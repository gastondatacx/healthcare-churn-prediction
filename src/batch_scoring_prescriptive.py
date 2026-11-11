import os
import json
import joblib
import numpy as np
import pandas as pd
from src.config import PROCESSED_DATA_DIR, MODELS_DIR


def asignar_accion_prescriptiva(row):
    """Determina el motivo raíz y la acción concreta para el equipo de Salud / CX."""
    if row["nivel_riesgo"] == "1. Riesgo Bajo (Monitoreo Habitual)":
        return "Sin Fricción Activa", "Flujo Digital Preventivo (Recordatorios de Cartilla / App)"

    if row["ordenes_rechazadas_trimestre_cant"] >= 1 or row["tiempo_resolucion_auditoria_hs"] > 48:
        return (
            "Fricción en Auditoría Médica",
            "Contacto Célula Médica (<24hs): Revisión de orden observada y turno prioritario en prestador de cartilla"
        )
    if row["reclamos_sin_resolver_cant"] >= 1 or row["llamadas_reincidentes_48hs_cant"] >= 2:
        return (
            "Falla de Resolución en Contact Center (Bajo FCR)",
            "Escalamiento a Supervisor CX Senior: Resolución inmediata de reclamo abierto y llamado de cierre"
        )
    if row["demora_reintegro_dias"] > 35:
        return (
            "Demora Crítica en Pago de Reintegros",
            "Vía Rápida Administración: Liquidación prioritaria de reintegro en cuenta y aviso por WhatsApp"
        )
    if row["aumento_cuota_trimestral_pct"] > 22 or row["facturas_impagas_cant"] >= 1:
        return (
            "Sensibilidad Tarifaria / Cuota",
            "Retención Comercial: Bonificación del 15% por 6 meses adhiriendo a débito automático o adecuación de plan"
        )

    return (
        "Desgaste General de Relación",
        "Llamado Preventivo de Fidelización y Chequeo Médico Preventivo Bonificado"
    )


def run_prescriptive_batch_scoring():
    print("Generando Listado Accionable para CRM con montos en Pesos ($ ARS) y probabilidades en %...")
    abt_path = os.path.join(PROCESSED_DATA_DIR, "abt_fuga_afiliados_salud.csv")
    model_path = os.path.join(MODELS_DIR, "champion_churn_pipeline.joblib")
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")

    df = pd.read_csv(abt_path)
    pipeline = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    opt_th_dec = meta["umbral_optimo_decimal"]
    features = meta["columnas_predictoras"]

    probs = pipeline.predict_proba(df[features])[:, 1]
    df["probabilidad_baja_pct"] = (probs * 100.0).round(1)

    corte_critico_pct = max(65.0, round(opt_th_dec * 100 + 15, 1))
    corte_moderado_pct = round(opt_th_dec * 100, 1)

    condiciones = [
        df["probabilidad_baja_pct"] >= corte_critico_pct,
        (df["probabilidad_baja_pct"] >= corte_moderado_pct) & (df["probabilidad_baja_pct"] < corte_critico_pct),
        df["probabilidad_baja_pct"] < corte_moderado_pct
    ]
    niveles = [
        "3. Riesgo Crítico (Llamar en < 24hs)",
        "2. Riesgo Moderado (Campaña Preventiva)",
        "1. Riesgo Bajo (Monitoreo Habitual)"
    ]
    df["nivel_riesgo"] = np.select(condiciones, niveles, default="1. Riesgo Bajo (Monitoreo Habitual)")

    # Facturación Anual y Dinero en Riesgo en Pesos Argentinos ($ ARS)
    df["aporte_anual_afiliado_ars"] = (df["cuota_mensual_ars"] * 12).round(0)
    df["dinero_anual_en_riesgo_ars"] = (df["aporte_anual_afiliado_ars"] * probs).round(0)

    acciones = df.apply(asignar_accion_prescriptiva, axis=1)
    df["motivo_raiz_principal"] = [a[0] for a in acciones]
    df["accion_recomendada_crm"] = [a[1] for a in acciones]

    columnas_salida = [
        "afiliado_id", "plan_salud", "tipo_contratacion", "region_residencia",
        "edad_anos", "antiguedad_meses", "cuota_mensual_ars", "aporte_anual_afiliado_ars",
        "probabilidad_baja_pct", "dinero_anual_en_riesgo_ars",
        "indice_friccion_afiliado_0a100", "tasa_rechazo_medico_pct",
        "nivel_riesgo", "motivo_raiz_principal", "accion_recomendada_crm"
    ]
    df_out = df[columnas_salida].sort_values(by="dinero_anual_en_riesgo_ars", ascending=False)
    scored_path = os.path.join(PROCESSED_DATA_DIR, "listado_accionable_retencion_crm.csv")
    df_out.to_csv(scored_path, index=False)

    # Resumen Ejecutivo Formateado con $ ARS y % para lectura inmediata
    resumen_grupos = []
    for nivel, grupo in df_out.groupby("nivel_riesgo"):
        cant = len(grupo)
        pct_cartera = (cant / len(df_out)) * 100
        prob_media = grupo["probabilidad_baja_pct"].mean()
        fact_millones = grupo["aporte_anual_afiliado_ars"].sum() / 1_000_000.0
        riesgo_millones = grupo["dinero_anual_en_riesgo_ars"].sum() / 1_000_000.0

        resumen_grupos.append({
            "Segmento de Riesgo": nivel,
            "Afiliados (Cant y %)": f"{cant:,} ({pct_cartera:.1f}%)",
            "Riesgo Promedio (%)": f"{prob_media:.1f}%",
            "Facturación Anual ($ ARS)": f"$ {fact_millones:,.1f} Millones",
            "Dinero en Riesgo ($ ARS)": f"$ {riesgo_millones:,.1f} Millones"
        })

    df_resumen = pd.DataFrame(resumen_grupos)
    print("\n--- RESUMEN EJECUTIVO DE CARTERA EN PESOS ARGENTINOS ($ ARS) ---")
    print(df_resumen.to_string(index=False))
    print(f"\n -> Base CRM lista para usar exportada en: {scored_path}")


if __name__ == "__main__":
    run_prescriptive_batch_scoring()