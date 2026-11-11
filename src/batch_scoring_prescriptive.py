import os
import json
import joblib
import numpy as np
import pandas as pd
from src.config import PROCESSED_DATA_DIR, MODELS_DIR


def asignar_accion_prescriptiva(row):
    """Determina el motivo raíz y la mejor acción comercial/operativa para cada afiliado."""
    if row["segmento_riesgo"] == "1. Riesgo Bajo (Monitoreo Habitual)":
        return "Sin Fricción Activa", "Flujo Digital Preventivo (Boletín de Salud / Autogestión App)"

    if row["ordenes_rechazadas_90d"] >= 1 or row["tiempo_medio_resolucion_hs"] > 48:
        return (
            "Fricción en Auditoría Médica",
            "Contacto Célula Médica (<24hs): Revisión de orden observada y derivación a prestador alternativo de cartilla"
        )
    if row["tickets_abiertos_sin_resolver"] >= 1 or row["llamadas_reincidentes_48h"] >= 2:
        return (
            "Falla de Resolución en Contact Center (Bajo FCR)",
            "Escalamiento a Supervisor CX Senior: Resolución prioritaria de ticket abierto y llamado de cierre"
        )
    if row["dias_demora_reintegro"] > 35:
        return (
            "Demora Crítica en Reintegros",
            "Vía Rápida Administración: Liquidación prioritaria de reintegro pendiente y aviso proactivo por WhatsApp"
        )
    if row["aumento_cuota_anual_pct"] > 22 or row["facturas_impagas_cant"] >= 1:
        return (
            "Sensibilidad Tarifaria / Mora",
            "Retención Comercial: Ofrecer bonificación temporal del 15% por adhesión a débito automático o adecuación de plan"
        )

    return (
        "Desgaste General de Relación",
        "Llamado Preventivo de Fidelización y Encuesta de Satisfacción Personalizada"
    )


def run_prescriptive_batch_scoring():
    print("Ejecutando Calificación de Cartera (Scoring) y Motor Prescriptivo de Retención...")
    abt_path = os.path.join(PROCESSED_DATA_DIR, "abt_fuga_afiliados_salud.csv")
    model_path = os.path.join(MODELS_DIR, "champion_churn_pipeline.joblib")
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")

    df = pd.read_csv(abt_path)
    pipeline = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    opt_th = meta["umbral_optimo"]
    features = meta["columnas_predictoras"]

    probs = pipeline.predict_proba(df[features])[:, 1]
    df["probabilidad_baja_pct"] = (probs * 100).round(2)

    # Segmentación por Niveles de Riesgo basada en el umbral financiero óptimo
    corte_critico = max(65.0, round(opt_th * 100 + 15, 1))
    corte_moderado = round(opt_th * 100, 1)

    condiciones = [
        df["probabilidad_baja_pct"] >= corte_critico,
        (df["probabilidad_baja_pct"] >= corte_moderado) & (df["probabilidad_baja_pct"] < corte_critico),
        df["probabilidad_baja_pct"] < corte_moderado
    ]
    niveles = [
        "3. Riesgo Crítico (Intervención < 24hs)",
        "2. Riesgo Moderado (Campaña Preventiva)",
        "1. Riesgo Bajo (Monitoreo Habitual)"
    ]
    df["segmento_riesgo"] = np.select(condiciones, niveles, default="1. Riesgo Bajo (Monitoreo Habitual)")

    # Valor Económico Anualizado en Riesgo
    df["facturacion_anual_usd"] = (df["cuota_mensual_usd"] * 12).round(2)
    df["perdida_anual_esperada_usd"] = (df["facturacion_anual_usd"] * probs).round(2)

    # Aplicar motor prescriptivo
    acciones = df.apply(asignar_accion_prescriptiva, axis=1)
    df["motivo_raiz_detectado"] = [a[0] for a in acciones]
    df["accion_recomendada_crm"] = [a[1] for a in acciones]

    # Exportar listado operativo para CRM / Power BI con columnas en español
    columnas_salida = [
        "afiliado_id", "plan_salud", "tipo_contratacion", "region", "antiguedad_meses",
        "cuota_mensual_usd", "facturacion_anual_usd", "probabilidad_baja_pct",
        "perdida_anual_esperada_usd", "segmento_riesgo", "motivo_raiz_detectado", "accion_recomendada_crm"
    ]
    df_out = df[columnas_salida].sort_values(by="perdida_anual_esperada_usd", ascending=False)
    scored_path = os.path.join(PROCESSED_DATA_DIR, "listado_accionable_retencion_crm.csv")
    df_out.to_csv(scored_path, index=False)

    # Resumen Ejecutivo por Segmento en Consola
    resumen = df_out.groupby("segmento_riesgo").agg(
        Cantidad_Afiliados=("afiliado_id", "count"),
        Probabilidad_Media_Pct=("probabilidad_baja_pct", lambda x: round(x.mean(), 2)),
        Facturacion_Anual_Cartera_USD=("facturacion_anual_usd", lambda x: round(x.sum(), 2)),
        Riesgo_Financiero_Esperado_USD=("perdida_anual_esperada_usd", lambda x: round(x.sum(), 2))
    ).reset_index()

    print("\n--- RESUMEN EJECUTIVO DE CARTERA Y RIESGO FINANCIERO ---")
    print(resumen.to_string(index=False))
    print(f"\n -> Listado accionable para CRM exportado en: {scored_path}")


if __name__ == "__main__":
    run_prescriptive_batch_scoring()