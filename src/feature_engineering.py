import os
import numpy as np
import pandas as pd
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def build_analytical_base_table():
    print("Consolidando tablas y calculando indicadores en porcentajes (0% a 100%) y Pesos ($ ARS)...")

    df_mem = pd.read_csv(os.path.join(RAW_DATA_DIR, "padron_afiliados.csv"))
    df_auth = pd.read_csv(os.path.join(RAW_DATA_DIR, "auditoria_medica_90d.csv"))
    df_cx = pd.read_csv(os.path.join(RAW_DATA_DIR, "interacciones_cx_90d.csv"))
    df_bill = pd.read_csv(os.path.join(RAW_DATA_DIR, "facturacion_reintegros_90d.csv"))

    abt = (
        df_mem
        .merge(df_auth, on="afiliado_id", how="left")
        .merge(df_cx, on="afiliado_id", how="left")
        .merge(df_bill, on="afiliado_id", how="left")
    )

    # 1. Tasa de Rechazo en Auditoría Médica (% de 0 a 100)
    abt["tasa_rechazo_medico_pct"] = np.where(
        abt["ordenes_solicitadas_trimestre_cant"] > 0,
        ((abt["ordenes_rechazadas_trimestre_cant"] + 0.5 * abt["ordenes_observadas_trimestre_cant"])
         / abt["ordenes_solicitadas_trimestre_cant"]) * 100.0,
        0.0
    ).round(1)

    # 2. Tasa de Resolución en Primer Contacto - FCR (% de 0 a 100)
    abt["tasa_resolucion_primer_contacto_pct"] = np.where(
        abt["llamadas_totales_trimestre_cant"] > 0,
        (1.0 - (abt["llamadas_reincidentes_48hs_cant"] / abt["llamadas_totales_trimestre_cant"])) * 100.0,
        100.0
    ).clip(0.0, 100.0).round(1)

    # 3. Peso de los Copagos sobre la Cuota Trimestral (% de 0 a 100)
    abt["peso_copagos_sobre_cuota_pct"] = (
        (abt["gasto_copagos_trimestre_ars"] / (abt["cuota_mensual_ars"] * 3.0)) * 100.0
    ).round(1)

    # 4. Índice de Fricción del Afiliado (Escala normalizada de 0 a 100 puntos)
    friccion_bruta = (
        (abt["ordenes_rechazadas_trimestre_cant"] * 12.0)
        + (abt["reclamos_sin_resolver_cant"] * 15.0)
        + (abt["llamadas_reincidentes_48hs_cant"] * 8.0)
        + np.where(abt["tiempo_resolucion_auditoria_hs"] > 48, 12.0, 0.0)
        + np.where(abt["demora_reintegro_dias"] > 45, 10.0, 0.0)
        + ((10.0 - abt["indice_tono_llamada_1a10"]) * 3.5)
    )
    abt["indice_friccion_afiliado_0a100"] = np.clip(friccion_bruta, 0.0, 100.0).round(1)

    # 5. Alerta de Doble Fricción Crítica (1 = Sí, 0 = No)
    abt["alerta_doble_friccion_bin"] = np.where(
        (abt["ordenes_rechazadas_trimestre_cant"] >= 1) & (abt["reclamos_sin_resolver_cant"] >= 1),
        1, 0
    )

    # 6. Porcentaje de Autogestión Digital vs Teléfono (% de 0 a 100)
    abt["porcentaje_autogestion_digital_pct"] = (
        (abt["ingresos_app_trimestre_cant"] /
         np.maximum(1, abt["llamadas_totales_trimestre_cant"] + abt["ingresos_app_trimestre_cant"])) * 100.0
    ).round(1)

    abt_path = os.path.join(PROCESSED_DATA_DIR, "abt_fuga_afiliados_salud.csv")
    abt.to_csv(abt_path, index=False)

    # Mostrar diccionario resumen de unidades en consola
    resumen_unidades = pd.DataFrame([
        {"Variable": "cuota_mensual_ars", "Unidad de Medida": "Pesos Argentinos ($ ARS)", "Promedio Cartera": f"$ {abt['cuota_mensual_ars'].mean():,.0f} ARS"},
        {"Variable": "gasto_copagos_trimestre_ars", "Unidad de Medida": "Pesos Argentinos ($ ARS)", "Promedio Cartera": f"$ {abt['gasto_copagos_trimestre_ars'].mean():,.0f} ARS"},
        {"Variable": "tasa_rechazo_medico_pct", "Unidad de Medida": "Porcentaje (0% a 100%)", "Promedio Cartera": f"{abt['tasa_rechazo_medico_pct'].mean():.1f}%"},
        {"Variable": "tasa_resolucion_primer_contacto_pct", "Unidad de Medida": "Porcentaje (0% a 100%)", "Promedio Cartera": f"{abt['tasa_resolucion_primer_contacto_pct'].mean():.1f}%"},
        {"Variable": "indice_friccion_afiliado_0a100", "Unidad de Medida": "Puntos (Escala 0 a 100)", "Promedio Cartera": f"{abt['indice_friccion_afiliado_0a100'].mean():.1f} pts"},
        {"Variable": "tiempo_resolucion_auditoria_hs", "Unidad de Medida": "Horas corridas (hs)", "Promedio Cartera": f"{abt['tiempo_resolucion_auditoria_hs'].mean():.1f} hs"},
        {"Variable": "demora_reintegro_dias", "Unidad de Medida": "Días corridos", "Promedio Cartera": f"{abt['demora_reintegro_dias'].mean():.1f} días"},
    ])
    print("\n--- DICCIONARIO DE UNIDADES Y PROMEDIOS DE LA CARTERA ---")
    print(resumen_unidades.to_string(index=False))
    print(f"\n -> Tabla Analítica guardada en: {abt_path}\n")
    return abt


if __name__ == "__main__":
    build_analytical_base_table()