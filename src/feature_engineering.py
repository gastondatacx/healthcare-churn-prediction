import os
import numpy as np
import pandas as pd
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def build_analytical_base_table():
    print("Iniciando Ingeniería de Variables y consolidación de la Tabla Base Analítica (ABT)...")

    df_mem = pd.read_csv(os.path.join(RAW_DATA_DIR, "padron_afiliados.csv"))
    df_auth = pd.read_csv(os.path.join(RAW_DATA_DIR, "auditoria_medica_90d.csv"))
    df_cx = pd.read_csv(os.path.join(RAW_DATA_DIR, "interacciones_cx_90d.csv"))
    df_bill = pd.read_csv(os.path.join(RAW_DATA_DIR, "facturacion_reintegros_90d.csv"))

    # Unión relacional de las 4 tablas
    abt = (
        df_mem
        .merge(df_auth, on="afiliado_id", how="left")
        .merge(df_cx, on="afiliado_id", how="left")
        .merge(df_bill, on="afiliado_id", how="left")
    )

    # 1. Tasa de Rechazo en Auditoría Médica
    abt["tasa_rechazo_medico"] = np.where(
        abt["total_ordenes_90d"] > 0,
        (abt["ordenes_rechazadas_90d"] + 0.5 * abt["ordenes_parciales_90d"]) / abt["total_ordenes_90d"],
        0.0
    ).round(4)

    # 2. Tasa Estimada de Resolución en Primer Contacto (FCR)
    abt["tasa_resolucion_primer_contacto"] = np.where(
        abt["llamadas_totales_90d"] > 0,
        1.0 - (abt["llamadas_reincidentes_48h"] / abt["llamadas_totales_90d"]),
        1.0
    ).clip(0.0, 1.0).round(4)

    # 3. Ratio de Carga Financiera Adicional (Copagos sobre Cuota Trimestral)
    abt["ratio_copago_sobre_cuota"] = (
        abt["gasto_copagos_90d_usd"] / (abt["cuota_mensual_usd"] * 3.0)
    ).round(4)

    # 4. Índice Compuesto de Fricción en Salud y CX
    abt["indice_friccion_salud"] = (
        (abt["ordenes_rechazadas_90d"] * 2.5)
        + (abt["tickets_abiertos_sin_resolver"] * 3.0)
        + (abt["llamadas_reincidentes_48h"] * 1.5)
        + np.where(abt["tiempo_medio_resolucion_hs"] > 48, 2.0, 0.0)
        + np.where(abt["dias_demora_reintegro"] > 45, 2.0, 0.0)
        - (abt["puntaje_sentimiento_nlp"] * 2.0)
    ).round(3)

    # 5. Alerta de Doble Fricción Crítica (Rechazo Médico + Reclamo Abierto)
    abt["alerta_doble_friccion_critica"] = np.where(
        (abt["ordenes_rechazadas_90d"] >= 1) & (abt["tickets_abiertos_sin_resolver"] >= 1),
        1, 0
    )

    # 6. Ratio de Autonomía Digital vs Teléfono
    abt["ratio_autonomia_digital"] = (
        (abt["ingresos_app_90d"] + 1) / (abt["llamadas_totales_90d"] + abt["ingresos_app_90d"] + 1)
    ).round(4)

    abt_path = os.path.join(PROCESSED_DATA_DIR, "abt_fuga_afiliados_salud.csv")
    abt.to_csv(abt_path, index=False)
    print(f" -> Tabla Maestra (ABT) generada con éxito: {abt.shape[0]:,} filas y {abt.shape[1]} columnas.")
    print(f" -> Archivo guardado en: {abt_path}\n")
    return abt


if __name__ == "__main__":
    build_analytical_base_table()