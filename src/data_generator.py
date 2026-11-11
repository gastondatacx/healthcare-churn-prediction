import os
import random
import numpy as np
import pandas as pd
from src.config import RAW_DATA_DIR, RANDOM_SEED, N_AFILIADOS


def generate_relational_healthcare_data():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print(f"[1/4] Generando Padrón de Afiliados ({N_AFILIADOS:,} registros)...")
    planes = ["Plan 100 - Básico", "Plan 200 - Integral", "Plan 300 - Superior", "Plan 400 - Exclusivo"]
    segmentos = ["Obligatorio / Relación Dependencia", "Voluntario / Particular", "Monotributo / Desregulado", "Corporativo"]
    regiones = ["Córdoba Capital", "Gran Córdoba", "Interior - Zona Sur", "Interior - Zona Este", "CABA / GBA"]
    condiciones = ["Ninguna", "Crónico Empadronado (HTA/DBT)", "Plan Materno Infantil (PMI)", "Complejidad Alta"]

    members = []
    for i in range(1, N_AFILIADOS + 1):
        afiliado_id = f"AF_{i:06d}"
        edad = int(np.clip(np.random.normal(41, 15), 18, 82))
        antiguedad_meses = int(np.random.exponential(scale=40)) + 1
        plan = np.random.choice(planes, p=[0.30, 0.40, 0.20, 0.10])
        segmento = np.random.choice(segmentos, p=[0.40, 0.30, 0.18, 0.12])
        region = np.random.choice(regiones, p=[0.45, 0.20, 0.15, 0.10, 0.10])
        condicion = np.random.choice(condiciones, p=[0.74, 0.16, 0.07, 0.03])
        grupo_familiar = int(np.random.choice([1, 2, 3, 4, 5], p=[0.38, 0.28, 0.18, 0.11, 0.05]))
        debito_auto = int(np.random.choice([1, 0], p=[0.68, 0.32]))
        uso_app_mensual = int(np.random.poisson(lam=3.5 if edad < 50 else 1.5))

        base_cuota = {
            "Plan 100 - Básico": 65,
            "Plan 200 - Integral": 110,
            "Plan 300 - Superior": 175,
            "Plan 400 - Exclusivo": 260
        }[plan]
        cuota_mensual = round(base_cuota * (1 + 0.35 * (grupo_familiar - 1)) * (1 + (edad - 30) * 0.008), 2)

        members.append({
            "afiliado_id": afiliado_id,
            "edad": edad,
            "antiguedad_meses": antiguedad_meses,
            "plan_salud": plan,
            "tipo_contratacion": segmento,
            "region": region,
            "condicion_cronica": condicion,
            "tamano_grupo_familiar": grupo_familiar,
            "adhesion_debito_auto": debito_auto,
            "ingresos_app_90d": uso_app_mensual * 3,
            "cuota_mensual_usd": cuota_mensual
        })

    df_members = pd.DataFrame(members)

    print("[2/4] Generando Transacciones de Auditoría Médica y Autorizaciones...")
    auth_records = []
    for row in members:
        af_id = row["afiliado_id"]
        lam_auth = 2.0 + (2.5 if row["condicion_cronica"] != "Ninguna" else 0) + (0.6 * row["tamano_grupo_familiar"])
        n_auths = np.random.poisson(lam=lam_auth)

        friction_profile = np.random.rand() < 0.22
        rej_prob = 0.35 if friction_profile else 0.06
        tat_base = 72 if friction_profile else 18

        rejected = np.random.binomial(n_auths, rej_prob) if n_auths > 0 else 0
        partial = np.random.binomial(n_auths - rejected, 0.12) if (n_auths - rejected) > 0 else 0
        approved = max(0, n_auths - rejected - partial)
        avg_tat_hours = round(max(1.0, np.random.normal(tat_base, 14)), 1) if n_auths > 0 else 0.0

        auth_records.append({
            "afiliado_id": af_id,
            "total_ordenes_90d": n_auths,
            "ordenes_aprobadas_90d": approved,
            "ordenes_parciales_90d": partial,
            "ordenes_rechazadas_90d": rejected,
            "tiempo_medio_resolucion_hs": avg_tat_hours
        })
    df_auth = pd.DataFrame(auth_records)

    print("[3/4] Generando Interacciones de Contact Center y Voz del Afiliado (VoC)...")
    cx_records = []
    for idx, row in enumerate(members):
        af_id = row["afiliado_id"]
        rej = auth_records[idx]["ordenes_rechazadas_90d"]
        tat = auth_records[idx]["tiempo_medio_resolucion_hs"]

        lam_calls = 1.0 + (1.4 * rej) + (1.2 if tat > 48 else 0)
        total_calls = np.random.poisson(lam=lam_calls)

        repeat_calls_48h = np.random.binomial(total_calls, 0.32 if rej > 0 else 0.10) if total_calls > 0 else 0
        unresolved_tickets = np.random.binomial(total_calls, 0.25 if rej > 0 else 0.05) if total_calls > 0 else 0
        aht_avg_sec = int(np.clip(np.random.normal(340 + (45 * rej), 90), 90, 1200)) if total_calls > 0 else 0

        base_sentiment = 0.35 - (0.28 * rej) - (0.30 * unresolved_tickets) - (0.15 * repeat_calls_48h)
        sentiment_score = round(float(np.clip(np.random.normal(base_sentiment, 0.25), -1.0, 1.0)), 3)
        csat_score = round(float(np.clip(3.2 + (1.6 * sentiment_score) + np.random.normal(0, 0.4), 1.0, 5.0)), 1)

        cx_records.append({
            "afiliado_id": af_id,
            "llamadas_totales_90d": total_calls,
            "llamadas_reincidentes_48h": repeat_calls_48h,
            "tickets_abiertos_sin_resolver": unresolved_tickets,
            "tiempo_medio_atencion_seg": aht_avg_sec,
            "puntaje_sentimiento_nlp": sentiment_score,
            "csat_promedio": csat_score
        })
    df_cx = pd.DataFrame(cx_records)

    print("[4/4] Generando Facturación, Reintegros y Estado de Baja (Churn)...")
    billing_records = []
    churn_labels = []

    for idx, row in enumerate(members):
        af_id = row["afiliado_id"]
        rej = auth_records[idx]["ordenes_rechazadas_90d"]
        tat = auth_records[idx]["tiempo_medio_resolucion_hs"]
        unres = cx_records[idx]["tickets_abiertos_sin_resolver"]
        rep_calls = cx_records[idx]["llamadas_reincidentes_48h"]
        sent = cx_records[idx]["puntaje_sentimiento_nlp"]

        aumento_pct = round(float(np.clip(np.random.normal(18.0, 7.5), 2.0, 48.0)), 1)
        copagos_usd = round(float(np.random.exponential(scale=28.0)), 2)
        reintegros_pendientes = int(np.random.choice([0, 1, 2, 3], p=[0.72, 0.16, 0.08, 0.04]))
        dias_demora_reintegro = int(np.clip(np.random.normal(38, 18), 7, 110)) if reintegros_pendientes > 0 else 0
        mora_facturas = int(np.random.choice([0, 1, 2], p=[0.84, 0.11, 0.05]))

        billing_records.append({
            "afiliado_id": af_id,
            "aumento_cuota_anual_pct": aumento_pct,
            "gasto_copagos_90d_usd": copagos_usd,
            "reintegros_pendientes_cant": reintegros_pendientes,
            "dias_demora_reintegro": dias_demora_reintegro,
            "facturas_impagas_cant": mora_facturas
        })

        logit = (
            -3.25
            + 0.62 * rej
            + 0.015 * max(0, tat - 24)
            + 0.78 * unres
            + 0.42 * rep_calls
            - 1.15 * sent
            + 0.045 * max(0, aumento_pct - 15)
            + 0.018 * dias_demora_reintegro
            + 0.55 * mora_facturas
            - 0.018 * row["antiguedad_meses"]
            - 0.35 * row["adhesion_debito_auto"]
            - 0.04 * row["ingresos_app_90d"]
            + (0.45 if row["tipo_contratacion"] == "Voluntario / Particular" else 0.0)
            + (0.85 if (rej >= 1 and unres >= 1) else 0.0)
        )

        prob = 1.0 / (1.0 + np.exp(-logit))
        baja = 1 if np.random.rand() < prob else 0
        churn_labels.append({"afiliado_id": af_id, "baja_afiliado": baja})

    df_billing = pd.DataFrame(billing_records)
    df_target = pd.DataFrame(churn_labels)

    df_members.merge(df_target, on="afiliado_id").to_csv(os.path.join(RAW_DATA_DIR, "padron_afiliados.csv"), index=False)
    df_auth.to_csv(os.path.join(RAW_DATA_DIR, "auditoria_medica_90d.csv"), index=False)
    df_cx.to_csv(os.path.join(RAW_DATA_DIR, "interacciones_cx_90d.csv"), index=False)
    df_billing.to_csv(os.path.join(RAW_DATA_DIR, "facturacion_reintegros_90d.csv"), index=False)

    tasa_baja = df_target["baja_afiliado"].mean() * 100
    print(f" -> Tablas relacionales guardadas en: {RAW_DATA_DIR}")
    print(f" -> Tasa global de Baja (Churn) generada: {tasa_baja:.2f}%\n")


if __name__ == "__main__":
    generate_relational_healthcare_data()