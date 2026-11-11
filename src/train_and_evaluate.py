import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_curve, precision_recall_curve,
    confusion_matrix
)

from src.config import (
    PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, RANDOM_SEED,
    COSTO_AFILIADO_PERDIDO_LTV, COSTO_ACCION_RETENCION, TASA_EXITO_RETENCION
)

plt.style.use("seaborn-v0_8-whitegrid")


def calcular_ahorro_financiero(y_true, y_prob, umbral):
    """Calcula el beneficio económico neto para un umbral de probabilidad dado."""
    y_pred = (y_prob >= umbral).astype(int)
    vn, fp, fn, vp = confusion_matrix(y_true, y_pred).ravel()

    # Afiliados en riesgo real detectados * efectividad de retención * LTV salvado
    ltv_bruto_recuperado = vp * TASA_EXITO_RETENCION * COSTO_AFILIADO_PERDIDO_LTV
    # Costo operativo de gestionar tanto a Verdaderos Positivos como Falsos Positivos
    costo_campana = (vp + fp) * COSTO_ACCION_RETENCION

    ahorro_neto = ltv_bruto_recuperado - costo_campana
    return ahorro_neto, vp, fp, fn, vn


def generar_reporte_clasificacion_es(y_true, y_pred):
    """Genera una tabla de métricas de clasificación íntegramente en español."""
    clases = [(0, "Permanece Activo (0)"), (1, "Baja / Fuga (1)")]
    filas = []
    for valor, etiqueta in clases:
        y_bin_true = (y_true == valor).astype(int)
        y_bin_pred = (y_pred == valor).astype(int)
        filas.append({
            "Clase / Estado": etiqueta,
            "Precisión": round(precision_score(y_bin_true, y_bin_pred, zero_division=0), 4),
            "Sensibilidad (Recall)": round(recall_score(y_bin_true, y_bin_pred, zero_division=0), 4),
            "Puntaje F1": round(f1_score(y_bin_true, y_bin_pred, zero_division=0), 4),
            "Cantidad Casos": int(y_bin_true.sum())
        })
    return pd.DataFrame(filas)


def train_and_select_champion():
    abt_path = os.path.join(PROCESSED_DATA_DIR, "abt_fuga_afiliados_salud.csv")
    df = pd.read_csv(abt_path)

    target_col = "baja_afiliado"
    ignore_cols = ["afiliado_id", target_col]

    feature_cols = [c for c in df.columns if c not in ignore_cols]
    categorical_cols = ["plan_salud", "tipo_contratacion", "region", "condicion_cronica"]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_cols)
        ]
    )

    modelos = {
        "Regresión Logística (Base)": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED
        ),
        "Random Forest (Balanceado)": RandomForestClassifier(
            n_estimators=250, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1
        ),
        "HistGradientBoosting (Campeón)": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.06, max_depth=6,
            class_weight="balanced", random_state=RANDOM_SEED
        )
    }

    print("Ejecutando Torneo de Modelos con Validación Cruzada Estratificada (5 Particiones)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    resultados_torneo = []
    pipelines_entrenados = {}

    # Figura 1: Curvas ROC y Precisión-Sensibilidad comparativas
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for nombre, clf in modelos.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        cv_scores = cross_validate(
            pipe, X_train, y_train, cv=cv,
            scoring=["roc_auc", "average_precision", "f1"], n_jobs=-1
        )

        pipe.fit(X_train, y_train)
        pipelines_entrenados[nombre] = pipe

        y_prob_test = pipe.predict_proba(X_test)[:, 1]
        y_pred_test = pipe.predict(X_test)

        roc_test = roc_auc_score(y_test, y_prob_test)
        pr_auc_test = average_precision_score(y_test, y_prob_test)
        rec_test = recall_score(y_test, y_pred_test)
        prec_test = precision_score(y_test, y_pred_test)
        f1_test = f1_score(y_test, y_pred_test)

        resultados_torneo.append({
            "Modelo": nombre,
            "ROC_AUC_Validacion": round(cv_scores["test_roc_auc"].mean(), 4),
            "ROC_AUC_Prueba": round(roc_test, 4),
            "PR_AUC_Prueba": round(pr_auc_test, 4),
            "Sensibilidad_Recall": round(rec_test, 4),
            "Precision": round(prec_test, 4),
            "Puntaje_F1": round(f1_test, 4)
        })

        fpr, tpr, _ = roc_curve(y_test, y_prob_test)
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob_test)

        axes[0].plot(fpr, tpr, lw=2, label=f"{nombre} (AUC = {roc_test:.3f})")
        axes[1].plot(rec_curve, prec_curve, lw=2, label=f"{nombre} (PR-AUC = {pr_auc_test:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_title("Curva ROC Comparativa (Conjunto de Prueba)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Tasa de Falsos Positivos (1 - Especificidad)")
    axes[0].set_ylabel("Tasa de Verdaderos Positivos (Sensibilidad)")
    axes[0].legend(loc="lower right")

    axes[1].set_title("Curva Precisión-Sensibilidad (Conjunto de Prueba)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Sensibilidad / Recall (Detección de Bajas)")
    axes[1].set_ylabel("Precisión (Pureza de la Alerta)")
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "01_model_comparison_roc_pr.png"), dpi=180)
    plt.close()

    df_bench = pd.DataFrame(resultados_torneo).sort_values(by="ROC_AUC_Prueba", ascending=False)
    print("\n--- TABLA COMPARATIVA DE RENDIMIENTO DE MODELOS ---")
    print(df_bench.to_string(index=False))

    nombre_campeon = df_bench.iloc[0]["Modelo"]
    pipe_campeon = pipelines_entrenados[nombre_campeon]
    y_prob_champ = pipe_campeon.predict_proba(X_test)[:, 1]

    # Optimización Financiera del Punto de Corte (Umbral)
    umbrales = np.linspace(0.10, 0.90, 81)
    lista_ahorros = []
    for umb in umbrales:
        ahorro_neto, _, _, _, _ = calcular_ahorro_financiero(y_test, y_prob_champ, umb)
        lista_ahorros.append(ahorro_neto)

    mejor_idx = int(np.argmax(lista_ahorros))
    umbral_optimo = round(float(umbrales[mejor_idx]), 2)
    ahorro_maximo, vp, fp, fn, vn = calcular_ahorro_financiero(y_test, y_prob_champ, umbral_optimo)

    print(f"\n -> Modelo Seleccionado (Campeón): {nombre_campeon}")
    print(f" -> Umbral Financiero Óptimo de Corte: {umbral_optimo} ({umbral_optimo * 100:.0f}% de probabilidad)")
    print(f" -> Ahorro Económico Neto en Conjunto de Prueba (2.000 afiliados): USD ${ahorro_maximo:,.2f}")
    print(f" -> Desglose Operativo: {vp} Bajas Reales Detectadas | {fp} Falsas Alarmas | {fn} Bajas No Detectadas")

    # Figura 2: Curva de Optimización de Costo/Beneficio Empresarial
    plt.figure(figsize=(9, 5))
    plt.plot(umbrales, lista_ahorros, color="#1f77b4", lw=2.5)
    plt.axvline(
        umbral_optimo, color="#d62728", linestyle="--", lw=2,
        label=f"Umbral Óptimo = {umbral_optimo} (Ahorro Neto: USD ${ahorro_maximo:,.0f})"
    )
    plt.title("Optimización Financiera: Ahorro Neto de Retención según Punto de Corte", fontsize=12, fontweight="bold")
    plt.xlabel("Umbral de Probabilidad de Baja")
    plt.ylabel("Ahorro Económico Neto Proyectado (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "02_financial_threshold_optimization.png"), dpi=180)
    plt.close()

    # Figura 3: Importancia de Variables en Español
    print("\nCalculando Importancia de Variables por Permutación en el Conjunto de Prueba...")
    perm_imp = permutation_importance(
        pipe_campeon, X_test, y_test, n_repeats=8, random_state=RANDOM_SEED, scoring="roc_auc", n_jobs=-1
    )

    nombres_legibles = {
        "indice_friccion_salud": "Índice Compuesto de Fricción Salud/CX",
        "puntaje_sentimiento_nlp": "Sentimiento NLP en Llamadas",
        "tickets_abiertos_sin_resolver": "Tickets Abiertos Sin Resolver",
        "antiguedad_meses": "Antigüedad del Afiliado (Meses)",
        "ordenes_rechazadas_90d": "Órdenes Médicas Rechazadas (90d)",
        "dias_demora_reintegro": "Días de Demora en Reintegros",
        "aumento_cuota_anual_pct": "% Aumento Anual de Cuota",
        "facturas_impagas_cant": "Cantidad de Facturas Impagas",
        "alerta_doble_friccion_critica": "Alerta Doble Fricción (Auditoría + CX)",
        "llamadas_reincidentes_48h": "Llamadas Reincidentes (< 48hs)",
        "tasa_rechazo_medico": "Tasa de Rechazo en Auditoría Médica",
        "adhesion_debito_auto": "Adhesión a Débito Automático",
        "tipo_contratacion": "Segmento / Tipo de Contratación",
        "ingresos_app_90d": "Frecuencia de Uso de App Móvil",
        "tasa_resolucion_primer_contacto": "Tasa de Resolución en 1er Contacto (FCR)"
    }

    df_imp = pd.DataFrame({
        "Variable": [nombres_legibles.get(c, c) for c in feature_cols],
        "Importancia_Media": perm_imp.importances_mean
    }).sort_values(by="Importancia_Media", ascending=True).tail(12)

    plt.figure(figsize=(10, 6))
    plt.barh(df_imp["Variable"], df_imp["Importancia_Media"], color="#2b5c8f")
    plt.title("Top 12 Factores Determinantes de Baja de Afiliados", fontsize=12, fontweight="bold")
    plt.xlabel("Impacto en Capacidad Predictiva (Caída en ROC-AUC al permutar)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "03_feature_importance.png"), dpi=180)
    plt.close()

    # Figura 4: Matriz de Confusión con Umbral Óptimo
    y_pred_opt = (y_prob_champ >= umbral_optimo).astype(int)
    cm = confusion_matrix(y_test, y_pred_opt)
    plt.figure(figsize=(6.5, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Permanece Activo (0)", "Baja / Fuga (1)"],
        yticklabels=["Permanece Activo (0)", "Baja / Fuga (1)"]
    )
    plt.title(f"Matriz de Confusión Operativa (Umbral = {umbral_optimo})", fontsize=12, fontweight="bold")
    plt.ylabel("Estado Real del Afiliado")
    plt.xlabel("Predicción del Sistema de Alerta")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "04_confusion_matrix_optimal.png"), dpi=180)
    plt.close()

    print("\n--- REPORTE DE CLASIFICACIÓN CON UMBRAL FINANCIERO ÓPTIMO ---")
    df_rep_es = generar_reporte_clasificacion_es(y_test, y_pred_opt)
    print(df_rep_es.to_string(index=False))

    # Guardar modelo y metadatos en español
    joblib.dump(pipe_campeon, os.path.join(MODELS_DIR, "champion_churn_pipeline.joblib"))
    metadata = {
        "modelo_campeon": nombre_campeon,
        "umbral_optimo": umbral_optimo,
        "roc_auc_prueba": float(df_bench.iloc[0]["ROC_AUC_Prueba"]),
        "pr_auc_prueba": float(df_bench.iloc[0]["PR_AUC_Prueba"]),
        "ahorro_neto_estimado_prueba_usd": round(float(ahorro_maximo), 2),
        "columnas_predictoras": feature_cols
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print(f"\n -> Modelo y metadatos exportados en: {MODELS_DIR}")
    print(f" -> Gráficos en español exportados en: {FIGURES_DIR}\n")


if __name__ == "__main__":
    train_and_select_champion()