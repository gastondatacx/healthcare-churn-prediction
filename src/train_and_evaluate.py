import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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
    precision_score, recall_score, roc_curve, confusion_matrix
)

from src.config import (
    PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, RANDOM_SEED,
    VALOR_VIDA_AFILIADO_ARS, COSTO_GESTION_RETENCION_ARS, TASA_EXITO_RETENCION
)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"


def calcular_ahorro_en_pesos(y_true, y_prob, umbral_decimal):
    """Calcula el ahorro económico neto en Pesos Argentinos ($ ARS)."""
    y_pred = (y_prob >= umbral_decimal).astype(int)
    vn, fp, fn, vp = confusion_matrix(y_true, y_pred).ravel()

    # Afiliados salvados = Verdaderos Positivos * Tasa de Éxito * Valor de Vida en ARS
    monto_recuperado_ars = vp * TASA_EXITO_RETENCION * VALOR_VIDA_AFILIADO_ARS
    # Costo de contactar a todos los alertados (Verdaderos Positivos + Falsas Alarmas)
    costo_operativo_ars = (vp + fp) * COSTO_GESTION_RETENCION_ARS

    ahorro_neto_ars = monto_recuperado_ars - costo_operativo_ars
    return ahorro_neto_ars, int(vp), int(fp), int(fn), int(vn)


def train_and_select_champion():
    abt_path = os.path.join(PROCESSED_DATA_DIR, "abt_fuga_afiliados_salud.csv")
    df = pd.read_csv(abt_path)

    target_col = "baja_afiliado"
    ignore_cols = ["afiliado_id", target_col]

    feature_cols = [c for c in df.columns if c not in ignore_cols]
    categorical_cols = ["plan_salud", "tipo_contratacion", "region_residencia", "condicion_medica"]
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
        "Regresión Logística": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1
        ),
        "Gradient Boosting (Campeón)": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.06, max_depth=6,
            class_weight="balanced", random_state=RANDOM_SEED
        )
    }

    print("Entrenando y comparando modelos predictivos (métricas expresadas en % de 0 a 100%)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    resultados_torneo = []
    pipelines_entrenados = {}
    curvas_roc = {}

    for nombre, clf in modelos.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        cv_scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=["roc_auc"], n_jobs=-1)

        pipe.fit(X_train, y_train)
        pipelines_entrenados[nombre] = pipe

        y_prob_test = pipe.predict_proba(X_test)[:, 1]
        y_pred_test = pipe.predict(X_test)

        roc_pct = roc_auc_score(y_test, y_prob_test) * 100
        rec_pct = recall_score(y_test, y_pred_test) * 100
        prec_pct = precision_score(y_test, y_pred_test) * 100
        f1_pct = f1_score(y_test, y_pred_test) * 100

        resultados_torneo.append({
            "Modelo Evaluado": nombre,
            "Capacidad_Discriminativa_AUC": f"{roc_pct:.1f}%",
            "Bajas_Detectadas_Recall": f"{rec_pct:.1f}%",
            "Acierto_de_Alerta_Precision": f"{prec_pct:.1f}%",
            "Equilibrio_Global_F1": f"{f1_pct:.1f}%",
            "_auc_num": roc_pct,
            "_rec_num": rec_pct,
            "_f1_num": f1_pct
        })

        fpr, tpr, _ = roc_curve(y_test, y_prob_test)
        curvas_roc[nombre] = (fpr * 100, tpr * 100, roc_pct)

    df_bench = pd.DataFrame(resultados_torneo).sort_values(by="_auc_num", ascending=False)
    print("\n--- COMPARATIVA DE RENDIMIENTO DE MODELOS (EN PORCENTAJES DE 0% A 100%) ---")
    cols_mostrar = [
        "Modelo Evaluado", "Capacidad_Discriminativa_AUC",
        "Bajas_Detectadas_Recall", "Acierto_de_Alerta_Precision", "Equilibrio_Global_F1"
    ]
    print(df_bench[cols_mostrar].to_string(index=False))

    # ==========================================================================
    # GRÁFICO 1: COMPARATIVA VISUAL DE MODELOS EN PORCENTAJES (0% A 100%)
    # ==========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel Izquierdo: Barras agrupadas con % explícito
    nombres_mod = df_bench["Modelo Evaluado"].tolist()
    x_pos = np.arange(len(nombres_mod))
    width = 0.35

    bars1 = axes[0].bar(x_pos - width/2, df_bench["_auc_num"], width, label="Poder Predictivo Global (AUC %)", color="#1f4e78")
    bars2 = axes[0].bar(x_pos + width/2, df_bench["_rec_num"], width, label="% de Bajas Reales Capturadas (Recall)", color="#2e8b57")

    axes[0].set_title("1. Comparación de Efectividad por Modelo (%)", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_ylabel("Efectividad (Escala de 0% a 100%)", fontsize=11)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(nombres_mod, fontsize=10, fontweight="bold")
    axes[0].set_ylim(0, 110)
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[0].legend(loc="upper right", frameon=True)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        axes[0].annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold"
        )

    # Panel Derecho: Curva de Detección en %
    colores_curva = {"Gradient Boosting (Campeón)": "#d95f02", "Random Forest": "#1f78b4", "Regresión Logística": "#7570b3"}
    for nombre, (fpr_pct, tpr_pct, auc_val) in curvas_roc.items():
        axes[1].plot(fpr_pct, tpr_pct, lw=2.5, color=colores_curva.get(nombre, "#333333"), label=f"{nombre} ({auc_val:.1f}%)")

    axes[1].plot([0, 100], [0, 100], "k--", alpha=0.4, label="Azar (50.0%)")
    axes[1].set_title("2. Curva de Captura de Fugas vs. Falsas Alarmas", fontsize=13, fontweight="bold", pad=12)
    axes[1].set_xlabel("% de Falsas Alarmas sobre Afiliados Activos", fontsize=11)
    axes[1].set_ylabel("% de Bajas Reales Detectadas a Tiempo", fontsize=11)
    axes[1].xaxis.set_major_formatter(mtick.PercentFormatter())
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[1].legend(loc="lower right", frameon=True)

    plt.suptitle("Evaluación de Modelos Predictivos de Fuga (Unidades: Porcentaje de 0% a 100%)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "01_model_comparison_roc_pr.png"), dpi=180, bbox_inches="tight")
    plt.close()

    # ==========================================================================
    # GRÁFICO 2: OPTIMIZACIÓN ECONÓMICA EN MILLONES DE PESOS ($ ARS)
    # ==========================================================================
    nombre_campeon = df_bench.iloc[0]["Modelo Evaluado"]
    pipe_campeon = pipelines_entrenados[nombre_campeon]
    y_prob_champ = pipe_campeon.predict_proba(X_test)[:, 1]

    umbrales_dec = np.linspace(0.10, 0.90, 81)
    ahorros_millones_ars = []
    for umb in umbrales_dec:
        ahorro_ars, _, _, _, _ = calcular_ahorro_en_pesos(y_test, y_prob_champ, umb)
        ahorros_millones_ars.append(ahorro_ars / 1_000_000.0)

    mejor_idx = int(np.argmax(ahorros_millones_ars))
    umbral_optimo_dec = round(float(umbrales_dec[mejor_idx]), 2)
    umbral_optimo_pct = int(round(umbral_optimo_dec * 100))
    ahorro_max_ars, vp, fp, fn, vn = calcular_ahorro_en_pesos(y_test, y_prob_champ, umbral_optimo_dec)
    ahorro_max_millones = ahorro_max_ars / 1_000_000.0

    plt.figure(figsize=(10.5, 5.8))
    umbrales_pct_eje = umbrales_dec * 100
    plt.plot(umbrales_pct_eje, ahorros_millones_ars, color="#1f4e78", lw=3, label="Ahorro Neto según Punto de Corte")
    plt.fill_between(umbrales_pct_eje, ahorros_millones_ars, 0, where=(np.array(ahorros_millones_ars) > 0), color="#1f4e78", alpha=0.12)

    plt.axvline(umbral_optimo_pct, color="#d62728", linestyle="--", lw=2.2)
    plt.scatter([umbral_optimo_pct], [ahorro_max_millones], color="#d62728", s=90, zorder=5)
    plt.annotate(
        f"PUNTO DE CORTE ÓPTIMO: {umbral_optimo_pct}% de probabilidad\n"
        f"Ahorro Neto: $ {ahorro_max_millones:,.1f} Millones ARS\n"
        f"({vp} bajas detectadas | {fp} alertas preventivas)",
        xy=(umbral_optimo_pct, ahorro_max_millones),
        xytext=(umbral_optimo_pct + 7, ahorro_max_millones * 0.78),
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.8),
        fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff8f8", edgecolor="#d62728")
    )

    plt.title(
        "Optimización Económica: ¿A partir de qué % de riesgo conviene llamar al afiliado?\n"
        f"(Valuación sobre muestra de 2.000 afiliados | LTV = $ {VALOR_VIDA_AFILIADO_ARS:,.0f} ARS | Costo Gestión = $ {COSTO_GESTION_RETENCION_ARS:,.0f} ARS)",
        fontsize=12, fontweight="bold", pad=12
    )
    plt.xlabel("Umbral de Probabilidad de Baja para Disparar Alerta (%)", fontsize=11)
    plt.ylabel("Ahorro Económico Neto en Millones de Pesos ($ ARS)", fontsize=11)
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter())
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f"$ {x:,.0f} M"))
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "02_financial_threshold_optimization.png"), dpi=180)
    plt.close()

    # ==========================================================================
    # GRÁFICO 3: IMPORTANCIA DE VARIABLES EN PORCENTAJE RELATIVO (SUMA = 100%)
    # ==========================================================================
    print("\nCalculando Peso Relativo (%) de cada factor en la decisión de baja...")
    perm_imp = permutation_importance(
        pipe_campeon, X_test, y_test, n_repeats=8, random_state=RANDOM_SEED, scoring="roc_auc", n_jobs=-1
    )

    nombres_claros = {
        "indice_friccion_afiliado_0a100": "Índice General de Fricción (0 a 100 pts)",
        "indice_tono_llamada_1a10": "Tono / Clima en Llamadas (Escala 1 a 10)",
        "reclamos_sin_resolver_cant": "Reclamos Abiertos Sin Resolver (Cantidad)",
        "antiguedad_meses": "Antigüedad del Afiliado (Meses)",
        "ordenes_rechazadas_trimestre_cant": "Órdenes Médicas Rechazadas (Cantidad)",
        "demora_reintegro_dias": "Demora en Pago de Reintegros (Días)",
        "aumento_cuota_trimestral_pct": "Aumento Trimestral de Cuota (%)",
        "facturas_impagas_cant": "Facturas Impagas en Mora (Cantidad)",
        "alerta_doble_friccion_bin": "Doble Fricción: Rechazo Médico + Reclamo",
        "llamadas_reincidentes_48hs_cant": "Llamadas Reincidentes en 48 hs (Cantidad)",
        "tasa_rechazo_medico_pct": "Tasa de Rechazo en Auditoría Médica (%)",
        "adhesion_debito_automatico": "Adhesión a Débito Automático (Sí/No)",
        "tipo_contratacion": "Tipo de Contratación (Voluntario / Obligatorio)",
        "ingresos_app_trimestre_cant": "Uso de App Móvil en el Trimestre (Ingresos)",
        "tasa_resolucion_primer_contacto_pct": "Resolución en Primer Contacto - FCR (%)"
    }

    imp_positiva = np.maximum(0, perm_imp.importances_mean)
    suma_imp = imp_positiva.sum() if imp_positiva.sum() > 0 else 1.0
    peso_relativo_pct = (imp_positiva / suma_imp) * 100.0

    df_imp = pd.DataFrame({
        "Factor Explicativo": [nombres_claros.get(c, c) for c in feature_cols],
        "Peso_Porcentual": peso_relativo_pct
    }).sort_values(by="Peso_Porcentual", ascending=True).tail(10)

    plt.figure(figsize=(11, 6.2))
    bars = plt.barh(df_imp["Factor Explicativo"], df_imp["Peso_Porcentual"], color="#2b5c8f", height=0.68)
    plt.title(
        "¿Qué motiva la baja de un afiliado? - Top 10 Factores por Peso Relativo (%)\n"
        "(Expresado como porcentaje de influencia sobre el 100% del poder predictivo del modelo)",
        fontsize=12.5, fontweight="bold", pad=12
    )
    plt.xlabel("Peso Relativo en la Predicción de Fuga (%)", fontsize=11)
    plt.xlim(0, df_imp["Peso_Porcentual"].max() * 1.18)
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter())

    for bar in bars:
        w = bar.get_width()
        plt.annotate(
            f" {w:.1f}%",
            xy=(w, bar.get_y() + bar.get_height() / 2),
            va="center", ha="left", fontsize=10, fontweight="bold", color="#1a3a5c"
        )

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "03_feature_importance.png"), dpi=180)
    plt.close()

    # ==========================================================================
    # GRÁFICO 4: MATRIZ DE CONFUSIÓN EXPLICADA CON CANTIDADES Y PORCENTAJES
    # ==========================================================================
    y_pred_opt = (y_prob_champ >= umbral_optimo_dec).astype(int)
    cm = confusion_matrix(y_test, y_pred_opt)
    total_muestra = cm.sum()

    etiquetas_celdas = np.array([
        [
            f"AFILIADOS ESTABLES\nBIEN CLASIFICADOS\n\n{vn:,} afiliados\n({vn/total_muestra*100:.1f}% del total)",
            f"ALERTAS PREVENTIVAS\n(Falsas Alarmas)\n\n{fp:,} afiliados\n({fp/total_muestra*100:.1f}% del total)"
        ],
        [
            f"FUGAS NO DETECTADAS\n(Falsos Negativos)\n\n{fn:,} afiliados\n({fn/total_muestra*100:.1f}% del total)",
            f"BAJAS REALES\nDETECTADAS A TIEMPO\n\n{vp:,} afiliados\n({vp/total_muestra*100:.1f}% del total)"
        ]
    ])

    plt.figure(figsize=(8.2, 6.2))
    sns.heatmap(
        cm, annot=etiquetas_celdas, fmt="", cmap="Blues", cbar=False,
        annot_kws={"size": 10.5, "weight": "bold"},
        xticklabels=["Predicción: Permanece Activo", "Predicción: Alerta de Baja"],
        yticklabels=["Realidad: Permanece Activo", "Realidad: Se da de Baja"]
    )
    plt.title(
        f"Resultado Operativo sobre 2.000 Afiliados de Prueba (Corte = {umbral_optimo_pct}%)\n"
        f"Se detectan {vp} de las {vp + fn} bajas reales ({vp/(vp+fn)*100:.1f}% de efectividad de captura)",
        fontsize=12, fontweight="bold", pad=14
    )
    plt.ylabel("Comportamiento Real del Afiliado", fontsize=11, fontweight="bold")
    plt.xlabel("Decisión del Sistema de Alerta Temprana", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "04_confusion_matrix_optimal.png"), dpi=180)
    plt.close()

    print("\n--- RESUMEN ECONÓMICO Y OPERATIVO DEL MODELO CAMPEÓN ---")
    print(f" -> Modelo Ganador: {nombre_campeon}")
    print(f" -> Punto de Corte Óptimo: {umbral_optimo_pct}% de probabilidad de baja")
    print(f" -> Bajas Reales Capturadas a Tiempo: {vp} de {vp + fn} ({vp/(vp+fn)*100:.1f}%)")
    print(f" -> Ahorro Neto en Muestra de Prueba (2.000 afiliados): $ {ahorro_max_ars:,.0f} ARS ($ {ahorro_max_millones:.2f} Millones)")
    print(f" -> Ahorro Neto Proyectado al Padrón Total (10.000 afiliados): $ {ahorro_max_ars * 5:,.0f} ARS\n")

    # Guardar artefactos y metadatos
    joblib.dump(pipe_campeon, os.path.join(MODELS_DIR, "champion_churn_pipeline.joblib"))
    metadata = {
        "modelo_campeon": nombre_campeon,
        "umbral_optimo_decimal": umbral_optimo_dec,
        "umbral_optimo_porcentaje": f"{umbral_optimo_pct}%",
        "roc_auc_porcentaje": round(float(df_bench.iloc[0]["_auc_num"]), 2),
        "ahorro_neto_muestra_ars": round(float(ahorro_max_ars), 2),
        "ahorro_neto_padron_10k_ars": round(float(ahorro_max_ars * 5), 2),
        "columnas_predictoras": feature_cols
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    train_and_select_champion()