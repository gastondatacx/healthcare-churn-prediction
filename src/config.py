import os

# Rutas del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")

for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR]:
    os.makedirs(directory, exist_ok=True)

RANDOM_SEED = 42
N_AFILIADOS = 10000

# ==============================================================================
# PARÁMETROS ECONÓMICOS DE NEGOCIO (EXPRESADOS EN PESOS ARGENTINOS - $ ARS)
# ==============================================================================
# Valor de Vida del Afiliado (LTV) promedio perdido cuando un titular se da de baja
# Equivale aprox. a 12 meses de aportes/cuota neta de margen ($ 1.800.000 ARS)
VALOR_VIDA_AFILIADO_ARS = 1_800_000.0

# Costo operativo promedio de ejecutar una acción de retención por afiliado alertado
# Incluye tiempo de agente especializado + bonificación comercial temporal ($ 95.000 ARS)
COSTO_GESTION_RETENCION_ARS = 95_000.0

# Porcentaje de éxito estimado cuando se interviene a tiempo a un afiliado en riesgo (45%)
TASA_EXITO_RETENCION = 0.45