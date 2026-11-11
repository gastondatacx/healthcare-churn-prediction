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

# Parámetros Financieros de Negocio (en USD equivalentes para estandarización)
# Valor de Vida del Afiliado (LTV) perdido si se da de baja sin ser retenido (Falso Negativo)
COSTO_AFILIADO_PERDIDO_LTV = 1200.0

# Costo operativo de ejecutar una acción de retención (gestión proactiva + beneficio/descuento)
COSTO_ACCION_RETENCION = 85.0

# Tasa de éxito estimada cuando se contacta proactivamente a un afiliado en riesgo
TASA_EXITO_RETENCION = 0.45