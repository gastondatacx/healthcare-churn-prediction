from src.data_generator import generate_relational_healthcare_data
from src.feature_engineering import build_analytical_base_table
from src.train_and_evaluate import train_and_select_champion
from src.batch_scoring_prescriptive import run_prescriptive_batch_scoring


def main():
    print("=========================================================================")
    print(" SISTEMA PREDICTIVO DE FUGA DE AFILIADOS Y MOTOR DE RETENCIÓN EN SALUD   ")
    print("=========================================================================\n")

    # Etapa 1: Simulación Relacional de Padrón, Auditoría, CX y Facturación
    generate_relational_healthcare_data()

    # Etapa 2: Ingeniería de Variables y Creación de la Tabla Base Analítica (ABT)
    build_analytical_base_table()

    # Etapa 3: Torneo de Modelos, Optimización Financiera y Gráficos en Español
    train_and_select_champion()

    # Etapa 4: Calificación de Cartera y Plan de Acción Prescriptivo para CRM
    run_prescriptive_batch_scoring()

    print("\n=========================================================================")
    print(" PIPELINE FINALIZADO EXITOSAMENTE                                        ")
    print("=========================================================================")


if __name__ == "__main__":
    main()