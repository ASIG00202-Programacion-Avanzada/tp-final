#!/usr/bin/env python3
"""
Script de ejemplo para ejecutar el análisis de precios de propiedades.
Este script demuestra cómo usar el proyecto paso a paso.
"""
import sys
from pathlib import Path
import logging

# Agregar src al path
sys.path.append(str(Path(__file__).parent / "src"))

from data_processing import DataProcessor
from models import ModelTrainer
from visualization import VisualizationManager
from config import *

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Función principal de ejemplo.
    """
    logger.info("=== EJEMPLO DE USO DEL PROYECTO ===")
    
    try:
        # 1. Inicializar componentes
        logger.info("1. Inicializando componentes...")
        processor = DataProcessor()
        trainer = ModelTrainer()
        viz = VisualizationManager()
        
        # 2. Cargar datos (ejemplo con datos dummy)
        logger.info("2. Cargando datos...")
        # Nota: En un caso real, aquí cargarías el dataset de Properati
        # df = processor.load_data("data/raw/properati_dataset.csv")
        
        # Para este ejemplo, creamos datos dummy
        import pandas as pd
        import numpy as np
        
        np.random.seed(42)
        n_samples = 1000
        
        df = pd.DataFrame({
            'property_type': np.random.choice(['Casa', 'Departamento', 'PH'], n_samples),
            'location': np.random.choice(['Buenos Aires', 'Córdoba', 'Rosario'], n_samples),
            'surface_total': np.random.normal(100, 30, n_samples),
            'surface_covered': np.random.normal(80, 25, n_samples),
            'rooms': np.random.randint(1, 6, n_samples),
            'bedrooms': np.random.randint(1, 4, n_samples),
            'bathrooms': np.random.randint(1, 3, n_samples),
            'price_usd': np.random.normal(150000, 50000, n_samples)
        })
        
        # Asegurar valores positivos
        df['surface_total'] = np.abs(df['surface_total'])
        df['surface_covered'] = np.abs(df['surface_covered'])
        df['price_usd'] = np.abs(df['price_usd'])
        
        logger.info(f"Datos cargados: {df.shape}")
        
        # 3. Explorar datos
        logger.info("3. Explorando datos...")
        info = processor.explore_data(df)
        logger.info(f"Información del dataset: {info['shape']}")
        
        # 4. Procesar datos
        logger.info("4. Procesando datos...")
        X_train, X_test, y_train, y_test, preprocessor = processor.process_pipeline(df)
        logger.info(f"Datos procesados - Train: {X_train.shape}, Test: {X_test.shape}")
        
        # 5. Entrenar modelos
        logger.info("5. Entrenando modelos...")
        results = trainer.compare_models(X_train, y_train, X_test, y_test)
        logger.info(f"Modelos entrenados: {len(results)}")
        
        # 6. Mostrar resultados
        logger.info("6. Resultados de modelos:")
        summary = trainer.get_model_summary()
        print(summary)
        
        # 7. Crear visualizaciones
        logger.info("7. Creando visualizaciones...")
        
        # Crear directorio de reportes si no existe
        REPORTS_DIR.mkdir(exist_ok=True)
        
        # Visualizar distribución de precios
        viz.plot_price_distribution(df, save_path=REPORTS_DIR / "price_distribution.png")
        
        # Visualizar comparación de modelos
        viz.plot_model_comparison(results, save_path=REPORTS_DIR / "model_comparison.png")
        
        # Visualizar predicciones vs reales (mejor modelo)
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
        best_results = results[best_model_name]
        viz.plot_predictions_vs_actual(
            y_test, 
            best_results['predictions']['test'], 
            best_model_name,
            save_path=REPORTS_DIR / "predictions_vs_actual.png"
        )
        
        logger.info("Ejemplo completado exitosamente!")
        logger.info(f"Visualizaciones guardadas en: {REPORTS_DIR}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error en el ejemplo: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
