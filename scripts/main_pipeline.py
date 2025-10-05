#!/usr/bin/env python3
"""
Script principal para el análisis de precios de propiedades de Properati Argentina.
Integra todo el pipeline: carga de datos, preprocesamiento, modelado y evaluación.
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_processing import DataProcessor
from models import ModelTrainer
from database import DatabaseManager


from large_data_handler import LargeDataHandler
from config import *

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reports/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProperatiAnalysisPipeline:
    """
    Pipeline completo para el análisis de precios de propiedades.
    """
    
    def __init__(self):
        self.data_processor = DataProcessor()
        self.model_trainer = ModelTrainer()
        self.db_manager = DatabaseManager()
        self.large_data_handler = LargeDataHandler()
        self.results = {}
        
    def download_data(self, kaggle_dataset: str = "alejandroczernikier/properati-argentina-dataset"):
        """
        Descarga el dataset de Kaggle.
        
        Args:
            kaggle_dataset: Nombre del dataset en Kaggle
        """
        try:
            import kaggle
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            api = KaggleApi()
            api.authenticate()
            
            # Descargar dataset
            api.dataset_download_files(
                kaggle_dataset, 
                path=str(RAW_DATA_DIR), 
                unzip=True
            )
            
            logger.info(f"Dataset descargado exitosamente en {RAW_DATA_DIR}")
            return True
            
        except Exception as e:
            logger.error(f"Error descargando dataset: {e}")
            logger.info("Por favor, descarga manualmente el dataset desde Kaggle")
            return False
    
    def load_and_explore_data(self, data_path: str = None, sample_size: int = 50000):
        """
        Carga y explora los datos, manejando datasets grandes.
        
        Args:
            data_path: Ruta al archivo de datos
            sample_size: Tamaño de la muestra para datasets grandes
            
        Returns:
            pd.DataFrame: Dataset cargado o muestreado
        """
        if data_path is None:
            # Buscar archivos CSV en el directorio raw
            csv_files = list(RAW_DATA_DIR.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError("No se encontraron archivos CSV en el directorio raw")
            data_path = csv_files[0]
        
        logger.info(f"Analizando dataset: {data_path}")
        
        # Obtener información del dataset
        dataset_info = self.large_data_handler.get_dataset_info(str(data_path))
        
        if dataset_info.get('total_rows', 0) > sample_size:
            logger.info(f"Dataset grande detectado: {dataset_info['total_rows']:,} filas")
            logger.info(f"Creando muestra estratificada de {sample_size:,} filas...")
            
            # Crear muestra estratificada
            df = self.large_data_handler.create_stratified_sample(
                str(data_path), 
                sample_size=sample_size,
                target_column='price_usd'
            )
            
            # Guardar muestra para uso futuro
            sample_path = PROCESSED_DATA_DIR / "dataset_sample.csv"
            df.to_csv(sample_path, index=False)
            logger.info(f"Muestra guardada en: {sample_path}")
            
        else:
            logger.info("Dataset de tamaño normal, cargando completo...")
            df = self.data_processor.load_data(str(data_path))
        
        # Optimizar DataFrame
        df = self.large_data_handler.optimize_dataframe(df)
        
        # Exploración básica
        logger.info(f"Shape del dataset: {df.shape}")
        logger.info(f"Columnas: {list(df.columns)}")
        logger.info(f"Memoria utilizada: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Guardar información de exploración
        exploration_info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'original_dataset_info': dataset_info
        }
        
        # Guardar en archivo
        exploration_file = REPORTS_DIR / "data_exploration.json"
        import json
        with open(exploration_file, 'w') as f:
            json.dump(exploration_info, f, indent=2, default=str)
        
        return df
    
    def preprocess_data(self, df):
        """
        Preprocesa los datos usando el pipeline completo.
        
        Args:
            df: Dataset original
            
        Returns:
            dict: Resultados del preprocesamiento
        """
        logger.info("Iniciando preprocesamiento de datos...")
        
        # Pipeline completo de procesamiento
        X_train, X_test, y_train, y_test, preprocessor = self.data_processor.process_pipeline(
            df, target_column='price_usd'
        )
        
        # Aplicar preprocesamiento
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Obtener nombres de características
        feature_names = self.data_processor.get_feature_importance_names(preprocessor)
        
        logger.info(f"Datos preprocesados. Train: {X_train_processed.shape}, Test: {X_test_processed.shape}")
        
        return {
            'X_train': X_train_processed,
            'X_test': X_test_processed,
            'y_train': y_train,
            'y_test': y_test,
            'preprocessor': preprocessor,
            'feature_names': feature_names
        }
    
    def train_models(self, processed_data):
        """
        Entrena y compara múltiples modelos.
        
        Args:
            processed_data: Datos preprocesados
            
        Returns:
            dict: Resultados de todos los modelos
        """
        logger.info("Iniciando entrenamiento de modelos...")
        
        # Entrenar todos los modelos
        results = self.model_trainer.compare_models(
            processed_data['X_train'],
            processed_data['y_train'],
            processed_data['X_test'],
            processed_data['y_test']
        )
        
        # Obtener resumen de modelos
        model_summary = self.model_trainer.get_model_summary()
        
        # Guardar resumen
        summary_file = REPORTS_DIR / "model_comparison.csv"
        model_summary.to_csv(summary_file, index=False)
        logger.info(f"Resumen de modelos guardado en: {summary_file}")
        
        return results
    

        """
        Ejecuta el pipeline completo de análisis.
        
        Args:
            data_path: Ruta al archivo de datos (opcional)
        """
        logger.info("=== INICIANDO PIPELINE COMPLETO ===")
        
        try:
            # 1. Cargar datos
            logger.info("Paso 1: Cargando datos...")
            df = self.load_and_explore_data(data_path)
            
            # 2. Preprocesar datos
            logger.info("Paso 2: Preprocesando datos...")
            processed_data = self.preprocess_data(df)
            
            # 3. Entrenar modelos
            logger.info("Paso 3: Entrenando modelos...")
            model_results = self.train_models(processed_data)
            
            # 4. Evaluar modelos
            logger.info("Paso 4: Evaluando modelos...")
            evaluation_metrics = self.evaluate_models(model_results)
            
            # 5. Crear visualizaciones
            logger.info("Paso 5: Creando visualizaciones...")
            self.create_visualizations(processed_data, model_results)
            
            # 6. Generar reporte
            logger.info("Paso 6: Generando reporte...")
            self.generate_report(model_results, evaluation_metrics)
            
            logger.info("=== PIPELINE COMPLETADO EXITOSAMENTE ===")
            
            return {
                'success': True,
                'model_results': model_results,
                'evaluation_metrics': evaluation_metrics
            }
            
        except Exception as e:
            logger.error(f"Error en el pipeline: {e}")
            return {
                'success': False,
                'error': str(e)
            }

def main():
    """
    Función principal para ejecutar el pipeline.
    """
    # Crear directorios necesarios
    for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR]:
        directory.mkdir(exist_ok=True)
    
    # Inicializar pipeline
    pipeline = ProperatiAnalysisPipeline()
    
    # Ejecutar pipeline completo
    results = pipeline.run_complete_pipeline()
    
    if results['success']:
        print("\n Pipeline completado exitosamente!")
        print(f" Resultados guardados en: {REPORTS_DIR}")
        print(f"  Base de datos configurada en: {DB_CONFIG['name']}")
    else:
        print(f"\n Error en el pipeline: {results['error']}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())


