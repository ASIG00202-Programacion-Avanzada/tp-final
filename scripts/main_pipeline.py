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
    
    def evaluate_models(self, model_results):
        """
        Evalúa los modelos y genera métricas detalladas.
        
        Args:
            model_results: Resultados de los modelos
            
        Returns:
            dict: Métricas de evaluación
        """
        logger.info("Evaluando modelos...")
        
        evaluation_metrics = {}
        
        for model_name, results in model_results.items():
            metrics = {
                'test_rmse': results['test_rmse'],
                'test_mae': results['test_mae'],
                'test_r2': results['test_r2'],
                'cv_rmse_mean': results['cv_mean'],
                'cv_rmse_std': results['cv_std']
            }
            
            evaluation_metrics[model_name] = metrics
            
            # Guardar en base de datos
            self.db_manager.store_model_results(
                model_name=model_name,
                model_version="1.0",
                metrics=metrics,
                hyperparameters={},
                feature_importance=None
            )
        
        return evaluation_metrics
    
    def create_visualizations(self, processed_data, model_results):
        """
        Crea visualizaciones de los resultados.
        
        Args:
            processed_data: Datos preprocesados
            model_results: Resultados de los modelos
        """
        logger.info("Creando visualizaciones...")
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Comparación de modelos
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Métricas de comparación
        model_names = list(model_results.keys())
        rmse_scores = [results['test_rmse'] for results in model_results.values()]
        r2_scores = [results['test_r2'] for results in model_results.values()]
        
        # RMSE por modelo
        axes[0, 0].bar(model_names, rmse_scores)
        axes[0, 0].set_title('RMSE por Modelo')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # R² por modelo
        axes[0, 1].bar(model_names, r2_scores)
        axes[0, 1].set_title('R² por Modelo')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Predicciones vs Valores reales (mejor modelo)
        best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['test_r2'])
        best_results = model_results[best_model_name]
        
        y_test = processed_data['y_test']
        y_pred = best_results['predictions']['test']
        
        axes[1, 0].scatter(y_test, y_pred, alpha=0.6)
        axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('Valores Reales')
        axes[1, 0].set_ylabel('Predicciones')
        axes[1, 0].set_title(f'Predicciones vs Reales - {best_model_name}')
        
        # Distribución de errores
        errors = y_test - y_pred
        axes[1, 1].hist(errors, bins=30, alpha=0.7)
        axes[1, 1].set_xlabel('Error de Predicción')
        axes[1, 1].set_ylabel('Frecuencia')
        axes[1, 1].set_title('Distribución de Errores')
        
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature importance (si está disponible)
        if hasattr(best_results['model'], 'feature_importances_'):
            feature_importance = best_results['model'].feature_importances_
            feature_names = processed_data['feature_names']
            
            # Crear DataFrame para feature importance
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            # Plot feature importance
            plt.figure(figsize=(10, 8))
            sns.barplot(data=importance_df.head(15), x='importance', y='feature')
            plt.title(f'Feature Importance - {best_model_name}')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig(REPORTS_DIR / 'feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("Visualizaciones guardadas en reports/")
    
    def generate_report(self, model_results, evaluation_metrics):
        """
        Genera un reporte final del análisis.
        
        Args:
            model_results: Resultados de los modelos
            evaluation_metrics: Métricas de evaluación
        """
        logger.info("Generando reporte final...")
        
        # Crear reporte en Markdown
        report_content = f"""
# Reporte de Análisis de Precios de Propiedades - Properati Argentina

## Resumen Ejecutivo

Este análisis utiliza técnicas de machine learning para predecir precios de propiedades en Argentina utilizando el dataset de Properati.

## Metodología

1. **Preprocesamiento de Datos**: Limpieza, feature engineering y normalización
2. **Modelado**: Comparación de múltiples algoritmos de regresión
3. **Evaluación**: Métricas RMSE, MAE y R²
4. **Almacenamiento**: Base de datos para persistencia de resultados

## Resultados de Modelos

| Modelo | RMSE | MAE | R² |
|-------|------|-----|-----|
"""
        
        for model_name, results in model_results.items():
            report_content += f"| {model_name} | {results['test_rmse']:.4f} | {results['test_mae']:.4f} | {results['test_r2']:.4f} |\n"
        
        # Mejor modelo
        best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['test_r2'])
        best_results = model_results[best_model_name]
        
        report_content += f"""
## Mejor Modelo: {best_model_name}

- **R²**: {best_results['test_r2']:.4f}
- **RMSE**: {best_results['test_rmse']:.4f}
- **MAE**: {best_results['test_mae']:.4f}

## Conclusiones

1. El modelo {best_model_name} mostró el mejor rendimiento con un R² de {best_results['test_r2']:.4f}
2. Las métricas indican una capacidad predictiva {'excelente' if best_results['test_r2'] > 0.8 else 'buena' if best_results['test_r2'] > 0.6 else 'moderada'}
3. Se recomienda continuar con la optimización de hiperparámetros para mejorar el rendimiento

## Archivos Generados

- `model_comparison.png`: Comparación visual de modelos
- `feature_importance.png`: Importancia de características
- `model_comparison.csv`: Tabla de resultados
- `data_exploration.json`: Información de exploración de datos

---
*Reporte generado el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # Guardar reporte
        report_file = REPORTS_DIR / "analysis_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Reporte guardado en: {report_file}")
    
    def run_complete_pipeline(self, data_path: str = None):
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


