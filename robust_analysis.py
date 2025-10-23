#!/usr/bin/env python3
"""
Script de Entrenamiento y Evaluación.

- Lee datos limpios desde la tabla 'input_data' de SQLite.
- Implementa un pipeline de preprocesamiento (ColumnTransformer) para
  evitar fugas de datos.
- Guarda los resultados (métricas e hiperparámetros) en la
  tabla 'model_results' de SQLite.
- Guarda los artefactos (preprocesador y modelos) en /models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib  
import os
import sys
from pathlib import Path
import logging


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.database import DatabaseManager
import config

# Imports de Scikit-learn para Pipelines
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data_from_db(db_manager):
    """
    Carga los datos de entrada preprocesados desde la base de datos.
    """
    logger.info("Cargando datos limpios desde la tabla 'input_data'...")
    try:
        df = db_manager.get_input_data() # Esta función debe devolver un DataFrame
        if df.empty:
            logger.error("La tabla 'input_data' está vacía.")
            logger.error("Por favor, ejecuta 'python src/data_processing.py' primero.")
            return None
        logger.info(f"Datos cargados desde la DB. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error al cargar datos desde la DB: {e}")
        return None

def create_preprocessor(X_train):
    """
    Crea el ColumnTransformer (pipeline de preprocesamiento) basado
    en las columnas del DataFrame de entrenamiento.
    """
    logger.info("Creando el pipeline de preprocesamiento (ColumnTransformer)...")
    
    # Identificar columnas automáticamente
    # Excluir columnas de ID o texto que no son features
    non_feature_cols = ['id','property_id', 'location', 'state_name', 'raw_data', 'created_at']
    
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    
    # Usar .select_dtypes(['object', 'category']) para features categóricas
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Asegurarse de que las columnas no-features no estén en las listas
    numeric_features = [col for col in numeric_features if col not in non_feature_cols]
    categorical_features = [col for col in categorical_features if col not in non_feature_cols]
    
    logger.info(f"Features numéricas identificadas: {numeric_features}")
    logger.info(f"Features categóricas identificadas: {categorical_features}")

    # Pipeline para características numéricas
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Imputación
        ('scaler', StandardScaler()) # Escalado
    ])
    
    # Pipeline para características categóricas
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Imputación
        ('onehot', OneHotEncoder(handle_unknown='ignore')) # One-Hot Encoding
    ])
    
    # Combinar preprocesadores en un ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' # Descarta columnas no especificadas (como 'location')
    )
    
    return preprocessor


def train_models(X_train, X_test, y_train, y_test, preprocessor, db_manager):
    """
    Entrena, evalúa y guarda los modelos y sus resultados en la DB.
    """
    logger.info("Entrenando modelos...")
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    results = {}
    
    # 1. Aplicar el preprocesamiento (fit_transform en train, transform en test)
    logger.info("Aplicando preprocesamiento a los datos (fit en train)...")
    X_train_processed = preprocessor.fit_transform(X_train)
    
    logger.info("Aplicando preprocesamiento a los datos (transform en test)...")
    X_test_processed = preprocessor.transform(X_test)
    
    for name, model in models.items():
        logger.info(f"    Entrenando {name}...")
        
        try:
            # 2. Entrenar el modelo con los datos procesados
            model.fit(X_train_processed, y_train)
            
            # 3. Evaluar
            y_pred_test = model.predict(X_test_processed)
            
            metrics = {
                'test_r2': r2_score(y_test, y_pred_test),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'test_mae': mean_absolute_error(y_test, y_pred_test)
            }
            
            logger.info(f"  {name}: R² = {metrics['test_r2']:.4f}, RMSE = {metrics['test_rmse']:.2f}")

            # 4. Preparar datos para la DB
            hyperparams = model.get_params()
            
            # (Opcional) Obtener feature importance si existe
            feat_importance_dict = None
            if hasattr(model, 'feature_importances_'):
                # Obtener los nombres de las features del preprocesador
                try:
                    feature_names = preprocessor.get_feature_names_out()
                    feat_importance_dict = dict(zip(feature_names, model.feature_importances_))
                except Exception:
                    feat_importance_dict = {"error": "No se pudieron obtener feature names"}
            
            # 5. Guardar resultados en la DB
            db_manager.store_model_results(
                model_name=name,
                model_version="1.0",
                metrics=metrics,
                hyperparameters=hyperparams,
                feature_importance=feat_importance_dict
            )
            
            results[name] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred_test
            }
            
        except Exception as e:
            logger.error(f"  Error entrenando {name}: {e}")
            continue
    
    return results

def create_visualizations(df, results, y_test):
    """
    Crea visualizaciones y las guarda en /reports.
    (Modificado para recibir y_test)
    """
    logger.info("Creando visualizaciones...")
    os.makedirs('reports', exist_ok=True)
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Gráfico 1: Distribución de Precios (del df original)
    sns.histplot(df['price_usd'], bins=50, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Distribución de Precios (Datos Limpios)')
    
    # Gráfico 2: Precio vs Superficie (del df original)
    sample_df = df.sample(n=min(2000, len(df)))
    sns.scatterplot(data=sample_df, x='surface_total', y='price_usd', 
                    hue='property_type', alpha=0.6, ax=axes[0, 1])
    axes[0, 1].set_title('Precio vs Superficie (por Tipo)')
    
    # Gráfico 3: Comparación de Modelos (R²)
    if results:
        model_names = list(results.keys())
        r2_scores = [results[name]['metrics']['test_r2'] for name in model_names]
        sns.barplot(x=model_names, y=r2_scores, ax=axes[1, 0])
        axes[1, 0].set_title('Comparación de Modelos (R²)')
        axes[1, 0].set_ylabel('R² Score')
    
    # Gráfico 4: Predicciones vs Reales (Mejor Modelo)
    if results:
        best_model_name = max(results, key=lambda name: results[name]['metrics']['test_r2'])
        y_pred = results[best_model_name]['predictions']
        
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, ax=axes[1, 1])
        axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1, 1].set_xlabel('Valores Reales (price_usd)')
        axes[1, 1].set_ylabel('Predicciones (price_usd)')
        axes[1, 1].set_title(f'Predicciones vs Reales - {best_model_name}')
    
    plt.tight_layout()
    plt.savefig('reports/analysis_results.png', dpi=300, bbox_inches='tight')
    logger.info("Visualizaciones guardadas en reports/analysis_results.png")

def main():
    logger.info("=== INICIANDO PIPELINE DE ENTRENAMIENTO Y EVALUACIÓN ===")
    
    # 1. Conectar a la DB y cargar datos
    db_manager = DatabaseManager()
    df = load_data_from_db(db_manager)
    
    if df is None:
        return

    # 2. Definir Target y Features (X, y)
    target_column = 'price_usd'
    
    # Columnas a excluir de las features (X)
    # 'raw_data' , 'created_at' y 'id' son de la DB, 'property_id' es un identificador
    columns_to_drop = [target_column, 'id','property_id', 'created_at', 'raw_data']
    
    y = df[target_column]
    X = df.drop(columns=columns_to_drop, errors='ignore')
    
    # 3. Dividir datos (ANTES de preprocesar)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Datos divididos - Train: {X_train.shape}, Test: {X_test.shape}")
    
    # 4. Crear el preprocesador (se ajusta solo en X_train)
    preprocessor = create_preprocessor(X_train)
    
    # 5. Entrenar modelos y guardar resultados en DB
    results = train_models(X_train, X_test, y_train, y_test, preprocessor, db_manager)
    
    if not results:
        logger.error("No se pudieron entrenar modelos.")
        return

    # 6. Guardar artefactos (Preprocesador y Modelos)
    os.makedirs('models', exist_ok=True)
    
    # Guardar el preprocesador
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    logger.info("Pipeline de preprocesamiento guardado en 'models/preprocessor.joblib'")
    
    # Guardar los modelos entrenados
    if 'Linear Regression' in results:
        joblib.dump(results['Linear Regression']['model'], 'models/linear_regression.joblib')
        logger.info("Modelo de Regresión Lineal guardado.")
        
    if 'Random Forest' in results:
        joblib.dump(results['Random Forest']['model'], 'models/random_forest.joblib')
        logger.info("Modelo de Random Forest guardado.")
    
    # 7. Crear Visualizaciones
    create_visualizations(df, results, y_test)
    
    logger.info("=== PIPELINE DE ENTRENAMIENTO FINALIZADO ===")
    return results

if __name__ == "__main__":
    results = main()