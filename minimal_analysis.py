#!/usr/bin/env python3
"""
Script de Entrenamiento y Evaluación Robusta (v4 - Reportes Unificados).

- Lee datos limpios desde la tabla 'input_data' de SQLite.
- Implementa un pipeline de preprocesamiento (ColumnTransformer).
- Evalúa 2 algoritmos de regresión usando validación cruzada 5-fold.
- Genera TODOS los reportes en /reports:
    - model_comparison.png (Gráfico 2x2 general)
    - feature_importance.png (Top 20 features del mejor modelo)
    - analysis_report.md (Resumen ejecutivo)
    - model_comparison.csv (Tabla detallada de métricas)
    - data_exploration.json (Metadatos del dataset)
- Guarda pipelines entrenados en /models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib  
import os
import sys
import json
from pathlib import Path
import logging

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT)) 

from src.database import DatabaseManager


from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =========================================
# FUNCIONES AUXILIARES
# =========================================

def load_data_from_db(db_manager):
    logger.info("Cargando datos limpios desde la tabla 'input_data'...")
    try:
        df = db_manager.get_input_data()
        if df.empty:
            logger.error("La tabla 'input_data' está vacía.")
            return None
        logger.info(f"Datos cargados. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error al cargar datos: {e}")
        return None

def create_preprocessor(X_train):
    logger.info("Creando pipeline de preprocesamiento...")
    non_feature_cols = ['id','property_id', 'location', 'state_name', 'raw_data', 'created_at']
    numeric_features = [col for col in X_train.select_dtypes(include=np.number).columns if col not in non_feature_cols]
    categorical_features = [col for col in X_train.select_dtypes(include=['object', 'category']).columns if col not in non_feature_cols]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    return preprocessor

def get_feature_importance(pipeline):
    try:
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        model = pipeline.named_steps['model']
        if hasattr(model, 'feature_importances_'):
            return dict(zip(feature_names, model.feature_importances_))
        if hasattr(model, 'coef_'):
            return dict(zip(feature_names, model.coef_))
        return None
    except Exception:
        return None

# =========================================
# NÚCLEO DE ENTRENAMIENTO
# =========================================

def train_models(X_train, X_test, y_train, y_test, preprocessor, db_manager):
    # Lista de modelos a evaluar
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15)
    }
    
    scoring_metrics = {'r2': 'r2', 'mae': 'neg_mean_absolute_error', 'rmse': 'neg_root_mean_squared_error'}
    results = {}

    for name, model in models.items():
        logger.info(f"--- Procesando {name} ---")
        full_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        
        try:
            # 1. Validación Cruzada (5-fold)
            logger.info(f"   Ejecutando 5-fold CV...")
            cv_results = cross_validate(full_pipeline, X_train, y_train, cv=5, scoring=scoring_metrics, n_jobs=-1)
            mean_metrics = {
                'cv_r2': np.mean(cv_results['test_r2']),
                'cv_rmse': -np.mean(cv_results['test_rmse']),
                'cv_mae': -np.mean(cv_results['test_mae'])
            }
            logger.info(f"   R² CV Promedio: {mean_metrics['cv_r2']:.4f}")

            # 2. Re-entrenamiento final
            logger.info(f"   Re-entrenando en todo el train set...")
            full_pipeline.fit(X_train, y_train)
            y_pred_test = full_pipeline.predict(X_test)
            feat_importance = get_feature_importance(full_pipeline)

            # 3. Guardar resultados
            db_manager.store_model_results(name, "1.0_cv5", mean_metrics, model.get_params(), feat_importance)
            results[name] = {
                'fitted_pipeline': full_pipeline,
                'cv_metrics': mean_metrics,
                'test_predictions': y_pred_test,
                'feature_importance': feat_importance
            }
        except Exception as e:
            logger.error(f"Error con {name}: {e}")
            continue
    return results

# =========================================
# GENERACIÓN DE REPORTES UNIFICADA
# =========================================

def generate_all_reports(df, results, X_test, y_test):
    """Genera todos los archivos de reporte en la carpeta /reports."""
    logger.info("=== Generando TODOS los reportes ===")
    reports_dir = 'reports'
    os.makedirs(reports_dir, exist_ok=True)
    
    # Configuración visual global
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    # --- 1. JSON de Exploración ---
    exploration_data = {
        "dataset_shape": df.shape,
        "columns": list(df.columns),
        "target_stats": df['price_usd'].describe().to_dict()
    }
    def convert_numpy(obj):
        if isinstance(obj, (np.int64, np.int32)): return int(obj)
        if isinstance(obj, (np.float64, np.float32)): return float(obj)
        raise TypeError
    with open(os.path.join(reports_dir, 'data_exploration.json'), 'w') as f:
        json.dump(exploration_data, f, indent=4, default=convert_numpy)

    # --- 2. CSV de Comparación ---
    comparison_data = []
    for name, data in results.items():
        metrics = data['cv_metrics'].copy()
        metrics['model'] = name
        comparison_data.append(metrics)
    comparison_df = pd.DataFrame(comparison_data).sort_values(by='cv_r2', ascending=False)
    comparison_df = comparison_df[['model', 'cv_r2', 'cv_rmse', 'cv_mae']] # Ordenar columnas
    comparison_df.to_csv(os.path.join(reports_dir, 'model_comparison.csv'), index=False)

    # --- 3. Markdown de Reporte ---
    best_model_name = comparison_df.iloc[0]['model']
    best_metrics = results[best_model_name]['cv_metrics']
    md_content = f"""# Reporte de Análisis Predictivo
## Resumen
- **Dataset:** {df.shape[0]} muestras.
- **Mejor Modelo:** {best_model_name} (R² CV: {best_metrics['cv_r2']:.4f})

## Comparativa (5-Fold CV)
| Modelo | R² | RMSE | MAE |
|---|---|---|---|
"""
    for _, row in comparison_df.iterrows():
        md_content += f"| {row['model']} | {row['cv_r2']:.4f} | {row['cv_rmse']:.0f} | {row['cv_mae']:.0f} |\n"
    with open(os.path.join(reports_dir, 'analysis_report.md'), 'w', encoding='utf-8') as f:
        f.write(md_content)

    # --- 4. PNG: model_comparison.png (Visión General 2x2) ---
    logger.info("Generando gráfico general (model_comparison.png)...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # 4.1 Distribución Precios
    sns.histplot(df['price_usd'], bins=50, kde=True, ax=axes[0,0], color='skyblue')
    axes[0,0].set_title('Distribución de Precios')
    
    # 4.2 Precio vs Superficie
    sample = df.sample(n=min(2000, len(df)), random_state=42)
    sns.scatterplot(data=sample, x='surface_total', y='price_usd', hue='property_type', alpha=0.5, ax=axes[0,1])
    axes[0,1].set_title('Precio vs Superficie (Muestra)')
    
    # 4.3 Comparativa R2
    sns.barplot(data=comparison_df, x='cv_r2', y='model', palette='viridis', ax=axes[1,0])
    axes[1,0].set_title('Comparativa de Modelos (R² CV)')
    axes[1,0].set_xlabel('R² Promedio')
    
    # 4.4 Predicciones vs Reales (Mejor Modelo)
    y_pred_best = results[best_model_name]['test_predictions']
    sns.scatterplot(x=y_test, y=y_pred_best, alpha=0.3, color='purple', ax=axes[1,1])
    axes[1,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1,1].set_title(f'Predicciones vs Reales ({best_model_name})')
    axes[1,1].set_xlabel('Real')
    axes[1,1].set_ylabel('Predicho')
    
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, 'model_comparison.png'), dpi=300)
    plt.close()

    # --- 5. PNG: feature_importance.png (Mejor Modelo) ---
    best_fi = results[best_model_name].get('feature_importance')
    if best_fi and isinstance(best_fi, dict):
        logger.info(f"Generando feature importance para {best_model_name}...")
        fi_df = pd.DataFrame(list(best_fi.items()), columns=['Feature', 'Importance'])
        fi_df = fi_df.sort_values(by='Importance', ascending=False).head(20)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=fi_df, x='Importance', y='Feature', palette='magma')
        plt.title(f'Top 20 Features - {best_model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, 'feature_importance.png'), dpi=300)
        plt.close()

    logger.info("Todos los reportes han sido generados en /reports")

# =========================================
# MAIN
# =========================================

def main():
    logger.info("=== INICIANDO PIPELINE FINAL (v4) ===")
    db_manager = DatabaseManager()
    df = load_data_from_db(db_manager)
    if df is None: return

    target_col = 'price_usd'
    df = df.dropna(subset=[target_col])
    X = df.drop(columns=[target_col, 'id', 'property_id', 'created_at', 'raw_data'], errors='ignore')
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = create_preprocessor(X_train)
    
    results = train_models(X_train, X_test, y_train, y_test, preprocessor, db_manager)
    if not results: return

    # Guardar el preprocesador
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    logger.info("Pipeline de preprocesamiento guardado en 'models/preprocessor.joblib'")
    
    # Guardar pipelines entrenados
    os.makedirs('models', exist_ok=True)
    for name, data in results.items():
        safe_name = name.lower().replace(' ', '_')
        joblib.dump(data['fitted_pipeline'], f'models/{safe_name}_pipeline.joblib')

    # Generar TODOS los reportes
    generate_all_reports(df, results, X_test, y_test)
    
    logger.info("=== PIPELINE FINALIZADO CON ÉXITO ===")

if __name__ == "__main__":
    main()