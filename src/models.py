"""
Módulo para la implementación y comparación de modelos de regresión.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import logging
import joblib
import os

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Clase para entrenar y comparar diferentes modelos de regresión.
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = -np.inf
        
    def define_models(self):
        """
        Define los modelos a comparar.
        
        Returns:
            dict: Diccionario con los modelos configurados
        """
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(random_state=42),
            'Lasso Regression': Lasso(random_state=42),
            'Random Forest': RandomForestRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'XGBoost': xgb.XGBRegressor(random_state=42),
            'LightGBM': lgb.LGBMRegressor(random_state=42, verbose=-1)
        }
        
        self.models = models
        logger.info(f"Modelos definidos: {list(models.keys())}")
        return models
    
    def define_hyperparameters(self):
        """
        Define los hiperparámetros para cada modelo.
        
        Returns:
            dict: Diccionario con los hiperparámetros para cada modelo
        """
        param_grids = {
            'Ridge Regression': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'Lasso Regression': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'LightGBM': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }
        
        logger.info("Hiperparámetros definidos para todos los modelos")
        return param_grids
    
    def train_model(self, model, X_train, y_train, X_test, y_test, model_name):
        """
        Entrena un modelo y evalúa su rendimiento.
        
        Args:
            model: Modelo a entrenar
            X_train, y_train: Datos de entrenamiento
            X_test, y_test: Datos de prueba
            model_name (str): Nombre del modelo
            
        Returns:
            dict: Resultados del modelo
        """
        try:
            # Entrenar modelo
            model.fit(X_train, y_train)
            
            # Hacer predicciones
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calcular métricas
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            # Validación cruzada
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            results = {
                'model_name': model_name,
                'model': model,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': {
                    'train': y_pred_train,
                    'test': y_pred_test
                }
            }
            
            logger.info(f"Modelo {model_name} entrenado exitosamente")
            logger.info(f"Test RMSE: {test_rmse:.4f}, Test R²: {test_r2:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error entrenando modelo {model_name}: {e}")
            return None
    
    def hyperparameter_tuning(self, model, param_grid, X_train, y_train, model_name):
        """
        Realiza ajuste de hiperparámetros usando GridSearchCV.
        
        Args:
            model: Modelo base
            param_grid (dict): Grid de hiperparámetros
            X_train, y_train: Datos de entrenamiento
            model_name (str): Nombre del modelo
            
        Returns:
            dict: Mejor modelo y resultados
        """
        try:
            grid_search = GridSearchCV(
                model, 
                param_grid, 
                cv=5, 
                scoring='r2',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            logger.info(f"Mejores hiperparámetros para {model_name}: {best_params}")
            logger.info(f"Mejor score CV: {best_score:.4f}")
            
            return {
                'best_model': best_model,
                'best_params': best_params,
                'best_score': best_score
            }
            
        except Exception as e:
            logger.error(f"Error en ajuste de hiperparámetros para {model_name}: {e}")
            return None
    
    def compare_models(self, X_train, y_train, X_test, y_test):
        """
        Compara todos los modelos definidos.
        
        Args:
            X_train, y_train: Datos de entrenamiento
            X_test, y_test: Datos de prueba
            
        Returns:
            dict: Resultados de todos los modelos
        """
        # Definir modelos
        models = self.define_models()
        
        # Definir hiperparámetros
        param_grids = self.define_hyperparameters()
        
        all_results = {}
        
        for model_name, model in models.items():
            logger.info(f"Entrenando modelo: {model_name}")
            
            # Ajuste de hiperparámetros si está disponible
            if model_name in param_grids:
                tuning_result = self.hyperparameter_tuning(
                    model, param_grids[model_name], X_train, y_train, model_name
                )
                if tuning_result:
                    model = tuning_result['best_model']
            
            # Entrenar y evaluar modelo
            results = self.train_model(model, X_train, y_train, X_test, y_test, model_name)
            
            if results:
                all_results[model_name] = results
                
                # Actualizar mejor modelo
                if results['test_r2'] > self.best_score:
                    self.best_score = results['test_r2']
                    self.best_model = results['model']
        
        self.results = all_results
        logger.info(f"Comparación completada. Mejor modelo: R² = {self.best_score:.4f}")
        
        return all_results
    
    def save_model(self, model, filepath):
        """
        Guarda un modelo entrenado.
        
        Args:
            model: Modelo a guardar
            filepath (str): Ruta donde guardar el modelo
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(model, filepath)
            logger.info(f"Modelo guardado en: {filepath}")
        except Exception as e:
            logger.error(f"Error guardando modelo: {e}")
    
    def load_model(self, filepath):
        """
        Carga un modelo guardado.
        
        Args:
            filepath (str): Ruta del modelo
            
        Returns:
            Modelo cargado
        """
        try:
            model = joblib.load(filepath)
            logger.info(f"Modelo cargado desde: {filepath}")
            return model
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            return None
    
    def get_model_summary(self):
        """
        Obtiene un resumen de todos los modelos entrenados.
        
        Returns:
            pd.DataFrame: Resumen de resultados
        """
        if not self.results:
            logger.warning("No hay resultados disponibles")
            return None
        
        summary_data = []
        for model_name, results in self.results.items():
            summary_data.append({
                'Modelo': model_name,
                'Test RMSE': results['test_rmse'],
                'Test MAE': results['test_mae'],
                'Test R²': results['test_r2'],
                'CV Mean': results['cv_mean'],
                'CV Std': results['cv_std']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Test R²', ascending=False)
        
        return summary_df
