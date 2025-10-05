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
            'Gradient Boosting': GradientBoostingRegressor(random_state=42)
        }
        
        self.models = models
        logger.info(f"Modelos definidos: {list(models.keys())}")
        return models
    
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