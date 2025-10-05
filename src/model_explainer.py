"""
Explicador de modelos para presentación.
Incluye explicaciones técnicas y justificaciones de decisiones.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class ModelExplainer:
    """
    Clase para explicar modelos y decisiones técnicas.
    """
    
    def __init__(self):
        """Inicializar explicador de modelos."""
        self.explanations = {}
    
    def explain_preprocessing_decisions(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Explicar decisiones de preprocesamiento.
        
        Args:
            df: DataFrame original
            
        Returns:
            Dict: Explicaciones de decisiones
        """
        explanations = {}
        
        # Análisis de valores faltantes
        missing_data = df.isnull().sum()
        total_missing = missing_data.sum()
        missing_percent = (total_missing / (df.shape[0] * df.shape[1])) * 100
        
        explanations['missing_values'] = f"""
        **Manejo de Valores Faltantes:**
        - Total de valores faltantes: {total_missing:,} ({missing_percent:.2f}%)
        - Estrategia: Eliminación de filas con valores faltantes en columnas críticas
        - Justificación: Mantener integridad de datos para modelado
        """
        
        # Análisis de outliers
        if 'price_usd' in df.columns:
            Q1 = df['price_usd'].quantile(0.25)
            Q3 = df['price_usd'].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df['price_usd'] < Q1 - 1.5*IQR) | (df['price_usd'] > Q3 + 1.5*IQR)]
            
            explanations['outliers'] = f"""
            **Manejo de Outliers:**
            - Outliers detectados: {len(outliers):,} ({len(outliers)/len(df)*100:.2f}%)
            - Método: IQR (Interquartile Range)
            - Rango normal: ${Q1-1.5*IQR:,.0f} - ${Q3+1.5*IQR:,.0f}
            - Justificación: Outliers pueden sesgar el modelo
            """
        
        # Análisis de características
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        explanations['feature_analysis'] = f"""
        **Análisis de Características:**
        - Características numéricas: {len(numeric_cols)}
        - Características categóricas: {len(categorical_cols)}
        - Estrategia: Normalización para numéricas, One-Hot Encoding para categóricas
        - Justificación: Algoritmos requieren datos numéricos normalizados
        """
        
        return explanations
    
    def explain_model_selection(self, model_results: Dict) -> Dict[str, str]:
        """
        Explicar selección de modelos.
        
        Args:
            model_results: Resultados de modelos
            
        Returns:
            Dict: Explicaciones de selección
        """
        explanations = {}
        
        # Mejor modelo
        best_model = max(model_results.keys(), key=lambda x: model_results[x]['test_r2'])
        best_r2 = model_results[best_model]['test_r2']
        
        explanations['model_selection'] = f"""
        **Selección del Mejor Modelo:**
        - Modelo seleccionado: {best_model}
        - R² Score: {best_r2:.4f}
        - Criterio: Mayor R² en conjunto de prueba
        - Justificación: R² indica qué porcentaje de la varianza es explicada
        """
        
        # Comparación de algoritmos
        algorithm_types = {
            'Linear Regression': 'Lineal',
            'Ridge Regression': 'Lineal con regularización L2',
            'Lasso Regression': 'Lineal con regularización L1',
            'Random Forest': 'Ensemble de árboles',
            'Gradient Boosting': 'Boosting de gradientes',
            'XGBoost': 'Gradient boosting optimizado',
            'LightGBM': 'Gradient boosting ligero',
            'SVR': 'Support Vector Regression'
        }
        
        explanations['algorithm_comparison'] = f"""
        **Comparación de Algoritmos:**
        """
        
        for model_name, results in model_results.items():
            algorithm_type = algorithm_types.get(model_name, 'Desconocido')
            r2 = results['test_r2']
            rmse = results['test_rmse']
            
            explanations['algorithm_comparison'] += f"""
        - {model_name} ({algorithm_type}): R²={r2:.4f}, RMSE={rmse:.2f}
        """
        
        return explanations
    
    def explain_metrics(self, model_results: Dict) -> Dict[str, str]:
        """
        Explicar métricas de evaluación.
        
        Args:
            model_results: Resultados de modelos
            
        Returns:
            Dict: Explicaciones de métricas
        """
        explanations = {}
        
        explanations['metrics_explanation'] = """
        **Métricas de Evaluación:**
        
        **1. R² (Coeficiente de Determinación):**
        - Rango: 0 a 1 (mejor = 1)
        - Interpretación: Porcentaje de varianza explicada
        - Fórmula: R² = 1 - (SS_res / SS_tot)
        
        **2. RMSE (Root Mean Square Error):**
        - Rango: 0 a ∞ (mejor = 0)
        - Interpretación: Error promedio en unidades de la variable objetivo
        - Fórmula: RMSE = √(Σ(y_true - y_pred)² / n)
        
        **3. MAE (Mean Absolute Error):**
        - Rango: 0 a ∞ (mejor = 0)
        - Interpretación: Error absoluto promedio
        - Fórmula: MAE = Σ|y_true - y_pred| / n
        
        **4. Validación Cruzada:**
        - Método: 5-fold cross-validation
        - Propósito: Evaluar estabilidad del modelo
        - Interpretación: Menor desviación estándar = modelo más estable
        """
        
        # Análisis de métricas por modelo
        best_model = max(model_results.keys(), key=lambda x: model_results[x]['test_r2'])
        best_results = model_results[best_model]
        
        explanations['best_model_metrics'] = f"""
        **Métricas del Mejor Modelo ({best_model}):**
        - R²: {best_results['test_r2']:.4f} ({best_results['test_r2']*100:.1f}% de varianza explicada)
        - RMSE: ${best_results['test_rmse']:,.0f} (error promedio en USD)
        - MAE: ${best_results['test_mae']:,.0f} (error absoluto promedio en USD)
        - CV R²: {best_results['cv_mean']:.4f} ± {best_results['cv_std']:.4f}
        """
        
        return explanations
    
    def explain_business_impact(self, model_results: Dict, df: pd.DataFrame) -> Dict[str, str]:
        """
        Explicar impacto en el negocio.
        
        Args:
            model_results: Resultados de modelos
            df: DataFrame con datos
            
        Returns:
            Dict: Explicaciones de impacto
        """
        explanations = {}
        
        best_model = max(model_results.keys(), key=lambda x: model_results[x]['test_r2'])
        best_results = model_results[best_model]
        
        # Análisis de precisión
        r2 = best_results['test_r2']
        rmse = best_results['test_rmse']
        
        if r2 > 0.8:
            precision_level = "Excelente"
        elif r2 > 0.6:
            precision_level = "Buena"
        elif r2 > 0.4:
            precision_level = "Moderada"
        else:
            precision_level = "Baja"
        
        explanations['business_impact'] = f"""
        **Impacto en el Negocio:**
        
        **1. Precisión del Modelo:**
        - Nivel de precisión: {precision_level}
        - R²: {r2:.4f} ({r2*100:.1f}% de varianza explicada)
        - Error promedio: ±${rmse:,.0f}
        
        **2. Aplicaciones Prácticas:**
        - Valuación automática de propiedades
        - Análisis de mercado inmobiliario
        - Soporte para decisiones de inversión
        - Estimación de precios para nuevos listados
        
        **3. Beneficios:**
        - Reducción de tiempo en valuaciones
        - Mayor consistencia en precios
        - Análisis de tendencias del mercado
        - Soporte para estrategias de pricing
        """
        
        # Análisis de características importantes
        if hasattr(best_results.get('model'), 'feature_importances_'):
            importance = best_results['model'].feature_importances_
            feature_names = [f'Feature {i}' for i in range(len(importance))]
            
            # Top 3 características más importantes
            top_features = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)[:3]
            
            explanations['feature_importance'] = f"""
            **Características Más Importantes:**
            """
            
            for i, (feature, importance) in enumerate(top_features, 1):
                explanations['feature_importance'] += f"""
            {i}. {feature}: {importance:.4f} ({importance*100:.1f}% de importancia)
            """
        
        return explanations
    
    def generate_presentation_summary(self, df: pd.DataFrame, model_results: Dict) -> str:
        """
        Generar resumen completo para presentación.
        
        Args:
            df: DataFrame con datos
            model_results: Resultados de modelos
            
        Returns:
            str: Resumen completo
        """
        # Obtener todas las explicaciones
        preprocessing_explanations = self.explain_preprocessing_decisions(df)
        model_explanations = self.explain_model_selection(model_results)
        metrics_explanations = self.explain_metrics(model_results)
        business_explanations = self.explain_business_impact(model_results, df)
        
        # Combinar explicaciones
        summary = f"""
# Resumen Técnico del Proyecto

## 1. Procesamiento de Datos
{preprocessing_explanations.get('missing_values', '')}
{preprocessing_explanations.get('outliers', '')}
{preprocessing_explanations.get('feature_analysis', '')}

## 2. Selección de Modelos
{model_explanations.get('model_selection', '')}
{model_explanations.get('algorithm_comparison', '')}

## 3. Métricas de Evaluación
{metrics_explanations.get('metrics_explanation', '')}
{metrics_explanations.get('best_model_metrics', '')}

## 4. Impacto en el Negocio
{business_explanations.get('business_impact', '')}
{business_explanations.get('feature_importance', '')}

## 5. Conclusiones y Recomendaciones

### Fortalezas del Modelo:
- Precisión {business_explanations.get('business_impact', '').split('Nivel de precisión: ')[1].split('\n')[0] if 'Nivel de precisión:' in business_explanations.get('business_impact', '') else 'Buena'}
- Métricas consistentes en validación cruzada
- Interpretabilidad de características importantes

### Limitaciones:
- Dependencia de calidad de datos de entrada
- Limitaciones geográficas del dataset
- Cambios en condiciones del mercado

### Recomendaciones:
1. Actualización regular del modelo con nuevos datos
2. Monitoreo continuo de performance
3. Integración con fuentes de datos adicionales
4. Validación con expertos del sector inmobiliario
        """
        
        return summary
