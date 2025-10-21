"""
Visualizaciones avanzadas para el proyecto.
Incluye dashboards interactivos y visualizaciones específicas para la presentación.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class AdvancedVisualization:
    """
    Clase para visualizaciones avanzadas del proyecto.
    """
    
    def __init__(self):
        """Inicializar visualizaciones avanzadas."""
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8'
        }
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_presentation_dashboard(self, df: pd.DataFrame, model_results: Dict, 
                                    save_path: str = None) -> go.Figure:
        """
        Crear dashboard completo para presentación.
        
        Args:
            df: DataFrame con datos
            model_results: Resultados de modelos
            save_path: Ruta para guardar
            
        Returns:
            go.Figure: Dashboard interactivo
        """
        # Crear subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'Distribución de Precios', 'Precio por Ubicación', 'Precio por Tipo',
                'Superficie vs Precio', 'Comparación de Modelos', 'Predicciones vs Reales',
                'Feature Importance', 'Residuals Plot', 'Model Performance'
            ],
            specs=[
                [{"type": "histogram"}, {"type": "bar"}, {"type": "box"}],
                [{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "scatter"}, {"type": "bar"}]
            ]
        )
        
        # 1. Distribución de precios
        fig.add_trace(
            go.Histogram(x=df['price_usd'], name='Precios', nbinsx=50),
            row=1, col=1
        )
        
        # 2. Precio por ubicación (top 10)
        if 'location' in df.columns:
            price_by_location = df.groupby('location')['price_usd'].mean().sort_values(ascending=False).head(10)
            fig.add_trace(
                go.Bar(x=price_by_location.index, y=price_by_location.values, name='Precio por Ubicación'),
                row=1, col=2
            )
        
        # 3. Box plot por tipo de propiedad
        if 'property_type' in df.columns:
            for prop_type in df['property_type'].unique()[:3]:
                data = df[df['property_type'] == prop_type]['price_usd']
                fig.add_trace(
                    go.Box(y=data, name=prop_type),
                    row=1, col=3
                )
        
        # 4. Scatter plot superficie vs precio
        if 'surface_total' in df.columns:
            sample_df = df.sample(n=min(2000, len(df)))
            fig.add_trace(
                go.Scatter(
                    x=sample_df['surface_total'], 
                    y=sample_df['price_usd'],
                    mode='markers',
                    name='Superficie vs Precio',
                    opacity=0.6
                ),
                row=2, col=1
            )
        
        # 5. Comparación de modelos
        model_names = list(model_results.keys())
        r2_scores = [results['test_r2'] for results in model_results.values()]
        fig.add_trace(
            go.Bar(x=model_names, y=r2_scores, name='R² Score'),
            row=2, col=2
        )
        
        # 6. Predicciones vs reales (mejor modelo)
        best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['test_r2'])
        best_results = model_results[best_model_name]
        if 'predictions' in best_results and 'test' in best_results['predictions']:
            y_test = best_results.get('y_test', [])
            y_pred = best_results['predictions']['test']
            if len(y_test) > 0 and len(y_pred) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=y_test, y=y_pred,
                        mode='markers',
                        name=f'Predicciones - {best_model_name}',
                        opacity=0.6
                    ),
                    row=2, col=3
                )
        
        # 7. Feature importance (si está disponible)
        if hasattr(best_results.get('model'), 'feature_importances_'):
            importance = best_results['model'].feature_importances_
            feature_names = [f'Feature {i}' for i in range(len(importance))]
            fig.add_trace(
                go.Bar(x=feature_names[:10], y=importance[:10], name='Feature Importance'),
                row=3, col=1
            )
        
        # 8. Residuals plot
        if 'predictions' in best_results and 'test' in best_results['predictions']:
            y_test = best_results.get('y_test', [])
            y_pred = best_results['predictions']['test']
            if len(y_test) > 0 and len(y_pred) > 0:
                residuals = y_test - y_pred
                fig.add_trace(
                    go.Scatter(
                        x=y_pred, y=residuals,
                        mode='markers',
                        name='Residuals',
                        opacity=0.6
                    ),
                    row=3, col=2
                )
        
        # 9. Métricas de modelos
        rmse_scores = [results['test_rmse'] for results in model_results.values()]
        mae_scores = [results['test_mae'] for results in model_results.values()]
        
        fig.add_trace(
            go.Bar(x=model_names, y=rmse_scores, name='RMSE', yaxis='y9'),
            row=3, col=3
        )
        
        # Actualizar layout
        fig.update_layout(
            height=1200,
            showlegend=True,
            title_text="Dashboard Completo - Análisis de Precios de Propiedades",
            title_x=0.5
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Dashboard guardado en: {save_path}")
        
        return fig
    
    def create_model_comparison_chart(self, model_results: Dict, save_path: str = None) -> go.Figure:
        """
        Crear gráfico de comparación de modelos.
        
        Args:
            model_results: Resultados de modelos
            save_path: Ruta para guardar
            
        Returns:
            go.Figure: Gráfico de comparación
        """
        model_names = list(model_results.keys())
        metrics = ['test_r2', 'test_rmse', 'test_mae']
        
        fig = go.Figure()
        
        for metric in metrics:
            values = [results[metric] for results in model_results.values()]
            fig.add_trace(go.Bar(
                name=metric,
                x=model_names,
                y=values
            ))
        
        fig.update_layout(
            title='Comparación de Modelos',
            xaxis_title='Modelos',
            yaxis_title='Valor de Métrica',
            barmode='group'
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Gráfico de comparación guardado en: {save_path}")
        
        return fig
    
    def create_geographic_analysis(self, df: pd.DataFrame, save_path: str = None) -> go.Figure:
        """
        Crear análisis geográfico interactivo.
        
        Args:
            df: DataFrame con datos
            save_path: Ruta para guardar
            
        Returns:
            go.Figure: Análisis geográfico
        """
        if 'location' not in df.columns:
            logger.warning("No hay columna 'location' para análisis geográfico")
            return go.Figure()
        
        # Análisis por ubicación
        location_stats = df.groupby('location').agg({
            'price_usd': ['count', 'mean', 'median', 'std'],
            'surface_total': 'mean'
        }).round(2)
        
        location_stats.columns = ['count', 'price_mean', 'price_median', 'price_std', 'surface_mean']
        location_stats = location_stats[location_stats['count'] >= 10].sort_values('price_mean', ascending=False)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Precio Promedio por Ubicación',
                'Cantidad de Propiedades por Ubicación', 
                'Precio vs Superficie por Ubicación',
                'Distribución de Precios por Ubicación'
            ]
        )
        
        # Precio promedio
        fig.add_trace(
            go.Bar(x=location_stats.index, y=location_stats['price_mean'], name='Precio Promedio'),
            row=1, col=1
        )
        
        # Cantidad de propiedades
        fig.add_trace(
            go.Bar(x=location_stats.index, y=location_stats['count'], name='Cantidad'),
            row=1, col=2
        )
        
        # Scatter precio vs superficie
        fig.add_trace(
            go.Scatter(
                x=location_stats['surface_mean'],
                y=location_stats['price_mean'],
                mode='markers+text',
                text=location_stats.index,
                name='Precio vs Superficie',
                marker=dict(size=location_stats['count']/10)
            ),
            row=2, col=1
        )
        
        # Box plot por ubicación (top 5)
        top_locations = location_stats.head(5).index
        for location in top_locations:
            data = df[df['location'] == location]['price_usd']
            fig.add_trace(
                go.Box(y=data, name=location),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Análisis Geográfico de Propiedades")
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Análisis geográfico guardado en: {save_path}")
        
        return fig
    
    def create_feature_analysis(self, df: pd.DataFrame, save_path: str = None) -> go.Figure:
        """
        Crear análisis de características.
        
        Args:
            df: DataFrame con datos
            save_path: Ruta para guardar
            
        Returns:
            go.Figure: Análisis de características
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'price_usd' in numeric_cols:
            numeric_cols.remove('price_usd')
        
        if len(numeric_cols) == 0:
            logger.warning("No hay columnas numéricas para análisis")
            return go.Figure()
        
        # Matriz de correlación
        corr_matrix = df[numeric_cols + ['price_usd']].corr()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Matriz de Correlación',
                'Distribución de Características',
                'Precio vs Características',
                'Análisis de Outliers'
            ]
        )
        
        # Matriz de correlación
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0
            ),
            row=1, col=1
        )
        
        # Distribución de características
        for i, col in enumerate(numeric_cols[:3]):  # Top 3 características
            fig.add_trace(
                go.Histogram(x=df[col], name=col, opacity=0.7),
                row=1, col=2
            )
        
        # Scatter plots precio vs características
        for i, col in enumerate(numeric_cols[:2]):  # Top 2 características
            sample_df = df.sample(n=min(1000, len(df)))
            fig.add_trace(
                go.Scatter(
                    x=sample_df[col],
                    y=sample_df['price_usd'],
                    mode='markers',
                    name=f'Precio vs {col}',
                    opacity=0.6
                ),
                row=2, col=1
            )
        
        # Box plots para outliers
        for col in numeric_cols[:3]:
            fig.add_trace(
                go.Box(y=df[col], name=col),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Análisis de Características")
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Análisis de características guardado en: {save_path}")
        
        return fig
