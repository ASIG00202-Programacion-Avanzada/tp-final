"""
Módulo de visualización para el análisis de precios de propiedades.
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

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisualizationManager:
    """
    Clase para manejar todas las visualizaciones del proyecto.
    """
    
    def __init__(self, style='seaborn-v0_8', palette='husl'):
        """
        Inicializar el manager de visualizaciones.
        
        Args:
            style: Estilo de matplotlib
            palette: Paleta de colores de seaborn
        """
        plt.style.use(style)
        sns.set_palette(palette)
        self.figures = {}
        
    def plot_price_distribution(self, df, price_col='price_usd', save_path=None):
        """
        Visualiza la distribución de precios.
        
        Args:
            df: DataFrame con los datos
            price_col: Columna de precios
            save_path: Ruta para guardar la figura
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Histograma
        axes[0, 0].hist(df[price_col], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Distribución de Precios')
        axes[0, 0].set_xlabel('Precio (USD)')
        axes[0, 0].set_ylabel('Frecuencia')
        
        # Box plot
        axes[0, 1].boxplot(df[price_col])
        axes[0, 1].set_title('Box Plot de Precios')
        axes[0, 1].set_ylabel('Precio (USD)')
        
        # Log scale
        axes[1, 0].hist(np.log10(df[price_col]), bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Distribución de Precios (Log Scale)')
        axes[1, 0].set_xlabel('Log10(Precio)')
        axes[1, 0].set_ylabel('Frecuencia')
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(df[price_col], dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot de Precios')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figura guardada en: {save_path}")
        
        self.figures['price_distribution'] = fig
        return fig

def plot_correlation_matrix(self, df, numeric_cols, save_path=None):
        """
        Visualiza la matriz de correlación.
        
        Args:
            df: DataFrame con los datos
            numeric_cols: Lista de columnas numéricas
            save_path: Ruta para guardar la figura
        """
        plt.figure(figsize=(12, 8))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Matriz de Correlación')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figura guardada en: {save_path}")
        
        self.figures['correlation_matrix'] = plt.gcf()
        return plt.gcf()
    
    def plot_price_by_location(self, df, location_col, price_col='price_usd', 
                              top_n=15, save_path=None):
        """
        Visualiza precios por ubicación.
        
        Args:
            df: DataFrame con los datos
            location_col: Columna de ubicación
            price_col: Columna de precios
            top_n: Número de ubicaciones a mostrar
            save_path: Ruta para guardar la figura
        """
        # Calcular precios promedio por ubicación
        price_by_location = df.groupby(location_col)[price_col].agg(['count', 'mean']).sort_values('mean', ascending=False)
        price_by_location = price_by_location[price_by_location['count'] >= 10]  # Al menos 10 propiedades
        
        plt.figure(figsize=(12, 8))
        price_by_location.head(top_n)['mean'].plot(kind='bar')
        plt.title(f'Precio Promedio por {location_col}')
        plt.xlabel(location_col)
        plt.ylabel('Precio Promedio (USD)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figura guardada en: {save_path}")
        
        self.figures['price_by_location'] = plt.gcf()
        return plt.gcf()
    
    def plot_price_by_property_type(self, df, property_type_col='property_type', 
                                   price_col='price_usd', save_path=None):
        """
        Visualiza precios por tipo de propiedad.
        
        Args:
            df: DataFrame con los datos
            property_type_col: Columna de tipo de propiedad
            price_col: Columna de precios
            save_path: Ruta para guardar la figura
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Distribución de tipos
        property_dist = df[property_type_col].value_counts()
        property_dist.plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Distribución de Tipos de Propiedad')
        axes[0, 0].set_xlabel('Tipo de Propiedad')
        axes[0, 0].set_ylabel('Cantidad')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Precio promedio por tipo
        price_by_type = df.groupby(property_type_col)[price_col].agg(['count', 'mean']).sort_values('mean', ascending=False)
        price_by_type['mean'].plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Precio Promedio por Tipo de Propiedad')
        axes[0, 1].set_xlabel('Tipo de Propiedad')
        axes[0, 1].set_ylabel('Precio Promedio (USD)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Box plot de precios por tipo
        df.boxplot(column=price_col, by=property_type_col, ax=axes[1, 0])
        axes[1, 0].set_title('Distribución de Precios por Tipo')
        axes[1, 0].set_xlabel('Tipo de Propiedad')
        axes[1, 0].set_ylabel('Precio (USD)')
        
        # Violin plot
        df_violin = df[df[property_type_col].isin(property_dist.head(5).index)]  # Top 5 tipos
        sns.violinplot(data=df_violin, x=property_type_col, y=price_col, ax=axes[1, 1])
        axes[1, 1].set_title('Distribución de Precios (Top 5 Tipos)')
        axes[1, 1].set_xlabel('Tipo de Propiedad')
        axes[1, 1].set_ylabel('Precio (USD)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figura guardada en: {save_path}")
        
        self.figures['price_by_property_type'] = fig
        return fig
    
    def plot_surface_analysis(self, df, surface_cols, price_col='price_usd', save_path=None):
        """
        Visualiza análisis de superficie.
        
        Args:
            df: DataFrame con los datos
            surface_cols: Lista de columnas de superficie
            price_col: Columna de precios
            save_path: Ruta para guardar la figura
        """
        available_surface_cols = [col for col in surface_cols if col in df.columns]
        
        if not available_surface_cols:
            logger.warning("No se encontraron columnas de superficie")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Distribución de superficie total
        if 'surface_total' in available_surface_cols:
            axes[0, 0].hist(df['surface_total'], bins=50, alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Distribución de Superficie Total')
            axes[0, 0].set_xlabel('Superficie Total (m²)')
            axes[0, 0].set_ylabel('Frecuencia')
        
        # Scatter plot: superficie vs precio
        if 'surface_total' in available_surface_cols:
            sample_df = df.sample(n=min(5000, len(df)))  # Muestra para visualización
            axes[0, 1].scatter(sample_df['surface_total'], sample_df[price_col], alpha=0.5)
            axes[0, 1].set_title('Superficie vs Precio')
            axes[0, 1].set_xlabel('Superficie Total (m²)')
            axes[0, 1].set_ylabel('Precio (USD)')
        
        # Precio por metro cuadrado
        if 'surface_total' in available_surface_cols:
            df['price_per_sqm'] = df[price_col] / df['surface_total']
            axes[1, 0].hist(df['price_per_sqm'], bins=50, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Distribución de Precio por m²')
            axes[1, 0].set_xlabel('Precio por m² (USD)')
            axes[1, 0].set_ylabel('Frecuencia')
        
        # Box plot de superficie por tipo de propiedad
        if 'surface_total' in available_surface_cols and 'property_type' in df.columns:
            df.boxplot(column='surface_total', by='property_type', ax=axes[1, 1])
            axes[1, 1].set_title('Superficie por Tipo de Propiedad')
            axes[1, 1].set_xlabel('Tipo de Propiedad')
            axes[1, 1].set_ylabel('Superficie Total (m²)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figura guardada en: {save_path}")
        
        self.figures['surface_analysis'] = fig
        return fig
    
    def plot_model_comparison(self, model_results, save_path=None):
        """
        Visualiza la comparación de modelos.
        
        Args:
            model_results: Diccionario con resultados de modelos
            save_path: Ruta para guardar la figura
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Métricas de comparación
        model_names = list(model_results.keys())
        rmse_scores = [results['test_rmse'] for results in model_results.values()]
        r2_scores = [results['test_r2'] for results in model_results.values()]
        mae_scores = [results['test_mae'] for results in model_results.values()]
        
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
        
        # MAE por modelo
        axes[1, 0].bar(model_names, mae_scores)
        axes[1, 0].set_title('MAE por Modelo')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Comparación de métricas
        x = np.arange(len(model_names))
        width = 0.25
        
        axes[1, 1].bar(x - width, rmse_scores, width, label='RMSE', alpha=0.8)
        axes[1, 1].bar(x, r2_scores, width, label='R²', alpha=0.8)
        axes[1, 1].bar(x + width, mae_scores, width, label='MAE', alpha=0.8)
        
        axes[1, 1].set_title('Comparación de Métricas')
        axes[1, 1].set_ylabel('Valor')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(model_names, rotation=45)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figura guardada en: {save_path}")
        
        self.figures['model_comparison'] = fig
        return fig
    
    def plot_predictions_vs_actual(self, y_actual, y_pred, model_name, save_path=None):
        """
        Visualiza predicciones vs valores reales.
        
        Args:
            y_actual: Valores reales
            y_pred: Predicciones
            model_name: Nombre del modelo
            save_path: Ruta para guardar la figura
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        axes[0].scatter(y_actual, y_pred, alpha=0.6)
        axes[0].plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2)
        axes[0].set_xlabel('Valores Reales')
        axes[0].set_ylabel('Predicciones')
        axes[0].set_title(f'Predicciones vs Reales - {model_name}')
        
        # Distribución de errores
        errors = y_actual - y_pred
        axes[1].hist(errors, bins=30, alpha=0.7)
        axes[1].set_xlabel('Error de Predicción')
        axes[1].set_ylabel('Frecuencia')
        axes[1].set_title('Distribución de Errores')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figura guardada en: {save_path}")
        
        self.figures['predictions_vs_actual'] = fig
        return fig
    
    def plot_feature_importance(self, feature_importance, feature_names, model_name, 
                                top_n=15, save_path=None):
        """
        Visualiza la importancia de características.
        
        Args:
            feature_importance: Array de importancia de características
            feature_names: Lista de nombres de características
            model_name: Nombre del modelo
            top_n: Número de características a mostrar
            save_path: Ruta para guardar la figura
        """
        if feature_importance is None:
            logger.warning("No hay información de importancia de características")
            return None
        
        # Crear DataFrame para feature importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df.head(top_n), x='importance', y='feature')
        plt.title(f'Feature Importance - {model_name}')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figura guardada en: {save_path}")
        
        self.figures['feature_importance'] = plt.gcf()
        return plt.gcf()

    def create_interactive_dashboard(self, df, price_col='price_usd'):
        """
        Crea un dashboard interactivo con Plotly.
        
        Args:
            df: DataFrame con los datos
            price_col: Columna de precios
        """
        # Crear subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Distribución de Precios', 'Precio por Ubicación', 
                           'Precio por Tipo de Propiedad', 'Superficie vs Precio'),
            specs=[[{"type": "histogram"}, {"type": "bar"}],
                   [{"type": "box"}, {"type": "scatter"}]]
        )
        
        # Histograma de precios
        fig.add_trace(
            go.Histogram(x=df[price_col], name='Precios'),
            row=1, col=1
        )
        
        # Precio por ubicación (top 10)
        if 'location' in df.columns:
            price_by_location = df.groupby('location')[price_col].mean().sort_values(ascending=False).head(10)
            fig.add_trace(
                go.Bar(x=price_by_location.index, y=price_by_location.values, name='Precio por Ubicación'),
                row=1, col=2
            )
        
        # Box plot por tipo de propiedad
        if 'property_type' in df.columns:
            for prop_type in df['property_type'].unique()[:5]:  # Top 5 tipos
                data = df[df['property_type'] == prop_type][price_col]
                fig.add_trace(
                    go.Box(y=data, name=prop_type),
                    row=2, col=1
                )
        
        # Scatter plot superficie vs precio
        if 'surface_total' in df.columns:
            sample_df = df.sample(n=min(2000, len(df)))
            fig.add_trace(
                    go.Scatter(x=sample_df['surface_total'], y=sample_df[price_col], 
                             mode='markers', name='Superficie vs Precio'),
                    row=2, col=2
                )
        
        fig.update_layout(height=800, showlegend=True, title_text="Dashboard Interactivo - Properati Argentina")
        
        return fig
    
    def save_all_figures(self, output_dir):
        """
        Guarda todas las figuras generadas.
        
        Args:
            output_dir: Directorio donde guardar las figuras
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for name, fig in self.figures.items():
            file_path = output_path / f"{name}.png"
            fig.savefig(file_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figura guardada: {file_path}")
    
    def close_all_figures(self):
        """
        Cierra todas las figuras abiertas.
        """
        plt.close('all')
        logger.info("Todas las figuras cerradas")
