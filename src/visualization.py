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
