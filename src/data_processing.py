"""
Módulo para el procesamiento y preprocesamiento de datos del dataset de Properati.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Clase para el procesamiento de datos del dataset de Properati.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.preprocessor = None
        
    def load_data(self, file_path):
        """
        Carga los datos desde un archivo CSV.
        
        Args:
            file_path (str): Ruta al archivo CSV
            
        Returns:
            pd.DataFrame: Dataset cargado
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Datos cargados exitosamente. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error al cargar datos: {e}")
            raise
