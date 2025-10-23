"""
Módulo para el procesamiento y preprocesamiento de datos del dataset de Properati.

"""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.database import DatabaseManager
import config

# logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Clase para el procesamiento de datos del dataset de Properati.
    """
    
    def __init__(self):
        try:
            self.db_manager = DatabaseManager()
            logger.info("DatabaseManager inicializado.")
        except Exception as e:
            logger.error(f"Error al inicializar DatabaseManager: {e}")
            raise
        
    def load_data(self, file_path):
        """
        Carga los datos desde un archivo CSV.
        """
        try:
            # Añadir 'id' para usar como 'property_id' si no existe
            df = pd.read_csv(file_path)
            if 'id' not in df.columns:
                 # Usamos el índice como ID único si no hay columna 'id'
                 df['property_id'] = df.index.astype(str)
            else:
                 df['property_id'] = df['id'].astype(str)

            logger.info(f"Datos cargados exitosamente. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error al cargar datos: {e}")
            raise
    
    def clean_data(self, df):
        """
        Limpia los datos eliminando outliers y valores atípicos.
        """
        df_clean = df.copy()

        if 'operation_type' in df_clean.columns:
            valid_ops = ['Venta', 'Alquiler']
            df_clean = df_clean[df_clean['operation_type'].isin(valid_ops)]
            logger.info(f"Filtrado por 'Venta' y 'Alquiler'. Shape: {df_clean.shape}")
        else:
            logger.warning("No se encontró la columna 'operation_type'.")
        
        # Renombrar 'price' a 'price_usd' si existe (para coincidir con la DB)
        if 'price' in df_clean.columns and 'price_usd' not in df_clean.columns:
            df_clean.rename(columns={'price': 'price_usd'}, inplace=True)

        # Eliminar duplicados
        initial_shape = df_clean.shape[0]
        df_clean = df_clean.drop_duplicates(subset=['property_id'])
        logger.info(f"Eliminados {initial_shape - df_clean.shape[0]} duplicados")
        
        # Eliminar filas con valores nulos en columnas críticas
        critical_columns = ['price_usd', 'surface_total']
        df_clean = df_clean.dropna(subset=critical_columns)
        
        # Eliminar outliers en precio (usando IQR)
        Q1 = df_clean['price_usd'].quantile(0.25)
        Q3 = df_clean['price_usd'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean['price_usd'] >= lower_bound) & 
                           (df_clean['price_usd'] <= upper_bound)]
        
        # Eliminar outliers en superficie
        df_clean = df_clean[df_clean['surface_total'] > 0]
        df_clean = df_clean[df_clean['surface_total'] < 2000]
        
        # Limpiar columnas de texto
        text_columns = ['property_type', 'location', 'state_name', 'operation_type']
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.strip().str.title()
        
        logger.info(f"Datos limpiados. Shape original: {df.shape}, Shape final: {df_clean.shape}")
        return df_clean
    
    def create_features(self, df):
        """
        Crea nuevas características (feature engineering).
        """
        df_features = df.copy()
        
        # Crear ratio superficie cubierta/total
        if 'surface_covered' in df_features.columns and 'surface_total' in df_features.columns:
            df_features['surface_ratio'] = df_features['surface_covered'] / df_features['surface_total']
            df_features['surface_ratio'] = df_features['surface_ratio'].fillna(1)
            df_features['surface_ratio'] = df_features['surface_ratio'].replace([np.inf, -np.inf], 1)
        
        
        # Crear total de habitaciones
        if 'bedrooms' in df_features.columns and 'bathrooms' in df_features.columns:
            df_features['total_rooms'] = df_features['bedrooms'] + df_features['bathrooms']
        
        # Crear densidad de habitaciones
        if 'rooms' in df_features.columns and 'surface_total' in df_features.columns:
            df_features['room_density'] = df_features['rooms'] / df_features['surface_total']
            df_features['room_density'] = df_features['room_density'].replace([np.inf, -np.inf], np.nan)
        
        # Extraer información de ubicación
        if 'location' in df_features.columns:
            df_features['city'] = df_features['location'].str.split(',').str[0].str.strip()
            df_features['province'] = df_features['location'].str.split(',').str[-1].str.strip()
        
        
        # Crear categorías de superficie
        if 'surface_total' in df_features.columns:
            df_features['surface_category'] = pd.cut(
                df_features['surface_total'],
                bins=5,
                labels=['Muy Pequeña', 'Pequeña', 'Mediana', 'Grande', 'Muy Grande']
            )
        
        logger.info("Feature engineering (sin fugas) completado")
        return df_features
    
    def run_pipeline(self, file_path):
        """
        Pipeline completo: Cargar, Limpiar, Crear Features y Guardar en DB.
        """
        try:
            # 1. Cargar datos
            df_raw = self.load_data(file_path)
            
            # 2. Limpiar datos
            df_clean = self.clean_data(df_raw)
            
            # 3. Crear características
            df_final = self.create_features(df_clean)
            
            # 4. Guardar en Base de Datos
            logger.info(f"Guardando {df_final.shape[0]} registros procesados en 'input_data'...")
            # Usamos el método de DatabaseManager que itera y mapea las columnas
            self.db_manager.store_input_data(df_final)
            logger.info("Datos guardados en la base de datos exitosamente.")
            
            return df_final
            
        except Exception as e:
            logger.error(f"El pipeline de procesamiento de datos falló: {e}")
            return None


if __name__ == "__main__":
    logger.info("Iniciando pipeline de procesamiento de datos...")
    
    # Ruta al archivo de datos crudos (desde config)
    # Asume que 'entrenamiento.csv' es el archivo descargado
    raw_data_file = config.RAW_DATA_DIR / "entrenamiento.csv" 
    
    if not raw_data_file.exists():
        logger.error(f"No se encontró el archivo de datos en: {raw_data_file}")
        logger.error("Por favor, ejecuta 'python scripts/download_data.py' primero.")
    else:
        processor = DataProcessor()
        processor.run_pipeline(raw_data_file)
        logger.info("Pipeline de procesamiento de datos finalizado.")