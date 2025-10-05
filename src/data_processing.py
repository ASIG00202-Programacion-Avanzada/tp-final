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
    
    def explore_data(self, df):
        """
        Realiza exploración básica de los datos.
        
        Args:
            df (pd.DataFrame): Dataset a explorar
            
        Returns:
            dict: Información básica del dataset
        """
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
        }
        
        logger.info(f"Exploración de datos completada. Shape: {info['shape']}")
        return info
    
    def clean_data(self, df):
        """
        Limpia los datos eliminando outliers y valores atípicos.
        
        Args:
            df (pd.DataFrame): Dataset a limpiar
            
        Returns:
            pd.DataFrame: Dataset limpio
        """
        df_clean = df.copy()
        
        # Eliminar duplicados
        initial_shape = df_clean.shape[0]
        df_clean = df_clean.drop_duplicates()
        logger.info(f"Eliminados {initial_shape - df_clean.shape[0]} duplicados")
        
        # Eliminar filas con valores nulos en columnas críticas
        critical_columns = ['price_usd', 'surface_total']
        available_critical = [col for col in critical_columns if col in df_clean.columns]
        if available_critical:
            df_clean = df_clean.dropna(subset=available_critical)
        
        # Eliminar outliers en precio (usando IQR)
        if 'price_usd' in df_clean.columns:
            Q1 = df_clean['price_usd'].quantile(0.25)
            Q3 = df_clean['price_usd'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean['price_usd'] >= lower_bound) & 
                               (df_clean['price_usd'] <= upper_bound)]
        
        # Eliminar outliers en superficie
        if 'surface_total' in df_clean.columns:
            df_clean = df_clean[df_clean['surface_total'] > 0]
            df_clean = df_clean[df_clean['surface_total'] < 2000]  # Eliminar propiedades muy grandes
        
        # Limpiar columnas de texto
        text_columns = ['property_type', 'location', 'state_name']
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.strip().str.title()
        
        logger.info(f"Datos limpiados. Shape original: {df.shape}, Shape final: {df_clean.shape}")
        return df_clean
    
    def create_features(self, df):
        """
        Crea nuevas características (feature engineering).
        
        Args:
            df (pd.DataFrame): Dataset original
            
        Returns:
            pd.DataFrame: Dataset con nuevas características
        """
        df_features = df.copy()
        
        # Crear ratio superficie cubierta/total
        if 'surface_covered' in df_features.columns and 'surface_total' in df_features.columns:
            df_features['surface_ratio'] = df_features['surface_covered'] / df_features['surface_total']
            df_features['surface_ratio'] = df_features['surface_ratio'].fillna(1)
            df_features['surface_ratio'] = df_features['surface_ratio'].replace([np.inf, -np.inf], 1)
        
        # Crear precio por metro cuadrado
        if 'price_usd' in df_features.columns and 'surface_total' in df_features.columns:
            df_features['price_per_sqm'] = df_features['price_usd'] / df_features['surface_total']
            df_features['price_per_sqm'] = df_features['price_per_sqm'].replace([np.inf, -np.inf], np.nan)
        
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
        
        # Crear categorías de precio
        if 'price_usd' in df_features.columns:
            df_features['price_category'] = pd.cut(
                df_features['price_usd'], 
                bins=5, 
                labels=['Muy Bajo', 'Bajo', 'Medio', 'Alto', 'Muy Alto']
            )
        
        # Crear categorías de superficie
        if 'surface_total' in df_features.columns:
            df_features['surface_category'] = pd.cut(
                df_features['surface_total'],
                bins=5,
                labels=['Muy Pequeña', 'Pequeña', 'Mediana', 'Grande', 'Muy Grande']
            )
        
        logger.info("Feature engineering completado")
        return df_features
    
    def prepare_preprocessor(self, df, target_column='price_usd'):
        """
        Prepara el preprocesador para las características.
        
        Args:
            df (pd.DataFrame): Dataset
            target_column (str): Nombre de la columna objetivo
            
        Returns:
            ColumnTransformer: Preprocesador configurado
        """
        from sklearn.preprocessing import OneHotEncoder
        
        # Separar características numéricas y categóricas
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_features:
            numeric_features.remove(target_column)
        
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        
        # Pipeline para características numéricas
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Pipeline para características categóricas
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combinar preprocesadores
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        self.preprocessor = preprocessor
        logger.info("Preprocesador configurado")
        return preprocessor
    
    def split_data(self, df, target_column='price_usd', test_size=0.2, random_state=42):
        """
        Divide los datos en conjuntos de entrenamiento y prueba.
        
        Args:
            df (pd.DataFrame): Dataset
            target_column (str): Columna objetivo
            test_size (float): Proporción del conjunto de prueba
            random_state (int): Semilla para reproducibilidad
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Datos divididos. Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def process_pipeline(self, df, target_column='price_usd'):
        """
        Pipeline completo de procesamiento de datos.
        
        Args:
            df (pd.DataFrame): Dataset original
            target_column (str): Columna objetivo
            
        Returns:
            tuple: X_train, X_test, y_train, y_test, preprocessor
        """
        # Limpiar datos
        df_clean = self.clean_data(df)
        
        # Crear características
        df_features = self.create_features(df_clean)
        
        # Preparar preprocesador
        preprocessor = self.prepare_preprocessor(df_features, target_column)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = self.split_data(df_features, target_column)
        
        return X_train, X_test, y_train, y_test, preprocessor
    
    def split_data(self, df, target_column='price_usd', test_size=0.2, random_state=42):
        """
        Divide los datos en conjuntos de entrenamiento y prueba.
        
        Args:
            df (pd.DataFrame): Dataset
            target_column (str): Columna objetivo
            test_size (float): Proporción del conjunto de prueba
            random_state (int): Semilla para reproducibilidad
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Datos divididos. Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def process_pipeline(self, df, target_column='price_usd'):
        """
        Pipeline completo de procesamiento de datos.
        
        Args:
            df (pd.DataFrame): Dataset original
            target_column (str): Columna objetivo
            
        Returns:
            tuple: X_train, X_test, y_train, y_test, preprocessor
        """
        # Limpiar datos
        df_clean = self.clean_data(df)
        
        # Crear características
        df_features = self.create_features(df_clean)
        
        # Preparar preprocesador
        preprocessor = self.prepare_preprocessor(df_features, target_column)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = self.split_data(df_features, target_column)
        
        return X_train, X_test, y_train, y_test, preprocessor
