#!/usr/bin/env python3
"""
Script para descargar el dataset de Properati Argentina desde Kaggle.
"""
import os
import sys
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_kaggle():
    """
    Configura la API de Kaggle.
    """
    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        # Verificar si existe el archivo de credenciales
        kaggle_dir = Path.home() / '.kaggle'
        kaggle_json = kaggle_dir / 'kaggle.json'
        
        if not kaggle_json.exists():
            logger.warning("No se encontró kaggle.json en ~/.kaggle/")
            logger.info("Por favor, descarga tu archivo kaggle.json desde:")
            logger.info("https://www.kaggle.com/account")
            logger.info("Y colócalo en ~/.kaggle/kaggle.json")
            return False
        
        # Configurar permisos
        os.chmod(kaggle_json, 0o600)
        
        api = KaggleApi()
        api.authenticate()
        
        logger.info("API de Kaggle configurada exitosamente")
        return True
        
    except Exception as e:
        logger.error(f"Error configurando Kaggle API: {e}")
        return False

def download_dataset(dataset_name="alejandroczernikier/properati-argentina-dataset", 
                    output_dir="data/raw"):
    """
    Descarga el dataset desde Kaggle.
    
    Args:
        dataset_name: Nombre del dataset en Kaggle
        output_dir: Directorio donde guardar los datos
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        # Crear directorio de salida
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Configurar API
        api = KaggleApi()
        api.authenticate()
        
        logger.info(f"Descargando dataset: {dataset_name}")
        logger.info(f"Directorio de destino: {output_path.absolute()}")
        
        # Descargar dataset
        api.dataset_download_files(
            dataset_name,
            path=str(output_path),
            unzip=True
        )
        
        # Verificar archivos descargados
        downloaded_files = list(output_path.glob("*"))
        logger.info(f"Archivos descargados: {[f.name for f in downloaded_files]}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error descargando dataset: {e}")
        return False

def main():
    """
    Función principal para descargar el dataset.
    """
    logger.info("=== DESCARGANDO DATASET DE PROPERATI ARGENTINA ===")
    
    # Configurar Kaggle
    if not setup_kaggle():
        logger.error("No se pudo configurar Kaggle API")
        return 1
    
    # Descargar dataset
    if download_dataset():
        logger.info("Dataset descargado exitosamente!")
        logger.info("Los archivos están en: data/raw/")
        return 0
    else:
        logger.error("Error descargando el dataset")
        return 1

if __name__ == "__main__":
    exit(main())
