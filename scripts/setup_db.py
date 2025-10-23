#!/usr/bin/env python3
"""
Script simplificado para configurar el proyecto.
"""
import sys
from pathlib import Path
import logging

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.database import DatabaseManager
from config import *

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_project():
    """
    Configura el proyecto completo.
    """
    logger.info("=== CONFIGURANDO PROYECTO ===")
    
    try:
        # 1. Crear directorios necesarios
        logger.info("1. Creando directorios...")
        for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, REPORTS_DIR]:
            directory.mkdir(exist_ok=True)
            logger.info(f"{directory}")
        
        # 2. Configurar base de datos SQLite
        logger.info("2. Configurando base de datos SQLite...")
        db_manager = DatabaseManager()
        logger.info(" Base de datos SQLite configurada")
        logger.info(f" Ubicación: {DATA_DIR / DB_CONFIG['name']}")
        
        # 3. Verificar archivos de datos
        logger.info("3. Verificando datos...")
        csv_files = list(RAW_DATA_DIR.glob("*.csv"))
        if csv_files:
            logger.info(f"Dataset encontrado: {csv_files[0].name}")
        else:
            logger.warning("No se encontraron archivos CSV en data/raw/")
            logger.info("Ejecuta python .\scripts\download_data.py")
        
        logger.info("Proyecto configurado exitosamente!")
        logger.info("\nPróximos pasos:")
        logger.info("   1. Ejecuta: python .\scripts\download_data.py")
        logger.info("   2. Ejecuta: python scripts/main_pipeline.py")
        logger.info("   3. O ejecuta: python run_example.py (para probar)")
        
        return True
        
    except Exception as e:
        logger.error(f"Error configurando proyecto: {e}")
        return False

def main():
    """
    Función principal.
    """
    if setup_project():
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit(main())
