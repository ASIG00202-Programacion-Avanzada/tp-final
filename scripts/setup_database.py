#!/usr/bin/env python3
"""
Script para configurar la base de datos del proyecto.
"""
import sys
from pathlib import Path
import logging

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from database import DatabaseManager
from config import DB_CONFIG

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_database():
    """
    Configura la base de datos y crea las tablas necesarias.
    """
    try:
        logger.info("Configurando base de datos...")
        logger.info(f"Tipo de BD: {DB_CONFIG['type']}")
        logger.info(f"Host: {DB_CONFIG['host']}")
        logger.info(f"Puerto: {DB_CONFIG['port']}")
        logger.info(f"Base de datos: {DB_CONFIG['name']}")
        
        # Inicializar manager de base de datos
        db_manager = DatabaseManager()
        
        logger.info("Base de datos configurada exitosamente!")
        logger.info("Tablas creadas:")
        logger.info("- input_data: Datos de entrada preprocesados")
        logger.info("- model_results: Resultados de modelos")
        logger.info("- model_config: Configuraciones de modelos")
        
        return True
        
    except Exception as e:
        logger.error(f"Error configurando base de datos: {e}")
        logger.error("Por favor, verifica la configuración en config.py")
        return False

def test_database_connection():
    """
    Prueba la conexión a la base de datos.
    """
    try:
        db_manager = DatabaseManager()
        
        # Probar conexión con una consulta simple
        session = db_manager.Session()
        session.execute("SELECT 1")
        session.close()
        
        logger.info("Conexión a base de datos exitosa!")
        return True
        
    except Exception as e:
        logger.error(f"Error conectando a la base de datos: {e}")
        return False

def main():
    """
    Función principal para configurar la base de datos.
    """
    logger.info("=== CONFIGURANDO BASE DE DATOS ===")
    
    # Configurar base de datos
    if not setup_database():
        logger.error("Error configurando la base de datos")
        return 1
    
    # Probar conexión
    if not test_database_connection():
        logger.error("Error conectando a la base de datos")
        return 1
    
    logger.info("Base de datos lista para usar!")
    return 0

if __name__ == "__main__":
    exit(main())
