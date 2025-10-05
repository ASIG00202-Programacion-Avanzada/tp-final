#!/usr/bin/env python3
"""
Script para manejar datasets grandes de Properati.
"""
import sys
from pathlib import Path
import logging

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from large_data_handler import LargeDataHandler
from config import *

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_dataset_size(file_path: str):
    """
    Analiza el tamaño del dataset y recomienda estrategias.
    """
    handler = LargeDataHandler()
    
    logger.info("=== ANÁLISIS DEL DATASET ===")
    
    # Obtener información
    info = handler.get_dataset_info(file_path)
    
    if not info:
        logger.error("No se pudo analizar el dataset")
        return
    
    logger.info(f"Filas totales: {info['total_rows']:,}")
    logger.info(f"Columnas: {len(info['columns'])}")
    logger.info(f"Tamaño estimado: {info['estimated_size_mb']:.1f} MB")
    
    # Recomendaciones
    if info['estimated_size_mb'] > 1000:  # > 1GB
        logger.info("\n DATASET MUY GRANDE (>1GB)")
        logger.info("Recomendaciones:")
        logger.info("  • Usar muestra estratificada de 50,000-100,000 filas")
        logger.info("  • Procesar en chunks")
        logger.info("  • Considerar usar Dask para procesamiento distribuido")
        
    elif info['estimated_size_mb'] > 500:  # > 500MB
        logger.info("\n DATASET GRANDE (500MB-1GB)")
        logger.info("Recomendaciones:")
        logger.info("  • Usar muestra estratificada de 100,000-200,000 filas")
        logger.info("  • Optimizar tipos de datos")
        
    else:
        logger.info("\n DATASET NORMAL (<500MB)")
        logger.info("Recomendaciones:")
        logger.info("  • Se puede procesar completo")
        logger.info("  • Usar muestra de 50,000 filas para desarrollo rápido")

def create_optimized_sample(file_path: str, sample_size: int = 50000, 
                          strategy: str = "stratified"):
    """
    Crea una muestra optimizada del dataset.
    
    Args:
        file_path: Ruta al dataset
        sample_size: Tamaño de la muestra
        strategy: Estrategia de muestreo ('stratified', 'geographic', 'random')
    """
    handler = LargeDataHandler()
    
    logger.info(f"=== CREANDO MUESTRA ({strategy.upper()}) ===")
    logger.info(f"Tamaño objetivo: {sample_size:,} filas")
    
    try:
        if strategy == "stratified":
            sample_df = handler.create_stratified_sample(
                file_path, sample_size, target_column='price_usd'
            )
        elif strategy == "geographic":
            sample_df = handler.create_geographic_sample(
                file_path, sample_size, location_column='location'
            )
        else:  # random
            sample_df = handler.create_stratified_sample(
                file_path, sample_size, target_column='price_usd'
            )
        
        if sample_df.empty:
            logger.error("No se pudo crear la muestra")
            return None
        
        # Optimizar DataFrame
        sample_df = handler.optimize_dataframe(sample_df)
        
        # Guardar muestra
        output_path = PROCESSED_DATA_DIR / f"dataset_sample_{strategy}_{sample_size}.csv"
        sample_df.to_csv(output_path, index=False)
        
        logger.info(f"Muestra creada exitosamente")
        logger.info(f"Guardada en: {output_path}")
        logger.info(f"Filas: {len(sample_df):,}")
        logger.info(f"Memoria: {sample_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error creando muestra: {e}")
        return None

def main():
    """
    Función principal para manejar datasets grandes.
    """
    logger.info("=== MANEJADOR DE DATASETS GRANDES ===")
    
    # Buscar archivos CSV en raw
    csv_files = list(RAW_DATA_DIR.glob("*.csv"))
    
    if not csv_files:
        logger.error("No se encontraron archivos CSV en data/raw/")
        logger.info("Por favor, coloca tu dataset en: data/raw/")
        return 1
    
    file_path = csv_files[0]
    logger.info(f"Procesando: {file_path.name}")
    
    # 1. Analizar tamaño
    analyze_dataset_size(str(file_path))
    
    # 2. Crear muestra estratificada
    sample_path = create_optimized_sample(
        str(file_path), 
        sample_size=50000, 
        strategy="stratified"
    )
    
    if sample_path:
        logger.info("\n ¡Muestra creada exitosamente!")
        logger.info(f"Archivo: {sample_path}")
        logger.info("\n Próximos pasos:")
        logger.info("   1. Ejecutar: python scripts/main_pipeline.py")
        logger.info("   2. O usar la muestra directamente en tu análisis")
        return 0
    else:
        logger.error(" Error creando la muestra")
        return 1

if __name__ == "__main__":
    exit(main())
