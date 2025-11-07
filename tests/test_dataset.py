#!/usr/bin/env python3
"""
Script simple para probar el dataset de Properati.
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Agregar src al path
sys.path.append(str(Path(__file__).parent / "src"))

def test_dataset():
    """
    Probar el dataset de Properati.
    """
    print("=== PROBANDO DATASET DE PROPERATI ===")
    
    # Verificar que existe el archivo
    data_path = Path("data/raw/entrenamiento.csv")
    if not data_path.exists():
        print("No se encontró el dataset en data/raw/entrenamiento.csv")
        return False
    
    print(f"Dataset encontrado: {data_path}")
    print(f"Tamaño del archivo: {data_path.stat().st_size / (1024*1024):.1f} MB")
    
    try:
        # Cargar una muestra del dataset
        print("\n Cargando muestra del dataset...")
        df = pd.read_csv(data_path, nrows=1000)
        
        print(f"Dataset cargado exitosamente")
        print(f" Shape de la muestra: {df.shape}")
        print(f" Columnas: {list(df.columns)}")
        
        # Información básica
        print(f"\n Información del dataset:")
        print(f"   • Filas en muestra: {len(df):,}")
        print(f"   • Columnas: {len(df.columns)}")
        print(f"   • Tipos de datos:")
        for col, dtype in df.dtypes.items():
            print(f"     - {col}: {dtype}")
        
        # Valores faltantes
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"\n  Valores faltantes:")
            for col, count in missing[missing > 0].items():
                print(f"   • {col}: {count} ({count/len(df)*100:.1f}%)")
        else:
            print(f"\n No hay valores faltantes en la muestra")
        
        # Estadísticas básicas
        if 'price_usd' in df.columns:
            print(f"\nEstadísticas de precios:")
            print(f"   Precio mínimo: ${df['price_usd'].min():,.0f}")
            print(f"   Precio máximo: ${df['price_usd'].max():,.0f}")
            print(f"   Precio promedio: ${df['price_usd'].mean():,.0f}")
            print(f"   Precio mediano: ${df['price_usd'].median():,.0f}")
        
        print(f"\n Dataset listo para usar!")
        return True
        
    except Exception as e:
        print(f" Error cargando dataset: {e}")
        return False

if __name__ == "__main__":
    success = test_dataset()
    if success:
        print("\n Prueba del dataset exitosa!")
    else:
        print("\n Error en la prueba del dataset")
