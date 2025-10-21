#!/usr/bin/env python3
"""
Análisis robusto del dataset de Properati Argentina.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data(file_path, sample_size=30000):
    """
    Cargar y limpiar el dataset de Properati.
    """
    print("Cargando dataset...")
    
    # Cargar muestra del dataset
    df = pd.read_csv(file_path, nrows=sample_size)
    print(f"Dataset cargado: {df.shape}")
    
    # Limpiar datos
    print("Limpiando datos...")
    
    # Eliminar filas con precio faltante
    df = df.dropna(subset=['price'])
    print(f"   Después de eliminar precios faltantes: {df.shape}")
    
    # Eliminar outliers en precio (más conservador)
    Q1 = df['price'].quantile(0.05)  # Usar percentiles más conservadores
    Q3 = df['price'].quantile(0.95)
    df = df[(df['price'] >= Q1) & (df['price'] <= Q3)]
    print(f"   Después de eliminar outliers de precio: {df.shape}")
    
    # Limpiar superficie
    if 'surface_total' in df.columns:
        df = df[df['surface_total'] > 0]
        df = df[df['surface_total'] < 1000]  # Más conservador
        print(f"   Después de limpiar superficie: {df.shape}")
    
    print(f"Datos limpiados: {df.shape}")
    return df

def create_features(df):
    """
    Crear características para el modelo.
    """
    print("Creando características...")
    
    df_features = df.copy()
    
    # Seleccionar solo las columnas numéricas más importantes
    important_cols = ['price', 'surface_total', 'surface_covered', 'rooms', 'bedrooms', 'bathrooms']
    available_cols = [col for col in important_cols if col in df_features.columns]
    
    # Crear dataset solo con columnas importantes
    df_features = df_features[available_cols].copy()
    
    # Crear características derivadas
    if 'surface_total' in df_features.columns and 'price' in df_features.columns:
        df_features['price_per_sqm'] = df_features['price'] / df_features['surface_total']
        df_features['price_per_sqm'] = df_features['price_per_sqm'].replace([np.inf, -np.inf], np.nan)
    
    if 'bedrooms' in df_features.columns and 'bathrooms' in df_features.columns:
        df_features['total_rooms'] = df_features['bedrooms'].fillna(0) + df_features['bathrooms'].fillna(0)
    
    # Usar imputador para valores faltantes
    imputer = SimpleImputer(strategy='median')
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns
    df_features[numeric_cols] = imputer.fit_transform(df_features[numeric_cols])
    
    # Eliminar filas que aún tengan valores infinitos
    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    df_features = df_features.dropna()
    
    print(f"Características creadas: {df_features.shape}")
    return df_features

def train_models(X_train, X_test, y_train, y_test):
    """
    Entrenar y comparar modelos.
    """
    print("Entrenando modelos...")
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"   Entrenando {name}...")
        
        try:
            # Entrenar modelo
            model.fit(X_train, y_train)
            
            # Predicciones
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Métricas
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            results[name] = {
                'model': model,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'predictions': y_pred_test
            }
            
            print(f" {name}: R² = {test_r2:.4f}, RMSE = {test_rmse:.2f}")
            
        except Exception as e:
            print(f" Error entrenando {name}: {e}")
            continue
    
    return results

def create_visualizations(df, results):
    """
    Crear visualizaciones de los resultados.
    """
    print("Creando visualizaciones...")
    
    # Crear directorio de reportes si no existe
    import os
    os.makedirs('reports', exist_ok=True)
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Distribución de precios
    axes[0, 0].hist(df['price'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Distribución de Precios')
    axes[0, 0].set_xlabel('Precio')
    axes[0, 0].set_ylabel('Frecuencia')
    
    # 2. Precio vs Superficie
    if 'surface_total' in df.columns:
        sample_df = df.sample(n=min(2000, len(df)))
        axes[0, 1].scatter(sample_df['surface_total'], sample_df['price'], alpha=0.5)
        axes[0, 1].set_title('Precio vs Superficie')
        axes[0, 1].set_xlabel('Superficie Total')
        axes[0, 1].set_ylabel('Precio')
    
    # 3. Comparación de modelos
    if results:
        model_names = list(results.keys())
        r2_scores = [results[name]['test_r2'] for name in model_names]
        axes[1, 0].bar(model_names, r2_scores)
        axes[1, 0].set_title('Comparación de Modelos (R²)')
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Predicciones vs reales (mejor modelo)
    if results:
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
        best_results = results[best_model_name]
        y_test = results[best_model_name].get('y_test', [])
        y_pred = best_results['predictions']
        
        if len(y_test) > 0:
            axes[1, 1].scatter(y_test, y_pred, alpha=0.6)
            axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            axes[1, 1].set_xlabel('Valores Reales')
            axes[1, 1].set_ylabel('Predicciones')
            axes[1, 1].set_title(f'Predicciones vs Reales - {best_model_name}')
    
    plt.tight_layout()
    plt.savefig('reports/analysis_results.png', dpi=300, bbox_inches='tight')
    print("Visualizaciones guardadas en reports/analysis_results.png")

def main():
    """
    Función principal del análisis.
    """
    print(" INICIANDO ANÁLISIS DE PROPERATI ARGENTINA")
    print("=" * 50)
    
    # 1. Cargar y limpiar datos
    df = load_and_clean_data("data/raw/entrenamiento.csv", sample_size=30000)
    
    if len(df) == 0:
        print("No hay datos suficientes después de la limpieza")
        return None
    
    # 2. Crear características
    df_features = create_features(df)
    
    if len(df_features) == 0:
        print(" No hay datos suficientes después de crear características")
        return None
    
    # 3. Preparar datos para modelado
    print("Preparando datos para modelado...")
    
    # Seleccionar características (excluyendo precio)
    feature_cols = [col for col in df_features.columns if col != 'price']
    X = df_features[feature_cols]
    y = df_features['price']
    
    print(f"   Características seleccionadas: {feature_cols}")
    print(f"   Shape de X: {X.shape}, Shape de y: {y.shape}")
    
    # Verificar que no hay valores NaN
    print(f"Verificando valores faltantes...")
    print(f"   NaN en X: {X.isnull().sum().sum()}")
    print(f"   NaN en y: {y.isnull().sum()}")
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Datos preparados - Train: {X_train.shape}, Test: {X_test.shape}")
    
    # 4. Entrenar modelos
    results = train_models(X_train, X_test, y_train, y_test)
    
    if not results:
        print("No se pudieron entrenar modelos")
        return None
    
    # Agregar y_test a los resultados para visualización
    for name in results:
        results[name]['y_test'] = y_test
    
    # 5. Mostrar resultados
    print("\nRESULTADOS FINALES:")
    print("=" * 30)
    for name, result in results.items():
        print(f"{name}:")
        print(f"  R²: {result['test_r2']:.4f}")
        print(f"  RMSE: {result['test_rmse']:.2f}")
        print(f"  MAE: {result['test_mae']:.2f}")
        print()
    
    # 6. Crear visualizaciones
    create_visualizations(df, results)
    
    print("Análisis completado exitosamente!")
    return results

if __name__ == "__main__":
    results = main()
