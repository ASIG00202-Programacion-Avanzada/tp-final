#!/usr/bin/env python3
"""
Análisis robusto del dataset de Properati Argentina.
MODIFICADO:
- Incluye 'operation_type' (Venta/Alquiler) como feature.
- Aplica One-Hot Encoding.
- Guarda los modelos entrenados y artefactos (imputer, columnas) con joblib.
- Corrige el target leak (price_per_sqm) y la imputación.
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
import joblib  
import os      

warnings.filterwarnings('ignore')

def load_and_clean_data(file_path, sample_size=30000):
    """
    Cargar y limpiar el dataset de Properati.
    """
    print("Cargando dataset...")
    
    # Cargar muestra del dataset
    try:
        # Cargamos todas las columnas, ya que filtraremos después
        df = pd.read_csv(file_path, nrows=sample_size)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en {file_path}")
        print("Asegúrate de que el archivo 'entrenamiento.csv' esté en la carpeta 'data/raw/'")
        return pd.DataFrame() # Devuelve un DataFrame vacío
        
    print(f"Dataset cargado: {df.shape}")
    
    # Limpiar datos
    print("Limpiando datos...")
    
    # --- MODIFICADO: Filtrar por operation_type ---
    if 'operation_type' in df.columns:
        valid_ops = ['Venta', 'Alquiler']
        df = df[df['operation_type'].isin(valid_ops)]
        print(f"    Después de filtrar por 'Venta' y 'Alquiler': {df.shape}")
    else:
        print("Advertencia: No se encontró la columna 'operation_type'. El modelo no distinguirá Venta/Alquiler.")

    # Eliminar filas con precio faltante
    df = df.dropna(subset=['price'])
    print(f"    Después de eliminar precios faltantes: {df.shape}")
    
    # Eliminar outliers en precio (más conservador)
    Q1 = df['price'].quantile(0.05)  
    Q3 = df['price'].quantile(0.95)
    df = df[(df['price'] >= Q1) & (df['price'] <= Q3)]
    print(f"    Después de eliminar outliers de precio: {df.shape}")
    
    # Limpiar superficie
    if 'surface_total' in df.columns:
        df = df[df['surface_total'] > 0]
        df = df[df['surface_total'] < 1000]  
        print(f"    Después de limpiar superficie: {df.shape}")
    
    print(f"Datos limpiados: {df.shape}")
    return df

def create_features(df):
    """
    Crear características para el modelo.
    DEVUELVE: X (features) e y (target) listos.
    """
    print("Creando características...")
    
    df_features = df.copy()
    
    # --- MODIFICADO: Añadir 'operation_type' ---
    important_cols = [
        'price', 'surface_total', 'surface_covered', 'rooms', 
        'bedrooms', 'bathrooms', 'operation_type' 
    ]
    available_cols = [col for col in important_cols if col in df_features.columns]
    
    # Crear dataset solo con columnas importantes
    df_features = df_features[available_cols].copy()
    
    # Crear características derivadas
    if 'bedrooms' in df_features.columns and 'bathrooms' in df_features.columns:
        df_features['total_rooms'] = df_features['bedrooms'].fillna(0) + df_features['bathrooms'].fillna(0)
    
    # Eliminar filas que aún tengan valores infinitos o NaN en el precio
    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    df_features = df_features.dropna(subset=['price'])

    # --- MODIFICADO: Aplicar One-Hot Encoding ANTES de separar X e y ---
    if 'operation_type' in df_features.columns:
        print("Aplicando One-Hot Encoding a 'operation_type'...")
        # drop_first=True evita multicolinealidad (crea 'operation_type_Venta' y 0 significa 'Alquiler')
        df_features = pd.get_dummies(df_features, columns=['operation_type'], drop_first=True)

    # --- CORRECCIÓN DE DATA LEAKAGE ---
    # 1. Separar 'y' (target)
    y = df_features['price']
    
    # 2. Separar 'X' (features), eliminando el target
    X = df_features.drop(columns=['price'], errors='ignore')
    
    # Guardar solo las columnas numéricas para la imputación
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = X.columns.tolist() # Lista completa de features (incluye dummies)

    # 3. Usar imputador SÓLO en columnas numéricas de X
    imputer = SimpleImputer(strategy='median')
    
    # Creamos un nuevo DataFrame X_imputed para evitar SettingWithCopyWarning
    X_imputed = X.copy()
    X_imputed[numeric_cols] = imputer.fit_transform(X[numeric_cols])
    X = X_imputed
    
    # --- GUARDAR ARTEFACTOS ---
    os.makedirs('models', exist_ok=True)
    joblib.dump(imputer, 'models/imputer.joblib')
    joblib.dump(feature_cols, 'models/feature_cols.joblib') # <--- Ahora incluye las dummies
    joblib.dump(numeric_cols, 'models/numeric_cols.joblib') # <--- Guardamos las cols a imputar
    
    print("Imputador y listas de columnas guardados en la carpeta 'models/'.")
    print(f"Características creadas (X shape): {X.shape}")
    print(f"Columnas de features: {feature_cols}")
    
    # Devolver X e y listos para el split
    return X, y

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
        print(f"    Entrenando {name}...")
        
        try:
            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)
            test_r2 = r2_score(y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

            results[name] = {
                'model': model,
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'test_mae': mean_absolute_error(y_test, y_pred_test),
                'predictions': y_pred_test
            }
            
            print(f"  {name}: R² = {test_r2:.4f}, RMSE = {test_rmse:.2f}")
            
        except Exception as e:
            print(f"  Error entrenando {name}: {e}")
            continue
    
    return results

def create_visualizations(df, results):
    # (Esta función no necesita cambios)
    print("Creando visualizaciones...")
    os.makedirs('reports', exist_ok=True)
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    if 'price' in df.columns:
        axes[0, 0].hist(df['price'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Distribución de Precios')
        axes[0, 0].set_xlabel('Precio')
        axes[0, 0].set_ylabel('Frecuencia')
    
    if 'surface_total' in df.columns and 'price' in df.columns:
        sample_df = df.sample(n=min(2000, len(df)))
        sns.scatterplot(data=sample_df, x='surface_total', y='price', hue='operation_type', alpha=0.6, ax=axes[0, 1])
        axes[0, 1].set_title('Precio vs Superficie (por Tipo)')
    
    if results:
        model_names = list(results.keys())
        r2_scores = [results[name]['test_r2'] for name in model_names]
        axes[1, 0].bar(model_names, r2_scores)
        axes[1, 0].set_title('Comparación de Modelos (R²)')
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    if results:
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
        y_test = results[best_model_name].get('y_test', [])
        y_pred = results[best_model_name]['predictions']
        
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
    print("  INICIANDO ANÁLISIS DE PROPERATI ARGENTINA")
    print("=" * 50)
    
    df = load_and_clean_data("data/raw/entrenamiento.csv", sample_size=50000) # Aumenté un poco el sample size
    
    if len(df) == 0:
        print("No hay datos suficientes después de la limpieza")
        return None
    
    X, y = create_features(df)
    
    if len(X) == 0:
        print("  No hay datos suficientes después de crear características")
        return None
    
    print("Preparando datos para modelado...")
    print(f"    NaN en X: {X.isnull().sum().sum()}")
    print(f"    NaN en y: {y.isnull().sum()}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=X[X.columns[X.columns.str.startswith('operation_type')]].iloc[:, 0] if any(X.columns.str.startswith('operation_type')) else None
    )
    print(f"Datos preparados - Train: {X_train.shape}, Test: {X_test.shape}")
    
    results = train_models(X_train, X_test, y_train, y_test)
    
    if results:
        if 'Linear Regression' in results:
            joblib.dump(results['Linear Regression']['model'], 'models/linear_regression.joblib')
            print("Modelo de Regresión Lineal guardado.")
        
        if 'Random Forest' in results:
            joblib.dump(results['Random Forest']['model'], 'models/random_forest.joblib')
            print("Modelo de Random Forest guardado.")
    else:
        print("No se pudieron entrenar modelos")
        return None
    
    for name in results:
        results[name]['y_test'] = y_test
    
    print("\nRESULTADOS FINALES:")
    print("=" * 30)
    for name, result in results.items():
        print(f"{name}:")
        print(f"    R²: {result['test_r2']:.4f}")
        print(f"    RMSE: {result['test_rmse']:.2f}")
    
    create_visualizations(df, results)
    
    print("Análisis completado exitosamente!")
    return results

if __name__ == "__main__":
    results = main()