import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))


# --- Configuración de la Página ---
st.set_page_config(
    page_title="Predictor de Precios de Propiedades",
    page_icon="🏠",
    layout="wide"
)

# --- Carga de Modelos y Artefactos ---
@st.cache_resource
def load_models_and_artifacts():
    """
    Carga los modelos y el preprocesador desde la carpeta 'models/'.
    """
    base_path = 'models'
    try:
        lr_model = joblib.load(os.path.join(base_path, 'linear_regression.joblib'))
        rf_model = joblib.load(os.path.join(base_path, 'random_forest.joblib'))
        
        # Cargamos el NUEVO preprocesador
        preprocessor = joblib.load(os.path.join(base_path, 'preprocessor.joblib'))
        
        models = {
            'Regresión Lineal': lr_model,
            'Random Forest': rf_model
        }
        return models, preprocessor
        
    except FileNotFoundError as e:
        st.error(f"Error al cargar archivos del modelo: {e}")
        st.error("Asegúrate de haber ejecutado 'robust_analysis.py' primero.")
        st.error("Este script genera 'preprocessor.joblib' y los modelos.")
        return None, None

models, preprocessor = load_models_and_artifacts()

# Si los modelos no se cargaron, detener la app
if models is None:
    st.stop()

# --- Interfaz de Usuario (Sidebar) ---
st.sidebar.title("🏠 Predictor de Precios")
st.sidebar.markdown("Ingrese las características de la propiedad.")

# Selector de modelo
model_choice = st.sidebar.selectbox(
    "Seleccione el modelo:",
    options=list(models.keys())
)

st.sidebar.header("Características Principales")

# --- MODIFICADO: Inputs para el nuevo modelo ---
operation_type = st.sidebar.selectbox(
    "Tipo de Operación:",
    options=['Venta', 'Alquiler'], # Asegúrate que coincida (Venta, Alquiler)
    index=0
)

property_type = st.sidebar.selectbox(
    "Tipo de Propiedad:",
    options=['Departamento', 'Casa', 'PH', 'Otro'], # Ajusta según tus datos
    index=0
)

surface_total = st.sidebar.number_input(
    "Superficie Total (m²)", 
    min_value=10, max_value=2000, value=60, step=5
)

surface_covered = st.sidebar.number_input(
    "Superficie Cubierta (m²)", 
    min_value=10, max_value=2000, value=50, step=5
)

rooms = st.sidebar.number_input(
    "Total de Ambientes", 
    min_value=1, max_value=10, value=2, step=1
)

bedrooms = st.sidebar.number_input(
    "Dormitorios", 
    min_value=0, max_value=10, value=1, step=1
)

bathrooms = st.sidebar.number_input(
    "Baños", 
    min_value=0, max_value=10, value=1, step=1
)


# Botón para predecir
predict_button = st.sidebar.button("Predecir Precio", type="primary")

# --- Lógica de Predicción y Visualización (MODIFICADO) ---
st.title("Estimación del Precio de la Propiedad")

if predict_button:
    
    # 1. Crear diccionario con datos de entrada
    # Debe contener TODAS las columnas que 'data_processing.py' crea
    # y 'robust_analysis.py' usa para entrenar.
    input_data = {
        'operation_type': operation_type,
        'property_type': property_type,
        'surface_total': surface_total,
        'surface_covered': surface_covered,
        'rooms': rooms,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms
    }
    
    # Replicar Feature Engineering de 'data_processing.py'
    try:
        input_data['total_rooms'] = input_data['bedrooms'] + input_data['bathrooms']
        
        if input_data['surface_total'] > 0:
            input_data['surface_ratio'] = input_data['surface_covered'] / input_data['surface_total']
            input_data['room_density'] = input_data['rooms'] / input_data['surface_total']
        else:
            input_data['surface_ratio'] = 1.0
            input_data['room_density'] = 0.0
            
        input_data['surface_category'] = pd.cut(
            [input_data['surface_total']],
            bins=5, # Usamos 5 bins como en data_processing
            labels=['Muy Pequeña', 'Pequeña', 'Mediana', 'Grande', 'Muy Grande'],
            right=False # Asegurar consistencia
        )[0]
        
        
        input_df = pd.DataFrame([input_data])
        
        
        input_df_processed = preprocessor.transform(input_df)
        
        
        selected_model = models[model_choice]
        prediction = selected_model.predict(input_df_processed)
        
        
        st.header(f"Resultado con {model_choice} para {operation_type}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"El precio estimado es:")
            st.metric(label="Precio (USD)", value=f"$ {prediction[0]:,.2f}")
        
        with col2:
            st.info("Datos Ingresados")
            # Mostrar los datos que el usuario ingresó
            display_data = {
                'Tipo Operación': operation_type,
                'Tipo Propiedad': property_type,
                'Superficie Total': surface_total,
                'Superficie Cubierta': surface_covered,
                'Ambientes': rooms,
                'Dormitorios': bedrooms,
                'Baños': bathrooms
            }
            st.dataframe(pd.DataFrame([display_data]).T.rename(columns={0: 'Valor'}))

    except Exception as e:
        st.error(f"Error durante la predicción: {e}")
        st.error("Verifica que las features de la app coincidan con las del entrenamiento.")
        
else:
    st.info("Por favor, ingrese los datos en la barra lateral y presione 'Predecir Precio'.")