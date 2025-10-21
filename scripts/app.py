import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

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
    Carga los modelos, el imputador y las listas de columnas desde la carpeta 'models/'.
    """
    base_path = 'models'
    try:
        lr_model = joblib.load(os.path.join(base_path, 'linear_regression.joblib'))
        rf_model = joblib.load(os.path.join(base_path, 'random_forest.joblib'))
        imputer = joblib.load(os.path.join(base_path, 'imputer.joblib'))
        feature_cols = joblib.load(os.path.join(base_path, 'feature_cols.joblib'))
        numeric_cols = joblib.load(os.path.join(base_path, 'numeric_cols.joblib')) # <--- MODIFICADO
        
        models = {
            'Regresión Lineal': lr_model,
            'Random Forest': rf_model
        }
        return models, imputer, feature_cols, numeric_cols
        
    except FileNotFoundError as e:
        st.error(f"Error al cargar archivos del modelo: {e}")
        st.error(f"Asegúrate de haber ejecutado el script 'train.py' primero para generar los archivos en la carpeta '{base_path}'.")
        return None, None, None, None

models, imputer, feature_cols, numeric_cols = load_models_and_artifacts()

# Si los modelos no se cargaron, detener la app
if models is None:
    st.stop()

# --- Interfaz de Usuario (Sidebar) ---
st.sidebar.title("🏠 Predictor de Precios")
st.sidebar.markdown("Ingrese las características de la propiedad para estimar su precio.")

# Selector de modelo
model_choice = st.sidebar.selectbox(
    "Seleccione el modelo de predicción:",
    options=list(models.keys())
)

# --- MODIFICADO: Selector de Tipo de Operación ---
op_type = st.sidebar.selectbox(
    "Tipo de Operación:",
    options=['Venta', 'Alquiler'],
    index=0 # Default en 'Venta'
)

st.sidebar.header("Características de la Propiedad")

# Las columnas numéricas que esperamos (basadas en numeric_cols.joblib)
# ['surface_total', 'surface_covered', 'rooms', 'bedrooms', 'bathrooms', 'total_rooms']
# Pediremos las 5 primeras, 'total_rooms' se calcula.

surface_total = st.sidebar.number_input(
    "Superficie Total (m²)", 
    min_value=10, max_value=1000, value=60, step=5
)

surface_covered = st.sidebar.number_input(
    "Superficie Cubierta (m²)", 
    min_value=10, max_value=1000, value=50, step=5
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

# --- Lógica de Predicción y Visualización ---
st.title("Estimación del Precio de la Propiedad")

if predict_button:
    # 1. Crear 'total_rooms' como en el entrenamiento
    total_rooms = bedrooms + bathrooms
    
    # 2. Crear diccionario con los datos de entrada numéricos
    input_data = {
        'surface_total': surface_total,
        'surface_covered': surface_covered,
        'rooms': rooms,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'total_rooms': total_rooms
    }
    
    # 3. Crear DataFrame solo con las columnas numéricas que se imputan
    # Filtramos el diccionario por las columnas que realmente están en numeric_cols
    input_numeric_data = {k: v for k, v in input_data.items() if k in numeric_cols}
    input_df = pd.DataFrame([input_numeric_data])

    # --- MODIFICADO: Procesamiento de 'operation_type' ---
    try:
        # 4. Añadir la columna categórica
        input_df['operation_type'] = op_type
        
        # 5. Aplicar get_dummies (creará 'operation_type_Venta' o 'operation_type_Alquiler')
        input_df_processed = pd.get_dummies(input_df, columns=['operation_type'])
        
        # 6. REINDEXAR: Esta es la parte clave.
        # Asegura que el DataFrame tenga EXACTAMENTE las mismas columnas que 'feature_cols'
        # Rellena con 0 las columnas dummy que no se crearon (ej. si se eligió 'Alquiler', 
        # 'operation_type_Venta' no existirá, así que reindex la crea y la pone en 0)
        input_df_final = input_df_processed.reindex(columns=feature_cols, fill_value=0)
        
        # 7. Aplicar el imputador (transform, NO fit) a las columnas numéricas
        # Copiamos para evitar advertencias
        input_df_imputed = input_df_final.copy()
        input_df_imputed[numeric_cols] = imputer.transform(input_df_final[numeric_cols])
        
        # 8. Seleccionar modelo y predecir
        selected_model = models[model_choice]
        prediction = selected_model.predict(input_df_imputed)
        
        # 9. Mostrar resultado
        st.header(f"Resultado con {model_choice} (para {op_type})")
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"El precio estimado es:")
            st.metric(label="Precio (USD)", value=f"$ {prediction[0]:,.2f}")
        
        with col2:
            st.info("Datos Ingresados")
            # Mostrar los datos originales que el usuario ingresó
            display_data = input_data.copy()
            display_data['operation_type'] = op_type
            st.dataframe(pd.DataFrame([display_data]).T.rename(columns={0: 'Valor'}))
            
            # (Opcional) Mostrar los datos procesados para debug
            # st.subheader("Datos procesados (para el modelo)")
            # st.dataframe(input_df_imputed)

    except Exception as e:
        st.error(f"Error durante la predicción: {e}")
        st.error("Verifica que los archivos 'models/*.joblib' estén actualizados con la nueva lógica.")
else:
    st.info("Por favor, ingrese los datos en la barra lateral y presione 'Predecir Precio'.")