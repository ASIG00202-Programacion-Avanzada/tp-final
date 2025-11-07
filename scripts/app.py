import streamlit as st
import pandas as pd
import joblib
import os
import sys
from pathlib import Path

# --- Configuración del Path ---
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Predictor Inmobiliario",
    page_icon="🏠",
    layout="centered"
)

# --- Carga Dinámica de Modelos ---
@st.cache_resource
def load_models_dynamic():
    """
    Explora la carpeta 'models/' y carga todos los pipelines disponibles
    automáticamente, sin importar cuántos sean.
    """
    models_dir = Path('models')
    if not models_dir.exists():
        st.error("La carpeta 'models/' no existe. Ejecuta primero un script de entrenamiento.")
        return None

    models = {}
    # Buscamos todos los archivos que terminen en _pipeline.joblib
    for model_file in models_dir.glob('*_pipeline.joblib'):
        try:
            # Convierte los nombres de archivos a nombre legible
            # Ej: 'linear_regression_pipeline.joblib' -> 'Linear Regression'
            model_name = model_file.stem.replace('_pipeline', '').replace('_', ' ').title()
            
            # Cargar el pipeline completo
            models[model_name] = joblib.load(model_file)
            print(f"Modelo cargado: {model_name}")
            
        except Exception as e:
            st.warning(f"No se pudo cargar {model_file.name}: {e}")

    if not models:
        st.error("No se encontraron modelos válidos en 'models/'.")
        return None
    
    return models

# Cargar modelos al iniciar la app
models_loaded = load_models_dynamic()
if models_loaded is None:
    st.stop()

# --- Interfaz Gráfica ---
st.title("🏠 Estimador de Valor de Propiedades")
st.markdown(f"**Modelos disponibles:** {len(models_loaded)}")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("🛠️ Configuración")
    
    # Selector dinámico basado en los modelos encontrados
    selected_model_name = st.selectbox(
        "Selecciona el Modelo Predictivo",
        options=sorted(list(models_loaded.keys()))
    )
    
    st.markdown("---")
    st.header("📋 Características de la Propiedad")
    
    operation_type = st.selectbox("Operación", ['Venta', 'Alquiler', 'Alquiler temporal'])
    property_type = st.selectbox("Tipo de Propiedad", ['Departamento', 'Casa', 'PH', 'Local comercial', 'Oficina', 'Lote'])
    
    col1, col2 = st.columns(2)
    with col1:
        surface_total = st.number_input("Sup. Total (m²)", min_value=10, value=70, step=5)
    with col2:
         surface_covered = st.number_input("Sup. Cubierta (m²)", min_value=10, value=60, step=5)
         
    rooms = st.slider("Ambientes", 1, 15, 3)
    col3, col4 = st.columns(2)
    with col3:
        bedrooms = st.number_input("Dormitorios", min_value=0, max_value=10, value=2)
    with col4:
        bathrooms = st.number_input("Baños", min_value=1, max_value=10, value=1)
    
    predict_btn = st.button("Calcular Precio!", type="primary", use_container_width=True)

# --- Lógica de Predicción ---
if predict_btn:
    # 1. Construir DataFrame con los inputs
    input_data = pd.DataFrame([{
        'operation_type': operation_type,
        'property_type': property_type,
        'surface_total': float(surface_total),
        'surface_covered': float(surface_covered),
        'rooms': int(rooms),
        'bedrooms': int(bedrooms),
        'bathrooms': int(bathrooms)
    }])
    
    # 2. Obtener el pipeline seleccionado y predecir
    pipeline = models_loaded[selected_model_name]
    
    try:
        with st.spinner(f"Calculando con {selected_model_name}..."):
            prediction = pipeline.predict(input_data)[0]
            
        st.success("Cálculo exitoso")
        
        # Muestra principal del resultado
        st.markdown("### Precio Estimado de Mercado")
        st.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>US$ {prediction:,.0f}</h1>", unsafe_allow_html=True)
        
        # Detalles adicionales
        with st.expander("ℹ️ Ver detalles de la entrada"):
            st.dataframe(input_data, hide_index=True)
            st.caption(f"Modelo utilizado: {selected_model_name}")

    except Exception as e:
        st.error("Error al realizar la predicción")
        st.write(e)
        st.warning("""
        **Posible causa:** Las características ingresadas no coinciden exactamente con las que el modelo espera.
        Revisa si faltan columnas (ej. 'location') que se usaron durante el entrenamiento.
        """)

else:
    st.info("Configura las características de la propiedad en el menú lateral para obtener una estimación.")