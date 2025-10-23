# Análisis de Precios de Propiedades - Properati Argentina

## Resumen del Proyecto

Este proyecto implementa un sistema completo de análisis de precios de propiedades utilizando técnicas de machine learning. El objetivo es predecir precios de propiedades en Argentina utilizando el dataset de Properati, comparando múltiples algoritmos de regresión y evaluando su rendimiento mediante métricas estándar.

### Objetivos Principales

- **Construcción del modelo**: Uso de pipelines para preprocesamiento y comparación de al menos 2 algoritmos de regresión
- **Almacenamiento de datos**: Base de datos relacional para datos de entrada, resultados y configuraciones
- **Visualización**: Gráficos y tablas para presentar resultados clave
- **Evaluación**: Métricas RMSE, MAE y R² para comparar modelos

### Tecnologías Utilizadas

- **Python 3.8+**
- **Scikit-learn**: Algoritmos de machine learning
- **Pandas & NumPy**: Manipulación de datos
- **Matplotlib & Seaborn**: Visualizaciones
- **SQLAlchemy**: Base de datos
- **XGBoost & LightGBM**: Algoritmos avanzados
- **Jupyter Notebooks**: Análisis exploratorio

---

## Estructura del Directorio

```
tp-final/
├── data/                          # Datos del proyecto
│   ├── raw/                       # Datos originales (dataset de Properati)
│   ├── processed/                 # Datos preprocesados
│   └── external/                  # Datos externos
├── src/                           # Código fuente
│   ├── data_processing.py         # Procesamiento y limpieza de datos
│   ├── models.py                  # Algoritmos de machine learning
│   ├── database.py                # Gestión de base de datos
│   ├── visualization.py           # Visualizaciones
│   └── config.py                  # Configuración del proyecto
├── scripts/                       # Scripts ejecutables
│   ├── main_pipeline.py          # Pipeline principal
│   ├── download_data.py          # Descarga de datos
│   └── setup_database.py         # Configuración de BD
├── notebooks/                     # Jupyter Notebooks
│   └── 01_data_exploration.ipynb  # Exploración de datos
├── reports/                       # Reportes y visualizaciones
├── requirements.txt               # Dependencias
└── README.md                     # Este archivo
```

---

## Instalación y Configuración

### 1. Clonar el Repositorio

```bash
git clone https://github.com/ASIG00202-Programacion-Avanzada/tp-final
cd tp-final
```

### 2. Configurar el Entorno Virtual y Dependencias

```bash
# Crear un entorno virtual
python -m venv venv

# Activar el entorno (en Windows)
.\venv\Scripts\activate

# Activar el entorno (en macOS/Linux)
source venv/bin/activate

# Instalar las librerías necesarias
pip install -r requirements.txt
```

### 4. Configurar Kaggle API

1. Crear cuenta en [Kaggle](https://www.kaggle.com/)
2. Descargar `kaggle.json` desde tu perfil
3. Colocar en `~/.kaggle/kaggle.json`

---

## Uso del Proyecto

### **Ejecución Rápida del Proyecto**


### **Paso 1. Adquisición de datos**

```bash
# Descarga del dataset desde Kanggle
python .\scripts\download_data.py
```

### **Paso 2. Armado de DB y preprocesamiento de datos **

```bash
# Verifica que este la data y armado de database
python scripts/setup_db.py
# Procesa la data y carga en DB
python .\scripts\data_processing.py
```
#### **Paso 3: Ejecutar Análisis**
```bash
# Análisis completo con visualizaciones
python robust_analysis.py
```

#### **Paso 4: App strealit**
```bash
# Prueba de modelos en app streamlit
streamlit run .\scripts\app.py
```

#### **Paso 5: Ver Resultados**
```bash
# Los resultados se guardan en:
# - reports/analysis_results.png (gráficos)
# - reports/analysis_summary.md (resumen)
```

### **¿Qué hace el análisis?**

1. **Carga** el dataset de Properati (muestra de 30,000 registros)
2. **Limpia** los datos (elimina outliers, valores faltantes)
3. **Crea características** (precio por m², total de habitaciones, etc.)
4. **Entrena modelos** (Linear Regression, Random Forest)
5. **Evalúa** con métricas (R², RMSE, MAE)
6. **Genera visualizaciones** automáticamente


## Algoritmos Implementados

### Algoritmos de Regresión

1. **Linear Regression**: Regresión lineal básica
2. **Ridge Regression**: Regresión con regularización L2
3. **Lasso Regression**: Regresión con regularización L1
4. **Random Forest**: Ensemble de árboles de decisión
5. **Gradient Boosting**: Boosting de gradientes
6. **XGBoost**: Gradient boosting optimizado
7. **LightGBM**: Gradient boosting ligero
8. **SVR**: Support Vector Regression

### Métricas de Evaluación

- **RMSE**: Root Mean Square Error**
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination
- **Validación Cruzada**: 5-fold cross-validation

---

## Base de Datos

### Tablas Implementadas

1. **`input_data`**: Datos de entrada preprocesados
   - `property_id`, `property_type`,`operation_type`, `location`
   - `surface_total`, `surface_covered`, `rooms`
   - `bedrooms`, `bathrooms`, `price_usd`

2. **`model_results`**: Resultados de modelos
   - `model_name`, `model_version`
   - `test_rmse`, `test_mae`, `test_r2`
   - `cv_rmse_mean`, `cv_mae_mean`, `cv_r2_mean`
   - `hyperparameters`, `feature_importance`

3. **`model_config`**: Configuraciones de modelos
   - `model_name`, `config_name`
   - `parameters`, `preprocessing_steps`
   - `feature_engineering`

---

## Visualizaciones

### Gráficos Generados

1. **Distribución de Precios**: Histogramas y box plots
2. **Análisis Geográfico**: Precios por ubicación
3. **Tipos de Propiedad**: Distribución y precios
4. **Análisis de Superficie**: Relación superficie-precio
5. **Comparación de Modelos**: Métricas de rendimiento
6. **Predicciones vs Reales**: Scatter plots
7. **Feature Importance**: Importancia de características

### Dashboard Interactivo

Se incluye un dashboard interactivo con Plotly para exploración avanzada de datos.

---

## Resultados Esperados

### Archivos de Salida

- `reports/model_comparison.png`: Comparación visual de modelos
- `reports/feature_importance.png`: Importancia de características
- `reports/analysis_report.md`: Reporte completo del análisis
- `reports/model_comparison.csv`: Tabla de resultados
- `reports/data_exploration.json`: Información de exploración

### Métricas de Evaluación

El proyecto evalúa modelos usando:
- **RMSE**: Error cuadrático medio
- **MAE**: Error absoluto medio  
- **R²**: Coeficiente de determinación
- **Validación Cruzada**: Estabilidad del modelo

---

## Configuración Avanzada

### Personalización de Modelos

Editar `src/models.py` para:
- Agregar nuevos algoritmos
- Modificar hiperparámetros
- Cambiar métricas de evaluación

---

## Documentación Adicional

### Notebooks de Análisis

- `01_data_exploration.ipynb`: Exploración inicial de datos
- Análisis de distribuciones, correlaciones y patrones
- Visualizaciones interactivas

### Scripts de Utilidad

- `download_data.py`: Descarga automática del dataset
- `setup_db.py`: Configuración de base de datos e inicialización
- `main_pipeline.py`: Pipeline completo automatizado

---

## Profesor

**Juan Carlos Cifuentes Duran**

---
