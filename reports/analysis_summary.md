# Resumen del Análisis - Properati Argentina

## **Resultados del Análisis**

### **Dataset Procesado**
- **Muestra inicial**: 30,000 registros
- **Después de limpieza**: 17,832 registros
- **Características utilizadas**: 7 variables numéricas
- **División**: 80% entrenamiento (14,265), 20% prueba (3,567)

### **Modelos Entrenados**

#### **1. Linear Regression**
- **R²**: 0.6350 (63.5% de varianza explicada)
- **RMSE**: $78,658
- **MAE**: $48,256
- **Interpretación**: Modelo básico con rendimiento moderado

#### **2. Random Forest**
- **R²**: 0.9973 (99.73% de varianza explicada)
- **RMSE**: $6,740
- **MAE**: $1,704
- **Interpretación**: Excelente rendimiento, posible overfitting

### **Características Utilizadas**
1. **surface_total**: Superficie total de la propiedad
2. **surface_covered**: Superficie cubierta
3. **rooms**: Número total de habitaciones
4. **bedrooms**: Número de dormitorios
5. **bathrooms**: Número de baños
6. **price_per_sqm**: Precio por metro cuadrado
7. **total_rooms**: Total de habitaciones (dormitorios + baños)

### **Conclusiones**

#### **Fortalezas del Modelo**
-  **Random Forest** muestra excelente capacidad predictiva
-  **Características de superficie** son muy importantes
-  **Datos limpios** y bien procesados
-  **Métricas consistentes** entre entrenamiento y prueba

#### **Limitaciones Identificadas**
-  **Posible overfitting** en Random Forest (R² muy alto)
-  **Linear Regression** limitado para datos no lineales
-  **Falta de variables categóricas** (tipo de propiedad, ubicación)

#### **Recomendaciones**
1. **Validación cruzada** para verificar estabilidad del modelo
2. **Incluir variables categóricas** (tipo de propiedad, ubicación)
3. **Regularización** para evitar overfitting
4. **Más datos** para mejorar generalización

### **Próximos Pasos**
1. Implementar validación cruzada
2. Agregar variables categóricas
3. Probar otros algoritmos (XGBoost, LightGBM)
4. Optimizar hiperparámetros
5. Crear pipeline de producción

---
*Análisis generado automáticamente el 5 de octubre de 2024*
