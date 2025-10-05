"""
Tests unitarios para el módulo de modelos.
"""
import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models import ModelTrainer
from data_processing import DataProcessor

class TestModels(unittest.TestCase):
    """Tests para la clase ModelTrainer."""
    
    def setUp(self):
        """Configurar datos de prueba."""
        np.random.seed(42)
        n_samples = 100
        
        self.X_train = np.random.randn(n_samples, 5)
        self.y_train = np.random.randn(n_samples)
        self.X_test = np.random.randn(20, 5)
        self.y_test = np.random.randn(20)
        
        self.trainer = ModelTrainer()
    
    def test_define_models(self):
        """Test que los modelos se definan correctamente."""
        models = self.trainer.define_models()
        
        self.assertIsInstance(models, dict)
        self.assertGreater(len(models), 0)
        
        # Verificar que todos los modelos son instancias válidas
        for name, model in models.items():
            self.assertIsNotNone(model)
    
    def test_train_single_model(self):
        """Test entrenamiento de un modelo individual."""
        from sklearn.linear_model import LinearRegression
        
        model = LinearRegression()
        results = self.trainer.train_model(
            model, self.X_train, self.y_train, 
            self.X_test, self.y_test, "Test Model"
        )
        
        self.assertIsNotNone(results)
        self.assertIn('test_r2', results)
        self.assertIn('test_rmse', results)
        self.assertIn('test_mae', results)
    
    def test_model_metrics(self):
        """Test que las métricas se calculen correctamente."""
        from sklearn.linear_model import LinearRegression
        
        model = LinearRegression()
        results = self.trainer.train_model(
            model, self.X_train, self.y_train, 
            self.X_test, self.y_test, "Test Model"
        )
        
        # Verificar que las métricas son números válidos
        self.assertIsInstance(results['test_r2'], (int, float))
        self.assertIsInstance(results['test_rmse'], (int, float))
        self.assertIsInstance(results['test_mae'], (int, float))
        
        # Verificar que RMSE y MAE son positivos
        self.assertGreaterEqual(results['test_rmse'], 0)
        self.assertGreaterEqual(results['test_mae'], 0)

class TestDataProcessor(unittest.TestCase):
    """Tests para la clase DataProcessor."""
    
    def setUp(self):
        """Configurar datos de prueba."""
        np.random.seed(42)
        n_samples = 100
        
        self.df = pd.DataFrame({
            'property_type': np.random.choice(['Casa', 'Departamento'], n_samples),
            'location': np.random.choice(['Buenos Aires', 'Córdoba'], n_samples),
            'surface_total': np.random.normal(100, 30, n_samples),
            'surface_covered': np.random.normal(80, 25, n_samples),
            'rooms': np.random.randint(1, 6, n_samples),
            'bedrooms': np.random.randint(1, 4, n_samples),
            'bathrooms': np.random.randint(1, 3, n_samples),
            'price_usd': np.random.normal(150000, 50000, n_samples)
        })
        
        # Asegurar valores positivos
        self.df['surface_total'] = np.abs(self.df['surface_total'])
        self.df['surface_covered'] = np.abs(self.df['surface_covered'])
        self.df['price_usd'] = np.abs(self.df['price_usd'])
        
        self.processor = DataProcessor()
    
    def test_load_data(self):
        """Test carga de datos."""
        # Crear archivo temporal
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            loaded_df = self.processor.load_data(temp_path)
            self.assertIsInstance(loaded_df, pd.DataFrame)
            self.assertGreater(len(loaded_df), 0)
        finally:
            os.unlink(temp_path)
    
    def test_clean_data(self):
        """Test limpieza de datos."""
        cleaned_df = self.processor.clean_data(self.df)
        
        self.assertIsInstance(cleaned_df, pd.DataFrame)
        self.assertLessEqual(len(cleaned_df), len(self.df))
    
    def test_create_features(self):
        """Test creación de características."""
        features_df = self.processor.create_features(self.df)
        
        self.assertIsInstance(features_df, pd.DataFrame)
        # Verificar que se crearon nuevas columnas
        self.assertGreater(len(features_df.columns), len(self.df.columns))

if __name__ == '__main__':
    unittest.main()
