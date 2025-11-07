"""
Database module for storing and retrieving data and model results.
"""
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON
from datetime import datetime
import json
from config import DB_CONFIG
from pathlib import Path 

Base = declarative_base()


def safe_cast(value, cast_type, default=None):
    """
    Intenta castear un valor, devolviendo un default si falla o es NaN.
    """
    if pd.isna(value):
        return default
    try:
        return cast_type(value)
    except (ValueError, TypeError):
        return default

class InputData(Base):
    """Table for storing preprocessed input data."""
    __tablename__ = 'input_data'
    
    id = Column(Integer, primary_key=True)
    property_id = Column(String(50), unique=True)
    property_type = Column(String(50))
    operation_type = Column(String(50))
    province = Column(String(100))
    department = Column(String(100))
    surface_total = Column(Float)
    surface_covered = Column(Float)
    rooms = Column(Integer)
    bedrooms = Column(Integer)
    bathrooms = Column(Integer)
    price_usd = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    raw_data = Column(JSON)

class ModelResults(Base):
    """Table for storing model predictions and metrics."""
    __tablename__ = 'model_results'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(100))
    model_version = Column(String(20))
    train_date = Column(DateTime, default=datetime.utcnow)
    test_rmse = Column(Float)
    test_mae = Column(Float)
    test_r2 = Column(Float)
    cv_rmse_mean = Column(Float)
    cv_rmse_std = Column(Float)
    cv_mae_mean = Column(Float)
    cv_mae_std = Column(Float)
    cv_r2_mean = Column(Float)
    cv_r2_std = Column(Float)
    hyperparameters = Column(JSON)
    feature_importance = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class ModelConfig(Base):
    """Table for storing model configuration and parameters."""
    __tablename__ = 'model_config'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(100))
    config_name = Column(String(100))
    parameters = Column(JSON)
    preprocessing_steps = Column(JSON)
    feature_engineering = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Integer, default=1)

class DatabaseManager:
    """Database manager for handling database operations."""
    
    def __init__(self):
        self.engine = None
        self.Session = None
        self._connect()
    
    def _connect(self):
        """Establish database connection."""
        if DB_CONFIG["type"] == "sqlite":
            # Para SQLite, usar ruta relativa al proyecto
            db_path = Path(__file__).parent.parent / "data" / DB_CONFIG['name']
            connection_string = f"sqlite:///{db_path}"
        else:
            raise ValueError(f"Database type {DB_CONFIG['type']} not supported")
        
        self.engine = create_engine(connection_string)
        self.Session = sessionmaker(bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(self.engine)
    

    def store_input_data(self, df: pd.DataFrame):
        """
        Almacena datos de entrada preprocesados.
        """
        session = self.Session()
        try:
            
            session.query(InputData).delete()
            session.commit()
            print("Datos antiguos de 'input_data' eliminados.")

            # 2. Preparar los datos para la inserción
            data_to_insert = []
            for _, row in df.iterrows():
                
                # 3. Serialización segura de raw_data
                # to_json() maneja dtypes de pandas (Categorical, nan)
                # default_handler=str es un seguro para tipos no esperados
                serializable_raw_data = json.loads(
                    row.to_json(default_handler=str, date_format='iso')
                )

                input_data = InputData(
                    # 4. Uso safe_cast para evitar errores de NaN -> int
                    property_id=str(row.get('property_id', '')),
                    property_type=safe_cast(row.get('property_type'), str, ''),
                    operation_type=safe_cast(row.get('operation_type'), str, ''),
                    province=safe_cast(row.get('province'), str, ''),
                    department=safe_cast(row.get('department'), str, ''),
                    surface_total=safe_cast(row.get('surface_total'), float, None),
                    surface_covered=safe_cast(row.get('surface_covered'), float, None),
                    rooms=safe_cast(row.get('rooms'), int, None),
                    bedrooms=safe_cast(row.get('bedrooms'), int, None),
                    bathrooms=safe_cast(row.get('bathrooms'), int, None),
                    price_usd=safe_cast(row.get('price_usd'), float, None),
                    raw_data=serializable_raw_data
                )
                data_to_insert.append(input_data)
            
            # 5. Usar bulk_save_objects para inserción rápida
            session.bulk_save_objects(data_to_insert)
            session.commit()
            print(f"Stored {len(data_to_insert)} records in input_data table")
            
        except Exception as e:
            session.rollback()
            # Imprime el error real
            print(f"Error storing input data: {e}") 
        finally:
            session.close()

    
    def store_model_results(self, model_name: str, model_version: str, 
                          metrics: dict, hyperparameters: dict, 
                          feature_importance: dict = None):
        """Store model results and metrics."""
        session = self.Session()
        try:
            result = ModelResults(
                model_name=model_name,
                model_version=model_version,
                test_rmse=metrics.get('test_rmse'),
                test_mae=metrics.get('test_mae'),
                test_r2=metrics.get('test_r2'),
                cv_rmse_mean=metrics.get('cv_rmse_mean'),
                cv_rmse_std=metrics.get('cv_rmse_std'),
                cv_mae_mean=metrics.get('cv_mae_mean'),
                cv_mae_std=metrics.get('cv_mae_std'),
                cv_r2_mean=metrics.get('cv_r2_mean'),
                cv_r2_std=metrics.get('cv_r2_std'),
                hyperparameters=hyperparameters,
                feature_importance=feature_importance
            )
            session.add(result)
            session.commit()
            print(f"Stored results for {model_name} v{model_version}")
        except Exception as e:
            session.rollback()
            print(f"Error storing model results: {e}")
        finally:
            session.close()
    
    def store_model_config(self, model_name: str, config_name: str, 
                          parameters: dict, preprocessing_steps: dict,
                          feature_engineering: dict = None):
        """Store model configuration."""
        session = self.Session()
        try:
            config = ModelConfig(
                model_name=model_name,
                config_name=config_name,
                parameters=parameters,
                preprocessing_steps=preprocessing_steps,
                feature_engineering=feature_engineering
            )
            session.add(config)
            session.commit()
            print(f"Stored configuration for {model_name}: {config_name}")
        except Exception as e:
            session.rollback()
            print(f"Error storing model configuration: {e}")
        finally:
            session.close()
    
    def get_latest_model_results(self, model_name: str = None):
        """Get latest model results."""
        session = self.Session()
        try:
            query = session.query(ModelResults)
            if model_name:
                query = query.filter(ModelResults.model_name == model_name)
            results = query.order_by(ModelResults.created_at.desc()).all()
            return results
        except Exception as e:
            print(f"Error retrieving model results: {e}")
            return []
        finally:
            session.close()
    
    def get_input_data(self, limit: int = None):
        """Get input data from database."""
        session = self.Session()
        try:
            query = session.query(InputData)
            if limit:
                query = query.limit(limit)
            df = pd.read_sql(query.statement, self.engine)
            return df
        except Exception as e:
            print(f"Error retrieving input data: {e}")
            return pd.DataFrame()
        finally:
            session.close()