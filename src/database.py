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

Base = declarative_base()

class InputData(Base):
    """Table for storing preprocessed input data."""
    __tablename__ = 'input_data'
    
    id = Column(Integer, primary_key=True)
    property_id = Column(String(50), unique=True)
    property_type = Column(String(50))
    location = Column(String(100))
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

Base = declarative_base()

class InputData(Base):
    """Table for storing preprocessed input data."""
    __tablename__ = 'input_data'
    
    id = Column(Integer, primary_key=True)
    property_id = Column(String(50), unique=True)
    property_type = Column(String(50))
    location = Column(String(100))
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
        if DB_CONFIG["type"] == "postgresql":
            connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['name']}"
        elif DB_CONFIG["type"] == "sqlite":
            # Para SQLite, usar ruta relativa al proyecto
            db_path = Path(__file__).parent.parent / "data" / DB_CONFIG['name']
            connection_string = f"sqlite:///{db_path}"
        else:
            raise ValueError(f"Database type {DB_CONFIG['type']} not supported")
        
        self.engine = create_engine(connection_string)
        self.Session = sessionmaker(bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(self.engine)
    
