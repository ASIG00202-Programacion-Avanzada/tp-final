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
