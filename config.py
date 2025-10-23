"""
Configuration file for the Properati Argentina Analysis project.
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"

# Database configuration
DB_NAME = "properati_analisis.db"
DB_PATH = PROJECT_ROOT / "data" / DB_NAME
DB_CONFIG = {
    "type": "sqlite",
    "name": DB_NAME,
    "path": str(DB_PATH) # Ruta completa al archivo .db
}

# Model configuration
MODEL_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    "target_column": "price_usd"
}

# Kaggle configuration
KAGGLE_CONFIG = {
    "username": os.getenv("KAGGLE_USERNAME", ""),
    "key": os.getenv("KAGGLE_KEY", "")
}

# Creando directorios si no existen
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, REPORTS_DIR]:
    directory.mkdir(exist_ok=True)
