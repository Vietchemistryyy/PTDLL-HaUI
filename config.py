"""Cấu hình Project Phân tích Bệnh Tim mạch"""
import os
from pathlib import Path

# Đường dẫn project
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"

# Tạo thư mục nếu chưa tồn tại
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# File paths
RAW_DATA_FILE = RAW_DATA_DIR / "cardio_train.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "cardio_processed.parquet"
MODEL_PATH = MODEL_DIR / "logistic_model"

# Cấu hình Spark
SPARK_CONFIG = {
    "app_name": "Cardiovascular Disease Analysis",
    "master": "local[*]",
    "driver_memory": "4g",
    "executor_memory": "4g"
}

# Cột dữ liệu
FEATURE_COLUMNS = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                   'cholesterol', 'gluc', 'smoke', 'alco', 'active']
TARGET_COLUMN = 'cardio'

# Phân loại features
NUMERICAL_FEATURES = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
CATEGORICAL_FEATURES = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

# Ngưỡng outliers
OUTLIER_THRESHOLDS = {
    'ap_hi': (80, 200),      # Huyết áp tâm thu
    'ap_lo': (60, 140),      # Huyết áp tâm trương
    'height': (120, 220),    # Chiều cao (cm)
    'weight': (30, 200),     # Cân nặng (kg)
    'age': (10950, 25550)    # Tuổi (days: 30-70 years)
}

# Model parameters
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# Streamlit config
APP_TITLE = "Phân tích & Dự báo Bệnh Tim mạch"
