import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "SMAP_MSL"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SMAP_CHANNEL = "P-1"
WINDOW_SIZE = 100
STEP_SIZE = 10
N_FEATURES_STAT = 8

LSTM_HIDDEN_DIM = 64
LSTM_NUM_LAYERS = 2
LSTM_LATENT_DIM = 32
LSTM_EPOCHS = 30
LSTM_BATCH_SIZE = 64
LSTM_LR = 1e-3
LSTM_DROPOUT = 0.2

IF_N_ESTIMATORS = 200
IF_CONTAMINATION = 0.05
IF_MAX_SAMPLES = "auto"
IF_RANDOM_STATE = 42

OCSVM_KERNEL = "rbf"
OCSVM_NU = 0.05
OCSVM_GAMMA = "scale"

ENSEMBLE_WEIGHTS = [0.25, 0.25, 0.50]

THRESHOLD_PERCENTILE = 97

STREAM_INTERVAL_SEC = 2.0

MLFLOW_EXPERIMENT_NAME = "realtime-anomaly-detection"
MLFLOW_TRACKING_URI = str(BASE_DIR / "mlruns")

API_HOST = "0.0.0.0"
API_PORT = 8000

ALERT_LOG_PATH = BASE_DIR / "reports" / "alerts.jsonl"
