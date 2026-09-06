"""
Configuration File for ESG-Aware Multi-Agent Stock System
Centralized settings, paths, hyperparameters, and agent weights.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()

# ============================================================
# 1. BASE DIRECTORIES & PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

IS_SERVERLESS = bool(os.getenv("VERCEL"))
RUNTIME_DATA_DIR = Path("/tmp/esg_multi_agent_stock_system") if IS_SERVERLESS else BASE_DIR / "data"
DATA_DIR = RUNTIME_DATA_DIR
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = RUNTIME_DATA_DIR / "processed"
NEWS_DATA_DIR = RUNTIME_DATA_DIR / "news"
ESG_DATA_DIR = BASE_DIR / "data" / "esg"
REPORTS_DATA_DIR = RUNTIME_DATA_DIR / "reports"
VECTOR_DB_DIR = RUNTIME_DATA_DIR / "vector_db"

MODELS_DIR = Path("/tmp/esg_multi_agent_models") if IS_SERVERLESS else BASE_DIR / "models"
LSTM_MODEL_DIR = MODELS_DIR / "lstm"
GRU_MODEL_DIR = MODELS_DIR / "gru"
BILSTM_MODEL_DIR = MODELS_DIR / "bilstm"
NEURALPROPHET_MODEL_DIR = MODELS_DIR / "neuralprophet"
SCALERS_DIR = MODELS_DIR / "scalers"

# Create all necessary directories on initialization
ALL_DIRS = [
    DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, NEWS_DATA_DIR,
    ESG_DATA_DIR, REPORTS_DATA_DIR, VECTOR_DB_DIR, MODELS_DIR,
    LSTM_MODEL_DIR, GRU_MODEL_DIR, BILSTM_MODEL_DIR, NEURALPROPHET_MODEL_DIR, SCALERS_DIR
]
for d in ALL_DIRS:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. MODEL SELECTION BENCHMARKS (FROM EXPERIMENTAL EVALUATION)
# ============================================================
# Results from model selection analysis on historical stock dataset:
# GRU:     MAE: 0.030419 | RMSE: 0.039738 | R2: 0.975537 | Accuracy: 88.10% (WINNER)
# BiLSTM:  MAE: 0.037912 | RMSE: 0.048852 | R2: 0.963028 | Accuracy: 81.77%
# LSTM:    MAE: 0.042117 | RMSE: 0.053688 | R2: 0.955346 | Accuracy: 83.47%

MODEL_BENCHMARKS = {
    "GRU": {"MAE": 0.030419, "RMSE": 0.039738, "R2": 0.975537, "Accuracy": 88.10},
    "BiLSTM": {"MAE": 0.037912, "RMSE": 0.048852, "R2": 0.963028, "Accuracy": 81.77},
    "LSTM": {"MAE": 0.042117, "RMSE": 0.053688, "R2": 0.955346, "Accuracy": 83.47}
}

# ============================================================
# 3. TIME-SERIES & FORECASTING HYPERPARAMETERS
# ============================================================
SEQUENCE_LENGTH = 60         # Historical window length in trading days
TRAIN_SPLIT = 0.70           # 70% Training
VAL_SPLIT = 0.15             # 15% Validation
TEST_SPLIT = 0.15            # 15% Testing
DEFAULT_FORECAST_HORIZON = 5 # Days ahead to predict

# Model Training Parameters
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
RANDOM_SEED = 42

# Forecasting Model Ensemble Weights (Calibrated according to empirical R2 & Accuracy)
FORECAST_ENSEMBLE_WEIGHTS = {
    "gru": 0.50,          # Winner (Highest R2 97.55%, Accuracy 88.10%)
    "bilstm": 0.25,       # Second Best (R2 96.30%)
    "lstm": 0.15,         # Third (R2 95.53%)
    "neuralprophet": 0.10 # Auxiliary trend component
}

# ============================================================
# 4. TECHNICAL ANALYSIS INDICATORS
# ============================================================
SMA_SHORT = 20
SMA_LONG = 50
EMA_SHORT = 12
EMA_LONG = 26
RSI_PERIOD = 14
RSI_OVERSOLD = 30.0
RSI_OVERBOUGHT = 70.0
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2.0
ATR_PERIOD = 14

# ============================================================
# 5. FINBERT SENTIMENT ANALYSIS
# ============================================================
FINBERT_MODEL_NAME = "ProsusAI/finbert"

# ============================================================
# 6. ESG ANALYSIS FACTOR WEIGHTS
# ============================================================
ESG_WEIGHTS = {
    "environmental": 0.40,
    "social": 0.30,
    "governance": 0.30
}

# ============================================================
# 7. RAG (RETRIEVAL-AUGMENTED GENERATION) PIPELINE
# ============================================================
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
RAG_TOP_K = 5
FAISS_INDEX_PATH = VECTOR_DB_DIR / "faiss_index.bin"
FAISS_METADATA_PATH = VECTOR_DB_DIR / "faiss_metadata.pkl"

# ============================================================
# 8. RISK ASSESSMENT METRICS
# ============================================================
RISK_FREE_RATE = 0.045       # 4.5% Annualized Risk-Free Rate
TRADING_DAYS_PER_YEAR = 252
VAR_CONFIDENCE_LEVEL = 0.95

# ============================================================
# 9. AGENT BASE WEIGHTS & CONSENSUS THRESHOLDS
# ============================================================
BASE_AGENT_WEIGHTS = {
    "forecast": 0.25,
    "technical": 0.15,
    "sentiment": 0.15,
    "esg": 0.20,
    "risk": 0.15,
    "rag_llm": 0.10
}

BUY_THRESHOLD = 0.20
SELL_THRESHOLD = -0.20

# Regime Weight Adjustments
REGIME_WEIGHT_MAP = {
    "BULL": {
        "forecast": 0.30,
        "technical": 0.22,
        "sentiment": 0.15,
        "esg": 0.18,
        "risk": 0.08,
        "rag_llm": 0.07
    },
    "BEAR": {
        "forecast": 0.15,
        "technical": 0.12,
        "sentiment": 0.20,
        "esg": 0.18,
        "risk": 0.25,
        "rag_llm": 0.10
    },
    "HIGH_VOLATILITY": {
        "forecast": 0.10,
        "technical": 0.15,
        "sentiment": 0.15,
        "esg": 0.15,
        "risk": 0.35,
        "rag_llm": 0.10
    },
    "SIDEWAYS": {
        "forecast": 0.18,
        "technical": 0.25,
        "sentiment": 0.20,
        "esg": 0.18,
        "risk": 0.12,
        "rag_llm": 0.07
    }
}

# ============================================================
# 10. OLLAMA / LLM CONFIGURATION
# ============================================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# ============================================================
# 11. SYSTEM & LOGGING
# ============================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = DATA_DIR / "system.log"
