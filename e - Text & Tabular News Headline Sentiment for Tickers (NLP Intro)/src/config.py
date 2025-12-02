"""Central configuration for project-wide constants."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

# Data files
RAW_DATA_PATH = DATA_DIR / "raw_headlines.csv"
LABELED_DATA_PATH = DATA_DIR / "labeled_headlines.csv"

# Modeling / feature constants
TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "SPY"]
LANGUAGE = "en"
MAX_ARTICLES_PER_TICKER = 60
CLASS_LABELS = ["Bullish", "Neutral", "Bearish"]

# Plot file names
CONFUSION_MATRIX_FILES = {
    "Logistic Regression": FIGURES_DIR / "cm_logreg.png",
    "Linear SVC": FIGURES_DIR / "cm_linearsvc.png",
}

TOP_TOKENS_FILES = {
    "Logistic Regression": FIGURES_DIR / "top_tokens_logreg.png",
    "Linear SVC": FIGURES_DIR / "top_tokens_linearsvc.png",
}

TOP_TOKENS_CSV_FILES = {
    "Logistic Regression": REPORTS_DIR / "top_tokens_logreg.csv",
    "Linear SVC": REPORTS_DIR / "top_tokens_linearsvc.csv",
}

MODEL_COMPARISON_CSV = REPORTS_DIR / "model_comparison.csv"
MODEL_COMPARISON_MD = REPORTS_DIR / "model_comparison.md"
METRICS_COMPARISON_FIG = FIGURES_DIR / "model_metrics.png"
PER_CLASS_FIG = FIGURES_DIR / "per_class_f1.png"

MODEL_ARTIFACT_PATHS = {
    "logreg_model": MODELS_DIR / "logreg_model.pkl",
    "linearsvc_model": MODELS_DIR / "linearsvc_model.pkl",
    "tfidf_vectorizer": MODELS_DIR / "tfidf_vectorizer.pkl",
    "ticker_encoder": MODELS_DIR / "ticker_encoder.pkl",
}


