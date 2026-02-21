"""Project configuration constants for paths and file locations."""

from pathlib import Path


PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
MODELS_DIR: Path = PROJECT_ROOT / "models"

RAW_EVICTION_PATH: Path = RAW_DIR / "eviction_county_month.csv"
FEATURE_TABLE_PATH: Path = PROCESSED_DIR / "eviction_features.csv"
MODEL_PATH: Path = MODELS_DIR / "eviction_risk_model.joblib"
METRICS_PATH: Path = REPORTS_DIR / "model_metrics.json"
LATEST_SCORES_PATH: Path = REPORTS_DIR / "county_risk_scores_latest.csv"
