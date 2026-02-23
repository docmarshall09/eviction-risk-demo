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

RAW_EVICTION_LAB_YEARLY_PATH: Path = RAW_DIR / "county_proprietary_valid_2000_2018.csv"
EVICTION_LAB_YEARLY_FEATURE_TABLE_PATH: Path = (
    PROCESSED_DIR / "eviction_lab_yearly_features.csv"
)
EVICTION_LAB_YEARLY_MODEL_PATH: Path = MODELS_DIR / "eviction_lab_yearly_model.joblib"
EVICTION_LAB_YEARLY_MODEL_METADATA_PATH: Path = (
    MODELS_DIR / "eviction_lab_yearly_model_metadata.json"
)
EVICTION_LAB_YEARLY_METRICS_PATH: Path = (
    REPORTS_DIR / "eviction_lab_yearly_model_metrics.json"
)
EVICTION_LAB_YEARLY_LATEST_SCORES_PATH: Path = (
    REPORTS_DIR / "eviction_lab_county_risk_scores_latest_year.csv"
)
EVICTION_LAB_BACKTEST_LAST_YEAR_PATH: Path = (
    REPORTS_DIR / "eviction_lab_yearly_backtest_last_year.json"
)
EVICTION_LAB_BACKTEST_LAST_TWO_YEARS_PATH: Path = (
    REPORTS_DIR / "eviction_lab_yearly_backtest_last_2_years.json"
)
EVICTION_LAB_HOLDOUT_DETAIL_LAST_YEAR_PATH: Path = (
    REPORTS_DIR / "eviction_lab_yearly_holdout_detail_last_year.csv"
)
EVICTION_LAB_HOLDOUT_DETAIL_LAST_TWO_YEARS_PATH: Path = (
    REPORTS_DIR / "eviction_lab_yearly_holdout_detail_last_2_years.csv"
)
EVICTION_LAB_BACKTEST_SUMMARY_PATH: Path = REPORTS_DIR / "backtest_summary.md"
