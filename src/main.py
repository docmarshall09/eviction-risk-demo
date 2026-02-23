"""CLI entry point for monthly and yearly eviction-risk model workflows."""

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import subprocess

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import (
    EVICTION_LAB_BACKTEST_LAST_TWO_YEARS_PATH,
    EVICTION_LAB_BACKTEST_LAST_YEAR_PATH,
    EVICTION_LAB_BACKTEST_SUMMARY_PATH,
    EVICTION_LAB_HOLDOUT_DETAIL_LAST_TWO_YEARS_PATH,
    EVICTION_LAB_HOLDOUT_DETAIL_LAST_YEAR_PATH,
    EVICTION_LAB_YEARLY_FEATURE_TABLE_PATH,
    EVICTION_LAB_YEARLY_LATEST_SCORES_PATH,
    EVICTION_LAB_YEARLY_MODEL_METADATA_PATH,
    EVICTION_LAB_YEARLY_METRICS_PATH,
    EVICTION_LAB_YEARLY_MODEL_PATH,
    FEATURE_TABLE_PATH,
    LATEST_SCORES_PATH,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    PROCESSED_DIR,
    RAW_EVICTION_LAB_YEARLY_PATH,
    RAW_EVICTION_PATH,
    REPORTS_DIR,
)
from src.datasets.eviction import load_raw_eviction_data, validate_and_clean_eviction_df
from src.datasets.eviction_lab_yearly import (
    clean_eviction_lab_yearly,
    load_eviction_lab_yearly,
)
from src.features.eviction_features import build_feature_frame
from src.features.eviction_lab_yearly_features import build_eviction_lab_yearly_features
from src.models.eviction_lab_yearly_model import (
    MODEL_FEATURE_COLUMNS,
    build_holdout_detail,
    evaluate_at_top_quartile,
    evaluate_eviction_lab_yearly_model,
    load_eviction_lab_yearly_model,
    save_eviction_lab_yearly_model,
    score_counties_yearly,
    score_latest_year,
    split_by_outcome_year,
    split_train_test_by_year,
    train_eviction_lab_yearly_model,
)
from src.models.eviction_risk_model import (
    evaluate_model,
    load_model,
    save_model,
    score_counties,
    split_train_test_by_time,
    train_model,
)
from src.reporting.eviction_lab_backtest_report import write_backtest_summary_report


LOGGER = logging.getLogger(__name__)
VALID_TASKS = [
    "train_eviction_model",
    "score_latest",
    "score_county",
    "train_eviction_lab_yearly",
    "score_eviction_lab_latest_year",
    "score_eviction_lab_county",
    "backtest_eviction_lab_yearly",
    "train_eviction_lab_yearly_final",
    "report_eviction_lab_backtest",
    "serve_api",
]


def _configure_logging() -> None:
    """Set up console logging for clear progress and validation messages."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def _normalize_fips_input(fips_value: str) -> str:
    """Normalize user-provided FIPS value to a 5-digit string."""
    digits = "".join(character for character in fips_value.strip() if character.isdigit())
    if digits == "" or len(digits) > 5:
        raise ValueError("FIPS must contain 1 to 5 digits.")
    return digits.zfill(5)


def _ensure_output_directories() -> None:
    """Create output directories used by training and scoring workflows."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _standardize_monthly_feature_df(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure monthly feature table dtypes are stable when loaded from CSV."""
    standardized_df = feature_df.copy()
    standardized_df["month"] = pd.to_datetime(standardized_df["month"])
    standardized_df["county_fips"] = (
        standardized_df["county_fips"].astype(str).str.strip().str.zfill(5)
    )
    return standardized_df


def _standardize_yearly_feature_df(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure yearly feature table dtypes are stable when loaded from CSV."""
    standardized_df = feature_df.copy()
    standardized_df["county_fips"] = (
        standardized_df["county_fips"].astype(str).str.strip().str.zfill(5)
    )
    standardized_df["year"] = pd.to_numeric(standardized_df["year"], errors="coerce")
    standardized_df["year"] = standardized_df["year"].astype(int)
    standardized_df["outcome_year"] = pd.to_numeric(
        standardized_df["outcome_year"], errors="coerce"
    )
    standardized_df["outcome_year"] = standardized_df["outcome_year"].astype(int)

    standardized_df["sample_weight"] = pd.to_numeric(
        standardized_df["sample_weight"],
        errors="coerce",
    ).fillna(1.0)

    standardized_df["y"] = pd.to_numeric(
        standardized_df["y"],
        errors="coerce",
    ).astype("Int64")
    return standardized_df


def _get_labeled_yearly_rows(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Return rows that are fully labeled and model-ready for yearly training."""
    required_columns = MODEL_FEATURE_COLUMNS + ["y", "sample_weight", "outcome_year"]
    labeled_df = feature_df.dropna(subset=required_columns).copy()
    labeled_df["y"] = labeled_df["y"].astype(int)
    return labeled_df


def _load_and_clean_raw_data() -> pd.DataFrame:
    """Load monthly raw CSV and return a validated clean dataframe."""
    raw_df = load_raw_eviction_data(str(RAW_EVICTION_PATH))
    clean_df = validate_and_clean_eviction_df(raw_df)
    return clean_df


def _load_and_clean_eviction_lab_yearly_data() -> pd.DataFrame:
    """Load yearly raw CSV and return a validated clean dataframe."""
    raw_df = load_eviction_lab_yearly(str(RAW_EVICTION_LAB_YEARLY_PATH))
    clean_df = clean_eviction_lab_yearly(raw_df)
    return clean_df


def _load_or_build_feature_table() -> pd.DataFrame:
    """Load monthly feature table if available, else rebuild from raw data."""
    if FEATURE_TABLE_PATH.exists():
        feature_df = pd.read_csv(FEATURE_TABLE_PATH)
        return _standardize_monthly_feature_df(feature_df)

    LOGGER.info(
        "Monthly feature table not found at %s. Rebuilding from raw data.",
        FEATURE_TABLE_PATH,
    )
    clean_df = _load_and_clean_raw_data()
    feature_df = build_feature_frame(clean_df)
    feature_df.to_csv(FEATURE_TABLE_PATH, index=False)
    return _standardize_monthly_feature_df(feature_df)


def _load_or_build_yearly_feature_table() -> pd.DataFrame:
    """Load yearly feature table if available, else rebuild from raw data."""
    if EVICTION_LAB_YEARLY_FEATURE_TABLE_PATH.exists():
        feature_df = pd.read_csv(EVICTION_LAB_YEARLY_FEATURE_TABLE_PATH)
        if "outcome_year" in feature_df.columns:
            return _standardize_yearly_feature_df(feature_df)

        LOGGER.info(
            "Yearly feature table at %s is missing outcome_year. "
            "Adding compatibility column from feature year.",
            EVICTION_LAB_YEARLY_FEATURE_TABLE_PATH,
        )
        feature_df["outcome_year"] = pd.to_numeric(
            feature_df["year"], errors="coerce"
        ) + 1
        feature_df.to_csv(EVICTION_LAB_YEARLY_FEATURE_TABLE_PATH, index=False)
        return _standardize_yearly_feature_df(feature_df)

    LOGGER.info(
        "Yearly feature table not found at %s. Rebuilding from raw data.",
        EVICTION_LAB_YEARLY_FEATURE_TABLE_PATH,
    )
    clean_df = _load_and_clean_eviction_lab_yearly_data()
    feature_df = build_eviction_lab_yearly_features(clean_df)
    feature_df.to_csv(EVICTION_LAB_YEARLY_FEATURE_TABLE_PATH, index=False)
    return _standardize_yearly_feature_df(feature_df)


def _write_json_report(report_path: Path, payload: dict) -> None:
    """Write a JSON payload to a report path with UTF-8 encoding."""
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(payload, report_file, indent=2)


def _read_json_report_if_exists(report_path: Path) -> dict | None:
    """Read a JSON report if present, otherwise return None."""
    if not report_path.exists():
        return None

    with report_path.open("r", encoding="utf-8") as report_file:
        return json.load(report_file)


def _get_git_short_hash() -> str | None:
    """Return the current git short hash, or None when unavailable."""
    try:
        command = ["git", "rev-parse", "--short", "HEAD"]
        output = subprocess.check_output(command, cwd=Path(__file__).resolve().parent.parent)
        git_hash = output.decode("utf-8").strip()
        if git_hash == "":
            return None
        return git_hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _build_model_version() -> str:
    """Build model version string from UTC timestamp and optional git hash."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    git_hash = _get_git_short_hash()
    if git_hash is None:
        return timestamp
    return f"{timestamp}_{git_hash}"


def _summarize_metrics_for_metadata(metrics_payload: dict | None) -> dict | None:
    """Create a compact metrics summary for metadata artifacts."""
    if metrics_payload is None:
        return None

    summary_keys = [
        "auc",
        "precision_at_0_5",
        "recall_at_0_5",
        "test_rows",
        "test_year_start",
        "test_year_end",
    ]

    summary: dict = {}
    for key in summary_keys:
        if key in metrics_payload:
            summary[key] = metrics_payload[key]

    if "overall" in metrics_payload:
        summary["overall"] = metrics_payload["overall"]

    if not summary:
        return None
    return summary


def _build_yearly_model_metadata_payload(
    training_df: pd.DataFrame,
    metrics_payload: dict | None,
) -> dict:
    """Build yearly model metadata payload for API docs and scoring responses."""
    model_version = _build_model_version()
    training_feature_years = training_df["year"]
    outcome_years = training_df["outcome_year"]

    return {
        "model_version": model_version,
        "model_type": "eviction_lab_yearly_logreg",
        "trained_on_dataset_name": "county_proprietary_valid_2000_2018.csv",
        "training_feature_year_range": {
            "min_year": int(training_feature_years.min()),
            "max_year": int(training_feature_years.max()),
        },
        "training_years": {
            "min_year": int(training_feature_years.min()),
            "max_year": int(training_feature_years.max()),
        },
        "outcome_year_range": {
            "min_year": int(outcome_years.min()),
            "max_year": int(outcome_years.max()),
        },
        "label_definition": "Top-quartile filing_rate in year t+1 across counties.",
        "feature_list": MODEL_FEATURE_COLUMNS,
        "metrics_summary": _summarize_metrics_for_metadata(metrics_payload),
        "limitations": [
            "Dataset ends at 2018 outcome year.",
            "Irregular county-year panel without gap filling.",
            "Scores depend on available county-year feature rows.",
        ],
    }


def _write_yearly_model_metadata(training_df: pd.DataFrame, metrics_payload: dict | None) -> None:
    """Write yearly model metadata artifact used by API metadata endpoint."""
    metadata_payload = _build_yearly_model_metadata_payload(training_df, metrics_payload)
    _write_json_report(EVICTION_LAB_YEARLY_MODEL_METADATA_PATH, metadata_payload)


def run_train_eviction_model() -> None:
    """Train monthly model, evaluate with a time split, and write artifacts."""
    _ensure_output_directories()

    clean_df = _load_and_clean_raw_data()
    feature_df = build_feature_frame(clean_df)

    if feature_df.empty:
        raise ValueError(
            "Feature table is empty after preprocessing. Check input data coverage."
        )

    train_df, test_df = split_train_test_by_time(feature_df)
    model = train_model(train_df)
    metrics = evaluate_model(model, test_df)

    feature_df.to_csv(FEATURE_TABLE_PATH, index=False)
    save_model(model, str(MODEL_PATH))
    _write_json_report(METRICS_PATH, metrics)

    LOGGER.info("Wrote monthly feature table to %s", FEATURE_TABLE_PATH)
    LOGGER.info("Saved monthly model to %s", MODEL_PATH)
    LOGGER.info("Saved monthly metrics to %s", METRICS_PATH)


def run_train_eviction_lab_yearly() -> None:
    """Train yearly model with feature-year split and write artifacts."""
    _ensure_output_directories()

    feature_df = _load_or_build_yearly_feature_table()
    labeled_df = _get_labeled_yearly_rows(feature_df)

    if labeled_df.empty:
        raise ValueError(
            "No labeled yearly rows available after preprocessing. Check input data coverage."
        )

    train_df, test_df = split_train_test_by_year(labeled_df)
    model = train_eviction_lab_yearly_model(train_df)
    metrics = evaluate_eviction_lab_yearly_model(model, test_df)

    save_eviction_lab_yearly_model(model, str(EVICTION_LAB_YEARLY_MODEL_PATH))
    _write_json_report(EVICTION_LAB_YEARLY_METRICS_PATH, metrics)
    _write_yearly_model_metadata(training_df=train_df, metrics_payload=metrics)

    LOGGER.info("Saved yearly model to %s", EVICTION_LAB_YEARLY_MODEL_PATH)
    LOGGER.info("Saved yearly metrics to %s", EVICTION_LAB_YEARLY_METRICS_PATH)
    LOGGER.info(
        "Saved yearly model metadata to %s",
        EVICTION_LAB_YEARLY_MODEL_METADATA_PATH,
    )


def _run_outcome_year_backtest(
    labeled_feature_df: pd.DataFrame,
    holdout_outcome_years: list[int],
    metrics_path: Path,
    detail_path: Path,
) -> None:
    """Train/test one outcome-year holdout configuration and write artifacts."""
    train_df, test_df = split_by_outcome_year(
        labeled_feature_df,
        holdout_outcome_years=holdout_outcome_years,
    )
    model = train_eviction_lab_yearly_model(train_df)

    metrics = evaluate_at_top_quartile(model, test_df)
    detail_df = build_holdout_detail(model, test_df)

    _write_json_report(metrics_path, metrics)
    detail_df.to_csv(detail_path, index=False)

    LOGGER.info("Wrote backtest metrics to %s", metrics_path)
    LOGGER.info("Wrote backtest detail to %s", detail_path)


def run_backtest_eviction_lab_yearly() -> None:
    """Run leakage-safe backtests using last 1 and last 2 outcome years."""
    _ensure_output_directories()

    feature_df = _load_or_build_yearly_feature_table()
    labeled_df = _get_labeled_yearly_rows(feature_df)

    outcome_years = sorted(labeled_df["outcome_year"].unique().tolist())
    if not outcome_years:
        raise ValueError("No labeled yearly rows found for backtesting.")

    last_outcome_year = int(outcome_years[-1])
    _run_outcome_year_backtest(
        labeled_feature_df=labeled_df,
        holdout_outcome_years=[last_outcome_year],
        metrics_path=EVICTION_LAB_BACKTEST_LAST_YEAR_PATH,
        detail_path=EVICTION_LAB_HOLDOUT_DETAIL_LAST_YEAR_PATH,
    )

    if len(outcome_years) < 2:
        raise ValueError("Need at least two outcome years for last-2-years backtest.")

    last_two_outcome_years = [int(outcome_years[-2]), int(outcome_years[-1])]
    _run_outcome_year_backtest(
        labeled_feature_df=labeled_df,
        holdout_outcome_years=last_two_outcome_years,
        metrics_path=EVICTION_LAB_BACKTEST_LAST_TWO_YEARS_PATH,
        detail_path=EVICTION_LAB_HOLDOUT_DETAIL_LAST_TWO_YEARS_PATH,
    )


def run_train_eviction_lab_yearly_final() -> None:
    """Train yearly model on all labeled rows and overwrite the model artifact."""
    _ensure_output_directories()

    feature_df = _load_or_build_yearly_feature_table()
    labeled_df = _get_labeled_yearly_rows(feature_df)

    if labeled_df.empty:
        raise ValueError("No labeled yearly rows available for final retraining.")

    model = train_eviction_lab_yearly_model(labeled_df)
    save_eviction_lab_yearly_model(model, str(EVICTION_LAB_YEARLY_MODEL_PATH))
    prior_metrics = _read_json_report_if_exists(EVICTION_LAB_YEARLY_METRICS_PATH)
    _write_yearly_model_metadata(training_df=labeled_df, metrics_payload=prior_metrics)

    LOGGER.info("Trained final yearly model on %d labeled rows.", len(labeled_df))
    LOGGER.info("Saved final yearly model to %s", EVICTION_LAB_YEARLY_MODEL_PATH)
    LOGGER.info(
        "Saved yearly model metadata to %s",
        EVICTION_LAB_YEARLY_MODEL_METADATA_PATH,
    )


def run_report_eviction_lab_backtest() -> None:
    """Run yearly backtests and write a human-readable markdown summary report."""
    _ensure_output_directories()

    # Always refresh artifacts so the markdown report reflects current behavior.
    run_backtest_eviction_lab_yearly()
    write_backtest_summary_report(EVICTION_LAB_BACKTEST_SUMMARY_PATH)

    LOGGER.info("Wrote backtest summary report to %s", EVICTION_LAB_BACKTEST_SUMMARY_PATH)


def run_serve_api() -> None:
    """Run the FastAPI app with uvicorn for local development."""
    port_text = os.getenv("PORT", "8000").strip()
    if not port_text.isdigit():
        raise ValueError("PORT must be an integer value.")

    port = int(port_text)
    if port <= 0 or port > 65535:
        raise ValueError("PORT must be between 1 and 65535.")

    import uvicorn

    LOGGER.info("Starting API server on http://127.0.0.1:%d", port)
    uvicorn.run("src.api.app:app", host="127.0.0.1", port=port)


def _load_monthly_model_or_raise(model_path: Path) -> Pipeline:
    """Load monthly model and raise a clear error when missing."""
    if not model_path.exists():
        raise FileNotFoundError(
            "Monthly model file not found at "
            f"{model_path}. Run '--task train_eviction_model' first."
        )
    return load_model(str(model_path))


def _load_yearly_model_or_raise(model_path: Path) -> LogisticRegression:
    """Load yearly model and raise a clear error when missing."""
    if not model_path.exists():
        raise FileNotFoundError(
            "Yearly model file not found at "
            f"{model_path}. Run '--task train_eviction_lab_yearly' or "
            "'--task train_eviction_lab_yearly_final' first."
        )
    return load_eviction_lab_yearly_model(str(model_path))


def run_score_latest() -> None:
    """Score all counties for the latest month and write a report CSV."""
    _ensure_output_directories()

    model = _load_monthly_model_or_raise(MODEL_PATH)
    feature_df = _load_or_build_feature_table()

    latest_month = feature_df["month"].max()
    latest_df = feature_df[feature_df["month"] == latest_month].copy()

    if latest_df.empty:
        raise ValueError("No rows available for the latest month in feature table.")

    scores_df = score_counties(model, latest_df)
    scores_df.to_csv(LATEST_SCORES_PATH, index=False)

    LOGGER.info("Latest month scored: %s", latest_month.date())
    LOGGER.info("Wrote monthly county scores to %s", LATEST_SCORES_PATH)


def run_score_county(fips: str) -> None:
    """Print the latest available monthly risk score for a single county."""
    model = _load_monthly_model_or_raise(MODEL_PATH)
    feature_df = _load_or_build_feature_table()

    normalized_fips = _normalize_fips_input(fips)
    county_df = feature_df[
        feature_df["county_fips"].astype(str) == normalized_fips
    ].copy()

    if county_df.empty:
        raise ValueError(
            f"No monthly feature rows found for county_fips={normalized_fips}. "
            "Check that the county exists in the input data."
        )

    latest_month = county_df["month"].max()
    latest_county_row = county_df[county_df["month"] == latest_month].copy()

    scores_df = score_counties(model, latest_county_row)
    risk_score = float(scores_df.iloc[0]["risk_score"])

    print(
        "county_fips="
        f"{normalized_fips}, month={latest_month.date()}, risk_score={risk_score:.6f}"
    )


def run_score_eviction_lab_latest_year() -> None:
    """Score all counties for the latest yearly observation and write CSV."""
    _ensure_output_directories()

    model = _load_yearly_model_or_raise(EVICTION_LAB_YEARLY_MODEL_PATH)
    feature_df = _load_or_build_yearly_feature_table()

    scores_df = score_latest_year(model, feature_df)
    if scores_df.empty:
        raise ValueError("No scoreable rows found in the latest year for yearly pipeline.")

    scores_df.to_csv(EVICTION_LAB_YEARLY_LATEST_SCORES_PATH, index=False)

    latest_year = int(scores_df["year"].max())
    LOGGER.info("Latest yearly panel scored: %d", latest_year)
    LOGGER.info(
        "Wrote yearly county scores to %s",
        EVICTION_LAB_YEARLY_LATEST_SCORES_PATH,
    )


def run_score_eviction_lab_county(fips: str) -> None:
    """Print the latest available yearly risk score for a single county."""
    model = _load_yearly_model_or_raise(EVICTION_LAB_YEARLY_MODEL_PATH)
    feature_df = _load_or_build_yearly_feature_table()

    normalized_fips = _normalize_fips_input(fips)
    county_df = feature_df[
        feature_df["county_fips"].astype(str) == normalized_fips
    ].copy()

    if county_df.empty:
        raise ValueError(
            f"No yearly feature rows found for county_fips={normalized_fips}. "
            "Check that the county exists in the yearly input data."
        )

    latest_year = int(county_df["year"].max())
    latest_county_row = county_df[county_df["year"] == latest_year].copy()

    scores_df = score_counties_yearly(model, latest_county_row)
    if scores_df.empty:
        raise ValueError(
            f"County {normalized_fips} has no scoreable latest-year row. "
            "This usually means not enough observed history for lag features."
        )

    risk_score = float(scores_df.iloc[0]["risk_score"])
    print(
        "county_fips="
        f"{normalized_fips}, year={latest_year}, risk_score={risk_score:.6f}"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the CLI."""
    parser = argparse.ArgumentParser(description="Eviction-risk POC CLI")
    parser.add_argument(
        "--task",
        choices=VALID_TASKS,
        required=True,
        help="Task to run.",
    )
    parser.add_argument(
        "--fips",
        default=None,
        help="County FIPS for county-scoring tasks.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the CLI task selected by the user."""
    _configure_logging()
    args = parse_args()

    try:
        if args.task == "train_eviction_model":
            run_train_eviction_model()
            return

        if args.task == "score_latest":
            run_score_latest()
            return

        if args.task == "score_county":
            if args.fips is None:
                raise ValueError("'--fips' is required when task is score_county.")
            run_score_county(args.fips)
            return

        if args.task == "train_eviction_lab_yearly":
            run_train_eviction_lab_yearly()
            return

        if args.task == "score_eviction_lab_latest_year":
            run_score_eviction_lab_latest_year()
            return

        if args.task == "score_eviction_lab_county":
            if args.fips is None:
                raise ValueError(
                    "'--fips' is required when task is score_eviction_lab_county."
                )
            run_score_eviction_lab_county(args.fips)
            return

        if args.task == "backtest_eviction_lab_yearly":
            run_backtest_eviction_lab_yearly()
            return

        if args.task == "train_eviction_lab_yearly_final":
            run_train_eviction_lab_yearly_final()
            return

        if args.task == "report_eviction_lab_backtest":
            run_report_eviction_lab_backtest()
            return

        if args.task == "serve_api":
            run_serve_api()
            return

    except FileNotFoundError as error:
        print(
            f"Error: {error}\n"
            "Ensure required raw files exist:\n"
            "- data/raw/eviction_county_month.csv (monthly)\n"
            "- data/raw/county_proprietary_valid_2000_2018.csv (yearly)"
        )
        raise SystemExit(1) from error
    except ValueError as error:
        print(f"Error: {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
