"""Training, backtesting, and scoring utilities for the yearly risk model."""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score


MODEL_FEATURE_COLUMNS = [
    "lag_1",
    "lag_3_mean_obs",
    "lag_5_mean_obs",
    "years_since_last_obs",
]
PROBABILITY_THRESHOLD = 0.5
TEST_YEARS = 2
CALIBRATION_DECILES = 10
DEFAULT_LOGISTIC_C = 0.3
LOGISTIC_C_GRID = [0.1, 0.3, 1.0]
MIN_ROWS_FOR_ISOTONIC = 1000
MIN_CLASS_ROWS_FOR_ISOTONIC = 200


def _get_labeled_rows(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Return rows that have valid labels and complete model features."""
    required_columns = MODEL_FEATURE_COLUMNS + ["y", "sample_weight"]
    labeled_df = feature_df.dropna(subset=required_columns).copy()
    labeled_df["y"] = labeled_df["y"].astype(int)
    return labeled_df


def _time_column_for_training(df: pd.DataFrame) -> str:
    """Return the preferred time column for train/calibration splits."""
    if "outcome_year" in df.columns:
        return "outcome_year"
    return "year"


def _split_fit_and_calibration_by_time(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split training rows into fit and calibration subsets by latest time bucket."""
    time_column = _time_column_for_training(train_df)
    unique_time_values = sorted(train_df[time_column].unique().tolist())

    if len(unique_time_values) < 2:
        raise ValueError(
            "Need at least two time buckets for leakage-safe calibration. "
            f"Found {len(unique_time_values)} in column {time_column}."
        )

    calibration_time_value = unique_time_values[-1]
    fit_df = train_df[train_df[time_column] < calibration_time_value].copy()
    calibration_df = train_df[train_df[time_column] == calibration_time_value].copy()

    if fit_df.empty or calibration_df.empty:
        raise ValueError("Failed to build non-empty fit/calibration split.")

    return fit_df, calibration_df


def _safe_auc(y_true: pd.Series, score: pd.Series) -> Optional[float]:
    """Compute AUC only when both classes are present."""
    if y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, score))


def _select_regularization_c(fit_df: pd.DataFrame, calibration_df: pd.DataFrame) -> float:
    """Select logistic C using calibration AUC, fallback to conservative default."""
    calibration_auc_available = calibration_df["y"].nunique() >= 2
    if not calibration_auc_available:
        return DEFAULT_LOGISTIC_C

    best_c = DEFAULT_LOGISTIC_C
    best_auc = float("-inf")

    for candidate_c in LOGISTIC_C_GRID:
        model = LogisticRegression(C=candidate_c, max_iter=1000)
        model.fit(
            fit_df[MODEL_FEATURE_COLUMNS],
            fit_df["y"],
            sample_weight=fit_df["sample_weight"],
        )

        candidate_score = pd.Series(
            model.predict_proba(calibration_df[MODEL_FEATURE_COLUMNS])[:, 1]
        )
        candidate_auc = _safe_auc(calibration_df["y"], candidate_score)

        if candidate_auc is None:
            continue

        # Break ties toward lower C for more conservative coefficients.
        if candidate_auc > best_auc or (
            math.isclose(candidate_auc, best_auc, rel_tol=1e-12)
            and candidate_c < best_c
        ):
            best_auc = candidate_auc
            best_c = candidate_c

    return float(best_c)


def _select_calibration_method(calibration_df: pd.DataFrame) -> str:
    """Choose isotonic when data is large enough, else sigmoid."""
    class_counts = calibration_df["y"].value_counts()
    if len(class_counts) < 2:
        return "sigmoid"

    if (
        len(calibration_df) >= MIN_ROWS_FOR_ISOTONIC
        and int(class_counts.min()) >= MIN_CLASS_ROWS_FOR_ISOTONIC
    ):
        return "isotonic"

    return "sigmoid"


def _build_prefit_calibrator(base_model: LogisticRegression, method: str) -> CalibratedClassifierCV:
    """Create a prefit calibrator compatible across sklearn versions."""
    try:
        from sklearn.frozen import FrozenEstimator

        frozen_estimator = FrozenEstimator(base_model)
        return CalibratedClassifierCV(estimator=frozen_estimator, method=method)
    except (ImportError, ModuleNotFoundError):
        pass

    try:
        return CalibratedClassifierCV(estimator=base_model, method=method, cv="prefit")
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base_model, method=method, cv="prefit")


def split_train_test_by_year(
    feature_df: pd.DataFrame,
    test_years: int = TEST_YEARS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split labeled data by feature year using latest years as holdout."""
    labeled_df = _get_labeled_rows(feature_df)
    unique_years = sorted(labeled_df["year"].unique().tolist())
    if len(unique_years) <= test_years:
        raise ValueError(
            "Not enough unique years for a time split. "
            f"Need more than {test_years}, found {len(unique_years)}."
        )

    held_out_years = set(unique_years[-test_years:])
    train_df = labeled_df[~labeled_df["year"].isin(held_out_years)].copy()
    test_df = labeled_df[labeled_df["year"].isin(held_out_years)].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Time split produced an empty train or test set.")

    return train_df, test_df


def split_by_outcome_year(
    feature_df: pd.DataFrame,
    holdout_outcome_years: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split labeled rows by outcome_year for leakage-safe backtesting.

    Args:
        feature_df: Yearly feature dataframe including outcome_year and y.
        holdout_outcome_years: Outcome years to reserve for testing.

    Returns:
        Tuple of (train_df, test_df).
    """
    if not holdout_outcome_years:
        raise ValueError("holdout_outcome_years must contain at least one year.")

    labeled_df = _get_labeled_rows(feature_df)
    if "outcome_year" not in labeled_df.columns:
        raise ValueError("feature_df must include an 'outcome_year' column.")

    holdout_set = set(int(year) for year in holdout_outcome_years)
    train_df = labeled_df[~labeled_df["outcome_year"].isin(holdout_set)].copy()
    test_df = labeled_df[labeled_df["outcome_year"].isin(holdout_set)].copy()

    if train_df.empty:
        raise ValueError("Outcome-year split produced an empty training set.")
    if test_df.empty:
        raise ValueError("Outcome-year split produced an empty test set.")

    return train_df, test_df


def train_eviction_lab_yearly_model(train_df: pd.DataFrame) -> Any:
    """Train a calibrated logistic regression model with conservative regularization."""
    labeled_train_df = _get_labeled_rows(train_df)
    unique_classes = sorted(labeled_train_df["y"].unique().tolist())
    if len(unique_classes) < 2:
        raise ValueError(
            "Yearly training data must contain both classes 0 and 1. "
            f"Found classes: {unique_classes}."
        )

    fit_df, calibration_df = _split_fit_and_calibration_by_time(labeled_train_df)
    chosen_c = _select_regularization_c(fit_df=fit_df, calibration_df=calibration_df)

    base_model = LogisticRegression(C=chosen_c, max_iter=1000)
    base_model.fit(
        fit_df[MODEL_FEATURE_COLUMNS],
        fit_df["y"],
        sample_weight=fit_df["sample_weight"],
    )

    calibration_method = _select_calibration_method(calibration_df)
    calibrated_model = _build_prefit_calibrator(base_model, method=calibration_method)
    calibrated_model.fit(
        calibration_df[MODEL_FEATURE_COLUMNS],
        calibration_df["y"],
    )

    # Store training details on the model for metadata/reporting.
    setattr(calibrated_model, "chosen_regularization_c", float(chosen_c))
    setattr(calibrated_model, "regularization_candidates", list(LOGISTIC_C_GRID))
    setattr(calibrated_model, "calibration_method", calibration_method)
    setattr(
        calibrated_model,
        "calibration_time_values",
        sorted(calibration_df[_time_column_for_training(calibration_df)].unique().tolist()),
    )

    return calibrated_model


def get_model_training_details(model: Any) -> Dict[str, Any]:
    """Extract model-training details stored on trained model objects."""
    return {
        "chosen_regularization_c": float(
            getattr(model, "chosen_regularization_c", DEFAULT_LOGISTIC_C)
        ),
        "regularization_candidates": list(
            getattr(model, "regularization_candidates", LOGISTIC_C_GRID)
        ),
        "calibration_method": str(getattr(model, "calibration_method", "sigmoid")),
        "calibration_time_values": list(getattr(model, "calibration_time_values", [])),
    }


def _build_calibration_summary(
    y_true: pd.Series,
    y_prob: pd.Series,
) -> List[Dict[str, Any]]:
    """Summarize calibration quality by probability deciles."""
    calibration_df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).copy()

    unique_probability_count = calibration_df["y_prob"].nunique(dropna=True)
    if unique_probability_count < 2:
        return [
            {
                "decile": 0,
                "count": int(len(calibration_df)),
                "mean_pred": float(calibration_df["y_prob"].mean()),
                "observed_rate": float(calibration_df["y_true"].mean()),
            }
        ]

    calibration_df["decile"] = pd.qcut(
        calibration_df["y_prob"],
        q=CALIBRATION_DECILES,
        labels=False,
        duplicates="drop",
    )

    grouped = calibration_df.groupby("decile", dropna=True)
    summary_df = grouped.agg(
        count=("y_true", "size"),
        mean_pred=("y_prob", "mean"),
        observed_rate=("y_true", "mean"),
    )

    summary: List[Dict[str, Any]] = []
    for decile, row in summary_df.iterrows():
        summary.append(
            {
                "decile": int(decile),
                "count": int(row["count"]),
                "mean_pred": float(row["mean_pred"]),
                "observed_rate": float(row["observed_rate"]),
            }
        )
    return summary


def evaluate_eviction_lab_yearly_model(
    model: Any,
    test_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Evaluate yearly model on holdout years with threshold 0.5 metrics."""
    labeled_test_df = _get_labeled_rows(test_df)
    y_true = labeled_test_df["y"]
    y_prob = model.predict_proba(labeled_test_df[MODEL_FEATURE_COLUMNS])[:, 1]
    y_pred = (y_prob >= PROBABILITY_THRESHOLD).astype(int)

    unique_classes = sorted(y_true.unique().tolist())
    auc: Optional[float]
    if len(unique_classes) < 2:
        auc = None
    else:
        auc = float(roc_auc_score(y_true, y_prob))

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))

    calibration_summary = _build_calibration_summary(
        y_true=y_true,
        y_prob=pd.Series(y_prob),
    )

    return {
        "test_rows": int(len(labeled_test_df)),
        "test_year_start": int(labeled_test_df["year"].min()),
        "test_year_end": int(labeled_test_df["year"].max()),
        "auc": auc,
        "precision_at_0_5": precision,
        "recall_at_0_5": recall,
        "calibration_by_decile": calibration_summary,
        "model_training_details": get_model_training_details(model),
        "note": "AUC is null when the test set contains a single class.",
    }


def _flag_top_quartile_predictions(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Flag the top 25% highest-risk counties per outcome year."""
    flagged_df = detail_df.copy()
    flagged_df["predicted_top_quartile"] = 0

    for _, group_df in flagged_df.groupby("outcome_year"):
        top_count = max(1, int(math.ceil(0.25 * len(group_df))))
        top_index = group_df.sort_values("risk_score", ascending=False).head(top_count).index
        flagged_df.loc[top_index, "predicted_top_quartile"] = 1

    return flagged_df


def build_holdout_detail(
    model: Any,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create per-county holdout detail rows for backtesting reports."""
    labeled_test_df = _get_labeled_rows(test_df)
    detail_df = labeled_test_df[["county_fips", "year", "outcome_year", "y"]].copy()
    detail_df["risk_score"] = model.predict_proba(labeled_test_df[MODEL_FEATURE_COLUMNS])[:, 1]

    detail_df = _flag_top_quartile_predictions(detail_df)
    detail_df = detail_df[
        [
            "county_fips",
            "year",
            "outcome_year",
            "risk_score",
            "y",
            "predicted_top_quartile",
        ]
    ]
    detail_df = detail_df.sort_values(["outcome_year", "risk_score"], ascending=[True, False])
    return detail_df.reset_index(drop=True)


def _build_top_quartile_metrics_for_slice(slice_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute precision, recall, and optional AUC for one evaluation slice."""
    y_true = slice_df["y"].astype(int)
    y_pred = slice_df["predicted_top_quartile"].astype(int)

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))

    unique_classes = sorted(y_true.unique().tolist())
    auc: Optional[float]
    if len(unique_classes) < 2:
        auc = None
    else:
        auc = float(roc_auc_score(y_true, slice_df["risk_score"]))

    return {
        "rows": int(len(slice_df)),
        "positive_rate": float(y_true.mean()),
        "precision_at_top_quartile": precision,
        "recall_at_top_quartile": recall,
        "auc": auc,
    }


def evaluate_at_top_quartile(
    model: Any,
    test_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Evaluate model by top-quartile ranking within each outcome year."""
    detail_df = build_holdout_detail(model, test_df)
    holdout_years = sorted(detail_df["outcome_year"].astype(int).unique().tolist())

    per_year_metrics: List[Dict[str, Any]] = []
    for outcome_year in holdout_years:
        year_slice = detail_df[detail_df["outcome_year"] == outcome_year].copy()
        year_metrics = _build_top_quartile_metrics_for_slice(year_slice)
        year_metrics["outcome_year"] = int(outcome_year)
        per_year_metrics.append(year_metrics)

    pooled_metrics = _build_top_quartile_metrics_for_slice(detail_df)

    return {
        "holdout_outcome_years": holdout_years,
        "test_rows": int(len(detail_df)),
        "per_outcome_year": per_year_metrics,
        "overall": pooled_metrics,
        "model_training_details": get_model_training_details(model),
        "note": "AUC is null for slices with a single observed class.",
    }


def save_eviction_lab_yearly_model(model: Any, path: str) -> None:
    """Persist the trained yearly model to disk with joblib."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def load_eviction_lab_yearly_model(path: str) -> Any:
    """Load a trained yearly model from disk."""
    loaded_model = joblib.load(path)
    return loaded_model


def score_counties_yearly(
    model: Any,
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """Score county-year rows and return risk_score outputs."""
    scored_df = feature_df.dropna(subset=MODEL_FEATURE_COLUMNS).copy()
    risk_scores = model.predict_proba(scored_df[MODEL_FEATURE_COLUMNS])[:, 1]
    output_df = scored_df[["county_fips", "year"]].copy()
    output_df["risk_score"] = risk_scores
    return output_df


def score_latest_year(
    model: Any,
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """Score all counties for the latest year in the feature table."""
    latest_year = int(feature_df["year"].max())
    latest_df = feature_df[feature_df["year"] == latest_year].copy()
    return score_counties_yearly(model, latest_df)
