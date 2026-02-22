"""Training, backtesting, and scoring utilities for the yearly risk model."""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd
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


def _get_labeled_rows(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Return rows that have valid labels and complete model features."""
    required_columns = MODEL_FEATURE_COLUMNS + ["y", "sample_weight"]
    labeled_df = feature_df.dropna(subset=required_columns).copy()
    labeled_df["y"] = labeled_df["y"].astype(int)
    return labeled_df


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


def train_eviction_lab_yearly_model(train_df: pd.DataFrame) -> LogisticRegression:
    """Train a logistic regression model using sample weights."""
    labeled_train_df = _get_labeled_rows(train_df)
    unique_classes = sorted(labeled_train_df["y"].unique().tolist())
    if len(unique_classes) < 2:
        raise ValueError(
            "Yearly training data must contain both classes 0 and 1. "
            f"Found classes: {unique_classes}."
        )

    model = LogisticRegression(max_iter=1000)
    model.fit(
        labeled_train_df[MODEL_FEATURE_COLUMNS],
        labeled_train_df["y"],
        sample_weight=labeled_train_df["sample_weight"],
    )
    return model


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
    model: LogisticRegression,
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
        "note": "AUC is null when the test set contains a single class.",
    }


def _flag_top_quartile_predictions(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Flag the top 25% highest-risk counties per outcome year."""
    flagged_df = detail_df.copy()
    flagged_df["predicted_top_quartile"] = 0

    for outcome_year, group_df in flagged_df.groupby("outcome_year"):
        top_count = max(1, int(math.ceil(0.25 * len(group_df))))
        top_index = group_df.sort_values("risk_score", ascending=False).head(top_count).index
        flagged_df.loc[top_index, "predicted_top_quartile"] = 1

    return flagged_df


def build_holdout_detail(
    model: LogisticRegression,
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
    model: LogisticRegression,
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
        "note": "AUC is null for slices with a single observed class.",
    }


def save_eviction_lab_yearly_model(model: LogisticRegression, path: str) -> None:
    """Persist the trained yearly model to disk with joblib."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def load_eviction_lab_yearly_model(path: str) -> LogisticRegression:
    """Load a trained yearly model from disk."""
    loaded_model = joblib.load(path)
    return loaded_model


def score_counties_yearly(
    model: LogisticRegression,
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """Score county-year rows and return risk_score outputs."""
    scored_df = feature_df.dropna(subset=MODEL_FEATURE_COLUMNS).copy()
    risk_scores = model.predict_proba(scored_df[MODEL_FEATURE_COLUMNS])[:, 1]
    output_df = scored_df[["county_fips", "year"]].copy()
    output_df["risk_score"] = risk_scores
    return output_df


def score_latest_year(
    model: LogisticRegression,
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """Score all counties for the latest year in the feature table."""
    latest_year = int(feature_df["year"].max())
    latest_df = feature_df[feature_df["year"] == latest_year].copy()
    return score_counties_yearly(model, latest_df)
