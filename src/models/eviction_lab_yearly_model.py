"""Training and scoring utilities for the Eviction Lab yearly risk model."""

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


def split_train_test_by_year(
    feature_df: pd.DataFrame,
    test_years: int = TEST_YEARS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test using the most recent years as test."""
    unique_years = sorted(feature_df["year"].unique().tolist())
    if len(unique_years) <= test_years:
        raise ValueError(
            "Not enough unique years for a time split. "
            f"Need more than {test_years}, found {len(unique_years)}."
        )

    held_out_years = set(unique_years[-test_years:])
    train_df = feature_df[~feature_df["year"].isin(held_out_years)].copy()
    test_df = feature_df[feature_df["year"].isin(held_out_years)].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Time split produced an empty train or test set.")

    return train_df, test_df


def train_eviction_lab_yearly_model(train_df: pd.DataFrame) -> LogisticRegression:
    """Train a logistic regression model using sample weights."""
    unique_classes = sorted(train_df["y"].unique().tolist())
    if len(unique_classes) < 2:
        raise ValueError(
            "Yearly training data must contain both classes 0 and 1. "
            f"Found classes: {unique_classes}."
        )

    model = LogisticRegression(max_iter=1000)
    model.fit(
        train_df[MODEL_FEATURE_COLUMNS],
        train_df["y"],
        sample_weight=train_df["sample_weight"],
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
    """Evaluate yearly model on holdout years."""
    y_true = test_df["y"]
    y_prob = model.predict_proba(test_df[MODEL_FEATURE_COLUMNS])[:, 1]
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
        "test_rows": int(len(test_df)),
        "test_year_start": int(test_df["year"].min()),
        "test_year_end": int(test_df["year"].max()),
        "auc": auc,
        "precision_at_0_5": precision,
        "recall_at_0_5": recall,
        "calibration_by_decile": calibration_summary,
        "note": "AUC is null when the test set contains a single class.",
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
    risk_scores = model.predict_proba(feature_df[MODEL_FEATURE_COLUMNS])[:, 1]
    output_df = feature_df[["county_fips", "year"]].copy()
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
