"""Model training, evaluation, persistence, and scoring for eviction risk."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


MODEL_FEATURE_COLUMNS = ["lag_1", "lag_3_mean", "lag_12_mean", "month_of_year"]
NUMERIC_FEATURE_COLUMNS = ["lag_1", "lag_3_mean", "lag_12_mean"]
CATEGORICAL_FEATURE_COLUMNS = ["month_of_year"]
PROBABILITY_THRESHOLD = 0.5
TEST_MONTHS = 6
CALIBRATION_DECILES = 10


def split_train_test_by_time(
    feature_df: pd.DataFrame,
    test_months: int = TEST_MONTHS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data using the most recent months as test set."""
    unique_months = sorted(feature_df["month"].unique())
    if len(unique_months) <= test_months:
        raise ValueError(
            "Not enough unique months for a time split. "
            f"Need more than {test_months}, found {len(unique_months)}."
        )

    test_month_values = set(unique_months[-test_months:])
    train_df = feature_df[~feature_df["month"].isin(test_month_values)].copy()
    test_df = feature_df[feature_df["month"].isin(test_month_values)].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Time split produced an empty train or test set.")

    return train_df, test_df


def train_model(train_df: pd.DataFrame) -> Pipeline:
    """Train a logistic regression model with month-of-year one-hot encoding."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "month_of_year_one_hot",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURE_COLUMNS,
            ),
            ("numeric_features", "passthrough", NUMERIC_FEATURE_COLUMNS),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    model.fit(train_df[MODEL_FEATURE_COLUMNS], train_df["y"])
    return model


def _build_calibration_summary(
    y_true: pd.Series,
    y_prob: pd.Series,
    bins: int = CALIBRATION_DECILES,
) -> list[dict[str, Any]]:
    """Build a decile calibration summary from predicted probabilities."""
    calibration_df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).copy()

    unique_probability_count = calibration_df["y_prob"].nunique(dropna=True)
    if unique_probability_count < 2:
        mean_pred = float(calibration_df["y_prob"].mean())
        observed_rate = float(calibration_df["y_true"].mean())
        return [
            {
                "decile": 0,
                "count": int(len(calibration_df)),
                "mean_pred": mean_pred,
                "observed_rate": observed_rate,
            }
        ]

    # qcut may create fewer bins when probabilities are repeated.
    calibration_df["decile"] = pd.qcut(
        calibration_df["y_prob"],
        q=bins,
        labels=False,
        duplicates="drop",
    )

    grouped = calibration_df.groupby("decile", dropna=True)
    summary_df = grouped.agg(
        count=("y_true", "size"),
        mean_pred=("y_prob", "mean"),
        observed_rate=("y_true", "mean"),
    )

    summary_df = summary_df.reset_index().sort_values("decile")
    summary: list[dict[str, Any]] = []
    for _, row in summary_df.iterrows():
        summary.append(
            {
                "decile": int(row["decile"]),
                "count": int(row["count"]),
                "mean_pred": float(row["mean_pred"]),
                "observed_rate": float(row["observed_rate"]),
            }
        )

    return summary


def evaluate_model(model: Pipeline, test_df: pd.DataFrame) -> dict[str, Any]:
    """Evaluate model performance on a time-based holdout set."""
    y_true = test_df["y"]
    y_prob = model.predict_proba(test_df[MODEL_FEATURE_COLUMNS])[:, 1]
    y_pred = (y_prob >= PROBABILITY_THRESHOLD).astype(int)

    unique_classes = sorted(y_true.unique().tolist())
    auc: float | None
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
        "test_month_start": str(test_df["month"].min().date()),
        "test_month_end": str(test_df["month"].max().date()),
        "auc": auc,
        "precision_at_0_5": precision,
        "recall_at_0_5": recall,
        "calibration_by_decile": calibration_summary,
        "note": "AUC is null when test set has only one class.",
    }


def save_model(model: Pipeline, path: str) -> None:
    """Save a trained model to disk with joblib."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def load_model(path: str) -> Pipeline:
    """Load a trained model from disk."""
    return joblib.load(path)


def score_counties(model: Pipeline, feature_df: pd.DataFrame) -> pd.DataFrame:
    """Score county-month rows and return a tidy score dataframe."""
    scores = model.predict_proba(feature_df[MODEL_FEATURE_COLUMNS])[:, 1]
    output_df = feature_df[["county_fips", "month"]].copy()
    output_df["risk_score"] = scores
    return output_df
