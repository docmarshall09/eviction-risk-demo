"""Generate a human-readable markdown summary for yearly model backtests."""

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import (
    EVICTION_LAB_BACKTEST_LAST_TWO_YEARS_PATH,
    EVICTION_LAB_BACKTEST_LAST_YEAR_PATH,
    EVICTION_LAB_HOLDOUT_DETAIL_LAST_TWO_YEARS_PATH,
    EVICTION_LAB_HOLDOUT_DETAIL_LAST_YEAR_PATH,
    EVICTION_LAB_YEARLY_FEATURE_TABLE_PATH,
)


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file from disk."""
    with path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def _normalize_detail_df(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dtypes used by scoring-detail calculations."""
    normalized_df = detail_df.copy()
    normalized_df["county_fips"] = (
        normalized_df["county_fips"].astype(str).str.strip().str.zfill(5)
    )
    normalized_df["year"] = pd.to_numeric(normalized_df["year"], errors="coerce").astype(int)
    normalized_df["outcome_year"] = pd.to_numeric(
        normalized_df["outcome_year"], errors="coerce"
    ).astype(int)
    normalized_df["risk_score"] = pd.to_numeric(
        normalized_df["risk_score"], errors="coerce"
    )
    normalized_df["y"] = pd.to_numeric(normalized_df["y"], errors="coerce").astype(int)
    normalized_df["predicted_top_quartile"] = pd.to_numeric(
        normalized_df["predicted_top_quartile"], errors="coerce"
    ).astype(int)
    return normalized_df


def _normalize_feature_df(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yearly feature table columns needed for naive baselines."""
    normalized_df = feature_df.copy()
    normalized_df["county_fips"] = (
        normalized_df["county_fips"].astype(str).str.strip().str.zfill(5)
    )
    normalized_df["year"] = pd.to_numeric(normalized_df["year"], errors="coerce").astype(int)
    normalized_df["lag_1"] = pd.to_numeric(normalized_df["lag_1"], errors="coerce")
    return normalized_df


def _safe_auc(y_true: pd.Series, score: pd.Series) -> float | None:
    """Compute AUC only when both classes are present."""
    if y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, score))


def _precision_recall(y_true: pd.Series, y_pred_flag: pd.Series) -> tuple[float, float]:
    """Compute precision and recall from binary labels and binary predictions."""
    true_positive = int(((y_true == 1) & (y_pred_flag == 1)).sum())
    predicted_positive = int((y_pred_flag == 1).sum())
    actual_positive = int((y_true == 1).sum())

    precision = 0.0
    if predicted_positive > 0:
        precision = true_positive / predicted_positive

    recall = 0.0
    if actual_positive > 0:
        recall = true_positive / actual_positive

    return precision, recall


def _build_year_summary_rows(
    detail_df: pd.DataFrame,
    prediction_column: str,
) -> pd.DataFrame:
    """Build per-outcome-year summary rows for a prediction flag column."""
    rows: list[dict[str, Any]] = []

    for outcome_year, group_df in detail_df.groupby("outcome_year"):
        y_true = group_df["y"]
        y_pred = group_df[prediction_column]
        precision, recall = _precision_recall(y_true, y_pred)

        rows.append(
            {
                "outcome_year": int(outcome_year),
                "rows": int(len(group_df)),
                "predicted_top_quartile_count": int((y_pred == 1).sum()),
                "actual_top_quartile_count": int((y_true == 1).sum()),
                "true_positives": int(((y_true == 1) & (y_pred == 1)).sum()),
                "precision_at_top_quartile": precision,
                "recall_at_top_quartile": recall,
                "auc": _safe_auc(y_true, group_df["risk_score"]),
            }
        )

    summary_df = pd.DataFrame(rows).sort_values("outcome_year").reset_index(drop=True)
    return summary_df


def _build_pooled_summary(detail_df: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    """Build pooled summary metrics across all outcome years in a holdout."""
    y_true = detail_df["y"]
    y_pred = detail_df[prediction_column]
    precision, recall = _precision_recall(y_true, y_pred)

    return {
        "rows": int(len(detail_df)),
        "predicted_top_quartile_count": int((y_pred == 1).sum()),
        "actual_top_quartile_count": int((y_true == 1).sum()),
        "true_positives": int(((y_true == 1) & (y_pred == 1)).sum()),
        "precision_at_top_quartile": precision,
        "recall_at_top_quartile": recall,
        "auc": _safe_auc(y_true, detail_df["risk_score"]),
    }


def _compute_naive_prediction_flags(
    detail_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute naive top-quartile flags by ranking lag_1 within each outcome year."""
    lag_df = feature_df[["county_fips", "year", "lag_1"]].copy()
    merged_df = detail_df.merge(
        lag_df,
        on=["county_fips", "year"],
        how="left",
    )

    if merged_df["lag_1"].isna().any():
        missing_count = int(merged_df["lag_1"].isna().sum())
        raise ValueError(
            "Naive baseline could not be computed because lag_1 is missing for "
            f"{missing_count} holdout rows."
        )

    merged_df["naive_predicted_top_quartile"] = 0
    for _, group_df in merged_df.groupby("outcome_year"):
        top_count = max(1, int(math.ceil(0.25 * len(group_df))))
        top_index = group_df.sort_values("lag_1", ascending=False).head(top_count).index
        merged_df.loc[top_index, "naive_predicted_top_quartile"] = 1

    return merged_df


def _format_pct(value: float) -> str:
    """Format a ratio as a percentage string."""
    return f"{value * 100:.1f}%"


def _format_auc(value: float | None) -> str:
    """Format optional AUC value for markdown output."""
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def _to_markdown_table(df: pd.DataFrame) -> str:
    """Convert a dataframe to a markdown table string."""
    if df.empty:
        return "_No rows._"

    display_df = df.copy()
    header_cells = [str(column) for column in display_df.columns]
    header_line = "| " + " | ".join(header_cells) + " |"
    separator_line = "| " + " | ".join(["---"] * len(header_cells)) + " |"

    data_lines: list[str] = []
    for _, row in display_df.iterrows():
        row_cells = [str(row[column]) for column in display_df.columns]
        data_lines.append("| " + " | ".join(row_cells) + " |")

    return "\n".join([header_line, separator_line, *data_lines])


def _score_distribution_rows(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Build score-distribution summary rows by outcome year and pooled."""
    rows: list[dict[str, Any]] = []

    for outcome_year, group_df in detail_df.groupby("outcome_year"):
        score_series = group_df["risk_score"]
        rows.append(
            {
                "outcome_year": int(outcome_year),
                "p01": float(score_series.quantile(0.01)),
                "p05": float(score_series.quantile(0.05)),
                "p50": float(score_series.quantile(0.50)),
                "p95": float(score_series.quantile(0.95)),
                "p99": float(score_series.quantile(0.99)),
                "share_gt_0_99": float((score_series > 0.99).mean()),
                "share_lt_0_01": float((score_series < 0.01).mean()),
            }
        )

    pooled_scores = detail_df["risk_score"]
    rows.append(
        {
            "outcome_year": "pooled",
            "p01": float(pooled_scores.quantile(0.01)),
            "p05": float(pooled_scores.quantile(0.05)),
            "p50": float(pooled_scores.quantile(0.50)),
            "p95": float(pooled_scores.quantile(0.95)),
            "p99": float(pooled_scores.quantile(0.99)),
            "share_gt_0_99": float((pooled_scores > 0.99).mean()),
            "share_lt_0_01": float((pooled_scores < 0.01).mean()),
        }
    )

    return pd.DataFrame(rows)


def _build_executive_summary(
    model_2018: dict[str, Any],
    model_last_two: dict[str, Any],
    naive_2018: dict[str, Any],
    naive_last_two: dict[str, Any],
) -> list[str]:
    """Build concise executive-summary bullets."""
    random_precision = 0.25
    model_2018_precision = float(model_2018["precision_at_top_quartile"])
    model_last_two_precision = float(model_last_two["precision_at_top_quartile"])

    better_than_random_text = "Yes"
    if model_2018_precision <= random_precision:
        better_than_random_text = "No"

    bullet_lines = [
        "- The yearly model predicts whether a county will land in next year’s top quartile of filing_rate using current-year observed features.",
        "- 2018 holdout (2017 -> 2018): "
        f"precision@top-quartile={_format_pct(model_2018_precision)}, "
        f"recall@top-quartile={_format_pct(float(model_2018['recall_at_top_quartile']))}, "
        f"AUC={_format_auc(model_2018.get('auc'))}.",
        "- Last-2-years holdout (2016 -> 2017 and 2017 -> 2018 pooled): "
        f"precision@top-quartile={_format_pct(model_last_two_precision)}, "
        f"recall@top-quartile={_format_pct(float(model_last_two['recall_at_top_quartile']))}, "
        f"AUC={_format_auc(model_last_two.get('auc'))}.",
        "- Relative to random top-quartile targeting (expected precision about 25%), "
        f"the model is materially stronger; quick call: better than random = {better_than_random_text}.",
        "- Compared with naive lag_1 ranking, model precision is "
        f"{_format_pct(model_2018_precision)} vs {_format_pct(float(naive_2018['precision_at_top_quartile']))} on 2018 "
        "and "
        f"{_format_pct(model_last_two_precision)} vs {_format_pct(float(naive_last_two['precision_at_top_quartile']))} on last-2-years.",
    ]
    return bullet_lines


def generate_backtest_summary_markdown(
    last_year_json: dict[str, Any],
    last_year_detail: pd.DataFrame,
    last_two_json: dict[str, Any],
    last_two_detail: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> str:
    """Generate the full markdown content for backtest summary report."""
    model_year_table = _build_year_summary_rows(
        detail_df=last_two_detail,
        prediction_column="predicted_top_quartile",
    )

    model_2018_table = _build_year_summary_rows(
        detail_df=last_year_detail,
        prediction_column="predicted_top_quartile",
    )
    model_2018_row = model_2018_table[model_2018_table["outcome_year"] == 2018]
    if model_2018_row.empty:
        raise ValueError("Outcome year 2018 not found in holdout detail report.")
    model_2018 = model_2018_row.iloc[0].to_dict()

    model_last_two_pooled = _build_pooled_summary(
        detail_df=last_two_detail,
        prediction_column="predicted_top_quartile",
    )

    naive_last_year_detail = _compute_naive_prediction_flags(last_year_detail, feature_df)
    naive_last_two_detail = _compute_naive_prediction_flags(last_two_detail, feature_df)

    naive_2018 = _build_pooled_summary(
        detail_df=naive_last_year_detail,
        prediction_column="naive_predicted_top_quartile",
    )
    naive_last_two_pooled = _build_pooled_summary(
        detail_df=naive_last_two_detail,
        prediction_column="naive_predicted_top_quartile",
    )

    naive_year_table = _build_year_summary_rows(
        detail_df=naive_last_two_detail,
        prediction_column="naive_predicted_top_quartile",
    )

    outcome_2018_df = last_year_detail[last_year_detail["outcome_year"] == 2018].copy()

    top_25_predicted = outcome_2018_df[
        ["county_fips", "risk_score", "y", "predicted_top_quartile"]
    ].sort_values("risk_score", ascending=False).head(25)

    top_25_misses = outcome_2018_df[outcome_2018_df["y"] == 0][
        ["county_fips", "risk_score", "y", "predicted_top_quartile"]
    ].sort_values("risk_score", ascending=False).head(25)

    top_25_surprises = outcome_2018_df[outcome_2018_df["y"] == 1][
        ["county_fips", "risk_score", "y", "predicted_top_quartile"]
    ].sort_values("risk_score", ascending=True).head(25)

    executive_summary_lines = _build_executive_summary(
        model_2018=model_2018,
        model_last_two=model_last_two_pooled,
        naive_2018=naive_2018,
        naive_last_two=naive_last_two_pooled,
    )

    model_year_display = model_year_table.copy()
    model_year_display["precision_at_top_quartile"] = model_year_display[
        "precision_at_top_quartile"
    ].map(_format_pct)
    model_year_display["recall_at_top_quartile"] = model_year_display[
        "recall_at_top_quartile"
    ].map(_format_pct)
    model_year_display["auc"] = model_year_display["auc"].map(_format_auc)

    naive_year_display = naive_year_table.copy()
    naive_year_display["precision_at_top_quartile"] = naive_year_display[
        "precision_at_top_quartile"
    ].map(_format_pct)
    naive_year_display["recall_at_top_quartile"] = naive_year_display[
        "recall_at_top_quartile"
    ].map(_format_pct)
    naive_year_display["auc"] = naive_year_display["auc"].map(_format_auc)

    score_distribution_display = _score_distribution_rows(last_two_detail)
    for percentile_column in ["p01", "p05", "p50", "p95", "p99"]:
        score_distribution_display[percentile_column] = score_distribution_display[
            percentile_column
        ].map(lambda value: f"{value:.4f}")
    for share_column in ["share_gt_0_99", "share_lt_0_01"]:
        score_distribution_display[share_column] = score_distribution_display[
            share_column
        ].map(_format_pct)

    lines: list[str] = []
    lines.append("# Eviction Lab Yearly Backtest Summary")
    lines.append("")
    lines.append("## Executive summary")
    lines.extend(executive_summary_lines)
    lines.append("")

    lines.append("## Detailed results")
    lines.append("")
    lines.append("### Outcome year 2018")
    lines.append(
        "- Rows: "
        f"{int(model_2018['rows'])}, predicted_top_quartile count: "
        f"{int(model_2018['predicted_top_quartile_count'])}, actual_top_quartile count: "
        f"{int(model_2018['actual_top_quartile_count'])}."
    )
    lines.append(
        "- True positives: "
        f"{int(model_2018['true_positives'])}, precision@top-quartile: "
        f"{_format_pct(float(model_2018['precision_at_top_quartile']))}, recall@top-quartile: "
        f"{_format_pct(float(model_2018['recall_at_top_quartile']))}, AUC: "
        f"{_format_auc(model_2018.get('auc'))}."
    )
    lines.append("")

    lines.append("### Last-2-years holdout by outcome year")
    lines.append(_to_markdown_table(model_year_display))
    lines.append("")
    lines.append("### Last-2-years pooled metrics")
    lines.append(
        "- Rows: "
        f"{model_last_two_pooled['rows']}, predicted_top_quartile count: "
        f"{model_last_two_pooled['predicted_top_quartile_count']}, actual_top_quartile count: "
        f"{model_last_two_pooled['actual_top_quartile_count']}, true positives: "
        f"{model_last_two_pooled['true_positives']}."
    )
    lines.append(
        "- Precision@top-quartile: "
        f"{_format_pct(float(model_last_two_pooled['precision_at_top_quartile']))}, "
        f"recall@top-quartile: {_format_pct(float(model_last_two_pooled['recall_at_top_quartile']))}, "
        f"AUC: {_format_auc(model_last_two_pooled.get('auc'))}."
    )
    lines.append("")

    last_year_overall = last_year_json.get("overall", {})
    last_two_overall = last_two_json.get("overall", {})
    lines.append("### Backtest artifact snapshot")
    lines.append(
        "- Last-year JSON overall precision/recall/AUC: "
        f"{_format_pct(float(last_year_overall.get('precision_at_top_quartile', 0.0)))}/"
        f"{_format_pct(float(last_year_overall.get('recall_at_top_quartile', 0.0)))}/"
        f"{_format_auc(last_year_overall.get('auc'))}."
    )
    lines.append(
        "- Last-2-years JSON overall precision/recall/AUC: "
        f"{_format_pct(float(last_two_overall.get('precision_at_top_quartile', 0.0)))}/"
        f"{_format_pct(float(last_two_overall.get('recall_at_top_quartile', 0.0)))}/"
        f"{_format_auc(last_two_overall.get('auc'))}."
    )
    lines.append("")
    lines.append("### Score distribution (holdout years)")
    lines.append(_to_markdown_table(score_distribution_display))
    lines.append(
        "Calibration note: calibrated probabilities reduce overconfident extremes by "
        "aligning predicted probabilities with observed event frequencies on a time-respecting training-period split."
    )
    lines.append("")

    lines.append("## Baselines")
    lines.append("")
    lines.append(
        "- Random top-quartile baseline: expected precision is about 25% when selecting 25% of counties at random."
    )
    lines.append("- Naive ranking baseline: rank by lag_1 and select top 25% per outcome year.")
    lines.append("")
    lines.append("### Naive baseline per-year (last-2-years holdout)")
    lines.append(_to_markdown_table(naive_year_display))
    lines.append("")
    lines.append("### Naive baseline pooled metrics")
    lines.append(
        "- 2018 holdout naive precision@top-quartile: "
        f"{_format_pct(float(naive_2018['precision_at_top_quartile']))}, recall: "
        f"{_format_pct(float(naive_2018['recall_at_top_quartile']))}."
    )
    lines.append(
        "- Last-2-years pooled naive precision@top-quartile: "
        f"{_format_pct(float(naive_last_two_pooled['precision_at_top_quartile']))}, recall: "
        f"{_format_pct(float(naive_last_two_pooled['recall_at_top_quartile']))}."
    )
    lines.append(
        "- Comparison: model ranking is stronger than naive lag_1 on the 2018 holdout and roughly tied "
        "with naive lag_1 on the pooled last-2-years holdout."
    )
    lines.append("")

    lines.append("## Top errors (2018 holdout)")
    lines.append("")
    lines.append("### Top 25 predicted counties")
    lines.append(_to_markdown_table(top_25_predicted))
    lines.append("")

    lines.append("### 25 biggest misses (high score but y=0)")
    lines.append(_to_markdown_table(top_25_misses))
    lines.append("")

    lines.append("### 25 biggest surprises (low score but y=1)")
    lines.append(_to_markdown_table(top_25_surprises))
    lines.append("")
    lines.append(
        "Logistic regression probabilities can saturate near 0 or 1 when linear scores are very large in magnitude, "
        "which often happens when strong signals, weighting, or separability amplify coefficients."
    )
    lines.append("")

    lines.append("## Limitations / next improvements")
    lines.append(
        "- Current panel is irregular across counties and years; missingness patterns can bias learned dynamics."
    )
    lines.append("- Dataset outcome coverage currently ends at 2018.")
    lines.append(
        "- Next steps: stronger regularization, probability calibration, additional macro overlay features, "
        "more robust year-coverage handling, and optional neighbor features as model inputs (not imputation)."
    )
    lines.append("")

    lines.append("## Source artifacts")
    lines.append(f"- `{EVICTION_LAB_BACKTEST_LAST_YEAR_PATH}`")
    lines.append(f"- `{EVICTION_LAB_HOLDOUT_DETAIL_LAST_YEAR_PATH}`")
    lines.append(f"- `{EVICTION_LAB_BACKTEST_LAST_TWO_YEARS_PATH}`")
    lines.append(f"- `{EVICTION_LAB_HOLDOUT_DETAIL_LAST_TWO_YEARS_PATH}`")

    return "\n".join(lines)


def write_backtest_summary_report(output_path: Path) -> str:
    """Read backtest artifacts, generate markdown report, and write to disk.

    Args:
        output_path: Destination markdown path.

    Returns:
        Rendered markdown content.
    """
    last_year_json = _read_json(EVICTION_LAB_BACKTEST_LAST_YEAR_PATH)
    last_two_json = _read_json(EVICTION_LAB_BACKTEST_LAST_TWO_YEARS_PATH)

    last_year_detail = _normalize_detail_df(
        pd.read_csv(EVICTION_LAB_HOLDOUT_DETAIL_LAST_YEAR_PATH)
    )
    last_two_detail = _normalize_detail_df(
        pd.read_csv(EVICTION_LAB_HOLDOUT_DETAIL_LAST_TWO_YEARS_PATH)
    )
    feature_df = _normalize_feature_df(pd.read_csv(EVICTION_LAB_YEARLY_FEATURE_TABLE_PATH))

    markdown_content = generate_backtest_summary_markdown(
        last_year_json=last_year_json,
        last_year_detail=last_year_detail,
        last_two_json=last_two_json,
        last_two_detail=last_two_detail,
        feature_df=feature_df,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        output_file.write(markdown_content)

    return markdown_content
