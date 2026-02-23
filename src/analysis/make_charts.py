"""Generate lightweight analysis charts for the static demo site.

Usage:
    python -m src.analysis.make_charts
"""

from pathlib import Path

import matplotlib
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from src.config import RAW_EVICTION_LAB_YEARLY_PATH
from src.datasets.eviction_lab_yearly import clean_eviction_lab_yearly, load_eviction_lab_yearly
from src.features.eviction_lab_yearly_features import build_eviction_lab_yearly_features
from src.models.eviction_lab_yearly_model import (
    CALIBRATION_DECILES,
    MODEL_FEATURE_COLUMNS,
    split_by_outcome_year,
    train_eviction_lab_yearly_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / "web" / "assets"
CHART_DPI = 240
CHART_SIZE = (9.2, 5.6)

# Use a headless backend so chart generation works in local/server environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "legend.fontsize": 11,
    }
)


def _build_labeled_feature_table() -> pd.DataFrame:
    """Load raw CSV and return fully labeled rows ready for backtest splits."""
    if not RAW_EVICTION_LAB_YEARLY_PATH.exists():
        raise FileNotFoundError(
            "Missing raw CSV at "
            f"{RAW_EVICTION_LAB_YEARLY_PATH}. Download it before generating charts."
        )

    raw_df = load_eviction_lab_yearly(str(RAW_EVICTION_LAB_YEARLY_PATH))
    clean_df = clean_eviction_lab_yearly(raw_df)
    feature_df = build_eviction_lab_yearly_features(clean_df)

    required = MODEL_FEATURE_COLUMNS + ["y", "sample_weight", "outcome_year"]
    labeled_df = feature_df.dropna(subset=required).copy()
    labeled_df["y"] = labeled_df["y"].astype(int)
    labeled_df["outcome_year"] = labeled_df["outcome_year"].astype(int)
    return labeled_df


def _fit_and_score_holdout(
    labeled_df: pd.DataFrame,
    holdout_outcome_years: list[int],
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Train on non-holdout outcome years, then return holdout scores."""
    train_df, test_df = split_by_outcome_year(
        labeled_df,
        holdout_outcome_years=holdout_outcome_years,
    )

    model = train_eviction_lab_yearly_model(train_df)
    y_true = test_df["y"].astype(int)
    model_score = pd.Series(
        model.predict_proba(test_df[MODEL_FEATURE_COLUMNS])[:, 1],
        index=test_df.index,
    )
    naive_lag_1_score = pd.to_numeric(test_df["lag_1"], errors="coerce")
    return test_df, y_true, model_score.fillna(0.0), naive_lag_1_score.fillna(0.0)


def _safe_auc(y_true: pd.Series, score: pd.Series) -> float:
    """Compute AUC when both classes exist."""
    if y_true.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def _plot_auc_delta_vs_baseline(
    auc_rows: list[dict[str, float | str]],
    output_path: Path,
) -> None:
    """Bar chart: delta AUC (model minus naive lag_1) across holdout setups."""
    labels = [str(row["label"]) for row in auc_rows]
    model_values = [float(row["model_auc"]) for row in auc_rows]
    naive_values = [float(row["naive_auc"]) for row in auc_rows]
    delta_values = [model_auc - naive_auc for model_auc, naive_auc in zip(model_values, naive_values)]

    x_positions = range(len(labels))
    fig, axis = plt.subplots(figsize=CHART_SIZE)
    bars = axis.bar(
        x=list(x_positions),
        height=delta_values,
        width=0.54,
        color="#2f6bff",
        edgecolor="#1f55ff",
        linewidth=0.9,
    )

    axis.axhline(0.0, color="#6e7f9d", linewidth=1.2, linestyle="--")
    axis.set_title("Delta AUC vs Naive lag_1 Baseline")
    axis.set_ylabel("Delta AUC (Model - Naive)")
    axis.set_xticks(list(x_positions), labels)
    axis.grid(axis="y", alpha=0.2, linewidth=0.9)

    min_delta = min(delta_values + [0.0])
    max_delta = max(delta_values + [0.0])
    delta_span = max_delta - min_delta
    padding = max(0.0003, delta_span * 0.5 if delta_span > 0 else 0.0006)
    axis.set_ylim(min_delta - padding, max_delta + padding)

    label_offset = max(0.00003, padding * 0.04)
    for bar, delta in zip(bars, delta_values):
        text_y = delta + label_offset if delta >= 0 else delta - label_offset
        vertical_align = "bottom" if delta >= 0 else "top"
        delta_text = f"{delta:+.4f}\n{delta * 10000:+.0f} bps"
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            text_y,
            delta_text,
            ha="center",
            va=vertical_align,
            fontsize=11,
            fontweight=600,
            color="#17386e",
        )

    absolute_auc_lines = [
        f"{label}: Model {model_auc:.4f}, Naive {naive_auc:.4f}"
        for label, model_auc, naive_auc in zip(labels, model_values, naive_values)
    ]
    absolute_auc_text = "Absolute AUCs\n" + "\n".join(absolute_auc_lines)
    fig.text(
        0.02,
        0.02,
        absolute_auc_text,
        fontsize=10.5,
        color="#27406b",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#f4f8ff",
            "edgecolor": "#cfdbf7",
        },
    )

    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(output_path, dpi=CHART_DPI)
    plt.close(fig)


def _plot_calibration_curve(y_true: pd.Series, y_prob: pd.Series, output_path: Path) -> None:
    """Calibration curve with quantile bins."""
    calibration_df = pd.DataFrame(
        {"y_true": y_true.astype(float), "y_prob": pd.to_numeric(y_prob, errors="coerce")}
    ).dropna()

    if calibration_df["y_prob"].nunique() < 2:
        calibration_df["bin"] = 0
    else:
        calibration_df["bin"] = pd.qcut(
            calibration_df["y_prob"],
            q=CALIBRATION_DECILES,
            labels=False,
            duplicates="drop",
        )

    grouped = calibration_df.groupby("bin", dropna=True).agg(
        mean_pred=("y_prob", "mean"),
        observed_rate=("y_true", "mean"),
    )
    grouped = grouped.sort_values("mean_pred")

    fig, axis = plt.subplots(figsize=CHART_SIZE)
    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        color="#8aa2cf",
        label="Perfect calibration",
    )
    axis.plot(
        grouped["mean_pred"],
        grouped["observed_rate"],
        marker="o",
        markersize=6.5,
        linewidth=2.6,
        color="#2f6bff",
        label="Model bins",
    )

    axis.set_title("Calibration on Last-2-Years Holdout")
    axis.set_xlabel("Predicted risk")
    axis.set_ylabel("Observed top-quartile rate")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.2, linewidth=0.9)
    axis.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=CHART_DPI)
    plt.close(fig)


def _plot_roc_curve(
    y_true: pd.Series,
    model_score: pd.Series,
    naive_score: pd.Series,
    output_path: Path,
) -> None:
    """ROC plot for model and naive lag_1 baseline."""
    fig, axis = plt.subplots(figsize=CHART_SIZE)

    if y_true.nunique() >= 2:
        model_fpr, model_tpr, _ = roc_curve(y_true, model_score)
        naive_fpr, naive_tpr, _ = roc_curve(y_true, naive_score)
        model_auc = roc_auc_score(y_true, model_score)
        naive_auc = roc_auc_score(y_true, naive_score)

        axis.plot(
            model_fpr,
            model_tpr,
            color="#2f6bff",
            linewidth=2.6,
            label=f"Model (AUC {model_auc:.4f})",
        )
        axis.plot(
            naive_fpr,
            naive_tpr,
            color="#9db7ff",
            linewidth=2.6,
            linestyle="--",
            label=f"Naive lag_1 (AUC {naive_auc:.4f})",
        )

    axis.plot([0, 1], [0, 1], color="#8aa2cf", linestyle=":", linewidth=1.6, label="Chance")
    axis.set_title("ROC Curve on Last-2-Years Holdout")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.2, linewidth=0.9)
    axis.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=CHART_DPI)
    plt.close(fig)


def main() -> None:
    """Generate demo analysis charts into web/assets."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    labeled_df = _build_labeled_feature_table()
    holdout_years = sorted(labeled_df["outcome_year"].unique().tolist())
    if len(holdout_years) < 2:
        raise ValueError("Need at least two outcome years to generate analysis charts.")

    last_year = int(holdout_years[-1])
    last_two = [int(holdout_years[-2]), int(holdout_years[-1])]

    _, y_last_year, model_last_year, naive_last_year = _fit_and_score_holdout(
        labeled_df=labeled_df,
        holdout_outcome_years=[last_year],
    )
    _, y_last_two, model_last_two, naive_last_two = _fit_and_score_holdout(
        labeled_df=labeled_df,
        holdout_outcome_years=last_two,
    )

    auc_rows = [
        {
            "label": f"Last year holdout ({last_year})",
            "model_auc": _safe_auc(y_last_year, model_last_year),
            "naive_auc": _safe_auc(y_last_year, naive_last_year),
        },
        {
            "label": f"Last 2 years pooled ({last_two[0]}-{last_two[1]})",
            "model_auc": _safe_auc(y_last_two, model_last_two),
            "naive_auc": _safe_auc(y_last_two, naive_last_two),
        },
    ]

    _plot_auc_delta_vs_baseline(
        auc_rows=auc_rows,
        output_path=ASSETS_DIR / "analysis_model_vs_naive_auc.png",
    )
    _plot_calibration_curve(
        y_true=y_last_two,
        y_prob=model_last_two,
        output_path=ASSETS_DIR / "analysis_calibration_curve.png",
    )
    _plot_roc_curve(
        y_true=y_last_two,
        model_score=model_last_two,
        naive_score=naive_last_two,
        output_path=ASSETS_DIR / "analysis_roc_curve.png",
    )

    print(f"Wrote charts to {ASSETS_DIR}")


if __name__ == "__main__":
    main()
