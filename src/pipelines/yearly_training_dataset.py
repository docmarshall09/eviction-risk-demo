"""Shared yearly training-dataset filters and audit helpers.

This module centralizes the exact row filter used to build the yearly
logistic-regression training dataframe so training code and audit exports
stay in sync.
"""

import pandas as pd

from src.models.eviction_lab_yearly_model import MODEL_FEATURE_COLUMNS


TRAINING_REQUIRED_COLUMNS = MODEL_FEATURE_COLUMNS + [
    "y",
    "sample_weight",
    "outcome_year",
]
AUDIT_REQUIRED_COLUMNS = [
    "county_fips",
    "year",
    "outcome_year",
] + TRAINING_REQUIRED_COLUMNS

DROP_REASON_BY_COLUMN = {
    "lag_1": "missing_lag_1",
    "lag_3_mean_obs": "missing_lag_3_mean_obs",
    "lag_5_mean_obs": "missing_lag_5_mean_obs",
    "years_since_last_obs": "missing_years_since_last_obs",
    "y": "missing_y",
    "sample_weight": "missing_sample_weight",
    "outcome_year": "missing_outcome_year",
}


def _validate_required_columns(
    feature_df: pd.DataFrame,
    required_columns: list[str],
) -> None:
    """Raise a clear error when expected columns are missing."""
    missing_columns = sorted(set(required_columns) - set(feature_df.columns))
    if missing_columns:
        raise ValueError(
            "Feature dataframe is missing required columns for yearly training: "
            f"{missing_columns}."
        )


def build_yearly_training_dataset(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Build the exact yearly training dataframe used by CLI training tasks.

    Args:
        feature_df: Yearly feature table from the shared pipeline.

    Returns:
        Dataframe filtered to rows with complete model inputs and labels.
    """
    _validate_required_columns(feature_df, TRAINING_REQUIRED_COLUMNS)
    labeled_df = feature_df.dropna(subset=TRAINING_REQUIRED_COLUMNS).copy()
    labeled_df["y"] = labeled_df["y"].astype(int)
    return labeled_df


def build_yearly_training_dataset_with_audit(
    feature_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build yearly training rows and return row-level and step-level audits.

    Args:
        feature_df: Yearly feature table from the shared pipeline.

    Returns:
        Tuple of:
            - filtered training dataframe
            - row-level audit dataframe
            - step-level filter counts dataframe
    """
    _validate_required_columns(feature_df, AUDIT_REQUIRED_COLUMNS)

    row_audit_df = pd.DataFrame(
        {
            "county_fips": feature_df["county_fips"].astype(str).str.strip().str.zfill(5),
            "target_year": feature_df["outcome_year"],
            "as_of_year": feature_df["year"],
            "kept": 0,
            "drop_reason": "",
        },
        index=feature_df.index,
    )

    remaining_mask = pd.Series(True, index=feature_df.index, dtype=bool)
    rows_remaining = int(remaining_mask.sum())
    counts_rows = [
        {
            "step_name": "input_feature_rows",
            "rows_remaining": rows_remaining,
            "rows_dropped": 0,
            "drop_reason": "",
        }
    ]

    for column_name in TRAINING_REQUIRED_COLUMNS:
        missing_mask = feature_df[column_name].isna()
        dropped_mask = remaining_mask & missing_mask
        dropped_count = int(dropped_mask.sum())
        remaining_mask = remaining_mask & ~missing_mask
        rows_remaining = int(remaining_mask.sum())

        counts_rows.append(
            {
                "step_name": f"require_non_null_{column_name}",
                "rows_remaining": rows_remaining,
                "rows_dropped": dropped_count,
                "drop_reason": DROP_REASON_BY_COLUMN[column_name],
            }
        )

    row_audit_df.loc[remaining_mask, "kept"] = 1

    for column_name in TRAINING_REQUIRED_COLUMNS:
        first_missing_mask = (
            (row_audit_df["kept"] == 0)
            & (row_audit_df["drop_reason"] == "")
            & feature_df[column_name].isna()
        )
        row_audit_df.loc[first_missing_mask, "drop_reason"] = (
            DROP_REASON_BY_COLUMN[column_name]
        )

    row_audit_df.loc[row_audit_df["kept"] == 1, "drop_reason"] = "kept"

    training_df = feature_df.loc[remaining_mask].copy()
    training_df["y"] = training_df["y"].astype(int)

    counts_rows.append(
        {
            "step_name": "cast_label_to_int",
            "rows_remaining": int(len(training_df)),
            "rows_dropped": 0,
            "drop_reason": "",
        }
    )
    counts_rows.append(
        {
            "step_name": "dedupe_rows_not_applied",
            "rows_remaining": int(len(training_df)),
            "rows_dropped": 0,
            "drop_reason": "no_dedupe_logic_in_training_dataset_builder",
        }
    )

    counts_df = pd.DataFrame(counts_rows)
    return training_df, row_audit_df.reset_index(drop=True), counts_df
