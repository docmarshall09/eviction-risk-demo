"""Feature engineering for county-level eviction risk modeling."""

import pandas as pd


FEATURE_COLUMNS = [
    "county_fips",
    "month",
    "y",
    "lag_1",
    "lag_3_mean",
    "lag_12_mean",
    "month_of_year",
]


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag-based features for each county over time."""
    features_df = df.copy()
    grouped = features_df.groupby("county_fips", group_keys=False)["eviction_filing_rate"]

    # lag_1 uses the current month value at time t by problem definition.
    features_df["lag_1"] = features_df["eviction_filing_rate"]
    features_df["lag_3_mean"] = grouped.transform(
        lambda series: series.rolling(window=3, min_periods=3).mean()
    )
    features_df["lag_12_mean"] = grouped.transform(
        lambda series: series.rolling(window=12, min_periods=12).mean()
    )
    features_df["month_of_year"] = features_df["month"].dt.month
    return features_df


def _build_next_month_label(df: pd.DataFrame) -> pd.DataFrame:
    """Build next-month elevated risk labels and align them back to month t."""
    label_df = df[["county_fips", "month", "eviction_filing_rate"]].copy()

    # For each month, compute the top-quartile threshold across counties.
    label_df["top_quartile_threshold"] = label_df.groupby("month")[
        "eviction_filing_rate"
    ].transform(lambda series: series.quantile(0.75))

    label_df["y"] = (
        label_df["eviction_filing_rate"] >= label_df["top_quartile_threshold"]
    ).astype(int)

    # Shift labels back one month so features at t predict elevated risk at t+1.
    label_df["month"] = label_df["month"] - pd.DateOffset(months=1)
    return label_df[["county_fips", "month", "y"]]


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Create the final modeling table for eviction risk.

    Args:
        df: Cleaned eviction dataframe with county_fips, month, and
            eviction_filing_rate.

    Returns:
        Tidy feature dataframe with model-ready columns.
    """
    base_df = df.copy().sort_values(["county_fips", "month"]).reset_index(drop=True)
    features_df = _add_lag_features(base_df)
    labels_df = _build_next_month_label(base_df)

    modeling_df = features_df.merge(
        labels_df,
        on=["county_fips", "month"],
        how="left",
    )

    modeling_df = modeling_df.dropna(
        subset=["y", "lag_1", "lag_3_mean", "lag_12_mean"]
    ).copy()
    modeling_df["y"] = modeling_df["y"].astype(int)

    modeling_df = modeling_df[FEATURE_COLUMNS].sort_values(
        ["county_fips", "month"]
    )
    return modeling_df.reset_index(drop=True)
