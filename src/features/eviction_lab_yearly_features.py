"""Feature engineering for Eviction Lab county-year eviction risk modeling."""

import pandas as pd


OUTPUT_COLUMNS = [
    "county_fips",
    "year",
    "outcome_year",
    "y",
    "lag_1",
    "lag_3_mean_obs",
    "lag_5_mean_obs",
    "years_since_last_obs",
    "sample_weight",
]


def _add_yearly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag features on observed rows for each county."""
    features_df = df.copy()
    grouped = features_df.groupby("county_fips", group_keys=False)

    features_df["lag_1"] = features_df["filing_rate"]
    features_df["lag_3_mean_obs"] = grouped["filing_rate"].transform(
        lambda series: series.rolling(window=3, min_periods=3).mean()
    )
    features_df["lag_5_mean_obs"] = grouped["filing_rate"].transform(
        lambda series: series.rolling(window=5, min_periods=5).mean()
    )

    previous_year = grouped["year"].shift(1)
    features_df["years_since_last_obs"] = features_df["year"] - previous_year

    # First observation for each county has no prior year; set to 0 explicitly.
    features_df["years_since_last_obs"] = (
        features_df["years_since_last_obs"].fillna(0).astype(int)
    )
    return features_df


def _build_next_year_label(df: pd.DataFrame) -> pd.DataFrame:
    """Build labels where y at year t indicates top-quartile risk at year t+1."""
    next_year_df = df[["county_fips", "year", "filing_rate"]].copy()

    next_year_df["top_quartile_threshold"] = next_year_df.groupby("year")[
        "filing_rate"
    ].transform(lambda series: series.quantile(0.75))

    next_year_df["y"] = (
        next_year_df["filing_rate"] >= next_year_df["top_quartile_threshold"]
    ).astype(int)

    # Shift labels backward one year so each row at t targets the event at t+1.
    next_year_df["year"] = next_year_df["year"] - 1
    return next_year_df[["county_fips", "year", "y"]]


def build_eviction_lab_yearly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build model features and labels for the Eviction Lab yearly panel.

    Args:
        df: Cleaned yearly dataframe with county_fips, year, filing_rate,
            and sample_weight.

    Returns:
        Tidy feature dataframe for yearly model training/scoring.
    """
    base_df = df.copy().sort_values(["county_fips", "year"]).reset_index(drop=True)
    feature_df = _add_yearly_features(base_df)
    label_df = _build_next_year_label(base_df)

    modeling_df = feature_df.merge(label_df, on=["county_fips", "year"], how="left")
    modeling_df["outcome_year"] = modeling_df["year"] + 1

    # Mark whether this county has an observed row at the target outcome year.
    outcome_observation_df = base_df[["county_fips", "year"]].copy()
    outcome_observation_df = outcome_observation_df.rename(
        columns={"year": "outcome_year"}
    )
    outcome_observation_df["has_outcome_observation"] = 1

    modeling_df = modeling_df.merge(
        outcome_observation_df,
        on=["county_fips", "outcome_year"],
        how="left",
    )

    # Keep y only when the county has an actual next-year observation.
    modeling_df.loc[modeling_df["has_outcome_observation"] != 1, "y"] = pd.NA
    modeling_df = modeling_df.dropna(
        subset=["lag_1", "lag_3_mean_obs", "lag_5_mean_obs"]
    ).copy()
    modeling_df["y"] = modeling_df["y"].astype("Int64")

    modeling_df = modeling_df[OUTPUT_COLUMNS].sort_values(["county_fips", "year"])
    return modeling_df.reset_index(drop=True)
