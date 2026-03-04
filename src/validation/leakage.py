"""Runtime guards against temporal leakage in time-series training data.

In this project, features from year t predict eviction outcomes in year t+1.
Temporal leakage occurs when a row's feature year (\"year\") is greater than or
equal to its outcome year (\"outcome_year\"), meaning the model would be trained
on future information to predict the past — the most dangerous ML bug in a
time-ordered pipeline.
"""

import pandas as pd


def assert_no_temporal_leakage(df: pd.DataFrame) -> None:
    """Raise ValueError if any row has year >= outcome_year.

    In a valid training row, features from year t predict the outcome in year
    t+1, so ``year`` must be strictly less than ``outcome_year``. Any row that
    violates this constraint means future data leaked into the feature set.

    Args:
        df: Labeled training DataFrame containing \"year\" and \"outcome_year\" columns.

    Raises:
        ValueError: If one or more rows have year >= outcome_year, including a
            count of violated rows and a sample of up to 5 offending records.
    """
    leaked = df[df["year"] >= df["outcome_year"]]
    if leaked.empty:
        return

    sample = leaked[["year", "outcome_year"]].head(5).to_string(index=True)
    raise ValueError(
        f"Temporal leakage detected: {len(leaked)} row(s) have year >= outcome_year. "
        f"Features must come from year t to predict outcome_year t+1.\n"
        f"Sample of offending rows:\n{sample}"
    )
