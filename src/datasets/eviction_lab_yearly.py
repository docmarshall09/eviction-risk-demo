"""Load and clean Eviction Lab county-year data for yearly risk modeling."""

import logging
from pathlib import Path
import pandas as pd


LOGGER = logging.getLogger(__name__)
MAX_FIPS_LENGTH = 5
DEFAULT_SAMPLE_WEIGHT = 1.0
WEIGHT_CAP_QUANTILE = 0.99


REQUIRED_YEARLY_COLUMNS = {
    "cofips": "county_fips",
    "year": "year",
    "filings": "filings",
    "filing_rate": "filing_rate",
}


def _normalize_column_name(column_name: str) -> str:
    """Normalize a column name for case-insensitive matching."""
    return "".join(character for character in column_name.lower() if character.isalnum())


def _normalize_fips(value: object) -> str | None:
    """Convert county FIPS values to zero-padded 5-character strings."""
    if pd.isna(value):
        return None

    text_value = str(value).strip()
    if text_value == "":
        return None

    digits_only = "".join(character for character in text_value if character.isdigit())
    if digits_only == "" or len(digits_only) > MAX_FIPS_LENGTH:
        return None

    return digits_only.zfill(MAX_FIPS_LENGTH)


def load_eviction_lab_yearly(path: str) -> pd.DataFrame:
    """Load Eviction Lab yearly county-level data from a CSV file.

    Args:
        path: File path to the Eviction Lab county-year CSV.

    Returns:
        Raw dataframe from the CSV file.

    Raises:
        FileNotFoundError: If the file path does not exist.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            "Eviction Lab yearly CSV not found at "
            f"{csv_path}. Add data/raw/county_proprietary_valid_2000_2018.csv."
        )

    return pd.read_csv(csv_path)


def _build_column_map(columns: pd.Index) -> dict[str, str]:
    """Map raw columns to canonical names using case-insensitive matching."""
    normalized_to_original = {
        _normalize_column_name(str(column)): str(column) for column in columns
    }

    column_map: dict[str, str] = {}
    for source_name, target_name in REQUIRED_YEARLY_COLUMNS.items():
        normalized_source = _normalize_column_name(source_name)
        original_column = normalized_to_original.get(normalized_source)
        if original_column is not None:
            column_map[original_column] = target_name

    return column_map


def _compute_sample_weights(df: pd.DataFrame) -> pd.Series:
    """Compute robust sample weights from filings and filing_rate.

    The implied renter-household proxy is filings / (filing_rate / 100). Rows with
    invalid inputs use a default weight of 1. We cap very large valid weights at
    the 99th percentile to avoid a small number of rows dominating training.
    """
    weights = pd.Series(DEFAULT_SAMPLE_WEIGHT, index=df.index, dtype="float64")

    valid_weight_mask = df["implied_renter_households"].notna() & (
        pd.to_numeric(df["implied_renter_households"], errors="coerce") > 0
    )

    invalid_weight_count = int((~valid_weight_mask).sum())
    if invalid_weight_count > 0:
        LOGGER.warning(
            "Using default sample weight for %d rows due to missing or non-positive "
            "filings/filing_rate values.",
            invalid_weight_count,
        )

    valid_weights = pd.to_numeric(
        df.loc[valid_weight_mask, "implied_renter_households"],
        errors="coerce",
    )

    if valid_weights.empty:
        return weights

    weight_cap = float(valid_weights.quantile(WEIGHT_CAP_QUANTILE))
    if weight_cap <= 0:
        return weights

    capped_weights = valid_weights.clip(upper=weight_cap)
    capped_count = int((valid_weights > weight_cap).sum())
    if capped_count > 0:
        LOGGER.warning(
            "Capped %d sample weights at the %.0fth percentile (%.2f).",
            capped_count,
            WEIGHT_CAP_QUANTILE * 100,
            weight_cap,
        )

    weights.loc[valid_weight_mask] = capped_weights
    return weights


def clean_eviction_lab_yearly(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean Eviction Lab yearly county-level data.

    Args:
        df: Raw Eviction Lab yearly dataframe.

    Returns:
        Cleaned dataframe with county_fips, year, filings, filing_rate,
        implied_renter_households, and sample_weight columns.

    Raises:
        ValueError: If required columns are missing.
    """
    column_map = _build_column_map(df.columns)
    required_targets = set(REQUIRED_YEARLY_COLUMNS.values())
    mapped_targets = set(column_map.values())

    missing_columns = sorted(required_targets - mapped_targets)
    if missing_columns:
        raise ValueError(
            "Missing required Eviction Lab columns after mapping: "
            f"{missing_columns}. Required columns are cofips, year, filings, filing_rate."
        )

    ordered_columns = ["county_fips", "year", "filings", "filing_rate"]
    cleaned_df = df.rename(columns=column_map)[ordered_columns].copy()
    cleaned_df["county_fips"] = cleaned_df["county_fips"].apply(_normalize_fips)
    cleaned_df["year"] = pd.to_numeric(cleaned_df["year"], errors="coerce")
    cleaned_df["filings"] = pd.to_numeric(cleaned_df["filings"], errors="coerce")
    cleaned_df["filing_rate"] = pd.to_numeric(cleaned_df["filing_rate"], errors="coerce")

    valid_mask = (
        cleaned_df["county_fips"].notna()
        & cleaned_df["year"].notna()
        & cleaned_df["filing_rate"].notna()
    )

    dropped_count = int((~valid_mask).sum())
    if dropped_count > 0:
        LOGGER.warning(
            "Dropped %d rows with invalid county_fips, year, or filing_rate.",
            dropped_count,
        )

    cleaned_df = cleaned_df.loc[valid_mask].copy()
    cleaned_df["year"] = cleaned_df["year"].astype(int)
    cleaned_df["implied_renter_households"] = pd.NA

    valid_implied_mask = (
        cleaned_df["filings"].notna()
        & (cleaned_df["filings"] >= 0)
        & (cleaned_df["filing_rate"] > 0)
    )
    cleaned_df.loc[valid_implied_mask, "implied_renter_households"] = (
        cleaned_df.loc[valid_implied_mask, "filings"]
        / (cleaned_df.loc[valid_implied_mask, "filing_rate"] / 100.0)
    )

    cleaned_df["sample_weight"] = _compute_sample_weights(cleaned_df)
    cleaned_df = cleaned_df.sort_values(["county_fips", "year"]).reset_index(drop=True)
    return cleaned_df
