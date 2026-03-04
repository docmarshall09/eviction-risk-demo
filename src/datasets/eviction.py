"""Load and validate county-level eviction filing time-series data."""

import logging
from pathlib import Path
import pandas as pd


LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "county_fips": "county_fips",
    "month": "month",
    "eviction_filing_rate": "eviction_filing_rate",
}


def _normalize_column_name(column_name: str) -> str:
    """Return a normalized column key for case-insensitive matching."""
    return "".join(character for character in column_name.lower() if character.isalnum())


def _build_column_map(columns: pd.Index) -> dict[str, str]:
    """Map raw input columns to canonical required names."""
    normalized_to_original = {
        _normalize_column_name(str(column)): str(column) for column in columns
    }

    required_lookup = {
        _normalize_column_name(column_name): canonical_name
        for canonical_name, column_name in REQUIRED_COLUMNS.items()
    }

    column_map: dict[str, str] = {}
    for normalized_required, canonical_name in required_lookup.items():
        if normalized_required in normalized_to_original:
            column_map[normalized_to_original[normalized_required]] = canonical_name

    return column_map


def _normalize_fips(value: object) -> str | None:
    """Normalize a county FIPS value to a zero-padded 5-digit string."""
    if pd.isna(value):
        return None

    text_value = str(value).strip()
    if text_value == "":
        return None

    # Many CSV exports serialize identifiers as numbers like "39049.0".
    if text_value.replace(".", "", 1).isdigit() and "." in text_value:
        try:
            text_value = str(int(float(text_value)))
        except ValueError:
            return None

    digits_only = "".join(character for character in text_value if character.isdigit())
    if digits_only == "" or len(digits_only) > 5:
        return None

    return digits_only.zfill(5)


def load_raw_eviction_data(path: str) -> pd.DataFrame:
    """Load raw eviction data from CSV.

    Args:
        path: Path to the raw CSV file.

    Returns:
        A raw dataframe from CSV.

    Raises:
        FileNotFoundError: If the CSV path does not exist.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            "Raw eviction CSV not found at "
            f"{csv_path}. Add data/raw/eviction_county_month.csv with columns "
            "county_fips, month, eviction_filing_rate."
        )

    return pd.read_csv(csv_path)


def validate_and_clean_eviction_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate required columns and return a cleaned eviction dataframe.

    This function keeps only required fields and drops invalid rows. It logs how
    many rows were dropped so data quality issues are visible to users.

    Args:
        df: Raw eviction dataframe.

    Returns:
        A cleaned dataframe sorted by county_fips and month.

    Raises:
        ValueError: If one or more required columns are missing.
    """
    column_map = _build_column_map(df.columns)
    required_output_columns = set(REQUIRED_COLUMNS.keys())
    mapped_output_columns = set(column_map.values())

    missing_columns = sorted(required_output_columns - mapped_output_columns)
    if missing_columns:
        raise ValueError(
            "Missing required columns after case-insensitive mapping: "
            f"{missing_columns}. Required columns are county_fips, month, "
            "eviction_filing_rate."
        )

    cleaned_df = df.rename(columns=column_map)[list(REQUIRED_COLUMNS.keys())].copy()

    cleaned_df["county_fips"] = cleaned_df["county_fips"].apply(_normalize_fips)

    parsed_month = pd.to_datetime(cleaned_df["month"], errors="coerce")
    cleaned_df["month"] = parsed_month.dt.to_period("M").dt.to_timestamp()

    cleaned_df["eviction_filing_rate"] = pd.to_numeric(
        cleaned_df["eviction_filing_rate"], errors="coerce"
    )

    valid_mask = (
        cleaned_df["county_fips"].notna()
        & cleaned_df["month"].notna()
        & cleaned_df["eviction_filing_rate"].notna()
    )

    dropped_count = int((~valid_mask).sum())
    if dropped_count > 0:
        LOGGER.warning(
            "Dropped %d rows with invalid county_fips, month, or eviction_filing_rate.",
            dropped_count,
        )

    cleaned_df = cleaned_df.loc[valid_mask].copy()
    cleaned_df = cleaned_df.sort_values(["county_fips", "month"]).reset_index(drop=True)
    return cleaned_df
