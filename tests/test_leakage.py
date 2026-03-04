"""Unit tests for the temporal leakage assertion guard."""

import pytest
import pandas as pd

from src.validation.leakage import assert_no_temporal_leakage


def test_clean_data_passes_silently():
    """Clean data where year < outcome_year for every row raises no error."""
    df = pd.DataFrame({
        "year": [2010, 2011, 2012],
        "outcome_year": [2011, 2012, 2013],
    })
    assert_no_temporal_leakage(df)  # must not raise


def test_leakage_equal_years_raises():
    """A row with year == outcome_year triggers ValueError with the leaked row count."""
    df = pd.DataFrame({
        "year": [2010, 2012, 2013],
        "outcome_year": [2011, 2012, 2014],  # middle row: 2012 == 2012 → leakage
    })
    with pytest.raises(ValueError, match="1 row"):
        assert_no_temporal_leakage(df)


def test_leakage_future_year_raises():
    """A row with year > outcome_year (future features) triggers ValueError."""
    df = pd.DataFrame({
        "year": [2015, 2016],
        "outcome_year": [2014, 2015],  # year > outcome_year for both rows
    })
    with pytest.raises(ValueError, match="2 row"):
        assert_no_temporal_leakage(df)


def test_error_message_contains_sample():
    """The ValueError message includes a sample of offending rows."""
    df = pd.DataFrame({
        "year": [2020],
        "outcome_year": [2019],
    })
    with pytest.raises(ValueError, match="outcome_year"):
        assert_no_temporal_leakage(df)
