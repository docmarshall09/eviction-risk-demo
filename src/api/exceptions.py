"""Custom API exception types for clean, sanitized HTTP error responses."""


class CountyNotFoundError(Exception):
    """Raised when county_fips does not exist in the feature table."""

    def __init__(self, county_fips: str) -> None:
        super().__init__(f"County not found: {county_fips}")
        self.county_fips = county_fips


class ScoringError(Exception):
    """Raised when county scoring fails due to an unresolvable input state."""

    pass
