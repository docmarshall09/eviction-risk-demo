"""Pydantic request/response schemas for the eviction-risk API."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _normalize_fips_for_api(value: str) -> str:
    """Normalize an input FIPS value to a 5-digit county code string.

    The API accepts both "39049" and "039049" for convenience.
    """
    digits_only = "".join(character for character in str(value).strip() if character.isdigit())
    if digits_only == "":
        raise ValueError("county_fips must include at least one digit.")

    if len(digits_only) == 6 and digits_only.startswith("0"):
        digits_only = digits_only[-5:]

    if len(digits_only) > 5:
        raise ValueError("county_fips must be at most 5 digits after normalization.")

    return digits_only.zfill(5)


class ScoreRequest(BaseModel):
    """Request schema for county-level risk scoring."""

    county_fips: str = Field(..., description="County FIPS code.")
    as_of_year: int | None = Field(
        default=None,
        ge=2000,
        le=2100,
        description="Feature year to score. If omitted, latest available year is used.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"county_fips": "39049"},
                {"county_fips": "39049", "as_of_year": 2017},
                {"county_fips": "039049", "as_of_year": 2017},
            ]
        }
    )

    @field_validator("county_fips")
    @classmethod
    def validate_county_fips(cls, value: str) -> str:
        """Validate and normalize county_fips to a 5-character string."""
        return _normalize_fips_for_api(value)


class ScoreResponse(BaseModel):
    """Response schema for county-level risk scoring."""

    county_fips: str = Field(..., description="Normalized 5-digit county FIPS code.")
    as_of_year: int = Field(..., description="Feature year used for scoring.")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Predicted risk score.")
    model_version: str = Field(..., description="Trained model version identifier.")
    model_type: str = Field(..., description="Model type identifier.")
    as_of_year_available: bool | None = Field(
        None,
        description="Whether requested as_of_year exists for this county in the feature table.",
    )
    available_years_min: int | None = Field(
        None,
        description="Minimum available feature year for this county, when known.",
    )
    available_years_max: int | None = Field(
        None,
        description="Maximum available feature year for this county, when known.",
    )
    risk_percentile_in_year: float | None = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Percentile rank of risk_score among scoreable counties in that year.",
    )
    features_used: dict[str, float] | None = Field(
        None,
        description="Model feature values used for this county-year score.",
    )
    notes: str | None = Field(None, description="Optional scoring note.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "county_fips": "39049",
                "as_of_year": 2017,
                "risk_score": 0.7421,
                "model_version": "20260223T160102Z_ab12cd3",
                "model_type": "eviction_lab_yearly_logreg",
                "as_of_year_available": True,
                "available_years_min": 2004,
                "available_years_max": 2017,
                "risk_percentile_in_year": 93.8,
                "features_used": {
                    "lag_1": 4.21,
                    "lag_3_mean_obs": 3.98,
                    "lag_5_mean_obs": 3.44,
                    "years_since_last_obs": 1.0,
                },
                "notes": None,
            }
        }
    )


class TrainingYears(BaseModel):
    """Training year range summary used in metadata responses."""

    min_year: int
    max_year: int


class MetadataResponse(BaseModel):
    """Response schema for model metadata and limitations."""

    model_version: str
    trained_on_dataset_name: str
    training_years: TrainingYears
    label_definition: str
    feature_list: list[str]
    metrics_summary: dict[str, Any] | None
    limitations: list[str]
    intercept: float | None = None
    coefficients: dict[str, float] | None = None
    feature_order: list[str] | None = None
    calibration_params: dict[str, Any] | None = None
    scaler_params: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_version": "20260223T160102Z_ab12cd3",
                "trained_on_dataset_name": "county_proprietary_valid_2000_2018.csv",
                "training_years": {"min_year": 2002, "max_year": 2017},
                "label_definition": "Top-quartile filing_rate in year t+1 across counties.",
                "feature_list": [
                    "lag_1",
                    "lag_3_mean_obs",
                    "lag_5_mean_obs",
                    "years_since_last_obs",
                ],
                "metrics_summary": {
                    "auc": 0.73,
                    "precision_at_0_5": 0.41,
                    "recall_at_0_5": 0.22,
                },
                "limitations": [
                    "Dataset ends at 2018 outcome year.",
                    "Irregular county-year panel without gap filling.",
                ],
                "intercept": -10.8238,
                "coefficients": {
                    "lag_1": 2.6394,
                    "lag_3_mean_obs": 0.9750,
                    "lag_5_mean_obs": -0.1943,
                    "years_since_last_obs": -0.4806,
                },
                "feature_order": [
                    "lag_1",
                    "lag_3_mean_obs",
                    "lag_5_mean_obs",
                    "years_since_last_obs",
                ],
                "calibration_params": {
                    "method": "sigmoid",
                    "params": {"a": -4.32, "b": 2.11},
                },
                "scaler_params": None,
            }
        }
    )
