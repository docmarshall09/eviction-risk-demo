"""Pydantic request/response schemas for the eviction-risk API."""

from typing import Any, Dict, List, Optional

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
    as_of_year: int = Field(
        ...,
        ge=2000,
        le=2100,
        description="Feature year to score.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
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
    notes: Optional[str] = Field(None, description="Optional scoring note.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "county_fips": "39049",
                "as_of_year": 2017,
                "risk_score": 0.7421,
                "model_version": "20260223T160102Z_ab12cd3",
                "model_type": "eviction_lab_yearly_logreg",
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
    feature_list: List[str]
    metrics_summary: Optional[Dict[str, Any]]
    limitations: List[str]

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
            }
        }
    )
