"""Service layer for loading artifacts and scoring Eviction Lab yearly requests."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.config import (
    EVICTION_LAB_YEARLY_FEATURE_TABLE_PATH,
    EVICTION_LAB_YEARLY_METRICS_PATH,
    EVICTION_LAB_YEARLY_MODEL_METADATA_PATH,
    EVICTION_LAB_YEARLY_MODEL_PATH,
)
from src.models.eviction_lab_yearly_model import MODEL_FEATURE_COLUMNS


MODEL_TYPE = "eviction_lab_yearly_logreg"
DEFAULT_LIMITATIONS = [
    "Dataset ends at 2018 outcome year.",
    "Irregular county-year panel without gap filling.",
    "Scores depend on feature-table availability for each county-year.",
]


class ScoringServiceError(Exception):
    """Service-layer error carrying an HTTP-friendly status code."""

    def __init__(self, message: str, status_code: int) -> None:
        """Initialize the service error.

        Args:
            message: Human-readable error message.
            status_code: HTTP status code to use in API responses.
        """
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def _normalize_county_fips(county_fips: str) -> str:
    """Normalize user county FIPS input to a 5-character county code."""
    digits_only = "".join(character for character in str(county_fips).strip() if character.isdigit())
    if digits_only == "":
        raise ScoringServiceError("county_fips must include at least one digit.", 400)

    if len(digits_only) == 6 and digits_only.startswith("0"):
        digits_only = digits_only[-5:]

    if len(digits_only) > 5:
        raise ScoringServiceError(
            "county_fips must be 5 digits (or 6 with one leading zero).",
            400,
        )

    return digits_only.zfill(5)


def _read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    """Read a JSON file from disk if present, else return None."""
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


class EvictionLabScoringService:
    """Lazy-loading scoring service for Eviction Lab yearly artifacts."""

    def __init__(
        self,
        model_path: Path = EVICTION_LAB_YEARLY_MODEL_PATH,
        feature_path: Path = EVICTION_LAB_YEARLY_FEATURE_TABLE_PATH,
        metadata_path: Path = EVICTION_LAB_YEARLY_MODEL_METADATA_PATH,
        metrics_path: Path = EVICTION_LAB_YEARLY_METRICS_PATH,
    ) -> None:
        """Initialize service with artifact paths.

        Args:
            model_path: Path to trained model joblib artifact.
            feature_path: Path to processed yearly feature CSV.
            metadata_path: Path to model metadata JSON artifact.
            metrics_path: Path to metrics JSON for optional metadata summary.
        """
        self._model_path = model_path
        self._feature_path = feature_path
        self._metadata_path = metadata_path
        self._metrics_path = metrics_path

        self._cached_model: Optional[Any] = None
        self._cached_feature_df: Optional[pd.DataFrame] = None
        self._cached_metadata: Optional[Dict[str, Any]] = None

    def _load_model(self) -> Any:
        """Load and cache model artifact on first use."""
        if self._cached_model is not None:
            return self._cached_model

        if not self._model_path.exists():
            raise ScoringServiceError(
                "Model artifact not found. Run yearly training first.",
                503,
            )

        import joblib

        loaded_model = joblib.load(self._model_path)
        self._cached_model = loaded_model
        return loaded_model

    def _load_feature_table(self) -> pd.DataFrame:
        """Load and cache processed feature table used for scoring."""
        if self._cached_feature_df is not None:
            return self._cached_feature_df

        if not self._feature_path.exists():
            raise ScoringServiceError(
                "Processed yearly feature table not found. "
                "Run yearly training or backtesting to generate it.",
                503,
            )

        feature_df = pd.read_csv(self._feature_path)

        required_columns = {"county_fips", "year", *MODEL_FEATURE_COLUMNS}
        missing_columns = sorted(required_columns - set(feature_df.columns))
        if missing_columns:
            raise ScoringServiceError(
                "Processed feature table is missing required columns: "
                f"{missing_columns}.",
                500,
            )

        feature_df["county_fips"] = (
            feature_df["county_fips"].astype(str).str.strip().str.zfill(5)
        )
        feature_df["year"] = pd.to_numeric(feature_df["year"], errors="coerce")
        feature_df = feature_df.dropna(subset=["year"]).copy()
        feature_df["year"] = feature_df["year"].astype(int)

        self._cached_feature_df = feature_df
        return feature_df

    def _read_metrics_summary(self) -> Optional[Dict[str, Any]]:
        """Read a compact metrics summary from the latest metrics report."""
        metrics_payload = _read_json_if_exists(self._metrics_path)
        if metrics_payload is None:
            return None

        # Keep the summary compact and stable for API consumers.
        summary_keys = [
            "auc",
            "precision_at_0_5",
            "recall_at_0_5",
            "test_rows",
            "test_year_start",
            "test_year_end",
        ]

        summary: Dict[str, Any] = {}
        for key in summary_keys:
            if key in metrics_payload:
                summary[key] = metrics_payload[key]

        if "overall" in metrics_payload:
            summary["overall"] = metrics_payload["overall"]

        return summary or None

    def _build_fallback_metadata(self) -> Dict[str, Any]:
        """Build metadata fallback when metadata artifact is missing."""
        feature_df = self._load_feature_table()
        available_years = sorted(feature_df["year"].unique().tolist())

        if not available_years:
            raise ScoringServiceError("Feature table has no scoreable years.", 500)

        return {
            "model_version": "unknown",
            "model_type": MODEL_TYPE,
            "trained_on_dataset_name": "county_proprietary_valid_2000_2018.csv",
            "training_years": {
                "min_year": int(available_years[0]),
                "max_year": int(available_years[-1]),
            },
            "label_definition": "Top-quartile filing_rate in year t+1 across counties.",
            "feature_list": MODEL_FEATURE_COLUMNS,
            "metrics_summary": self._read_metrics_summary(),
            "limitations": DEFAULT_LIMITATIONS,
        }

    def _load_metadata(self) -> Dict[str, Any]:
        """Load and cache model metadata artifact for API responses."""
        if self._cached_metadata is not None:
            return self._cached_metadata

        metadata_payload = _read_json_if_exists(self._metadata_path)
        if metadata_payload is None:
            metadata_payload = self._build_fallback_metadata()

        if "training_years" not in metadata_payload:
            year_range = metadata_payload.get("training_feature_year_range", {})
            metadata_payload["training_years"] = {
                "min_year": year_range.get("min_year"),
                "max_year": year_range.get("max_year"),
            }

        training_years = metadata_payload.get("training_years", {})
        min_year = training_years.get("min_year")
        max_year = training_years.get("max_year")
        if min_year is None or max_year is None:
            feature_df = self._load_feature_table()
            available_years = sorted(feature_df["year"].unique().tolist())
            metadata_payload["training_years"] = {
                "min_year": int(available_years[0]),
                "max_year": int(available_years[-1]),
            }

        if metadata_payload.get("metrics_summary") in (None, {}):
            metadata_payload["metrics_summary"] = self._read_metrics_summary()

        metadata_payload.setdefault("model_type", MODEL_TYPE)
        metadata_payload.setdefault("feature_list", MODEL_FEATURE_COLUMNS)
        metadata_payload.setdefault("limitations", DEFAULT_LIMITATIONS)
        metadata_payload.setdefault(
            "label_definition",
            "Top-quartile filing_rate in year t+1 across counties.",
        )
        metadata_payload.setdefault(
            "trained_on_dataset_name",
            "county_proprietary_valid_2000_2018.csv",
        )
        metadata_payload.setdefault("model_version", "unknown")

        self._cached_metadata = metadata_payload
        return metadata_payload

    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata for API About/docs usage."""
        metadata = self._load_metadata().copy()

        return {
            "model_version": metadata["model_version"],
            "trained_on_dataset_name": metadata["trained_on_dataset_name"],
            "training_years": metadata["training_years"],
            "label_definition": metadata["label_definition"],
            "feature_list": metadata["feature_list"],
            "metrics_summary": metadata.get("metrics_summary"),
            "limitations": metadata["limitations"],
        }

    def score_county(self, county_fips: str, as_of_year: int) -> Dict[str, Any]:
        """Score one county-year request using model and processed feature table.

        Args:
            county_fips: County FIPS code input.
            as_of_year: Year of feature row to score.

        Returns:
            API-ready score response dictionary.
        """
        model = self._load_model()
        feature_df = self._load_feature_table()

        normalized_fips = _normalize_county_fips(county_fips)

        available_years = sorted(feature_df["year"].unique().tolist())
        if as_of_year not in available_years:
            raise ScoringServiceError(
                "as_of_year is not present in processed feature data. "
                f"Available years: {available_years}",
                400,
            )

        county_year_df = feature_df[
            (feature_df["county_fips"] == normalized_fips)
            & (feature_df["year"] == as_of_year)
        ].copy()

        if county_year_df.empty:
            raise ScoringServiceError(
                "No scoreable row found for the requested county_fips and as_of_year.",
                400,
            )

        if county_year_df[MODEL_FEATURE_COLUMNS].isna().any(axis=None):
            raise ScoringServiceError(
                "Requested county-year row has missing model features and cannot be scored.",
                400,
            )

        risk_score = float(model.predict_proba(county_year_df[MODEL_FEATURE_COLUMNS])[:, 1][0])
        metadata = self._load_metadata()

        return {
            "county_fips": normalized_fips,
            "as_of_year": int(as_of_year),
            "risk_score": risk_score,
            "model_version": str(metadata.get("model_version", "unknown")),
            "model_type": str(metadata.get("model_type", MODEL_TYPE)),
            "notes": None,
        }


_SERVICE_INSTANCE: Optional[EvictionLabScoringService] = None


def get_scoring_service() -> EvictionLabScoringService:
    """Return a singleton scoring service for API handlers."""
    global _SERVICE_INSTANCE
    if _SERVICE_INSTANCE is None:
        _SERVICE_INSTANCE = EvictionLabScoringService()
    return _SERVICE_INSTANCE
