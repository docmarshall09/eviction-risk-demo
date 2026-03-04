"""Service layer for loading artifacts and scoring Eviction Lab yearly requests."""

import json
from pathlib import Path
from typing import Any

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

    def __init__(
        self,
        message: str,
        status_code: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the service error."""
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.details = details or {}

    def to_detail(self) -> dict[str, Any]:
        """Return structured detail payload for HTTP error responses."""
        payload = {"message": self.message}
        payload.update(self.details)
        return payload


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


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    """Read a JSON file from disk if present, else return None."""
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _to_float_list(values: Any) -> list[float]:
    """Convert an array-like object to a list of floats."""
    if values is None:
        return []
    try:
        return [float(value) for value in list(values)]
    except TypeError:
        return []


def _candidate_estimators(model: Any) -> list[Any]:
    """Collect likely estimator objects for metadata extraction."""
    candidates: list[Any] = [model]

    direct_estimator = getattr(model, "estimator", None)
    if direct_estimator is not None:
        candidates.append(direct_estimator)
        nested_estimator = getattr(direct_estimator, "estimator", None)
        if nested_estimator is not None:
            candidates.append(nested_estimator)

    calibrated_estimators = getattr(model, "calibrated_classifiers_", [])
    if calibrated_estimators:
        first_calibrated = calibrated_estimators[0]
        calibrated_estimator = getattr(first_calibrated, "estimator", None)
        if calibrated_estimator is not None:
            candidates.append(calibrated_estimator)
            nested_estimator = getattr(calibrated_estimator, "estimator", None)
            if nested_estimator is not None:
                candidates.append(nested_estimator)

    return candidates


def _final_step_if_pipeline(estimator: Any) -> Any:
    """Return the final model step for pipeline-like estimators."""
    named_steps = getattr(estimator, "named_steps", None)
    if named_steps is None:
        return estimator

    if not named_steps:
        return estimator

    step_names = list(named_steps.keys())
    last_step_name = step_names[-1]
    return named_steps[last_step_name]


def _extract_linear_model_params(model: Any) -> dict[str, Any]:
    """Extract intercept/coefficients/feature order from the loaded model."""
    intercept: float | None = None
    coefficients: dict[str, float] | None = None

    feature_order: list[str] = []
    if hasattr(model, "feature_names_in_"):
        feature_order = [str(name) for name in list(model.feature_names_in_)]
    else:
        feature_order = list(MODEL_FEATURE_COLUMNS)

    for candidate in _candidate_estimators(model):
        estimator = _final_step_if_pipeline(candidate)
        coef_values = getattr(estimator, "coef_", None)
        intercept_values = getattr(estimator, "intercept_", None)

        if coef_values is None or intercept_values is None:
            continue

        intercept_list = _to_float_list(intercept_values)
        if intercept_list:
            intercept = float(intercept_list[0])

        coef_rows = list(coef_values)
        if not coef_rows:
            continue

        coef_vector = _to_float_list(coef_rows[0])
        if not coef_vector:
            continue

        if len(feature_order) != len(coef_vector):
            feature_order = list(MODEL_FEATURE_COLUMNS)

        coefficients = {
            feature_name: float(coef_vector[index])
            for index, feature_name in enumerate(feature_order)
            if index < len(coef_vector)
        }
        break

    return {
        "intercept": intercept,
        "coefficients": coefficients,
        "feature_order": feature_order,
    }


def _extract_calibration_params(model: Any) -> dict[str, Any] | None:
    """Extract calibration details from calibrated models when present."""
    calibrated_estimators = getattr(model, "calibrated_classifiers_", [])
    if not calibrated_estimators:
        return None

    first_calibrated = calibrated_estimators[0]
    calibration_method = str(
        getattr(model, "calibration_method", getattr(first_calibrated, "method", "unknown"))
    )
    payload: dict[str, Any] = {"method": calibration_method}

    calibration_time_values = getattr(model, "calibration_time_values", None)
    if calibration_time_values is not None:
        payload["calibration_time_values"] = list(calibration_time_values)

    calibrators = getattr(first_calibrated, "calibrators", None)
    if calibrators and len(calibrators) > 0:
        calibrator = calibrators[0]

        if hasattr(calibrator, "a_") and hasattr(calibrator, "b_"):
            payload["params"] = {
                "a": float(getattr(calibrator, "a_")),
                "b": float(getattr(calibrator, "b_")),
            }
            return payload

        if hasattr(calibrator, "X_thresholds_") and hasattr(calibrator, "y_thresholds_"):
            x_thresholds = _to_float_list(getattr(calibrator, "X_thresholds_"))
            y_thresholds = _to_float_list(getattr(calibrator, "y_thresholds_"))
            if x_thresholds and y_thresholds:
                payload["params"] = {
                    "n_thresholds": len(x_thresholds),
                    "x_min": float(min(x_thresholds)),
                    "x_max": float(max(x_thresholds)),
                    "y_min": float(min(y_thresholds)),
                    "y_max": float(max(y_thresholds)),
                    "out_of_bounds": getattr(calibrator, "out_of_bounds", None),
                }
                return payload

    return payload


def _extract_scaler_params(model: Any) -> dict[str, Any] | None:
    """Extract scaler parameters when a scaler is present in the model stack."""
    for candidate in _candidate_estimators(model):
        pipeline_steps = getattr(candidate, "named_steps", None)
        if pipeline_steps is not None:
            for step_name, step_value in pipeline_steps.items():
                mean_values = _to_float_list(getattr(step_value, "mean_", None))
                scale_values = _to_float_list(getattr(step_value, "scale_", None))
                var_values = _to_float_list(getattr(step_value, "var_", None))
                if not mean_values and not scale_values and not var_values:
                    continue
                return {
                    "step_name": str(step_name),
                    "class_name": step_value.__class__.__name__,
                    "mean": mean_values or None,
                    "scale": scale_values or None,
                    "var": var_values or None,
                }

        mean_values = _to_float_list(getattr(candidate, "mean_", None))
        scale_values = _to_float_list(getattr(candidate, "scale_", None))
        var_values = _to_float_list(getattr(candidate, "var_", None))
        if not mean_values and not scale_values and not var_values:
            continue
        return {
            "class_name": candidate.__class__.__name__,
            "mean": mean_values or None,
            "scale": scale_values or None,
            "var": var_values or None,
        }

    return None


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

        self._cached_model: Any | None = None
        self._cached_feature_df: pd.DataFrame | None = None
        self._cached_metadata: dict[str, Any] | None = None
        self._cached_year_scores: dict[int, pd.Series] = {}

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

    def _read_metrics_summary(self) -> dict[str, Any] | None:
        """Read a compact metrics summary from the latest metrics report."""
        metrics_payload = _read_json_if_exists(self._metrics_path)
        if metrics_payload is None:
            return None

        summary_keys = [
            "auc",
            "precision_at_0_5",
            "recall_at_0_5",
            "test_rows",
            "test_year_start",
            "test_year_end",
        ]

        summary: dict[str, Any] = {}
        for key in summary_keys:
            if key in metrics_payload:
                summary[key] = metrics_payload[key]

        if "overall" in metrics_payload:
            summary["overall"] = metrics_payload["overall"]

        return summary or None

    def _build_fallback_metadata(self) -> dict[str, Any]:
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

    def _load_metadata(self) -> dict[str, Any]:
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

    def _get_year_score_distribution(self, as_of_year: int) -> pd.Series:
        """Return cached year-level score distribution for percentile context."""
        if as_of_year in self._cached_year_scores:
            return self._cached_year_scores[as_of_year]

        model = self._load_model()
        feature_df = self._load_feature_table()

        year_df = feature_df[feature_df["year"] == as_of_year].copy()
        year_df = year_df.dropna(subset=MODEL_FEATURE_COLUMNS)

        if year_df.empty:
            raise ScoringServiceError(
                f"No scoreable rows found for as_of_year={as_of_year}.",
                400,
                details={"as_of_year_available": False},
            )

        try:
            year_scores = pd.Series(
                model.predict_proba(year_df[MODEL_FEATURE_COLUMNS])[:, 1],
                dtype="float64",
            )
        except (KeyError, ValueError, IndexError) as exc:
            raise ScoringServiceError(
                f"Model inference failed for as_of_year={as_of_year}.", 500
            ) from exc
        self._cached_year_scores[as_of_year] = year_scores
        return year_scores

    def get_metadata(self) -> dict[str, Any]:
        """Return metadata for API About/docs usage."""
        metadata = self._load_metadata().copy()
        model = self._load_model()
        linear_params = _extract_linear_model_params(model)
        calibration_params = _extract_calibration_params(model)
        scaler_params = _extract_scaler_params(model)

        return {
            "model_version": metadata["model_version"],
            "trained_on_dataset_name": metadata["trained_on_dataset_name"],
            "training_years": metadata["training_years"],
            "label_definition": metadata["label_definition"],
            "feature_list": metadata["feature_list"],
            "metrics_summary": metadata.get("metrics_summary"),
            "limitations": metadata["limitations"],
            "intercept": linear_params.get("intercept"),
            "coefficients": linear_params.get("coefficients"),
            "feature_order": linear_params.get("feature_order"),
            "calibration_params": calibration_params,
            "scaler_params": scaler_params,
            "provenance": metadata.get("provenance"),
        }

    def _get_county_feature_rows(
        self,
        feature_df: pd.DataFrame,
        county_fips: str,
    ) -> pd.DataFrame:
        """Return all feature rows for one county."""
        county_df = feature_df[feature_df["county_fips"] == county_fips].copy()
        if county_df.empty:
            raise ScoringServiceError(
                f"Unknown county_fips '{county_fips}'.",
                404,
                details={
                    "county_fips": county_fips,
                    "as_of_year_available": False,
                    "available_years_min": None,
                    "available_years_max": None,
                },
            )
        return county_df

    def _get_scoreable_county_feature_rows(self, county_df: pd.DataFrame) -> pd.DataFrame:
        """Return county rows that have complete model features."""
        scoreable_df = county_df.dropna(subset=MODEL_FEATURE_COLUMNS).copy()
        if scoreable_df.empty:
            county_fips = str(county_df["county_fips"].iloc[0])
            raise ScoringServiceError(
                (
                    f"County '{county_fips}' exists but has no scoreable years in the "
                    "processed feature table."
                ),
                422,
                details={
                    "county_fips": county_fips,
                    "as_of_year_available": False,
                    "available_years_min": None,
                    "available_years_max": None,
                },
            )
        return scoreable_df

    def _get_available_year_bounds(
        self,
        county_scoreable_df: pd.DataFrame,
    ) -> tuple[int, int]:
        """Get min and max available scoreable years for one county."""
        min_year = int(county_scoreable_df["year"].min())
        max_year = int(county_scoreable_df["year"].max())
        return min_year, max_year

    def _resolve_as_of_year(
        self,
        county_fips: str,
        requested_as_of_year: int | None,
        available_years: list[int],
    ) -> tuple[int, str | None]:
        """Resolve the feature year to use for scoring.

        Args:
            county_fips: Normalized county code used in responses.
            requested_as_of_year: User-supplied year, if provided.
            available_years: Sorted scoreable years for this county.

        Returns:
            Tuple of (resolved_year, optional_note).

        Raises:
            ScoringServiceError: If a requested year is not scoreable.
        """
        min_year = int(available_years[0])
        max_year = int(available_years[-1])

        if requested_as_of_year is None:
            return max_year, (
                f"as_of_year was not provided. Used latest available year ({max_year})."
            )

        if requested_as_of_year in set(available_years):
            return int(requested_as_of_year), None

        raise ScoringServiceError(
            (
                "as_of_year is not available for this county. "
                "This API returns an error when an unavailable as_of_year is provided. "
                f"Try a year in [{min_year}, {max_year}]."
            ),
            400,
            details={
                "county_fips": county_fips,
                "as_of_year": int(requested_as_of_year),
                "as_of_year_available": False,
                "available_years_min": min_year,
                "available_years_max": max_year,
            },
        )

    def score_county(
        self,
        county_fips: str,
        as_of_year: int | None,
    ) -> dict[str, Any]:
        """Score one county-year request using model and processed feature table.

        Args:
            county_fips: County FIPS code input.
            as_of_year: Year of feature row to score. If None, latest is used.

        Returns:
            API-ready score response dictionary.
        """
        model = self._load_model()
        feature_df = self._load_feature_table()

        normalized_fips = _normalize_county_fips(county_fips)
        county_df = self._get_county_feature_rows(feature_df, normalized_fips)
        county_scoreable_df = self._get_scoreable_county_feature_rows(county_df)
        available_years = sorted(county_scoreable_df["year"].unique().tolist())
        available_years_min, available_years_max = self._get_available_year_bounds(
            county_scoreable_df
        )
        resolved_as_of_year, note = self._resolve_as_of_year(
            county_fips=normalized_fips,
            requested_as_of_year=as_of_year,
            available_years=available_years,
        )

        county_year_df = county_scoreable_df[
            county_scoreable_df["year"] == resolved_as_of_year
        ].copy()
        if county_year_df.empty:
            raise ScoringServiceError(
                "Requested county-year is not scoreable in the processed feature table.",
                400,
                details={
                    "county_fips": normalized_fips,
                    "as_of_year": int(resolved_as_of_year),
                    "as_of_year_available": False,
                    "available_years_min": available_years_min,
                    "available_years_max": available_years_max,
                },
            )

        try:
            risk_score = float(
                model.predict_proba(county_year_df[MODEL_FEATURE_COLUMNS])[:, 1][0]
            )
        except (KeyError, ValueError, IndexError) as exc:
            raise ScoringServiceError(
                f"Model inference failed for county '{normalized_fips}'.", 500
            ) from exc

        year_scores = self._get_year_score_distribution(resolved_as_of_year)
        risk_percentile = float((year_scores <= risk_score).mean() * 100.0)

        feature_row = county_year_df.iloc[0]
        features_used = {
            "lag_1": float(feature_row["lag_1"]),
            "lag_3_mean_obs": float(feature_row["lag_3_mean_obs"]),
            "lag_5_mean_obs": float(feature_row["lag_5_mean_obs"]),
            "years_since_last_obs": float(feature_row["years_since_last_obs"]),
        }

        metadata = self._load_metadata()

        return {
            "county_fips": normalized_fips,
            "as_of_year": int(resolved_as_of_year),
            "risk_score": risk_score,
            "model_version": str(metadata.get("model_version", "unknown")),
            "model_type": str(metadata.get("model_type", MODEL_TYPE)),
            "as_of_year_available": True,
            "available_years_min": available_years_min,
            "available_years_max": available_years_max,
            "risk_percentile_in_year": risk_percentile,
            "features_used": features_used,
            "notes": note,
        }


_SERVICE_INSTANCE: EvictionLabScoringService | None = None


def get_scoring_service() -> EvictionLabScoringService:
    """Return a singleton scoring service for API handlers."""
    global _SERVICE_INSTANCE
    if _SERVICE_INSTANCE is None:
        _SERVICE_INSTANCE = EvictionLabScoringService()
    return _SERVICE_INSTANCE
