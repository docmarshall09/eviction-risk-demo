"""Smoke tests for the FastAPI scoring API.

Run from the repository root:
    pytest tests/ -v

All tests use FastAPI's TestClient — no subprocess server is started.
The scoring service is mocked so no model artifacts are required on disk.
"""

import pytest
from unittest.mock import MagicMock, patch
from starlette.testclient import TestClient

from src.api.app import app
from src.services.eviction_lab_scoring_service import ScoringServiceError


_MOCK_METADATA = {
    "model_version": "20260101T000000Z_abc1234",
    "trained_on_dataset_name": "county_proprietary_valid_2000_2018.csv",
    "training_years": {"min_year": 2002, "max_year": 2017},
    "label_definition": "Top-quartile filing_rate in year t+1 across counties.",
    "feature_list": ["lag_1", "lag_3_mean_obs", "lag_5_mean_obs", "years_since_last_obs"],
    "metrics_summary": {"auc": 0.73},
    "limitations": ["Dataset ends at 2018 outcome year."],
    "intercept": -1.0,
    "coefficients": {
        "lag_1": 0.5,
        "lag_3_mean_obs": 0.3,
        "lag_5_mean_obs": -0.1,
        "years_since_last_obs": -0.4,
    },
    "feature_order": ["lag_1", "lag_3_mean_obs", "lag_5_mean_obs", "years_since_last_obs"],
    "calibration_params": None,
    "scaler_params": None,
    "provenance": {
        "git_sha": "abc1234",
        "trained_at_utc": "2026-01-01T00:00:00+00:00",
        "python_version": "3.11.0",
        "sklearn_version": "1.8.0",
        "pandas_version": "3.0.1",
        "numpy_version": "2.4.2",
    },
}

_MOCK_SCORE = {
    "county_fips": "39049",
    "as_of_year": 2015,
    "risk_score": 0.65,
    "model_version": "20260101T000000Z_abc1234",
    "model_type": "eviction_lab_yearly_logreg",
    "as_of_year_available": True,
    "available_years_min": 2002,
    "available_years_max": 2017,
    "risk_percentile_in_year": 75.0,
    "features_used": {
        "lag_1": 4.2,
        "lag_3_mean_obs": 3.9,
        "lag_5_mean_obs": 3.5,
        "years_since_last_obs": 1.0,
    },
    "notes": None,
}


@pytest.fixture()
def client():
    """Return a TestClient bound to the FastAPI app."""
    return TestClient(app)


@pytest.fixture()
def mock_service():
    """Return a mock EvictionLabScoringService with canned responses."""
    service = MagicMock()
    service.get_metadata.return_value = _MOCK_METADATA
    service.score_county.return_value = _MOCK_SCORE
    return service


def test_health(client):
    """GET /health returns 200 with {status: ok} — validates basic app availability."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metadata(client, mock_service):
    """GET /metadata returns 200 and includes model_version — validates metadata endpoint shape."""
    with patch("src.api.app.get_scoring_service", return_value=mock_service):
        response = client.get("/metadata")
    assert response.status_code == 200
    body = response.json()
    assert "model_version" in body
    assert body["model_version"] == _MOCK_METADATA["model_version"]
    assert "training_years" in body
    assert "feature_list" in body


def test_score_valid(client, mock_service):
    """POST /score with county_fips=39049, as_of_year=2015 returns 200 with risk_score in [0, 1]."""
    with patch("src.api.app.get_scoring_service", return_value=mock_service):
        response = client.post(
            "/score", json={"county_fips": "39049", "as_of_year": 2015}
        )
    assert response.status_code == 200
    body = response.json()
    assert "risk_score" in body
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["county_fips"] == "39049"
    assert body["as_of_year"] == 2015
    mock_service.score_county.assert_called_once_with(county_fips="39049", as_of_year=2015)


def test_score_invalid_fips(client, mock_service):
    """POST /score with an unknown FIPS returns 404 — validates service-layer 404 propagation."""
    mock_service.score_county.side_effect = ScoringServiceError(
        "Unknown county_fips '00000'.", 404
    )
    with patch("src.api.app.get_scoring_service", return_value=mock_service):
        response = client.post("/score", json={"county_fips": "00000"})
    assert response.status_code == 404


def test_score_malformed_input(client):
    """POST /score with no digits in county_fips returns 422 — validates Pydantic input rejection."""
    response = client.post("/score", json={"county_fips": "AAAA"})
    assert response.status_code == 422


def test_metadata_provenance(client, mock_service):
    """GET /metadata response includes a provenance block with all 6 required keys."""
    with patch("src.api.app.get_scoring_service", return_value=mock_service):
        response = client.get("/metadata")
    assert response.status_code == 200
    body = response.json()
    assert "provenance" in body
    provenance = body["provenance"]
    for key in ("git_sha", "trained_at_utc", "python_version", "sklearn_version", "pandas_version", "numpy_version"):
        assert key in provenance, f"provenance missing key: {key}"
