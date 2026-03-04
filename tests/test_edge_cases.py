"""Edge case tests for the scoring API.

These exist because we thought "what if someone tries THIS?" — not from a spec.
"""

from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from src.api.app import app
from src.services.eviction_lab_scoring_service import ScoringServiceError


def _score_payload(county_fips="39049", as_of_year=2015, notes=None):
    return {
        "county_fips": county_fips,
        "as_of_year": as_of_year,
        "risk_score": 0.55,
        "model_version": "test-ver",
        "model_type": "eviction_lab_yearly_logreg",
        "as_of_year_available": True,
        "available_years_min": 2002,
        "available_years_max": 2017,
        "risk_percentile_in_year": 60.0,
        "features_used": {
            "lag_1": 3.5,
            "lag_3_mean_obs": 3.2,
            "lag_5_mean_obs": 3.0,
            "years_since_last_obs": 1.0,
        },
        "notes": notes,
    }


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def ok_service():
    svc = MagicMock()
    svc.score_county.return_value = _score_payload()
    return svc


def test_fips_leading_zero_round_trips(client, ok_service):
    # LA county (06037) — leading zero must survive schema normalization and come back intact
    ok_service.score_county.return_value = _score_payload(county_fips="06037")
    with patch("src.api.app.get_scoring_service", return_value=ok_service):
        resp = client.post("/score", json={"county_fips": "06037", "as_of_year": 2015})
    assert resp.status_code == 200
    assert resp.json()["county_fips"] == "06037"
    ok_service.score_county.assert_called_once_with(county_fips="06037", as_of_year=2015)


def test_fips_missing_leading_zero_gets_padded(client, ok_service):
    # Real users type "6037" not "06037" — should zero-pad before hitting the service
    ok_service.score_county.return_value = _score_payload(county_fips="06037")
    with patch("src.api.app.get_scoring_service", return_value=ok_service):
        resp = client.post("/score", json={"county_fips": "6037", "as_of_year": 2015})
    assert resp.status_code == 200
    ok_service.score_county.assert_called_once_with(county_fips="06037", as_of_year=2015)


def test_fips_with_surrounding_whitespace_normalizes(client, ok_service):
    # Copy-paste from a spreadsheet often includes trailing spaces
    with patch("src.api.app.get_scoring_service", return_value=ok_service):
        resp = client.post("/score", json={"county_fips": " 39049 ", "as_of_year": 2015})
    assert resp.status_code == 200
    ok_service.score_county.assert_called_once_with(county_fips="39049", as_of_year=2015)


def test_six_digit_fips_with_leading_zero_stripped(client, ok_service):
    # "039049" — 6 chars but the leading 0 is padding, should resolve to "39049"
    with patch("src.api.app.get_scoring_service", return_value=ok_service):
        resp = client.post("/score", json={"county_fips": "039049", "as_of_year": 2015})
    assert resp.status_code == 200
    ok_service.score_county.assert_called_once_with(county_fips="39049", as_of_year=2015)


def test_whitespace_only_fips_rejected(client):
    # "   " has no digits — schema should reject it before it ever reaches the service
    resp = client.post("/score", json={"county_fips": "   "})
    assert resp.status_code == 422


def test_year_1999_rejected_by_schema(client):
    # Eviction Lab data starts at 2000; 1999 should fail Pydantic's ge=2000 constraint
    resp = client.post("/score", json={"county_fips": "39049", "as_of_year": 1999})
    assert resp.status_code == 422


def test_year_2101_above_schema_max_rejected(client):
    # le=2100 on as_of_year — 2101 should be caught by schema, not service
    resp = client.post("/score", json={"county_fips": "39049", "as_of_year": 2101})
    assert resp.status_code == 422


def test_year_2100_at_boundary_reaches_service(client, ok_service):
    # 2100 is the schema max — it should pass validation and reach the service (even if service returns 400)
    ok_service.score_county.side_effect = ScoringServiceError(
        "as_of_year is not available for this county. Try a year in [2002, 2017].", 400
    )
    with patch("src.api.app.get_scoring_service", return_value=ok_service):
        resp = client.post("/score", json={"county_fips": "39049", "as_of_year": 2100})
    assert resp.status_code == 400
    # confirm schema didn't swallow the call
    ok_service.score_county.assert_called_once_with(county_fips="39049", as_of_year=2100)


def test_year_after_dataset_end_returns_400_not_500(client, ok_service):
    # 2025 is post-2018 dataset cutoff — valid schema, but service says unavailable
    ok_service.score_county.side_effect = ScoringServiceError(
        "as_of_year is not available for this county. Try a year in [2002, 2017].", 400
    )
    with patch("src.api.app.get_scoring_service", return_value=ok_service):
        resp = client.post("/score", json={"county_fips": "39049", "as_of_year": 2025})
    assert resp.status_code == 400


def test_omitting_year_calls_service_with_none(client, ok_service):
    # No as_of_year in the payload — service should be called with None, not blow up
    ok_service.score_county.return_value = _score_payload(
        notes="as_of_year was not provided. Used latest available year (2017)."
    )
    with patch("src.api.app.get_scoring_service", return_value=ok_service):
        resp = client.post("/score", json={"county_fips": "39049"})
    assert resp.status_code == 200
    ok_service.score_county.assert_called_once_with(county_fips="39049", as_of_year=None)
    assert "2017" in resp.json()["notes"]


def test_batch_empty_list_returns_200_with_empty_array(client, ok_service):
    # /score/batch with [] — should return an empty array, not a 422 or 500
    with patch("src.api.app.get_scoring_service", return_value=ok_service):
        resp = client.post("/score/batch", json=[])
    assert resp.status_code == 200
    assert resp.json() == []


def test_year_as_numeric_string_behavior(client, ok_service):
    # Pydantic v2 lax mode coerces "2015" (str) → 2015 (int) — document that it doesn't 500
    with patch("src.api.app.get_scoring_service", return_value=ok_service):
        resp = client.post("/score", json={"county_fips": "39049", "as_of_year": "2015"})
    # coerced to int in lax mode → 200; strict mode would give 422
    assert resp.status_code in (200, 422)
