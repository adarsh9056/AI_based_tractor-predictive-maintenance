"""API tests (require trained artifacts under backend/)."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND = Path(__file__).resolve().parent.parent / "backend"

pytestmark = pytest.mark.skipif(
    not (BACKEND / "model.pkl").is_file(),
    reason="Run python scripts/train_model.py to create backend/model.pkl",
)


def test_health() -> None:
    from app import app

    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_model_info() -> None:
    from app import app

    c = TestClient(app)
    r = c.get("/model_info")
    assert r.status_code == 200
    body = r.json()
    assert "feature_names" in body
    assert "power_kw" in body["feature_names"]
    assert body.get("pwf_reference_band_kw") == [3.5, 9.0]


def test_predict_validation_rejects_out_of_range() -> None:
    from app import app

    c = TestClient(app)
    r = c.post(
        "/predict",
        json={
            "air_temp": 100.0,
            "process_temp": 308.0,
            "rpm": 1500.0,
            "torque": 35.0,
            "tool_wear": 50.0,
        },
    )
    assert r.status_code == 422


def test_predict_ok_includes_explanation() -> None:
    from app import app

    c = TestClient(app)
    r = c.post(
        "/predict",
        json={
            "air_temp": 298.0,
            "process_temp": 308.0,
            "rpm": 1500.0,
            "torque": 35.0,
            "tool_wear": 50.0,
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert "prediction" in data
    assert "explanation_factors" in data
    assert isinstance(data["explanation_factors"], list)
    assert "derived" in data
    assert "power_kw" in data["derived"]


def test_logs_pagination_shape() -> None:
    from app import app

    c = TestClient(app)
    r = c.get("/logs", params={"limit": 5, "offset": 0})
    assert r.status_code == 200
    data = r.json()
    assert "items" in data
    assert "total" in data
    assert "limit" in data
    assert isinstance(data["items"], list)
