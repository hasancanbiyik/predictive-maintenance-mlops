"""
API contract tests for the FastAPI serving layer.

Strategy: we override `get_model_bundle` with a fake bundle so these tests
don't depend on MLflow being set up or a model being trained. The point of
these tests is the *contract* (schema validation, response shape, status
codes), not whether the real model produces the right numbers.

There's a separate integration test (the MLflow logging one in Phase 2) that
exercises a real trained model end-to-end.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.serving import app as app_module
from src.serving.app import ModelBundle, app, get_model_bundle


class _FakeClassifier:
    """Sklearn-shaped stub: returns 0.42 probability for everything."""

    def predict_proba(self, X):
        n = len(X)
        return np.array([[0.58, 0.42]] * n)


@pytest.fixture
def client():
    bundle = ModelBundle(model=_FakeClassifier(), name="predictive_maintenance", version="9", alias="test")
    app.dependency_overrides[get_model_bundle] = lambda: bundle
    yield TestClient(app)
    app.dependency_overrides.clear()
    # Bust the lru_cache between tests in case anything triggered it.
    app_module._clear_model_cache()


VALID_PAYLOAD = {
    "type": "L",
    "air_temp_k": 298.1,
    "process_temp_k": 308.6,
    "rot_speed_rpm": 1551,
    "torque_nm": 42.8,
    "tool_wear_min": 0,
}


def test_health_ok_when_model_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_name"] == "predictive_maintenance"
    assert body["model_version"] == "9"
    assert body["model_alias"] == "test"


def test_predict_returns_prediction_and_probability(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200, r.text
    body = r.json()
    # 0.42 < 0.5 threshold => prediction 0
    assert body["prediction"] == 0
    assert body["probability"] == pytest.approx(0.42, abs=1e-6)
    assert body["model_name"] == "predictive_maintenance"
    assert body["model_version"] == "9"
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_rejects_missing_field(client):
    bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "torque_nm"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
    assert "torque_nm" in r.text


def test_predict_rejects_invalid_type_enum(client):
    bad = {**VALID_PAYLOAD, "type": "X"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_rejects_negative_torque(client):
    bad = {**VALID_PAYLOAD, "torque_nm": -5}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_rejects_zero_kelvin(client):
    bad = {**VALID_PAYLOAD, "air_temp_k": 0}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_health_reports_degraded_when_no_model():
    """If the registry is empty, /health still returns 200 with status='degraded'."""
    app.dependency_overrides[get_model_bundle] = lambda: None
    try:
        c = TestClient(app)
        r = c.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "degraded"
        # And /predict returns 503 in the same state.
        r2 = c.post("/predict", json=VALID_PAYLOAD)
        assert r2.status_code == 503
    finally:
        app.dependency_overrides.clear()
        app_module._clear_model_cache()


def test_ready_returns_200_when_model_loaded(client):
    """/ready is the K8s readiness probe -- 200 means 'send me traffic'."""
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_metrics_endpoint_exposes_prometheus_text(client):
    """Phase 7: /metrics should be Prometheus exposition format with our custom keys."""
    # Drive one prediction first so the custom counters appear in output.
    client.post("/predict", json=VALID_PAYLOAD)

    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    # Built-in HTTP metrics from the instrumentator
    assert "http_request_duration_seconds" in body
    # Our custom ML metrics
    assert "pdm_predictions_total" in body
    assert "pdm_prediction_probability" in body


def test_ready_returns_503_when_no_model():
    """/ready returns 503 when the model isn't loaded so K8s pulls the pod
    out of the Service's rotation. /health stays 200 so K8s doesn't restart."""
    app.dependency_overrides[get_model_bundle] = lambda: None
    try:
        c = TestClient(app)
        r_ready = c.get("/ready")
        r_health = c.get("/health")
        assert r_ready.status_code == 503
        assert r_health.status_code == 200  # liveness still alive
    finally:
        app.dependency_overrides.clear()
        app_module._clear_model_cache()
