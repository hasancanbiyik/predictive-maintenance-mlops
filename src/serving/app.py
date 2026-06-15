"""
Phase 3: FastAPI service that serves the registered model.

What this proves in an interview:
- The serving layer loads the model through a stable registry handle
  (`models:/predictive_maintenance@staging`), not a file path. When Phase 8
  retrains and re-points the alias, the service picks up the new model on
  restart -- zero code change. That's the whole point of a model registry.
- Pydantic enforces the request *shape* at the door, so the model never sees
  garbage. Distribution drift is a separate concern (Phase 7).
- The model is loaded once per process and cached. The first hit pays the
  load cost; subsequent hits are warm.

Run:
    uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

Then visit:
    http://127.0.0.1:8000/docs   -- Swagger UI
    http://127.0.0.1:8000/health -- liveness + model info
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from src.data.load import CLEAN_NUMERIC_FEATURES
from src.serving.schemas import HealthResponse, PredictRequest, PredictResponse

# --- Configuration (env-overridable for Phase 4 Docker / Phase 5 K8s) ---

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
REGISTERED_MODEL_NAME = os.getenv("MODEL_NAME", "predictive_maintenance")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "staging")
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))

# Phase 7: where to capture live prediction inputs for offline drift analysis.
# Each line is a JSON object: timestamp + model version + the raw request fields.
PREDICTIONS_LOG = Path(os.getenv("PREDICTIONS_LOG", "/var/log/pdm/predictions.jsonl"))
_log_lock = threading.Lock()

# Feature column order the trained model expects. Must match what
# `src.data.load.build_features()` produces, in order.
TRAINING_FEATURE_ORDER = CLEAN_NUMERIC_FEATURES + ["Type_H", "Type_L", "Type_M"]

logger = logging.getLogger("predictive_maintenance.serving")


@dataclass
class ModelBundle:
    """The loaded model plus the metadata we want in /predict responses."""

    model: Any  # sklearn-compatible estimator with .predict_proba
    name: str
    version: str
    alias: str


def _load_classifier(uri: str) -> Any:
    """Load the registered model regardless of which flavor it was logged with.

    We registered RandomForest via mlflow.sklearn and XGBoost via mlflow.xgboost.
    Either loader will fail noisily on the wrong flavor, so try sklearn first
    (broader coverage) and fall back to xgboost. Both flavors return estimators
    that expose .predict_proba().
    """
    try:
        return mlflow.sklearn.load_model(uri)
    except Exception as e:
        logger.info("sklearn loader failed (%s); falling back to xgboost", e.__class__.__name__)
        return mlflow.xgboost.load_model(uri)


# Cache only SUCCESSFUL bundle loads. lru_cache would cache None too, so a
# pod that boots before MLflow has a model would never recover until restart.
# A simple module-level dict + lock gives us "cache the win, retry the miss".
_bundle_cache: dict[str, ModelBundle] = {}
_bundle_lock = threading.Lock()


def get_model_bundle() -> ModelBundle | None:
    """Resolve `@alias` -> version, load the model, cache only on success.

    Returns None (instead of raising) when the registry has no match. This
    makes the function safe to use as a FastAPI dependency in both endpoints:
    /predict treats None as 503; /health treats None as 'degraded' (still
    200 OK -- the service is alive even if the model isn't).

    Caching only successes means the API auto-recovers when a model gets
    registered later. The trade-off: every request before first success
    triggers a registry lookup. That's fine -- /ready hits this once per
    10s probe, not per /predict call.
    """
    cached = _bundle_cache.get("bundle")
    if cached is not None:
        return cached

    with _bundle_lock:
        # Re-check after acquiring the lock (double-checked locking pattern).
        cached = _bundle_cache.get("bundle")
        if cached is not None:
            return cached

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        try:
            mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)
        except MlflowException as e:
            logger.warning(
                "Model alias %s@%s not found: %s", REGISTERED_MODEL_NAME, MODEL_ALIAS, e
            )
            return None

        uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
        try:
            model = _load_classifier(uri)
        except Exception as e:
            logger.error("Failed to load model %s: %s", uri, e)
            return None

        bundle = ModelBundle(model=model, name=REGISTERED_MODEL_NAME, version=mv.version, alias=MODEL_ALIAS)
        _bundle_cache["bundle"] = bundle
        return bundle


def _clear_model_cache() -> None:
    """Test helper -- forces the next call to re-resolve the registry."""
    _bundle_cache.clear()


# --- App ----------------------------------------------------------------

app = FastAPI(
    title="Predictive Maintenance API",
    description="Predicts machine failure from sensor readings. Model loaded via MLflow Model Registry.",
    version="0.7.0",
)

# Phase 7: Prometheus metrics.
#  - The Instrumentator auto-exposes /metrics with HTTP request rate, latency
#    histograms, and status-code counters keyed on endpoint -- no code per
#    endpoint.
#  - Custom metrics capture what makes this an *ML* service rather than a
#    plain web app: how many predictions, by class, and the probability
#    distribution. Drift in either is a signal.
Instrumentator().instrument(app).expose(app)

PREDICTIONS_COUNTER = Counter(
    "pdm_predictions_total",
    "Total predictions served",
    ["model_version", "prediction"],
)
PROBABILITY_HISTOGRAM = Histogram(
    "pdm_prediction_probability",
    "Distribution of predicted failure probabilities",
    ["model_version"],
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99),
)


def _append_prediction_log(req: PredictRequest, proba: float, bundle: "ModelBundle") -> None:
    """Append a single JSON line for offline drift analysis.

    Best-effort: a logging-disk failure must not break /predict. Production
    versions of this would push to Kafka/Kinesis instead of a local file.
    """
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "model_version": bundle.version,
        "model_alias": bundle.alias,
        "probability": proba,
        **req.model_dump(),
    }
    try:
        PREDICTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _log_lock, PREDICTIONS_LOG.open("a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as e:
        logger.warning("Failed to append prediction log: %s", e)


def _to_feature_frame(req: PredictRequest) -> pd.DataFrame:
    """Build the single-row DataFrame the model was trained on."""
    row = {
        "air_temp_k": req.air_temp_k,
        "process_temp_k": req.process_temp_k,
        "rot_speed_rpm": req.rot_speed_rpm,
        "torque_nm": req.torque_nm,
        "tool_wear_min": req.tool_wear_min,
        "Type_H": int(req.type == "H"),
        "Type_L": int(req.type == "L"),
        "Type_M": int(req.type == "M"),
    }
    return pd.DataFrame([row], columns=TRAINING_FEATURE_ORDER)


@app.get("/health", response_model=HealthResponse)
def health(bundle: ModelBundle | None = Depends(get_model_bundle)) -> HealthResponse:
    """Liveness probe. Always 200. The service is alive whether or not the
    model is loaded. K8s livenessProbe uses this -- restarting a pod when
    only the model failed to load won't fix anything and just causes flapping.
    """
    if bundle is None:
        return HealthResponse(status="degraded")
    return HealthResponse(
        status="ok",
        model_name=bundle.name,
        model_version=str(bundle.version),
        model_alias=bundle.alias,
    )


@app.get("/ready", response_model=HealthResponse)
def ready(bundle: ModelBundle | None = Depends(get_model_bundle)) -> HealthResponse:
    """Readiness probe. Returns 503 when the model isn't loaded.

    K8s readinessProbe uses this -- a pod is excluded from the Service's load-
    balancer rotation until /ready returns 200. This is what makes rolling
    updates safe: new pods don't get traffic until they can actually predict.
    """
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )
    return HealthResponse(
        status="ok",
        model_name=bundle.name,
        model_version=str(bundle.version),
        model_alias=bundle.alias,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(
    req: PredictRequest,
    bundle: ModelBundle | None = Depends(get_model_bundle),
) -> PredictResponse:
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"No model registered at {REGISTERED_MODEL_NAME}@{MODEL_ALIAS}",
        )
    X = _to_feature_frame(req)
    proba = float(bundle.model.predict_proba(X)[0, 1])
    pred = int(proba >= DECISION_THRESHOLD)

    # Phase 7 instrumentation: counter + histogram, then JSONL drift log.
    PREDICTIONS_COUNTER.labels(model_version=str(bundle.version), prediction=str(pred)).inc()
    PROBABILITY_HISTOGRAM.labels(model_version=str(bundle.version)).observe(proba)
    _append_prediction_log(req, proba, bundle)

    return PredictResponse(
        prediction=pred,
        probability=proba,
        model_name=bundle.name,
        model_version=str(bundle.version),
        model_alias=bundle.alias,
        decision_threshold=DECISION_THRESHOLD,
    )
