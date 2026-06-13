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

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import mlflow
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.data.load import CLEAN_NUMERIC_FEATURES
from src.serving.schemas import HealthResponse, PredictRequest, PredictResponse

# --- Configuration (env-overridable for Phase 4 Docker / Phase 5 K8s) ---

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
REGISTERED_MODEL_NAME = os.getenv("MODEL_NAME", "predictive_maintenance")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "staging")
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))

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


@lru_cache(maxsize=1)
def get_model_bundle() -> ModelBundle | None:
    """Resolve `@alias` -> version, load the model, cache the bundle.

    Returns None (instead of raising) when the registry has no match. This
    makes the function safe to use as a FastAPI dependency in both endpoints:
    /predict treats None as a 503; /health treats None as 'degraded' (still
    200 OK, because the *service* is alive even if the *model* isn't).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)
    except MlflowException as e:
        logger.warning("Model alias %s@%s not found: %s", REGISTERED_MODEL_NAME, MODEL_ALIAS, e)
        return None
    uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
    try:
        model = _load_classifier(uri)
    except Exception as e:
        logger.error("Failed to load model %s: %s", uri, e)
        return None
    return ModelBundle(model=model, name=REGISTERED_MODEL_NAME, version=mv.version, alias=MODEL_ALIAS)


# --- App ----------------------------------------------------------------

app = FastAPI(
    title="Predictive Maintenance API",
    description="Predicts machine failure from sensor readings. Model loaded via MLflow Model Registry.",
    version="0.3.0",
)


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
    """Liveness probe. Reports model identity if loaded, 'degraded' if not.

    Always returns 200 -- the service is alive. The body tells you whether
    the *model* is loaded. Kubernetes liveness probes care about the service;
    readiness probes can be wired to this body in Phase 5.
    """
    if bundle is None:
        return HealthResponse(status="degraded")
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
    return PredictResponse(
        prediction=pred,
        probability=proba,
        model_name=bundle.name,
        model_version=str(bundle.version),
        model_alias=bundle.alias,
        decision_threshold=DECISION_THRESHOLD,
    )
