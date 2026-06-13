"""
Pydantic schemas for the prediction API.

Design notes (interview-ready):
- The API speaks the *raw* feature vocabulary (`type: "L"`), not our internal
  one-hot encoding (`Type_L/M/H`). Forcing callers to one-hot their inputs
  would be a leaky abstraction -- they'd have to know how we trained.
- Validation bounds here are *shape* checks (Kelvin > 0, type is L/M/H), not
  *distribution* checks. Detecting "torque is suddenly 3x higher than training"
  is the drift monitor's job in Phase 7. Don't conflate them.
- Every prediction response carries the model name + version + alias so logs
  and dashboards can attribute predictions to a specific model artifact.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """One sensor reading from a single machine at a single moment."""

    type: Literal["L", "M", "H"] = Field(
        ..., description="Product quality variant: Low / Medium / High"
    )
    air_temp_k: float = Field(..., gt=200, lt=400, description="Ambient air temperature [K]")
    process_temp_k: float = Field(..., gt=200, lt=400, description="Process temperature [K]")
    rot_speed_rpm: float = Field(..., ge=0, lt=10_000, description="Rotational speed [rpm]")
    torque_nm: float = Field(..., ge=0, lt=500, description="Torque [Nm]")
    tool_wear_min: float = Field(..., ge=0, lt=10_000, description="Tool wear [min]")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "type": "L",
                    "air_temp_k": 298.1,
                    "process_temp_k": 308.6,
                    "rot_speed_rpm": 1551,
                    "torque_nm": 42.8,
                    "tool_wear_min": 0,
                }
            ]
        }
    }


class PredictResponse(BaseModel):
    prediction: int = Field(..., description="1 = predicted failure, 0 = no failure")
    probability: float = Field(..., ge=0, le=1, description="P(failure)")
    model_name: str
    model_version: str
    model_alias: str
    decision_threshold: float = 0.5


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_name: str | None = None
    model_version: str | None = None
    model_alias: str | None = None
