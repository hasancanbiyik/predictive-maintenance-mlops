"""
Tests for the retrain orchestrator. Three decision paths to cover:

  1. No drift -> no-op exit.
  2. Drift + challenger beats champion -> promote (registry version advances).
  3. Drift + challenger same/worse than champion -> EXIT_DRIFT_BUT_NO_IMPROVEMENT.

We mock the drift gate and the trainer to keep these tests fast; the real
training pipeline is exercised by test_mlflow_logging.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from src.orchestrator import retrain


@pytest.fixture
def isolated_mlflow(tmp_path, monkeypatch):
    """Throwaway MLflow store per test so registry state can't leak."""
    db_path = tmp_path / "mlflow.db"
    uri = f"sqlite:///{db_path}"
    monkeypatch.setattr(retrain, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(retrain, "PREDICTIONS_LOG", tmp_path / "preds.jsonl")
    monkeypatch.setattr(retrain, "DRIFT_OUT_DIR", tmp_path / "drift")
    mlflow.set_tracking_uri(uri)
    # All tests run from project root so DEFAULT_CSV resolves.
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    yield uri


def _fake_train_pipeline(f1: float, winner: str = "XGBoost") -> mock.Mock:
    """Replace train_baseline.train_pipeline with a deterministic stub."""
    return mock.Mock(return_value={
        "winner_name": winner,
        "winner_run_id": "fake-run-id",
        "winner_f1": f1,
        "winner_passes": f1 >= 0.70,
        "results": {},
        "run_ids": {},
    })


def test_no_drift_returns_noop(isolated_mlflow, monkeypatch):
    monkeypatch.setattr(retrain, "check_drift", lambda: (False, {"drift_detected": False}))
    rc = retrain.main([])
    assert rc == retrain.EXIT_NOOP_NO_DRIFT


def test_drift_with_better_challenger_promotes(isolated_mlflow, monkeypatch):
    monkeypatch.setattr(retrain, "check_drift", lambda: (True, {"drift_detected": True, "n_drifted": 5}))

    # No champion yet -> any qualifying challenger promotes.
    monkeypatch.setattr(retrain, "get_champion", lambda *a, **kw: retrain.Champion(version=None, f1=None))

    monkeypatch.setattr(retrain.train_baseline, "train_pipeline", _fake_train_pipeline(0.80))
    monkeypatch.setattr(
        retrain.train_baseline,
        "register_winner",
        mock.Mock(return_value=1),
    )

    rc = retrain.main([])
    assert rc == retrain.EXIT_PROMOTED
    record = json.loads((retrain.DRIFT_OUT_DIR / "retrain_decision.json").read_text())
    assert record["decision"] in {"promote", "promote_first"}
    assert record["challenger"]["f1"] == 0.80


def test_drift_with_worse_challenger_does_not_promote(isolated_mlflow, monkeypatch):
    monkeypatch.setattr(retrain, "check_drift", lambda: (True, {"drift_detected": True, "n_drifted": 5}))
    monkeypatch.setattr(retrain, "get_champion", lambda *a, **kw: retrain.Champion(version="3", f1=0.85))

    monkeypatch.setattr(retrain.train_baseline, "train_pipeline", _fake_train_pipeline(0.80))
    register_mock = mock.Mock()
    monkeypatch.setattr(retrain.train_baseline, "register_winner", register_mock)

    rc = retrain.main([])
    assert rc == retrain.EXIT_DRIFT_BUT_NO_IMPROVEMENT
    register_mock.assert_not_called()


def test_tie_does_not_promote(isolated_mlflow, monkeypatch):
    """Exact F1 tie must NOT promote -- otherwise the registry churns on noise."""
    monkeypatch.setattr(retrain, "check_drift", lambda: (True, {"drift_detected": True, "n_drifted": 5}))
    monkeypatch.setattr(retrain, "get_champion", lambda *a, **kw: retrain.Champion(version="5", f1=0.80))

    monkeypatch.setattr(retrain.train_baseline, "train_pipeline", _fake_train_pipeline(0.80))
    register_mock = mock.Mock()
    monkeypatch.setattr(retrain.train_baseline, "register_winner", register_mock)

    rc = retrain.main([])
    assert rc == retrain.EXIT_DRIFT_BUT_NO_IMPROVEMENT
    register_mock.assert_not_called()


def test_force_bypasses_drift_gate(isolated_mlflow, monkeypatch):
    # check_drift would raise if called -- the --force path must skip it.
    monkeypatch.setattr(
        retrain,
        "check_drift",
        mock.Mock(side_effect=AssertionError("check_drift was called despite --force")),
    )
    monkeypatch.setattr(retrain, "get_champion", lambda *a, **kw: retrain.Champion(version=None, f1=None))
    monkeypatch.setattr(retrain.train_baseline, "train_pipeline", _fake_train_pipeline(0.75))
    monkeypatch.setattr(retrain.train_baseline, "register_winner", mock.Mock(return_value=1))

    rc = retrain.main(["--force"])
    assert rc == retrain.EXIT_PROMOTED
