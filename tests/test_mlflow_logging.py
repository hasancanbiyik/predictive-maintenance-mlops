"""
Smoke test for the MLflow integration. Runs the trainer end-to-end against a
temporary SQLite store (so it can't clobber the real `mlflow.db`) and checks:

  1. Two runs were logged (one per model family).
  2. Each run has the expected metric keys.
  3. A model named 'predictive_maintenance' got registered with at least 1 version.
  4. The @staging alias resolves to a real version.

This is a real integration test (it actually trains the models), so it's the
slowest test in the suite -- but still well under a minute on a laptop CPU.
"""

from __future__ import annotations

import os
from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from src.training import train_baseline


@pytest.fixture
def isolated_mlflow(tmp_path, monkeypatch):
    """Point MLflow at a throwaway SQLite store under tmp_path."""
    db_path = tmp_path / "test_mlflow.db"
    uri = f"sqlite:///{db_path}"
    monkeypatch.setattr(train_baseline, "MLFLOW_TRACKING_URI", uri)
    # Also reset any previously-set tracking URI on the mlflow module itself.
    mlflow.set_tracking_uri(uri)
    # Run training from the project root so data/raw/ai4i2020.csv resolves.
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    yield uri


def test_training_run_logs_two_runs_and_registers_winner(isolated_mlflow):
    rc = train_baseline.main()
    assert rc == 0, "Training should pass the F1 target"

    client = MlflowClient(tracking_uri=isolated_mlflow)
    exp = client.get_experiment_by_name(train_baseline.EXPERIMENT_NAME)
    assert exp is not None
    runs = client.search_runs(exp.experiment_id)
    assert len(runs) == 2, f"Expected 2 runs (RF + XGB), got {len(runs)}"

    expected_metric_keys = {
        "precision_failure",
        "recall_failure",
        "f1_failure",
        "roc_auc",
        "pr_auc",
    }
    for run in runs:
        assert expected_metric_keys.issubset(run.data.metrics.keys()), (
            f"Run {run.info.run_name} missing metrics: {expected_metric_keys - run.data.metrics.keys()}"
        )
        assert "dataset_sha256" in run.data.params

    # Registered model + @staging alias
    rm = client.get_registered_model(train_baseline.REGISTERED_MODEL_NAME)
    assert len(rm.latest_versions) >= 1
    mv = client.get_model_version_by_alias(
        train_baseline.REGISTERED_MODEL_NAME, train_baseline.STAGING_ALIAS
    )
    assert mv is not None
    assert int(mv.version) >= 1
