"""
Phase 2: train RF + XGBoost, log everything to MLflow, register the winner.

What this script proves in an interview:
- Experiment tracking: every run is reproducible because params, metrics, dataset
  hash, and the trained artifact are all captured together.
- Model registry: the winning model gets a versioned entry under a stable name
  so downstream services (Phase 3 API) load it by name, not by file path.
- Backend choice: SQLite tracking store. File-backend MLflow doesn't support
  the Model Registry; SQLite gives full feature parity with zero infra.

Run:
    python -m src.training.train_baseline

Browse afterwards:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.data.load import DEFAULT_CSV, load_dataset

# --- Configuration --------------------------------------------------------

TARGET_F1 = 0.70
# Default to local SQLite for laptop use; compose / K8s override via env var
# to point at the running MLflow server (`http://mlflow:5000`).
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = "predictive_maintenance"
REGISTERED_MODEL_NAME = "predictive_maintenance"
STAGING_ALIAS = "staging"  # Phase 3 API will load `predictive_maintenance@staging`


# --- Result container -----------------------------------------------------


@dataclass
class EvalResult:
    model_name: str
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    confusion: list[list[int]]

    def passes(self, target: float = TARGET_F1) -> bool:
        return self.f1 >= target


def _evaluate(model_name: str, y_true, y_pred, y_proba) -> EvalResult:
    return EvalResult(
        model_name=model_name,
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_proba)),
        pr_auc=float(average_precision_score(y_true, y_proba)),
        confusion=confusion_matrix(y_true, y_pred).tolist(),
    )


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _log_common(
    res: EvalResult,
    feature_names: list[str],
    dataset_path: Path,
    dataset_sha: str,
    n_train: int,
    n_test: int,
    n_pos_train: int,
) -> None:
    """Log everything not specific to a particular estimator."""
    mlflow.log_params(
        {
            "model_family": res.model_name,
            "n_features": len(feature_names),
            "n_train": n_train,
            "n_test": n_test,
            "n_positives_train": n_pos_train,
            "dataset_path": str(dataset_path),
            "dataset_sha256": dataset_sha,
            "decision_threshold": 0.5,
        }
    )
    mlflow.log_metrics(
        {
            "precision_failure": res.precision,
            "recall_failure": res.recall,
            "f1_failure": res.f1,
            "roc_auc": res.roc_auc,
            "pr_auc": res.pr_auc,
        }
    )
    # Log the confusion matrix as a JSON artifact -- searchable, diffable.
    Path("models").mkdir(exist_ok=True)
    cm_path = Path("models") / f"confusion_{res.model_name}.json"
    cm_path.write_text(json.dumps(res.confusion))
    mlflow.log_artifact(str(cm_path))
    mlflow.set_tag("metric_of_record", "f1_failure")
    mlflow.set_tag("passes_target", str(res.passes()))


def train_random_forest(X_train, y_train, X_test, y_test, ctx: dict) -> tuple[Any, EvalResult, str]:
    with mlflow.start_run(run_name="random_forest") as run:
        params = {"n_estimators": 300, "max_depth": None, "class_weight": "balanced", "random_state": 42}
        mlflow.log_params(params)
        clf = RandomForestClassifier(**params, n_jobs=-1)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        res = _evaluate("RandomForest", y_test, pred, proba)
        _log_common(res, **ctx)
        mlflow.sklearn.log_model(clf, name="model", input_example=X_train.iloc[:2])
        return clf, res, run.info.run_id


def train_xgboost(X_train, y_train, X_test, y_test, ctx: dict) -> tuple[Any, EvalResult, str]:
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    spw = n_neg / max(n_pos, 1)

    with mlflow.start_run(run_name="xgboost") as run:
        params = {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.1,
            "scale_pos_weight": spw,
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "random_state": 42,
        }
        mlflow.log_params(params)
        clf = XGBClassifier(**params, n_jobs=-1)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        res = _evaluate("XGBoost", y_test, pred, proba)
        _log_common(res, **ctx)
        mlflow.xgboost.log_model(clf, name="model", input_example=X_train.iloc[:2])
        return clf, res, run.info.run_id


def pretty_print(res: EvalResult) -> None:
    print(f"\n=== {res.model_name} ===")
    print(f"Precision (failure): {res.precision:.3f}")
    print(f"Recall    (failure): {res.recall:.3f}")
    print(f"F1        (failure): {res.f1:.3f}  {'PASS' if res.passes() else 'FAIL'} (target >= {TARGET_F1})")
    print(f"ROC-AUC            : {res.roc_auc:.3f}")
    print(f"PR-AUC             : {res.pr_auc:.3f}")
    tn, fp, fn, tp = (res.confusion[0][0], res.confusion[0][1], res.confusion[1][0], res.confusion[1][1])
    print(f"Confusion          : TN={tn} FP={fp} FN={fn} TP={tp}")


def register_winner(run_id: str, model_name: str) -> int:
    """Register the winning run's model and point the @staging alias at it."""
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    # Aliases (modern MLflow) are stable handles; better than the deprecated
    # 'Production'/'Staging' stages. Phase 3 will load 'predictive_maintenance@staging'.
    client.set_registered_model_alias(name=model_name, alias=STAGING_ALIAS, version=mv.version)
    return int(mv.version)


def train_pipeline() -> dict:
    """Train RF + XGBoost, log to MLflow, return a dict the orchestrator
    needs to make a register/skip decision.

    Caller is responsible for calling `mlflow.set_tracking_uri` and
    `mlflow.set_experiment` first -- they're cheap and harmless to call
    multiple times, so main() / the orchestrator each call them.

    Returns:
        {
            "winner_name": "XGBoost" | "RandomForest",
            "winner_run_id": str,
            "winner_f1": float,
            "winner_passes": bool,   # F1 >= TARGET_F1
            "results": {"RandomForest": EvalResult, "XGBoost": EvalResult},
            "run_ids": {"RandomForest": str, "XGBoost": str},
        }
    """
    print("Loading dataset...")
    ds = load_dataset()
    dataset_sha = _file_sha256(DEFAULT_CSV)
    print(f"Train: {ds.X_train.shape}, Test: {ds.X_test.shape}")
    print(f"Dataset SHA256: {dataset_sha[:16]}...")

    ctx = dict(
        feature_names=ds.feature_names,
        dataset_path=DEFAULT_CSV,
        dataset_sha=dataset_sha,
        n_train=len(ds.X_train),
        n_test=len(ds.X_test),
        n_pos_train=int(ds.y_train.sum()),
    )

    _, rf_res, rf_run = train_random_forest(ds.X_train, ds.y_train, ds.X_test, ds.y_test, ctx)
    _, xgb_res, xgb_run = train_xgboost(ds.X_train, ds.y_train, ds.X_test, ds.y_test, ctx)

    pretty_print(rf_res)
    pretty_print(xgb_res)

    if xgb_res.f1 >= rf_res.f1:
        winner_name, winner_res, winner_run = "XGBoost", xgb_res, xgb_run
    else:
        winner_name, winner_res, winner_run = "RandomForest", rf_res, rf_run

    print(f"\n>>> Winner: {winner_name} (F1={winner_res.f1:.3f})")

    return {
        "winner_name": winner_name,
        "winner_run_id": winner_run,
        "winner_f1": winner_res.f1,
        "winner_passes": winner_res.passes(),
        "results": {"RandomForest": rf_res, "XGBoost": xgb_res},
        "run_ids": {"RandomForest": rf_run, "XGBoost": xgb_run},
    }


def main() -> int:
    """CLI entrypoint: train + auto-register if winner clears TARGET_F1.

    Behavior preserved from Phase 2 so the MLflow integration test and
    `python -m src.training.train_baseline` still work the same way. The
    Phase 8 orchestrator bypasses this and uses `train_pipeline()` directly
    so it can apply champion-challenger logic before registering.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    info = train_pipeline()

    if not info["winner_passes"]:
        print(f"[!] Below target ({TARGET_F1}). Not registering.")
        return 1

    version = register_winner(info["winner_run_id"], REGISTERED_MODEL_NAME)
    print(f">>> Registered: {REGISTERED_MODEL_NAME} v{version} (alias @{STAGING_ALIAS})")
    print(f">>> MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    print(f">>> Browse with: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
