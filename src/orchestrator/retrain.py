"""
Phase 8: drift-triggered, champion-challenger guarded auto-retraining.

Pipeline (the production loop, finally closed):

  1. Read the prediction log written by /predict (live distribution).
  2. Run Evidently's drift check vs the training distribution.
  3. If no drift AND not forced -> exit "no-op".
  4. If drift detected OR --force:
     a. Train RF + XGBoost candidates (without auto-registering).
     b. Pick the better of the two as the *challenger*.
     c. Look up the current *champion* (@staging) and its F1 from MLflow.
     d. If challenger F1 > champion F1, register + re-point @staging.
     e. Else, log "drift detected but no improvement -- investigate data".

Honest about what production would add (interview talking points):
  - **Shadow deployment**: serve the challenger alongside the champion on
    live traffic for days, compare without exposing the new model to users.
  - **Statistical significance**: a single-shot F1 delta of 0.005 is noise.
    Real systems require N runs and a p-value, or a 5%+ relative gain.
  - **Manual approval gate**: prod model swaps in regulated domains require
    a human sign-off. We auto-promote here for demo purposes.
  - **Time-based fallback**: schedule a retrain every N days regardless of
    drift, because covariate drift isn't the only failure mode.

Memory hook: drift detection is the smoke detector. Champion-challenger is
the rule that says "don't call the fire department for a candle."

Run:
    python -m src.orchestrator.retrain
    python -m src.orchestrator.retrain --force        # skip drift gate
    python -m src.orchestrator.retrain --min-gain 0.01  # require 1% F1 gain
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.monitoring import drift as drift_mod
from src.training import train_baseline

# Same defaults the trainer uses; env-overridable for compose / K8s.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
REGISTERED_MODEL_NAME = os.getenv("MODEL_NAME", "predictive_maintenance")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "staging")
PREDICTIONS_LOG = Path(os.getenv("PREDICTIONS_LOG", "data/predictions/predictions.jsonl"))
DRIFT_OUT_DIR = Path(os.getenv("DRIFT_OUT_DIR", "artifacts/drift"))

# Exit codes communicate the orchestrator's decision -- useful when a
# scheduler (cron / GitHub Actions / Argo) wants to alert on retrains.
EXIT_NOOP_NO_DRIFT = 0
EXIT_PROMOTED = 0
EXIT_DRIFT_BUT_NO_IMPROVEMENT = 2
EXIT_TRAINING_FAILED = 3
EXIT_NO_PREDICTIONS = 4


@dataclass
class Champion:
    version: str | None
    f1: float | None


def get_champion(client: MlflowClient, model_name: str, alias: str) -> Champion:
    """The currently-deployed model + its training F1. None if no model yet."""
    try:
        mv = client.get_model_version_by_alias(model_name, alias)
    except MlflowException:
        return Champion(version=None, f1=None)
    try:
        run = client.get_run(mv.run_id)
        f1 = run.data.metrics.get("f1_failure")
    except MlflowException:
        f1 = None
    return Champion(version=str(mv.version), f1=f1)


def check_drift() -> tuple[bool, dict]:
    """Returns (drift_detected, summary). Raises if no predictions to compare."""
    if not PREDICTIONS_LOG.exists():
        raise FileNotFoundError(
            f"No prediction log at {PREDICTIONS_LOG}. Drive traffic to /predict first."
        )

    reference = drift_mod.load_reference(drift_mod.DEFAULT_CSV)
    current = drift_mod.load_current(PREDICTIONS_LOG)
    result = drift_mod.run_drift_report(reference, current, DRIFT_OUT_DIR)
    summary = result["summary"]
    detected = bool(summary.get("drift_detected"))
    return detected, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Drift-triggered retrain orchestrator.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the drift gate and always train a challenger.",
    )
    parser.add_argument(
        "--min-gain",
        type=float,
        default=0.0,
        help=(
            "Required absolute F1 gain over the champion (strict inequality). "
            "Default 0.0 means 'any positive improvement promotes; ties do NOT'. "
            "Production should set this to e.g. 0.01 to avoid promoting noise."
        ),
    )
    args = parser.parse_args(argv)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(train_baseline.EXPERIMENT_NAME)
    client = MlflowClient()

    # --- 1) Drift gate ---------------------------------------------------
    if args.force:
        print(">>> --force: skipping drift gate.")
        drift_detected, summary = True, {"drift_detected": "forced"}
    else:
        try:
            drift_detected, summary = check_drift()
        except FileNotFoundError as e:
            print(f"[!] {e}")
            return EXIT_NO_PREDICTIONS
        print(f">>> Drift detected: {drift_detected}  ({summary.get('n_drifted')} columns)")
        if not drift_detected:
            print(">>> No drift. No retrain needed.")
            return EXIT_NOOP_NO_DRIFT

    # --- 2) Train challenger (without auto-registering) -----------------
    print("\n>>> Training challenger...")
    try:
        info = train_baseline.train_pipeline()
    except Exception as e:
        print(f"[!] Training failed: {e}")
        return EXIT_TRAINING_FAILED

    if not info["winner_passes"]:
        print(
            f"[!] Challenger F1 {info['winner_f1']:.3f} below "
            f"TARGET_F1 {train_baseline.TARGET_F1}. Not promoting."
        )
        return EXIT_DRIFT_BUT_NO_IMPROVEMENT

    # --- 3) Champion comparison -----------------------------------------
    champion = get_champion(client, REGISTERED_MODEL_NAME, MODEL_ALIAS)
    print(f"\n>>> Champion: v{champion.version} (F1={champion.f1})")
    print(f">>> Challenger: {info['winner_name']} (F1={info['winner_f1']:.3f})")

    if champion.f1 is None:
        print(">>> No champion registered. Promoting challenger by default.")
        decision = "promote_first"
    else:
        gain = info["winner_f1"] - champion.f1
        # Strict inequality so an exact tie does NOT churn the registry.
        # Pass a *negative* --min-gain if you want to promote on ties for debug.
        if gain > args.min_gain:
            print(f">>> Gain {gain:+.4f} > min_gain {args.min_gain}. Promoting.")
            decision = "promote"
        else:
            print(
                f">>> Gain {gain:+.4f} <= min_gain {args.min_gain}. "
                "Drift detected but no improvement -- investigate data, not model."
            )
            return EXIT_DRIFT_BUT_NO_IMPROVEMENT

    # --- 4) Register + re-point alias -----------------------------------
    new_version = train_baseline.register_winner(info["winner_run_id"], REGISTERED_MODEL_NAME)
    print(f"\n>>> Registered: {REGISTERED_MODEL_NAME} v{new_version} (alias @{MODEL_ALIAS})")
    print(">>> Restart the API to load the new model:")
    print("    docker compose restart api    # or `kubectl rollout restart deployment/pdm-api -n pdm`")

    # Persist a structured record of the decision for downstream tools.
    decision_record = {
        "decision": decision,
        "drift_summary": summary,
        "champion": {"version": champion.version, "f1": champion.f1},
        "challenger": {
            "name": info["winner_name"],
            "run_id": info["winner_run_id"],
            "f1": info["winner_f1"],
            "registered_version": new_version,
        },
        "min_gain": args.min_gain,
    }
    DRIFT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    (DRIFT_OUT_DIR / "retrain_decision.json").write_text(json.dumps(decision_record, indent=2))
    return EXIT_PROMOTED


if __name__ == "__main__":
    sys.exit(main())
