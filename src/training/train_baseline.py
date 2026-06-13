"""
Phase 1 baseline: train RandomForest and XGBoost on AI4I, compare, save the best.

Design notes (interview-ready):
- We try the *simplest* class-imbalance fix first (`class_weight='balanced'` for
  RF, `scale_pos_weight` for XGBoost). If F1 on the minority class clears 0.70
  we stop. If not, we escalate to SMOTE in a follow-up. Always reach for the
  blunt tool before the sharp one.
- The metric of record is **F1 on the failure class** (label=1). On a 3.4%-
  positive dataset, accuracy is uninformative — a model that predicts "no
  failure" forever scores 96.6%.
- We *also* log precision, recall, ROC-AUC, and PR-AUC. PR-AUC > ROC-AUC for
  rare-event detection because ROC is optimistic when negatives dominate.
- Test set is held out at split time and never touched during training.

Run: `python -m src.training.train_baseline`
"""

from __future__ import annotations

import json
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.data.load import load_dataset

MODELS_DIR = Path("models")
TARGET_F1 = 0.70  # minimum F1 on the minority class to consider Phase 1 "done"


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


def evaluate(model_name: str, y_true, y_pred, y_proba) -> EvalResult:
    return EvalResult(
        model_name=model_name,
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_proba)),
        pr_auc=float(average_precision_score(y_true, y_proba)),
        confusion=confusion_matrix(y_true, y_pred).tolist(),
    )


def train_random_forest(X_train, y_train, X_test, y_test) -> tuple[Any, EvalResult]:
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",  # the cheap imbalance fix
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return clf, evaluate("RandomForest", y_test, pred, proba)


def train_xgboost(X_train, y_train, X_test, y_test) -> tuple[Any, EvalResult]:
    # scale_pos_weight = neg/pos -- XGBoost's analog of class_weight='balanced'
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    spw = n_neg / max(n_pos, 1)

    clf = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=spw,
        eval_metric="aucpr",  # PR-AUC is more honest on rare events
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return clf, evaluate("XGBoost", y_test, pred, proba)


def pretty_print(res: EvalResult, y_test) -> None:
    print(f"\n=== {res.model_name} ===")
    print(f"Precision (failure): {res.precision:.3f}")
    print(f"Recall    (failure): {res.recall:.3f}")
    print(f"F1        (failure): {res.f1:.3f}  {'PASS' if res.passes() else 'FAIL'} (target ≥ {TARGET_F1})")
    print(f"ROC-AUC            : {res.roc_auc:.3f}")
    print(f"PR-AUC             : {res.pr_auc:.3f}")
    tn, fp, fn, tp = (res.confusion[0][0], res.confusion[0][1], res.confusion[1][0], res.confusion[1][1])
    print(f"Confusion          : TN={tn} FP={fp} FN={fn} TP={tp}")


def main() -> int:
    print(f"Loading dataset...")
    ds = load_dataset()
    print(f"Train: {ds.X_train.shape}, Test: {ds.X_test.shape}")
    print(f"Train positives: {int(ds.y_train.sum())} / {len(ds.y_train)}")
    print(f"Test  positives: {int(ds.y_test.sum())} / {len(ds.y_test)}")
    print(f"Features ({len(ds.feature_names)}): {ds.feature_names}")

    rf_model, rf_res = train_random_forest(ds.X_train, ds.y_train, ds.X_test, ds.y_test)
    xgb_model, xgb_res = train_xgboost(ds.X_train, ds.y_train, ds.X_test, ds.y_test)

    pretty_print(rf_res, ds.y_test)
    pretty_print(xgb_res, ds.y_test)

    # Pick winner by F1 on minority class
    if xgb_res.f1 >= rf_res.f1:
        best_model, best_res = xgb_model, xgb_res
    else:
        best_model, best_res = rf_model, rf_res

    print(f"\n>>> Winner: {best_res.model_name} (F1={best_res.f1:.3f})")

    # Persist
    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / "baseline.pkl"
    metrics_path = MODELS_DIR / "baseline_metrics.json"

    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": best_model,
                "feature_names": ds.feature_names,
                "metric_threshold": 0.5,
            },
            f,
        )
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "winner": best_res.model_name,
                "random_forest": asdict(rf_res),
                "xgboost": asdict(xgb_res),
            },
            f,
            indent=2,
        )

    print(f"Saved model    -> {model_path}")
    print(f"Saved metrics  -> {metrics_path}")

    if not best_res.passes():
        print(
            f"\n[!] Best F1 ({best_res.f1:.3f}) below target ({TARGET_F1}). "
            "Escalation path: add SMOTE on the train fold only, or tune the decision threshold."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
