"""
Phase 7: offline data-drift analysis.

Compares the *current* distribution of live inputs (captured in the JSONL
prediction log) against the *reference* distribution (the training dataset)
using Evidently AI's DataDriftPreset. Outputs:
  - drift_report.html  -- human-readable, what you'd hand a stakeholder
  - drift_summary.json -- machine-readable, what Phase 8 will read to decide
                          whether to trigger a retrain

Why a separate script instead of in-process:
  - Drift detection is a batch operation over many predictions, not per
    request.
  - Keeping it out of the API hot path keeps p99 latency stable.
  - Phase 8 wraps this exact script in a scheduled job. The same CLI you can
    `python -m src.monitoring.drift` today is what cron / GitHub Actions /
    Argo Workflows will invoke later.

Run:
    python -m src.monitoring.drift
    python -m src.monitoring.drift --out artifacts/drift
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.data.load import (
    CLEAN_NUMERIC_FEATURES,
    DEFAULT_CSV,
    FEATURE_RENAME,
)

DEFAULT_PREDICTIONS_LOG = Path("data/predictions/predictions.jsonl")
DEFAULT_OUT_DIR = Path("artifacts/drift")
# Feature columns the drift check considers. Numerics + Type (as a categorical
# string, NOT the one-hot encoding -- Evidently handles categoricals natively).
FEATURE_COLUMNS = CLEAN_NUMERIC_FEATURES + ["type"]


def load_reference(csv_path: Path = DEFAULT_CSV) -> pd.DataFrame:
    """The training distribution. Renames raw CSV columns to API vocabulary."""
    df = pd.read_csv(csv_path)
    df.columns = [c.lstrip("﻿") for c in df.columns]
    df = df.rename(columns=FEATURE_RENAME)
    df = df.rename(columns={"Type": "type"})
    return df[FEATURE_COLUMNS].copy()


def load_current(log_path: Path) -> pd.DataFrame:
    """The live distribution. Read JSONL prediction log; one row per request."""
    if not log_path.exists():
        raise FileNotFoundError(
            f"No prediction log at {log_path}. Drive some traffic to /predict first."
        )
    rows = []
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Prediction log {log_path} is empty.")
    df = pd.DataFrame(rows)
    # The API stores raw 'type' (L/M/H), same vocabulary as reference.
    return df[FEATURE_COLUMNS].copy()


def _import_evidently():
    """Evidently restructured its public API at 0.7. Try newer path first,
    fall back to the 0.6.x path. The Report.run() + save_html() shape has
    stayed stable across both."""
    try:  # Evidently >= 0.7
        from evidently import Report  # type: ignore[attr-defined]
        from evidently.presets import DataDriftPreset  # type: ignore[attr-defined]
        return Report, DataDriftPreset
    except ImportError:
        pass
    try:  # Evidently 0.4 - 0.6.x
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        return Report, DataDriftPreset
    except ImportError as e:
        raise SystemExit(
            "Could not import Evidently. Install with `pip install evidently>=0.6`."
        ) from e


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame, out_dir: Path) -> dict:
    """Run Evidently's DataDriftPreset and write artifacts.

    Evidently 0.7+ changed the API:
      report.run(...) returns a Snapshot; .save_html and .dict live on it.
    For 0.6.x and earlier, the Report itself carried those methods. We
    handle both by checking what the call returned.
    """
    Report, DataDriftPreset = _import_evidently()

    out_dir.mkdir(parents=True, exist_ok=True)
    report = Report([DataDriftPreset()])
    snapshot = report.run(reference_data=reference, current_data=current)

    # Newer Evidently returns a Snapshot; older versions return None and the
    # Report itself is the artifact. Pick whichever has save_html.
    artifact = snapshot if hasattr(snapshot, "save_html") else report

    html_path = out_dir / "drift_report.html"
    json_path = out_dir / "drift_summary.json"
    artifact.save_html(str(html_path))

    full = _as_dict(artifact)
    summary = _extract_summary(full)
    json_path.write_text(json.dumps(summary, indent=2))

    return {"html": str(html_path), "json": str(json_path), "summary": summary}


def _as_dict(artifact) -> dict:
    """Get the report payload as a plain dict across Evidently versions."""
    for attr in ("dict", "as_dict", "json"):
        fn = getattr(artifact, attr, None)
        if callable(fn):
            try:
                out = fn()
                # `.json()` returns a string; the others return dict.
                if isinstance(out, str):
                    return json.loads(out)
                return out
            except Exception:
                continue
    return {}


def _extract_summary(report_dict: dict) -> dict:
    """Pull the few fields Phase 8 will key off.

    Two shapes coexist across Evidently versions:
      - 0.6.x: each metric's `result` is a dict (e.g. {'dataset_drift': True,
        'drift_by_columns': {...}}).
      - 0.7+:  each metric's `value` is a scalar (np.float64) and the metric's
        identity is in `metric_name` like 'DriftedColumnsCount(drift_share=0.5)'.

    For Phase 8's purposes we only need a single signal: did drift happen?
    The HTML report has the per-feature drill-down for humans.
    """
    out: dict = {"drift_detected": None, "n_drifted": None, "share_drifted": None, "by_feature": {}}

    for m in report_dict.get("metrics") or []:
        metric_id = m.get("metric") or m.get("metric_id") or m.get("metric_name") or ""
        value = m.get("value")
        result = m.get("result")
        result_dict = result if isinstance(result, dict) else {}

        # 0.6.x shape: dataset-level summary lives in result dict.
        if "dataset_drift" in result_dict:
            out["drift_detected"] = result_dict.get("dataset_drift")
        if "number_of_drifted_columns" in result_dict:
            out["n_drifted"] = result_dict.get("number_of_drifted_columns")
        if "share_of_drifted_columns" in result_dict:
            out["share_drifted"] = result_dict.get("share_of_drifted_columns")

        # 0.6.x shape: per-column table.
        if "drift_by_columns" in result_dict:
            for feature, details in (result_dict.get("drift_by_columns") or {}).items():
                if isinstance(details, dict):
                    out["by_feature"][feature] = {
                        "drift_detected": details.get("drift_detected"),
                        "drift_score": details.get("drift_score"),
                        "stattest": details.get("stattest_name"),
                    }

        # 0.7+ shape: DriftedColumnsCount returns a scalar count.
        if "DriftedColumns" in metric_id and isinstance(value, (int, float)):
            count = int(value)
            out["n_drifted"] = count
            if out["drift_detected"] is None:
                out["drift_detected"] = count > 0

    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline data-drift report.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_LOG)
    parser.add_argument("--reference", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    print(f"Reference data: {args.reference}")
    print(f"Live data:      {args.predictions}")
    reference = load_reference(args.reference)
    current = load_current(args.predictions)
    print(f"Reference rows: {len(reference)}   Live rows: {len(current)}")

    result = run_drift_report(reference, current, args.out)
    s = result["summary"]
    print(f"\nDrift detected: {s['drift_detected']}")
    if s["by_feature"]:
        print(f"Drifted features ({s['n_drifted']}):")
        for feature, info in s["by_feature"].items():
            flag = "DRIFT" if info["drift_detected"] else "ok   "
            score = info.get("drift_score")
            score_s = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
            print(f"  {flag}  {feature:<18}  score={score_s}  test={info.get('stattest')}")
    else:
        # Newer Evidently versions don't surface per-feature stattest in the
        # summary dict by default; the HTML report shows it. We log the
        # top-level signal so callers can still react.
        print("Per-feature breakdown unavailable in this Evidently version's summary dict.")
        print("See the HTML report for the full drill-down.")

    print(f"\nReport:  {result['html']}")
    print(f"Summary: {result['json']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
