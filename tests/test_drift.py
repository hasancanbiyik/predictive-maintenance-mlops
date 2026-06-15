"""
Smoke tests for the Evidently drift script. Generates a small synthetic
prediction log so the test doesn't depend on a running API.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.monitoring import drift


def _make_synthetic_log(path: Path, n: int = 50) -> None:
    """Synthetic JSONL log matching what the API writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for i in range(n):
            rec = {
                "ts": "2026-06-14T00:00:00+00:00",
                "model_version": "1",
                "model_alias": "staging",
                "probability": 0.1 + (i % 9) * 0.1,
                "type": ["L", "M", "H"][i % 3],
                "air_temp_k": 298.0 + (i % 7),
                "process_temp_k": 308.0 + (i % 7),
                "rot_speed_rpm": 1500 + (i % 100),
                "torque_nm": 40.0 + (i % 30),
                "tool_wear_min": (i * 5) % 250,
            }
            f.write(json.dumps(rec) + "\n")


def test_drift_script_runs_end_to_end(tmp_path, monkeypatch):
    log_path = tmp_path / "predictions.jsonl"
    out_dir = tmp_path / "drift"
    _make_synthetic_log(log_path)

    # Run from project root so DEFAULT_CSV resolves.
    monkeypatch.chdir(Path(__file__).resolve().parents[1])

    rc = drift.main([
        "--predictions", str(log_path),
        "--out", str(out_dir),
    ])
    assert rc == 0
    assert (out_dir / "drift_report.html").exists()
    summary_path = out_dir / "drift_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert "drift_detected" in summary
    assert "by_feature" in summary
