# Predictive Maintenance MLOps Pipeline

End-to-end MLOps pipeline for predicting industrial machine failure from sensor data.
The model is a placeholder — the **production loop** (train → serve → containerize → deploy
→ monitor → auto-retrain) is the deliverable.

## Phase status

| # | Phase | Status |
|---|---|---|
| 0 | Repo + env setup | Done |
| 1 | Data + baseline model (XGBoost F1 = 0.768 on test) | Done |
| 2 | MLflow experiment tracking + registry | In progress |
| 3 | FastAPI serving (`/health`, `/predict`) | Pending |
| 4 | Docker + docker-compose (API + MLflow + monitoring stack) | Pending |
| 5 | Kubernetes deploy (kind/minikube → AWS EKS) | Pending |
| 6 | CI/CD via GitHub Actions | Pending |
| 7 | Prometheus + Grafana + Evidently drift reports | Pending |
| 8 | Drift-triggered / scheduled auto-retraining (+ **DVC** for dataset versioning here) | Pending |
| 9 | Terraform IaC + architecture diagram + demo | Pending |

> **DVC note:** intentionally deferred from Phase 2 to Phase 8. Versioning a single
> 522 KB static CSV is busywork; DVC earns its keep when the retrain loop starts
> producing new dataset snapshots.

## Dataset

**Source:** UCI Machine Learning Repository — AI4I 2020 Predictive Maintenance Dataset
https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

| Field | Value |
|---|---|
| File | `data/raw/ai4i2020.csv` |
| Rows | 10,000 |
| Size | 522,048 bytes |
| SHA-256 | `dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e` |
| Target | `Machine failure` (binary, ~3.4% positive class) |
| Sub-targets | TWF, HDF, PWF, OSF, RNF (failure-type labels) |

Verify with:

```bash
shasum -a 256 data/raw/ai4i2020.csv
```

## Repo layout

```
.
├── data/raw/             # immutable raw dataset (not versioned via DVC yet — see note above)
├── notebooks/            # EDA, scratch
├── src/
│   ├── data/             # loading, preprocessing
│   ├── training/         # training pipelines
│   └── serving/          # FastAPI app (Phase 3+)
├── tests/                # pytest
├── configs/              # hyperparams, env configs
├── requirements.txt
└── README.md
```

## Setup (Phase 0)

```bash
# 1. Create + activate venv
python3 -m venv .venv
source .venv/bin/activate

# 2. Install deps
pip install --upgrade pip
pip install -r requirements.txt

# 3. Sanity check
python -c "import pandas, sklearn, xgboost, imblearn, mlflow; print('OK')"
```

## MLflow (Phase 2)

Every training run logs to a local SQLite store (`mlflow.db`) — registry works
with SQLite but **not** with the default file backend.

```bash
# Train both models, register the winner under @staging
python -m src.training.train_baseline

# Browse experiments + registry in your browser (http://127.0.0.1:5000)
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

The winning model is registered as `predictive_maintenance` and the
`@staging` alias is pointed at the new version. **Phase 3's FastAPI service
will load `models:/predictive_maintenance@staging`** — that handle stays
stable no matter how many times we retrain.

## Conventions

- **Metric of record:** F1 on the minority class (failure). Accuracy is misleading on a
  ~3.4%-positive dataset.
- **Model registry:** MLflow Model Registry (Phase 2 onward). Don't commit `.pkl` files.
- **Reproducibility:** every training run logs params, metrics, and the dataset hash to MLflow.
