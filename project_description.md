# Predictive-Maintenance MLOps Pipeline — Build Guide

**Goal:** Build *new* — a deployed, monitored, self-retraining failure-prediction service on
industrial sensor data. This is the project that reframes you from "NLP researcher" to
"engineer who ships production ML," and gives you manufacturing-domain signal.

**The mindset for the whole thing:** the model is a placeholder; the *production loop* is the
deliverable. Nobody hiring for these roles cares about your F1 on a public dataset. They care
that you can deploy, monitor, detect drift, retrain, and define infrastructure as code. Spend
your effort there, not on squeezing model accuracy.

---

## Dataset

- **Start here: UCI AI4I 2020 Predictive Maintenance.** ~10K rows, tabular, binary "machine
  failure" target (with failure-type sub-labels). A RandomForest/XGBoost gets you a working
  model in an afternoon. Class-imbalanced (failures are rare) — good excuse to show SMOTE /
  class weighting.
- **Optional upgrade: NASA CMAPSS** (turbofan remaining-useful-life). Time-series, needs an
  LSTM/GRU. More impressive *as a model*, but more time sunk in the part that matters least.
  Do this only as a v2 if you want a deep-learning variant.

---

## Resources (fork / follow — ranked)

1. **MLOps Zoomcamp — DataTalks.Club (FREE, the spine).** 6 modules + a portfolio capstone:
   Docker, MLflow tracking, deployment (batch/real-time), monitoring with Prometheus / Grafana /
   Evidently, CI/CD with GitHub Actions. Work through it, but use the predictive-maintenance
   dataset as your capstone so the course output *is* this project.
   `datatalks.club/blog/mlops-zoomcamp.html`
2. **Best fork base — `Sa1f27/predictive-maintenance-mlops`.** Closest to the target stack:
   sensor data (temp/torque/speed/tool-wear — AI4I-style), scikit-learn + SMOTE, FastAPI +
   Pydantic, MLflow registry, Docker + Compose, GitHub Actions CI/CD, AWS ECR/ECS, drift
   detection. Read it, rebuild it your way, swap the dataset, understand every line.
3. **Kubernetes reference — `simonbbbb/mlops-predictive-maintenance`** (Balázs Simon, write-up on
   blog.devops.dev). Use this for the K8s deployment piece specifically.
4. **Step-by-step walkthrough — Analytics Vidhya, "Machine Predictive Maintenance with MLOps"**
   (Karthik Ponna, Mar 2025). Docker + FastAPI + AWS + GitHub Actions, narrated end to end.
5. **Also useful:** `typhonshambo/End-to-End-ML-Pipeline-for-Predictive-Maintenance`
   (RandomForest + MLflow), `Abhi0323/Agile-MLOps-Deployment-Docker-AWS-CI-CD-Pipeline`
   (modular Docker + AWS + CI/CD).
6. **CMAPSS model bootstraps (only if you do the LSTM variant):** see GitHub topic `cmapss`.
7. **Terraform (if you need to learn IaC):** Data Engineering Zoomcamp (DataTalks.Club) teaches
   Docker + Terraform directly.

> On forking: cloning a template to learn from is normal and fine. But rebuild it, change the
> data and structure, and be able to explain it as your own work. A verbatim clone you can't
> walk through is worse than no project.

---

## The phased build

Build a **thin vertical slice first** (Phases 1–5: model → API → container → one deployment),
get it *actually running*, then layer on the impressive parts (6–9). A deployed simple pipeline
beats a half-built sophisticated one every time.

### Phase 0 — Setup
- **Do:** Create a GitHub repo. Set up a modular structure (`src/data/`, `src/training/`,
  `src/serving/`, `tests/`). Pin a Python env (`requirements.txt` or `pyproject`). Start MLOps
  Zoomcamp modules 1–2.
- **Proves:** Git/GitHub, modular code organization, reproducible environments.

### Phase 1 — Data + baseline model *(keep this fast)*
- **Do:** Load AI4I 2020. Quick EDA. Train RandomForest or XGBoost. Handle class imbalance
  (SMOTE or `class_weight`). Evaluate with precision/recall/F1 (not accuracy — failures are
  rare). Stop as soon as it's decent.
- **Proves:** scikit-learn / XGBoost, feature engineering, class-imbalance handling, model
  evaluation.

### Phase 2 — Experiment tracking + data versioning
- **Do:** Wrap training in **MLflow** (log params, metrics, artifacts; register the model in the
  MLflow Model Registry). Version the dataset + model with **DVC**.
- **Proves:** MLflow, DVC, experiment tracking, reproducibility, model registry.

### Phase 3 — Serve it
- **Do:** Wrap the registered model in a **FastAPI** service: `/health` + `/predict`, with
  **Pydantic** request/response validation. Test locally.
- **Proves:** FastAPI, REST API design, Pydantic, model serving.

### Phase 4 — Containerize
- **Do:** Write a (multi-stage) **Dockerfile**. Add **docker-compose** to run API + MLflow +
  monitoring locally as one stack.
- **Proves:** Docker, containerization, multi-service local orchestration.

### Phase 5 — Deploy *(this is where your Kubernetes claim becomes real)*
- **Do:** Deploy to **Kubernetes**. Acceptable: a local cluster (minikube/kind) with manifests
  you wrote yourself (Deployment, Service, ConfigMap). Better: managed cloud — push image to
  **AWS ECR**, run on **EKS** (or ECS if you want simpler). Write the YAML yourself; understand
  pods, services, deployments, scaling.
- **Proves:** Kubernetes, cloud deployment, container registries, scaling. *(If asked about K8s
  in an interview, this project is your answer — so actually run it on a cluster, don't just
  commit YAML.)*

### Phase 6 — CI/CD
- **Do:** **GitHub Actions** workflow: lint → run **pytest** → build image → push to registry →
  deploy. Trigger on push to main.
- **Proves:** CI/CD, GitHub Actions, automated testing, release automation.

### Phase 7 — Monitoring + drift detection
- **Do:** **Prometheus** scrapes API/system metrics; **Grafana** dashboards them. Add
  **Evidently AI** to generate data-drift / prediction-drift reports comparing live inputs to
  the training distribution.
- **Proves:** monitoring, observability, Prometheus, Grafana, Evidently, data/model drift
  detection.

### Phase 8 — Automated retraining *(closes the loop — the "automation engineer" payoff)*
- **Do:** When Evidently flags drift past a threshold (or on a schedule), trigger a retraining
  job → evaluate → if better, register the new version → redeploy. Even a scheduled GitHub
  Actions / cron-triggered retrain that re-registers the model demonstrates the concept.
- **Proves:** automated retraining, drift-triggered pipelines, full ML lifecycle ownership.

### Phase 9 — Infrastructure-as-code + polish
- **Do:** Define the cloud infra (cluster, registry, IAM/roles) in **Terraform**. Write a README
  with an **architecture diagram** and a **live demo link** (or a short Loom walkthrough if you
  tear the cloud resources down to save cost).
- **Proves:** Terraform, IaC, technical documentation, system design communication.

---

## What to put on the resume (one bullet group)

> **Predictive-Maintenance MLOps Pipeline** — built and deployed an end-to-end failure-prediction
> service on industrial sensor data: model (scikit-learn/XGBoost) tracked with MLflow and
> versioned with DVC, served via FastAPI, containerized with Docker and deployed on Kubernetes
> (AWS), with GitHub Actions CI/CD, Prometheus/Grafana + Evidently drift monitoring, and
> automated drift-triggered retraining; infrastructure defined in Terraform.

That single entry legitimately carries: scikit-learn, MLflow, DVC, FastAPI, Docker, Kubernetes,
AWS, CI/CD, GitHub Actions, Prometheus, Grafana, Evidently, drift detection, retraining,
Terraform. Plus the manufacturing domain.

## LinkedIn (recruiter attention)

Post a thread, one stage at a time, with the architecture diagram up top:
*"End-to-end MLOps for predictive maintenance: model → Docker → K8s → CI/CD → drift monitoring →
auto-retrain. Here's how each piece fits."* Tag the tools. This is the content that pulls eyes
off your NLP-only profile.

---

## Honest cautions

1. **Scope creep is the #1 failure mode.** Ship the thin slice (Phases 1–5) end-to-end before
   touching monitoring/retraining/Terraform. "Deployed and simple" > "sophisticated and
   half-finished."
2. **Don't gold-plate the model.** 90% on AI4I is plenty. Time spent chasing 93% is time stolen
   from the part that gets you hired.
3. **It has to be real and explainable.** You must be able to demo it live (or on video) and
   explain every box in the diagram — these become interview questions. A README-only project
   that doesn't run is a liability, not an asset.
4. **Kubernetes: make the claim true here.** You added K8s to your resume; this project is where
   you earn it. Run it on an actual cluster and write the manifests yourself.
5. **Cost:** managed cloud (EKS) costs money idle. Build it, capture the demo/screenshots, then
   tear it down with `terraform destroy`. The Terraform code + recording is the proof; you don't
   need it running 24/7.
