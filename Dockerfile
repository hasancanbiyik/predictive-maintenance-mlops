# syntax=docker/dockerfile:1.7
#
# Multi-stage build for the FastAPI serving container.
#
# Why multi-stage:
#  - Stage 1 has the build toolchain (gcc, headers) needed to compile any wheels
#    that don't ship binaries. We throw it away.
#  - Stage 2 is a clean python:slim with just the installed site-packages and
#    our code. ~200MB instead of ~1.2GB.
#
# Why no model weights in the image:
#  - The model is fetched at runtime from the MLflow server via the registry
#    alias (`models:/predictive_maintenance@staging`). The image is dataset-
#    and version-agnostic. Retraining doesn't require an image rebuild.

# ---------- Stage 1: builder ----------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install only what's needed to *build* wheels (not what runtime needs).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .

# Install into an isolated prefix so we can copy it cleanly into the runtime
# stage. `--prefix` keeps everything under /install for easy COPY.
RUN pip install --prefix=/install -r requirements.txt


# ---------- Stage 2: runtime ----------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # API talks to MLflow over HTTP inside the compose network
    MLFLOW_TRACKING_URI=http://mlflow:5000 \
    MODEL_NAME=predictive_maintenance \
    MODEL_ALIAS=staging \
    DECISION_THRESHOLD=0.5

# Non-root user. Containers running as root are a long-standing source of
# CVE exposure; this is the kind of detail K8s security reviewers flag.
RUN useradd --create-home --uid 10001 appuser

# Bring in the installed packages from the builder stage.
COPY --from=builder /install /usr/local

WORKDIR /app
COPY src/ ./src/

USER appuser
EXPOSE 8000

# A failing healthcheck lets Docker / K8s notice a broken pod without us
# writing extra liveness logic. Note: this is Docker's HEALTHCHECK; K8s uses
# its own probe config (Phase 5) but pointing at the same endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
