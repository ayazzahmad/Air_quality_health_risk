# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – trainer
#   Installs the ML deps, trains the model, and writes artifacts/model.joblib
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.13-slim AS trainer

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source
COPY data/       data/
COPY pipeline/   pipeline/
COPY train.py    .

# Train the model (fast mode uses the tuned hyperparameters, skips grid-search;
# the result is identical to the tuned model). Remove --fast to rerun the full
# grid-search (several minutes of extra build time).
RUN python train.py --data data/city_day.csv --out artifacts/model.joblib --fast


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 – api
#   Minimal image: only the FastAPI app + the trained artifact
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.13-slim AS api

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_PATH=artifacts/model.joblib

WORKDIR /app

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# Copy the trained artifact from the trainer stage
COPY --from=trainer /app/artifacts/ artifacts/

# Copy the API and pipeline source (the API reuses pipeline/ for features)
COPY app/      app/
COPY pipeline/ pipeline/

# Run as an unprivileged user
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

# python:slim has no curl, so the health check uses the standard library
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=5).status == 200 else 1)"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
