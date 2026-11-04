# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – trainer
#   Installs heavy ML deps, trains the model, and writes artifacts/model.joblib
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS trainer

WORKDIR /app

# System deps (needed by some numpy/scipy wheels on slim images)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY data/       data/
COPY pipeline/   pipeline/
COPY train.py    .

# Train the model → produces artifacts/model.joblib
RUN python train.py --data data/city_day.csv --out artifacts/model.joblib


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 – api
#   Minimal image: only the FastAPI app + the trained artifact
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS api

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy the trained artifact from the trainer stage
COPY --from=trainer /app/artifacts/ artifacts/

# Copy the API and pipeline source
COPY app/      app/
COPY pipeline/ pipeline/

# Expose the API port
EXPOSE 8000

# MODEL_PATH can be overridden via docker run -e MODEL_PATH=...
ENV MODEL_PATH=artifacts/model.joblib

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

