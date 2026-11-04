"""
app/main.py – FastAPI service for Air Quality Health Risk prediction.

Endpoints
---------
GET  /              Health check
GET  /health        Detailed health (model loaded, feature list)
POST /predict       Single-row prediction from raw pollutant readings
POST /predict/batch Batch prediction (list of readings)
GET  /docs          Auto-generated Swagger UI  (provided by FastAPI)
"""
import os
import pathlib
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Load model artifact ──────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/model.joblib")

try:
    _artifact = joblib.load(MODEL_PATH)
    _model         = _artifact["model"]
    _scaler        = _artifact["scaler"]
    _features      = _artifact["features"]
    _classes       = _artifact["classes"]
    _low_th        = _artifact["low_th"]
    _high_th       = _artifact["high_th"]
    _train_medians = _artifact["train_medians"]
    _model_loaded  = True
except Exception as e:
    _model_loaded = False
    _load_error   = str(e)

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Air Quality Health Risk API",
    description=(
        "Predicts the **next-week air-pollution health risk** (Low / Moderate / High) "
        "for Indian cities from daily pollutant measurements.\n\n"
        "Train the model first with `python train.py`, then start the server with "
        "`uvicorn app.main:app --reload`."
    ),
    version="1.0.0",
)


# ── Schemas ──────────────────────────────────────────────────────────────────
class PollutantReading(BaseModel):
    """
    Raw daily pollutant concentrations for a single city-day.
    All fields are optional; missing values will be imputed with training medians.
    """
    date: str = Field(..., example="2020-01-15",
                      description="ISO date string (YYYY-MM-DD)")
    city: str = Field(..., example="Delhi",
                      description="City name (used only for display)")

    # Pollutants used for the HRS composite score
    PM2_5: Optional[float] = Field(None, alias="PM2.5", example=98.4)
    PM10:  Optional[float] = Field(None, example=152.0)
    NO2:   Optional[float] = Field(None, example=43.1)
    SO2:   Optional[float] = Field(None, example=11.2)
    CO:    Optional[float] = Field(None, example=1.8)
    O3:    Optional[float] = Field(None, example=29.5)

    # VOC group – used for VOC_index feature
    Benzene: Optional[float] = Field(None, example=2.1)
    Toluene: Optional[float] = Field(None, example=4.3)
    Xylene:  Optional[float] = Field(None, example=1.0)

    # Pre-computed rolling / lag features (optional – will be computed from
    # the pollutants above if not supplied)
    HRS_lag1:    Optional[float] = Field(None, example=0.072)
    HRS_7day_std: Optional[float] = Field(None, example=0.018)

    class Config:
        populate_by_name = True  # allow both alias and field name


class PredictionResponse(BaseModel):
    city:        str
    date:        str
    risk_level:  str
    probabilities: dict


class BatchRequest(BaseModel):
    readings: List[PollutantReading]


class BatchResponse(BaseModel):
    results: List[PredictionResponse]


# ── Helpers ──────────────────────────────────────────────────────────────────
POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]


def _reading_to_feature_vector(r: PollutantReading) -> pd.DataFrame:
    """
    Convert a PollutantReading into a one-row DataFrame with the 8 model features.
    Missing values are filled with training-set medians.
    """
    if not _model_loaded:
        raise HTTPException(503, detail=f"Model not loaded: {_load_error}")

    # --- build raw pollutant row ---
    raw = {
        "PM2.5": r.PM2_5,
        "PM10":  r.PM10,
        "NO2":   r.NO2,
        "SO2":   r.SO2,
        "CO":    r.CO,
        "O3":    r.O3,
    }
    raw_df = pd.DataFrame([raw])

    # --- MinMax scale the pollutants (using saved scaler) ---
    for col, val in raw.items():
        if val is None:
            # will become NaN → impute with scaler center ≈ training median
            raw_df[col] = np.nan

    # Replace NaNs with column medians before scaling
    for col in POLLUTANTS:
        if raw_df[col].isna().any():
            raw_df[col] = _train_medians.get(col, 0)  # fallback 0

    normed = _scaler.transform(raw_df[POLLUTANTS])
    PM25_n, PM10_n, NO2_n, SO2_n, CO_n, O3_n = normed[0]

    daily_hrs = (
        0.35 * PM25_n +
        0.20 * PM10_n +
        0.15 * NO2_n  +
        0.10 * SO2_n  +
        0.10 * CO_n   +
        0.10 * O3_n
    )

    # --- temporal features ---
    dt = pd.Timestamp(r.date)
    month      = dt.month
    dayofweek  = dt.dayofweek
    is_winter  = int(month in [11, 12, 1, 2])
    month_sin  = np.sin(2 * np.pi * month / 12)
    month_cos  = np.cos(2 * np.pi * month / 12)
    dow_sin    = np.sin(2 * np.pi * dayofweek / 7)
    dow_cos    = np.cos(2 * np.pi * dayofweek / 7)

    # --- lag / std – use provided values or fall back to training median ---
    hrs_lag1     = r.HRS_lag1    if r.HRS_lag1    is not None else _train_medians.get("HRS_lag1",    daily_hrs)
    hrs_7day_std = r.HRS_7day_std if r.HRS_7day_std is not None else _train_medians.get("HRS_7day_std", 0.01)

    # --- VOC index ---
    voc_vals = [v for v in [r.Benzene, r.Toluene, r.Xylene] if v is not None]
    voc_index = sum(voc_vals) if voc_vals else _train_medians.get("VOC_index", 0.0)

    row = {
        "month_sin":      month_sin,
        "month_cos":      month_cos,
        "dayofweek_sin":  dow_sin,
        "dayofweek_cos":  dow_cos,
        "is_winter":      is_winter,
        "HRS_lag1":       hrs_lag1,
        "HRS_7day_std":   hrs_7day_std,
        "VOC_index":      voc_index,
    }
    return pd.DataFrame([row])[_features]


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", tags=["Meta"])
def root():
    return {
        "service": "Air Quality Health Risk API",
        "status":  "ok",
        "docs":    "/docs",
    }


@app.get("/health", tags=["Meta"])
def health():
    if not _model_loaded:
        raise HTTPException(503, detail=f"Model not loaded: {_load_error}")
    return {
        "status":        "ok",
        "model":         "RandomForestClassifier",
        "classes":       _classes,
        "features":      _features,
        "risk_thresholds": {"low": _low_th, "high": _high_th},
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(reading: PollutantReading):
    """
    Predict the health risk level for a **single** city-day reading.

    - `risk_level` → one of `Low`, `Moderate`, `High`
    - `probabilities` → model class probabilities
    """
    X = _reading_to_feature_vector(reading)
    pred       = _model.predict(X)[0]
    proba      = _model.predict_proba(X)[0]
    proba_dict = dict(zip(_model.classes_, [round(float(p), 4) for p in proba]))
    return PredictionResponse(
        city=reading.city,
        date=reading.date,
        risk_level=pred,
        probabilities=proba_dict,
    )


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch(payload: BatchRequest):
    """
    Predict health risk levels for a **list** of city-day readings in one call.
    """
    results = []
    for r in payload.readings:
        X     = _reading_to_feature_vector(r)
        pred  = _model.predict(X)[0]
        proba = _model.predict_proba(X)[0]
        proba_dict = dict(zip(_model.classes_, [round(float(p), 4) for p in proba]))
        results.append(PredictionResponse(
            city=r.city,
            date=r.date,
            risk_level=pred,
            probabilities=proba_dict,
        ))
    return BatchResponse(results=results)

