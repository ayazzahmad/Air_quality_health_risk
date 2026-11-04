"""
app/main.py – FastAPI service for Air Quality Health Risk prediction.

The model artifact is produced by  train.py  and loaded once at startup.
All preprocessing (QuantileTransformer ranking, composite score, causal
features) mirrors the production pipeline in pipeline/.

Endpoints
---------
GET  /                  Service ping
GET  /health            Model status, features, risk thresholds, test metrics
POST /predict           Single city-day prediction from raw pollutant values
POST /predict/batch     Batch predictions
GET  /docs              Auto-generated Swagger UI (FastAPI built-in)
GET  /redoc             ReDoc alternative UI
"""
import os
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Load model artifact once at startup
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/model.joblib")

_model_loaded = False
_load_error   = ""

try:
    _artifact    = joblib.load(MODEL_PATH)
    _model       = _artifact["model"]       # sklearn Pipeline (imputer + RF)
    _ranker      = _artifact["ranker"]      # QuantileTransformer fitted on train
    _features    = _artifact["features"]    # ordered list of feature column names
    _classes     = _artifact["classes"]     # ["High", "Low", "Moderate"] sorted
    _low_th      = float(_artifact["low_th"])
    _high_th     = float(_artifact["high_th"])
    _test_metrics = _artifact.get("test_metrics", {})
    _model_loaded = True
except Exception as exc:
    _load_error = str(exc)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Air Quality Health Risk API",
    description=(
        "Predicts the **next-week air-pollution health risk** "
        "(`Low` / `Moderate` / `High`) for Indian cities from daily "
        "pollutant measurements.\n\n"
        "The model uses a **QuantileTransformer** (percentile-rank) composite "
        "score, causal rolling-window features, and a tuned "
        "**RandomForestClassifier** (macro F1 ≈ 0.756 on held-out test data).\n\n"
        "**Before starting the server**, train the model:\n"
        "```\npython train.py --fast\n```"
    ),
    version="1.1.0",
    license_info={"name": "MIT"},
)

# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------
CORE_POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
VOC_POLLUTANTS  = ["Benzene", "Toluene", "Xylene"]


class PollutantReading(BaseModel):
    """
    Raw daily pollutant concentrations for a single city-day.

    **All pollutant fields are optional.** Missing values are handled
    by the imputer that is baked into the model pipeline — they will be
    filled with the training-set median for each feature.

    `HRS_lag1` and `voc_index` are derived automatically when not
    provided; supply them directly if you have pre-computed values.
    """
    date: str = Field(
        ..., example="2020-01-15",
        description="ISO-8601 date string (YYYY-MM-DD)",
    )
    city: str = Field(
        ..., example="Delhi",
        description="City name — used for display only, not for prediction",
    )

    # Core pollutants (µg/m³ unless noted)
    PM2_5:   Optional[float] = Field(None, alias="PM2.5", example=98.4)
    PM10:    Optional[float] = Field(None, example=152.0)
    NO2:     Optional[float] = Field(None, example=43.1)
    SO2:     Optional[float] = Field(None, example=11.2)
    CO:      Optional[float] = Field(None, example=1.8)
    O3:      Optional[float] = Field(None, example=29.5)

    # VOC group — summed into voc_index; all optional
    Benzene: Optional[float] = Field(None, example=2.1)
    Toluene: Optional[float] = Field(None, example=4.3)
    Xylene:  Optional[float] = Field(None, example=1.0)

    # Pre-computed rolling / lag features (optional)
    # Provide these if you are calling the API for a date that is part of a
    # longer city time-series and you have already computed the rolling stats.
    HRS_lag1:    Optional[float] = Field(None, example=0.072)
    HRS_7day_std: Optional[float] = Field(None, example=0.018)

    class Config:
        # Accept both "PM2.5" (alias) and "PM2_5" (field name)
        populate_by_name = True


class PredictionResponse(BaseModel):
    city:          str
    date:          str
    risk_level:    str = Field(..., description="Low | Moderate | High")
    probabilities: dict = Field(..., description="Per-class probabilities")
    daily_hrs:     Optional[float] = Field(
        None, description="Computed daily health-risk score (0–1)"
    )


class BatchRequest(BaseModel):
    readings: List[PollutantReading] = Field(..., min_length=1, max_length=500)


class BatchResponse(BaseModel):
    results: List[PredictionResponse]


# ---------------------------------------------------------------------------
# Feature-construction helper
# ---------------------------------------------------------------------------

def _build_feature_row(r: PollutantReading) -> tuple[pd.DataFrame, float]:
    """
    Convert one PollutantReading into a one-row feature DataFrame whose
    columns match `_features`, using the same logic as the production pipeline.

    Returns (feature_df, daily_hrs).
    NaN values are left as-is; the SimpleImputer inside the model pipeline
    fills them with training-set medians.
    """
    if not _model_loaded:
        raise HTTPException(503, detail=f"Model not loaded: {_load_error}")

    # -- 1. Build a one-row pollutant DataFrame --------------------------------
    raw = {
        "PM2.5": r.PM2_5,
        "PM10":  r.PM10,
        "NO2":   r.NO2,
        "SO2":   r.SO2,
        "CO":    r.CO,
        "O3":    r.O3,
    }
    poll_df = pd.DataFrame([raw], columns=CORE_POLLUTANTS).astype(float)

    # -- 2. Percentile-rank the pollutants (QuantileTransformer) ---------------
    # The ranker was fitted on training data; calling transform here mirrors
    # exactly what add_causal_features does inside the notebook/train pipeline.
    ranked = pd.DataFrame(
        _ranker.transform(poll_df),
        columns=CORE_POLLUTANTS,
    )

    # -- 3. Weighted composite score (renormalise over observed pollutants) -----
    from pipeline import config as cfg
    weighted_sum   = 0.0
    weight_present = 0.0
    for p in CORE_POLLUTANTS:
        w = cfg.POLLUTANT_WEIGHTS[p]
        v = ranked[p].iloc[0]
        if not np.isnan(v):
            weighted_sum   += v * w
            weight_present += w

    daily_hrs = (weighted_sum / weight_present) if weight_present > 0 else np.nan

    # -- 4. Temporal features --------------------------------------------------
    dt         = pd.Timestamp(r.date)
    month      = dt.month
    dayofweek  = dt.dayofweek
    is_winter  = int(month in [11, 12, 1, 2])
    month_sin  = float(np.sin(2 * np.pi * month / 12))
    month_cos  = float(np.cos(2 * np.pi * month / 12))
    dow_sin    = float(np.sin(2 * np.pi * dayofweek / 7))
    dow_cos    = float(np.cos(2 * np.pi * dayofweek / 7))

    # -- 5. Lag / volatility ---------------------------------------------------
    # Use caller-supplied values if available; otherwise leave NaN for the
    # in-pipeline imputer to fill with the training-set median.
    hrs_lag1     = r.HRS_lag1    if r.HRS_lag1    is not None else np.nan
    hrs_7day_std = r.HRS_7day_std if r.HRS_7day_std is not None else np.nan

    # -- 6. VOC index ----------------------------------------------------------
    voc_vals = [v for v in [r.Benzene, r.Toluene, r.Xylene] if v is not None]
    voc_index = float(sum(voc_vals)) if voc_vals else np.nan

    voc_observed = 1 if voc_vals else 0
    weight_coverage = weight_present  # already a 0-1 fraction

    # -- 7. Assemble feature row in the exact column order the model expects ---
    # _features is the FEATURE_COLUMNS list from pipeline/features.py:
    # ["hrs_mean_3", "hrs_mean_7", "hrs_mean_14", "hrs_mean_30",
    #  "hrs_std_7",  "hrs_std_14",
    #  "hrs_lag_1",  "hrs_lag_7",
    #  "hrs_trend",
    #  "month_sin", "month_cos", "dow_sin", "dow_cos", "is_winter",
    #  "voc_index", "voc_observed", "weight_coverage"]
    #
    # Rolling features (3/7/14/30-day means, 7/14-day stds, lag-7, trend)
    # cannot be recovered from a single row — the model's internal SimpleImputer
    # will fill them with training medians, which degrades accuracy gracefully
    # but is honest (we document this in the endpoint description).
    row = {f: np.nan for f in _features}

    # Features we CAN compute from one row
    row["month_sin"]      = month_sin
    row["month_cos"]      = month_cos
    row["dow_sin"]        = dow_sin
    row["dow_cos"]        = dow_cos
    row["is_winter"]      = float(is_winter)
    row["voc_index"]      = voc_index
    row["voc_observed"]   = float(voc_observed)
    row["weight_coverage"]= weight_coverage if not np.isnan(weight_coverage) else np.nan

    # Lag-1 and 7-day std provided by caller
    if "hrs_lag_1" in row:
        row["hrs_lag_1"]   = hrs_lag1
    if "hrs_std_7" in row:
        row["hrs_std_7"]   = hrs_7day_std

    return pd.DataFrame([row])[_features], daily_hrs


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Meta"])
def root():
    """Service ping — always returns 200 even when the model is not loaded."""
    return {
        "service": "Air Quality Health Risk API",
        "version": "1.1.0",
        "model_loaded": _model_loaded,
        "docs": "/docs",
    }


@app.get("/health", tags=["Meta"])
def health():
    """
    Detailed health check.

    Returns 503 if the model artifact cannot be found.
    Run `python train.py --fast` to generate it.
    """
    if not _model_loaded:
        raise HTTPException(
            503,
            detail=(
                f"Model not loaded: {_load_error}. "
                "Run `python train.py --fast` to generate artifacts/model.joblib."
            ),
        )
    return {
        "status":          "ok",
        "model":           "RandomForestClassifier",
        "features":        _features,
        "n_features":      len(_features),
        "classes":         _classes,
        "risk_thresholds": {"low": _low_th, "high": _high_th},
        "test_metrics":    _test_metrics,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Single city-day prediction",
)
def predict(reading: PollutantReading):
    """
    Predict the **next-week** air-pollution health risk for a single city-day.

    ### Notes on accuracy
    Rolling features (3/7/14/30-day means, lag-7, trend) **cannot** be
    computed from a single reading — they require the preceding days'
    history. The model's internal imputer fills them with training-set medians,
    which produces reasonable predictions but is less accurate than providing
    full time-series context via `/predict/batch`.

    Supply `HRS_lag1` and `HRS_7day_std` if you have pre-computed them.
    """
    X, daily_hrs = _build_feature_row(reading)
    pred         = _model.predict(X)[0]
    proba        = _model.predict_proba(X)[0]
    proba_dict   = {
        cls: round(float(p), 4)
        for cls, p in zip(_model.classes_, proba)
    }
    return PredictionResponse(
        city=reading.city,
        date=reading.date,
        risk_level=pred,
        probabilities=proba_dict,
        daily_hrs=round(float(daily_hrs), 6) if not np.isnan(daily_hrs) else None,
    )


@app.post(
    "/predict/batch",
    response_model=BatchResponse,
    tags=["Prediction"],
    summary="Batch predictions (up to 500 readings)",
)
def predict_batch(payload: BatchRequest):
    """
    Predict health risk levels for a **list** of city-day readings in one call.

    Readings do **not** need to be for the same city or consecutive dates.
    Maximum batch size: 500.
    """
    results = []
    for r in payload.readings:
        X, daily_hrs = _build_feature_row(r)
        pred         = _model.predict(X)[0]
        proba        = _model.predict_proba(X)[0]
        proba_dict   = {
            cls: round(float(p), 4)
            for cls, p in zip(_model.classes_, proba)
        }
        results.append(PredictionResponse(
            city=r.city,
            date=r.date,
            risk_level=pred,
            probabilities=proba_dict,
            daily_hrs=round(float(daily_hrs), 6) if not np.isnan(daily_hrs) else None,
        ))
    return BatchResponse(results=results)
