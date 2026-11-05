"""
app/main.py – FastAPI service for Air Quality Health Risk prediction.

The model artifact is produced by  train.py  and loaded once at startup.
Features are built with the same functions the training pipeline uses
(pipeline/data.py and pipeline/features.py), so serving cannot drift from
training.

Endpoints
---------
GET  /                  Service ping
GET  /health            Model status, features, risk thresholds, test metrics
POST /predict           Single city-day prediction from raw pollutant values
POST /predict/batch     Batch predictions; consecutive days of a city are used
                        as history for the rolling features
GET  /docs              Auto-generated Swagger UI (FastAPI built-in)
GET  /redoc             ReDoc alternative UI
"""
import datetime as dt
import os
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from pipeline import config as cfg
from pipeline import data as data_mod
from pipeline import features as feat

API_VERSION = "1.3.0"

# ---------------------------------------------------------------------------
# Load model artifact once at startup
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/model.joblib")

_model_loaded = False
_load_error   = ""
_regressor    = None   # optional: artifacts from before v1.3 have no regressor

try:
    _artifact     = joblib.load(MODEL_PATH)
    _model        = _artifact["model"]       # sklearn Pipeline (imputer + RF)
    _ranker       = _artifact["ranker"]      # QuantileTransformer fitted on train
    _features     = _artifact["features"]    # ordered list of feature column names
    _classes      = _artifact["classes"]     # model.classes_ order
    _low_th       = float(_artifact["low_th"])
    _high_th      = float(_artifact["high_th"])
    _test_metrics = _artifact.get("test_metrics", {})
    _regressor    = _artifact.get("regressor")           # next-week exposure change
    _reg_features = _artifact.get("regressor_features", [])
    _reg_metrics  = _artifact.get("regressor_metrics", {})
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
        "pollutant measurements, plus a forecast of next week's mean "
        "**exposure score** (0–1) that also uses daily weather.\n\n"
        "The models use a **QuantileTransformer** (percentile-rank) composite "
        "score, causal rolling-window features, a tuned "
        "**RandomForestClassifier** for the risk level and a "
        "**RandomForestRegressor** for the exposure forecast. Held-out test "
        "metrics for both are reported by `GET /health`.\n\n"
        "**Before starting the server**, train the model:\n"
        "```\npython train.py --fast\n```"
    ),
    version=API_VERSION,
    license_info={"name": "MIT"},
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
# A batch may span long gaps; gaps wider than the longest feature window are
# split into separate segments instead of being filled with empty days.
_MAX_LOOKBACK_DAYS = max(cfg.ROLLING_WINDOWS + cfg.VOLATILITY_WINDOWS + cfg.LAGS)


def _conc(example: float, alias: Optional[str] = None):
    """Optional, non-negative pollutant concentration field."""
    return Field(None, alias=alias, ge=0, examples=[example])


class PollutantReading(BaseModel):
    """
    Raw daily pollutant concentrations for a single city-day.

    **All pollutant fields are optional.** As in training, a day whose
    observed pollutants cover less than 60% of the composite weight gets no
    exposure score; missing features are filled with training-set medians by
    the imputer inside the model pipeline.

    `HRS_lag1` and `HRS_7day_std` are only used when they cannot be computed
    from earlier readings of the same city in the request.

    **Weather fields are optional** and only feed the exposure forecast. Use
    daily values for the same city and date (e.g. from Open-Meteo); missing
    weather is filled with training-set medians.
    """
    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    date: dt.date = Field(
        ..., examples=["2020-01-15"],
        description="ISO-8601 date (YYYY-MM-DD)",
    )
    city: str = Field(
        ..., min_length=1, examples=["Delhi"],
        description="City name — groups readings into a time series; "
                    "not itself a model feature",
    )

    # Core pollutants (µg/m³; CO in mg/m³)
    PM2_5:   Optional[float] = _conc(98.4, alias="PM2.5")
    PM10:    Optional[float] = _conc(152.0)
    NO2:     Optional[float] = _conc(43.1)
    SO2:     Optional[float] = _conc(11.2)
    CO:      Optional[float] = _conc(1.8)
    O3:      Optional[float] = _conc(29.5)

    # VOC group — summed into voc_index; all optional
    Benzene: Optional[float] = _conc(2.1)
    Toluene: Optional[float] = _conc(4.3)
    Xylene:  Optional[float] = _conc(1.0)

    # Pre-computed history features (optional fallbacks)
    HRS_lag1:     Optional[float] = Field(
        None, ge=0, le=1, examples=[0.72],
        description="Previous day's composite score (0–1)",
    )
    HRS_7day_std: Optional[float] = Field(
        None, ge=0, examples=[0.05],
        description="Std. dev. of the composite score over the last 7 days",
    )

    # Daily weather (optional; exposure forecast only)
    temp_mean:     Optional[float] = Field(
        None, ge=-60, le=60, examples=[15.4], description="Mean temperature at 2 m (°C)")
    precip_sum:    Optional[float] = Field(
        None, ge=0, examples=[0.0], description="Total precipitation (mm)")
    wind_mean:     Optional[float] = Field(
        None, ge=0, examples=[9.8], description="Mean wind speed at 10 m (km/h)")
    wind_dir:      Optional[float] = Field(
        None, ge=0, le=360, examples=[288], description="Dominant wind direction (degrees)")
    humidity_mean: Optional[float] = Field(
        None, ge=0, le=100, examples=[70], description="Mean relative humidity (%)")
    blh_max:       Optional[float] = Field(
        None, ge=0, examples=[996], description="Daily maximum boundary-layer height (m)")


# Weather inputs the exposure features are built from
_WEATHER_INPUTS = cfg.WEATHER_FEATURE_VARS + ["wind_dir"]


class PredictionResponse(BaseModel):
    city:          str
    date:          dt.date
    risk_level:    str = Field(..., description="Low | Moderate | High")
    probabilities: Dict[str, float] = Field(..., description="Per-class probabilities")
    daily_hrs:     Optional[float] = Field(
        None, description="Computed daily health-risk score (0–1); null when "
                          "too few pollutants were observed",
    )
    history_days:  int = Field(
        ..., description="Earlier days of this city available in the request "
                         "for the rolling features (30 gives full context)",
    )
    forecast_exposure: Optional[float] = Field(
        None, description="Forecast mean exposure score (0–1) over the next 7 "
                          "days; null without 7 days of scored history",
    )


class BatchRequest(BaseModel):
    readings: List[PollutantReading] = Field(..., min_length=1, max_length=500)


class BatchResponse(BaseModel):
    results: List[PredictionResponse]


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def _require_model() -> None:
    if not _model_loaded:
        raise HTTPException(
            503,
            detail=(
                f"Model not loaded: {_load_error}. "
                "Run `python train.py --fast` to generate artifacts/model.joblib."
            ),
        )


def _build_features(readings: List[PollutantReading]) -> pd.DataFrame:
    """
    Turn readings into model features using the training pipeline's own
    functions. Returns one row per reading, in request order, with the
    feature columns plus `hrs` and `history_days`.

    Readings of the same city are treated as one daily time series: the frame
    is reindexed to a contiguous calendar (so row-based windows equal day
    windows, as in training), forward-filled causally, scored, and passed
    through `add_causal_features` and `add_weather_features`. Features of a
    day therefore only use that day and earlier ones.
    """
    raw = pd.DataFrame([
        {
            "_pos": i,
            "_city": r.city,
            "Date": pd.Timestamp(r.date),
            "PM2.5": r.PM2_5, "PM10": r.PM10, "NO2": r.NO2,
            "SO2": r.SO2, "CO": r.CO, "O3": r.O3,
            "Benzene": r.Benzene, "Toluene": r.Toluene, "Xylene": r.Xylene,
            "_lag1": r.HRS_lag1, "_std7": r.HRS_7day_std,
            **{w: getattr(r, w) for w in _WEATHER_INPUTS},
        }
        for i, r in enumerate(readings)
    ])
    numeric_cols = (cfg.CORE_POLLUTANTS + cfg.VOC_POLLUTANTS
                    + ["_lag1", "_std7"] + _WEATHER_INPUTS)
    raw[numeric_cols] = raw[numeric_cols].astype(float)

    dupes = raw[raw.duplicated(["_city", "Date"], keep=False)]
    if not dupes.empty:
        first = dupes.iloc[0]
        raise HTTPException(
            422,
            detail=f"Duplicate reading for city '{first['_city']}' on "
                   f"{first['Date'].date()}",
        )

    # Split each city into segments separated by gaps longer than any feature
    # window, then reindex each segment to a contiguous daily calendar.
    raw = raw.sort_values(["_city", "Date"])
    gap = raw.groupby("_city")["Date"].diff().dt.days
    raw["_segment"] = (gap.isna() | (gap > _MAX_LOOKBACK_DAYS)).cumsum()

    frames = []
    for seg, g in raw.groupby("_segment", sort=False):
        days = pd.date_range(g["Date"].min(), g["Date"].max(), freq="D")
        g = g.set_index("Date").reindex(days).rename_axis("Date").reset_index()
        g["City"] = f"segment-{seg}"
        frames.append(g)
    df = pd.concat(frames, ignore_index=True)

    # Same cleaning and scoring path as pipeline.data.build_clean_frame
    df = data_mod.causal_forward_fill(df)
    df = data_mod.add_weight_coverage(df)
    df["hrs_valid"] = df["weight_coverage"] >= cfg.MIN_ROW_WEIGHT_COVERAGE
    df = feat.apply_composite_score(df, _ranker)
    df = feat.add_causal_features(df)
    df = feat.add_weather_features(df)
    df["history_days"] = df.groupby("City").cumcount().clip(upper=_MAX_LOOKBACK_DAYS)

    out = df[df["_pos"].notna()].sort_values("_pos").reset_index(drop=True)

    # Caller-supplied history features fill in only what the request could not
    if "hrs_lag_1" in out:
        out["hrs_lag_1"] = out["hrs_lag_1"].fillna(out["_lag1"])
    if "hrs_std_7" in out:
        out["hrs_std_7"] = out["hrs_std_7"].fillna(out["_std7"])
    return out


def _predict(readings: List[PollutantReading]) -> List[PredictionResponse]:
    _require_model()
    feats = _build_features(readings)
    X     = feats[_features]
    preds = _model.predict(X)
    proba = _model.predict_proba(X)

    # Exposure forecast = this week's mean + predicted change; needs a full
    # 7-day mean, so it is null for readings without that history.
    forecast = pd.Series(np.nan, index=feats.index)
    if _regressor is not None:
        change = _regressor.predict(feats[_reg_features])
        forecast = (feats["hrs_mean_7"] + change).clip(0, 1)

    results = []
    for i, r in enumerate(readings):
        hrs = feats.at[i, "hrs"]
        results.append(PredictionResponse(
            city=r.city,
            date=r.date,
            risk_level=str(preds[i]),
            probabilities={
                str(cls): round(float(p), 4)
                for cls, p in zip(_model.classes_, proba[i])
            },
            daily_hrs=round(float(hrs), 6) if pd.notna(hrs) else None,
            history_days=int(feats.at[i, "history_days"]),
            forecast_exposure=(round(float(forecast[i]), 4)
                               if pd.notna(forecast[i]) else None),
        ))
    return results


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Meta"])
def root():
    """Service ping — always returns 200 even when the model is not loaded."""
    return {
        "service": "Air Quality Health Risk API",
        "version": API_VERSION,
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
    _require_model()
    return {
        "status":          "ok",
        "model":           "RandomForestClassifier",
        "features":        _features,
        "n_features":      len(_features),
        "classes":         _classes,
        "risk_thresholds": {"low": _low_th, "high": _high_th},
        "test_metrics":    _test_metrics,
        "exposure_forecast": {
            "available":    _regressor is not None,
            "model":        "RandomForestRegressor" if _regressor is not None else None,
            "n_features":   len(_reg_features) if _regressor is not None else 0,
            "test_metrics": _reg_metrics if _regressor is not None else {},
        },
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
    The model's strongest features are 3/7/14/30-day rolling means of the
    exposure score, which **cannot** be computed from a single reading. They
    are filled with training-set medians, so a lone reading mostly reflects
    the season and the supplied `HRS_lag1` / `HRS_7day_std`.

    For real forecasts send the preceding 30 days of the city to
    `/predict/batch`; the response's `history_days` shows how much history
    each prediction actually had. `forecast_exposure` needs at least 7 days of
    history, so it is always null here.
    """
    return _predict([reading])[0]


@app.post(
    "/predict/batch",
    response_model=BatchResponse,
    tags=["Prediction"],
    summary="Batch predictions (up to 500 readings)",
)
def predict_batch(payload: BatchRequest):
    """
    Predict health risk levels for a **list** of city-day readings in one call.

    Readings are grouped by `city` and ordered by `date`. Each reading's
    rolling, lag and volatility features are computed from that reading and
    the **earlier** readings of the same city in the request — never later
    ones — exactly as during training. Readings may mix cities and need not be
    consecutive; missing days are treated as unmeasured.

    Results are returned in request order. Maximum batch size: 500.
    """
    return BatchResponse(results=_predict(payload.readings))
