"""
pipeline/preprocess.py
Reusable preprocessing helpers extracted from the notebook.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

POLLUTANTS      = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
IMPUTE_COLS     = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3"]
VOC_COLS        = ["Benzene", "Toluene", "Xylene"]
FEATURE_COLUMNS = [
    "month_sin", "month_cos",
    "dayofweek_sin", "dayofweek_cos",
    "is_winter",
    "HRS_lag1", "HRS_7day_std",
    "VOC_index",
]


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load city_day.csv, mask negatives, forward-fill per city."""
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values(["City", "Date"]).reset_index(drop=True)

    # Mask negative concentrations
    all_pollutant_cols = [
        "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
        "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"
    ]
    existing = [c for c in all_pollutant_cols if c in df.columns]
    df[existing] = df[existing].mask(df[existing] < 0)

    # City-level median imputation for core pollutants
    imp_existing = [c for c in IMPUTE_COLS if c in df.columns]
    df[imp_existing] = (
        df.groupby("City")[imp_existing]
        .transform(lambda x: x.fillna(x.median()))
    )

    # Keep only cities with >= 365 observations
    city_counts = df["City"].value_counts()
    valid_cities = city_counts[city_counts >= 365].index
    df = df[df["City"].isin(valid_cities)].reset_index(drop=True)

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"]      = df["Date"].dt.month
    df["dayofweek"]  = df["Date"].dt.dayofweek
    df["is_winter"]  = df["month"].isin([11, 12, 1, 2]).astype(int)
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


def build_exposure_features(df: pd.DataFrame, scaler: MinMaxScaler) -> pd.DataFrame:
    """
    Compute Daily_HRS, rolling windows, lag, VOC index.
    `scaler` must already be fitted on training pollutant data.
    """
    df = df.copy()
    normed = scaler.transform(df[POLLUTANTS])
    for i, p in enumerate(POLLUTANTS):
        df[p + "_norm"] = normed[:, i]

    df["Daily_HRS"] = (
        0.35 * df["PM2.5_norm"] +
        0.20 * df["PM10_norm"]  +
        0.15 * df["NO2_norm"]   +
        0.10 * df["SO2_norm"]   +
        0.10 * df["CO_norm"]    +
        0.10 * df["O3_norm"]
    )

    df = df.sort_values(["City", "Date"])
    df["HRS_3day"] = (
        df.groupby("City")["Daily_HRS"]
          .rolling(3, min_periods=1).mean()
          .reset_index(level=0, drop=True)
    )
    df["HRS_7day"] = (
        df.groupby("City")["Daily_HRS"]
          .rolling(7, min_periods=1).mean()
          .reset_index(level=0, drop=True)
    )
    df["HRS_lag1"] = df.groupby("City")["Daily_HRS"].shift(1)
    df["HRS_7day_std"] = (
        df.groupby("City")["Daily_HRS"]
          .transform(lambda x: x.rolling(7, min_periods=3).std())
    )

    voc_existing = [c for c in VOC_COLS if c in df.columns]
    df["VOC_index"] = df[voc_existing].sum(axis=1, min_count=1)

    return df


def apply_risk_labels(
    df: pd.DataFrame,
    low_th: float,
    high_th: float,
    col: str = "HRS_7day",
) -> pd.Series:
    def _label(x):
        if x <= low_th:
            return "Low"
        elif x <= high_th:
            return "Moderate"
        return "High"
    return df[col].apply(_label)

