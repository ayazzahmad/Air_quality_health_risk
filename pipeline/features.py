"""Composite exposure score, causal features, and the forecasting label.

This module carries the central correction. In the original pipeline the label
was a 7-day rolling mean of the exposure score while the features were 3-day
and 1-day statistics of that same score -- so the "prediction" task was mostly
reconstructing a smoothed series from its own components, and the reported 89%
accuracy measured that reconstruction rather than any forecasting ability.

Here:
  * features use only days up to and including t,
  * the label covers days t+1 .. t+LABEL_HORIZON_DAYS,
  * the two windows never overlap.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

from . import config as cfg


# --- Pollutant normalisation -------------------------------------------------

def fit_pollutant_ranker(train_df: pd.DataFrame) -> QuantileTransformer:
    """Fit a percentile-rank transform on TRAIN pollutant values only.

    Why not MinMaxScaler (the original choice): these distributions have
    extreme tails -- CO runs to 175 against a 75th percentile of 1.5, PM10 to
    1000 against 130. Min-max maps almost every ordinary day into a sliver near
    zero, so a pollutant's *nominal* weight stops matching its actual
    contribution to the composite and the weighting scheme becomes decorative.
    Percentile ranking is outlier-robust, stays bounded in [0, 1], and keeps
    each pollutant's contribution proportional to its assigned weight.
    """
    ranker = QuantileTransformer(
        output_distribution="uniform",
        n_quantiles=min(1000, len(train_df)),
        random_state=cfg.RANDOM_STATE,
    )
    ranker.fit(train_df[cfg.CORE_POLLUTANTS])
    return ranker


def apply_composite_score(df: pd.DataFrame, ranker: QuantileTransformer) -> pd.DataFrame:
    """Weighted composite of percentile-ranked pollutants.

    Weights are renormalised over the pollutants actually observed on each row,
    so a partially-measured day is scored on the same [0, 1] scale as a fully
    measured one instead of being silently penalised for missing inputs. No
    pollutant value is ever invented.
    """
    df = df.copy()
    ranked = pd.DataFrame(
        ranker.transform(df[cfg.CORE_POLLUTANTS]),
        columns=cfg.CORE_POLLUTANTS,
        index=df.index,
    )

    weighted_sum = np.zeros(len(df))
    weight_present = np.zeros(len(df))
    for p in cfg.CORE_POLLUTANTS:
        w = cfg.POLLUTANT_WEIGHTS[p]
        observed = ranked[p].notna().to_numpy()
        weighted_sum += np.where(observed, ranked[p].fillna(0.0).to_numpy() * w, 0.0)
        weight_present += observed * w

    with np.errstate(invalid="ignore", divide="ignore"):
        hrs = np.where(weight_present > 0, weighted_sum / weight_present, np.nan)

    # Rows that failed the coverage gate get no score, which keeps incomplete
    # rolling windows incomplete rather than quietly filling them.
    df["hrs"] = np.where(df["hrs_valid"].to_numpy(), hrs, np.nan)
    return df


# --- Causal (backward-only) features ----------------------------------------

def add_causal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features computed strictly from days <= t, within each city.

    `min_periods` equals the window length everywhere, so a feature only exists
    when the full window of genuine observations exists. The original used
    `min_periods=1`, which meant a row's "7-day mean" could be a single day.

    Note on the train/test boundary: a test-period row legitimately uses its own
    earlier days, some of which fall in the training period. That is not
    leakage -- it is exactly what a deployed model would have access to. What
    must not happen (and does not, here) is a *training* label being built from
    test-period days; that is what the embargo in split_frame handles.
    """
    df = df.sort_values(["City", "Date"]).copy()
    g = df.groupby("City", sort=False)["hrs"]

    for w in cfg.ROLLING_WINDOWS:
        df[f"hrs_mean_{w}"] = g.transform(
            lambda s, w=w: s.rolling(w, min_periods=w).mean()
        )

    # Volatility is computed on the RAW daily score. The original computed it on
    # a series that had already been overwritten by its own 7-day rolling mean,
    # so the "volatility" feature was measuring the variability of smoothed
    # data -- close to the opposite of what the name implies.
    for w in cfg.VOLATILITY_WINDOWS:
        df[f"hrs_std_{w}"] = g.transform(
            lambda s, w=w: s.rolling(w, min_periods=w).std()
        )

    for lag in cfg.LAGS:
        df[f"hrs_lag_{lag}"] = g.transform(lambda s, lag=lag: s.shift(lag))

    # Direction of travel: is the recent week worse than the fortnight?
    df["hrs_trend"] = df["hrs_mean_7"] - df["hrs_mean_14"]

    # Calendar features as cyclical pairs. As raw integers (the original's
    # approach) December=12 and January=1 sit at opposite ends of the range
    # despite being adjacent seasons, which is actively misleading for a linear
    # model.
    month = df["Date"].dt.month
    dow = df["Date"].dt.day_of_week
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["is_winter"] = month.isin([11, 12, 1, 2]).astype(int)

    # VOCs are too sparse to impute quietly (Xylene is ~61% missing), so the
    # model gets an explicit observed/not-observed flag alongside the value.
    voc = df[cfg.VOC_POLLUTANTS].sum(axis=1, min_count=1)
    df["voc_observed"] = voc.notna().astype(int)
    df["voc_index"] = voc

    return df


POLLUTION_FEATURES = (
    [f"hrs_mean_{w}" for w in cfg.ROLLING_WINDOWS]
    + [f"hrs_std_{w}" for w in cfg.VOLATILITY_WINDOWS]
    + [f"hrs_lag_{lag}" for lag in cfg.LAGS]
    + ["hrs_trend", "month_sin", "month_cos", "dow_sin", "dow_cos", "is_winter",
       "voc_index", "voc_observed", "weight_coverage"]
)


# --- Weather features --------------------------------------------------------

def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Weather summaries computed strictly from days <= t, within each city.

    Same availability rule as the pollution features: day t's weather is
    observed by the end of day t, and nothing later is used. Windows require
    complete data (`min_periods == window`).
    """
    df = df.sort_values(["City", "Date"]).copy()
    g = df.groupby("City", sort=False)

    for var in cfg.WEATHER_FEATURE_VARS:
        for w in cfg.WEATHER_WINDOWS:
            df[f"{var}_{w}d"] = g[var].transform(
                lambda s, w=w: s.rolling(w, min_periods=w).mean()
            )

    # Wind direction is circular (359 deg is next to 0 deg), so it is averaged
    # as a unit vector rather than as a raw angle.
    rad = np.deg2rad(df["wind_dir"])
    df["_wind_dir_sin"], df["_wind_dir_cos"] = np.sin(rad), np.cos(rad)
    for comp in ("sin", "cos"):
        df[f"wind_dir_{comp}_3"] = g[f"_wind_dir_{comp}"].transform(
            lambda s: s.rolling(3, min_periods=3).mean()
        )
    df = df.drop(columns=["_wind_dir_sin", "_wind_dir_cos"])

    # Season turning: is the last week colder or warmer than the last month?
    temp_30 = g["temp_mean"].transform(lambda s: s.rolling(30, min_periods=30).mean())
    df["temp_trend"] = df["temp_mean_7d"] - temp_30

    # Stagnant, pollution-trapping days in the past week
    calm_dry = ((df["wind_mean"] < cfg.CALM_WIND_KMH)
                & (df["precip_sum"] < cfg.DRY_PRECIP_MM)).astype(float)
    calm_dry = calm_dry.where(df[["wind_mean", "precip_sum"]].notna().all(axis=1))
    df["calm_dry_days_7"] = calm_dry.groupby(df["City"], sort=False).transform(
        lambda s: s.rolling(7, min_periods=7).sum()
    )
    return df


WEATHER_FEATURES = (
    [f"{v}_{w}d" for v in cfg.WEATHER_FEATURE_VARS for w in cfg.WEATHER_WINDOWS]
    + ["wind_dir_sin_3", "wind_dir_cos_3", "temp_trend", "calm_dry_days_7"]
)


def add_oracle_weather(df: pd.DataFrame) -> pd.DataFrame:
    """OBSERVED weather over the label window t+1 .. t+H.

    This is NOT available at forecast time. It stands in for a perfect weather
    forecast, to measure how much a real forecast feed could add at most. It
    must never be part of the production feature set; verify_no_leakage.py
    checks that these columns do move when the future changes.
    """
    h = cfg.LABEL_HORIZON_DAYS
    df = df.sort_values(["City", "Date"]).copy()
    g = df.groupby("City", sort=False)
    for var in cfg.WEATHER_FEATURE_VARS:
        df[f"{var}_next_{h}"] = g[var].transform(
            lambda s: s.rolling(h, min_periods=h).mean().shift(-h)
        )
    return df


ORACLE_WEATHER_FEATURES = [
    f"{v}_next_{cfg.LABEL_HORIZON_DAYS}" for v in cfg.WEATHER_FEATURE_VARS
]

# The production feature set
FEATURE_COLUMNS = POLLUTION_FEATURES


# --- Forward-looking label ---------------------------------------------------

def add_forward_target(df: pd.DataFrame) -> pd.DataFrame:
    """Mean exposure over days t+1 .. t+H, per city.

    `rolling(H).mean()` at index i is the mean of [i-H+1 .. i]; shifting by -H
    moves that to index i-H, giving the mean of [i+1 .. i+H] at row i. Requiring
    a complete window means the last H days of each city have no label and drop
    out, which is correct -- their future is not in the dataset.
    """
    h = cfg.LABEL_HORIZON_DAYS
    df = df.sort_values(["City", "Date"]).copy()
    df["target_exposure"] = df.groupby("City", sort=False)["hrs"].transform(
        lambda s: s.rolling(h, min_periods=h).mean().shift(-h)
    )
    return df


def fit_risk_thresholds(train_target: pd.Series) -> tuple[float, float]:
    """Risk cut points from TRAIN exposure only.

    The original computed these quantiles over the entire dataset, so the label
    definition itself encoded the test distribution.
    """
    lo, hi = train_target.quantile(list(cfg.RISK_QUANTILES))
    return float(lo), float(hi)


def apply_risk_labels(target: pd.Series, lo: float, hi: float) -> pd.Series:
    """Bin continuous future exposure into ordered risk levels."""
    return pd.Series(
        np.select(
            [target <= lo, target <= hi],
            [cfg.RISK_LABELS[0], cfg.RISK_LABELS[1]],
            default=cfg.RISK_LABELS[2],
        ),
        index=target.index,
    ).where(target.notna())
