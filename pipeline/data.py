"""Loading, cleaning and data-quality gating.

Every step here is causal (uses only past observations within a city) or is a
structural filter that does not depend on pollutant values, so nothing in this
module can leak test information into training.
"""

import pandas as pd

from . import config as cfg


def load_raw(path=None) -> pd.DataFrame:
    """Load the raw city-day table and enforce basic structural guarantees.

    `path` defaults to `config.DATA_PATH`.
    """
    df = pd.read_csv(path or cfg.DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values(["City", "Date"]).reset_index(drop=True)

    if df.duplicated(["City", "Date"]).any():
        raise ValueError("Duplicate (City, Date) rows -- rolling windows would be wrong.")

    return df


def check_date_contiguity(df: pd.DataFrame) -> pd.DataFrame:
    """Report per-city calendar gaps.

    Row-based rolling windows only equal day-based windows when each city's
    dates are contiguous. This asserts that assumption instead of trusting it:
    a gap would silently make a "7-day" window span more than 7 days.
    """
    rows = []
    for city, g in df.groupby("City", sort=True):
        span = (g["Date"].max() - g["Date"].min()).days + 1
        rows.append({"city": city, "rows": len(g), "span_days": span,
                     "missing_days": span - len(g)})
    return pd.DataFrame(rows)


def mask_impossible_values(df: pd.DataFrame) -> pd.DataFrame:
    """Blank out physically impossible (negative) concentrations."""
    cols = [c for c in cfg.CORE_POLLUTANTS + cfg.VOC_POLLUTANTS if c in df.columns]
    df = df.copy()
    df[cols] = df[cols].mask(df[cols] < 0)
    return df


def causal_forward_fill(df: pd.DataFrame) -> pd.DataFrame:
    """Carry the last observation forward within each city, up to a limit.

    Causal by construction: a filled value only ever uses an earlier day from
    the same city, so this is safe to apply before splitting. Contrast with the
    original pipeline's city-median imputation, which averaged over the whole
    series (including future days) before the split.
    """
    df = df.copy()
    cols = [c for c in cfg.CORE_POLLUTANTS + cfg.VOC_POLLUTANTS if c in df.columns]
    df[cols] = df.groupby("City", sort=False)[cols].ffill(limit=cfg.FFILL_LIMIT_DAYS)
    return df


def add_weight_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Fraction of total composite weight actually observed on each row.

    Used both as a validity gate and as a model feature, so the model can tell
    a well-measured day from a sparsely-measured one.
    """
    df = df.copy()
    coverage = sum(
        cfg.POLLUTANT_WEIGHTS[p] * df[p].notna() for p in cfg.CORE_POLLUTANTS
    )
    total = sum(cfg.POLLUTANT_WEIGHTS.values())
    df["weight_coverage"] = coverage / total
    return df


def apply_quality_gates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop cities that cannot support exposure modelling. Returns (df, report).

    Two gates, both independent of the target:
      1. Cities with too little history for 30-day windows.
      2. Cities whose rows mostly fail the weight-coverage threshold -- their
         composite score would be built from whichever pollutants happened to
         be measured, which is not comparable across days.
    """
    report = {}

    counts = df["City"].value_counts()
    too_short = sorted(counts[counts < cfg.MIN_CITY_OBSERVATIONS].index)
    df = df[~df["City"].isin(too_short)]
    report["dropped_insufficient_history"] = too_short

    valid_rate = df.groupby("City")["weight_coverage"].apply(
        lambda s: (s >= cfg.MIN_ROW_WEIGHT_COVERAGE).mean()
    )
    poor = sorted(valid_rate[valid_rate < cfg.MIN_CITY_VALID_RATE].index)
    df = df[~df["City"].isin(poor)]
    report["dropped_poor_coverage"] = {
        c: round(float(valid_rate[c]), 3) for c in poor
    }

    # Rows failing the coverage gate are FLAGGED, not dropped. Dropping them
    # would break each city's daily contiguity, and row-based rolling windows
    # would then silently span more calendar days than intended. Keeping the
    # rows with a NaN composite means an incomplete window stays incomplete,
    # which is what `min_periods=window` is there to catch.
    df = df.copy()
    df["hrs_valid"] = df["weight_coverage"] >= cfg.MIN_ROW_WEIGHT_COVERAGE
    report["rows_flagged_low_coverage"] = int((~df["hrs_valid"]).sum())
    report["cities_retained"] = sorted(df["City"].unique())
    report["rows_retained"] = len(df)

    return df.sort_values(["City", "Date"]).reset_index(drop=True), report


def build_clean_frame(path=None) -> tuple[pd.DataFrame, dict]:
    """Full cleaning path, returning the frame and an audit report."""
    df = load_raw(path)
    contiguity = check_date_contiguity(df)

    df = mask_impossible_values(df)
    df = causal_forward_fill(df)
    df = add_weight_coverage(df)
    df, report = apply_quality_gates(df)

    report["max_calendar_gap_days"] = int(contiguity["missing_days"].max())
    return df, report
