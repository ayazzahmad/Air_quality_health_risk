"""Configuration for the exposure-aware air quality risk pipeline.

Every value that shapes a result lives here so it can be found and challenged in
one place, rather than being buried as a literal inside a notebook cell.
"""

from pathlib import Path

# --- Paths (relative, so the project runs on any machine) ---------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "city_day.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

RANDOM_STATE = 42

# --- Pollutants and composite weights ----------------------------------------
# Weights are domain-informed (PM2.5 weighted highest for established health
# impact), NOT learned. They are applied to *percentile-ranked* pollutant values,
# so each pollutant's contribution is proportional to its weight -- see
# features.fit_pollutant_ranker for why raw min-max scaling breaks this.
CORE_POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
POLLUTANT_WEIGHTS = {
    "PM2.5": 0.35,
    "PM10": 0.20,
    "NO2": 0.15,
    "SO2": 0.10,
    "CO": 0.10,
    "O3": 0.10,
}
VOC_POLLUTANTS = ["Benzene", "Toluene", "Xylene"]

# --- Data quality gates ------------------------------------------------------
# Forward-fill is causal (uses only past observations), so it cannot leak.
FFILL_LIMIT_DAYS = 3

# A row's composite score is only trusted if at least this fraction of total
# pollutant weight was actually observed. Below it, the composite would be
# dominated by whichever pollutants happened to be measured.
MIN_ROW_WEIGHT_COVERAGE = 0.60

# Cities need enough history for 30-day rolling windows to be meaningful.
MIN_CITY_OBSERVATIONS = 365

# Cities whose rows mostly fail the coverage gate cannot support rolling
# exposure features at all. Mumbai (39% coverage) is excluded by this.
MIN_CITY_VALID_RATE = 0.70

# --- Forecasting task definition ---------------------------------------------
# THE central fix. The label is the mean exposure over the NEXT
# LABEL_HORIZON_DAYS, and every feature is computed from data up to and
# including day t. Feature and label windows do not overlap, so the task is
# genuine forecasting rather than reconstructing a smoothed series from its own
# components.
LABEL_HORIZON_DAYS = 7

# Backward-looking windows for features (all end at day t).
ROLLING_WINDOWS = [3, 7, 14, 30]
VOLATILITY_WINDOWS = [7, 14]
LAGS = [1, 7]

# --- Temporal split ----------------------------------------------------------
# Every retained city has data starting before this date, so no city appears
# only in test. Chosen to leave ~9 months of held-out future.
TEST_START = "2019-10-01"

# Training rows whose label window would reach past TEST_START are dropped, so
# no training label is built from data the model is later tested on.
EMBARGO_DAYS = LABEL_HORIZON_DAYS

# India's COVID-19 lockdown began 2020-03-25 and sharply changed air quality.
# It sits inside the test period, so results are reported with and without it.
COVID_START = "2020-03-25"

# Risk-level cut points. Applied to TRAIN quantiles only, then reused unchanged
# on test -- the original computed these on the full dataset, leaking the test
# distribution into the label definition.
RISK_QUANTILES = (0.33, 0.66)
RISK_LABELS = ["Low", "Moderate", "High"]
