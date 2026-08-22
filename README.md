# Exposure-Aware Air Quality Health Risk Forecasting

Forecasting next-week air-pollution health risk (**Low / Moderate / High**) for
18 Indian cities from multi-pollutant daily measurements, 2015–2020.

This is a corrected rebuild of an earlier version of this project. The earlier
version reported **89% accuracy**; that number was an artifact of building the
label out of the same rolling window the features were computed from. Rebuilt as
a genuine forecasting task, the honest number is **75.6% accuracy / 0.756 macro
F1** — and only **+0.019 macro F1 above a trivial persistence rule**. Full
defect-by-defect accounting in [FIXES.md](FIXES.md).

## The task

At day *t*, using only data from day *t* and earlier, predict the risk level of
mean pollution exposure over days **t+1 … t+7**.

Feature windows end at *t*. The label window starts at *t+1*. They never
overlap — this is the correction that matters most, because the original label
(a 7-day mean) was being predicted from its own 3-day and 1-day components.

## Results

Test period: 2019-10-01 → 2020-07-01 (4,482 days-city observations, never seen
during training or tuning).

| | accuracy | macro F1 |
|---|---|---|
| Majority class | 0.272 | 0.142 |
| **Persistence** ("next week = this week") | **0.737** | **0.737** |
| Logistic regression | 0.748 | 0.750 |
| **Random forest** (tuned) | **0.751** | **0.756** |

**The honest headline: the tuned model beats persistence by 1.9 macro-F1
points.** Pollution exposure is strongly autocorrelated, so most of what looks
like predictive skill is just inertia. Any evaluation of this problem that omits
the persistence baseline will overstate the model's contribution — which is
exactly what happened in the original.

Random forest, per class:

| class | precision | recall | F1 | n |
|---|---|---|---|---|
| Low | 0.829 | 0.772 | 0.800 | 1,744 |
| Moderate | 0.625 | 0.702 | 0.661 | 1,521 |
| High | 0.828 | 0.785 | 0.806 | 1,217 |

Confusion matrix (rows actual, columns predicted):

| | Low | Moderate | High |
|---|---|---|---|
| **Low** | 1346 | 386 | 12 |
| **Moderate** | 268 | 1067 | 186 |
| **High** | 9 | 253 | 955 |

Errors are almost entirely between adjacent classes: only 12 Low days are called
High and 9 High days are called Low, out of 4,482. Moderate is the weak class
(F1 0.661) because it is bounded on both sides — the same structural pattern as
the "CANDIDATE" class in my [Kepler KOI
project](https://github.com/ayazzahmad/kepler-koi-classification), and for the
same reason: a middle band defined by thresholds is intrinsically fuzzier than
the extremes.

Tuning selected `max_depth=4, min_samples_leaf=5, n_estimators=600` (CV macro F1
0.775) against a train macro F1 of 0.800 and test 0.756 — a modest, visible
generalisation gap. The original used `max_depth=50` with no search and never
reported a training score, so it had no way to see overfitting at all.

### Distribution shift is real and reported

India's COVID-19 lockdown (2020-03-25) falls inside the test window and changed
air quality sharply, so metrics are reported on both sides of it:

| segment | n | persistence macro F1 | RF macro F1 |
|---|---|---|---|
| pre-lockdown | 3,018 | 0.695 | 0.710 |
| lockdown | 1,464 | 0.634 | 0.657 |

Both degrade under the regime change, as expected. The model's margin over
persistence widens slightly during lockdown (+0.023 vs +0.015) — when yesterday
stops being a good guide to tomorrow, the extra features earn a little more of
their keep.

### What actually drives predictions

Permutation importance on held-out test data (drop in macro F1 when shuffled):

| feature | importance |
|---|---|
| `hrs_mean_3` | +0.0198 |
| `hrs_mean_7` | +0.0112 |
| `hrs_mean_14` | +0.0103 |
| `hrs_mean_30` | +0.0055 |
| `hrs_lag_1` | +0.0026 |
| `month_cos` | +0.0016 |
| `hrs_std_7` | +0.0001 |
| `dow_sin` / `dow_cos` | ~0 |

Two corrections to claims made in the original write-up:

- **Volatility carries essentially no signal** (`hrs_std_7` +0.0001,
  `hrs_std_14` +0.0003). The original claimed "volatility provides secondary but
  meaningful signal" — that claim does not survive removing the leakage. It also
  could not have been assessed properly there, because that feature was computed
  on a series which had been accidentally overwritten by its own rolling mean.
- **No day-of-week effect.** Both cyclical day-of-week terms sit at zero.

All importances are small in absolute terms, which is consistent with the
narrow margin over persistence: the rolling means carry largely redundant
information, so removing any single one costs little.

## Method

**Composite exposure score.** Weighted sum of six pollutants (PM2.5 0.35, PM10
0.20, NO₂ 0.15, SO₂/CO/O₃ 0.10 each), applied to **percentile-ranked** values.
Ranking rather than min-max scaling matters here: CO reaches 175 against a 75th
percentile of 1.5, and PM10 reaches 1000 against 130. Under min-max, ordinary
days compress into a sliver near zero and a pollutant's nominal weight stops
matching its real contribution — the weighting scheme becomes decorative. The
ranker is fitted on training rows only.

Weights are renormalised over the pollutants actually **observed** on each row,
so no pollutant value is ever invented.

**Features** (all ending at day *t*, all requiring complete windows):
rolling means over 3/7/14/30 days; rolling standard deviations over 7/14 days
computed on the *raw* daily score; lags at 1 and 7 days; a 7-vs-14-day trend
term; cyclical month and day-of-week encodings; a winter indicator; VOC index
with an explicit observed/not-observed flag; and per-row pollutant coverage as a
data-quality signal.

**Split.** Global date cutoff at 2019-10-01, with a 7-day embargo before it so
no training label is built from a test-period day. Every retained city has
history on both sides of the cutoff. Hyperparameters tuned with forward-chaining
`TimeSeriesSplit` on macro F1.

**Data quality.** Negative concentrations masked; causal forward-fill within
city (limit 3 days, uses only past values); cities with under 365 days of
history dropped; rows with under 60% of pollutant weight observed flagged
unscoreable but **retained**, so each city's daily contiguity survives and
incomplete rolling windows stay incomplete.

Two cities are excluded for coverage: **Mumbai** (39% of rows scoreable — around
60% missing across most pollutants) and **Ahmedabad** (70%). Six more are
excluded for insufficient history (Aizawl, Bhopal, Chandigarh, Ernakulam, Kochi,
Shillong). 18 cities and 24,173 rows remain; 19,955 have complete feature
history and a label.

This dataset has severe, uneven missingness that the original glossed over with
city-median imputation — Lucknow's PM10 and Ernakulam's O₃ are **100% missing**,
Patna's PM10 is 90%, Chennai's 85%. Median-filling a column that is 85–100%
absent produces a constant masquerading as a measurement.

## Limitations

Methodological limits that remain, stated plainly:

- **The margin over persistence is small** (+0.019 macro F1). This model is a
  modest improvement on inertia, not a strong forecaster.
- **Labels are relative, not clinical.** Risk levels are training-set exposure
  terciles, not medical or regulatory thresholds. There is no health-outcome
  data anywhere in this dataset, so "health risk" is a proxy throughout.
- **Pollutant weights are hand-assigned**, not learned or validated against
  health outcomes. A learned weighting would need outcome data this dataset
  lacks.
- **No meteorology.** Wind, temperature, humidity and rainfall drive pollutant
  dispersion and are absent.
- **One pooled model across 18 cities**, which differ in baseline pollution,
  measurement quality and seasonality. Per-city or hierarchical models are not
  explored, and the pooled model is likely dominated by the long-history cities.
- **The test window contains a structural break** (COVID-19), so headline test
  metrics blend ordinary forecasting with regime change. Both are reported, but
  a longer evaluation period would be better.
- **Only two model families** (regularised linear, tree ensemble). Gradient
  boosting and proper sequence models are untried.

## Running it

```bash
pip install -r requirements.txt
python run.py                  # full pipeline
python verify_no_leakage.py    # checks the anti-leakage invariants
```

`run.py` goes top to bottom from a clean interpreter in a few minutes and writes
to `results/`: `metrics.json`, `run_log.txt`, `model.joblib`,
`permutation_importance.csv`, and three plots. Verified deterministic — deleting
`results/` and rerunning reproduces every metric to four decimal places.

```
pipeline/
  config.py           all tunable constants and split dates
  data.py             loading, cleaning, data-quality gates
  features.py         composite score, causal features, forward label
  models.py           baselines, models, tuning, evaluation
run.py                end-to-end entry point
verify_no_leakage.py  executable checks on the core invariants
data/city_day.csv     source data (~2.5 MB), so the project runs as cloned
notebooks/            the original notebook, kept unchanged for provenance
results/              generated outputs (model.joblib is gitignored)
```

`notebooks/air_quality_health_risk_analysis.ipynb` is the original, flawed
analysis, left in place deliberately so the claims in [FIXES.md](FIXES.md) can be
checked against the code they describe. It does not run top to bottom — one of
the defects listed there. Nothing in `pipeline/` imports it.

### Verified invariants

`verify_no_leakage.py` checks the properties this rebuild depends on, against
independent recomputation rather than assertion — the original README claimed
"strict temporal ordering to prevent leakage" while its split was alphabetical
by city, so these are worth actually testing:

```
[PASS] label at t == mean(hrs[t+1..t+7])          2,002 rows, max diff 3.3e-16
[PASS] features at t unchanged when the future is randomised
                                                  11,573 rows x 9 features fixed
[PASS] feature window ends at t, label window starts at t+1
[PASS] embargo holds: last train label 2019-09-30 < test start 2019-10-01
[PASS] train-only thresholds differ from full-data thresholds
                        train (0.4032, 0.5935) vs full (0.3941, 0.5805)
[PASS] split is temporal, not by city             18/18 cities in both
```

The second check is the strongest evidence: every pollution value after
2018-06-30 is replaced with noise, and every feature at or before that date
stays bit-identical. Nothing looks forward.

The fifth quantifies the original's threshold leak — fitting cut points on the
full dataset shifts them by roughly 0.01, because the test period's
distribution was being folded into the label definition.

## Why this rebuild exists

The original notebook reported 89% accuracy and read as a solid result. It was
not, for four compounding reasons: the label was built from the same window as
the features; the "chronological" split was actually alphabetical by city; the
label thresholds, scaler and imputer were all fitted on the full dataset; and
there was no baseline to compare against.

Fixing those drops the number from 89% to 75.6%, and reveals that a one-line
persistence rule already achieves 73.7%. That is a much less impressive result
and a much more useful one — it says what this data can actually support.

The full list of defects and fixes, cell by cell, is in [FIXES.md](FIXES.md).
