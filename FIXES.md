# Defect → fix mapping

Every issue found in the original `notebooks/air_quality_health_risk_analysis.ipynb`,
with the cell it occurred in and where it is addressed. Cell numbers refer to
positional index in the original notebook.

## Tier 1 — invalidated the reported 89%

| # | Defect | Original | Fix |
|---|---|---|---|
| 1 | **Target reconstructed from its own components.** Label was quantile-binned `HRS_7day` = mean(HRS[t-6…t]); features were `HRS_3day` (subset of that window), `HRS_lag1` (inside it) and `HRS_7day_std` (same window). Importances confirmed it: `HRS_3day` 0.457 + `HRS_lag1` 0.302 = 76%. | cells 31–45 | Reframed as genuine forecasting. Label is mean exposure over **t+1 … t+7**; every feature ends at **t**. Zero window overlap. `features.add_forward_target`, `features.add_causal_features` |
| 2 | **Split was city-based, not temporal.** `train_test_split(shuffle=False)` on a frame sorted by `['City','Date']` takes the alphabetically-last cities entire, not the latest dates. No temporal validation at all, in a project premised on temporal exposure. | cells 8, 48 | Global date cutoff (`TEST_START`), so train is strictly earlier than test and every city appears in both. `run.main` step 2 |
| 3a | **Label thresholds fitted on the full dataset**, leaking the test distribution into the label definition. | cell 39 | `features.fit_risk_thresholds` is called on training rows only, then reused unchanged on test. |
| 3b | **`MinMaxScaler` fitted on the full dataset** before the split. | cell 28 | `features.fit_pollutant_ranker` fits on training rows only. |
| 3c | **`X.fillna(X.median())` computed on the full dataset** before the split. | cell 46 | `SimpleImputer` lives inside the sklearn `Pipeline`, so it is refitted per training fold. `models.build_logistic`, `models.build_random_forest` |
| 4 | **No baseline.** 89% was reported against nothing. | — | Two baselines reported alongside every model: majority-class, and **persistence** ("next week looks like this week"), which is the real bar for a forecaster. `models.majority_baseline`, `models.persistence_baseline` |
| 5 | **No embargo.** Even with a date split, a training row's label window would have reached into the test period. | — | Training rows within `EMBARGO_DAYS` of the cutoff are discarded, so no training label is built from a test-period day. `run.main` step 2 |

## Tier 2 — substantive method problems

| # | Defect | Original | Fix |
|---|---|---|---|
| 6 | **Min-max scaling destroyed the weighting scheme.** Weights (PM2.5 0.35, CO 0.10, …) were applied to min-max values, but CO ranges to 175 against a p75 of 1.5 and PM10 to 1000 against 130 — so ordinary days compress toward zero and a pollutant's nominal weight stops matching its real contribution. | cells 28–29 | Percentile-rank transform (`QuantileTransformer`, uniform) — outlier-robust, bounded [0,1], keeps contribution proportional to weight. `features.fit_pollutant_ranker` |
| 7 | **`Health_Risk_Score` overwritten by its own rolling mean** (accidental self-assignment), so `HRS_7day_std` measured the volatility of *smoothed* data — roughly the opposite of its name. Visible in cell 36: `Health_Risk_Score` count 26163 vs `HRS_3day` 26182. | cell 33 | Removed. Volatility is computed on the raw daily score. `features.add_causal_features` |
| 8 | **Logistic regression fitted on unscaled features** spanning `month` (1–12), HRS (~0.01–0.4) and `VOC_index` (another scale) — handicapping the model and making the 86%-vs-89% comparison unfair. | cell 49 | `StandardScaler` inside the pipeline. `models.build_logistic` |
| 9 | **No cross-validation or tuning.** `max_depth=50, n_estimators=200` chosen arbitrarily; effectively unbounded depth on 7 features; no training score reported, so overfitting was invisible. | cell 50 | `GridSearchCV` over 48 configs with forward-chaining `TimeSeriesSplit`, scored on macro F1. Train **and** test metrics reported for every model. `models.tune_random_forest`, `models.full_report` |
| 10 | **`month` / `dayofweek` as raw integers** — December (12) and January (1) sit at opposite ends of the range despite being adjacent seasons. | cells 16, 45 | Cyclical sin/cos encoding. `features.add_causal_features` |
| 11 | **`min_periods=1`** meant a row's "7-day mean" could be a single day; also inconsistent across the notebook (1, 2, 3) with no rationale. | cells 31, 33, 44 | `min_periods == window` everywhere; rows without complete history are dropped, not filled. |
| 12 | **`VOC_index` mostly imputed constant.** Built from Benzene/Toluene/Xylene where Xylene is ~61% missing, deliberately un-imputed, then globally median-filled anyway — carrying 0.066 importance on largely synthetic values. | cells 34, 46 | Kept with an explicit `voc_observed` indicator so the model can distinguish measured from imputed. |
| 13 | **All-NaN city×pollutant groups ignored.** Cell 11 emitted `RuntimeWarning: Mean of empty slice`; cell 12 confirmed PM10/NH3 still had NaN after imputation. Lucknow PM10 and Ernakulam O3 are **100% missing**; Chennai 85%, Patna 90%, Mumbai ~60% across most pollutants. Never investigated. | cell 11 | Explicit data-quality gates: per-row weight coverage, per-city valid rate, weights renormalised over *observed* pollutants only — no value is invented. Mumbai (39% coverage) is excluded and the exclusion is reported. `data.add_weight_coverage`, `data.apply_quality_gates` |
| 14 | **Row-based rolling assumed daily contiguity** without checking. | cells 31, 44 | `data.check_date_contiguity` asserts it (all cities are in fact gap-free). Low-coverage rows are **flagged, not dropped**, so contiguity survives and incomplete windows stay incomplete. |

## Tier 3 — interpretation

| # | Defect | Fix |
|---|---|---|
| 15 | README presented *"short-term accumulated exposure dominates predictions"* as a domain insight. It was a mechanical consequence of building the label from `HRS_3day`'s window. | Removed. Importance is now measured by **permutation on held-out test data** (`models.importance_report`) rather than training-set impurity, and interpreted against the persistence baseline. |
| 16 | *"Balanced class distribution (~33% each)"* stated as a property of the data — it is a tautology of binning at the 0.33/0.66 quantiles. | Stated as a design choice; train→test class drift is reported explicitly. |
| 17 | *"Maintained strict temporal ordering to prevent leakage"* — contradicted by defect #2. | Corrected; the actual split is documented with dates. |
| 18 | Limitations section listed only non-methodological caveats, omitting every issue above. | Rewritten to lead with the methodological limits that remain. |
| 19 | COVID-19 lockdown sits inside the evaluation window and sharply changed air quality — never acknowledged. | Test metrics reported split around `2020-03-25`. `run.main` step 7 |

## Tier 4 — reproducibility

| # | Defect | Fix |
|---|---|---|
| 20 | **A fresh top-to-bottom run crashed.** Cell 24 plotted `Health_Risk_Score`/`HRS_7day`, created at cells 29/31. Proof of out-of-order execution: cell 6 — in Section 1 — printed a column list already containing `Health_Risk_Level` and `HRS_7day_std`, created at cells 40/44. | Linear, importable modules with a single entry point (`python run.py`), verified from a clean interpreter. |
| 21 | **Hardcoded absolute path** `C:\Users\ayazp\Downloads\...` while the repo ships `data/city_day.csv`. | Relative paths resolved from the package root. `config.PROJECT_ROOT` |
| 22 | **`!pip install scikit-learn` inside the notebook**; no `requirements.txt`. | `requirements.txt` added; no installs in code. |
| 23 | **No saved model artifact** (`results/` held only PNGs). | Model, ranker and thresholds persisted to `results/model.joblib`; all metrics to `results/metrics.json`. |
| 24 | **Error analysis was `errors.head()`** — five printed rows, no aggregation. | Confusion matrix, per-class report, permutation importance, and the regime split. |
| 25 | **Notebook hygiene:** Section 4 (Feature Engineering) came *after* Section 3 (EDA) plotted those same engineered features; `pollutants` defined twice (18, 20); `city_counts` twice (13, 14); columns printed twice (5, 6); dead exploratory cells (12, 15, 36, 41, 42). | Removed; each step is a named function with a single responsibility. |
