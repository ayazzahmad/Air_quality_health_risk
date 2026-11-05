"""
train.py – Train the tuned RandomForest model using the production pipeline
and save a self-contained artifact to artifacts/model.joblib.

Usage
-----
    python train.py                                    # defaults
    python train.py --data data/city_day.csv \\
                    --out  artifacts/model.joblib \\
                    --fast                             # skip grid-search, use best params
"""
import argparse
import pathlib

import joblib
import numpy as np
import pandas as pd

from pipeline import config as cfg
from pipeline.data import build_clean_frame
from pipeline.features import (
    EXPOSURE_FEATURES,
    FEATURE_COLUMNS,
    add_causal_features,
    add_forward_target,
    add_weather_features,
    apply_composite_score,
    apply_risk_labels,
    fit_pollutant_ranker,
    fit_risk_thresholds,
)
from pipeline.models import (
    build_exposure_regressor,
    exposure_report,
    full_report,
    tune_exposure_regressor,
    tune_random_forest,
)


def main(data_path: str, out_path: str, fast: bool = False, weather_path=None) -> None:
    # ── 1. Load & clean (causal imputation, quality gates) ───────────────────
    print(f"[train] Loading {data_path}")
    df, audit = build_clean_frame(data_path, weather_path)
    print(f"[train] Cities retained : {len(audit['cities_retained'])}")
    print(f"[train] Rows retained   : {audit['rows_retained']}")
    print(f"[train] Weather coverage: {audit['weather_coverage']:.1%}")

    # ── 2. Temporal split ─────────────────────────────────────────────────────
    test_start  = pd.Timestamp(cfg.TEST_START)
    train_cut   = test_start - pd.Timedelta(days=cfg.EMBARGO_DAYS)

    # ── 3. Fit pollutant ranker (QuantileTransformer) on TRAIN only ───────────
    ranker = fit_pollutant_ranker(df[df["Date"] < train_cut])

    # ── 4. Build composite score + causal features + forward label ────────────
    df = apply_composite_score(df, ranker)
    df = add_causal_features(df)
    df = add_weather_features(df)
    df = add_forward_target(df)

    # ── 5. Split ──────────────────────────────────────────────────────────────
    # Drop rows that have any required feature or the label missing
    required = [c for c in FEATURE_COLUMNS if c != "voc_index"]
    ok       = df.dropna(subset=required + ["target_exposure"])
    train    = ok[ok["Date"] <  train_cut]
    test     = ok[ok["Date"] >= test_start]
    # Date order, as in tune_random_forest: the forest's bootstrap samples
    # depend on row order, so this keeps --fast identical to the tuned refit.
    train    = train.iloc[np.argsort(train["Date"].to_numpy(), kind="stable")]
    print(f"[train] Train rows: {len(train)}  |  Test rows: {len(test)}")

    # ── 6. Risk thresholds from TRAIN only ────────────────────────────────────
    lo, hi = fit_risk_thresholds(train["target_exposure"])
    y_train = apply_risk_labels(train["target_exposure"], lo, hi)
    y_test  = apply_risk_labels(test["target_exposure"],  lo, hi)

    # ── 7. Train (grid-search or best known params) ───────────────────────────
    if fast:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        print("[train] Fast mode - using best known params (no grid search)")
        model = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=600, max_depth=4, min_samples_leaf=5,
                max_features="sqrt", class_weight="balanced",
                random_state=cfg.RANDOM_STATE, n_jobs=-1,
            )),
        ])
        model.fit(train[FEATURE_COLUMNS], y_train)
    else:
        print("[train] Running grid-search (this takes a few minutes) ...")
        search = tune_random_forest(
            train[FEATURE_COLUMNS], y_train, train["Date"]
        )
        model = search.best_estimator_
        print(f"[train] Best params : {search.best_params_}")
        print(f"[train] CV macro F1 : {search.best_score_:.4f}")

    # ── 8. Evaluate ───────────────────────────────────────────────────────────
    report = full_report(
        "RandomForest", model,
        train[FEATURE_COLUMNS], y_train,
        test[FEATURE_COLUMNS],  y_test,
    )
    print(f"[train] Test accuracy : {report['test']['accuracy']:.4f}")
    print(f"[train] Test macro F1 : {report['test']['macro_f1']:.4f}")

    # ── 9. Exposure forecast: next-week change from this week's mean ──────────
    change_train = train["target_exposure"] - train["hrs_mean_7"]
    if fast:
        regressor = build_exposure_regressor(n_jobs=-1)
        regressor.fit(train[EXPOSURE_FEATURES], change_train)
    else:
        print("[train] Tuning the exposure regressor ...")
        reg_search = tune_exposure_regressor(
            train[EXPOSURE_FEATURES], change_train, train["Date"]
        )
        regressor = reg_search.best_estimator_
        print(f"[train] Best params : {reg_search.best_params_}")
        print(f"[train] CV R2 change: {reg_search.best_score_:.4f}")
    reg_metrics = exposure_report(
        regressor, test[EXPOSURE_FEATURES], test["hrs_mean_7"], test["target_exposure"]
    )
    print(f"[train] Exposure MAE  : {reg_metrics['mae']:.4f} "
          f"(no-change baseline {reg_metrics['mae_persistence']:.4f})")
    print(f"[train] R2 of change  : {reg_metrics['r2_change']:.4f}")

    # ── 10. Save artifact ─────────────────────────────────────────────────────
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model":         model,
        "ranker":        ranker,
        "features":      FEATURE_COLUMNS,
        # The model's own class order, i.e. the column order of predict_proba
        "classes":       [str(c) for c in model.classes_],
        "low_th":        lo,
        "high_th":       hi,
        "test_metrics":  report["test"],
        # Next-week exposure forecast (predicts the change from hrs_mean_7)
        "regressor":            regressor,
        "regressor_features":   EXPOSURE_FEATURES,
        "regressor_metrics":    reg_metrics,
    }
    joblib.dump(artifact, out_path)
    print(f"[train] Saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the Air Quality Health Risk forecasting model."
    )
    parser.add_argument(
        "--data", default=str(cfg.DATA_PATH),
        help="Path to city_day.csv (default: data/city_day.csv)"
    )
    parser.add_argument(
        "--weather", default=str(cfg.WEATHER_PATH),
        help="Path to weather_daily.csv (default: data/weather_daily.csv)"
    )
    parser.add_argument(
        "--out", default="artifacts/model.joblib",
        help="Output path for the saved artifact"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip grid-search, use best-known hyperparameters"
    )
    args = parser.parse_args()
    main(args.data, args.out, args.fast, args.weather)
