"""
train.py – Train the tuned RandomForest model using the production pipeline
and save a self-contained artifact to artifacts/model.joblib.

Usage
-----
    python train.py                                    # defaults
    python train.py --data data/city_day.csv \\
                    --out  artifacts/model.joblib \\
                    --fast                             # skip grid-search, use best params
    python train.py --fast --mlflow                    # also log the run to MLflow
    python train.py --fast --mlflow --exposure-features pollution   # ablation run
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
    POLLUTION_FEATURES,
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
    persistence_baseline,
    tune_exposure_regressor,
    tune_random_forest,
)


EXPOSURE_FEATURE_SETS = {"weather": EXPOSURE_FEATURES, "pollution": POLLUTION_FEATURES}


def main(data_path: str, out_path: str, fast: bool = False, weather_path=None,
         exposure_set: str = "weather", track: bool = False, experiment=None) -> None:
    exposure_cols = EXPOSURE_FEATURE_SETS[exposure_set]
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
        clf_cv = None
    else:
        print("[train] Running grid-search (this takes a few minutes) ...")
        search = tune_random_forest(
            train[FEATURE_COLUMNS], y_train, train["Date"]
        )
        model = search.best_estimator_
        clf_cv = search.best_score_
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
    persistence, _ = persistence_baseline(test["hrs_mean_7"], lo, hi, y_test)

    # ── 9. Exposure forecast: next-week change from this week's mean ──────────
    change_train = train["target_exposure"] - train["hrs_mean_7"]
    print(f"[train] Exposure features: {exposure_set} ({len(exposure_cols)})")
    if fast:
        regressor = build_exposure_regressor(n_jobs=-1)
        regressor.fit(train[exposure_cols], change_train)
        reg_cv = None
    else:
        print("[train] Tuning the exposure regressor ...")
        reg_search = tune_exposure_regressor(
            train[exposure_cols], change_train, train["Date"]
        )
        regressor = reg_search.best_estimator_
        reg_cv = reg_search.best_score_
        print(f"[train] Best params : {reg_search.best_params_}")
        print(f"[train] CV R2 change: {reg_search.best_score_:.4f}")
    reg_metrics = exposure_report(
        regressor, test[exposure_cols], test["hrs_mean_7"], test["target_exposure"]
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
        "regressor_features":   exposure_cols,
        "regressor_metrics":    reg_metrics,
    }
    joblib.dump(artifact, out_path)
    print(f"[train] Saved -> {out_path}")

    # ── 11. Optional: log the run to MLflow ───────────────────────────────────
    if track:
        from pipeline.tracking import DEFAULT_EXPERIMENT, log_run

        clf_params = model.named_steps["clf"].get_params()
        reg_params = regressor.named_steps["reg"].get_params()
        keys = ("n_estimators", "max_depth", "min_samples_leaf", "max_features")
        params = {
            "mode": "fast" if fast else "grid_search",
            "exposure_features": exposure_set,
            "n_classifier_features": len(FEATURE_COLUMNS),
            "n_exposure_features": len(exposure_cols),
            "test_start": cfg.TEST_START,
            "label_horizon_days": cfg.LABEL_HORIZON_DAYS,
            "risk_threshold_low": round(lo, 4),
            "risk_threshold_high": round(hi, 4),
            "train_rows": len(train),
            "test_rows": len(test),
            **{f"clf_{k}": clf_params[k] for k in keys},
            **{f"reg_{k}": reg_params.get(k) for k in keys},
        }
        metrics = {
            "clf_test_accuracy": report["test"]["accuracy"],
            "clf_test_macro_f1": report["test"]["macro_f1"],
            "clf_train_macro_f1": report["train"]["macro_f1"],
            "clf_persistence_macro_f1": persistence["macro_f1"],
            "clf_gain_over_persistence": report["test"]["macro_f1"] - persistence["macro_f1"],
            "reg_test_mae": reg_metrics["mae"],
            "reg_no_change_mae": reg_metrics["mae_persistence"],
            "reg_error_reduction": 1 - reg_metrics["mae"] / reg_metrics["mae_persistence"],
            "reg_test_r2_change": reg_metrics["r2_change"],
            "reg_test_r2_level": reg_metrics["r2_level"],
        }
        if clf_cv is not None:
            metrics["clf_cv_macro_f1"] = clf_cv
        if reg_cv is not None:
            metrics["reg_cv_r2_change"] = reg_cv
        run_id, uri = log_run(
            run_name=f"{params['mode']}-{exposure_set}",
            params=params, metrics=metrics,
            tags={"cities": len(audit["cities_retained"])},
            artifact_path=out_path,
            experiment=experiment or DEFAULT_EXPERIMENT,
        )
        print(f"[train] MLflow run {run_id} -> {uri}")


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
    parser.add_argument(
        "--exposure-features", choices=sorted(EXPOSURE_FEATURE_SETS), default="weather",
        help="Features for the exposure forecast: pollution + past weather "
             "(default, production) or pollution only (ablation)"
    )
    parser.add_argument(
        "--mlflow", action="store_true",
        help="Log params, metrics and the artifact to MLflow (needs requirements-dev.txt)"
    )
    parser.add_argument(
        "--experiment", default=None,
        help="MLflow experiment name (default: air-quality-health-risk)"
    )
    args = parser.parse_args()
    main(args.data, args.out, args.fast, args.weather,
         exposure_set=args.exposure_features, track=args.mlflow, experiment=args.experiment)
