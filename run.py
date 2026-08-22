"""End-to-end run: clean -> split -> feature-build -> baselines -> models -> report.

Runnable top to bottom from a clean interpreter:

    python run.py

Ordering here is deliberate. The temporal split happens *before* anything is
fitted, so the pollutant ranker, the risk thresholds, the imputers and the
scalers all see training rows only.
"""

import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline import config as cfg
from pipeline import data as data_mod
from pipeline import features as feat
from pipeline import models as mdl

warnings.filterwarnings("ignore", category=FutureWarning)


def log(msg=""):
    print(msg, flush=True)


def section(title):
    log()
    log("=" * 78)
    log(title)
    log("=" * 78)


def main():
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}

    # ---------------------------------------------------------------- cleaning
    section("1. LOAD & DATA-QUALITY AUDIT")
    df, report = data_mod.build_clean_frame()
    log(f"max calendar gap within any city : {report['max_calendar_gap_days']} days")
    log(f"dropped (history < {cfg.MIN_CITY_OBSERVATIONS} days)   : {report['dropped_insufficient_history']}")
    log(f"dropped (weight coverage < {cfg.MIN_CITY_VALID_RATE:.0%}) : {report['dropped_poor_coverage']}")
    log(f"rows flagged unscoreable         : {report['rows_flagged_low_coverage']:,}")
    log(f"cities retained ({len(report['cities_retained'])})            : {', '.join(report['cities_retained'])}")
    log(f"rows retained                    : {report['rows_retained']:,}")
    summary["data_audit"] = report

    # ------------------------------------------------------------------- split
    section("2. TEMPORAL SPLIT")
    test_start = pd.Timestamp(cfg.TEST_START)
    train_end = test_start - pd.Timedelta(days=cfg.EMBARGO_DAYS)

    is_train = df["Date"] < train_end
    is_test = df["Date"] >= test_start
    log(f"train : {df.loc[is_train, 'Date'].min().date()} -> {df.loc[is_train, 'Date'].max().date()}")
    log(f"embargo gap of {cfg.EMBARGO_DAYS} days ({train_end.date()} -> {test_start.date()}) discarded,")
    log(f"        so no training label is built from a test-period day")
    log(f"test  : {df.loc[is_test, 'Date'].min().date()} -> {df.loc[is_test, 'Date'].max().date()}")

    # --------------------------------------------------------- feature building
    section("3. COMPOSITE SCORE & CAUSAL FEATURES")
    ranker = feat.fit_pollutant_ranker(df.loc[is_train])
    log("percentile-rank transform fitted on TRAIN rows only")

    df = feat.apply_composite_score(df, ranker)
    df = feat.add_causal_features(df)
    df = feat.add_forward_target(df)
    log(f"label = mean exposure over days t+1..t+{cfg.LABEL_HORIZON_DAYS}")
    log(f"features use days <= t only; windows {cfg.ROLLING_WINDOWS} require full history")

    hrs_features = [c for c in feat.FEATURE_COLUMNS if c.startswith("hrs_")]
    complete = df[hrs_features].notna().all(axis=1) & df["target_exposure"].notna()
    log(f"rows with complete history + label: {int(complete.sum()):,} of {len(df):,}")

    # Thresholds from training exposure only.
    lo, hi = feat.fit_risk_thresholds(df.loc[is_train & complete, "target_exposure"])
    log(f"risk thresholds from TRAIN only   : Low <= {lo:.4f} < Moderate <= {hi:.4f} < High")
    df["risk"] = feat.apply_risk_labels(df["target_exposure"], lo, hi)

    train = df[is_train & complete]
    test = df[is_test & complete]
    X_train, y_train = train[feat.FEATURE_COLUMNS], train["risk"].to_numpy()
    X_test, y_test = test[feat.FEATURE_COLUMNS], test["risk"].to_numpy()
    log(f"train rows {len(X_train):,} | test rows {len(X_test):,} | features {len(feat.FEATURE_COLUMNS)}")

    log()
    log("class balance (train -> test):")
    tr = pd.Series(y_train).value_counts(normalize=True)
    te = pd.Series(y_test).value_counts(normalize=True)
    for lbl in cfg.RISK_LABELS:
        log(f"  {lbl:9s} {tr.get(lbl, 0):6.1%} -> {te.get(lbl, 0):6.1%}")

    summary["split"] = {
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "train_end": str(train_end.date()), "test_start": str(test_start.date()),
        "thresholds": {"low": lo, "high": hi},
        "n_features": len(feat.FEATURE_COLUMNS),
    }

    # --------------------------------------------------------------- baselines
    section("4. BASELINES")
    maj_metrics, _ = mdl.majority_baseline(y_train, y_test)
    log(f"majority class    acc {maj_metrics['accuracy']:.3f}  macroF1 {maj_metrics['macro_f1']:.3f}")

    pers_metrics, _ = mdl.persistence_baseline(test["hrs_mean_7"], lo, hi, y_test)
    log(f"persistence       acc {pers_metrics['accuracy']:.3f}  macroF1 {pers_metrics['macro_f1']:.3f}")
    log("  (persistence = 'next week looks like this week' -- the bar a forecaster must clear)")
    summary["baselines"] = {"majority": maj_metrics, "persistence": pers_metrics}

    # ------------------------------------------------------------------ models
    section("5. MODELS")
    log("logistic regression (scaled, in-pipeline)...")
    logit = mdl.build_logistic().fit(X_train, y_train)
    logit_rep = mdl.full_report("logistic_regression", logit, X_train, y_train, X_test, y_test)
    log(f"  train macroF1 {logit_rep['train']['macro_f1']:.3f} | test macroF1 {logit_rep['test']['macro_f1']:.3f}"
        f" | test acc {logit_rep['test']['accuracy']:.3f}")

    log()
    log(f"random forest -- GridSearchCV over {np.prod([len(v) for v in mdl.RF_GRID.values()])} configs,")
    log("  forward-chaining TimeSeriesSplit, scored on macro F1...")
    search = mdl.tune_random_forest(X_train, y_train, train["Date"])
    log(f"  best params : {search.best_params_}")
    log(f"  best CV macroF1 : {search.best_score_:.3f}")
    rf = search.best_estimator_
    rf_rep = mdl.full_report("random_forest", rf, X_train, y_train, X_test, y_test)
    log(f"  train macroF1 {rf_rep['train']['macro_f1']:.3f} | test macroF1 {rf_rep['test']['macro_f1']:.3f}"
        f" | test acc {rf_rep['test']['accuracy']:.3f}")

    summary["models"] = {"logistic_regression": logit_rep, "random_forest": rf_rep}
    summary["rf_best_params"] = {k: str(v) for k, v in search.best_params_.items()}
    summary["rf_cv_macro_f1"] = float(search.best_score_)

    # ------------------------------------------------------------- comparison
    section("6. RESULTS vs BASELINES")
    rows = [
        ("majority class", maj_metrics["accuracy"], maj_metrics["macro_f1"]),
        ("persistence", pers_metrics["accuracy"], pers_metrics["macro_f1"]),
        ("logistic regression", logit_rep["test"]["accuracy"], logit_rep["test"]["macro_f1"]),
        ("random forest", rf_rep["test"]["accuracy"], rf_rep["test"]["macro_f1"]),
    ]
    log(f"{'':22s} {'accuracy':>9s} {'macro F1':>9s}")
    for name, acc, f1 in rows:
        log(f"{name:22s} {acc:9.3f} {f1:9.3f}")

    best_model_f1 = max(logit_rep["test"]["macro_f1"], rf_rep["test"]["macro_f1"])
    lift = best_model_f1 - pers_metrics["macro_f1"]
    log()
    log(f"best model beats persistence by {lift:+.3f} macro F1")

    # ------------------------------------------------- distribution-shift check
    section("7. COVID-19 REGIME CHECK")
    covid_start = pd.Timestamp(cfg.COVID_START)
    pre = test["Date"] < covid_start
    log(f"India's lockdown ({covid_start.date()}) falls inside the test window and")
    log("changed air quality sharply, so test metrics are split around it:")
    shift = {}
    for label, mask in [("pre-lockdown", pre), ("lockdown", ~pre)]:
        if mask.sum() == 0:
            continue
        m = mask.to_numpy()
        seg = {
            "n": int(m.sum()),
            "persistence": mdl.score(y_test[m], feat.apply_risk_labels(
                test["hrs_mean_7"], lo, hi).to_numpy()[m]),
            "random_forest": mdl.score(y_test[m], rf.predict(X_test[m])),
        }
        shift[label] = seg
        log(f"  {label:14s} n={seg['n']:5d}  persistence macroF1 {seg['persistence']['macro_f1']:.3f}"
            f"  RF macroF1 {seg['random_forest']['macro_f1']:.3f}")
    summary["covid_split"] = shift

    # ------------------------------------------------------------- importance
    section("8. PERMUTATION IMPORTANCE (held-out test)")
    imp = mdl.importance_report(rf, X_test, y_test)
    for _, r in imp.head(12).iterrows():
        log(f"  {r['feature']:18s} {r['importance']:+.4f} +/- {r['std']:.4f}")
    imp.to_csv(cfg.RESULTS_DIR / "permutation_importance.csv", index=False)
    summary["permutation_importance"] = imp.to_dict(orient="records")

    # ------------------------------------------------------------------ output
    section("9. ARTIFACTS")
    joblib.dump({"model": rf, "ranker": ranker, "thresholds": (lo, hi),
                 "features": feat.FEATURE_COLUMNS},
                cfg.RESULTS_DIR / "model.joblib")

    with open(cfg.RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    _plot_confusion(rf_rep, cfg.RESULTS_DIR / "confusion_matrix.png")
    _plot_comparison(rows, cfg.RESULTS_DIR / "model_vs_baselines.png")
    _plot_importance(imp, cfg.RESULTS_DIR / "permutation_importance.png")

    for p in sorted(cfg.RESULTS_DIR.iterdir()):
        log(f"  wrote {p.name}")
    log()
    log("done.")


def _plot_confusion(rep, path):
    cm = np.array(rep["confusion_matrix"])
    labels = rep["confusion_labels"]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("Random forest -- next-week risk")
    thresh = cm.max() / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_comparison(rows, path):
    names = [r[0] for r in rows]
    f1s = [r[2] for r in rows]
    colors = ["#999999", "#d08770", "#5e81ac", "#a3be8c"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.barh(names, f1s, color=colors)
    ax.set_xlabel("macro F1 (test)")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_title("Models vs baselines")
    for b, v in zip(bars, f1s):
        ax.text(v + 0.012, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_importance(imp, path):
    top = imp.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color="#5e81ac")
    ax.set_xlabel("drop in macro F1 when shuffled")
    ax.set_title("Permutation importance (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
