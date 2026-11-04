"""
train.py – Train the RandomForest model and save artifacts/model.joblib.

Usage:
    python train.py
    python train.py --data data/city_day.csv --out artifacts/model.joblib
"""
import argparse
import os
import pathlib

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from pipeline.preprocess import (
    FEATURE_COLUMNS,
    POLLUTANTS,
    add_temporal_features,
    apply_risk_labels,
    build_exposure_features,
    load_and_clean,
)
from sklearn.preprocessing import MinMaxScaler


def main(data_path: str, out_path: str) -> None:
    # ── 1. Load & clean ──────────────────────────────────────────────────────
    print(f"[train] Loading {data_path}")
    df = load_and_clean(data_path)
    df = add_temporal_features(df)

    # ── 2. Temporal split (80 / 20 chronological) ────────────────────────────
    cutoff = df["Date"].sort_values().iloc[int(len(df) * 0.8)]
    train_mask = df["Date"] < cutoff
    print(f"[train] Cutoff date: {cutoff.date()}  train={train_mask.sum()}  test={(~train_mask).sum()}")

    # ── 3. Fit scaler on training rows only ──────────────────────────────────
    scaler = MinMaxScaler()
    scaler.fit(df.loc[train_mask, POLLUTANTS])

    # ── 4. Build exposure features ───────────────────────────────────────────
    df = build_exposure_features(df, scaler)
    df = df.dropna(subset=["HRS_7day"]).reset_index(drop=True)

    # Recompute masks after dropna
    cutoff = df["Date"].sort_values().iloc[int(len(df) * 0.8)]
    train_mask = df["Date"] < cutoff

    # ── 5. Risk thresholds fitted on training set ────────────────────────────
    low_th, high_th = df.loc[train_mask, "HRS_7day"].quantile([0.33, 0.66])
    df["Health_Risk_Level"] = apply_risk_labels(df, low_th, high_th)

    # ── 6. Impute missing feature values with training medians ────────────────
    X = df[FEATURE_COLUMNS]
    y = df["Health_Risk_Level"]
    train_medians = X.loc[train_mask].median()
    X = X.fillna(train_medians)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[~train_mask], y[~train_mask]

    # ── 7. Train ─────────────────────────────────────────────────────────────
    print("[train] Fitting RandomForestClassifier …")
    rf = RandomForestClassifier(n_estimators=200, max_depth=50, random_state=42)
    rf.fit(X_train, y_train)

    print("\n[train] Test classification report:")
    print(classification_report(y_test, rf.predict(X_test)))

    # ── 8. Save artifacts ────────────────────────────────────────────────────
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model":        rf,
        "scaler":       scaler,
        "features":     FEATURE_COLUMNS,
        "classes":      list(rf.classes_),
        "low_th":       float(low_th),
        "high_th":      float(high_th),
        "train_medians": train_medians.to_dict(),
    }
    joblib.dump(artifact, out_path)
    print(f"[train] Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/city_day.csv")
    parser.add_argument("--out",  default="artifacts/model.joblib")
    args = parser.parse_args()
    main(args.data, args.out)

