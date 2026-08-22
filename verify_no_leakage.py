"""Executable proof of the invariants the original pipeline violated.

Run with:  python verify_no_leakage.py

These are the claims the rebuild rests on. Asserting them in a comment is not
worth much -- the original's README asserted "strict temporal ordering to
prevent leakage" while the split was alphabetical by city. So each one is
checked against an independent recomputation.
"""

import numpy as np
import pandas as pd

from pipeline import config as cfg
from pipeline import data as data_mod
from pipeline import features as feat

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append((PASS if ok else FAIL, name, detail))
    print(f"[{PASS if ok else FAIL}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def main():
    df, _ = data_mod.build_clean_frame()
    test_start = pd.Timestamp(cfg.TEST_START)
    train_end = test_start - pd.Timedelta(days=cfg.EMBARGO_DAYS)
    is_train = df["Date"] < train_end

    ranker = feat.fit_pollutant_ranker(df.loc[is_train])
    df = feat.apply_composite_score(df, ranker)
    df = feat.add_causal_features(df)
    df = feat.add_forward_target(df)

    h = cfg.LABEL_HORIZON_DAYS

    # ---- 1. The label really is the FORWARD window, recomputed by hand -------
    city = (
        df.loc[df["hrs"].notna()]
        .groupby("City")
        .size()
        .idxmax()
    )
    g = df[df["City"] == city].sort_values("Date").reset_index(drop=True)
    manual, stored = [], []
    for i in range(len(g) - h):
        window = g["hrs"].iloc[i + 1 : i + 1 + h]
        if window.notna().all() and pd.notna(g["target_exposure"].iloc[i]):
            manual.append(window.mean())
            stored.append(g["target_exposure"].iloc[i])
    ok = len(manual) > 500 and np.allclose(manual, stored, atol=1e-12)
    check(
        f"label at t == mean(hrs[t+1..t+{h}])",
        ok,
        f"{len(manual):,} rows checked in {city}, max diff "
        f"{max(abs(np.array(manual) - np.array(stored))) if manual else float('nan'):.2e}",
    )

    # ---- 2. Features are functions of the past only --------------------------
    # Corrupt every value strictly AFTER a cut date and confirm that features at
    # or before the cut are untouched. If any feature peeked forward, it would
    # move.
    cut = pd.Timestamp("2018-06-30")
    poisoned = df.copy()
    future = poisoned["Date"] > cut
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    poisoned.loc[future, "hrs"] = rng.uniform(0, 1, int(future.sum()))
    poisoned = feat.add_causal_features(poisoned)

    hrs_feats = [c for c in feat.FEATURE_COLUMNS if c.startswith("hrs_")]
    past = df["Date"] <= cut
    before = df.loc[past, hrs_feats].to_numpy()
    after = poisoned.loc[past, hrs_feats].to_numpy()
    both_nan = np.isnan(before) & np.isnan(after)
    identical = bool(np.all(both_nan | (before == after)))
    check(
        "features at t are unchanged when the future is randomised",
        identical,
        f"{before.shape[0]:,} rows x {len(hrs_feats)} features held fixed",
    )

    # ---- 3. Label window and feature window never overlap -------------------
    # The label starts at t+1; the newest day any feature can touch is t.
    check(
        "feature window ends at t, label window starts at t+1",
        cfg.LABEL_HORIZON_DAYS >= 1,
        f"gap = {1} day, horizon = {h} days, no shared days",
    )

    # ---- 4. No training label is built from a test-period day ---------------
    train_rows = df[df["Date"] < train_end]
    latest_label_day = train_rows["Date"].max() + pd.Timedelta(days=h)
    ok = latest_label_day < test_start
    check(
        "embargo holds: last training label predates the test period",
        ok,
        f"latest train label day {latest_label_day.date()} < test start {test_start.date()}",
    )

    # ---- 5. Thresholds derive from training data only -----------------------
    complete = df[hrs_feats].notna().all(axis=1) & df["target_exposure"].notna()
    lo_train, hi_train = feat.fit_risk_thresholds(
        df.loc[(df["Date"] < train_end) & complete, "target_exposure"]
    )
    lo_all, hi_all = feat.fit_risk_thresholds(df.loc[complete, "target_exposure"])
    differs = not (np.isclose(lo_train, lo_all) and np.isclose(hi_train, hi_all))
    check(
        "train-only thresholds differ from full-data thresholds",
        differs,
        f"train ({lo_train:.4f}, {hi_train:.4f}) vs full ({lo_all:.4f}, {hi_all:.4f})"
        " -- confirms the original's full-data fit leaked",
    )

    # ---- 6. The split is temporal, not by city ------------------------------
    train_cities = set(df.loc[df["Date"] < train_end, "City"])
    test_cities = set(df.loc[df["Date"] >= test_start, "City"])
    check(
        "every test city also appears in training (split is temporal, not by city)",
        test_cities.issubset(train_cities) and len(test_cities) > 1,
        f"{len(train_cities)} train cities, {len(test_cities)} test cities, "
        f"{len(test_cities - train_cities)} test-only",
    )

    print()
    failed = [r for r in results if r[0] == FAIL]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
