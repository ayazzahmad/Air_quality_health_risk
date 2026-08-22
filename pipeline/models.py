"""Baselines, models, tuning and evaluation.

The baselines matter as much as the models. The original reported 89% accuracy
with nothing to compare it against, which made a largely tautological result
look strong. A forecaster's real bar is persistence -- "next week will look like
this week" -- and any model that cannot beat it is not adding information.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as cfg


def score(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


# --- Baselines ---------------------------------------------------------------

def majority_baseline(y_train, y_test) -> tuple[dict, np.ndarray]:
    """Always predict the most common training class."""
    majority = pd.Series(y_train).value_counts().idxmax()
    pred = np.full(len(y_test), majority, dtype=object)
    return score(y_test, pred), pred


def persistence_baseline(current_week_mean, lo: float, hi: float, y_test) -> tuple[dict, np.ndarray]:
    """Predict next week's risk level as the current week's risk level.

    This is the honest bar for a forecasting task, and it is the baseline whose
    absence let the original result look impressive: because the original label
    was built from the same window as its features, a rule this simple would
    have recovered most of that reported 89%.
    """
    from .features import apply_risk_labels
    pred = apply_risk_labels(current_week_mean, lo, hi).to_numpy()
    return score(y_test, pred), pred


# --- Models ------------------------------------------------------------------

def build_logistic() -> Pipeline:
    """Logistic regression WITH imputation and scaling inside the pipeline.

    The original fitted logistic regression on unscaled features spanning very
    different ranges, which both handicapped the model and made its coefficients
    meaningless -- so the reported 86%-vs-89% was not a fair comparison.
    Wrapping the steps in a Pipeline also guarantees the imputer and scaler are
    fitted on training folds only.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=cfg.RANDOM_STATE)),
    ])


def build_random_forest(n_jobs: int = 1) -> Pipeline:
    """Random forest pipeline.

    `n_jobs` defaults to 1 because this estimator is run *inside* a parallel
    GridSearchCV. Leaving both at -1 nests parallelism: every grid worker spawns
    a full set of tree workers, oversubscribing the CPU and slowing the search
    down rather than speeding it up.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            random_state=cfg.RANDOM_STATE,
            class_weight="balanced",
            n_jobs=n_jobs,
        )),
    ])


RF_GRID = {
    "clf__n_estimators": [300, 600],
    "clf__max_depth": [4, 6, 10, 16, None],
    "clf__min_samples_leaf": [1, 5, 20],
    # With ~17 features, sqrt and log2 both land near 4, so searching both adds
    # runtime without adding coverage.
    "clf__max_features": ["sqrt"],
}


def tune_random_forest(X_train, y_train, dates_train, n_splits: int = 4):
    """Tune on macro F1 with forward-chaining CV.

    The original picked `max_depth=50, n_estimators=200` with no search and no
    cross-validation, and never reported a training score -- so there was no way
    to see overfitting. TimeSeriesSplit is applied to date-ordered rows so each
    fold validates on a period strictly later than it trained on.
    """
    order = np.argsort(dates_train.to_numpy(), kind="stable")
    Xo = X_train.iloc[order]
    yo = np.asarray(y_train)[order]

    search = GridSearchCV(
        build_random_forest(),
        RF_GRID,
        scoring="f1_macro",
        cv=TimeSeriesSplit(n_splits=n_splits),
        n_jobs=-1,
        refit=True,
    )
    search.fit(Xo, yo)
    return search


# --- Reporting ---------------------------------------------------------------

def full_report(name, model, X_train, y_train, X_test, y_test) -> dict:
    """Train and test metrics side by side, so overfitting is visible."""
    return {
        "model": name,
        "train": score(y_train, model.predict(X_train)),
        "test": score(y_test, model.predict(X_test)),
        "test_report": classification_report(
            y_test, model.predict(X_test), output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_test, model.predict(X_test), labels=cfg.RISK_LABELS
        ).tolist(),
        "confusion_labels": cfg.RISK_LABELS,
    }


def importance_report(model, X_test, y_test, n_repeats: int = 5) -> pd.DataFrame:
    """Permutation importance on held-out data.

    Preferred over tree impurity importance (the original's choice), which is
    computed on training data and is biased toward high-cardinality features.
    """
    result = permutation_importance(
        model, X_test, y_test,
        scoring="f1_macro",
        n_repeats=n_repeats,
        random_state=cfg.RANDOM_STATE,
        n_jobs=-1,
    )
    return (
        pd.DataFrame({
            "feature": X_test.columns,
            "importance": result.importances_mean,
            "std": result.importances_std,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
