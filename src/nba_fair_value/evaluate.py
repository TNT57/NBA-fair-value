"""Validation: season-wise cross-validation and importance.

The key rigour upgrade over the original project. When >= 2 seasons are present
we use leave-one-season-out CV (train on past/other seasons, test on a held-out
season) so reported skill reflects generalisation, not memorisation. With a
single season we fall back to K-fold and say so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut

from .models import make_fair_value_model


def cross_validate(df: pd.DataFrame, feature_cols: list[str],
                   kind: str = "gbm") -> dict:
    """Return CV metrics dict with per-fold R2/MAE and the CV scheme used."""
    X = df[feature_cols].to_numpy()
    y = df["salary_pct_of_cap"].to_numpy()
    groups = df["season"].to_numpy()
    n_seasons = df["season"].nunique()

    if n_seasons >= 2:
        splitter = LeaveOneGroupOut().split(X, y, groups)
        scheme = "leave-one-season-out"
    else:
        splitter = KFold(n_splits=5, shuffle=True, random_state=42).split(X)
        scheme = "5-fold (single season; add seasons for season-wise CV)"

    r2s, maes, fold_names = [], [], []
    for train_idx, test_idx in splitter:
        model = make_fair_value_model(kind)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        r2s.append(r2_score(y[test_idx], pred))
        maes.append(mean_absolute_error(y[test_idx], pred))
        held = np.unique(groups[test_idx])
        fold_names.append(held[0] if len(held) == 1 else "fold")

    return {
        "scheme": scheme,
        "kind": kind,
        "fold_names": fold_names,
        "r2_per_fold": r2s,
        "mae_per_fold": maes,
        "r2_mean": float(np.mean(r2s)),
        "mae_mean": float(np.mean(maes)),
    }


def importance(df: pd.DataFrame, feature_cols: list[str], model,
               n_repeats: int = 20) -> pd.DataFrame:
    """Permutation importance (SHAP-free fallback). Higher => more predictive
    of fair value. Returns a sorted DataFrame."""
    X = df[feature_cols].to_numpy()
    y = df["salary_pct_of_cap"].to_numpy()
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=42,
        scoring="r2",
    )
    return (
        pd.DataFrame({
            "feature": feature_cols,
            "importance": result.importances_mean,
            "std": result.importances_std,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def print_cv_report(metrics: dict) -> None:
    print(f"\nCross-validation ({metrics['scheme']}), model={metrics['kind']}")
    print("-" * 60)
    for name, r2, mae in zip(metrics["fold_names"],
                             metrics["r2_per_fold"],
                             metrics["mae_per_fold"]):
        print(f"  held-out {name!s:>10}:  R2={r2:6.3f}   MAE(cap%)={mae:6.4f}")
    print("-" * 60)
    print(f"  MEAN            :  R2={metrics['r2_mean']:6.3f}   "
          f"MAE(cap%)={metrics['mae_mean']:6.4f}")
