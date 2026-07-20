"""Models: the fair-value regressor and the Isolation Forest / LOF lens.

- :func:`make_fair_value_model` builds the salary-from-performance regressor.
- :func:`fit_predict_fair_value` fits it and returns predictions + residuals in
  dollars (the over/underpaid signal).
- :func:`outlier_lens` runs Isolation Forest and LOF on the SAME performance-only
  feature space as a complementary "statistically unusual player" view.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    IsolationForest,
)
from sklearn.linear_model import Ridge
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import SALARY_CAP_BY_SEASON

RANDOM_STATE = 42


def make_fair_value_model(kind: str = "gbm"):
    """Return an unfitted regressor predicting salary_pct_of_cap.

    kind='linear' -> standardised Ridge (interpretable baseline)
    kind='gbm'    -> HistGradientBoosting (stronger, non-linear)
    """
    if kind == "linear":
        return Pipeline([
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ])
    if kind == "gbm":
        return HistGradientBoostingRegressor(
            max_depth=3,
            learning_rate=0.05,
            max_iter=400,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unknown model kind: {kind!r}")


def fit_predict_fair_value(df: pd.DataFrame, feature_cols: list[str],
                           kind: str = "gbm") -> pd.DataFrame:
    """Fit on all rows and attach fair-value predictions and residuals.

    Adds columns:
      pred_pct_of_cap  - predicted salary as fraction of cap
      fair_salary      - predicted salary in dollars (pred * season cap)
      residual_dollars - actual - fair (positive => overpaid)
      residual_pct     - residual as fraction of cap
    """
    out = df.copy()
    X = out[feature_cols].to_numpy()
    y = out["salary_pct_of_cap"].to_numpy()

    model = make_fair_value_model(kind)
    model.fit(X, y)
    out["pred_pct_of_cap"] = model.predict(X)

    caps = out["season"].map(SALARY_CAP_BY_SEASON).to_numpy()
    out["fair_salary"] = out["pred_pct_of_cap"] * caps
    out["residual_dollars"] = out["Salary"] - out["fair_salary"]
    out["residual_pct"] = out["salary_pct_of_cap"] - out["pred_pct_of_cap"]
    return out, model


def outlier_lens(df: pd.DataFrame, feature_cols: list[str],
                 contamination: float = 0.1) -> pd.DataFrame:
    """Isolation Forest + LOF on the performance-only space. Second view only."""
    out = df.copy()
    X = StandardScaler().fit_transform(out[feature_cols].to_numpy())

    iso = IsolationForest(
        n_estimators=200, contamination=contamination,
        max_samples="auto", random_state=RANDOM_STATE,
    )
    out["IF_label"] = iso.fit_predict(X)          # 1 normal, -1 outlier
    out["IF_score"] = iso.decision_function(X)    # lower => more anomalous

    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    out["LOF_label"] = lof.fit_predict(X)
    out["LOF_score"] = lof.negative_outlier_factor_
    return out
