"""Per-player explanation via the linear model's standardized contributions.

Works without SHAP: for the Ridge pipeline, contribution_i = coef_i * z_i where
z_i is the player's standardized feature value. The signed contributions sum
(plus intercept) to the predicted salary %-of-cap, so they read as "this stat
raised/lowered the player's fair value by X". Interpretable and dependency-free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline


def linear_contributions(pipeline: Pipeline, feature_cols: list[str],
                         row: pd.Series) -> pd.Series:
    """Return a signed contribution (in cap-% units) per feature for one player.

    ``pipeline`` must be the StandardScaler+Ridge pipeline from
    :func:`nba_fair_value.models.make_fair_value_model('linear')`, already fit.
    """
    scaler = pipeline.named_steps["scale"]
    ridge: Ridge = pipeline.named_steps["ridge"]

    x = row[feature_cols].to_numpy(dtype=float).reshape(1, -1)
    z = scaler.transform(x)[0]
    contrib = ridge.coef_ * z
    return pd.Series(contrib, index=feature_cols)
