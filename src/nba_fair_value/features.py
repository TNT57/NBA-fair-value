"""Feature engineering for the fair-value model.

The single most important rule here: **model inputs are performance-only**. No
salary and no salary-derived quantity is ever returned by :func:`build_features`.
Salary ratios that were used as model inputs in the original assignment now live
purely in the reporting layer (:func:`add_reporting_columns`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BASE_PERF_FEATURES, LEAKAGE_TOKENS, PER36_SOURCE_COLS


def add_per36(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-36-minute rate stats. MP is per-game minutes in this schema, so
    per-36 = stat * 36 / MP normalises for playing time without leaking salary."""
    out = df.copy()
    mp = out["MP"].replace(0, np.nan)
    for col in PER36_SOURCE_COLS:
        out[f"{col}_per36"] = out[col] * 36.0 / mp
    return out


def add_age_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Age and Age^2 capture the non-linear NBA earnings/production curve."""
    out = df.copy()
    out["Age2"] = out["Age"] ** 2
    return out


def add_position_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot the primary position (first listed for hybrids like 'PG-SG')."""
    out = df.copy()
    primary = out["Position"].astype(str).str.split("-").str[0]
    dummies = pd.get_dummies(primary, prefix="pos")
    return pd.concat([out, dummies], axis=1)


def add_reporting_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Salary-derived quantities for DISPLAY ONLY (never model inputs).

    These reproduce the intuitive value ratios from the original assignment but
    are kept strictly out of the feature matrix.
    """
    out = df.copy()
    out["salary_per_WS"] = out["Salary"] / (out["WS"].abs() + 1.0)
    out["pts_per_dollar"] = out["PTS"] / (out["Salary"] / 1_000_000)
    out["PER_salary_ratio"] = out["PER"] / (out["Salary"] / 1_000_000)
    return out


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return ``(engineered_df, feature_cols)``.

    ``engineered_df`` contains reporting columns AND the engineered feature
    columns; ``feature_cols`` names ONLY the performance-only model inputs.
    """
    out = add_per36(df)
    out = add_age_curve(out)
    out = add_position_dummies(out)
    out = add_reporting_columns(out)

    per36_cols = [f"{c}_per36" for c in PER36_SOURCE_COLS]
    pos_cols = [c for c in out.columns if c.startswith("pos_")]
    feature_cols = BASE_PERF_FEATURES + ["Age2"] + per36_cols + pos_cols

    # Fill any residual NaN/inf in the feature matrix (e.g. MP==0 edge cases).
    out[feature_cols] = (
        out[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    _assert_no_leakage(feature_cols)
    return out, feature_cols


def _assert_no_leakage(feature_cols: list[str]) -> None:
    """Raise if any feature name looks salary-derived. Cheap safety net that
    mirrors the check in tests/test_features.py."""
    for col in feature_cols:
        low = col.lower()
        for token in LEAKAGE_TOKENS:
            if token in low:
                raise ValueError(
                    f"Leakage: feature '{col}' matches banned token '{token}'."
                )
