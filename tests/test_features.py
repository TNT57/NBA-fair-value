"""Unit tests for feature engineering — the leakage guard is the important one."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_fair_value.config import LEAKAGE_TOKENS  # noqa: E402
from nba_fair_value.features import add_per36, build_features  # noqa: E402


@pytest.fixture
def sample() -> pd.DataFrame:
    """A tiny two-row frame with the columns build_features needs."""
    return pd.DataFrame({
        "Player Name": ["A. Player", "B. Player"],
        "Team": ["LAL", "BOS"],
        "Position": ["PG", "C-PF"],
        "season": ["2022-23", "2022-23"],
        "Salary": [30_000_000, 2_000_000],
        "salary_pct_of_cap": [0.24, 0.016],
        "Age": [30, 22],
        "MP": [34.0, 18.0],
        "GP": [70, 60],
        "PTS": [25.0, 8.0], "AST": [6.0, 1.0], "TRB": [5.0, 7.0],
        "STL": [1.2, 0.4], "BLK": [0.3, 1.5], "TOV": [3.0, 0.8],
        "WS": [8.0, 3.0], "PER": [22.0, 15.0], "USG%": [30.0, 14.0],
        "BPM": [4.0, -1.0], "VORP": [3.5, 0.4], "TS%": [0.60, 0.58],
    })


def test_no_salary_leakage_in_model_features(sample):
    """Model feature names must not contain any salary-derived token."""
    _, feature_cols = build_features(sample)
    for col in feature_cols:
        low = col.lower()
        assert not any(tok in low for tok in LEAKAGE_TOKENS), (
            f"Leaked feature '{col}' entered the model matrix"
        )


def test_feature_matrix_is_finite(sample):
    """No NaN or inf may reach the model, even for MP==0 edge cases."""
    edge = sample.copy()
    edge.loc[1, "MP"] = 0.0  # would make per-36 divide by zero
    engineered, feature_cols = build_features(edge)
    X = engineered[feature_cols].to_numpy(dtype=float)
    assert np.isfinite(X).all()


def test_per36_math(sample):
    """per36 = stat * 36 / MP for a known row."""
    out = add_per36(sample)
    expected = 25.0 * 36.0 / 34.0
    assert out.loc[0, "PTS_per36"] == pytest.approx(expected)


def test_position_dummies_use_primary(sample):
    """Hybrid 'C-PF' should one-hot to the primary position (C)."""
    engineered, _ = build_features(sample)
    assert engineered.loc[1, "pos_C"] == 1
    assert "pos_PG" in engineered.columns
