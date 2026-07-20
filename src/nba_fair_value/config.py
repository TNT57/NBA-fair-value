"""Project-wide constants: paths, feature definitions, and season salary caps.

Keeping these in one place makes the leakage rule auditable: the model may ONLY
consume names listed in ``MODEL_FEATURES`` (performance-only). Salary and any
salary-derived quantity live in ``REPORTING_COLS`` and never enter the model.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS = PROJECT_ROOT / "outputs"
PROCESSED_PARQUET = DATA_PROCESSED / "players.parquet"

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
# Minimum total minutes for a player-season to be modelled. Filters out
# deep-bench players whose tiny samples produce noisy advanced stats.
MIN_TOTAL_MINUTES = 500

# ---------------------------------------------------------------------------
# NBA salary cap by season ($). Used to normalise salaries across seasons so a
# 2019 contract is comparable to a 2024 one. Source: Basketball-Reference /
# NBA CBA. VERIFY each value against a canonical source when adding a season.
# ---------------------------------------------------------------------------
SALARY_CAP_BY_SEASON = {
    "2019-20": 109_140_000,
    "2020-21": 109_140_000,  # flat vs prior year (COVID)
    "2021-22": 112_414_000,
    "2022-23": 123_655_000,
    "2023-24": 136_021_000,
    "2024-25": 140_588_000,
}

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
# Performance-only inputs to the fair-value model. NO salary, NO salary ratios.
# These are the columns present in the Basketball-Reference schema.
BASE_PERF_FEATURES = [
    "PTS", "AST", "TRB", "STL", "BLK", "TOV",
    "WS", "PER", "USG%", "BPM", "VORP", "TS%",
    "MP", "GP", "Age",
]

# Per-36-minute rate features engineered in features.py (non-leaky, pace-free).
PER36_SOURCE_COLS = ["PTS", "AST", "TRB", "STL", "BLK", "TOV"]

# Columns used only for reporting / the second (outlier) lens — NEVER model
# inputs. salary_per_WS etc. are computed for display, not prediction.
REPORTING_COLS = [
    "Player Name", "Team", "Position", "season",
    "Salary", "salary_pct_of_cap",
]

# Prediction target: salary as a fraction of that season's cap.
TARGET = "salary_pct_of_cap"

# Guard list: if any of these ever appears in the model's feature matrix it is
# leakage. Enforced by tests/test_features.py.
LEAKAGE_TOKENS = ["salary", "log_salary", "pct_of_cap", "per_dollar", "per_ws"]
