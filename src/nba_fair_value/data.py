"""Data acquisition, cleaning, and multi-season merge.

Reads every ``data/raw/nba_<season>.csv`` file (Basketball-Reference schema),
cleans it, normalises salary to a fraction of that season's cap, stacks all
seasons, and writes ``data/processed/players.parquet``.

Run as a script:  ``python -m nba_fair_value.data``
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .config import (
    DATA_PROCESSED,
    DATA_RAW,
    MIN_TOTAL_MINUTES,
    PROCESSED_PARQUET,
    SALARY_CAP_BY_SEASON,
)

_SEASON_RE = re.compile(r"nba_(\d{4}-\d{2})", re.IGNORECASE)


def season_from_filename(path: Path) -> str:
    """Extract a season label like '2022-23' from a raw CSV filename."""
    m = _SEASON_RE.search(path.stem)
    if not m:
        raise ValueError(
            f"Cannot parse season from '{path.name}'. "
            "Expected a name like 'nba_2022-23.csv'."
        )
    return m.group(1)


def load_raw_season(path: Path) -> pd.DataFrame:
    """Load and lightly clean one raw season CSV.

    - drops the unnamed index column Basketball-Reference exports leave behind
    - resolves traded players (Team like 'LAL/LAC') to their last team
    - coerces the modelling/reporting columns to numeric
    - attaches ``season`` and ``salary_pct_of_cap``
    """
    season = season_from_filename(path)
    df = pd.read_csv(path)

    # Drop the leading unnamed index column if present.
    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        df = df.drop(columns=df.columns[0])

    df["season"] = season

    # A traded player shows as 'TOT' rows or 'LAL/LAC'; keep the aggregate row
    # (Basketball-Reference's combined season line) and shorten the team label.
    df["Team"] = df["Team"].astype(str).str.split("/").str[-1]

    # Deduplicate on player within a season (keep the highest-minutes row).
    if "Total Minutes" in df.columns:
        df = (
            df.sort_values("Total Minutes", ascending=False)
            .drop_duplicates(subset=["Player Name"], keep="first")
            .reset_index(drop=True)
        )

    cap = SALARY_CAP_BY_SEASON.get(season)
    if cap is None:
        raise KeyError(
            f"No salary cap recorded for season '{season}' in config.py. "
            "Add and verify it before processing this season."
        )
    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
    df["salary_pct_of_cap"] = df["Salary"] / cap

    return df


def load_all_seasons(raw_dir: Path = DATA_RAW) -> pd.DataFrame:
    """Load and stack every raw season CSV found in ``raw_dir``."""
    paths = sorted(raw_dir.glob("nba_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No raw season files (nba_*.csv) found in {raw_dir}")
    frames = [load_raw_season(p) for p in paths]
    merged = pd.concat(frames, ignore_index=True)
    return merged


def filter_modellable(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only player-seasons with enough minutes and a valid salary."""
    before = len(df)
    out = df[df["Total Minutes"] >= MIN_TOTAL_MINUTES].copy()
    out = out.dropna(subset=["Salary"])
    out = out[out["Salary"] > 0]
    out = out.reset_index(drop=True)
    print(f"  Filtered {before} -> {len(out)} player-seasons "
          f"(Total Minutes >= {MIN_TOTAL_MINUTES}, valid Salary)")
    return out


def build_processed(raw_dir: Path = DATA_RAW,
                    out_path: Path = PROCESSED_PARQUET) -> pd.DataFrame:
    """Full pipeline: load raw -> filter -> write parquet. Returns the frame."""
    print("Loading raw seasons...")
    df = load_all_seasons(raw_dir)
    print(f"  Loaded {len(df)} player-seasons across "
          f"{df['season'].nunique()} season(s): {sorted(df['season'].unique())}")
    df = filter_modellable(df)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} ({len(df)} rows, {df.shape[1]} cols)")
    return df


if __name__ == "__main__":
    build_processed()
