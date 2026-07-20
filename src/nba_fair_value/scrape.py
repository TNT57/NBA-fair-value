"""Reproducible multi-season data acquisition.

Builds ``data/raw/nba_<season>.csv`` files matching the original Kaggle schema
from two public sources:

- **Stats**  : Basketball-Reference league pages (per-game + advanced tables).
- **Salary** : ESPN per-season salary pages (paginated).

Player rows are joined within a season on a normalised name key. Multi-team
players keep their combined ("2TM"/"3TM") season row.

Run:  ``python -m nba_fair_value.scrape 2019-20 2020-21 2021-22 2022-23 2023-24 2024-25``
(no args -> every season in SALARY_CAP_BY_SEASON)

Be polite: a crawl delay is enforced between requests.
"""

from __future__ import annotations

import io
import sys
import time
import unicodedata
import urllib.request

import numpy as np
import pandas as pd

from .config import DATA_RAW, SALARY_CAP_BY_SEASON

_UA = {"User-Agent": "Mozilla/5.0 (research; fair-value project)"}
_CRAWL_DELAY_S = 3.5  # Basketball-Reference asks for a slow crawl.


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "ignore")


def normalize_name(name: str) -> str:
    """Lowercase, strip accents/punctuation/suffixes for a robust join key."""
    name = str(name).split(",")[0]  # ESPN appends ", G" position
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = name.lower().replace(".", "").replace("'", "")
    for suf in (" jr", " sr", " ii", " iii", " iv"):
        if name.endswith(suf):
            name = name[: -len(suf)]
    return name.strip()


def season_to_end_year(season: str) -> int:
    """'2022-23' -> 2023 (Basketball-Reference / ESPN use the end year)."""
    start = int(season.split("-")[0])
    return start + 1


# ---------------------------------------------------------------------------
# Basketball-Reference stats
# ---------------------------------------------------------------------------
def _bbr_table(year: int, kind: str) -> pd.DataFrame:
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_{kind}.html"
    df = pd.read_html(io.StringIO(_get(url)))[0]
    df = df[df["Player"] != "Player"].copy()  # drop repeated header rows
    # One row per player: keep the max-games row (the combined multi-team line).
    df["G"] = pd.to_numeric(df["G"], errors="coerce")
    df = (df.sort_values("G", ascending=False)
            .drop_duplicates(subset=["Player"], keep="first"))
    return df


def scrape_bbr_stats(year: int) -> pd.DataFrame:
    """Merge per-game and advanced tables into one player-per-row frame."""
    per = _bbr_table(year, "per_game")
    time.sleep(_CRAWL_DELAY_S)
    adv = _bbr_table(year, "advanced")

    keep_per = ["Player", "Pos", "Age", "Team", "G", "GS", "MP",
                "FG%", "3P%", "FT%", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS"]
    keep_adv = ["Player", "PER", "TS%", "USG%", "OWS", "DWS", "WS", "WS/48",
                "OBPM", "DBPM", "BPM", "VORP"]
    per = per[[c for c in keep_per if c in per.columns]]
    adv = adv[[c for c in keep_adv if c in adv.columns]]

    df = per.merge(adv, on="Player", how="inner")
    df["_key"] = df["Player"].map(normalize_name)
    return df


# ---------------------------------------------------------------------------
# ESPN salaries
# ---------------------------------------------------------------------------
def scrape_espn_salaries(year: int, max_pages: int = 20) -> pd.DataFrame:
    """Loop ESPN salary pages until one returns no new rows."""
    rows = []
    for page in range(1, max_pages + 1):
        url = f"https://www.espn.com/nba/salaries/_/year/{year}/page/{page}"
        try:
            tab = pd.read_html(io.StringIO(_get(url)))[0]
        except (ValueError, urllib.error.HTTPError):
            break
        tab = tab[tab[0] != "RK"]           # drop repeated header rows
        if tab.empty:
            break
        rows.append(tab)
        if len(tab) < 40:                    # last (short) page
            break
        time.sleep(1.0)
    if not rows:
        return pd.DataFrame(columns=["_key", "Salary"])
    allrows = pd.concat(rows, ignore_index=True)
    out = pd.DataFrame({
        "_key": allrows[1].map(normalize_name),
        "Salary": (allrows[3].astype(str)
                   .str.replace(r"[\$,]", "", regex=True)
                   .pipe(pd.to_numeric, errors="coerce")),
    })
    return out.dropna(subset=["Salary"]).drop_duplicates("_key")


# ---------------------------------------------------------------------------
# Assemble one season into the target schema
# ---------------------------------------------------------------------------
def build_season(season: str) -> pd.DataFrame:
    year = season_to_end_year(season)
    print(f"[{season}] scraping Basketball-Reference stats (year {year})...")
    stats = scrape_bbr_stats(year)
    time.sleep(_CRAWL_DELAY_S)
    print(f"[{season}] scraping ESPN salaries...")
    sal = scrape_espn_salaries(year)

    df = stats.merge(sal, on="_key", how="inner")
    matched = len(df)
    print(f"[{season}] stats={len(stats)} salaries={len(sal)} "
          f"joined={matched}")

    # Coerce numerics and match the original column names.
    num_cols = ["Age", "G", "GS", "MP", "TRB", "AST", "STL", "BLK", "TOV", "PF",
                "PTS", "PER", "TS%", "USG%", "OWS", "DWS", "WS", "WS/48",
                "OBPM", "DBPM", "BPM", "VORP", "FG%", "3P%", "FT%"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Total Minutes"] = (df["G"] * df["MP"]).round(0)
    df = df.rename(columns={"Player": "Player Name", "Pos": "Position",
                            "G": "GP"})
    df = df.drop(columns=["_key"])
    return df


def scrape_and_save(seasons: list[str]) -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        if season not in SALARY_CAP_BY_SEASON:
            print(f"[{season}] SKIP — no verified salary cap in config.py")
            continue
        out_path = DATA_RAW / f"nba_{season}.csv"
        df = build_season(season)
        df.to_csv(out_path, index=False)
        print(f"[{season}] wrote {out_path} ({len(df)} players)\n")
        time.sleep(_CRAWL_DELAY_S)


if __name__ == "__main__":
    seasons = sys.argv[1:] or list(SALARY_CAP_BY_SEASON.keys())
    scrape_and_save(seasons)
