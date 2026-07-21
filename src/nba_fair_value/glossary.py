"""Plain-English glossary for the NBA stat abbreviations used across the app.

Advanced basketball stats (WS, PER, USG%, BPM, VORP, TS%) are opaque to a
general audience, and even some box-score codes (TRB, TOV) are not obvious.
This module is the single source of truth for turning a feature/column code
into (a) a short human-readable label for chart axes and (b) a one-line
explanation for tooltips and the glossary panel.
"""

from __future__ import annotations

# code -> (short label, plain-English explanation)
STAT_GLOSSARY: dict[str, tuple[str, str]] = {
    "PTS": ("Points", "Points scored per game."),
    "AST": ("Assists", "Passes that directly lead to a teammate scoring, per game."),
    "TRB": ("Rebounds", "Missed shots grabbed off the rim, on offense or defense, per game."),
    "STL": ("Steals", "Times the player takes the ball away from the opponent, per game."),
    "BLK": ("Blocks", "Opponent shots swatted away, per game."),
    "TOV": ("Turnovers", "Times the player loses the ball to the other team, per game — lower is better."),
    "WS": ("Win Shares", "An estimate of how many of the team's wins a player is responsible for."),
    "PER": ("Efficiency (PER)", "Player Efficiency Rating: an all-in-one per-minute productivity score. 15 = league average."),
    "USG%": ("Usage %", "Share of the team's plays a player finishes (a shot, assist, or turnover) while on court — how central he is to the offense."),
    "BPM": ("Box Plus/Minus", "Points per 100 possessions the player adds above a league-average player."),
    "VORP": ("VORP", "Value Over Replacement Player: total value versus a freely-available bench player."),
    "TS%": ("True Shooting %", "Scoring efficiency that counts 2-pointers, 3-pointers, and free throws together."),
    "MP": ("Minutes", "Minutes played per game."),
    "GP": ("Games", "Games played — a measure of availability and health."),
    "Age": ("Age", "Player's age during the season."),
}

# The subset most worth explaining to a casual fan (the non-obvious ones).
GLOSSARY_HIGHLIGHT = ["WS", "PER", "USG%", "BPM", "VORP", "TS%", "TRB", "TOV"]

_POSITIONS = {
    "PG": "Point Guard",
    "SG": "Shooting Guard",
    "SF": "Small Forward",
    "PF": "Power Forward",
    "C": "Center",
}


def stat_label(code: str) -> str:
    """Short human-readable label for a stat code (falls back to the code)."""
    return STAT_GLOSSARY.get(code, (code, ""))[0]


def stat_help(code: str) -> str:
    """One-line explanation for a stat code (empty string if unknown)."""
    entry = STAT_GLOSSARY.get(code)
    return entry[1] if entry else ""


def pretty_feature_name(col: str) -> str:
    """Human-readable label for a model feature column, for chart axes.

    Handles the engineered variants: ``pos_C`` -> 'Center',
    ``PTS_per36`` -> 'Points /36 min', ``Age2`` -> 'Age (curve)'.
    """
    if col.startswith("pos_"):
        code = col[len("pos_"):]
        return _POSITIONS.get(code, code)
    if col == "Age2":
        return "Age (curve)"
    if col.endswith("_per36"):
        return f"{stat_label(col[:-len('_per36')])} /36 min"
    return stat_label(col)
