"""Shared visual system for the app and the README figures.

Over/underpaid is a *polarity* (diverging) encoding, so it uses a validated
blue<->red diverging pair with a neutral gray midpoint (dataviz palette):
underpaid = blue, overpaid = red. One place defines the palette so every figure
reads as one system.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- palette (from the validated reference instance) ------------------------
PALETTE = {
    "overpaid": "#e34948",   # red pole  (paid above fair value)
    "underpaid": "#2a78d6",  # blue pole (paid below fair value)
    "neutral": "#b8b7b1",    # gray midpoint / fairly-paid
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "baseline": "#c3c2b7",
}


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": PALETTE["surface"],
        "axes.facecolor": PALETTE["surface"],
        "axes.edgecolor": PALETTE["baseline"],
        "axes.labelcolor": PALETTE["secondary"],
        "axes.titlecolor": PALETTE["ink"],
        "xtick.color": PALETTE["muted"],
        "ytick.color": PALETTE["muted"],
        "grid.color": PALETTE["grid"],
        "text.color": PALETTE["ink"],
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 130,
        "savefig.dpi": 130,
    })


def residual_color(residual: float) -> str:
    if residual > 0:
        return PALETTE["overpaid"]
    if residual < 0:
        return PALETTE["underpaid"]
    return PALETTE["neutral"]


def fair_vs_actual(df: pd.DataFrame, highlight: str | None = None,
                   ax=None):
    """Scatter of fair (x) vs actual (y) salary in $M, colored by residual
    sign. The y=x diagonal is 'paid exactly fair value'."""
    apply_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    fair = df["fair_salary"] / 1e6
    actual = df["Salary"] / 1e6
    colors = df["residual_dollars"].map(residual_color)

    ax.scatter(fair, actual, c=colors, s=26, alpha=0.72,
               edgecolors="white", linewidth=0.4, zorder=2)

    lim = float(max(fair.max(), actual.max())) * 1.05
    ax.plot([0, lim], [0, lim], color=PALETTE["baseline"], lw=1.4,
            ls="--", zorder=1)
    ax.text(lim * 0.72, lim * 0.78, "paid = fair value",
            color=PALETTE["muted"], fontsize=9, rotation=45, ha="left")

    if highlight is not None and (df["Player Name"] == highlight).any():
        r = df[df["Player Name"] == highlight].iloc[0]
        ax.scatter([r["fair_salary"] / 1e6], [r["Salary"] / 1e6],
                   s=140, facecolor="none",
                   edgecolor=PALETTE["ink"], linewidth=1.8, zorder=3)
        ax.annotate(highlight,
                    (r["fair_salary"] / 1e6, r["Salary"] / 1e6),
                    color=PALETTE["ink"], fontsize=10, fontweight="bold",
                    xytext=(8, 8), textcoords="offset points")

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Fair value ($M)")
    ax.set_ylabel("Actual salary ($M)")
    ax.grid(True, lw=0.6, alpha=0.7)
    return ax


def contribution_bars(contrib: pd.Series, ax=None, top: int = 10):
    """Horizontal diverging bars: how each stat pushes a player's fair value
    up (red) or down (blue) relative to the league baseline."""
    apply_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    s = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(top)
    s = s.iloc[::-1]
    colors = [PALETTE["overpaid"] if v > 0 else PALETTE["underpaid"] for v in s]
    ax.barh(s.index, s.values, color=colors, height=0.68, zorder=2)
    ax.axvline(0, color=PALETTE["baseline"], lw=1.2)
    ax.set_xlabel("Effect on fair value  (blue = lowers, red = raises)")
    ax.grid(True, axis="x", lw=0.6, alpha=0.7)
    ax.tick_params(length=0)
    return ax
