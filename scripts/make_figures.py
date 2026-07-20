"""Regenerate the portfolio figures used in the README.

Run:  python scripts/make_figures.py
Writes PNGs to outputs/figures/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_fair_value.config import OUTPUTS, PROCESSED_PARQUET  # noqa: E402
from nba_fair_value.evaluate import cross_validate  # noqa: E402
from nba_fair_value.features import build_features  # noqa: E402
from nba_fair_value.models import fit_predict_fair_value  # noqa: E402
from nba_fair_value.viz import PALETTE, apply_style, fair_vs_actual  # noqa: E402

FIG_DIR = OUTPUTS / "figures"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()

    df = pd.read_parquet(PROCESSED_PARQUET)
    engineered, feature_cols = build_features(df)
    scored, _ = fit_predict_fair_value(engineered, feature_cols, kind="gbm")

    # 1. Fair vs actual for the most recent season -------------------------
    latest = sorted(scored["season"].unique())[-1]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    fair_vs_actual(scored[scored["season"] == latest], ax=ax)
    ax.set_title(f"Fair value vs actual salary — {latest}\n"
                 "red = overpaid, blue = underpaid", loc="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fair_vs_actual.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Leave-one-season-out CV R^2 by held-out season --------------------
    m_lin = cross_validate(engineered, feature_cols, kind="linear")
    m_gbm = cross_validate(engineered, feature_cols, kind="gbm")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = range(len(m_gbm["fold_names"]))
    ax.bar([i - 0.2 for i in x], m_lin["r2_per_fold"], width=0.38,
           color=PALETTE["neutral"], label=f"Ridge (mean {m_lin['r2_mean']:.2f})")
    ax.bar([i + 0.2 for i in x], m_gbm["r2_per_fold"], width=0.38,
           color=PALETTE["underpaid"], label=f"GBM (mean {m_gbm['r2_mean']:.2f})")
    ax.set_xticks(list(x))
    ax.set_xticklabels(m_gbm["fold_names"])
    ax.set_ylabel("R² on held-out season")
    ax.set_title("Generalisation: leave-one-season-out cross-validation",
                 loc="left")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", lw=0.6, alpha=0.7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cv_by_season.png", bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
