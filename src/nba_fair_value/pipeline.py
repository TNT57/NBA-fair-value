"""End-to-end pipeline: processed data -> features -> model -> results.

Run:  ``python -m nba_fair_value.pipeline``
Regenerates the fair-value results CSV and prints the CV report and the top
over/underpaid players (residual in dollars).
"""

from __future__ import annotations

import pandas as pd

from .config import OUTPUTS, PROCESSED_PARQUET
from .evaluate import cross_validate, importance, print_cv_report
from .features import build_features
from .models import fit_predict_fair_value, outlier_lens


def run(model_kind: str = "gbm") -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_PARQUET)
    print(f"Loaded {len(df)} player-seasons, "
          f"seasons={sorted(df['season'].unique())}")

    engineered, feature_cols = build_features(df)
    print(f"Built {len(feature_cols)} performance-only features.")

    # Validation first (honest skill), then fit on all data for reporting.
    for kind in ("linear", model_kind):
        metrics = cross_validate(engineered, feature_cols, kind=kind)
        print_cv_report(metrics)

    scored, model = fit_predict_fair_value(engineered, feature_cols, kind=model_kind)
    scored = outlier_lens(scored, feature_cols)

    print("\nTop feature drivers of fair value (permutation importance):")
    imp = importance(engineered, feature_cols, model)
    for _, r in imp.head(8).iterrows():
        print(f"  {r['feature']:>14}: {r['importance']:.4f}")

    _print_extremes(scored)

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    cols = [
        "Player Name", "season", "Team", "Position", "Age", "Salary",
        "fair_salary", "residual_dollars", "residual_pct",
        "PTS", "AST", "TRB", "WS", "PER", "VORP",
        "IF_label", "IF_score", "LOF_label", "LOF_score",
    ]
    out_path = OUTPUTS / "fair_value_results.csv"
    scored[cols].sort_values("residual_dollars", ascending=False).to_csv(
        out_path, index=False)
    print(f"\nWrote {out_path}")
    return scored


def _print_extremes(scored: pd.DataFrame, n: int = 8) -> None:
    s = scored.sort_values("residual_dollars", ascending=False)
    print(f"\nMost OVERPAID (actual >> fair value):")
    print("-" * 78)
    for _, r in s.head(n).iterrows():
        print(f"  {r['Player Name']:<22} {r['season']} | "
              f"paid ${r['Salary']/1e6:5.1f}M  fair ${r['fair_salary']/1e6:5.1f}M  "
              f"=> +${r['residual_dollars']/1e6:5.1f}M | WS {r['WS']:4.1f} PER {r['PER']:4.1f}")
    print(f"\nMost UNDERPAID (actual << fair value):")
    print("-" * 78)
    for _, r in s.tail(n).iloc[::-1].iterrows():
        print(f"  {r['Player Name']:<22} {r['season']} | "
              f"paid ${r['Salary']/1e6:5.1f}M  fair ${r['fair_salary']/1e6:5.1f}M  "
              f"=> ${r['residual_dollars']/1e6:6.1f}M | WS {r['WS']:4.1f} PER {r['PER']:4.1f}")


if __name__ == "__main__":
    run()
