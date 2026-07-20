"""NBA Fair-Value Explorer — interactive demo.

Run locally:  streamlit run app/streamlit_app.py
Deploy:       Streamlit Community Cloud, entrypoint app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_fair_value.config import PROCESSED_PARQUET  # noqa: E402
from nba_fair_value.explain import linear_contributions  # noqa: E402
from nba_fair_value.features import build_features  # noqa: E402
from nba_fair_value.models import (  # noqa: E402
    fit_predict_fair_value,
    make_fair_value_model,
)
from nba_fair_value.viz import (  # noqa: E402
    PALETTE,
    contribution_bars,
    fair_vs_actual,
)

st.set_page_config(page_title="NBA Fair-Value Explorer", page_icon="🏀",
                   layout="wide")


@st.cache_data(show_spinner="Loading data and fitting models…")
def load_scored():
    df = pd.read_parquet(PROCESSED_PARQUET)
    engineered, feature_cols = build_features(df)
    scored, _ = fit_predict_fair_value(engineered, feature_cols, kind="gbm")
    # A linear model too, fit once, for per-player explanations.
    lin = make_fair_value_model("linear")
    lin.fit(engineered[feature_cols].to_numpy(),
            engineered["salary_pct_of_cap"].to_numpy())
    return scored, feature_cols, lin


scored, feature_cols, lin = load_scored()

st.title("🏀 NBA Fair-Value Explorer")
st.caption(
    "A model predicts each player's **fair salary from on-court performance "
    "only** (no salary leakage). The gap between actual pay and fair value = "
    "how over- or under-paid they are. Leave-one-season-out CV R² ≈ 0.73."
)

# --- filters ---------------------------------------------------------------
seasons = sorted(scored["season"].unique())
c1, c2, c3 = st.columns(3)
season = c1.selectbox("Season", seasons, index=len(seasons) - 1)
view = scored[scored["season"] == season].copy()
teams = ["All"] + sorted(view["Team"].dropna().unique())
team = c2.selectbox("Team", teams)
if team != "All":
    view = view[view["Team"] == team]
positions = ["All"] + sorted(view["Position"].astype(str).str[:2].unique())
pos = c3.selectbox("Position", positions)
if pos != "All":
    view = view[view["Position"].astype(str).str.startswith(pos)]

st.divider()

# --- player detail ---------------------------------------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("Player breakdown")
    player = st.selectbox(
        "Pick a player",
        view.sort_values("Salary", ascending=False)["Player Name"].tolist(),
    )
    row = view[view["Player Name"] == player].iloc[0]
    resid = row["residual_dollars"]
    verdict = ("Overpaid" if resid > 1.5e6 else
               "Underpaid" if resid < -1.5e6 else "Fairly paid")
    color = (PALETTE["overpaid"] if verdict == "Overpaid" else
             PALETTE["underpaid"] if verdict == "Underpaid" else PALETTE["muted"])

    m1, m2, m3 = st.columns(3)
    m1.metric("Actual salary", f"${row['Salary']/1e6:.1f}M")
    m2.metric("Fair value", f"${row['fair_salary']/1e6:.1f}M")
    m3.metric("Difference", f"${resid/1e6:+.1f}M")
    st.markdown(
        f"<h3 style='color:{color};margin-top:0'>{verdict}</h3>",
        unsafe_allow_html=True,
    )

    contrib = linear_contributions(lin, feature_cols, row)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    contribution_bars(contrib, ax=ax, top=8)
    ax.set_title(f"What drives {player}'s fair value")
    st.pyplot(fig, use_container_width=True)

with right:
    st.subheader(f"{season}: fair value vs actual pay")
    fig2, ax2 = plt.subplots(figsize=(6, 5.4))
    fair_vs_actual(view, highlight=player, ax=ax2)
    st.pyplot(fig2, use_container_width=True)
    st.caption("Red = overpaid (above the line) · Blue = underpaid (below).")

st.divider()

# --- leaderboards ----------------------------------------------------------
def _fmt(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["Player Name", "Team", "Position", "Salary", "fair_salary",
              "residual_dollars", "WS", "PER"]].copy()
    for c in ("Salary", "fair_salary", "residual_dollars"):
        out[c] = (out[c] / 1e6).round(1)
    return out.rename(columns={"Salary": "Paid $M", "fair_salary": "Fair $M",
                               "residual_dollars": "Diff $M"})


lc, rc = st.columns(2)
lc.subheader("Most overpaid")
lc.dataframe(_fmt(view.sort_values("residual_dollars", ascending=False).head(12)),
             hide_index=True, use_container_width=True)
rc.subheader("Most underpaid")
rc.dataframe(_fmt(view.sort_values("residual_dollars").head(12)),
             hide_index=True, use_container_width=True)

st.caption(
    "Salaries are ESPN season cap-hit figures; waived/bought-out players (e.g. "
    "dead-money contracts) can show anomalously low pay. Model is descriptive, "
    "not a contract recommendation."
)
