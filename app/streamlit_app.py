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
from nba_fair_value.data import build_processed  # noqa: E402
from nba_fair_value.explain import linear_contributions  # noqa: E402
from nba_fair_value.features import build_features  # noqa: E402
from nba_fair_value.glossary import (  # noqa: E402
    GLOSSARY_HIGHLIGHT,
    STAT_GLOSSARY,
    pretty_feature_name,
    stat_help,
)
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

CV_R2 = 0.73  # validated leave-one-season-out CV R² (see README)


@st.cache_data(show_spinner="Loading data and fitting models…")
def load_scored():
    # On a fresh deploy (e.g. Streamlit Cloud) the processed parquet is
    # git-ignored and absent — rebuild it from the tracked raw CSVs.
    if not PROCESSED_PARQUET.exists():
        build_processed()
    df = pd.read_parquet(PROCESSED_PARQUET)
    engineered, feature_cols = build_features(df)
    scored, _ = fit_predict_fair_value(engineered, feature_cols, kind="gbm")
    # A linear model too, fit once, for per-player explanations.
    lin = make_fair_value_model("linear")
    lin.fit(engineered[feature_cols].to_numpy(),
            engineered["salary_pct_of_cap"].to_numpy())
    return scored, feature_cols, lin


scored, feature_cols, lin = load_scored()

# --- sidebar: what this is + a plain-English glossary ----------------------
with st.sidebar:
    st.header("About")
    st.markdown(
        "A machine-learning model learns each player's **fair salary from "
        "on-court performance only** — it never sees what anyone is paid. "
        "The gap between actual pay and that fair value is how **over- or "
        "under-paid** a player is, in dollars."
    )
    st.markdown(
        f"- **{scored['season'].nunique()} seasons**, "
        f"**{len(scored):,} player-seasons**\n"
        f"- Validated by leave-one-season-out CV (**R² ≈ {CV_R2:.2f}**)\n"
        "- Salaries: ESPN cap hit · Stats: Basketball-Reference"
    )
    st.divider()
    st.subheader("📖 What the stats mean")
    st.caption("The less-common basketball terms used below.")
    for term in GLOSSARY_HIGHLIGHT:
        label, desc = STAT_GLOSSARY[term]
        st.markdown(f"**{term} — {label}**  \n<span style='color:#898781'>"
                    f"{desc}</span>", unsafe_allow_html=True)

# --- header ----------------------------------------------------------------
st.title("🏀 NBA Fair-Value Explorer")
st.caption(
    "Which NBA players are overpaid or underpaid — and by how many dollars? "
    "Pick a player to see their fair value, or scan the leaderboards below."
)

m1, m2, m3 = st.columns(3)
m1.metric("Seasons covered", f"{scored['season'].nunique()}")
m2.metric("Player-seasons", f"{len(scored):,}")
m3.metric("Model accuracy (R²)", f"{CV_R2:.2f}",
          help="Leave-one-season-out cross-validation: the model is scored on "
               "a season it never trained on. 1.0 is perfect; 0 is no better "
               "than guessing the average.")

st.divider()

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
    m1.metric("Actual salary", f"${row['Salary']/1e6:.1f}M",
              help="What the player is actually paid this season "
                   "(ESPN cap-hit figure).")
    m2.metric("Fair value", f"${row['fair_salary']/1e6:.1f}M",
              help="What the model predicts he'd earn based on on-court "
                   "performance alone.")
    m3.metric("Difference", f"${resid/1e6:+.1f}M",
              help="Actual minus fair value. Positive = paid above his "
                   "production; negative = below.")
    st.markdown(
        f"<h3 style='color:{color};margin-top:0'>{verdict}</h3>",
        unsafe_allow_html=True,
    )

    gap = abs(resid) / 1e6
    fair_m = row["fair_salary"] / 1e6
    paid_m = row["Salary"] / 1e6
    if verdict == "Overpaid":
        st.markdown(
            f"The model values **{player}**'s production at about "
            f"**${fair_m:.1f}M**, but he's paid **${paid_m:.1f}M** — roughly "
            f"**${gap:.1f}M above** fair value.")
    elif verdict == "Underpaid":
        st.markdown(
            f"The model values **{player}**'s production at about "
            f"**${fair_m:.1f}M**, yet he's paid only **${paid_m:.1f}M** — a "
            f"bargain of roughly **${gap:.1f}M**.")
    else:
        st.markdown(
            f"**{player}** is paid close to what the model thinks his "
            f"production is worth (**${paid_m:.1f}M** vs **${fair_m:.1f}M**).")

    contrib = linear_contributions(lin, feature_cols, row)
    contrib.index = [pretty_feature_name(c) for c in contrib.index]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    contribution_bars(contrib, ax=ax, top=8)
    ax.set_title(f"What raises or lowers {player}'s fair value")
    st.pyplot(fig, use_container_width=True)
    st.caption("Each bar is a stat pushing the fair-value estimate up (red) or "
               "down (blue) relative to a league-average player.")

with right:
    st.subheader(f"{season}: fair value vs actual pay")
    fig2, ax2 = plt.subplots(figsize=(6, 5.4))
    fair_vs_actual(view, highlight=player, ax=ax2)
    st.pyplot(fig2, use_container_width=True)
    st.caption("Every dot is a player. Above the dashed line = paid more than "
               "fair value (**red, overpaid**); below it = **blue, underpaid**. "
               "Your selected player is circled.")

st.divider()

# --- leaderboards ----------------------------------------------------------
LEADERBOARD_COLS = ["Player Name", "Team", "Position", "Salary", "fair_salary",
                    "residual_dollars", "WS", "PER"]

COLUMN_CONFIG = {
    "Player Name": st.column_config.TextColumn("Player"),
    "Salary": st.column_config.NumberColumn(
        "Paid", format="$%.1fM", help=stat_help("Salary") or
        "Actual salary this season."),
    "fair_salary": st.column_config.NumberColumn(
        "Fair value", format="$%.1fM",
        help="Model's fair salary from performance alone."),
    "residual_dollars": st.column_config.NumberColumn(
        "Diff", format="$%.1fM",
        help="Paid minus fair value. Positive = overpaid."),
    "WS": st.column_config.NumberColumn("WS", help=stat_help("WS")),
    "PER": st.column_config.NumberColumn("PER", format="%.1f",
                                         help=stat_help("PER")),
}


def _fmt(df: pd.DataFrame) -> pd.DataFrame:
    out = df[LEADERBOARD_COLS].copy()
    for c in ("Salary", "fair_salary", "residual_dollars"):
        out[c] = out[c] / 1e6
    return out


lc, rc = st.columns(2)
lc.subheader("💸 Most overpaid")
lc.dataframe(_fmt(view.sort_values("residual_dollars", ascending=False).head(12)),
             hide_index=True, use_container_width=True, column_config=COLUMN_CONFIG)
rc.subheader("💎 Most underpaid")
rc.dataframe(_fmt(view.sort_values("residual_dollars").head(12)),
             hide_index=True, use_container_width=True, column_config=COLUMN_CONFIG)

st.caption(
    "Salaries are ESPN season cap-hit figures; waived or bought-out players "
    "(dead-money contracts) can show anomalously low pay. The model is "
    "descriptive of how the market has paid for production — not a contract "
    "recommendation."
)
