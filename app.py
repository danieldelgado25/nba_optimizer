"""
app.py
Streamlit front end. Reads the committed data/snapshot/ (see `nbaopt snapshot`)
instead of calling stats.nba.com, which blocks most cloud hosts.

Run locally:
    streamlit run app.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from nba_api.stats.static import teams as nba_teams_static  # noqa: E402

from nbaopt.config import metrics_path, model_path, snapshot_dir  # noqa: E402
from nbaopt.models.injury_impact import batch_injury_report  # noqa: E402
from nbaopt.models.predictor import LEAGUE_AVG_DEF_RATING, predict_player_score  # noqa: E402
from nbaopt.optimizer.lineup import optimize_lineup  # noqa: E402

st.set_page_config(page_title="NBA Lineup Optimizer", page_icon="🏀", layout="wide")


@st.cache_data
def load_snapshot():
    def read(name):
        return pd.read_csv(snapshot_dir() / name)

    rosters = read("rosters.csv")
    rosters["POSITION"] = rosters["POSITION"].fillna("")
    fetched_at = (snapshot_dir() / "fetched_at.txt").read_text().strip()
    return rosters, read("game_logs.csv"), read("def_ratings.csv"), fetched_at


@st.cache_data
def analyze(team_id: int, opp_rating: float, injured: tuple[str, ...]):
    roster = rosters[rosters["TEAM_ID"] == team_id]
    scores = {
        p["PLAYER"]: predict_player_score(
            logs[logs["PLAYER_ID"] == p["PLAYER_ID"]], opp_rating, injured=p["PLAYER"] in injured
        )
        for _, p in roster.iterrows()
    }
    return scores, optimize_lineup(roster, scores), batch_injury_report(roster, scores)


rosters, logs, def_ratings, fetched_at = load_snapshot()
team_ids = {t["full_name"]: t["id"] for t in sorted(nba_teams_static.get_teams(), key=lambda t: t["full_name"])}

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    team = st.selectbox("Team", list(team_ids), index=list(team_ids).index("Boston Celtics"))
    opponent = st.selectbox("Opponent", ["League average"] + [n for n in team_ids if n != team])
    roster_names = rosters.loc[rosters["TEAM_ID"] == team_ids[team], "PLAYER"].tolist()
    injured = st.multiselect("Ruled out", roster_names)
    st.caption(f"Data snapshot: {fetched_at}")

opp_rating = LEAGUE_AVG_DEF_RATING
if opponent != "League average":
    match = def_ratings.loc[def_ratings["TEAM_ID"] == team_ids[opponent], "DEF_RATING"]
    if not match.empty:
        opp_rating = float(match.iloc[0])

scores, result, impact = analyze(team_ids[team], opp_rating, tuple(injured))

# ── Lineup ────────────────────────────────────────────────────────────────────

st.title("🏀 NBA Lineup Optimizer")
st.caption(
    "Next-game fantasy projections from an LSTM over each player's last 10 games, "
    "adjusted for the opponent's defense, then the best position-valid starting five."
)

if result["lineup"] is None:
    st.error(result.get("error", "Could not build a lineup."))
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Projected lineup score", f"{result['total_predicted_score']:.1f}")
c2.metric("Opponent DEF rating", f"{opp_rating:.1f}")
c3.metric("Active players", result["pool_size"])

lineup = result["lineup"].assign(model=lambda d: d["name"].map(lambda n: scores[n]["model_used"]))
st.subheader("Optimal lineup")
st.dataframe(
    lineup[["slot", "name", "position", "predicted_score", "avg_minutes", "model"]],
    hide_index=True,
    width="stretch",
    column_config={
        "slot": "Slot",
        "name": "Player",
        "position": "Pos",
        "predicted_score": st.column_config.NumberColumn("Projected", format="%.1f"),
        "avg_minutes": st.column_config.NumberColumn("Avg min", format="%.1f"),
        "model": "Model",
    },
)
if result["excluded_players"]:
    st.caption(f"Out / no data: {', '.join(result['excluded_players'])}")

# ── Injury impact ─────────────────────────────────────────────────────────────

st.subheader("Injury impact")
st.caption(
    "How much the optimal lineup's projected total drops if each player is ruled out "
    "and the lineup is re-optimized from the bench."
)
st.dataframe(
    pd.DataFrame(
        [
            {
                "Player": r["injured_player"],
                "Projected": r["player_projected_score"],
                "Lineup drop": r["score_delta"],
                "Impact %": r["impact_percentage"],
                "Level": r["impact_level"],
                "Replaced by": ", ".join(r["replaced_by"]) or "—",
            }
            for r in impact
        ]
    ),
    hide_index=True,
    width="stretch",
    column_config={
        "Projected": st.column_config.NumberColumn(format="%.1f"),
        "Lineup drop": st.column_config.NumberColumn(format="%.1f"),
        "Impact %": st.column_config.NumberColumn(format="%.1f%%"),
    },
)

# ── Model ─────────────────────────────────────────────────────────────────────

metrics_file = metrics_path(model_path())
if metrics_file.exists():
    m = json.loads(metrics_file.read_text())
    with st.expander("Model evaluation"):
        gain = 1 - m["lstm"]["mae"] / m["baseline"]["mae"]
        st.write(
            f"On {m['val_games']:,} held-out games (the last 20% of each player's season), "
            f"the LSTM's mean absolute error is **{m['lstm']['mae']:.2f}** fantasy points vs "
            f"**{m['baseline']['mae']:.2f}** for a 10-game rolling average ({gain:+.1%}). "
            "Players with fewer than 10 games this season use the rolling average."
        )
