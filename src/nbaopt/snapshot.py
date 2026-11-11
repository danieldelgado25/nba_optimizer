"""
snapshot.py
Fetches current-season rosters, recent game logs, and defensive ratings for
all 30 teams into data/snapshot/, which the Streamlit app reads.

stats.nba.com blocks most cloud IPs, so run this locally and commit the
result to refresh the deployed app.

Usage:
    nbaopt snapshot
"""

from datetime import UTC, datetime

import pandas as pd
from nba_api.stats.static import teams as nba_teams_static

from nbaopt.config import current_season, snapshot_dir
from nbaopt.data.fetcher import (
    get_injury_report,
    get_player_game_log,
    get_team_defensive_rating,
    get_team_roster,
)
from nbaopt.models.predictor import SEQ_LEN


def main() -> None:
    out = snapshot_dir()
    out.mkdir(parents=True, exist_ok=True)
    print(f"Snapshotting {current_season()}...")

    get_team_defensive_rating().to_csv(out / "def_ratings.csv", index=False)
    get_injury_report().to_csv(out / "injuries.csv", index=False)  # fallback when the app can't reach ESPN

    rosters, logs = [], []
    for t_i, team in enumerate(nba_teams_static.get_teams(), 1):
        roster = get_team_roster(team["id"])
        roster["TEAM_ID"] = team["id"]
        rosters.append(roster)

        for p_i, (_, player) in enumerate(roster.iterrows(), 1):
            print(
                f"  [{t_i:>2}/30] {team['abbreviation']} [{p_i:>2}/{len(roster)}] {player['PLAYER']:<28}",
                end="\r",
            )
            try:
                log = get_player_game_log(player["PLAYER_ID"], last_n=SEQ_LEN)
            except Exception as e:
                print(f"\n  Skipped {player['PLAYER']}: {e}")
                continue
            if not log.empty:
                log["PLAYER_ID"] = player["PLAYER_ID"]
                logs.append(log)

    pd.concat(rosters).to_csv(out / "rosters.csv", index=False)
    pd.concat(logs).to_csv(out / "game_logs.csv", index=False)
    (out / "fetched_at.txt").write_text(datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"))
    print(f"\nDone. {len(logs)} player logs → {out}")


if __name__ == "__main__":
    main()
