"""
fetcher.py
Pulls player stats and injury data from nba_api.
"""

from nba_api.stats.endpoints import (
    playergamelog,
    commonteamroster,
    leaguedashteamstats,
    leaguedashplayerstats,
    scoreboardv2,
)
from nba_api.stats.static import teams as nba_teams_static, players as nba_players_static
import pandas as pd
import time

CURRENT_SEASON = "2024-25"


def get_team_id(team_name: str) -> int | None:
    """Return NBA team ID by full name, city, or abbreviation."""
    all_teams = nba_teams_static.get_teams()
    team_name_lower = team_name.lower()
    for t in all_teams:
        if (
            team_name_lower in t["full_name"].lower()
            or team_name_lower == t["abbreviation"].lower()
            or team_name_lower in t["nickname"].lower()
        ):
            return t["id"]
    return None


def get_team_roster(team_id: int) -> pd.DataFrame:
    """Return current roster for a team."""
    time.sleep(0.6)
    roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=CURRENT_SEASON)
    df = roster.get_data_frames()[0]
    return df[["PLAYER_ID", "PLAYER", "NUM", "POSITION"]]


def get_player_game_log(player_id: int, last_n: int = 10) -> pd.DataFrame:
    """Return last N games for a player."""
    time.sleep(0.6)
    log = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=CURRENT_SEASON,
        season_type_all_star="Regular Season",
    )
    df = log.get_data_frames()[0]
    if df.empty:
        return df
    cols = ["GAME_DATE", "MATCHUP", "WL", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG_PCT", "FG3_PCT", "FT_PCT", "PLUS_MINUS"]
    df = df[cols].head(last_n)
    df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce").fillna(0)
    return df


def get_league_player_stats() -> pd.DataFrame:
    """Season averages for all players."""
    time.sleep(0.6)
    stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=CURRENT_SEASON,
        per_mode_simple="PerGame",
    )
    df = stats.get_data_frames()[0]
    return df


def get_team_defensive_rating() -> pd.DataFrame:
    """Opponent points allowed per 100 possessions for every team."""
    time.sleep(0.6)
    stats = leaguedashteamstats.LeagueDashTeamStats(
        season=CURRENT_SEASON,
        measure_type_simple="Advanced",
        per_mode_simple="PerGame",
    )
    df = stats.get_data_frames()[0]
    return df[["TEAM_ID", "TEAM_NAME", "DEF_RATING", "OPP_PTS_OFF_TOV", "OPP_PTS_2ND_CHANCE"]]


def get_todays_games() -> pd.DataFrame:
    """Return today's scheduled matchups."""
    time.sleep(0.6)
    board = scoreboardv2.ScoreboardV2()
    games = board.get_data_frames()[0]
    if games.empty:
        return pd.DataFrame()
    return games[["GAME_ID", "HOME_TEAM_ID", "VISITOR_TEAM_ID", "GAME_STATUS_TEXT"]]
