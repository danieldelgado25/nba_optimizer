"""
fetcher.py
Pulls player stats from nba_api and the injury report from ESPN.
"""

import json
import time
import unicodedata
import urllib.request

import pandas as pd
from nba_api.stats.endpoints import (
    commonteamroster,
    leaguedashteamstats,
    playergamelog,
)
from nba_api.stats.static import teams as nba_teams_static

from nbaopt.config import current_season


def get_team_id(team_name: str) -> int | None:
    """
    Return NBA team ID by abbreviation, nickname, or full name (exact match),
    else by a substring of the full name if it matches exactly one team.
    """
    all_teams = nba_teams_static.get_teams()
    q = team_name.strip().lower()
    for t in all_teams:
        if q in (t["abbreviation"].lower(), t["nickname"].lower(), t["full_name"].lower()):
            return t["id"]
    matches = [t for t in all_teams if q in t["full_name"].lower()]
    return matches[0]["id"] if len(matches) == 1 else None


def get_team_roster(team_id: int, season: str | None = None) -> pd.DataFrame:
    """Return roster for a team."""
    time.sleep(0.6)
    roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season or current_season())
    df = roster.get_data_frames()[0]
    return df[["PLAYER_ID", "PLAYER", "NUM", "POSITION"]]


def get_player_game_log(player_id: int, last_n: int = 10, season: str | None = None) -> pd.DataFrame:
    """Return a player's last N games, ordered oldest -> newest."""
    time.sleep(0.6)
    log = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season or current_season(),
        season_type_all_star="Regular Season",
    )
    df = log.get_data_frames()[0]
    if df.empty:
        return df
    cols = [
        "GAME_DATE",
        "MATCHUP",
        "WL",
        "MIN",
        "PTS",
        "REB",
        "AST",
        "STL",
        "BLK",
        "TOV",
        "FG_PCT",
        "FG3_PCT",
        "FT_PCT",
        "PLUS_MINUS",
    ]
    # nba_api returns newest first: take the latest N, then flip to chronological
    df = df[cols].head(last_n).iloc[::-1].reset_index(drop=True)
    df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce").fillna(0)
    return df


def get_team_defensive_rating(season: str | None = None) -> pd.DataFrame:
    """Opponent points allowed per 100 possessions for every team."""
    time.sleep(0.6)
    stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season or current_season(),
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
    )
    df = stats.get_data_frames()[0]
    return df[["TEAM_ID", "TEAM_NAME", "DEF_RATING"]]


# Unofficial, undocumented ESPN endpoint (no key). Callers must survive it failing.
ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
INJURY_COLUMNS = ["PLAYER", "TEAM_ID", "STATUS", "COMMENT", "DATE"]


def get_injury_report(timeout: float = 10) -> pd.DataFrame:
    """
    League-wide injury report: one row per listed player.

    STATUS is ESPN's ("Out", "Day-To-Day", ...). PLAYER is ESPN's display name,
    which can lack the accents nba_api uses ("Doncic" vs "Dončić").
    """
    req = urllib.request.Request(ESPN_INJURIES_URL, headers={"User-Agent": "nbaopt"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    rows = [
        {
            "PLAYER": i["athlete"]["displayName"],
            # nickname, since ESPN says "LA Clippers" where nba_api says "Los Angeles Clippers"
            "TEAM_ID": get_team_id(team["displayName"].split()[-1]),
            "STATUS": i["status"],
            "COMMENT": i.get("shortComment", ""),
            "DATE": i.get("date", ""),
        }
        for team in data.get("injuries", [])
        for i in team.get("injuries", [])
    ]
    return pd.DataFrame(rows, columns=INJURY_COLUMNS).astype({"TEAM_ID": "Int64"})


def name_key(name: str) -> str:
    """Normalize a player name for matching across sources: 'Luka Dončić' == 'luka doncic'."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return " ".join("".join(c for c in ascii_name.lower() if c.isalnum() or c.isspace()).split())


def opponent_abbr(matchup: pd.Series) -> pd.Series:
    """'BOS vs. MIA' / 'BOS @ MIA' -> 'MIA'."""
    return matchup.str.split().str[-1]


def def_rating_by_abbr(def_df: pd.DataFrame) -> dict[str, float]:
    """Map team abbreviation -> DEF_RATING, for joining against MATCHUP strings."""
    id_to_abbr = {t["id"]: t["abbreviation"] for t in nba_teams_static.get_teams()}
    return {
        id_to_abbr[tid]: float(r)
        for tid, r in zip(def_df["TEAM_ID"], def_df["DEF_RATING"], strict=True)
        if tid in id_to_abbr
    }
