import pandas as pd
import pytest

from conftest import make_game_log
from nbaopt.data import fetcher


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(fetcher.time, "sleep", lambda _s: None)


class FakeEndpoint:
    calls: list[dict] = []
    frame = pd.DataFrame()

    def __init__(self, **kwargs):
        FakeEndpoint.calls.append(kwargs)

    def get_data_frames(self):
        return [FakeEndpoint.frame]


@pytest.fixture
def fake(monkeypatch):
    FakeEndpoint.calls = []
    return FakeEndpoint


@pytest.mark.parametrize("name", ["Boston Celtics", "celtics", "BOS", "boston"])
def test_get_team_id_matches_name_nickname_and_abbreviation(name):
    assert fetcher.get_team_id(name) == 1610612738


def test_get_team_id_unknown():
    assert fetcher.get_team_id("Seattle Supersonics") is None


def test_get_team_id_prefers_exact_nickname_over_substring():
    # "nets" is also a substring of "Charlotte Hornets"
    assert fetcher.get_team_id("Nets") == fetcher.get_team_id("BKN")
    assert fetcher.get_team_id("hornets") == fetcher.get_team_id("CHA")
    assert fetcher.get_team_id("BKN") != fetcher.get_team_id("CHA")


def test_get_team_id_ambiguous_substring():
    assert fetcher.get_team_id("Los Angeles") is None  # Lakers or Clippers


def test_get_player_game_log_trims_and_coerces(monkeypatch, fake):
    raw = make_game_log(15)  # newest first, like nba_api
    raw["MIN"] = raw["MIN"].astype(str)
    raw.loc[0, "MIN"] = "bad"
    raw["EXTRA"] = 1
    fake.frame = raw
    monkeypatch.setattr(fetcher.playergamelog, "PlayerGameLog", fake)
    monkeypatch.setenv("NBAOPT_SEASON", "2025-26")

    log = fetcher.get_player_game_log(42, last_n=10)

    assert len(log) == 10
    assert "EXTRA" not in log.columns
    assert log["MIN"].iloc[-1] == 0  # the newest game, now last
    assert pd.api.types.is_numeric_dtype(log["MIN"])
    assert fake.calls[0]["player_id"] == 42
    assert fake.calls[0]["season"] == "2025-26"


def test_get_player_game_log_keeps_latest_games_oldest_first(monkeypatch, fake):
    raw = make_game_log(15)
    fake.frame = raw
    monkeypatch.setattr(fetcher.playergamelog, "PlayerGameLog", fake)

    log = fetcher.get_player_game_log(42, last_n=10)

    dates = pd.to_datetime(log["GAME_DATE"], format="%b %d, %Y")
    assert dates.is_monotonic_increasing
    assert log["GAME_DATE"].iloc[-1] == raw["GAME_DATE"].iloc[0]  # newest game kept
    assert log["GAME_DATE"].iloc[0] == raw["GAME_DATE"].iloc[9]  # 5 oldest dropped


def test_opponent_abbr():
    assert fetcher.opponent_abbr(pd.Series(["BOS vs. MIA", "BOS @ LAL"])).tolist() == ["MIA", "LAL"]


def test_def_rating_by_abbr():
    df = pd.DataFrame({"TEAM_ID": [1610612738, 999], "DEF_RATING": [110.5, 100.0]})
    assert fetcher.def_rating_by_abbr(df) == {"BOS": 110.5}


def test_get_player_game_log_empty(monkeypatch, fake):
    fake.frame = pd.DataFrame()
    monkeypatch.setattr(fetcher.playergamelog, "PlayerGameLog", fake)
    assert fetcher.get_player_game_log(1).empty


def test_explicit_season_overrides_default(monkeypatch, fake):
    fake.frame = pd.DataFrame([{"PLAYER_ID": 1, "PLAYER": "A", "NUM": "0", "POSITION": "G", "OTHER": 1}])
    monkeypatch.setattr(fetcher.commonteamroster, "CommonTeamRoster", fake)
    roster = fetcher.get_team_roster(1610612738, season="2019-20")
    assert list(roster.columns) == ["PLAYER_ID", "PLAYER", "NUM", "POSITION"]
    assert fake.calls[0]["season"] == "2019-20"


def test_get_injury_report_parses_espn(monkeypatch):
    import io
    import json

    payload = {
        "injuries": [
            {
                "displayName": "LA Clippers",  # ESPN's name; nba_api says "Los Angeles Clippers"
                "injuries": [
                    {
                        "athlete": {"displayName": "Bradley Beal"},
                        "status": "Out",
                        "shortComment": "Hip.",
                        "date": "2026-11-01T00:00Z",
                    }
                ],
            },
            {
                "displayName": "Minnesota Timberwolves",
                "injuries": [{"athlete": {"displayName": "Anthony Edwards"}, "status": "Day-To-Day"}],
            },
        ]
    }
    monkeypatch.setattr(
        fetcher.urllib.request, "urlopen", lambda req, timeout: io.BytesIO(json.dumps(payload).encode())
    )

    df = fetcher.get_injury_report()

    assert df.to_dict("records") == [
        {
            "PLAYER": "Bradley Beal",
            "TEAM_ID": 1610612746,
            "STATUS": "Out",
            "COMMENT": "Hip.",
            "DATE": "2026-11-01T00:00Z",
        },
        {"PLAYER": "Anthony Edwards", "TEAM_ID": 1610612750, "STATUS": "Day-To-Day", "COMMENT": "", "DATE": ""},
    ]


def test_get_injury_report_empty(monkeypatch):
    import io

    monkeypatch.setattr(fetcher.urllib.request, "urlopen", lambda req, timeout: io.BytesIO(b"{}"))
    df = fetcher.get_injury_report()
    assert df.empty
    assert list(df.columns) == fetcher.INJURY_COLUMNS


@pytest.mark.parametrize(
    ("a", "b"),
    [("Luka Dončić", "Luka Doncic"), ("Ernest Udeh, Jr.", "Ernest Udeh Jr"), ("Nae'Qwan Tomlin", "naeqwan  tomlin")],
)
def test_name_key_matches_across_sources(a, b):
    assert fetcher.name_key(a) == fetcher.name_key(b)
