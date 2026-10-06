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


def test_get_player_game_log_trims_and_coerces(monkeypatch, fake):
    raw = make_game_log(15)
    raw["MIN"] = raw["MIN"].astype(str)
    raw.loc[0, "MIN"] = "bad"
    raw["EXTRA"] = 1
    fake.frame = raw
    monkeypatch.setattr(fetcher.playergamelog, "PlayerGameLog", fake)
    monkeypatch.setenv("NBAOPT_SEASON", "2025-26")

    log = fetcher.get_player_game_log(42, last_n=10)

    assert len(log) == 10
    assert "EXTRA" not in log.columns
    assert log["MIN"].iloc[0] == 0
    assert pd.api.types.is_numeric_dtype(log["MIN"])
    assert fake.calls[0]["player_id"] == 42
    assert fake.calls[0]["season"] == "2025-26"


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
