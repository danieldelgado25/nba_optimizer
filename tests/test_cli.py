import pandas as pd
import pytest

from conftest import make_game_log, make_roster
from nbaopt import cli

ROSTER = make_roster(
    [
        ("Guard One", "G"),
        ("Guard Two", "G"),
        ("Wing One", "F"),
        ("Wing Two", "F"),
        ("Big One", "C"),
        ("Bench Guard", "G"),
    ]
)


@pytest.fixture
def offline(monkeypatch, no_model):
    """Replace every network call the CLI makes with synthetic data."""
    captured: dict[str, list] = {"logs": []}

    def game_log(pid, last_n=10, season=None):
        log = make_game_log(last_n, pts=10 + (pid - 1000) * 3, seed=pid)
        return log

    def predict(log, opp_def_rating=113.5, injured=False):
        captured["logs"].append(log)
        return real_predict(log, opp_def_rating, injured)

    real_predict = cli.predict_player_score
    monkeypatch.setattr(cli, "get_team_roster", lambda team_id, season=None: ROSTER)
    monkeypatch.setattr(cli, "get_player_game_log", game_log)
    monkeypatch.setattr(cli, "predict_player_score", predict)
    monkeypatch.setattr(
        cli,
        "get_team_defensive_rating",
        lambda season=None: pd.DataFrame([{"TEAM_ID": 1610612744, "DEF_RATING": 110.0}]),
    )
    return captured


def test_parser_subcommands():
    parser = cli.build_parser()
    args = parser.parse_args(["lineup", "--team", "BOS", "--impact-report", "--season", "2024-25"])
    assert (args.command, args.team, args.impact_report, args.season) == ("lineup", "BOS", True, "2024-25")
    args = parser.parse_args(["train", "--team", "BOS", "--team", "LAL", "--epochs", "3"])
    assert (args.command, args.teams, args.epochs) == ("train", ["BOS", "LAL"], 3)


def test_lineup_end_to_end(offline, capsys):
    cli.main(["lineup", "--team", "Celtics", "--opponent", "Warriors", "--injured", "Bench Guard", "--impact-report"])
    out = capsys.readouterr().out
    assert "OPTIMAL LINEUP" in out
    assert "DEF_RATING: 110.0" in out
    assert "FULL INJURY IMPACT REPORT" in out
    assert "EXCLUDED (OUT/NO DATA): Bench Guard" in out


def test_lineup_main_alias(offline, capsys):
    cli.lineup_main(["--team", "BOS", "--injury-impact", "Big One"])
    assert "INJURY IMPACT: Big One" in capsys.readouterr().out


def test_unknown_team_exits(offline):
    with pytest.raises(SystemExit):
        cli.main(["lineup", "--team", "Nowhere FC"])


@pytest.mark.xfail(strict=True, reason="Known bug: game logs reach the predictor newest-first (fixed in P2)")
def test_predictor_receives_logs_oldest_first(offline):
    cli.main(["lineup", "--team", "BOS"])
    dates = pd.to_datetime(offline["logs"][0]["GAME_DATE"], format="%b %d, %Y")
    assert dates.is_monotonic_increasing
