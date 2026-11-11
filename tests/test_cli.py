import pandas as pd
import pytest

from conftest import make_game_log, make_roster
from nbaopt import cli
from nbaopt.data import fetcher

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

    def game_log(pid, last_n=10, season=None):
        return make_game_log(last_n, pts=10 + (pid - 1000) * 3, newest_first=False, seed=pid)

    monkeypatch.setattr(cli, "get_team_roster", lambda team_id, season=None: ROSTER)
    monkeypatch.setattr(cli, "get_player_game_log", game_log)
    monkeypatch.setattr(
        cli,
        "get_team_defensive_rating",
        lambda season=None: pd.DataFrame([{"TEAM_ID": 1610612744, "DEF_RATING": 110.0}]),
    )
    captured: dict[str, list] = {"injuries": []}
    monkeypatch.setattr(
        cli, "get_injury_report", lambda: pd.DataFrame(captured["injuries"], columns=fetcher.INJURY_COLUMNS)
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


def test_injured_accepts_unique_partial_names(offline, capsys):
    cli.main(["lineup", "--team", "BOS", "--injured", "bench guard, big one"])
    assert "Marked as OUT: Bench Guard, Big One" in capsys.readouterr().out


@pytest.mark.parametrize("flag", ["--injured", "--injury-impact"])
def test_ambiguous_player_name_exits(offline, capsys, flag):
    with pytest.raises(SystemExit):
        cli.main(["lineup", "--team", "BOS", flag, "Guard"])  # Guard One, Guard Two, Bench Guard
    assert "matches no player or several" in capsys.readouterr().out


def test_resolve_player():
    names = ["Jaylen Brown", "Bruce Brown", "Jayson Tatum"]
    assert cli.resolve_player("Brown", names) is None
    assert cli.resolve_player("jaylen brown", names) == "Jaylen Brown"
    assert cli.resolve_player("Tatum", names) == "Jayson Tatum"
    assert cli.resolve_player("Nobody", names) is None


def injury(name, status):
    return {"PLAYER": name, "TEAM_ID": 0, "STATUS": status, "COMMENT": "", "DATE": ""}


def test_injury_report_marks_out_players(offline, capsys):
    offline["injuries"] = [injury("Big One", "Out"), injury("Wing One", "Day-To-Day"), injury("Elsewhere", "Out")]
    cli.main(["lineup", "--team", "BOS"])
    out = capsys.readouterr().out
    assert "Marked as OUT: Big One" in out
    assert "Wing One — Day-To-Day (kept active" in out
    assert "Elsewhere" not in out
    assert "EXCLUDED (OUT/NO DATA): Big One" in out


def test_injury_report_skipped_when_disabled_or_past_season(offline, capsys, monkeypatch):
    monkeypatch.setenv("NBAOPT_SEASON", "2026-27")
    offline["injuries"] = [injury("Big One", "Out")]
    cli.main(["lineup", "--team", "BOS", "--no-injury-report"])
    cli.main(["lineup", "--team", "BOS", "--season", "2024-25"])
    assert "Marked as OUT" not in capsys.readouterr().out


def test_injury_report_failure_is_not_fatal(offline, capsys, monkeypatch):
    def boom():
        raise OSError("blocked")

    monkeypatch.setattr(cli, "get_injury_report", boom)
    cli.main(["lineup", "--team", "BOS"])
    out = capsys.readouterr().out
    assert "Could not fetch the injury report" in out
    assert "OPTIMAL LINEUP" in out
