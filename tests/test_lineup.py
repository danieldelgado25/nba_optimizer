import pytest

from conftest import active, make_roster
from nbaopt.optimizer.lineup import (
    brute_force_best_five,
    build_player_pool,
    greedy_positional_lineup,
    optimize_lineup,
    parse_positions,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("G", {"PG", "SG"}),
        ("F", {"SF", "PF"}),
        ("C", {"C"}),
        ("G-F", {"PG", "SG", "SF", "PF"}),
        ("F-C", {"SF", "PF", "C"}),
        ("pg/sg", {"PG", "SG"}),
        ("", set()),
    ],
)
def test_parse_positions(raw, expected):
    assert set(parse_positions(raw)) == expected


ROSTER = make_roster(
    [
        ("Guard One", "G"),
        ("Guard Two", "G"),
        ("Wing One", "F"),
        ("Wing Two", "F"),
        ("Big One", "C"),
        ("Bench Guard", "G"),
        ("Bench Big", "C"),
    ]
)
SCORES = {
    "Guard One": active(40),
    "Guard Two": active(30),
    "Wing One": active(35),
    "Wing Two": active(25),
    "Big One": active(32),
    "Bench Guard": active(10),
    "Bench Big": active(8),
}


def test_build_player_pool_excludes_inactive():
    scores = {**SCORES, "Wing Two": {"predicted_score": 0.0, "status": "OUT", "avg_minutes": 0, "rolling_stats": {}}}
    pool = build_player_pool(ROSTER, scores)
    assert "Wing Two" not in set(pool["name"])
    assert len(pool) == 6


def test_greedy_fills_every_slot_by_position():
    lineup = greedy_positional_lineup(build_player_pool(ROSTER, SCORES))
    assert lineup is not None
    assert dict(zip(lineup["slot"], lineup["name"], strict=True)) == {
        "PG": "Guard One",
        "SG": "Guard Two",
        "SF": "Wing One",
        "PF": "Wing Two",
        "C": "Big One",
    }


def test_brute_force_picks_top_five_by_score():
    lineup = brute_force_best_five(build_player_pool(ROSTER, SCORES))
    assert lineup is not None
    assert set(lineup["name"]) == {"Guard One", "Guard Two", "Wing One", "Wing Two", "Big One"}


def test_optimize_lineup_reports_totals():
    result = optimize_lineup(ROSTER, SCORES)
    assert result["method"] == "positional_greedy"
    assert result["total_predicted_score"] == pytest.approx(162)
    assert result["pool_size"] == 7


def test_optimize_lineup_needs_five_active_players():
    scores = {name: active(10) for name in ["Guard One", "Guard Two", "Wing One", "Wing Two"]}
    result = optimize_lineup(ROSTER, scores)
    assert result["lineup"] is None
    assert "need at least 5" in result["error"]


@pytest.mark.xfail(strict=True, reason="Known bug: greedy slot filling is not optimal (fixed in P5)")
def test_lineup_is_optimal_under_position_constraints():
    roster = make_roster(
        [
            ("Combo Guard", "PG-SG"),
            ("Pure Point", "PG"),
            ("Pure Shooter", "SG"),
            ("Wing", "SF"),
            ("Forward", "PF"),
            ("Center", "C"),
        ]
    )
    scores = {
        "Combo Guard": active(30),
        "Pure Point": active(25),
        "Pure Shooter": active(5),
        "Wing": active(20),
        "Forward": active(20),
        "Center": active(20),
    }
    result = optimize_lineup(roster, scores)
    # Best legal lineup: Pure Point at PG, Combo Guard at SG.
    assert result["total_predicted_score"] == pytest.approx(115)
