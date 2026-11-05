import pytest

from conftest import active, make_roster
from nbaopt.optimizer.lineup import (
    STANDARD_POSITIONS,
    best_positional_lineup,
    brute_force_best_five,
    build_player_pool,
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
        (float("nan"), set()),  # missing POSITION in a CSV-loaded roster
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


def assert_legal(lineup):
    assert list(lineup["slot"]) == STANDARD_POSITIONS
    for _, row in lineup.iterrows():
        assert row["slot"] in row["positions"]


def test_build_player_pool_excludes_inactive():
    scores = {**SCORES, "Wing Two": {"predicted_score": 0.0, "status": "OUT", "avg_minutes": 0, "rolling_stats": {}}}
    pool = build_player_pool(ROSTER, scores)
    assert "Wing Two" not in set(pool["name"])
    assert len(pool) == 6


def test_positional_lineup_fills_every_slot():
    lineup = best_positional_lineup(build_player_pool(ROSTER, SCORES))
    assert lineup is not None
    assert_legal(lineup)
    assert set(lineup["name"]) == {"Guard One", "Guard Two", "Wing One", "Wing Two", "Big One"}


def test_positional_lineup_none_when_no_legal_five():
    roster = make_roster([(f"Guard {i}", "G") for i in range(6)])
    assert best_positional_lineup(build_player_pool(roster, {f"Guard {i}": active(10) for i in range(6)})) is None


def test_brute_force_picks_top_five_by_score():
    lineup = brute_force_best_five(build_player_pool(ROSTER, SCORES))
    assert lineup is not None
    assert set(lineup["name"]) == {"Guard One", "Guard Two", "Wing One", "Wing Two", "Big One"}


def test_optimize_lineup_reports_totals():
    result = optimize_lineup(ROSTER, SCORES)
    assert result["method"] == "positional_optimal"
    assert result["total_predicted_score"] == pytest.approx(162)
    assert result["pool_size"] == 7


def test_optimize_lineup_falls_back_without_a_legal_five():
    roster = make_roster([(f"Guard {i}", "G") for i in range(6)])
    result = optimize_lineup(roster, {f"Guard {i}": active(10 + i) for i in range(6)})
    assert result["method"] == "brute_force_top5"
    assert result["total_predicted_score"] == pytest.approx(11 + 12 + 13 + 14 + 15)


def test_optimize_lineup_needs_five_active_players():
    scores = {name: active(10) for name in ["Guard One", "Guard Two", "Wing One", "Wing Two"]}
    result = optimize_lineup(ROSTER, scores)
    assert result["lineup"] is None
    assert "need at least 5" in result["error"]


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
    assert_legal(result["lineup"])


# The 2026-27 Celtics case the old greedy fill got wrong (benched a 23-pt guard)
CELTICS = [
    ("Tatum", "F-G", 44.57),
    ("White", "G", 23.24),
    ("George", "F", 38.92),
    ("Robinson", "C-F", 13.75),
    ("Queta", "C", 21.21),
    ("Pritchard", "G", 23.08),
    ("Hauser", "F", 12.58),
]


def test_celtics_regression():
    roster = make_roster([(n, p) for n, p, _ in CELTICS])
    result = optimize_lineup(roster, {n: active(s) for n, _, s in CELTICS})
    assert result["method"] == "positional_optimal"
    assert set(result["lineup"]["name"]) == {"Tatum", "White", "George", "Queta", "Pritchard"}
    assert result["total_predicted_score"] == pytest.approx(151.02)
    assert_legal(result["lineup"])
