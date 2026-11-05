import pytest

from conftest import active, make_roster
from nbaopt.models.injury_impact import batch_injury_report, compute_injury_impact

CELTICS = [
    ("Tatum", "F-G", 44.57),
    ("White", "G", 23.24),
    ("George", "F", 38.92),
    ("Robinson", "C-F", 13.75),
    ("Queta", "C", 21.21),
    ("Pritchard", "G", 23.08),
    ("Hauser", "F", 12.58),
]
ROSTER = make_roster([(n, p) for n, p, _ in CELTICS])
SCORES = {n: active(s) for n, _, s in CELTICS}


def test_impact_is_drop_in_reoptimized_lineup():
    report = compute_injury_impact("Tatum", ROSTER, SCORES)
    assert report["team_score_with"] == pytest.approx(151.02)
    assert report["score_delta"] == pytest.approx(44.57 - 13.75)
    assert report["team_score_without"] == pytest.approx(151.02 - (44.57 - 13.75))
    assert report["replaced_by"] == ["Robinson"]
    assert report["impact_percentage"] == pytest.approx(round(100 * (44.57 - 13.75) / 151.02, 1))
    assert "CRITICAL" in report["impact_level"]


def test_bench_player_has_zero_impact():
    report = compute_injury_impact("Hauser", ROSTER, SCORES)
    assert report["score_delta"] == 0
    assert report["replaced_by"] == []
    assert "LOW" in report["impact_level"]


@pytest.mark.parametrize("name", list(SCORES))
def test_impact_is_never_negative(name):
    assert compute_injury_impact(name, ROSTER, SCORES)["score_delta"] >= 0


@pytest.mark.parametrize(
    ("sub_score", "level"),
    [(19, "LOW"), (16, "MODERATE"), (12, "HIGH"), (5, "CRITICAL")],
)
def test_impact_levels(sub_score, level):
    # Five 20-point starters (total 100) and one backup center: losing the
    # starting center costs exactly 20 - sub_score, i.e. that many percent.
    roster = make_roster([("G1", "G"), ("G2", "G"), ("F1", "F"), ("F2", "F"), ("C1", "C"), ("Sub", "C")])
    scores = {n: active(20) for n in ["G1", "G2", "F1", "F2", "C1"]} | {"Sub": active(sub_score)}
    report = compute_injury_impact("C1", roster, scores)
    assert report["impact_percentage"] == pytest.approx(20 - sub_score)
    assert level in report["impact_level"]


def test_unknown_player_returns_error():
    assert "error" in compute_injury_impact("Nobody", ROSTER, SCORES)


def test_no_replacement_available_returns_error():
    roster = make_roster([("G1", "G"), ("G2", "G"), ("F1", "F"), ("F2", "F"), ("C1", "C")])
    scores = {n: active(20) for n in roster["PLAYER"]}
    assert "error" in compute_injury_impact("C1", roster, scores)


def test_batch_report_skips_inactive_and_sorts_by_impact():
    scores = {**SCORES, "Queta": {"predicted_score": 0.0, "status": "OUT", "avg_minutes": 0, "rolling_stats": {}}}
    reports = batch_injury_report(ROSTER, scores)
    names = [r["injured_player"] for r in reports]
    assert "Queta" not in names
    assert len(names) == len(SCORES) - 1
    pcts = [r["impact_percentage"] for r in reports]
    assert pcts == sorted(pcts, reverse=True)
    assert names[0] == "Tatum"
