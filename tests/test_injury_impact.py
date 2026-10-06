import pytest

from conftest import active
from nbaopt.models.injury_impact import batch_injury_report, compute_injury_impact, estimate_usage_redistribution

ROSTER_SCORES = {
    "Star": active(50, 36),
    "Starter": active(30, 30),
    "Role": active(20, 24),
    "Out Guy": {"predicted_score": 0.0, "status": "OUT", "avg_minutes": 0, "rolling_stats": {}},
}


def test_redistribution_is_proportional_and_recovers_70_percent():
    boosts = estimate_usage_redistribution("Star", ROSTER_SCORES, 50, 36)
    assert set(boosts) == {"Starter", "Role"}
    assert boosts["Starter"] == pytest.approx(50 * 0.7 * 30 / 50)
    assert boosts["Role"] == pytest.approx(50 * 0.7 * 20 / 50)
    assert sum(boosts.values()) == pytest.approx(35)


def test_redistribution_with_no_active_teammates():
    assert estimate_usage_redistribution("Star", {"Star": active(50)}, 50, 36) == {}


def test_compute_injury_impact():
    report = compute_injury_impact("Star", ROSTER_SCORES)
    assert report["team_score_with"] == pytest.approx(100)
    assert report["team_score_without"] == pytest.approx(85)
    assert report["score_delta"] == pytest.approx(15)
    assert report["impact_percentage"] == pytest.approx(15.0)
    assert "CRITICAL" in report["impact_level"]
    assert report["usage_absorbed_by"][0][0] == "Starter"


@pytest.mark.parametrize(
    ("role_score", "level"),
    [(1, "LOW"), (12, "MODERATE"), (40, "HIGH")],
)
def test_impact_levels(role_score, level):
    scores = {"Star": active(100), "Role": active(role_score)}
    report = compute_injury_impact("Role", scores)
    assert level in report["impact_level"]


def test_unknown_player_returns_error():
    assert "error" in compute_injury_impact("Nobody", ROSTER_SCORES)


def test_batch_report_skips_inactive_and_sorts_by_impact():
    reports = batch_injury_report(ROSTER_SCORES)
    assert [r["injured_player"] for r in reports] == ["Star", "Starter", "Role"]
