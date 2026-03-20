"""
injury_impact.py
Models the impact of removing a player from the lineup.

Calculates:
  - Team projected score WITH player
  - Team projected score WITHOUT player (redistributing usage)
  - Impact delta and impact percentage
  - Which players absorb the usage
"""

import pandas as pd
from typing import Optional


def estimate_usage_redistribution(
    player_name: str,
    roster_scores: dict[str, dict],
    injured_player_score: float,
    injured_player_minutes: float,
) -> dict[str, float]:
    """
    When a player is out, distribute their projected score contribution
    across remaining active players proportional to their own scores.

    Returns a dict of player_name → score_boost.
    """
    active = {
        name: data
        for name, data in roster_scores.items()
        if name != player_name and data["status"] == "ACTIVE"
    }

    total_active_score = sum(d["predicted_score"] for d in active.values())
    if total_active_score == 0:
        return {}

    redistribution = {}
    for name, data in active.items():
        share = data["predicted_score"] / total_active_score
        redistribution[name] = round(share * injured_player_score * 0.7, 2)  # 70% recovery assumption

    return redistribution


def compute_injury_impact(
    injured_player_name: str,
    roster_scores: dict[str, dict],
) -> dict:
    """
    Full injury impact report for one player being ruled out.

    Args:
        injured_player_name: Name of player who is OUT
        roster_scores: Dict of player_name → predict_player_score() output

    Returns:
        Impact report dict
    """
    if injured_player_name not in roster_scores:
        return {"error": f"{injured_player_name} not found in roster scores."}

    injured_data = roster_scores[injured_player_name]
    lost_score = injured_data.get("predicted_score", 0)
    lost_minutes = injured_data.get("avg_minutes", 0)

    # Team total WITH player
    team_score_with = sum(d["predicted_score"] for d in roster_scores.values())

    # Team total WITHOUT player (with redistribution)
    redistribution = estimate_usage_redistribution(
        injured_player_name,
        roster_scores,
        lost_score,
        lost_minutes,
    )

    team_score_without = team_score_with - lost_score + sum(redistribution.values())

    delta = round(team_score_with - team_score_without, 2)
    impact_pct = round((delta / team_score_with) * 100, 1) if team_score_with > 0 else 0

    # Classify impact level
    if impact_pct >= 12:
        impact_level = "🔴 CRITICAL"
    elif impact_pct >= 7:
        impact_level = "🟠 HIGH"
    elif impact_pct >= 3:
        impact_level = "🟡 MODERATE"
    else:
        impact_level = "🟢 LOW"

    top_absorbers = sorted(redistribution.items(), key=lambda x: x[1], reverse=True)[:3]

    return {
        "injured_player": injured_player_name,
        "player_projected_score": lost_score,
        "player_avg_minutes": lost_minutes,
        "team_score_with": round(team_score_with, 2),
        "team_score_without": round(team_score_without, 2),
        "score_delta": delta,
        "impact_percentage": impact_pct,
        "impact_level": impact_level,
        "usage_absorbed_by": top_absorbers,
        "redistribution_detail": redistribution,
    }


def batch_injury_report(roster_scores: dict[str, dict]) -> list[dict]:
    """
    Run injury impact analysis for every player on the roster.
    Useful for showing 'what if X goes down?' for the full team.
    Returns sorted list by impact percentage descending.
    """
    reports = []
    for player_name in roster_scores:
        if roster_scores[player_name]["status"] == "ACTIVE":
            report = compute_injury_impact(player_name, roster_scores)
            reports.append(report)

    return sorted(reports, key=lambda x: x.get("impact_percentage", 0), reverse=True)
