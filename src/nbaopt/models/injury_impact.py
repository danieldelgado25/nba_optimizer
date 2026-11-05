"""
injury_impact.py
Models the impact of removing a player from the lineup.

Calculates:
  - Optimal lineup projected score WITH player
  - Optimal lineup projected score WITHOUT player (re-optimized from the bench)
  - Impact delta and impact percentage
  - Which players enter the lineup to replace them
"""

import pandas as pd

from nbaopt.optimizer.lineup import optimize_lineup


def compute_injury_impact(
    injured_player_name: str,
    roster_df: pd.DataFrame,
    roster_scores: dict[str, dict],
) -> dict:
    """
    Full injury impact report for one player being ruled out.

    Impact = how much the optimal lineup's projected total drops when the
    player is OUT and the lineup is re-optimized. The replacement absorbs the
    lost minutes, so the delta is never negative, and a bench player who
    doesn't make the optimal lineup has zero impact.

    Args:
        injured_player_name: Name of player who is OUT
        roster_df: Roster from get_team_roster() (for positions)
        roster_scores: Dict of player_name → predict_player_score() output

    Returns:
        Impact report dict
    """
    if injured_player_name not in roster_scores:
        return {"error": f"{injured_player_name} not found in roster scores."}

    injured_data = roster_scores[injured_player_name]
    lost_score = injured_data.get("predicted_score", 0)
    lost_minutes = injured_data.get("avg_minutes", 0)

    with_result = optimize_lineup(roster_df, roster_scores)
    if with_result["lineup"] is None:
        return {"error": "Not enough active players to build a lineup."}

    scores_without = {**roster_scores, injured_player_name: {**injured_data, "status": "OUT", "predicted_score": 0.0}}
    without_result = optimize_lineup(roster_df, scores_without)
    if without_result["lineup"] is None:
        return {"error": "Not enough active players to build a lineup without this player."}

    team_score_with = with_result["total_predicted_score"]
    team_score_without = without_result["total_predicted_score"]
    replaced_by = sorted(set(without_result["lineup"]["name"]) - set(with_result["lineup"]["name"]))

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

    return {
        "injured_player": injured_player_name,
        "player_projected_score": lost_score,
        "player_avg_minutes": lost_minutes,
        "team_score_with": round(team_score_with, 2),
        "team_score_without": round(team_score_without, 2),
        "score_delta": delta,
        "impact_percentage": impact_pct,
        "impact_level": impact_level,
        "replaced_by": replaced_by,
    }


def batch_injury_report(roster_df: pd.DataFrame, roster_scores: dict[str, dict]) -> list[dict]:
    """
    Run injury impact analysis for every player on the roster.
    Useful for showing 'what if X goes down?' for the full team.
    Returns sorted list by impact percentage descending.
    """
    reports = []
    for player_name in roster_scores:
        if roster_scores[player_name]["status"] == "ACTIVE":
            report = compute_injury_impact(player_name, roster_df, roster_scores)
            if "error" not in report:
                reports.append(report)

    return sorted(reports, key=lambda x: x.get("impact_percentage", 0), reverse=True)
