"""
lineup.py
Selects the optimal 5-man lineup from a roster using predicted scores.

Positional logic:
  - Tries to fill: PG, SG, SF, PF, C (standard 5)
  - Falls back to best-5-by-score if positional data is incomplete
  - Respects injury status (OUT players excluded)
"""

import pandas as pd
from itertools import combinations
from typing import Optional

STANDARD_POSITIONS = ["PG", "SG", "SF", "PF", "C"]


def parse_positions(position_str: str) -> list[str]:
    """Parse position strings like 'PG-SG' or 'F' into a list."""
    if not position_str:
        return []
    # Normalize common abbreviations
    pos = position_str.upper().replace(" ", "")
    parts = pos.replace("-", " ").replace("/", " ").split()
    expanded = []
    for p in parts:
        if p == "G":
            expanded += ["PG", "SG"]
        elif p == "F":
            expanded += ["SF", "PF"]
        elif p == "C":
            expanded.append("C")
        else:
            expanded.append(p)
    return list(set(expanded))


def build_player_pool(
    roster_df: pd.DataFrame,
    roster_scores: dict[str, dict],
) -> pd.DataFrame:
    """
    Merge roster info with predicted scores into one DataFrame.
    Filters out injured/no-data players.
    """
    rows = []
    for _, player in roster_df.iterrows():
        name = player["PLAYER"]
        score_data = roster_scores.get(name, {})
        if score_data.get("status") not in ("ACTIVE",):
            continue
        rows.append(
            {
                "name": name,
                "position": player.get("POSITION", ""),
                "positions": parse_positions(player.get("POSITION", "")),
                "predicted_score": score_data.get("predicted_score", 0),
                "avg_minutes": score_data.get("avg_minutes", 0),
                "rolling_stats": score_data.get("rolling_stats", {}),
            }
        )
    return pd.DataFrame(rows)


def greedy_positional_lineup(pool: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Greedy approach: fill each position slot with the highest-scoring
    eligible player who hasn't already been assigned.
    """
    assigned = set()
    lineup = []

    pool_sorted = pool.sort_values("predicted_score", ascending=False)

    for pos in STANDARD_POSITIONS:
        eligible = pool_sorted[
            pool_sorted["positions"].apply(lambda p: pos in p)
            & ~pool_sorted["name"].isin(assigned)
        ]
        if eligible.empty:
            # fallback: any unassigned player
            remaining = pool_sorted[~pool_sorted["name"].isin(assigned)]
            if remaining.empty:
                return None
            best = remaining.iloc[0]
        else:
            best = eligible.iloc[0]

        assigned.add(best["name"])
        lineup.append({**best.to_dict(), "slot": pos})

    return pd.DataFrame(lineup)


def brute_force_best_five(pool: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Exhaustive search for the highest total projected score
    from any combination of 5 players (no positional constraint).
    Used as fallback or for comparison.
    """
    if len(pool) < 5:
        return None

    best_score = -1
    best_combo = None

    for combo in combinations(pool.itertuples(index=False), 5):
        total = sum(p.predicted_score for p in combo)
        if total > best_score:
            best_score = total
            best_combo = combo

    if not best_combo:
        return None

    rows = [c._asdict() for c in best_combo]
    df = pd.DataFrame(rows)
    df["slot"] = STANDARD_POSITIONS[: len(df)]
    return df


def optimize_lineup(
    roster_df: pd.DataFrame,
    roster_scores: dict[str, dict],
    use_positions: bool = True,
) -> dict:
    """
    Main lineup optimizer entry point.

    Returns:
        {
          lineup: DataFrame of 5 players,
          total_predicted_score: float,
          method: str,
          excluded_players: list of OUT/no-data players
        }
    """
    pool = build_player_pool(roster_df, roster_scores)

    excluded = [
        name
        for name, data in roster_scores.items()
        if data.get("status") != "ACTIVE"
    ]

    if len(pool) < 5:
        return {
            "lineup": None,
            "total_predicted_score": 0,
            "method": "ERROR",
            "excluded_players": excluded,
            "error": f"Only {len(pool)} active players available, need at least 5.",
        }

    lineup = greedy_positional_lineup(pool) if use_positions else None

    if lineup is None or len(lineup) < 5:
        lineup = brute_force_best_five(pool)
        method = "brute_force_top5"
    else:
        method = "positional_greedy"

    total = round(lineup["predicted_score"].sum(), 2) if lineup is not None else 0

    return {
        "lineup": lineup,
        "total_predicted_score": total,
        "method": method,
        "excluded_players": excluded,
        "pool_size": len(pool),
    }
