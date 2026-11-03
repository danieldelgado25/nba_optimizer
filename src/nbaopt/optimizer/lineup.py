"""
lineup.py
Selects the optimal 5-man lineup from a roster using predicted scores.

Positional logic:
  - Fills PG, SG, SF, PF, C (standard 5) with the highest-total eligible players
  - Falls back to best-5-by-score if no positionally valid lineup exists
  - Respects injury status (OUT players excluded)
"""

from itertools import combinations, permutations

import pandas as pd

STANDARD_POSITIONS = ["PG", "SG", "SF", "PF", "C"]


def parse_positions(position_str: str) -> list[str]:
    """Parse position strings like 'PG-SG' or 'F' into a list."""
    if not isinstance(position_str, str) or not position_str:  # NaN from CSV-loaded rosters
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


def best_positional_lineup(pool: pd.DataFrame) -> pd.DataFrame | None:
    """
    Exact search: the highest-total 5 players that can fill PG/SG/SF/PF/C
    with every slot held by a position-eligible player.

    Combos are tried best-total first, so the first feasible one is optimal.
    """
    rows = list(pool.itertuples(index=False))
    combos = sorted(combinations(rows, 5), key=lambda c: -sum(p.predicted_score for p in c))
    for combo in combos:
        for perm in permutations(combo):
            if all(pos in p.positions for pos, p in zip(STANDARD_POSITIONS, perm, strict=True)):
                df = pd.DataFrame([p._asdict() for p in perm])
                df["slot"] = STANDARD_POSITIONS
                return df
    return None


def brute_force_best_five(pool: pd.DataFrame) -> pd.DataFrame | None:
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

    excluded = [name for name, data in roster_scores.items() if data.get("status") != "ACTIVE"]

    if len(pool) < 5:
        return {
            "lineup": None,
            "total_predicted_score": 0,
            "method": "ERROR",
            "excluded_players": excluded,
            "error": f"Only {len(pool)} active players available, need at least 5.",
        }

    lineup = best_positional_lineup(pool) if use_positions else None

    if lineup is None:
        lineup = brute_force_best_five(pool)
        method = "brute_force_top5"
    else:
        method = "positional_optimal"

    total = round(lineup["predicted_score"].sum(), 2) if lineup is not None else 0

    return {
        "lineup": lineup,
        "total_predicted_score": total,
        "method": method,
        "excluded_players": excluded,
        "pool_size": len(pool),
    }
