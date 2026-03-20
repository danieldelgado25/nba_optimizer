"""
main.py
NBA Lineup Optimizer — CLI Entry Point

Usage:
    python main.py --team "Boston Celtics"
    python main.py --team "BOS" --injured "Jaylen Brown"
    python main.py --team "Lakers" --injured "LeBron James" --opponent "Golden State Warriors"
    python main.py --team "Celtics" --impact-report
"""

import argparse
import sys
import time
import pandas as pd

from data.fetcher import (
    get_team_id,
    get_team_roster,
    get_player_game_log,
    get_team_defensive_rating,
)
from models.predictor import predict_player_score
from models.injury_impact import compute_injury_impact, batch_injury_report
from optimizer.lineup import optimize_lineup


# ── Formatting helpers ────────────────────────────────────────────────────────

DIVIDER = "─" * 62


def header(title: str):
    print(f"\n{'═' * 62}")
    print(f"  {title}")
    print(f"{'═' * 62}")


def section(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def print_lineup(lineup_result: dict):
    lineup = lineup_result.get("lineup")
    if lineup is None:
        print(f"\n  ❌ {lineup_result.get('error', 'Could not build lineup.')}")
        return

    section("🏀 OPTIMAL LINEUP")
    print(f"  {'SLOT':<5} {'PLAYER':<26} {'POS':<8} {'PROJ SCORE':>10}  {'AVG MIN':>7}")
    print(f"  {'-'*5} {'-'*26} {'-'*8} {'-'*10}  {'-'*7}")

    for _, row in lineup.iterrows():
        pos_display = row.get("position", "—")[:8]
        print(
            f"  {row['slot']:<5} {row['name']:<26} {pos_display:<8} "
            f"{row['predicted_score']:>10.2f}  {row['avg_minutes']:>7.1f}"
        )

    print(f"\n  {'TOTAL PROJECTED SCORE':.<40} {lineup_result['total_predicted_score']:>8.2f}")
    print(f"  {'METHOD':.<40} {lineup_result['method']:>8}")
    print(f"  {'ACTIVE PLAYERS IN POOL':.<40} {lineup_result['pool_size']:>8}")

    if lineup_result["excluded_players"]:
        print(f"\n  ⚠️  EXCLUDED (OUT/NO DATA): {', '.join(lineup_result['excluded_players'])}")


def print_injury_impact(report: dict):
    if "error" in report:
        print(f"  ❌ {report['error']}")
        return

    print(f"\n  Player         : {report['injured_player']}")
    print(f"  Proj Score     : {report['player_projected_score']:.2f} pts")
    print(f"  Avg Minutes    : {report['player_avg_minutes']:.1f} min")
    print(f"  Team Score WITH: {report['team_score_with']:.2f}")
    print(f"  Team Score OUT : {report['team_score_without']:.2f}")
    print(f"  Score Delta    : -{report['score_delta']:.2f} ({report['impact_percentage']}%)")
    print(f"  Impact Level   : {report['impact_level']}")

    if report["usage_absorbed_by"]:
        print(f"\n  Top Usage Absorbers:")
        for name, boost in report["usage_absorbed_by"]:
            print(f"    +{boost:.2f}  {name}")


def print_full_impact_report(reports: list[dict]):
    section("📊 FULL INJURY IMPACT REPORT (all active players)")
    print(f"  {'PLAYER':<26} {'PROJ SCORE':>10}  {'DELTA':>7}  {'IMPACT%':>8}  LEVEL")
    print(f"  {'-'*26} {'-'*10}  {'-'*7}  {'-'*8}  {'-'*14}")
    for r in reports:
        print(
            f"  {r['injured_player']:<26} {r['player_projected_score']:>10.2f}  "
            f"{r['score_delta']:>7.2f}  {r['impact_percentage']:>7.1f}%  {r['impact_level']}"
        )


# ── Core pipeline ─────────────────────────────────────────────────────────────

def build_roster_scores(
    roster_df: pd.DataFrame,
    injured_names: list[str],
    opp_def_rating: float,
    last_n: int = 10,
) -> dict[str, dict]:
    """Fetch game logs and compute predicted scores for every player."""
    roster_scores = {}
    total = len(roster_df)

    print(f"\n  Fetching game logs for {total} players (this takes ~{total * 0.7:.0f}s)...\n")

    for i, (_, player) in enumerate(roster_df.iterrows(), 1):
        name = player["PLAYER"]
        pid = player["PLAYER_ID"]
        is_injured = any(inj.lower() in name.lower() for inj in injured_names)

        print(f"  [{i:>2}/{total}] {name} {'⛔ OUT' if is_injured else '...'}", end="\r")

        if is_injured:
            roster_scores[name] = {"predicted_score": 0.0, "status": "OUT", "avg_minutes": 0, "rolling_stats": {}}
            continue

        try:
            log = get_player_game_log(pid, last_n=last_n)
            roster_scores[name] = predict_player_score(log, opp_def_rating)
        except Exception as e:
            roster_scores[name] = {"predicted_score": 0.0, "status": "NO_DATA", "avg_minutes": 0, "rolling_stats": {}}

    print(f"  Done fetching.{' ' * 40}")
    return roster_scores


def run(args):
    header("🏀 NBA LINEUP OPTIMIZER")

    # ── Resolve team ──────────────────────────────────────────────────────────
    team_id = get_team_id(args.team)
    if not team_id:
        print(f"\n  ❌ Could not find team: '{args.team}'. Try full name or abbreviation.")
        sys.exit(1)
    print(f"\n  ✅ Team found: {args.team}")

    # ── Opponent defensive rating ─────────────────────────────────────────────
    opp_def_rating = 113.5  # default league avg
    if args.opponent:
        opp_id = get_team_id(args.opponent)
        if opp_id:
            try:
                def_df = get_team_defensive_rating()
                match = def_df[def_df["TEAM_ID"] == opp_id]
                if not match.empty:
                    opp_def_rating = float(match.iloc[0]["DEF_RATING"])
                    print(f"  ✅ Opponent '{args.opponent}' DEF_RATING: {opp_def_rating:.1f}")
            except Exception:
                print(f"  ⚠️  Could not fetch opponent DEF_RATING, using league average.")
        else:
            print(f"  ⚠️  Opponent '{args.opponent}' not found, using league average DEF_RATING.")

    # ── Roster ────────────────────────────────────────────────────────────────
    print(f"  Fetching roster...")
    roster_df = get_team_roster(team_id)
    print(f"  ✅ Roster loaded: {len(roster_df)} players")

    injured_names = [n.strip() for n in args.injured.split(",")] if args.injured else []
    if injured_names:
        print(f"  ⛔ Marked as OUT: {', '.join(injured_names)}")

    # ── Compute scores ────────────────────────────────────────────────────────
    roster_scores = build_roster_scores(roster_df, injured_names, opp_def_rating)

    # ── Optimize lineup ───────────────────────────────────────────────────────
    result = optimize_lineup(roster_df, roster_scores)
    print_lineup(result)

    # ── Injury impact for specific player ─────────────────────────────────────
    if args.injury_impact:
        section(f"🩺 INJURY IMPACT: {args.injury_impact}")
        report = compute_injury_impact(args.injury_impact, roster_scores)
        print_injury_impact(report)

    # ── Full impact report ────────────────────────────────────────────────────
    if args.impact_report:
        reports = batch_injury_report(roster_scores)
        print_full_impact_report(reports)

    print(f"\n{'═' * 62}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NBA Lineup Optimizer with Injury Impact Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --team "Boston Celtics"
  python main.py --team "BOS" --injured "Jaylen Brown"
  python main.py --team "Lakers" --opponent "Warriors" --injured "LeBron James"
  python main.py --team "Celtics" --injury-impact "Jayson Tatum"
  python main.py --team "Celtics" --impact-report
        """,
    )
    parser.add_argument("--team", required=True, help="Team name or abbreviation (e.g. 'Celtics', 'BOS')")
    parser.add_argument("--opponent", default=None, help="Opponent team (used for defensive rating adjustment)")
    parser.add_argument("--injured", default=None, help="Comma-separated list of injured/out players")
    parser.add_argument("--injury-impact", default=None, dest="injury_impact",
                        help="Show impact analysis if this specific player is ruled out")
    parser.add_argument("--impact-report", action="store_true", dest="impact_report",
                        help="Show full injury impact report for all players")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
