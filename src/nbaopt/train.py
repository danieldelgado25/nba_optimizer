"""
train.py
Builds and saves the LSTM model weights used by predictor.py.

Fetches a season of game logs for every player on the given teams (default:
all 30, last completed season), trains the LSTM, evaluates it against the
rolling-average baseline on held-out games, and writes:
  artifacts/lstm_weights.pt
  artifacts/lstm_weights.pt.norm.npz
  artifacts/lstm_weights.pt.metrics.json

Usage
-----
# All 30 teams, last completed season (default)
nbaopt train

# Specific teams / season / hyperparameters
nbaopt train --team "Boston Celtics" --team "Denver Nuggets"
nbaopt train --season 2024-25 --epochs 10 --lr 0.0005
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from nba_api.stats.static import teams as nba_teams_static

from nbaopt.config import previous_season
from nbaopt.data.fetcher import (
    def_rating_by_abbr,
    get_player_game_log,
    get_team_defensive_rating,
    get_team_id,
    get_team_roster,
    opponent_abbr,
)
from nbaopt.models.predictor import LEAGUE_AVG_DEF_RATING, train_model


def fetch_training_data(
    team_names: list[str],
    games_per_player: int = 82,
    season: str | None = None,
) -> pd.DataFrame:
    """
    Pull game logs for all players on the given teams for one season.
    Attaches each game's opponent DEF_RATING (parsed from MATCHUP).
    Returns a concatenated DataFrame ready for train_model().
    """
    season = season or previous_season()
    print(f"\nFetching {season} defensive ratings for all teams...")
    try:
        rating_map = def_rating_by_abbr(get_team_defensive_rating(season=season))
    except Exception as e:
        print(f"  Warning: could not fetch DEF ratings ({e}), using league avg.")
        rating_map = {}

    all_logs = []
    seen_ids = set()  # players traded mid-season can appear on two rosters

    for team_name in team_names:
        team_id = get_team_id(team_name)
        if not team_id:
            print(f"  Skipping unknown team: '{team_name}'")
            continue

        print(f"\nFetching roster: {team_name}")
        try:
            roster = get_team_roster(team_id, season=season)
        except Exception as e:
            print(f"  Could not fetch roster: {e}")
            continue

        for i, (_, player) in enumerate(roster.iterrows(), 1):
            name = player["PLAYER"]
            pid = player["PLAYER_ID"]
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            print(f"  [{i:>2}/{len(roster)}] {name} ...", end="\r")

            try:
                log = get_player_game_log(pid, last_n=games_per_player, season=season)
                if log.empty:
                    continue
                log["PLAYER_ID"] = pid
                log["OPP_DEF_RATING"] = opponent_abbr(log["MATCHUP"]).map(rating_map).fillna(LEAGUE_AVG_DEF_RATING)
                all_logs.append(log)

            except Exception as e:
                print(f"\n  Skipped {name}: {e}")
                continue

    print(f"\n\nDone. Collected logs for {len(all_logs)} players.")

    if not all_logs:
        raise ValueError("No game logs collected. Check team names and API availability.")

    combined = pd.concat(all_logs, ignore_index=True)
    print(f"Total game rows: {len(combined)}")
    return combined


TRAIN_EXAMPLES = """
Examples:
  nbaopt train
  nbaopt train --team "Celtics" --team "Lakers" --team "Warriors"
  nbaopt train --season 2024-25 --games 82 --lr 0.0005 --batch-size 64
"""


def add_train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.epilog = TRAIN_EXAMPLES
    parser.add_argument(
        "--team",
        action="append",
        dest="teams",
        default=None,
        help="Team name(s) to pull training data from (can pass multiple times). Default: all 30.",
    )
    # 6: held-out MAE bottoms out around epoch 5 on 2025-26 data, then overfits
    parser.add_argument("--epochs", type=int, default=6, help="Training epochs (default: 6)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 0.001)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--games", type=int, default=82, help="Games per player to fetch (default: 82)")
    parser.add_argument(
        "--val-frac", type=float, default=0.2, help="Last fraction of each player's games held out (default: 0.2)"
    )
    parser.add_argument("--season", default=None, help="Season like '2025-26' (default: last completed season)")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Where to write weights (default: $NBAOPT_MODEL_PATH or ./artifacts/lstm_weights.pt)",
    )


def run_training(args: argparse.Namespace) -> None:
    season = args.season or previous_season()
    teams = args.teams or [t["full_name"] for t in nba_teams_static.get_teams()]

    print("=" * 62)
    print("  NBA LSTM Predictor — Training")
    print("=" * 62)
    print(f"\n  Season  : {season}")
    print(f"  Teams   : {'all 30' if not args.teams else ', '.join(teams)}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  LR      : {args.lr}")
    print(f"  Batch   : {args.batch_size}")
    print(f"  Games   : {args.games} per player")

    # ponytail: cache key ignores --team/--games; delete the file to refetch
    cache = Path.cwd() / "data" / f"train_{season}.csv"
    if cache.exists():
        print(f"\n  Using cached game logs: {cache}")
        df = pd.read_csv(cache)
    else:
        df = fetch_training_data(teams, games_per_player=args.games, season=season)
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)

    print("\n" + "=" * 62)
    print("  Training LSTM...")
    print("=" * 62 + "\n")

    train_model(
        df,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        save_path=args.model_path,
        verbose=True,
    )

    print("\n" + "=" * 62)
    print("  Training complete.")
    print("  Run `nbaopt lineup` to use the trained model for lineup optimization.")
    print("=" * 62 + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the NBA LSTM performance predictor.")
    add_train_arguments(parser)
    run_training(parser.parse_args(argv))


if __name__ == "__main__":
    main()
