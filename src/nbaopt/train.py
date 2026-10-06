"""
train.py
Builds and saves the LSTM model weights used by predictor.py.

Fetches historical game logs for every player on a given team (or a list
of player IDs), trains the LSTM, and writes:
  models/lstm_weights.pt
  models/lstm_weights.pt.norm.npz

Usage
-----
# Train on one team's current roster
nbaopt train --team "Boston Celtics"

# Train on multiple teams for a larger, more generalizable model
nbaopt train --team "Boston Celtics" --team "Los Angeles Lakers" --team "Denver Nuggets"

# Control training hyperparameters
nbaopt train --team "Celtics" --epochs 50 --games 82 --lr 0.0005
"""

import argparse
from collections.abc import Sequence

import pandas as pd

from nbaopt.data.fetcher import get_player_game_log, get_team_defensive_rating, get_team_id, get_team_roster
from nbaopt.models.predictor import train_model


def fetch_training_data(
    team_names: list[str],
    games_per_player: int = 40,
    season: str | None = None,
) -> pd.DataFrame:
    """
    Pull game logs for all players on the given teams.
    Attempts to attach opponent DEF_RATING to each game row.
    Returns a concatenated DataFrame ready for train_model().
    """
    print("\nFetching defensive ratings for all teams...")
    try:
        def_df = get_team_defensive_rating(season=season)
        def_rating_map = dict(zip(def_df["TEAM_ID"], def_df["DEF_RATING"], strict=True))  # noqa: F841 (unused until P2)
    except Exception as e:
        print(f"  Warning: could not fetch DEF ratings ({e}), using league avg.")
        def_rating_map = {}  # noqa: F841

    all_logs = []

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
            print(f"  [{i:>2}/{len(roster)}] {name} ...", end="\r")

            try:
                log = get_player_game_log(pid, last_n=games_per_player, season=season)
                if log.empty:
                    continue

                # Try to attach opponent DEF_RATING from the MATCHUP column
                # nba_api MATCHUP format: "BOS vs. MIA" or "BOS @ MIA"
                # We approximate by assigning a fixed team avg for now;
                # a more precise approach requires parsing matchup strings.
                log["OPP_DEF_RATING"] = 113.5  # league avg fallback

                # Compute fantasy score label
                for col in ["PTS", "REB", "AST", "STL", "BLK", "TOV"]:
                    if col in log.columns:
                        log[col] = pd.to_numeric(log[col], errors="coerce").fillna(0)
                log["FANTASY_SCORE"] = (
                    log.get("PTS", 0) * 1.0
                    + log.get("REB", 0) * 1.2
                    + log.get("AST", 0) * 1.5
                    + log.get("STL", 0) * 3.0
                    + log.get("BLK", 0) * 3.0
                    + log.get("TOV", 0) * -1.0
                )

                # Reverse so oldest game is first (LSTM needs chronological order)
                log = log.iloc[::-1].reset_index(drop=True)
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
  nbaopt train --team "Boston Celtics"
  nbaopt train --team "Celtics" --team "Lakers" --team "Warriors" --epochs 50
  nbaopt train --team "Celtics" --games 82 --lr 0.0005 --batch-size 64
"""


def add_train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.epilog = TRAIN_EXAMPLES
    parser.add_argument(
        "--team",
        action="append",
        dest="teams",
        required=True,
        help="Team name(s) to pull training data from (can pass multiple times).",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (default: 30)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 0.001)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--games", type=int, default=40, help="Games per player to fetch (default: 40)")
    parser.add_argument("--season", default=None, help="Season like '2025-26' (default: current season)")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Where to write weights (default: $NBAOPT_MODEL_PATH or ./artifacts/lstm_weights.pt)",
    )


def run_training(args: argparse.Namespace) -> None:
    print("=" * 62)
    print("  NBA LSTM Predictor — Training")
    print("=" * 62)
    print(f"\n  Teams   : {', '.join(args.teams)}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  LR      : {args.lr}")
    print(f"  Batch   : {args.batch_size}")
    print(f"  Games   : {args.games} per player")

    df = fetch_training_data(args.teams, games_per_player=args.games, season=args.season)

    print("\n" + "=" * 62)
    print("  Training LSTM...")
    print("=" * 62 + "\n")

    train_model(
        df,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
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
