# NBA Lineup Optimizer

A Python CLI tool that builds optimal NBA lineups with injury impact analysis.

## Features
- Pulls live player stats and roster data via `nba_api`
- Predicts player performance with a PyTorch LSTM over recent games, falling back to rolling averages + opponent defensive rating adjustment when no trained model is available
- Injury impact modeling: see how much a player's absence hurts the team
- Full team "what if everyone went down?" impact report
- Optimal 5-man lineup selection with positional constraints

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"        # or: pip install -r requirements.txt
pre-commit install             # optional, runs ruff on commit
```

## Usage

```bash
# Basic lineup for a team
nbaopt lineup --team "Boston Celtics"

# With opponent defensive rating adjustment
nbaopt lineup --team "Celtics" --opponent "Warriors"

# Mark players as OUT
nbaopt lineup --team "Celtics" --injured "Jaylen Brown, Al Horford"

# See how much one player's absence impacts the team
nbaopt lineup --team "Celtics" --injury-impact "Jayson Tatum"

# Full roster injury impact report (sorted by most impactful)
nbaopt lineup --team "Celtics" --impact-report

# Pick a specific season (default: current season, rolling over each October)
nbaopt lineup --team "Lakers" --opponent "Nuggets" --injured "LeBron James" --impact-report --season 2024-25

# Train the LSTM (writes ./artifacts/lstm_weights.pt + .norm.npz)
nbaopt train --team "Boston Celtics" --team "Denver Nuggets" --epochs 50
```

`python main.py ...` and `python train.py ...` still work as aliases for `nbaopt lineup` and `nbaopt train`.

### Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `NBAOPT_SEASON` | derived from today's date | Season used for all API calls (`--season` overrides) |
| `NBAOPT_MODEL_PATH` | `./artifacts/lstm_weights.pt` | LSTM weights; the normalizer is stored next to it as `<path>.norm.npz` |

## Sample Output

```
══════════════════════════════════════════════════════════════
  🏀 OPTIMAL LINEUP
──────────────────────────────────────────────────────────────
  SLOT  PLAYER                     POS       PROJ SCORE  AVG MIN
  ----- -------------------------- --------  ----------  -------
  PG    Jaylen Brown               SG-SF          42.10     35.2
  SG    Derrick White              PG-SG          28.75     30.1
  SF    Jayson Tatum               SF-PF          51.30     36.8
  PF    Al Horford                 C-PF           22.40     28.3
  C     Kristaps Porzingis         C              31.20     27.9

  TOTAL PROJECTED SCORE............................ 175.75
```

## Project Structure

```
nba_optimizer/
├── src/nbaopt/
│   ├── cli.py                # `nbaopt` entry point (lineup + train subcommands)
│   ├── config.py             # Season / model path settings
│   ├── train.py              # LSTM training pipeline
│   ├── data/fetcher.py       # nba_api data layer
│   ├── models/predictor.py   # LSTM + rolling-average scoring
│   ├── models/injury_impact.py
│   └── optimizer/lineup.py   # 5-man lineup selection
├── tests/                    # Offline pytest suite (no network)
├── main.py, train.py         # Backwards-compatible aliases
└── pyproject.toml
```

## How the Prediction Works

1. **Fantasy scoring** — Weighted sum: pts ×1.0, reb ×1.2, ast ×1.5, stl/blk ×3.0, tov −1.0
2. **LSTM** — If trained weights exist and a player has 5+ games, the last 10 games (box score stats + opponent DEF_RATING) are fed to a 2-layer LSTM that predicts the next game's fantasy score
3. **Fallback** — Otherwise: rolling average of the last 10 games, scaled by opponent DEF_RATING vs league average (±15% cap)
4. **Minutes penalty** — Players under 15 avg minutes are scaled down proportionally

## How Injury Impact Works

When a player is ruled OUT:
- Their projected score is removed from team total
- Their "usage" is redistributed to remaining active players (proportional to their own scores, with a 70% recovery assumption)
- Impact level is classified: 🔴 Critical (12%+), 🟠 High (7%+), 🟡 Moderate (3%+), 🟢 Low

## Development

```bash
make check        # ruff + mypy + pytest (or run them individually)
pytest            # all tests run offline against synthetic fixtures
```

Known bugs are captured as strict `xfail` tests (see `pytest -ra`); fixing one flips its test to a failure until the marker is removed.
