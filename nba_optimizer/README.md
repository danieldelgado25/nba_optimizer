# NBA Lineup Optimizer

A Python CLI tool that builds optimal NBA lineups with injury impact analysis.

## Features
- Pulls live player stats and roster data via `nba_api`
- Predicts player performance using rolling averages + opponent defensive rating adjustment
- Injury impact modeling: see how much a player's absence hurts the team
- Full team "what if everyone went down?" impact report
- Optimal 5-man lineup selection with positional constraints

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Basic lineup for a team
python main.py --team "Boston Celtics"

# With opponent defensive rating adjustment
python main.py --team "Celtics" --opponent "Warriors"

# Mark players as OUT
python main.py --team "Celtics" --injured "Jaylen Brown, Al Horford"

# See how much one player's absence impacts the team
python main.py --team "Celtics" --injury-impact "Jayson Tatum"

# Full roster injury impact report (sorted by most impactful)
python main.py --team "Celtics" --impact-report

# Full combo
python main.py --team "Lakers" --opponent "Nuggets" --injured "LeBron James" --impact-report
```

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
├── data/
│   └── fetcher.py          # nba_api data layer
├── models/
│   ├── predictor.py        # Rolling avg + opp adjustment scoring
│   └── injury_impact.py    # Team impact modeling
├── optimizer/
│   └── lineup.py           # 5-man lineup selection
├── main.py                 # CLI entry point
├── requirements.txt
└── README.md
```

## How the Prediction Works

1. **Rolling averages** — Last 10 games of pts/reb/ast/stl/blk/tov
2. **Fantasy scoring** — Weighted sum (pts ×1.0, ast ×1.5, stl/blk ×3.0, tov −1.0)
3. **Opponent adjustment** — Scale by opponent DEF_RATING vs league average (±15% cap)
4. **Minutes penalty** — Players under 15 avg minutes are scaled down proportionally

## How Injury Impact Works

When a player is ruled OUT:
- Their projected score is removed from team total
- Their "usage" is redistributed to remaining active players (proportional to their own scores, with a 70% recovery assumption)
- Impact level is classified: 🔴 Critical (12%+), 🟠 High (7%+), 🟡 Moderate (3%+), 🟢 Low
