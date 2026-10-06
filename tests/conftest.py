from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from nbaopt.models import predictor

GAME_LOG_COLUMNS = [
    "GAME_DATE",
    "MATCHUP",
    "WL",
    "MIN",
    "PTS",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "FG_PCT",
    "FG3_PCT",
    "FT_PCT",
    "PLUS_MINUS",
]


def make_game_log(
    n_games: int = 10,
    pts: float = 20.0,
    minutes: float = 32.0,
    newest_first: bool = True,
    seed: int = 0,
    start: date = date(2025, 10, 22),
) -> pd.DataFrame:
    """Synthetic game log shaped like nba_api PlayerGameLog output (newest game first by default)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_games):
        rows.append(
            {
                "GAME_DATE": (start + timedelta(days=2 * i)).strftime("%b %d, %Y").upper(),
                "MATCHUP": "BOS vs. MIA" if i % 2 == 0 else "BOS @ NYK",
                "WL": "W" if i % 3 else "L",
                "MIN": minutes,
                "PTS": pts + float(rng.integers(-3, 4)),
                "REB": 5.0,
                "AST": 4.0,
                "STL": 1.0,
                "BLK": 0.5,
                "TOV": 2.0,
                "FG_PCT": 0.48,
                "FG3_PCT": 0.36,
                "FT_PCT": 0.82,
                "PLUS_MINUS": float(rng.integers(-10, 11)),
            }
        )
    df = pd.DataFrame(rows, columns=GAME_LOG_COLUMNS)
    return df.iloc[::-1].reset_index(drop=True) if newest_first else df


def make_roster(players: list[tuple[str, str]]) -> pd.DataFrame:
    rows = [
        {"PLAYER_ID": 1000 + i, "PLAYER": name, "NUM": str(i), "POSITION": pos} for i, (name, pos) in enumerate(players)
    ]
    return pd.DataFrame(rows)


def active(score: float, minutes: float = 30.0) -> dict:
    return {"predicted_score": score, "status": "ACTIVE", "avg_minutes": minutes, "rolling_stats": {}}


@pytest.fixture
def game_log() -> pd.DataFrame:
    return make_game_log()


@pytest.fixture
def no_model(tmp_path, monkeypatch):
    """Point the predictor at an empty model location and reset its module-level cache."""
    monkeypatch.setenv("NBAOPT_MODEL_PATH", str(tmp_path / "missing.pt"))
    monkeypatch.setattr(predictor, "_MODEL_LOADED", False)
    monkeypatch.setattr(predictor, "_MODEL", None)
    monkeypatch.setattr(predictor, "_NORMALIZER", None)
    return tmp_path


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    """Like no_model, but tests are expected to train a model into tmp_path/lstm.pt."""
    path = tmp_path / "lstm.pt"
    monkeypatch.setenv("NBAOPT_MODEL_PATH", str(path))
    monkeypatch.setattr(predictor, "_MODEL_LOADED", False)
    monkeypatch.setattr(predictor, "_MODEL", None)
    monkeypatch.setattr(predictor, "_NORMALIZER", None)
    return path
