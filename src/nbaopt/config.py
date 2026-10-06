"""
config.py
Runtime settings resolved from environment variables with sensible defaults.

  NBAOPT_SEASON      Season string like "2025-26" (default: derived from today's date)
  NBAOPT_MODEL_PATH  LSTM weights file (default: ./artifacts/lstm_weights.pt)
"""

import os
from datetime import date
from pathlib import Path

SEASON_ROLLOVER_MONTH = 10  # NBA seasons start in October


def season_for_date(day: date) -> str:
    """Return the NBA season string (e.g. "2025-26") that contains the given date."""
    start_year = day.year if day.month >= SEASON_ROLLOVER_MONTH else day.year - 1
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def current_season() -> str:
    return os.environ.get("NBAOPT_SEASON") or season_for_date(date.today())


def model_path() -> Path:
    env = os.environ.get("NBAOPT_MODEL_PATH")
    return Path(env) if env else Path.cwd() / "artifacts" / "lstm_weights.pt"


def normalizer_path(weights_path: Path) -> Path:
    return weights_path.with_name(weights_path.name + ".norm.npz")
