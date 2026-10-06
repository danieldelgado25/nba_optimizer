from datetime import date
from pathlib import Path

import pytest

from nbaopt import config


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (date(2026, 10, 6), "2026-27"),
        (date(2026, 10, 1), "2026-27"),
        (date(2026, 9, 30), "2025-26"),
        (date(2026, 6, 15), "2025-26"),
        (date(2025, 1, 10), "2024-25"),
        (date(1999, 11, 1), "1999-00"),
    ],
)
def test_season_for_date(day, expected):
    assert config.season_for_date(day) == expected


def test_current_season_env_override(monkeypatch):
    monkeypatch.setenv("NBAOPT_SEASON", "2023-24")
    assert config.current_season() == "2023-24"


def test_current_season_defaults_to_today(monkeypatch):
    monkeypatch.delenv("NBAOPT_SEASON", raising=False)
    assert config.current_season() == config.season_for_date(date.today())


def test_model_path_default_and_override(monkeypatch, tmp_path):
    monkeypatch.delenv("NBAOPT_MODEL_PATH", raising=False)
    assert config.model_path() == Path.cwd() / "artifacts" / "lstm_weights.pt"
    monkeypatch.setenv("NBAOPT_MODEL_PATH", str(tmp_path / "w.pt"))
    assert config.model_path() == tmp_path / "w.pt"
    assert config.normalizer_path(tmp_path / "w.pt") == tmp_path / "w.pt.norm.npz"
