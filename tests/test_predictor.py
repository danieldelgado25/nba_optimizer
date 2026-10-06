import numpy as np
import pandas as pd
import pytest
import torch

from conftest import make_game_log
from nbaopt.models import predictor
from nbaopt.models.predictor import (
    LEAGUE_AVG_DEF_RATING,
    GameSequenceDataset,
    compute_fantasy_score,
    predict_player_score,
    train_model,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


def training_frame(n_players: int = 3, n_games: int = 20) -> pd.DataFrame:
    logs = []
    for p in range(n_players):
        log = make_game_log(n_games, pts=15 + 5 * p, newest_first=False, seed=p)
        log["PLAYER_ID"] = p
        log["OPP_DEF_RATING"] = LEAGUE_AVG_DEF_RATING
        logs.append(log)
    return pd.concat(logs, ignore_index=True)


def test_compute_fantasy_score_uses_weights():
    row = pd.Series({"PTS": 20, "REB": 10, "AST": 5, "STL": 2, "BLK": 1, "TOV": 3})
    assert compute_fantasy_score(row) == pytest.approx(20 + 12 + 7.5 + 6 + 3 - 3)


def test_injured_player_is_out(game_log):
    result = predict_player_score(game_log, injured=True)
    assert result["status"] == "OUT"
    assert result["predicted_score"] == 0.0


def test_empty_log_is_no_data():
    assert predict_player_score(pd.DataFrame())["status"] == "NO_DATA"


def test_rolling_average_fallback_without_model(no_model, game_log):
    result = predict_player_score(game_log)
    expected = compute_fantasy_score(game_log[["PTS", "REB", "AST", "STL", "BLK", "TOV"]].mean())
    assert result["model_used"] == "rolling_avg_fallback"
    assert result["status"] == "ACTIVE"
    assert result["games_sampled"] == 10
    assert result["predicted_score"] == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize(("rating", "ratio"), [(LEAGUE_AVG_DEF_RATING, 1.0), (200.0, 1.15), (50.0, 0.85)])
def test_opponent_adjustment_is_capped(no_model, game_log, rating, ratio):
    base = predict_player_score(game_log)["base_score"]
    assert predict_player_score(game_log, rating)["base_score"] == pytest.approx(base * ratio, abs=0.02)


def test_minutes_penalty_scales_low_minute_players(no_model):
    log = make_game_log(minutes=7.5)
    result = predict_player_score(log)
    assert result["predicted_score"] == pytest.approx(result["base_score"] * 0.5, abs=0.01)


def test_dataset_builds_next_game_targets():
    df = training_frame(n_players=1, n_games=12)
    ds = GameSequenceDataset(df, seq_len=10)
    assert len(ds) == 2
    x, y = ds[0]
    assert tuple(x.shape) == (10, predictor.INPUT_SIZE)
    assert float(y) == pytest.approx(compute_fantasy_score(df.iloc[10]))


def test_train_model_writes_weights_and_normalizer(model_dir):
    train_model(training_frame(), epochs=2, verbose=False)
    assert model_dir.exists()
    assert model_dir.with_name(model_dir.name + ".norm.npz").exists()


@pytest.mark.xfail(strict=True, reason="Known bug: float64 normalizer output crashes the float32 LSTM (fixed in P2)")
def test_trained_model_is_used_for_inference(model_dir, game_log):
    train_model(training_frame(), epochs=2, verbose=False)
    result = predict_player_score(game_log)
    assert result["model_used"] == "lstm"
    assert result["predicted_score"] >= 0


def test_train_model_rejects_too_little_data(model_dir):
    with pytest.raises(ValueError, match="Not enough rows"):
        train_model(training_frame(n_players=1, n_games=5), epochs=1, verbose=False)


@pytest.mark.xfail(strict=True, reason="Known bug: windows span player boundaries (fixed in P2)")
def test_dataset_does_not_cross_player_boundaries():
    ds = GameSequenceDataset(training_frame(n_players=2, n_games=12), seq_len=10)
    assert len(ds) == 4


@pytest.mark.xfail(strict=True, reason="Known bug: constant OPP_DEF_RATING explodes normalization (fixed in P2)")
def test_opponent_rating_normalizes_to_a_sane_range(model_dir, game_log):
    _, normalizer = train_model(training_frame(), epochs=1, verbose=False)
    seq = predictor._build_inference_sequence(game_log, opponent_def_rating=110.0)
    assert np.abs(normalizer.transform(seq)).max() < 10
