import json

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import make_game_log
from nbaopt.config import metrics_path, normalizer_path
from nbaopt.models import predictor
from nbaopt.models.predictor import (
    LEAGUE_AVG_DEF_RATING,
    SEQ_LEN,
    FeatureNormalizer,
    build_sequences,
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


# ── Inference ─────────────────────────────────────────────────────────────────


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


def test_trained_model_is_used_for_inference(model_dir, game_log):
    train_model(training_frame(), epochs=2, verbose=False)
    result = predict_player_score(game_log)
    assert result["model_used"] == "lstm"
    assert result["predicted_score"] >= 0


def test_model_waits_for_a_full_window(model_dir):
    train_model(training_frame(), epochs=1, verbose=False)
    result = predict_player_score(make_game_log(SEQ_LEN - 1))
    assert result["model_used"] == "rolling_avg_fallback"


# ── Sequences ─────────────────────────────────────────────────────────────────


def test_build_sequences_targets_the_next_game():
    df = training_frame(n_players=1, n_games=12)
    X, y, _, is_val = build_sequences(df, seq_len=10)
    assert X.shape == (2, 10, predictor.INPUT_SIZE)
    assert X.dtype == np.float32
    assert y[0] == pytest.approx(compute_fantasy_score(df.iloc[10]), abs=1e-4)
    assert not is_val.any()  # val_frac defaults to 0


def test_build_sequences_does_not_cross_player_boundaries():
    X, *_ = build_sequences(training_frame(n_players=2, n_games=12), seq_len=10)
    assert len(X) == 4


def test_build_sequences_opponent_baseline_and_holdout():
    n = SEQ_LEN + 5

    def player(pid, pts):
        return pd.DataFrame({"PLAYER_ID": pid, "PTS": pts, "OPP_DEF_RATING": np.arange(n) + 100.0})

    a, b = player(1, np.arange(n, dtype=float)), player(2, 1000.0)
    X, y, base, is_val = build_sequences(pd.concat([a, b]), val_frac=0.2)

    assert len(y) == 2 * 5
    assert (y[:5] == np.arange(SEQ_LEN, n)).all()  # target = next game, chronological
    assert (y[5:] == 1000).all() and (X[5:, :, 0] == 1000).all()
    assert (X[0, :, -1] == 100 + SEQ_LEN).all()  # opponent feature = target game's opponent
    ratio = np.clip((100 + SEQ_LEN) / LEAGUE_AVG_DEF_RATING, 0.85, 1.15)
    assert base[0] == pytest.approx(np.arange(SEQ_LEN).mean() * ratio)
    assert is_val.tolist() == [False] * 4 + [True] + [False] * 4 + [True]


# ── Normalizer ────────────────────────────────────────────────────────────────


def test_normalizer_constant_column():
    norm = FeatureNormalizer()
    norm.fit(np.array([[1.0, 113.5], [3.0, 113.5]]))
    out = norm.transform(np.array([[2.0, 110.0]]))
    assert np.allclose(out, [[0.0, -3.5]])


def test_normalizer_must_be_fit_before_saving(tmp_path):
    with pytest.raises(ValueError, match="fit before saving"):
        FeatureNormalizer().save(str(tmp_path / "norm.npz"))


def test_opponent_rating_normalizes_to_a_sane_range(model_dir, game_log):
    _, normalizer, _ = train_model(training_frame(), epochs=1, verbose=False)
    seq = predictor._build_inference_sequence(game_log, opponent_def_rating=110.0)
    assert np.abs(normalizer.transform(seq)).max() < 10


# ── Training ──────────────────────────────────────────────────────────────────


def test_train_model_writes_weights_normalizer_and_metrics(model_dir):
    _, _, metrics = train_model(training_frame(), epochs=2, verbose=False)
    assert model_dir.exists()
    assert normalizer_path(model_dir).exists()
    saved = json.loads(metrics_path(model_dir).read_text())
    assert set(metrics) == {"lstm", "baseline"}
    assert set(metrics["lstm"]) == {"mae", "rmse"}
    assert saved["val_games"] == 3 * 2  # ceil(10 windows * 0.2) per player
    assert saved["train_samples"] == 3 * 8


def test_train_model_without_holdout_has_no_metrics(model_dir):
    _, _, metrics = train_model(training_frame(), epochs=1, val_frac=0.0, verbose=False)
    assert metrics == {}


def test_train_model_rejects_too_little_data(model_dir):
    with pytest.raises(ValueError, match="Not enough games"):
        train_model(training_frame(n_players=1, n_games=5), epochs=1, verbose=False)
