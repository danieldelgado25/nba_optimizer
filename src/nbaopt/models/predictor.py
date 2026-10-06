"""
predictor.py
Predicts a fantasy-style performance score for each player using a PyTorch LSTM.

Architecture:
  - Input sequence: last N games, each game encoded as a feature vector
  - LSTM layers learn temporal patterns (hot streaks, fatigue, matchup trends)
  - Opponent DEF_RATING injected as a context feature at inference time
  - Minutes penalty applied post-prediction for low-usage players
  - Falls back to rolling-average baseline if < MIN_GAMES_FOR_MODEL games exist

Drop-in replacement for the statistical predictor.py — predict_player_score()
signature is identical.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from nbaopt.config import model_path, normalizer_path

# ── Constants ─────────────────────────────────────────────────────────────────

LEAGUE_AVG_DEF_RATING = 113.5
MIN_GAMES_FOR_MODEL = 5  # fall back to rolling avg below this
SEQ_LEN = 10  # games fed into LSTM

# Raw stat columns pulled from nba_api game log
STAT_COLS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "MIN", "FG_PCT", "FG3_PCT", "FT_PCT", "PLUS_MINUS"]

# Fantasy scoring weights (used for labels during training & baseline fallback)
SCORE_WEIGHTS = {
    "PTS": 1.0,
    "REB": 1.2,
    "AST": 1.5,
    "STL": 3.0,
    "BLK": 3.0,
    "TOV": -1.0,
}

INPUT_SIZE = len(STAT_COLS) + 1  # stats + opponent DEF_RATING
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2


# ── Fantasy score helper (used for labels + fallback) ─────────────────────────


def compute_fantasy_score(row: pd.Series) -> float:
    score = sum(row.get(s, 0) * w for s, w in SCORE_WEIGHTS.items())
    return round(float(score), 2)


# ── LSTM Model ────────────────────────────────────────────────────────────────


class PlayerLSTM(nn.Module):
    """
    Sequence-to-one LSTM that ingests a window of recent games and
    outputs a single predicted fantasy score.

    Input:  (batch, seq_len, input_size)
    Output: (batch,)
    """

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)
        last_hidden = lstm_out[:, -1, :]  # take final timestep
        dropped = self.dropout(last_hidden)
        return self.head(dropped).squeeze(-1)  # (batch,)


# ── Dataset ───────────────────────────────────────────────────────────────────


class GameSequenceDataset(Dataset):
    """
    Builds (sequence, label) pairs from a DataFrame of game logs.

    Each sample:
      X: (SEQ_LEN, INPUT_SIZE) — window of past games
      y: float — fantasy score of the NEXT game (the one after the window)

    Expects a DataFrame with STAT_COLS + 'OPP_DEF_RATING' + 'FANTASY_SCORE'.
    Games should be ordered oldest -> newest.
    """

    def __init__(self, df: pd.DataFrame, seq_len: int = SEQ_LEN):
        self.seq_len = seq_len
        self.samples: list[tuple[np.ndarray, float]] = []

        feature_cols = STAT_COLS + ["OPP_DEF_RATING"]
        df = df.copy()
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

        if "FANTASY_SCORE" not in df.columns:
            df["FANTASY_SCORE"] = df.apply(compute_fantasy_score, axis=1)

        values = df[feature_cols].values.astype(np.float32)
        labels = df["FANTASY_SCORE"].values.astype(np.float32)

        for i in range(len(df) - seq_len):
            X = values[i : i + seq_len]
            y = labels[i + seq_len]
            self.samples.append((X, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y = self.samples[idx]
        return torch.tensor(X), torch.tensor(y)


# ── Normalization ─────────────────────────────────────────────────────────────


class FeatureNormalizer:
    """
    Per-feature mean/std normalizer.
    Fit on training data, reused at inference.
    Saved alongside model weights as a .npz file.
    """

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def save(self, path: str):
        np.savez(path, mean=self.mean_, std=self.std_)

    @classmethod
    def load(cls, path: str) -> "FeatureNormalizer":
        data = np.load(path)
        n = cls()
        n.mean_ = data["mean"]
        n.std_ = data["std"]
        return n


# ── Training ──────────────────────────────────────────────────────────────────


def train_model(
    game_logs_df: pd.DataFrame,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 32,
    seq_len: int = SEQ_LEN,
    save_path: str | Path | None = None,
    verbose: bool = True,
) -> tuple["PlayerLSTM", "FeatureNormalizer"]:
    """
    Train the LSTM on a historical game-log DataFrame.

    game_logs_df should have columns matching STAT_COLS + 'OPP_DEF_RATING'.
    Rows are individual games sorted oldest -> newest.
    Multiple players' logs can be concatenated together for a shared model.

    Saves:
      save_path           — model weights (.pt), default config.model_path()
      save_path.norm.npz  — normalizer stats

    Returns (model, normalizer).

    Example
    -------
    from data.fetcher import get_player_game_log
    from models.predictor import train_model
    import pandas as pd

    logs = []
    for pid in player_ids:
        df = get_player_game_log(pid, last_n=82)
        df["OPP_DEF_RATING"] = 113.5   # add opponent rating per game if available
        logs.append(df)

    all_logs = pd.concat(logs, ignore_index=True)
    model, normalizer = train_model(all_logs, epochs=30)
    """
    df = game_logs_df.copy()
    feature_cols = STAT_COLS + ["OPP_DEF_RATING"]

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    if "FANTASY_SCORE" not in df.columns:
        df["FANTASY_SCORE"] = df.apply(compute_fantasy_score, axis=1)

    normalizer = FeatureNormalizer()
    df[feature_cols] = normalizer.fit_transform(df[feature_cols].values)

    dataset = GameSequenceDataset(df, seq_len=seq_len)
    if len(dataset) == 0:
        raise ValueError(f"Not enough rows to build sequences. Need > {seq_len} games, got {len(df)}.")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = PlayerLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        if verbose and (epoch % 5 == 0 or epoch == 1):
            avg_loss = epoch_loss / len(loader)
            print(f"  Epoch {epoch:>3}/{epochs}  MSE Loss: {avg_loss:.4f}")

    weights_path = Path(save_path) if save_path else model_path()
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    normalizer.save(str(normalizer_path(weights_path)))
    if verbose:
        print(f"\n  Model saved  → {weights_path}")
        print(f"  Normalizer   → {normalizer_path(weights_path)}")

    return model, normalizer


# ── Inference helpers ─────────────────────────────────────────────────────────


def _load_model_and_normalizer() -> tuple["PlayerLSTM | None", "FeatureNormalizer | None"]:
    """Load saved weights + normalizer. Returns (None, None) if files not found."""
    weights_path = model_path()
    norm_path = normalizer_path(weights_path)
    if not weights_path.exists() or not norm_path.exists():
        return None, None
    model = PlayerLSTM()
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    normalizer = FeatureNormalizer.load(str(norm_path))
    return model, normalizer


def _build_inference_sequence(
    game_log: pd.DataFrame,
    opponent_def_rating: float,
    seq_len: int = SEQ_LEN,
) -> np.ndarray:
    """
    Convert a player's recent game log into a (seq_len, input_size) float32 array.
    Pads with zeros on the left if fewer than seq_len games exist.
    """
    feature_cols = STAT_COLS + ["OPP_DEF_RATING"]
    df = game_log.copy()
    df["OPP_DEF_RATING"] = opponent_def_rating

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    values = df[feature_cols].values[-seq_len:].astype(np.float32)
    if len(values) < seq_len:
        pad = np.zeros((seq_len - len(values), len(feature_cols)), dtype=np.float32)
        values = np.vstack([pad, values])

    return values


def _lstm_predict(
    model: "PlayerLSTM",
    normalizer: "FeatureNormalizer",
    sequence: np.ndarray,
) -> float:
    """Run one (seq_len, input_size) sequence through the LSTM and return a score."""
    normed = normalizer.transform(sequence)
    tensor = torch.tensor(normed).unsqueeze(0)  # (1, seq_len, input_size)
    with torch.no_grad():
        score = model(tensor).item()
    return max(0.0, round(score, 2))


def _rolling_avg_predict(
    game_log: pd.DataFrame,
    opponent_def_rating: float,
) -> float:
    """
    Statistical fallback: rolling averages + opponent DEF_RATING scaling.
    Used when the trained model is not available or data is insufficient.
    """
    stat_cols = [c for c in ["PTS", "REB", "AST", "STL", "BLK", "TOV", "MIN"] if c in game_log.columns]
    numeric = game_log[stat_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    avgs = pd.Series(numeric.mean().to_dict())
    base = compute_fantasy_score(avgs)
    ratio = float(np.clip(opponent_def_rating / LEAGUE_AVG_DEF_RATING, 0.85, 1.15))
    return round(base * ratio, 2)


def _minutes_penalty(score: float, avg_minutes: float) -> float:
    """Scale down predicted score for players averaging under 15 minutes."""
    if avg_minutes >= 15:
        return score
    return round(score * (avg_minutes / 15.0), 2)


# ── Module-level model cache ──────────────────────────────────────────────────
# Loaded once on first call, reused for all subsequent players in the same run.

_MODEL: "PlayerLSTM | None" = None
_NORMALIZER: "FeatureNormalizer | None" = None
_MODEL_LOADED: bool = False


def _ensure_model_loaded():
    global _MODEL, _NORMALIZER, _MODEL_LOADED
    if not _MODEL_LOADED:
        _MODEL, _NORMALIZER = _load_model_and_normalizer()
        _MODEL_LOADED = True


# ── Public API ────────────────────────────────────────────────────────────────


def predict_player_score(
    game_log: pd.DataFrame,
    opponent_def_rating: float = LEAGUE_AVG_DEF_RATING,
    injured: bool = False,
) -> dict:
    """
    Predict the fantasy score for one player's next game.

    This is a drop-in replacement for the original statistical predictor.
    The return dict is identical in structure; 'model_used' is the only
    new key added.

    Decision logic:
      1. injured=True           → return OUT immediately, score = 0
      2. empty game log         → return NO_DATA, score = 0
      3. trained model on disk  → LSTM inference
      4. no model / < 5 games   → rolling-average fallback

    Parameters
    ----------
    game_log : pd.DataFrame
        Recent game log from get_player_game_log(). Oldest row first.
    opponent_def_rating : float
        Defensive rating of the upcoming opponent (from get_team_defensive_rating()).
        Defaults to league average if not provided.
    injured : bool
        Set True to mark this player OUT — skips all computation.

    Returns
    -------
    dict with keys:
        predicted_score   float   Final score after minutes penalty
        base_score        float   Score before minutes penalty
        model_used        str     "lstm" | "rolling_avg_fallback" | "n/a"
        status            str     "ACTIVE" | "OUT" | "NO_DATA"
        avg_minutes       float
        games_sampled     int
        rolling_stats     dict    Per-stat rolling averages (always computed)
    """
    # ── Injury shortcut ───────────────────────────────────────────────────────
    if injured:
        return {
            "predicted_score": 0.0,
            "base_score": 0.0,
            "model_used": "n/a",
            "status": "OUT",
            "avg_minutes": 0,
            "games_sampled": 0,
            "rolling_stats": {},
        }

    # ── No data ───────────────────────────────────────────────────────────────
    if game_log is None or game_log.empty:
        return {
            "predicted_score": 0.0,
            "base_score": 0.0,
            "model_used": "n/a",
            "status": "NO_DATA",
            "avg_minutes": 0,
            "games_sampled": 0,
            "rolling_stats": {},
        }

    # ── Rolling stats (always computed — used for CLI display + fallback) ─────
    display_cols = [c for c in ["PTS", "REB", "AST", "STL", "BLK", "TOV", "MIN"] if c in game_log.columns]
    numeric = game_log[display_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    avgs = numeric.mean().to_dict()
    avg_min = float(avgs.get("MIN", 0))
    n_games = len(game_log)

    # ── Choose prediction method ──────────────────────────────────────────────
    _ensure_model_loaded()

    if n_games >= MIN_GAMES_FOR_MODEL and _MODEL is not None and _NORMALIZER is not None:
        sequence = _build_inference_sequence(game_log, opponent_def_rating)
        base_score = _lstm_predict(_MODEL, _NORMALIZER, sequence)
        method = "lstm"
    else:
        base_score = _rolling_avg_predict(game_log, opponent_def_rating)
        method = "rolling_avg_fallback"

    final_score = _minutes_penalty(base_score, avg_min)

    return {
        "predicted_score": final_score,
        "base_score": base_score,
        "model_used": method,
        "status": "ACTIVE",
        "avg_minutes": round(avg_min, 1),
        "games_sampled": n_games,
        "rolling_stats": {k: round(v, 2) for k, v in avgs.items()},
    }
