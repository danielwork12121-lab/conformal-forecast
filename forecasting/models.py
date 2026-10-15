"""Point-forecast models.

Every model exposes the same minimal interface so the conformal wrapper
and the CLI can treat them interchangeably:

    model.fit(X_train, y_train)          # X: (n, lookback), y: (n, horizon)
    model.predict(X) -> (n, horizon)

Two dependency-free baselines (Naive, SeasonalNaive) are included because
a forecasting benchmark without them is not trustworthy -- any "smart"
model needs to actually beat these to justify its complexity. The
LSTMForecaster is a small from-scratch PyTorch recurrent model, and (v0.20)
TransformerForecaster is a second, architecturally distinct from-scratch
model (self-attention instead of recurrence) added to check whether the
LSTM's small-data weaknesses (see README) are specific to that one
architecture or a general property of neural approaches at this scale.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


class NaiveForecaster:
    """Predicts the last observed value, repeated for the whole horizon."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaiveForecaster":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        horizon = 1
        last = X[:, -1:]
        return np.repeat(last, horizon, axis=1)


class SeasonalNaiveForecaster:
    """Predicts the value observed exactly `period` steps before the target.

    Falls back to the plain last-value naive forecast if the lookback
    window is shorter than `period`.
    """

    def __init__(self, period: int):
        self.period = period

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SeasonalNaiveForecaster":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        lookback = X.shape[1]
        if self.period <= lookback:
            col = X[:, lookback - self.period]
        else:
            col = X[:, -1]
        return col.reshape(-1, 1)


class DeltaWrapper:
    """Wraps any model to predict the *change* from the last observed value
    instead of the absolute level, then adds that last value back at
    prediction time.

    Why this matters: a neural network trained on windows from the first
    60% of a trending series (e.g. airline passenger counts, which roughly
    quadruple from 1949 to 1960) has never seen the absolute *level* of
    values in the held-out test period -- it has to extrapolate, which
    feedforward/recurrent nets are notoriously bad at. The one-step
    *change* between consecutive points, however, has a much more
    stationary distribution even when the level trends -- so a model
    predicting delta-from-last-value only has to interpolate within a
    range it has actually seen, and the trend is reconstructed for free
    by adding it back to the (already known) last observed value. This is
    the neural-forecasting analogue of the "I" (integrated/differencing)
    term in ARIMA.
    """

    def __init__(self, base_model):
        self.base_model = base_model

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DeltaWrapper":
        last = X[:, -1:]
        self.base_model.fit(X, y - last)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        last = X[:, -1:]
        return last + self.base_model.predict(X)

    def __getattr__(self, name):
        # Forward attribute access (e.g. .train_losses) to the wrapped model
        # for introspection/plotting, without re-implementing its whole API.
        return getattr(self.base_model, name)


class LSTMModule(nn.Module):
    def __init__(self, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, 1)
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden)


class LSTMForecaster:
    """A small LSTM regressor trained with Adam + early stopping.

    Inputs/targets are expected to already be normalized by the caller
    (see forecasting.data.Normalizer) -- this class does no scaling of
    its own so that scaling stays a single, auditable step shared by
    every model in a benchmark run.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        num_layers: int = 1,
        lr: float = 1e-2,
        max_epochs: int = 200,
        patience: int = 15,
        seed: int = 0,
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self.model: LSTMModule | None = None
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray, val_frac: float = 0.15) -> "LSTMForecaster":
        torch.manual_seed(self.seed)
        n = X.shape[0]
        n_val = max(1, int(n * val_frac))
        n_train = n - n_val

        X_t = torch.from_numpy(X).float().unsqueeze(-1)  # (n, lookback, 1)
        y_t = torch.from_numpy(y).float()  # (n, horizon)

        X_train, y_train = X_t[:n_train], y_t[:n_train]
        X_val, y_val = X_t[n_train:], y_t[n_train:]

        self.model = LSTMModule(self.hidden_size, self.num_layers)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        best_val = float("inf")
        best_state = None
        epochs_since_improvement = 0

        for _epoch in range(self.max_epochs):
            self.model.train()
            optimizer.zero_grad()
            pred = self.model(X_train)
            loss = loss_fn(pred, y_train)
            loss.backward()
            optimizer.step()
            self.train_losses.append(float(loss.item()))

            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val)
                val_loss = float(loss_fn(val_pred, y_val).item())
            self.val_losses.append(val_loss)

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LSTMForecaster.predict called before fit()")
        self.model.eval()
        X_t = torch.from_numpy(X).float().unsqueeze(-1)
        with torch.no_grad():
            pred = self.model(X_t)
        return pred.numpy()


def _sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """Standard (Vaswani et al. 2017) fixed sinusoidal positional encoding.

    Computed fresh for whatever `seq_len` the input actually has, rather
    than baked in as a fixed-size learned parameter at construction time --
    this keeps TransformerModule, like LSTMModule, agnostic to the
    lookback length used at fit/predict time (a plain learned positional
    *embedding* table would silently break if ever called with a different
    lookback than it was built for).
    """
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe


class TransformerModule(nn.Module):
    def __init__(
        self,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, 1)
        seq_len = x.shape[1]
        pe = _sinusoidal_positional_encoding(seq_len, self.d_model).to(x.device)
        h = self.input_proj(x) + pe  # broadcasts over the batch dim
        out = self.encoder(h)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden)


class TransformerForecaster:
    """A small from-scratch Transformer-encoder regressor, trained with
    Adam + early stopping -- deliberately the same training discipline as
    LSTMForecaster (same optimizer, same loss, same early-stopping rule),
    so a benchmark comparison between the two isolates the *architecture*
    (recurrence vs. self-attention) rather than also varying how each one
    is trained.

    Treats the lookback window as a length-`lookback` sequence of scalar
    tokens, projects each to `d_model`, adds a fixed (non-learned)
    sinusoidal positional encoding, runs it through `num_layers` of
    self-attention, and reads off a point forecast from the final
    position's hidden state -- the same "read off the last position"
    convention LSTMForecaster uses, so the two are as structurally
    comparable as a recurrent vs. attention model can be.

    Inputs/targets are expected to already be normalized by the caller
    (see forecasting.data.Normalizer), exactly like LSTMForecaster.
    """

    def __init__(
        self,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 64,
        lr: float = 1e-2,
        max_epochs: int = 200,
        patience: int = 15,
        seed: int = 0,
    ):
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self.model: TransformerModule | None = None
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray, val_frac: float = 0.15) -> "TransformerForecaster":
        torch.manual_seed(self.seed)
        n = X.shape[0]
        n_val = max(1, int(n * val_frac))
        n_train = n - n_val

        X_t = torch.from_numpy(X).float().unsqueeze(-1)  # (n, lookback, 1)
        y_t = torch.from_numpy(y).float()  # (n, horizon)

        X_train, y_train = X_t[:n_train], y_t[:n_train]
        X_val, y_val = X_t[n_train:], y_t[n_train:]

        self.model = TransformerModule(
            self.d_model, self.nhead, self.num_layers, self.dim_feedforward
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        best_val = float("inf")
        best_state = None
        epochs_since_improvement = 0

        for _epoch in range(self.max_epochs):
            self.model.train()
            optimizer.zero_grad()
            pred = self.model(X_train)
            loss = loss_fn(pred, y_train)
            loss.backward()
            optimizer.step()
            self.train_losses.append(float(loss.item()))

            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val)
                val_loss = float(loss_fn(val_pred, y_val).item())
            self.val_losses.append(val_loss)

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("TransformerForecaster.predict called before fit()")
        self.model.eval()
        X_t = torch.from_numpy(X).float().unsqueeze(-1)
        with torch.no_grad():
            pred = self.model(X_t)
        return pred.numpy()
