"""Point-forecast models.

Every model exposes the same minimal interface so the conformal wrapper
and the CLI can treat them interchangeably:

    model.fit(X_train, y_train)          # X: (n, lookback), y: (n, horizon)
    model.predict(X) -> (n, horizon)

Two dependency-free baselines (Naive, SeasonalNaive) are included because
a forecasting benchmark without them is not trustworthy -- any "smart"
model needs to actually beat these to justify its complexity. The
LSTMForecaster is a small from-scratch PyTorch recurrent model.
"""
from __future__ import annotations

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
