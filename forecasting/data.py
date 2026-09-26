"""Dataset loading and windowing utilities.

Two real, public univariate time series (bundled as CSVs in data/) are
supported out of the box:

- airline-passengers.csv: the classic Box & Jenkins monthly airline
  passenger counts, 1949-1960 (144 points). Strong trend + yearly
  seasonality, very little noise.
- daily-min-temperatures.csv: daily minimum temperatures in Melbourne,
  Australia, 1981-1990 (3650 points). Strong yearly seasonality with
  substantial day-to-day noise -- a good stress test for interval
  calibration since point forecasts are much less exact here.

A synthetic generator is also provided for controlled testing of the
conformal-calibration machinery itself (known noise distribution ->
we can sanity-check that empirical coverage tracks the nominal level).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Split:
    """Chronological train/calibration/test split of a 1-D series."""

    train: np.ndarray
    calibration: np.ndarray
    test: np.ndarray

    def full(self) -> np.ndarray:
        return np.concatenate([self.train, self.calibration, self.test])


def load_airline_passengers() -> np.ndarray:
    df = pd.read_csv(DATA_DIR / "airline-passengers.csv")
    return df["Passengers"].to_numpy(dtype=np.float64)


def load_daily_min_temperatures() -> np.ndarray:
    df = pd.read_csv(DATA_DIR / "daily-min-temperatures.csv")
    return df["Temp"].to_numpy(dtype=np.float64)


def generate_synthetic(
    n: int = 800,
    period: int = 24,
    trend_slope: float = 0.05,
    noise_std: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Trend + seasonality + i.i.d. Gaussian noise, with a *known* noise_std.

    Used only to validate the conformal-prediction machinery: since we
    know the true noise distribution, we can check that a nominal (1-alpha)
    interval actually achieves close to (1-alpha) empirical coverage.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    trend = trend_slope * t
    seasonal = 10.0 * np.sin(2 * np.pi * t / period)
    noise = rng.normal(0.0, noise_std, size=n)
    return 50.0 + trend + seasonal + noise


DATASETS = {
    "airline": load_airline_passengers,
    "temperature": load_daily_min_temperatures,
    "synthetic": generate_synthetic,
}


def load_dataset(name: str) -> np.ndarray:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Options: {list(DATASETS)}")
    return DATASETS[name]()


def chronological_split(
    series: np.ndarray, train_frac: float = 0.6, cal_frac: float = 0.2
) -> Split:
    """Split a series into train / calibration / test, in time order.

    Never shuffles -- shuffling a time series before splitting would leak
    future information into training and invalidate both the forecast
    evaluation and the conformal-coverage guarantee (which relies on
    exchangeability of the *calibration* residuals with the *test*
    residuals, not on the series itself being i.i.d. over time).
    """
    n = len(series)
    n_train = int(n * train_frac)
    n_cal = int(n * cal_frac)
    if n_train < 2 or n_cal < 2 or n - n_train - n_cal < 2:
        raise ValueError(f"Series of length {n} too short for this split.")
    return Split(
        train=series[:n_train],
        calibration=series[n_train : n_train + n_cal],
        test=series[n_train + n_cal :],
    )


def make_windows(series: np.ndarray, lookback: int, horizon: int = 1):
    """Slide a fixed-size window over `series`, producing (X, y) pairs.

    X[i] = series[i : i+lookback]
    y[i] = series[i+lookback : i+lookback+horizon]

    Returns float32 numpy arrays shaped (num_windows, lookback) and
    (num_windows, horizon).
    """
    n = len(series)
    num_windows = n - lookback - horizon + 1
    if num_windows <= 0:
        raise ValueError(
            f"Series of length {n} too short for lookback={lookback}, horizon={horizon}"
        )
    X = np.empty((num_windows, lookback), dtype=np.float32)
    y = np.empty((num_windows, horizon), dtype=np.float32)
    for i in range(num_windows):
        X[i] = series[i : i + lookback]
        y[i] = series[i + lookback : i + lookback + horizon]
    return X, y


class Normalizer:
    """Simple z-score normalizer fit on training data only.

    Fitting on train-only (never on calibration/test) avoids leaking
    distributional information from held-out data into the model.
    """

    def __init__(self):
        self.mean_ = 0.0
        self.std_ = 1.0

    def fit(self, x: np.ndarray) -> "Normalizer":
        self.mean_ = float(np.mean(x))
        self.std_ = float(np.std(x)) or 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std_ + self.mean_
