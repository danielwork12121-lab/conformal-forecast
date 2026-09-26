"""End-to-end experiment: load data -> window -> fit models -> evaluate.

This is the one place that wires data.py + models.py + conformal.py +
metrics.py together, so both the CLI and the test suite exercise the
exact same code path a user would run.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from forecasting.conformal import SplitConformalForecaster, evaluate_coverage
from forecasting.data import Normalizer, chronological_split, load_dataset, make_windows
from forecasting.metrics import mae, mape, rmse
from forecasting.models import DeltaWrapper, LSTMForecaster, NaiveForecaster, SeasonalNaiveForecaster

DEFAULT_LOOKBACKS = {"airline": 12, "temperature": 30, "synthetic": 24}
DEFAULT_PERIODS = {"airline": 12, "temperature": 365, "synthetic": 24}


@dataclass
class ModelResult:
    name: str
    mae: float
    rmse: float
    mape: float
    coverage: dict = field(default_factory=dict)
    point_pred: np.ndarray | None = None
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    y_true: np.ndarray | None = None


def run_experiment(
    dataset: str,
    alpha: float = 0.1,
    lookback: int | None = None,
    period: int | None = None,
    seed: int = 0,
) -> list[ModelResult]:
    series = load_dataset(dataset)
    lookback = lookback or DEFAULT_LOOKBACKS.get(dataset, 12)
    period = period or DEFAULT_PERIODS.get(dataset, 12)

    split = chronological_split(series, train_frac=0.6, cal_frac=0.2)

    normalizer = Normalizer().fit(split.train)
    norm_series = normalizer.transform(split.full())

    X, y = make_windows(norm_series, lookback=lookback, horizon=1)

    n_train_raw, n_cal_raw = len(split.train), len(split.calibration)
    # Window i uses series[i : i+lookback] to predict series[i+lookback].
    # A window's *target* index in the original series is i + lookback.
    n_train_windows = max(0, n_train_raw - lookback)
    n_cal_windows = n_cal_raw  # calibration windows: targets fall in the cal region

    X_train, y_train = X[:n_train_windows], y[:n_train_windows]
    X_cal, y_cal = X[n_train_windows : n_train_windows + n_cal_windows], y[
        n_train_windows : n_train_windows + n_cal_windows
    ]
    X_test, y_test = X[n_train_windows + n_cal_windows :], y[n_train_windows + n_cal_windows :]

    if len(X_train) < 5 or len(X_cal) < 5 or len(X_test) < 5:
        raise ValueError(
            f"Not enough windows for dataset={dataset} lookback={lookback}: "
            f"train={len(X_train)} cal={len(X_cal)} test={len(X_test)}"
        )

    results = []
    model_factories = {
        "Naive": lambda: NaiveForecaster(),
        "SeasonalNaive": lambda: SeasonalNaiveForecaster(period=period),
        "LSTM (delta)": lambda: DeltaWrapper(LSTMForecaster(hidden_size=32, num_layers=1, seed=seed)),
    }

    for name, factory in model_factories.items():
        model = factory()
        conformal = SplitConformalForecaster(model, alpha=alpha)
        # Naive/SeasonalNaive have no learnable state, but we still fit them
        # on train+cal-appropriate data through the same interface so every
        # model in the benchmark goes through *identical* code.
        conformal.fit(X_train, y_train, X_cal, y_cal)
        pred = conformal.predict(X_test)

        point_denorm = normalizer.inverse_transform(pred.point)
        lower_denorm = normalizer.inverse_transform(pred.lower)
        upper_denorm = normalizer.inverse_transform(pred.upper)
        y_test_denorm = normalizer.inverse_transform(y_test)

        cov = evaluate_coverage(pred, y_test)  # coverage is scale-invariant; use normalized

        results.append(
            ModelResult(
                name=name,
                mae=mae(y_test_denorm, point_denorm),
                rmse=rmse(y_test_denorm, point_denorm),
                mape=mape(y_test_denorm, point_denorm),
                coverage=cov,
                point_pred=point_denorm,
                lower=lower_denorm,
                upper=upper_denorm,
                y_true=y_test_denorm,
            )
        )

    return results


def format_results_table(dataset: str, alpha: float, results: list[ModelResult]) -> str:
    nominal_pct = int(round((1 - alpha) * 100))
    lines = [
        f"### {dataset} (nominal {nominal_pct}% intervals)",
        "",
        "| Model | MAE | RMSE | MAPE (%) | Empirical coverage | Mean interval width |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.name} | {r.mae:.3f} | {r.rmse:.3f} | {r.mape:.2f} | "
            f"{r.coverage['empirical_coverage'] * 100:.1f}% | {r.coverage['mean_interval_width']:.3f} |"
        )
    return "\n".join(lines)
