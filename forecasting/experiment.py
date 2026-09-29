"""End-to-end experiment: load data -> window -> fit models -> evaluate.

This is the one place that wires data.py + models.py + conformal.py +
adaptive.py + metrics.py together, so both the CLI and the test suite
exercise the exact same code path a user would run.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from forecasting.adaptive import (
    AdaptiveConformalForecaster,
    SlidingWindowAdaptiveConformalForecaster,
    evaluate_adaptive_coverage,
)
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


@dataclass
class AdaptiveComparison:
    dataset: str
    alpha: float
    gamma: float
    static: dict  # evaluate_coverage() output for static split conformal
    adaptive: dict  # evaluate_adaptive_coverage() output for ACI
    alpha_t: np.ndarray  # (n_test,) -- ACI's adapted alpha_t over the test window


def _prepare_split(
    dataset: str,
    lookback: int | None,
    period: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Normalizer, int]:
    """Shared data-prep path for both `run_experiment` and
    `run_adaptive_comparison`, so the two can never silently drift apart on
    how a dataset gets split/windowed/normalized.
    """
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

    return X_train, y_train, X_cal, y_cal, X_test, y_test, normalizer, period


def run_experiment(
    dataset: str,
    alpha: float = 0.1,
    lookback: int | None = None,
    period: int | None = None,
    seed: int = 0,
) -> list[ModelResult]:
    X_train, y_train, X_cal, y_cal, X_test, y_test, normalizer, period = _prepare_split(dataset, lookback, period)

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


def run_adaptive_comparison(
    dataset: str,
    alpha: float = 0.1,
    gamma: float = 0.05,
    lookback: int | None = None,
    period: int | None = None,
    seed: int = 0,
) -> AdaptiveComparison:
    """Run the LSTM (delta) model -- the one model in the benchmark that
    static split conformal under-covers on -- through both static split
    conformal and Adaptive Conformal Inference on the *same* fit, split, and
    test window, so the two are a fair, apples-to-apples comparison of the
    calibration method only, not of a different model or a different split.
    """
    X_train, y_train, X_cal, y_cal, X_test, y_test, normalizer, period = _prepare_split(dataset, lookback, period)

    model = DeltaWrapper(LSTMForecaster(hidden_size=32, num_layers=1, seed=seed))
    static = SplitConformalForecaster(model, alpha=alpha).fit(X_train, y_train, X_cal, y_cal)
    static_pred = static.predict(X_test)
    static_result = evaluate_coverage(static_pred, y_test)

    # A second, freshly-initialized model instance for ACI, fit identically,
    # so ACI's result isn't contaminated by any state the static run left on
    # a *shared* model object (there isn't any today, but keeping the two
    # runs fully independent is the honest way to compare them).
    adaptive_model = DeltaWrapper(LSTMForecaster(hidden_size=32, num_layers=1, seed=seed))
    adaptive = AdaptiveConformalForecaster(adaptive_model, alpha=alpha, gamma=gamma).fit(X_train, y_train, X_cal, y_cal)
    adaptive_pred = adaptive.predict_sequential(X_test, y_test.reshape(-1))
    adaptive_result = evaluate_adaptive_coverage(adaptive_pred, y_test)

    return AdaptiveComparison(
        dataset=dataset,
        alpha=alpha,
        gamma=gamma,
        static=static_result,
        adaptive=adaptive_result,
        alpha_t=adaptive_pred.alpha_t,
    )


def format_adaptive_comparison(comparison: AdaptiveComparison) -> str:
    nominal_pct = int(round((1 - comparison.alpha) * 100))
    s, a = comparison.static, comparison.adaptive
    lines = [
        f"### {comparison.dataset}: static split conformal vs. Adaptive Conformal Inference "
        f"(LSTM (delta), nominal {nominal_pct}%, gamma={comparison.gamma})",
        "",
        "| Method | Empirical coverage | Coverage gap | Mean interval width |",
        "|---|---|---|---|",
        f"| Static split conformal | {s['empirical_coverage'] * 100:.1f}% | "
        f"{s['coverage_gap'] * 100:+.1f}pp | {s['mean_interval_width']:.3f} |",
        f"| Adaptive Conformal Inference | {a['empirical_coverage'] * 100:.1f}% | "
        f"{a['coverage_gap'] * 100:+.1f}pp | {a['mean_interval_width']:.3f} |",
    ]
    return "\n".join(lines)


@dataclass
class SlidingWindowComparison:
    dataset: str
    alpha: float
    gamma: float
    window: int | None  # None = unbounded growing buffer
    static: dict  # evaluate_coverage() output for static split conformal
    fixed_pool: dict  # evaluate_adaptive_coverage() output for fixed-pool ACI
    sliding: dict  # evaluate_adaptive_coverage() output for the sliding/growing-pool variant
    pool_size: np.ndarray  # (n_test,) -- sliding variant's pool size at each step


def run_sliding_window_comparison(
    dataset: str,
    alpha: float = 0.1,
    gamma: float = 0.05,
    window: int | None = None,
    lookback: int | None = None,
    period: int | None = None,
    seed: int = 0,
) -> SlidingWindowComparison:
    """Three-way, same-fit comparison: static split conformal, fixed-pool ACI,
    and the sliding/growing-pool ACI variant -- all on the same fitted LSTM
    and the same test window, so any difference is attributable to the
    calibration method alone.

    This is the direct follow-up to `run_adaptive_comparison`'s own airline
    finding: fixed-pool ACI barely moved the airline number because the
    calibration pool's own maximum residual puts a hard ceiling on how wide
    an interval it can ever propose. Folding each step's *observed* residual
    into the pool (this function's `sliding` result) removes that ceiling.
    Whether that actually helps on real data -- and by how much -- is
    measured here, not assumed; see the README for the honest numbers.
    """
    X_train, y_train, X_cal, y_cal, X_test, y_test, normalizer, period = _prepare_split(dataset, lookback, period)

    static_model = DeltaWrapper(LSTMForecaster(hidden_size=32, num_layers=1, seed=seed))
    static = SplitConformalForecaster(static_model, alpha=alpha).fit(X_train, y_train, X_cal, y_cal)
    static_result = evaluate_coverage(static.predict(X_test), y_test)

    # Three independently-initialized model instances, fit identically, so
    # none of the three runs can be contaminated by state a *shared* model
    # object left over from another run (there isn't any today, but keeping
    # them fully independent is the honest way to compare calibration methods
    # rather than accidentally comparing side effects of object reuse).
    fixed_model = DeltaWrapper(LSTMForecaster(hidden_size=32, num_layers=1, seed=seed))
    fixed = AdaptiveConformalForecaster(fixed_model, alpha=alpha, gamma=gamma).fit(X_train, y_train, X_cal, y_cal)
    fixed_pred = fixed.predict_sequential(X_test, y_test.reshape(-1))
    fixed_result = evaluate_adaptive_coverage(fixed_pred, y_test)

    sliding_model = DeltaWrapper(LSTMForecaster(hidden_size=32, num_layers=1, seed=seed))
    sliding = SlidingWindowAdaptiveConformalForecaster(sliding_model, alpha=alpha, gamma=gamma, window=window).fit(
        X_train, y_train, X_cal, y_cal
    )
    sliding_pred = sliding.predict_sequential(X_test, y_test.reshape(-1))
    sliding_result = evaluate_adaptive_coverage(sliding_pred, y_test)

    return SlidingWindowComparison(
        dataset=dataset,
        alpha=alpha,
        gamma=gamma,
        window=window,
        static=static_result,
        fixed_pool=fixed_result,
        sliding=sliding_result,
        pool_size=sliding_pred.pool_size,
    )


def format_sliding_window_comparison(comparison: SlidingWindowComparison) -> str:
    nominal_pct = int(round((1 - comparison.alpha) * 100))
    window_desc = "unbounded (growing buffer)" if comparison.window is None else f"{comparison.window} (sliding)"
    s, f, w = comparison.static, comparison.fixed_pool, comparison.sliding
    lines = [
        f"### {comparison.dataset}: static vs. fixed-pool ACI vs. sliding/growing-pool ACI "
        f"(LSTM (delta), nominal {nominal_pct}%, gamma={comparison.gamma}, window={window_desc})",
        "",
        "| Method | Empirical coverage | Coverage gap | Mean interval width |",
        "|---|---|---|---|",
        f"| Static split conformal | {s['empirical_coverage'] * 100:.1f}% | "
        f"{s['coverage_gap'] * 100:+.1f}pp | {s['mean_interval_width']:.3f} |",
        f"| Fixed-pool ACI | {f['empirical_coverage'] * 100:.1f}% | "
        f"{f['coverage_gap'] * 100:+.1f}pp | {f['mean_interval_width']:.3f} |",
        f"| Sliding/growing-pool ACI | {w['empirical_coverage'] * 100:.1f}% | "
        f"{w['coverage_gap'] * 100:+.1f}pp | {w['mean_interval_width']:.3f} |",
    ]
    return "\n".join(lines)
