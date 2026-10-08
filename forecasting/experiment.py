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


@dataclass
class WindowSweepResult:
    window: int | None  # None = unbounded growing buffer
    sliding: dict  # evaluate_adaptive_coverage() output for this window value


@dataclass
class WindowSweepComparison:
    dataset: str
    alpha: float
    gamma: float
    static: dict  # evaluate_coverage() output for static split conformal (reference line)
    fixed_pool: dict  # evaluate_adaptive_coverage() output for fixed-pool ACI (reference line)
    results: list[WindowSweepResult]  # one per swept window value, in the order given


def run_window_sweep_comparison(
    dataset: str,
    windows: list[int | None],
    alpha: float = 0.1,
    gamma: float = 0.05,
    lookback: int | None = None,
    period: int | None = None,
    seed: int = 0,
) -> WindowSweepComparison:
    """Sweep `window` for `SlidingWindowAdaptiveConformalForecaster` on one
    dataset, holding everything else -- the trained model, the calibration
    residual pool, the test window -- fixed across the sweep.

    `window` is a property of *how the sliding-pool ACI reads its residual
    pool*; it has no effect on model fitting at all. So the LSTM is trained
    exactly once here and its fitted weights + calibration residuals are
    reused for every window value in the sweep, rather than training a fresh
    model per window (which `run_sliding_window_comparison` does, since each
    of its three methods there really is independent). Retraining per window
    would waste `len(windows)` LSTM training runs for no benefit, and worse,
    would let train-time randomness (not `window`) explain part of any
    difference between window values -- the opposite of what a clean
    hyperparameter sweep needs. Sharing one fitted model isolates the effect
    of `window` alone, which is the actual question this function answers.

    `static` and `fixed_pool` are included once, as reference lines, so a
    caller (the CLI / README table) can show the sweep against the two
    baselines from `run_sliding_window_comparison` without re-running them.
    """
    X_train, y_train, X_cal, y_cal, X_test, y_test, normalizer, period = _prepare_split(dataset, lookback, period)

    static_model = DeltaWrapper(LSTMForecaster(hidden_size=32, num_layers=1, seed=seed))
    static = SplitConformalForecaster(static_model, alpha=alpha).fit(X_train, y_train, X_cal, y_cal)
    static_result = evaluate_coverage(static.predict(X_test), y_test)

    fixed_model = DeltaWrapper(LSTMForecaster(hidden_size=32, num_layers=1, seed=seed))
    fixed = AdaptiveConformalForecaster(fixed_model, alpha=alpha, gamma=gamma).fit(X_train, y_train, X_cal, y_cal)
    fixed_pred = fixed.predict_sequential(X_test, y_test.reshape(-1))
    fixed_result = evaluate_adaptive_coverage(fixed_pred, y_test)

    sweep_model = DeltaWrapper(LSTMForecaster(hidden_size=32, num_layers=1, seed=seed))
    fitted = SlidingWindowAdaptiveConformalForecaster(sweep_model, alpha=alpha, gamma=gamma, window=None).fit(
        X_train, y_train, X_cal, y_cal
    )

    results = []
    for window in windows:
        forecaster = SlidingWindowAdaptiveConformalForecaster(fitted.model, alpha=alpha, gamma=gamma, window=window)
        forecaster._residuals = fitted._residuals  # reuse the one fit; window doesn't affect fitting
        pred = forecaster.predict_sequential(X_test, y_test.reshape(-1))
        results.append(WindowSweepResult(window=window, sliding=evaluate_adaptive_coverage(pred, y_test)))

    return WindowSweepComparison(
        dataset=dataset, alpha=alpha, gamma=gamma, static=static_result, fixed_pool=fixed_result, results=results
    )


def format_window_sweep_comparison(comparison: WindowSweepComparison) -> str:
    nominal_pct = int(round((1 - comparison.alpha) * 100))
    s, f = comparison.static, comparison.fixed_pool
    lines = [
        f"### {comparison.dataset}: sliding-pool ACI window sweep "
        f"(LSTM (delta), nominal {nominal_pct}%, gamma={comparison.gamma})",
        "",
        "| Window | Empirical coverage | Coverage gap | Mean interval width |",
        "|---|---|---|---|",
        f"| *static split conformal (reference)* | {s['empirical_coverage'] * 100:.1f}% | "
        f"{s['coverage_gap'] * 100:+.1f}pp | {s['mean_interval_width']:.3f} |",
        f"| *fixed-pool ACI (reference)* | {f['empirical_coverage'] * 100:.1f}% | "
        f"{f['coverage_gap'] * 100:+.1f}pp | {f['mean_interval_width']:.3f} |",
    ]
    for r in comparison.results:
        wlabel = "unbounded (growing buffer)" if r.window is None else str(r.window)
        w = r.sliding
        lines.append(
            f"| {wlabel} | {w['empirical_coverage'] * 100:.1f}% | "
            f"{w['coverage_gap'] * 100:+.1f}pp | {w['mean_interval_width']:.3f} |"
        )
    return "\n".join(lines)


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


# ---------------------------------------------------------------------------
# Automatic window selection (v0.5) -- the README's own "What's next" item:
# "pick the window minimizing |coverage gap| on a held-out slice of the
# calibration data ... instead of requiring a user to eyeball a window-sweep
# plot themselves." v0.4 found real, dataset-dependent optima but left
# finding them to a human. This automates that, and -- per this repo's own
# "measure, don't assert" discipline -- also checks, honestly, whether the
# automatic choice actually generalizes to the real test set, rather than
# just trusting that a held-out-calibration-slice proxy works.
# ---------------------------------------------------------------------------


@dataclass
class WindowSelectionCandidate:
    window: int | None  # None = unbounded growing buffer
    holdout_coverage_gap: float  # signed; measured on the selection-holdout slice ONLY
    holdout_mean_interval_width: float
    test_coverage_gap: float  # signed; the REAL test-set gap for this window -- reported
    # for honest post-hoc comparison only, never used to choose `selected_window`


@dataclass
class AutoWindowResult:
    dataset: str
    alpha: float
    gamma: float
    holdout_frac: float
    n_select_cal: int
    n_select_holdout: int
    candidates: list[WindowSelectionCandidate]  # one per candidate window, in the order given
    selected_window: int | None  # chosen using holdout data only
    best_test_window: int | None  # oracle: smallest |test gap| in hindsight -- comparison only
    static: dict  # evaluate_coverage() output, reference line (from the real test set)
    fixed_pool: dict  # evaluate_adaptive_coverage() output, reference line (from the real test set)
    selected_test_result: dict  # evaluate_adaptive_coverage() output for `selected_window` on the real test set


def _select_best_window(
    windows: list[int | None],
    gap_by_window: dict,
    width_by_window: dict | None = None,
) -> int | None:
    """Pure selection rule, no model/data dependency at all, so the
    selection *logic* can be unit-tested directly and fast, separate from
    the (slow, LSTM-training) integration path that produces the gaps (and
    widths) it's given.

    Primary criterion: argmin |coverage gap| -- the thing that actually
    matters (calibration).

    With a holdout slice small enough to only support a handful of distinct
    outcomes (this repo's own airline/synthetic holdout slices are 8-48
    points -- see the README), several windows routinely tie exactly on
    |gap|. This was checked directly during development (not assumed): on
    `synthetic`, windows 10/20/50/unbounded all tied at the same holdout
    gap, and picking among them mattered a lot for the real test-set
    outcome. Two tie-breaks, in order:
    1. Smaller mean interval width -- the standard efficiency criterion in
       conformal prediction (among equally-calibrated choices, prefer the
       sharper one). Only applied when `width_by_window` is given.
    2. Larger window -- a last-resort, fully deterministic tie-break: fewer
       residuals makes for a noisier quantile estimate, so prefer the option
       that discards less information. `None` (unbounded) is "infinitely
       large" here, the correct direction for this specific tie-break.
    """
    if not windows:
        raise ValueError("Need at least one candidate window to select from")

    def key(w):
        size = float("inf") if w is None else w
        width = width_by_window[w] if width_by_window is not None else 0.0
        return (round(abs(gap_by_window[w]), 10), width, -size)

    return min(windows, key=key)


def run_auto_window_selection(
    dataset: str,
    windows: list[int | None],
    alpha: float = 0.1,
    gamma: float = 0.05,
    holdout_frac: float = 0.3,
    lookback: int | None = None,
    period: int | None = None,
    seed: int = 0,
) -> AutoWindowResult:
    """Automatically pick `window` for `SlidingWindowAdaptiveConformalForecaster`.

    Method: the real calibration set (X_cal, y_cal) -- never the test set --
    is itself split chronologically into a selection-calibration slice (the
    first `1 - holdout_frac`) and a selection-holdout slice (the last
    `holdout_frac`). The LSTM is trained once, on X_train only (window
    selection must not see X_test, and the model doesn't depend on
    calibration data). For each candidate window, the sliding-pool ACI
    forecaster is seeded with ONLY the selection-calibration residuals and
    run sequentially over the selection-holdout slice -- exactly as it would
    later run over the real test set -- and scored by |coverage gap| there.
    `_select_best_window` then picks the window with the smallest |gap| on
    that holdout slice.

    This never touches X_test/y_test to make the selection. X_test is used
    only afterward, to honestly check (via `run_window_sweep_comparison`,
    the same tested machinery v0.4 shipped) whether the holdout-based choice
    actually generalizes -- reported as `test_coverage_gap` per candidate
    and `best_test_window` (the oracle best-in-hindsight), so this feature's
    own claim ("holdout selection finds a good window") is itself measured,
    not assumed, matching every other claim in this repo.

    Once selected, the window is NOT used to retrain the model (window has
    no effect on model fitting, only on how the residual pool is read --
    `run_window_sweep_comparison` already documents the same reasoning);
    `selected_test_result` comes from that function's own run of the
    selected window against the FULL calibration set (X_cal, not just the
    selection-calibration slice), so the deployed forecaster isn't left
    throwing away real calibration data it doesn't need to.
    """
    X_train, y_train, X_cal, y_cal, X_test, y_test, normalizer, period = _prepare_split(dataset, lookback, period)

    if not (0.0 < holdout_frac < 1.0):
        raise ValueError("holdout_frac must be in (0, 1)")

    n_cal = len(X_cal)
    n_select_holdout = int(round(n_cal * holdout_frac))
    n_select_cal = n_cal - n_select_holdout
    if n_select_cal < 5 or n_select_holdout < 5:
        raise ValueError(
            f"Calibration set too small to hold out a selection slice: n_cal={n_cal}, "
            f"holdout_frac={holdout_frac} -> select_cal={n_select_cal}, "
            f"select_holdout={n_select_holdout} (need >= 5 each -- try a smaller holdout_frac "
            "or a dataset/lookback with a larger calibration set)"
        )
    X_select_cal, y_select_cal = X_cal[:n_select_cal], y_cal[:n_select_cal]
    X_select_holdout, y_select_holdout = X_cal[n_select_cal:], y_cal[n_select_cal:]

    base_model = DeltaWrapper(LSTMForecaster(hidden_size=32, num_layers=1, seed=seed))
    base_model.fit(X_train, y_train)
    select_cal_residuals = np.abs(
        y_select_cal.reshape(-1) - base_model.predict(X_select_cal).reshape(-1)
    )

    holdout_gap_by_window: dict = {}
    holdout_width_by_window: dict = {}
    for window in windows:
        forecaster = SlidingWindowAdaptiveConformalForecaster(base_model, alpha=alpha, gamma=gamma, window=window)
        forecaster._residuals = select_cal_residuals
        pred = forecaster.predict_sequential(X_select_holdout, y_select_holdout.reshape(-1))
        result = evaluate_adaptive_coverage(pred, y_select_holdout)
        holdout_gap_by_window[window] = result["coverage_gap"]
        holdout_width_by_window[window] = result["mean_interval_width"]

    selected_window = _select_best_window(windows, holdout_gap_by_window, holdout_width_by_window)

    # Honest post-hoc check, using the already-tested v0.4 sweep machinery:
    # the REAL test-set gap for every candidate. X_test plays no role above
    # -- it's used here only to check, after the fact, whether the
    # holdout-based choice generalized.
    sweep = run_window_sweep_comparison(
        dataset, windows=windows, alpha=alpha, gamma=gamma, lookback=lookback, period=period, seed=seed
    )
    test_gap_by_window = {r.window: r.sliding["coverage_gap"] for r in sweep.results}
    test_width_by_window = {r.window: r.sliding["mean_interval_width"] for r in sweep.results}
    test_result_by_window = {r.window: r.sliding for r in sweep.results}

    candidates = [
        WindowSelectionCandidate(
            window=w,
            holdout_coverage_gap=holdout_gap_by_window[w],
            holdout_mean_interval_width=holdout_width_by_window[w],
            test_coverage_gap=test_gap_by_window[w],
        )
        for w in windows
    ]
    # Oracle/hindsight best, for comparison only -- same tie-break rule,
    # applied to the real test-set numbers instead of the holdout ones.
    best_test_window = _select_best_window(windows, test_gap_by_window, test_width_by_window)

    return AutoWindowResult(
        dataset=dataset,
        alpha=alpha,
        gamma=gamma,
        holdout_frac=holdout_frac,
        n_select_cal=n_select_cal,
        n_select_holdout=n_select_holdout,
        candidates=candidates,
        selected_window=selected_window,
        best_test_window=best_test_window,
        static=sweep.static,
        fixed_pool=sweep.fixed_pool,
        selected_test_result=test_result_by_window[selected_window],
    )


def format_auto_window_selection(result: AutoWindowResult) -> str:
    nominal_pct = int(round((1 - result.alpha) * 100))
    s, f = result.static, result.fixed_pool
    lines = [
        f"### {result.dataset}: automatic window selection "
        f"(LSTM (delta), nominal {nominal_pct}%, gamma={result.gamma}, "
        f"holdout_frac={result.holdout_frac}, n_select_cal={result.n_select_cal}, "
        f"n_select_holdout={result.n_select_holdout})",
        "",
        "| Window | Holdout gap (selection only) | Real test gap (post-hoc check) |",
        "|---|---|---|",
    ]
    for c in result.candidates:
        wlabel = "unbounded (growing buffer)" if c.window is None else str(c.window)
        marker = ""
        if c.window == result.selected_window:
            marker += " **<- selected**"
        if c.window == result.best_test_window:
            marker += " *(best in hindsight)*"
        lines.append(
            f"| {wlabel}{marker} | {c.holdout_coverage_gap * 100:+.1f}pp | {c.test_coverage_gap * 100:+.1f}pp |"
        )
    lines += [
        "",
        f"*Reference, real test set:* static split conformal {s['coverage_gap'] * 100:+.1f}pp, "
        f"fixed-pool ACI {f['coverage_gap'] * 100:+.1f}pp.",
        "",
        f"**Selected window: {('unbounded' if result.selected_window is None else result.selected_window)}** "
        f"(chosen from the holdout column only) -> real test-set gap "
        f"{result.selected_test_result['coverage_gap'] * 100:+.1f}pp, mean interval width "
        f"{result.selected_test_result['mean_interval_width']:.3f}.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rolling-origin cross-validated window selection (v0.6) -- follows up on
# v0.5's own documented limitation: a single holdout slice is, by
# construction, drawn entirely from the calibration period, so on a dataset
# with a small calibration set (airline: 28 windows) every candidate window
# ties exactly on that one slice's hit/miss outcome (see
# tests/test_auto_window_selection.py::test_auto_window_selection_on_airline_is_honestly_a_near_tie_across_candidates).
# Rolling-origin CV does NOT fix that structural problem -- no amount of
# slicing within the calibration period can see a shift that only happens in
# the test period -- but a *single* small holdout slice is also a
# high-variance estimate in its own right: which points happen to land in it
# (entirely decided by one `holdout_frac` cut point) can flip which window
# looks best. Averaging the score over several expanding-window folds is the
# standard time-series fix for exactly that single-split variance. Whether
# it actually selects better windows than v0.5's one-shot holdout is
# measured here, across multiple seeds, not assumed -- see the README for
# the honest (mixed) multi-seed comparison.
# ---------------------------------------------------------------------------


def _make_rolling_folds(n: int, n_folds: int, min_initial: int, min_fold_size: int) -> list[tuple[int, int]] | None:
    """Pure fold-boundary logic, no data/model dependency, so it's tested
    directly and fast, separate from the (slow, LSTM-training) integration
    path that uses it.

    Expanding-window ("blocked", chronological) folds over indices
    `[0, n)`: fold 0's seed is `[0, min_initial)`, its validation chunk is
    the next `fold_size` points; fold 1's seed extends to cover fold 0's
    validation chunk too (expanding, never sliding or shrinking back down --
    more residuals can only help a quantile estimate), and so on. The last
    fold's validation chunk absorbs the remainder of `usable // actual_folds`,
    so every index from `min_initial` to `n` is covered by exactly one
    fold's validation chunk, with no gap and no overlap.

    Returns a list of `(train_end, test_end)` pairs, or `None` if `n` is too
    small to form even one fold of at least `min_fold_size` validation
    points after reserving `min_initial` points for the very first seed.
    `n_folds` is a request, not a guarantee: if the data can't support that
    many folds of `min_fold_size` each, as many as it can support are
    returned instead (always at least 1, or `None`).
    """
    if n_folds < 1:
        raise ValueError("n_folds must be a positive integer")
    if min_initial < 1 or min_fold_size < 1:
        raise ValueError("min_initial and min_fold_size must be positive integers")

    usable = n - min_initial
    if usable < min_fold_size:
        return None

    actual_folds = min(n_folds, usable // min_fold_size)
    fold_size = usable // actual_folds

    folds = []
    start = min_initial
    for k in range(actual_folds):
        end = start + fold_size if k < actual_folds - 1 else n
        folds.append((start, end))
        start = end
    return folds


def _fold_seed_start(train_end: int, min_initial: int, fold_scheme: str) -> int:
    """How much calibration history feeds each fold's conformal-pool seed.

    This is deliberately separate from `_make_rolling_folds`, which only
    decides the (train_end, test_end) validation partition -- that
    partition is identical under both schemes below. This function decides
    where each fold's *seed* residuals start, which is the only thing that
    differs between them:

    - 'expanding' (v0.6's original behavior, still the default): every
      fold's seed is `cal_residuals[0:train_end]` -- it only grows, never
      drops old residuals, so later folds carry every residual since the
      very start of the calibration set forever.
    - 'sliding': every fold's seed is capped at the most recent
      `min_initial` residuals before its own validation chunk
      (`cal_residuals[max(0, train_end - min_initial):train_end]`) -- later
      folds stop carrying arbitrarily old, possibly-stale residuals, at the
      cost of a smaller (and for the early folds, identical) seed than
      'expanding' gets. This is the README's own "What's next" follow-up
      from v0.6: "trying a *sliding* (not just expanding) fold scheme so
      later folds don't carry early, possibly-stale calibration residuals
      forever."

    Whether 'sliding' actually selects better windows than 'expanding' is
    measured, not assumed -- see the README's multi-seed comparison.
    """
    if fold_scheme == "expanding":
        return 0
    if fold_scheme == "sliding":
        return max(0, train_end - min_initial)
    raise ValueError("fold_scheme must be 'expanding' or 'sliding'")


@dataclass
class CVWindowCandidate:
    window: int | None  # None = unbounded growing buffer
    mean_abs_fold_gap: float  # mean of |coverage gap| across folds -- used to select
    mean_fold_gap: float  # mean of SIGNED coverage gap across folds -- display only
    mean_fold_width: float
    test_coverage_gap: float  # real test-set gap, honest post-hoc check only


@dataclass
class CVWindowSelectionResult:
    dataset: str
    alpha: float
    gamma: float
    fold_scheme: str  # 'expanding' (v0.6 default) or 'sliding' (v0.7) -- see _fold_seed_start
    n_folds: int  # actually achieved; may be less than requested, see _make_rolling_folds
    fold_bounds: list[tuple[int, int]]  # (train_end, test_end) indices into the calibration set
    candidates: list[CVWindowCandidate]  # one per candidate window, in the order given
    selected_window: int | None  # chosen using the fold columns only
    best_test_window: int | None  # oracle: smallest |test gap| in hindsight -- comparison only
    static: dict  # evaluate_coverage() output, reference line (from the real test set)
    fixed_pool: dict  # evaluate_adaptive_coverage() output, reference line (from the real test set)
    selected_test_result: dict  # evaluate_adaptive_coverage() output for `selected_window` on the real test set


def run_cv_window_selection(
    dataset: str,
    windows: list[int | None],
    alpha: float = 0.1,
    gamma: float = 0.05,
    n_folds: int = 4,
    min_initial_frac: float = 0.2,
    min_fold_frac: float = 0.1,
    fold_scheme: str = "expanding",
    lookback: int | None = None,
    period: int | None = None,
    seed: int = 0,
) -> CVWindowSelectionResult:
    """Select `window` for the sliding-pool ACI using rolling-origin
    cross-validation over the calibration set, instead of
    `run_auto_window_selection`'s single static holdout slice.

    Method: the LSTM is trained once, on X_train only (identical discipline
    to `run_auto_window_selection` -- selection must never see X_test, and
    the model doesn't depend on calibration data at all). The calibration
    set is partitioned into folds via `_make_rolling_folds` (this
    partition -- which indices are each fold's validation chunk -- is the
    same regardless of `fold_scheme`). What differs by `fold_scheme` is how
    much calibration history feeds each fold's seed residuals, decided by
    `_fold_seed_start`: 'expanding' (the default, v0.6's original behavior)
    seeds every fold from the very start of the calibration set; 'sliding'
    (v0.7) caps each fold's seed at the most recent `min_initial` residuals
    before its own validation chunk, so later folds don't carry arbitrarily
    old residuals forever. For each candidate window, the sliding-pool ACI
    forecaster is seeded accordingly and run sequentially over that fold's
    validation chunk, scored by |coverage gap| there. The per-window score
    is the MEAN of |gap| across folds --
    not the mean of signed gap, so a window that swings over-covered on one
    fold and under-covered on another doesn't look artificially
    well-calibrated on average; it has to be *consistently* close to
    nominal across several different slices of the calibration period to
    win. `_select_best_window` then picks the smallest mean-|gap| window,
    tie-broken by mean interval width (same rule as v0.5) and, failing
    that, the larger window.

    This never touches X_test/y_test to make the selection. X_test is used
    only afterward, via the already-tested `run_window_sweep_comparison`,
    to honestly report whether this selection method's choice generalized
    -- same post-hoc-only discipline as `run_auto_window_selection`.

    Raises ValueError if the calibration set is too small to form even one
    fold of `min_fold_frac * n_cal` points after reserving
    `min_initial_frac * n_cal` points for the first fold's seed.
    """
    X_train, y_train, X_cal, y_cal, X_test, y_test, normalizer, period = _prepare_split(dataset, lookback, period)

    if n_folds < 1:
        raise ValueError("n_folds must be a positive integer")
    if not (0.0 < min_initial_frac < 1.0) or not (0.0 < min_fold_frac < 1.0):
        raise ValueError("min_initial_frac and min_fold_frac must be in (0, 1)")
    if fold_scheme not in ("expanding", "sliding"):
        raise ValueError("fold_scheme must be 'expanding' or 'sliding'")

    n_cal = len(X_cal)
    min_initial = max(5, round(n_cal * min_initial_frac))
    min_fold_size = max(3, round(n_cal * min_fold_frac))
    fold_bounds = _make_rolling_folds(n_cal, n_folds, min_initial, min_fold_size)
    if fold_bounds is None:
        raise ValueError(
            f"Calibration set too small for rolling-origin CV: n_cal={n_cal}, "
            f"min_initial={min_initial}, min_fold_size={min_fold_size} (from min_initial_frac="
            f"{min_initial_frac}, min_fold_frac={min_fold_frac}) -- try a smaller n_folds / "
            "min_initial_frac / min_fold_frac, or a dataset/lookback with a larger calibration set"
        )

    base_model = DeltaWrapper(LSTMForecaster(hidden_size=32, num_layers=1, seed=seed))
    base_model.fit(X_train, y_train)
    cal_residuals = np.abs(y_cal.reshape(-1) - base_model.predict(X_cal).reshape(-1))

    mean_abs_gap_by_window: dict = {}
    mean_gap_by_window: dict = {}
    mean_width_by_window: dict = {}
    for window in windows:
        abs_gaps, signed_gaps, widths = [], [], []
        for train_end, test_end in fold_bounds:
            forecaster = SlidingWindowAdaptiveConformalForecaster(base_model, alpha=alpha, gamma=gamma, window=window)
            seed_start = _fold_seed_start(train_end, min_initial, fold_scheme)
            forecaster._residuals = cal_residuals[seed_start:train_end]
            X_fold, y_fold = X_cal[train_end:test_end], y_cal[train_end:test_end]
            pred = forecaster.predict_sequential(X_fold, y_fold.reshape(-1))
            result = evaluate_adaptive_coverage(pred, y_fold)
            abs_gaps.append(abs(result["coverage_gap"]))
            signed_gaps.append(result["coverage_gap"])
            widths.append(result["mean_interval_width"])
        mean_abs_gap_by_window[window] = float(np.mean(abs_gaps))
        mean_gap_by_window[window] = float(np.mean(signed_gaps))
        mean_width_by_window[window] = float(np.mean(widths))

    selected_window = _select_best_window(windows, mean_abs_gap_by_window, mean_width_by_window)

    # Honest post-hoc check, using the already-tested v0.4 sweep machinery --
    # same pattern as run_auto_window_selection.
    sweep = run_window_sweep_comparison(
        dataset, windows=windows, alpha=alpha, gamma=gamma, lookback=lookback, period=period, seed=seed
    )
    test_gap_by_window = {r.window: r.sliding["coverage_gap"] for r in sweep.results}
    test_width_by_window = {r.window: r.sliding["mean_interval_width"] for r in sweep.results}
    test_result_by_window = {r.window: r.sliding for r in sweep.results}

    candidates = [
        CVWindowCandidate(
            window=w,
            mean_abs_fold_gap=mean_abs_gap_by_window[w],
            mean_fold_gap=mean_gap_by_window[w],
            mean_fold_width=mean_width_by_window[w],
            test_coverage_gap=test_gap_by_window[w],
        )
        for w in windows
    ]
    best_test_window = _select_best_window(windows, test_gap_by_window, test_width_by_window)

    return CVWindowSelectionResult(
        dataset=dataset,
        alpha=alpha,
        gamma=gamma,
        fold_scheme=fold_scheme,
        n_folds=len(fold_bounds),
        fold_bounds=fold_bounds,
        candidates=candidates,
        selected_window=selected_window,
        best_test_window=best_test_window,
        static=sweep.static,
        fixed_pool=sweep.fixed_pool,
        selected_test_result=test_result_by_window[selected_window],
    )


def format_cv_window_selection(result: CVWindowSelectionResult) -> str:
    nominal_pct = int(round((1 - result.alpha) * 100))
    s, f = result.static, result.fixed_pool
    fold_desc = ", ".join(f"[{a}:{b})" for a, b in result.fold_bounds)
    lines = [
        f"### {result.dataset}: rolling-origin CV window selection "
        f"(LSTM (delta), nominal {nominal_pct}%, gamma={result.gamma}, "
        f"fold_scheme={result.fold_scheme}, n_folds={result.n_folds}, folds={fold_desc})",
        "",
        "| Window | Mean &#124;fold gap&#124; (used to select) | Mean fold gap (signed) | Real test gap (post-hoc check) |",
        "|---|---|---|---|",
    ]
    for c in result.candidates:
        wlabel = "unbounded (growing buffer)" if c.window is None else str(c.window)
        marker = ""
        if c.window == result.selected_window:
            marker += " **<- selected**"
        if c.window == result.best_test_window:
            marker += " *(best in hindsight)*"
        lines.append(
            f"| {wlabel}{marker} | {c.mean_abs_fold_gap * 100:.1f}pp | {c.mean_fold_gap * 100:+.1f}pp | "
            f"{c.test_coverage_gap * 100:+.1f}pp |"
        )
    lines += [
        "",
        f"*Reference, real test set:* static split conformal {s['coverage_gap'] * 100:+.1f}pp, "
        f"fixed-pool ACI {f['coverage_gap'] * 100:+.1f}pp.",
        "",
        f"**Selected window: {('unbounded' if result.selected_window is None else result.selected_window)}** "
        f"(chosen from the {result.n_folds} fold columns only) -> real test-set gap "
        f"{result.selected_test_result['coverage_gap'] * 100:+.1f}pp, mean interval width "
        f"{result.selected_test_result['mean_interval_width']:.3f}."
    ]
    return "\n".join(lines)
