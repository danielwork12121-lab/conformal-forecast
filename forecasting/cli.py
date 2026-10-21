"""Command-line entry point.

Examples
--------
Run the full benchmark (all bundled datasets, all models) and print
markdown tables:

    python -m forecasting.cli benchmark

Run one dataset and also save a forecast+interval plot:

    python -m forecasting.cli benchmark --dataset airline --plot results/airline.png

Compare static split conformal against Adaptive Conformal Inference on the
LSTM model (the one the plain benchmark shows under-covering) and save a
plot of how ACI's alpha_t adapts over the test window:

    python -m forecasting.cli adaptive --dataset airline --plot results/airline_adaptive.png

Compare static split conformal, fixed-pool ACI, and the sliding/growing-pool
ACI variant (which folds each step's own residual into the pool instead of
staying frozen at the calibration set) and save a plot of the pool growing
over the test window:

    python -m forecasting.cli sliding-window --dataset airline --plot results/airline_sliding_window.png

Sweep the sliding-pool ACI's `window` size and see how coverage gap responds
(a bounded window can beat the unbounded growing buffer -- see the README):

    python -m forecasting.cli window-sweep --dataset airline --windows 10,15,20,30,50,unbounded --plot results/airline_window_sweep.png

Jointly sweep `window` AND `gamma`, aggregated across several seeds, to check
whether the best window actually depends on gamma (the single-`window`-sweep
above holds gamma fixed at its default throughout, so this is the first time
the two are varied together):

    python -m forecasting.cli window-gamma-sweep --dataset airline --windows 10,15,20,30,50,unbounded --gammas 0.01,0.02,0.05,0.1,0.2 --seeds 0,1,2,3,4,5 --plot results/airline_window_gamma_sweep.png

Automatically pick a `window` instead of eyeballing a window-sweep plot: the
calibration set is itself split into a selection-calibration slice and a
held-out slice, each candidate window is scored by |coverage gap| on that
held-out slice only, and the smallest-gap window is selected -- then, as an
honest post-hoc check (never used for the selection itself), the real
test-set gap for every candidate is reported too, so you can see whether the
holdout-based choice actually generalized:

    python -m forecasting.cli auto-window --dataset airline --candidates 5,7,10,15,20,30,50,unbounded --plot results/airline_auto_window.png

Select `window` via rolling-origin cross-validation instead of a single
holdout slice: the calibration set is split into several folds (fold k's
"validation chunk" is a slice of the calibration set; its "seed" is the
residuals feeding the conformal pool before that chunk -- see
`--fold-scheme` below), each candidate window is scored by the MEAN of
|coverage gap| across every fold (not just one slice), and the
smallest-mean-|gap| window is selected -- then, as with `auto-window`, the
real test-set gap for every candidate is reported as an honest post-hoc
check:

    python -m forecasting.cli cv-window-select --dataset airline --candidates 5,7,10,15,20,30,50,unbounded --plot results/airline_cv_window_select.png

`--fold-scheme` controls how much calibration history feeds each fold's
seed: 'expanding' (default) grows it from the very start of the calibration
set every fold; 'sliding' caps it at the most recent `--min-initial-frac`
residuals, so later folds don't carry arbitrarily old residuals forever --
see the README for an honest multi-seed comparison of the two:

    python -m forecasting.cli cv-window-select --dataset airline --candidates 5,7,10,15,20,30,50,unbounded --fold-scheme sliding --plot results/airline_cv_window_select_sliding.png

`--auto-folds` replaces `--n-folds`/`--min-initial-frac`/`--min-fold-frac`
(ignored when this is set) with values chosen automatically from the
calibration set's own size, instead of the same fixed fractions applied
regardless of dataset -- see the README for whether more, data-sized folds
actually select better windows:

    python -m forecasting.cli cv-window-select --dataset temperature --candidates 5,7,10,15,20,30,50,unbounded --auto-folds --plot results/temperature_cv_window_select_auto.png

Check whether `cv-window-select`'s own pick holds up across different model
training seeds, rather than only ever having been checked at seed=0 -- each
seed is a fully independent re-run (a different seed means a different base
LSTM, not just a different reading of an existing one):

    python -m forecasting.cli cv-window-select-seed-check --dataset airline --candidates 5,7,10,15,20,30,50,unbounded --seeds 0,1,2,3,4 --plot results/airline_cv_window_select_seed_check.png

Compare static split conformal, fixed-pool ACI, and EnbPI (a bootstrap
ensemble of independently-trained LSTMs, whose online adaptivity comes from
sliding each member's out-of-bag residuals forward, instead of adapting a
single model's significance level the way ACI does -- see
`forecasting/enbpi.py`) and save a plot of EnbPI's own residual pool size
over the test window:

    python -m forecasting.cli enbpi --dataset airline --plot results/airline_enbpi.png

Sweep EnbPI's own `window` size the same way `window-sweep` already does
for ACI's sliding window (the bootstrap ensemble is trained once and
reused across every value, since window has no effect on fitting -- see
`forecasting/experiment.py`'s `run_enbpi_window_sweep_comparison`):

    python -m forecasting.cli enbpi-window-sweep --dataset airline --windows 5,10,15,20,30,50,unbounded --plot results/airline_enbpi_window_sweep.png

Select EnbPI's own `window` via the same rolling-origin cross-validation
`cv-window-select` already does for ACI's sliding window, instead of
eyeballing an `enbpi-window-sweep` table: the calibration set is split into
several folds, each candidate window is scored by the mean of |coverage
gap| across every fold's residual pool (seeded from EnbPI's own
out-of-bag training residuals plus that fold's calibration prefix -- see
`forecasting/experiment.py`'s `run_enbpi_cv_window_selection`), and the
smallest-mean-|gap| window is selected -- then, as an honest post-hoc
check, the real test-set gap for every candidate is reported too:

    python -m forecasting.cli enbpi-cv-window-select --dataset airline --candidates 5,10,15,20,30,50,unbounded --plot results/airline_enbpi_cv_window_select.png

Sweep the interpretable N-BEATS variant's `num_harmonics` (its seasonality
basis's own hyperparameter), holding `trend_degree` at its shipped default
(2) throughout -- the direct follow-up to the `trend_degree` sweep (v0.25)
`run_nbeats_basis_sweep_comparison`'s own docstring named but did not
attempt (see `forecasting/experiment.py`'s `run_num_harmonics_sweep_comparison`):

    python -m forecasting.cli num-harmonics-sweep --dataset airline --num-harmonics-values 1,2,3,4,5 --plot results/airline_num_harmonics_sweep.png

Jointly sweep the interpretable N-BEATS variant's `trend_degree` AND
`num_harmonics` (v0.28) -- the real follow-up both single-hyperparameter
basis sweeps' own README "What's next" named but neither attempted:
whether the two interact, rather than only ever being measured one at a
time (see `forecasting/experiment.py`'s `run_nbeats_joint_basis_sweep_comparison`):

    python -m forecasting.cli nbeats-joint-basis-sweep --dataset airline --trend-degrees 1,2,3,4,5 --num-harmonics-values 1,2,3,4,5 --plot results/airline_nbeats_joint_basis_sweep.png

Jointly sweep the interpretable N-BEATS variant's stack *depth*,
`num_trend_blocks` AND `num_seasonality_blocks` (v0.29) -- every prior
N-BEATS sweep in this repo (v0.24 through v0.28) varied a basis's own
per-block *shape* (`trend_degree`, `num_harmonics`) but left block *count*
at the constructor default (2, 2) throughout; this is the real follow-up
question neither basis-shape sweep could answer (see
`forecasting/experiment.py`'s `run_nbeats_block_count_sweep_comparison`):

    python -m forecasting.cli nbeats-block-count-sweep --dataset airline --num-trend-blocks-values 1,2,3,4,5 --num-seasonality-blocks-values 1,2,3,4,5 --plot results/airline_nbeats_block_count_sweep.png

Select the interpretable N-BEATS variant's `trend_degree` AND `num_harmonics`
via the same rolling-origin cross-validation `cv-window-select` already does
for ACI's `window` (v0.30) -- every prior basis sweep (v0.25, v0.26, v0.28's
joint grid) picked its "best" cell by reading the real test set directly;
this instead scores each `(trend_degree, num_harmonics)` candidate by the
mean of |coverage gap| across several rolling folds of the calibration set
only, then, as an honest post-hoc check, reports whether the CV-selected
cell actually matches the real test set's own best cell (see
`forecasting/experiment.py`'s `run_nbeats_basis_cv_selection`):

    python -m forecasting.cli nbeats-basis-cv-select --dataset airline --trend-degrees 1,2,3,4,5 --num-harmonics-values 1,2,3,4,5 --plot results/airline_nbeats_basis_cv_select.png

Serve the calibrated forecasts over a minimal REST API (see
`forecasting/api.py` and the README's "Serving it as an API" section --
demo-quality, not production: the LSTM refits per request):

    python -m forecasting.cli serve --port 8000
"""
from __future__ import annotations

import argparse

from forecasting.experiment import (
    format_adaptive_comparison,
    format_auto_window_selection,
    format_cv_window_selection,
    format_cv_window_selection_seed_robustness,
    format_enbpi_comparison,
    format_enbpi_cv_window_selection,
    format_enbpi_window_sweep_comparison,
    format_nbeats_basis_cv_selection,
    format_nbeats_basis_sweep_comparison,
    format_nbeats_block_count_sweep_comparison,
    format_nbeats_joint_basis_sweep_comparison,
    format_num_harmonics_sweep_comparison,
    format_results_table,
    format_sliding_window_comparison,
    format_window_gamma_sweep_comparison,
    format_window_sweep_comparison,
    run_adaptive_comparison,
    run_auto_window_selection,
    run_cv_window_selection,
    run_cv_window_selection_seed_robustness,
    run_enbpi_comparison,
    run_enbpi_cv_window_selection,
    run_enbpi_window_sweep_comparison,
    run_experiment,
    run_nbeats_basis_cv_selection,
    run_nbeats_basis_sweep_comparison,
    run_nbeats_block_count_sweep_comparison,
    run_nbeats_joint_basis_sweep_comparison,
    run_num_harmonics_sweep_comparison,
    run_sliding_window_comparison,
    run_window_gamma_sweep_comparison,
    run_window_sweep_comparison,
)


def _plot(dataset: str, results, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lstm = next(r for r in results if r.name == "LSTM (delta)")
    n = len(lstm.y_true)
    x = range(n)

    nominal_pct = int(round(lstm.coverage["nominal_coverage"] * 100))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, lstm.y_true, label="actual", color="#222222", linewidth=1.5)
    ax.plot(x, lstm.point_pred.reshape(-1), label="LSTM forecast", color="#1f77b4", linewidth=1.2)
    ax.fill_between(
        x,
        lstm.lower.reshape(-1),
        lstm.upper.reshape(-1),
        color="#1f77b4",
        alpha=0.2,
        label=f"{nominal_pct}% conformal interval",
    )
    ax.set_title(
        f"{dataset}: LSTM forecast with split-conformal interval "
        f"(empirical coverage {lstm.coverage['empirical_coverage'] * 100:.1f}%, "
        f"nominal {lstm.coverage['nominal_coverage'] * 100:.0f}%)"
    )
    ax.set_xlabel("test-set time step")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_adaptive(dataset: str, comparison, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alpha_t = comparison.alpha_t
    x = range(len(alpha_t))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, alpha_t, color="#d62728", linewidth=1.3, label="ACI's adapted alpha_t")
    ax.axhline(comparison.alpha, color="#222222", linewidth=1.0, linestyle="--", label=f"nominal alpha={comparison.alpha}")
    ax.set_title(
        f"{dataset}: ACI's miscoverage rate over the test window "
        f"(static coverage {comparison.static['empirical_coverage'] * 100:.1f}% -> "
        f"adaptive {comparison.adaptive['empirical_coverage'] * 100:.1f}%, nominal "
        f"{int(round((1 - comparison.alpha) * 100))}%)"
    )
    ax.set_xlabel("test-set time step")
    ax.set_ylabel("alpha_t (lower = wider interval)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_sliding_window(dataset: str, comparison, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pool_size = comparison.pool_size
    x = range(len(pool_size))

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(x, pool_size, color="#2ca02c", linewidth=1.5, label="residual pool size")
    ax1.set_xlabel("test-set time step")
    ax1.set_ylabel("pool size (# residuals)", color="#2ca02c")
    ax1.tick_params(axis="y", labelcolor="#2ca02c")
    window_desc = "unbounded" if comparison.window is None else str(comparison.window)
    s, f, w = comparison.static, comparison.fixed_pool, comparison.sliding
    ax1.set_title(
        f"{dataset}: sliding-pool ACI's residual pool size over the test window (window={window_desc})\n"
        f"coverage -- static {s['empirical_coverage'] * 100:.1f}%, fixed-pool ACI {f['empirical_coverage'] * 100:.1f}%, "
        f"sliding-pool ACI {w['empirical_coverage'] * 100:.1f}% (nominal {int(round((1 - comparison.alpha) * 100))}%)",
        fontsize=10,
    )
    ax1.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_enbpi(dataset: str, comparison, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pool_size = comparison.pool_size
    x = range(len(pool_size))

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(x, pool_size, color="#9467bd", linewidth=1.5, label="EnbPI residual pool size")
    ax1.set_xlabel("test-set time step")
    ax1.set_ylabel("pool size (# residuals)", color="#9467bd")
    ax1.tick_params(axis="y", labelcolor="#9467bd")
    window_desc = "unbounded" if comparison.window is None else str(comparison.window)
    s, f, e = comparison.static, comparison.fixed_pool, comparison.enbpi
    ax1.set_title(
        f"{dataset}: EnbPI's residual pool size over the test window "
        f"(n_bootstrap={comparison.n_bootstrap}, window={window_desc})\n"
        f"coverage -- static {s['empirical_coverage'] * 100:.1f}%, fixed-pool ACI {f['empirical_coverage'] * 100:.1f}%, "
        f"EnbPI {e['empirical_coverage'] * 100:.1f}% (nominal {int(round((1 - comparison.alpha) * 100))}%)",
        fontsize=10,
    )
    ax1.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_enbpi_window_sweep(dataset: str, comparison, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    windows = [r.window for r in comparison.results]
    gaps = [r.enbpi["coverage_gap"] * 100 for r in comparison.results]
    x = range(len(windows))
    labels = ["unbounded" if w is None else str(w) for w in windows]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, gaps, color="#9467bd", linewidth=1.5, marker="o", label="EnbPI")
    ax.axhline(
        comparison.fixed_pool["coverage_gap"] * 100,
        color="#1f77b4",
        linewidth=1.2,
        linestyle="--",
        label="fixed-pool ACI (window-independent)",
    )
    ax.axhline(
        comparison.static["coverage_gap"] * 100,
        color="#888888",
        linewidth=1.0,
        linestyle=":",
        label="static split conformal",
    )
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel("window (max residual-pool size)")
    ax.set_ylabel("coverage gap (pp, closer to 0 is better)")
    ax.set_title(f"{dataset}: EnbPI coverage gap by window size")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_window_sweep(dataset: str, comparison, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    windows = [r.window for r in comparison.results]
    gaps = [r.sliding["coverage_gap"] * 100 for r in comparison.results]
    # Plot against an evenly-spaced index, not the raw window values, so an
    # "unbounded" entry (window=None) doesn't have to pretend to be a number
    # on the x-axis -- it's drawn as its own labeled tick instead.
    x = range(len(windows))
    labels = ["unbounded" if w is None else str(w) for w in windows]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, gaps, color="#2ca02c", linewidth=1.5, marker="o", label="sliding-pool ACI")
    ax.axhline(
        comparison.fixed_pool["coverage_gap"] * 100,
        color="#1f77b4",
        linewidth=1.2,
        linestyle="--",
        label="fixed-pool ACI (window-independent)",
    )
    ax.axhline(
        comparison.static["coverage_gap"] * 100,
        color="#888888",
        linewidth=1.0,
        linestyle=":",
        label="static split conformal",
    )
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel("window (max residual-pool size)")
    ax.set_ylabel("coverage gap (pp, closer to 0 is better)")
    ax.set_title(f"{dataset}: sliding-pool ACI coverage gap by window size")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_window_gamma_sweep(dataset: str, comparison, path: str) -> None:
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    windows = comparison.windows
    gammas = comparison.gammas
    wlabels = ["unbounded" if w is None else str(w) for w in windows]
    grid = np.zeros((len(windows), len(gammas)))
    cell_by_pos = {(c.window, c.gamma): c for c in comparison.cells}
    for i, w in enumerate(windows):
        for j, g in enumerate(gammas):
            grid[i, j] = cell_by_pos[(w, g)].mean_abs_gap * 100

    fig, ax = plt.subplots(figsize=(1.6 * len(gammas) + 2, 1.1 * len(windows) + 2))
    im = ax.imshow(grid, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(gammas)))
    ax.set_xticklabels([str(g) for g in gammas])
    ax.set_yticks(range(len(windows)))
    ax.set_yticklabels(wlabels)
    ax.set_xlabel("gamma (ACI step size)")
    ax.set_ylabel("window (max residual-pool size)")
    ax.set_title(f"{dataset}: window x gamma joint sweep, mean |coverage gap| (pp)\n{len(comparison.seeds)} seeds")
    for i in range(len(windows)):
        for j in range(len(gammas)):
            ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="mean |coverage gap| (pp)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_nbeats_basis_sweep(dataset: str, comparison, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    degrees = [r.trend_degree for r in comparison.results]
    gaps = [r.interpretable["coverage_gap"] * 100 for r in comparison.results]
    x = range(len(degrees))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, gaps, color="#d62728", linewidth=1.5, marker="o", label="interpretable N-BEATS")
    ax.axhline(
        comparison.generic["coverage_gap"] * 100,
        color="#888888",
        linewidth=1.0,
        linestyle=":",
        label="generic N-BEATS (reference)",
    )
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(d) for d in degrees])
    if 2 in degrees:
        ax.axvline(degrees.index(2), color="#9467bd", linewidth=0.8, linestyle="--", label="shipped default (degree=2)")
    ax.set_xlabel("trend_degree")
    ax.set_ylabel("coverage gap (pp, closer to 0 is better)")
    ax.set_title(f"{dataset}: interpretable N-BEATS coverage gap by trend_degree")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_num_harmonics_sweep(dataset: str, comparison, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    harmonics = [r.num_harmonics for r in comparison.results]
    gaps = [r.interpretable["coverage_gap"] * 100 for r in comparison.results]
    x = range(len(harmonics))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, gaps, color="#1f77b4", linewidth=1.5, marker="o", label="interpretable N-BEATS")
    ax.axhline(
        comparison.generic["coverage_gap"] * 100,
        color="#888888",
        linewidth=1.0,
        linestyle=":",
        label="generic N-BEATS (reference)",
    )
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(h) for h in harmonics])
    if comparison.shipped_default in harmonics:
        ax.axvline(
            harmonics.index(comparison.shipped_default),
            color="#9467bd",
            linewidth=0.8,
            linestyle="--",
            label=f"shipped default (num_harmonics={comparison.shipped_default})",
        )
    ax.set_xlabel("num_harmonics")
    ax.set_ylabel("coverage gap (pp, closer to 0 is better)")
    ax.set_title(f"{dataset}: interpretable N-BEATS coverage gap by num_harmonics")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_nbeats_joint_basis_sweep(dataset: str, comparison, path: str) -> None:
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    degrees = comparison.trend_degrees
    harmonics = comparison.num_harmonics_values
    grid = np.zeros((len(degrees), len(harmonics)))
    cell_by_pos = {(c.trend_degree, c.num_harmonics): c for c in comparison.cells}
    for i, d in enumerate(degrees):
        for j, h in enumerate(harmonics):
            grid[i, j] = abs(cell_by_pos[(d, h)].coverage_gap) * 100

    fig, ax = plt.subplots(figsize=(1.6 * len(harmonics) + 2, 1.1 * len(degrees) + 2))
    im = ax.imshow(grid, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(harmonics)))
    ax.set_xticklabels([str(h) for h in harmonics])
    ax.set_yticks(range(len(degrees)))
    ax.set_yticklabels([str(d) for d in degrees])
    ax.set_xlabel("num_harmonics")
    ax.set_ylabel("trend_degree")
    ax.set_title(f"{dataset}: trend_degree x num_harmonics joint sweep, |coverage gap| (pp)\nseed={comparison.seed}")
    for i in range(len(degrees)):
        for j in range(len(harmonics)):
            ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="|coverage gap| (pp)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_nbeats_block_count_sweep(dataset: str, comparison, path: str) -> None:
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_trend_values = comparison.num_trend_blocks_values
    n_season_values = comparison.num_seasonality_blocks_values
    grid = np.zeros((len(n_trend_values), len(n_season_values)))
    cell_by_pos = {(c.num_trend_blocks, c.num_seasonality_blocks): c for c in comparison.cells}
    for i, nt in enumerate(n_trend_values):
        for j, ns in enumerate(n_season_values):
            grid[i, j] = abs(cell_by_pos[(nt, ns)].coverage_gap) * 100

    fig, ax = plt.subplots(figsize=(1.6 * len(n_season_values) + 2, 1.1 * len(n_trend_values) + 2))
    im = ax.imshow(grid, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(n_season_values)))
    ax.set_xticklabels([str(n) for n in n_season_values])
    ax.set_yticks(range(len(n_trend_values)))
    ax.set_yticklabels([str(n) for n in n_trend_values])
    ax.set_xlabel("num_seasonality_blocks")
    ax.set_ylabel("num_trend_blocks")
    ax.set_title(
        f"{dataset}: num_trend_blocks x num_seasonality_blocks joint sweep, |coverage gap| (pp)\n"
        f"trend_degree={comparison.trend_degree}, num_harmonics={comparison.num_harmonics} "
        f"(both shipped defaults), seed={comparison.seed}"
    )
    for i in range(len(n_trend_values)):
        for j in range(len(n_season_values)):
            ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="|coverage gap| (pp)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_nbeats_basis_cv_selection(dataset: str, result, path: str) -> None:
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    degrees = result.trend_degrees
    harmonics_values = result.num_harmonics_values
    fold_grid = np.zeros((len(degrees), len(harmonics_values)))
    test_grid = np.zeros((len(degrees), len(harmonics_values)))
    by_key = {(c.trend_degree, c.num_harmonics): c for c in result.candidates}
    for i, d in enumerate(degrees):
        for j, h in enumerate(harmonics_values):
            c = by_key[(d, h)]
            fold_grid[i, j] = c.mean_abs_fold_gap * 100
            test_grid[i, j] = abs(c.test_coverage_gap) * 100

    fig, axes = plt.subplots(1, 2, figsize=(2.2 * len(harmonics_values) + 4, 1.1 * len(degrees) + 2))
    for ax, grid, title in (
        (axes[0], fold_grid, f"CV: mean |fold gap| (used to select)"),
        (axes[1], test_grid, "Real test-set |gap| (post-hoc check)"),
    ):
        im = ax.imshow(grid, cmap="viridis_r", aspect="auto")
        ax.set_xticks(range(len(harmonics_values)))
        ax.set_xticklabels([str(h) for h in harmonics_values])
        ax.set_yticks(range(len(degrees)))
        ax.set_yticklabels([str(d) for d in degrees])
        ax.set_xlabel("num_harmonics")
        ax.set_ylabel("trend_degree")
        ax.set_title(title, fontsize=10)
        for i in range(len(degrees)):
            for j in range(len(harmonics_values)):
                ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, label="|gap| (pp)")

    sel_i, sel_j = degrees.index(result.selected[0]), harmonics_values.index(result.selected[1])
    axes[0].add_patch(plt.Rectangle((sel_j - 0.5, sel_i - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2.5))
    axes[1].add_patch(plt.Rectangle((sel_j - 0.5, sel_i - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2.5))
    best_i, best_j = degrees.index(result.best_test[0]), harmonics_values.index(result.best_test[1])
    axes[1].add_patch(plt.Rectangle((best_j - 0.5, best_i - 0.5), 1, 1, fill=False, edgecolor="orange", linewidth=2.5, linestyle="--"))

    folds_label = f"{result.n_folds} folds, auto" if result.auto_folds else f"{result.n_folds} folds"
    fig.suptitle(
        f"{dataset}: trend_degree x num_harmonics CV selection ({result.fold_scheme}, {folds_label})\n"
        f"red box = selected (trend_degree={result.selected[0]}, num_harmonics={result.selected[1]}); "
        f"dashed orange = best in hindsight (trend_degree={result.best_test[0]}, num_harmonics={result.best_test[1]})",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_auto_window_selection(dataset: str, result, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    windows = [c.window for c in result.candidates]
    holdout_gaps = [c.holdout_coverage_gap * 100 for c in result.candidates]
    test_gaps = [c.test_coverage_gap * 100 for c in result.candidates]
    x = list(range(len(windows)))
    labels = ["unbounded" if w is None else str(w) for w in windows]
    selected_idx = windows.index(result.selected_window)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, holdout_gaps, color="#9467bd", linewidth=1.5, marker="o", label="holdout gap (used to select)")
    ax.plot(x, test_gaps, color="#2ca02c", linewidth=1.5, marker="s", label="real test-set gap (post-hoc check)")
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.axvline(selected_idx, color="#d62728", linewidth=1.0, linestyle="--", label=f"selected (window={labels[selected_idx]})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("window (max residual-pool size)")
    ax.set_ylabel("coverage gap (pp, closer to 0 is better)")
    ax.set_title(
        f"{dataset}: automatic window selection -- holdout-based choice vs. real test-set outcome\n"
        f"selected window={labels[selected_idx]}, real test-set gap "
        f"{result.selected_test_result['coverage_gap'] * 100:+.1f}pp",
        fontsize=10,
    )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_cv_window_selection(dataset: str, result, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    windows = [c.window for c in result.candidates]
    fold_gaps = [c.mean_abs_fold_gap * 100 for c in result.candidates]
    test_gaps = [c.test_coverage_gap * 100 for c in result.candidates]
    x = list(range(len(windows)))
    labels = ["unbounded" if w is None else str(w) for w in windows]
    selected_idx = windows.index(result.selected_window)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, fold_gaps, color="#9467bd", linewidth=1.5, marker="o", label=f"mean |gap| across {result.n_folds} folds (used to select)")
    ax.plot(x, test_gaps, color="#2ca02c", linewidth=1.5, marker="s", label="real test-set gap (post-hoc check)")
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.axvline(selected_idx, color="#d62728", linewidth=1.0, linestyle="--", label=f"selected (window={labels[selected_idx]})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("window (max residual-pool size)")
    ax.set_ylabel("coverage gap (pp; fold column is |gap|, test column is signed)")
    folds_label = f"{result.n_folds} folds, auto" if result.auto_folds else f"{result.n_folds} folds"
    ax.set_title(
        f"{dataset}: rolling-origin CV window selection ({result.fold_scheme}, {folds_label}) "
        f"vs. real test-set outcome\n"
        f"selected window={labels[selected_idx]}, real test-set gap "
        f"{result.selected_test_result['coverage_gap'] * 100:+.1f}pp",
        fontsize=10,
    )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_cv_window_selection_seed_robustness(dataset: str, result, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    windows_order = [w for w in result.windows]
    labels = ["unbounded" if w is None else str(w) for w in windows_order]
    x = list(range(len(result.per_seed)))
    picked_idx = [windows_order.index(c.selected_window) for c in result.per_seed]
    test_gaps = [c.selected_test_gap * 100 for c in result.per_seed]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    ax1.scatter(x, picked_idx, color="#d62728", marker="D", s=60)
    ax1.set_yticks(range(len(windows_order)))
    ax1.set_yticklabels(labels)
    ax1.set_ylabel("window picked\nby fold metric")
    ax1.set_title(
        f"{dataset}: is cv-window-select's pick seed-robust? ({len(result.per_seed)} seeds)\n"
        + ("robust: every seed picked the same window" if result.is_window_robust else "NOT robust: picks differ by seed"),
        fontsize=10,
    )

    ax2.axhline(0.0, color="#222222", linewidth=0.8)
    ax2.plot(x, test_gaps, color="#2ca02c", linewidth=1.5, marker="s")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(c.seed) for c in result.per_seed])
    ax2.set_xlabel("model training seed")
    ax2.set_ylabel("real test-set gap (pp)\nfor that seed's pick")

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_enbpi_cv_window_selection(dataset: str, result, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    windows = [c.window for c in result.candidates]
    fold_gaps = [c.mean_abs_fold_gap * 100 for c in result.candidates]
    test_gaps = [c.test_coverage_gap * 100 for c in result.candidates]
    x = list(range(len(windows)))
    labels = ["unbounded" if w is None else str(w) for w in windows]
    selected_idx = windows.index(result.selected_window)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(
        x,
        fold_gaps,
        color="#9467bd",
        linewidth=1.5,
        marker="o",
        label=f"mean |gap| across {result.n_folds} folds (used to select)",
    )
    ax.plot(x, test_gaps, color="#2ca02c", linewidth=1.5, marker="s", label="real test-set gap (post-hoc check)")
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.axvline(selected_idx, color="#d62728", linewidth=1.0, linestyle="--", label=f"selected (window={labels[selected_idx]})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("window (max residual-pool size)")
    ax.set_ylabel("coverage gap (pp; fold column is |gap|, test column is signed)")
    folds_label = f"{result.n_folds} folds, auto" if result.auto_folds else f"{result.n_folds} folds"
    ax.set_title(
        f"{dataset}: EnbPI rolling-origin CV window selection ({result.fold_scheme}, {folds_label}) "
        f"vs. real test-set outcome\n"
        f"selected window={labels[selected_idx]}, real test-set gap "
        f"{result.selected_test_result['coverage_gap'] * 100:+.1f}pp",
        fontsize=10,
    )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="conformal-forecast benchmark runner")
    sub = parser.add_subparsers(dest="command", required=True)

    bench = sub.add_parser("benchmark", help="run the model benchmark on one or all datasets")
    bench.add_argument(
        "--dataset",
        choices=["airline", "temperature", "synthetic", "all"],
        default="all",
    )
    bench.add_argument("--alpha", type=float, default=0.1, help="miscoverage rate (default 0.1 -> 90%% intervals)")
    bench.add_argument("--plot", type=str, default=None, help="save a forecast+interval PNG for --dataset (not 'all')")
    bench.add_argument("--seed", type=int, default=0)

    adaptive = sub.add_parser(
        "adaptive",
        help="compare static split conformal vs. Adaptive Conformal Inference on the LSTM model",
    )
    adaptive.add_argument("--dataset", choices=["airline", "temperature", "synthetic"], default="airline")
    adaptive.add_argument("--alpha", type=float, default=0.1)
    adaptive.add_argument("--gamma", type=float, default=0.05, help="ACI step size")
    adaptive.add_argument("--plot", type=str, default=None, help="save a plot of alpha_t over the test window")
    adaptive.add_argument("--seed", type=int, default=0)

    sliding = sub.add_parser(
        "sliding-window",
        help="compare static split conformal, fixed-pool ACI, and sliding/growing-pool ACI on the LSTM model",
    )
    sliding.add_argument("--dataset", choices=["airline", "temperature", "synthetic"], default="airline")
    sliding.add_argument("--alpha", type=float, default=0.1)
    sliding.add_argument("--gamma", type=float, default=0.05, help="ACI step size")
    sliding.add_argument(
        "--window",
        type=int,
        default=None,
        help="max residual-pool size (default: unbounded growing buffer)",
    )
    sliding.add_argument("--plot", type=str, default=None, help="save a plot of the residual pool size over the test window")
    sliding.add_argument("--seed", type=int, default=0)

    enbpi = sub.add_parser(
        "enbpi",
        help="compare static split conformal, fixed-pool ACI, and EnbPI (bootstrap-ensemble conformal) on the LSTM model",
    )
    enbpi.add_argument("--dataset", choices=["airline", "temperature", "synthetic"], default="airline")
    enbpi.add_argument("--alpha", type=float, default=0.1)
    enbpi.add_argument("--gamma", type=float, default=0.05, help="fixed-pool ACI's step size (reference line only)")
    enbpi.add_argument("--n-bootstrap", type=int, default=20, help="number of bootstrap ensemble members")
    enbpi.add_argument(
        "--window",
        type=int,
        default=None,
        help="max residual-pool size (default: unbounded growing buffer)",
    )
    enbpi.add_argument("--plot", type=str, default=None, help="save a plot of EnbPI's residual pool size over the test window")
    enbpi.add_argument("--seed", type=int, default=0)

    enbpi_sweep = sub.add_parser(
        "enbpi-window-sweep",
        help="sweep EnbPI's window size and compare coverage gap across values (mirrors window-sweep for ACI)",
    )
    enbpi_sweep.add_argument("--dataset", choices=["airline", "temperature", "synthetic"], default="airline")
    enbpi_sweep.add_argument("--alpha", type=float, default=0.1)
    enbpi_sweep.add_argument("--gamma", type=float, default=0.05, help="fixed-pool ACI's step size (reference line only)")
    enbpi_sweep.add_argument("--n-bootstrap", type=int, default=20, help="number of bootstrap ensemble members")
    enbpi_sweep.add_argument(
        "--windows",
        type=str,
        default="5,10,15,20,30,50,unbounded",
        help="comma-separated list of window sizes to try; include 'unbounded' for the growing buffer",
    )
    enbpi_sweep.add_argument("--plot", type=str, default=None, help="save a plot of coverage gap vs. window size")
    enbpi_sweep.add_argument("--seed", type=int, default=0)

    sweep = sub.add_parser(
        "window-sweep",
        help="sweep the sliding-pool ACI's window size and compare coverage gap across values",
    )
    sweep.add_argument("--dataset", choices=["airline", "temperature", "synthetic"], default="airline")
    sweep.add_argument("--alpha", type=float, default=0.1)
    sweep.add_argument("--gamma", type=float, default=0.05, help="ACI step size")
    sweep.add_argument(
        "--windows",
        type=str,
        default="10,15,20,30,50,unbounded",
        help="comma-separated list of window sizes to try; include 'unbounded' for the growing buffer",
    )
    sweep.add_argument("--plot", type=str, default=None, help="save a plot of coverage gap vs. window size")
    sweep.add_argument("--seed", type=int, default=0)

    gamma_sweep = sub.add_parser(
        "window-gamma-sweep",
        help="jointly sweep the sliding-pool ACI's window AND gamma, aggregated across several seeds "
        "(window-sweep's own 'What's next' follow-up: window and gamma were never swept together before)",
    )
    gamma_sweep.add_argument("--dataset", choices=["airline", "temperature", "synthetic"], default="airline")
    gamma_sweep.add_argument("--alpha", type=float, default=0.1)
    gamma_sweep.add_argument(
        "--windows",
        type=str,
        default="10,15,20,30,50,unbounded",
        help="comma-separated list of window sizes to try; include 'unbounded' for the growing buffer",
    )
    gamma_sweep.add_argument(
        "--gammas",
        type=str,
        default="0.01,0.02,0.05,0.1,0.2",
        help="comma-separated list of ACI step sizes to try",
    )
    gamma_sweep.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4,5",
        help="comma-separated list of seeds to aggregate across (matches this repo's own 6-seed README convention)",
    )
    gamma_sweep.add_argument("--plot", type=str, default=None, help="save a window x gamma heatmap of mean |gap|")

    nbeats_sweep = sub.add_parser(
        "nbeats-basis-sweep",
        help="sweep the interpretable N-BEATS trend basis's polynomial degree and compare coverage/MAE across values",
    )
    nbeats_sweep.add_argument("--dataset", choices=["airline", "temperature", "synthetic"], default="airline")
    nbeats_sweep.add_argument("--alpha", type=float, default=0.1)
    nbeats_sweep.add_argument(
        "--trend-degrees",
        type=str,
        default="1,2,3,4,5",
        help="comma-separated list of trend_degree values to try (num_harmonics stays at its default throughout)",
    )
    nbeats_sweep.add_argument("--plot", type=str, default=None, help="save a plot of coverage gap vs. trend_degree")
    nbeats_sweep.add_argument("--seed", type=int, default=0)

    harmonics_sweep = sub.add_parser(
        "num-harmonics-sweep",
        help="sweep the interpretable N-BEATS seasonality basis's num_harmonics and compare coverage/MAE across values",
    )
    harmonics_sweep.add_argument("--dataset", choices=["airline", "temperature", "synthetic"], default="airline")
    harmonics_sweep.add_argument("--alpha", type=float, default=0.1)
    harmonics_sweep.add_argument(
        "--num-harmonics-values",
        type=str,
        default="1,2,3,4,5",
        help="comma-separated list of num_harmonics values to try (trend_degree stays at its shipped default, 2, throughout)",
    )
    harmonics_sweep.add_argument("--plot", type=str, default=None, help="save a plot of coverage gap vs. num_harmonics")
    harmonics_sweep.add_argument("--seed", type=int, default=0)

    joint_basis_sweep = sub.add_parser(
        "nbeats-joint-basis-sweep",
        help="jointly sweep the interpretable N-BEATS variant's trend_degree AND num_harmonics "
        "(the real follow-up neither single-hyperparameter basis sweep attempted: whether they interact)",
    )
    joint_basis_sweep.add_argument("--dataset", choices=["airline", "temperature", "synthetic"], default="airline")
    joint_basis_sweep.add_argument("--alpha", type=float, default=0.1)
    joint_basis_sweep.add_argument(
        "--trend-degrees",
        type=str,
        default="1,2,3,4,5",
        help="comma-separated list of trend_degree values to try",
    )
    joint_basis_sweep.add_argument(
        "--num-harmonics-values",
        type=str,
        default="1,2,3,4,5",
        help="comma-separated list of num_harmonics values to try",
    )
    joint_basis_sweep.add_argument(
        "--plot", type=str, default=None, help="save a trend_degree x num_harmonics heatmap of coverage gap"
    )
    joint_basis_sweep.add_argument("--seed", type=int, default=0)

    block_count_sweep = sub.add_parser(
        "nbeats-block-count-sweep",
        help="jointly sweep the interpretable N-BEATS variant's stack depth, num_trend_blocks AND "
        "num_seasonality_blocks (trend_degree/num_harmonics stay at their shipped defaults throughout) "
        "-- the real follow-up neither basis-shape sweep (trend_degree, num_harmonics, or their v0.28 "
        "joint sweep) attempted: does block count matter, not just basis shape",
    )
    block_count_sweep.add_argument("--dataset", choices=["airline", "temperature", "synthetic"], default="airline")
    block_count_sweep.add_argument("--alpha", type=float, default=0.1)
    block_count_sweep.add_argument(
        "--num-trend-blocks-values",
        type=str,
        default="1,2,3,4,5",
        help="comma-separated list of num_trend_blocks values to try",
    )
    block_count_sweep.add_argument(
        "--num-seasonality-blocks-values",
        type=str,
        default="1,2,3,4,5",
        help="comma-separated list of num_seasonality_blocks values to try",
    )
    block_count_sweep.add_argument(
        "--plot", type=str, default=None, help="save a num_trend_blocks x num_seasonality_blocks heatmap of coverage gap"
    )
    block_count_sweep.add_argument("--seed", type=int, default=0)

    basis_cv = sub.add_parser(
        "nbeats-basis-cv-select",
        help="select the interpretable N-BEATS variant's trend_degree AND num_harmonics via rolling-origin "
        "cross-validation over the calibration set, instead of v0.28's joint sweep reading its best cell "
        "off the real test set directly (mirrors cv-window-select for ACI's window)",
    )
    basis_cv.add_argument("--dataset", choices=["airline", "temperature", "synthetic"], default="airline")
    basis_cv.add_argument("--alpha", type=float, default=0.1)
    basis_cv.add_argument(
        "--trend-degrees", type=str, default="1,2,3,4,5", help="comma-separated list of trend_degree values to try"
    )
    basis_cv.add_argument(
        "--num-harmonics-values",
        type=str,
        default="1,2,3,4,5",
        help="comma-separated list of num_harmonics values to try",
    )
    basis_cv.add_argument("--n-folds", type=int, default=4, help="requested number of rolling-origin folds (may be fewer if the calibration set is small)")
    basis_cv.add_argument("--min-initial-frac", type=float, default=0.2, help="fraction of the calibration set reserved as the first fold's seed")
    basis_cv.add_argument("--min-fold-frac", type=float, default=0.1, help="minimum fraction of the calibration set each fold's validation chunk must have")
    basis_cv.add_argument(
        "--fold-scheme",
        choices=["expanding", "sliding"],
        default="expanding",
        help="'expanding' (default): every fold's seed grows from the start of the calibration set. "
        "'sliding': each fold's seed is capped at the most recent min-initial-frac residuals",
    )
    basis_cv.add_argument(
        "--auto-folds",
        action="store_true",
        help="pick n-folds/min-initial-frac/min-fold-frac automatically from the calibration set's size "
        "instead of the fixed defaults above (which are then ignored)",
    )
    basis_cv.add_argument(
        "--plot", type=str, default=None, help="save a trend_degree x num_harmonics heatmap pair (CV fold gap vs. real test gap)"
    )
    basis_cv.add_argument("--seed", type=int, default=0)

    auto = sub.add_parser(
        "auto-window",
        help="automatically select the sliding-pool ACI's window size from a held-out calibration slice",
    )
    auto.add_argument("--dataset", choices=["airline", "temperature", "synthetic"], default="airline")
    auto.add_argument("--alpha", type=float, default=0.1)
    auto.add_argument("--gamma", type=float, default=0.05, help="ACI step size")
    auto.add_argument(
        "--candidates",
        type=str,
        default="5,7,10,15,20,30,50,unbounded",
        help="comma-separated list of candidate window sizes to select from; include 'unbounded' for the growing buffer",
    )
    auto.add_argument(
        "--holdout-frac",
        type=float,
        default=0.3,
        help="fraction of the calibration set held out (chronologically, from the end) to score each candidate window",
    )
    auto.add_argument("--plot", type=str, default=None, help="save a plot comparing holdout gap vs. real test-set gap by window")
    auto.add_argument("--seed", type=int, default=0)

    cv = sub.add_parser(
        "cv-window-select",
        help="select the sliding-pool ACI's window size via rolling-origin cross-validation over the calibration set",
    )
    cv.add_argument("--dataset", choices=["airline", "temperature", "synthetic"], default="airline")
    cv.add_argument("--alpha", type=float, default=0.1)
    cv.add_argument("--gamma", type=float, default=0.05, help="ACI step size")
    cv.add_argument(
        "--candidates",
        type=str,
        default="5,7,10,15,20,30,50,unbounded",
        help="comma-separated list of candidate window sizes to select from; include 'unbounded' for the growing buffer",
    )
    cv.add_argument("--n-folds", type=int, default=4, help="requested number of rolling-origin folds (may be fewer if the calibration set is small)")
    cv.add_argument("--min-initial-frac", type=float, default=0.2, help="fraction of the calibration set reserved as the first fold's seed")
    cv.add_argument("--min-fold-frac", type=float, default=0.1, help="minimum fraction of the calibration set each fold's validation chunk must have")
    cv.add_argument(
        "--fold-scheme",
        choices=["expanding", "sliding"],
        default="expanding",
        help="'expanding' (default): every fold's seed grows from the start of the calibration set. "
        "'sliding': each fold's seed is capped at the most recent min-initial-frac residuals, so later "
        "folds don't carry arbitrarily old residuals forever",
    )
    cv.add_argument(
        "--auto-folds",
        action="store_true",
        help="pick n-folds/min-initial-frac/min-fold-frac automatically from the calibration set's size "
        "instead of the fixed defaults above (which are then ignored) -- see _auto_cv_fold_params's "
        "docstring for the heuristic",
    )
    cv.add_argument("--plot", type=str, default=None, help="save a plot comparing mean fold |gap| vs. real test-set gap by window")
    cv.add_argument("--seed", type=int, default=0)

    cv_seed = sub.add_parser(
        "cv-window-select-seed-check",
        help="does cv-window-select's own pick stay the same across different model training seeds?",
    )
    cv_seed.add_argument("--dataset", choices=["airline", "temperature", "synthetic"], default="airline")
    cv_seed.add_argument("--alpha", type=float, default=0.1)
    cv_seed.add_argument("--gamma", type=float, default=0.05, help="ACI step size")
    cv_seed.add_argument(
        "--candidates",
        type=str,
        default="5,7,10,15,20,30,50,unbounded",
        help="comma-separated list of candidate window sizes to select from; include 'unbounded' for the growing buffer",
    )
    cv_seed.add_argument("--n-folds", type=int, default=4)
    cv_seed.add_argument("--min-initial-frac", type=float, default=0.2)
    cv_seed.add_argument("--min-fold-frac", type=float, default=0.1)
    cv_seed.add_argument("--fold-scheme", choices=["expanding", "sliding"], default="expanding")
    cv_seed.add_argument("--auto-folds", action="store_true")
    cv_seed.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="comma-separated list of model training seeds to independently re-run cv-window-select at",
    )
    cv_seed.add_argument("--plot", type=str, default=None, help="save a plot of each seed's selected window vs. its real test-set gap")

    enbpi_cv = sub.add_parser(
        "enbpi-cv-window-select",
        help="select EnbPI's window size via rolling-origin cross-validation over the calibration set (mirrors cv-window-select for ACI)",
    )
    enbpi_cv.add_argument("--dataset", choices=["airline", "temperature", "synthetic"], default="airline")
    enbpi_cv.add_argument("--alpha", type=float, default=0.1)
    enbpi_cv.add_argument("--gamma", type=float, default=0.05, help="fixed-pool ACI's step size (reference line only)")
    enbpi_cv.add_argument("--n-bootstrap", type=int, default=20, help="number of bootstrap ensemble members")
    enbpi_cv.add_argument(
        "--candidates",
        type=str,
        default="5,7,10,15,20,30,50,unbounded",
        help="comma-separated list of candidate window sizes to select from; include 'unbounded' for the growing buffer",
    )
    enbpi_cv.add_argument("--n-folds", type=int, default=4, help="requested number of rolling-origin folds (may be fewer if the calibration set is small)")
    enbpi_cv.add_argument("--min-initial-frac", type=float, default=0.2, help="fraction of the calibration set reserved as the first fold's seed")
    enbpi_cv.add_argument("--min-fold-frac", type=float, default=0.1, help="minimum fraction of the calibration set each fold's validation chunk must have")
    enbpi_cv.add_argument(
        "--fold-scheme",
        choices=["expanding", "sliding"],
        default="expanding",
        help="'expanding' (default): every fold's seed grows from the start of the calibration set. "
        "'sliding': each fold's seed is capped at the most recent min-initial-frac residuals",
    )
    enbpi_cv.add_argument(
        "--auto-folds",
        action="store_true",
        help="pick n-folds/min-initial-frac/min-fold-frac automatically from the calibration set's size "
        "instead of the fixed defaults above (which are then ignored)",
    )
    enbpi_cv.add_argument("--plot", type=str, default=None, help="save a plot comparing mean fold |gap| vs. real test-set gap by window")
    enbpi_cv.add_argument("--seed", type=int, default=0)

    serve = sub.add_parser(
        "serve",
        help="run forecasting/api.py (GET /health, GET /datasets, POST /forecast) with uvicorn",
    )
    serve.add_argument("--host", type=str, default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true", help="auto-reload on code changes (development only)")

    args = parser.parse_args()

    if args.command == "benchmark":
        datasets = ["airline", "temperature", "synthetic"] if args.dataset == "all" else [args.dataset]
        if args.plot and args.dataset == "all":
            parser.error("--plot requires a single --dataset, not 'all'")
        for ds in datasets:
            results = run_experiment(ds, alpha=args.alpha, seed=args.seed)
            print(format_results_table(ds, args.alpha, results))
            print()
            if args.plot:
                _plot(ds, results, args.plot)
                print(f"Saved plot to {args.plot}")
    elif args.command == "adaptive":
        comparison = run_adaptive_comparison(args.dataset, alpha=args.alpha, gamma=args.gamma, seed=args.seed)
        print(format_adaptive_comparison(comparison))
        print()
        if args.plot:
            _plot_adaptive(args.dataset, comparison, args.plot)
            print(f"Saved plot to {args.plot}")
    elif args.command == "sliding-window":
        comparison = run_sliding_window_comparison(
            args.dataset, alpha=args.alpha, gamma=args.gamma, window=args.window, seed=args.seed
        )
        print(format_sliding_window_comparison(comparison))
        print()
        if args.plot:
            _plot_sliding_window(args.dataset, comparison, args.plot)
            print(f"Saved plot to {args.plot}")
    elif args.command == "enbpi":
        comparison = run_enbpi_comparison(
            args.dataset,
            alpha=args.alpha,
            gamma=args.gamma,
            n_bootstrap=args.n_bootstrap,
            window=args.window,
            seed=args.seed,
        )
        print(format_enbpi_comparison(comparison))
        print()
        if args.plot:
            _plot_enbpi(args.dataset, comparison, args.plot)
            print(f"Saved plot to {args.plot}")
    elif args.command == "enbpi-window-sweep":
        windows = [None if w.strip().lower() == "unbounded" else int(w) for w in args.windows.split(",")]
        comparison = run_enbpi_window_sweep_comparison(
            args.dataset,
            windows=windows,
            alpha=args.alpha,
            gamma=args.gamma,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
        print(format_enbpi_window_sweep_comparison(comparison))
        print()
        if args.plot:
            _plot_enbpi_window_sweep(args.dataset, comparison, args.plot)
            print(f"Saved plot to {args.plot}")
    elif args.command == "window-sweep":
        windows = [None if w.strip().lower() == "unbounded" else int(w) for w in args.windows.split(",")]
        comparison = run_window_sweep_comparison(
            args.dataset, windows=windows, alpha=args.alpha, gamma=args.gamma, seed=args.seed
        )
        print(format_window_sweep_comparison(comparison))
        print()
        if args.plot:
            _plot_window_sweep(args.dataset, comparison, args.plot)
            print(f"Saved plot to {args.plot}")
    elif args.command == "window-gamma-sweep":
        windows = [None if w.strip().lower() == "unbounded" else int(w) for w in args.windows.split(",")]
        gammas = [float(g) for g in args.gammas.split(",")]
        seeds = [int(s) for s in args.seeds.split(",")]
        comparison = run_window_gamma_sweep_comparison(
            args.dataset, windows=windows, gammas=gammas, alpha=args.alpha, seeds=seeds
        )
        print(format_window_gamma_sweep_comparison(comparison))
        print()
        if args.plot:
            _plot_window_gamma_sweep(args.dataset, comparison, args.plot)
            print(f"Saved plot to {args.plot}")
    elif args.command == "nbeats-basis-sweep":
        trend_degrees = [int(d) for d in args.trend_degrees.split(",")]
        comparison = run_nbeats_basis_sweep_comparison(
            args.dataset, trend_degrees=trend_degrees, alpha=args.alpha, seed=args.seed
        )
        print(format_nbeats_basis_sweep_comparison(comparison))
        print()
        if args.plot:
            _plot_nbeats_basis_sweep(args.dataset, comparison, args.plot)
            print(f"Saved plot to {args.plot}")
    elif args.command == "num-harmonics-sweep":
        num_harmonics_values = [int(h) for h in args.num_harmonics_values.split(",")]
        comparison = run_num_harmonics_sweep_comparison(
            args.dataset, num_harmonics_values=num_harmonics_values, alpha=args.alpha, seed=args.seed
        )
        print(format_num_harmonics_sweep_comparison(comparison))
        print()
        if args.plot:
            _plot_num_harmonics_sweep(args.dataset, comparison, args.plot)
            print(f"Saved plot to {args.plot}")
    elif args.command == "nbeats-joint-basis-sweep":
        trend_degrees = [int(d) for d in args.trend_degrees.split(",")]
        num_harmonics_values = [int(h) for h in args.num_harmonics_values.split(",")]
        comparison = run_nbeats_joint_basis_sweep_comparison(
            args.dataset,
            trend_degrees=trend_degrees,
            num_harmonics_values=num_harmonics_values,
            alpha=args.alpha,
            seed=args.seed,
        )
        print(format_nbeats_joint_basis_sweep_comparison(comparison))
        print()
        if args.plot:
            _plot_nbeats_joint_basis_sweep(args.dataset, comparison, args.plot)
            print(f"Saved plot to {args.plot}")
    elif args.command == "nbeats-block-count-sweep":
        num_trend_blocks_values = [int(n) for n in args.num_trend_blocks_values.split(",")]
        num_seasonality_blocks_values = [int(n) for n in args.num_seasonality_blocks_values.split(",")]
        comparison = run_nbeats_block_count_sweep_comparison(
            args.dataset,
            num_trend_blocks_values=num_trend_blocks_values,
            num_seasonality_blocks_values=num_seasonality_blocks_values,
            alpha=args.alpha,
            seed=args.seed,
        )
        print(format_nbeats_block_count_sweep_comparison(comparison))
        print()
        if args.plot:
            _plot_nbeats_block_count_sweep(args.dataset, comparison, args.plot)
            print(f"Saved plot to {args.plot}")
    elif args.command == "nbeats-basis-cv-select":
        trend_degrees = [int(d) for d in args.trend_degrees.split(",")]
        num_harmonics_values = [int(h) for h in args.num_harmonics_values.split(",")]
        result = run_nbeats_basis_cv_selection(
            args.dataset,
            trend_degrees=trend_degrees,
            num_harmonics_values=num_harmonics_values,
            alpha=args.alpha,
            n_folds=args.n_folds,
            min_initial_frac=args.min_initial_frac,
            min_fold_frac=args.min_fold_frac,
            fold_scheme=args.fold_scheme,
            auto_folds=args.auto_folds,
            seed=args.seed,
        )
        print(format_nbeats_basis_cv_selection(result))
        print()
        if args.plot:
            _plot_nbeats_basis_cv_selection(args.dataset, result, args.plot)
            print(f"Saved plot to {args.plot}")
    elif args.command == "auto-window":
        candidates = [None if w.strip().lower() == "unbounded" else int(w) for w in args.candidates.split(",")]
        result = run_auto_window_selection(
            args.dataset,
            windows=candidates,
            alpha=args.alpha,
            gamma=args.gamma,
            holdout_frac=args.holdout_frac,
            seed=args.seed,
        )
        print(format_auto_window_selection(result))
        print()
        if args.plot:
            _plot_auto_window_selection(args.dataset, result, args.plot)
            print(f"Saved plot to {args.plot}")
    elif args.command == "cv-window-select":
        candidates = [None if w.strip().lower() == "unbounded" else int(w) for w in args.candidates.split(",")]
        result = run_cv_window_selection(
            args.dataset,
            windows=candidates,
            alpha=args.alpha,
            gamma=args.gamma,
            n_folds=args.n_folds,
            min_initial_frac=args.min_initial_frac,
            min_fold_frac=args.min_fold_frac,
            fold_scheme=args.fold_scheme,
            auto_folds=args.auto_folds,
            seed=args.seed,
        )
        print(format_cv_window_selection(result))
        print()
        if args.plot:
            _plot_cv_window_selection(args.dataset, result, args.plot)
            print(f"Saved plot to {args.plot}")
    elif args.command == "cv-window-select-seed-check":
        candidates = [None if w.strip().lower() == "unbounded" else int(w) for w in args.candidates.split(",")]
        seeds = [int(s) for s in args.seeds.split(",")]
        result = run_cv_window_selection_seed_robustness(
            args.dataset,
            windows=candidates,
            seeds=seeds,
            alpha=args.alpha,
            gamma=args.gamma,
            n_folds=args.n_folds,
            min_initial_frac=args.min_initial_frac,
            min_fold_frac=args.min_fold_frac,
            fold_scheme=args.fold_scheme,
            auto_folds=args.auto_folds,
        )
        print(format_cv_window_selection_seed_robustness(result))
        print()
        if args.plot:
            _plot_cv_window_selection_seed_robustness(args.dataset, result, args.plot)
            print(f"Saved plot to {args.plot}")
    elif args.command == "enbpi-cv-window-select":
        candidates = [None if w.strip().lower() == "unbounded" else int(w) for w in args.candidates.split(",")]
        result = run_enbpi_cv_window_selection(
            args.dataset,
            windows=candidates,
            alpha=args.alpha,
            gamma=args.gamma,
            n_bootstrap=args.n_bootstrap,
            n_folds=args.n_folds,
            min_initial_frac=args.min_initial_frac,
            min_fold_frac=args.min_fold_frac,
            fold_scheme=args.fold_scheme,
            auto_folds=args.auto_folds,
            seed=args.seed,
        )
        print(format_enbpi_cv_window_selection(result))
        print()
        if args.plot:
            _plot_enbpi_cv_window_selection(args.dataset, result, args.plot)
            print(f"Saved plot to {args.plot}")
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("forecasting.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
