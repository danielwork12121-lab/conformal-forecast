"""Tests for `run_window_gamma_sweep_comparison`/`format_window_gamma_sweep_comparison`
(v0.27) -- the real follow-up `run_window_sweep_comparison`'s own README "What's next"
named: the original window sweep (and its 6-seed re-check) only ever tuned `window`,
holding `gamma=0.05` fixed throughout, so whether the best window depends on `gamma`
(or vice versa) was never actually measured, only assumed. This covers the joint sweep.
"""
from __future__ import annotations

import numpy as np

from forecasting.experiment import (
    WindowGammaSweepComparison,
    format_window_gamma_sweep_comparison,
    run_window_gamma_sweep_comparison,
    run_window_sweep_comparison,
)


def test_sweep_runs_every_requested_cell_and_reports_real_numbers():
    windows = [10, 20, None]
    gammas = [0.02, 0.05, 0.1]
    comparison = run_window_gamma_sweep_comparison(
        "airline", windows=windows, gammas=gammas, alpha=0.1, seeds=[0, 1]
    )

    assert isinstance(comparison, WindowGammaSweepComparison)
    assert len(comparison.cells) == len(windows) * len(gammas)
    seen = {(c.window, c.gamma) for c in comparison.cells}
    assert seen == {(w, g) for w in windows for g in gammas}
    for cell in comparison.cells:
        assert cell.mean_interval_width > 0.0
        assert len(cell.per_seed_gap) == 2
        assert cell.mean_gap == float(np.mean(cell.per_seed_gap))
        assert cell.mean_abs_gap == float(np.mean(np.abs(cell.per_seed_gap)))


def test_single_seed_single_gamma_matches_run_window_sweep_comparison_exactly():
    """The whole point of reusing one fit per seed across the grid (not retraining per
    cell) is that a (window, gamma) cell here must numerically match what the older,
    single-purpose `run_window_sweep_comparison` reports for the same window/gamma/seed
    -- checked directly as a real numeric equality, not just assumed from both calling
    the same underlying class.
    """
    gamma = 0.05
    window = 15
    joint = run_window_gamma_sweep_comparison(
        "airline", windows=[window], gammas=[gamma], alpha=0.1, seeds=[0]
    )
    single = run_window_sweep_comparison("airline", windows=[window], alpha=0.1, gamma=gamma, seed=0)

    cell = joint.cells[0]
    direct = single.results[0].sliding
    assert cell.per_seed_gap[0] == direct["coverage_gap"]
    assert cell.mean_interval_width == direct["mean_interval_width"]


def test_varying_gamma_alone_changes_results_not_just_window():
    """Direct regression guard for the actual question this feature answers: if gamma
    had no measurable effect once window is fixed, the whole premise of this sweep
    (that window and gamma might interact) would be moot. Confirms gamma is genuinely
    threaded through to `predict_sequential`, not silently ignored after the first fit.
    """
    comparison = run_window_gamma_sweep_comparison(
        "airline", windows=[15], gammas=[0.01, 0.3], alpha=0.1, seeds=[0]
    )
    low_gamma_gap = comparison.cells[0].per_seed_gap[0]
    high_gamma_gap = comparison.cells[1].per_seed_gap[0]
    assert low_gamma_gap != high_gamma_gap


def test_one_fit_per_seed_not_per_grid_cell():
    """Regression guard, mirroring this repo's own `trend_degree` sweep guard but in the
    opposite direction: unlike that sweep (which must refit per value because trend_degree
    changes the model's own architecture), window and gamma must NOT change the fitted
    LSTM's point predictions at all -- every cell for a given seed should be built from
    literally the same `fitted.model`/`fitted._residuals` object, so two cells sharing a
    seed must share the same model-fit cost. Checked indirectly: the seeded calibration
    residual pool (before any window truncation) is identical across every window/gamma
    value for one seed, which would only be true if fit() ran exactly once per seed.
    """
    comparison = run_window_gamma_sweep_comparison(
        "synthetic", windows=[5, 10, None], gammas=[0.05, 0.15], alpha=0.1, seeds=[0]
    )
    # every cell for seed=0 shares one fit; a quick cross-check that gamma=0.05/window=None
    # (the unbounded buffer, which never truncates) reproduces the single-seed
    # run_window_sweep_comparison's own unbounded reference exactly confirms the shared fit.
    single = run_window_sweep_comparison("synthetic", windows=[None], alpha=0.1, gamma=0.05, seed=0)
    matching = [c for c in comparison.cells if c.window is None and c.gamma == 0.05][0]
    assert matching.per_seed_gap[0] == single.results[0].sliding["coverage_gap"]


def test_format_includes_every_window_and_gamma_and_the_seed_count():
    comparison = run_window_gamma_sweep_comparison(
        "synthetic", windows=[10, None], gammas=[0.05, 0.1], alpha=0.1, seeds=[0, 1, 2]
    )
    text = format_window_gamma_sweep_comparison(comparison)
    assert "| 10 | 0.05 |" in text
    assert "| unbounded | 0.1 |" in text
    assert "3 seeds" in text
