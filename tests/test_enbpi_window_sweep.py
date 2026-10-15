"""Tests for `run_enbpi_window_sweep_comparison`/`format_enbpi_window_sweep_comparison`
(v0.21) -- the EnbPI analog of `tests/test_experiment.py`'s window-sweep
coverage for ACI, and the "What's next" item named in both
`forecasting/enbpi.py`'s own docstring and the project docs: "a principled,
held-out-data way to pick window ... or an enbpi-window-sweep CLI command
mirroring the existing window-sweep one for ACI." This picks the second,
simpler alternative.
"""
from __future__ import annotations

from forecasting.experiment import (
    EnbPIWindowSweepComparison,
    format_enbpi_window_sweep_comparison,
    run_enbpi_window_sweep_comparison,
)


def test_sweep_runs_every_requested_window_and_reports_reference_lines():
    windows = [5, 20, None]
    comparison = run_enbpi_window_sweep_comparison(
        "synthetic", windows=windows, alpha=0.1, gamma=0.05, n_bootstrap=5, seed=0
    )

    assert isinstance(comparison, EnbPIWindowSweepComparison)
    assert [r.window for r in comparison.results] == windows
    assert 0.0 <= comparison.static["empirical_coverage"] <= 1.0
    assert 0.0 <= comparison.fixed_pool["empirical_coverage"] <= 1.0
    for r in comparison.results:
        assert 0.0 <= r.enbpi["empirical_coverage"] <= 1.0
        assert r.enbpi["n_test"] == comparison.static["n_test"]


def test_sweep_matches_run_enbpi_comparison_for_the_same_window_and_seed():
    """The sweep's whole premise is that it reuses one ensemble fit instead
    of calling `run_enbpi_comparison` once per window -- but it must still
    produce the SAME result `run_enbpi_comparison` would for any individual
    window value, since fitting is deterministic given the same seed/model
    factory. Checked directly here (not just assumed from the docstring):
    if the sweep accidentally reused a stale/wrong model or pool, this
    would catch it as a real numeric mismatch, not just a style nit.
    """
    from forecasting.experiment import run_enbpi_comparison

    sweep = run_enbpi_window_sweep_comparison(
        "synthetic", windows=[7, None], alpha=0.1, gamma=0.05, n_bootstrap=4, seed=3
    )
    direct_bounded = run_enbpi_comparison("synthetic", alpha=0.1, gamma=0.05, n_bootstrap=4, window=7, seed=3)
    direct_unbounded = run_enbpi_comparison("synthetic", alpha=0.1, gamma=0.05, n_bootstrap=4, window=None, seed=3)

    by_window = {r.window: r.enbpi for r in sweep.results}
    assert by_window[7] == direct_bounded.enbpi
    assert by_window[None] == direct_unbounded.enbpi


def test_window_bounds_the_swept_results_mean_interval_width_sanely():
    """A direct regression guard, through the sweep path specifically, for
    the exact pool-truncation bug `forecasting/enbpi.py` documents finding
    and fixing (a pool seeded above `window` that never actually shrinks
    without an explicit pre-loop truncation -- `run_enbpi_window_sweep_comparison`
    builds its own fresh `EnbPIForecaster` per window value rather than
    mutating the one `fit()` returns, so this is worth checking on this
    path too, not just the one `forecasting/enbpi.py`'s own tests cover).
    A severely under-bounded window (1) should give a visibly different,
    less-informed interval than a generous one (50) on the same fixed
    ensemble/pool -- if truncation silently didn't apply, every window
    would collapse to the same (full-pool) result.
    """
    comparison = run_enbpi_window_sweep_comparison(
        "synthetic", windows=[1, 50], alpha=0.1, gamma=0.05, n_bootstrap=4, seed=4
    )
    tiny, generous = comparison.results[0], comparison.results[1]
    assert tiny.enbpi["mean_interval_width"] != generous.enbpi["mean_interval_width"]


def test_format_includes_every_window_and_both_reference_lines():
    comparison = run_enbpi_window_sweep_comparison(
        "synthetic", windows=[5, None], alpha=0.1, gamma=0.05, n_bootstrap=4, seed=0
    )
    text = format_enbpi_window_sweep_comparison(comparison)
    assert "static split conformal (reference)" in text
    assert "fixed-pool ACI (reference)" in text
    assert "| 5 |" in text
    assert "unbounded (growing buffer)" in text
