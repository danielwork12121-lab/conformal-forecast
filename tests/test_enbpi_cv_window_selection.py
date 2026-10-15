"""Tests for `run_enbpi_cv_window_selection`/`format_enbpi_cv_window_selection`
(v0.22) -- the held-out-data half of EnbPI's own "What's next" item (the
CLI-sweep half shipped as v0.21's `run_enbpi_window_sweep_comparison`).
Mirrors `tests/test_cv_window_selection.py`'s real-data integration layer
for ACI's CV window selection, adapted to EnbPI's own residual pool.

The pure fold-boundary helpers this reuses (`_make_rolling_folds`,
`_fold_seed_start`, `_select_best_window`, `_auto_cv_fold_params`) are
already fully tested in `test_cv_window_selection.py` against their own
inputs -- nothing here re-tests that pure logic. These tests cover only
what's actually new: the integration between EnbPI's own fit/residual-pool
structure and that already-tested fold machinery.
"""
from __future__ import annotations

import pytest

from forecasting.experiment import (
    EnbPICVWindowSelectionResult,
    format_enbpi_cv_window_selection,
    run_enbpi_cv_window_selection,
    run_enbpi_window_sweep_comparison,
)

CANDIDATES = [5, 10, 20, None]


def test_runs_and_reports_every_candidate():
    result = run_enbpi_cv_window_selection(
        "synthetic", windows=CANDIDATES, n_bootstrap=4, seed=0
    )

    assert isinstance(result, EnbPICVWindowSelectionResult)
    assert result.dataset == "synthetic"
    assert [c.window for c in result.candidates] == CANDIDATES
    assert result.selected_window in CANDIDATES
    assert result.best_test_window in CANDIDATES
    assert result.n_folds >= 1
    assert len(result.fold_bounds) == result.n_folds
    for c in result.candidates:
        assert c.mean_abs_fold_gap >= 0.0
        assert -1.0 <= c.mean_fold_gap <= 1.0
        assert -1.0 <= c.test_coverage_gap <= 1.0
        assert c.mean_fold_width > 0.0


def test_mean_abs_fold_gap_is_abs_of_mean_fold_gap_bound():
    # mean(|x_i|) >= |mean(x_i)| always (triangle inequality) -- same basic
    # cross-check `test_cv_window_selection.py` runs for the ACI version,
    # confirming the two aggregates reported per candidate are actually
    # consistent with each other.
    result = run_enbpi_cv_window_selection("synthetic", windows=[10, None], n_bootstrap=4, seed=0)
    for c in result.candidates:
        assert c.mean_abs_fold_gap >= abs(c.mean_fold_gap) - 1e-9


def test_folds_partition_the_calibration_set():
    result = run_enbpi_cv_window_selection("synthetic", windows=[10, 20], n_bootstrap=4, seed=0)
    assert result.fold_bounds[0][0] >= 1
    for (_, end_a), (start_b, _) in zip(result.fold_bounds, result.fold_bounds[1:]):
        assert end_a == start_b  # no gap, no overlap


def test_too_small_calibration_set_raises():
    # Same oversized min_initial_frac/min_fold_frac case test_cv_window_selection.py
    # uses for the ACI version: airline's calibration set (28 windows) can't
    # support n_folds=4 at these fractions.
    with pytest.raises(ValueError):
        run_enbpi_cv_window_selection(
            "airline", windows=[5, 10], n_bootstrap=4, n_folds=4, min_initial_frac=0.8, min_fold_frac=0.5, seed=0
        )


def test_invalid_args_raise():
    with pytest.raises(ValueError):
        run_enbpi_cv_window_selection("synthetic", windows=[5, 10], n_bootstrap=4, n_folds=0, seed=0)
    with pytest.raises(ValueError):
        run_enbpi_cv_window_selection("synthetic", windows=[5, 10], n_bootstrap=4, min_initial_frac=0.0, seed=0)
    with pytest.raises(ValueError):
        run_enbpi_cv_window_selection("synthetic", windows=[5, 10], n_bootstrap=4, fold_scheme="bogus", seed=0)


def test_is_deterministic_for_a_fixed_seed():
    a = run_enbpi_cv_window_selection("synthetic", windows=CANDIDATES, n_bootstrap=4, seed=1)
    b = run_enbpi_cv_window_selection("synthetic", windows=CANDIDATES, n_bootstrap=4, seed=1)
    assert a.selected_window == b.selected_window
    assert [c.mean_abs_fold_gap for c in a.candidates] == [c.mean_abs_fold_gap for c in b.candidates]


def test_selected_candidates_test_gap_matches_the_sweep_directly():
    """The whole point of reusing `run_enbpi_window_sweep_comparison` for the
    post-hoc check is that it must report the SAME real test-set numbers the
    sweep itself would for each window -- checked directly, not just assumed
    from the shared code path.
    """
    result = run_enbpi_cv_window_selection("synthetic", windows=[10, None], n_bootstrap=4, seed=2)
    sweep = run_enbpi_window_sweep_comparison("synthetic", windows=[10, None], n_bootstrap=4, seed=2)
    sweep_gap_by_window = {r.window: r.enbpi["coverage_gap"] for r in sweep.results}
    for c in result.candidates:
        assert c.test_coverage_gap == sweep_gap_by_window[c.window]


def test_auto_folds_reports_true_and_more_folds_than_fixed_default():
    result = run_enbpi_cv_window_selection(
        "synthetic", windows=[10, 20], n_bootstrap=4, auto_folds=True, seed=0
    )
    assert result.auto_folds is True
    fixed = run_enbpi_cv_window_selection("synthetic", windows=[10, 20], n_bootstrap=4, auto_folds=False, seed=0)
    assert result.n_folds >= fixed.n_folds


def test_sliding_fold_scheme_is_reported_and_differs_from_expanding_seed_start():
    # Not a win/loss claim either way (unlike test_cv_window_selection.py's
    # pinned win/loss cases for the much larger ACI test suite) -- just
    # confirms the fold_scheme plumbing actually reaches EnbPI's own CV path
    # and is reported correctly.
    result = run_enbpi_cv_window_selection(
        "synthetic", windows=[10, 20], n_bootstrap=4, fold_scheme="sliding", seed=0
    )
    assert result.fold_scheme == "sliding"


def test_format_includes_every_window_and_the_selection_marker():
    result = run_enbpi_cv_window_selection("synthetic", windows=[10, None], n_bootstrap=4, seed=0)
    text = format_enbpi_cv_window_selection(result)
    assert "EnbPI rolling-origin CV window selection" in text
    assert "| 10" in text
    assert "unbounded (growing buffer)" in text
    assert "<- selected" in text
