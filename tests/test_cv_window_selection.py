"""Tests for rolling-origin cross-validated window selection (v0.6) --
`_make_rolling_folds` / `run_cv_window_selection` in `forecasting/experiment.py`.

Same two-layer split as `test_auto_window_selection.py`:

1. `_make_rolling_folds` is pure (no model, no data loading) and is tested
   directly and fast for its fold-boundary logic.
2. `run_cv_window_selection` is tested against the real bundled datasets,
   because the actual question this feature answers -- "does averaging over
   several rolling folds pick a better window than v0.5's single holdout
   slice?" -- can only be checked with real data. Per this repo's "measure,
   don't assert" discipline, the answer here is honestly mixed (see the
   README's multi-seed comparison), not a clean win, so these tests pin
   down structural correctness and the cases that *are* consistent, not a
   blanket "CV always wins" claim that the real numbers don't support.
"""
from __future__ import annotations

import pytest

from forecasting.experiment import (
    _make_rolling_folds,
    _select_best_window,
    run_auto_window_selection,
    run_cv_window_selection,
)


# ---------------------------------------------------------------------------
# Layer 1: pure fold-boundary logic, no training involved.
# ---------------------------------------------------------------------------


def test_make_rolling_folds_exact_division():
    folds = _make_rolling_folds(n=28, n_folds=4, min_initial=6, min_fold_size=3)
    assert folds == [(6, 11), (11, 16), (16, 21), (21, 28)]


def test_make_rolling_folds_cover_range_exactly_with_no_gap_or_overlap():
    folds = _make_rolling_folds(n=160, n_folds=4, min_initial=32, min_fold_size=16)
    assert folds[0][0] == 32
    assert folds[-1][1] == 160
    for (_, end_a), (start_b, _) in zip(folds, folds[1:]):
        assert end_a == start_b  # no gap, no overlap


def test_make_rolling_folds_expanding_not_sliding():
    # Every fold's "seed" (everything before its own start) only grows --
    # fold k's start is always fold (k-1)'s end, never resets backward.
    folds = _make_rolling_folds(n=100, n_folds=5, min_initial=10, min_fold_size=5)
    starts = [f[0] for f in folds]
    assert starts == sorted(starts)
    assert starts[0] == 10


def test_make_rolling_folds_fewer_achievable_folds_than_requested():
    # Only enough data for 3 folds of size >= 5 (usable=15, 15 // 5 = 3)
    # even though 6 were requested.
    folds = _make_rolling_folds(n=20, n_folds=6, min_initial=5, min_fold_size=5)
    assert folds is not None
    assert len(folds) == 3


def test_make_rolling_folds_too_small_returns_none():
    assert _make_rolling_folds(n=10, n_folds=4, min_initial=8, min_fold_size=5) is None


def test_make_rolling_folds_single_fold_requested():
    folds = _make_rolling_folds(n=20, n_folds=1, min_initial=5, min_fold_size=5)
    assert folds == [(5, 20)]


def test_make_rolling_folds_invalid_args_raise():
    with pytest.raises(ValueError):
        _make_rolling_folds(n=20, n_folds=0, min_initial=5, min_fold_size=5)
    with pytest.raises(ValueError):
        _make_rolling_folds(n=20, n_folds=2, min_initial=0, min_fold_size=5)
    with pytest.raises(ValueError):
        _make_rolling_folds(n=20, n_folds=2, min_initial=5, min_fold_size=0)


# ---------------------------------------------------------------------------
# Layer 2: real-data integration, via run_cv_window_selection.
# ---------------------------------------------------------------------------

CANDIDATES = [5, 7, 10, 15, 20, 30, 50, None]


def test_cv_window_selection_runs_and_reports_every_candidate():
    result = run_cv_window_selection("temperature", windows=[5, 10, 20, 30, 50, None], seed=0)

    assert result.dataset == "temperature"
    assert [c.window for c in result.candidates] == [5, 10, 20, 30, 50, None]
    assert result.selected_window in [5, 10, 20, 30, 50, None]
    assert result.best_test_window in [5, 10, 20, 30, 50, None]
    assert result.n_folds >= 1
    assert len(result.fold_bounds) == result.n_folds
    for c in result.candidates:
        assert c.mean_abs_fold_gap >= 0.0
        assert -1.0 <= c.mean_fold_gap <= 1.0
        assert -1.0 <= c.test_coverage_gap <= 1.0
        assert c.mean_fold_width > 0.0


def test_cv_window_selection_mean_abs_fold_gap_is_abs_of_mean_fold_gap_bound():
    # mean(|x_i|) >= |mean(x_i)| always (triangle inequality) -- a basic
    # sanity check that the two aggregates reported per candidate are
    # actually consistent with each other, not computed from unrelated data.
    result = run_cv_window_selection("temperature", windows=[10, 30, None], seed=0)
    for c in result.candidates:
        assert c.mean_abs_fold_gap >= abs(c.mean_fold_gap) - 1e-9


def test_cv_window_selection_folds_partition_the_calibration_set():
    result = run_cv_window_selection("synthetic", windows=[10, 30], seed=0)
    assert result.fold_bounds[0][0] >= 1
    for (_, end_a), (start_b, _) in zip(result.fold_bounds, result.fold_bounds[1:]):
        assert end_a == start_b


def test_cv_window_selection_too_small_calibration_set_raises():
    # airline's calibration set (28 windows): reserving 80% of it as the
    # first fold's seed (min_initial=22) leaves only 6 points usable, below
    # the 14-point minimum fold size an oversized min_fold_frac=0.5 demands.
    with pytest.raises(ValueError):
        run_cv_window_selection("airline", windows=[5, 10], n_folds=4, min_initial_frac=0.8, min_fold_frac=0.5, seed=0)


def test_cv_window_selection_invalid_args_raise():
    with pytest.raises(ValueError):
        run_cv_window_selection("temperature", windows=[5, 10], n_folds=0, seed=0)
    with pytest.raises(ValueError):
        run_cv_window_selection("temperature", windows=[5, 10], min_initial_frac=0.0, seed=0)
    with pytest.raises(ValueError):
        run_cv_window_selection("temperature", windows=[5, 10], min_fold_frac=1.0, seed=0)


def test_cv_window_selection_is_deterministic_for_a_fixed_seed():
    a = run_cv_window_selection("synthetic", windows=CANDIDATES, seed=0)
    b = run_cv_window_selection("synthetic", windows=CANDIDATES, seed=0)
    assert a.selected_window == b.selected_window
    assert [c.mean_abs_fold_gap for c in a.candidates] == [c.mean_abs_fold_gap for c in b.candidates]


def test_cv_window_selection_matches_best_in_hindsight_on_temperature():
    """Same clean success case as v0.5's own temperature test: a large
    calibration set (730 windows here) gives the fold columns enough points
    to actually discriminate between candidates, and the CV-selected
    window's real test-set gap should land close to the oracle's.
    """
    result = run_cv_window_selection("temperature", windows=CANDIDATES, seed=0)
    best_gap = result.selected_test_result["coverage_gap"]
    oracle_gap = [c.test_coverage_gap for c in result.candidates if c.window == result.best_test_window][0]
    assert abs(best_gap - oracle_gap) <= 0.02, (
        f"selected window={result.selected_window} real test gap={best_gap:+.4f} vs. "
        f"best-in-hindsight window={result.best_test_window} gap={oracle_gap:+.4f}"
    )


def test_cv_window_selection_does_not_uniformly_beat_single_holdout():
    """Honest, not a cherry-pick: this repo's own multi-seed exploration
    (see active-project.md / the README's CV section) found rolling-origin
    CV selection is NOT a strict improvement over v0.5's single holdout --
    it wins on some seeds/datasets, ties on others, and loses on at least
    one (synthetic, seed=2). This test pins down that specific honestly-
    documented case so a future change can't silently start claiming CV is
    strictly better than it's actually measured to be, without that claim
    also being re-verified here.
    """
    cv = run_cv_window_selection("synthetic", windows=CANDIDATES, seed=2)
    holdout = run_auto_window_selection("synthetic", windows=CANDIDATES, holdout_frac=0.3, seed=2)
    cv_gap = abs(cv.selected_test_result["coverage_gap"])
    holdout_gap = abs(holdout.selected_test_result["coverage_gap"])
    assert cv_gap > holdout_gap, (
        f"expected this specific documented case (synthetic, seed=2) where CV selection "
        f"(window={cv.selected_window}, |gap|={cv_gap:.4f}) does worse than the single-holdout "
        f"selection (window={holdout.selected_window}, |gap|={holdout_gap:.4f}) -- if this now "
        "passes with CV doing *better* or equal, the README's honest-mixed-results claim needs "
        "updating to match, not this test loosened to hide the change"
    )
