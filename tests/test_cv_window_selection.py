"""Tests for rolling-origin cross-validated window selection (v0.6) --
`_make_rolling_folds` / `run_cv_window_selection` in `forecasting/experiment.py`
-- and its `fold_scheme` option (v0.7) -- `_fold_seed_start`.

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
    _fold_seed_start,
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
# Layer 1b: `_fold_seed_start` (v0.7) -- pure, no model/data dependency,
# same testing split as `_make_rolling_folds` above. This decides how much
# calibration history feeds each fold's seed; `_make_rolling_folds` itself
# is unchanged by `fold_scheme` -- the (train_end, test_end) validation
# partition is identical either way, only the seed start differs.
# ---------------------------------------------------------------------------


def test_fold_seed_start_expanding_is_always_zero():
    for train_end in [10, 20, 55, 100]:
        assert _fold_seed_start(train_end, min_initial=15, fold_scheme="expanding") == 0


def test_fold_seed_start_sliding_caps_at_min_initial():
    assert _fold_seed_start(100, min_initial=20, fold_scheme="sliding") == 80
    assert _fold_seed_start(45, min_initial=20, fold_scheme="sliding") == 25


def test_fold_seed_start_sliding_never_goes_negative_on_the_first_fold():
    # The very first fold's seed is exactly [0, min_initial) under both
    # schemes -- 'sliding' only starts dropping history once there's more
    # than min_initial residuals behind the fold.
    assert _fold_seed_start(15, min_initial=15, fold_scheme="sliding") == 0
    assert _fold_seed_start(10, min_initial=15, fold_scheme="sliding") == 0


def test_fold_seed_start_sliding_window_size_is_constant_after_the_first_fold():
    # Once train_end exceeds min_initial, the sliding seed window's *size*
    # (train_end - seed_start) stays pinned at min_initial rather than
    # growing -- that's the whole point of 'sliding' vs. 'expanding'.
    for train_end in [20, 45, 70, 100]:
        seed_start = _fold_seed_start(train_end, min_initial=20, fold_scheme="sliding")
        assert train_end - seed_start == 20


def test_fold_seed_start_invalid_scheme_raises():
    with pytest.raises(ValueError):
        _fold_seed_start(50, min_initial=10, fold_scheme="bogus")


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


# ---------------------------------------------------------------------------
# Layer 2b: `fold_scheme` (v0.7) -- real-data integration tests for the
# 'expanding' vs. 'sliding' seed-window comparison. Same "measure, don't
# assert" discipline as the layer-2 tests above: the honest, 6-seed x
# 3-dataset result (see the README's own "Sliding vs. expanding fold
# scheme" section) is mixed, so these pin the specific win/loss/tie cases
# that were actually measured, not a blanket claim either direction.
# ---------------------------------------------------------------------------


def test_cv_window_selection_default_fold_scheme_is_expanding():
    result = run_cv_window_selection("temperature", windows=[10, 30, None], seed=0)
    assert result.fold_scheme == "expanding"


def test_cv_window_selection_reports_the_requested_fold_scheme():
    result = run_cv_window_selection("synthetic", windows=[10, 30], fold_scheme="sliding", seed=0)
    assert result.fold_scheme == "sliding"


def test_cv_window_selection_invalid_fold_scheme_raises():
    with pytest.raises(ValueError):
        run_cv_window_selection("temperature", windows=[5, 10], fold_scheme="bogus", seed=0)


def test_cv_window_selection_sliding_scheme_is_deterministic_for_a_fixed_seed():
    a = run_cv_window_selection("synthetic", windows=CANDIDATES, fold_scheme="sliding", seed=0)
    b = run_cv_window_selection("synthetic", windows=CANDIDATES, fold_scheme="sliding", seed=0)
    assert a.selected_window == b.selected_window
    assert [c.mean_abs_fold_gap for c in a.candidates] == [c.mean_abs_fold_gap for c in b.candidates]


def test_cv_window_selection_sliding_scheme_helps_on_airline_seed_3():
    """Honest win case for `sliding`, pinned so a future change can't
    silently claim it does better than this -- or stop doing this well --
    without the README's own table being re-verified. See the README's
    "Sliding vs. expanding fold scheme" section for the full 6-seed table
    this is drawn from.
    """
    expanding = run_cv_window_selection("airline", windows=CANDIDATES, fold_scheme="expanding", seed=3)
    sliding = run_cv_window_selection("airline", windows=CANDIDATES, fold_scheme="sliding", seed=3)
    expanding_gap = abs(expanding.selected_test_result["coverage_gap"])
    sliding_gap = abs(sliding.selected_test_result["coverage_gap"])
    assert sliding_gap < expanding_gap, (
        f"expected this specific documented case (airline, seed=3) where sliding "
        f"(window={sliding.selected_window}, |gap|={sliding_gap:.4f}) beats expanding "
        f"(window={expanding.selected_window}, |gap|={expanding_gap:.4f}) -- if this no longer "
        "holds, the README's honest comparison needs updating to match, not this test loosened"
    )


def test_cv_window_selection_sliding_scheme_hurts_on_airline_seed_2():
    """Honest loss case for `sliding` -- included for the same
    not-just-the-win-case reason `test_cv_window_selection_does_not_uniformly_beat_single_holdout`
    pins a loss case for CV vs. single-holdout above. `sliding` is NOT a
    strict upgrade over `expanding`, and this is one of the measured cases
    where it's worse.
    """
    expanding = run_cv_window_selection("airline", windows=CANDIDATES, fold_scheme="expanding", seed=2)
    sliding = run_cv_window_selection("airline", windows=CANDIDATES, fold_scheme="sliding", seed=2)
    expanding_gap = abs(expanding.selected_test_result["coverage_gap"])
    sliding_gap = abs(sliding.selected_test_result["coverage_gap"])
    assert sliding_gap > expanding_gap, (
        f"expected this specific documented case (airline, seed=2) where sliding "
        f"(window={sliding.selected_window}, |gap|={sliding_gap:.4f}) does worse than expanding "
        f"(window={expanding.selected_window}, |gap|={expanding_gap:.4f}) -- if this no longer "
        "holds, the README's honest comparison needs updating to match, not this test loosened"
    )


def test_cv_window_selection_fold_scheme_has_no_effect_on_temperature_seed_0():
    """Tie case: on a calibration set as large as `temperature`'s
    (n_cal=730), capping fold history at `min_initial_frac * n_cal` never
    actually truncates a fold's seed differently in a way that changes
    which window wins -- both schemes land on the exact same selection and
    gap. Measured across all 6 README seeds, not just this one; seed=0
    pinned here as the representative, reproducible case.
    """
    expanding = run_cv_window_selection("temperature", windows=CANDIDATES, fold_scheme="expanding", seed=0)
    sliding = run_cv_window_selection("temperature", windows=CANDIDATES, fold_scheme="sliding", seed=0)
    assert expanding.selected_window == sliding.selected_window
    assert expanding.selected_test_result["coverage_gap"] == sliding.selected_test_result["coverage_gap"]
