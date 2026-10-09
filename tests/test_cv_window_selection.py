"""Tests for rolling-origin cross-validated window selection (v0.6) --
`_make_rolling_folds` / `run_cv_window_selection` in `forecasting/experiment.py`
-- its `fold_scheme` option (v0.7) -- `_fold_seed_start` -- its
`auto_folds` option (v0.8) -- `_auto_cv_fold_params` -- and whether the two
options compound when used together (v0.9, no new production code -- see
the `_sliding_plus_auto_folds_` tests below).

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
    _auto_cv_fold_params,
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


# ---------------------------------------------------------------------------
# Layer 1c: `_auto_cv_fold_params` (v0.8) -- pure, no model/data dependency,
# same testing split as `_make_rolling_folds` / `_fold_seed_start` above.
# Decides (n_folds, min_initial_frac, min_fold_frac) from n_cal instead of
# the v0.6/v0.7 fixed defaults (4, 0.2, 0.1) applied regardless of
# calibration-set size.
# ---------------------------------------------------------------------------


def test_auto_cv_fold_params_requests_max_folds_regardless_of_n_cal():
    # The requested n_folds is deliberately a constant upper bound --
    # _make_rolling_folds's own capping (actual_folds = min(n_folds, usable
    # // min_fold_size)) is what adapts it down per dataset, not this
    # function directly reducing its request for a smaller n_cal.
    for n_cal in [10, 28, 160, 730, 5000]:
        n_folds, _, _ = _auto_cv_fold_params(n_cal, max_folds=15)
        assert n_folds == 15


def test_auto_cv_fold_params_fractions_shrink_as_n_cal_grows():
    # Larger n_cal -> smaller fractions, because the function targets
    # roughly constant ABSOLUTE fold sizes, not a constant fraction.
    _, mi_small, mf_small = _auto_cv_fold_params(28)
    _, mi_mid, mf_mid = _auto_cv_fold_params(160)
    _, mi_large, mf_large = _auto_cv_fold_params(730)
    assert mi_small > mi_mid > mi_large
    assert mf_small > mf_mid > mf_large


def test_auto_cv_fold_params_targets_translate_to_absolute_point_counts():
    # min_initial_frac * n_cal should land close to target_min_initial (and
    # likewise for min_fold_frac/target_fold_size) whenever the 0.4/0.2 cap
    # isn't binding -- confirms the "roughly constant absolute size" framing
    # in the docstring is actually what the numbers do, not just a claim.
    n_cal = 400
    n_folds, min_initial_frac, min_fold_frac = _auto_cv_fold_params(
        n_cal, target_min_initial=8, target_fold_size=4
    )
    assert abs(min_initial_frac * n_cal - 8) < 1e-6
    assert abs(min_fold_frac * n_cal - 4) < 1e-6


def test_auto_cv_fold_params_caps_fractions_on_a_tiny_calibration_set():
    # Below the point where target_min_initial/target_fold_size would push
    # the fraction past the cap, the 0.4/0.2 ceiling takes over instead of
    # requesting an ever-larger fraction as n_cal shrinks further.
    n_folds, min_initial_frac, min_fold_frac = _auto_cv_fold_params(5, target_min_initial=8, target_fold_size=4)
    assert min_initial_frac == 0.4
    assert min_fold_frac == 0.2


# ---------------------------------------------------------------------------
# Layer 2c: real-data integration, via run_cv_window_selection(auto_folds=True).
# Same "measure, don't assert" discipline: the honest 6-seed x 3-dataset
# result (see the README's "Automatic fold count" section) is that
# auto_folds never did worse than the v0.6/v0.7 fixed defaults across the
# 18 seed/dataset combinations actually measured, and sometimes did
# noticeably better -- reported as measured, not claimed as a general
# guarantee beyond what was tested.
# ---------------------------------------------------------------------------


def test_cv_window_selection_default_auto_folds_is_false():
    result = run_cv_window_selection("temperature", windows=[10, 30, None], seed=0)
    assert result.auto_folds is False
    assert result.n_folds == 4  # the v0.6 fixed default, unchanged


def test_cv_window_selection_auto_folds_reports_true_and_the_params_used():
    result = run_cv_window_selection("synthetic", windows=[10, 30], auto_folds=True, seed=0)
    assert result.auto_folds is True
    assert result.min_initial_frac == pytest.approx(min(0.4, 8 / 160))
    assert result.min_fold_frac == pytest.approx(min(0.2, 4 / 160))


def test_cv_window_selection_auto_folds_achieves_more_folds_on_larger_datasets():
    # The whole point: a larger calibration set should get MORE folds
    # automatically, not just bigger ones at the same fixed count of 4.
    airline = run_cv_window_selection("airline", windows=[10, 30], auto_folds=True, seed=0)
    synthetic = run_cv_window_selection("synthetic", windows=[10, 30], auto_folds=True, seed=0)
    temperature = run_cv_window_selection("temperature", windows=[10, 30], auto_folds=True, seed=0)
    assert airline.n_folds > 4  # still beats the old fixed default, even on the smallest set
    assert synthetic.n_folds > airline.n_folds
    assert temperature.n_folds >= synthetic.n_folds


def test_cv_window_selection_auto_folds_is_deterministic_for_a_fixed_seed():
    a = run_cv_window_selection("synthetic", windows=CANDIDATES, auto_folds=True, seed=0)
    b = run_cv_window_selection("synthetic", windows=CANDIDATES, auto_folds=True, seed=0)
    assert a.selected_window == b.selected_window
    assert a.n_folds == b.n_folds
    assert [c.mean_abs_fold_gap for c in a.candidates] == [c.mean_abs_fold_gap for c in b.candidates]


def test_cv_window_selection_auto_folds_beats_fixed_default_on_synthetic_seed_2():
    """Honest win case for `auto_folds=True`, pinned the same way the
    fold_scheme win/loss cases above are -- so a future change can't
    silently claim it does better or stop doing this well without the
    README's own table being re-verified. See the README's "Automatic
    fold count" section for the full 6-seed x 3-dataset table.
    """
    fixed = run_cv_window_selection("synthetic", windows=CANDIDATES, auto_folds=False, seed=2)
    auto = run_cv_window_selection("synthetic", windows=CANDIDATES, auto_folds=True, seed=2)
    fixed_gap = abs(fixed.selected_test_result["coverage_gap"])
    auto_gap = abs(auto.selected_test_result["coverage_gap"])
    assert auto_gap < fixed_gap, (
        f"expected this specific documented case (synthetic, seed=2) where auto_folds "
        f"(n_folds={auto.n_folds}, window={auto.selected_window}, |gap|={auto_gap:.4f}) beats the "
        f"fixed default (n_folds={fixed.n_folds}, window={fixed.selected_window}, |gap|={fixed_gap:.4f}) "
        "-- if this no longer holds, the README's honest comparison needs updating to match, not "
        "this test loosened"
    )


def test_cv_window_selection_auto_folds_does_not_lose_on_airline_seed_1():
    """Not-just-ties/wins case: pins the specific `airline` seed=1 result
    where auto_folds does measurably better (not just tie) even on the
    smallest, most fold-constrained dataset -- where `fold_scheme=sliding`
    (v0.7) was measured to be slightly *worse* on average. Across all 18
    seed/dataset combinations measured for this README section, auto_folds
    was never observed to do worse than the fixed default -- this is one
    concrete instance of that, not a claim that it can never lose on data
    not tested here.
    """
    fixed = run_cv_window_selection("airline", windows=CANDIDATES, auto_folds=False, seed=1)
    auto = run_cv_window_selection("airline", windows=CANDIDATES, auto_folds=True, seed=1)
    fixed_gap = abs(fixed.selected_test_result["coverage_gap"])
    auto_gap = abs(auto.selected_test_result["coverage_gap"])
    assert auto_gap <= fixed_gap, (
        f"expected auto_folds (|gap|={auto_gap:.4f}) to not do worse than the fixed default "
        f"(|gap|={fixed_gap:.4f}) on this documented case (airline, seed=1)"
    )


def test_cv_window_selection_auto_folds_loses_on_airline_seed_16():
    """v0.10: the 6-seed table above (seeds 0-5) never found a case where
    `auto_folds` did worse than the fixed default. Re-checking across 20
    seeds (0-19) found 3 that do -- this is one of them. The fixed default
    happens to land on an exact 0.0pp test gap here; auto_folds picks a
    smaller window (more, smaller folds push it toward window=10 instead of
    15) that measurably undershoots. Small in absolute terms (+3.3pp), but
    a real loss, not a tie -- see the README's "does 'never lost' hold up
    with more seeds?" section for the full 20-seed table this is drawn from.
    """
    fixed = run_cv_window_selection("airline", windows=CANDIDATES, auto_folds=False, seed=16)
    auto = run_cv_window_selection("airline", windows=CANDIDATES, auto_folds=True, seed=16)
    fixed_gap = abs(fixed.selected_test_result["coverage_gap"])
    auto_gap = abs(auto.selected_test_result["coverage_gap"])
    assert auto_gap > fixed_gap, (
        f"expected this documented case (airline, seed=16) where auto_folds "
        f"(n_folds={auto.n_folds}, window={auto.selected_window}, |gap|={auto_gap:.4f}) does WORSE than "
        f"the fixed default (n_folds={fixed.n_folds}, window={fixed.selected_window}, |gap|={fixed_gap:.4f}) "
        "-- if this no longer holds, the README's 20-seed table needs updating to match, not this test loosened"
    )


def test_cv_window_selection_auto_folds_loses_on_synthetic_seed_6():
    """v0.10: second of the 3 new loss cases found at 20 seeds (the other
    is seed=11, same dataset, same exact numbers) -- see
    test_cv_window_selection_auto_folds_loses_on_airline_seed_16 and the
    README section it references for the full context.
    """
    fixed = run_cv_window_selection("synthetic", windows=CANDIDATES, auto_folds=False, seed=6)
    auto = run_cv_window_selection("synthetic", windows=CANDIDATES, auto_folds=True, seed=6)
    fixed_gap = abs(fixed.selected_test_result["coverage_gap"])
    auto_gap = abs(auto.selected_test_result["coverage_gap"])
    assert auto_gap > fixed_gap, (
        f"expected this documented case (synthetic, seed=6) where auto_folds "
        f"(n_folds={auto.n_folds}, window={auto.selected_window}, |gap|={auto_gap:.4f}) does WORSE than "
        f"the fixed default (n_folds={fixed.n_folds}, window={fixed.selected_window}, |gap|={fixed_gap:.4f})"
    )


def test_cv_window_selection_auto_folds_loses_on_synthetic_seed_11():
    """v0.10: third of the 3 new loss cases found at 20 seeds -- included
    alongside seed=6 (same dataset) so the loss isn't read as a one-off
    fluke of a single seed.
    """
    fixed = run_cv_window_selection("synthetic", windows=CANDIDATES, auto_folds=False, seed=11)
    auto = run_cv_window_selection("synthetic", windows=CANDIDATES, auto_folds=True, seed=11)
    fixed_gap = abs(fixed.selected_test_result["coverage_gap"])
    auto_gap = abs(auto.selected_test_result["coverage_gap"])
    assert auto_gap > fixed_gap, (
        f"expected this documented case (synthetic, seed=11) where auto_folds "
        f"(n_folds={auto.n_folds}, window={auto.selected_window}, |gap|={auto_gap:.4f}) does WORSE than "
        f"the fixed default (n_folds={fixed.n_folds}, window={fixed.selected_window}, |gap|={fixed_gap:.4f})"
    )


def test_cv_window_selection_sliding_plus_auto_folds_loses_to_auto_alone_on_synthetic_seed_2():
    """v0.9: does combining `fold_scheme='sliding'` (v0.7) with
    `auto_folds=True` (v0.8) compound -- i.e. do better than `auto_folds`
    alone, since both are independently non-losing/improving options?
    Measured across the same 6 seeds x 3 datasets as every other
    comparison in this README, the honest answer is no: on `synthetic`
    specifically, adding `sliding` on top of `auto_folds` makes things
    measurably *worse* on 4 of 6 seeds (mean |gap| 0.42pp -> 1.04pp), with
    this seed=2 case as the clearest single instance. See the README's "Do
    fold_scheme and auto_folds compound?" section for the full table.
    """
    auto_alone = run_cv_window_selection("synthetic", windows=CANDIDATES, auto_folds=True, seed=2)
    compound = run_cv_window_selection(
        "synthetic", windows=CANDIDATES, fold_scheme="sliding", auto_folds=True, seed=2
    )
    auto_alone_gap = abs(auto_alone.selected_test_result["coverage_gap"])
    compound_gap = abs(compound.selected_test_result["coverage_gap"])
    assert compound_gap > auto_alone_gap, (
        f"expected this specific documented case (synthetic, seed=2) where adding "
        f"fold_scheme='sliding' on top of auto_folds (window={compound.selected_window}, "
        f"|gap|={compound_gap:.4f}) does WORSE than auto_folds alone (window={auto_alone.selected_window}, "
        f"|gap|={auto_alone_gap:.4f}) -- if this no longer holds, the README's honest comparison "
        "needs updating to match, not this test loosened"
    )


def test_cv_window_selection_sliding_plus_auto_folds_loses_to_auto_alone_on_airline_seed_4():
    """Second loss case for the compound combination, on the smallest
    dataset this time -- included for the same not-just-one-case reason
    every other honest comparison in this test file pins more than a
    single example.
    """
    auto_alone = run_cv_window_selection("airline", windows=CANDIDATES, auto_folds=True, seed=4)
    compound = run_cv_window_selection(
        "airline", windows=CANDIDATES, fold_scheme="sliding", auto_folds=True, seed=4
    )
    auto_alone_gap = abs(auto_alone.selected_test_result["coverage_gap"])
    compound_gap = abs(compound.selected_test_result["coverage_gap"])
    assert compound_gap > auto_alone_gap, (
        f"expected this documented case (airline, seed=4) where the compound combination "
        f"(window={compound.selected_window}, |gap|={compound_gap:.4f}) does worse than auto_folds "
        f"alone (window={auto_alone.selected_window}, |gap|={auto_alone_gap:.4f})"
    )


def test_cv_window_selection_sliding_plus_auto_folds_matches_auto_alone_on_temperature_seed_0():
    """On `temperature` -- the largest calibration set, where `auto_folds`
    alone already ties the fixed default -- the compound combination also
    ties `auto_folds` alone on every one of the 6 seeds measured. Included
    so the compound finding isn't read as "always hurts": on this dataset
    it's simply orthogonal, same as `fold_scheme` was on its own (see the
    "Sliding vs. expanding fold scheme" section above).
    """
    auto_alone = run_cv_window_selection("temperature", windows=CANDIDATES, auto_folds=True, seed=0)
    compound = run_cv_window_selection(
        "temperature", windows=CANDIDATES, fold_scheme="sliding", auto_folds=True, seed=0
    )
    auto_alone_gap = abs(auto_alone.selected_test_result["coverage_gap"])
    compound_gap = abs(compound.selected_test_result["coverage_gap"])
    assert compound_gap == pytest.approx(auto_alone_gap, abs=1e-9), (
        f"expected the compound combination (|gap|={compound_gap:.4f}) to tie auto_folds alone "
        f"(|gap|={auto_alone_gap:.4f}) on this documented case (temperature, seed=0)"
    )
