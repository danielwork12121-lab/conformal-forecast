"""Tests for automatic window selection (v0.5) -- `select_best_window` /
`run_auto_window_selection` in `forecasting/experiment.py`.

Two layers, matching this repo's established split between fast/pure and
slow/real-data tests (see `test_sliding_window.py` and `test_window_sweep.py`):

1. `_select_best_window` is pure (no model, no data loading) and is tested
   directly and fast for its tie-break behavior -- this is where the actual
   selection *logic* lives, so it's worth pinning down exactly.
2. `run_auto_window_selection` is tested against the real bundled datasets,
   because the actual question this feature answers -- "does a holdout-based
   choice generalize to the real test set?" -- can only be checked with real
   data. Per this repo's "measure, don't assert" discipline, this includes a
   documented, honest FAILURE case (airline), not just success cases.
"""
from __future__ import annotations

import pytest

from forecasting.experiment import (
    _select_best_window,
    run_auto_window_selection,
)


# ---------------------------------------------------------------------------
# Layer 1: pure tie-break logic, no training involved.
# ---------------------------------------------------------------------------


def test_select_best_window_clear_winner_no_ties():
    windows = [5, 10, 20]
    gaps = {5: 0.05, 10: 0.01, 20: 0.03}
    assert _select_best_window(windows, gaps) == 10


def test_select_best_window_compares_absolute_value_not_signed():
    windows = [5, 10]
    gaps = {5: -0.02, 10: 0.015}
    # |0.015| < |-0.02|, so 10 wins even though 5's raw value looks smaller.
    assert _select_best_window(windows, gaps) == 10


def test_select_best_window_tie_without_widths_prefers_larger_window():
    windows = [5, 10, 20]
    gaps = {5: 0.01, 10: 0.01, 20: 0.01}
    assert _select_best_window(windows, gaps) == 20


def test_select_best_window_tie_without_widths_prefers_unbounded_over_any_finite_window():
    windows = [5, 10, 50, None]
    gaps = {5: 0.01, 10: 0.01, 50: 0.01, None: 0.01}
    assert _select_best_window(windows, gaps) is None


def test_select_best_window_tie_broken_by_smaller_width_before_larger_window():
    """The actual bug this test guards: without a width tie-break, a 4-way
    gap tie on the `synthetic` dataset (windows 10/20/50/unbounded) picked
    `unbounded`, the single worst real-test-set outcome among the tied
    group -- found during this feature's own development (see
    `active-project.md`). Width is a principled secondary criterion (prefer
    the sharper interval among equally-calibrated choices), and it resolves
    this exact tie correctly: the narrower-width candidate should win even
    though it's a *smaller* window than `None`.
    """
    windows = [10, 20, 50, None]
    gaps = {10: 0.02, 20: 0.02, 50: 0.02, None: 0.02}
    widths = {10: 0.54, 20: 0.57, 50: 0.545, None: 0.553}
    assert _select_best_window(windows, gaps, widths) == 10


def test_select_best_window_full_tie_falls_back_to_larger_window():
    windows = [5, 10, 20, None]
    gaps = {5: 0.02, 10: 0.02, 20: 0.02, None: 0.02}
    widths = {5: 1.0, 10: 1.0, 20: 1.0, None: 1.0}  # identical widths too
    assert _select_best_window(windows, gaps, widths) is None


def test_select_best_window_empty_candidates_raises():
    with pytest.raises(ValueError):
        _select_best_window([], {})


# ---------------------------------------------------------------------------
# Layer 2: real-data integration, via run_auto_window_selection.
# ---------------------------------------------------------------------------

CANDIDATES = [5, 7, 10, 15, 20, 30, 50, None]


def test_auto_window_selection_runs_and_reports_every_candidate():
    result = run_auto_window_selection("temperature", windows=[5, 10, 20, 30, 50, None], seed=0)

    assert result.dataset == "temperature"
    assert [c.window for c in result.candidates] == [5, 10, 20, 30, 50, None]
    assert result.selected_window in [5, 10, 20, 30, 50, None]
    assert result.best_test_window in [5, 10, 20, 30, 50, None]
    for c in result.candidates:
        assert -1.0 <= c.holdout_coverage_gap <= 1.0
        assert -1.0 <= c.test_coverage_gap <= 1.0
        assert c.holdout_mean_interval_width > 0.0


def test_auto_window_selection_holdout_and_selection_cal_partition_the_calibration_set():
    result = run_auto_window_selection("temperature", windows=[10, 20, None], holdout_frac=0.3, seed=0)
    # 0.2 of the full series is calibration (chronological_split's default
    # cal_frac); the holdout split then further divides *that* into
    # select_cal / select_holdout at the given fraction. Both pieces must be
    # positive and roughly match the requested fraction.
    total = result.n_select_cal + result.n_select_holdout
    assert total > 0
    assert result.n_select_cal >= 5 and result.n_select_holdout >= 5
    observed_frac = result.n_select_holdout / total
    assert abs(observed_frac - 0.3) < 0.05


def test_auto_window_selection_too_small_holdout_raises():
    # airline's calibration set has only 28 windows; an extreme holdout_frac
    # pushes one side below the minimum of 5 required for a meaningful score.
    with pytest.raises(ValueError):
        run_auto_window_selection("airline", windows=[5, 10], holdout_frac=0.98, seed=0)
    with pytest.raises(ValueError):
        run_auto_window_selection("airline", windows=[5, 10], holdout_frac=0.02, seed=0)


def test_auto_window_selection_invalid_holdout_frac_raises():
    with pytest.raises(ValueError):
        run_auto_window_selection("temperature", windows=[5, 10], holdout_frac=0.0, seed=0)
    with pytest.raises(ValueError):
        run_auto_window_selection("temperature", windows=[5, 10], holdout_frac=1.0, seed=0)


def test_auto_window_selection_matches_best_in_hindsight_on_temperature():
    """The clean success case: temperature's calibration set is large enough
    (511 selection-calibration windows, 219-point holdout at the default
    0.3 fraction) for the holdout slice to actually discriminate between
    candidates. The automatically selected window's real test-set gap
    should land close to the oracle best-in-hindsight's gap -- not
    necessarily identical (the holdout is a proxy, not the test set
    itself), but clearly in the same regime, not a bad pick.
    """
    result = run_auto_window_selection("temperature", windows=CANDIDATES[:-2] + [None], seed=0)
    best_gap = result.selected_test_result["coverage_gap"]
    oracle_gap = [c.test_coverage_gap for c in result.candidates if c.window == result.best_test_window][0]
    assert abs(best_gap - oracle_gap) <= 0.02, (
        f"selected window={result.selected_window} real test gap={best_gap:+.4f} vs. "
        f"best-in-hindsight window={result.best_test_window} gap={oracle_gap:+.4f} -- "
        "expected the holdout-based choice to land within 2pp of the oracle on a dataset "
        "whose holdout slice is large enough to discriminate between candidates"
    )


def test_auto_window_selection_beats_unbounded_on_synthetic():
    """A second success case, and the comparison this feature exists to
    automate: the selected window's real test-set |gap| should be smaller
    than the unbounded growing buffer's -- the same "bounded can beat
    unbounded" finding `test_window_sweep.py` already established manually,
    now checked for the *automatically selected* window specifically.
    """
    result = run_auto_window_selection("synthetic", windows=CANDIDATES, seed=0)
    unbounded_gap = [c.test_coverage_gap for c in result.candidates if c.window is None][0]
    selected_gap = result.selected_test_result["coverage_gap"]
    assert abs(selected_gap) < abs(unbounded_gap), (
        f"selected window={result.selected_window} real test gap={selected_gap:+.4f} vs. "
        f"unbounded gap={unbounded_gap:+.4f} -- expected the automatically selected window "
        "to beat the unbounded buffer on synthetic, per this run's own measurement"
    )


def test_auto_window_selection_on_airline_is_honestly_a_near_tie_across_candidates():
    """Documented limitation, not a bug: airline's calibration set (28
    windows) is small enough that its holdout slice sees essentially
    identical hit/miss outcomes for every candidate window, regardless of
    `holdout_frac` (checked at 0.3/0.4/0.5/0.6 during development -- all
    produced the exact same holdout gap for every single candidate).
    Structurally, this isn't surprising: the holdout slice is still drawn
    from the *calibration* period, and airline's whole story (see the
    README) is that calibration-period residuals are smaller than
    test-period residuals -- the one thing a calibration-only holdout
    cannot see is a shift that hasn't happened yet. This test pins down
    that every candidate really does tie on the holdout metric here, so a
    future change that silently breaks this observation (e.g. a selection
    method that starts overfitting noise on a tiny holdout) gets caught.
    """
    result = run_auto_window_selection("airline", windows=CANDIDATES, seed=0)
    holdout_gaps = {round(c.holdout_coverage_gap, 6) for c in result.candidates}
    assert len(holdout_gaps) == 1, (
        f"expected every candidate to tie on holdout gap for airline (n_select_holdout="
        f"{result.n_select_holdout} is too small to discriminate), got distinct values: {holdout_gaps}"
    )
