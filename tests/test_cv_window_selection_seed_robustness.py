"""Tests for whether `run_cv_window_selection`'s own pick (v0.6-v0.9) is
robust to the base LSTM's random training seed -- v0.31,
`run_cv_window_selection_seed_robustness` /
`format_cv_window_selection_seed_robustness` / `_unique_in_order` in
`forecasting/experiment.py`. The README's own "What's next" named this
open question directly: every prior CV-window-selection result in this
repo was only ever checked at seed=0.

Same two-layer split as `test_cv_window_selection.py`:

1. `_unique_in_order` is pure (no model, no data loading) and is tested
   directly and fast for its own bookkeeping logic.
2. `run_cv_window_selection_seed_robustness` is tested against the real
   bundled datasets, because the actual question -- does a different
   random LSTM initialization change which window the fold metric picks?
   -- can only be answered with real data, not asserted.
"""
from __future__ import annotations

import pytest

from forecasting.experiment import (
    _unique_in_order,
    run_cv_window_selection,
    run_cv_window_selection_seed_robustness,
)


# ---------------------------------------------------------------------------
# Layer 1: pure bookkeeping logic, no training involved.
# ---------------------------------------------------------------------------


def test_unique_in_order_dedupes_preserving_first_seen_order():
    assert _unique_in_order([10, 30, 10, None, 30, 10]) == [10, 30, None]


def test_unique_in_order_all_same_returns_one_element():
    assert _unique_in_order([20, 20, 20]) == [20]


def test_unique_in_order_empty_list():
    assert _unique_in_order([]) == []


def test_unique_in_order_handles_none_as_a_real_value():
    # None means "unbounded growing buffer" throughout this repo -- it must
    # be treated as a normal, distinct value, not skipped or coerced.
    assert _unique_in_order([None, None, 10]) == [None, 10]


# ---------------------------------------------------------------------------
# Layer 2: real-data integration, via run_cv_window_selection_seed_robustness.
# ---------------------------------------------------------------------------

CANDIDATES = [5, 7, 10, 15, 20, 30, 50, None]


def test_seed_robustness_requires_a_non_empty_seed_list():
    with pytest.raises(ValueError):
        run_cv_window_selection_seed_robustness("synthetic", windows=[10, 30], seeds=[])


def test_seed_robustness_one_seed_matches_run_cv_window_selection_exactly():
    """Cross-check against the already-tested single-seed function, same
    discipline `run_nbeats_basis_cv_selection` uses against
    `run_nbeats_joint_basis_sweep_comparison`: with exactly one seed, this
    function's per-seed numbers must be the plain-old result read back
    verbatim, not recomputed a second way.
    """
    direct = run_cv_window_selection("synthetic", windows=[10, 30, None], seed=3)
    robustness = run_cv_window_selection_seed_robustness("synthetic", windows=[10, 30, None], seeds=[3])

    assert len(robustness.per_seed) == 1
    only = robustness.per_seed[0]
    assert only.seed == 3
    assert only.selected_window == direct.selected_window
    assert only.selected_test_gap == direct.selected_test_result["coverage_gap"]
    assert only.best_test_window == direct.best_test_window
    assert robustness.unique_selected_windows == [direct.selected_window]
    assert robustness.is_window_robust is True  # trivially true with one seed


def test_seed_robustness_reports_one_row_per_seed():
    seeds = [0, 1, 2]
    result = run_cv_window_selection_seed_robustness("synthetic", windows=CANDIDATES, seeds=seeds)

    assert result.dataset == "synthetic"
    assert result.seeds == seeds
    assert len(result.per_seed) == len(seeds)
    assert [c.seed for c in result.per_seed] == seeds
    for c in result.per_seed:
        assert c.selected_window in CANDIDATES
        assert c.best_test_window in CANDIDATES
        assert -1.0 <= c.selected_test_gap <= 1.0


def test_seed_robustness_verdict_is_consistent_with_unique_windows():
    # Structural invariant, true regardless of which real dataset/windows
    # are measured: the robustness flag and the de-duplicated list must
    # always agree.
    result = run_cv_window_selection_seed_robustness("synthetic", windows=CANDIDATES, seeds=[0, 1, 2, 3, 4])
    assert result.is_window_robust == (len(result.unique_selected_windows) == 1)
    assert result.unique_selected_windows == _unique_in_order([c.selected_window for c in result.per_seed])


def test_seed_robustness_is_deterministic_for_a_fixed_seed_list():
    a = run_cv_window_selection_seed_robustness("temperature", windows=[10, 30, None], seeds=[0, 1])
    b = run_cv_window_selection_seed_robustness("temperature", windows=[10, 30, None], seeds=[0, 1])
    assert [c.selected_window for c in a.per_seed] == [c.selected_window for c in b.per_seed]
    assert [c.selected_test_gap for c in a.per_seed] == [c.selected_test_gap for c in b.per_seed]


def test_seed_robustness_propagates_fold_scheme_and_auto_folds_to_every_seed():
    result = run_cv_window_selection_seed_robustness(
        "synthetic", windows=[10, 30], seeds=[0, 1], fold_scheme="sliding", auto_folds=False
    )
    assert result.fold_scheme == "sliding"
    assert result.auto_folds is False
    # Cross-check seed 0's row against the single-seed function called the
    # same way, to confirm the kwarg actually reached the underlying call.
    direct = run_cv_window_selection("synthetic", windows=[10, 30], seed=0, fold_scheme="sliding")
    assert result.per_seed[0].selected_window == direct.selected_window
