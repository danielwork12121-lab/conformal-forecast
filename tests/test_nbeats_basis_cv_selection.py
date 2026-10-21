"""Tests for `run_nbeats_basis_cv_selection`/`format_nbeats_basis_cv_selection`
(v0.30) -- the real follow-up v0.29's own README "What's next" named:
every `trend_degree`/`num_harmonics` sweep so far (v0.25, v0.26, v0.28's
joint grid) reads its "best" cell straight off the real test set, which
is honest for *reporting* the hyperparameters' effect but isn't a
deployable selection method. This mirrors `run_cv_window_selection`'s
rolling-origin cross-validation (already shipped for ACI's `window`) to
select `(trend_degree, num_harmonics)` from the calibration set alone.

`_select_best_cell` is pure (no model/data dependency) and is tested
directly and fast, same two-layer split `test_cv_window_selection.py`
already uses for `_select_best_window`. `run_nbeats_basis_cv_selection`
itself is tested against the real bundled datasets, since the question it
answers -- "does a held-out-data method recover the real test set's own
best cell?" -- can only be checked with real data; per this repo's
"measure, don't assert" discipline, these tests pin down structural
correctness (every cell evaluated, the selection never touches X_test,
each cell gets its own independent fit) rather than asserting the CV
pick always matches the oracle.
"""
from __future__ import annotations

import pytest

from forecasting.experiment import (
    NBEATSBasisCVSelectionResult,
    _select_best_cell,
    format_nbeats_basis_cv_selection,
    run_nbeats_basis_cv_selection,
    run_nbeats_joint_basis_sweep_comparison,
)


# ---------------------------------------------------------------------------
# Layer 1: pure selection logic, no training involved.
# ---------------------------------------------------------------------------


def test_select_best_cell_picks_smallest_abs_gap():
    keys = [(1, 1), (2, 1), (3, 2)]
    gap_by_key = {(1, 1): 0.20, (2, 1): -0.05, (3, 2): 0.10}
    assert _select_best_cell(keys, gap_by_key) == (2, 1)


def test_select_best_cell_tie_breaks_on_width_then_complexity():
    keys = [(1, 1), (2, 2), (1, 2)]
    # (1, 1) and (2, 2) tie exactly on |gap|; (2, 2) has the smaller width, so it wins.
    gap_by_key = {(1, 1): 0.05, (2, 2): -0.05, (1, 2): 0.05}
    width_by_key = {(1, 1): 2.0, (2, 2): 1.0, (1, 2): 2.0}
    assert _select_best_cell(keys, gap_by_key, width_by_key) == (2, 2)

    # With no width given (or all tied), the simplest combination (lowest
    # trend_degree + num_harmonics) wins as the last-resort tie-break.
    gap_by_key2 = {(1, 1): 0.05, (2, 2): 0.05, (1, 2): 0.05}
    assert _select_best_cell(keys, gap_by_key2) == (1, 1)


def test_select_best_cell_empty_raises():
    with pytest.raises(ValueError):
        _select_best_cell([], {})


# ---------------------------------------------------------------------------
# Layer 2: integration against real bundled data.
# ---------------------------------------------------------------------------


def test_selection_runs_every_requested_cell_and_reports_real_numbers():
    trend_degrees = [1, 2, 3]
    num_harmonics_values = [1, 2]
    result = run_nbeats_basis_cv_selection(
        "synthetic",
        trend_degrees=trend_degrees,
        num_harmonics_values=num_harmonics_values,
        alpha=0.1,
        n_folds=3,
        seed=0,
    )

    assert isinstance(result, NBEATSBasisCVSelectionResult)
    assert len(result.candidates) == len(trend_degrees) * len(num_harmonics_values)
    seen = {(c.trend_degree, c.num_harmonics) for c in result.candidates}
    assert seen == {(d, h) for d in trend_degrees for h in num_harmonics_values}
    for c in result.candidates:
        assert c.mean_fold_width > 0.0
    assert result.selected in seen
    assert result.best_test in seen


def test_selected_cell_matches_the_fold_columns_own_argmin():
    """Direct regression guard that `selected` is actually computed from
    `mean_abs_fold_gap` (the fold columns), not accidentally from
    `test_coverage_gap` (which would silently let the selection peek at
    the test set)."""
    result = run_nbeats_basis_cv_selection(
        "airline",
        trend_degrees=[1, 2, 3],
        num_harmonics_values=[1, 2],
        alpha=0.1,
        n_folds=3,
        seed=0,
    )
    by_key = {(c.trend_degree, c.num_harmonics): c for c in result.candidates}
    expected = min(by_key, key=lambda k: (round(abs(by_key[k].mean_abs_fold_gap), 10), by_key[k].mean_fold_width, k[0] + k[1]))
    assert result.selected == expected


def test_every_cell_gets_its_own_independent_fit():
    """Same regression guard the block-count-sweep test file uses: varying
    either hyperparameter alone must change the mean fold width (a direct
    artifact of a real, independent fit -- not one fit relabeled across
    cells)."""
    result = run_nbeats_basis_cv_selection(
        "airline",
        trend_degrees=[1, 5],
        num_harmonics_values=[1, 5],
        alpha=0.1,
        n_folds=3,
        seed=0,
    )
    by_key = {(c.trend_degree, c.num_harmonics): c for c in result.candidates}
    assert by_key[(1, 1)].mean_fold_width != by_key[(5, 1)].mean_fold_width
    assert by_key[(1, 1)].mean_fold_width != by_key[(1, 5)].mean_fold_width


def test_fold_scheme_and_auto_folds_options_pass_through():
    result = run_nbeats_basis_cv_selection(
        "temperature",
        trend_degrees=[2],
        num_harmonics_values=[1],
        alpha=0.1,
        fold_scheme="sliding",
        auto_folds=True,
        seed=0,
    )
    assert result.fold_scheme == "sliding"
    assert result.auto_folds is True
    assert result.n_folds >= 1


def test_too_small_calibration_set_raises():
    # airline's calibration set (28 windows): reserving 80% of it as the
    # first fold's seed (min_initial=22) leaves only 6 points usable, below
    # the 14-point minimum fold size an oversized min_fold_frac=0.5 demands
    # -- same configuration `test_cv_window_selection.py` uses for the
    # analogous ACI-window case.
    with pytest.raises(ValueError):
        run_nbeats_basis_cv_selection(
            "airline",
            trend_degrees=[2],
            num_harmonics_values=[1],
            alpha=0.1,
            n_folds=4,
            min_initial_frac=0.8,
            min_fold_frac=0.5,
            seed=0,
        )


def test_invalid_args_raise():
    with pytest.raises(ValueError):
        run_nbeats_basis_cv_selection("temperature", trend_degrees=[2], num_harmonics_values=[1], n_folds=0, seed=0)
    with pytest.raises(ValueError):
        run_nbeats_basis_cv_selection(
            "temperature", trend_degrees=[2], num_harmonics_values=[1], min_initial_frac=0.0, seed=0
        )
    with pytest.raises(ValueError):
        run_nbeats_basis_cv_selection(
            "temperature", trend_degrees=[2], num_harmonics_values=[1], min_fold_frac=1.0, seed=0
        )
    with pytest.raises(ValueError):
        run_nbeats_basis_cv_selection(
            "temperature", trend_degrees=[2], num_harmonics_values=[1], fold_scheme="bogus", seed=0
        )


def test_format_includes_every_cell_the_generic_reference_and_both_markers():
    result = run_nbeats_basis_cv_selection(
        "synthetic",
        trend_degrees=[1, 2],
        num_harmonics_values=[1, 2],
        alpha=0.1,
        n_folds=3,
        seed=0,
    )
    text = format_nbeats_basis_cv_selection(result)
    assert "generic N-BEATS" in text
    assert "| 1 | 1" in text
    assert "| 2 | 2" in text
    assert "selected" in text
    assert "Selected: trend_degree=" in text


def test_single_cell_grid_selected_test_result_matches_joint_sweep():
    """`selected_test_result` for a single-candidate grid must exactly
    match `run_nbeats_joint_basis_sweep_comparison`'s own already-tested
    cell for the same (dataset, trend_degree, num_harmonics, seed) --
    confirms the post-hoc check is reusing that real, already-verified
    machinery rather than recomputing it differently."""
    result = run_nbeats_basis_cv_selection(
        "airline", trend_degrees=[2], num_harmonics_values=[1], alpha=0.1, n_folds=3, seed=0
    )
    joint = run_nbeats_joint_basis_sweep_comparison(
        "airline", trend_degrees=[2], num_harmonics_values=[1], alpha=0.1, seed=0
    )
    assert result.selected_test_result.coverage_gap == joint.cells[0].coverage_gap
    assert result.selected_test_result.mae == joint.cells[0].mae
