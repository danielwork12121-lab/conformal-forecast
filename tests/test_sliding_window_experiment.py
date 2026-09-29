"""Integration test: `run_sliding_window_comparison` wired to the real
airline dataset -- the exact case flagged in the README and in
`tests/test_adaptive_experiment.py` as the one where fixed-pool ACI barely
moves the needle (63.3% -> 60.0% empirical coverage, essentially noise, with
only 28 calibration windows).

The honest result checked here, and reported in the README: folding each
test step's own observed residual into the pool (removing the fixed-pool
ceiling) recovers real coverage on this exact dataset -- verified against
the actual numbers produced by a live run, not a number invented ahead of
time. This isn't a universal fix (see `tests/test_sliding_window.py`'s own
docstring on aggregate-vs-per-seed framing, and the README's temperature
row, where there's no gap to close in the first place), but on the one case
this repo's whole "what's next" section was built around, it works.
"""
from __future__ import annotations

from forecasting.experiment import run_sliding_window_comparison


def test_sliding_window_meaningfully_improves_airline_coverage_over_fixed_pool():
    comparison = run_sliding_window_comparison("airline", alpha=0.1, gamma=0.05, window=None, seed=0)

    # Both variants actually ran and produced real coverage numbers.
    assert 0.0 <= comparison.static["empirical_coverage"] <= 1.0
    assert 0.0 <= comparison.fixed_pool["empirical_coverage"] <= 1.0
    assert 0.0 <= comparison.sliding["empirical_coverage"] <= 1.0
    assert len(comparison.pool_size) == comparison.static["n_test"]

    # The pool actually grew (this dataset has n_cal=28, n_test=30 -- see
    # DEFAULT_LOOKBACKS/_prepare_split -- so it should grow from 28 to 57).
    assert comparison.pool_size[0] == 28
    assert comparison.pool_size[-1] == 57
    assert list(comparison.pool_size) == list(range(28, 58))

    # The actual claim: sliding-pool ACI closes a real, substantial fraction
    # of the gap that fixed-pool ACI left almost entirely untouched here.
    static_gap = abs(comparison.static["coverage_gap"])
    fixed_gap = abs(comparison.fixed_pool["coverage_gap"])
    sliding_gap = abs(comparison.sliding["coverage_gap"])

    assert sliding_gap < fixed_gap, (
        f"sliding-pool coverage_gap={comparison.sliding['coverage_gap']:.3f} is not smaller in magnitude "
        f"than fixed-pool's {comparison.fixed_pool['coverage_gap']:.3f} on airline -- expected the pool "
        "update to actually help on the exact case that motivated building it"
    )
    # A generous but bounded bar: on a 30-point test set, one extra hit/miss
    # is a 3.3pp swing, so this checks for a real, multi-point recovery, not
    # single-point noise -- while not hard-coding the exact live number (see
    # README for the specific figures this test's own run produced).
    assert sliding_gap < static_gap * 0.6, (
        f"static gap={comparison.static['coverage_gap']:.3f}, sliding gap={comparison.sliding['coverage_gap']:.3f} "
        "-- expected sliding-pool ACI to recover a real, substantial fraction of static's coverage shortfall "
        "on airline, the dataset that originally motivated this work"
    )
