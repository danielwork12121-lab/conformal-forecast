"""Integration test: `run_online_retraining_comparison` wired to the real
LSTM and the real airline dataset -- the exact README "What's next" item
this feature answers: does letting the point model itself keep learning
from newly-revealed test data (not just the conformal layer around it)
actually help, measured on real data, not assumed.

Mirrors `tests/test_sliding_window_experiment.py`'s discipline: confirm the
real mechanics (pool growth, retrain cadence) match what the dataset's own
known sizes predict, then report the real, measured result -- whichever
way it comes out -- rather than asserting a particular winner ahead of
time. The README is where the honest, reported conclusion lives; this test
only pins down that the comparison actually ran and produced sane,
internally-consistent numbers.
"""
from __future__ import annotations

from forecasting.experiment import run_online_retraining_comparison


def test_online_retraining_runs_end_to_end_on_airline_with_resolved_cadence():
    comparison = run_online_retraining_comparison(
        "airline", alpha=0.1, gamma=0.05, window=None, n_retrains_target=8, seed=0
    )

    # Airline has n_cal=28, n_test=30 (see test_sliding_window_experiment.py).
    # retrain_every is resolved from n_retrains_target: max(1, 30 // 8) = 3.
    assert comparison.n_test == 30
    assert comparison.retrain_every == 3
    # Retrains fire at t=3,6,...,27 (never at t=0, which .fit() already covers,
    # and never beyond the last test index, 29).
    assert comparison.n_retrains == 9
    assert list(comparison.retrained_at_steps) == [3, 6, 9, 12, 15, 18, 21, 24, 27]

    # The pool-growth mechanics are identical to the frozen sliding-pool
    # variant (retraining the model and sliding the pool are independent
    # mechanisms) -- same assertion as test_sliding_window_experiment.py.
    assert len(comparison.pool_size) == comparison.n_test
    assert comparison.pool_size[0] == 28
    assert comparison.pool_size[-1] == 57
    assert list(comparison.pool_size) == list(range(28, 58))

    # All three methods actually ran and produced real, well-formed
    # coverage numbers.
    for result in (comparison.static, comparison.frozen_sliding, comparison.retraining):
        assert 0.0 <= result["empirical_coverage"] <= 1.0
        assert result["n_test"] == comparison.n_test
        assert result["mean_interval_width"] >= 0.0


def test_online_retraining_honors_an_explicit_retrain_every_over_the_target():
    """When `retrain_every` is given explicitly, it should be used as-is,
    not overridden by `n_retrains_target` -- the target is only a fallback
    for resolving a cadence when the caller doesn't want to hand-pick one.
    """
    comparison = run_online_retraining_comparison(
        "airline", alpha=0.1, gamma=0.05, window=None, retrain_every=5, n_retrains_target=8, seed=0
    )
    assert comparison.retrain_every == 5
    # n_test=30: retrains fire at t=5,10,15,20,25 (5 of them).
    assert comparison.n_retrains == 5
    assert list(comparison.retrained_at_steps) == [5, 10, 15, 20, 25]
