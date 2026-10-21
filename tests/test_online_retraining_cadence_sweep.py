"""Integration tests for `run_online_retraining_cadence_sweep` (v0.34) --
v0.32's own "What's next" follow-up, left open at the time: does a *finer*
retraining cadence than the shipped `n_retrains_target=8` default close
more of the coverage gap, or has the coarse default already captured most
of the real benefit? This sweeps `retrain_every` directly on the real
airline dataset (small enough to afford every candidate down to
`retrain_every=1`, the literal every-step case) and checks the real,
measured mechanics and shape of the result -- not a particular winner
assumed ahead of time, same discipline as `test_window_sweep.py`.
"""
from __future__ import annotations

import pytest

from forecasting.experiment import run_online_retraining_cadence_sweep


def test_cadence_sweep_runs_and_reports_real_numbers_for_every_candidate():
    candidates = [10, 5, 2, 1]
    sweep = run_online_retraining_cadence_sweep("airline", candidates, seed=0)

    assert sweep.dataset == "airline"
    assert sweep.n_test == 30
    assert [r.retrain_every for r in sweep.results] == candidates
    for r in sweep.results:
        assert 0.0 <= r.adaptive["empirical_coverage"] <= 1.0
        assert r.adaptive["mean_interval_width"] > 0.0
        assert r.n_retrains >= 0

    # Reference lines don't depend on retrain_every at all -- computed once,
    # shared by every row, same pattern as run_window_sweep_comparison's
    # static/fixed_pool reference lines.
    assert 0.0 <= sweep.static["empirical_coverage"] <= 1.0
    assert 0.0 <= sweep.frozen_sliding["empirical_coverage"] <= 1.0


def test_finer_retrain_every_strictly_increases_the_real_retrain_count():
    """A smaller retrain_every must never produce *fewer* real retrains over
    the same fixed test window than a larger one -- a direct, hand-checkable
    regression test for the sweep wiring itself (not the model), since
    n_test=30 makes the exact counts easy to verify by hand: retrains fire
    at every positive multiple of retrain_every below n_test=30.
    """
    sweep = run_online_retraining_cadence_sweep("airline", [10, 5, 2, 1], seed=0)
    counts = {r.retrain_every: r.n_retrains for r in sweep.results}

    assert counts[10] == 2  # t=10,20
    assert counts[5] == 5  # t=5,10,15,20,25
    assert counts[2] == 14  # t=2,4,...,28
    assert counts[1] == 29  # t=1,2,...,29 (every step after the initial fit)
    # Monotonically non-decreasing as the cadence gets finer (candidates
    # given coarsest-to-finest here).
    assert counts[10] <= counts[5] <= counts[2] <= counts[1]


def test_retrain_every_one_is_the_literal_every_step_case():
    """`retrain_every=1` must retrain at every single test step after the
    first (t=0 is already covered by the initial .fit()) -- the sweep's
    finest candidate should match RetrainingSlidingWindowAdaptiveConformalForecaster's
    own documented every-step case exactly, not an approximation of it.
    """
    sweep = run_online_retraining_cadence_sweep("airline", [1], seed=0)
    result = sweep.results[0]
    assert result.retrain_every == 1
    assert result.n_retrains == sweep.n_test - 1


def test_two_sweeps_with_the_same_seed_and_candidates_are_reproducible():
    """Same discipline as every other deterministic-given-a-seed feature in
    this repo: running the sweep twice with identical inputs must produce
    byte-identical real numbers, not just "close" ones -- a silent
    non-determinism regression (e.g. an un-seeded model init creeping back
    in) would show up here immediately.
    """
    sweep_a = run_online_retraining_cadence_sweep("airline", [5, 1], seed=0)
    sweep_b = run_online_retraining_cadence_sweep("airline", [5, 1], seed=0)

    assert [r.adaptive["empirical_coverage"] for r in sweep_a.results] == [
        r.adaptive["empirical_coverage"] for r in sweep_b.results
    ]
    assert [r.n_retrains for r in sweep_a.results] == [r.n_retrains for r in sweep_b.results]
    assert sweep_a.static["empirical_coverage"] == sweep_b.static["empirical_coverage"]


def test_rejects_a_non_positive_retrain_every_candidate():
    with pytest.raises(ValueError, match="positive integer"):
        run_online_retraining_cadence_sweep("airline", [5, 0, 1], seed=0)
    with pytest.raises(ValueError, match="positive integer"):
        run_online_retraining_cadence_sweep("airline", [-3], seed=0)


def test_rejects_an_empty_candidate_list():
    with pytest.raises(ValueError, match="non-empty"):
        run_online_retraining_cadence_sweep("airline", [], seed=0)
