"""Integration test: `run_adaptive_comparison` wired to a real dataset.

`tests/test_adaptive.py` proves ACI's mechanics and statistical behavior in
isolation, including a synthetic scenario where it substantially reduces a
coverage gap under real distribution drift. This file checks the actual,
real-world case flagged in the README: on the airline dataset,
`forecasting.experiment.run_experiment` already shows the LSTM
under-covering (63.3% vs. 90% nominal).

The honest result here, verified below and reported in the README, is that
this module's "fixed calibration pool" ACI variant (see its own docstring)
does *not* meaningfully fix that specific case: with only ~28 calibration
windows, `_quantile_with_finite_sample_correction` already sits within a
couple of residuals of the pool's true max at the default alpha=0.1, so
there's very little headroom left for ACI to adapt into. This test checks
the mechanism worked correctly (alpha_t reacted to misses, coverage didn't
get meaningfully worse) rather than asserting an improvement this specific,
small-sample case doesn't actually produce -- exactly the "measure, don't
assert" discipline the rest of this repo follows.
"""
from __future__ import annotations

from forecasting.experiment import run_adaptive_comparison


def test_adaptive_comparison_reacts_without_meaningfully_worsening_coverage_on_airline():
    comparison = run_adaptive_comparison("airline", alpha=0.1, gamma=0.05, seed=0)

    # Both methods actually ran and produced real coverage numbers.
    assert 0.0 <= comparison.static["empirical_coverage"] <= 1.0
    assert 0.0 <= comparison.adaptive["empirical_coverage"] <= 1.0
    assert len(comparison.alpha_t) == comparison.static["n_test"]

    # ACI visibly reacted to the known under-coverage (alpha_t dropped well
    # below the nominal alpha at some point) -- it isn't a no-op here, even
    # though (see module docstring) reacting doesn't buy much headroom with
    # only ~28 calibration windows.
    assert comparison.alpha_t.min() < comparison.alpha - 1e-9, (
        "ACI's alpha_t never dropped below the nominal alpha on airline -- "
        "expected it to react to the known under-coverage by widening"
    )

    # With a test set of only 30 points, one extra hit/miss is a 3.3
    # percentage-point swing -- allow a generous but bounded tolerance so
    # this test catches a real regression (e.g. a sign error making ACI
    # much worse) without being sensitive to that single-point noise.
    assert abs(comparison.adaptive["coverage_gap"]) <= abs(comparison.static["coverage_gap"]) + 0.1, (
        f"adaptive coverage_gap={comparison.adaptive['coverage_gap']:.3f} is meaningfully worse than "
        f"static's {comparison.static['coverage_gap']:.3f} -- more than noise from n_test=30 can explain"
    )
