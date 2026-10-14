"""Tests for the v0.16 public `prepare_split_from_series` wrapper.

This wrapper exists so a downstream project depending on conformal-forecast
as a real library (not copy-pasting its internals) has a stable, public name
for the shared split/window/normalize/validate core, instead of reaching
into the underscore-prefixed `_prepare_split_from_series`. These tests only
need to confirm it's a behavior-identical pass-through -- the actual
split/window/normalize logic is already covered by the existing
`_prepare_split_from_series`-based tests in test_api.py and elsewhere.
"""
from __future__ import annotations

import numpy as np

from forecasting.data import generate_synthetic
from forecasting.experiment import _prepare_split_from_series, prepare_split_from_series


def test_public_wrapper_matches_private_function_exactly():
    series = generate_synthetic(n=400, period=24, seed=0)

    private_result = _prepare_split_from_series(series, lookback=24, period=24, label="series")
    public_result = prepare_split_from_series(series, lookback=24, period=24, label="series")

    assert len(private_result) == len(public_result) == 8
    for priv, pub in zip(private_result[:6], public_result[:6]):
        np.testing.assert_array_equal(priv, pub)
    # Index 6 is a Normalizer instance (fit on the same train split -> same params).
    assert private_result[6].mean_ == public_result[6].mean_
    assert private_result[6].std_ == public_result[6].std_
    # Index 7 is the (unchanged) period value.
    assert private_result[7] == public_result[7] == 24


def test_public_wrapper_raises_same_error_for_too_short_series():
    too_short = np.arange(10, dtype=np.float64)
    try:
        prepare_split_from_series(too_short, lookback=24, period=24, label="series")
        raised = False
    except ValueError:
        raised = True
    assert raised, "expected a ValueError for a series too short to window, same as the private function"
