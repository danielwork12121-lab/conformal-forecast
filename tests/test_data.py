import numpy as np
import pytest

from forecasting.data import (
    Normalizer,
    chronological_split,
    generate_synthetic,
    load_airline_passengers,
    load_daily_min_temperatures,
    make_windows,
)


def test_make_windows_shapes_and_alignment():
    series = np.arange(10, dtype=np.float64)
    X, y = make_windows(series, lookback=3, horizon=1)
    assert X.shape == (7, 3)
    assert y.shape == (7, 1)
    # window 0: X=[0,1,2] -> y=[3]
    np.testing.assert_array_equal(X[0], [0, 1, 2])
    np.testing.assert_array_equal(y[0], [3])
    # last window: X=[6,7,8] -> y=[9]
    np.testing.assert_array_equal(X[-1], [6, 7, 8])
    np.testing.assert_array_equal(y[-1], [9])


def test_make_windows_too_short_raises():
    with pytest.raises(ValueError):
        make_windows(np.arange(3), lookback=5, horizon=1)


def test_chronological_split_preserves_order_and_sizes():
    series = np.arange(100, dtype=np.float64)
    split = chronological_split(series, train_frac=0.6, cal_frac=0.2)
    assert len(split.train) == 60
    assert len(split.calibration) == 20
    assert len(split.test) == 20
    # order preserved: train comes before calibration comes before test
    assert split.train[-1] < split.calibration[0]
    assert split.calibration[-1] < split.test[0]
    np.testing.assert_array_equal(split.full(), series)


def test_normalizer_round_trip():
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 2.0, size=200)
    norm = Normalizer().fit(x)
    z = norm.transform(x)
    assert abs(z.mean()) < 1e-6
    assert abs(z.std() - 1.0) < 1e-6
    recovered = norm.inverse_transform(z)
    np.testing.assert_allclose(recovered, x, atol=1e-8)


def test_normalizer_fit_on_train_only_can_diverge_from_full_series_stats():
    # Regression guard: Normalizer must not silently refit on whatever it's
    # last given -- transform() should keep using the *fit* mean/std even
    # when handed different data.
    train = np.array([1.0, 2.0, 3.0])
    other = np.array([100.0, 200.0, 300.0])
    norm = Normalizer().fit(train)
    z_other = norm.transform(other)
    assert z_other[0] != pytest.approx(0.0, abs=1e-6)


def test_generate_synthetic_is_deterministic_given_seed():
    a = generate_synthetic(n=50, seed=1)
    b = generate_synthetic(n=50, seed=1)
    c = generate_synthetic(n=50, seed=2)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_bundled_real_datasets_load_and_have_expected_shape():
    airline = load_airline_passengers()
    assert airline.shape == (144,)
    assert airline.min() > 0  # passenger counts are positive

    temps = load_daily_min_temperatures()
    assert temps.shape == (3650,)
    assert -10 < temps.min() and temps.max() < 40  # sane Celsius range for Melbourne
