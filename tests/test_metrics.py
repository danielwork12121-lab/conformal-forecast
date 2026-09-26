import numpy as np

from forecasting.metrics import empirical_coverage, mae, mape, mean_interval_width, rmse


def test_mae_rmse_zero_for_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0


def test_mae_known_value():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 2.0])
    assert abs(mae(y_true, y_pred) - (2.0 / 3.0)) < 1e-9


def test_rmse_known_value():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([3.0, 4.0])
    # sqrt(mean(9, 16)) = sqrt(12.5)
    assert abs(rmse(y_true, y_pred) - (12.5 ** 0.5)) < 1e-9


def test_mape_known_value():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([12.0, 18.0])
    # |2/10| and |2/20| -> 20% and 10% -> mean 15%
    assert abs(mape(y_true, y_pred) - 15.0) < 1e-6


def test_empirical_coverage_all_inside():
    y = np.array([1.0, 2.0, 3.0])
    lower = np.array([0.0, 1.0, 2.0])
    upper = np.array([2.0, 3.0, 4.0])
    assert empirical_coverage(y, lower, upper) == 1.0


def test_empirical_coverage_partial():
    y = np.array([1.0, 5.0, 3.0])
    lower = np.array([0.0, 1.0, 2.0])
    upper = np.array([2.0, 3.0, 4.0])
    # only indices 0 and 2 are inside -> 2/3
    assert abs(empirical_coverage(y, lower, upper) - (2.0 / 3.0)) < 1e-9


def test_mean_interval_width():
    lower = np.array([0.0, 1.0])
    upper = np.array([1.0, 3.0])
    assert mean_interval_width(lower, upper) == 1.5
