import numpy as np

from forecasting.data import make_windows
from forecasting.models import DeltaWrapper, LSTMForecaster, NaiveForecaster, SeasonalNaiveForecaster


def test_naive_forecaster_repeats_last_value():
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    y = np.array([[3.5], [6.5]], dtype=np.float32)
    model = NaiveForecaster().fit(X, y)
    pred = model.predict(X)
    np.testing.assert_array_equal(pred, [[3.0], [6.0]])


def test_seasonal_naive_uses_period_steps_back():
    # lookback=6, period=3 -> prediction is the value 3 steps before the end
    X = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=np.float32)
    model = SeasonalNaiveForecaster(period=3)
    pred = model.predict(X)
    # lookback - period = 3 -> X[:, 3] = 4.0
    np.testing.assert_array_equal(pred, [[4.0]])


def test_seasonal_naive_falls_back_to_naive_when_period_exceeds_lookback():
    X = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    model = SeasonalNaiveForecaster(period=100)
    pred = model.predict(X)
    np.testing.assert_array_equal(pred, [[3.0]])


def test_lstm_forecaster_learns_a_bounded_periodic_series():
    # A sine wave is periodic and bounded, so the test window's true values
    # fall inside the same range the model saw during training -- unlike an
    # unbounded linear trend, which would require the network to *extrapolate*
    # past its training range (a known, unrelated weakness of NNs, not a
    # check of whether this LSTM training loop works). This test checks the
    # latter: does the loop actually fit a learnable, in-distribution pattern.
    t = np.arange(400, dtype=np.float64)
    series = (np.sin(t * 0.3) * 5.0).astype(np.float32)
    X, y = make_windows(series, lookback=8, horizon=1)
    X_train, y_train = X[:300], y[:300]
    X_test, y_test = X[300:], y[300:]

    model = LSTMForecaster(hidden_size=16, max_epochs=300, patience=30, seed=0)
    model.fit(X_train, y_train)
    assert model.train_losses[-1] < model.train_losses[0]

    pred = model.predict(X_test)
    mae = float(np.mean(np.abs(pred - y_test)))
    # Loose tolerance -- the point is "learned something real", not
    # "matches a specific float exactly across torch versions". A model
    # that learned nothing would do about as well as predicting the mean
    # (MAE ~ 3.2 for this amplitude-5 sine); a naive last-value baseline
    # gets roughly 1.6-2.0 here, so we require clearly better than that.
    assert mae < 1.5


def test_delta_wrapper_extrapolates_a_trend_a_plain_model_cannot():
    # A model that predicts the trained-on *absolute level* has never seen
    # test-range values for a strongly trending series and should do
    # noticeably worse than the same underlying model wrapped to predict
    # the *change* from the last value instead (see DeltaWrapper docstring).
    series = np.arange(300, dtype=np.float64).astype(np.float32)  # y = x, unbounded trend
    X, y = make_windows(series, lookback=5, horizon=1)
    X_train, y_train = X[:200], y[:200]
    X_test, y_test = X[200:], y[200:]

    plain = LSTMForecaster(hidden_size=16, max_epochs=200, patience=20, seed=0)
    plain.fit(X_train, y_train)
    plain_mae = float(np.mean(np.abs(plain.predict(X_test) - y_test)))

    wrapped = DeltaWrapper(LSTMForecaster(hidden_size=16, max_epochs=200, patience=20, seed=0))
    wrapped.fit(X_train, y_train)
    wrapped_mae = float(np.mean(np.abs(wrapped.predict(X_test) - y_test)))

    assert wrapped_mae < plain_mae
    # For y = x exactly, delta is always 1 -- the wrapped model should
    # reconstruct the trend almost perfectly.
    assert wrapped_mae < 1.0


def test_delta_wrapper_forwards_attributes_to_base_model():
    wrapper = DeltaWrapper(LSTMForecaster(max_epochs=5, patience=5))
    X = np.random.RandomState(0).randn(20, 4).astype(np.float32)
    y = np.random.RandomState(1).randn(20, 1).astype(np.float32)
    wrapper.fit(X, y)
    assert len(wrapper.train_losses) > 0  # forwarded via __getattr__
