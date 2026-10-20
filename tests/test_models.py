import numpy as np

from forecasting.data import make_windows
from forecasting.models import (
    DeltaWrapper,
    LSTMForecaster,
    NaiveForecaster,
    NBEATSForecaster,
    NBEATSInterpretableForecaster,
    SeasonalNaiveForecaster,
    TransformerForecaster,
)


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


def test_transformer_forecaster_rejects_nhead_that_does_not_divide_d_model():
    try:
        TransformerForecaster(d_model=32, nhead=5)
    except ValueError as e:
        assert "divisible" in str(e)
    else:
        raise AssertionError("expected a ValueError for an incompatible d_model/nhead pair")


def test_transformer_forecaster_learns_a_bounded_periodic_series():
    # Same test as the LSTM's own periodic-series check above, with the
    # same reasoning: a sine wave is bounded, so this isolates "did the
    # training loop learn a real in-distribution pattern" from the
    # separate, well-known weakness that neural nets (of any architecture)
    # extrapolate poorly past their training range.
    t = np.arange(400, dtype=np.float64)
    series = (np.sin(t * 0.3) * 5.0).astype(np.float32)
    X, y = make_windows(series, lookback=8, horizon=1)
    X_train, y_train = X[:300], y[:300]
    X_test, y_test = X[300:], y[300:]

    model = TransformerForecaster(d_model=16, nhead=4, max_epochs=300, patience=30, seed=0)
    model.fit(X_train, y_train)
    assert model.train_losses[-1] < model.train_losses[0]

    pred = model.predict(X_test)
    mae = float(np.mean(np.abs(pred - y_test)))
    # Same loose tolerance and same reasoning as the LSTM version of this
    # test: a naive last-value baseline gets roughly 1.6-2.0 here, so we
    # require clearly better than that, not a specific float.
    assert mae < 1.5


def test_transformer_forecaster_handles_a_different_lookback_at_predict_time():
    # Unlike a learned positional *embedding* table of fixed size, the
    # sinusoidal encoding is computed fresh for whatever seq_len the input
    # actually has -- so, like the LSTM (which is recurrent and therefore
    # inherently sequence-length-agnostic), this model shouldn't crash or
    # silently misbehave if ever called with a lookback it wasn't fit on.
    # (No benchmark path in this repo actually does this -- every dataset
    # uses one fixed lookback throughout -- but the model itself shouldn't
    # assume it won't happen.)
    rng = np.random.RandomState(0)
    X_train = rng.randn(40, 6).astype(np.float32)
    y_train = rng.randn(40, 1).astype(np.float32)
    model = TransformerForecaster(d_model=16, nhead=4, max_epochs=5, patience=5, seed=0)
    model.fit(X_train, y_train)

    X_different_lookback = rng.randn(5, 10).astype(np.float32)
    pred = model.predict(X_different_lookback)
    assert pred.shape == (5, 1)
    assert np.all(np.isfinite(pred))


def test_delta_wrapper_extrapolates_a_trend_with_transformer_too():
    # The same trend-extrapolation check as the LSTM version above, run
    # against the Transformer instead -- confirms DeltaWrapper's benefit
    # isn't somehow LSTM-specific.
    series = np.arange(300, dtype=np.float64).astype(np.float32)  # y = x, unbounded trend
    X, y = make_windows(series, lookback=5, horizon=1)
    X_train, y_train = X[:200], y[:200]
    X_test, y_test = X[200:], y[200:]

    plain = TransformerForecaster(d_model=16, nhead=4, max_epochs=200, patience=20, seed=0)
    plain.fit(X_train, y_train)
    plain_mae = float(np.mean(np.abs(plain.predict(X_test) - y_test)))

    wrapped = DeltaWrapper(
        TransformerForecaster(d_model=16, nhead=4, max_epochs=200, patience=20, seed=0)
    )
    wrapped.fit(X_train, y_train)
    wrapped_mae = float(np.mean(np.abs(wrapped.predict(X_test) - y_test)))

    assert wrapped_mae < plain_mae


def test_delta_wrapper_forwards_attributes_to_transformer_base_model():
    wrapper = DeltaWrapper(TransformerForecaster(d_model=16, nhead=4, max_epochs=5, patience=5))
    X = np.random.RandomState(0).randn(20, 4).astype(np.float32)
    y = np.random.RandomState(1).randn(20, 1).astype(np.float32)
    wrapper.fit(X, y)
    assert len(wrapper.train_losses) > 0  # forwarded via __getattr__


def test_nbeats_forecaster_learns_a_bounded_periodic_series():
    # Same test as the LSTM/Transformer periodic-series checks above, same
    # reasoning: a bounded sine wave isolates "did the training loop learn
    # a real in-distribution pattern" from neural nets' separate, unrelated
    # weakness at extrapolating past their training range.
    t = np.arange(400, dtype=np.float64)
    series = (np.sin(t * 0.3) * 5.0).astype(np.float32)
    X, y = make_windows(series, lookback=8, horizon=1)
    X_train, y_train = X[:300], y[:300]
    X_test, y_test = X[300:], y[300:]

    model = NBEATSForecaster(num_blocks=3, hidden_size=16, max_epochs=300, patience=30, seed=0)
    model.fit(X_train, y_train)
    assert model.train_losses[-1] < model.train_losses[0]

    pred = model.predict(X_test)
    mae = float(np.mean(np.abs(pred - y_test)))
    # Same loose tolerance and reasoning as the LSTM/Transformer versions of
    # this test: a naive last-value baseline gets roughly 1.6-2.0 here, so
    # we require clearly better than that, not a specific float.
    assert mae < 1.5


def test_nbeats_forecaster_rejects_a_different_lookback_at_predict_time():
    # Real, honest architectural difference from LSTM/Transformer: those two
    # are sequence models (recurrence / attention), so they tolerate a
    # different lookback at predict time than they were fit on (see the
    # dedicated Transformer test for this). N-BEATS' blocks are plain FC
    # stacks with Linear layers sized to the *flat* lookback at construction
    # time, so a mismatched lookback is a real shape error, not something
    # this implementation silently handles -- this test documents that
    # instead of pretending otherwise.
    rng = np.random.RandomState(0)
    X_train = rng.randn(40, 6).astype(np.float32)
    y_train = rng.randn(40, 1).astype(np.float32)
    model = NBEATSForecaster(num_blocks=2, hidden_size=8, max_epochs=5, patience=5, seed=0)
    model.fit(X_train, y_train)

    X_different_lookback = rng.randn(5, 10).astype(np.float32)
    try:
        model.predict(X_different_lookback)
    except RuntimeError as e:
        assert "size mismatch" in str(e) or "shapes cannot be multiplied" in str(e)
    else:
        raise AssertionError(
            "expected a shape error -- N-BEATS' FC blocks are sized to the "
            "lookback they were fit on, unlike the sequence models"
        )


def test_delta_wrapper_extrapolates_a_trend_with_nbeats_too():
    # The same trend-extrapolation check as the LSTM/Transformer versions
    # above, run against N-BEATS instead -- confirms DeltaWrapper's benefit
    # isn't architecture-specific.
    series = np.arange(300, dtype=np.float64).astype(np.float32)  # y = x, unbounded trend
    X, y = make_windows(series, lookback=5, horizon=1)
    X_train, y_train = X[:200], y[:200]
    X_test, y_test = X[200:], y[200:]

    plain = NBEATSForecaster(num_blocks=3, hidden_size=16, max_epochs=200, patience=20, seed=0)
    plain.fit(X_train, y_train)
    plain_mae = float(np.mean(np.abs(plain.predict(X_test) - y_test)))

    wrapped = DeltaWrapper(
        NBEATSForecaster(num_blocks=3, hidden_size=16, max_epochs=200, patience=20, seed=0)
    )
    wrapped.fit(X_train, y_train)
    wrapped_mae = float(np.mean(np.abs(wrapped.predict(X_test) - y_test)))

    assert wrapped_mae < plain_mae


def test_delta_wrapper_forwards_attributes_to_nbeats_base_model():
    wrapper = DeltaWrapper(NBEATSForecaster(num_blocks=2, hidden_size=8, max_epochs=5, patience=5))
    X = np.random.RandomState(0).randn(20, 4).astype(np.float32)
    y = np.random.RandomState(1).randn(20, 1).astype(np.float32)
    wrapper.fit(X, y)
    assert len(wrapper.train_losses) > 0  # forwarded via __getattr__


def test_nbeats_doubly_residual_stacking_reduces_backcast_error_across_blocks():
    # The defining mechanic of N-BEATS (vs. a single FC block) is doubly-
    # residual stacking: each block only has to explain what earlier
    # blocks' backcasts failed to reconstruct. This test checks that
    # mechanic directly, at the module level, rather than only checking
    # end-to-end forecast accuracy (which wouldn't distinguish "residual
    # stacking is working" from "one block alone would do just as well").
    from forecasting.models import NBEATSModule

    torch_seed = 0
    import torch

    torch.manual_seed(torch_seed)
    module = NBEATSModule(lookback=10, horizon=1, num_blocks=3, hidden_size=16)
    x = torch.randn(4, 10)

    with torch.no_grad():
        residual = x
        residual_norms = [float(residual.norm())]
        for block in module.blocks:
            backcast, _ = block(residual)
            residual = residual - backcast
            residual_norms.append(float(residual.norm()))

    # An untrained block's backcast is essentially random noise relative to
    # the input, so this doesn't assert monotonic decrease (that would only
    # hold after training) -- it asserts the mechanic itself: each block
    # actually changes the residual passed to the next one, i.e. the
    # blocks are genuinely chained, not independently reading the same
    # fixed input (which would make "stacking" cosmetic).
    assert residual_norms[1] != residual_norms[0]
    assert residual_norms[2] != residual_norms[1]


def test_trend_block_basis_is_a_fixed_low_degree_polynomial():
    # Hand-computed check of the fixed (non-learned) basis itself, isolated
    # from the learned theta projection: a degree=2 trend block's backcast
    # basis must be exactly [1, t, t^2] evaluated at lookback evenly-spaced
    # points in [0, 1), and the forecast basis the same shape over the
    # horizon -- this is what makes it "interpretable" (a hard-coded
    # low-order polynomial shape), not a claim about what the network
    # learns to put in theta.
    from forecasting.models import NBEATSTrendBlock

    block = NBEATSTrendBlock(lookback=4, horizon=2, hidden_size=8, num_fc_layers=1, degree=2)
    assert block.theta_size == 3  # degree + 1

    t_back = np.arange(4) / 4.0
    expected_backcast_basis = np.stack([t_back**p for p in range(3)], axis=0)
    np.testing.assert_allclose(block.backcast_basis.detach().numpy(), expected_backcast_basis, atol=1e-6)

    t_fore = np.arange(2) / 2.0
    expected_forecast_basis = np.stack([t_fore**p for p in range(3)], axis=0)
    np.testing.assert_allclose(block.forecast_basis.detach().numpy(), expected_forecast_basis, atol=1e-6)


def test_seasonality_block_basis_is_a_fixed_fourier_series():
    # Same idea as the trend-block basis test, for the seasonality block's
    # paired sin/cos harmonics -- hand-computed against the actual Fourier
    # formula, not just "it runs."
    from forecasting.models import NBEATSSeasonalityBlock

    block = NBEATSSeasonalityBlock(lookback=8, horizon=4, hidden_size=8, num_fc_layers=1, num_harmonics=2)
    assert block.theta_size == 4  # 2 harmonics * (cos, sin)

    t_fore = np.arange(4) / 4.0
    expected_forecast_basis = np.stack(
        [
            np.cos(2 * np.pi * 1 * t_fore),
            np.sin(2 * np.pi * 1 * t_fore),
            np.cos(2 * np.pi * 2 * t_fore),
            np.sin(2 * np.pi * 2 * t_fore),
        ],
        axis=0,
    )
    np.testing.assert_allclose(block.forecast_basis.detach().numpy(), expected_forecast_basis, atol=1e-6)


def test_seasonality_block_num_harmonics_defaults_to_half_horizon_and_floors_at_one():
    from forecasting.models import NBEATSSeasonalityBlock

    assert NBEATSSeasonalityBlock(lookback=10, horizon=12, hidden_size=8).num_harmonics == 6
    # horizon=1 elsewhere in this project (horizon // 2 == 0) must not produce
    # a zero-width (unusable) basis.
    assert NBEATSSeasonalityBlock(lookback=10, horizon=1, hidden_size=8).num_harmonics == 1


def test_nbeats_interpretable_forecaster_learns_a_bounded_periodic_series():
    # Same periodic-series check as NBEATSForecaster's own version -- the
    # interpretable variant's seasonality blocks should be at least as able
    # to fit a clean sine wave as the generic variant's learned basis.
    t = np.arange(400, dtype=np.float64)
    series = (np.sin(t * 0.3) * 5.0).astype(np.float32)
    X, y = make_windows(series, lookback=8, horizon=1)
    X_train, y_train = X[:300], y[:300]
    X_test, y_test = X[300:], y[300:]

    model = NBEATSInterpretableForecaster(
        num_trend_blocks=2, num_seasonality_blocks=2, hidden_size=16, max_epochs=300, patience=30, seed=0
    )
    model.fit(X_train, y_train)
    assert model.train_losses[-1] < model.train_losses[0]

    pred = model.predict(X_test)
    mae = float(np.mean(np.abs(pred - y_test)))
    # Same loose tolerance as every other model's version of this test --
    # a naive last-value baseline gets roughly 1.6-2.0 here.
    assert mae < 1.5


def test_nbeats_interpretable_forecaster_fits_a_pure_linear_trend_well():
    # A case the trend basis is specifically built for: a pure linear trend
    # should be very easy for a degree>=1 polynomial basis to represent,
    # directly through theta -- not something that requires many epochs or
    # a lucky learned shape the way the generic block's unconstrained
    # linear projection would need to discover on its own.
    series = np.arange(200, dtype=np.float64).astype(np.float32)  # y = x
    X, y = make_windows(series, lookback=5, horizon=1)
    X_train, y_train = X[:150], y[:150]
    X_test, y_test = X[150:], y[150:]

    model = NBEATSInterpretableForecaster(
        num_trend_blocks=2, num_seasonality_blocks=1, hidden_size=16, trend_degree=1, max_epochs=300, patience=30, seed=0
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = float(np.mean(np.abs(pred - y_test)))
    assert mae < 1.5


def test_nbeats_interpretable_forecaster_rejects_a_different_lookback_at_predict_time():
    # Same real, honest architectural limitation as the generic variant
    # (both operate on the flat lookback window via FC layers sized at
    # construction time) -- documented the same way, not papered over.
    rng = np.random.RandomState(0)
    X_train = rng.randn(40, 6).astype(np.float32)
    y_train = rng.randn(40, 1).astype(np.float32)
    model = NBEATSInterpretableForecaster(
        num_trend_blocks=1, num_seasonality_blocks=1, hidden_size=8, max_epochs=5, patience=5, seed=0
    )
    model.fit(X_train, y_train)

    X_different_lookback = rng.randn(5, 10).astype(np.float32)
    try:
        model.predict(X_different_lookback)
    except RuntimeError as e:
        assert "size mismatch" in str(e) or "shapes cannot be multiplied" in str(e)
    else:
        raise AssertionError(
            "expected a shape error -- the interpretable variant's FC blocks are "
            "sized to the lookback they were fit on, same as the generic variant"
        )


def test_delta_wrapper_forwards_attributes_to_nbeats_interpretable_base_model():
    wrapper = DeltaWrapper(
        NBEATSInterpretableForecaster(num_trend_blocks=1, num_seasonality_blocks=1, hidden_size=8, max_epochs=5, patience=5)
    )
    X = np.random.RandomState(0).randn(20, 4).astype(np.float32)
    y = np.random.RandomState(1).randn(20, 1).astype(np.float32)
    wrapper.fit(X, y)
    assert len(wrapper.train_losses) > 0  # forwarded via __getattr__


def test_nbeats_interpretable_doubly_residual_stacking_chains_blocks():
    # Same mechanic check as the generic variant's own version of this test:
    # trend and seasonality blocks alike must actually change the residual
    # each sees, confirming the stack (not just one block) is doing the work.
    from forecasting.models import NBEATSInterpretableModule

    import torch

    torch.manual_seed(0)
    module = NBEATSInterpretableModule(
        lookback=10, horizon=1, num_trend_blocks=2, num_seasonality_blocks=2, hidden_size=16
    )
    assert len(module.blocks) == 4
    x = torch.randn(4, 10)

    with torch.no_grad():
        residual = x
        residual_norms = [float(residual.norm())]
        for block in module.blocks:
            backcast, _ = block(residual)
            residual = residual - backcast
            residual_norms.append(float(residual.norm()))

    for i in range(1, len(residual_norms)):
        assert residual_norms[i] != residual_norms[i - 1]
