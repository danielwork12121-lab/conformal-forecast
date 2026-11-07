# conformal-forecast

Time-series forecasting with calibrated prediction intervals, built from scratch in PyTorch + NumPy, with empirical coverage checked rather than assumed.

| Airline passengers, nominal 90% intervals (30-point test window) | Empirical coverage |
|---|---|
| SeasonalNaive baseline (MAE 40.8), static split conformal | 86.7% |
| LSTM (MAE 122.6 -- loses badly to the baseline), static split conformal | 63.3% |
| Same LSTM + EnbPI (20 bootstrap members, `window=10`) | 86.7% |

```bash
pip install -e ".[dev,api]"            # or: pip install -r requirements.txt
python -m pytest                        # 298 tests
python -m forecasting.cli benchmark     # all three datasets, static split conformal
```

Caveat: every coverage figure above comes from one small test window (30 points on airline, so one miss moves coverage by 3.3pp). Treat them as evidence of a mechanism, not as strong statistics.

## What it does

If a forecaster reports a "90% interval", does the truth land inside it about 90% of the time? That property is *coverage*, which [conformal prediction](https://en.wikipedia.org/wiki/Conformal_prediction) is designed to guarantee. This repo wraps several point forecasters in conformal calibration, then measures coverage on real and synthetic data, including the cases where the guarantee visibly breaks.

The textbook guarantee for split conformal assumes calibration and test residuals are exchangeable. For non-stationary series, or models whose error drifts between the calibration and test windows, that holds only approximately. The benchmarks below include cases where it fails, and the later sections test methods that try to repair it.

## Components

- **Point models** (`forecasting/models.py`, shared `fit`/`predict` interface): `Naive`, `SeasonalNaive`, a from-scratch `LSTM`, `Transformer` encoder, generic `N-BEATS` (Oreshkin et al. 2020), and an "interpretable" N-BEATS with fixed trend (polynomial) and seasonality (Fourier) bases. Neural models share one training recipe (Adam, MSE, early stopping), so comparisons isolate architecture.
- **`DeltaWrapper`**: predicts the change from the last observation instead of the level, the neural analogue of differencing. A small LSTM cannot extrapolate a trend whose level it never saw in training.
- **Split conformal** (`conformal.py`): finite-sample-corrected quantile (Lei et al. 2018, Vovk et al. 2005).
- **Adaptive Conformal Inference** (`adaptive.py`, Gibbs & Candes 2021): fixed-pool and sliding/growing-pool variants, plus a variant that periodically retrains the model.
- **EnbPI** (`enbpi.py`, Xu & Xie 2021): bootstrap-ensemble conformal with a sliding out-of-bag residual pool. Two disclosed simplifications versus the paper: symmetric residual quantiles, and an unbounded pool by default.
- **Datasets** bundled in `data/`: classic airline passengers (144 points), Melbourne daily minimum temperature (3,650 points), and a synthetic trend + seasonality + Gaussian-noise generator used as a calibration sanity check.

## Benchmark results (static split conformal, nominal 90%)

Produced by `python -m forecasting.cli benchmark`. **Mean interval widths in this README are in normalized (z-score) units**: they are comparable between rows of the same table but not interpretable in the series' original units. MAE is denormalized. Coverage and coverage gap are scale-invariant.

**airline** (strong trend + yearly seasonality, ~86 training windows, 28 calibration windows):

| Model | MAE | RMSE | MAPE (%) | Coverage | Width |
|---|---|---|---|---|---|
| Naive | 45.100 | 52.612 | 10.30 | 80.0% | 2.270 |
| SeasonalNaive | 40.800 | 45.555 | 9.02 | 86.7% | 1.999 |
| LSTM (delta) | 122.645 | 134.268 | 27.76 | 63.3% | 4.873 |
| Transformer (delta) | 36.276 | 44.416 | 8.22 | 66.7% | 1.509 |
| N-BEATS (delta) | 45.260 | 49.689 | 10.64 | 56.7% | 1.484 |
| N-BEATS interpretable (delta) | 36.092 | 43.731 | 8.58 | 66.7% | 1.372 |

**temperature** and **synthetic** (MAE / coverage):

| Model | temperature | synthetic |
|---|---|---|
| Naive | 1.953 / 93.6% | 1.910 / 87.5% |
| SeasonalNaive | 1.953 / 93.6% | 1.515 / 90.0% |
| LSTM (delta) | 1.730 / 93.0% | 1.978 / 63.7% |
| Transformer (delta) | 1.721 / 92.9% | 1.556 / 81.2% |
| N-BEATS (delta) | 1.758 / 92.3% | 2.240 / 58.1% |
| N-BEATS interpretable (delta) | 1.789 / 91.9% | 1.099 / 89.4% |

Reading these honestly:

- **The LSTM loses to both baselines on airline** (MAE 122.6 vs 40.8 for SeasonalNaive), and its 90% interval covers only 63.3% of test points. With ~86 training windows a small neural model cannot reliably beat a strong seasonal baseline, and its error grows between the calibration and test windows, which is exactly the exchangeability failure described above.
- **The Transformer is more accurate but not better calibrated** on airline: best-tier MAE (36.3, beating SeasonalNaive) with 66.7% coverage. On synthetic it closes much of the LSTM's gap (63.7% to 81.2%). On temperature, which has ~1,950 training windows, the neural models roughly tie.
- **Generic N-BEATS is the weakest architecture** here: worst coverage on airline (56.7%) and synthetic (58.1%), worse than the LSTM it was meant to improve on. Its interpretable sibling is the best model on synthetic (a dataset whose generating process is trend + seasonality, matching its built-in bias) and strong on airline, but is **not** a win on temperature (worst MAE of the four neural models).
- On temperature, `SeasonalNaive` equals `Naive` because the 30-day lookback is shorter than the 365-day period, so it falls back to the last-value rule. A 365-day lookback was tried and persistence still won (MAE 1.95 vs 2.97).
- Takeaway across all three: under-coverage is not LSTM-specific, not purely a point-accuracy problem, and not fixed by swapping architecture. Calibration-side adaptation (below) is the lever that worked.

## Fixing under-coverage: adaptive and ensemble calibration

All comparisons use the same fitted LSTM and the same test window, so only the calibration method changes.

**Fixed-pool ACI** re-slices a frozen calibration pool, so it can never propose an interval wider than the pool's largest residual:

| Dataset | Static coverage | ACI coverage | Static / ACI width |
|---|---|---|---|
| airline | 63.3% | 60.0% | 4.873 / 4.880 |
| temperature | 93.0% | 90.0% | 1.929 / 1.774 |
| synthetic | 63.7% | 81.9% | 0.471 / 0.663 |

It helps on synthetic and tightens temperature by 8%, but on airline (28 calibration windows) there is almost no headroom and it barely moves, within the noise of a 30-point test set.

**Sliding/growing-pool ACI** folds each scored step's residual into the pool (strictly after that step's interval is computed, so nothing leaks), removing the ceiling:

| Dataset | Static | Fixed-pool ACI | Sliding-pool ACI | Sliding width |
|---|---|---|---|---|
| airline | 63.3% | 60.0% | 80.0% | 5.918 |
| temperature | 93.0% | 90.0% | 90.0% | 1.785 |
| synthetic | 63.7% | 81.9% | 86.9% | 0.803 |

The cost is real: intervals are wider on every dataset. Coverage recovered by widening is a trade, not a free lunch.

**Window tuning.** Bounding the sliding pool helps. On airline, `window=7-15` gives 86.7% (gap -3.3pp) against 80.0% unbounded; averaged over 6 seeds, mean |gap| is 6.1pp for `window=10` vs 10.6pp unbounded, and `window=10` was never worse on any seed. `window=5` is too small. On synthetic, windows of 30-50 land on 90.0%. There is no single best default; the optimum is dataset-dependent and rests on two datasets, one `gamma`, one `alpha`. `window-sweep` finds it for a new series. This sweep also exposed a bug in the first sliding-window implementation: the pool was seeded with the whole calibration set and then did one append and one evict per step, so a pool seeded above `window` never shrank, and `window=10/15/20` silently produced identical results on airline. Fixed by truncating the seed pool before the first step, with regression tests.

**EnbPI** answers the same problem differently (a bootstrap ensemble trained once, plus a sliding out-of-bag residual pool, with no `alpha_t` adaptation). On airline with the same LSTM:

| Method | Coverage | Gap | Width |
|---|---|---|---|
| Static split conformal | 63.3% | -26.7pp | 4.873 |
| Fixed-pool ACI | 60.0% | -30.0pp | 4.880 |
| EnbPI (`n_bootstrap=20`, `window=10`) | **86.7%** | **-3.3pp** | 3.492 |

This beats both ACI variants on this case. The result depends strongly on the window: the gap degrades smoothly as the window grows and an unbounded window matches static conformal almost exactly. The `window=10` choice came from a sweep over `{10, 15, 20, 30, 40, 50, 60, 75, 90, 100, unbounded}` (not a single lucky setting); seeds 1-3 all reach exactly 90.0% at `window=10`. It is still one dataset and one 30-point test window.

**Retraining the model online** (a fresh LSTM refit about 8 times across the test window, sliding pool unbounded):

| Dataset | Frozen model + ACI | Retrained + ACI | Width frozen / retrained |
|---|---|---|---|
| airline | 80.0% | 93.3% | 5.918 / 3.353 |
| synthetic | 86.9% | 89.4% | 0.803 / 0.593 |
| temperature | 90.0% | 90.0% | 1.785 / 1.803 |

Retraining helps most where data is scarcest and never measurably hurt. Retraining is coarse-grained by design (refitting every step is too slow). A finer cadence does *not* reliably help further: airline plateaus at 93.3%, and on synthetic the coarsest cadence tested was the best (90.0%) while finer cadences gave up a little coverage.

## Selecting hyperparameters without the test set

Several selection methods choose the pool `window` (or N-BEATS basis settings) from calibration data only, using the test set purely as a post-hoc check. The results are mixed, and that is the finding:

- **Single holdout (`auto-window`)** works when the calibration set is large (temperature, synthetic). On airline every candidate ties, for a structural reason: a holdout drawn from the calibration period cannot see the calibration-to-test shift.
- **Rolling-origin CV (`cv-window-select`)** is not a strict upgrade. Mean |test gap| over 6 seeds, CV vs single-holdout vs oracle: airline 8.3 / 10.0 / 6.1pp; synthetic 1.1 / 0.9 / 0.1pp; temperature 0.1 / 0.1 / 0.0pp. It helps on small calibration sets and is a coin flip on larger ones. A sliding fold scheme is situational, `auto_folds` improves the average but lost in 3 of 60 seed/dataset measurements, and the two options do not compound.
- **Not seed-robust.** Re-running the CV pick with 5 training seeds on airline selected 4 different windows, with real test gaps from +0.0pp to -16.7pp. Synthetic spans +0.6pp to -1.9pp. Temperature is practically robust (+0.1pp every seed) because its calibration set is large. Any single-seed selection result should be read as one sample from a noisy distribution.
- **EnbPI window CV** gave a clean win on synthetic (+1.2pp gap) and temperature (+1.0pp), and an honest **loss on airline**: it picked `window=20` (-6.7pp) where the eyeballed sweep value `window=10` gave -3.3pp.
- **N-BEATS basis hyperparameters** (`trend_degree`, `num_harmonics`, block counts) are dataset-dependent. On airline the shipped `trend_degree=2` is among the worse sweep points (`degree=5` improves the gap from -23.3pp to -6.7pp), and the joint grid finds an interaction neither single sweep can reach (`trend_degree=1, num_harmonics=3`: +3.3pp, MAE 17.298). On synthetic the shipped defaults are near-optimal. Block-depth sweeps show no clean pattern on airline. CV selection of the basis is **not** a reliable substitute for reading the test set: it picks a clearly suboptimal cell on airline, the right degree but the wrong harmonics on synthetic (-5.0pp vs -0.6pp for the oracle), and lands close on temperature, where nearly any choice works.

Recurring lesson: calibration-set size (28 / 160 / 730 windows) dominates how well any of these automated methods behave.

## Serving it as an API

`python -m forecasting.cli serve --port 8000` starts a FastAPI app (Swagger UI at `/docs`) around the same split + conformal pipeline the CLI uses.

- `GET /health`, `GET /datasets`
- `POST /forecast` with exactly one of `dataset`, `series` (at least 30 points), or `series_id`; returns per-point actuals, forecasts, `[lower, upper]` intervals, and a summary (MAE, empirical/nominal coverage, gap, mean interval width, `q_hat`). Invalid input returns 422, an unknown `series_id` 404.
- `POST/GET/PATCH/DELETE /series` registers a series once and returns an id; entries live in a bounded LRU store with idle-TTL eviction.
- `POST /series/{id}/score` scores newly observed points against a cached fit without refitting. It does not retrain the model.
- `GET /cache/stats`, `DELETE /cache` manage a bounded LRU cache of fitted forecasters.

`summary.mean_interval_width` and `summary.q_hat` are in normalized (z-score) units, not the series' original units; `summary.mae` is denormalized.

## Architecture

```
forecasting/
  data.py        loading, chronological split, sliding windows, z-score normalizer
  models.py      Naive, SeasonalNaive, LSTM, Transformer, N-BEATS (generic + interpretable), DeltaWrapper
  metrics.py     MAE / RMSE / MAPE, empirical coverage, mean interval width
  conformal.py   SplitConformalForecaster, evaluate_coverage
  adaptive.py    fixed-pool ACI, sliding/growing-pool ACI, retraining variant
  enbpi.py       EnbPIForecaster
  experiment.py  end-to-end experiments, window/basis sweeps, CV selection
  api.py         FastAPI serving layer
  cli.py         benchmark, adaptive, sliding-window, enbpi, window-sweep, cv-window-select, ...
tests/           298 tests, including statistical coverage-tracking checks
data/            bundled datasets (no network needed)
results/         generated plots
```

`prepare_split_from_series` is exposed as a public wrapper over the shared split/window/normalize pipeline so downstream projects can reuse the exact pipeline this library validates its coverage against. [kalshi-commodity-forecaster](https://github.com/danielwork12121-lab/kalshi-commodity-forecaster) depends on this repo as a real library.

## Usage

```bash
python -m forecasting.cli benchmark --dataset airline --plot results/airline_forecast.png
python -m forecasting.cli adaptive --dataset airline        # static vs. fixed-pool ACI
python -m forecasting.cli sliding-window --dataset airline  # + sliding-pool ACI
python -m forecasting.cli enbpi --dataset airline --n-bootstrap 20 --window 10
python -m forecasting.cli window-sweep --dataset airline --windows 5,7,10,15,20,30,50,unbounded
python -m forecasting.cli cv-window-select --dataset airline --candidates 5,7,10,15,20,30,50,unbounded
python -m forecasting.cli online-retraining --dataset airline
```

## Limitations

- Coverage figures come from small test windows (airline: 30 points, 28 calibration windows). One miss is 3.3pp, and several differences reported above are within that noise.
- Conformal guarantees assume exchangeability, which fails for the drifting-error cases shown here. The adaptive methods repair coverage empirically, usually by widening intervals.
- Hyperparameters (`window`, retrain cadence, N-BEATS basis) are dataset-dependent and measured on three datasets, mostly at one seed. Single-seed selection results are noisy.
- Neural models are small, from-scratch, and lightly tuned; the benchmark says nothing about what tuned or large models would do.
- Interval widths are reported in normalized units.

## Changelog

- 0.11-0.19: FastAPI service, fit caching, user-supplied series, series registry with TTL eviction.
- 0.16: installable as a dependency (`[build-system]`, `prepare_split_from_series`).
- 0.20-0.24: Transformer, generic and interpretable N-BEATS; EnbPI.
- 0.25-0.30: N-BEATS basis and depth sweeps; CV selection of basis hyperparameters.
- 0.31: seed-robustness check of CV window selection.
- 0.32-0.34: online retraining, `/series/{id}/score`, retraining-cadence sweep.

## License

MIT, see [LICENSE](LICENSE).
