# Price forecast for the stochastic CRRA implementation

The validation-selected model is a four-layer, width-512 normalized GLU with
120 known one-second return lags and 58 latest production features. Its price
law has an explicit exact-zero atom and a 256-knot triangular continuous
nonzero-return density. It was retrained because the previous checkpoints had
been deleted. Selection uses distribution likelihood, not backtest returns.

## Model choice and data corrections

The production-basis density experiment from August 17 had the strongest
documented one-step return NLL among the proper price-density records reviewed.
The September 10 five-branch structured experiment predicts point feature states
by MSE; its correlation does not establish a superior probability distribution.
Embedding-space density losses likewise cannot be ranked as price likelihoods.

This run uses that GLU density recipe with two corrections. The old targets
excluded future exact-zero returns, so they did not describe the all-tick law.
The old shared knot fit also used full-history statistics extending beyond the
training interval. Here every chronological tick is retained and every
data-dependent density coordinate, prior mass, zero probability and feature
normalization is fitted only on training observations. The asinh-logistic
transform has fixed alpha 4; its scale comes from training returns. Quantile and
regular coordinates together give 256 distinct knots.

The preserved production59 timeline allows exact recovery of all ticks because
`timestamp - 1000 * originIndex` is constant throughout each split. The input
at origin `i` uses return channel 0 at `i-119 ... i` and channels 1 through 58 at
`i`. Its target is return channel 0 at `i+1`. Export additionally checks those
targets against the historical close-price ratios.

| Split | Origin interval, UTC | Seconds |
| --- | --- | ---: |
| Train | July 18 04:01:59 through July 29 17:28:40 | 998,802 |
| Validation | August 3 00:00:01 through August 4 16:23:57 | 145,437 |
| Test | August 10 00:00:01 through August 11 18:08:52 | 151,732 |

Training used seed 1337, batch 4096, learning rate 0.0001, the existing hybrid
Muon/AdamW optimizer, no dropout/weight decay, and float32 with high matmul
precision. The declared cap was 64 epochs with patience 12 on validation NLL.
The best checkpoint was epoch 1; training stopped at epoch 13, in 78.7 seconds
on the available RTX 3070. Test targets were loaded after checkpoint selection.

## Validation comparison

A separate unconditional baseline fits the same train-only triangular basis
with 100 EM iterations and half-count smoothing. Its zero atom is the training
zero frequency. This comparison selects exclusively by validation likelihood;
the selection file is persisted before computing its test metrics.

| Model | Validation mixed NLL | Test mixed NLL | Validation MSE skill | Test MSE skill |
| --- | ---: | ---: | ---: | ---: |
| Conditional GLU, epoch 1 | **-5.041420** | -5.921930 | 2.6732% | 3.5713% |
| Unconditional fitted marginal | -4.782361 | -5.757708 | -0.0001% | 0.0005% |

The initial empirical histogram model also participates as epoch 0 in neural
checkpoint selection; its validation NLL was -4.798307. The conditional GLU
beats both unconditional candidates on the declared criterion. This establishes
the best model among these corrected candidates, not a universal best model.
These mixed-law scores include a discrete zero mass and cannot be compared
numerically with historical nonzero-only continuous NLL scores.

## Scenario and inference contract

`ml/stochastic_price_forecast.py` exposes the fitted law, mixed log likelihood,
causal lag update and finite scenarios. For each node, three equal-probability
bins of the nonzero law are represented by their conditional mean, with a
separate zero branch. Conditional means use 64-point Gauss-Legendre quadrature
per triangular-knot interval. They retain the forecast first moment to numerical
precision; midpoint quantiles would omit substantial tail contributions in this
sharply concentrated distribution. The maximum first-return mean discrepancy in
the full test export was 7.74e-12.

At the second hypothetical tick the 120 return lags are updated with the branch
return, while the 58 other observed features remain fixed. Those features include
derived price features as well as external observations. Freezing them is an
explicit approximation in the forecast model; actual future observations are
never injected into a hypothetical branch.

The exporter emits contiguous JSONL rows with `timestamp`, `observedPrice`,
`realizedNextTimestamp`, `realizedNextPrice`, and a two-step `tree.children`
structure. Each child has `probability`, `price`, and, at the first level,
`children`. Realized prices are separate evaluation fields and are never model
inputs. Predicted prices are unrounded; exchange price-grid rounding belongs to
the execution adapter.

The full test forecast has 151,732 rows and took 32.45 seconds to export. A
separate first-hour file supplies a fixed 3,600-second smoke/backtest interval.

For a live bot provider, `ml/serve_stochastic_price_forecast.py` loads the selected
checkpoint once and accepts JSONL requests `{id, observedPrice, features}` with
exactly 178 finite current features and a positive observed price. It emits
`{id, tree}` or an explicit `{id, error: {code, message}}` response. No future
observations are accepted as extra request fields. `--device auto` chooses CUDA
when available; CPU and CUDA may also be requested explicitly. Only protocol
responses go to stdout. Both exporter and worker use the same two-tick scenario
and price-tree functions.

An actual-checkpoint worker response exactly matched a single-row historical
export. Compared with the previously generated 512-row CUDA batch, the largest
differences on that sample were 0.000727 USDT and 2.81e-5 probability, from the
different float32/high-matmul batch kernels. This does not establish bitwise
forecast identity across devices or batch sizes. The exact policy solver acts
on the concrete probabilities and prices it receives.

## Scope of optimality and market data

The historical target is **BTCUSDT spot one-second closes**, with some futures
features. Applying a derivatives fee/leverage configuration to this path is a
synthetic execution replay, not a recorded USD-M futures fill backtest.

The Lean Bellman guarantee can apply to the supplied finite forecast tree and
feasible action menu. The learned law, frozen future features and finite bin
representatives are additional statistical/numerical approximations. In
particular, bin means do not preserve nonlinear CRRA utility or liquidation
probabilities. Under risk aversion at least one, positive-probability zero wealth
has utility negative infinity; discarded adverse tails can therefore materially
change the continuous-law optimum. No continuous-law optimality certificate is
claimed by this forecast exporter.

## Reproduction and artifacts

From the repository root with the configured ML environment:

```powershell
.venv-ml/Scripts/python.exe ml/train_stochastic_price_forecast.py
.venv-ml/Scripts/python.exe ml/compare_stochastic_price_forecasts.py
.venv-ml/Scripts/python.exe ml/export_stochastic_price_forecast.py --output data/backtests/stochastic-crra/test-full-3bins.jsonl
.venv-ml/Scripts/python.exe -u ml/serve_stochastic_price_forecast.py --device auto
.venv-ml/Scripts/python.exe -m unittest discover -s ml -p 'test*stochastic_price_forecast.py'
```

The trainer refuses to overwrite an existing `best.pt`; give a fresh `--output`
for another run and pass that checkpoint to comparison/export commands.

- Dataset: `data/training/datasets/next-return-production-basis-history120-4l-train512k-v1`.
- Run: `data/training/runs/stochastic-crra-production178-zeroatom-k256-v1`.
- Selected checkpoint: `best.pt`, SHA256 `5f7eb5252bc411d0d8919644f3cbbed3b0382d8f3624e81ce6f0e15a44979029`.
- Baseline: `marginal.pt`; validation choice: `model-selection.json`; comparison: `model-comparison.json`.
- Forecasts: `data/backtests/stochastic-crra/test-first-hour-3bins.jsonl` and `test-full-3bins.jsonl`, each with a `.manifest.json` sidecar.

Eight focused tests cover zero mass, scenario normalization/mean preservation,
causal hypothetical updates, past-only timeline features, the train-fitted
basis contract, worker validation/error responses and worker/export equality.
They do not replace the separate policy/execution proof checks.
