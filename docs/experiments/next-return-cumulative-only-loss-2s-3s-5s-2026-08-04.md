# Cumulative-only path loss for 2s, 3s, and 5s forecasts

Date: 2026-08-04

## Question

Does the short-horizon return predictor improve when mean, variance, minimum,
and maximum are removed from the training objective, leaving only individual
candle accuracy and cumulative return?

The new objective is:

`candle normalized MSE + cumulative-return normalized MSE`

where cumulative return is `expm1(sum(log returns))` over the predicted path.
Mean, variance, minimum, and maximum are still calculated as diagnostics, but
their loss weights are zero.

## Protocol

- Input: 120 completed BTCUSDT one-second log returns.
- Horizons: 2, 3, and 5 future one-second returns.
- Models: linear and one-layer width-512 normalized GLU.
- Train and validation rows: identical to the previous cumulative-weighted
  screen for each matching horizon.
- Train-only normalization: reused from the previous linear dataset manifest.
- Selection: lowest candle-plus-cumulative validation objective.
- Maximum epochs: 16; early-stopping patience: 5.
- Test evaluation: disabled. All six prior test blocks remain sealed for this
  experiment.

The runner now supports zero-weight diagnostic summaries and validation-only
completion. It also avoids validating the constant summary-weight tensor with a
GPU-to-CPU synchronization inside every batch.

## Validation results

| Horizon | Model | Best epoch | Objective | Candle MSE skill vs zero | Direction accuracy | Correlation | Cumulative normalized MSE |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5s | Linear | 10 | 2.019408 | +0.0848% | 53.8296% | 0.03002 | 1.005507 |
| 5s | GLU | 15 | **2.012766** | **+0.2413%** | 52.1551% | **0.04925** | **1.000454** |
| 3s | Linear | 0 | 2.018156 | +0.1232% | 55.3701% | 0.03622 | 1.004638 |
| 3s | GLU | 15 | **2.010466** | **+0.3706%** | 52.4305% | **0.06109** | **0.999457** |
| 2s | Linear | 0 | 2.018071 | +0.1946% | 55.6743% | 0.04476 | 1.005276 |
| 2s | GLU | 14 | **2.009919** | **+0.5013%** | 53.6848% | **0.07096** | **1.000235** |

Every model now beats the raw zero-return candle MSE baseline on validation.
The GLU wins the requested objective, raw candle MSE skill, and correlation at
every horizon. The 2-second GLU is the strongest result, followed by 3 seconds
and then 5 seconds.

Direction accuracy is higher for the linear models, but this can be inflated
by a persistent sign bias. The GLUs have materially higher correlation and MSE
skill, which better reflect prediction magnitude as well as direction.

## Before and after removing statistics loss

For an apples-to-apples comparison, the old checkpoint is rescored as its
candle normalized MSE plus cumulative-return normalized MSE. The old checkpoint
was originally selected using extrema and variance losses; the new checkpoint
is selected directly on this two-term objective.

| Horizon | Model | Old direct objective | New direct objective | Old candle MSE skill | New candle MSE skill |
|---:|---|---:|---:|---:|---:|
| 5s | Linear | 2.053879 | **2.019408** | -3.3238% | **+0.0848%** |
| 5s | GLU | 2.053026 | **2.012766** | -3.7486% | **+0.2413%** |
| 3s | Linear | 2.041090 | **2.018156** | -2.1022% | **+0.1232%** |
| 3s | GLU | 2.035811 | **2.010466** | -2.1482% | **+0.3706%** |
| 2s | Linear | 2.027283 | **2.018071** | -0.7217% | **+0.1946%** |
| 2s | GLU | 2.022095 | **2.009919** | -0.7188% | **+0.5013%** |

The previous loss was the cause of the negative candle skill. Exact min/max and
variance matching encouraged artificial return dispersion at selected future
positions. Removing those terms eliminates that failure mode and changes every
validation candle result from negative to positive.

## Per-lead validation MSE skill

| Horizon/model | +1s | +2s | +3s | +4s | +5s |
|---|---:|---:|---:|---:|---:|
| 5s linear | +0.314% | +0.107% | +0.016% | -0.002% | -0.012% |
| 5s GLU | **+0.761%** | **+0.250%** | **+0.103%** | **+0.058%** | **+0.034%** |
| 3s linear | +0.299% | +0.075% | -0.005% | — | — |
| 3s GLU | **+0.770%** | **+0.247%** | **+0.094%** | — | — |
| 2s linear | +0.301% | +0.088% | — | — | — |
| 2s GLU | **+0.764%** | **+0.239%** | — | — | — |

The first-lead GLU signal is stable across independently trained horizons at
approximately +0.76% validation MSE skill. Skill then decays with lead time but
remains positive through +5s for the GLU. This is much cleaner than the earlier
fixed-position spike behavior.

## Interpretation

The result supports two conclusions:

1. The historical 120-second return window contains a small short-horizon
   predictive signal.
2. Mean/variance/min/max path supervision overwhelmed that signal and made the
   ordered return predictions worse than zero.

The cumulative-return normalized MSE remains close to 1. The model has not
found a large cumulative-return advantage; most of the improvement is in the
individual candle path. The useful effect of keeping cumulative return is not
yet separated from simply training on candle MSE, so a candle-only ablation is
the next clean comparison.

The reported edge is small, and the 15.5 million validation examples are
strongly overlapping 120-second windows. Their effective independent sample
count is much lower. Before treating +0.24% to +0.50% MSE skill as tradable,
evaluate significance using non-overlapping time blocks and include fees,
spread, latency, and position-selection rules.

## Artifacts

Plans:

- `ml/training-plans/horizon-cumulative-only-linear-5s-v1.json`
- `ml/training-plans/horizon-cumulative-only-glu-5s-v1.json`
- `ml/training-plans/horizon-cumulative-only-linear-3s-v1.json`
- `ml/training-plans/horizon-cumulative-only-glu-3s-v1.json`
- `ml/training-plans/horizon-cumulative-only-linear-2s-v1.json`
- `ml/training-plans/horizon-cumulative-only-glu-2s-v1.json`

Validation-only results:

- `data/training/runs/horizon-cumulative-only-linear-5s-v1/state/result.json`
- `data/training/runs/horizon-cumulative-only-glu-5s-v1/state/result.json`
- `data/training/runs/horizon-cumulative-only-linear-3s-v1/state/result.json`
- `data/training/runs/horizon-cumulative-only-glu-3s-v1/state/result.json`
- `data/training/runs/horizon-cumulative-only-linear-2s-v1/state/result.json`
- `data/training/runs/horizon-cumulative-only-glu-2s-v1/state/result.json`

