# Fifteen-second return-path prediction with a one-layer normalized GLU

Date: 2026-08-04

## Objective

Predict the next 15 completed BTCUSDT one-second close-to-close log returns
from the preceding 120 completed one-second log returns.

This extends the next-return model to a configurable horizon `T`: the model
output contains one return for each of the next `T` candles, the dataset embargo
covers `120 + T` seconds, and the output layer and normalization contain `T`
lead-specific values.

## Multi-metric loss

For each predicted 15-return path, training minimizes:

1. Mean normalized MSE across the 15 individual candle returns.
2. The average normalized MSE across five whole-path summaries:
   mean, population variance, minimum, maximum, and compounded cumulative
   return.

Compounded cumulative return is computed from log returns as
`expm1(sum(log returns))`. Every candle lead and every summary metric is scaled
by its standard deviation calculated from the training split only.

The experiment assigns weight 1 to the per-candle loss family and weight 1 to
the averaged summary loss family. Thus the two families have equal influence
on the composite validation objective:

`objective = candle normalized MSE + average summary normalized MSE`

The weights are plan parameters and can be changed for later horizons.

## Dataset

- Horizon: 15 seconds.
- Input history: 120 seconds.
- Cross-split embargo: 135 seconds.
- Training examples: 33,219,855.
- Validation examples: 15,505,425.
- Sealed chronological test examples: 1,000,000.
- Corpus fingerprint:
  `1408252043a11885cab0630f4bd6938264036883b3e3fd17c54171979df649e9`.

The test tail was opened only after epoch selection was frozen using the
composite validation objective.

## Model and training

- One learned-radius normalized fused GLU layer of width 512.
- Independent learned value and gate metric-centering matrices.
- Fifteen-output linear return head.
- Trainable parameters: 1,181,201.
- Optimizer: hybrid Muon and AdamW.
- Maximum epochs: 24.
- Completed epochs: 24.
- Best checkpoint: epoch index 21, the 22nd completed epoch.
- Best validation composite objective: 1.845259.

## Aggregate results

| Metric | Validation | Sealed test |
|---|---:|---:|
| Composite objective | 1.845259 | 0.781794 |
| Candle normalized MSE | 1.103393 | 0.415730 |
| Average summary normalized MSE | 0.741866 | 0.366063 |
| Raw per-candle MSE | 7.697782e-9 | 2.900327e-9 |
| Per-candle MSE skill versus zero | -8.7666% | -8.2172% |
| Per-candle direction accuracy | 52.9970% | 53.2180% |
| Per-candle correlation | 0.01018 | 0.02248 |
| Prediction standard deviation | 2.577988e-5 | 1.604942e-5 |
| Target standard deviation | 8.412694e-5 | 5.176966e-5 |

The test objective is numerically lower partly because test-return and summary
variance differs from training and validation. It should not be interpreted as
a percentage improvement. MSE skill versus zero is the direct comparison with
always predicting zero returns, and it is negative.

## Path-summary results

| Test summary | Normalized MSE | Raw MSE | MAE |
|---|---:|---:|---:|
| Mean | 0.46335 | 2.599530e-10 | 9.295112e-6 |
| Variance | 0.01532 | 1.305417e-16 | 2.227977e-9 |
| Minimum | 0.41296 | 8.925173e-9 | 5.531619e-5 |
| Maximum | 0.47531 | 1.036609e-8 | 5.644351e-5 |
| Compounded cumulative return | 0.46338 | 5.849880e-8 | 1.394273e-4 |

The model improved the requested summary objective, especially variance, but
did so without accurately predicting the complete ordered return path.

## Per-lead behavior

| Lead | MSE skill vs zero | Direction accuracy | Correlation |
|---:|---:|---:|---:|
| 1s | -58.079% | 55.189% | 0.1145 |
| 2s | **+0.649%** | **57.480%** | 0.0818 |
| 3s | +0.352% | 55.972% | 0.0618 |
| 4s | +0.203% | 55.893% | 0.0473 |
| 5s | +0.103% | 56.717% | 0.0334 |
| 6s | +0.070% | 56.284% | 0.0277 |
| 7s | +0.035% | 55.870% | 0.0191 |
| 8s | -64.239% | 45.982% | -0.0107 |
| 9s | +0.012% | 52.822% | 0.0111 |
| 10s | +0.007% | 53.144% | 0.0088 |
| 11s | -0.006% | 49.862% | 0.0014 |
| 12s | -2.339% | 49.959% | -0.0009 |
| 13s | -0.006% | 51.701% | 0.0019 |
| 14s | -0.004% | 52.025% | 0.0021 |
| 15s | -0.016% | 49.369% | -0.0041 |

## Interpretation

The equal-weight multi-metric objective produced a clear objective-conflict
failure mode. Exact minimum and maximum losses send gradients through the
predicted extrema. The model learned high-variance outputs at a few fixed lead
positions, especially leads 1 and 8, to improve minimum, maximum, and variance
matching. Those spikes dominate squared candle error. Other leads remain very
small and several of leads 2 through 7 individually beat the zero-return MSE
baseline.

Consequently, aggregate sign agreement of 53.22% is not sufficient evidence of
a useful path predictor: aggregate candle MSE is 8.22% worse than zero. The
experiment successfully demonstrates configurable horizon training and summary
supervision, but equal weighting of exact extrema is not a good final loss.

A safer follow-up should keep per-candle prediction primary and treat summary
matching as a weaker regularizer, for example summary weight 0.05 to 0.2. It may
also replace exact min/max with smooth extrema or quantile losses, and add
prefix-cumulative-return losses so metric improvement cannot be obtained by
placing arbitrary spikes at fixed future leads. Any such choice should be made
on validation without reusing this already-opened test tail for selection.

## Cumulative-weighted follow-up

A second run changed the summary aggregation to the explicit weighted average:

`(mean + variance + minimum + maximum + 2 * cumulative return) / 6`

The top-level objective remains per-candle normalized MSE plus this weighted
summary average. It therefore gives cumulative return twice the influence of
each other summary metric without changing the total scale when weights change.

The follow-up used the same train and validation rows. Its test set is the
one-million-example block immediately preceding the test tail used above, so
the two runs have disjoint test rows. Their test metrics should not be compared
as if they were evaluated on the same market period.

### Shared-validation comparison

| Metric | Equal summary weights | 2x cumulative weight |
|---|---:|---:|
| Best epoch index | 21 | 21 |
| Per-candle normalized MSE | 1.103393 | **1.086676** |
| Summary weighted normalized MSE | **0.784957** | 0.800695 |
| Recomputed weighted objective | 1.888350 | **1.887370** |
| Cumulative-return normalized MSE | 1.000413 | **1.000042** |
| Minimum normalized MSE | **0.748074** | 0.791543 |
| Maximum normalized MSE | **0.766606** | 0.815751 |
| Per-candle MSE skill versus zero | -8.7666% | **-7.1186%** |
| Direction accuracy | 52.9970% | **56.8393%** |

The weighted run marginally wins its requested validation objective. It shifts
capacity away from matching exact extrema, improves cumulative and per-candle
error, and raises sign agreement. Cumulative-return normalized MSE remains near
1, however, so the model has not found a strong cumulative-return signal.

### Fresh disjoint test window

| Metric | 2x cumulative-weight run |
|---|---:|
| Composite objective | 1.420383 |
| Per-candle normalized MSE | 0.761246 |
| Summary weighted normalized MSE | 0.659136 |
| Raw per-candle MSE | 5.310804e-9 |
| MSE skill versus zero | -6.8319% |
| Direction accuracy | 56.1000% |
| Correlation | 0.02481 |
| Prediction standard deviation | 2.025979e-5 |
| Target standard deviation | 7.050657e-5 |

The fresh test result remains worse than predicting zero by MSE. Large extrema
are still concentrated at particular leads, now especially leads 1, 8, and 12.
Thus weighted averaging changes the tradeoff in the intended direction but does
not solve the exact-min/max gradient problem.

## Artifacts

- Plan:
  `ml/training-plans/normalized-glu-next-15-second-returns-v1.json`.
- Dataset manifest:
  `data/training/datasets/normalized-glu-next-15-second-returns-one-layer-v1/dataset.json`.
- Result:
  `data/training/runs/normalized-glu-next-15-second-returns-one-layer-v1/state/result.json`.
- Best checkpoint:
  `data/training/runs/normalized-glu-next-15-second-returns-one-layer-v1/checkpoints/best.json`.
- Training log:
  `data/training/runs/normalized-glu-next-15-second-returns-one-layer-v1/logs/training.jsonl`.

### Cumulative-weighted follow-up

- Plan:
  `ml/training-plans/normalized-glu-next-15-second-returns-cumulative-weighted-v2.json`.
- Result:
  `data/training/runs/normalized-glu-next-15-second-returns-cumulative-weighted-v2/state/result.json`.
- Best checkpoint:
  `data/training/runs/normalized-glu-next-15-second-returns-cumulative-weighted-v2/checkpoints/best.json`.
- Training log:
  `data/training/runs/normalized-glu-next-15-second-returns-cumulative-weighted-v2/logs/training.jsonl`.

Run the configured experiment with:

```powershell
npm run mlp:experiment:normalized-glu-next-15-second-returns
```

Run the cumulative-weighted follow-up with:

```powershell
npm run mlp:experiment:normalized-glu-next-15-second-returns:cumulative-weighted
```

## Linear cumulative-weighted follow-up

A final run replaced the GLU with a strictly linear map from 120 standardized
history returns to 15 standardized future returns. It has 1,815 parameters:
`120 * 15` coefficients and 15 intercepts.

Although the model is linear, the requested variance, min, max, and compounded
return objective is not a linear least-squares problem. The parameters were
therefore optimized with AdamW using the same weighted loss:

`candle MSE + (mean + variance + minimum + maximum + 2 * cumulative) / 6`

The run used the remaining untouched 500,000-example test block, offset by two
million examples from the source-test tail. It does not overlap either GLU test
window.

- Completed epochs: 20 of 24 before early stopping.
- Best epoch index: 11.
- Best validation objective: 1.922833.
- Test composite objective: 0.906653.
- Test per-candle MSE: 3.183343e-9.
- Test MSE skill versus zero: -6.5390%.
- Test direction accuracy: 49.9669%.
- Test correlation: 0.00984.
- Best individual test lead: +1 second with +1.4608% MSE skill and 53.4254%
  direction accuracy.

The linear model reproduced the same structural failure as the GLUs: exact
extrema supervision concentrated large prediction variance at selected leads,
especially leads 2, 8, 10, 11, and 12. Consequently the aggregate path is worse
than zero despite a useful first-lead return estimate on this test block.

### Runtime note

Epoch 0 took 28.8 seconds and later epochs averaged 19.1 seconds. Compilation
and first-pass caching account for roughly 9 to 10 seconds of the initial epoch,
but most recurring time is not linear-layer computation. Each epoch streams
33.2 million training and 15.5 million validation paths, reads roughly 23 GB of
input-feature values, processes 1,164 shard-bounded batches, and accumulates
per-lead and path-summary metrics in FP64.

### Linear artifacts

- Plan:
  `ml/training-plans/linear-next-15-second-returns-cumulative-weighted-v1.json`.
- Result:
  `data/training/runs/linear-next-15-second-returns-cumulative-weighted-v1/state/result.json`.
- Best checkpoint:
  `data/training/runs/linear-next-15-second-returns-cumulative-weighted-v1/checkpoints/best.json`.

Run it with:

```powershell
npm run mlp:experiment:linear-next-15-second-returns:cumulative-weighted
```
