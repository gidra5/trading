# Five-day BTC return prediction with correlation loss

Date: 2026-08-06

## Question

Can the two-layer normalized GLU learn the general movement of the next five
daily BTCUSDT returns more effectively when trained directly for correlation,
instead of per-candle or cumulative-return error?

## Setup

This experiment keeps the corpus, chronological 60/20/20 split, model, and four
joint telescoping-MA input variants from the daily MSE experiment unchanged:

- 120 daily-spaced input positions and five direct daily-return outputs.
- 881 training, 361 validation, and 362 sealed test examples.
- Two normalized-GLU layers of width 512.
- Raw daily returns and MA decompositions through 1w, 1M, and 3M.
- Maximum 32 epochs, early stopping patience 8, fixed seed 1337.
- Only the validation-selected MA depth is evaluated on test.

The loss is one minus the average of the five weighted Pearson correlations:

`loss = 1 - mean(correlation(prediction[:, lead], target[:, lead]))`

Each lead is correlated across examples in the training batch. There is no MSE,
cumulative-return, or other calibration term in the training loss. Validation
selection uses the same mean-per-lead correlation, calculated over the entire
validation split.

Correlation is invariant to prediction offset and positive scale. MSE,
direction accuracy, prediction mean, and prediction standard deviation are
therefore retained only as diagnostics; they do not affect training or model
selection.

## Validation MA-depth screen

| Maximum window | Parameters | Best epoch | Epochs run | Mean lead correlation | Flat correlation | MSE skill vs zero | Direction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw (`1d`) | 2,750,985 | 2 | 10 | 0.03211 | 0.03275 | +0.3264% | 52.08% |
| **`1w`** | **2,873,865** | **2** | **10** | **0.03393** | **0.03414** | **+0.3306%** | **52.41%** |
| `1M` | 2,996,745 | 3 | 11 | 0.00277 | 0.00488 | +0.2619% | 52.35% |
| `3M` | 3,119,625 | 4 | 12 | 0.03194 | 0.03239 | +0.3243% | 52.02% |

All four variants early-stopped. The 1w decomposition won, but its advantage
over raw and 3M inputs is very small. At the final epochs, training mean-lead
correlation was 0.85-0.87 while validation remained near zero. This is strong
overfitting on the small daily corpus, not a failure to optimize the training
objective.

## Sealed test result

| Metric | Correlation-trained 1w GLU |
| --- | ---: |
| Mean of five per-lead correlations | **0.02128** |
| Correlation after flattening all five outputs | 0.02134 |
| Direction accuracy | 48.01% |
| Aggregate MSE skill vs zero | -0.4190% |
| Prediction standard deviation | 0.000295 |
| Target standard deviation | 0.022671 |

Per-day test diagnostics:

| Lead | Correlation | MSE skill vs zero | Direction |
| --- | ---: | ---: | ---: |
| +1d | 0.04850 | -0.3111% | 47.79% |
| +2d | 0.01546 | -0.4597% | 48.07% |
| +3d | 0.01171 | -0.4531% | 48.34% |
| +4d | -0.02406 | -0.5372% | 48.07% |
| +5d | 0.05476 | -0.3338% | 47.79% |

The prediction standard deviation is only about 1.3% of the target standard
deviation. That is legal under a pure correlation objective: multiplying every
prediction by any positive constant leaves the objective unchanged. A later
calibration fit could restore amplitude without changing correlation, but it
cannot repair the weak test correlation.

## Comparison with the MSE-plus-cumulative model

The prior daily 1w GLU, selected using normalized per-candle MSE plus cumulative
five-day-return MSE, achieved a higher test mean-lead correlation of 0.04730
and flat correlation of 0.04736. The pure-correlation model reaches only 0.02128
and 0.02134 respectively.

Directly optimizing correlation therefore did not improve general movement
prediction on the sealed daily period. The limiting factor is generalization
from the small, non-stationary daily dataset rather than the choice between MSE
and correlation loss.

## Reproduce

```powershell
npm run mlp:experiment:multiscale-daily-correlation
```

The matrix results and checkpoints are under
`data/training/runs/multiscale-candle-1d-*-joint-5path-32epoch-daily-correlation-v1/`.
The selected test result is the 1w run's `state/result.json`.
