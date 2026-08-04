# Candle-only loss ablation for 2s, 3s, and 5s forecasts

Date: 2026-08-04

## Question

Does cumulative-return supervision improve the short-horizon predictor, or
should the model optimize only individual future-candle returns?

The candle-only objective is:

`mean lead-normalized candle MSE`

It is compared with the immediately preceding objective:

`mean lead-normalized candle MSE + cumulative-return normalized MSE`

Mean, variance, minimum, maximum, and cumulative return remain validation
diagnostics, but no path summary contributes gradient in the candle-only run.

## Protocol

- Input: 120 completed BTCUSDT one-second log returns.
- Horizons: 2, 3, and 5 future one-second returns.
- Models: linear and one-layer width-512 normalized GLU.
- Train/validation data, normalization, architecture, optimizer, seed, batch
  size, scheduler, and early stopping are unchanged.
- The candle-only training path skips summary calculation inside the gradient
  objective.
- Test evaluation is disabled. Existing test blocks remain sealed.

## Candle-only validation results

| Horizon | Model | Best epoch | Candle objective | Raw MSE skill vs zero | Direction accuracy | Correlation | Cumulative normalized MSE (diagnostic) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5s | Linear | 0 | 1.013994 | +0.0755% | 55.3780% | 0.02872 | 1.005401 |
| 5s | GLU | 14 | **1.012416** | **+0.2311%** | 52.5907% | **0.04822** | **1.000556** |
| 3s | Linear | 0 | 1.013384 | +0.1365% | 55.6632% | 0.03761 | 1.004562 |
| 3s | GLU | 14 | **1.011135** | **+0.3581%** | 53.2034% | **0.05999** | **0.999575** |
| 2s | Linear | 0 | 1.012717 | +0.2024% | 55.9198% | 0.04539 | 1.005286 |
| 2s | GLU | 14 | **1.009793** | **+0.4905%** | 53.8803% | **0.07023** | **1.000312** |

All six candle-only models retain positive validation MSE skill. The GLU still
wins raw MSE skill and correlation at every horizon, and shorter horizons
remain better.

## Does cumulative return help?

| Horizon | Model | Candle + cumulative MSE skill | Candle-only MSE skill | Difference from dropping cumulative |
|---:|---|---:|---:|---:|
| 5s | Linear | +0.0848% | +0.0755% | -0.0092 pp |
| 5s | GLU | **+0.2413%** | +0.2311% | -0.0102 pp |
| 3s | Linear | +0.1232% | **+0.1365%** | +0.0133 pp |
| 3s | GLU | **+0.3706%** | +0.3581% | -0.0125 pp |
| 2s | Linear | +0.1946% | **+0.2024%** | +0.0077 pp |
| 2s | GLU | **+0.5013%** | +0.4905% | -0.0108 pp |

For the GLU, cumulative-return supervision improves raw candle MSE skill at
all three horizons. The improvement is small but consistent: about 0.01
percentage point. It also improves GLU correlation at every horizon:

- 5s: 0.04925 with cumulative versus 0.04822 candle-only.
- 3s: 0.06109 versus 0.05999.
- 2s: 0.07096 versus 0.07023.

For the linear model, candle-only is slightly better at 2s and 3s, while
cumulative supervision is slightly better at 5s. These differences are also
very small.

## Per-lead GLU MSE skill

| Horizon/objective | +1s | +2s | +3s | +4s | +5s |
|---|---:|---:|---:|---:|---:|
| 5s candle + cumulative | **+0.761%** | **+0.250%** | **+0.103%** | **+0.058%** | **+0.034%** |
| 5s candle-only | +0.731% | +0.240% | +0.097% | +0.055% | +0.033% |
| 3s candle + cumulative | **+0.770%** | **+0.247%** | **+0.094%** | — | — |
| 3s candle-only | +0.742% | +0.239% | +0.094% | — | — |
| 2s candle + cumulative | **+0.764%** | **+0.239%** | — | — | — |
| 2s candle-only | +0.745% | +0.236% | — | — | — |

The cumulative term does not trade later-lead accuracy for aggregate path
accuracy. Instead, it slightly improves or preserves nearly every GLU lead,
with the largest difference at +1s.

## Conclusion

Keep cumulative-return supervision for the normalized GLU. It is not the
source of the earlier failure; mean/variance/min/max losses were. The preferred
objective remains:

`candle normalized MSE + cumulative-return normalized MSE`

The benefit over candle-only is small enough that it should not be considered
proven economically significant. However, its consistency across 2s, 3s, and
5s validation screens makes it the better current default.

The best current configuration remains the 2-second GLU with cumulative-return
supervision: +0.5013% validation candle MSE skill and 0.07096 correlation.

## Artifacts

Plans:

- `ml/training-plans/horizon-candle-only-linear-5s-v1.json`
- `ml/training-plans/horizon-candle-only-glu-5s-v1.json`
- `ml/training-plans/horizon-candle-only-linear-3s-v1.json`
- `ml/training-plans/horizon-candle-only-glu-3s-v1.json`
- `ml/training-plans/horizon-candle-only-linear-2s-v1.json`
- `ml/training-plans/horizon-candle-only-glu-2s-v1.json`

Validation-only results:

- `data/training/runs/horizon-candle-only-linear-5s-v1/state/result.json`
- `data/training/runs/horizon-candle-only-glu-5s-v1/state/result.json`
- `data/training/runs/horizon-candle-only-linear-3s-v1/state/result.json`
- `data/training/runs/horizon-candle-only-glu-3s-v1/state/result.json`
- `data/training/runs/horizon-candle-only-linear-2s-v1/state/result.json`
- `data/training/runs/horizon-candle-only-glu-2s-v1/state/result.json`

