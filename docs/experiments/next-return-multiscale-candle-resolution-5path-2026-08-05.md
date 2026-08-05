# Five-candle return prediction at 1m and 1h resolution

Date: 2026-08-05

## Question

For models that directly predict the next five larger candles, how many
telescoping moving-average levels should be included in the joint input?

This is a candle-resolution experiment. It does not predict more one-second
candles:

- The 1m model consumes 120 consecutive 1m positions and predicts the next
  five 1m log returns.
- The 1h model consumes 120 consecutive 1h positions and predicts the next
  five 1h log returns.

All positions and targets are aligned to real UTC candle boundaries.

## Inputs

At each position, the raw return is decomposed into telescoping components.
For example, the 1m model through the 1d MA uses:

1. `ma_1d`
2. `ma_1h - ma_1d`
3. `return_1m - ma_1h`

Adding the components reconstructs the raw aligned candle return (within
float32 arithmetic). The model receives all components jointly. Separate
component models are intentionally excluded from this screen.

The screened maximum windows are:

- 1m resolution: 1m, 1h, 1d, 1w, 1M, and 3M (one through six components).
- 1h resolution: 1h, 1d, 1w, 1M, and 3M (one through five components).

## Model and selection

- Two normalized-GLU layers, width 512 each.
- Fixed training-set normalization for every component and lag position.
- Five raw candle-return outputs at the selected resolution.
- Training objective: normalized per-candle MSE plus normalized cumulative
  five-candle-return MSE.
- Up to 32 epochs, with early stopping selected on validation objective.
- The test split remains sealed while selecting MA depth.

The common-corpus aligned example counts are:

- 1m: 536,965 train and 248,405 validation examples.
- 1h: 8,589 train and 2,380 validation examples.

Every depth at a given resolution uses these same timestamps. All depths are
trimmed to the history available to the widest 3M case, so the comparison is
not biased by different sample ranges.

## Results

The common-corpus sweep completed all 11 configurations. The table reports
validation measurements at the checkpoint selected by the combined training
objective. Positive MSE skill means lower error than predicting zero; negative
skill means the model is worse.

### 1m resolution

| Maximum window | MA levels | Terms | Best epoch | Aggregate MSE skill | Direction accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1m (raw only) | 0 | 1 | 1 | **+0.004686%** | 50.7049% |
| 1h | 1 | 2 | 1 | +0.002191% | 51.1149% |
| 1d | 2 | 3 | 1 | -0.003142% | 51.1571% |
| 1w | 3 | 4 | 1 | -0.006887% | 50.8289% |
| 1M | 4 | 5 | 1 | -0.008898% | 51.0228% |
| 3M | 5 | 6 | 3 | -0.009987% | 50.0958% |

The raw-only winner's MSE skill by future 1m candle is:

| +1m | +2m | +3m | +4m | +5m |
| ---: | ---: | ---: | ---: | ---: |
| +0.023984% | +0.000190% | -0.000907% | +0.002076% | -0.001904% |

**Selection:** zero moving-average levels (one raw-return term). The aggregate
gain is only 0.004686%, and three of the last four leads are approximately
zero, so this is not evidence of a useful five-minute forecasting edge. It is
only the least complex and least bad choice in this screen.

The 1m model with an added 1h MA term has the lowest combined validation
objective (1.9090303 versus 1.9091291 for raw-only), because the objective also
includes the five-candle cumulative return. It is not selected here because
the stated selection metric is per-candle MSE skill, where raw-only is better.

### 1h resolution

| Maximum window | MA levels | Terms | Best epoch | Aggregate MSE skill | Direction accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1h (raw only) | 0 | 1 | 1 | -0.068839% | 50.8151% |
| 1d | 1 | 2 | 1 | -0.061343% | 50.8655% |
| 1w | 2 | 3 | 1 | -0.063562% | 51.1933% |
| 1M | 3 | 4 | 1 | **-0.038251%** | 50.3613% |
| 3M | 4 | 5 | 1 | -0.098672% | 50.2269% |

The 1M-depth winner's MSE skill by future 1h candle is:

| +1h | +2h | +3h | +4h | +5h |
| ---: | ---: | ---: | ---: | ---: |
| -0.004800% | -0.072475% | -0.055226% | -0.028477% | -0.030251% |

**Selection:** three moving-average levels through 1M (four telescoping
terms), if a depth must be selected. This configuration also has the lowest
combined validation objective. However, it is worse than the zero predictor
in aggregate and at every lead, so it should not be treated as a successful
1h return model.

### Test policy

The test split was not evaluated for any configuration. These are validation
results used only for selecting MA depth.

An initial sweep that trimmed each depth independently was discarded before
selection because its 3M sample range differed from the shallower cases.
