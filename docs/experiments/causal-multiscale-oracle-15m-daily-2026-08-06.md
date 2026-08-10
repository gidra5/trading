# Causal multiscale 15-minute oracle distribution (2026-08-06)

## Question

Can the learned-radius GLU that fitted oracle distributions from a hindsight
return path learn the same 15-minute trading distribution from the richer
causal market-history inputs, using the entropy-gated target-temperature
schedule?

## Contract

- One prediction per completed 1-minute candle.
- Input: the first 771 values of feature schema 6, retaining the 1-second,
  1-minute, 1-hour, and 1-day history blocks and dropping month/quarter blocks.
- Target: the exact differentiable 15-minute exposure-value oracle used by the
  learned-oracle bot backtest: one-candle delay and hold, 0.175% transition
  friction, 101 actions from -100 to +100 exposure, and temperature 0.01.
- Split boundaries purge the final 15 minutes so no target path crosses into a
  different split.
- Examples: 552,930 train, 258,105 validation, 43,185 sealed test.
- Model: 16-layer shrinking learned-radius fused GLU, widths 512 down to 272.
- Training loss: mean forward KL + batch p50 forward KL, equal weights, plus
  the inherited soft normalization and soft weight-bound regularizers.
- Best-checkpoint selection: validation mean production-temperature KL + exact
  validation p50 production-temperature KL.
- Temperature: entropy-gated schedule from 0.25 toward 0.01; advance only when
  validation mean curriculum-target KL is at most 0.05.
- LR: start at 1e-4, reduce after 16 stale epochs, restore the complete best
  model/optimizer/temperature trajectory on every reduction, floor at 1e-6.
- Intended automatic stop: LR floor + stale temperature + 100 stale validation
  epochs. The run was manually stopped at epoch 184 after the user judged the
  floor-LR saturation evidence sufficient.

## Training result

The best checkpoint was epoch 62. Training reached the 1e-6 LR floor at epoch
158 and was manually stopped at epoch 184. The final target temperature did
not progress below 0.168468: after two early temperature steps, curriculum KL
settled around 0.059, above the 0.05 gate.

Best epoch 62 validation metrics:

| Metric | Value |
| --- | ---: |
| Selection objective (mean KL + p50 KL) | 2.657265 |
| Mean raw KL | 1.372887 |
| Raw KL p50 | 1.284378 |
| Raw KL p90 | 2.420356 |
| Raw KL p95 | 2.683228 |
| Mean curriculum KL at T=0.168468 | 0.059603 |
| Curriculum KL p50 | 0.020192 |
| Curriculum KL p90 | 0.172061 |
| Curriculum KL p95 | 0.249550 |

The sealed test split was not used for checkpoint selection and was not
evaluated before backtesting.

## Bot backtest

Checkpoint epoch 62 was served as a direct 101-action distribution predictor.
Inference used the frozen canonical feature components directly and held at
most one decompressed feature day in memory; predicted distributions were not
cached to disk. The strategy used static confidence 0.75 and all other bot
parameters at their defaults.

Results over the 33 standard static windows:

| Metric | Result |
| --- | ---: |
| Measured 1-minute candles | 887,040 |
| Oracle decisions | 887,039 |
| Raw nonzero modal decisions | 59,308 (6.686%) |
| Nonzero after transition-cost conditioning | 0 |
| Signals / trades | 0 / 0 |
| Profitable windows | 0 / 33 |
| Return | 0% in every window |
| Model inference duration | 52.947 s |
| Whole suite wall duration | 73.876 s |

`latest-3m` was excluded because its newest days lie outside the frozen
594-day rich-feature corpus. The long `fit-full` window was inferred once and
reused for its four subwindows, as in the standard suite.

## Interpretation

The model learned nonzero directional modes, but not enough distributional
separation to overcome the backtest's 0.175% transition-cost conditioning.
The temperature gate plateau is consistent with this: causal history could
fit the softened distribution moderately well, but could not meet the gate
needed to approach the sharp production oracle. Under the current execution
defaults, this checkpoint has no tradable edge.

## Validation-only logit sharpening

To test whether the zero-trade result was primarily a calibration problem, a
single positive sharpening factor was fitted on validation only. For each
candidate factor, model logits were multiplied by the factor before softmax,
then both prediction and target were passed through the exact 0.175%
transition-cost conditioning for current exposures -100, -50, 0, +50, and
+100. Selection minimized conditioned mean forward KL plus conditioned p50
forward KL. The sealed test split remained untouched.

The selected factor was 16.0:

| Validation metric | Factor 1.0 | Factor 16.0 |
| --- | ---: | ---: |
| Conditioned mean KL + p50 KL | 4.713628 | 2.363043 |
| Raw mean KL | 1.372887 | 1.995838 |
| Raw p50 KL | 1.284378 | 0.570394 |
| Raw p90 KL | 2.420356 | 6.446474 |
| Raw p95 KL | 2.683228 | 8.630499 |
| Zero-state predicted nonzero mode | 0 / 258,105 | 0 / 258,105 |

Sharpening substantially improved the selected conditioned KL objective and
the median example, but worsened the upper-tail and mean raw KL. More
importantly, it did not produce a nonzero post-cost modal decision on any
validation example; this was also true for every tested factor through 32.

The complete 33-window backtest with factor 16.0 confirmed that result:

| Metric | Unsharpened | Sharpened |
| --- | ---: | ---: |
| Oracle decisions | 887,039 | 887,039 |
| Raw nonzero modal decisions | 59,308 | 59,308 |
| Mean predicted confidence | 30.890% | 38.006% |
| Nonzero after transition-cost conditioning | 0 | 0 |
| Signals / trades | 0 / 0 | 0 / 0 |
| Return | 0% | 0% |

This rejects the narrow hypothesis that broad output distributions alone
caused the zero-trade result. A scalar factor cannot change which raw action
has the largest logit, and even the increased nonzero logit margins were
insufficient to overcome the transaction-cost penalty. The next model change
should train explicitly through the transition-conditioned decision or a
post-cost utility objective rather than relying on post-hoc sharpening.

Artifacts:

- Plan: `ml/training-plans/causal-multiscale-oracle-daily-v1.json`
- Trainer: `ml/train_causal_multiscale_oracle.py`
- Best checkpoint: `data/training/runs/causal-multiscale-oracle-15m-daily-v1/checkpoints/best.json`
- Backtest report: `data/benchmarks/causal-multiscale-oracle-15m-suite-2026-08-06.json`
- Sharpening calibration: `data/training/runs/causal-multiscale-oracle-15m-daily-v1/calibration/logit-sharpening.json`
- Sharpened backtest report: `data/benchmarks/causal-multiscale-oracle-15m-sharpened-suite-2026-08-06.json`
