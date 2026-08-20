# Production-basis next-return GLU

Date: 2026-08-17  
Status: complete

## Question

Does adding the currently usable production input basis from
`recommended-model-input-basis-2026-08-17.md` to the existing 120-return input
improve a four-layer next-1s GLU?

## Dataset contract

- Target: next completed BTCUSDT 1s signed log return.
- Exact-zero targets are excluded from train, validation, and test metrics.
- Zero returns remain in the 120-return input histories and activity features.
- Each split contains exactly 65,536 retained examples.
- Train starts 2026-07-18, validation starts 2026-08-03, and untouched test
  starts 2026-08-10.
- All features are causal at the end of the origin second. Minute and hour
  features use only completed bars.

The exported matrix has 178 inputs:

- 120 raw signed 1s return lags;
- two volatility-normalized return lags and their exact-zero flags;
- 10s/60s active fractions and clipped log zero-run age;
- invertible log-RMS volatility anchor/differences for 5s, 15s, 1m, 5m,
  15m, 30m, 1h, and 4h state;
- log mean absolute return at 5s, 15s, 1m, 5m, 15m, and 1h;
- RSI(2), EMA acceleration(2,1), and EMA slope(8,8), using the canonical
  mapped/volatility-normalized forms;
- last completed 1h log base volume, observed flag, and age;
- completed 1s normalized range and close location;
- normalized 16s Haar contrast and both signed 16s efficiency ratios;
- one-hot last aggressor side for the latest and 2s-old completed flow bins;
- latest spot quote imbalance, its EMA(2s)/EMA(8s), and aggregate-count
  imbalance;
- last completed USD-M futures 1m log trade count and range, plus observed/age;
- ETH-minus-BTC 30m/60m log-RMS volatility, plus observed/age;
- cyclic UTC second, minute, hour, and day-of-week sine/cosine pairs.

The exact coordinate list and constructions are stored in the generated
dataset `manifest.json`.

## Omitted for insufficient history

- Spot order book: only 18,954 fresh recent origins, below one 65,536-example
  split.
- Cross-exchange books, Deribit, liquidations, GDELT, mempool, and macro/ETF
  feeds were unavailable at this dataset's cutoff or had insufficient point-in-time
  history for its fixed 65,536-example splits. This is a dataset-availability
  statement, not a claim that every family remains unmeasured: fast books,
  Binance BTC liquidations, and Deribit perpetual/option trade flow received a
  separate early causal screen on 2026-08-19 using 29.701h of target coverage
  (20.155h for books).
- Futures basis/funding/positioning and research-only coordinates were not
  added because the recommendation explicitly excludes them from the current
  production basis.

## Model and training

- Four fused-GLU layers, each width 512.
- Static centering matrix, learned radius, no dropout or weight decay.
- 3,861,001 trainable parameters; 5,958,153 total parameters including fixed
  centering matrices.
- 512 epochs, batch size 4,096, learning rate 1e-4, hybrid Muon/AdamW.
- Training-set mean/std normalization for every input coordinate and target.
- Separate best-validation-MSE and best-validation-correlation checkpoints.

## Results

| selection | split | MSE skill vs zero | correlation | direction | normalized MSE |
|---|---|---:|---:|---:|---:|
| best validation MSE, epoch 9 | train | +9.727% | 0.32276 | 52.321% | 0.90274 |
| best validation MSE, epoch 9 | validation | +2.694% | 0.16948 | 50.459% | 3.64684 |
| best validation MSE, epoch 9 | test | +4.200% | 0.20545 | 48.662% | 1.71422 |
| best validation correlation, epoch 7 | validation | +2.581% | 0.17180 | 49.858% | 3.65105 |
| best validation correlation, epoch 7 | test | +4.165% | 0.20827 | 47.621% | 1.71485 |

Normalized MSE uses the training target variance as its denominator, so it is
not directly comparable between splits whose return volatility differs. MSE
skill uses each split's zero-prediction MSE and is the more interpretable
within-split score.

## Interpretation

The augmented model has a real held-out magnitude signal: MSE skill and
correlation are positive on both validation and untouched test, and the test
result is stronger than validation. It does not yet provide a reliable raw
sign classifier. Direction is approximately chance on validation and below
50% on test despite positive correlation, consistent with a small output bias
and weak predictions concentrated near zero.

The full-budget trajectory strongly overfits after the first few epochs. Train
skill eventually exceeds 99.8%, while held-out performance peaks at epochs
7--9. Future comparisons should therefore retain per-epoch validation and use
the early selected checkpoint. A matched raw-120-return control on these exact
three clean splits is still required to attribute the gain specifically to the
new indicators rather than to this recent market window.

## Multi-step and 15-minute episode evaluation

The best-validation-MSE checkpoint was also evaluated beyond its trained
one-step objective. For the step-2 and step-3 rows below, the scalar prediction
is appended to the return history and the model is called again. Zero realized
returns are excluded from the target paths, matching the clean-example
contract.

| split | active-return step | MSE skill vs zero | correlation | direction |
|---|---:|---:|---:|---:|
| train | 1 | +9.727% | 0.32277 | 52.321% |
| train | 2 | -1.560% | 0.14138 | 58.176% |
| train | 3 | -8.003% | 0.09823 | 49.881% |
| validation | 1 | +2.694% | 0.16947 | 50.458% |
| validation | 2 | -0.065% | 0.07187 | 60.183% |
| validation | 3 | -0.727% | 0.05911 | 47.983% |
| test | 1 | +4.200% | 0.20545 | 48.663% |
| test | 2 | -1.241% | 0.09370 | 62.807% |
| test | 3 | -3.414% | 0.07703 | 47.018% |

The direction score at step 2 is not accompanied by positive MSE skill. It is
therefore not evidence of a well-calibrated magnitude forecast; a directional
imbalance or recurrent output bias can produce this combination.

For the longer test, each wall-clock 15-minute episode is cleaned of exact-zero
realized returns and the model generates the same number of active returns
open-loop. Endogenous return-history features are updated from predictions.
Future-unknown exogenous features are frozen at the episode origin, avoiding
look-ahead leakage.

| split | episodes | active candles | pooled MSE skill | pooled corr. | pooled direction | mean within-episode corr. | cumulative-path corr. |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 162 | 65,536 | -6.915% | -0.00068 | 50.200% | -0.00640 | 0.03432 |
| test | 169 | 65,536 | -13.176% | 0.01107 | 50.587% | -0.00040 | 0.11381 |

This is a sharp separation between one-step and rollout behavior. The model
has useful one-step magnitude information, but errors in predicted inputs
compound rapidly. Across complete 15-minute open-loop paths it is worse than
predicting zero in MSE and has essentially zero candle-return correlation.
The positive test cumulative-path correlation is weak and does not offset the
large negative MSE skill.

The same metrics are stored for the best-validation-correlation checkpoint in
`checkpoint-selection-comparison.json`. That checkpoint is slightly less bad
in rollout MSE (-5.593% validation and -11.338% test pooled skill), but its
within-episode correlations remain effectively zero.

## Artifacts

- Dataset exporter: `scripts/export-next-return-production-basis.ts`
- Training plan: `ml/training-plans/next-return-production-basis-4l-65k-v1.json`
- Trainer: `ml/train_feature_augmented_next_return.py`
- Multi-step evaluator: `ml/evaluate_feature_augmented_next_return.py`
- Dataset: `data/training/datasets/next-return-production-basis-4l-65k-v1`
- Run: `data/training/runs/next-return-production-basis-4l-65k-v1`
