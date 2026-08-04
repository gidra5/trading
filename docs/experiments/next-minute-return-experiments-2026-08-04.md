# Next-minute return prediction experiments

Date: 2026-08-04

## Objective

Predict the next completed BTCUSDT one-minute close-to-close log return from
the preceding 120 completed one-minute log returns, representing two hours of
price history.

Three models were completed on the same corpus:

1. A linear ridge regression baseline.
2. A 16-layer shrinking learned-radius normalized GLU.
3. A one-layer learned-radius normalized GLU of width 512.

These experiments evaluate return prediction only. They are not trading
backtests and do not include fees, spread, slippage, execution latency, or a
mapping from predicted return to exposure.

## Dataset and split contract

- Input: 120 adjacent completed one-minute close log returns.
- Target: the immediately following completed one-minute close log return.
- Logical training examples: 33,169,980.
- Logical validation examples: 15,327,300.
- Logical test examples: a sealed chronological tail of 1,000,000.
- Distinct weighted minute rows: 553,616 training, 255,810 validation, and
  16,690 test.
- Cross-split embargo: 121 minutes, or 7,260,000 milliseconds, covering the
  complete input and target candle span.
- Normalization: calculated from the training split only.
- Corpus fingerprint:
  `4669d93cb0f83b83e290230acf71ae75f2d5c9f3c88010445b92bcd297dd3544`.

The logical counts retain the oracle dataset's source-second weighting. Source
seconds that resolve to the same completed minute history are compacted into
one minute row with a multiplicity weight. Consequently, the 1,000,000-example
test tail represents 16,690 distinct minute rows rather than one million
distinct minute histories.

All model and epoch choices were frozen using validation before evaluating the
test tail.

## Models and training

### Linear ridge regression

The linear model predicts:

`next return = intercept + sum(lag coefficient * historical return)`

It contains 120 lag coefficients and one intercept, for 121 fitted parameters.
The model was fitted in closed form; it does not use epochs. Ridge lambda values
from 0 through 100 and the training-mean limit were compared using validation
MSE. Validation selected lambda = 1.0. The intercept was not regularized.

The largest absolute raw coefficient is lag 8 at approximately -0.00630. The
most recent return, lag 1, has a coefficient of approximately +0.00401. All
coefficients are small, consistent with the model producing predictions whose
variance is much lower than actual minute-return variance.

### 16-layer normalized GLU

- Widths: 512, 496, 480, 464, 448, 432, 416, 400, 384, 368, 352, 336, 320,
  304, 288, and 272.
- Trainable parameters: 15,082,289.
- Each fused GLU layer has independent learned value/gate metric-centering
  matrices and learned-radius normalization.
- Dropout: 0.05, applied with configured rate 0.5.
- Optimizer: hybrid Muon and AdamW.
- Initial learning rate: 0.0001.
- Maximum epochs: 24; early-stopping patience: 8.
- Completed epochs: 9, indexed 0 through 8.
- Best checkpoint: epoch index 0, the first completed epoch.

Training performance continued improving while validation performance quickly
deteriorated. For example, training MSE skill versus zero reached 9.57% by
epoch 8, while that epoch's validation skill was -8.98%. This is strong
overfitting, so the first checkpoint was retained.

### One-layer normalized GLU

- One fused GLU layer of width 512.
- Trainable parameters: 1,174,019.
- Independent learned 512 by 512 value/gate metric-centering matrices,
  learned-radius normalization, and a scalar return head.
- Dropout: 0.05, applied with configured rate 0.5.
- Optimizer: hybrid Muon and AdamW.
- Initial learning rate: 0.0001.
- Maximum epochs: 24; early-stopping patience: 8.
- Completed epochs: 10, indexed 0 through 9.
- Best checkpoint: epoch index 1, the second completed epoch.

This model overfit more slowly than the 16-layer model, but validation stopped
improving after the second epoch.

## Validation results

| Metric | Linear ridge | 16-layer GLU | One-layer GLU |
|---|---:|---:|---:|
| Parameters | **121** | 15,082,289 | 1,174,019 |
| Selected lambda | 1.0 | n/a | n/a |
| Best epoch index | n/a | 0 | 1 |
| MSE | 5.078084e-7 | 5.078980e-7 | **5.077320e-7** |
| RMSE | 7.126068e-4 | 7.126697e-4 | **7.125532e-4** |
| MAE | **4.168005e-4** | 4.169241e-4 | 4.173904e-4 |
| Direction accuracy | **51.2642%** | 50.7768% | 51.0775% |
| Correlation | 0.01724 | 0.01200 | **0.02122** |
| MSE skill versus zero | 0.02830% | 0.01065% | **0.04333%** |
| Prediction standard deviation | 1.234551e-5 | 5.220054e-6 | 1.629406e-5 |
| Target standard deviation | 7.127067e-4 | 7.127067e-4 | 7.127067e-4 |

All validation improvements over predicting zero are very small. The one-layer
GLU has the best validation MSE and correlation; the linear model has the best
validation MAE and sign agreement.

## Sealed test-tail results

| Metric | Linear ridge | 16-layer GLU | One-layer GLU |
|---|---:|---:|---:|
| MSE | 2.592686e-7 | **2.592238e-7** | 2.594960e-7 |
| RMSE | 5.091842e-4 | **5.091403e-4** | 5.094075e-4 |
| MAE | 3.289960e-4 | **3.289592e-4** | 3.296210e-4 |
| Direction accuracy | **50.7040%** | 50.5681% | 50.4640% |
| Correlation | **-0.00066** | -0.00191 | -0.00346 |
| MSE skill versus zero | -0.03120% | **-0.01393%** | -0.11896% |
| Prediction standard deviation | 8.710176e-6 | 5.173951e-6 | 1.590896e-5 |
| Target standard deviation | 5.091036e-4 | 5.091036e-4 | 5.091036e-4 |

The zero-prediction test MSE is 2.591877e-7. Therefore none of the three models
beats the zero-return predictor on test MSE. The least-negative MSE skill belongs
to the 16-layer GLU, but its -0.01393% result is still a failure to generalize.

Direction accuracy measures agreement between predicted and actual signs. A
value near 50% does not by itself demonstrate useful prediction, especially
without comparison to the majority-sign baseline. That baseline was not
recorded in these runs. The slightly positive direction accuracies coexist with
near-zero or negative test correlations and negative MSE skill.

## Interpretation

The one-minute experiments found no convincing out-of-sample next-return
signal. Validation chose nontrivial models, but every selected model performed
worse than predicting zero on the sealed test tail. Test correlations are also
slightly negative.

Predictions from every model are strongly shrunk. The test prediction standard
deviation is about 1.7% of target standard deviation for linear ridge, 1.0% for
the 16-layer GLU, and 3.1% for the one-layer GLU. The models mostly predict tiny
returns near the training mean.

The large GLU clearly memorized or exploited training-specific structure that
did not transfer to validation. Reducing it to one layer greatly reduced model
size and slowed overfitting, but did not improve the sealed test result. The
121-parameter ridge regression remained competitive with both neural models.

The experiment does not support using these models as a trading strategy. A
future attempt should first establish a stable walk-forward signal against
zero-return and majority-sign baselines, then evaluate it with fees, spread,
slippage, latency, and explicit exposure sizing.

## Artifacts

### Linear ridge

- Model and metrics:
  `data/training/runs/simple-linear-next-return-v1/model.json`.
- Dataset manifest:
  `data/training/datasets/simple-linear-next-return-v1/dataset.json`.
- Training log:
  `data/training/runs/simple-linear-next-return-v1/logs/training.jsonl`.

### 16-layer GLU

- Frozen plan:
  `data/training/runs/normalized-glu-next-return-v1/state/plan.json`.
- Result:
  `data/training/runs/normalized-glu-next-return-v1/state/result.json`.
- Training log:
  `data/training/runs/normalized-glu-next-return-v1/logs/training.jsonl`.
- Best checkpoint:
  `data/training/runs/normalized-glu-next-return-v1/checkpoints/best.json`.

### One-layer GLU

- Frozen plan:
  `data/training/runs/normalized-glu-next-return-one-layer-v1/state/plan.json`.
- Result:
  `data/training/runs/normalized-glu-next-return-one-layer-v1/state/result.json`.
- Training log:
  `data/training/runs/normalized-glu-next-return-one-layer-v1/logs/training.jsonl`.
- Best checkpoint:
  `data/training/runs/normalized-glu-next-return-one-layer-v1/checkpoints/best.json`.

These are archival one-minute runs. The active next-return experiment scripts
and package commands now target the later one-second experiments, so the
one-minute commands should not be inferred from the current package scripts.
