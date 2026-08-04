# Next-second return prediction: linear ridge versus one-layer normalized GLU

Date: 2026-08-04

## Objective

Predict the next completed BTCUSDT one-second close-to-close log return from
the preceding 120 completed one-second log returns.

This experiment compares a small linear baseline with the one-layer
learned-radius normalized GLU on exactly the same examples. It measures raw
return prediction only; it is not a trading backtest and does not include
fees, spread, slippage, or execution latency.

## Dataset and split contract

- Input: 120 adjacent completed one-second close log returns.
- Target: the immediately following completed one-second close log return.
- Training examples: 33,219,953.
- Validation examples: 15,505,775.
- Test examples: a sealed chronological tail of 1,000,000 examples.
- Cross-split embargo: 121 seconds, covering the full input and target span.
- Normalization: feature means/standard deviations and target mean/standard
  deviation are calculated from the training split only.
- Corpus fingerprint:
  `879aaeaa955e077dd44a29bf67003871de94e73ac6db9a3446c9b0cd9f597a85`.

The test split was not used to select an epoch, model, or ridge strength.

## Models

### Linear ridge

The linear model has 120 lag coefficients and one intercept, for 121 fitted
parameters. It was fitted in closed form from streamed sufficient statistics;
there are no training epochs.

Ridge strengths from 0 through 100 were evaluated using validation normalized
MSE. Validation selected lambda = 0.1. The intercept is not regularized.

The immediately preceding return has the largest raw coefficient, approximately
0.04738. The second and third preceding returns have coefficients approximately
0.02475 and 0.00927. This is consistent with short-lived positive
autocorrelation in the observed one-second closes, though it does not establish
that the effect survives execution costs or latency.

### One-layer normalized GLU

The nonlinear model contains one fused GLU layer of width 512, independent
learned value/gate metric-centering matrices, learned-radius normalization, and
a scalar return head. It has 1,174,019 trainable parameters.

Training allowed at most 24 epochs. Early stopping ended training after 15
completed epochs; the best validation checkpoint was epoch index 6, meaning the
seventh completed epoch.

## Results

### Validation

| Metric | Linear ridge | One-layer GLU |
|---|---:|---:|
| MSE | 7.056992e-9 | **7.028960e-9** |
| MAE | **2.900707e-5** | 3.026790e-5 |
| MSE skill versus zero | 0.3241% | **0.7200%** |
| Direction accuracy | **54.6655%** | 49.5769% |
| Correlation | 0.05694 | **0.08487** |
| Prediction standard deviation | 4.852609e-6 | 7.178308e-6 |
| Target standard deviation | 8.414237e-5 | 8.414237e-5 |

### Sealed test tail

| Metric | Linear ridge | One-layer GLU |
|---|---:|---:|
| MSE | 2.650663e-9 | **2.627060e-9** |
| RMSE | 5.148459e-5 | **5.125485e-5** |
| MAE | **1.556210e-5** | 1.692952e-5 |
| MSE skill versus zero | 1.0990% | **1.9796%** |
| Direction accuracy | **55.2207%** | 50.7012% |
| Correlation | 0.12171 | **0.14413** |
| Prediction standard deviation | 3.099806e-6 | 5.848473e-6 |
| Target standard deviation | 5.176984e-5 | 5.176984e-5 |

Direction accuracy means that the sign of the prediction and the sign of the
actual return agreed. Therefore the linear model's 55.2207% test direction
accuracy means sign agreement on about 552,207 of the 1,000,000 test examples.

MSE skill versus zero is `1 - model MSE / zero-prediction MSE`. Positive skill
means the model improves on always predicting a return of zero. It is a relative
error metric, not a percentage trading return.

## Interpretation

The linear model is the stronger sign classifier and has lower MAE. The GLU has
better correlation and MSE, indicating better prediction of return magnitude,
especially on larger errors that MSE penalizes heavily.

Both models produce conservative predictions: their prediction standard
deviations are much smaller than the actual return standard deviation. On the
test tail, the linear prediction standard deviation is about 6.0% of the target
standard deviation and the GLU prediction standard deviation is about 11.3%.

The GLU improves test MSE by 1.98% over zero, compared with 1.10% for the linear
model, despite using roughly 9,700 times as many parameters. The linear result
is therefore a useful low-complexity baseline, while the GLU captures additional
nonlinear magnitude information.

These results do not yet demonstrate a profitable strategy. The next useful
evaluation is a latency-aware trading backtest with spread, fees, slippage, an
explicit mapping from predicted return to exposure, and model-selection choices
kept separate from the final test period.

## Reproduction and artifacts

- Linear plan: `ml/training-plans/linear-next-second-return-v1.json`.
- Linear implementation: `ml/linear_next_second_return.py`.
- Linear fitter: `ml/fit_linear_next_second_return.py`.
- Linear result: `data/training/runs/linear-next-second-return-v1/state/result.json`.
- Linear model: `data/training/runs/linear-next-second-return-v1/model.json`.
- GLU plan: `ml/training-plans/normalized-glu-next-second-return-v1.json`.
- GLU result:
  `data/training/runs/normalized-glu-next-second-return-one-layer-v1/state/result.json`.

Run the linear experiment with:

```powershell
npm run mlp:experiment:linear-next-second-return
```

Run the GLU experiment with:

```powershell
npm run mlp:experiment:normalized-glu-next-second-return
```
