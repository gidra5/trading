# Autoregressive next-minute GLU on the latest four months (2026-08-07)

## Split

The experiment reads the immutable one-second candles directly rather than
reusing the historical oracle-corpus split:

| Split | Target dates | Minute-aligned examples |
| --- | --- | ---: |
| Train | 2026-04-01 through 2026-05-31 | 87,838 |
| Validation | 2026-06-01 through 2026-06-30 | 43,198 |
| Test | 2026-07-01 through 2026-07-23 | 33,120 |

The candle archive ends on July 24. July 24 is used as future source data for
the final July 23 targets but is not itself a target day, because July 25 is not
available.

The last training and validation decision is three minutes before the first
decision in the next split. Each decision touches a 180-second span: 120
seconds of input plus 60 seconds of target. This boundary embargo prevents a
return used as a late training target from also appearing in the first
validation input. It removes only two minute-aligned decisions at each split
boundary.

## Model and training

- Input: 120 completed one-second log returns.
- Shared model: one-layer normalized GLU, width 512, 1,174,532 parameters.
- Internal rollout: predict two seconds, append both predictions, and repeat 30
  times while retaining the complete gradient graph.
- External prediction and loss: one scalar next-minute log return, equal to the
  sum of the 60 generated returns.
- Initialization: random. The earlier two-second checkpoint was deliberately
  not reused because it had already seen parts of the new chronological splits.
- Selection: lowest validation normalized MSE; test remained sealed until the
  validation-selected checkpoint was fixed.
- Maximum 32 epochs, with eight-stale-epoch early stopping.

Training stopped after epoch 14 and selected epoch 6.

## Results

| Metric | Validation (June) | Sealed test (July 1-23) |
| --- | ---: | ---: |
| MSE | 5.189492e-7 | 2.394873e-7 |
| Zero-return MSE | 5.194784e-7 | 2.396068e-7 |
| MSE skill versus zero | **+0.10188%** | **+0.04986%** |
| Direction accuracy | 50.9399% | **52.0169%** |
| Correlation | 0.03311 | 0.03562 |
| Prediction standard deviation | 2.989736e-5 | 3.096913e-5 |
| Target standard deviation | 7.207291e-4 | 4.894863e-4 |

The normalized validation MSE is 2.2314 because normalization uses the quieter
April-May training standard deviation. MSE skill remains the appropriate
within-split comparison because both the model and zero baseline are evaluated
against the same June targets.

## Interpretation

The positive validation result survives the one-time chronological test, but
the error improvement remains tiny: about 0.05% on July. Direction agreement
is more encouraging at 52.02%, with positive correlation in both holdout
months. The model still predicts a narrow range relative to realized returns,
so these numbers establish weak forecast information, not profitability.

Artifacts:

- Plan: `ml/training-plans/autoregressive-glu-2s-to-1m-return-recent-4m-v2.json`
- Trainer: `ml/train_autoregressive_minute_return.py`
- Result: `data/training/runs/autoregressive-glu-2s-to-1m-return-recent-4m-v2/state/result.json`
- Best checkpoint: `data/training/runs/autoregressive-glu-2s-to-1m-return-recent-4m-v2/checkpoints/best.json`

## Single-pass ablation

A matched model was then trained without autoregressive propagation. It uses
the same 120 returns, split, normalization source, width, optimizer, seed, and
selection policy, but calls the GLU once and emits the next-minute return
directly. It has 1,174,019 parameters, 513 fewer than the two-output rollout
core.

| Metric | Autoregressive validation | Direct validation | Autoregressive test | Direct test |
| --- | ---: | ---: | ---: | ---: |
| MSE skill vs zero | +0.10188% | **+0.10910%** | +0.04986% | **+0.05997%** |
| Direction accuracy | 50.9399% | **51.3357%** | **52.0169%** | 51.6516% |
| Correlation | 0.03311 | **0.03553** | 0.03562 | **0.03935** |

The direct checkpoint was epoch 4 and early stopping ended after epoch 12.
Warm epoch time was about 0.35 seconds, compared with about 3.35 seconds for
the 30-step rollout. The direct model is therefore roughly ten times faster to
train and is slightly better on validation and test MSE and correlation. The
autoregressive model only wins the test sign-agreement metric.

Direct artifacts:

- Plan: `ml/training-plans/direct-glu-to-1m-return-recent-4m-v1.json`
- Result: `data/training/runs/direct-glu-to-1m-return-recent-4m-v1/state/result.json`
- Best checkpoint: `data/training/runs/direct-glu-to-1m-return-recent-4m-v1/checkpoints/best.json`
