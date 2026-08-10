# Autoregressive GLU for one next-minute return (2026-08-07)

## Question

Can the best two-second return GLU be trained autoregressively to predict one
return over the immediately following minute?

## Contract

- Input: 120 completed one-second BTCUSDT log returns.
- Internal rollout: the pretrained two-second, one-layer normalized GLU predicts
  two returns, appends both predictions to the input, and repeats this 30 times.
- External output: one scalar, the sum of the 60 generated log returns.
- Target: the actual close-to-close log return over the next 60 seconds.
- Loss: normalized MSE on that single minute return. Gradients propagate through
  all 30 recurrent applications and every appended prediction.
- Examples are aligned one minute apart, so adjacent targets do not overlap.
- Split: 553,659 train, 258,405 validation, and 250 sealed test examples after
  applying the 180-second input-plus-target embargo and minute alignment.
- Initialization: validation-selected epoch 62 of
  `horizon-cumulative-only-glu-2s-long-v1`.
- Model parameters: 1,174,532 shared across every rollout step.
- Training: bfloat16 CUDA, batch 4,096, initial LR `1e-5`, at most 32 epochs,
  validation early stopping after eight stale epochs. Test evaluation remained
  disabled.

## Result

Training stopped after epoch 16 and selected epoch 8.

| Validation metric | Result |
| --- | ---: |
| Examples | 258,405 |
| MSE | 5.233406e-7 |
| Zero-return baseline MSE | 5.236791e-7 |
| MSE skill versus zero | **+0.06464%** |
| Direction accuracy | 50.9820% |
| Correlation | 0.03167 |
| Prediction standard deviation | 3.623690e-5 |
| Target standard deviation | 7.236563e-4 |

Training MSE continued improving after epoch 8, reaching +0.4704% training
skill at epoch 16, while validation deteriorated. The validation checkpoint
therefore correctly stopped the beginning of overfit.

## Interpretation

Full autoregressive training can extract a positive but very small one-minute
MSE advantage. The forecast variance is only about 5% of the target variance,
so the model mostly stays near zero and learns a weak directional component.
This is statistically more promising than a negative-skill model, but the
effect is far too small to assume tradability without a sealed evaluation and
an execution-cost backtest.

Artifacts:

- Model: `ml/autoregressive_minute_return.py`
- Trainer: `ml/train_autoregressive_minute_return.py`
- Plan: `ml/training-plans/autoregressive-glu-2s-to-1m-return-v1.json`
- Result: `data/training/runs/autoregressive-glu-2s-to-1m-return-v1/state/result.json`
- Best checkpoint: `data/training/runs/autoregressive-glu-2s-to-1m-return-v1/checkpoints/best.json`
