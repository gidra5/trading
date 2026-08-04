# Longer training for the 2-second one-layer GLU

Date: 2026-08-04

## Question

Can the one-layer GLU that jointly predicts the next two one-second returns
match the next-second predictor if it is trained substantially longer?

The remembered result of almost 2% was the old one-second GLU's **sealed-test
MSE skill of 1.9796%**. Its comparable validation MSE skill was **0.7200%**.
This distinction matters because model selection must compare validation with
validation. The long 2-second run did not evaluate its sealed test split.

## Protocol

- Input: 120 completed BTCUSDT one-second log returns.
- Output: the next two one-second log returns, predicted jointly.
- Model: one fused normalized-GLU layer of width 512 with 1,174,532 trainable
  parameters.
- Loss: normalized per-candle MSE plus normalized compounded cumulative-return
  MSE. Mean, variance, minimum, and maximum remain diagnostics with zero loss
  weight.
- Training examples: 33,219,946.
- Validation examples: 15,505,750.
- Maximum epochs: 64; all 64 completed.
- Batch size: 262,144; validation batch size: 131,072.
- Optimizer: hybrid Muon/AdamW, initial learning rate `1e-4`.
- Learning-rate reductions: `5e-5` at epoch 45 and `2.5e-5` at epoch 58.
- Selection: lowest validation candle-plus-cumulative objective.
- Test evaluation: disabled and sealed.
- Epoch compute time: approximately 34.1 minutes in total, excluding setup and
  the interrupted reporter/resume overhead.

The selected checkpoint was epoch index 62, the 63rd completed epoch. It was
also the best observed epoch for both aggregate candle MSE skill and first-lead
MSE skill, so there is no checkpoint-selection conflict in the final result.

## Validation result

| Metric | Short 2s GLU (16 epochs) | Long 2s GLU (64 epochs) | Change |
|---|---:|---:|---:|
| Best epoch index | 14 | 62 | +48 |
| Selection objective | 2.009919 | **2.008253** | -0.001665 |
| Aggregate two-lead MSE skill | +0.5013% | **+0.5511%** | +0.0498 pp |
| First-lead MSE skill | +0.7638% | **+0.8524%** | +0.0886 pp |
| Second-lead MSE skill | +0.2389% | **+0.2498%** | +0.0109 pp |
| Aggregate direction accuracy | **53.6848%** | 52.0245% | -1.6603 pp |
| Aggregate correlation | 0.07096 | **0.07432** | +0.00336 |
| First-lead correlation | 0.08751 | **0.09237** | +0.00486 |
| Cumulative-return normalized MSE | 1.000235 | **0.999075** | -0.001160 |

Longer training improved aggregate MSE skill by about 9.9% relative and
first-lead MSE skill by about 11.6% relative. Most of the gain went to the first
predicted second; the second lead improved only slightly.

## Comparison with the one-second GLU

The first output of the 2-second model is the closest comparison with the old
one-second model. The validation periods are effectively the same, although
the 2-second horizon's embargo removes 25 additional rows.

| Validation metric | Old 1s GLU | Long 2s GLU, first lead |
|---|---:|---:|
| MSE skill versus zero | +0.7200% | **+0.8524%** |
| Direction accuracy | 49.5769% | **52.1955%** |
| Correlation | 0.08487 | **0.09237** |
| MSE | 7.028960e-9 | **7.019438e-9** |

On validation, the hypothesis is supported: the 2-second model matches and
slightly exceeds the one-second model on its first predicted return. This is
true for MSE skill, direction agreement, correlation, and raw MSE.

It does **not** establish that the 2-second model matches the old 1.9796% test
skill. That number came from a different, unusually easier sealed test tail.
Evaluating the new model on its test split now would turn that split into model
selection feedback, so it remains sealed until a final model is frozen.

## Interpretation

The aggregate two-second skill remains lower than first-lead skill because the
second return is harder to predict: +0.2498% versus +0.8524%. Therefore adding
the second output does not preserve the same average predictability over both
seconds, even though it does not harm the first output.

Direction accuracy and MSE skill are different metrics. Direction accuracy is
the fraction of signs that agree; MSE skill is the relative squared-error
reduction versus always predicting zero. The almost-2% historical figure was
MSE skill, not sign accuracy.

## Artifacts

- Plan: `ml/training-plans/horizon-cumulative-only-glu-2s-long-v1.json`
- Result: `data/training/runs/horizon-cumulative-only-glu-2s-long-v1/state/result.json`
- Best checkpoint metadata:
  `data/training/runs/horizon-cumulative-only-glu-2s-long-v1/checkpoints/best.json`
- Training log:
  `data/training/runs/horizon-cumulative-only-glu-2s-long-v1/logs/training.jsonl`

