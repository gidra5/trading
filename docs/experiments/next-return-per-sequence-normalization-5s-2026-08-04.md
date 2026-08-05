# Per-sequence input normalization for the two-layer 5-second GLU

Date: 2026-08-04

## Question

Does normalizing each 120-second input history by its own mean and standard
deviation improve the two-layer GLU that predicts the next five one-second
BTCUSDT returns?

## Change

The baseline applies fixed training-split statistics independently to each lag
position:

`(x[j] - training_mean[j]) / training_std[j]`

The variant instead uses only the current example's 120 input returns:

`(x - mean(x)) / sqrt(mean((x - mean(x))^2))`

The denominator has a `1e-8` floor for constant or nearly constant histories.
No validation or future values enter this calculation. Target normalization,
summary-loss scaling, internal learned-radius GLU normalization, architecture,
loss, optimizer, data split, and seed remain unchanged.

Per-sequence normalization makes the input invariant to adding a constant to
all 120 returns and, for positive scales above the floor, multiplying all 120
returns by a common factor. It therefore removes the recent history's absolute
mean return and volatility scale from the model input.

## Protocol

- Input: 120 completed one-second log returns.
- Output: the next five one-second log returns.
- Model: two width-512 normalized GLU layers, 2,750,985 parameters.
- Loss: normalized candle MSE plus normalized cumulative-return MSE.
- Training batch size: 131,072.
- Epoch cap: 32; early-stopping patience: 8.
- Actual epochs: 15, indices 0 through 14.
- Selected checkpoint: epoch 6.
- Test evaluation: disabled and sealed.
- Warm GPU utilization: 100%.
- Total recorded epoch time: approximately 23.3 minutes. Most epochs took
  roughly 50-85 seconds, but two epochs slowed under near-capacity WDDM VRAM
  pressure.

## Validation comparison

| Metric | Training-position normalization | Per-sequence normalization |
|---|---:|---:|
| Best epoch index | 11 | 6 |
| Selection objective | **2.010381** | 2.013016 |
| Aggregate five-lead MSE skill | **+0.3055%** | +0.2404% |
| Direction accuracy | **52.7201%** | 51.2990% |
| Correlation | **0.05540** | 0.04903 |
| Cumulative-return normalized MSE | **0.998720** | 1.000694 |

Per-sequence normalization reduces aggregate MSE skill by 0.0651 percentage
point, or 21.3% relative. It is worse on the selection objective, direction
accuracy, correlation, and cumulative-return error.

## Per-lead MSE skill

| Input normalization | +1s | +2s | +3s | +4s | +5s |
|---|---:|---:|---:|---:|---:|
| Training-position | **+1.0158%** | **+0.2900%** | **+0.1211%** | **+0.0643%** | **+0.0362%** |
| Per-sequence | +0.8162% | +0.2231% | +0.0896% | +0.0474% | +0.0255% |

Every lead becomes worse, not only the first one. The result is consistent with
the absolute mean and/or volatility of the recent 120-second history carrying
useful predictive information. Shape-only normalization discards that signal.

## Conclusion

Retain the fixed training-position input normalization for now. Per-sequence
normalization is a useful supported ablation, but it should not replace the
current default. A possible future hybrid would pass the sequence mean and
standard deviation as two additional features while normalizing the 120-return
shape; that would separate scale from shape without destroying information.

## Artifacts

- Plan:
  `ml/training-plans/horizon-cumulative-only-glu-5s-2layer-sequence-normalized-32epoch-v1.json`
- Result:
  `data/training/runs/horizon-cumulative-only-glu-5s-2layer-sequence-normalized-32epoch-v1/state/result.json`
- Best checkpoint metadata:
  `data/training/runs/horizon-cumulative-only-glu-5s-2layer-sequence-normalized-32epoch-v1/checkpoints/best.json`
- Training log:
  `data/training/runs/horizon-cumulative-only-glu-5s-2layer-sequence-normalized-32epoch-v1/logs/training.jsonl`

