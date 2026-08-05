# Reversible per-sequence normalization with mean/std side features

Date: 2026-08-04

## Question

Can the two-layer 5-second GLU benefit from normalizing every 120-return input
window locally, predicting the five future returns in the same local coordinate
system, and reversing the transform—provided the removed window mean and
standard deviation are explicitly passed to the model?

## Transformation

For each example, let `x` contain the 120 known past log returns. Calculate:

`mu = mean(x)`

`sigma = max(population_std(x), 1e-8)`

The GLU receives 122 values:

1. The 120 normalized returns `(x - mu) / sigma`.
2. The window mean as `(mu - training_mean) / training_std`.
3. The window standard deviation as `sigma / training_std - 1`.

Here `training_mean` and `training_std` are averages of the fixed per-position
training statistics. The two side features are therefore dimensionless and on
roughly unit scale rather than being passed as tiny raw return values.

The five-dimensional head predicts `z` in the local coordinate system. Before
the loss or metrics are computed, predictions are converted back to raw log
returns:

`predicted_return = mu + sigma * z`

The existing globally scaled candle MSE and cumulative-return MSE are then
computed in raw return space, preserving direct metric comparability.

## Protocol

- Input: 120 completed BTCUSDT one-second log returns plus two side features.
- Output: the next five one-second log returns.
- Model: two width-512 normalized GLU layers.
- Parameters: 2,753,033, only 2,048 more than the standard model.
- Data split, target/loss scales, loss weights, optimizer, seed, and epoch cap:
  identical to the other two-layer 5-second experiments.
- Training batch: 131,072; validation batch: 65,536.
- Epoch cap: 32; early-stopping patience: 8.
- Actual epochs: 16, indices 0 through 15.
- Selected checkpoint: epoch 7.
- Recorded total training time: approximately 19.8 minutes.
- Test evaluation: disabled and sealed.

An initial reversible run without the two side features was stopped at the
user's request after epoch 10. Its validation skill remained negative because
the network saw only normalized shape while the inverse transform added a mean
and scale it could not observe. It was replaced rather than treated as a final
experiment.

## Validation comparison

| Metric | Training-position | Per-sequence shape only | Reversible + mean/std |
|---|---:|---:|---:|
| Parameters | 2,750,985 | 2,750,985 | 2,753,033 |
| Best epoch | 11 | 6 | 7 |
| Selection objective | **2.010381** | 2.013016 | 2.015227 |
| Aggregate five-lead MSE skill | **+0.3055%** | +0.2404% | +0.1864% |
| Direction accuracy | **52.7201%** | 51.2990% | 48.4292% |
| Correlation | **0.05540** | 0.04903 | 0.04596 |
| Cumulative-return normalized MSE | **0.998720** | 1.000694 | 1.002357 |

Passing the missing statistics fixes the catastrophic negative-skill behavior
of the preliminary reversible run, but it does not make reversible local
normalization competitive. Aggregate skill is 39.0% lower than the standard
model and 22.5% lower than the shape-only local-normalization model.

## Per-lead MSE skill

| Normalization | +1s | +2s | +3s | +4s | +5s |
|---|---:|---:|---:|---:|---:|
| Training-position | **+1.0158%** | **+0.2900%** | **+0.1211%** | **+0.0643%** | **+0.0362%** |
| Shape only | +0.8162% | +0.2231% | +0.0896% | +0.0474% | +0.0255% |
| Reversible + mean/std | +0.7528% | +0.2123% | +0.0446% | -0.0199% | -0.0580% |

The reversible model retains positive signal at +1s through +3s, but becomes
worse than predicting zero at +4s and +5s. Local mean and volatility are useful
as information, but forcing all outputs through `mu + sigma * z` is a harmful
output parameterization for this return forecast.

## Conclusion

Keep the training-position normalization and globally scaled output head as the
default. If local mean and volatility are tested again, the cleaner next
ablation is to append them as side features while retaining the global output
transform. That would add information without forcing the forecast to inherit
the local input window's mean and scale.

## Artifacts

- Plan:
  `ml/training-plans/horizon-cumulative-only-glu-5s-2layer-sequence-reversible-stats-32epoch-v1.json`
- Result:
  `data/training/runs/horizon-cumulative-only-glu-5s-2layer-sequence-reversible-stats-32epoch-v1/state/result.json`
- Best checkpoint metadata:
  `data/training/runs/horizon-cumulative-only-glu-5s-2layer-sequence-reversible-stats-32epoch-v1/checkpoints/best.json`
- Training log:
  `data/training/runs/horizon-cumulative-only-glu-5s-2layer-sequence-reversible-stats-32epoch-v1/logs/training.jsonl`
