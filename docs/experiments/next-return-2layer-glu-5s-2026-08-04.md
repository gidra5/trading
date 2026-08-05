# Two-layer normalized GLU for a 5-second return path

Date: 2026-08-04

## Question

Does adding a second width-512 normalized GLU layer improve the joint prediction
of the next five one-second BTCUSDT returns?

## Implementation

The next-return GLU was generalized from an exactly-one-layer model to a
configurable stack. A `widths` list now creates one fused GLU block per entry,
with every block retaining independent learned value/gate metric centering,
learned-radius normalization, full post-normalization transforms, biases, and
dropout. A one-entry list preserves the previous one-layer computation.

The architecture contract is now:

`next-return-fused-glu-stack-independent-branch-centering-sqrt-learned-radius-full-a-post-bias-v2`

## Protocol

- Input: 120 completed BTCUSDT one-second log returns.
- Output: the next five one-second log returns, predicted jointly.
- Architecture: two width-512 fused normalized GLU layers.
- Parameters: 2,750,985, versus 1,176,071 in the one-layer comparator.
- Loss: normalized per-candle MSE plus normalized compounded cumulative-return
  MSE. Mean, variance, minimum, and maximum have zero loss weight.
- Training examples: 33,219,925.
- Validation examples: 15,505,675.
- Batch size: 131,072 for training and validation.
- Epoch cap: 32.
- Early-stopping patience: 8 validation epochs.
- Actual epochs: 20, indices 0 through 19. Validation selected epoch 11.
- Learning rate: `1e-4`, reduced to `5e-5` at epoch 15 and `2.5e-5` at
  epoch 19.
- Epoch compute time: approximately 17.0 minutes total.
- Observed warm GPU utilization: 99%, using 7.37 of 8.19 GB VRAM.
- Test evaluation: disabled and sealed.

The 32-epoch value was a cap rather than a forced epoch count. The run stopped
at epoch 19 after eight consecutive checkpoints failed to improve on epoch 11.

## Validation comparison

| Metric | One layer, 5s | Two layers, 5s | Change |
|---|---:|---:|---:|
| Best epoch index | 15 | 11 | -4 |
| Parameters | 1,176,071 | 2,750,985 | 2.34x |
| Selection objective | 2.012766 | **2.010381** | -0.002385 |
| Aggregate five-lead MSE skill | +0.2413% | **+0.3055%** | +0.0641 pp |
| Direction accuracy | 52.1551% | **52.7201%** | +0.5651 pp |
| Correlation | 0.04925 | **0.05540** | +0.00614 |
| Cumulative-return normalized MSE | 1.000454 | **0.998720** | -0.001735 |

The second layer improves aggregate MSE skill by about 26.6% relative. It also
improves the formal selection objective, direction agreement, correlation, and
cumulative-return error. The improvement is therefore not confined to one
diagnostic.

## Per-lead MSE skill

| Model | +1s | +2s | +3s | +4s | +5s |
|---|---:|---:|---:|---:|---:|
| One layer | +0.7608% | +0.2505% | +0.1027% | +0.0585% | +0.0342% |
| Two layers | **+1.0158%** | **+0.2900%** | **+0.1211%** | **+0.0643%** | **+0.0362%** |

Every lead improves, but the gain is concentrated at +1 second. The first-lead
improvement is +0.2550 percentage point, while the +5-second improvement is
only +0.0020 percentage point. Depth extracts more of the strong nearest-return
signal but does not stop predictability from decaying quickly with horizon.

## Conclusion

Two layers are better than one for this five-output task under the same split
and loss. The result justifies retaining configurable depth and testing whether
the second layer also helps other horizons. It does not yet establish trading
profitability, because the metrics exclude fees, spread, slippage, latency, and
position sizing.

## Artifacts

- Plan:
  `ml/training-plans/horizon-cumulative-only-glu-5s-2layer-32epoch-v1.json`
- Result:
  `data/training/runs/horizon-cumulative-only-glu-5s-2layer-32epoch-v1/state/result.json`
- Best checkpoint metadata:
  `data/training/runs/horizon-cumulative-only-glu-5s-2layer-32epoch-v1/checkpoints/best.json`
- Training log:
  `data/training/runs/horizon-cumulative-only-glu-5s-2layer-32epoch-v1/logs/training.jsonl`

