# Three-layer normalized GLU for a 5-second return path

Date: 2026-08-04

## Question

Does a third width-512 normalized GLU layer improve the joint prediction of the
next five BTCUSDT one-second returns beyond the two-layer model?

## Protocol

- Input: 120 completed BTCUSDT one-second log returns.
- Output: the next five one-second log returns, predicted jointly.
- Architecture: three width-512 fused normalized GLU layers.
- Parameters: 4,325,899.
- Loss: normalized per-candle MSE plus normalized compounded cumulative-return
  MSE. Mean, variance, minimum, and maximum have zero loss weight.
- Dataset, chronological split, normalization, seed, optimizer family, and
  validation selection objective: identical to the one- and two-layer 5-second
  runs.
- Training batch size: 65,536. The third layer did not fit safely with the
  two-layer run's 131,072 training batch on the 8 GB GPU.
- Validation batch size: 131,072.
- Epoch cap: 32; early-stopping patience: 8.
- Actual epochs: 15, indices 0 through 14.
- Selected checkpoint: epoch 6.
- Learning-rate reductions: `5e-5` at epoch 10 and `2.5e-5` at epoch 14.
- Epoch compute time: approximately 22.2 minutes total.
- Observed warm GPU utilization: 99%, using 6.24 of 8.19 GB VRAM.
- Test evaluation: disabled and sealed.

The smaller training batch is a necessary memory accommodation and means this
is not a perfectly isolated depth ablation: the three-layer model takes twice
as many optimizer steps per epoch as the two-layer model.

## Validation comparison by depth

| Metric | 1 layer | 2 layers | 3 layers |
|---|---:|---:|---:|
| Parameters | 1,176,071 | 2,750,985 | 4,325,899 |
| Best epoch index | 15 | 11 | 6 |
| Selection objective | 2.012766 | **2.010381** | 2.010553 |
| Aggregate five-lead MSE skill | +0.2413% | **+0.3055%** | +0.3049% |
| Direction accuracy | 52.1551% | **52.7201%** | 51.1036% |
| Correlation | 0.04925 | **0.05540** | 0.05526 |
| Cumulative-return normalized MSE | 1.000454 | **0.998720** | 0.998887 |

Three layers essentially tie two layers on magnitude-sensitive metrics, but do
not beat it. The aggregate skill difference is only 0.00054 percentage point,
yet two layers also have the better selection objective, correlation,
cumulative-return error, and direction accuracy while using 36% fewer
parameters than three layers.

## Per-lead MSE skill

| Depth | +1s | +2s | +3s | +4s | +5s |
|---|---:|---:|---:|---:|---:|
| 1 layer | +0.7608% | +0.2505% | +0.1027% | +0.0585% | +0.0342% |
| 2 layers | +1.0158% | **+0.2900%** | **+0.1211%** | **+0.0643%** | **+0.0362%** |
| 3 layers | **+1.0262%** | +0.2825% | +0.1186% | +0.0632% | +0.0341% |

The third layer improves only the first lead, by 0.0104 percentage point over
two layers. Leads 2 through 5 are all slightly worse, so the added capacity does
not improve the whole path.

## Training stability

The three-layer validation objective improved through epoch 6, then degraded
continuously. Aggregate skill fell from +0.3049% at the selected checkpoint to
negative territory by epoch 13. Reducing the learning rate after the plateau
did not reverse the deterioration before early stopping.

This behavior suggests that the third layer needs different optimization, such
as a lower initial learning rate, residual connections, or layer-specific
scaling. Under the current normalized-GLU training recipe, two layers remain
the preferred 5-second model.

## Artifacts

- Plan:
  `ml/training-plans/horizon-cumulative-only-glu-5s-3layer-32epoch-v1.json`
- Result:
  `data/training/runs/horizon-cumulative-only-glu-5s-3layer-32epoch-v1/state/result.json`
- Best checkpoint metadata:
  `data/training/runs/horizon-cumulative-only-glu-5s-3layer-32epoch-v1/checkpoints/best.json`
- Training log:
  `data/training/runs/horizon-cumulative-only-glu-5s-3layer-32epoch-v1/logs/training.jsonl`

