# Next-second GLU replication on the latest four months (2026-08-07)

## Question

The earlier one-layer GLU reached about 2% next-second MSE skill on the old
dataset split. This experiment repeats that task on the current chronological
windows to determine whether the result survives a dataset change.

## Matched setup

- Input: 120 completed one-second log returns.
- Output: the next single one-second log return.
- Model: one-layer normalized GLU, width 512, 1,174,019 parameters.
- Examples: one decision per second.
- Initialization: random; no earlier checkpoint was reused.
- Training: 24 epochs, batch size 32,768, learning rate 1e-4, hybrid
  Muon/AdamW optimizer, seed 1337, and validation-based checkpoint selection.
- Test policy: the July test was evaluated once, after selecting the best
  checkpoint using June validation only.

The chronological split is:

| Split | Target dates | Examples |
| --- | --- | ---: |
| Train | 2026-04-01 through 2026-05-31 | 5,270,280 |
| Validation | 2026-06-01 through 2026-06-30 | 2,591,880 |
| Test | 2026-07-01 through 2026-07-23 | 1,987,200 |

At the train-validation and validation-test boundaries, the last 120 adjacent
examples are omitted. Consequently, the final decision timestamp in one split
and the first in the next are 121 seconds apart: 120 seconds of input plus the
one-second target. This prevents any return used by a late target from
reappearing in the next split's first input.

## Results

Training completed all 24 epochs and selected epoch 22.

| Metric | Validation (June) | Sealed test (July 1-23) |
| --- | ---: | ---: |
| MSE | 5.621043e-9 | 2.438762e-9 |
| Zero-return MSE | 5.747291e-9 | 2.502585e-9 |
| MSE skill versus zero | **+2.19665%** | **+2.55029%** |
| Direction accuracy | 55.1296% | 54.0797% |
| Correlation | 0.14835 | 0.16017 |
| Prediction standard deviation | 1.172570e-5 | 8.629853e-6 |
| Target standard deviation | 7.581084e-5 | 5.002582e-5 |

## Comparison with the earlier run

| Metric | Old validation | Current validation | Old test | Current test |
| --- | ---: | ---: | ---: | ---: |
| MSE skill versus zero | +0.72004% | **+2.19665%** | +1.97963% | **+2.55029%** |
| Direction accuracy | 49.5769% | **55.1296%** | 50.7012% | **54.0797%** |
| Correlation | 0.08487 | **0.14835** | 0.14413 | **0.16017** |

The approximately 2% result is therefore replicated and exceeded on the new
calendar split. The held-out July improvement is 0.57066 percentage points
above the old test result. It is not merely an artifact of training longer:
the current run used the same 24-epoch budget and the same architecture and
optimizer contract.

This establishes measurable one-second forecast information for this split.
It does not by itself establish tradable profit: overlapping one-second
decisions, spread, fees, latency, and turnover remain outside the MSE metric.
The difference between validation months also demonstrates that the signal is
non-stationary, so future chronological windows should be evaluated in the
same sealed manner.

Artifacts:

- Plan: `ml/training-plans/direct-glu-to-next-1s-recent-4m-v1.json`
- Trainer: `ml/train_autoregressive_minute_return.py`
- Result: `data/training/runs/direct-glu-to-next-1s-recent-4m-v1/state/result.json`
- Best checkpoint: `data/training/runs/direct-glu-to-next-1s-recent-4m-v1/checkpoints/best.json`

## Two-layer depth ablation

A matched run added a second width-512 normalized GLU layer. All data,
normalization, target, optimization, seed, and selection settings remained
unchanged. Parameter count increased from 1,174,019 to 2,748,933.

| Metric | One layer validation | Two layers validation | One layer test | Two layers test |
| --- | ---: | ---: | ---: | ---: |
| MSE skill versus zero | **+2.19665%** | +2.14788% | **+2.55029%** | +2.43486% |
| Direction accuracy | 55.1296% | **55.1631%** | **54.0797%** | 53.9161% |
| Correlation | **0.14835** | 0.14665 | **0.16017** | 0.15698 |

The two-layer run selected zero-based epoch 6 and early-stopped after epoch 14
when validation had remained stale for eight epochs. Its test MSE skill was
0.11544 percentage points below the one-layer model. The extra layer therefore
adds about 2.34 times as many parameters without improving the primary metric;
the one-layer checkpoint remains the selected next-second model.

Two-layer artifacts:

- Plan: `ml/training-plans/direct-glu-to-next-1s-recent-4m-2-layer-v1.json`
- Result: `data/training/runs/direct-glu-to-next-1s-recent-4m-2-layer-v1/state/result.json`
- Best checkpoint: `data/training/runs/direct-glu-to-next-1s-recent-4m-2-layer-v1/checkpoints/best.json`

## One-layer extended-patience ablation

The one-layer model was rerun with a maximum of 256 epochs and early-stopping
patience 32. All other settings remained matched to the 24-epoch replication.
It retained zero-based epoch 22 as its best checkpoint and stopped after epoch
54, once 32 subsequent epochs had failed to improve validation.

| Metric | 24-epoch validation | Extended validation | 24-epoch test | Extended test |
| --- | ---: | ---: | ---: | ---: |
| MSE skill versus zero | **+2.19665%** | +2.19178% | **+2.55029%** | +2.54966% |
| Direction accuracy | 55.1296% | **55.1790%** | 54.0797% | **54.1752%** |
| Correlation | **0.14835** | 0.14815 | **0.16017** | 0.16006 |

The MSE result is effectively unchanged and very slightly worse. Increasing
the ceiling and patience therefore provided no evidence that the one-layer
model was undertrained; the original 24-epoch checkpoint remains the primary
selection by validation MSE.

Extended-run artifacts:

- Plan: `ml/training-plans/direct-glu-to-next-1s-recent-4m-1-layer-long-v2.json`
- Result: `data/training/runs/direct-glu-to-next-1s-recent-4m-1-layer-long-v2/state/result.json`
- Best checkpoint: `data/training/runs/direct-glu-to-next-1s-recent-4m-1-layer-long-v2/checkpoints/best.json`

## Two-layer extended-patience ablation

The two-layer width-512 model was also run with the 256-epoch ceiling and
early-stopping patience 32. It selected zero-based epoch 6 and stopped after
epoch 38. Training MSE continued to improve while validation deteriorated,
which is direct evidence of overfitting rather than insufficient patience.

| Metric | Extended one layer validation | Extended two layers validation | Extended one layer test | Extended two layers test |
| --- | ---: | ---: | ---: | ---: |
| MSE skill versus zero | **+2.19178%** | +2.14582% | **+2.54966%** | +2.43802% |
| Direction accuracy | **55.1790%** | 55.1736% | **54.1752%** | 53.9411% |
| Correlation | **0.14815** | 0.14660 | **0.16006** | 0.15705 |

The long two-layer result is effectively the same as its short run: test skill
moved only from +2.43486% to +2.43802%. It remains 0.11164 percentage points
below the extended one-layer model on sealed test MSE skill. The one-layer
architecture remains preferred.

Extended two-layer artifacts:

- Plan: `ml/training-plans/direct-glu-to-next-1s-recent-4m-2-layer-long-v2.json`
- Result: `data/training/runs/direct-glu-to-next-1s-recent-4m-2-layer-long-v2/state/result.json`
- Best checkpoint: `data/training/runs/direct-glu-to-next-1s-recent-4m-2-layer-long-v2/checkpoints/best.json`

## Four-layer extended-patience ablation

A four-layer width-512 model used the same 256-epoch ceiling and patience 32.
It selected zero-based epoch 2 and stopped after epoch 34. Its parameter count
was 5,898,761. The gap between improving training MSE and collapsing validation
MSE appeared even earlier than for two layers.

| Metric | One layer | Two layers | Four layers |
| --- | ---: | ---: | ---: |
| Parameters | 1,174,019 | 2,748,933 | 5,898,761 |
| Best epoch (zero-based) | 22 | 6 | 2 |
| Validation MSE skill | **+2.19178%** | +2.14582% | +2.03479% |
| Test MSE skill | **+2.54966%** | +2.43802% | +2.33415% |
| Test direction accuracy | **54.1752%** | 53.9411% | 52.2694% |
| Test correlation | **0.16006** | 0.15705 | 0.15317 |

Generalization degrades monotonically as depth grows from one to two to four
layers. The four-layer test skill is 0.21551 percentage points below one layer,
and its direction accuracy loses 1.9058 percentage points. The one-layer model
remains decisively preferred for this input and target.

Four-layer artifacts:

- Plan: `ml/training-plans/direct-glu-to-next-1s-recent-4m-4-layer-long-v2.json`
- Result: `data/training/runs/direct-glu-to-next-1s-recent-4m-4-layer-long-v2/state/result.json`
- Best checkpoint: `data/training/runs/direct-glu-to-next-1s-recent-4m-4-layer-long-v2/checkpoints/best.json`

## Four-layer fixed-subset memorization diagnostic

To distinguish insufficient capacity from generalization-oriented stopping, a
separate diagnostic trained the four-layer model on exactly 65,536 contiguous
examples from April 1. It used no validation or test data, zero dropout, zero
weight decay, float32 computation, a fixed 1e-4 learning rate, and no early
stopping or validation-driven scheduler. Metrics were recomputed after every
epoch in deterministic evaluation mode.

The run completed its 512-epoch ceiling and selected epoch 505:

| Deterministic train metric | Result |
| --- | ---: |
| Normalized MSE | 0.00155458 |
| MSE skill versus zero | **+99.84454%** |
| MSE | 7.813960e-12 |
| RMSE | 2.795346e-6 |
| Correlation | 0.999224 |
| Direction accuracy | 65.1581% |

This decisively shows that the architecture can memorize a smaller fixed
subset. It did not reach the deliberately strict 1e-4 normalized-MSE stopping
target, but removed 99.84% of baseline error. The earlier full-corpus run did
not approach this regime because it contained 5.27 million examples, retained
dropout, reduced learning rate according to validation, and stopped after
validation became stale.

Direction accuracy is not a contradiction: 41.92% of targets in this one-second
subset are exactly zero. Their fitted predictions are extremely close to zero,
but the direction metric treats zero as positive, so an infinitesimal negative
prediction counts as a sign error while contributing almost no MSE.

Memorization artifacts:

- Plan: `ml/training-plans/next-second-4-layer-memorization-65k-v1.json`
- Trainer: `ml/train_next_return_memorization.py`
- Result: `data/training/runs/next-second-4-layer-memorization-65k-v1/state/result.json`
- Best checkpoint: `data/training/runs/next-second-4-layer-memorization-65k-v1/checkpoints/best.json`

## Static-centering memorization ablation

The fixed-subset diagnostic was repeated with an exactly matched plan except
that every value- and gate-branch centering matrix was frozen at the canonical
projector `C = I - 11^T/d`. The normalization radius remained trainable. This
reduced the trainable count by 2,097,152 parameters, from 5,898,761 to
3,801,609, while leaving the model's stored state size unchanged.

Both runs used the same 65,536 examples, seed, batches, optimizer, fixed 1e-4
learning rate, and 512-epoch ceiling:

| Deterministic train metric | Learned C | Static C |
| --- | ---: | ---: |
| Trainable parameters | 5,898,761 | 3,801,609 |
| Best epoch (zero-based) | 505 | 500 |
| Normalized MSE | 0.00155458 | **0.00078884** |
| MSE skill versus zero | +99.84454% | **+99.92112%** |
| MSE | 7.813960e-12 | **3.965019e-12** |
| RMSE | 2.795346e-6 | **1.991236e-6** |
| Correlation | 0.999224 | **0.999613** |
| Direction accuracy | **65.1581%** | 62.6480% |

Static centering produced 49.27% less MSE than learned centering in this
memorization diagnostic despite using 35.55% fewer trainable parameters. The
lower sign score is not evidence of worse magnitude fit because 41.92% of the
targets are exactly zero and the sign metric is unstable for tiny predictions
around zero. Neither run reached the deliberately strict 0.0001 normalized-MSE
target.

This result supports static centering as an easier optimization problem for
fitting this subset. It is not evidence of better out-of-sample forecasting;
that requires a matched validation/test experiment.

Static-centering artifacts:

- Plan: `ml/training-plans/next-second-4-layer-memorization-65k-static-c-v1.json`
- Result: `data/training/runs/next-second-4-layer-memorization-65k-static-c-v1/state/result.json`
- Best checkpoint: `data/training/runs/next-second-4-layer-memorization-65k-static-c-v1/checkpoints/best.json`

## Static-centering dataset scaling: 128k

The static-centering memorization protocol was next scaled from 65,536 to
131,072 contiguous examples. The larger subset spans all 86,400 examples from
April 1 and the first 44,672 examples from April 2. Architecture, optimizer,
seed, batch sizes, fixed learning rate, lack of regularization, and the
512-epoch ceiling were unchanged.

| Deterministic train metric | 65k static C | 128k static C |
| --- | ---: | ---: |
| Examples | 65,536 | 131,072 |
| Best epoch (zero-based) | 500 | 493 |
| Normalized MSE | **0.00078884** | 0.00144216 |
| MSE skill versus zero | **+99.92112%** | +99.85579% |
| MSE | **3.965019e-12** | 7.206799e-12 |
| RMSE | **1.991236e-6** | 2.684548e-6 |
| Correlation | **0.999613** | 0.999289 |
| Direction accuracy | 62.6480% | **66.3818%** |

Doubling the examples increased residual normalized MSE by 82.82%, but the
same 3,801,609 trainable parameters still removed 99.8558% of zero-baseline
error. The run did not reach the deliberately strict 0.0001 normalized-MSE
target. This establishes 128k as the second controlled point on the static-C
memorization scaling curve; it does not measure out-of-sample forecasting.

128k artifacts:

- Plan: `ml/training-plans/next-second-4-layer-memorization-128k-static-c-v1.json`
- Result: `data/training/runs/next-second-4-layer-memorization-128k-static-c-v1/state/result.json`
- Best checkpoint: `data/training/runs/next-second-4-layer-memorization-128k-static-c-v1/checkpoints/best.json`

## Static-centering dataset scaling: 256k

The next matched point doubled the contiguous subset again to 262,144
examples. It spans April 1 through the first 2,944 seconds of April 4. All
model and training settings remained identical to the 65k and 128k runs.

| Deterministic train metric | 65k | 128k | 256k |
| --- | ---: | ---: | ---: |
| Examples | 65,536 | 131,072 | 262,144 |
| Best epoch (zero-based) | 500 | 493 | 488 |
| Normalized MSE | **0.00078884** | 0.00144216 | 0.00340727 |
| MSE skill versus zero | **+99.92112%** | +99.85579% | +99.65927% |
| MSE | **3.965019e-12** | 7.206799e-12 | 1.306867e-11 |
| RMSE | **1.991236e-6** | 2.684548e-6 | 3.615061e-6 |
| Correlation | **0.999613** | 0.999289 | 0.998396 |
| Direction accuracy | 62.6480% | **66.3818%** | 60.0590% |

Doubling from 128k to 256k increased residual normalized MSE by 136.26%.
Across the complete scaling range, the 256k residual is 4.32 times the 65k
residual. The model nevertheless removed 99.6593% of zero-baseline error from
all 262,144 training examples. The increasingly superlinear residual growth
shows that fixed model capacity and/or the fixed optimization budget is
becoming limiting; this remains a memorization diagnostic rather than an
out-of-sample result.

256k artifacts:

- Plan: `ml/training-plans/next-second-4-layer-memorization-256k-static-c-v1.json`
- Result: `data/training/runs/next-second-4-layer-memorization-256k-static-c-v1/state/result.json`
- Best checkpoint: `data/training/runs/next-second-4-layer-memorization-256k-static-c-v1/checkpoints/best.json`

## Static-centering dataset scaling: 512k

The subset was doubled once more to 524,288 contiguous examples, spanning
April 1 through the first 5,888 seconds of April 7. The architecture and full
512-epoch training protocol again remained unchanged.

| Deterministic train metric | 65k | 128k | 256k | 512k |
| --- | ---: | ---: | ---: | ---: |
| Examples | 65,536 | 131,072 | 262,144 | 524,288 |
| Best epoch (zero-based) | 500 | 493 | 488 | 507 |
| Normalized MSE | **0.00078884** | 0.00144216 | 0.00340727 | 0.00750875 |
| MSE skill versus zero | **+99.92112%** | +99.85579% | +99.65927% | +99.24912% |
| MSE | **3.965019e-12** | 7.206799e-12 | 1.306867e-11 | 2.318900e-11 |
| RMSE | **1.991236e-6** | 2.684548e-6 | 3.615061e-6 | 4.815496e-6 |
| Correlation | **0.999613** | 0.999289 | 0.998396 | 0.996242 |
| Direction accuracy | 62.6480% | **66.3818%** | 60.0590% | 59.6582% |

Doubling from 256k to 512k increased residual normalized MSE by 120.37%.
Across the complete range, increasing the dataset eightfold increased the
residual by 9.52 times, approximately an `N^1.08` scaling law over these four
points. The model still removed 99.2491% of zero-baseline training error, but
the residual grows slightly faster than the example count under a fixed model
and fixed 512-epoch optimization budget.

512k artifacts:

- Plan: `ml/training-plans/next-second-4-layer-memorization-512k-static-c-v1.json`
- Result: `data/training/runs/next-second-4-layer-memorization-512k-static-c-v1/state/result.json`
- Best checkpoint: `data/training/runs/next-second-4-layer-memorization-512k-static-c-v1/checkpoints/best.json`

## Static-centering depth ablation: eight layers on 512k

The 524,288-example experiment was repeated with eight width-512 GLU layers
instead of four. Static centering, data, seed, optimizer, batch sizes, fixed
1e-4 learning rate, lack of regularization, and the 512-epoch ceiling were all
matched. The deeper model had 8,004,113 trainable parameters versus 3,801,609.

| Deterministic train metric | Four layers | Eight layers |
| --- | ---: | ---: |
| Stored parameters | 5,898,761 | 12,198,417 |
| Trainable parameters | 3,801,609 | 8,004,113 |
| Best epoch (zero-based) | 507 | 479 |
| Normalized MSE | 0.00750875 | **0.00664631** |
| MSE skill versus zero | +99.24912% | **+99.33537%** |
| MSE | 2.318900e-11 | **2.052554e-11** |
| RMSE | 4.815496e-6 | **4.530512e-6** |
| MAE | 2.598223e-6 | **1.671817e-6** |
| Correlation | 0.996242 | **0.996699** |
| Direction accuracy | **59.6582%** | 54.7632% |

Eight layers reduced residual normalized MSE by 11.49% and MAE by 35.66%.
The improvement is real but modest relative to the 110.55% increase in
trainable parameters. Training was also much noisier under the matched fixed
learning rate: the winning checkpoint appeared late at epoch 479, and nearby
epochs frequently had materially worse MSE. The lower sign score should again
be interpreted cautiously because the one-second target contains many exact
zeros and tiny near-zero errors dominate that metric.

This establishes that additional depth can improve memorization capacity at
512k, but the gain is optimization-inefficient under the unchanged schedule.
It remains a training interpolation result, not evidence of better forecasting
generalization.

Eight-layer artifacts:

- Plan: `ml/training-plans/next-second-8-layer-memorization-512k-static-c-v1.json`
- Result: `data/training/runs/next-second-8-layer-memorization-512k-static-c-v1/state/result.json`
- Best checkpoint: `data/training/runs/next-second-8-layer-memorization-512k-static-c-v1/checkpoints/best.json`

## Eight-layer depth-width quadrant training on 512k

The matched eight-layer experiment was repeated while updating only one fixed
parameter quadrant per epoch. The eight layers were divided into front layers
0--3 and back layers 4--7, and each width-512 hidden representation was divided
into channels 0--255 and 256--511. Their Cartesian product formed four fixed
depth-width quadrants. Epoch `e` updated quadrant `e mod 4`; gradients and
optimizer state were masked for the other three quadrants. Frozen values were
snapshotted at the start of every epoch and verified bit-for-bit unchanged at
its end.

The dense and transform-matrix rows follow their output-channel half. Output
head weights follow their input-channel half and belong to the back-depth
quadrants. The two learned scalar branch radii are split between the two width
halves. Static centering matrices remain frozen independently. This assigned
every trainable scalar exactly once, with quadrant sizes 1,900,548; 1,900,548;
2,101,509; and 2,101,508, totaling all 8,004,113 trainable parameters. Each
quadrant received exactly 128 active epochs during the 512-epoch run.

| Deterministic train metric | Four layers, full update | Eight layers, full update | Eight layers, quadrant update |
| --- | ---: | ---: | ---: |
| Trainable parameters | 3,801,609 | 8,004,113 | 8,004,113 |
| Best epoch (zero-based) | 507 | 479 | 435 |
| Normalized MSE | **0.00750875** | **0.00664631** | 0.00766130 |
| MSE skill versus zero | +99.24912% | **+99.33537%** | +99.23387% |
| MSE | 2.318900e-11 | **2.052554e-11** | 2.366012e-11 |
| RMSE | 4.815496e-6 | **4.530512e-6** | 4.864167e-6 |
| MAE | 2.598223e-6 | **1.671817e-6** | 2.739325e-6 |
| Correlation | 0.996242 | **0.996699** | 0.996204 |
| Direction accuracy | **59.6582%** | 54.7632% | 57.3511% |
| Summed epoch runtime | 2,044.4 s | 3,868.5 s | 4,251.7 s |

Relative to updating all eight layers together, quadrant training left 15.27%
more normalized-MSE residual, 63.85% more MAE, and 0.000495 less correlation.
It did raise direction accuracy by 2.59 percentage points, although that metric
is especially unstable for tiny and zero one-second returns. It also took 9.91%
more wall time because all four quadrants still participate in every forward
and backward pass; masking updates does not reduce dense-compute cost.

The quadrant run came within 2.03% of the four-layer model's normalized-MSE
residual, but did not beat it. Its per-epoch curve showed a strong four-step
sawtooth: updating one quadrant often damaged the coordination established by
the others, especially for the front-depth/high-width quadrant, before later
quadrants repaired it. Under an equal 512-epoch budget this fixed rotating
scheme is therefore less optimization-efficient than joint training. A fair
active-update-count comparison would require 2,048 total epochs so every
quadrant receives 512 updates, but would cost roughly four times as many full
model passes and was not run here.

Quadrant-training artifacts:

- Plan: `ml/training-plans/next-second-8-layer-memorization-512k-static-c-quadrants-v1.json`
- Result: `data/training/runs/next-second-8-layer-memorization-512k-static-c-quadrants-v1/state/result.json`
- Best checkpoint: `data/training/runs/next-second-8-layer-memorization-512k-static-c-quadrants-v1/checkpoints/best.json`
