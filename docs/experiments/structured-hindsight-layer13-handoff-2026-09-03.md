# Replace second conditioned readout with stream-only layer (13)

Prepared successor: `structured-production59-w256-selfattn256-k1-2-k2-2-hindsight-residual-r128-s1-layer2h512-layer13h512-nocarry-corrgate-train256k-b10240-ema4-v8`.

The user requested the switch after 128 completed epochs of the independent
two-layer-(2) v7 run. The handoff uses the persisted completed epoch and
checkpoint, not elapsed time. A zero-based epoch >=127 must exist in both
the complete log and a loadable last checkpoint before stopping the current
trainer. It remains unchanged and running until that threshold.

## New readout

```
z0 = concat(learned_R128, matching_noisy_O59)
z1 = z0 + layer2(concat(expected_W256, z0, variance))
z2 = z1 + layer13(z1)
output = z2[128:]
```

Layer (2): 444 -> hidden512 -> 187.
Layer (13): 187 -> hidden512 -> 187; NO direct feature embedding, variance,
or new hindsight input. It can of course use information already encoded
in z1. The two blocks have independent parameters, each shared across
forecast seconds. Their full residual stream and gradients connect within
an output, then reset before the next second. No second layer-(2) module.

Total: **5,911,924 parameters**, versus 6,175,092 for two conditioned readouts.
Keep K1=K2=2, full 59-feature MSE and hindsight sample, R128, all backbone and
attention widths, batch/evaluation batch 10,240, 256k examples, seed1337,
EMA4 and no regularization. Keep both readout hidden widths 512 and small
0.001 residual-projection initialization. Start fresh at variance zero with
the unchanged correlation-gated +0.01 noise curriculum, not a weight transplant.
Pure-noise forecasting remains separate from reconstruction diagnostics.

## Preparation and handoff

All 41 structured-model tests passed. Added coverage checks exact layer13
stream-only arguments and dimensions, independent parameters shared across
forecast seconds, no cross-second hindsight/state leakage, gradient agreement
with explicit unrolling, correct optimizer groups, checkpoint reload,
near-passthrough, and exact 5,911,924 parameter count. The new CUDA smoke and
full run are intentionally deferred until the current run reaches threshold.

Handoff heartbeat (deleted after successful verification):
`replace-second-layer2-with-layer13-at-epoch128`, every minute.
It stays quiet below the threshold, verifies complete checkpoint/log state,
stops only the exact trainer processes, evaluates best-MSE/best-correlation/last,
preserves the reconstruction diagnostic, and cleans only unreferenced weights.
All logs/results/snapshots/status/evaluations are retained. It then runs the
full-batch CUDA smoke, starts fresh full training, verifies a full epoch and
all checkpoint pointers, and deletes itself. It must report rather than
silently advance if the current run dies early or the smoke fails.

## Handoff execution

The monitor observed the threshold and verified a complete checkpoint at
zero-based epoch 130. Checkpoint inspection through the Node wrapper included
post-process storage maintenance; by the exact-process shutdown, the last
completed epoch was 132 (133 completed epochs, five beyond the requested
threshold). No earlier checkpoint or partial epoch was used for finalization.

V7's three policy evaluations were persisted:

| Policy | Zero-based epoch | Validation forecast next-1s MSE skill | Correlation |
| --- | ---: | ---: | ---: |
| Best validation MSE | 132 | -23.943247% | 0.0010089414 |
| Best validation correlation | 103 | -24.315157% | 0.0011866904 |
| Last | 132 | -23.943247% | 0.0010089414 |

At variance 0.01, the separate raw-weight reconstruction diagnostic had
training correlation 0.9989152526 and held-out correlation 0.9993056287.
These target-assisted values are NOT pure-noise forecast accuracy. The
diagnostic was saved before removing checkpoints.

Cleanup removed three checkpoint pointers and three now-unreferenced objects;
hashes of six preserved log/status/result/evaluation/diagnostic/plan files
were unchanged. Training, smoke and launcher logs remain.

The full-batch compiled CUDA smoke passed with 5,911,924 parameters, finite
gradients and both per-step rows. Initial feature MSE was 5.911623e-7;
zero-noise training reconstruction correlation was 0.9998996283 and held-out
reconstruction correlation was 0.9999817985. The gate selected variance 0.01.
Only the benign small-GPU autotune warning appeared on smoke stderr.

The full run started fresh at epoch 0 and was verified through at least eight
completed epochs. Warm epochs take about 6.88-6.89 seconds including diagnostics
(3.81-3.83 seconds optimization). The intended one layer-(2) plus layer-(13)
configuration, 5,911,924 parameters, equal 10,240 batches and both per-step rows
are persisted. The gate advanced from variance 0 to 0.01, then held correctly.

All three live launcher/worker processes were verified, full stderr was empty,
and the dashboard reported the successor as training. Last/best-MSE/best-correlation
immutable checkpoints were loaded successfully and all model tensors were
finite; their layer13 projection had 187 stream-only inputs, with no second
conditioned layer2 module. The heartbeat was deleted after these checks.
