# Independent readout positions, shared across forecast seconds

Plan: `structured-production59-w256-selfattn256-k1-2-k2-2-hindsight-residual-r128-s1-layer2h512x2-independent-no13-nocarry-corrgate-train256k-b10240-ema4-v7`.

Correction to v6: the two layer-(2) passes must have independent parameters.
Sharing applies across forecast steps, NOT between readout positions.

For each predicted second s:

```
z0 = concat(layer12, noisy_features[s])
z1 = z0 + A(concat(W[s], z0, variance))
z2 = z1 + B(concat(W[s], z1, variance))
output[s] = z2[128:]
```

A is `layer2`; B is `layer2_refinements[0]`. They own disjoint parameters and
are independently initialized and trained. Each second uses the SAME A and
B. Full register+sample state and gradients flow from A to B, but neither
registers nor samples carry to the next predicted second. W and variance
stay fixed within the two passes. No layer (13) is added.

The shared-pass execution path was replaced after v6 finalization; the new
multi-pass configuration explicitly requires
`residualParameterSharing: across-forecast-steps-only`.

## Capacity and controls

Two GNGLU readouts, each 444 -> hidden512 -> 187, have 1,076,925 parameters
apiece. Total parameters: **6,175,092** (previously 5,098,167). Additional
readouts initialize after the existing modules to preserve their seeded
initialization. Both output projections start at identity*0.001, biases zero.

Fresh seed 1337, K1=K2=2, I=O=59, S=1 full-feature sample, R=128 learned
initial registers. Same W/PF/NF256, M/P/F128 and Q/K/V/O256 attention; full
standardized feature MSE, 256,000 training examples, equal train/evaluation
batch 10,240, EMA4, 512 epochs, no regularization. Variance starts at zero,
increasing by 0.01 only when training reconstruction next-1s correlation
reaches 2^(-1/3600). Pure-noise forecasts remain separate from target-assisted
reconstruction diagnostics.

## Superseded run

V6's exact Node/Python processes were stopped after zero-based epoch 45
(46 completed epochs), following complete-checkpoint verification. The
best-validation-MSE, best-validation-correlation, and last policies were
evaluated and persisted before cleanup:

| Policy | Zero-based epoch | Validation forecast next-1s MSE skill | Correlation |
| --- | ---: | ---: | ---: |
| Best validation MSE | 45 | -27.872373% | -0.0005305285 |
| Best validation correlation | 0 | -58.391234% | 0.0011230298 |
| Last | 45 | -27.872373% | -0.0005305285 |

At variance 0.01, the raw checkpoint's target-assisted reconstruction
correlation was 0.9981879462 on training and 0.9990128219 on validation;
these are not forecast scores. The offline diagnostic was saved as well.

Cleanup removed three pointers and three unreferenced immutable checkpoint
objects. Logs/results/status/snapshots/evaluations/diagnostics are retained;
SHA-256 checks verified six key preserved files unchanged.

## Verification

All 39 structured-model tests passed. They verify disjoint A/B parameters,
independent initialization with the existing model unchanged, A/B reuse
across forecast steps, correct Muon/Adam parameter routing without duplicates,
exact manual-unroll outputs/gradients, A-register gradients reaching B,
no cross-second state/noise leakage, near-identity initialization, exact
parameter counts, and checkpoint restoration with both readouts.

Compiled CUDA smoke at full batch 10,240 passed with 6,175,092 parameters,
finite gradients, both per-step rows, and online feature MSE 7.016468e-7.
At variance zero, the training reconstruction correlation was 0.9998884891
and held-out reconstruction correlation was 0.9999871470. The gate correctly
selected variance 0.01. The only stderr output was PyTorch's benign small-GPU
autotuning warning.

The full run started fresh at epoch 0 and was verified through six completed
epochs with both independent readouts, 6,175,092 parameters, equal batches,
and two per-step rows. Initial warm epochs take about 6.9 seconds, including
diagnostics (3.84-3.86 seconds optimization). The gate advanced to variance
0.01 then correctly held. Last/best-MSE/best-correlation pointers exist,
all three exact launcher/worker processes are alive, full stderr is empty,
and the dashboard lists the new run as training.
