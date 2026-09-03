# Two shared layer-(2) residual passes per output

Plan: `structured-production59-w256-selfattn256-k1-2-k2-2-hindsight-residual-r128-s1-layer2h512x2-no13-nocarry-corrgate-train256k-b10240-ema4-v6`.

Repeat the same numbered layer (2) twice before emitting each predicted
second's feature vector. Both calls share parameters, as required by the
numbered-layer convention. This adds computation, not parameters: the model
still has 5,098,167 trainable parameters. K1=K2=2 remains unchanged.

For each predicted second s independently:

```
z0 = concat(layer12, noisy_features[s])   # R128 + S1*O59 = 187
z1 = z0 + layer2(concat(W[s], z0, v))    # 444 -> hidden512 -> 187
z2 = z1 + layer2(concat(W[s], z1, v))    # same layer2, W, and v
output[s] = z2[128:]                     # full 59-feature prediction
```

The first pass's updated registers AND sample enter the second pass, with
full backpropagation through both. Registers are now useful within each
output: the first update can affect the second sample correction. They still
reset to the learned layer-(12) initializer before the next predicted second.
No layer (13), per-layer recurrent memory, extra diffusion noise draw,
noise-level change, or feedback into structured/attention states is added.

`hindsightConditioning.residualPasses` controls this count (default one).
Model traces index residuals and updated streams by output step, then pass.
Epoch logs distinguish one overall model/denoising forward from two
`layer2ResidualPasses` inside the readout.

## Controlled setup

Fresh initialization, seed 1337; same 0.001 residual projection initialization,
learned R128 zero-initialized registers, one complete noisy 59-feature sample,
W/PF/NF256, M/P/F128, attention Q/K/V/O256, and hidden512 readout. Full-feature
standardized MSE over both output steps. 256,000 training examples, equal
train/evaluation batch 10,240, EMA half-life 4, 512 epochs. No regularization.

Start at variance 0; add 0.01 only when the training-only raw-weight next-1s
reconstruction correlation reaches 2^(-1/3600), or 0.9998074776513175.
Forecast evaluation/checkpoint selection still uses target-independent
variance-1 Gaussian samples. Current-noise reconstruction diagnostics are
separate and must not be reported as forecasting performance.

## Predecessor completion

The one-pass residual-register v5 run completed all 512 epochs naturally.
All three evaluation policies were persisted before cleanup:

| Policy | Zero-based epoch | Validation next-1s forecast MSE skill | Correlation |
| --- | ---: | ---: | ---: |
| Best validation MSE | 510 | -18.134846% | 0.0048083320 |
| Best validation correlation | 511 | -18.137666% | 0.0048255546 |
| Last | 511 | -18.137666% | 0.0048255546 |

Its curriculum remained at variance 0.01. The final raw-checkpoint
reconstruction diagnostic reproduced the logged training correlation
0.9990815332 and held-out correlation 0.9993708747. These use target-assisted
hindsight and are NOT forecast scores.

Cleanup removed three checkpoint pointers and three now-unreferenced
immutable objects. Training logs, snapshots, status, results, all policy
evaluations, launcher/smoke logs, and reconstruction diagnostics were kept.
SHA-256 checks verified six key preserved files were unchanged.

## Verification

All 39 structured-model tests passed, including explicit two-pass unrolling
and matching gradients, shared parameter shapes/count, gradient flow through
updated registers, exact no-carry boundaries, near-passthrough initialization,
literal residual additions, checkpoint reload, and noise/target isolation.

The compiled CUDA smoke completed one optimizer step at the full 10,240
train/evaluation batch, with finite loss and gradients. Initial standardized
feature MSE was 1.244133e-6. At zero noise, the training reconstruction
correlation was 0.9998909952 and independent validation reconstruction was
0.9999862378; the gate selected variance 0.01 for the next epoch. Both
per-step rows were present. Stderr contained only PyTorch's benign warning
about insufficient SMs for max-autotune GEMM.

The full run started fresh at epoch 0 and was verified through seven completed
epochs. Training-start reports 5,098,167 parameters, two shared residual
passes, and equal 10,240 batches. Both per-step rows and the three checkpoint
pointers are present; the exact Node/Python parent/worker processes are alive,
full-run stderr is empty, and the dashboard lists the run as training.
Initial warm epochs take about 6.69-6.70 seconds, including diagnostics
(about 3.70 seconds for optimization). The curriculum advanced from variance
0 to 0.01, then correctly held below the required reconstruction correlation.
