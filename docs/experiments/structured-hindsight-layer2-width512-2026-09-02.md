# Widen only the final hindsight correction layer

Plan:
`structured-production59-w256-selfattn256-k1-2-k2-2-hindsight-layer2-h512-identity-corrgate-step001-train256k-b10240-ema4-v4`.

Predecessor:
`structured-production59-w256-selfattn256-k1-2-k2-2-hindsight-layer2-identity-corrgate-step001-train256k-b10240-ema4-v3`.

## Controlled change

Fresh weights. Set `architecture.hindsightConditioning.correctionHiddenWidth`
to 512 instead of the default 187. The layer-2 correction remains a single
GNGLU with 316 inputs (W256, noisy O59 and scalar variance), now with 512
hidden coordinates, and 59 outputs. No extra layers or recurrent state.

The learnable bypass still initializes to 1 and the correction gain to 0.001:
`prediction = a * noisy_features + b * GNGLU(W, noisy_features, variance)`.
Hindsight and variance never enter the structured recurrence or attention.

Everything else is unchanged: K1=K2=2, production-59 direct feature MSE,
W/PF/NF=256, M/P/F=128, causal attention Q/K/V/O=256, 256,000 training
examples, equal train/evaluation batch 10,240, optimizer and seed 1337,
EMA half-life 4, 512 epochs, no SAM/dropout/adversarial regularization.

| Correction hidden width | Layer-2 parameters | Total parameters |
| --- | ---: | ---: |
| 187 (predecessor) | 200,082 | 4,221,196 |
| 512 (new) | 880,307 | 4,901,421 |

Noise starts at variance zero. Only the fixed 65,536-example training probe,
using raw weights, controls advancement: next-1s correlation must reach
`2**(-1/3600) = 0.9998074776513175` before variance increases by 0.01.

## Diagnostic separation

Pure-noise forecasting and checkpoint selection remain unchanged: variance 1,
no target contribution, EMA weights. New epoch logs also include
`hindsight.validationReconstruction`: raw-weight, target-assisted validation
at the current training variance, using independent seed 982451654. This
diagnostic never affects the loss, curriculum gate, or checkpoint selection.
Both predicted steps have diagnostic rows. Test data is not used here.

`hindsight.optimizationSeconds` records training-only time;
`hindsight.validationReconstructionSeconds` isolates the extra diagnostic
cost so a shorter epoch count is not mistaken for faster wall-clock learning.

The reusable `ml/evaluate_hindsight_reconstruction.py` can evaluate the last
raw checkpoint without replacing forecast results or changing training state.
It preserves an existing diagnostic instead of silently overwriting it.

## Predecessor completion and comparison baseline

The predecessor completed all 512 epochs naturally (last zero-based epoch
511; training/epoch-evaluation elapsed time 2,828.843 s). It remained at
variance 0.01. All three final forecast policies were already persisted:

| Policy | Epoch (zero-based) | Validation next-1s correlation | Validation MSE skill |
| --- | ---: | ---: | ---: |
| Best MSE | 509 | 0.0034859262 | -22.911176% |
| Best correlation | 502 | 0.0035105877 | -22.955898% |
| Last | 511 | 0.0034650389 | -22.932830% |

Before checkpoint cleanup, the last raw checkpoint was also measured at
variance 0.01 on fixed train/validation probes. These are target-assisted
reconstruction results, **not pure-noise forecasting accuracy**:

| Probe | Next-1s correlation | Next-1s normalized MSE | MSE skill |
| --- | ---: | ---: | ---: |
| Training (65,536) | 0.9988036809 | 0.0010898359 | 99.760499% |
| Validation (65,534) | 0.9992322690 | 0.0026163219 | 99.846481% |

Saved to the predecessor's `state/hindsight-reconstruction.json`. The
training value exactly reproduced the last logged gate probe. Correlation
and skill depend on target variance, so do not infer a generalization gain
merely because validation correlation exceeds training correlation.

## Verification

All 36 structured-model tests pass. They cover exact parameter counts,
unchanged shapes outside layer 2, near-identity initialization, finite
nonzero gradients, correct optimizer routing, causal isolation of hindsight,
checkpoint reconstruction with the width override, and held-out diagnostic
isolation from the gate, model parameters, and training random-number stream.

The full-batch compiled CUDA smoke completed one optimizer step with exactly
4,901,421 parameters. At variance zero its online feature MSE was 3.3298e-7;
next-1s reconstruction correlation was 0.9999992333 on training and
0.9999999456 on validation. The gate advanced to variance 0.01 using only the
training result. The sole stderr entry was PyTorch's benign small-GPU
autotuning warning. The plan's training settings exactly match the predecessor;
its only architecture difference is `correctionHiddenWidth=512`.

Cleanup removed the predecessor's three checkpoint pointers and three
unreferenced immutable checkpoint objects. Training-log SHA-256 was unchanged;
all smoke/launcher logs, final evaluations, reconstruction diagnostics,
status, result, and saved plan remain.

The full run started fresh at epoch 0. Its first complete epoch (25 optimizer
updates) reached training reconstruction correlation 0.9999999908 at variance
zero, advancing to 0.01. The next epoch used 0.01 and held it, with training
correlation 0.9890808104 and held-out reconstruction correlation 0.9970364343.
At verification, five full epochs had completed; last/best-MSE/best-correlation
pointers existed, the exact Node/Python launcher and workers were alive, and
full-run stderr was empty. The dashboard API recognized the new training run.
Warm epochs initially took 6.27-6.58 s total, including 0.63-0.64 s of extra
held-out reconstruction evaluation and 3.45-3.58 s of optimization.
