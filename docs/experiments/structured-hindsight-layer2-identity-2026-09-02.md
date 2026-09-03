# Final-layer-only hindsight with near-identity initialization

New fresh run:
`structured-production59-w256-selfattn256-k1-2-k2-2-hindsight-layer2-identity-corrgate-step001-train256k-b10240-ema4-v3`.

Keep the W256 structured self-attention model, two preceding 1s production-59
states, two next 1s feature states, standardized feature MSE, equal batch
10,240, EMA 4 and the correlation gate from the predecessor. The gate remains
next-1s return reconstruction correlation >= `2**(-1/3600)` on the fixed
65,536-example training probe. Increase variance by 0.01 for the next epoch
only when the gate passes; cap at 1, with no timed fallback.

## Readout change

Remove the early NF-state hindsight projection completely. No noisy future
feature value or noise variance enters the feature encoder, attention, or
market/prefix/feature-distribution recurrent states. At each output step, only
the shared final layer (2) receives:

`concat(expected W256 embedding, matching noisy O59 feature row, scalar variance)`.

The readout computes, in standardized feature coordinates:

`prediction = a * noisy_features + b * GNGLU(concatenated_readout_inputs)`.

Both `a` and `b` are trainable per-feature vectors, initially 1 and 0.001.
The identity bypass preserves the supplied values' magnitudes directly; the
small correction keeps gradients connected through the model at initialization.
Neither path is frozen, and the readout can learn to suppress/correct noise.
The complete model has 4,221,196 parameters (49,778 fewer than its predecessor).

This is the last architectural layer of **each** predicted step, not only the
second time step. Readout outputs are not fed into the recurrent states.
Changing a step's noisy value affects only that step's final readout. Changing
the scalar variance may affect both readouts, but never the latent trajectories.

## Evaluation and verification

Normal forecast charts and checkpoint selection still use pure Gaussian input
at layer (2), never target-assisted validation. The separate current-noise
training reconstruction probe controls the gate. Identity initialization does
not establish forecasting accuracy at pure noise.

Tests verify near-identity initialization, finite nonzero correction/backbone
gradients, complete optimizer routing, unchanged latent trajectories under
changed hindsight/variance, independent per-step readouts, and the existing
gate/checkpoint/evaluation contracts. The old NF-injection implementation is
replaced rather than retained as a second runtime path.

The predecessor was stopped after zero-based epoch 180 and evaluated for
best-validation-MSE, best-validation-correlation, and last before replacement.
Its last training reconstruction correlation was 0.9989701 at variance zero.
Its logs and completed evaluations are preserved during checkpoint cleanup.

Verification: all 34 structured-model tests passed, including restoration of
the new readout for stopped evaluation. The full-batch compiled CUDA smoke
had standardized feature MSE 3.1728e-7 and training-probe next-1s correlation
0.9999991753 at variance zero. It correctly selected variance 0.01 for the next
epoch. The only smoke stderr output was PyTorch's benign GPU-autotuning warning.

The fresh full run started from epoch 0 with 4,221,196 trainable parameters,
equal train/evaluation batches of 10,240, and zero initial noise variance.
After its first 25-update epoch, the 65,536-example probe returned next-1s
correlation 0.9999999881 and advanced variance to 0.01. The next epoch used
0.01 and held it because correlation 0.9890804 was below the gate. All three
last/best-MSE/best-correlation checkpoint pointers are durable, the dashboard
API reports the run as training, and full-run launcher stderr is empty.
