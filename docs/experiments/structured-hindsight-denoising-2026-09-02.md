# Structured W256: one-pass noisy hindsight

## Control

Basis: `structured-shared-io-feature-process-k1-2-k2-2-i59-o59-w256-m128-p128-f128-pf256-nf256-layer8-selfattn256-productiondense-train256k-batch10240-mse-ema4-v1`.

Two consecutive preceding 1s production-59 states predict the next two 1s
production-59 states. Keep W/PF/NF=256, M/P/F=128, causal single-head layer-8
self-attention with Q/K/V/O=256, direct 59-feature output, standardized feature
MSE, 256,000 training examples, equal train/evaluation batch 10,240, EMA 4,
and the existing optimizer. No separate return head or per-layer recurrent memory.

The previously quoted 7.027% next-1s MSE skill and 0.26709 correlation are
**test**, not validation, metrics. The saved best-correlation policy has
validation skill 6.1783% and correlation 0.248679. Do not use test scores to
select checkpoints in this experiment.

## Change

New plan: `structured-production59-w256-selfattn256-k1-2-k2-2-hindsight-vpnoise-e48-train256k-b10240-ema4-v1`.

For each output step, standardize the true target with training-only statistics
and form `z = sqrt(1-v) * target + sqrt(v) * epsilon`, with independent standard
Gaussian `epsilon`. The scalar `v` is the noise variance fraction, not its
standard deviation. Targets are detached from the conditioning branch.

A single shared GNGLU projects the 59 noisy coordinates plus `v` to NF256.
Add this to the existing layer-7 output before causal attention and before the
next recurrence update. Each output sees its own and earlier noisy tokens,
never later tokens. This adds 110,226 parameters for a total of 4,270,974.
The original numbered layers and shared recurrence remain intact.

Use a cosine noise-variance curriculum:

`v(e) = sin(pi/2 * min(e,48)/48)^2`.

At zero-based epoch 0 the side input is exact hindsight. At epoch 24, signal
and noise each have coefficient `sqrt(0.5)`. At epoch 48 and all later epochs,
the side input is exactly Gaussian: no residual target term is computed.
The endpoint is a configurable plan parameter; this first run uses 48.

The training noise stream is seeded independently by epoch. The run starts
from scratch, with the same initialization seed as the control, for 512 epochs.

## Evaluation and interpretation

Every evaluation, including during the hindsight curriculum, uses `v=1` and
Gaussian noise generated without access to targets. Fixed distinct train,
validation and test noise streams keep the measurement repeatable and do not
advance the training RNG. Noise draws are invariant to evaluation batch
partitioning. Prediction remains one forward pass, without noise averaging.

Headlines and checkpoint selection use pure-noise validation next-1s return
MSE/correlation; complete-feature and both per-step metrics remain available.
The optimization loss remains the original full-feature MSE. Preserve
best-MSE, best-correlation and last checkpoints for final evaluation.

This is a one-pass conditional denoising curriculum, not yet a calibrated
diffusion distribution or an iterative sampler. At the pure-noise endpoint,
MSE favors the conditional mean and may train the model to ignore the noise.
The experiment tests whether the curriculum improves forecasting; it does not
establish distributional calibration.

## Verification

Unit tests cover exact schedule endpoints, detached targets, future-token
causality, finite side-projection gradients, optimizer coverage, and seeded
noise reproducibility. An evaluation regression changes held-out targets and
checks that the predictions remain bit-identical. CUDA smoke checks use the
full 10,240 batch, with both eager and compiled execution before launch.

All 29 structured-model tests passed. The eager/compiled smoke objectives were
1.8235504627 / 1.8235502243. Full training was verified through 8 completed
epochs, with roughly 5 seconds per steady-state epoch, an empty full-run stderr,
and durable last/best-MSE/best-correlation pointers. Both smoke logs and all
launcher/training logs were preserved. Initial curriculum scores are not
evidence of improvement; compare validation after the pure-noise phase trains.
