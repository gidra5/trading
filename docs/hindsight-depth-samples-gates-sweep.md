# Hindsight depth, samples, and correlation-gate sweep

## Cancelled on 2026-09-03

The user skipped the remaining sweep. Runs 1-7 completed; the 0.90 gate run
was stopped after 58 completed epochs for final evaluation and checkpoint-only
cleanup. Gates 0.80, 0.50, 0.30, and 0.27 will not run. The queue heartbeat was
deleted. Saved plans, logs, results, and evaluations remain available.

Replacement: a fresh 0.99-gate run with **3 total** independently learned
layer13 blocks and S16 affine256 samples. See
`docs/experiments/structured-hindsight-depth3-s16-gate099-2026-09-03.md`.
The queue and handoff instructions below are historical, not active.

Queue: `ml/experiment-queues/hindsight-depth-samples-gates-e256.json`.

Twelve fresh seed-1337 runs, each with a built-in **256 completed epoch** limit.
K1=K2=2, full59 feature input/output, W/PF/NF256 and M/P/F128, layer8 causal
self-attention256, R128 reset for each forecast second, readout hidden512,
equal train/evaluation batch10240, train256000, EMA4. No SAM, dropout,
adversarial training, or weight decay. Original dense dataset retained.

| Order | Change from original S1 / one-layer13 baseline | Parameters |
| --- | --- | ---: |
| 1 | Add2 layer13 blocks: **3 total** | 7,539,438 |
| 2 | Add3 layer13 blocks: **4 total** | 8,353,195 |
| 3 | Add4 layer13 blocks: **5 total** | 9,166,952 |
| 4 | S4, concat236 -> affine256 -> residual stream -> affine236 | 6,638,826 |
| 5 | S16, concat944 -> affine256 -> residual stream -> affine944 | 7,002,030 |
| 6–12 | Original S1 / one-layer13, gates .99/.95/.9/.8/.5/.3/.27 | 5,911,924 each |

Depth positions have independent parameters. Each position shares its weights
across forecast seconds only. They are not separate diffusion iterations. The
sample runs return to **one** layer13, isolating sample count from extra depth.
The user explicitly selected the original one-sample model for softer gates.

## Multi-sample semantics

Generate independent Gaussian noise for S copies of each matching clean
standardized target vector. Concatenate the S vectors, affine-project to256,
then concatenate learned R128 registers. Condition layer2 on W, this stream,
and the scalar variance. Layer13 sees only the updated stream. Affine-decode
the final256 sample coordinates to S full59 vectors. The loss is the mean MSE
over **every sample**, both seconds, and every channel; it is not MSE of their
average. Forecast metrics and the training-only gate use the sample mean.
Neither registers nor corrected samples feed the next forecast second.

Initialization: the first59 embedding coordinates average matching sample
features; the decoder initially repeats those coordinates S times. Extra
embedding rows are random and their initial decoder columns zero, allowing
learning without dead zero/zero pairs. Thus clean identical samples pass
through before the small residual corrections. At nonzero noise, averaging
already reduces independent-noise variance by S; improvement is **not solely
evidence of learned denoising**. Record this distinction in comparisons.

## Evaluation and handoff

Training variance starts0 and increases by .01 only after the training-probe
step1 return correlation reaches the plan's threshold. The first five plans
retain 2^(-1/3600). Gate: fixed65536 training examples, raw weights. Separate
validation reconstruction is target-assisted and never selects checkpoints.
Headline forecasting and best-checkpoint selection always use target-free
Gaussian noise at variance1 with EMA weights and fixed split-specific seeds.

Use the same-thread Codex heartbeat every15 minutes. The trainer's epoch limit,
not heartbeat timing, stops new runs at epoch255. At each terminal boundary,
verify/evaluate best-validation-mse, best-validation-correlation, and last;
preserve all logs, result/status/plan and evaluation artifacts; clean only
checkpoint pointers and unreferenced immutable objects. CUDA-smoke each next
plan at its actual batch/architecture, then launch hidden with --replace-smoke
and compile-mode default. Verify the first complete full epoch and all three
checkpoint pointers. Never silently reduce batch, change architecture, or
advance past failed/incomplete training. After the last run completes,
evaluate and clean it too, report the comparison, and delete the heartbeat.

Predecessor v8 was externally stopped at259 completed epochs (3 beyond the
intended256 boundary during checkpoint verification). It is not one of the
twelve epoch-capped runs. Its three policies and raw-weight reconstruction
were evaluated; checkpoint objects cleaned, all logs/results retained.

Tests cover independent residual positions, register reset, per-sample MSE,
affine gradients, near-identity clean initialization, fixed-noise batching,
and existing target-independent forecast evaluation. The baseline's original
checkpoint state keys and parameter count remain loadable.
