# Combined hindsight depth, samples, and 0.99 gate

Requested on 2026-09-03, replacing the unfinished depth/samples/gates sweep.

Plan: `ml/training-plans/structured-production59-w256-hindsight-r128-depth13x3-samples16-affine256-gate0p99-k2x2-h512-train256k-b10240-e256-ema4-v1.json`.

Fresh seed-1337 training, not a warm start from the earlier 0.99 checkpoint.
The original 0.99-gate setup is combined with three **total** independently
learned layer13 blocks and sixteen independent noisy full59 feature samples.

## Configuration

- K1=K2=2: two preceding 1s production59 states, two forecast seconds.
- W/PF/NF=256, M/P/F=128, layer8 causal self-attention with Q/K/V/O=256.
- R128 learned registers reset separately at each forecast second.
- S16 full59 samples: concatenate944 -> learned affine256 -> residual stream.
- One feature-conditioned layer2, then three stream-only layer13 residuals;
  each residual block has hidden width512 and independent parameters.
- Weights share across forecast seconds only. No register/sample state carries
  between forecast seconds; there are no extra diffusion iterations.
- Affine unembedding256 -> 944, yielding sixteen complete59 predictions.
- Training minimizes mean standardized MSE over every sample, step and feature.
  Reported forecasts and the curriculum gate use the arithmetic sample mean.
- Variance starts0 and advances by0.01 only after training-probe next1s return
  correlation reaches0.99. Probe: 65,536 fixed training examples, raw weights.
- Target-assisted validation reconstruction is diagnostic only. Forecast
  metrics and checkpoint selection use target-independent Gaussian noise at
  variance1, EMA weights, and fixed split-specific noise seeds.
- 256,000 training examples, original dense dataset, 256 epochs; equal training
  and evaluation batch10,240; EMA4; no SAM, dropout, adversarial loss or decay.
- Exactly **9,235,122 trainable parameters**, validated by model construction.

Sample averaging itself reduces independent Gaussian variance; reconstruction
gains must not be attributed entirely to learned denoising.

## Verification

All44 structured-model unit tests passed, including combined S16/depth3
coverage of independent blocks, reset registers, finite nonzero gradients,
per-sample loss, and clean-copy initialization after zeroing residuals.
Full-batch compiled CUDA smoke passed with equal train/evaluation batch10,240.
Fresh full training was launched hidden with `--replace-smoke` and compile-mode
default. Verified `startEpoch=0`, 256 epochs, all requested settings, three live
launcher/worker processes, a complete first epoch (25 optimizer steps), both
per-step metric rows, and all three finite immutable checkpoint policies.
The first full epoch took17.594s including startup work and passed the gate at
variance0 with raw training-probe correlation0.9999669; the next variance is0.01.
Only the benign PyTorch insufficient-SMs autotune warning appeared; no OOM or
traceback. Smoke and full-launch logs are preserved. No further sweep is queued.

## Previous queue closure

Deleted its heartbeat; gates0.80/0.50/0.30/0.27 are canceled. Their saved plans
remain for reference. The 0.90 run stopped after58 completed epochs (zero-based
epoch57), then all three policies were evaluated on full train/validation/test.

| Checkpoint | Completed epoch | Validation next1s MSE skill | Validation next1s correlation |
| --- | ---: | ---: | ---: |
| Best MSE | 49 | -7.2593% | 0.02282 |
| Best correlation | 58 | -7.7236% | 0.02529 |
| Last | 58 | -7.7236% | 0.02529 |

Last raw-weight reconstruction at variance0.57: train correlation0.93174,
validation correlation0.94686. These are target-assisted, not forecast scores.

Removed three checkpoint pointers and three unreferenced immutable objects.
All14 remaining run files passed unchanged SHA256 verification, including
training/smoke/launcher logs, results, status, plan and evaluation artifacts.
