# Codex task: continue BTCUSDT Hawkes study locally

Read `SPEC.md` completely before changing code.

## Main objective

Build a reproducible local research pipeline for BTCUSDT buy/sell aggregate-trade arrivals and test whether the observed dynamics are best explained by:

1. a flexible 2-channel multiscale Hawkes process,
2. nonlinear cross-scale state-dependent excitation,
3. a genuine third **latent Hawkes event channel** `L`,
4. both nonlinear excitation and latent `L`,
5. optionally, continuous stochastic latent excitation in addition to `L`.

## Non-negotiable requirements

- Chronological split: 60% train / 20% validation / 20% untouched test.
- Architecture, state count, timescales, regularization and stopping are selected using train+validation only.
- Preserve equal-ms event ordering in event-level code.
- Fit independent source→target Hawkes kernels; do not assume shared scale counts.
- Optimize both model order `K_ij` and continuous timescales `tau_ij,k`.
- Refit all existing modes after adding/removing a mode; use forward addition + backward deletion, not one-shot greedy frozen scales.
- Explicitly enforce/check stability for additive linear Hawkes models.
- Do not call residual-filtered pseudo-events a latent Hawkes channel. A true `L` model must include and infer `B→L`, `S→L`, `L→L`, `L→B`, and `L→S` interactions in a generative likelihood/inference scheme.
- Keep point-process/state-space semantics clear: deterministic Hawkes traces, latent event counts, and continuous latent diffusion states are different objects.
- Report failures, non-identifiability and degenerate solutions rather than hiding them.

## First implementation milestone

Use 1-second binned B/S counts for architecture search because it is computationally manageable and aligns with previous tests. Implement:

- `M0`: optimized 2-channel exponential-mixture Hawkes.
- `M2`: M0 + slow-state modulation of faster event excitation amplitudes.
- `ML`: true generative 3-channel Hawkes with latent count channel L.
- `M2L`: M2 + true latent channel L.

Then evaluate open-loop forecasting at 1s, 5s, 15s, 60s, 300s, 900s, and 3600s.

After architecture selection, optionally repeat the final comparison on event-level timestamps.

## Preferred outputs

- `results/model_selection.csv`
- `results/final_test_metrics.csv`
- `results/horizon_metrics.csv`
- `results/fitted_scales.csv`
- `results/stability.csv`
- `results/latent_diagnostics.csv`
- serialized model parameters
- a short Markdown report with plots and conclusions
- tests for causality/no leakage, trace recursion, stability calculation, and open-loop forecast recursion

Reference scripts in `experiments/` reproduce the recent screening logic and are useful only as sanity checks.
