# Dense all-layer residuals with learnable C baseline

Recorded on 2026-07-30 before the fixed-centering ablation.

## Preserved run

The complete run directory, logs, status, and best/last checkpoints are in:

`data/ml-runs/return-oracle-ce-shrinking-v1-dense-all-layer-residuals-learnable-c-baseline-20260730-224319`

Architecture contract:

`second-layer-peak60to1024to255-fused-glu-shared-layer-centering-tanh-learned-radius-full-a-post-bias-full-normalized-global-input-glu-dense-all-prior-layer-residuals-shared-c-path-shared-value-gate-a-v36`

## Configuration

- 60 training-position-standardized close-only simple returns.
- Eight hidden widths: `60, 1024, 896, 768, 640, 511, 383, 255`.
- A main GLU and original-input residual GLU at every layer.
- Twenty-one additional additive GLU paths from every non-adjacent earlier
  hidden layer to every later target layer.
- One learnable full `C` shared across every incoming value/gate branch at
  each target layer.
- One full value `A_v` and one separate gate `A_g` shared across all incoming
  paths at each target layer.
- Path-specific projection matrices, projection biases, learned tanh radii,
  and post-`A` biases.
- 27,931,525 trainable parameters.
- 33,195,600 training, 15,418,800 validation, and 1,000,000 test examples.
- Hybrid Muon/AdamW, learning rate `1e-4`, three Newton-Schulz steps,
  gradient-norm clipping at `100`.
- Cross-entropy weight `10`; reverse KL and entropy-sharpness expected weights
  `0.1` each with independent 10% inverse-probability-corrected gates.
- Soft-LayerNorm weight `1`; centering idempotence and symmetry weights `1`.

## Observed result

The run reached epoch 9 before preservation. Its best completed validation
objective was `37.669299024682694` at epoch 6.

At best epoch 6, validation metrics included:

- cross entropy: `3.6485105894058907`
- base KL divergence: `0.3076189654854701`
- reverse KL divergence: `0.5091750177822463`
- predicted entropy: `3.6943342886654653`
- oracle entropy: `3.3408916048877795`
- soft-LayerNorm loss: `1.1082012644305064`
- centering constraint: `0.00017153745284304023`

After the compile-heavy first epoch, completed epochs took approximately
62–74 seconds including validation. This configuration showed strong early
learning and is the baseline for testing the same dense residual graph with
`C` frozen at the canonical centering projector.
