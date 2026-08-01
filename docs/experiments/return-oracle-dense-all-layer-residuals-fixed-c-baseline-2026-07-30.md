# Dense all-layer residuals with fixed C baseline

Recorded on 2026-07-30 before reducing every hidden layer to width 256.

## Preserved run

The complete run, logs, and best/last checkpoints are in:

`data/training/runs/return-oracle-ce-shrinking-v1-dense-all-layer-residuals-fixed-c-baseline-20260730-232337`

Architecture contract:

`second-layer-peak60to1024to255-fused-glu-shared-layer-centering-tanh-learned-radius-full-a-post-bias-full-normalized-global-input-glu-dense-all-prior-layer-residuals-fixed-c-path-shared-value-gate-a-v37`

## Configuration

- Hidden widths: `60, 1024, 896, 768, 640, 511, 383, 255`.
- Original-input residual at every layer plus 21 additive GLU residuals from
  every non-adjacent earlier hidden layer.
- `C = I - 11^T/d` fixed and shared across all paths into a target layer.
- Learnable `A_v` and `A_g` separately shared across all paths into a target.
- Path-specific projections, biases, tanh radii, and post-`A` biases.
- 24,604,274 trainable parameters.
- Cross-entropy weight `10`; reverse KL and entropy-sharpness expected weights
  `0.1`; soft-LayerNorm weight `1`; gradient clip `100`.

## Observed result

The run reached epoch 10 before preservation. Its best completed validation
objective was `37.603173643032946` at epoch 6.

At epoch 6, validation metrics included:

- cross entropy: `3.6391292340895185`
- base KL divergence: `0.29823760950806094`
- reverse KL divergence: `0.5072435255143065`
- predicted entropy: `3.6564211585993194`
- oracle entropy: `3.3408916048877795`
- soft-LayerNorm loss: `1.1395370682356598`
- centering constraint: exactly `0` and not computed

Completed epochs generally took about 62–70 seconds including validation.
Freezing `C` did not materially impair the strong early behavior of the
learnable-`C` dense-residual baseline.
