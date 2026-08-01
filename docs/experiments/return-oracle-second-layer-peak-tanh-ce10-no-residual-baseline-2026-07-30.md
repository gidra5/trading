# Return-oracle second-layer-peak tanh CE10 no-residual baseline — 2026-07-30

## Preserved run

```text
data/training/runs/return-oracle-ce-shrinking-v1-secondlayerpeak-tanh-ce10-noresidual-baseline-20260730-174907
```

The run was stopped intentionally while healthy to test learned projections
of the shared model input into every hidden-layer output. Its checkpoints,
status, plan snapshot, and append-only training log remain in the archived
directory.

## Configuration

- Inputs: 60 close-only simple returns standardized with frozen training-split
  per-position population mean and standard deviation.
- Hidden widths: `60, 1024, 896, 768, 640, 511, 383, 255`.
- Output: 255 raw base-action logits trained against the stored soft oracle
  distribution.
- Parameters: 18,957,668.
- Residual paths: none.
- Branch normalization:
  `A C h tanh(r/s) / r + b`, independently for the value and gate branches.
- Learnable centering: separate full value/gate `C` matrices per layer,
  initialized to `I - 11^T/d`, with idempotence and symmetry losses.
- Radius initialization: `s = sqrt(1e-5)` with positive softplus
  parameterization.
- Activation: GLU followed by intermittent 5% dropout with application rate
  0.5.
- Objective: soft-target CE weight 10, independently gated skew reverse KL and
  one-sided entropy sharpness with expected weight 0.02 each, soft LayerNorm
  weight 1, centering constraints weight 1 each, and soft weight-bound weight
  0.01.
- Optimizer: hybrid Muon/AdamW, three Newton–Schulz iterations, BF16 compiled
  execution, two source segments per optimizer update, initial learning rate
  `1e-4`.
- Examples: 25,423,200 training, 15,418,800 non-overlapping validation, and
  the final 1,000,000 chronological test examples.

## Result when archived

The run was archived during epoch 46 with finite gradients and no overflow.
Epoch 40 held the best combined validation objective:

- combined validation loss: `35.12662653334411`;
- raw cross-entropy: `3.4928166596321146`;
- forward `KL(Q || P)`: `0.15192503447572872`;
- skew reverse KL: `0.2766313542140691`;
- probability MSE: `2.8871240629566882e-5`;
- oracle entropy: `3.3408916048877795`;
- predicted entropy: `3.54910186921885`;
- soft-LayerNorm penalty: `0.1899455271116796`; and
- learning rate: `1e-4`.

Epoch 40 took `21.969` seconds, including `19.578` seconds of training and
`2.360` seconds of validation.
