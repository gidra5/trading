# Return-oracle hard-LayerNorm baseline (2026-07-28)

This records the effective hard-LayerNorm configuration that preceded the
no-hard-LayerNorm comparison. It was intentionally stopped during epoch 150
and preserved at:

```text
data/ml-runs/return-oracle-ce-shrinking-v1-hard-layernorm-baseline-20260728-195427
```

The archive contains `best.pt`, `last.pt`, `training.log`, and the complete
MLP-page status/history. Its best combined validation loss was
`3.5067436467747988` at epoch 132.

## Model and data

- 60 close-only one-minute simple returns.
- Frozen population mean/std computed independently for each return position
  from the selected training split only.
- Hidden widths: 1024, 976, 928, 880, 832, 784, 736, 688, 640, 592, 544,
  496, 448, 400, 352, 304.
- Each hidden layer used independent value and gate-logit affine transforms
  implemented as one fused projection.
- Separate affine hard LayerNorms were applied to the raw value and gate-logit
  branches before `value * sigmoid(gate)`.
- The soft-LayerNorm loss was measured on both raw branches before hard
  normalization.
- 255 raw oracle-policy logits.
- 14,764,239 trainable parameters and 15,055,200 training examples.
- Architecture contract:
  `shrinking-fused-glu-training-input-norm-branch-layer-norm-16-layer-v4`.

## Objective

```text
1.0  * soft-target cross-entropy
1.5  * soft LayerNorm
0.01 * soft weight bound
0.01 * width-normalized distribution unit-sum penalty
0.01 * width-normalized distribution non-negativity penalty
```

Soft LayerNorm used population variance and variance weight `1`. The soft
weight bound used desired magnitude `1`, sharpness `10`, and absolute epsilon
`1e-8`. AdamW weight decay was disabled.

## Training

- Batch size: 8,192; validation batch size: 32,768.
- Initial learning rate: `1e-4`.
- Reduce-on-validation-plateau: factor `0.5`, patience `32`, absolute threshold
  `1e-5`, floor `1e-6`.
- Gradient clipping: `5`.
- BF16 autocast without gradient scaling.
- Compiled dynamic objective without CUDA graphs.
- Four component-loading workers and prefetch factor `2`.
- Activation dropout: `0.05`, intermittent application rate `0.5`; pass and
  layer gates each used probability `sqrt(0.5)`.
- Seed: `1337`.
- Stop rule: more than 1,024 stale validation epochs.

At the time it was preserved, the learning rate was `5e-5`, the latest
completed validation at epoch 148 had total loss `3.5143282907081552`,
cross-entropy `3.4708198310101324`, base KL `0.1299282234889397`, and
probability MSE `2.3781047322227224e-5`.
