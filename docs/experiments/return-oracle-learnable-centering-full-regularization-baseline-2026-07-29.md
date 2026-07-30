# Learnable centering with full activation regularization baseline

This record preserves the successful learnable-centering configuration before
soft LayerNorm and both hidden-distribution penalties were disabled. It was
intentionally stopped during epoch 25 on 2026-07-29 after completing epoch 24,
so the full run and its best checkpoint could be retained for comparison with
the activation-regularizer ablation.

The complete run is archived at:

```text
data/ml-runs/return-oracle-ce-shrinking-v1-learnable-c-full-regularization-baseline-20260729-144909
```

`best.pt`, `last.pt`, `status.json`, and the complete append-only
`training.log` are present in that directory.

## Data and network contract

- Inputs are exactly 60 adjacent simple returns derived only from completed
  one-minute close values.
- Every return coordinate uses its own training-split population mean and
  standard deviation. Validation and test reuse those frozen statistics.
- Targets are the stored raw 255-cell minute-oracle distributions aligned to
  the same close path.
- The 16 effective hidden widths shrink from 512 to 272 in steps of 16.
- Each fused projection produces raw value and gate branches.
- One full learnable centering matrix `C` is shared by the branches at each
  layer and initialized to `I - 11^T / d`.
- Each centered branch is divided by its per-example population RMS and gets
  its own learnable offset, with no learnable normalization scale.
- Separate full square post-normalization transforms `A_a` and `A_b` are
  identity-initialized before the GLU.
- The model has 12,544,495 parameters versus 15,055,200 expanded training
  examples.

## Training contract

- Soft-target cross-entropy weight: `1`
- Soft LayerNorm weight: `1.5`
- Soft LayerNorm variance weight: `1`
- Distribution unit-sum weight: `0.01`
- Distribution non-negativity weight: `0.01`
- Soft weight-bound weight: `0.01`
- Centering idempotence weight: `1`
- Centering symmetry weight: `1`
- Desired weight magnitude: `1`
- Weight-bound sharpness: `10`
- Activation dropout: `0.05`
- Dropout application rate: `0.5`
- Gradient clipping: `5`
- Mixed precision: BF16
- Initial learning rate: `1e-4`
- Validation plateau reduction: factor `0.5`, patience `32`, floor `1e-6`
- Early stopping: more than `1024` stale validation epochs

Muon updates the fused projections and the 32 post-normalization `A` matrices.
AdamW updates the output head, affine biases, normalization offsets, and all 16
learnable centering matrices. Weight decay is disabled for both optimizers.

## Observed result

The best checkpoint was epoch 24 at global step 52,200:

- validation loss: `3.5543347498544673`
- validation cross-entropy: `3.4541087737152587`
- validation base-action KL: `0.11321715464827434`
- validation probability MSE: `0.00002488539960285854`
- validation soft LayerNorm penalty: `0.06414568355820544`
- validation distribution-layer loss: `0.002174015150146681`
- validation centering idempotence penalty: `0.0013435763539746404`
- validation centering symmetry penalty: `0.0004897921462543309`

At that epoch, training loss was `3.557548066099998`, training
cross-entropy was `3.4471553221957025`, and training base-action KL was
`0.10955485429363183`. This is the direct baseline for testing whether the
learnable centering operator and hard per-example RMS normalization make the
soft activation penalties redundant.
