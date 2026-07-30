# Bias-only LayerNorm plus full post-normalization A baseline

This record preserves the successful return-oracle configuration that preceded
the learnable-centering experiment. It was intentionally stopped during epoch
59 on 2026-07-29 so its checkpoints and metrics could be retained before
changing the LayerNorm centering operator.

The complete run is archived at:

```text
data/ml-runs/return-oracle-ce-shrinking-v1-bias-ln-full-a-baseline-20260729-132700
```

`best.pt`, `last.pt`, `status.json`, and the complete append-only
`training.log` are present in that directory.

## Data contract

- Inputs are exactly 60 adjacent simple returns derived only from completed
  one-minute close values.
- Every return coordinate uses its own training-split population mean and
  standard deviation. Validation and test reuse those frozen statistics.
- Targets are the stored raw 255-cell minute-oracle distributions aligned to
  the same close path.
- Training contains 15,055,200 expanded examples, validation contains
  15,418,800 examples, and test is the final 1,000,000 chronological examples.

## Network contract

The 16 effective hidden widths are:

```text
512, 496, 480, 464, 448, 432, 416, 400,
384, 368, 352, 336, 320, 304, 288, 272
```

For each layer:

```text
(a0, b0) = split(Px + c)
a_norm = fixed_LayerNorm(a0) + beta_a
b_norm = fixed_LayerNorm(b0) + beta_b
a = A_a a_norm
b = A_b b_norm
h = a * sigmoid(b)
```

- `P` absorbs the earlier input transform `B`.
- Fixed LayerNorm performs exact per-example centering and population-variance
  normalization.
- LayerNorm scale is fixed at one.
- `beta_a` and `beta_b` are separate learnable per-neuron offsets.
- `A_a` and `A_b` are separate full square matrices, initialized to identity.
- Soft LayerNorm continues to measure the raw `a0` and `b0` projections.
- The final affine head emits 255 raw action logits.

The model contains 9,998,831 parameters:

- 4,812,800 fused value/gate projection weights;
- 5,091,328 post-normalization `A` weights;
- 12,544 LayerNorm offsets; and
- 82,159 projection biases and output-head parameters.

## Training contract

- Soft-target cross-entropy weight: `1`
- Soft LayerNorm weight: `1.5`
- Soft LayerNorm variance weight: `1`
- Distribution unit-sum weight: `0.01`
- Distribution non-negativity weight: `0.01`
- Soft weight-bound weight: `0.01`
- Desired weight magnitude: `1`
- Weight-bound sharpness: `10`
- Activation dropout: `0.05`
- Dropout application rate: `0.5`
- Gradient clipping: `5`
- Mixed precision: BF16
- Initial learning rate: `1e-4`
- Validation plateau reduction: factor `0.5`, patience `32`, floor `1e-6`
- Early stopping: more than `1024` stale validation epochs

Muon updates the 16 fused projections and 32 `A` matrices, totaling 9,904,128
parameters. AdamW updates the output head, all affine biases, and all 32
LayerNorm offsets, totaling 94,703 parameters. Weight decay is disabled for
both optimizers.

## Observed result

The best checkpoint was epoch 57 at global step 121,104:

- validation loss: `3.5366934129221788`
- validation cross-entropy: `3.4554504256669434`
- validation base-action KL: `0.11455881691365653`
- validation probability MSE: `0.000024979822894206575`
- validation soft LayerNorm penalty: `0.0529194073842732`
- validation distribution-layer loss: `0.0018638793692040067`

At the same epoch, training loss was `3.490984397916613` and training
base-action KL was `0.06730088288162665`. This configuration is therefore a
useful stable baseline for judging whether a learnable constrained centering
operator improves generalization.
