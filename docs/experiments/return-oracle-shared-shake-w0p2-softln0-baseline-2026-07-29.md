# Shared 1% output shake at expected weight 0.2 with diagnostic soft LayerNorm

This record preserves the run immediately before splitting the skew
reverse-KL and entropy-sharpness gates, reducing their expected weights from
`0.2` to `0.02`, and restoring soft LayerNorm at weight `1`. It was
intentionally stopped after completing epoch 64 on 2026-07-29.

The complete run is archived at:

```text
data/ml-runs/return-oracle-ce-shrinking-v1-shared-shake-w0p2-softln0-baseline-20260729-191415
```

`best.pt`, `last.pt`, `status.json`, and the complete append-only
`training.log` are present in that directory.

## Controlled configuration

- Feature contract:
  `train-position-standardized-completed-minute-close-only-simple-returns-v2`
- Architecture contract:
  `shrinking-fused-glu-learnable-shared-centering-bias-ln-full-a-v12`
- Model parameters: `12,544,495`
- Effective hidden widths: 16 layers shrinking from `512` to `272`
- Batch capacity: `86,400`
- Gradient accumulation: two temporal-segment microbatches
- Optimizer updates per epoch: `182`
- Muon Newton-Schulz steps: `3`
- Cross-entropy weight: `1`
- Skew reverse-KL prediction-mixture epsilon: `0.01`
- Skew reverse-KL expected weight: `0.2`
- Entropy-sharpness expected weight: `0.2`
- Output-loss application probability: `0.01`
- Gate coupling: one shared Bernoulli draw per optimizer update
- Inverse-probability scale: `100`
- Active skew reverse-KL and entropy-sharpness weights: `20` each
- Soft LayerNorm weight: `0` (diagnostic only)
- Distribution unit-sum and non-negativity weights: `0` (diagnostic only)
- Soft weight-bound weight: `0.01`
- Centering idempotence and symmetry weights: `1` each
- Gradient clipping: `5`
- Mixed precision: BF16

The shared output gate was sampled once per optimizer update and reused for
both accumulated microbatches. Validation used the deterministic expected
weight `0.2` for both output terms.

## Observed result

The best checkpoint was epoch 61 at global step 11,284:

- validation loss: `3.5100566521516234`
- validation cross-entropy: `3.463752645158111`
- validation base-action KL: `0.12286103300153584`
- validation skew reverse KL: `0.17563819514551426`
- validation entropy gap: `0.11066280776809205`
- validation entropy sharpness: `0.0549307994992663`
- validation probability MSE: `0.000026346837307426635`
- validation soft-LayerNorm diagnostic: `0.686041862802363`
- validation centering constraint: `0.00019012956181541085`

Epoch 64 was the final completed epoch. Its validation loss was
`3.5257548058345978`, with cross-entropy `3.4735182441063026`. The epoch took
`21.329` seconds, of which `17.625` seconds were training and `3.672` seconds
were validation.
