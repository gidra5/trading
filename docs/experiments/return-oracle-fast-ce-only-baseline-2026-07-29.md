# Fast accumulated CE-only baseline

This record preserves the last complete fast run before adding smoothed
reverse KL to the objective.

The complete run is archived at:

```text
data/training/runs/return-oracle-ce-shrinking-v1-fast-ce-only-baseline-20260729-163122
```

`best.pt`, `last.pt`, `status.json`, and the complete append-only
`training.log` are present in that directory. The run completed epoch 61 and
was stopped during epoch 62.

## Controlled configuration

- Training microbatch capacity: `86,400`
- Gradient accumulation: `2` consecutive chronological segment microbatches
- Microbatches per epoch: `364`
- Optimizer updates per epoch: `182`
- Evaluation batch size: `86,400`
- Muon Newton-Schulz steps: `3`
- Muon and AdamW learning rates: `1e-4`
- Soft-target cross-entropy weight: `1`
- Reverse-KL weight: `0` (not yet implemented)
- Soft LayerNorm weight: `0` (diagnostic only)
- Distribution unit-sum and non-negativity weights: `0` (diagnostic only)
- Soft weight-bound weight: `0.01`
- Centering idempotence weight: `1`
- Centering symmetry weight: `1`
- Model parameters: `12,544,495`

## Observed result

The best checkpoint was epoch 61:

- validation loss: `3.4658628049063887`
- validation cross-entropy: `3.4656705046935863`
- validation forward/base-action KL: `0.12477890432542858`
- validation probability MSE: `0.00002627549448333153`

The steady complete epoch time was approximately `20.781` seconds:
`17.656` seconds for training and `3.094` seconds for validation.

This is the direct comparison baseline for the otherwise identical
`CE(Q,P) + 0.05 KL(P||Q_epsilon)` run.
