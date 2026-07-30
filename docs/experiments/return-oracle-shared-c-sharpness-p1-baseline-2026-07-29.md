# Shared-centering, 10% sharpness-gate baseline

This record preserves the final run before separating the value and gate
centering matrices. It was intentionally stopped during epoch 136 on
2026-07-29 after completing epoch 135.

The complete run is archived at:

```text
data/ml-runs/return-oracle-ce-shrinking-v1-shared-c-sharpness-p1-baseline-20260729-205615
```

`best.pt`, `last.pt`, `status.json`, and the complete append-only
`training.log` are present in that directory.

## Controlled configuration

- Model parameters: `12,544,495`
- Hidden layers: 16, shrinking from 512 to 272
- Centering matrices: 16 full matrices, one per layer and shared by value and
  gate
- Post-normalization transforms: 32 full identity-initialized `A` matrices
- Training examples: `15,055,200`
- Training batch capacity: `86,400`
- Gradient accumulation: two chronological segments per optimizer update
- Muon Newton-Schulz steps: `3`
- Soft-target cross-entropy weight: `1`
- Skew reverse-KL expected weight/probability: `0.02` / `0.01`
- Entropy-sharpness expected weight/probability: `0.02` / `0.1`
- Soft LayerNorm weight: `1`
- Distribution unit-sum and non-negativity weights: `0` (diagnostic only)
- Centering idempotence and symmetry weights: `1` each
- Soft weight-bound weight: `0.01`

## Observed result

The best checkpoint was epoch 133 at global step 24,388:

- validation loss: `3.494138002289944`
- validation cross-entropy: `3.4336747026543684`
- validation base-action KL: `0.09278308785658138`
- validation probability MSE: `0.000019204605201509964`
- validation soft-LayerNorm penalty: `0.05677778166924268`
- validation centering idempotence penalty: `0.0002908017486333847`
- validation centering symmetry penalty: `0.00016641238471493125`

Epoch 133 took `20.641` seconds: `17.422` seconds of training and `3.188`
seconds of validation. The run completed epoch 135 and was interrupted partway
through epoch 136 so the independent value/gate centering experiment could
start.
