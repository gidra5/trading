# Independent 1% output gates at expected weight 0.02 with soft LayerNorm

This record preserves the run immediately before increasing only the
entropy-sharpness application probability from `0.01` to `0.1`. It was
intentionally stopped after completing epoch 91 on 2026-07-29.

The complete run is archived at:

```text
data/training/runs/return-oracle-ce-shrinking-v1-independent-p01-w002-softln1-baseline-20260729-195620
```

The directory contains `best.pt`, `last.pt`, `status.json`, the complete
append-only `training.log`, and both process launch logs.

## Controlled configuration

- Architecture:
  `shrinking-fused-glu-learnable-shared-centering-bias-ln-full-a-v12`
- Parameters: `12,544,495`
- Optimizer updates per epoch: `182`
- Muon Newton-Schulz steps: `3`
- Cross-entropy weight: `1`
- Skew reverse-KL expected weight: `0.02`
- Entropy-sharpness expected weight: `0.02`
- Reverse-KL application probability: `0.01`
- Entropy-sharpness application probability: `0.01`
- Gate coupling: independent Bernoulli draws per optimizer update
- Inverse-probability scale: `100` for both losses
- Active reverse-KL and entropy-sharpness weights: `2` each
- Soft LayerNorm weight: `1`
- Distribution-layer losses: diagnostic only
- Soft weight-bound weight: `0.01`
- Centering idempotence and symmetry weights: `1` each
- Gradient clipping: `5`
- Mixed precision: BF16

## Observed result

The best checkpoint was the final completed epoch, epoch 91 at global step
16,744:

- validation loss: `3.5122174832166793`
- validation cross-entropy: `3.4498756383864255`
- validation base-action KL: `0.10898402702940066`
- validation skew reverse KL: `0.15832186962286293`
- validation entropy gap: `0.04729103245006467`
- validation entropy sharpness: `0.03385579167966849`
- validation probability MSE: `0.00002234530230303134`
- validation soft-LayerNorm penalty: `0.05812769770322612`
- validation centering constraint: `0.00037051294930279255`

Epoch 91 took `21.203` seconds, including `17.828` seconds of training and
`3.344` seconds of validation.
