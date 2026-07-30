# Learnable centering diagnostic-activation baseline at batch 8,192

This record preserves the activation-loss ablation at the original training
batch size before testing a larger batch to reduce Muon's per-step overhead.
It was intentionally stopped during epoch 20 on 2026-07-29 after completing
epoch 19.

The complete run is archived at:

```text
data/ml-runs/return-oracle-ce-shrinking-v1-batch8192-diagnostic-activations-baseline-20260729-155008
```

`best.pt`, `last.pt`, `status.json`, and the complete append-only
`training.log` are present in that directory.

## Controlled configuration

- Training batch size: `8,192`
- Evaluation batch size: `32,768`
- Batches per training epoch: `2,088`
- Initial Muon and AdamW learning rates: `1e-4`
- Muon Newton-Schulz steps: `5`
- Model parameters: `12,544,495`
- Soft-target cross-entropy weight: `1`
- Soft LayerNorm weight: `0` (diagnostic only)
- Distribution unit-sum weight: `0` (diagnostic only)
- Distribution non-negativity weight: `0` (diagnostic only)
- Soft weight-bound weight: `0.01`
- Centering idempotence weight: `1`
- Centering symmetry weight: `1`

All data, model, optimizer, dropout, normalization, and scheduling settings
match the subsequent 65,536-example batch experiment.

## Observed result

The best checkpoint was epoch 19 at global step 41,760:

- validation loss: `3.4868639398316983`
- validation cross-entropy: `3.4854610849122647`
- validation base-action KL: `0.14456946728697861`
- validation probability MSE: `0.00003150927715206789`
- validation centering idempotence penalty: `0.0010077455081045628`
- validation centering symmetry penalty: `0.00039508522604592144`

At epoch 19, training loss was `3.45546348202475`, training cross-entropy was
`3.4541094337803724`, and training base-action KL was
`0.11650896697298858`. Recent complete epochs took approximately 150–175
seconds, with about 142–166 seconds spent in training and 8–10 seconds in
validation.
