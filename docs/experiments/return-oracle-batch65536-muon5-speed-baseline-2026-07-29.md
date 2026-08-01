# Batch 65,536 with five-step Muon speed baseline

This record preserves the first large-batch speed experiment before moving to
one chronological segment per batch, reducing Muon's Newton-Schulz iteration
count, and eliminating redundant validation regularizer calculations.

The complete run is archived at:

```text
data/training/runs/return-oracle-ce-shrinking-v1-batch65536-muon5-baseline-20260729-160152
```

The model and objective match the active learnable-centering diagnostic-loss
experiment. Its controlled speed settings were:

- training batch size: `65,536`;
- training optimizer steps per epoch: `528`;
- evaluation batch size: `32,768`;
- Muon Newton-Schulz steps: `5`; and
- Muon and AdamW learning rates: `1e-4`.

Steady epochs generally took `43.7–54.7` seconds. Training took approximately
`36.0–45.4` seconds and validation `7.6–9.5` seconds. Full-batch throughput
was approximately `321k–382k` expanded examples per second, with no gradient
overflow.

The best checkpoint was epoch 8 at global step 4,752:

- validation loss: `3.624963726776319`
- validation cross-entropy: `3.6248590550934923`
- validation base-action KL: `0.2839674443061061`

The run was stopped during epoch 10 after epoch 9 completed.
