# Fixed-smoothed reverse-KL baseline

This record preserves the run using a fixed uniformly smoothed oracle in the
reverse-KL term before switching to an unbiased prediction-mixture reference.

The complete run is archived at:

```text
data/training/runs/return-oracle-ce-shrinking-v1-smoothed-reverse-kl-w005-baseline-20260729-170500
```

`best.pt`, `last.pt`, `status.json`, and the complete append-only
`training.log` are present. The run completed epoch 73 and was stopped during
epoch 74.

## Controlled objective

```text
Q_epsilon = (1 - 1e-4) Q + 1e-4 / 255
L = CE(Q,P) + 0.05 KL(P||Q_epsilon)
    + 0.01 L_weight-bound
    + L_idempotence
    + L_symmetry
```

All model, data, optimizer, batching, dropout, and scheduling settings match
the subsequent skew reverse-KL run.

## Observed result

The best checkpoint was epoch 72:

- validation loss: `3.471482120577515`
- validation cross-entropy: `3.459634064570393`
- validation forward/base-action KL: `0.11874246701011933`
- validation fixed-smoothed reverse KL: `0.23258978719604528`
- validation probability MSE: `0.00002612030865215346`

The last complete epoch took `20.594` seconds. This run is retained for
comparison because its reverse term has a slightly different ideal target
from the raw oracle.
