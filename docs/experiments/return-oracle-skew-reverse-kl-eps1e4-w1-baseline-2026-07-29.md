# Skew reverse-KL epsilon 1e-4 baseline

This record preserves the first weight-`1` skew reverse-KL run before its
prediction-mixture epsilon was increased from `1e-4` to `1e-2`.

The complete run is archived at:

```text
data/ml-runs/return-oracle-ce-shrinking-v1-skew-reverse-kl-eps1e4-w1-baseline-20260729-173400
```

`best.pt`, `last.pt`, `status.json`, and the complete append-only
`training.log` are present. The run completed epoch 66 and was stopped during
epoch 67.

## Controlled objective

```text
M = (1 - 1e-4) Q + 1e-4 P
L = CE(Q,P) + KL(P||M)
    + 0.01 L_weight-bound
    + L_idempotence
    + L_symmetry
```

All other model, data, optimizer, batching, dropout, and scheduling settings
match the subsequent epsilon-`1e-2` run.

## Observed result

The best checkpoint was epoch 64:

- validation loss: `3.6559933166606364`
- validation cross-entropy: `3.481865638164669`
- validation forward/base-action KL: `0.1409740085782496`
- validation skew reverse KL: `0.17393384645048277`
- validation probability MSE: `0.0000316148636650267`

Epoch 64 took `19.969` seconds. This run is the direct comparison baseline for
measuring how softening the zero-support penalty from `log(10,000)` to
`log(100)` affects convergence.
