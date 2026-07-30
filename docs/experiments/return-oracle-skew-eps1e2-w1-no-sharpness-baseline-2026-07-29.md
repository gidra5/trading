# Skew reverse-KL epsilon 1e-2 baseline without entropy sharpness

This record preserves the weight-`1`, epsilon-`1e-2` skew reverse-KL run before
adding the one-sided output-entropy sharpness loss.

The complete run is archived at:

```text
data/ml-runs/return-oracle-ce-shrinking-v1-skew-eps1e2-w1-no-sharpness-baseline-20260729-175200
```

`best.pt`, `last.pt`, `status.json`, and the complete append-only
`training.log` are present. The run completed epoch 46 and was stopped during
epoch 47.

## Controlled objective

```text
M = 0.99 Q + 0.01 P
L = CE(Q,P) + KL(P||M)
    + 0.01 L_weight-bound
    + L_idempotence
    + L_symmetry
```

All other model, data, optimizer, batching, dropout, and scheduling settings
match the subsequent entropy-sharpness run.

## Observed result

The best checkpoint was epoch 41:

- validation loss: `3.655137795468254`
- validation cross-entropy: `3.486699896718252`
- validation forward/base-action KL: `0.14580829096481834`
- validation skew reverse KL: `0.16829434829061066`
- validation predicted entropy: `3.4119454813482273`
- validation oracle entropy: `3.3408916048877795`
- validation entropy gap: `0.0710538764604478`
- validation probability MSE: `0.00003172185554666872`

Epoch 41 took `18.281` seconds. This is the direct comparison baseline for
the otherwise identical weight-`1` one-sided entropy-sharpness run.
