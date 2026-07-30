# Continuous weight-1 skew-KL and entropy-sharpness baseline

This record preserves the run where skew reverse KL and one-sided output
entropy sharpness were both applied continuously at weight `1`, before moving
them to a shared corrected 1% optimizer-update gate.

The complete run is archived at:

```text
data/ml-runs/return-oracle-ce-shrinking-v1-continuous-skew-sharpness-w1-baseline-20260729-184300
```

`best.pt`, `last.pt`, `status.json`, and the complete append-only
`training.log` are present. The run completed epoch 131.

The best checkpoint was epoch 120:

- validation loss: `3.593227272343378`
- validation cross-entropy: `3.4530559038130613`
- validation forward/base-action KL: `0.11216429875023748`
- validation skew reverse KL: `0.12046236442169338`
- validation entropy gap: `-0.0008347956617490063`
- validation entropy-sharpness penalty: `0.01934487004651374`
- validation probability MSE: `0.00002262048255474996`

Epoch 120 took `20.438` seconds. This run demonstrated that continuous
sharpness pressure could move mean validation entropy slightly below the
oracle, but it also motivates testing a rarer shake rather than applying both
output regularizers on every update.
