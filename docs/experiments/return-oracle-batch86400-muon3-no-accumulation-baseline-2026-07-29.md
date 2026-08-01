# One-segment batch with three-step Muon baseline

This record preserves the intermediate speed configuration before
two-segment gradient accumulation and the larger validation batch were
enabled.

The complete run is archived at:

```text
data/training/runs/return-oracle-ce-shrinking-v1-batch86400-muon3-no-accum-baseline-20260729-160827
```

Its controlled speed settings were:

- training batch capacity: `86,400`, producing one microbatch per segment;
- optimizer updates per epoch: `364`;
- evaluation batch size: `32,768`;
- Muon Newton-Schulz steps: `3`;
- batch-invariant validation regularizers computed once per pass; and
- Muon and AdamW learning rates: `1e-4`.

Steady epochs took approximately `28–29` seconds: about `23` seconds for
training and `5–6` seconds for validation.

The best checkpoint was epoch 4 at global step 1,820:

- validation loss: `3.878100654889037`
- validation cross-entropy: `3.8780421770645415`
- validation base-action KL: `0.5371505553756957`
- validation probability MSE: `0.00009716735385498709`

The run was stopped during epoch 5 after proving the speed and numerical
stability of three-step Muon.
