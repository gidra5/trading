# Return-oracle learned-radius shrinking baseline — 2026-07-30

## Preserved run

```text
data/ml-runs/return-oracle-ce-shrinking-v1-learned-radius-shrinking-baseline-20260730-104920
```

The run was stopped intentionally while still healthy so the next experiment
could widen every hidden layer to 512 and increase the training corpus
proportionally. Its `best.pt`, `last.pt`, status, and full append-only training
log remain in the archived directory.

## Configuration

- Inputs: 60 close-only simple returns, standardized with training-split
  per-position population statistics.
- Hidden widths: 16 GLU layers shrinking from 512 to 272 in steps of 16.
- Parameters: 15,090,191.
- Branch normalization:
  `A C x / sqrt(r^2 + s^2) + b`, with separate learnable value/gate `C`, `A`,
  bias, and scalar radius per layer.
- Radius initialization: `s = sqrt(1e-5)`, exactly reproducing the preceding
  hard-RMS denominator at initialization.
- Training examples: 15,400,800.
- Validation examples: 15,418,800, disjoint from training.
- Test examples: final 1,000,000 chronological examples.
- Optimizer: hybrid Muon/AdamW, three Newton–Schulz iterations, two source
  segments accumulated per optimizer update, BF16 compiled execution.
- Objective: soft-target CE plus independently gated skew reverse KL and
  one-sided entropy sharpness, soft LayerNorm weight 1, centering-projector
  constraints, and the soft weight-bound loss.

## Best validation result

Epoch 1,345 produced the best combined validation loss:

- combined validation loss: `3.471345953149787`;
- cross-entropy: `3.4120866245347266`;
- forward base-action `KL(Q || P)`: `0.0711950351548919`;
- skew reverse KL: `0.0896365071718409`;
- probability MSE: `1.2857891456921933e-5`;
- oracle entropy: `3.3408916048877795`;
- predicted entropy: `3.3796239891760322`;
- entropy gap: `0.038732382743350835`; and
- soft-LayerNorm penalty: `0.056100392473647176`.

The learning rate had reached its configured floor of `1e-6`. That epoch took
`25.406` seconds: `21.813` seconds training and `3.562` seconds validation.
The run was archived during epoch 1,396 with 50 stale epochs, finite gradients,
and no overflow.
