# Exposure-value distillation objective

The VW-KAMA search can rank candidates by the proposed oracle-value distillation loss instead
of transition F1/agreement:

```bash
npm run search:kama -- \
  --objective value-distillation \
  --exposure-grid-size 21 \
  --exposure-min -1 --exposure-max 1 \
  --oracle-temperature 0.01 \
  --strategy-sigma 0.15
```

`--accelerator auto` and `--accelerator cuda` support this objective. The friction-aware oracle
is prepared once per candle segment and shared by all CPU workers or copied once into each CUDA
batch.

## Oracle value

For every scored time `t` and grid exposure `a`, the oracle computes `Q_t(a)`: force exposure
`a` across the next price move, then follow the optimal policy until the end of the evaluation
segment. The continuation is a backward Bellman recurrence over the exposure grid.

Rebalancing uses the exact buy/sell fee equations recorded in `tasks.md`. Maintenance applies
the configured per-candle quote lend, quote borrow, and asset borrow rates. If marked effective
exposure leaves the configured grid bounds, the position is liquidated to quote using friction
and continuation resumes from flat.

The oracle preference is:

`p_t(a) = softmax(Q_t(a) / oracleTemperature)`

Training examples are weighted by:

`W_t = max_a Q_t(a) - min_a Q_t(a) + opportunityEpsilon`

This first implementation uses return until the end of the current continuous evaluation
segment. A fixed-horizon value target can be added later without changing the candidate loss.

## Candidate distribution and loss

The existing strategy emits one marked exposure rather than a distribution. For evaluation it
is interpreted as a discretized Gaussian distribution over the same exposure grid, centered on
the candidate exposure with standard deviation `strategySigma`.

The case loss is the normalized opportunity-weighted cross-entropy:

`CE = sum_t W_t * [-sum_a p_t(a) log s_t(a)] / sum_t W_t`

Genetic fitness is `-CE`, so larger fitness remains better. The report displays positive
cross-entropy, KL divergence, `exp(-KL)`, mean opportunity, and the previous signal score as a
diagnostic.

Because `log s_t(a)` is quadratic in exposure, exact cross-entropy only needs the oracle
distribution's mean, second moment, and entropy. These sufficient statistics keep the shared
oracle cache to five floats per candle and avoid an exposure-grid loop for oracle probabilities
inside each candidate evaluation. CUDA and CPU calculate the same loss.

## Current scope

- Single asset and a fixed exposure grid.
- Close-to-close price observations.
- End-of-segment value horizon.
- Per-candle borrow/lend rates, defaulting to zero.
- The strategy distribution width is a search-level calibration value, not part of the genome.
- Existing transition metrics remain diagnostic and can still be selected with
  `--objective signal`.
