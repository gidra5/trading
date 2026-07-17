# Exposure-value distillation objective

The VW-KAMA search can rank candidates by the proposed oracle-value distillation loss instead
of transition F1/agreement:

```bash
npm run search:kama -- \
  --objective value-distillation \
  --exposure-grid-size 21 \
  --exposure-min -1 --exposure-max 1 \
  --value-horizon-mode oracle-average-trade \
  --value-horizon 1h \
  --oracle-temperature 0.01 \
  --strategy-temperature 0.001 \
  --strategy-volatility-scaling true
```

`--accelerator auto` and `--accelerator cuda` support this objective. CUDA can prepare the
friction-aware Bellman oracle itself, parallelizing each exposure-grid step, and then evaluate
candidate batches. The oracle is prepared once per candle segment and shared by CPU workers or
copied once into each CUDA candidate batch. `cuda` forces both preparation and candidate
evaluation onto CUDA. `auto` uses the numerically equivalent CPU recurrence below 129 grid
points, where the sequential time dependency outweighs grid parallelism on the benchmarked
consumer GPU; it selects CUDA for grids from 129 through the native 1,024-point limit when CUDA
is available. On an actual 86,400-candle BTC one-second day, grid 101 took 20.40s on CPU versus
21.91s CUDA, while grid 129 took 32.80s CPU versus 26.79s CUDA. The inspector exposes the full
native grid range, follows the same crossover, and retains full probability rows for the
selected chart point.

## Oracle value

For every scored time `t` and grid exposure `a`, the oracle computes `Q_t(a)`: force exposure
`a` for the next value horizon `H`, then follow the optimal policy. The target exposure remains
exactly `a` throughout `H`, so price drift is corrected by intermediate rebalances using the
same friction model. The continuation is a backward Bellman recurrence from `t + H` over the
exposure grid. Near the segment end, the hold is truncated at the last available close.

`--value-horizon-mode fixed` uses `--value-horizon` directly. With
`--value-horizon-mode oracle-average-trade`, each continuous scored segment resolves `H` as the
arithmetic mean of the elapsed times between consecutive executable perfect-margin-oracle state
changes. The result is rounded to the nearest candle interval. If the segment has fewer than two
oracle transitions, `--value-horizon` is used as the fallback. The inspector displays the resolved
`H`; search preparation logs its range and holdout reports include it per case.

Rebalancing uses the exact buy/sell fee equations recorded in `tasks.md`. Maintenance applies
the configured per-candle quote lend, quote borrow, and asset borrow rates. If marked effective
exposure leaves the configured grid bounds, the position is liquidated to quote using friction
and continuation resumes from flat.

The oracle preference is:

`p_t(a) = softmax(Q_t(a) / oracleTemperature)`

Training examples are weighted by:

`W_t = max_a Q_t(a) - min_a Q_t(a) + opportunityEpsilon`

## Candidate distribution and loss

For evaluation, the signed KAMA rate defines a truncated exponential distribution over the
exposure grid:

`s_t(a) = exp(kappa_t * a) / sum_A exp(kappa_t * A)`

`kappa_t = (rateBpsHour_t / 10000) * (H / 1h) / effectiveTemperature_t`

A positive rate biases probability toward the maximum exposure, a negative rate toward the
minimum, and a zero rate is uniform. The effective temperature is:

`effectiveTemperature = strategyTemperature * sqrt(H / dt)`

Enabling `strategyVolatilityScaling` additionally multiplies effective temperature by the
population standard deviation of simple close-to-close returns over the same trailing `H`
interval. It is causal: only returns known at the current candle close are included. Thus
high-volatility periods make the exposure preference softer and low-volatility periods sharper.

The case loss is the normalized opportunity-weighted cross-entropy:

`CE = sum_t W_t * [-sum_a p_t(a) log s_t(a)] / sum_t W_t`

Genetic fitness is `-CE`, so larger fitness remains better. The report displays positive
cross-entropy, KL divergence, `exp(-KL)`, mean opportunity, and the previous signal score as a
diagnostic.

The standard search also writes its validation-selected volume-aware and canonical finalists
to `data/benchmarks/vw-kama-global-presets.json`. The KAMA inspector loads these presets and
uses the exact horizon, oracle temperature, exposure grid, strategy temperature, and volatility
setting stored with the run.

Because `log s_t(a) = kappa_t*a - log Z_t`, exact cross-entropy is simply
`log Z_t - kappa_t*E_p[a]`; it only needs the oracle distribution mean. The oracle cache retains
the second moment for diagnostics, but candidate loss does not need it. CUDA and CPU calculate
the same loss.

## Inspector visualization

The KAMA inspector retains full oracle probabilities only for interactive analysis. Click or
hover a rendered price-chart candle to compare grouped bars for `p_t(a)` and `s_t(a)`. The
panel also shows both distribution means, cross-entropy, oracle entropy, and opportunity. Full
probability rows are not retained by global-search workers, so the high-throughput search path
continues to use the compact sufficient statistics.

## Return measurements

The evaluator marks two hypothetical close-to-close portfolios alongside the loss:

- the strategy uses its actual price-marked exposure;
- the hindsight oracle uses the Bellman policy conditioned on its current marked exposure,
  maximizing `log(rebalanceFactor(current, a)) + Q_t(a)` at each H-step decision, then keeping
  that target until the next decision. The unconditioned distribution mode and soft mean
  `E_p[a]` remain visible in the chart but are not labeled as oracle return.

Every continuous segment starts with equity `1` and exposure `0`. Each target rebalance uses
the same exact fee equation as the Bellman oracle, followed by the configured quote/asset
maintenance rates and the next close. Final equity is marked at the last close without a
synthetic terminal liquidation. Reports include compounded return, maximum drawdown, and
turnover; return is diagnostic and does not enter genetic fitness.

## Production signal baseline

Every global run includes `production-current`. Its KAMA durations, rate mode, threshold,
clamp mode, adaptive-noise settings, and signal friction are derived from
`createPeakValleyStrategyConfig()` rather than copied into the search script. It is always
evaluated on fit, validation, and holdout, even when it does not rank among the learned
finalists.

The live strategy and batch runner share the same causal KAMA rate, threshold clamp,
adaptive-threshold adjustment, and consumed-edge friction-memory implementation. An
equivalence test compares the runner's transitions against `PeakValleyStrategy` directly.
Experimental confirmation and mean-reversion layers are disabled for this baseline.

This is a production **signal** comparison. The runner maps each accepted direction to a
normalized exposure for the distillation loss and return diagnostic; it does not simulate the
production grid bot, leverage, order placement, or exchange fills. Use the bot backtest for
execution-level PnL.

## Current scope

- Single asset and a fixed exposure grid.
- Close-to-close price observations.
- Fixed or per-segment oracle-average `H` target-exposure hold followed by perfect-oracle
  continuation.
- Per-candle borrow/lend rates, defaulting to zero.
- Strategy temperature and optional volatility normalization are search-level calibration
  values, not part of the genome.
- Existing transition metrics remain diagnostic and can still be selected with
  `--objective signal`.
