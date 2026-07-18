# Exposure-value distillation objective

The VW-KAMA search can rank candidates by the proposed oracle-value distillation loss instead
of transition F1/agreement:

```bash
npm run search:kama -- \
  --objective value-distillation \
  --exposure-grid-size 150 \
  --exposure-min -100 --exposure-max 100 \
  --max-effective-exposure 250 \
  --value-holding-period-mode fixed \
  --value-holding-period 1s \
  --value-horizon 1s \
  --value-horizon-end-mode truncate \
  --oracle-temperature 0.01 \
  --strategy-temperature-range 0.000001..0.1 \
  --strategy-quadratic-scale-range 0..1000000 \
  --strategy-quadratic-volatility-window-range 1m..24h \
  --threshold-noise-responses proportional,inverse \
  --signal-friction-fraction-range 0..1 \
  --strategy-volatility-scaling true
```

`--accelerator auto` and `--accelerator cuda` support this objective. The exact finite-horizon
Bellman update uses separable buy/sell fee factors, reducing each time step from `O(grid²)` to
`O(grid)`. CPU and CUDA both evaluate the rolling horizon in holding-period blocks; CUDA launches
all scored starting times in parallel at each Bellman depth. The oracle is prepared once per
candle segment and retained by the prepared-stage cache.

Candidate fitness cases remain resident on the GPU across generations. Search therefore uploads
columns and oracle sufficient statistics once, then communicates only packed parameters and
fitness results. Fitness-only evaluation omits transition alignment, return diagnostics, oracle
paths, and other candidate-independent columns; oracle first and second moments remain resident
for the quadratic cross-entropy. Full diagnostics still run for
reported fit/validation/holdout results. Multiple resident cases are launched on independent CUDA
streams, and recurrent change rings use packed per-candidate offsets instead of a rectangular
`candidateCount × maximumPeriod × 2` allocation.

`cuda` forces CUDA oracle preparation and candidate evaluation. `auto` uses CUDA oracle
preparation after its normal size checks. Candidate scheduling
uses CUDA at four or more candidate-case lanes. The inspector uses CUDA for at least 10 million
scored candle-grid cells, otherwise CPU avoids one-request startup overhead. The
native CUDA grid limit and the UI maximum are both 1,024 grid points.

The repeatable benchmark is `npm run benchmark:kama:value:cuda`. On the RTX 3060 Laptop test
device, 20,000 candles × grid 101 × `H=300` improved from 4.85s to about 0.37s on CPU and from
5.01s to about 0.14s of CUDA kernel time. For 384 mixed-feature candidates, fitness-only
evaluation is about 63ms versus 246ms for full diagnostics; scheduling four resident cases takes
about 80ms versus 280ms serially. CPU/CUDA parity tests cover retained probabilities, policy,
cross-entropy, returns, and scheduled fitness.

## Oracle value

For every scored time `t` and grid exposure `a`, the oracle computes `Q_t(a)`: force exposure
`a` for holding period `H`, then follow the optimal policy until final-equity time
`T = min(t + valueHorizon, segmentEnd)`. The target exposure remains exactly `a` throughout `H`,
so price drift is corrected by intermediate rebalances using the same friction model. The finite
Bellman continuation runs from `t + H` through `T`.

`--value-holding-period-mode fixed` uses `--value-holding-period` directly and defaults to one
candle (`1s` is one step at every supported scale). With
`--value-holding-period-mode oracle-half-average-trade`, each continuous scored segment resolves `H` as
half the arithmetic mean of the elapsed times between consecutive executable
perfect-margin-oracle state changes. The result is rounded to the nearest candle interval. If the
segment has fewer than two oracle transitions, `--value-holding-period` is used as the fallback.
`--value-horizon` independently sets `T-t` and must be at least the configured fallback `H`; if an
adaptive `H` is longer, preparation fails and reports that the horizon must be increased. With
`--value-horizon-end-mode truncate` (the default), `T` is capped at the scored window's last close.
With `--value-horizon-end-mode extend`, preparation loads post-window candles and keeps
`T=t+valueHorizon` for every scored example. Candidate scoring still ends at the window boundary;
only the hindsight oracle target may read beyond it. Missing or discontinuous post-window candles
are an error in `extend` mode. The inspector and reports display both durations and the end mode.

Rebalancing uses the exact buy/sell fee equations recorded in `tasks.md`. Maintenance applies
the configured per-candle quote lend, quote borrow, and asset borrow rates. Trades may select only
targets inside `[exposure-min, exposure-max]`. Price and fee drift may carry the marked position
outside that target range; liquidation to quote occurs only when its absolute fee-adjusted
exposure exceeds `max-effective-exposure` (250× by default).

The oracle preference is:

`p_t(a) = softmax(Q_t(a) / oracleTemperature)`

Training examples are weighted by:

`W_t = max_a Q_t(a) - min_a Q_t(a) + opportunityEpsilon`

## Candidate distribution and loss

For evaluation, the signed KAMA rate and volatility-scaled quadratic coefficient define a
quadratic exponential distribution over the exposure grid:

`s_t(a) = exp(kappa_t * a + b2_t * a²) / sum_A exp(kappa_t * A + b2_t * A²)`

`kappa_t = (rateBpsHour_t / 10000) * (H / 1h) / effectiveTemperature_t`

`v_t = populationStdDev(log(close_i / close_(i-1)))` over the causal trailing quadratic
volatility window

`b2_t = -strategyQuadraticScale * v_t²`

A positive rate biases probability toward the maximum exposure, a negative rate toward the
minimum. The configured nonnegative scale `b2'` therefore always produces a concave quadratic:
larger volatility concentrates probability more strongly around its interior mode. A zero scale
or zero measured volatility gives `b2=0` and recovers the original truncated exponential with
its exact constant-time partition. The effective temperature is:

Strategy temperature, quadratic scale, quadratic volatility window, threshold-noise response,
and candidate friction fraction are signal parameters searched independently for every candidate
and stored directly in each selected preset. The quadratic volatility window search defaults to
`1m..24h`. At each candle scale it is rounded to the
nearest positive candle count. If it resolves to one return, its population standard deviation
is zero and the quadratic term is inactive at that scale.

`effectiveTemperature = strategyTemperature * sqrt(H / dt)`

Enabling `strategyVolatilityScaling` additionally multiplies effective temperature by the same
calculation over the holding period `H`, independently of the quadratic volatility window. Both
measurements are causal and include only log returns known at the current candle close. Thus the
optional temperature scaling softens the linear rate preference using trailing-H volatility,
while the quadratic term grows with volatility measured over its separately configured window.

The case loss is the normalized opportunity-weighted cross-entropy:

`CE = sum_t W_t * [-sum_a p_t(a) log s_t(a)] / sum_t W_t`

Genetic fitness is `-CE`, so larger fitness remains better. The report displays positive
cross-entropy, KL divergence, `exp(-KL)`, mean opportunity, and the previous signal score as a
diagnostic.

The standard search also writes its validation-selected volume-aware and canonical finalists
to `data/benchmarks/vw-kama-global-presets.json`. The KAMA inspector loads these presets and
uses the exact holding period, value horizon, exposure limits, temperatures, quadratic scale, quadratic volatility
window, and temperature-volatility setting stored with the run.

Because `log s_t(a) = kappa_t*a + b2_t*a² - log Z_t`, exact cross-entropy is
`log Z_t - kappa_t*E_p[a] - b2_t*E_p[a²]`. The oracle cache therefore retains both sufficient
moments. CUDA and CPU calculate the same per-candle coefficient and loss. `b2_t=0` stays on the
constant-time geometric-series path. For negative `b2_t`, concavity bounds a fixed-width window
around the discrete mode; omitted tails are below Float64 resolution on CPU and Float32 resolution
on CUDA. The forward quadratic recurrence gives every GPU candidate the same loop count at each
candle. The generic positive-quadratic helper retains the full-grid sum.

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
- Fixed or per-segment half-oracle-average `H` target-exposure hold followed by perfect-oracle
  continuation.
- Per-candle borrow/lend rates, defaulting to zero.
- Strategy temperature and optional volatility normalization are search-level calibration
  values, not part of the genome.
- Existing transition metrics remain diagnostic and can still be selected with
  `--objective signal`.
