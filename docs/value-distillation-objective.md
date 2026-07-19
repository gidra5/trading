# Exposure-value distillation objective

The VW-KAMA search can rank candidates by the proposed oracle-value distillation loss instead
of transition F1/agreement:

```bash
npm run search:kama -- \
  --objective value-distillation \
  --exposure-grid-size 151 \
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
  --strategy-normal-mixture-range 0..1 \
  --strategy-normal-sigma-range 1..200 \
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
for the pure quadratic cross-entropy. Runs that search a nonzero normal mixture also retain the
oracle post-action probabilities needed by the exact mixture cross-entropy. Full diagnostics still run for
reported fit/validation/holdout results. Multiple resident cases are launched on independent CUDA
streams, and recurrent change rings use packed per-candidate offsets instead of a rectangular
`candidateCount × maximumPeriod × 2` allocation.

`cuda` forces CUDA oracle preparation and candidate evaluation. `auto` uses CUDA oracle
preparation after its normal size checks. Candidate scheduling
uses CUDA at four or more candidate-case lanes. The inspector uses CUDA for at least 10 million
scored candle-grid cells, otherwise CPU avoids one-request startup overhead. The
native CUDA grid limit and the UI maximum are both 1,024 grid points.

The repeatable benchmark is `npm run benchmark:kama:value:cuda`. On the RTX 3060 Laptop test
device, the exact untouched-hold recurrence for 20,000 candles × grid 101 × `H=300` takes about
11.69s on CPU and 2.41s of CUDA wall time (1.92s in CUDA kernels), a 4.9× wall-time speedup. For
384 mixed-feature candidates, fitness-only evaluation takes about 174ms versus 364ms for full
diagnostics; scheduling four resident cases takes about 177ms versus 661ms serially. CPU/CUDA
parity tests cover retained probabilities, policy, drifted exposure paths, cross-entropy, returns,
and scheduled fitness.

## Oracle value

For every scored time `t` and grid exposure `a`, the oracle computes `Q_t(a)`: rebalance to
`a` once, leave the resulting portfolio untouched for holding period `H`, then follow the optimal
policy until final-equity time `T = min(t + valueHorizon, segmentEnd)`. Exposure drifts naturally
throughout `H`; maintenance is applied but no target-restoring trades occur. At `t + H`, the
Bellman continuation compares grid rebalances with keeping the exact drifted exposure unchanged.

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

Write `F_t(a)` for the value after forcing target `a`, before paying the transition from the
current marked exposure. The complete oracle policy is

`Q_t(x,a) = F_t(a) + log R(x→a)`

`p_t(a|x) = softmax_a(Q_t(x,a) / oracleTemperature)`.

The configured grid supplies both current states `x` and targets `a`. The operational prior is
uniform over `x`; no balance-derived or path-frequency weighting is applied. This deliberately
trains the whole map so the policy remains defined after an external deposit, withdrawal, or
other execution-layer event moves exposure away from the usual oracle path.

Time weights use average regret rather than the diagonal value span:

`avgRegret_t = mean_x [max_a Q_t(x,a) - mean_a Q_t(x,a)]`

`W_t = avgRegret_t + opportunityEpsilon`.

The old post-action span `max_a F_t(a)-min_a F_t(a)` remains a diagnostic named opportunity,
but it no longer weights training.

## Candidate distribution and loss

For evaluation, the signed KAMA rate and volatility-scaled quadratic coefficient define a
quadratic-exponential post-action component over the exposure grid:

`s_t(a) = exp(kappa_t * a + b2_t * a²) / sum_A exp(kappa_t * A + b2_t * A²)`

`kappa_t = (rateBpsHour_t / 10000) * (H / 1h) / effectiveTemperature_t`

`v_t = populationStdDev(log(close_i / close_(i-1)))` over the causal trailing quadratic
volatility window

`b2_t = -strategyQuadraticScale * v_t²`

The complete post-action predictor is a mixture with a separately normalized, truncated normal
centered on the candidate target exposure `c_t`:

`n_t(a) = exp(-0.5*((a-c_t)/sigma)^2) / sum_A exp(-0.5*((A-c_t)/sigma)^2)`

`m_t(a) = (1-rho)*s_t(a) + rho*n_t(a)`

`rho = strategyNormalMixture`, `sigma = strategyNormalSigma`.

The first component is itself equivalent to a truncated normal whenever `b2_t < 0`, but its
center and width are coupled to the signed rate and trailing volatility. The second component is
not redundant: it is anchored to the strategy's requested, price-marked target and has an
independently controlled width. `rho=0` exactly recovers the former model and is the UI/preset
default.

A positive rate biases probability toward the maximum exposure, a negative rate toward the
minimum. The configured nonnegative scale `b2'` therefore always produces a concave quadratic:
larger volatility concentrates probability more strongly around its interior mode. A zero scale
or zero measured volatility gives `b2=0` and recovers the original truncated exponential with
its exact constant-time partition. The effective temperature is:

Strategy temperature, quadratic scale, quadratic volatility window, target-normal mixture and
sigma, threshold-noise response, and candidate friction fraction are signal parameters searched independently for every candidate
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

The policy used by training is conditioned on the input exposure and uses the same exact
rebalance factor as the oracle:

`s_t(a|x) ∝ m_t(a) * R(x→a)^(signalFrictionFraction/effectiveTemperature_t)`.

The candidate friction fraction controls how strongly the predictor internalizes transition
friction. Zero recovers a current-state-independent policy; one uses the full configured oracle
friction.

The case loss is cross-entropy over every input-output pair. The current-state dimension is
uniform, while time uses the average-regret weight above:

`CE_t = mean_x [-sum_a p_t(a|x) log s_t(a|x)]`

`CE = sum_t W_t*CE_t / sum_t W_t`.

Genetic fitness is `-CE`, so larger fitness remains better. The report displays positive
cross-entropy, KL divergence, `exp(-KL)`, mean opportunity, and the previous signal score as a
diagnostic.

The standard search also writes its validation-selected volume-aware and canonical finalists
to `data/benchmarks/vw-kama-global-presets.json`. The KAMA inspector loads these presets and
uses the exact holding period, value horizon, exposure limits, temperatures, quadratic scale, quadratic volatility
window, and temperature-volatility setting stored with the run.

The buy and sell fee equations are separable. CPU and CUDA compute the row-averaged log partition,
target moments, log-rebalance moment, entropy, and the oracle expectation of the mixture log-density
with prefix/suffix scans in `O(grid)` per candle, without materializing the `grid²` matrix. The
last quantity makes the mixture cross-entropy exact rather than a moment-based surrogate. Oracle preparation runs grid current states in parallel
on CUDA, then uniformly reduces their sufficient statistics. At extremely low temperature, both
paths use the hold-dominant limit once every off-diagonal transition is below floating-point
relevance. Precise oracle MI uses an additional binned conditional-policy pass; approximate mode
uses the fused conditional moments.

## Inspector visualization

The main price chart uses one shared inspection timestamp for the selected-point diagnostics and
the lower indicator charts. Hovering previews a timestamp, a movement-thresholded click pins it,
and double-clicking releases the pin so every dependent chart follows hover again. Dragging still
pans without accidentally pinning a timestamp.

The KAMA inspector retains the grouped bars as post-action (`x=a`, zero transition cost)
preferences. Below them it renders the actual transition-aware conditional distributions used by
the loss: oracle `p(a | x)`, prediction `s(a | x)`, and the pointwise probability difference
`s(a | x) - p(a | x)`. Rows are current exposure and columns are target exposure; every
probability row sums to one, and both probability panels share one robust color scale. Hovering a
cell shows exact `x`, `a`, probability, and both row-modal targets. Cyan and violet points mark the
oracle and predicted mode for every input-exposure row. A solid cyan outline marks the oracle-path
state row and a dashed green outline marks the candidate-current row. Conditional slices show the
oracle at its reconstructed path, the oracle at the candidate's current state, and the prediction
at that same candidate state. The oracle, prediction, and difference heatmap panels can be toggled
independently; visible panels reflow to fill the available width. Oracle and prediction row modes,
row means, the oracle-path row, candidate row, and `x + a = 0` trace also have independent shared
marking toggles that apply to every visible heatmap.
Grid-squared matrices are derived only for the selected point in the browser; workers and API
payloads keep compact sufficient statistics.

Hovering a cell in the oracle or prediction heatmap adds two curves to the conditional-slices
chart: the normalized row `p(a | x_hover)` across target exposure and the raw column
`p(a_hover | x)` across current exposure. The column is a cross-section through independently
normalized rows and therefore does not itself sum to one. A single click pins the heatmap source
and cell, while double-clicking any heatmap (or pressing **Follow hover**) releases the pin. The
pinned source cell is outlined on its heatmap.

The post-action distribution chart keeps the raw oracle masses as points, overlays their
quadratic-exponential pointwise probability-MSE fit, and renders the exact mixture prediction as a
continuous line. The conditional chart fits the oracle separately at the
oracle-path and candidate-current states. It also fits one shared coefficient pair over the full
conditional map using the uniform current-state prior. All conditional fits use the predictor's
fixed transition scale in
`softmax(b1*a + b2*a^2 + transitionScale*log(rebalanceFactor(x,a)))`, constrain `b2 <= 0` to the
actual predictor family, and display KL divergence. The whole-map fit is plotted at the candidate
state alongside the exact predicted curve. Every sample set and curve has its own toggle below its
chart; hiding a series also removes it from the chart's probability-scale calculation. A further anti-diagonal
cross-section plots the raw oracle and prediction cells satisfying `x + a = 0`; it is highlighted
in gold on each heatmap. Since its current state changes with every target, this cross-section is
not renormalized and should not be interpreted as a single conditional row summing to one.

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
- Uniform operational weighting over all grid current exposures; no minimum-order or absolute
  balance state.
- Existing transition metrics remain diagnostic and can still be selected with
  `--objective signal`.
