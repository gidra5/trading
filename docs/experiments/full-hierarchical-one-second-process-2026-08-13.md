# Full hierarchical one-second forecast process

Date: 2026-08-13  
Symbol: BTCUSDT spot  
Machine-readable report: `data/benchmarks/full-hierarchical-one-second-process.json`

## Result

The fitted process now produces fresh 1-second candle paths that remain coherent
from 1 second through 1 day. On a final untouched 91-day chronological holdout it
passes all predefined acceptance gates:

| Audit | Untouched result | Gate |
| --- | ---: | ---: |
| 1s return-histogram JS divergence | 0.002047 bits | < 0.005 |
| Zero-change gap JS divergence | 0.000486 bits | < 0.001 |
| Signed-return ACF mean absolute error | 0.008151 | diagnostic |
| Absolute-return ACF mean absolute error | 0.047508 | < 0.05 |
| Activity ACF mean absolute error | 0.006992 | < 0.05 |
| Minute return constraint error | < 6.6e-12 bps | < 1e-8 |
| Minute variance constraint error | < 6.3e-12 bps squared | < 1e-8 |
| Minute activity-count error | 0 | exactly 0 |
| Cross-scale coherence error | 0 | exactly 0 |

This is a scenario forecast, not a prediction of the single realized future
path. Passing the audit means the generated ensemble is statistically plausible
under the tested regime and split. It does not establish trading profitability or
guarantee calibration after a regime change.

## Leakage-safe chronological design

The 365 forecast origins from 2025-07-25 through 2026-07-24 are split in order:

1. 183 origins, through 2026-01-23: history-window and marginal-calibrator
   selection.
2. 91 origins, 2026-01-24 through 2026-04-24: hierarchy architecture,
   reconciliation weights, and micro-kernel selection.
3. 91 origins, 2026-04-25 through 2026-07-24: untouched confirmation only.

The second-level fitted process uses data only before 2026-04-25. Final-holdout
outcomes are not used for window selection, reconciliation, copula weights,
microstructure calibration, or fallback decisions.

No historical candle, activity mask, sign mask, magnitude profile, or day is
resampled during generation. Historical data are reduced to fitted parameters and
calibration residual distributions.

## Modeled process

### 1. Independent scale forecasts

At each daily origin the process refits local target models at:

`1d -> 8h -> 4h -> 2h -> 1h -> 30m -> 15m -> 1m`

For each scale it forecasts:

- period log return in bps;
- integrated one-second realized variance in bps squared;
- active one-second count.

Variance is generated in log space with a shrinkage AR(1) Student-t process.
Return efficiency, $R / \sqrt{Q}$, is generated conditionally on variance.
Activity is generated in logit space conditionally on log variance. The chosen
history window depends on scale and feature; the report records all candidates
and selection scores.

Marginal residual calibration is walk-forward. Calibrated quantiles are returned
to the original simulated member ranks at every timestamp. This detail is
essential: sorting all timestamps into the same member order creates an artificial
perfect temporal copula and invalid aggregate forecasts.

### 2. Hierarchical conditioning and fallback

The process starts with coherent 1-minute paths and considers parents bottom-up.
At each edge, a parent ensemble is rank-coupled to the current child sum. A
validation-selected weight $w$ forms the target

$$
T = (1-w)\sum_i C_i + wP.
$$

Only immediate children are adjusted to $T$; their inner subtrees are preserved.
Candidate weights are `0, 0.25, 0.5, 0.75, 1`. Weight zero is the fallback.
A nonzero weight must improve every chronological architecture subfold.

Return selected full parent conditioning. Variance and activity selected a
resolution-preserving copula-only mode: parent conditioning determines temporal
member ranks, but every minute is mapped back to the validated 1-minute marginal.
This retains useful clustering without allowing coarse conditioning to distort the
1-second distribution.

Against the coherent bottom-up 1-minute fallback, untouched joint energy-score
skill is positive for all targets:

| Target | Joint energy skill |
| --- | ---: |
| Return | +0.103% |
| Integrated variance | +0.456% |
| Active seconds | +0.197% |

Marginal CRPS versus that coherent fallback is no worse by more than 1% at any
tested level. The separately fitted scale-specific forecasts remain in the report
as a diagnostic oracle, but they cannot all be realized by one candle path and are
not the correct deployable baseline.

### 3. Long-memory volatility copula

The calibrated minute variance marginal was adequate, but its member ordering had
too little persistence. A fitted AR-factor volatility copula is blended into the
minute member ranks without changing any per-minute values. The selected blend
weight is 0.75.

On the untouched test, mean absolute error of minute log-variance ACF at lags
1, 5, 15, 60, 300, and 900 minutes is 0.0314. For example:

| Lag | Observed | Generated |
| --- | ---: | ---: |
| 1m | 0.457 | 0.537 |
| 5m | 0.385 | 0.389 |
| 1h | 0.330 | 0.290 |
| 5h | 0.234 | 0.245 |
| 15h | 0.214 | 0.238 |

### 4. Cross-feature path coupling

Return, variance, and activity marginals are forecast separately, so arbitrary
member pairing can produce rare infeasible triples. Whole-day variance and
activity paths are assigned to return paths with a forecast-only Hungarian
matching step. It permutes complete generated paths; no marginal or temporal path
is altered.

The remaining feasibility repair adds 209.9 bps squared over 9,503,722.7 bps
squared of forecast variance, or 0.00221%. A very small-variance edge case still
has a large multiplicative ratio, so both the ratio and the economically relevant
aggregate addition are retained in the report.

### 5. One-second generation

For every generated minute, continuous activity is balanced-rounded to an integer
$N \in [0,60]$. Let the minute targets be return $R$, integrated variance $Q$,
and activity $N$. The generator creates a fresh activity mask and fresh signed
magnitudes, then projects the active vector $x$ so that

$$
\sum_{j=1}^{60} x_j = R,
\qquad
\sum_{j=1}^{60} x_j^2 = Q,
\qquad
\sum_{j=1}^{60} \mathbf{1}[x_j \ne 0] = N.
$$

The fitted second kernel contains:

- clustered activity timing;
- state-conditional micro-return probability;
- state-conditional magnitude concentration;
- Gaussian-AR sign timing, selected at rho 0.65;
- persistent normalized magnitude shares, selected at rho 0.999.

The last two parameters are chosen on 14 days immediately before the untouched
test. Magnitude persistence is calibrated with generated hierarchical targets,
not realized minute targets, so calibration includes attenuation from the actual
feasibility and exact-moment projection workflow.

Every slower return is then a direct sum of these generated 1-second returns.

## Untouched distribution results

The return-histogram comparison uses the real returns from the final 91 days and
all generated ensemble members at each scale.

| Scale | Independent JS bits | Hierarchical coherent JS bits |
| --- | ---: | ---: |
| 1s | 0.003930 | 0.002047 |
| 1m | 0.026870 | 0.013766 |
| 15m | 0.006123 | 0.005951 |
| 1h | 0.013710 | 0.014376 |
| 4h | 0.028792 | 0.025043 |
| 1d | 0.084080 | 0.078866 |

Daily JS is noisy because the untouched sample contains only 91 realized daily
returns. It is reported rather than used as a hard acceptance gate.

The generated zero-gap survival is also close to the holdout: probability of a
gap lasting at least 5 seconds is 10.05% generated versus 9.30% observed; at
least 15 seconds is 0.131% versus 0.147%.

## What is reliable, and what is not

Supported by this experiment:

- the process can generate fresh coherent candles all the way to 1 second;
- its 1-second marginal, zero-delay behavior, signed dependence, activity
  dependence, and absolute-return dependence pass the specified test gates;
- hierarchy conditioning improves joint forecasts over the deployable coherent
  bottom-up baseline without sacrificing any scale by more than the 1% CRPS gate;
- all target constraints and aggregation identities are exact to floating-point
  tolerance.

Not established:

- the realized next path can be predicted candle by candle;
- the same calibration survives an arbitrary future regime;
- the generated paths yield a profitable strategy after fees and slippage;
- one 91-day BTCUSDT confirmation period is sufficient for universal claims.

The appropriate next audit is repeated walk-forward deployment: freeze all rules,
advance the cutoff, refit using only newly available history, and accumulate the
same gate vector over multiple future blocks and other liquid symbols.

## Reproduction

```powershell
npm run analysis:full-hierarchical-one-second:test
npm run analysis:full-hierarchical-one-second
```

The evaluator caches the minute history, selected calibrated forecast ensembles,
and causal one-second fit under `data/benchmarks/` so reconciliation iterations do
not repeatedly scan all 158 million source candles.
