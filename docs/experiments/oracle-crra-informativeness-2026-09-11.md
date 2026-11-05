# Forecast informativeness, fees and horizon comparison

With gamma zero, the forecasts produce changing trading decisions when fees
are removed. At the configured fees, their predicted gains are too small to
justify entry in the tested menus. Training direct forecasts over 30 and 120
seconds increased predicted move sizes, but did not clear the economic gate.

This is a completed negative result for paid trading under these forecasts,
inputs, horizons and execution assumptions. A requirement to predict profitable
trades would not make the underlying price process more predictable.

## Requirements proved

For the full observable history and account state `h`, feasible action `a`, and
feasible reference action `a0`, define the expected terminal wealth difference
using Oracle execution and optimal observable continuation:

\[
\Delta(h,a)=Q(h,a)-Q(h,a_0).
\]

The reference is optimal exactly when all feasible differences are nonpositive.
A strictly positive difference forces an optimal policy to choose another action.
Different strict winners across contexts force nonconstant decisions; changing
decisions through arbitrary tie-breaking is insufficient. Constant decisions
can nevertheless be correct when the optimal-action sets share a common member.

If each of the two estimated action values is within `epsilon` of its true value,
an estimated difference greater than `2 epsilon` guarantees true improvement.
The theorem requires the error bounds; this experiment does not establish a
uniform bound from a finite, temporally dependent validation sample.

For a surviving long entry from cash followed by a valid full close, constant
funding factor `k`, and entry/exit fees `f_in`, `f_out`, the criterion specializes to

\[
\frac{\mathbb E[P_{t+H}\mid h]}{P_t}>
\frac{1}{(1-f_{in})k(1-f_{out})}.
\]

At 5 basis points per side and unit funding, the required expected gross return
is **10.007505 basis points**. Leverage scales the gain and cost in this formula;
it does not lower the threshold. With maintenance, liquidation, pending limits
or an incomplete capped close, use the full action difference instead of assuming
this special formula applies.

Thirteen new results are in
[PredictionDecisionConditions.lean](../theory/lean/TradingTheory/PredictionDecisionConditions.lean).
The assumptions and training implications are explained in
[Decision-informative distributions](../theory/Decision-informative%20distributions.md).

## Identical forecasts, different fees

The old validation-selected model's finite forecast trees were held fixed. The
first 600 one-second observations, starting at 2026-08-10 00:00:01 UTC, were replayed
with gamma zero and one-step planning. All evaluated forecast hashes and action
menus match across fee cases. Both maker and taker fees were scaled together.

| Fee scale | Taker fee per side, bps | Return | Market trades | Limit placements / fills | Direction changes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | +0.718623% | 306 | 0 / 0 | 52 |
| 0.01 | 0.05 | +0.719425% | 87 | 3 / 2 | 22 |
| 0.1 | 0.5 | 0% | 0 | 0 / 0 | 0 |
| 1 | 5 | 0% | 0 | 0 / 0 | 0 |

The fee-free run included 435 long and 165 short decisions. The two paid cases
at scales 0.1 and 1 stayed in cash. The smallest positive fee changed trading
decisions and slightly improved the realized result on this short prefix; this
is not a monotonicity claim about realized performance or a test-selected model.

A separate first-minute, two-step comparison also traded with zero fees
(+0.068547%, 34 market trades) and stayed in cash at configured fees. It retains
the original observable tree, including its approximation for future features.

The full menu includes hold, exposure targets `[-2,-1,0,1,2]`, one-tick passive
limit orders, and cancellation choices. Research parameters are taker 5 bps,
maker 2 bps before scaling, entry bounds `[-2,2]`, maintenance `[-3,3]`, quote
order bounds `[5,10000]`, price step `0.01`, one pending order, and unit funding.
Only the both-zero fee diagnostic is newly allowed; positive equal maker/taker
fees remain invalid.

Terminal wealth includes any residual after a valid capped close. In particular,
the fee-free ten-minute run retains approximately `-0.1559231464` asset units.
Its reported wealth is not a completely cash-liquidated account.

## Training and validation

Six candidates were trained: horizons 1, 30 and 120 seconds, each with NLL alone
and NLL plus a normalized squared error for expected **price return**. All six
used the same validation selection score: NLL plus train-normalized price-return
MSE. No checkpoint or horizon was selected using test PnL.

Inputs remain 178 causal features. Targets use endpoint price ratios and preserve
the exact-zero atom. The distribution uses a triangular knot density in Gaussian
CDF coordinates of log return. Its exponential moments are finite, so expected
price exists. The previous continuous density's polynomial log-return tails did
not satisfy that requirement; the old finite-tree replays themselves remain
well-defined finite models.

The auxiliary loss improves estimation of the moment relevant to gamma zero
without imposing positive expected returns or a minimum trading frequency.

| Horizon | Selected mean-loss weight | Validation price-MSE skill versus zero forecast | Largest predicted validation mean, bps | Paid long/cash switching gate |
| --- | ---: | ---: | ---: | --- |
| 1 second | 1 | 2.909% | 0.2718 | Failed |
| 30 seconds | 0 | 1.288% | 1.1179 | Failed |
| 120 seconds | 0 | 0.580% | 1.5297 | Failed |

Economic validation uses nonoverlapping horizon labels: 145,437, 4,847 and 1,211
contexts respectively. None of the selected models clears even the lower
2.0003-bps round-trip threshold corresponding to 1 bp per side. Every model has
both long and cash recommendations at zero fees. Conditional realized returns
when forecasting a positive mean are +0.04583, +0.32367 and +0.43840 bps,
respectively; these are sample conditional means, not compounded returns or
certified conditional expectations.

Each `validation-economic-gate.json` explicitly records
`selectedModelGatePassed: false` and the absence of a uniform value-error bound.
The separate audit reads only validation data and does not change model
selection. It was added after some already selected candidates had been test
scored; it is not described as a preregistered test-blind acceptance decision.
Future trainer runs persist these diagnostics before loading test data.

See [the ML methods, metrics and reproduction report](oracle-crra-direct-horizon-forecasts-2026-09-11.md).

## Controlled direct-horizon replay

Endpoint forecasts alone cannot describe intermediate liquidation, funding or
limit crossings in an unrestricted leveraged account. These comparisons use
nonnegative quote and asset balances, no pending orders, unit funding, entry and
maintenance bounds `[0,1]`, and the menu hold or target exposure `0,0.5,1`.
Quote caps, fees, price grid and residual-close handling remain active.

[OracleEndpointHold.lean](../theory/lean/TradingTheory/OracleEndpointHold.lean)
proves seven results: every positive intermediate mark survives under these
restrictions, a committed no-trade hold ends at the direct endpoint account,
and the same terminal function produces the same wealth. The actual replay
still applies every intermediate one-second price. It replans every `H` seconds
and executes only the selected net trade; it does not artificially close and
reopen at every horizon boundary.

All three first-hour comparisons cover the same 3,600 observed seconds, with
the same Gaussian-CDF forecast family, seven continuous bins plus zero, and
the same restricted action menu.

| Horizon | Zero-fee return | Zero-fee market trades | Zero-fee maximum drawdown | Configured-fee return / trades |
| --- | ---: | ---: | ---: | ---: |
| 1 second | +1.420836% | 1,168 | 0.093914% | 0% / 0 |
| 30 seconds | +0.483652% | 71 | 0.254869% | 0% / 0 |
| 120 seconds | +0.249392% | 15 | 0.210746% | 0% / 0 |

Buy-and-hold over this hour returns +0.129415% without fees and +0.029325% at
configured fees. The one-second zero-fee policy retains `0.0021867090` asset units
after its capped terminal close; the other two zero-fee runs finish flat.

The configured-fee comparisons were also run over all available held-out data.
They all stayed in cash: 151,732 one-second decisions, 5,057 thirty-second
decisions, and 1,264 two-minute decisions. The latter horizons omit incomplete
final intervals and therefore cover 151,710 and 151,680 observed seconds.

These experiments use a BTCUSDT **spot** one-second tape with synthetic Oracle
account rules, not actual USD-M funding, order-book execution or live trading.
Short fee-free profits do not establish profitable trading with actual costs.
No model passed the requested paid economic gate.

## Verification and reproduction

The Lean build and axiom audit pass for **1,102 theorem declarations**. The
TypeScript Oracle suites pass **44 tests**, the ML horizon suite passes **8**,
and both the server typecheck and focused endpoint adapter typecheck pass.

The exact checker passed **33 saved decision-certificate records and 108
execution fixtures** against the declared finite quantized forecast trees and
action menus. Continuous binning approximately
preserves the quadrature mean of price before price-grid rounding; this does not
certify continuous-law policy optimality, nonlinear capped-close payoffs, model
calibration, or the full-history optimum. The commitment to hold for `H` seconds
is also a restriction, not a theorem about optimal decision cadence.

All numerical results and comparison hashes are consolidated in
`data/backtests/stochastic-crra/informativeness-comparison/report.json`.
The script checks matching forecast hashes, menus, time intervals and validation
selection before combining the 33 saved decision-certificate records.

```powershell
npm run test:oracle-crra
npm run test:oracle-horizons
npm run typecheck -w @trading/server
npm run theory:check
node scripts/report-oracle-informativeness.mjs
node scripts/verify-oracle-crra.mjs --certificates data/backtests/stochastic-crra/informativeness-comparison/decisions.jsonl --output data/backtests/stochastic-crra/informativeness-comparison/lean-verification.json
```

Replay examples (after the ML report's training/export commands):

```powershell
npm run backtest:oracle-crra -- --forecasts data/backtests/stochastic-crra/test-first-hour-3bins.jsonl --horizon 1 --count 600 --fee-scale 0 --output data/backtests/stochastic-crra/ablation-h1-fees0 --certificates 1
npm run backtest:oracle-endpoint -- --forecasts data/backtests/oracle-crra-horizon-h30.jsonl --count 120 --fee-scale 0 --output data/backtests/stochastic-crra/endpoint-h30-fees0
```

Repeat the fee ablation with scales `0.01`, `0.1`, `1`. For the endpoint first
hour use `(horizon,count)` pairs `(1,3600)`, `(30,120)`, `(120,30)` and fee scales
`0,1`; omit `--count` for the full held-out replay. Outputs and checkpoints remain
research artifacts. These comparisons did not change the running bot provider
or enable trading.
