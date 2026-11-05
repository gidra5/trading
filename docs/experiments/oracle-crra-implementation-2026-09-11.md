# Stochastic Oracle CRRA bot: implementation and held-out replay

This report records the original gamma-two experiment. The bot now defaults
to gamma zero; see the [controlled comparison](oracle-crra-gamma0-2026-09-11.md).

The bot now solves a finite stochastic price tree by exact backward induction
with constant relative risk aversion 2, executes its selected full account
decision through Oracle's transition, and can request the trained price model
through a persistent Python worker. The tested configuration stayed in cash.
This experiment does not establish a profitable strategy or unrestricted
optimality over the market's unknown distribution.

## Backtest results

All intervals start at the beginning of the sealed test split, August 10, 2026,
00:00:01 UTC. Model selection uses validation likelihood, before test trading
outcomes. The one-hour interval and the full one-step diagnostic were selected
before their results. Initial wealth is 10,000 quote units.

| Replay | Observed decisions | Planning horizon | Bot terminal wealth | Bot return | Same-engine buy-and-hold return | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| First hour | 3,600 | 2 seconds | 10,000.00 | 0% | +0.029325% | 301.0 s |
| First hour, shorter-horizon diagnostic | 3,600 | 1 second | 10,000.00 | 0% | +0.029325% | 4.9 s |
| Full held-out interval, shorter-horizon diagnostic | 151,732 | 1 second | 10,000.00 | 0% | -2.165959% | 194.3 s |
| First ten minutes, fractional market menu | 600 | 2 seconds | 10,000.00 | 0% | +0.134093% | 106.3 s |

All four runs made zero market trades, zero limit placements/fills and had zero
drawdown or liquidations. The full test ends at August 11, 2026, 18:08:53 UTC,
after its last observed price move. It is **not** a full-interval two-step replay.
The buy-and-hold comparator enters at exposure one through the same fee model,
coasts through the same marks, and performs the same capped terminal close.

The conditional forecast improves distribution likelihood, but that improvement
does not produce a trade worth its costs within these very short planning
windows and menus. Cash being selected in these experiments says nothing about
longer horizons or larger action spaces.

The additional fractional-menu diagnostic includes market targets -0.5 and
+0.5 over the first 600 ticks. It still selects cash; this checks that omitting
those two targets did not explain the result on that fixed prefix. Its report
is `data/backtests/stochastic-crra/first-ten-minutes-fractional-menu/report.json`.

Reports and exact evaluated-input hashes are in:

- `data/backtests/stochastic-crra/first-hour/report.json`
- `data/backtests/stochastic-crra/first-hour-horizon1/report.json`
- `data/backtests/stochastic-crra/full-test-horizon1/report.json`

The first-hour forecast prefix SHA256 is
`1bbf163dbefdd79a1f857db1fb5131c98bf99208a1f7f33d6648d94334b9a1f3`.
The full-test prefix SHA256 is
`9ebd5abea00f9e50352c66dc0d24dc6b9cc2e21cecefe2a943fd76db6c401af6`.

## Distribution model

The strongest documented price-density architecture was retrained: four
512-wide GLU layers, 120 known return lags and 58 latest production features,
a 256-knot continuous density and a separate zero-return atom. There are
998,802 training ticks and 145,437 validation ticks. Both density coordinates
and feature normalization use training data only. The previous nonzero-only
target filter was removed.

Validation selected epoch 1, with mixed-law NLL **-5.041420**, ahead of the
train-fitted unconditional marginal's **-4.782361** and the initial empirical
baseline's **-4.798307**. The selected model's held-out NLL is **-5.921930**.
These results establish the winner among the rebuilt candidates, not every
possible prediction architecture. See
[the model report](stochastic-crra-price-forecast-2026-09-10.md) for training,
split definitions, checkpoint hash and forecast approximation details.

Each exported tree has two conditional steps, with three continuous-bin means
plus the exact-zero atom at each step. Rollout updates the 120 return lags and
freezes the other 58 currently observed features. It does not inject their
realized future values. Batch/device numerical differences can change forecast
probabilities; the exact policy operates on the concrete tree it receives.

## Execution and finite action contract

The new path is separate from `GridTradingBot` and the existing USD-M
backtester. Neither adapter implements the same transition as the proved
Oracle model, so their sizing, confidence controls and fill rules are not
applied to these decisions.

The research configuration is explicit:

| Parameter | Value |
| --- | --- |
| Received-amount market / limit fee | 0.0005 / 0.0002 |
| Entry raw exposure interval | [-2, 2] |
| Maintenance effective exposure interval | [-3, 3] |
| Reservation leverage | 2 |
| Physical order quote minimum / aggregate decision maximum | 5 / 10,000 |
| Price step | 0.01 |
| Simultaneously active limit orders | At most 1 |
| Market exposure targets | -2, -1, 0, 1, 2, plus unchanged holdings |
| New passive limits | Buy or sell one price tick away, notional half current equity, capped by the physical order maximum |
| Cancellation menu | Every subset of existing orders |
| Funding | Known unit signed-asset factor at each tick |

Market, cancellation and placement choices form a Cartesian menu. Exact
fee-aware market requests are capped by the remaining aggregate decision
budget; the resulting actual account is revalidated. Entry targeting is not
silently converted to a grid or multiplied by confidence. The public solver
accepts other declared menus; `--targets` changes the replay's market grid.

The model retains received-amount fees, signed-asset funding, effective-equity
maintenance and zero-account liquidation. Mature crossed limits fill before
cancellations, at their specified prices. Fills and market orders share one
decision budget. Pending orders reserve notional/leverage and share collateral
with existing positions. New orders are strictly passive and cannot fill in
their placement decision. Prices obey the declared grid. Maintenance-safe
coasting can remain outside the entry band when no actual trade occurs.

At the terminal mark, committed arrival fills happen first. Remaining orders
are canceled, then at most one valid nonovershooting close uses the remaining
decision capacity. Uncloseable inventory remains and is valued at the raw
mark. This is an explicit terminal functional, not a promise of complete
cash liquidation under order constraints. As in the Lean closing formulation,
the terminal close is governed by its own valid-order contract rather than
requiring a capped residual to re-enter the entry exposure band.

The source tape is **BTCUSDT spot one-second closes**. These parameters define
a synthetic Oracle replay; they are not claimed to be actual historical
Binance USD-M fee tiers, funding payments, order-book fills or maintenance
tiers. Actual non-unit Oracle funding is covered by execution tests and proof
fixtures, but is not present in this market-data replay.

## What is verified

The policy uses canonical BigInt rationals for execution, probabilities,
terminal utility, expectations and comparison. Floating point is restricted
to the neural forecast inputs/outputs and display metrics. Price rounding is
an explicit forecast transformation. Branches that round to the same observed
price are merged into a conditional mixture **before** continuation is
optimized, preventing access to hidden scenario labels.

The TypeScript score is `-1 / wealth` for positive wealth, and negative
infinity for nonpositive wealth. Original Lean CRRA uses `1 - 1 / wealth`.
`CRRANormalization.lean` proves the exact +1 correspondence, normalized
expectation correspondence, and invariance of argmax, tie selection and
certificates, including ruin. Zero probability does not turn an impossible
ruin branch into negative-infinity expectation.

Verification completed:

- **1,076 public Lean results** pass the axiom audit. This includes rational
  CRRA/Bellman selection, the utility-normalization bridge, rational Oracle
  primitive refinement and the original-engine limit trace.
- **108 exact TypeScript execution fixtures** pass the Lean checker against
  the refined Oracle primitives.
- A literal complete limit-order trace is checked both by original Lean
  execution/admissibility proofs and a TypeScript regression, including
  aggregate-cap rejection. It is not merely two copies of the TS engine
  agreeing with each other.
- **Three sampled two-step decisions**, including all their supplied
  candidate continuations, pass the independent Lean certificate checker.
- **39 TypeScript tests**, **8 Python model/worker tests**, the focused Oracle
  package build and the **full server typecheck** pass. Actual persistent
  Node-to-Python inference also matches the standalone worker output.

`ExactCRRACheck.lean` rejects nonoptimal root or child choices, invalid
probabilities and duplicate action identifiers. It checks every candidate
provided in a certificate. It does **not** prove that the producer included
all feasible actions or that supplied terminal wealth was produced by the
TypeScript engine. Primitive refinement theorems and differential fixtures
provide a separate execution check; finite tests are not a universal proof
about the TypeScript interpreter, JSON parser or pending-order implementation.

The Bellman guarantee is therefore exact **for the supplied finite observable
tree and supplied feasible menu**, under its execution/refinement assumptions.
It is not an unrestricted continuous-exposure theorem. Replanning a two-step
window every second also does not establish global optimality over the whole
backtest interval. Finally, bin means preserve the forecast mean numerically,
but not nonlinear CRRA utility or liquidation tail probabilities; the learned
continuous law and the actual market law are not certified by these finite
tree certificates.

## Code and use

- `packages/bot-algo/src/oracle-rational.ts`: exact arithmetic.
- `packages/bot-algo/src/oracle-execution.ts`: full Oracle account phases.
- `packages/bot-algo/src/oracle-crra-policy.ts`: observable-tree construction,
  finite menu, Bellman optimization and certificate generation.
- `packages/bot-algo/src/oracle-crra-bot.ts`: exact account/tick state machine.
- `apps/server/src/oracle-forecast-provider.ts`: persistent inference worker.
- `apps/server/src/bot.ts`: bot factory and serial stochastic controller.
- `ml/serve_stochastic_price_forecast.py`: trained model inference protocol.

`OracleForecastProvider.predict({observedPrice, features})` accepts exactly 178
finite, currently known features. Their order is the trained contract: 120
oldest-to-newest log-return lags, then channels 1 through 58 of the production
feature basis at the current tick. The trained normalizer is inside the model.
This controller requires that observed-feature stream; it does not fabricate
the missing features from a spot price alone.

Create an `OracleCrraBot` with the full initial account, rules and known
funding schedule, pair it with an `OracleForecastProvider` in
`StochasticOracleController`, and call `onTick` for each completed observation.
Call `finish` at the next terminal mark to perform the specified close and
stop inference. Failed inference can be retried before executing the tick;
duplicate ticks and skipped observations are rejected. The server HTTP
start/stop controls remain the user's existing unconnected interface; this
change does not activate live Binance trading.

With dependencies, the retained production dataset and candle files present,
the experiment can be reproduced from the repository root with:

```sh
npm run experiment:oracle-crra -- --gamma 2
```

It reuses the checkpoint if present, trains if it is missing, compares the
distribution candidates by validation likelihood, exports the selected model,
runs the first-hour replay, and verifies execution fixtures and sampled
decision certificates. The ML environment can be installed with
`node scripts/bootstrap-ml.mjs`; Lean setup is handled by the experiment if
its local cache is missing. Model training/export currently requires CUDA;
the persistent inference worker also supports CPU.

Individual commands:

```sh
npm run test:oracle-crra
npm run typecheck -w @trading/server
node scripts/run-ml-python.mjs -m unittest discover -s ml -p 'test*stochastic_price_forecast.py'
npm run theory:check
npm run backtest:oracle-crra -- --gamma 2 --output data/backtests/stochastic-crra/first-hour
npm run verify:oracle-crra -- --certificates data/backtests/stochastic-crra/first-hour/decisions.jsonl
npm run backtest:oracle-crra -- --gamma 2 --forecasts data/backtests/stochastic-crra/test-full-3bins.jsonl --count 151732 --horizon 1 --certificates 0 --output data/backtests/stochastic-crra/full-test-horizon1
```

The JSON reports and checkpoints are local generated artifacts. The proof
sources, test fixtures, inference provider and reproduction scripts are
workspace source files.
