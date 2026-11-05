# Five- and fifteen-minute Oracle policy comparison

The longer horizons contain price moves large enough to cover the configured
fees, but the selected models do not predict a sufficiently large conditional
mean. Both paid policies stayed in cash throughout the same 42-hour test window.
The corresponding fee-free policies traded and lost money over that window.

## Forecasts versus realized opportunities

The 10.007505-bps diagnostic is a threshold for **conditional expected gross
return**, for a surviving cash-starting long with a valid full close, unit funding
and 5-bps entry/exit fees. It is not a threshold on volatility or on a move that
can only be identified after it happens. The actual replay uses Oracle's capped
close and residual valuation when evaluating its finite action menu.

| Horizon | Positive realized test moves above 10.0075 bps | Largest predicted validation mean | Largest predicted mean across all test origins |
| --- | ---: | ---: | ---: |
| 5 minutes | 30 / 504 (5.95%) | 1.5909 bps | 2.2144 bps |
| 15 minutes | 19 / 168 (11.31%) | 2.3680 bps | 3.8429 bps |

Realized counts use the common 42-hour nonoverlapping replay intervals. The last
column includes every one-second forecast origin with a complete within-split
target, including intervals beyond the aligned replay end. Even that broader
search has no predicted mean above the configured threshold. It is a diagnostic;
it does not change the selected checkpoint or trading cadence.

The median absolute realized test move is 3.52 bps at five minutes and 6.75 bps at
fifteen minutes; the 90th percentiles are 12.27 and 21.16 bps. On validation,
positive threshold crossings occur in 10.74% and 20.50% of intervals. Larger
realized moves therefore exist at the longer horizon, but their size alone does
not establish predictable direction or a positive expected net payoff.

## Training and selection

Four fresh models were trained with horizons 300 and 900 seconds and auxiliary
mean-loss weights 0 and 1. Each was compared with a train-fitted unconditional
density. All checkpoint/model decisions use validation NLL plus train-normalized
price-return MSE. Both selected the conditional GLU with auxiliary weight 1 at
epoch 1. Test results did not select a candidate.

Inputs remain the same 178 causal features: 120 past/current one-second return
lags and 58 other current observed features. Gaussian-CDF coordinates preserve
finite exponential price moments. Targets use actual endpoint price ratios,
preserve exact zeros, and never cross a chronological split boundary. This
experiment extends the prediction horizon; it does not extend the raw history
input or test every possible model for those horizons.

Selected mean-price MSE skill relative to predicting zero return is +0.493% and
-0.063% on validation, and +0.259% and -0.738% on test, for five and fifteen
minutes respectively. The fifteen-minute selected model is worse than the zero
mean baseline on that metric, despite having the best declared composite
validation score. Longer-horizon prediction has not reliably improved the
available signal in this experiment.

Both saved paid switching gates are false. At the lower diagnostic fee of 1 bp
per side, the fifteen-minute model crosses the 2.0003-bps threshold in four
validation contexts, with a sample mean net return of +2.90 bps. That small
sample is not a calibration guarantee and was not used to choose the model.

Detailed candidate metrics and training commands are in
[the forecast report](oracle-crra-5m15m-forecasts-2026-09-11.md).

## Exact account replay

All four replays cover **2026-08-10 00:00:01 UTC through 2026-08-11 18:00:01 UTC**:
151,200 observed seconds, with 504 five-minute or 168 fifteen-minute decisions.
The five-minute export has an additional complete interval, which is excluded
so that both horizons see precisely the same price period.

| Horizon | Fee scale | Terminal wealth from 10,000 | Return | Market trades | Maximum drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5 minutes | 0 | 9,937.6894 | -0.623106% | 240 | 1.387730% |
| 5 minutes | 1 | 10,000 | 0% | 0 | 0% |
| 15 minutes | 0 | 9,865.8601 | -1.341399% | 84 | 1.774541% |
| 15 minutes | 1 | 10,000 | 0% | 0 | 0% |

Buy-and-hold returns -1.978872% without fees and -2.076869% with configured fees.
The fee-free strategies outperform that declining-market benchmark but
underperform cash. All four policy runs finish without residual inventory.
These 42-hour results should not be compared directly with earlier one-hour
fee-free results as if only the horizon had changed.

The replay uses gamma zero, nonnegative quote and asset balances, entry and
maintenance bounds `[0,1]`, unit funding, no pending orders, and hold or exposure
targets `0,0.5,1`. Quote trade bounds `[5,10000]`, the `0.01` price grid and valid
capped terminal close are active. Every actual intermediate one-second mark is
applied. Trades at the next decision execute only the selected net adjustment;
there is no artificial close and reopen at every horizon boundary.

The existing endpoint-hold theorem applies under these restrictions. Endpoint
forecasts do not describe intermediate liquidation or limit-order fills for an
unrestricted leveraged account. The tape is BTCUSDT spot, with synthetic Oracle
account parameters, rather than an actual USD-M funding and order-book replay.

## Verification and artifacts

`comparison-plan.json` was recorded before these backtests. The report assembler
verifies both stages of validation selection, actual checkpoint bytes, actual
consumed forecast-prefix hashes, manifest paths and horizons, and the matching
42-hour periods. All four comparisons passed those checks.

The existing Lean endpoint theorem is generic in the number of intermediate
marks; no special 300- or 900-second proof is needed. Twelve saved decision
certificates (the first three from each replay) and 108 execution fixtures all
passed the exact Lean checker. They check finite quantized
scenario/menu decisions, not every realized decision, model calibration or the
unrestricted continuous-law optimum.

Artifacts are under `data/backtests/oracle-horizons-5m15m`:

- `comparison-plan.json` and `comparison.json` record the plan and measured results.
- `h300-fees0`, `h300-fees1`, `h900-fees0`, `h900-fees1` contain individual replays.
- `decisions.jsonl` and `lean-verification.json` contain the sampled proof checks.
- Full endpoint exports retain all actual intermediate prices.

The independent realized-movement audit is
`data/backtests/stochastic-crra/h300-h900-realized-moves.json`; its hindsight
conditional averages are explicitly noncausal and never used as predictions.

After the forecast report's training/export commands, reproduce a replay with:

```powershell
npm run backtest:oracle-endpoint -- --forecasts data/backtests/oracle-horizons-5m15m/oracle-crra-horizon-h300.jsonl --count 504 --fee-scale 1 --output data/backtests/oracle-horizons-5m15m/h300-fees1
```

Use `h900`, count `168`, and the matching output for fifteen minutes; repeat both
with `--fee-scale 0` for the diagnostic. Then assemble and verify:

```powershell
node scripts/report-oracle-horizon-comparison.mjs
node scripts/verify-oracle-crra.mjs --certificates data/backtests/oracle-horizons-5m15m/decisions.jsonl --output data/backtests/oracle-horizons-5m15m/lean-verification.json
node scripts/run-ml-python.mjs ml/audit_horizon_realized_moves.py
```

No shared forecast/execution code, running provider or live trading settings
were changed. The new selected checkpoints remain available under
`data/training/runs/stochastic-crra-direct-horizons-5m15m-v1`.
