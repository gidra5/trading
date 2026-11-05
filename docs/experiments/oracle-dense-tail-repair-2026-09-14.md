# Checked backward repair of dense Bellman fields

## Completed 90-minute replay

`data/backtests/oracle-dense-repair-terminal-90m` completed all 5,400 one-second observations in **9,215.32 seconds (2h 33m 35s)**, including initial construction, periodic full refreshes and ten independent full-field reference sweeps. It finished flat at **$10,000: 0% return, zero trades and zero fills**. This is a completed execution evaluation, but it does not demonstrate a profitable learned policy.

The independent ledger audit checked all 5,400 transitions and terminal closing; all 5,401 initial/observed prices matched the immutable candle source exactly. The source remains BTCUSDT **spot** one-second closes with the stated Oracle rules and unit funding, not an exchange futures execution replay.

Ten reference sweeps found zero feasibility disagreements and exactly matching root fields. The largest interior discrepancy was `7.157107314137434e-6` log units, within its recorded allowance. This isolates numerical reuse error under the same finite operator; it does not bound account-grid, action-menu, market-projection or forecast errors.

| Workload | Updates | Mean seconds | Median seconds |
| --- | ---: | ---: | ---: |
| Rolling one-hour repair | 1,770 | 2.41944 | 2.42085 |
| Scheduled full refresh | 30 | 88.14655 | 88.00428 |
| Countdown to the fixed endpoint | 3,599 | 0.41085 | 0.40793 |

The entire replay's median update was 0.46550 seconds, but that figure is dominated by the cheaper countdown. It is not a real-time throughput claim for a perpetually rolling one-hour planner.

Predefined policies on the same prices and execution rules returned **0% cash**, **+0.035678% half-long**, and **-0.135764% half-short**, each with a valid final flat close. The half-long baseline paid approximately $5.0043 total entry/exit fees. These are fixed baselines, not hindsight-selected policies.

Authoritative artifacts: `results.json`, `summary.json`, `ledger-audit.json`, `baselines.json` and `comparisons.json` in that directory. The earlier partial observations below are retained as experiment history.

## Shared stateful controller

`ml/oracle_dense_controller.py` now owns arrival, funding, committed fills, TD-error recording, market rolling, value repair, action selection and strict endpoint closing. The backtest driver calls this controller. It rejects skipped/duplicate seconds, inconsistent observation pairs and off-tick prices before execution. A numerical failure after arrival stops the controller so a retry cannot charge funding or execute fills twice. An invalid full close retains the remaining position and reports failure.

Seven CPU tests exercise horizon countdown, exact terminal closing, funding, pending fills sharing the terminal quote cap, retained dust, history validation, failure handling and restoration. A four-second GPU replay through this controller exactly matched the previous driver in the complete one-hour value field, front proposals, sampled market cloud, error statistics and realized decisions (`oracle-dense-controller-equivalence/equivalence.json`).

`ml/serve_oracle_dense.py` provides a persistent JSONL worker, and `apps/server/src/oracle-dense-controller.ts` provides the server paper adapter, exported from `bot.ts`. Initialization receives two observed seconds, an already-arrived account, its consumed quote budget and a fixed terminal timestamp. Subsequent requests contain only the next observed pair. The Python controller owns account execution; the server receives the executed state and must not apply the same trade again. Orders are carried as a list without assigning predictive significance to their slot index. The worker uses a fixed checkpoint hash and currently supports the log-utility numerical planner.

Six Python worker tests and seven server protocol tests pass, covering arrival budgets, serial decisions, malformed responses, timeouts, duplicate requests and terminal-state checks. Server TypeScript typechecking passes. A real CPU NSF worker was also exercised through the server adapter for a two-tick synthetic flat continuation of a saved observed state: initialization, both updates and final flat closing completed in 3.04 seconds (`oracle-dense-server-worker-smoke/results.json`). This is an integration smoke test, not a market backtest or GPU throughput measurement. Checkpoint persistence and dedicated paper API controls are now implemented and tested; see [worker runtime and deployment](oracle-dense-worker-runtime-2026-09-14.md). No exchange orders are submitted.

The worker config in `data/backtests/oracle-dense-server-worker-config.json` matches the coarse replay's one-hour model, grid and execution parameters. Start it directly with the following command; stdout is reserved for JSONL responses. Initialization can take a full sweep, so the server's default initialization timeout is 20 minutes and its per-tick timeout is three minutes. These are failure limits, not promises of real-time throughput.

```powershell
.venv-ml/Scripts/python.exe -u ml/serve_oracle_dense.py --config data/backtests/oracle-dense-server-worker-config.json
node --conditions=development --import tsx apps/server/scripts/check-oracle-dense-worker.ts .venv-ml/Scripts/python.exe data/backtests/oracle-dense-server-worker-smoke/config.json data/backtests/oracle-dense-server-worker-smoke/input.json data/backtests/oracle-dense-server-worker-smoke/results.json
```

The server adapter accepts `initialize({observation, account, terminalSeconds, consumedQuote, filled})`, then `onTick(observation)`. Each observation has `features: [previous59, current59]`, `prices: [previous, current]` and `timestamps: [previousUnixSeconds, currentUnixSeconds]`. Initial account fields are quote, asset, previousPrice and orders; orders have side, price and notional. `consumedQuote` is the aggregate quote already used by this tick's committed fills, and `filled` records whether such fills occurred. Returned accounts already include the chosen paper trades. The initial horizon must fit before the specified terminal time; final closing happens automatically on that last observed tick.

The separate GPU restart test also passed: two updates plus two resumed updates matched four uninterrupted updates exactly, including the complete value field, market cloud, front and residual statistics (`oracle-dense-resume-equivalence/results.json`). That test preceded controller extraction; the new controller has an additional CPU restoration check and GPU continuation check.

**Latest correction:** the terminal-close guard now also applies inside the backward sweep, and rolling repair excludes both operators affected by moving the terminal date. The earlier 90-minute attempt was stopped after 349 observed seconds. Its partial metrics are preserved, but it is not a completed backtest. A corrected run is in `data/backtests/oracle-dense-repair-terminal-90m`.

The solver can now repair the new tail of a rolling market map, reuse an unchanged earlier prefix, and recompute the actual root. The implementation records residuals from earlier repair boundaries instead of treating every copied field as exact. It uses all conditional samples in the repaired layers; this is distinct from the paired Monte Carlo estimator.

## Conditions and error accounting

Repair checks the account grid and execution parameters, the absolute-time candidate schedule, and equality of the retained interior market nodes and raw conditional rows. A subsampled field without residual accounting cannot initialize repair. Each candidate stopping boundary must have the same finite/nonfinite support as its aligned previous field.

Let `b` be the newly recomputed boundary and `delta` the maximum difference between its field and the previous field at the same absolute time. Layers `1..b-1` are copied, while the actual root is recomputed. Their previous local Bellman residual bounds are shifted with the copied values. At the boundary immediately preceding `b`, `delta` is added to the old residual. Residuals in the recomputed tail are removed. The stopping allowance includes all retained residuals, the new boundary difference and a double-precision roundoff allowance.

The real-arithmetic reasoning is that positive interpolation with total weight at most one, conditional probability averaging and maximization do not amplify a uniform value error on common feasible support. Local Bellman residuals accumulate through the remaining prefix. A greedy action selected with uniformly inaccurate scores has at most twice the score error as its gap within the compared action set.

`TradingTheory/BellmanResidualRepair.lean` formalizes positive-interpolation error, residual propagation, the copied-boundary residual and the greedy score bound. The module is imported and its four public theorems are included in `Audit.lean`. **`npm run theory:check` passed**, including the axiom audit. The Lean result is conditional on the specified support and operator properties; it is not a machine proof of Triton, the floating-point allowance, market quantization or continuous-action coverage.

The numerical allowance uses an explicit float64 operation/magnitude margin. It is deliberately reported as an allowance rather than a machine-verified interval enclosure. Full recomputation checks below test it against the implemented finite-grid solver. Market-law approximation, account interpolation and omitted continuous actions remain separate errors.

## Implementation and execution correction

`oracle_dense_repair.py` owns the alignment checks and residual accounting. `DenseValueMap.sweep(..., estimator='repair')` performs full conditional backups until it can reuse a prefix within the requested tolerance. Saved `bellman.json` and `bellman-residuals.npy` preserve the schedule and residual history across reloads. Copied dispersion estimates are explicitly stale scheduling statistics, not newly measured Monte Carlo variance.

For rolling replay, maps can live in RAM and be archived explicitly. Only two maps are retained, avoiding a multi-gigabyte archive at every observed second. The GPU work list reuses previously feasible account representatives and checks the newly introduced parent prices, so it cannot omit an account cell that becomes feasible at a new root or tail parent.

When the remaining backtest interval becomes shorter than one hour, the terminal date stays fixed. The market updater predicts the new root without appending a tail. The unchanged interior value field can be shifted directly, while preserving any earlier residual allowance. The final step requires a valid full close and cancellation; an uncloseable residual is reported as a terminal failure, not erased.

A further execution defect was corrected before rolling evaluation: an interpolated finite value could previously mask liquidation on an actual conditional next-price sample. Both CPU and GPU post-state evaluation now reject that action. All arrival outcomes are checked even if an earlier outcome cannot close immediately. Failure to close immediately can still use a longer continuation; invalid arrival cannot.

## Accuracy and timing

The initial two-update experiment `oracle-dense-tail-repair` reduced updates to 5.78 and 5.14 seconds, with identical root scores to its saved full references. Those measurements preceded the additional arrival-feasibility correction and remain historical; their execution/value archives are rejected by the newer kernel hash check.

The corrected rolling smoke test is `data/backtests/oracle-dense-repair-smoke`. It replayed 30 actual seconds of a planned 90-minute interval, using a 3,600-step forecast at every decision and a normalized log-value allowance of `1e-6` (0.01 bps). Initial full construction took 90.37 seconds. Median actual-update time was **5.10 seconds**, with a **6.44-second** maximum. The bot remained at $10,000 cash; the interval was deliberately stopped early, so it has no terminal-return result.

| Actual update | Maximum full-field log error | Maximum root error | Recorded allowance | Feasibility disagreements |
| --- | ---: | ---: | ---: | ---: |
| 15 | 6.2561e-7 | 0 | 9.1494e-7 | 0 |
| 30 | 6.2836e-7 | 0 | 9.1427e-7 | 0 |

The full references used the same finite market rows and candidate schedules. They took about 87 seconds each. These checks support the reuse implementation; they do not establish optimality relative to the underlying continuous NSF law.

**58 Python tests passed** in the final regression run. Coverage includes repeated nonzero repair seams, refusal of changed transitions or schedules, refusal of an unaccounted subsampled parent, in-memory export/reload, newly feasible account representatives, fixed-terminal shrinking to one step, nonzero residuals during shrinking, actual-arrival liquidation, and the existing funding/fees/order-cap/limit-order execution comparisons.

## Longer evaluation

`backtest_oracle_dense_repair.py` now provides a causal realized-price replay with fresh NSF root laws every second, full reference checks, terminal handling and a wall-clock cap. A 90-minute evaluation was launched in `data/backtests/oracle-dense-repair-90m` with a `1e-5` log-value allowance (0.1 bps), a full value refresh every 60 actual seconds while the horizon is rolling, and separate full-reference checks every 601 seconds. Initial values were loaded from the verified complete smoke-test initial map. The run has a three-hour wall-clock cap; terminal return is reported only if it completes the valid final close.

The first 59 actual updates took about 148 seconds, before the first scheduled full refresh. This is early runtime evidence, not the final backtest result. `progress.json`, `updates.jsonl` and the eventual `results.json` are the authoritative run records.

That original run was subsequently interrupted at 349 flushed updates / 1,270.85 wall seconds after the short exact-account audit found the sweep defect below. It remained at $10,000 cash throughout, with zero fills, orders or market trades. Ordinary updates had a 2.42-second median; five full refreshes averaged 87.88 seconds. These are partial measurements of the preceding implementation. `interruption.json` and `summary.json` record its status explicitly, with no terminal return.

```powershell
.venv-ml/Scripts/python.exe ml/backtest_oracle_dense_repair.py --market-source data/backtests/oracle-dense-raw-price-paired --initial-map data/backtests/oracle-dense-repair-smoke/initial --output data/backtests/oracle-dense-repair-90m-new --seconds 5400 --repair-tolerance 1e-5 --refresh-every 60 --reference-every 601 --max-wall-seconds 10800
```

## Exact terminal decisions and corrected evaluation

The observed root already suppressed interpolation at its last decision. The CPU and GPU backward sweeps did not: they could score a hypothetical final action using interpolated terminal values instead of its actual next-arrival closing value. A two-step, exact-account enumeration reproduced a discrepancy for an initial pending buy at 99.99, two first prices 99.97 / 101, and conditional final prices [99.83, 99.83] / [101.49, 99.98]. At a represented account state, the old backup differed by 9.0168e-8 log units, despite no account interpolation being necessary in this two-step backup. The regression failed before the fix and passes for both backends afterward.

The last future decision now uses exact arrivals and valid terminal closes for every candidate. Moving the terminal date changes both the terminal-arrival operator and the preceding last-decision operator. The repair boundary must therefore be at most `H-3`; allowing `H-2` copied a layer with a different operator and failed the repeated-residual regression. Fixed-terminal shrinking retains the aligned operators and remains supported. Saved maps now include hashes of the complete Bellman operator, and the repair contract version has changed, so old values cannot initialize the corrected solver.

**65 Python tests passed** after these changes, including the exact-account regression, GPU/CPU conformance, repeated nonzero repair seams, shrinking to one step, archive reloads, and reporting tests. The Lean nonexpansive-operator result is unchanged; the corrected boundary restores its same-operator premise in the implementation.

The corrected 90-minute replay starts from a fresh full sweep, which took 91.11 seconds (92.37 seconds including setup/export). It uses the same model, finite grid, 0.1-bps reuse allowance and three-hour wall limit. Extra full-reference checks at updates 1 and 30 verify the corrected rolling implementation before the regular 601-update checks. Complete account states are now logged at each decision, and reference checks are persisted immediately rather than only at completion.

The update-1 reference passed: maximum field error 7.1571e-6 log units, maximum root error zero, allowance 7.4618e-6, and zero feasibility disagreements. Full recomputation took 87.71 seconds. This is an early accuracy check, not the completed replay result.

Update 30 also passed: maximum field error 3.7824e-6, root error zero, allowance 9.4278e-6 and zero feasibility disagreements; reference time was 87.69 seconds. The run remained in cash through update 41 and was still running. Additional targeted archive-signature and interrupted-report checks passed after the full 65-test regression run.

The update-601 reference also passed: maximum field error 6.0757e-6, root error zero, allowance 6.3811e-6, no feasibility disagreements, and 87.74 seconds for the reference. The replay remained active and in cash through the subsequently checked update 646.

Update 1,202 passed with maximum field error 4.3243e-6, root error zero, allowance 4.6247e-6, no feasibility disagreements, and 87.72 seconds for the reference. A later process check confirmed the replay was still running; progress had reached update 1,253 / 5,220 wall seconds, still at $10,000 cash. These remain partial results. The separate ledger audit now checks 1,176 updates and all 1,177 associated prices against the immutable source, with exact price matches and no execution discrepancies.

After update 1,800, the horizon shrinks toward the unchanged replay endpoint. The update-1,803 full reference matched the reused field exactly: zero field error, zero root error and zero feasibility disagreements. The first 432 shrinking-horizon updates averaged 0.512 seconds, with a 0.500-second median and a 0.603-second maximum. Ordinary rolling-horizon repair had a 2.421-second median, while the 30 scheduled full refreshes averaged 88.147 seconds. These are different planning workloads; the cheaper countdown must not be described as the cost of perpetually extending a one-hour horizon. At the time of this report the replay had reached 2,232 updates and remained in cash, without a terminal return yet.

The finer 41-exposure GPU profile is queued behind the actual replay process, so the measurements do not compete for the GPU. It will start only after that process exits and produces `results.json`. Full-field and root-score agreement above concerns reuse under the same grid; account/action refinement is a separate accuracy check.

```powershell
.venv-ml/Scripts/python.exe ml/backtest_oracle_dense_repair.py --market-source data/backtests/oracle-dense-raw-price-paired --output data/backtests/oracle-dense-repair-terminal-90m-new --seconds 5400 --repair-tolerance 1e-5 --refresh-every 60 --reference-every 601 --reference-updates 1 30 --max-wall-seconds 10800
```

The separate [account interpolation audit](oracle-dense-account-interpolation-2026-09-14.md) measures a remaining approximation that repair accuracy does not cover. Small reuse error must not be interpreted as small total planning error.

A [separate market-projection comparison](oracle-market-projection-2026-09-14.md) evaluates matching retained nodes by the NSF latent posterior. It improves several conditional-distribution diagnostics, but has not yet established lower policy-value error. The ongoing replay's matching rule is unchanged.

## Independent realized-execution audit

`audit_oracle_dense_ledger.py` reconstructs signed funding, received-amount fees, crossing fills and market inventory changes using plain NumPy arithmetic rather than importing the planner's execution kernels. It checks remaining aggregate quote capacity, entry leverage, reservations, retained versus newly placed orders, passivity and tick alignment. On completion it reconstructs the terminal arrival and full close, rather than accepting the reported terminal wealth alone. Balance comparisons use explicit floating-point tolerances; this is a numerical ledger audit, not a machine proof.

Five focused tests pass, including a fee-aware round trip ending in cash, two same-side fills with funding, upward/flat crossing behavior, and rejection of missing fees, excess capacity, nonpassive placement and reused reserved equity. The audit passed the first 359 recorded actual updates. All 360 prices, including the initial price, matched the immutable source candles exactly at their recorded timestamps. That prefix remained entirely in cash, so its live ledger check does not exercise trading; the synthetic tests cover those transitions.

The actual price source is **BTCUSDT spot one-second candle closes**, with the declared Oracle fee/leverage/order rules and unit funding. This is not a replay of USD-M futures fills, exchange queues or depth. The checker does not establish forecast accuracy, optimality, or correctness of unlogged steps. It is rerunnable on the growing log and records its checked prefix in `ledger-audit.json`.

```powershell
.venv-ml/Scripts/python.exe ml/audit_oracle_dense_ledger.py data/backtests/oracle-dense-repair-terminal-90m --price-source data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s
```

## Resume a cleanly stopped replay

The replay driver now accepts `--resume` with a new output directory. It requires a finalized, incomplete result without an execution failure, a complete flushed log, a matching saved account/front, and a complete value map at the same time boundary. It checks the model identifier, forecast generators, execution rules, account grid, action schedule, conditional sample count and residual accounting. The saved boundary must also match the actual causal observation loaded for that second. A failed liquidation or invalid terminal close cannot be resumed as if it were a clean pause.

Continuation starts at the next original update. It preserves the original absolute-time seeds, endpoint, counters, pending orders, TD-error statistics and Bellman residuals. The new log contains the prior observations followed by the new segment; cumulative elapsed time includes the earlier segment. The original initial archive is referenced by absolute path rather than duplicating its large value arrays. Reports and the independent ledger audit follow this reference. `--max-updates` limits new updates in the current segment, and `--max-wall-seconds` is a per-segment runtime cap.

Six CPU tests pass, including a real export/load, a carried pending order, changed-policy rejection, wrong timestamp/log/account rejection, and reports using the shared initial archive. The existing three reporting and five execution-ledger tests also pass. A separate GPU comparison is queued to test a two-update run followed by two resumed updates against four uninterrupted updates, including the complete value field, market cloud, front proposals and error statistics. GPU restart equivalence remains pending until that comparison finishes.

The already running long replay retains its startup driver snapshot in its `source` directory. These driver changes do not change its in-memory execution or the five files hashed as its Bellman operator.

```powershell
.venv-ml/Scripts/python.exe ml/backtest_oracle_dense_repair.py --market-source data/backtests/oracle-dense-raw-price-paired --resume PATH_TO_CLEANLY_STOPPED_RUN --output data/backtests/oracle-dense-resumed --max-wall-seconds 3600
.venv-ml/Scripts/python.exe -m unittest discover -s ml -p test_oracle_dense_checkpoint.py -q
.venv-ml/Scripts/python.exe ml/check_oracle_dense_resume.py --market-source data/backtests/oracle-dense-raw-price-paired --initial-map data/backtests/oracle-dense-repair-terminal-90m/initial --output data/backtests/oracle-dense-resume-equivalence-new
```

## Realized baselines

`report_oracle_dense_baselines.py` compares a completed replay with three predefined policies: cash, enter half-long and hold, and enter half-short and hold. The two directional policies keep funded inventory without resizing and execute a valid full close at the same final tick. Entry sizing includes received-amount fees, and every arrival applies the replay's funding, maintenance and liquidation rules. The final close obeys the aggregate quote cap and minimum size; an infeasible baseline does not receive a valid terminal-return number.

Four tests pass for fee-aware exposure, long/short/cash returns, funding on every transition, and rejection of liquidation or an uncloseable final position. The report requires a completed observed interval and passes the independent execution ledger before producing a comparison. No baseline results have been attributed to the currently unfinished long replay.

```powershell
.venv-ml/Scripts/python.exe ml/report_oracle_dense_baselines.py data/backtests/oracle-dense-repair-terminal-90m
```
