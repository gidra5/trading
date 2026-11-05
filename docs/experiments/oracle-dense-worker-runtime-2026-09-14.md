# Stateful Oracle paper execution

The server paper API and replay driver now call the same `DenseController`. It advances the account, funds inventory, fills eligible carried orders, predicts a fresh next-second law, updates the value map and selects a post-decision account. It automatically requires a valid full close on the specified terminal tick. This is the finite-grid **log-utility (gamma=1)** planner, not a certificate for the continuous optimum or the true price law.

## Server setup and controls

`data/backtests/oracle-dense-server-worker-config.json` contains the model hash and the coarse one-hour replay's execution/grid settings. It fixes the model, rules and numerical configuration for a worker's lifetime. Paths in the server environment resolve relative to the repository, including when npm launches the server from its workspace directory.

```powershell
$env:ORACLE_DENSE_CONFIG = 'data/backtests/oracle-dense-server-worker-config.json'
$env:ORACLE_DENSE_STATE = 'data/paper/oracle-dense'
$env:ORACLE_DENSE_SYMBOL = 'BTCUSDT'
npm run dev -w @trading/server
```

`ORACLE_DENSE_PYTHON` optionally overrides the repository's Python runtime. The configured account is explicitly paper-only. The existing exchange bot endpoints and credentials retain their separate meanings; this worker never submits exchange orders.

| Method and path | Operation |
| --- | --- |
| `GET /bot/oracle-paper` | Current status and last completed decision |
| `POST /bot/oracle-paper/start` | Initialize a new episode, decide and checkpoint |
| `POST /bot/oracle-paper/tick` | Process the next observed second |
| `POST /bot/oracle-paper/checkpoint` | Save the current completed decision boundary |
| `POST /bot/oracle-paper/stop` | Checkpoint and stop the worker, preserving exposure |
| `POST /bot/oracle-paper/resume` | Restore the saved account and continuation map |
| `POST /bot/oracle-paper/reset` | Explicitly clear the stopped account's pointers for a new episode |

Start takes `observation`, `account`, `terminalSeconds`, `consumedQuote` and `filled`. An observation contains two 59-feature rows, two prices and their consecutive Unix-second timestamps. The account contains quote, asset, previousPrice and pending orders with side, price and notional. It is already at the arrival phase: funding and committed fills for that initial tick have been applied. `consumedQuote` and `filled` prevent spending that tick's order capacity twice. Subsequent tick requests contain only the next observation pair.

Returned accounts have already executed the selected paper action. Do not apply that action again. The caller supplies the causal 59-feature feed; the API does not invent missing features or skip observations. Resume reports the checkpoint's time boundary. Intervening observations must be replayed in order before later ticks are accepted. This is simulated recovery, not evidence that exchange trades occurred during a server outage.

The service checkpoints every 60 processed seconds by default and on stop. Completion writes a terminal marker and closes the worker. Resume refuses a completed episode, so it cannot repeat its final trades. Reset removes only the small active/completed pointers; it preserves historical snapshots. Concurrent mutations receive a conflict response rather than executing twice.

Status now includes a bounded `failure` message and reports an exited worker as not running, even if it failed between requests. Resume disposes that failed process and restores the last committed snapshot. A tick whose required checkpoint fails is not published as an accepted decision. Local history validation errors leave a healthy worker available. Recovery may return an earlier timestamp than the last response when the checkpoint interval exceeds one; the caller must replay from the returned boundary. Five service tests and seven controller tests pass, and the real API/server-restart smoke check passes in 7.82 seconds (`api-recovery-results.json` under the smoke-test directory).

## Checkpoint integrity

Snapshots contain the account/front, complete value field, market cloud, conditional sample counts, TD-error statistics, residual accounting, episode deadline, original update count and random schedule. Every file is checksummed. The model, execution rules, grid and implementation hashes must match before restoration. Saving also refuses source files changed since the worker started.

The snapshot files are flushed before an atomic pointer replacement. An interrupted save leaves the previous pointer usable. Rotation retains the current and previous snapshots, plus a restored snapshot while its arrays remain memory-mapped. Cleanup stays within the managed state directory and defers files that cannot be removed. Active, failed and terminal boundaries are distinguished; a failed or incomplete decision cannot be saved as a successful resumable state.

Six checkpoint tests pass, including exact continuation after restoration, interrupted pointer replacement, model/configuration mismatch, corrupted arrays, invalid paths and rotation with open Windows memory maps. Six worker protocol tests and seven server controller tests also pass. The paper service and route tests cover start, stop, resume, reset and terminal completion.

The real Fastify API/CPU NSF smoke test stopped after one tick, recreated the server, restored its account and completed the second tick with a valid flat close in **7.88 seconds**. Artifact: `data/backtests/oracle-dense-server-worker-smoke/api-results.json`. This is a two-tick synthetic continuation for integration checking, not a profitability or one-hour throughput result.

`scripts/smoke-oracle-paper.ts` now provides a repeatable real localhost HTTP version of this check. It accepts explicit `--config`, `--input` and a new `--output` directory, starts the actual persistent Python worker, recreates the Fastify server after the first tick, checks exact restoration of the account boundary, and requires flat terminal completion. It preserves responses and operation timings. The CPU NSF run completed in 7.45 seconds with terminal cash 10,000 (`data/backtests/oracle-paper-http-cpu-smoke/results.json`). This checks the transport and recovery path; the incremental CUDA result follows below.

The incremental fixture uses fourteen actual consecutive test observations, an eight-second rolling horizon, four conditional outcomes and two-draw interior refreshes. Its two market nodes and small account grid make it an integration diagnostic, not a throughput benchmark. The real HTTP/CUDA NSF run passed in **15.73 seconds**, including 8.32 seconds for initial worker startup/solve and 3.12 seconds for restart/restore. The restored account matched exactly at update one, all fourteen ticks completed, and terminal cash was 10,000 with no holdings or orders. This validates the incremental worker path, including rolling and shrinking horizons, but does not establish long-horizon forecasting accuracy or profitability. Artifact: `data/backtests/oracle-incremental-http-smoke/results.json`.

Reproduce with a new output directory:

```powershell
node --conditions=development --import tsx scripts/smoke-oracle-paper.ts --config data/backtests/oracle-incremental-http-fixture/config.json --input data/backtests/oracle-incremental-http-fixture/input.json --output data/backtests/oracle-incremental-http-smoke
```

The broader server suite currently has **93 passing tests and one unrelated failure**: WebSocket status tests expect payload forwarding and boolean results, while the current `WebClient` status methods return neither. That interface has not been changed by this work. Server TypeScript checks pass.

## Removing repeated metadata construction

Profiling found that the 41-exposure root search rebuilt all grid coordinates 1,845 times per decision. `oracle_dense_metadata.py` supplies a view of the map's existing metadata to the original sampling and cell-indexing methods. It shares immutable arrays, preserves random draws and arithmetic, and leaves the Bellman operator and archived grid identity unchanged.

At the saved fine initial state, root search fell from **5.2943 to 0.0644 seconds**, and counterfactual error recording from **1.5004 to 0.4004 seconds**. All front proposals, scores, chosen states and error statistics matched exactly. The first **79 actual replay seconds** also matched the previous implementation exactly. Artifacts: `oracle-dense-prepared-metadata.json` and `oracle-dense-exposure41-prepared/prefix-equivalence.json` under `data/backtests`.

The slow fine replay was interrupted after 79 observations; it has no terminal return. Its complete initial map was reused for `oracle-dense-exposure41-prepared`, avoiding another 13.7-minute construction. That replay completed 3,600 observed seconds in **4,288.97 seconds (71.48 minutes)**. It ended flat at **9,976.4756**, a **-0.23524%** net return, with 23 market trades and five individual limit fills over three fill ticks. Maximum observed drawdown was 0.73255%. The independent ledger verified all 3,600 transitions, the actual terminal close, and all 3,601 prices against immutable candle closes. This run uses a fixed terminal date and a shrinking planning horizon; it does not measure a perpetually rolling one-hour planner. Artifacts: `results.json`, `summary.json` and `ledger-audit.json` in that directory.

The fine rolling-horizon profile keeps 3,600 future decisions while appending a new tail. Its first two updates took **11.3763 and 26.7206 seconds** (mean 19.0484). They recomputed 41 and 121 layers respectively, reusing the rest. Their finite-field repair allowances were 5.37e-6 and 9.34e-6. This two-update diagnostic is too small to estimate sustained runtime reliably, but shows that the shrinking-horizon timing cannot justify a long fine rolling replay within a few hours. Artifact: `data/backtests/oracle-dense-exposure41-rolling-profile/updates.jsonl`.

## Parallel counterfactual error updates

`oracle_dense_observe_batch.py` now evaluates the independent front proposals concurrently using the existing Numba arrival and decision kernels. Per-proposal arithmetic and candidate ordering stay unchanged. Cell grouping, averaging and running-error updates still occur in the original order, so parallel reductions cannot change error statistics. The prepared map view uses this implementation; the original map remains available for conformance checks. The server worker and backtest driver use at most four Numba threads, respecting a lower environment limit.

On the fine archived field, 405 finite proposals took a median **0.3902 seconds** through the original prepared loop, **0.3530 seconds** with one batched thread, **0.1997 seconds** with two, and **0.1169 seconds** with four. This integrated measurement is a **3.34x** improvement for the error-update stage; the preliminary five-repeat measurement was 3.60x. Both were measured while the existing GPU replay was active. This is not an end-to-end planner speedup. Artifact: `data/backtests/oracle-dense-observe-batch-integrated.json`.

Every observation summary and all cell counts, means and second moments matched exactly. Tests cover final/nonfinal decisions, funding, upward/downward fills, invalid arrivals, empty proposals and one/two-thread execution. The combined controller, metadata, batching, worker and checkpoint suite passes **23 tests**. A reference cycle found during integration was removed; the existing Windows checkpoint-rotation tests verify that mapped files are released.

The real Fastify/NSF restart smoke test also passes with the batched worker in **7.83 seconds**, ending at a valid flat cash balance of 10,000. It is the same two-tick synthetic integration check, not a profitability result. Artifact: `data/backtests/oracle-dense-server-worker-smoke/api-batched-results.json`.

The completed fine replay retained its startup implementation and source snapshot. Subsequent workers and the completed rolling-horizon timing profile use batching. The new helper is included in backtest source snapshots and worker checkpoint identity checks.
