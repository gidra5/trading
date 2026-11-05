# GPU Bellman scaling with the latent NSF

Implemented a batched PyTorch float64 counterpart of the finite-tree Oracle
Bellman solver. The NSF already used CUDA; the new evaluator moves account
transitions, candidate actions, conditional expectations, and maximization onto
the GPU as well. The exact rational TypeScript solver remains the reference.

The experiment uses the existing frozen sampled-skip latent NSF checkpoint,
SHA256 `80fdc3df78a5f60d9980b236f9bbaa296904dc562bd64195394831120dde01db`.
There is no retraining or model promotion. Hardware is an NVIDIA RTX 3070.

## What is searched

Exposure targets cover [-2, 2] in increments of 0.01: 401 targets. The menu also
includes zero trade, which preserves inventory and differs from target exposure
zero. Fee-aware target requests obey the aggregate quote cap; targets that
execute the same capped trade are deduplicated. Chunking and deduplication do
not deliberately remove distinct feasible actions from this declared menu.

The one-slot limit-order menu retains a buy or sell one price tick away, with
notional 25% of current equity, and cancellation choices. Limit prices and sizes
are still coarse. Simultaneous nonzero market adjustment and new placement are
excluded in these runs. The tensor evaluator can accept broader limit grids
and joint actions through `Menu`, with additional computational cost.

Execution retains the reference experiment's received-amount fees, signed asset
funding, reservation, maintenance liquidation, activation and crossing rules,
aggregate order limits, and valid capped terminal close with residual inventory.
Parameters: $10,000 initial equity; maker/taker fees 0.02%/0.05%; entry exposure
[-2,2]; maintenance exposure [-3,3]; reservation leverage 2; quote bounds
$5–$10,000; price tick $0.01; one pending order. Gamma is zero. The replay uses
funding factor one; it does not incorporate historical funding payments.

Actions are shared by all branches before the next observation arrives. At
each child, continuation actions can depend on that child's observable market
history and account state. Expected continuation values are reduced across
children before maximizing over the current action. Hidden NSF latent samples
are never used as observable branch labels. The tree importer rejects duplicate
observable siblings; callers must merge those with conditional continuation
weights before solving.

## Measured scaling

These are individual planning benchmarks at the same forecast origin. Times
below are Bellman solve times, excluding NSF inference and model loading. They
are representative measurements, not sustained service latency guarantees.

| Sampled branches at successive transitions | Future transitions | Exposure spacing at successive decisions | Terminal account evaluations | GPU solve | Peak allocated GPU memory |
|---|---|---|---:|---:|---:|
| 64 | +1 second | 0.01 | 13,056 | 0.084 s | 18 MiB |
| 8, 8 | +1, +2 seconds | 0.01, 0.01 | 2,682,632 | 0.161 s | 103 MiB |
| 64, 2 | +1, +2 seconds | 0.01, 0.01 | 5,365,262 | 0.894 s | 59 MiB |
| 64, 64 | +1, +2 seconds | 0.01, 0.01 | 171,688,384 | 2.243 s | 633 MiB |
| 8, 2, 2 | +1 second, midpoint, archive terminal | 0.01, 1, 1 | 329,870 | 0.054 s | 72 MiB |

The 64-by-64 tree contains 4,160 nonroot market nodes and 4,096 terminal market
paths. Account evaluations are much more numerous because candidate trades
produce different account states on those same market paths. Invalid actions
and duplicate capped requests explain why counts differ from the naive full
Cartesian product. GPU memory figures are PyTorch allocations, not total device
memory including the driver and allocator cache.

NSF inference took about 0.02–0.17 seconds for the warmed-up trees, with about
0.29 seconds for the first one-step query. The fixed-terminal benchmark's exact
CPU solve took 16.890 seconds versus 0.054 seconds on GPU, approximately 311x
faster. The smaller one-step comparison was about 3x faster. Both comparison
values matched the rational reference at displayed float64 precision.

The one- and two-second terminal benchmarks are computational tests. They do
not replace the user's full-interval wealth objective in the replay below.

## One-second replay with the original terminal

Ran the first 120 seconds of the archived test interval, making a decision every
second. Each plan retains the original terminal timestamp at the end of the
144,533-second archive. Hypothetical decision offsets are +1 second, +2 seconds,
approximately halfway through the remaining interval, and the fixed terminal.
Branching is 8,2,2,2: 120 nonroot market nodes and 64 terminal market paths.
The current action grid is 0.01; future action grids are 1.0.

The run completed in 30.333 seconds including forecasts and planning: mean
decision time 0.252 seconds, 95th percentile 0.505 seconds. It evaluated about
200 million terminal accounts in total. This configuration was faster than the
one-second decision interval over this short test; latency and state dependence
over a whole archive remain unmeasured.

An independent exact rational execution replay accepted all 120 actions. There
were 2 market decisions, 10 limit placements, 2 cancellations, 8 fills, and no
liquidation. Fees paid were $4.099. Marked equity at the end of this two-minute
slice, before the next arrival phase, was $9,986.147 (-0.13853%). Inventory
remained open. This is **not a completed backtest or a terminal return**, and
does not establish that the refined policy trades profitably.

## Validation and limitations

Eight unit tests cover action timing, observable feedback, chunk invariance,
liquidation ordering, residual terminal holdings, activation, tick boundaries,
leverage boundaries, and rejection when the work budget is exhausted. Across
36 differential cases, maximum expected-wealth error was $1.82e-12. Exact
re-evaluation of each chosen action gave zero selection regret at reported
precision. Independent rational replay of the 120 actual decisions found
maximum numerical account/trade drift of $1.56e-12.

The larger comparison caught two bugs during development: binary price
subtraction could miss exact tick crossings, and a float32 scalar conversion
could erase the roundoff tolerance at an exact leverage boundary. Crossings now
compare integer tick coordinates, and comparison tolerances remain float64.
Regression tests cover both. The interrupted original `second-replay/` and
`benchmark-before-tick-fix.json` are invalid/superseded artifacts; use
`second-replay-refined/` and the current validated benchmark files.

These checks do not constitute a Lean proof of floating-point refinement or a
uniform error certificate. The kernel uses a small numerical tolerance at
constraint boundaries; sampled exact feasibility checks cannot guarantee all
future numerical decisions. The formal finite Bellman theorem continues to
apply to exact arithmetic and the stated finite search problem. No changes
were made to the Lean theorems.

GPU batching does not remove exponential depth. With 64 branches per second,
ten future seconds already produce 64^10 ≈ 1.15e18 terminal market paths,
before counting alternative account actions. A full every-second tree to the
40-hour terminal remains infeasible. A fixed pool of 64 nodes at each time
would require a different approximation, such as state aggregation and a
continuation-value approximation; it is not equivalent to 64 branches per node.

The replay still omits intermediate hypothetical prices, fills, liquidation
checks, observations, and actions across its two long jumps. Refining the root
action grid and actual decision frequency does not resolve those omissions.
Learned NSF model error, finite sampling error, coarse limit orders, and
continuous-action coverage also remain. Neither these measurements nor the
existing conditional Lean bounds provide a global epsilon-optimality claim.

## Reproduction

Implementation: `ml/gpu_oracle_bellman.py`. Benchmark/forecast generation:
`ml/benchmark_gpu_oracle.py`. Replay: `ml/backtest_gpu_nsf.py`.
Exact audits: `scripts/audit-gpu-oracle.ts`, `scripts/audit-gpu-benchmarks.ts`,
and `scripts/audit-gpu-replay.ts`.

```powershell
node --conditions=development --import tsx scripts/audit-gpu-oracle.ts
.venv-ml/Scripts/python.exe ml/benchmark_gpu_oracle.py
node --conditions=development --import tsx scripts/audit-gpu-oracle.ts --check
node --conditions=development --import tsx scripts/audit-gpu-benchmarks.ts
.venv-ml/Scripts/python.exe ml/backtest_gpu_nsf.py --decisions 120 --branches 8 --output data/backtests/oracle-gpu-scaling/new-replay
node --conditions=development --import tsx scripts/audit-gpu-replay.ts data/backtests/oracle-gpu-scaling/new-replay
.venv-ml/Scripts/python.exe -m unittest discover -s ml -p test_gpu_oracle_bellman.py
```

`--future-step` controls future exposure grid spacing, `--near-seconds` adds
consecutive future one-second transitions, and `--chunk` controls account batch
size. The replay rejects existing output plans. Forecast construction rejects
trees above 100,000 nonroot nodes; Bellman search rejects work-budget overruns
without returning a partial decision. Increasing a budget permits more work,
not a better complexity class.

Results and source snapshots are under `data/backtests/oracle-gpu-scaling/`.
`validation.json` identifies the validated artifacts, superseded trials, source
hashes, and absence of a global optimality certificate.
