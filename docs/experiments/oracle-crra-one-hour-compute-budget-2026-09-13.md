# Compute budget for rolling one-hour, one-second planning

Follow-up: [reusable continuation experiment](oracle-crra-reusable-continuation-2026-09-13.md)
implements GPU log-value backups and completes the full archive replay in
238.53 seconds using a fitted continuation function. That replaces explicit
deep search with an additional approximation; its accuracy is not certified.
The measurements below describe the preceding explicit scenario-graph method.

The current implementation is not practical for a complete one-hour-deep
policy search at every backtest second. Forecast construction alone is
estimated at **9.9 minutes per origin on the RTX 3070**, using 256 retained
market nodes per layer and 64 conditional samples per parent. Account-state
search is a separate, exponential bottleneck and is not included in that number.

## Measurement and extrapolation

Using the same frozen sampled-skip NSF checkpoint, current source, independent
256-node compression and consecutive one-second transitions:

| Future layers | Graph construction | Peak PyTorch GPU allocation |
| --- | ---: | ---: |
| 64 (prior measurement) | 10.5057 s | 272.47 MiB |
| 256 (new measurement) | 42.1491 s | 345.09 MiB |
| 3,600 (linear estimate) | 593.27 s / 9.89 min | 1,609.98 MiB |

Timing covers the construction call, including NSF sampling, compression and
diagnostics; it excludes checkpoint loading, initial dataset loading and policy
optimization. The new probe stores 65,345 market nodes and 4,165,696 sampled
edges. Forecasting is independent of the utility's gamma value.

At 3,600 layers, the same shape would contain approximately 921,409 retained
market nodes and 58,953,792 sampled edges. Retained CPU feature/price/time
arrays would use about 443 MiB. The extrapolated GPU number is allocated
tensor memory, not total process/driver memory; allocator caches and other
applications also use the 8 GiB GPU. No 3,600-layer run was performed, so this
does not verify long-rollout numerical stability or actual peak memory.

The depth slope is measured from the 64- and 256-layer runs. Extrapolation
assumes comparable utilization and a full retained width, with no optimization
to batch or reuse work across forecast origins. The older nested splitter
took about 18% longer at 64 layers; the table uses the faster independent fitter.

If a fresh full-hour graph is built for every realized second:

| Backtest duration | Forecast origins | Forecast-only RTX 3070 time estimate |
| --- | ---: | ---: |
| 1 hour | 3,600 | 24.7 GPU-days |
| 24 hours | 86,400 | 593 GPU-days |
| Existing 40h08m53s archive | 144,533 | 992 GPU-days |

These are naive sequential cost projections, not irreducible computational
lower bounds. They omit shortening the planning horizon near the archive's
final flat-close deadline. They also exclude complete backtest execution,
account optimization, and verification. Batching forecast origins or reusing
an appropriate conditional kernel may improve throughput, but neither saving
was measured here. Reuse must retain dependence on observable state and clock;
later forecasts based on realized future observations cannot be read early.

## Why this is not the complete planner budget

The existing 256-node cap compresses market histories only. At the same market
node, different action sequences produce different quote balances, inventory,
pending limit prices/sizes and reservations. `Bellman._value` recursively
evaluates those account continuations instead of merging them into a bounded
state/value representation. Even one sampled market path does not remove the
branching over candidate actions.

The 0.01 exposure grid has 401 target exposures over [-2,2], plus zero trade,
cancellations and the declared limit placements. Two-second GPU benchmarks
already evaluated 171.7 million terminal accounts with 64 samples per parent
and this fine action grid. That benchmark took 2.24 seconds; it cannot be
extrapolated linearly to 3,600 decisions. The exact recursive search is
exponential in decision depth. There is no credible practical end-to-end GPU
budget for running that algorithm unchanged to one hour.

The GPU evaluator currently implements gamma zero; the gamma-one interval
solver added in the previous experiment is a CPU finite-tree evaluator. A GPU
log-utility backup would be straightforward relative to the state-search
problem, but would not resolve its scaling.

A practical every-second construction needs a reusable continuation-value
representation or account-state aggregation in addition to the market graph,
with measured approximation error. Absolute order caps/minima and pending
orders mean that exposure alone does not determine the full account state.
All modeled seconds can then include trading, fills, funding and liquidation,
but finite sampling, account interpolation and action/order grids still need
their own accuracy assessment. Calling that merely a full-fidelity version of
the existing 256-node forecast would conceal additional approximations.

## Scope of the earlier 95% interpretation

For the gamma-one five-stage experiment, the 4.653991% upper contribution means
the one-hour certainty-equivalent wealth is at least

    1 / 1.04653991264750871 = 95.552973%

of the 48-hour certainty-equivalent wealth for the same fixed current action.
That is a model-value statement, not a realized-wealth guarantee. Its forecast
law has only five decision checkpoints; it has not been established for the
larger model with a decision, observation, fill and liquidation check every
second. This compute estimate does not transfer that certificate between models.

## Artifacts

New measured construction and source snapshot:
`data/backtests/oracle-1h-compute-probe-20260913/`.
Derived counts and timing estimates:
`data/backtests/oracle-1h-compute-probe-20260913/compute-estimate.json`.

```powershell
.venv-ml/Scripts/python.exe ml/benchmark_nsf_scenario_graph.py --depth 256 --width 256 --samples 64 --construction-only --refinement independent --output data/backtests/new-one-hour-compute-probe
```

Only a forecast-construction benchmark was run. No new trading backtest or
full-depth policy solve is claimed.
