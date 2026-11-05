# Incremental continuation updates

The continuation computation can retain its existing market nodes, conditional
laws and account-value functions. Rebuilding a complete market graph at every
forecast origin is not required by the Bellman construction. This experiment
tests exact reuse in the current supporting-plane implementation.

## What can stay cached

With the NSF checkpoint frozen, its conditional kernel at the same two complete
observation rows and absolute clock time stays the same. A new observation
changes the current conditioning state. It does not automatically change the
kernel at every existing hypothetical state. This statement concerns the
implemented two-observation kernel, not sufficiency of two observations for the
real market or a model with an additional evolving posterior.

Cache identity must include the market state and clock, remaining horizon,
forecast-model version and execution parameters. Account values must retain
cash, inventory and pending-order dependence. Reusing storage slots is not
enough: if a slot is assigned a different state, its cached value is invalid.

A rolling one-hour horizon moves the flat terminal boundary. Dropping the old
first layer is cheap, but appending a new second changes the continuation of
existing nodes. Those changes need to be propagated or bounded. The unchanged
market rows can still be reused.

## Delta equation

On states where the relevant expected utilities are finite, write a Bellman
action value as `Q_a = r_a + P_a V`, incorporating the Oracle account transition
into the action-dependent kernel. For updated inputs, the exact identity is

    delta Q_a = delta r_a + (P'_a - P_a) V + P'_a delta V
    delta V   = max_a (Q_a + delta Q_a) - max_a Q_a.

For terminal-wealth CRRA, the intermediate additive utility reward is zero;
wealth changes occur inside the state transition. The changed terminal boundary
initializes the backward delta. The maximum remains outside the conditional
average. Keeping an old maximizing action without checking its competitors is
not an exact update.

This identity supports caching and incremental evaluation. Small numerical
deltas alone do not imply small computational cost. Caching the previous
maximizers or selected support faces may make updates cheaper when their
optimality conditions remain valid. Approximate stopping additionally needs an
error bound at the root; this experiment does not introduce a delta cutoff.

## Implemented experiment

`ml/oracle_incremental_value.py` stores a graph and its account-value planes. It
detects changed conditional rows and prices, invalidates their predecessors,
and reuses unchanged value arrays. Price changes invalidate both the current
node's backup and incoming price transitions. A change is propagated only if
the recomputed integer coefficients differ. Backup-parameter changes invalidate
all rows. Inputs and unchanged arrays are shared without mutation.

The numerical operation is the existing `rational_order_backup`. These are
supporting upper bounds used by the current approximate feedback policy, not
exact log Bellman values. Cache equivalence does not improve or certify their
approximation error.

The benchmark starts with the saved first-test-hour graph: 3,600 seconds,
256 nodes per layer, 64 outcomes per conditional row and 65 support points.
It performs three cases:

1. No change: reuse every value.
2. Change one sample destination in the root's conditional law: update the
   root while retaining all deeper values. This is a localized-law diagnostic,
   not a new observed-history replay.
3. Generate one additional genuine NSF second from all final representative
   states, using the original clustering metric, masses and seed schedule.
   Retain every previous market row, extend the flat terminal by one second,
   and update the values backwards.

For the third case an independent complete backward sweep compares every
integer coefficient at every node against the incremental result. Numerical
kernels are warmed before timing. The full-reference timing also includes
coefficient comparisons.

## Measurements

| Operation | Time | Recomputed value rows |
| --- | ---: | ---: |
| Verify and reuse an unchanged graph | 0.294 s | 0 |
| Change only the root conditional row | 0.295 s | 1 |
| Append one NSF market layer | 0.961 s | Generates 256 new nodes from 16,384 outcomes |
| Incrementally propagate the extended endpoint | 175.626 s | 921,409 / 921,409 |
| Full backward reference, including comparison | 192.215 s | 921,409 / 921,409 |

All compared coefficients are **bit-identical**, with no mismatched depth.
All 51 related Python tests pass, including the six incremental-cache tests.
Unchanged and localized timings include scanning/validating the stored graph;
they are not the intrinsic cost of evaluating one already-cached function.

The market extension reused all 921,153 pre-existing conditional rows. Its
price envelope did not require changing the reachable-leverage parameter.
The value update saved about 8.6% of the full-reference time, whose comparison
overhead also contributes to this difference. Exact invalidation therefore
avoided rebuilding market forecasts but did not eliminate the backward sweep:
every nonterminal value row changed.

The old root cash upper ratio was `1.0034849867224693`; after the extra second
it was `1.0034848665818572`. This small change is in the constructed upper
envelope. It neither measures the maximum change over all action values nor
certifies that a stale action remains near-optimal. A tighter constructed upper
bound can decrease even though the true optimal value for an extended feasible
horizon cannot decrease.

At three minutes of planning per one-second historical decision, a 40-hour
replay would require roughly 300 days. Cheap reuse or bounded approximate delta
updates are therefore essential for the intended runtime. This experiment
establishes equivalence of exact incremental scheduling; it does not establish
that the desired rolling planner is already fast enough.

Results: [machine-readable report](E:/Projects/trading/data/backtests/oracle-incremental-value-test-hour0/report.json).

## Scope

This isolates reuse of an existing market graph and propagation of a changed
endpoint. It does not test mapping a newly observed current state into the
graph, changing representative coverage, or a complete rolling historical
backtest. The extension has 3,601 transitions from the old origin; after
advancing one existing graph step, its retained continuation spans 3,600
transitions. Creating the new observed root remains a separate operation.

The updater still performs the full support search within each affected row.
It does not yet implement cached-maximizer delta arithmetic or a bounded
approximate stopping rule. Consequently this benchmark does not measure the
full speed potential of those further optimizations.

No forecast checkpoint, execution rule, prior backtest or Lean theorem was
changed. The six cache tests cover unchanged inputs, changed prices, changed
probability rows, shorter and longer endpoints, parameter invalidation,
equivalence with full recomputation, and preservation of the prior cache.

## Reproduction

    .venv-ml/Scripts/python.exe ml/benchmark_oracle_incremental_value.py --graph data/backtests/oracle-accuracy-graph-test-hour0 --value data/backtests/oracle-accuracy-value-test-hour0 --output data/backtests/oracle-incremental-value-test-hour0

The output directory is exclusive; use a new directory for another run. It
contains the source snapshots and hashes, the appended forecast layer, per-layer
update counts, and the final comparison report. The benchmark log is
`data/backtests/oracle-incremental-value-test-hour0.log`.
