# Flat terminal policies

Completed policies must end with exactly zero holdings in every asset and no
pending orders. This is a feasibility constraint on every supported terminal
branch, and is propagated backward through the Bellman construction. A zero
target or a positive marked residual does not satisfy it.

The terminal arrival executes first. Its fills count against the aggregate
quote cap; cancellation cannot undo them. The final market close pays the
original received-amount fee, obeys the minimum and remaining maximum, and
cannot overshoot. If that decision cannot close everything, earlier decisions
must have reduced the position. No additional tick, free close, dust write-off,
or order splitting within a decision is introduced. Liquidated zero accounts
remain permissible under gamma=0.

## Formal construction

`ContinuousStochasticPolicy.Model.terminalConstraint` defaults to asset = 0.
It is part of `terminalFeasible`, hence of `viable`, `allowed`, and policy
feasibility. The original continuous-action construction and conservative
Bellman evaluator use that restricted feasible set. The exposure-control
parameterization preserves the terminal predicate.

`FlatTerminalPolicy.constructed_optimal_and_flat` proves that the constructed
policy is feasible, ends flat on every supported branch, and dominates every
other feasible policy. Its hypotheses still require viable initial state and
attained Bellman optima. Separate results prove residual infeasibility, zero
terminal exposure, terminal value equal to cash, and cancellation of the book.
The portfolio model also requires all holdings and books to be empty;
`feasible_vector_ends_flat` connects its valid batch sequence to that ending.

`ContinuousEndpointBounds` additionally proves the full long close's actual
Oracle cash/asset formulas, its finite-distribution expectation, and the affine
interval endpoint bound used by the restricted numerical endpoint controller.
Floating-point arithmetic and distribution integration are not Lean-certified.

The four historical convergence-counterexample models explicitly keep their
previous marked-residual predicate. Their witnesses cannot be passed off as
feasible flat-ending policies. The bounded-price maintenance witness is now
also tested for rejection by the strict finalizer. This removes that witness;
it does not supply a general finite convergence rate or a one-hour error bound.

## Executable policy and price-law feasibility

The exact rational finalizer rejects any nonzero residual, so `solveCrra`
rejects a candidate if even one supported branch cannot finish flat. This also
applies to the bot and rational action-menu refinements. The GPU tree/graph
solver assigns an infeasible leaf negative infinity and excludes it at every
ancestor. Numeric underflow cannot turn a nonzero holding into a valid zero
closing request. A completely infeasible root produces an explicit error.

The physical `closeWithinCapacityChecked` helper still returns the real partial
execution and residual, for account operations and the historical comparisons.
It is not the completed-policy terminal evaluator.

The earlier continuous endpoint controller has a narrower domain: nonnegative
cash and long inventory, gamma=0, unit funding, no orders, no intermediate
trades, and maintenance containing [0,1]. Its positive post-entry inventory a
must lie in

    minimum / ((1-fee) * lowest supported price)
      <= a <=
    maximum / ((1-fee) * highest supported price).

Zero inventory is a separate feasible point. The optimizer intersects these
constraints with the continuous buy/sell intervals and the current order-size
limits. Within each feasible interval, expected complete-close cash is affine;
its endpoints suffice. Spread still affects feasibility through the support.

An unbounded upper tail and finite terminal cap allow no positive inventory in
this endpoint domain. Those gamma/knot forecasts consequently select cash, or
reject an initial state that cannot reach cash. Structural support is retained
even when a tiny tail probability underflows or CDF subtraction rounds to zero.
Imposing a numerical price ceiling requires an explicitly changed forecast law;
this update does not silently truncate it. Multi-step/leveraged planning retains
its full transition, including liquidation, and must be analyzed separately.

Backtests now require a valid complete close before reporting terminal wealth
and terminal return. Unfinished replays report marked wealth separately. The
NSF replay records an invalid terminal result with `complete: false`, null
terminal wealth/return, and its failure reason. Buy-and-hold benchmarks that
cannot close under the same cap are explicitly infeasible, with null returns.
Old result files were not relabeled as compliant with this new contract.

## Checks and replay

- 50 TypeScript tests passed, including the server bot, exact execution,
  Bellman, refinement, conservative comparisons and backtest integration.
- 34 Python tests passed, covering continuous actions, tail support, GPU
  Bellman, scenario graphs and nested compression.
- The Oracle TypeScript project typecheck passed.
- Full Lean build and public-result axiom audit: `node scripts/theory-lean.mjs check`.
- GPU versus exact rational Bellman: 30 cases, maximum value difference
  1.8189894035458565e-12 dollars. Every GPU-selected root action was exactly
  feasible, with zero regret after exact re-evaluation of its continuation.
  These are conformance checks, not a universal numerical error certificate.

The existing sampled-skip latent NSF archive was replayed through the fixed end
of its complete 144,533-second interval. It used 161 decisions, original fees,
unit funding, one limit-order slot, and the archived finite scenario trees.
Action level 2 uses targets [-2,-1,0,1,2] plus zero trade and the configured limit
menu; actual funding/price/liquidation execution is checked at every second.
Hypothetical decision times remain coarse. This run verifies the new contract,
not convergence to the unrestricted continuous learned-law optimum.

| Metric | Flat-ending replay |
| --- | ---: |
| Initial cash | $10,000 |
| Final cash | $9,570.379045448408 |
| Return | -4.2962095% |
| Maximum drawdown | 5.0064269% |
| Fees at execution prices | $26.4220993 |
| Market orders | 14 |
| Limit placements / fills | 11 / 11 |
| Liquidations | 0 |
| Terminal inventory / pending orders | 0 / 0 |
| Runtime | 240.86 seconds |

The old replay with the same archive and action level reported -4.0167041%,
but retained 0.1489560706 asset units and only $125.955361 in cash. That old
marked terminal result does not satisfy the current constraint; imposing the
constraint changes decisions throughout the plan, not only its final valuation.

Artifacts are in `data/backtests/oracle-flat-terminal-20260912/`: `nsf-replay/`
contains the plan, exact decisions and report; `gpu/` contains the reference,
GPU values and exact selection audit. Verification logs are
`data/backtests/oracle-flat-terminal-{tests,python-tests,lean}.log`.

Reproduction commands from the repository root:

```powershell
node --import tsx scripts/backtest-nsf-oracle.ts --source data/backtests/oracle-nsf-planning --action-level 2 --output <new-replay-directory>
node --import tsx scripts/audit-gpu-oracle.ts --output <new-audit-directory>
.venv-ml/Scripts/python.exe ml/benchmark_gpu_oracle.py --audit-only --output <new-audit-directory>
node --import tsx scripts/audit-gpu-oracle.ts --check --output <new-audit-directory>
node scripts/theory-lean.mjs check
```
