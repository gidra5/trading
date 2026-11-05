# Fixed-terminal stochastic Oracle backtest: construction audit

The requested backtest ends at **2026-08-11 16:08:54 UTC**. Its first decision
is **2026-08-10 00:00:01 UTC**, giving 144,533 one-second transitions. The
terminal objective is gamma-zero CRRA utility of Oracle terminal wealth,
including the valid capped closing procedure and marked residual. Intermediate
decisions may retain inventory and pending orders. There are no intermediate
forced closes, rolling terminal times, or separate roundtrip episodes.

**This full-interval backtest has not been run.** The previous H1 replay is
not its implementation, and no new return figure is available. The audit
found missing inputs and an unimplemented optimization domain, rather than a
parameter that can simply be changed in the previous runner.

**Subsequent formalization:** [Continuous stochastic policy](../theory/Continuous%20stochastic%20policy.md)
now constructs policies over real exposures with time/history-dependent,
countably supported price laws. Exact optimality follows from closed bounded
viable action sets and upper semicontinuous Bellman scores; a near-optimal
construction only needs finite Bellman values. An Oracle dust/mandatory-close
counterexample proves unconditional exact attainment false. These are
noncomputable Lean constructions, so the numerical solver, future conditional
inputs and full-interval backtest obligations below remain open.

## What the existing Lean construction establishes

`CRRAPolicy.constructed_crra_policy_optimal` optimizes expected utility over
nonanticipative complete scenario plans. Its global conclusion requires
`cover`: every feasible plan belongs to the supplied finite list. The theorem
`crra_search_optimal_on_menu` establishes optimality within a supplied menu
without that assumption. A finite list cannot literally enumerate an interval
of distinct feasible continuous trade quantities. No theorem reducing this
full account problem to the implemented finite menu has been established.

`ExactCRRAKernel.bellman_step` establishes the Bellman composition step given
optimal feasible child continuations and an optimal root comparison.
`ExactRiskNeutral` supplies the gamma-zero utility correspondence. Neither
theorem computes the conditional market law, eliminates the need for future
continuation values, nor makes exhaustive search tractable over this interval.

The exact TypeScript `solveCrra` implements recursive search over a supplied
finite observable price tree and finite action menus, including market trades,
limit placements and cancellations. It carries holdings forward and closes
only at the supplied tree's leaves. Repeatedly supplying short trees would
still change the requested fixed-terminal objective.

## Missing construction inputs and implementation

1. **Future conditional information.** The fresh direct59/base17 law adapter
   accepts two observed production59 states and returns a next-second law.
   The native network predicts feature states as point estimates, not a joint
   stochastic transition law for future observed features and calibration
   state. The saved mixture streams contain forecasts at actual observed
   histories, every four seconds. Those later rows were not available to an
   earlier decision and cannot be inserted as its hypothetical continuations.
   Freezing features or propagating point estimates would introduce additional
   modeling assumptions, not recover an already supplied joint law.
2. **Continuous-action optimization.** The existing exact finite-menu solver
   does not cover all feasible continuous quote flows or order notionals.
   The continuous endpoint branch-and-bound implementation covers one terminal
   step, not arbitrary child value functions with carried positions/orders.
3. **Full remaining horizon.** Even after specifying a continuation model,
   direct tree expansion does not scale to 144,533 steps. A binary fully
   branching tree already has 2^144533 terminal paths. This is a limitation of
   the current exhaustive algorithm, not a proof that every possible exact
   algorithm must enumerate those paths. A valid state reduction, exploitable
   Bellman structure, or explicitly bounded approximation is still needed.

Thus the existing files do not constitute an executable, proved globally
optimal continuous-action policy for this full interval. The earlier language
connecting the endpoint backtest to that guarantee was incorrect. Better
one-second prediction MSE does not discharge these missing obligations.

## Completed checks

- All **12 Oracle CRRA policy tests** pass.
- A new regression shows the concrete horizon error: at prices 100 then
  100.06, followed by 100.18 with probability 3/4 or 99.99 with probability
  1/4, both one-tick cash-start roundtrip optimizers choose cash at a 0.05%
  received-amount fee per side. The two-step Bellman solver enters at 100,
  chooses zero trade at 100.06, retains its exact inventory, and pays its exit
  fee only at the final tick. Its expected terminal wealth exceeds initial
  wealth. This is a synthetic regression, not a historical trading result.
- The **six Lean risk-certificate regressions** pass, including four intended
  rejections of invalid/nonoptimal certificates. They validate the checker;
  they are not certificates for uncomputed full-interval decisions.

Sources:

- `docs/theory/lean/TradingTheory/CRRAPolicy.lean`
- `docs/theory/lean/TradingTheory/ExactCRRAKernel.lean`
- `docs/theory/lean/TradingTheory/ExactRiskNeutral.lean`
- `packages/bot-algo/src/oracle-crra-policy.ts`
- `packages/bot-algo/test/oracle-crra-policy.test.ts`
- `ml/refitted_price_forecast.py`
- `ml/backtest_refitted_ensemble.py`

The next implementation must supply the conditional continuation model and
an optimizer for the fixed terminal objective, then establish the applicable
execution, action-coverage and value-bound obligations. Any finite scenario
or action approximation must be reported with its own scope and error bounds;
it cannot inherit unrestricted optimality by citing the Bellman theorem.
