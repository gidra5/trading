# Error bounds with a flat terminal account

The flat-terminal requirement yields a finite upper bound for the original
Oracle gamma=0 Bellman problem and a valid stopping certificate. It does not
yield a universal rate at which an arbitrary stochastic continuation loses
influence on the first action.

This follows the [flat-terminal implementation](oracle-crra-flat-terminal-2026-09-12.md).
All completed policies must close their holdings through valid orders and
finish with no pending orders. The original fees, funding, liquidation,
per-decision aggregate order cap and price constraints remain in force.

## An upper bound from actual execution

Let q be the current quote balance. Suppose the total absolute quote movement
at every decision, including arrival fills and market trades together, is at
most M >= 0. Assume the same bound at every supported hypothetical history.

A rebalance can increase quote by at most M. Signed-asset funding and price
marking leave quote unchanged. Liquidation replaces the account by zero, so
the positive part of quote satisfies

    max(q_next, 0) <= max(q, 0) + M.

At a valid flat terminal account, wealth equals quote. With r transitions
remaining, followed by the scalar model's existing terminal closing decision,

    V_r(s) <= max(q_s, 0) + (r + 1) M.

The final +1 counts the specified terminal decision; it does not introduce an
additional tick or an exemption from the close's size limit. Terminal arrival
fills share that decision's M with the final market close.

The proof is pathwise before taking expectations. It therefore allows arbitrary
normalized, history-dependent next-price PMFs and continuous feasible controls.
It requires neither a price ceiling nor a funding ceiling. Restrictions on
orders, leverage, price grids and terminal feasibility only restrict the
policies to which the inequality is applied.

`OracleFlatTailBound.bellman_cash_upper` proves this for the actual scalar
continuous Bellman construction. `finite_values_from_cap` consequently proves
its `FiniteValues` condition; finiteness is no longer a separate assumption.
`action_cash_upper` also bounds a first action by the expected cash bound at
its successor states.

For the portfolio engine, use the shared aggregate quote cap M, or a bound on
the sum of individual caps. The same inequality applies to shared quote,
including all assets' simultaneous fills and trades. The portfolio plan counts
its closing trades among its n batches, so its bound is max(q,0) + n M.
`vector_run_positive_quote_upper` and `vector_flat_wealth_upper` prove this for
every feasible plan in the existing concrete vector model. This is a pathwise
bound, and taking expectations preserves it. It is not a new formalization of
a stochastic vector Bellman selector.

The conservative assessment ledger subtracts nonnegative debits from actual
terminal wealth, clipping at zero. The same nonnegative cash upper bound
therefore also bounds that assessed payoff. It is not necessary to replace
the physical funding map with a different execution rule.

## A certified stopping rule

Fix the actual terminal date T. Write Q_T(a) for the best full-horizon expected
terminal wealth after first action a. At search depth K, construct intervals

    L_K(a) <= Q_T(a) <= U_K(a).

A lower bound must come from a feasible continuation through T. Closing at K
and waiting in cash is one possible lower policy when that close and subsequent
cash waiting are admissible. The upper search must permit inventory and orders
at K: imposing terminal flatness at an artificial cutoff would remove policies
that are allowed to close later at T and could invalidate an upper bound.
At each upper-search frontier, the cash bound above is one valid continuation
bound; sharper valid relaxations can replace it.

For a feasible selected action a_hat and any bound U_best on all feasible
actions' upper values,

    sup_a Q_T(a) - Q_T(a_hat) <= U_best - L_K(a_hat).

Proof: Q_T(a) <= U_best for every competitor, while
Q_T(a_hat) >= L_K(a_hat). Subtract these inequalities.
`action_interval_regret` proves the certificate. It applies to the original
continuous feasible action set only if the upper bound covers that entire set;
taking the maximum over an incomplete action menu is insufficient.

If L = L_K(a_hat) > 0, the relative regret is bounded by

    (V_star - Q_T(a_hat)) / V_star <= (U_best - L) / L.

`relative_interval_regret` proves this. Stop when the relevant absolute or
relative gap meets the requested accuracy. Small differences between two
successive truncated-horizon values do not establish this interval.

Separately, `approximate_policy_bound` specializes the existing continuous
approximate-selector theorem using the newly proved finiteness result:

    V_star - V_approx <= n * delta.

For n > 0, per-decision Bellman selection error at most epsilon/n suffices for
total value loss at most epsilon. This is an existence and error theorem for
the Lean approximate construction. The selection error is relative to the
true continuation value under the supplied model, not just the largest score
on a numerical grid. The theorem does not certify that the present numerical
planner attains those local gaps or give it an executable continuous optimizer.

## Why no universal shrinking lookahead bound follows

A delayed opportunity can preserve a first-action gap even when every position
closes immediately after the move it trades. The account retains cash through
the quiet interval, and that cash can be correlated with later opportunities.

Consider this fixed Oracle configuration:

| Parameter | Value |
| --- | ---: |
| Initial cash, holdings, price | 1000, 0, 100 |
| Entry and maintenance exposure bounds | [0, 1] |
| Quote order minimum / aggregate maximum | 5 / 10000 |
| Taker fee | 0.0005 |
| Signed-asset funding multiplier | 1 |
| Conservative position assessment rate | 0.00001 per held tick |
| Price grid / strict ceiling | 0.01 / 111 |
| Active order slots | 0 |

This is an unleveraged long/cash, market-only specialization. The numbers are
illustrative fixed model parameters, not a fitted exchange funding schedule.
The account is 200 times the order minimum and has fully feasible entries and
closes. Restricting the configuration this way suffices to test a proposed
guarantee for all Oracle configurations; it does not characterize the particular
leveraged NSF law used by the bot.

The first price is 110 or 90 with equal probability. It then stays at that
branch's price for d quiet ticks. Afterwards, only the 110 branch moves to 100
and then back to 110; the 90 branch stays at 90. End flat after this last move.
The branch is observed at the first tick, so subsequent actions are observable.

Any truncated prefix after the first move and before the later opportunity has
cash as its optimal first action. For initial purchase cost x >= 0, its expected
wealth is at most

    1000 - 0.0005 x.

All later prices in that prefix are flat, so later trades and assessment cannot
increase this bound. Waiting in cash achieves 1000. The upper argument permits
all continuous long/cash trades, not only selected example policies.

In the full problem, require the first action to remain cash. Even ignoring
all later fees and assessment, the 110 branch can grow that cash by at most
1.1 and the 90 branch cannot grow it. Thus

    Q_T(first action cash) <= (1100 + 1000)/2 = 1050.

Now consider a feasible competing policy: buy with all 1000 initially, close
fully at the first observed price, wait in cash through the quiet interval,
and in the upper branch buy at the later 100 and close at 110. Under the exact
Oracle received-amount fee convention, it produces:

| Branch | Terminal cash minus conservative assessment |
| --- | ---: |
| Initially 110, later 100 -> 110 | 1207.560835886827 |
| Initially 90, then flat | 899.09023 |
| Expected value | 1053.3255329434135 |

The exact expectation is 2106651065886827 / 2000000000000. It is independent
of d. The positions incur strictly negative assessed maintenance while held;
cash waiting has zero assessment. Both branches close completely, and every
nonzero quote trade lies between 5 and 10000.

The optimal full value is therefore more than 3 above the best full continuation
of the prefix-selected first action. Its relative first-action regret exceeds
3/1055, about 0.284%, for arbitrarily long quiet prefixes. This concerns loss
from choosing the first action, not the separate profits omitted when shortening
the terminal date.

**This is a family with arbitrarily delayed opportunities.** It rules out a
uniform bound tending to zero as K grows that depends only on these asset
parameters and K, uniformly over the allowed laws. It does not prove failure of
convergence for each fixed infinite law. For each member of this family,
extending the prefix through the final opportunity resolves it. A rate for a
specific law requires a justified bound on that law's remaining continuation
influence; the asset restrictions alone do not supply it.

## Formal and executable verification

`OracleFlatHorizonCounterexample` proves the actual Oracle transition's
long/cash growth inequality and lifts it over arbitrary real-trade traces.
It proves the prefix and cash-first upper bounds under those trace conditions,
the exact fee/assessment witness arithmetic, identity of an arbitrarily long
cash wait, flatness, order-size inequalities, and the resulting absolute and
relative first-action gap inequalities. The scenario-tree argument above
instantiates those conditions mathematically. A complete instantiation as a
`ContinuousStochasticPolicy.Model` and composition with its selected `Policy`
is not included; the gap theorem must not be presented as that larger Lean
construction.

The new exact rational execution test independently replays both witness
branches with delays 0, 1, 10, 1000 and 10000. It checks original action
admissibility, funding, positive equity, assessment, and exact flat finalization.
All five delays have the same rational expected payoff.

Validation:

- Full Lean build and public-theorem axiom audit.
- 35 execution, policy and new delayed-opportunity tests.
- Oracle TypeScript project typecheck.

Reproduce from the repository root:

```powershell
node scripts/theory-lean.mjs check
node --conditions=development --import tsx --test packages/bot-algo/test/oracle-flat-horizon.test.ts packages/bot-algo/test/oracle-crra-policy.test.ts packages/bot-algo/test/oracle-execution.test.ts
node node_modules/typescript/bin/tsc -p packages/bot-algo/tsconfig.oracle.json --noEmit
```

Logs: `data/backtests/oracle-flat-tail-bound-{lean,tests}.log`.
The current NSF planner does not yet compute certified upper/lower intervals
for its continuous action and learned transition approximations. No new
backtest or automatic stopping implementation is claimed by these proofs.
