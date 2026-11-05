# Oracle optimal-value convergence: an exact obstruction

Scope update: the user subsequently required equity to be much larger than
the minimum order. That excludes the small-account example below. It remains
a result about unrestricted Oracle accounts, but does not settle the revised
claim. The well-funded case is treated separately in
`oracle-crra-funded-value-convergence-2026-09-12.md`.

The unrestricted convergence statement is false for the actual Oracle model.
This result uses `ContinuousStochasticPolicy.Model`, its original admissibility
checker, and its complete continuous-control Bellman recursion. It is not an
assigned reward schedule or a forced-hold control parameterization.

The objective is the existing gamma-zero expected terminal wealth, divided by
the fixed initial equity as described in Oracle.md. No discount, time average,
log-wealth objective, or horizon-dependent normalization is introduced.

## An admissible instance

| Parameter | Value |
|---|---:|
| Initial quote balance | 0 |
| Initial asset quantity | 0.001 |
| Initial price | 100 |
| Initial equity | 0.100 |
| Subsequent deterministic prices | 101, 100, 101, 100, ... |
| Taker received-amount fee | 0.0005 |
| Maker received-amount fee | 0.0002 |
| Signed-asset funding multiplier | 1 |
| Raw entry exposure bounds | -2 to +2 |
| Effective maintenance bounds | -3 to +3 |
| Minimum nonzero quote order | 5 |
| Maximum aggregate quote traffic per decision | 10,000 |
| Maximum active limit orders | 1 |
| Reservation leverage | 2 |
| Allowed order-price grid | 0.01 |

The price law is a normalized PMF with probability one on the next specified
price. Its integer-valued support lies on the stated price grid. All real
market quote amounts, cancellation lists and passive limit placements remain
available as candidate controls. `controlAllowed` is the default True.

This is a small residual account, not the 10,000 initial cash account used in
the model backtests. A claim applying to every admissible Oracle account must
include it. The one-asset instance also obstructs a universal portfolio claim.

## Constraints imply no balance-changing action is feasible

At every mark, equity lies between 0.100 and 0.101. A received-amount market
trade with a nonnegative fee cannot increase current-mark raw equity. The raw
entry exposure bound gives |asset notional| <= 2*post-trade equity, hence

    |post-trade quote| <= 3*post-trade equity <= 0.303.

Initial quote is zero, so post-trade quote equals the actual signed market
quote movement dq. Every nonzero dq must have magnitude at least 5. Therefore
dq = 0 for every admissible action. This is `market_must_be_zero`.

Every valid new limit order has gross notional at least 5: a buy spends its
notional, while a sell receives its notional times 1-makerFee. At leverage 2 it
reserves at least 2.5, greater than the whole account equity. Hence no nonempty
placement list is admissible. Cancellation of an empty book changes nothing.
`admissible_rebalance_unchanged` proves these facts using the full checker.

The account stays wholly invested with effective exposure 1. Unit funding and
either positive mark preserve it through the actual liquidation check
(`dust_survives`, `next_state_exact`). Liquidation is enabled, not omitted.

The available received quote from closing the entire position is at most

    (1-0.0005)*0.001*101 = 0.1009495 < 5.

The close is nonovershooting and must obey the same size minimum. Therefore
zero is the only valid close; the residual is retained and marked.
`dust_close_only_zero` proves this rather than assuming that closing is disabled.

## Exact full Bellman value

For every horizon n, time t and recorded history h on this path,

    bellman(model, n, t, h, state_t) = ofReal(price_(t+n)/1000).

`bellman_exact` proves the equality by induction over the original recursion,
including the normalized expectation and supremum over all viable controls.
`every_policy_exact` proves that every feasible policy has this same value.
`hold_policy_feasible` and `hold_attains_bellman` exhibit an attained optimum;
the argument does not rely on nonattainment or infinite expectations.

Consequently, starting at time zero and dividing by initial equity,

    V_(2n) = 1,
    V_(2n+1) = 101/100.

`normalized_optimal_values_do_not_converge` proves that no finite real limit
exists. All optimal values are finite and bounded. The relative differences
between neighboring values also do not vanish in this instance. Optimality
does not ensure a positive one-step wealth increment: the fall from 101 to 100
reduces wealth even under the optimal feasible policy.

## What the result does and does not settle

Oracle's execution rules and normalized price laws alone do not establish the
requested convergence of optimal terminal values. In particular, the earlier
positive-additive-reward argument cannot be assumed to apply to every Oracle
state. A proof of that universal statement would contradict the checked result.

This does not show nonconvergence for the fitted NSF from the backtest's initial
cash state. Such a claim requires analysis of that law and those reachable
states. It also does not show bad action choices: holding is already optimal at
every horizon here. The failure concerns convergence of wealth levels across
different terminal dates, which is stronger than what is needed for action
selection in this example.

For a fixed backtest terminal date, a useful stopping certificate must bound
the same full-terminal action values while varying computation depth. Changing
the terminal date to the cutoff is a different sequence of optimization
problems. Existing interval and relative-advantage theorems remain applicable
when actual bounds for that fixed-terminal continuation are supplied; this
counterexample supplies no general one-hour or R/K truncation guarantee.

Formal source: `docs/theory/lean/TradingTheory/OracleValueConvergence.lean`.
All public results are included in `Audit.lean`. The full library build and
axiom audit passed with `node scripts/theory-lean.mjs check`; only the permitted
standard Lean axioms were reported. No proof holes or convergence axioms were
introduced. No forecasting, execution, or backtest implementation was changed.
