# Oracle optimal values with equity far above the order minimum

The revised assumption removes the earlier account with only 0.100 equity.
It does not, by itself, turn every order-minimum effect into a vanishing
function of lookahead. The new example has arbitrarily large cash, feasible
nonzero market orders, and the original continuous-control Bellman recursion.

The result concerns gamma-zero terminal wealth divided by fixed initial
equity. It does not concern distance between optimizing actions, nor does it
replace the objective with average reward or introduce a discount.

## Exact instance

| Parameter | Value |
|---|---:|
| Initial quote balance | C, any fixed C >= 10,000 |
| Initial asset quantity | 0.001 |
| Initial price | 100 |
| Subsequent deterministic prices | 100.01, 100, 100.01, 100, ... |
| Initial equity | C + 0.1 |
| Taker received-amount fee | 0.0005 |
| Maker received-amount fee | 0.0002 |
| Signed-asset funding multiplier | 1 |
| Raw entry exposure cap | 2 in absolute value |
| Effective maintenance bounds | -3 to +3 |
| Minimum actual quote order | 5 |
| Maximum aggregate quote traffic per decision | 10,000 |
| Maximum active limit orders | 1 |
| Reservation leverage | 2 |
| Price grid | 0.01 |

These are an explicit Oracle instance, not a claim about a particular live
exchange's current fee tier or the fitted NSF distribution.

All real market quote amounts, cancellation lists, and limit-order placements
remain candidate controls. `controlAllowed` remains True. Original feasibility
checks enforce their size, collateral, passivity, activation, grid, slot and
aggregate-cap constraints. The theorem `nonzero_market_admissible` checks an
actual five-quote-unit buy from the initial state for every C >= 10,000.
`equity_large` proves equity exceeds 2,000 times the minimum along the optimal
holding path. C can be increased without an upper bound; the example is not
limited to that ratio.

Finalization first executes eligible arrival fills, then uses the remaining
aggregate capacity for a valid nonovershooting close. A remaining executable
amount below 5 becomes zero. Unfilled orders are canceled under the existing
terminal contract, and asset residuals remain marked. The closing amount is
computed from each actual terminal account and its fills, rather than set to
zero for all policies.

## Why no alternative sequence can improve the value

Fix any terminal date T and its price P_T. Value each intermediate account by

    S_T(q,a) = q + a P_T.

For a trade at price p and received-amount fee f, Oracle's exact buy/sell
formulas give S_T(after) <= S_T(before) whenever

    (1-f) P_T <= p,
    (1-f) p <= P_T.

All market execution prices lie in [100,100.01]. Every limit order that
actually fills must have its execution price between the two crossing prices,
so its price lies in that interval too. Both inequalities therefore follow from

    (1-0.0002) * 100.01 = 99.989998 <= 100.

This covers arbitrary buys and sells, resizing, shorting, and every filled
limit order. There is no restriction to fixed holdings in the upper bound.
Placement and cancellation do not change balances. Unit signed-asset funding
and marking leave S_T unchanged. Liquidation to zero preserves the nonnegative
capital upper bound even if an intermediate shadow valuation became negative.
The same trade inequality covers the actual final close.

`OracleLimitShadowBound` proves those statements for the actual chronological
machine, including arrival fills and `finish_is_oracle_liquidation`.
`bellman_upper` applies them inductively to the original Bellman expectation
and supremum over all viable continuous controls. Consequently,

    V_T <= C + 0.001 P_T.

Holding attains that bound. Its tiny asset balance remains below the closing
minimum, but its account is well funded and can trade normally. Increasing
the position to make a later close possible, or selling enough to overshoot
flat before finalization, incurs fees and cannot improve the bound.
`hold_policy_feasible`, `hold_value`, and `bellman_exact` prove attainment and
the exact optimal value, including the actual terminal procedure:

    V_T = C + 0.001 P_T.

## A small constant factor is not a horizon decay factor

Write v_T = V_T / (C+0.1). The checked identities are

    v_(2n)   = 1,
    v_(2n+1) = 1 + delta(C),
    delta(C) = 1 / (100000 * (C+0.1)) > 0.

For the fixed minimum m=5 and initial equity W0=C+0.1, this is

    delta(C) = (m/W0) / 500000.

The minimum/equity ratio really is a constant multiplier here, but the
remaining factor is also constant. There is no factor 1/n. Increasing n does
not reduce the oscillation. Increasing C does reduce it; those are different
limits. `persistent_gap` and `normalized_optimal_values_do_not_converge` prove
the fixed-account statements. `gap_small_for_large_cash` proves that raising
cash makes the gap smaller than any prescribed positive tolerance.

If the large-equity condition is additionally imposed throughout a policy's
path, the holding policy satisfies it at every tick. Removing competing
policies that violate that condition cannot improve the upper bound or remove
this attaining policy, so the same optimal values remain.

At C=10,000, the full optimal dollar values alternate between 10000.10000 and
10000.10001. Their relative difference is approximately 1e-9. This is an exact
nonconvergence result with a practically negligible amplitude.

`all_horizon_value_width` also proves the useful positive result

    |v_n - v_m| <= delta(C) for every pair n,m.

Thus every horizon is already within that tiny tolerance of every other
horizon, even though there is no single limiting value. This bound is specific
to this price-band instance; it is not a universal error floor or a universal
bound proportional to the order minimum for arbitrary Oracle price laws.

## Consequence for the planner

If a bound of the form constant * error(n) were established with error(n)->0,
the user is correct that multiplying by the constant would preserve
convergence. Large equity alone does not establish such a bound.

This example does not demonstrate a planning failure: `hold_attains_bellman`
proves that the same policy is optimal at every horizon, so its action regret
is exactly zero. It separates exact convergence of optimal wealth levels from
the practical goal of a sufficiently accurate action. A general computable
stopping certificate for the NSF still needs bounds on its omitted
continuation; no R/n rate or one-hour guarantee follows from this example.

The old small-account report now explicitly states that it is excluded by the
revised large-equity condition. No model training, backtest, or execution
implementation was changed by this proof work.

Formal sources:

- `docs/theory/lean/TradingTheory/OracleLimitShadowBound.lean`
- `docs/theory/lean/TradingTheory/OracleFundedValueConvergence.lean`

Both modules are imported into `TradingTheory.lean`, with every public theorem
listed in `Audit.lean`. The full `node scripts/theory-lean.mjs check` build and
axiom audit passed; only propext, Classical.choice and Quot.sound were reported.
No proof holes or additional axioms were introduced. The validation log is
`data/backtests/oracle-funded-value-convergence-lean.log`.
