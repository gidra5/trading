# Bounded prices expose a terminal-valuation obstruction

**Historical objective:** this counterexample permits a marked residual after
the terminal close. The subsequently required zero-holdings terminal constraint
excludes its witness. The four historical convergence models explicitly retain
their alternative predicate; current constructed policies and backtests require
a valid complete close. See [the flat-terminal follow-up](oracle-crra-flat-terminal-2026-09-12.md).

The user added a uniform price ceiling: there is a fixed P such that every
future price remains below P. For a stochastic law, apply the restriction at
every supported hypothetical history, not only to the observed training path.

`BoundedPriceConservativeBellman.PriceCeiling` records this assumption. Together
with the existing positive price grid it gives finite one-step support:

    j < floor(P / priceStep).

This excludes the preceding counterexample with unbounded prices. It does
not by itself make the implemented conservative terminal values converge.
There is another example with prices confined to {100, 100.04}, constant
parameters and strictly negative assessed maintenance. It isolates valuation
of the inventory left after a capped terminal close.

## Domain and execution

Keep the actual Oracle engine, gamma=0, and its conservative assessment ledger.
Funding and borrowing are alternatives; only signed-asset funding is used here.
The artificial assessment does not change real execution balances.

| Parameter | Value |
| --- | ---: |
| Initial cash / asset | 1,000,000 / 0 |
| Price at even / odd ticks | 100 / 100.04 |
| Uniform strict ceiling P | 101 |
| Maker / taker fee | 0.0002 / 0.0005 |
| Signed-asset funding multiplier k | 1.0002 per decision, fixed |
| Extra assessment rate eta | 0.00001 per decision, fixed |
| Entry exposure / maintenance bounds | [-2,2] / [-3,3] |
| Quote minimum / aggregate maximum | 5 / 10,000 |
| Simultaneous limit orders | At most 1 |
| Price grid | 0.01 |

The deterministic next-price PMF is identical at every history with the same
time parity. It never leaves the bounded price domain. These are illustrative
per-decision parameters, not calibrated live funding or return estimates.

After actual arrival fills and rebalancing, record

    d = max((k-1)*a*p, 0) + eta*|a*p|.

For a long the funding credit is removed from assessment, and the remaining
assessed maintenance contribution is -eta*a*p. For a short it is
-(k-1+eta)*|a*p|. Both are strictly negative for a nonzero position, before
marking at the next price.

Finalization executes the available valid close, with total quote movement
at most 10,000, and retains the remaining inventory at the current mark.
Its conservative payoff is max(0, final actual wealth minus accumulated d).

## Upper bound over every admissible policy at low-price horizons

Use two auxiliary inventory valuations:

    C = 100/(1-maker) = 100.0200040008...
    b(100) = 100
    b(100.04) = (1-maker)*100.04 = 100.019992.

Define

    F(q,a,p,D) = max(q+a*C, q+a*b(p)) - D.

Since C>=b(p), this values a long at C and a short at b(p).
The complete Oracle transition cannot increase F:

1. Each market trade cannot increase either relevant auxiliary valuation.
   Buying costs at least the auxiliary long value; selling receives at most
   the auxiliary short value. This includes trades crossing through flat.
2. All crossed limit fills obey the same inequalities. At price 100, arrivals
   can only be buys; their execution prices are at least 100. At price 100.04,
   the maker spread covers both auxiliary valuations. The proof permits any
   admissible pending book; it does not assume the optimal policy avoids limits.
3. For a long, funding followed by the assessment changes F by

       a * ((k-1)*(C-p) - eta*p) <= 0.

4. For a short, the change is

       a * (k*b(p_next) - b(p) + eta*p) <= 0,

   because a<0, b(p_next)>=100, and k*100=100.02>b(p).
5. Liquidation maps the real account to zero and hence F to -D<=0.
   Placement and cancellation do not change balances. The terminal closing
   trade also cannot increase F.

These account and fill inequalities are proved in
`BoundedPriceConservativePotential.lean`. The theorem
`BoundedPriceConservativeBellman.bellman_upper_low` lifts them through the
original conservative Bellman recursion, including continuous market amounts,
limit orders, fees, funding, liquidation and the specified terminal close.

Initially F=1,000,000. At the low price, actual equity minus D is at most F,
for either sign of inventory. Consequently every feasible policy satisfies

    lowerV_(2n) <= 1,000,000.

Remaining in cash attains 1,000,000, with no funding assessment, orders,
liquidation or closing costs. Thus the low-horizon optimum is exactly the
initial cash balance. This bound is against all policies, not against a
preselected strategy.

## A feasible high-horizon witness

For terminal tick 21:

1. At tick 0 remain in cash.
2. At high-price ticks 1,3,...,19, place one limit buy at price 100 with
   gross quote notional 10,000.
3. Each order first fills at the next low-price tick 2,4,...,20. Its gross
   quote cost is 10,000 and it receives 99.98 asset units, under the exact
   received-asset maker fee.
4. Hold the accumulated inventory, applying the actual funding multiplier
   and recording the assessment every decision.
5. At high terminal tick 21, execute the permitted 10,000-quote close.

The rule uses at most one active order, all placements are strictly passive,
every fill is delayed to the next tick, and all prices lie on the fixed grid.
There is at most one 10,000 fill per decision and no market trade at those
decisions. Cash remains at least 900,000, collateral and leverage constraints
hold, and no liquidation occurs. The terminal close leaves positive inventory.

Exact rational execution gives, rounded here for display:

    Actual terminal marked wealth: 1,000,215.2757469141
    Accumulated assessment debit:        210.2543143341
    Conservative payoff:          1,000,005.0214325800
    Residual asset quantity:              901.7920406529

There is a direct rational expression for this payoff. Let u=99.98 and
N=10,000. A lot filled 2j+1 decisions before the terminal date contributes
u*k^(2j+1) terminal asset units. Its assessment is

    (k-1+eta)*u *
      [100*sum(i=0..j, k^(2i)) + 100.04*k*sum(i=0..j-1, k^(2i))].

Sum over j=0,...,9, subtract the 10*N purchase cost and the terminal fee
taker*N/(1-taker). `witness_gain_positive` proves in Lean that the resulting
gain is strictly between 5 and 5.03. The exact execution test independently
checks the original engine's states, feasible actions, charges and payoff.

For any later odd terminal date T>=21, remain in cash for T-21 ticks, then
execute this same 21-tick block. T-21 is even, so the price sequence and all
constant parameters match exactly. Cash waiting costs nothing; placement
dates shift with the block. The payoff is unchanged. Therefore

    lowerV_(2n) = 1,000,000,
    lowerV_(2n+1) >= 1,000,005.02143258... for n>=10.

These subsequences cannot converge to a common value. The relative separation
exceeds 5/1,000,000, or 0.05 basis points, and does not shrink as the horizon
increases. This concerns terminal dates moving along a fixed infinite path,
not numerical refinements at one fixed terminal date.

## What remains in the way

The optimizer can build fresh inventory shortly before each high terminal
date. A price ceiling and a positive carrying cost can penalize a stale position
without excluding this behavior. The marked residual makes the high-date
payoff larger even though the position cannot be completely sold at that tick.

Charging the closing fee on the entire residual removes this witness's gain:
its full fee-adjusted liquidation valuation is below the initial 1,000,000.
That calculation is a valuation comparison, not an executable order above the
aggregate cap. A realizable liquidation objective would instead have to specify
the later valid closing decisions, their prices, funding and costs.

This identifies a terminal-valuation issue to resolve before claiming a
convergence theorem. Neither changing the terminal valuation nor adding a
price ceiling by itself proves a general bound on the whole continuation.
The bot's actual terminal execution has not been changed by this analysis.

## Verification scope

- The price ceiling, finite supported-index bound, full-transition potential,
  full Bellman upper bound at every low horizon, and exact rational gain
  inequality are proved in the two new Lean modules and audited.
- The complete finite witness is replayed with exact rational execution for
  terminal ticks 21,23,101,1021. All four payoffs agree exactly, including each
  prefix of cash waiting. All six maintenance tests pass.
- The analytical witness construction and cash-prefix extension above give
  the other side of the counterexample. Its complete policy feasibility and
  equality to the displayed closed form are not yet composed into a single
  Lean `Policy` nonconvergence theorem. Do not confuse the engine replay with
  such a theorem or with a compiler-correctness result.

Commands:

    node scripts/theory-lean.mjs check
    node --import tsx --test packages/bot-algo/test/oracle-conservative-maintenance.test.ts

The full proof/audit log is `data/backtests/oracle-bounded-counter-lean.log`.
