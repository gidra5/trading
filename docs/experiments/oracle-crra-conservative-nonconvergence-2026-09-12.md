# A fixed positive carrying charge does not imply value convergence

The subsequently requested uniform price ceiling excludes this particular
example. The [bounded-price analysis](oracle-crra-bounded-price-maintenance-2026-09-12.md)
examines the strengthened domain and the separate terminal-valuation issue.

The implemented conservative Oracle evaluator has a bounded, nonconvergent
optimal-value sequence, even with a fixed strictly positive assessment rate,
large account equity and the full continuous action space. This is a complete
Bellman counterexample, not merely a nonconvergent holding-policy payoff.

The statement concerns gamma=0 and terminal dates T increasing along one
fixed infinite deterministic price path. It does not compare distances between
optimizing actions. It also does not compare approximations to a fixed terminal
date while preserving the same terminal continuation value.

## Exact execution and evaluation

The counterexample uses the original Oracle funding engine and the conservative
ledger from `ConservativeOraclePolicy.lean`. Artificial assessment debits change
the payoff, not the balances used for actual trades, margin and liquidation.

| Parameter | Value |
| --- | ---: |
| Initial quote | 10,000 |
| Initial asset quantity | 1,000 |
| Initial price | 4,000 |
| Initial equity W0 | 4,010,000 |
| Maker / taker fee | 0.0002 / 0.0005 |
| Actual funding multiplier | 1 |
| Extra assessment rate eta | 0.00001 per decision, fixed |
| Entry exposure bounds | -2 to 2 |
| Maintenance exposure bounds | -3 to 3 |
| Minimum / aggregate maximum quote trade | 5 / 10,000 |
| Maximum simultaneous limit orders | 1 |
| Price grid delta | 0.01 |

Each decision uses the original arrival fills, rebalance, funding and liquidation.
All real market quantities, limit orders, cancellations and placements remain
subject to the original feasibility checks. There is no artificial restriction
to a holding strategy or a finite action menu. The terminal decision executes
arrival fills and the largest valid nonovershooting close using remaining
aggregate capacity, then marks the residual position.

With actual funding equal to one, the post-rebalance assessment is

    d_t = eta * |a_t p_t|.
    D_(t+1) = D_t + d_t.
    lowerW_T = max(0, actualTerminalWealth_T - D_T).

Every nonzero holding has a strictly negative assessed maintenance contribution
at the current mark. Subsequent price gains are still allowed. This is not a
claim that wealth must fall after both maintenance AND the next price move.

## One coherent deterministic path on a fixed price grid

Define the reference level B and the actual traded price p by

    B_0 = 4000
    p_t = delta * floor(B_t/delta) + 2*delta*(t mod 2)
    B_(t+1) = B_t + eta*p_t
    e_t = p_t - B_t

These definitions do not depend on the terminal horizon. They give

    -0.01 < e_t <= 0.02
    -0.01 < e_(2n) <= 0
     0.01 < e_(2n+1) <= 0.02.

Prices remain positive and on the 0.01 grid. They strictly increase, starting
4000, 4000.06, 4000.08, 4000.14, 4000.16, ... . At every step,

    p_(t+1) = (1+eta)*p_t + e_(t+1)-e_t.

The relative price increase is less than 0.00175% per decision. Prices grow
without bound over infinite time; this is an explicit assumption of the example,
not a claim about a fitted market law. The conservative optimal values below
remain bounded despite that growth.

## Upper bound for every feasible policy

For a terminal date T define a shadow price

    phi_t = B_t + e_T.

It satisfies phi_T=p_T, phi_(t+1)-phi_t=eta*p_t, and
|phi_t-p_t|<0.03. Consider the potential

    F_t = q_t + a_t*phi_t - D_t.

The full original execution cannot increase this potential:

1. A market trade at the current price, or any crossed limit fill, cannot
   increase q+a*phi. The maker fee already covers the local price envelope
   [p_(t-1)-0.03, p_t+0.03]:

       (1-makerFee)*(p_t+0.03) <= p_(t-1)-0.03.

   `OracleLimitShadowBound` derives this statement using the exact Oracle buy,
   sell and fill formulas. The bound covers pending orders as well as current
   placements, without assuming that an optimizer avoids limit orders.

2. Between decisions, actual funding is one, and the change in the potential
   from shadow-price growth and the assessment debit is

       eta*p_t*(a_t-|a_t|) <= 0.

   For a long it is exactly zero; for a short it is nonpositive.

3. Liquidation zeroes the real account and leaves potential -D<=0, preserving
   the positive upper bound. Cancellations and placements do not change account
   balances. The terminal close also cannot increase the potential.

At the initial state F_0=W0+1000*e_T. At the terminal date phi_T=p_T, so
the potential is actual marked equity minus the assessment ledger. Its positive
part is exactly the conservative terminal payoff after the actual terminal close.
Therefore, for the supremum over all admissible complete policies,

    lowerV_T <= W0 + 1000*e_T.

In particular every even-horizon optimal value is at most W0. The Lean proof
`bellman_upper` includes the original Bellman recursion, not just fixed-inventory
arithmetic or an assumption that holding attains the optimum.

## Feasible lower bound

Hold the original long with no orders or trades until the terminal close.
Cash is positive, exposure is below one, and the account remains solvent.
The initial account equity is more than 800,000 times the minimum order.
The holding policy and its terminal close pass the original feasibility checker.

The assessment telescopes:

    D_T = 1000*(B_T-4000).

The terminal close sells enough asset to receive 10,000 quote units. Its fee
in marked equity is exactly

    c = takerFee*10000/(1-takerFee) = 10000/1999
      = 5.002501250625... .

Most of the position remains marked because the maximum is aggregate per
decision. This is not an order-minimum dust example. Full closing across earlier
decisions is available to competing policies and is covered by the upper bound.

The holding witness therefore has conservative payoff

    W0 + 1000*e_T - c > 0.

This is a lower bound for the optimum; it need not itself be optimal. Combining
it with the universal upper bound gives

    W0 + 1000*e_T - c <= lowerV_T <= W0 + 1000*e_T.

Consequently, for every n,

    W0-16 < lowerV_n <= W0+20
    lowerV_(2n) <= W0
    lowerV_(2n+1) > W0+4.

The sequence of optimal conservative values is bounded but cannot converge.
The gap persists after dividing by initial equity: it always exceeds
4/4,010,000, or approximately 0.009975 basis points. This is small and may be
acceptable for a practical tolerance; it is not a factor that tends to zero
with a larger planning horizon.

## Scope of the obstruction

The two mechanisms used here are growth that offsets a fixed carrying charge
and terminal exposure that cannot all be closed in a single capped decision.
Strictly negative maintenance alone excludes neither. Quantization and fees
allow the remaining adjusted-value oscillation to survive optimization.

This does not prove that all funding or borrowing configurations fail to converge,
or that a one-hour approximation is practically inaccurate. It disproves exact
general convergence under the current stated assumptions. It also does not
establish a result for an alternative account that physically pays artificial
charges and computes margin on those reduced balances. That is a different
execution, for which a new simulation/dominance proof would be needed.

The earlier scalar retention example only showed that strictly decreasing
retention need not go to zero when its rates approach zero. This example instead
keeps eta fixed and proves bounds on the full conservative Oracle Bellman value.

## Verification

- `ConservativeCounterPath.lean`: the infinite path, fixed price grid, parity
  bounds, monotonicity and fee envelope.
- `ConservativeValueCounterexample.lean`: the all-policy upper bound, legal
  holding witness, finite bounded optimal values, persistent gap and failure
  of convergence, including values normalized by initial equity.
- The exact rational execution test in
  `packages/bot-algo/test/oracle-conservative-maintenance.test.ts` replays 4096
  holding decisions, checking every assessment, terminal close, residual and
  payoff formula. A finite replay supplements, rather than replaces, the proof.

Validation commands:

    node scripts/theory-lean.mjs check
    node --import tsx --test packages/bot-algo/test/oracle-conservative-maintenance.test.ts

The full proof/audit log is `data/backtests/oracle-conservative-counter-lean.log`.
The new modules are imported by `TradingTheory.lean`; all 49 public theorems
are listed in `Audit.lean`. No model training or trading-policy backtest is
required for this deterministic counterexample.
