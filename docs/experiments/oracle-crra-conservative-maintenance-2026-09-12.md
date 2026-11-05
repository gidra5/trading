# Conservative maintenance: valid lower values and an explicit error budget

Funding and borrowing are alternatives here. No result composes their costs.
The user authorized an artificial carrying cost for every nonzero position in
order to obtain conservative action values and avoid relying on persistent
residual holdings.

The lower-bound part can be proved. Strictly negative maintenance alone does
not establish a general convergence rate, a small approximation error, or a
one-hour cutoff. This construction records its pessimism explicitly so those
claims can be checked rather than assumed.

## Assessment rule

At each decision, first execute the actual arrival fills and rebalance. Let
N_t=|a_t p_t| be gross held notional after those executions. Apply the selected
actual maintenance rule at the current mark and measure its wealth change g_t.
For an extra rate eta>0, record an additional evaluation debit

    d_t = max(g_t,0) + eta N_t.

The assessed maintenance contribution is

    g_t - d_t = min(g_t,0) - eta N_t < 0 whenever N_t > 0.

It is also no greater than the actual maintenance contribution. This matters:
an arbitrary negative replacement cost could still be less costly than actual
maintenance, and would then fail to give a lower value.

For Oracle signed-asset funding with multiplier k_t,

    g_t = (k_t-1) a_t p_t.

For Oracle's separate borrowing alternative with nonnegative rates,

    g_t = -r_q max(-q_t,0) - p_t r_a max(-a_t,0) <= 0,
    d_t = eta N_t.

The borrowing equation is proved equivalent to the positive/negative-part
formula in Oracle.md. Borrowing and funding are not applied together.

The assessment occurs before the next price mark. Price returns are not
silently reclassified as maintenance credits and removed. Cancellation,
placement and reservations retain their original execution semantics.

## Full-policy lower bound

The debit is an evaluation ledger D, initially zero, updated by D'=D+d_t.
The real account retains its actual balances, funding/borrowing, price drift,
fees, liquidation checks, order sizes, aggregate cap and pending fills. A
policy may use its observable account and this ledger when deciding actions.

At the same specified terminal date T, perform the real terminal close and
retain residuals under the original contract. If its nonnegative wealth is
W_T, evaluate

    lowerW_T = max(0, W_T - D_T).

The floor is safe because feasible actual terminal wealth is nonnegative.
The strict maintenance statement concerns the raw assessed phase contribution
before this final floor, not a claim that a zero lower payoff decreases further.

For every feasible complete policy and every path,

    0 <= lowerW_T <= W_T.

Taking the actual path expectation, then optimizing over the same legal
continuations with the same forced first action a, gives

    lowerQ_T(s,a) <= Q_T(s,a),
    lowerV_T(s) <= V_T(s).

`ConservativeOraclePolicy.lowerValue`, `lowerBellman` and `lowerScore` implement
this recursion with the original continuous action set and original price PMF
at each history. `lower_action_value` and `lower_bellman_le_actual` prove the
inequalities. They do not rely on a finite action menu, finite price support,
an attained optimizer, or a heuristic averaging of deterministic policies.

The full Bellman integration uses the existing funding execution engine.
`lower_optimum_for_any_execution` proves the payoff comparison independently
of the engine: it also applies when the actual legal policy set, terminal
wealth and path law come from the borrowing alternative. The borrowing phase
and its assessment are formalized separately; a new complete borrowing
execution backend was not silently substituted into the funding engine.

Changing the physical balances of a simulated account instead would require
a separate proof that its actions and future order fills can be implemented
in the actual account and that terminal payoffs remain ordered. A one-step
equity inequality does not establish that simulation relation. The ledger
construction avoids needing such an assumption.

## How much pessimism was introduced?

A lower bound alone can be very loose. Suppose B_n(t,h,s) covers the maximum
expected remaining assessment debit, in the following checkable sense for
every allowed action:

    E[d_t + B_n(t+1,h',s')] <= B_(n+1)(t,h,s).

The nonnegative base budget may be zero. `bellman_budget_bound` proves

    lowerV_n(s,D) <= V_n(s) <= lowerV_n(s,D) + D + B_n(s).

For a forced first action, `action_budget_bound` provides the tighter width

    Q_(n+1)(s,a) <= lowerQ_(n+1)(s,a,D)
                   + D + E[d_t + B_n(s')].

These budgets can depend on the real state and history. Using infinity
outside a forward-invariant reachable set avoids needing a finite global
wealth bound on every conceivable account. Bounds must cover all supported
outcomes and allowed actions in their claimed domain; observed training paths
or a finite sample of forecast scenarios alone are not such a certificate.

If every extra debit is bounded by c on the relevant domain, the simpler
bound is D+n*c for an n-transition value. The corresponding action-value width
is D+(n+1)*c. The proved `chosen_action_bound` shows that etaOpt-optimal ranking
of the conservative first-action scores loses at most

    (n+1)*c + etaOpt

against the actual best first action when the starting ledger is zero. This
is a bias bound, not a claim that increasing the horizon makes the bias vanish.
For funding, `funding_debit_upper` gives the useful pointwise estimate

    d_t <= (|k_t-1| + eta) N_t.

An actual stopping rule still needs lower/upper intervals for the same full
terminal objective, covering numerical/planning errors as well as this debit
budget. The existing `ContinuationValueBounds.interval_stop` proves the
epsilon-regret certificate when max_a upper(a) <= lower(chosen)+epsilon.
No automatic computational truncation is derived merely by adding a charge.

## Application to the earlier residual example

The initial real state has cash 10,000 and 0.001 asset at price 100: total
equity 10000.1. Maker/taker fees are 0.0002/0.0005, the quote minimum is 5,
the aggregate maximum is 10,000, leverage is 2, maintenance bounds are -3/3,
and the price grid is 0.01. Funding is one; the deterministic next prices
are 100.01,100,100.01,100,... .

Take eta=0.00001 per decision as an explicit illustrative assessment rate.
It is not a calibrated live carrying-cost estimate.

A feasible continuation clears the residual:

1. Buy 5 quote units at price 100, satisfying the minimum. Asset quantity
   becomes 0.050975. The assessment debit for that holding interval is
   0.000050975.
2. At price 100.01, sell the whole asset. Received quote is 5.095460745125,
   satisfying both order-size bounds. The account is now entirely cash with
   no resting orders.
3. Remain in cash through the remaining terminal date. There is no further
   position assessment or closing fee. Price movements and positive funding
   multipliers do not change a cash-only account.

Exact rational execution gives

    actual cash after cleanup = 10000.095460745125,
    conservative payoff      = 10000.095409770125.

This is a feasible-policy lower bound for every horizon containing those two
decisions. It is not an assertion that this policy maximizes the conservative
Bellman recursion. That optimum is at least as large as this witness.

The previously proved full optimal value in this price-band instance is at
most 10000.10001. Therefore the displayed conservative payoff is within

    0.004600229875 quote units,
    less than 0.005 basis points of initial equity,

of that actual optimum, uniformly over all those terminal dates. The lower
payoff is constant after cleanup, even though the actual optimal marked
residual value keeps oscillating. This removes the residual problem from this
particular lower-bound policy at a quantified, small cost. The difference
from initial equity is a loss; inherited asset value is not trading profit.

The numerical witness is checked using the real rational Oracle execution
functions. The formal policy inequalities are general Lean proofs; the
numerical engine is not claimed to be extracted from Lean or verified by a
compiler-correctness theorem.

## What is still not implied

Strictly decreasing maintenance does not imply a uniform rate of retention
decay. For example, for any 0<l<1,

    retention_n = 1-l + l/(n+1)

strictly decreases from one but stays above 1-l. The associated per-step
retention factors are all below one; nevertheless a positive residual remains.
These sequence facts are proved in `ConservativeMaintenance`. They explain
why an unquantified strict inequality is insufficient; they are not presented
as a complete Oracle Bellman counterexample.

Even a fixed positive position charge is not a Bellman discount: equal debits
preserve differences between sufficiently positive value levels, price gains
can offset carrying costs, and policies can open new positions. Thus neither
a uniform contraction nor convergence follows from the maintenance assumption
alone. A complete fixed-rate Oracle Bellman counterexample is now proved in
`ConservativeValueCounterexample.lean` and explained in
[the counterexample report](oracle-crra-conservative-nonconvergence-2026-09-12.md).
Its conservative optimal values are bounded but separated at even and odd
horizons; it uses large equity, the aggregate maximum close and a fixed price
grid, without a shrinking assessment rate. A convergent conservative approximation
may also retain bias relative to actual value. Its interval width must be
bounded explicitly before claiming it is accurate enough.

## Implementation and verification

- `ConservativeMaintenance.lean`: the two separate actual maintenance phases,
  strictly negative assessment, and limits of strict-decrease reasoning.
- `ConservativeOraclePolicy.lean`: original-Oracle lower values, action bounds,
  monotonicity under additional charges, and explicit bias budgets.
- `oracle-conservative-maintenance.ts`: exact rational assessment helpers and
  the terminal payoff ledger. The live policy and execution balances are not
  changed automatically by adding this opt-in evaluator.
- `oracle-conservative-maintenance.test.ts`: separate funding/borrowing cases,
  strict phase timing, and the valid residual-cleanup witness above.

Verification commands:

    node scripts/theory-lean.mjs check
    node --import tsx --test packages/bot-algo/test/oracle-conservative-maintenance.test.ts
    node node_modules/typescript/bin/tsc -p packages/bot-algo/tsconfig.oracle.json --noEmit

The full Lean build and axiom audit passed, including every public theorem in
both new modules. Only propext, Classical.choice and Quot.sound were reported;
no proof holes or new axioms were introduced. The log is
`data/backtests/oracle-conservative-maintenance-lean.log`. Four exact-execution
tests and the Oracle TypeScript typecheck passed. No forecast training or NSF
backtest was run for this theorem work.
