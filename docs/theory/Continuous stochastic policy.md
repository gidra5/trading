# Continuous stochastic Oracle policy

The Lean construction now optimizes over real trade controls rather than a
finite menu. It permits a different next-price distribution at every tick and
observed price history, and optimizes expected wealth at one fixed terminal
tick. This retains the requested risk aversion **gamma = 0**.

There is an important distinction between defining a supremum and attaining
it. Exact optimality is proved under compactness and upper semicontinuity
conditions. Without those conditions, finite Bellman values still permit an
arbitrarily near-optimal policy. Oracle's valid-close/dust convention can
prevent an exact optimum from existing; a concrete example is also proved.

## Price laws and execution

`ContinuousStochasticPolicy.Model.law` has type
`Nat -> History -> PMF Nat`. Outcome index `j` means price `(j + 1) * priceStep`,
with a positive grid step. The support may be countably infinite. This matches
Oracle's price grid without discretizing trade quantities. It does not claim
to cover non-atomic probability measures on arbitrary real prices.

For the bounded-price convergence investigation, additionally require a uniform
ceiling P over every supported hypothetical history, as well as a current
price in the bounded domain. `BoundedPriceConservativeBellman.PriceCeiling`
records this restriction; `ceiling_bounds_supported_index` proves
`j < floor(P / priceStep)` for every supported outcome. The general Bellman
construction remains valid without this extra assumption. This is a theory
domain restriction, not a claim that the trained NSF already has bounded support.
See the [bounded-price analysis](../experiments/oracle-crra-bounded-price-maintenance-2026-09-12.md)
for the remaining issue with residual terminal valuation.

For a time-dependent sequence of next-price laws, supply
`law := fun t _ => laws t`. For conditional laws, use the observed history too.
Histories store the newest observation first. Laws for hypothetical future
histories must be supplied when evaluating their continuations. Forecasts
saved later on the actual realized path are not known counterfactual inputs.

The original `LimitOrderMachine.step` performs arrival fills, cancellation,
market execution, placement, signed-asset funding, marking, and liquidation.
Fees, funding and maintenance terms are known at the current decision; only
the template's next price is replaced by the sampled outcome. All original
`LimitOrderAdmissibility.Admissible` restrictions remain: quote-size minima,
aggregate trade caps, price restrictions, passive orders, activation, slots,
reservations and configured account constraints. Control-specific restrictions
are additional, not replacements for these checks.

The terminal contract performs arrival fills and one specified valid,
nonovershooting closing order, cancels the remaining book, and values the
actual residual account. Arrival fills and the close share the terminal
decision's trade cap. Additional closing decisions belong in the supplied
horizon; there is no unlimited terminal order allowance. Terminal feasibility
requires nonnegative actual equity. Its conversion to a nonnegative extended
real is proved to preserve that equity exactly.

## Continuous exposure construction

`exposureControls` specializes the execution environment to `Control = Real`.
A target can be any real value inside its current lower and upper exposure
bounds, provided its exact physical trade satisfies the order and account
rules. Minimum sizes can leave gaps inside the band. Targets are not rounded
to a grid or silently clamped to a different exposure.

`admissible_exposure_reaches` proves that an admissible target is the exposure
actually achieved after arrival fills and fee-aware market execution. Target
zero is available as a no-op for an account already zeroed by liquidation.
The scalar specialization retains existing orders but does not place or cancel
them. For simultaneous market/order control, the same Bellman construction
accepts `Control = LimitOrderMachine.Action` with an identity encoder. Its
exact existence conditions must then hold on that full action space.

Write `n` for the number of remaining decisions, `t` for current time, `h` for
observed history and `s` for the full account/book state. Let `A_n(t,h,s)`
contain current controls satisfying the original rules and admitting feasible
continuations after every positive-probability next price. The construction is

\[
V_0(t,h,s)=W_{\mathrm{terminal}}(t,h,s),
\]
\[
Q_n(t,h,s,a)=\sum_{j=0}^{\infty}P_t(j\mid h)
  V_n(t+1,j::h,F_t(s,a,j)),
\]
\[
V_{n+1}(t,h,s)=\sup_{a\in A_n(t,h,s)}Q_n(t,h,s,a).
\]

The action is chosen before observing `j`; each child policy is selected only
after that observation. `every_policy_le_bellman` bounds every feasible
nonanticipative contingent policy, including ones with different actions and
subsequent holdings. The comparison is over the entire remaining horizon.

`compact_usc_attainment` derives a maximizing action from a nonempty compact
viable action set and an upper semicontinuous Bellman score on that set.
`construct` recursively selects it. `construct_attains` proves feasibility
and equality with the Bellman value. `continuous_exposure_policy_optimal`
specializes the global optimality result to bounded real exposures; it needs
closed viable action sets and upper semicontinuous scores. Boundedness follows
from the encoded exposure band.

These are sufficient conditions, not assertions that Oracle always satisfies
them. In particular, continuity of the price forecast alone does not establish
continuity of the account value across liquidation or closing boundaries.

`constructUntil m T t h s` uses `T - t` remaining decisions.
`fixed_terminal_optimal` proves the endpoint remains exactly `T`, rather than
moving forward each time the policy is called. These results use arbitrary
finite horizons; they do not assert infinite-horizon attainment.

## Near-optimal construction when a maximum does not exist

Assume the Bellman values at viable states are finite. For any positive
per-decision tolerance `delta`, `approximate_action_exists` obtains a feasible
action whose Bellman score is within `delta` of the supremum. Recursing gives
`approxConstruct`, and `approx_construct_bound` proves

\[
V_n(t,h,s)\le
\mathbb E[W_T^{\pi_\delta}\mid t,h,s]+n\delta.
\]

`approx_construct_optimal` therefore bounds the expected-wealth advantage of
every competing feasible policy by `n * delta`. For positive `n`, setting
`delta = epsilon / n` gives any desired positive total tolerance. No
compactness, continuity, exact maximizer, finite action menu or finite price
support is required for this result. This tolerance is an expected terminal
wealth error; it is not an exposure rounding tolerance or a pathwise guarantee.

Both selectors use classical choice and exact Bellman values. These are Lean
mathematical constructions, not executable numerical solvers. A deployed
approximation must establish its value/optimization errors separately; it
cannot inherit the theorem merely by using a Bellman-shaped calculation.

## Why unconditional exact existence would be false

`ContinuousTradeNonattainment` uses the original fee and funding execution:

- Initial cash 10, no holdings, initial price 1.
- Deterministic next price `4003 / 4000` (a 7.5-bps increase), unit funding.
- Received-amount fee `1 / 2000` (5 bps per side).
- Unleveraged long holdings; maintenance bounds `[0,1]` and no liquidation.
- Actual order minimum 5 and maximum 10, also applying to the final close.
- At the endpoint, close as much as validly possible and mark uncloseable dust.

Let `x` be the initial gross quote spent, so `x = 0` or `5 <= x <= 10`.
The quote obtainable by closing is `k*x`, where
`k = 15995992003 / 16000000000 < 1`. Let `c = 5/k`, which lies strictly
between 5 and 10. Exact terminal wealth is

\[
W(x)=
\begin{cases}
10+\frac{1997}{8000000}x,&x<c,\\
10-\frac{4007997}{16000000000}x,&c\le x\le10.
\end{cases}
\]

Below `c`, the close is too small to execute and marked wealth increases with
`x`. At `c`, the valid closing order incurs its fee and wealth jumps downward.
Every feasible `x<c` has a better feasible choice closer to `c`; every
`x>=c` is worse than `x=5`, and cash is also worse than `x=5`.
`no_optimal_entry` proves there is no maximizing entry. The companion results
check exact entry balances, funding/liquidation, valid final closing and its
actual Oracle wealth. Both prices lie on the grid with step `1/4000`.

This obstruction concerns the specified mandatory capped closing contract.
Allowing a discretionary decision to skip otherwise executable terminal
closing changes that optimization problem. It is not a claim that every
configuration with a minimum order size lacks an optimum.

## Verification and source

- `lean/TradingTheory/ContinuousStochasticPolicy.lean`
- `lean/TradingTheory/ContinuousTradeNonattainment.lean`

Both modules are imported by `lean/TradingTheory.lean`; their public theorems
are listed in `lean/Audit.lean`. Validate with `npm run theory:check`, which
builds the library and rejects unapproved proof axioms.

Verified on 2026-09-12: the complete library build and axiom audit passed,
including all 38 public theorems in these two modules. The audit permits only
`propext`, `Classical.choice` and `Quot.sound`; no proof holes or custom axioms
were introduced. Log: `data/backtests/continuous-stochastic-theory-check.log`
from the repository root.
