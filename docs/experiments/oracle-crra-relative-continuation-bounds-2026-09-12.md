# Relative continuation influence and certified stopping

The relevant target is the influence of omitted continuation on the ranking of
actions now. Total future wealth can grow without bound while this influence
shrinks. The two questions are distinct. The current objective has risk aversion
gamma zero and maximizes expected terminal wealth; it has no time discount.

## A bound that addresses the relative effect

Fix the current full account and market information, the same admissible first
actions, and a terminal horizon T. Let Q_T(a) be the optimal expected terminal
wealth conditional on first action a. Future controls remain nonanticipative.
Let q_K(a) be the score obtained using K steps and a specified terminal
approximation. Choose a common positive wealth scale S_T independently of the
action. The scale must have an economic interpretation fixed before certification;
making it arbitrarily large would make a relative accuracy claim meaningless.

For example, if comparing K- and T-horizon expected wealth, one may use
q_K(a) = Q_K(a)/S_K and study

    d_K(a) = Q_T(a)/S_T - q_K(a).

The influence relevant to current action selection is the oscillation

    E_K = sup_{a,b feasible} |d_K(a) - d_K(b)|.

This removes any common additive continuation contribution. The normalization
also permits removal of a common growth factor. A growth factor that depends on
the future state cannot simply be pulled outside an expectation: wealth and that
factor may be correlated.

If a_K maximizes q_K and E_K <= epsilon, then, for every feasible a,

    (Q_T(a) - Q_T(a_K))/S_T
      = q_K(a) - q_K(a_K) + d_K(a) - d_K(a_K)
      <= epsilon.

Thus the selected first action has normalized regret at most epsilon. No bound
on the common level of Q_T is necessary. Approximate maximization of q_K with
error eta adds eta to this guarantee. This certifies the first decision; repeating
such decisions requires a separate accumulated policy-error bound.

The Lean theorem `relative_continuation_bound` proves the equivalent affine
form Q_T(a) = scale*q_K(a) + offset + residual(a), assuming pairwise residual
differences are bounded by scale*epsilon. `centered_continuation_bound` proves
the absolute-regret version. These are conditional theorems, not assumptions
that the residual actually decays in Oracle.

## When decay would follow

If a valid bound on this normalized action-dependent influence satisfies

    E_(K+1) <= rho * E_K,  0 <= rho < 1,  E_0 <= D,

then E_K <= rho^K D. `contraction_bound` formally proves the finite-step bound.
For 0 < rho < 1 and 0 < epsilon < D, the sufficient depth is

    K >= ceil(log(epsilon/D) / log(rho)).

A one-hour statement at one-second steps would require rho^3600 D <= epsilon,
or another certified bound meeting epsilon at K = 3600. There is currently no
verified rho, D, or one-hour residual bound for the NSF/Oracle planner.

Mixing of prices alone is insufficient: leverage, inventory, remaining order
capacity, pending orders and liquidation can retain the consequences of a first
action. A contraction must control the relevant normalized Bellman differences
on the full account and predictive state. Ordinary undiscounted Bellman
nonexpansion does not imply strict contraction.

Even relative decay has no universal elapsed-time guarantee under the allowed
family of account constraints. For an illustrative deterministic Oracle instance,
take initial cash W, price 1, funding factor 1, and fee 0 < f < 1/2. Permit a
market purchase spending W now; set subsequent aggregate trade caps to zero,
including the close, so residual inventory is retained and marked. Choose a
maintenance band containing exposure 1. The price remains 1 for an arbitrarily
long duration and then becomes 2. Prices lie on a compatible tick lattice.
Cash is worth W; the purchase is worth (1-f)W before the jump and 2(1-f)W after
it. The normalized pairwise advantage changes by 1-f, regardless of how many
flat ticks precede the jump. There is no liquidation in these two trajectories.
Lower-fee limit orders do not enable a buy on this flat-then-up path under the
strict crossing rule, and later zero trade capacity forbids fills in any case.
This example uses time-varying trade capacity; it is not a measurement or a
claim that the current fixed-cap fitted NSF exhibits the same behavior.

The example explains why a decay premise needs a proof or a distribution-specific
certificate. It does not establish that useful practical decay is absent.

## A computable stopping certificate

If full-horizon scores have certified intervals L_K(a) <= Q_T(a) <= U_K(a),
choose a_K maximizing L_K. It is safe to stop when

    sup_a U_K(a) - L_K(a_K) <= epsilon * S_T.

`interval_stop` proves this implication without requiring a maximizer of the
upper bounds. Direct intervals for pairwise advantages can be tighter, since
they cancel common continuation uncertainty.

Bounds must cover every admissible first action and the original modeled
future law. Finite action menus, compressed scenarios, skipped execution times,
numerical error and Monte Carlo error all require inclusion in the certificate.
A statistical certificate must specify its coverage probability. Agreement
between two lookahead depths, or between two sampled graphs, does not bound an
unseen continuation. There is no automatic stopping mode added to the backtester
by this change because these bounds are not yet supplied by the NSF planner.

## Bounded rewards: adjacent increments versus the whole remainder

An additional formalization in `TradingTheory/BoundedRewardIncrements.lean`
addresses the proposed linear-accumulation argument directly. These are generic
mathematical support lemmas, not a replacement trading execution model.

Normalized nonnegative probability weights preserve any uniform reward interval
[lo, hi]. No mixing restriction is needed for that fact. If expected additive
rewards satisfy 0 <= r_t <= R, their cumulative value Q_K lies in [0, K*R].
This is a linear upper bound, not a positive linear growth guarantee.

If additionally Q_K >= c*K for some c > 0, and the next increment is at most R,
then

    (Q_(K+1) - Q_K)/Q_K <= R/(c*K).

This is a decreasing upper envelope; the actual ratios need not decrease at
each step. A positive lower bound r_t >= c is sufficient for Q_K >= c*K.
Probability normalization and an upper reward bound do not imply that lower
bound. An all-cash gamma-zero account, for example, can keep constant wealth.

There is also a bound without positive linear growth: when 0 <= Q_K <= K*R and
the next increment lies in [0,R],

    |Q_(K+1)/(K+1) - Q_K/K| <= R/(K+1).

This bounds neighboring time averages. Dividing by K does not change the
preferred action within one fixed horizon, but does not make different
horizons equivalent.

Neither adjacent bound certifies an arbitrarily long remainder. With N additional
steps the elementary bound is

    (Q_(K+N) - Q_K)/Q_K <= N*R/(c*K).

The N factor cannot be omitted. Bounds of order 1/K for consecutive normalized
scores are not, by themselves, a summable tail bound or a Cauchy certificate.

The formal arithmetic example uses two cumulative reward schedules. Schedule A
earns 1 for its first K steps, then 1/4 per step. Schedule B earns 1/2 throughout.
All rewards lie in [1/4,1], so even positive linear growth is present. At horizon
K, A has value K and B has K/2. At horizon 4K, A has 7K/4 and B has 2K. Choosing
the K-horizon winner is therefore 1/8 below the 4K-horizon winner, independently
of how large K is. `horizon_reversal` proves these comparisons and the exact
ratio; `example_increments_bounded` verifies every increment. This refutes the
inference from bounded positive increments to a small distant action-ranking
effect. It is not presented as an Oracle execution counterexample.

For an additive bounded-reward problem with compatible policy restriction and
extension across horizons and zero terminal payoff, the optimal action scores
satisfy Q_K(a) <= Q_T(a) <= Q_K(a)+(T-K)R. This gives a legitimate, often loose,
first-action regret bound (T-K)R for the K-horizon optimizer. A different terminal
contract, including Oracle closing, needs its own boundary-error allowance.

There is a separate Oracle applicability obligation: bounded percentage moves
bound increments relative to current wealth and exposure, not by a constant
dollar amount. Multiplicative wealth growth need not accumulate linearly. The
new generic lemmas do not silently change gamma-zero terminal wealth to an
additive return or average-reward objective.

## Auxiliary Oracle bounds and formalization scope

The same Lean module proves that a received-amount market trade with nonnegative
fee below one cannot increase raw equity at its current mark, and proves the
concrete scalar funding/mark identity. Portfolio algebra gives

    W_mark = W_posttrade + sum_i v_i r_i
           <= W_posttrade * (1 + Lambda * m)

when sum_i |v_i| <= Lambda * W_posttrade and every funded return
r_i = funding_i * P'_i/P_i - 1 satisfies |r_i| <= m. The gross leverage bound
must be derived from the actual margin constraints, including coasting; it is
not automatically the entry leverage cap. Liquidation to zero preserves any
nonnegative upper bound even when pre-liquidation raw equity is negative.

Crossed limit fills have the same current-mark nonincrease property algebraically:
a buy spending N at L >= P changes marked equity by -N+(1-f)NP/L <= 0;
a sell receiving N at L <= P changes it by N-NP/((1-f)L) <= 0.
Reservation changes do not change balances. The new module does not prove the
full order-event/admissibility-to-portfolio-cap bridge.

`expected_terminal_upper` proves a finite-outcome expectation recursion under a
uniform conditional one-step expected-wealth bound. Uniform conditional return
moment envelopes, when available, give multiplicative wealth envelopes without
independence. Unconditional average moments cannot replace conditional bounds.
This absolute envelope can be useful for interval construction but does not
establish relative decay.

The full NSF law has unbounded latent-conditioned scales; uniform moment and
relative-contraction bounds have not been established. Finite sampled maxima
and sampled conditional moments are not certificates for that law. Separate
discounted-tail theorems are included for comparison; applying them would
require a discounted bounded-reward objective, which is not our current one.

Formal source: `docs/theory/lean/TradingTheory/ContinuationValueBounds.lean`.
The module is imported by the library and its main theorems are listed in
`Audit.lean`. Verification command: `node scripts/theory-lean.mjs check`.
The complete library build and axiom audit passed. No decay hypothesis was added
as an axiom, and no execution or forecasting model was changed.
The subsequent `BoundedRewardIncrements` module also passed the complete build
and axiom audit, including the increment, remainder and horizon-reversal results.
