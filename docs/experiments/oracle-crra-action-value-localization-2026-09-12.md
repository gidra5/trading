# Action localization from continuation bounds

Subsequent clarification: the user's target is convergence of optimal values,
not geometric distance between optimizing actions. The alternating-action
example below has normalized optimal value 1+1/(4*(K+1)), which does converge
to 1. It refutes action-distance localization, not optimal-value convergence.
See `oracle-crra-optimal-value-convergence-2026-09-12.md` for the intended claim.

This extends the relative-continuation analysis. Shrinking relative differences
between neighboring value functions must be distinguished from three stronger
claims: a bound against the complete continuation, a bound on value regret, and
a bound on distance between optimal actions. Each implication has its own
hypotheses. The new Lean module contains generic mathematical support lemmas;
it does not introduce an alternative trading execution model.

## A valid adjacent-horizon value neighborhood

Let Q_K(a) and Q_(K+1)(a) concern the same feasible first actions, and suppose

    0 <= Q_(K+1)(a) - Q_K(a) <= R

for every action. If a_K maximizes Q_K, then

    Q_(K+1)(a) - Q_(K+1)(a_K) <= R

for every competitor a. Dividing by K+1 gives an R/(K+1) average-value regret
bound. `adjacent_horizon_regret` proves this result with only the pointwise
inequalities needed for the comparison. The bound is against the next horizon;
it is not a certificate against all omitted steps. In Oracle, horizon-dependent
terminal close costs and the absence of a uniform dollar reward bound require
separate treatment before these assumptions can be used.

## Uniform ratios approaching one do not localize actions

For the continuous action interval [-1,1], consider the arithmetic score family

    Q_K(a) = K + 1 + (-1)^K * a/4,  K >= 0.

K indexes a prefix containing K+1 rewards. The first reward is 1+a/4, between
3/4 and 5/4. Each subsequent increment is

    Q_(K+1)(a) - Q_K(a) = 1 - (-1)^K * a/2,

between 1/2 and 3/2. Thus this is one consistent family of cumulative, bounded,
strictly positive rewards. The Lean proofs establish, uniformly in a,

    |Q_(K+1)(a)/Q_K(a) - 1| <= 2/(K+1).

Nevertheless, the unique maximizing action is +1 for every even K and -1 for
every odd K. The distance between neighboring optimal actions is always 2.
The normalized scores Q_K/(K+1) converge uniformly to the constant function 1;
their small action-dependent term still selects opposite endpoints.

`oscillating_increment_bounds`, `oscillating_ratio_bound`, and
`oscillating_unique_maximizers` formalize these claims. This is a counterexample
to the proposed mathematical implication, not an Oracle strategy or a claim
that the fitted NSF produces these scores. It shows why probability
normalization, bounded positive rewards and even uniform relative convergence
are insufficient premises for action-space localization.

## Conditions that do localize actions

Let F be the complete-horizon action score after a specified common economic
normalization, and let F_K be its approximation. Suppose a* maximizes F, a_K
maximizes F_K, and the full-continuation approximation error satisfies

    sup_a |F_K(a)-F(a)| <= epsilon_K.

Then F(a*)-F(a_K) <= 2*epsilon_K (`score_error_regret`). Bounds at just the two
compared actions suffice. The preceding adjacent-ratio theorem does not itself
provide this full-continuation error bound.

If F has a quadratic margin around its optimizer,

    F(a*)-F(a) >= (mu/2) * distance(a,a*)^2,  mu > 0,

then

    distance(a_K,a*)^2 <= 4*epsilon_K/mu.

This is `quadratic_action_localization`. Distance is therefore at most
2*sqrt(epsilon_K/mu). An O(1/K) value error yields an O(1/sqrt(K)) distance
bound under this hypothesis. Strong concavity can supply a quadratic margin
on an appropriate convex feasible domain, but it has not been established for
the full Oracle action space.

A sharper linear margin F(a*)-F(a) >= mu*distance(a,a*) gives
distance(a_K,a*) <= 2*epsilon_K/mu (`sharp_action_localization`), which would
yield the proposed O(1/K) distance bound if epsilon_K is O(1/K). This margin
is an additional property of the objective, not a consequence of normalization.
For portfolio actions, distance requires a chosen norm. Limit-order placement,
cancellation and order count also have discrete components, so an exposure-only
metric cannot certify that the complete decision is close.

## A candidate set without curvature assumptions

If L_K(a) <= F(a) <= U_K(a) for every admissible action, define

    C_K = {a feasible : U_K(a) >= sup_b L_K(b)}.

Every true maximizer belongs to C_K (`optimizer_in_candidate_set`). If the
certified candidate set fits in a small ball, that is an action-localization
certificate. Without a margin, it may remain wide even when every interval is
narrow. For value accuracy alone, the interval stopping theorem from the
previous module can suffice without geometric localization.

These are conditional certificates. The current NSF planner does not yet
provide full-continuation intervals, continuous-action coverage bounds, or
verified action margins. No stopping mode, model, objective or bot execution
rule was changed by this formalization.

Source: `docs/theory/lean/TradingTheory/ActionValueStability.lean`.
The module is imported into the library and public theorems are included in
the axiom audit. Verification command: `node scripts/theory-lean.mjs check`.
The complete library build and axiom audit passed, including the uniform-ratio
counterexample and both conditional action-distance bounds.
