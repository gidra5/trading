# Convergence of optimal continuation values

Oracle-specific follow-up: `oracle-crra-value-convergence-obstruction-2026-09-12.md`
now gives an exact counterexample using the full original continuous-control
Bellman model. Its optimal terminal wealth divided by initial equity does not
converge. The generic results below remain conditional on their stated hypotheses.

The intended claim is convergence of the optimal value, sup_a q_K(a), to the
full-continuation optimal value sup_a q(a). It does not concern distance between
actions. Supremum notation allows continuous-action optima that are not attained.
All scores must use compatible units and any economic normalization must be
specified. For a relative claim with a positive finite reference value, divide
the absolute value-error bound by that reference value.

## Direct optimal-value theorem

Suppose q_K(a) <= q(a), q_K(a) converges to q(a) for each feasible a, and
V = sup_a q(a) is a finite real number. Then

    V_K = sup_a q_K(a) converges to V.

Monotone convergence q_K(a) increasing to q(a) is one sufficient instance.
No action metric, curvature, positive action gap or uniform convergence rate
is necessary. To prove the result, choose an action a_e with q(a_e) within
epsilon/3 of V. For sufficiently large K, q_K(a_e) is within epsilon/3 of
q(a_e). Since q_K(a_e) <= V_K <= V, the difference V-V_K is less than epsilon.

`optimal_values_converge_from_below` formalizes this proof using explicit
supremum approximation and eventual pointwise approximation hypotheses. It
does not require that either optimum is attained.

If a_K approximately maximizes q_K with error eta, the lower approximation
also gives

    V - q(a_K) <= V - V_K + eta.

Thus the complete-continuation value of that first action approaches optimality.
`first_action_regret_from_lower_values` formalizes the inequality. Here q(a_K)
uses optimal continuation after the first action; this statement alone does
not analyze all errors of repeatedly applying a truncated controller.

## A quantitative version

For general approximations that need not lie below the full scores, if

    sup_a |q_K(a)-q(a)| <= epsilon_K,

then

    |sup_a q_K(a)-sup_a q(a)| <= epsilon_K.

`supremum_error_bound` proves this for finite real least upper bounds without
assuming maximizers exist. If epsilon_K is bounded by C/K, the same rate holds
for the optimal values. The theorem propagates a full-continuation error bound;
it does not infer that bound from neighboring-horizon ratios.

## What remains to establish for Oracle

The earlier bounded-reward results establish adjacent-increment or adjacent
time-average bounds. They do not establish a finite full-continuation limit
or a bound against that limit. In particular, neighboring ratios approaching
one are weaker than convergence of normalized values.

An elementary illustration needs only one action. Let all rewards lie in [1,2].
Starting after one initial reward, append alternating blocks of reward 1 and
reward 2; each new block is nine times as long as all preceding history. At the
end of a reward-1 block, the cumulative average lies in [1,1.1]. At the end of
a reward-2 block, it lies in [1.9,2]. These endpoints occur arbitrarily late,
so the average has no limit. Yet, with cumulative reward S_K >= K,

    0 <= S_(K+1)/S_K - 1 <= 2/K.

Thus even strictly positive bounded increments and vanishing adjacent relative
increments do not establish an average-value limit. This generic mathematical
example is not an Oracle execution model and is not claimed as a Lean theorem
in this change.

The previously formalized alternating-action example is different: its normalized
optimal values do converge to 1. It is compatible with the user's value target
and should not be used as a counterexample to that target.

For finite full additive values, nonnegative rewards can give monotone prefix
convergence; bounded rewards alone do not make an infinite total finite. For
gamma-zero Oracle wealth, compounding, terminal close rules, losses and the chosen
normalization must be included when establishing the hypotheses above. No new
probability mixing restriction is introduced by these optimal-value theorems.

Formal source: `docs/theory/lean/TradingTheory/OptimalValueConvergence.lean`.
The module is imported by the library and its public results are audited.
Verification command: `node scripts/theory-lean.mjs check`.
The complete library build and axiom audit passed for all three new theorems.
