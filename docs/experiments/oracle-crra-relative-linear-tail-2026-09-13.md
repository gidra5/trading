# A relative tail envelope from the computed prefix

The linear quote-flow theorem already supplies a finite tail envelope. The
remaining issue is its tightness, rather than the existence of an envelope.
This follow-up connects it explicitly to the computed one-hour action value
without evaluating price laws or policies beyond that hour.

Let H be the number of evaluated transitions, r the number still omitted,
and T = H + r the original terminal horizon. The terminal closing decision is
specified separately, as in the existing scalar model. Fix the current action
a. Write Q_H(a) for its positive optimal expected cash wealth with a valid
complete close at H, and Q_T(a) for its full-horizon value with a valid close
at T. These are gamma=0 action values under the same causal model.

The requested relative contribution is

    (Q_T(a) - Q_H(a)) / Q_H(a).

It is not first-action regret relative to another action. It also includes any
changes in the optimal decisions inside the first hour induced by the longer
horizon. When the H-ending policy can close and then wait legally in cash
through T, its cash value is a feasible full-horizon lower bound and the
contribution is nonnegative. No initial-cash fallback replaces Q_H(a).

## Use the bound at the frontier

At a frontier state s after H transitions, the previously proved envelope is

    Phi_r(s) = max(s.quote, 0) + (r + 1) M.

M bounds aggregate absolute quote movement at every decision, including
arrivals and market orders together. This is the original Oracle model:
fees, funding, liquidation, price constraints and order constraints remain in
the transitions. The extra 1 accounts for the specified closing decision.

Define an upper prefix calculation by replacing the remaining continuation
with Phi_r and performing Bellman backups through only the evaluated prefix.
At each internal decision keep original one-step admissibility, but relax
the requirement that its successor be viable through the omitted tail.
The upper calculation does not require flatness at H. Full feasible policies
can arrive at H with holdings and orders that they close later at T.

`OracleRelativeTailBound.prefixUpper` defines this calculation directly over
the original continuous controls and countable next-price PMFs. It only reads
laws and terms for the evaluated prefix. Its terminal expression reads the
frontier quote, r and M. It does not query future viability, future price laws
or a future optimal policy.

For the fixed first action, let U_(H,r)(a) be the resulting upper score.
`bellman_le_prefix_upper` and `action_le_prefix_upper` prove

    Q_T(a) <= U_(H,r)(a).

Each full-horizon feasible action is present in the relaxed prefix calculation.
At the frontier, Phi_r dominates its continuation by the existing cash-bound
theorem. Expectation and supremum preserve those inequalities on the way back.
`prefix_upper_cash_bound` and `action_prefix_cash_bound` also prove this upper
calculation finite under the uniform M bound.

Consequently,

    (Q_T(a) - Q_H(a)) / Q_H(a)
        <= (U_(H,r)(a) - Q_H(a)) / Q_H(a).

The theorem `relative_continuation_bound` supplies this comparison for any
positive supplied baseline, including the correctly computed Q_H(a).

## Reuse the prefix for different continuation lengths

Phi_r(s) = Phi_0(s) + r M. Probability normalization preserves an additive
constant under expectation. The relaxed Bellman supremum cannot increase
that shift; if a relaxed action set is empty, its ENNReal supremum is zero
and the one-sided inequality still holds. Thus the new formal results
`prefix_upper_linear_tail` and `action_prefix_linear_tail` give

    U_(H,r)(a) <= U_(H,0)(a) + r M.

Both Q_H(a) and U_(H,0)(a) are calculations confined to the first hour.
Define the computed prefix relaxation gap

    C_H(a) = U_(H,0)(a) - Q_H(a).

This gap is nonnegative for a feasible H-horizon action. It is caused by the
upper calculation's relaxation of terminal feasibility and valuation, rather
than by numerical rounding. The resulting relative tail certificate is

    (Q_T(a) - Q_H(a)) / Q_H(a)
        <= [C_H(a) + r M] / Q_H(a).

`computed_prefix_relative_envelope` proves this for the actual continuous
Oracle action value on the left. Neither the proof nor evaluation of the
right side requires solving the tail. For a required relative tolerance
epsilon, the stopping condition is

    C_H(a) + r M <= epsilon * Q_H(a).

The more direct U_(H,r) calculation can be used if it gives a tighter upper
bound; no claim is made that the two upper expressions always coincide.

For reference, the earlier closed-form root bound gives the looser envelope

    [max(q_0,0) + (H+1)M - Q_H(a) + r M] / Q_H(a).

`linear_relative_envelope` proves this version. The prefix upper calculation
cannot exceed that root bound. We cannot simply infer Q_T-Q_H <= r M by
subtracting two upper bounds on Q_T and Q_H: Q_H need not attain its bound,
and their optimizing policies can reach different frontier accounts.

## What linearity establishes

The tail envelope is affine in the omitted length r. Dividing by the computed
Q_H converts it into the requested relative bound. Smallness is then a numeric
inequality to check, not an assumption about how long one hour feels.

An O(r) bound on a tail does not by itself become an O(1/H) bound for all
remaining time. A fixed-length extension may be relatively small compared
with a large Q_H. If r is large or unbounded, the factor r remains. For a fixed
terminal T, extending H decreases r = T-H; the prefix relaxation gap must also
be controlled. Exact evaluation at T uses the actual terminal contract, not
the loose Phi_0 relaxation.

Certified quiet-block compression can reduce computational work, but it does
not permit substituting the number of price changes for r in the quote-cap
theorem. A skipped block's retained trading capacity and account effects must
still be accounted for, or separately proved irrelevant for every competing
policy covered by the upper calculation.

## Verification and implementation scope

The new module proves eleven public results, all imported and axiom-audited.

    node scripts/theory-lean.mjs check

The build and public-theorem audit passed; log:
`data/backtests/oracle-relative-tail-bound-lean.log`.

The prefix-upper specification is noncomputable because its continuous
suprema and exact expectations retain the original model. Its dependency on
price laws is restricted to the prefix; that fact does not certify a finite
menu or sampled approximation as an upper bound. The current NSF numerical
planner has not been modified to produce these certificates, and this report
contains no invented precise one-hour value or new backtest result.
