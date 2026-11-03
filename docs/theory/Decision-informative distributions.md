# Decision-informative distributions

For gamma zero, an Oracle policy maximizes expected terminal wealth. A useful
forecast must estimate economically relevant conditional differences accurately;
neither high variance nor frequent trades is a sufficient requirement.

Let `h` contain the observable history and the complete current account state,
including exposure and pending orders. Let `A(h)` be the declared feasible action
menu. Define

\[
Q_P(h,a)=\mathbb E_P[W_T\mid h,\ a\text{ now, optimal observable continuation}],
\qquad
\Delta_P(h,a)=Q_P(h,a)-Q_P(h,a_0),
\]

where `a_0` is a feasible reference action. Every term uses the same Oracle
received-amount fees, signed asset funding, maintenance/liquidation, aggregate
order caps, price grid, pending-order behavior and terminal-close convention.
Holding an existing position is not the same reference as staying in cash from
a flat account. Pending orders can also give a hold action a random payoff.

For a multi-step decision, continuation is optimized subject to observation and
nonanticipativity. It cannot use the deterministic optimum of each complete
future path as a clairvoyant continuation.

## Exact requirements

For a feasible reference action:

\[
a_0\in\arg\max_{a\in A(h)} Q_P(h,a)
\iff
\Delta_P(h,a)\le0\quad\text{for every }a\in A(h).
\]

Therefore **a strictly positive feasible expected gap is necessary and sufficient
to make the reference suboptimal**. It forces every optimizer to choose another
action. Merely changing the selected action in a tie does not demonstrate useful
information.

This differs from requiring a nonconstant policy over contexts. A constant
action can be optimal everywhere whenever the optimal-action sets have a common
member. Different unique optimal actions at two contexts force a policy switch.
A model can consistently recommend a useful long position without meeting that
stronger switching requirement, and can alternate bad decisions without being
useful.

The statements also apply to a restricted menu. They do not establish that the
menu contains the unrestricted optimum.

## The one-step fee and funding threshold

Start with cash `C`, buy using positive quote spend `x` at price `p`, and close
completely after one price/funding transition. Let the entry and exit fees be
`f_in` and `f_out`, and let signed asset funding multiply the position by `k`.
If the account survives and both the entry and full close are valid in every
scenario, Oracle gives

\[
W=C+x\left((1-f_{out})k(1-f_{in})\frac{P'}p-1\right).
\]

For constant positive `k` and finite conditional mean price,

\[
\boxed{
\mathbb E[W\mid h]>C
\iff
\frac{\mathbb E[P'\mid h]}p>
\frac1{(1-f_{out})k(1-f_{in})}.
}
\]

The threshold is independent of the positive spend. Increasing leverage cannot
turn a negative coefficient into a positive one. Under these assumptions,
distribution variance does not enter the decision except through the feasibility
and survival assumptions already required.

For the research configuration `f_in = f_out = 0.0005` and `k = 1`, the required
conditional mean return is approximately **10.0075 basis points**. This is a
property of that declared configuration, not a statement of current exchange
fees. At zero fees and unit funding, the threshold is zero expected price return.

When funding is random, the relevant first moment is `E[k P' | h]`, including its
dependence on price. With liquidation, pending limits or capped residual closes,
the affine formula may not describe actual wealth. Use the complete `Delta`
criterion instead of applying this threshold outside its hypotheses.

## Finite moments are required before comparing expected wealth

The finite-scenario solver assigns finite wealth to finitely many leaves, so its
expectations exist. A continuous forecast needs a separate integrability check.
For each feasible candidate continuation being compared, require

\[
\mathbb E_P[|W_T|\mid h]<\infty.
\]

For the preceding long specialization, finite `E[k P' | h]` is sufficient. A
finite mean or variance of the **log return** does not imply a finite mean price:
the needed moment is `E[exp(R) | h]`. Nor do finite one-step marginal means alone
justify every multi-step product or adaptive strategy; integrability must cover
the continuation payoff being optimized.

A distribution with polynomial tails in log return can have infinite expected
price. A finite scenario approximation of it still returns a finite number,
but that number cannot be claimed to preserve its gamma-zero objective merely
by preserving the log-return mean.

One suitable construction is a bounded density `f_U` on `(0,1)` followed by
`R = s Phi^{-1}(U)`, with `s > 0` and `Phi` the standard normal CDF. If
`f_U <= M`, change of variables gives

\[
f_R(r)\le \frac{M}{s}\phi(r/s),\qquad
\mathbb E[e^{\alpha R}]\le M e^{\alpha^2 s^2/2}<\infty.
\]

A finite piecewise-linear knot density is bounded. Mixing it with an exact-zero
atom preserves finite exponential moments. This analytic sufficient condition
does not by itself prove that a trained forecast is calibrated.

For the affine endpoint-only long task, discretization must preserve bin
probability and the conditional mean of the **price ratio**, not the mean log
return. Full leveraged policies with limit orders also depend on intermediate
crossings and maintenance, so an endpoint distribution alone is insufficient.

An endpoint-only comparison is valid for a restricted committed hold: after
entry, quote and asset balances must both be nonnegative and not both zero;
funding is one; there are no pending orders or intermediate trades; every price
is positive; fees lie in `[0,1)`; and maintenance bounds contain `[0,1]`.
`OracleEndpointHold.lean` proves that effective equity remains positive and
effective exposure stays in `[0,1]` at every intermediate mark. All liquidation
checks are therefore inactive and the final account equals a direct endpoint
mark. Any identical terminal function consequently gives the same result,
including a capped close retaining residual inventory. This last fact preserves
endpoint sufficiency, but does not make a residual-close payoff affine or prove
that this commitment is optimal among policies allowed to trade inside it.

## Accuracy margins and training

Suppose the true and estimated action values satisfy

\[
|\widehat Q(h,a)-Q_P(h,a)|\le\epsilon
\]

for the action and reference. Then a predicted gap greater than `2 epsilon`
certifies a true positive gap. If the difference itself has a certified error
at most `epsilon`, the sufficient margin is `epsilon`. A true best action
separated from each competitor by more than `2 epsilon` remains the best under
the value approximation.

These are conditional implications. A backtest does not automatically provide
uniform error bounds, especially with dependent observations or unbounded
payoffs. Validation should report out-of-sample calibration and errors of the
relevant price-ratio moments and action gaps, not substitute trading frequency
for those measurements.

Training can improve the estimate through a proper distribution score,
well-defined tails, appropriate horizons, and decision-aware validation.
It cannot manufacture a true positive conditional gap when the data-generating
distribution has none. In particular, forcing a minimum predicted edge or a
minimum fraction of trade decisions may produce fabricated confidence.

For a binary event with true probability `p` and forecast `q`, the population
Brier score obeys

\[
L(p,q)-L(p,p)=(q-p)^2.
\]

If a trade requires `q > c` but `p <= c`, forcing the forecast past `c` strictly
worsens this proper score. A useful training acceptance criterion is therefore
held-out evidence of accurately estimated positive gaps where they exist,
together with honest no-trade predictions elsewhere. Do not impose positive
gaps on every context or use test profitability to select the model.

## Formal verification scope

`lean/TradingTheory/PredictionDecisionConditions.lean` proves thirteen results:
the exact reference-optimality criterion, positive-gap selection, error-margin
guarantees, common/strict optimizer switching conditions, expected wealth-gap
identity, Oracle full-close expectation and threshold, and the proper-score
counterexample. The continuous integrability discussion and Gaussian domination
argument above are mathematical derivations, not newly machine-checked integral
theorems. The finite expectation and Oracle threshold theorems are checked by
Lean and included in the repository axiom audit.
