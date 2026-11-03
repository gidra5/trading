Predicting direction and magnitude separately is a valid representation of a
price forecast. Its exactness depends on retaining the sign-conditional laws.
The Lean results are in [SignSizeForecast.lean](lean/TradingTheory/SignSizeForecast.lean).

Let the simple price return over the chosen horizon be
\(R=p_{t+H}/p_t-1\), its sign \(S\in\{-1,0,1\}\), and magnitude
\(M=|R|\). All probabilities below are conditional on the information available
at the current decision. Write
\[
p_+=\Pr(R>0),\qquad p_-=\Pr(R<0),\qquad p_0=\Pr(R=0),
\]
\[
m_+=\mathbb E[M\mid R>0],\qquad m_-=\mathbb E[M\mid R<0].
\]
A conditional mean for an empty sign group can be assigned arbitrarily: its
probability is zero. The formalization handles these empty groups explicitly.
Then
\[
\mathbb E[R]=p_+m_+-p_-m_-,\qquad
\mathbb E[|R|]=p_+m_++p_-m_-.
\]
These are `expected_signed_return` and `expected_absolute_return`. With
\(p=\Pr(R>0\mid R\ne0)\), the first formula becomes
\((1-p_0)[p m_+-(1-p)m_-]\), which is the forecast implementation's convention.
There is no assumption that direction and size are independent.

Average movement is an opportunity measure when direction is known. For an
upward return with probability one, expected return equals expected magnitude
(`known_upward_sign`). With uncertainty, the two differ. A return of either
\(+0.3\%\) or \(-0.3\%\) has expected absolute size \(0.3\%\) under both
75%-up and 25%-up laws. Their signed expected returns are respectively
\(+0.15\%\) and \(-0.15\%\) (`same_average_size_opposite_means`). Even a
99% upward probability can have negative mean when the rare loss is sufficiently
large (`high_sign_probability_can_have_negative_mean`). In general
\(\mathbb E[SM]\ne\mathbb E[S]\mathbb E[M]\); unequal sign-conditional
magnitudes already give a counterexample with equally likely signs.

For any fixed admissible policy and any finite full-scenario law, regrouping
scenarios by return sign preserves its expected payoff:
\[
\mathbb E[F_a(\omega)]
=\sum_{s\in\{-,0,+\}}p_s\,
  \mathbb E[F_a(\omega)\mid S=s].
\]
Here each conditional law retains the whole original scenario, including its
intermediate prices and observation schedule. Therefore every action score and
the set of optimal actions are unchanged (`regrouped_action_payoff`,
`regrouped_action_optimality`). The existing CRRA problem, including feasibility
and bankruptcy utility, is preserved when the reconstructed law equals the
original law (`full_conditional_law_preserves_crra_optimality`). The future sign
is a latent forecast outcome; this representation does not reveal it to the
current policy.

Replacing each conditional law by its mean instead gives a three-point law on
\(\{-m_-,0,m_+\}\). This preserves the supplied first signed and absolute
moments. It does not generally preserve nonlinear utilities, intermediate limit
fills, liquidation, capped terminal close values, or continuation values. A
policy can be optimal for that approximate law without being optimal for the
original law. Predicting both conditional size means alone is therefore not an
exact reduction of the unrestricted Oracle problem.

There is an exact economic specialization for the [Oracle execution model](Oracle.md).
Start in cash, spend \(x>0\) in a valid buy, hold through unit funding while
remaining solvent, and execute a valid complete sell at the horizon. The fees
are Oracle's received-amount fees, at the same rate \(f\) on each side. The
actual account path must establish this complete-close contract; order limits
or remaining inventory cannot be ignored to obtain it. Its terminal wealth is
\[
W'=W+x\big[(1-f)^2(1+R)-1\big].
\]
Consequently, for \(\gamma=0\), positive spending improves expected terminal
wealth exactly when
\[
p_+m_+-p_-m_->\frac{1}{(1-f)^2}-1.
\]
This is `oracle_long_profitable_iff_signed_size`, derived from the existing
Oracle funding and round-trip theorem. At \(f=0.0005\), the threshold is
10.007505 basis points. The equal-average-size examples above lie on opposite
sides of that fee-adjusted entry decision (`same_average_size_opposite_fee_gaps`).
For this affine full-close payoff the three-point law's preserved mean is
sufficient. Other Oracle states and execution rules need their actual payoff
law evaluated.

The formal module adds 14 public audited results. Verification:
`npm run theory:check`.
