# Deriving and Fitting the 17-Parameter Conditional Regret Model

This document accompanies [Conditional Four-Segment Exponential Model with Log-Slope-Change Parameters](./conditional-four-segment-compact-support.md). It gives a direct procedure for deriving or estimating every trained scalar from the underlying regret surface, while allowing the regret to contain kinks.

## 1. Oracle value, regret, and probability

Let \(x\) denote current exposure and \(a\) the candidate target exposure. A generic oracle value for the candidate transition is

\[
J_n^{H,T}(x,a)
=-C_n(x,a)
+\sum_{h=1}^{H}U_{n+h}(a)
+V_{n+H,T}(a),
\]

where:

- \(C_n(x,a)\) is the immediate transition cost;
- \(U_{n+h}(a)\) is the log-return at exposure \(a\), minus maintenance, funding, and other holding costs;
- \(V_{n+H,T}(a)\) is the optimal continuation value for the following \(T\) steps, starting from exposure \(a\).

The oracle regret is

\[
R_n^{H,T}(x,a)
=\max_u J_n^{H,T}(x,u)-J_n^{H,T}(x,a).
\]

Assume the conditional distribution is constructed as

\[
p(a\mid x)
\propto
\exp\!\left(-\frac{R(x,a)}{\tau}\right),
\]

where \(\tau>0\) is the regret temperature. If the construction uses \(e^{-R}\), set \(\tau=1\).

Define the score

\[
Y(x,a)=-\frac{R(x,a)}{\tau}.
\]

Because the best attainable oracle value is constant with respect to the candidate \(a\),

\[
Y(x,a)
=\frac{J(x,a)}{\tau}+K(x)
\]

for some slice-dependent constant \(K(x)\). Conditional normalization also changes only that constant. All shape parameters can therefore be fitted from \(Y\) without knowing \(K(x)\).

## 2. The 17 trained parameters

Scale current exposure to

\[
\xi(x)=\frac{2x-X_--X_+}{X_+-X_-}\in[-1,1].
\]

The parameter functions are

\[
\begin{aligned}
b(\xi)&=b_0+b_1\xi,\\
\beta_{c_1}(\xi)&=\beta_{c_1,0}+\beta_{c_1,1}\xi,\\
\beta_x(\xi)&=\beta_{x,0}+\beta_{x,1}\xi,\\
\beta_{c_2}(\xi)&=\beta_{c_2,0}+\beta_{c_2,1}\xi.
\end{aligned}
\]

The 17 trained scalars are:

| Group | Scalars | Count |
|---|---|---:|
| Fixed breakpoints | \(c_1,c_2\) | 2 |
| Baseline slope | \(b_0,b_1\) | 2 |
| First fixed jump | \(\beta_{c_1,0},\beta_{c_1,1}\) | 2 |
| Moving jump | \(\beta_{x,0},\beta_{x,1}\) | 2 |
| Second fixed jump | \(\beta_{c_2,0},\beta_{c_2,1}\) | 2 |
| Transition sharpness | \(\kappa_{c_1},\kappa_x,\kappa_{c_2}\) | 3 |
| Support taper widths | \(w_L,w_R\) | 2 |
| Support taper sharpness | \(\rho_L,\rho_R\) | 2 |
| **Total** |  | **17** |

The first eight coefficients are linear once the remaining nine shape parameters are fixed.

## 3. Complete score model

Define

\[
\operatorname{SP}_{\kappa}(z)
=\frac{1}{\kappa}\log(1+e^{\kappa z}).
\]

Let \([A_-,A_+]\) be the full latent support. The fitted score is

\[
\begin{aligned}
Y_\theta(x,a)={}&\alpha(x)
+b(\xi)(a-A_-)\\
&+\beta_{c_1}(\xi)
  \operatorname{SP}_{\kappa_{c_1}}(a-c_1)\\
&+\beta_x(\xi)
  \operatorname{SP}_{\kappa_x}(a-x)\\
&+\beta_{c_2}(\xi)
  \operatorname{SP}_{\kappa_{c_2}}(a-c_2)\\
&+\log G_L(a;w_L,\rho_L)
+\log G_R(a;w_R,\rho_R).
\end{aligned}
\]

The nuisance offset \(\alpha(x)\) is not one of the 17 model parameters. It absorbs the arbitrary score offset for each conditional slice and disappears after normalization.

The visible interval, such as \([-100,100]\), is a hard observational truncation. It is not used in \(G_L\) or \(G_R\). The gates belong only to the full latent interval, such as \([-250,250]\).

## 4. Regret need not be differentiable

The model should normally be fitted to the regret values \(Y=-R/\tau\), not to numerical derivatives. Full differentiability is unnecessary.

For a probability distribution to exist, it is enough that \(R\) be measurable and that

\[
\int\exp\!\left(-\frac{R(x,a)}{\tau}\right)da<\infty.
\]

For slope-jump interpretation, it is convenient for regret to be continuous and piecewise differentiable with finite one-sided derivatives. At a breakpoint \(d\), define

\[
R'_-(x,d)=\lim_{a\uparrow d}R_a(x,a),
\qquad
R'_+(x,d)=\lim_{a\downarrow d}R_a(x,a).
\]

Then

\[
\boxed{
\beta_d(x)
=-\frac{R'_+(x,d)-R'_-(x,d)}{\tau}.
}
\]

Thus a slope kink is allowed and has a direct \(\beta\) interpretation. A jump discontinuity in regret is different: a continuous softplus cannot represent it exactly and an explicit step, point mass, or smoothing rule is needed.

## 5. Direct fitting objective

Suppose regret is available on grid points \((x_j,a_i)\). Set

\[
Y_{ji}=-\frac{R(x_j,a_i)}{\tau}.
\]

Fit the model by minimizing

\[
\boxed{
\mathcal L(\theta,\alpha)
=\sum_{j,i}w_{ji}
\left[Y_{ji}-Y_\theta(x_j,a_i)\right]^2
+\mathcal R(\theta).
}
\]

Possible weights include:

- equal weights for a global approximation;
- larger weights near low-regret actions if the peak is most important;
- quadrature weights if the exposure grid is nonuniform;
- robust-loss weights when the oracle table contains numerical artifacts.

For fixed structural parameters, the best slice offset is

\[
\widehat\alpha_j
=\frac{\sum_iw_{ji}[Y_{ji}-F_\theta(x_j,a_i)]}
{\sum_iw_{ji}},
\]

where \(F_\theta\) is the modeled score without \(\alpha\). Thus the offsets can be eliminated analytically rather than trained as persistent parameters.

## 6. The inner linear fit for eight coefficients

Fix

\[
c_1,c_2,
\kappa_{c_1},\kappa_x,\kappa_{c_2},
w_L,w_R,\rho_L,\rho_R.
\]

For every sample \((x_j,a_i)\), define

\[
\begin{aligned}
z_0&=a_i-A_-,\\
z_1&=\xi_j(a_i-A_-),\\
z_2&=\operatorname{SP}_{\kappa_{c_1}}(a_i-c_1),\\
z_3&=\xi_jz_2,\\
z_4&=\operatorname{SP}_{\kappa_x}(a_i-x_j),\\
z_5&=\xi_jz_4,\\
z_6&=\operatorname{SP}_{\kappa_{c_2}}(a_i-c_2),\\
z_7&=\xi_jz_6.
\end{aligned}
\]

The coefficient vector is

\[
\theta_{\mathrm{lin}}=
\begin{bmatrix}
b_0&b_1&
\beta_{c_1,0}&\beta_{c_1,1}&
\beta_{x,0}&\beta_{x,1}&
\beta_{c_2,0}&\beta_{c_2,1}
\end{bmatrix}^{\mathsf T}.
\]

After subtracting the log gates and including one nuisance-intercept column per \(x_j\), this is a weighted linear regression. In matrix form,

\[
\boxed{
\widehat\theta_{\mathrm{lin}}
=(Z^{\mathsf T}WZ+\lambda I)^{-1}Z^{\mathsf T}Wy.
}
\]

The ridge term \(\lambda I\) is optional but useful when transitions overlap. The nuisance intercept columns should not be penalized.

This simultaneous solve is preferable to measuring jumps separately because it remains valid when \(x\) crosses or approaches \(c_1,c_2\).

## 7. Derivative form for interpretation and diagnostics

Where the regret derivative exists, define

\[
m(x,a)
=-\frac{R_a(x,a)}{\tau}.
\]

The fitted marginal score is

\[
\begin{aligned}
m_\theta(x,a)={}&b_0+b_1\xi\\
&+(\beta_{c_1,0}+\beta_{c_1,1}\xi)
  \sigma\!\left(\kappa_{c_1}(a-c_1)\right)\\
&+(\beta_{x,0}+\beta_{x,1}\xi)
  \sigma\!\left(\kappa_x(a-x)\right)\\
&+(\beta_{c_2,0}+\beta_{c_2,1}\xi)
  \sigma\!\left(\kappa_{c_2}(a-c_2)\right)\\
&+\partial_a\log G_L(a)
+\partial_a\log G_R(a).
\end{aligned}
\]

For fixed nonlinear parameters, this derivative model is also linear in the same eight coefficients. Its feature vector is

\[
\begin{bmatrix}
1&\xi&
\sigma_{c_1}&\xi\sigma_{c_1}&
\sigma_x&\xi\sigma_x&
\sigma_{c_2}&\xi\sigma_{c_2}
\end{bmatrix}.
\]

Use this form for checking the fitted interpretation. Use the score-value fit in Section 5 as the primary estimator because numerical differentiation amplifies noise.

## 8. Parameters 1–2: fixed breakpoints \(c_1,c_2\)

The fixed breakpoints are persistent exposure locations where the marginal score changes with \(a\), independent of the moving location \(a=x\).

If a smoothed curvature diagnostic is available, define

\[
q(x,a)
=\partial_am(x,a)
=-\frac{R_{aa}(x,a)}{\tau}.
\]

Remove or mask the moving band around \(a=x\), then form

\[
E_{\mathrm{fixed}}(a)
=\sum_jw_jq_{\mathrm{res}}(x_j,a)^2.
\]

The two persistent peaks of \(E_{\mathrm{fixed}}\) give initial values for \(c_1,c_2\). Because curvature estimates are noisy, the final values should be obtained by minimizing the direct score loss, subject to

\[
A_-<c_1<c_2<A_+.
\]

If the trading specification contains known exposure thresholds, such as maintenance or funding tiers, those values can initialize or fix \(c_1,c_2\) directly.

The breakpoints are not necessarily fundamental market parameters. If the underlying regret is smoothly curved, they are effective knot locations chosen by the four-segment approximation.

## 9. Parameters 3–4: baseline \(b_0,b_1\)

The baseline marginal score is

\[
b(\xi)=b_0+b_1\xi.
\]

In a region to the left of every transition and outside the latent support taper,

\[
m(x,a)\approx b_0+b_1\xi.
\]

If such a region is well observed, estimate a baseline value \(\widehat b_j\) for each \(x_j\), then use

\[
b_1
=\frac{\sum_jw_j(\xi_j-\bar\xi)(\widehat b_j-\bar b)}
{\sum_jw_j(\xi_j-\bar\xi)^2},
\qquad
b_0=\bar b-b_1\bar\xi.
\]

In general, take \(b_0,b_1\) from the simultaneous linear solve in Section 6.

From the oracle formulation,

\[
\boxed{
\tau b(\xi)
=-C_a(x,a)
+\sum_{h=1}^{H}U_{n+h,a}(a)
+V_{n+H,T,a}(a)
}
\]

in the baseline regime, excluding the support-gate derivative. Thus \(b\) combines marginal return, maintenance, immediate cost, and continuation value. It is not normally derivable from fees alone.

## 10. Parameters 5–6: jump at \(c_1\)

The slope change at \(c_1\) is

\[
j_{c_1}(x)
=m(x,c_1^+)-m(x,c_1^-).
\]

For a smooth fitted transition, the left and right values mean the two asymptotic levels outside the transition layer, not infinitesimally separated evaluations at its center.

The model assumes

\[
j_{c_1}(x)
=\beta_{c_1,0}+\beta_{c_1,1}\xi.
\]

If slice-wise jump estimates are used,

\[
\beta_{c_1,1}
=\frac{\operatorname{Cov}_w(\xi,j_{c_1})}
{\operatorname{Var}_w(\xi)},
\qquad
\beta_{c_1,0}
=\bar j_{c_1}-\beta_{c_1,1}\bar\xi.
\]

The structural decomposition is

\[
\boxed{
\tau\beta_{c_1}(x)
=-\Delta_{c_1}C_a
+\sum_{h=1}^{H}\Delta_{c_1}U_{n+h,a}
+\Delta_{c_1}V_{n+H,T,a}.
}
\]

For a per-step maintenance cost \(M(a)\),

\[
\beta_{c_1}^{\mathrm{maintenance}}
=-\frac{H}{\tau}
\left[M'(c_1^+)-M'(c_1^-)\right].
\]

## 11. Parameters 7–8: moving jump at \(a=x\)

The moving jump is

\[
j_x(x)=m(x,x^+)-m(x,x^-)
=\beta_{x,0}+\beta_{x,1}\xi.
\]

The two linear coefficients can be obtained from slice-wise jumps by

\[
\beta_{x,1}
=\frac{\operatorname{Cov}_w(\xi,j_x)}
{\operatorname{Var}_w(\xi)},
\qquad
\beta_{x,0}
=\bar j_x-\beta_{x,1}\bar\xi,
\]

or, preferably, from the joint linear solve.

For asymmetric proportional transition cost

\[
C(x,a)
=q_{\mathrm{buy}}(a-x)_+
+q_{\mathrm{sell}}(x-a)_+,
\]

the direct fee contribution is

\[
\boxed{
\beta_x^{\mathrm{fee}}
=-\frac{q_{\mathrm{buy}}+q_{\mathrm{sell}}}{\tau}.
}
\]

For a symmetric rate \(q\),

\[
\beta_x^{\mathrm{fee}}=-\frac{2q}{\tau}.
\]

A fixed fee for any nonzero trade produces a jump in regret rather than merely a slope change and is not represented exactly by this \(\beta_x\) term.

## 12. Parameters 9–10: jump at \(c_2\)

Define

\[
j_{c_2}(x)
=m(x,c_2^+)-m(x,c_2^-)
=\beta_{c_2,0}+\beta_{c_2,1}\xi.
\]

Then

\[
\beta_{c_2,1}
=\frac{\operatorname{Cov}_w(\xi,j_{c_2})}
{\operatorname{Var}_w(\xi)},
\qquad
\beta_{c_2,0}
=\bar j_{c_2}-\beta_{c_2,1}\bar\xi.
\]

Its structural decomposition is

\[
\boxed{
\tau\beta_{c_2}(x)
=-\Delta_{c_2}C_a
+\sum_{h=1}^{H}\Delta_{c_2}U_{n+h,a}
+\Delta_{c_2}V_{n+H,T,a}.
}
\]

## 13. Parameters 11–13: transition sharpnesses

For an isolated transition,

\[
m(a)=s_-+\beta\sigma(\kappa(a-c)).
\]

Normalize the transition:

\[
z(a)=\frac{m(a)-s_-}{\beta}.
\]

Let \(a_{10}\) and \(a_{90}\) satisfy \(z(a_{10})=0.1\) and \(z(a_{90})=0.9\). Then

\[
w_{10\text{--}90}=a_{90}-a_{10}
\]

and

\[
\boxed{
\kappa=\frac{2\log9}{w_{10\text{--}90}}
\approx\frac{4.394}{w_{10\text{--}90}}.
}
\]

Therefore

\[
\kappa_{c_1}
\approx\frac{4.394}{w_{c_1,10\text{--}90}},
\qquad
\kappa_{c_2}
\approx\frac{4.394}{w_{c_2,10\text{--}90}}.
\]

For \(\kappa_x\), align all slices by

\[
u=a-x,
\]

subtract the baseline and fixed transitions, pool the moving residuals, and measure their 10--90 percent width in \(u\):

\[
\kappa_x
\approx\frac{4.394}{w_{x,10\text{--}90}}.
\]

These formulas are initial estimators. Final values come from minimizing the direct score loss.

A literal regret kink corresponds to \(\kappa\to\infty\). A finite fitted value may instead reflect grid resolution, regularization, history aggregation, a varying true breakpoint, or genuinely smooth costs. When a transition is poorly resolved, the local curvature product

\[
\frac{\beta\kappa}{4}
\]

may be much better identified than \(\beta\) and \(\kappa\) separately.

## 14. Compact support gates

Define the compact smoothstep

\[
H_\rho(t)=
\begin{cases}
0,&t\le0,\\[3pt]
\displaystyle
\sigma\!\left(\rho\left[\frac{1}{1-t}-\frac{1}{t}\right]\right),
&0<t<1,\\[10pt]
1,&t\ge1.
\end{cases}
\]

The endpoint gates are

\[
G_L(a)
=H_{\rho_L}\!\left(\frac{a-A_-}{w_L}\right),
\qquad
G_R(a)
=H_{\rho_R}\!\left(\frac{A_+-a}{w_R}\right).
\]

In regret units, these gates correspond to the additional barrier

\[
\boxed{
R_{\mathrm{gate}}(a)
=-\tau\log G_L(a)-\tau\log G_R(a).
}
\]

The support parameters are derivable from regret only if this boundary barrier is actually present in the oracle score or if the oracle regret is evaluated throughout the latent boundary layers.

## 15. Parameters 14–15: taper widths \(w_L,w_R\)

After fitting the interior score, define the residual

\[
e(x,a)
=Y(x,a)-Y_{\mathrm{interior}}(x,a)-\alpha(x).
\]

Choose \(\alpha(x)\) so that the residual is approximately zero on the central region where both gates equal one. Average over \(x\), then define

\[
\widehat G(a)=\exp(\bar e(a)).
\]

Near the left endpoint, \(G_R=1\), so \(\widehat G\approx G_L\). Because

\[
H_\rho(1/2)=1/2,
\]

if \(a_{L,1/2}\) is the point where \(\widehat G=1/2\), then

\[
\boxed{
w_L=2(a_{L,1/2}-A_-).
}
\]

Similarly, if \(a_{R,1/2}\) is the right half-gate point,

\[
\boxed{
w_R=2(A_+-a_{R,1/2}).
}

These are initial values. Refine them with the full score objective under

\[
w_L>0,
\qquad
w_R>0,
\qquad
w_L+w_R<A_+-A_-.
\]

## 16. Parameters 16–17: taper shapes \(\rho_L,\rho_R\)

For the left gate, let

\[
t=\frac{a-A_-}{w_L},
\qquad 0<t<1.
\]

Then

\[
\operatorname{logit}G_L(a)
=\rho_L
\left[\frac{1}{1-t}-\frac{1}{t}\right].
\]

Therefore \(\rho_L\) is the slope in the regression

\[
\operatorname{logit}\widehat G_L(a)
=\rho_L f(t),
\qquad
f(t)=\frac{1}{1-t}-\frac{1}{t}.
\]

The pointwise formula is

\[
\boxed{
\rho_L
=\frac{\operatorname{logit}\widehat G_L(a)}
{\frac{1}{1-t}-\frac{1}{t}}.
}
\]

Use multiple points and omit \(t=1/2\), where both numerator and denominator are zero.

For the right gate, use

\[
t=\frac{A_+-a}{w_R}
\]

and fit

\[
\boxed{
\rho_R
=\frac{\operatorname{logit}\widehat G_R(a)}
{\frac{1}{1-t}-\frac{1}{t}}.
}
\]

Again, these are initial estimates followed by joint nonlinear refinement.

## 17. What is and is not identifiable from visible data

If the fitted data contain only the visible interval \([V_-,V_+]\), for example \([-100,100]\), while the true latent support is \([-250,250]\), then the visible score contains no samples from the endpoint taper layers.

Consequently,

\[
\boxed{
w_L,w_R,\rho_L,\rho_R
\text{ cannot be identified from the visible distribution alone.}
}
\]

They must be:

- fitted from oracle regret evaluated near the latent endpoints;
- derived from an explicit exposure-limit penalty;
- or fixed as modeling choices.

Even complete latent data do not determine these parameters if the oracle merely declares \(a\notin[A_-,A_+]\) infeasible while keeping regret finite up to the boundary. In that case the smooth taper is not produced by the economic regret and should not be described as market-derived.

## 18. Separating the effects of \(H\) and \(T\)

The marginal score is

\[
\boxed{
-\frac{R_a(x,a)}{\tau}
=\frac{1}{\tau}
\left[
-C_a(x,a)
+\sum_{h=1}^{H}U_{n+h,a}(a)
+V_{n+H,T,a}(a)
\right].
}
\]

This suggests fitting the 17-parameter model over a grid of \((H,T)\):

- the immediate transition-cost contribution is paid once and should not scale with \(H\);
- per-step maintenance contributions to a fixed jump should scale approximately linearly with \(H\);
- cumulative returns shift \(b\) and introduce smooth curvature as \(H\) changes;
- setting \(T=0\) removes the continuation value;
- comparing \(T>0\) with \(T=0\) isolates continuation-policy effects;
- varying the fee multiplier isolates the direct fee contribution to \(\beta_x\).

Because changing costs can also change the optimal continuation path, exact attribution may contain interactions. Counterfactual oracle reruns are more reliable than assuming perfect additivity.

## 19. Outer nonlinear fit

The nine nonlinear parameters are

\[
\theta_{\mathrm{nonlin}}
=\left(
c_1,c_2,
\kappa_{c_1},\kappa_x,\kappa_{c_2},
w_L,w_R,\rho_L,\rho_R
\right).
\]

For each trial value of these parameters:

1. construct the softplus and gate features;
2. solve exactly or by ridge regression for the eight linear coefficients and nuisance slice offsets;
3. evaluate the weighted residual loss;
4. update only the nine nonlinear parameters in the outer optimizer.

This is a variable-projection fit. It is more stable than asking one optimizer to learn all 17 parameters simultaneously without exploiting the model's linear structure.

Use transformed unconstrained variables to maintain validity:

\[
\kappa_d=\kappa_{\min}+\operatorname{softplus}(q_d),
\qquad
\rho_d=\rho_{\min}+\operatorname{softplus}(r_d),
\]

and an ordered mapping for \(c_1,c_2\). Constrain the taper widths so that they are positive and do not overlap.

Use several starting points because breakpoint and width objectives may have local minima.

## 20. When only the optimal path is retained

If the oracle output contains only

\[
a^*(x)=\arg\min_aR(x,a),
\]

then it supplies only the first-order or subgradient condition at the optimum. It does not determine the regret away from that optimum and cannot identify 17 shape parameters.

The full counterfactual regret grid—or at least a sufficiently broad collection of regret differences—is required. This is precisely the information that distinguishes alternative target exposures that the oracle did not select.

## 21. Recommended implementation sequence

1. Fix \(\tau\), the latent support, and the visible observation window.
2. Build \(Y_{ji}=-R(x_j,a_i)/\tau\) from the full counterfactual regret table.
3. Initialize \(c_1,c_2\) from persistent curvature bands or known trading thresholds.
4. Initialize the three \(\kappa\)'s from measured 10--90 percent widths.
5. Fix the four taper parameters unless latent boundary-layer regret is available.
6. For every nonlinear-parameter proposal, solve the eight interior coefficients and the nuisance slice offsets by weighted linear regression.
7. Optimize the nine nonlinear parameters with multiple starts.
8. Optionally refine all identifiable parameters jointly using automatic differentiation.
9. Verify the fitted one-sided slope changes against local linear fits to regret.
10. Refit across several \((H,T)\) and fee settings to distinguish holding, continuation, and immediate-cost effects.

## 22. Final parameter-by-parameter summary

| Parameter | Direct source in regret | Closed form? |
|---|---|---|
| \(c_1\) | First persistent fixed location of marginal-regret change | Initialization from curvature; final nonlinear fit |
| \(c_2\) | Second persistent fixed location | Initialization from curvature; final nonlinear fit |
| \(b_0\) | Mean baseline marginal score | Yes, conditional on nonlinear parameters |
| \(b_1\) | Linear dependence of baseline marginal score on \(\xi\) | Yes, conditional on nonlinear parameters |
| \(\beta_{c_1,0}\) | Mean slope change at \(c_1\) | Yes, conditional on nonlinear parameters |
| \(\beta_{c_1,1}\) | \(\xi\)-dependence of that change | Yes, conditional on nonlinear parameters |
| \(\beta_{x,0}\) | Mean slope change at \(a=x\) | Yes; fee part may be analytic |
| \(\beta_{x,1}\) | \(\xi\)-dependence of moving change | Yes, conditional on nonlinear parameters |
| \(\beta_{c_2,0}\) | Mean slope change at \(c_2\) | Yes, conditional on nonlinear parameters |
| \(\beta_{c_2,1}\) | \(\xi\)-dependence of that change | Yes, conditional on nonlinear parameters |
| \(\kappa_{c_1}\) | Width of marginal-score transition at \(c_1\) | Width estimate, then nonlinear fit |
| \(\kappa_x\) | Width after aligning by \(a-x\) | Width estimate, then nonlinear fit |
| \(\kappa_{c_2}\) | Width at \(c_2\) | Width estimate, then nonlinear fit |
| \(w_L\) | Physical extent of latent left boundary barrier | Only with latent endpoint data |
| \(w_R\) | Physical extent of latent right boundary barrier | Only with latent endpoint data |
| \(\rho_L\) | Shape of left boundary barrier | Only with latent endpoint data |
| \(\rho_R\) | Shape of right boundary barrier | Only with latent endpoint data |

The central practical rule is

\[
\boxed{
\text{Fit regret values directly; use one-sided derivatives only to interpret the fitted jumps.}
}
\]

# Regret Differentiability and Direct Fitting

This note accompanies [Conditional Four-Segment Exponential Model with Log-Slope-Change Parameters](./conditional-four-segment-compact-support.md). It explains what regularity the regret function needs, how nondifferentiable regret is interpreted, and how the model should be fitted without requiring numerical derivatives.

## 1. Basic regret construction

Let \(R(x,a)\) be the regret associated with moving from current exposure \(x\) to target exposure \(a\). Suppose the conditional density is constructed as

\[
p(a\mid x)
\propto
\exp\!\left(-\frac{R(x,a)}{\tau}\right),
\]

where \(\tau>0\) is a temperature or regret-scale parameter. If the construction uses \(e^{-R}\), take \(\tau=1\).

Define the log-score

\[
S(x,a)=-\frac{R(x,a)}{\tau}.
\]

Conditional normalization adds a term depending on \(x\) but not \(a\):

\[
\log p(a\mid x)
=S(x,a)-\log Z(x).
\]

Therefore, wherever the derivative exists,

\[
\partial_a\log p(a\mid x)
=-\frac{1}{\tau}\partial_aR(x,a).
\]

## 2. Full differentiability is not required

The regret does **not** need to be differentiable everywhere. Proportional trading costs, maintenance-cost thresholds, exposure constraints, and changes in the oracle's optimal continuation policy naturally produce kinks.

For the density itself to be well defined, it is enough that \(R\) be measurable and that

\[
\int \exp\!\left(-\frac{R(x,a)}{\tau}\right)da
<\infty
\]

for every relevant \(x\).

For the log-slope-change interpretation, a convenient sufficient condition is that regret be:

- continuous inside the feasible exposure interval;
- differentiable between a finite number of breakpoints;
- equipped with finite one-sided derivatives at those breakpoints.

These conditions are useful but not required for fitting the regret values directly.

## 3. One-sided derivatives at a kink

At a breakpoint \(d\), define

\[
R'_-(x,d)
=\lim_{a\uparrow d}\partial_aR(x,a),
\qquad
R'_+(x,d)
=\lim_{a\downarrow d}\partial_aR(x,a).
\]

The fitted \(\beta_d(x)\) is the change in the log-density slope when the breakpoint is crossed:

\[
\boxed{
\beta_d(x)
=-\frac{1}{\tau}
\left[R'_+(x,d)-R'_-(x,d)\right].
}
\]

Thus the ordinary derivative may fail to exist at \(d\). The two one-sided derivatives contain exactly the information needed by the model.

### Example: proportional transition cost

Consider symmetric proportional regret

\[
R(x,a)=q|a-x|.
\]

At \(a=x\),

\[
R'_-(x,x)=-q,
\qquad
R'_+(x,x)=q.
\]

Therefore

\[
\boxed{
\beta_x
=-\frac{2q}{\tau}.
}
\]

The regret is not differentiable at \(a=x\), but its moving log-slope change is perfectly well defined.

For asymmetric proportional costs

\[
R(x,a)
=q_{\mathrm{buy}}(a-x)_+
+q_{\mathrm{sell}}(x-a)_+,
\]

the result is

\[
\boxed{
\beta_x
=-\frac{q_{\mathrm{buy}}+q_{\mathrm{sell}}}{\tau}.
}
\]

## 4. Direct fitting does not require derivatives

Differentiating regret is useful for interpretation and initialization, but it is not necessary for estimation. The preferred approach is to fit the sampled regret values directly.

For the conditional four-segment model, write

\[
\begin{aligned}
-\frac{R(x,a)}{\tau}
\approx{}&\alpha(x)
+b(\xi)(a-A_-)\\
&+\beta_{c_1}(\xi)
  \operatorname{SP}_{\kappa_{c_1}}(a-c_1)\\
&+\beta_x(\xi)
  \operatorname{SP}_{\kappa_x}(a-x)\\
&+\beta_{c_2}(\xi)
  \operatorname{SP}_{\kappa_{c_2}}(a-c_2)\\
&+\log G(a),
\end{aligned}
\]

where

\[
\operatorname{SP}_{\kappa}(z)
=\frac{1}{\kappa}\log(1+e^{\kappa z}).
\]

The term \(\alpha(x)\) is a free additive constant for each conditional slice. It is necessary because the model's unnormalized log-kernel and \(-R/\tau\) may use different slice-wise offsets. It disappears when the density is normalized.

Direct value fitting works when regret is:

- kinked;
- available only on a discrete exposure grid;
- mildly noisy;
- generated by an oracle whose policy changes discontinuously;
- not reliable enough to differentiate numerically.

## 5. Why direct fitting is preferable

Numerical differentiation amplifies grid noise. Numerical second derivatives amplify it further. Fitting regret values avoids this amplification and uses all points in the surface simultaneously.

The derivative-based picture can still be used after fitting. The fitted marginal log-score is

\[
\begin{aligned}
\partial_a S_\theta(x,a)={}&b(\xi)\\
&+\beta_{c_1}(\xi)
  \sigma\!\left(\kappa_{c_1}(a-c_1)\right)\\
&+\beta_x(\xi)
  \sigma\!\left(\kappa_x(a-x)\right)\\
&+\beta_{c_2}(\xi)
  \sigma\!\left(\kappa_{c_2}(a-c_2)\right)\\
&+\partial_a\log G(a).
\end{aligned}
\]

This provides a smooth estimate of the marginal regret even if the original regret grid is nondifferentiable or noisy.

## 6. Interpretation of finite transition sharpness

A literal slope kink has zero transition width. In the softplus model, it corresponds to the limit

\[
\kappa\rightarrow\infty.
\]

A finite fitted \(\kappa\) means that the model represents the slope change over a nonzero exposure width. That width may arise from:

- genuinely smooth transaction cost or market impact;
- discrete exposure-grid resolution;
- regularization;
- interpolation of the regret table;
- aggregation over market histories;
- variation in the true breakpoint across histories;
- approximation of broad return curvature with a small number of transitions.

The sigmoid's 10--90 percent transition width is

\[
w_{10\text{--}90}
=\frac{2\log 9}{\kappa}
\approx\frac{4.394}{\kappa}.
\]

Consequently, a fitted \(\kappa\) should not automatically be interpreted as a primitive trading parameter. It is structural only when the underlying cost or value transition is itself smooth over that exposure scale.

## 7. Subgradients

If regret is convex but nondifferentiable, its subdifferential at \(d\) is

\[
\partial R(x,d)
=\left[R'_-(x,d),R'_+(x,d)\right].
\]

The width of this interval determines the slope jump:

\[
\beta_d(x)
=-\frac{\operatorname{width}(\partial R(x,d))}{\tau}
\]

when the derivative increases across the kink. More generally, retain the signed formula

\[
\beta_d(x)
=-\frac{R'_+(x,d)-R'_-(x,d)}{\tau}.
\]

This is often the correct language for proportional fees and oracle value functions defined by maxima over alternative trading paths.

## 8. Discrete regret grids

If regret is known only at exposure grid points \(a_1,\ldots,a_M\), no differentiability assumption is required. Fit

\[
\left\{-R(x_j,a_i)/\tau\right\}_{i,j}
\]

directly by weighted least squares, robust loss, or a likelihood appropriate to the way the target distribution was generated.

If approximate slopes are desired for diagnostics, estimate them with local linear fits on each side of a suspected breakpoint. Do not use a single central difference across a kink, because it averages the two one-sided slopes.

For example, near \(d\), fit

\[
R(x,a)\approx u_-+v_-(a-d)
\quad\text{for }a<d
\]

and

\[
R(x,a)\approx u_++v_+(a-d)
\quad\text{for }a>d.
\]

Then estimate

\[
\widehat\beta_d(x)
=-\frac{v_+-v_-}{\tau}.
\]

## 9. Cases not represented exactly by softplus transitions

The current model handles continuous functions with changes in slope. Other irregularities require care.

### Slope kink

A continuous regret with different left and right slopes is handled naturally. A large \(\kappa\) approximates a sharp kink.

### Flat optimum interval

If regret is constant over an interval, the density has a modal plateau rather than a unique optimum. The model can approximate this with near-zero log-slope over that region, although breakpoint placement may be weakly identified.

### Multiple isolated optima

The density can be multimodal. Three transitions may or may not be sufficient to reproduce all modes; additional slope changes may be required.

### Jump discontinuity in regret

A fixed fee paid for any nonzero trade can create a jump between \(a=x\) and nearby \(a\ne x\). A continuous softplus term cannot represent this exactly. Possible treatments are:

- add an explicit step or point-mass component;
- model the no-trade action separately;
- smooth the fixed-cost rule over a chosen resolution scale.

### Infinite regret outside the feasible range

This is handled separately by the strict-support construction or an indicator of feasibility. It does not require differentiability at the support boundary.

### Dense policy switching

If the oracle value has many closely spaced switches, three transitions may be insufficient. Use more knots, a spline representation, or a direct nonparametric approximation.

## 10. Minimal recommended workflow

1. Compute the complete counterfactual regret grid \(R(x,a)\).
2. Convert it to the target score \(-R(x,a)/\tau\).
3. Fit the score values directly, including a free additive offset \(\alpha(x)\) for each conditional slice.
4. Use local one-sided linear fits only to initialize or interpret breakpoint jumps.
5. Interpret

   \[
   \beta_d(x)
   =-\frac{R'_+(x,d)-R'_-(x,d)}{\tau}
   \]

   whenever the one-sided slopes exist.
6. Treat finite \(\kappa\) as an effective transition width unless the underlying trading model contains a genuinely smooth transition.
7. Add a separate model component if regret has actual jumps rather than merely slope jumps.

The central conclusion is:

\[
\boxed{
\text{Fit regret values directly; use derivatives only for interpretation.}
}
\]
