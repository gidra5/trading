# Handcrafted Indicator-Based Prediction of the Conditional Regret Model

This document describes a deterministic alternative to machine learning for predicting the conditional regret distribution and its 17 fitted parameters.

> Inspector implementation note: the original forecast-regret projection described here remains available as the handcrafted model. The separate [direct indicator to 17-parameter predictor](./direct-indicator-17-parameter-predictor.md) skips the two-dimensional regret surface and computes the analytic parameters from the one-dimensional forecast background.

It builds on:

- [Conditional Four-Segment Exponential Model with Log-Slope-Change Parameters](./conditional-four-segment-compact-support.md); and
- [Deriving and Fitting the 17-Parameter Conditional Regret Model](./regret-to-17-parameter-fit.md).

The proposed pipeline is

\[
\boxed{
\text{historical indicators}
\longrightarrow
\text{forecasted returns and costs}
\longrightarrow
\text{approximate dynamic-programming oracle}
\longrightarrow
\widehat R(x,a)
\longrightarrow
\text{17-parameter projection}.
}
\]

The method is fully causal: every indicator uses information available at the decision time. It does not attempt to reproduce the perfect future-aware oracle exactly. Instead, it approximates the appropriate ex-ante oracle under explicit indicator forecasts.

## 1. Inputs and notation

At decision time \(t\), let:

- \(x\) be current exposure;
- \(a\) be the candidate target exposure;
- \(H\) be the number of steps for which \(a\) must initially be held;
- \(T\) be the number of subsequent optimally controlled steps;
- \([A_-,A_+]\) be the full latent exposure support;
- \([V_-,V_+]\subset(A_-,A_+)\) be the visible observation window;
- \(\tau>0\) be the regret temperature.

The trading specification supplies:

- buy and sell fees;
- spread and slippage assumptions;
- maintenance or funding costs \(M_t(a)\);
- exposure constraints;
- optional market-impact parameters.

The market history available at time \(t\) supplies returns, volatility, volume, spread, funding, borrow rates, and other causal indicators.

## 2. Causal market indicators

The minimum useful indicator set contains forecasts of return, variance, and execution cost.

### 2.1 Expected return

Let \(r_t\) be the one-step asset return. An exponentially weighted estimate is

\[
\widehat\mu_t
=(1-\lambda_\mu)
\sum_{j=0}^{L-1}\lambda_\mu^jr_{t-j},
\qquad 0<\lambda_\mu<1.
\]

Forecast the signal with a chosen persistence:

\[
\widehat\mu_{t+h\mid t}
=\phi_\mu^{h-1}\widehat\mu_t,
\qquad h\ge1.
\]

A signal half-life \(h_\mu\) corresponds to

\[
\phi_\mu=2^{-1/h_\mu}.
\]

A richer but still handcrafted signal can combine trend and mean reversion:

\[
\widehat\mu_t
=\gamma_{\mathrm{trend}}z_{\mathrm{trend},t}
-\gamma_{\mathrm{MR}}z_{\mathrm{deviation},t},
\]

where the \(z\)-scores are constructed from historical prices and volatility. The coefficients and half-lives are a small set of transparent calibration constants.

### 2.2 Variance

Use an EWMA estimate

\[
\widehat v_t
=(1-\lambda_v)
\sum_{j=0}^{L-1}
\lambda_v^j(r_{t-j}-\widehat\mu_t)^2.
\]

A mean-reverting forecast is

\[
\widehat v_{t+h\mid t}
=\bar v+\phi_v^{h-1}(\widehat v_t-\bar v),
\]

where \(\bar v\) is a long-run variance estimate and \(0\le\phi_v<1\).

The forecasted second moment is

\[
\widehat s_{2,t+h\mid t}
=\widehat v_{t+h\mid t}
+\widehat\mu_{t+h\mid t}^{,2}.
\]

### 2.3 Liquidity and execution cost

Construct effective marginal costs from known fees and current liquidity:

\[
q_{\mathrm{buy},t}
=f_{\mathrm{buy}}
+\frac{\operatorname{spread}_t}{2}
+I_{\mathrm{buy},t},
\]

\[
q_{\mathrm{sell},t}
=f_{\mathrm{sell}}
+\frac{\operatorname{spread}_t}{2}
+I_{\mathrm{sell},t}.
\]

A simple impact rule is

\[
I_t
=\eta
\frac{\text{expected trade size}}
{\max(\text{recent depth or volume},\varepsilon)}.
\]

The transition-cost function can then be written as

\[
C_t(x,a)
=q_{\mathrm{buy},t}(a-x)_+
+q_{\mathrm{sell},t}(x-a)_+
+C_{\mathrm{nonlinear},t}(x,a).
\]

The nonlinear term can include quadratic impact, fixed costs, or any other known execution rule.

### 2.4 Maintenance and funding

If future rates are contractually fixed over the horizon, use them directly. Otherwise use a transparent forecast such as

\[
\widehat f_{t+h\mid t}
=\bar f+\phi_f^{h-1}(f_t-\bar f).
\]

Substitute the forecasted rate into the maintenance function

\[
M_{t+h\mid t}(a).
\]

Piecewise-linear maintenance is especially convenient because its slope changes have a direct relationship to fitted \(\beta_{c_i}\) jumps.

## 3. Forecasted one-step log utility

For small exposure-return products, approximate expected log growth by

\[
\boxed{
\widehat u_h(a)
=a\widehat\mu_{t+h\mid t}
-\frac{a^2}{2}\widehat s_{2,t+h\mid t}
-M_{t+h\mid t}(a).
}
\]

The terms have clear roles:

- \(a\widehat\mu\) rewards exposure in the forecast direction;
- \(-a^2\widehat s_2/2\) penalizes return uncertainty and leverage;
- \(-M(a)\) accounts for maintenance, funding, and borrow costs.

### 3.1 Empirical alternative

When \(|ar|\) is not small, replace the quadratic approximation with a weighted empirical expectation:

\[
\widehat u_h(a)
=\sum_{j=1}^{L}\omega_{j,h}
\log(1+ar_{t-j})
-M_{t+h\mid t}(a),
\]

where

\[
\omega_{j,h}\ge0,
\qquad
\sum_j\omega_{j,h}=1.
\]

The weights may emphasize recent observations or observations with similar volatility, trend, funding, and liquidity indicators. This remains a deterministic historical estimator rather than an opaque predictive model.

## 4. Handcrafted continuation oracle

The continuation value should be calculated by backward dynamic programming on an exposure grid. This reproduces the structure of the perfect oracle while replacing the unknown future path with indicator forecasts.

Choose a grid

\[
\mathcal A=\{a_1,\ldots,a_M\}\subset[A_-,A_+].
\]

Set the value after the final continuation step to zero:

\[
V_{T+1}(z)=0.
\]

For \(h=T,T-1,\ldots,1\), compute

\[
\boxed{
V_h(z)
=\max_{y\in\mathcal A}
\left[
-C_{t+H+h}(z,y)
+\widehat u_{H+h}(y)
+V_{h+1}(y)
\right].
}
\]

Here:

- \(z\) is the exposure before the continuation transition;
- \(y\) is the selected next exposure;
- all costs and utilities use forecasts made at time \(t\).

The function \(V_1(a)\) is the predicted value of proceeding optimally for \(T\) steps after arriving at the continuation period with exposure \(a\).

### 4.1 Computational cost

The direct grid recursion costs approximately

\[
O(TM^2).
\]

For a one-dimensional exposure grid this is usually practical. If costs are convex or proportional, monotonicity and convex-optimization methods can reduce the cost, but the simple grid implementation is the best initial reference.

## 5. Predicted value of holding \(a\) for \(H\) steps

For every pair \((x,a)\), calculate

\[
\boxed{
\widehat J_t^{H,T}(x,a)
=-C_t(x,a)
+\sum_{h=1}^{H}\widehat u_h(a)
+V_1(a).
}
\]

This has the same structure as the perfect oracle label:

1. transition from current exposure \(x\) to candidate \(a\);
2. hold \(a\) for \(H\) steps;
3. continue optimally for \(T\) forecasted steps.

The predicted optimal candidate is

\[
\widehat a^*(x)
=\arg\max_{a\in\mathcal A}
\widehat J_t^{H,T}(x,a).
\]

Define predicted regret by

\[
\boxed{
\widehat R_t^{H,T}(x,a)
=\max_{y\in\mathcal A}\widehat J_t^{H,T}(x,y)
-\widehat J_t^{H,T}(x,a).
}
\]

Then

\[
\widehat R(x,a)\ge0,
\qquad
\min_a\widehat R(x,a)=0.
\]

## 6. Predicted conditional distribution

Construct the latent density

\[
\widehat q(a\mid x)
\propto
\exp\!\left(-\frac{\widehat R(x,a)}{\tau}\right)
\]

on \([A_-,A_+]\).

If only \([V_-,V_+]\) is visible, form the hard-truncated conditional density

\[
\boxed{
\widehat p_{\mathrm{vis}}(a\mid x)
=
\frac{
\exp[-\widehat R(x,a)/\tau]
}
{
\displaystyle
\int_{V_-}^{V_+}
\exp[-\widehat R(x,u)/\tau]du
}
}
\]

for \(a\in[V_-,V_+]\), and zero outside the visible interval as an observed density. No smoothing is applied at \(V_-\) or \(V_+\).

The indicator-based regret surface and distribution can be used directly. Projection into 17 parameters is necessary only when a compact analytic representation is required.

## 7. Projection into the 17-parameter model

Fit the score

\[
\widehat Y(x,a)
=-\frac{\widehat R(x,a)}{\tau}
\]

with

\[
\begin{aligned}
\widehat Y(x,a)\approx{}&\alpha(x)
+(b_0+b_1\xi)(a-A_-)\\
&+(\beta_{c_1,0}+\beta_{c_1,1}\xi)
  \operatorname{SP}_{\kappa_{c_1}}(a-c_1)\\
&+(\beta_{x,0}+\beta_{x,1}\xi)
  \operatorname{SP}_{\kappa_x}(a-x)\\
&+(\beta_{c_2,0}+\beta_{c_2,1}\xi)
  \operatorname{SP}_{\kappa_{c_2}}(a-c_2)\\
&+\log G_L(a)+\log G_R(a).
\end{aligned}
\]

For fixed breakpoints, transition sharpnesses, and support gates, solve the eight \(b,\beta\) coefficients and the nuisance slice offsets by weighted linear least squares. Optimize the nonlinear parameters in an outer loop as described in the 17-parameter fitting guide.

Because \(\widehat R\) is deterministic, this projection requires no parameter-prediction model.

## 8. Closed-form approximation for fixed breakpoints

The dynamic program is the preferred construction, but a simpler approximation provides interpretable initial values for \(c_1,c_2\).

Approximate a continuation block by

\[
F_T(y)
=A_Ty-\frac{B_T}{2}y^2-TM(y),
\]

where

\[
A_T
=\sum_{h=1}^{T}
\widehat\mu_{t+H+h\mid t},
\]

\[
B_T
=\sum_{h=1}^{T}
\widehat s_{2,t+H+h\mid t}.
\]

With asymmetric proportional costs, approximate the continuation value by

\[
V_T(a)
=\max_y
\left[
F_T(y)
-q_{\mathrm{buy}}(y-a)_+
-q_{\mathrm{sell}}(a-y)_+
\right].
\]

The no-trade boundaries satisfy

\[
F_T'(c_1)=q_{\mathrm{buy}},
\qquad
F_T'(c_2)=-q_{\mathrm{sell}}.
\]

Since

\[
F_T'(y)=A_T-B_Ty-TM'(y),
\]

the approximate boundaries are

\[
\boxed{
c_1
\approx
\frac{A_T-TM'(c_1)-q_{\mathrm{buy}}}{B_T},
}
\]

\[
\boxed{
c_2
\approx
\frac{A_T-TM'(c_2)+q_{\mathrm{sell}}}{B_T}.
}
\]

Clip them to the latent support. When maintenance is piecewise linear, solve the formula in each region and keep the solution consistent with that region's derivative.

These are no-trade boundaries of the simplified continuation problem. The final fitted breakpoints may differ because the full dynamic program, finite horizon, and four-segment approximation contain additional structure.

## 9. Handcrafted parameter interpretation

Define the predicted marginal score

\[
\widehat m(x,a)
=\frac{1}{\tau}
\partial_a\widehat J(x,a)
=-\frac{1}{\tau}
\partial_a\widehat R(x,a)
\]

where one-sided derivatives are used at kinks.

### 9.1 Baseline \(b_0,b_1\)

Measure the marginal-score level before any transition has been crossed and outside the latent support taper. Fit

\[
b(\xi)=b_0+b_1\xi.
\]

It incorporates:

- cumulative forecast return over \(H\);
- forecast variance and log-growth curvature;
- baseline maintenance slope;
- the relevant side of the current transition cost;
- continuation marginal value.

### 9.2 Moving jump

For asymmetric proportional immediate costs, the directly known contribution is

\[
\boxed{
\beta_x^{\mathrm{fee}}
=-\frac{q_{\mathrm{buy}}+q_{\mathrm{sell}}}{\tau}.
}
\]

If the cost rates do not depend on \(x\), then their direct contribution to \(\beta_{x,1}\) is zero. Additional moving-jump behavior can arise from the continuation value or approximation interactions.

### 9.3 Fixed jumps

Measure

\[
\beta_{c_i}(x)
=\widehat m(x,c_i^+)-\widehat m(x,c_i^-)
\]

using one-sided levels outside the transition layer. Then regress

\[
\beta_{c_i}(x)
=\beta_{c_i,0}+\beta_{c_i,1}\xi.
\]

If maintenance has a slope change at \(c_i\), its known holding-period contribution is

\[
\boxed{
\beta_{c_i}^{\mathrm{maintenance}}
=-\frac{H}{\tau}
\left[M'(c_i^+)-M'(c_i^-)\right].
}
\]

### 9.4 Transition sharpness

For every transition, measure the exposure interval over which the marginal score completes 10 to 90 percent of its change:

\[
\boxed{
\kappa_d
=\frac{4.394}{w_{d,10\text{--}90}}.
}
\]

For the moving transition, align slices by \(u=a-x\) before measuring the width.

If a cost rule creates a literal kink, select a small effective width based on exposure resolution, for example

\[
w_{d,10\text{--}90}=k_\Delta\Delta a
\]

with \(k_\Delta\) between roughly one and several grid cells, then refine it through projection error.

### 9.5 Support tapers

Keep

\[
w_L,w_R,\rho_L,\rho_R
\]

fixed unless the indicator-based oracle includes a genuine diverging penalty near the latent endpoints. These parameters close the chosen latent support and are not forecasts of current market conditions.

## 10. Using indicator uncertainty to choose \(\kappa\)

The transition widths can also reflect uncertainty in the handcrafted indicators.

Suppose a breakpoint estimate has uncertainty \(\sigma_c\). Averaging a sharp transition over uncertain boundary locations produces a smooth transition. A useful logistic approximation is

\[
\boxed{
\kappa\approx\frac{1.7}{\sigma_c}.
}
\]

Estimate \(\sigma_c\) by deterministic sensitivity analysis:

1. perturb \(\widehat\mu\) upward and downward;
2. perturb forecast variance;
3. perturb spread, impact, funding, and maintenance;
4. recompute the dynamic program or closed-form boundary;
5. calculate the standard deviation or robust range of the resulting breakpoint.

This gives the desired qualitative behavior:

- confident indicators produce sharp transitions and large \(\kappa\);
- uncertain indicators produce broad transitions and small \(\kappa\).

## 11. Scenario-based uncertainty without machine learning

A single indicator forecast can be replaced by a small explicit scenario set. For example:

\[
s\in\{\text{bearish},\text{base},\text{bullish}\}
\times
\{\text{low-volatility},\text{high-volatility}\}.
\]

Assign scenario weights \(\pi_s\) from indicator rules and construct a regret surface \(\widehat R_s\) under each scenario.

For a minimum expected-regret decision, use

\[
\widehat R_{\mathrm{mean}}(x,a)
=\sum_s\pi_s\widehat R_s(x,a).
\]

For a conservative decision, use

\[
\widehat R_{\mathrm{robust}}(x,a)
=\sum_s\pi_s\widehat R_s(x,a)
+\lambda_{\mathrm{risk}}
\operatorname{SD}_s[\widehat R_s(x,a)].
\]

Do not confuse the density produced from mean regret with the average of scenario densities:

\[
\exp\!\left(-\frac{\mathbb E_s[R_s]}{\tau}\right)
\ne
\mathbb E_s
\left[
\exp\!\left(-\frac{R_s}{\tau}\right)
\right].
\]

The first targets expected regret. The second describes a mixture of scenario-conditional oracle distributions.

## 12. Regime-table alternative

A simpler historical method is a handcrafted lookup table.

Define discrete regimes such as:

- trend: bearish, neutral, bullish;
- volatility: low, medium, high;
- liquidity: liquid, normal, stressed;
- funding: favorable, neutral, expensive.

For every regime and \((H,T)\) bucket, store either:

- the historical average oracle regret surface; or
- the historical average fitted interior parameters.

At runtime:

1. calculate the current indicator regime;
2. retrieve neighboring stored surfaces;
3. interpolate between regime cells;
4. replace their fee and maintenance contributions with the current exact values;
5. project the resulting score onto the 17-parameter model.

This is transparent but becomes data-hungry when many indicators and regimes are used. Averaging regret surfaces is generally more stable than averaging fitted parameters.

## 13. Calibration without opaque prediction

The handcrafted system still contains a small number of constants:

- indicator half-lives;
- trend and mean-reversion weights;
- variance mean-reversion rate;
- impact coefficients;
- scenario perturbation sizes and weights;
- effective minimum transition width.

Calibrate these constants with walk-forward historical evaluation. The primary objective should be economic regret of the chosen exposure:

\[
\mathcal L_{\mathrm{decision}}
=\sum_t
R_t^{\mathrm{oracle}}
\left(
x_t,widehat a_t^*
\right).
\]

Secondary diagnostics include:

- error in the predicted regret surface;
- error in the optimal exposure;
- KL divergence between oracle and handcrafted conditional distributions;
- stability across market regimes;
- turnover and realized net log-return.

Use chronological walk-forward testing with an embargo of at least the maximum \(H+T\). Randomly splitting overlapping historical windows would leak future label information.

## 14. Recommended first implementation

Use the following minimal version:

1. Estimate drift with an EWMA return signal.
2. Decay the drift forecast exponentially with a fixed signal half-life.
3. Estimate variance by EWMA and mean-revert it toward a long-run level.
4. Use exact fee, spread, maintenance, and exposure-constraint formulas.
5. Approximate expected one-step log utility with

   \[
   \widehat u_h(a)
   =a\widehat\mu_h
   -\frac{a^2}{2}\widehat s_{2,h}
   -M_h(a).
   \]

6. Run the one-dimensional backward dynamic program for \(T\) steps.
7. Add the initial transition cost and \(H\)-step hold value.
8. Construct \(\widehat R(x,a)\) and the corresponding conditional distribution.
9. Use the predicted regret surface directly or project it into the 17 parameters.
10. Fix all four latent support-taper parameters.
11. Initialize transition widths from grid resolution or indicator sensitivity.
12. Calibrate only the small set of indicator constants by walk-forward decision regret.

## 15. Pseudocode

```text
input:
    market history through time t
    H, T
    fees, spread, impact, maintenance rules
    exposure grid A

indicators:
    mu0  = ewma_return(history)
    var0 = ewma_variance(history)
    current liquidity and funding state

for h = 1 .. H + T:
    mu[h]  = forecast_drift(mu0, h)
    var[h] = forecast_variance(var0, h)

    for a in A:
        utility[h, a] =
            a * mu[h]
            - 0.5 * a^2 * (var[h] + mu[h]^2)
            - maintenance_forecast(h, a)

for z in A:
    V[T + 1, z] = 0

for h = T down to 1:
    calendar_step = H + h

    for z in A:
        V[h, z] = max over y in A of:
            - transition_cost(calendar_step, z, y)
            + utility[calendar_step, y]
            + V[h + 1, y]

for x in A:
    for a in A:
        J[x, a] =
            - transition_cost(0, x, a)
            + sum over h = 1 .. H of utility[h, a]
            + V[1, a]

    best = max over a in A of J[x, a]

    for a in A:
        regret[x, a] = best - J[x, a]

optional:
    fit the 17-parameter score model to -regret / tau

output:
    predicted regret surface
    predicted conditional distribution
    optional 17-parameter representation
```

## 16. What this method can and cannot do

It can:

- incorporate \(H,T\), fees, maintenance, spread, and constraints explicitly;
- produce a complete counterfactual regret surface;
- preserve the dynamic structure of the perfect oracle;
- remain transparent and debuggable;
- provide all 17 parameters through deterministic projection;
- generate scenario-based uncertainty without a learned model.

It cannot:

- know the realized future returns used by the perfect oracle;
- discover predictive relationships absent from the selected indicators;
- infer latent endpoint-taper behavior from a visibly truncated sample;
- guarantee that a simple EWMA signal describes every market regime;
- reproduce nonlinear liquidity or continuation effects omitted from the handcrafted cost model.

The method should therefore be understood as an interpretable forecast oracle: it solves the same decision problem as the perfect oracle, but under explicit causal forecasts rather than realized future information.
