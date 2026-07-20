# Direct indicator to 17-parameter conditional predictor

This is the inspector's third exposure predictor, named **Direct indicator → 17-parameter distribution**. It shares the causal EWMA drift and variance state from the handcrafted forecast, but it does not construct a conditional regret surface and does not project or fit one. Its runtime path is:

1. Compute the one-dimensional forecast background
   \(F(a)=\sum_{h=1}^{H}\hat u_h(a)+V_1(a)\) on the oracle's latent current-exposure grid. The continuation scan uses the oracle's executable target grid and exact rebalance factor, while retaining only one value per background exposure.
2. Solve the decreasing marginal-background equations
   \(F'(c_1)=q_{buy}(c_1)\) and \(F'(c_2)=-q_{sell}(c_2)\), then enforce strict ordered interior boundaries.
3. Form the three background secants between \(L_0\), \(c_1\), \(c_2\), and \(U_0\).
4. Decode the eight slope-change coefficients directly and evaluate the existing analytic conditional four-segment distribution.

For the project's exact proportional rebalance rule,

\[
q_{buy}(a)=\frac{f}{1-f+fa},\qquad
q_{sell}(a)=\frac{f}{1-fa}.
\]

The action derivative depends on the target \(a\), while current exposure contributes only a row-wise additive constant. Consequently the four current-state coefficients \(b_1,\beta_{c_1,1},\beta_{x,1},\beta_{c_2,1}\) are exactly zero. The decoder uses exact fee secants on the outer background segments rather than replacing the fee with a constant penalty.

Maintenance uses the oracle's quote-lend, quote-borrow, and asset-borrow equity factor in log space. Its piecewise exposure dependence is therefore already present in the numerical derivative of \(F\) used to solve both boundaries.

## Fixed support and smoothing convention

The oracle's effective-exposure limits are the latent bounds. The executable target limits are only the visible hard truncation and do not set the tapers. Both support widths are fixed at 20% of the latent span and both support sharpnesses are fixed at 1. These are design settings, not fitted forecast constants.

When a 10–90% transition width is resolvable from the one-dimensional background marginal slope, the decoder uses \(\kappa=4.394/w_{10-90}\). A literal or numerically unresolved kink uses `transitionWidthGridCells` times the median executable-action grid spacing. The moving current-exposure transition is a literal fee kink and always uses this grid convention. This is the seventh direct-model calibration constant; the other six are the shared causal drift/variance constants. Oracle \(H\), \(T\), temperature, fees, maintenance rates, constraints, and grids are never fitted predictor constants.

## Stored calibration

`scripts/calibrate-direct-indicator-predictor.ts` uses the same 144 deterministic candidates, four coordinate-refinement rounds, eight causal samples per static window, and equal-window weighting as the handcrafted calibration. It stores one global fit and a separately executed local optimization for every static inspector window.

The 2026-07-20 real-history calibration used all 33 static windows and 264 causal samples:

- direct global mean conditional decision regret: 0.015293869;
- direct global uniform-current conditional cross-entropy: 6.402041493;
- previous handcrafted global regret on the same conditional comparison: 0.018302050;
- previous handcrafted global conditional cross-entropy: 6.460336984.

The direct default, before calibration, scored regret 0.016648632 and conditional cross-entropy 6.134009843. The calibrated CE is higher because the inherited optimization objective minimizes decision regret; CE remains a diagnostic metric.
