# Direct indicator to six-parameter quadratic predictor

The direct predictor converts its causal one-dimensional forecast background
into the same quadratic conditional policy used by the MLP. It does not build a
two-dimensional regret surface.

1. Compute the background values and finite-difference marginal slopes.
2. Locate ordered `c1/c2` from the buy/sell marginal-cost crossings.
3. Estimate transition widths and their positive kappas.
4. Subtract the fee-derived moving transition.
5. Solve the weighted linear design for `b`, `lambda`, `betaC1`, and `betaC2`.

The reported six-value vector is

```text
[c1, c2, b, lambda, betaC1, betaC2]
```

Support geometry, `betaX`, and kappas are diagnostics/decoder values, not
members of this learned vector. The quadratic separates smooth background
curvature from localized changes at `c1/c2`.
