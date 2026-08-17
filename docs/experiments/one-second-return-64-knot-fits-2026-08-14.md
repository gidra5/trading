# Static 64-knot one-second return-density fits

Generated 2026-08-14T15:08:32.387219Z. The fit covers nonzero BTCUSDT 1-second log returns from 2021-07-25T00:00:00.000Z through 2026-07-25T00:00:00.000Z; the exact-zero mass of 35.727134% remains separate.

## Method

The return transform is frozen:

```text
u = sigmoid(4.043721227 * asinh((r - (-0.002437250 bps)) / 2.313554716 bps))
```

Each fit uses 64 movable knots on the complete unit interval and a normalized continuous piecewise-linear density. The four fits optimize JS bits, forward KL bits, original-return CDF-L2, and the equal relative joint objective respectively.

## Accuracy

| optimized for | JS (bits) | KL (bits) | CDF-L2 (bps) | normalized CDF-L2 | total variation | max CDF error |
|---|---:|---:|---:|---:|---:|---:|
| js | 0.003157429 | 0.013044961 | 1.93748441e-06 | 1.79158356e-06 | 0.028113329 | 0.003197279 |
| kl | 0.003188343 | 0.012913390 | 1.95260427e-06 | 1.80556484e-06 | 0.027657429 | 0.001823358 |
| cdfL2 | 0.003208372 | 0.013023947 | 9.53395889e-07 | 8.8160111e-07 | 0.029284245 | 0.002101147 |
| joint | 0.003208372 | 0.013023947 | 9.53395889e-07 | 8.8160111e-07 | 0.029284245 | 0.002101147 |

The full-distribution equivalents include the common exact-zero point mass. Because that mass is identical in every target and fit, full JS and KL equal their active values times the active probability; CDF-L2 is multiplied by the squared active probability.

## Pairwise fit differences

| pair | JS (bits) | total variation | max CDF difference | CDF-L2 (bps) | median knot difference (bps) | max knot difference (bps) |
|---|---:|---:|---:|---:|---:|---:|
| jsVsKl | 0.000059093 | 0.006029633 | 0.001778632 | 5.3791042e-07 | 0.00057488133 | 0.0153266253 |
| jsVsCdfL2 | 0.000077461 | 0.007454103 | 0.003259018 | 1.01614985e-06 | 0.000362308229 | 0.140131544 |
| jsVsJoint | 0.000077461 | 0.007454103 | 0.003259018 | 1.01614985e-06 | 0.000362308229 | 0.140131544 |
| klVsCdfL2 | 0.000044298 | 0.005978717 | 0.002343357 | 1.13366e-06 | 0.000283529207 | 0.131672336 |
| klVsJoint | 0.000044298 | 0.005978717 | 0.002343357 | 1.13366e-06 | 0.000283529207 | 0.131672336 |
| cdfL2VsJoint | 0.000000000 | 0.000000000 | 0.000000000 | 2.65601035e-30 | 0 | 0 |

## Interpretation

- Lowest JS: `js`.
- Lowest KL: `kl`.
- Lowest CDF-L2: `cdfL2`.
- Pairwise JS and total variation measure differences between the reconstructed probability laws; knot-coordinate differences alone can exaggerate functional differences because different knot layouts may reconstruct nearly the same density.
- This is an in-sample representation audit. It does not measure whether a conditional history model forecasts the next return distribution.

## Reproducibility

```text
node scripts/run-ml-python.mjs ml/fit_return_density_knots.py
```

The machine-readable knots, weights, density heights, convergence details, and all cross-metrics are stored in `data/benchmarks/one-second-return-64-knot-fits.json`.
