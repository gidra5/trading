# Static 64-knot one-second return-density fits

Generated 2026-08-17T15:55:11.654977Z. The fit covers nonzero BTCUSDT 1-second log returns from 2021-07-25T00:00:00.000Z through 2026-07-25T00:00:00.000Z; the exact-zero mass of 35.727134% remains separate.

## Method

The return transform is frozen:

```text
u = sigmoid(4.043721227 * asinh((r - (-0.002437250 bps)) / 2.313554716 bps))
```

Each fit uses 64 movable knots on the complete unit interval and a normalized continuous piecewise-linear density. The four fits optimize JS bits, forward KL bits, original-return CDF-L2, and the equal relative joint objective respectively.

## Accuracy

| optimized for | JS (bits) | KL (bits) | CDF-L2 (bps) | normalized CDF-L2 | total variation | max CDF error |
|---|---:|---:|---:|---:|---:|---:|
| js | 0.015457393 | 0.063296881 | 4.42105381e-06 | 4.08812959e-06 | 0.083692690 | 0.011191788 |
| kl | 0.015498552 | 0.061788532 | 2.51310227e-06 | 2.32385495e-06 | 0.082512104 | 0.009748270 |
| cdfL2 | 0.022036957 | 0.081930974 | 1.17765356e-06 | 1.08897122e-06 | 0.096993885 | 0.007119215 |
| joint | 0.015568806 | 0.063245762 | 1.31268151e-06 | 1.21383099e-06 | 0.084307592 | 0.009836677 |

The full-distribution equivalents include the common exact-zero point mass. Because that mass is identical in every target and fit, full JS and KL equal their active values times the active probability; CDF-L2 is multiplied by the squared active probability.

## Pairwise fit differences

| pair | JS (bits) | total variation | max CDF difference | CDF-L2 (bps) | median knot difference (bps) | max knot difference (bps) |
|---|---:|---:|---:|---:|---:|---:|
| jsVsKl | 0.000224413 | 0.011373121 | 0.003268167 | 7.15898552e-06 | 0.000852434993 | 0.00470757014 |
| jsVsCdfL2 | 0.006845057 | 0.038086660 | 0.010067443 | 3.41552193e-06 | 0.00364577618 | 0.0925341855 |
| jsVsJoint | 0.000264993 | 0.009086944 | 0.002766794 | 3.1166145e-06 | 7.67825024e-05 | 0.00503147528 |
| klVsCdfL2 | 0.007168590 | 0.038533929 | 0.007480075 | 1.52655395e-06 | 0.00550906386 | 0.0971139985 |
| klVsJoint | 0.000468370 | 0.013197451 | 0.002654897 | 1.03188301e-06 | 0.000590257534 | 0.00507756553 |
| cdfL2VsJoint | 0.005871115 | 0.033391963 | 0.007466146 | 3.10724432e-07 | 0.00410510644 | 0.0974556289 |

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
