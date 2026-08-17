# Hierarchical ensemble point forecasts

Generated: 2026-08-13T03:43:10.369035Z

This audit asks whether the fitted probabilistic process can forecast the realized future path, not merely reproduce its unconditional statistics. All forecasts originate at 00:00 UTC; the final 91 days remain untouched by model, estimator, or member selection.

## Key findings

- Validated signed-return point edge found: **no**. Every scale and cumulative horizon selected the zero-return fallback after the pre-holdout fold gate.
- At 1s, the raw ensemble mean correlation was -0.000536; its MSE skill versus zero was -0.054335.
- The process contains more scale information than direction information: expected absolute 1m return correlated 0.146064 with realized absolute return, while 1s activity probability AUC was only 0.510109.
- The ensemble-mean cumulative price path had mean within-day correlation 0.013354 with the realized path.
- Mean is still the correct ensemble summary for squared-error decisions and median for absolute-error decisions, but here their forecast-only validation says to shrink them completely to zero for signed returns.

## Results

| scale | estimator | correlation | MAE (bps) | RMSE (bps) | MSE skill vs zero |
|---|---|---:|---:|---:|---:|
| 1s | ensembleMean | -0.000536 | 0.248991 | 0.589700 | -0.054335 |
| 1s | ensembleMedian | -0.000400 | 0.172037 | 0.574348 | -0.000153 |
| 1s | localDensityMode | 0.000513 | 0.171721 | 0.574310 | -0.000023 |
| 1s | validationSelectedShrinkageBlend | n/a | 0.171670 | 0.574304 | 0.000000 |
| 1s | pathMedoid | -0.000173 | 0.336967 | 0.791158 | -0.897767 |
| 1m | ensembleMean | 0.005688 | 3.535998 | 5.598343 | -0.001540 |
| 1m | ensembleMedian | 0.002185 | 3.535502 | 5.599209 | -0.001850 |
| 1m | localDensityMode | -0.000512 | 3.628334 | 5.671473 | -0.027877 |
| 1m | validationSelectedShrinkageBlend | n/a | 3.520615 | 5.594037 | 0.000000 |
| 1m | pathMedoid | -0.003954 | 5.093877 | 7.226213 | -0.668671 |
| 15m | ensembleMean | -0.004785 | 14.020948 | 21.379673 | -0.000357 |
| 15m | ensembleMedian | 0.015172 | 14.111002 | 21.450772 | -0.007022 |
| 15m | localDensityMode | -0.009548 | 15.346870 | 22.675459 | -0.125292 |
| 15m | validationSelectedShrinkageBlend | n/a | 14.018870 | 21.375854 | 0.000000 |
| 15m | pathMedoid | -0.025123 | 21.002596 | 29.240935 | -0.871266 |
| 30m | ensembleMean | -0.008411 | 19.644286 | 30.239021 | -0.000572 |
| 30m | ensembleMedian | 0.004192 | 19.881453 | 30.428600 | -0.013157 |
| 30m | localDensityMode | -0.000496 | 22.065164 | 32.440510 | -0.151564 |
| 30m | validationSelectedShrinkageBlend | n/a | 19.650783 | 30.230377 | 0.000000 |
| 30m | pathMedoid | -0.034431 | 29.325108 | 41.129037 | -0.851015 |
| 1h | ensembleMean | -0.027515 | 27.989604 | 43.218725 | -0.001482 |
| 1h | ensembleMedian | 0.049299 | 28.071788 | 43.151278 | 0.001641 |
| 1h | localDensityMode | 0.035317 | 29.565710 | 44.514246 | -0.062423 |
| 1h | validationSelectedShrinkageBlend | n/a | 27.982463 | 43.186733 | 0.000000 |
| 1h | pathMedoid | -0.037213 | 40.323256 | 56.904302 | -0.736159 |
| 2h | ensembleMean | -0.004155 | 40.496535 | 61.422541 | -0.000746 |
| 2h | ensembleMedian | 0.009675 | 41.196517 | 61.727583 | -0.010710 |
| 2h | localDensityMode | 0.006108 | 45.792179 | 65.548395 | -0.139705 |
| 2h | validationSelectedShrinkageBlend | n/a | 40.530289 | 61.399649 | 0.000000 |
| 2h | pathMedoid | -0.023324 | 55.386712 | 76.272404 | -0.543132 |
| 4h | ensembleMean | 0.001415 | 57.393155 | 82.116336 | -0.000970 |
| 4h | ensembleMedian | -0.047508 | 58.613842 | 82.966226 | -0.021797 |
| 4h | localDensityMode | -0.011196 | 63.282892 | 86.645713 | -0.114439 |
| 4h | validationSelectedShrinkageBlend | n/a | 57.397745 | 82.076530 | 0.000000 |
| 4h | pathMedoid | -0.006985 | 74.312735 | 99.786267 | -0.478099 |
| 8h | ensembleMean | 0.025348 | 79.553650 | 110.883212 | 0.000665 |
| 8h | ensembleMedian | 0.002700 | 80.748188 | 111.415366 | -0.008950 |
| 8h | localDensityMode | 0.029467 | 84.060272 | 113.697304 | -0.050703 |
| 8h | validationSelectedShrinkageBlend | n/a | 79.807211 | 110.920099 | 0.000000 |
| 8h | pathMedoid | -0.021821 | 99.479596 | 131.219231 | -0.399505 |
| 1d | ensembleMean | -0.033465 | 140.992779 | 182.227067 | -0.012404 |
| 1d | ensembleMedian | 0.090646 | 144.042219 | 181.211974 | -0.001156 |
| 1d | localDensityMode | 0.194932 | 155.148265 | 194.326008 | -0.151304 |
| 1d | validationSelectedShrinkageBlend | n/a | 140.711594 | 181.107317 | 0.000000 |
| 1d | pathMedoid | 0.010243 | 159.668259 | 201.654571 | -0.239779 |

## Interpretation

The ensemble mean is the correct point summary when optimizing squared return error, the median when optimizing absolute error, and the mode only when the most likely local value is the desired decision. At one second the exact-zero atom makes the mode frequently zero. A pointwise median or mode is not necessarily a dynamically valid sampled path; the medoid is included when one coherent representative scenario is required.

The validation-selected blend is admitted only when its shrinkage-only coefficients beat a zero-return forecast overall and in every pre-holdout chronological validation fold. If no candidate passes, it is exactly the zero-return baseline; this prevents an apparently useful final-holdout correlation from being selected after it is observed.

Signed-return accuracy and distributional scenario quality answer different questions. Low signed correlation does not invalidate calibrated uncertainty, volatility, or activity forecasts. Conversely, a plausible generated distribution does not establish that a particular realized candle path is predictable.

Machine-readable results: `data/benchmarks/hierarchical-point-forecasts.json`.
