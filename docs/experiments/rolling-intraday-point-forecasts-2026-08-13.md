# Rolling intraday 15m–1h point forecasts

Generated: 2026-08-13T07:33:15.392638Z

These forecasts are refreshed immediately before every non-overlapping target window. History-window and ensemble-summary choices end before the final 91-day holdout.

## Key findings

- Refreshing immediately before the target window does **not** create a validated signed-return point forecast. Every pre-holdout blend gate still selects zero return.
- Raw signed-return correlations are -0.0148 at 15m, 0.0142 at 30m, and 0.0257 at 1h; all raw mean forecasts have negative MSE skill versus zero.
- Fresh state matters strongly for endpoint scale: expected absolute-return correlation is 0.370 at 15m, 0.350 at 30m, and 0.348 at 1h.
- Expected realized-variance correlation is 0.567–0.628, and expected activity correlation is 0.808–0.832.
- A pre-holdout 1.25× spread correction substantially improves interval coverage, although 90% coverage remains only 84–86% and CRPS becomes slightly worse.
- The selected history is 30d at all three horizons, but its advantage over 7–14d is small; this should be treated as a stable smoothing preference, not a sharp optimum.
- After materializing 377,395,200 generated seconds, 1s ensemble-mean correlations are -0.000008, -0.000084, and -0.000370; the internal realized path remains unpredictable.

## Untouched results

| horizon | origins | selected history | estimator | correlation | MAE (bps) | RMSE (bps) | MSE skill vs zero |
|---|---:|---:|---|---:|---:|---:|---:|
| 15m | 8736 | 30d | ensembleMean | -0.014844 | 14.629554 | 22.167174 | -0.075409 |
| 15m | 8736 | 30d | ensembleMedian | -0.016766 | 14.577315 | 22.118132 | -0.070656 |
| 15m | 8736 | 30d | localDensityMode | -0.027776 | 15.830876 | 23.794122 | -0.239060 |
| 15m | 8736 | 30d | validationSelectedShrinkageBlend | n/a | 14.018870 | 21.375854 | 0.000000 |
| 15m | 8736 | 30d | lastBlockReturnBaseline | -0.017005 | 20.422047 | 30.484350 | -1.033794 |
| 30m | 4368 | 30d | ensembleMean | 0.014170 | 20.278493 | 30.945030 | -0.047839 |
| 30m | 4368 | 30d | ensembleMedian | -0.000688 | 20.302728 | 31.044907 | -0.054614 |
| 30m | 4368 | 30d | localDensityMode | -0.013031 | 22.158240 | 33.136699 | -0.201521 |
| 30m | 4368 | 30d | validationSelectedShrinkageBlend | n/a | 19.650783 | 30.230377 | 0.000000 |
| 30m | 4368 | 30d | lastBlockReturnBaseline | -0.033481 | 28.666437 | 43.457546 | -1.066537 |
| 1h | 2184 | 30d | ensembleMean | 0.025669 | 28.684747 | 44.042703 | -0.040033 |
| 1h | 2184 | 30d | ensembleMedian | 0.034390 | 28.715631 | 43.965360 | -0.036384 |
| 1h | 2184 | 30d | localDensityMode | 0.004888 | 31.420410 | 47.098203 | -0.189345 |
| 1h | 2184 | 30d | validationSelectedShrinkageBlend | n/a | 27.982463 | 43.186733 | 0.000000 |
| 1h | 2184 | 30d | lastBlockReturnBaseline | -0.042594 | 41.428722 | 62.350784 | -1.084410 |

## Probabilistic calibration

| horizon | selected spread | raw CRPS | calibrated CRPS | raw 90% coverage | calibrated 90% coverage |
|---|---:|---:|---:|---:|---:|
| 15m | 1.25 | 10.940729 | 11.032501 | 0.792582 | 0.861149 |
| 30m | 1.25 | 15.222528 | 15.343605 | 0.799679 | 0.863324 |
| 1h | 1.25 | 21.684313 | 21.859332 | 0.800366 | 0.870879 |

## Protocol

- Window selection: days 92–183 of the forecast year.
- Blend stability validation: days 184–274, split into three chronological folds.
- Untouched test: final 91 days.
- Candidate histories: 1d, 3d, 7d, 14d, and 30d.
- Each ensemble has 16 freshly sampled target paths; no historical block is resampled.
- A complete-window return is tested before materializing its 1s allocation, because second-level projection preserves that endpoint exactly.

## Materialized one-second paths

| horizon | generated seconds | mean 1s correlation | mean 1s RMSE (bps) | expected absolute-return correlation | activity AUC | mean cumulative-path correlation |
|---|---:|---:|---:|---:|---:|---:|
| 15m | 125798400 | -0.000008 | 0.592495 | 0.187090 | 0.550777 | -0.008307 |
| 30m | 125798400 | -0.000084 | 0.592028 | 0.156011 | 0.548039 | 0.004671 |
| 1h | 125798400 | -0.000370 | 0.592072 | 0.120567 | 0.543346 | 0.016989 |

Machine-readable results: `data/benchmarks/rolling-intraday-point-forecasts.json`.
