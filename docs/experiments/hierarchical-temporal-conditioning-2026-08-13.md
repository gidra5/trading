# Hierarchical temporal conditioning: 1d to 8h to 4h

Date: 2026-08-13  
Instrument: Binance spot BTCUSDT  
Forecast features: period log return, integrated 1s variance, and active-second count

## Result

Hierarchical conditioning is feasible and can improve forecast accuracy, but the useful rule is feature-specific. Exact top-down conditioning everywhere is not beneficial.

The strongest result comes from preserving a selected daily scenario, correcting the six 4h child allocations from the latest completed hierarchical forecast errors, and deriving the three 8h totals from those children. On the chronological test this improves joint variance accuracy by 11.8% and joint activity accuracy by 18.1% relative to independently calibrated forecasts. Return gains are limited to a 2.0% improvement in the joint scenario score and do not improve marginal CRPS.

The final hierarchy is exactly coherent to floating-point precision:

- daily return equals the sum of three 8h returns and six 4h returns;
- daily integrated variance equals the sum of all child variances;
- daily active seconds equal the sum of all child active-second counts;
- each 8h node equals its two 4h children.

## Chronological design

There are 365 UTC-day forecast origins from 2025-07-25 through 2026-07-24. Each origin generates 64 possible next-day target hierarchies.

- First 183 origins: history-window selection, calibration-family selection, covariance/shrinkage estimation, and hierarchy-method selection.
- Last 182 origins beginning 2026-01-24: chronological test.
- Every forecast uses only information available before its origin.
- Online calibration uses only the latest 90 completed forecast/outcome pairs.

The child forecasts are sequential. The three 8h nodes and six 4h nodes describe distinct consecutive parts of the next day; they are not repeated samples of the first block.

## Strongest independent fits selected before the test

| Level | Return history | Variance history | Activity history |
| --- | ---: | ---: | ---: |
| 1d | 365d | 30d | 365d |
| 8h | 90d | 30d | 30d |
| 4h | 90d | 7d | 30d |

These choices differ from the previous one-prefix-per-day horizon audit because this experiment scores every sequential child node and selects windows only on its own earlier calibration segment.

## Reconciliation methods

The experiment compares:

1. Independently calibrated forecasts at 1d, 8h, and 4h.
2. Hard top-down reconciliation: keep the sampled daily target fixed and bridge/allocate all children to it.
3. Error-weighted reconciliation: project all ten node forecasts onto the exact hierarchy according to a shrunk forecast-error second-moment matrix.
4. Both coherent methods with and without rank-preserving cross-scale scenario pairing.

For return, bridges are additive. For variance and activity, allocations remain nonnegative. Activity is additionally bounded by the number of seconds in each node.

The earlier segment selected these exact methods:

| Feature | Pre-test selected method |
| --- | --- |
| Return | Hard top-down with rank-coupled scenarios |
| Integrated variance | Error-weighted, uncoupled members |
| Active seconds | Error-weighted, uncoupled members |

All three selected methods are exactly coherent; independence was a permitted selection option but was not chosen.

## Confirmatory result for the pre-test choices

Positive skill means lower error than the independently calibrated baseline.

| Feature | Joint energy-score skill | 1d CRPS skill | 8h CRPS skill | 4h CRPS skill | All-node CRPS skill |
| --- | ---: | ---: | ---: | ---: | ---: |
| Return | +1.30% | 0.00% | -0.37% | -0.86% | -0.52% |
| Integrated variance | +1.87% | +5.72% | +3.98% | -3.20% | +1.75% |
| Active seconds | +3.96% | -1.70% | +1.25% | -0.66% | -0.29% |

Thus the pre-test selection confirms modest joint gains and a real variance CRPS gain, but not a general marginal improvement at every scale.

## Parent-preserving child residual model

The basic reconciliation left child distributions miscalibrated. A follow-up layer therefore uses the latest 90 completed six-dimensional 4h residual vectors to correct child allocations, then bridges them back to the already selected daily ensemble member and aggregates upward. Positive variance corrections are bounded to prevent near-zero projected children from creating unbounded log ratios.

| Feature | Joint energy-score skill | 1d CRPS skill | 8h CRPS skill | 4h CRPS skill | All-node CRPS skill |
| --- | ---: | ---: | ---: | ---: | ---: |
| Return | +2.04% | 0.00% | -0.77% | -1.22% | -0.82% |
| Integrated variance | **+11.83%** | +5.72% | +4.34% | +5.63% | **+5.23%** |
| Active seconds | **+18.05%** | -1.70% | +8.82% | **+16.23%** | **+9.17%** |

This is strong evidence that the information is in the conditional child-allocation residuals, especially for activity and variance.

However, this follow-up architecture was added after inspecting the initial reconciliation result. Its per-origin parameters remain causal, but these percentages are exploratory rather than a fresh confirmatory holdout.

## Calibration

The independent online-calibrated forecasts start from good marginal central-interval coverage. Reconciliation improves joint accuracy but can distort predictive ranks.

For the parent-preserving residual model, maximum absolute central-interval coverage errors at 1d / 8h / 4h are:

- Return: 4.4% / 5.3% / 3.3%; PIT tests pass separately at all three levels.
- Variance: 15.9% / 11.9% / 9.9%; PIT tests reject at all levels.
- Activity: 11.9% / 7.6% / 5.8%; PIT tests reject despite strong CRPS improvement.

So the variance/activity hierarchy is sharper and more accurate by CRPS and joint energy score, but it is not yet a fully trustworthy probabilistic distribution. Accuracy and calibration are improving along different axes.

## Conclusion and next extension

The experiment supports extending the hierarchy downward, with three constraints:

- Keep return conditioning soft because its marginal predictive gain is negligible.
- Model variance and activity as conditional child shares/residual vectors rather than blindly imposing the parent total on independent children.
- Calibrate at the bottom of each subtree and aggregate upward so calibration never breaks exact coherence.

The next tree should add 2h, 1h, 30m, and 15m nodes using the same parent-preserving residual-share construction. Only after those levels pass a new chronological audit should the process allocate 1m and 1s variance/activity and generate full candles.

## Artifacts

- Evaluator: `ml/evaluate_hierarchical_temporal_reconciliation.py`
- Tests: `ml/test_evaluate_hierarchical_temporal_reconciliation.py`
- Full report: `data/benchmarks/hierarchical-temporal-reconciliation.json`
- Commands: `npm run analysis:hierarchical-reconciliation` and `npm run analysis:hierarchical-reconciliation:test`
