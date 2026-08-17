# Walk-forward calibration of rolling candle forecasts

Date: 2026-08-13  
Instrument: Binance spot BTCUSDT  
Forecast horizons: 15m, 30m, 1h, 2h, 4h, 8h, and 1d

## Question

Can past forecast errors be used to make the simulated distribution of the next period more reliable without looking at that period's outcome?

This experiment calibrates probabilities, not future values. If a forecast repeatedly supplies a central 90% interval, approximately 90% of later realized values should fall inside it. The full rank or probability-integral-transform (PIT) distribution should also be uniform.

## Chronological design

The source forecast archive contains 365 rolling-origin forecasts from 2025-07-25 through 2026-07-24. Every raw forecast was already made using only candles preceding its origin.

- Calibration segment: first 183 origins, 2025-07-25 through 2026-01-23.
- Inner method fit: first 91 calibration origins.
- Inner method selection: following 92 calibration origins.
- Untouched test: final 182 origins, 2026-01-24 through 2026-07-24.

Five correction families were considered: no correction, empirical residual dressing, spread-standardized residual dressing, shrinkage-affine residual dressing, and shrinkage-affine standardized residual dressing. Method and raw model history window were selected without test outcomes.

Positive quantities are calibrated in log space, activity in logit space, return in its original space, and correlation in Fisher-$z$ space. This preserves their natural support.

Two applications of the selected correction are evaluated:

1. **Frozen:** fit once on all 183 calibration origins and do not update during the test.
2. **Online 90d:** before each test forecast, refit from the most recent 90 completed forecast/outcome pairs. The current outcome and all future outcomes remain unavailable.

Both output 16 calibrated ensemble members, matching the raw ensemble size.

## Main result

The correction must adapt. A frozen correction improves central coverage but retains substantial rank distortion. Refreshing it from completed forecasts makes PIT ranks much closer to uniform.

Across all seven measured path statistics, seven horizons, and the history window selected before the untouched test:

| Test result | Raw | Frozen calibration | Online 90d calibration |
| --- | ---: | ---: | ---: |
| Cases with PIT uniformity not rejected at 5% | 11 / 49 | 8 / 49 | **36 / 49** |
| Cases with every 50/80/90/95% coverage error within 5 points | 2 / 49 | **14 / 49** | 5 / 49 |
| Cases with lower CRPS than raw | — | 34 / 49 | 33 / 49 |
| Cases with lower mean coverage error than raw | — | 36 / 49 | 36 / 49 |

The frozen calibration is better at hitting the four specifically checked central intervals, while the online calibration is much better across the complete rank distribution. This is not contradictory: four central-interval checks do not fully characterize a predictive distribution.

For the five main features—period return, integrated 1s variance, active fraction, minute path range, and maximum absolute minute return—the online result is:

| Horizon | Mean CRPS improvement | Features with lower CRPS | PIT passes | Raw mean maximum coverage error | Online mean maximum coverage error | Raw mean absolute 90% error | Online mean absolute 90% error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15m | 13.4% | 5 / 5 | 5 / 5 | 13.1% | 6.1% | 11.2% | 4.5% |
| 30m | 8.9% | 5 / 5 | 4 / 5 | 11.8% | 7.2% | 11.0% | 5.8% |
| 1h | 2.0% | 3 / 5 | 2 / 5 | 6.8% | 6.5% | 5.3% | 4.9% |
| 2h | 1.7% | 2 / 5 | 4 / 5 | 11.3% | 7.4% | 10.1% | 5.3% |
| 4h | 1.7% | 2 / 5 | 2 / 5 | 9.6% | 7.2% | 8.2% | 5.2% |
| 8h | 5.2% | 3 / 5 | 4 / 5 | 15.2% | 7.2% | 13.8% | 5.6% |
| 1d | 4.2% | 4 / 5 | 4 / 5 | 15.4% | 7.1% | 13.8% | 5.4% |

Across these 35 main-feature cases, online calibration improves CRPS in 24, reduces mean interval error in 26, and increases PIT passes from 8 to 25.

## Interpretation

The experiment confirms that recent realized forecast errors contain information not captured by the candle process's raw historical parameter fit. In particular, they reveal whether the current generator is too narrow, too broad, or biased for a feature and horizon.

The strongest improvements occur at 15m and 30m. At 1h through 4h, raw forecasts are already closer in sharpness and the correction is less consistently useful. At 8h and 1d, online correction again helps because regime drift creates large raw coverage errors.

This is not complete calibration yet:

- Thirteen of 49 online cases still reject uniform PIT ranks.
- Only five of 49 have every checked central-interval coverage error within five percentage points.
- Corrections are currently applied to distributions of path statistics. They do not yet create one jointly coherent candle path that simultaneously realizes corrected return, variance, activity, range, maximum move, and autocorrelation.
- The 90-day online correction history was specified before examining its test result. Other correction histories need their own nested chronological selection rather than selection on this test.

## Recommended model change

Use the online correction as a forecast layer, then feed the calibrated return, variance, and activity targets back into the candle generator with rank-preserving ensemble coupling. Range and maximum-move calibration require an additional conditional intraperiod shape/extreme-move layer. After that change, rerun the full candle-level audit rather than assuming independently calibrated summaries produce a calibrated joint path.

## Artifacts

- Raw rolling evaluator: `ml/evaluate_rolling_refit_next_day_process.py`
- Calibration evaluator: `ml/calibrate_rolling_forecasts.py`
- Calibration tests: `ml/test_calibrate_rolling_forecasts.py`
- Forecast archive: `data/benchmarks/rolling-refit-next-day-process-forecasts.npz`
- Full calibration report: `data/benchmarks/calibrated-rolling-forecasts.json`
- Commands: `npm run analysis:rolling-refit-next-day` and `npm run analysis:calibrate-rolling-forecasts`

## Hierarchical follow-up

The next experiment reconciles the calibrated 1d, 8h, and 4h forecasts into exact temporal hierarchies. Results and the parent-preserving child-allocation model are documented in `docs/experiments/hierarchical-temporal-conditioning-2026-08-13.md`.
