# Parametric process next-day holdout audit

Date: 2026-08-13  
Instrument: Binance spot BTCUSDT  
Target: determine whether parameters fitted only on historical data can generate calibrated samples of the next day's candles

## Verdict

The fitted process contains useful out-of-sample information, but it is **not yet reliable as a complete next-day candle sampler**.

It already works reasonably well as a scenario generator for broad return marginals. A recent one-year fit reaches out-of-sample JS divergence of 0.00233 at 1s and 0.00391 at 1m. State conditioning materially improves next-day integrated-variance forecasts. However, PIT and interval-coverage tests reject calibration for daily variance and most path features, especially the fraction of zero-return seconds and the maximum minute move.

The right architecture is therefore hybrid:

- estimate stable nonzero-return shape and tail parameters from a long window;
- update volatility state from recent observations;
- give daily activity/zero probability its own dynamic target instead of assuming a stationary all-history marginal;
- recalibrate path extremes on rolling holdouts.

The generated paths can currently be called plausible scenarios. Their quoted probabilities cannot yet be trusted as calibrated next-day probabilities.

## Leakage-free design

The untouched holdout is 2025-07-25 through 2026-07-24: 365 sequential next-day outcomes.

Two fits are compared against the same holdout:

| Fit | Training window | Training days | Holdout days |
| --- | --- | ---: | ---: |
| Long | 2021-07-25 through 2025-07-24 | 1,461 | 365 |
| Recent | 2024-07-25 through 2025-07-24 | 365 | 365 |

For each target day:

1. Parameters are fixed using only the stated training window.
2. The posterior state of each fitted AR-factor process contains observations only through the preceding day.
3. 2,048 daily-return and daily-variance samples are drawn.
4. Sixty-four complete minute paths are drawn conditionally on the latest volatility/activity state.
5. One full 86,400-return second path is materialized.
6. The realized day is scored, and only then is it allowed to update the state for the following forecast.

No historical candle, minute template, day, sign mask, or magnitude vector is resampled.

### Proper forecast scores

The audit uses:

- PIT/rank uniformity;
- empirical coverage of central 50%, 80%, 90%, and 95% intervals;
- continuous ranked probability score (CRPS);
- kernel negative log density;
- JS divergence for pooled multiscale marginals;
- calibration of daily path range, minute-return quadratic variation, zero probability, and maximum absolute minute return.

A fit is not considered calibrated merely because its mean or variance is close.

## Daily one-step forecasts

### Daily signed return

| Fit | PIT KS p-value | Maximum coverage error | Mean CRPS | CRPS skill vs Gaussian | Benefit from state conditioning |
| --- | ---: | ---: | ---: | ---: | ---: |
| Four years | 0.0155 | 8.49 points | 119.95 bp | +3.17% | +0.06% |
| One year | **0.1719** | 6.85 points | **119.23 bp** | +1.93% | +0.33% |

The one-year marginal passes the PIT uniformity test, but its 80% interval covers 86.85%, outside the five-point tolerance. Directional state contributes essentially nothing: daily signed returns remain close to white noise.

This means the process can sample a reasonable next-day close distribution, but it does not predict which close will occur. The samples are uncertainty scenarios, not directional forecasts.

### Daily integrated variance

| Fit | PIT KS p-value | Maximum coverage error | Mean CRPS | CRPS skill vs lognormal | Benefit from state conditioning |
| --- | ---: | ---: | ---: | ---: | ---: |
| Four years | 0.00615 | 11.92 points | **15,202 bp²** | **+36.37%** | **+36.85%** |
| One year | 0.00993 | 10.27 points | 15,631 bp² | +22.64% | +22.64% |

Volatility state is genuinely forecastable: conditioning cuts the four-year model's CRPS by 36.85% relative to starting from the stationary distribution. The 90% and 95% intervals from that model cover 89.86% and 95.07%, almost exactly their targets.

The central distribution is still wrong. Its 50% interval covers 61.92%, and PIT ranks are too concentrated. The model is too wide around the center while its kernel log score is worse than the simpler lognormal baseline. This is useful but not fully calibrated volatility forecasting.

## Out-of-sample candle marginals

JS divergence compares all realized holdout returns with generated returns on the same fixed bins.

| Scale | Four-year fit JS | One-year fit JS | Four-year variance ratio | One-year variance ratio |
| --- | ---: | ---: | ---: | ---: |
| 1s | 0.00563 | **0.00233** | 0.921 | 0.870 |
| 1m | 0.00763 | **0.00391** | 1.131 | 0.905 |
| 15m | 0.00413 | **0.00372** | 1.086 | 0.880 |
| 1h | **0.00406** | 0.00520 | 0.976 | 0.823 |
| 4h | **0.01114** | 0.01220 | 0.845 | 0.779 |

Variance ratios are observed/model; one is ideal.

Recency clearly improves 1s through 15m shape, but the one-year fit generates too much variance and too much tail mass from 1m upward. The longer fit is more stable at 1h. At 4h the holdout has only 2,190 observations: median one-sample JS floors are about 0.0091 and 0.0096, so much of the measured 4h JS is finite-sample noise.

The 1s JS values are far above their roughly $4.5\times10^{-6}$ sampling floors. The remaining difference is genuine distribution shift rather than histogram noise.

## Path-feature calibration

| Feature | Four-year PIT p | One-year PIT p | Four-year max coverage error | One-year max coverage error |
| --- | ---: | ---: | ---: | ---: |
| Minute-path range | 0.0484 | 0.00754 | **3.77 points** | **3.77 points** |
| Minute-return quadratic variation | $6.9\times10^{-15}$ | $2.0\times10^{-6}$ | 3.70 points | 8.63 points |
| Zero-second probability | $8.1\times10^{-289}$ | $5.9\times10^{-136}$ | 81.30 points | 46.85 points |
| Maximum absolute minute return | $1.2\times10^{-18}$ | $3.0\times10^{-7}$ | 14.11 points | 8.90 points |

The daily range is close enough to be useful, although PIT still detects shape error. The two decisive failures are activity and extremes:

- The one-year fit overpredicts zero probability by 5.51 percentage points at its median; the four-year fit overpredicts it by 9.36 points.
- Both fits underpredict the maximum minute move, by 11.76 bp and 15.67 bp at the median.

Annual exact-zero fractions demonstrate why a stationary activity marginal is fragile:

| Year beginning | Exact-zero fraction |
| --- | ---: |
| 2021-07-25 | 0.2759 |
| 2022-07-25 | 0.1754 |
| 2023-07-25 | 0.4485 |
| 2024-07-25 | 0.4241 |
| Holdout: 2025-07-25 | 0.4621 |

This is structural parameter drift, not just short-memory autocorrelation. Filtering a stationary activity factor cannot fully repair a drifting marginal and in the four-year fit overreacts to the inferred low-activity state.

## What can be trusted now

The current process can support:

- Monte Carlo scenarios with broadly realistic 1s through 1h return shapes;
- next-day volatility ranking and wide-tail risk intervals;
- approximate daily range scenarios;
- stress testing where exact probability calibration is not required.

It should not yet be used for:

- probability claims about how many seconds will change price;
- calibrated likelihoods of extreme minute candles;
- sizing positions from nominal predictive quantiles without an extra safety margin;
- claims that an individual future candle direction is predictable.

## Next model refinement

The most direct improvement is a daily activity-budget layer analogous to the existing daily variance budget:

1. Model the daily active fraction on the logit scale with a slowly drifting level plus a filtered short-memory state.
2. Generate the next day's total active-second budget from that predictive distribution.
3. Condition minute activity counts on volatility, but normalize or project them so the daily total matches the generated activity budget.
4. Fit stable intraminute timing and nonzero magnitude shapes on several years, while updating only the activity level from a recent 30- to 365-day window.

After that, add a conditional extreme-move layer for daily maximum minute magnitude and couple the daily return target to the daily variance budget. Validate with multiple rolling origins rather than this single final-year holdout, reserving a last untouched period for final calibration confirmation.

## Artifacts

- Evaluator: `ml/evaluate_parametric_next_day_process.py`
- Tests: `ml/test_evaluate_parametric_next_day_process.py`
- Four-year fit report: `data/benchmarks/parametric-next-day-holdout.json`
- One-year fit report: `data/benchmarks/parametric-next-day-holdout-365d-fit.json`
- Commands: `npm run analysis:parametric-next-day` and `npm run analysis:parametric-next-day:test`
