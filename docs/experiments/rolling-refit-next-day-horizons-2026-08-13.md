# Rolling local-statistics next-candle simulation across horizons

Date: 2026-08-13  
Instrument: Binance spot BTCUSDT  
Holdout: 2025-07-25 through 2026-07-24  
Rolling origins: 365

## Question

Does this procedure generate future candles whose statistics match the realized future period?

1. At a forecast boundary, take exactly the latest $X$ historical 1s candles.
2. Re-estimate the time-varying process statistics without looking beyond that boundary.
3. Generate the next $H$ one-second candles.
4. Compute the same statistics on the generated and realized periods.
5. Repeat over many forecast boundaries, history windows, and prediction horizons.

History windows are 7, 30, 90, 365, and 730 days. Prediction horizons are 15m, 30m, 1h, 2h, 4h, 8h, and 1d.

## Implementation

The stable intraminute return-shape parameters—nonzero magnitude mixtures, the micro-return layer, activity timing, and minute-level copula structure—are fitted only on the pre-holdout period from 2021-07-25 through 2025-07-24.

Before every target period, three time-varying targets are re-estimated from historical blocks having the same duration as the prediction horizon:

- integrated 1s realized variance;
- signed return divided by square-root realized variance;
- active-second fraction, conditionally adjusted for variance.

Each target uses a shrinkage AR(1) location model with a fitted Student-t innovation. Thus a 15m forecast trained on seven days sees 672 historical 15m blocks; a 1d forecast sees seven daily blocks. No historical return block is resampled.

Each history-window/horizon/origin case gets 16 generated paths. One path is fully materialized at 1s resolution. Across the audit this creates 12,775 complete second-path cases. Common random numbers are shared across history windows at each origin and horizon so their comparison is less affected by Monte Carlo luck.

Shorter forecasts are generated independently. They are not prefixes of a path constrained to a 1d endpoint.

## Main result

There is no universally best history length. The useful adaptation window increases with prediction horizon.

| Horizon | Best overall window | Best return window | Best variance/activity window | Best pooled 1s-shape window |
| --- | ---: | ---: | ---: | ---: |
| 15m | 7d | 7d, but differences are negligible | 7d | 730d |
| 30m | 7d | 30d, but differences are negligible | 7d | 730d |
| 1h | 730d | 365d | variance 730d; activity 7d | 730d |
| 2h | 30d | 730d | 30d | 730d |
| 4h | 30–90d | 90d | 30d | 30d |
| 8h | 90d | 90d | 90d | 90d |
| 1d | 730d | 365d | variance 730d; activity 365d | 90d |

“Best overall” is the lowest mean normalized CRPS across period return, realized variance, active fraction, path range, and maximum minute move. It is a relative ranking, not a claim that the resulting forecast is calibrated.

The CRPS advantage between history windows is small at short horizons and grows at longer horizons:

| Horizon | Best versus worst aggregate CRPS behavior |
| --- | --- |
| 15m–30m | Usually only 0–5% depending on statistic |
| 1h | Mostly 1–3% |
| 2h | Up to 4.6% for variance |
| 4h | Up to 8.4% for variance and 5.2% for activity |
| 8h | About 6–7% for variance/activity |
| 1d | About 5% for return/variance and 9% for activity |

This is consistent with an effective-sample-size tradeoff: short forecast blocks provide many recent examples, whereas day-scale targets need hundreds of days to estimate their tails and persistence.

## Best-window statistics by horizon

The following rows use the best window for the named target, not necessarily one common window for all columns.

| Horizon | Return CRPS | Variance CRPS | Activity CRPS | Path-range CRPS | Max-minute CRPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 15m | 14.00 bp (7d) | 240 bp² (7d) | 0.0400 (7d) | 9.89 bp (730d) | 3.42 bp (7d) |
| 30m | 17.14 bp (30d) | 490 bp² (7d) | 0.0367 (7d) | 13.54 bp (730d) | 4.02 bp (7d) |
| 1h | 22.82 bp (365d) | 712 bp² (730d) | 0.0327 (7d) | 17.06 bp (730d) | 4.15 bp (730d) |
| 2h | 33.87 bp (730d) | 1,429 bp² (30d) | 0.0354 (30d) | 27.12 bp (30d) | 6.05 bp (30d) |
| 4h | 47.61 bp (90d) | 2,294 bp² (30d) | 0.0343 (30d) | 36.82 bp (90d) | 6.87 bp (30d) |
| 8h | 64.05 bp (90d) | 4,444 bp² (90d) | 0.0384 (90d) | 50.19 bp (30d) | 8.25 bp (90d) |
| 1d | 122.81 bp (365d) | 16,617 bp² (730d) | 0.0334 (365d) | 95.41 bp (730d) | 16.85 bp (730d) |

The return-window differences at 15m and 30m are below 0.5%, so their nominal winners should not be interpreted as meaningful directional predictability.

## Pooled candle-distribution shape

One fully materialized path per origin/window/horizon is pooled to compare its return histogram with the corresponding realized prefixes.

| Horizon | Lowest 1s JS | Window | Lowest 1m JS | Window |
| --- | ---: | ---: | ---: | ---: |
| 15m | 0.00561 | 730d | 0.04610 | 7d |
| 30m | 0.00398 | 730d | 0.03838 | 7d |
| 1h | 0.00227 | 730d | 0.02381 | 7d |
| 2h | 0.00262 | 730d | 0.02172 | 7d |
| 4h | 0.00151 | 30d | 0.01319 | 7d |
| 8h | 0.00171 | 90d | 0.01210 | 90d |
| 1d | 0.00195 | 90d | 0.00774 | 730d |

The 1s marginal extrapolates much better than the 1m marginal. This means matching local variance, activity, and second-level magnitude shape still does not fully reproduce how signed second returns combine into future minute returns.

## Calibration verdict

Most forecasts are not probabilistically calibrated, even when they rank best among the tested windows.

Examples:

- The best 15m variance model generates mean variance of 255 bp² versus 472 bp² realized and its nominal 90% interval covers only 74.2%.
- At 1h, the 730d return model has acceptable PIT rank uniformity, but variance, activity, range, and maximum-minute PIT tests reject calibration.
- The 4h 30–90d models improve CRPS, but their central intervals still under-cover path range and maximum minute movement.
- At 1d, the 730d variance model is the only best-target variance model whose PIT test barely passes at 5% ($p=0.0514$). Its nominal 90% interval covers 84.9%.
- The best daily maximum-minute model underpredicts the median maximum by about 16.4 bp.

Therefore the experiment does **not** show that recomputing statistics from a rolling history window makes the entire next period's candle distribution reliable. It shows which parameters benefit from recent data and which need long samples.

## Comparison with the fixed-parameter audit

For 1d forecasts:

| Method | Return CRPS | Variance CRPS | Activity CRPS | Range CRPS | 1s JS | 1m JS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed four-year shape/state | 119.95 | **15,202** | 0.0795 | **91.43** | 0.00563 | 0.00763 |
| Fixed one-year shape/state | **119.23** | 15,631 | 0.0428 | 92.65 | 0.00233 | **0.00391** |
| Rolling local targets, best per metric | 122.81 | 16,617 | **0.0334** | 95.41 | **0.00195** | 0.00774 |

Rolling local refitting substantially improves activity and pooled 1s shape, but it worsens daily return, variance, path range, and 1m shape compared with the best fixed/state-conditioned model. Blindly refitting every parameter from recent data is therefore not the answer.

## Recommended process

Use separate estimation horizons for separate model layers:

- Stable nonzero 1s magnitude/tail shape: 1–2 years.
- Activity level and short-horizon variance:
  - 7d for 15–60m forecasts;
  - 30d for 2–4h;
  - 90d for 8h;
  - 365d for daily activity.
- Daily integrated variance and path-scale tails: 730d, with the continuously filtered current volatility state retained.
- Signed return direction: use long windows; shorter refits do not produce a meaningful gain and no horizon shows strong directional predictability.

The next implementation step should combine these layer-specific windows rather than choosing one common $X$. It should also introduce a conditional extreme-minute layer, since maximum moves remain systematically too small.

## Limitations

- There are 365 independent forecast origins. Each origin has 16 generated paths, enough for comparative CRPS but coarse for extreme quantile calibration.
- Only the first period of each UTC day is scored at each horizon. A full intraday rolling-origin audit would test whether the optimal windows depend on time of day.
- Stable intraminute shape parameters are not re-estimated at every origin. Only the time-varying variance, signed-efficiency, and activity targets use exactly the latest $X$ days. This isolates adaptation without trying to infer rare-tail mixtures from seven days.
- Results are for one final-year BTCUSDT holdout and require confirmation on additional chronological periods or instruments.

## Artifacts

- Evaluator: `ml/evaluate_rolling_refit_next_day_process.py`
- Tests: `ml/test_evaluate_rolling_refit_next_day_process.py`
- Full machine-readable report: `data/benchmarks/rolling-refit-next-day-process.json`
- Commands: `npm run analysis:rolling-refit-next-day` and `npm run analysis:rolling-refit-next-day:test`

## Follow-up calibration audit

A chronological calibration/test split and a live-style 90-day rolling correction are evaluated in `docs/experiments/rolling-forecast-calibration-2026-08-13.md`. The rolling correction materially improves forecast ranks and interval errors, but it does not yet make every path statistic or their joint distribution calibrated.
