# Extrema Signal Timing Benchmark: base-start-signals

Generated: 2026-07-07 07:19
Raw results: `data/benchmarks/extrema-signal-base-start-2026-07-07.jsonl`

## Scope

- Market: BTCUSDT 1m spot candles, UTC day-inclusive intervals from `tasks.md`.
- Strategy: static sigma `buySigma=0.1`, `sellSigma=0.1`, mode both, futures-margin shorts, borrow depths `999/999`, max leverage `1x`.
- Derivative source: `price`; derivative clamp mode: `deadband`.

## Summary

| Avg return | Median return | Positive | Avg DD | Max DD | Avg trades |
| ---: | ---: | ---: | ---: | ---: | ---: |
| -2.647% | -2.084% | 6/28 | 4.720% | 15.692% | 747.6 |

## Intervals

| Group | Case | Interval | Market | Return | Max DD | Trades | Win rate | Stop |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| choppy-week | highest OHLC churn | `2022-07-28..2022-08-03` | -0.592% | -5.442% | 7.923% | 1,398 | 95.290% | completed |
| choppy-week | highest close churn | `2022-05-14..2022-05-20` | -0.294% | -3.189% | 6.433% | 2,120 | 95.610% | completed |
| choppy-week | very choppy 2021-12 | `2021-12-14..2021-12-20` | +0.453% | -7.448% | 7.448% | 1,426 | 94.020% | completed |
| choppy-week | very choppy 2021-09 | `2021-09-08..2021-09-14` | +0.518% | -2.506% | 6.348% | 1,246 | 93.630% | completed |
| choppy-week | near-flat close | `2023-03-18..2023-03-24` | +0.217% | +2.371% | 3.132% | 1,455 | 94.010% | completed |
| regime-week | uptrend | `2023-03-11..2023-03-17` | +35.951% | -8.054% | 8.815% | 1,542 | 98.260% | completed |
| regime-week | sideways | `2026-04-22..2026-04-28` | +0.009% | +2.597% | 0.721% | 401 | 92.330% | completed |
| regime-week | downtrend | `2022-06-12..2022-06-18` | -33.260% | -4.893% | 11.874% | 2,368 | 97.690% | completed |
| 3d-regime | uptrend low churn | `2024-02-24..2024-02-26` | +7.355% | -0.308% | 0.308% | 24 | 0.000% | completed |
| 3d-regime | uptrend high churn | `2022-06-19..2022-06-21` | +9.239% | -4.111% | 4.317% | 1,182 | 98.510% | completed |
| 3d-regime | downtrend low churn | `2023-06-03..2023-06-05` | -5.559% | -4.905% | 5.773% | 57 | 100.000% | completed |
| 3d-regime | downtrend high churn | `2022-06-13..2022-06-15` | -15.017% | +10.090% | 5.111% | 1,204 | 98.260% | completed |
| 3d-regime | sideways high bias churn | `2021-10-19..2021-10-21` | +0.302% | -0.378% | 1.937% | 558 | 99.290% | completed |
| 3d-regime | sideways high bias low churn | `2025-02-14..2025-02-16` | -0.507% | -1.296% | 1.387% | 69 | 100.000% | completed |
| 3d-regime | sideways low bias churn | `2024-07-07..2024-07-09` | -0.309% | -0.883% | 1.438% | 440 | 100.000% | completed |
| 3d-regime | sideways low bias low churn | `2025-07-04..2025-07-06` | -0.348% | -0.043% | 0.043% | 6 | 0.000% | completed |
| 3d-regime | sideways mid bias churn | `2024-01-02..2024-01-04` | -0.064% | +1.719% | 3.069% | 223 | 100.000% | completed |
| 3d-regime | sideways mid bias low churn | `2023-09-15..2023-09-17` | +0.018% | -0.043% | 0.052% | 5 | 0.000% | completed |
| trend-sharpe | 3d up 2024-11 | `2024-11-09..2024-11-11` | +15.865% | +0.254% | 0.189% | 475 | 100.000% | completed |
| trend-sharpe | 3d up 2023-12 | `2023-12-03..2023-12-05` | +11.719% | -2.092% | 2.121% | 145 | 100.000% | completed |
| trend-sharpe | 3d down 2026-06 | `2026-06-01..2026-06-03` | -12.938% | +0.796% | 0.477% | 375 | 100.000% | completed |
| trend-sharpe | 3d down 2023-03 | `2023-03-07..2023-03-09` | -9.135% | -4.704% | 6.359% | 214 | 100.000% | completed |
| trend-sharpe | 7d up 2023-12 | `2023-11-29..2023-12-05` | +16.538% | -1.843% | 2.097% | 239 | 100.000% | completed |
| trend-sharpe | 7d up 2024-11 | `2024-11-05..2024-11-11` | +30.653% | -2.077% | 2.836% | 907 | 100.000% | completed |
| trend-sharpe | 7d down 2023-03 | `2023-03-03..2023-03-09` | -13.224% | -4.309% | 5.986% | 293 | 100.000% | completed |
| trend-sharpe | 7d down 2026-06 | `2026-05-27..2026-06-02` | -12.076% | -4.904% | 4.981% | 444 | 100.000% | completed |
| stress | 3d down 2022-06 | `2022-06-11..2022-06-13` | -22.702% | -13.247% | 15.692% | 844 | 98.250% | completed |
| stress | 7d down 2022-06 | `2022-06-07..2022-06-13` | -28.323% | -15.280% | 15.280% | 1,273 | 99.890% | completed |

