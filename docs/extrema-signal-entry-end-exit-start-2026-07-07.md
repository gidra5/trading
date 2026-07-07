# Extrema Signal Timing Benchmark: entry-end-exit-start

Generated: 2026-07-07 07:25
Raw results: `data/benchmarks/extrema-signal-entry-end-exit-start-2026-07-07.jsonl`

## Scope

- Market: BTCUSDT 1m spot candles, UTC day-inclusive intervals from `tasks.md`.
- Strategy: static sigma `buySigma=0.1`, `sellSigma=0.1`, mode both, futures-margin shorts, borrow depths `999/999`, max leverage `1x`.
- Derivative source: `price`; derivative clamp mode: `deadband`.

## Summary

| Avg return | Median return | Positive | Avg DD | Max DD | Avg trades |
| ---: | ---: | ---: | ---: | ---: | ---: |
| -1.730% | -0.048% | 13/28 | 3.833% | 21.389% | 651.8 |

## Intervals

| Group | Case | Interval | Market | Return | Max DD | Trades | Win rate | Stop |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| choppy-week | highest OHLC churn | `2022-07-28..2022-08-03` | -0.592% | -6.438% | 7.545% | 1,241 | 85.280% | completed |
| choppy-week | highest close churn | `2022-05-14..2022-05-20` | -0.294% | +0.028% | 3.706% | 1,582 | 99.100% | completed |
| choppy-week | very choppy 2021-12 | `2021-12-14..2021-12-20` | +0.453% | -5.074% | 5.461% | 1,044 | 94.670% | completed |
| choppy-week | very choppy 2021-09 | `2021-09-08..2021-09-14` | +0.518% | -1.629% | 5.435% | 1,108 | 96.340% | completed |
| choppy-week | near-flat close | `2023-03-18..2023-03-24` | +0.217% | +3.413% | 3.035% | 1,210 | 83.200% | completed |
| regime-week | uptrend | `2023-03-11..2023-03-17` | +35.951% | -21.333% | 21.389% | 1,027 | 97.230% | completed |
| regime-week | sideways | `2026-04-22..2026-04-28` | +0.009% | +0.419% | 1.447% | 248 | 94.000% | completed |
| regime-week | downtrend | `2022-06-12..2022-06-18` | -33.260% | -3.779% | 9.453% | 2,140 | 94.170% | completed |
| 3d-regime | uptrend low churn | `2024-02-24..2024-02-26` | +7.355% | +0.413% | 0.070% | 22 | 0.000% | completed |
| 3d-regime | uptrend high churn | `2022-06-19..2022-06-21` | +9.239% | -1.791% | 2.115% | 1,018 | 96.900% | completed |
| 3d-regime | downtrend low churn | `2023-06-03..2023-06-05` | -5.559% | +0.067% | 0.070% | 56 | 100.000% | completed |
| 3d-regime | downtrend high churn | `2022-06-13..2022-06-15` | -15.017% | +6.970% | 6.458% | 1,062 | 98.140% | completed |
| 3d-regime | sideways high bias churn | `2021-10-19..2021-10-21` | +0.302% | -0.532% | 1.899% | 509 | 95.420% | completed |
| 3d-regime | sideways high bias low churn | `2025-02-14..2025-02-16` | -0.507% | +0.022% | 0.115% | 64 | 100.000% | completed |
| 3d-regime | sideways low bias churn | `2024-07-07..2024-07-09` | -0.309% | +0.470% | 2.922% | 319 | 83.820% | completed |
| 3d-regime | sideways low bias low churn | `2025-07-04..2025-07-06` | -0.348% | -0.084% | 0.084% | 6 | 0.000% | completed |
| 3d-regime | sideways mid bias churn | `2024-01-02..2024-01-04` | -0.064% | -0.272% | 1.437% | 258 | 100.000% | completed |
| 3d-regime | sideways mid bias low churn | `2023-09-15..2023-09-17` | +0.018% | -0.011% | 0.013% | 4 | 0.000% | completed |
| trend-sharpe | 3d up 2024-11 | `2024-11-09..2024-11-11` | +15.865% | +1.122% | 0.072% | 463 | 100.000% | completed |
| trend-sharpe | 3d up 2023-12 | `2023-12-03..2023-12-05` | +11.719% | -0.965% | 1.706% | 316 | 100.000% | completed |
| trend-sharpe | 3d down 2026-06 | `2026-06-01..2026-06-03` | -12.938% | +0.887% | 0.540% | 301 | 100.000% | completed |
| trend-sharpe | 3d down 2023-03 | `2023-03-07..2023-03-09` | -9.135% | +0.251% | 1.220% | 219 | 100.000% | completed |
| trend-sharpe | 7d up 2023-12 | `2023-11-29..2023-12-05` | +16.538% | -1.393% | 1.723% | 390 | 100.000% | completed |
| trend-sharpe | 7d up 2024-11 | `2024-11-05..2024-11-11` | +30.653% | +0.887% | 0.543% | 1,095 | 100.000% | completed |
| trend-sharpe | 7d down 2023-03 | `2023-03-03..2023-03-09` | -13.224% | +0.716% | 1.438% | 306 | 100.000% | completed |
| trend-sharpe | 7d down 2026-06 | `2026-05-27..2026-06-02` | -12.076% | -4.486% | 4.569% | 421 | 100.000% | completed |
| stress | 3d down 2022-06 | `2022-06-11..2022-06-13` | -22.702% | -0.087% | 3.381% | 676 | 97.230% | completed |
| stress | 7d down 2022-06 | `2022-06-07..2022-06-13` | -28.323% | -16.244% | 19.472% | 1,145 | 98.330% | completed |

