# Extrema Signal Timing Benchmark: anticipatory-grid-5m

Generated: 2026-07-07 09:54
Raw results: `data/benchmarks/extrema-signal-anticipatory-grid-5m-2026-07-07.jsonl`

## Scope

- Market: BTCUSDT 1m spot candles, UTC day-inclusive intervals from `tasks.md`.
- Strategy: static sigma `buySigma=0.1`, `sellSigma=0.1`, mode both, futures-margin shorts, borrow depths `999/999`, max leverage `1x`.
- Derivative source: `price`; derivative clamp mode: `deadband`.
- Anticipatory entry/exit grid window: `300s`.
- Moving average type(s): `sma`. EMA uses a continuous-time alpha with `tau = window / 2`, matching the SMA window's average sample age.

## Summary

| Average | Avg return | Median return | Positive | Avg DD | Max DD | Avg trades |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SMA | -0.806% | -0.201% | 10/28 | 1.255% | 8.298% | 643.9 |

## Intervals

| Average | Group | Case | Interval | Market | Return | Max DD | Trades | Win rate | Stop |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| SMA | choppy-week | highest OHLC churn | `2022-07-28..2022-08-03` | -0.592% | -1.448% | 1.616% | 964 | 100.000% | completed |
| SMA | choppy-week | highest close churn | `2022-05-14..2022-05-20` | -0.294% | +0.098% | 1.292% | 1,785 | 96.980% | completed |
| SMA | choppy-week | very choppy 2021-12 | `2021-12-14..2021-12-20` | +0.453% | +0.059% | 0.543% | 1,501 | 99.830% | completed |
| SMA | choppy-week | very choppy 2021-09 | `2021-09-08..2021-09-14` | +0.518% | +1.694% | 1.176% | 1,389 | 99.550% | completed |
| SMA | choppy-week | near-flat close | `2023-03-18..2023-03-24` | +0.217% | +0.312% | 0.713% | 1,124 | 99.190% | completed |
| SMA | regime-week | uptrend | `2023-03-11..2023-03-17` | +35.951% | -1.344% | 1.405% | 1,000 | 99.680% | completed |
| SMA | regime-week | sideways | `2026-04-22..2026-04-28` | +0.009% | +0.253% | 0.256% | 180 | 100.000% | completed |
| SMA | regime-week | downtrend | `2022-06-12..2022-06-18` | -33.260% | -5.436% | 6.365% | 2,224 | 98.750% | completed |
| SMA | 3d-regime | uptrend low churn | `2024-02-24..2024-02-26` | +7.355% | -0.308% | 0.308% | 24 | 0.000% | completed |
| SMA | 3d-regime | uptrend high churn | `2022-06-19..2022-06-21` | +9.239% | -1.542% | 1.584% | 790 | 97.960% | completed |
| SMA | 3d-regime | downtrend low churn | `2023-06-03..2023-06-05` | -5.559% | +0.046% | 0.039% | 46 | 100.000% | completed |
| SMA | 3d-regime | downtrend high churn | `2022-06-13..2022-06-15` | -15.017% | -0.180% | 1.645% | 1,091 | 97.470% | completed |
| SMA | 3d-regime | sideways high bias churn | `2021-10-19..2021-10-21` | +0.302% | -0.530% | 0.530% | 415 | 99.640% | completed |
| SMA | 3d-regime | sideways high bias low churn | `2025-02-14..2025-02-16` | -0.507% | +0.066% | 0.104% | 43 | 100.000% | completed |
| SMA | 3d-regime | sideways low bias churn | `2024-07-07..2024-07-09` | -0.309% | +0.431% | 0.193% | 439 | 100.000% | completed |
| SMA | 3d-regime | sideways low bias low churn | `2025-07-04..2025-07-06` | -0.348% | -0.043% | 0.043% | 5 | 0.000% | completed |
| SMA | 3d-regime | sideways mid bias churn | `2024-01-02..2024-01-04` | -0.064% | +0.835% | 0.259% | 393 | 100.000% | completed |
| SMA | 3d-regime | sideways mid bias low churn | `2023-09-15..2023-09-17` | +0.018% | -0.043% | 0.052% | 5 | 0.000% | completed |
| SMA | trend-sharpe | 3d up 2024-11 | `2024-11-09..2024-11-11` | +15.865% | -0.320% | 0.320% | 278 | 100.000% | completed |
| SMA | trend-sharpe | 3d up 2023-12 | `2023-12-03..2023-12-05` | +11.719% | -0.462% | 0.492% | 223 | 100.000% | completed |
| SMA | trend-sharpe | 3d down 2026-06 | `2026-06-01..2026-06-03` | -12.938% | +0.313% | 0.257% | 376 | 100.000% | completed |
| SMA | trend-sharpe | 3d down 2023-03 | `2023-03-07..2023-03-09` | -9.135% | -0.768% | 0.768% | 229 | 100.000% | completed |
| SMA | trend-sharpe | 7d up 2023-12 | `2023-11-29..2023-12-05` | +16.538% | -0.131% | 0.454% | 304 | 100.000% | completed |
| SMA | trend-sharpe | 7d up 2024-11 | `2024-11-05..2024-11-11` | +30.653% | -1.060% | 1.060% | 528 | 100.000% | completed |
| SMA | trend-sharpe | 7d down 2023-03 | `2023-03-03..2023-03-09` | -13.224% | -1.921% | 1.921% | 318 | 100.000% | completed |
| SMA | trend-sharpe | 7d down 2026-06 | `2026-05-27..2026-06-02` | -12.076% | -0.222% | 0.400% | 389 | 100.000% | completed |
| SMA | stress | 3d down 2022-06 | `2022-06-11..2022-06-13` | -22.702% | -2.621% | 3.042% | 762 | 97.580% | completed |
| SMA | stress | 7d down 2022-06 | `2022-06-07..2022-06-13` | -28.323% | -8.298% | 8.298% | 1,205 | 98.490% | completed |

