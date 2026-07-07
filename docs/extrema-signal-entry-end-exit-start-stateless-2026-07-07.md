# Extrema Signal Timing Benchmark: entry-end-exit-start-stateless

Generated: 2026-07-07 07:43
Raw results: `data/benchmarks/extrema-signal-entry-end-exit-start-stateless-2026-07-07.jsonl`

## Scope

- Market: BTCUSDT 1m spot candles, UTC day-inclusive intervals from `tasks.md`.
- Strategy: static sigma `buySigma=0.1`, `sellSigma=0.1`, mode both, futures-margin shorts, borrow depths `999/999`, max leverage `1x`.
- Derivative source: `price`; derivative clamp mode: `deadband`.

## Summary

| Avg return | Median return | Positive | Avg DD | Max DD | Avg trades |
| ---: | ---: | ---: | ---: | ---: | ---: |
| -4.817% | -2.046% | 8/28 | 6.685% | 29.557% | 800 |

## Intervals

| Group | Case | Interval | Market | Return | Max DD | Trades | Win rate | Stop |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| choppy-week | highest OHLC churn | `2022-07-28..2022-08-03` | -0.592% | -13.450% | 13.450% | 1,767 | 90.710% | completed |
| choppy-week | highest close churn | `2022-05-14..2022-05-20` | -0.294% | -1.578% | 2.986% | 1,841 | 98.140% | completed |
| choppy-week | very choppy 2021-12 | `2021-12-14..2021-12-20` | +0.453% | -5.716% | 8.568% | 1,786 | 97.440% | completed |
| choppy-week | very choppy 2021-09 | `2021-09-08..2021-09-14` | +0.518% | -6.402% | 7.760% | 1,392 | 94.430% | completed |
| choppy-week | near-flat close | `2023-03-18..2023-03-24` | +0.217% | +3.161% | 2.799% | 1,687 | 87.150% | completed |
| regime-week | uptrend | `2023-03-11..2023-03-17` | +35.951% | -16.281% | 16.281% | 1,266 | 98.300% | completed |
| regime-week | sideways | `2026-04-22..2026-04-28` | +0.009% | -1.588% | 2.044% | 253 | 96.220% | completed |
| regime-week | downtrend | `2022-06-12..2022-06-18` | -33.260% | -23.921% | 24.415% | 2,016 | 98.160% | completed |
| 3d-regime | uptrend low churn | `2024-02-24..2024-02-26` | +7.355% | +4.201% | 0.367% | 34 | 0.000% | completed |
| 3d-regime | uptrend high churn | `2022-06-19..2022-06-21` | +9.239% | -3.513% | 3.513% | 1,185 | 95.590% | completed |
| 3d-regime | downtrend low churn | `2023-06-03..2023-06-05` | -5.559% | +0.063% | 0.137% | 68 | 100.000% | completed |
| 3d-regime | downtrend high churn | `2022-06-13..2022-06-15` | -15.017% | +7.232% | 6.278% | 1,142 | 93.930% | completed |
| 3d-regime | sideways high bias churn | `2021-10-19..2021-10-21` | +0.302% | -2.021% | 3.224% | 654 | 100.000% | completed |
| 3d-regime | sideways high bias low churn | `2025-02-14..2025-02-16` | -0.507% | +0.003% | 0.074% | 64 | 100.000% | completed |
| 3d-regime | sideways low bias churn | `2024-07-07..2024-07-09` | -0.309% | +1.087% | 2.927% | 426 | 91.080% | completed |
| 3d-regime | sideways low bias low churn | `2025-07-04..2025-07-06` | -0.348% | -0.100% | 0.100% | 9 | 0.000% | completed |
| 3d-regime | sideways mid bias churn | `2024-01-02..2024-01-04` | -0.064% | -6.042% | 6.423% | 398 | 100.000% | completed |
| 3d-regime | sideways mid bias low churn | `2023-09-15..2023-09-17` | +0.018% | -0.031% | 0.032% | 6 | 0.000% | completed |
| trend-sharpe | 3d up 2024-11 | `2024-11-09..2024-11-11` | +15.865% | +1.023% | 0.270% | 588 | 100.000% | completed |
| trend-sharpe | 3d up 2023-12 | `2023-12-03..2023-12-05` | +11.719% | -1.724% | 2.828% | 416 | 100.000% | completed |
| trend-sharpe | 3d down 2026-06 | `2026-06-01..2026-06-03` | -12.938% | -2.072% | 4.736% | 450 | 100.000% | completed |
| trend-sharpe | 3d down 2023-03 | `2023-03-07..2023-03-09` | -9.135% | -5.462% | 6.465% | 92 | 100.000% | completed |
| trend-sharpe | 7d up 2023-12 | `2023-11-29..2023-12-05` | +16.538% | -8.232% | 8.371% | 534 | 100.000% | completed |
| trend-sharpe | 7d up 2024-11 | `2024-11-05..2024-11-11` | +30.653% | +2.576% | 1.087% | 1,219 | 100.000% | completed |
| trend-sharpe | 7d down 2023-03 | `2023-03-03..2023-03-09` | -13.224% | -8.362% | 8.715% | 208 | 100.000% | completed |
| trend-sharpe | 7d down 2026-06 | `2026-05-27..2026-06-02` | -12.076% | -5.854% | 5.928% | 627 | 98.640% | completed |
| stress | 3d down 2022-06 | `2022-06-11..2022-06-13` | -22.702% | -16.506% | 17.833% | 817 | 100.000% | completed |
| stress | 7d down 2022-06 | `2022-06-07..2022-06-13` | -28.323% | -25.373% | 29.557% | 1,456 | 100.000% | completed |

