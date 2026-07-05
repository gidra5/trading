# Derivative Clamp Mode Benchmark

Generated: 2026-07-05 17:44
Raw results: `data/benchmarks/derivative-clamp-mode-2026-07-05-173814.jsonl`

## Scope

- Market: BTCUSDT 1m spot candles, UTC day-inclusive intervals from `tasks.md`.
- Strategy: static sigma `buySigma=0.1`, `sellSigma=0.1`, mode both, futures-margin shorts, borrow depths `999/999`, max leverage `1x`.
- Derivative source: `kama`.
- KAMA: `erLen=20`, `fastLen=20`, `slowLen=200`, `power=1`.
- Compared modes: `deadband` is the previous stateless threshold clamp; `hysteresis` keeps an active derivative sign until the raw derivative crosses zero.

## Summary

| Mode | Avg return | Median return | Positive | Avg DD | Max DD | Avg trades |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `deadband` | -0.163% | +0.117% | 17/28 | 2.175% | 15.333% | 106.6 |
| `hysteresis` | -6.256% | -5.210% | 5/28 | 8.225% | 22.353% | 272.8 |

Hysteresis had higher return on `7/28` intervals and lower max drawdown on `1/28` intervals.

## Intervals

| Group | Case | Interval | Market | Deadband | Hysteresis | Delta | Deadband DD | Hysteresis DD | Trades |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| choppy-week | highest OHLC churn | `2022-07-28..2022-08-03` | -0.592% | -3.272% | -13.653% | -10.380% | 3.460% | 13.653% | 49/425 |
| choppy-week | highest close churn | `2022-05-14..2022-05-20` | -0.294% | -0.116% | -3.070% | -2.953% | 3.681% | 7.669% | 313/732 |
| choppy-week | very choppy 2021-12 | `2021-12-14..2021-12-20` | +0.453% | +1.052% | -9.568% | -10.620% | 2.525% | 9.568% | 175/734 |
| choppy-week | very choppy 2021-09 | `2021-09-08..2021-09-14` | +0.518% | +1.074% | -1.633% | -2.707% | 0.477% | 6.781% | 161/615 |
| choppy-week | near-flat close | `2023-03-18..2023-03-24` | +0.217% | -0.128% | -1.906% | -1.778% | 2.701% | 3.881% | 262/860 |
| regime-week | uptrend | `2023-03-11..2023-03-17` | +35.951% | -15.331% | -18.318% | -2.988% | 15.333% | 21.800% | 107/218 |
| regime-week | sideways | `2026-04-22..2026-04-28` | +0.009% | -0.345% | -0.104% | +0.242% | 0.768% | 2.148% | 4/53 |
| regime-week | downtrend | `2022-06-12..2022-06-18` | -33.260% | +5.056% | -16.071% | -21.127% | 1.450% | 21.071% | 542/1141 |
| 3d-regime | uptrend low churn | `2024-02-24..2024-02-26` | +7.355% | -0.014% | +5.345% | +5.359% | 0.014% | 0.652% | 2/36 |
| 3d-regime | uptrend high churn | `2022-06-19..2022-06-21` | +9.239% | -11.672% | -10.937% | +0.736% | 11.677% | 11.065% | 68/247 |
| 3d-regime | downtrend low churn | `2023-06-03..2023-06-05` | -5.559% | +0.059% | +2.447% | +2.388% | 0.062% | 1.010% | 6/52 |
| 3d-regime | downtrend high churn | `2022-06-13..2022-06-15` | -15.017% | +0.489% | -0.750% | -1.240% | 0.415% | 4.087% | 266/492 |
| 3d-regime | sideways high bias churn | `2021-10-19..2021-10-21` | +0.302% | -0.057% | -4.499% | -4.442% | 0.986% | 5.517% | 92/221 |
| 3d-regime | sideways high bias low churn | `2025-02-14..2025-02-16` | -0.507% | +0.003% | +0.772% | +0.769% | 0.004% | 1.397% | 1/4 |
| 3d-regime | sideways low bias churn | `2024-07-07..2024-07-09` | -0.309% | -1.314% | -5.922% | -4.608% | 3.450% | 6.376% | 13/99 |
| 3d-regime | sideways low bias low churn | `2025-07-04..2025-07-06` | -0.348% | +0.000% | +0.118% | +0.118% | 0.000% | 0.514% | 1/2 |
| 3d-regime | sideways mid bias churn | `2024-01-02..2024-01-04` | -0.064% | +0.563% | -13.829% | -14.392% | 0.333% | 13.862% | 51/162 |
| 3d-regime | sideways mid bias low churn | `2023-09-15..2023-09-17` | +0.018% | +0.000% | +0.000% | +0.000% | 0.000% | 0.000% | 0/0 |
| trend-sharpe | 3d up 2024-11 | `2024-11-09..2024-11-11` | +15.865% | +1.859% | -9.843% | -11.702% | 0.781% | 10.756% | 51/27 |
| trend-sharpe | 3d up 2023-12 | `2023-12-03..2023-12-05` | +11.719% | +0.280% | -4.334% | -4.613% | 1.149% | 5.064% | 66/50 |
| trend-sharpe | 3d down 2026-06 | `2026-06-01..2026-06-03` | -12.938% | -2.416% | -0.257% | +2.159% | 2.887% | 3.163% | 49/156 |
| trend-sharpe | 3d down 2023-03 | `2023-03-07..2023-03-09` | -9.135% | +3.693% | -6.366% | -10.058% | 1.976% | 7.654% | 16/28 |
| trend-sharpe | 7d up 2023-12 | `2023-11-29..2023-12-05` | +16.538% | +0.434% | -8.863% | -9.296% | 0.418% | 9.136% | 81/67 |
| trend-sharpe | 7d up 2024-11 | `2024-11-05..2024-11-11` | +30.653% | +2.003% | -14.444% | -16.447% | 1.111% | 14.444% | 128/114 |
| trend-sharpe | 7d down 2023-03 | `2023-03-03..2023-03-09` | -13.224% | +2.342% | -6.563% | -8.904% | 1.548% | 7.415% | 37/117 |
| trend-sharpe | 7d down 2026-06 | `2026-05-27..2026-06-02` | -12.076% | +0.176% | +0.030% | -0.145% | 0.002% | 4.789% | 3/63 |
| stress | 3d down 2022-06 | `2022-06-11..2022-06-13` | -22.702% | +5.535% | -10.591% | -16.127% | 2.177% | 14.483% | 197/314 |
| stress | 7d down 2022-06 | `2022-06-07..2022-06-13` | -28.323% | +5.483% | -22.353% | -27.836% | 1.527% | 22.353% | 245/608 |

