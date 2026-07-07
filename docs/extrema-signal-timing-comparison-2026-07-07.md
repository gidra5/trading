# Extrema Signal Timing Comparison

Generated: 2026-07-07 10:25 UTC

Raw results:

- Baseline start signals: `data/benchmarks/extrema-signal-base-start-2026-07-07.jsonl`
- Entry-end / exit-start candidate: `data/benchmarks/extrema-signal-entry-end-exit-start-2026-07-07.jsonl`

## Scope

- Market: BTCUSDT 1m spot candles, UTC day-inclusive intervals from `tasks.md`.
- Strategy: `legacy-valley-peak`, static sigma `buySigma=0.1`, `sellSigma=0.1`, both sides enabled, futures-margin shorts, borrow depths `999/999`, max leverage `1x`.
- Baseline: entry and exit both used the start edge of the clamped-derivative extrema.
- Candidate: entries use the end edge of the extrema; exits keep the start edge.

## Summary

| Run | Avg return | Median return | Positive | Avg DD | Max DD | Avg trades |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline start signals | -2.647% | -2.084% | 6/28 | 4.720% | 15.692% | 747.6 |
| Entry-end / exit-start | -1.730% | -0.048% | 13/28 | 3.833% | 21.389% | 651.8 |

Candidate versus baseline:

- Higher return on 20/28 intervals.
- Lower max drawdown on 21/28 intervals.
- Fewer trades on 21/28 intervals.
- Average return delta: +0.917%.
- Average max drawdown delta: -0.887%.
- Average trade delta: -95.8 trades.

The main concern is the `2023-03-11..2023-03-17` uptrend case: return fell from `-8.054%` to `-21.333%`, and max drawdown rose from `8.815%` to `21.389%`. The largest improvement was the `2022-06-11..2022-06-13` stress case: return improved from `-13.247%` to `-0.087%`, and max drawdown fell from `15.692%` to `3.381%`.

## Intervals

| Group | Case | Interval | Base | Candidate | Delta | Base DD | Candidate DD | Trades |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choppy-week | highest OHLC churn | `2022-07-28..2022-08-03` | -5.442% | -6.438% | -0.996% | 7.923% | 7.545% | 1,398/1,241 |
| choppy-week | highest close churn | `2022-05-14..2022-05-20` | -3.189% | +0.028% | +3.217% | 6.433% | 3.706% | 2,120/1,582 |
| choppy-week | very choppy 2021-12 | `2021-12-14..2021-12-20` | -7.448% | -5.074% | +2.374% | 7.448% | 5.461% | 1,426/1,044 |
| choppy-week | very choppy 2021-09 | `2021-09-08..2021-09-14` | -2.506% | -1.629% | +0.877% | 6.348% | 5.435% | 1,246/1,108 |
| choppy-week | near-flat close | `2023-03-18..2023-03-24` | +2.371% | +3.413% | +1.042% | 3.132% | 3.035% | 1,455/1,210 |
| regime-week | uptrend | `2023-03-11..2023-03-17` | -8.054% | -21.333% | -13.279% | 8.815% | 21.389% | 1,542/1,027 |
| regime-week | sideways | `2026-04-22..2026-04-28` | +2.597% | +0.419% | -2.178% | 0.721% | 1.447% | 401/248 |
| regime-week | downtrend | `2022-06-12..2022-06-18` | -4.893% | -3.779% | +1.114% | 11.874% | 9.453% | 2,368/2,140 |
| 3d-regime | uptrend low churn | `2024-02-24..2024-02-26` | -0.308% | +0.413% | +0.721% | 0.308% | 0.070% | 24/22 |
| 3d-regime | uptrend high churn | `2022-06-19..2022-06-21` | -4.111% | -1.791% | +2.319% | 4.317% | 2.115% | 1,182/1,018 |
| 3d-regime | downtrend low churn | `2023-06-03..2023-06-05` | -4.905% | +0.067% | +4.972% | 5.773% | 0.070% | 57/56 |
| 3d-regime | downtrend high churn | `2022-06-13..2022-06-15` | +10.090% | +6.970% | -3.120% | 5.111% | 6.458% | 1,204/1,062 |
| 3d-regime | sideways high bias churn | `2021-10-19..2021-10-21` | -0.378% | -0.532% | -0.154% | 1.937% | 1.899% | 558/509 |
| 3d-regime | sideways high bias low churn | `2025-02-14..2025-02-16` | -1.296% | +0.022% | +1.319% | 1.387% | 0.115% | 69/64 |
| 3d-regime | sideways low bias churn | `2024-07-07..2024-07-09` | -0.883% | +0.470% | +1.353% | 1.438% | 2.922% | 440/319 |
| 3d-regime | sideways low bias low churn | `2025-07-04..2025-07-06` | -0.043% | -0.084% | -0.041% | 0.043% | 0.084% | 6/6 |
| 3d-regime | sideways mid bias churn | `2024-01-02..2024-01-04` | +1.719% | -0.272% | -1.991% | 3.069% | 1.437% | 223/258 |
| 3d-regime | sideways mid bias low churn | `2023-09-15..2023-09-17` | -0.043% | -0.011% | +0.032% | 0.052% | 0.013% | 5/4 |
| trend-sharpe | 3d up 2024-11 | `2024-11-09..2024-11-11` | +0.254% | +1.122% | +0.868% | 0.189% | 0.072% | 475/463 |
| trend-sharpe | 3d up 2023-12 | `2023-12-03..2023-12-05` | -2.092% | -0.965% | +1.126% | 2.121% | 1.706% | 145/316 |
| trend-sharpe | 3d down 2026-06 | `2026-06-01..2026-06-03` | +0.796% | +0.887% | +0.090% | 0.477% | 0.540% | 375/301 |
| trend-sharpe | 3d down 2023-03 | `2023-03-07..2023-03-09` | -4.704% | +0.251% | +4.955% | 6.359% | 1.220% | 214/219 |
| trend-sharpe | 7d up 2023-12 | `2023-11-29..2023-12-05` | -1.843% | -1.393% | +0.450% | 2.097% | 1.723% | 239/390 |
| trend-sharpe | 7d up 2024-11 | `2024-11-05..2024-11-11` | -2.077% | +0.887% | +2.964% | 2.836% | 0.543% | 907/1,095 |
| trend-sharpe | 7d down 2023-03 | `2023-03-03..2023-03-09` | -4.309% | +0.716% | +5.025% | 5.986% | 1.438% | 293/306 |
| trend-sharpe | 7d down 2026-06 | `2026-05-27..2026-06-02` | -4.904% | -4.486% | +0.418% | 4.981% | 4.569% | 444/421 |
| stress | 3d down 2022-06 | `2022-06-11..2022-06-13` | -13.247% | -0.087% | +13.160% | 15.692% | 3.381% | 844/676 |
| stress | 7d down 2022-06 | `2022-06-07..2022-06-13` | -15.280% | -16.244% | -0.964% | 15.280% | 19.472% | 1,273/1,145 |
