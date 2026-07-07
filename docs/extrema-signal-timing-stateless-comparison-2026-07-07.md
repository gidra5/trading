# Stateless Extrema Signal Timing Comparison

Generated: 2026-07-07 10:34 UTC

Raw results:

- Baseline start signals: `data/benchmarks/extrema-signal-base-start-2026-07-07.jsonl`
- Origin-tracked entry-end / exit-start: `data/benchmarks/extrema-signal-entry-end-exit-start-2026-07-07.jsonl`
- Stateless entry-end / exit-start: `data/benchmarks/extrema-signal-entry-end-exit-start-stateless-2026-07-07.jsonl`

## Scope

- Market: BTCUSDT 1m spot candles, UTC day-inclusive intervals from `tasks.md`.
- Strategy: `legacy-valley-peak`, static sigma `buySigma=0.1`, `sellSigma=0.1`, both sides enabled, futures-margin shorts, borrow depths `999/999`, max leverage `1x`.
- Stateless entry-end logic:

```text
valley end = latest.rateClamped > 0 and previous.rateClamped <= 0
peak end   = latest.rateClamped < 0 and previous.rateClamped >= 0
```

## Summary

| Run | Avg return | Median return | Positive | Avg DD | Max DD | Avg trades |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline start signals | -2.647% | -2.084% | 6/28 | 4.720% | 15.692% | 747.6 |
| Origin-tracked entry-end / exit-start | -1.730% | -0.048% | 13/28 | 3.833% | 21.389% | 651.8 |
| Stateless entry-end / exit-start | -4.817% | -2.046% | 8/28 | 6.685% | 29.557% | 800.0 |

Stateless versus baseline:

- Higher return on 12/28 intervals.
- Lower max drawdown on 7/28 intervals.
- Fewer trades on 10/28 intervals.
- Average return delta: -2.170%.
- Average max drawdown delta: +1.965%.
- Average trade delta: +52.4 trades.

Stateless versus origin-tracked:

- Higher return on 5/28 intervals.
- Lower max drawdown on 5/28 intervals.
- Fewer trades on 3/28 intervals.
- Average return delta: -3.087%.
- Average max drawdown delta: +2.852%.
- Average trade delta: +148.3 trades.

The stateless interpretation is simpler but materially worse on this frame. The largest regressions versus baseline were:

| Case | Return delta | Return | Max DD |
| --- | ---: | ---: | ---: |
| `2022-06-12..2022-06-18` downtrend | -19.028% | -4.893% -> -23.921% | 11.874% -> 24.415% |
| `2022-06-07..2022-06-13` 7d down stress | -10.094% | -15.280% -> -25.373% | 15.280% -> 29.557% |
| `2023-03-11..2023-03-17` uptrend | -8.226% | -8.054% -> -16.281% | 8.815% -> 16.281% |
| `2022-07-28..2022-08-03` highest OHLC churn | -8.008% | -5.442% -> -13.450% | 7.923% -> 13.450% |
| `2024-01-02..2024-01-04` sideways mid bias churn | -7.761% | +1.719% -> -6.042% | 3.069% -> 6.423% |

Best improvements versus baseline were smaller and concentrated in a few windows:

| Case | Return delta | Return | Max DD |
| --- | ---: | ---: | ---: |
| `2023-06-03..2023-06-05` downtrend low churn | +4.968% | -4.905% -> +0.063% | 5.773% -> 0.137% |
| `2024-11-05..2024-11-11` 7d up | +4.653% | -2.077% -> +2.576% | 2.836% -> 1.087% |
| `2024-02-24..2024-02-26` uptrend low churn | +4.509% | -0.308% -> +4.201% | 0.308% -> 0.367% |
| `2024-07-07..2024-07-09` sideways low bias churn | +1.971% | -0.883% -> +1.087% | 1.438% -> 2.927% |
| `2021-12-14..2021-12-20` very choppy | +1.732% | -7.448% -> -5.716% | 7.448% -> 8.568% |

Conclusion: reject stateless as a default on this frame. It confirms that `0 -> sign`
breakouts are too noisy when treated as entries without remembering the direction that
entered the deadband.
