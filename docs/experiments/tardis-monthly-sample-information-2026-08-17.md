# Tardis monthly-sample external information audit

Generated 2026-08-19T19:21:53.094Z. Free Tardis samples cover 10 independent UTC days.

## Outcome

These results replace an unqualified `awaiting-data` label with sparse monthly-sample evidence. A pass here is still weaker than a continuous point-in-time archive because only the first UTC day of each month is public.

| target | candidates | stable | best feature | lookback | primary bits | transfer bits | sign bits | magnitude bits |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 1s | 118 | 0 | binance-spot-btcusdt L1 quantity imbalance | latest | — | — | — | — |
| 5s | 118 | 0 | binance-spot-btcusdt L1 quantity imbalance | latest | — | — | — | — |
| 15s | 118 | 0 | binance-spot-btcusdt L1 quantity imbalance | latest | — | — | — | — |
| 1m | 118 | 0 | binance-spot-btcusdt L1 quantity imbalance | latest | — | — | — | — |
| 5m | 118 | 0 | binance-spot-btcusdt L1 quantity imbalance | latest | — | — | — | — |
| 15m | 118 | 0 | binance-spot-btcusdt L1 quantity imbalance | latest | — | — | — | — |
| 30m | 118 | 0 | binance-spot-btcusdt L1 quantity imbalance | latest | — | — | — | — |
| 1h | 118 | 0 | binance-spot-btcusdt L1 quantity imbalance | latest | — | — | — | — |

## Causal and validation constraints

- Tardis collector local_timestamp; the last quote observed in each second predicts later Binance quote mids.
- quartiles of same-horizon trailing Binance return and realized volatility.
- training-only quartiles; missing observations matched before scoring.
- positive held-out bits in Aug-Sep 2025, Oct-Nov 2025, and May-Jun 2026.
- first UTC day of each month only; this is independent multi-regime sample evidence, not continuous-history evidence.

Complete rankings and frozen quantile edges are in `data/benchmarks/tardis-monthly-sample-information.json`.
