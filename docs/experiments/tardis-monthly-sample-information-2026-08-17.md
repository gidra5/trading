# Tardis monthly-sample external information audit

Generated 2026-08-17T10:14:25.738Z. Free Tardis samples cover 10 independent UTC days.

## Outcome

These results replace an unqualified `awaiting-data` label with sparse monthly-sample evidence. A pass here is still weaker than a continuous point-in-time archive because only the first UTC day of each month is public.

| target | candidates | stable | best feature | lookback | primary bits | transfer bits | sign bits | magnitude bits |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 1s | 118 | 103 | binance-spot-btcusdt L1 quantity imbalance | latest | 0.046000 | 0.044636 | 0.044670 | 0.017566 |
| 5s | 118 | 93 | binance-spot-btcusdt L1 quantity imbalance | latest | 0.071323 | 0.075743 | 0.056154 | 0.014432 |
| 15s | 118 | 74 | coinbase-spot-btcusd realized volatility | 300s | 0.060975 | 0.041179 | 0.012799 | 0.069473 |
| 1m | 118 | 16 | coinbase-spot-btcusd realized volatility | 300s | 0.013722 | 0.006016 | -0.000592 | 0.017479 |
| 5m | 118 | 0 | binance-spot-btcusdt trailing return | 300s | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 15m | 118 | 0 | BTC liquidation count | 1s | -0.001622 | -0.001267 | -0.000038 | -0.001085 |
| 30m | 118 | 0 | kraken-spot-xbtusd realized volatility | 2s | -0.000091 | 0.001194 | -0.000181 | 0.000090 |
| 1h | 118 | 0 | deribit-btc-perpetual trailing return | 1s | -0.002399 | -0.003050 | -0.000098 | -0.001495 |

## Causal and validation constraints

- Tardis collector local_timestamp; the last quote observed in each second predicts later Binance quote mids.
- quartiles of same-horizon trailing Binance return and realized volatility.
- training-only quartiles; missing observations matched before scoring.
- positive held-out bits in Aug-Sep 2025, Oct-Nov 2025, and May-Jun 2026.
- first UTC day of each month only; this is independent multi-regime sample evidence, not continuous-history evidence.

Complete rankings and frozen quantile edges are in `data/benchmarks/tardis-monthly-sample-information.json`.
