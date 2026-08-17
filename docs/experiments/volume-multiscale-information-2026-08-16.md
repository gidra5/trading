# Volume and completed higher-timeframe information for the next 1s return

Generated 2026-08-16T17:01:28.761Z from BTCUSDT spot OHLCV history.

## Answer

The earlier price-only basis was not the best tested basis. **last completed 1h log(1 + volume)** adds 0.098779680 bits/target after the price core-three, versus 0.010725766 for the previous fourth feature.

The price core-three itself contributes 0.14229402 bits/target beyond the latest-return state. The best external feature changes geometric assigned probability by 7.0867% on top of that state.

## Strongest additions after the price core-three

| rank | candidate | scale | family | individual bits | marginal bits | cumulative bits | positive years |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | last completed 1h log(1 + volume) | 1h | completed-bar volume level | 0.13396077 | 0.098779680 | 0.24111009 | 4/4 |
| 2 | last completed 15m log(1 + volume) | 15m | completed-bar volume level | 0.13358049 | 0.097217325 | 0.23955528 | 4/4 |
| 3 | last completed 4h log(1 + volume) | 4h | completed-bar volume level | 0.12929366 | 0.096515554 | 0.23895043 | 4/4 |
| 4 | last completed 5m log(1 + volume) | 5m | completed-bar volume level | 0.13131678 | 0.095239832 | 0.23754294 | 4/4 |
| 5 | last completed 1m log(1 + volume) | 1m | completed-bar volume level | 0.12093925 | 0.085819171 | 0.22811319 | 4/4 |
| 6 | 1s high-low range | 1s | candle shape | 0.086738468 | 0.080208281 | 0.22250230 | 4/4 |
| 7 | last completed 15s log(1 + volume) | 15s | completed-bar volume level | 0.10355760 | 0.070695557 | 0.21298957 | 4/4 |
| 8 | last completed 5s log(1 + volume) | 5s | completed-bar volume level | 0.084230412 | 0.054370902 | 0.19666492 | 4/4 |
| 9 | 1s close location in range | 1s | candle shape | 0.075315372 | 0.054076734 | 0.19637075 | 4/4 |
| 10 | last completed 5s close location in range | 5s | completed-bar candle shape | 0.079353978 | 0.044625712 | 0.18691973 | 4/4 |
| 11 | 1s log-volume surprise vs EMA(512s) | 1s | relative volume | 0.057674305 | 0.037758785 | 0.18005280 | 4/4 |
| 12 | 1s log-volume surprise vs EMA(2048s) | 1s | relative volume | 0.056668996 | 0.037559396 | 0.17985341 | 4/4 |
| 13 | 1s log-volume surprise vs EMA(128s) | 1s | relative volume | 0.056616847 | 0.036739922 | 0.17903394 | 4/4 |
| 14 | 1s log-volume surprise vs EMA(32s) | 1s | relative volume | 0.050596649 | 0.032384533 | 0.17467855 | 4/4 |
| 15 | last completed 5s high-low range | 5s | completed-bar candle shape | 0.070548453 | 0.030182085 | 0.17247610 | 4/4 |
| 16 | last completed 15s close location in range | 15s | completed-bar candle shape | 0.056529672 | 0.029797536 | 0.17209155 | 4/4 |
| 17 | last completed 15s high-low range | 15s | completed-bar candle shape | 0.056956171 | 0.022808782 | 0.16510280 | 4/4 |
| 18 | 1s log-volume surprise vs EMA(8s) | 1s | relative volume | 0.037972905 | 0.022772443 | 0.16506646 | 4/4 |
| 19 | last completed 1m high-low range | 1m | completed-bar candle shape | 0.050745249 | 0.022104239 | 0.16439825 | 4/4 |
| 20 | last completed 5s absolute return | 5s | completed-bar volatility | 0.045754348 | 0.019768436 | 0.16206245 | 4/4 |
| 21 | last completed 5m high-low range | 5m | completed-bar candle shape | 0.044360381 | 0.019316453 | 0.16161956 | 4/4 |
| 22 | last completed 15m high-low range | 15m | completed-bar candle shape | 0.037456860 | 0.014618991 | 0.15695695 | 4/4 |
| 23 | last completed 15s absolute return | 15s | completed-bar volatility | 0.031971154 | 0.010994516 | 0.15328853 | 4/4 |
| 24 | EMA acceleration(n=2s,k=2s) (price-basis benchmark) | 1s | EMA acceleration | 0.025725255 | 0.010725766 | 0.15301978 | 4/4 |
| 25 | last completed 1m absolute return | 1m | completed-bar volatility | 0.027074834 | 0.010385204 | 0.15267922 | 4/4 |
| 26 | last completed 15s close−EMA(8 bars) | 15s | completed-bar indicator | 0.024274996 | 0.0090047288 | 0.15129874 | 4/4 |
| 27 | last completed 5s EMA acceleration(n=2,k=1 bars) | 5s | completed-bar indicator | 0.024676646 | 0.0085742792 | 0.15086829 | 4/4 |
| 28 | Price−EMA(8s) (price-basis benchmark) | 1s | EMA | 0.037623316 | 0.0085328145 | 0.15082683 | 4/4 |
| 29 | last completed 5s EMA slope(n=8,k=8 bars) | 5s | completed-bar indicator | 0.022983790 | 0.0085284601 | 0.15082248 | 4/4 |
| 30 | last completed 1m close−EMA(8 bars) | 1m | completed-bar indicator | 0.022596670 | 0.0084205942 | 0.15071461 | 4/4 |

## Best new feature at each resolution

| scale | candidate | marginal bits | positive years |
|---|---|---:|---:|
| 1s | 1s high-low range | 0.080208281 | 4/4 |
| 5s | last completed 5s log(1 + volume) | 0.054370902 | 4/4 |
| 15s | last completed 15s log(1 + volume) | 0.070695557 | 4/4 |
| 1m | last completed 1m log(1 + volume) | 0.085819171 | 4/4 |
| 5m | last completed 5m log(1 + volume) | 0.095239832 | 4/4 |
| 15m | last completed 15m log(1 + volume) | 0.097217325 | 4/4 |
| 1h | last completed 1h log(1 + volume) | 0.098779680 | 4/4 |
| 4h | last completed 4h log(1 + volume) | 0.096515554 | 4/4 |

## Causal construction

For a target return from second `t` to `t+1`, every input contains data only through second `t`. A UTC-aligned 5s, 15s, 1m, 5m, 15m, 1h, or 4h bar becomes visible only when its final constituent 1s candle has closed. The current partial bar is never exposed.

All candidate features are split into four year-1 equal-mass cells. We score the exact next-return distribution with rolling held-out log information:

```text
mean_test[log2 P_train(R | previous_return, price_core3, candidate)
        - log2 P_train(R | previous_return, price_core3)]
```

The deterministic evaluation sample uses one of every 4 targets. The history is still large enough to score four separate annual test epochs, each trained only on earlier years.

## Limits

- The spot candle archive contains base volume but not quote volume, trade count, or taker-buy/sell imbalance.
- This scan tests additions after the previously selected three-feature price basis; it does not exhaustively enumerate every mixed price/volume subset.
- Each candidate is quartile-quantized. Continuous models may retain useful within-cell variation.
- Aligned higher-timeframe bars are deliberately stale between closes; partial bars are excluded to prevent future leakage.
- Predictive log information is not a trading-PnL estimate and does not include fees, spread, latency, or impact.

## Reproducibility

```text
node --conditions=development --import tsx scripts/analyze-volume-multiscale-information.ts
```

Complete rankings are stored in `data/benchmarks/volume-multiscale-information.json`.
