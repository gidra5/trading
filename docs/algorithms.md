# Algorithms

## Moving Average

The default strategy compares short and long moving averages, places limit buys when the fast average is above the slow average, and places sells for trend reversal, take-profit, or stop-loss conditions.

## Legacy Valley/Peak

This is the TypeScript port of the deleted `model/index.py` Python experiment. It keeps rolling averages over these default windows:

```text
1s, 1m, 10m, 30m, 1h, 4h, 12h
```

For every window it estimates the derivative of the rolling average. It buys when the shortest buy-price derivative crosses from negative to non-negative and the longest confirmation window is still negative. It sells when the one-minute sell-price derivative crosses from positive to non-positive and the 10-minute and 30-minute confirmation windows are also positive.

Trade size follows the original Gaussian derivative sizing:

```text
quote buy size ~= quote balance * buyRate * gaussian(derivative, 0, buySigma)
base sell size ~= base balance * sellRate * gaussian(derivative, 0, sellSigma)
```

The live UI exposes the main parameters: warmup period, buy/sell rate, buy/sell sigma, min/max trade quote, and confirmation offset. Applying algorithm changes resets the paper bot so old orders and old strategy memory do not mix with the new parameters.

## Bidirectional Market-Rebalance Strategies

The bot now supports automated long/short strategies that target signed exposure:

```text
positive target = long
negative target = short
zero target = flat
```

These strategies use immediate paper `market` fills instead of resting limit orders. Fills apply configured fee and market slippage, update signed base inventory, maintain both long and short average entry prices, record realized PnL when closing either side, and pass through the leverage guard. Long target notional is capped by `maxPositionQuote` and current equity times `maxLeverage`. Short target notional is additionally capped against the debt-leverage guard because borrowed base counts as external debt in the position ledger.

Current bidirectional strategies:

- `trend-following` - goes long when fast and slow returns are both positive enough, short when both are negative enough, and exits when momentum fades.
- `volatility-breakout` - goes long above the recent rolling high, short below the recent rolling low, and exits when price mean-reverts toward the range midpoint.
- `mean-reversion` - goes long on large negative z-score deviations and short on large positive z-score deviations when the larger trend filter is not too strong against the trade.

The dashboard exposes each strategy's windows, entry/exit thresholds, and target exposure percentage. These are research baselines, not production-ready alpha. Initial BTCUSDT benchmarks show that naive long/short signals still lose after fees/slippage, especially when cooldown is short and turnover is high.

## Benchmark Controls

The benchmark harness also includes explicit control algorithms:

- `benchmark-always-flat`
- `benchmark-always-long`
- `benchmark-always-short`
- `benchmark-random-sign`

These controls use the same paper fill path as the bidirectional strategies. The random-sign control is deterministic and seeded through `benchmarkRandomSeed`; it is opt-in for CLI benchmarks because it intentionally creates high churn and can be slow on long windows.
