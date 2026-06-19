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
