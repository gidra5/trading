# Legacy Valley/Peak Algorithm

The project currently has one automated strategy: `legacy-valley-peak`.

Legacy valley/peak is a long-only mean-turning strategy. It watches rolling price
averages across several timeframes, places a buy limit order after a confirmed local
valley, and places a sell limit order after a confirmed local peak. It does not open
automated shorts, does not use online outcome labels, and does not force positions out
with separate quality-exit rules.

Manual fills and manual short positions still exist for inspection and position
management. The automated strategy itself only buys base with quote and sells base it
already holds.

## Runtime Flow

Every accepted price tick runs this sequence:

1. Store the latest price and timestamp.
2. Cancel open limit orders older than `staleOrderMs`, releasing reserved quote/base.
3. Fill any open limit orders crossed by the current tick.
4. Append the latest price to public memory.
5. If the bot is running, evaluate the legacy valley/peak detector.
6. If a signal is emitted and cooldown/open-order limits allow it, place one new limit
   order.
7. Recalculate metrics unless the caller requested a replay fast path.

The strategy never fills a signal immediately. A signal creates a resting limit order:

- buy orders rest below the current market by `limitOffsetBps`
- sell orders rest above the current market by `limitOffsetBps`
- orders fill later only if replay/live price crosses the limit
- stale orders are cancelled after `staleOrderMs`
- the bot will not place more than `maxOpenOrders` open automated orders

## Rolling Average State

The detector keeps rolling averages for these default windows:

```text
1s, 1m, 10m, 30m, 1h, 4h, 12h
```

For each window it stores:

- raw price entries and timestamps
- the current rolling average
- a derivative estimate sampled inside the rolling window
- a clamped derivative point used for turning-point detection

Derivative clamping is important. A small derivative is treated as zero unless it
exceeds the configured positive or negative threshold for that window:

```text
rateClamped = derivative >= highThreshold ? derivative
            : derivative <= -lowThreshold ? derivative
            : 0
```

This prevents tiny average movements from constantly generating valley/peak flips.

The detector shares the same average array for buy and sell memory. `buyAverages` and
`sellAverages` are separate fields for state compatibility, but current code points both
at the same rolling average objects.

## Valley Signal

A valley is detected when the selected window's clamped derivative turns upward:

```text
latest.rateClamped >= 0
previous.rateClamped < 0
```

The default buy source is the shortest window:

```text
legacyValleyPeak.buyDataIndex = 0
```

The buy confirmation window is:

```text
buyDataIndex + buyConfirmationOffset
```

With the default `buyConfirmationOffset = 6`, a 1s valley is only accepted while the
12h confirmation average is still falling (`rateClamped < 0`). That makes the legacy
entry contrarian: it buys a short-term upward turn while the broader window has not yet
turned up.

## Peak Signal

A peak is detected when the selected window's clamped derivative turns downward:

```text
latest.rateClamped <= 0
previous.rateClamped > 0
```

The default sell source is the 1m window:

```text
legacyValleyPeak.sellDataIndex = 1
```

The default sell confirmations are:

```text
legacyValleyPeak.sellConfirmationOffsets = [2, 1]
```

That checks the 30m and 10m windows relative to the 1m source and requires each
confirmation window to still be rising (`rateClamped > 0`). In practice, the detector
sells a short-term peak while larger local averages still show upward movement.

## Warmup

The detector records rolling state immediately but refuses to trade before warmup:

```text
legacyValleyPeak.saturationSec = 3600
```

This avoids trading from empty or underfilled rolling windows. During warmup,
`evaluateLegacyValleyPeak` always returns `hold`.

## Buy Sizing

On a confirmed valley, the strategy computes a desired quote spend from available quote
balance and the 30m derivative:

```text
desired buy quote =
  quoteFree
  * legacyValleyPeak.buySpendRate
  * gaussian(derivative30m, 0, legacyValleyPeak.buySigma)
```

The Gaussian term is largest when the derivative is near zero and smaller when the
average is moving sharply. That keeps the largest buys near smooth turning points.

The final buy quote is clamped by:

- `legacyValleyPeak.minTradeQuote`
- `legacyValleyPeak.maxTradeQuote`
- `quoteFree * 0.98`
- remaining long capacity under `maxPositionQuote`

If the final quote is below the minimum order size, no buy order is created.

## Sell Sizing

On a confirmed peak, the strategy computes a desired base quantity from available base
and the same 30m derivative style:

```text
desired sell base =
  baseFree
  * legacyValleyPeak.sellAmountRate
  * gaussian(derivative30m, 0, legacyValleyPeak.sellSigma)
```

The final sell quantity is clamped by:

- `legacyValleyPeak.minTradeQuote / currentPrice`
- `legacyValleyPeak.maxTradeQuote / currentPrice`
- available `baseFree`

If the final notional is below the minimum order size, no sell order is created. Because
the automated strategy checks `quantity <= baseFree`, it cannot sell through flat or
open a short.

## Limit Order Accounting

Buy order creation:

- computes limit price as `marketPrice * (1 - limitOffsetBps / 10000)`
- computes quantity from the clamped quote size and limit price
- reserves estimated quote cost including fee
- pushes an open limit order into state

Sell order creation:

- computes limit price as `marketPrice * (1 + limitOffsetBps / 10000)`
- reserves base quantity
- pushes an open limit order into state

Order fill checks are simple OHLC/tick crossing rules:

- a buy fills when `tick.price <= order.price`
- a sell fills when `tick.price >= order.price`

When a buy fills, reserved quote is released/spent, base increases, fees are recorded,
and long average entry price is updated. When a sell fills, reserved base is released,
quote increases, realized PnL is recorded against `avgEntryPrice`, fees are recorded,
and average entry resets when the long is fully closed.

Every fill runs the leverage guard. If the guard rejects a fill, the state rolls back,
the order reserve is released, and the order is cancelled with a leverage-limit reason.

## Current Defaults To Watch

| Config | Default | Meaning |
| --- | ---: | --- |
| `algorithm` | `legacy-valley-peak` | The only automated algorithm key. |
| `maxLeverage` | `1` | Leverage guard for the account; automated legacy does not open shorts. |
| `maxPositionQuote` | `4500` | Maximum long notional the strategy can build. |
| `limitOffsetBps` | `2` | Distance from current price for new limit orders. |
| `maxOpenOrders` | `3` | Maximum resting automated orders. |
| `cooldownMs` | `30000` | Minimum delay between newly created automated orders. |
| `staleOrderMs` | `180000` | Time before an unfilled limit order is cancelled. |
| `minOrderQuote` | `25` | Global minimum order notional. |
| `legacyValleyPeak.saturationSec` | `3600` | Rolling detector warmup before signals can trade. |
| `legacyValleyPeak.maxTradeQuote` | `3000` | Per-signal quote/notional clip. |
| `legacyValleyPeak.buySigma` | `0.1` | Buy Gaussian width around the sizing derivative. |
| `legacyValleyPeak.sellSigma` | `0.1` | Sell Gaussian width around the sizing derivative. |

## Known Limitations

Legacy valley/peak is a heuristic turning-point detector. It can buy early in a falling
trend, miss fast reversals when limit orders do not fill, and churn when local averages
oscillate around zero after fees. It does not currently use funding, spread, order-book
depth, liquidation data, open interest, learned labels, or regime-aware position sizing.

Perfect-margin capture remains the north-star benchmark, not an achievable target.
