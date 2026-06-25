# Legacy Valley/Peak Algorithm

The project currently has one automated strategy: `legacy-valley-peak`.

Legacy valley/peak is a bidirectional mean-turning strategy. It watches rolling price
averages across several timeframes. The current default is the relaxed per-lot exit-grid
mode: it opens long lots at confirmed valleys, opens short lots at confirmed peaks, and
manages each lot with its own resettable partial-exit ladder.

It does not use online outcome labels or separate quality-exit rules.
`longSideEnabled` and `shortSideEnabled` can disable one side for comparison runs, but
the default strategy now trades both sides.

The account, order, lot, and leverage model is documented separately in
[Trading Model and Position Ledger](position-ledger.md). UI-driven manual controls are
documented in [Manual Position Management](manual-position-management.md).
The automated entry and exit-management workflow is documented in
[Automated Position Management](automated-position-management.md).

## The model

### The market model

The assumed model is very simple. Treat the market price as a time series consisting of the alternation of 3 states: up, down, and flat. The core assumption is that we can't have two identical states in a row, which means optimal decisions can be made only at the transition points. 

To detect these 3 states, we use a derivative of the simple moving average. Given a up/down trend thresholds and a timeframe, we compute the derivative of the SMA over the timeframe and compare against thresholds:
1. if its withing the thresholds around 0, we consider it a sideways trend
2. if it is above the threshold upper, we consider it an uptrend
3. if it is below the threshold, we consider it a downtrend

### Timeframes

We distinguish between 3 timeframes: short, medium, and long. We use them to determine market state at different scales and make decisions based on that. 
Given some candle size ratio, for each of the timeframes we determine the following:
1. trend
2. volatility - average size of the candle bodies and wicks.
3. extrema - min and max price over the timeframe

short range = execution / grid / stop noise
mid range = leverage and trade lifetime
long range = account-level exposure and “do I still want to hold this asset?”

compute recent avg and max candle size for each range as candle interval

### The trading model

The trading model is much more complex that traditionally used on the platforms.

The fundamental concept is a lot. A lot is an entry into the market at some price and direction. It has the following properties:
1. size - the amount of the asset associated with the lot
2. cost - the price we paid for the corresponding size
3. break even price - price at which the lot an be closed and yield zero PnL
4. internal borrow - 
   1. the lot lends the cost
   2. the size of the lended cost.
5. external borrow - the part of the cost that borrowed from the platform through leverage.
   - derived leverage of the lot=(cost-internal borrow)/(cost-internal borrow-external borrow)
6. fractions required to sell that will move break even price to the extrema corresponding to one of the timeframes.

The most notable change from the traditional model is that we have notion of internal borrow. This allow us to sell the asset associated with the lot to open the opposing lot instead of closing the current one, but with the condition that the new lot will return back at least the borrowed amount. We may also borrow from multiple lots and externally as well to build up the resulting position.

That may create chains of borrowing and lending and keep money fluid, without freezing it in bad positions.

In ideal prediction of the market, the chain depth is limited, or even non existent.

All lots together form an aggregated position, which is basically a sum of the lots. The main difference from simple lot is that it cant have internal borrow, only external. It also has additional liquidation price

Besides regular lots we can create hedged lots. These are always created in pairs of opposite lots, that are supposed to hedge each other. The main advantage is that we can create them without actually executing anything.

### The entry strategy model

Each up/down state represents a profit opportunity, so we must enter at the start of an trend and exit when the trend reverses. Entering during the ongoing up/down trend does not make sense, because we will immediately get loss without any opportunities for profit until we cross break-even price.

Thus for long positions we look for low price in overall uptrend, and symmetrically for short positions we look for high price in overall downtrend. This basically exploits the zig-zag market pattern.

Once entry price is decided, we must determine the target size. We scale it proportionally to the slope of the overall trend - the faster it moves, the smaller the target size make sense, if we want to concentrate most of the capital in certain extrema.

We also must consider the current free capital and limits on order size.

### The exit strategy model

https://chatgpt.com/c/6a3cfff7-2978-83eb-bb5b-4c0449627fff

When we hit an extrema, for every profitable lot we want to optimize the PnL given uncertainty in where the price will move.

At an extrema, define a sell distribution between current price and an anchor price. The realized grid will approximate this distribution given the constraints on order sizes, price steps, and order count.

the shape of the distribution is determined by our confidence in the peak quality - how likely it is that the price will move down past some threshold, like short break even price. It also should optimize expected profit given opportunity cost of skipping it.

the anchor price is between break even price of the lot and break even of the overall account. if the account break even above the lot break even, we should prioritize profits on individual lots and dont risk staying below the break even longer.

if after the grid was created, the market improved, we may reset the grid according to new prices if it is more profitable to do so, given some reset price. More concretely, it may be the case if the sell distribution changes significantly and the remaining orders approximate it very badly. 

In total this defines these parameters:
1. the distribution of the sell orders over the range from current price to the anchor price
2. realized grid constraints: order size, price step, order count
3. the anchor price
4. the confidence in the extrema quality
5. the reset condition
6. the reset price

### Sell distribution model

The distribution is a kind of (beta distribution)[https://chatgpt.com/c/6a3cfff7-2978-83eb-bb5b-4c0449627fff]. 
It is supposed to be skewed towards the peak if we are confident in the peak quality, and uniform otherwise.

First consider the opportunity cost of the sell. Given some expected down move D and up move U, and our confidence q in the peak quality as probability between choosing either, we can compute the expected opportunity cost of the sell:

C = D * q + U * (1 - q)

Then the optimal sell fraction with the expectation that the peak is true, is:

f = D * q / C

The short-term price range and re-entry price can be used to estimate the D and U.



## Runtime Flow

Every accepted price tick runs this sequence:

1. Store the latest price and timestamp.
2. Cancel open limit orders older than `staleOrderMs`, releasing reserved quote/base.
3. Fill any open limit orders crossed by the current tick.
4. Append the latest price to public memory.
5. If the bot is running, evaluate the legacy valley/peak detector.
6. If a signal is emitted and cooldown/open-order limits allow it, place the configured
   order action.
7. Recalculate metrics unless the caller requested a replay fast path.

With `legacyValleyPeak.exitGridEnabled = false`, the older limit-only behavior never
fills a signal immediately. A signal creates one resting limit order:

- long-entry buy and short-cover buy orders rest below the current market by
  `limitOffsetBps`
- long-exit sell and short-entry sell orders rest above the current market by
  `limitOffsetBps`
- orders fill later only if replay/live price crosses the limit
- stale orders are cancelled after `staleOrderMs`

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

For each configured window, the detector also keeps `candleRanges`, a rolling view of
the last 1000 synthetic candles at that window size. Each completed candle stores
`(high - low) / open * 100`; the latest point reports average percent range, maximum
percent range, current in-progress candle range, and sample count.

The detector also keeps fixed `priceRanges` for `1y`, `3m`, and `2w`. These track
rolling minimum and maximum prices using 5-minute high/low buckets, so long windows do
not retain every raw trade tick. The `3m` window is modeled as 90 days.

## Valley Signal

A valley is detected when the selected window's clamped derivative turns upward:

```text
latest.rateClamped >= 0
previous.rateClamped < 0
```

The default buy source is the 1m window:

```text
legacyValleyPeak.buyDataIndex = 1
```

The default buy confirmation windows are:

```text
legacyValleyPeak.buyConfirmationOffsets = [2, 1]
```

That checks the 30m and 10m windows relative to the 1m source and requires each
confirmation window to still be rising (`rateClamped > 0`). That makes valley entries
stricter, slower, and aligned with broader upward context.

## Peak Signal

A peak is detected when the selected window's clamped derivative turns downward:

```text
latest.rateClamped <= 0
previous.rateClamped > 0
```

The default sell source matches the buy source:

```text
legacyValleyPeak.sellDataIndex = 1
```

The default sell confirmations are:

```text
legacyValleyPeak.sellConfirmationOffsets = [2, 1]
```

That checks the 30m and 10m windows relative to the 1m source and requires each
confirmation window to still be falling (`rateClamped < 0`). This is the
strict-symmetric baseline used by the `0.35%` 180d anchor replay.

The asymmetric short-favoring detector is saved as a reference config. It keeps strict
buys, but uses `sellDataIndex = 0` and `sellConfirmationOffsets = [6]` for faster peak
detection in broader downward context.

## Warmup

The detector records rolling state immediately but refuses to trade before warmup:

```text
legacyValleyPeak.saturationSec = 3600
```

This avoids trading from empty or underfilled rolling windows. During warmup,
`evaluateLegacyValleyPeak` always returns `hold`.

## Entry Leverage

New long entries now use a target leverage procedure. The procedure currently returns
the configured maximum leverage, clamped to a positive account limit:

```text
target entry leverage = maxLeverage
```

The hook is intentionally narrow so future versions can replace the constant with a
range-aware value derived from historical volatility or recent swing width.

Entry buying power is the remaining quote notional that can be added without exceeding
either `maxPositionQuote` or the target leverage guard after fees. Open buy orders count
as committed future long exposure, so multiple resting entries cannot each reuse the
same leverage headroom:

```text
entry buying power =
  min(
    maxPositionQuote - committed long notional,
    target entry leverage * equity - committed long notional, fee-adjusted
  )
```

## Buy Sizing

On a confirmed valley, the strategy computes a desired quote notional from leveraged
entry buying power and the 30m derivative:

```text
desired buy quote =
  entry buying power
  * legacyValleyPeak.buySpendRate
  * gaussian(derivative30m, 0, legacyValleyPeak.buySigma)
```

The Gaussian term is largest when the derivative is near zero and smaller when the
average is moving sharply. That keeps the largest buys near smooth turning points.

The final buy quote is clamped by:

- `legacyValleyPeak.minTradeQuote`
- `legacyValleyPeak.maxTradeQuote`
- entry buying power
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

## Peak Exit Grid Mode

With the current default `legacyValleyPeak.exitGridEnabled = true`, the same
valley/peak detector drives a different execution model:

- a valley opens a new long lot, using a market buy when
  `legacyValleyPeak.exitGridMarketEntry` and `longSideEnabled` are true
- a valley also recreates buy-to-cover ladders for active short lots when price has
  moved far enough below their break-even buy price
- additional valleys can open additional long lots while older lots remain active
- a peak opens a new short lot, using a market sell when
  `legacyValleyPeak.exitGridMarketEntry` is true and `shortSideEnabled` is true
- a peak also recreates sell ladders for active long lots when price has moved far
  enough above their break-even sell price
- the strategy reads active long and short lots from the position ledger and tracks
  entry price, original quantity, highest observed price per long lot, and lowest
  observed price per short lot
- a confirmed peak creates targeted sell orders between each lot's tracked peak and
  break-even sell price
- a confirmed valley creates targeted buy orders between each short lot's tracked trough
  and break-even buy price
- the grid distribution procedure chooses both ladder prices and per-order sizes
- a later peak cancels that lot's open grid orders and recreates its ladder when the
  peak rises above the previous grid peak or crosses back above a filled point from
  the currently active grid
- a later valley mirrors the reset for short lots when price makes a lower trough than
  the previous grid trough or crosses below a filled buy-cover point from the currently
  active grid

The default distribution is a uniform price grid with geometric size decay:

- `exitGridPriceDistribution = "uniform"` spaces prices evenly from peak down to
  break-even or entry.
- `exitGridPriceDistribution = "geometric"` spaces prices by equal percentage ratios
  across the same interval.
- `exitGridSizeDistribution = "geometric"` sells `exitGridSellFraction` of remaining
  base at every non-final level, then sweeps the rest at the final level.
- `exitGridSizeDistribution = "linear"` sells linearly decreasing quantities as the
  ladder moves down from peak toward break-even.
- `exitGridSizeDistribution = "constant"` divides remaining quantity evenly across the
  remaining levels.
- if a planned partial exit would create or place a below-`minOrderQuote` remainder,
  the ladder sweeps the whole remaining lot at that grid level when the full remainder
  is tradable.

The simulator represents these exit-grid orders as sell limit orders with
`trigger = "below"`, so they fill when replay/live price falls through a ladder level.
That models the intended peak-to-entry exit ladder, but it is stop-like behavior rather
than a normal exchange sell limit resting above market.
Short cover ladders mirror this with buy limit orders using `trigger = "above"`, so they
fill when replay/live price rises back through a ladder level after a trough.
After creating a new exit grid, the simulator immediately evaluates the new orders
against the same tick. This lets the extrema-side grid point fill on the signal tick
instead of waiting for the next replay/live price update.

When a short is opened while long lots are active, the ledger records which long lots
supplied internal borrowed base, prioritizing underwater longs. The short sale proceeds
immediately reduce those long lots' remaining cost basis and break-even, without
treating the short open as a long close. Buying that short back charges the cover cost
back to the same source longs, so the completed borrow cycle applies the short's net
profit or loss to the long cost basis.

Long entries mirror this with quote borrowed from active short lots. The borrowed quote
is tied to the base quantity it funded, and closing that long returns the corresponding
sale value to the same source shorts. Borrow chains are limited by `longBorrowDepth`
for chains that start from a long lender and `shortBorrowDepth` for chains that start
from a short lender.

Entry capacity is split into two buckets. Free-capital capacity is limited by the
per-side cap and leverage headroom. Internal borrow capacity is the sum of currently
unlent opposite-side lots and is added on top of free-capital capacity. That lets a
short entry borrow base from several long lots, or a long entry borrow quote from
several short lots, when one lender lot is too small for the desired entry. Settlement
already follows the recorded allocation list, so borrower profit or loss is returned to
each source lot proportionally.

When `lockBorrowedLenderCollateral` is enabled, a lender lot cannot place exit-grid
orders against collateral that is currently lent to a borrower. Long lenders lock
`lentQuantity`; short lenders lock the cover quantity represented by `lentQuote` at
the current break-even grid price. `borrowerProfitShareToLender` controls how much of a
profitable borrower close is credited back into the lender's lot basis. `1` preserves
the older behavior where the lender receives the full borrower profit, while `0` keeps
the borrower's profit free at the account level instead of lowering the lender's
break-even further. Borrower losses are still charged back through the lender basis.

## Current Defaults To Watch

| Config | Default | Meaning |
| --- | ---: | --- |
| `algorithm` | `legacy-valley-peak` | The only automated algorithm key. |
| `maxLeverage` | `1` | Baseline leverage guard from the strict-symmetric `0.35%` anchor replay. |
| `shortMarginModel` | `futures-margin` | Collateral-backed short accounting used by the current baseline. |
| `longBorrowDepth` | `999` | Number of alternating internal borrow hops allowed from an original long lender. |
| `shortBorrowDepth` | `999` | Number of alternating internal borrow hops allowed from an original short lender. |
| `lockBorrowedLenderCollateral` | `false` | Lender lots can still exit-grid collateral currently lent to borrowers. |
| `borrowerProfitShareToLender` | `1` | Fraction of profitable borrower closes credited back to the lender lot basis. |
| `maxPositionQuote` | `10000` | Maximum notional the strategy can build per side; matches the `$10000` starting quote. |
| `limitOffsetBps` | `2` | Distance from current price for new limit orders. |
| `maxOpenOrders` | `1024` | Maximum number of resting automated orders across entries and exit-grid ladders. |
| `cooldownMs` | `300000` | Minimum delay between newly created automated orders. |
| `staleOrderMs` | `2592000000` | Time before an unfilled limit order is cancelled; current grid ladders can persist for 30 days. |
| `minOrderQuote` | `5` | Global minimum order notional. |
| `legacyValleyPeak.saturationSec` | `3600` | Rolling detector warmup before signals can trade. |
| `legacyValleyPeak.maxTradeQuote` | `50000` | Per-signal quote/notional clip. |
| `legacyValleyPeak.longSideEnabled` | `true` | Allows confirmed valleys to open long lots and confirmed peaks to close them. |
| `legacyValleyPeak.shortSideEnabled` | `true` | Allows confirmed peaks to open short lots and confirmed valleys to cover them. |
| `legacyValleyPeak.buySigma` | `0.3` | Buy Gaussian width around the sizing derivative. |
| `legacyValleyPeak.sellSigma` | `0.1` | Sell Gaussian width around the sizing derivative. |
| `legacyValleyPeak.exitGridEnabled` | `true` | Enables the peak-to-entry exit ladder. |
| `legacyValleyPeak.exitGridMarketEntry` | `true` | Uses immediate market-style valley entries in grid mode. |
| `legacyValleyPeak.exitGridOrderCount` | `200` | Maximum orders in a recreated exit ladder. |
| `legacyValleyPeak.exitGridMaxStepPct` | `0.006` | Maximum target grid price step as a percentage of current price; `0` disables this cap. |
| `legacyValleyPeak.exitGridPriceDistribution` | `uniform` | Spacing procedure for ladder prices. |
| `legacyValleyPeak.exitGridSizeDistribution` | `geometric` | Sizing procedure for ladder quantities. |
| `legacyValleyPeak.exitGridSellFraction` | `0.35` | Fraction of remaining base sold by each non-final grid order. |
| `legacyValleyPeak.exitGridMinProfitBps` | `20` | Minimum peak distance above break-even before creating a ladder. |
| `legacyValleyPeak.exitGridResetBps` | `10` | Higher-peak improvement used only when manually selecting strict reset mode. |
| `legacyValleyPeak.exitGridPositionMode` | `per-lot` | Uses position-ledger lots for independent ladders. |
| `legacyValleyPeak.exitGridResetMode` | `filled-grid` | Resets after a new higher/lower extreme or after crossing a filled point from the active grid. |

## Current Default Backtests

### Short-Side Comparison

Focused comparisons after adding mirrored short entries and buy-to-cover grids. All rows
use relaxed per-lot filled-grid mode, `$10000` starting quote, `300s` cooldown, and no
liquidations. Historical rows use local BTCUSDT `1m` candles from 2026-05-23 to
2026-06-21. Synthetic rows use 30 days of default sine-plus-noise candles with seed
`1337`. These historical rows used the then-default `spot-borrow` short margin model.
The `1x` short-only rows have zero fills because the spot-margin debt guard rejects
borrowed-base short entries at `1x`; use `shortMarginModel = "futures-margin"` to test
standalone collateral-backed shorts without leverage.

| Source | Max lev | Strategy | Return | Max DD | Risk Ret | Sharpe | Trades | Profitable closed positions | Oracle Capture | Reinvest Cap |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical | `1x` | Long/Short | `-7.90%` | `12.10%` | `-0.653` | `-3.488` | `347` | `19/73 (26.0%)` | `-13.146%` | `-9.610847%` |
| Historical | `1x` | Long Only | `-9.85%` | `17.91%` | `-0.550` | `-2.546` | `45` | `3/10 (30.0%)` | `-16.386%` | `-11.979724%` |
| Historical | `1x` | Short Only | `0.00%` | `0.00%` | `-` | `-` | `0` | `0/0 (0.0%)` | `0.000%` | `0.000000%` |
| Historical | `5x` | Long/Short | `-44.46%` | `78.27%` | `-0.568` | `0.134` | `105` | `7/13 (53.8%)` | `-14.791%` | `-2.379744%` |
| Historical | `5x` | Long Only | `-49.49%` | `84.47%` | `-0.586` | `0.691` | `59` | `4/12 (33.3%)` | `-16.464%` | `-2.648954%` |
| Historical | `5x` | Short Only | `9.34%` | `16.34%` | `0.572` | `2.237` | `283` | `30/30 (100.0%)` | `3.106%` | `0.499801%` |
| Synthetic | `1x` | Long/Short | `61.48%` | `19.40%` | `3.170` | `4.945` | `992` | `132/263 (50.2%)` | `-` | `-` |
| Synthetic | `1x` | Long Only | `64.32%` | `22.53%` | `2.855` | `4.579` | `996` | `131/281 (46.6%)` | `-` | `-` |
| Synthetic | `1x` | Short Only | `0.00%` | `0.00%` | `-` | `-` | `0` | `0/0 (0.0%)` | `-` | `-` |
| Synthetic | `5x` | Long/Short | `319.69%` | `50.91%` | `6.280` | `5.535` | `2213` | `340/503 (67.6%)` | `-` | `-` |
| Synthetic | `5x` | Long Only | `299.73%` | `69.57%` | `4.308` | `5.706` | `1661` | `255/363 (70.2%)` | `-` | `-` |
| Synthetic | `5x` | Short Only | `16.30%` | `87.77%` | `0.186` | `4.806` | `959` | `16/16 (100.0%)` | `-` | `-` |

With `shortMarginModel = "futures-margin"` and `1x` max leverage, the short-only row
fills normally: the same historical window returned `2.24%` with `8.22%` drawdown over
`134` fills, and the same synthetic setup returned `2.31%` with `28.80%` drawdown over
`335` fills. Current relaxed long/short at `1x` under `futures-margin` returned
`4.65%` with `5.46%` drawdown on the historical window and `4.65%` with `15.34%`
drawdown on the synthetic series. All checks had zero liquidations.

The broader reference rows below were captured before the short-side default changed.
They use relaxed per-lot exit grid with filled-grid resets, `5x` max leverage, `$10000`
starting quote, `$50000` target position cap, `$39200` initial debt cap, account-level
liquidation, and a `300s` cooldown. Historical rows use local `BTCUSDT` `1m` candles
through June 21, 2026. Random and synthetic runs use seed `1337`.
`Reinvest Cap` is capture versus the compounded perfect-margin upper bound; `-` means
the corresponding oracle PnL was not positive.

### Historical BTCUSDT

| Backtest | Window | Return | Max DD | Risk Ret | Sharpe | Trades | Profitable closed positions | Liquidated positions | Oracle Capture | Reinvest Ret | Reinvest Cap | Extra |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Fixed 7d | 2026-06-15 to 2026-06-21 | `9.45%` | `15.98%` | `0.591` | `4.298` | `48` | `6/7 (85.7%)` | `0` | `35.941%` | `29.88%` | `31.638647%` | - |
| Fixed 30d | 2026-05-23 to 2026-06-21 | `-49.49%` | `84.47%` | `-0.586` | `0.691` | `59` | `4/12 (33.3%)` | `0` | `-16.464%` | `1868.20%` | `-2.648954%` | - |
| Fixed 1y | 2025-06-22 to 2026-06-21 | `-100.29%` | `100.16%` | `-1.001` | `-0.417` | `4,066` | `555/1,081 (51.3%)` | `30` | `-5.883%` | `1.68e+9%` | `-0.000006%` | stopped by account liquidation |
| Random 24 windows | 2021-07-16 to 2026-06-21 cache | `6.56% avg` | `46.28% avg` | `0.653 avg` | `1.012 avg` | `602.0 avg` | `88.0/126.1 (59.3%) avg` | `1.4 avg` | `1.875% avg` | - | `1.141360% avg` | `14/24` profitable, median `6.11%`, P10 `-87.62%`, best `145.65%`, worst `-100.30%` |
| Six folded windows | 2021-07-16 to 2026-06-21 | `110.16% avg` | `63.41% avg` | `4.095 avg` | `0.440 avg` | `2818.5 avg` | `426.3/660.7 (64.4%) avg` | `7.0 avg` | `0.689% avg` | - | `-0.000530% avg` | `4/6` profitable, worst fold `-100.30%` |

### Synthetic Regimes

Synthetic candles use start price `$100000`, amplitude `8%`, and 30 days of generated
1m OHLC data. The Brownian component is sampled at candle open, midpoint, and close, so
the detector sees realistic close/open boundary movement instead of a perfectly
continuous curve.

| Regime | Frequency | Trend | Daily noise | Return | Max DD | Risk Ret | Sharpe | Trades | Profitable closed positions | Liquidated positions | Oracle Capture | Reinvest Ret | Reinvest Cap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sideways, low noise | `2/day` | `0.00%/day` | `0.50%` | `31.83%` | `45.15%` | `0.705` | `2.762` | `180` | `22/43 (51.2%)` | `0` | `-` | `0.00%` | `-` |
| Sideways, medium noise | `2/day` | `0.00%/day` | `2.00%` | `232.70%` | `40.40%` | `5.760` | `5.565` | `1,208` | `161/275 (58.5%)` | `0` | `-` | `0.00%` | `-` |
| Sideways, high noise | `2/day` | `0.00%/day` | `6.00%` | `233.89%` | `61.39%` | `3.810` | `5.364` | `1,660` | `194/402 (48.3%)` | `0` | `37.848%` | `47371.63%` | `0.493738%` |
| Uptrend, low noise | `1.5/day` | `0.30%/day` | `0.50%` | `148.32%` | `47.80%` | `3.103` | `4.634` | `987` | `96/188 (51.1%)` | `0` | `-` | `0.00%` | `-` |
| Uptrend, medium noise | `1.5/day` | `0.30%/day` | `3.00%` | `154.89%` | `57.49%` | `2.694` | `4.621` | `567` | `63/109 (57.8%)` | `0` | `-` | `0.00%` | `-` |
| Uptrend, high noise | `1.5/day` | `0.30%/day` | `6.00%` | `172.45%` | `50.79%` | `3.396` | `4.854` | `777` | `92/144 (63.9%)` | `0` | `43.440%` | `5146.51%` | `3.350728%` |
| Downtrend, low noise | `1.5/day` | `-0.30%/day` | `0.50%` | `1063.27%` | `48.73%` | `21.822` | `9.202` | `7,700` | `1,150/1,693 (67.9%)` | `0` | `-` | `0.00%` | `-` |
| Downtrend, medium noise | `1.5/day` | `-0.30%/day` | `3.00%` | `487.60%` | `58.74%` | `8.300` | `6.462` | `2,472` | `338/594 (56.9%)` | `0` | `74529.797%` | `0.66%` | `74341.579440%` |
| Downtrend, high noise | `1.5/day` | `-0.30%/day` | `6.00%` | `574.71%` | `56.06%` | `10.252` | `6.942` | `2,877` | `362/718 (50.4%)` | `0` | `70.238%` | `349075.91%` | `0.164637%` |

## Known Limitations

Legacy valley/peak is a heuristic turning-point detector. It can buy early in a falling
trend, miss fast reversals when limit orders do not fill, and churn when local averages
oscillate around zero after fees. It does not currently use funding, spread, order-book
depth, liquidation data, open interest, learned labels, or regime-aware position sizing.

Perfect-margin capture remains the north-star benchmark, not an achievable target.
