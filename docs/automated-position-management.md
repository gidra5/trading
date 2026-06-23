# Automated Position Management

This document describes how automated strategy actions use the trading model. It is
separate from the neutral account and lot model in
[Trading Model and Position Ledger](position-ledger.md). Manual operator controls are
documented in [Manual Position Management](manual-position-management.md).

## Current Scope

The only automated strategy is `legacy-valley-peak`. Its current default execution
mode is long-only relaxed per-lot peak exit grid:

- it opens long lots at confirmed valleys
- it waits for confirmed peaks
- it manages each active long lot with its own resettable partial-exit ladder

The automated strategy does not open shorts. Short lots can exist through explicit
paper fills, but they are not managed by this strategy today.

## Entry Management

Confirmed valleys can open new long lots.

In the default grid mode:

- valley entries are market-style immediate buys
- entry fills are marked with `positionEffect = "open"`
- every accepted valley creates a distinct long lot
- additional valleys may open additional lots while older lots remain active

Entry size is clipped by:

- `legacyValleyPeak.minTradeQuote`
- `legacyValleyPeak.maxTradeQuote`
- `maxPositionQuote`
- target leverage buying power
- open-order limits
- cooldown

New automated long entries use a target entry leverage procedure. It currently returns
`maxLeverage`, so the default target is `5x`.

Entry buying power is capped by both `maxPositionQuote` and the leverage guard after
fees. Open buy orders count as committed future long exposure, so multiple resting
entries cannot reuse the same headroom.

## Lot Tracking

The strategy reads active long lots from the fill-derived ledger model. For performance,
the bot keeps an incremental long-lot cache while replaying backtests and live ticks.

Each tracked lot contributes:

- lot id
- average entry price
- original quantity
- remaining quantity
- remaining cost quote
- break-even sell price

When a targeted sell fills, the remaining quantity and break-even price for that lot are
updated. When the remaining quantity reaches the base floor, the lot is removed from the
active set.

## Peak Exit Grid

Confirmed peaks inspect all active long lots.

Each lot has exit-grid memory:

- `lotId`
- `entryPrice`
- `entryQuantity`
- highest observed `peakPrice`
- last `gridPeakPrice`
- open `gridOrderIds`

A grid is created only when the tracked peak is above break-even by at least
`exitGridMinProfitBps`.

Grid sell levels span from the tracked peak down to the greater of:

- the lot entry price
- the lot break-even sell price

The grid distribution procedure controls price spacing and order size:

- `exitGridPriceDistribution = "uniform"` spaces prices linearly across the interval.
- `exitGridPriceDistribution = "geometric"` spaces prices by equal percentage ratios.
- `exitGridSizeDistribution = "geometric"` sells `exitGridSellFraction` of remaining
  base at each non-final level and closes the rest at the final level.
- `exitGridSizeDistribution = "linear"` sells linearly decreasing quantities from the
  peak level down to break-even.
- `exitGridSizeDistribution = "constant"` divides remaining quantity evenly across the
  remaining grid levels.

The current default is uniform prices with geometric size decay.

Exit-grid orders are targeted closes:

- they carry the long lot id as `targetPositionId`
- they use `positionEffect = "close"`
- their fills allocate proceeds back to that exact lot

## Grid Reset

The current default reset mode is `filled-grid`.

If price later crosses above at least one already filled grid point for a lot, the bot:

1. Cancels that lot's open grid orders.
2. Releases the reserved base from those cancelled orders.
3. Rebuilds the ladder from the new peak-to-break-even interval.

The stricter `higher-peak` reset mode still exists for comparison. In that mode the
tracked peak must exceed the previous grid peak by `exitGridResetBps` before the ladder
is reset.

When a lot closes, its exit-grid memory is removed.

## Trigger Behavior

Exit-grid sell orders are represented as sell limit orders with `trigger = "below"`.
They fill when replay or live price falls through the ladder level. This models the
intended peak-to-entry stop ladder.

That is different from a normal exchange sell limit resting above market. The current
simulator behavior may overstate fills around gaps because OHLC replay only approximates
intra-candle path.

## Older Non-Grid Mode

The older non-grid mode still exists behind `legacyValleyPeak.exitGridEnabled = false`.

In that mode:

- a valley creates a resting buy limit order below market
- a peak creates a resting sell limit order above market
- orders fill later only if price crosses the limit
- stale orders are cancelled after `staleOrderMs`
- sells are not targeted to specific lots

Because non-grid sells are not targeted, their fills close active long lots through
ordinary chronological allocation.

## Boundaries

- Automated management only handles long lots today.
- It does not automatically manage manual short lots.
- It does not execute the ledger's advisory risk recommendations directly.
- Account-level liquidation is simulated by the backtest account model, not by strategy
  decision logic.
- It does not model funding, borrow interest, maintenance margin, or exchange margin
  tiers.
