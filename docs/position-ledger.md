# Trading Model and Position Ledger

The bot keeps one persisted paper-trading state and derives the position ledger from
that state. The persisted state is the source of truth for balances, orders, fills,
strategy memory, and metrics. The ledger is a reconstructed view used by position-risk
helpers, benchmark metrics, and presentation layers.

The implementation lives in:

- `packages/bot-algo/src/bot.ts` for account state, order lifecycle, fills, leverage
  checks, and strategy execution
- `packages/bot-algo/src/position-ledger.ts` for derived long/short lots, break-even
  prices, borrow attribution, and risk suggestions

## Account State

The account is modeled as base and quote balances:

- `quoteFree` is spendable quote.
- `quoteReserved` is quote reserved by open buy orders, including estimated fees.
- `baseFree` is sellable base.
- `baseReserved` is base reserved by open sell orders.
- `avgEntryPrice` is the aggregate long entry price used for simple PnL accounting.
- `avgShortEntryPrice` is inferred from derived short lots.

Balances can go negative. A negative quote balance represents borrowed quote used to
hold leveraged long exposure. A negative net base balance represents borrowed base used
to hold short exposure. Equity is still computed from marked-to-market balances:

```text
equity = quoteFree + quoteReserved + (baseFree + baseReserved) * currentPrice
```

The state model can represent both long and short lots. Which side a specific strategy
opens is strategy behavior, not part of the ledger model.

## Orders and Fills

Orders are persisted in `state.orders`; fills are persisted in `state.fills`.

An order has:

- `side`: `buy` or `sell`
- `type`: `limit` or `market`
- optional `trigger`: `above` or `below`
- `status`: `open`, `filled`, or `cancelled`
- `price`, `quantity`, estimated quote cost, timestamps, reason, and optional target lot

Open order accounting is reserve-based:

- creating a buy order reserves estimated quote cost and moves it from `quoteFree` to
  `quoteReserved`
- creating a sell order reserves base and moves it from `baseFree` to `baseReserved`
- cancelling an order releases its reserve
- filling an order consumes the reserve, creates a `TradeFill`, updates balances, and
  records fees

When `trigger` is absent, limit orders use exchange-like crossing rules:

- buy fills when price moves at or below the order price
- sell fills when price moves at or above the order price

When `trigger` is present, it controls the crossing direction directly:

- `trigger = "below"` fills when price moves at or below the order price
- `trigger = "above"` fills when price moves at or above the order price

This lets strategy-specific management represent stop-like orders without changing the
ledger accounting model.

Market orders are modeled as orders that are immediately filled on the same tick. If
the leverage guard rejects the fill, the state rolls back and the order is cancelled.

## Position Lots

The position ledger is rebuilt by replaying fills in chronological order, then adding
currently open orders as pending lots.

Long lots:

- come from filled buys that are not consumed by earlier short lots
- store original quantity, remaining quantity, cost quote including fee, average price,
  closed quantity, and allocated sell proceeds
- close when later sells are allocated to them

Short lots:

- come from filled sells that are not consumed by earlier long lots
- store original quantity, remaining quantity, proceeds quote after fee, average price,
  closed quantity, and allocated buyback cost
- close when later buys are allocated to them

Open buy orders appear as pending long lots. Open sell orders appear as pending short
lots. Pending lots use the limit price and estimated fee-adjusted quote amount so
callers can see the break-even levels that would apply if the order fills. Pending lots
are reported separately in ledger summary totals; they are not counted as active
exposure until filled.

## Fill Allocation

Fill allocation is controlled by `targetPositionId` and `positionEffect`.

- If `targetPositionId` is present, the fill first closes that specific lot.
- If `positionEffect` is `open`, any remaining quantity opens a new lot even if
  opposite-side lots exist.
- Otherwise, the fill closes opposite-side lots in chronological order before any
  leftover quantity opens a new lot.

This gives callers two behaviors:

- targeted closes, used by explicit close fills
- ordinary netting, used when a non-targeted fill should reduce existing opposing
  exposure before opening new exposure

Callers can force a new lot with `positionEffect = "open"` or close a specific lot with
`targetPositionId`.

## Leverage and Borrowing

The account does not model exchange margin buckets directly. It derives debt from
balances and attributes that debt back to lots for inspection.

`shortMarginModel` controls how standalone shorts count against the leverage guard:

- `spot-borrow` is the default. A short is modeled as borrowed base, so an unhedged
  short at `1x` has no headroom because the borrowed base value counts as external
  debt.
- `futures-margin` treats shorts as collateral-backed futures-style exposure. Account
  balances are still represented the same way, but summary leverage and fill rejection
  use gross notional exposure over equity.

In `spot-borrow`, summary leverage uses external borrowed value:

```text
effective leverage = 1 + externalBorrowedQuote / equity
```

In `futures-margin`, summary leverage uses gross exposure:

```text
effective leverage = grossExposureQuote / equity
```

Lot leverage uses the portion of that lot's exposure that required external borrowing:

```text
lot leverage = exposureQuote / (exposureQuote - externalBorrowedQuote)
```

Borrow attribution distinguishes internal and external borrowing:

- A short can borrow base internally from active long lots; any remaining borrowed base
  is external. Internal short borrow records the source long lot without closing that
  long.
- Opening a borrowed short credits its sale proceeds to the source long lot's remaining
  cost, which lowers that long's break-even immediately.
- Buying the borrowed short back charges the cover cost back to the same source long.
  The completed effect is the short's realized profit or loss applied to the long cost
  basis.
- A long can use owned quote capital first, then borrow quote internally from active
  short proceeds, then borrow quote externally if the account quote balance is negative.
- In `spot-borrow`, `externalBorrowedQuote` is the amount that counts toward effective
  leverage.
- In `futures-margin`, `externalBorrowedQuote` is still shown for inspection, but gross
  exposure controls effective leverage.
- `internalBorrowedQuote` and `internalBorrowedQuantity` show how much opposing
  position value is being reused inside the account model.

The leverage guard runs after every fill. In `spot-borrow`, it uses a fast debt-leverage
estimate before rebuilding the full ledger if needed. In `futures-margin`, it rebuilds
the ledger and checks gross exposure directly. If effective leverage is above
`maxLeverage`, the fill is rolled back and the order is cancelled with a leverage-limit
reason.

## Account Liquidation

Backtests model liquidation at account level, not per position. The liquidation check
uses aggregate balances, so long and short exposure can compensate for each other
through net base and quote.

For a net long account with borrowed quote:

```text
liquidation price = -quote balance / net base quantity
```

For a net short account with borrowed base:

```text
liquidation price = quote balance / -net base quantity
```

If replay price crosses that account liquidation price, the simulator cancels open
orders, closes all active exposure at the liquidation price, marks the fills as
liquidations, and records how many active ledger positions were liquidated. This is an
equity-zero model; it does not include maintenance margin tiers or exchange-specific
buffer rules.

## Risk Fields

The ledger mirrors the `sol` spreadsheet section from rows 111 onward. It computes
break-even and suggested partial-close levels for each lot. These fields are advisory;
the ledger does not execute the recommended risk actions by itself.

Long lots use the spreadsheet recovery model:

```text
average buy price = cost quote / bought quantity
remaining cost = original cost - allocated sell proceeds
break-even sell price = remaining cost / max(remaining quantity, quantity floor) * (1 + fee + slippage)
max-loss sell price = max(0, remaining cost - max loss pct * original cost) / max(remaining quantity, quantity floor) * (1 + fee + slippage)
sell-now quote = max(0, (remaining cost - lower baseline * remaining quantity) / (net market sell - lower baseline) * net market sell)
can reach lower baseline = sell-now quote > 0 and sell-now quote < remaining cost and quantity is available
```

Short lots use the mirrored sell-side formulas:

```text
average sell price = proceeds quote / sold quantity
remaining proceeds = original proceeds - allocated buyback cost
break-even buy price = remaining proceeds / max(remaining quantity, quantity floor) / (1 + fee + slippage)^2
max-loss buy price = remaining proceeds / max(remaining quantity - max loss pct * original quantity, quantity floor) / (1 + fee + slippage)^2
buy-now quote = max(0, (remaining proceeds - upper baseline * remaining quantity) / (gross market buy - upper baseline) * gross market buy)
can reach upper baseline = buy-now quote > 0 and buy-now quote < remaining proceeds and quantity is available
```

The default quantity floor is `0.01`, matching the spreadsheet guard against
divide-by-zero rows. Fee is the bot configured Binance fee. Slippage is a separate
market-order estimate.

## Workflow Boundaries

Manual fills are represented by the same order and fill records as automated fills, with
`manual: true`. They are replayed by the same ledger allocator and can create or close
the same lot types.

The operator controls for creating those fills are not part of the trading model. They are
documented in [Manual Position Management](manual-position-management.md).

Automated strategy management is also outside the neutral ledger model. The current
strategy behavior is documented in
[Automated Position Management](automated-position-management.md).

## Known Gaps

- Funding, borrow interest, exchange margin tiers, and maintenance margin are not
  modeled yet.
- The account-level average entry fields are aggregate conveniences. Per-lot behavior
  comes from the fill-derived ledger.
- Internal borrowing is a paper-accounting operation, not a live exchange borrow or
  transfer operation.
