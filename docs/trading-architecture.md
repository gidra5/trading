# Trading contracts

The bot depends on the small provider-neutral surface in `trading-api.ts`. Live,
paper, and backtest implementations bind one API instance to one market, so no
method accepts a symbol.

## Direction

There are two directional concepts, but only one belongs to orders:

- order side: `buy` or `sell`;
- position side: `long` or `short`.

The bot interprets an order against its target position:

| Order | Position | Meaning |
| --- | --- | --- |
| buy | long | open/increase long |
| sell | long | close/decrease long |
| sell | short | open/increase short |
| buy | short | close/decrease short |

Orders contain only buy/sell. Long/short is stored on positions and never crosses the
trading API boundary. There is no separate position-effect type. Stop direction
is conventional and implicit: buy stops trigger above and sell stops below.

## Orders and events

Every provider adapter assigns one bot-visible order ID, including locally held
overflow orders and rejected submissions. Exchange-specific IDs remain private
to the server adapter.

Order creation returns accepted/rejected plus the current order state. Rejection
details stay in provider/server logs; the bot only needs the order ID. Stop
orders may return `pending`. The bot receives only four asynchronous updates:
`open`, `partial-fill`, `fill`, and `rejected`. Cancellation is handled directly
from `cancelOrder`'s boolean result.

Fill events contain only exact net asset and quote amounts plus remaining size.
Fees and execution friction are already reflected in the net amounts. Price is
derived when needed, and `partial-fill` versus `fill` supplies terminal state.

## Market and account inputs

Tick quantity is required; synthetic/backtest ticks use the simulated quantity
or zero. `tick.candle` is non-null only when a tick represents a candle update or
replay step, allowing candle-based indicators to consume the same bot callback.

History requests contain only interval and count. `EquitySnapshot` exposes only
available, reserved, and explicitly named unleveraged amounts for quote and asset.
`TradingMarketRules` exposes provider price, quantity, notional, and leverage
constraints without exposing symbol or exchange-native contract details.
Friction is one proportional number returned by `getFriction()`.

## Strategy

The strategy consumes market ticks and owns its indicator state. It does not
receive equity, friction, positions, or orders. Entry and exit signals contain a
size in `[0, 1]`: entry size is a fraction of available execution capacity, while
exit size is a fraction of the matching bot-owned position.

Signal reasons are not execution data. They stay in strategy diagnostics or a
separate reporter. During `warmup()`, the concrete strategy uses its injected
`getHistory` function to fetch whatever candles its indicators require; the bot
only delegates to `strategy.warmup()`.

## Ownership

The provider/server owns symbols, credentials, exchange IDs, filters, leverage
limits, balances, reconciliation, liquidation, persistence, and account/market
metrics. The bot owns config, active orders, fill-derived positions, entry/exit
grids, internal borrow attribution, strategy snapshots, and raw diagnostics.
Position and lot mean the same thing; the code uses only `TradingPosition`.
The bot creates a position together with its entry grid; its asset and quote are
zero until the first fill. If all entry orders are rejected/cancelled, it removes
the empty position. Fully closed positions are also removed and retained by the server.

Every order lives in a position entry or exit grid, including a single market
order represented as a one-order grid. Grid orders keep one `filled` amount: quote
for long entry/short exit, and asset for short entry/long exit. There is no
separate persisted orders collection; implementations may build an in-memory ID index.

Internal lending optionally locks exactly the asset and quote amounts recorded in
each `PositionBorrow`. When locking is disabled, closing a lender resolves or
forgives the affected internal debt.

The server owns the single complete default bot config. The bot and strategy do
not contain hidden defaults or provider settings.
