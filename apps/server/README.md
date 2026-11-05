# Server

Run from the repository root:

```sh
npm run dev -w @trading/server
```

Build and start:

```sh
npm run build -w @trading/server
npm start
```

The server listens on `0.0.0.0:3001` by default. Set `HOST` and `PORT` to
override the address. `GET /health` returns `{ "status": "ok" }`.

WebSocket connections use `ws://localhost:3001/ws` on the same server.
`src/webclient.ts` wraps each connection for server-to-client JSON status messages.
From the frontend, use `/backend/ws` through the Vite proxy:

```js
const url = new URL("/backend/ws", window.location.href);
url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
const socket = new WebSocket(url);
socket.onmessage = (event) => {
  const { type, message } = JSON.parse(event.data);
  console.log(type, message);
};
```

`SIGINT` and `SIGTERM` close the server gracefully. Startup failures exit
with a nonzero status.

## Endpoint schemas

`src/schemas.ts` defines Zod request and response schemas and their inferred
TypeScript types. `src/routes.ts` registers the routes and their schemas;
`src/app.ts` provides the default application context.
`src/index.ts` configures Fastify, Zod validation and serialization, WebSocket
support, error handling, and the process lifecycle. Handlers receive validated
bodies and their responses are checked before serialization.

| Endpoint | JSON body | Success response |
| --- | --- | --- |
| `GET /health` | None | `{ "status": "ok" }` |
| `POST /bot/start` | `{ "symbol": "BTCUSDT" }` | `{ "symbol": "BTCUSDT", "status": "running" }` |
| `POST /bot/stop` | `{ "symbol": "BTCUSDT" }` | `{ "symbol": "BTCUSDT", "status": "stopped" }` |
| `POST /bot/reset` | `{ "symbol": "BTCUSDT" }` | `{ "symbol": "BTCUSDT", "status": "reset" }` |
| `POST /bot/position/open` | `{ "symbol": "BTCUSDT", "side": "long", "quantity": 0.1 }` | `{ "symbol": "BTCUSDT", "status": "accepted" }` |
| `POST /bot/position/close` | `{ "symbol": "BTCUSDT", "side": "long", "quantity": 0.1 }` | `{ "symbol": "BTCUSDT", "status": "accepted" }` |
| `POST /bot/configure` | `{ "network": "demo", "credentials": { "apiKey": "...", "apiSecret": "..." } }` | `{ "network": "demo", "status": "configured" }` |

`network` accepts `live`, `demo`, or `testnet`. Symbols allow hyphens for option contracts. Request objects reject unknown fields.
Invalid bodies return HTTP 400. Invalid responses return HTTP 500. Error
responses use `{ statusCode, error, message }`; internal error details stay in
server logs. Response objects omit undeclared fields, including credentials.

Supply an `ApiContext` from `src/app.ts` as the second argument to
`createServer(options, context)` exported by `src/index.ts`
to connect the route handlers to `context.bot.start`, `.stop`, `.reset`,
`.configure`, `.openPosition`, and `.closePosition`.
The default `/bot/configure` handler persists configuration. Other bot controls
return HTTP 501 until connected to the runtime. Position requests accept
`side: "long" | "short"` and a positive numeric `quantity`. Opening also accepts
an optional `lifecycle` with `lifetimeMs`, `stopLossPrice`, and `takeProfitPrice`;
closing accepts an optional `positionId`. Acceptance is not confirmation of a fill.

`context.addWebClient(client)` registers a WebSocket and returns a cleanup
function called when it disconnects. Send updates through `client.botStatus(payload)`,
`.connectionStatus(payload)`, `.marketStatus(payload)`, `.backtestStatus(payload)`,
or `.accountStatus(payload)`. Each sends `{ type, message }`; sends after closing
return `false`. The default context owns a separate connection registry per server.
`/ws` does not echo incoming messages or use HTTP body/response schemas.

To add a route, export its Zod schemas and register them as
`schema: { body: RequestSchema, response: { 200: ResponseSchema, ...ErrorResponseSchemas } }`.
Fastify infers the request and response types automatically; handlers do not
need a separate `.parse(request.body)` call. This uses the
[Fastify Zod type provider](https://github.com/turkerdev/fastify-type-provider-zod).

## SQLite database

The server uses Drizzle ORM with `better-sqlite3`. Startup opens
`data/trading.sqlite` at the repository root and applies the committed migrations
in `apps/server/drizzle`. Shutdown closes the connection. SQLite uses WAL mode,
foreign-key enforcement, a five-second busy timeout, and full synchronization.
Set `DATABASE_PATH` to use another file; relative paths resolve from the process's
working directory. Keep `apps/server/drizzle` alongside `dist` when deploying.

`BotSettingsSchema` and `BotStateSchema` in `src/schemas.ts` are the source of
the database record types. `src/db/schema.ts` declares the tables and SQL
constraints. The column helpers in `src/db/columns.ts` infer their types from
Zod and parse values on reads and writes, including nested JSON configuration.
Raw SQL bypasses Zod, so database constraints remain explicit.

- `bot_settings`: one configuration record, reusing `BotConfigureRequestSchema`.
- `bot_states`: desired activity per symbol and its update timestamp. A saved
  active flag does not automatically start a bot or report it as running.

Use `createBotStore(db)` for configuration and state updates, or the typed Drizzle
`db` for queries and transactions. `createContext(store)` exposes this store to
the application. The SQLite file contains API credentials in unencrypted JSON;
keep its directory or mounted volume private. Database and WAL files are ignored
by Git.

From the repository root:

```sh
npm run db:generate -w @trading/server -- --name describe_change
npm run db:migrate -w @trading/server
npm run db:studio -w @trading/server
```

After changing the Zod models and table mappings, generate and review a migration.
`db:migrate` applies pending migrations explicitly; normal server startup also
applies them. Commit SQL migrations and their `meta` files together.
Tests can use `openDatabase(":memory:")` or
`createServer({ databasePath: ":memory:" })` for isolation.

See the [Drizzle SQLite documentation](https://orm.drizzle.team/docs/get-started-sqlite).

## Trading adapters

### Binance

The `binance` Node SDK supplies transport and signing. `BinanceClient` owns
lazy SDK clients; use `.market()` to bind several symbols to one connection.

```ts
import { BinanceClient } from "./clients/binance.js";

const client = new BinanceClient({
  environment: "demo", // "live" or "testnet" also available for Spot/USD-M
  apiKey: process.env.BINANCE_API_KEY!,
  apiSecret: process.env.BINANCE_API_SECRET!,
});
const futures = await client.market({ product: "futures", symbol: "BTCUSDT", frictionDecayRate: 5 });
const spot = await client.market({ product: "spot", symbol: "BTCUSDT" });
const eth = await client.market({ product: "futures", symbol: "ETHUSDT" });

await futures.initialize();
const history = await futures.getHistory({ intervalMs: 60_000, count: 2000 });
const equity = await futures.getEquity();
const ticks = await futures.getTicks({ startTime: Date.now() - 60_000, endTime: Date.now() - 1 });
```

For one symbol, `await getBinanceTradingApi({ ...connection, product, symbol })`
from `./clients/binance.js` returns the product adapter. Spot, futures, and options
load and validate metadata in their static async `create()` factories before
constructing the adapter. `new BinanceClient(...)` itself sends no requests.
Repeated `.market()` calls for these products share creation and the resulting adapter; failed
creation can be retried by calling `.market()` again.
Use credentials issued for the selected environment and product.

| Product | Environments in this adapter | Orders | Account model |
| --- | --- | --- | --- |
| `futures` | live, demo, testnet | Market, limit, stop-market, stop-limit | USD-M, one-way, cross margin, single-asset collateral |
| `spot` | live, demo, testnet | Market, limit, stops allowed by the symbol | Free/locked base and quote balances; no borrowing |
| `options` | live | Limit buys and `reduceOnly` limit sells | Premium-funded, contract unit 1; no uncovered writing |
| `predictions` | live | Quoted market and limit orders | Separate wallet quote API |

Unsupported environments fail locally; they never fall back to live.
Options and prediction endpoints use the SDK's generic signed methods because
this SDK version has no named clients for their current APIs.

Futures `initialize()` selects the maximum leverage tier compatible with
existing position and pending bid/ask notional. It applies the venue's returned
leverage and notional cap. Higher leverage may reduce the permitted position
notional. Initialization runs once per cached market adapter, including before
the first order. Read methods report the currently known leverage without
changing it. Account modes are validated and never changed automatically.
Recreate the client after changing account modes or leverage elsewhere.

Futures market/limit orders use regular orders; stops use conditional algo
orders with contract-price triggers. `reduceOnly` prevents opening/reversal.
Spot rejects `reduceOnly`. Options reject market/stops and uncovered sells.
Order IDs carry product namespaces; options cancellation uses client IDs to
avoid rounding Binance's large integer IDs.
See [futures orders](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/trade)
and [options orders](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-options/api/rest-api/trade).

`getFriction()` estimates taker fees plus exponentially weighted average slippage.
Pass a base quantity, e.g. `getFriction(0.01)`, to walk the current order book for
exactly that size instead of averaging. The supplied quantity must satisfy the
quantity limits and step; insufficient depth throws rather than clamping it.
This is exact for the observed book, not a guarantee of a future fill.
For each size it uses the worse buy/sell VWAP cost relative to the midpoint,
including spread. The range follows quantity min/max/step limits (market lots
for Spot/futures, limit lots for options), capped by visible depth on both sides.
It does not model account capacity or price-dependent notional filters.
Weights are proportional to `exp(-frictionDecayRate * (size - min) / (max - min))`.
The market config's optional `frictionDecayRate` defaults to `5`; `0` gives uniform
weights, and larger rates favor smaller orders more strongly. It must be finite
and nonnegative, and does not affect explicit-size estimates.
Up to 256 valid lots are summed exactly; larger grids and continuous
ranges use 256 deterministic midpoint probability quantiles. A single valid size
uses its actual cost. Empty books or ranges without valid quantities throw.
This is a distribution-based estimate, not a fill guarantee. Spot includes tax/special commissions and
ignores conditional BNB discounts. Options fees use the underlying index,
[capped relative to premium](https://www.binance.com/en/support/faq/detail/5326e5de61c34fed98abe28d2f175a23).

History contains completed candles, ascending, paginated at venue limits.
Spot supports native 1s candles. USD-M 1s candles are aggregated from trades
over the last requested seconds (maximum 3600); seconds without trades remain
absent. `getTicks()` returns aggregate trades, not individual fills, for an
inclusive range shorter than one hour, capped at one million records.
Options support fixed candle intervals from 1m; aggregate tick history is unavailable.
Other supported fixed intervals run through 1w; unsupported intervals throw.

#### Predictions

Prediction buys specify **collateral spend**, while sells specify **shares**.
They use `BinancePredictionApi`, returned by `client.market()`, rather than
misrepresenting exact spend as `TradingApi`'s exact base quantity.
Amounts are decimal strings converted exactly to 18-decimal wei.

```ts
const live = new BinanceClient({
  environment: "live",
  apiKey: process.env.BINANCE_PREDICTION_API_KEY!,
  apiSecret: process.env.BINANCE_PREDICTION_API_SECRET!,
});
const predictions = await live.market({
  product: "predictions",
  walletAddress: process.env.BINANCE_PREDICTION_WALLET_ADDRESS!,
  walletId: process.env.BINANCE_PREDICTION_WALLET_ID!,
  tokenId: "112233", // select the actual outcome token
  marketId: 5567895,
  vendor: "predict_fun",
  accountType: "SPOT",
  fundingSource: "CEX",
  slippageBps: 100,
});
await predictions.initialize();
const quote = await predictions.getQuote({ side: "buy", type: "market", amount: "10" });
// Inspect quote.amountOut, feeAmount, minReceive and expireAt before placing.
const order = await predictions.placeQuote(quote.quoteId);
```

Requires a registered prediction wallet, prediction API credentials and existing
SAS authorization. The adapter does not authorize a wallet or auto-transfer funds.
Market quotes use FOK; limit quotes carry the same limit price into GTC placement.
Accepted submissions return `pending`, not a claimed fill.
`getOrderBook()`, `getPosition()`, `getPaymentBalances()`, paginated
`getActiveOrders()`, and `cancelOrder()` are available.
Payment balances are CEX funding choices, not an invented combined wallet equity.
Historical candles, native stops, cash-style capacity and settlement/redemption
are not exposed by this adapter.
See [prediction trade APIs](https://developers.binance.com/en/docs/catalog/web3-wallet-prediction-trading/api/rest-api/trade)
and [wallet APIs](https://developers.binance.com/en/docs/catalog/web3-wallet-prediction-trading/api/rest-api/wallet).

Transport failures are never retried automatically. `BinanceOrderSubmissionError`
carries the reconciliation ID (client order ID, or prediction quote ID).
Query the venue before retrying an uncertain submission. Prediction quotes
are single-attempt within the adapter and expire; callers must also persist
reconciliation state across restarts. Live streams and later fill events are
outside these REST adapters.

Run the offline adapter tests from the repository root:

```sh
node --conditions=development --import tsx --test apps/server/test/binance-client.test.ts apps/server/test/binance-products.test.ts apps/server/test/binance-friction.test.ts
```

### Backtesting

```ts
import { BacktestTradingApi } from "./clients/backtest.js";

const api = new BacktestTradingApi(candles, {
  initialQuote: 10_000,
  leverage: 5,
  startIndex: 200, // first 200 completed candles are available to warmup
  candlePath: "open-high-low-close",
  feeRate: 0.0005,
  slippageRate: 0.0001,
  maintenanceMarginRate: 0.005, // explicit flat-rate model
  liquidationFeeRate: 0.002,
  marketRules: await binance.getMarketRules(),
  // funding: [{ timestamp, rate, markPrice }, ...],
});

// Construct your bot with this api before running the loop.
await bot.warmup();
for (let tick = api.advance(); tick; tick = api.advance()) {
  for (const event of api.drainOrderEvents()) await bot.onOrder(event);
  await bot.onTick(tick);
  for (const event of api.drainOrderEvents()) await bot.onOrder(event);
}
console.log(api.snapshot());
```

Configs, candles, rules, and returned histories are readonly data contracts.
Adapters share their references; callers must not mutate them. The backtester
keeps one source history and advances a cursor, exposing only completed candles.
Order and rule changes produce new snapshots, so previously returned values stay stable.
This is a TypeScript contract, without runtime freezing or defensive deep copies.

The supplied history must be chronological, non-overlapping OHLCV
equal-duration candles with millisecond timestamps and inclusive `closeTime`. `startIndex`
defaults to zero. `advance()` processes the next candle and returns its closing
tick, or `null` at the end. `getHistory()` only exposes completed replay history;
larger intervals aggregate complete, contiguous buckets aligned to Unix time.
With no warmup candles, advance once to establish a price before placing orders.
Rates and market rules are explicit configuration; there are no implicit venue defaults.

Execution assumptions:

- The required `candlePath` is either `open-high-low-close` or
  `open-low-high-close`. Price crossings are processed in that order, with equal
  time between the four points. OHLC does not reveal the real intrabar path.
- New market orders fill at the next candle open. Limits and stops start
  matching on the next candle; a stop-limit cannot use a price touched before
  its trigger. Gaps use the new open, including limit-price improvement.
  Liquidation takes precedence over an order at the same crossing. Orders at
  the same crossing execute in submission order.
- Fills are all-or-reject with unlimited modeled liquidity. Fees apply to every
  fill; slippage applies to market execution. There is no order-book queue or
  partial-fill model. Reduce-only orders exceeding the remaining position reject.
- Pending orders reserve estimated incremental margin and costs. Margin is
  checked again at execution. The reservation model is conservative and is not
  Binance's exact conditional-order margin algorithm.
- Supplied funding records debit longs and credit shorts for positive rates.
  Omitted funding means zero modeled funding. Candle prices stand in for mark
  prices when valuing equity and checking liquidation; funding uses its supplied
  mark price. Maintenance margin and liquidation fees are configurable flat
  rates, not Binance's full tiered liquidation engine.
- End-of-history positions remain open and are marked at the final close.
  Pending orders are canceled at exhaustion. `snapshot()` reports wallet balance,
  position, P&L, fees, funding and liquidation;
  `getOrder(id)` inspects an order and `drainOrderEvents()` drains queued events.

For both adapters, `quoteAvailable` is free collateral and `quoteReserved` is
initial margin plus pending-order reserves. `assetUnleveraged` is signed futures
exposure, while `quoteUnleveraged` is synthetic cash, preserving
`equity = quoteUnleveraged + assetUnleveraged * price`. Binance collateral is
shared with other cross-margin positions; the backtester models a dedicated
single-symbol account. Capacity is an estimate at the requested price and
clamps requested leverage to the configured/available leverage.

Run the offline adapter tests from the repository root:

```sh
npm test -w @trading/server
```
