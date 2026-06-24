# Binance Paper Trading

The server can connect to Binance testnet/demo trading endpoints separately from the
local simulator. This is forward paper trading only; historical backtests still run
locally.

demo binance: https://demo.binance.com/en/trade/BTC_USDT?type=spot

Enable it with separate paper credentials:

```bash
TRADING_BINANCE_PAPER_ENABLED=true
TRADING_BINANCE_PAPER_MODE=auto
BINANCE_PAPER_API_KEY=...
BINANCE_PAPER_API_SECRET=...
```

Supported modes:

- `auto`: `spot` markets use Spot Testnet, USD-M futures use USD-M Futures Testnet,
  and COIN-M futures use COIN-M Futures Testnet.
- `spot-testnet`: Spot Testnet REST and WebSocket streams.
- `spot-demo`: Spot Demo Mode REST and WebSocket streams.
- `usdm-futures-testnet`: USD-M Futures Testnet.
- `coinm-futures-testnet`: COIN-M Futures Testnet.

Optional settings:

```bash
TRADING_BINANCE_PAPER_AUTO_SUBMIT=false
BINANCE_PAPER_RECV_WINDOW_MS=5000
BINANCE_PAPER_BASE_URL=https://custom-endpoint.example
```

Futures paper trading:

```bash
TRADING_MARKET_ID=usdm-futures:BTCUSDT
TRADING_BINANCE_PAPER_ENABLED=true
TRADING_BINANCE_PAPER_MODE=usdm-futures-testnet
BINANCE_PAPER_API_KEY=...
BINANCE_PAPER_API_SECRET=...
TRADING_SHORT_MARGIN_MODEL=futures-margin
TRADING_MAX_LEVERAGE=999
```

`TRADING_MAX_LEVERAGE` is treated as the requested strategy cap. For futures
paper modes the server fetches Binance notional/leverage brackets through the
active paper endpoint and caps the running bot config to the exchange maximum.
Leverage changes submitted through the dashboard/API are capped the same way.

On startup the live bot warms its SMA/valley-peak memory from stored closed
candles. If the local candle store does not cover the configured averaging window
plus saturation period, the server fetches recent klines from the active Binance
paper/demo endpoint. That means a fresh process does not need to sit idle for the
whole SMA warmup period before the strategy can evaluate signals.

When paper trading is enabled, the dashboard exposes:

- exchange account sync
- paper order placement
- open order cancellation
- futures leverage updates
- exchange balances, positions, and open orders
- a position close control that pauses the bot and cancels open strategy orders;
  by default it closes only positions that are profitable at the current price,
  while the force option closes all open positions

The server normalizes every outgoing paper order against Binance symbol filters
from `exchangeInfo`: price tick size, quantity step size, min/max quantity, and
min/max notional. It also fetches account commission rates when the active
paper endpoint supports them and updates the running bot `feeBps` from the taker
rate. Binance does not expose a static slippage constant, so the server estimates
`positionRisk.marketSlippageBps` from the current best bid/ask half-spread.

`TRADING_BINANCE_PAPER_AUTO_SUBMIT=true` submits strategy-created orders to the
active paper exchange. In this mode local tick processing does not fill or stale-cancel
open orders; Binance order/trade updates are the order-status source. Futures
paper modes open a private user-data websocket, keep its listen key alive, and
apply bot-linked `ORDER_TRADE_UPDATE` fills immediately using the deterministic
`bot_<localOrderId>` client order id. Each user-data event also triggers an
exchange sync so balances, positions, and REST order history stay consistent.
Server startup runs the same sync so restart recovery restores accepted,
cancelled, and filled bot exchange orders.

Spot paper modes still use explicit REST sync for account updates. Binance removed
the old Spot listen-key stream path; adding Spot user-data streaming needs the new
WebSocket API authentication flow.

Keep live trading disabled until the same reconciliation path has been exercised
for long-running futures sessions.

to run demo futures bot server locally execute this:
```bash
TRADING_MARKET_ID=usdm-futures:BTCUSDT \
TRADING_BINANCE_PAPER_ENABLED=true \
TRADING_BINANCE_PAPER_MODE=usdm-futures-testnet \
BINANCE_PAPER_API_KEY=KifpW53tsEfJPBHiiUNrhvlAwmOL7tk54B8xexX7XhDDjy84Kszj3Ah9f5iPvV9S \
BINANCE_PAPER_API_SECRET=pkePwOWIyVLJ86HvmOMfm5fpmjnvP2dCX3t9SPrznRzlfKWUL0UYXmzmluEjKWaf \
TRADING_SHORT_MARGIN_MODEL=futures-margin \
TRADING_MAX_LEVERAGE=1 \
TRADING_BINANCE_PAPER_AUTO_SUBMIT=true \
npm run dev -w @trading/server
```

optional web ui for it:
```bash
npm run dev -w @trading/web
```
