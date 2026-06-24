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

When paper trading is enabled, the dashboard exposes:

- exchange account sync
- paper order placement
- open order cancellation
- futures leverage updates
- exchange balances, positions, and open orders

The server normalizes every outgoing paper order against Binance symbol filters
from `exchangeInfo`: price tick size, quantity step size, min/max quantity, and
min/max notional.

`TRADING_BINANCE_PAPER_AUTO_SUBMIT=true` submits strategy-created orders to the
active paper exchange. In this mode local tick processing does not fill or stale-cancel
open orders; Binance order/trade history is the order-status source. Exchange sync
reconciles bot-linked orders and fills back into the local ledger using the deterministic
`bot_<localOrderId>` client order id. Server startup runs the same sync so restart
recovery restores accepted, cancelled, and filled bot exchange orders.

Keep live trading disabled until the same reconciliation path has been exercised
for long-running futures sessions and user-data websocket streaming is added for
lower-latency fill updates.

to run demo futures bot server locally execute this:
```bash
TRADING_MARKET_ID=usdm-futures:BTCUSDT \
TRADING_BINANCE_PAPER_ENABLED=true \
TRADING_BINANCE_PAPER_MODE=usdm-futures-testnet \
BINANCE_PAPER_API_KEY=KifpW53tsEfJPBHiiUNrhvlAwmOL7tk54B8xexX7XhDDjy84Kszj3Ah9f5iPvV9S \
BINANCE_PAPER_API_SECRET=pkePwOWIyVLJ86HvmOMfm5fpmjnvP2dCX3t9SPrznRzlfKWUL0UYXmzmluEjKWaf \
TRADING_SHORT_MARGIN_MODEL=futures-margin \
TRADING_MAX_LEVERAGE=999 \
npm run dev -w @trading/server
```