# Binance Paper Trading

The server can connect to Binance testnet/demo trading endpoints separately from the
local simulator. This is forward paper trading only; historical backtests still run
locally.

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

When paper trading is enabled, the dashboard exposes:

- exchange account sync
- paper order placement
- open order cancellation
- futures leverage updates
- exchange balances, positions, and open orders

`TRADING_BINANCE_PAPER_AUTO_SUBMIT=true` shadow-submits strategy-created orders to
the active paper exchange. The local bot state remains the source of strategy memory
and chart annotations; exchange fills are shown separately in the Binance Paper panel.
Keep this off until strategy/exchange state reconciliation is implemented.
