# trading

Handmade Binance trading interface with three TypeScript workspace projects:

- `apps/server` - Fastify backend, Binance market websocket ingestion, JSONL market history, paper bot state, bot controls, and backtest API.
- `apps/web` - SolidJS + UnoCSS dashboard for candles, balances, bot performance, orders, fills, order book, and saved-history backtests.
- `packages/bot-algo` - reusable simulated trading bot and backtest engine.

## Run

```bash
npm install
npm run dev
```

Dashboard: http://localhost:5173  
Backend API: http://localhost:3001

The dev server writes live state and saved market data under `data/`.

## Build

```bash
npm run typecheck
npm run build
npm start
```

`npm start` serves the built Fastify backend. If `apps/web/dist` exists, the backend also serves the built frontend.

## Notes

Algorithm behavior is summarized in [docs/algorithms.md](docs/algorithms.md).
Backtesting behavior and current performance debt are tracked in [docs/backtesting.md](docs/backtesting.md).
Spreadsheet-derived position ledger formulas are documented in [docs/position-ledger.md](docs/position-ledger.md).

## Environment

```bash
PORT=3001
TRADING_SYMBOL=BTCUSDT
TRADING_INTERVAL=1m
TRADING_STARTING_QUOTE=10000
TRADING_DATA_DIR=/path/to/data
TRADING_HISTORY_CACHE_MAX_BYTES=512mb
TRADING_HISTORY_CACHE_MIN_FREE_BYTES=512mb
```
