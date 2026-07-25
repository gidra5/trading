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

## Experiments

```bash
npm run benchmark:strategies
npm run experiment:loop
npm run basis:binance
npm run basis:compare-scales
```

`benchmark:strategies` defaults to random-length BTCUSDT samples across the available
five-year cycle instead of a recent 30-day window. `experiment:loop` repeatedly runs
that benchmark, prompts a Codex agent to review and improve the master adaptive
strategy, runs typecheck, and writes iteration logs under `data/experiments/agent-loop`.

`basis:binance` discovers economic underlyings across Binance products and selects
a near-orthogonal subset from aligned returns. It supports exact sample counts,
such as `--candles 360 --interval 4h`, and prefers the largest mean absolute
candle return among near-equivalent orthogonality pivots at that interval.
Methodology and options are documented in
[docs/binance-portfolio-basis.md](docs/binance-portfolio-basis.md).
`basis:compare-scales` compares independently selected 360-return bases at 1d,
4h, 1h, 15m, and 1m without mixing return amplitudes across intervals. It also
produces a 5%-capped, liquidity-size-weighted index for every scale and an
equal-weight aggregate of the five scale sleeves; see
[docs/binance-multiscale-basis.md](docs/binance-multiscale-basis.md).

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
Strategy research notes and benchmark observations are tracked in [docs/strategy-research.md](docs/strategy-research.md).
The current experiment plan and latest run results are recorded in [docs/experiment-plan.md](docs/experiment-plan.md).
The trading model and spreadsheet-derived position ledger formulas are documented in [docs/position-ledger.md](docs/position-ledger.md).
The causal MLP predictor, training workflow, artifact contract, and GPU inference settings are documented in [docs/mlp-exposure-predictor.md](docs/mlp-exposure-predictor.md).
Automated position management is documented in [docs/automated-position-management.md](docs/automated-position-management.md).
UI-driven manual fill workflows are documented in [docs/manual-position-management.md](docs/manual-position-management.md).

## Environment

```bash
PORT=3001
TRADING_SYMBOL=BTCUSDT
TRADING_INTERVAL=1m
TRADING_STARTING_QUOTE=10000
TRADING_MAX_LEVERAGE=1
TRADING_EXCHANGE_ACCOUNT_GUARD_HARD_STOP=false
TRADING_BINANCE_LIVE_ENABLED=false
TRADING_BINANCE_EXCHANGE_MODE=auto
TRADING_DATA_DIR=/path/to/data
TRADING_HISTORY_CACHE_MAX_BYTES=512mb
TRADING_HISTORY_CACHE_MIN_FREE_BYTES=512mb
TRADING_MLP_EXECUTION_PROVIDER=auto
TRADING_MLP_BATCH_SIZE=1024
TRADING_MLP_CUDNN_DIR=
BINANCE_API_KEY=
BINANCE_API_SECRET=
```
