# trading

Handmade Binance trading interface with three TypeScript workspace projects:

- `apps/server` - Fastify backend, Binance market websocket ingestion, JSONL market history, paper bot state, bot controls, and backtest API.
- `apps/web` - SolidJS + UnoCSS dashboard for candles, balances, bot performance, orders, fills, order book, and saved-history backtests.
- `packages/bot-algo` - reusable simulated trading bot and backtest engine.

## Run

```bash
npm run setup
npm run dev
```

Dashboard: http://localhost:5173  
Backend API: http://localhost:3001

The dev server writes live state and saved market data under `data/`.

`npm run setup` installs the JavaScript workspaces, creates the Python 3.12
environment, installs the CUDA-enabled PyTorch/Triton stack, builds the native
oracle kernel, and builds all three workspaces. On Windows, the ML runtime is
native and uses workspace-local CUDA compiler files under `.tools/`; Visual
Studio 2022 C++ Build Tools must be installed. Linux continues to use the
system CUDA toolkit.

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

### Kronos candle forecasts

Install the pinned public Kronos mini, small, and base checkpoints, then run the
probabilistic 15x1m benchmark across the non-fit inspector windows:

```bash
npm run kronos:setup
npm run kronos:benchmark -- --models all --temperature 0.8 --sample-count 20
```

The benchmark retains all Monte Carlo paths, reports candle/correlation/oracle
distribution metrics, and repairs invalid OHLC quantiles with KQSP. Predictor
fine-tuning is available through `npm run kronos:finetune`; the documented final
configuration excludes every policy-calibration inspector episode from both
training and checkpoint-selection loss. Causal forecast
artifacts can be calibrated and replayed through the real fee/slippage/borrow-
aware bot with `npm run kronos:backtest`; its `calibrate` and `validate` phases
enforce the chronological policy split. After fine-tuning,
`npm run kronos:final-pipeline` resumes and chains dense inference, policy
freezing, the guarded one-shot bot validation, and a hash-bound completion
audit. The audit can also be rerun explicitly with `npm run kronos:audit`. The leakage-free data split, exact
commands, limitations, and current results are documented in
[the Kronos report](docs/experiments/kronos-report-2026-08-07.md), with the full
experiment notebook in
[docs/experiments/kronos-btcusdt-1m-15-candle-2026-08-06.md](docs/experiments/kronos-btcusdt-1m-15-candle-2026-08-06.md).

### Financial foundation forecast models

FinCast, TiRex-2, and Chronos-2 use a separate pinned Python environment and a
shared native-quantile adapter:

```bash
npm run forecast-models:setup
npm run forecast-models:smoke
npm run forecast-models:screen
npm run forecast-models:benchmark
```

Chronos-2 can be adapted on the strictly pre-test BTC history with
`forecast-models:finetune:chronos2` and compared with its base checkpoint using
`forecast-models:validate:chronos2`. Dense causal artifacts use the same real
bot simulator via `forecast-models:backtest`. The KAMA inspector catalog exposes
the complete global and per-window benchmark summary. Exact pins, the
leakage contract, LoRA selection, all-window results, and simulator evidence are
recorded in
[docs/experiments/foundation-forecast-models-btcusdt-1m-2026-08-07.md](docs/experiments/foundation-forecast-models-btcusdt-1m-2026-08-07.md).

## Historical data

The downloader is resumable and stores independently compressed daily shards,
so long downloads can run in the background while the server and trainer use
completed days:

```bash
npm run fetch:candles -- --symbol BTCUSDT --interval 1s --days 1826 --end 2026-07-24 --compression gzip --fill-gaps --data-dir data
```

MLP dataset components are also independent daily shards. Features, full
86,400-row raw one-second oracle distributions, and physical 1,441-row
completed-minute oracle distributions use Zstandard-compressed Float32 arrays.
The trainer decompresses one shard at a time for bounded-memory streaming and
does not synthesize minute targets at training time.

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

## MLP training

Prepare and run a specific plan from PowerShell, cmd, or a POSIX shell:

```bash
npm run mlp:run -- --plan ml/training-plans/PLAN.json
```

The command is resumable. Runtime status is written under the plan's
`data/ml-runs/...` directory and is exposed on the dashboard's MLP training
page while the server is running.
