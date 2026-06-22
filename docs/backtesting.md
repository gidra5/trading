# Backtesting Notes

Historical backtests can run over saved data, recent Binance candle ranges, fixed
week/month/year ranges, or random-window aggregates. Long Binance candle runs emit
progress over the dashboard websocket, can be cancelled, and stop early when simulated
equity falls below 1% of starting capital.

The only automated strategy under test is `legacy-valley-peak`.

## Historical Candle Cache

Historical Binance candle backtests use a local day-sharded cache under:

```text
data/historical/<symbol>/<interval>/YYYY-MM-DD.jsonl
```

Newer market-aware cache layouts may add venue and market-id path segments. The
benchmark loader chooses the deepest local cache with the most matching day files.

The cache stores only missing candle ranges, so repeated week/month/year runs should
reuse local data and avoid hitting Binance for already cached days. The current day may
still fetch newly formed candles.

Storage limits are configurable:

```bash
TRADING_HISTORY_CACHE_MAX_BYTES=512mb
TRADING_HISTORY_CACHE_MIN_FREE_BYTES=512mb
```

When the cache is over budget or disk free space falls below the configured floor, the
server evicts the oldest cache shards first. Shards needed by the currently running
backtest are protected.

## Replay Model

Candle backtests replay each candle as deterministic open/high/low/close ticks through
the same bot path used by the dashboard. This keeps server backtests and UI backtests
consistent, but OHLC-only replay still depends on a synthetic intra-candle path.

The replay path optimizes long runs by:

- disabling event cloning during replay
- sampling equity curves instead of marking metrics on every synthetic tick
- capping returned orders/fills while keeping full summary counts
- filling resting legacy limit orders when synthetic ticks cross their price
- using a fast debt-leverage estimate before rebuilding the full position ledger
- preloading merged candle windows for random-window and random-length batches

Because legacy uses resting limit orders, OHLC replay can materially affect fill timing.
Results should be treated as deterministic simulator evidence, not a promise that live
orders would have filled at the same prices.

## Strategy Benchmark Script

Local legacy valley/peak benchmarks can run without starting the server:

```bash
npm run benchmark:strategies
```

Useful options:

```bash
npm run benchmark:strategies -- --days 7
npm run benchmark:strategies -- --mode year
npm run benchmark:strategies -- --mode random-lengths
npm run benchmark:strategies -- --mode grid-search --days 1825 --grid-folds 6
npm run benchmark:strategies -- --mode portfolio --days 1825
npm run benchmark:strategies -- --leverage 3 --cooldown-sec 300
npm run benchmark:strategies -- --only "legacy"
```

The script reads local historical candles, runs `legacy-valley-peak` with identical
starting capital, leverage guard, position cap, fees, limit offset, and cooldown, then
reports return, drawdown, trade count, win rate, risk-adjusted return, Sharpe, and
perfect-margin capture.

Random-length mode defaults to deterministic sampled windows. Tune it with:

```bash
--samples
--min-window-days
--max-window-days
--lookback-days
--seed
```

Grid-search mode runs folded tests over legacy parameter variants only. Current grid
candidates adjust valley/peak clip size, warmup, sigma, buy/sell rates, and confirmation
strictness.

Portfolio mode is still useful for multi-symbol allocation experiments when enough
symbols are cached, but it should allocate around legacy valley/peak equity curves
rather than multiple strategy families.

`Risk Ret` is `returnPct / maxDrawdownPct`. `Sharpe` is annualized from the sampled
equity curve with zero risk-free rate, so it is a coarse comparison metric rather than a
tick-perfect realized Sharpe.

## Promotion Standard

A legacy valley/peak parameter change should not be promoted from a single recent fixed
window. Use cycle-wide random-length windows and folded checks, then record:

- command and seed
- date range and candle count
- leverage guard, position cap, cooldown, fee, and limit offset
- average, median, and P10 return
- max drawdown and risk-adjusted return
- trade count, open-order count, and win rate
- perfect-margin capture

Failed experiments should stay in the docs as observations, but removed algorithms
should not be kept as runnable controls.
