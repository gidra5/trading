# Backtesting Notes

Historical backtests can run over saved data, recent Binance candle ranges, fixed
week/month/year ranges, random-window aggregates, or synthetic candle series. Long
Binance candle runs emit progress over the dashboard websocket, can be cancelled, and
stop early when simulated equity falls below 1% of starting capital or when account-level
liquidation is triggered.

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
- liquidating the aggregate account when replay price crosses the account liquidation
  price
- using a fast debt-leverage estimate before rebuilding the full position ledger
- preloading merged candle windows for random-window and random-length batches

Because legacy uses resting limit orders, OHLC replay can materially affect fill timing.
Results should be treated as deterministic simulator evidence, not a promise that live
orders would have filled at the same prices.

## Backtest Coverage

Backtesting should cover several market generators and sampling surfaces because each
one exposes different failure modes:

1. Synthetic graphs: sine waves, constant linear slopes, and Brownian noise isolate
   behavior under sideways chop, smooth trends, and noisy reversals.
2. Direct historical windows: last X days, months, or years test the strategy against
   real chronological market structure and real clustered volatility.
3. Random historical intervals: sampled windows from the candle cache reduce dependence
   on one recent period and provide median, P10, worst-case, and liquidation evidence.
4. Shuffled-increment historical samples: planned tests should shuffle historical
   returns or increments inside a sample to preserve local return distribution while
   breaking the exact chronology. This can separate statistical exposure from accidental
   fit to one market path.
5. Multiple markets: each symbol contributes a different volatility, liquidity, trend,
   and liquidation regime. Sampling across markets should make validation less
   BTC-specific once enough synchronized history is cached.

Promotion decisions should prefer improvements that survive across these surfaces,
rather than wins on one hand-picked window.

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
npm run benchmark:strategies -- --mode synthetic --synthetic-frequency 2 --synthetic-amplitude 0.08
npm run benchmark:strategies -- --cooldown-sec 300
npm run benchmark:strategies -- --only "legacy"
npm run benchmark:strategies -- --mode days --days 30 --only relaxed
npm run benchmark:strategies -- --mode synthetic --days 30 --only relaxed
npm run benchmark:strategies -- --mode random-lengths --leverage 1 --only relaxed
npm run benchmark:strategies -- --mode random-lengths --leverage 5 --only relaxed
npm run benchmark:strategies -- --mode days --days 30 --leverage 1 --only "short only" --short-margin futures-margin
```

The script reads local historical candles, runs `legacy-valley-peak` with identical
starting capital, leverage guard, position cap, fees, limit offset, and cooldown, then
reports return, drawdown, trade count, trade win rate, profitable closed positions,
liquidated positions, risk-adjusted return, Sharpe, additive perfect-margin capture,
and compounded perfect-margin capture.
The benchmark script defaults to `1x` max leverage. Fixed historical, synthetic, grid,
and portfolio checks should generally stay no-leverage first; run paired `1x` and `5x`
comparisons mainly for random-window validation. The `--only relaxed` comparison includes
the current relaxed per-lot long/short default, long-only, and short-only variants.
Shorts default to the `spot-borrow` margin model. Use `--short-margin futures-margin`
when testing standalone unlevered shorts against collateral-backed futures-style gross
exposure instead of borrowed-base spot-margin debt.

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

Synthetic mode generates deterministic OHLC candles from:

```text
price = startPrice * (1 + amplitude * sin(2pi * frequency * days) + trend * days + brownian)
```

The Brownian component is accumulated from normally distributed increments sampled at
candle open, midpoint, and close. The default synthetic start price is `100000`, which
keeps BTCUSDT-scale detector thresholds meaningful. Tune it with:

```bash
--synthetic-candles
--synthetic-start-price
--synthetic-frequency  # sine cycles per day
--synthetic-amplitude  # fraction of start price
--synthetic-trend      # linear slope as fraction of start price per day
--synthetic-noise      # Brownian daily volatility
--seed
```

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
- trade count, open-order count, trade win rate, profitable closed-position count, and
  liquidated-position count
- additive and compounded perfect-margin capture

Failed experiments should stay in the docs as observations, but removed algorithms
should not be kept as runnable controls.
