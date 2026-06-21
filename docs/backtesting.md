# Backtesting Notes

Current historical backtests can run over saved data, a configurable `Last X days`
Binance candle range, fixed Binance candle ranges for the last week/month/year, or a
random-window aggregate mode. Long Binance candle runs emit progress over the dashboard
websocket, can be cancelled from the dashboard, and stop early when simulated equity
falls below 1% of starting capital.

The random-window modes run many smaller historical candle backtests and aggregate the
strategy result metrics across the samples. `Random weeks` defaults to 40 one-week
windows sampled from the last year and averages raw sample results. `Random lengths`
defaults to 40 windows with random lengths from 1 to 30 days sampled from the last year;
because sample durations differ, it also reports average profit/day and return/day rates.
The dashboard exposes inputs for sample count, fixed/random window length, lookback,
and extra random pairs. Extra pairs are sampled from the same market group, venue, and
quote asset as the currently selected pair, then each random window is replayed against
the selected pair plus those extra pairs.
The API can tune this with `historicalDays`, `randomSampleCount`, `randomWindowDays`,
`randomMinWindowDays`, `randomMaxWindowDays`, `randomLookbackDays`, and
`randomPairCount` on `POST /api/backtest`.

## Historical Candle Cache

Historical Binance candle backtests use a local day-sharded cache under:

```text
data/historical/<symbol>/<interval>/YYYY-MM-DD.jsonl
```

The cache stores only missing candle ranges, so repeated week/month/year runs should reuse local data and avoid hitting Binance for already cached days. The current day may still fetch newly formed candles.

Storage limits are configurable:

```bash
TRADING_HISTORY_CACHE_MAX_BYTES=512mb
TRADING_HISTORY_CACHE_MIN_FREE_BYTES=512mb
```

When the cache is over budget or disk free space falls below the configured floor, the server evicts the oldest cache shards first. Shards needed by the currently running backtest are protected; if the requested range alone is larger than the configured cache limit, the backtest fails with a clear cache-limit error instead of repeatedly refetching data.

## Performance Notes

The replay path is optimized for long candle runs:

- backtests use the same paper bot state machine but disable event cloning during replay
- candle backtests replay deterministic open/high/low/close ticks through the same strategy path as the dashboard historical backtest; this matches the UI, but it also means OHLC-only backtests depend on a synthetic intra-candle path assumption
- metrics are marked to market only at sampled candle boundaries instead of every synthetic tick
- equity curves are capped to roughly 800 points by default
- returned backtest orders/fills are capped to the latest 2,000 each by default; summary metrics still count the full run
- open order bookkeeping uses indexed set entries instead of scanning all historical orders on every fill
- leverage checks use a fast debt-leverage estimate first and rebuild the full position ledger only when the estimate is near or above the configured limit
- historical replay emits progress in larger candle batches and dashboard websocket broadcasts are throttled so UI snapshots do not block replay
- directional strategies keep private rolling stats for volatility, mean/std, and breakout ranges instead of rescanning price windows every tick
- replay uses a trusted no-event tick path, skipping symbol checks, unused volume-derived quantities, and per-tick metrics updates
- cached full-day shards are validated with line count plus first/last candle checks instead of parsing every cached candle before replay
- random-window and random-length runs prefetch the merged union of sample windows once per market, then skip cache validation for each individual sample
- random-window and random-length runs preload the merged candle union for moderately sized sample sets and replay individual samples by index, avoiding repeated JSONL parsing of overlapping windows

## Strategy Benchmark Script

Local strategy comparisons can be run without starting the server:

```bash
npm run benchmark:strategies
```

Useful options:

```bash
npm run benchmark:strategies -- --days 7
npm run benchmark:strategies -- --mode year
npm run benchmark:strategies -- --mode random-lengths
npm run benchmark:strategies -- --mode grid-search --days 30 --grid-folds 3
npm run benchmark:strategies -- --mode portfolio --days 30
npm run benchmark:strategies -- --include-random-sign --only random
npm run benchmark:strategies -- --leverage 1 --cooldown-sec 1800
npm run benchmark:strategies -- --only trend
```

The script reads local historical candles from `data/historical/<symbol>/<interval>`,
runs the built-in strategies with identical starting capital, leverage, position cap,
fees, slippage, and cooldown, then reports return, drawdown, trade count, win rate, and
perfect-margin capture. Random-length mode defaults to 40 deterministic samples,
1-30 day windows, 365 day lookback, and seed `1337`; tune it with `--samples`,
`--min-window-days`, `--max-window-days`, `--lookback-days`, and `--seed`.
Grid-search mode runs a small folded parameter search over the current trend,
breakout, and reversion families. Portfolio mode tests allocation across strategy
equity curves and attempts multi-symbol allocation when at least two cached symbols
exist. The deterministic random-sign control is opt-in because it is intentionally
churn-heavy.
`Risk Ret` is `returnPct / maxDrawdownPct`. `Sharpe` is annualized from the sampled
equity curve with zero risk-free rate, so it is a coarse comparison metric rather than a
tick-perfect realized Sharpe.

Latest local BTCUSDT 1m benchmark, using 30 cached day files from 2026-05-22 through
2026-06-20, `3x` max leverage, target cap `$25,500`, initial short debt cap
`$19,600`, and `300s` cooldown:

| Strategy | Return | Net PnL | Max DD | Risk Ret | Sharpe | Trades | Win Rate | Oracle Capture |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline always flat | 0.00% | $0.00 | 0.00% | - | - | 0 | 0.0% | 0.000% |
| Baseline always long | -47.11% | -$4,711.03 | 60.74% | -0.776 | -3.642 | 1 | 0.0% | -26.249% |
| Baseline always short | 35.52% | $3,552.42 | 13.55% | 2.622 | 6.632 | 1 | 0.0% | 19.793% |
| Moving average | -39.99% | -$3,998.58 | 39.99% | -1.000 | -40.971 | 4,519 | 16.1% | -22.279% |
| Legacy valley/peak | -15.70% | -$1,569.67 | 23.50% | -0.668 | -4.804 | 258 | 7.2% | -8.746% |
| Trend following L/S | -61.84% | -$6,184.28 | 61.84% | -1.000 | -73.217 | 2,056 | 7.0% | -34.458% |
| Vol breakout L/S | -17.22% | -$1,721.90 | 17.22% | -1.000 | -23.038 | 332 | 22.2% | -9.594% |
| Mean reversion L/S | -72.65% | -$7,265.09 | 72.65% | -1.000 | -129.283 | 2,373 | 2.7% | -40.480% |

This result is intentionally recorded as a failed baseline: the mechanics work, but the
simple first-pass long/short signals do not yet have a cost-adjusted edge on this sample.
The passive short control winning both the 30-day and last-year windows is a warning that
the active rules are mostly adding churn, not timing skill.

Latest local BTCUSDT 1m last-year benchmark, using 365 cached day files from
2025-06-21 through 2026-06-20, `3x` max leverage, target cap `$25,500`, initial
short debt cap `$19,600`, and `300s` cooldown:

| Strategy | Return | Net PnL | Max DD | Risk Ret | Sharpe | Trades | Win Rate | Oracle Capture |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline always flat | 0.00% | $0.00 | 0.00% | - | - | 0 | 0.0% | 0.000% |
| Baseline always long | -98.91% | -$9,890.78 | 103.79% | -0.953 | 1.126 | 1 | 0.0% | -4.080% |
| Baseline always short | 75.34% | $7,533.73 | 45.52% | 1.655 | 1.125 | 1 | 0.0% | 3.107% |
| Moving average | -99.74% | -$9,974.50 | 99.74% | -1.000 | -13.761 | 17,489 | 15.7% | -4.114% |
| Legacy valley/peak | -35.98% | -$3,597.54 | 48.08% | -0.748 | -0.952 | 4,918 | 41.0% | -1.484% |
| Trend following L/S | -99.97% | -$9,996.69 | 99.97% | -1.000 | -27.298 | 16,557 | 17.9% | -4.123% |
| Vol breakout L/S | -85.62% | -$8,561.53 | 85.62% | -1.000 | -16.541 | 3,197 | 25.0% | -3.531% |
| Mean reversion L/S | -100.00% | -$9,999.97 | 100.00% | -1.000 | -62.639 | 21,695 | 7.1% | -4.125% |

Latest local BTCUSDT 1m random-length benchmark, using 40 deterministic samples,
1-30 day windows, 365 day lookback, seed `1337`, `3x` max leverage, and `300s`
cooldown:

| Strategy | Samples | Profitable | Avg Return | Avg Net PnL | Avg PnL/day | Avg Max DD | Avg Risk Ret | Avg Sharpe | Avg Trades | Avg Capture | Best | Worst |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline always flat | 40 | 0/40 | 0.00% | $0.00 | $0.00 | 0.00% | - | - | 0.0 | 0.000% | 0.00% | 0.00% |
| Baseline always long | 40 | 17/40 | -5.87% | -$587.05 | -$18.07 | 21.99% | 0.372 | 0.517 | 1.0 | 150.410% | 27.98% | -67.63% |
| Baseline always short | 40 | 19/40 | 3.83% | $382.63 | $5.92 | 13.25% | 0.694 | 0.594 | 1.0 | -137.020% | 51.30% | -22.20% |
| Moving average | 40 | 0/40 | -12.99% | -$1,299.07 | -$97.36 | 13.24% | -0.965 | -31.721 | 1,830.4 | -129.510% | -1.57% | -35.09% |
| Legacy valley/peak | 40 | 19/40 | -2.55% | -$255.24 | -$10.59 | 7.33% | 0.395 | 0.287 | 175.7 | 4.945% | 6.49% | -22.95% |
| Trend following L/S | 40 | 0/40 | -24.49% | -$2,448.66 | -$187.36 | 24.56% | -0.991 | -65.727 | 700.5 | -210.048% | -1.87% | -78.15% |
| Vol breakout L/S | 40 | 2/40 | -5.74% | -$574.50 | -$42.86 | 6.12% | -0.836 | -18.539 | 103.5 | -48.714% | 0.34% | -28.78% |
| Mean reversion L/S | 40 | 0/40 | -41.81% | -$4,180.99 | -$354.63 | 41.82% | -1.000 | -148.066 | 1,086.4 | -683.772% | -8.97% | -71.56% |

The detailed experiment plan, grid-search result, and portfolio result are recorded in
[`docs/experiment-plan.md`](experiment-plan.md).

The original synthetic benchmark was too optimistic because it did not match the active
dashboard path. With cached Binance one-minute candles, the API-backed historical path
currently processes roughly:

- 101k candles/s over a cached month `master-adaptive` backtest with a live dashboard websocket
- 109k candles/s over a cached month `master-adaptive` backtest through the direct API path
- 133k candles/s over a cached year `master-adaptive` backtest through the direct API path

The in-memory replay path is faster because it excludes cache batch reads: the same
master strategy processed the latest cached 30-day window at about 110k candles/s.
Simpler synthetic strategy benchmarks and lighter algorithms can still be faster, but
they should not be treated as representative of the live dashboard backtest path.

The biggest replay wins came from removing hot-loop allocations and duplicate work:
candle replay no longer allocates tick arrays, legacy buy/sell rolling averages share
the same price stream, rolling-average samples no longer allocate objects per update,
directional rolling-window helpers use cached rolling stats, and dashboard progress
updates no longer force a full public websocket snapshot every 1,000 candles.
Repeated cached runs also avoid a full JSON parse pass during cache validation for
complete day shards.
Legacy-only replay can still be much faster; the master-adaptive path is the current
conservative benchmark because it exercises the heaviest strategy stack.
For random-length runs, the same 40-sample `master-adaptive` configuration dropped from
roughly 19s wall-clock to about 8.5-9s wall-clock in the server path. A cached local
CLI run over the same random-length shape completed in about 5.1s by replaying sample
index ranges against one loaded candle array.

Remaining work before treating year-scale runs as cheap:

- improve the historical cache importer with resume/checkpoint metadata and batch verification
- store long historical data in a queryable binary or columnar format instead of JSONL
- run long backtests in a dedicated worker/job queue so UI and live trading remain isolated
- add optional full trade-log export for cases where the latest 2,000 returned fills are not enough
