# Backtesting Notes

Current historical backtests can run over saved data, a configurable `Last X days`
Binance candle range, fixed Binance candle ranges for the last week/month/year, or a
random-window aggregate mode. Long Binance candle runs emit progress over the dashboard
websocket and stop early when simulated equity falls below 1% of starting capital.

The random-window modes run many smaller historical candle backtests and aggregate the
strategy result metrics across the samples. `Random weeks` defaults to 40 one-week
windows sampled from the last year and averages raw sample results. `Random lengths`
defaults to 40 windows with random lengths from 1 to 30 days sampled from the last year;
because sample durations differ, it also reports average profit/day and return/day rates.
The dashboard exposes inputs for sample count, fixed/random window length, and lookback.
The API can tune this with `historicalDays`, `randomSampleCount`, `randomWindowDays`,
`randomMinWindowDays`, `randomMaxWindowDays`, and `randomLookbackDays` on
`POST /api/backtest`.

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
- metrics are marked to market only at sampled candle boundaries instead of every synthetic tick
- equity curves are capped to roughly 800 points by default
- returned backtest orders/fills are capped to the latest 2,000 each by default; summary metrics still count the full run
- open order bookkeeping uses indexed set entries instead of scanning all historical orders on every fill

The original synthetic benchmark was too optimistic because it did not match the active
dashboard path. With the default `legacy-valley-peak` algorithm and cached Binance
one-minute candles, the API-backed dashboard path currently processes roughly:

- 32k candles/s over a cached month backtest
- 35k candles/s over a cached year backtest

The in-memory replay path for the same cached month is closer to 40k candles/s. Simpler
synthetic strategy benchmarks can still be much faster, but they should not be treated as
representative of the live dashboard backtest path.

Remaining work before treating year-scale runs as cheap:

- improve the historical cache importer with resume/checkpoint metadata and batch verification
- store long historical data in a queryable binary or columnar format instead of JSONL
- reduce remaining strategy/accounting overhead enough to reach 100k+ candles/s on real cached data
- run long backtests in a dedicated worker/job queue so UI and live trading remain isolated
- add optional full trade-log export for cases where the latest 2,000 returned fills are not enough
