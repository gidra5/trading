# Backtesting Notes

Current historical backtests can run over saved data or Binance candle ranges for the last week, month, or year. Long Binance candle runs emit progress over the dashboard websocket and stop early when simulated equity falls below 1% of starting capital.

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

## Performance Work Needed

Backtest performance is still a known weak point. The current replay path feeds synthetic OHLC ticks through the same paper bot used for live simulation, and the bot still scans accumulated order arrays while replaying. Before relying on year-scale tests heavily, improve this path:

- keep an indexed open-order collection instead of filtering all historical orders on every tick
- improve the historical cache importer with resume/checkpoint metadata and batch verification
- store long historical data in a queryable format instead of large JSON arrays
- run long backtests in a dedicated worker/job queue so UI and live trading remain isolated
- sample equity curves and persisted fills intentionally instead of writing very large result payloads
