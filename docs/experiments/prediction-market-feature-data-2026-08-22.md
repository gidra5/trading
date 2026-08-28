# Point-in-time prediction-market data for the 1s and 1m BTC models

Date: 2026-08-22  
Status: acquisition path implemented and causality-tested; predictive contribution not yet measured.

## Decision

Yes for the numeric market prediction: we can recover the market-implied YES price at every relevant 1s or 1m model origin. We can also recover the contract's current canonical question, YES/NO outcome definition, strike when applicable, and open/close time. One qualification matters: the historical API does not expose field-by-field metadata revisions, so exact earlier wording or a strike that was populated after market open cannot always be reconstructed retrospectively.

The first production-quality source is Kalshi:

- Public trades contain contract quantity, YES price, side, trade ID, and a precise event timestamp.
- Public 1-minute candles contain completed-minute YES bid/ask OHLC, trade-price OHLC/mean, volume, and open interest.
- Live and older history are split at the provider's moving historical cutoff, but both partitions are publicly queryable.
- No settlement result, settlement value, or post-settlement current price is admitted to the feature rows.

The implemented importer is [fetch-prediction-market-history.ts](../../scripts/fetch-prediction-market-history.ts). Its deterministic reconstruction is in [prediction-market-data.ts](../../scripts/lib/prediction-market-data.ts), with leakage tests in [fetch-prediction-market-history.test.ts](../../scripts/fetch-prediction-market-history.test.ts).

## What “the prediction at time t” means

A contract such as “BTC price up in next 15 mins?” pays 1 dollar if YES and 0 otherwise. The price of YES is an executable market consensus proxy for the probability of YES, but it is not guaranteed to be a calibrated physical probability.

At a model origin `t`:

- 1s trade state uses only trades with `created_time < t`. A trade stamped exactly at `t` is excluded because it was not known before the new candle began.
- 1m quote state uses only a candle with `end_period_ts <= t`.
- `lastProbability` is the last traded YES price. Its age is always supplied so a stale print is not treated as a current quote.
- `quoteMidProbability` is the completed-minute midpoint of YES bid and ask. It is accompanied by spread, so a wide or illiquid quote receives less trust.
- The exact question and outcome wording are joined by `marketId` from the immutable metadata sidecar.

The metadata sidecar records `metadataUpdatedTime` and `mutableMetadataAvailableAt`. Mutable question/strike fields must be masked at origins earlier than that timestamp. This is conservative: it prevents a later-populated strike from leaking backward, although it can also hide a field that existed before an unrelated metadata update. Live lifecycle archiving is required for exact field-level historical wording from now on.

Historical trades do not reconstruct an exact second-by-second order book. Therefore the 1s historical representation has last price, VWAP, activity, range, and taker-side flow, but no exact bid/ask spread. Exact 1s quote and book state must be archived from the live WebSocket from now on. The 1m historical endpoint already supplies quote candles.

## Current selected-asset coverage

The certified feature bases currently contain 54 distinct asset identifiers at 1s and 171 at 1m. Direct liquid prediction markets do not exist for all of them. The strongest clean intersection is:

| selected asset | directly relevant recurring Kalshi series | cadence | current total series volume observed on 2026-08-22 | disposition |
|---|---|---|---:|---|
| BTC | `KXBTC15M`, `KXBTCD`, `KXBTC` | 15m direction; hourly thresholds/ranges | 12.735B; 6.495B; 407.163M | primary 1s and 1m candidate |
| ETH | `KXETHD`, `KXETH` | hourly thresholds/ranges | 277.967M; 100.308M | cross-asset 1m candidate; slower state for 1s |
| SOL | `KXSOL15M` | 15m direction | 241.643M | cross-asset 1s and 1m candidate |
| XRP | `KXXRP15M`, `KXXRPD`, `KXXRP` | 15m direction; hourly thresholds/ranges | 239.032M; 11.164M; 4.052M | cross-asset 1s and 1m candidate |
| HYPE | `KXBTCVSHYPE` | weekly BTC-versus-HYPE | 234 | too sparse to treat as a reliable fast feature |

There are currently 273 Kalshi crypto series in total. Other 1s/1m selected identifiers can still be mentioned by a one-off crypto, company, election, regulatory, or technology question, but this is intermittent coverage rather than a stable per-asset input. Discovery must use canonical entity names and provider tags, not raw ticker substring matching: selected identifiers such as `sign`, `the`, `ai`, and `v` are ambiguous English tokens.

## Global-event coverage

Useful recurring global series already include:

| family | examples | cadence | role in the BTC model |
|---|---|---|---|
| monetary policy | Fed decision, target rate, number of cuts | meeting/annual | probability changes and disagreement around policy expectations |
| inflation | CPI level and year-over-year inflation | monthly releases | pre-release expectation and post-release repricing shock |
| labor | payrolls and unemployment | monthly releases | risk/liquidity regime and surprise transmission |
| growth | GDP and recession | quarterly/annual | slower regime state; change is more useful than level at 1s |
| non-US central banks | ECB, BoE, BoJ, BoC, China-related policy questions when listed | event-dependent | global risk and FX/liquidity shocks |
| politics/geopolitics/regulation | elections, wars, shutdowns, tariffs, crypto legislation/reserves | event-dependent | sparse event set; retain category, horizon, liquidity, and age |

The recurring US macro series are liquid enough to merit screening: Fed-decision series volume was about 519.0M; inflation 35.0M; rate-cut count 34.8M; CPI 23.5M; payrolls 18.7M; GDP 14.8M; unemployment 10.8M; and recession 9.0M when queried on 2026-08-22. These are source-coverage facts, not evidence that they improve BTC prediction.

## Stored artifacts

Each bounded backfill writes only reduced gzip data, not raw trade pages:

| artifact | content |
|---|---|
| `*.markets.json` | `marketId`, provider/series/ticker, current canonical question/outcome wording, metadata availability/update time, creation/open/close/expiry times, strike semantics; settlement outcome is omitted |
| `*.1s.ndjson.gz` | state at each origin: last YES price and age, preceding-second trade count/contract volume/VWAP/min/max/change/taker fraction, time to contract close |
| `*.1m.ndjson.gz` | completed-minute YES bid/ask close, midpoint/spread, trade OHLC/mean/previous price, volume, open interest, time to contract close |
| `*.manifest.json` | request interval, series/profile, historical cutoff, causal semantics, row counts, and any source failures |

Overlapping backfill chunks should be deduplicated by `(marketId, availableAt)` after resolving the metadata sidecars to provider ticker. Missing values remain null. They are never replaced with 0 or 0.5.

## Validated real pilot

The importer was run against the official public API for `KXBTC15M` from 2026-08-22 12:30:00Z through 12:31:00Z. It found one contract:

- Question: “BTC price up in next 15 mins?”
- YES definition: target price 77,309.82 dollars
- Contract window: 12:30Z–12:45Z
- Output: 61 causal origin states at 1s and one completed 1m candle; no failures
- First active second: last YES price 0.50, age 80ms, 4 trades, 9.02 contracts, VWAP 0.500022
- Completed minute: bid 0.44, ask 0.45, midpoint 0.445, trade mean 0.4411, volume 107,845.47, open interest 71,504.36

The compact pilot is stored in `data/runtime-cache/prediction-market-pilot`. The 1s and 1m gzip artifacts were about 2.6KB and 235 bytes respectively for this interval.

## Recommended feature representation

Do not allocate a permanent scalar input to every ephemeral contract. During one tested hour there were 4 simultaneous/repeating BTC 15m contracts, 4 SOL contracts, 4 XRP 15m contracts, 10 XRP threshold contracts, 60 Fed-decision outcomes, 87 Fed-rate outcomes, and 21 annual rate-cut outcomes. The set of questions changes over time.

Use two branches.

### Direct asset surface

For BTC, ETH, SOL, and XRP markets, retain each contract as a set element with:

| coordinate | construction at origin t |
|---|---|
| probability state | `logit(clamp(p_yes, 0.01, 0.99))`, changes over 1s/5s/15s/1m where available |
| execution quality | bid/ask spread at 1m; last-trade age, trade count and volume at 1s |
| event geometry | time to close; direction/range/threshold type; strike distance divided by current asset volatility |
| market dynamics | preceding-interval VWAP, min/max, probability change, taker-side fraction, open-interest and volume changes |
| identity | asset ID, provider, series type, and observed/missing mask; never infer identity from title text alone |

Use either a small attention/set encoder or deterministic liquidity-weighted surface interpolation. For BTC threshold markets, interpolate probability across normalized strike distance and sample a fixed grid. This preserves changing contract sets without changing the model input width.

### Global-event state

Encode each active event with category, time to resolution, logit probability, trailing probability changes, spread/age, volume, and open interest. Aggregate with a set encoder, or initially screen a fixed summary:

- liquidity-weighted mean and dispersion of 1m/5m/15m logit changes;
- positive and negative probability-shock extrema;
- fraction of liquid markets rising, active-market count, and total volume;
- probability entropy/disagreement and median quote/trade age;
- the same summaries by macro, politics, geopolitics, regulation, and crypto categories.

The weights must depend only on trailing volume/liquidity available before `t`. Selecting the retrospectively highest-volume resolved contracts would leak future information.

## How to test contribution

Prediction-market features are provisional until they beat the certified 1s/1m bases on untouched time:

1. Backfill at least 30 days, reducing trades to the causal rows during download.
2. Freeze discovery aliases, transformations, strike grid, and liquidity thresholds on the training segment.
3. Compare the current 150-input 1s and 338-input 1m routed bases against `basis + prediction-market branch` on full-distribution negative log likelihood in bits.
4. Also score inactive/active state, sign, magnitude thresholds, calibration, and tail CRPS. The branch may help magnitude/volatility without helping sign.
5. Report conditional gain, block stability, bootstrap interval, coverage, and performance conditioned on high versus low market activity.
6. Require positive transfer to a later calendar block before adding any coordinate to production.

This is a conditional-information test. A large standalone association is insufficient if the existing volatility, cross-asset, order-flow, and calendar basis already explains it.

## Reproduction

```text
npm run test:prediction-markets

npm run fetch:prediction-markets -- --start 2026-07-22 --end 2026-08-22 --profile asset --resolution both

npm run fetch:prediction-markets -- --start 2026-07-22 --end 2026-08-22 --profile global --resolution 1m
```

`--all-series-at-1s` is available for an explicit global tick-level experiment, but the default restricts 1s expansion to fast recurring asset series to control API load and storage. The importer is all-or-nothing: it uses partial files, fails if pagination or any requested market fetch is incomplete, and only publishes final artifacts after the whole run succeeds.

## Other public sources

- [Kalshi market history partition](https://docs.kalshi.com/getting_started/historical_data), [trades](https://docs.kalshi.com/api-reference/market/get-trades), and [1m candles](https://docs.kalshi.com/api-reference/market/get-market-candlesticks) are the implemented primary source.
- [Polymarket market discovery](https://docs.polymarket.com/market-data/discover-markets) can broaden one-off asset and global-event coverage. Its standard [price-history endpoint](https://docs.polymarket.com/api-reference/markets/get-prices-history) has minute-valued fidelity, so it is naturally a 1m source; exact public trades can later supply a second 1s trade-state branch.
- Both Polymarket API hostnames currently resolve to localhost in this development environment and refuse connections. No routing workaround was attempted, so Polymarket is documented but not represented in the validated pilot.
- Manifold exposes public probability-changing bets, but its data/API terms restrict commercial model training without a license. It should not enter this trading model until licensing is resolved.

Kalshi and Polymarket can disagree because of venue population, rules, fees, liquidity, and resolution wording. Preserve provider identity and question semantics; do not average superficially similar markets until their outcomes and deadlines are normalized.
