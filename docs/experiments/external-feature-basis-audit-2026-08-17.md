# External feature basis audit

Generated 2026-08-17T12:21:06.874Z. This is the canonical inventory for external BTC return-distribution features from 1s through 1h.

## Outcome

The requested list is fully enumerated. Public backfills, matched crypto minute histories, and ten synchronized monthly quote/liquidation samples have been measured. Selected: 3; provisional: 3; rejected: 2; collecting live: 6; proxy-only: 1; credential-required: 1. There is no generic awaiting-data bucket.

The matched 15m neural test remains negative: adding all 231 existing forward inputs changed validation objective from 2.6894290 to 2.6957000 (0.233% worse). This is why the basis must be selected conditionally instead of concatenating every candidate.

## Thirty-day common-coverage update

The freely backfillable recent layer is no longer awaiting data. For 2026-07-18 through 2026-08-16, the local store contains checksum-verified Binance spot 1s/1m candles, spot aggregate-trade flow, USD-M perpetual 1m candles and 5m metrics, and BTC/ETH/SOL/BNB/DOGE 1m candles. These overlap the recent local Binance spot-book capture where that stream is present.

The resulting 147-coordinate joint dataset has 42,901 causal minute-boundary origins and uses 16d train / 7d primary / 7d untouched transfer. Futures 1m log trade count transfers jointly at the 1m head. No external coordinate is promoted at 15m or 1h; the recent 1h mixture fails transfer. Full results are in `docs/experiments/global-return-feature-basis-30d-2026-08-17.md`.

This backfill does not manufacture unavailable history. Continuous cross-exchange L2, full option surfaces/OI, exact liquidation history, historical mempool snapshots, and point-in-time macro consensus remain live/vendor-only categories.

## Evidence-backed basis

| family | selected lookback | useful target horizons | result |
|---|---|---|---|
| Spot queue imbalance and microprice | latest fresh snapshot, training history: expanding/all pre-cutoff | 1s, 5s, 15s, 1m | 0.014954081 primary and 0.022972245 transfer bits/target for L1 quantity imbalance, positive in 4/4 sub-blocks. |
| Spot aggressor flow and trade sequence | 1s, 2s | 1s, 5s, 15s, 1m | Last aggressor side adds 0.12525481 primary and 0.10207176 transfer bits/target at 1s; 0.087281126 at age 2s and -0.0046537989 at age 3s. |
| Cross-market returns, volatility, and lead/lag | ETH realized volatility 30m for 1m, ETH realized volatility 60m for 5m/15m | 1m, 5m, 15m | ETH realized volatility adds 0.069524 primary / 0.036094 transfer bits at a 30m lookback for the 1m target; a 60m lookback adds 0.026971 / 0.017613 at 5m and 0.010539 / 0.005732 at 15m. The gain is magnitude information; no feature was stable at 30m or 1h. |

## Provisional basis

| family | provisional lookback | reason |
|---|---|---|
| Spot order additions, cancellations, and executions | latest synchronized update, 1s, 5s | 0.0058477776 primary and 0.020244919 transfer bits/target for the snapshot OFI proxy; one of four chronological blocks is negative. |
| Cross-exchange spot and perpetual books | Coinbase trailing return 5s, Coinbase realized volatility 5m | Sparse monthly samples show stable cross-venue information through 1m: Coinbase 5s trailing return adds 0.017814736 primary / 0.018444009 transfer bits at the 1s target; Coinbase 5m realized volatility adds 0.013721870 / 0.0060156912 at 1m. No cross-venue feature was stable at 5m or longer. |
| Liquidations and liquidation bursts | 15m count for 1s/5s targets, 1h count for 15s/1m targets | Zero-aware sparse-sample bins show stable liquidation-count information through 1m: a 15m count adds 0.0070383517 primary / 0.0075641338 transfer bits for 1s, and a 1h count adds 0.0071318950 / 0.0061818179 for 1m. No liquidation feature was stable at 5m or longer. |

## Unresolved evidence, classified

| status | families | meaning |
|---|---:|---|
| collecting-live | 6 | Causal collection is active, but there is not yet a separated continuous-history test block. |
| proxy-only | 1 | Public historical proxies were tested; the exact requested point-in-time series remains unavailable. |
| credential-required | 1 | The exact historical series requires a vendor key or licensed data. |

## Complete feature inventory

| family | data | evidence | candidate lookbacks | result |
|---|---|---|---|---|
| Spot order additions, cancellations, and executions | live-local | provisional | raw event, 1s, 2s, 5s, 15s, 30s, 1m | 0.0058477776 primary and 0.020244919 transfer bits/target for the snapshot OFI proxy; one of four chronological blocks is negative. |
| Spot queue imbalance and microprice | live-local | selected | latest fresh snapshot, 1s, 2s, 5s, 15s, 1m | 0.014954081 primary and 0.022972245 transfer bits/target for L1 quantity imbalance, positive in 4/4 sub-blocks. |
| Cross-exchange spot and perpetual books | historical-and-live | provisional | raw event, 1s, 2s, 5s, 15s, 30s, 1m | Sparse monthly samples show stable cross-venue information through 1m: Coinbase 5s trailing return adds 0.017814736 primary / 0.018444009 transfer bits at the 1s target; Coinbase 5m realized volatility adds 0.013721870 / 0.0060156912 at 1m. No cross-venue feature was stable at 5m or longer. |
| Liquidations and liquidation bursts | historical-and-live | provisional | 1s, 5s, 15s, 1m, 5m, 15m, 30m, 1h | Zero-aware sparse-sample bins show stable liquidation-count information through 1m: a 15m count adds 0.0070383517 primary / 0.0075641338 transfer bits for 1s, and a 1h count adds 0.0071318950 / 0.0061818179 for 1m. No liquidation feature was stable at 5m or longer. |
| Futures premium, basis, funding, and spot/perpetual disagreement | historical-local | rejected | 1m, 2m, 5m, 15m, 30m, 1h, 4h | Basis level added -0.018264236 primary bits/target; no positioning feature was stable. Futures log trade count, not premium, was strongest at 0.031064383. |
| Spot aggressor flow and trade sequence | historical-local | selected | raw event, 1s, 2s, 5s, 15s, 30s, 1m | Last aggressor side adds 0.12525481 primary and 0.10207176 transfer bits/target at 1s; 0.087281126 at age 2s and -0.0046537989 at age 3s. |
| Cross-market returns, volatility, and lead/lag | historical-local | selected | 1s, 2s, 5s, 15s, 30s, 1m, 2m, 5m, 15m, 30m, 1h, 4h | ETH realized volatility adds 0.069524 primary / 0.036094 transfer bits at a 30m lookback for the 1m target; a 60m lookback adds 0.026971 / 0.017613 at 5m and 0.010539 / 0.005732 at 15m. The gain is magnitude information; no feature was stable at 30m or 1h. |
| ATM implied volatility and DVOL | free-backfill | collecting-live | 1m, 2m, 5m, 15m, 30m, 1h, 4h | All 17 DVOL level/change/implied-realized candidates failed stability from 5m through 1h; historical ATM surface features remain unavailable and are accumulating live. |
| 25-delta put/call skew | live-local | collecting-live | 1m, 2m, 5m, 15m, 30m, 1h, 4h | Not yet measured: the first point-in-time surface exists, but a chronological test block does not. |
| 1d/7d/30d volatility term structure | live-local | collecting-live | 1m, 2m, 5m, 15m, 30m, 1h, 4h | Not yet measured: the first point-in-time surface exists, but a chronological test block does not. |
| Option OI, major strikes, and expiration pressure | live-local | collecting-live | current surface, 5m change, 15m change, 1h change, 4h change, 1d change | Not yet measured: the first point-in-time OI surface exists, but a chronological test block does not. |
| News events, GDELT intensity, and sentiment shocks | live-local | collecting-live | 5m decay, 15m decay, 30m decay, 1h decay, 4h decay | Not yet measured: the DOC proxy was excluded because HTTP 429 throttling stopped it before the primary/transfer span completed. Point-in-time GDELT collection remains active. |
| Scheduled macro events and standardized surprises | paid-pit | credential-required | pre-event 4h, pre-event 1h, post 1m, post 5m, post 15m, post 30m, post 1h | Not measured. |
| Exchange, whale, miner, ETF, and treasury flows | historical-local | proxy-only | 1h, 4h, 1d, 3d, 7d, 30d | No exchange, whale, miner, funding, OI, liquidation, premium, or stablecoin-flow coordinate was stable at 15m, 30m, or 1h. The test is retrospective and ETF/treasury point-in-time histories remain unavailable. |
| Live Bitcoin mempool pressure | live-local | collecting-live | 1m, 2m, 5m, 15m, 30m, 1h, 4h | All 36 mined-block fee/size/reward proxy candidates failed stability at 15m–1h. True unconfirmed-mempool snapshots are accumulating live for a later test. |
| Slow futures percentage-depth snapshots | historical-local | rejected | latest 30s snapshot, 1m, 5m, 15m | Best full-distribution result was -0.0015874785 primary bits/target and was unstable. |

## Per-family causal contract

### Spot order additions, cancellations, and executions

- Source: [Binance spot diff-depth stream; existing top-10 snapshots provide only an aggregate OFI proxy](https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams)
- History: 2026-07-25T10:51:29.758Z..2026-08-16T12:27:47.123Z snapshot history; synchronized diff-depth and aggregate-trade events have been retained since 2026-08-17
- Candidate features: bid/ask add notional, bid/ask cancel notional, execution notional, normalized OFI, queue depletion, event intensity, cancel/add ratio, price-level distance moments.
- Target horizons: 1s, 5s, 15s, 1m.
- Timing: Timestamp by exchange event and local receive time; fit transforms on training history only.
- Leakage risk: A size decrease cannot be separated into cancellation versus execution without joining trades and synchronized depth updates.
- Decision: **provisional**. 0.0058477776 primary and 0.020244919 transfer bits/target for the snapshot OFI proxy; one of four chronological blocks is negative.

### Spot queue imbalance and microprice

- Source: [Locally recorded Binance spot top-10 snapshots](https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams)
- History: 823,992 snapshots across a gappy three-week interval
- Candidate features: L1/L2/L5/L10 quantity imbalance, notional imbalance, microprice offset, spread, depth slope, snapshot deltas.
- Target horizons: 1s, 5s, 15s, 1m.
- Timing: Snapshot must predate the target boundary; discard after 5s staleness.
- Leakage risk: Local capture gaps and a short calendar span make regime transfer uncertain.
- Decision: **selected**. 0.014954081 primary and 0.022972245 transfer bits/target for L1 quantity imbalance, positive in 4/4 sub-blocks.

### Cross-exchange spot and perpetual books

- Source: [Binance + Coinbase BTC-USD level2_batch + Kraken BTC/USD book v2 + Deribit BTC-PERPETUAL](https://docs.cdp.coinbase.com/exchange/websocket-feed/channels)
- History: Ten free Tardis first-of-month UTC days across 2025-04..2025-11 and 2026-05..2026-06, plus continuous point-in-time collection since 2026-08-17
- Candidate features: venue mid returns, venue queue imbalance, spread, microprice, cross-venue mid dispersion, best executable cross-venue spread, venue lead residuals, staleness.
- Target horizons: 1s, 5s, 15s, 1m, 5m.
- Timing: Use receive time for every venue and require a maximum age per venue.
- Leakage risk: Exchange clocks are not directly comparable; event-time joins create false lead/lag unless receive-time latency is retained.
- Decision: **provisional**. Sparse monthly samples show stable cross-venue information through 1m: Coinbase 5s trailing return adds 0.017814736 primary / 0.018444009 transfer bits at the 1s target; Coinbase 5m realized volatility adds 0.013721870 / 0.0060156912 at 1m. No cross-venue feature was stable at 5m or longer.

### Liquidations and liquidation bursts

- Source: [Binance USD-M forceOrder live stream; free monthly Tardis samples; optional paid Tardis/Kaiko continuous history](https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/Liquidation-Order-Streams)
- History: 11,064 BTC liquidation events across ten free Tardis first-of-month days, plus point-in-time Binance and Deribit collection since 2026-08-17
- Candidate features: long/short liquidation notional, count, max event, signed imbalance, burst z-score, inter-arrival gap, cross-symbol breadth, post-burst decay.
- Target horizons: 1s, 5s, 15s, 1m, 5m, 15m, 30m, 1h.
- Timing: Timestamp by exchange event and local receive time; fit transforms on training history only.
- Leakage risk: Binance's stream reports only the latest liquidation per symbol in each 1s window, so it is censored burst data.
- Decision: **provisional**. Zero-aware sparse-sample bins show stable liquidation-count information through 1m: a 15m count adds 0.0070383517 primary / 0.0075641338 transfer bits for 1s, and a 1h count adds 0.0071318950 / 0.0061818179 for 1m. No liquidation feature was stable at 5m or longer.

### Futures premium, basis, funding, and spot/perpetual disagreement

- Source: [Existing Binance spot and USD-M 1m histories plus 5m positioning metrics](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api)
- History: 2025-03-18..2025-11-13 plus 2026-04-21..2026-06-23
- Candidate features: basis level, basis change, basis EMA deviation, relative spot/futures return, funding level/change, premium z-score, spot/futures activity ratio.
- Target horizons: 1s, 1m, 5m, 15m, 30m, 1h.
- Timing: A futures minute becomes usable only after close; 5m metrics retain a full 5m lag.
- Leakage risk: Repeated 5m values reduce effective sample size.
- Decision: **rejected**. Basis level added -0.018264236 primary bits/target; no positioning feature was stable. Futures log trade count, not premium, was strongest at 0.031064383.

### Spot aggressor flow and trade sequence

- Source: [Existing Binance spot aggregate-trade archive](https://github.com/binance/binance-public-data)
- History: 2025-03-18..2025-11-13 plus 2026-04-21..2026-06-23
- Candidate features: last aggressor side, trade-count imbalance, quote/base imbalance, large-trade skew, arrival centroid, flow surprise.
- Target horizons: 1s, 5s, 15s, 1m.
- Timing: Only the preceding completed one-second bin is used.
- Leakage risk: The effect is execution-latency sensitive and does not imply hour-ahead directional skill.
- Decision: **selected**. Last aggressor side adds 0.12525481 primary and 0.10207176 transfer bits/target at 1s; 0.087281126 at age 2s and -0.0046537989 at age 3s.

### Cross-market returns, volatility, and lead/lag

- Source: [Matched Binance BTC/ETH/SOL/BNB/DOGE minute histories; Databento CME/ICE remains optional for macro assets](https://databento.com/docs/knowledge-base/datasets)
- History: 462,240 aligned 1m candles per alt market across the 2025 primary and separated 2026 transfer blocks; macro assets still require a Databento account
- Candidate features: lagged return by venue/asset, realized volatility, beta residual, rolling correlation, lead-lag residual, session-open flag, data age.
- Target horizons: 1m, 5m, 15m.
- Timing: Use venue receive time where possible and closed bars otherwise; add market-open and staleness masks.
- Leakage risk: Back-adjusted continuous futures and forward-filled closed markets can manufacture lead/lag.
- Decision: **selected**. ETH realized volatility adds 0.069524 primary / 0.036094 transfer bits at a 30m lookback for the 1m target; a 60m lookback adds 0.026971 / 0.017613 at 5m and 0.010539 / 0.005732 at 15m. The gain is magnitude information; no feature was stable at 30m or 1h.

### ATM implied volatility and DVOL

- Source: [Deribit option mark-price/ticker streams and historical BTC DVOL endpoint](https://docs.deribit.com/api-reference/market-data/public-get_volatility_index_data)
- History: 47,338 hourly BTC DVOL rows are local; full Deribit option-surface collection began 2026-08-17
- Candidate features: 1d/7d/30d ATM IV, BTC DVOL, ATM IV slope, ATM bid/ask IV spread, IV z-score.
- Target horizons: 1m, 5m, 15m, 30m, 1h.
- Timing: Surface snapshot must predate prediction time; interpolate in total variance across strike and expiry.
- Leakage risk: Trade-implied IV is selection-biased; use quote/mark surfaces rather than only option trades.
- Decision: **collecting-live**. All 17 DVOL level/change/implied-realized candidates failed stability from 5m through 1h; historical ATM surface features remain unavailable and are accumulating live.

### 25-delta put/call skew

- Source: [Deribit full BTC option surface](https://docs.deribit.com/api-reference/market-data/public-ticker)
- History: Full Deribit option-surface summaries and periodic raw snapshots have been recorded since 2026-08-17
- Candidate features: 25d put IV minus 25d call IV by expiry, risk reversal, skew slope, skew curvature, skew change.
- Target horizons: 1m, 5m, 15m, 30m, 1h.
- Timing: Calculate delta from the same timestamped surface and freeze the interpolation method.
- Leakage risk: Nearest listed option can change discontinuously; interpolate by delta and maturity instead of selecting a contract after the fact.
- Decision: **collecting-live**. Not yet measured: the first point-in-time surface exists, but a chronological test block does not.

### 1d/7d/30d volatility term structure

- Source: [Deribit full BTC option surface plus local realized volatility](https://docs.deribit.com/subscriptions/market-data/markpriceoptionsindex_name)
- History: Full Deribit option-surface summaries and periodic raw snapshots have been recorded since 2026-08-17
- Candidate features: 7d-1d forward variance, 30d-7d forward variance, term slope/curvature, ATM IV minus realized volatility at matched horizon, variance-risk premium.
- Target horizons: 5m, 15m, 30m, 1h.
- Timing: Interpolate total variance, then compare with strictly trailing realized variance.
- Leakage risk: Comparing annualized IV with differently scaled realized volatility creates unit leakage and misleading levels.
- Decision: **collecting-live**. Not yet measured: the first point-in-time surface exists, but a chronological test block does not.

### Option OI, major strikes, and expiration pressure

- Source: [Deribit instrument ticker/open-interest surface](https://docs.deribit.com/api-reference/market-data/public-get_book_summary_by_instrument)
- History: Full Deribit option OI/strike/expiry snapshots have been recorded since 2026-08-17
- Candidate features: call/put OI imbalance, delta-weighted OI, gamma-weighted OI, distance to top OI strikes, distance to max-pain proxy, time to expiry, expiring notional.
- Target horizons: 5m, 15m, 30m, 1h.
- Timing: Use only open interest reported before the prediction time and preserve instrument lifecycle metadata.
- Leakage risk: Historical OI reconstructed from today's instrument set omits expired contracts and is invalid.
- Decision: **collecting-live**. Not yet measured: the first point-in-time OI surface exists, but a chronological test block does not.

### News events, GDELT intensity, and sentiment shocks

- Source: [GDELT 2.0 GKG/GCAM, optionally a lower-latency paid news feed](https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/)
- History: Filtered GDELT unigram intensity and GKG theme/tone snapshots have been recorded since 2026-08-17; a retrospective DOC proxy has 926 checkpointed 15m observations for 2026-07-18..2026-07-29 but no transfer block
- Candidate features: deduplicated story count, abnormal intensity, signed sentiment, absolute sentiment shock, sentiment change, topic/entity flags, source breadth, novelty.
- Target horizons: 15m, 30m, 1h.
- Timing: Use first observed feed time, not an article's editable publication time; deduplicate syndication.
- Leakage risk: Backfilled publication timestamps can precede the time an article entered the feed.
- Decision: **collecting-live**. Not yet measured: the DOC proxy was excluded because HTTP 429 throttling stopped it before the primary/transfer span completed. Point-in-time GDELT collection remains active.

### Scheduled macro events and standardized surprises

- Source: [BLS/Fed/FRED/ALFRED actuals plus Trading Economics point-in-time consensus calendar](https://tradingeconomics.com/api/calendar.aspx)
- History: Official actual/vintage data is free; historical pre-release consensus requires credentials
- Candidate features: event countdown, event type/importance, actual-consensus surprise, revision surprise, post-release age, simultaneous NQ/DXY/2Y reaction.
- Target horizons: 5m, 15m, 30m, 1h.
- Timing: Consensus must be the last value observed strictly before release; actual becomes visible only at release/update time.
- Leakage risk: Today's revised previous value and final consensus must never replace what was displayed before a historical release.
- Decision: **credential-required**. Not measured.

### Exchange, whale, miner, ETF, and treasury flows

- Source: [Coin Metrics network data; CryptoQuant/Glassnode labeled flows; issuer ETF holdings; SEC filings](https://docs.coinmetrics.io/api/v4/)
- History: 1,972 Coin Metrics daily rows plus 1,360 CC-BY community daily rows across 29 whale/miner/liquidation/derivatives metrics; values are retrospectively revised
- Candidate features: exchange inflow/outflow/netflow, large transfers, miner-to-exchange flow, miner reserves, ETF net holdings change, treasury filing impulse, flow z-scores.
- Target horizons: 15m, 30m, 1h.
- Timing: Use vendor observation/publication time and carry the most recently known value with an explicit age feature.
- Leakage risk: Wallet labels are revised retrospectively; vendor history is not point-in-time unless snapshots were archived live.
- Decision: **proxy-only**. No exchange, whale, miner, funding, OI, liquidation, premium, or stablecoin-flow coordinate was stable at 15m, 30m, or 1h. The test is retrospective and ETF/treasury point-in-time histories remain unavailable.

### Live Bitcoin mempool pressure

- Source: [mempool.space public API or a self-hosted Bitcoin Core node](https://mempool.space/docs/api/rest)
- History: 2,195 three-year mined-block proxy rows plus point-in-time mempool snapshots recorded since 2026-08-17
- Candidate features: mempool vbytes, transaction count, fee histogram, projected-block depth, recommended fee curve, arrival rate, large-transaction rate, block-arrival shock.
- Target horizons: 1m, 5m, 15m, 30m, 1h.
- Timing: Archive local receive time and node height; reset rate features after capture gaps.
- Leakage risk: Public mempool snapshots are observer-dependent; a later full-node reconstruction cannot reproduce what was unconfirmed earlier.
- Decision: **collecting-live**. All 36 mined-block fee/size/reward proxy candidates failed stability at 15m–1h. True unconfirmed-mempool snapshots are accumulating live for a later test.

### Slow futures percentage-depth snapshots

- Source: [Existing Binance USD-M ±1–5% percentage-depth archive](https://data.binance.vision/)
- History: 368 days
- Candidate features: depth imbalance, total depth, near/far concentration, snapshot changes.
- Target horizons: 1s, 1m, 5m, 15m.
- Timing: Use only a snapshot from an earlier second; discard after 120s.
- Leakage risk: Cumulative percentage bands omit top-of-book state and are too slow for queue dynamics.
- Decision: **rejected**. Best full-distribution result was -0.0015874785 primary bits/target and was unstable.

## Reproduction and live acquisition

- `npm run fetch:research-forward-market -- --ranges 2026-07-18..2026-08-16` rebuilds the research-only spot-flow and Binance derivatives layer without touching the sealed oracle corpus. Official SHA-256 checksums are verified and source ZIPs are discarded after reduction.
- `npm run fetch:gdelt-history` builds the reduced 15-minute retrospective crypto-news count/tone proxy in sub-72-hour API chunks. Missing intervals remain null, HTTP 429 responses back off, and completed chunks are checkpointed before the immutable artifact is written.
- `npm run analysis:global-feature-basis:export-30d` reconstructs the 147-coordinate common-coverage dataset. `npm run analysis:global-feature-basis -- --input-dir data/runtime-cache/global-feature-basis-30d --output data/benchmarks/global-return-feature-basis-30d.json --report docs/experiments/global-return-feature-basis-30d-2026-08-17.md` reruns the joint search.
- `npm run fetch:external-public` refreshes the free DVOL, Coin Metrics, community daily, mempool-proxy, and current Deribit surface data.
- `npm run analysis:external-public` rebuilds the held-out information screen in `docs/experiments/public-external-feature-information-2026-08-17.md` and its machine-readable JSON artifact.
- `npm run fetch:tardis-samples` streams and reduces the ten free first-of-month cross-venue quote and liquidation samples without retaining raw tick CSV; `npm run analysis:tardis-samples` rebuilds their 1s–1h screen.
- `npm run collect:external-live` records synchronized exchange, liquidation, premium, Deribit surface, GDELT, and mempool observations under `data/market/mutable/external-live`.
- `npm run analysis:external-live-early` rebuilds the horizon-aware live readiness, storage, feature-variation, and early held-out likelihood screen. The active watcher records immutable 1h and 24h checkpoints as collection reaches them.
- The concrete model tensor, horizon routing, causal lookbacks, and selected/provisional/excluded split are summarized in `docs/experiments/recommended-model-input-basis-2026-08-17.md`.
- Live JSONL is stored as concatenated complete gzip members. Each source flushes at least once per second, so files remain readable while the collector runs and a crash loses at most the current in-memory chunk.
- Set `TRADING_ECONOMICS_API_KEY` for point-in-time macro consensus and `BLOCKWORKS_API_KEY` for ETF flows. Every public source operates without credentials.

## Acquisition blockers recorded

- Binance no longer publishes its historical liquidationSnapshot archive. Ten free first-of-month Tardis samples now provide sparse event evidence; continuous history still requires a vendor or new live collection.
- Ten free first-of-month Tardis quote samples now cover synchronized Binance, Coinbase, Kraken, and Deribit top-of-book state. Continuous L2/L3 books and complete historical Deribit option surfaces remain vendor datasets.
- Binance's public BTCUSDT option EOHSummary archive ends on 2023-10-23, so it cannot provide a recent 30-day surface substitute; current BVOL index data remains available but does not contain strike/delta/OI structure.
- Macro consensus forecasts need a point-in-time calendar vendor. Official BLS/FRED data supplies actuals and revisions but not historical pre-release consensus.
- CryptoQuant documents that exchange-wallet clustering revisions make its historical exchange-flow endpoint non-point-in-time.
- The current environment has no Databento, Trading Economics, Blockworks, Glassnode, CryptoQuant, or FRED credentials.
- Live point-in-time collection is now running. Order events, option surfaces, GDELT, and true mempool state still need enough forward history before a continuous-history impact test is possible.

## Live evidence timing and retention

- One hour is a feed-health and very-large-effect smoke test for 1s–15s targets, not a feature rejection window.
- One day is the first early screen for 1s–1m targets. It is still one market regime, so a negative score alone does not justify deletion.
- GDELT has about 96 observations/day; use at least seven separated days for its first screen. A 1h target has only 24 non-overlapping outcomes/day and needs roughly 30–90 days.
- The unresolved options, premium, GDELT, and mempool feeds are low-volume. The raw exchange books dominate storage; compact those to causal 1s derived state after reconstruction tests instead of cutting the slow candidates early.

## Selection protocol once data exists

1. Freeze transformations and quantile cuts on training history only.
2. Evaluate every declared lookback at 1s, 5s, 15s, 1m, 5m, 15m, 30m, and 1h where the source cadence permits.
3. Score incremental held-out log likelihood after the current basis, plus sign, zero-gate, CRPS, and tail calibration.
4. Require positive improvement in two chronological primary halves and one separated transfer block.
5. Within a correlated family, keep the shortest stable lookback; keep another only when it adds conditional information.
6. Repeat greedy forward selection after every accepted family. Individual marginal gains are not additive.

Machine-readable details are stored in `data/benchmarks/external-feature-basis-audit.json`.
