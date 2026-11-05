# Collector storage capacity — 2026-11-01

For **460 manifest entries / 687 distinct Binance venue+symbol routes** (261 spot, 426 USD-M), budget 30 days of 24/7 recording: **2,592,000 seconds**. The main raw-public-replay estimate is **3.31 TB (3.01 TiB)**; the 2–3x stress range is **6.62 TB (6.02 TiB)–9.93 TB (9.03 TiB)**. These figures include public peripheral feeds and explicit metadata/checkpoint allowances. Optional derived rows are shown separately. Source snapshots were taken November 1; later report regeneration does not refresh those observations.

| Profile | Base gzip | 2x volume | 3x volume |
|---|---:|---:|---:|
| Current BTC compact collector, with raw trade feeds | 6.48 GB (6.03 GiB) | 12.95 GB (12.06 GiB) | 19.43 GB (18.09 GiB) |
| All routes, hypothetical 1s features only (lossy) | 995.29 GB (926.94 GiB) | 1.99 TB (1.81 TiB) | 2.99 TB (2.72 TiB) |
| All routes, full public replay + current BTC peripherals (default) | 3.31 TB (3.01 TiB) | 6.62 TB (6.02 TiB) | 9.93 TB (9.03 TiB) |
| Full replay plus optional per-event derived rows | 4.10 TB (3.73 TiB) | 8.20 TB (7.46 TiB) | 12.31 TB (11.19 TiB) |

Full profile uncompressed JSONL: **20.90 TB (19.01 TiB)** base. Add at least 25% free-space/rotation/index allowance: **12.41 TB (11.29 TiB)** usable capacity for the 3x case. An independent backup adds another copy. This is a capacity envelope; the short sample cannot establish a month-long stochastic traffic distribution.

## What is retained

The default full profile retains spot individual trades, USD-M aggregate trades at the exchange's available granularity, every published 100ms L2 diff, unthrottled BBO, local 5-minute checkpoints, session/sequence/gap markers, 1s futures mark/funding and rotating 300s open interest. REST initial/resync seed allowance is two per route per day. Checkpoints are local reconstructed known levels, not repeated REST downloads. Seed sizes are measured at 5,000 spot / 1,000 futures levels per side; the model reserves 2x their size for checkpoint growth. Optional per-depth BBO and per-trade causal book-alignment rows duplicate reconstructible information and are off by default; the fourth profile includes their explicit storage allowance. Turning those derived rows off preserves public replay quality.

Public L2 does not distinguish cancellations from executions, reveal trader IDs, or provide complete queue/L3 state. Sparse local checkpoints cannot restore unknown depth beyond the initial REST range. Sequence gaps remain gaps until resynchronized. The full profile is replay of what the public feed publishes, not all matching-engine events.

The feature-only row is a hypothetical lower-volume design. It does not preserve within-second order, cancellation/depletion paths or event timing, and is not selectable as the implemented multi-asset full collector. The existing BTC compact collector also keeps raw aggregate/perpetual/option trades, so it is not a pure feature-only profile.

## Empirical basis and assumptions

Inputs: [saved measurement basis](collector-storage-capacity-2026-11-01-inputs.json); machine-readable [estimate](collector-storage-capacity-2026-11-01.json) and [per-route CSV](collector-storage-capacity-2026-11-01.csv). Universe SHA-256: `13530728d11831f51bd35dcbc2c4dd7768ee197e0ee4aa1efa923ff45227c26d`.

Activity classes per venue: `{"spot":{"quiet":99,"medium":152,"busy":9,"very-busy":1},"usdm-futures":{"quiet":0,"medium":362,"busy":50,"very-busy":14}}`. TradFi routes: **344**; missing ticker rows: **0**. Spot message counts use 24-hour individual fill counts. Futures aggregate rates divide fill counts by the empirically sampled fills-per-message ratio. Depth/BBO rates and bytes use measured own-market samples where available, then matched venue/activity classes. Crypto tiers are <0.1, <1, <10 and ≥10 fills/s. Markets are not all assigned BTC rates.

TradFi receives positive planning floors of 0.5 fills/s, 5 L2 messages/s and 10 BBO messages/s, running 24/7, because the November 1 observation was Sunday. Those are conservative assumptions, not measured weekday averages or claims about venue trading hours.

Actual full-recorder calibration uses **12,295 wire messages** from the 35s BTC spot/futures smoke in `data/runtime-cache/collector-capacity/full-btc-smoke`. For each venue/stream, the same messages are encoded using the actual full envelope and a reconstructed legacy envelope, then regrouped into 1s gzip level-6 members. The measured byte ratio scales each activity class's legacy-shaped samples. Initial snapshots/control records are excluded from that ratio and budgeted separately; the complete smoke's 1,328,605 gzip bytes are not divided by 35s and extrapolated as recurring traffic.

| Full-recorder stream | Rows | gzip B/row, 1s groups | Full / legacy gzip |
|---|---:|---:|---:|
| spot:depth_bbo | 338 | 81.4 | derived row |
| spot:bookTicker | 2,135 | 34.7 | 1.837 |
| spot:depth@100ms | 338 | 196.1 | 1.378 |
| spot:trade | 795 | 43.0 | 2.133 |
| spot:trade_book_alignment | 795 | 45.9 | derived row |
| usdm-futures:depth_bbo | 303 | 92.8 | derived row |
| usdm-futures:markPrice@1s | 31 | 494.2 | 2.411 |
| usdm-futures:aggTrade | 203 | 108.0 | 2.059 |
| usdm-futures:trade_book_alignment | 203 | 108.0 | derived row |
| usdm-futures:bookTicker | 8,490 | 30.7 | 1.677 |
| usdm-futures:depth@100ms | 303 | 537.2 | 1.160 |

At default parameters, these measured ratios and derived-row sizes replace the fallback 1.5x / 250B / 220B assumptions. The raw-envelope CLI multiplier scales calibration relative to 1.5; the derived-byte CLI values scale their measured sizes relative to 250/220. Only a missing calibration uses those fallback sizes literally. Quiet markets may have different envelope amortization; the 2–3x scenarios cover volume sensitivity rather than establish confidence bounds.

Measured legacy archive examples (largest available file/source, not all history):

| Source | Rows | Supported hours | gzip B/row | JSONL B/row |
|---|---:|---:|---:|---:|
| binance-spot-aggtrade | 2,495,347 | 24.00 | 32.5 | 278.5 |
| binance-spot-depth-diff | 346,195 | 9.62 | 172.2 | 1007.4 |
| binance-spot-depth-snapshot | 1 | 0.00 | 48016.0 | 320094.0 |
| binance-usdm-book-ticker | 223,291 | 9.62 | 60.3 | 275.1 |
| binance-usdm-liquidations | 90,997 | 23.68 | 95.0 | 320.6 |
| binance-usdm-mark-price | 86,130 | 23.88 | 205.3 | 294.0 |
| binance-usdm-premium-index | 1,440 | 24.00 | 201.6 | 271.0 |
| coinbase-btcusd-level2 | 619,248 | 9.62 | 121.9 | 690.4 |
| cross-exchange-book-1s | 86,104 | 24.00 | 791.1 | 1735.6 |
| deribit-btc-option-summary | 1,376 | 22.93 | 1798.3 | 6610.7 |
| deribit-btc-option-trades | 24,715 | 23.16 | 260.3 | 610.0 |
| deribit-btc-perpetual-book | 113,539 | 9.62 | 259.1 | 1031.8 |
| deribit-btc-perpetual-trades | 58,111 | 23.95 | 317.2 | 1373.7 |
| gdelt-crypto-news | 98 | 23.92 | 1121.6 | 3360.6 |
| kraken-btcusd-book | 3,041,955 | 9.62 | 29.3 | 306.4 |
| mempool-live | 1,440 | 24.00 | 3355.9 | 8321.1 |

Supported hours exclude entire receive-time deltas above 60s for event streams, 2.5s for 1s streams and 2.5 polling periods for low-cadence rows. This deliberately does not divide by the Aug 17–30 calendar span. It is a heuristic: it can exclude genuine sparse-market silence and cannot detect short packet losses. Single seed snapshots have no observation-duration rate; their bytes are used only per snapshot.

Compression experiments group the same received rows into synthetic 1s and 60s gzip level-6 members. Larger batches reduce repetitive metadata overhead, especially on quiet feeds; the input JSON retains both measured byte totals. The live production assumption remains 1s batches. Binary/Parquet/Zstd savings are unmeasured and are not credited. Checkpoint gzip is measured as a standalone snapshot, not a long homogeneous tape.

For evaluating an optimized fixed-schema binary/Parquet+Zstd archive, an **unmeasured what-if** range of 0.4–0.8 times this gzip base would be **1.32 TB (1.20 TiB)–2.65 TB (2.41 TiB)**. This is a sensitivity assumption, not an observed compression ratio or a provisioning recommendation. Preserve raw payloads or a proven lossless reconstruction path and benchmark representative busy and quiet days before claiming those savings.

## Optional participant-observable DEX pilot

Hyperliquid BTC trades/BBO/limited-depth periodic L2 snapshots are separate from the Binance total. The 20s sample excludes the initial replay of historical trades for rate fitting. A 12s watched-address smoke confirms channel accessibility but includes initialization and is not a population-rate estimate. The configurable account allowance is 2 messages/s/address at 400 compressed bytes/message across order/fill/state updates. Ten watched accounts are a bounded pilot, not all exchange participants. Address remains a proxy for identity; economic entity, strategy, non-actions and off-venue information are latent.

```json
{
  "publicTapeMonthGzipBytes": 1448623013.0960712,
  "publicTape3xBytes": 4345869039.288214,
  "accountAllowances": [
    {
      "watchedAccounts": 0,
      "monthGzipBytes": 0
    },
    {
      "watchedAccounts": 1,
      "monthGzipBytes": 2073600000
    },
    {
      "watchedAccounts": 10,
      "monthGzipBytes": 20736000000
    }
  ]
}
```

## Reproduce

```powershell
node --import tsx scripts/estimate-collector-storage.ts --universe docs/portfolio/restored-market-universe.json
node --import tsx --test scripts/lib/collector-storage.test.ts
```

Optional measurement refresh (opens bounded public connections for the live/REST commands; the archive audit is local):

```powershell
node --import tsx scripts/measure-collector-storage.ts --sample-seconds 60
node --import tsx scripts/measure-collector-storage.ts --sample-seconds 20 --output data/research/collector-storage/sector-live-measurements-2026-11-01.json
node --import tsx scripts/measure-collector-storage.ts --rest
node --import tsx scripts/measure-collector-storage.ts --audit-archives
node --import tsx scripts/estimate-collector-storage.ts --refresh-inputs
```

The measurement modes save distinct live/rest/archive files. All live samples stop themselves; none starts a persistent collector. The normal estimate command is offline and reads the committed input basis, including the optional DEX sample budget.

- Capacity scenarios, not guaranteed event-rate forecasts. 60s crypto and 20s sector samples are short, Sunday-biased observations.
- TradFi floors budget 24/7 operation even when underlying cash markets close; no zero-activity month is extrapolated from Sunday.
- Full public replay means exact published L2 deltas, trades or exchange aggregates, BBO, bounded checkpoints and explicit gaps. Public L2 cannot identify cancellations separately from fills or expose participant IDs.
- Rows with unavailable credentials/history are not invented as stored feeds. Existing peripheral collection is BTC-specific, not every asset.
- Compressed baseline uses per-source one-second gzip members. Rich envelope overhead and optional derived-row sizes are calibrated from the 35s full-recorder smoke; broad market traffic and quiet-feed transfer remain uncertain.
- Archive observed time is a receive-time gap-clipped support heuristic, not calendar span, connectivity certification or a guarantee of complete delivery.
- Stress multipliers cover changing activity, changed levels per depth message and size growth, not merely message cadence; maintain separate free-space and backup capacity.
- Binary/Parquet/Zstd alternatives have not been measured; do not apply an assumed 5x or 10x saving to provision storage.
