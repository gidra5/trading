# Can all-event, all-market collection fit 1 TB/month?

Scope remains **460 economic identities / 687 physical routes**, every published event, 30 days. This report changes no collector, universe, or retention settings. All amounts use decimal TB/GB.

The earlier per-route gzip model is **3.309 TB base / 9.928 TB at 3× activity**. A 1 TB physical cap with 20% reserved for transient WAL, indices, rotation and working space permits **800.0 GB of monthly archive data**. The complete archive would need to retain at most **8.06%** of the earlier gzip footprint, a **12.41× reduction**. This is a feasibility target, not an observed compression result.

The model contains 77.74 billion records at 3×: the 800 GB archive allowance averages **10.29 bytes per modeled record**. That is not an entropy lower bound—batching can share metadata—but it makes a 10× claim demanding.

## Where the modeled bytes go

| Component | Base gzip | 3× gzip | Share |
|---|---:|---:|---:|
| usdm-futures/BBO | 872.1 GB | 2616.4 GB | 26.4% |
| usdm-futures/depth-diff | 707.1 GB | 2121.4 GB | 21.4% |
| usdm-futures/mark-price-funding | 545.7 GB | 1637.1 GB | 16.5% |
| spot/BBO | 507.7 GB | 1523.0 GB | 15.3% |
| spot/depth-diff | 297.0 GB | 891.1 GB | 9.0% |
| usdm-futures/aggTrade | 183.0 GB | 549.0 GB | 5.5% |
| usdm-futures/local-checkpoint | 65.5 GB | 196.5 GB | 2.0% |
| spot/local-checkpoint | 56.0 GB | 167.9 GB | 1.7% |
| spot/trade | 48.4 GB | 145.1 GB | 1.5% |
| BTC-peripherals/kraken-btcusd-book | 6.7 GB | 20.0 GB | 0.2% |
| BTC-peripherals/coinbase-btcusd-level2 | 5.7 GB | 17.0 GB | 0.2% |
| usdm-futures/session-gap-health | 4.6 GB | 13.8 GB | 0.1% |
| spot/session-gap-health | 2.8 GB | 8.5 GB | 0.1% |
| BTC-peripherals/deribit-btc-perpetual-book | 2.2 GB | 6.6 GB | 0.1% |
| BTC-peripherals/cross-exchange-book-1s | 2.0 GB | 6.1 GB | 0.1% |
| usdm-futures/open-interest | 0.9 GB | 2.8 GB | 0.0% |
| BTC-peripherals/deribit-btc-perpetual-trades | 0.6 GB | 1.7 GB | 0.0% |
| BTC-peripherals/binance-usdm-liquidations | 0.3 GB | 0.8 GB | 0.0% |
| usdm-futures/REST-seed-and-resync | 0.2 GB | 0.7 GB | 0.0% |
| BTC-peripherals/option-surface-raw | 0.2 GB | 0.6 GB | 0.0% |
| BTC-peripherals/deribit-btc-option-trades | 0.2 GB | 0.6 GB | 0.0% |
| spot/REST-seed-and-resync | 0.2 GB | 0.6 GB | 0.0% |
| BTC-peripherals/mempool-live | 0.1 GB | 0.4 GB | 0.0% |
| BTC-peripherals/deribit-btc-option-summary | 0.1 GB | 0.2 GB | 0.0% |
| BTC-peripherals/binance-usdm-premium-index | 0.0 GB | 0.0 GB | 0.0% |
| BTC-peripherals/gdelt-crypto-news | 0.0 GB | 0.0 GB | 0.0% |

## Measured lossless result

Candidate: **lexical-zstd19-3600s**. Worst measured ratio per component across corresponding full-recorder and legacy windows. Legacy aggregate trades are a conservative granularity transfer, not a new individual-trade measurement. Unmeasured sources remain unchanged. No additional global multi-market baseline correction is credited.

Measured component mappings cover 3161.0 GB of the earlier base footprint; 148.3 GB remains unmeasured and receives no compression saving.

No global shard-baseline correction is applied. The measured-component projection retains 35.3% of the earlier conservative modeled baseline, including unchanged unmeasured sources. This is a conditional transfer estimate, not a direct month-long 687-route measurement.

| Component | Worst measured encoded / 1s gzip | Byte-exact round trip |
|---|---:|---|
| spot/BBO | 22.4% | yes |
| spot/depth-diff | 52.2% | yes |
| spot/trade | 22.5% | yes |
| usdm-futures/mark-price-funding | 15.6% | yes |
| usdm-futures/aggTrade | 23.3% | yes |
| usdm-futures/BBO | 24.2% | yes |
| usdm-futures/depth-diff | 56.5% | yes |

| Activity scenario | Projected archive | Archive + 200 GB reserve | Within 1 TB? |
|---|---:|---:|---|
| 1× | 1.167 TB | 1.367 TB | no |
| 2× | 2.334 TB | 2.534 TB | no |
| 3× | 3.500 TB | 3.700 TB | no |

The existing 3.31 TB baseline prices gzip using separate per-route/per-stream 1s groups. Production shares metadata across multi-market socket shards; especially the 1s mark/funding rows can be cheaper in a matched shard. Benchmarking must distinguish this baseline correction from the additional gain of the lossless codec. BTC-only busy-stream ratios cannot establish quiet TradFi or 687-route traffic costs.

Matched physical-shard measurements (each candidate uses exactly the baseline's same rows; short windows are not full-hour observations):

| Dataset | Observed seconds | Kind-separated 1s gzip | Physical-shard 1s gzip | Encoded / physical shard |
|---|---:|---:|---:|---:|
| full-btc-smoke/spot-public-000 | 33.83 | 174,495 B | 154,646 B | 37.6% |
| full-btc-smoke/usdm-futures-market-000 | 30.31 | 37,243 B | 26,922 B | 17.1% |
| full-btc-smoke/usdm-futures-public-000 | 30.82 | 423,580 B | 431,271 B | 33.8% |
| multiasset-lossless-sample/usdm-futures-market-000 | 88.79 | 608,780 B | 556,096 B | 24.2% |
| multiasset-lossless-sample/usdm-futures-public-000 | 21.24 | 1,826,716 B | 1,901,474 B | 34.2% |

These are matching selected wire records regrouped by physical source, not extrapolated bootstrap totals. Optional derived rows and separately stored checkpoints/control records are outside these codec samples and receive no unmeasured saving. The multi-market benchmark's per-kind groups already combine markets. Their grouping comparison isolates additional sharing across stream kinds, not the full difference between 20 markets and 20 isolated one-market files. No further unmeasured sharing factor is multiplied into the global projection.

All accepted codec ratios must be measured against the same input records and pass byte-exact round-trip checks, including raw public payload strings, timestamps, ordering, identifiers, session/sequence fields and gaps. Recompression, dictionary/schema references, reversible integer/delta encoding and longer archive frames can reduce repetition. Lossless recompression does not supply missing exchange events or resolve public L2 into cancellation identities.

## Operational interpretation

A future archiver can persist short durable WAL chunks, repack only sealed chunks into longer compressed frames, verify exact reconstruction, and retain originals until verification finishes. Transient duplicated chunks and indices need working capacity. Increasing the live flush interval alone trades away crash durability; it is not required to obtain a longer offline compression window.

Local full-book checkpoints are derived caches of retained REST seeds plus L2 deltas. Replacing full checkpoint copies with replay offsets, or constructing them on demand, preserves all published events while changing restore/random-access work; that additional unmeasured encoding gain is not credited. Independent BBO events cannot generally be reconstructed from the 100ms depth feed and must remain in an all-event archive.

The following future-cadence scenarios reduce only duplicate local checkpoint snapshots. All published wire messages, REST initial/resync seeds and session/gap/quality records remain. This preserves market-event information; a longer initial replay is required when seeking or restarting. Existing archived checkpoint rows are not removed or rewritten by this report.

| Future local checkpoint cadence | Base before codec | 3× before codec | 3× saving vs 5m |
|---|---:|---:|---:|
| 5 min | 3.309 TB | 9.928 TB | 0.0 GB |
| 1 hour | 3.198 TB | 9.594 TB | 334.1 GB |
| 1 day | 3.188 TB | 9.565 TB | 363.2 GB |

With the same measured codec transfers, changing only the future checkpoint cadence gives:

| Cadence | Base archive | 3× archive | 3× + working reserve |
|---|---:|---:|---:|
| 300s | 1.167 TB | 3.500 TB | 3.700 TB |
| 3600s | 1.055 TB | 3.166 TB | 3.366 TB |
| 86400s | 1.046 TB | 3.137 TB | 3.337 TB |

A physical 1 TB cap cannot guarantee 30 days of unrestricted future event volume. If the measured all-event projection exceeds the cap, the answer is additional lossless engineering, additional capacity, or an explicit user-approved change of scope. This report does not discard assets, downsample the book, shorten 30-day raw history, or silently drop records to enforce a cap.

## Reproduce

```powershell
node --import tsx scripts/report-collector-one-tb.ts
node --import tsx --test scripts/report-collector-one-tb.test.ts
```

Inputs: [earlier capacity model](collector-storage-capacity-2026-11-01.json), [actual lossless benchmark](collector-lossless-codec-2026-11-02.json), [explicit transfer assumptions](collector-one-tb-2026-11-02-codec-basis.json). Machine-readable result: [JSON](collector-one-tb-2026-11-02.json).
