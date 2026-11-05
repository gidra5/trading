# Lossless collector compression benchmark

Generated 2026-11-02T09:48:12.633Z, Node v25.9.0. Experimental measurement only; no production archival integration.

Every selected original JSONL byte roundtrips: raw exchange strings, decimal spelling, unknown fields, unsafe integer lexemes, whitespace, key ordering and row ordering. No events or fields are removed within the measured windows. The lexical codec uses repeated structural templates and field dictionaries, BigInt integer/decimal deltas, and Zstandard. Decimal text is reconstructed exactly, rather than rounded or converted to floating point.

Selected candidate: **lexical-zstd19-3600s**. The table uses the **worst observed ratio per component** across the measured full-recorder and legacy windows. Baseline is the same records grouped into 1-second gzip level-6 members, matching the previous conservative per-component capacity calibration.

| Component | Remaining bytes vs baseline | Compression improvement | Evidence windows |
| --- | ---: | ---: | ---: |
| spot/BBO | 22.40% | 4.46× | 1 |
| spot/depth-diff | 52.20% | 1.92× | 2 |
| spot/trade | 22.52% | 4.44× | 2 |
| usdm-futures/mark-price-funding | 15.61% | 6.41× | 3 |
| usdm-futures/aggTrade | 23.26% | 4.30× | 2 |
| usdm-futures/BBO | 24.24% | 4.12× | 3 |
| usdm-futures/depth-diff | 56.52% | 1.77× | 2 |

The 3× activity estimate of 9.93 TB requires remaining storage below approximately 10.07% to fit 1 TB, or 8.06% with 20% capacity reserved. These measured component ratios do not demonstrate that reduction. In particular, depth remains above half its previous gzip size in the conservative transfer, which prevents the measured design from meeting the 1 TB stressed-month target. This is a result for the measured design, not an information-theoretic impossibility proof for every lossless codec.

| Actual dataset | Rows | Observed seconds | Selected uncompressed MiB |
| --- | ---: | ---: | ---: |
| full-btc-smoke/spot-public-000 | 3268 | 33.826 | 2.78 |
| full-btc-smoke/usdm-futures-market-000 | 234 | 30.308 | 0.20 |
| full-btc-smoke/usdm-futures-public-000 | 8793 | 30.821 | 8.25 |
| multiasset-lossless-sample/usdm-futures-market-000 | 7888 | 88.793 | 7.32 |
| multiasset-lossless-sample/usdm-futures-public-000 | 32276 | 21.242 | 32.00 |
| legacy/binance-spot-aggtrade | 40000 | 1180.581 | 10.63 |
| legacy/binance-spot-depth-diff | 35978 | 3599.967 | 22.63 |
| legacy/binance-usdm-book-ticker | 18175 | 3599.972 | 4.77 |
| legacy/binance-usdm-mark-price | 3600 | 3598.992 | 1.01 |

The 20-market capture ran for 90 seconds. Its busy public shard reaches the selected 32 MiB window cap after about 21 seconds; that shorter observed duration is explicit above. The market shard spans about 89 seconds. Legacy depth/BBO/mark windows are approximately one hour, while the trade window reaches the 40,000-row cap after about 20 minutes. Legacy spot aggregate trades have different granularity from the newer individual trade feed.

| Actual shard | Whole 1s gzip / separated kinds | Chosen encoded bytes |
| --- | ---: | ---: |
| full-btc-smoke/spot-public-000 | 88.62% | 58110 |
| full-btc-smoke/usdm-futures-market-000 | 72.29% | 4606 |
| full-btc-smoke/usdm-futures-public-000 | 101.82% | 145844 |
| multiasset-lossless-sample/usdm-futures-market-000 | 91.35% | 134568 |
| multiasset-lossless-sample/usdm-futures-public-000 | 104.09% | 649513 |

These shard rows directly test cross-stream/market envelope amortization. The mixed market shard saves about 9% versus separated component gzip members; the mixed public shard is slightly larger. They do not support an additional order-of-magnitude baseline discount. Do not multiply these shard ratios into the component codec ratios as independent savings.

| Dataset / component | 60s gzip ratio | Hour-bucket raw Zstd12 ratio | Hour-bucket lexical Zstd19 ratio | Chosen encode / decode ms |
| --- | ---: | ---: | ---: | ---: |
| full-btc-smoke/spot-public-000 / spot/BBO | 78.26% | 50.47% | 22.40% | 158.9 / 12.4 |
| full-btc-smoke/spot-public-000 / spot/depth-diff | 69.03% | 57.03% | 52.20% | 65.9 / 10.5 |
| full-btc-smoke/spot-public-000 / spot/trade | 60.50% | 39.36% | 18.07% | 63.3 / 6.0 |
| full-btc-smoke/spot-public-000 / whole-actual-shard | 85.97% | 61.27% | 37.58% | 278.9 / 30.9 |
| full-btc-smoke/usdm-futures-market-000 / usdm-futures/mark-price-funding | 13.80% | 11.90% | 11.05% | 5.6 / 0.5 |
| full-btc-smoke/usdm-futures-market-000 / usdm-futures/aggTrade | 44.13% | 30.35% | 15.26% | 18.1 / 2.4 |
| full-btc-smoke/usdm-futures-market-000 / whole-actual-shard | 43.04% | 30.31% | 17.11% | 21.2 / 2.1 |
| full-btc-smoke/usdm-futures-public-000 / usdm-futures/BBO | 93.95% | 63.23% | 20.48% | 640.3 / 58.3 |
| full-btc-smoke/usdm-futures-public-000 / usdm-futures/depth-diff | 83.47% | 71.46% | 56.52% | 179.8 / 37.0 |
| full-btc-smoke/usdm-futures-public-000 / whole-actual-shard | 94.69% | 68.72% | 33.82% | 805.5 / 101.5 |
| multiasset-lossless-sample/usdm-futures-market-000 / usdm-futures/aggTrade | 84.36% | 55.11% | 23.26% | 521.4 / 50.1 |
| multiasset-lossless-sample/usdm-futures-market-000 / usdm-futures/mark-price-funding | 49.96% | 32.38% | 15.61% | 149.8 / 13.7 |
| multiasset-lossless-sample/usdm-futures-market-000 / whole-actual-shard | 87.92% | 53.37% | 24.20% | 717.1 / 71.6 |
| multiasset-lossless-sample/usdm-futures-public-000 / usdm-futures/BBO | 98.55% | 61.08% | 24.24% | 2634.3 / 220.9 |
| multiasset-lossless-sample/usdm-futures-public-000 / usdm-futures/depth-diff | 97.40% | 73.26% | 51.60% | 678.5 / 148.9 |
| multiasset-lossless-sample/usdm-futures-public-000 / whole-actual-shard | 99.14% | 66.63% | 34.16% | 3221.5 / 360.8 |
| legacy/binance-spot-aggtrade / spot/trade | 84.93% | 65.94% | 22.52% | 1345.7 / 154.1 |
| legacy/binance-spot-depth-diff / spot/depth-diff | 74.41% | 64.54% | 44.86% | 4039.6 / 460.8 |
| legacy/binance-usdm-book-ticker / usdm-futures/BBO | 52.97% | 46.47% | 19.77% | 609.6 / 61.0 |
| legacy/binance-usdm-mark-price / usdm-futures/mark-price-funding | 18.53% | 15.65% | 7.14% | 125.7 / 12.5 |

An hour bucket is not an observed full hour when the source is shorter or capped. The complete JSON also records 60-second lexical/Zstd candidates, baseline and encoded byte counts, timings, source paths and SHA-256 hashes of every selected original record window. The measured codec header/payload overhead is included. Compression timings are single-run local observations, not production throughput guarantees.

Unmeasured checkpoints, REST seeds, open interest, coverage/control and peripheral feed kinds retain a 1.0 transfer ratio in capacity calculations. A short Sunday mixed TradFi sample does not establish weekday entropy, month-long activity or 3× spike compression behavior. Activity extrapolation remains a separate capacity scenario.

The prototype decoder is for bounded trusted benchmark inputs. It has no production integrity checksum or decoded-size/structure budget in its on-disk header; the benchmark independently compares every decoded block against the original bytes and records source SHA-256. Production packing would additionally require verified archival integrity, bounded decoding, crash recovery, indexes and retention design. No source files are deleted or replaced.

Reproduce measurements:

```powershell
node --import tsx scripts/benchmark-collector-lossless.ts --level19
node --import tsx --test scripts/lib/lossless-jsonl-block.test.ts
```

Regenerate summary without rerunning compression:

```powershell
node --import tsx scripts/benchmark-collector-lossless.ts --summarize
```
