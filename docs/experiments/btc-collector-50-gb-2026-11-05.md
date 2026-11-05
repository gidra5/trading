# BTC-only collector storage at 50 GB per month

The core archive covers Binance BTCUSDT spot and USD-M perpetual, the
Bitcoin mainnet confirmed block journal, and reconstruction data. The
exchange tape retains published trades at each market's available
granularity, BBO, depth updates, and futures mark/funding. Open interest,
REST snapshots, checkpoints and coverage records receive a separate
allowance. This scope covers one spot route and one perpetual route.

## Measured rates

A 105-second live run on November 5 recorded 80,840 wire records on the
three full exchange shards. The run also recorded one spot startup
snapshot retry and a subsequent bridge. The estimates multiply bytes
per observed second by 2,592,000 and use decimal GB. Each Zstd result
restored the complete original JSONL bytes, including the raw payload
strings and within-file ordering. Gzip sizes are from the live files.

| Feed | Records | Observed seconds | Live gzip GB / 30 d | Lexical Zstd19 GB / 30 d |
| --- | ---: | ---: | ---: | ---: |
| Binance spot tape | 26,567 | 103.766 | 27.71 | 8.93 |
| Binance futures BBO and depth | 52,112 | 100.909 | 66.11 | 18.02 |
| Binance futures trades and mark/funding | 2,161 | 100.652 | 3.91 | 0.65 |
| Full exchange tape subtotal | 80,840 | | 97.73 | 27.60 |
| Bitcoin mainnet blocks | | | 4.89 | 4.45 using plain Zstd3 |
| REST seeds, five-minute checkpoints, OI and health | | | ~1.00 | ~1.00 allowance |
| **Core total** | | | **~104** | **~33** |

Eight Bitcoin blocks, sampled every 18 heights from 974548 to 974674
over 20.66 hours of block timestamps, give 4.89 GB/month in the live gzip
format and 4.45 GB/month with per-block Zstd3 at 4,320 blocks per 30 days.
The earlier single-block sample projected 6.59 GB/month with gzip. The
~1 GB exchange allowance uses measured spot and futures seed sizes,
five-minute checkpoints at twice seed size, two seeds per route per day,
plus the earlier OI and health allowances. [Prior capacity model](collector-storage-capacity-2026-11-01.md), [Bitcoin block capture](../collector/blockchain-stages.md#storage-evidence)

Plain Zstd3 alone projects 92.35 GB/month for the full exchange tape in
this sample. The byte-exact lexical codec encodes repeated structure
and numeric deltas before Zstd. The earlier BTC smoke projected 17.11
GB/month for the same three exchange shards with that prototype. Both
exchange samples are short and the difference shows sensitivity to
activity. The live recorder uses gzip. [Earlier codec benchmark](collector-lossless-codec-2026-11-02.md)

## What fits

| Profile | Sampled exchange rate | 2× exchange rate |
| --- | ---: | ---: |
| Core with experimental lossless packing | ~33 GB | ~61 GB |
| Core plus current supplemental Binance files and 1s summary, packed | ~47 GB | ~88 GB |

The supplemental files added 13.70 GB/month with the prototype, mainly
the separate futures BBO and spot depth subscriptions, which overlap
the main tape. This row also includes the collector's all-symbol
liquidation snapshots. A BTC-only selection of that published feed
would add less than the measured 0.07 GB/month packed for the full
liquidation stream. The estimates keep the Bitcoin and reconstruction
allowances fixed when stressing exchange event volume. With a physical
50 GB disk and a 10 GB work-space reserve, the archive allowance is
40 GB.

Coinbase, Kraken, Deribit, full pending Bitcoin transaction observations,
and other BTC trading pairs are outside the core estimate. The existing
once-a-minute mempool summaries were previously estimated at roughly
0.15 GB/month gzip. The earlier Coinbase, Kraken and Deribit peripheral
feed estimate was roughly 16 GB/month in gzip, separate from the
Binance tape. [Peripheral inputs](collector-one-tb-2026-11-02.md#where-the-modeled-bytes-go)

The live sample and exact compression measurements are stored in
`data/runtime-cache/btc-storage-50gb-2026-11-05`, including
`compression-sample.json`. The benchmark uses the existing
`scripts/lib/lossless-jsonl-block.ts` codec and compares the recovered
bytes with the original. The sample establishes feasibility at its
observed rate. A busy month can exceed the cap, and the lossless
prototype still needs production archival integration.
