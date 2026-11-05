# Seven EVM chains, live receipt capture and storage

On November 5, 2026, the collector ran for 70 seconds with gzip level 6,
durable cursor writes, and the seven-network
[EVM example](../collector/blockchain-evm.example.json). It recorded 475 blocks,
29,512 transactions and matching receipts, and 155,919 logs. For each chain,
the archive was contiguous from its recorded ancestry anchor and the final
saved cursor matched the last archived block. No coverage errors were recorded.
This initial run read one block at a time and exited cleanly.

The block and receipt responses totalled **193.75 MB** before JSONL framing,
**204.71 MB** as block JSONL, and **23.14 MB** on disk including chain journal
metadata. MB and GB are decimal throughout.

| Network | First–last height | Blocks | Transactions and receipts | Logs | Source span | Block gzip MB | Projected GB / 30 days |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ethereum | 26349087–26349092 | 6 | 1,961 | 5,121 | 60 s | 1.134 | 40.8 |
| BNB Smart Chain | 131829271–131829398 | 128 | 13,925 | 83,816 | 57 s | 11.140 | 502.6 |
| Base | 53548930–53548964 | 35 | 8,661 | 33,235 | 68 s | 6.108 | 226.2 |
| Arbitrum One | 521495311–521495470 | 160 | 1,116 | 3,252 | 42 s | 0.876 | 53.7 |
| Polygon PoS | 96787786–96787831 | 46 | 2,768 | 27,275 | 68 s | 3.117 | 116.2 |
| Avalanche C-Chain | 99499484–99499550 | 67 | 293 | 889 | 58 s | 0.283 | 12.5 |
| OP Mainnet | 159144215–159144247 | 33 | 788 | 2,331 | 64 s | 0.479 | 18.8 |
| **Total** | | **475** | **29,512** | **155,919** | | **23.138** | **970.9** |

The projection multiplies each network's mean gzip bytes per block by
2,592,000 seconds divided by its observed mean time between consecutive
blocks. Source span uses the first and last block timestamps so the collector's
backlog does not lower the estimated chain data rate. The block gzip column
uses the journal's gzip level and one member per block, excluding the small
session and ancestry records. Each network contributed 42–68 seconds of
source time. The result is a short-window illustration with substantial
activity uncertainty. It excludes traces, pending observations, other chains,
and working storage. The seven EVM archives alone project to about **0.97 TB**
at this activity. The earlier [exchange-only projection](collector-one-tb-2026-11-02.md)
adds about 1.17 TB at its base rate.

At shutdown the Arbitrum, BSC and Polygon cursors trailed their last persisted
observed targets by 103, 25 and 1 blocks, respectively. The run establishes
that those archived ranges are complete. It does not establish that polling
these endpoints can sustain the live arrival rate or recover from long
outages. The other four cursors reached their last persisted targets during
this run. Longer runs should track head lag, outage catch-up and hourly size
variation.

## Lossless Zstandard replay

Recompressed the original 475 block JSONL records as raw bytes and verified
every decompressed byte against the gzip journal. No parsing or event
selection occurs in the compression step. The per-block result keeps the
existing one-frame-per-block boundary. The per-source result packs each
chain's complete sample into a single frame as an offline archive.

| Network | Gzip GB / 30 days | Zstd 3 per block | Zstd 3 per source | Zstd 19 per block | Zstd 19 per source |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ethereum | 40.8 | 31.7 | 31.3 | 26.8 | 24.5 |
| BNB Smart Chain | 502.6 | 392.9 | 352.9 | 333.3 | 250.2 |
| Base | 226.2 | 166.7 | 158.7 | 135.0 | 110.2 |
| Arbitrum One | 53.7 | 49.8 | 34.6 | 44.9 | 25.4 |
| Polygon PoS | 116.2 | 91.5 | 85.8 | 75.9 | 61.1 |
| Avalanche C-Chain | 12.5 | 12.1 | 7.6 | 10.9 | 5.9 |
| OP Mainnet | 18.8 | 16.0 | 11.8 | 13.7 | 9.0 |
| **Total** | **970.9** | **760.7** | **682.7** | **640.5** | **486.5** |

Zstd level 3 produced 18.04 MB with separate block frames and 16.31 MB with
one frame per source. Level 19 produced 15.14 MB and 11.62 MB, respectively.
The original gzip block records occupy 23.14 MB. Entire source files including
journal metadata also round-tripped byte for byte in both levels.
Each framing mode took about 0.4 seconds to encode at level 3 and 72–79
seconds at level 19 on this workstation. These are offline replay results
using the first source windows and the same 30-day scaling. The live journal
still writes gzip.

## Parallel read follow-up

A second 70-second run read up to four blocks concurrently per chain and
committed them in parent order. It exited cleanly with **408 contiguous
blocks**, **26,448 transactions and receipts**, **152,956 logs**, and
**20.80 MB gzip**. The same short-window calculation yields **882.5 GB
per 30 days**. The first run's 970.9 GB and this estimate reflect different
blocks and source windows.

Six networks reached their last observed targets, including BSC and Polygon.
Arbitrum One recorded three HTTP 429 responses from its official endpoint
and finished **150 blocks behind** its last observed target, with 74
contiguous blocks committed. At this endpoint, the four-way mode hit a rate
limit even though it reduced the lag on BSC and Polygon. The observed cursor
gap measures backfill still owed at shutdown.

## Receipt method and endpoint evidence

Block mode saves the original full block response plus the original
`eth_getBlockReceipts` response requested by exact block hash. The collector
requires one receipt for every transaction hash and checks each receipt and
log's block and transaction identity before advancing. For sampled heights
26349070 (Ethereum), 131828828 (BSC), 521494550 (Arbitrum) and 96787656
(Polygon), all 515 parsed receipts and their nested logs matched those fetched
individually from the same endpoints. The complete response envelopes are
preserved in each mode. Block mode reduces receipt requests from one per
transaction to one per block. [Ethereum Execution APIs](https://ethereum.github.io/execution-apis/api/methods/eth_getBlockReceipts/)

| Network | Chain ID | Environment variable | Tested RPC | Chain identity source |
| --- | ---: | --- | --- | --- |
| Ethereum | 1 | `ETHEREUM_RPC_URL` | `https://ethereum-rpc.publicnode.com` | [Geth](https://geth.ethereum.org/docs/getting-started) |
| BNB Smart Chain | 56 | `BSC_RPC_URL` | `https://bsc-dataseed.bnbchain.org` | [BNB Chain](https://docs.bnbchain.org/bnb-smart-chain/developers/json_rpc/json-rpc-endpoint/) |
| Base | 8453 | `BASE_RPC_URL` | `https://base-rpc.publicnode.com` | [Base](https://docs.base.org/base-chain/api-reference/ethereum-json-rpc-api/eth_chainId) |
| Arbitrum One | 42161 | `ARBITRUM_RPC_URL` | `https://arb1.arbitrum.io/rpc` | [Arbitrum](https://docs.arbitrum.io/chain-info) |
| Polygon PoS | 137 | `POLYGON_RPC_URL` | `https://polygon.drpc.org` | [Polygon](https://docs.polygon.technology/pos/reference/rpc-endpoints) |
| Avalanche C-Chain | 43114 | `AVALANCHE_C_RPC_URL` | `https://avalanche-c-chain-rpc.publicnode.com` | [Avalanche](https://docs.avax.network/docs/rpcs/c-chain) |
| OP Mainnet | 10 | `OPTIMISM_RPC_URL` | `https://optimism-rpc.publicnode.com` | [Optimism Superchain registry](https://github.com/ethereum-optimism/superchain-registry/blob/main/superchain/configs/mainnet/op.toml) |

The official Base public RPC returned HTTP 429 during the initial individual
receipt test and responded to the subsequent bulk probe. The official OP
Mainnet public RPC returned HTTP 403 for the bulk method and the official
Avalanche C-Chain public RPC returned method unavailable (`-32601`). The
integrated run used the endpoints in the table. Verify the method on the
specific provider used for deployment. [PublicNode RPC directory](https://www.publicnode.com/)

Set the environment variables for the networks included in the config. Copy
the example and remove entries to test a smaller set. Use a fresh output
directory for a bounded run.

```powershell
node --import tsx scripts/collect-live-external-features.ts --rest-only --fsync true --binance-only --blockchain-config docs/collector/blockchain-evm.example.json --duration-seconds 70 --output-dir data/research/blockchain-samples/evm-stage2/integrated
```
