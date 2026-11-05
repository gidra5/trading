# Native chain capture and lossless storage samples

The November 5 samples use mainnet data and decimal storage units. Gzip
and offline Zstd were checked against the exact source JSONL bytes. The
30-day figures use short source windows and measure finalized or validated
payloads at sampled activity. They exclude pending observations, backups
and working storage.

| Chain and sample | Transactions | JSONL | Gzip | Zstd | Gzip / 30 d | Zstd / 30 d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Solana, 6 spaced blocks | 2,005 | 11.272 MB | 2.746 MB | 1.386 MB (19) | 5.57 TB | 2.81 TB |
| Solana, 177 consecutive blocks | 67,346 | 399.210 MB | 95.110 MB | 51.668 MB (3) | 6.45 TB | 3.50 TB |
| XRP Ledger, 3 consecutive ledgers | 879 | 1.487 MB | 0.424 MB | 0.318 MB (19) | 81.3 GB | 61.1 GB |
| XRP Ledger, second run, 2 consecutive ledgers | 727 | 1.205 MB | 0.348 MB | 0.276 MB (19) | 56.4 GB | 44.7 GB |
| XRP Ledger, alternate provider, 5 consecutive ledgers | 1,326 | 2.307 MB | 0.630 MB | 0.406 MB (19) | 68.7 GB | 44.3 GB |
| TRON, 6 consecutive blocks | 3,647 | 5.170 MB | 0.939 MB | 0.598 MB (19) | 135.3 GB | 86.1 GB |
| TRON, integrated 6 consecutive blocks | 3,450 | 4.978 MB | 0.894 MB | 0.570 MB (19) | 128.8 GB | 82.1 GB |

## Sources and projections

- **Solana.** Six evenly spaced finalized blocks between slots 465729301
  and 465731701 came from the official mainnet RPC. The source interval was
  509 seconds and the slot listing found 2,389 produced blocks in the
  sampled window. The projection scales mean compressed bytes per block
  by that production rate, about 12.17 million blocks per 30 days. The
  sample included 957 v1 transactions. The integrated 32-second run
  verified four consecutive produced blocks, then recorded HTTP 429 while
  the cursor stayed at the last durable block. A 40-second integrated run
  on PublicNode captured slots 465733731–465733907 over 38 seconds of block
  time, finishing four slots behind the last observed target with no
  coverage errors. [Solana public RPC limits](https://solana.com/docs/references/clusters), [PublicNode Solana RPC](https://solana.publicnode.com/)
- **XRP Ledger.** The original integrated samples used the XRP Ledger
  Foundation community server. They cover sequences 108132925–108132927
  and 108132989–108132990 across nine and eight seconds respectively.
  The estimates scale mean bytes per ledger by the observed mean close
  interval. The provider did not return a validated response for the
  next requested sequence in either initial run, so the cursor stopped
  at the captured prefix. The example config now reads two ledgers behind
  the provider's validated head. With that lag and Ripple's full-history
  public endpoint, the 40-second integrated run captured 108133021–108133025
  over 19 seconds of close time and reached its observed target without
  coverage errors. [XRPL public server directory](https://xrpl.org/docs/tutorials/public-servers)
- **TRON.** PublicNode's Solidity HTTP endpoint provided six consecutive
  solidified blocks, 87738006–87738011, and all matching receipts. The
  average block interval was three seconds. The receipts included 727
  with event logs. A separate paced TronGrid sample captured block
  87737894 with 694 matching receipts, including three receipts with
  provider-reported internal transactions. Its one-block projection was
  152.5 GB gzip or 105.0 GB Zstd19 for 30 days. Anonymous TronGrid returned
  HTTP 429 in the integrated batch, which retained its starting boundary
  and captured zero blocks from that endpoint. The integrated PublicNode
  run archived 87738031–87738036, reached its last observed target, and
  recorded no coverage errors. [PublicNode Solidity RPC](https://tron-solidity.publicnode.com/), [TronGrid rate limits](https://developers.tron.network/reference/rate-limits)

Adding the 2.81 TB Solana illustration, the integrated XRP Ledger and TRON
Zstd19 results, the seven-chain EVM Zstd19 projection, the Bitcoin gzip
sample, and the base exchange estimate gives roughly **4.6 TB per 30 days**
for this subset. The subtotal combines different sample windows and archive
formats. The live collector writes gzip. Zstd here measures offline savings,
with production throughput and month-scale activity still to be verified.
[EVM study](blockchain-evm-storage-2026-11-05.md), [exchange study](collector-one-tb-2026-11-02.md), [Bitcoin sample](../collector/blockchain-stages.md#storage-evidence)
