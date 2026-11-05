# Blockchain economic activity: staged coverage

Blockchain activity is a separate source family alongside exchange activity.
Transfers, contract calls, DEX swaps/liquidity changes, issuance/burns, staking
and bridge operations can matter without assigning real-world identities.
Transfers are not automatically buys/sells, exchange deposits, or changes in
beneficial ownership. The collector preserves observations before those labels.

## Confirmed and finalized feeds

`--blockchain-config FILE` enables the confirmed-block journal in the existing
collector. It is independent of `--binance-only` and can run without exchange
sockets using `--rest-only --fsync true`.

- **Bitcoin / Esplora:** network genesis verification, block identity and parent
  links, complete raw block bytes, raw-byte SHA-256 and independently computed
  header hash. Transactions, witnesses and scripts remain lossless in the raw
  block. This is not independent full consensus/merkle verification or a UTXO
  snapshot. Values of inputs spending pre-capture outputs require earlier
  blocks/prevout backfill.
- **EVM-compatible networks:** expected chain-ID verification, complete block
  transaction objects and every transaction's raw receipt/log response. Block
  mode requests all receipts by exact block hash, verifies one receipt for
  every transaction and checks log identities. It fetches up to four blocks
  in parallel and journals the successful prefix in height order.
  Per-transaction mode remains available for endpoints without bulk receipts.
- **Solana:** finalized blocks with serialized legacy, v0 and v1 transactions,
  block rewards and the metadata published by the node, including inner
  instructions and balance changes where available. The collector reads up
  to eight blocks at once. It emits `skipped_slots` only after a block's
  parent slot and hash link to the preceding captured block or initial
  ancestry anchor. Missing/pruned blocks can stall the cursor.
- **XRP Ledger:** validated binary ledger headers, serialized transactions and
  execution metadata. The adapter checks the configured network ID, recomputes
  the ledger header hash and individual transaction IDs, and stores the
  original response. Transaction and metadata inclusion relies on the RPC
  provider's ledger transaction set.
- **TRON:** solidified transaction bodies and matching solidified execution
  receipts, including log and internal transaction fields when returned by
  the node. The adapter verifies the network's genesis ID, hashes transaction
  bodies and checks the receipt set against the block's transaction IDs.
- **Common journal:** block order and parent checks, wall-clock observation
  times, durable cursor after sink flush, restart backfill, ancestry anchors,
  explicit reorg detach records, and status-file chain cursor/errors. Orphaned
  observations are retained. Polling does not reconstruct unobserved pending
  events or forks that appeared and vanished between observations.

The journal retains 128 ancestry identities. A reorg outside that retained
history stops cursor advancement and records an error requiring an explicit
rescan; it never silently deletes/relabels historical data. Crash recovery may
replay an already durable block; use `(chain, block hash)` for idempotency.
`confirmationLag` counts height units behind the observed head, with slots
as the unit on Solana. Solana, XRP Ledger and TRON use the provider's finalized,
validated and solidified views respectively. Bitcoin and EVM use an observed
head with a configurable confirmation lag. Omitting `startHeight` persists
the first observed target before attempting its payload, so retries cannot
move the start forward. Set `startHeight` for backfill.

Example: bounded Bitcoin-only capture from the repository root:

```powershell
npm run collect:external-live -- --rest-only --fsync true --binance-only --blockchain-config docs/collector/blockchain-bitcoin.json --duration-seconds 30 --output-dir data/research/blockchain-capture
```

To include it in exchange recording, use the same `--blockchain-config` with
`--quality full --universe docs/portfolio/restored-market-universe.json`.
`blockchain-evm.example.json` configures Ethereum, BNB Smart Chain, Base,
Arbitrum One, Polygon PoS, Avalanche C-Chain and OP Mainnet with environment
variables for each RPC URL. `blockchain-native.example.json` uses
`SOLANA_RPC_URL`, `XRPL_RPC_URL` and `TRON_RPC_URL`. TRON requires the
Solidity HTTP API paths, such as `/walletsolidity/getblockbynum`. For
authenticated TronGrid, add `"tronApiKeyEnv": "TRONGRID_API_KEY"` to that chain
and set the environment variable. Copy and trim either example to collect a
subset, or combine the chains in one config. Use a fresh output directory
when changing a chain's cursor configuration. The inventory is descriptive,
and capture requires an explicit config.

Endpoint capacity determines catch-up feasibility. The collector honors
HTTP Retry-After and records failures while retaining its durable cursor.
See the measurements below before choosing providers for unattended use.

## Inventory and following stages

`blockchain-coverage.json` covers all **212 crypto economic identities** in the
current Binance research universe. The November 5 snapshot resolves all 212
catalog identities through existing canonical IDs, exact Binance physical
routes or venue names, and reviewed listings. It records 62 provider-reported
deployment platforms and 28 documented native network surfaces. The deduplicated
union is 81 surfaces, including indexing overlays. Of those, 46 have an
unambiguous non-testnet EIP-155 registry match. Registry IDs can collide, so
the inventory flags ambiguous registry evidence separately. Reported contract
addresses still require deployment verification.

Rebuild/replay the inventory:

```powershell
node --import tsx scripts/build-blockchain-coverage.ts
node --import tsx scripts/build-blockchain-coverage.ts --offline
```

Subsequent stages are explicit:

1. Verify native and bridged asset contracts and expand the tested EVM
   configuration. Add execution traces/state changes for internal native
   transfers beyond ordinary receipt/log coverage.
2. Add represented native families such as Cosmos/Substrate, Move-based
   chains, TON, Stellar, Nano and other UTXO networks. Cover distinct network
   surfaces independently, such as HyperCore and HyperEVM.
3. Add observer-specific pending transaction streams, replacements/drops and
   first-seen timestamps. Historical confirmed blocks cannot recreate what a
   particular observer knew before inclusion. Protocol decoders for swaps,
   liquidity, staking and bridges must remain reproducible views over raw data.

Native-chain public observability varies; storing all public data does not
create private/shielded transaction values or participant identities. Coverage
must be evaluated per chain and event family, not counted from asset symbols.

## Storage evidence

One live Bitcoin block at height 974267 was captured and header-checked on
November 2: **1,760,375 raw bytes**, **2,347,728 JSONL bytes** including base64,
and **1,526,386 gzip bytes**. Repeating that one block's size at an expected
4,320 blocks per 30 days gives about **6.59 GB/month gzip**. This is a single
block illustration, not a measured month or upper bound. Bitcoin's block-weight
limit bounds an individual block, not the number of blocks arriving in a month.

The seven-network EVM and three native-network studies use short samples. See
the [EVM results](../experiments/blockchain-evm-storage-2026-11-05.md) and the
[native-chain results](../experiments/blockchain-native-storage-2026-11-05.md)
for source spans, compression and capture-rate limits. Solana's full finalized
blocks alone projected to about **2.81 TB per 30 days with Zstd19** in the
sample. The separate exchange compression experiment projected **1.17 TB**
at base activity and **3.50 TB** at 3x activity. The evidence does not support
a 1 TB combined archive while preserving all captured events.

## Primary references

- [Esplora raw block and transaction API](https://github.com/Blockstream/esplora/blob/master/API.md)
- [Ethereum JSON-RPC methods](https://ethereum.org/developers/docs/apis/json-rpc/)
- [Ethereum execution API for block receipts](https://ethereum.github.io/execution-apis/api/methods/eth_getBlockReceipts/)
- [Solana block response structures](https://solana.com/docs/rpc/json-structures)
- [XRP Ledger method](https://xrpl.org/docs/references/http-websocket-apis/public-api-methods/ledger-methods/ledger)
- [TRON solidification and receipt semantics](https://developers.tron.network/docs/confirmation-semantics)
- [Block-hash-specific log queries and reorg reasoning](https://eips.ethereum.org/EIPS/eip-234)
- [Bitcoin block weight](https://bitcoin.org/bip/141/)
- [CoinGecko platform metadata](https://api.coingecko.com/api/v3/asset_platforms)
