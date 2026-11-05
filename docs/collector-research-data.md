# Research capture and economic coverage

The collector can record published Binance spot and USD-M perpetual market
messages for the physical routes in `docs/portfolio/restored-market-universe.json`.
The manifest preserves historical research membership and adds current economic
exposures. It is a data collection universe, not investment weights or a claim
to span every part of the global economy.

The [blockchain runbook](collector/blockchain-stages.md) covers the Bitcoin,
EVM, Solana, XRP Ledger and TRON adapters, along with the asset inventory,
capture boundaries and measured storage. The earlier mempool summaries
remain a separate observation stream.

The November 1 snapshot contains 460 economic identities and 687 physical
routes (261 spot, 426 USD-M). The 248 noncrypto identities include 229 equity
or fund exposures, eight commodity underlyings and eleven FX exposures. See
`docs/portfolio/restored-market-universe-sectors.md` for sectors, proxy types
and gaps. All currently identified noncrypto instruments are retained; this is
a broad coverage inventory rather than a new fitted minimal return basis.

## What is observable

| Research requirement | Capture support | Boundary |
| --- | --- | --- |
| Price-level liquidity state | Initial REST depth, published incremental L2 updates, sequence checks and local checkpoints | Initially at most 5,000 levels per side for spot and 1,000 for USD-M; unchanged deeper levels remain unknown |
| Order flow | Spot individual trades; USD-M aggregate trade messages; net price-level quantity changes | Net reductions do not uniquely identify cancellations versus executions; public L2 is not an individual-order/L3 lifecycle |
| Spot/futures prices | BBO updates, reconstructed depth BBO/microprice, exchange and receive timestamps, book age at a trade | Spot `bookTicker` has no exchange timestamp; depth-derived prices have the depth publication cadence, not matching-engine precision |
| Observation continuity | Connection/session records, depth epochs, sequence gaps, bootstrap validity and coverage status | No assurance of completeness across disconnects; trade-ID continuity does not prove that every exchange action is public |
| Derivative state | Mark/index/funding stream and periodic open-interest requests across selected USD-M routes | Aggregation and publication cadence are those of the provider |
| Participant observations | Optional Hyperliquid public trades with buyer/seller addresses; explicitly watched accounts' orders, fills and account state | Binance public feeds expose no participant identities or other accounts' positions; an address does not establish an economic entity or strategy |
| Temporal/regime diversity | Persistent append-only capture across the selected routes and successive sessions | Diversity of market regimes requires elapsed observation time and coverage audits; code cannot create it immediately |

Public Binance market feeds and private account feeds have different access
boundaries. A private key would provide observations for its own account, not
the market's entire participant population. The research record must not label
net L2 removals as exact cancellations or latent cohorts as identified traders.

## Full capture

From the repository root, record a bounded two-market check:

```powershell
npm run collect:external-live -- --quality full --universe docs/portfolio/restored-market-universe.json --symbols BTCUSDT --venues spot,usdm-futures --binance-only --duration-seconds 45 --output-dir data/research/collector-smoke
```

For the complete manifest, omit the symbol filter and duration. Supplemental
BTC exchange books, options, mempool and news remain available when
`--binance-only` is omitted. Supplying a universe requires full capture;
the existing compact mode is the BTC supplemental collection, not a smaller
multi-asset event archive.

Full mode retains each published public Binance message without local event
downsampling, its original wire JSON, exchange times where supplied, receive
wall and monotonic times, and ordering/session metadata. Monetary source
strings are preserved. Local book and alignment rows are derived observations;
their timestamps and validity must be retained when building a model dataset.
High-frequency derived BBO and trade/book alignment rows are optional through
`--derived-rows true`; the default full capture omits these duplicates. The
raw tape and checkpoints retain the inputs needed to reconstruct them offline.

Full mode uses gzip and synchronizes flushed chunks by default. The sink is
bounded and a write/backlog failure stops collection instead of silently
dropping records. Complete gzip members are flushed approximately every second
or at the size threshold. A process or host failure can still interrupt the
last member; recovery must exclude incomplete tails and audit the next session
boundary. Local fsync does not provide a remote backup or guarantee an entire
month of uninterrupted observation.

Books begin invalid and become usable only after a snapshot bridges buffered
updates. Bootstrap starts are paced to fit REST request weights; the full
universe is not observed simultaneously from the instant the command starts.
Connection liveness and absence of trades are separate concepts, particularly
for quiet instruments and closed underlying markets.

To run this configuration under the existing restart supervisor:

```powershell
npm run collect:external-live:start -- --quality full --universe docs/portfolio/restored-market-universe.json
npm run collect:external-live:status
npm run collect:external-live:stop
```

The manager saves the supplied arguments for subsequent restarts. Confirm
storage capacity from the capacity report before starting the broad universe;
the implementation work and bounded smoke checks do not start this indefinite
recording job.

## Optional participant pilot

```powershell
npm run collect:external-live -- --quality full --symbols BTCUSDT --binance-only --hyperliquid --hyperliquid-coins BTC --duration-seconds 45 --output-dir data/research/participant-smoke
```

`--hyperliquid-users` accepts an explicit comma-separated list of at most ten
account addresses. Without that list, only public market observations are
requested. The selected-account channels include fills, order updates, open
orders, perpetual account state, spot state and non-funding ledger updates.
This limit also applies across other user subscriptions sharing the same IP.

Hyperliquid's public L2 feed contains bounded snapshots, not every individual
order action. Initial messages can replay earlier trades or fills; retain
initial-snapshot flags and session boundaries and deduplicate provider trade
identifiers when constructing a dataset. Missing events across reconnects must
not be filled with assumed inactivity. Account state for ten addresses is a
pilot, not a census of all participants.

## Capacity and reproducibility

See `docs/experiments/collector-storage-capacity-2026-11-01.md` for the measured
30-day estimate, sample periods, compression settings, route counts and stress
allowances. Compression ratios depend strongly on batching and message rate;
a short Sunday sample is insufficient to certify a trading-month capacity.
Continuous capture also needs CPU, network, disk throughput and free-space
monitoring, independently of the final compressed archive size.

The universe's saved provider inputs allow offline membership replay. Keep the
manifest used for each capture with its session metadata; refreshing current
listings must not retrospectively change which markets an older session meant.

## Primary protocol references

- [Binance spot market streams](https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams)
- [Binance endpoint security and account-data distinction](https://developers.binance.com/en/docs/products/spot/rest-api)
- [Hyperliquid subscriptions and data schemas](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket/subscriptions)
- [Hyperliquid IP and user subscription limits](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/rate-limits-and-user-limits)
