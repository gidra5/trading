# Live data, research universe, and stochastic-model data gaps

Recovered on November 1, 2026 from the current source tree, local archives,
saved portfolio reports, and the regular ChatGPT conversations linked below.
This is a historical recovery record; earlier acquisition blockers are not
claims that a provider still has the same access restrictions today.

Follow-up implementation: `docs/collector-research-data.md` describes the
expanded collector and economic coverage. The restored universe was extended
after this initial audit; use its current manifest and sector report for
current membership, rather than the initial 223-asset restoration below.

## Existing collector

Implementation: `scripts/collect-live-external-features.ts`, with book state in
`scripts/lib/live-book-compactor.ts` and restart management in
`scripts/manage-live-external-collector.mjs` and
`scripts/supervise-live-external-collector.mjs`.

The collector is BTC-focused, rather than a collector for the representative
asset universe. Binance's all-symbol liquidation subscription is the exception.
It has no Binance perpetual trade subscription: the perpetual trade tape in
this collector is Deribit's. Binance futures trades used in the spot/futures
chat experiment came from a separately supplied dataset.

| Source | Recorded information | Cadence/storage |
| --- | --- | --- |
| Binance spot BTCUSDT | Aggregate trades, including price, quantity, aggressor side, aggregate IDs and exchange times | Every received event, raw |
| Binance spot BTCUSDT book | Incremental L2 updates, reconstructed from a 5,000-level snapshot | Ingests 100ms updates; normal output is a 1s summary |
| Coinbase BTC-USD | Reconstructed `level2_batch` book and heartbeat | 1s book summaries |
| Kraken BTC/USD | Reconstructed 100-level book | 1s book summaries |
| Deribit BTC perpetual | Book updates plus perpetual trades | Book summaries at 1s; trades raw |
| Binance BTCUSDT perpetual | Best bid/ask and quantities | 1s book summaries; no full perpetual depth |
| Binance USD-M liquidations | All-market `forceOrder` messages | Raw received events, subject to exchange feed censoring |
| Binance BTCUSDT mark price | Mark/index price and funding-related fields | Raw 1s stream |
| Binance BTCUSDT premium index | Mark/index/settlement and funding fields | REST every minute |
| Deribit BTC options | Option trades; derived ATM IV and 25-delta skew at 1d/7d/30d, term slopes, OI, expiry/strike state | Trades raw; summary every minute; full raw surface roughly every 15 minutes |
| Bitcoin mempool | Transaction count/vsize/fees, recommended fees, projected blocks, recent transactions | Every minute |
| GDELT | Crypto-news intensity, story/source counts, tone and article metadata | Every 15 minutes |
| Trading Economics, optional | US importance-2 macro calendar/consensus | Every minute, requires configured key |
| Blockworks, optional | Bitcoin ETF flow metric | Every hour, requires configured key |

The compact book rows retain bid/ask, quantities, mid, spread, L1/top-five
imbalance, validity/age, cross-spot dispersion, executable spread, perpetual
basis, and Binance spot quote-depth additions/removals. Removed depth is not a
separately identified cancellation: executions and cancellations can both
reduce a price level. The perpetual BBO does not supply true top-five depth.

Full raw book messages and Binance snapshots require `--raw-books`; the normal
managed collector does not enable it. Even raw L2 records are aggregated
price-level changes, not a participant-attributed order lifecycle/L3 tape.

Most output uses append-only daily gzip JSONL under
`data/market/mutable/external-live/<source>/YYYY-MM-DD.jsonl.gz`. Raw option
surfaces use separate gzip JSON snapshots. Buffers flush every second or
256 KiB. There is no automatic age-based pruning in these collector scripts.

Local audit on November 1:

- Managed status returned `{"running":false}`; the runtime status directory was
  absent. This does not establish the status of a remote or independently
  launched process.
- Archive: 1,392 files / 2,122,024,814 bytes, beginning August 17.
- Latest inspected spot-trade receipt: August 30, 2026 07:10:09.710 UTC.
- Latest compact-book receipt: August 30, 2026 07:10:08.850 UTC.
- No optional macro-calendar or ETF-flow archive directories were present.
- Calendar span is not continuous observed coverage; the archive has outages.

## Recovered asset sets

The original compact 365-day/daily set was **BTC, VELVET, BULLA, JELLYJELLY,
SIREN, JST, ALCH, TA, M, BR, OG, FHE**. It survives in
`docs/portfolio/runs/2026-07-22-all-usdt-pearson-k12.json`.

The later independent-scale studies selected 117 daily, 182 four-hour,
221 hourly, 234 fifteen-minute, and 257 one-minute assets, with a union of 437.
Their saved membership is documented in `docs/portfolio/scale-comparison/latest.md`.

The actual cross-asset research universe used the **257-asset one-minute basis
plus ETH, SOL, XRP and HYPE = 261 assets**. Its exact basis manifest is
`docs/portfolio/runs/2026-07-22-all-usdt-1m-360c-q4-amp5pct-pearson-k257.json`;
the additions are explicit in `scripts/export-binance-cross-asset-minute-axis.ts`
and `docs/experiments/binance-cross-asset-component-feature-bases-2026-08-20.md`.

Selection maximized residual standardized-return variance using pivoted QR,
anchored at BTC. Within 5% of the best residual, larger mean absolute returns
won the tie. There was no default liquidity floor. Thus the set represented
independent return directions, not capitalization share. A correlated large
market could be omitted even when it carried substantial economic weight.

`docs/portfolio/latest.json` is a different August 20 hourly 12-asset run with
only 80 eligible assets and weak market coverage. It is not the archived
cross-asset universe. The former selector commands also import server modules
currently deleted in the working tree; recovery must not silently undo that
ongoing server rewrite.

For restoration, the user's market-share definition is **global market
capitalization**. Preserve the historical 261 constituents separately from a
current tradable union, then add a capitalization leader core regardless of
correlation. This deliberately permits correlated leaders. Capitalization
coverage must name its denominator, distinguish stable/wrapped assets, and
report unknown or ambiguous provider mappings. Binance turnover is not a
substitute for global capitalization. This restoration is membership for data
research, not portfolio weights or a new fitted return basis.

The standalone restoration is now saved in
`docs/portfolio/restored-market-universe.json`, generated by
`scripts/restore-research-market-universe.ts`. It preserves all historical 261
in order, finds 220 with active USDT spot/perpetual routes, and adds BNB, ZEC
and DOGE for a current union of **223**. The cap core is **BTC, ETH, BNB, XRP,
SOL, TRX, HYPE, ZEC, DOGE, XMR**. It covers 95.4417% of capitalization among
265 uniquely mapped, non-excluded, Binance-eligible assets in a fetched
999-coin CoinGecko snapshot. This is not 95% of total global capitalization.
Stablecoin, wrapped-token, liquid-staking and liquid-restaking categories are
excluded from the cap core; historical diversity membership is preserved.
Provider inputs, identities, retrieval times and response hashes are saved in
`docs/portfolio/restored-market-universe-sources/` for offline replay.
This manifest does not reconfigure or start the existing BTC collector.

## Recovered regular ChatGPT discussions

The clearest missing-data discussion is
[Branch · Branch · Recall SDE Data Analysis](https://chatgpt.com/c/6ac2b5a1-d010-83ed-9a8b-433df287cefd),
October 5, 2026. The earlier
[Recall SDE Data Analysis](https://chatgpt.com/c/6abf6588-d5f0-83ed-83b9-8e5fc601b6ed),
October 3, covers synchronized spot/futures observations. Both conversations
were read during this recovery; this is not inferred solely from their titles.

### Specific observations absent from the modeling dataset

1. **LOB state and event types:** the model corpus contained aggregate trades
   `(time, price, quantity, aggressor side)`, without order submissions,
   cancellations, queue depth or the book state preceding each action. The chat
   proposed incremental depth plus reconstruction snapshots synchronized to
   trades; even BBO/spread/imbalance/queue sizes would permit an initial
   queue-reactive Hawkes test. Trade-only observations could not distinguish
   event self-excitation from persistent omitted liquidity state.
2. **Participant attribution and state:** persistent participant IDs, per-agent
   order/cancel/fill histories, inventory/positions, and observable strategy
   switching were missing. The proposed hierarchy was agent -> strategy/cohort
   -> event intensity -> LOB. Several latent decompositions could explain the
   same aggregate flow, so behavioral labels were underidentified. This is a
   missing observation channel, not something longer aggregate-trade history
   alone resolves.
3. **Synchronous spot/futures prices:** last-trade prices were asynchronous and
   could create artificial basis movements. The original chat wanted both
   markets' BBO/mid/microprice at event time. Its reduced joint experiment had
   only 9.77 hours of strict overlap, and 100ms bins obscured sub-bin ordering.
4. **Independent time periods and regimes:** older turns in the branch warned
   that long-lag ACF fluctuations across blocks exceeded 1%, and a mode much
   longer than a six-hour fitting window was effectively constant over that
   window. The archive's calendar length was not sufficient evidence for a
   precise physical long-memory law.

The branch considered blockchain exchange inflows/outflows and participant
cohorts as auxiliary observations, but acknowledged that chain transfers cannot
reveal individual internal Binance orders. It then explored Uniswap and
Hyperliquid as more observable research venues. Persistent addresses,
orders/cancellations/fills, positions, wallet state and a reconstructed book
were the desired dataset. This was a historical proposal, not a completed
data pipeline or a verification of current archive completeness. Even an
address-attributed venue leaves strategy, address-to-entity linkage, inaction
and off-chain information partly unobserved.

### Local corpus and broader acquisition gaps

`docs/experiments/hawkes-latent-resume-2026-10-03.md` records the actual input:
14,464,973 Binance spot BTCUSDT aggregate trades, August 17-30. It had 81
aggregate-ID discontinuities, 884,285 missing IDs, and 41 gaps longer than
60 seconds. The conservative gap mask excluded 132,377 seconds and could not
certify shorter outages. This trade-only export must not be confused with all
the richer sources present in the collector archive.

The separate August feature-acquisition ledger,
`docs/experiments/external-feature-basis-audit-2026-08-17.md`, records broader
historical blockers: continuous cross-venue L2/L3 books, full historical options
surfaces/OI, exact continuous liquidations, true historical unconfirmed-mempool
state, point-in-time macro consensus and labeled wallet/ETF/treasury flows,
credentialed intraday macro markets and signed Binance borrow-rate history.
Sparse vendor sample days and coarse proxies did not supply those observations.

The cross-asset study further states that additions/cancellations, liquidations,
exact aggregate-trade sequence/size, options, chain flows and non-Binance books
were not uniformly available over the representative universe. Broad candle,
funding, five-minute positioning and percentage-depth backfills existed; rich
event history with comparable coverage across assets and regimes did not.

The existing BTC collector therefore supplies some omitted state in compact
form, but does not yet satisfy the participant-attributed, synchronized,
multi-asset event dataset envisioned by the stochastic-model discussion.
