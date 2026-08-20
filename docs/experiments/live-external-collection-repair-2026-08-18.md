# Live external collection repair — 2026-08-18

## Outcome

The continuous point-in-time collector is running again under a persistent supervisor. The earlier archive was not a 24-hour sample: it contains **9.598 hours of observed BTC trade coverage** from 2026-08-17 09:44–19:21 UTC. A watcher had used wall-clock time since collector start, so downtime was incorrectly counted as collected evidence. Coverage and checkpoints now advance only from observed target seconds.

The legacy 287.63 MiB archive is preserved. New normal operation no longer stores every high-frequency book update. It writes one causal `cross-exchange-book-1s` record per second; `--raw-books` is available only for short reconstruction diagnostics.

The first 110 seconds of the final supervised session imply a startup-biased upper rate of about 136 MiB/day, versus roughly 676 MiB/day for the old raw-book policy. The estimate should settle after the immediate options/GDELT/mempool startup pulls are amortized; the 1-hour checkpoint will provide the first useful steady-state storage rate.

## Repairs

- Migrated Binance USD-M sockets to the current `/market/ws` and `/public/ws` paths for liquidation, mark price, and book ticker streams.
- Added per-stream stale-message reconnects for streams that should be active continuously. The sparse liquidation stream is not declared dead merely because no liquidation occurs.
- Added an atomic collector heartbeat with per-source message counts, last-message ages, opens, and closes.
- Added a supervisor that restarts the collector after process exit or a stale heartbeat. An intentional child-process termination was recovered automatically: restart count became 1 and collection resumed under a new PID.
- Changed the checkpoint watcher from elapsed wall time to accumulated observed coverage.
- Invalidated return windows that cross a market-data outage, instead of forward-filling a price across the outage.
- Prevented low-effective-sample long-horizon scores from appearing in the best-score ranking; fewer than 16 effective held-out outcomes is always `inconclusive`.

## Live validation

A 25-second isolated smoke run produced:

| stream | messages |
|---|---:|
| Binance spot depth diff | 244 |
| Binance spot aggregate trades | 27 |
| Binance USD-M mark price | 24 |
| Binance USD-M book ticker | 541 |
| Coinbase level 2 | 422 |
| Kraken level 2 | 805 |
| Deribit combined book/trade socket | 77 |

It produced 24 one-second compact book rows. Binance futures and Deribit were valid in 24/24; Binance spot, Coinbase, and Kraken were valid in 23/24 because their initial snapshots completed after the first timer tick. A subsequent supervised run also observed a real Binance liquidation event, confirming the migrated liquidation path.

## Compact 1-second book record

For Binance spot, Coinbase spot, Kraken spot, Deribit perpetual, and Binance perpetual the row contains:

- receive-time age and validity;
- best bid/ask and quantities, midpoint, spread in basis points;
- level-1 and top-five quantity imbalance (top-five equals level-1 for the Binance futures BBO stream);
- cross-spot midpoint dispersion, best executable cross-venue spread, and Binance perpetual basis;
- Binance spot bid/ask quote depth added and removed during the second.

`removed` deliberately includes both cancellations and trade depletion. Binance level-2 depth updates do not identify which mechanism removed displayed quantity, so treating it as exact cancellation volume would manufacture information not present in the source.

Binance spot snapshots and diff-update IDs are sequence checked. A sequence gap invalidates the book and triggers a fresh REST snapshot before the venue becomes valid again.

## What the first archive supports

The 9.598-hour sample is sufficient to validate feed shape, timestamp alignment, and catch only unusually large effects at 1–15 seconds. It is not enough to reject features or support a 1-hour forecast conclusion. The old apparent 1-hour gains have only about three effective held-out outcomes and are therefore inconclusive.

In the original uninterrupted sample, Deribit total call/put open-interest imbalance at the 1-second target measured about +0.035 bits/target and was positive in 5/6 chronological blocks. After collection resumed, adding only a small new tail across the outage changed the single-split ranking and pushed funding/premium bins to the top. That degree of ranking sensitivity is itself evidence that none of these is a reliable feature conclusion yet; the current table is a pipeline smoke test, not a selected basis. GDELT produced only 39 observations in the original sample and could not be scored.

Practical evidence targets remain:

| target/use | minimum interpretation window |
|---|---|
| feed health and very-large 1–15s effects | 1 hour |
| early 1s–1m screen | 1 day |
| useful separated-day evidence for minute-cadence options/mempool | 3–14 days depending on target |
| GDELT and 15m targets | 14–30 days |
| credible 1h target validation | 30–90 days |

## Operations

```text
npm run collect:external-live:start
npm run collect:external-live:status
npm run collect:external-live:stop
npm run analysis:external-live-early
```

Runtime state and logs are under `data/runtime-cache/external-live-collector`. Collected observations remain under `data/market/mutable/external-live`. `TRADING_ECONOMICS_API_KEY` and `BLOCKWORKS_API_KEY` are still absent, so point-in-time macro-surprise consensus and ETF-flow feeds remain disabled rather than being represented by invalid substitute data.
