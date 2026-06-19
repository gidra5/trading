# Position Ledger

The position ledger mirrors the `sol` spreadsheet section from rows 111 onward. It is implemented in `packages/bot-algo/src/position-ledger.ts` and is derived from the paper bot fills plus currently open limit orders.

## Lot Model

Each filled buy that is not consumed by earlier shorts becomes a long lot. Each filled sell that is not consumed by earlier longs becomes a short lot. Closing trades are allocated FIFO.

Open buy limit orders are shown as pending long lots. Open sell limit orders are shown as pending short lots. Pending lots use the limit price and estimated fee-adjusted quote amount so the UI can show the break-even levels that would apply if the order fills.

Every lot also keeps explicit closed quantity and closed quote. For long lots that quote value is allocated sell proceeds. For short lots it is allocated buyback cost.

## Spreadsheet Formulas

Long lots use the spreadsheet recovery model:

```text
average buy price = cost quote / bought quantity
remaining cost = original cost - allocated sell proceeds
break-even sell price = remaining cost / max(remaining quantity, quantity floor) * (1 + fee + slippage)
max-loss sell price = max(0, remaining cost - max loss pct * original cost) / max(remaining quantity, quantity floor) * (1 + fee + slippage)
sell-now quote = max(0, (remaining cost - lower baseline * remaining quantity) / (net market sell - lower baseline) * net market sell)
can reach lower baseline = sell-now quote > 0 and sell-now quote < remaining cost and quantity is available
```

Short lots use the mirrored sell-side formulas:

```text
average sell price = proceeds quote / sold quantity
remaining proceeds = original proceeds - allocated buyback cost
break-even buy price = remaining proceeds / max(remaining quantity, quantity floor) / (1 + fee + slippage)^2
max-loss buy price = remaining proceeds / max(remaining quantity - max loss pct * original quantity, quantity floor) / (1 + fee + slippage)^2
buy-now quote = max(0, (remaining proceeds - upper baseline * remaining quantity) / (gross market buy - upper baseline) * gross market buy)
can reach upper baseline = buy-now quote > 0 and buy-now quote < remaining proceeds and quantity is available
```

The default quantity floor is `0.01`, matching the spreadsheet guard against divide-by-zero rows. Fee is the bot configured Binance fee. Slippage is a separate market-order estimate exposed in the UI.

## Manual Table Actions

The ledger table can record manual paper fills:

- `Long` records a manual buy fill and creates a long lot unless it closes an existing short.
- `Short` records a manual sell fill and creates a short lot unless it closes an existing long.
- row `Close` records the opposite-side fill against that specific lot id. Quantity is editable, so the same action supports partial closes.

Manual fills are stored as normal filled paper orders and trade fills with `manual: true`; they persist with the rest of the bot state and are replayed by the ledger allocator.
