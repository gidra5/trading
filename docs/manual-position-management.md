# Manual Position Management

This document describes the UI workflow for recording manual paper fills. It is separate
from the trading model because these controls are operator actions, not automated
strategy behavior. The underlying account and lot model is documented in
[Trading Model and Position Ledger](position-ledger.md).

Manual fills are stored as normal filled paper orders and trade fills with
`manual: true`. They persist with the bot state and are replayed by the same ledger
allocator as automated fills.

## Ledger Controls

The break-even ledger panel exposes three manual actions:

- `Long` records a manual buy fill with `positionEffect = "open"` and creates a new
  long lot.
- `Short` records a manual sell fill with `positionEffect = "open"` and creates a new
  short lot.
- row `Close` records the opposite-side fill against that specific lot id with
  `targetPositionId` and `positionEffect = "close"`.

The `Long` and `Short` controls intentionally force open new lots. They do not auto-net
against opposite-side lots. Targeted row closes are the UI path for reducing a specific
existing lot.

## Manual Fill Form

For new `Long` and `Short` fills, the form asks for quote amount and price. Quantity is
derived from:

```text
quantity = quote amount / selected price
```

For row closes, the form starts with the lot's remaining quantity. Quantity is editable,
so the same close action supports partial closes.

The price mode can be:

- `Current`, using the latest bot price
- `Limit`, using a manually entered price for the recorded fill

These are paper fills. Recording a limit price here does not create a resting exchange
order; it records a fill at that selected price.

## Resulting Model Effects

Manual `Long`:

- creates a filled buy order
- appends a buy fill
- opens a long lot because `positionEffect = "open"`

Manual `Short`:

- creates a filled sell order
- appends a sell fill
- opens a short lot because `positionEffect = "open"`

Manual close of a long:

- creates a filled sell order
- appends a sell fill with the target long lot id
- closes all or part of that long lot

Manual close of a short:

- creates a filled buy order
- appends a buy fill with the target short lot id
- closes all or part of that short lot

All manual fills still run through normal balance updates, fee accounting, realized PnL
updates, and the leverage guard.
