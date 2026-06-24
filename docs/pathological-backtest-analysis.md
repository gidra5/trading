# Pathological Backtest Analysis

This report explains the weak PnL and low oracle-capture cases saved in
[pathological-backtest-windows.json](pathological-backtest-windows.json). The cases were
sampled from BTCUSDT `1m` historical candles with the default `legacy-valley-peak`
strategy, `1x` max leverage, `futures-margin` shorts, `999/999` borrow depth,
`maxOpenOrders = 1024`, and `300s` cooldown.

Reproduce the saved windows with:

```bash
npm run backtest:pathologies -- --mode rerun
```

The follow-up analysis reran the same windows with all orders and fills retained, then
reconstructed trade side, realized PnL, final inventory, and approximate exposure over
time.

## Main Failure Modes

The weak cases are mostly not isolated simulator bugs. They expose the current strategy
shape:

- `legacy-valley-peak` is a mean-reversion and exit-grid strategy, not an oracle-like
  directional switcher.
- The bot is usually net long, including in bearish windows.
- Short entries are often much smaller than long entries.
- Several positive returns are partly or mostly open-inventory mark-to-market gains,
  not fully closed profit.
- The perfect-margin oracle is large in volatile windows because it earns from every
  replayed OHLC leg. Any idle time, late entry, under-sizing, or wrong-side inventory
  creates a large capture gap.

`Net PnL` in a backtest is mark-to-market equity. It includes realized PnL from closed
fills plus unrealized PnL on still-open lots at the final candle. A window can therefore
look profitable because it ends while the bot is holding favorable inventory. That
profit is not locked in unless exit fills have closed the position.

## Case Notes

| Case | Return | Oracle Capture | Primary Cause |
| --- | ---: | ---: | --- |
| `2022-03-22 1d` | `-1.00%` | `-13.556%` | False/late valley entry after the rally. The bot bought near `43.2k`, stayed net long for `95%` of the window, never became net short, and held open long inventory through the pullback. |
| `2022-08-28 3d` | `0.00%` | `0.000%` | Complete signal silence. The raw detector produced `0` buy and `0` sell decisions across `17,280` replay ticks, while the oracle found `5.38%` worth of OHLC movement. |
| `2022-11-11 5d` | `1.99%` | `1.977%` | Extreme volatility after the FTX crash, but short participation was tiny. Short entries totaled only about `$148` versus `$8,654` long entries. Most profit was open long inventory after the bounce. |
| `2022-05-07 1m` | `39.03%` | `4.791%` | Bear-market selloff with strong long bias. BTC closed `-12.5%`, but the bot was net long about `95%` of the time. It made money buying rebounds, yet missed most downside oracle opportunity. |
| `2021-12-22 3m` | `69.62%` | `6.951%` | Long-biased bear-market behavior. BTC closed `-12.4%` and had a `36%` close-to-close drawdown. The bot was net long about `95%` of the time; short exposure existed but was too small. |
| `2022-03-29 1m` | `9.88%` | `10.148%` | Late start and long bias. First entry came after about `44.6h`; BTC then closed `-15.1%`. The bot was net long about `89.8%` of the time and fees consumed about `12.9%` of net PnL. |
| `2023-09-10 1w` | `1.72%` | `13.213%` | Undertraded rally/chop. Only two long entries fired, with no exits and no shorts. The positive return was open long inventory marked at the final candle, not harvested closed PnL. |
| `2024-01-26 2m` | `76.90%` | `21.323%` | Strong bull trend where the bot was late and grid-like. First entry came after about `77.5h`; the bot took profit slices and opened shorts against the trend instead of staying maximally long. |
| `2024-11-24 3m` | `90.52%` | `28.177%` | High-volatility range with huge churn. The bot made money, but stayed net long almost all the time, carried very large gross exposure, left many open lots/orders, and paid about `$853` in fees. |
| `2024-04-24 2w` | `11.34%` | `32.746%` | Bearish/rebound window with long bias. BTC closed `-3.2%` and traded down `12%` from the start; the bot was net long about `93.8%` of the time and shorts were comparatively small. |

## Implications

The current strategy needs better regime and side control before parameter tuning can
meaningfully chase oracle capture:

- Add explicit trend/regime filters so valley buys are reduced or disabled in persistent
  downside trends.
- Make short sizing competitive with long sizing when downside momentum dominates.
- Track and report closed PnL separately from unrealized mark-to-market PnL in benchmark
  tables.
- Penalize stale inventory and open-order backlog in promotion criteria.
- Add a no-trade diagnostic for windows where the raw detector emits no decisions, so
  signal silence is visible instead of appearing as a flat `0%` result.
