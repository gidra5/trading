# Strategy Research Notes

The target remains the "perfect margin trader": full long on every upward move, full
short on every downward move, scaled by maximum leverage. This is an oracle benchmark,
not an achievable live strategy, because the real bot only sees past/current data and
pays fees, spread, slippage, funding, and liquidation risk.

The codebase now keeps one automated strategy: `legacy-valley-peak`. Removed strategy
families and deleted adaptive controls are no longer available as runtime choices or
benchmark controls. Research should therefore improve legacy valley/peak itself, its
execution model, its sizing, its filters, or the data it uses.

## Strategy Components

Every strategy improvement should be evaluated as an improvement to one of three basic
components:

1. Signal quality: decide when price is near a useful low or high and therefore when
   the bot should buy, sell, hold, or ignore the move.
2. Sizing: choose how much exposure to open or close based on current equity, borrowed
   balances, open orders, existing lots, volatility, and remaining opportunity budget.
3. Execution price: choose whether to trade immediately or wait for a better level,
   while accounting for fees, spread, slippage, missed fills, and stale-order risk.

The better these components are, the closer the realized strategy can move toward the
perfect-margin oracle. The signal decides whether there is an opportunity. The sizing
procedure decides how much of that opportunity to take. The execution price decides
whether the expected edge survives trading costs and fill risk.

Optimal size is not simply maximum leverage on every signal. It should preserve enough
capital to keep buying a deeper dip, but not preserve so much that the strategy fails to
extract profit from the current dip. Optimal execution price has the same tension: trade
too frequently and fees eat the edge; wait too long and volatility opportunities are
missed.

We also want to maximize utility of every dollar available. basically the more of our total capital we allocate at quality valleys/peaks, the more utility per dollar we get.

## Current Legacy Summary

Legacy valley/peak is a rolling-average turning-point strategy. The current default is
the relaxed per-lot peak exit-grid execution model:

- detects local valleys from clamped rolling-average derivative sign changes
- opens long lots at confirmed valleys with market-style entries
- recreates buy-to-cover ladders for active short lots at confirmed valleys
- detects local peaks from clamped rolling-average derivative sign changes
- opens short lots at confirmed peaks with market-style entries when `shortSideEnabled`
  is true
- places resettable targeted sell ladders between each lot's tracked peak and
  break-even sell price
- places resettable targeted buy ladders between each short lot's tracked trough and
  break-even buy price
- cancels stale unfilled orders after `staleOrderMs`
- tracks quote/base reserves while orders are open
- records fees, realized PnL, ledger-derived long/short average entries, drawdown, and
  win rate when orders fill

In the simulator those ladder orders use a below-price trigger, which is closer to
stop-ladder behavior than normal exchange sell-limit behavior. Short cover ladders
mirror this with above-price buy triggers. The older limit-only mode still exists as a
configuration path, but it is no longer the default.

The detailed implementation flow and current default parameters are documented in
[Algorithms](algorithms.md).

## Validation Standard

A legacy valley/peak change is not promotable unless it improves robust evidence, not
just one recent window. Preferred validation:

- cycle-wide random-length BTCUSDT windows
- folded grid search for parameter changes
- oracle capture versus the perfect-margin benchmark
- average, median, and P10 return
- max drawdown and risk-adjusted return
- trade count, fee burden, and stale-order rate
- explicit note of sample count, date range, leverage guard, cooldown, limit offset, and
  seed

The benchmark harness now produces legacy-only comparisons. Grid search should compare
legacy parameter variants, not separate strategy families.

## Current Observations

Legacy valley/peak remains useful but fragile:

- The detector can identify local turns, but it can buy too early during persistent
  downside trends.
- Resting limit orders improve nominal price but introduce missed-fill risk after fast
  reversals.
- Confirmation windows reduce noise but can delay exits.
- Gaussian sizing keeps largest trades near smooth derivative turns, but sigma and
  spend-rate settings are easy to overfit.
- Fees and stale orders remain large enough that high-churn parameter sets should be
  rejected quickly.
- Current BTCUSDT tests show three useful exit-grid regimes. On the 2026-05-23 to
  2026-06-21 fixed window, aggregate grid returned `-2.16%` with `5.16%` drawdown,
  per-lot strict returned `-10.18%` with `17.94%` drawdown, per-lot filled-grid
  relaxed returned `-10.20%` with `17.94%` drawdown, and default legacy returned
  `-14.99%` with `21.65%` drawdown.
- On the last-year 2025-06-22 to 2026-06-21 BTCUSDT window, aggregate grid was the
  safest variant (`-11.06%` return, `13.41%` drawdown). Default legacy returned
  `-33.77%`, while per-lot strict returned `-38.46%` and per-lot filled-grid relaxed
  returned `-38.96%` with `49.02%` drawdown.
- On 24 random 7-120 day BTCUSDT windows over the local 2021-07-16 to 2026-06-21
  cache, aggregate grid had the best downside profile (`0.69%` average return,
  `2.57%` average drawdown, `-5.18%` worst sample). Per-lot filled-grid relaxed had
  the best average return among grid variants (`1.73%`, `11.05%` average drawdown,
  `-12.44%` worst sample), slightly ahead of per-lot strict (`1.64%`). Default legacy
  averaged `-0.22%` with `13.96%` average drawdown and a `-25.92%` worst sample.
- On six folded BTCUSDT windows from 2021-07-16 to 2026-06-21, aggregate grid ranked
  first by risk-adjusted return (`2.67%` average return, `2.385` average risk return,
  `4.93%` average drawdown, `-10.19%` worst fold). Per-lot filled-grid relaxed ranked
  seventh (`14.06%` average return, `1.529` average risk return, `29.15%` average
  drawdown, `-42.24%` worst fold), ahead of per-lot strict by risk-adjusted return
  but behind default legacy on this folded run.
- Full folded validation became practical after replacing repeated position-ledger
  rebuilds with an incremental long-lot cache inside the strategy. Random-window runtime
  for per-lot rows dropped from roughly `40-43s` per row to `16-18s` per row on the same
  sample set.
- Portfolio mode over the four strategy equity curves favored a rolling-winner strategy
  mix (`316.19%` return, `71.39%` drawdown, `4.429` risk return). Inverse-vol had the
  best drawdown-adjusted shape (`158.02%` return, `42.07%` drawdown, `3.756` risk
  return). This is strategy allocation evidence only; multi-symbol allocation was
  skipped because the cached symbol set did not yield enough aligned usable series for
  that routine.
- After adding mirrored peak shorts and valley buy-to-cover grids, the focused
  2026-05-23 to 2026-06-21 BTCUSDT `1m` comparison improved the relaxed per-lot
  baseline but long/short remained unprofitable. At `1x`, long/short returned
  `-7.90%` with `12.10%` drawdown versus long-only `-9.85%` with `17.91%` drawdown.
  Short-only had zero fills at `1x` under the then-default `spot-borrow` model because
  the debt guard rejects borrowed-base shorts with no leverage headroom. At `5x`,
  long/short returned `-44.46%` with `78.27%` drawdown, long-only returned `-49.49%`
  with `84.47%` drawdown, and short-only returned `9.34%` with `16.34%` drawdown. All
  rows had zero liquidations. Use `shortMarginModel = "futures-margin"` for
  collateral-backed unlevered short-only validation.
- On the 30-day default synthetic sine-plus-noise series, `1x` long/short returned
  `61.48%` with `19.40%` drawdown and long-only returned `64.32%` with `22.53%`
  drawdown; short-only again had zero fills at `1x` under `spot-borrow`. At `5x`,
  long/short returned `319.69%` with `50.91%` drawdown, long-only returned `299.73%`
  with `69.57%` drawdown, and short-only returned `16.30%` with `87.77%` drawdown.
- After adding the `futures-margin` short model, `1x` short-only no longer has the
  zero-fill problem. On the same 2026-05-23 to 2026-06-21 historical BTCUSDT window,
  short-only returned `2.24%` with `8.22%` drawdown over `134` fills. On the 30-day
  default synthetic series, short-only returned `2.31%` with `28.80%` drawdown over
  `335` fills. Current relaxed long/short at `1x` under `futures-margin` returned
  `4.65%` with `5.46%` drawdown on the historical window and `4.65%` with `15.34%`
  drawdown on the synthetic series. All futures-margin checks had zero liquidations.
- After replacing the no-chain borrow rule with separate `longBorrowDepth` and
  `shortBorrowDepth`, relaxed per-lot long/short was tested at `1x` with
  `futures-margin`, `maxOpenOrders = 1024`, and seed `1337`. On 48 random 7-day BTCUSDT
  windows, `L2/S2` had the best average return at `0.79%`, followed by `L1/S2` at
  `0.77%`; both reduced the worst weekly result from roughly `-26%` to `-13.91%`.
  On the 2025-06-22 to 2026-06-21 one-year window, however, `L0/S0` was least bad at
  `-1.74%`, while deeper borrow chains increased drawdown and losses. All checked
  combinations had zero liquidations.

Borrow-depth matrix, random 7-day windows:

| Depth L/S | Avg Return | Median | Avg DD | Avg Trades |
| --- | ---: | ---: | ---: | ---: |
| `0/0` | `-0.11%` | `0.06%` | `3.09%` | `66.6` |
| `1/0` | `0.08%` | `0.12%` | `3.05%` | `76.8` |
| `0/1` | `0.05%` | `0.04%` | `3.23%` | `88.0` |
| `1/1` | `0.12%` | `0.06%` | `3.17%` | `84.3` |
| `1/2` | `0.77%` | `0.12%` | `2.78%` | `83.5` |
| `2/1` | `0.14%` | `0.06%` | `3.22%` | `90.9` |
| `2/2` | `0.79%` | `0.12%` | `2.83%` | `94.3` |

Borrow-depth matrix, 1-year window:

| Depth L/S | Return | Max DD | Trades |
| --- | ---: | ---: | ---: |
| `0/0` | `-1.74%` | `9.84%` | `637` |
| `1/0` | `-5.61%` | `12.28%` | `646` |
| `0/1` | `-8.99%` | `16.59%` | `890` |
| `1/1` | `-6.39%` | `13.70%` | `836` |
| `1/2` | `-6.56%` | `13.10%` | `818` |
| `2/1` | `-12.56%` | `20.06%` | `957` |
| `2/2` | `-10.29%` | `18.05%` | `865` |

Longer random-window borrow-depth checks used relaxed per-lot long/short, `1x`,
`futures-margin`, `maxOpenOrders = 1024`, `300s` cooldown, seed `1337`, and the
spot BTCUSDT 1m cache. The 365-day sample set only produced two heavily
overlapping April 2022-April 2023 windows, so treat it as a single-regime stress
check rather than broad one-year coverage.

Borrow-depth matrix, random 30-day windows, 8 samples:

| Depth L/S | Avg Return | Avg DD | Avg Trades | Best | Worst |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0/0` | `-0.61%` | `6.80%` | `134.9` | `5.62%` | `-13.48%` |
| `1/0` | `1.19%` | `6.68%` | `148.9` | `5.76%` | `-4.04%` |
| `0/1` | `-0.63%` | `7.13%` | `230.6` | `5.62%` | `-13.80%` |
| `1/1` | `-0.51%` | `6.31%` | `172.1` | `5.76%` | `-13.80%` |
| `1/2` | `2.23%` | `4.13%` | `190.4` | `5.76%` | `-1.54%` |
| `2/1` | `-0.41%` | `6.99%` | `238.6` | `5.76%` | `-13.80%` |
| `2/2` | `1.99%` | `4.79%` | `274.6` | `5.93%` | `-4.04%` |

Borrow-depth matrix, random 180-day windows, 3 samples:

| Depth L/S | Avg Return | Avg DD | Avg Trades | Best | Worst |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0/0` | `-2.81%` | `19.14%` | `300.3` | `5.92%` | `-12.76%` |
| `1/0` | `12.81%` | `11.04%` | `477.7` | `15.73%` | `9.50%` |
| `0/1` | `1.84%` | `21.48%` | `399.0` | `22.06%` | `-20.06%` |
| `1/1` | `4.19%` | `16.90%` | `396.3` | `28.57%` | `-18.93%` |
| `1/2` | `14.81%` | `8.40%` | `518.0` | `18.03%` | `12.67%` |
| `2/1` | `-7.14%` | `23.35%` | `339.3` | `3.91%` | `-19.47%` |
| `2/2` | `18.35%` | `10.72%` | `605.0` | `23.23%` | `13.82%` |

Borrow-depth matrix, random 365-day windows, 2 overlapping samples:

| Depth L/S | Avg Return | Avg DD | Avg Trades | Best | Worst |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0/0` | `17.95%` | `24.11%` | `514.5` | `19.04%` | `16.86%` |
| `1/0` | `34.23%` | `7.21%` | `990.0` | `39.05%` | `29.42%` |
| `0/1` | `2.57%` | `27.71%` | `501.5` | `2.99%` | `2.15%` |
| `1/1` | `23.06%` | `7.83%` | `743.5` | `35.42%` | `10.71%` |
| `1/2` | `48.70%` | `6.62%` | `956.5` | `58.59%` | `38.82%` |
| `2/1` | `12.01%` | `22.07%` | `643.0` | `16.05%` | `7.97%` |
| `2/2` | `47.77%` | `12.60%` | `1054.0` | `63.94%` | `31.60%` |

Activity diagnostic on the 2022-05-12 to 2022-11-08 180-day sample:

| Depth L/S | Return | Max DD | Trades | Fees | Orders | Cancelled | Exit-grid orders | Active L/S lots | Gross Exposure | Internal Borrow |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1/0` | `23.86%` | `11.34%` | `474` | `$93.73` | `694` | `219` | `481` | `81/94` | `$12559.85` | `0.17114899 BTC / $3174.34` |
| `1/2` | `19.66%` | `18.69%` | `544` | `$112.38` | `1167` | `620` | `963` | `66/75` | `$12103.77` | `0.15008842 BTC / $3401.08` |
| `2/1` | `-6.23%` | `23.16%` | `435` | `$104.38` | `683` | `243` | `532` | `67/40` | `$9407.01` | `0.02696449 BTC / $2248.64` |
| `2/2` | `28.36%` | `8.46%` | `672` | `$163.96` | `1910` | `1219` | `1690` | `55/76` | `$12958.04` | `0.09058885 BTC / $5007.26` |

Longer windows show two competing effects. `S2` often improves sampled returns by
letting profitable opposing positions subsidize bad inventory, but every extra hop
materially increases lots, exit-grid resets, cancellations, fees, and pending
inventory. The suspicious case is `L2/S1`: it is worse in both 180-day and
365-day checks, with high drawdown despite many profitable closed positions. That
suggests the second long-origin hop can over-recycle capital into local valleys
inside larger down moves, leaving the book with stale long exposure while the
closed-position stats still look healthy. Since the extrema detector is local and
not trend-aware, deeper chains amplify false reversals instead of adding edge; the
strategy needs regime/signal qualification before promoting deeper borrow limits.

The 2021-12-03 to 2022-06-01 worst 180-day window exposed a grid reset edge case:
tiny extrema-side partials could fall below `minOrderQuote`, get skipped, and leave
only break-even-side orders to be repeatedly cancelled on resets. The grid builder now
sweeps the whole remaining lot when a partial would create a below-minimum remainder,
and filled-grid resets only reuse filled points from the active grid unless price makes
a new higher peak or lower trough than the previous reset.

Exact 1m replay of that window after the fix:

| Depth L/S | Return | Max DD | Trades |
| --- | ---: | ---: | ---: |
| `0/0` | `-34.65%` | `37.66%` | `404` |
| `1/0` | `53.35%` | `9.13%` | `1209` |
| `0/1` | `-36.40%` | `39.98%` | `358` |
| `1/1` | `-7.66%` | `14.12%` | `1020` |
| `1/2` | `58.63%` | `8.38%` | `1336` |
| `2/1` | `-8.77%` | `14.44%` | `1056` |
| `2/2` | `53.54%` | `9.28%` | `1779` |

After swapping the extrema procedures, the default detector now uses the old peak
procedure for valley buys (`1m` source with `30m`/`10m` rising confirmations) and the
old valley procedure for peak sells (`1s` source with `12h` falling confirmation). This
turns long entries into broader-uptrend pullbacks and short entries into broader-downtrend
bounce peaks. Exact 1m replay of the same 2021-12-03 to 2022-06-01 window:

| Depth L/S | Return | Max DD | Trades |
| --- | ---: | ---: | ---: |
| `0/0` | `25.27%` | `10.47%` | `496` |
| `1/0` | `29.44%` | `11.95%` | `469` |
| `0/1` | `44.54%` | `17.86%` | `407` |
| `1/1` | `46.38%` | `17.10%` | `448` |
| `1/2` | `27.72%` | `8.88%` | `503` |
| `2/1` | `46.38%` | `17.10%` | `448` |
| `2/2` | `46.31%` | `17.10%` | `448` |

Symmetric detector checks on the same exact window:

- Symmetric permissive means both sides use the `1s` primary with `12h` trend
  confirmation. Every borrow-depth case converged to the same path: `54.27%`
  return, `26.94%` max drawdown, and `229` trades.
- Symmetric strict means both sides use the `1m` primary with `30m`/`10m` trend
  confirmations.

| Strict Depth L/S | Return | Max DD | Trades |
| --- | ---: | ---: | ---: |
| `0/0` | `29.46%` | `0.85%` | `926` |
| `1/0` | `28.38%` | `3.20%` | `898` |
| `0/1` | `73.32%` | `6.20%` | `2688` |
| `1/1` | `36.68%` | `2.95%` | `1123` |
| `1/2` | `35.01%` | `3.48%` | `1171` |
| `2/1` | `85.78%` | `6.27%` | `3163` |
| `2/2` | `45.74%` | `1.77%` | `1889` |

Odd-depth strict-symmetric check on the same window:

| Strict Depth L/S | Return | Max DD | Trades |
| --- | ---: | ---: | ---: |
| `1/1` | `36.68%` | `2.95%` | `1123` |
| `1/3` | `44.66%` | `4.44%` | `1432` |
| `3/1` | `50.64%` | `5.81%` | `1866` |
| `3/3` | `88.72%` | `5.98%` | `3011` |

Next odd-depth strict-symmetric check:

| Strict Depth L/S | Return | Max DD | Trades |
| --- | ---: | ---: | ---: |
| `3/5` | `98.05%` | `2.59%` | `3782` |
| `5/3` | `94.70%` | `7.29%` | `3204` |
| `5/5` | `104.65%` | `1.80%` | `3879` |

An effectively unlimited strict-symmetric depth run with `999/999` produced the same
result as `5/5` on this window: `104.65%` return, `1.80%` max drawdown, and `3879`
trades. The natural chain depth therefore appears to saturate by depth 5 for this
specific replay.

Strict-symmetric fixed last-window anchor, captured in the Codex session log on
2026-06-23, used `1x`, `futures-margin`, exact BTCUSDT 1m candles, `300s` cooldown,
`maxOpenOrders = 1024`, `maxPositionQuote = 10000`, `exitGridResetMode = "filled-grid"`,
and no borrow-lock/profit-share config. The detector was strict symmetric:
`buyDataIndex = 1`, `sellDataIndex = 1`, `buyConfirmationOffsets = [2, 1]`, and
`sellConfirmationOffsets = [2, 1]`.

This detector is the baseline default and is exported as
`legacyValleyPeakStrictSymmetricConfig`, with the reference key
`strictSymmetric035Anchor`.

| Window | Depth | Interval | Return | Net PnL | Max DD | Trades | Win Rate | Prof Pos |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 30d | `5/5` | 2026-05-22..2026-06-21 | `5.12%` | `$512.39` | `2.85%` | `878` | `74.9%` | `92/109` |
| 30d | `999/999` | 2026-05-22..2026-06-21 | `5.12%` | `$512.39` | `2.85%` | `878` | `74.9%` | `92/109` |
| 180d | `5/5` | 2025-12-23..2026-06-21 | `0.35%` | `$34.79` | `8.43%` | `1093` | `67.7%` | `104/131` |
| 180d | `999/999` | 2025-12-23..2026-06-21 | `0.35%` | `$34.79` | `8.43%` | `1093` | `67.7%` | `104/131` |
| 365d | `5/5` | 2025-06-21..2026-06-21 | `23.11%` | `$2311.47` | `7.98%` | `1570` | `80.3%` | `152/169` |
| 365d | `999/999` | 2025-06-21..2026-06-21 | `23.11%` | `$2311.47` | `7.98%` | `1570` | `80.3%` | `152/169` |

The exact 180d anchor interval starts at `2025-12-23T20:22:00Z`, not midnight. A later
audit reran that exact interval with current code and explicit `lockBorrowedLenderCollateral = false`
plus `borrowerProfitShareToLender = 1`; it reproduced `0.35%`, `$34.79`, and `1093`
trades. Raising only `maxPositionQuote` from `10000` to `1000000` did not change that
strict `5/5` anchor row, so the position cap was not the cause of the original near-flat
result.

Strict-symmetric random 90d exact 1m matrix, same anchor config, seed `1337`, 50 samples:

| Depth | Samples | Profitable | Avg Return | Median | P10 | Avg Max DD | Avg Trades | Best | Worst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `5/5` | `50` | `48/50` | `28.38%` | `27.86%` | `4.42%` | `5.98%` | `989.9` | `82.91%` | `-4.40%` |
| `999/999` | `50` | `48/50` | `28.47%` | `27.86%` | `4.42%` | `5.98%` | `996.8` | `83.18%` | `-4.87%` |

The 90d sample set was strong and mostly profitable, but the 180d fixed anchor showed
that one recent half-year path could still give back most of the early gains. Depth
`999/999` was essentially identical to `5/5` on 30d/90d/180d/365d recent checks, so
unlimited depth was not useful there.

Full-cycle strict-symmetric 1800d checks changed that conclusion slightly. The `5/5`
run returned `253.50%` on `2021-07-17..2026-06-21`. A tracked `999/999` run returned
`261.14%`, `$26114.14` net PnL, `4794` trades, `81.3%` win rate, and `659/769`
profitable closed positions. The tracked run reached max long depth `6` and max short
depth `7`, first in late 2021. So full-cycle unlimited depth exceeded `5/5`, but only
modestly; it did not grow without bound.

Asymmetric short-favoring detector checks used strict buys with permissive sells:
`buyDataIndex = 1`, `buyConfirmationOffsets = [2, 1]`, `sellDataIndex = 0`, and
`sellConfirmationOffsets = [6]`, with `7/7`, `1x`, `futures-margin`, and exact 1m
replay.

This detector is retained as `legacyValleyPeakAsymmetricShortFavoringConfig`, with the
reference key `asymmetricShortFavoring`, but it is not the baseline default.

| Window | Interval | Market | Return | Max DD | Trades | Prof Pos |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 180d recent | 2025-12-23..2026-06-21 | - | `27.89%` | `14.05%` | `405` | `10/10` |
| 180d uptrend | 2023-09-16..2024-03-13 | `174.71%` | `107.15%` | `4.86%` | `400` | `43/45` |
| 1800d full cycle | 2021-07-17..2026-06-21 | `101.62%` | `213.97%` | `51.78%` | `1686` | `230/253` |

This variant was much better on the troublesome recent 180d interval, but the 1800d
drawdown was very high. It looks more like a directional short-trigger bias than a
robust all-regime improvement.

Random 30d strict-symmetric `7/7` leverage scaling, 50 exact 1m samples:

| Leverage | Profitable | Stopped | Avg Return | Median | P10 | Avg Max DD | Avg Trades | Liquidated Positions | Best | Worst |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1x` | `43/50` | `0` | `12.41%` | `9.38%` | `-1.01%` | `5.13%` | `488.2` | `0` | `42.13%` | `-5.20%` |
| `2x` | `46/50` | `0` | `33.19%` | `30.49%` | `4.76%` | `9.04%` | `884.4` | `0` | `91.53%` | `-3.06%` |
| `3.5x` | `44/50` | `2` | `56.65%` | `56.27%` | `-0.29%` | `17.31%` | `1041.1` | `15` | `171.97%` | `-100.15%` |
| `5x` | `44/50` | `2` | `85.95%` | `91.31%` | `-0.41%` | `22.01%` | `1109.4` | `13` | `257.76%` | `-100.24%` |

Leverage helped average returns but introduced liquidation/tail-loss failures at `3.5x`
and `5x`. The practical no-leverage-first rule still stands; paired leverage tests are
useful for stress validation, not for selecting parameters by average return alone.

Borrow-lock/profit-share experiments were added after observing that lender lots could
still exit-grid free account base while another position had borrowed from that lot.
The historical note below was captured during that implementation pass on the
troublesome recent 180d replay with `7/7`, `1x`, `futures-margin`, and exact 1m
candles:

| Lock Lent Collateral | Borrower Profit Share To Lender | Return | Max DD | Trades | Prof Pos |
| --- | ---: | ---: | ---: | ---: | ---: |
| off | `1.00` | `0.35%` | `8.43%` | `1093` | `104/131` |
| off | `0.50` | `2.21%` | `7.73%` | `1163` | `115/148` |
| off | `0.00` | `1.32%` | `7.19%` | `1273` | `130/164` |
| on | `1.00` | `1.81%` | `6.86%` | `523` | `16/17` |
| on | `0.50` | `1.81%` | `6.86%` | `523` | `16/17` |
| on | `0.00` | `1.81%` | `6.86%` | `523` | `16/17` |

The later audit found that the `off / 1.00` row was conflated with the strict-symmetric
fixed-window anchor above. Treat this table as historical context for why locking was
added, not as a clean current-code reproduction. Under the asymmetric-detector checkpoint
with `7/7`, the same calendar span measured around `5.8%` to `6.9%` depending on lock
and cap settings.

Current-code cap checks on the same calendar replay:

| Borrow Lock | Cap | Return | Max DD | Trades |
| --- | ---: | ---: | ---: | ---: |
| off | `1x` | `5.94%` | `9.73%` | `329` |
| off | `100x` | `6.89%` | `9.40%` | `326` |
| on | `1x` | `5.82%` | `9.47%` | `326` |
| on | `100x` | `6.58%` | `9.40%` | `320` |

The reproduction audit also tried nearby current-code variants on the same calendar
span. The current swapped detector returned about `5.94%` with the old `10000` cap and
about `6.89%` once the cap was raised. A strict-symmetric override returned `3.83%`
regardless of `10000`, `50000`, or `1000000` cap, so the cap was not the reason it
failed to reproduce `0.35%`. Reset-mode changes also did not reproduce the anchor:
swapped/filled returned `6.68%`, strict/higher returned `3.83%`, and strict/filled
returned `3.53%`. Reverting the sell-confirmation sign produced `2.51%` at both caps,
also not the anchor.

Strict-symmetric grid order-count probes on that calendar span were non-monotonic:
`3` orders returned `-5.90%`, `6` returned `3.83%`, `8` returned `6.99%`, `10`
returned `10.97%`, and `12` returned `8.50%`. More grid levels can help, but this is
parameter sensitivity, not a clean explanation for the original 180d near-flat row.

The cap increase from `startingQuote * leverage` to `startingQuote * 100` removed a
hidden per-side bucket during low-leverage tests, but it did not remove the main
capacity blocker. Instrumentation on the locked `7/7`, `1x`, futures-margin replay
showed that March-June peak signals were mostly blocked by gross exposure capacity,
not by borrow depth or by short-side cap:

| Month | Peak Signals | Quote OK | Gross Cap Blocked | Sell Opens | Min Short Cap | Max Gross Exposure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-03 | `955` | `0` | `955` | `0` | `$1745.18` | `$13019.56` |
| 2026-04 | `302` | `0` | `302` | `0` | `$1366.59` | `$13616.69` |
| 2026-05 | `247` | `0` | `247` | `0` | `$1001.16` | `$14193.03` |
| 2026-06 | `753` | `27` | `726` | `27` | `$1920.56` | `$12692.70` |

The exact condition was `grossEntryLeverageCapacityQuote(...) < minOrderQuote`, which
makes short entry selling power zero before borrow allocation is even attempted. Internal
borrow improves lot basis accounting, but the current guard still counts both long and
short exposure toward gross leverage capacity at `1x`.

Implementation follow-up: entry power now adds internal opposite-side borrow capacity on
top of free-capital side-cap/leverage capacity, and `futures-margin` leverage checks use
gross exposure net of internally borrowed paired exposure. The strict-symmetric exact
anchor replay with the same `1x`, `futures-margin`, `5/5`, `maxPositionQuote = 10000`,
and `300s` settings moved from `0.35%` to `3.07%` return, `13.58%` max drawdown, and
`1808` trades. March 2026 no longer had zero activity: it produced `58` short opens and
`16` long opens, though March-June still had no later close fills inside that anchor.

Current `999/999` baseline verification, captured on 2026-06-24 after the internal-borrow
capacity and futures-margin netting changes, used exact BTCUSDT `1m` candles from
`data/historical/spot-btcusdt/btcusdt/1m` with no resampling. Commands used the
strict-symmetric reference via `--only strict-symmetric`.

Exact config:

- `startingQuote = 10000`, `maxPositionQuote = 10000`, `maxLeverage = 1`
- `shortMarginModel = "futures-margin"`
- `longBorrowDepth = 999`, `shortBorrowDepth = 999`
- `lockBorrowedLenderCollateral = false`
- `borrowerProfitShareToLender = 1`
- `maxOpenOrders = 1024`, `cooldownMs = 300000`, `staleOrderMs = 2592000000`
- `feeBps = 7.5`, `limitOffsetBps = 2`, `minOrderQuote = 25`
- detector: `buyDataIndex = 1`, `sellDataIndex = 1`,
  `buyConfirmationOffsets = [2, 1]`, `sellConfirmationOffsets = [2, 1]`
- detector sizing: `saturationSec = 3600`, `buySpendRate = 1`,
  `sellAmountRate = 1`, `trendSigmaA = 1`, `trendSigmaSellB1 = 1`,
  `trendSigmaBuyB2 = 1`,
  `minTradeQuote = 25`, `maxTradeQuote = 50000`
- exit grid: enabled, market entry enabled, `exitGridOrderCount = 6`,
  `exitGridPriceDistribution = "uniform"`,
  `exitGridSizeDistribution = "geometric"`, `exitGridSellFraction = 0.35`,
  `exitGridMinProfitBps = 20`, `exitGridResetBps = 10`,
  `exitGridPositionMode = "per-lot"`, `exitGridResetMode = "filled-grid"`

Fixed-window exact runs:

| Window | Interval               |     Candles |      Return |       Net PnL |  Max DD |   Risk Ret |  Sharpe |   Trades | Win Rate |              Prof Pos | Liq Pos | Oracle Capture |
| ------ | ---------------------- | ----------: | ----------: | ------------: | ------: | ---------: | ------: | -------: | -------: | --------------------: | ------: | -------------: |
| 30d    | 2026-05-23..2026-06-21 |    `42,982` |     `7.94%` |     `$794.47` | `3.05%` |    `2.602` | `3.977` |    `970` |  `81.5%` |     `120/135 (88.9%)` |     `0` |      `13.216%` |
| 180d   | 2025-12-24..2026-06-21 |   `258,982` |   `124.98%` |   `$12497.64` | `9.34%` |   `13.374` | `5.982` |   `7227` |  `81.1%` |   `1048/1191 (88.0%)` |     `0` |      `24.729%` |
| 365d   | 2025-06-22..2026-06-21 |   `525,382` |   `457.22%` |   `$45722.47` | `6.21%` |   `73.676` | `7.180` |  `22250` |  `79.2%` |   `2879/3362 (85.6%)` |     `0` |      `56.592%` |
| 1800d  | 2021-07-18..2026-06-21 | `2,591,312` | `13212.88%` | `$1321287.67` | `4.79%` | `2759.891` | `6.070` | `100436` |  `79.7%` | `10228/12837 (79.7%)` |     `0` |     `130.083%` |

The fixed-window commands were:

```bash
npm run benchmark:strategies -- --mode days --days 30 --only strict-symmetric
npm run benchmark:strategies -- --mode days --days 180 --only strict-symmetric
npm run benchmark:strategies -- --mode days --days 365 --only strict-symmetric
npm run benchmark:strategies -- --mode days --days 1800 --only strict-symmetric
```

The 1800d exact run took `6245949ms` (`1h44m06s`). It also reported
`9.60e+45%` reinvested return and `0.000000%` reinvested oracle capture; the additive
oracle capture is the more useful comparator for this row.

Random 90d exact run, seed `1337`, `50` samples, `90-90` day windows, `1825` day
lookback, cache span `2021-07-16..2026-06-21`, first sample
`2022-05-27..2022-08-25`, last sample `2026-01-09..2026-04-09`:

| Samples | Profitable | Avg Return |   Median |      P10 | Avg Net PnL | Avg PnL/day | Avg Max DD | Avg Risk Ret | Avg Sharpe | Avg Trades |          Avg Prof Pos | Avg Liq Pos | Avg Capture |     Best |   Worst |
| ------: | ---------: | ---------: | -------: | -------: | ----------: | ----------: | ---------: | -----------: | ---------: | ---------: | --------------------: | ----------: | ----------: | -------: | ------: |
|    `50` |    `50/50` |   `47.99%` | `46.55%` | `13.89%` |  `$4799.29` |    `$53.33` |    `7.00%` |      `9.992` |    `5.821` |   `2180.6` | `304.7/345.7 (90.4%)` |       `0.0` |   `21.067%` | `96.34%` | `3.73%` |

The random-window command was:

```bash
npm run benchmark:strategies -- --mode random-lengths --min-window-days 90 --max-window-days 90 --samples 50 --only strict-symmetric
```

Depth-tracking replays used `scripts/track-borrow-depth.ts` with the same config and
same exact candle replay sequence. These runs measure used chain depth as
`configuredDepth - borrowDepthRemaining` while tracked lots are active:

| Window |    Return |  Max DD |  Trades |            Prof Pos | Liq Pos | Max Long Depth | Max Short Depth |
| ------ | --------: | ------: | ------: | ------------------: | ------: | -------------: | --------------: |
| 30d    |   `7.94%` | `3.05%` |   `970` |   `120/135 (88.9%)` |     `0` |            `3` |             `4` |
| 180d   | `124.98%` | `9.34%` |  `7227` | `1048/1191 (88.0%)` |     `0` |            `6` |             `7` |
| 365d   | `457.22%` | `6.21%` | `22250` | `2879/3362 (85.6%)` |     `0` |            `8` |             `9` |

The depth-tracking commands were:

```bash
npx tsx scripts/track-borrow-depth.ts --days 30
npx tsx scripts/track-borrow-depth.ts --days 180
npx tsx scripts/track-borrow-depth.ts --days 365
```

Absolute-rate no-leverage baseline, captured on 2026-06-25 before relative rates became
the default, used exact BTCUSDT `1m` candles with
`--only "Legacy Valley/Peak Long/Short"` and no `--leverage` override. This kept the
benchmark runner at `1x max leverage`, `futures-margin`, `999/999` borrow depth,
`maxPositionQuote = 10000`, `300s` cooldown, and the old absolute derivative thresholds.

Fixed-window exact reruns:

| Window | Interval | Candles | Return | Net PnL | Max DD | Risk Ret | Sharpe | Trades | Win Rate | Prof Pos | Liq Pos | Oracle Ret | Oracle Capture | Reinvest Ret | Reinvest Capture |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30d | 2026-05-26..2026-06-24 | `42,940` | `30.25%` | `$3025.42` | `2.87%` | `10.528` | `13.258` | `2360` | `82.0%` | `214/239 (89.5%)` | `0` | `60.77%` | `49.784%` | `83.43%` | `36.263083%` |
| 180d | 2025-12-27..2026-06-24 | `258,940` | `237.76%` | `$23776.29` | `1.00%` | `237.906` | `17.021` | `16193` | `84.4%` | `1533/1672 (91.7%)` | `0` | `505.68%` | `47.019%` | `15462.04%` | `1.537720%` |
| 365d | 2025-06-25..2026-06-24 | `525,340` | `487.87%` | `$48786.66` | `1.18%` | `414.355` | `15.496` | `40074` | `81.4%` | `3746/4172 (89.8%)` | `0` | `809.31%` | `60.282%` | `318674.00%` | `0.153093%` |

The same 365d benchmark with `legacyValleyPeak.relativeRateEnabled=true`, now the
default detector mode, was rerun with `--relative-rates` before the default flip:

| Window | Interval | Candles | Return | Net PnL | Max DD | Risk Ret | Sharpe | Trades | Win Rate | Prof Pos | Liq Pos | Oracle Ret | Oracle Capture | Reinvest Ret | Reinvest Capture |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 365d relative rates | 2025-06-25..2026-06-24 | `525,340` | `563.10%` | `$56310.37` | `1.17%` | `482.495` | `16.952` | `51123` | `81.1%` | `4838/5400 (89.6%)` | `0` | `809.31%` | `69.578%` | `318674.00%` | `0.176702%` |

The fixed-window rerun commands were:

```bash
npm run benchmark:strategies -- --mode days --days 30 --interval 1m --symbol BTCUSDT --only "Legacy Valley/Peak Long/Short" --absolute-rates
npm run benchmark:strategies -- --mode days --days 180 --interval 1m --symbol BTCUSDT --only "Legacy Valley/Peak Long/Short" --absolute-rates
npm run benchmark:strategies -- --mode days --days 365 --interval 1m --symbol BTCUSDT --only "Legacy Valley/Peak Long/Short" --absolute-rates
npm run benchmark:strategies -- --mode days --days 365 --interval 1m --symbol BTCUSDT --only "Legacy Valley/Peak Long/Short" --relative-rates
```

After relative rates became the default, rerun old absolute-rate baselines with
`--absolute-rates`.

Previous `$10000`-cap padded-range leverage baseline, captured on 2026-06-25, used
`maxLeverage = 5`, `maxPositionQuote = 10000`, and the relative-rate detector default. Entry
leverage is selected from the 1y price range with `3%` padding: long liquidation below
`1y min * 0.97`, short liquidation above `1y max * 1.03`. The near-edge lifetime branch
is disabled. `Max Entry Lev` is the highest selected entry leverage on created entry
orders; `Max Eff Lev` is the highest sampled account effective leverage.

Fixed-window exact runs:

| Window | Interval | Candles | Return | Net PnL | Max DD | Risk Ret | Sharpe | Max Entry Lev | Max Eff Lev | Trades | Win Rate | Prof Pos | Liq Pos | Oracle Ret | Oracle Capture | Reinvest Ret | Reinvest Capture |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30d | 2026-05-26..2026-06-24 | `42,940` | `48.61%` | `$4861.11` | `0.86%` | `56.684` | `23.731` | `5.000x` | `1.000x` | `5485` | `78.9%` | `537/603 (89.1%)` | `0` | `303.85%` | `15.998%` | `1934.45%` | `2.512919%` |
| 180d | 2025-12-27..2026-06-24 | `258,940` | `323.70%` | `$32369.79` | `0.67%` | `479.893` | `20.894` | `5.000x` | `1.000x` | `29868` | `81.1%` | `2861/3176 (90.1%)` | `0` | `2528.40%` | `12.803%` | `7.60e+12%` | `0.000000%` |
| 365d | 2025-06-25..2026-06-24 | `525,340` | `559.16%` | `$55916.39` | `1.24%` | `450.350` | `17.101` | `5.000x` | `1.000x` | `51136` | `81.3%` | `4859/5401 (90.0%)` | `0` | `4046.56%` | `13.818%` | `2.02e+19%` | `0.000000%` |

The 30d and 180d commands were run after `5x` became the benchmark default, but before
the default position cap changed to `startingQuote * maxLeverage`. To reproduce those
rows now, add `--max-position-quote 10000`:

```bash
npm run benchmark:strategies -- --mode days --days 30 --only "Legacy Valley/Peak Long/Short"
npm run benchmark:strategies -- --mode days --days 180 --only "Legacy Valley/Peak Long/Short"
```

The 365d row was run with the equivalent explicit flag before the benchmark default was
updated:

```bash
npm run benchmark:strategies -- --mode days --days 365 --leverage 5 --only "Legacy Valley/Peak Long/Short"
```

Random 90d exact rerun, seed `1337`, `48` samples, `90-90` day windows, `1825` day
lookback, cache span `2021-07-16..2026-06-24`, first sample
`2022-05-28..2022-08-26`, last sample `2023-10-02..2023-12-31`:

| Samples | Profitable | Avg Return |   Median |      P10 | Avg Net PnL | Avg PnL/day | Avg Max DD | Avg Risk Ret | Avg Sharpe | Avg Trades |          Avg Prof Pos | Avg Liq Pos | Avg Capture | Avg Reinvest Cap |      Best |   Worst |
| ------: | ---------: | ---------: | -------: | -------: | ----------: | ----------: | ---------: | -----------: | ---------: | ---------: | --------------------: | ----------: | ----------: | ---------------: | --------: | ------: |
|    `48` |    `48/48` |   `99.13%` | `89.01%` | `20.67%` |  `$9912.59` |   `$110.14` |    `4.33%` |     `64.528` |   `11.824` |   `5107.6` | `475.5/520.0 (93.6%)` |       `0.0` |   `40.592%` |     `19.717953%` | `229.48%` | `6.53%` |

The random-window rerun command was:

```bash
npm run benchmark:strategies -- --mode random-lengths --interval 1m --symbol BTCUSDT --min-window-days 90 --max-window-days 90 --samples 48 --seed 1337 --lookback-days 1825 --only "Legacy Valley/Peak Long/Short"
```

Higher-sample checkpoint, 30m OHLC proxy: exact 1m 50-sample long-window matrices
were too slow for a 30-minute checkpoint, so the benchmark runner was extended with
`--borrow-depth-matrix` and `--resample-minutes`. The 30d and 180d matrices below
completed with 50 samples per depth case. A 365d 50-sample pass was stopped before
producing a table; a reduced 10-sample 365d pass also exceeded the checkpoint while
still inside `L0/S1`, so the previous exact 2-sample 365d table remains the only
complete 1-year depth matrix for now.

Borrow-depth matrix, random 30-day windows, 50 samples, 30m OHLC proxy:

| Depth L/S | Avg Return | Median | P10 | Avg DD | Avg Trades | Best | Worst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0/0` | `3.83%` | `3.75%` | `-3.75%` | `8.55%` | `277.4` | `22.94%` | `-27.21%` |
| `1/0` | `7.74%` | `5.38%` | `2.92%` | `4.93%` | `464.4` | `27.69%` | `-3.13%` |
| `0/1` | `4.20%` | `4.48%` | `-3.78%` | `8.99%` | `292.7` | `23.73%` | `-27.21%` |
| `1/1` | `5.45%` | `5.58%` | `-1.36%` | `7.19%` | `381.3` | `22.12%` | `-27.25%` |
| `1/2` | `8.10%` | `5.91%` | `3.03%` | `4.91%` | `472.7` | `27.70%` | `-3.13%` |
| `2/1` | `4.79%` | `4.61%` | `-3.50%` | `8.12%` | `350.6` | `22.90%` | `-27.25%` |
| `2/2` | `6.93%` | `5.60%` | `1.82%` | `5.58%` | `460.1` | `22.97%` | `-12.01%` |

Borrow-depth matrix, random 180-day windows, 50 samples, 30m OHLC proxy:

| Depth L/S | Avg Return | Median | P10 | Avg DD | Avg Trades | Best | Worst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0/0` | `11.00%` | `11.57%` | `-14.16%` | `14.12%` | `649.3` | `47.27%` | `-43.92%` |
| `1/0` | `23.64%` | `16.26%` | `5.31%` | `7.96%` | `1084.5` | `84.22%` | `-43.92%` |
| `0/1` | `10.94%` | `15.25%` | `-17.80%` | `16.57%` | `838.1` | `34.17%` | `-43.92%` |
| `1/1` | `20.94%` | `17.91%` | `6.47%` | `10.89%` | `976.6` | `81.39%` | `-43.75%` |
| `1/2` | `24.73%` | `18.89%` | `4.89%` | `8.54%` | `1087.6` | `84.22%` | `-43.92%` |
| `2/1` | `13.50%` | `16.67%` | `-10.83%` | `13.84%` | `955.5` | `36.43%` | `-43.75%` |
| `2/2` | `22.33%` | `23.10%` | `6.86%` | `10.07%` | `1200.7` | `70.87%` | `-43.92%` |

The higher-sample proxy keeps the same broad ranking as the smaller exact checks:
`L1/S2` and `L1/S0` are strongest, `L0/S1` adds weaker short-origin churn, and
`L2/S1` remains inferior to the one-hop long-origin variants. The proxy also makes
the cost of deeper chains clearer: the best-return variants roughly double trade
count versus `0/0`, and `2/2` is the highest-activity case.

## Skipped Position-Cap Baselines, 2026-06-25

The default position cap was changed to uncapped on 2026-06-25. These rows use exact
BTCUSDT `1m` candle replays, `futures-margin`, `999/999` borrow depth, borrow lock off,
lender profit share `1`, open-order cap `1024`, and `300s` cooldown. `Max Eff Lev` is
account-level gross exposure divided by current equity after mark-to-market. The
effective leverage cap now force-reduces exposure when mark-to-market drift would push
the account over the configured max leverage.

Standard capital, default `$5` minimum order, `5x` max leverage, no position cap:

| Window | Return | Net PnL | Max DD | Risk Ret | Sharpe | Max Entry | Max Eff | Trades | Prof Pos | Liq |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7d | `24.12%` | `$2411.79` | `4.61%` | `5.230` | `11.056` | `5.000x` | `5.000x` | `831` | `69/88 (78.4%)` | `0` |
| 30d | `367.56%` | `$36755.55` | `10.26%` | `35.828` | `15.704` | `5.000x` | `4.993x` | `7259` | `492/644 (76.4%)` | `0` |
| 90d | `2147.91%` | `$214790.90` | `10.09%` | `212.948` | `14.518` | `5.000x` | `5.000x` | `20746` | `1142/1302 (87.7%)` | `0` |
| 180d | `9124.22%` | `$912421.87` | `3.38%` | `2696.861` | `14.489` | `5.000x` | `5.000x` | `59620` | `3197/3601 (88.8%)` | `0` |
| 365d | `16396.91%` | `$1639691.25` | `6.77%` | `2421.286` | `10.359` | `5.000x` | `4.993x` | `104465` | `5410/6089 (88.8%)` | `0` |

Small capital, `$134` starting quote, `$50` minimum order, `5x` max leverage, no
position cap:

| Window | Return | Net PnL | Max DD | Risk Ret | Sharpe | Max Entry | Max Eff | Trades | Prof Pos | Liq |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7d | `19.08%` | `$25.56` | `5.23%` | `3.646` | `9.449` | `5.000x` | `4.935x` | `93` | `19/27 (70.4%)` | `0` |
| 30d | `289.88%` | `$388.44` | `13.01%` | `22.287` | `13.729` | `5.000x` | `4.978x` | `802` | `148/214 (69.2%)` | `0` |
| 180d | `283531.06%` | `$379931.63` | `24.67%` | `11492.184` | `13.238` | `5.000x` | `5.000x` | `24555` | `2001/2431 (82.3%)` | `0` |

Small capital, `$134` starting quote, `$50` minimum order, `100x` max leverage, no
position cap:

| Window | Return | Net PnL | Max DD | Risk Ret | Sharpe | Max Entry | Max Eff | Trades | Prof Pos | Liq |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7d | `210.65%` | `$282.27` | `20.71%` | `10.170` | `14.177` | `27.164x` | `28.123x` | `235` | `44/47 (93.6%)` | `0` |
| 30d | `18118.42%` | `$24278.68` | `34.47%` | `525.671` | `15.572` | `26.258x` | `27.645x` | `3392` | `356/410 (86.8%)` | `0` |
| 90d | `116978.32%` | `$156750.95` | `20.78%` | `5630.115` | `14.528` | `27.345x` | `23.554x` | `12024` | `931/1064 (87.5%)` | `0` |
| 180d | `680515.10%` | `$911890.24` | `15.75%` | `43210.629` | `11.306` | `26.797x` | `24.285x` | `39631` | `2642/2998 (88.1%)` | `0` |
| 365d | `1197896.60%` | `$1605181.44` | `27.89%` | `42943.453` | `8.178` | `27.751x` | `26.689x` | `70714` | `4566/5198 (87.8%)` | `0` |

Tiny capital, `$34` starting quote, `$50` minimum order, `100x` max leverage, no
position cap:

| Window | Return | Net PnL | Max DD | Risk Ret | Sharpe | Max Entry | Max Eff | Trades | Prof Pos | Liq |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7d | `111.60%` | `$37.94` | `21.37%` | `5.222` | `11.089` | `27.164x` | `25.422x` | `98` | `19/26 (73.1%)` | `0` |
| 30d | `5656.74%` | `$1923.29` | `77.36%` | `73.119` | `9.819` | `26.258x` | `70.038x` | `1281` | `207/260 (79.6%)` | `0` |

With `100x` max leverage the padded-range selector still usually chooses only about
`26x` to `28x`. Effective leverage can exceed selected entry leverage after
mark-to-market because account equity changes while exposure remains open; the enforced
hard cap is the configured max leverage.

The most important next improvement is better signal qualification. The strategy needs
to know whether a detected valley/peak has enough expected move after costs and fill
risk before it opens or exits exposure.

## Reproducible Candidate Configs

### Static 0.1 Anticipatory Confirmation 10%

Recorded on 2026-07-05 after changing anticipation to confirm signals instead of
placing anticipatory limit orders. This variant keeps the normal valley/peak primary
trigger, but allows exactly one lagging confirmation to pass when a quadratic fit of
the recent 30m price trend predicts the matching extremum within 10% of the fit
window. If both confirmations are still failing, the signal remains blocked.

Backtest scope:

- Market: BTCUSDT `1m` candles from `data/historical/spot-btcusdt/btcusdt/1m`.
- Files: `2026-06-22.jsonl` through `2026-06-28.jsonl`.
- Exact replay span: `2026-06-22T00:00:00.000Z` to
  `2026-06-28T17:23:59.999Z`.
- Candles: `9,684`.
- Account/model: default strategy account settings, including `10000` USDT starting
  quote, `5x` max leverage, futures-margin shorts, and default borrow depths.

Config override:

```ts
const config = {
  symbol: "BTCUSDT",
  algorithm: "legacy-valley-peak",
  startingQuote: defaultStrategyConfig.startingQuote,
  maxLeverage: defaultStrategyConfig.maxLeverage,
  shortMarginModel: defaultStrategyConfig.shortMarginModel,
  longBorrowDepth: defaultStrategyConfig.longBorrowDepth,
  shortBorrowDepth: defaultStrategyConfig.shortBorrowDepth,
  internalBorrowAccounting: defaultStrategyConfig.internalBorrowAccounting,
  borrowerProfitShareToLender: defaultStrategyConfig.borrowerProfitShareToLender,
  maxPositionQuote: defaultStrategyConfig.maxPositionQuote,
  minOrderQuote: defaultStrategyConfig.minOrderQuote,
  maxOpenOrders: defaultStrategyConfig.maxOpenOrders,
  cooldownMs: defaultStrategyConfig.cooldownMs,
  staleOrderMs: defaultStrategyConfig.staleOrderMs,
  legacyValleyPeak: {
    sigmaMode: "static",
    buySigma: 0.1,
    sellSigma: 0.1,
    exitGridEnabled: true,
    exitGridMarketEntry: true,
    exitGridPositionMode: "per-lot",
    exitGridResetMode: "filled-grid",
    anticipatoryConfirmationEnabled: true,
    anticipatoryConfirmationWindowSec: 30 * 60,
    anticipatoryConfirmationLookaheadFraction: 0.1,
  },
};
```

Observed result:

| Return | Net PnL | Final Equity | Max DD | Oracle Return | Oracle Capture | Trades | Win Rate | Closed Profitable | Liq |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `28.6894%` | `$2868.94` | `$12868.94` | `7.2934%` | `675.2428%` | `4.2487%` | `1118` | `99.2639%` | `90/90` | `0` |

Extrema order-mass metrics:

| Side | Fills | Quote | Threshold Mass | P99 Frame | P99 Price Distance |
| --- | ---: | ---: | ---: | ---: | ---: |
| Buy near valleys | `754` | `$273279.77` | `2.7686%` | `40.00m` | `0.8743%` |
| Sell near peaks | `364` | `$285247.35` | `0.0000%` | `41.66m` | `1.0139%` |

Baseline on the same window with `anticipatoryConfirmationEnabled=false` returned
`-1.2826%` with `-0.1899%` oracle capture. The same anticipation rule made the current
default sizing worse on this window (`-37.3103%` return), so this is a reproducible
signal/sizing candidate rather than a promoted default.

Stress test on hand-picked intervals from `tasks.md`, using fresh memory per interval
and UTC day-inclusive windows. The candidate was profitable on `4/18` intervals, had
zero liquidations, and averaged roughly `-9.26%` return across the set. This failed the
robustness check: it worked on some high-churn reversal/range intervals, but lost
heavily in sustained trend weeks and several choppy near-flat weeks.

| Group | Interval | Market | Return | Max DD | Capture | Trades |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Choppy week | `2022-07-28..2022-08-03` | `-0.59%` | `-29.38%` | `29.95%` | `-2.49%` | `1987` |
| Choppy week | `2022-05-14..2022-05-20` | `-0.29%` | `-19.27%` | `29.50%` | `-1.32%` | `3258` |
| Choppy week | `2021-12-14..2021-12-20` | `+0.45%` | `-29.48%` | `31.72%` | `-2.14%` | `2216` |
| Choppy week | `2021-09-08..2021-09-14` | `+0.52%` | `-6.02%` | `30.10%` | `-0.41%` | `1960` |
| Choppy week | `2023-03-18..2023-03-24` | `+0.22%` | `+20.20%` | `10.15%` | `+1.46%` | `2381` |
| Regime week up | `2023-03-11..2023-03-17` | `+35.95%` | `-42.93%` | `43.61%` | `-1.76%` | `1768` |
| Regime week sideways | `2026-04-22..2026-04-28` | `+0.01%` | `+12.82%` | `6.91%` | `+4.66%` | `612` |
| Regime week down | `2022-06-12..2022-06-18` | `-33.26%` | `-46.38%` | `59.26%` | `-0.68%` | `3163` |
| 3d up low churn | `2024-02-24..2024-02-26` | `+7.36%` | `-1.84%` | `1.95%` | `-1.48%` | `24` |
| 3d up high churn | `2022-06-19..2022-06-21` | `+9.24%` | `-4.53%` | `14.53%` | `-0.24%` | `1955` |
| 3d down low churn | `2023-06-03..2023-06-05` | `-5.56%` | `-23.77%` | `25.79%` | `-14.63%` | `27` |
| 3d down high churn | `2022-06-13..2022-06-15` | `-15.02%` | `+21.39%` | `32.59%` | `+0.46%` | `1662` |
| 3d sideways high bias churn | `2021-10-19..2021-10-21` | `+0.30%` | `-4.76%` | `17.96%` | `-0.67%` | `918` |
| 3d sideways high bias low churn | `2025-02-14..2025-02-16` | `-0.51%` | `-6.46%` | `6.87%` | `-8.53%` | `98` |
| 3d sideways low bias churn | `2024-07-07..2024-07-09` | `-0.31%` | `-13.64%` | `16.09%` | `-2.81%` | `570` |
| 3d sideways low bias low churn | `2025-07-04..2025-07-06` | `-0.35%` | `-0.22%` | `0.22%` | `-0.59%` | `6` |
| 3d sideways mid bias churn | `2024-01-02..2024-01-04` | `-0.06%` | `+7.82%` | `14.03%` | `+1.23%` | `363` |
| 3d sideways mid bias low churn | `2023-09-15..2023-09-17` | `+0.02%` | `-0.19%` | `0.24%` | `-0.39%` | `5` |

## KAMA Derivative Source Test

Implemented `legacyValleyPeak.derivativeSource = "kama"` as a primary extrema
source only. In `price` mode, the primary signal source remains the configured raw
price SMA, usually the 1m SMA. In `kama` mode, the primary valley/peak extrema come
from KAMA, while confirmation windows, exit confirmations, trend sigma sizing, market
state, and chart SMAs still use raw prices.

The test below interprets the provided KAMA lengths as minutes. The candle backtest
replays each 1m candle as four OHLC ticks, so `erLen=5m`, `fastLen=5m`,
`slowLen=50m` were run as `kamaErLen=20`, `kamaFastLen=20`, `kamaSlowLen=200`,
`kamaPower=1`. Static sigmas are `buySigma=0.1`, `sellSigma=0.1`, mode `both`,
borrow depths `999/999`.

Corrected KAMA is still better than raw price source on average, but it is not a
universal improvement. It cuts average drawdown and trade count substantially, and
rescues the June 2022 stress downtrends, but introduces a severe `-11.67%` loss on
the `2022-06-19..2022-06-21` high-churn uptrend.

| Version | Avg return | Avg DD | Avg trades | Positive / flat / negative |
| --- | ---: | ---: | ---: | ---: |
| Raw price source | `-2.38%` | `4.22%` | `449` | `4 / 0 / 16` |
| KAMA primary source | `+0.37%` | `1.53%` | `68.7` | `14 / 1 / 5` |

| Case | Interval | KAMA return | KAMA DD | KAMA trades |
| --- | --- | ---: | ---: | ---: |
| Up low churn | `2024-02-24..2024-02-26` | `-0.01%` | `0.01%` | `2` |
| Up high churn | `2022-06-19..2022-06-21` | `-11.67%` | `11.68%` | `68` |
| Down low churn | `2023-06-03..2023-06-05` | `+0.06%` | `0.06%` | `6` |
| Down high churn | `2022-06-13..2022-06-15` | `+0.49%` | `0.41%` | `266` |
| Side high bias churn | `2021-10-19..2021-10-21` | `-0.06%` | `0.99%` | `92` |
| Side high bias low churn | `2025-02-14..2025-02-16` | `+0.00%` | `0.00%` | `1` |
| Side low bias churn | `2024-07-07..2024-07-09` | `-1.31%` | `3.45%` | `13` |
| Side low bias low churn | `2025-07-04..2025-07-06` | `+0.00%` | `0.00%` | `1` |
| Side mid bias churn | `2024-01-02..2024-01-04` | `+0.56%` | `0.33%` | `51` |
| Side mid bias low churn | `2023-09-15..2023-09-17` | `0.00%` | `0.00%` | `0` |
| Trend 3d up 2024-11 | `2024-11-09..2024-11-11` | `+1.86%` | `0.78%` | `51` |
| Trend 3d up 2023-12 | `2023-12-03..2023-12-05` | `+0.28%` | `1.15%` | `66` |
| Trend 3d down 2026-06 | `2026-06-01..2026-06-03` | `-2.42%` | `2.89%` | `49` |
| Trend 3d down 2023-03 | `2023-03-07..2023-03-09` | `+3.69%` | `1.98%` | `16` |
| Trend 7d up 2023-12 | `2023-11-29..2023-12-05` | `+0.43%` | `0.42%` | `81` |
| Trend 7d up 2024-11 | `2024-11-05..2024-11-11` | `+2.00%` | `1.11%` | `128` |
| Trend 7d down 2023-03 | `2023-03-03..2023-03-09` | `+2.34%` | `1.55%` | `37` |
| Trend 7d down 2026-06 | `2026-05-27..2026-06-02` | `+0.18%` | `0.00%` | `3` |
| Stress 3d down 2022-06 | `2022-06-11..2022-06-13` | `+5.54%` | `2.18%` | `197` |
| Stress 7d down 2022-06 | `2022-06-07..2022-06-13` | `+5.48%` | `1.53%` | `245` |

## Derivative Clamp Hysteresis Test

Implemented `legacyValleyPeak.derivativeClampMode = "hysteresis"` as an experimental
alternative to the previous stateless deadband clamp. In this mode a derivative leaves
zero only after crossing the configured threshold, then keeps its active positive or
negative sign until the raw derivative crosses zero.

Benchmark report:
`docs/derivative-clamp-mode-benchmark-2026-07-05-170933.md`

Scope: the 28 UTC day-inclusive intervals from `tasks.md`, BTCUSDT 1m candles, static
`buySigma=0.1`, `sellSigma=0.1`, price derivative source, both sides enabled,
futures-margin shorts, borrow depths `999/999`, and `1x` max leverage.

Result: hysteresis was worse overall. It improved return on only `6/28` intervals and
lowered max drawdown on `4/28`. Average return moved from `-2.647%` with deadband to
`-8.485%`; average max drawdown moved from `5.038%` to `10.321%`; average trades almost
doubled from `747.6` to `1485`.

Main read: zero-cross persistence is too slow for the current price-source detector.
It keeps directional state active through churn, causing more orders and deeper adverse
inventory. Keep `deadband` as the default; use hysteresis only as an explicit experiment
or retest it later with a smoother derivative source/regime gate.

KAMA follow-up report:
`docs/derivative-clamp-mode-benchmark-2026-07-05-173814.md`

Using the same benchmark scope with `derivativeSource=kama` and minute-scaled KAMA
parameters `erLen=20`, `fastLen=20`, `slowLen=200`, hysteresis still failed. KAMA
deadband averaged `-0.163%` return, `2.175%` max drawdown, and `106.6` trades;
KAMA hysteresis averaged `-6.256%` return, `8.225%` max drawdown, and `272.8`
trades. Hysteresis improved return on only `7/28` intervals and lowered drawdown on
only `1/28`. The June 2022 stress cases are decisive: KAMA deadband returned
`+5.535%` and `+5.483%`, while KAMA hysteresis returned `-10.591%` and `-22.353%`.

Inner-threshold follow-up report:
`docs/derivative-threshold-grid-2026-07-05-180107.md`

Added `legacyValleyPeak.derivativeClampInnerThresholdRatio` so hysteresis can exit
at a threshold between zero and the outer deadband instead of waiting for raw zero
cross. On the KAMA grid, inner thresholds did reduce the damage from zero-cross
persistence, but they did not beat the clean larger-deadband candidate. Best practical
setting in this sweep was KAMA `deadband` with `2x` outer thresholds: average return
improved from `-0.163%` to `+0.247%`, average max drawdown fell from `1.960%` to
`0.492%`, max drawdown fell from `15.333%` to `5.833%`, and average trades fell from
`106.6` to `14.5`.

The top average-return row was KAMA `4x` zero-cross hysteresis at `+0.292%`, but it
was mostly a sparse stress-case capture: only `9/28` intervals were positive and the
median return was `0.000%`. Treat it as an experiment, not a default candidate. The
more actionable region is KAMA with moderately wider outer thresholds, especially
`2x` deadband or `2x` with `inner=1`, which is effectively the same boundary behavior.

## Research Backlog

| Direction | Why it may help legacy | Main risk |
| --- | --- | --- |
| Limit-execution tuning | Jointly tune `limitOffsetBps`, stale timing, and order count to improve fill quality. | Can overfit synthetic OHLC fill paths. |
| Peak exit-grid tuning | Tune grid order count, price distribution, size distribution, sell fraction, reset policy, stale timing, and entry sizing. | Current below-trigger simulator may overstate fills after gaps. |
| Triple-barrier labels | Label candidate entries by take-profit, stop-loss, timeout, and net return after costs. | Easy to overfit barrier and horizon settings. |
| Cost-aware accept/reject model | Learn when a detected valley/peak is worth trading after fees and missed-fill risk. | Data leakage and unstable calibration. |
| Funding and premium features | Perpetual funding can dominate flat price movement and crowded positioning. | Funding extremes can persist for long periods. |
| Open-interest and liquidation features | Helps distinguish real continuation from weak squeeze/liquidation moves. | Exchange data quality and interpretation are regime-dependent. |
| Order-flow imbalance | May improve very short-horizon entry timing around detected valleys/peaks. | Requires realistic spread, latency, depth, and partial-fill simulation. |
| Volatility targeting | Reduce exposure when realized volatility makes long inventory fragile. | Can reduce size before profitable volatility expansion. |
| Correlation-aware portfolio layer | Avoid running many copies of the same BTC-beta exposure across symbols. | Needs synchronized multi-symbol history. |
| Anti-overfit reporting | Track trials, folds, and statistical confidence before promoting parameters. | Adds process overhead, but prevents false winners. |

## Near-Term Implementation Steps

1. Tune peak exit-grid parameters across random-length and folded validation.
2. Compare aggregate low-drawdown grid against per-lot filled-grid relaxed as separate
   risk profiles rather than forcing one winner too early.
3. Add benchmark columns for created, filled, stale-cancelled, and leverage-cancelled
   orders.
4. Tune limit offset and stale timing across folded/random-length validation.
5. Add triple-barrier labels around legacy candidate entries.
6. Train a small accept/reject model using only features available at signal time.
7. Add funding, mark/index premium, open interest, and liquidation history for futures
   markets.
8. Cache enough liquid symbols to evaluate legacy as a portfolio-level allocator.
9. Delay RL until the simulator includes funding, liquidation, latency, spread, and
   partial-fill assumptions.

## References

| Area | Reference | Useful idea | Application here |
| --- | --- | --- | --- |
| Backtest overfitting | [Bailey, Borwein, Lopez de Prado, Zhu - The Probability of Backtest Overfitting](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253) | Strategy search can select false winners. | Track number of trials, walk-forward splits, and later PBO-style reporting. |
| Deflated Sharpe | [Bailey and Lopez de Prado - The Deflated Sharpe Ratio](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551) | Sharpe needs correction for selection bias and non-normal returns. | Add confidence checks before promoting legacy parameters. |
| Triple-barrier labeling | [Mlfin.py triple-barrier and meta-labeling docs](https://mlfinpy.readthedocs.io/en/latest/Labelling.html) | Labels should include profit, stop, and timeout outcomes. | Add cost-aware labels for legacy valley/peak candidates. |
| Time-series momentum | [Moskowitz, Ooi, Pedersen - Time Series Momentum](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2089463) | Past return can predict continuation across futures markets. | Add trend/regime filters before trading a valley or peak. |
| Volatility timing | [Moreira and Muir - Volatility Managed Portfolios](https://www.nber.org/papers/w22208) | Reducing exposure when volatility is high can improve risk-adjusted returns. | Add volatility throttles around legacy sizing. |
| Order-flow imbalance | [Cont, Kukanov, Stoikov - The Price Impact of Order Book Events](https://arxiv.org/abs/1011.6402) | Bid/ask order-flow imbalance can explain short-horizon price changes. | Add order-book features only after execution simulation is realistic. |
| Perpetual futures mechanics | [He, Manela, Ross, von Wachter - Fundamentals of Perpetual Futures](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4301150) | Funding links perpetual futures and spot prices. | Include funding and basis in PnL and features. |
| Binance funding data | [Binance funding-rate FAQ](https://www.binance.com/en/support/faq/detail/360033525031) and [funding-rate history API](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History) | Funding history is available from the exchange. | Cache funding beside candles. |
| Online portfolio selection | [Li and Hoi - Online Portfolio Selection: A Survey](https://arxiv.org/abs/1212.2129) | Sequential allocation can be benchmarked with simple online allocators. | Use portfolio allocation as a later layer around legacy signals. |
| Covariance shrinkage | [Ledoit and Wolf - A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices](https://www.ledoit.net/ole1a.pdf) | Sample covariance is unstable with short history. | Use shrinkage before multi-symbol risk allocation. |
| RL caution | [Deep Reinforcement Learning in Quantitative Algorithmic Trading: A Review](https://arxiv.org/abs/2106.00123) | Many DRL trading studies use unrealistic settings. | Keep RL behind simulator realism and robust validation. |
