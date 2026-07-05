# Recent KAMA 2x 100x Benchmark

Generated: 2026-07-05 20:13 UTC

Raw partial outputs:

- `data/benchmarks/recent-kama-thresholds-2026-07-05-192609.jsonl`
- `data/benchmarks/recent-kama-thresholds-2026-07-05-195705.jsonl`
- Aborted broad-sweep baseline rows: `data/benchmarks/recent-kama-thresholds-2026-07-05-191122.jsonl`

## Scope

- Market: BTCUSDT 1m spot candles from `data/historical/spot-btcusdt/btcusdt/1m`.
- Local cache used here ends at `2026-06-28 17:23 UTC`; it does not include July 2026 candles after that point.
- Strategy: `legacy-valley-peak`, `derivativeSource=kama`, `derivativeClampMode=deadband`, `buySigma=0.1`, `sellSigma=0.1`.
- Candidate: `2x` derivative thresholds, so the default 60s threshold `0.25` becomes `0.5`.
- KAMA: `erLen=20`, `fastLen=20`, `slowLen=200`, `power=1`.
- Account: `10000` USDT starting quote, futures-margin shorts, borrow depths `999/999`, `maxLeverage=100`.

## Status

The requested 90d full-resolution row did not complete in a reasonable runtime. Both
90d attempts were CPU-bound and were stopped manually:

- Normal range-leverage run: stopped during the 90d row after the 3d/7d/30d rows completed.
- Forced max-entry leverage run: stopped during the 90d row after the 3d/7d/30d rows completed.

The completed rows below are valid. The 90d result is missing, not negative or flat.

## Normal Range-Leverage Run

This uses `maxLeverage=100` as the account ceiling, but leaves the default
range-leverage selector enabled. Actual max entry leverage stayed near `4.9x`, and
max effective account leverage stayed at `1.0x`.

| Window | Dates | Actual | Market | Bot | Net PnL | Max DD | Trades | Closed win | Liq | Max entry lev | Max eff lev |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3d | `2026-06-26 00:00..2026-06-28 17:23` | `2.725d` | `-0.219%` | `+0.000%` | `+0.00` | `0.000%` | `0` | `0/0` | `0` | `1.000x` | `1.000x` |
| 7d | `2026-06-22 00:00..2026-06-28 17:23` | `6.725d` | `-5.762%` | `+0.008%` | `+0.75` | `0.018%` | `4` | `0/0` | `0` | `4.861x` | `1.000x` |
| 30d | `2026-05-30 00:00..2026-06-28 17:23` | `29.725d` | `-18.781%` | `+0.113%` | `+11.31` | `0.067%` | `34` | `9/9` | `0` | `4.888x` | `1.000x` |
| 90d | `2026-03-31 00:00..2026-06-28 17:23` | `89.725d` | pending | pending | pending | pending | pending | pending | pending | pending | pending |

## Forced Max-Entry Run

This disables the range-leverage selector so eligible entries request the configured
`100x` leverage. The 2x threshold still produced sparse exposure, so max effective
account leverage stayed at `1.0x` on completed rows.

| Window | Dates | Actual | Market | Bot | Net PnL | Max DD | Trades | Closed win | Liq | Max entry lev | Max eff lev |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3d | `2026-06-26 00:00..2026-06-28 17:23` | `2.725d` | `-0.219%` | `+0.000%` | `+0.00` | `0.000%` | `0` | `0/0` | `0` | `1.000x` | `1.000x` |
| 7d | `2026-06-22 00:00..2026-06-28 17:23` | `6.725d` | `-5.762%` | `+0.009%` | `+0.93` | `0.016%` | `7` | `1/1` | `0` | `100.000x` | `1.000x` |
| 30d | `2026-05-30 00:00..2026-06-28 17:23` | `29.725d` | `-18.781%` | `+1.089%` | `+108.91` | `0.857%` | `54` | `8/8` | `0` | `100.000x` | `1.000x` |
| 90d | `2026-03-31 00:00..2026-06-28 17:23` | `89.725d` | pending | pending | pending | pending | pending | pending | pending | pending | pending |

## Baseline Context

The aborted broader sweep completed only the `1x` threshold rows for 3d/7d/30d before
the `1x` 90d row became too slow. Those partial rows are still useful context:

| Window | Threshold | Market | Bot | Max DD | Trades | Liq |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3d | `1x` | `-0.219%` | `-0.293%` | `2.093%` | `4` | `0` |
| 7d | `1x` | `-5.762%` | `-0.335%` | `1.421%` | `58` | `0` |
| 30d | `1x` | `-18.781%` | `-21.445%` | `29.498%` | `565` | `0` |

## Read

On the completed recent suffixes, `2x KAMA deadband` is much more defensive than the
`1x` KAMA baseline. It avoids the severe recent 30d drawdown and remains positive in
both normal range-leverage and forced-entry modes.

This is not yet a complete long-window validation because the 90d full-resolution
backtest did not finish. The next engineering task is benchmark-runner performance:
either add checkpoint/progress reporting inside `runBacktestFromCandles`, or create a
faster long-window mode that periodically compacts resting order state without changing
trading semantics.
