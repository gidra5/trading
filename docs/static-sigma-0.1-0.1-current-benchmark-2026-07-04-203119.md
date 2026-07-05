# Current Strategy Static Sigma 0.1/0.1 Benchmark

Generated: 2026-07-05T01:58:39.657Z
Status: partial
Raw results: `data/benchmarks/static-sigma-0.1-0.1-current-2026-07-04-203119.jsonl`

## Scope

- Market: BTCUSDT 1m from `data/historical/spot-btcusdt/btcusdt/1m`.
- Cache span: 2021-07-16T07:52:00Z to 2026-06-28T17:23:59.999Z (1,809 day files).
- Strategy: current `legacy-valley-peak` defaults with only sigma mode overridden.
- Static sigma override: `sigmaMode=static`, `buySigma=0.1`, `sellSigma=0.1`.
- Account model: 10000 USDT starting quote, 5x max leverage, futures-margin shorts, borrow depths 999/999.
- Random window seed: 1337.
- Completed rows: 3/6 fixed windows, 0/70 random samples.
- Note: the requested 5y fixed window is limited by available local cache to 1,809 day files.

## Stop Note

This run was stopped after the `6m` fixed window had been running for several
hours without appending a completed row. The durable JSONL contains only fully
completed benchmark rows, so the report includes `1w`, `1m`, and `3m`. The
requested `6m`, `1y`, `5y`, `30d` random samples, and `1w` random samples are
not included.

The throughput issue is benchmark-runner/simulator related rather than data
availability related: the process stayed CPU-bound and memory-stable, but the
current 5x static-sigma configuration made long continuous replays impractical
in this path.

## Fixed Windows

| window | actual span | candles | market | return | net PnL | max DD | perfect | capture | trades | closed win | liq | stop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1w | 6.725d | 9,684 | -5.7622% | -1.1896% | -$118.96 | 4.3601% | +675.2428% | -0.1762% | 622 | 74/74 | 0 | completed |
| 1m | 29.725d | 42,804 | -18.7814% | -10.9877% | -$1098.77 | 13.6200% | +2912.5712% | -0.3772% | 4,040 | 498/498 | 0 | completed |
| 3m | 89.725d | 129,204 | -10.6794% | -69.1663% | -$6916.63 | 70.8125% | +5799.8653% | -1.1926% | 8,282 | 803/803 | 0 | completed |

## Random Window Aggregates

| group | samples | profitable | avg return | median | p10 | worst | best | avg/day | avg max DD | avg capture | avg trades | avg liq |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30d random | 0/20 | 0/0 | - | - | - | - | - | - | - | - | - | - |
| 1w random | 0/50 | 0/0 | - | - | - | - | - | - | - | - | - | - |
| _none completed_ |  |  |  |  |  |  |  |  |  |  |  |  |

## Random Sample Details

### 30d random

| sample | dates | market | return | net PnL | max DD | capture | trades | closed win | liq | stop |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| _none completed_ |  |  |  |  |  |  |  |  |  |  |

### 1w random

| sample | dates | market | return | net PnL | max DD | capture | trades | closed win | liq | stop |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| _none completed_ |  |  |  |  |  |  |  |  |  |  |
