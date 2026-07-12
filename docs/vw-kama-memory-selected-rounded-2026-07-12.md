# Volume-weighted KAMA oracle approximation search

Generated: 2026-07-11T23:48:43.586Z
Raw results: `data/benchmarks/vw-kama-memory-selected-rounded-2026-07-12.jsonl`

The fit ranking never reads validation or holdout scores. The best validated volume-aware and canonical (`volumePower=0`) candidates are the only candidates evaluated on holdout.

## Data and objective

- Source: `data/historical/spot-btcusdt/btcusdt/1s` (1s); target scales: 1s, 5s, 15s, 1m, 5m, 15m, 1h.
- Windows: fit 2021-09-08..2021-12-20, 2022-05-14..2022-08-03, 2023-03-03..2023-12-05; validation 2024-01-02..2024-02-26, 2024-07-07..2024-11-11; test 2025-02-14..2025-07-06, 2026-04-22..2026-07-10.
- Each continuous segment reserves 3d before scoring.
- Signal: completed-candle volume-weighted KAMA derivative rate; candidates either go flat inside the deadband or hold their prior exposure until the opposite threshold.
- Signal memory: after the first signal, a candidate state change emits only when the current close is strictly more than 17.5 bps from the last emitted signal price; rejected changes retain the prior state. This is the same friction used by the oracle.
- Matching is symmetric nearest-event comparison by direction. It permits multiple signals around one oracle event; timing credit halves every configured half-life, while distant chatter and missed events reduce precision and recall.
- Case score weights timing-credited transition F1 80% and graded oracle-state agreement 20% (exact 1, flat/directional 0.5, opposite 0). Trading returns and execution prices are neither computed nor ranked.
- Candidate objective equally weights the median and P10 case score; every scale/window case has equal weight.

## Holdout finalists

| family | id | objective | median | P10 | precision | recall | F1 | agreement | signals/day | lag P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| volume | v0001 | 26.14% | 35.83% | 16.45% | 23.52% | 44.73% | 30.82% | 57.1% | 28.33 | 1442500ms |
| canonical | k0001 | 26.14% | 35.83% | 16.45% | 23.52% | 44.73% | 30.82% | 57.1% | 28.33 | 1442500ms |

## Validation finalists

| family | id | objective | median | P10 | precision | recall | F1 | agreement | signals/day | lag P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical | k0001 | 39.04% | 54.85% | 23.22% | 43.57% | 65.63% | 52.37% | 63.83% | 82.75 | 11m |
| volume | v0001 | 39.04% | 54.85% | 23.22% | 43.57% | 65.63% | 52.37% | 63.83% | 82.75 | 11m |

## Best fit candidates

| family | id | objective | median | P10 | precision | recall | F1 | agreement | signals/day | lag P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical | k0001 | 36.58% | 57.45% | 15.72% | 47.19% | 68.81% | 55.99% | 61.4% | 93.98 | 10m |
| volume | v0001 | 36.58% | 57.45% | 15.72% | 47.19% | 68.81% | 55.99% | 61.4% | 93.98 | 10m |

## Finalist parameters

| family | id | efficiency | fast | slow | power | volume | cap | volume power | deadband | mode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| volume | v0001 | 840000ms | 1680000ms | 9180000ms | 0.49 | 7800000ms | 2.65 | 0 | 67.567 bps/hour | flat |
| canonical | k0001 | 840000ms | 1680000ms | 9180000ms | 0.49 | 7800000ms | 2.65 | 0 | 67.567 bps/hour | flat |

## Holdout cases

| candidate | scale/window | score | precision | recall | F1 | agreement | signals/day | lag P50 | lag P90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v0001 | 1s:test-1 | 41.92% | 26.41% | 56.22% | 35.94% | 65.87% | 35 | 1303500ms | 4240300ms |
| v0001 | 1s:test-2 | 56.88% | 45.88% | 71.35% | 55.85% | 60.99% | 90.24 | 674s | 3409100ms |
| v0001 | 5s:test-1 | 35.7% | 23.9% | 47.21% | 31.74% | 51.58% | 30.83 | 1385s | 4765s |
| v0001 | 5s:test-2 | 54.08% | 42.27% | 67.65% | 52.03% | 62.3% | 79.17 | 770s | 3535s |
| v0001 | 15s:test-1 | 38.26% | 26.33% | 47.6% | 33.9% | 55.67% | 25.83 | 1147500ms | 4057500ms |
| v0001 | 15s:test-2 | 50.97% | 39.01% | 63.71% | 48.39% | 61.26% | 70.6 | 855s | 65m |
| v0001 | 1m:test-1 | 32.21% | 20.01% | 36.99% | 25.97% | 57.16% | 22.33 | 27m | 80m |
| v0001 | 1m:test-2 | 46.41% | 33.53% | 58.51% | 42.63% | 61.53% | 59.29 | 18m | 74m |
| v0001 | 5m:test-1 | 26.31% | 15.75% | 25.6% | 19.5% | 53.53% | 16.17 | 30m | 80m |
| v0001 | 5m:test-2 | 35.95% | 23.14% | 42.25% | 29.9% | 60.15% | 37.64 | 25m | 85m |
| v0001 | 15m:test-1 | 18.04% | 9.45% | 9.84% | 9.64% | 51.65% | 5.5 | 2250s | 6030000ms |
| v0001 | 15m:test-2 | 27.02% | 15.38% | 26.71% | 19.52% | 57.03% | 20.1 | 30m | 105m |
| v0001 | 1h:test-1 | 10.6% | 0.67% | 0.29% | 0.41% | 51.39% | 1.17 | 1h | 1h |
| v0001 | 1h:test-2 | 15.77% | 8.32% | 5.65% | 6.73% | 51.93% | 3.43 | 1h | 2h |
| k0001 | 1s:test-1 | 41.92% | 26.41% | 56.22% | 35.94% | 65.87% | 35 | 1303500ms | 4240300ms |
| k0001 | 1s:test-2 | 56.88% | 45.88% | 71.35% | 55.85% | 60.99% | 90.24 | 674s | 3409100ms |
| k0001 | 5s:test-1 | 35.7% | 23.9% | 47.21% | 31.74% | 51.58% | 30.83 | 1385s | 4765s |
| k0001 | 5s:test-2 | 54.08% | 42.27% | 67.65% | 52.03% | 62.3% | 79.17 | 770s | 3535s |
| k0001 | 15s:test-1 | 38.26% | 26.33% | 47.6% | 33.9% | 55.67% | 25.83 | 1147500ms | 4057500ms |
| k0001 | 15s:test-2 | 50.97% | 39.01% | 63.71% | 48.39% | 61.26% | 70.6 | 855s | 65m |
| k0001 | 1m:test-1 | 32.21% | 20.01% | 36.99% | 25.97% | 57.16% | 22.33 | 27m | 80m |
| k0001 | 1m:test-2 | 46.41% | 33.53% | 58.51% | 42.63% | 61.53% | 59.29 | 18m | 74m |
| k0001 | 5m:test-1 | 26.31% | 15.75% | 25.6% | 19.5% | 53.53% | 16.17 | 30m | 80m |
| k0001 | 5m:test-2 | 35.95% | 23.14% | 42.25% | 29.9% | 60.15% | 37.64 | 25m | 85m |
| k0001 | 15m:test-1 | 18.04% | 9.45% | 9.84% | 9.64% | 51.65% | 5.5 | 2250s | 6030000ms |
| k0001 | 15m:test-2 | 27.02% | 15.38% | 26.71% | 19.52% | 57.03% | 20.1 | 30m | 105m |
| k0001 | 1h:test-1 | 10.6% | 0.67% | 0.29% | 0.41% | 51.39% | 1.17 | 1h | 1h |
| k0001 | 1h:test-2 | 15.77% | 8.32% | 5.65% | 6.73% | 51.93% | 3.43 | 1h | 2h |
