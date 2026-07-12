# Volume-weighted KAMA oracle approximation search

Generated: 2026-07-11T23:35:57.509Z
Raw results: `data/benchmarks/vw-kama-memory-multiscale-k0214-baseline-2026-07-12.jsonl`

The fit ranking never reads validation or holdout scores. The best validated volume-aware and canonical (`volumePower=0`) candidates are the only candidates evaluated on holdout.

## Data and objective

- Source: `data/historical/spot-btcusdt/btcusdt/1s` (1s); target scales: 1s, 5s, 15s, 1m, 5m, 15m, 1h.
- Windows: fit 2021-09-08..2021-12-20, 2022-05-14..2022-08-03, 2023-03-03..2023-12-05; validation 2024-01-02..2024-02-26, 2024-07-07..2024-11-11; test 2025-02-14..2025-07-06, 2026-04-22..2026-07-10.
- Signal: completed-candle volume-weighted KAMA derivative rate; candidates either go flat inside the deadband or hold their prior exposure until the opposite threshold.
- Signal memory: after the first signal, a candidate state change emits only when the current close is strictly more than 17.5 bps from the last emitted signal price; rejected changes retain the prior state. This is the same friction used by the oracle.
- Matching is symmetric nearest-event comparison by direction. It permits multiple signals around one oracle event; timing credit halves every configured half-life, while distant chatter and missed events reduce precision and recall.
- Case score weights timing-credited transition F1 80% and graded oracle-state agreement 20% (exact 1, flat/directional 0.5, opposite 0). Trading returns and execution prices are neither computed nor ranked.
- Candidate objective equally weights the median and P10 case score; every scale/window case has equal weight.

## Holdout finalists

| family | id | objective | median | P10 | precision | recall | F1 | agreement | signals/day | lag P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| volume | v0001 | 26.54% | 39.26% | 13.83% | 27.31% | 50.22% | 33.96% | 58.12% | 32.65 | 1271s |
| canonical | k0001 | 26.54% | 39.26% | 13.83% | 27.31% | 50.22% | 33.96% | 58.12% | 32.65 | 1271s |

## Validation finalists

| family | id | objective | median | P10 | precision | recall | F1 | agreement | signals/day | lag P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical | k0001 | 36.44% | 58.24% | 14.65% | 46.45% | 74.02% | 57.08% | 60.67% | 111.65 | 11m |
| volume | v0001 | 36.44% | 58.24% | 14.65% | 46.45% | 74.02% | 57.08% | 60.67% | 111.65 | 11m |

## Best fit candidates

| family | id | objective | median | P10 | precision | recall | F1 | agreement | signals/day | lag P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical | k0001 | 38.03% | 60.61% | 15.44% | 49.45% | 76.71% | 60.14% | 59.85% | 137.95 | 9m |
| volume | v0001 | 38.03% | 60.61% | 15.44% | 49.45% | 76.71% | 60.14% | 59.85% | 137.95 | 9m |

## Finalist parameters

| family | id | efficiency | fast | slow | power | volume | cap | volume power | deadband | mode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| volume | v0001 | 124173ms | 3206ms | 6122590ms | 1.604 | 86400000ms | 1 | 0 | 191.739 bps/hour | flat |
| canonical | k0001 | 124173ms | 3206ms | 6122590ms | 1.604 | 86400000ms | 1 | 0 | 191.739 bps/hour | flat |

## Holdout cases

| candidate | scale/window | score | precision | recall | F1 | agreement | signals/day | lag P50 | lag P90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v0001 | 1s:test-1 | 41.77% | 28.19% | 60.18% | 38.4% | 55.24% | 39.33 | 1271s | 4394s |
| v0001 | 1s:test-2 | 57.95% | 46.5% | 74.21% | 57.17% | 61.04% | 106.57 | 650s | 3364800ms |
| v0001 | 5s:test-1 | 43.65% | 29.69% | 56.72% | 38.98% | 62.33% | 33.33 | 1132500ms | 3830s |
| v0001 | 5s:test-2 | 55.67% | 43.63% | 72.33% | 54.43% | 60.61% | 102.52 | 760s | 3535s |
| v0001 | 15s:test-1 | 43.02% | 26.43% | 56.74% | 36.07% | 70.83% | 33.33 | 1290s | 4623000ms |
| v0001 | 15s:test-2 | 53.32% | 41.18% | 69.35% | 51.67% | 59.92% | 91.1 | 14m | 3675s |
| v0001 | 1m:test-1 | 33.84% | 19.87% | 43.72% | 27.32% | 59.9% | 28.83 | 1650s | 4776s |
| v0001 | 1m:test-2 | 48.52% | 35.47% | 63.51% | 45.52% | 60.53% | 71.67 | 17m | 71m |
| v0001 | 5m:test-1 | 26.54% | 19.83% | 20.84% | 20.32% | 51.39% | 10.17 | 25m | 4140000ms |
| v0001 | 5m:test-2 | 36.74% | 26.25% | 40.48% | 31.85% | 56.34% | 31.98 | 25m | 80m |
| v0001 | 15m:test-1 | 25.72% | 38.21% | 13.12% | 19.54% | 50.43% | 2 | 15m | 2970000ms |
| v0001 | 15m:test-2 | 22.56% | 22.49% | 11.84% | 15.51% | 50.74% | 5.64 | 30m | 90m |
| v0001 | 1h:test-1 | 10% | 0% | 0% | 0% | 50% | 0 | - | - |
| v0001 | 1h:test-2 | 10.09% | 0.6% | 0.04% | 0.08% | 50.1% | 0.31 | 1h | 2h |
| k0001 | 1s:test-1 | 41.77% | 28.19% | 60.18% | 38.4% | 55.24% | 39.33 | 1271s | 4394s |
| k0001 | 1s:test-2 | 57.95% | 46.5% | 74.21% | 57.17% | 61.04% | 106.57 | 650s | 3364800ms |
| k0001 | 5s:test-1 | 43.65% | 29.69% | 56.72% | 38.98% | 62.33% | 33.33 | 1132500ms | 3830s |
| k0001 | 5s:test-2 | 55.67% | 43.63% | 72.33% | 54.43% | 60.61% | 102.52 | 760s | 3535s |
| k0001 | 15s:test-1 | 43.02% | 26.43% | 56.74% | 36.07% | 70.83% | 33.33 | 1290s | 4623000ms |
| k0001 | 15s:test-2 | 53.32% | 41.18% | 69.35% | 51.67% | 59.92% | 91.1 | 14m | 3675s |
| k0001 | 1m:test-1 | 33.84% | 19.87% | 43.72% | 27.32% | 59.9% | 28.83 | 1650s | 4776s |
| k0001 | 1m:test-2 | 48.52% | 35.47% | 63.51% | 45.52% | 60.53% | 71.67 | 17m | 71m |
| k0001 | 5m:test-1 | 26.54% | 19.83% | 20.84% | 20.32% | 51.39% | 10.17 | 25m | 4140000ms |
| k0001 | 5m:test-2 | 36.74% | 26.25% | 40.48% | 31.85% | 56.34% | 31.98 | 25m | 80m |
| k0001 | 15m:test-1 | 25.72% | 38.21% | 13.12% | 19.54% | 50.43% | 2 | 15m | 2970000ms |
| k0001 | 15m:test-2 | 22.56% | 22.49% | 11.84% | 15.51% | 50.74% | 5.64 | 30m | 90m |
| k0001 | 1h:test-1 | 10% | 0% | 0% | 0% | 50% | 0 | - | - |
| k0001 | 1h:test-2 | 10.09% | 0.6% | 0.04% | 0.08% | 50.1% | 0.31 | 1h | 2h |
