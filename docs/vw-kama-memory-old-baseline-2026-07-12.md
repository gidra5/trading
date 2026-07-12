# Volume-weighted KAMA oracle approximation search

Generated: 2026-07-11T23:14:37.283Z
Raw results: `data/benchmarks/vw-kama-memory-old-baseline-2026-07-12.jsonl`

The fit ranking never reads validation or holdout scores. The best validated volume-aware and canonical (`volumePower=0`) candidates are the only candidates evaluated on holdout.

## Data and objective

- Source: `data/historical/spot-btcusdt/btcusdt/1m` (1m); target scales: 1m.
- Windows: fit 2021-08-01..2022-01-31, 2022-02-01..2022-07-31, 2022-08-01..2023-01-31, 2023-02-01..2023-12-31; validation 2024-01-01..2024-06-30, 2024-07-01..2024-12-31; test 2025-01-01..2025-12-31, 2026-01-01..2026-07-10.
- Signal: completed-candle volume-weighted KAMA derivative rate; candidates either go flat inside the deadband or hold their prior exposure until the opposite threshold.
- Signal memory: after the first signal, a candidate state change emits only when the current close is strictly more than 17.5 bps from the last emitted signal price; rejected changes retain the prior state. This is the same friction used by the oracle.
- Matching is symmetric nearest-event comparison by direction. It permits multiple signals around one oracle event; timing credit halves every configured half-life, while distant chatter and missed events reduce precision and recall.
- Case score weights timing-credited transition F1 80% and graded oracle-state agreement 20% (exact 1, flat/directional 0.5, opposite 0). Trading returns and execution prices are neither computed nor ranked.
- Candidate objective equally weights the median and P10 case score; every scale/window case has equal weight.

## Holdout finalists

| family | id | objective | median | P10 | precision | recall | F1 | agreement | signals/day | lag P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| volume | v0001 | 50.73% | 50.87% | 50.59% | 45.26% | 53.53% | 49.05% | 58.16% | 39.85 | 11m |
| canonical | k0001 | 50.93% | 50.97% | 50.89% | 45.48% | 53.43% | 49.13% | 58.29% | 39.16 | 11m |

## Validation finalists

| family | id | objective | median | P10 | precision | recall | F1 | agreement | signals/day | lag P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical | k0001 | 53.26% | 53.34% | 53.18% | 47.92% | 56.87% | 52.01% | 58.62% | 50.7 | 630s |
| volume | v0001 | 53.18% | 53.26% | 53.09% | 47.95% | 56.95% | 52.07% | 58.06% | 51.42 | 630s |

## Best fit candidates

| family | id | objective | median | P10 | precision | recall | F1 | agreement | signals/day | lag P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| volume | v0001 | 51.95% | 54.83% | 49.06% | 49.02% | 59.76% | 53.86% | 58.57% | 59.18 | 10m |
| canonical | k0001 | 51.63% | 54.64% | 48.63% | 48.99% | 59.48% | 53.72% | 58.6% | 58.56 | 10m |

## Finalist parameters

| family | id | efficiency | fast | slow | power | volume | cap | volume power | deadband | mode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| volume | v0001 | 360000ms | 60000ms | 300000ms | 0.405 | 35400000ms | 3.201 | 2.362 | 398.722 bps/hour | hold |
| canonical | k0001 | 360000ms | 60000ms | 300000ms | 0.405 | 35400000ms | 3.201 | 0 | 398.722 bps/hour | hold |

## Holdout cases

| candidate | scale/window | score | precision | recall | F1 | agreement | signals/day | lag P50 | lag P90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v0001 | 1m:test-1 | 50.52% | 45.04% | 52.96% | 48.68% | 57.89% | 37.6 | 11m | 55m |
| v0001 | 1m:test-2 | 51.22% | 45.48% | 54.1% | 49.42% | 58.43% | 42.1 | 11m | 52m |
| k0001 | 1m:test-1 | 50.86% | 45.45% | 53.03% | 48.95% | 58.54% | 36.85 | 11m | 54m |
| k0001 | 1m:test-2 | 51.07% | 45.52% | 53.82% | 49.32% | 58.05% | 41.46 | 11m | 53m |
