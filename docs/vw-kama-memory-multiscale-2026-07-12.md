# Volume-weighted KAMA oracle approximation search

Generated: 2026-07-11T23:36:30.506Z
Raw results: `data/benchmarks/vw-kama-memory-multiscale-2026-07-12.jsonl`

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
| volume | v0010 | 25.71% | 34.66% | 16.76% | 22.05% | 40.9% | 28.63% | 58.8% | 25.39 | 25m |
| canonical | k0010 | 26.58% | 36.42% | 16.75% | 23.47% | 43.62% | 30.51% | 59.9% | 26.69 | 25m |

## Validation finalists

| family | id | objective | median | P10 | precision | recall | F1 | agreement | signals/day | lag P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| volume | v0010 | 37.74% | 52.51% | 22.97% | 42.04% | 61.49% | 50% | 61.47% | 73.06 | 705s |
| canonical | k0010 | 37.49% | 52.29% | 22.68% | 41.98% | 61.89% | 50.03% | 62.23% | 73.71 | 12m |
| volume | v0046 | 36.79% | 55.47% | 18.11% | 44.72% | 69.63% | 54.46% | 59.17% | 101.13 | 12m |
| volume | v0015 | 36.51% | 53.85% | 19.16% | 43.79% | 67.52% | 53.13% | 59.31% | 93.49 | 12m |
| volume | v0014 | 36.5% | 54.04% | 18.96% | 43.58% | 65.16% | 52.23% | 62.17% | 71.59 | 690s |
| canonical | k0048 | 36.32% | 54.96% | 17.67% | 43.95% | 71.24% | 54.36% | 59.62% | 95.43 | 12m |
| canonical | k0040 | 36.21% | 54.9% | 17.52% | 44.52% | 67.17% | 53.83% | 61.77% | 72.9 | 690s |
| volume | v0021 | 36.12% | 54.54% | 17.69% | 43.87% | 70.03% | 53.94% | 61.08% | 89.45 | 12m |
| canonical | k0019 | 36.04% | 54.35% | 17.74% | 43.4% | 70.83% | 53.82% | 55.68% | 88.11 | 750s |
| canonical | k0021 | 35.99% | 54.26% | 17.73% | 43.89% | 71.11% | 54.28% | 58.53% | 89.7 | 750s |
| canonical | k0041 | 35.65% | 53.61% | 17.7% | 43.08% | 69.56% | 53.2% | 55.27% | 86.33 | 13m |
| volume | v0018 | 35.15% | 52.33% | 17.98% | 44.31% | 60.23% | 51.06% | 57.19% | 76.19 | 690s |

## Best fit candidates

| family | id | objective | median | P10 | precision | recall | F1 | agreement | signals/day | lag P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| volume | v0046 | 37% | 58.05% | 15.95% | 48.26% | 72.65% | 57.99% | 57.73% | 127.81 | 10m |
| volume | v0015 | 36.8% | 58.13% | 15.48% | 48.57% | 71.86% | 57.97% | 57.47% | 120.65 | 10m |
| canonical | k0010 | 35.87% | 54.95% | 16.79% | 46.11% | 62.08% | 53.16% | 58.81% | 78.33 | 11m |
| volume | v0021 | 35.87% | 58.66% | 13.08% | 48.63% | 74.79% | 58.93% | 57.24% | 118.62 | 10m |
| canonical | k0021 | 35.74% | 58.41% | 13.08% | 48.2% | 75.39% | 58.8% | 56.34% | 105.95 | 11m |
| volume | v0014 | 35.71% | 57.53% | 13.89% | 48.26% | 68.48% | 56.62% | 59.13% | 79.36 | 10m |
| volume | v0018 | 35.71% | 55.07% | 16.34% | 48.65% | 62.65% | 54.77% | 56.75% | 94.49 | 10m |
| volume | v0010 | 35.65% | 54.52% | 16.79% | 45.75% | 62.59% | 52.86% | 58.86% | 80.48 | 11m |
| canonical | k0040 | 35.65% | 58.09% | 13.21% | 48.33% | 70.8% | 57.44% | 59.37% | 79.48 | 10m |
| volume | v0048 | 35.63% | 58.92% | 12.34% | 48.25% | 75.55% | 58.89% | 57.93% | 111.28 | 10m |
| volume | v0040 | 35.63% | 58.04% | 13.21% | 48.28% | 70.75% | 57.4% | 59.59% | 79.85 | 10m |
| volume | v0007 | 35.51% | 58.79% | 12.23% | 49.34% | 73.1% | 58.91% | 57.65% | 94.26 | 10m |

## Finalist parameters

| family | id | efficiency | fast | slow | power | volume | cap | volume power | deadband | mode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| volume | v0010 | 363717ms | 1559054ms | 3070088ms | 0.876 | 5935244ms | 1.089 | 2.462 | 59.206 bps/hour | flat |
| canonical | k0010 | 363717ms | 1559054ms | 3070088ms | 0.876 | 5935244ms | 1.089 | 0 | 59.206 bps/hour | flat |

## Holdout cases

| candidate | scale/window | score | precision | recall | F1 | agreement | signals/day | lag P50 | lag P90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v0010 | 1s:test-1 | 39.73% | 25.62% | 54.73% | 34.9% | 59.04% | 30.2 | 1270500ms | 4837800ms |
| v0010 | 1s:test-2 | 51.31% | 40.45% | 61.69% | 48.86% | 61.11% | 70.93 | 789s | 3726600ms |
| v0010 | 5s:test-1 | 38.4% | 25.36% | 47.69% | 33.11% | 59.57% | 26.12 | 1340s | 4543s |
| v0010 | 5s:test-2 | 50.11% | 38.79% | 60.53% | 47.28% | 61.42% | 69.3 | 850s | 3790s |
| v0010 | 15s:test-1 | 34.34% | 21.29% | 42.01% | 28.26% | 58.68% | 24.65 | 1665s | 4909500ms |
| v0010 | 15s:test-2 | 47.36% | 36.17% | 56.96% | 44.25% | 59.83% | 63.31 | 16m | 3915s |
| v0010 | 1m:test-1 | 28.51% | 16.63% | 31.17% | 21.68% | 55.82% | 20.08 | 29m | 4884s |
| v0010 | 1m:test-2 | 44.37% | 32.61% | 53.88% | 40.63% | 59.33% | 53.14 | 18m | 72m |
| v0010 | 5m:test-1 | 26.71% | 17.94% | 24.74% | 20.8% | 50.37% | 12.73 | 25m | 5760000ms |
| v0010 | 5m:test-2 | 34.99% | 22.82% | 39.78% | 29% | 58.93% | 34.46 | 25m | 85m |
| v0010 | 15m:test-1 | 19.33% | 10.96% | 11.12% | 11.04% | 52.47% | 5.22 | 30m | 105m |
| v0010 | 15m:test-2 | 26.29% | 15.08% | 24.75% | 18.74% | 56.47% | 18.7 | 30m | 105m |
| v0010 | 1h:test-1 | 10.61% | 0.7% | 0.39% | 0.5% | 51.02% | 1.47 | 1h | 5760000ms |
| v0010 | 1h:test-2 | 15.66% | 7.28% | 6.19% | 6.69% | 51.53% | 4.31 | 1h | 2h |
| k0010 | 1s:test-1 | 40.14% | 24.53% | 50.6% | 33.04% | 68.54% | 30.2 | 1326s | 4820400ms |
| k0010 | 1s:test-2 | 51.47% | 40.59% | 62.33% | 49.17% | 60.68% | 71.74 | 788500ms | 3683200ms |
| k0010 | 5s:test-1 | 39.83% | 27.49% | 47.22% | 34.75% | 60.15% | 26.12 | 1180s | 4383s |
| k0010 | 5s:test-2 | 49.76% | 38.88% | 59.58% | 47.05% | 60.58% | 67.2 | 855s | 3820s |
| k0010 | 15s:test-1 | 37.03% | 23.73% | 45.59% | 31.22% | 60.28% | 27.27 | 1582500ms | 4777500ms |
| k0010 | 15s:test-2 | 47.41% | 36.3% | 56.7% | 44.26% | 59.97% | 62.44 | 16m | 3990s |
| k0010 | 1m:test-1 | 28.91% | 17.11% | 31% | 22.05% | 56.37% | 19.76 | 29m | 82m |
| k0010 | 1m:test-2 | 44.78% | 32.6% | 54.24% | 40.73% | 60.98% | 54.41 | 18m | 72m |
| k0010 | 5m:test-1 | 27.75% | 18.11% | 27.92% | 21.97% | 50.85% | 14.37 | 25m | 5340000ms |
| k0010 | 5m:test-2 | 35.81% | 23.21% | 41.65% | 29.81% | 59.82% | 36.09 | 25m | 85m |
| k0010 | 15m:test-1 | 19.3% | 10.68% | 11.12% | 10.9% | 52.89% | 5.55 | 30m | 105m |
| k0010 | 15m:test-2 | 26.07% | 14.61% | 24.81% | 18.39% | 56.79% | 19.74 | 30m | 105m |
| k0010 | 1h:test-1 | 10.61% | 0.7% | 0.39% | 0.5% | 51.02% | 1.47 | 1h | 5760000ms |
| k0010 | 1h:test-2 | 15.66% | 7.28% | 6.19% | 6.69% | 51.53% | 4.31 | 1h | 2h |
