# Volume-weighted KAMA oracle approximation search

Generated: 2026-07-11T23:44:25.880Z
Raw results: `data/benchmarks/vw-kama-memory-multiscale-refined-2026-07-12.jsonl`

The fit ranking never reads validation or holdout scores. The best validated volume-aware and canonical (`volumePower=0`) candidates are the only candidates evaluated on holdout.

## Data and objective

- Source: `data/historical/spot-btcusdt/btcusdt/1s` (1s); target scales: 1s, 5s, 15s, 1m, 5m, 15m, 1h.
- Windows: fit 2021-09-08..2021-12-20, 2022-05-14..2022-08-03, 2023-03-03..2023-12-05; validation 2024-01-02..2024-02-26, 2024-07-07..2024-11-11; test 2025-02-14..2025-07-06, 2026-04-22..2026-07-10.
- Each continuous segment reserves 72h before scoring.
- Signal: completed-candle volume-weighted KAMA derivative rate; candidates either go flat inside the deadband or hold their prior exposure until the opposite threshold.
- Signal memory: after the first signal, a candidate state change emits only when the current close is strictly more than 17.5 bps from the last emitted signal price; rejected changes retain the prior state. This is the same friction used by the oracle.
- Matching is symmetric nearest-event comparison by direction. It permits multiple signals around one oracle event; timing credit halves every configured half-life, while distant chatter and missed events reduce precision and recall.
- Case score weights timing-credited transition F1 80% and graded oracle-state agreement 20% (exact 1, flat/directional 0.5, opposite 0). Trading returns and execution prices are neither computed nor ranked.
- Candidate objective equally weights the median and P10 case score; every scale/window case has equal weight.

## Holdout finalists

| family | id | objective | median | P10 | precision | recall | F1 | agreement | signals/day | lag P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| volume | v0048 | 27.79% | 37.22% | 18.35% | 23.89% | 45.87% | 31.77% | 60.83% | 29.25 | 1267500ms |
| canonical | k0021 | 26.14% | 35.83% | 16.45% | 23.52% | 44.73% | 30.82% | 57.1% | 28.33 | 1442500ms |

## Validation finalists

| family | id | objective | median | P10 | precision | recall | F1 | agreement | signals/day | lag P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical | k0021 | 39.04% | 54.85% | 23.22% | 43.57% | 65.63% | 52.37% | 63.83% | 82.75 | 11m |
| canonical | k0043 | 38.89% | 54.31% | 23.46% | 44.07% | 60.93% | 52.52% | 62.48% | 84 | 11m |
| canonical | k0014 | 38.46% | 55.87% | 21.04% | 44.18% | 68.18% | 53.61% | 62.93% | 89.47 | 11m |
| canonical | k0048 | 38.45% | 54.4% | 22.51% | 43.2% | 63.4% | 51.65% | 64.4% | 78.17 | 11m |
| volume | v0048 | 38.39% | 54.33% | 22.45% | 43.12% | 64.21% | 51.62% | 63.87% | 79.17 | 690s |
| volume | v0035 | 38.32% | 53.2% | 23.44% | 43.31% | 56.53% | 51.53% | 57.94% | 76.08 | 690s |
| volume | v0014 | 38.24% | 55.23% | 21.26% | 43.54% | 68.07% | 53.11% | 63.23% | 90.3 | 690s |
| volume | v0043 | 38.05% | 53.42% | 22.69% | 43.15% | 62.09% | 52.01% | 59.68% | 84.75 | 12m |
| volume | v0016 | 37.44% | 55.61% | 19.27% | 43.88% | 66.46% | 52.86% | 63.85% | 77.88 | 690s |
| canonical | k0016 | 37.37% | 55.47% | 19.27% | 43.9% | 66.1% | 52.76% | 63.93% | 77.98 | 690s |
| volume | v0060 | 36.27% | 55.03% | 17.5% | 44.47% | 67.98% | 53.76% | 58.33% | 93.2 | 11m |
| canonical | k0060 | 36.25% | 55.09% | 17.4% | 44.39% | 67.82% | 53.66% | 59.72% | 93.23 | 690s |
| volume | v0015 | 36.06% | 54.49% | 17.62% | 44.84% | 66.9% | 53.68% | 57.73% | 92.35 | 11m |
| canonical | k0015 | 36.02% | 54.52% | 17.52% | 44.96% | 66.19% | 53.54% | 59.06% | 88.72 | 11m |
| volume | v0008 | 35.92% | 55.65% | 16.19% | 45.5% | 68.77% | 54.75% | 57.39% | 94.85 | 11m |
| canonical | k0008 | 35.42% | 54.96% | 15.87% | 44.44% | 68.26% | 53.83% | 58.4% | 95.75 | 690s |

## Best fit candidates

| family | id | objective | median | P10 | precision | recall | F1 | agreement | signals/day | lag P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| volume | v0016 | 37.69% | 58.07% | 17.3% | 47.93% | 69.66% | 56.79% | 61.25% | 93.03 | 10m |
| canonical | k0016 | 37.47% | 57.65% | 17.3% | 47.61% | 68.99% | 56.34% | 61.13% | 92.82 | 10m |
| canonical | k0060 | 37.46% | 58.84% | 16.08% | 47.89% | 72.88% | 57.8% | 60.41% | 105.97 | 10m |
| volume | v0015 | 37.45% | 58.85% | 16.06% | 48.74% | 71.95% | 58.11% | 58.68% | 106.35 | 9m |
| canonical | k0008 | 37.44% | 58.51% | 16.37% | 47.67% | 72.65% | 57.57% | 60.05% | 108.41 | 10m |
| volume | v0060 | 37.39% | 58.71% | 16.08% | 47.67% | 72.69% | 57.58% | 61.01% | 105.97 | 10m |
| volume | v0048 | 37.38% | 57.04% | 17.72% | 47.46% | 64.74% | 55.52% | 60.07% | 83.38 | 10m |
| volume | v0008 | 37.27% | 59% | 15.54% | 48.24% | 73.48% | 58.24% | 59.57% | 112.29 | 10m |
| canonical | k0015 | 37.22% | 58.39% | 16.06% | 48.23% | 71.21% | 57.51% | 59.66% | 103.8 | 10m |
| canonical | k0048 | 37.19% | 56.66% | 17.72% | 47.23% | 64.22% | 55.12% | 60.59% | 79.44 | 10m |
| volume | v0014 | 36.95% | 59.21% | 14.7% | 48.23% | 72.65% | 57.97% | 61.52% | 104 | 10m |
| canonical | k0043 | 36.94% | 56.97% | 16.9% | 47.52% | 64.68% | 56.52% | 58.9% | 86.54 | 10m |

## Finalist parameters

| family | id | efficiency | fast | slow | power | volume | cap | volume power | deadband | mode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| volume | v0048 | 330355ms | 749200ms | 5043799ms | 0.857 | 5322623ms | 2.617 | 0.563 | 38.742 bps/hour | flat |
| canonical | k0021 | 842179ms | 1678833ms | 9187995ms | 0.49 | 7820651ms | 2.65 | 0 | 67.567 bps/hour | flat |

## Holdout cases

| candidate | scale/window | score | precision | recall | F1 | agreement | signals/day | lag P50 | lag P90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v0048 | 1s:test-1 | 39.76% | 25.07% | 54.75% | 34.39% | 61.22% | 31.33 | 1124s | 4657s |
| v0048 | 1s:test-2 | 51.05% | 40.43% | 60.76% | 48.56% | 61.03% | 67 | 795s | 3755200ms |
| v0048 | 5s:test-1 | 38.33% | 25.87% | 48.66% | 33.78% | 56.55% | 28 | 1125s | 4361500ms |
| v0048 | 5s:test-2 | 50.63% | 39.49% | 60.87% | 47.91% | 61.51% | 68.05 | 830s | 3803000ms |
| v0048 | 15s:test-1 | 42.05% | 26.03% | 55.07% | 35.35% | 68.84% | 30.5 | 1155s | 4713s |
| v0048 | 15s:test-2 | 49.19% | 37.95% | 59.45% | 46.33% | 60.63% | 65.33 | 15m | 3886500ms |
| v0048 | 1m:test-1 | 35.14% | 21.24% | 43.03% | 28.44% | 61.93% | 22.67 | 23m | 78m |
| v0048 | 1m:test-2 | 45.82% | 33.23% | 56.62% | 41.88% | 61.56% | 57.48 | 18m | 73m |
| v0048 | 5m:test-1 | 24.17% | 13.59% | 27.09% | 18.1% | 48.47% | 18.83 | 2250s | 5130000ms |
| v0048 | 5m:test-2 | 36.1% | 22.72% | 43.09% | 29.75% | 61.52% | 37.86 | 30m | 5220000ms |
| v0048 | 15m:test-1 | 20% | 8.41% | 17.47% | 11.35% | 54.6% | 11.67 | 45m | 105m |
| v0048 | 15m:test-2 | 26.87% | 13.97% | 27.62% | 18.56% | 60.13% | 23.48 | 30m | 105m |
| v0048 | 1h:test-1 | 10.36% | 0.59% | 0.49% | 0.54% | 49.65% | 2.67 | 1h | 2h |
| v0048 | 1h:test-2 | 17.65% | 7.51% | 10.25% | 8.67% | 53.57% | 6.95 | 1h | 2h |
| k0021 | 1s:test-1 | 41.62% | 26.41% | 56.22% | 35.94% | 64.35% | 35 | 1303500ms | 4240300ms |
| k0021 | 1s:test-2 | 56.9% | 45.89% | 71.41% | 55.87% | 60.98% | 90.31 | 674s | 3408200ms |
| k0021 | 5s:test-1 | 35.7% | 23.9% | 47.21% | 31.74% | 51.58% | 30.83 | 1385s | 4765s |
| k0021 | 5s:test-2 | 54.08% | 42.27% | 67.65% | 52.03% | 62.3% | 79.17 | 770s | 3535s |
| k0021 | 15s:test-1 | 38.26% | 26.33% | 47.6% | 33.9% | 55.67% | 25.83 | 1147500ms | 4057500ms |
| k0021 | 15s:test-2 | 50.97% | 39.01% | 63.71% | 48.39% | 61.26% | 70.6 | 855s | 65m |
| k0021 | 1m:test-1 | 32.21% | 20.01% | 36.99% | 25.97% | 57.16% | 22.33 | 27m | 80m |
| k0021 | 1m:test-2 | 46.41% | 33.53% | 58.51% | 42.63% | 61.53% | 59.29 | 18m | 74m |
| k0021 | 5m:test-1 | 26.31% | 15.75% | 25.6% | 19.5% | 53.53% | 16.17 | 30m | 80m |
| k0021 | 5m:test-2 | 35.95% | 23.14% | 42.25% | 29.9% | 60.15% | 37.64 | 25m | 85m |
| k0021 | 15m:test-1 | 18.04% | 9.45% | 9.84% | 9.64% | 51.65% | 5.5 | 2250s | 6030000ms |
| k0021 | 15m:test-2 | 27.02% | 15.38% | 26.71% | 19.52% | 57.03% | 20.1 | 30m | 105m |
| k0021 | 1h:test-1 | 10.6% | 0.67% | 0.29% | 0.41% | 51.39% | 1.17 | 1h | 1h |
| k0021 | 1h:test-2 | 15.77% | 8.32% | 5.65% | 6.73% | 51.93% | 3.43 | 1h | 2h |
