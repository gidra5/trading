# Volume-weighted KAMA oracle approximation search

Generated: 2026-07-11T14:41:59.945Z
Raw results: `data/benchmarks/vw-kama-signal-only-2026-07-11.jsonl`

The fit ranking never reads validation or holdout scores. The best validated volume-aware and canonical (`volumePower=0`) candidates are the only candidates evaluated on holdout.

## Data and objective

- Source: `data/historical/spot-btcusdt/btcusdt/1m` (1m); target scales: 1m, 5m, 15m, 1h.
- Windows: fit 2021-08-01..2022-01-31, 2022-02-01..2022-07-31, 2022-08-01..2023-01-31, 2023-02-01..2023-12-31; validation 2024-01-01..2024-06-30, 2024-07-01..2024-12-31; test 2025-01-01..2025-12-31, 2026-01-01..2026-07-10.
- Signal: completed-candle volume-weighted KAMA derivative rate; candidates either go flat inside the deadband or hold their prior exposure until the opposite threshold.
- Matching is symmetric nearest-event comparison by direction. It permits multiple signals around one oracle event; timing credit halves every configured half-life, while distant chatter and missed events reduce precision and recall.
- Case score weights timing-credited transition F1 80% and graded oracle-state agreement 20% (exact 1, flat/directional 0.5, opposite 0). Trading returns and execution prices are neither computed nor ranked.
- Candidate objective equally weights the median and P10 case score; every scale/window case has equal weight.

## Holdout finalists

| family | id | objective | median | P10 | precision | recall | F1 | agreement | signals/day | lag P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| volume | v0037 | 24.01% | 33.14% | 14.88% | 19.16% | 52.75% | 27.69% | 54.96% | 68.15 | 30m |
| canonical | k0037 | 23.96% | 33.04% | 14.88% | 18.82% | 53.87% | 27.49% | 55.29% | 72.46 | 30m |

## Validation finalists

| family | id | objective | median | P10 | precision | recall | F1 | agreement | signals/day | lag P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| volume | v0037 | 25.31% | 35.26% | 15.36% | 21% | 54.48% | 30.08% | 55.97% | 77.42 | 30m |
| volume | v0086 | 25.3% | 33.81% | 16.79% | 18.52% | 57.16% | 27.87% | 57.5% | 98.4 | 30m |
| canonical | k0037 | 25.22% | 35.08% | 15.36% | 20.52% | 55.43% | 29.75% | 56.33% | 82.51 | 30m |
| volume | v0055 | 25.21% | 33.68% | 16.74% | 20.84% | 45.01% | 28.44% | 54.61% | 49.94 | 1650s |
| canonical | k0055 | 25.19% | 33.49% | 16.9% | 21.14% | 42.47% | 28.18% | 54.69% | 45.93 | 1650s |
| volume | v0093 | 25.12% | 31.95% | 18.3% | 15.96% | 56.46% | 24.88% | 60.24% | 109.13 | 40m |
| canonical | k0086 | 25.12% | 33.44% | 16.79% | 18.02% | 57.48% | 27.36% | 57.77% | 102.47 | 35m |
| canonical | k0093 | 25.04% | 31.79% | 18.3% | 15.81% | 56.54% | 24.69% | 60.18% | 110.47 | 40m |
| canonical | k0040 | 25% | 32.89% | 17.11% | 17.22% | 57.83% | 26.47% | 58.56% | 107.66 | 40m |
| canonical | k0094 | 25% | 32.31% | 17.69% | 16.39% | 56.58% | 25.41% | 59.91% | 105.82 | 40m |
| volume | v0094 | 25% | 32.31% | 17.69% | 16.39% | 56.58% | 25.41% | 59.91% | 105.82 | 40m |
| volume | v0040 | 25% | 32.88% | 17.11% | 17.21% | 57.94% | 26.47% | 58.49% | 107.85 | 40m |
| volume | v0017 | 24.89% | 32.04% | 17.75% | 16.19% | 57.97% | 25.27% | 58.57% | 114.4 | 40m |
| canonical | k0041 | 24.82% | 30.88% | 18.75% | 14.94% | 57.38% | 23.68% | 59.72% | 121.8 | 40m |
| canonical | k0077 | 24.8% | 32.11% | 17.48% | 16.4% | 58.44% | 25.53% | 58.43% | 118.52 | 40m |
| canonical | k0017 | 24.7% | 31.65% | 17.75% | 15.83% | 58.27% | 24.85% | 58.32% | 121.19 | 40m |
| volume | v0041 | 24.7% | 30.94% | 18.46% | 14.98% | 56.76% | 23.68% | 59.98% | 118.82 | 40m |
| volume | v0077 | 24.54% | 32.82% | 16.26% | 17.47% | 57.68% | 26.75% | 57.33% | 103.31 | 1950s |
| canonical | k0075 | 24.38% | 30.66% | 18.1% | 17.07% | 33.14% | 24.04% | 56.06% | 44.05 | 2250s |
| volume | v0075 | 24.34% | 31.33% | 17.35% | 17.66% | 44.76% | 25.31% | 54.13% | 58.55 | 2250s |

## Best fit candidates

| family | id | objective | median | P10 | precision | recall | F1 | agreement | signals/day | lag P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical | k0055 | 25.15% | 32.78% | 17.53% | 20.56% | 42.55% | 27.48% | 54% | 40.83 | 30m |
| volume | v0055 | 25.09% | 32.85% | 17.32% | 20.16% | 45.18% | 27.56% | 54.02% | 44.41 | 30m |
| volume | v0037 | 25.03% | 34.35% | 15.72% | 20.35% | 55.68% | 29.11% | 55.59% | 70.9 | 30m |
| canonical | k0037 | 24.96% | 34.19% | 15.72% | 20.13% | 56.4% | 28.85% | 55.9% | 75.09 | 30m |
| canonical | k0094 | 24.93% | 31.2% | 18.66% | 16.71% | 56.74% | 24.31% | 59.05% | 100.74 | 2070s |
| volume | v0094 | 24.93% | 31.2% | 18.66% | 16.71% | 56.74% | 24.31% | 59.05% | 100.74 | 2070s |
| canonical | k0040 | 24.92% | 32.08% | 17.76% | 17.81% | 58.36% | 25.54% | 57.83% | 102.81 | 2070s |
| volume | v0040 | 24.92% | 32.08% | 17.76% | 17.77% | 58.46% | 25.54% | 57.63% | 103.09 | 2070s |
| volume | v0093 | 24.91% | 30.97% | 18.85% | 16.45% | 56.82% | 23.8% | 59.26% | 105.99 | 2010s |
| canonical | k0017 | 24.9% | 31.25% | 18.54% | 16.23% | 58.4% | 24.27% | 56.68% | 120.36 | 2250s |
| volume | v0017 | 24.9% | 31.25% | 18.54% | 16.53% | 58.09% | 24.27% | 56.85% | 111.86 | 34m |
| canonical | k0093 | 24.9% | 30.94% | 18.85% | 16.38% | 56.84% | 23.77% | 59.24% | 107.59 | 2010s |

## Finalist parameters

| family | id | efficiency | fast | slow | power | volume | cap | volume power | deadband | mode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| volume | v0037 | 886696ms | 63566ms | 1905088ms | 0.557 | 8044750ms | 5.977 | 1.702 | 92.373 bps/hour | flat |
| canonical | k0037 | 886696ms | 63566ms | 1905088ms | 0.557 | 8044750ms | 5.977 | 0 | 92.373 bps/hour | flat |

## Holdout cases

| candidate | scale/window | score | precision | recall | F1 | agreement | signals/day | lag P50 | lag P90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v0037 | 1m:test-1 | 40.8% | 22.95% | 92.7% | 36.8% | 56.8% | 567.44 | 28m | 85m |
| v0037 | 1m:test-2 | 42.91% | 25.05% | 92.65% | 39.44% | 56.82% | 566.51 | 26m | 81m |
| v0037 | 5m:test-1 | 36.83% | 20.69% | 70.22% | 31.96% | 56.29% | 107.88 | 30m | 90m |
| v0037 | 5m:test-2 | 38.26% | 22.13% | 70.51% | 33.69% | 56.54% | 109.9 | 30m | 90m |
| v0037 | 15m:test-1 | 29.38% | 17.63% | 34.53% | 23.34% | 53.52% | 25.55 | 30m | 90m |
| v0037 | 15m:test-2 | 29.46% | 17.53% | 35.27% | 23.42% | 53.63% | 28.42 | 30m | 90m |
| v0037 | 1h:test-1 | 14.32% | 8.46% | 3.86% | 5.3% | 50.37% | 2.21 | 1h | 2h |
| v0037 | 1h:test-2 | 15.13% | 9.86% | 4.66% | 6.33% | 50.34% | 2.48 | 1h | 2h |
| k0037 | 1m:test-1 | 40.55% | 22.51% | 92.33% | 36.2% | 57.93% | 584.57 | 29m | 85m |
| k0037 | 1m:test-2 | 42.41% | 24.35% | 92.28% | 38.54% | 57.91% | 596.53 | 27m | 81m |
| k0037 | 5m:test-1 | 36.48% | 20.19% | 70.86% | 31.43% | 56.67% | 114.25 | 35m | 90m |
| k0037 | 5m:test-2 | 37.53% | 21.21% | 71.1% | 32.68% | 56.93% | 119.26 | 30m | 90m |
| k0037 | 15m:test-1 | 29.6% | 17.44% | 36.29% | 23.56% | 53.74% | 27.61 | 30m | 90m |
| k0037 | 15m:test-2 | 29.53% | 17.18% | 36.87% | 23.43% | 53.91% | 30.67 | 30m | 90m |
| k0037 | 1h:test-1 | 14.32% | 8.46% | 3.86% | 5.3% | 50.37% | 2.21 | 1h | 2h |
| k0037 | 1h:test-2 | 15.13% | 9.86% | 4.66% | 6.33% | 50.34% | 2.48 | 1h | 2h |
