# Volume-weighted KAMA oracle approximation search

Generated: 2026-07-11T23:14:05.114Z
Raw results: `data/benchmarks/vw-kama-memory-broad-2026-07-12.jsonl`

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
| volume | v0001 | 51.82% | 51.98% | 51.67% | 40.25% | 66.58% | 50.17% | 59.24% | 77.38 | 14m |
| canonical | k0072 | 52.43% | 52.62% | 52.25% | 44% | 62.92% | 51.78% | 55.98% | 66.96 | 12m |

## Validation finalists

| family | id | objective | median | P10 | precision | recall | F1 | agreement | signals/day | lag P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical | k0072 | 54.95% | 55.13% | 54.78% | 46.68% | 66.34% | 54.8% | 56.44% | 86.14 | 11m |
| volume | v0001 | 54.92% | 55.1% | 54.74% | 43.77% | 69.9% | 53.83% | 60.17% | 96.69 | 750s |
| canonical | k0001 | 54.77% | 55% | 54.55% | 43.89% | 69.37% | 53.76% | 59.94% | 95.2 | 750s |
| volume | v0072 | 54.75% | 54.9% | 54.61% | 46.18% | 66.23% | 54.41% | 56.86% | 86.12 | 11m |
| volume | v0093 | 54.59% | 54.67% | 54.5% | 42.48% | 70.7% | 53.07% | 61.07% | 95.14 | 13m |
| volume | v0097 | 54.34% | 54.4% | 54.27% | 42.19% | 69.28% | 52.44% | 62.22% | 90.77 | 13m |
| volume | v0168 | 54.33% | 54.43% | 54.22% | 42.28% | 70.5% | 52.86% | 60.71% | 96.29 | 13m |
| canonical | k0093 | 54.28% | 54.44% | 54.12% | 42.28% | 70.3% | 52.81% | 60.96% | 93.94 | 13m |
| canonical | k0097 | 54.27% | 54.36% | 54.17% | 42.31% | 67.76% | 52.09% | 63.44% | 83.74 | 750s |
| volume | v0070 | 54.25% | 54.31% | 54.2% | 42.46% | 68.47% | 52.41% | 61.89% | 90.62 | 750s |
| volume | v0025 | 54.23% | 54.32% | 54.13% | 42.28% | 70.23% | 52.79% | 60.46% | 94.43 | 13m |
| volume | v0145 | 54.21% | 54.28% | 54.14% | 42.93% | 67.99% | 52.63% | 60.9% | 88.43 | 750s |
| volume | v0063 | 54.19% | 54.2% | 54.17% | 42.36% | 69.28% | 52.57% | 60.71% | 95.7 | 750s |
| volume | v0126 | 54.12% | 54.13% | 54.12% | 42.25% | 69% | 52.41% | 61.01% | 94.63 | 13m |
| canonical | k0188 | 54.08% | 54.2% | 53.95% | 42.21% | 68.76% | 52.31% | 61.78% | 89.58 | 750s |
| canonical | k0168 | 54.05% | 54.12% | 53.97% | 42.18% | 70.43% | 52.76% | 59.56% | 93.72 | 810s |
| volume | v0115 | 54.01% | 54.17% | 53.84% | 43.22% | 68.18% | 52.91% | 59.21% | 90.11 | 750s |
| canonical | k0115 | 53.94% | 54.14% | 53.75% | 43.89% | 67.1% | 53.07% | 58.42% | 87.95 | 690s |
| volume | v0064 | 53.94% | 54.03% | 53.85% | 42.21% | 67.38% | 51.9% | 62.55% | 81.38 | 750s |
| canonical | k0137 | 53.94% | 53.94% | 53.93% | 41.91% | 69.45% | 52.27% | 60.63% | 87.99 | 810s |
| canonical | k0042 | 53.93% | 54.04% | 53.83% | 42.97% | 67.47% | 52.5% | 60.19% | 87.44 | 750s |
| volume | v0137 | 53.92% | 54% | 53.84% | 41.94% | 69.88% | 52.42% | 60.31% | 89.51 | 810s |
| canonical | k0025 | 53.92% | 54.02% | 53.82% | 42.11% | 70.06% | 52.6% | 59.69% | 93.22 | 810s |
| canonical | k0182 | 53.89% | 53.89% | 53.89% | 42.06% | 66.53% | 51.54% | 63.33% | 83.47 | 750s |
| volume | v0042 | 53.84% | 53.97% | 53.71% | 42.89% | 67.91% | 52.57% | 59.56% | 90.69 | 750s |
| canonical | k0099 | 53.81% | 53.94% | 53.68% | 47.13% | 60.44% | 52.96% | 57.86% | 57.12 | 11m |
| canonical | k0064 | 53.81% | 53.93% | 53.68% | 42.06% | 67.82% | 51.92% | 61.99% | 80.3 | 13m |
| canonical | k0181 | 53.79% | 53.87% | 53.72% | 41.8% | 69.89% | 52.32% | 60.08% | 87.52 | 810s |
| canonical | k0070 | 53.77% | 53.83% | 53.71% | 42.26% | 66.82% | 51.78% | 62.03% | 85.06 | 750s |
| canonical | k0063 | 53.65% | 53.74% | 53.56% | 41.79% | 67.93% | 51.75% | 61.71% | 89.28 | 750s |
| volume | v0011 | 53.53% | 53.68% | 53.38% | 47.41% | 60.75% | 53.26% | 55.35% | 68.52 | 10m |
| volume | v0190 | 53.12% | 53.29% | 52.95% | 44% | 64.62% | 52.35% | 57.03% | 79.83 | 12m |

## Best fit candidates

| family | id | objective | median | P10 | precision | recall | F1 | agreement | signals/day | lag P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical | k0072 | 53.37% | 56.22% | 50.53% | 47.71% | 68.26% | 56.16% | 56.45% | 98.15 | 10m |
| volume | v0072 | 53.37% | 56.26% | 50.48% | 47.62% | 68.38% | 56.14% | 56.73% | 97.61 | 10m |
| canonical | k0001 | 53.22% | 56.24% | 50.2% | 45.15% | 71.35% | 55.29% | 59.87% | 105.17 | 690s |
| volume | v0001 | 53.1% | 56.06% | 50.14% | 44.99% | 71.36% | 55.17% | 59.62% | 106.28 | 690s |
| canonical | k0093 | 52.72% | 55.59% | 49.85% | 44.03% | 71.83% | 54.56% | 60.46% | 101.02 | 750s |
| volume | v0097 | 52.69% | 55.68% | 49.7% | 43.99% | 70.7% | 54.2% | 61.5% | 97.29 | 750s |
| volume | v0093 | 52.65% | 55.57% | 49.73% | 43.92% | 71.99% | 54.52% | 60.49% | 101.49 | 750s |
| volume | v0025 | 52.52% | 55.46% | 49.58% | 43.85% | 71.64% | 54.37% | 59.75% | 101.41 | 750s |
| volume | v0063 | 52.51% | 55.32% | 49.7% | 43.87% | 70.83% | 54.16% | 60.04% | 104.46 | 690s |
| canonical | k0115 | 52.44% | 55.26% | 49.62% | 45.39% | 68.67% | 54.65% | 57.55% | 96.56 | 690s |
| volume | v0115 | 52.44% | 55.31% | 49.57% | 44.84% | 69.45% | 54.48% | 58.34% | 98.37 | 690s |
| volume | v0168 | 52.41% | 55.37% | 49.45% | 43.73% | 71.75% | 54.31% | 59.94% | 103.06 | 750s |

## Finalist parameters

| family | id | efficiency | fast | slow | power | volume | cap | volume power | deadband | mode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| volume | v0001 | 591798ms | 4312ms | 23446123ms | 0.319 | 1517182ms | 5.594 | 1.589 | 310.966 bps/hour | flat |
| canonical | k0072 | 147219ms | 2136ms | 8964581ms | 1.426 | 132865ms | 4.049 | 0 | 565.617 bps/hour | flat |

## Holdout cases

| candidate | scale/window | score | precision | recall | F1 | agreement | signals/day | lag P50 | lag P90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v0001 | 1m:test-1 | 51.59% | 39.71% | 66.02% | 49.59% | 59.57% | 73.92 | 14m | 62m |
| v0001 | 1m:test-2 | 52.37% | 40.78% | 67.13% | 50.74% | 58.91% | 80.84 | 14m | 59m |
| k0072 | 1m:test-1 | 52.15% | 43.73% | 61.93% | 51.26% | 55.7% | 63.08 | 12m | 55m |
| k0072 | 1m:test-2 | 53.09% | 44.26% | 63.91% | 52.3% | 56.25% | 70.83 | 12m | 53m |
