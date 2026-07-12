# Volume-weighted KAMA oracle approximation search

Generated: 2026-07-11T23:19:55.477Z
Raw results: `data/benchmarks/vw-kama-memory-refined-1m-2026-07-12.jsonl`

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
| volume | v0132 | 52.04% | 52.22% | 51.86% | 39.72% | 67.11% | 49.9% | 61.5% | 78.57 | 810s |
| canonical | k0214 | 52.27% | 52.44% | 52.1% | 40.08% | 67.67% | 50.34% | 60.86% | 81.42 | 810s |

## Validation finalists

| family | id | objective | median | P10 | precision | recall | F1 | agreement | signals/day | lag P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical | k0214 | 55.48% | 55.63% | 55.33% | 43.72% | 70.96% | 54.11% | 61.73% | 101.23 | 12m |
| canonical | k0222 | 55.46% | 55.63% | 55.3% | 43.75% | 70.91% | 54.11% | 61.68% | 101.16 | 12m |
| canonical | k0114 | 55.41% | 55.54% | 55.28% | 43.43% | 70.96% | 53.88% | 62.2% | 99.74 | 750s |
| volume | v0132 | 55.35% | 55.45% | 55.25% | 43.47% | 70.18% | 53.69% | 62.49% | 97.02 | 12m |
| volume | v0103 | 55.34% | 55.47% | 55.21% | 43.77% | 70.77% | 54.09% | 61.01% | 100.79 | 12m |
| canonical | k0167 | 55.34% | 55.49% | 55.19% | 43.42% | 70.96% | 53.87% | 61.97% | 100 | 750s |
| volume | v0114 | 55.31% | 55.47% | 55.16% | 43.42% | 70.87% | 53.85% | 61.94% | 99.6 | 750s |
| volume | v0098 | 55.29% | 55.37% | 55.22% | 43.15% | 70.55% | 53.55% | 62.66% | 96 | 750s |
| volume | v0214 | 55.27% | 55.44% | 55.09% | 43.72% | 70.69% | 54.03% | 61.07% | 101.39 | 12m |
| canonical | k0132 | 55.26% | 55.36% | 55.15% | 43.38% | 69.99% | 53.56% | 62.55% | 95.74 | 12m |
| canonical | k0013 | 55.14% | 55.25% | 55.03% | 45.69% | 67.95% | 54.64% | 57.71% | 92.04 | 11m |
| volume | v0003 | 55.13% | 55.3% | 54.95% | 43.35% | 70.76% | 53.76% | 61.45% | 101.03 | 12m |
| volume | v0110 | 55.12% | 55.26% | 54.98% | 44.09% | 69.74% | 54.03% | 60.21% | 95.61 | 690s |
| canonical | k0242 | 55.11% | 55.26% | 54.97% | 43.5% | 70.42% | 53.78% | 61.18% | 98.32 | 750s |
| volume | v0237 | 55.09% | 55.26% | 54.92% | 43.09% | 70.72% | 53.55% | 62.08% | 97.63 | 750s |
| canonical | k0110 | 55.08% | 55.22% | 54.94% | 44.11% | 69.57% | 53.99% | 60.17% | 95.64 | 12m |
| canonical | k0103 | 55.07% | 55.19% | 54.95% | 43.35% | 70.65% | 53.73% | 61.03% | 99.63 | 750s |
| canonical | k0156 | 55.05% | 55.2% | 54.9% | 45.84% | 67.58% | 54.63% | 57.5% | 90.73 | 690s |
| volume | v0200 | 55.05% | 55.18% | 54.91% | 44.39% | 69.19% | 54.08% | 59.58% | 95.56 | 690s |
| canonical | k0200 | 55.04% | 55.17% | 54.9% | 44.35% | 69.19% | 54.05% | 59.66% | 96.13 | 690s |
| volume | v0056 | 55.03% | 55.16% | 54.9% | 43.23% | 70.58% | 53.62% | 61.31% | 98.67 | 750s |
| canonical | k0229 | 55.03% | 55.15% | 54.9% | 44.6% | 68.67% | 54.08% | 59.46% | 93.22 | 690s |
| volume | v0183 | 55.02% | 55.16% | 54.88% | 43.22% | 70.57% | 53.61% | 61.37% | 98.8 | 750s |
| canonical | k0237 | 55.02% | 55.19% | 54.85% | 43.06% | 70.7% | 53.52% | 61.87% | 97.71 | 750s |
| volume | v0233 | 55.02% | 55.16% | 54.87% | 44.97% | 68.8% | 54.39% | 58.28% | 93.28 | 690s |
| volume | v0242 | 55% | 55.11% | 54.88% | 43.4% | 70.26% | 53.66% | 60.95% | 98.14 | 750s |
| volume | v0156 | 54.99% | 55.11% | 54.88% | 45.82% | 67.36% | 54.54% | 57.39% | 90.4 | 11m |
| volume | v0252 | 54.99% | 55.17% | 54.82% | 43.08% | 70.71% | 53.54% | 61.68% | 97.51 | 750s |
| canonical | k0181 | 54.99% | 55.15% | 54.83% | 46.36% | 66.77% | 54.72% | 56.84% | 88.51 | 11m |
| volume | v0184 | 54.91% | 55.05% | 54.77% | 46.41% | 66.41% | 54.64% | 56.7% | 87.05 | 11m |
| canonical | k0184 | 54.89% | 55% | 54.77% | 46.14% | 66.77% | 54.57% | 56.73% | 87.87 | 11m |
| canonical | k0149 | 54.87% | 55% | 54.74% | 44.99% | 67.81% | 54.09% | 58.64% | 90.13 | 690s |
| canonical | k0233 | 54.82% | 54.98% | 54.66% | 44.69% | 68.77% | 54.17% | 58.2% | 94.13 | 690s |
| canonical | k0016 | 54.81% | 54.95% | 54.68% | 43.68% | 69.56% | 53.66% | 60.09% | 97.22 | 12m |
| volume | v0046 | 54.78% | 54.82% | 54.74% | 42.86% | 68.85% | 52.83% | 62.79% | 91.35 | 750s |
| volume | v0144 | 54.76% | 54.91% | 54.61% | 44.24% | 69.17% | 53.97% | 58.7% | 95.55 | 12m |
| canonical | k0068 | 54.76% | 54.94% | 54.58% | 45.08% | 67.86% | 54.17% | 58.03% | 92.03 | 690s |
| canonical | k0144 | 54.69% | 54.85% | 54.53% | 44.1% | 69.2% | 53.87% | 58.78% | 95.94 | 12m |
| volume | v0149 | 54.68% | 54.82% | 54.54% | 44.03% | 68.97% | 53.75% | 59.12% | 93.94 | 690s |
| volume | v0229 | 54.55% | 54.72% | 54.38% | 43.93% | 68.68% | 53.58% | 59.26% | 93.26 | 690s |

## Best fit candidates

| family | id | objective | median | P10 | precision | recall | F1 | agreement | signals/day | lag P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| volume | v0214 | 53.72% | 56.75% | 50.7% | 45.33% | 72.36% | 55.72% | 60.6% | 111.79 | 690s |
| canonical | k0114 | 53.64% | 56.56% | 50.72% | 44.93% | 72.31% | 55.4% | 61.62% | 107.37 | 690s |
| volume | v0103 | 53.62% | 56.56% | 50.67% | 45.18% | 72.27% | 55.58% | 60.26% | 110.12 | 690s |
| canonical | k0200 | 53.6% | 56.52% | 50.67% | 45.87% | 71.06% | 55.75% | 59.41% | 105.96 | 690s |
| canonical | k0167 | 53.59% | 56.53% | 50.65% | 44.91% | 72.25% | 55.37% | 61.43% | 107.55 | 690s |
| canonical | k0013 | 53.59% | 56.49% | 50.68% | 46.88% | 69.99% | 56.15% | 57.81% | 103.63 | 11m |
| canonical | k0156 | 53.57% | 56.45% | 50.69% | 46.83% | 69.98% | 56.11% | 57.83% | 102.61 | 11m |
| canonical | k0214 | 53.56% | 56.49% | 50.63% | 45.09% | 72.06% | 55.45% | 60.83% | 109.87 | 690s |
| volume | v0156 | 53.54% | 56.42% | 50.66% | 46.84% | 69.84% | 56.07% | 57.82% | 103.2 | 11m |
| canonical | k0222 | 53.54% | 56.49% | 50.58% | 45.1% | 72.06% | 55.46% | 60.8% | 109.99 | 690s |
| volume | v0003 | 53.53% | 56.41% | 50.64% | 44.99% | 71.86% | 55.32% | 60.7% | 109.6 | 690s |
| volume | v0200 | 53.5% | 56.47% | 50.53% | 45.89% | 71.02% | 55.75% | 59.34% | 105.8 | 690s |

## Finalist parameters

| family | id | efficiency | fast | slow | power | volume | cap | volume power | deadband | mode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| volume | v0132 | 106067ms | 93193ms | 1219297ms | 1.639 | 216931ms | 6.68 | 1.205 | 116.476 bps/hour | flat |
| canonical | k0214 | 124173ms | 3206ms | 6122590ms | 1.604 | 113691ms | 7.851 | 0 | 191.739 bps/hour | flat |

## Holdout cases

| candidate | scale/window | score | precision | recall | F1 | agreement | signals/day | lag P50 | lag P90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v0132 | 1m:test-1 | 51.77% | 39.03% | 66.74% | 49.26% | 61.81% | 75.57 | 14m | 64m |
| v0132 | 1m:test-2 | 52.68% | 40.41% | 67.47% | 50.55% | 61.2% | 81.57 | 13m | 1h |
| k0214 | 1m:test-1 | 52.02% | 39.48% | 67.26% | 49.76% | 61.07% | 78.38 | 14m | 63m |
| k0214 | 1m:test-2 | 52.87% | 40.67% | 68.08% | 50.92% | 60.64% | 84.45 | 13m | 59m |
