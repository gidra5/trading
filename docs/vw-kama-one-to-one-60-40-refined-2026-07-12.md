# Volume-weighted KAMA oracle approximation search

Generated: 2026-07-12T11:24:58.253Z
Raw results: `data/benchmarks/vw-kama-one-to-one-60-40-refined-2026-07-12.jsonl`

The fit ranking never reads validation or holdout scores. The best validated volume-aware and canonical (`volumePower=0`) candidates are the only candidates evaluated on holdout.

## Data and objective

- Source: `data/historical/spot-btcusdt/btcusdt/1s` (1s); target scales: 1s, 5s, 15s, 1m, 5m, 15m, 1h.
- Windows: fit 2021-09-08..2021-12-20, 2022-05-14..2022-08-03, 2023-03-03..2023-12-05; validation 2024-01-02..2024-02-26, 2024-07-07..2024-11-11; test 2025-02-14..2025-07-06, 2026-04-22..2026-07-10.
- Each continuous segment reserves 3d before scoring.
- Signal: completed-candle volume-weighted KAMA derivative rate; candidates either go flat inside the deadband or hold their prior exposure until the opposite threshold.
- Signal memory: after the first signal, a candidate state change emits only when the current close is strictly more than 17.5 bps from the last emitted signal price; rejected changes retain the prior state. This is the same friction used by the oracle.
- Matching is one chronological one-to-one alignment by resulting state. It maximizes total timing credit, so extra candidate transitions reduce precision and uncovered oracle transitions reduce recall.
- Case score weights timing-credited transition F1 60% and graded oracle-state agreement 40% (exact 1, flat/directional 0.5, opposite 0). Trading returns and execution prices are neither computed nor ranked.
- Candidate objective equally weights the median and P10 case score; every scale/window case has equal weight.

## Holdout finalists

| family | id | objective | median | P10 | precision | recall | F1 | agreement | signals/day | timing error P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| volume | v0016 | 30.88% | 35.76% | 26% | 17.78% | 39.91% | 25.8% | 57.65% | 22.4 | 828750ms |
| canonical | k0016 | 31.57% | 36.91% | 26.22% | 18.63% | 40.8% | 25.97% | 58.22% | 22.08 | 768750ms |

## Validation finalists

| family | id | objective | median | P10 | precision | recall | F1 | agreement | signals/day | timing error P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| volume | v0016 | 39.95% | 50.65% | 29.26% | 35.67% | 54.12% | 43% | 62.08% | 54.87 | 6m |
| canonical | k0016 | 39.9% | 50.58% | 29.23% | 35.41% | 53.62% | 42.65% | 62.19% | 54.8 | 6m |
| volume | v0185 | 39.71% | 50.02% | 29.39% | 34.18% | 58.15% | 43.28% | 60.94% | 61.15 | 330s |
| volume | v0067 | 39.63% | 50.3% | 28.96% | 33.31% | 58.97% | 42.57% | 60.94% | 64.1 | 330s |
| canonical | k0045 | 39.58% | 50.37% | 28.8% | 34.75% | 59.36% | 43.83% | 60.76% | 61.83 | 330s |
| volume | v0006 | 39.53% | 50.03% | 29.03% | 33.35% | 59.05% | 42.99% | 61.11% | 63.27 | 330s |
| volume | v0126 | 39.53% | 50.24% | 28.81% | 34.76% | 58.6% | 43.63% | 62.33% | 61.08 | 330s |
| volume | v0214 | 39.52% | 49.94% | 29.11% | 32.71% | 67.58% | 44.09% | 58.91% | 73.75 | 4m |
| volume | v0045 | 39.47% | 50.07% | 28.87% | 34.62% | 58.88% | 43.6% | 60.65% | 61.6 | 330s |
| volume | v0222 | 39.45% | 49.93% | 28.96% | 32.58% | 65.58% | 43.52% | 59.69% | 72 | 270s |
| volume | v0091 | 39.44% | 49.83% | 29.05% | 33.37% | 56.17% | 42.52% | 61.31% | 62.67 | 330s |
| canonical | k0039 | 39.4% | 49.53% | 29.27% | 33.5% | 61.87% | 43.46% | 59.83% | 66.88 | 5m |
| volume | v0112 | 39.34% | 49.47% | 29.21% | 33.76% | 57.21% | 42.97% | 61.63% | 60.17 | 330s |
| volume | v0250 | 39.31% | 49.79% | 28.84% | 34.34% | 59.54% | 43.56% | 59.7% | 62.72 | 5m |
| canonical | k0222 | 39.31% | 49.64% | 28.97% | 33.1% | 61.27% | 42.97% | 60.1% | 67.1 | 270s |
| canonical | k0091 | 39.3% | 49.53% | 29.07% | 33.3% | 57.35% | 42.11% | 61.04% | 61.92 | 330s |
| volume | v0089 | 39.25% | 49.81% | 28.68% | 36.45% | 51.45% | 42.69% | 63.25% | 51.18 | 390s |
| volume | v0240 | 39.23% | 49.7% | 28.75% | 34.77% | 54.22% | 42.82% | 61.25% | 56.33 | 6m |
| canonical | k0142 | 39.21% | 49.6% | 28.83% | 34% | 59.39% | 43.24% | 60.11% | 63.32 | 5m |
| canonical | k0112 | 39.21% | 49.69% | 28.74% | 34.1% | 56.88% | 43.41% | 61.19% | 58.65 | 330s |
| canonical | k0008 | 39.15% | 49.5% | 28.81% | 33.55% | 55.67% | 41.95% | 59.36% | 59.97 | 330s |
| canonical | k0172 | 39.12% | 49.24% | 29% | 32.81% | 59.81% | 43.4% | 59.41% | 63.57 | 330s |
| canonical | k0250 | 39.09% | 49.34% | 28.84% | 34.04% | 60.57% | 43.59% | 59.42% | 64.35 | 5m |
| volume | v0008 | 39.02% | 49.26% | 28.79% | 33.9% | 55.55% | 42.23% | 59.16% | 59.2 | 330s |
| canonical | k0049 | 38.96% | 48.64% | 29.28% | 32.8% | 60.18% | 42.44% | 60.46% | 66.62 | 5m |
| canonical | k0006 | 38.93% | 48.67% | 29.19% | 33.78% | 57.6% | 42.97% | 59.57% | 60.93 | 330s |
| volume | v0142 | 38.9% | 48.98% | 28.81% | 32.76% | 59.72% | 42.31% | 59.92% | 65.95 | 5m |
| canonical | k0240 | 38.74% | 48.65% | 28.82% | 34.19% | 54.93% | 42.44% | 61.62% | 57.67 | 345s |
| volume | v0192 | 38.67% | 48.35% | 29% | 32.98% | 61.58% | 43.06% | 59.88% | 65.17 | 311250ms |
| canonical | k0179 | 38.62% | 48.42% | 28.82% | 32.1% | 59.83% | 41.78% | 59.41% | 67.33 | 330s |
| canonical | k0157 | 38.51% | 48.19% | 28.82% | 32.1% | 58.07% | 42.84% | 60% | 61.95 | 5m |
| canonical | k0244 | 37.24% | 45.7% | 28.78% | 30.46% | 60.97% | 41.53% | 59.11% | 72.67 | 262500ms |

## Best fit candidates

| family | id | objective | median | P10 | precision | recall | F1 | agreement | signals/day | timing error P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| volume | v0192 | 38.52% | 52.92% | 24.13% | 36.53% | 64.59% | 47.35% | 59.62% | 74.52 | 4m |
| volume | v0045 | 38.51% | 52.65% | 24.37% | 37.12% | 62.21% | 46.5% | 59.44% | 74.61 | 5m |
| volume | v0142 | 38.43% | 52.49% | 24.37% | 36.6% | 63.87% | 46.54% | 59.54% | 74.49 | 5m |
| volume | v0112 | 38.41% | 52.74% | 24.08% | 36.25% | 60.25% | 46.38% | 59.21% | 76.4 | 5m |
| canonical | k0045 | 38.33% | 52.3% | 24.37% | 36.66% | 62.05% | 46.09% | 59.32% | 74.67 | 5m |
| volume | v0006 | 38.32% | 52.79% | 23.84% | 37.09% | 63.6% | 46.99% | 58.94% | 77.65 | 5m |
| canonical | k0142 | 38.27% | 52.17% | 24.37% | 36.64% | 61.66% | 45.97% | 59.56% | 74.03 | 5m |
| volume | v0091 | 38.21% | 51.95% | 24.47% | 36.83% | 53.08% | 45.67% | 59.78% | 69.86 | 5m |
| canonical | k0157 | 38.21% | 52.27% | 24.14% | 37.06% | 58.01% | 45.68% | 59.87% | 76 | 5m |
| volume | v0214 | 38.2% | 52.56% | 23.84% | 35.38% | 68.86% | 46.74% | 59.24% | 79.97 | 4m |
| volume | v0126 | 38.17% | 53.38% | 22.96% | 38.04% | 59.26% | 46.33% | 59.61% | 71.02 | 5m |
| canonical | k0112 | 38.16% | 52.44% | 23.89% | 36.18% | 58.46% | 45.67% | 59.49% | 74.69 | 5m |

## Finalist parameters

| family | id | efficiency | fast | slow | power | volume | cap | volume power | deadband | mode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| volume | v0016 | 1856036ms | 1873481ms | 3701806ms | 0.614 | 440028ms | 1.973 | 1.727 | 0.444 bps/hour | hold |
| canonical | k0016 | 1856036ms | 1873481ms | 3701806ms | 0.614 | 440028ms | 1.973 | 0 | 0.444 bps/hour | hold |

## Holdout cases

| candidate | scale/window | score | precision | recall | F1 | agreement | signals/day | timing error P50 | timing error P90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v0016 | 1s:test-1 | 34.94% | 16.34% | 53.58% | 25.05% | 49.77% | 29.5 | 385500ms | 3013600ms |
| v0016 | 1s:test-2 | 47.2% | 30.03% | 67.21% | 41.51% | 55.74% | 69.9 | 218s | 1320400ms |
| v0016 | 5s:test-1 | 36.58% | 19.04% | 51.84% | 27.85% | 49.68% | 24.5 | 412500ms | 2719500ms |
| v0016 | 5s:test-2 | 47.8% | 30.41% | 61.84% | 40.77% | 58.35% | 59.17 | 285s | 1597000ms |
| v0016 | 15s:test-1 | 39.96% | 18.93% | 44.42% | 26.55% | 60.08% | 20.33 | 757500ms | 2908500ms |
| v0016 | 15s:test-2 | 46.63% | 29.56% | 55.73% | 38.63% | 58.63% | 50.76 | 367500ms | 1942500ms |
| v0016 | 1m:test-1 | 32.39% | 15.21% | 32.01% | 20.62% | 50.05% | 16.83 | 23m | 4356000ms |
| v0016 | 1m:test-2 | 46.53% | 28.2% | 50.93% | 36.3% | 61.87% | 40.12 | 450s | 32m |
| v0016 | 5m:test-1 | 33.88% | 16.63% | 30.17% | 21.44% | 52.55% | 13 | 15m | 55m |
| v0016 | 5m:test-2 | 40.98% | 21.65% | 35.39% | 26.87% | 62.16% | 24.48 | 15m | 50m |
| v0016 | 15m:test-1 | 33.69% | 8.91% | 18.11% | 11.94% | 66.32% | 10.5 | 30m | 75m |
| v0016 | 15m:test-2 | 33.19% | 11.78% | 19.89% | 14.8% | 60.79% | 16.76 | 15m | 75m |
| v0016 | 1h:test-1 | 20.84% | 0.36% | 0.69% | 0.47% | 51.39% | 5.17 | 1h | 2h |
| v0016 | 1h:test-2 | 23.27% | 0.68% | 1.02% | 0.81% | 56.94% | 7.55 | 1h | 2h |
| k0016 | 1s:test-1 | 35.55% | 16.79% | 55.04% | 25.73% | 50.28% | 29.5 | 5m | 3039s |
| k0016 | 1s:test-2 | 46.96% | 29.86% | 67.05% | 41.32% | 55.43% | 70.14 | 224s | 1303800ms |
| k0016 | 5s:test-1 | 35.74% | 18.78% | 49.73% | 27.26% | 48.46% | 23.83 | 475s | 2790s |
| k0016 | 5s:test-2 | 47.74% | 30.32% | 61.86% | 40.7% | 58.3% | 59.36 | 275s | 1603500ms |
| k0016 | 15s:test-1 | 39.76% | 18.87% | 44.26% | 26.46% | 59.73% | 20.33 | 757500ms | 2925000ms |
| k0016 | 15s:test-2 | 46.63% | 29.53% | 55.76% | 38.61% | 58.66% | 50.86 | 375s | 1875s |
| k0016 | 1m:test-1 | 38.09% | 18.47% | 37.33% | 24.72% | 58.15% | 16.17 | 13m | 4452000ms |
| k0016 | 1m:test-2 | 46.47% | 28.01% | 50.95% | 36.15% | 61.96% | 40.4 | 8m | 31m |
| k0016 | 5m:test-1 | 33.44% | 16.18% | 29.35% | 20.86% | 52.31% | 13 | 15m | 55m |
| k0016 | 5m:test-2 | 40.3% | 21.12% | 34.52% | 26.21% | 61.43% | 24.48 | 15m | 50m |
| k0016 | 15m:test-1 | 33.42% | 9.45% | 18.59% | 12.52% | 64.76% | 10.17 | 30m | 75m |
| k0016 | 15m:test-2 | 33.11% | 11.74% | 19.82% | 14.74% | 60.66% | 16.76 | 15m | 1h |
| k0016 | 1h:test-1 | 20.84% | 0.36% | 0.69% | 0.47% | 51.39% | 5.17 | 1h | 2h |
| k0016 | 1h:test-2 | 23.27% | 0.68% | 1.02% | 0.81% | 56.94% | 7.55 | 1h | 2h |
