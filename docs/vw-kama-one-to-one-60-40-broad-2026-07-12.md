# Volume-weighted KAMA oracle approximation search

Generated: 2026-07-12T10:56:33.101Z
Raw results: `data/benchmarks/vw-kama-one-to-one-60-40-broad-2026-07-12.jsonl`

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
| volume | v0153 | 32.34% | 37.87% | 26.81% | 18.89% | 47.86% | 27.03% | 57.77% | 25.71 | 532500ms |
| canonical | k0213 | 32.22% | 38.45% | 25.99% | 17.34% | 45.79% | 25.83% | 57.03% | 25.25 | 543750ms |

## Validation finalists

| family | id | objective | median | P10 | precision | recall | F1 | agreement | signals/day | timing error P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical | k0213 | 39.59% | 51.46% | 27.73% | 33.87% | 59.12% | 43.06% | 60.06% | 63.22 | 5m |
| canonical | k0153 | 39.5% | 49.77% | 29.23% | 34.67% | 56.75% | 43.03% | 60.32% | 59.42 | 330s |
| volume | v0153 | 39.5% | 49.77% | 29.23% | 34.67% | 56.75% | 43.03% | 60.32% | 59.42 | 330s |
| canonical | k0240 | 39.42% | 49.85% | 28.99% | 34.91% | 54.77% | 42.64% | 61.44% | 55.58 | 6m |
| canonical | k0145 | 39.21% | 49.58% | 28.84% | 33.19% | 65.45% | 44.41% | 58.48% | 70.57 | 270s |
| canonical | k0023 | 39.16% | 49.54% | 28.78% | 33.53% | 60.38% | 43.24% | 60.49% | 64.88 | 330s |
| volume | v0145 | 39.09% | 49.43% | 28.75% | 32.95% | 65.8% | 44.54% | 58.08% | 70.8 | 270s |
| volume | v0110 | 39.06% | 49.64% | 28.47% | 36.91% | 48.38% | 42.52% | 63.65% | 49.13 | 7m |
| volume | v0176 | 38.99% | 49.15% | 28.83% | 32.03% | 63.45% | 43.45% | 60.25% | 72.58 | 225s |
| volume | v0240 | 38.96% | 49.32% | 28.6% | 34.3% | 56.59% | 42.7% | 60.99% | 59.8 | 345s |
| canonical | k0020 | 38.96% | 48.92% | 28.99% | 32.87% | 61.48% | 43.06% | 59.58% | 67.1 | 5m |
| volume | v0020 | 38.94% | 48.89% | 28.99% | 32.88% | 62.24% | 43.28% | 59.51% | 67.83 | 270s |
| volume | v0213 | 38.93% | 49.81% | 28.05% | 33.3% | 59.91% | 42.8% | 60.11% | 64.17 | 5m |
| volume | v0023 | 38.84% | 48.87% | 28.81% | 32.83% | 63.08% | 43.19% | 59.54% | 69.5 | 285s |
| canonical | k0191 | 38.83% | 48.71% | 28.95% | 31.67% | 63.34% | 43.56% | 59.12% | 69.27 | 270s |
| volume | v0191 | 38.82% | 48.68% | 28.95% | 31.46% | 63.8% | 43.23% | 58.61% | 70.83 | 270s |
| canonical | k0229 | 38.75% | 49.44% | 28.05% | 34.48% | 57.03% | 43.29% | 60.23% | 59.17 | 5m |
| canonical | k0045 | 38.67% | 48.51% | 28.82% | 32.17% | 62.11% | 43.28% | 59.43% | 70.75 | 4m |
| canonical | k0091 | 38.62% | 50.69% | 26.55% | 33.73% | 59.62% | 43.08% | 59.6% | 64.03 | 5m |
| volume | v0229 | 38.45% | 49% | 27.9% | 33.72% | 57.4% | 42.48% | 60.33% | 61.63 | 330s |
| volume | v0045 | 38.44% | 48.05% | 28.82% | 31.58% | 61.91% | 43% | 60.68% | 73.08 | 4m |
| volume | v0144 | 38.27% | 47.78% | 28.75% | 31.17% | 68.25% | 42.78% | 58.21% | 79.43 | 210s |
| canonical | k0002 | 38.24% | 47.65% | 28.84% | 32.69% | 58.18% | 42.7% | 59.17% | 67.67 | 5m |
| canonical | k0081 | 38.16% | 49.27% | 27.04% | 34.13% | 56.97% | 43.08% | 59.64% | 60.58 | 307500ms |
| canonical | k0006 | 38.11% | 46.97% | 29.25% | 30.98% | 62.54% | 42.47% | 58.32% | 70.5 | 5m |
| volume | v0044 | 37.9% | 46.85% | 28.95% | 31.11% | 55.48% | 41.1% | 58.82% | 61.7 | 5m |
| volume | v0081 | 37.74% | 48.45% | 27.04% | 31.18% | 64.87% | 42.97% | 56.93% | 73.28 | 270s |
| canonical | k0133 | 37.62% | 48.66% | 26.58% | 31.49% | 63.77% | 42.81% | 58.53% | 75.67 | 4m |
| volume | v0161 | 37.59% | 47.3% | 27.89% | 30.72% | 56.33% | 40.61% | 61.21% | 64.5 | 5m |
| volume | v0091 | 37.57% | 48.58% | 26.55% | 32.55% | 63.33% | 43.69% | 58.26% | 68.95 | 270s |
| canonical | k0211 | 37.31% | 46.71% | 27.91% | 30.02% | 63.59% | 42.7% | 58.18% | 71.1 | 5m |
| canonical | k0238 | 37.2% | 45.58% | 28.82% | 30.16% | 65.28% | 41.68% | 57.2% | 77 | 270s |

## Best fit candidates

| family | id | objective | median | P10 | precision | recall | F1 | agreement | signals/day | timing error P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical | k0153 | 38.46% | 52.55% | 24.37% | 37.39% | 61.72% | 46.56% | 59.88% | 74.75 | 5m |
| volume | v0153 | 38.46% | 52.55% | 24.37% | 37.39% | 61.72% | 46.56% | 59.88% | 74.75 | 5m |
| canonical | k0020 | 38.29% | 52.21% | 24.37% | 35.66% | 65.93% | 46.29% | 58.74% | 80.75 | 4m |
| volume | v0240 | 38.29% | 52.74% | 23.84% | 37.96% | 59.17% | 46.25% | 59.62% | 69.28 | 5m |
| volume | v0023 | 38.29% | 52.93% | 23.65% | 36.86% | 65.74% | 47.41% | 59.38% | 81.5 | 4m |
| canonical | k0023 | 38.28% | 52.92% | 23.65% | 36.59% | 64.28% | 47.25% | 59.77% | 78.78 | 4m |
| volume | v0020 | 38.23% | 52.09% | 24.37% | 35.76% | 66.03% | 46.4% | 58.41% | 80.81 | 4m |
| volume | v0145 | 37.88% | 51.97% | 23.79% | 34.71% | 67.15% | 45.76% | 58.27% | 83.96 | 4m |
| canonical | k0240 | 37.83% | 52.06% | 23.6% | 38.33% | 56.75% | 45.76% | 59.5% | 64.7 | 6m |
| canonical | k0145 | 37.79% | 51.74% | 23.84% | 34.64% | 67.06% | 45.68% | 58.01% | 83.04 | 4m |
| canonical | k0229 | 37.75% | 52.36% | 23.15% | 38.28% | 57.27% | 46.03% | 59.31% | 73.47 | 5m |
| volume | v0213 | 37.7% | 52.12% | 23.29% | 36.31% | 62.97% | 46.06% | 59.36% | 72.32 | 5m |

## Finalist parameters

| family | id | efficiency | fast | slow | power | volume | cap | volume power | deadband | mode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| volume | v0153 | 4177477ms | 1624855ms | 1624855ms | 0.607 | 207412ms | 4.192 | 1.968 | 2.583 bps/hour | hold |
| canonical | k0213 | 922511ms | 425399ms | 51988734ms | 0.585 | 190895ms | 3.251 | 0 | 9.073 bps/hour | hold |

## Holdout cases

| candidate | scale/window | score | precision | recall | F1 | agreement | signals/day | timing error P50 | timing error P90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v0153 | 1s:test-1 | 36.96% | 16.31% | 59.5% | 25.6% | 54% | 32.83 | 257s | 1890000ms |
| v0153 | 1s:test-2 | 45.42% | 28.07% | 69.41% | 39.97% | 53.6% | 77.26 | 211s | 1177s |
| v0153 | 5s:test-1 | 37.33% | 18.17% | 53.85% | 27.18% | 52.56% | 26.67 | 465s | 2397000ms |
| v0153 | 5s:test-2 | 47.27% | 29.19% | 65% | 40.29% | 57.74% | 64.79 | 250s | 1493000ms |
| v0153 | 15s:test-1 | 36.49% | 18.48% | 52.94% | 27.39% | 50.13% | 24.83 | 390s | 1935s |
| v0153 | 15s:test-2 | 46.67% | 29.23% | 59.79% | 39.26% | 57.79% | 55.1 | 5m | 29m |
| v0153 | 1m:test-1 | 39.78% | 19.3% | 43.03% | 26.65% | 59.47% | 17.83 | 12m | 3756s |
| v0153 | 1m:test-2 | 45.11% | 26.64% | 52.69% | 35.39% | 59.69% | 43.93 | 7m | 33m |
| v0153 | 5m:test-1 | 38.41% | 21.68% | 37.81% | 27.55% | 54.69% | 12.5 | 10m | 3030000ms |
| v0153 | 5m:test-2 | 40.57% | 21.01% | 37.31% | 26.88% | 61.1% | 26.6 | 15m | 50m |
| v0153 | 15m:test-1 | 33.19% | 9.15% | 18.59% | 12.26% | 64.58% | 10.5 | 30m | 75m |
| v0153 | 15m:test-2 | 33.06% | 11.87% | 19.76% | 14.84% | 60.39% | 16.52 | 15m | 75m |
| v0153 | 1h:test-1 | 21.12% | 0.37% | 0.69% | 0.48% | 52.08% | 5 | 1h | 2h |
| v0153 | 1h:test-2 | 24.13% | 0.68% | 0.97% | 0.8% | 59.13% | 7.24 | 1h | 2h |
| k0213 | 1s:test-1 | 39.67% | 17.59% | 58.64% | 27.07% | 58.59% | 30 | 261s | 1901700ms |
| k0213 | 1s:test-2 | 46.82% | 30.01% | 67.48% | 41.55% | 54.74% | 70.24 | 219s | 1338700ms |
| k0213 | 5s:test-1 | 35.49% | 17.09% | 50.63% | 25.55% | 50.4% | 26.67 | 450s | 3570s |
| k0213 | 5s:test-2 | 47.21% | 29.32% | 63.17% | 40.05% | 57.96% | 62.69 | 265s | 1475s |
| k0213 | 15s:test-1 | 37.76% | 17.81% | 48.97% | 26.12% | 55.22% | 23.83 | 487500ms | 2901000ms |
| k0213 | 15s:test-2 | 45.48% | 28.77% | 58.53% | 38.58% | 55.83% | 54.79 | 315s | 1915500ms |
| k0213 | 1m:test-1 | 39.5% | 15.66% | 36.21% | 21.86% | 65.97% | 18.5 | 1050s | 58m |
| k0213 | 1m:test-2 | 44.81% | 26.54% | 54.77% | 35.76% | 58.4% | 45.83 | 7m | 1956000ms |
| k0213 | 5m:test-1 | 30.22% | 15.96% | 31.18% | 21.12% | 43.87% | 14 | 20m | 70m |
| k0213 | 5m:test-2 | 39.14% | 19.93% | 42.61% | 27.15% | 57.11% | 32.02 | 10m | 50m |
| k0213 | 15m:test-1 | 34.1% | 10.5% | 19.99% | 13.77% | 64.58% | 9.83 | 15m | 75m |
| k0213 | 15m:test-2 | 31.6% | 11.32% | 20.93% | 14.7% | 56.94% | 18.36 | 15m | 1h |
| k0213 | 1h:test-1 | 20.86% | 0.39% | 0.69% | 0.5% | 51.39% | 4.67 | 1h | 2h |
| k0213 | 1h:test-2 | 24.18% | 0.71% | 0.97% | 0.82% | 59.23% | 6.86 | 1h | 2h |
