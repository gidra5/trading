# Volume-weighted KAMA oracle approximation search

Generated: 2026-07-12T19:15:31.668Z
Score version: 2
Raw results: `data/benchmarks/vw-kama-global-score-v2-de-2026-07-12.jsonl`

The fit ranking never reads validation or holdout scores. The best validated volume-aware and canonical (`volumePower=0`) candidates are the only candidates evaluated on holdout.

## Data and objective

- Source: `data/historical/spot-btcusdt/btcusdt/1s` (1s); target scales: 1s, 5s, 15s, 1m, 5m, 15m, 1h.
- Windows: fit 2021-09-08..2021-12-20, 2022-05-14..2022-08-03, 2023-03-03..2023-12-05; validation 2024-01-02..2024-02-26, 2024-07-07..2024-11-11; test 2025-02-14..2025-07-06, 2026-04-22..2026-07-10.
- Each continuous segment reserves 3d before scoring.
- Signal: completed-candle volume-weighted KAMA derivative rate; accepted directional transitions use separate buy/sell maxima and zero-centered Gaussian rate curves. Sizing mode price-marks the fraction, while confidence mode holds it as uncertainty until the next signal.
- Signal memory: after the first signal, a candidate state change emits only when the current close is strictly more than 17.5 bps from the last emitted signal price; rejected changes retain the prior state. This is the same friction used by the oracle.
- Matching is one chronological one-to-one alignment by resulting state. It maximizes total timing credit, so extra candidate transitions reduce precision and uncovered oracle transitions reduce recall.
- Case score weights timing-credited transition F1 60%, the selected sizing/confidence agreement 30%, and signal cleanliness 10%. Cleanliness is matched / (matched + extra); the displayed noise/signal ratio is extra / matched. Trading returns are neither computed nor ranked.
- Candidate objective equally weights the median and P10 case score; every scale/window case has equal weight.

## Holdout finalists

| family | id | strategy | objective | median | P10 | precision | recall | F1 | agreement | cleanliness | noise/signal | signals/day | timing error P50 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| volume | global-clean-v0182 | sizing | 31.74% | 39.05% | 24.43% | 19.28% | 39.87% | 26.44% | 61.34% | 54.39% | 0.839 | 18.75 | 752500ms |
| canonical | global-clean-k0050 | sizing | 30.81% | 39.38% | 22.23% | 20.55% | 44.26% | 27.58% | 59.09% | 50.82% | 0.968 | 21.58 | 645s |

## Validation finalists

| family | id | strategy | objective | median | P10 | precision | recall | F1 | agreement | cleanliness | noise/signal | signals/day | timing error P50 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| volume | global-clean-v0182 | sizing | 41.39% | 51.99% | 30.78% | 37.73% | 51.67% | 43.61% | 62.93% | 63.83% | 0.567 | 49.5 | 6m |
| volume | global-clean-v0182-confidence | confidence | 41.39% | 51.99% | 30.78% | 37.73% | 51.67% | 43.61% | 62.93% | 63.83% | 0.567 | 49.5 | 6m |
| canonical | global-clean-k0050 | sizing | 40.8% | 51.75% | 29.86% | 36.2% | 54.64% | 43.55% | 62.72% | 60.42% | 0.655 | 54.6 | 6m |
| canonical | global-clean-k0050-confidence | confidence | 40.8% | 51.75% | 29.86% | 36.2% | 54.64% | 43.55% | 62.72% | 60.42% | 0.655 | 54.6 | 6m |
| volume | global-refined-v0016 | sizing | 39.71% | 50.54% | 28.89% | 35.67% | 54.12% | 43% | 62.08% | 58.08% | 0.722 | 54.87 | 6m |
| volume | global-refined-v0016-confidence | confidence | 39.71% | 50.54% | 28.89% | 35.67% | 54.12% | 43% | 62.08% | 58.08% | 0.722 | 54.87 | 6m |
| canonical | global-refined-k0016 | sizing | 39.64% | 50.43% | 28.85% | 35.41% | 53.62% | 42.65% | 62.19% | 58.78% | 0.701 | 54.8 | 6m |
| canonical | global-refined-k0016-confidence | confidence | 39.64% | 50.43% | 28.85% | 35.41% | 53.62% | 42.65% | 62.19% | 58.78% | 0.701 | 54.8 | 6m |
| canonical | k0103 | sizing | 36.46% | 46.36% | 26.56% | 36.7% | 42.11% | 39.26% | 54.84% | 67.75% | 0.477 | 43 | 465s |
| volume | v0103 | sizing | 36.28% | 45.99% | 26.57% | 36.57% | 41.92% | 39.14% | 54.84% | 67.77% | 0.477 | 43.5 | 465s |
| volume | v0150 | confidence | 35.96% | 43.68% | 28.25% | 32.37% | 46.57% | 38.18% | 50.82% | 59.86% | 0.671 | 52.33 | 7m |
| canonical | k0150 | confidence | 35.94% | 44.41% | 27.47% | 35.43% | 29.26% | 33.79% | 59.42% | 73.73% | 0.357 | 33 | 11m |
| canonical | k0246 | sizing | 34.59% | 41.43% | 27.74% | 30.92% | 62.1% | 42.47% | 49.47% | 49.8% | 1.008 | 69.78 | 270s |
| volume | v0145 | sizing | 34.08% | 40.14% | 28.01% | 31.78% | 65.78% | 42.86% | 41.98% | 47.96% | 1.085 | 74.88 | 4m |
| volume | v0054 | sizing | 33.55% | 42.92% | 24.19% | 33.61% | 56.58% | 42.2% | 42.51% | 56.28% | 0.777 | 58.42 | 5m |
| canonical | k0054 | sizing | 33.48% | 42.78% | 24.19% | 33.61% | 56.55% | 42.15% | 42.56% | 56.28% | 0.777 | 58.42 | 5m |
| canonical | k0145 | sizing | 33.45% | 38.95% | 27.95% | 30.59% | 67.07% | 42.15% | 39.31% | 46.18% | 1.165 | 79.08 | 4m |
| canonical | k0193 | sizing | 33.02% | 38.3% | 27.74% | 27.52% | 66.32% | 39.32% | 34.84% | 47.67% | 1.098 | 85.58 | 4m |
| volume | v0014 | sizing | 32.93% | 39.02% | 26.85% | 30.95% | 63.6% | 43.22% | 41.12% | 50.32% | 0.988 | 70.45 | 270s |
| canonical | k0014 | sizing | 32.92% | 39% | 26.85% | 30.97% | 63.62% | 43.22% | 41.11% | 50.29% | 0.989 | 70.48 | 270s |
| canonical | k0127 | sizing | 32.64% | 37.59% | 27.69% | 28.43% | 67.01% | 40.38% | 37.95% | 43.05% | 1.323 | 84.03 | 270s |
| volume | v0037 | sizing | 32.48% | 38.76% | 26.19% | 31.66% | 62.15% | 43.43% | 41.34% | 51.8% | 0.931 | 67.38 | 270s |
| canonical | k0037 | sizing | 32.47% | 38.74% | 26.19% | 31.83% | 61.99% | 43.53% | 41.53% | 52.23% | 0.915 | 66.88 | 270s |
| volume | v0246 | sizing | 32.47% | 37.45% | 27.48% | 27.2% | 66.77% | 40.8% | 41.73% | 46.97% | 1.131 | 76.58 | 4m |
| volume | v0026 | sizing | 32.46% | 40.01% | 24.92% | 32.59% | 61.77% | 43.37% | 43.05% | 52.07% | 0.921 | 66.97 | 5m |
| volume | v0127 | sizing | 32.38% | 37.05% | 27.72% | 28.07% | 67.21% | 40.26% | 34.77% | 42.47% | 1.355 | 84.7 | 270s |
| canonical | k0055 | sizing | 32.3% | 40.59% | 24.01% | 35.04% | 29.3% | 32.15% | 44.44% | 74.19% | 0.348 | 29.75 | 735s |
| canonical | k0026 | sizing | 32.1% | 39.28% | 24.93% | 31.76% | 61.9% | 43.3% | 42.77% | 52.27% | 0.913 | 67.07 | 5m |
| canonical | k0041 | confidence | 31.93% | 36.3% | 27.55% | 27.66% | 65.82% | 40.91% | 45.52% | 46.2% | 1.166 | 77.38 | 270s |
| volume | v0041 | confidence | 31.92% | 36.17% | 27.66% | 27.71% | 65.76% | 40.83% | 45.28% | 46.03% | 1.174 | 77.63 | 270s |
| volume | v0193 | sizing | 31.89% | 36.09% | 27.68% | 26.26% | 67.11% | 38.49% | 35.56% | 40.25% | 1.485 | 88.93 | 270s |
| canonical | k0182 | confidence | 31.68% | 39.2% | 24.17% | 27.95% | 40.09% | 32.44% | 49.83% | 52.52% | 0.904 | 56.92 | 555s |
| volume | v0104 | confidence | 31.54% | 37.34% | 25.74% | 30.48% | 67.58% | 43.08% | 34.01% | 47.11% | 1.124 | 76.38 | 255s |
| volume | v0182 | confidence | 31.52% | 39% | 24.04% | 27.74% | 38.86% | 31.87% | 49.74% | 52.33% | 0.911 | 56.42 | 553750ms |
| canonical | k0161 | sizing | 31.34% | 37.05% | 25.64% | 28.95% | 67.61% | 41.47% | 35.35% | 43.61% | 1.293 | 81.5 | 210s |
| volume | v0161 | sizing | 31.31% | 36.98% | 25.64% | 28.95% | 67.56% | 41.47% | 35.32% | 43.61% | 1.293 | 81.5 | 210s |
| canonical | k0116 | confidence | 31.16% | 37.99% | 24.34% | 30.06% | 63.3% | 42.73% | 37.59% | 51.15% | 0.956 | 69.27 | 270s |
| canonical | k0104 | confidence | 31.15% | 36.57% | 25.73% | 30.34% | 67.85% | 43% | 33.47% | 46.82% | 1.137 | 76.82 | 255s |
| canonical | k0108 | confidence | 30.9% | 36.23% | 25.58% | 28.68% | 68.61% | 40.44% | 30.6% | 44.07% | 1.269 | 84.08 | 4m |
| canonical | k0191 | sizing | 30.81% | 36.22% | 25.4% | 29.48% | 67.56% | 41.99% | 34.07% | 44.78% | 1.236 | 80.17 | 4m |
| volume | v0143 | sizing | 30.7% | 37.42% | 23.99% | 31.24% | 60.35% | 43.1% | 36.99% | 53.77% | 0.86 | 64.42 | 330s |
| canonical | k0143 | sizing | 30.7% | 37.42% | 23.99% | 31.24% | 60.35% | 43.1% | 37% | 53.77% | 0.86 | 64.42 | 330s |
| canonical | k0176 | sizing | 30.68% | 38.88% | 22.48% | 35.45% | 50.51% | 41.66% | 27.25% | 62.08% | 0.611 | 51.48 | 450s |
| volume | v0191 | sizing | 30.58% | 35.65% | 25.5% | 29.22% | 69.05% | 42.07% | 32.63% | 43.53% | 1.297 | 82.58 | 4m |
| canonical | k0093 | confidence | 30.51% | 36.35% | 24.67% | 29.29% | 66.83% | 41.65% | 32.83% | 45.67% | 1.19 | 77.77 | 4m |
| canonical | k0071 | confidence | 30.49% | 34.89% | 26.09% | 27.46% | 65.78% | 40.65% | 32.58% | 46.11% | 1.17 | 77.32 | 270s |
| volume | v0176 | sizing | 30.44% | 38.09% | 22.8% | 33.05% | 53.51% | 40.85% | 27.1% | 59.49% | 0.681 | 58.42 | 390s |
| canonical | k0007 | sizing | 30.43% | 34.88% | 25.98% | 28.01% | 68.42% | 41.56% | 26.95% | 45.01% | 1.223 | 80.15 | 4m |
| volume | v0007 | sizing | 30.42% | 34.85% | 26% | 28.03% | 68.42% | 41.6% | 26.93% | 45.01% | 1.223 | 80.15 | 4m |
| volume | v0080 | confidence | 30.35% | 35.14% | 25.55% | 27.9% | 55.65% | 39.6% | 22.56% | 53.27% | 0.879 | 64.9 | 7m |
| canonical | k0074 | confidence | 30.2% | 39.8% | 20.6% | 33.86% | 61.8% | 43.75% | 27.28% | 52.92% | 0.89 | 65.58 | 5m |
| volume | v0074 | confidence | 30.13% | 39.68% | 20.58% | 33.78% | 61.83% | 43.69% | 27.25% | 53.32% | 0.875 | 65.58 | 5m |
| volume | v0233 | confidence | 29.78% | 33.72% | 25.83% | 28.59% | 20.74% | 16.68% | 54.73% | 66.14% | 0.514 | 19.93 | 842500ms |
| volume | v0254 | sizing | 29.75% | 35.55% | 23.95% | 28.02% | 69.39% | 40.56% | 26.02% | 41.79% | 1.393 | 87.83 | 210s |
| volume | v0071 | confidence | 29.67% | 33.5% | 25.83% | 26.49% | 68.79% | 39.96% | 32.6% | 43.93% | 1.277 | 82.27 | 210s |
| volume | v0055 | sizing | 29.6% | 35.63% | 23.57% | 29.81% | 52.39% | 40.3% | 36.49% | 55.82% | 0.795 | 58.05 | 6m |
| volume | v0110 | confidence | 29.51% | 32.27% | 26.75% | 25.92% | 70.97% | 39.23% | 22.32% | 38.59% | 1.592 | 93.25 | 210s |
| canonical | k0052 | confidence | 29.39% | 37.98% | 20.8% | 34.37% | 54.51% | 43.42% | 30.83% | 58.22% | 0.718 | 54.77 | 330s |
| volume | v0034 | confidence | 29.35% | 37.93% | 20.77% | 34.55% | 54.19% | 42.89% | 29.7% | 57.88% | 0.728 | 55.28 | 6m |
| canonical | k0034 | confidence | 29.33% | 37.89% | 20.77% | 34.59% | 54.4% | 42.99% | 29.7% | 57.88% | 0.728 | 55.43 | 6m |
| canonical | k0254 | sizing | 29.29% | 34.63% | 23.95% | 27.55% | 67.55% | 39.79% | 26.32% | 41.52% | 1.408 | 86.78 | 225s |
| volume | v0108 | confidence | 29.07% | 33.12% | 25.02% | 24.87% | 67.26% | 36.57% | 28.42% | 40.95% | 1.443 | 86.83 | 4m |
| canonical | k0249 | confidence | 28.71% | 36.69% | 20.72% | 30.75% | 62.82% | 42.17% | 28.06% | 49.31% | 1.03 | 71.77 | 5m |
| volume | v0234 | sizing | 28.13% | 33.19% | 23.07% | 23.14% | 47.49% | 31.53% | 34.19% | 46.36% | 1.157 | 79.17 | 357500ms |

## Best fit candidates

| family | id | strategy | objective | median | P10 | precision | recall | F1 | agreement | cleanliness | noise/signal | signals/day | timing error P50 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| volume | global-clean-v0182 | sizing | 39.53% | 53.06% | 26% | 39.96% | 53.48% | 45.74% | 61.13% | 69.07% | 0.448 | 57.2 | 6m |
| volume | global-clean-v0182-confidence | confidence | 39.53% | 53.06% | 26% | 39.96% | 53.48% | 45.74% | 61.13% | 69.07% | 0.448 | 57.2 | 6m |
| canonical | global-clean-k0050 | sizing | 39.04% | 52.42% | 25.66% | 38.21% | 56.89% | 45.72% | 60.56% | 64.68% | 0.546 | 63.14 | 5m |
| canonical | global-clean-k0050-confidence | confidence | 39.04% | 52.42% | 25.66% | 38.21% | 56.89% | 45.72% | 60.56% | 64.68% | 0.546 | 63.14 | 5m |
| volume | global-refined-v0016 | sizing | 38.41% | 52.2% | 24.61% | 38.31% | 55.34% | 45.27% | 60.24% | 63.34% | 0.579 | 64.96 | 6m |
| volume | global-refined-v0016-confidence | confidence | 38.41% | 52.2% | 24.61% | 38.31% | 55.34% | 45.27% | 60.24% | 63.34% | 0.579 | 64.96 | 6m |
| canonical | global-refined-k0016 | sizing | 38.18% | 51.75% | 24.61% | 37.97% | 54.85% | 44.88% | 60.17% | 63.01% | 0.587 | 64.67 | 6m |
| canonical | global-refined-k0016-confidence | confidence | 38.18% | 51.75% | 24.61% | 37.97% | 54.85% | 44.88% | 60.17% | 63.01% | 0.587 | 64.67 | 6m |
| volume | v0150 | confidence | 35.92% | 46.62% | 25.21% | 39.05% | 39.23% | 43.62% | 52.02% | 66.7% | 0.499 | 57.23 | 6m |
| canonical | k0054 | sizing | 34.36% | 47.04% | 21.68% | 36.87% | 58.93% | 45.36% | 44.35% | 62.01% | 0.613 | 64.5 | 5m |
| volume | v0054 | sizing | 34.35% | 47.03% | 21.68% | 36.9% | 58.8% | 45.34% | 44.31% | 62.12% | 0.61 | 64.15 | 5m |
| volume | v0103 | sizing | 34.21% | 45.67% | 22.75% | 38.6% | 38.43% | 38.7% | 52.11% | 74.66% | 0.339 | 46.31 | 8m |

## Finalist parameters

| family | id | agreement | efficiency | fast | slow | power | volume | cap | volume power | base threshold | state mode | threshold mode | noise lookback | noise multiplier | buy max | sell max | buy sigma | sell sigma |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| volume | global-clean-v0182 | sizing | 265846ms | 1113418ms | 18h | 0.625 | 283525ms | 5.081 | 0.485 | 30 bps/hour | hold | adaptive | 2457073ms | 0 | 100% | 100% | 1000000000000 | 1000000000000 |
| canonical | global-clean-k0050 | sizing | 1420250ms | 976600ms | 976600ms | 0.796 | 82113ms | 4.274 | 0 | 21.215 bps/hour | hold | static | 15m | 5.607 | 100% | 100% | 1000000000000 | 1000000000000 |

## Holdout cases

| candidate | scale/window | score | precision | recall | F1 | agreement | cleanliness | noise/signal | signals/day | timing error P50 | timing error P90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| global-clean-v0182 | 1s:test-1 | 34.58% | 18.6% | 49.6% | 27.05% | 49.82% | 34.03% | 1.939 | 24 | 368s | 2890400ms |
| global-clean-v0182 | 1s:test-2 | 48.19% | 32.94% | 57.92% | 41.99% | 58.75% | 53.66% | 0.863 | 54.93 | 307500ms | 1630500ms |
| global-clean-v0182 | 5s:test-1 | 41.32% | 19.96% | 43.62% | 27.39% | 68.83% | 42.37% | 1.36 | 19.67 | 605s | 2821s |
| global-clean-v0182 | 5s:test-2 | 47.89% | 32.08% | 54.64% | 40.43% | 60.4% | 55.12% | 0.814 | 49.55 | 370s | 1726000ms |
| global-clean-v0182 | 15s:test-1 | 34.31% | 17.55% | 36.12% | 23.62% | 52.16% | 44.86% | 1.229 | 17.83 | 937500ms | 3436500ms |
| global-clean-v0182 | 15s:test-2 | 47.8% | 31.88% | 51.13% | 39.27% | 61.39% | 58.16% | 0.719 | 43.19 | 7m | 1995s |
| global-clean-v0182 | 1m:test-1 | 43.67% | 22.95% | 44% | 30.17% | 69.65% | 46.74% | 1.14 | 15.33 | 8m | 3516000ms |
| global-clean-v0182 | 1m:test-2 | 46.22% | 29.46% | 46.98% | 36.21% | 62.36% | 57.93% | 0.726 | 35.43 | 8m | 35m |
| global-clean-v0182 | 5m:test-1 | 38.34% | 16.68% | 24.44% | 19.83% | 69.1% | 57.14% | 0.75 | 10.5 | 1050s | 70m |
| global-clean-v0182 | 5m:test-2 | 39.75% | 21.1% | 33.28% | 25.83% | 61.28% | 58.67% | 0.704 | 23.62 | 15m | 55m |
| global-clean-v0182 | 15m:test-1 | 30.29% | 9.44% | 13.4% | 11.08% | 62.15% | 50% | 1 | 7.33 | 30m | 5310000ms |
| global-clean-v0182 | 15m:test-2 | 33.03% | 11.7% | 17.67% | 14.08% | 62.67% | 57.78% | 0.731 | 15 | 30m | 75m |
| global-clean-v0182 | 1h:test-1 | 15.26% | 0.47% | 0.29% | 0.36% | 36.81% | 40% | 1.5 | 1.67 | 1h | 102m |
| global-clean-v0182 | 1h:test-2 | 21.92% | 0.75% | 0.44% | 0.55% | 48.31% | 70.97% | 0.409 | 2.95 | 1h | 2h |
| global-clean-k0050 | 1s:test-1 | 35.68% | 17.89% | 46.05% | 25.77% | 55.64% | 35.25% | 1.837 | 23.17 | 472s | 3335000ms |
| global-clean-k0050 | 1s:test-2 | 48.39% | 32.14% | 63.93% | 42.77% | 59.39% | 49.16% | 1.034 | 62.14 | 251s | 1469800ms |
| global-clean-k0050 | 5s:test-1 | 38.34% | 21.06% | 53.05% | 30.16% | 54.98% | 37.5% | 1.667 | 22.67 | 335s | 2170s |
| global-clean-k0050 | 5s:test-2 | 47.06% | 30.87% | 58.98% | 40.53% | 58.8% | 51.05% | 0.959 | 55.6 | 325s | 1738000ms |
| global-clean-k0050 | 15s:test-1 | 37.93% | 20.03% | 47.37% | 28.15% | 56.32% | 41.46% | 1.412 | 20.5 | 690s | 48m |
| global-clean-k0050 | 15s:test-2 | 46.65% | 30.06% | 54.62% | 38.78% | 60.41% | 52.6% | 0.901 | 48.93 | 375s | 32m |
| global-clean-k0050 | 1m:test-1 | 40.66% | 19.94% | 35.31% | 25.49% | 67.7% | 50.59% | 0.977 | 14.17 | 14m | 4404000ms |
| global-clean-k0050 | 1m:test-2 | 45.86% | 28.51% | 51.19% | 36.62% | 61.83% | 53.31% | 0.876 | 39.88 | 8m | 1908000ms |
| global-clean-k0050 | 5m:test-1 | 46.34% | 24.35% | 42.47% | 30.95% | 73.9% | 56% | 0.786 | 12.5 | 10m | 2670000ms |
| global-clean-k0050 | 5m:test-2 | 40.43% | 21.64% | 35.88% | 27% | 62.09% | 55.99% | 0.786 | 24.83 | 15m | 55m |
| global-clean-k0050 | 15m:test-1 | 21.86% | 8.4% | 13.55% | 10.37% | 36.81% | 46% | 1.174 | 8.33 | 30m | 5220000ms |
| global-clean-k0050 | 15m:test-2 | 32.11% | 11.94% | 19.84% | 14.9% | 59.62% | 52.81% | 0.893 | 16.5 | 15m | 1h |
| global-clean-k0050 | 1h:test-1 | 20.05% | 0.37% | 0.49% | 0.42% | 54.86% | 33.33% | 2 | 3.5 | 1h | 2h |
| global-clean-k0050 | 1h:test-2 | 23.09% | 0.75% | 0.71% | 0.73% | 53.97% | 64.65% | 0.547 | 4.71 | 1h | 2h |
