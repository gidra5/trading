# Volume-weighted KAMA oracle approximation search

Generated: 2026-07-13T04:52:31.492Z
Score version: 2
Raw results: `data/benchmarks/vw-kama-regime-confirmations-multicore-de-2026-07-13.jsonl`

The fit ranking never reads validation or holdout scores. The best validated volume-aware and canonical (`volumePower=0`) candidates are the only candidates evaluated on holdout.

## Data and objective

- Source: `data/historical/spot-btcusdt/btcusdt/1s` (1s); target scales: 1s, 5s, 15s, 1m, 5m, 15m, 1h.
- Windows: fit 2021-09-08..2021-12-20, 2022-05-14..2022-08-03, 2023-03-03..2023-12-05; validation 2024-01-02..2024-02-26, 2024-07-07..2024-11-11; test 2025-02-14..2025-07-06, 2026-04-22..2026-07-10.
- Each continuous segment reserves 3d before scoring.
- Candidate evaluation uses 6 bounded worker processes.
- Signal: completed-candle volume-weighted KAMA derivative rate with flat, hold, or hysteresis state handling. A causal logistic confirmation combines KAMA acceleration, price overextension, independent slow-EMA trend, RSI, and ADX-strength-weighted DMI direction, then scales or filters the signal. Sizing mode price-marks the fraction, while confidence mode holds it as uncertainty until the next signal.
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
| canonical | k0439 | confidence | 38.58% | 48.74% | 28.43% | 35.06% | 56.57% | 43.28% | 57.3% | 56.59% | 0.767 | 58.47 | 330s |
| canonical | k0137 | confidence | 38.42% | 48.71% | 28.13% | 35.88% | 52.33% | 42.59% | 56.53% | 61.91% | 0.615 | 52.5 | 375s |
| volume | v0558 | confidence | 38.37% | 48.61% | 28.12% | 36.1% | 40.25% | 41.05% | 57.5% | 64.56% | 0.549 | 47 | 7m |
| volume | v0439 | confidence | 38.33% | 48.18% | 28.47% | 35.07% | 58.36% | 43.81% | 55.79% | 56.73% | 0.763 | 60.23 | 330s |
| canonical | k0257 | sizing | 38.28% | 48.6% | 27.96% | 37.86% | 41.18% | 39.45% | 57.39% | 71.27% | 0.403 | 39.27 | 510s |
| volume | v0137 | confidence | 38.1% | 47.82% | 28.38% | 35.68% | 48.34% | 41.05% | 56.61% | 64.56% | 0.549 | 48.98 | 7m |
| canonical | k0253 | confidence | 37.92% | 47.98% | 27.87% | 35.99% | 53.25% | 42.95% | 56.98% | 58.68% | 0.704 | 53.45 | 6m |
| canonical | k0413 | confidence | 37.87% | 47.86% | 27.89% | 34.79% | 51.14% | 41.41% | 54.84% | 59.06% | 0.693 | 53.2 | 390s |
| volume | v0385 | confidence | 37.6% | 46.84% | 28.37% | 39.13% | 32.69% | 36.77% | 60.23% | 77.05% | 0.298 | 34.42 | 570s |
| canonical | k0712 | sizing | 37.59% | 47.29% | 27.88% | 37.37% | 49.34% | 42.53% | 55.18% | 64.12% | 0.56 | 47.8 | 390s |
| volume | v0712 | sizing | 37.59% | 47.29% | 27.88% | 37.37% | 49.34% | 42.53% | 55.18% | 64.12% | 0.56 | 47.8 | 390s |
| canonical | k0131 | sizing | 37.22% | 46.42% | 28.02% | 35.66% | 43.24% | 39.08% | 57.46% | 63.66% | 0.571 | 43.7 | 510s |
| volume | v0253 | confidence | 37.14% | 46.66% | 27.63% | 37.4% | 52.26% | 43.6% | 55.11% | 59.48% | 0.681 | 50.62 | 6m |
| volume | v0008 | sizing | 37.1% | 46.31% | 27.89% | 35.06% | 56.52% | 43.28% | 53.82% | 57.46% | 0.74 | 58.35 | 6m |
| canonical | k0008 | sizing | 37.07% | 46.41% | 27.72% | 35.01% | 55.81% | 43.02% | 54.49% | 56.66% | 0.765 | 57.82 | 330s |
| volume | v0441 | sizing | 36.37% | 44.51% | 28.22% | 34.21% | 46.09% | 39.27% | 54.75% | 61.52% | 0.625 | 48.82 | 8m |

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
| canonical | k0131 | sizing | 37.1% | 49.57% | 24.63% | 39.77% | 40.51% | 40% | 55.49% | 71.37% | 0.401 | 49.94 | 8m |
| volume | v0137 | confidence | 37.05% | 49.37% | 24.73% | 38.38% | 49.17% | 43.41% | 53.81% | 70.1% | 0.427 | 53.35 | 7m |
| canonical | k0257 | sizing | 37.01% | 49.88% | 24.13% | 41.13% | 41.48% | 41.31% | 54.14% | 77.23% | 0.295 | 48.08 | 510s |
| canonical | k0439 | confidence | 36.94% | 49.79% | 24.1% | 37.51% | 57.62% | 45.44% | 52.95% | 62.51% | 0.6 | 66.67 | 5m |

## Finalist parameters

| family | id | agreement | efficiency | fast | slow | power | volume | cap | volume power | base threshold | state mode | threshold mode | noise lookback | noise multiplier | buy max | sell max | buy sigma | sell sigma |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| volume | global-clean-v0182 | sizing | 265846ms | 1113418ms | 18h | 0.625 | 283525ms | 5.081 | 0.485 | 30 bps/hour | hold | adaptive | 2457073ms | 0 | 100% | 100% | 1000000000000 | 1000000000000 |
| canonical | global-clean-k0050 | sizing | 1420250ms | 976600ms | 976600ms | 0.796 | 82113ms | 4.274 | 0 | 21.215 bps/hour | hold | static | 15m | 5.607 | 100% | 100% | 1000000000000 | 1000000000000 |

## Finalist confirmation parameters

| candidate | mix | minimum quality | acceleration lookback | distance lookback | acceleration weight | overextension weight | bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| global-clean-v0182 | 0% | 0% | 1h | 1h | 1 | 1 | 0 |
| global-clean-k0050 | 0% | 0% | 1h | 1h | 1 | 1 | 0 |

| candidate | hysteresis release | slow EMA | EMA tolerance | EMA weight/gate | RSI period | RSI tolerance/weight | DMI period | DMI weight | ADX threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| global-clean-v0182 | 25% | 1h | 0 bps/h | 0 / 0% | 14m | 0 / 0 | 14m | 0 | 20 |
| global-clean-k0050 | 25% | 1h | 0 bps/h | 0 / 0% | 14m | 0 / 0 | 14m | 0 | 20 |

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
