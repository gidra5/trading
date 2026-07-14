# Volume-weighted KAMA oracle approximation search

Generated: 2026-07-14T17:17:27.672Z
Score version: 3
Raw results: `data/benchmarks/vw-kama-1s-mean-reversion-agreement60-de-2026-07-13.jsonl`

The fit ranking never reads validation or holdout scores. The best validated volume-aware and canonical (both volume powers are zero) candidates are the only candidates evaluated on holdout.

## Data and objective

- Source: `data/historical/spot-btcusdt/btcusdt/1s` (1s); target scales: 1s.
- Windows: fit 2021-09-08..2021-12-20, 2022-05-14..2022-08-03, 2023-03-03..2023-12-05; validation 2024-01-02..2024-02-26, 2024-07-07..2024-11-11; test 2025-02-14..2025-07-06, 2026-04-22..2026-07-10.
- Each continuous segment reserves 3d before scoring.
- Candidate evaluation uses 12 persistent shared-memory worker threads, dynamic batching, cross-generation score caching, and stage-wide prepared candle/oracle caching.
- Generation zero evaluates all 852 warm genomes plus 2048 Latin-hypercube genomes, then selects 2048 by 70% score / 30% parameter novelty. Warm sources: `data/benchmarks/vw-kama-regime-confirmations-multicore-de-2026-07-13.jsonl`, `data/benchmarks/vw-kama-window-presets-fit-only.json`.
- Search uses 2 independent restart(s), adaptive current-to-pbest differential evolution, family/agreement/confirmation-mask islands with cross-island migration, rotating fit folds with a full-fit pass every 4 generations, and 4 shrinking elite-refinement round(s).
- Signal: completed-candle volume-weighted KAMA derivative rate with flat, hold, or hysteresis state handling. ER optionally weights every price move by `(volume / causal volume EMA)^ER volume power`; zero recovers standard ER. A causal volatility-normalized distance-from-EMA regime blends the local trend direction into its countertrend direction as mean-reversion strength rises. A causal logistic confirmation combines KAMA acceleration, price overextension, independent slow-EMA trend, RSI, and ADX-strength-weighted DMI direction, then scales or filters the signal. Sizing mode price-marks the fraction, while confidence mode holds it as uncertainty until the next signal.
- Signal memory: after the first signal, a candidate state change emits only when the current close is strictly more than 17.5 bps from the last emitted signal price; rejected changes retain the prior state. This is the same friction used by the oracle.
- Matching is one chronological one-to-one alignment by resulting state. It maximizes total timing credit, so extra candidate transitions reduce precision and uncovered oracle transitions reduce recall.
- Case score weights timing-credited transition F1 20%, the selected sizing/confidence agreement 60%, and signal cleanliness 20%. Cleanliness is matched / (matched + extra); the displayed noise/signal ratio is extra / matched. Trading returns are neither computed nor ranked.
- Candidate objective equally weights the median and P10 case score; every scale/window case has equal weight.

## Holdout finalists

| family | id | strategy | objective | median | P10 | precision | recall | F1 | agreement | cleanliness | noise/signal | signals/day | timing error P50 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| volume | v00129 | confidence | 51.74% | 53.24% | 50.25% | 25.69% | 37.74% | 30.13% | 59.7% | 56.97% | 0.82 | 27.42 | 683500ms |
| canonical | k00459 | confidence | 49.88% | 51.68% | 48.08% | 27.93% | 39.21% | 32.51% | 55.4% | 59.66% | 0.702 | 27.2 | 721250ms |

## Validation finalists

| family | id | strategy | objective | median | P10 | precision | recall | F1 | agreement | cleanliness | noise/signal | signals/day | timing error P50 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical | k00459 | confidence | 60.3% | 60.93% | 59.67% | 46.66% | 43.41% | 44.83% | 61.15% | 76.38% | 0.313 | 57.33 | 348s |
| canonical | k00329 | sizing | 59.99% | 60.12% | 59.86% | 48.29% | 49.35% | 48.51% | 58.91% | 75.35% | 0.331 | 63.3 | 264500ms |
| volume | v00129 | confidence | 59.97% | 60.52% | 59.43% | 43.6% | 33.71% | 37.77% | 62.55% | 77.17% | 0.296 | 47.37 | 466s |
| canonical | k00489 | confidence | 59.97% | 60.48% | 59.46% | 43.41% | 33.5% | 37.57% | 62.52% | 77.24% | 0.295 | 47.32 | 466s |
| canonical | k00387 | confidence | 59.9% | 60.07% | 59.73% | 44.89% | 42.27% | 43.44% | 61.09% | 73.65% | 0.36 | 57.9 | 357500ms |
| volume | v00131 | confidence | 59.86% | 60.35% | 59.37% | 43.38% | 33.55% | 37.58% | 62.29% | 77.28% | 0.294 | 47.42 | 469s |
| volume | v00132 | confidence | 59.73% | 59.91% | 59.55% | 45.27% | 41.37% | 43.09% | 60.5% | 74.94% | 0.335 | 56.23 | 370500ms |
| volume | v00074 | sizing | 59.66% | 60.15% | 59.16% | 46.81% | 43.33% | 44.89% | 59.79% | 76.51% | 0.31 | 56.98 | 342500ms |
| canonical | k00458 | confidence | 59.52% | 59.54% | 59.5% | 46.77% | 42.85% | 44.6% | 59.02% | 76.02% | 0.317 | 56.33 | 337250ms |
| volume | v00925 | confidence | 59.12% | 59.52% | 58.71% | 45.17% | 40.82% | 42.72% | 60.13% | 74.52% | 0.342 | 55.6 | 370s |
| volume | v01221 | confidence | 58.7% | 59.54% | 57.86% | 42.41% | 33.62% | 37.33% | 61.55% | 75.73% | 0.321 | 48.42 | 469500ms |
| volume | v00073 | sizing | 58.7% | 58.81% | 58.59% | 45.56% | 41.9% | 43.56% | 58.4% | 75.26% | 0.331 | 56.43 | 353s |
| volume | v00130 | confidence | 58.65% | 59.24% | 58.06% | 45.6% | 41.81% | 43.45% | 59.38% | 74.62% | 0.341 | 56.38 | 352s |
| canonical | k00385 | confidence | 58.64% | 59.3% | 57.97% | 43.67% | 36.52% | 39.64% | 60.82% | 74.41% | 0.344 | 51.13 | 407750ms |
| canonical | k00425 | confidence | 58.61% | 58.97% | 58.24% | 43.69% | 47.15% | 45.21% | 60.45% | 68.28% | 0.465 | 66.2 | 313500ms |
| canonical | k00388 | confidence | 58.51% | 58.85% | 58.16% | 42.91% | 36.4% | 39.16% | 60.49% | 73.62% | 0.359 | 52.13 | 416500ms |
| canonical | k00345 | sizing | 58.36% | 58.78% | 57.94% | 44.86% | 49.51% | 46.81% | 59.49% | 68.63% | 0.459 | 67.97 | 281s |
| volume | v00065 | sizing | 57.82% | 58.11% | 57.52% | 45.23% | 50.18% | 47.35% | 57.97% | 69.3% | 0.445 | 68.42 | 269s |
| canonical | k00386 | confidence | 57.76% | 58.6% | 56.91% | 43.76% | 37.74% | 40.31% | 59.18% | 75.17% | 0.331 | 52.98 | 418750ms |
| volume | v00067 | sizing | 57.66% | 57.68% | 57.63% | 45.03% | 47.45% | 46.01% | 57.25% | 70.66% | 0.419 | 65 | 291s |
| volume | v00068 | sizing | 57.46% | 58.14% | 56.78% | 44.95% | 47.07% | 45.82% | 58.27% | 70.08% | 0.427 | 64.25 | 284500ms |
| canonical | k00457 | confidence | 57.38% | 58.14% | 56.61% | 46.99% | 50.39% | 48.3% | 57.01% | 71.39% | 0.402 | 66.22 | 265s |
| canonical | k00330 | sizing | 57.14% | 58.04% | 56.25% | 44.97% | 50.21% | 47.18% | 58.13% | 68.62% | 0.462 | 68.9 | 266500ms |
| volume | v00066 | sizing | 56.23% | 57.57% | 54.89% | 44.2% | 49.58% | 46.59% | 58.01% | 67.23% | 0.488 | 68.67 | 280250ms |

## Best fit candidates

| family | id | strategy | objective | median | P10 | precision | recall | F1 | agreement | cleanliness | noise/signal | signals/day | timing error P50 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| volume | v00073 | sizing | 61.25% | 61.49% | 61.02% | 52.66% | 42.13% | 48.27% | 58.86% | 83.27% | 0.201 | 73.12 | 326s |
| canonical | k00457 | confidence | 61.24% | 61.75% | 60.72% | 52.1% | 50.09% | 52.71% | 57.68% | 78.42% | 0.275 | 88.47 | 258s |
| volume | v00065 | sizing | 61.18% | 62.23% | 60.12% | 50.79% | 49.66% | 52.58% | 58.17% | 75.57% | 0.323 | 92.71 | 263s |
| volume | v00066 | sizing | 61.02% | 61.73% | 60.31% | 50.35% | 49.67% | 52.27% | 58.22% | 75.54% | 0.324 | 93.24 | 262s |
| volume | v00067 | sizing | 61.01% | 61.91% | 60.1% | 50.61% | 49.06% | 51.73% | 57.56% | 76.22% | 0.312 | 90.29 | 272s |
| volume | v00074 | sizing | 61% | 61.25% | 60.74% | 51.88% | 42.44% | 47.83% | 58.79% | 81.13% | 0.233 | 73.88 | 312s |
| volume | v00068 | sizing | 60.94% | 61.76% | 60.11% | 50.73% | 47.75% | 51.41% | 57.75% | 77.2% | 0.295 | 88.76 | 279s |
| canonical | k00345 | sizing | 60.93% | 61.87% | 59.99% | 50.15% | 50.31% | 52.23% | 57.8% | 74.9% | 0.335 | 93.88 | 257s |
| canonical | k00458 | confidence | 60.92% | 61.31% | 60.53% | 52.48% | 41.96% | 48.6% | 57.89% | 82.16% | 0.217 | 74.53 | 319s |
| canonical | k00385 | confidence | 60.92% | 61.45% | 60.39% | 49.82% | 37.35% | 44.81% | 59.54% | 83.33% | 0.2 | 72 | 389s |
| canonical | k00459 | confidence | 60.92% | 61.67% | 60.16% | 52.4% | 42.78% | 48.92% | 57.19% | 81.8% | 0.222 | 75.65 | 308s |
| volume | v00129 | confidence | 60.91% | 61.44% | 60.37% | 49.29% | 34.42% | 43.05% | 59.61% | 85.78% | 0.166 | 67 | 402500ms |

## Finalist parameters

| family | id | agreement | efficiency | ER volume EMA/power | fast | slow | power | volume | cap | volume power | base threshold | state mode | threshold mode | noise lookback | noise multiplier | buy max | sell max | buy sigma | sell sigma |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| volume | v00129 | confidence | 1230229ms | 60000ms / 0.529 | 1544713ms | 1698128ms | 1.006 | 1m | 1 | 0 | 1.005 bps/hour | hold | static | 8636054ms | 8.448 | 99.97% | 99.9% | 289.318 | 297.878 |
| canonical | k00459 | confidence | 1202152ms | 1m / 0 | 664729ms | 2558602ms | 1.914 | 1m | 1 | 0 | 0.05 bps/hour | hold | static | 7895320ms | 17.285 | 100% | 99.85% | 66.122 | 300 |

## Finalist confirmation parameters

| candidate | mix | minimum quality | acceleration lookback | distance lookback | acceleration weight | overextension weight | bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| v00129 | 0% | 0% | 60000ms | 21600000ms | 0 | 0 | -2.877 |
| k00459 | 100% | 95% | 4092127ms | 754173ms | 0 | 0 | -5 |

| candidate | hysteresis release | slow EMA | EMA tolerance | EMA weight/gate | RSI period | RSI tolerance/weight | DMI period | DMI weight | ADX threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v00129 | 9.66% | 9955971ms | 1.735 bps/h | 0 / 0% | 5956951ms | 3.51 / 0 | 2054925ms | 0 | 42.52 |
| k00459 | 91.29% | 540971ms | 73.078 bps/h | 4.92 / 47.13% | 9655087ms | 19.6 / 0 | 3483510ms | 0.453 | 14.26 |

## Finalist mean-reversion parameters

| candidate | blend | mean | volatility | switch threshold |
|---|---:|---:|---:|---:|
| v00129 | 0% | 1m | 1m | 0.25σ |
| k00459 | 100% | 60160ms | 80240ms | 0.274σ |

## Holdout cases

| candidate | scale/window | score | precision | recall | F1 | agreement | cleanliness | noise/signal | signals/day | timing error P50 | timing error P90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v00129 | 1s:test-1 | 49.5% | 18.56% | 36.43% | 24.59% | 58.89% | 46.23% | 1.163 | 17.67 | 13m | 2995000ms |
| v00129 | 1s:test-2 | 56.98% | 32.83% | 39.06% | 35.67% | 60.5% | 67.71% | 0.477 | 37.17 | 587s | 2203200ms |
| k00459 | 1s:test-1 | 47.18% | 20.58% | 33.53% | 25.5% | 52.7% | 52.27% | 0.913 | 14.67 | 956500ms | 3555500ms |
| k00459 | 1s:test-2 | 56.18% | 35.28% | 44.88% | 39.51% | 58.11% | 67.05% | 0.492 | 39.74 | 486s | 2129s |
