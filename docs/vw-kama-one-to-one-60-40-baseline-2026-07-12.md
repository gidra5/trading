# Volume-weighted KAMA oracle approximation search

Generated: 2026-07-12T10:32:13.380Z
Raw results: `data/benchmarks/vw-kama-one-to-one-60-40-baseline-2026-07-12.jsonl`

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
| volume | v0001 | 26.77% | 31.76% | 21.78% | 10.53% | 29.54% | 15.69% | 57.09% | 28.33 | 990s |
| canonical | k0001 | 26.77% | 31.76% | 21.78% | 10.53% | 29.54% | 15.69% | 57.09% | 28.33 | 990s |

## Validation finalists

| family | id | objective | median | P10 | precision | recall | F1 | agreement | signals/day | timing error P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical | k0001 | 34.59% | 44.8% | 24.37% | 22.59% | 52.03% | 31.5% | 63.82% | 82.75 | 450s |
| volume | v0001 | 34.59% | 44.8% | 24.37% | 22.59% | 52.03% | 31.5% | 63.82% | 82.75 | 450s |

## Best fit candidates

| family | id | objective | median | P10 | precision | recall | F1 | agreement | signals/day | timing error P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical | k0001 | 32.91% | 44.69% | 21.13% | 23.25% | 52.74% | 32.28% | 61.4% | 93.98 | 7m |
| volume | v0001 | 32.91% | 44.69% | 21.13% | 23.25% | 52.74% | 32.28% | 61.4% | 93.98 | 7m |

## Finalist parameters

| family | id | efficiency | fast | slow | power | volume | cap | volume power | deadband | mode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| volume | v0001 | 842179ms | 1678833ms | 9187995ms | 0.49 | 7820651ms | 2.65 | 0 | 67.567 bps/hour | flat |
| canonical | k0001 | 842179ms | 1678833ms | 9187995ms | 0.49 | 7820651ms | 2.65 | 0 | 67.567 bps/hour | flat |

## Holdout cases

| candidate | scale/window | score | precision | recall | F1 | agreement | signals/day | timing error P50 | timing error P90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v0001 | 1s:test-1 | 36.23% | 10.99% | 42.73% | 17.48% | 64.35% | 35 | 746s | 3257000ms |
| v0001 | 1s:test-2 | 43.2% | 21.1% | 60.99% | 31.35% | 60.98% | 90.31 | 326500ms | 1519700ms |
| v0001 | 5s:test-1 | 31.09% | 11.26% | 38.56% | 17.43% | 51.58% | 30.83 | 695s | 3212000ms |
| v0001 | 5s:test-2 | 42.54% | 20.09% | 54.65% | 29.38% | 62.3% | 79.17 | 440s | 1805s |
| v0001 | 15s:test-1 | 32.44% | 11.32% | 33.74% | 16.95% | 55.66% | 25.83 | 14m | 63m |
| v0001 | 15s:test-2 | 40.96% | 18.94% | 49.66% | 27.42% | 61.26% | 70.6 | 525s | 2010s |
| v0001 | 1m:test-1 | 30.88% | 9.07% | 25.32% | 13.36% | 57.16% | 22.33 | 19m | 58m |
| v0001 | 1m:test-2 | 38.64% | 16.07% | 42.9% | 23.38% | 61.53% | 59.29 | 11m | 38m |
| v0001 | 5m:test-1 | 26.04% | 5.56% | 12.55% | 7.71% | 53.53% | 16.17 | 25m | 5040000ms |
| v0001 | 5m:test-2 | 32.71% | 10.08% | 25.33% | 14.42% | 60.15% | 37.64 | 20m | 65m |
| v0001 | 15m:test-1 | 23.77% | 5.03% | 5.36% | 5.19% | 51.65% | 5.5 | 45m | 105m |
| v0001 | 15m:test-2 | 27.87% | 6.3% | 12.75% | 8.43% | 57.03% | 20.1 | 30m | 90m |
| v0001 | 1h:test-1 | 20.64% | 0.22% | 0.1% | 0.14% | 51.39% | 1.17 | 1h | 1h |
| v0001 | 1h:test-2 | 20.93% | 0.32% | 0.22% | 0.26% | 51.93% | 3.43 | 1h | 2h |
| k0001 | 1s:test-1 | 36.23% | 10.99% | 42.73% | 17.48% | 64.35% | 35 | 746s | 3257000ms |
| k0001 | 1s:test-2 | 43.2% | 21.1% | 60.99% | 31.35% | 60.98% | 90.31 | 326500ms | 1519700ms |
| k0001 | 5s:test-1 | 31.09% | 11.26% | 38.56% | 17.43% | 51.58% | 30.83 | 695s | 3212000ms |
| k0001 | 5s:test-2 | 42.54% | 20.09% | 54.65% | 29.38% | 62.3% | 79.17 | 440s | 1805s |
| k0001 | 15s:test-1 | 32.44% | 11.32% | 33.74% | 16.95% | 55.66% | 25.83 | 14m | 63m |
| k0001 | 15s:test-2 | 40.96% | 18.94% | 49.66% | 27.42% | 61.26% | 70.6 | 525s | 2010s |
| k0001 | 1m:test-1 | 30.88% | 9.07% | 25.32% | 13.36% | 57.16% | 22.33 | 19m | 58m |
| k0001 | 1m:test-2 | 38.64% | 16.07% | 42.9% | 23.38% | 61.53% | 59.29 | 11m | 38m |
| k0001 | 5m:test-1 | 26.04% | 5.56% | 12.55% | 7.71% | 53.53% | 16.17 | 25m | 5040000ms |
| k0001 | 5m:test-2 | 32.71% | 10.08% | 25.33% | 14.42% | 60.15% | 37.64 | 20m | 65m |
| k0001 | 15m:test-1 | 23.77% | 5.03% | 5.36% | 5.19% | 51.65% | 5.5 | 45m | 105m |
| k0001 | 15m:test-2 | 27.87% | 6.3% | 12.75% | 8.43% | 57.03% | 20.1 | 30m | 90m |
| k0001 | 1h:test-1 | 20.64% | 0.22% | 0.1% | 0.14% | 51.39% | 1.17 | 1h | 1h |
| k0001 | 1h:test-2 | 20.93% | 0.32% | 0.22% | 0.26% | 51.93% | 3.43 | 1h | 2h |
