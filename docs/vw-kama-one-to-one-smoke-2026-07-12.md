# Volume-weighted KAMA oracle approximation search

Generated: 2026-07-12T09:55:43.171Z
Raw results: `data/benchmarks/vw-kama-one-to-one-smoke-2026-07-12.jsonl`

The fit ranking never reads validation or holdout scores. The best validated volume-aware and canonical (`volumePower=0`) candidates are the only candidates evaluated on holdout.

## Data and objective

- Source: `data/historical/spot-btcusdt/btcusdt/1s` (1s); target scales: 1s, 15s, 1m.
- Windows: fit 2022-06-13..2022-06-15; validation 2024-02-24..2024-02-26; test 2025-02-14..2025-02-16.
- Each continuous segment reserves 3d before scoring.
- Signal: completed-candle volume-weighted KAMA derivative rate; candidates either go flat inside the deadband or hold their prior exposure until the opposite threshold.
- Signal memory: after the first signal, a candidate state change emits only when the current close is strictly more than 17.5 bps from the last emitted signal price; rejected changes retain the prior state. This is the same friction used by the oracle.
- Matching is one chronological one-to-one alignment by resulting state. It maximizes total timing credit, so extra candidate transitions reduce precision and uncovered oracle transitions reduce recall.
- Case score weights timing-credited transition F1 80% and graded oracle-state agreement 20% (exact 1, flat/directional 0.5, opposite 0). Trading returns and execution prices are neither computed nor ranked.
- Candidate objective equally weights the median and P10 case score; every scale/window case has equal weight.

## Holdout finalists

| family | id | objective | median | P10 | precision | recall | F1 | agreement | signals/day | timing error P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| volume | v0001 | 26% | 27.24% | 24.76% | 11.98% | 36.76% | 19.34% | 61.92% | 30.67 | 810s |
| canonical | k0001 | 26% | 27.24% | 24.76% | 11.98% | 36.76% | 19.34% | 61.92% | 30.67 | 810s |

## Validation finalists

| family | id | objective | median | P10 | precision | recall | F1 | agreement | signals/day | timing error P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical | k0001 | 31.3% | 32.59% | 30.02% | 16.3% | 44.82% | 23.9% | 66.32% | 47.67 | 645s |
| volume | v0001 | 31.3% | 32.59% | 30.02% | 16.3% | 44.82% | 23.9% | 66.32% | 47.67 | 645s |

## Best fit candidates

| family | id | objective | median | P10 | precision | recall | F1 | agreement | signals/day | timing error P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical | k0001 | 58.06% | 60.17% | 55.94% | 53.02% | 67.36% | 59.33% | 63.53% | 386.67 | 2m |
| volume | v0001 | 58.06% | 60.17% | 55.94% | 53.02% | 67.36% | 59.33% | 63.53% | 386.67 | 2m |

## Finalist parameters

| family | id | efficiency | fast | slow | power | volume | cap | volume power | deadband | mode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| volume | v0001 | 842179ms | 1678833ms | 9187995ms | 0.49 | 7820651ms | 2.65 | 0 | 67.567 bps/hour | flat |
| canonical | k0001 | 842179ms | 1678833ms | 9187995ms | 0.49 | 7820651ms | 2.65 | 0 | 67.567 bps/hour | flat |

## Holdout cases

| candidate | scale/window | score | precision | recall | F1 | agreement | signals/day | timing error P50 | timing error P90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v0001 | 1s:test-1 | 28.56% | 11.98% | 50.11% | 19.34% | 65.45% | 46 | 498s | 3677400ms |
| v0001 | 15s:test-1 | 27.24% | 13.19% | 36.76% | 19.41% | 58.56% | 30.67 | 810s | 3397500ms |
| v0001 | 1m:test-1 | 24.14% | 10.04% | 27.36% | 14.69% | 61.92% | 26.33 | 19m | 3108000ms |
| k0001 | 1s:test-1 | 28.56% | 11.98% | 50.11% | 19.34% | 65.45% | 46 | 498s | 3677400ms |
| k0001 | 15s:test-1 | 27.24% | 13.19% | 36.76% | 19.41% | 58.56% | 30.67 | 810s | 3397500ms |
| k0001 | 1m:test-1 | 24.14% | 10.04% | 27.36% | 14.69% | 61.92% | 26.33 | 19m | 3108000ms |
