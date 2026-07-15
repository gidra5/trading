# VW-KAMA per-window optimization

Generated: 2026-07-15T08:47:19.325Z
Score version: 2
Algorithm: de; 2048 population; 64 generations × 2 restart(s); 5 refinement round(s); 1s.
Workers: 12; single-window CUDA candidate batches.
Windows: regime-flat-2026-04 (2026-04-22..2026-04-28).
Warm start: `data/benchmarks/vw-kama-global-score-v2-de-2026-07-12.jsonl`, `data/benchmarks/vw-kama-window-presets.json`.

## CUDA stress result

Target: the score-v2 finalist `global-clean-k0050` on `regime-flat-2026-04` at 1s. Its exact CPU score-v2 baseline is 40.1509%, making it the weakest seven-day 1s candidate/window pair in the CUDA comparison matrix.

- Device: NVIDIA GeForce RTX 3060 Laptop GPU; CUDA was forced for every optimizer batch.
- Work: 2,048 members × 64 generations × 2 independent restarts, plus a 3,442-candidate seeded initialization, five 96-candidate refinement batches, and a 2,058-candidate final selection.
- Approximate total: 270,000 complete candidate evaluations and 163.4 billion scored candidate-candles.
- Wall time: 1,165,340ms (19m25s), about 140.2 million scored candidate-candles/s end to end.
- Representative 2,048-candidate batches took 7.1–7.6s, of which 6.6–7.1s was CUDA kernel time.
- All 128 generations, both restarts, all refinements, and final selection completed without CUDA, allocation, alignment, or evaluator errors. Batch time did not increase over the run.

## CPU verification of the CUDA winner

| evaluation | score v2 | F1 | agreement | cleanliness | signals | matched |
|---|---:|---:|---:|---:|---:|---:|
| original `global-clean-k0050`, CPU Float64 | 40.1509% | 33.0715% | 52.8905% | 44.4079% | 304 | 135 |
| deep winner, CUDA Float32 | 50.7650% | 40.6077% | 65.6608% | 67.0213% | 188 | 126 |
| deep winner, CPU Float64 | 48.8995% | 39.0140% | 62.6898% | 66.8421% | 190 | 127 |

The selected winner retains an exact CPU improvement of 8.7486 score points over the bad baseline. CUDA overstates this winner by 1.8655 points, mostly through +2.9710 agreement points and +1.5937 F1 points. This is larger than the fixed-candidate stress matrix: a deep search can preferentially discover parameters near Float32-sensitive recurrent thresholds even when ordinary configurations have small drift.

This supports keeping the entire population search on CUDA for throughput, but it also provides concrete evidence that one final CPU evaluation is useful. Verifying only the final winner preserves essentially all GPU speedup; widening CPU verification to a shortlist is only necessary when the exact winner decision matters.

These are hindsight upper-bound configurations for comparison, not deployable validation results.

Every selected preset is scored against the global candidates at the same window and candle scale; the global score is therefore a hard lower bound.

Existing presets are re-scored with the current evaluator before replacement; `incumbent` is therefore directly comparable even when evaluator semantics changed.

| window | scale | score | incumbent | delta | preset |
|---|---:|---:|---:|---:|---|
| regime-flat-2026-04 | 1s | 50.77% | — | — | window-regime-flat-2026-04-1s |

## Selected parameters

| preset | ER | ER volume EMA/power | fast | slow | KAMA power | post-ER volume EMA/cap/power | threshold | state/agreement | confirmation mix/gate | mean reversion |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| window-regime-flat-2026-04-1s | 2258280ms | 1m / 0 | 1090ms | 1500837ms | 8.948 | 1m / 1 / 0 | 0.13 bps/h + static/0.446 | hold/sizing | 0.953 / 0.087 | 0.997 · 300000ms / 31242036ms @ 1.095σ |
