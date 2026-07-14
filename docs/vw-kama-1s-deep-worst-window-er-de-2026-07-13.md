# VW-KAMA per-window optimization

Generated: 2026-07-13T17:15:04.339Z
Score version: 2
Algorithm: de; 2048 population; 64 generations × 2 restart(s); 5 refinement round(s); 1s.
Workers: 12; candidate-parallel single-window search.
Windows: shape-flat-low-bias-low-2025-07 (2025-07-04..2025-07-06).
Warm start: `data/benchmarks/vw-kama-regime-confirmations-multicore-de-2026-07-13.jsonl`, `data/benchmarks/vw-kama-window-presets.json`.

These are hindsight upper-bound configurations for comparison, not deployable validation results.

Every selected preset is scored against the global candidates at the same window and candle scale; the global score is therefore a hard lower bound.

Existing presets are re-scored with the current evaluator before replacement; `incumbent` is therefore directly comparable even when evaluator semantics changed.

| window | scale | score | incumbent | delta | preset |
|---|---:|---:|---:|---:|---|
| shape-flat-low-bias-low-2025-07 | 1s | 55.62% | 55.12% | +0.51% | window-shape-flat-low-bias-low-2025-07-1s |

## Selected parameters

| preset | ER | ER volume EMA/power | fast | slow | KAMA power | post-ER volume EMA/cap/power | threshold | state/agreement | confirmation mix/gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| window-shape-flat-low-bias-low-2025-07-1s | 112751ms | 468377ms / 0.05 | 2100000ms | 2152705ms | 0.854 | 1m / 1 / 0 | 37.795 bps/h + static/0 | hold/confidence | 0.952 / 0.035 |
