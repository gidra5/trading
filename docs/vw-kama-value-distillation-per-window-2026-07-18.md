# VW-KAMA per-window optimization

Generated: 2026-07-18T11:08:16.189Z
Score version: 8
Algorithm: de; 384 population; 12 generations × 2 restart(s); 3 refinement round(s); 1m, 5m, 15m, 1h.
Workers: 4; serial windows with CUDA candidate batches.
Windows: fit-1 (2025-03-19..2025-05-17), fit-2 (2025-05-18..2025-07-16), fit-3 (2025-07-17..2025-09-14), fit-4 (2025-09-15..2025-11-13).
Warm start: `data/benchmarks/vw-kama-global-presets.json`.

These are hindsight upper-bound configurations for comparison, not deployable validation results.

Every selected preset is scored against the global candidates at the same window and candle scale; the global score is therefore a hard lower bound.

Existing presets are re-scored with the current evaluator before replacement; `incumbent` is therefore directly comparable even when evaluator semantics changed.

| window | scale | cross-entropy | incumbent | improvement | preset |
|---|---:|---:|---:|---:|---|
| fit-1 | 1m | CE 5.01058 | — | — | window-fit-1-1m |
| fit-1 | 5m | CE 5.00997 | — | — | window-fit-1-5m |
| fit-1 | 15m | CE 5.01808 | — | — | window-fit-1-15m |
| fit-1 | 1h | CE 6.59066 | — | — | window-fit-1-1h |
| fit-2 | 1m | CE 5.01044 | — | — | window-fit-2-1m |
| fit-2 | 5m | CE 5.01019 | — | — | window-fit-2-5m |
| fit-2 | 15m | CE 5.01712 | — | — | window-fit-2-15m |
| fit-2 | 1h | CE 6.39895 | — | — | window-fit-2-1h |
| fit-3 | 1m | CE 5.01063 | — | — | window-fit-3-1m |
| fit-3 | 5m | CE 5.01058 | — | — | window-fit-3-5m |
| fit-3 | 15m | CE 5.01701 | — | — | window-fit-3-15m |
| fit-3 | 1h | CE 6.06852 | — | — | window-fit-3-1h |
| fit-4 | 1m | CE 5.01051 | — | — | window-fit-4-1m |
| fit-4 | 5m | CE 5.01049 | — | — | window-fit-4-5m |
| fit-4 | 15m | CE 5.07253 | — | — | window-fit-4-15m |
| fit-4 | 1h | CE 6.38818 | — | — | window-fit-4-1h |

## Selected parameters

| preset | ER | ER volume EMA/power | fast | slow | KAMA power | post-ER volume EMA/cap/power | threshold | state/agreement | confirmation mix/gate | mean reversion |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| window-fit-1-1m | 7200000ms | 60000ms / 2.736 | 1000ms | 86400000ms | 5 | 317239ms / 1 / 3 | 2.929 bps/h + noise×1.606 | hold/sizing | 0.834 / 0 | 1m ER / 1m–5m KAMA / 1m vol @ 0σ suppress / 0σ reverse |
| window-fit-1-5m | 5168402ms | 1m / 0 | 1800000ms | 70996315ms | 3.865 | 1m / 1 / 0 | 582.501 bps/h + noise×1.506 | hysteresis/sizing | 0 / 0 | 5151226ms ER / 21600000ms–21600000ms KAMA / 86400000ms vol @ 3σ suppress / 6σ reverse |
| window-fit-1-15m | 7200000ms | 410788ms / 2.421 | 1800000ms | 86400000ms | 5 | 1973052ms / 1 / 3 | 23.151 bps/h + noise×0.721 | hold/sizing | 0.05 / 0 | 1m ER / 1m–5m KAMA / 1m vol @ 0σ suppress / 0σ reverse |
| window-fit-1-1h | 7200000ms | 410788ms / 2.421 | 1800000ms | 86400000ms | 5 | 1973052ms / 1 / 3 | 23.151 bps/h + noise×0.721 | hold/sizing | 0.05 / 0 | 1m ER / 1m–5m KAMA / 1m vol @ 0σ suppress / 0σ reverse |
| window-fit-2-1m | 6879014ms | 1m / 0 | 136773ms | 76602937ms | 5 | 85919ms / 1 / 0.383 | 2000 bps/h + noise×7.965 | hold/sizing | 0.525 / 0 | 1m ER / 1m–5m KAMA / 1m vol @ 0σ suppress / 0σ reverse |
| window-fit-2-5m | 7200000ms | 1m / 0 | 1800000ms | 12638678ms | 5 | 60000ms / 4.641 / 3 | 0.078 bps/h + noise×8 | hold/confidence | 0.05 / 0 | 1m ER / 1m–5m KAMA / 1m vol @ 0σ suppress / 0σ reverse |
| window-fit-2-15m | 3884604ms | 405804ms / 0.666 | 1618765ms | 9573021ms | 5 | 280416ms / 1 / 3 | 0.05 bps/h + noise×8 | hold/confidence | 0.05 / 0 | 1m ER / 1m–5m KAMA / 1m vol @ 0σ suppress / 0σ reverse |
| window-fit-2-1h | 7200000ms | 60000ms / 0.427 | 1800000ms | 86400000ms | 5 | 60000ms / 1 / 2.532 | 0.223 bps/h + noise×7.813 | hysteresis/sizing | 0.518 / 0 | 86400000ms ER / 21600000ms–86400000ms KAMA / 86400000ms vol @ 3σ suppress / 3.628σ reverse |
| window-fit-3-1m | 4508120ms | 1m / 0 | 1439902ms | 3692122ms | 5 | 1m / 1 / 0 | 0.05 bps/h + noise×2.307 | hold/sizing | 0.895 / 1 | 759747ms ER / 62820ms–86400000ms KAMA / 1993824ms vol @ 1.363σ suppress / 5.264σ reverse |
| window-fit-3-5m | 4887146ms | 1m / 0 | 1800000ms | 86400000ms | 3.655 | 1m / 1 / 0 | 0.05 bps/h + noise×8 | flat/confidence | 0.144 / 0.416 | 1m ER / 1m–5m KAMA / 1m vol @ 0σ suppress / 0σ reverse |
| window-fit-3-15m | 6137232ms | 1m / 0 | 1800000ms | 51586712ms | 4.995 | 4107115ms / 1 / 2.954 | 0.05 bps/h + noise×7.872 | flat/confidence | 0.05 / 0 | 2993905ms ER / 315353ms–543344ms KAMA / 4441187ms vol @ 1.002σ suppress / 1.784σ reverse |
| window-fit-3-1h | 6187401ms | 60000ms / 3.999 | 1800000ms | 86400000ms | 5 | 60473ms / 1 / 3 | 0.05 bps/h + noise×3.151 | hold/sizing | 0.081 / 0 | 1m ER / 1m–5m KAMA / 1m vol @ 0σ suppress / 0σ reverse |
| window-fit-4-1m | 7200000ms | 60000ms / 3.955 | 720785ms | 47609425ms | 3.146 | 3698077ms / 8.784 / 0.836 | 1.597 bps/h + noise×8 | flat/confidence | 0.05 / 0.931 | 5287394ms ER / 21600000ms–86400000ms KAMA / 11856997ms vol @ 2.364σ suppress / 5.117σ reverse |
| window-fit-4-5m | 2141751ms | 60000ms / 1.419 | 1738082ms | 67476022ms | 5 | 1m / 1 / 0 | 0.05 bps/h + noise×7.91 | flat/confidence | 0.901 / 0.242 | 1m ER / 1m–5m KAMA / 1m vol @ 0σ suppress / 0σ reverse |
| window-fit-4-15m | 7200000ms | 1m / 0 | 1800000ms | 86400000ms | 5 | 60000ms / 1 / 0.585 | 0.05 bps/h + noise×8 | hold/sizing | 0.05 / 0 | 1m ER / 1m–5m KAMA / 1m vol @ 0σ suppress / 0σ reverse |
| window-fit-4-1h | 7200000ms | 1m / 0 | 1800000ms | 24165856ms | 5 | 164680ms / 1 / 2.254 | 0.05 bps/h + noise×8 | hold/confidence | 0.076 / 0 | 1m ER / 1m–5m KAMA / 1m vol @ 0σ suppress / 0σ reverse |

## Selected distribution and friction parameters

| preset | noise response | candidate friction | strategy temperature | quadratic scale | quadratic volatility window |
|---|---|---:|---:|---:|---:|
| window-fit-1-1m | inverse | 1 | 0.082175901 | 0 | 6714353ms |
| window-fit-1-5m | inverse | 0.85657 | 0.1 | 0 | 685780ms |
| window-fit-1-15m | proportional | 0 | 0.1 | 0 | 1d |
| window-fit-1-1h | proportional | 0 | 0.1 | 0 | 1d |
| window-fit-2-1m | proportional | 0.08788 | 0.08446115 | 0 | 49867572ms |
| window-fit-2-5m | proportional | 0 | 0.1 | 0 | 1d |
| window-fit-2-15m | inverse | 0.02797 | 0.1 | 0 | 1d |
| window-fit-2-1h | inverse | 0.32973 | 0.1 | 899158.52707 | 1m |
| window-fit-3-1m | proportional | 1 | 0.099533257 | 0 | 1d |
| window-fit-3-5m | proportional | 1 | 0.08772791 | 0 | 1d |
| window-fit-3-15m | inverse | 0 | 0.1 | 0 | 1d |
| window-fit-3-1h | proportional | 0.00043 | 0.1 | 0 | 63580ms |
| window-fit-4-1m | inverse | 0.01509 | 0.097369125 | 0 | 10623163ms |
| window-fit-4-5m | inverse | 0.94612 | 0.1 | 0 | 1m |
| window-fit-4-15m | inverse | 0.02237 | 0.1 | 0 | 1d |
| window-fit-4-1h | inverse | 0.01119 | 0.1 | 0 | 69987169ms |
