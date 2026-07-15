# VW-KAMA evaluator CPU/shared-memory optimization

Date: 2026-07-14

The exact search evaluator now:

- accumulates exposure agreement in its primary candle loop;
- allocates full state/exposure arrays only when a UI trace is requested;
- prepares oracle states and transitions once per case;
- stores OHLCV data in shared columnar `Float64Array` buffers;
- loads and aggregates candles once in the parent process;
- evaluates dynamic candidate batches with `worker_threads` over the shared data.

The `Float64` object, streaming, and shared-columnar paths produced exactly equal metrics and matched transitions in the regression suite. A separate check over 86,400 real Binance 1-second candles also produced exactly equal metrics and all 31 candidate transitions. The Float32 CUDA path uses the same columns with small accepted numerical drift. Automatic optimization sends batches of four or more candidates to CUDA and keeps smaller batches on CPU because of the measured throughput crossover; `--accelerator cpu` remains available for exact Float64 runs.

## Benchmark

The isolated benchmark used 12 workers, 266 candidates, and one 864,000-candle 1-second case with the global search parameter ranges.

| Measure | Result |
| --- | ---: |
| Total elapsed, including preparation and small validation/test cases | 24.38 s |
| CPU user time | 226.14 s |
| Peak RSS | 1,556,144 KiB |
| Shared fit columns | 49.2 MB |
| Estimated candidate-candle throughput | 9.43 million/s |

The prior child-process run used roughly 9.5 GB RSS and projected 25–28 hours. The isolated evaluator throughput initially projected 8–10 hours, so the old run was replaced at generation 6. The complete two-restart search also pays for alternating folds, periodic full-fit checkpoints, candidate bookkeeping, refinement, and final validation/test scoring. Its observed end-to-end pace instead projects roughly 20–23 hours total, still finishing before the following day's 14:00–16:00 target. The live process has ranged from roughly 2.4 GB to a 4.4 GB observed peak while all three fit datasets occupied the full-fit stage cache, still less than half the prior footprint.
