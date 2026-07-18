# VW-KAMA value-search performance pass — 2026-07-18

This pass optimized the exact value-distillation implementation without changing its objective,
fee equations, `H`-step holding semantics, segment-ending `T`, or deterministic lower-exposure tie break.

## Retained changes

- Algebra: `b2=0` evaluates the uniform truncated-exponential partition in constant time with a
  stable geometric-series formula. The optional quadratic distribution uses a stable exact grid
  log-sum-exp and the oracle's resident second moment.
- Bellman recurrence: separable buy/sell fee factors use prefix and suffix maxima, reducing the
  continuation update from `O(G²)` to `O(G)` on CPU and `O(log G)` scan depth on CUDA.
- Holding values: CPU reuses marked outcomes and continuation prefixes. CUDA computes one-step
  values in parallel, builds continuation prefixes, and assembles all `H`-step holding returns.
  A segment-ending `T` uses the independent backward residue chains; a finite rolling `T` launches
  all scored starting times concurrently for each holding-period block.
- Fitness specialization: evolutionary value fitness calculates only strategy state and weighted
  cross-entropy. Transition alignment, lag percentiles, oracle/strategy returns, drawdown, and
  turnover are deferred to diagnostic evaluation.
- Candidate-independent work: weighted oracle entropy, opportunity totals, weight totals, and the
  oracle return path are computed once per case rather than once per candidate.
- Residency and transfers: case columns, value means/second moments/weights, and unit
  strategy-temperature scales are converted to aligned Float32 structure-of-arrays and uploaded
  once. Candidate temperature, quadratic scale, and quadratic-volatility window travel in the
  packed parameter row; each CUDA lane maintains its own causal rolling volatility moments.
  Generations transfer only parameter and result buffers. The device LRU budget is controlled by
  `VW_KAMA_CUDA_CASE_CACHE_BYTES` and defaults to 1.5 GB.
- Layout: recurrent change storage uses packed per-candidate offsets, eliminating the previous
  `candidateCount × maximumPeriod × 2` allocation. CPU policy rows are stored only for the scored
  suffix.
- Scheduling: independent resident cases launch on non-blocking CUDA streams. Auto routing uses
  candidate-case lane count for evaluation and estimated candle-grid cells for oracle preparation.
- Diagnostics: a bounded first-pass transition capture normally avoids rerunning the full strategy
  kernel. Exact compact replay remains as an overflow fallback.
- Lifecycle: prepared host cases and resident device cases are cached across generations and
  released at search shutdown; device-memory admission prevents oversized oracle grids from
  exhausting the GPU.

## Measurements

Device: NVIDIA GeForce RTX 3060 Laptop GPU. Oracle workload: 20,000 synthetic candles, 19,000
scored candles, grid 101, `H=300`, with `T` at the segment end. Times vary slightly with clocks; ratios below use repeated warm
runs from the same implementation steps.

| Oracle implementation | Time | Relative to original |
| --- | ---: | ---: |
| Original CPU `O(TG²)` | 4,846 ms | 1.0× |
| CPU separable Bellman scan | ~700 ms | 6.9× |
| CPU outcome/continuation rings | ~365 ms | 13.3× |
| Original CUDA one-block kernel | 5,013 ms kernel | 1.0× |
| CUDA separable parallel fee scan | 4,395 ms kernel | 1.14× |
| CUDA `H` residue chains | 1,141 ms kernel | 4.4× |
| CUDA parallel steps + prefix assembly + `H` chains | ~140 ms kernel | 35.8× |

For 384 mixed-feature candidates × 20,000 candles:

| Candidate path | Time |
| --- | ---: |
| Full diagnostics after first-pass transition capture | ~246 ms kernel |
| Resident fitness-only evaluation | ~63 ms kernel |
| Four resident cases, serial launches | ~280 ms wall |
| Four resident cases, stream scheduled | ~79 ms wall |

The base-feature transfer microbenchmark fell from 54.3 ms one-shot wall time to 47.7 ms after
case residency. Four-case scheduling improved mixed-feature wall time by about 3.5×. Fitness and
diagnostic weighted cross-entropy had zero measured drift.

With the new one-candle default, the same 20,000-candle/grid-101 oracle takes about 315 ms on the
optimized CPU but 2.8 s in the CUDA kernel because `H=1` exposes only one sequential backward
residue chain. Auto routing therefore keeps fixed `H<8` oracle preparation on CPU while candidate
fitness remains on CUDA. At grid 101, the earlier constant-coefficient benchmark increased the
384-candidate fitness kernel from about 64 ms (`b2=0`) to 99 ms (`b2=-0.2`); four-case stream
scheduling still delivered 3.66× wall-time throughput. The production parameter is now a
nonnegative scale with the causal per-candle coefficient `b2=-b2'*v²`, where `v` is log-return
volatility over a separately configured trailing duration.

The concave quadratic normalizer now sums a curvature-derived, mode-centered fixed window using
the quadratic's constant second difference. Its tail bound is below the target numeric precision;
the zero-coefficient geometric path and positive-quadratic full-grid fallback are unchanged. On
20,000 candles, grid 150, `H=300`, segment-ending `T`, 384 candidates, four cases, and quadratic scale 200,000, the
resident fitness kernel improved from 120.2 ms to 107.5 ms (10.6%), and the scheduled four-case
kernel improved from 127.1 ms to 114.5 ms (9.9%), with zero fitness/diagnostic loss drift.

## Rejected changes

- A serial CUDA prefix/suffix fee scan was removed after increasing the oracle kernel from about
  5.01s to 12.45s. The retained implementation uses a parallel scan and precomputed fee factors.
- Sorting candidates into execution-feature buckets was removed after making the mixed-feature
  kernel slower (about 63.0ms to 67.3ms). Packed rings remain, but candidate order is preserved.
- Candidate blocks of 64 and 128 threads did not meaningfully improve the mixed workload; 128 was
  slower. The 32-thread launch keeps more independent blocks for small evolutionary batches.
- A bidirectional mode-outward tail walk was removed after increasing the same nonzero-quadratic
  fitness kernel from 120.2 ms to 316.3 ms. Its variable loop lengths caused warp divergence; the
  retained fixed-width forward recurrence keeps the tail bound without that scheduling penalty.

## Verification

Run:

```bash
npm run build:cuda
npm run benchmark:kama:value:cuda -- 384 20000 101 300 20000 4
# final optional argument selects nonnegative quadratic scale b2'
npm run benchmark:kama:value:cuda -- 384 20000 101 300 20000 4 200000
npm test -w @trading/bot-algo
npm run typecheck
npm run build
```

Tests compare CPU and CUDA oracle means, moments, entropy, opportunity, retained probabilities,
modal exposure, exact policy targets, fitness-only/scheduled cross-entropy, and strategy/oracle
return diagnostics.
