# Causal MLP exposure predictor

The inspector's `MLP` predictor maps causal multi-resolution market features and
margin state to the 8 learned raw parameters of the conditional exposure distribution.
The existing policy decoder then produces the exposure probabilities, simulated
trades, return series, statistics, and chart values. There is one prediction path:
both full analyses and exact timestamp hover requests use the same evaluator.

## Input and output contract

Every input row has 909 `float32` values in this fixed order:

| Window | Candles | Values per candle | Following fill fraction |
| --- | ---: | ---: | ---: |
| 1 second | 64 | 4 | none (finest resolution) |
| 1 minute | 64 | 4 | 1 |
| 1 hour | 32 | 4 | 1 |
| 1 calendar day | 32 | 4 | 1 |
| 1 calendar month | 16 | 4 | 1 |
| 1 calendar quarter | 16 | 4 | 1 |

Each candle is encoded as close/open log return, upward log deviation from the
geometric open/close midpoint, downward log deviation from that midpoint, and
log volume relative to a slow causal EMA. The last candle in every window is the
candle containing the requested timestamp and may be partially complete. Partial
candles are built causally and hierarchically: visible seconds form the current
minute, completed minutes plus that minute form the current hour, and the same
procedure continues through day, calendar month, and calendar quarter. Each
coarse block is immediately followed by its wall-clock fill fraction in `[0, 1]`.
The latest visible one-second candle is treated as the indivisible finest
observation and therefore has no separate fill fraction. Missing older history
is zero-padded.

The final eight values are fee rate, minimum/maximum usable exposure,
minimum/maximum effective exposure, quote lend rate, quote borrow rate, and asset
borrow rate. Spread is not an MLP parameter. Feature normalization is fitted on
the training split only and is stored inside the ONNX graph.

The graph is exactly `909 -> 16 x 1024 -> 8`. Its output order is
`[c1, c2, b, lambda, betaC1, betaC2, cutoffLower, cutoffUpper]`. It uses LayerNorm, SiLU, small dropout,
residual hidden blocks, and raw-output constraints identical to the shared
TypeScript decoder. `betaX` is derived from fee and temperature; the three
transition sharpnesses are fixed effective-span-scaled decoder constants. The
last two coordinates decode to a lower cutoff in `[effectiveLower, 0]` and an
upper cutoff in `[0, effectiveUpper]`; probability is exactly zero outside that
interval.

Each target is produced by the revised conditional four-segment fitter on CUDA.
The oracle and teacher temporarily use the complete effective range as the
visible action range. The production decoder then truncates the fitted latent
policy to usable leverage. This makes a survival boundary identifiable even
when it lies outside the executable window.
The optimizer continues to fit that complete `[-250, 250]` effective domain,
but fit quality is measured on the executable `[-100, 100]` surface only. The
oracle and fitted rows are re-normalized across the 61 usable action cells, and
cross-entropy, KL divergence, and probability MSE are averaged across the 61
usable current-exposure cells. Those visible-only diagnostics drive rejection,
adaptive refinement, and temporal candidate selection.
The streaming first pass uses batches of up to `2880` examples, `40`
variable-projection steps, `30` batched BFGS steps, `3` structural starts, `31`
sampled states, `61` sampled actions, and `1e-8` tolerance. Variable projection
subtracts the fixed fee transition, eliminates each row's normalization offset,
and analytically solves `[b, lambda, betaC1, betaC2]` while optimizing `c1/c2`.
This is the screenshot design `[1, a, -a²/2]` extended by the two fixed-location
smooth hinges. Only each resulting 4x4 correlated system is solved in float64.
All three projected starts are scored, then only the per-example winner enters
full-memory BFGS. Examples above either quality threshold are compacted into an
immediate bounded refinement lane; only cases that still fail that lane enter
the durable refinement queue. On the current
one-target-per-minute plan a complete day remains one 1,440-row batch; the wider
setting is for multi-day and future per-second jobs.
For each timestamp, the two cutoff labels come from the mandatory `H`-step
equity/maintenance recursion. Zero exposure anchors the feasible interval;
each side is bisected between zero and the effective limit. Any liquidation
during the hold makes that action infeasible. Actions outside maximum effective
leverage are infeasible by definition. The cutoffs are hard masks, so their
training signal is the standardized parameter loss rather than a soft-gate
surrogate.
Training evaluates the fitted teacher and prediction on a fixed, evenly spaced
current-exposure grid; there is no random current-state sampling. Timestamp
weights use the visible-policy distance imbalance

```text
B_t(x) = E[a - x | x] / (E[|a - x| | x] + 1e-6)
W_t = 1e-6 + |mean_{visible x} B_t(x)|
w_t = W_t / mean_batch(W_t)
```

This distinguishes equally probable left/right moves when one tail extends
farther, while symmetric inward pressure at opposite current exposures cancels
before the absolute value is taken. Advice with `|B_t| >= 0.25` also carries a
causal same-side evidence counter. Each earlier same-side important point adds
`0.25` to the next point's multiplier, capped at `4x`. Weak timestamps decay the
evidence with a 15-step half-life; opposite advice or a gap longer than 60 steps
resets it. Persistence is prepared chronologically within each split before
training shuffles examples, so future advice never affects an older example.
It also removes the old average-regret weight files: weights are
derived directly from each fitted teacher during training. The loss is:

```text
cross entropy
+ probability MSE
+ normalized fitted-parameter MSE
+ squared normalized excess entropy
- State MI
- approximate Oracle MI
```

All distribution objectives—cross entropy, probability MSE, excess entropy,
State MI, Oracle MI, and distance-imbalance weighting—use a 151-action by
17-current-state surface spanning only the visible usable range. Teacher and
predicted policies are normalized on that surface, so latent-only cells do not
dilute any optimized metric. Both score policies use the teacher's hard feasible
mask during distribution losses; this prevents exact-zero prediction failures
while the cutoff outputs are still learning. Parameter MSE remains defined over
all eight fitted coordinates, so the complete effective-range score parameters
and survival cutoffs retain direct supervision. Cross entropy, probability MSE, and
parameter MSE have coefficient `1`; excess
entropy, State MI, and Oracle MI have coefficient `0.1`. The MI terms are
rewards, so they are subtracted. Parameter error is standardized per coordinate from training data,
which keeps an equal coefficient meaningful across differently scaled raw
parameters. Training logs also report the mean raw distance-imbalance weight and
its effective-sample ratio; both appear on the live training page.

## Build a dataset

Bootstrap the isolated Python environment once:

```bash
npm run mlp:bootstrap
```

The versioned plan is [training-plan.json](../ml/training-plan.json). It stores the
requested future fee, leverage, effective-leverage, and maintenance sweeps, plus
the conservative values used by this run. Build resumable daily memory-mapped
shards from locally cached 1-second and 1-minute BTCUSDT histories:

```bash
npm run mlp:dataset
```

When only the causal candle schema changes, refresh all feature shards without
re-running the oracle or teacher fitter:

```bash
npm run mlp:features
```

The combined training runner performs this check after teacher refinement and
will not start optimization against stale feature dimensions.

For every non-aggregate inspector window, its first time half is training and its
second half is validation. The latest 30 complete cached days form a held-out test
set. Overlaps are de-duplicated with `test > validation > train` priority, so a
timestamp can never leak between splits. The redundant `fit-full` aggregate is
excluded because its four constituent windows already cover the same dates.

This plan samples one target per minute and trains at 17.5 bps fees, `[-100, 100]`
usable exposure, `[-250, 250]` effective exposure, and 10 bps/hour for the three
configured maintenance rates. Every teacher target stores the 8 fitted raw
parameters and its cross-entropy, KL divergence, probability MSE, iterations,
screened structural starts, and convergence flag. Fits above either quality
threshold are retained as the best available target and also appended to
`teacher-refinement-queue.json` with their timestamp and visible-range
diagnostics. They never block later shards and can be regenerated in a focused
refinement pass. Each shard records its diagnostic bounds, so a later refinement
automatically rebuilds shards whose stored metrics came from an older scope.
Completed shards are retained and reused after a restart. Both oracle preparation and
teacher fitting are CUDA-required for this dataset; there is no silent CPU
fallback that could mix different numeric procedures within one run.

The fitter first solves every timestamp independently, then runs two wide CUDA
warm-start passes from the preceding timestamp. On multi-batch jobs, a
high-priority CUDA stream projects and independently solves batch `N+1` while a
low-priority stream runs the compacted adaptive retries, compatibility fallback,
and temporal selection for batch `N`. The next refinement waits only for the
previous refined tail, so chronological continuity is preserved without holding
the foreground lane idle. Shards are still persisted in timestamp order. The
independent BFGS is a
Triton work queue: one GPU program owns one fit, evaluates the complete
eight-step Armijo ladder branchlessly, and can stop without forcing neighboring
fits to stop or continue. Every 32 iterations the remaining work list is
compacted before the next dispatch. Fits still above the strict quality gate get
96 steps of the established PyTorch BFGS as a selective compatibility fallback;
accepted fits never pay for it. The structural projection already uses batched
`einsum` normal matrices and 4-by-4 solves. A warm result remains eligible
only when its visible 61-by-61-grid cross-entropy is better than, or within
`5e-5 + 2e-5 * abs(crossEntropy)` of, the independent fit. A two-state dynamic
program finally chooses the minimum-total-jump chronological path through the
independent and eligible warm candidates. Continuity is therefore a tie-break
between quality-equivalent fits, not a loss term that can trade away oracle fit.
The final fit of one 2,880-row chunk seeds the first temporal candidate in the
next chunk, so a future per-second day has no artificial parameter discontinuity
at a batching boundary.

Input rows are packed as 160 float32 values: 151 probabilities, two cutoff
coordinates, and seven padding values, making every row 640 bytes (five
128-byte cache lines). Sampled GPU targets use 64 action lanes, or 256 bytes per
state row. The 8-coordinate state uses an 8-by-8 inverse-Hessian layout, while
only the first six smooth score coordinates are optimized by BFGS; the exact
cutoff labels remain fixed.
Four pinned-host batches are prepared ahead of the consumer, so file decoding
and host copies overlap the current GPU batch. This keeps the dense layout
`[fit, state, action]`, with action as the contiguous dimension used by each
warp. Projection uses a local explicit Adam update rather than the shared
multi-tensor optimizer implementation, making concurrent foreground projection
and background autograd fallback safe. A completed refinement stream is also a
lifetime fence for foreground-owned target buffers before the allocator may
reuse them.

On the 192-case stratified rejection corpus, the warm queued implementation
processes `427` fits/second before compatibility fallback versus `61.8` for the
former batched path. The 96-step fallback preserves the former rejection count,
improves mean KL (`0.01275` versus `0.01279`), and still reaches `149`
fits/second even though almost every row in that corpus is pathological. Wide
1,440- and 2,880-fit queued benchmarks reach about `721` and `750` fits/second
at `0.90` and `1.78 GiB` peak allocated memory. The complete production path,
including the selective fallback and temporal passes, sustains `369`
fits/second on a 2,880-row batch made entirely from the rejection corpus. During
the five-pass wide benchmark, one-second samples averaged `94.8%` SM activity
and about `70%` memory-controller activity. A 5,760-fit batch had no
throughput benefit and reserved essentially the full device, so 2,880 is the
configured safety/performance point.

The retained chronological continuity benchmark selected a warm candidate for
`10.1%` of timestamps,
reduced median normalized parameter step from `1.619` to `1.314`, 90th-percentile
step from `5.449` to `5.242`, and median normalized second difference from
`3.431` to `2.970`. Those benchmark quality values predate the visible-only
diagnostic gate and are retained only as historical throughput/continuity data.
Every warm selection now passes the usable-surface quality guard. Rejected timestamps are
durably retained for slower refinement rather than stalling the daily stream. A
quadratic-free quality-matched ablation reached KL `0.00280` and MSE `1.71e-6`.

Daily Bellman-oracle construction also runs through the native CUDA helper. On a
real 86,400-candle day with grid 151, `H=60`, `T=3600`, fees, leverage bounds,
and all three nonzero maintenance rates, it takes `10.38 s` of kernel time and
`12.31 s` wall time. The second day took `10.36 s` kernel / `10.42 s` wall.
Together with the warm teacher worker, the first two complete dataset days took
48 seconds after startup instead of roughly four minutes on the former
CPU-oracle path. `mlp:run` rebuilds the native helper before resuming so the
loaded ABI always matches the TypeScript wrapper.

### Grid-size ablation for the per-second run

The current v7 artifact remains on 151 action points because its dataset and
checkpoint already use that contract. For the next dataset, 127 action points
with 63 sampled fitter actions is the preferred layout. The native oracle rounds
its chain block to the next power of two: 151 points therefore use 256 threads,
whereas 127 points use 128 threads with one inactive lane and half the per-chain
shared workspace. Both counts are odd and include zero; 129 is the unfavorable
boundary because it returns to a 256-thread block.

On a complete 86,400-candle day, 127 points took `2.878 s` of oracle kernel time
versus `3.547 s` at 151, an `18.9%` reduction. Policy-mean RMSE was `0.324`
exposure units, path-exposure RMSE was `0.500`, and final log return differed by
`-7.36e-4`. On the matched 192-case rejection corpus, the 127-point fit had
lower KL in `64.6%` of cases and a median KL change of `-5.09e-5`; it accepted
two additional cases. One structural outlier, rejected under both grids,
dominates the worse untrimmed mean and remains isolated by the quality queue.
Using 63 sampled actions also fills the fitter's existing padded 64-lane row,
includes zero, and samples every second point of the 127-point grid exactly.
Per-bin probability MSE is not invariant to grid size; its purely geometric
151-to-127 scaling is about `1.414`. A 127-point plan should therefore start its
MSE gate near `4.24e-6` and recalibrate it on the accepted-fit distribution,
while retaining KL as the directly comparable quality criterion.

Repeat the oracle timing and numerical comparison with:

```bash
npm run mlp:benchmark:oracle -- \
  --candles 86400 --iterations 2 --grid-sizes 127,129,151,255
```

## Train and verify

The normal entry point builds/resumes the dataset, trains/resumes on CUDA, exports
the best validation checkpoint, verifies PyTorch/ONNX and CPU/GPU parity, and then
makes the artifact discoverable by the backend:

```bash
npm run mlp:run
```

In another terminal, watch daily oracle preparation, CUDA teacher throughput,
allocated GPU memory, mean teacher KL/MSE, or model-training updates (Ctrl+C only
stops the watcher):

```bash
npm run mlp:status
```

Once the model-training stage has begun, request a stop after its current GPU
update, then validate, restore the best checkpoint, test, export, and verify the
artifact. A request made during dataset construction remains pending until model
training starts:

```bash
npm run mlp:finalize
```

Early stopping and model selection use validation loss; the test set is evaluated
once after the best checkpoint is restored. Training uses AdamW, a warmup/cosine
schedule, AMP, gradient clipping, deterministic seeds, checkpoint/resume support,
and the configured CUDA device. `data/ml-runs/mlp-conservative-quadratic-temporal-matmul-cuda-v8/training.log`
contains the complete append-only event stream. The status display includes loss,
CE, probability/parameter MSE, excess entropy, both MI rewards, learning rate,
gradient norm, throughput, and GPU memory.

The v8 continuation initializes model and optimizer state from v7's last
checkpoint, then runs a fresh 15-epoch warmup/cosine schedule. Its excess
entropy, State MI, and Oracle MI weights are 0.1; the remaining loss weights stay
at 1. Validation selection and early-stopping state restart under the revised
objective rather than inheriting incomparable v7 loss values.

Verification checks the ONNX graph contract, PyTorch/ONNX numeric parity, and
deterministic CPU/CUDA parity, then records the provider, errors, and timestamp in
`manifest.json`. For a CPU-only
host, set `TRADING_MLP_VERIFY_CUDA=false`. To require GPU inference at runtime, set
`TRADING_MLP_EXECUTION_PROVIDER=cuda`; `auto` attempts CUDA and safely falls back
to CPU. Both `npm run dev` and `npm start` automatically expose cuDNN from
`.venv-ml` when present.
Use `TRADING_MLP_CUDNN_DIR` when cuDNN is installed elsewhere.

## Runtime and UI

Artifacts are discovered under either `models/mlp/<model-id>` or
`$TRADING_DATA_DIR/models/mlp/<model-id>`. A valid artifact contains at least:

```text
manifest.json
model.onnx
```

Open the VW-KAMA inspector, select `MLP`, then choose the artifact. Analysis,
statistics, distributions, simulated exposure, returns, and charts are recomputed
from that model. Hovering a candle can call `POST /api/kama-inspector/predict` with
the inspector request plus `time`; the UI uses a 300 ms trailing throttle,
cancels stale requests, and retains the 64 most recent exact distribution points.

Artifacts from the earlier 13-coordinate feature schema are intentionally
ignored by the current runtime. Once v8 verifies,
`mlp-conservative-quadratic-temporal-matmul-cuda-v8` appears alongside the
preserved `mlp-conservative-quadratic-temporal-matmul-cuda-v7` artifact without
a server restart.
