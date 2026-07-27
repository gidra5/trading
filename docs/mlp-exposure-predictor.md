# Causal MLP exposure predictor

The inspector's `MLP` predictor maps causal multi-resolution market features
directly to 255 base-action logits. The deterministic fee/current-exposure
transition then produces the conditional exposure probabilities, simulated
trades, return series, statistics, and chart values. There is one prediction path:
both full analyses and exact timestamp hover requests use the same evaluator.

## Input and output contract

Every schema-5 input row has 901 `float32` values in this fixed order:

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

There are no execution-state values in the current network input. Fee, usable
and effective leverage, and maintenance rates remain fixed teacher/execution
configuration in v12; they are stored with the dataset and still govern policy
decoding, but the MLP does not receive eight constant columns. Schema 5 is the
only accepted feature contract; older 909-input datasets and artifacts must be
rebuilt rather than migrated. Feature
normalization is fitted on the training split only and stored inside the ONNX graph.

The active v12 learnability plan has `predictionDelayMs = 3600000`. For an execution or
prediction timestamp `p`, its input row contains only candles causally available
through `p`, while its oracle target is timestamped `p - 3600000`. The
network therefore reproduces a policy with exactly one hour of hindsight and
one hour of response lag; the delay is not another input feature. It is stored
in the dataset and model manifests and shown in both the training page and model
selector. Changing the training-plan value creates a different pairing over the
same timestamp-keyed components.

The configured joint study compares `0`, `1`, `30`, and `60` minutes at every
one of the 33 joint loss-weight design points. All 132 combinations use the same
component files, prediction-time split assignment, seed, architecture, and
training settings; the builder replaces only the row pairing and its causal
time weights between delay groups. At 60 minutes, all 3,600 seconds in the teacher's value
horizon are already historical at prediction time. This makes the oracle a
deterministic function of the available *raw history* and fixed execution
configuration. It does not make validation error pure neural-network
approximation, because the 901-value feature encoder is a compressed view of
that history; residual error also contains feature insufficiency, optimization,
and generalization error. The deployable target is read directly from the raw
oracle and therefore has no fitted-parameter approximation error.

The current graph is exactly `901 -> 16 x 1024 -> 255`. It uses LayerNorm,
SiLU, small dropout, and residual hidden blocks. The output cells correspond to
the dataset's stored `[-250, 250]` action grid. Training masks and normalizes
them on the executable `[-100, 100]` range; inference interpolates those logits
onto the runtime visible grid and applies the same deterministic transaction
transition as the oracle.

The deployable target is the raw oracle base distribution, not the fitted
parameters. The controlled learnability run uses a close-only one-minute oracle
ending at the latest completed UTC minute available at each timestamp. This target is
exact on second `59`; otherwise it is conservatively shifted back by 1–59
seconds. At the active 60-minute prediction delay its complete 60-step path is therefore
a deterministic subset of the model's causal one-minute inputs, without hidden
within-minute ordering. Visible base-distribution Jensen-Shannon divergence is
persisted as resolution metadata.
The original distance/persistence weight is retained in `baseTimeWeights`; the
combined weight is retained in `timeWeights`. This keeps the one-minute target,
the resolution diagnostic, and both weight stages independently auditable.
`train_mlp.py --target minuteOracleProbabilities` uses that aligned coarse
distribution directly for the controlled 60-minute-delay learnability test.
The target distributions themselves are intentionally not persisted: four
deterministic CPU workers compute each unique target UTC day once into shared
memory before training. The current full-corpus run measures whether the
existing architecture can learn the one-minute oracle over all 33,177,600
production examples before another architecture is considered.

The revised conditional four-segment fitter is retained as an offline
diagnostic only. Production-wide direct-target preparation does not run it:
the CUDA diagnostics pass computes target entropy, exact hard cutoffs, and the
whole-example distance imbalance directly from the raw oracle surface at about
170k rows/s. Existing completed fitted components remain reusable for visual
diagnostics, but missing production dates no longer spend hours optimizing
parameters that the MLP does not predict. The historical diagnostic fitter
Its first stage solves the old compact `[-100, 100]` problem, including hard
cutoffs clipped into that support. It then analytically remaps `c1`, `c2`, `b`,
`lambda`, `betaC1`, and `betaC2` into the `[-250, 250]` coordinate system. The
remapped score has exactly the same usable-range probabilities (up to float32
roundoff and an action-independent softmax constant), while the original exact
effective-range cutoff coordinates are restored. No second network or second
complete fit is required. Selective recovery retains 75% usable-domain samples
and 25% outer anchors. The production decoder truncates the extended latent
policy to usable leverage, while survival boundaries remain identifiable
outside the executable window.

Fit quality is measured on the executable `[-100, 100]` surface only. The
oracle and fitted rows are re-normalized across the 101 usable action cells, and
cross-entropy, KL divergence, and probability MSE are averaged across the 101
usable current-exposure cells. Those visible-only diagnostics drive rejection,
adaptive refinement, and temporal candidate selection.
The streaming first pass uses batches of up to `4096` examples, `32`
variable-projection steps, `30` batched BFGS steps, `3` structural starts, `31`
sampled states, `63` sampled actions, and `1e-8` tolerance. Variable projection
subtracts the fixed fee transition, eliminates each row's normalization offset,
and analytically solves `[b, lambda, betaC1, betaC2]` while optimizing `c1/c2`.
This is the screenshot design `[1, a, -a²/2]` extended by the two fixed-location
smooth hinges. Only each resulting 4x4 correlated system is solved in float64.
All three projected starts are scored, then only the per-example winner enters
full-memory BFGS. Examples above either quality threshold are compacted into an
immediate bounded refinement lane; only cases that still fail that lane enter
the durable refinement queue. The current plan emits one target per second, so a
complete day contains 86,400 rows and is processed as 21 full 4,096-row batches
plus one 384-row tail batch.
For each timestamp, the two cutoff labels come from the mandatory `H`-step
equity/maintenance recursion. Zero exposure anchors the feasible interval;
each side is bisected between zero and the effective limit. Any liquidation
during the hold makes that action infeasible. Actions outside maximum effective
leverage are infeasible by definition. The cutoffs are hard masks, so their
training signal is the standardized parameter loss rather than a soft-gate
surrogate.
Training evaluates the fitted teacher and prediction on a fixed, evenly spaced
current-exposure grid; there is no random current-state sampling. Dataset
construction computes one signed distance-imbalance scalar for each complete
timestamp example from its exact cutoff-applied raw oracle map on the visible
portion of the stored 255-by-255 grid:

```text
D_t = sum_{visible x} E[a - x | x]
      / (sum_{visible x} E[|a - x| | x] + 1e-6)
W_t = (1e-6 + |D_t| * persistence_multiplier)
      * (1 + resolutionDivergenceMultiplier * JSD(oracle_1s, oracle_1m))
w_t = W_t / mean_batch(W_t)
```

This distinguishes equally probable left/right moves when one tail extends
farther, while symmetric inward pressure at opposite current exposures cancels
before the absolute value is taken. Unlike a mean of normalized row imbalances,
the ratio of global integrals weights each current-exposure row by its expected
action distance. Advice with `|D_t| >= 0.25` also carries a
causal same-side evidence counter. Each earlier same-side important point adds
`0.25` to the next point's multiplier, capped at `4x`. Weak timestamps decay the
evidence with a 15-step half-life; opposite advice or a gap longer than 60 steps
resets it. Under the current one-second cadence, these are 15-second and
60-second durations. Persistence is prepared chronologically within each split
and the resulting unnormalized whole-example scalar is written to the shard
before training can shuffle examples, so future advice never affects an older
example. `D_t` is stored as the `distanceImbalance` teacher-metadata field. The
distance/persistence-only value is stored in `baseTimeWeights`; `W_t` is stored
in the shard's aligned `timeWeights` file; and the visible 1s↔1m JSD is stored
in `resolutionDivergence`. Training neither decodes fitted teacher parameters
nor recomputes persistence or resolution; it only applies
`W_t / mean_batch(W_t)`. The loss is:

```text
cross entropy
+ probability MSE
+ squared normalized excess entropy
- conditional Gaussian Oracle MI
```

Conditional cross entropy, conditional probability MSE, and excess entropy use
a 255-action by 31-current-state surface spanning only the visible usable
range. The action-only CE/pMSE terms and Oracle MI use the emitted 255-action
base distribution. The persisted distance imbalance uses
all visible cells of the raw 255-by-255 oracle map rather than that training
resampling. Teacher and
predicted policies are normalized on that surface, so latent-only cells do not
dilute any optimized metric. For the current one-minute-oracle learnability
run, conditional and action-only cross entropy and probability MSE all have
coefficient `1`; excess entropy is disabled and Oracle MI has coefficient
`0.5`. There is no parameter-MSE term in the deployable direct-output model.
Training logs also
report the mean raw distance-imbalance weight and
its effective-sample ratio; both appear on the live training page.

For fast studies, `npm run mlp:study:frozen-production` performs a resumable
two-stage preparation. It first scans all 33,177,600 production timestamps,
stores raw direct-oracle factors plus weights, and deliberately skips the
roughly 63 GB full feature materialization. It then freezes non-overlapping
3,600-second temporal blocks with seed `1337`: 192 training blocks are sampled
without replacement in proportion to their full combined `timeWeights`, while
150 validation and 42 test blocks are sampled uniformly so evaluation remains
unbiased. This exactly preserves the existing study sizes of 691,200 /
540,000 / 151,200 rows. The second stage materializes features and completed
minute targets only for those 384 frozen blocks. Exact source weights are copied
into the compact dataset, including same-side persistence accumulated before a
selected block, rather than being recomputed after unselected rows disappear.
The same command then trains all 691,200 frozen training rows, evaluates all
540,000 validation rows, tests all 151,200 test rows, verifies the exported
ONNX artifact, and leaves the resulting model available to the backend/UI.
Validation KL and probability MSE are accumulated as per-example weighted
population moments over the complete split. Logs and the training UI show the
mean, variance, and standard deviation of both metrics, rather than the
variance of minibatch averages.

The full production learnability run is:

```bash
npm run mlp:train:production-minute
```

It reuses the prepared raw-oracle factors, per-example weights, and resolution
metadata. Missing source candle days are fetched and validated once. Causal
901-value inputs are generated as complete UTC days by three persistent CPU
workers, converted to float16, and stored as independent zstd frames. This
keeps decompression day-local and resumable; the trainer shuffles day groups
and contiguous blocks without repeatedly expanding unrelated days. One-minute
targets remain memory-only for every epoch.

Oracle MI uses the same time/state layout. For every current-exposure state, it
computes weighted predicted and teacher means, total variances, and their
covariance over the contiguous minibatch time axis. It converts squared
correlation to the Gaussian approximation
`-0.5 * log(1-rho^2) / log(|A|)` and then averages uniformly across current
exposures. This avoids cancellation between conditioning states; cross entropy
and both MSE terms continue to enforce the correct correlation direction and
full distribution shape.

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
prediction timestamp can never leak between splits. Split assignment is made
before subtracting the configured policy delay: training, validation, and test
therefore describe execution time, while the associated oracle timestamp is
earlier. The redundant `fit-full` aggregate is excluded because its four
constituent windows already cover the same dates.

Every scored day, its warm-up day, and the day containing its future horizon
must contain valid, contiguous one-second candles. The builder tries both local
JSONL encodings and, when any required day is missing, malformed, or incomplete,
immediately downloads the Binance daily archive, verifies its checksum and
strict one-second continuity, installs it atomically, and resumes preparation in
the same run. A day enters
`source-rejection-queue.json` only if that recovery also fails; completed shards
remain resumable while the failed source is retried on the next run.

The MLP teacher always uses extended fixed-horizon targets. Each 86,400-row
oracle component loads another 3,600 one-second candles after midnight, so every
oracle timestamp—including 23:59:59—has the complete one-hour value horizon.
Oracle probabilities and mandatory-hold cutoffs may read those future prices,
while the paired MLP feature row remains causally bounded at its later prediction
timestamp. The oracle path and persisted component still stop at the UTC day
boundary.

This plan samples one target per second and trains at 17.5 bps fees, `[-100, 100]`
usable exposure, `[-250, 250]` effective exposure, and 10 bps/hour for the three
configured maintenance rates. Every teacher target stores the 8 fitted raw
parameters and its cross-entropy, KL divergence, probability MSE, iterations,
screened structural starts, convergence flag, and raw-oracle `D_t` metadata.
Fits above either quality
threshold are retained as the best available target and also appended to
`teacher-refinement-queue.json` with their timestamp and visible-range
diagnostics. They never block later shards and can be regenerated in a focused
refinement pass. Each shard records its diagnostic bounds, so a later refinement
automatically rebuilds shards whose stored metrics came from an older scope.
Completed shards are retained and reused after a restart. Both oracle preparation and
teacher fitting are CUDA-required for this dataset; there is no silent CPU
fallback that could mix different numeric procedures within one run.

Each oracle timestamp retains the complete normalized 255-current-state by
255-target-action raw oracle map. It is stored losslessly in factorized form:
`*.raw-oracle-probabilities.f32` contains the 255-value CUDA-oracle base row, and
the other map axis is the deterministic transaction transition defined by the
manifest's current grid, action grid, friction, and temperature. The shared
`materialize_raw_oracle_policy_map` helper expands a batch to
`[batch, 255, 255]` directly on CPU or GPU. This is mathematically identical to
storing every dense row, not a low-rank approximation or quantization. Passing
the stored `teacherParameters[6:8]` cutoff coordinates to the same helper emits
the exact hard-masked, renormalized map used as the fitter target.

The exact hard cutoff is retained in the last two fitted-parameter coordinates,
so the fitter's masked target map is reconstructible while the unmodified oracle
map remains available for direct-map training. Inputs and oracle targets are no
longer duplicated into aligned delay-specific shards. Instead,
`components/inputs` and `components/oracle` use row-addressable UTC-day files
keyed by prediction and oracle time respectively. Production components are
complete days. Screening components are sparse files whose materialized row
ranges cover only the configured contiguous blocks and the union of rows needed
by every tested delay. Dataset shards contain only component row offsets/strides plus the
delay-specific time weights. A new delay can therefore rebuild the lightweight
pairing manifest without recomputing completed feature days or refitting
completed oracle days. Delay experiments may use distinct model/run IDs while
sharing the same `datasetDir` and `componentStoreId`; only the small pairing
manifest and time-weight files are replaced. Sparse screening files are also
valid production seeds: the production builder imports them, fits only missing
seconds, and atomically writes the merged full-day component. Normal production
completion preserves already-fitted study seconds; a later explicit quality-
refinement pass may still replace rows that fail the configured KL/MSE gates. A complete
86,400-example day uses 88,128,000 bytes (84.0 MiB) for the lossless factors;
dense float32 maps would use 22,472,640,000 bytes (20.9 GiB) per day and roughly
7.9 TiB for the current plan. Every oracle component records a hash of its
teacher-fit contract. When only the teacher definition changes, the builder
reads the stored raw oracle factor and refits the parameters directly; it does
not rerun the oracle kernel. Refinement passes replace the stable teacher
component atomically instead of duplicating the raw grid.

The fitter first solves every timestamp independently on compact usable support,
analytically extends it to effective support, then runs two wide CUDA
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
compacted before the next dispatch. Masked logits use the finite Float32 floor,
not IEEE negative infinity, so the padded 64th action lane cannot contaminate
the queued objective with `0 * -inf`. The queued kernel also receives the
explicit 200-exposure hinge calibration instead of inferring it from the
500-exposure effective support. One 250-step adaptive pass handles hard cases;
the former second pass changed mean KL by only `3.4e-8` and recovered one of
4,096 cases, so it is left to the durable rejected-fit refinement rather than
charged to every first-pass batch. Fits still above
the strict quality gate get
96 steps of the established PyTorch BFGS as a selective compatibility fallback;
accepted fits never pay for it. The structural projection already uses batched
`einsum` normal matrices and 4-by-4 solves. A warm result remains eligible
only when its visible 101-by-101-grid cross-entropy is better than, or within
`5e-5 + 2e-5 * abs(crossEntropy)` of, the independent fit. A two-state dynamic
program finally chooses the minimum-total-jump chronological path through the
independent and eligible warm candidates. Continuity is therefore a tie-break
between quality-equivalent fits, not a loss term that can trade away oracle fit.
The final fit of one 4,096-row chunk seeds the first temporal candidate in the
next chunk, so a future per-second day has no artificial parameter discontinuity
at a batching boundary.

Input rows are packed as 288 float32 values: 255 probabilities, two cutoff
coordinates, and 31 padding values, making every row 1,152 bytes (nine
128-byte cache lines). Sampled GPU targets use 64 lanes for 63 action cells, or
256 bytes per state row. The 8-coordinate state uses an 8-by-8 inverse-Hessian layout, while
only the first six smooth score coordinates are optimized by BFGS; the exact
cutoff labels remain fixed.
Four pinned-host batches are prepared ahead of the consumer, so file decoding
and host copies overlap the current GPU batch. This keeps the dense layout
`[fit, state, action]`, with action as the contiguous dimension used by each
warp. Projection uses a local explicit Adam update rather than the shared
multi-tensor optimizer implementation, making concurrent foreground projection
and background autograd fallback safe. A completed refinement stream is also a
lifetime fence for foreground-owned target buffers before the allocator may
reuse them. The Python CUDA worker is deliberately recycled after each complete
daily fit job. This preserves within-day graph compilation and stream overlap,
but releases allocator arenas and compiled dynamic-shape execution state before
the next 86,400-row regime. A retained worker fell from `343.8` fits/second on
the first day to `7.8` fits/second at the start of the second; the exact same
second-day rows reached `248.8` fits/second in a fresh two-batch pipelined worker
and `183.7` fits/second for an isolated full production batch. Per-day recycling
therefore trades one small cached-JIT startup for a guard against a roughly 24x
cross-job throughput collapse without changing fitted parameters or quality
gates.

On the current 4,096-case profiling input, the corrected queued objective plus
the 32-step projection and single adaptive pass sustained `344.9` fits/second,
up from `329.7`. Mean visible KL improved from `0.0058734` to `0.0057285`, mean
probability MSE improved from `5.5953e-6` to `5.4960e-6`, and the rejection count
did not increase (`2,060`). The queued objective agrees with the shared compiled
objective within `6e-7` on the explicit padded-action/200-span parity test.

On the 192-case stratified rejection corpus, the warm queued implementation
processes `427` fits/second before compatibility fallback versus `61.8` for the
former batched path. The 96-step fallback preserves the former rejection count,
improves mean KL (`0.01275` versus `0.01279`), and still reaches `149`
fits/second even though almost every row in that corpus is pathological.

The last pre-255 effective-grid, visible-metric two-lane benchmark used 151 action
cells over `[-250,250]`, measured the 61-by-61 usable surface, and forced every
synthetic row through adaptive refinement, compatibility fallback, and temporal
selection. Batch sizes `2,880`, `4,096`, `4,800`, `5,120`, and `5,760` sustain
`320`, `339`, `344`, `343`, and `349` fits/second while reserving `2.94`, `4.13`,
`4.80`, `5.11`, and `5.81 GiB`. Repeating `4,800` reproduced `343.9` fits/second.
The 1.4% gain at `5,760` consumes 94.5% of the 6-GiB device, while `5,120` is
strictly slower, so `4,800` is the exclusive-GPU throughput point and `4,096`
is safer when another CUDA workload must share memory. The all-accepted
four-batch path reaches `726` fits/second at `3.78 GiB` reserved, corresponding
to about 119 seconds for 86,400 fits before oracle construction and persistence.
The all-rejected bound is about 251 seconds at `4,800`. Production graph
compilation remains inside the memory envelope (`4.23 GiB` reserved) but adds a
one-time cold-start cost that amortizes over the complete one-second day.

The retained chronological continuity benchmark selected a warm candidate for
`10.1%` of timestamps,
reduced median normalized parameter step from `1.619` to `1.314`, 90th-percentile
step from `5.449` to `5.242`, and median normalized second difference from
`3.431` to `2.970`. Those benchmark quality values predate the visible-only
diagnostic gate and are retained only as historical throughput/continuity data.
Every warm selection now passes the usable-surface quality guard. Rejected timestamps are
durably retained for slower refinement rather than stalling the daily stream. A
quadratic-free quality-matched ablation reached KL `0.00280` and MSE `1.71e-6`.

A July 2026 cadence audit then measured the active 255-cell fitter on three real
one-second slices from 2022, 2023, and 2026 (8,192 examples and 8,189 adjacent
candidate pairs). Only `25.4%` of final rows selected the warm candidate;
`38.6%` of warm candidates were quality-equivalent to their direct fit, `6.7%`
were materially better, and `1.7%` independently passed both strict visible-range
quality gates. The one-second selection rates varied from `12.5%` to `41.6%`
across regimes, so proximity in time does not make the independent basin search
redundant. A true previous-second chain anchored once per minute was also tested:
only `12.3%` of its 2,013 intermediate fits remained quality-equivalent, and
increasing its iteration allowance from 20 to 128 did not recover the lost
basins. Consequently, v11 keeps direct and warm candidates at every second. The
proposed 30-minute direct / one-minute hybrid / sub-minute warm-only scheduler is
not enabled because it would silently replace most targets with worse fits.
Production progress now reports warm equivalence, strict acceptance, improvement,
and final selection separately. Re-run the audit with `npm run
mlp:benchmark:temporal`.

Daily Bellman-oracle construction also runs through the native CUDA helper. On a
real 86,400-candle day with grid 151, `H=60`, `T=3600`, fees, leverage bounds,
and all three nonzero maintenance rates, it takes `10.38 s` of kernel time and
`12.31 s` wall time. The second day took `10.36 s` kernel / `10.42 s` wall.
Together with the warm teacher worker, the first two complete dataset days took
48 seconds after startup instead of roughly four minutes on the former
CPU-oracle path. `mlp:run` rebuilds the native helper before resuming so the
loaded ABI always matches the TypeScript wrapper.

### Grid-size ablation for the per-second run

The v11 dataset uses 255 full effective-range cells, 63 sampled fitter actions, 31 sampled fitter
states, and 31 current states per MLP training example. Every symmetric grid is
therefore `2^n-1`: zero is the exact center while the padded CUDA/Triton work
dimensions are powers of two. The native oracle runs the 255-cell chain in a
256-thread block with one inactive lane.

The earlier ablation remains useful performance context: on a complete
86,400-candle day, 127 points took `2.878 s` of oracle kernel time
versus `3.547 s` at 151, an `18.9%` reduction. Policy-mean RMSE was `0.324`
exposure units, path-exposure RMSE was `0.500`, and final log return differed by
`-7.36e-4`. On the matched 192-case rejection corpus, the 127-point fit had
lower KL in `64.6%` of cases and a median KL change of `-5.09e-5`; it accepted
two additional cases. One structural outlier, rejected under both grids,
dominates the worse untrimmed mean and remains isolated by the quality queue.
Using 63 sampled actions fills the fitter's padded 64-lane row and includes
zero. Per-bin probability MSE is not invariant to grid size. Moving the metric
surface from 61 to 101 action cells geometrically scales the former `3e-6` gate
by `(61/101)^2`; v11 therefore starts at `1.1e-6`, while KL remains the directly
comparable quality criterion.

On the 4,096-row profiling slice, the former uniform effective-support fitter
accepted 209 rows and rejected 3,887 at the unchanged KL `0.003` / probability
MSE `1.1e-6` gates. Compact initialization plus the restored 200-wide hinge
calibration accepted 2,035 and rejected 2,061. Mean visible KL fell from
`0.01980` to `0.00586`, mean probability MSE from `1.52e-5` to `5.60e-6`, and
steady execution remained around 300 fits/second. A parallel full-support fit
added only four accepted rows, so it is not worth doubling the optimizer work.

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
allocated/reserved/whole-device GPU memory, mean teacher KL/MSE, or
model-training updates (Ctrl+C only stops the watcher):

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
compiled model-plus-loss execution, and the configured CUDA device. The physical
training batch is 256 with no gradient accumulation, preserving the former
effective optimizer batch of 256; evaluation uses batches of 1,024. Data workers
coalesce each contiguous memory-mapped block into five batched tensors rather
than constructing and sharing one tensor tuple per example. On the current
691,200-example study split this reduced a measured train-plus-validation epoch
from about 616 seconds to 47.3 seconds; the complete one-epoch process took 52.7
seconds. A tested 512/2,048 configuration was slower on its cold compiled pass,
halved optimizer updates, and had worse one-epoch validation KL, so it is not the
default. `data/ml-runs/mlp-conservative-quadratic-cutoff-temporal-matmul-cuda-v11-delay-60s/training.log`
contains the complete append-only event stream. The status display includes loss,
conditional/base KL divergence, probability MSE, excess entropy, oracle MI,
learning rate, gradient norm, throughput, and GPU memory.

The direct-distribution trainer keeps compressed minute-oracle components lazy
inside Windows DataLoader workers instead of pickling every decoded day into
each process. The shuffled training loader owns the only persistent worker
pool. Minute-curriculum validation uses two temporary workers to materialize a
bounded GPU-resident cache once; they exit before training workers start.
Recurring validation therefore performs neither decompression nor host-to-device
copies. Final test evaluation remains in the main process. All uncached transfers
reuse one run-lifetime CUDA copy stream so allocator blocks stay reusable across
epochs instead of becoming associated with new streams. The compiled objectives
use static Inductor graphs with CUDA graphs disabled; only the full and final
partial batch shapes need workspaces. At epoch boundaries the trainer releases
inactive CUDA allocator blocks only after reserved memory exceeds half of device
capacity. This is now an emergency fragmentation guard rather than routine
cleanup. Training events report current tensor allocation, PyTorch reservation,
whole-device usage, validation-cache storage, and whether cache trimming occurred
separately. Throughput is a rolling completed-example rate across training
segments and excludes validation time.

The training loader queues the next epoch before validation starts, so its
persistent workers decompress and pin the first batches while the fixed
validation cache occupies the GPU. This removes the recurring first-batch wait
from the train/validation boundary. Resumable checkpoints default to every eight
epochs, plus every newly improved best state and every terminal epoch. Each
checkpoint takes one bounded frozen CPU snapshot, then serializes it atomically
on a single background writer while the GPU continues. The writer must finish
before another snapshot is accepted, so disk backpressure cannot accumulate
checkpoint copies in memory. A newly improved best-model state shares the
checkpoint's frozen model tensors and is written by the same ordered job.
Shutdown and finalization always flush the pending job. With the current
one-minute workload, an interruption can therefore repeat at most seven epochs,
roughly twenty seconds of work.

Target-only curriculum phases keep their requested 256-epoch cosine schedule,
then continue for up to 4,096 additional epochs at the schedule floor inside
the same Python process. This avoids a model reload, validation-cache rebuild,
and compiled-kernel warm-up every 256 epochs. If the unusually large continuation
ceiling is exhausted, the curriculum adds another continuation block and remains
resumable.

The v11 run starts from a fresh random initialization and runs at most 256 epochs
with 64-epoch early-stopping patience. CE and both MI rewards have weight `1`;
probability MSE, parameter MSE, and excess entropy have weight `0.1`. Only a checkpoint from
this exact v11 architecture, delay, and dataset contract can be resumed.

### Joint prediction-delay and loss-weight response study

Loss-term effects are not assumed to be independent. The configured screening
study uses a 32-run resolution-VI half-factorial design plus one center run. In
the factorial runs, all six coefficients vary jointly between `0.25x` and `4x`
their production values; the center uses the production `1x` weights. This
estimates every main effect and all 15 pairwise interactions without aliasing
them with each other. Main effects remain aliased with five-way interactions,
and pairwise effects with four-way interactions, which are assumed negligible
only for this initial screen. The complete 33-setting design is repeated at
each of `0`, `1`, `30`, and `60` minutes, producing 132 delay/weight
combinations. Consequently, a weight's main effect and every pairwise
weight interaction are estimated independently at each delay rather than
assuming the tasks share one optimum.

Every run uses the same seed, contiguous minibatches, initialization procedure,
and dataset ordering. Model selection uses validation KL and the test split is
never used for checkpoint selection. Runs are sequential so one model owns the
GPU and the shared pairing cannot change under a live data loader. Before any
training, the supervisor prioritizes a 64-calendar-day stratified subset: 32
training days, one day from each of 25 disjoint validation regimes, and 7 test
days. Four contiguous 90-minute UTC blocks per day provide 691,200 training,
540,000 validation, and 151,200 test examples. The oracle fitter prepares the
union required by `0`, `1`, `30`, and `60` minute delays once. Completed
production components are hard-linked into the subset when available; newly
fitted sparse components later seed the resumed production build. The
supervisor then completes all 33 weight runs at one delay before changing the
pairing and restores the priority pairing at the end:

```bash
npm run mlp:study-loss-weights
```

The configured response screen uses at most 6 epochs with patience 2, for a
worst case of 792 epoch-runs across all 132 combinations.
It is resumable and writes a live `summary.json` plus a human-readable
`summary.md` under `lossWeightStudy.outputDir`. Intermediate run metrics are
available immediately, but a delay's factorial contrasts remain blank until all
32 corners for that delay are complete, preventing an unbalanced partial design
from being misread as an effect. Feature normalization is computed once for the
shared prediction rows, while target-parameter scale is computed once per delay;
the remaining 128 runs reuse those exact cached statistics instead of rescanning
the complete dataset.
Study variants also skip validation of the random untrained initialization;
every trained epoch is still validated and eligible for checkpoint selection.
The response screen uses a deterministic, time-stratified 25% of every validation
run. This cuts repeated screening cost while preserving coverage across regimes.
Promoted confirmation runs set the fraction to 100%, so final model selection and
reported final validation metrics still use the complete validation set.
Every completed artifact is immediately ONNX-verified and retained below the
joint-study output directory. The backend discovers these artifacts recursively,
so every study model remains selectable in the inspector UI together with its
delay, loss weights, best epoch, validation metrics, and test metrics.

The current plan stops model training after the reduced screen at a manual
review gate. Production dataset preparation remains paused while the screen owns
the GPU; neither a production-data resume nor a larger confirmation run starts
without an explicit follow-up decision.
After reviewing the report, promotion can be explicitly enabled to train the
best two settings per delay on the same 64 complete days and then confirm the
best two on the production dataset. Promotion artifacts use the same retained,
verified, UI-discoverable contract.

Watch the supervisor without stopping it:

```bash
npm run mlp:study-loss-weights:status
```

The supervisor writes its own five-second heartbeat separately from child
training progress. A training child that emits no progress event for 120 seconds
is treated as stalled. Before restarting it, the watchdog writes process/thread
state and `nvidia-smi` output under
`lossWeightStudy.runDir/diagnostics`, sends `SIGUSR1` to dump every Python thread
into the append-only study log, then terminates the complete child process group.
Training resumes from the last epoch checkpoint and receives at most three
automatic attempts. The status command shows heartbeat age, child PID, attempt,
no-progress duration, watchdog threshold, and the latest diagnostic path.
Thresholds can be overridden for a diagnostic run with
`TRADING_MLP_TRAINING_STALL_TIMEOUT_MS`,
`TRADING_MLP_WATCHDOG_STACK_DUMP_GRACE_MS`,
`TRADING_MLP_WATCHDOG_TERMINATION_GRACE_MS`, and
`TRADING_MLP_TRAINING_MAX_ATTEMPTS`.
Under WSL, the supervisor also starts a Windows execution-state request so an
idle timeout cannot place the machine and its CUDA context into Modern Standby
while the study is running. The request uses a native-Windows heartbeat file and
releases automatically three minutes after the WSL supervisor stops refreshing
it, including after a WSL crash. The status command reports whether this
inhibitor is active.
Observed training processes force Python multiprocessing to use `spawn`.
PyTorch initializes CUDA before its train and validation loaders are iterated;
using Linux's default `fork` at those lazy worker-start boundaries can clone a
live CUDA context and wedge WSL's GPU bridge. Spawned workers import the dataset
module into clean processes instead. Batch sampling, seeds, checkpoints, and the
experiment fingerprint are unchanged.

After a candidate has a complete study result and its ONNX artifact passes
runtime verification, the supervisor removes its optimizer checkpoint and
duplicate PyTorch best-weight file. The verified ONNX model, manifest,
verification fixtures, study metrics, and summaries remain available to the
backend and UI. This prevents completed candidates from retaining roughly
250 MiB of training-only state apiece. Existing verified studies can be pruned
with:

```bash
node scripts/mlp-study-retention.mjs
```

The supervisor records free space for both the Linux filesystem and the Windows
C: drive in `status.json`; the status command displays both. It refuses to
start another stage below 5 GiB on either filesystem, because the WSL VHDX
cannot safely grow when its Windows host volume is full even if ext4 still
reports free blocks. Override the guard only deliberately with
`TRADING_MLP_MIN_LINUX_FREE_GIB` or `TRADING_MLP_MIN_WINDOWS_FREE_GIB`.

The report includes validation KL, its validation-set standard deviation,
probability MSE, parameter MSE, excess entropy, and Oracle MI. For validation KL, a negative
main effect means increasing that term helped on average. A pairwise
difference-of-differences far from zero means the effect of either term depends
on the other's weight. Variation in the same weight contrast across delay
columns is the measured delay × weight interaction. The JSON report retains all
10 pairwise interactions at every delay; the Markdown report shows the
per-delay main effects, their cross-delay ranges, the center controls, and the
best combinations. A deliberately cheaper fresh design can override the budget:

```bash
npm run mlp:study-loss-weights -- --epochs 16 --patience 4
```

### Exhaustive loss-weight grid

The follow-up exhaustive screen uses all `4^6 = 4,096` combinations of relative
loss scales `[0, 0.25, 1, 4]` at prediction delays 0, 0.5, 1, 30, and 60
minutes. It retains every configuration and validation metric in the summary,
including four-level marginal curves and all fifteen 4×4 pairwise response
surfaces. It deliberately stops for review after the complete zero-delay block.
Continue the remaining delays explicitly with `--continue-after-review`.

Only the lowest-validation-KL verified ONNX model at each completed delay is
retained. A new winner is exported and verified before the former winner is
removed, while non-winning optimizer state and duplicate PyTorch weights are
discarded immediately. Completed fractional-screen metrics at matching
configurations are reused. Their artifacts are migrated only when they are the
current winner, and the old source artifact is then retired.

A single persistent population trainer shares one dataset, loader pool, and
compiled objective across the exhaustive delay block. Four identical MLP
replicas are stacked along a leading population dimension, so each layer and all
six loss objectives execute as batched tensor operations instead of four Python
processes competing for the GPU. AdamW moments, per-member gradient clipping,
best epoch, patience, and metrics remain independent. Every completed
four-candidate population is a resume boundary and writes four standard
`study.json` files. For this six-epoch screen, an interruption redoes at most the
active four candidates instead of writing an approximately 800 MiB optimizer
checkpoint every epoch. Best states stay in RAM within the active population,
and only the current validation-KL winner is written as `best-model.pt`. If four
replicas do not fit, the supervisor retries with two and then one without
invalidating completed candidates.

On the target RTX 3060 Laptop GPU, the exact compiled 256-row training shape
peaks near 1.34 GiB at population four. A width sweep measured roughly 27k
effective examples/s at width four after replacing parameter-by-parameter
gradient clipping with a paired multi-tensor norm/scale pass. Width eight used
about 2.61 GiB but fell to roughly 10k examples/s because its batched GEMMs enter
an unfavorable scheduling/cache regime on this 30-SM GPU, so the configured
width remains four. CUDA AdamW is explicitly kept on the multi-tensor path;
PyTorch's fused AdamW and a single flattened parameter allocation were both
measured slower and are not used.

Queue the exhaustive screen behind the currently running fractional study:

```bash
npm run mlp:study-loss-weights:exhaustive:queue
```

Watch the queue or exhaustive runner:

```bash
npm run mlp:study-loss-weights:exhaustive:status
```

Verification checks the ONNX graph contract, PyTorch/ONNX numeric parity, and
deterministic CPU/CUDA parity, then records the provider, errors, and timestamp in
`manifest.json`. For a CPU-only
host, set `TRADING_MLP_VERIFY_CUDA=false`. To require GPU inference at runtime, set
`TRADING_MLP_EXECUTION_PROVIDER=cuda`; `auto` attempts CUDA and safely falls back
to CPU. Both `npm run dev` and `npm start` automatically expose cuDNN from
`.venv-ml` when present.
Use `TRADING_MLP_CUDNN_DIR` when cuDNN is installed elsewhere.

### Dynamic delay/weight curriculum pilot

The incremental curriculum search starts from the two best verified 60-minute
models in the completed fractional response study. It then searches the fixed
delay schedule `[30, 30, 1, 1, 0, 0]` minutes. At every one-epoch transition,
every surviving parent is crossed with all 33 fractional loss-weight profiles.
This explicitly tests every next weight step from every retained history; it
does not assume that epoch effects commute or that the best static weights are
the best continuation weights.

The theoretical `33^6` tree is reduced only after observing the child models.
Each child persists a deterministic policy-surface signature over 16
time-stratified validation examples, 15 visible current exposures, and 31 target
actions. Children within the configured validation-KL tolerance and mean
Jensen-Shannon policy divergence are treated as functionally equivalent. The
better representative survives. The remaining candidates are Pareto ranked on
both mean validation KL and validation KL standard deviation, and a four-model
beam continues. Thus the pilot measures how aggressively real trained states
can be collapsed before attempting a wider search; it does not claim that an
unvisited descendant of a pruned non-equivalent state is impossible.

Each branch uses 256 training minibatches, or 65,536 examples, while validation
uses the same deterministic 25% temporal sample as the response screen. There
are at most 726 one-epoch branches, vectorized four at a time. Every completed
population is resumable, rejected checkpoints are removed only after the stage
summary is durable, and all metrics and lineages remain. The best survivor at
each visited delay is evaluated on the test split, exported, verified, and made
selectable in the UI.

Queue the pilot behind the fractional response study and inspect it with:

```bash
npm run mlp:study-dynamic:queue
npm run mlp:study-dynamic:status
```

Preview the exact stage budget without starting work:

```bash
npm run mlp:study-dynamic:dry-run
```

### Adaptive absolute-weight and delay curriculum

The production follow-up removes the fixed 33-profile/fixed-delay restriction.
Every loss receives an absolute value from `[0, 0.25, 1, 4]`; these values are
not multiplied by the production loss weights. At least one distribution
matching term—cross entropy, probability MSE, or parameter MSE—must be nonzero.
That leaves 1,008 valid raw five-way tuples. Globally proportional tuples are
represented once by the largest member that is still on the configured
absolute grid, leaving 774 distinct weight directions. Excess entropy and
Oracle MI may independently be zero.

It would still be wasteful to fully train all 3,330 descendants of every
surviving model. For each parent, the runner computes the five individual
training-loss gradients, the validation-KL gradient, the 5×5 training-gradient
Gram matrix, and five validation-Hessian/vector products. It evaluates the local
clipped-update approximation

`KL(w) ≈ KL₀ - η bᵀw + ½ η² wᵀAw`

for every valid six-way tuple. This is a ranking prior, not a substituted
validation result. Forty-eight candidates per parent are actually trained using
a mixture of projected winners, weight-space coverage, and deterministic
exploration. A Gaussian process fitted to measured-minus-projected residuals
then requests 16 more candidates per parent by lower confidence bound.

Measured candidates pass through successive fidelities of 32, 96, and 128
additional minibatches. Promotion widths are 16, 8, and 4, and validation grows
from 1/256 to 1/32 to 1/4 of the study validation split. A finalist therefore
receives 256 minibatches, or 65,536 examples, in one adaptive round. Only real
validation KL and its example-level standard deviation drive promotion.
Full-fidelity models then use the same policy-JSD equivalence collapse and
KL-mean/KL-standard-deviation Pareto beam as the pilot.

The delay controller starts at a 60-minute source model and first tries a
30-minute step. Delay is stored as integer seconds, so subsequent trust-region
steps can land anywhere from 3,600 seconds through one second. A transition may
dwell for up to eight adaptive rounds while accuracy recovers. It advances when
it reaches the delay-specific direct-training KL reference within the configured
absolute/relative tolerance, halves the step and restores the last accepted
parent after a plateau, and can continue one second at a time when the minimum
step is reached. Direct-reference KL is log-delay interpolated between measured
0, 1, 30, and 60-minute studies; raw KL values at inequivalent delays are never
compared as if their difficulty were equal.

Round progress, projection matrices, actual probe results, acquisition
diagnostics, promotions, equivalence collapses, lineages, and controller state
are written before obsolete branch checkpoints are removed. The same run command
therefore resumes at the last durable phase of an interrupted round. The best
model at every accepted delay is exported and verified immediately under
`data/ml-dynamic-studies`, so it becomes selectable in the UI while later delay
steps are still running:

Checkpoint retention follows the durable dependency graph. Once every child in
a fidelity is published, its parent fidelity's `best-model.pt` files are
removed; after a round summary is published, superseded continuation parents are
removed as well. A startup sweep repairs retention after abrupt shutdowns while
keeping the current round, current anchor/trial parents, accepted finalists, all
`study.json` metrics, signatures, and verified ONNX artifacts. Thus temporary
branch weights remain bounded to approximately one active round instead of
growing by roughly 1.8 GiB per completed round.

```bash
npm run mlp:study-adaptive:dry-run
npm run mlp:study-adaptive:run
npm run mlp:study-adaptive:status
npm run mlp:study-adaptive:watch
```

The target RTX 3060 uses four simultaneous models because the measured
population-eight path fits in 2.64 GiB but is slower on its 30 SMs. Compilation
is enabled only when a child invocation contains at least 512 population-group
minibatches; shorter acquisition and promotion invocations remain eager to avoid
paying a roughly minute-long compile for seconds of useful training.

For each evaluation split, KL dispersion is the weighted population standard
deviation over examples,
`sqrt(sum_t w_t (KL_t - mean_KL)^2 / sum_t w_t)`. Stable centered moments are
merged across minibatches, so this is not an average of minibatch standard
deviations. The live training page plots validation mean KL and KL standard
deviation together, and both values are retained in study results and model
manifests.

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

The runtime accepts only feature schema 5 with 901 inputs, 16 hidden layers of
width 1024, and 8 outputs. Incompatible artifacts are ignored during discovery
instead of being adapted at inference time.
