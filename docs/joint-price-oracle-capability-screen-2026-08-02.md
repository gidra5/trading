# Joint price/oracle capability screen — 2026-08-02

## Decision

Promote the learned-radius shrinking decoder for a long saturation run. Do not
promote any tested deterministic price predictor or causal oracle-policy model
to a long run. The decoder has demonstrated the required capacity when the
realized future is supplied; the causal models have not demonstrated that the
required future information is present in candle history.

The final low-capacity residual audits also reject a fixed six-hour close/scale
bypass, causal within-minute OHLCV microstructure, and true Binance Spot
aggressor flow, as well as standalone USD-M open-interest/crowding metrics.
Do not implement the proposed v29 branch or another price/trade-only or
positioning-only feature branch; the decoder is the only tuple promoted to a
long run.

The untouched test splits remain sealed. Every result below is a training or
validation result.

The chronological validation region has been reused across the broader
capability search. A fusion's second half is untouched by that fusion's scalar
selection and separated by a full forecast-horizon embargo, but it is a
within-audit development holdout rather than a globally fresh test. Only the
still-sealed test split has the latter role.

## Fixed contracts

- Oracle target: 1 hour value horizon, 1 minute execution delay, 1 minute hold,
  temperature 0.01, and the verified 101-action usable exposure grid.
- Decoder inputs: the 60 realized future one-minute close returns in `(t,
  t + 60m]`. This is a capability proof and is intentionally not causal.
- Predictor inputs: candles ending at or before `t`; prediction targets start
  after `t` and split boundaries include the complete history-plus-target
  embargo.
- Causal policy inputs: 3,601 one-second closes ending at `t`, converted to all
  3,600 one-second transitions, 60 boundary-complete minute tokens, and eight
  causal hour-scale trend/volatility statistics.
- Selection metric for the oracle policy is raw, uncalibrated validation
  `KL(Q || P)` at target temperature 0.01. F1 and bot-return calibration are not
  used to select these capability models.
- Clustering oracle distributions is not a separate learning signal. Training
  every history against a cluster assignment quantizes the same conditional
  target, while direct soft-label cross-entropy has the Bayes optimum
  `P*(a | h) = E[Q(a) | h]`.
- Every representation, fusion, and curriculum result below is ultimately
  scored against the original 101-action T=0.01 oracle distribution. No
  prototype assignment or softened distribution is treated as alternate
  supervision, and all test payloads remain untouched.

## Screen results

| Layer | Best configuration | Validation result | Decision |
| --- | --- | ---: | --- |
| Decoder, current full-corpus screen | 60 future 1m returns → 16-layer shrinking fused-GLU learned-radius MLP, direct T=0.01 | raw KL `0.266811650` at epoch 23 | Promote to saturation run |
| Decoder, preserved historical run | Same 15,090,191-parameter family with the historical regularization curriculum | raw KL `0.071195035` at epoch 1,345 | Proven decoder capacity |
| 1h deterministic path predictor | 6h close history → Patch-TCN / RLinear / DLinear / Patch-TiDE | all at or worse than the conditional-mean baseline | Reject |
| Easier deterministic predictor | 6h of 5m closes → next 5m Patch-TCN | normalized MSE `0.953058479` vs zero-return `0.953415157`; correlation `0.0207584` | Real but too small for the 1h oracle |
| Leakage-safe linear audit | recent 12×5m returns plus 2h/3h/6h sums → ridge | internal selection chose the training-mean limit; validation MSE `0.953472651` | Reject |
| Direct causal oracle policy | corrected boundary-complete minute TCN | raw KL `0.969834961` at epoch 7 | Retain only as causal control |
| Long-context causal policy | v28 six-hour exact-receptive-field TCN, direct T=0.01 | raw KL `0.967228995` at epoch 7; matched v18 `0.969076740` | Best causal diagnostic; reject long run because gain `0.001847745` misses gate |
| Matured-oracle lag | Exact causal oracle distribution from 60m earlier, train-selected prior backoff | raw KL `1.071323907` | Reject |
| K16 prototype-mixture policy | v18 encoder → continuous mixture over fixed soft prototypes | raw KL `0.970259275` at epoch 7 | Reject; no gain over v18 |
| Calendar fusion | v18 plus train-only calendar prior | within-audit holdout raw KL `1.00169874` vs v18 `1.00214279` | Reject; gain `0.000444` |
| OHLCV fusion | v18 plus causal completed-minute geometry prior | within-audit holdout raw KL `1.00199325` vs v18 `1.00214279` | Reject; gain `0.000150` |
| Six-hour close residual | Frozen v18 plus train-selected fixed return/RMS/MA information | within-audit holdout raw KL `1.001682938` vs v18 `1.002509750` | Reject; gain `0.000826811` misses gate |
| One-second OHLCV residual | Frozen v18 plus causal path/range/volume microstructure | within-audit holdout raw KL `1.001903598` vs v18 `1.002142786` | Reject; gain `0.000239188` |
| Spot aggressor-flow residual | Frozen v18 plus 85 true buy/sell flow features and matched magnitude controls | within-audit holdout raw KL `1.002405121`, identical to v18 after zero selected fusion weight | Reject; gain `0.000000` |
| USD-M positioning residual | Frozen v18 plus 80 lagged open-interest/crowding/taker-ratio features and matched magnitude/OHLCV controls | within-audit holdout raw KL `1.002405121`, identical to v18 after zero selected fusion weight | Reject; gain `0.000000` |
| USD-M/Spot basis-flow residual | Frozen v18 plus completed-minute perp basis, taker imbalance, relative activity, and matched controls | within-audit holdout raw KL `1.002405121`, identical to v18 after a neutral signed backoff | Reject; gain `0.000000` |

The decoder corpus has 33,195,600 training, 15,418,800 validation, and
1,000,000 sealed test examples. The canonical 5m linear audit has 33,067,500
training and 14,961,300 validation examples; it selected/read zero test
examples. The causal policy corpus has 535,620 training and 43,080 validation
examples, with 43,140 sealed test examples. The six-hour v28 receptive field
leaves 42,780 ordered validation rows after excluding its additional 300-row
history prefix.

## Promoted decoder tuple

- Feature format: 60 close-only simple returns, standardized by training-split
  per-position population statistics. The representation is scale-free and
  reconstructs the supplied future close path.
- Architecture: widths `512, 496, ..., 272`; 16 fused-GLU layers; independent
  value/gate branch centering; learned-radius RMS normalization; 15,090,191
  parameters. Value and gate projections are fused into one affine projection
  and split afterward.
- Target/curriculum: train directly on the production T=0.01 soft
  distribution. The 0.04→0.02→0.01 curriculum was worse at matched compute
  (`0.297427192` versus direct `0.288694215` through 16 epochs), and sharpening
  a softened prediction increased raw T=0.01 KL.
- Optimizer: hybrid Muon/AdamW, BF16, gradient accumulation 2, initial LR
  `1e-4`, reduce-on-raw-validation-KL patience 48, and early-stop patience 160.
- Saturation ceiling: 1,600 epochs. This is deliberately above the historical
  epoch-1,345 optimum; early stopping remains the normal termination path.
- Plan:
  `ml/training-plans/return-oracle-decoder-learned-radius-direct-long-v1.json`.
- Promotion gate: retain raw validation KL continuously, require approximately
  `0.10 ± 0.01` or better before treating the fresh run as a successful
  decoder, and never select on the sealed test split.

The current 24-epoch direct screen remains resumable at
`data/training/runs/return-oracle-decoder-screen-learned-radius-direct-v1`.
The historical `0.071195035` run remains archived and independently resumable
at
`data/training/runs/return-oracle-ce-shrinking-v1-learned-radius-shrinking-baseline-20260730-104920`.

## Predictor findings

The following axes were actually screened rather than inferred:

- close-only returns, exact additive MA bands, OHLC geometry, and relative
  volume plus a zero-volume mask;
- 1–6 hour histories and 1m, 5m, and 15m aggregation;
- next-5m, next-15m, next-1h, and complete next-hour targets;
- point-return, cumulative-path, and multiscale objectives; and
- RLinear/DLinear, causal Patch-TCN, and Patch-TiDE variants.

For the complete next-hour task, the models converged to the conditional mean.
The best 6h close-only Patch-TCN normalized MSE was `0.96433546` versus the
mean baseline `0.96450067`, only a `0.017%` improvement, and the advantage did
not persist through epoch 16. MA, OHLC, and volume decompositions did not
improve the one-epoch matched screen. Patch-TiDE diverged on both the
multiscale and easier 5m tasks.

The next-5m Patch-TCN is the only retained forecasting diagnostic. At epoch 11
it improved zero-return MSE by `0.0374%`, predicted only `2.60%` of the target's
standard-deviation scale, and had `50.3728%` direction accuracy. Its plan and
best checkpoint are preserved, but the effect is too small to promote a
200-epoch run or to serve as the deterministic input to the hour decoder.

The independent ridge audit prevents a misleading post-hoc conclusion. On its
training-internal chronological calibration split, increasing regularization
kept improving MSE until the selected solution became the training-mean
limit. OLS scored `0.952798509` on external validation, but it was not selected
internally and is recorded as `diagnostic-unselected`; it is not eligible for
downstream use. The canonical reproducible artifact is
`data/training/immutable/refs/models/future-price-resolution/future-price-resolution-6h-5m-next-5m-ridge-v3/model.json`.

## Causal policy findings

The corrected v18 model includes all 3,600 second-to-second transitions that
the earlier minute tokenizer accidentally omitted and adds causal hour-scale
trend/scale statistics. Exact overlapping sequence-core reuse reduces warm
epochs from about 55 seconds to about 10 seconds without changing logits,
losses, or the 372 AdamW updates per epoch.

| Run | Change | Best raw validation KL | Best epoch | Final observed train / validation KL |
| --- | --- | ---: | ---: | ---: |
| v18 reuse | LR `2e-4`, dropout 0, WD 0.01 | `0.969834961` | 7 | `0.696510 / 1.284159` at epoch 64 |
| v28 reuse | Six-hour exact receptive field, 535,794 parameters | `0.967228995` | 7 | five-epoch medians `0.814542 / 1.079253` through epoch 39 |
| v23 | LR `1e-4`, earlier LR reductions | `0.970381474` | 14 | `0.875637 / 1.018604` at epoch 64 |
| v25 | independent policy-head dropout 0.05 | `0.970006480` | 7 | `0.847550 / 1.046350` at epoch 40 |
| v26 | policy dropout 0.05 and WD 0.05 | `0.969985332` | 7 | `0.854032 / 1.033740` at epoch 43 |

v25 and v26 were stopped after two consecutive five-epoch windows met the
predeclared divergence rule: median validation KL exceeded best+0.05 while
median training KL had improved by more than 0.03 from the train value at the
best epoch. Those four earlier runs retain their best and last checkpoints and
are resumable.

v28 is the strongest causal model in the screen. It consumes every one of the
21,600 one-second transitions in a six-hour exact receptive field while the
scale-statistics bypass remains restricted to the last 60 minutes. It has
535,794 parameters and uses the same direct T=0.01 target, seed, optimizer, and
scheduler as v18. Exact overlapping-core reuse keeps warm epochs near 20
seconds, and all 73 focused model, reuse, integration, and trainer tests pass.

The initially apparent full-score improvement over v18 was confounded by the
different valid row counts: v18 evaluates 43,080 validation rows, whereas v28
must exclude a 300-row prefix and evaluates 42,780. A read-only ordered-row
audit reproduced v18's full score as `0.969834926`, then scored that checkpoint
as `0.969076740` on exactly the v28 rows. The true matched v28 gain is therefore
`0.001847745`, only `0.000152255` below the predeclared `0.002` promotion gate.
Matched ordered rows, not the superficially larger full-score difference, are
the canonical comparison.

Because improvements can follow long plateaus, v28 continued through epoch 39.
The epoch-30–34 median train/validation KL was `0.845537 / 1.035897`; at epochs
35–39 it was `0.814542 / 1.079253`. Training kept improving while validation
diverged, and the epoch-7 best did not change. The run is cleanly paused at
epoch 39 and global step 14,508, with both best and last state resumable. It is
retained as the best causal diagnostic, but is not promoted to a long run or
the bot. Its training and validation passes read zero test payloads and zero
test references.

Other causal architectures were materially different but converged to the
same region: fast patch mixer `0.970121`, learned PatchTST-like aggregates
`0.972114`, boundary-incomplete minute TCN `0.971774`, residual mixer
`0.972652`, DCT residual mixer `0.973629`, patch transformer `0.975156`, and
minute MLP `0.980052`. Exact MA bands did not improve the corrected TCN.

The constant training prior scores `1.078395` raw validation KL and simple
causal volatility bins score `0.983599`. The best neural model therefore
explains only about 10.3% of the prior KL, while reaching KL 0.1 would require
explaining roughly 90.7% of it. Longer optimization cannot close that gap when
training improves and validation diverges.

## Additional causal-information audits

### A matured oracle is not a useful causal feature

The exact oracle distribution from 60 minutes earlier is available without
hindsight, so it was tested directly against the current target. Of 535,620
training rows, 534,540 have a valid predecessor; the 1,080 gap-start rows were
masked. All 43,080 validation predecessors are valid. The unmodified lagged
distribution scores raw validation KL `10.753011953`. A convex backoff selected
only on training uses lag weight `0.125996326` and prior weight `0.874003674`;
it scores `1.071323907`. That is a real `0.007071` improvement over the constant
prior, but it is `0.101489` worse than v18. The matured oracle is therefore
rejected as a model input or shortcut.

### Soft policy prototypes preserve outputs, not future information

A forward-KL prototype basis was fitted only from training targets, then its
representation error was measured on validation. The projected column is an
oracle-aware lower bound obtained by fitting mixture weights to each validation
target; it is not a causal prediction score.

| Prototype count | Nearest-prototype KL | Continuous-mixture projected KL |
| ---: | ---: | ---: |
| 8 | `0.073761` | `0.039570` |
| 16 | `0.034723` | `0.014198` |
| 32 | `0.016533` | `0.006312` |
| 64 | `0.010097` | `0.003853` |

This proves that a small learned policy basis can represent the oracle
distribution nearly losslessly. It does not prove that candle history can
select the correct mixture. In particular, prototypes are not cluster labels,
alternate targets, or extra information: the causal model must still infer
their continuous weights from the same history while optimizing the original
soft target.

The K16 v27 screen made exactly that test. It reused the v18 encoder, emitted a
continuous softmax mixture over the fixed 16×101 basis, and trained by direct
soft-label cross-entropy. Its best raw validation KL was `0.970259275` at epoch
7, `0.000424` worse than v18. Extending through the possible plateau did not
uncover a later improvement: by epoch 39 train KL had fallen to `0.787744`
while validation KL had risen to `1.160044`. The run is cleanly paused and
resumable at epoch 39, but the prototype head is rejected for a long run.

### Calendar and completed-minute OHLCV add too little to v18

Train-only calendar tables expose a genuine distribution shift. Standalone raw
validation KL is `1.074421` for hour of day, `1.038050` for hour × weekday,
`1.074452` for minute of day, and `1.037686` for minute of week, compared with
the constant prior's `1.078395`. Hour × weekday therefore retains nearly all
of the useful calendar signal with 60 times fewer cells than minute of week.

Fusion weights were selected on the first chronological half of validation and
evaluated on the untouched second half. The verified v18 epoch-7 checkpoint
scores `0.93752705` on the selection half and `1.00214279` on the clean half.
A convex calendar fusion selects weight `0.0081853` and scores `1.00169874` on
the clean half, a gain of only `0.000444`. A log-ratio fusion independently
scores `1.00180756`, a gain of `0.000335`. Both miss the predeclared `0.002`
promotion gate, so calendar features were not added to the neural trainer.

The OHLCV audit uses 32 scale-free features from completed one-minute candles.
For a target at `hh:mm:00.999`, the newest feature candle ends at
`hh:mm-1:59.999`; therefore no target-minute high, low, close, or volume leaks
into the input. Standalone raw validation KL is `0.98687612` for a matched
close-only regime, `0.98242126` for OHLC geometry, and `0.99393961` for direct
joint bins. A train-selected convex backoff with `45.6949%` close and `54.3051%`
geometry scores `0.979024530`: geometry adds `0.007852` over the matched close
control, but remains `0.009190` worse than v18. Volume was weak.

The clean v18 fusion audit is decisive. Direct convex fusion selected zero
OHLCV weight. Log-ratio correction selected weight `0.147792871` on the first
half and scored `1.00199325` on the untouched second half, only `0.00014954`
better than v18, or 7.5% of the promotion gate. The full validation score was
`0.96969171`, but it includes the half used to select the scalar and is not the
clean decision metric. OHLCV integration is rejected at this evidence level.

### Six-hour close summaries do not justify a v29 residual branch

The final close-only audit tests the low-capacity information hypothesis behind
the proposed frozen-v18 v29 branch without training another neural model. Its
13 strictly causal completed-minute features comprise trailing return and RMS
at 1h, 2h, 3h, and 6h plus an exact additive moving-average-band
decomposition. All regime tables and backoff weights were selected from
training data. The selected six-hour table retained return/RMS features at 1h,
3h, and 6h, yet received only `15.2956%` weight when backed off against the
selected short-context table.

The fusion audit uses exactly the 42,780 ordered rows available to v28. The
frozen v18 checkpoint scores raw T=0.01 KL `0.935643712` on the selection half,
`1.002509750` on the clean second half, and `0.969076731` in full. The primary
log-ratio fusion selected weight `0.549004` on the first half, then scored
`1.001682938` on the clean half, an improvement of only `0.000826811`. Its full
score is `0.968553937`, a gain of `0.000522794`. Both results miss the `0.002`
promotion gate. The extra six-hour scale/trend information is real but too weak
to justify implementing or training v29.

### Within-minute OHLCV microstructure is also exhausted

A separate paired audit derives causal one-second OHLCV features from preceding
completed minutes plus the current already-closed second. It compares each
path/range/volume microstructure table with an aggregate-only control under the
same train selection. The best microstructure family was path dynamics, but
every joint aggregate-plus-micro table was already worse than its control on
the training-internal calibration split.

On validation, frozen v18 scores `0.937527049` on the fusion-selection half and
`1.002142786` on the untouched second half. Convex fusion selected only
`0.032518` microstructure weight and reached clean KL `1.001903598`, a gain of
`0.000239188`. Log-ratio fusion selected exactly zero weight and reproduced
v18. Thus neither completed-minute OHLCV nor finer causal within-minute candle
geometry supplies a useful residual at the current gate.

Both candle audits are read-only: they trained no neural model and read zero
test references and zero test payloads.

### True Spot aggressor flow adds no usable residual

The next information gate used the official Binance Spot BTCUSDT `aggTrades`
archive rather than another candle transformation. A checksum-verified,
resumable ingestion produced 420 immutable one-second shards covering every
train/validation target day and required predecessor day. They contain
614,704,620 valid aggregate trades representing 1,371,148,191 constituent
fills from 8,872,038,319 compressed source bytes. Aggregate IDs, row totals,
constituent-fill totals, timestamp units, object hashes, and per-second base
volume against the canonical candle corpus all reconciled. No sealed-test date
was downloaded or opened.

The audit exposed 85 strictly causal trade-flow features at 1s, 5s, 60s, and
5m scales: aggressive buy/sell quote and trade-count imbalance, activity,
order-size concentration/skew, VWAP and arrival-time gaps, side flips, last
aggressor, and rate surprises. Together with 32 OHLCV controls and two
price-sign volume controls, this made a 119-feature screen. Every directional
regime had an exact absolute-magnitude control; regime, bins, shrinkage, and
backoff were selected only on embargoed training data.

The largest train-internal directional reduction was only `0.000139600` raw
KL. The selected true-flow-versus-price-sign regime assigned its directional
table `3.6855%` weight. On reused validation development data, both the
predeclared log-ratio residual and the convex diagnostic selected exactly zero
flow weight. Frozen v18 therefore remained `0.937527057` on scalar-fit rows,
`1.002405121` on the forecast-embargoed within-audit holdout, and
`0.969834926` on all validation rows. The gain is exactly zero against the
`0.002` promotion gate, so a neural Spot trade-flow branch is not justified.

### USD-M positioning metrics also add no usable residual

The next gate used the official Binance USD-M BTCUSDT five-minute metrics
archive: open interest and notional open interest, top-trader account and
position long/short ratios, the global account ratio, and futures taker
buy/sell ratio. Checksum-pinned ingestion produced the exact 420 immutable
train/validation/predecessor references and no sealed-test reference. Every
row is delayed by one complete five-minute bin. Missing fields are carried
forward causally with independent age/observed indicators; non-grid source
timestamps are discarded rather than rounded backward.

The 420 archives contain 120,947 source rows. The dense representation retains
120,692 on-grid rows, marks 268 grid slots missing, discards 51 off-grid rows,
and drops 204 next-day boundary rows. The latter creates conservative midnight
staleness rather than leakage and affects 0.17% of corpus slots. All stored
rows have `timestampAdjustedRows=0`; source checksums, payload hashes, axes,
per-field validity masks, quality counters, and the five-minute availability
contract reconcile.

The screen exposed 80 causal positioning features plus completed-minute OHLCV
controls. Paired train-only regimes covered open-interest pressure, notional
pressure, implied futures-price dislocation, futures taker pressure, global
crowding, and top-trader divergence. The largest train-internal joint-over-
control reduction was only `0.000358597`; the selected implied-mark-dislocation table
received `4.8539%` weight.

A stricter complementarity pass retained independently train-selected tables
for all six regimes, then selected regime plus residual scalar only on the
first validation half. Five regimes selected exactly zero residual weight;
top-trader divergence won a numerical tie with weight `0.000014708`. It also
made no measurable change. Frozen v18 remained `0.937527057` on scalar-fit
rows, `1.002405121` on the embargoed within-audit holdout, and `0.969834926` on
all validation rows. Every candidate's holdout gain was `0.000000`. The neural
futures-positioning branch is rejected.

### Completed USD-M/Spot basis and flow add no residual signal

The paired source gate then joined all 420 checksum-pinned USD-M one-minute
kline references with the same 420 Spot aggTrade references. Target minute
`k` uses only futures candle and Spot flow minute `k-1`; source-row presence,
official no-trade rows, live prices, and causal observation ages remain
separate. The 604,800 futures rows comprise 604,782 live rows and 18 official
no-trade rows, with no missing, off-grid, filled, or adjusted rows. No sealed
test reference or payload was opened.

Five train-fitted paired regimes covered basis level/reversion, basis change,
futures taker pressure, cross-market flow divergence/agreement, and relative
activity. The first validation half selected basis change. Its train-selected
signed-table backoff was exactly zero, so the nominal first-half residual
scalar of `4` multiplies an identically neutral log ratio and changes no
probability. After the 60-row horizon embargo, frozen v18 remains
`1.002405121`; full validation remains `0.969834926`. The gain is exactly
`0.000000`, below the `0.002` gate, so no neural basis/flow branch should run.
The saved audit is
`data/runtime/logs/v18-futures-basis-audit-2026-08-02-r1.json` (SHA-256
`69728272155A52E20F4265EB90E9B8BE3DA26CD99832CE2E6D68DB18A897B294`).

The next bounded information gate is the compact historical USD-M order-book
depth source, with corrupt/stale snapshots masked rather than repaired.

### Soft-temperature transfer does not recover the sharp target

The temperature audit fitted each estimator and any scalar output calibration
on training only. Every number in this table is raw validation KL against the
original T=0.01 oracle distribution.

| Estimator | Source target T | No transform | Analytic inverse sharpening | Train-calibrated scalar |
| --- | ---: | ---: | ---: | ---: |
| Price regime | 0.01 | `0.989364` | `0.989364` | `0.989222` |
| Price regime | 0.02 | `1.069619` | `1.067896` | `1.014597` |
| Price regime | 0.04 | `1.244386` | `1.218024` | `1.082243` |
| Price regime | 0.08 | `1.408614` | `1.377818` | `1.204671` |
| Calendar | 0.01 | `1.037686` | `1.037686` | `1.037710` |
| Calendar | 0.02 | `1.117904` | `1.087530` | `1.054079` |
| Calendar | 0.04 | `1.292897` | `1.209306` | `1.123140` |
| Calendar | 0.08 | `1.447678` | `1.409598` | `1.282853` |

The train-selected price calibration powers for source temperatures 0.02,
0.04, and 0.08 were only `1.493`, `2.456`, and `4.125`, rather than the analytic
`2`, `4`, and `8`. Sharpening the conditional mean of softened distributions
cannot in general recover the conditional mean of sharp distributions. Even
the calibrated softened estimators lose to direct T=0.01 training. This audit
does not support a neural softened-target screen, while the existing decoder
0.04→0.02→0.01 curriculum already lost to direct training at matched compute.

### Archived second-phase targets are contract-incompatible

The old 33-million-example decoder corpus does contain raw oracle components at
additional within-minute phases, but they were generated under an older oracle
contract and cannot be relabeled as the current 1h/1m/1m target. On a matched
day and action-grid subset, its nominal phase-zero target differs from the
current target by mean KL `2.2495` and maximum KL `28.74`. Nearby shifts do not
repair the mismatch: mean KL is `2.315` at -1 second, `2.283` at +1 second, and
`6.17` at +60 seconds. Exact all-phase augmentation would require recomputing
the current oracle contract; reusing these archived payloads would silently
train on a different policy.

All audits in this section used training and validation payloads only. Their
test-payload and test-reference read counts are zero.

## What should run next

1. If GPU time is assigned to a long capability run, use only the promoted
   decoder plan. It is a clean test of decoder saturation, not a deployable
   causal bot.
2. Do not spend a 200-epoch run on the current candle-only deterministic
   predictors or causal policy variants. v28 is the closest causal candidate,
   but its exact-row gain still misses the promotion gate and its extension
   overfits. The matched six-hour and one-second OHLCV residual audits also
   reject v29 and further candle-only feature branches.
3. Any next causal predictor screen must add information beyond Spot price,
   Spot trade flow, standalone futures positioning, and completed-minute
   futures/Spot basis-flow. The next bounded gate is strict historical USD-M
   order-book-depth imbalance; later options are related market/asset context
   or other timestamped exogenous data. If those inputs are modeled
   with a probabilistic multi-future latent representation, it must marginalize
   its future heads to one final action distribution and still select by raw
   validation KL; clustering labels is not a substitute for direct KL.
4. Only after a new non-candle causal input/architecture beats its exact-row
   v18 control by at least `0.002` raw KL and improves a five-epoch rolling
   median by at least `0.001` should it be repeated across two additional seeds
   or considered for a long run.

The unrelated archived fitter process remains OS-suspended at its original
checkpoint and can be resumed independently. None of these screens accessed
the sealed test payloads or replaced the hindsight oracle in the live bot.
