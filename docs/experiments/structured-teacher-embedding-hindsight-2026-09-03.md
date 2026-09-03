# Direct teacher-embedding hindsight experiment

Plan: `ml/training-plans/structured-production59-w256-teacher64-embed512-r128-depth13x3-s16-affine256-gate099-k2x2-train256k-b10240-e256-ema4-v1.json`.

Fresh W256 student, K1=K2=2, original dense production59 population, 256,000
training examples, 65,534 validation/test origins. Three total independently
learned layer13 blocks, S16, compressed sample width256, R128 reset per forecast
second, hidden512 readouts, equal train/evaluation batch10,240, 256 epochs,
EMA4, no SAM/dropout/adversarial loss/weight decay. The student predicts the
complete59 feature vector with the same per-sample standardized MSE objective.
Student: **11,090,610 trainable parameters**; external frozen teacher:28,385,589.

## Teacher and direct connection

Frozen 64-key, all512, base-support embedding-density teacher, checkpoint
`best-validation-correlation` after13 completed epochs (original selection was
feature-wide validation correlation). Its saved next1s validation return-only
skill/correlation are5.1366%/0.249811. No test-selected checkpoint is used.

Teacher SHA256: `c3f71a19c1c0072da72ea5ac056e2fe0bfb03163bf875e81a08c3c4c1682edc2`.
A copied immutable pointer under this run's `checkpoints/dependencies/teacher.json`
pins those weights against checkpoint garbage collection. A checked teacher-plan
snapshot lives at `state/teacher-plan.json`. Teacher parameters are not members
of the student model, optimizer, or EMA and never receive gradients.

For each origin the frozen teacher predicts64 primitive support paths and their
per-step masses. Each full support path is causally derived into59 feature states
before sampling. Sample16 per-step component centers using those prior masses;
encode the selected states with frozen teacher layer1 and unit-RMS normalization.
This is discrete component-center sampling, **not** sampling the sigma4 Gaussian
spread used by the teacher's embedding NLL. It makes no additional joint-path
sampling claim for the teacher's separately emitted per-step masses.

The clean endpoint is the true future59 features encoded through that same frozen
encoder. The curriculum is `z=(1-v)*encode(target)+v*sampled_teacher_center`.
Here v means **teacher replacement fraction**, not Gaussian noise variance.
It starts0 and advances0.01 after the training-probe next1s correlation reaches
0.99, otherwise holding its value. The existing numerical noise-variance fields
remain the dashboard's scalar channel, accompanied by explicit parameterMeaning.

The S16 x512 embedding values concatenate to8192 and go directly through the
learned affine8192->256 compression. R128 is appended, followed by the existing
conditioned layer2 and three stream-only layer13 residual blocks. The existing
final affine256->16x59 produces outputs. There is **no intermediate feature-space
decoder**. Both affines learn; an embedding mean is not assumed to decode to the
teacher's exact mean features, and initialization is not an identity passthrough.

## Evaluation and leakage controls

Forecast evaluation always sets v=1. It passes no targets into the teacher
inference function. Teacher context is constructed directly from origin-time
causal histories, excluding future primitive targets. Fixed split-specific
uniform streams select components invariant to batch partitioning; training uses
an independently seeded stream per epoch. Teacher inference is internally chunked
at512 examples, independent of the student's unchanged logical batch10,240.
No large embedding cache, quantization, or trainable teacher is introduced.

The teacher trained on the same training population, so this is an in-sample
distillation pilot, not out-of-fold stacking. The training reconstruction gate
can be optimistic. Target-free held-out forecasting metrics determine success;
current-fraction validation reconstruction is diagnostic only.

The input audit verifies matching origins, endpoint invariance to replacing true
targets with NaNs, exact agreement of clean hindsight with the frozen teacher
encoder, finite sampled embeddings, and frozen teacher parameters. Its results
are persisted in `state/teacher-input-audit.json`. Unit tests cover direct
512-wide input compression to feature outputs, endpoint isolation, component
sampling, finite gradients, independent readout blocks and per-step register reset.
All47 structured unit tests passed. The full-batch compiled CUDA smoke completed
one optimizer step with finite loss1.63130, two per-step evaluation rows, and three
finite checkpoint policies. Its teacher-replacement source/512-dimensional inputs
were verified from the persisted training event; only the benign PyTorch
insufficient-SMs autotune warning appeared. Fresh full training uses
`--replace-smoke --compile-mode default`; all smoke/launcher logs are retained.

Full-run verification: fresh startEpoch0, 256 planned epochs, and all three exact
launcher/worker processes alive. By completed epoch12, steady-state epochs took
about21-22 seconds (first epoch56.94 seconds including compilation). The full
launcher stderr was empty. Last, best-MSE, and best-correlation immutable student
checkpoints were loaded on CPU and contained finite weights. The teacher pointer,
snapshot, audit, and preserved smoke log remained present.

At completed epoch12, the training reconstruction gate correlation was0.95748,
so replacement remained0. Target-free next1s validation return correlation was
0.18655 and MSE skill3.34598%. These are early training observations, not evidence
of superiority to the frozen teacher or a completed-run comparison.

## Extension to 512 total epochs

The user requested continuing the same experiment to512 total epochs. The original
plan and its SHA256 remain unchanged so all existing checkpoint policies retain
their exact identity. A plan-hash-bound `state/epoch-limit.json` sets the active
total to512 and is read on every launch, including future resumes. The trainer
reports both the original plan total and the active total in `training-start`.
Dashboard metadata uses the reported runtime total, and `state/display.json`
updates this run's label without rewriting the immutable plan snapshot.

The extension does not change any teacher, student, optimizer, EMA, batch, dataset,
loss, or0.99-gate setting; it does not add target-free training examples. The runner
requires a checkpoint resume because its existing live loop captured256 at startup.
The extension tests reject mismatched plan hashes and invalid or reduced totals.
All48 structured Python tests and8 dashboard metrics tests passed; server TypeScript
typechecking passed. No checkpoints, logs, or evaluation artifacts are removed.

Verified handoff: checkpoint epoch167/globalStep4200 was complete and finite;
only the exact old Node/Python process tree was stopped. The new hidden launcher
resumed at startEpoch168 with the saved optimizer, EMA, and curriculum noiseStep74.
Its first full epoch completed at globalStep4225 (169 completed epochs), with
finite loss0.00924152 and gate correlation0.990347, advancing the next replacement
fraction to0.75. All three worker/launcher processes were alive, stderr was empty,
and the dashboard reader reported512 total epochs. Original launcher/smoke logs,
selected checkpoints, the frozen teacher dependency, and the input audit remained.

## Saved epoch342, then 32 target-free epochs

The user superseded the512-epoch gated continuation: pin the current checkpoint
and train32 more epochs with no true-future conditioning. The exact latest complete
checkpoint was zero-based epoch341/globalStep8550 (342 completed epochs), at
replacement0.83. Its raw gate correlation was0.793211; the EMA target-free next1s
validation correlation/skill were0.149446/-5.95818%. No better checkpoint was
substituted for the requested current one.

The full raw model, EMA, optimizers, best selections, and old curriculum are pinned
at `checkpoints/milestones/before-target-free-e342.json`, immutable object SHA256
`307227a76c3ac158644acd26bc2bac47f7bc5d783ecade45dfcfd36a7517ac9e`.
This milestone is separate from the rolling last pointer; the standard cleanup
script does not remove milestone pointers, and its object remains referenced.

`state/target-free-phase.json` binds the32-epoch phase to that source object and
the unchanged plan hash. It fixes replacement1.0, disables the correlation gate,
and resumes the same model/optimizer/EMA at zero-based epoch342. The total limit
is374; the phase covers epochs342..373. Teacher samples are the only side inputs.
Actual future features are still MSE labels, but are not supplied to the teacher,
embedding encoder, or student as conditioning. Sampling, batch10,240, loss, teacher,
architecture, and learning rate remain unchanged. Redundant target-assisted gate
probes are no longer run; target-free train/validation evaluation and final
best-MSE/best-correlation/last evaluation remain in place.

Phase metadata is persisted inside each new last checkpoint. Resume validation
checks the pinned source at the first phase epoch and checks phase identity for
subsequent resumes. All50 structured tests passed, including the constant endpoint,
source identity, exact epoch budget, and within-phase resume guards. Old training,
smoke, and launcher logs remain intact; new launcher logs are timestamped.

Launch verified through phase epoch1 / total completed epoch343, globalStep8575.
The event records replacement1, signalScale0, trainingTargetContribution0, and
gateEnabledfalse. Loss0.261199 was finite; both per-step evaluation rows persisted.
First-phase EMA next1s validation skill/correlation were1.97191%/0.162544 (early
observation, not a final comparison). The new last checkpoint contains the phase
metadata and updated finite weights/EMA, and passes within-phase resume validation.
The pinned epoch342 object is unchanged. All three exact launcher/worker processes
were alive and stderr was empty. The first phase epoch took56.05 seconds including
startup compilation; the phase stops after32 completed epochs at total374.

## Previous run

The Gaussian S16/depth3/0.99 predecessor completed all256 epochs and persisted
best-MSE, best-correlation and last evaluations. Its final raw reconstruction at
variance0.80 had training correlation0.9895868 and validation0.9908181; final
target-free validation forecasting correlation was0.0421253 with MSE skill-0.6709%.
Three checkpoint pointers and three unreferenced objects were deleted only after
evaluation. All12 other run files passed unchanged SHA256 verification; training,
smoke and launcher logs, plan, status, results and evaluations were preserved.
