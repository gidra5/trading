# Correlation-gated one-pass hindsight training

Fresh run requested by the user, starting from new weights and variance zero:
`structured-production59-w256-selfattn256-k1-2-k2-2-hindsight-corrgate-step001-train256k-b10240-ema4-v2`.

Keep the predecessor's W256 self-attention architecture, two preceding 1s
production-59 states, two future 1s states, 4,270,974 parameters, standardized
feature MSE, equal train/evaluation batches of 10,240, and EMA 4. Keep the
existing 512-epoch budget. No additional regularizers.

Replace the timed cosine ramp with a curriculum:

1. Train at the current noise variance `v`, initially zero.
2. At each completed epoch, measure next-1s return Pearson correlation using
   the current raw training weights and noisy hindsight at that same `v`.
3. If correlation is at least `2**(-1/3600)` (approximately 0.9998074777), set
   the next epoch's variance to `min(1, v + 0.01)`. Otherwise leave it unchanged.

The increment is in **variance**, not standard deviation. Corruption stays
`sqrt(1-v)*standardized_target + sqrt(v)*standard_normal`. There is at most one
increment per complete epoch. No time-based fallback or threshold relaxation.
The model may remain at one variance for the rest of the epoch budget; reaching
pure noise is not guaranteed.

The gate uses a fixed 65,536-example **training-only reconstruction probe**,
with its own deterministic Gaussian seed, 982451653. It is not held-out forecast
accuracy, feature-wide correlation, direction accuracy, or EMA accuracy. The
probe and its next-step decision are logged under the epoch's `hindsight` field.

All normal train/validation/test forecasting evaluations still receive pure
Gaussian noise, regardless of the curriculum stage. Validation next-1s MSE and
correlation continue to select best checkpoints. The gate does not use test or
validation targets and does not change the displayed forecast metrics.

Persist the integer noise step, last completed gate epoch, and last gate
correlation inside the durable last checkpoint. Resuming must restore this
state; it must neither return to zero nor replay a successful increment.

The predecessor was stopped after zero-based epoch 210. Best-validation-MSE,
best-validation-correlation, and last were evaluated. Both best policies chose
epoch 71: validation next-1s MSE skill 6.2621295%, correlation 0.2504473.
Its logs and evaluation artifacts are preserved during checkpoint cleanup.

Verification: 32 structured-model tests passed, including threshold boundary,
no timed advancement, single-increment/cap behavior, checkpoint state, and
training-only probing. A CUDA compiled smoke passed and was resumed from its
checkpoint successfully. The full run then restarted fresh and completed its
first two full epochs at variance zero. The probe used all 65,536 configured
training examples; full-run stderr was empty, and the dashboard API reported
the new run as active. Three unreferenced predecessor checkpoint objects were
removed; training, smoke, launcher, result and evaluation logs/files remain.
