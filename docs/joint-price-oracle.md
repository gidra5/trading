# Causal 1-second joint price/oracle model

This experiment predicts the verified `1h horizon / 1m execution delay / 1m
hold` oracle without exposing any post-decision candle to the model:

```text
3,600 one-second closes through decision time t
  -> causal trend/residual decomposition
  -> next 3,600 one-second close forecasts
  -> 101 usable oracle-action logits at t
  -> current-exposure conditioning
  -> regular bot target-exposure execution
```

The realized future is used only to construct the supervised oracle label and
the auxiliary forecast loss. At inference the ONNX graph receives closes
ending at `t` and nothing after `t`.

## Target and split contract

The canonical targets are the immutable references under
`data/training/immutable/refs/oracle/1s/hindsight-bot-71391c44b323e044e6ab`.
Every reference is contract-checked before training:

- input interval: 1 second;
- decision interval and hold: 60 seconds;
- oracle execution delay: 60 seconds;
- value horizon: 3,600 seconds;
- latent exposure grid: 255 actions over `[-250, 250]`;
- deployed policy grid: the 101 actions inside approximately `[-100, 100]`;
- friction: 17.5 bps; and
- long/short maintenance: 10 bps/hour.

There is no additional response lag: the close context and oracle decision are
both timestamped at `t`. Labels are one minute apart, while every input and
forecast position remains one second apart.

The 432 available target days are split chronologically. The last 30 days are
untouched test data, the preceding 30 are validation, and the earlier 372 are
training. Input and future-price windows are removed around split boundaries
so a close used by one split cannot enter another split's model window. The
checked corpus contains 535,620 training, 43,081 validation, and 43,141 test
decisions.

## Architecture

Positive closes are converted to log closes and decomposed exactly:

```text
trend_t = hybrid(causal 60s MA_t, learned causal 60s patch aggregate_t)
residual_t = log(close_t) - trend_t
```

The learned patch aggregate is implemented as a grouped causal convolution.
Uniform weights equal the trailing moving average exactly. A learned sigmoid
can retain the MA, replace it, or mix both.

Trend and residual are normalized independently to per-example mean zero and
variance one. Each stream then has both a rank-64 RLinear/DLinear temporal path
and its own four-layer, 128-wide TiDE encoder/3,600-step decoder. The two
forecasts are inverse-normalized and recombined in log-price space. Initial
weights produce exact no-change persistence.

Only the resulting forecast movement path enters four 192-wide policy layers.
Those layers use the return-oracle design: each main and residual path has a
single fused affine value/gate projection, then shared canonical centering,
learned-radius tanh RMS normalization, full value/gate transforms, and fused
GLU activation. The model has 18,994,388 trainable parameters with the 101-way
verified policy output.

The v3 objective combines raw policy cross-entropy, exact
transition/friction-conditioned policy cross-entropy at short, flat, and long
exposure anchors, a volatility-scaled cumulative forecast Huber loss, and a
0.01 soft-normalization penalty. Best-checkpoint selection uses conditioned
validation KL divergence.

The value and gate projections are fused: every main and residual policy path
uses one affine projection whose output is split into value and gate halves.
They are not two separately executed projections.

## Training, export, and deployment

The active plan is
`ml/training-plans/joint-price-oracle-decision-conditioned-v3.json`. It starts
from the best v2 weights and owns a separate optimizer and checkpoint stream.
Direct Python invocation is safe while another checkpoint object is open; the
generic training wrapper also performs storage maintenance and should not be
started concurrently with an unrelated trainer.

```text
# validate all target contracts and one real batch per split
.venv-ml/Scripts/python.exe ml/train_joint_price_oracle.py \
  --plan ml/training-plans/joint-price-oracle-decision-conditioned-v3.json \
  --check-data

# start or resume
npm run mlp:experiment:joint-price-oracle

# export the current best checkpoint and verify ONNX against PyTorch
npm run mlp:experiment:joint-price-oracle:export

# sweep action temperatures on validation only and bind the selected report
# to the exported artifact
npm run mlp:experiment:joint-price-oracle:calibrate

# run the chronological 30-day bot holdout
npm run mlp:experiment:joint-price-oracle:backtest
```

V3 training is durably paused after epoch 17 at
`data/training/runs/joint-price-oracle-decision-conditioned-v3-1h-1m-1m`;
epoch 10 is the best conditioned-validation checkpoint. Resume it with
`npm run mlp:experiment:joint-price-oracle`. The raw v2 run remains independently
resumable after epoch 16 with
`npm run mlp:experiment:joint-price-oracle:v2:resume`.
`checkpoints/last.json` and `checkpoints/best.json` are immutable artifact
pointers; resume verifies the model, data contract, split fingerprint, and RNG
state.

The exported best artifact is
`data/models/joint-price-oracle/joint-price-oracle-decision-conditioned-v3-1h-1m-1m`.
Its ONNX SHA-256 is
`255fda10ab378ec0fb2d97f1e46d4313d9449113e7a854e4e891d0cc3efa7e6c`;
maximum absolute ONNX/PyTorch logit error is `6.67572021484375e-6`.

The earlier calibration selected `1.3717149224602072` solely because it reduced
raw validation KL from `1.2620121940` to `1.2282814329`. That is not an
executable-policy calibration: the extra softening let transition/friction
conditioning choose flat everywhere. The value remains a historical
diagnostic and must not be promoted to a new artifact.

Calibration now runs the Python action evaluator once over the complete,
ordered validation split and scores every requested logit temperature on the
same logits. Selection maximizes signed transition F1. Exact ties prefer, in
order, signed precision, signed recall, exact transition F1, path direction,
lower path error, target-like turnover, distance from identity, and finally
the lower numeric temperature. Raw KL is retained as a diagnostic but is not a
selection key. The calibration command has no test-split option and always
loads the best checkpoint with `split=validation` and `allow_test=false`.

The full sweep, raw metrics, executable action/path metrics, resolved plan,
checkpoint epoch/step, split timeline, and dataset fingerprint are written to
`calibration.json` beside the ONNX model. The manifest records the report's
SHA-256, the deterministic selection contract, selected and identity metrics,
and the applied temperature. When the plan opts into an
`actionObjective.executionPolicy`, the evaluator uses that resolved leverage
cap and confidence policy for every candidate and records it in both the
report and manifest together with the rollout-score contract version. The
server verifies that hash and provenance before
loading an action-calibrated artifact. More aggressive temperatures in
the legacy v3 model did trade but failed economically: at temperature `0.05`,
a five-day validation backtest returned `-28.8721%` with 925 trades, `2437.17`
in fees, and `470.91` in maintenance charges. That model has not met the
profitability acceptance gate.

The one-time chronological test split covers 2026-06-24 through 2026-07-23
(43,200 decisions and 2,592,000 one-second candles). At the validation-selected
temperature and the learned policy's 1x cap it made zero trades, returned
`0%`, and had zero drawdown. Its raw modal action was nonzero for 21,856
decisions, but transition/friction conditioning rejected every one. The stored
hindsight target produced 4,470 nonzero conditioned decisions and 495,512 bot
fills at its separate 100x ceiling. This is a rejection result: the learned
distribution is too diffuse to clear costs, not a profitable oracle
replacement. The full report is
`data/benchmarks/joint-price-oracle-decision-conditioned-v3-1h-1m-1m.json`.

Historical backtests select `learned-oracle-1s`, precompute only causal
distributions, and pass them through the same target-exposure strategy and
simulator as the hindsight ceiling. Live use remains deliberately opt-in:

```text
TRADING_INTERVAL=1s
TRADING_JOINT_PRICE_ORACLE_MODEL=latest
TRADING_JOINT_PRICE_ORACLE_MAX_LEVERAGE=1
```

The learned policy defaults to a hard 1x leverage cap. The live strategy
retains 3,600 one-second closes, infers at the target's exact decision phase
(`closeTime % 60000 == 999`), conditions the base policy on actual marked
exposure and friction, and holds through the standard one-minute bot cooldown.

The previously running archive experiment is resumable with:

```text
.venv-ml/Scripts/python.exe ml/resume_return_oracle_archives.py
```

Its durable pause point is recorded in
`data/training/archive-resume/status.json`.

## 2026-08-02 capability-screen update

The decoder, deterministic price-predictor, and corrected causal-policy
screens are consolidated in
[`joint-price-oracle-capability-screen-2026-08-02.md`](joint-price-oracle-capability-screen-2026-08-02.md).
The learned-radius shrinking decoder is the only component promoted to a long
saturation run. Candle-only deterministic predictors and direct causal policy
variants did not pass their validation promotion gates, so the integrated
learned oracle remains rejected for live use.
