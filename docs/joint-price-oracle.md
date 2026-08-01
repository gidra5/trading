# Causal 1-second joint price/oracle model

This experiment predicts the raw 255-action oracle policy without putting any
post-decision price into the network input. It replaces the controlled
60-minute-hindsight reconstruction task with an actual forecasting task:

```text
past closes through t
  -> causal trend/residual decomposition
  -> next 3,600 one-second close forecasts
  -> learned raw-oracle policy at t
```

The oracle still uses realized future prices to create the supervised label.
That is target construction, not an input available to the model. At inference,
the graph receives closes ending at `t` and nothing after `t`.

## Timestamp and split contract

For every prediction timestamp `t`:

- the input is the 3,600 one-second BTCUSDT closes ending at `t`;
- the auxiliary price target is the 3,600 closes from `t + 1s` through
  `t + 3,600s`;
- the policy target is the source dataset's exact
  `rawOracleProbabilities` row timestamped at `t`;
- `predictionDelayMs` is exactly `0`; and
- the forecast horizon equals the source oracle's 3,600-step value horizon.

The loader constructs the close windows directly from the timestamped
historical one-second files. It does not use the source MLP's 901-value feature
encoding. Train, validation, and test windows are purged at split boundaries on
both sides: no input context or forecast label can cross into a differently
assigned split. The last 1,000,000 chronological test examples are reserved.

The reused source component store does not contain same-timestamp raw-oracle
days for `2023-06-17` and `2023-09-24`; those dates are explicitly excluded
instead of silently retargeted or shifted. The current checked corpus has:

- 32,932,807 training examples;
- 15,393,625 validation examples; and
- 1,000,000 test examples.

## Forecast architecture

The model consumes positive raw closes, converts them to log closes for stable
additive decomposition, and forms two exact streams:

```text
trend_t = hybrid(causal MA_t, learned causal patch aggregate_t)
residual_t = log(close_t) - trend_t
```

The learned aggregate uses overlapping trailing 60-second patches with one
learned softmax weighting per variable. Its uniform initialization is exactly
the 60-second causal moving average. A learned sigmoid mixes the fixed and
learned aggregates, so training can retain the MA, replace it, or use both.
Changing a later input close cannot change an earlier aggregate.

Each trend/residual stream is independently standardized across its 3,600
historical positions to per-example mean zero and variance one. The statistics
are detached and retained for the matching inverse transform. This is the
RLinear-style reversible instance-normalization path; no validation/test
statistics enter training.

The model module accepts `[example, time, variable]` closes and keeps learned
patch weights and temporal linear factors independent by variable. The current
BTCUSDT corpus supplies one variable; extending the dataset loader to aligned
symbols does not require changing the network contract.

Each normalized stream is sent through both:

- a separate rank-64 temporal linear projection initialized to exact
  last-value persistence (the factorized DLinear/RLinear path); and
- a four-layer, 128-wide TiDE encoder and 3,600-step temporal decoder.

The TiDE decoder is a learned residual on the linear forecast. Trend and
residual forecasts are independently denormalized and added in log-price space,
then exponentiated to positive close forecasts. Before training, this produces
an exact no-change forecast at every horizon rather than arbitrary prices.
The production configuration has 19,302,638 trainable parameters, fewer than
the 32,932,807 selected training examples.

The GLU layers reuse the active return-oracle model's design:

- fused value/gate projections;
- a fixed canonical centering matrix shared by the layer's main and
  global-residual value/gate paths;
- learned-radius tanh RMS normalization;
- full value and gate transforms;
- post-transform offsets; and
- a GLU global-input residual into every layer.

## Policy architecture and objective

Only the predicted 3,600-step log-movement path enters the learned policy head.
There is no shortcut from observed closes to policy logits. Four
oracle-style 192-wide GLU layers produce the 255 raw base-action logits. Policy
cross-entropy therefore propagates through the forecast decoder, while an
explicit price loss keeps the intermediate movement path identifiable.

The joint loss is:

```text
L = CE(raw oracle policy, predicted policy)
  + Huber(
      (predicted cumulative log move - realized cumulative log move)
      / (causal past volatility * sqrt(horizon))
    )
  + 0.01 * soft-LayerNorm penalty
```

Validation checkpoint selection uses raw-policy KL divergence. The runner also
reports probability MSE, normalized forecast loss, next-second log-movement
RMSE, next-second direction accuracy, and the normalization penalties.

## Operation

Validate all real-data timestamp and component pairings without using CUDA:

```text
npm run mlp:experiment:joint-price-oracle:check
```

Start or resume training:

```text
npm run mlp:experiment:joint-price-oracle
```

The plan is
`ml/training-plans/joint-price-oracle-tide-rlinear-dlinear-v1.json`.
Checkpoints and observability files are isolated under
`data/training/runs/joint-price-oracle-tide-rlinear-dlinear-v1`. `last.pt` includes
the model, optimizer, scheduler, RNG states, exact model/data contracts, and a
fingerprint of every selected segment. A checkpoint is rejected if any of
those resume-critical values changes.

The batch loader keeps a bounded CPU prefetch queue so gzip/zstd decoding and
window construction overlap the current GPU update without duplicating the
large per-day oracle cache.

This experiment deliberately does not start automatically while another CUDA
training run owns the GPU. Its source, plan, status, and checkpoint paths do not
modify the active `return-oracle-ce-shrinking-v1` run.
