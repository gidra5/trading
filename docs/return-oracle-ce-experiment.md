# 60-return close-only oracle cross-entropy experiment

This standalone experiment asks whether the close-only 255-action oracle policy
can be reconstructed from exactly the price path that generated it. It is intentionally
not a deployable predictor and is not exposed in the inspector.

## Pairing

For each one-second inspector example with prediction timestamp `p`:

- the target is the source dataset's stored `minuteOracleProbabilities`;
- both target and input use the close-only one-minute path ending at the latest
  completed UTC minute available at `p`;
- 61 completed one-minute close values produce 60 adjacent simple returns;
- the same aligned input/target pair repeats for source seconds until another
  one-minute candle completes, so storage and GPU batches collapse those exact
  repeats to one row with an equivalent multiplicity weight;
- each of the 60 return positions is standardized with its own population mean
  and standard deviation calculated from the selected training split only;
  validation and test examples reuse those frozen training statistics; and
- the output cells retain the source dataset's complete 255-cell action grid.

The experiment reuses the original source corpus and retains the prior random
training chunks. For the fully normalized residual model, seed `20260731`
selects nine additional non-overlapping 10-day chunks from unused complete
pre-test history, adding 90 training days. The resulting source corpus
contains 51,321,600 examples. It applies a
60-minute purge whenever the chronological source split changes so no
one-minute input candle can cross from one split into another. After the purge:

- training has 33,195,600 examples;
- validation has 15,418,800 examples; and
- testing uses exactly the final 1,000,000 chronological test examples.

## Network

The graph is:

```text
60 inputs → 256 → 256 → 256 → 256 → 256 → 256 → 256 → 256
→ 255 logits
```

All eight current hidden layers have width 256. The displayed widths are
effective hidden widths. Every layer directly
projects its `d`-wide input into raw value and gate-logit branches, normalizes
those branches with the target layer's fixed shared centering matrix and
learned-radius tanh RMS denominators, applies separate full transforms and post-transform
offsets, then applies the GLU:

```text
(a0, b0) = split(Px + c)
C_a_initial = C_b_initial = I - 11^T / d
a_centered = C_a a0
b_centered = C_b b0
r_a = sqrt(mean(a_centered^2))
r_b = sqrt(mean(b_centered^2))
s_a = softplus(rho_a) + 1e-4
s_b = softplus(rho_b) + 1e-4
g_tanh(u) = u / tanh(u)
a = A_a (a_centered / (s_a g_tanh(r_a / s_a))) + beta_a
b = A_b (b_centered / (s_b g_tanh(r_b / s_b))) + beta_b
h = a * sigmoid(b)
r0_v, r0_g = split(W_res x0 + b_res)
r_v = A_res,v Norm(C_res,v r0_v, s_res,v) + beta_res,v
r_g = A_res,g Norm(C_res,g r0_g, s_res,g) + beta_res,g
r = r_v * sigmoid(r_g)
h = h + r
```

The former square input transform `B` is absorbed into both halves of the
fused matrix because moving LayerNorm after projection makes the composite
matrices directly learnable: `P_a = W_a B` and `P_b = W_b B`. Thus there is
one fused matrix multiplication before the two branch norms and no separate
input transform.

The fused projection is algebraically identical to separate affine transforms
for `a0` and `b0`, while using one matrix multiplication. Each hidden layer
uses one fixed centering matrix `C` shared across the main value,
main gate, residual value, and residual gate branches. The main and residual
value branches share `A_v`; the main and residual gate branches share a
separate `A_g`. Branch radii and biases remain independent. `C` stays at the
canonical centering projector `I - 11^T / d`, while `A` starts at identity.
RMS
normalization after `C` is replaced by a learnable family with one positive
radius `s` per branch. For the tanh family the expression simplifies to
`C x tanh(r/s) / r`, whose output RMS is `tanh(r/s)`: linear near zero and
saturating at one for large inputs. All 16 radii start at
`sqrt(1e-5) = 0.0031622776601683794`. The radii use
`softplus(rho) + 1e-4` to remain positive and can learn the transition between
the near-zero linear region and unit-RMS normalization. Separate full square
matrices `A_a` and `A_b` are initialized to identity and are unrestricted;
the per-neuron offsets `beta` are added after `A`, matching
`A C x / (s g(r/s)) + beta`. The raw branches before `C` still contribute
separate statistics that are averaged, reported, and regularized by the
soft-LayerNorm objective.
Every hidden layer also applies a fully normalized residual GLU to the same
standardized 60-return input `x0`, then adds it to the main GLU output:
`h = GLU(main) + GLU(input)`. The residual projection has shape
`2*hiddenWidth x 60`. Its value half starts at zero, its gate half uses
Kaiming initialization, and both biases start at zero. Consequently the
residual initially contributes exactly zero while remaining trainable.
Projection and the layer-shared `A` matrices are optimized by Muon; biases,
radii, and constrained `C` matrices use AdamW.
In addition, each target layer receives an additive normalized GLU residual
from every earlier hidden layer except the immediately preceding one, which
already supplies the main path. Across the eight-layer network this adds 21
dense residual projections. For target layer `l`, all incoming paths share
that target's `C`, `A_v`, and `A_g`; each path retains independent projection
weights, projection biases, normalization radii, and post-`A` biases.
Training uses 5% activation dropout intermittently with an application rate
of 50%. One
pass-level gate and one independent gate per layer both use probability
`sqrt(0.5)`, making the marginal probability of applying dropout to a
particular layer in a particular forward batch exactly 50%. Validation and
inference never apply dropout. The fused input-to-value/gate projection
weights contain 948,224 parameters and their biases contain 4,096. The 16
path-shared `A` matrices contain 1,048,576 parameters, and the eight fixed
layer-shared `C` matrices contain 524,288 non-trainable values. Across all
paths, the 74 post-`A` offsets contain 18,944 parameters and the 74 scalar
radii add 74. The eight fused global-input residual projections contain
245,760 weights and 4,096 biases. The 21 dense residual paths contain
2,752,512 weights and 10,752 projection biases. The output head contains
65,535 parameters. With `C` frozen, the model has 5,098,569 trainable
parameters overall, so
the 33,195,600 training examples
continue to outnumber parameters.

The preceding hard-LayerNorm configuration and its preserved checkpoints are
recorded in
[`experiments/return-oracle-hard-layernorm-baseline-2026-07-28.md`](experiments/return-oracle-hard-layernorm-baseline-2026-07-28.md).
The immediately preceding bias-only LayerNorm plus full-`A` configuration is
recorded with its best checkpoint and metrics in
[`experiments/return-oracle-bias-ln-full-a-baseline-2026-07-29.md`](experiments/return-oracle-bias-ln-full-a-baseline-2026-07-29.md).
The learnable-centering configuration with all activation regularizers enabled
is preserved in
[`experiments/return-oracle-learnable-centering-full-regularization-baseline-2026-07-29.md`](experiments/return-oracle-learnable-centering-full-regularization-baseline-2026-07-29.md).
The same current objective at the former 8,192-example training batch is
preserved in
[`experiments/return-oracle-learnable-centering-diagnostic-activations-batch8192-baseline-2026-07-29.md`](experiments/return-oracle-learnable-centering-diagnostic-activations-batch8192-baseline-2026-07-29.md).
The subsequent 65,536-batch, five-step Muon speed baseline is preserved in
[`experiments/return-oracle-batch65536-muon5-speed-baseline-2026-07-29.md`](experiments/return-oracle-batch65536-muon5-speed-baseline-2026-07-29.md).
The intermediate one-segment, three-step Muon run without gradient
accumulation is preserved in
[`experiments/return-oracle-batch86400-muon3-no-accumulation-baseline-2026-07-29.md`](experiments/return-oracle-batch86400-muon3-no-accumulation-baseline-2026-07-29.md).
The final fast CE-only comparison run before adding reverse KL is preserved in
[`experiments/return-oracle-fast-ce-only-baseline-2026-07-29.md`](experiments/return-oracle-fast-ce-only-baseline-2026-07-29.md).
The subsequent fixed-smoothed reverse-KL run with weight `0.05` is preserved
in
[`experiments/return-oracle-smoothed-reverse-kl-w005-baseline-2026-07-29.md`](experiments/return-oracle-smoothed-reverse-kl-w005-baseline-2026-07-29.md).
The first unbiased skew reverse-KL run with weight `1` and epsilon `1e-4` is
preserved in
[`experiments/return-oracle-skew-reverse-kl-eps1e4-w1-baseline-2026-07-29.md`](experiments/return-oracle-skew-reverse-kl-eps1e4-w1-baseline-2026-07-29.md).
The epsilon-`1e-2` skew reverse-KL run before output-entropy regularization is
preserved in
[`experiments/return-oracle-skew-eps1e2-w1-no-sharpness-baseline-2026-07-29.md`](experiments/return-oracle-skew-eps1e2-w1-no-sharpness-baseline-2026-07-29.md).
The run applying skew reverse KL and entropy sharpness continuously at weight
`1` is preserved in
[`experiments/return-oracle-continuous-skew-sharpness-w1-baseline-2026-07-29.md`](experiments/return-oracle-continuous-skew-sharpness-w1-baseline-2026-07-29.md).
The subsequent run using a shared 1% gate, inverse-probability correction,
expected output-loss weights `0.2`, and diagnostic-only soft LayerNorm is
preserved in
[`experiments/return-oracle-shared-shake-w0p2-softln0-baseline-2026-07-29.md`](experiments/return-oracle-shared-shake-w0p2-softln0-baseline-2026-07-29.md).
The following run with independent 1% gates, expected output-loss weights
`0.02`, and soft LayerNorm weight `1` is preserved in
[`experiments/return-oracle-independent-p01-w002-softln1-baseline-2026-07-29.md`](experiments/return-oracle-independent-p01-w002-softln1-baseline-2026-07-29.md).
The immediately preceding run with one centering matrix shared between each
layer's value and gate branches, including the later 10% sharpness gate, is
preserved in
[`experiments/return-oracle-shared-c-sharpness-p1-baseline-2026-07-29.md`](experiments/return-oracle-shared-c-sharpness-p1-baseline-2026-07-29.md).
The directly preceding exact-RMS run with independent centering matrices and
10% reverse-KL/10% entropy-sharpness gates is archived at
`data/ml-runs/return-oracle-ce-shrinking-v1-hard-rms-reverse-p1-sharpness-p1-baseline-20260729-235300`.
The subsequent tanh-family soft-RMS run is archived at
`data/ml-runs/return-oracle-ce-shrinking-v1-tanh-soft-rms-baseline-20260730-002736`.
The following square-root-family run that used
`C x / sqrt(1 + (r/s)^2)` with `s = 1` is archived at
`data/ml-runs/return-oracle-ce-shrinking-v1-sqrt-output-scale-baseline-20260730-002937`.
That form learned slowly because it bounded branch RMS by `s` and did not
reproduce the amplification of the preceding hard RMS operation.
The directly preceding learned-radius run with hidden widths shrinking from
512 to 272 is recorded in
[`experiments/return-oracle-learned-radius-shrinking-baseline-2026-07-30.md`](experiments/return-oracle-learned-radius-shrinking-baseline-2026-07-30.md).

## Cross-entropy plus skew reverse KL and regularizers

Let `Q` be the stored close-only oracle policy and `P` the model policy. The
optimizer minimizes categorical cross-entropy with objective weight `10`
against every probability in the stored soft target:

```text
CE(Q, P) = -sum_a Q(a) log P(a)
         = H(Q) + KL(Q || P).
```

The oracle entropy `H(Q)` is constant with respect to the model, so minimizing
cross-entropy has exactly the same gradient and optimum as minimizing
`KL(Q || P)`. Cross-entropy does not approach zero for a soft oracle; its
minimum is the oracle entropy. Reported base-action KL subtracts that entropy
and therefore approaches zero when `P = Q`. Probability MSE is diagnostic only.

The optimizer also uses a reverse-direction skew KL. Its reference
distribution mixes the raw oracle with the live prediction:

```text
M_epsilon(a) = (1 - epsilon) Q(a) + epsilon P(a)
L_reverse = KL(P || M_epsilon)
epsilon = 1e-2
```

Because `M_epsilon >= epsilon P` elementwise, this divergence is finite and
bounded above by `log(1 / epsilon)`. Where `Q(a) = 0`, that action contributes
exactly `P(a) log(1 / epsilon)`, giving a finite gradient that suppresses
probability outside the oracle support. The prediction remains connected to
autograd in both the numerator and the mixture. For `0 < epsilon < 1`, the
term is zero exactly when `P = Q`, so it shares the raw cross-entropy's ideal
target instead of introducing a fixed smoothing bias.

The output also receives a one-sided per-example entropy-sharpness penalty:

```text
Delta_H = H(P) - H(Q)
L_entropy-sharpness = ReLU(Delta_H)^2
```

The squared hinge is computed separately for every compact example before
the source-multiplicity-weighted mean. It contributes only while a prediction
is more entropic than its matching oracle and becomes exactly zero at or
below the oracle entropy. There is no below-oracle margin and no stochastic
margin. Consequently, `P = Q` remains an exact common optimum, while natural
entropy undershoot is allowed rather than rewarded.

The skew reverse KL and entropy-sharpness terms now use independent stochastic
gates drawn for every optimizer update. Each pair of draws is reused across
both accumulated microbatches:

```text
g_reverse ~ Bernoulli(0.1)
g_entropy ~ Bernoulli(0.1), independently
L_output =
    (g_reverse / 0.1) * 0.1 * L_reverse
    + (g_entropy / 0.1) * 0.1 * L_entropy-sharpness
```

Inverse-probability correction makes both active coefficients `1.0`, while
preserving a long-run expected coefficient of `0.1` for each term. No
per-example gating is used because millions of independently gated examples
would average into a nearly constant weak regularizer. Validation does not
sample either gate; it deterministically uses the expected coefficients
`0.1` and `0.1`, keeping scheduling and checkpoint selection stable.

For every hidden layer and compact example, the raw value branch `a` and raw
gate-logit branch `b` are separately measured using statistics across that
example's neurons:

```text
mu = mean(h)
v = mean((h - mu)^2)
L_soft-LN = mu^2 + lambda_v * (v - 1)^2
```

The implementation averages the value and gate-logit penalties, averages that
result across all 8 hidden layers, then takes the example mean using the
compact rows' source-example multiplicities.
Weight matrices, but not biases, also contribute a smooth magnitude-bound
penalty. For every scalar weight `w`:

```text
z(w) = sqrt(w^2 + epsilon) - desiredMagnitude
smoothExcess(w) = log(1 + exp(beta * z(w))) / beta
L_weight-bound = mean_w(smoothExcess(w)^2)
```

Each hidden layer's post-GLU activations `h`, measured before dropout, also
contribute the screenshot's two distribution-like diagnostic components:

```text
D_sum = ((sum_i h_i - 1) / d)^2
D_negative = mean_i ReLU(-h_i)^2
```

The two raw components are calculated per example, averaged across all 8
hidden layers, and then averaged with the compact rows' source-example
multiplicities. Measuring before dropout keeps the target deterministic.
Dividing by layer width preserves exactly the same zero-loss conditions as the
screenshot formula while preventing the sum term from scaling quadratically
with each layer's width. Both components remain explicit
training-plan metrics, but their current objective weights are zero.

The fixed centering matrices remain exact orthogonal projectors. Their
diagnostic components are still computed for each layer:

```text
L_idempotence = mean((C^2 - C)^2)
L_symmetry = mean((C^T - C)^2)
```

Both terms remain numerically zero at `C = I - 11^T/d`; `C` is excluded from
both optimizers, so these terms are diagnostic and cannot update the model.

In the current ablation, the training objective is:

```text
L_train = CE(Q, P)
    + (g_reverse / 0.1)
        * 0.1 * KL(P || (1 - epsilon) Q + epsilon P)
    + (g_entropy / 0.1)
        * 0.1 * mean(ReLU(H(P) - H(Q))^2)
    + 1.0 * L_soft-LN
    + 0.01 * L_weight-bound
    + 1.0 * L_idempotence
    + 1.0 * L_symmetry
g_reverse ~ Bernoulli(0.1) once per optimizer update
g_entropy ~ Bernoulli(0.1) independently once per optimizer update
lambda_v = 1
desiredMagnitude = 1
beta = 10
epsilon = 1e-8
skew reverse-KL epsilon = 1e-2
```

The deterministic validation objective replaces both corrected gates with
their expectation, making each output-loss coefficient exactly `0.1`.

The soft-LayerNorm loss weight is `1` in this run, in addition to the
square-root learned-radius per-example branch normalization. Its
penalty therefore contributes continuously to training and to the
deterministic validation objective. The distribution unit-sum and
distribution non-negativity loss weights remain `0`; their values are still
computed and reported as diagnostics.

The weight penalty is exponentially small within the
desired magnitude, becomes
approximately quadratic outside it, and does not push every valid weight
toward zero. Optimizer weight decay is disabled for both Muon and AdamW, so
this explicit term is the only magnitude regularizer. The diagnostic
distribution metrics continue to show whether hidden transformations have
nonnegative activation mass summing to one, alongside the soft-LayerNorm and
soft weight-bound metrics.

Training stops only after validation loss has failed to improve for more than
1,024 complete epochs. The final evaluation restores the best-validation
checkpoint and scores the reserved 1,000,000-example test tail.

Training uses two optimizers. Muon updates the 8 main fused projections, the
16 path-shared post-normalization `A` matrices, and the 8 global-input fused
residual projections. The 21 dense residual projection matrices are also
routed to Muon, bringing its total to 4,995,072 parameters. It uses
momentum `0.95`, Nesterov momentum,
three Newton-Schulz steps, and `match_rms_adamw` rectangular-matrix
learning-rate adjustment. AdamW updates the output head, all projection
biases, all 74 post-`A` offsets, and all 74 scalar normalization radii,
totaling 103,497 parameters. The eight fixed `C`
matrices are excluded from both optimizers.

Both optimizers start at `1e-4`. If the combined validation objective fails to
improve materially for 32 epochs, paired `ReduceLROnPlateau` schedulers halve
both learning rates down to a floor of `1e-6`. Both optimizer and scheduler
states are stored in the resume checkpoint.
CUDA training uses BF16 autocast so rare high-variance soft-LayerNorm batches
retain FP32-like exponent range instead of overflowing FP16.

Training now loads one microbatch per chronological source segment. The
configured capacity is `86,400`, large enough for every selected segment's
examples. Gradients from two consecutive segment microbatches are combined
using their expanded example counts before every optimizer update. This
preserves the multiplicity-weighted combined objective while reducing
expensive Muon updates to `302` per epoch across `604` microbatches. Exact repeated minute
rows remain compacted with multiplicity weights. Validation also uses
`86,400`, reducing its batches from `714` to `355`.

Muon now uses three Newton-Schulz iterations instead of five. Learning rates
remain `1e-4`, making this a speed-focused optimizer-compute ablation.
During validation and final testing, the weight-bound and centering-matrix
regularizers are batch invariant because model parameters do not change.
They are therefore computed once per evaluation pass and added back to the
aggregated metrics and loss exactly, rather than recomputed in every
validation batch.

The constant-width run starts from a fresh initialization. After the one-time
compiled-graph warm-up, epoch 1 took `37.218` seconds: `33.703` seconds
training and `3.453` seconds validation. Validation loss improved from
`4.999210` at epoch 0 to `4.759717` at epoch 1; validation CE was `4.413778`
and forward base-action KL was `1.072886`. Gradients remained finite with no
overflow.

## Operation and observability

Run:

```text
npm run mlp:experiment:return-oracle-ce
```

The plan is `ml/training-plans/return-oracle-ce-shrinking-v1.json`. Because it
uses the standard `status.json` and append-only `training.log` event contract,
the run is automatically available in the MLP page's training-run selector.
The page shows live total loss, cross-entropy, the three soft-LayerNorm
components, all three distribution-layer metrics, the soft weight-bound
penalty and configured desired magnitude, validation loss, raw-policy KL,
probability MSE, throughput, GPU memory, the best validation value, and
stale-epoch count.

The best and resume checkpoints remain under the experiment run directory.
They are not copied into `data/models`, exported to ONNX, registered with the
runtime, or offered in the inspector.

If a checkpoint is explicitly exported later,
`fold_input_normalization_for_export` creates a logits-only copy and folds the
frozen training statistics into the first fused projection:

```text
P_export[:, i] = P[:, i] / std_train[i]
c_export = c - P_export @ mean_train
```

The exported graph therefore consumes raw simple returns and contains neither
the training mean/std buffers nor a separate input-normalization operation.

The training loop uses the same execution pipeline as the established MLP
runner: four parallel component-loading workers with two-deep prefetch,
pinned-host batches copied on a dedicated CUDA stream, BF16 autocast without
loss scaling, dynamically compiled full-objective execution, GPU-resident
metric accumulation, next-epoch CPU prefetch during validation, and
asynchronous checkpoint serialization. Exact per-second repeats are compacted
within every original batch; multiplicity-weighted CE preserves the original
loss, metrics, and optimizer-step boundaries while avoiding redundant network
evaluations.
