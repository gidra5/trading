# Neural direction and size specialists

The retrained specialists improve expected absolute-size prediction by
**14.39%, 18.98% and 15.99%** at 15 minutes, 30 minutes and two hours. The native
one-second direction model achieves **84.38% active-sign accuracy**, but the
trained longer-horizon classifiers remain near chance. Combining the specialists
does not produce profitable trading on this test: the paid 15- and 30-minute
accounts stay in cash, while the two-hour account loses **1.042219%**. Its sign
checkpoint is initialized epoch zero because training made its validation score
worse; that result is explicitly not evidence of learned two-hour direction.

Both historically strong structured neural backbones were retrained from
scratch with separate task objectives and combined for the existing gamma-zero
Oracle policy. Checkpoint selection uses validation scores; every prespecified
trading horizon and control is reported. Training and evaluation took 424.78
seconds on the local RTX 3070. All 1,092 policy decision certificates and 108
execution fixtures pass the exact Lean checker.

## Measured results

| Horizon | Selected sign epoch | Selected size epoch | Test active-sign accuracy | Test sign log loss / train-constant log loss | Test average-size MSE improvement | Test signed-mean MSE improvement |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| 1 second | 12 | 9 | 84.379% | 0.386841 / 0.694152 | +7.587% | +3.294% |
| 15 minutes | 4 | 7 | 50.530% | 0.697138 / 0.693303 | +14.387% | -2.097% |
| 30 minutes | 3 | 7 | 49.515% | 0.700279 / 0.693898 | +18.984% | -10.391% |
| 2 hours | **0, initialized** | 6 | 52.991% | 0.693062 / 0.703377 | +15.992% | +0.673% |

The sign run stopped at epoch 32 and the size run at epoch 29 under their
prespecified patience rule. Later training losses continued to fall while
validation losses worsened. No checkpoint was selected using test scores.
The corresponding validation sign accuracies are 81.212%, 49.561%, 51.634%
and 49.236%. At all three longer horizons, even the selected sign checkpoint
has worse validation log loss than the training-constant probability. Those
neural forecasts remain in the primary experiment as declared, while fixed
probabilities appear only in labelled controls.

Forecast metrics use 36,134 common test origins at a four-second stride, each
complete through the two-hour endpoint. Sign accuracy excludes exact zero
returns. Average-size MSE compares the combined expected absolute return against
its train-constant baseline; signed-mean MSE compares against zero return.
These are different targets.

The [matched joint-component audit](oracle-crra-joint-marginal-audit-2026-09-11.md)
separately compares these specialists with surviving joint return densities on
identical origins. The historical roughly 7% active-log-return scores are not
directly comparable with this experiment's all-tick simple-return skill.

Size regression is conditioned separately on each
sign: test upward/downward magnitude MSE improvements are +24.28%/+18.53% at
one second, -2.10%/+17.14% at 15 minutes, -12.61%/+17.77% at 30 minutes, and
-14.19%/+14.67% at two hours. Better aggregate size scores do not mean both
conditional heads improve separately.

The one-second model's magnitude-weighted direction accuracy is **58.564%**,
compared with 84.379% unweighted accuracy. Its forecast mean reaches at most
0.4703 bps on this metric grid, far below the 10.0075-bps full-close entry
diagnostic. High accuracy on numerous tiny moves therefore does not establish
a tradable edge at that horizon.

| Horizon | Primary mixture: paid return | Paid market trades | Zero-fee return | Zero-fee market trades | Largest mean on replay origins | Entry-threshold crossings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 15 minutes | 0% | 0 | -1.159890% | 85 | 5.2497 bps | 0 / 168 |
| 30 minutes | 0% | 0 | -0.528599% | 40 | 6.2615 bps | 0 / 84 |
| 2 hours | -1.042219% | 2 | -2.594118% | 11 | 15.6359 bps | 1 / 21 |

Both paid controls—learned sign with fixed conditional sizes, and fixed sign
with learned sizes—stay in cash at every trading horizon. The two-hour primary
account makes one round trip, finishes at 9,895.7781 from 10,000, and has maximum
drawdown 1.630093%. Every account ends with zero residual inventory. Paid
buy-and-hold loses 2.076869% and cash earns zero on the common tape. Removing
fees changes selected actions as well as costs; it is a separate policy replay.

The 30-minute model exceeds the threshold at some four-second metric origins,
but none of its prespecified nonoverlapping replay origins. Likewise, the
two-hour model has 923 crossings on the heavily overlapping metric grid and
only one among 21 replay origins. Those counts must not be interpreted as
hundreds of independent trade opportunities. No downside-size clipping occurs
in evaluation or any exported forecast.

This experiment supports learning magnitude and direction separately as
measurable prediction tasks. It does not establish a profitable long-horizon
directional model or a benefit from the untrained two-hour sign head. Better
average-size MSE alone is insufficient to make the combined policy profitable
on the observed test tape.

## Historical models and the specialization

The direction backbone is the base17 causal self-attention model from the
[next-second sign audit](structured-next-second-sign-audit-2026-09-04.md).
Its feature-MSE-selected checkpoint previously achieved 76.8059% validation
and 81.5012% test accuracy on **nonzero next-one-second returns**. Those are
historical scores under a different objective, not measurements of this run or
of longer-horizon predictability. Its original backbone has 4,130,613 parameters.

The size backbone is the
[five-branch direct59 attention model](structured-layer4-five-branches-2026-09-10.md),
the strongest pure feature-MSE backbone in the audited family. Its historical
feature-MSE checkpoint achieved 6.3869% validation and 6.9971% test improvement
in signed next-second return MSE relative to zero. Its original backbone has
9,067,378 parameters. A separate hybrid feature/density-loss run scored higher
on signed-return MSE; this experiment chooses the pure-MSE architecture for
dedicated conditional-magnitude regression. No historical result establishes
that this backbone is already best at absolute-size prediction.

All old checkpoints had been removed. Both specialists are therefore trained
from scratch. Their widths, two observed input steps, two latent output steps,
normalization, shared numbered blocks, and causal attention match their source
recipes. The size backbone retains all five independent layer-four branches.
A new linear readout uses the flattened two-step latent outputs. Those outputs
are now task features, not supervised predictions of the original primitive
feature graph. This is architecture reuse with new losses, not a reproduction
of the old primitive-feature training objective.

The sign model predicts active-up logits for 1, 900, 1800 and 7200 seconds and
minimizes binary cross-entropy on nonzero returns. The separate size model
predicts positive upward and downward conditional mean sizes for each horizon,
minimizing sign-masked squared errors normalized by their training conditional
means. Upward/downward tasks and horizons are balanced. The specialists share
no parameters with each other; horizons within each specialist share a backbone.

Training uses the source Muon/AdamW recipe at learning rate `0.0001`, effective
batch size 10,240 accumulated over microbatches of 2,560, float32 computations,
and an EMA with a four-epoch half-life. There is no dropout or weight decay;
gradient norm is capped at 100. Each specialist has a maximum of 96 epochs,
minimum 24, and stops after 20 epochs without improvement on any horizon.
Each horizon independently selects its best EMA checkpoint by validation log
loss or balanced normalized conditional-size MSE. Initialized epoch zero is
also scored and explicitly identified if selected. The task readouts add 140
parameters to the sign backbone and 952 to the size backbone.

Targets are **simple** endpoint returns `R = P[t+H]/P[t] - 1`, using actual
close prices. Input tensors contain only two successive observed production59
states, at `t-1` and `t`. No future feature states enter the model. Every training
and validation origin has a complete target through the maximum horizon within
its chronological split. Normalization, sign priors, zero mass, and size scales
are fitted from training data only.
There are 247,901 training and 34,560 validation origins at a four-second stride.
Long-horizon labels overlap substantially, so these counts are not independent
sample counts. The fixed zero and sign priors use Jeffreys half-count smoothing.

## Combination and controls

For `p = Pr(R>0 | R≠0, inputs)`, training zero probability `p0`, and conditional
size predictions `m+` and `m-`, the executable law is:

| Return support | Probability |
| --- | --- |
| `-m-` | `(1-p0)(1-p)` |
| `0` | `p0` |
| `+m+` | `(1-p0)p` |

The sign specialist gates the two size components. Expected signed return is
`(1-p0)[p*m+ - (1-p)*m-]`; expected absolute size uses a plus between the terms.
This permits different upward and downward sizes and does not assume sign-size
independence. Downward size is bounded below a 100% loss, with every clipped
prediction counted. The zero probability is a fixed train-fitted component,
not a third trained specialist.

The primary forecast retains both learned specialists even when a constant
scores better. Two fixed controls isolate their contributions: retain learned
sign with train-constant conditional sizes, or retain learned conditional sizes
with the train-constant active-up probability. The primary combination is
replayed both with the declared fees and with zero fees; controls use paid fees.
There is no test-driven sharpening, horizon selection, or replacement of a
poor learned head hidden inside the primary result.

## Policy and verification scope

The 15-minute, 30-minute and two-hour forecasts use a common 42-hour test tape:
2026-08-10 00:00:01 UTC to 2026-08-11 18:00:01 UTC. This is the previously
inspected benchmark, not a newly untouched holdout. New checkpoint choices use
validation only, but repeated experimentation on this tape still makes its
results exploratory. The one-second task is a forecast-accuracy control.

The existing endpoint policy chooses hold or target exposure `0`, `0.5`, `1`
with gamma zero, received-amount taker fees of 5 bps per side, quote trade bounds
`[5,10000]`, price grid `0.01`, nonnegative cash and inventory, unit funding,
and no pending orders. It processes every actual intermediate one-second mark.
Decisions execute net adjustments; final closing obeys the cap and retains any
residual. The data are BTCUSDT spot closes with synthetic Oracle parameters,
not a USD-M futures funding/order-book simulation.

Each decision maximizes the supplied finite scenario problem over that menu.
The same Lean checker used in earlier runs verifies exact rational payoff
comparisons; execution conformance is checked separately. The mixture preserves
its supplied first moments. Replacing each entire conditional size law by one
mean can change nonlinear capped-close payoffs. Neither finite-menu optimality
nor the sign/size identities prove that these fitted three-point forecasts
recover the unknown full distribution, or optimize a leveraged policy with
intrahorizon liquidation, limit orders, or a longer continuation horizon.

The theory build and axiom audit pass with 1,116 audited declarations. Thirteen
specialist tests cover architecture fidelity, all branch/attention gradients,
checkpoint reconstruction, polarity-specific scaling and masking, uneven
microbatch accumulation, exact endpoint labels, causal boundaries, and mixture
weights. All 33 relevant existing Oracle policy, execution and endpoint tests
also pass. Artifact checks verify checkpoint/selection/forecast hashes, the
same complete price tape across all twelve replays, positive scenario prices,
normalized probabilities, and certificate coverage for every decision. Lean
then accepts all 1,092 decision records plus 108 execution fixtures.

## Reproduction

The frozen source recipes live in `ml/neural_sign_size_forecast.py`; training
and selection are in `ml/train_neural_sign_size_forecast.py`. The combined law
and fixed controls are exported by `ml/export_neural_sign_size_forecast.py`.
The complete replay schedule is recorded before training in
`data/backtests/oracle-neural-specialists/replay-plan.json`.
Frozen checkpoints, selection metadata, validation/test metrics and the training
curve are in `data/training/runs/stochastic-crra-neural-specialists-v1`.
The trainer refuses to overwrite an existing experiment plan.

```powershell
node scripts/run-ml-python.mjs ml/train_neural_sign_size_forecast.py
node scripts/run-ml-python.mjs ml/export_neural_sign_size_forecast.py
node scripts/run-oracle-neural-specialists.mjs
```

Each exported file is consumed by `scripts/backtest-oracle-endpoint-hold.ts`
with its full decision count also passed to `--certificate-count`.
`scripts/report-oracle-neural-specialists.mjs` checks artifact hashes, the common
price tape, scenario support and decision coverage, then joins the certificates.
`scripts/verify-oracle-crra.mjs` checks them in Lean.
