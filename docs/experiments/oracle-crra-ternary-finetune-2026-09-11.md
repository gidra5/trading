# Three-state sign pretraining and joint mean fine-tuning

Implemented and completed the requested two-stage construction: separately
pretrain a down/zero/up sign expert and a conditional size expert, then unfreeze
both and fine-tune their combined signed-return mean. This run did not improve
overall test accuracy. At one second, joint tuning raised validation MSE skill
from 3.179% to 3.236%, but lowered test skill from 2.257% to 2.165%. The previous
binary specialists scored 3.294% on those same test examples.

At 15 minutes, 30 minutes and two hours, every fine-tuned checkpoint had worse
validation MSE than its separately pretrained initialization. All three runs
therefore retained epoch zero. With fees, both stages stayed in cash throughout
the common 42-hour replay; without fees, all three horizons lost money. These
results do not establish that the construction cannot work with another fitted
model, but they do not support replacing the previous model with this run.

## Matched forecast results

Skill is `100 * (1 - MSE / mean(actualReturn^2))`, using all ticks and actual
simple endpoint returns. A negative skill means a zero-return forecast has
lower MSE. Every model below uses the same 36,134 test origins, with complete
targets through two hours inside the split. These overlapping examples are not
36,134 independent market outcomes.

| Horizon | Previous binary specialists | Three-state, separately pretrained | After validation-selected joint fine-tuning |
| --- | ---: | ---: | ---: |
| 1 second | +3.294% | +2.257% | +2.165% |
| 15 minutes | -2.097% | -2.247% | -2.247% |
| 30 minutes | -10.391% | -7.854% | -7.854% |
| 2 hours | +0.673% | -4.306% | -4.306% |

The new three-state pretraining improves the 30-minute comparison, but its
error still exceeds the zero-return baseline. The other horizons are worse
than the preceding binary-specialist run. Joint tuning adds no test gain.
The matched surviving joint-density model's one-second skill was 3.699%; the
historical roughly 6.5-7% figures use different labels and populations, as
documented in the earlier matched audit.

Training and evaluation completed in 851.55 seconds (14.19 minutes). Ternary
pretraining ran 62 epochs. The size initializations were already separately
trained; they were reused, not trained again.

| Horizon | Selected sign epoch | Selected size epoch | Fine-tuning epochs completed | Selected fine-tuning epoch | Validation mean skill, before → after |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 second | 42 | 9 | 14 | 2 | +3.179% → +3.236% |
| 15 minutes | 7 | 7 | 12 | 0 | -1.134% → -1.134% |
| 30 minutes | 1 | 7 | 12 | 0 | -1.002% → -1.002% |
| 2 hours | 0 | 6 | 12 | 0 | +0.215% → +0.215% |

The two-hour sign selection retains its training-prior initialization. The
three longer fine-tuning runs were performed in full through their registered
stopping rule; their selected tensors exactly equal their saved baseline
tensors. All four final selections were frozen before test data was loaded.

## What changed on zero and moving ticks

The one-second test population has 57.090% exact zero returns. The new model
predicts mean zero probability 51.048% before fine-tuning and 49.949% afterward.
Fine-tuning slightly improves sign accuracy conditional on a nonzero move,
from 83.696% to 83.760%, but worsens three-class log loss from 0.863568 to
0.865378 and zero-probability Brier error from 0.242884 to 0.244298.

The composed mean's MSE decomposes exactly into contributions from moving and
zero ticks. On moving ticks, conditional MSE decreases from 0.1927891 to
0.1924718 bps². But the zero-tick contribution to overall MSE increases from
0.00229751 to 0.00251352 bps². That increase outweighs the improvement on moving
ticks: total MSE rises from 0.08502277 to 0.08510265 bps². This accounts for the
observed test regression; lower training MSE did not guarantee lower test MSE.

The separate activity-conditioned diagnostic, which supplies the realized
future nonzero gate, scores 5.363% for the previous binary specialists, 3.865%
for ternary pretraining and 3.928% after joint tuning. It is not a deployable
all-tick score. Three-class argmax accuracy is also different: 56.468% before
and 56.000% after tuning, versus 57.090% for always predicting zero. Fine-tuned
recalls are 19.034% down, 76.412% zero and 38.952% up. Probability forecasts are
scored with proper losses as well, because argmax accuracy alone loses the
probability and magnitude information used by the policy.

The independent audit reproduced every new all-tick score and checked the
archived baseline's dataset, checkpoint selection, sample counts and origin
bounds. No probability-floor or physical-downside-cap effects occurred.

## Policy results and verification

The separately pretrained and selected fine-tuned forecasts are identical at
the three replay horizons, so their realized results are identical too.

| Horizon | Maximum predicted mean on replay schedule | Fee-paying return / trades | Zero-fee return / trades |
| --- | ---: | ---: | ---: |
| 15 minutes | 4.984 bps | 0.000% / 0 | -1.224% / 106 |
| 30 minutes | 4.530 bps | 0.000% / 0 | -1.714% / 51 |
| 2 hours | 8.160 bps | 0.000% / 0 | -1.496% / 14 |

These maxima are below the approximately 10.0075-bps long round-trip threshold
for the tested 5-bps received-amount taker fee. Counts of threshold crossings
on the overlapping accuracy population use different decision times and must
not be substituted for the actual nonoverlapping replay schedule.

All 12 replay artifacts passed the frozen-checkpoint, forecast, common-price-
tape and certificate coverage checks. Lean checked **1,092 decision
certificates and 108 execution conformance fixtures**, all passing. The 15 new
Python model/training/export/audit tests and the three endpoint replay tests
also passed. The proof scope is the supplied finite scenario and action menu;
it does not establish that a fitted distribution matches the true market law.

## Model and training objectives

The sign expert uses the historical base17 attention backbone and emits three
logits per horizon in **down, zero, up** order. Softmax produces probabilities
`Pdown`, `Pzero`, `Pup`; exact equal endpoint prices are labelled zero. It is
trained by ordinary three-class cross-entropy on every example, without class
weights that would change the probability target. The new readout begins with
zero weights and half-count-smoothed training log-prior biases.

The size expert retains the five-branch direct59 attention backbone. Its
independently trained upward/downward conditional simple-return magnitude
checkpoints from `stochastic-crra-neural-specialists-v1` are reused unchanged as
pretraining inputs. Their data manifest, normalization, conditional scales and
checkpoint hashes are verified. This is reuse of already completed separate
training, not fitting size using the sign model or test outcomes.

Both backbones read only two observed production59 states, at `t-1` and `t`.
The horizons are 1, 900, 1800 and 7200 seconds. The training/validation origin
schedule remains the common four-second stride with all horizon endpoints
inside their chronological split. This matches the preceding specialist run.
There are 247,901 training and 34,560 validation origins. The new sign model
has 4,131,033 parameters. Each single-horizon fine-tuning pair has 13,198,334
parameters, with float32 networks and float64 physical scales, simple-return
targets and composed mean.

The combined prediction is

\[
\widehat R=P_{\rm up}\widehat m_{\rm up}
-P_{\rm down}\widehat m_{\rm down}.
\]

Both sizes are nonnegative; downward size is capped below a 100% simple loss
using a bound representable in the output dtype. Training and export use the
same physical sizes and the same mean function. The zero state's contribution
to this mean is exactly zero, while its probability changes the available
upward/downward mass.

Sign pretraining shares one backbone across horizons and selects the best
checkpoint for each horizon by validation three-class cross-entropy. Each
fine-tuning run then copies that horizon's selected sign backbone and readout,
plus its independently selected size backbone and readout, into a one-horizon
pair. Other horizon heads are removed; the copied pair shares no trainable
weights with another horizon. This avoids trying to combine different selected
pretraining epochs into a single shared backbone.
Sign pretraining allows 96 epochs, at least 24, and stops after 20 epochs
without improvement on any horizon; its EMA half-life is four epochs. Each
fine-tuning run allows 64 epochs, at least 12, with patience 12 and an EMA
half-life of two epochs.

Fine-tuning unfreezes **both entire experts** and minimizes only

\[
L_H=\frac{\mathbb E_{\rm train}[(\widehat R_H-R_H)^2]}
{\mathbb E_{\rm train}[R_H^2]}.
\]

All ticks, including zeros, contribute. The train-only denominator rescales
the loss; it does not change the single-horizon MSE minimizer. There is no
direction loss, magnitude loss, entropy penalty, or calibration penalty in
this fine-tuning stage. The learning rate is reduced to `1e-5`; pretraining
uses `1e-4`. Both use Muon/AdamW, effective batch 10,240, microbatches of 2,560,
and gradient norm limit 100. Fine-tuning selects the best EMA checkpoint by
validation signed-return MSE, including its pretraining initialization as
epoch zero. Test data does not select a checkpoint or horizon.

## What this objective establishes

Pure signed-mean MSE identifies the mean, not its factorization. Multiple
probability/size combinations can produce the same mean. After fine-tuning,
the output still defines a positive-price three-point law, but the individual
heads are not automatically calibrated probabilities or conditional means.
We therefore score three-class cross-entropy, zero-probability Brier error,
active-sign accuracy/log loss and conditional size MSE before and after tuning,
alongside the overall mean error requested by the user.

Exact extraction of sign-conditioned means from an existing joint law would
preserve its mean. This experiment changes fitted parameters and their training
objective; its benefit is an empirical question. The historical roughly 6.5–7%
active-log-return scores remain a different metric/population from all-tick
simple-return skill. The previous
[matched audit](oracle-crra-joint-marginal-audit-2026-09-11.md) records that
distinction and the available joint-density comparison.

## Policy comparison and proof scope

Both the separately pretrained mixture and the selected fine-tuned mixture
are replayed with paid and zero fees at 15 minutes, 30 minutes and two hours.
All twelve replays use the same 151,200 observed seconds, from
2026-08-10 00:00:01 UTC to 2026-08-11 18:00:01 UTC. The one-second task remains
a forecast-accuracy control. This historical test tape was inspected in earlier
experiments; it is not a new untouched confirmation period.

The policy uses the existing restricted long/cash endpoint account: gamma zero,
unit funding, no pending orders, nonnegative cash/inventory, exposure targets
`0`, `0.5`, `1` plus hold, 5-bps received-amount taker fees per side, quote bounds
`[5,10000]`, price grid `0.01`, and valid capped final closing with residual
marked. Every actual intermediate one-second mark is processed. The source
is BTCUSDT spot data with synthetic Oracle parameters.

The policy optimizes each supplied finite scenario/menu problem. Its exact
rational payoff comparisons are checked with the existing Lean certificate
checker. That does not prove forecast calibration, identification of the true
return law, or optimality over unrestricted leveraged, limit-order or full-path
policies. Conditional spread remains approximated by one size per direction.
Export records floating softmax normalization and the resulting numerical
difference from the training mean.

## Artifacts and reproduction

The new model contract lives in `ml/ternary_sign_size_forecast.py`; the training
driver is `ml/train_ternary_sign_size_forecast.py`. Model checkpoints, stagewise
selection, training curves and metric results live under
`data/training/runs/stochastic-crra-ternary-finetune-v1`.

```powershell
node scripts/run-ml-python.mjs ml/train_ternary_sign_size_forecast.py
node scripts/run-ml-python.mjs ml/export_ternary_sign_size_forecast.py
node scripts/run-ml-python.mjs ml/audit_ternary_finetune.py
node scripts/run-oracle-neural-specialists.mjs --output data/backtests/oracle-ternary-finetune
```

The trainer refuses to overwrite an existing experiment plan. The replay plan
is recorded before training in `data/backtests/oracle-ternary-finetune/replay-plan.json`.
Both stages' frozen checkpoint hashes and the complete common price tape are
validated by `scripts/report-oracle-ternary-finetune.mjs` before every decision
certificate is passed to Lean.

Primary result artifacts are `result.json` and `selection.json` in the training
run, and `diagnostics.json`, `comparison.json` and `lean-verification.json` in
`data/backtests/oracle-ternary-finetune`. The frozen selection SHA-256 is
`3791387539b40732d1f9cdd81d2ccb0fffd06fd15403193129a69a8caaeb8845`.
