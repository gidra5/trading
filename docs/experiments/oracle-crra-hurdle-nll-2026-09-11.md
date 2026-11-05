# Separate zero expert and full likelihood fine-tuning

The completed experiment shows a small forecast improvement from a separate
zero expert followed by full-likelihood fine-tuning. One-second test mean-MSE
skill is **3.294% with the original binary specialists, 3.322% with a separate
zero expert, and 3.337% after likelihood fine-tuning**. This recovers the
performance lost by the preceding shared three-class/MSE run, which scored
2.165%, but it remains below the surviving joint-density model's matched
3.699%. The small gain over the original specialists is an observed result on
the reused test period, not evidence of a statistically established advantage.

Likelihood fine-tuning improves test likelihood and mean MSE at one second,
15 minutes and 30 minutes. The two-hour model retains its pretrained starting
point. These improvements do not produce a profitable fee-paying backtest.

The experiment replaces the original binary specialists' constant zero
probability with an independently trained zero expert, then fine-tunes the
zero, conditional-sign and size experts by full return negative log likelihood
(NLL). Existing sign and size checkpoints are independently trained
initializers. Comparisons use the same causal inputs, chronological splits and
complete horizon origins as the preceding experiments.

## Matched forecast results

All-tick simple-return MSE skill is `100*(1-MSE/mean(actualReturn^2))` on the
same 36,134 test origins. Negative skill means a zero-return prediction is
better under MSE.

| Horizon | Original binary specialists | Prior shared three-class + MSE | Separate zero, before NLL tuning | After NLL tuning |
| --- | ---: | ---: | ---: | ---: |
| 1 second | +3.294% | +2.165% | +3.322% | +3.337% |
| 15 minutes | -2.097% | -2.247% | -2.097% | -2.021% |
| 30 minutes | -10.391% | -7.854% | -10.391% | -7.612% |
| 2 hours | +0.673% | -4.306% | +0.673% | +0.673% |

Before NLL tuning, every tensor in the original sign and size experts is
unchanged. Replacing the predicted zero probability with the original fixed
prior reconstructs the archived binary model's mean skill within `2.72e-10`
absolute. This isolates the new zero expert's contribution within floating
precision. The larger difference from the prior shared three-class model also
reflects restoring the original conditional-sign expert; it cannot all be
credited to learning zero probability.

The separate zero expert lowers one-second zero Brier error from 0.251796
for the fixed prior to 0.240000. After NLL tuning it is 0.240269.
Thus even this successful mean-MSE fine-tuning step slightly worsens zero
calibration on test. Conditional sign accuracy improves from 84.379% to
84.450%; three-class cross-entropy improves from 0.838979 to 0.838293.
Predicted mean zero probability changes from 52.486% to 52.280%, against
57.090% observed zeros. Overall absolute-size MSE skill rises from 7.900% to
8.000%.

| Horizon | Test full NLL before | Test full NLL after | Test absolute-size MSE skill before → after |
| --- | ---: | ---: | ---: |
| 1 second | -3.998239 | -3.999551 | +7.900% → +8.000% |
| 15 minutes | -5.357993 | -5.361137 | +14.387% → +14.816% |
| 30 minutes | -5.018492 | -5.026308 | +18.986% → +19.635% |
| 2 hours | -4.238063 | -4.238063 | +15.992% → +15.992% |

Lower NLL is better. The 30-minute validation mean skill actually worsens from
-1.491% to -3.297% while validation NLL improves, although both test metrics
improve. This illustrates why likelihood selection does not guarantee a mean-
MSE ranking, even within one experiment.

Zero pretraining completed 69 epochs and selected epochs `[15,8,49,7]` for
`[1,900,1800,7200]` seconds. Fine-tuning completed `[15,15,15,12]` epochs and
selected `[3,3,3,0]`. All choices froze before test loading. The two-hour
selected model exactly equals its pretrained model.

## Backtest results

Every row uses the same 42-hour actual price tape. Returns below are for the
restricted gamma-zero Oracle endpoint policy described later.

| Horizon | Fees, before → after | Zero fees, before → after |
| --- | ---: | ---: |
| 15 minutes | 0.000% → 0.000% | -1.160% → -1.694% |
| 30 minutes | 0.000% → 0.000% | -0.529% → -0.180% |
| 2 hours | -1.042% → -1.042% | -2.594% → -2.594% |

The fee-paying 15- and 30-minute runs stay in cash. Their maximum predicted
mean moves after NLL tuning are 5.146 and 6.097 bps on the replay schedule,
below the approximately 10.0075-bps round-trip threshold. The two-hour model
exceeds that threshold once (maximum 15.636 bps), executes two market trades
including its close, and loses 1.042%. Better average forecast scores alone
have not established a useful trading advantage.

## Why the previous fine-tuning regression is possible

The preceding one-second mean-MSE run reduced selected validation MSE by only
0.05854% relative, while test MSE increased by 0.09395%. An independent audit
now evaluates both saved old checkpoints on exactly the same 247,901 training
origins, removing the ambiguity of comparing online training logs with EMA
validation scores.

| Old MSE run, fixed train checkpoints | Before | After |
| --- | ---: | ---: |
| Total MSE, bps² | 0.13782260248 | 0.13708217423 |
| Moving-tick contribution to total MSE, bps² | 0.13561794041 | 0.13470153005 |
| Zero-tick contribution to total MSE, bps² | 0.00220466206 | 0.00238064417 |
| All-tick MSE skill | 6.72276% | 7.22388% |

The old model **did learn a compensation on training**: moving-tick errors
contributed 0.00091641036 bps² less, while zero-tick errors contributed
0.00017598211 bps² more. Total training MSE fell. On test, the moving-tick gain
was only 0.00013613395 bps², less than the extra 0.00021601056 bps² from zero
ticks, so total test MSE rose. Minimizing total training error does not require
every subgroup to improve, and this particular trade-off generalized poorly.

Mean MSE trains the product of probabilities and sizes. For example, an
overestimated active probability can be offset by underestimated conditional
sizes. That can fit the mean while degrading the component probability
forecasts. It does not imply the model can identify and fix its future test
errors from the training sample. The earlier audit measured a small improvement
on moving test ticks outweighed by increased error on zero-return test ticks.

In the new NLL run, fixed training-checkpoint mean skills rise from 6.813% to
7.035% at one second, 3.388% to 4.396% at 15 minutes, and 4.060% to 5.742% at
30 minutes. The corresponding smaller or negative test skills still limit
what can be concluded about generalization.

## Independent experts and the loss

Let `z=P(R=0|I)` and `u=P(R>0|R!=0,I)`. The zero expert predicts `z`; a separate
binary sign expert predicts `u`. The three class masses are

\[
(p_{\rm down},p_0,p_{\rm up})=((1-z)(1-u),z,(1-z)u).
\]

The zero event means an unchanged **endpoint price**. The price can move and
return during the horizon. Exact equality is preserved; there is no arbitrary
small-return deadband.

The size expert emits upward/downward conditional magnitude means `m+`, `m-`.
To give size outputs a valid likelihood, the declared positive magnitude law
is Gamma with shape `alpha+` and rate `alpha+/m+`; the downward magnitude law
is Beta with parameters `alpha-` and `alpha-*(1-m-)/m-`. These laws have exactly
the requested means, with downward support strictly below a 100% simple loss.
The combined mean is

\[
\widehat R=(1-z)[u m_+-(1-u)m_-].
\]

For `A=1[R!=0]`, `S=1[R>0]`, and `M=|R|`, the per-example objective is

\[
\ell=\operatorname{BCE}(z,1-A)
+A\operatorname{BCE}(u,S)
-1_{R>0}\log f_+(M)-1_{R<0}\log f_-(M).
\]

Terms are summed and averaged over all examples. The first two terms equal
ordinary three-class cross-entropy exactly. Cross-entropy on the three class
probabilities alone has no gradient for the independent size outputs; it is
insufficient to train all three experts. A distribution consisting only of
point masses at the two predicted sizes would assign zero likelihood to
almost every observed nonzero magnitude, which is why the full likelihood
uses continuous conditional densities.

The experts own independent parameters. Consequently, this joint objective
factorizes: it trains each probability or size law directly and does not let
one expert compensate for another through the combined mean. The separate
architecture changes optimization and capacity, not the information in the
three class probabilities. Gradient clipping, learning-rate scheduling and
selection of a whole checkpoint are still shared optimization decisions.

NLL and MSE have different finite-model objectives. In particular, a fitted
Beta conditional law under misspecification need not have the true conditional
arithmetic mean. Better NLL therefore need not mean better mean-MSE or better
trading. Both are reported separately. On tick-quantized data, the nonzero
point-density likelihood is a continuous approximation, not exact probability
mass assigned to every price tick.

Mixed atom/density NLL can be negative. Its absolute level also changes when
the zero/nonzero frequency changes between splits. Generalization is assessed
by the before/after change within each split, with the four likelihood terms
reported separately, rather than by treating a raw train/test NLL gap as a
direct measure of overfitting.

## Protocol and numerical implementation

Horizons are 1, 900, 1800 and 7200 seconds. Inputs are the two observed
production59 states at `t-1,t`. Data and source checkpoint hashes, train-only
normalization and size scales are checked against
`stochastic-crra-neural-specialists-v1`. The common stride-four origin schedule
has 247,901 training, 34,560 validation and 36,134 test examples; overlapping
horizons are not independent observations. The historical test tape has
already been inspected in previous experiments.

Zero pretraining uses a fresh base17 sign backbone with zero readout weights
and training-prior logit bias. It optimizes unweighted zero/nonzero BCE,
selecting each horizon by validation BCE. It permits 96 epochs, minimum 24,
patience 20, learning rate `1e-4`, and EMA half-life four epochs. Conditional
sign and size initialize from their already selected independent checkpoints.
Each horizon then receives its own triplet of full backbones and copied
readout rows, with no parameter sharing between experts or horizon triplets.

Fine-tuning unfreezes all three backbones and the per-horizon conditional-size
shape parameters. It uses only full NLL, learning rate `1e-5`, EMA half-life
two epochs, maximum 64 epochs, minimum 12, patience 12, and selects by
validation NLL including the separately pretrained initialization. All final
horizon choices freeze before loading test data. Mean-MSE, class CE, zero
Brier and each likelihood term are diagnostics rather than selection rules.

Both stages retain the existing Muon/AdamW recipe, effective batch 10,240,
microbatch 2,560, gradient limit 100 and seed 1337. Networks are float32;
physical scales, returned means, targets and likelihood calculations are
float64. Physical mean magnitudes are bounded below by `1e-12`; downward mean
is at most `1-1e-12`. Shapes lie between `0.001` and `100`. Any boundary hits
are reported. Initial shapes use the inverse mean squared relative size
residual on training examples, clipped to `[0.01,50]`; the Beta counterpart
is a moment-based initialization heuristic, not maximum-likelihood fitting.
Large-Beta log normalizers use a checked log-Gamma ratio
expansion to avoid cancellation; independent value and derivative checks use
80-digit arithmetic. Inactive likelihood branches receive safe inputs before
masking, avoiding invalid logarithms and NaN gradients on zero ticks.

The first attempt to compile the complete hurdle graph failed on Windows
before any fine-tuning optimizer update because a generated CPU kernel needed
an unavailable C++ compiler. Completed zero selections were preserved. The
run resumed with each expert's forward callable compiled separately and the
canonical distribution packing and likelihood evaluated eagerly. The immutable
statistical plan is retained; `runtime-adjustment.json` records the execution
change, resume guards and provenance checks.

## Policy comparison and formal-check scope

Before and after fine-tuning, the same 42-hour tape is replayed at 15 minutes,
30 minutes and two hours, with paid and zero fees. The backtest preserves the
preceding restricted Oracle long/cash account: gamma zero, unit funding, no
orders, exposure menu `0,0.5,1` plus hold, 5-bps received-amount taker fees,
quote limits `[5,10000]`, price grid `0.01`, second-by-second observed marks,
and valid capped final closing with residual inventory marked.

For comparability, the endpoint policy consumes the three-point approximation
at the conditional means; this preserves the fitted one-step mean but discards
conditional spread. The underlying magnitude likelihood is evaluated in the
forecast scores. Certificates check each supplied finite scenario/menu
decision and execution fixtures. They do not prove the learned density is the
true law, that conditional spread is irrelevant to arbitrary account rules,
or optimality over unrestricted full-path policies.

## Reproduction

```powershell
node scripts/run-ml-python.mjs ml/train_hurdle_sign_size_forecast.py
node scripts/run-ml-python.mjs ml/export_hurdle_sign_size_forecast.py
node scripts/run-oracle-neural-specialists.mjs --output data/backtests/oracle-hurdle-nll
```

The trainer refuses to overwrite its existing plan. Checkpoints, frozen
selection and train/validation/test metrics are under
`data/training/runs/stochastic-crra-hurdle-nll-v1`; forecast exports, replay
artifacts and Lean checks are under `data/backtests/oracle-hurdle-nll`.

The independent audit is reproduced with
`node scripts/run-ml-python.mjs ml/audit_hurdle_finetune.py` after training.
It verifies common origin/target hashes, old baseline provenance, unchanged
pretrained sign/size tensors, train/export loader parity and all new forecast
scores. No mean-floor or physical-downside-cap hits occurred. Model, trainer
and exporter-parity coverage comprises 11 passing tests. All 20 referenced
checkpoint hashes passed verification; eight combined states were finite.

All 12 replay artifacts passed consistency and certificate-coverage checks.
Lean verified **1,092 decision certificates and 108 execution conformance
fixtures**, all passing. These checks establish the stated finite-menu
decision and execution properties for the supplied forecasts, not their
accuracy or calibration against the market.

Frozen selection SHA-256:
`8ac7d25215c733b284d26cf97c09b741ffc81cc786eae20ef195547286777317`.
