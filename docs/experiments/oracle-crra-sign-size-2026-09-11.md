# Predicting direction and size separately

Separate size prediction is useful in this experiment: test mean-squared error
for expected absolute movement improves **8–13%** over fixed training averages.
The direction classifiers do not improve validation log loss over their training
constants. All four horizons therefore retain constant sign probabilities and
learn separate, changing upward and downward sizes.

The resulting thirty-minute model crosses the fee threshold on two validation
intervals and one test interval. Its test account makes one round trip and loses
**0.665393% after fees**. The other horizons stay in cash. Predicting size better
has not established a profitable directional edge on this test period.

## What is predicted and what the policy uses

Let `R = P[t+H]/P[t] - 1` be the **simple** endpoint return. The model estimates
the probability of an upward move conditional on a nonzero move, a fixed
train-fitted zero probability, and two sizes:

\[
m_+=\mathbb E[|R|\mid R>0,I_t],\qquad
m_-=\mathbb E[|R|\mid R<0,I_t].
\]

If `p+` and `p-` are the unconditional upward and downward probabilities,

\[
\underbrace{\mathbb E[|R|\mid I_t]}_{\text{average size}}
=p_+m_++p_-m_-,\qquad
\underbrace{\mathbb E[R\mid I_t]}_{\text{expected return}}
=p_+m_+-p_-m_-.
\]

The model does not assume that sign and size are independent. Even a constant
sign probability can give a changing expected return when upward and downward
sizes change differently. That is what generates this experiment's signals.
When the upward sign is known with certainty, average size equals expected
return. With uncertain sign, the loss side must also enter the calculation.

For the existing Oracle cash-starting, surviving, fully closable long with unit
funding and 5-bps fees per side, expected return must exceed **10.007505 bps**.
Average size does not replace that threshold's sign-weighted mean. The actual
replay uses the full finite scenario payoff comparison, including capped closes
and marked residuals, instead of substituting the full-close formula everywhere.

An audit of the old selected joint densities already finds predicted mean
absolute movement of 9.576, 13.378, 20.634 and 28.640 bps at 15, 30, 60 and 120
minutes on the nonoverlapping test origins. The two-hour model thus already
predicts approximately **0.286% average size**, while its signed mean is only
0.0206 bps because positive and negative contributions cancel. Separating the
training tasks can improve estimation; algebraically refactoring an unchanged
joint density does not change its decisions.

## Training and fixed controls

The four prespecified horizons are 900, 1800, 3600 and 7200 seconds. Inputs are
the same 178 causal features: 120 observed one-second return lags and 58 latest
observed features. Targets use actual endpoint prices and never cross a
chronological split boundary. Training and validation selection use every eighth
eligible origin. Long-horizon labels overlap heavily; their count does not
represent independent observations.

Each horizon fits three separate histogram gradient-boosting models:

- Sign classifier: log loss on active upward/downward outcomes.
- Upward size regressor: positive simple-return magnitude, scaled by the
  training upward mean, with a Poisson log link.
- Downward size regressor: the corresponding downward magnitude, also positive
  and scaled by its own training mean. Simple losses are bounded below 100%.

The Poisson loss is a fitting objective for the conditional mean, not a claim
that returns are counts or follow a Poisson distribution. Model defaults are
150 maximum iterations, 15 leaves, minimum leaf size 128, learning rate 0.05,
L2 regularization 1, four CPU threads and seed 1337. Early stopping uses the
explicit chronological validation data, with patience 12. The best iteration is
restored by deterministic refitting on the training data only. Each final head
then competes with its train-only constant, using sign log loss or conditional
size MSE; constants win ties.

| Horizon | Sign head | Upward size best iteration | Downward size best iteration |
| --- | --- | ---: | ---: |
| 15 minutes | Training constant | 77 | 16 |
| 30 minutes | Training constant | 53 | 6 |
| 1 hour | Training constant | 37 | 4 |
| 2 hours | Training constant | 18 | 1 |

All twelve head choices were saved before any test data was loaded. All four
horizons and fixed controls are reported; test PnL does not choose a winner.
Fit and metric evaluation took 27.56 seconds on CPU. This is a new nonlinear
model family and training objective, so differences from previous neural or
ridge results cannot be attributed to factorization alone.

The primary control keeps the same selected sign predictor and replaces both
size predictors with their separate training constants. Additional saved
diagnostics use one common constant size, or a fair sign probability with the
same predicted size on both sides. The fair-sign symmetric control has exactly
zero expected return; it does not treat an uncertain predicted sign as known.

## Forecast accuracy and economic diagnostics

| Horizon | Test average-size MSE improvement vs training constant | Test signed-return MSE improvement vs zero | Maximum validation mean | Validation crossings | Maximum test mean | Test crossings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 15 minutes | +11.10% | -3.13% | 9.1362 bps | 0 / 161 | 7.5458 bps | 0 / 168 |
| 30 minutes | +13.32% | -3.98% | 14.3981 bps | 2 / 80 | 12.8914 bps | 1 / 84 |
| 1 hour | +13.17% | -4.97% | 10.5466 bps | 1 / 40 | 5.2564 bps | 0 / 42 |
| 2 hours | +8.14% | -2.98% | 23.5921 bps | 1 / 20 | 4.2623 bps | 0 / 21 |

The MSE columns use the fixed eight-second evaluation stride over complete test
targets. Mean maxima and crossings use nonoverlapping decision origins; test
crossings use the common 42-hour replay period. They are different sampling
grids. Better size MSE is a measured predictive improvement, not a proof of lower
irreducible variance or a general guarantee about sign/size models.

At thirty minutes the two validation crossings average +33.962 bps in the
isolated cash-start/full-close diagnostic. The one test crossing realizes
-66.539 bps. Those counts are too small to establish calibration. The direction
classifiers select iteration zero at every horizon: the varying predictions
come entirely from the conditional size regressors. Signed-return MSE is worse
than predicting zero at every horizon despite the size improvements.

## Common-period Oracle account comparison

All twelve replays use gamma zero and the same 151,200 observed one-second marks,
from **2026-08-10 00:00:01 UTC through 2026-08-11 18:00:01 UTC**. There are
168, 84, 42 or 21 decisions according to horizon.

| Horizon | Learned sizes, paid return | Paid market trades | Fixed sizes, paid return | Learned sizes, zero-fee return | Zero-fee market trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| 15 minutes | 0% | 0 | 0% | -0.636734% | 58 |
| 30 minutes | -0.665393% | 2 | 0% | -1.191342% | 28 |
| 1 hour | 0% | 0 | 0% | -1.169416% | 22 |
| 2 hours | 0% | 0 | 0% | -1.332819% | 10 |

Paid buy-and-hold returns -2.076869% on this declining tape; cash returns zero.
The thirty-minute paid account finishes at 9,933.4607 from 10,000, with maximum
drawdown 0.683371%. Every replay finishes without residual inventory. Removing
fees changes the selected actions and generates more trades; it does not isolate
the fee subtraction from a fixed action sequence.

The account restrictions are unchanged: nonnegative cash and asset balances,
entry and maintenance bounds `[0,1]`, unit funding, no pending orders, targets
`0,0.5,1` plus hold, quote trade bounds `[5,10000]`, price grid 0.01 and valid
capped terminal close retaining any residual. Actual intermediate marks are
applied every second, and decisions execute net adjustments. The tape is
BTCUSDT spot with synthetic Oracle account parameters, not an actual USD-M
funding or order-book replay.

## Distribution approximation and proof scope

The implemented endpoint law has three support points, `-m-`, zero and `+m+`,
with their predicted sign probabilities. It preserves its supplied sign and
conditional first moments. No downside clipping occurred on the evaluated
predictions. Export checks confirm positive price support, normalized weights,
mean conservation, target alignment and every intermediate observed mark.

Replacing each conditional magnitude distribution with one mean omits its
spread. It can alter expected nonlinear capped-close payoffs, liquidation,
limit fills or nonlinear CRRA utility. The general symmetry and regrouping
results do not make that approximation exact.

The new [Lean sign/size module](../theory/lean/TradingTheory/SignSizeForecast.lean)
proves fourteen public results. Retaining the **full** sign-conditioned scenario
laws preserves every action's payoff and the existing CRRA policy problem,
including observation timing. The mean/average-size identities and the
restricted Oracle fee-threshold specialization are also proved. Cases with
zero-probability signs need no fictitious scenario support. The repository theory
check passes with 1,116 audited results.

All **945 finite scenario/menu decision certificates**, including every decision
in every replay, pass the exact Lean checker. The **108 execution fixtures** also
pass. These verify the supplied finite payoff comparisons and execution fixtures;
they do not certify calibration or global optimality of the rolling policy under
the true market distribution. Four regression tests cover dependent sign/sizes,
equal-size counterexamples, the symmetric fair-sign control and causal split
boundaries. A separate synthetic learnability audit checks the training path.
With independently generated training and validation sets, sign log loss falls
from 0.69328 to 0.28870, and upward/downward size MSE falls by 87.3%/87.5%.
The unchanged production fitting function selects positive iteration counts in
all three cases. The real-data constant-sign outcome is not an always-select-zero
bug in the early-stopping or scoring code.

The report assembler verifies each declared head choice, frozen global and
per-horizon selections, actual model/selection/forecast hashes, identical full
price tapes across horizons and controls, scenario support and all certificate
counts. No shared execution rule or live provider was changed.

## Artifacts and reproduction

Models, frozen choices and metrics are in
`data/training/runs/stochastic-crra-sign-size-v1`. The eight exports, twelve
replays, `comparison.json`, `decisions.jsonl`, `lean-verification.json` and old
joint-density decomposition audit are in `data/backtests/oracle-sign-size`.

Train into a fresh directory to preserve the recorded run:

```powershell
node scripts/run-ml-python.mjs ml/train_sign_size_forecast.py --output data/training/runs/stochastic-crra-sign-size-reproduction
```

Export the existing frozen models and reproduce an account run:

```powershell
node scripts/run-ml-python.mjs ml/export_sign_size_forecast.py
npm run backtest:oracle-endpoint -- --forecasts data/backtests/oracle-sign-size/factorized-h1800.jsonl --count 84 --certificate-count 84 --fee-scale 1 --output data/backtests/oracle-sign-size/factorized-h1800-fees1
```

Repeat using the horizons/counts and variants in `replay-plan.json`. Then:

```powershell
node scripts/report-oracle-sign-size.mjs
node scripts/verify-oracle-crra.mjs --certificates data/backtests/oracle-sign-size/decisions.jsonl --output data/backtests/oracle-sign-size/lean-verification.json
node scripts/run-ml-python.mjs ml/audit_existing_sign_size.py
node scripts/run-ml-python.mjs ml/audit_sign_size_learnability.py
node scripts/run-ml-python.mjs -m unittest discover -s ml -p test_sign_size_forecast.py
npm run theory:check
```
