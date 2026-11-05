# Joint-law components versus separately trained specialists

On identical one-second test examples, the available joint density scores
**3.6988%** signed-mean MSE skill and the specialists **3.2940%**. That is a
0.4048-percentage-point skill gap, or **0.4204% more actual MSE** for the
specialists. The specialists nevertheless score better on sign log loss,
sign accuracy, both conditional size MSEs, and unconditional absolute-size MSE.
On validation the signed-mean ranking reverses: specialists **3.2707%**, joint
density **3.1026%**. This does not establish a stable joint-model advantage.

The earlier roughly 7% MSE skill and the specialist mixture's 3.29% are not
scores from a controlled comparison. The older headline used active next-second
log returns from a feature-state regression model. The new figure evaluates
simple returns on a four-second-stride sample of all ticks, including exact
zero returns, with every origin complete through a two-hour endpoint. The
specialists' separate size improvements were against training constants, not
against components extracted from that older model.

## Measured one-second components

The models below use the same 36,134 test origins, simple-return targets and
training-constant baselines. Sign and conditional-size scores use the matching
realized nonzero/upward/downward subsets. The joint column extracts probabilities
and conditional first moments from its original density without refitting.

| Metric | Specialists | Joint density's components |
| --- | ---: | ---: |
| Active-sign accuracy | **84.379%** | 81.954% |
| Active-sign log loss, lower is better | **0.386841** | 0.425710 |
| Magnitude-weighted sign accuracy | **58.564%** | 54.872% |
| Upward conditional-size MSE skill | **24.284%** | 20.423% |
| Downward conditional-size MSE skill | **18.528%** | 16.012% |
| Unconditional absolute-size MSE skill | **7.587%** | 6.909% |
| All-tick signed-mean MSE skill | 3.294% | **3.699%** |
| Active-only conditional-mean MSE skill | 5.363% | **6.296%** |

The final row evaluates the active-conditional forecast only where the realized
return is nonzero. It does not assert that activity is known when trading. It
also does not reproduce the deleted historical model's original 65,534-example
log-return evaluation. The specialist's active-only 5.363% versus all-tick
3.294% illustrates a substantial change in metric scope, without explaining
away every model difference.

The one-second joint model is the previously selected Gaussian-coordinate
`lambda1` density, trained with negative log likelihood **plus normalized
signed-price-mean MSE**. The separate specialist losses do not include their
composed mean's MSE. The better component scores and worse test mean score are
empirical evidence that those objectives are not interchangeable; they do not
prove that every joint architecture trains better.

## Component swaps

Except for the two explicitly named zero-probability swaps, these diagnostics
use the specialist's same fixed zero probability. They do not choose a new
ensemble or tune anything on test data.

| One-second combination | Test signed-mean MSE skill |
| --- | ---: |
| Both specialist components, specialist zero probability | 3.2940% |
| Both joint components, joint zero probability | 3.6988% |
| Both joint components, specialist zero probability | 3.5282% |
| Joint sign + specialist sizes, specialist zero probability | 2.2254% |
| Specialist sign + joint sizes, specialist zero probability | 3.2747% |
| Both specialist components, joint zero probability | 3.2143% |

Neither replacing the specialist sign nor replacing its sizes improves the
specialist combination. The joint's varying zero probability helps its own
combination by 0.1706 skill points but worsens the specialists' combination.
Thus a missing learned zero gate is not, by itself, an established explanation
or fix for the joint-versus-specialist test gap.

## Other horizons

| Horizon | Test sign log loss: specialists / joint | Test upward-size MSE skill: specialists / joint | Test downward-size MSE skill: specialists / joint | Test signed-mean MSE skill: specialists / joint |
| --- | --- | --- | --- | --- |
| 15 minutes | 0.697138 / 0.696050 | -2.100% / +4.569% | +17.139% / +17.851% | -2.097% / -0.713% |
| 30 minutes | 0.700279 / 0.693858 | -12.607% / -1.974% | +17.770% / +22.099% | -10.391% / +0.079% |
| 2 hours | 0.693062 / 0.703301 | -14.187% / -0.155% | +14.673% / -0.001% | +0.673% / -0.033% |

At 15 and 30 minutes, the joint distribution's sign log loss and both
conditional-size MSEs are better on test. Specialist sign accuracy at 15
minutes is higher (50.530% versus 48.525%), but its probability log loss is
worse. Aggregate expected absolute-size skill still favors specialists:
14.387% versus 13.324%, and 18.984% versus 16.635%, respectively.

The two-hour row is not a comparison of two successfully trained directional
models. The specialist sign head is initialized epoch zero. Although the joint
selection manifest calls its chosen model `conditionalGlu`, its checkpoint is
also epoch zero and its output weights are zero: it is an initialized
train-fitted constant density. Both saved loss-weight candidates retain only
epoch-zero checkpoints; later trained states were not saved. The audit records
that inventory explicitly and does not invent a trained two-hour candidate.

## What can be extracted from a joint distribution

Let `R` be the next endpoint simple return and `I` the observed inputs. A joint
return law determines:

- `p0 = Pr(R=0 | I)`;
- `p = Pr(R>0 | R≠0, I)`;
- `m+ = E[R | R>0, I]`;
- `m- = E[-R | R<0, I]`.

These components reconstruct its signed mean exactly:

\[
\mu(I)=(1-p_0)\{p m_+-(1-p)m_-\}.
\]

Extracting and recombining those quantities cannot inherently lower the joint
law's signed-mean MSE. Its expected absolute size is the same formula with a
plus between the two terms. The two conditional sizes are essential: keeping
only marginal sign probability and marginal average absolute size would discard
their dependence. This experiment does not make that independence assumption.

A point forecast of the mean does not identify these probabilities and
conditional sizes. For example, many distributions with very different sign
probabilities and sizes have mean zero. Calling a shared multi-feature point
regressor a joint model does not make it a joint probability distribution.

The historical 6.9971% point-model checkpoint and the related approximately
7.6% scalar/joint log-density checkpoints were deleted. Their saved evaluations
contain aggregate metrics without recoverable per-example forecasts. The exact
old models therefore cannot be marginalized or rescored from the surviving
artifacts. The historical density models also used polynomial tails in log
return; their reported expected-log-return scores are not interchangeable with
finite expected-price-return scores. The present audit uses the surviving
Gaussian-coordinate density models with finite simple-return moments.

## Why the specialized scores do not order overall MSE

Sign accuracy and sign log loss score direction over active examples without
weighting each error by its economic magnitude. The size model minimizes
separate sign-masked, train-normalized magnitude MSEs. The final prediction is a
product and difference of these estimates, so those training objectives do not
minimize the same finite-model loss as signed-return MSE.

For fixed zero probability, let component errors be `deltaP`, `deltaUp` and
`deltaDown`. The error in the active conditional mean is

\[
\widehat\mu_{\rm active}-\mu_{\rm active}
=(m_++m_-)\delta p+p\delta m_+-(1-p)\delta m_-
+\delta p(\delta m_++\delta m_-).
\]

Direction-probability errors are multiplied by movement size. Errors can also
cancel in one model and reinforce in another. Better individual component
metrics therefore do not imply better MSE after combination.

As a mathematical example, suppose the true return is equally likely to be
`+1%` or `-1%`. A forecast with `(p,m+,m-)=(0.6,1.2%,1.8%)` gets the mean exactly
right at zero. A second forecast `(0.55,1.1%,1.1%)` improves probability log loss,
both conditional size MSEs, and expected absolute-size MSE, yet predicts a
signed mean of `0.11%`, increasing signed-return MSE by 1.21%.
This is an illustration of component errors, not a trading calibration result.

## Active-only and all-tick evaluation

The specialist mixture includes a train-fitted constant zero probability; it
does not learn when the next tick will be unchanged. Its one-second test sample
contains approximately 57.09% unchanged ticks. The older active-return metric
does not score those contexts. Simply interpreting the two skill percentages
as an architecture or factorization comparison is therefore invalid.

In the illustrative special case where the probability `a` of an active tick
is independent of the inputs, an active conditional forecast `h(I)` becomes
the all-tick forecast `a*h(I)`. With the same active-conditional context
distribution and zero-return baseline, its all-tick MSE skill equals `a` times
its active-only skill. A roughly one-half activity rate can therefore halve
the percentage without worsening the active conditional predictor. This is
an explanatory special case, not an assumption imposed on the market data.

The audit scores the actual active-conditional and deployed unconditional
predictions separately and swaps the zero-probability component separately
from sign and size.

## Matched audit protocol

The existing Gaussian-coordinate return-density checkpoints are frozen; no
model is trained, selected again, or calibrated during this audit. The chosen
one-second, 15-minute, 30-minute and two-hour checkpoints are those selected
earlier using validation. They are different runs from the deleted historical
roughly 7% feature-regression checkpoint.

Both model families are scored on the exact specialist validation and test
origins, targets and common train-constant baselines. Joint distributions retain
their own predicted zero mass. Fixed-zero controls and component swaps isolate
zero probability, active sign, and sign-conditional size contributions. These
swaps are diagnostics; test scores do not select a deployed ensemble.

Integration splits the density at zero and compares numerical quadrature
resolutions. The reconstructed signed mean is checked against the original
joint expectation. This is a numerical forecast audit, not a new Lean proof or
an account backtest. The previously inspected test tape remains exploratory.

The largest observed reconstruction error is below `2e-18` in simple-return
units. Doubling quadrature from 192 to 384 points changes signed means by less
than `6e-8` bps and conditional sizes by less than `1.3e-7` bps. No impossible-side
conditional means or downside clipping occurred. Three tests check the
integration split, conditional-moment reconstruction and separate zero-probability
effects. Inference uses the original specialist batch size 2,560, matching its
previous scores within `1e-12`.

Reproduce without training or changing selection:

```powershell
node scripts/run-ml-python.mjs ml/audit_joint_vs_neural_specialists.py
node scripts/run-ml-python.mjs -m unittest discover -s ml -p test_audit_joint_vs_neural_specialists.py
```

Results and checkpoint provenance are saved in
`data/backtests/oracle-neural-specialists/joint-marginal-audit.json`; the audit
plan is saved alongside it as `joint-marginal-audit.plan.json`.
