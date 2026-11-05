# Agreement and combinations of the two one-second predictors

The frozen joint knot density and latest input/output-calibrated specialists have complementary mean-prediction errors. Combining their predictions improves mean MSE on this historical test sample. Requiring agreement adds almost nothing beyond averaging, and the more elaborate gate does not beat simple averaging on test.

This experiment concerns the same one-second forecasts as the preceding MSE audit. It does not establish results for the 15-minute, 30-minute or two-hour models, or new trading returns.

## How often do they agree?

On 36,134 matched test predictions:

| Comparison | Agreement |
| --- | ---: |
| Sign of expected log return | 64.770% |
| Active UP/DOWN classification, threshold 0.5 | 92.110% |
| Most likely DOWN/ZERO/UP class | 85.847% |

Expected-return sign differs from the active sign classifier because conditional UP and DOWN sizes can differ. Means have correlation 0.76656. The median absolute difference between predicted means is 0.01538 bps; the 90th and 99th percentiles are 0.07014 and 0.18340 bps. Opposite signs can therefore describe two very small forecasts, rather than strong opposing predictions.

Their full distributions differ much more substantially. Exact bin masses are integrated on a common partition derived from the joint model's training knots and uniform Gaussian-CDF coordinates, with the zero atom and infinite tails retained. The median binned total-variation distance is **0.42461**, with 10th/90th percentiles **0.34252/0.50293**. Every tested pair exceeds 0.25. A distance of 0.42461 means at least that much probability mass must be reassigned to make these binned laws match.

These are lower bounds on the continuous-law total-variation distance, not exact continuous distances. Refining the nested uniform component from 512 to 1,024 intervals increases TV by at most 0.001975 and changes no classification at the prespecified 0.10/0.25 thresholds. The separate atom is not blurred into a central continuous bin. Median binned Jensen–Shannon divergence is 0.24232 nats.

These comparisons refer to the declared continuous log-return laws with an exact zero atom. They are not distances after projecting every forecast to the exchange's price lattice.

## Is there a clear winner for each observation?

The specialist mean has lower squared error on **55.070%** of test examples; the knot mean wins **44.930%**. Despite winning fewer individual examples, the knot model has lower total MSE because error magnitudes matter. There are no exactly tied squared errors in this sample.

On opposite-mean-sign predictions, the specialists win 60.888% of individual squared-error comparisons, yet the knot model still has marginally smaller aggregate MSE within that group. Counts of wins do not determine the MSE-optimal mixture weight.

For **48.185%** of all examples, the difference in absolute errors is less than 0.01 bps. Only **4.104%** have an absolute-error difference of at least 0.1 bps. The numerical winner is therefore often only slightly better. These fixed tolerances describe the comparison; they were not tuned to select a combination.

Mean accuracy and full-law accuracy can disagree. The knot model assigns higher probability/density to the realized outcome on 53.005% of all ticks and 92.519% of nonzero ticks. It wins the zero-atom likelihood comparison on only 23.307% of unchanged ticks. The per-example MSE winner and likelihood winner coincide on only 53.219% of the sample.

One realized price allows proper-score comparisons, but cannot reveal which entire conditional distribution was true. A retrospective best-of-two chooser would score 9.192545% MSE skill, but it uses the future label and is not an executable policy. None of the tested rules receives that winner label at prediction time.

## Combining forecasts without future information

A distribution mixture is

```
P_mix = w * P_knot + (1-w) * P_specialists,     0 <= w <= 1.
```

It preserves both complete predictive laws. Its mean is the corresponding weighted mean; its zero atom and density are mixed with the same weight. Each weight below depends only on already-issued forecasts and parameters fitted on validation data.

| Rule | Held-out validation-half skill | Test skill |
| --- | ---: | ---: |
| Joint knot density | 3.351051% | 3.698769% |
| Current specialists | 3.672276% | 3.445867% |
| 50/50 distribution mixture | 4.010984% | **4.196924%** |
| Constant mixture fitted for MSE | 4.020319% | 4.176922% |
| Constant mixture fitted for likelihood | 4.023277% | 4.087843% |
| Average only when mean signs agree; zero signal otherwise | 3.993340% | 4.195178% |
| Average only when mean signs disagree; zero signal otherwise | 0.017644% | 0.001746% |
| Smaller absolute mean when signs agree; zero otherwise | 3.233979% | 3.746553% |
| Agreement-dependent distribution mixture, validation-selected | **4.073192%** | 4.129103% |
| Forecast-feature distribution gate | 4.022708% | 4.125374% |

Agreement-abstention rows are point-signal diagnostics. A zero signal does not mean the return distribution is an atom at zero and is not automatically equivalent to closing or holding every possible account. These rows have no claimed full-distribution likelihood score or trading backtest.

The test-leading 50/50 mixture was a prespecified candidate, not a test-fitted weight. However, test results were not used to replace the validation-selected primary rule. The held-out validation selection chose the agreement-dependent mixture. After refitting its parameters on all validation examples, its frozen weights are:

```
same mean sign:     38.2738% knot + 61.7262% specialists
opposite signs:    67.2889% knot + 32.7111% specialists.
```

This rule improves test skill by 0.430334 points over the knot model and 0.683235 points over the specialists. A paired 3,000-replicate bootstrap over 41 hourly blocks gives exploratory 95% percentile intervals of [0.12194, 0.70728] and [0.50828, 0.90425] skill points respectively. This is about a 0.447% reduction in actual test MSE versus the knot model, not a 0.43% increase in returns or a large percentage reduction in error. Hour-block resampling does not remove repeated-research selection effects or establish future market stability.

## What the more elaborate gate does

The gate outputs a logistic mixture weight from 18 forecast-only features: both zero/sign probabilities, log conditional UP/DOWN means, signed means, absolute mean gap, mean-sign agreement, binned TV/JS, and active expected absolute magnitudes. It uses no realized future feature.

Ridge choices 0.0001, 0.001, 0.01, 0.1 and 1 were compared. Fitting used the first chronological validation half; ridge and rule-family selection used the second half. All fitted mixture parameters were then refit on the complete validation set and saved before test forecasts were processed. The validation halves are separated by four seconds, exceeding the one-second target maturity. The test period had already been inspected in previous research; it is not a new untouched holdout.

The selected full-validation constant MSE mixture gives the knot model 44.7815% weight. The likelihood-fitted mixture gives it 76.5604%. Neither is a test-fitted weight.

## MSE improvements do not mean every distribution score improves

Full-law test negative log likelihood, in the same log-return coordinate and atom/density convention, is:

| Forecast | Test NLL, lower is better |
| --- | ---: |
| Joint knot density | **-5.903018** |
| Current specialists | -4.005978 |
| 50/50 mixture | -5.699083 |
| Validation-selected agreement mixture | -5.676012 |

Thus the mixtures improve mean MSE while the knot law retains better test density likelihood. A model should be assessed against the intended downstream objective, including the full distribution when account payoffs depend on it. No trading policy was changed or new profitability claimed by this forecast experiment.

These are continuous-density scores for nonzero outcomes, not exact probability masses of exchange price-tick cells. Numerical point-density evaluation is sensitive to very narrow knot intervals. The independent native-evaluator check found that its forced float32 CDF-coordinate rounding changes individual log likelihood by up to 0.07903 nats on 2,560 checked rows, with mean difference -0.00006253 nats. Matching that rounding reproduces the native evaluator within 0.000001593 nats; the audit consistently evaluates the declared law at float64 coordinates. That numerical effect is much smaller than the observed mean likelihood gap, and does not affect the saved mean-prediction comparisons.

## Artifacts and checks

`ml/audit_forecast_agreement.py` verifies forecast, checkpoint, population and dataset hashes; confirms normalized nonnegative bin masses and nested-partition monotonicity; reproduces the specialist likelihood; freezes selection before test; and saves per-example comparisons and all candidate predictions. Five tests independently check triangular CDF integrals, zero-atom TV and symmetry, partition refinement, signed exponential special cases, and mixture likelihoods.

Artifacts are in `data/backtests/oracle-forecast-agreement`: `protocol.json`, `selection.json`, `audit.json`, per-split `*-predictions.npz`, and `native-likelihood-check.json`. The primary selected rule is recorded as `twoRegimeMixture`. All candidates remain diagnostics; no live forecast source was promoted.

```powershell
$env:PYTHONPATH='ml'
.venv-ml/Scripts/python.exe ml/audit_forecast_agreement.py
.venv-ml/Scripts/python.exe -m unittest ml/test_forecast_agreement.py
```
