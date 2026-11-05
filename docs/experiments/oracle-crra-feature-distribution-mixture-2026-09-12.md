# Separate sign and magnitude distribution mixtures

Separating feature mixture weights gives a small test improvement in expected
log-return MSE: **4.6142% to 4.6561% skill**. This is **0.0419 skill percentage
points**, or approximately **0.044% lower MSE**. The paired hour-block bootstrap
interval for the gain is **[-0.1083, +0.1696] skill points**, so these data do not
establish a reliable improvement over the existing complete-law mixture.

The selected variant has slightly worse nonzero-return MSE and worse full-law
likelihood. It improves the all-tick score mainly by reducing the prediction
error on zero-return ticks. No bot setting, base checkpoint or retained mixture
was replaced.

## Common population and frozen models

The experiment uses the same next-second BTC **log-return** population as the
previous mixture experiment: 34,560 validation examples and 36,134 test examples,
including 15,505 nonzero test outcomes. All component origins, timestamps and
targets are checked for equality. These are one-second forecasts issued at
the archived four-second cadence, not four-second returns.

The four frozen models are the original joint knot law, the current calibrated
sign/size specialists, fresh direct59 with its knot probability head, and fresh
base17 with its knot probability head. Existing online-calibrated streams are
reused unchanged; new mixture weights are static after validation fitting.
There is no new calibration or base-model training in this experiment.

Pools are the original pair, the retained specialist/direct59/base17 trio, and
all four. Fit each candidate on the first chronological validation half;
select its pool and the best mixture family on the second half. Refit chosen
pools on full validation, save and hash the selection, then load and score
test. Target maturity is checked at both boundaries. The historical test has
been inspected in earlier experiments, and the original checkpoints used
validation, so this remains an exploratory comparison.

## What is mixed

For model `i`, extract its complete law as zero probability `z_i`, conditional
up probability `p_i`, and magnitude densities `f_i^+` and `f_i^-` conditional
on positive/negative return. Their conditional means are `u_i` and `d_i`.

Three layouts are tested:

1. **Sign + magnitude:** one simplex mixes the three-class probabilities
   `(down, zero, up)`. A second simplex mixes the magnitude densities, using
   the same magnitude weights for the positive and negative conditional laws.
2. **Zero + sign + magnitude:** independent weights for zero probability,
   up probability conditional on a nonzero return, and conditional size laws.
3. **Zero + sign + separate sizes:** independent weights for zero, conditional
   sign, positive conditional magnitude and negative conditional magnitude.

For the third layout, for example,

\[
z=\sum_i w_i^0z_i,\qquad p=\sum_i w_i^s p_i,
\]
\[
f^+(m)=\sum_i w_i^+f_i^+(m),\qquad
f^-(m)=\sum_i w_i^-f_i^-(m).
\]

The atom is `P(Y=0)=z`; positive density is `(1-z)*p*f^+(y)` and negative
density is `(1-z)*(1-p)*f^-(-y)`. All weights are nonnegative and each group
sums to one. This is normalized without an additional partition function.
The conditional magnitude laws remain intact; neither knot masses nor Gamma
parameters are replaced with representative outcomes. Magnitude remains
conditional on direction, so this does not impose sign/magnitude independence.

The resulting expected log return is

\[
\widehat{Y}=(1-z)\left[p\sum_iw_i^+u_i-(1-p)\sum_iw_i^-d_i\right].
\]

Only the mean calculation needs the component means. Full-law scoring uses
the original conditional densities. The observed sign selects a density only
when scoring an outcome; forecast probabilities, sizes and means are computed
without that outcome.

Two training objectives are tested. **NLL** fits each feature group separately
with its appropriate likelihood, using convex simplex optimization. **MSE**
jointly fits the feature weights through the reconstructed mean; this objective
is nonconvex, so deterministic multistart SLSQP uses NLL, uniform and singleton
initializations. It has no global optimization certificate. Neither procedure
changes the underlying networks.

## Test results

MSE skill uses zero predicted log return as baseline. The nonzero column
scores the same deployed expectation on nonzero outcomes; it does not remove
the forecast zero probability. Direction accuracy uses conditional sign
probabilities on nonzero outcomes. Lower NLL is better.

| Forecast / mixture objective | All-tick MSE skill | Nonzero MSE skill | Nonzero sign accuracy | Full-law NLL |
|---|---:|---:|---:|---:|
| Original joint knot | 3.6988% | 5.4013% | 81.9542% | -5.903018 |
| Current calibrated specialists | 3.4459% | 5.7563% | 84.4695% | -4.005978 |
| Direct59 refit | 4.5298% | 6.3815% | 85.8110% | **-6.029284** |
| Base17 refit | 4.0943% | 5.2278% | 85.7336% | -6.007046 |
| Previous complete-law MSE mixture | 4.6142% | 6.3688% | **86.0110%** | -5.993936 |
| Sign + magnitude, MSE fit | 4.6238% | 6.3809% | 85.9852% | -5.918758 |
| Zero + sign + magnitude, MSE fit | 4.6536% | **6.3990%** | 85.9787% | -5.916795 |
| Zero + sign + separate sizes, MSE fit | **4.6561%** | 6.3359% | 85.9787% | -5.921939 |
| Zero + sign + magnitude, NLL fit | 4.6215% | 6.1201% | 85.9852% | -5.980631 |

Validation selects **zero/sign/separate sizes** among the MSE families, before
test inspection: its second-half skill is 4.5930%, versus 4.4744% for the
complete-law MSE pool. The selected four-model pool puts zero weight on the
old joint model. The three-model and four-model fits are effectively tied
on this half; the approximately 0.0000003-point difference is numerical and
does not establish a useful contribution from the old joint model.

The best feature family by validation NLL is **zero/sign/shared magnitude**.
Its test skill is 4.6215%. Its test NLL is only about 0.000384 nats better than
the newly fitted complete-law NLL pool (-5.980247), and is worse than direct59
alone. On tuning NLL the complete-law pool beats the separate-feature families;
there is no validation evidence to replace it with feature mixing for likelihood.

## Learned feature weights

The validation-selected MSE variant assigns the following percentages after
full-validation refitting; the original joint model receives zero throughout.

| Feature | Current specialists | Direct59 | Base17 |
|---|---:|---:|---:|
| Zero probability | 0.00% | 35.86% | 64.14% |
| Up probability given nonzero | 4.24% | 67.31% | 28.46% |
| Upward magnitude law | 45.00% | 55.00% | 0.00% |
| Downward magnitude law | 19.09% | 45.60% | 35.31% |

For the separately NLL-trained winner, the weights are:

| Feature | Current specialists | Direct59 | Base17 |
|---|---:|---:|---:|
| Zero probability | 55.71% | 27.68% | 16.61% |
| Up probability given nonzero | 16.06% | 39.35% | 44.59% |
| Both conditional magnitude laws | 21.12% | 72.12% | 6.76% |

Different weights reflect different objectives. The weights that improve the
expected return's MSE need not be the best probability calibration for each
feature separately.

## Where the small MSE gain comes from

For the selected MSE variant versus the previous complete-law mixture:

- Nonzero deployed skill decreases by **0.0329 points**.
- The zero-outcome contribution to total squared error decreases by
  **0.0748 skill points**; zero-only prediction RMSE drops from 0.051705 to
  0.050591 bps. These two changes sum to the net **0.0419-point** improvement.
- Conditional magnitude MSE, evaluated using the realized direction, drops
  from **0.139427 to 0.138711 bps squared**, about **0.513%**.
- Deployable all-tick absolute-magnitude MSE drops from **0.076338 to 0.076044
  bps squared**, about **0.385%**. This forecast integrates both unknown sign
  and zero occurrence; the conditional-size metric above does not.
- Sign accuracy is slightly lower, 86.0110% to 85.9787%. Full density NLL also
  worsens, despite the small mean-MSE improvement.

The 41-block paired bootstrap gain interval overlaps zero both versus the old
mixture and versus direct59 alone (the latter interval is [-0.0905, +0.3341]
skill points). These intervals do not account for the historical model search.
The experiment supplies no new trading-profit or policy-optimality result.

## Verification and artifacts

Five focused tests pass: independent factor selection, numerical normalization
and first-moment integration including the zero atom, singleton recovery of
the original laws, finite-difference gradients through every feature, and
factor-NLL fitting versus singleton baselines. The experiment additionally
checks every source model against its archived MSE, factorized singleton
likelihood/mean reconstruction, row alignment, probabilities, source hashes,
and 1536-versus-768-point conditional-moment quadrature. The reconstructed old
complete-law mixture reproduces 4.61421787% test skill.

- `ml/feature_distribution_mixture.py`: normalized factor pools and fitting.
- `ml/evaluate_feature_distribution_mixture.py`: fit/select/freeze/test runner.
- `ml/test_feature_distribution_mixture.py`: focused numerical tests.
- `data/backtests/oracle-feature-mixture/plan.json`: declared experiment.
- `data/backtests/oracle-feature-mixture/selection.json`: frozen feature weights.
- `data/backtests/oracle-feature-mixture/evaluation.json`: all metrics, tuning
  trials, paired intervals and source hashes.
- `data/backtests/oracle-feature-mixture/{validation,test}-predictions.npz`:
  component summaries and resulting forecast probabilities/means.
- `data/backtests/oracle-feature-mixture-run.log`: completed run log.

Reproduce in a new output directory with:

```powershell
$env:PYTHONPATH='ml'
.venv-ml/Scripts/python.exe -m unittest ml.test_feature_distribution_mixture
.venv-ml/Scripts/python.exe ml/evaluate_feature_distribution_mixture.py --output data/backtests/oracle-feature-mixture-rerun
```
