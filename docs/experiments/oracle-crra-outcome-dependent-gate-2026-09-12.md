# Learned outcome-dependent distribution mixture

The normalized outcome-dependent gate was trained on the same frozen one-second joint knot density and input/output-calibrated specialist forecasts. It improves test mean MSE over either predictor alone, but does not beat the existing 50/50 mixture on MSE. It improves test density likelihood over 50/50, while the knot predictor alone still has better likelihood.

The validation winner among the two new gate families was the return-range-only gate. When the original predictors and constant likelihood mixture were included as controls, validation selected the original knot predictor for the primary NLL objective. No new predictor was promoted.

## Results

Same 36,134 test origins, log-return targets, and zero-forecast MSE baseline. Lower NLL is better; these are mixed zero-atom/continuous-density scores, not probability masses after price-grid projection.

| Forecast | Test MSE skill | Test nonzero deployed-mean skill | Test NLL |
| --- | ---: | ---: | ---: |
| Original joint knot density | 3.698769% | 5.401253% | **-5.903018** |
| Current specialists | 3.445867% | 5.756278% | -4.005978 |
| Existing 50/50 full-distribution mixture | **4.196924%** | 5.934212% | -5.699083 |
| Existing constant likelihood-fitted mixture | 4.087843% | 5.739615% | -5.831804 |
| New return-range gate | 4.068159% | 5.729356% | -5.821920 |
| New return-range + forecast-context gate | 4.013343% | 5.811173% | -5.779377 |

The new gate does not improve both MSE and likelihood over the existing constant likelihood-fitted mixture. This is evidence against promoting this particular architecture/protocol, not a proof that all outcome-dependent mixtures are ineffective.

The return-range gate's MSE gain over the knot model is 0.369390 skill points. An exploratory 41-hour-block paired bootstrap gives a 95% percentile interval of [0.25214, 0.47700] points. This does not establish a fresh out-of-sample result: the historical test period has already been inspected in earlier experiments.

## What was trained

For the currently observed context I and a candidate log return r:

```
q(r | I) = w(r,I)*P_knot(r | I) + [1-w(r,I)]*P_specialists(r | I)
Z(I) = total mass of q, including the exact-zero atom
P_gate(r | I) = q(r | I) / Z(I).
```

The gate is constant within each of 14 continuous return regions, with a separate fifteenth weight for the zero atom. Region boundaries, in basis points of log return, were fixed before the trial:

```
-infinity, -5, -1, -0.2, -0.05, -0.01, -0.003,
0,
0.003, 0.01, 0.05, 0.2, 1, 5, infinity.
```

Within each region, both original continuous densities are retained. These are gate regions, not representative outcome prices or a replacement histogram for the original densities. The gate is piecewise constant rather than a continuously interpolated neural function; adjacent continuous-region coefficients receive a smoothing penalty. The zero atom has its own coefficient.

The first family learns 15 outcome-region logits. The second learns 135 coefficients: a sigmoid of a region-specific linear function of an intercept and eight forecast-context features. The features are the two zero probabilities, two active UP probabilities, log active expected magnitudes from each model, and the two active-conditional signed log means in bps. Normalization statistics are fitted only on the corresponding training portion.

Both original models and their archived online calibration outputs remain frozen. The gate itself is fitted offline on validation forecasts; this trial does not jointly retrain the underlying predictors or add an online update to the new gate.

## Normalization and learning

For gate region k, let J_k and S_k be the two original probability masses, and M^J_k and M^S_k their signed first log-return moments. Then

```
Z = sum_k [w_k*J_k + (1-w_k)*S_k]
E_gate[r] = sum_k [w_k*M^J_k + (1-w_k)*M^S_k] / Z.
```

The zero atom participates in the mass sum and contributes zero first moment. Infinite tails are included in the original distributions' terminal regions. The normalizer uses exact region-CDF identities evaluated numerically; mean evaluation uses analytic incomplete-Gamma moments and 768-point Gauss–Legendre integration per intersected knot interval.

Training minimizes

```
-log q(observed return | I) + log Z(I) + regularization.
```

The analytic gradient includes both terms. Omitting log Z would reward creating extra probability mass. No auxiliary mean-MSE objective was used.

Regularization penalizes coefficient deviations from a fitted constant-likelihood-mixture initialization and differences between adjacent continuous-region coefficients. Ridge choices 0.001, 0.01 and 0.1 were tried for both families. Both families selected 0.1.

## Validation protocol

The first chronological half of the 34,560 validation examples fits each candidate; the second half selects the family and regularization using unregularized NLL. The boundary is four seconds apart, exceeding the one-second label maturity. Each family's selected candidate is then refitted on all validation examples and saved before processing test forecasts.

Held-out validation-half NLL:

| Candidate | NLL |
| --- | ---: |
| Original knot predictor | **-5.517029** |
| Constant mixture fitted on first half | -5.281835 |
| Best new return-range gate | -5.231977 |
| Best new return-range + context gate | -5.074220 |

Thus the new gates did not beat the controls at the actual selection boundary. Full-validation refit scores are training diagnostics, not replacement held-out selection scores. Model selection and weights did not change after test scoring.

## Learned weights

After full-validation refitting, the simpler gate's knot weights range from approximately 70.65% to 79.50% across nonzero return regions, with 74.26% at the zero atom. The constant likelihood mixture fitted on full validation has 76.56% knot weight. The learned rule therefore remains fairly close to a constant mixture rather than strongly assigning entirely different outcomes to different predictors.

For the simpler gate, the test normalizer ranges from 0.98062 to 1.01614, with median 0.99664. The contextual gate varies more: 0.72935 to 1.08957, median 0.97459. In both cases these normalizers are applied per prediction; the unnormalized pool is never treated as a probability law.

## Verification and artifacts

Seven tests passed: constant-weight mixture equivalence; normalization including the zero atom; the complete loss gradient against finite differences; identical-source invariance; prediction independence from future labels and serialization; Gamma region mass/mean identities; and separation of the zero atom from continuous regions.

Across validation/test outputs, normalized masses sum to one within 6.67e-16. Changing joint moment integration from order 384 to 768 changes the predicted mean by at most 2.35e-12 bps. Summing source regional moments reproduces the archived source means. Predictor, calibration, dataset and forecast hashes are checked before and after the run.

This is a forecast experiment. No live bot was changed, no new trading backtest was performed, and no Lean claim is made about the numerical fitting procedure.

Core implementation: `ml/outcome_distribution_gate.py`. Training/evaluation: `ml/train_outcome_distribution_gate.py`. Artifacts: `data/backtests/oracle-outcome-gate/{plan,selection,evaluation}.json`, per-split region integrals and forecast NPZ files. Training log: `data/backtests/outcome-gate-training.log`.

```powershell
$env:PYTHONPATH='ml'
.venv-ml/Scripts/python.exe -m unittest ml/test_outcome_distribution_gate.py
.venv-ml/Scripts/python.exe ml/train_outcome_distribution_gate.py --output data/backtests/oracle-outcome-gate-recheck
```
