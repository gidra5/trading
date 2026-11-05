# What explains the current one-second MSE gap?

The surviving joint knot density's test advantage comes from smaller false mean predictions on unchanged ticks. The latest input/output-calibrated log specialists have better deployed-mean MSE on nonzero returns, better sign scores, better conditional magnitude scores, and a better zero-probability Brier score. Those advantages do not imply that their composed mean has lower overall MSE.

This audit uses the latest `joint-calibrated` specialist variant, not the earlier mean-only specialists. The word "joint" in that variant refers to input/output calibration; the comparison's "joint knot density" is the older single return-distribution model.

## Matched test scores

Both models use identical 36,134 one-second test origins, with 15,505 nonzero returns and 20,629 unchanged ticks. Forecasts and targets are log returns. Conditional size skills use the same training-constant baselines; signed-mean skills use the zero forecast baseline. The knot model's probabilities and conditional log magnitudes are integrated from its unchanged law.

| Metric | Joint knot density | Current calibrated specialists |
| --- | ---: | ---: |
| All-tick deployed-mean MSE skill | **3.698769%** | 3.445867% |
| Deployed-mean MSE skill on nonzero returns | 5.401253% | **5.756278%** |
| Active sign accuracy | 81.954208% | **84.469526%** |
| Active sign log loss | 0.425710 | **0.380669** |
| UP conditional log-magnitude MSE skill | 20.424860% | **24.086437%** |
| DOWN conditional log-magnitude MSE skill | 16.010038% | **17.593718%** |
| Zero-probability Brier score | 0.238859 | **0.237718** |
| Mean prediction RMSE on unchanged ticks | **0.050932 bps** | 0.059333 bps |
| Active-conditional mean RMSE on unchanged ticks | **0.087079 bps** | 0.110268 bps |

The active-conditional mean removes the zero gate. The last row is a diagnostic on quiet contexts, not a proposal to supply future activity to the model.

## Exact attribution of the overall skill gap

Since the actual return is zero on unchanged ticks, the squared-target baseline sum is identical over all ticks and over the nonzero subset. Consequently:

```
all-tick skill = nonzero deployed skill - 100*SSE_on_zero_ticks / sum(actual_log_return^2)

joint density:       5.401253 - 1.702484 = 3.698769
current specialists: 5.756278 - 2.310411 = 3.445867
```

The current model gains 0.355025 skill points on nonzero outcomes, but incurs an additional 0.607927 points of squared-error penalty on unchanged outcomes. Their difference is the joint density's 0.252901-point overall advantage. This identifies where the observed test gap arises; it is not a causal attribution to an architecture or training change.

The current model also has less squared error on the 1–5 bp and >=5 bp outcome groups. It has more error on unchanged and sub-1 bp outcomes. There are only five >=5 bp observations in this test sample, so this does not establish general tail superiority.

## Which distribution component is responsible?

For a log-return law with zero probability z, active UP probability p, and conditional magnitude means m+ and m-, the deployed mean is

```
mu = (1-z) * [p*m+ - (1-p)*m-].
```

These four quantities determine its mean completely. Once they are fixed, distribution spread or higher moments cannot further change log-return mean MSE. Separate component metrics do not score the products, subtraction, or the alignment of their errors across contexts. Sign-masked size metrics also only score the realized side, whereas both conditional means enter each deployed prediction.

On quiet contexts the joint density has a smaller net conditional signed mean even before its zero gate is applied. The current zero gate itself has better Brier score. Swapping it into the joint density's sign and sizes improves the joint density's test mean skill further, so the missing improvement is not simply more accurate zero classification.

## Component swaps

Here S means the current specialist component and J means the older joint density component. The three positions select zero probability, active sign probability, and the pair of conditional magnitude means respectively. All metrics below are all-tick log-return mean MSE skill, with no refitting.

| Zero / sign / sizes | Validation | Test |
| --- | ---: | ---: |
| S / S / S, current model | **3.347704%** | 3.445867% |
| J / J / J, joint density | 3.102474% | 3.698769% |
| S / J / J | 3.082725% | **3.780676%** |
| S / J / S | 2.108724% | 2.208171% |
| S / S / J | 3.061801% | 3.360192% |
| J / S / S | 3.287695% | 3.163359% |

Neither swapping in the joint sign alone nor its sizes alone improves the current composition. Their paired predictions work better together on test. The old joint training objective included normalized signed-price-mean MSE alongside density likelihood, while the latest specialists use likelihood-based fine-tuning/calibration. That is a plausible mechanism for different component-error balance; these diagnostics do not isolate the effect of the loss from architecture, distribution family, or training.

Validation reverses the overall ranking: current specialists score 3.347704%, joint density 3.102474%. The test-leading hybrid does not improve validation. No hybrid was selected or promoted, and the test gap is not evidence of universal superiority.

## Reproduction and checks

`ml/audit_current_forecast_mse.py` verifies source checkpoint, forecast, market and dataset hashes; aligns origins, timestamps and log targets; reconstructs the joint law's signed mean from its marginals; checks quadrature refinement; and reproduces the archived whole-mean skills to absolute tolerance 2e-12. It runs inference on the frozen joint checkpoint and reads the already-issued calibrated specialist forecasts. No predictor, calibration or model selection changes.

Results and all eight component swaps per split are in `data/backtests/oracle-current-mse-audit/audit.json`. The hashed per-example components, targets, origins and timestamps are retained in `validation-components.npz` and `test-components.npz` for further diagnostics.

```powershell
$env:PYTHONPATH='ml'
.venv-ml/Scripts/python.exe ml/audit_current_forecast_mse.py
```
