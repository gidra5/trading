# Joint online input and output calibration

Implemented and replayed joint online calibration on the corrected, independently trained per-horizon log-return specialists. All predictor weights, distribution shapes and training normalization remain frozen. A single current likelihood loss backpropagates through the predictor to both adapters.

The joint variant improves test all-tick log-return MSE skill over the same-optimizer output-only ablation at 2/4 horizons, and over the retained Fisher output calibration at 3/4 horizons. At one second, joint calibration scores 3.445867%, versus 3.444572% for the optimizer-matched output-only ablation and 3.512772% for the retained output calibrator.

## Comparable forecast results

Log-return MSE skill is `100*(1-MSE/mean(target^2))`, measured on the same 36,134 test origins. Negative skill loses to predicting zero. The retained column is the previous causal Fisher output calibration on the same frozen log checkpoints.

| Horizon | Retained output calibration | Output-only, matched Adam | Joint input + output |
| --- | ---: | ---: | ---: |
| 1 second | 3.512772% | 3.444572% | 3.445867% |
| 15 minutes | -1.708833% | -0.882122% | -0.871397% |
| 30 minutes | -6.538630% | -4.306413% | -4.306691% |
| 2 hours | 1.125755% | 2.124064% | 2.121018% |

For the same 15,505 nonzero one-second outcomes:

| Metric | Retained output calibration | Output-only, matched Adam | Joint input + output |
| --- | ---: | ---: | ---: |
| Normal forecast, nonzero outcomes | 5.729022% | 5.672998% | 5.756278% |
| Conditional-on-nonzero forecast | 5.375497% | 5.233175% | 5.218515% |

The conditional-on-nonzero score removes the forecast zero-atom weight. It is a diagnostic, because future activity is unknown at issuance.

Test log-return negative log likelihood (lower is better; the update objective):

| Horizon | Retained output calibration | Output-only, matched Adam | Joint input + output |
| --- | ---: | ---: | ---: |
| 1 second | -4.004764 | -4.003792 | -4.005978 |
| 15 minutes | -5.359984 | -5.362477 | -5.362310 |
| 30 minutes | -5.026616 | -5.034958 | -5.034852 |
| 2 hours | -4.234555 | -4.239129 | -4.239057 |

Validation log-return MSE skill, shown separately from test:

| Horizon | Retained output calibration | Output-only, matched Adam | Joint input + output |
| --- | ---: | ---: | ---: |
| 1 second | 3.442681% | 3.278972% | 3.347704% |
| 15 minutes | -1.693444% | -1.216842% | -1.194309% |
| 30 minutes | -2.867613% | -2.817511% | -2.793760% |
| 2 hours | -4.871459% | -4.233872% | -4.230922% |

## Joint training and causal timing

Each horizon owns 122 adaptive parameters: 59 log gains and 59 shifts on the normalized inputs, plus four output offsets. The input transform is shared over the two observed ticks and the three experts. In standardized coordinates it is `z -> exp(g)*z+b`. The output adapter shifts the zero/sign logits and the two magnitude means. UP uses mean/shape log odds to preserve finite first/second positive price moments; shapes stay fixed.

One Adam optimizer updates both adapters from full log-hurdle negative log likelihood, with an identity-centered quadratic prior. Input learning rate is 0.001, output learning rate 0.01; respective prior precisions are 32 and 8. Gains are bounded to [0.5,2], shifts to one training standard deviation, probability offsets to ±2 and magnitude offsets to ±log(2). These settings were fixed before validation/test evaluation; no test-driven selection or tuning occurred.

Updates happen on a fixed 60-second clock. At time t, only forecasts issued earlier whose endpoint t_origin+H has already been observed are eligible. The loss uses the last 3,600 seconds of mature labels, exponentially weighted with a 3,600-second half-life. At most 128 deterministic midpoint representatives approximate that window, preserving each bin’s total weight. The previous overlap weight min(1,4/H) is retained as a heuristic for effective information.

A single loss backward pass reaches both adapters through the frozen predictor. This is joint end-to-end online optimization, with no claim that one gradient step finds the exact window optimum. Earlier features are replayed under the current adapters when their labels mature; historical issued forecasts are never rewritten. Prediction queries do not train, and changing the trading-query frequency cannot create extra updates.

All variants start at identity independently for each split/horizon, using the same four-second issuance stream and frozen base checkpoints. The output-only ablation uses exactly the same clock, objective, window, prior and output optimizer, with input parameters disabled. Archived Fisher calibration remains a separate reference. Saved online state includes both adapters, Adam state and the last update timestamp for deterministic restart with the same base checkpoint.

## Backtests

Gamma remains zero. The same 42-hour BTCUSDT spot tape is replayed through the restricted synthetic Oracle long/cash endpoint account. Paid runs charge received-amount taker fees of 0.05% per side.

| Horizon | Output-only, fees | Joint, fees | Joint trades | Output-only, zero fees | Joint, zero fees |
| --- | ---: | ---: | ---: | ---: | ---: |
| 15 minutes | 0.000000% | 0.000000% | 0 | -0.656531% | -0.587085% |
| 30 minutes | 0.000000% | 0.000000% | 0 | -0.114951% | -0.114951% |
| 2 hours | -1.042219% | -1.042219% | 2 | -2.594118% | -2.594118% |

These historical backtests reduced the forecast law to three conditional price means and used target exposures 0,0.5,1 plus a zero trade. Zero trade holds the account; zero target exposure attempts to close it. This was a matched comparison under a restricted controller, not a comparison using continuous optimal actions. The [continuous-action, full-distribution replay](oracle-crra-continuous-full-law-2026-09-12.md) supersedes this controller comparison. Both experiments use unit funding, no pending orders, quote-size caps, price quantization, one-second maintenance marks and valid final closing with residuals.

## Verification

The compiler specialization limit is 64 and exceeding it raises an error. The default limit caused an execution fallback during an incomplete validation stream; that partial stream was restarted before test data loaded. Completed validation streams were retained, and the original source, plan and progress are preserved under `implementation-history`, with the change recorded in `runtime-correction.json`. Calibration settings did not change.

All 23 relevant tests passed, including seven new checks for identity/physical output transforms, a common gradient updating both adapters, frozen predictor weights, maturity rejection, future-prefix invariance, query invariance, the output-only ablation and deterministic restart.

The independent audit checked 584,450 canonical forecasts and 38,300 optimizer events, reconstructing maturity, representative indices and decayed weights. It replayed all 287,245 adapted joint forecasts from saved input/output traces with the recorded compiled batch schedule. Maximum replayed logit error was 0; maximum relative magnitude error was 4.22627471e-16. Cold forecasts match the archived base predictions exactly.

Log/simple expected-return scores were independently reconstructed with maximum skill error 2.22044605e-16. All 12 replay artifact checks passed on the identical tape. Lean verified 1,092 supplied decision certificates and 108 execution fixtures. This checks the supplied finite scenario/menu decisions, not forecast quality or unrestricted optimality.

This is a first fixed-configuration calibration trial on an already inspected historical test period. It does not establish that input calibration will improve every model, horizon or future period. Existing retained forecast references are preserved; no live deployment is changed by the experiment.

- Frozen protocol: `data\backtests\oracle-joint-calibration\plan.json`.
- Forecast results and provenance: `data\backtests\oracle-joint-calibration\evaluation.json`.
- Causal/forward audit: `data\backtests\oracle-joint-calibration\calibration-diagnostics.json`.
- Replays and Lean: `data\backtests\oracle-joint-calibration\comparison.json`, `data\backtests\oracle-joint-calibration\lean-verification.json`.
- Per-split/horizon traces and restart state: `data\backtests\oracle-joint-calibration\cache`.

```powershell
.venv-ml/Scripts/python.exe ml/evaluate_joint_hurdle_calibration.py
.venv-ml/Scripts/python.exe ml/audit_joint_hurdle_calibration.py
node scripts/run-oracle-neural-specialists.mjs --output data/backtests/oracle-joint-calibration
.venv-ml/Scripts/python.exe ml/report_joint_hurdle_calibration.py
```
