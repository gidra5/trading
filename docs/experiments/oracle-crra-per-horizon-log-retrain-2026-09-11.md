# Corrected log-target retrain with per-horizon copies

The subsequent [joint online input/output calibration experiment](oracle-crra-joint-input-output-calibration-2026-09-11.md)
uses these frozen checkpoints and compares it with both the retained output
calibrator and an optimizer-matched output-only ablation.

This run restores the retained model architecture: four independent zero/sign/size triplets, with each horizon selecting its own checkpoints. It corrects the preceding log-target experiment, which used the weaker shared configuration.

The correction restores the retained forecast performance. Calibrated one-second all-tick log-return MSE skill is 3.512772%, versus 3.513035% for the retained per-horizon model and 2.588907% for the shared log model. The corrected copies outperform the shared log model at all four horizons; changing the targets to log returns produces only small differences from the retained per-horizon baseline.

Keep the existing forecast references: the earlier joint density still leads the matched one-second all-tick and active-conditional MSE comparisons, while the original per-horizon specialists remain the specialist reference. The new log checkpoints are preserved as the corrected retrain; these small mixed score changes do not establish a new overall winner.

The original zero and active-sign pretrained checkpoints are reused because their labels, inputs and pretraining objectives are unchanged by a log transform. Size pretraining is rerun on log-return magnitudes. All weights and magnitude shapes are then unfrozen and fine-tuned separately for each horizon.

## Matched forecast results

All models are scored on the same 36,134 test origins. MSE skill is `100*(1-MSE/mean(target^2))`, relative to predicting zero. Positive is better than zero. The next table uses **log-return** targets and actual expected log-return forecasts.

| Horizon | Retained per-horizon, calibrated | Corrected log, raw | Corrected log, calibrated | Previous shared log, calibrated |
| --- | ---: | ---: | ---: | ---: |
| 1 second | 3.513035% | 3.337578% | 3.512772% | 2.588907% |
| 15 minutes | -1.721150% | -1.970777% | -1.708833% | -3.342413% |
| 30 minutes | -6.554300% | -7.503050% | -6.538630% | -9.545352% |
| 2 hours | 1.122225% | 0.880552% | 1.125755% | -2.298190% |

For the same **15,505 nonzero one-second outcomes**, alongside the all-tick reference:

| Log-return MSE skill | Retained joint density | Retained per-horizon specialists | Corrected per-horizon log specialists |
| --- | ---: | ---: | ---: |
| Conditional-on-nonzero forecast | 6.295851% | 5.375472% | 5.375497% |
| Normal forecast, nonzero outcomes | 5.401253% | 5.728953% | 5.729022% |
| Normal forecast, all outcomes | 3.698769% | 3.513035% | 3.512772% |

Conditioning the forecast on a nonzero move removes its predicted zero-atom weight. This is a diagnostic: future activity is not known when forecasting. Filtering scoring rows alone keeps the normal forecast unchanged.

The corresponding calibrated **simple-return** all-tick skills are 1 second: 3.512195%, 15 minutes: -1.745846%, 30 minutes: -6.629164%, 2 hours: 0.923186%. Expected log returns and expected simple returns are integrated separately; neither is obtained by applying a logarithm/exponential to the other mean.

## Training selection

There are 69,315,656 retained parameters across 12 expert backbones, matching the original per-horizon model. Each deployed horizon has its own zero, sign, size and shape parameters. Multi-horizon size pretraining retains a separate validation-best checkpoint for each horizon before making the independent copies.

| Horizon | Zero initializer epoch | Sign initializer epoch | Log-size epoch | Fine-tune selected / completed |
| --- | ---: | ---: | ---: | ---: |
| 1 second | 15 | 12 | 9 | 3 / 15 |
| 15 minutes | 8 | 4 | 7 | 3 / 15 |
| 30 minutes | 49 | 3 | 7 | 3 / 15 |
| 2 hours | 7 | 0 | 6 | 0 / 12 |

A selected epoch zero means the pretrained combination won that horizon; it does not mean training was skipped. Fine-tuning uses full log-return NLL, averaged over all rows, and includes epoch zero in validation selection. No global average across horizons can veto a horizon-specific improvement.

The original sample protocol is preserved: 247,901 training and 34,560 validation origins, four-second stride, complete endpoints through 7,200 seconds, two observed production59 rows at `t-1,t`, and training-only normalization. Log targets are `log1p(P[t+H]/P[t]-1)` in float64, preserving exact endpoint zeros.

Log-size pretraining uses conditional normalized magnitude MSE, equal directions/horizons, learning rate `1e-4`, EMA half-life four epochs, and limits maximum 96/minimum 24/patience 20. Fine-tuning uses `1e-5`, EMA half-life two epochs, and 64/12/12 independently for each copy. Both use the original Muon/AdamW settings, seed 1337, gradient cap 100, batch 10,240 and microbatch 2,560. Networks use float32/high matmul; targets, scales and likelihood use float64.

Training took 982.453 seconds and evaluation 85.890 seconds. All choices were frozen before the evaluator loaded test data.

## Return law and calibration

The log law is the same as in the preceding log experiment: an exact zero atom and Gamma distributions for UP/DOWN log magnitudes. UP scale is smoothly bounded by 0.49 to ensure finite first and second positive price moments. DOWN log magnitudes remain unbounded. The current controller retains gamma zero; additional CRRA choices can require other moment conditions.

This corrects the architecture/selection confound. It still changes the magnitude family relative to the simple-return baseline: a Beta DOWN simple-return law becomes Gamma in log-magnitude coordinates. The experiment does not isolate a coordinate relabeling while holding the probability family fixed.

Calibration retains the old validation-selected modes/settings: full at one second, probability-only at the longer horizons, half-life 3,600 seconds, ridge 32/8 respectively. Updates use the new log likelihood only when an issued forecast reaches its actual endpoint. Each split/horizon/stage starts from identity. There is no test-based calibration search.

## Backtests

The same 42-hour tape and gamma-zero Oracle endpoint controller are used. Results for the independently selected fine-tuned copies:

| Horizon | Fees, raw | Fees, calibrated | Calibrated trades | Zero fees, calibrated |
| --- | ---: | ---: | ---: | ---: |
| 15 minutes | 0.000000% | 0.000000% | 0 | -1.859928% |
| 30 minutes | 0.000000% | 0.000000% | 0 | 0.007885% |
| 2 hours | -1.042219% | -1.042219% | 2 | -2.594118% |

The restricted menu remains long/cash `0,0.5,1` plus hold, unit funding, no pending orders, quote caps, price quantization, one-second intermediate marks, and valid terminal closing with residuals. Paid runs use received-amount taker fees of 0.05% per side. The tape is BTCUSDT spot history with synthetic Oracle parameters. Three conditional price means preserve those means but discard magnitude spread.

## Verification and artifacts

All 28 relevant tests passed. The independent training audit verifies target/statistic reconstruction, each horizon's validation minimum, singleton checkpoint shape, exact original classifier-state transfer and retained source integrity. The calibration audit checks 584,450 canonical queries across 32 streams and independently recomputes log/simple mean-MSE.

All 24 replay artifact checks passed. Lean verified 2,184 supplied decision certificates plus 108 execution fixtures. This checks the finite scenario/menu decisions, not predictive accuracy or unrestricted stochastic-policy optimality.

This is the previously inspected historical test period, not a fresh blind holdout. No weights, epochs or calibration settings were selected from these test results.

- Selection: `data\training\runs\stochastic-crra-per-horizon-log-v1\selection.json`; SHA-256 `97050630b636ea3e9e9773b6e9e2ae7ad7f2044496b1e8ff769c855867db3dbf`.
- Evaluation: `data\backtests\oracle-per-horizon-log\evaluation.json`.
- Training audit: `data\backtests\oracle-per-horizon-log\training-diagnostics.json`.
- Calibration audit: `data\backtests\oracle-per-horizon-log\calibration-diagnostics.json`.
- Replay comparison and Lean report: `data\backtests\oracle-per-horizon-log\comparison.json`, `data\backtests\oracle-per-horizon-log\lean-verification.json`.

```powershell
.venv-ml/Scripts/python.exe ml/train_per_horizon_log_hurdle.py
.venv-ml/Scripts/python.exe ml/evaluate_per_horizon_log_hurdle.py
.venv-ml/Scripts/python.exe ml/audit_per_horizon_log_hurdle.py
.venv-ml/Scripts/python.exe ml/audit_log_hurdle_calibration.py --evaluation data/backtests/oracle-per-horizon-log/evaluation.json --output data/backtests/oracle-per-horizon-log/calibration-diagnostics.json
node scripts/run-oracle-neural-specialists.mjs --output data/backtests/oracle-per-horizon-log
.venv-ml/Scripts/python.exe ml/report_per_horizon_log_hurdle.py
```

Reproduction requires new output directories; the original frozen checkpoints and completed experiment artifacts are retained.
