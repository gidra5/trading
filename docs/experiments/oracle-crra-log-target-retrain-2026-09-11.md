# Retraining the shared specialists on log returns

This experiment used the shared architecture. The requested correction using
independent per-horizon copies is now documented in
[the corrected log-target retrain](oracle-crra-per-horizon-log-retrain-2026-09-11.md).
It restores one-second calibrated log MSE skill to 3.512772% and closely matches
the retained per-horizon baseline. The results below remain the shared trial.

The zero, sign and size specialists were retrained from fresh initialization,
with **log-return magnitude targets** and joint likelihood fine-tuning in
log-return coordinates. The calibrated model improves the previous shared
model's one-second forecast, but does not beat the retained separate specialists
or the earlier joint density. Retain those stronger existing references.

## Matched forecast results

Every model uses the same 36,134 test origins. MSE skill is
`100*(1-MSE/mean(actualReturn^2))`, relative to predicting zero, computed in
the return coordinate and population specified by each table.

All-tick **log-return** MSE skill, with the same previously selected causal
calibration settings:

| Horizon | Retained separate specialists | Previous shared specialists | Retrained shared log specialists |
| --- | ---: | ---: | ---: |
| 1 second | 3.513035% | 1.047327% | 2.588907% |
| 15 minutes | -1.721150% | -2.776664% | -3.342413% |
| 30 minutes | -6.554300% | -10.067903% | -9.545352% |
| 2 hours | 1.122225% | -2.292012% | -2.298190% |

For the same **15,505 nonzero one-second outcomes**:

| Log-return MSE skill | Earlier joint density | Retained separate specialists | Retrained shared log specialists |
| --- | ---: | ---: | ---: |
| Conditional-on-nonzero forecast | 6.295851% | 5.375472% | 3.988041% |
| Normal forecast, scored only on nonzero outcomes | 5.401253% | 5.728953% | 4.576043% |
| Normal forecast, all outcomes | 3.698769% | 3.513035% | 2.588907% |

Conditioning the forecast on a nonzero move removes its predicted zero-atom
weight. It is a diagnostic, since future activity is unknown when forecasting.
Simply filtering the scoring population leaves the normal forecast unchanged.
All log scores integrate `E[log(P_next/P_now)]` under the forecast law; they
never use `log(1+E[simple return])`.

The new one-second active sign accuracy is 83.811674% after calibration,
versus 84.424379% for the retained separate specialists. Its all-tick log MSE
skill improves from 2.343025% without calibration to 2.588907% with calibration.
Conditional sign cross entropy is worse than the retained separate model
(0.427257 versus 0.383853), while zero Brier error and conditional log-size
scores are slightly better. These component scores identify a sign-forecast
deficit, but do not by themselves prove its causal contribution to mean MSE.

For comparison with previous **simple-return** tables, the new calibrated
all-tick skills are 2.588721%, -3.386247%, -9.650370% and -2.536285% at
1, 900, 1800 and 7200 seconds respectively. Switching evaluation coordinates
still has little effect at one second. This retrain does not recover the
historical higher headline score.

## Training and selection

Each of the three independent experts shares its backbone across all four
horizons, with four zero logits, four active-sign logits and eight conditional
log-magnitude outputs. The combined model has **17,329,844 parameters**.

The sample protocol and historical architectures match the preceding shared
experiment: two observed production59 states at `t-1,t`; 247,901 training and
34,560 validation origins; four-second stride; every scored origin complete
through the maximum horizon. Input normalization and conditional log-magnitude
scales are computed from training data only. Exact unchanged endpoint prices
remain zero; sign labels are unchanged by the coordinate transformation.

Pretraining uses all-row zero BCE, active-only sign BCE and sign-conditional
log-magnitude MSE normalized by its training mean. Direction and horizon
weights are equal within the size objective. All three experts start from
seed 1337; previous weights are not used as initializers.

| Stage | Completed epochs | Selected epoch | Global validation objective |
| --- | ---: | ---: | --- |
| Zero pretraining | 36 | 16 | Mean all-row zero BCE |
| Sign pretraining | 26 | 6 | Mean active-sign BCE |
| Log-size pretraining | 29 | 9 | Mean balanced conditional normalized log-size MSE |
| Joint log-likelihood fine-tuning | 12 | 0 | Mean full log-hurdle NLL |

One intact checkpoint per expert minimizes the equal-horizon validation
average. Joint fine-tuning uses one global checkpoint for the entire triplet.
It averages the full mixed-law NLL over all rows and horizons, without an
auxiliary MSE term. Every expert weight and the distribution shapes can learn.

Joint validation NLL worsened from **-4.374933** at the pretrained combination
to **-4.310195** at epoch 12; lower is better. The fixed stopping rule therefore
retained epoch zero. The exported "finetuned" selection is the newly pretrained
triplet, and its forecasts exactly match the pretrained stage. It does not
represent a successful joint fine-tuning improvement.

Training uses Muon/AdamW, zero weight decay, gradient norm cap 100, effective
batches of 10,240 and microbatches of 2,560. Pretraining uses learning rate
`1e-4`, EMA half-life four epochs, maximum 96 epochs, minimum 24 and patience
20. Joint fine-tuning uses `1e-5`, EMA half-life two epochs, maximum 64, minimum
12 and patience 12. Neural calculations use float32/high matmul precision;
log targets, physical means, scales, shapes and likelihood use float64.
Training took **564.968 seconds** on the RTX 3070; evaluation took **93.687
seconds**. The trainer never loads test data.

## Log-law and calibration changes

Let `X=log(P_next/P_now)`. The law has an exact zero atom and two Gamma
distributions for the positive magnitudes of `X`, one for each direction.
For conditional log means `muUp,muDown` and shapes `aUp,aDown`:

```text
E[X | UP] = muUp
E[X | DOWN] = -muDown
E[exp(X) | UP] = (1 - muUp/aUp)^(-aUp)
E[exp(X) | DOWN] = (1 + muDown/aDown)^(-aDown)
```

The UP scale is smoothly bounded using
`muUp=aUp*0.49*tanh(rawMean/(aUp*0.49))`. This ensures finite first and second
positive price moments and is close to the identity for small means. It
bounds a distribution parameter; positive realized log returns still have
unbounded support. DOWN log magnitudes have no simple-return lower-bound
restriction. Arbitrary CRRA risk aversion can require additional moments;
this experiment retains the existing **gamma=0** controller.

The independent bound audit found no floor hits or saturation. The largest
test reduction in a conditional UP log mean was 0.002866% relative, or
0.002559 bps absolute. The one-second maximum relative reduction was
0.0006235%. The numerical effect of this bound is small on these forecasts.

Calibration reuses the preceding validation-selected configurations: full
calibration at one second, probability-only calibration at longer horizons,
half-life 3,600 seconds, ridge 32 at one second and 8 otherwise. It uses the
new log-law likelihood, with the proper UP parameterization derivative and
ordinary Gamma DOWN log-mean update. No calibration hyperparameters are
reselected. Each split/checkpoint/horizon starts at identity and consumes
labels only when their `origin+horizon` endpoint has become observable.

This is a complete log-target retrain, **not an isolated coordinate-only
ablation**. Relative to the old simple-return family, DOWN Beta becomes Gamma
in log-magnitude coordinates and UP gains an exponential-moment constraint.
Fresh global pretraining selection also differs from the previous shared
experiment's selection among retained candidate checkpoints. These results
do not establish that log returns or shared architectures are inherently worse.

## Backtests and verification

All 24 replays use the same 42-hour observed price tape. Pretrained and selected
combined results are identical. None of the fee-paying runs earns a positive
return.

| Horizon | After fees, raw | After fees, calibrated | Calibrated trades | Zero fees, calibrated |
| --- | ---: | ---: | ---: | ---: |
| 15 minutes | 0.000000% | 0.000000% | 0 | -0.703436% |
| 30 minutes | 0.000000% | 0.000000% | 0 | -0.418302% |
| 2 hours | -0.627555% | -0.627555% | 4 | -1.219372% |

The unchanged controller uses gamma zero, long/cash targets `0,0.5,1` plus
hold, unit funding, no pending orders, quote caps, price quantization,
one-second intermediate marks and valid final closing with residuals.
Paid runs use received-amount taker fees of 0.05% per side. The tape is
BTCUSDT spot history with synthetic Oracle account parameters, not a live
perpetual funding book. Exports preserve the three conditional price means
and zero mass; they discard within-direction magnitude spread.

All **26 new tests passed**, covering independent likelihood and moment
formulas, coordinate conversion, gradients, uneven microbatch accumulation,
all-horizon shared ownership, calibration derivatives, exact maturity and
future-suffix isolation. The training audit independently reconstructs train
and validation targets, constants, normalization and time bounds, verifies
all four global checkpoint minima, and checks exact pretrained state transfer.
All retained source checkpoints remain unchanged.

The evaluator checks matching origin and target hashes, reproduces the
retained simple-return scores and joint one-second log audit, and compares
raw and calibrated forecasts for both new stages. It verifies 137 artifact
and source hashes. The independent calibration audit checks 32 saved streams
and 584,450 canonical calibration queries: exact maturity, fixed settings,
cold identity, unchanged shapes and correct coefficient transformations.
Independent analytic log/simple MSE recomputation differs by at most
`4.55e-13` in the reported numeric metrics.

All replay artifact checks passed. **Lean passed all 2,184 decision
certificates and 108 execution fixtures.** These verify the supplied finite
scenario/menu decisions under the declared execution model. They do not
prove forecast accuracy, the approximation's general optimality, or optimality
over unrestricted policies.

## Artifacts and reproduction

Training artifacts: `data/training/runs/stochastic-crra-log-hurdle-v1/`.
Evaluation and replay artifacts: `data/backtests/oracle-log-hurdle/`.

- Frozen training plan SHA-256:
  `3dbe43cbea1822fb847f27e648ef0c73ee16d681fec04a260133e4b61c5de461`.
- Frozen selection SHA-256:
  `235d9693cea5e3c0c778050968269ba600e99864a9709e6eab4822de79730e0e`.
- Selected combined checkpoint SHA-256:
  `2979f62ef4cb441542cae2de60b1a96a00c174d985e348416c3f8336ff5db19c`.
- Evaluation SHA-256:
  `91f78374d6803a5bbd9c8ee66a4d4f57fa589105d0dba493b75743f15049afd7`.

```powershell
.venv-ml/Scripts/python.exe ml/train_log_hurdle_sign_size_forecast.py --prepare-only
.venv-ml/Scripts/python.exe ml/train_log_hurdle_sign_size_forecast.py --run-prepared
.venv-ml/Scripts/python.exe ml/evaluate_log_hurdle_forecast.py
.venv-ml/Scripts/python.exe ml/audit_log_hurdle_training.py
.venv-ml/Scripts/python.exe ml/audit_log_hurdle_calibration.py
node scripts/run-oracle-neural-specialists.mjs --output data/backtests/oracle-log-hurdle
```

Reproduction requires fresh output paths; frozen plans and completed results
are protected from overwriting. The test period was already inspected in
previous experiments, so it is a reused benchmark rather than a fresh blind
holdout. Test results do not select new weights or calibration settings.
