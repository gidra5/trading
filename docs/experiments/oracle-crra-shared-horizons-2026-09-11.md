# One zero, sign and size expert across all horizons

This experiment replaces four separately fine-tuned hurdle triplets with one
triplet whose zero, active-sign and conditional-size experts each predict all
four horizons. Separate fine-tuning was an implementation choice that allowed
independent adaptation and checkpoint selection; the likelihood and policy
do not require that separation.

The requested architecture shares each expert's backbone across 1 second,
15 minutes, 30 minutes and 2 hours. The three tasks retain independent
backbones. One forward pass through each of the three experts produces all
four hurdle laws, and one global validation score selects one checkpoint for
the whole triplet.

The shared implementation works and reduces model parameters by about 75%,
but **this run has worse mean-MSE and likelihood scores than the separate
models at every tested horizon**. Fine-tuning completed 12 epochs without
improving the combined validation objective, so the frozen selection retained epoch zero. The
reported shared model is consequently the intact pretrained triplet, with
optional causal online calibration, rather than a successful fine-tuning
improvement.

Following the comparison, the user chose to keep the stronger existing
models. The shared trial is not promoted. Retained checkpoint references
and matched nonzero-return scores are recorded in
`oracle-crra-retained-nonzero-scores-2026-09-11.md`.

## Forecast results

All-tick simple-return mean-MSE skill is
`100*(1-MSE/mean(actualReturn^2))`, on the same 36,134 test origins. Positive
skill beats predicting a zero return. The two online columns use the same
previously selected settings, with no recalibration hyperparameter search.

| Horizon | Separate, raw | Shared, raw | Separate, online | Shared, online |
| --- | ---: | ---: | ---: | ---: |
| 1 second | +3.337% | +0.550% | +3.512% | +1.047% |
| 15 minutes | -2.021% | -3.114% | -1.758% | -2.819% |
| 30 minutes | -7.612% | -11.379% | -6.645% | -10.178% |
| 2 hours | +0.673% | -2.919% | +0.920% | -2.532% |

Shared-pretrained and selected shared raw metrics are identical because
epoch zero was selected. Online calibration improves the shared model's
mean-MSE scores, but does not close the gap to the separate models.

This comparison includes both effects required to consolidate the old
models: choosing one initial checkpoint per expert across horizons, and
selecting one joint fine-tuning checkpoint. It does not isolate sharing
alone under identical per-horizon starting weights, which a single shared
backbone cannot retain when those starting weights differ. It also does
not establish that shared training is inherently inferior; it shows that
this fixed initialization/objective/optimization protocol did not improve
the retained shared model.

| Horizon | Separate raw NLL | Shared raw NLL | Separate online NLL | Shared online NLL |
| --- | ---: | ---: | ---: | ---: |
| 1 second | -3.999551 | -3.954900 | -4.004881 | -3.959942 |
| 15 minutes | -5.361137 | -5.341851 | -5.359991 | -5.340611 |
| 30 minutes | -5.026308 | -4.995345 | -5.026785 | -4.996103 |
| 2 hours | -4.238063 | -4.143339 | -4.235554 | -4.141067 |

Lower NLL is better. The one-second conditional sign accuracy, for example,
is 81.92% for shared online versus 84.42% for separate online. Zero Brier
error is similar (0.237666 versus 0.237695). The weaker shared one-second
return forecast therefore cannot be explained solely by zero-frequency
calibration.

## Backtest results

The 24 replays all use the same 42-hour tape. Calibrated policies produce
the following returns after ordinary fees:

| Horizon | Separate online | Shared online | Market trades, separate → shared |
| --- | ---: | ---: | ---: |
| 15 minutes | 0.000% | 0.000% | 0 → 0 |
| 30 minutes | 0.000% | 0.000% | 0 → 0 |
| 2 hours | -1.042% | -0.628% | 2 → 4 |

Without calibration, separate/shared after-fee results are 0%/0% at both
shorter horizons and -1.042%/-0.976% at two hours. The shared two-hour policy
loses less on this particular replay even though its average forecast scores
are worse. That is an execution-path outcome, not evidence that the shared
forecast law is more accurate. None of the fee-paying runs earns a positive
return.

| Horizon | Zero fees, separate raw | Separate online | Shared raw | Shared online |
| --- | ---: | ---: | ---: | ---: |
| 15 minutes | -1.694% | -1.860% | -1.072% | -1.317% |
| 30 minutes | -0.180% | +0.008% | -0.774% | -0.601% |
| 2 hours | -2.594% | -2.594% | -1.449% | -1.219% |

## Fine-tuning result

The selected starting mean validation NLL was **-4.377488**. Every trained
EMA checkpoint had a worse aggregate score; after epoch 12 it was
**-4.336567**. The first and last training epoch losses decreased from
-4.868697 to -4.989241, so the improving training objective did not generalize
to the combined validation objective. The individual validation NLLs show
the observed trade-off:

| Horizon | Starting validation NLL | Epoch 12 validation NLL |
| --- | ---: | ---: |
| 1 second | -3.947637 | -3.976851 |
| 15 minutes | -5.081846 | -5.042879 |
| 30 minutes | -4.698765 | -4.631155 |
| 2 hours | -3.781702 | -3.695383 |

Lower is better. Only the one-second validation likelihood improved;
losses at the other horizons outweighed that gain. The experiment therefore
kept the starting checkpoint under its fixed selection rule. These scores
do not by themselves identify whether gradient interference, distribution
shift or another generalization effect caused the regression.

Training plus fixed training/validation checkpoint evaluation took
204.359 seconds; forecast evaluation, calibration and exports took
62.422 seconds. Peak reported training GPU allocation was about 2,052 MiB.
The shared selection hash is
`ca0f90eaf013ff54a422dd9fbf14a9ccfed93d743ace82c4aea30b14e817a908`;
the evaluation plan hash is
`c04e28a980f16908c7735bf1ad4ff8b4975da023ebdfb33ae52fe12eeffa71b6`.

## Architecture and objective

| Model | Backbone copies | Trainable parameters |
| --- | ---: | ---: |
| Four separate horizon triplets | 12 | 69,315,656 |
| One shared multi-horizon triplet | 3 | 17,329,844 |

The shared zero and sign experts each have 4,130,753 parameters and four
logit outputs. The size expert has 9,068,330 parameters and eight conditional
mean outputs, UP/DOWN for each horizon. Eight learned global shape parameters
complete the hurdle law. This reduces retained inference parameters by about
75%; it is an architecture count, not a measured latency claim.

Fine-tuning minimizes

```text
mean over training rows and horizons of
    zero BCE
  + 1[return != 0] * conditional-sign BCE
  + 1[return > 0] * UP Gamma negative log likelihood
  + 1[return < 0] * DOWN Beta negative log likelihood
```

All expert weights, readouts and distribution shapes can learn. The loss
uses the exact effective-batch denominator across rows and horizons, even
with uneven microbatches. It contains no auxiliary mean-MSE objective.
Mean validation NLL across all horizons selects a single EMA checkpoint,
including the unmodified starting point. Per-horizon diagnostics do not
select separate epochs or splice their weights into the shared model.

The objective averages four marginal hurdle NLLs. Shared weights jointly
learn those marginals; this does not estimate cross-horizon dependence or
assume overlapping returns are independent.

NLL levels vary by horizon, but an additive horizon-specific constant would
not change the shared checkpoint ranking. No return-size normalization is
added to the proper hurdle NLL. Sharing can still introduce competing
gradients and different generalization trade-offs across horizons.

## Initialization and fixed protocol

The prior pretraining retained multi-horizon checkpoints selected at different
epochs for individual horizons. A shared model needs one intact checkpoint
per expert. This experiment scores every retained candidate on validation
using its original specialist objective, then selects the smallest equal-
horizon average: all-row BCE for zero, active-only BCE for sign, and
conditional normalized MSE averaged equally over directions and horizons
for size. Candidate order breaks exact ties. This is a search among retained
checkpoints, not among missing historical epochs.

Selected initializer epochs are zero **15**, sign **4** and size **9**.
Each entire backbone and all horizon readout rows are copied intact. No
per-horizon weights are stitched together. Initial UP/DOWN shape estimates
use only training residuals and are computed independently for all eight
horizon/direction combinations.

The source neural selection is
`data/training/runs/stochastic-crra-hurdle-nll-v1/selection.json`, hash
`8ac7d25215c733b284d26cf97c09b741ffc81cc786eae20ef195547286777317`.
The new training plan was frozen with hash
`56d5b05108f516da36702548ca24f98cc24b0de779b4b9a7b6554c0654c8d7c1`.

The dataset, inputs and accuracy populations match previous experiments:
two observed production59 states at `t-1,t`; 247,901 training, 34,560
validation and 36,134 test origins; four-second stride; all scored origins
complete through the maximum horizon. Input normalization and conditional
size scales are training-only quantities.

Training uses Muon/AdamW, learning rate `1e-5`, zero weight decay, gradient
norm cap 100, EMA half-life two epochs, seed 1337, effective batches of
10,240 and microbatches of 2,560. The fixed limit is 64 epochs, with a
minimum of 12 and patience of 12 epochs without global validation improvement.
Neural experts use float32, with float64 physical means, shapes, simple-return
targets and hurdle likelihood. Each expert is compiled separately; the small
physical packing and likelihood remain eager. The trainer never loads test.

## Forecast and execution comparison

The comparison has four variants:

- `separate-raw`: previous per-horizon fine-tuned models, uncalibrated.
- `separate-online`: those models with the prior validation-selected online
  calibration for each horizon.
- `shared-raw`: the new global shared checkpoint, uncalibrated.
- `shared-online`: the new checkpoint with exactly the same per-horizon
  calibration settings as `separate-online`.

The prior calibration selection hash is
`3a2987241eace925c2a1281f936e5d641524bba048d6322fa482a04c3d694f63`.
Its selected mode is full calibration at one second and probability-only
calibration at longer horizons. Settings are reused without reselection on
the shared model. This holds the calibration procedure fixed; it does not
claim those settings are optimal for the new architecture.

Both architectures start calibration from identity independently on each
split. Labels can update only when their `origin+horizon` endpoint is
observed. The canonical four-second issue stream and overlap weight
`min(1,4/horizon)` match the prior experiment. The weight remains a heuristic
for overlapping outcomes. Prediction queries do not create observations.

Separate-model raw and calibrated arrays are reused after checking hashes,
origins and provenance. Shared inference produces a single `[N,4,6]` packed
forecast matrix per checkpoint/split. Longer horizons use only their complete
prefixes of the union of forecast origins. Shared-pretrained raw scores are
reported to distinguish fine-tuning changes from the required consolidation
of the initial checkpoints.

The same 42-hour observed tape drives 24 replays: four variants, three
trading horizons and ordinary/zero fees. Gamma remains zero. The unchanged
restricted Oracle controller uses long/cash targets `0,0.5,1` plus hold,
unit funding, no pending orders, quote caps, price quantization, one-second
intermediate marks and valid final closing with residuals. Ordinary
received-amount taker fees are 0.05% per side. Exports use three conditional-
mean support points, which discard within-direction magnitude spread.

Lean verifies supplied finite-menu decisions under that execution model.
It does not establish forecast accuracy or true-market optimality. The
reused BTCUSDT spot tape and synthetic account parameters do not represent
a live perpetual funding book, and this previously inspected historical
test period is not a fresh blind holdout.

## Verification

All **25 relevant tests passed**: four shared-training tests, four hurdle
training/transfer tests, seven hurdle likelihood tests and ten causal online
calibration tests. These cover three-expert ownership, full-horizon gradient
flow and shape initialization, exact effective-batch loss accumulation and
one global checkpoint even when individual horizons prefer different epochs.

The independent audit verified all retained initializer candidates against
their original training logs, exact transfer of every pretrained state
tensor, one global validation minimum and unchanged selected tensors at
epoch zero. It checked **1,168,900 canonical queries across 32 trajectories**,
105 artifact hashes, fixed calibration settings, frozen distribution shapes,
causal label maturity and baseline artifact/metric parity. NumPy/SciPy
independently recomputed hurdle NLL, mean-MSE and zero-calibration scores.

The audit reproduced **73,999 cached forecast rows bit-for-bit** in 30
original GPU batch contexts. A separate numerical check found that changing
device/batch context under PyTorch's `high` float32 matmul precision changes
sampled signed mean returns by up to **0.0242 bps**. Conditional magnitude
means differed by up to **0.0652 bps**. Exact reproduction is therefore tied
to the recorded GPU precision and batch contexts; cross-device bitwise
equivalence is not claimed. No frozen forecasts or algorithm settings were
changed to resolve this check.

The full audit artifact is
`data/backtests/oracle-shared-hurdle/diagnostics.json`, SHA-256
`2a5fcd1a1247784d9bd3384741e7d3c4ffad4aab1397d37bfb66407630c15913`.

The replay artifact checks passed for all 12 exports and all 24 replays.
**Lean passed every one of the 2,184 decision certificates, plus all 108
execution conformance fixtures.** The certificate file hash is
`e26cafeee588ab235883e54f2ec48ed3ce188e170992ed800b97ce954fb88f32`.
These certify supplied finite-menu decisions under the declared execution
model; they do not prove forecast accuracy or market optimality.

## Implementation and reproduction

The current `ml/train_hurdle_sign_size_forecast.py` now performs shared
initialization, all-horizon shape initialization, one jointly trained triplet
and one global checkpoint selection. It replaces the default singleton
fine-tuning path. Historical checkpoints and results remain intact for the
comparison.

Additional implementation:

- `ml/evaluate_shared_hurdle_forecast.py`: all-horizon shared inference,
  fixed online calibration, matched scoring and replay exports.
- `ml/test_shared_hurdle_training.py`: shared parameter ownership, per-horizon
  gradients, global selection and full-horizon shape initialization checks.
- `ml/audit_shared_hurdle_forecast.py`: independent initialization, selection,
  cache inference, calibration timing and forecast metric checks.
- `scripts/report-oracle-shared-hurdle.mjs`: replay and artifact checks and
  collection of every decision certificate.

Artifacts are saved under
`data/training/runs/stochastic-crra-shared-hurdle-v1/` and
`data/backtests/oracle-shared-hurdle/`.

```powershell
.venv-ml/Scripts/python.exe ml/train_hurdle_sign_size_forecast.py --prepare-only
.venv-ml/Scripts/python.exe ml/train_hurdle_sign_size_forecast.py --run-prepared
.venv-ml/Scripts/python.exe ml/evaluate_shared_hurdle_forecast.py --prepare-only
.venv-ml/Scripts/python.exe ml/evaluate_shared_hurdle_forecast.py --run-prepared
.venv-ml/Scripts/python.exe ml/audit_shared_hurdle_forecast.py
node scripts/run-oracle-neural-specialists.mjs --output data/backtests/oracle-shared-hurdle
```

The commands reject overwriting frozen plans or completed evaluations.
Reproduction requires fresh output paths and a matching replay plan.
