# Horizon input added to trained price models

This experiment starts from the existing direct59 and base17 price-law checkpoints. Both extended models reproduce their parent's complete one-second distribution exactly at initialization. Training only the added horizon weights also preserves that distribution exactly after training. This addresses preservation of existing capability, which the preceding from-scratch horizon trial did not test.

The two parents are strong standalone constituents of the existing ensemble. This run does not initialize from or replace the full feature-mixture ensemble or its other calibrated specialists. No existing checkpoint or bot configuration was overwritten.

## Construction

For integer horizon `h` in 1–86,400 seconds, the first GLU preactivation becomes

\[
W\,\operatorname{normalize}(x)+b+
v\left[\tanh(h/21600)-\tanh(1/21600)\right].
\]

The original network parameters, normalizers, knot positions, Gaussian-CDF coordinate scale, and zero-atom output are copied unchanged. Only `v` is added, initialized to zero. Each parent has a width-128 GLU, so this adds 256 trainable scalars: one input weight for each value/gate preactivation. The horizon contribution uses a separate multiply/add to leave the original feature matrix multiplication unchanged.

The raw return coordinate remains `log(P[t+h]/P[t]) / sqrt(h)`. At `h=1` this scaling is exactly one. Thus, at initialization, identical logits and the identical basis imply an identical full one-second law, including zero probability, sign probabilities, magnitude distribution and moments. No distribution refitting, distillation, or moment-only matching is involved.

At `h=1` the new preactivation term is zero for **any** `v`. Consequently, freezing the parent and training only `v` preserves its H1 law algebraically throughout training. Jointly updating the parent has only the initialization guarantee.

## Training

Two arms per parent:

- **Horizon weights only:** parent head and native feature model frozen; train `v` at learning rate 0.001.
- **Joint fine-tuning:** same initialization; train the entire price head at 0.0001 and `v` at 0.001. The native feature model remains frozen.

Horizon sampling and labels are unchanged from the preceding trial. `Pr(H=h)` is proportional to `tanh(h/21600)-tanh((h-1)/21600)`, truncated at 24 hours. Two horizons are sampled per origin each epoch. No extra one-second oversampling or retention penalty is added. The loss is full raw-return mixed-law NLL plus scaled-return mean MSE divided by the training scaled-return second moment.

Each parent uses 103,693 complete training origins and 14,760 complete validation origins, with all 24-hour targets inside their own splits. Native training precedes these head-training origins. The same random sampled pairs are used across arms. Validation pairs are frozen. Adam batch size is 4096; budget is 48 epochs, with patience 12 after at least 24 epochs. Selection uses validation loss, separately for each arm. All selections freeze before test scoring.

The two horizon-only arms selected epoch 48, the budget boundary; convergence is not established. Each saw 9,954,528 training pairs. Joint arms ran 24 epochs, with 4,977,264 pairs each, selecting epoch 5 for direct59 and epoch 9 for base17.

| Parent and arm | Initial validation loss | Selected validation loss | Selected epoch |
|---|---:|---:|---:|
| direct59, horizon weights only | 1.716688 | 0.308873 | 48 |
| direct59, joint fine-tuning | 1.716688 | -0.574073 | 5 |
| base17, horizon weights only | 1.128917 | -0.086730 | 48 |
| base17, joint fine-tuning | 1.128917 | -0.679921 | 9 |

These are checkpointed standalone refits without online calibration. The experiment does not silently add, remove, or refit calibration from another ensemble member.

## Results

H1 uses all 36,134 original test origins, making the scores comparable to the parent results. Longer horizons use the same 16,334 complete 24-hour origins as the previous horizon trial. These long-horizon targets overlap heavily; the test was also inspected in earlier experiments. Long-horizon scores should not be treated as independent-sample evidence or a profitability result.

Log-return MSE skill relative to predicting zero:

| Model | 1 second | 5 minutes | 15 minutes | 2 hours |
|---|---:|---:|---:|---:|
| direct59, initial extension | 4.5298% | -0.8278% | -2.2540% | -2.1167% |
| direct59, horizon weights only | **4.5298%** | -0.8995% | -2.6629% | -3.5550% |
| direct59, joint fine-tuning | -29.9777% | -11.0155% | -10.0594% | -2.0052% |
| base17, initial extension | 4.0943% | -0.2856% | -1.1931% | -1.2708% |
| base17, horizon weights only | **4.0943%** | -0.3234% | -1.4287% | -3.2069% |
| base17, joint fine-tuning | -10.9569% | -2.1427% | 0.6806% | 10.3291% |

The horizon-only weights learn distribution changes despite negative long-horizon mean skill. At two hours, direct59 NLL improves from -0.6734 to -1.7961, and base17 from -1.1591 to -2.3156; lower is better. Full joint fine-tuning gives -3.6158 and -3.9791, respectively, at the cost of changing one-second forecasts.

The capability result is therefore positive and narrower than uniform accuracy improvement: the added parameter can condition the existing distribution on horizon while preserving the complete original one-second law. This run does not demonstrate improved mean predictions at every horizon. Joint training with the broad horizon weighting can sacrifice short-horizon accuracy; initialization equality alone does not prevent that. Training only the added weights prevents H1 forgetting by construction but restricts adaptation of the rest of the network.

## Verification

Before fitting, both parents passed bit-for-bit H1 logit equality on all 125,293 cached training and 34,560 validation origins. After all selections froze, they also passed on all 36,134 test origins.

The independent audit checked:

- Every copied initial head state tensor equals its original checkpoint tensor.
- Training-source, parent-checkpoint and selected-checkpoint hashes match.
- For the horizon-only arms, every head tensor remains unchanged after training; the H1 saved means, likelihoods, zero probabilities and conditional sign probabilities are exactly unchanged.
- Causal native features and trained adapter logits reconstructed from 2,560 actual observed-state inputs match the corresponding cached-input computation exactly for both parents.
- Learned two-hour logits differ from initial logits (maximum absolute differences 4.4121 and 4.9838), confirming that the added horizon weights affect the distribution.

Three unit tests cover complete H1-law equality, learning at another horizon while H1 remains fixed, and checkpoint reload with nonzero horizon weights. All passed. Final moments use the same float64 quadrature/scoring method for parents and extensions; exact equality is asserted between identically evaluated laws rather than between differently rounded archival NLL reports.

## Files and use

Implementation:

- `ml/warm_target_horizon_forecast.py`
- `ml/train_warm_target_horizon_forecast.py`
- `ml/audit_warm_target_horizon_forecast.py`
- `ml/test_warm_target_horizon_forecast.py`

Artifacts: `data/backtests/oracle-warm-target-horizon/`, containing initial, selected and final diagnostic checkpoints, frozen plan/selection, per-horizon forecasts, evaluation, audit and training histories. Selected horizon-only checkpoints are `direct59-horizonWeightsOnly.pt` and `base17-horizonWeightsOnly.pt`.

`load_warm_observed_forecast(checkpoint, device)` loads a causal adapter accepting two observed production59 states and a target horizon in seconds. `load_warm_forecast` loads the head for already-computed native features. Full zero-plus-knot distributions are available through the head API; the NPZ evaluation summaries are not full-law exports.

```powershell
.venv-ml/Scripts/python.exe -m unittest discover -s ml -p test_warm_target_horizon_forecast.py
.venv-ml/Scripts/python.exe ml/train_warm_target_horizon_forecast.py --output data/backtests/oracle-warm-target-horizon-new
```

This is a forecast experiment. It does not add a joint stochastic path model, backtest a policy, or establish a new Lean optimality result.
