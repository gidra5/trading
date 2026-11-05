# Target-horizon-conditioned price distribution

Implemented and trained a shared endpoint distribution with a requested horizon in seconds. This first trial did not establish useful conditioning on market features: validation selected epoch 0, the training-fitted initial distribution, for both the feature-conditioned model and the horizon-only control. Existing best models remain unchanged.

## Horizon input and sampling

The new input is

\[
z(h)=\tanh(h/21600),\qquad h\in\{1,\ldots,86400\}\text{ seconds}.
\]

At two hours, the argument is 1/3 and the input is 0.321513, 3.55% below the linear approximation. The horizon coordinate is appended after feature normalization and is not standardized again.

Integer horizon probabilities integrate the derivative over one-second bins:

\[
\Pr(H=h)=\frac{\tanh(h/21600)-\tanh((h-1)/21600)}{\tanh(86400/21600)}.
\]

Implementation samples uniform `u` between zero and `tanh(4)`, then takes `ceil(21600 * atanh(u))`. Every integer horizon has positive sampling probability. Sampling is independent of origin, and new horizons are drawn each epoch.

The chosen linear region and derivative weighting imply about 32.2% of examples at or below two hours, 44.0% between two and six hours, and 23.8% beyond six hours. Only about 3.5% exceed twelve hours. They do **not** imply that most examples lie below two hours; those two requirements would require a different sampling rule or a smaller linear region.

The feature-conditioned run drew 3,318,176 training pairs over 16 epochs. It visited 73,549 distinct integer horizons. Sampling gives positive support to all seconds but does not guarantee every tail second appears in a finite run. Horizon 1 appeared 145 times; horizon 7,200 appeared 146 times; horizon 86,400 was not drawn. Long-tail predictions therefore include interpolation across nearby sampled horizons.

## Forecast and training contract

- Inputs: the existing frozen direct59 native model's two predicted feature states, flattened to 118 channels, plus the target-horizon coordinate. Native training ends before the head's first training origin. The old one-second price head is not reused or overwritten.
- Output: the existing Gaussian-CDF triangular knot distribution and a separate exact-zero atom. Magnitudes are not replaced with Gamma distributions or collapsed to conditional means.
- Target: `log(P[t+h]/P[t])`, computed directly from observed endpoint prices. Equal endpoint prices produce an exact zero even if the intervening path moved.
- Internal return coordinate: `log(P[t+h]/P[t]) / sqrt(h)`. This is an invertible numerical scaling. The entire law can change with horizon; independent increments or Brownian price dynamics are not assumed.
- Density conversion: for nonzero raw return `r`, `f_h(r) = f_scaled(r / sqrt(h)) / sqrt(h)`. The zero atom's probability is unchanged by scaling. Raw-return likelihood includes this Jacobian.
- Head: two width-128 GLU layers; knot basis and normalizers fitted on training data only.
- Loss: raw-return mixed-law NLL plus mean-prediction MSE in the scaled coordinate, divided by its training second moment. There is no activity or predicted-advantage reward.
- Training: Adam, learning rate 0.001, batch 4096, two independently sampled horizons per origin per epoch, maximum 32 epochs, patience 8 after at least 16 epochs. Both models stopped at epoch 16.
- Control: independently trained horizon-only head with the same output law, sampled targets, loss, and budget. It receives no market features.
- Calibration: none in this architecture screen. This does not claim to include the earlier online calibration methods.

The model estimates `P(log(P[t+h]/P[t]) | observed information at t, h)`. Different horizon marginals are not constrained to form a consistent joint path law. This extension alone does not supply `P(P[t+h+1] | hypothetical information at t+h)`, future feature transitions, or the path dependence needed for multi-step Bellman planning. No policy backtest or new Lean optimality claim is made here.

## Split protection and comparison

Training uses 103,693 origins from the cached native training tail; validation uses 14,760 origins; test uses 16,334 origins. Forecast cadence is four seconds. Every retained origin has a full 24-hour target inside its own available split. The same origins are used at every horizon, avoiding horizon-dependent sample composition and avoiding a sampling bias from truncating late-origin horizons.

Validation selection uses two fixed random horizon draws per origin. All of this validation subset is used for checkpoint selection; it is not a reserved second-stage mixture test. Selection and checkpoint hashes are frozen before this script opens test forecasts or prices. The historical test period has been inspected by earlier experiments and is not a new untouched holdout.

Test origins span approximately 18.15 hours, with targets extending a further day. These overlapping long-horizon examples are strongly dependent; thousands of origins are not thousands of independent day-long outcomes. The common-origin purge also makes the one-second comparison smaller than the earlier 36,134-origin evaluation. Old-model figures below are rescored on exactly these 16,334 origins.

## Outcome

Both models selected epoch 0, so their selected forecasts coincide. At that point, the network ignores market features and the horizon input; raw distributions still scale with `sqrt(h)` through the return coordinate. The positive long-horizon MSE scores below are scores of this training-fitted baseline, **not evidence that the new network learned a useful long-horizon signal**.

| Target horizon | Selected new model log-return MSE skill | Horizon-only control skill |
|---|---:|---:|
| 1 second | -2.2421% | -2.2421% |
| 5 minutes | 0.3971% | 0.3971% |
| 15 minutes | 2.1638% | 2.1638% |
| 1 hour | 6.8088% | 6.8088% |
| 2 hours | 7.9496% | 7.9496% |
| 6 hours | 9.9065% | 9.9065% |
| 24 hours | 20.5386% | 20.5386% |

The frozen direct59 one-second head scores **4.0187%** on the matched one-second sample, versus **-2.2421%** for this shared model. Its mixed-law NLL is -6.19394, versus -0.24633 for the new selected baseline. Broad derivative sampling allocates only about 0.00463% of draws to each short second, and the initial zero-atom frequency is fitted over the broad horizon distribution. Short-horizon quality is therefore not preserved by this experiment.

An exact replay of the first feature-conditioned training epoch found:

| Combined loss; lower is better | Before training | After epoch 1 |
|---|---:|---:|
| Training pairs | -3.073757 | -3.937341 |
| Validation pairs | -0.907798 | 0.522824 |

Optimization reduced training loss while worsening validation loss. That supports a generalization failure in this trial rather than an optimizer that never learned. It does not isolate whether the main cause is regime shift, limited effective long-horizon data, representation, or regularization. The diagnostic epoch-1 trained network is retained separately and is not promoted over the validation choice.

## Verification and artifacts

Seven unit tests passed: derivative-weighted sampling, split purging, exact endpoint/zero labels, invalid horizons, density Jacobian, normalized mixed laws and causal scoring, gradients and checkpoint reload.

The separate audit verified training-source and selected-checkpoint hashes, reproduced epoch-1 validation loss, and reconstructed the 118 cached native features from actual observed states. On 2,560 origins, both native features and the diagnostic trained network's logits matched exactly. Validation selection remained unchanged throughout the audit.

Final means use float64 768-point quadrature checked against 384-point quadrature. The largest reported difference is below `5e-15` log return. Output laws remain available through knot weights, exact-zero probabilities, and horizon-scaled return coordinates; the saved evaluation NPZ summaries are not themselves complete-law exports.

Source:

- `ml/target_horizon_forecast.py`: horizon contract, sampler, full distribution head, and causal observed-state inference adapter.
- `ml/train_target_horizon_forecast.py`: training, frozen selection, matched evaluation.
- `ml/audit_target_horizon_forecast.py`: first-epoch replay and causal feature audit.
- `ml/test_target_horizon_forecast.py`: tests.

Artifacts in `data/backtests/oracle-target-horizon/`:

- `featuresAndHorizon.pt`, `horizonOnly.pt`: validation-selected checkpoints.
- `diagnostic-trained-epoch1.pt`: trained model retained for inspection, not selected.
- `plan.json`, `selection.json`, `evaluation.json`, `audit.json`.
- Training histories, sampled horizon counts, fixed validation pairs, and per-horizon test predictions.

Commands from the repository root:

```powershell
.venv-ml/Scripts/python.exe -m unittest discover -s ml -p test_target_horizon_forecast.py
.venv-ml/Scripts/python.exe ml/train_target_horizon_forecast.py --output data/backtests/oracle-target-horizon-new
```

For inference, `load_observed_target_horizon_forecast(checkpoint, device)` returns an adapter accepting two observed production59 states and either one integer horizon or one integer horizon per batch row. Horizons outside 1–86,400 seconds are rejected rather than silently clipped or extrapolated.
