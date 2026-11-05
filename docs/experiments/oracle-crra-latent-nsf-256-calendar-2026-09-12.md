# Latent NSF with 256 dimensions and calendar time

This experiment implements the requested larger latent state, multiple horizon features, advancing calendar inputs, and uniform horizon sampling. It consumes the 59 observed production features and observed prices directly. No direct59 predicted feature summary is passed to the encoder, transition, or decoder.

## Architecture and time contract

The causal GRU has 256 hidden units and emits a 256-dimensional diagonal Gaussian latent posterior. The transition has four affine coupling layers with width-256 conditioners; the observation decoder has two width-256 hidden layers. The forecasting model has 1,741,432 parameters, excluding its training-only bridge. All parameters are trained from initialization; this is not a warm start from a previous predictor.

For horizon `h` in seconds, use physical duration `d = h / 21600`. The encoder, transition, and bridge receive neural horizon features `[d, tanh(d), log1p(h)]`. Drift, diffusion, and coupling multipliers still use `d` and `sqrt(d)`. Tanh is not substituted for physical time.

Calendar features are sine/cosine pairs with UTC daily and weekly periods. The encoder receives the observation's calendar features. The transition receives start and endpoint clock features. The bridge receives start, intermediate, and endpoint clock features. Composed transitions advance their start time by the elapsed duration; they do not freeze the forecast-origin clock. These are periodic calendar inputs, not an unrestricted secular timestamp trend.

The public `distribution` API requires float64 Unix timestamps in seconds for each observed row. It checks their differences against elapsed seconds. Phase reduction uses float64 before casting clock features to network precision; float32 Unix epochs would lose one-second resolution. Internally, transition start times are Unix seconds divided by 21600, so the common flow-loss composition `start + duration` remains valid.

The observed history is still two completed rows one second apart. A training window contains `[-1, 0, floor(h/2), h]` relative to the origin. Future observations infer target posteriors during training; they never enter the forecast-origin encoder state. Gaussian decoder uncertainty and uncertainty over latent states are both retained in the forecast mixture. Zero-horizon price is the known current price. Arbitrary fractional horizons from 0 to 86400 seconds are accepted, although supervised targets are on integer seconds.

## Training and selection

Horizon targets are uniform over integers 1 through 86400 inclusive. Every integer has probability 1/86400; the tanh derivative no longer affects sampling. Short horizons consequently receive no special sampling preference. Validation also uses a fixed uniform sample of 2048 horizons; its aggregate score must not be compared directly with the older tanh-weighted validation score.

Training uses the previous matched pools of 103,693 training origins and 14,760 validation origins, each with complete 24-hour targets inside its own split. Archived native data provide only origin alignment and dataset provenance; their forecast values do not enter this model. Normalization uses training observations only.

Each epoch draws 32,768 windows, batch size 1024. AdamW learning rate is 0.0003. Maximum epochs 32, minimum 16, patience 10. Both arms retain adjacent KL weight 0.1. The second arm adds skip KL weight 0.1 and flow-loss weight 0.01. KL terms sum over latent dimensions, so increasing state size also changes the aggregate regularization scale; this is not an isolated width ablation.

Selection minimizes mean forecast CRPS(log return)/sqrt(h), using 64 latent components and 128 decoder draws. Checkpoints are selected on validation and frozen before test evaluation. Test reporting uses the same fixed horizons and strided origins as previous experiments, 256 latent components and 512 decoder draws. It separately reports the decoder-integrated mean and a larger-component integration check.

No online calibration or trading-policy backtest is part of this architecture experiment. The Gaussian observation law still has no explicit zero-move atom; tick rounding induces discrete probabilities. Calendar conditioning and a larger state do not establish a sufficient state, exact flow consistency, or optimality for the real market distribution.

## Results

Both arms selected trained checkpoints. The initial uniform-validation scaled CRPS was 0.390682 bps. Adjacent-only selected epoch 8 at 0.342006 (12.46% reduction); skip+flow selected epoch 1 at 0.365468 (6.45% reduction). They stopped after 18 and 16 epochs respectively, with 589,824 and 524,288 sampled windows. Total training took 199 seconds.

Test MSE skill uses the decoder-integrated conditional mean log return, relative to predicting zero log return. CRPS is in log-return bps; lower is better. These are descriptive scores on the previously inspected historical test, with 4517 H1 origins and 2042 complete-24-hour origins at every longer horizon. Long-horizon targets overlap heavily.

| Horizon | Adjacent MSE skill % | Skip+flow MSE skill % | Adjacent CRPS | Skip+flow CRPS | Previous 8D initialization CRPS |
|---|---:|---:|---:|---:|---:|
| 1 second | -1.7250 | -12.9236 | 0.1530 | 0.1705 | 0.1609 |
| 5 minutes | 0.2361 | 0.1913 | 4.7573 | 4.7642 | 4.7672 |
| 15 minutes | 1.1481 | 0.2262 | 8.0147 | 8.0414 | 8.0451 |
| 2 hours | 9.4077 | -0.7415 | 22.5384 | 23.5034 | 23.4053 |
| 6 hours | 5.1331 | -3.0400 | 48.6641 | 51.0439 | 50.4659 |
| 24 hours | -983.8092 | -53.8889 | 371.2803 | 102.8405 | 83.2846 |

The adjacent-only arm learns useful conditional means at 5 minutes through 6 hours in this sample; the previous selected 8D model had zero integrated-mean skill at all horizons. However, H1 is worse than the zero-return mean, and the adjacent-only 24-hour forecast fails severely. The skip+flow arm reduces that 24-hour failure but does not improve the intermediate-horizon mean scores over adjacent-only. The previous direct59 predictor remains substantially stronger at H1 (5.19% skill on these matched strided origins). The earlier base17 joint horizon model had 10.78% skill at 2 hours and 39.27% at 24 hours on the matched sample.

The new adjacent-only two-hour CRPS of 22.5384 compares with 22.7742 for the earlier base17 warm horizon model. This small difference is not a significance result. Changing latent size, network width, calendar conditioning, time features, and sampling together prevents attributing gains or failures to any single change. No existing best predictor is replaced.

The independent 512-component check on 128 origins produced mean differences below 0.056 bps at every horizon for both arms. The severe 24-hour mean error is therefore not explained by the measured integration discrepancy. This finite check is not a uniform numerical error certificate.

## Verification

Eight latent unit tests passed, plus the seven existing observed NSF tests and a full 256-dimensional GPU forward/backward smoke test with finite gradients. Tests cover causal encoding, duplicate times, zero-duration identity, inversion/Jacobians, Brownian bridge consistency, advancing the clock in composition, distinguishable one-second calendar phases, weekly periodicity, calendar-dependent transition parameters, mixed decoder uncertainty, arbitrary horizons, and rejection of low-precision/inconsistent timestamp histories.

The selected-checkpoint audit verifies source/checkpoint hashes and confirms that changing future teacher observations leaves the current posterior exactly unchanged. At a 1234.5-second query, both current/transition latent uncertainty and decoder conditional uncertainty are retained. Shifting all observed timestamps by six hours while retaining features and prices changes the predictive mean: RMS differences were 0.01263 bps for adjacent-only and 0.01777 bps for skip+flow on 32 validation origins. This verifies that learned clock inputs affect forecasts; it does not establish that the effect improves accuracy.

Sampling replay confirmed roughly equal counts in the four six-hour horizon bins. Each arm happened to sample exactly seven one-second examples over its complete training run. Equal per-second sampling gives very sparse supervision at any particular short horizon; the model relies on shared parameters to generalize across horizons.

## Reproduction

```powershell
.venv-ml/Scripts/python.exe -m unittest discover -s ml -p test_latent_neural_stochastic_flow.py
.venv-ml/Scripts/python.exe ml/train_latent_neural_stochastic_flow.py
.venv-ml/Scripts/python.exe ml/evaluate_latent_neural_stochastic_flow.py
.venv-ml/Scripts/python.exe ml/audit_latent_neural_stochastic_flow.py
```

Training refuses to overwrite an existing experiment. Outputs are in `data/backtests/oracle-latent-nsf-256-calendar`. Original eight-dimensional experiment sources were archived in `data/backtests/oracle-latent-nsf/source` before updating the active implementation; its old checkpoints and reports remain historical artifacts.
