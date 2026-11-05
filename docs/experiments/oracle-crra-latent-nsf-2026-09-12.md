# Latent NSF: from a latent transition to a price distribution

The latent version supports the requested interface: infer a current latent **distribution**, advance it to an arbitrary target horizon with NSF, and integrate the observation decoder. Implemented this as a causal variational model and tested a first training run. It did not improve validation forecasting: both arms selected epoch 0. Existing predictors remain unchanged.

## Predictive distribution

The paper's latent construction combines a recurrent variational encoder, NSF latent transitions and a probabilistic observation decoder. Its training objective includes observation reconstruction, posterior-to-transition KL terms, skip-ahead KL and flow consistency regularization. See [Neural Stochastic Flows, section 4](https://proceedings.neurips.cc/paper_files/paper/2025/file/18abbeef8cfe9203fdf9053c9c4fe191-Paper-Conference.pdf).

For observations available now, `I_t`, the model-based price forecast is approximated by

\[
\widehat p(P_{t+h}\mid I_t)
=\iint p_\psi(P_{t+h}\mid z')\,
K_\theta^h(dz'\mid z)\,q_\phi(dz\mid I_t).
\]

Here `q` approximates the filtering posterior, `K` is the learned latent NSF transition, and the decoder's price marginal is `p`. This is a model-based approximation to the market conditional law, not an exact recovery guarantee.

Sampling current states `z_m ~ q`, then endpoint states `z'_m ~ K_h(. | z_m)`, produces

\[
\widehat p_M(P_{t+h}\mid I_t)
=\frac1M\sum_{m=1}^{M}p_\psi(P_{t+h}\mid z'_m).
\]

The price decoder in this experiment is Gaussian in log return, so this is a finite mixture of lognormal price distributions. Decoder noise remains inside every component. It is not equivalent to propagating a single latent mean and returning one decoded price.

Two distinct approximations remain: learned model/variational posterior error, and finite integration over latent states. For a fixed finite mixture, its CDF and log-return density are evaluated analytically, including decoder uncertainty. The finite mixture approximates the original latent integral.

## Implemented model

- Observations: the 59 raw production features and log price relative to the known forecast-origin price, scaled by 0.01. Feature normalization is fitted only on training origins.
- Current history: two completed observations one second apart, matching the raw history supplied to the previously frozen native models. The native forecast weights are not reused; the latent encoder learns from observed features directly.
- Encoder: causal GRU with 64 hidden units, producing an eight-dimensional diagonal Gaussian posterior at each observation.
- Initialization: a learnable price skip connection anchors one latent coordinate and the price decoder to observed log price; the remaining seven coordinates are learned. All latent posterior coordinates remain stochastic. The skip connections are initialization aids, not fixed observed-state constraints.
- Transition: eight-dimensional NSF, four alternating affine coupling layers, two width-64 SiLU layers per conditioner. Drift/diffusion and coupling effects use physical elapsed time divided by 21,600; the neural time coordinate uses tanh. Scale and shift networks are bounded, as in the preceding local NSF implementation.
- Decoder: a diagonal Gaussian observation law for price and all 59 normalized features. Its means and scales depend on the latent state. Price uncertainty includes both variation between decoded latent states and the decoder's conditional variance.
- Bridge: an eight-dimensional conditional flow with Brownian-interpolation base, used for the bidirectional flow loss.
- No fixed origin feature context is passed into future latent transitions. The inferred latent state carries the learned information forward.

Inference uses only the observed prefix. Future observations are used during training to infer target posteriors for reconstruction and KL matching; they are not supplied to the current forecast. Price coordinates are anchored to the known forecast-origin price throughout each training window. This is a forecast-window construction, not a separately proved online filtering implementation invariant to every choice of window origin.

The API accepts any real horizon from 0 to 86,400 seconds. Training observations lie on integer seconds; fractional-horizon queries interpolate the learned time-dependent model. A zero-horizon query returns the known current price exactly. The latent transition also satisfies its own zero-duration identity.

## Training protocol

The same complete-24-hour origin pools as the prior experiment are used: 103,693 training origins and 14,760 validation origins. Each training window contains observations at `[-1, 0, floor(h/2), h]` relative to the forecast origin. When `h=1`, the duplicate midpoint at time zero does not update the GRU or sample a new latent state, and its zero-duration KL is omitted.

Integer horizons are sampled with probabilities proportional to `tanh(h/21600) - tanh((h-1)/21600)`, truncated at 24 hours. Each epoch draws 32,768 windows. All endpoints stay within their own data split.

Two arms:

| Arm | Adjacent KL weight | Skip KL weight | Flow weight |
|---|---:|---:|---:|
| Latent adjacent | 0.1 | 0 | 0 |
| Latent skip + flow | 0.1 | 0.1 | 0.01 |

The objective combines time-averaged observation NLL, time-averaged adjacent KL (including the initial standard-normal prior), the direct current-to-endpoint skip KL and, in the second arm, both bridge flow bounds. One inner bridge update precedes each main update. This uses the paper's objective components with local weighting/time-averaging choices; it is not a reproduction of its benchmark configuration.

AdamW learning rate is 0.0003, batch size 1,024, maximum 32 epochs, patience 10 after at least 16 epochs. Both arms stopped at epoch 16, each having seen 524,288 windows. Total training time was about 156 seconds on the local RTX 3070.

Checkpoint selection uses a **forecast** score rather than reconstruction: mean CRPS of log return divided by sqrt(h), on 2,048 fixed validation origin/horizon pairs. The forecast is generated from current observations only, with 64 latent components and 128 observation draws. All selections freeze before test evaluation. This selection objective differs from the joint-state NLL used in the preceding observed-state NSF experiment.

No online calibration, explicit zero atom, trading objective or policy backtest was added. Gaussian price observations can be rounded to 0.01 ticks to induce discrete price probabilities. This may still fail to represent the large one-second zero-move mass accurately.

## Results and interpretation

Both arms selected their initial checkpoint, with validation scaled CRPS **0.412662 bps**. At epoch 16:

| Arm | Initial validation scaled CRPS | Epoch-16 validation scaled CRPS | Epoch-16 reconstruction NLL |
|---|---:|---:|---:|
| Latent adjacent | 0.412662 | 0.426975 | 33.4910 |
| Latent skip + flow | 0.412662 | 0.428614 | 33.8568 |

Reconstruction improved substantially during training, but validation forecasting did not. Reconstructing observations and matching latent posteriors did not produce a better price forecast in this first configuration.

At initialization, the posterior and transition do not yet exploit the market feature content. The selected model has a zero conditional mean log return and a Gaussian-like uncertainty profile. Its decoder-integrated MSE skill is therefore **0% at every reported horizon**, and the two selected arms have identical predictive laws. Tiny nonzero skills in sample-only reports are integration/rounding artifacts, not learned price information. Do not interpret these selected checkpoint scores as the performance of a successfully learned latent representation.

Matched test CRPS in log-return bps, lower is better:

| Forecast | 1 second | 5 minutes | 15 minutes | 2 hours |
|---|---:|---:|---:|---:|
| Selected latent initialization | 0.1609 | 4.7672 | 8.0451 | 23.4053 |
| Earlier trained observed-state NSF with flow loss | 0.1471 | 4.6492 | 7.8367 | 23.8053 |
| Existing direct59 H1 / untrained sqrt(h) extension | 0.0705 | 5.9405 | 10.2443 | 29.9249 |
| Earlier base17 joint horizon fine-tune | 0.1291 | 4.7552 | 7.9556 | 22.7742 |

These use the same strided test origins as the previous NSF comparison: 4,517 at H1 and 2,042 at longer horizons. The test was inspected in earlier experiments, and long-horizon outcomes overlap heavily. The table is descriptive; no significance or overall superiority claim is made. The small two-hour CRPS advantage over observed-state NSF belongs to the untrained initialization, not learned latent dynamics.

The approximation machinery works, but this trial does not establish that the latent architecture improves forecasting. No previous best model was promoted, overwritten or replaced.

## Verification

Five unit tests passed, covering causal encoding, duplicate-time handling, finite variational gradients, latent flow inversion and exact Brownian-bridge consistency, integration of decoder uncertainty, and arbitrary/zero-horizon queries.

A separate audit reproduced the first adjacent-training epoch exactly, including validation score **0.422483 bps**, and saved it as `diagnostic-trained-epoch1.pt`. It is retained for inspection and is not the validation-selected model. On this genuinely updated encoder, changing all future teacher observations left current posterior means and variances exactly unchanged.

The trained diagnostic's 1,234.5-second query returned 256 Gaussian price components per example. Mean variance between component log-return means was approximately `2.690e-6`; mean conditional decoder variance was `1.041e-10`. Both sources were retained. The current posterior and latent transition both contribute to variation between component means.

Training-source and selected-checkpoint hashes were verified. Test integration uses 256 latent components and 512 observation draws, with a second 512-component integration on a subset. Selection and existing parent checkpoint hashes remain unchanged.

## Usage

```python
from pathlib import Path
import torch
from latent_neural_stochastic_flow import load_latent_nsf

model, metadata = load_latent_nsf(
    Path('data/backtests/oracle-latent-nsf/latentSkipFlow.pt'),
    torch.device('cuda'),
)
law = model.distribution(
    observed_features,  # [batch, history, 59]
    observed_prices,    # [batch, history], positive prices
    elapsed_seconds,    # [batch, history], e.g. [0, 1] for the trained prefix
    horizon_seconds=1234.5,
    components=256,
)
prices = law.sample_prices(samples=1024)
cdf_at_current = law.cdf(observed_prices[:, -1])
unchanged_probability = law.tick_probability(observed_prices[:, -1])
mean_log_return = law.mean_log_return()
mean_price = law.mean_price()
```

The returned mixture also exposes log-return density. CDF and tick-probability queries in this implementation expect positive price thresholds. Observation sampling and tick rounding are optional numerical representations of the mixture; the decoder distribution itself is not reduced to sampled prices.

Source files:

- `ml/latent_neural_stochastic_flow.py`
- `ml/train_latent_neural_stochastic_flow.py`
- `ml/evaluate_latent_neural_stochastic_flow.py`
- `ml/audit_latent_neural_stochastic_flow.py`
- `ml/test_latent_neural_stochastic_flow.py`

Artifacts are in `data/backtests/oracle-latent-nsf/`: selected checkpoints, diagnostic trained checkpoint, plan, selection, histories, marginal evaluations and audit.

```powershell
.venv-ml/Scripts/python.exe -m unittest discover -s ml -p test_latent_neural_stochastic_flow.py
.venv-ml/Scripts/python.exe ml/train_latent_neural_stochastic_flow.py --output data/backtests/oracle-latent-nsf-new
```

No exact market-law, filtering, Chapman–Kolmogorov or policy-optimality guarantee follows from this experiment.
