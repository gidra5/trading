# Neural Stochastic Flow with arbitrary target horizons

Implemented and trained an NSF price-distribution experiment. The model accepts **any real-valued horizon from 0 to 86,400 seconds**, including horizons absent from the evaluation table, and returns a price marginal directly. A 1,234.5-second query was verified through the observed-state inference adapter.

The experiment does not replace the current predictor. Its one-second accuracy is substantially worse, and its longer-horizon improvements are mixed.

## Paper and implementation

Source: [Kiyohara, Johns and Li, Neural Stochastic Flows, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/18abbeef8cfe9203fdf9053c9c4fe191-Paper-Conference.pdf); [authors' implementation](https://github.com/nkiyohara/jax-nsf).

This is a PyTorch implementation of the paper's equations, not a run of the authors' JAX code. It uses a Gaussian base with drift proportional to elapsed time and diffusion proportional to its square root, followed by four alternating affine coupling layers. A conditional bridge estimates both variational KL bounds for direct-versus-two-step consistency. Each main update in the regularized arm is preceded by one bridge update.

The zero-duration transition is exactly identity. The flow loss encourages Chapman–Kolmogorov consistency; it does not enforce or certify exact equality. This first experiment uses an observed two-dimensional state rather than the paper's latent encoder/decoder extension.

## Market adaptation

State:

1. Log-price displacement divided by 0.01.
2. Log of causal one-second return volatility relative to 0.0001. Volatility is an EWMA standard deviation with 300-second half-life and a 0.000001 floor.

The network also receives the existing direct59 model's 118 predicted native feature coordinates, computed entirely from observations available at the forecast origin. Native weights remain frozen. This feature context remains fixed when composing hypothetical transitions; sampled price and volatility evolve. That is an explicit model assumption, not a claim that these two state variables suffice for the real market.

We enforce translation equivariance in log price. Transitions are autonomous conditional on the fixed context. Network inputs use `tanh(h/21600)`; physical drift, noise and coupling factors use the additive duration `h/21600`, **not** the tanh value. Thus splitting a duration does not mistakenly add normalized tanh coordinates.

Each Gaussian/coupling conditioner has two width-64 SiLU layers. Scale and shift outputs are bounded for numerical stability. Density evaluation includes both the Gaussian base's scale Jacobian and all coupling Jacobians.

The flow defines a continuous joint state distribution. Its price marginal is the projection of that complete distribution, not a collapsed mean or three-point approximation. Price samples can be rounded to the 0.01 tick grid, which induces point masses, including an unchanged-price probability. Unlike the old head, this first NSF has **no separately learned zero atom**. Density fitting treats observed state pairs as continuous; this is a modeling limitation for tick-level data.

The previous knot heads cannot be copied exactly into this different flow architecture. Their checkpoints were retained unchanged and evaluated as controls. Unlike the preceding warm horizon-weight adapter, NSF initialization does not reproduce the old H1 distribution.

## Training and evaluation

- Training pool: 103,693 origins; validation pool: 14,760 origins. Every retained origin has complete 24-hour targets inside its split. Native-model training precedes these head-training origins.
- Supervised targets: actual endpoint log prices and causal endpoint volatility states. No later forecast cache is used as a current input.
- Horizons: integer training draws weighted by the integrated derivative of `tanh(h/21600)`, between 1 second and 24 hours. Each epoch samples 65,536 origin/horizon pairs. The function supports fractional-second queries, but those are interpolation queries without fractional-second supervised labels.
- Optimizer: AdamW, learning rate 0.0003, batch 2048. Maximum 32 epochs; patience 10 after at least 16 epochs.
- Baseline NSF: joint-state NLL only.
- Regularized NSF: joint-state NLL plus 0.1 times the sum of both bridge KL bounds. Flow minibatches use 128 examples. The intermediate time is a uniform fraction between 0.1 and 0.9 of the complete duration. Generated intermediate states, not observed future features, are used in this regularizer.
- Selection: minimum joint-state NLL on a fixed, horizon-sampled validation set. Both arms selected epoch 6 and stopped after epoch 16. Each saw 1,048,576 supervised pairs. Training took about 98 seconds on the local RTX 3070.
- Validation joint NLL: initial 2.52906; selected unregularized 2.24688; selected regularized 2.25352. These are joint state scores, not comparable to the old price-only NLL.
- No new online calibration and no policy backtest.

All selections and source/checkpoint hashes were frozen before test evaluation. The historical test was used in earlier experiments and is not a newly untouched holdout.

Price evaluation uses every eighth cached origin: 4,517 origins for H1 and 2,042 common complete-24-hour origins for longer horizons. All comparison models use these same origin sets. These long-horizon targets overlap; effective independent sample counts are much smaller. This experiment also has a different training budget and objective from the earlier knot-head experiments.

Price marginals are evaluated using 512 deterministic scrambled Sobol integration draws, with antithetic Gaussian noise for NSF, and 0.01 price rounding for every model. MSE skill and CRPS therefore evaluate the same observable, but include numerical integration error. A second scramble with 1,024 draws checks NSF means on 256 origins per horizon. For the regularized model, RMS mean differences are about 0.00011 bps at 15 minutes, 0.00637 bps at two hours, and 0.787 bps at 24 hours. These checks are numerical diagnostics, not certified error bounds.

## Price results

MSE skill relative to a zero log-return forecast:

| Model | 1 second | 5 minutes | 15 minutes | 2 hours | 24 hours |
|---|---:|---:|---:|---:|---:|
| Existing direct59, with sqrt(h) extension beyond H1 | 5.19% | -1.79% | -4.37% | -3.27% | -3.03% |
| Earlier base17 joint horizon fine-tune | -10.14% | -1.97% | 0.84% | **10.78%** | **39.26%** |
| NSF without flow loss | 0.01% | 0.62% | 1.04% | -0.85% | -76.23% |
| NSF with flow loss | 0.01% | **0.63%** | **1.07%** | -0.85% | -69.13% |

The existing direct59 H1 score differs from its earlier 4.53% because this table uses a strided subset and numerical marginal integration. The checkpoint itself is unchanged. Beyond one second, its untrained sqrt(h) extension is a weak baseline; the table also includes the previously trained multi-horizon base17 model for that reason.

CRPS in log-return basis points; lower is better:

| Model | 1 second | 5 minutes | 15 minutes | 2 hours |
|---|---:|---:|---:|---:|
| Existing direct59 / sqrt(h) extension | **0.0705** | 5.9405 | 10.2443 | 29.9249 |
| Earlier base17 joint horizon fine-tune | 0.1291 | 4.7552 | 7.9556 | **22.7742** |
| NSF without flow loss | 0.1478 | 4.6503 | 7.8392 | 23.8315 |
| NSF with flow loss | 0.1471 | **4.6492** | **7.8367** | 23.8053 |

NSF improves the five- and fifteen-minute distribution scores in this screen, but the margins over the previously trained horizon model are small and are not established as statistically robust. It is inferior to that model at two hours and beyond. At one second, zero-move calibration is especially poor: zero-event Brier is about 0.5699, versus 0.2831 for the existing explicit-atom predictor. Tick rounding alone did not recover that capability.

## Consistency and verification

On 64 contexts, the mean price Wasserstein discrepancy between a direct draw and two half-duration transitions was:

| Total duration | NSF without flow loss | NSF with flow loss | Approximate direct-vs-direct sampling floor |
|---|---:|---:|---:|
| 5 minutes | 0.2423 bps | 0.2406 bps | 0.12 bps |
| 2 hours | 1.4048 bps | 1.3836 bps | 0.78 bps |
| 6 hours | 5.0954 bps | 4.8925 bps | 1.93 bps |

The regularizer made a modest improvement in this diagnostic. The discrepancies remain nonzero, and part of each estimate is integration noise. This is not a proof of exact path consistency.

Seven unit tests passed: zero-duration identity, coupling inversion/Jacobian, Gaussian density normalization terms, vanishing KL bounds for an exact Brownian bridge, finite gradients through both losses, price translation/time stationarity, and arbitrary-horizon price sampling/CDF/grid behavior.

The trained-checkpoint audit verified source and selection hashes, zero-duration identity, finite sample log densities, and inverse errors below 0.000001 in state coordinates. The small joint MMD diagnostic was recomputed in float64 after float32 cancellation produced spurious negative values. The pre-correction evaluation is retained; price scores and model selection were unchanged.

The observed-state adapter was checked on 2,560 validation origins at horizon 1,234.5 seconds. It reconstructed cached native features exactly and produced finite price samples of shape `[2560, 8]`.

## Interface and artifacts

```python
from pathlib import Path
import torch
from nsf_price_forecast import load_nsf_price_forecast

model = load_nsf_price_forecast(
    Path('data/backtests/oracle-neural-stochastic-flow/nsfFlowLoss.pt'),
    torch.device('cuda'),
)
law = model.distribution(
    observed_states,      # [batch, 2, 59], completed observed features
    current_price,        # [batch]
    ewma_volatility,      # [batch], causal 300s-half-life std of 1s log returns
    horizon_seconds=1234.5,
)
prices = law.sample_prices(samples=1024)  # [batch, samples], rounded to .01
probability_below_current = law.cdf(current_price)
```

The price marginal remains a full generative distribution. `cdf` is numerical integration, not a closed-form exact marginal CDF. The full two-dimensional transition has an analytic change-of-variables density. Horizons outside the trained 0–24-hour range are rejected. No neural SDE solver is invoked for a horizon query.

Source files:

- `ml/neural_stochastic_flow.py`: transition, bridge, KL bounds, price marginal API and checkpoint loader.
- `ml/nsf_price_forecast.py`: observed-state inference adapter.
- `ml/train_neural_stochastic_flow.py` and `ml/evaluate_neural_stochastic_flow.py`.
- `ml/audit_neural_stochastic_flow.py` and `ml/test_neural_stochastic_flow.py`.

Artifacts: `data/backtests/oracle-neural-stochastic-flow/`, including both selected checkpoints, frozen training/selection records, per-horizon forecasts, full evaluation, warm-model comparison, numerical audit and adapter audit.

```powershell
.venv-ml/Scripts/python.exe -m unittest discover -s ml -p test_neural_stochastic_flow.py
.venv-ml/Scripts/python.exe ml/train_neural_stochastic_flow.py --output data/backtests/oracle-neural-stochastic-flow-new
```

NSF capability is implemented, including arbitrary-horizon distribution queries. This first market adaptation is not an overall accuracy improvement and does not establish Bellman-policy optimality.
