# Three consecutive BTCUSDT log returns across scales and windows

Generated 2026-08-17T13:56:45.785Z. Exact common history: **2021-08-17T00:00:00.000Z to 2026-08-17T00:00:00.000Z** (exclusive end).

## Result

- Signed dependence remains small beyond one step: lag-2 return correlations range from -0.0124 to 0.0289. Magnitude dependence persists, with lag-2 absolute-return correlations from 0.089 to 0.350.
- Three adjacent $>2\sigma$ magnitudes occur 7.3–49.4 times more often than independent marginals predict. The main three-step structure is a persistent common volatility state, not directional continuation.
- AIC winners by scale are Trivariate generalized t at 1s, 1m, 15m, 1h, 4h, 1d.
- At 1d the closest tested family is **Trivariate generalized t** with $p=0.924$; Trivariate generalized Gaussian follows at $\Delta$AIC 0.83.
- At 1s, 9.91% of triples are exactly $(0,0,0)$ and 64.85% touch at least one zero plane. Continuous fits exclude those singular components.

## Full-history triple statistics

| Scale | Triples | Corr lag 1 | Corr lag 2 | Magnitude corr lag 1 | Magnitude corr lag 2 | All-3 >2σ lift | (0,0,0) % | GGD p | AIC winner |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1s | 157,766,398 | 0.0468 | 0.0289 | 0.365 | 0.332 | 38.0 | 9.91 | 0.150 | Trivariate generalized t |
| 1m | 2,629,438 | 0.0044 | -0.0093 | 0.390 | 0.350 | 49.4 | 0.16 | 0.515 | Trivariate generalized t |
| 15m | 175,294 | -0.0022 | -0.0124 | 0.321 | 0.278 | 33.4 | 0.01 | 0.540 | Trivariate generalized t |
| 1h | 43,822 | -0.0076 | -0.0059 | 0.275 | 0.206 | 18.6 | 0.00 | 0.521 | Trivariate generalized t |
| 4h | 10,954 | -0.0011 | 0.0055 | 0.195 | 0.132 | 9.8 | 0.00 | 0.551 | Trivariate generalized t |
| 1d | 1,824 | -0.0209 | 0.0042 | 0.167 | 0.089 | 7.3 | 0.00 | 0.739 | Trivariate generalized t |

`All-3 >2σ lift` is the observed probability that all three standardized magnitudes exceed two, divided by the product of the three marginal exceedance probabilities.

## Closest continuous distribution

For covariance-whitened $z\in\mathbb{R}^3$, the trivariate generalized Gaussian is

$$
f(z)=\frac{p}{4\pi s^3\Gamma(3/p)}
\exp\!\left[-\left(\frac{\lVert z\rVert_2}{s}\right)^p\right],
$$

and the generalized t is

$$
f(z)=\frac{p}{4\pi s^3B(3/p,q-3/p)}
\left[1+\left(\frac{\lVert z\rVert_2}{s}\right)^p\right]^{-q}.
$$

For the radial-lognormal candidate, $\log\rho\sim\mathcal N(m,\tau^2)$ for $\rho=\lVert z\rVert_2$, so

$$
f(z)=\frac{1}{4\pi\sqrt{2\pi}\tau\rho^3}
\exp\!\left[-\frac{(\log\rho-m)^2}{2\tau^2}\right].
$$

| Scale | Candidate | p | q | ν | mean log ρ | sd log ρ | NLL / triple | ΔAIC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1s | Trivariate generalized t | 0.150 | 86.876 | n/a | n/a | n/a | -24.78778 | 0.00 |
| 1s | Trivariate generalized Gaussian | 0.150 | n/a | n/a | n/a | n/a | -24.77685 | 653.8 |
| 1s | Trivariate radial lognormal | n/a | n/a | n/a | -0.841 | 2.024 | -24.75939 | 1701 |
| 1s | Trivariate Student t | 2.000 | 2.500 | 2.00 | n/a | n/a | -23.94702 | 50443 |
| 1s | Trivariate Gaussian | n/a | n/a | n/a | n/a | n/a | -22.69866 | 125343 |
| 1m | Trivariate generalized t | 0.919 | 9.033 | n/a | n/a | n/a | -18.09595 | 0.00 |
| 1m | Trivariate generalized Gaussian | 0.515 | n/a | n/a | n/a | n/a | -18.08949 | 385.6 |
| 1m | Trivariate Student t | 2.000 | 2.600 | 2.20 | n/a | n/a | -18.07911 | 1008 |
| 1m | Trivariate radial lognormal | n/a | n/a | n/a | -0.103 | 0.848 | -18.06080 | 2107 |
| 1m | Trivariate Gaussian | n/a | n/a | n/a | n/a | n/a | -17.21819 | 52662 |
| 15m | Trivariate generalized t | 1.246 | 5.225 | n/a | n/a | n/a | -14.12568 | 0.00 |
| 15m | Trivariate radial lognormal | n/a | n/a | n/a | -0.062 | 0.776 | -14.12290 | 164.5 |
| 15m | Trivariate Student t | 2.000 | 2.629 | 2.26 | n/a | n/a | -14.11885 | 407.6 |
| 15m | Trivariate generalized Gaussian | 0.540 | n/a | n/a | n/a | n/a | -14.11082 | 889.3 |
| 15m | Trivariate Gaussian | n/a | n/a | n/a | n/a | n/a | -13.36020 | 45925 |
| 1h | Trivariate generalized t | 0.992 | 7.550 | n/a | n/a | n/a | -12.03855 | 0.00 |
| 1h | Trivariate radial lognormal | n/a | n/a | n/a | -0.056 | 0.803 | -12.03612 | 144.0 |
| 1h | Trivariate generalized Gaussian | 0.521 | n/a | n/a | n/a | n/a | -12.03155 | 417.7 |
| 1h | Trivariate Student t | 2.000 | 2.554 | 2.11 | n/a | n/a | -12.02496 | 813.6 |
| 1h | Trivariate Gaussian | n/a | n/a | n/a | n/a | n/a | -11.34262 | 41752 |
| 4h | Trivariate generalized t | 0.635 | 40.252 | n/a | n/a | n/a | -9.83725 | 0.00 |
| 4h | Trivariate generalized Gaussian | 0.551 | n/a | n/a | n/a | n/a | -9.83700 | 3.46 |
| 4h | Trivariate radial lognormal | n/a | n/a | n/a | 0.002 | 0.806 | -9.82678 | 227.3 |
| 4h | Trivariate Student t | 2.000 | 2.607 | 2.21 | n/a | n/a | -9.80686 | 663.7 |
| 4h | Trivariate Gaussian | n/a | n/a | n/a | n/a | n/a | -9.31135 | 11517 |
| 1d | Trivariate generalized t | 0.924 | 19.677 | n/a | n/a | n/a | -6.83332 | 0.00 |
| 1d | Trivariate generalized Gaussian | 0.739 | n/a | n/a | n/a | n/a | -6.83255 | 0.83 |
| 1d | Trivariate Student t | 2.000 | 3.159 | 3.32 | n/a | n/a | -6.81992 | 46.9 |
| 1d | Trivariate radial lognormal | n/a | n/a | n/a | 0.128 | 0.710 | -6.81302 | 72.1 |
| 1d | Trivariate Gaussian | n/a | n/a | n/a | n/a | n/a | -6.54929 | 1032 |

## Similarity across scales

The comparison averages Jensen–Shannon divergence across all three standardized pairwise projections: $(r_t,r_{t+1})$, $(r_{t+1},r_{t+2})$, and $(r_t,r_{t+2})$.

| Adjacent scales | Mean projection JS bits |
| --- | --- |
| 1s → 1m | 0.0852 |
| 1m → 15m | 0.0035 |
| 15m → 1h | 0.0041 |
| 1h → 4h | 0.0153 |
| 4h → 1d | 0.0767 |

## Stability across one-year epochs

| Scale | Lag-1 magnitude range | Lag-2 magnitude range | GGD p range | Median JS to 5y | Max JS to 5y |
| --- | --- | --- | --- | --- | --- |
| 1s | 0.307–0.403 | 0.265–0.378 | 0.150–0.279 | 0.0803 | 0.2340 |
| 1m | 0.351–0.415 | 0.312–0.361 | 0.449–0.624 | 0.0042 | 0.0045 |
| 15m | 0.253–0.352 | 0.223–0.284 | 0.471–0.662 | 0.0071 | 0.0088 |
| 1h | 0.226–0.286 | 0.158–0.202 | 0.452–0.668 | 0.0166 | 0.0208 |
| 4h | 0.115–0.252 | 0.058–0.147 | 0.456–0.718 | 0.0581 | 0.0632 |
| 1d | 0.027–0.299 | -0.015–0.147 | 0.541–1.072 | 0.2356 | 0.2644 |

At 4h and 1d, annual projection distances have a large finite-sample noise floor. Daily model selection uses only about 1,824 triples, annual daily estimates use about 363, and the daily all-three $>2\sigma$ lift is based on only 2 observed triples.

## Interpretation

- The three-dimensional law strengthens the pairwise conclusion: a shared radial distribution is much closer than a Gaussian, while direct signed correlations remain close to zero.
- The generalized t captures a sharp center and polynomial joint tails. A generalized Gaussian becomes competitive only when aggregation has softened the power-law component.
- The 1s fit remains dominated by tick/no-trade microstructure and reaches the allowed lower generalized-Gaussian power bound; its power estimate is not a literal continuous-law parameter.
- Close-to-close triples are descriptive and omit intrabar extremes, spread, fees, slippage, and liquidation paths.

## Reproduction

Run `npm run analysis:return-triples`, then `npm run analysis:return-triples:render`. Machine-readable results are in `data/benchmarks/consecutive-log-return-triples.json`.
