# Consecutive BTCUSDT log-return pairs across scales and windows

Generated 2026-08-17T13:56:35.627Z. The exact common history is **2021-08-17T00:00:00.000Z to 2026-08-17T00:00:00.000Z** (exclusive end).

## Result

- Consecutive signed returns are nearly uncorrelated, but their magnitudes are not: full-history return correlations span -0.0209 to 0.0468, while absolute-return correlations span 0.167 to 0.390. The joint distribution therefore contains volatility-state dependence that two independent marginals miss.
- After removing exact-zero axes and standardizing each window, adjacent-scale 2D shape distances are 0.0040–0.1301 bits of Jensen–Shannon divergence. The shapes are related, but not scale-invariant.
- AIC winners by scale are Elliptical generalized t at 1s, 1m, 15m, 1h, 4h; Elliptical generalized Gaussian at 1d.
- At 1d the closest tested continuous family is **Elliptical generalized Gaussian** with power $p=0.811$; Elliptical generalized t follows at $\Delta$AIC 0.13.
- The 1s distribution is a mixed distribution, not a single continuous density: 18.19% of pairs are exactly $(0,0)$ and 54.12% lie on at least one zero axis. Model selection below applies only where both returns are nonzero.

## Full-history pair statistics

| Scale | Pairs | Corr r | Corr magnitude | Same sign % | Joint >2σ lift | (0,0) % | GGD p | AIC winner |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1s | 157,766,399 | 0.0468 | 0.365 | 44.03 | 4.85 | 18.19 | 0.150 | Elliptical generalized t |
| 1m | 2,629,439 | 0.0044 | 0.390 | 50.29 | 5.60 | 0.52 | 0.617 | Elliptical generalized t |
| 15m | 175,295 | -0.0021 | 0.321 | 47.68 | 4.96 | 0.01 | 0.661 | Elliptical generalized t |
| 1h | 43,823 | -0.0076 | 0.275 | 47.00 | 3.65 | 0.00 | 0.637 | Elliptical generalized t |
| 4h | 10,955 | -0.0012 | 0.195 | 45.90 | 2.48 | 0.00 | 0.645 | Elliptical generalized t |
| 1d | 1,825 | -0.0209 | 0.167 | 47.84 | 2.33 | 0.00 | 0.811 | Elliptical generalized Gaussian |

`Joint >2σ lift` is $P(|X|>2,|Y|>2)/[P(|X|>2)P(|Y|>2)]$. A value above one means large adjacent moves cluster more than independent marginals predict. `GGD p` is a common elliptical generalized-Gaussian shape fit used as a comparable shape index even where another family wins.

## Closest continuous distribution

The elliptical bivariate generalized Gaussian is

$$
f(z)=\frac{p}{2\pi s^2|R|^{1/2}\Gamma(2/p)}
\exp\!\left[-\left(\frac{\sqrt{z^\top R^{-1}z}}{s}\right)^p\right].
$$

The elliptical generalized t replaces the stretched-exponential kernel with

$$
f(z)=\frac{p}{2\pi s^2|R|^{1/2}B(2/p,q-2/p)}
\left[1+\left(\frac{\sqrt{z^\top R^{-1}z}}{s}\right)^p\right]^{-q}.
$$

For radial-lognormal $m$ and $\tau$, the covariance-whitened radius $\rho=\lVert z\rVert_2$ has $\log\rho\sim\mathcal N(m,\tau^2)$, giving

$$
f(z)=\frac{1}{2\pi\sqrt{2\pi}\tau\rho^2}
\exp\!\left[-\frac{(\log\rho-m)^2}{2\tau^2}\right].
$$

The product variants apply the analogous 1D density independently along the whitened common/difference axes. They test whether an axis-shaped density is closer than an elliptical common-radius density. AIC uses the same deterministic continuous-pair sample for all candidates within a scale; only differences within a scale are meaningful.

| Scale | Candidate | p | q | ν | mean log ρ | sd log ρ | NLL / pair | ΔAIC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1s | Elliptical generalized t | 0.150 | 35.292 | n/a | n/a | n/a | -17.31400 | 0.00 |
| 1s | Elliptical radial lognormal | n/a | n/a | n/a | -1.714 | 2.575 | -17.30769 | 377.0 |
| 1s | Elliptical generalized Gaussian | 0.150 | n/a | n/a | n/a | n/a | -17.27158 | 2543 |
| 1s | Product generalized Gaussian | 0.172 | n/a | n/a | n/a | n/a | -16.68671 | 37636 |
| 1s | Product generalized t | 0.199 | 40.795 | n/a | n/a | n/a | -16.66777 | 38774 |
| 1s | Student t | 2.000 | 2.000 | 2.00 | n/a | n/a | -16.17575 | 68293 |
| 1s | Gaussian | n/a | n/a | n/a | n/a | n/a | -15.26978 | 122649 |
| 1m | Elliptical generalized t | 0.994 | 7.218 | n/a | n/a | n/a | -12.05550 | 0.00 |
| 1m | Elliptical generalized Gaussian | 0.617 | n/a | n/a | n/a | n/a | -12.05102 | 267.3 |
| 1m | Student t | 2.000 | 2.118 | 2.24 | n/a | n/a | -12.04367 | 708.2 |
| 1m | Product generalized t | 1.315 | 3.675 | n/a | n/a | n/a | -11.97818 | 4640 |
| 1m | Product generalized Gaussian | 0.813 | n/a | n/a | n/a | n/a | -11.96625 | 5353 |
| 1m | Elliptical radial lognormal | n/a | n/a | n/a | -0.449 | 1.059 | -11.95991 | 5734 |
| 1m | Gaussian | n/a | n/a | n/a | n/a | n/a | -11.58248 | 28377 |
| 15m | Elliptical generalized t | 1.441 | 3.580 | n/a | n/a | n/a | -9.38738 | 0.00 |
| 15m | Student t | 2.000 | 2.168 | 2.34 | n/a | n/a | -9.38442 | 175.8 |
| 15m | Elliptical radial lognormal | n/a | n/a | n/a | -0.374 | 0.879 | -9.37576 | 695.2 |
| 15m | Elliptical generalized Gaussian | 0.661 | n/a | n/a | n/a | n/a | -9.37140 | 957.2 |
| 15m | Product generalized t | 1.541 | 2.693 | n/a | n/a | n/a | -9.31072 | 4600 |
| 15m | Product generalized Gaussian | 0.838 | n/a | n/a | n/a | n/a | -9.29037 | 5819 |
| 15m | Gaussian | n/a | n/a | n/a | n/a | n/a | -8.89493 | 29543 |
| 1h | Elliptical generalized t | 1.212 | 4.621 | n/a | n/a | n/a | -8.00795 | 0.00 |
| 1h | Student t | 2.000 | 2.081 | 2.16 | n/a | n/a | -8.00134 | 395.1 |
| 1h | Elliptical radial lognormal | n/a | n/a | n/a | -0.378 | 0.904 | -7.99965 | 496.3 |
| 1h | Elliptical generalized Gaussian | 0.637 | n/a | n/a | n/a | n/a | -7.99862 | 558.2 |
| 1h | Product generalized t | 1.299 | 3.776 | n/a | n/a | n/a | -7.91283 | 5707 |
| 1h | Product generalized Gaussian | 0.829 | n/a | n/a | n/a | n/a | -7.90270 | 6313 |
| 1h | Gaussian | n/a | n/a | n/a | n/a | n/a | -7.56611 | 26507 |
| 4h | Elliptical generalized t | 0.842 | 12.725 | n/a | n/a | n/a | -6.56266 | 0.00 |
| 4h | Elliptical generalized Gaussian | 0.645 | n/a | n/a | n/a | n/a | -6.56134 | 26.9 |
| 4h | Elliptical radial lognormal | n/a | n/a | n/a | -0.341 | 0.925 | -6.54949 | 286.6 |
| 4h | Student t | 2.000 | 2.078 | 2.16 | n/a | n/a | -6.54410 | 404.7 |
| 4h | Product generalized t | 1.102 | 7.737 | n/a | n/a | n/a | -6.45265 | 2410 |
| 4h | Product generalized Gaussian | 0.883 | n/a | n/a | n/a | n/a | -6.45031 | 2460 |
| 4h | Gaussian | n/a | n/a | n/a | n/a | n/a | -6.20760 | 7775 |
| 1d | Elliptical generalized Gaussian | 0.811 | n/a | n/a | n/a | n/a | -4.56412 | 0.00 |
| 1d | Elliptical generalized t | 0.954 | 18.715 | n/a | n/a | n/a | -4.56463 | 0.13 |
| 1d | Student t | 2.000 | 2.554 | 3.11 | n/a | n/a | -4.55218 | 43.6 |
| 1d | Elliptical radial lognormal | n/a | n/a | n/a | -0.220 | 0.870 | -4.52605 | 139.0 |
| 1d | Product generalized t | 1.406 | 6.260 | n/a | n/a | n/a | -4.49209 | 264.9 |
| 1d | Product generalized Gaussian | 1.092 | n/a | n/a | n/a | n/a | -4.48878 | 275.0 |
| 1d | Gaussian | n/a | n/a | n/a | n/a | n/a | -4.36657 | 719.0 |

## Shape similarity across scales

Every histogram is standardized within its own scale before comparison. The unconditional comparison retains zero atoms; the continuous comparison removes every pair with either return exactly zero.

| Adjacent scales | JS bits, all pairs | JS bits, continuous |
| --- | --- | --- |
| 1s → 1m | 0.3210 | 0.1301 |
| 1m → 15m | 0.0103 | 0.0040 |
| 15m → 1h | 0.0041 | 0.0041 |
| 1h → 4h | 0.0152 | 0.0152 |
| 4h → 1d | 0.0796 | 0.0796 |

Jensen–Shannon divergence is zero only for identical binned shapes and at most one bit. Daily comparisons are noisier because the five-year daily sample contains only about 1,825 pairs.

## Stability across one-year epochs

| Scale | Corr r range | Corr magnitude range | GGD p range | Median JS to 5y | Max JS to 5y |
| --- | --- | --- | --- | --- | --- |
| 1s | -0.0126–0.1272 | 0.307–0.403 | 0.150–0.260 | 0.0510 | 0.1989 |
| 1m | -0.0054–0.0194 | 0.351–0.415 | 0.552–0.732 | 0.0044 | 0.0054 |
| 15m | -0.0319–0.0221 | 0.253–0.352 | 0.585–0.778 | 0.0072 | 0.0089 |
| 1h | -0.0316–0.0078 | 0.226–0.286 | 0.557–0.774 | 0.0161 | 0.0205 |
| 4h | -0.0551–0.0283 | 0.115–0.251 | 0.549–0.776 | 0.0599 | 0.0649 |
| 1d | -0.0879–0.0099 | 0.027–0.298 | 0.633–1.089 | 0.2371 | 0.2614 |

The machine-readable JSON also contains trailing 30d, 90d, and 365d windows, quadrant probabilities, squared-return correlation, joint 2σ/3σ lifts, the response of next-return magnitude after a 2σ move, whitened radial quantiles, angular non-ellipticity, and the normalized 2D histograms.

## Interpretation

- The best family here is an unconditional descriptive law, not a predictive transition model. Magnitude correlation and joint-tail lift show that $r_{t+1}$ is not independent of $r_t$ even when signed correlation is near zero.
- A symmetric elliptical density can represent common stochastic volatility through a shared radius, but it cannot represent time-varying volatility regimes exactly. The product alternatives test a different contour shape, not a full volatility model.
- The daily result has limited power for distinguishing flexible families. Five years provide thousands of minute-scale pairs but only roughly 1,825 daily pairs and roughly 365 per annual epoch.
- The 4h and especially 1d annual-epoch JS distances have a substantial sparse-histogram noise floor; use their parameter ranges as uncertainty indicators, not as proof of regime changes. The 1s generalized-Gaussian proxy also reaches the fit's lower power bound and should not be interpreted literally.
- Close prices omit intrabar extremes, spreads, fees, slippage, and liquidation paths. One-second last-trade closes also have tick-size and no-trade artifacts.

## Reproduction

Run `npm run analysis:return-pairs`, then `npm run analysis:return-pairs:render` for the six-scale heatmap. Machine-readable results are in `data/benchmarks/consecutive-log-return-pairs.json`.
