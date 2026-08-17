# BTCUSDT log-return distributions across scales and history windows

Generated 2026-08-12T13:11:45.702Z. Data ends at **2026-07-25T00:00:00.000Z**.

## Result

- The unconditional distribution is sharply peaked and fat-tailed at every scale: full-history central-mass ratios run from 1.55× to 3.64× Gaussian, while three-sigma events occur 5.8× to 7.9× as often.
- Aggregation makes the center more Gaussian but does not eliminate tail risk: excess kurtosis changes from 421.3 at 1s to 4.0 at 1d.
- The latest 365-day 1s annualized volatility is 38.04%, versus 48.69% over the full sample; distribution scale is therefore not stationary.
- Window placement matters materially: one-year 1s annualized volatility ranges from 38.04% (2025-07-25 to 2026-07-25) to 69.45% (2021-07-25 to 2022-07-25).
- Absolute-return lag-1 correlation is 0.364 at 1s and 0.164 at 1d, direct evidence that volatility clustering contributes to the unconditional mixture shape.
- At 1s, 35.73% of close-to-close returns are exactly zero, so the center contains market microstructure and tick-size effects in addition to continuous price variation.

## Full-history shape

| Scale | n | σ (bp) | Ann. vol % | Skew | Excess kurt. | Peak / normal | >3σ / normal | ACF |r| lag 1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1s | 157,766,399 | 0.867 | 48.69 | 0.34 | 421.3 | 3.64 | 7.9 | 0.364 |
| 1m | 2,629,440 | 7.474 | 54.19 | -0.24 | 109.4 | 1.89 | 5.8 | 0.388 |
| 15m | 175,296 | 28.319 | 53.01 | -0.33 | 31.7 | 1.70 | 6.5 | 0.319 |
| 1h | 43,824 | 55.834 | 52.26 | -0.18 | 12.7 | 1.75 | 7.5 | 0.275 |
| 4h | 10,956 | 110.131 | 51.54 | -0.13 | 7.4 | 1.76 | 7.2 | 0.192 |
| 1d | 1,826 | 275.344 | 52.60 | -0.21 | 4.0 | 1.55 | 6.5 | 0.164 |

`Peak / normal` is the observed mass within ±0.25 sample standard deviations,
divided by the Gaussian expectation. `>3σ / normal` is the observed two-sided
three-sigma exceedance rate divided by the Gaussian expectation. Values above
one mean a sharper center or heavier tail than a Gaussian with the same variance.

## Dependence on trailing-window length

All rows end at 2026-07-25T00:00:00.000Z; only the amount of history changes.

| Window | Scale | n | Ann. vol % | Skew | Excess kurt. | >3σ / normal |
| --- | --- | --- | --- | --- | --- | --- |
| 7d | 1s | 604,800 | 22.08 | 0.29 | 212.5 | 9.3 |
| 7d | 1m | 10,080 | 28.74 | -0.24 | 11.0 | 5.4 |
| 7d | 15m | 672 | 27.99 | 0.21 | 4.2 | 7.2 |
| 7d | 1h | 168 | 28.34 | 1.14 | 8.0 | 6.6 |
| 7d | 4h | 42 | 24.95 | -0.47 | 0.7 | 0.0 |
| 7d | 1d | 7 | 26.55 | 0.18 | -1.7 | 0.0 |
| 30d | 1s | 2,592,000 | 31.48 | 1.39 | 145.6 | 8.3 |
| 30d | 1m | 43,200 | 39.57 | 0.56 | 18.8 | 6.1 |
| 30d | 15m | 2,880 | 39.54 | -0.86 | 21.7 | 6.2 |
| 30d | 1h | 720 | 40.84 | -1.16 | 26.9 | 5.7 |
| 30d | 4h | 180 | 35.53 | 0.29 | 2.3 | 6.2 |
| 30d | 1d | 30 | 30.91 | 0.30 | -0.0 | 0.0 |
| 90d | 1s | 7,776,000 | 32.41 | 0.58 | 176.0 | 8.5 |
| 90d | 1m | 129,600 | 40.75 | 0.48 | 20.9 | 6.5 |
| 90d | 15m | 8,640 | 40.21 | -0.05 | 15.1 | 6.4 |
| 90d | 1h | 2,160 | 40.62 | -0.45 | 12.8 | 7.4 |
| 90d | 4h | 540 | 38.59 | -0.17 | 3.2 | 7.5 |
| 90d | 1d | 90 | 34.75 | -0.39 | 1.2 | 4.1 |
| 365d | 1s | 31,536,000 | 38.04 | 2.77 | 2053.9 | 8.2 |
| 365d | 1m | 525,600 | 45.14 | -0.40 | 81.8 | 6.3 |
| 365d | 15m | 35,040 | 44.52 | -0.16 | 16.7 | 6.5 |
| 365d | 1h | 8,760 | 43.13 | -0.33 | 8.7 | 7.0 |
| 365d | 4h | 2,190 | 41.99 | -0.33 | 3.9 | 6.3 |
| 365d | 1d | 365 | 43.24 | -0.53 | 6.9 | 3.0 |

Daily-scale rows with short lookbacks have too few observations for reliable
shape inference. The JSON includes a `sampleWarning` on every affected row.

## Dependence on the particular historical epoch

The one-year epochs are non-overlapping and share the same calendar boundaries.

| Epoch start | Scale | Ann. vol % | Skew | Excess kurt. | >3σ / normal | Down/up p99 |
| --- | --- | --- | --- | --- | --- | --- |
| 2021-07-25 | 1s | 69.45 | 0.29 | 226.7 | 7.3 | 0.99 |
| 2021-07-25 | 1m | 72.46 | 0.24 | 71.8 | 5.4 | 0.96 |
| 2021-07-25 | 15m | 70.24 | -0.01 | 18.9 | 6.1 | 0.99 |
| 2021-07-25 | 1h | 69.09 | -0.10 | 8.2 | 6.8 | 0.95 |
| 2021-07-25 | 4h | 68.85 | -0.12 | 4.6 | 6.8 | 0.99 |
| 2021-07-25 | 1d | 68.92 | -0.38 | 2.1 | 6.1 | 1.27 |
| 2022-07-25 | 1s | 47.86 | -0.14 | 188.4 | 7.1 | 0.99 |
| 2022-07-25 | 1m | 49.49 | -0.78 | 239.3 | 5.5 | 1.00 |
| 2022-07-25 | 15m | 48.32 | -0.41 | 31.6 | 6.9 | 1.02 |
| 2022-07-25 | 1h | 48.69 | -0.30 | 20.8 | 7.5 | 1.06 |
| 2022-07-25 | 4h | 47.67 | -0.13 | 11.0 | 7.3 | 0.94 |
| 2022-07-25 | 1d | 51.34 | -0.28 | 5.8 | 9.1 | 0.87 |
| 2023-07-25 | 1s | 39.54 | -0.60 | 353.0 | 8.0 | 1.00 |
| 2023-07-25 | 1m | 48.60 | -1.16 | 104.6 | 5.7 | 1.00 |
| 2023-07-25 | 15m | 48.06 | -1.38 | 80.3 | 6.0 | 1.01 |
| 2023-07-25 | 1h | 46.27 | -0.17 | 13.1 | 7.4 | 1.00 |
| 2023-07-25 | 4h | 44.98 | -0.16 | 7.2 | 7.9 | 0.99 |
| 2023-07-25 | 1d | 47.86 | 0.21 | 2.4 | 7.1 | 0.90 |
| 2024-07-25 | 1s | 41.59 | -0.04 | 130.8 | 8.5 | 1.01 |
| 2024-07-25 | 1m | 50.83 | -0.15 | 54.3 | 6.1 | 0.99 |
| 2024-07-25 | 15m | 49.91 | -0.21 | 17.5 | 6.4 | 1.02 |
| 2024-07-25 | 1h | 50.11 | -0.08 | 10.2 | 7.6 | 1.05 |
| 2024-07-25 | 4h | 49.82 | 0.05 | 7.0 | 6.4 | 1.02 |
| 2024-07-25 | 1d | 47.74 | 0.40 | 2.5 | 7.1 | 0.79 |
| 2025-07-25 | 1s | 38.04 | 2.77 | 2053.9 | 8.2 | 1.00 |
| 2025-07-25 | 1m | 45.14 | -0.40 | 81.8 | 6.3 | 1.00 |
| 2025-07-25 | 15m | 44.52 | -0.16 | 16.7 | 6.5 | 1.01 |
| 2025-07-25 | 1h | 43.13 | -0.33 | 8.7 | 7.0 | 1.05 |
| 2025-07-25 | 4h | 41.99 | -0.33 | 3.9 | 6.3 | 1.04 |
| 2025-07-25 | 1d | 43.24 | -0.53 | 6.9 | 3.0 | 1.06 |

`Down/up p99` compares the magnitude of the centered 1st-percentile loss with
the centered 99th-percentile gain. Values above one indicate the left tail is
larger at that quantile.

## Method

- Source: 157,766,400 BTCUSDT spot one-second candles across 1,826 complete daily shards, plus 2,669,760 one-minute candles across 1,854 shards.
- Return: native one-second close-to-close log returns and non-overlapping UTC-aligned 1m, 15m, 1h, 4h, and 1d close-to-close log returns.
- The 1m through 1d scales share one minute-close source. The 1s source covers the same full comparison window. Incomplete buckets and returns across gaps are omitted.
- Annualized volatility uses 365-day crypto trading. Shape statistics use each window's own sample mean and sample standard deviation.
- The machine-readable result at `data/benchmarks/log-return-distributions.json` also contains quantiles, zero mass, robust scale, return autocorrelation, absolute-return autocorrelation, standardized absolute-tail survival curves, and rolling 90-day estimates stepped every 30 days. Moments and direct counts are exact at every scale; 1s quantiles and standardized probability masses use a deterministic tail-dense weighted sketch to stay memory-bounded.

## Interpretation limits

- These are unconditional sample distributions, not forecasts and not evidence of independent or identically distributed returns.
- Excess kurtosis and sigma-tail multiples are strongly regime- and sample-dependent because volatility is clustered. A conditional volatility model would remove part, but not all, of the apparent heavy tail.
- Close-to-close returns omit intrabar extremes, spread, fees, slippage, and liquidation paths. One-second closes are last-trade samples, not executable bid/ask returns; tick-size and no-trade seconds create a discrete mass at zero.
- The latest common complete UTC boundary is used; no live or partial candle is included.

## Continuous return-distribution fit

At 1s and 1m, the exact-zero and microstructure peak should not determine the
continuous return fit. Their single zero-containing histogram bin, of width
$0.1\sigma$, is excluded. At 15m, 1h, 4h, and 1d, that bin is included and the
fitted density covers the full distribution through zero.

Let $x=r/\sigma$, where $r$ is the log return and $\sigma$ is the
full-history standard deviation at that scale. A symmetric generalized t is
selected by AIC from 1s through 1h:

$$
f(x\mid\mu,s,p,q)=
\frac{p}{2sB(1/p,q-1/p)}
\left[1+\left(\frac{|x-\mu|}{s}\right)^p\right]^{-q},
\qquad q>1/p.
$$

At 4h and 1d, AIC instead selects the symmetric generalized normal:

$$
f(x\mid\mu,s,p)=
\frac{p}{2s\Gamma(1/p)}
\exp\!\left[-\left(\frac{|x-\mu|}{s}\right)^p\right].
$$

The density in native return units is

$$
f_r(r)=\frac{1}{\sigma}f\!\left(\frac{r}{\sigma}\right).
$$

For 1s and 1m, the overlay is normalized to the observed mass outside the
excluded central bin $C=[-0.05\sigma,0.05\sigma)$:

$$
\widehat{\Pr}(r\in A)=
\frac{m_{\mathrm{out}}}{\Pr_f(X\notin C)}
\int_{A/\sigma}f(x)\,dx,
\qquad A\cap C=\varnothing.
$$

For 15m and slower, no bin is excluded and the fitted density integrates to one
over the entire support, including zero.

| Scale | Family | $\sigma$ (bp) | $\mu/\sigma$ | $s/\sigma$ | $p$ | $q$ | PDF tail exponent $pq$ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1s | Generalized t | 0.867 | -0.0028 | 2.6109 | 1.2777 | 3.9474 | 5.0437 |
| 1m | Generalized t | 7.474 | -0.0001 | 1.2163 | 1.4681 | 2.9505 | 4.3316 |
| 15m | Generalized t | 28.319 | 0.0020 | 1.0224 | 1.6355 | 2.4147 | 3.9492 |
| 1h | Generalized t | 55.834 | 0.0011 | 1.2251 | 1.3829 | 3.0956 | 4.2809 |
| 4h | Generalized normal | 110.131 | 0.0000 | 0.3683 | 0.7351 | — | — |
| 1d | Generalized normal | 275.344 | 0.0000 | 0.5885 | 0.9020 | — | — |

For the generalized-t fits, the asymptotic density and two-sided survival behave
as

$$
f(x)\propto |x|^{-pq},\qquad
\Pr(|X|\ge x)\propto x^{-(pq-1)}.
$$

The generalized normal has stretched-exponential rather than power-law tails.
At 1d only 1,826 observations are available, so its shape parameters should not
be interpreted precisely. Reproducible parameters and model comparisons are stored in
`data/benchmarks/log-return-distribution-fits.json`, generated by
`scripts/fit-log-return-distributions.ts`.

## Hierarchical aggregation: observed versus IID convolution

The scale hierarchy can be used directly instead of trying to regress the
generalized-t parameter $q$ on interval length. For aligned close-to-close log
returns, aggregation is an exact sum:

$$
R_h(t)=\log\!\left(\frac{P_t}{P_{t-h}}\right)
=\sum_{j=0}^{h-1}\log\!\left(\frac{P_{t-j}}{P_{t-j-1}}\right).
$$

If the one-second returns had the observed one-second marginal distribution but
were independent, the characteristic function at horizon $h$ would be

$$
\phi_{h,\mathrm{IID}}(u)=\left[\phi_{1\mathrm{s}}(u)\right]^h.
$$

The empirical one-second probability mass was therefore convolved $h$ times by
FFT and rebinned onto each slower scale's native $0.1\sigma$ grid. This gives a
non-parametric prediction of how the distribution would evolve under IID
aggregation; it does not require $p$ or $q$ to remain fixed.

A second, local comparison uses the immediate parent in the hierarchy:

$$
\phi_{\mathrm{child},\mathrm{parent\text{-}IID}}(u)
=\left[\phi_{\mathrm{parent}}(u)\right]^m,
$$

where $m$ is 60 for $1\mathrm{s}\to1\mathrm{m}$, 15 for
$1\mathrm{m}\to15\mathrm{m}$, 4 for $15\mathrm{m}\to1\mathrm{h}$ and
$1\mathrm{h}\to4\mathrm{h}$, and 6 for $4\mathrm{h}\to1\mathrm{d}$. This
preserves all dependence already accumulated inside a parent block and tests
only what is added between adjacent parent blocks.

| Step | Variance / parent IID | Center mass / parent IID | $>3\sigma$ tail / parent IID | $>3\sigma$ tail / 1s IID |
| --- | ---: | ---: | ---: | ---: |
| 1s × 60 → 1m | 1.239 | 1.54 | 3.32 | 3.32 |
| 1m × 15 → 15m | 0.957 | 1.52 | 1.81 | 9.10 |
| 15m × 4 → 1h | 0.972 | 1.50 | 1.52 | 12.38 |
| 1h × 4 → 4h | 0.973 | 1.53 | 1.36 | 11.48 |
| 4h × 6 → 1d | 1.042 | 1.27 | 2.05 | 13.60 |

The important split is that variance becomes almost additive above one minute,
but distribution shape does not. For parent returns $r_t$ and an $m$-block
child return, the variance ratio is

$$
\frac{\operatorname{Var}(\sum_{t=1}^{m}r_t)}
{m\operatorname{Var}(r_t)}
=1+2\sum_{k=1}^{m-1}\left(1-\frac{k}{m}\right)\rho_k.
$$

After the 1s→1m step, the local variance ratios stay between 0.957 and 1.042,
so signed linear return correlations add little variance at those slower steps.
Nevertheless, every observed child distribution is roughly $1.5\times$ as
concentrated near zero and
$1.4\times$ to $2.1\times$ as likely beyond three standard deviations as the
independent-parent convolution. That remaining difference is higher-order
temporal structure: volatility clustering, activity persistence, and temporal
concentration of jumps are plausible contributors.

The plotted binwise residual is

$$
D_h(x)=\log_{10}\!\left(
\frac{\Pr_{\mathrm{observed}}(R_h\text{ in bin }x)}
{\Pr_{\mathrm{IID}}(R_h\text{ in bin }x)}
\right).
$$

Positive values identify returns that occur more often than the chosen IID
aggregation predicts. The full residual curve retains the changes in peak
curvature and tail decay that a single fitted $q$ would collapse. It is an
aggregate dependence signature, not a unique decomposition of the underlying
mechanisms.

The hierarchy alignment was checked on 17,280 minute endpoints sampled across
12 days; all one-second and one-minute closes matched exactly. Machine-readable
results are stored in `data/benchmarks/return-aggregation-hierarchy.json`,
generated by `scripts/analyze-return-aggregation-hierarchy.ts`. The interactive
chart is generated by `scripts/render-return-aggregation-hierarchy.ts`.

## How much the zero-change delay law explains at 1 minute

The delay distribution can be incorporated as a marked renewal process. The
positive zero-run distribution by itself is not sufficient: it must be extended
with zero-length gaps between adjacent nonzero returns. Let $D$ be this complete
interarrival time, let $N_{60}$ be the number of arrivals in a stationary
60-second window, and let $X_j$ be a nonzero one-second return drawn from the
empirical conditional mark distribution. The delay-only model is

$$
R_{60}^{\mathrm{renewal}}=\sum_{j=1}^{N_{60}}X_j,
\qquad X_j\perp D_j.
$$

Its characteristic function is a count mixture of convolution powers:

$$
\phi_{R_{60}}(u)
=\operatorname{E}\!\left[\phi_X(u)^{N_{60}}\right]
=G_{N_{60}}\!\left(\phi_X(u)\right),
$$

where $G_{N_{60}}$ is the probability-generating function of the stationary
renewal count. This retains the empirical 1s nonzero-return distribution and
the full empirical delay distribution, while removing sign ordering, mark-size
ordering, delay/mark coupling, and volatility regimes.

| 1m baseline | Model count variance | Observed/model variance | Center observed/model | $>3\sigma$ observed/model | JS divergence (bits) |
| --- | ---: | ---: | ---: | ---: | ---: |
| IID one-second returns | 13.78 | 1.238 | 1.542 | 3.318 | 0.02713 |
| IID empirical delays + IID nonzero marks | 24.80 | 1.238 | 1.536 | 3.283 | 0.02669 |
| Observed 1m nonzero counts + IID nonzero marks | 145.56 | 1.238 | 1.482 | 2.922 | 0.02306 |

The measured delay distribution therefore explains only **1.6%** of the IID
model's Jensen-Shannon distribution gap. Even granting the model the exact
observed distribution of per-minute activity counts explains only **15.0%**.
Neither activity-count model changes the missing variance: with nearly
zero-mean IID marks, randomizing the count changes peakedness and tails but not
the expected sum of mark variances.

This exposes another hierarchical difference. The empirical delay marginal
does produce more count dispersion than independent Bernoulli activity (Fano
factor 0.64 versus 0.36), but the actual minute count Fano factor is **3.77**.
Thus zero and nonzero seconds themselves arrive in persistent activity regimes;
IID resampling of gap lengths discards most of that dependence. The fitted
multi-hour gaps do reproduce the probability of an entirely inactive minute,
but not the rest of the activity-count distribution.

For forecasting, the fitted delay survival still has a useful role. If a
zero-change run has already lasted $a$ seconds, its next-second continuation
probability is

$$
\Pr(L\ge a+1\mid L\ge a)=\frac{S(a+1)}{S(a)},
$$

so the current run age and fitted survival can drive an activity hazard. To
account for the remaining 1s→1m shape, that hazard should be embedded in a
marked state model such as

$$
\Pr(A_t=1\mid a_t,z_t),
\qquad
X_t\mid A_t=1,a_t,z_t,
$$

where $A_t$ indicates a nonzero move, $a_t$ is inactivity age, and $z_t$ is a
persistent volatility/activity state. The second distribution must retain
return magnitude and sign dependence; the delay law alone cannot supply them.

The reproducible result is stored in
`data/benchmarks/delay-aware-minute-return-distribution.json`, generated by
`scripts/analyze-delay-aware-minute-returns.ts`.
