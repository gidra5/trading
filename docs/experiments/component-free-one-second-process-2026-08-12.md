# Component-free fitted one-second return process

Date: 2026-08-12  
Updated: 2026-08-13  
Instrument: Binance spot BTCUSDT  
Fit window: 2021-07-25 through 2026-07-25  
Fit sample: 157,766,399 one-second returns, 2,629,439 complete minutes, and 1,825 complete days

## Result

This experiment replaces historical component resampling with a fitted hierarchical stochastic process. Generation never selects a historical candle, day, minute, activity mask, sign mask, magnitude profile, or return vector. It retains only fitted quantile splines, categorical probabilities, regression coefficients, calendar harmonics, AR-factor weights, mixture parameters, and scalar calibration constants.

The final acceptance results are:

- 1s JS divergence: **0.000680 bits**, below the 0.005 target.
- Long-run 1d JS divergence: **0.003466 bits**, below the 0.01 target.
- No daily projection required a feasibility fallback.

All slower returns remain sums of the generated one-second process. The daily layer is a higher-level latent constraint in the generative hierarchy, not a separately substituted return series.

## Fitted process

### 1. Daily and minute realized variance

For minute $m$, define realized variance

$$
Q_m=\sum_{j=1}^{60}r_{m,j}^2.
$$

The marginal distribution of $\log(Q_m+q_0)$ is represented by a bounded quantile spline $G_Q$. Its Gaussian-copula score combines daily and weekly Fourier terms with a causal mixture of stationary AR(1) factors:

$$
Z_m^Q=s_Q(m)+\sum_k\sqrt{w_k^Q}X_{k,m}+\sqrt{w_0^Q}\epsilon_m^Q,
$$

$$
X_{k,m}=\phi_kX_{k,m-1}+\sqrt{1-\phi_k^2}\eta_{k,m},
\qquad \phi_k=\exp(-1/\tau_k).
$$

The preliminary generated minute variance is

$$
\widetilde Q_m=\exp\!\left[G_Q\!\left(\Phi(Z_m^Q)\right)\right]-q_0.
$$

A separate bounded quantile spline and long-memory AR-factor process generates each complete day's integrated variance $V_d$. Preliminary minute variances inside day $d$ are normalized exactly:

$$
Q_m=\widetilde Q_m
\frac{V_d}{\sum_{l\in d}\widetilde Q_l},
\qquad m\in d.
$$

This preserves the generated intraday volatility shape while making its daily total equal to a freshly generated daily budget. No historical volatility trajectory is replayed.

### 2. Activity count, zero returns, and timing

The active-second count $N_m\in\{0,\ldots,60\}$ uses a fitted 61-category marginal $F_N$ and a Gaussian-copula score coupled to volatility:

$$
Z_m^N=c_0+c_1Z_m^Q+c_2\left[(Z_m^Q)^2-1\right]
+G_A\!\left(\Phi(U_m)\right)+\delta_N,
$$

$$
N_m=F_N^{-1}\!\left(\Phi(Z_m^N)\right).
$$

$U_m$ is a separate causal AR-factor mixture and $G_A$ is a fitted residual quantile spline. The calibrated location $\delta_N=0.22067$ matches the mean active count exactly:

$$
E[N_m]=38.5637,
\qquad P(r_t=0)=0.3572714.
$$

Within a minute, correlated Gaussian timing scores are generated and the $N_m$ largest positions are marked active. The fitted Gaussian timing correlation is $0.126806$. The resulting adjacent-active probability is 0.46331 versus 0.46343 observed.

### 3. Preliminary minute signed innovation

Define minute signed efficiency

$$
E_m=\frac{R_m}{\sqrt{Q_m}},
\qquad R_m=\sum_{j=1}^{60}r_{m,j}.
$$

Its bounded quantile spline $G_E$ has a fitted quadratic dependence on minute volatility and an otherwise white innovation:

$$
E_m=G_E\!\left(\Phi\!\left[
d_0+d_1Z_m^Q+d_2\left((Z_m^Q)^2-1\right)+\epsilon_m^E
\right]\right).
$$

The preliminary minute target is $\widetilde R_m=E_m\sqrt{Q_m}$. It is clipped to the geometric feasibility bound $|\widetilde R_m|\leq\sqrt{N_mQ_m}$, with $N_m=1$ fixed to the only feasible magnitude $\sqrt{Q_m}$.

### 4. Daily signed-return layer

Healthy minute statistics were not sufficient to reproduce the daily marginal: the missing piece was a slow directional constraint. A bounded fitted daily-return marginal with its own causal AR-factor Gaussian copula generates a fresh signed target $D_d$ for each day.

The preliminary minute returns are tilted so their direct daily sum equals this target. In the unconstrained case,

$$
R_m=\widetilde R_m+
\left(D_d-\sum_{l\in d}\widetilde R_l\right)
\frac{Q_m^*}{\sum_{l\in d}Q_l^*},
\qquad m\in d,
$$

where $Q_m^*=Q_m$ for minutes with at least two active seconds and zero otherwise. If a proposed value crosses $|R_m|\leq\sqrt{N_mQ_m}$, a bounded one-dimensional water-filling projection is used. In the full validation run this fallback was needed zero times.

This makes daily direction part of the one-second generative process: generated second returns still aggregate to these adjusted minute returns, and those minute returns aggregate to $D_d$.

### 5. Nonzero magnitudes and the micro-return spike

The exact-zero probability alone does not explain the very sharp nonzero mass around zero. Excluding it caused the original 1s JS divergence of 0.0562 bits.

Minutes are divided into 16 states using activity-count and volatility quartiles. In each state, a three-component Gaussian mixture models the logarithm of the concentration controlling squared-return shares. Correlated logistic-normal scores then generate fresh active-second energy weights. A separate fitted state-conditional micro-return probability generates small nonzero values inside half of the central histogram bin.

The global energy-dispersion and micro-probability scales are simulation-calibrated against the post-projection 1s histogram. The selected values are 0.96 and 0.95, respectively. This is what reduces 1s JS to 0.000680 without sampling historical intraminute templates.

### 6. Exact second-level constraints

For each minute, the generated active-second base vector is projected so that

$$
\sum_j r_{m,j}=R_m,
\qquad
\sum_j r_{m,j}^2=Q_m.
$$

The projection therefore preserves both the modeled minute return and minute realized variance to floating-point precision. Every slower return is obtained by direct addition of these same generated returns.

## Validation

All distribution-shape columns except JS are observed/model ratios, so one is ideal. JS divergence is in bits.

| Scale | Variance | Center | $>3\sigma$ | $>5\sigma$ | JS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1s | 0.99997 | 0.98914 | 0.98509 | 0.83040 | **0.000680** |
| 1m | 1.00076 | 0.85659 | 0.81946 | 0.75020 | 0.004727 |
| 15m | 0.95471 | 0.93126 | 0.87282 | 0.93637 | 0.002316 |
| 1h | 0.94540 | 1.03449 | 0.98667 | 1.15129 | 0.002791 |
| 4h | 0.93786 | 1.17723 | 1.07682 | 1.23066 | 0.008285 |
| 1d, finite 1,825-day path | 1.08111 | 1.00666 | 1.19165 | 0.99945 | 0.015004 |
| 1d, long-run 50,000 days | 1.00808 | 0.98912 | 0.98589 | 1.15700 | **0.003466** |

### Why the two daily JS values differ

The observed daily histogram contains only 1,825 returns. A multinomial sampling audit using that histogram gives:

| Comparison at 1,825 observations | Median JS | 97.5th percentile |
| --- | ---: | ---: |
| One synthetic sample versus its population | 0.00860 | 0.01156 |
| Two independent samples | 0.01633 | 0.02176 |

The finite generated path and observed history are two samples, so its 0.01500 JS is ordinary sample noise at this resolution. The 50,000-day run estimates the generated population more stably and is the acceptance metric. It still aggregates the same modeled minute-return layer. The internal 60-second projection is omitted only in this long-run calculation because that projection preserves each minute sum exactly; materializing 4.32 billion second returns cannot change a daily sum.

## Interpretation and limitations

- The two requested targets pass: 1s JS is below 0.005 and long-run 1d JS is below 0.01.
- The result identifies a genuine hierarchical feature: daily signed direction cannot be inferred from healthy unconditional minute marginals alone, so it must enter as a slow latent constraint.
- Intermediate shapes remain plausible, but the 1m center and tails and the 4h center can still be improved.
- Activity-count persistence is too weak: generated correlations are 0.659 at lag 1 minute and 0.483 at one day, versus 0.881 and 0.725 observed.
- The long-run daily marginal is an in-sample distribution fit. A chronological train/test audit is required before treating it as evidence of predictive power.
- This is component-free but not a tiny closed-form family. Bounded marginal quantile splines are fitted distribution summaries, not replayed observations.

## Artifacts

- Generator and fitter: `ml/analyze_parametric_one_second_process.py`
- Tests: `ml/test_analyze_parametric_one_second_process.py`
- Fitted parameters and validation: `data/benchmarks/parametric-one-second-process.json`
- Commands: `npm run analysis:parametric-one-second` and `npm run analysis:parametric-one-second:test`
