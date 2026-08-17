# BTCUSDT one-second dependence model without a separate jump process

Generated from the complete common BTCUSDT spot history ending
2026-07-25T00:00:00Z: 157,766,399 one-second returns and 2,629,439 complete
60-second windows.

## Objective

The IID convolution of the empirical 1s return marginal substantially misses
the observed 1m distribution. The previous delay experiment showed that the
zero-gap marginal explains only 1.6% of that distribution discrepancy, and
even the exact per-minute activity-count distribution explains only 15.0%.

This experiment models the remaining dependence without creating a separate
jump process. Extreme returns remain in the empirical volatility and normalized
intraminute-shape tails. They are not labeled, removed, or assigned a separate
arrival intensity.

The process separates:

1. zero/nonzero activity;
2. slowly changing activity and volatility state;
3. intraminute relative magnitude clustering;
4. signed return ordering;
5. an empirical heavy-tailed residual shape.

## Measured dependence used as input

For a centered 1s return $r_t$, activity indicator $A_t=\mathbf 1[r_t\ne0]$,
and magnitude $M_t=|r_t|$, the measured inputs are

$$
\rho_r(k),\qquad
\rho_A(k),\qquad
\rho_M(k),\qquad
\rho_{\log(1+M)}(k),\qquad 1\le k\le900.
$$

Squared-return correlation is retained in the machine-readable report, but it
is not the primary volatility input because it is exceptionally sensitive to a
small number of extreme returns. Absolute and log-absolute correlations are
more stable when jumps are not modeled separately.

| Lag | Return ACF | Activity ACF | Absolute-return ACF | Log-absolute ACF |
| ---: | ---: | ---: | ---: | ---: |
| 1s | 0.04571 | 0.22170 | 0.36445 | 0.39436 |
| 2s | 0.02814 | 0.17692 | 0.33067 | 0.35106 |
| 5s | 0.00388 | 0.16547 | 0.30983 | 0.33036 |
| 10s | 0.00421 | 0.16184 | 0.29728 | 0.32077 |
| 30s | 0.00127 | 0.15731 | 0.27596 | 0.30422 |
| 60s | 0.00305 | 0.15674 | 0.26646 | 0.29914 |
| 300s | 0.00179 | 0.15182 | 0.23097 | 0.27417 |
| 900s | 0.00173 | 0.14868 | 0.20302 | 0.25327 |

The signed ACF is short and small at each individual lag, but its weighted sum
is material. For $R_h=\sum_{j=1}^h r_j$,

$$
\frac{\operatorname{Var}(R_h)}{h\operatorname{Var}(r_t)}
=1+2\sum_{k=1}^{h-1}\left(1-\frac{k}{h}\right)\rho_r(k).
$$

At $h=60$, the measured lags imply a variance ratio of 1.2259. The direct
observed 1m/1s variance ratio is 1.2386; the small difference comes from daily
FFT boundaries, histogram discretization, and the difference between the exact
aligned return samples used by the two calculations.

The activity and volatility state is much more persistent than a first-order
second-level model suggests. Per-minute nonzero counts have correlations 0.881
at one minute, 0.830 at 15 minutes, and 0.725 at one day. Per-minute log RMS
nonzero magnitude has correlations 0.570, 0.480, and 0.299 at those horizons.

## State and template definitions

For historical minute $m$, define

$$
N_m=\sum_{j=1}^{60}A_{m,j},
\qquad
Q_m=\sum_{j=1}^{60}r_{m,j}^2,
$$

and the RMS active-return magnitude

$$
U_m=\sqrt{\frac{Q_m}{\max(N_m,1)}}.
$$

Each minute is assigned a discrete state $Z_m$ from the Cartesian product of:

- the quartile of $N_m$, using upper edges 29, 36, and 47 active seconds;
- the quartile of $\log U_m$, using upper edges -1.06682, -0.51803, and
  -0.05832 when $U_m$ is measured in basis points.

This gives 16 observed joint activity/volatility states. The full 16×16
one-minute transition matrix is saved for inspection, but a first-order Markov
chain is not used as the final state generator because it forgets the measured
long persistence too quickly.

For each historical minute with $Q_m>0$, define a unit-energy normalized
magnitude template

$$
W_{m,j}=\frac{|r_{m,j}|}{\sqrt{Q_m}},
\qquad
\sum_{j=1}^{60}W_{m,j}^2=1.
$$

The corresponding activity and sign templates are

$$
A_{m,j}=\mathbf 1[r_{m,j}\ne0],
\qquad
S_{m,j}=\operatorname{sign}(r_{m,j}).
$$

The tuple $(A_{m,1:60},W_{m,1:60},S_{m,1:60})$ is called the normalized
intraminute shape. It contains no absolute volatility scale because $Q_m$ has
been divided out.

## Final modeled process

The final process is a semi-parametric block-state marked activity model.

### 1. Generate the slow state and scale driver

The historical sequence $(Z_m,Q_m)$ is split into contiguous 1,440-minute
blocks. A random permutation of these blocks without replacement produces the
synthetic driver sequence

$$
(\widetilde Z_m,\widetilde Q_m).
$$

Using full-day blocks preserves the empirical slow decay of activity and
volatility correlations over the 1s–15m range. Sampling without replacement
preserves the exact full-history marginal distribution of $Q_m$, including its
rare extremes. Dependence across permuted day boundaries is intentionally
broken.

### 2. Draw an independent intraminute shape

For each generated minute $m$, independently sample a historical template
index $I_m$ uniformly from minutes satisfying

$$
Z_{I_m}=\widetilde Z_m.
$$

Only the state must match. The template minute is not the minute supplying
$\widetilde Q_m$. This independent recombination prevents the generator from
replaying historical returns.

### 3. Apply the independently generated volatility scale

The synthetic 1s returns are

$$
\widetilde r_{m,j}
=A_{I_m,j}S_{I_m,j}W_{I_m,j}\sqrt{\widetilde Q_m}.
$$

Consequently,

$$
\sum_{j=1}^{60}\widetilde r_{m,j}^2=\widetilde Q_m
$$

up to float16 template-storage precision. Exact zeros, clustered relative
magnitudes, sign order, and the independently generated minute volatility scale
are all present in the resulting second path.

### 4. No separate jump process

There is no Bernoulli jump indicator, Poisson intensity, or special jump-size
law. A rare extreme can enter through either:

- an extreme emitted scale $\widetilde Q_m$;
- an unusually concentrated normalized shape $W_{I_m,1:60}$.

This deliberately treats such observations as the far tail of the same
activity/volatility/innovation process.

## Incremental controls

Four models are compared on the observed 1m histogram:

1. **IID seconds:** convolve the full empirical 1s marginal 60 times.
2. **Activity counts:** use the exact observed distribution of $N_m$, but draw
   nonzero marks IID.
3. **Clustered volatility:** use the block-state driver and normalized
   activity/magnitude templates, but replace signs independently using the
   empirical positive-return probability.
4. **Full non-jump dependence:** restore the sign template paired with the
   activity and normalized magnitude shape.

| Model | Observed/model variance | Center observed/model | $>3\sigma$ observed/model | $>5\sigma$ observed/model | JS bits |
| --- | ---: | ---: | ---: | ---: | ---: |
| IID seconds | 1.238 | 1.542 | 3.318 | 8.052 | 0.02713 |
| Exact activity counts + IID marks | 1.238 | 1.482 | 2.922 | 7.655 | 0.02306 |
| Clustered activity/volatility + IID signs | 1.236 | 0.837 | 1.186 | 1.171 | 0.00510 |
| Full non-jump dependence | 1.026 | 0.973 | 0.982 | 0.985 | 0.00138 |

Activity clustering alone improves the fit only modestly. Adding the persistent
volatility scale and intraminute magnitude shape removes most of the center and
tail discrepancy, but IID signs cannot create the observed covariance, so its
variance remains approximately the IID variance.

Restoring the empirical signed shape supplies that covariance. The full model
is 2.6% low in variance, 2.7% high in central mass, 1.9% high in the three-sigma
tail, and 1.5% high in the five-sigma tail. Its Jensen-Shannon divergence is
0.00138 bits, a 94.9% reduction from the IID model.

## Dependence validation

The full model's ACF was measured on a separately generated contiguous sample
of 100,000 minutes.

| Quantity | Observed | Model |
| --- | ---: | ---: |
| Return ACF, 1s | 0.04571 | 0.05446 |
| Return ACF, 2s | 0.02814 | 0.02812 |
| Return-implied 60s variance ratio | 1.2259 | 1.2327 |
| Activity ACF, 1s | 0.22170 | 0.22582 |
| Activity ACF, 60s | 0.15674 | 0.14618 |
| Activity ACF, 900s | 0.14868 | 0.13529 |
| Absolute-return ACF, 1s | 0.36445 | 0.33914 |
| Absolute-return ACF, 60s | 0.26646 | 0.24164 |
| Absolute-return ACF, 900s | 0.20302 | 0.18443 |

The model matches the weighted signed covariance that controls 1m variance and
reproduces most activity and magnitude persistence. Its remaining systematic
shortfall is about 7–10% in absolute-return and activity correlation. This is
consistent with dependence broken at state-conditioned template recombination
and block boundaries.

## What this model is and is not

This is a descriptive and generative distribution model. It establishes that
the 1s→1m discrepancy can be explained almost entirely by persistent
activity/volatility state plus normalized intraminute signed shapes, without a
separate jump mechanism.

It is not yet a causal forecasting model:

- $Z_m$ is defined from the completed minute, so it cannot be observed at that
  minute's first second;
- full-history empirical emissions include nonstationarity and future regimes;
- daily block permutation is a dependence-preserving bootstrap, not a compact
  online state equation;
- dependence beyond a permuted block boundary is not modeled.

For real-time use, replace the descriptive state with a filtered state based
only on completed data, for example trailing activity and realized volatility.
The empirical emission templates can remain as the innovation model, while a
causal state filter predicts the distribution of the next $(Z_m,Q_m)$.

Machine-readable measurements, the transition matrix, complete ACF curves, and
validation results are stored in
`data/benchmarks/one-second-dependence-model.json`. They are generated by
`ml/analyze_one_second_dependence_model.py`.

## Propagation to slower return scales

The generated 1s path was also aggregated directly, without refitting, to 15m,
1h, 4h, and 1d. Thus this tests whether the modeled dependence propagates
through the exact identity

$$
R_H=\sum_{t=1}^{H}r_t,
$$

rather than whether a new model can be fitted at each horizon.

| Scale | Observed/model variance | Center observed/model | $>3\sigma$ observed/model | $>5\sigma$ observed/model | JS bits |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1m | 1.026 | 0.973 | 0.982 | 0.985 | 0.00138 |
| 15m | 0.991 | 0.984 | 1.024 | 1.202 | 0.00156 |
| 1h | 0.972 | 1.070 | 1.162 | 1.458 | 0.00385 |
| 4h | 0.947 | 1.187 | 1.050 | 0.889 | 0.00885 |
| 1d | 1.005 | 1.029 | 0.999 | 2.998 | 0.01924 |

The model transfers very well through 15m, reasonably through 1h, and only
partially at 4h. The daily center, three-sigma tail, and variance happen to
match, but the daily sample has only 1,826 observations and three observed
five-sigma events; the five-sigma ratio and JS distance are therefore dominated
by sampling noise. This is not strong evidence of a validated daily model.

The degradation at 1h and 4h shows that a minute-state driver plus independently
recombined minute templates is not a complete scale-invariant model. It retains
most slow activity/volatility persistence, but it does not explicitly model the
joint evolution of signed returns and volatility across adjacent minutes and
hours.

## Feature relevance by scale

Not every 1s feature should remain an explicit feature at every slower scale.
Some become negligible; others are absorbed into coarser sufficient summaries.

| Feature | 1s→1m | 1m→15m | 15m→1h | 1h→4h | 4h→1d |
| --- | --- | --- | --- | --- | --- |
| Exact-zero delay / activity timing | High | Low–moderate | Low | Negligible for return shape | Negligible for return shape |
| Signed short-lag dependence | High for variance | Small but useful | Very small | Very small | Uncertain |
| Persistent volatility state | Dominant | Dominant | Dominant | Dominant | Dominant but poorly sampled |
| Normalized intraminute shape | High | Useful through aggregation | Mostly absorbed into minute innovations | Absorbed | Absorbed |
| Heavy-tailed innovation | High | High | High | Moderate | Still present, imprecisely measured |
| Slow activity/volatility state history | Moderate | High | High | High | Requires longer than one-day state model |

The empirical reasons are:

- Exact-zero probability falls from 35.73% at 1s to 2.60% at 1m, 0.063% at
  15m, 0.023% at 1h, and zero at 4h and 1d. At slow scales, activity should
  influence returns through integrated variance and liquidity state, not as an
  explicit zero-gap mechanism.
- Lag-1 signed-return correlation falls from 0.0457 at 1s to 0.00515 at 1m,
  -0.00071 at 15m, -0.00567 at 1h, and 0.00041 at 4h. Correspondingly, local
  parent-to-child variance ratios after 1m remain between 0.957 and 1.042.
- Lag-1 absolute-return correlation remains 0.388 at 1m, 0.319 at 15m, 0.275
  at 1h, 0.192 at 4h, and 0.164 at 1d. Volatility clustering is therefore the
  only measured feature that is clearly important at every scale.
- Excess kurtosis declines from 421 at 1s to 109 at 1m, 31.7 at 15m, 12.7 at
  1h, 7.4 at 4h, and 4.0 at 1d. Heavy tails weaken under aggregation but do not
  disappear.

For a compact model intended to operate at multiple horizons, the explicit 1s
activity and template process should generate the next minute. Above one
minute, its output can be summarized by minute return $R_m$, realized variance
$Q_m$, activity count $N_m$, and the filtered slow state. A separate
multiscale state equation should then evolve volatility over 15m, 1h, 4h, and
daily horizons. Carrying every 60-bit activity or sign template into the
hourly state would add complexity without retaining useful independent
information.

## Component-free successor

The historical daily-block and minute-template sampling described above has
now been replaced by a fitted stochastic generator that does not select any
historical candle component during generation. Its process definition,
parameters, and direct 1s-to-1d validation are recorded in
`docs/experiments/component-free-one-second-process-2026-08-12.md`.
