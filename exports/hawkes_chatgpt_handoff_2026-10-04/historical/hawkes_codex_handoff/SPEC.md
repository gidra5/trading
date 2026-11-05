# BTCUSDT multiscale Hawkes + latent-channel model specification

## 1. Research question

We want a causal probabilistic model of BTCUSDT spot aggressive buy/sell aggregate-trade arrivals over milliseconds to hours. The focus is not merely one-step prediction; the model should explain multiscale clustering and support stable open-loop forecasts up to 1 hour.

The current research question is whether structure not captured by a flexible 2-channel Hawkes process should be modeled as:

- more linear exponential modes,
- nonlinear cross-scale/state-dependent excitation,
- a third unobserved self-/cross-exciting event stream,
- continuous stochastic latent state(s),
- or a combination.

## 2. Dataset

Raw file used in ChatGPT runs:

`BTCUSDT-aggTrades-live-2026-08-17_2026-08-30.jsonl.gz`

Known dataset facts:

- 14,464,973 aggregate trades
- about 12.90 days
- 2026-08-17 09:34:49.910 UTC through 2026-08-30 07:10:09.636 UTC
- average ~12.98 aggregate trades/s
- fields: `timestampMs`, `price`, `quantity`, `aggressorSide`
- buys: 7,459,754
- sells: 7,005,219
- timestamps monotonic
- equal-ms records are frequent and **their file ordering is the true arrival order**

For the previously built 1-second B/S count series:

- shape: `(1,114,521, 2)`
- mean counts/s approximately `[6.69324, 6.28541]`
- total count exactly 14,464,973

For the latest screening work, a 5-second series had:

- 222,905 bins
- mean counts per 5s approximately `[33.4661, 31.4269]`

## 3. Evaluation protocol

Always use chronological split:

- first 60%: training
- next 20%: validation
- final 20%: untouched test

Use validation for:

- number of modes
- timescales
- interaction structure
- regularization
- optimizer/early-stopping choices

Do not repeatedly tune against test metrics.

Primary probabilistic metric:

- held-out Poisson or chosen observation-model negative log-likelihood/log score

Forecast metrics:

- MAE
- RMSE
- Pearson correlation
- R²
- side-wise MAE
- imbalance correlation

Open-loop horizons:

`1s, 5s, 15s, 60s, 300s, 900s, 3600s`

Where feasible, block-bootstrap important metric differences, especially 1-hour MAE/log score.

## 4. Base 2-channel Hawkes model

Observed channels:

\[
\mathbf N=(N_B,N_S)^T.
\]

For target channel `i` and source channel `j`, use an independent exponential mixture:

\[
\phi_{ij}(u)=\sum_{k=1}^{K_{ij}} \alpha_{ij,k} e^{-u/\tau_{ij,k}}1_{u>0}.
\]

Equivalent deterministic traces:

\[
dx_{ij,k}=-\frac{x_{ij,k}}{\tau_{ij,k}}dt+dN_j(t).
\]

Intensity, for an additive model:

\[
\lambda_i(t)=\mu_i+\sum_j\sum_k \alpha_{ij,k}x_{ij,k}(t).
\]

A positivity link may be used if needed, but note that ordinary linear Hawkes branching stability then no longer applies directly.

### Model order

Do **not** freeze previous greedy counts. Search over

\[
K_{BB},K_{BS},K_{SB},K_{SS}.
\]

Recommended selection:

1. start from a compact model,
2. try adding one mode to each edge,
3. for every candidate refit **all** amplitudes and all continuous timescales,
4. take the best validation improvement,
5. perform backward deletion/refit,
6. stop when added complexity no longer gives a meaningful validation improvement.

This is important because the previous `B→S` one-scale result may have been a consequence of staged greedy optimization.

## 5. Stability

For a standard additive multivariate Hawkes process, define the integrated branching matrix

\[
G_{ij}=\int_0^\infty \phi_{ij}(u)du.
\]

If the basis is unnormalized exponential amplitudes,

\[
G_{ij}=\sum_k \alpha_{ij,k}\tau_{ij,k}.
\]

For normalized traces/kernel basis, use the corresponding integrated coefficients.

Require and report

\[
\rho(G)<1,
\]

preferably with a safety margin such as `<= 0.95` during fitting.

Be careful: several early scripts mixed normalized and unnormalized trace conventions. Never quote a branching matrix without confirming the convention.

## 6. Linear cross-scale coupling (`M1`)

General linear state realization:

\[
dx=A x\,dt+B\,dN,\qquad \lambda=\mu+Cx.
\]

Kernel:

\[
\Phi(t)=Ce^{At}B.
\]

If `A` is diagonalizable this reduces to exponential eigenmodes. Therefore an unrestricted `M1` is largely a different realization of the same linear rational/exponential kernel class as a sufficiently flexible exponential mixture.

Use `M1` only if testing whether a **lower-dimensional structured realization** matches the performance of a larger diagonal mixture. Do not count arbitrary linear cross-scale coupling as a fundamentally richer model class by itself.

## 7. Nonlinear cross-scale excitation (`M2`)

The most promising nonlinear extension is slow-state modulation of the amplitude of faster event excitation:

\[
dx_k=-\beta_kx_kdt+
\left[B_k^{(0)}+\sum_{\ell:\tau_\ell>\tau_k}B_{k\ell}^{(1)}x_\ell\right]dN.
\]

Interpretation:

- fast activity can accumulate into slower modes,
- slow regime controls how strongly new events excite fast modes.

Prefer amplitude modulation before making timescales state-dependent; it preserves efficient exponential state recursion and is easier to identify.

A more general nonlinear drift model (`M3`) may include quadratic state interactions such as

\[
dx_k=\left[-\beta_kx_k+\sum_{lm}Q_{klm}x_lx_m\right]dt+B_k(x)dN.
\]

Only retain this if it improves held-out metrics after M0/M2 model order has been properly optimized.

## 8. Latent event channel `L`

User's intended extension is a genuine third unobserved event stream:

\[
\mathbf N=(N_B,N_S,N_L)^T.
\]

All source-target interactions should in principle exist:

\[
\lambda_i(t)=\mu_i+
\sum_{j\in\{B,S,L\}}\int \phi_{ij}(u)dN_j(t-u),
\qquad i\in\{B,S,L\}.
\]

This gives the full 3x3 set:

- `B→B`, `B→S`, `B→L`
- `S→B`, `S→S`, `S→L`
- `L→B`, `L→S`, `L→L`

Each edge may have an independent number of exponential modes and independent optimized timescales.

### Important semantic point

`N_L` is a **latent point process**, not a residual filter and not a continuous OU state.

A proper latent-channel fit must infer latent event counts/times under the joint generative model. `B→L`, `S→L` and `L→L` must enter the inference/likelihood, not be added afterward.

Possible inference routes for the binned count version:

- variational inference over latent count sequence
- particle filter / particle smoother / SMC
- EM with an approximate smoother
- PyTorch/JAX gradient optimization of a differentiable variational posterior

For event-level continuous time, latent-event inference is harder; do it only after the binned model is validated.

### Why L may explain apparent nonlinearity

After marginalizing `L`, cascades such as

\[
B\to L\to S
\]

and

\[
B\to L\to L\to S
\]

can produce higher-order and cross-scale structure in observed B/S even if the complete 3-channel model is linear Hawkes. Therefore compare a true latent-channel model directly with M2 before claiming genuine state-dependent excitation.

## 9. Continuous stochastic latent state

This is a distinct extension from latent events.

A pure continuous mode:

\[
dz_k=-\beta_kz_kdt+L_kdW_k.
\]

A mixed event + diffusion mode:

\[
dx_k=-\beta_kx_kdt+B_kdN+L_kdW_k.
\]

If a Hawkes state and OU state share exactly the same decay operator, they can be merged into one mixed state. If their timescales differ, their sum is not a closed 1-D Markov state and they must remain distinct.

The latent event channel and diffusion noise are not equivalent:

- `dN_L`: discrete hidden impulses / clustered hidden events
- `dW`: continuous stochastic variation

Test the diffusion extension only after M0/M2/ML/M2L are understood.

## 10. Recommended nested models

Use the following hierarchy:

### M0

Optimized 2-channel multiscale Hawkes.

### M2

M0 + slow-state-dependent excitation amplitude.

### ML

True 3-channel Hawkes with latent `L`; otherwise linear.

### M2L

M2 + true latent event channel.

### M2LZ

M2L + continuous stochastic latent excitation / mixed diffusion modes.

This hierarchy lets us test whether nonlinear structure and latent events are substitutes or additive mechanisms.

## 11. Previous empirical results to reproduce/check

These results are context, not ground truth. Some older runs were approximate/staged.

### Stable two-stage Hawkes + latent-background model

Using 1s B/S counts and a stable normalized-trace Hawkes plus causal latent residual-state filter:

| horizon | Frozen MAE | Multi-latent MAE | gain |
|---|---:|---:|---:|
| 1s | 8.1587 | 8.1045 | 0.67% |
| 5s | 28.7979 | 28.5142 | 0.99% |
| 15s | 64.0196 | 62.9863 | 1.61% |
| 1m | 198.122 | 189.177 | 4.52% |
| 5m | 917.778 | 824.908 | 10.12% |
| 15m | 3171.671 | 2751.426 | 13.25% |
| 1h | 16265.527 | 13851.665 | 14.84% |

At 1h, activity correlation actually dropped slightly while MAE/R² improved, suggesting calibration/level correction rather than better regime ranking.

This was a two-stage approximation, not joint latent-state maximum likelihood.

### Independent-scale staged Hawkes selection

Approximate selected scales before continuous refinement:

- `BB`: ~1.73s, 136s, 3600s
- `S→B`: ~136s, 1208s
- `B→S`: ~235s
- `SS`: ~1s, 2.98s, 26.5s, 136s, 405s, 3600s

Do not interpret the one-scale B→S result as final; proper model-order refitting is still needed.

### Continuous timescale refinement of that staged model

Validation objective improved from about `-18.0984` to `-18.1932`.

Optimized scales found:

- `BB`: ~1.16s, 75.6s, 13548s (~3.76h)
- `S→B`: ~25.0s, 11009s (~3.06h)
- `B→S`: ~191s
- `SS`: ~1.00s, 5.43s, 65.3s, 165s, 332s, 7625s (~2.12h)

Weakness: that refinement did not enforce stability correctly for the unnormalized trace convention. Refit before using these as final parameters.

### 5-second cross-scale structural screening

Latest clean screening result:

| model | validation B+S Poisson loss | test B+S loss | test total 5s MAE |
|---|---:|---:|---:|
| M0 linear | -204.220 | -106.086 | 29.422 |
| M2 modulation | -205.337 | -106.300 | 29.277 |
| M3 broader nonlinear | -205.648 | -106.430 | 29.306 |

M2 improved 5s total MAE by ~0.50%. M3 improved likelihood further but not MAE.

This used finite-bank traces and feature proxies, not a final continuous-time nonlinear point process.

### Residual-proxy latent-channel screening

| model | validation loss | test loss | test total 5s MAE |
|---|---:|---:|---:|
| 2ch | -204.220 | -106.086 | 29.422 |
| proxy L K=1 | -204.631 | -106.257 | 29.263 |
| proxy L K=2 | -204.632 | -106.257 | 29.264 |
| proxy L K=3 | -204.633 | -106.258 | 29.264 |

Most gain came from a ~10s proxy latent mode. Again: this is **not** a true latent Hawkes channel.

### M2 + residual-proxy L

| model | validation loss | test loss | test total 5s MAE |
|---|---:|---:|---:|
| M0 | -204.220 | -106.086 | 29.422 |
| M2 | -205.337 | -106.300 | 29.277 |
| M0 + proxy L | -204.641 | -106.241 | 29.302 |
| M2 + proxy L | -205.562 | -106.314 | 29.443 |

The proxy `L` and M2 were largely competing for the same conditional-mean structure. `M2+L` improved likelihood a little but worsened MAE. This motivates a proper generative latent-channel fit.

## 12. Observed memory shape

Raw total-count activity ACF was long-memory-like:

- 1s: ~0.436
- 5s: ~0.317
- 1m: ~0.182
- 5m: ~0.128
- 15m: ~0.104
- 1h: ~0.071
- 3h: ~0.039
- 6h: ~0.035

A single exponential fits poorly. Multiple exponentials, power-law-like kernels, stretched/tempered laws fit much better.

Directional B-S imbalance memory was much shorter.

Important confound: raw count covariance mixes Hawkes excitation and latent exogenous intensity. Do not identify it directly with the latent covariance kernel.

## 13. Approximation / realization result

For raw activity ACF over 1s–6h, optimized exponential mixtures achieved approximately:

- K=1: log-kernel RMSE 0.369
- K=2: 0.140
- K=3: 0.079
- K=4: 0.057
- K=5: 0.055
- K=6: 0.053

A 4-state optimized exponential realization was roughly as good as a fixed 10-state log grid.

This supports exponential-state realization as a computationally efficient basis even when the apparent law is power-law-like.

## 14. Implementation guidance

### Binned first

Use 1s counts for the main architecture search. This makes latent-count inference and long-horizon forecasting tractable.

### Event-level later

After selecting an architecture, repeat the final comparison on raw timestamps if feasible. Preserve same-ms order and treat zero interarrival time within a millisecond as no observed decay between those ordered events.

### Acceleration

Local compute is available, so prefer compiled/vectorized/JIT code rather than Python loops:

- NumPy/SciPy vectorization
- Numba for recursions
- JAX or PyTorch for differentiable latent inference
- parallel candidate model fits

### Parameterization

Use log-timescales and positivity transforms for positive parameters. Order scales or penalize near-duplicates to avoid redundant modes.

### Identifiability diagnostics

For latent `L` report at least:

- posterior event rate over time
- fraction of bins with non-negligible latent count
- inferred `B→L`, `S→L`, `L→L`, `L→B`, `L→S` integrated masses
- whether L collapses to zero
- whether L duplicates an observed channel
- whether two latent modes collapse to the same timescale
- sensitivity to initialization
- likelihood gain after a complexity penalty / held-out validation

If `L` is not identifiable, say so rather than assigning it a physical interpretation.

## 15. Tests that should exist

- no future leakage in trace construction
- same-ms order preservation at event level
- exponential trace recursion agrees with direct convolution on a toy sequence
- integrated branching matrix calculation agrees with numerical integration
- stability constraint enforced
- forward/backward model-order search can recover a synthetic known mixture
- latent inference can recover a synthetic hidden third channel
- open-loop forecast uses expected future events only and does not read future observations
- train/validation/test slicing is immutable

## 16. Final decision criteria

Do not choose a model by one-step likelihood alone. Prefer the smallest model that gives reproducible validation/test improvements in both probabilistic fit and useful open-loop horizons.

The key scientific decision is:

Does a true hidden event stream `L` explain the apparent cross-scale/nonlinear structure, or does state-dependent excitation remain necessary even after `L` is inferred?

The decisive comparison is therefore:

`M0 vs M2 vs ML vs M2L`, with model order and timescales fairly optimized for every candidate.
