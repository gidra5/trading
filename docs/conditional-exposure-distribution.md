# Eight-parameter quadratic distribution with hard survival cutoffs

The production policy is a compact conditional density for target exposure
`a` given current exposure `x`. Its learned raw vector is

```text
[c1, c2, b, lambda, betaC1, betaC2, cutoffLower, cutoffUpper]
```

The fee transition `betaX` and the three transition sharpnesses are decoder
values rather than learned coordinates.

## Score

Let `[A-, A+]` be maximum effective support, `[V-, V+]` executable usable
support, and `m=(A-+A+)/2`. Define

\[
\operatorname{SP}_\kappa(z)=\frac{\log(1+e^{\kappa z})}{\kappa}.
\]

Inside latent support the unnormalized log score is

\[
\begin{aligned}
\ell(a,x)={}&b(a-A_-)-\frac{\lambda}{2}(a-m)^2\\
&+\beta_{c_1}\operatorname{SP}_{\kappa_{c_1}}(a-c_1)\\
&+\beta_x\operatorname{SP}_{\kappa_x}(a-x)\\
&+\beta_{c_2}\operatorname{SP}_{\kappa_{c_2}}(a-c_2).
\end{aligned}
\]

The fixed compact envelope is replaced by learned hard survival bounds `[L,U]`.
The score is used only inside both `[V-,V+]` and `[L,U]`; probability is exactly
zero elsewhere. The policy is

\[
p(a\mid x)=\frac{\mathbf 1\{a\in[V_-,V_+]\cap[L,U]\}\exp\ell(a,x)}
{\sum_u\mathbf 1\{u\in[V_-,V_+]\cap[L,U]\}\exp\ell(u,x)}.
\]

The action derivative is

\[
\partial_a\ell=b-\lambda(a-m)
+\beta_{c_1}\sigma(\kappa_{c_1}(a-c_1))
+\beta_x\sigma(\kappa_x(a-x))
+\beta_{c_2}\sigma(\kappa_{c_2}(a-c_2)).
\]

Thus `lambda` models smooth slope drift. Each beta only models a localized
slope change and no longer needs to absorb global curvature.

## Raw decoder

The two location outputs are ordered logistic coordinates:

\[
c_1=A_-+(A_+-A_-)\sigma(r_1),\qquad
c_2=c_1+(A_+-c_1)\sigma(r_2).
\]

For effective span `S=A+-A-` and half span `h=S/2`:

\[
b=r_b/S,\quad \lambda=r_\lambda/h^2,\quad
\beta_{c_1}=r_{\beta_1}/S,\quad
\beta_{c_2}=r_{\beta_2}/S.
\]

The fee-derived moving change is evaluated at zero exposure:

\[
\beta_x=-\frac{f/(1-f)+f}{\tau},
\]

where `f` is one-way friction and `tau` is policy temperature. The cutoff
coordinates decode around the always-feasible zero action:

```text
L = A- + (0 - A-) sigmoid(rL)
U = A+ sigmoid(rU)
```

Raw endpoints decode exactly to the effective limits. Current calibration fixes

\[
\kappa_{c_1}=82/S,\qquad \kappa_x=678/S,\qquad
\kappa_{c_2}=82/S.
\]

For the standard effective/usable supports `[-250,250]` and `[-100,100]`, the
score basis is fitted over `[-250,250]`; usable truncation is applied only when
the policy is executed.

## Teacher fit

For each current-exposure row, start with target log probability or
`-regret/temperature`. Subtract the known moving fee hinge. With `c1/c2` fixed,
solve the weighted linear design

\[
y_n=\alpha(x)+b(a_n-A_-)-\frac{\lambda}{2}(a_n-m)^2
+\beta_{c_1}H_{c_1}(a_n)+\beta_{c_2}H_{c_2}(a_n).
\]

The teacher uses the complete effective action grid during this fit. Separately,
it simulates the mandatory `H`-step equity and maintenance recursion. Any
liquidation makes the action infeasible. Because zero is feasible, each boundary
is refined by bisection against zero and encoded directly; BFGS optimizes only
the first six smooth score coordinates.

The per-row nuisance intercept `alpha(x)` is eliminated by weighted centering.
The remaining 4x4 normal system is solved in float64 while the large CUDA
reductions remain float32. Several deterministic `c1/c2` starts are projected;
only the best start per example enters batched BFGS cross-entropy refinement.

The screenshot slope correction follows directly. If `s_j` is average log
slope in segment `j` and `aBar_j` is its midpoint, then

\[
s_j=b-\lambda(\bar a_j-m)+\sum_{i\le j}\beta_i,
\]

so adjacent localized changes are initialized as

\[
\beta_j=s_j-s_{j-1}+\lambda(\bar a_j-\bar a_{j-1}).
\]

This prevents the smooth `-lambda*a` derivative from being counted twice in a
beta transition.

## Continuity

The formula is continuous as `x` crosses `c1` or `c2`; breakpoint ordering is
irrelevant because the score is a sum. This algebraic continuity does not imply
that independently fitted raw parameters will be temporally smooth. The dataset
fitter therefore generates quality-equivalent warm-start candidates from each
preceding timestamp and selects the globally lowest-jump path between warm and
independent fits. The executable-surface cross-entropy guard is applied before
continuity selection, so smoothing cannot compensate for a materially worse
useful distribution. Optimization can still use the complete effective domain;
the diagnostic re-normalizes action rows and averages current states only inside
the configured visible interval.
On the retained 1,440-timestamp sequence the optimized branchless fitter reduced
median normalized parameter step by `18.9%`, 90th-percentile step by `3.8%`, and
median normalized second difference by `13.4%`, while slightly improving mean KL
and probability MSE over independent fitting.

## Required checks

1. Every row normalizes to one and is exactly zero outside usable support or the learned survival interval.
2. TypeScript and PyTorch decoders agree on all eight raw coordinates.
3. `betaX` changes when fee or temperature changes but is never emitted by the MLP.
4. Quadratic slope differences equal `-lambda*(a-m)` using the effective-range center.
5. Teacher acceptance records cross-entropy, KL, probability MSE, iterations,
   restarts, and convergence; rejected timestamps go to the refinement queue.
6. Pipelined and sequential multi-batch runs produce byte-identical teacher
   parameters and diagnostics for the same inputs.
7. Six-coordinate artifacts retain visible-span scaling and default to the full effective survival interval.
