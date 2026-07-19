# Conditional Four-Segment Exponential Model with Log-Slope-Change Parameters

This document extends [Four-Segment Smooth Exponential Density with Strict Compact Support](./four-segment-compact-support-exponential.md) to a conditional density in \(a\) given \(x\).

The construction uses the base document's soft exponential transitions and smooth compact-support envelope, but changes the parameter interpretation:

\[
\boxed{\text{Each }\beta\text{ is a change in log-slope at a breakpoint, not an absolute segment slope.}}
\]

This makes the formula independent of the order of the fixed breakpoints \(c_1,c_2\) and the moving breakpoint \(a=x\).

The latent density may live on a wide interval such as \([-250,250]\), while only its hard truncation to a smaller interval such as \([-100,100]\) is visible. No smoothing is applied at the visible truncation points.

## 1. Domain assumptions

Let

\[
a\in[A_-,A_+],
\qquad
x\in[X_-,X_+]
\]

describe the latent model domain. Let

\[
[V_-,V_+]\subset(A_-,A_+)
\]

be the visible observation window. For example,

\[
[A_-,A_+]=[-250,250],
\qquad
[V_-,V_+]=[-100,100].
\]

The fixed breakpoints satisfy

\[
A_-<c_1<c_2<A_+.
\]

The moving breakpoint is located at \(a=x\). If it must always lie inside the latent support, require

\[
x\in(A_-,A_+).
\]

There is no required ordering between \(x\), \(c_1\), and \(c_2\). In particular, all of

\[
x<c_1<c_2,
\qquad
c_1<x<c_2,
\qquad
c_1<c_2<x
\]

are valid.

The fixed breakpoints may lie inside or outside the visible observation window, provided they lie inside the latent support. A transition outside the visible window can still affect the visible tail.

## 2. Soft transitions

Define the scaled softplus and sigmoid

\[
\operatorname{SP}_{\kappa}(z)
=\frac{1}{\kappa}\log(1+e^{\kappa z}),
\qquad
s_{\kappa}(z)
=\frac{1}{1+e^{-\kappa z}}.
\]

They satisfy

\[
\frac{d}{dz}\operatorname{SP}_{\kappa}(z)
=s_{\kappa}(z).
\]

Consequently, a term

\[
\beta\operatorname{SP}_{\kappa}(a-d)
\]

changes the log-slope smoothly by \(\beta\) around \(a=d\):

\[
\frac{\partial}{\partial a}
\left[
\beta\operatorname{SP}_{\kappa}(a-d)
\right]
=\beta s_{\kappa}(a-d).
\]

Far to the left of \(d\), the contribution to the log-slope is approximately zero. Far to the right, it is approximately \(\beta\). This is why \(\beta\) is naturally interpreted as a log-slope change.

## 3. Parameters

For each \(x\), define:

- \(b(x)\in\mathbb R\): the baseline log-slope before any breakpoint has been crossed;
- \(\beta_{c_1}(x)\in\mathbb R\): log-slope change at \(a=c_1\);
- \(\beta_x(x)\in\mathbb R\): log-slope change at the moving breakpoint \(a=x\);
- \(\beta_{c_2}(x)\in\mathbb R\): log-slope change at \(a=c_2\);
- \(\kappa_{c_1}(x),\kappa_x(x),\kappa_{c_2}(x)>0\): transition sharpnesses.

All \(\beta\) parameters are unrestricted real numbers:

- \(\beta_d>0\) increases the log-slope when crossing \(d\);
- \(\beta_d<0\) decreases it;
- \(\beta_d=0\) removes that transition.

The baseline \(b\) is also unrestricted. Only the \(\kappa\) parameters require positivity.

## 4. Order-independent latent log-kernel

Define

\[
\begin{aligned}
h(a,x)={}&b(x)(a-A_-)\\
&+\beta_{c_1}(x)
  \operatorname{SP}_{\kappa_{c_1}(x)}(a-c_1)\\
&+\beta_x(x)
  \operatorname{SP}_{\kappa_x(x)}(a-x)\\
&+\beta_{c_2}(x)
  \operatorname{SP}_{\kappa_{c_2}(x)}(a-c_2).
\end{aligned}
\]

Let \(G(a;A_-,A_+)\) be the strict compact-support envelope from the base document:

\[
G(a)>0\quad\text{for }A_-<a<A_+,
\qquad
G(a)=0\quad\text{otherwise}.
\]

The complete latent log-kernel is

\[
\ell(a,x)=h(a,x)+\log G(a;A_-,A_+).
\]

Its exact log-slope is

\[
\boxed{
\begin{aligned}
\partial_a\ell(a,x)={}&b(x)\\
&+\beta_{c_1}(x)s_{\kappa_{c_1}(x)}(a-c_1)\\
&+\beta_x(x)s_{\kappa_x(x)}(a-x)\\
&+\beta_{c_2}(x)s_{\kappa_{c_2}(x)}(a-c_2)\\
&+\partial_a\log G(a;A_-,A_+).
\end{aligned}}
\]

This formula does not depend on how \(x,c_1,c_2\) are ordered.

## 5. Recovering the actual segment slopes

Away from smooth transition zones and the true support boundaries, replace each sigmoid by a step function. The approximate log-slope is then

\[
\boxed{
s(a,x)=b(x)
+\beta_{c_1}(x)\mathbf 1\{a>c_1\}
+\beta_x(x)\mathbf 1\{a>x\}
+\beta_{c_2}(x)\mathbf 1\{a>c_2\}.}
\]

Thus the rate in a region is the baseline plus all slope changes whose breakpoints have already been crossed.

### Case 1: \(x<c_1<c_2\)

\[
\begin{array}{c|c}
\text{Region} & \text{Approximate log-slope}\\
\hline
a<x & b\\
x<a<c_1 & b+\beta_x\\
c_1<a<c_2 & b+\beta_x+\beta_{c_1}\\
a>c_2 & b+\beta_x+\beta_{c_1}+\beta_{c_2}
\end{array}
\]

### Case 2: \(c_1<x<c_2\)

\[
\begin{array}{c|c}
\text{Region} & \text{Approximate log-slope}\\
\hline
a<c_1 & b\\
c_1<a<x & b+\beta_{c_1}\\
x<a<c_2 & b+\beta_{c_1}+\beta_x\\
a>c_2 & b+\beta_{c_1}+\beta_x+\beta_{c_2}
\end{array}
\]

### Case 3: \(c_1<c_2<x\)

\[
\begin{array}{c|c}
\text{Region} & \text{Approximate log-slope}\\
\hline
a<c_1 & b\\
c_1<a<c_2 & b+\beta_{c_1}\\
c_2<a<x & b+\beta_{c_1}+\beta_{c_2}\\
a>x & b+\beta_{c_1}+\beta_{c_2}+\beta_x
\end{array}
\]

There are always four nominal segments when the three breakpoints are distinct. Their absolute slopes are derived quantities rather than model parameters.

## 6. Relation to an absolute-rate parameterization

When \(c_1<x<c_2\), suppose the four absolute segment slopes are denoted by \(r_1,r_2,r_3,r_4\). Then

\[
\begin{aligned}
b&=r_1,\\
\beta_{c_1}&=r_2-r_1,\\
\beta_x&=r_3-r_2,\\
\beta_{c_2}&=r_4-r_3.
\end{aligned}
\]

Conversely,

\[
\begin{aligned}
r_1&=b,\\
r_2&=b+\beta_{c_1},\\
r_3&=b+\beta_{c_1}+\beta_x,\\
r_4&=b+\beta_{c_1}+\beta_x+\beta_{c_2}.
\end{aligned}
\]

Therefore the jump parameterization has exactly the same number of degrees of freedom as four absolute rates. Its advantage is that \(\beta_x\) remains attached to the moving transition when \(x\) crosses a fixed breakpoint.

## 7. Latent and hard-truncated visible PDFs

The latent normalizer is

\[
Z_{\mathrm{lat}}(x)
=\int_{A_-}^{A_+}\exp(\ell(u,x))\,du.
\]

The latent conditional PDF is

\[
q(a\mid x)=
\begin{cases}
\displaystyle
\frac{\exp(\ell(a,x))}{Z_{\mathrm{lat}}(x)},
&A_-<a<A_+,\\[1.2ex]
0,&\text{otherwise}.
\end{cases}
\]

Only \([V_-,V_+]\) is observed. Define

\[
Z_{\mathrm{vis}}(x)
=\int_{V_-}^{V_+}\exp(\ell(u,x))\,du.
\]

The visible conditional density is

\[
\boxed{
p_{\mathrm{vis}}(a\mid x)=
\begin{cases}
\displaystyle
\frac{\exp(\ell(a,x))}{Z_{\mathrm{vis}}(x)},
&V_-\le a\le V_+,\\[1.2ex]
0,&\text{otherwise as an observed density}.
\end{cases}}
\]

There is no support envelope or smoothed transition at \(V_-\) or \(V_+\). The latent density continues smoothly through them. The visible zero outside the window is caused solely by the hard observation rule.

The truncation normalizer does not depend on \(a\), so it changes the conditional height but not any log-slope or stationary point inside the visible interval.

## 8. Parameterizing the moving slope change

For a peak-like moving transition, use

\[
\boxed{
\beta_x(x)=-2r(x),
\qquad
r(x)\ge0.}
\]

Thus the log-slope decreases by \(2r\) when crossing \(a=x\).

Define the background log-slope at \(a=x\), excluding the moving transition but including both fixed transitions and the support envelope:

\[
\begin{aligned}
B(x)={}&b(x)
+\beta_{c_1}(x)u_1(x)
+\beta_{c_2}(x)u_2(x)
+g_x,
\end{aligned}
\]

where

\[
u_1(x)=s_{\kappa_{c_1}(x)}(x-c_1),
\qquad
u_2(x)=s_{\kappa_{c_2}(x)}(x-c_2),
\]

and

\[
g_x=left.\partial_a\log G(a;A_-,A_+)\right|_{a=x}.
\]

Because the moving sigmoid equals \(1/2\) at its center,

\[
\left.\partial_a\ell(a,x)\right|_{a=x}
=B(x)+\frac{\beta_x(x)}{2}
=B(x)-r(x).
\]

Introduce a local tilt parameter

\[
t(x)=B(x)-r(x).
\]

Then the approximate local slopes on the two sides of the moving transition are

\[
s_-(x)\approx t(x)+r(x),
\qquad
s_+(x)\approx t(x)-r(x),
\]

provided the other transitions vary little over the width of the moving transition.

This recovers the useful tilt-strength interpretation without treating the neighboring segment slopes as primary \(\beta\) parameters.

## 9. Stationary-point condition at \(a=x\)

If \(a=x\) must be a stationary point of the conditional slice, require

\[
\left.\partial_a\ell(a,x)\right|_{a=x}=0.
\]

In the jump parameterization this is simply

\[
\boxed{
B(x)+\frac{\beta_x(x)}{2}=0.}
\]

Equivalently,

\[
\boxed{
\beta_x(x)=-2B(x).}
\]

Under \(\beta_x=-2r\), the condition becomes

\[
B(x)=r(x),
\qquad
t(x)=0.
\]

An exact and convenient parameterization is therefore

\[
\boxed{
b(x)=t(x)+r(x)
-\beta_{c_1}(x)u_1(x)
-\beta_{c_2}(x)u_2(x)
-g_x,}
\]

together with

\[
\beta_x(x)=-2r(x).
\]

This construction guarantees

\[
\left.\partial_a\ell(a,x)\right|_{a=x}=t(x).
\]

Set \(t(x)=0\) when \(a=x\) must follow a stationary ridge. Fit \(t(x)\) freely when the moving transition is allowed to have a nonzero local slope.

A stationary point is a local mode only if

\[
\left.\partial_a^2\ell(a,x)\right|_{a=x}<0.
\]

If \(a=x\) is merely the location of a moving rate change rather than a mode, fit \(b(x)\) and \(\beta_x(x)\) freely and do not impose the stationary condition.

## 10. Curvature and local width

The curvature contributed at its center by a transition with slope change \(\beta\) is

\[
\frac{\beta\kappa}{4}.
\]

For \(\beta_x=-2r\), the moving contribution is

\[
-\frac{r(x)\kappa_x(x)}{2}.
\]

If other curvature terms are small, matching this to the curvature \(-1/s_d^2\) of a Gaussian log-density gives

\[
\boxed{
s_d^2(x)\approx\frac{2}{r(x)\kappa_x(x)},
\qquad
\kappa_x(x)\approx\frac{2}{r(x)s_d^2(x)}.}
\]

The full curvature at \(a=x\) is

\[
\begin{aligned}
C_x={}&
\beta_{c_1}\kappa_{c_1}u_1(1-u_1)
+\beta_{c_2}\kappa_{c_2}u_2(1-u_2)\\
&+\frac{\beta_x\kappa_x}{4}
+\left.\partial_a^2\log G(a)\right|_{a=x}.
\end{aligned}
\]

For an exact local mode, require \(C_x<0\).

## 11. Transition widths

The distance over which a sigmoid transition rises from \(0.1\) to \(0.9\) is

\[
w_{10\text{--}90}
=\frac{2\log 9}{\kappa}
\approx\frac{4.394}{\kappa}.
\]

Hence

\[
\kappa\approx\frac{4.394}{w_{10\text{--}90}}.
\]

This applies to the transitions at \(c_1\), \(x\), and \(c_2\). It does not apply to \(V_-\) and \(V_+\), because those are hard observation cutoffs with no transition width.

## 12. Smooth parameter functions of \(x\)

Scale the conditioning coordinate to

\[
\xi(x)=\frac{2x-X_--X_+}{X_+-X_-}\in[-1,1].
\]

A compact starting model is

\[
\begin{aligned}
\beta_{c_1}(x)
&=p_{10}+p_{11}\xi+p_{12}\xi^2,\\
\beta_{c_2}(x)
&=p_{20}+p_{21}\xi+p_{22}\xi^2,\\
t(x)&=t_0+t_1\xi+t_2\xi^2,\\
r(x)&=r_{\min}
+\operatorname{softplus}(r_0+r_1\xi+r_2\xi^2),\\
s_d(x)&=s_{\min}
+\operatorname{softplus}(s_0+s_1\xi+s_2\xi^2).
\end{aligned}
\]

Then set

\[
\beta_x(x)=-2r(x),
\qquad
\kappa_x(x)=\frac{2}{r(x)s_d^2(x)}.
\]

For an exact tilt parameterization, derive \(b(x)\) from

\[
b=t+r-\beta_{c_1}u_1-\beta_{c_2}u_2-g_x.
\]

Otherwise, fit \(b(x)\) independently as a polynomial or spline.

Parameterize the fixed-transition sharpnesses by

\[
\kappa_{c_j}(x)=\kappa_{\min}
+\operatorname{softplus}(q_{j0}+q_{j1}\xi+q_{j2}\xi^2),
\qquad j\in\{1,2\}.
\]

Begin with constant or linear functions. Add quadratic terms or smooth splines only when supported by held-out likelihood.

## 13. What happens when breakpoints cross

No formula changes when \(x\) crosses \(c_1\) or \(c_2\). The three softplus terms are additive and remain attached to their own transition locations.

At \(x=c_j\), the moving and fixed transitions are co-centered. If their sharpnesses are equal,

\[
\beta_x\operatorname{SP}_{\kappa}(a-x)
+\beta_{c_j}\operatorname{SP}_{\kappa}(a-c_j)
=
(\beta_x+\beta_{c_j})
\operatorname{SP}_{\kappa}(a-c_j).
\]

Thus their effective slope changes add. If their sharpnesses differ, they form two co-centered transitions with different widths. The density is still smooth and valid.

Near a crossing, \(\beta_x\) and \(\beta_{c_j}\) may be difficult to identify separately because their effects overlap. This is an estimation issue, not a validity issue. Smooth parameter functions and regularization help prevent the two terms from canceling with unnecessarily large magnitudes.

## 14. JavaScript implementation

The following implementation assumes that the base document supplies `logCompactEnvelope` for the true latent endpoints.

```js
function sigmoid(z) {
  if (z >= 0) {
    const e = Math.exp(-z);
    return 1 / (1 + e);
  }
  const e = Math.exp(z);
  return e / (1 + e);
}

function softplus(z) {
  if (z > 35) return z;
  if (z < -35) return Math.exp(z);
  return Math.log1p(Math.exp(z));
}

function scaledSoftplus(z, kappa) {
  return softplus(kappa * z) / kappa;
}

function positive(raw, floor = 1e-8) {
  return floor + softplus(raw);
}

function latentLogKernel(a, x, p) {
  if (!(a > p.latentLower && a < p.latentUpper)) {
    return -Infinity;
  }

  const logGate = logCompactEnvelope(
    a,
    p.latentLower,
    p.latentUpper,
    p.supportParams
  );

  return (
    p.baseSlope * (a - p.latentLower)
    + p.betaC1 * scaledSoftplus(a - p.c1, p.kappaC1)
    + p.betaX  * scaledSoftplus(a - x,    p.kappaX)
    + p.betaC2 * scaledSoftplus(a - p.c2, p.kappaC2)
    + logGate
  );
}
```

Here `betaC1`, `betaX`, and `betaC2` are log-slope changes. They are not absolute segment rates.

### Parameter evaluation

```js
function linear(c, z) {
  return c[0] + c[1] * z;
}

function parametersAtX(x, theta) {
  const xi = (
    (2 * x - theta.xLower - theta.xUpper)
    / (theta.xUpper - theta.xLower)
  );

  const baseSlope = linear(theta.baseSlopeCoef, xi);
  const betaC1 = linear(theta.betaC1Coef, xi);
  const betaX = linear(theta.betaXCoef, xi);
  const betaC2 = linear(theta.betaC2Coef, xi);

  const kappaC1 = positive(theta.kappaC1Raw, theta.minKappa);
  const kappaX = positive(theta.kappaXRaw, theta.minKappa);
  const kappaC2 = positive(theta.kappaC2Raw, theta.minKappa);

  return {
    latentLower: theta.latentLower,
    latentUpper: theta.latentUpper,
    visibleLower: theta.visibleLower,
    visibleUpper: theta.visibleUpper,
    c1: theta.c1,
    c2: theta.c2,
    baseSlope,
    betaC1,
    betaX,
    betaC2,
    kappaC1,
    kappaX,
    kappaC2,
    supportParams: theta.supportParams
  };
}
```

This free form is the default fitted model because a moving slope change need not be a stationary ridge. When exact centering is known to be appropriate, use the constrained construction from Section 9 instead.

### Visible hard truncation

```js
function visibleLogKernel(a, x, p) {
  if (a < p.visibleLower || a > p.visibleUpper) {
    return -Infinity;
  }
  return latentLogKernel(a, x, p);
}
```

Do not pass `visibleLower` or `visibleUpper` to `logCompactEnvelope`.

## 15. Conditional normalization

For a stable numerical integral, subtract the largest sampled log-kernel before exponentiation:

```js
function logIntegralAtX(x, p, nodes, weights) {
  const values = nodes.map(a => latentLogKernel(a, x, p));
  const m = Math.max(...values);

  let sum = 0;
  for (let i = 0; i < nodes.length; i++) {
    sum += weights[i] * Math.exp(values[i] - m);
  }

  return m + Math.log(sum);
}

function visibleLogPdf(a, x, theta, visibleNodes, visibleWeights) {
  const p = parametersAtX(x, theta);

  if (a < p.visibleLower || a > p.visibleUpper) {
    return -Infinity;
  }

  const logZVisible = logIntegralAtX(
    x,
    p,
    visibleNodes,
    visibleWeights
  );

  return latentLogKernel(a, x, p) - logZVisible;
}
```

For the visible truncated likelihood, map the quadrature nodes to \([V_-,V_+]\). To evaluate the full latent PDF, use a separate quadrature rule on \([A_-,A_+]\).

## 16. Likelihood for truncated observations

For observed pairs \((a_n,x_n)\) with \(a_n\in[V_-,V_+]\), minimize

\[
\mathcal L(\theta)
=-\sum_{n=1}^N
\left[
\ell(a_n,x_n;\theta)
-\log Z_{\mathrm{vis}}(x_n;\theta)
\right]
+\mathcal R(\theta).
\]

If only accepted observations are available, this is the correct likelihood. If the numbers of observations rejected outside the window are also known, additionally model

\[
P_{\mathrm{vis}}(x)
=\frac{Z_{\mathrm{vis}}(x)}{Z_{\mathrm{lat}}(x)}.
\]

Useful regularization includes:

- penalties on rapid variation of \(b(x)\), the \(\beta(x)\) functions, and the \(\kappa(x)\) functions;
- penalties on unnecessarily large, canceling slope changes;
- a centering penalty \(\lambda\mathbb E_x[t(x)^2]\) when \(a=x\) is expected to be approximately rather than exactly stationary.

## 17. Initialization

For several representative values of \(x\):

1. Estimate approximate absolute log-slopes in each visible region away from transitions.
2. Set \(b\) to the slope before the leftmost breakpoint.
3. At every breakpoint, initialize its \(\beta\) as

   \[
   \beta_d
   =\text{slope immediately to the right of }d
   -\text{slope immediately to the left of }d.
   \]

4. Attach the change at the moving breakpoint to \(\beta_x\), regardless of where \(x\) occurs relative to \(c_1,c_2\).
5. If a peak is intended at \(a=x\), initialize

   \[
   r=-\frac{\beta_x}{2}>0
   \]

   and estimate \(\kappa_x\approx2/(rs_d^2)\).
6. Estimate each fixed \(\kappa\) from its transition width using \(\kappa\approx4.394/w_{10\text{--}90}\).
7. Regress the slice-wise jump estimates on scaled \(x\) to initialize the smooth parameter functions.
8. Run several deterministic starts, including one with both fixed breakpoints outside the visible window.
9. Optimize the full truncated conditional likelihood with analytic gradients and report iteration-limit or line-search termination separately from convergence.

Near \(x=c_1\) or \(x=c_2\), initialize using nearby slices where the two transitions are separated. Co-centered transitions are generally difficult to decompose from a single slice.

## 18. Recommended initial model

Start with:

- fixed latent support, for example \([-250,250]\);
- fixed visible window, for example \([-100,100]\);
- the compact-support envelope only at the latent endpoints;
- independently trainable positive compact-envelope sharpnesses \(\rho_L,\rho_R\) and physical taper widths \(w_L,w_R\), with an ordered positive transform enforcing \(w_L+w_R<A_+-A_-\);
- no smoothing at the visible endpoints;
- globally trainable \(c_1<c_2\) over the entire latent support, fixed with respect to \(x\) and represented by an ordered transform;
- no restriction on whether \(x\) is below, between, or above \(c_1,c_2\);
- independently fitted linear functions of scaled \(x\) for \(b\), \(\beta_{c_1}\), \(\beta_x\), and \(\beta_{c_2}\);
- independently trainable positive constants \(\kappa_{c_1}\), \(\kappa_x\), and \(\kappa_{c_2}\);
- empirical initialization from row-wise visible log-gradients;
- deterministic multi-start optimization of the truncated cross-entropy with analytic gradients;
- a separate visible normalizer for every \(x\).

## 19. Model checks

After fitting, verify:

1. \(p_{\mathrm{vis}}(a\mid x)\) integrates to one on \([V_-,V_+]\) for a dense grid of \(x\).
2. The latent density is exactly zero outside \([A_-,A_+]\).
3. No gate or taper has been applied at \(V_-\) or \(V_+\).
4. Each fitted \(\beta_d\) agrees with the empirical change in log-slope across its transition.
5. Physical segment slopes reconstructed by cumulative addition agree with the observed regional slopes.
6. The formula remains continuous as \(x\) passes through \(c_1\) and \(c_2\).
7. Parameters do not develop large canceling jumps near breakpoint crossings.
8. If \(a=x\) is intended to be stationary, verify

   \[
   B(x)+\frac{\beta_x(x)}{2}=0.
   \]

9. If it is intended to be a mode, also verify that the curvature \(C_x\) is negative.
10. Compare held-out likelihood against simpler parameter functions and a flexible reference estimator.

The central implementation rule is:

\[
\boxed{
\text{actual log-slope}
=\text{baseline slope}
+\sum_{\text{crossed breakpoints}}
\text{fitted }\beta\text{ jump}.}
\]
