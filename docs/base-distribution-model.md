# Four-Segment Smooth Exponential Density with Strict Compact Support

This is the strict-support version of the model. It has:

- support exactly on \([L,U]\);
- four locally exponential interior regimes;
- three ordinary sigmoid transitions between those regimes; and
- compact sigmoid transitions to zero at \(L\) and \(U\).

## 1. Why an ordinary sigmoid is insufficient at the endpoints

The logistic sigmoid

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

satisfies

\[
0<\sigma(z)<1
\]

for every finite \(z\). Therefore multiplying a density by

\[
\sigma\!\left(\kappa(x-L)\right)
\sigma\!\left(\kappa(U-x)\right)
\]

does **not** give strict support on \([L,U]\). It only makes the exterior values small.

To obtain exact compact support while retaining a sigmoid-like smooth transition, use a compact sigmoid smoothstep.

## 2. Compact sigmoid smoothstep

Define

\[
H_\rho(t)=
\begin{cases}
0,&t\le0,\\[4pt]
\displaystyle
\sigma\!\left(
\rho\left[
\frac{1}{1-t}-\frac{1}{t}
\right]
\right),&0<t<1,\\[12pt]
1,&t\ge1,
\end{cases}
\]

where \(\rho>0\) controls sharpness inside the transition layer.

This function has the properties

\[
H_\rho(t)=0\quad(t\le0),
\qquad
H_\rho(t)=1\quad(t\ge1),
\]

and it is \(C^\infty\) at \(t=0\) and \(t=1\). Every derivative approaches zero at both endpoints.

It is also symmetric:

\[
H_\rho(1-t)=1-H_\rho(t).
\]

Increasing \(\rho\) makes the change inside \((0,1)\) sharper.

## 3. Endpoint gates

Choose positive boundary-layer widths \(w_L,w_R\) satisfying

\[
w_L+w_R<U-L.
\]

Define

\[
G_L(x)=H_{\rho_L}\!\left(\frac{x-L}{w_L}\right)
\]

and

\[
G_R(x)=H_{\rho_R}\!\left(\frac{U-x}{w_R}\right).
\]

Then:

- \(G_L(x)=0\) for \(x\le L\);
- \(G_L(x)=1\) for \(x\ge L+w_L\);
- \(G_R(x)=1\) for \(x\le U-w_R\); and
- \(G_R(x)=0\) for \(x\ge U\).

Thus \(G_L(x)G_R(x)\) is exactly zero outside \([L,U]\), smoothly rises from zero near \(L\), and smoothly returns to zero near \(U\).

## 4. Four interior exponential regimes

Choose

\[
L<c_1<c_2<c_3<U
\]

and four log-slopes

\[
\beta_1,\beta_2,\beta_3,\beta_4\in\mathbb R.
\]

Let

\[
\kappa_1,\kappa_2,\kappa_3>0
\]

control the three interior join sharpnesses.

Define

\[
\operatorname{softplus}(z)=\log(1+e^z).
\]

The interior unnormalized log-kernel is

\[
\boxed{
\ell_{\mathrm{int}}(x)
=
\beta_1(x-L)
+
\sum_{j=1}^{3}
\frac{\beta_{j+1}-\beta_j}{\kappa_j}
\operatorname{softplus}\!\left(\kappa_j(x-c_j)\right).
}
\]

Its derivative is

\[
\boxed{
\ell_{\mathrm{int}}'(x)
=
\beta_1
+
\sum_{j=1}^{3}
(\beta_{j+1}-\beta_j)
\sigma\!\left(\kappa_j(x-c_j)\right).
}
\]

Therefore its approximate log-slopes are \(\beta_1,\beta_2,\beta_3,\beta_4\) in the four consecutive interior regions.

## 5. Strictly supported kernel and PDF

Define

\[
g(x)=
e^{\ell_{\mathrm{int}}(x)}G_L(x)G_R(x).
\]

Because the gates are exactly zero outside the interval,

\[
g(x)=0\qquad\text{for }x\notin[L,U].
\]

The normalizing constant is

\[
Z=\int_L^U
e^{\ell_{\mathrm{int}}(t)}G_L(t)G_R(t)\,dt.
\]

The normalized PDF is

\[
\boxed{
f(x)=
\begin{cases}
\displaystyle
\frac{
e^{\ell_{\mathrm{int}}(x)}G_L(x)G_R(x)
}{Z},&L<x<U,\\[14pt]
0,&\text{otherwise}.
\end{cases}
}
\]

This PDF has support exactly \([L,U]\) and is \(C^\infty\) across both endpoints.

## 6. Stable numerical functions

```js
function softplus(z) {
  if (z > 35) return z;
  if (z < -35) return Math.exp(z);
  return Math.log1p(Math.exp(z));
}

function logSigmoid(z) {
  return -softplus(-z);
}

function sigmoid(z) {
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const ez = Math.exp(z);
  return ez / (1 + ez);
}
```

## 7. Compact step implementation

```js
function compactStep(t, rho = 1) {
  if (t <= 0) return 0;
  if (t >= 1) return 1;

  const argument = rho * (
    1 / (1 - t) - 1 / t
  );

  return sigmoid(argument);
}
```

For density calculations, use its logarithm to avoid underflow near the endpoints:

```js
function logCompactStep(t, rho = 1) {
  if (t <= 0) return -Infinity;
  if (t >= 1) return 0;

  const argument = rho * (
    1 / (1 - t) - 1 / t
  );

  return logSigmoid(argument);
}
```

## 8. Interior log-kernel implementation

```js
function interiorLogKernel(x, L, breaks, beta, kappa) {
  // breaks = [c1, c2, c3]
  // beta   = [beta1, beta2, beta3, beta4]
  // kappa  = [kappa1, kappa2, kappa3]

  let value = beta[0] * (x - L);

  for (let j = 0; j < 3; j++) {
    value +=
      (beta[j + 1] - beta[j]) *
      softplus(kappa[j] * (x - breaks[j])) /
      kappa[j];
  }

  return value;
}
```

## 9. Complete compact-support log-kernel

```js
function compactLogKernel(
  x,
  {
    L,
    U,
    breaks,
    beta,
    kappa,
    leftWidth,
    rightWidth,
    leftSharpness = 1,
    rightSharpness = 1
  }
) {
  if (x <= L || x >= U) return -Infinity;

  const leftCoordinate = (x - L) / leftWidth;
  const rightCoordinate = (U - x) / rightWidth;

  return (
    interiorLogKernel(x, L, breaks, beta, kappa) +
    logCompactStep(leftCoordinate, leftSharpness) +
    logCompactStep(rightCoordinate, rightSharpness)
  );
}
```

## 10. Numerical normalization

The compact gates do not lead to a useful elementary normalizing constant. The integral should be evaluated numerically on \([L,U]\).

```js
function buildCompactPdf(parameters) {
  const { L, U, kappa, leftWidth, rightWidth } = parameters;

  if (!(L < U)) {
    throw new Error("L must be smaller than U");
  }

  if (!(leftWidth > 0 && rightWidth > 0)) {
    throw new Error("Boundary widths must be positive");
  }

  if (!(leftWidth + rightWidth < U - L)) {
    throw new Error("Boundary layers must not cover the whole interval");
  }

  const maxKappa = Math.max(...kappa);

  const targetStep = Math.min(
    (U - L) / 2000,
    leftWidth / 100,
    rightWidth / 100,
    1 / (10 * maxKappa)
  );

  let N = Math.ceil((U - L) / targetStep);
  if (N % 2 !== 0) N += 1;

  const h = (U - L) / N;
  const logs = new Float64Array(N + 1);
  let maximum = -Infinity;

  for (let i = 0; i <= N; i++) {
    const x = L + i * h;
    logs[i] = compactLogKernel(x, parameters);
    maximum = Math.max(maximum, logs[i]);
  }

  let scaledSum = 0;

  for (let i = 0; i <= N; i++) {
    const weight =
      i === 0 || i === N ? 1 :
      i % 2 === 1 ? 4 : 2;

    const scaledValue = Number.isFinite(logs[i])
      ? Math.exp(logs[i] - maximum)
      : 0;

    scaledSum += weight * scaledValue;
  }

  const logZ =
    maximum + Math.log(h / 3) + Math.log(scaledSum);

  return {
    logZ,

    logPdf(x) {
      if (x <= L || x >= U) return -Infinity;
      return compactLogKernel(x, parameters) - logZ;
    },

    pdf(x) {
      if (x <= L || x >= U) return 0;
      return Math.exp(
        compactLogKernel(x, parameters) - logZ
      );
    }
  };
}
```

## 11. Example

```js
const distribution = buildCompactPdf({
  L: -4,
  U: 4,

  breaks: [-2, 0.5, 2],
  beta: [1.5, 0.2, -0.5, -1.3],
  kappa: [5, 5, 5],

  leftWidth: 0.5,
  rightWidth: 0.5,

  leftSharpness: 1,
  rightSharpness: 1
});

console.log(distribution.pdf(-4)); // 0
console.log(distribution.pdf(0));
console.log(distribution.pdf(4));  // 0
```

## 12. Interpretation

The parameters have separate roles:

- \(\beta_1,\ldots,\beta_4\): interior exponential log-slopes;
- \(c_1,c_2,c_3\): interior transition locations;
- \(\kappa_1,\kappa_2,\kappa_3\): interior transition sharpnesses;
- \(w_L,w_R\): physical widths of the endpoint taper regions;
- \(\rho_L,\rho_R\): shapes of the compact sigmoid tapers; and
- \(L,U\): exact support boundaries.

The ordinary interior sigmoid joins and compact endpoint joins cannot be literally identical functions: exact zeros require a compact smoothstep. The construction above is the closest strict-support analogue and remains infinitely differentiable at every transition.
