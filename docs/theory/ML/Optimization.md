Suppose we are optimizing a loss function $L(x)$. 

It might be constrained to solutions that must satisfy some kind of $\phi(x)<0$ or $\phi(x)=0$.
The observations might be noisy, with some distribution $\epsilon \approx p_{\epsilon}(x)$ with $\mathbb E_x[\epsilon]=0$
We might assume a certain distribution $X \approx p_X(x)$ of observations as well.
## Local optimization

generally use derivatives to find local minima.

convex loss functions have a properly that any local optimizer is global.

We can use roots of unity or fractional steps to arbitrarily accelerate the convergence, as they remove progressively more degrees.

we can treat optimization as a dynamic system, that evolves according to PDE, then solutions are infinite time realizations of its solution. that allows for things like hamiltonian dynamics.

damped Gauss–Newton and the Levenberg–Marquardt method

**BFGS** (Broyden–Fletcher–Goldfarb–Shanno) is a **quasi-Newton optimization algorithm**. It approximates Newton's method without explicitly computing the Hessian.
## 2. Existing literature

### A. Austin, Kravanja & Trefethen (2014)

**[Numerical Algorithms Based on Analytic Function Values at Roots of Unity](https://doi.org/10.1137/130931035?utm_source=chatgpt.com)**

This is the most directly relevant paper I found.

It studies numerical algorithms that evaluate analytic functions at roots of unity, including:

- Polynomial interpolation.
- Cauchy integral approximations.
- Numerical differentiation.
- Rational approximation.
- Numerical stability and convergence.

The authors show how roots-of-unity sampling relates to interpolation and contour integration. [SIAM](https://epubs.siam.org/doi/10.1137/130931035?utm_source=chatgpt.com)

[Full paper (PDF)](https://faculty.nps.edu/aaustin/pubs/AKT2014.pdf?utm_source=chatgpt.com)

### B. Lyness & Moler (1967)

**[Numerical Differentiation of Analytic Functions](https://doi.org/10.1137/0704019?utm_source=chatgpt.com)**

This is an early foundational treatment of evaluating analytic functions at complex points to obtain numerical derivatives.

The roots-of-unity construction appears naturally when Cauchy integrals are discretized using equally spaced points on a circle. [SIAM](https://epubs.siam.org/doi/10.1137/0704019?utm_source=chatgpt.com)

### C. Lyness (1968)

**[Differentiation Formulas for Analytic Functions](https://doi.org/10.1090/S0025-5718-1968-0230468-5?utm_source=chatgpt.com)**

This extends earlier complex-evaluation methods and studies their truncation errors.

It is particularly relevant because the error can be represented as a convergent series of terms obtained from complex-plane evaluations—the same structure your extrapolation would exploit. [Bishtref](https://bishtref.com/articles/10.1090/s0025-5718-1968-0230468-5?utm_source=chatgpt.com)

### D. Trefethen — Approximation Theory and Approximation Practice

**[Appendix C: ATAP for the Unit Disk](https://epubs.siam.org/doi/abs/10.1137/1.9781611975949.appc?utm_source=chatgpt.com)**

This develops the broader mathematical framework behind algorithms using roots of unity, including extensions beyond polynomial approximation.

It's useful for understanding how far the method generalizes to other functional spaces.
## Global optimization

Generally what we want is a kind of certificate of how close we are to the minimum, preferably in terms of the objective value. That means for every $\epsilon$ we can find $x$ such that its loss is within this distance of global optimum $x^*=\arg\min f(x)$:
$$||f(x^*)-f(x)||<\varepsilon$$
Then the problem is to iteratively improve it.

We can also study any related functions that share optimizers within some bounds. If we specifically want global optimums to match, we dont need arbitrary uniform accuracy between two functions, only that it preserves the global minima specifically:
$$ |\min g-\min f|<\varepsilon \\ |\arg\min g-\arg\min f|<\delta$$

Then if finding optimum of $g$ is simpler in any sense, we can instead solve for its global optimum. Moreover, even if we restrict $g$ to be convex, $f$ still can be arbitrarily nonconvex.

For example, a region could satisfy the Polyak–Łojasiewicz inequality:

\[ \boxed{ \frac12\|\nabla f(x)\|^2 \geq \mu(f(x)-f_i^*) } \]

for some \(\mu>0\), where \(f_i^*\) is the minimum over that region.

This is weaker than strong convexity.

A nonconvex function can satisfy this inequality, allowing gradient descent to converge to a global minimum of the region under appropriate smoothness and feasibility assumptions.

We can separate this problem into broad-phase and narrow-phase:
1. Initially we simply try to narrow down the neighborhood of the optimum - certify the region where it must be.
2. Then search for the region where the global optimum is the attractor.
3. Then we can apply any local method to approach it arbitrarily close.

https://arxiv.org/abs/1803.02021?utm_source=chatgpt.com

basin detection - take f and test point x, run de solution until found the minima, evaluate. If two points converge to the same minima they are the same basin.
### Symmetry
| Approach                                                                                                                                                                                | What it discovers or exploits                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **[AI Feynman](https://pmc.ncbi.nlm.nih.gov/articles/PMC7159912/)**                                                                                                                     | Fits a neural surrogate and tests candidate transformations. For example, if shifting \(x_1\) and \(x_2\) by the same amount leaves the prediction unchanged, it tries rewriting the function in terms of \(x_1-x_2\). Its goal is symbolic regression, but the surrogate → symmetry test → reduced model loop is close to what you described. ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7159912?utm_source=chatgpt.com "AI Feynman: A physics-inspired method for symbolic regression")) |
| **[Gaussian processes with built-in dimensionality reduction](https://arxiv.org/abs/1602.04550)**                                                                                       | Learns a projection \(W^\top x\) from function values. In the exact model \(f(x)=g(W^\top x)\), every translation in \(\ker W^\top\) is a symmetry. This is a way to discover flat directions that aren’t aligned with the original axes. ([arxiv.org](https://arxiv.org/abs/1602.04550?utm_source=chatgpt.com "Gaussian processes with built-in dimensionality reduction: Applications in high-dimensional uncertainty propagation"))                                                          |
| **[A Unified Framework to Enforce, Discover, and Promote Symmetry](https://jmlr.org/papers/v26/24-1315.html)**                                                                          | Studies how to recover continuous symmetry generators from a fitted model or data, and how to fit models that allow the data to break a candidate symmetry. This is the closest reference if you want to search for transformations beyond coordinate shifts. ([jmlr.org](https://jmlr.org/papers/v26/24-1315.html?utm_source=chatgpt.com "A Unified Framework to Enforce, Discover, and Promote Symmetry in Machine Learning"))                                                                |
| **[Sample-efficient Bayesian Optimisation Using Known Invariances](https://proceedings.nips.cc/paper_files/paper/2024/hash/55aeba84b402008d3ed10440d906b4e1-Abstract-Conference.html)** | Builds a GP kernel that shares information across symmetry-related points. It gives sample-complexity bounds for BO when the invariance is supplied in advance. ([proceedings.nips.cc](https://proceedings.nips.cc/paper_files/paper/2024/hash/55aeba84b402008d3ed10440d906b4e1-Abstract-Conference.html?utm_source=chatgpt.com "Sample-efficient Bayesian Optimisation Using Known Invariances"))                                                                                              |
### Polynomial optimization
https://epubs.siam.org/doi/10.1137/S1052623400366802?utm_source=chatgpt.com
https://ir.cwi.nl/pub/25212?utm_source=chatgpt.com

On a bounded interval lets define a continuous function $f:[a,b]\to\mathbb R$.
The Weierstrass approximation theorem states that
$$ \boxed{ \forall\varepsilon>0,\quad \exists p_n\in\mathcal P_n: \|f-p_n\|_\infty<\varepsilon } $$
where
$$ \|f-p_n\|_\infty =\sup_{x\in[a,b]}|f(x)-p_n(x)| $$
For multivariate functions, the Stone–Weierstrass theorem gives an analogous result on compact subsets of $\mathbb R^d$.

Suppose $\|f-p_n\|_\infty\leq\varepsilon$
Let $x^*=\arg\min_x f(x)$ and $\hat x=\arg\min_x p_n(x)$
Then
$$\begin{aligned} f(\hat x) &\leq p_n(\hat x)+\varepsilon\\ &\leq p_n(x^*)+\varepsilon\\ &\leq f(x^*)+2\varepsilon. \end{aligned}$$
Therefore,
$$\boxed{ 0\leq f(\hat x)-f(x^*)\leq2\varepsilon. }$$

**This is a global optimality guarantee, independent of the number of local extrema of $f$.**

If we want an objective error below $\delta$, it suffices to construct a polynomial approximation with $\varepsilon\leq\frac{\delta}{2}$. Then we can then optimize that polynomial globally.

And global optimization of finite degree polynomials is quite simple algorithmically. First, any global optima must have $\nabla f(x)=0$. For polynomials there are at most $d$ such points for $d$ degree polynomial. Thus we need to find all zeroes of the derivative of the polynomial. This can be done recursively through interval splitting and binary search as shown in the [video](https://www.youtube.com/watch?v=R4P1_c4Y0Ec).

Then once we enumerated all extrema, we simply choose the smallest one.

For a function $f$ we can find the optimal polynomial with minimax polynomial approximation through Remez algorithm. Whatever that means.

### Smoothness
https://arxiv.org/html/1703.02628
https://chatgpt.com/share/6ae4a3d4-7504-83ed-bc25-c7da589951b7
If the loss has bounded variation, in a sense that its a L-lipschitz function with:
$$||f(x_2)-f(x_1)||<L||x_1-x_2||$$
then we can have globally optimizable method with accuracy guarantees.
### Stochastic methods
sample random points on the domain and intelligently decide if the solution is global

Things like Monte Carlo, bayesian, Pure Random Opt, Genetic algorithms

Expectations also optimize a loss, which means we can transform any loss into stochastic monte carlo estimator.
https://arxiv.org/abs/1611.01838?utm_source=chatgpt.com
[hj mad](https://arxiv.org/pdf/2202.11014)
Adam, muon
#### Averaging as optimization
https://arxiv.org/abs/2012.12762?utm_source=chatgpt.com
https://arxiv.org/html/2301.10980v2
https://www.jmlr.org/papers/volume6/banerjee05b/banerjee05b.pdf
Fréchet mean is defined through optimization objective:
$$m=\arg\min_x\mathbb E_y[d(x,y)^2]$$
For different distance functions like Euclidean or log distance we get the usual minimizers in the form of arithmetic and geometric mean.

We can formulate our optimization objective $L(x)$ in terms of this mean by transforming

### Deflation technique
We can turn local optimizer into a global one through enumeration of extrema. For polynomials this can be done in finite amount of steps exactly. Moreover we can have an polynomial approximation to a function of a finite degree with uniformly bounded error, which means optimizing it finds actual minima up to the error. Then if the minima is sufficiently localized, we can finish the job with local optimization.

https://doi.org/10.1137/140984798?utm_source=chatgpt.com

### Convexification
We can find a convex loss whose optimum coincides with original ones.
For that we can smooth out with gaussian and sharpen with softmax. in the limit it approaches the minimum.
[On Graduated Optimization for Stochastic Non-Convex Problems](https://proceedings.mlr.press/v48/hazanb16.html?utm_source=chatgpt.com)
2. [A Theoretical Analysis of Optimization by Gaussian Continuation](https://people.csail.mit.edu/hmobahi/?utm_source=chatgpt.com)
3. [On the Link Between Gaussian Homotopy Continuation and Convex Envelopes](https://people.csail.mit.edu/hmobahi/pubs/gaussian_convenv_2015.pdf?utm_source=chatgpt.com)
entropic convexification through affine coupling rather than quadratic proximity
- Graduated nonconvexity (GNC): progressively transform a simple objective into the original nonconvex one.
    
- Gaussian smoothing and continuation: optimize \(F_\sigma=F*\mathcal N(0,\sigma^2I)\) while decreasing \(\sigma\).
    
- Homotopy methods: track solutions of a parameterized family of optimization problems.
    
- Scale-space theory: study how critical points appear, disappear, or move as smoothing changes.
    
- Certified global optimization: use approximation error bounds to ensure that refinement does not exclude the true global optimum. 
- proximal convexification method


Suppose smoothing introduces a uniform approximation error

\[ \sup_x|F_\sigma(x)-F(x)|\le\delta_\sigma. \]

Let \(x_\sigma^*\) globally minimize \(F_\sigma\), and \(x^*\) globally minimize \(F\).

Then

\[ \begin{aligned} F(x_\sigma^*) &\le F_\sigma(x_\sigma^*)+\delta_\sigma\\ &\le F_\sigma(x^*)+\delta_\sigma\\ &\le F(x^*)+2\delta_\sigma. \end{aligned} \]

Therefore,

\[ \boxed{ F(x_\sigma^*)-\min_xF(x)\le2\delta_\sigma. } \]

| Hessian condition                  | Implication                       |
| ---------------------------------- | --------------------------------- |
| $\nabla^2F(x)\succeq0$ everywhere  | Convex                            |
| $\nabla^2F(x)\succ0$ everywhere    | Strictly convex                   |
| $\nabla^2F(x)\succeq\mu I,\ \mu>0$ | $\mu$-strongly convex             |
| $\nabla^2F(x)\succeq-\rho I$       | $\rho$-weakly convex (semiconvex) |
| $\nabla^2F(x)\preceq0$ everywhere  | Concave                           |

Two important caveats:

- The domain must be convex. A positive semidefinite Hessian on a disconnected or nonconvex domain does not generally establish global convexity across the domain.
    
- The condition must hold everywhere, not merely at stationary points. A positive definite Hessian at a stationary point only establishes a strict local minimum.
### 1. DC decomposition

Consider the minimization problem

\[ \min_{x\in C} F(x), \]

where \(C\) is convex and

\[ \boxed{F(x)=g(x)-h(x)} \]

with both \(g\) and \(h\) convex.

Such an objective is called a DC function.

For example,

\[ F(x)=x^4-3x^2 \]

has the decomposition

\[ g(x)=x^4,\qquad h(x)=3x^2. \]

Both components are convex, even though their difference is nonconvex.
#### 2. The DC Algorithm (DCA)

The standard DC algorithm uses the convexity of \(h\).

At the current iterate \(x_t\), choose

\[ u_t\in\partial h(x_t). \]

By convexity,

\[ h(x)\ge h(x_t)+\langle u_t,x-x_t\rangle. \]

Negating gives an upper bound on \(F\):

\[ F(x)\le g(x)-h(x_t)-\langle u_t,x-x_t\rangle. \]

We therefore minimize this convex upper bound:

\[ \boxed{ x_{t+1}\in \arg\min_{x\in C} \left[g(x)-\langle u_t,x\rangle\right]. } \]

Every iteration requires solving a convex optimization problem.

Under standard assumptions, DCA produces nonincreasing objective values:

\[ F(x_{t+1})\le F(x_t). \]

However, DCA does not generally guarantee a global minimum. It typically converges toward a DC-critical point under appropriate conditions.

### Example

Take

\[ f(x)=(x^2-1)^2,\qquad x\in[-2,2]. \]

Its global minima are

\[ x=\pm1. \]

The convex envelope is

\[ g(x)= \begin{cases} 0,& |x|\leq1,\\ (x^2-1)^2,&1<|x|\leq2. \end{cases} \]

It is convex, and

\[ \min g=\min f=0. \]

But now every point in \([-1,1]\) minimizes \(g\).

This illustrates the tradeoff: convexification preserves the optimal value but may lose information about the location of the optimum.
### Consensus based methods

https://arxiv.org/html/2103.15130v4
### Population-based and evolutionary methods
| Algorithm                          |      Year | Main idea                                                             | Original paper                                                                                  |
| ---------------------------------- | --------: | --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Differential Evolution (DE)        |      1997 | Generate candidates from differences between population members       | [Storn & Price (1997)](https://doi.org/10.1023/a%3A1008202821328?utm_source=chatgpt.com)        |
| CMA-ES                             | 1996/2001 | Adapt a multivariate Gaussian search distribution                     | [Hansen & Ostermeier (2001)](https://doi.org/10.1162/106365601750190398?utm_source=chatgpt.com) |
| Particle Swarm Optimization (PSO)  |      1995 | Update candidate positions using individual and collective experience | [Kennedy & Eberhart (1995)](https://doi.org/10.1109/ICNN.1995.488968?utm_source=chatgpt.com)    |
| JADE                               |      2009 | Adaptive differential evolution with an external archive              | [Zhang & Sanderson (2009)](https://doi.org/10.1109/TEVC.2009.2014613?utm_source=chatgpt.com)    |
| SHADE                              |      2013 | Adapt DE parameters using historical successful values                | [Tanabe & Fukunaga (2013)](https://doi.org/10.1109/CEC.2013.6557555?utm_source=chatgpt.com)     |
| L-SHADE                            |      2014 | SHADE with linear population-size reduction                           | [Tanabe & Fukunaga (2014)](https://doi.org/10.1109/CEC.2014.6900380?utm_source=chatgpt.com)     |
| Natural Evolution Strategies (NES) |      2009 | Optimize parameters of a search distribution using natural gradients  | [Wierstra et al. (2009)](https://arxiv.org/abs/1106.4487?utm_source=chatgpt.com)                |
### Surrogate-based optimization
| Algorithm                           | Year | Main idea                                                       | Original paper                                                                                        |
| ----------------------------------- | ---: | --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Efficient Global Optimization (EGO) | 1998 | Gaussian-process surrogate and expected improvement             | [Jones et al. (1998)](https://doi.org/10.1023%2FA%3A1008306431147?utm_source=chatgpt.com)             |
| GP-UCB                              | 2010 | Choose candidates using confidence bounds on a Gaussian process | [Srinivas et al. (2010)](https://arxiv.org/abs/0912.3995?utm_source=chatgpt.com)                      |
| TuRBO                               | 2019 | Bayesian optimization using multiple local trust regions        | [Eriksson et al. (2019)](https://arxiv.org/abs/1910.01739?utm_source=chatgpt.com)                     |
| SMAC                                | 2011 | Sequential model-based optimization using surrogate models      | [Hutter et al. (2011)](https://www.cs.ubc.ca/~hutter/papers/11-LION5-SMAC.pdf?utm_source=chatgpt.com) |
### Deterministic and partition-based methods

| Algorithm                                  |      Year | Main idea                                                          | Original paper                                                                                                                                                       |
| ------------------------------------------ | --------: | ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DIRECT                                     |      1993 | Recursively partition the domain into promising rectangles         | [Jones et al. (1993)](https://dumas.perso.math.cnrs.fr/DIRECT.pdf?utm_source=chatgpt.com)                                                                            |
| Multilevel Coordinate Search (MCS)         |      1999 | Hierarchical partitioning with local search                        | [Huyer & Neumaier (1999)](https://doi.org/10.1006/jcph.1999.6329?utm_source=chatgpt.com)                                                                             |
| SHGO                                       |      2018 | Use simplicial complexes and topology to identify promising minima | [Endres et al. (2018)](https://stefan-endres.github.io/shgo/files/shgo_softx.pdf?utm_source=chatgpt.com)                                                             |
| Simultaneous Optimistic Optimization (SOO) |      2011 | Optimistic hierarchical search without a known smoothness constant | [Munos (2011)](https://papers.nips.cc/paper/4309-optimistic-optimization-of-a-deterministic-function-without-the-knowledge-of-its-smoothness?utm_source=chatgpt.com) |
| Piyavskii–Shubert                          | 1967/1972 | Construct lower bounds using a Lipschitz constant                  | [Shubert (1972)](https://doi.org/10.1137/0709029?utm_source=chatgpt.com)                                                                                             |

### basin detection
**Scale-space theory** studies how extrema and other geometric features change under smoothing. Gaussian scale space provides a continuous family of landscapes indexed by smoothing scale.

**Morse theory and Morse–Smale complexes** describe critical points, their stable and unstable manifolds, and how these partition the domain. They provide a natural language for the basin structure.

**Graduated optimization and continuation methods** optimize a sequence of progressively less-smoothed objectives. Your construction differs because it first removes within-basin variation and then smooths the resulting attractor-value landscape.

finite-volume methods

## 1. Computing the flow map

For gradient descent, define

\[ \dot X_t=-\nabla f(X_t),\qquad X_0=x. \]

The flow map is

\[ \Phi_t(x)=X_t. \]

It satisfies the integral equation

\[ \boxed{ \Phi_t(x)=x-\int_0^t\nabla f(\Phi_s(x))\,ds. } \]

This is exact but implicit. For general nonlinear \(f\), there is no closed-form expression for \(\Phi_t\).

Thus,

\[ \boxed{ \Phi_t(x) = x-t\nabla f(x) +\frac{t^2}{2}\nabla^2f(x)\nabla f(x) +\cdots. } \]

This avoids numerical time integration when the series converges and can be evaluated efficiently, but it does not generally eliminate the need to approximate the flow. The series may have a limited convergence interval, and higher-order terms become expensive.

we can define an attractor map based on this and evaluate its value. then we have a partition of the domain by the local minima attractors, and now we can explore this discrete space instead of the continuum.


# 3. Exploiting backward instability efficiently

Near the minimum,

\[ \dot u=Hu. \]

Therefore,

\[ u(t)=e^{Ht}u(0). \]

Along eigenvector \(q_i\),

\[ u_i(t)=e^{\lambda_i t}u_i(0). \]

This means directions with larger Hessian eigenvalues expand faster.

But that does not necessarily mean they explore the basin more effectively.

In fact, the fastest-expanding directions may dominate the trajectory distribution, causing many samples to follow similar paths.

### 3.1. Compensate for anisotropic expansion

Suppose we want the backward-expanded points to cover directions approximately uniformly.

For a target expansion time \(t\), initialize

\[ \boxed{ u_0=e^{-Ht}z, } \]

where \(z\) is sampled isotropically.

Then, under the linearized dynamics,

\[ u(t)=z. \]

This compensates for the different expansion rates.

More generally, if we know the backward-flow Jacobian

\[ J_{-t}(x)=D\Phi_{-t}(x), \]

we can use it to adjust the sampling distribution.

The idea is to sample initial perturbations so that their expanded images cover the basin efficiently, rather than concentrating along a few unstable directions.

expanding basins helps us to find how much of the domain we *dont* need to explore further.




# 3. Basin-hopping: the original research

morse theory

The paper most directly connected to our discussion is:

**David J. Wales and Jonathan P. K. Doye (1997)**  
_Global Optimization by Basin-Hopping and the Lowest Energy Structures of Lennard-Jones Clusters Containing up to 110 Atoms._

[Original paper — Journal of Physical Chemistry A](https://pubs.acs.org/doi/10.1021/jp970984n?utm_source=chatgpt.com)  
[Full-text PDF from the author's website](https://www-wales.ch.cam.ac.uk/pdf/JPCA.101.5111.1997.pdf?utm_source=chatgpt.com)

This paper is remarkably close to the first step of your construction.

It defines a transformed energy landscape in which every point is assigned the energy of the local minimum reached by optimization from that point:

\[ \boxed{ \widetilde f(x)=f(A(x)). } \]

This is exactly our attractor-value function \(g(x)\).

The transformed landscape consists of plateaus corresponding to attraction basins. The global minimum value is preserved, while barriers between local minima are removed from the transformed objective. The authors then use stochastic moves and acceptance criteria to search this landscape. [ACS Publications](https://pubs.acs.org/doi/10.1021/jp970984n?utm_source=chatgpt.com)

The difference from your proposal is significant.

**Basin-hopping does not attempt to construct a smooth gradient field that systematically enlarges the attraction basin of the global optimum.**

It uses the collapsed landscape to make stochastic exploration more effective.

Your proposed continuation—interpolating boundary information into basin interiors, changing their stability, and recursively merging basins—is a different direction.

### An even earlier precursor

**Zhenqin Li and Harold A. Scheraga (1987)**  
_Monte Carlo-minimization approach to the multiple-minima problem in protein folding._

[Original paper — PNAS](https://pmc.ncbi.nlm.nih.gov/articles/PMC299132/?page=0&utm_source=chatgpt.com)

This combines local energy minimization with Monte Carlo moves between minima. It is an important precursor to modern basin-hopping methods. [PubMed](https://pubmed.ncbi.nlm.nih.gov/3477791/?utm_source=chatgpt.com)

### Why basin-hopping can work

A useful follow-up is:

**Doye, Wales and Miller (1998)**  
_Thermodynamics and the Global Optimization of Lennard-Jones Clusters._

[Paper and abstract](https://doye.chem.ox.ac.uk/abstracts/jcp98c.html?utm_source=chatgpt.com)

This investigates why transforming the energy landscape improves global search. A central observation is that the transformed landscape can increase the probability of occupying globally optimal structures at temperatures where transitions between competing regions remain possible.

This is particularly relevant to your goal of increasing the effective size of the global basin. [Doye Group](https://doye.chem.ox.ac.uk/abstracts/jcp98c.html?utm_source=chatgpt.com)

# 4. Other research closely related to your construction

The literature contains several approaches that correspond to different parts of what we discussed.

| Research                 | Main idea                                                          | Relationship to your proposal                     |
| ------------------------ | ------------------------------------------------------------------ | ------------------------------------------------- |
| Basin-hopping            | Collapse each basin to its minimum value                           | Almost exactly your \(g(x)=f(A(x))\)              |
| Filled-function methods  | Modify the objective to escape known local minima                  | Very close to your basin inversion                |
| Tunneling methods        | Construct a new search problem after finding a local minimum       | Similar to eliminating the current basin          |
| Minima hopping           | Explore neighboring minima using dynamically informed escape moves | Related to backward-flow exploration              |
| Gentlest ascent dynamics | Modify dynamics to converge to saddles separating basins           | Directly relevant to boundary discovery           |
| Metadynamics             | Progressively fill visited minima with a bias potential            | Related to recursive basin flattening             |
| Graduated nonconvexity   | Solve a sequence of increasingly difficult landscapes              | Related to your hierarchical smoothing            |
| Simulated annealing      | Use stochastic dynamics to escape local minima                     | Provides asymptotic global convergence guarantees |

Here are the papers I would prioritize.

### A. Filled-function methods — changing local minima into escape regions

**R. P. Ge and collaborators (1990)**  
_The Globally Convexized Filled Functions for Global Optimization._

[Original paper](https://www.sciencedirect.com/science/article/pii/009630039090114I?utm_source=chatgpt.com)

This is especially relevant to your idea of replacing a basin's interior with a simple potential that no longer attracts trajectories.

Filled-function methods construct an auxiliary objective after discovering a local minimum. The auxiliary function is designed to discourage returning to the same minimum and help locate a better region.

The key distinction is that your proposal uses the geometry of the attraction basin and its boundary to construct the replacement field, whereas filled-function methods typically work from known minima and objective values. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/009630039090114I?utm_source=chatgpt.com)

### B. Tunneling algorithms — searching for better basins

**A. V. Levy and A. Montalvo**  
_The Tunneling Algorithm for the Global Minimization of Functions._

[Original paper](https://doi.org/10.1137/0906002?utm_source=chatgpt.com)

The algorithm alternates between local minimization and a tunneling phase.

After finding a local minimum, it constructs a search procedure intended to locate another region from which optimization can reach an equally good or better minimum.

This is closely related to your proposed rule of escaping a basin, crossing its boundary, and descending toward a new attractor. [DOI](https://doi.org/10.1137/0906002?utm_source=chatgpt.com)

### C. Minima hopping — using geometry to choose escape directions

**Stefan Goedecker (2004)**  
_Minima Hopping: An Efficient Search Method for the Global Minimum of the Potential Energy Surface of Complex Molecular Systems._

[Original paper](https://pubmed.ncbi.nlm.nih.gov/15268009/?utm_source=chatgpt.com)

This is particularly relevant to your interest in using Hessian information and trajectories.

The algorithm uses short molecular-dynamics trajectories to escape minima and adaptive feedback to avoid revisiting previously explored configurations.

An important empirical observation is that low-barrier escape paths often lead toward low-energy minima.

This suggests that exploring basin boundaries intelligently may be more efficient than uniformly sampling radial directions. [PubMed](https://pubmed.ncbi.nlm.nih.gov/15268009/?utm_source=chatgpt.com)

### D. Gentlest ascent dynamics — finding the boundary saddles

**Weinan E and Xiang Zhou (2011)**  
_The Gentlest Ascent Dynamics._

[Original paper](https://collaborate.princeton.edu/en/publications/the-gentlest-ascent-dynamics/?utm_source=chatgpt.com)

This paper is particularly close to our discussion of modifying the gradient field.

It constructs dynamics whose stable equilibria are index-one saddle points of the original system.

For a gradient system, the central idea is to reverse the gradient component along a selected unstable direction while retaining descent behavior in other directions.

Schematically,

\[ \boxed{ \dot x = -\nabla f(x) + 2(v^\top\nabla f(x))v, } \]

where \(v\) is an appropriately evolved direction associated with the lowest Hessian mode.

This changes the stability of the original landscape so that certain saddles become attracting.

Those saddles are important because their stable manifolds often form boundaries between attraction basins.

For your approach, this could provide a more efficient way to discover neighboring basins than reconstructing their entire boundaries. [Princeton University](https://collaborate.princeton.edu/en/publications/the-gentlest-ascent-dynamics/?utm_source=chatgpt.com)

### E. Metadynamics — progressively filling attraction basins

**Alessandro Laio and Michele Parrinello (2002)**  
_Escaping Free-Energy Minima._

[Original paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC130499/?utm_source=chatgpt.com)

This may be the closest conceptual match to your repeated basin-flattening idea.

Metadynamics adds a history-dependent bias potential to discourage returning to previously visited regions.

A typical form is

\[ F_t(x) = f(x)+ \sum_{k<t} w_k \exp\left( -\frac{\|x-x_k\|^2}{2\sigma_k^2} \right). \]

As the algorithm explores a region, the accumulated bias raises its effective energy.

This gradually fills wells and encourages exploration of new regions.

The original method generally operates in selected collective coordinates, rather than necessarily in the full configuration space.

Your approach differs by attempting to reconstruct basin geometry and replace its interior directly, potentially avoiding repeated local deposition of bias terms. [PubMed Central (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC130499/?utm_source=chatgpt.com)

### F. Diffusions for global optimization — enlarging the global basin probabilistically

**Stuart Geman and Chii-Ruey Hwang (1986)**  
_Diffusions for Global Optimization._

[Original paper](https://epubs.siam.org/doi/abs/10.1137/0324060?journalCode=sjcodc&utm_source=chatgpt.com)

This provides a different perspective on your objective.

Consider Langevin dynamics:

\[ dX_t = -\nabla f(X_t)\,dt + \sqrt{2T}\,dW_t. \]

At fixed positive temperature, the invariant density under suitable conditions is

\[ p_T(x)\propto e^{-f(x)/T}. \]

As \(T\to0\), this distribution concentrates near global minima.

With sufficiently slow cooling and suitable assumptions, simulated annealing can converge in distribution toward the set of global minima.

In a sense, the algorithm makes the global optimum dominate the long-term probability distribution, even when its deterministic attraction basin is small.

The limitation is that the cooling schedule required for theoretical convergence can be extremely slow. [SIAM](https://epubs.siam.org/doi/abs/10.1137/0324060?journalCode=sjcodc&utm_source=chatgpt.com)

trust-region methods. Tensor methods, higher-order Taylor optimization. Random search, Bayesian optimization, Nelder–Mead

[A Hamilton–Jacobi-based Proximal Operator](https://pmc.ncbi.nlm.nih.gov/articles/PMC10083605/?utm_source=chatgpt.com)
[Variational Characterizations of Local Entropy and Heat Regularization in Deep Learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC7515000/?utm_source=chatgpt.com)

### Tunneling and filling
These are the most useful sources I found that expose actual formulas rather than only abstracts.

[A novel parameter-free differentiable filled function (2026)](https://www.nature.com/articles/s41598-026-46030-x?utm_source=chatgpt.com)

Full open-access paper. Particularly useful for its explicit construction, derivatives, and proofs of stationary-point elimination.

[A New Parameterless Filled Function Method (2022)](https://www.mdpi.com/2075-1680/11/12/746?utm_source=chatgpt.com)

Open access, with a historical review of filled functions and their parameter-selection problems.

[Filled-function research paper (Optimization Online, PDF)](https://optimization-online.org/wp-content/uploads/2004/03/846.pdf?utm_source=chatgpt.com)

Free PDF discussing globally concavized versus globally convexized filled functions, with mathematical definitions and constructions.

[Tunneling algorithm with explicit formulas](https://pmc.ncbi.nlm.nih.gov/articles/PMC5621012/?utm_source=chatgpt.com)

Open-access application reproducing the classical inverse-power tunneling function and modified variants.

# Another formulation
A useful necessary-and-sufficient condition is simply the convexity inequality:

\[ \boxed{ \inf_y[f(y)+c(\theta x_1+(1-\theta)x_2,y)] \le \theta g(x_1)+(1-\theta)g(x_2) } \]

for all \(x_1,x_2\) and \(\theta\in[0,1]\).

But this is not yet a constructive condition on \(c\).

A much more useful sufficient condition is that

\[ \boxed{ (x,y)\mapsto f(y)+c(x,y) } \]

is jointly convex.

Then partial minimization over \(y\) preserves convexity, provided the feasible set is convex.
## 1. Formal setup

Assume \(f:\mathbb R^d\to\mathbb R\) is continuous, coercive, and has a unique global minimizer

\[ a_*=\arg\min_y f(y). \]

Let \(h\) be continuous and strictly increasing, and define

\[ F(y)=h(f(y)). \]

Then \(F\) has the same unique minimizer \(a_*\).

Consider

\[ \boxed{ g(x)=\inf_y\left[F(y)+c(x,y)\right]. } \]

Suppose

\[ c(x,y)\ge0,\qquad c(x,y)=0\iff x=y. \]

Under suitable attainment and continuity assumptions, these conditions guarantee

\[ \arg\min g=\{a_*\}. \]

But they do not eliminate nonglobal stationary points.

To achieve that, impose the stronger radial condition

\[ \boxed{ \langle\nabla g(x),x-a_*\rangle>0, \qquad x\ne a_*. } \tag{1} \]

This is sufficient for global attraction under gradient flow, assuming the dynamics are well-defined.

In fact, if \(g\) is differentiable and (1) holds, then

\[ \frac{d}{dt}\frac12\|x(t)-a_*\|^2 = -\langle\nabla g(x),x-a_*\rangle<0. \]

Thus the squared distance to \(a_*\) is a strict Lyapunov function.

## 2. Derive the condition directly on \(c\)

Define the set of inner minimizers

\[ Y(x)=\arg\min_y[F(y)+c(x,y)]. \]

Assume for now that \(Y(x)=\{y_x\}\), with sufficient smoothness.

By the envelope theorem,

\[ \boxed{ \nabla g(x)=\nabla_xc(x,y_x). } \]

The derivative of \(F(y_x)\) does not appear explicitly because the inner minimizer satisfies

\[ \nabla F(y_x)+\nabla_yc(x,y_x)=0. \]

Therefore, the global attraction condition becomes

\[ \boxed{ \langle\nabla_xc(x,y_x),x-a_*\rangle>0. } \tag{2} \]

This is the fundamental constraint.

It depends only on the derivative of the transition cost at the optimal destination.

If we want to eliminate the implicit minimizer \(y_x\), we can use softmin approximations.

## 3. Replace the inner infimum with log-sum-exp

Choose a reference measure \(\mu\) with positive density around the relevant minimizers.

Define

\[ \boxed{ g_\beta(x) = -\beta\log \int e^{-[F(y)+c(x,y)]/\beta}\,\mu(dy). } \]

As \(\beta\to0\), under Laplace-principle assumptions,

\[ g_\beta(x)\longrightarrow g(x). \]

Define the conditional Gibbs distribution

\[ \boxed{ \pi_\beta(dy\mid x) = \frac{ e^{-[F(y)+c(x,y)]/\beta}\mu(dy) }{ \int e^{-[F(z)+c(x,z)]/\beta}\mu(dz) }. } \]

Differentiating under the integral,

\[ \boxed{ \nabla g_\beta(x) = \mathbb E_{\pi_\beta(\cdot\mid x)} [\nabla_xc(x,Y)]. } \tag{3} \]

This is exact.

Therefore, a sufficient condition for global attraction of the softened objective is

\[ \boxed{ \mathbb E_{\pi_\beta(\cdot\mid x)} \left[ \langle\nabla_xc(x,Y),x-a_*\rangle \right]>0. } \tag{4} \]

This replaces the inner optimization with an expectation.

Notice that \(h\) enters through the Gibbs weights, while \(c\) enters through both the weights and its gradient.

## 4. Replace the global minimizer with a Gibbs expectation

Now define a second Gibbs distribution, based only on \(f\):

\[ \boxed{ \nu_\tau(dz) = \frac{e^{-f(z)/\tau}\mu(dz)} {\int e^{-f(w)/\tau}\mu(dw)}. } \]

Let

\[ \boxed{ a_\tau=\mathbb E_{\nu_\tau}[Z]. } \]

For a unique global minimizer \(a_*\), suitable coercivity and integrability conditions imply

\[ a_\tau\longrightarrow a_* \qquad(\tau\to0). \]

This gives a soft approximation of the minimizer's location.

Substituting into (4),

\[ \boxed{ \lim_{\tau\to0} \mathbb E_{\pi_\beta(\cdot\mid x)} \left[ \left\langle \nabla_xc(x,Y),x-a_\tau \right\rangle \right]>0. } \tag{5} \]

Equivalently, because the two Gibbs distributions are independent,

\[ \boxed{ \lim_{\tau\to0} \mathbb E_{\substack{Y\sim\pi_\beta(\cdot\mid x)\\Z\sim\nu_\tau}} \left[ \langle\nabla_xc(x,Y),x-Z\rangle \right]>0. } \tag{6} \]

This is an explicit integral constraint on \(c\) and \(h\), with no occurrence of \(\arg\min f\).

## constraints on h
Recall

\[ g(x)=\inf_y\left[h(f(y))+c(x,y)\right], \]

with unique global minimizer \(a_*\).

For a smooth, uniquely attained inner minimum \(y_x\),

\[ \nabla g(x)=\nabla_xc(x,y_x). \]

The attraction condition is

\[ \boxed{ \langle\nabla_xc(x,y_x),x-a_*\rangle>0. } \]

Although \(h\) does not appear explicitly, it determines \(y_x\) through

\[ \boxed{ h'(f(y_x))\nabla f(y_x)+\nabla_yc(x,y_x)=0. } \]

Thus \(h'\) determines how strongly the objective pulls the optimal destination away from the transition-cost minimizer \(y=x\).


I would distinguish two desirable properties of \(h\):

Global separation: Make nonglobal low-valued regions expensive enough that the inner minimization prefers the true optimum.

Local sharpness: Make the transformed objective grow sufficiently rapidly away from the optimum that the proximal destination contracts toward it.

These are related, but not identical. A function that sharply separates objective values near \(m\) might compress differences farther away, and vice versa.

A particularly interesting family is

\[ \boxed{ h(z)=A(z-m)^q,\qquad 0<q\le1, } \]

or its smooth approximations, with \(q\) adapted to the local growth exponent of \(f\).

There is also a potentially more powerful construction: if we could estimate the global sublevel-volume function

\[ V(s)=\mu\{y:f(y)\le s\}, \]

then \(h\) could be designed using the distribution of objective values, not just their numerical scale. That would let us amplify precisely the objective ranges where competing near-optimal points are concentrated.

# Bayesian Optimization

Start with **Frazier’s tutorial** for the mechanics, then read **TuRBO**, **dimension-scaled vanilla BO**, and the **BBOB comparison** for your high-dimensional black-box optimization question. All the papers below have freely readable versions.

## Foundations

|Resource|What to read it for|
|---|---|
|**Frazier (2018), [A Tutorial on Bayesian Optimization](https://arxiv.org/abs/1807.02811?utm_source=chatgpt.com)**|The best starting point. Walks through Gaussian process regression and how expected improvement, entropy search, and knowledge gradient each decide where to evaluate next. [arxiv.org](https://arxiv.org/abs/1807.02811?utm_source=chatgpt.com)|
|**Rasmussen & Williams, [Gaussian Processes for Machine Learning](https://gaussianprocess.org/gpml/chapters/?utm_source=chatgpt.com)**|Chapters 2 and 4 cover GP regression and covariance functions. Read these if you want to understand how a kernel determines what a new observation tells you about the rest of the domain. [gaussianprocess.org](https://gaussianprocess.org/gpml/chapters/?utm_source=chatgpt.com)|
|**Jones, Schonlau & Welch (1998), [Efficient Global Optimization](https://www.schonlau.net/publication/jogo98.pdf?utm_source=chatgpt.com)**|The classic expected improvement paper. Read it for the original EGO algorithm and its worked examples. [schonlau.net](https://www.schonlau.net/publication/jogo98.pdf?utm_source=chatgpt.com)|
|**Srinivas et al. (2010), [GP-UCB](https://www.dna.caltech.edu/Papers/Srinivas-2010-regret-ICML.pdf?utm_source=chatgpt.com)** and **Bull (2011), [Convergence Rates of Efficient Global Optimization Algorithms](https://jmlr.csail.mit.edu/beta/papers/v12/bull11a.html?utm_source=chatgpt.com)**|Two routes into the theory. GP-UCB bounds **cumulative regret** using information gain. Bull studies convergence for **the best point found** by expected improvement, which is closer to your goal if you only care about the final optimizer. [dna.caltech.edu](https://www.dna.caltech.edu/Papers/Srinivas-2010-regret-ICML.pdf?utm_source=chatgpt.com)|

## High-dimensional global optimization

|Paper|How it handles dimension|
|---|---|
|**Eriksson et al. (2019), [TuRBO](https://arxiv.org/abs/1910.01739?utm_source=chatgpt.com)**|Fits local GP models in adaptive trust regions and allocates evaluations across them. Closest to your basin-search interests, since it decides which local searches deserve more evaluations. [papers.neurips.cc](https://papers.neurips.cc/paper_files/paper/2019/hash/6c990b7aca7bc7058f5e98ea909e924b-Abstract.html?utm_source=chatgpt.com)|
|**Eriksson & Jankowiak (2021), [SAASBO](https://proceedings.mlr.press/v161/eriksson21a.html?utm_source=chatgpt.com)**|Uses a sparsity prior on GP inverse length scales to learn which input coordinates matter. The structural assumption is that relatively few original coordinates drive the objective. [proceedings.mlr.press](https://proceedings.mlr.press/v161/eriksson21a.html?utm_source=chatgpt.com)|
|**Hvarfner, Hellsten & Nardi (2024), [Vanilla Bayesian Optimization Performs Great in High Dimensions](https://proceedings.mlr.press/v235/hvarfner24a.html?utm_source=chatgpt.com)**|Scales the GP length-scale prior with dimension. A particularly relevant baseline alongside TuRBO and SAASBO because it isolates how much of the difficulty comes from the surrogate prior. [proceedings.mlr.press](https://proceedings.mlr.press/v235/hvarfner24a.html?utm_source=chatgpt.com)|
|**Santoni et al., [Comparison of High-Dimensional Bayesian Optimization Algorithms on BBOB](https://arxiv.org/abs/2303.00890?utm_source=chatgpt.com)**|Compares BO variants with CMA-ES across 24 COCO functions at 10–60 dimensions. The authors report an advantage for BO at limited evaluation budgets, with performance varying by landscape and budget. The 2024 dimension-scaled baseline is worth reading alongside this comparison. [arxiv.org](https://arxiv.org/abs/2303.00890?utm_source=chatgpt.com)|

## Information gain and implementation

- **[Entropy Search](https://www.jmlr.org/beta/papers/v13/hennig12a.html?utm_source=chatgpt.com)** and **[Max-value Entropy Search](https://proceedings.mlr.press/v70/wang17e.html?utm_source=chatgpt.com)** connect directly to your idea of choosing where to sample based on uncertainty. The first asks how much an observation would tell you about **where** the global optimum is. The second asks how much it would tell you about **what value** the optimum attains. [jmlr.org](https://www.jmlr.org/beta/papers/v13/hennig12a.html?utm_source=chatgpt.com)
- **[Unexpected Improvements to Expected Improvement](https://proceedings.neurips.cc/paper/2023/hash/419f72cbd568ad62183f8132a3605a2a-Abstract-Conference.html?utm_source=chatgpt.com)** explains why ordinary EI can become numerically hard to optimize and develops LogEI variants. The **[BoTorch closed-loop tutorial](https://botorch.org/docs/next/tutorials/closed_loop_botorch_only?utm_source=chatgpt.com)** gives runnable code for batch LogEI and noisy batch LogEI.

# RL based optimization

The most direct version is **train a policy to decide where to evaluate a black-box function next**. There are also methods where the policy controls an existing optimizer, and tree-search methods that learn where to focus within a single run.

|Approach|What it learns or decides|Paper|
|---|---|---|
|**Learn the optimizer itself**|The policy observes the search history and proposes the next iterate. Training across objectives lets it learn search behavior that transfers to new instances.|Li & Malik, [Learning to Optimize](https://arxiv.org/abs/1606.01885?utm_source=chatgpt.com). An early formulation of the optimizer as an RL policy. [arxiv.org](https://arxiv.org/abs/1606.01885?utm_source=chatgpt.com)|
|**Learn to control an existing optimizer**|An RL agent observes the state of a lower-level optimizer and adjusts how it searches. The lower-level optimizer still generates candidates.|Ma et al., [MetaBox](https://arxiv.org/abs/2310.08252?utm_source=chatgpt.com). A benchmark with learned and traditional baselines, plus [code and tutorials](https://metaevo.github.io/MetaBox/?utm_source=chatgpt.com). [arxiv.org](https://arxiv.org/abs/2310.08252?utm_source=chatgpt.com)|
|**Learn where to search within one run**|LA-MCTS learns a partition from the current function’s evaluations, uses bandit-style tree traversal to choose a region, and runs a solver such as BO or TuRBO inside it.|Wang, Fonseca & Tian, [LA-MCTS](https://arxiv.org/abs/2007.00708?utm_source=chatgpt.com). The closest of these to your idea of allocating search effort across basins, although its learned regions need not coincide with actual attraction basins. [arxiv.org](https://arxiv.org/abs/2007.00708?utm_source=chatgpt.com)|

### How the RL objective differs from expected improvement

Suppose you’re minimizing a deterministic function with a budget of \(T\) evaluations. The policy sees the history \(H_t=\{(x_i,f(x_i))\}_{i=1}^t\), chooses \(x_{t+1}\), and gets reward \(r_t=b_t-b_{t+1}\), where \(b_t\) is the best value observed so far. The undiscounted return telescopes:

\[ \sum_{t=t_0}^{T-1} r_t=b_{t_0}-b_T. \]

So a policy trained on this return can learn to make a **probe that yields zero immediate improvement** if the result helps it find a better point later. Standard one-step expected improvement scores the immediate gain. The RL formulation gives credit for downstream gains over the remaining budget.

For your basin-search idea, a natural action would be **continue refining this region, switch to a known region, or probe somewhere new**. The distinction I’d use to pick a paper is whether you have _one unknown objective_ or _a family of related objectives_. **LA-MCTS** is the more direct starting point for one objective because it learns the partition during the run. **MetaBox** is the better starting point if you want to train a search controller across instances and test whether it beats TuRBO or CMA-ES on held-out functions at the same evaluation budget. That comparison measures transfer, which is where the upfront RL training can pay off. [proceedings.neurips.cc](https://proceedings.neurips.cc/paper/2020/file/e2ce14e81dba66dbff9cbc35ecfdb704-Paper.pdf?utm_source=chatgpt.com)

## Beam search

Suppose a candidate is built through a sequence of decisions, such as composing a function one operation at a time. At each step, beam search expands the candidates it kept from the previous step, scores their children, and keeps only the best \(B\). The number \(B\) is the **beam width**.

For example, with width 3 and 10 possible operations, you score up to 30 extensions per step and keep 3. Width 1 gives you greedy search. A wider beam gives a candidate that looks mediocre now a chance to become good after another step. Once its branch is pruned, though, the search can miss the best final composition.

For continuous optimization, you have to **define what a branch means**. In your basin-search setup, you could keep \(B\) active local searches, advance each for a small evaluation budget, and periodically replace some with new starts. That’s closer to a beam-inspired multistart optimizer than classical beam search. The key choice is how to rank branches. If you rank solely by the best value seen so far, several starts in the same easy basin could crowd out a slower but more promising basin.

## Latin hypercube sampling

“Latin square search” most likely referred to **Latin hypercube sampling (LHS)**. If you have \(N\) evaluations to place in a \(d\)-dimensional box, split **each coordinate axis** into \(N\) equal intervals. Then choose \(N\) points so that, along every coordinate, exactly one point lands in each interval.

For example, with 5 points in 2D, draw a \(5\times5\) grid and place one point in each row and each column. You get coverage of both axes with **5 evaluations**, while evaluating every cell would take 25. The guarantee is about each coordinate separately. A design could put all five points along the diagonal and still satisfy the rule, so the pairing of intervals matters for joint coverage.

The names are related. A **Latin square** fills every cell with a symbol, with each symbol appearing once per row and column. A **2D Latin hypercube sample** marks one cell per row and column.

### What the contour _can_ count

There’s an exact version of your counting idea if you track **how many times the gradient direction rotates** around the loop. On a simple, regular level curve, the gradient is normal to the curve, so it makes one full turn whether the curve is convex or wavy. If the critical points inside are nondegenerate, the planar index theorem gives

\[ \#\text{minima}+\#\text{maxima}-\#\text{saddles}=1. \]

A minimum and a saddle cancel in this count. That’s exactly what happens in the circular example, where \(2+0-1=1\).

**Multiparameter continuation** is the established version of tangent-tracing idea. **Sublevel merge trees** capture your idea of raising the contour level and watching wells merge. If you want the actual boundary between attraction basins, the relevant methods compute **stable manifolds of saddles**.

|What you want to recover|Existing method|Starting point|
|---|---|---|
|**The geometry of one surface \(f(x)=c\)**|**Multiparameter continuation.** Extend a collection of overlapping local patches across the implicit surface. This is the higher-dimensional analogue of stepping tangentially and correcting back onto the contour.|Henderson, [Multiple Parameter Continuation: Computing Implicitly Defined \(k\)-Manifolds](https://research.ibm.com/publications/multiple-parameter-continuation-computing-implicitly-defined-k-manifolds?utm_source=chatgpt.com). [IBM Research](https://research.ibm.com/publications/multiple-parameter-continuation-computing-implicitly-defined-k-manifolds?utm_source=chatgpt.com)|
|**Which wells merge as \(c\) rises**|**Sublevel merge trees**, also called **join trees**. Sweep through the values of a sampled function on a mesh and track which connected components appear and merge. The output is a hierarchy of wells and their merge levels.|Carr, Snoeyink & Axen, [Computing Contour Trees in All Dimensions](https://www.cs.ubc.ca/tr/1999/tr-99-09?utm_source=chatgpt.com). Look for the join-tree construction. [Computer Science at UBC](https://www.cs.ubc.ca/tr/1999/tr-99-09?utm_source=chatgpt.com)|
|**A barrier hierarchy without meshing the entire domain**|**Disconnectivity graphs.** Build a network of discovered minima and transition states, then group minima by the lowest energy threshold at which the network connects them. Missed transition states can make an estimated barrier too high.|Wales group, [PATHSAMPLE disconnectivity graph example](https://www-wales.ch.cam.ac.uk/examples/PATHSAMPLE/DisconnectivityGraphs/?utm_source=chatgpt.com). [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10523320/?utm_source=chatgpt.com)|
|**The actual boundary between gradient-flow basins**|**Stable-manifold computation.** When a saddle with one unstable direction separates two minima, points on its stable manifold flow _toward the saddle_. Nearby points on opposite sides can flow to different minima. One numerical approach starts near the saddle and extends the manifold backward along the flow.|Kuznetsov, [Elements of Applied Bifurcation Theory](https://www.ma.imperial.ac.uk/~dturaev/kuznetsov.pdf?utm_source=chatgpt.com), on computing invariant manifolds. [ma.imperial.ac.uk](https://www.ma.imperial.ac.uk/~dturaev/kuznetsov.pdf?utm_source=chatgpt.com)|

The double well from earlier makes the distinction concrete. For \(f(x,y)=(x^2-1)^2+y^2\), the **merge tree** tells you the two wells connect at energy \(1\). The **stable manifold** tells you their basin boundary is the entire line \(x=0\). Along that line, \(f(0,y)=1+y^2\), so a single contour only intersects the boundary at a few points.

There’s also a direct connection to your idea of **running descent backward to expand a basin**. Let \(C\) be a small trapped sublevel component around a minimum \(m\), with \(m\) in its interior, and let \(\Phi_t\) be the gradient flow. Then

\[ B(m)=\bigcup_{t\ge 0}\{x:\Phi_t(x)\in C\}. \]

Every point converging to \(m\) eventually enters \(C\), and every point that enters \(C\) converges to \(m\). So backward expansion of a certified core recovers the basin _in principle_. Numerically, the choice is what to represent. **Continuation** follows the surface, a **disconnectivity graph** compresses the barrier information to minima and transitions, and **stable-manifold computation** targets the separator directly.

| Global optimizer                                                                                                                       | How the information changes the search                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| -------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **[Pure Adaptive Search](https://deepblue.lib.umich.edu/bitstream/handle/2027.42/47920/10107_2005_Article_BF01582296.pdf?sequence=1)** | Given the current best value \(b\), sample uniformly from the entire improving region \(\{x:f(x)<b\}\), including disconnected components. This directly exploits sublevel geometry. The hard part is drawing that sample. Tracing the contour of one component gives only local information about where improvements are possible. ([deepblue.lib.umich.edu](https://deepblue.lib.umich.edu/bitstream/handle/2027.42/47920/10107_2005_Article_BF01582296.pdf?sequence=1&utm_source=chatgpt.com "Mathematical Programming 43 (1988) 317-328"))       |
| **[Minima Hopping](https://edoc.unibas.ch/entities/publication/a571b9bc-a627-4e05-8066-cd710ae4ef02)**                                 | Start at a known minimum, launch a molecular-dynamics trajectory biased toward soft directions, then minimize from the endpoint. It adapts the escape energy when it revisits known minima. The goal is to cross a nearby low barrier without having to reconstruct the basin boundary. ([edoc.unibas.ch](https://edoc.unibas.ch/entities/publication/a571b9bc-a627-4e05-8066-cd710ae4ef02?utm_source=chatgpt.com "Minima hopping: An efficient search method for the global minimum of the potential energy surface of complex molecular systems")) |
| **[Multilevel Single Linkage](https://mathopt.zib.de/COAL-Issues/COAL11.pdf)**                                                         | Sample candidate starts and skip a local solve when a nearby, better sample suggests it would be redundant. It uses proximity as a proxy for sharing a basin. Your proposal would replace that heuristic, wherever possible, with a region that is certified to drain to a known minimum. ([mathopt.zib.de](https://mathopt.zib.de/COAL-Issues/COAL11.pdf?utm_source=chatgpt.com "THE MULTI LEVEL SINGLE LINKAGE METHOD FOR GLOBAL OPTIMIZATION"))                                                                                                   |

[Stochastic Adaptive Search Methods](https://web.archive.org/web/20180712174243/http://ie.sharif.edu/~so/(International%20Series%20in%20Operations%20Research%20&%20Management%20Science%20216)%20Michael%20C%20Fu%20(eds.)-Handbook%20of%20Simulation%20Optimization-Springer-Verlag%20New%20York%20(2015).pdf)
[Segment Tracing Using Local Lipschitz Bounds](https://hal.science/hal-02507361)
[Forward inclusion functions for ray-tracing implicit surfaces](https://inria.hal.science/hal-04129922/file/Preprint-2.pdf)
[global opt at work](https://edepot.wur.nl/201647)

