Suppose we are optimizing a loss function $L(x): \mathbb R^n \to \mathbb R^m$.

It might be constrained to solutions that must satisfy some condition $\phi(x)\le0$
The function loss observations might be noisy, with some distribution $\epsilon \approx p_{\epsilon}(x)$ with $\mathbb E_x[\epsilon]=0$
We might assume a certain distribution $X \approx p_X(x)$ of observations under which the loss is tested as well.

For a point in global optimum set we use $x^* \in \arg\min L(x)$ and $L^*=L(x^*)$ for its value.

multi objective
https://chatgpt.com/c/6aff2f52-d41c-83eb-b9ea-3137b667f9e7?no_universal_links=1
# Characterization
The basic characterization is is within $\delta$-neighborhood of the optimum and the suboptimality/$\varepsilon$-optimality:
$$ \|L(x)-L^*\|<\varepsilon \\ \|x-x^*\|<\delta$$
But this is a bit restrictive, as this is most suitable for characterizing local optimization.

For global optimization a more natural thing is a probability of a specific optimality condition being met at some step $k$:
$$ P_\varepsilon(k) = \Pr(\|L(x_k)-L^*\|\leq\varepsilon) \\ P_\delta(k) = \Pr(\|x_k-x^*\|\leq\delta)$$
because we cant really guarantee them over the whole space by checking everything.
We can then characterize performance using the expected hitting time $\tau$:
$$\tau = \mathbb E[k] $$

Global optimization can be split into search and convergence. So we must have a guarantee that we eventually improve, and that every improvement is sufficiently strong to always converge to the global optimum
## Local optimization
Suppose we want to find a minima of the loss function. At any minima we necessarily must have these conditions be true:
$$L'(x)=0 \quad L''(x)>0$$
The first means we must find a saddle point, and the second one means that any further movement will only increase the loss, because it curves upward. While this is true for all global minima, not every point satisfying these conditions is a global minima. Instead, the points satisfying these conditions are not necessarily globally optimal, and we call locally optimal, as they consider only local information to decide that we cannot improve anymore further.

For local optimization we can characterize how quickly do we converge to a local minima by error order $p$ and rate constant $C$:
$$\varepsilon=O(h^p)\\ \delta_{k+1}=C\|x_{k}-x^*\|$$
Order characterizes improvement rate relative to the step size, while rate $C$ characterizes base of the exponential convergence to the optimum.
## Convexity
Notably, there is a class of functions for which every local optimum must be global. Such function are called convex, and they satisfy this property:
$$L(tx+(1-t)y)\le tL(x)+(1-t)L(y)$$
This inequality basically says that for any two points, if we evaluate at any point inbetween, the loss will be lower than simple mix of the endpoints.

Alternative definition can be made through curvature - it should be positive at every point:
$$\nabla^2 L>0$$
We could also say the function is $\mu$-strongly or $\rho$-weakly convex (aka semiconvex):
$$\nabla^2 L>\mu \qquad \nabla^2 L>-\rho$$
These are basically the same as simple convexity, but more specific about how much convexity there is.

This characterizes how optimal the current value is, and within what bound. Note that parameter distance to the optimum can be arbitrary in general - any point in the domain can be arbitrarily close to the optimal value. If the certified bound on the loss is enough for us, then this can be treated as sufficient.

We can bound upon this in a few ways. First interesting case is when $\varepsilon$ related to the gradient, as in Polyak–Łojasiewicz inequality:
$$ \boxed{ \frac12\|\nabla L(x)\|^2 \geq\mu||L^*-L(x)||. } $$
where $\mu$ is some global constant. It effectively states a bound on variation of the loss through the gradient - rate of change should generally be stronger than root of the suboptimality of the solution.

Another related one is Lipschitz inequality:
$$||f(x_2)-f(x_1)||<L||x_1-x_2||$$
It also bounds variation, but does so through a relation to parameter distance, rather than gradient.

Another more direct condition on gradient direction is this:
$$ \boxed{ \langle\nabla L(x),x-x^*\rangle\geq0 \quad\forall x\neq x^* } $$
Which means that gradient at any point should move us towards $x^*$. It can even be extended to general iterations by replacing $\nabla L$ with $\Delta x_k$, meaning that any update must move us in the direction of optimum. That is quite flexible, as it doesnt require us to constrain the target function at all, but then actual set of globally solvable objectives heavily depends on the iteration procedure.

We can even combine all of these into a composite condition, that is more flexible than any one by itself, and may even cover new possibilities for optimization.

All of these are weaker than convexity.
## Surrogates
We can also study any related functions that share optimizers within some bounds. If we specifically want global optimums to match, we dont need arbitrary uniform accuracy between two functions, only that it preserves the global minima specifically:
$$ |\min g-\min f|<\varepsilon \\ |\arg\min g-\arg\min f|<\delta$$
Then if finding optimum of $g$ is simpler in any sense, we can instead solve for its global optimum. Moreover, even if we restrict $g$ to be convex, $f$ still can be arbitrarily nonconvex.

## Constrained
When optimizing with constraints, we can make generally two choices:
1. Project any moves onto a subset that preserves the constraints.
2. Use Lagrange multipliers to modify the loss and optimize a tradeoff between constraint satisfaction and target loss optimization softly.

## Noisy
When optimizing with noise, we can only optimize up to noise strength, as at some point we cannot distinguish truly suboptimal local point from a noisy observation of the optimal local point.

## RL
Generally global optimization is more suitably characterized by RL formulation - we can describe objective improvement as reward, and then choose action (point eval) that would yield the best improvement. Note that even if the algorithm chose best point just with a few iterations, it still needs to prove optimality bounds on it. So the best action boils down to the one that would narrow it down the most.

## Basin graph
Yes. In fact, representing optimization as movement between basins gives us a much richer characterization of global optimization performance than convergence rates alone.

We can separate optimization into two processes:

1. Intra-basin optimization: convergence toward a local minimum.
    
2. Inter-basin optimization: exploration and transitions between basins.
    

The second process can be modeled as a graph search, Markov chain, or optimal-control problem. This connects directly to energy landscapes, metastability, transition path theory, and basin-hopping optimization.

### 1. Constructing the basin graph

Let \(f:\mathcal X\to\mathbb R\), and let \(L(x)\) be a deterministic local optimizer that maps an initial point to its limiting local minimum.

Define an equivalence relation:

\[ x\sim y \iff L(x)=L(y). \]

Each equivalence class is a basin:

\[ \boxed{B_i=\{x:L(x)=m_i\}} \]

where \(m_i\) is a local minimum.

We can then construct a graph

\[ G=(V,E) \]

with one vertex for each basin.

Illustrative basin-transition graph

There are several possible definitions of an edge:

- Geometric adjacency: two basins share a boundary.
    
- Transition accessibility: a particular perturbation mechanism can move from one basin to another.
    
- Energy barrier: two minima are connected through a saddle or minimum-energy path.
    
- Probabilistic transition: the optimizer has nonzero probability of moving between the basins.
    

These definitions produce different graphs. For evaluating an actual algorithm, transition probabilities are often the most informative.

### 2. Performance as a Markov chain over basins

Suppose we run a local optimizer, perturb the resulting point, and optimize again.

This is the basic structure of basin hopping.

Let

\[ P_{ij}=\Pr(\text{next basin}=B_j\mid\text{current basin}=B_i). \]

If the transition mechanism depends only on the current basin, we obtain a Markov chain:

\[ \boxed{p_{k+1}=p_kP} \]

where \(p_k\) is the distribution over basins after \(k\) transitions.

Example transition matrix

\[ P= \begin{pmatrix} 0.7&0.2&0.1\\ 0.3&0.5&0.2\\ 0.1&0.2&0.7 \end{pmatrix}. \]

Transition simulation

Start in B₁

Number of transitions

3

B1

48.8%

B2

27.8%

B3

23.4%

Illustrative three-basin process. B₃ is designated as the global-optimum basin.

This immediately gives us new optimization performance metrics.

### 3. Hitting time of the global optimum

Let \(B_*\) denote the basin containing the global minimum.

Define

\[ \tau_*=\inf\{k:X_k=B_*\}. \]

The expected hitting time

\[ \boxed{H_i=\mathbb E[\tau_*\mid X_0=B_i]} \]

characterizes how difficult it is to reach the global optimum from each basin.

For a Markov chain,

\[ H_*=0, \]

and for every other basin,

\[ \boxed{ H_i=1+\sum_jP_{ij}H_j. } \]

This is a linear system.

If \(Q\) is the transition submatrix over non-target basins, then

\[ H=(I-Q)^{-1}\mathbf 1 \]

when the expected hitting times are finite.

This is a powerful characterization because it incorporates the entire graph structure rather than just local convergence rates.

We can also calculate the probability of finding the global basin within \(k\) transitions:

\[ \boxed{ \Pr(\tau_*\leq k) = 1-p_0^{\neg *}Q^k\mathbf 1. } \]

This is the basin-level equivalent of the success probability discussed earlier.

### 4. Barrier heights and transition difficulty

For a continuous objective, basin transitions may require crossing regions of higher objective value.

Define the communication height

\[ \Phi(i,j)= \inf_{\gamma:m_i\to m_j} \max_{t\in[0,1]}f(\gamma(t)). \]

The barrier for escaping basin \(i\) toward basin \(j\) is

\[ \boxed{ \Delta_{ij}=\Phi(i,j)-f(m_i). } \]

For small-noise gradient dynamics under suitable assumptions, transition rates often have Arrhenius-type behavior:

\[ k_{ij}\asymp A_{ij}e^{-\Delta_{ij}/T}. \]

Here \(T\) controls noise intensity.

This reveals something important: the objective difference between minima does not determine transition difficulty.

Two minima may have similar objective values but be separated by a large barrier. Conversely, a substantially better minimum might be accessible through a small barrier.

The graph should therefore carry more information than just the value of each minimum.

### 5. Global optimization as graph search

Once the basin graph is defined, optimization resembles a search problem with unknown structure.

A transition has a cost \(c_{ij}\), potentially including:

- Perturbation and proposal generation.
    
- Objective evaluations needed to cross between basins.
    
- Local optimization after entering a new basin.
    
- Probability of a failed transition.
    

If the graph and costs were known, we could formulate an optimal-control problem.

Let \(a\) denote an action, such as a perturbation scale, search direction, or temperature.

Then

\[ P_{ij}(a) \]

describes how that action changes the transition probabilities.

The optimal expected cost to reach the global basin satisfies a Bellman equation:

\[ \boxed{ V(i)= \min_a \left[ c(i,a)+\sum_jP_{ij}(a)V(j) \right] } \]

with \(V(*)=0\).

This suggests an interesting distinction:

An optimizer should not necessarily maximize its probability of improving the objective on the next transition. It should minimize the expected cost of reaching a sufficiently good basin.

Sometimes that requires moving temporarily into a worse basin because it provides better access to other regions.

### 6. Graph-theoretic performance characteristics

We can now characterize an optimization landscape using graph properties.

|Characteristic|Optimization interpretation|
|---|---|
|Basin volume|Probability of entering a basin under random initialization|
|Node degree|Number of accessible neighboring basins|
|Edge transition probability|Ease of moving between basins|
|Barrier height|Difficulty of crossing between minima|
|Hitting time|Expected effort to reach a target basin|
|Conductance|Ease of escaping a group of basins|
|Spectral gap|Rate of mixing across the landscape|
|Bottlenecks|Regions that trap the optimizer|
|Effective resistance|A measure related to commute times for reversible random walks|
|Graph diameter|Maximum shortest-path distance between basins|
|Connectivity|Whether every basin can eventually be reached|

Two especially useful characteristics deserve attention.

#### Conductance

For a reversible Markov chain with stationary distribution \(\pi\), define

\[ \Phi(S)= \frac{\sum_{i\in S,j\notin S}\pi_iP_{ij}} {\pi(S)} \]

for a subset \(S\) of basins.

Low conductance indicates that the optimizer is likely to remain trapped in \(S\) for a long time.

This is a more general notion of trapping than considering individual local minima.

An entire collection of basins can form a metastable region.

#### Spectral gap

For an irreducible reversible Markov chain, the spectral gap

\[ \gamma=1-\lambda_2(P) \]

characterizes mixing speed, where \(\lambda_2\) is the second-largest eigenvalue.

A small spectral gap often indicates slow exploration due to bottlenecks.

However, mixing rapidly is not necessarily equivalent to finding the global optimum rapidly. An algorithm might mix efficiently while spending very little time in the global basin.

Therefore, target hitting times and mixing times should be characterized separately.

### 7. Basins at multiple scales

There is another extension that I think is particularly relevant: a basin graph need not have a fixed resolution.

At a fine scale, each local minimum is a separate node.

At a coarser scale, several nearby minima can be grouped into a larger metastable region.

Conceptual hierarchy: fine basins can be grouped into larger regions with slower transitions between them.

This connects to:

- Disconnectivity graphs and energy-landscape analysis.
    
- Morse theory and merge trees.
    
- Markov state models and metastability.
    
- Hierarchical basin hopping.
    
- Multiscale optimization and coarse-graining.
    

A hierarchy lets us distinguish fast transitions within a cluster of basins from slow transitions between clusters.

One can also define a scale-dependent basin graph \(G_\lambda\), where the scale controls smoothing, perturbation size, or the transition-time resolution.

This may be more informative than a single graph because the effective optimization landscape changes with the algorithm's exploration scale.

### 8. Constraints can be incorporated into the graph

For constrained optimization, the graph can represent feasible basins or regions.

Let

\[ \mathcal C=\{x:g_i(x)\leq0\}. \]

There are two fundamentally different transition models.

Feasibility-preserving transitions: every intermediate state must remain in \(\mathcal C\). The graph may become disconnected even if the unconstrained basin graph is connected.

Endpoint-feasible transitions: intermediate states may violate constraints, but the accepted destination must be feasible.

The latter can introduce edges that are impossible under feasibility-preserving movement.

We can assign an edge a feasibility cost:

\[ c_{ij}^{\mathrm{viol}} = \inf_{\gamma:i\to j} \int_0^1 V(\gamma(t))\,dt. \]

Or characterize the smallest unavoidable maximum violation:

\[ \boxed{ b_{ij}^{\mathrm{viol}} = \inf_{\gamma:i\to j} \max_t V(\gamma(t)). } \]

This is analogous to an energy barrier, but for constraint violation.

It provides a direct way to characterize the difficulty of transitioning between feasible regions when temporary infeasibility is permitted.

For algorithms that can jump directly between endpoints without evaluating the intervening path, geometric path barriers may not represent the actual transition cost. In that case, proposal probabilities and endpoint feasibility are more relevant.

### 9. An important limitation: basins are not always Markov states

The basin index alone may not contain enough information to predict the next transition.

For example, if the perturbation is applied to the current point before local optimization, different points inside the same basin can have very different probabilities of escaping.

Thus

\[ \Pr(B_{k+1}\mid B_k) \]

may depend on the optimizer's history or its precise location within \(B_k\).

A basin-level Markov model is exact only when the transition distribution depends solely on the current basin, or under a suitable state-resetting or lumpability condition.

Otherwise, we can:

- Include additional state variables such as position, momentum, temperature, or step size.
    
- Use semi-Markov models that account for transition durations.
    
- Refine the partition until the dynamics are approximately Markovian.
    
- Estimate effective transitions after local equilibration inside each basin.
    

This is especially important for noisy optimizers, where trajectories may spend significant time exploring a basin before escaping.

### 10. A unified performance measure

We can combine local convergence, graph transitions, and computational cost.

Let \(B_k\) denote the basin occupied after \(k\) transitions, and \(x_k\) the current point.

Then define

\[ \boxed{ R(B)= \mathbb E\left[ f(x_B)-f^* \right] } \]

where \(B\) is total computational budget, including local optimization and basin transitions.

We can decompose the objective gap:

\[ \begin{aligned} f(x_B)-f^* ={}& \underbrace{f(x_B)-f(m_{B_B})}_{\text{within-basin error}}\\ &+ \underbrace{f(m_{B_B})-f^*}_{\text{basin-selection error}}. \end{aligned} \]

Here \(m_{B_B}\) is the local minimum associated with the basin containing the final point. This decomposition assumes the basin map is well defined and that the local minimum is no worse than the current point.

The first term characterizes local optimization quality.

The second characterizes global exploration quality.

This is a particularly useful separation because improvements to the local optimizer primarily reduce the first term, while better exploration policies primarily reduce the second.

The most interesting implication is that global optimization can be characterized as a stochastic shortest-path problem over a graph that is generally unknown and must itself be learned. The algorithm is simultaneously discovering the basin structure, estimating transition probabilities, and choosing where to search next.

That suggests evaluating global optimizers not only by objective regret, but also by how efficiently they learn and exploit the transition structure of the landscape.
### 1. Main performance characteristics

| Characteristic      | What it measures                       | Typical quantities                       |
| ------------------- | -------------------------------------- | ---------------------------------------- |
| Local convergence   | Speed near a solution                  | Order \(p\), rate constant \(C\)         |
| Global convergence  | Behavior from arbitrary initialization | Basin of attraction, success probability |
| Solution quality    | How close the result is to optimal     | Objective gap, distance, stationarity    |
| Robustness          | Sensitivity to perturbations           | Stability region, noise amplification    |
| Scalability         | Dependence on problem dimension        | Complexity in \(n\), conditioning        |
| Exploration         | Ability to escape local minima         | Hitting time, barrier crossing           |
| Asymptotic accuracy | Best attainable precision              | Error floor, convergence bias            |

These characteristics are not interchangeable. For example, Newton's method has excellent local convergence but can fail globally, while gradient descent often has slower local convergence but stronger guarantees from arbitrary starting points under convexity assumptions.

### 2. Local convergence

Suppose an algorithm produces

\[ x_{k+1}=T(x_k) \]

and converges to \(x^*\). Define the error

\[ e_k=\|x_k-x^*\|. \]

#### 2.1 Order of convergence

The standard characterization is

\[ \boxed{e_{k+1}=C e_k^p+o(e_k^p)} \]

where:

- \(p\) is the convergence order.
    
- \(C\) is the asymptotic error constant.
    

Common cases:

|Order|Name|Example|
|---|---|---|
|\(p=1,\ C<1\)|Linear|Gradient descent on strongly convex functions|
|\(1<p<2\)|Superlinear|Secant method, \(p\approx1.618\)|
|\(p=2\)|Quadratic|Newton's method|
|\(p=3\)|Cubic|Halley's method|
|\(p=m\)|Higher order|Higher-order root-finding methods|

Illustration: error after each iteration

![Chart](data:image/png;base64,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)

■ Linear

■ Quadratic

■ Cubic

Illustrative recurrences: linear eₖ₊₁ = 0.5eₖ, quadratic eₖ₊₁ = eₖ², cubic eₖ₊₁ = eₖ³; e₀ = 0.2.

An important subtlety: order \(p\) only describes asymptotic behavior. A higher-order method may perform worse before reaching the region where its asymptotic approximation is accurate.

#### 2.2 Convergence rate constant

Even with identical order, two algorithms can behave very differently.

For linear convergence,

\[ e_{k+1}\leq q e_k,\qquad 0<q<1. \]

The smaller \(q\), the faster the method.

For gradient descent on a strongly convex quadratic with Hessian eigenvalues in \([\mu,L]\), the optimal fixed step size gives

\[ q=\frac{L-\mu}{L+\mu} =\frac{\kappa-1}{\kappa+1}, \qquad \kappa=\frac L\mu. \]

Thus the condition number \(\kappa\) directly characterizes local performance.

#### 2.3 Basin of attraction

Define

\[ B(x^*)=\{x_0:T^k(x_0)\to x^*\}. \]

This is the set of initial conditions from which the algorithm converges to the particular solution.

Two algorithms might have identical quadratic convergence near \(x^*\), but one might converge from a much larger region.

A local guarantee is often expressed as a convergence radius:

\[ \|x_0-x^*\|<r \implies x_k\to x^*. \]

The radius \(r\) is another important performance characteristic.

#### 2.4 Local stability

For differentiable iteration maps, linearization gives

$$ e_{k+1}^{\mathrm{vec}}\approx DT(x^*)e_k^{\mathrm{vec}}. $$

If

$$ \rho(DT(x^*))<1, $$

the fixed point is locally asymptotically stable.

The spectral radius characterizes the asymptotic linear rate, although nonnormality can cause transient amplification.

For Newton's method at a regular root, \(DT(x^*)=0\), explaining why its convergence can be quadratic.

### 3. Global convergence

#### 3.2 Global optimality

For minimization, define the objective gap

$$ \boxed{\Delta_k=f(x_k)-f^*} $$

Global optimization performance can be characterized by how quickly $\Delta_k$ approaches zero.

Examples:

| Function class / method           | Typical objective-gap guarantee                                 |
| --------------------------------- | --------------------------------------------------------------- |
| Convex, gradient descent          | $O(1/k)$                                                        |
| Convex, accelerated gradient      | $O(1/k^2)$                                                      |
| Strongly convex, gradient descent | $O(q^k)$                                                        |
| Strongly convex, accelerated      | $O((1-1/\sqrt{\kappa})^k)$                                      |
| General nonconvex                 | No universal global-optimum rate without additional assumptions |

The first four are worst-case upper bounds under standard smoothness assumptions.

### 4. Computational complexity

Iteration count alone is insufficient because different methods have different costs per iteration.

A useful general characterization is

\[ \boxed{ N(\epsilon,n,\mathcal F) } \]

the computational resources needed to reach accuracy \(\epsilon\), in dimension \(n\), over a function class \(\mathcal F\).

We can distinguish:

- Iteration complexity: number of algorithm steps.
    
- Oracle complexity: number of function, gradient, Hessian, or higher-derivative evaluations.
    
- Arithmetic complexity: number of operations, including linear-system solves.
    
- Memory complexity: amount of storage required.
    
- Parallel complexity: critical-path depth and communication costs.
    

For example, Newton's method may require far fewer iterations than gradient descent, but computing and solving with a dense Hessian can cost \(O(n^3)\) per iteration for a generic factorization, in addition to derivative evaluation.

Higher-order convergence is therefore not automatically better in total computational cost.

### 5. Robustness and sensitivity

An optimizer may converge rapidly on exact functions but perform poorly under noisy evaluations.

Suppose

$$ \widehat{\nabla f}(x)=\nabla f(x)+\xi. $$

We can characterize performance by the sensitivity of the result to \(\xi\).

Relevant quantities include the stationary error floor

$$ \limsup_{k\to\infty}\mathbb E\|x_k-x^*\|^2, $$

the variance of the iterates, and the probability of divergence.

For stochastic optimization, one often obtains bounds resembling

$$ \mathbb E[\Delta_k] \leq A e^{-ck}+B\sigma^2, $$

where \(\sigma^2\) characterizes gradient noise and \(B\sigma^2\) is a noise-dependent error floor. The precise form depends on the method, step-size schedule, and assumptions.

A method can have excellent convergence order but poor robustness because its higher-order derivatives amplify noise.

### 6. A unified characterization: performance as a function of initialization

There is an especially useful way to combine local and global performance.

Instead of characterizing convergence from one starting point, define

\[ \boxed{ E_k(x_0)=f(T^k(x_0))-f^* } \]

for a deterministic optimizer.

This is an entire function describing optimization performance over initial conditions.

If initial conditions follow a distribution \(p_0(x)\), we obtain

\[ \boxed{ R_k=\mathbb E_{x_0\sim p_0}[E_k(x_0)]. } \]

This allows several additional characterizations.

Average-case performance:

\[ R_k=\int E_k(x)p_0(x)\,dx. \]

Worst-case performance:

\[ W_k=\sup_{x\in D}E_k(x). \]

Success probability:

\[ P_\epsilon(k) = \int \mathbf 1[E_k(x)\leq\epsilon]p_0(x)\,dx. \]

Convergence-time distribution:

\[ F_\epsilon(k)=\Pr(\tau_\epsilon\leq k). \]

These quantities reveal different aspects of the same optimization algorithm.

For stochastic optimizers, we additionally average over the algorithm's randomness.

#### A useful interpretation

We can view local and global convergence as two regimes of the same performance function:

- Local regime: \(x_0\) approaches an optimum, and the asymptotic order and rate constants dominate.
    
- Global regime: \(x_0\) varies over the entire domain, and basins, barriers, and exploration dominate.
    
- Intermediate regime: the algorithm must reach a region where fast local convergence becomes effective.
    

This suggests characterizing the algorithm by its time to enter a fast-convergence neighborhood, followed by its local convergence rate.

For example,

\[ T_\epsilon(x_0) \approx T_{\mathrm{entry}}(x_0) + T_{\mathrm{local}}(\epsilon). \]

This decomposition is particularly useful when comparing hybrid global-local optimization methods.

### 7. What is the most complete characterization?

There is no single scalar metric that captures all aspects of optimization performance.

A reasonably complete characterization for an optimizer \(A\) over a function class \(\mathcal F\) would be its anytime performance distribution:

$$ \boxed{ \Pr\left( \min_{j:\,\mathrm{cost}(j)\leq B} f(x_j)-f^* \leq\epsilon \;\middle|\; f,p_0,A \right). } $$

Here:

- \(B\) is the computational budget.
    
- \(\epsilon\) is the desired accuracy.
    
- \(f\) is the objective.
    
- \(p_0\) is the initialization distribution.
    
- The probability includes algorithmic randomness.
    

By varying \(B,\epsilon,f,p_0\), we can recover expected error, success probabilities, hitting-time distributions, robustness to initialization, and many other performance metrics.

Local convergence order, condition-number dependence, and oracle complexity then provide analytical descriptions or bounds for particular regions of this broader performance characterization.

One further distinction is important: all these metrics depend on the function class. Without restrictions on the objective, no optimizer can have uniformly superior global performance over all possible black-box functions. This is the motivation behind worst-case complexity theory, oracle lower bounds, and the no-free-lunch results.

For comparing new optimization methods, I would focus on three complementary objects: the local error recurrence, the distribution of time-to-accuracy, and the cost required to achieve that accuracy. Together they capture local speed, global reliability, and actual computational efficiency.

## constrained/noisy
Constrained and noisy optimization introduce additional performance dimensions that cannot be captured by objective gap or convergence order alone.

The key differences are:

- Constrained optimization: We must characterize feasibility, optimality within the feasible set, and convergence near its boundary.
    
- Noisy optimization: We must characterize estimation error, uncertainty, noise amplification, and the probability of reaching a desired accuracy.
    
- Combined: We must account for uncertainty in both the objective and constraints, potentially requiring probabilistic guarantees of feasibility.
    

### 1. Constrained optimization

Consider

\[ \begin{aligned} \min_x\quad & f(x)\\ \text{s.t.}\quad &g_i(x)\leq 0,\quad i=1,\ldots,m,\\ &h_j(x)=0,\quad j=1,\ldots,p. \end{aligned} \]

Let the feasible set be

\[ \mathcal C=\{x:g_i(x)\leq0,\ h_j(x)=0\}. \]

#### 1.1 Feasibility error

Before evaluating optimality, we need to measure constraint violation.

A common measure is

\[ \boxed{ V(x)= \sum_i[g_i(x)]_+^2+ \sum_j h_j(x)^2 } \]

where \([z]_+=\max(0,z)\).

An optimizer can have a very low objective value while remaining infeasible.

Consequently, constrained optimization has at least two independent errors:

\[ \big(f(x)-f^*,\ V(x)\big). \]

Note that the objective gap can be negative for infeasible points.

We can also characterize feasibility through distance to the feasible set:

\[ d_{\mathcal C}(x)=\inf_{y\in\mathcal C}\|x-y\|. \]

Under suitable constraint regularity, violation measures can bound this distance locally.

#### 1.2 Constrained stationarity

Ordinary gradient norm is no longer an appropriate stationarity measure.

At a constrained optimum, generally

\[ \nabla f(x^*)\neq0. \]

Instead, the first-order condition for a differentiable objective over a closed convex feasible set is

\[ \boxed{ 0\in\nabla f(x^*)+N_{\mathcal C}(x^*) } \]

where \(N_{\mathcal C}\) is the normal cone.

For general smooth nonlinear constraints, the corresponding KKT conditions are

\[ \nabla f(x)+ \sum_i\lambda_i\nabla g_i(x)+ \sum_j\nu_j\nabla h_j(x)=0, \]

with

\[ \lambda_i\geq0,\quad g_i(x)\leq0,\quad \lambda_i g_i(x)=0,\quad h_j(x)=0. \]

These conditions require appropriate constraint qualifications at a local optimum.

We can define a KKT residual:

\[ R_{\mathrm{KKT}}= \left\| \nabla f+\sum_i\lambda_i\nabla g_i+ \sum_j\nu_j\nabla h_j \right\| + V_{\mathrm{KKT}}, \]

where \(V_{\mathrm{KKT}}\) measures primal feasibility, dual feasibility, and complementarity.

This is analogous to the gradient norm in unconstrained optimization.

#### 1.3 Local convergence changes

The local convergence order can remain linear, quadratic, or higher, but additional properties matter.

|Property|Meaning|
|---|---|
|Active-set identification|How quickly the algorithm discovers which constraints bind|
|Tangential convergence|Convergence along feasible directions|
|Normal convergence|Correction of constraint violations|
|KKT conditioning|Sensitivity of the constrained solution|
|Constraint qualification|Whether the local optimality system is regular|
|Feasibility preservation|Whether iterates remain feasible|

For example, sequential quadratic programming (SQP) can achieve local quadratic convergence under suitable regularity and second-order conditions.

But its performance depends on the conditioning of the entire KKT system, not just the objective Hessian.

#### 1.4 Global constrained performance

We can generalize the earlier performance distribution to two dimensions:

\[ \boxed{ P_{\epsilon,\delta}(B)= \Pr\left( f(x_B)-f^*\leq\epsilon,\quad V(x_B)\leq\delta \right). } \]

Here \(B\) is computational budget.

This characterizes the probability of simultaneously achieving sufficient objective quality and feasibility.

For feasible outputs, we can use the simpler constrained objective gap. For infeasible outputs, it is important to report feasibility separately.

### 2. Noisy optimization

Suppose we cannot observe the objective exactly:

\[ Y(x)=f(x)+\xi(x), \]

where \(\xi\) is observation noise.

Alternatively, we might observe noisy gradients:

\[ G(x)=\nabla f(x)+\eta(x). \]

These are distinct oracle models and can lead to very different complexity bounds.

#### 2.1 Bias and variance

For a stochastic gradient estimator \(G(x)\), define

\[ b(x)=\mathbb E[G(x)]-\nabla f(x) \]

and

\[ \Sigma(x)=\operatorname{Cov}(G(x)). \]

Then

\[ \mathbb E\|G(x)-\nabla f(x)\|^2 = \|b(x)\|^2+\operatorname{tr}\Sigma(x). \]

This decomposition separates two fundamentally different problems:

- Variance can often be reduced through repeated sampling.
    
- Persistent bias generally cannot be eliminated merely by averaging.
    

An optimizer with small variance but significant bias may converge reliably to the wrong point.

#### 2.2 Convergence in expectation and probability

For deterministic optimization, we usually study

\[ f(x_k)-f^*. \]

For stochastic optimization, we can characterize several different notions.

Expected convergence:

\[ \mathbb E[f(x_k)-f^*]\to0. \]

Convergence in probability:

\[ \Pr(\|x_k-x^*\|>\epsilon)\to0. \]

Almost-sure convergence:

\[ \Pr\left(\lim_{k\to\infty}x_k=x^*\right)=1. \]

These are different guarantees. Expected objective convergence alone does not generally imply convergence of the iterates to a unique optimizer without additional assumptions.

#### 2.3 Noise-dependent convergence rates

Consider stochastic gradient descent on a \(\mu\)-strongly convex, smooth objective, with unbiased gradient noise of bounded variance \(\sigma^2\).

With an appropriate decreasing learning-rate schedule, a typical expected objective-gap bound is

\[ \boxed{ \mathbb E[f(x_k)-f^*] = O\left(\frac{\sigma^2}{\mu k}\right) } \]

up to smoothness-dependent constants, initialization terms, and assumptions on the stochastic oracle.

Compare this with deterministic gradient descent:

\[ f(x_k)-f^*=O(q^k). \]

The noise changes the asymptotic behavior from geometric convergence to a typically polynomial rate.

With constant learning rates, SGD instead generally approaches a noise-dependent neighborhood of the optimum.

#### 2.4 Noise amplification and stability

An optimizer can be highly sensitive to noisy gradients even if its deterministic convergence is excellent.

Consider Newton's method:

\[ x_{k+1}=x_k-H(x_k)^{-1}G(x_k). \]

With gradient noise covariance \(\Sigma\), the noise-induced update covariance is

\[ \boxed{ \operatorname{Cov}(\Delta x) = H^{-1}\Sigma H^{-T} } \]

when \(H\) is treated as exact and fixed at the current point.

Small Hessian eigenvalues can strongly amplify noise.

This gives another useful performance characteristic: noise gain, or the mapping from oracle noise to update uncertainty.

It also helps explain why regularization, trust regions, and damping can improve practical Newton-type methods.

### 3. Noise versus computational budget

A central tradeoff is whether to spend computational resources on more optimization steps or more accurate estimates per step.

Suppose averaging \(m\) independent gradient samples reduces variance to

\[ \operatorname{Var}(\bar G)=\frac{\sigma^2}{m}. \]

If the total budget is \(B=km\), increasing \(m\) leaves fewer iterations \(k\).

This leads to a more informative characterization:

\[ \boxed{ R(B)= \inf_{\text{sampling and update policies}} \mathbb E[f(x_B)-f^*]. } \]

It measures the best attainable accuracy under a fixed total computational budget, for a specified oracle and algorithm class.

This also connects to adaptive sampling: use fewer samples when far from the optimum, and increase accuracy as the gradient becomes small.

### 4. Stochastic constraints

Now consider

\[ \min_x \mathbb E_\xi[F(x,\xi)] \]

subject to uncertain constraints.

There are several fundamentally different formulations.

#### Expected constraints

\[ \mathbb E_\xi[g(x,\xi)]\leq0. \]

These constrain average behavior but permit individual violations.

#### Chance constraints

\[ \boxed{ \Pr(g(x,\xi)\leq0)\geq1-\alpha. } \]

Here \(\alpha\) is the permitted violation probability.

#### Robust constraints

\[ g(x,\xi)\leq0 \quad\forall\xi\in\mathcal U. \]

These require feasibility for every uncertainty realization in the specified set.

#### Distributionally robust constraints

\[ \inf_{P\in\mathcal P} P(g(x,\xi)\leq0) \geq1-\alpha. \]

These require a chance-constraint guarantee across an entire family of plausible distributions.

Their performance cannot be evaluated using the same feasibility metric without specifying the uncertainty model.

For chance-constrained optimization, for example, we can characterize both objective quality and actual constraint violation probability.

### 5. Additional characteristics for noisy and constrained optimization

|Dimension|Characterization|
|---|---|
|Objective accuracy|\(f(x)-f^*\)|
|Feasibility|\(V(x)\) or \(d_{\mathcal C}(x)\)|
|Stationarity|KKT residual or projected-gradient residual|
|Statistical accuracy|Bias, variance, confidence intervals|
|Reliability|Probability of reaching an \(\epsilon\)-optimal feasible point|
|Sample efficiency|Number of noisy oracle calls|
|Stability|Sensitivity to noise and perturbations|
|Safety|Probability or magnitude of constraint violations during optimization|
|Asymptotic consistency|Whether error vanishes with unlimited observations|
|Robustness to misspecification|Sensitivity to incorrect noise or constraint models|

The distinction between final feasibility and feasibility throughout optimization is especially important. An algorithm might return a perfectly feasible solution but evaluate thousands of infeasible points while searching.

In physical experiments or safety-critical systems, those intermediate violations may be unacceptable.

### 6. A general performance formulation

We can extend the earlier anytime-performance distribution.

Let an algorithm interact with a noisy objective and constraint oracle, using a total computational budget \(B\).

Define

\[ \boxed{ \mathcal P_A(\epsilon,\delta,B) = \Pr\left( f(x_B)-f^*\leq\epsilon,\; V(x_B)\leq\delta \right). } \]

The probability can incorporate initialization, oracle noise, and algorithm randomness.

But this still does not capture everything. We may also need to constrain the probability of unsafe intermediate evaluations:

\[ \Pr\left( x_t\in\mathcal C \;\forall t\leq B \right)\geq1-\alpha. \]

And we can introduce uncertainty about the objective itself by evaluating performance across a distribution of functions \(P(f)\).

That yields a general framework:

\[ \boxed{ \mathcal P_A(\epsilon,\delta,B;P(f),P(x_0),P(\xi)). } \]

It characterizes how reliably an algorithm achieves a specified objective and feasibility accuracy, given the available budget and distributions of problems, initializations, and noise.

### 7. An important additional distinction: statistical versus optimization error

For noisy optimization, the observed objective may itself only approximate the quantity we actually want to minimize.

Suppose

\[ f(x)=\mathbb E_\xi[F(x,\xi)] \]

but we optimize a finite-sample estimate

\[ \widehat f_N(x)=\frac1N\sum_{i=1}^N F(x,\xi_i). \]

Let \(\widehat x_N^*\) minimize \(\widehat f_N\). Then

\[ \begin{aligned} f(x_k)-f(x^*) ={}& \underbrace{\widehat f_N(x_k)-\widehat f_N(\widehat x_N^*)}_{\text{optimization error}}\\ &+ \underbrace{f(x_k)-\widehat f_N(x_k) +\widehat f_N(\widehat x_N^*)-f(x^*)}_{\text{statistical discrepancy}}. \end{aligned} \]

The second term is not necessarily nonnegative, but can be bounded using uniform generalization estimates or other statistical arguments.

This distinction matters because improving the numerical optimizer cannot eliminate the error caused by insufficient or unrepresentative observations.

The broader conclusion: optimization performance is not just convergence speed. It is the relationship between computational resources, objective accuracy, constraint satisfaction, and statistical reliability.

For deterministic unconstrained problems, much of this collapses to convergence rate and complexity. For constrained and noisy problems, these dimensions remain genuinely independent.
# Local optimization


We can think of optimization as iterative process, where for some existing point $x_k$ we propose next $x_{k+1}$ that is closer to minima or improves the objective. The simplest method is just following the gradient (or derivative in this case):
$$x_{k+1}=x_k-\eta\nabla L(x_k)$$
This simply looks at where does the function increases the most and walks in an opposite direction. Then it either decreases indefinitely (gradient is never zero) or eventually cant decrease further.
## Optimization as root-finding
A natural way to extend this is to treat the optimization as finding the root $\nabla L(x)=0$. Then classic gradient descent is a solution to an approximation of the loss as linear loss function:
$$L(x+h)\approx L(x)+\nabla L(x)h$$
But we can actually do better by using more taylor terms. Recall vector taylor expansion:
$$\nabla L(x+h)=\nabla L(x)+\nabla^2 L(x)h+\nabla^3 L(x)\frac {h^2} 2+...$$
### Newton's method
We can simply solve for $h$ that makes $\nabla L(x+h)=0$. Truncating at the linear term gives us the classic newton's method:
$$x_{k+1}=x_k-\eta (\nabla^2L)^{-1}\nabla L$$
But directly solving for $h$ in higher order cases becomes increasingly complex and generally intractable. Instead we can split it into a sequence of steps $h_k$ that each remove the next order. We can iteratively express next step in terms of the prev ones and eliminate more orders.

Lets assume that $h_k$ are coeffs in the expansion of some $h(t)=\sum_i t^ih_i$. At $t=0$ we get the origin, while at $t=1$ exactly the overall step. Then by requiring $\nabla L(x+h(t))=0$ and matching powers we get a sequence of linear equations basically.

By collecting all combinations of indices whose sum is $k$, we obtain:
$$ \boxed{ \nabla^2Lh_k= -\sum_{m=2}^{k}\frac1{m!} \sum_{\substack{i_1+\cdots+i_m=k\\i_j\ge1}} \nabla^mL(x)[h_{i_1},\ldots,h_{i_m}] } $$
with initial term
$$ Jh_1=-F(x)$$
In the nonlinear terms, $m\ge2$, and all indices are positive. If $i_1+\cdots+i_m=k$, then necessarily $i_j\le k-m+1\le k-1$. So none of those terms can contain $h_k$. The only occurrence of $h_k$ is therefore the linear Taylor term $\nabla^2Lh_k$. These steps can be viewed as instances of taylor/lagrange inversion.

### Broyden's method
We can approximate these gradients "as we go", which generally make optimization cheaper, as we need less explicit evaluations of the gradients. At iteration $k$, maintain an approximation $B_k\approx J(x_k)$. Compute 
$$B_ks_k=-F(x_k),\qquad x_{k+1}=x_k+s_k$$

After evaluating $F(x_{k+1})$, define $y_k=F(x_{k+1})-F(x_k)$. The secant condition is $B_{k+1}s_k=y_k$.
One update satisfying this is
$$ \boxed{ B_{k+1} =B_k+ \frac{(y_k-B_ks_k)s_k^\top}{s_k^\top s_k} }$$
This is Broyden's good update, a rank-one correction.

The intuition is that every iteration provides a small experiment: _moving by \(s_k\) caused the function to change by \(y_k\)._ The approximation should therefore map \(s_k\) to \(y_k\). Among matrices satisfying that requirement, the first Broyden update makes a particularly small rank-one correction to the previous approximation, allowing information accumulated during earlier iterations to be retained.

We could instead maintain $H_k\approx J_F(x_k)^{-1}$. The inverse secant condition is $H_{k+1}y_k=s_k$.
Bad Broyden minimizes the Frobenius-norm change to the inverse approximation:
$$ \min_H\|H-H_k\|_F^2 \quad\text{subject to}\quad Hy_k=s_k$$
Its solution is
$$ \boxed{ H_{k+1} =H_k+\frac{(s_k-H_ky_k)y_k^\top}{y_k^\top y_k} } $$
This is also a rank-one update. The next step is simply $x_{k+1}=x_k-H_kF(x_k)$. Unlike an implementation that explicitly solves with $B_k$, this uses a matrix-vector multiplication with $H_k$.

In many common optimization scenarios it appears to perform worse, hence "bad" update. Although its not a universal characteristic and it actually can be better in other cases. We could even try maintaining both and measuring the disagreement to compensate the different errors they give.

### Newton–Krylov method
Instead of maintaining explicit estimations, we can apply Krylov solver to find steps in high dimensional contexts.

For large \(n\), explicitly constructing and inverting an \(n\times n\) Hessian is expensive.

Newton's method requires solving

\[ Hh=-g, \]

but solving this system does not necessarily require storing \(H\).

Krylov methods such as conjugate gradients (for positive-definite \(H\)) or GMRES can approximately solve the system using only products

\[ v\mapsto Hv. \]

For optimization, these are Hessian-vector products.

We can compute them through directional differentiation:

\[ Hv =\left.\frac{d}{d\epsilon}\nabla f(x+\epsilon v)\right|_{\epsilon=0}. \]

With automatic differentiation, an exact Hessian-vector product (up to floating-point error) can often be computed at a cost comparable to a small number of gradient evaluations.

Algorithm

1. Compute \(g=\nabla f(x)\).
    
2. Define an operator \(v\mapsto Hv\).
    
3. Approximately solve \(Hh=-g\) using a Krylov solver.
    
4. Update \(x\leftarrow x+\alpha h\), using a line search or trust region when appropriate.
    

For an indefinite Hessian, additional safeguards are needed to ensure a useful optimization direction.
### Homotopy continuation: follow stationary points across objectives

Instead of solving the original root equation directly, construct a family of easier problems.

For example,

\[ F(x,t)=(1-t)\nabla q(x)+t\nabla f(x), \qquad t\in[0,1], \]

where \(q\) is a simple function with a known stationary point.

At \(t=0\), solve

\[ \nabla q(x)=0. \]

Then gradually increase \(t\), tracking a solution \(x(t)\) satisfying

\[ F(x(t),t)=0. \]

Differentiate with respect to \(t\):

\[ J_F(x(t),t)\frac{dx}{dt} +\frac{\partial F}{\partial t}=0. \]

Therefore,

\[ \boxed{ \frac{dx}{dt} =-J_F^{-1}\frac{\partial F}{\partial t}. } \]

For the particular interpolation above,

\[ \frac{dx}{dt} =-\big[(1-t)\nabla^2q+t\nabla^2f\big]^{-1} \big[\nabla f-\nabla q\big]. \]

This provides a differential equation describing how a stationary point moves as the objective changes.

### Householder's methods
A particularly interesting family is Householder's methods. For scalar \(F\), their general form is

\[ \boxed{ x_{\mathrm{new}} =x+p\frac{(1/F)^{(p-1)}}{(1/F)^{(p)}} } \]

where \(p\ge1\), and derivatives are taken with respect to \(x\).

For \(p=1\), this gives Newton's method. For \(p=2\), it gives Halley's method. Under standard simple-root assumptions, the method has local convergence order \(p+1\).

There is no equally simple, universally canonical multivariate version of this formula, because division and reciprocal differentiation of vector-valued functions are not directly analogous to the scalar case.

By considering how exactly do we 
https://chatgpt.com/c/6af786c2-6cc4-83eb-9dbb-a8aebdf8a726?no_universal_links=1
https://chatgpt.com/c/6af82c17-04d4-83ed-801d-139b43bebb02?no_universal_links=1

predictor-corrector
We can use roots of unity or fractional steps to arbitrarily accelerate the convergence, as they remove progressively more degrees.

try multiple steps (2h, 3h, etc, Richardson extrapolation), root's of unity, and implicit methods, finite differences, bisection, stochastic

### Implicit root-finding with higher-order approximations

We can go further than implicit gradient descent.

For a root \(x^*\), the fundamental theorem of calculus gives

\[ F(x^*)-F(x) =\int_0^1 J_F(x+t(x^*-x))(x^*-x)\,dt. \]

Since \(F(x^*)=0\), writing \(h=x^*-x\),

\[ \boxed{ F(x)+\left[\int_0^1J_F(x+th)\,dt\right]h=0. } \]

This is an exact implicit root equation.

Ordinary Newton approximates the integral by its value at \(t=0\):

\[ \int_0^1J_F(x+th)\,dt\approx J_F(x). \]

But we can use better quadrature formulas.

### Newton's finite-difference expansion

Taylor's formula is

\[ F(x+h)=\sum_{k=0}^{\infty}\frac{F^{(k)}(x)}{k!}h^k. \]

For a uniform grid with spacing \(s\), define the forward difference operator

\[ \Delta_sF(x)=F(x+s)-F(x). \]

Higher differences are

\[ \Delta_s^2F(x)=F(x+2s)-2F(x+s)+F(x), \]

and, generally,

\[ \Delta_s^kF(x) =\sum_{j=0}^k(-1)^{k-j}\binom kjF(x+js). \]

Newton's forward-difference interpolation formula is

\[ \boxed{ F(x+h)\approx \sum_{k=0}^{m} \binom{h/s}{k}\Delta_s^kF(x) } \]

where

\[ \binom tk=\frac{t(t-1)\cdots(t-k+1)}{k!}. \]

This is the finite-difference analogue of a truncated Taylor expansion.

The difference is the basis:

- Taylor uses powers \(h^k\).
    
- Newton interpolation uses falling factorials \(\binom{h/s}{k}\).
    

Both represent the same polynomial space through degree \(m\), but they use different information to determine its coefficients.

### First-order finite-difference root-finding

Keep only \(k=0,1\):

\[ F(x+h)\approx F(x)+\frac hs\Delta_sF(x). \]

Set this approximation to zero:

\[ F(x)+\frac hs[F(x+s)-F(x)]=0. \]

Solving for \(h\),

\[ \boxed{ h=-s\frac{F(x)}{F(x+s)-F(x)}. } \]

This is a finite-difference Newton step.

If we choose \(s=x_k-x_{k-1}\), then \(x+s\) is not the previous point, so instead use the backward difference

\[ F'(x_k)\approx \frac{F(x_k)-F(x_{k-1})}{x_k-x_{k-1}}. \]

We obtain the familiar secant method:

\[ \boxed{ x_{k+1} =x_k-F(x_k) \frac{x_k-x_{k-1}}{F(x_k)-F(x_{k-1})}. } \]

Its local convergence order is

\[ p=\frac{1+\sqrt5}{2}\approx1.618, \]

under standard simple-root assumptions.

This is slower per iteration than Newton's quadratic convergence, but it requires no derivative evaluations.

### Second-order finite-difference root-finding

Retain the second difference:

\[ F(x+h)\approx F(x)+t\Delta_sF(x) +\frac{t(t-1)}2\Delta_s^2F(x), \]

where \(t=h/s\).

Set the interpolant equal to zero:

\[ \boxed{ F(x)+t\Delta_sF(x) +\frac{t(t-1)}2\Delta_s^2F(x)=0. } \]

This is a quadratic equation for \(t\), and hence for the next root estimate \(x+h\).

It uses three function evaluations:

\[ F(x),\quad F(x+s),\quad F(x+2s). \]

Unlike the quadratic Taylor method, it does not require \(F'\) or \(F''\).

This is closely related to Müller's method, which fits a quadratic polynomial through three points and selects one of its roots as the next iterate.

Müller's method has local convergence order approximately

\[ \boxed{p\approx1.8393} \]

under suitable conditions, when using the usual three-point iteration.

An important distinction: fitting a quadratic to three samples does not automatically yield cubic convergence. The interpolation error depends on how far the sample points are from the actual root, and those points themselves evolve during iteration.

### Higher-order finite-difference root solvers

We can generalize immediately:

\[ P_m(h)= \sum_{k=0}^{m} \binom{h/s}{k}\Delta_s^kF(x). \]

Then choose

\[ \boxed{h:\ P_m(h)=0.} \]

For increasing \(m\), this constructs higher-degree polynomial approximations to \(F\).

But there is an interesting alternative.

Instead of approximating \(F(x)\) and solving the resulting polynomial, approximate its inverse:

\[ x=G(y),\qquad y=F(x). \]

We know the desired value of \(y\): zero.

Given samples

\[ (x_0,F(x_0)),\ldots,(x_m,F(x_m)), \]

interpolate the inverse function \(G\) and evaluate it at zero.

For example, quadratic inverse interpolation gives

\[ \boxed{ x_{\mathrm{new}} =\sum_{i=0}^{2} x_i\prod_{j\ne i} \frac{-F(x_j)}{F(x_i)-F(x_j)}. } \]

This is inverse quadratic interpolation, an important component of Brent-type root solvers.

Unlike direct polynomial interpolation, inverse interpolation produces a new root estimate without having to solve a polynomial equation.

For a simple root, the inverse function is locally smooth, making this a natural representation.
### Bisection: root-finding through interval elimination

Suppose

\[ F:\mathbb R\to\mathbb R \]

is continuous and we know

\[ F(a)<0,\qquad F(b)>0. \]

By the intermediate value theorem, there is at least one root in \([a,b]\).

Evaluate the midpoint

\[ m=\frac{a+b}{2}. \]

If \(F(m)<0\), replace \(a\) with \(m\). Otherwise replace \(b\) with \(m\), unless \(F(m)=0\).

Each evaluation halves the interval.

Interactive bisection of \(F(x)=x^2-2\)

Bisection steps

4

Lower bound

1.375000

Upper bound

1.437500

Interval width

0.062500

After \(n\) steps,

\[ \boxed{b_n-a_n=\frac{b_0-a_0}{2^n}.} \]

This gives a guaranteed error bound for a root inside the interval.

Unlike Newton's method, bisection does not require differentiability or a good initial guess near the root. It only requires continuity and a valid sign-changing bracket.

However, it converges linearly, not quadratically.

### Can we improve bisection?

Yes. The most straightforward improvement is to combine bracketing with interpolation.

For example, the secant estimate between bracket endpoints is

\[ c=\frac{aF(b)-bF(a)}{F(b)-F(a)}. \]

This is the regula falsi or false-position estimate.

Rather than choosing the midpoint, it estimates where the graph crosses zero.

Algorithms such as Brent's method combine bisection, secant steps, and inverse quadratic interpolation.

They retain the safety of bracketing while often converging much faster than pure bisection.

### Bisection applied to optimization

For a one-dimensional differentiable objective,

\[ F(x)=f'(x). \]

If

\[ f'(a)<0,\qquad f'(b)>0, \]

then there is at least one stationary point inside the interval.

If \(f\) is convex, this brackets a global minimizer over the interval.

But for a nonconvex function, it only guarantees a stationary point, not necessarily the global minimum.

There is also a subtle distinction: binary search for a minimum need not use gradients at all.

For a unimodal function, we can use ternary search or golden-section search, which compare function values and eliminate portions of the interval.

That is a different kind of information from root bracketing.

### Stochastic root-finding: noisy evaluations

Suppose we cannot evaluate \(F(x)\) exactly.

Instead, we observe

\[ Y(x,\xi)=F(x)+\varepsilon(x,\xi) \]

with

\[ \mathbb E[Y(x,\xi)\mid x]=F(x). \]

The goal is still

\[ F(x^*)=0, \]

but every observation is noisy.

This is the setting of Robbins–Monro stochastic approximation.

A basic iteration is

\[ \boxed{ x_{k+1}=x_k-\eta_kY(x_k,\xi_k). } \]

This looks like gradient descent, but its interpretation is broader: it is a stochastic root solver.

For \(F=\nabla f\), it becomes stochastic gradient descent.

### Convergence

For a strongly monotone \(F\), appropriate noise assumptions, and step sizes satisfying

\[ \sum_k\eta_k=\infty,\qquad \sum_k\eta_k^2<\infty, \]

the iteration can converge almost surely to the root under standard additional regularity conditions.

A common choice is

\[ \eta_k=\frac{c}{k+1}. \]

The decreasing step size prevents persistent noise from producing an irreducible random walk around the solution.

Under suitable strongly monotone conditions, averaged stochastic approximation can achieve expected squared error of order \(O(1/n)\), corresponding to root-mean-square error \(O(n^{-1/2})\).

This is fundamentally different from deterministic Newton's quadratic local convergence.

### Stochastic Newton methods

We can combine stochastic approximation with curvature information.

Suppose we have a noisy estimate \(\widehat F_k\) and a Jacobian estimate \(\widehat J_k\).

Then

\[ \boxed{ x_{k+1} =x_k-\eta_k\widehat J_k^{-1}\widehat F_k. } \]

For optimization, this becomes stochastic Newton or stochastic quasi-Newton.

The Jacobian approximation can be obtained from minibatches, finite differences, running secant estimates, or other curvature estimators.

The main difficulty is that noise affects both the residual and the inverse Jacobian. In particular,

\[ \mathbb E[\widehat J^{-1}] \ne(\mathbb E[\widehat J])^{-1} \]

in general.

This creates bias and potentially severe instability.

Regularization, damping, averaging, and positive-definite approximations are therefore especially important.

### Stochastic bisection

Now combine the first two ideas.

Suppose \(F\) is continuous, strictly increasing, and has a unique root.

We maintain a bracket \([a,b]\), but at each midpoint we observe a noisy sign.

For example,

\[ Y(m)=F(m)+\varepsilon. \]

A single observation may have the wrong sign.

If we immediately discard half the interval based on that observation, we can accidentally eliminate the true root.

One solution is to take repeated samples at the midpoint:

\[ \overline Y_N(m)=\frac1N\sum_{i=1}^NY_i(m). \]

If the noise is sub-Gaussian with variance proxy \(\sigma^2\), then

\[ P\left( |\overline Y_N-F(m)|\ge t \right) \le2\exp\left(-\frac{Nt^2}{2\sigma^2}\right). \]

Therefore, to determine the sign with error probability at most \(\delta\), a sufficient sample size is approximately

\[ \boxed{ N\ge \frac{2\sigma^2}{F(m)^2} \log\frac2\delta. } \]

This reveals an important limitation: as the midpoint approaches the root, \(F(m)\to0\), so identifying its sign becomes increasingly expensive.

### Probabilistic bisection

Instead of making a hard decision, maintain a probability distribution over the root location:

\[ p_k(x^*). \]

After observing a noisy sign at query point \(q\), update it using Bayes' rule:

\[ p_{k+1}(x^*) \propto P(Y_q\mid x^*)p_k(x^*). \]

This is the basis of probabilistic bisection algorithms.

A natural query is the posterior median, since it divides the current probability mass into equal halves.

With a known, constant sign-observation accuracy greater than \(1/2\), this can be interpreted as noisy twenty-questions search.

For additive observation noise, however, sign accuracy depends on the distance to the root, so the model must account for that.

### Stochastic methods can also estimate derivatives

Another approach is to use random perturbations to estimate the Jacobian.

For scalar \(F:\mathbb R^n\to\mathbb R\), choose a random vector \(u\) with

\[ \mathbb E[uu^\top]=I. \]

Then the estimator

\[ \widehat{\nabla F}(x) = \frac{F(x+hu)-F(x-hu)}{2h}u \]

satisfies

\[ \mathbb E[\widehat{\nabla F}(x)] =\nabla F(x)+O(h^2) \]

under suitable smoothness and moment assumptions.

For vector-valued \(F\), the analogous random Jacobian estimator is

\[ \boxed{ \widehat J(x) = \frac{[F(x+hu)-F(x-hu)]u^\top}{2h}. } \]

This connects root-finding to random-direction finite differences and simultaneous perturbation methods.

For optimization, where \(F=\nabla f\), these estimators approximate the Hessian using gradient evaluations.

If only noisy objective values are available, simultaneous perturbation stochastic approximation (SPSA) can estimate gradients with only two objective evaluations per iteration, independent of dimension.

That can be useful in high-dimensional black-box optimization, although the estimator variance may grow substantially with dimension.

### Randomized Newton and sketching

There is another kind of stochastic root-finding where the randomness is not observational noise.

Suppose we want to solve

\[ J(x)h=-F(x). \]

Instead of solving the entire linear system, randomly select a subset of equations or project the system into a smaller subspace.

For example, choose a random sketching matrix \(S\) and solve

\[ SJ(x)h=-SF(x). \]

With suitable regularization or a minimum-norm criterion, this gives a randomized approximate Newton direction.

Related methods include randomized Kaczmarz, randomized coordinate Newton, and sketch-and-project methods.

Their advantage is computational: they may make each iteration much cheaper, even if more iterations are required.

They are particularly useful when \(n\) is large and the Jacobian is sparse, structured, or expensive to factorize.


### Can binary search generalize to multiple dimensions?

There is no direct analogue of scalar sign bracketing for arbitrary vector-valued root problems.

In one dimension, the sign of a continuous function at two endpoints certifies a root between them.

In multiple dimensions, knowing the signs of components of \(F\) at a few points generally does not provide the same geometric guarantee.

However, there are several related approaches.

Coordinate-wise bisection: If the problem has suitable monotonicity or separability, solve one coordinate equation at a time while holding others fixed.

Interval Newton methods: Maintain a box \(X\subset\mathbb R^n\) and use interval bounds on the Jacobian to certify whether a root exists or is unique within that box.

Topological methods: Use conditions such as the Poincaré–Miranda theorem, a multidimensional analogue of the intermediate value theorem, to certify roots from boundary sign conditions.

Subdivision methods: Repeatedly partition the search domain into smaller boxes and eliminate regions that cannot contain a root.

For general nonconvex optimization, these methods can be combined with interval bounds on \(\nabla f\) and \(f\), producing deterministic global optimization algorithms. Their worst-case cost, however, typically grows exponentially with dimension.

### Non polynomial root-finding
### Trigonometric, Fourier, exp/log families

For

\[ F(x)=a\sin(bx+c)+d, \]

roots are available using inverse trigonometric functions, with periodic branches.

More interestingly, consider a finite Fourier series:

\[ F(x)=\sum_{k=-m}^{m}c_ke^{ikx}. \]

Substitute

\[ z=e^{ix}. \]

Then

\[ F(x)=\sum_{k=-m}^{m}c_kz^k. \]

Multiplying by \(z^m\) gives

\[ \boxed{ P(z)=\sum_{k=-m}^{m}c_kz^{k+m}=0. } \]

We have reduced a trigonometric polynomial to an ordinary polynomial of degree at most \(2m\).

To recover real \(x\), retain roots satisfying

\[ |z|=1 \]

and take

\[ x=\arg z\pmod{2\pi}. \]

This is useful for oscillatory functions, where a Taylor polynomial may require many terms over a large interval.

It also connects naturally to your earlier interest in Fourier and chirplet approximations.
## Optimization as differential equation evolution

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
# Global optimization

Generally what we want is a kind of certificate of how close we are to the minimum, preferably in terms of the objective value. That means for every $\epsilon$ we can find $x$ such that its loss is within this distance of global optimum $x^*=\arg\min f(x)$:
$$||f(x^*)-f(x)||<\varepsilon$$
Then the problem is to iteratively improve it.

We can also study any related functions that share optimizers within some bounds. If we specifically want global optimums to match, we dont need arbitrary uniform accuracy between two functions, only that it preserves the global minima specifically:
$$ |\min g-\min f|<\varepsilon \\ |\arg\min g-\arg\min f|<\delta$$

Then if finding optimum of $g$ is simpler in any sense, we can instead solve for its global optimum. Moreover, even if we restrict $g$ to be convex, $f$ still can be arbitrarily nonconvex.

For example, a region could satisfy the Polyak–Łojasiewicz inequality:

$$ \boxed{ \frac12\|\nabla f(x)\|^2 \geq \mu(f(x)-f_i^*) } $$

for some $\mu>0$, where $f_i^*$ is the minimum over that region.

This is weaker than strong convexity.

A nonconvex function can satisfy this inequality, allowing gradient descent to converge to a global minimum of the region under appropriate smoothness and feasibility assumptions.

We can separate this problem into broad-phase and narrow-phase:
1. Initially we simply try to narrow down the neighborhood of the optimum - certify the region where it must be.
2. Then search for the region where the global optimum is the attractor.
3. Then we can apply any local method to approach it arbitrarily close.

https://arxiv.org/abs/1803.02021?utm_source=chatgpt.com

basin detection - take f and test point x, run de solution until found the minima, evaluate. If two points converge to the same minima they are the same basin.

annealing and tempering
https://chatgpt.com/c/6af4c9ed-5048-83ed-a172-25a4957e9031
https://www.semanticscholar.org/paper/Collective-Annealing-by-Switching-Temperatures%3A-a-Blondeel-Pareschi/6b3b7512d74d964b4988fcc39c4321ad92d3afa2

tunneling
https://www.semanticscholar.org/paper/The-Tunneling-Algorithm-for-the-Global-Minimization-Levy-Montalvo/3da72047ef0687e917cdad2a7c9971126c1a82e8

[Global optimization in Hilbert space](https://link.springer.com/article/10.1007/s10107-017-1215-7?fromPaywallRec=true)
[A novel parameter-free differentiable filled function for global optimization](https://link.springer.com/article/10.1038/s41598-026-46030-x?fromPaywallRec=true)
paywalled https://www.semanticscholar.org/paper/Novel-Global-Optimization-Algorithm-with-a-Curve-Wang-Yang/3a76d7e262b097c9cccc8fb7fd1c9546cc210de6
paywalled https://www.semanticscholar.org/paper/The-parallel-tunneling-method-G%C3%B3mez-Castillo/6fb8a9c7fad15cb20d1203e1ee40d50fb76308c8
paywalled https://www.semanticscholar.org/paper/A-Transformation-Function-for-Finding-Better-Minima-Wang-Shang/61626dc05ce49807421b74c2dd4268b12016b4ad
paywalled https://www.semanticscholar.org/paper/New-Classes-of-Globally-Convexized-Filled-Functions-Lucidi-Piccialli/15600389d1f7ef02d22da7c6dbb003322c297acf
paywalled https://www.semanticscholar.org/paper/Finding-Global-Minima-with-a-Computable-Filled-Liu/4dbd9e60b24c6d7b6acf758449faeaefcd491def
paywalled https://www.semanticscholar.org/paper/An-Evolutionary-Algorithm-for-Global-Optimization-Wang-Dang/3b131f176da316f2251f8feaba1547c32f0e44f2

A useful heuristic is therefore

$$ \boxed{ \Delta\beta \sim \frac{c}{\sqrt{\operatorname{Var}_{\pi_\beta}(f)}}}$$
### Gibbs distribution

Even if we know where all minima are, the Gibbs distribution becomes increasingly concentrated as \(\beta\) increases.

Near a nondegenerate minimum \(x^*\),

\[ f(x)\approx f(x^*)+\frac12f''(x^*)(x-x^*)^2. \]

Therefore,

\[ p_\beta(x)\propto \exp\left( -\frac{\beta f''(x^*)}{2}(x-x^*)^2 \right). \]

The peak width scales as

\[ \boxed{\sigma\sim\frac1{\sqrt{\beta f''(x^*)}}.} \]

A uniform integration grid must resolve that width. For fixed curvature, this requires \(N=O(\sqrt{\beta})\) points over a fixed one-dimensional domain.

Adaptive quadrature can reduce this cost dramatically if it can locate the peaks first.

Inverse CDF sampling works on unbounded intervals as well. The main difference is that we must handle infinite integration limits and ensure that the Gibbs distribution is normalizable.

For \(x\in\mathbb R\),

\[ p_\beta(x)=\frac{e^{-\beta f(x)}}{Z_\beta}, \qquad Z_\beta=\int_{-\infty}^{\infty}e^{-\beta f(t)}\,dt. \]

The CDF is

\[ F_\beta(x)= \frac{\int_{-\infty}^{x}e^{-\beta f(t)}\,dt} {Z_\beta}. \]

As before, sample \(u\sim U(0,1)\) and solve \(F_\beta(x)=u\).

#### 1. First requirement: finite partition function

We need

\[ \boxed{Z_\beta<\infty.} \]

This is not guaranteed merely because \(f\) has a global minimum or grows toward infinity.

For example:

|Objective \(f(x)\)|Normalizable on \(\mathbb R\)?|
|---|---|
|\(x^2\)|Yes, for \(\beta>0\)|
|\(\lvert x\rvert\)|Yes, for \(\beta>0\)|
|\(\log(1+x^2)\)|Only for \(\beta>1/2\)|
|\(\log\log(e+\lvert x\rvert)\)|No, for any finite \(\beta>0\)|
|\(\sin x\)|No|

The important property is how rapidly \(e^{-\beta f(x)}\) decays compared with the volume of the domain.

In \(d\) dimensions, even logarithmic growth can suffice, but it must overcome the increasing volume:

\[ f(x)\sim c\log\|x\| \quad\Longrightarrow\quad Z_\beta<\infty \ \text{if }\beta c>d, \]

under suitable regularity assumptions on the asymptotic behavior.

[gibbs corrections](https://chatgpt.com/c/6aff60ec-cb90-83eb-8513-c4ba9e8e5418?no_universal_links=1)
[infinum principles](https://chatgpt.com/c/6aff63ff-0e00-83ed-9c40-c160ef307229?no_universal_links=1)
[ball proximal and convexity](https://chatgpt.com/c/6affee69-31a8-83eb-9fd2-a4a2eed5bcec?no_universal_links=1)
[cbo non unique optimums](https://chatgpt.com/c/6affeb74-2464-83ed-a041-f4e35313c636)
#### probabilistic characterization of global optimality
Define

\[ \boxed{ I_q(x)=\Pr_{Y\sim q(\cdot\mid x)}[f(Y)<f(x)]. } \]

This is the probability of finding a better point with one proposal.

If \(q\) has full support and \(f\) is continuous on a suitable domain, then

\[ I_q(x)=0 \quad\Longleftrightarrow\quad x\text{ is a global minimizer}, \]

assuming every nonglobal point has a positive-measure region of strictly better values.

This is a probabilistic characterization of global optimality.

And with \(n\) independent proposals, the probability of missing every improvement is

\[ \boxed{ \Pr(\text{no improvement in }n\text{ trials}) = (1-I_q(x))^n. } \]

So if we can lower-bound the probability of proposing an improvement whenever the current point is more than \(\varepsilon\) from optimality, we can obtain finite-sample guarantees.

For example, if

\[ f(x)-f^*>\varepsilon \implies I_q(x)\ge c_\varepsilon>0, \]

then observing no improvements in \(n\) independent proposals has probability at most

\[ (1-c_\varepsilon)^n \]

whenever the point is not \(\varepsilon\)-optimal.
#### 2. Numerical approach: truncate the tails

The simplest approach is to find \(a,b\) such that

\[ P(X<a)+P(X>b)\leq\varepsilon. \]

Then compute the CDF on \([a,b]\).

If we sample from the truncated distribution, its total variation distance from the original distribution is exactly the excluded tail probability.

The challenge is determining \(a,b\) reliably.

If we know a lower bound on the growth of \(f\), we can bound the omitted mass.

For example, suppose

\[ f(x)\geq c|x|^p+C \]

for sufficiently large \(|x|\), with \(c,p>0\).

Then

\[ \int_R^\infty e^{-\beta f(x)}dx \leq e^{-\beta C} \int_R^\infty e^{-\beta cx^p}dx. \]

The right-hand side is expressible through the incomplete gamma function:

\[ \int_R^\infty e^{-\beta cx^p}dx = \frac{\Gamma(1/p,\beta cR^p)} {p(\beta c)^{1/p}}. \]

So we can choose \(R\) to achieve a specified tail-mass error, provided we also have a suitable lower bound on \(Z_\beta\).

#### 3. Alternative: map the infinite domain to a finite interval

We can transform \(x\in\mathbb R\) into \(t\in(0,1)\).

For example,

\[ x(t)=\tan\left(\pi(t-\tfrac12)\right). \]

Then

\[ \frac{dx}{dt} = \pi\sec^2\left(\pi(t-\tfrac12)\right). \]

The partition function becomes

\[ Z_\beta= \int_0^1 e^{-\beta f(x(t))} \pi\sec^2\left(\pi(t-\tfrac12)\right)\,dt. \]

Now we can use ordinary finite-interval quadrature and inverse CDF sampling in \(t\).

The transformed density is

\[ \boxed{ p_T(t)\propto e^{-\beta f(x(t))} \left|\frac{dx}{dt}\right|. } \]

This is exact mathematically, but it doesn't automatically make integration easy.

The Jacobian can create strong endpoint behavior, and a poor transformation can concentrate nearly all the probability in a tiny part of \((0,1)\).

A better transformation should approximately match the tails of the target Gibbs distribution.

For example, a Gaussian reference distribution is often preferable when \(f(x)\) grows quadratically.

#### 4. A more useful approach: choose a reference distribution

Let \(q(x)\) be a distribution with a known inverse CDF \(Q^{-1}\).

Set

\[ x=Q^{-1}(u),\qquad u\in(0,1). \]

Since

\[ \frac{dx}{du}=\frac1{q(x)}, \]

the transformed Gibbs density is

\[ \boxed{ p_U(u)\propto \frac{e^{-\beta f(Q^{-1}(u))}} {q(Q^{-1}(u))}. } \]

If \(q\) is close to the Gibbs distribution, this transformed density is approximately uniform.

In the ideal case \(q=p_\beta\), it is exactly uniform, and sampling requires no further correction.

This gives us a useful interpretation:

Inverse CDF sampling is easiest when we can find a reference distribution that already captures most of the target's structure.

For instance, if

\[ f(x)=\frac{x^2}{2}+\epsilon\sin x, \]

we can choose

\[ q(x)=\mathcal N(0,1/\beta). \]

Then

\[ p_U(u)\propto \exp\left[-\beta\epsilon \sin\left(Q^{-1}(u)\right)\right]. \]

The quadratic component disappears completely from the transformed integration problem.

We only need to integrate the residual correction.

This is closely related to importance sampling and transport methods.
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

antithetic sampling

Quasi-Monte Carlo. Choose points that cover the integration domain systematically rather than independently. For sufficiently regular functions, randomized QMC can achieve substantially better rates than ordinary Monte Carlo, and deterministic QMC has error bounds approaching \(O(N^{-1})\) for certain function classes.

Higher-order quadrature. If the function is smooth, polynomial or spectral approximations can converge much faster, sometimes exponentially in the approximation order for analytic functions.

Multilevel Monte Carlo. Construct increasingly accurate approximations \(f_\ell\) and estimate differences between levels. These differences become progressively smaller, allowing fewer samples on expensive levels.

sampling corresponding gibbs distribution with temp approaching 0.

[inverse cdf sampling](https://chatgpt.com/c/6afb13ef-fc54-83ed-8239-510cc25217cc?no_universal_links=1)
[distribution decomposition](https://chatgpt.com/c/6afb19a7-a554-83ed-bef5-1b9504204e49?no_universal_links=1)
[moment generating function (MFG)](https://chatgpt.com/c/6afb5a92-e0bc-83eb-8fc5-c6074520fc83?no_universal_links=1)
[levelset curvature and implicit methods for constrained optimization](https://chatgpt.com/c/6afc08a2-18f4-83eb-a6c5-706baff0af72?no_universal_links=1)
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
https://chatgpt.com/c/6ad63b91-80a8-83eb-84d7-6a1c9be7c831

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
https://chatgpt.com/c/6aff3a6d-ce64-83ed-9ef1-1f06bf6df42f?no_universal_links=1
[distribution of "where to sample next"](https://chatgpt.com/c/6aff1ed2-a7fc-83eb-9d9c-a2b448dfbcf6?no_universal_links=1)

https://www.semanticscholar.org/paper/The-Tunneling-Algorithm-for-the-Global-Minimization-Levy-Montalvo/3da72047ef0687e917cdad2a7c9971126c1a82e8
https://www.semanticscholar.org/paper/GTCOA%3A-a-novel-golden-t-distribution-chaotic-and-in-Liu-Wang/ca1809a1882ffe4d4a569ba5316484439268eda7
https://www.semanticscholar.org/paper/CBO-algorithm-with-average-drift-and-applications-Bae-Ha/da11ea94cb9c6d3cc82a1aa1e786b11c9b1aa429
https://www.semanticscholar.org/paper/Swarm-based-optimization-with-jumps%3A-a-kinetic-BGK-Borghi-Im/6c776c9d67fb6e1cc96ecc0bf0baebf342ef4961
[adaptive importance sampling](https://www.semanticscholar.org/paper/Consensus-based-rare-event-estimation-Althaus-Papaioannou/7ebc5bc2da5d0f9c4f52e1f27ee68753ce516100)
https://www.semanticscholar.org/paper/Adaptive-State-Dependent-Diffusion-for-Optimization-Engquist-Ren/67bf0bea4ab441e1b4cbb14d9852b63d6813eaab
https://www.semanticscholar.org/paper/Adaptive-State-Dependent-Diffusion-for-Optimization-Engquist-Ren/67bf0bea4ab441e1b4cbb14d9852b63d6813eaab
Tensor Cross Interpolation
https://www.semanticscholar.org/paper/Kinetic-models-for-optimization%3A-a-unified-for-Borghi-Herty/9597ca9d7e4727e578db92a235ab87c789308015
https://www.semanticscholar.org/paper/Kinetic-description-and-convergence-analysis-of-for-Borghi-Pareschi/9858ff923b2d4067e64daab22b24b4cf6ec76fad
https://www.semanticscholar.org/paper/Binary-Interaction-Methods-for-High-Dimensional-and-Benfenati-Borghi/3d4e681fe3bd77c678b96acc6e2bbefbd9c41701
https://chatgpt.com/c/6aff5314-46c0-83eb-952a-18647f48cfe5?no_universal_links=1
https://www.semanticscholar.org/paper/Convergence-of-Anisotropic-Consensus-Based-in-Law-Fornasier-Klock/a33b7aca0c39b694f7a0ed48f4381f2ece16986d
https://www.semanticscholar.org/paper/Enhancing-Exploration-in-Global-Optimization-by-in-Serr%C3%A9-Germain/76468582cc075ed06ed47612fe55e560059c6f43
https://www.semanticscholar.org/paper/Importance-Sampling-Optimization-with-Laplace-Dragomir-Portier/c80d58d2d46b239a23d58f252e35831a5e795242
https://www.semanticscholar.org/paper/Consensus-based-optimization-with-%24/alpha%24-stable-Aceves-S%C3%A1nchez-Albi/31fb269aef2851c65232f11e4d7a0c71105dd96b
https://www.semanticscholar.org/paper/Strong-Global-Convergence-of-the-Consensus-Based-Bonandin-Riedl/7b24308b057113f62d7b92bdc005ee4414753c03
https://www.semanticscholar.org/paper/A-modified-Consensus-Based-Optimization-model%3A-and-Choi-Lee/76bd40f0cab5d72e8ed8dd3bf31b41907921f03b
https://www.semanticscholar.org/paper/Consensus-Based-Optimization-Beyond-Finite-Time-Bianchi-Dragomir/e4856697fce97f0a5024bb9a3605336319c70b55
https://www.semanticscholar.org/paper/Model-predictive-control-strategies-using-Borghi-Herty/e43d9d92a125b35d8bb991d4315fe12e854f8ef7
https://www.semanticscholar.org/paper/Consensus-Based-Stochastic-Control-Lyu-Chen/25bf18839006da55b0dea7c1c29f6856037534d1
https://www.semanticscholar.org/paper/A-Unifying-Framework-for-Global-Optimization%3A-From-Serr%C3%A9-Kalogeratos/20ea596b0997dce26b7db1c945350c4d5179ada1
https://www.semanticscholar.org/paper/A-Consensus-Based-Algorithm-for-Multi-Objective-and-Borghi-Herty/595fc8f9031de3574848aa31776a1d9b5aaeec09
https://www.semanticscholar.org/paper/Convergence-of-time-discrete-finite-particle-based-Herty-Huang/5581ef3dda27ddc772fdf5c7bfa8f2b687ef06fd
https://www.semanticscholar.org/paper/An-Optimal-Structured-Zeroth-order-Algorithm-for-Rando-Molinari/048d457ec432ecfcbc090a1d92478410e7be9fe5
https://www.semanticscholar.org/paper/Consensus-Based-Optimization-of-Path-Integral-Free-Inoue-Huang/2ac83e0917924869aa536648fd6dac096e925f4f
https://www.semanticscholar.org/paper/Uniform-in-time-propagation-of-chaos-for-Gerber-Hoffmann/807f504655e991782327c711d4efbdae913d7b7c
https://www.semanticscholar.org/paper/A-consensus-based-optimization-method-for-nonsmooth-Wei-Wu/20ed12588f15df8dcad6b56bca47354769ddebc6
https://www.semanticscholar.org/paper/A-discrete-Consensus-Based-Global-Optimization-with-Bellavia-Malaspina/6017c955fe322d0cb75a22795f0a995b7285fe28
https://www.semanticscholar.org/paper/Consensus-based-optimization-with-memory-effects%3A-Borghi-Grassi/51318b09746da8ae9e3d3883838e8f8a20f7300c
https://www.semanticscholar.org/paper/Consensus-based-optimization-for-multi-objective-a-Klamroth-Stiglmayr/a5e01da689baa5b569b98cdd12712ceee6a87fdb
[CB2O](https://arxiv.org/abs/2411.13394)
https://www.semanticscholar.org/paper/Fast-and-Robust-Consensus-Based-Optimization-via-Huang-Herty/adac321f7a13388e44fdf7d8fb2c9fa4cc401154
https://www.semanticscholar.org/paper/Consensus-Based-Algorithms-for-Stochastic-Problems-Bonandin-Herty/9c750a9ded4fd494f670f1a2351cd7d556766115
https://www.semanticscholar.org/paper/Ensemble-based-gradient-inference-for-particle-in-Schillings-Totzeck/d8bd62c4f2c2cf746b85981343b6663c10f03b24
https://www.semanticscholar.org/paper/Constrained-consensus-based-optimization-Borghi-Herty/668077e10838c9492912fb79d8be52157b7f194a
https://www.semanticscholar.org/paper/Proximal-basin-hopping%3A-global-optimization-with-Lauga-Molinari/4e4fd2d0800c8b6eed93b2d8bd12e195dad121b0
https://www.semanticscholar.org/paper/Broximal-Alignment-for-Global-Non-Convex-Gruntkowska-Li/05edb0531c78426bda10637f3bb09f7c1e060203
https://www.semanticscholar.org/paper/Global-Optimization-via-Softmin-Energy-Minimization-Agazzi-Carlei/6ca140322fdace44fcce1596c93742efdb9bea6d
https://www.semanticscholar.org/paper/Kinetic-simulated-annealing-optimization-with-rate-Herty-Zanella/00448bff1d965d0bf93f982d04d28014e169ece6
https://www.semanticscholar.org/paper/Swarm-based-gradient-descent-meets-simulated-Ding-Guerra/b79d9574dfc91258d1a0b683e00ccf99c4cfb1e1
https://www.semanticscholar.org/paper/Stein-Boltzmann-Sampling%3A-A-Variational-Approach-Serr%C3%A9-Kalogeratos/70c211d3a150a2ee3e7b8f759381f99ff4247a23
[sample where it matters and cbo](https://www.semanticscholar.org/paper/SCHOOL-OF-COMPUTATION%2C-INFORMATION-AND-TECHNOLOGY-%E2%80%94/56aed06929239304e33c77cd7a00297d52fb33a0)
https://www.semanticscholar.org/paper/Discrete-Consensus-Based-Optimization-Byeon-Ha/5d3cb1a06f322305c857bd6242c1fe3bc0ab2433
https://www.semanticscholar.org/paper/Gradient-Is-All-You-Need-How-Consensus-Based-Can-Be-Riedl-Klock/af9a09f94c8b4f74181e939dfe08bedcc1a18b1d
https://www.semanticscholar.org/paper/Constrained-consensus-based-optimization-and-for-Beddrich-Chenchene/a39da437e5b828f8a72196e296284f3d722a8f0c
https://www.semanticscholar.org/paper/Exploiting-Structure-with-Anisotropic-Optimization-Bonandin-Riedl/a3b33d19327b25efb532404c26a29e68967cd26c
https://www.semanticscholar.org/paper/Exploiting-Structure-with-Anisotropic-Optimization-Bonandin-Riedl/a3b33d19327b25efb532404c26a29e68967cd26c
https://www.semanticscholar.org/paper/A-PDE-framework-of-consensus-based-optimization-for-Fornasier-Sun/8dcb17dd7339dc66f6c34e5e24cc23fff4335ec4
combining with adam seems to improve empirical risk in nn quite significantly [link](https://www.semanticscholar.org/paper/Mean-Field-Model-for-Two-Layer-Neural-Networks-with-Deyn-Herty/4938c968853f79e6d32e1bd6abf375f02c8cec66)
[binary interaction methods](https://chatgpt.com/c/6aff5314-46c0-83eb-952a-18647f48cfe5?no_universal_links=1)
[cbo perf and dynamic population counts](https://chatgpt.com/c/6aff3a6d-ce64-83ed-9ef1-1f06bf6df42f)
### Composite optimization
https://www.semanticscholar.org/paper/ProxiCBO%3A-A-Provably-Convergent-Consensus-Based-for-Zhang-Ma/e236d39885bd1f07467de9e429c49c069506ac54

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
https://chatgpt.com/c/6af09320-9380-83ed-9113-ad53098030a0
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

# metropolis acceptance and greedy opt

Your observation suggests a useful unification:

\[ \boxed{ \text{Optimization algorithm} = \text{Proposal mechanism} + \text{Selection mechanism}. } \]

The selection mechanism can be interpreted thermodynamically:

\[ \alpha_\beta(x,y)=\min(1,e^{-\beta\Delta f}). \]

At finite \(\beta\), it allows uphill moves.

At infinite \(\beta\), it becomes greedy optimization.

The proposal mechanism determines the geometry and reachability of the search.

This means many evolutionary algorithms can indeed be interpreted as zero-temperature Metropolis-like processes, even though they do not necessarily sample the canonical zero-temperature Gibbs measure.

coupling of proposal and current state - we can design proposal distribution to target specific behavior relative to current state, like exploration, exploitation, or independence.
## 3. Choosing the coupling by optimization

We could optimize the coupling itself.

For example, suppose we want proposals that improve the objective:

\[ \boxed{ \min_{C\in\Pi(p,r)} \mathbb E_{(X,Y)\sim C}[f(Y)-f(X)]_+. } \]

Here \(\Pi(p,r)\) is the set of all couplings with marginals \(p,r\).

But there's a surprising issue:

\[ \mathbb E_C[f(Y)-f(X)] = \mathbb E_r[f(Y)]-\mathbb E_p[f(X)]. \]

The expected signed improvement is independent of the coupling.

Therefore, optimizing the coupling only matters for nonlinear objectives such as:

\[ \mathbb E_C[(f(Y)-f(X))_+], \]

\[ \Pr_C(f(Y)<f(X)), \]

or expected accepted movement.

For a greedy optimizer, a particularly natural objective is

\[ \boxed{ \max_{C\in\Pi(p,r)} \Pr_C(f(Y)<f(X)). } \]

This is an optimal transport problem with cost

\[ c(x,y)=-\mathbf1_{\{f(y)<f(x)\}}. \]

It chooses how to pair current states and proposals so that as many pairs as possible represent improvements.

Importantly, this can change the distribution of accepted states even though the proposal marginal \(r\) is fixed.
## 4. For Gibbs sampling, optimize accepted movement instead

If our target is Gibbs,

\[ \pi_\beta(x)\propto e^{-\beta f(x)}, \]

we must include the MH correction.

For a coupling \(C\) whose first marginal is \(\pi_\beta\), its conditional proposal is

\[ q_C(y\mid x)=\frac{C(x,y)}{\pi_\beta(x)}. \]

Then

\[ \pi_\beta(x)q_C(y\mid x)=C(x,y). \]

The MH accepted flow is

\[ \boxed{ S_C(x,y)=\min(C(x,y),C(y,x)). } \]

Thus, the expected accepted squared jumping distance is

\[ \boxed{ J(C)= \iint\|x-y\|^2\min(C(x,y),C(y,x))\,dx\,dy. } \]

We can therefore formulate coupling design as

\[ \boxed{ \max_{C\in\Pi(\pi_\beta,r)}J(C). } \]

This is an interesting constrained optimization problem over joint distributions.

It tries to construct a proposal that moves far while preserving the desired stationary distribution after acceptance.

However, maximizing jump distance alone does not necessarily maximize mixing or the spectral gap. A deterministic oscillation between two regions can have large jumps while exploring poorly.
# Population based and bayesian opt connection
There is a genuine connection here, and it is stronger than merely saying that Bayesian optimization and population optimization both maintain distributions. But the Gibbs correspondence is exact only under certain conditions, and it does not generally make the joint law redundant.

## 1. Gibbs gives a correspondence between objectives and distributions

For a fixed reference measure \(m\), define

\[ p_f(x)=\frac{e^{-\beta f(x)}}{Z_f}, \qquad Z_f=\int e^{-\beta f(x)}m(dx). \]

Conversely, for a strictly positive density \(p\) with respect to \(m\),

\[ \boxed{ f(x)=-\frac1\beta\log p(x)+C. } \]

Therefore, modulo additive constants, Gibbs gives a one-to-one correspondence between suitable objectives and strictly positive probability densities.

So your observation is correct in this sense:

\[ \boxed{ \text{Objective functions modulo constants} \;\longleftrightarrow\; \text{Positive Gibbs densities}. } \]

And this extends to uncertainty over objectives.

If Bayesian optimization maintains a posterior

\[ P_t(df), \]

we can push it through the Gibbs map \(G_\beta:f\mapsto p_f\):

\[ \boxed{ Q_t=(G_\beta)_\#P_t. } \]

Now \(Q_t\) is a distribution over probability distributions:

\[ Q_t\in\mathcal P(\mathcal P(E)). \]

Thus, a Bayesian posterior over objective functions can indeed be represented as a distribution over Gibbs distributions, without losing information about the objective modulo additive constants.

This is precisely the kind of higher-order probability object we encountered in population optimization.

## 2. But the two distributions over distributions are not generally the same

The distinction is what their randomness represents.

For Bayesian optimization,

\[ Q_t=\operatorname{Law}(G_\beta(F)\mid D_t). \]

It represents uncertainty about the unknown objective, expressed as uncertainty about its Gibbs distribution.

For population optimization,

\[ \Pi_t^N=\operatorname{Law}(\mu_t^N). \]

It represents randomness in the population generated by the algorithm.

Even when the objective \(f\) is known and fixed, \(\Pi_t^N\) can remain nondegenerate.

Conversely, even when the population is completely deterministic, the Bayesian posterior \(Q_t\) can remain highly uncertain.

So these are generally different probability measures on the same space.

The Gibbs transformation does not by itself identify them.

## 3. There is nevertheless a natural way to make them coincide

Suppose the objective is random:

\[ F\sim P_t. \]

Given \(F=f\), independently sample a population from its Gibbs distribution:

\[ X_1,\ldots,X_N\mid F=f \overset{\mathrm{iid}}{\sim}G_\beta(f). \]

Then define

\[ \mu^N=\frac1N\sum_i\delta_{X_i}. \]

We obtain a hierarchical model:

\[ F\longrightarrow G_\beta(F)\longrightarrow\mu^N. \]

As \(N\to\infty\),

\[ \mu^N\to G_\beta(F) \]

almost surely, conditional on \(F\), under standard assumptions.

Consequently,

\[ \boxed{ \operatorname{Law}(\mu^N) \Rightarrow (G_\beta)_\#P_t=Q_t. } \]

This is an exact connection between the two representations in the infinite-population limit.

For finite populations, the empirical distribution introduces additional sampling noise.

However, ordinary evolutionary algorithms do not generally generate conditionally independent Gibbs samples. Their population laws need not satisfy this relationship.

## 4. What this means for optimization algorithms

Your interpretation suggests treating the Gibbs distribution as the common underlying state:

\[ p_f(x)\propto e^{-\beta f(x)}. \]

Then different algorithms can be understood as manipulating different representations of it.

For example:

|Algorithm|Representation|
|---|---|
|Bayesian optimization|Posterior over \(f\), hence posterior over \(p_f\)|
|Gibbs sampling / MCMC|Particles approximating \(p_f\)|
|Variational inference|Parametric approximation \(q_\theta\approx p_f\)|
|Population optimization|Evolving empirical distribution \(\mu_t^N\), potentially toward \(p_f\)|
|Transport / diffusion|Transformations mapping a simple distribution toward \(p_f\)|

This is a useful unification, but there is a crucial distinction: not every population optimizer is actually approximating the Gibbs distribution.

For example, greedy DE tends to eliminate worse candidates and may collapse its population. Its limiting distribution can differ substantially from any finite-temperature Gibbs distribution.

Nevertheless, we can interpret its population as an approximation to some effective objective:

\[ \boxed{ f_t^{\mathrm{eff}}(x) = -\frac1\beta\log p_t(x)+C_t. } \]

This works whenever \(p_t\) has a strictly positive density.

Then the population optimizer induces an evolution of effective objectives:

\[ p_{t+1}=\mathcal T(p_t) \]

corresponding to

\[ \boxed{ f_{t+1}^{\mathrm{eff}} = G_\beta^{-1}\circ\mathcal T\circ G_\beta (f_t^{\mathrm{eff}}). } \]

This is a conjugation of the distributional evolution by the Gibbs transformation.

It allows us to reinterpret population dynamics as dynamics over functions.

But \(f_t^{\mathrm{eff}}\) is generally not the true objective \(f\), nor a Bayesian estimate of it. It describes the current search distribution.

## 5. The deeper connection: optimization as evolution of an effective energy

This is where I think your idea becomes particularly useful.

Instead of treating the objective \(f\) and the population distribution \(p_t\) as unrelated objects, define the effective energy

\[ E_t(x)=-\frac1\beta\log p_t(x). \]

Then ask how a population optimizer transforms \(E_t\).

For example, consider pure Gibbs reweighting:

\[ p_{t+1}(x) = \frac{p_t(x)e^{-\eta f(x)}} {\int p_t(y)e^{-\eta f(y)}dy}. \]

Taking negative logarithms,

\[ -\log p_{t+1}(x) = -\log p_t(x)+\eta f(x)+C_t. \]

Therefore,

\[ \boxed{ E_{t+1}(x) = E_t(x)+\frac{\eta}{\beta}f(x)+C_t'. } \]

So multiplicative selection in probability space becomes additive accumulation of objective information in energy space.

If initially \(p_0\) is uniform on a bounded domain, then after \(t\) such updates,

\[ p_t(x)\propto e^{-t\eta f(x)}. \]

This is exactly a Gibbs distribution with increasing inverse temperature.

Other population operators give different energy transformations.

For example, convolution mutation gives

\[ E_{t+1}(x) = -\frac1\beta \log\left[ p_t*k_t \right](x)+C, \]

which is a log-convolution or soft-min operation.

That connects directly to the Moreau/Hamilton–Jacobi smoothing you have been investigating.

Thus, population mutation and selection can be viewed as alternating smoothing and sharpening operations on an effective energy landscape.

## 6. Where your proposed equivalence is strongest

The strongest common representation is not simply a distribution over distributions, but a probability measure over energy landscapes modulo additive constants.

Through Gibbs, this is equivalent to a measure over strictly positive probability densities.

Bayesian optimization evolves such a measure by conditioning on observed objective values.

Population-based optimization evolves a random empirical measure through stochastic search operators.

When population particles are explicitly interpreted as samples from uncertain Gibbs distributions, these representations can be connected through a hierarchical sampling model.

But without that assumption, their laws are different.

There is also a subtle issue with normalization: Gibbs removes additive constants from \(f\). If Bayesian observations contain absolute objective values, those constants matter to the likelihood. To preserve the full Bayesian posterior, we must either retain the additive constant or explicitly work with objectives modulo constants.

The useful unification is therefore an algebra of transformations on energy landscapes and probability measures, rather than an unconditional identification of Bayesian posteriors with population laws.

This gives a concrete way to compare algorithms: translate each population operation into its induced transformation of the effective energy, then compare that transformation with Bayesian updating, Gibbs reweighting, and Moreau smoothing.

# moreau envelope generalization

We generalized the Moreau envelope by replacing its quadratic distance penalty with an arbitrary two-point cost:

\[ \boxed{ M_cf(x)=\inf_{y\in\mathcal X} \left[f(y)+c(x,y)\right]. } \]

The main idea was to find conditions on \(c(x,y)\) that preserve the global optimum while making the transformed objective easier to optimize, ideally convex or at least pseudoconvex.

We identified several levels of conditions.

## 1. Preserve the global minimum value

We required

\[ \boxed{c(x,y)\ge0,\qquad c(x,x)=0.} \]

These guarantee

\[ \min_x M_cf(x)=\min_y f(y), \]

assuming the minima exist.

The reasoning is straightforward:

- Nonnegative cost prevents the envelope from going below \(f^*\).
    
- Zero diagonal cost lets us recover \(f^*\) at any global minimizer.
    

## 2. Preserve exactly the global minimizers

We strengthened the condition to

\[ \boxed{c(x,y)=0\iff x=y.} \]

Then, assuming the relevant infima are attained,

\[ \boxed{\arg\min M_cf=\arg\min f.} \]

This was important because a cost that vanishes at distinct points can introduce additional global minimizers.

Strict positivity away from the diagonal is sufficient, although not necessary for a particular objective.

## 3. Make the transformed objective convex

We considered

\[ M_cf(x)=\inf_y F(x,y), \qquad F(x,y)=f(y)+c(x,y). \]

A sufficient condition is that \(F(x,y)\) be jointly convex in \((x,y)\), because partial minimization preserves convexity.

However, this is generally too restrictive when \(f\) is nonconvex.

We also established a fundamental limitation: if \(f\) has multiple separated global minimizers, exact preservation of its minimizer set is incompatible with convexity of \(M_cf\).

This motivated looking for weaker properties, such as pseudoconvexity.

## 4. Make every point move toward the global minimizer

For a unique minimizer \(x^*\), we considered the envelope gradient.

If

\[ y_x\in\arg\min_y[f(y)+c(x,y)] \]

is unique and sufficient differentiability conditions hold, Danskin's theorem gives

\[ \nabla M_cf(x)=\nabla_xc(x,y_x). \]

We derived a sufficient directional condition:

\[ \boxed{ \left\langle \nabla_xc(x,y_x),x-x^* \right\rangle>0 \quad\forall x\ne x^*. } \]

This guarantees that the negative gradient points inward relative to \(x^*\), so gradient flow decreases the squared distance to the optimum.

It is stronger than merely requiring pseudoconvexity, but gives a useful geometric guarantee.

For the soft envelope,

\[ M_{c,\beta}f(x) =-\frac1\beta\log \int e^{-\beta[f(y)+c(x,y)]}\,dy, \]

the corresponding condition becomes

\[ \boxed{ \left\langle \mathbb E_{Y\mid x}[\nabla_xc(x,Y)], x-x^* \right\rangle>0. } \]

This connects directly to our Gibbs-sampling discussion.

## 5. Relate the cost to growth of the objective

In our more recent discussion, we examined when the ordinary quadratic Moreau cost can completely eliminate nonglobal minima.

We found that an upper Lipschitz bound alone is insufficient. What matters is a lower growth bound around the global minimizer:

\[ f(y)-f^*\ge\omega(\|y-x^*\|). \]

For a domain contained in a ball of radius \(R\) around \(x^*\), we obtained the sufficient quadratic-cost condition

\[ \lambda\ge \sup_{0<r\le R} \frac{(2Rr-r^2)_+}{2\omega(r)}. \]

Under this condition,

\[ M_\lambda f(x) =f^*+\frac{\|x-x^*\|^2}{2\lambda}. \]

Thus the envelope becomes exactly quadratic, not merely convex.

For linear growth \(\omega(r)=\mu r\), the simpler sufficient bound is

\[ \lambda\ge\frac R\mu. \]

## 6. Higher-order costs

We also considered extending the quadratic penalty with higher-order terms:

\[ c_t(x,y)= \frac{\|x-y\|^2}{2t} +\sum_{j=3}^{k}t^{-\alpha_j}C_j(x,y). \]

The motivation was to cancel higher-order terms in local expansions, potentially producing an envelope closer to a quadratic around the optimum.

But these corrections must preserve the nonnegativity and zero-set conditions if we want exact preservation of global minimizers.

## Where this leaves us with pseudoconvexity

We can now state a less restrictive target than convexification:

\[ \boxed{ \begin{aligned} &c(x,y)\ge0,\\ &c(x,y)=0\iff x=y,\\ &M_cf\text{ is pseudoconvex}. \end{aligned} } \]

For differentiable \(M_cf\), the last condition can be expressed directly in terms of the cost:

\[ \boxed{ \langle\nabla_xc(x,y_x),z-x\rangle\ge0 \implies M_cf(z)\ge M_cf(x). } \]

This is a condition on the resulting envelope, not a simple pointwise condition on \(c\) alone.

The remaining research question is whether we can derive tractable sufficient conditions directly on \(c(x,y)\) that guarantee pseudoconvexity without requiring joint convexity of \(f(y)+c(x,y)\).

That would be especially valuable because it could retain the global optimization guarantees we want while allowing much more flexible two-point costs than the quadratic Moreau penalty.

We can formulate this as a minimum-cost convexification or pseudoconvexification problem: find the smallest two-point cost \(c(x,y)\) that makes the soft envelope satisfy the desired gradient condition.

Recall the soft envelope:

\[ F_{c,\beta}(x) =-\frac1\beta\log\int_{\mathcal X} e^{-\beta[f(y)+c(x,y)]}\,dy. \]

Its gradient is

\[ \nabla F_{c,\beta}(x) =\mathbb E_{Y\sim q_c(\cdot\mid x)} [\nabla_xc(x,Y)], \]

where

\[ q_c(y\mid x)= \frac{e^{-\beta[f(y)+c(x,y)]}} {\int e^{-\beta[f(z)+c(x,z)]}\,dz}. \]

For a known unique global minimizer \(x^*\), our earlier directional condition was

\[ \boxed{ \left\langle \mathbb E_{q_c}[\nabla_xc(x,Y)], x-x^* \right\rangle>0 \quad (x\ne x^*). } \]

We want the smallest cost satisfying this condition.

There is an important distinction: minimizing the magnitude of \(c\) is not necessarily equivalent to minimizing how much the envelope changes \(f\).

## 1. The minimum-magnitude formulation

A natural optimization problem is

\[ \boxed{ \begin{aligned} \min_{c}\quad &\|c\|\\ \text{s.t.}\quad &c(x,y)\ge0,\\ &c(x,x)=0,\\ &\langle\nabla F_{c,\beta}(x),x-x^*\rangle>0 \quad(x\ne x^*). \end{aligned} } \]

Here \(\|c\|\) could be a supremum norm, an integral norm, or an expected cost.

However, this problem generally has no nonzero minimum.

Consider the quadratic cost

\[ c_\varepsilon(x,y)=\frac{\varepsilon}{2}\|x-y\|^2. \]

For arbitrarily small \(\varepsilon>0\), the cost approaches zero. On a compact domain, its soft envelope approaches a constant, but its gradient can still point toward the optimum.

For example, with \(f(y)=y^2/2\) on \(\mathbb R\), direct integration gives

\[ F_{c_\varepsilon,\beta}(x) =\text{constant}+ \frac{\varepsilon}{2(1+\varepsilon)}x^2. \]

Therefore,

\[ xF'_{c_\varepsilon,\beta}(x) =\frac{\varepsilon}{1+\varepsilon}x^2>0 \]

for every \(x\ne0\), regardless of how small \(\varepsilon\) is.

So the infimum cost is zero, but the limiting envelope is constant and no longer satisfies the strict condition.

We need a quantitative condition, not just a positive gradient direction.

## 2. Require a minimum strength of convergence

One possibility is

\[ \boxed{ \langle\nabla F_{c,\beta}(x),x-x^*\rangle \ge\mu\|x-x^*\|^2 } \]

for some prescribed \(\mu>0\).

This ensures that gradient flow converges toward \(x^*\) at least exponentially in distance:

\[ \|x_t-x^*\|\le e^{-\mu t}\|x_0-x^*\|. \]

Now minimizing the cost is meaningful because arbitrarily weak costs cannot satisfy a fixed positive \(\mu\).

For the Gaussian example above, the condition requires

\[ \frac{\varepsilon}{1+\varepsilon}\ge\mu, \]

so, for \(0<\mu<1\),

\[ \boxed{\varepsilon_{\min}=\frac{\mu}{1-\mu}.} \]

This gives an exact minimum within the quadratic-cost family.

## 3. But minimizing distortion of \(f\) is more interesting

Suppose we want the envelope to remain as close as possible to the original objective.

Instead of minimizing \(\|c\|\), we could solve

\[ \boxed{ \begin{aligned} \min_c\quad& \|F_{c,\beta}-f\|_{L^2(\rho)}^2\\ \text{s.t.}\quad& \langle\nabla F_{c,\beta}(x),x-x^*\rangle \ge\mu\|x-x^*\|^2. \end{aligned} } \]

Here \(\rho\) is a reference distribution over the domain.

We might also allow an additive constant, since constants do not affect optimization:

\[ \min_{c,a}\|F_{c,\beta}-f-a\|_{L^2(\rho)}^2. \]

This is closer to your original goal: remove only the parts of \(f\) that prevent global convergence, preserving everything else.

There is an important complication: the soft envelope does not generally preserve minimizers merely because \(c\ge0\) and \(c(x,x)=0\). Those conditions work for the hard infimum, but integration introduces entropy effects. We therefore need to impose mode preservation separately, or ensure it through additional assumptions.

## 4. The ideal transformed function can be characterized directly

We can first optimize over the transformed function \(F\), without restricting it to a particular two-point cost.

For example, define

\[ \boxed{ \begin{aligned} F^*=\arg\min_F\quad& \frac12\|F-f\|_{L^2(\rho)}^2\\ \text{s.t.}\quad& \langle\nabla F(x),x-x^*\rangle \ge\mu\|x-x^*\|^2. \end{aligned} } \]

This has a useful property: the optimization problem is convex in \(F\).

The objective is quadratic, and the gradient inequality is a linear constraint on \(F\).

In a suitable Sobolev function space, with appropriate boundary and regularity assumptions, this becomes a projection problem onto a convex set of functions.

It tells us the minimum distortion required to obtain the desired directional geometry.

We could then investigate which two-point costs realize or approximate that projection.

## 5. What about pseudoconvexity instead?

For pseudoconvexity, we can use the weaker condition

\[ \nabla F(x)^\top(y-x)\ge0 \implies F(y)\ge F(x). \]

But unlike the fixed directional condition, this constraint is generally nonconvex in \(F\).

A practical alternative is to require only that no nonglobal stationary points exist, supplemented by a quantitative gradient condition to prevent arbitrarily slow convergence.

That leads to three different optimization targets:

|Target|What it minimizes|Main difficulty|
|---|---|---|
|Minimum \(\\|c\\|\)|Size of the two-point penalty|Degenerate zero-cost limit|
|Minimum \(\\|F-f\\|\)|Distortion of the original objective|Realizing \(F\) as a soft envelope|
|Minimum change subject to pseudoconvexity|Modification of optimization geometry|Nonconvex feasibility condition|

The second formulation is particularly promising because we can characterize the optimal transformed function independently of how we implement it.

One further possibility is to minimize the change in the Gibbs distribution itself:

\[ \boxed{ \min_c D_{\mathrm{KL}}\left( p_{\beta,f}\,\|\,p_{\beta,F_{c,\beta}} \right) } \]

subject to the same directional condition.

That would preserve the original distribution as much as possible while changing its optimization geometry.

The central insight is that the smallest useful cost is determined by the strength of the convergence guarantee we demand. Without specifying that strength, the cost can often approach zero while the transformed objective becomes arbitrarily flat.

Yes. The quadratic (Gaussian) cost is only one possibility. We can derive entire families of costs from the geometry we want the soft envelope to have.

The most interesting alternatives are nonquadratic radial costs, anisotropic costs, Bregman divergences, and costs designed directly from the gradient condition.

## 1. General framework

Recall

\[ F_{c,\beta}(x) =-\frac1\beta\log\int e^{-\beta[f(y)+c(x,y)]}\,dy. \]

The induced conditional distribution is

\[ q_c(y\mid x)\propto e^{-\beta[f(y)+c(x,y)]}. \]

And the envelope gradient is

\[ \boxed{ \nabla F_{c,\beta}(x) =\mathbb E_{q_c}[\nabla_xc(x,Y)]. } \]

We want a cost that is nonnegative, vanishes on the diagonal, and makes this gradient point toward the global minimizer.

## 2. Families worth considering

|Cost \(c(x,y)\)|Induced kernel|Main characteristic|
|---|---|---|
|\(\frac{\lambda}{2}\\|x-y\\|^2\)|Gaussian|Smooth, analytically convenient|
|\(\lambda\\|x-y\\|\)|Laplace-type|Heavier tails than Gaussian|
|\(\frac{\lambda}{p}\\|x-y\\|^p\)|Generalized Gaussian|Adjustable tail decay|
|\(\lambda\log(1+\\|x-y\\|^2/s^2)\)|Student-type|Polynomial tails|
|\(\frac12(x-y)^\top A(x-y)\)|Anisotropic Gaussian|Direction-dependent smoothing|
|\(D_\phi(y,x)\)|Bregman kernel|Non-Euclidean geometry|
|Huber\((\\|x-y\\|)\)|Hybrid|Quadratic locally, linear in tails|

How different costs change the sampling kernel

Power \(p\)

3.0

Unnormalized kernels \(e^{-|x-y|^p/p}\), with a Gaussian reference. Larger \(p\) produces flatter centers and faster tail decay.

## 3. Derive the cost from a desired gradient

Rather than choosing a kernel and hoping it works, we can reverse the problem.

Suppose we want

\[ \nabla F_{c,\beta}(x)=g(x) \]

for a prescribed vector field \(g\).

Then the cost must satisfy

\[ \boxed{ \frac{\int\nabla_xc(x,y)e^{-\beta[f(y)+c(x,y)]}\,dy} {\int e^{-\beta[f(y)+c(x,y)]}\,dy} =g(x). } \]

This is a nonlinear integral equation for \(c\).

If we choose

\[ g(x)=\mu(x-x^*), \]

the desired envelope is quadratic:

\[ F(x)=\frac{\mu}{2}\|x-x^*\|^2+C. \]

That is the strongest possible form of the directional condition, but requires knowing \(x^*\).

A more practical approach is to parameterize \(c\) and minimize the violation of the gradient condition.

## 4. Radial costs give a useful simplification

Let

\[ c(x,y)=h(\|x-y\|). \]

Writing \(r=\|x-y\|\), we obtain

\[ \nabla_xc(x,y) =\frac{h'(r)}{r}(x-y). \]

Therefore,

\[ \boxed{ \nabla F_{c,\beta}(x) =\mathbb E_{q_c} \left[ \frac{h'(R)}{R}(x-Y) \right]. } \]

Define

\[ w(r)=\frac{h'(r)}r. \]

Then

\[ \nabla F_{c,\beta}(x) =\mathbb E[w(R)(x-Y)]. \]

This makes the design problem particularly clear: the radial cost determines how strongly samples at different distances influence the gradient.

For a quadratic cost, \(w(r)=\lambda\), so all distances receive the same gradient coefficient.

For \(h(r)=\lambda r^p/p\),

\[ w(r)=\lambda r^{p-2}. \]

- \(p<2\): relatively greater influence from nearby samples.
    
- \(p=2\): distance-independent coefficient.
    
- \(p>2\): relatively greater influence from distant samples.
    

These are gradient coefficients; the actual sampling probabilities also change with \(p\).

## 5. An especially interesting alternative: Bregman costs

Given a differentiable strictly convex function \(\phi\), define

\[ D_\phi(y,x) =\phi(y)-\phi(x)-\nabla\phi(x)^\top(y-x). \]

Then

\[ c(x,y)=\lambda D_\phi(y,x). \]

Its gradient is

\[ \boxed{ \nabla_xc(x,y) =\lambda\nabla^2\phi(x)(x-y). } \]

Thus

\[ \nabla F_{c,\beta}(x) =\lambda\nabla^2\phi(x) \left(x-\mathbb E[Y\mid x]\right). \]

The Hessian of \(\phi\) acts as a position-dependent metric.

This is useful when the objective is highly anisotropic or when Euclidean distance poorly represents its geometry.

For example, choosing \(\phi\) to approximate local curvature could produce a cost adapted to the shape of \(f\), rather than using isotropic Gaussian smoothing.

However, the orientation of the Bregman divergence matters: \(D_\phi(x,y)\) produces a different gradient.

## 6. Can we derive a minimum-distortion cost?

There is another approach that connects directly to your previous question.

Choose a baseline cost \(c_0\) and perturb it:

\[ c_\varepsilon(x,y)=c_0(x,y)+\varepsilon h(x,y). \]

Differentiating the soft envelope with respect to \(\varepsilon\) gives

\[ \boxed{ \left.\frac{\partial F_{c_\varepsilon,\beta}(x)} {\partial\varepsilon}\right|_{\varepsilon=0} =\mathbb E_{q_{c_0}}[h(x,Y)]. } \]

And the first-order change in its gradient is

\[ \boxed{ \delta\nabla F(x) = \mathbb E[\nabla_xh(x,Y)] -\beta\operatorname{Cov} \left(\nabla_xc_0(x,Y),h(x,Y)\right). } \]

This formula is especially useful.

It tells us exactly how a small modification of the cost changes the optimization geometry.

We could select \(h\) to maximize improvement in the directional condition per unit cost perturbation.

For example, define

\[ G_c(x)= \langle\nabla F_{c,\beta}(x),x-x^*\rangle. \]

We could solve the local problem

\[ \boxed{ \begin{aligned} \min_h\quad&\|h\|^2\\ \text{s.t.}\quad& \delta G_c(x)\ge \mu\|x-x^*\|^2-G_c(x) \end{aligned} } \]

for points where the constraint is violated.

Because \(\delta G_c\) is linear in \(h\), this is a convex quadratic problem for an appropriate Hilbert-space norm.

It provides a principled way to learn the cost rather than select one from a fixed family.

There are still constraints to enforce, such as \(h(x,x)=0\) and \(c_0+\varepsilon h\ge0\), and the first-order approximation must be checked against the full nonlinear envelope.

## 7. One subtlety: soft-envelope normalization

A cost \(c(x,y)\) also determines the mass of its kernel:

\[ A_c(x)=\int e^{-\beta c(x,y)}\,dy. \]

For translation-invariant costs on \(\mathbb R^d\), this is constant in \(x\).

For position-dependent costs, it generally is not.

Consequently, even with \(f\equiv0\), the unnormalized soft envelope may have a nonzero gradient.

To avoid artificial drift, we can instead use the normalized kernel

\[ K_c(y\mid x) =\frac{e^{-\beta c(x,y)}}{A_c(x)} \]

and define

\[ \widetilde F_{c,\beta}(x) =-\frac1\beta\log \mathbb E_{Y\sim K_c(\cdot\mid x)} [e^{-\beta f(Y)}]. \]

This is a different envelope, but it cleanly separates smoothing of \(f\) from changes in kernel volume.

My preferred direction would be to start with radial costs \(h(r)\), derive their influence function \(w(r)=h'(r)/r\), and optimize that function directly. This reduces the search over arbitrary two-point costs to a one-dimensional functional optimization problem.

Then, if radial costs are insufficient, extend the same construction to anisotropic or Bregman costs.

The main question is whether the cost should be designed to achieve pseudoconvexity, the stronger inward-gradient condition, or a quantitative convergence rate. These are different constraints and can lead to substantially different optimal kernels.

# Multistart methods for global optimization

Multistart optimization is a family of global optimization methods that repeatedly run a local optimizer from different initial points, attempting to discover multiple basins of attraction and ultimately the global minimum.

The central idea is to turn global optimization into two subproblems:

1. Exploration: choose promising starting points across the search domain.
    
2. Exploitation: use a local optimizer to converge from each starting point to a nearby local minimum.
    

Consider

\[ \min_{x\in\Omega} f(x). \]

Let \(L(x)\) denote the result of running a local optimizer initialized at \(x\). A multistart method generates \(x_1,\ldots,x_N\) and returns

\[ \boxed{x_{\mathrm{best}}=\arg\min_{i=1,\ldots,N}f(L(x_i)).} \]

## 1. Why multistart works

Every local minimum \(x_j^*\) has a basin of attraction

\[ B_j=\{x\in\Omega:L(x)=x_j^*\}. \]

If starting points are sampled independently from a distribution \(q\), the probability of reaching minimum \(j\) in one run is

\[ p_j=\int_{B_j}q(x)\,dx. \]

For the global minimum, the probability of discovering it after \(N\) starts is

\[ \boxed{P_N=1-(1-p_*)^N.} \]

For example, if its basin occupies 1% of the sampling distribution:

![Chart](data:image/png;base64,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u3bt0q9iHgEEEEAAAQQQQAAB/xDw41IQoPpx41C0nBHYHxOn6XNW6daBn6hOl5Hq/drX+uGXf2RP5rUjREaEqUn9ynr1oau0+L1uGvbgVWp8YSVFJC+39emHk046SYMGDdLMmTP17LPP6v7779cTTzyhsWPHavLkybr00kvT78I8AggggAACCCCAAAIIZEIgGALUTFSTTUJNIC4+QbN+XKN7XpjpgtL7XpqtL39ap/iEREcRHuZRw5oVNPjeK5KD0lv11uPN1eKSKoqOSnlSr9voOIldTW3durW6d++uzp07q06dOsfZg9UIIIAAAggggAACCCBwLAEC1GPpsE5S4CAkJCZp7tINenjYV7rgppG647nP9PF3f+jAwXhXCbtLt071cup/x6VaMKqrxg5oqfaNz1bRQlFuPQkCCCCAAAIIIIAAAgjkrwABav76c/RsCiQlSYtWbFK/N+eo3i3v6qZ+M/Thl79rz/5YX85nVz5Jj3a5SPPevlmTB7XVzdfUUOniBX3r83WCgyOAAAIIIIAAAggggIBPgADVR8FEIAksX/ufnh/9gxrePkZtH52s0Z/8om27DviqUKl8Cd13w4X6+vVO+uzlG3RX2wtU/qQi4l9oCVBbBBBAAAEEEEAAgcASIEANrPYK6dKu+3eXXpn4k668e5ya3z9Rb0xZrH+27vGZnFK6iG5vVUsfDWmvb5ID014d66lycqDq24AJBBDISQHyQgABBBBAAAEEclyAADXHSckwJwU2b9+nt6YtVYsHPtBld47V0HEL9OffO3yHKFUsWp2bnacPnmut79+5WU90baAaZ5bxrWcCAQQQCEwBSo0AAggggEBoChCghma7+3Wtd+6J0djPflX73lP1v26j9My78/TLn1t9ZS5SKEptr6iu0f1a6KfR3TTwzkaqd0552UOQfBsxgQACCCCAwNEEWI4AAggg4LcCBKh+2zShVbB9B+I0+euVurn/R6p787vq88a3WrB8o+whSCYRHRWh5hefqTcea6bFY7rppfuuVKMLTpd1F2PrGRBAAAEEEEDAPwQoBQIIIJAdAQLU7Oixb7YEDsYm6LPv/9Rdgz7XBV3e0YMvf6FvF//l66s0IjxMl9epqKG9GmvRe930f49erasvOkNRkeHZOi47I4AAAggggAACASpAsREIegEC1KBvYv+qYHxCor5ZtF4PDP3C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+77lXWdjPrMO65iHd8DtsAvn3pT3Wk5+f/N4PO67tL3e0r/WDstLYaln/GHauo2x35DaJWC7RJ26TPbUXps/7bTTbJThYMGn3d7bvHlzt96unlofqPaX/ho1asgC1ypVqmjFihXy5uc2zOXEPlCsyxy7BflY5c/lYpA9AggggAACIS5A9RFAAAEE/FnA7wJUu03Vbh+wp/mmDyDtvnf7a4b3ymh6WNvHrk62bNnSXUm19XYrr/1uxfbz/rXMHq5kPyJP/5sR254BAQQQQAABBBBA4AQF2A0BBBDIpoDfBah2Gdl+S2o/xP/ll1981bPgc926dbKuZipXruxb7p2whyvZ7z3r1q3rrpJ6l9vvPuzqaeorshb4ZtSNjXcfxggggAACCCCAAAII+JsA5UEgFAT8LkA19HPPPVd16tTRnDlztHbtWvdAo08++cR1P9O6dWt377Jtl3rw3sZrD05Kvdzj8ah+/fratWuX7OmtdhV25cqVsh+B29Xa1NsyjQACCCCAAAIIIIAAAiEpQKX9RMAvA1SPx6POnTvLupEZOnSoe/quPdK9d+/eOuOMM46gs6DTHoRkT/D1eDxHrK9Vq5batGmj0aNH6+mnn5Y9NcqerGtXa4/YmAUIIIAAAggggAACCCCAAAI5KJD5rMIyv2nebmm/F73uuutkT+i1LmR69uypMmXKZFiI8uXLq0ePHipcuHCG6z0ejy655BINGTLEdUdz9913H3XbDDNgIQIIIIAAAggggAACCCCAQK4L+G2Amus1z8YB2BUBBBBAAAEEEEAAAQQQQCDnBQhQc96UHLMnwN4IIIAAAggggAACCCAQogIEqCHa8FQ7VAWoNwIIIIAAAggggAAC/itAgOq/bUPJEEAg0AQoLwIIIIAAAggggEC2BAhQs8XHzggggAACeSXAcRBAAAEEEEAg+AUIUPOojePj4/Xjjz+6bm5Wr16dR0flMDklkJSUpPXr18u6PZo+ffpRsz148KCmTZsme+q0PS3ankK9Y8eOo27PivwRsPb8/vvv9dhjj6l79+665557NGrUKO3fvz/DAq1bt851d2XbeocBAwZo3759GW7PwvwTyGxbWddl9v686667XNva+9rO0/lX8nw/sl8UwN6bmTnXZrb9EhMTNWfOHD3wwAOytn7mmWf0999/+0VdQ6EQMTExev/9991nop07H374Ydce1i7p62/LMttW9hqxbgPvvPNOWZ5z586VvXbS58l87gjYe8jrb9913nrrLe3ZsyfDg2W2rfj+lCFfni60Nhg8eLD7XmTvVxvuvfderVixIk05bLvMfNfNyns6zQGSZwhQkxFy+/9vv/3mTtCTJk2Sfajm9vHIP2cFtm3bpsmTJ+u9995zb1J7w2V0BPtya33tLliwwAU+L730kqKiolxQ+99//2W0C8vySeCHH37QhAkTXNdV5557rsLCwmTLLGDJKOj86aefZP0mFy9eXN7h4osvpruqfGq/Yx02M221efNm974sUKCAXn75ZT344IOy9+2YMWOUkJBwrOxZl4sCmT3XZrb9LGCZMWOGpk6dqttvv12vvvqqqlev7rqcW7NmTS7WJKOsQ2+ZfSbae2rRokXOvWLFitq7d6/GjRsnaxdrH6+KTduyzLTVqlWr3PvWzt3/93//p1tuucW18ccff+zNjnEuCvz55596++23Vbt2bbVr104nnXSSFi5cKPsjb/rzZ2bbyl4rfH/KxUbLZNZr166VnYe933NsbOfMypUr+3LIbFtl5T3tyzzVBAFqKozcmrST6M0336wLLrggtw5BvrkoULp0aXcSbt26tSIjI496pO+++07Lli1z21rfvNHR0WrVqpUOHDgg++C0N+tRd2ZFngnYFe358+friSeecIGJXe22qyqVKlWSnZytHVMXZtOmTdq4caP69+8v+8uid2jcuHHqzZj2A4HMtJV9uE6cOFEej0ft27eXBamnnXaamjZtqsWLF+vnn3/2g5qEZhEyc67NSvvZH4e//PJLNWnSRGeffbasf/Wrr75aJUqUcAGNXQUITem8qbW9l+wPe3bOtL7qH3/8cXcltWDBgu4qql2F85Yks221e/dujR8/XqeccoquueYa98dF+45Vv359ff311+5OJ2+ejHNewN5/v//+u7tqfe2118o+Bx955BGdfvrp7s6EXbt2+Q6albayz12+P/no8mXC/rjw7bffqmvXrmm+69gVcvuc9BYqs22V2fe0N9/0YwLU9CK5OG8n5VzMnqxzWcCuhtqVtowOY0Go3TJatGhRpf5LU5kyZXTqqafK3qj2V/+M9mVZ3gpYO9iHatmyZX0HtnazAMXad926db7lNmFBS5UqVbhaahh+PmSmreyPEH/88Yf7QlWyZElfjayN7Y9K9uFr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)

Assumes independent starting points and deterministic local convergence.

This gives a simple probabilistic convergence guarantee, provided the global optimum's basin has positive sampling probability and local optimization reliably converges within it.

However, it also exposes the central limitation: a very narrow global basin can be exceptionally difficult to discover, even when local optimization is nearly free.

## 2. Main variants

|Method|How it selects starts|Main advantage|
|---|---|---|
|Random multistart|Independent random points|Simple, parallelizable|
|Sobol / Latin hypercube multistart|Space-filling samples|Better initial coverage|
|MLSL|Samples points, filters redundant local searches using proximity and objective values|Avoids repeatedly exploring the same minima|
|APOSMM|Coordinates sampling and multiple local runs|Efficient parallel exploration|
|Basin-hopping|Perturbs an existing minimum and locally optimizes again|Can move between neighboring basins|
|Surrogate-guided multistart|Uses a learned approximation of the objective to suggest starts|Potentially reduces expensive evaluations|

Two particularly interesting methods are:

Multi-Level Single-Linkage (MLSL). Instead of blindly optimizing every sampled point, MLSL uses clustering-like rules to avoid redundant searches. Under suitable regularity assumptions, it can identify all local minima almost surely while initiating only finitely many local optimization runs. This is a stronger efficiency result than ordinary random multistart.

NLopt Documentation+1

APOSMM (Asynchronously Parallel Optimization Solver for Finding Multiple Minima). It evaluates candidate points and advances multiple local searches concurrently. It also uses information from previously evaluated points to decide which local searches are worthwhile.

DOI+1

## 3. Connection to basin mapping

There is a useful mathematical interpretation.

Define the locally minimized objective:

\[ \boxed{F(x)=f(L(x)).} \]

If the local optimizer converges to a fixed minimum throughout each basin, \(F\) is constant on the interior of that basin.

Conceptual illustration: local optimization maps every starting point to the minimum of its basin. The transformed landscape is piecewise constant.

Multistart can therefore be viewed as searching over the discrete set of local minima, using continuous points to discover them.

This suggests more advanced approaches: estimate basin membership from observed optimization trajectories, learn the boundaries between basins, or construct proposals that preferentially enter previously unexplored basins.

An important caveat is that Euclidean proximity alone does not determine basin membership: nearby points can converge to different minima, while distant points can reach the same one.

## 4. Comparison with other global methods

- Multistart: global exploration is mostly about locating new basins.
    
- Basin-hopping: explicitly explores transitions between local minima, often with a Metropolis acceptance rule.
    
    SciPy v1.18.0 Manual
    
- Parallel tempering: maintains samples at different temperatures and exchanges them to improve mixing.
    
- Bayesian optimization: models the objective to decide where evaluations are most valuable.
    
- Differential evolution / CMA-ES: adapt populations or search distributions rather than relying primarily on separate local optimizations.
    

The main advantage of multistart is that it can exploit an excellent local optimizer without requiring a sophisticated global model.

Its main weakness is sample complexity: in high dimensions, narrow basins can have exponentially small sampling probabilities.

## 5. Practical implementation

For a first experiment, I would compare three methods using the same local optimizer and total function-evaluation budget:

1. Sobol-sequence multistart.
    
2. MLSL with redundant-start filtering.
    
3. Basin-hopping with adaptive perturbation scale.
    

These are available through established implementations:

- [NLopt — MLSL and MLSL-LDS](https://nlopt.readthedocs.io/en/stable/NLopt_Algorithms/?utm_source=chatgpt.com#mlsl-multi-level-single-linkage)
    
- [libEnsemble — APOSMM](https://libensemble.readthedocs.io/en/main/examples/aposmm.html?utm_source=chatgpt.com)
    
- [SciPy — Basin-hopping](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html?utm_source=chatgpt.com)
    

The particularly interesting research direction is adaptive multistart based on learned basin structure: instead of independently sampling points, use accumulated optimization trajectories to estimate where undiscovered minima are most likely to exist, while preserving sufficient global exploration for convergence guarantees.