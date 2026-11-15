A vector of $N$ dimensions can represent at most $N$ independent features. Independence here means that inner product of basis vectors is 0.

But if we allow overlap of features by restricting inner product to be at most $\varepsilon$, then we can potentially represent exponentially more independent directions in the same space.

Note that representation is relative to what we actually observe. The domain of all possible values can be infinitely large, but if not all values are equally likely we can neglect representing rare values exactly. That is basically what the entropy of a distribution measures.

Fundamentally, whatever task we want to learn can be represented by a function that maps inputs to desired outputs.

[fgd ar](https://arxiv.org/abs/2606.16926?utm_source=chatgpt.com)
We can represent a function via a complete orthonormal functional basis $\phi_i$ as a sum:
$$f(x; \theta)=\sum_{i=0}^{N} \theta_i \phi_i(x)$$
This representation is particularly nice as it allows us to add functional derivatives represented in the same basis and applied linearly. Suppose objective loss creates a functional gradient of f:
$$L(f)=\frac 1 2 \|f-g\|^2 \\
\nabla L(f)=f-g=r\\
f \leftarrow f-\eta \nabla L(f) \iff \theta \leftarrow \theta-\eta \Delta \theta $$

With functional basis representation the update is trivial, as $r=\sum_{i=0}^{N} \Delta\theta_i \phi_i(x)$ and step in functional space coincides with step in parameter space exactly.

We can quantify effectiveness of representation through residual. After obtaining the approximate functional gradient $r_{\theta}\approx r$ with truncation to the represented basis, we compute unrepresented residual error $\|r_{\theta}-r\|$. This residual is orthogonal to every represented basis term and this measures unexplained fraction of actual gradient. Once it starts to dominate we must extend the representation, which yields this condition:
$$ \boxed{ \epsilon= \frac{\|r_{\theta}-r\|}{\|r_{\theta}\|} <\frac12 } $$
If the condition holds, the direction is accepted and a sufficiently small learning rate guarantees descent in the functional-space formulation. The permitted learning-rate interval is
$$ 0<\eta< \frac{1-2\epsilon}{1+2\epsilon}$$

So FGD provides a quantitative test of whether the current local representation can follow the desired functional gradient sufficiently well. 

For small updates we can approximate $\nabla L(f)$ with Jacobian:
$$f(x;\theta+\Delta\theta) \approx f(x;\theta)+J_\theta\Delta\theta\\ J_\theta=\frac{\partial f}{\partial\theta}$$

But the term $J_\theta\Delta\theta=g$ generally does not identify a unique parameter update, when we have overcomplete basis.

For example, if $J_\theta$ has a nontrivial nullspace,
$$ J_\theta v=0$$
then
$$ J_\theta(\Delta\theta+v) =J_\theta\Delta\theta$$
There are infinitely many parameter updates with the same first-order functional effect.

[CAGE-NAS](https://arxiv.org/abs/2610.01173?utm_source=chatgpt.com) approximates the desired functional direction by solving a regularized least-squares problem:
$$ \boxed{ \Delta\theta_\lambda = \arg\min_{\Delta\theta} \left[ \|J_\theta\Delta\theta-g\|^2 +\lambda\|\Delta\theta\|^2 \right] } $$

giving

$$\Delta\theta_\lambda = (J_\theta^\top J_\theta+\lambda I)^{-1} J_\theta^\top g$$

The corresponding functional direction is
$$ \nabla L(f)=J_\theta\Delta\theta_\lambda$$
If no tested damping value produces an acceptable direction, the algorithm attempts to grow the representation.


3 dimentions of representation:
1. width - how many terms in a basis
2. depth - how many composition components
3. latent dimension - dimensionality of intermediate components.

We can extend/shrink each dimension with function preserving changes under this representation:
$$f=g_k \quad g_{i+1}(x)=g_i(P_ix+h_i(x)) \quad g_1(x)=h_0(x)\\
g_i:X_{i-1}\to X_{i} \quad f: X\to Y \quad X_0=X \quad X_{k}=Y\\
h_k(x)=\sum_i a_i\phi_i(x) \quad \phi_i(x): X_{i-1}\to\mathbb R \quad a_i\in X_i\\
$$
Where $X_i=\mathbb R^{n_i}$ are intermediate domains, $P_i$ simply drops coordinates (projects) or pads them with 0 (embeds) to match the required domain, $h_k$ are the residual transforms, $g_i$ are the composition components, $\phi_i$ are the functional basis terms and $a_i$ are their corresponding coefficients/values.

 3. Exact interaction expansion using parameter multi-indices

Collect all residual coefficients into one vector

\[ b=(b_{11},b_{21},\ldots,b_{1L},b_{2L},\ldots). \]

At \(b=0\), every residual layer is the identity:

\[ f(x;0)=g(x). \]

Assuming analyticity in the parameters near zero, we have

\[ \boxed{ f(x;b)= \sum_{\alpha\in\mathbb N^N} \frac{b^\alpha}{\alpha!} \left. \partial_b^\alpha f(x;b) \right|_{b=0}. } \]

This is the most general interaction expansion.

Define

\[ \boxed{ \Phi_\alpha(x)= \left. \partial_b^\alpha f(x;b) \right|_{b=0}. } \]

Then

\[ \boxed{ f(x;b)= \sum_\alpha \underbrace{\frac{b^\alpha}{\alpha!}}_{c_\alpha} \underbrace{\Phi_\alpha(x)}_{\text{effective basis function}}. } \]

Because \(g\) is itself expanded in \(\phi_{i0}\), we can separate its coefficients:

\[ \boxed{ f(x;b) = \sum_i a_i \sum_\alpha \frac{b^\alpha}{\alpha!} \Phi_{i,\alpha}(x), } \]

where

\[ \boxed{ \Phi_{i,\alpha}(x)= \left. \partial_b^\alpha \left[ \phi_{i0}\circ h_L\circ\cdots\circ h_1 \right](x) \right|_{b=0}. } \]

Therefore, the effective coefficient of each indexed interaction is

\[ \boxed{ c_{i,\alpha} = a_i\frac{b^\alpha}{\alpha!}. } \]

This formula works at arbitrary depth and arbitrary interaction order.

**The depth-dependent complexity is entirely contained in the ==effective basis== functions \(\Phi_{i,\alpha}\).**

With this representation we can conservatively mutate the representation without affecting the resulting function:
1. Add or remove a term with 0 value
2. Add or remove a composition component with 0 residual
3. Add or remove a dimension for internal components with independent variables manipulated through embedding-projection pairs, merged into adjacent components.



Whats important is that new degrees of freedom have a gradient flowing through them even when they generally preserve the function. This creates complications for dimension expansion, because naive embed-project pair will keep this new variable hidden from the gradient, as it did not yet contribute anything to the computation. For that reason we treat new embeded state as a new basis term, that we are not yet folding into expansion.

For an existing residual transform $g(x)=x+h(x)$ ordinary basis expansion introduces $g'(x)=x+h(x)+a\phi(x)$. We instead interpret it as introducing a latent domain $W$ and separates the new term into two operations:
1. **Expansion:**
$$ \boxed{ e_\phi(x)=(h(x),\phi(x)) } $$
2. **Projection:**
$$ \boxed{ p_a(x,w)=x+aw } $$
Then
$$ \boxed{ p_a\circ e_\phi(x)=x+aL(\phi(x))=g' } $$
This exactly reproduces ordinary basis expansion, while actually introducing another internal dimension.

for $f=g_1\circ g_2$ with $f: X\to Z$, $g_1: X\to Y$, $g_2: Y\to Z$ we expand intermediate $Y=\mathbb R^n$ into $(Y=\mathbb R^n,\mathbb R)\cong \mathbb W=R^{n+1}$:
$$f=g_1\circ g_2=g_1\circ p\circ e\circ g_2=(g_1\circ p)\circ (e\circ g_2)=g_1'\circ g_2'$$
With $g_1':X\to W$, $g_2': W\to Z$ now.

Note that projection here is not simple discarding of a dimension, but a linear combination. To represent within the proposed framework we must split these further into computation and projection, and similarly into embedding and computation as we can basically treat now the new dimension as truly independent before/after the transformation. But either way we must merge these with adjacent components to preserve component count. we can, right?


We expand terms to support approximation rate.
We expand composition to new directions along the gradients of basis functions.
We expand dimension to reuse values.
Try to match basis expansion to the first order.

We can treat choice of a particular expansion method as RL action, with reward being (future) improvement. While jacobian technically already shows us that effective degree of freedom only increases as much as we add new parameters, but some choices may yield faster converges as they correlate efficiently?

decomposition can be understood as sharing a restricted domain.

Weierstrass factorization theorem

- **Functional analysis:** classifying spaces of functions.
- **Representation theory:** finding coordinates in which transformations become simple.
- **Differential algebra:** characterizing functions through equations they satisfy.
- **Algebraic geometry:** decomposing families of functions into structured components.
- **Approximation theory:** determining which representation is most efficient for a given function.

While complete functional basis is useful for finding a representation of the target solution itself, this solution may be suboptimal for many reasons, like storage and computational efficiency cost, interpolation and extrapolation characteristics, overall generalization to unseen datapoints.

This creates justification for overcomplete basis, and general computational graphs. They are redundant for function representation, yet crucial for further optimization.

Platonic representation hypothesis
https://youtu.be/u8cxqSPyT8s?si=L-kpF9A7-rJJiHhY

[chirplets](https://chatgpt.com/c/6aef7db9-a130-83eb-99a8-bbe367ea78da)