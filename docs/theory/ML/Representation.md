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