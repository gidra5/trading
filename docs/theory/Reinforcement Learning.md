
https://www.youtube.com/watch?v=NFo9v_yKQXA
The dynamics of the environment are described by a (transition) distribution function $p(s',r\ |\ s,a)$ - a probability of transitioning to $s'$ with a reward $r$ given current state $s$ and an action $a$. In our setup it represents how much return and what portfolio we will get, given our current portfolio and a chosen portfolio change.

The distribution acts on the 3 sets - set of states $\mathcal S$, set of rewards $\mathcal R$, and set of actions in any given state $\mathcal A(s)$.

Note that in that setup the sets are not finite, but we can discretize them to get a finite approximation.

The oracle has a policy $\pi(a\ |\ s)$ - a distribution function for an action $a$ given state $s$.

We call $g_t$ a *score* of the policy, defined as a weighted sum of all future rewards:
$$g_t=\sum_{t'\ge t}w_t*r_t$$
The weight essentially describes how much the *agent* values the reward, while reward's values itself is some objective measure of the reward.

Not all weights are equally nice. A nice weighting is factorizable over all prev time:
$$w_t=\prod_{t\ge t'}w_{t'}'$$
That way we can express score recursively as:
$$g_t=r_t+w_{t+1}' g_{t+1}$$

A simple, nice weighting is exponential, scaling each future score by $\gamma$:
$$g_t=r_t+\gamma g_{t+1}$$
That values immediate reward more than the later one.

We define an *action value function* $q$:
$$q(s, a)=\mathbb{E}[G_t\ |\ s, a]=\sum_{s'\in\mathcal S, r\in\mathcal R} p(s',r\ |\ s, a)[r+w*v(s')]$$
Then *state value function* $v$:
$$v(s)=\mathbb{E}[G_t\ |\ s]=\sum_{a\in\mathcal A(s)} \pi(a\ |\ s)q(s,a)$$
And *policy score* as simply:
$$\mathbb{E}[G_t]$$
The usual goal of the agent is to maximize it:
$$\max_{\pi}\mathbb{E}[G_t]$$
That means, we seek such a policy $\pi$, that will generate highest *expected score*. While it is a statistical goal, it is still reasonable formulation for deterministic goals.

For optimal policy we will also have optimal value functions:
$$ {v_o(s)=\max_{\pi} v(s)} \quad {q_o(s,a)=\max_{\pi} q(s,a)}$$
Another properly of optimal policy is that is is greedy with respect to its action value function:
$$\pi_o(a\ |\ s)=\arg \max_a q(s,a)$$

Note that although we define dynamics and policy probabilistically, they can be deterministic, when described by delta distributions.

By substituting action value and state value function into each other, we can get recursive definitions for each function, which are called **Bellman equations**:
$$
\begin{aligned}
q(s, a)&=\sum_{s'\in\mathcal S, r\in\mathcal R} p(s',r\ |\ s, a)[r+ w(s')*\sum_{a\in\mathcal A(s)} \pi(a\ |\ s')q(s',a)]\\
&=\mathbb{E}[R_t\ |\ s,a]+\sum_{s'\in\mathcal S, r\in\mathcal R} \sum_{a\in\mathcal A(s')} w(s')*p(s',r\ |\ s, a)\pi(a\ |\ s')q(s',a)
\end{aligned}$$
$$v(s)=\sum_{a\in\mathcal A(s)} \sum_{s'\in\mathcal S, r\in\mathcal R} \pi(a\ |\ s)p(s',r\ |\ s, a)[r+w(s')*v(s')]$$
# Policy Evaluation

Given some policy it is useful to be able to compute both value functions.

If we dont have access to $p$, then we cannot directly compute the functions, but importantly if we have action value function, we can easily compute state value action, since usually the policy is known exactly, or can be evaluated at any point.

Another problem is that we might not be able to evaluate next state values, if they are non terminating and acyclical. That is impossible for finite sets, but even if they are finite, they can be extremely large, so we cannot reach terminal state.

To get around these problems we can simply approximate the value function and then adjust it according to Bellman equations until we reach fixed point.

This procedure applies to both action value and state value functions.