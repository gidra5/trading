In other documents we considered the RL formulation for the trading problem.

The state formulation was deliberately simplified, although still complete. For actual optimal strategy it might be worth adding more abstractions that create actual positions and manage them explicitly, instead of it being implicit in the policy.

As a brief recap - the base model looks as follows:
1. The state is a tuple of price, portfolio/leveraged equity allocation across assets and resting limit orders. portfolio+orders is account.
2. Actions are describing target state of the portfolio and orders. Although for implementation it might still be preferable to use deltas for actions instead (open/cancel limit order, execute market order).

Now we replace the portfolio-orders tuple we control with an equivalent formulation in terms of *positions* that have a *lifecycle*, similar to orders. The equivalence will be shown as mapping from one to the other.

The state structure for account be as follows:
1. store available quote and asset amounts. That basically matches regular portfolio state.
2. store list of positions instead of resting orders. These are basically tuple of state (entering, exiting), resting order for entering and position size for exiting.
3. each position roughly corresponds to entering at some price and exiting at another. The lifecycle is like this:
	1. Create order to enter at a given price
	2. Once entered exit with market order at chosen time.
	3. We might soften entry/exit execution to happen gradually through a grid of orders.
4. The positions cannot be modified outside lifecycle: we cannot add more to the position size
5. The action set changes to open position with given size and optional price, and close a specific position within limits of the portfolio.

This structure roughly decomposes sequence on individual market/limit orders into separate episodes with fixed sequence of actions that happen for each. Basically the only thing it does it pair each order with opposing side order of the same size.

The positions can be allowed to hedge each other - as long as there is enough corresponding asset being available for this, we can create/close any positions. But that means opening/closing positions cannot be independent of other positions. Thats not an issue for the optimal policy, but it is for the simplicity of actual model.

We can trivially collapse it to base formulation - each entering position's orders are mapped back to resting order set and thats it.

This formulation also can be scaled to multiple assets - every asset now has its own list of positions. Or alternatively we label each position with whichever asset it corresponds to.

To motivate this structure consider how optimal policy for exiting long position would look like:
1. Assume the agent is penalized with friction on every transaction and holding step with $f$ and $c_h$
2. Furthermore single transaction is bound by min and max size $s_{min}$ and $s_{max}$.
3. Then assuming the next price is fixed $p_{t+1}$, we can split it into a few regions, relative to current price $p_{t}$:
	1. $p_{t+1}>p_{t}+a_{t,h}$ - is price rises enough to compensate for holding friction, we trivially hold
	2. $p_{t+1}\in[p_{t}+a_{t,h}, p_{t}-a_{t,min}]$ - a small region just below current price, where we cannot reliably decide best action using just the next price. If later it rises enough, its better to hold, if it falls its better to sell. The threshold $a_{min}$ is determined by fees and min order.
	3. $p_{t+1}\in[p_{t}-a_{t,min}, p_{t}-a_{t,max}]$ - an intermediate region where we need to balance risk and return.
	4. $p_{t}-a_{t,max}>p_{t+1}$ - we sell as much as we can. $a_{max}$ is determined by max order size.
4. within the undetermined region $p_{t+1}\in[p_{t}+a_{t,h}, p_{t}-a_{t,min}]$:
	1. second move gets us above the band, then if we hold we are sure to get better position at the cost of the intermediate drawdown of size $\Delta p_t$. Selling and buying at $p_{t+1}$ is worse if $a_{t,h}<a_t$, since we pay roundtrip cost at the same price movement. The condition $a_{t,h}<a_t$ is guaranteed to be true by construction, since we cannot actually execute the plausible $c$. That places break-even $p_{t+2}$ strictly above holding's break even $p_t$. Thus we always hold.
	2. In case it is smaller, then neither is better, we need to look at the next price move. Next move is same direction, and if in total it moves us below $p_t-a_{t,min}>p_{t+3}$, then now its better to sell.
	3. Thus we can conclude that staying in this range necessarily requires looking further into the future.
	4. As we look deeper, the cost of holding, as well as partial close increases linearly with $h$ due to continuous maintenance.
	5. By induction it is preferrable to hold when $p_{t+h}>p_t+a_{t:t+h,h}$ and $p_{\tau}>p_t-a_{t:t+\tau,min}$ for any $\tau \in [t,t+h]$. Similarly for selling - it is preferrable when $p_{t+h}>p_t-c_{t:t+h,min}$ and $p_t+a_{t:t+\tau,h}>p_{\tau}$ for any $\tau \in [t,t+h]$.
	6. Note that $a_{t:t+h,min}$ increases with $h$ a bit slower than the pure holding cost $a_{t:t+h,h}$ itself, because it allows for holding only a fraction of the position.
	7. As $h \to \infty$, if we are still within the range, then the asset can be considered frozen, as there are no useful opportunities and the price moves within the margin of error. In which case it is better to invest in another, more volatile asset. And that means selling all and never touching it again.

Note that the overall PnL is the sum of each position's PnL, so to maximize it its enough to consider optimal policy only for one position. Moreover, an optimal policy will have every position closed with best possible RoI. And if the market transition is considered deterministic (only the realized price is considered as possible) then positions from optimal policy will be disjoint - it doesnt need to open multiple simultaneous positions, because these delayed positions can be merged into the earliest one and trivially improve without any downside.

Another property of the optimal policy is that every position has to have 0 drawdown. Otherwise it could've opened later to both improve returns and reduce drawdown.

# Policy
For policy we consider optimizing expected wealth utility parametrized with a relative risk-aversion factor. Use normalized CRRA utility (a special case of [HARA](https://en.wikipedia.org/wiki/Hyperbolic_absolute_risk_aversion)), with expected log wealth as the limiting $\gamma=1$ case, since it is equivalent to optimizing for returns:

$$
U_\gamma(E)=\dfrac{E^{1-\gamma}-1}{1-\gamma}
\qquad \gamma>0, E>0.
$$

## Wealth multiplier from an allocation

Let current portfolio wealth be $W_t$, and let $w\in[0,1]$ be the fraction allocated to the asset. The remaining fraction $1-w$ stays in quote currency. Define the cumulative net log return of holding the asset over period $h$ as

$$
 R_{t,h} = \log\frac{p_{t+h}}{p_t} - H_h = r_{t,h} - h*a_h 
$$
At time $t$, the asset allocation is worth $wW_t$. At price $p_t$, it buys

$$
a_t=\frac{wW_t}{p_t}
$$

asset units. The quote allocation is

$$
W_t^{quote}=(1-w)W_t
$$

At time $t+1$, the asset units are worth

$$
\begin{aligned}
W_{t+1}^{asset}
&=a_tp_{t+1}(1+r_a)\\
&=\frac{wW_t}{p_t}p_{t+1}(1+r_a)\\
&=wW_t\frac{p_{t+1}}{p_t}(1+r_a)\\
&=wW_te^{r_{t+1}-ln(\frac{1}{1+r_a})}\\
&=wW_te^{r_{t+1}-H_1}\\
&=wW_te^{R_{t,1}}
\end{aligned}
$$

If quote has zero return over the period, its value remains

$$
W_{t+1}^{quote}=(1-w)W_t
$$

Total terminal wealth is therefore

$$
W_{t+1}
=W_t\left[(1-w)+we^{R_{t,1}}\right]
$$

The gross portfolio wealth multiplier is terminal wealth relative to current wealth:

$$
\boxed{
M_{t+1}
=\frac{W_{t+1}}{W_t}
=(1-w)+we^{R_{t,1}}
}
$$

Since

$$
(1-w)+we^{R_{t,1}}
=1+w(e^{R_{t,1}}-1)
$$

we can also write

$$
\boxed{
M_{t+1}=1+w\rho_{t+1},
\qquad
\rho_{t+1}=e^{R_{t,1}}-1
}
$$

where $\rho_{t+1}$ is the asset's simple return. The portfolio's simple return is consequently

$$
\frac{W_{t+1}-W_t}{W_t}
=M_{t+1}-1
=w\rho_{t+1}
$$

If the portfolio starts fully allocated to the asset and closes fraction $s$, then $w=1-s$ and, before costs,

$$
M_{t+1}=s+(1-s)e^{R_{t,1}}.
$$

The friction-adjusted model below replaces the quote multiplier $1$ by the close-and-re-enter multiplier $K_h$.

Let $K_h$ be the terminal wealth factor for closing into quote and later re-entering. If quote earns no return and both conversions charge proportional fee $f$,

$$ K_h=(1-f)^2 $$

Starting with unit wealth, the terminal wealth factor after retaining fraction $w$ is

$$ M_h(w,R_{t,h}) = we^{R_{t,h}}+(1-w)K_h. $$

## Optimal fraction

Let the log return $r_{t+1}$ have conditional distribution $P(r_{t+1}\mid\mathcal F_t)$. The retained fraction is chosen before its realization:

$$
w_t^*
=\arg\max_{w\in[0,1]}
\mathbb E\left[
U_\gamma\left(M_h(w,R_{t,h})\right)
\mid\mathcal F_t
\right].
$$

The close fraction is

$$
s_t^*=1-w_t^*.
$$

Since

$$
\frac{\partial M_h}{\partial w}=e^{R_{t,h}}-K_h,
\qquad
U_\gamma'(M)=M^{-\gamma},
$$

an interior optimum $0<w_t^*<1$ requires these two conditions:
$$U'(s^*)=0,\qquad U''(s^*)<0$$
Which means
$$
\boxed{
\mathbb E\left[
(e^{R_{t,h}}-K_h)
M_h(w_t^*,R_{t,h})^{-\gamma}
\mid\mathcal F_t
\right]=0.
}
$$
From that also follows that $U'(s)$ is non-increasing:
$$ \boxed{J'(s_1)\ge J'(s_2), \qquad s_1<s_2} $$
This ordering eliminates the apparently missing quadrant.
For expected log wealth, $\gamma=1$, this becomes

$$
\boxed{
\mathbb E\left[
\frac{e^{R_{t,h}}-K_h}
{w_t^*e^{R_{t,h}}+(1-w_t^*)K_h}
\mathrel{\Big|}\mathcal F_t
\right]=0.
}
$$

For $\gamma>0$,

$$
\frac{\partial^2}{\partial w^2}
\mathbb E[U_\gamma(M_h)]
=-\gamma\,
\mathbb E\left[
(e^{R_{t,h}}-K_h)^2M_h^{-\gamma-1}
\right]
\le0.
$$

Therefore the objective is concave in $w$ and is strictly concave whenever the two branches have different payoffs with positive probability. The first-order equation then has at most one interior solution. If it has no solution in $(0,1)$, the optimum is one of the endpoints.

This fraction comes from uncertainty combined with concave utility. If $Y_h$ is known deterministically, the utility is a monotone function of an affine wealth multiplier and the optimum is bang-bang: retain everything when $e^{Y_h}>K_h$ and close everything when $e^{Y_h}<K_h$.

## The three policy regions

Let

$$
J_t(w)=
\mathbb E\left[
U_\gamma(M_h(w,R_{t,h}))
\mid\mathcal F_t
\right].
$$

Because $J_t$ is concave, the signs of its marginal value at $w=0$ and $w=1$ completely determine whether the solution is full close, fractional or full hold.


### Close everything

At $w=0$,

$$
J_t'(0)
=K_h^{-\gamma}
\left(
\mathbb E[e^{R_{t,h}}\mid\mathcal F_t]-K_h
\right).
$$

If

$$
\boxed{
\mathbb E[e^{R_{t,h}}\mid\mathcal F_t]\le K_h,
}
$$

then adding even a marginal amount of asset exposure does not improve utility. Concavity implies

$$
w_t^*=0,
\qquad
s_t^*=1.
$$

### Hold everything

At $w=1$,

$$
J_t'(1)
=\mathbb E\left[
(e^{R_{t,h}}-K_h)e^{-\gamma R_{t,h}}
\mid\mathcal F_t
\right].
$$

If

$$
\boxed{
\mathbb E\left[
(e^{R_{t,h}}-K_h)e^{-\gamma R_{t,h}}
\mid\mathcal F_t
\right]\ge0,
}
$$

then utility is still increasing at the largest admissible retained fraction. Hence

$$
w_t^*=1,
\qquad
s_t^*=0.
$$

For expected log wealth this condition simplifies to

$$
\boxed{
\mathbb E\left[
K_he^{-R_{t,h}}
\mid\mathcal F_t
\right]\le1.
}
$$

### Close a fraction

The unique optimum is interior when

$$
J_t'(0)>0
\qquad\text{and}\qquad
J_t'(1)<0.
$$

Equivalently,

$$
\boxed{
\mathbb E[e^{R_{t,h}}\mid\mathcal F_t]>K_h
}
$$

and

$$
\boxed{
\mathbb E\left[
(e^{R_{t,h}}-K_h)e^{-\gamma Y_h}
\mid\mathcal F_t
\right]<0.
}
$$

For expected log wealth, the second inequality becomes

$$
\boxed{
\mathbb E[K_he^{-R_{t,h}}\mid\mathcal F_t]>1.
}
$$

In this region the distribution contains enough upside that closing everything is suboptimal, but enough downside that holding everything is also suboptimal. Solve the first-order equation for $w_t^*$ and use $s_t^*=1-w_t^*$.

The three regions are regions of the **conditional return distribution**, not intervals between already-observed future prices. As the distribution converges to a delta at a deterministic return, the fractional region collapses to the single indifference boundary $e^{Y_h}=K_h$ and the policy converges to the deterministic hold-or-close rule, subject to continuity and tail conditions.

## etc

The position decomposition allows us to consider a restricted problem of optimally opening and then closing that single position, instead of managing whole portfolio with arbitrary possible action sequences.

Note that opening price primarily affects drawdown resulting from the position, while closing price affects its final return.

We can restrict ourselves according to properties of the optimal policy:
1. We close only when PnL/RoI>0
2. Positions are independent - we dont need to consider global availability for opening closing.
3. Positions dont strictly allocate capital, simply track designated capital optimization.
4. We need to maximize log wealth (aka RoI) and minimize drawdown per position.
5. Size of the position is irrelevant.

These are reasonable if we merely assume that next price distribution $P(p'_{t+1})$ converges to $\delta(p'_{t+1}-p_{t+1})$ eventually - we have enough data to predict next return exactly in principle. In that case there is no uncertainty in the next price and the policy becomes an oracle that can predict future.

Thus we consider lifecycle of a single position. Lets say we open at price $p_o$, time $t_o$ and close at $p_c$, time $t_c$. In the interval $[t_o,t_c]$ there are two prices $p_h$ at time $t_h$ and $p_l$ at $t_l$ for highest and lowest prices. For optimal policy these are equal to close and open prices. The difference corresponds to missed return and drawdown.

Assume we entered at lowest price. Then the task is simply to close optimally. Using restriction (1) we need to only consider the case when $p_c>p_o$. Thus, assume we are in such a state and considering to close at price $p_t$. For simplicity lets also assume that every next step is strictly opposite in direction. Another way to put it is that every step is accumulation of same direction changes.

Under no friction, for $p_{t+1}>p_t$ we trivially hold the position for the next tick and get more of the return. Otherwise we can close and reopen at the lower price. It can be written down as a more general "it gets better before it gets worse" rule.

With friction it gets more complicated. Now for any action to become actually beneficial the price must move past the friction costs. Lets consider two simpler actions for now - fully closing or not at all.

If we hold, that means the return will be bigger than if we sold and reopened later. Since holding doesnt cost anything, if price rises we trivially can benefit from it. Thus if $p_{t+1}>p_t$, we still hold.

In the opposite case holding should be more favorable then selling and buying later. Assume that roundtrip cost is $c$, then it is more beneficial to sell if price drops below $p_t-c$ . The band $[p_t, p_t-c]$ is the ambiguous case - if price is within it, then neither case is trivially benefitial. If we hold and price stays within this limit, then we are worse, but selling is also worse because it does not fall low enough so we can benefit.

For the inbetween case it seems to be better to partially close such that $c<-\Delta p_t$ and we can control that with order size.

Next addition that complicates the optimal policy, is that there is a minimal order size, which in turn creates a cap on how small the roundtrip cost we can make. We are back to square one, when price is within $[p_t, p_t-c_{min}]$.

For high frequency trading the last decision branch becomes increasingly important, since more and more deltas are within this range.

A reasonable assumption is that probability of price path $p_{t:t+h}$ fully staying within $[p_t,p_t-c_{min}]$ goes to 0 as $h \to \infty$. You could write it down as:
$$\prod_{t=T}^{H}p(p_{t}\in[p_{T},p_{T}-c_{min}])\to0$$
In this case we can safely ignore that case, which means the rest of the policy covers all possible price paths optimally.

## Global portfolio position decomposition

Treat the whole portfolio exposure to an asset as one signed global position $x_t$, measured in asset units immediately after trading. We want to represent its trajectory by positions $i$ with sign $\sigma_i\in\{-1,1\}$ and remaining nonnegative size $q_{i,t}$ such that

$$
x_t=\sum_i\sigma_iq_{i,t}.
$$

Each local position may be created once, but after creation it may only be held or reduced:

$$
q_{i,t+1}\le q_{i,t}.
$$

### Constructive discrete-time proof

Maintain a collection of currently active positions and process each change in the global position.

1. If the global policy increases exposure on the current side, create a new position for exactly the added amount. Do not add it to an existing position.
2. If the policy reduces the current absolute exposure from $|x_t|$ by $c_t\in[0,|x_t|]$, define the global close fraction

   $$
   s_t=\frac{c_t}{|x_t|}.
   $$

   Reduce every active position on that side pro rata:

   $$
   q_{i,t+1}=(1-s_t)q_{i,t}.
   $$

3. If exposure crosses zero, first use $s_t=1$ to close every position on the old side, then create one new position for the residual exposure on the opposite side.

The representation is preserved by induction. In the reduction case,

$$
\sum_i\sigma_iq_{i,t+1}
=(1-s_t)\sum_i\sigma_iq_{i,t}
=(1-s_t)x_t,
$$

which is exactly the reduced global position. The opening and sign-crossing cases preserve the equality by construction. Every local position has one creation event and thereafter a nonincreasing size, so it obeys the required lifecycle. Holds correspond to $s_t=0$, partial closes to $0<s_t<1$, and full closes to $s_t=1$.

This also gives a pathwise decomposition without using the pro-rata convention: every positive exposure increment creates a lot, and later negative increments are matched to existing lots using FIFO, LIFO or any other matching rule. Pro rata is special because it makes every active local position follow the same fractional policy.

### In what sense the positions follow the policy independently

Let $z_t$ contain the market information used by the close policy. Suppose its close amount for a position of size $q$ is

$$
C(q,z_t)=s(z_t)q,
\qquad 0\le s(z_t)\le1.
$$

This policy is positively homogeneous and additive in size. For any partition $q=\sum_iq_i$,

$$
C(q,z_t)
=s(z_t)\sum_iq_i
=\sum_iC(q_i,z_t).
$$

Therefore applying the policy once to the whole portfolio gives exactly the same aggregate trade as applying it separately to every active local position. Each position needs only its own remaining size and the shared exogenous state $z_t$; it does not need the sizes or lifecycle states of the other positions. Whenever the global policy adds exposure, a new constrained position is created instead of increasing an existing one. This proves the intended global-to-local equivalence.

The independence here is **policy and accounting separability**, not independence of the random returns: all positions still observe the same price path. It also assumes that feasibility is separable. Global collateral, liquidation and availability constraints may still couple which collection of positions can be opened.

### Separable feasibility

Let $y_i$ be the lifecycle state of position $i$ and let $a_i$ be its proposed local action. Denote the actions individually allowed for that position by $\mathcal A_i(y_i)$. Feasibility is separable when the jointly allowed action set is the Cartesian product

$$
\mathcal A(y_1,\ldots,y_n)
=\mathcal A_1(y_1)\times\cdots\times\mathcal A_n(y_n).
$$

This equality has two directions:

1. Every globally feasible action can be represented by individually feasible local actions.
2. More importantly, **every combination** of individually feasible local actions is jointly feasible.

The second direction is what permits positions to act without consulting one another. If each local position independently chooses $a_i\in\mathcal A_i(y_i)$, the combined action $(a_1,\ldots,a_n)$ is guaranteed to remain valid.

Suppose, in addition, that the global action value is additive,

$$
Q(y_1,\ldots,y_n,a_1,\ldots,a_n,z)
=\sum_iQ_i(y_i,a_i,z),
$$

where $z$ is shared exogenous market state. Then

$$
\begin{aligned}
\max_{(a_1,\ldots,a_n)\in\mathcal A}
\sum_iQ_i(y_i,a_i,z)
&=
\sum_i\max_{a_i\in\mathcal A_i(y_i)}Q_i(y_i,a_i,z).
\end{aligned}
$$

Thus the global optimum is obtained by independently optimizing every position. The positions need not have statistically independent returns: they may all observe the same future price. What matters is that, conditional on the shared market path, each local transition and payoff depends only on its own state and action, and that arbitrary local actions can be combined feasibly.

Separable feasibility is **not required for the bookkeeping decomposition**. Any already-feasible global exposure trajectory can still be split into lifecycle positions. It is required for the stronger statement that autonomous local policies can choose actions independently and that their aggregate will remain feasible and globally optimal.

#### Examples of separable feasibility

Closing long spot lots is separable when each lot has exclusive ownership of its assigned asset quantity, any amount down to zero can be sold, closing cannot increase account risk, and execution costs are additive. Each lot can then choose any close amount in

$$
0\le u_i\le q_i
$$

without consuming a resource needed by another lot.

A per-position maximum close fraction is also separable:

$$
0\le u_i\le s_{max,i}q_i.
$$

Every position can reach its own bound simultaneously because the bounds do not share a common budget.

Shared market randomness does not break feasibility separability. Neither does a position-specific holding cost or risk limit, provided it depends only on that position and does not enter a portfolio-level constraint.

#### Common sources of coupled feasibility

Opening positions is normally coupled by a shared capital constraint such as

$$
\sum_i m_i(a_i)\le E,
$$

where $m_i$ is required cash or margin and $E$ is available equity. Two openings may each be feasible in isolation but infeasible together.

Other important couplings include:

1. **Cross margin and liquidation.** Account health depends on total equity, gross exposure, net exposure and maintenance margin. Closing one side of a hedge can increase net risk even though the individual position becomes smaller.
2. **Shared borrow or inventory.** Several positions can compete for the same asset balance, borrow limit or credit line.
3. **Aggregate execution limits.** A constraint such as $\sum_i u_i\le u_{max}$ couples otherwise valid local closes. A per-position cap is separable; one shared cap is not.
4. **Minimum order size and rounding.** Two virtual closes may each be below the exchange minimum while their sum is executable. Conversely, rounding every local action separately can make the aggregate differ from the global target.
5. **Netting and hedging.** The exchange observes the net account order, while lifecycle positions may contain opposing virtual actions. Executing one without the other can change exposure and margin.
6. **Market impact and shared liquidity.** If cost is $C(\sum_i u_i)$ rather than $\sum_iC_i(u_i)$, one position's order changes the feasible or optimal execution of the others.
7. **Portfolio risk limits.** Drawdown, VaR, expected shortfall, leverage and concentration are generally functions of the whole portfolio rather than of one position at a time.
8. **Global operational limits.** Limits on open orders, request rate, turnover or traded notional are common resources.

For example, suppose two positions can each close at most $0.7$ units locally, but the account can execute at most one unit in total. The local action sets permit $(u_1,u_2)=(0.7,0.7)$, while the global constraint rejects it because $u_1+u_2=1.4$. Therefore the global feasible set is a strict subset of $\mathcal A_1\times\mathcal A_2$, and independent maximization is invalid.

#### Recovering conditional separability

A practical system can preserve local lifecycle policies by placing a global coordination layer around them.

1. **Reserve resources at creation.** Assign each position exclusive inventory, collateral and risk budget. Conditional on those reservations, local actions become separable.
2. **Centralize opening, decentralize safe closing.** A global allocator approves new positions using account-wide constraints. Local closing remains independent only when reducing a position is guaranteed not to worsen any global constraint.
3. **Aggregate virtual orders.** Let positions propose desired closes $\hat u_i$. Net and aggregate them into exchange-valid orders, then allocate fills and costs back to positions. This is particularly useful for minimum order size and rounding.
4. **Project onto the global feasible set.** Replace independent proposals by the closest or highest-value jointly feasible action:

   $$
   (u_1^*,\ldots,u_n^*)
   =\arg\max_{u\in\mathcal A_{global}}
   \sum_iQ_i(y_i,u_i,z).
   $$

5. **Use shadow prices for shared budgets.** For a constraint $\sum_i g_i(a_i)\le B$, introduce multiplier $\lambda\ge0$ and solve local problems

   $$
   \max_{a_i\in\mathcal A_i}
   \left[Q_i(y_i,a_i,z)-\lambda g_i(a_i)\right].
   $$

   A global controller adjusts $\lambda$ until the aggregate resource use satisfies the budget. Positions then optimize independently conditional on a common resource price.

The resulting architecture is hierarchical rather than completely independent: positions generate local value curves or desired actions, while the account layer enforces shared feasibility. This is sufficient to retain the position decomposition without incorrectly assuming that capital, margin, liquidity and portfolio risk have disappeared.

### Additivity of value

If PnL, proportional execution costs and holding costs are linear in size, the value of the global trajectory is the sum of the local values. For long positions, for example,

$$
\operatorname{PnL}_{global}
=\sum_i\operatorname{PnL}_i.
$$

This follows by distributing every price increment and every proportional cost across the identity $x_t=\sum_i\sigma_iq_{i,t}$. Consequently, the decomposition preserves not only exposure and trades but also the objective whenever that objective is additive over positions.

### PnL, and ROI/log wealth composition across positions

These quantities compose in different ways. Confusing them is one reason that optimizing every position independently need not optimize the account.

Let account equity at the beginning of an interval be $W_0>0$, and let $\Pi_i$ be the net PnL attributed to position $i$ over the interval. If all fees, funding and other costs have either been included in the position PnLs or retained as an explicit account-level term $C_{global}$, then

$$
W_T=W_0+\sum_i\Pi_i-C_{global}.
$$

Thus PnL is additive whenever the underlying payoff and the cost attribution are additive:

$$
\Pi_{portfolio}=\sum_i\Pi_i-C_{global}.
$$

Nonlinear impact, liquidation and other shared costs should first be computed from the aggregate account action. Allocating them back to positions is then an accounting convention and need not be unique.

#### ROI is capital-weighted

Suppose position $i$ is assigned capital $K_i>0$ and its ROI is defined by

$$
R_i=\frac{\Pi_i}{K_i}.
$$

Define its initial account-capital weight as

$$
\alpha_i=\frac{K_i}{W_0}.
$$

If the assigned capitals form a disjoint partition of account equity, $\sum_iK_i=W_0$, and all costs are included, then portfolio ROI is

$$
\begin{aligned}
R_{portfolio}
&=\frac{W_T-W_0}{W_0}\\
&=\frac{\sum_i\Pi_i}{W_0}\\
&=\sum_i\alpha_iR_i.
\end{aligned}
$$

Therefore portfolio ROI is a capital-weighted mixture of position ROIs, not their unweighted sum. Unallocated quote can be included as another component with its own return, normally zero before funding or interest.

For leveraged or cross-margin positions, the choice of $K_i$ may not be canonical. Several positions can use the same account collateral, so assigning the full shared equity to every position would double-count capital. One can reserve explicit capital or risk budgets per position, but otherwise position ROI is a diagnostic ratio rather than a quantity that composes uniquely. The always-defined account-level contribution is

$$
\rho_i=\frac{\Pi_i}{W_0},
\qquad
R_{portfolio}=\sum_i\rho_i-\frac{C_{global}}{W_0}.
$$

#### Simultaneous position returns do not add logarithmically

Let

$$
M_i=1+R_i=e^{g_i}
$$

be the gross return multiplier and log return of a position with exclusively assigned capital. For simultaneous unleveraged positions with nonnegative weights $\alpha_i$ summing to one, the portfolio gross multiplier is

$$
M_{portfolio}
=\sum_i\alpha_iM_i
=\sum_i\alpha_ie^{g_i}.
$$

The portfolio log return is therefore

$$
\boxed{
g_{portfolio}
=\log\left(\sum_i\alpha_ie^{g_i}\right)
}
$$

rather than $\sum_i\alpha_ig_i$ or $\sum_ig_i$. For nonnegative weights, concavity of the logarithm gives

$$
g_{portfolio}
\ge\sum_i\alpha_ig_i,
$$

with equality when all active gross multipliers are equal. This inequality does not apply directly to signed or leveraged weights.

The most general account-level expression uses PnL contributions:

$$
\boxed{
g_{portfolio}
=\log\left(
1+\sum_i\frac{\Pi_i}{W_0}
-\frac{C_{global}}{W_0}
\right),
}
$$

which requires the quantity inside the logarithm to remain positive. A liquidation or loss that makes equity nonpositive lies outside the domain of log utility and must be modeled as ruin or an absorbing boundary.

It is generally incorrect to assign every lifecycle lot its own account-level log utility and add the results:

$$
\log\left(1+\sum_i\rho_i\right)
\ne
\sum_i\log(1+\rho_i).
$$

The right-hand side treats each lot as if it owned a separate copy of account equity. For positions sharing one account, PnL must be aggregated first and the logarithm applied once.

#### Log wealth adds across time

Log returns compose additively across consecutive time intervals:

$$
\begin{aligned}
\log\frac{W_T}{W_0}
&=\log\prod_{t=0}^{T-1}\frac{W_{t+1}}{W_t}\\
&=\sum_{t=0}^{T-1}\log\frac{W_{t+1}}{W_t}\\
&=\sum_{t=0}^{T-1}
\log\left(1+\sum_i\rho_{i,t}-c_t\right),
\end{aligned}
$$

where $\rho_{i,t}=\Pi_{i,t}/W_t$ and $c_t$ is the account-level cost as a fraction of current equity. This is the natural additive reward for a sequential trading policy.

If positions use the entire account one after another without overlapping, with gross multipliers $M_1,\ldots,M_n$, then

$$
W_T=W_0\prod_iM_i,
\qquad
\log\frac{W_T}{W_0}=\sum_i\log M_i.
$$

This special sequential case is the circumstance in which position log returns genuinely add. It does not hold for positions active simultaneously on different portions of the same account.

#### Consequences for lifecycle decomposition and optimization

Splitting one global position into lifecycle lots preserves account log wealth because the split preserves aggregate PnL. If

$$
\Pi=\sum_i\Pi_i,
$$

then

$$
\log\left(1+\frac{\Pi}{W_0}\right)
=\log\left(1+\sum_i\frac{\Pi_i}{W_0}\right)
$$

regardless of whether FIFO, LIFO or pro-rata labels are used. The labels change attribution but not account wealth.

However, expected log wealth across simultaneous positions is

$$
\mathbb E\left[
\log\left(1+\sum_i\rho_i\right)
\right],
$$

which depends on their **joint distribution**, including correlations and common tail events. It is not equal to a sum of independently optimized expected position log returns. Therefore the position decomposition justifies local lifecycle bookkeeping and, under separability, local action proposals; it does not by itself justify independently maximizing each position's ROI or log utility.

Partial exits introduce another attribution issue: returned capital may be reused by later positions. Summing the ROIs of both positions then counts the same capital more than once. Global time-indexed equity and cumulative log growth remain well-defined, while per-position ROI requires a fixed capital-allocation and cash-flow convention.

In continuous time, the same result can be expressed by treating positive exposure variation as a measure of position births and negative variation as a measure of liquidations, then matching liquidation mass only to positions born earlier. The remaining mass of every born position is nonincreasing. A pro-rata liquidation rate gives the continuous analogue of the common fractional update above.

### Consequence for the position-based state representation

This establishes the missing equivalence used earlier in the document:

1. **Collapse:** given a set of lifecycle-constrained positions, summing their signed remaining sizes recovers the global portfolio exposure, and summing their orders recovers the global order flow.
2. **Lift:** given a trajectory of the global portfolio position, the construction above represents every exposure increase by a newly created position and every exposure decrease by reductions of existing positions. No created position ever needs to grow.

Therefore replacing one globally managed portfolio position with a set of positions that can only be held or reduced after entry does not restrict the attainable portfolio trajectories under the separability assumptions. It adds lifecycle labels and a local decomposition of the same trades. The labels are not unique unless a canonical allocation rule such as pro rata, FIFO or LIFO is fixed; this non-uniqueness does not affect aggregate exposure or additive PnL.

Opening remains globally constrained by available equity, collateral and leverage. Once a collection of positions is feasible, however, their close updates can be evaluated locally and combined additively when the fractional close policy and costs are separable. This is the precise sense in which the earlier split into a set of positions is justified.

The equivalence can fail for fixed per-order fees, minimum order sizes applied separately to each local position, nonlinear market impact, position-specific rules based on different entry prices or ages, or non-additive risk objectives. In those cases the global trade need not equal the sum of independently chosen local trades. The trajectory can still be split into lots as bookkeeping, but those lots cannot necessarily reproduce the global policy as autonomous local policies.

### etc
(\*1) Note that from this policy it follows that we can partition all positions by continuous holding and partial close decisions.

Maintenance costs also become increasingly important, since now holding can also be costly, making overall costs constantly grow with time. Lets call holding costs $c_h$, then policy changes as follows:
1. strictly outside $[p_t+c_h, p_t-c_{max}]$ - hold if $p_{t+1}>p_t+c_h$, sell if $p_{t+1}<p_t-c_{max}$
2. within $[p_t+c_h, p_t-c_{min}]$:
	1. second move larger than first by $c_h$, $|p_{t+2}|>|p_{t+1}|+c_h$, then necessarily $p_{t+2}>p_t+c_h$. If we hold we are sure to get better position at the cost of the intermediate drawdown of size $\Delta p_t$. Selling and buying at $p_{t+1}$ is worse if $c_h<c$, since we pay roundtrip cost at the same price movement. The condition $c_h<c$ is guaranteed to be false by construction. That places break-even $p_{t+2}$ strictly above holding's break even $p_t$. Thus we always hold.
	2. In case it is smaller, then neither is better, we need to look at the next price move. Next move is same direction, and if in total it moves us below $p_t-c_{min}>p_{t+3}$, then now its better to sell.
	3. Thus we can conclude that staying in this range necessarily requires looking further into the future.
	4. As we look deeper, the cost of holding, as well as partial close increases linearly with $h$.
	5. By induction it is preferrable to hold when $p_{t+h}>p_t+h*c_h$ and $p_{\tau}>p_t-c_{min}$ for any $\tau \in [t,t+h]$. Similarly for selling - it is preferrable when $p_{t+h}>p_t-c_{min}$ and $p_t+h*c_h>p_{\tau}$ for any $\tau \in [t,t+h]$.
	6. Note that $c_{min}=f*s_{min}*p_t+(1-s_{min})*h*c_h+f*s_{min}*p_{t+1}$ increases with $h$ a bit slower than the pure holding cost itself.
	7. As $h \to \infty$, if we are still within the range, then the asset can be considered frozen, as there are no useful opportunities. In which case it is better to invest in another, more volatile asset. And that means selling all and never touching it again.

We could naturally express action in this case as sell fraction $s_t$ at each time step. Each branch has its own probability in terms of next price probabilities, which means we can compute expectation of $s_t$ for each time step.

(\*3) Notice that we also can apply basically the same policy for short positions. And if we consider 0 exposure position as short with respect to asset, then we can apply it for determining entry points as well.

$$\begin{aligned}
p_t&>p_{t+1}\\
(1-s)p_{t+1}+s(1-f)^2p_t&\ge p_t \quad \text{equity after price drop and repurchase of s is not reduced}\\
1-s+s(1-f)^2p_t/p_{t+1}&\ge p_t/p_{t+1}\\
1-s+skR&\ge R\\
s(kR-1)&\ge R-1\\
s&\ge \frac {R-1} {kR-1}>1 \quad R>kR>1,\ \text{change is bigger than roundtrip cost}\\
s&\le \frac {R-1} {kR-1}<0 \quad 1> kR\ge k,\ \text{change is smaller than roundtrip cost while price dropped}\\
s&\ge \frac {R-1} {kR-1}>0 \quad 1> R > kR,\ \text{change is positive and we will preserve equity if we sell fraction s now and observe price }p_{t+1}>p_t\\
\end{aligned}
$$

$$\begin{aligned}
p_t&>p_{t+1}\\
(1-s)p_{t+1}+s(1-f)^2p_t&\ge p_{t+1} \quad \text{equity after price drop and repurchase of s is better than holding}\\
(1-s)p_{t+1}/p_{t}+s(1-f)^2&\ge p_{t+1}/p_{t}\\
(1-s)R+sk&\ge R\\
-sR+sk&\ge 0\\
s&\ge \frac {R-1} {kR-1}>1 \quad R>kR>1,\ \text{change is bigger than roundtrip cost}\\
s&\le \frac {R-1} {kR-1}<0 \quad 1> kR\ge k,\ \text{change is smaller than roundtrip cost while price dropped}\\
\end{aligned}
$$
### Continuously selling partially as price falls

For any exit-only path, define the fraction of the then-current inventory sold at time $t$ by

$$
s_t=\frac{u_t}{q_t}\in[0,1],\qquad
q_{t+1}=q_t(1-s_t).
$$

Thus a sequence of partial sells is a valid way to represent liquidation. However, nothing above proves that $s_t>0$ whenever price falls, that $s_t$ changes continuously, or that it increases with the size of a fall. Indeed, under linear PnL and proportional fees the size cancels from the hold-versus-sell comparison, producing a bang-bang decision except at exact indifference.

A monotone partial-sale theorem would require additional structure. One possible route is to define a Bellman action value $Q(s,u)$ with nonlinear impact or a convex risk penalty, prove concavity in the sold amount $u$, and prove increasing differences between the adverse-price state and $u$. The first-order condition can then give an interior fraction, while monotone comparative statics gives a nondecreasing sale amount as the state becomes more adverse. Those assumptions and that proof are absent from the current model, so the marked statement should be treated as a design intuition, not a derived property.

## Long/short and entry/exit symmetry

We can reduce optimal policy through symmetry to just consider exiting long positions and then derive entry/short behavior from that.

**both long/short reflection and the base-quote entry/exit duality have conditional proofs, but they are different symmetries. Zero asset exposure is a relative short under a numeraire swap, not a literal borrowed margin short.**

Suppose inventory $x$ is signed, one-period trading PnL is linear in inventory,

$$
r(x,\Delta p)=x\Delta p,
$$

the trading cost is sign-symmetric, $C(\Delta x)=C(-\Delta x)$, long and short actions have mirrored constraints, and funding, borrow availability and liquidation rules are also symmetric. Under the sign transformation

$$
(x,\Delta p,\Delta x)\mapsto(-x,-\Delta p,-\Delta x),
$$

both reward and costs are unchanged. Applying this transformation to every step maps every admissible long trajectory bijectively to a short trajectory with the same objective value. Therefore the optimal short policy is the sign-reflection of the optimal long policy evaluated under the return-reflected state. This is the proof sketch for the valid part of the claim.

Real margin trading violates some of these assumptions through asymmetric borrow costs, borrow limits, liquidation and possibly fee schedules, so the symmetry must be tested against the actual account model.

There is a second, distinct symmetry that supports the entry-point intuition. Suppose the account holds $Q$ units of quote currency and the asset price is $p$ quote units per asset. Its wealth measured in quote is constant,

$$
W^{quote}=Q,
$$

but its wealth measured in asset units is

$$
W^{asset}=\frac{Q}{p}.
$$

If price falls from $p_0$ to $p_1$, the return of the quote holding in the asset numeraire is

$$
\frac{Q/p_1}{Q/p_0}-1=\frac{p_0}{p_1}-1>0.
$$

Thus, after exchanging the roles of base and quote, the relevant price is $\tilde p=1/p$, which rises whenever $p$ falls. The symmetry is exact for log returns:

$$
\Delta\log\tilde p=-\Delta\log p.
$$

Consequently, holding quote is the opposite **relative position** to holding the asset. Buying the asset closes the quote position when the portfolio is described in the asset numeraire. The mapping is

$$
\begin{aligned}
p&\longleftrightarrow 1/p,\\
\text{asset}&\longleftrightarrow\text{quote},\\
\text{enter asset}&\longleftrightarrow\text{exit quote},\\
\text{exit asset}&\longleftrightarrow\text{enter quote}.
\end{aligned}
$$

This proves the entry/exit duality when the portfolio contains only the base and quote assets, conversion costs and admissible allocations remain symmetric after exchanging them, and the objective is invariant to the numeraire change. A pathwise terminal-wealth objective has this invariance because every policy's terminal wealth is multiplied by the same positive factor $1/p_T$. Expected log wealth also has it because

$$
\log W_T^{asset}=\log W_T^{quote}-\log p_T,
$$

and the second term is independent of the policy when the market path is exogenous. Expected arithmetic wealth is not generally invariant: multiplying the random policy wealth by $1/p_T$ can change its expectation and the ranking of policies.

This relative position is still not operationally identical to a margin short. A borrowed short earns positive quote-denominated PnL when $p$ falls, whereas an unleveraged quote holding keeps the same quote value and gains only relative to the asset. Borrowing costs, leverage, liquidation, asymmetric constraints, a third asset, or a non-invariant objective can therefore break the mapping.

Under the stated symmetry, an optimal exit rule for the quote position at reciprocal price $1/p$ transforms into an optimal entry rule for the asset at price $p$. With transaction costs, the two transformed switching rules can still form a no-trade region; the duality does not imply identical entry and exit thresholds expressed directly in $p$.

## Bellman symmetries and the minimal optimization domain

The preceding fraction is a terminal or myopic optimization unless its objective is also the relevant Bellman action value. To determine which entry and exit problems can be obtained from one another, the symmetry has to be applied to the complete controlled process, not only to the one-period PnL formula.

Let $x_t$ be the complete Markov state. It includes at least wealth, prices, current signed exposure, position lifecycle state, collateral and resting orders. Let $a_t$ be a target-exposure or order action, let $\mathcal A_t(x_t)$ be its feasible set, and let $K_t(dx'\mid x,a)$ be the transition kernel. The Bellman equations are

$$
Q_t^*(x,a)
=
\int
\left[
r_t(x,a,x')+\beta V_{t+1}^*(x')
\right]
K_t(dx'\mid x,a),
$$

$$
V_t^*(x)=\sup_{a\in\mathcal A_t(x)}Q_t^*(x,a).
$$

For the forecast-randomized formulation used above, a policy generated from possible next-price forecasts is

$$
\pi_\phi(ds_t\mid\mathcal F_t)
=
\int
P_t(d\bar p_{t+1}\mid\mathcal F_t)
\delta_{\phi(x_t,\bar p_{t+1})}(ds_t).
$$

This policy is Bellman-optimal only if its support consists of Bellman-optimal actions:

$$
\phi(x_t,\bar p_{t+1})
\in\arg\max_s Q_t^*(x_t,s)
\quad\text{for }P_t\text{-almost every }\bar p_{t+1}.
$$

If $Q_t^*(x_t,\cdot)$ is strictly concave, its optimum is unique and the condition reduces to

$$
\phi(x_t,\bar p_{t+1})=s_t^*
\quad P_t\text{-almost surely}.
$$

Thus a non-degenerate pushforward over several different current actions is not optimal for the ordinary expected-utility Bellman problem unless all those actions are tied. The complete predictive distribution normally determines one current optimal action. The pushforward is still useful for describing a distribution of later actions, an explicit randomized strategy, or an oracle benchmark.

### Bellman symmetry theorem

Let $S:x\mapsto Sx$ transform states and let $T_x:a\mapsto T_xa$ be a bijective transformation of actions at state $x$. The transformation is a symmetry of the Bellman problem when all of the following hold:

1. Feasible actions are mapped exactly:

   $$
   \mathcal A_t(Sx)=T_x\mathcal A_t(x).
   $$

2. Rewards are preserved:

   $$
   r_t(Sx,T_xa,Sx')=r_t(x,a,x').
   $$

3. The transition kernel is equivariant. For every measurable next-state set $B$,

   $$
   K_t(SB\mid Sx,T_xa)=K_t(B\mid x,a).
   $$

4. Terminal utility is preserved:

   $$
   V_T(Sx)=V_T(x).
   $$

The equalities may also contain a common positive scaling or an action-independent Bellman-consistent offset, since neither changes the maximizing action. They may not contain an action-dependent correction.

Under these conditions, backward induction gives

$$
\boxed{V_t^*(Sx)=V_t^*(x)},
\qquad
\boxed{Q_t^*(Sx,T_xa)=Q_t^*(x,a)}.
$$

Indeed, the result is true at the terminal time by condition 4. If it is true for $V_{t+1}^*$, conditions 2 and 3 make the transformed continuation integral equal to the original one. Condition 1 then makes maximization over the transformed feasible set equivalent to maximization over the original set. This proves the result inductively. For an infinite discounted problem the same conclusion follows because the Bellman operator preserves the symmetry and has a unique fixed point.

The optimal policy therefore transforms as

$$
\boxed{
\pi_t^*(\,\cdot\mid Sx)
=(T_x)_\#\pi_t^*(\,\cdot\mid x)
}.
$$

When the optimum is unique and deterministic,

$$
\boxed{a_t^*(Sx)=T_xa_t^*(x)}.
$$

If there are several optimal actions, a particular arbitrary selection need not look symmetric, but the set of optimal actions is transformed exactly and a symmetric stochastic selection can be chosen.

### Signed target exposure as the common action

Let

$$
\ell_t
=
\frac{\text{signed risky notional}}{\text{account equity}}
$$

be the target exposure. Positive $\ell$ is long, negative $\ell$ is short and $|\ell|>1$ is leveraged. If the current exposure is $\ell_0$, a simple one-period wealth transition has the form

$$
M_{t+1}(\ell;\ell_0,\rho)
=
1+\ell\rho-B_t(\ell)-C_t(\ell-\ell_0),
$$

where $\rho=p_{t+1}/p_t-1$, $B_t$ contains borrowing and funding costs, and $C_t$ contains turnover costs. The general Bellman action is

$$
\ell_t^*
=
\arg\max_{\ell\in\mathcal L_t(x_t)}
Q_t^*(x_t,\ell).
$$

Entry, holding, exit and reversal are restrictions or regions of this one target-exposure action:

| Operation   | Current exposure | Target exposure |
| ----------- | ---------------: | --------------: |
| long entry  |              $0$ |            $+L$ |
| long exit   |           $+L_0$ |     $+(1-s)L_0$ |
| short entry |              $0$ |            $-L$ |
| short exit  |           $-L_0$ |     $-(1-s)L_0$ |

Here $L\ge0$ and $s\in[0,1]$. For either sign, an exit-only lifecycle is obtained by imposing

$$
\ell=(1-s)\ell_0,
\qquad 0\le s\le1.
$$

Consequently, the close fraction corresponding to a Bellman-optimal target that remains on the same side is

$$
s_t^*=1-\frac{\ell_t^*}{\ell_0}.
$$

This shows the precise relationship between the optimal-fraction section and the account-level Bellman policy: the fraction is a reparametrization of a target-exposure action restricted to the interval between the existing exposure and zero.

### Long-short symmetry

Long-short reflection uses

$$
(\ell,\rho)\mapsto(-\ell,-\rho).
$$

It is an exact Bellman symmetry only if the transformation is also applied to the conditional return law, fees, borrowing, funding, liquidation, collateral, leverage bounds and every future feasible action. In the symmetric one-period model,

$$
M_{short}(L,\rho)
=1-L\rho-C(-L)
=M_{long}(L,-\rho)
$$

when $C(-L)=C(L)$. If $P_{-\rho}=(-\operatorname{id})_\#P_\rho$ is the reflected return distribution, then

$$
\boxed{
L_{short}^*(P_\rho)
=L_{long}^*(P_{-\rho})
},
$$

or, in signed-exposure notation,

$$
\boxed{
\ell^*(Sx)=-\ell^*(x)
}.
$$

This relates two transformed belief states. It does not imply equal long and short actions under the same asymmetric return distribution. Asset-borrow fees, unequal funding, asymmetric leverage limits and liquidation rules all break the symmetry and require the short side to be optimized separately.

### Base-quote symmetry

Swapping the asset and quote roles maps

$$
p\mapsto\frac1p
$$

and transforms holdings, orders, liabilities and return distributions accordingly. Under this transformation, closing an asset position can be viewed as entering the quote asset when wealth is measured in asset units. This proves a relative-exposure symmetry, but not automatically a margin-short symmetry: zero asset exposure is short relative to the asset numeraire, while a margin short contains a borrowed negative asset balance.

Expected log wealth has especially useful numeraire behavior:

$$
\log W^{asset}=\log W^{quote}-\log p.
$$

If the price transition is exogenous to the action, the additional log-price term can be action-independent, allowing the transformed Bellman argmax to be preserved. General CRRA utility is not automatically numeraire-invariant because

$$
U_\gamma\left(\frac Wp\right)
=
\frac{p^{\gamma-1}W^{1-\gamma}-1}{1-\gamma}.
$$

The random factor $p^{\gamma-1}$ changes scenario weights when $\gamma\ne1$ and can change the optimal policy. A base-quote reduction therefore requires a proof that utility, transition probabilities and all account mechanics transform equivariantly; price inversion by itself is insufficient.

### Entry-exit is a restriction, not generally a symmetry

Long entry and long exit use the same Bellman action-value function but evaluate it at different states and over different feasible sets:

$$
\ell_{entry}^*
=
\arg\max_{\ell\ge0}Q_t^*(x_{flat},\ell),
$$

$$
\ell_{exit}^*
=
\arg\max_{0\le\ell\le L_0}Q_t^*(x_{long},\ell).
$$

Entry creates exposure and its future optionality, while exit removes them. Current inventory, unrealized PnL, sunk costs, liquidation distance, holding duration, minimum orders and future funding can all distinguish the two states. Therefore the optimal exit cannot generally be recovered from the optimal flat-state entry merely by reversing the trade.

Entry-exit becomes an exact symmetry only if an explicit reversible transformation maps the entire flat-entry state to the invested-exit state while preserving rewards, transition probabilities, feasible continuation actions and terminal utility. Without that strong condition, exit must remain in the optimization domain as a different current-exposure state, although its action is still just the restriction $\ell\in[0,L_0]$.

### Minimal domain that must be optimized

Let $G$ be the collection of transformations that satisfy the complete Bellman symmetry conditions. States related by such transformations form an orbit

$$
[x]=\{Sx:S\in G\}.
$$

It is sufficient to optimize one representative from each orbit, or equivalently the quotient state space $\mathcal X/G$. Values and policies on the other states are reconstructed by

$$
V_t^*(Sx)=V_t^*(x),
\qquad
\pi_t^*(\,\cdot\mid Sx)=(T_x)_\#\pi_t^*(\,\cdot\mid x).
$$

For this trading problem, the smallest defensible domain depends on which symmetries have actually been proved:

1. With exact long-short symmetry, optimize only one exposure sign together with one canonical reflected return distribution. Recover the other sign by reflecting the state, belief and action. Current exposure magnitude, including zero versus nonzero exposure, must still remain in the state.
2. With exact base-quote symmetry as well, choose one canonical numeraire and reconstruct swapped asset/quote states. This reduction is most natural for expected log wealth and symmetric conversion mechanics.
3. Without entry-exit reversibility, retain at least two lifecycle classes: flat states, where entry leverage and direction are chosen, and invested states, where the target exposure may be held, reduced, increased or reversed. An exit-only position further restricts the invested action set to the interval between its current exposure and zero.
4. If borrowing, funding, liquidation or execution is asymmetric, retain separate long and short state classes. If utility is not numeraire-equivariant, retain separate base- and quote-measured problems as well.

Accordingly, the generally safe minimal primitive is not long entry alone. It is the Bellman optimization of signed target exposure conditional on current signed exposure and the complete account state. Long entry can serve as the only explicitly optimized operation only after long-short, base-quote and entry-exit transformations have each been shown to satisfy the full symmetry theorem. In the more realistic model, long and short may be related by a useful approximate symmetry, while entry and exit remain different restrictions of the same target-exposure Bellman problem.

## Event-compressed price predictions and decision points

The exit-policy regions suggest a smaller prediction target than an arbitrary-depth raw price path. Consecutive raw movements can be combined until they reach a point at which the policy can make a materially different decision. The resulting process is indexed by decision events rather than by every exchange candle.

The exact construction below has two layers: compute an executable buy/sell/no-trade table, then follow it and merge its actions into directional runs, coasting intervals and cash intervals. The running-extremum scan supplies the representation; the decision table supplies the switching test. A price-only threshold scan is not substituted for that test.

### Maintenance coordinates and elementary blocks

First classify and group consecutive candles with the same maintenance-adjusted preference. Keep the raw decision points inside these provisional blocks: grouping is an index over the history, not permission to discard their execution opportunities. The forward scan below subsequently resolves counter-moves and flat blocks. An elementary flat block is not necessarily a constant-price interval; it is an interval where neither direction has positive carrying return before execution fees.

For candle $i$, let

$$
r_i=\log\frac{p_{i+1}}{p_i},
\qquad
h_i=\int_{t_i}^{t_{i+1}}c(u)\,du.
$$

Under the symmetric per-unit directional-log maintenance model, classify it as

$$
\boxed{
\begin{cases}
\mathrm{long\text{-}favorable},&r_i>h_i,\\
\mathrm{flat},&|r_i|\le h_i,\\
\mathrm{short\text{-}favorable},&r_i<-h_i.
\end{cases}
}
$$

Equality is a carrying-return tie, assigned to flat by convention. This is the fee-free local classification at the selected candle resolution.

The corresponding directional curves are

$$
\boxed{
L_{\mathrm{long}}=\log p-\int c\,dt,
\qquad
L_{\mathrm{short}}=-\log p-\int c\,dt.
}
$$

A favorable short run is an increase of $L_{\mathrm{short}}$. Since $L_{\mathrm{short}}\ne-L_{\mathrm{long}}$ when maintenance is positive, an unfavorable long move need not be favorable for shorting. Use the actual side-specific rates when they differ.

For the long-side multiplicative asset-unit maintenance convention, the transformation is exact even with leverage. If $da=-ca\,dt$ between trades, define

$$
C_t=\int_0^t c(u)\,du,\qquad
\widetilde a_t=a_te^{C_t},\qquad
\widetilde p_t=p_te^{-C_t}.
$$

Then $\widetilde a$ and quote holdings are constant between trades,

$$
W=q+\widetilde a\widetilde p,\qquad
e=\frac{\widetilde a\widetilde p}{W},
$$

and proportional execution fees retain their original form in the transformed coordinates. An adjusted displacement already includes maintenance; do not subtract its duration cost again.

Other maintenance conventions require transforming execution and constraints as well. For a fixed balance-sign regime with $dq=g_q q\,dt$ and $da=g_a a\,dt$, let $B_q=\exp(\int g_qdt)$ and $B_a=\exp(\int g_adt)$. Then

$$
\widehat q=q/B_q,\qquad
\widehat a=a/B_a,\qquad
\widehat p=pB_a/B_q,\qquad
W/B_q=\widehat q+\widehat a\widehat p.
$$

This removes maintenance within that regime and preserves proportional fees, but borrowing rates that change when balances change sign require retaining the regime and its factors. For quote-debited long maintenance $dq=-cpa\,dt$, $da=0$, setting $I_t=\int_0^t c_up_u\,du$ gives $\widehat q=q+aI_t$ and $\widehat p=p-I_t$. The transformed ask and bid are $p/(1-f)-I_t$ and $(1-f)p-I_t$, not a constant proportional spread around $\widehat p$. Original exposure is still $ap/W$, not $a\widehat p/W$; the adjusted mark may also cease to be positive. Such a transform is useful accounting, not automatically the same log-price compression with unchanged thresholds.

### Operational construction: exact decisions, then run compression

The objective is deterministic hindsight maximization of terminal wealth, including final liquidation, for one asset and quote. Equivalently, maximize terminal log wealth. The procedure below is exact on a **specified finite execution grid**, with a continuous interval of target exposures. It does not approximate that interval with a sampled action grid.

The inputs and scope are:

- prices $p_0,\ldots,p_N>0$, timestamps, initial wealth and exposure;
- executable targets $E=[-\lambda_-,\lambda_+]$, including zero;
- a possibly wider effective-exposure interval, with no trade allowed outside $E$ while still inside that effective interval;
- proportional execution fees $f$, known maintenance factors, and compulsory liquidation to cash at $N$;
- maintenance of the form $q'=q_tq+k_tap_t$, $a'=a_ta$ with known coefficients within finitely many regimes; this includes Oracle.md's borrowing and multiplicative funding models, and proportional notional costs debited from quote;
- no fixed fees, minimum order, lot rounding, market impact, pending orders or additional path-dependent account state;
- closed feasible exposure pieces on which equity and the execution factors stay positive. For $f>0$, use an effective interval strictly inside $(-(1-f)/f,1/f)$; for $f=0$ this extra fee-solvency bound is absent. Include all specified intermediate margin checkpoints in these pieces.

Thus this is a grid-time theorem, not a claim about unrestricted continuous-time execution or stochastic optimality. Rates may vary with time. The necessary factors use physical duration even when adjusted prices are used for drawing the runs.

#### 1. Turn each candle into explicit account coefficients

Write $x$ for current exposure and $e$ for the selected post-trade exposure. Let $b=1-f$. The exact execution multiplier is

$$
R(x,e)=
\begin{cases}
\dfrac{b+fx}{b+fe},&e>x,\\[5pt]
\dfrac{1-fx}{1-fe},&e<x,\\[5pt]
1,&e=x.
\end{cases}
$$

Within one maintenance regime, let $q_t>0$ and $a_t>0$ be the quote-balance and asset-unit multipliers over the candle, $k_t$ the signed quote credit per unit of starting asset notional, and $\rho_t=p_{t+1}/p_t$. Usually $k_t=0$; a quote-debited notional charge is represented by its appropriate side-specific $k_t$. Define

$$
\alpha_t=q_t,\qquad \eta_t=a_t\rho_t-q_t+k_t,\qquad \chi_t=a_t\rho_t.
$$

After the order, maintenance and price move give

$$
\boxed{
g_t(e)=\alpha_t+\eta_te,\qquad
d_t(e)=\frac{\chi_te}{\alpha_t+\eta_te},\qquad
(W_{t+1},x_{t+1})=(W_tR(x,e)g_t(e),d_t(e)).
}
$$

For debt maintenance, split at $e=0$ and $e=1$: the asset is borrowed below zero and quote is borrowed above one. Use the actual factors separately on each piece. For a no-maintenance candle, $(\alpha_t,\eta_t,\chi_t)=(1,\rho_t-1,\rho_t)$.

Feasibility is also explicit. At each checkpoint with coefficients $(\alpha,\eta,\chi)$, intersect with

$$
\alpha+\eta e>0,\qquad
E_{\mathrm{eff},-}(\alpha+\eta e)\le\chi e
\le E_{\mathrm{eff},+}(\alpha+\eta e).
$$

On the stated closed safe domain these are interval cuts. Here $E_{\mathrm{eff},-}<0$ is the signed lower bound. Evaluate no trade using $e=x$ against these same checks; do not clamp a feasible coasting exposure to the target interval.

#### 2. Build the decision table using endpoints and three comparisons

Let $v_t(x)$ be optimal terminal cash per unit of current marked wealth. This is a wealth multiplier, not a log value. Store it as affine pieces $(l,u,A,B)$, meaning $v_t(x)=A+Bx$ on $[l,u]$, together with the winning action. Store boundary values explicitly when needed.

**Initialize the last candle with the actual closeout:**

$$
v_N(x)=R(x,0)=
\begin{cases}
1+fx/b,&x\le0,\\
1-fx,&x\ge0.
\end{cases}
$$

Then process $t=N-1,N-2,\ldots,0$:

1. **Pull back the next table's intervals.** For each next-value piece $A+By$ and each maintenance regime, find the interval where $d_t(e)\in[l,u]$. A finite inverse boundary is
   $$e=\frac{\alpha_t y}{\chi_t-\eta_t y},\qquad y\in\{l,u\}.$$
   If the denominator vanishes, use the defining interval inequalities instead of creating an infinite endpoint. Intersect with the regime and feasibility intervals.
2. **Write the continuation on that interval as a line.** It is exactly
   $$H_t(e)=g_t(e)v_{t+1}(d_t(e))
   =A\alpha_t+(A\eta_t+B\chi_t)e.$$
   Retain $H_t$ over the effective domain for evaluating no trade. Separately intersect its pieces with $E$ to obtain executable trade targets.
3. **Collect the finite target list $u_1<\cdots<u_m$.** Take all endpoints of those feasible target pieces, including maintenance boundaries and preimages of the next table's boundaries. Evaluate $H_t(u_j)$ from the actual applicable pieces. No unknown optimal fraction is used in constructing this list.
4. **Make two running-record arrays.** At each target compute
   $$B_j=\frac{H_t(u_j)}{b+fu_j},\qquad S_j=\frac{H_t(u_j)}{1-fu_j}.$$
   Scan right-to-left, retaining the largest $B_j$ and its target; scan left-to-right, retaining the largest $S_j$ and its target. These are suffix and prefix maximum arrays.
5. **At any current exposure $x$, compare exactly three scores:**
   $$\boxed{
   \begin{aligned}
   C_{\mathrm{hold}}(x)&=H_t(x),\\
   C_{\mathrm{buy}}(x)&=(b+fx)\max_{u_j>x}B_j,\\
   C_{\mathrm{sell}}(x)&=(1-fx)\max_{u_j<x}S_j.
   \end{aligned}}
   $$
   Assign $-\infty$ to an infeasible hold or an empty trade set. The largest score determines the action: no trade, buy to the stored suffix target, or sell to the stored prefix target. Prefer no trade at an exact tie; otherwise prefer the tied target nearest $x$, then the smaller target.
6. **Store the resulting intervals.** Between consecutive target and $H_t$ boundaries, the three scores are affine in $x$. Split at their pairwise intersections, retain the winning line and action on each subinterval, and handle boundary ties with the same rule. This produces $v_t$ and its executable action table $\mathcal D_t$.

In particular, there is no instruction here to “find the best program” or solve an unspecified optimization problem. The operations are interval intersection, evaluation of linear formulas, two record scans, and intersection of at most three score lines on each interval. The number of pieces can grow with the horizon; this is an exact construction, not a claim of constant-size memory.

The actual policy boundaries are equally explicit. If two competing score lines are $a_1+b_1x$ and $a_2+b_2x$, their boundary is

$$
\boxed{x_* = \frac{a_2-a_1}{b_1-b_2}}
$$

when the slopes differ and the crossing lies in the interval. Equal slopes mean dominance or a tie throughout it. These boundaries depend on the remaining path and account model. They are not universally one constant displacement from a price extremum.

#### 3. Follow the table and construct runs with a running extremum

Keep this scanner state:

$$
(\mathrm{side},\ e_{\mathrm{follow}},\ t_{\mathrm{start}},\ M,\ t_M,\
\mathrm{pending},\ t_{\mathrm{execution}},\ W,x).
$$

Here side is long, cash or short; $e_{\mathrm{follow}}$ is the last selected target; $M$ is the running maximum of the active directional curve $L_{\mathrm{side}}$; pending stores the counter-move's start and most adverse value/time. The execution anchor is the time of the last actual order, not necessarily $t_M$. Initialize side from the initial holdings, $e_{\mathrm{follow}}=x_0$ and $t_{\mathrm{start}}=0$, with pending empty and $M=L_{\mathrm{side}}(0)$ when invested. If the previous execution anchor is unknown, record it as unknown rather than inferring one from the initial price.

For each raw candle boundary $t$, **before applying that candle's return**, update the active extremum using the price already observed at $t$:

- Below $M$, start or extend pending and track its most adverse value.
- At or above $M$, absorb a pending no-trade counter-move, clear it, and update $M,t_M$ on a strict new high. This happens before any restoration order at the recovery price.
- Apply this update on order candles as well as no-trade candles. Do not let the high-water mark become stale while exposure is being restored.

Then look up $\mathcal D_t(x)$ and execute the following branches:

1. **Currently cash.**

   - If the selected target is zero, extend the cash interval.
   - If it is positive or negative, end the cash interval at $t$, emit the entry, set the new side, $e_{\mathrm{follow}}=e$ and both start/execution times to $t$, and initialize $M=L_{\mathrm{side}}(t)$, $t_M=t$ and pending to empty.
2. **Currently invested, and the action is no trade.** Extend the current interval and keep the execution anchor unchanged. The extremum update keeps its counter-move pending until recovery. An elementary flat block is bridged in exactly the same way when the policy remains invested through it.
3. **The selected target is zero.** End the invested interval at $t$, emit its exit, start cash with $e_{\mathrm{follow}}=0$ and start/execution times $t$, and clear pending. A pending adverse tail ending here is retained as part of the old interval; do not pretend it recovered.
4. **The selected target has the opposite sign.** End the old interval and begin the opposite run at $t$. Execute one direct reversal using $R(x,e)$, not an artificial close-to-zero followed by a separate entry. Initialize the new side's state exactly as for an entry.
5. **An order keeps the same side.** Compare $e$ with $e_{\mathrm{follow}}$ before updating it. A restoration of the same target on a favorable extension is an internal run order. A different target or an order inside a pending counter-move ends the current follower interval and begins a same-side adjustment interval at $t$; it is not relabeled as no-trade coasting. After an adjustment, set $M=L_{\mathrm{side}}(t)$, $t_M=t$ and pending to empty. Ordinary favorable restorations retain the updated high-water mark. In either case record the order, set $e_{\mathrm{follow}}=e$, and update the execution anchor to $t$.
6. **Advance the account** using $W\leftarrow WR(x,e)g_t(e)$ and $x\leftarrow d_t(e)$, with $e=x$ and $R=1$ for no trade. At $N$, close to cash and terminate. Never wait beyond the horizon for a pending move to resolve.

For short runs, favorable movement is an increase of $L_{\mathrm{short}}$, so the same maximum-and-drawdown code applies to that curve. In raw prices this corresponds to tracking a trough. With maintenance, do not replace it with the negative of the long curve.

This retains the usable structure of the original outline: **extend a run; remember its extremum; keep a counter-move pending; absorb it on recovery; otherwise emit cash, a reversal or an adjustment when its action branch wins.** The change is that the branch test is now fully computed rather than assumed to be a universal $\kappa$ crossing. The threshold examples below demonstrate why that change is necessary.

```text
tables[N] = terminal_closeout_pieces()
for t = N-1 down to 0:
    H = pull_back_and_clip(tables[t+1], candle[t], maintenance[t], limits)
    U = endpoints(intersect_pieces(H, target_interval))
    buy_records  = suffix_max(H(U) / (1-f+f*U))
    sell_records = prefix_max(H(U) / (1-f*U))
    tables[t] = upper_envelope_with_actions(
        H(x), (1-f+f*x)*buy_records_above(x),
        (1-f*x)*sell_records_below(x))

account = initial_account
scanner = initialize_from(account)
for t = 0 to N-1:
    update_directional_extremum_and_pending(scanner, price[t])
    action = lookup(tables[t], account.exposure)
    update_runs_using_branches_1_to_5(scanner, action, t)
    account = execute_then_maintain_then_mark(account, action, candle[t])
emit_final_close_and_finish(scanner, account, N)
```

There is no backdating during this pass: the hindsight table already incorporates the later path when it selects an earlier action. A live policy must instead form its decision from the information then available.

#### 4. Merge records without deleting their account transition

Merge adjacent cash records. Merge adjacent same-side records with the same follower rule, retaining their internal target-restoring orders, coasting intervals and execution anchors. Do not merge across an emitted reversal or adjustment boundary. Same-side runs separated by a retained cash interval remain separate.

Within a fixed execution branch and maintenance regime, the holdings update is a linear map $\psi' = M\psi$, with $\psi=(q,a)^\top$. A fixed target gives a linear map too: $W'=R(x,e)W$ is linear in $(q,a)$ on its buy or sell branch, and post-trade holdings are $(W'(1-e),W'e/p)$. No trade is the identity before maintenance. Thus a compiled block stores

$$
\boxed{M_{[i,j]}=M_{j-1}\cdots M_i}
$$

and the input-state interval on which every chosen branch and intermediate feasibility check remains valid. Obtain that interval by pulling each branch's exposure inequalities back through the preceding maps; they are linear inequalities in initial holdings. Also retain the internal order program if it must be replayed.

Consequently a block preserves terminal holdings, wealth and feasibility on its recorded domain, not merely cumulative price return. Outside that domain, use the appropriate branch of the decision table instead of reusing the block. This is lossless policy compression for the specified hindsight path; it does not establish that endpoint price and duration alone suffice for an unknown future path.

### Completed optimality argument for the operational construction

**Theorem.** Under the finite-grid, proportional, piecewise-linear account model stated above, the endpoint/record-scan construction maximizes terminal cash over all feasible continuous target exposures and no-trade decisions. The forward follower attains that value, and block compilation preserves it.

**Proof.** Terminal liquidation is the displayed piecewise-affine $v_N$. Suppose $v_{t+1}$ is correctly represented. Pullback through each maintenance/price transition makes $H_t(e)=g_t(e)v_{t+1}(d_t(e))$ affine on finitely many feasible intervals, exactly as in step 2.

Fix a current exposure $x$. On one such interval write $H_t(e)=A+Be$. On its buy portion,

$$
\frac{d}{de}\frac{A+Be}{b+fe}
=\frac{Bb-fA}{(b+fe)^2};
$$

on its sell portion,

$$
\frac{d}{de}\frac{A+Be}{1-fe}
=\frac{B+fA}{(1-fe)^2}.
$$

Each derivative has constant sign or is identically zero. Multiplication by the positive current-state factors $b+fx$ or $1-fx$ does not change that conclusion. Hence the optimum on each piece is at an endpoint, or everywhere on a tied piece. Splitting at $e=x$ adds only the no-trade action. All other endpoints are exactly the finite targets collected by the algorithm. The prefix/suffix scans therefore evaluate the maximum over **every feasible real target**, not just maximum long, cash and maximum short.

The upper envelope records that maximum for every feasible current state, so it is the exact Bellman value $v_t$. Backward induction proves optimality for the whole horizon. Forward execution follows the stored maximizing actions, including the prescribed terminal close. Finally, composition of the linear account maps changes neither the chosen orders nor their result, and the recorded guards preserve feasibility. This proves the compression claim as well. $\square$

This resolves the previous completeness gap by deriving the complete finite action list rather than assuming a three-target list is complete. Additional targets are generated mechanically at funding-regime, continuation-value or feasibility boundaries. They must not be deleted to force an alternating maximum-exposure description.

For example, under Oracle.md's **borrowing** convention, take a single $1\%$ price rise, $f=0.001$, quote debt growth $1.02$, initial cash $1$, and maximum long target $3$. Target $1$ finishes with $1.00798101$ after both execution fees; target $3$ finishes with $0.984015$. Below exposure $1$ the exact entry-to-liquidation payoff increases, and above $1$ it decreases, so the optimum is $1$, not $0$ or $3$. This example concerns borrowed-quote maintenance, not the alternative uniform per-unit funding model. It establishes that a universal maximum-target-only theorem across the account models discussed here would be false.

The exact result is therefore: **compute the account-dependent action boundaries, follow them, and compress the resulting runs.** A running-extremum scan with predetermined fee strips alone is a different algorithm; it is not the proved replacement.

### Coasting anchor and feasibility

For a long with exposure $\lambda\ge1$ last established at adjusted-price anchor $\widetilde p_a$, let $u=\widetilde p_t/\widetilde p_a$. With no discretionary trades afterward,

$$
\boxed{
\frac{W_t}{W_a}=1-\lambda+\lambda u,
\qquad
e_t=\frac{\lambda u}{1-\lambda+\lambda u}.
}
$$

For $\lambda>1$, an adjusted decline raises exposure above $\lambda$; recovery to the anchor restores it automatically. Above the anchor, exposure falls below $\lambda$. An absorbed coasting pullback contains no discretionary trades, so its execution anchor remains unchanged. A counter-move selected for trading becomes a separate planned interval and receives the appropriate new execution anchor.

For a positive effective-exposure ceiling $E_{\mathrm{eff},+}>1$, coasting requires positive wealth and $e_t\le E_{\mathrm{eff},+}$. In this simple model the latter condition is

$$
u\ge
\frac{E_{\mathrm{eff},+}(\lambda-1)}
{\lambda(E_{\mathrm{eff},+}-1)}.
$$

For a short initially at exposure $-\lambda$, let $d\ge0$ be the adverse displacement of its directional curve. With zero maintenance, this means $p_t/p_a=e^d$. With asset-borrow growth $e^H$, use the liability multiplier $(p_t/p_a)e^H=e^d$. Then

$$
\boxed{
\frac{W_t}{W_a}=1+\lambda-\lambda e^d,
\qquad
e_t=-\frac{\lambda e^d}{1+\lambda-\lambda e^d}.
}
$$

For a short effective-exposure magnitude ceiling $E_{\mathrm{eff},-}\ge\lambda$, its adverse log displacement must satisfy

$$
d\le
\log\frac{E_{\mathrm{eff},-}(1+\lambda)}
{\lambda(1+E_{\mathrm{eff},-})}.
$$

The positive symbol $E_{\mathrm{eff},-}$ here denotes the magnitude of the negative short limit, not a signed exposure.

Choose the actual side-specific coasting bands strictly inside these effective-exposure limits, with any additional exchange margin buffer. If the model includes unseen intracandle excursions, endpoint checks alone do not certify safety. Under this invariant, no risk-driven trim occurs inside an accepted coasting block. A block that cannot satisfy it is outside the safe coasting domain and must not be absorbed.

For the fee-only band $d=\kappa$, positive marked equity on both sides is ensured by

$$
\boxed{(\lambda+1)(1-K)<1,\qquad K=(1-f)^2.}
$$

The simpler $2f(\lambda+1)<1$ is sufficient. These are solvency conditions through that band, not substitutes for the effective-exposure checks, and must be recomputed for any wider execution-aware band. If $\lambda>1$ is itself a hard continuous ceiling, no adverse long coasting is permitted. An unleveraged all-asset long, $\lambda=1$, does not have this exposure drift.

For log returns, same-direction composition is exact at the price level:

$$
\log\frac{p_{t+h}}{p_t}
=
\sum_{j=0}^{h-1}\log\frac{p_{t+j+1}}{p_{t+j}}.
$$

Pure turning-point segments can alternate price direction, but the updated exposure regimes need not alternate long and short: cash may separate two runs of the same side. A forecast made in the middle of an unfinished run must also allow same-sign continuation.

### Compressed state

Let $p_a$ be the anchor price at which the current run started, let $p_t$ be the current observed price, and let $d_t$ be its elapsed physical duration in candle count. A minimal candidate state is

$$
z_t=\left(p_a,p_t,d_t,\ell_t,m_t,\eta_t,x_t^{account},\xi_t\right),
$$

where $\ell_t$ is current signed exposure, $m_t\in\{\mathrm{long},\mathrm{flat},\mathrm{short}\}$ is the active exposure regime, and $\eta_t$ records the execution anchor, running adverse/favorable extrema and timestamps, pending counter-moves, and maintenance transformation factors not already represented elsewhere. The run-start anchor $p_a$ need not be the latest high-water execution anchor. The account state $x_t^{account}$ contains the remaining holdings, constraints and lifecycle state, and $\xi_t$ contains market-regime information required to predict the next event. The raw cumulative return is not an independent state variable because

$$
R_t^{run}=\log\frac{p_t}{p_a}.
$$

The raw return direction, away from zero, is derivable as

$$
\sigma_t=\operatorname{sign}R_t^{run}.
$$

This raw direction does not determine the active exposure regime. Maintenance, cash intervals and unresolved pullbacks can distinguish them. The updated algorithm uses hysteresis and therefore retains $m_t$ and the pending-run metadata explicitly.

### Predictive target

Let $\bar p_{t+1}$ denote the endpoint of the next decision-relevant cumulative movement, not necessarily the next raw candle, and let $D_{t+1}$ be the remaining physical time needed to realize it. The compressed prediction target is the signed joint distribution
$$
\boxed{
P_t(dR_{t+1},dD_{t+1}\mid z_t),
\qquad
R_{t+1}=\log\frac{\bar p_{t+1}}{p_t}.
}
$$

The return must remain signed. At the anchor, it describes the next completed compressed movement. In the middle of a run, its support must also allow continuation in the current direction.

Separate probabilities of leaving through the upper and lower significant boundaries do not need to be predicted independently. They are marginals of the endpoint distribution and are true according to whichever endpoint occurs by the definition of the compressed event:

$$
P(R_{t+1}>a_+\mid z_t)
=
\int_{R>a_+}P_t(dR,dD\mid z_t),
$$

$$
P(R_{t+1}<-a_-\mid z_t)
=
\int_{R<-a_-}P_t(dR,dD\mid z_t).
$$

If an immediately significant movement is counted as one raw period, use $D=1$. If $D$ counts only additional waiting depth beyond the first movement, the same case has $D=0$. The longer price stays inside the indecision band, the larger $D$. The convention does not affect the policy as long as duration-dependent costs use it consistently.

### Policy and current-price execution

The predicted endpoint is a forecast target. The current action is always executed at the current price $p_t$:

$$
a_t=\phi(z_t,\bar p_{t+1},D_{t+1}).
$$

The corresponding forecast-pushforward policy is

$$
\pi_\phi(da_t\mid\mathcal F_t)
=
\int
P_t(d\bar p,dD\mid z_t)
\delta_{\phi(z_t,\bar p,D)}(da_t).
$$

The Bellman-optimal deterministic alternative uses the same event distribution inside its action value and selects one current action:

$$
a_t^*
=
\arg\max_a
\int
\left[
G(z_t,a,\bar p,D)
+\bar\beta V^*(T(z_t,a,\bar p,D))
\right]
P_t(d\bar p,dD\mid z_t).
$$

As established above, the randomized pushforward is optimal for the ordinary expected-utility Bellman objective only when all actions in its support maximize the same current action value. The event compression itself is compatible with either policy construction; it changes the prediction kernel, not that optimality condition.

### Event-time Bellman equation

Use the compressed event index as the Bellman clock. Every transition advances this clock by exactly one event, so the Bellman duration is always one:

$$
D^{Bellman}=1.
$$

Assuming event frequency is proportional to physical time, accumulated discounting can be represented by one constant event discount $\bar\beta$. The action value is

$$
\boxed{
\bar Q(z,a)
=
\int
\left[
G(z,a,\bar p,D)
+\bar\beta\bar V(T(z,a,\bar p,D))
\right]
P(d\bar p,dD\mid z).
}
$$

When using raw prices, physical duration affects the actual maintenance and funding transition. For a log-wealth objective, record the net account multiplier, including internal execution:

$$
G=\log\frac{W_{\mathrm{event\ end}}}{W_{\mathrm{event\ start}}}.
$$

Since adjusted-price transition already absorbs maintenance, we do not charge that cost again in $G$. Duration can be omitted from that particular payoff only when no remaining funding-regime, (\*) execution, (\*) feasibility or predictive-state dependency requires it.

Using a constant $\bar\beta$ is exact for an objective defined in event time. It is an approximation to a physical-time discounted objective when event duration is random, unless the physical discount accumulated over an event is also included in $G$ or the transition kernel. The event-time convention is adopted here.

### Causality and online use

Using a future compressed endpoint as a training label is not lookahead leakage by itself. The causal online procedure is:

1. Construct $z_t$ only from information available at time $t$.
2. Predict $P_t(\bar p,D\mid z_t)$.
3. Select and execute the action at current price $p_t$.
4. Update the current run as new raw movements arrive.
5. Recompute the compressed distribution and policy from the new state.

Leakage occurs only if a future-defined endpoint is included in the model input, if a backtest treats the predicted endpoint as known, or if it executes retroactively at a turning point that could only be confirmed later.

When the current movement continues in the same direction, the next prediction may also have that sign. When direction changes, the new state exposes exactly the new opportunity the compression is intended to represent.

### When endpoint and duration are sufficient

Let $C(h_t)=z_t$ map a raw history to its compressed state. The compression is Bellman-exact if any two raw histories with the same $z_t$ have, for every admissible action:

1. the same feasible action set;
2. the same conditional distribution of compressed reward;
3. the same conditional distribution of $(\bar p,D)$ and the next compressed state;
4. the same liquidation, collateral and lifecycle consequences;
5. no omitted intermediate decision capable of improving the value.

Equivalently, the joint law $\mathcal L\left(G,\bar p,D,z_{t+1}\mid h_t,a_t\right)$ must depend on the raw history only through $z_t$. Then there exists a compressed value function such that

$$
V^*(h_t)=\bar V^*(C(h_t)),
$$

and the full raw next-price distribution can be discarded after constructing the event kernel.

For the updated run construction, the intended internal actions are favorable-extension rebalancing, feasible coasting through absorbed counter-moves, or remaining in cash. These actions must be represented by the compressed transition rather than treated as a constant untraded position. Direction changes and flat blocks are candidate boundaries; execution costs can make retaining them or merging them preferable. Constraint events can require additional boundaries or internal adjustments.

The recovered-pullback calculations below compare selected alternatives at matching exposure states. They do not establish that every intermediate action or every nested sequence is dominated. That is the outstanding no-omitted-decision proof obligation.
For the proposed exit-policy outline, the no-omitted-decision condition has the following intended justification:

1. Inside the indecision band, no different useful action is available by definition of the band and minimum order restriction.
2. During favorable same-sign continuation, the policy continues to hold.
3. After a full close, additional close actions leave exposure at zero and are idempotent.
4. A direction change is a new decision opportunity and therefore becomes an event boundary.
5. During adverse same-sign continuation, the only remaining nontrivial case is repeated fractional/full closure.

Under these claims, intermediate observations do not create a distinct option that needs the full raw path distribution. The first four cases compress directly. The fifth requires additional treatment.

The definition of the indecision band cannot by itself prove that no useful action exists inside it. That property must follow from the raw Bellman problem, the minimum-order constraint, or a verified dominance bound. Otherwise defining the compression from the desired policy and then using it to prove that policy optimal would be circular.

### Reoptimization does not preserve an earlier point forecast

Bellman optimality is a closed-loop property, not a commitment to the action implied by an earlier sampled or point forecast. At event time $t$, the policy chooses

$$
a_t^*\in\arg\max_a\bar Q^*(z_t,a).
$$

After observing new information and reaching state $z_{t+1}$, it chooses

$$
a_{t+1}^*\in\arg\max_a\bar Q^*(z_{t+1},a).
$$

There is no requirement that $a_{t+1}^*$ equal the action that the earlier point forecast would have prescribed for that future time. Dynamic consistency means that each action maximizes the continuation value conditional on the information then available. It does not mean that forecasts or actions remain unchanged as the filtration grows.

This distinction is harmless for compression whenever all omitted intermediate states have the same optimal operation: hold, remain fully closed, or continue the same macro-policy. If an intermediate state changes the Bellman-optimal target exposure, it must remain an event boundary or be handled inside the macro-policy.

### Fractional closure and composition

CRRA utility guarantees concavity and scale invariance, but it does not make fractions compose automatically across time. Consider two consecutive returns $\rho_1$ and $\rho_2$. Reoptimizing or rebalancing a fractional allocation between them gives a multiplier of the form

$$
M_{sequential}
=(1+w_1\rho_1)(1+w_2\rho_2).
$$

Collapsing both returns and applying one initial fraction gives

$$
M_{collapsed}
=1+w\left[(1+\rho_1)(1+\rho_2)-1\right].
$$

They agree automatically at the bang-bang actions $w=0$ and $w=1$, but not at a general interior CRRA fraction.

For an exit-only remaining-position rule

$$
q_{new}=g(R,D)q_{old},
\qquad g=1-s,
$$

two sequential reductions produce

$$
q_2
=g(R_2,D_2)g(R_1,D_1)q_0.
$$

One compressed reduction produces the same remaining size only if

$$
\boxed{
g(R_1+R_2,D_1+D_2)
=g(R_2,D_2)g(R_1,D_1).
}
$$

Equality of remaining size is still insufficient if the sequential reductions incur different execution costs or change wealth at different prices. Exact compression must preserve the complete macro reward and next account state.

This semigroup equation is not a separate requirement for Bellman optimality when every state at which the optimal action changes remains an event boundary. Bellman recursion then reoptimizes normally at each event. It is required only when several internal partial-close decisions are to be replaced by one initial compressed action.

### Minimal predictive object

Under the exit-policy dominance assumptions and with fractional decisions handled by one of the methods above, the full raw path distribution can be replaced by

$$
\boxed{
P_t(\bar p_{t+1},D_{t+1}\mid z_t).
}
$$

This remains a distribution, not a single point prediction. Concave utility and leverage sizing depend on the probability assigned to favorable and adverse endpoints as well as their duration.

The kernel is sufficient when reward and next compressed state are deterministic functions of

$$
(z_t,a_t,\bar p_{t+1},D_{t+1}).
$$

If intra-event liquidation, drawdown, order fills or partial-execution costs are not recoverable from those variables, augment the kernel only with the missing policy-relevant summaries. There is no need to retain the raw sequence once these summaries form a controlled sufficient statistic.

This provides a finite and decision-oriented prediction target: predict the next significant signed cumulative movement and the time required to realize it, then let the event-time Bellman recursion represent all later opportunities.

## Compression thresholds

This section records closed-form local comparisons and explains the fee strips used to interpret compressed runs. They are not required to complete the exact finite-grid algorithm above: its action table supplies the decision even when no scalar price threshold describes it. For multiplicative asset-unit maintenance $m_i$ per period, the long-side adjusted price is

$$
\widetilde p_t=p_t\prod_{i<t}(1-m_i),
\qquad
L_{\mathrm{long},t}=\log p_t+\sum_{i<t}\log(1-m_i).
$$

The continuous-rate version is $L_{\mathrm{long}}=\log p-\int c\,dt$. A maintenance-compensating raw price change therefore becomes zero adjusted movement. Use the appropriate separate short-side coordinate rather than negating the long curve when maintenance is positive.

Under the fee convention in Oracle.md, selling a unit receives $(1-f)p$ and buying a unit costs $p/(1-f)$. Define

$$
\boxed{
K=(1-f)^2,\qquad \kappa=-\log K=-2\log(1-f).
}
$$

With no other costs, a unit bought at $p$ and later sold at the known hindsight exit price $p_T$ is profitable when $p<Kp_T$. A unit sold short at $p$ and later bought back at $p_T$ is profitable when $p>p_T/K$. Thus the favorable log move must exceed $\kappa$ for this isolated incremental round trip. A valid cost-removing transform can express this in adjusted prices without another maintenance deduction.

Relative to a long execution anchor $L_a$, the fee-only strip $-\kappa<L-L_a<0$ describes a candidate adverse excursion. It is not a complete long/cash/short decision rule. The associated adjusted-price distance is

$$
a_{\mathrm{fee}}=\widetilde p_a(1-K),
$$

which does not depend on minimum order size: proportional proceeds and costs scale together. Order-size constraints, if introduced, require separate feasibility checks.

The leverage adequacy condition $(\lambda+1)(1-K)<1$ and its sufficient bound $2f(\lambda+1)<1$ ensure positive marked equity through the fee-only coasting band under the models described above. They are not an exact formula for the loss on a maximum-exposure round trip, and they do not determine the exchange's effective-exposure ceiling.

The following comparison scopes determine which threshold to use:

| Threshold                   | Alternatives being compared                                                                  | Scope                                                                              |
| --------------------------- | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| $\kappa$                    | Add an incremental position now versus omit that isolated round trip.                        | Known closing price and the stated fee-only or equivalent transformed model.       |
| $\kappa_{\mathrm{maint}}$   | Restore maximum long exposure throughout an adverse leg versus close/reopen through cash.    | Compare at that leg's end, both at $+\lambda$.                                     |
| $\kappa_{\mathrm{coast}}$   | Coast through a recovered long pullback versus close at its anchor and reopen at its trough. | Compare at recovery, both at $+\lambda$, with the specified recovery follower.     |
| $\kappa_{\mathrm{reverse}}$ | Coast versus a padded short reversal followed by long recovery.                              | Zero maintenance, symmetric targets, the specified fee convention, and $d>\kappa$. |

These are branch comparisons, not four interchangeable constants defining a universal band. Outside a formula's scope, it is not a switching rule; the exact finite-grid action table handles the decision within its stated model. Constant slippage may be included in $f$ as a model assumption; variable slippage, impact or fixed charges lie outside that theorem's scope.

### Fee-only padding and elementary holding boundaries

For multiplicative asset-unit maintenance $m$ per period over $h$ periods, the raw-price rise that offsets maintenance on the held unit is

$$
a_{\mathrm{hold}}=p_t\left((1-m)^{-h}-1\right).
$$

For a candidate flat block $[a,b]$ between two retained runs of the same direction, its accumulated adverse directional displacement is

$$
A_+=L_{\mathrm{long}}(a)-L_{\mathrm{long}}(b)
=\sum_i(h_i-r_i),
\qquad
A_-=L_{\mathrm{short}}(a)-L_{\mathrm{short}}(b)
=\sum_i(h_i+r_i).
$$

These are simply cumulative adverse returns in adjusted coordinates, not additional maintenance costs. Elementary classification guarantees they are nonnegative, not that they exceed a switching threshold. Once a same-side gap has been retained because it passed the applicable bridge test, repeating that test is redundant.

Only in the additive directional-log surrogate with constant round-trip switching cost $\kappa$ per unit of exposure does the test reduce to bridge if $A_\pm<\kappa$, separate if $A_\pm>\kappa$, and tie at equality. The leveraged execution model uses the corresponding candidate-value comparison below. The surrounding runs must themselves be retained for this same-side comparison to apply.

### Recovered long-pullback benchmark

The next formulas use a specific common comparison scenario:

1. At an adjusted-price anchor normalized to $1$, equity is $W_a$ and long exposure is at $\lambda\ge1$. Initial entry fees are already paid.
2. Adjusted price declines monotonically to $u=e^{-d}<1$, then recovers monotonically to $1$. Here $d>0$ is log drawdown, not the physical duration $D$ used in the prediction kernel.
3. Compare candidates at recovery with exposure restored to $+\lambda$, so subsequent identical policies have comparable account states. Recovery is a comparison point, not necessarily the final end of the upward run.
4. Fees are constant and proportional, orders have no minimum/maximum size or fixed charge, and specified rebalancing is continuous along a smooth price path. Require $f\lambda<1$, positive wealth and valid execution multipliers.
5. For long/cash comparisons, maintenance is zero or removed by the long-side transform that preserves the same fee and quote-account model. Coasting must respect the effective-exposure and liquidation limits. The reversal extension below additionally assumes zero maintenance and symmetric target bounds.

Finite-candle execution must instead compose the exact per-step account transitions. Its numbers need not equal the continuous-rebalancing powers below.

Define the maintained-long growth coefficients and close/reopen fee multiplier:

$$
\boxed{
\alpha_\uparrow=\frac{\lambda}{1-f+f\lambda},
\qquad
\alpha_\downarrow=\frac{\lambda(1-f)}{1-f\lambda},
\qquad
B_\lambda=\frac{(1-f\lambda)(1-f)}{1-f+f\lambda}.
}
$$

Maintaining $A=\lambda W$ while buying on a favorable extension or selling on an adverse movement, and including the respective fee flows, gives $d\log W=\alpha_\uparrow\,d\log\widetilde p$ on an increase and $d\log W=\alpha_\downarrow\,d\log\widetilde p$ on a decrease. The product of the fee-aware $\lambda\to0$ and $0\to\lambda$ transitions gives $B_\lambda$.

At adjusted-price recovery the three candidate wealth multipliers are

$$
\boxed{
\begin{aligned}
G_{\mathrm{coast}}&=1,\\
G_{\mathrm{maintain}}&=
u^{\alpha_\downarrow-\alpha_\uparrow}
=e^{-(\alpha_\downarrow-\alpha_\uparrow)d},\\
G_{\mathrm{cash}}&=
B_\lambda u^{-\alpha_\uparrow}
=B_\lambda e^{\alpha_\uparrow d}.
\end{aligned}
}
$$

Coasting makes no trades during either leg. Maintaining restores $\lambda$ continuously during both legs. The cash candidate closes at the anchor, reopens to $\lambda$ at the trough, and maintains $\lambda$ during recovery.

For $f>0$ and $\lambda>1$, $\alpha_\downarrow>\alpha_\uparrow$, so feasible coasting beats continuously restoring the target through this fully recovered pullback. At $\lambda=1$, coasting and maintaining coincide.

The cash-versus-coasting threshold is

$$
\boxed{
\kappa_{\mathrm{coast}}(\lambda)
 =-\frac{\log B_\lambda}{\alpha_\uparrow}.
}
$$

Among those two candidates, coast for $d<\kappa_{\mathrm{coast}}$, close/reopen for $d>\kappa_{\mathrm{coast}}$, and tie at equality. This does not yet compare against shorting the counter-move.

For completeness, comparing continuously maintained exposure against close/reopen at the *end of the adverse leg*, with both candidates then at exposure $\lambda$, gives a different threshold:

$$
\boxed{
\kappa_{\mathrm{maint}}(\lambda)
 =-\frac{\log B_\lambda}{\alpha_\downarrow}.
}
$$

The maintained candidate there has multiplier $e^{-\alpha_\downarrow d}$ and the cash candidate has multiplier $B_\lambda$. A coasting candidate has a different exposure at the trough, so its equity cannot be inserted into this comparison without evaluating continuation or the trades needed to match states. For an unleveraged long, both thresholds reduce to $\kappa$.

### Padded reversal versus coasting

For the same recovered price path with *zero maintenance*, symmetric target exposures $\pm\lambda$, and the same fee convention, define

$$
\boxed{
\alpha_s=\frac{\lambda(1-f)}{1+f\lambda}.
}
$$

This is the favorable-decline growth coefficient of a continuously maintained short. It is not obtained by merely negating the maintained-long coefficient.

Consider this additional candidate for $d>\kappa$:

1. Reverse from $+\lambda$ to $-\lambda$ at the anchor.
2. Maintain the maximum short until price reaches $u/K$. At that point the remaining decline to $u$ is exactly the fee-only round-trip width.
3. Stop adding short exposure and coast through the last $\kappa$ of the decline.
4. Reverse to $+\lambda$ at the trough and maintain it during recovery.

Integrating the first short leg, applying the passive final decline and fee-aware reversal, and then applying the long recovery gives

$$
\boxed{
G_{\mathrm{reverse}}
=B_\lambda
\exp\left[(\alpha_\uparrow+\alpha_s)d-\alpha_s\kappa\right],
\qquad d>\kappa.
}
$$

Consequently, the crossover against coasting for this candidate is

$$
\boxed{
\kappa_{\mathrm{reverse}}(\lambda)
=\frac{-\log B_\lambda+\alpha_s\kappa}
{\alpha_\uparrow+\alpha_s}.
}
$$

Evaluate the candidate values under their feasibility conditions; do not extrapolate the padded formula to $d\le\kappa$, where its prescribed stopping point would precede the start of the decline. For $d>\kappa$, this reversal candidate beats the cash candidate by the factor $e^{\alpha_s(d-\kappa)}$. Positive maintenance, side-specific borrowing, asymmetric leverage bounds, finite-step execution, or a different terminal state require recomputing the comparison; the zero-maintenance reversal formula does not become general by inserting the long-side adjusted price into both legs.

These are comparisons among specified policies. They do not prove that partial adjustments or other schedules cannot improve on all of them.

### Numerical check and threshold selection

For $f=0.01$, $\lambda=10$:

$$
\begin{aligned}
\kappa&\approx0.020100672,\\
\kappa_{\mathrm{maint}}&\approx0.018326232,\\
\kappa_{\mathrm{coast}}&\approx0.021973152,\\
\kappa_{\mathrm{reverse}}&\approx0.021045891.
\end{aligned}
$$

For the zero-maintenance benchmark, starting with equity $100$ and comparing at recovered price $1$, all candidates below end at exposure $+10$:

| Log drawdown $d$ |        Coast | Close/reopen through cash | Padded short reversal |
| ---------------: | -----------: | ------------------------: | --------------------: |
|         $0.0210$ | $100.000000$ |               $99.111174$ |           $99.916631$ |
|         $0.0215$ | $100.000000$ |               $99.566857$ |          $100.828726$ |

An effective-exposure ceiling of $15$ permits coasting in both examples; peak effective exposures are approximately $12.361$ and $12.432$. Initial entry fees are common sunk costs and are not charged again.

The fee-only $\kappa$ would preserve both declines as reversal opportunities and lose against coasting in the first row. Using only $\kappa_{\mathrm{coast}}$ would favor coasting in both rows and miss the profitable short reversal in the second. Thus a single anchored strip $-\kappa<L-L_{\mathrm{anchor}}<0$ cannot substitute for all the branch comparisons.

For a valid long-side transform, $d=H_{a:m}-\log(p_m/p_a)$ from anchor $a$ to trough $m$. An applicable long-side boundary $d=\kappa_{\mathrm{case}}$ translates to

$$
p_m=p_a\exp(H_{a:m}-\kappa_{\mathrm{case}}).
$$

There is no extra maintenance deduction after this conversion. The raw-price displacement may even be positive when maintenance outweighs raw appreciation. Duration can disappear from a particular comparison only when all relevant costs and state dependencies have already been absorbed.

### Empirical study
The following stochastic one-period fraction thresholds are separate from the deterministic run-merge thresholds above. They do not prove the optimality of hindsight compression.

For the one-second policy, use the full empirical histogram as the residual-return law rather than extrapolating the fitted generalized-$t$ tails. This makes the thresholds finite and computes them from the complete distribution instead of replacing the next return by a point prediction.

Let histogram bin $i$ have probability $q_i$ and midpoint $b_i$ in basis points. Centre the histogram and convert it to log-return units:

$$
x_i=10^{-4}(b_i-\bar b),
\qquad
\bar b=\sum_iq_ib_i,
\qquad
\sum_iq_i=1.
$$

For the stored full-history BTCUSDT 1s histogram,

$$
\bar b=0.000043202202384\text{ bp},
$$

and the empirical law contains $1{,}353$ populated bins over $157{,}766{,}399$ observations. Treating the bin-midpoint distribution as the model, every expectation below is the finite sum over those bins. The computation is therefore exact for the stored binned law; it does not use a delta-distribution or small-variance approximation.

Write the predicted next log return as

$$
R=\mu+X-H_1,
$$

where $P(X=x_i)=q_i$, $\mu$ is the predicted location of the distribution, and $H_1$ is one second of holding or funding cost in log-return units. For close fraction $s$, expected log wealth is

$$
J(s;\mu)
=\sum_iq_i\log\left((1-s)e^{\mu+x_i-H_1}+sK\right).
$$

For any fixed $0<s<1$, define $d_s$ as the unique solution of

$$
\boxed{
\sum_i
\frac{q_i}
{s+(1-s)e^{d_s+x_i}}
=1.
}
$$

The predicted-location threshold at which the unconstrained optimum has close fraction $s$ is

$$
\boxed{
\mu_s=H_1+\log K+d_s.
}
$$

The two endpoint constants are also finite:

$$
\boxed{
d_0=\log\sum_iq_ie^{-x_i},
\qquad
d_1=-\log\sum_iq_ie^{x_i}.
}
$$

Here $d_0$ is the full-hold boundary and $d_1$ is the full-close boundary. The exact 1s values are:

| Close fraction $s$ |      $d_s$ in log bp |
| -----------------: | -------------------: |
|                $0$ | $+0.000037600003040$ |
|           $0.0005$ | $+0.000037562410959$ |
|            $0.001$ | $+0.000037524796603$ |
|             $0.01$ | $+0.000036848022411$ |
|             $0.10$ | $+0.000030080109514$ |
|             $0.25$ | $+0.000018800222490$ |
|             $0.50$ | $+0.000000000181817$ |
|             $0.75$ | $-0.000018800124790$ |
|             $0.90$ | $-0.000030080420377$ |
|             $0.99$ | $-0.000036848661900$ |
|            $0.999$ | $-0.000037525906826$ |
|                $1$ | $-0.000037600740228$ |

As the forecast location falls, the optimal close fraction increases. Thus the four one-period regions, before adding continuation value, are separated by $\mu_0$, $\mu_{s_{min}}$, and $\mu_{s_{max}}$:

$$
\begin{cases}
\mu\ge\mu_0, & s^*=0,\\
\mu_{s_{min}}<\mu<\mu_0, & 0<s^*<s_{min}\text{ would be desired},\\
\mu_{s_{max}}<\mu\le\mu_{s_{min}}, & s_{min}\le s^*<s_{max},\\
\mu\le\mu_{s_{max}}, & s^*\ge s_{max}\text{ would be desired}.
\end{cases}
$$

Because an exchange does not permit the second region's unconstrained action, the actual one-period switch between holding and executing the minimum order is obtained by comparing their values, not by using a derivative:

$$
\boxed{
\sum_iq_i
\log\left((1-s_{min})+s_{min}e^{-d_{switch}-x_i}\right)
=0.
}
$$

For $s_{min}=0.0005$,

$$
d_{switch}=0.000037581204815\text{ bp}.
$$

Above $H_1+\log K+d_{switch}$, holding has greater one-period expected log utility; below it, executing at least the minimum order has greater one-period expected log utility. In the Bellman problem, the continuation value replaces this terminal comparison and can move the switch, but the same finite-sum construction applies.

To express a location threshold as the document's price-distance constant, use

$$
\boxed{
a_t(s)=p_t\left(1-e^{\mu_s}\right).
}
$$

This is a constant fraction of current price, not a constant number of quote-currency units. A constant one-second holding cost $H_1$ simply shifts every log threshold by the same amount.

For the current example of a $10{,}000$ quote position, the $5$ minimum order gives $s_{min}=0.0005$. If the per-side friction includes the configured $7.5$ bp fee and $10$ bp modeled slippage, then

$$
f=0.00175,
\qquad
K=(1-f)^2=0.9965030625,
\qquad
\log K=-35.030660776127\text{ bp}.
$$

With $H_1=0$, the calibrated boundaries are:

| Boundary                                       | Forecast location in log bp | Price decrease in bp of $p_t$ |
| ---------------------------------------------- | --------------------------: | ----------------------------: |
| full hold, $s=0$                               |          $-35.030623176124$ |             $34.969337531482$ |
| hold versus minimum-order switch               |          $-35.030623194922$ |             $34.969337550215$ |
| unconstrained optimum reaches $s_{min}=0.0005$ |          $-35.030623213716$ |             $34.969337568942$ |
| $s=0.10$                                       |          $-35.030630696017$ |             $34.969345025079$ |
| $s=0.50$                                       |          $-35.030660775945$ |             $34.969374999819$ |
| $s=0.90$                                       |          $-35.030690856547$ |             $34.969404975231$ |
| full close, $s=1$                              |          $-35.030698376867$ |             $34.969412469252$ |

Therefore, under this one-second unconditional law and cost example, the uncertainty-created fractional band is extremely narrow relative to transaction friction. This is a result of the exact finite-sum calculation: the return dispersion affects log-utility thresholds at second order in the very small one-second log-return scale.

## Complete portfolio-management policy

The complete policy has three logically separate layers:

1. **Prediction:** transform raw market histories into compressed decision states and predict the next joint decision-relevant event distribution.
2. **Account allocation:** choose one jointly feasible vector of signed target exposures by maximizing the account-level Bellman action value.
3. **Position and execution accounting:** convert the change in global exposure into lifecycle-position entries and exits, net it into executable orders, and reconcile fills back into the account state.

The layers may be implemented by separate models, but they form one policy. Local positions do not independently create physical capital or orders; the account-level optimizer owns global feasibility and expected utility.

### Complete state

Let there be assets $i=1,\ldots,m$. At portfolio decision event $n$, define

$$
X_n
=
\left(
W_n,
b_n,
d_n,
o_n,
\mathcal P_n,
z_{1,n},\ldots,z_{m,n},
\xi_n
\right),
$$

where:

- $W_n$ is account equity in the chosen numeraire;
- $b_n$ contains available asset and quote balances;
- $d_n$ contains quote debt, asset borrow and other liabilities;
- $o_n$ is the set of resting and partially filled orders;
- $\mathcal P_n$ is the set of lifecycle positions;
- $z_{i,n}$ is the compressed event state of asset $i$;
- $\xi_n$ contains cross-asset and market-regime information needed by the predictor.

For position $j$ in asset $i$, store at least

$$
P_{i,j,n}
=
\left(
\sigma_{i,j},
q_{i,j,n},
p_{i,j}^{entry},
t_{i,j}^{entry},
c_{i,j,n},
\lambda_{i,j,n}
\right),
$$

where $\sigma_{i,j}\in\{-1,1\}$ is its side, $q_{i,j,n}\ge0$ is remaining size, $c_{i,j,n}$ contains attributed costs and funding, and $\lambda_{i,j,n}$ is lifecycle state such as entering, active or exiting.

The physical net exposure to asset $i$ is

$$
x_{i,n}
=
\sum_j\sigma_{i,j}q_{i,j,n},
$$

and its signed account leverage is

$$
\ell_{i,n}
=
\frac{p_{i,n}x_{i,n}}{W_n}.
$$

Gross exposure, borrow and liquidation risk cannot in general be recovered from net exposure alone when simultaneous long and short virtual positions are retained. Those quantities must remain separately available in $X_n$.

### Portfolio event clock

Each asset can complete compressed movements at different physical times. The portfolio event clock advances whenever any event can change the optimal account action, including:

1. a new decision-relevant price event in any asset;
2. a material update of an event distribution;
3. an order fill, cancellation or rejection;
4. a funding, interest, collateral or liquidation-state change;
5. an external risk-limit change.

At a portfolio event, all asset states are marked to the same current physical time. Assets without a new completed movement retain their current incomplete-run state and update their conditional endpoint distribution from the information then available.

### Joint compressed forecast

For each asset, the basic predictive object is

$$
P_{i,n}(d\bar p_i,dD_i\mid X_n).
$$

Account utility depends on simultaneous outcomes, so marginal forecasts are not sufficient unless conditional independence has been justified. The allocator needs a joint scenario kernel

$$
\boxed{
\mathcal P_n
\left(
d\bar{\mathbf p},
d\mathbf D,
d\eta
\mid X_n
\right),
}
$$

where $\bar{\mathbf p}=(\bar p_1,\ldots,\bar p_m)$, $\mathbf D=(D_1,\ldots,D_m)$ and $\eta$ contains any additional policy-relevant joint event variables. This kernel must preserve cross-asset dependence, because diversification, joint drawdown, collateral use and liquidation depend on correlations.

It can be implemented directly as a multivariate model or indirectly with a common latent market factor and conditionally independent asset-specific event models. Sampling each marginal independently is valid only if the resulting loss of dependence is intentional.

### The account-level Bellman action

Use signed target exposure

$$
\boldsymbol\ell_n'
=
(\ell_{1,n}',\ldots,\ell_{m,n}')
$$

as the main economic action. Resting-order targets or execution instructions may be included in an additional action component $u_n$. The full action is

$$
a_n=(\boldsymbol\ell_n',u_n).
$$

Let $\mathcal A(X_n)$ be the jointly feasible action set after accounting for:

- available balances and collateral;
- long and short leverage limits;
- quote and asset borrow limits;
- scenario-dependent maintenance margin and liquidation;
- minimum and maximum order sizes;
- aggregate execution limits;
- resting-order commitments;
- portfolio risk and drawdown constraints.

The optimal account action is

$$
\boxed{
a_n^*
=
\arg\max_{a\in\mathcal A(X_n)}
\int
\left[
G(X_n,a,y)
+\bar\beta V^*(T(X_n,a,y))
\right]
\mathcal P_n(dy\mid X_n),
}
$$

where $y=(\bar{\mathbf p},\mathbf D,\eta)$ is a joint composite-event scenario.

For terminal expected utility this reduces to

$$
a_n^*
=
\arg\max_{a\in\mathcal A(X_n)}
\mathbb E_n
\left[
U_\gamma(W_{n+1}(X_n,a,Y_{n+1}))
\right].
$$

For repeated expected log wealth, the account reward can be the event log multiplier

$$
G_n
=
\log\frac{W_{n+1}}{W_n},
$$

provided every fee, funding payment and liability change is included in $W_{n+1}$.

The optimization is account-level rather than a sum of independently optimized position ROIs. A position can have attractive standalone expected return while increasing account risk because it is correlated with existing exposure or consumes scarce collateral. Local objectives separate exactly only under the additive-value and separable-feasibility conditions established earlier, or conditionally after introducing correct global shadow prices.

### Approximate one-event exposure objective

For a simple scenario return vector $\boldsymbol\rho$ and current-to-target turnover

$$
\Delta\boldsymbol\ell
=
\boldsymbol\ell'-\boldsymbol\ell_n,
$$

a useful one-event account multiplier is

$$
M
\left(
\boldsymbol\ell';
\boldsymbol\ell_n,
\boldsymbol\rho,
\mathbf D
\right)
=
1
+\boldsymbol\ell'^{\mathsf T}\boldsymbol\rho
-B(\boldsymbol\ell',\mathbf D)
-C(\Delta\boldsymbol\ell),
$$

where $B$ contains funding and borrowing and $C$ contains fees, spread and impact. The corresponding CRRA allocation problem is

$$
\boldsymbol\ell_n^*
=
\arg\max_{\boldsymbol\ell'\in\mathcal L(X_n)}
\mathbb E_n
\left[
U_\gamma
\left(
M(\boldsymbol\ell';\boldsymbol\ell_n,\boldsymbol\rho,\mathbf D)
\right)
\right].
$$

This is an exact Bellman action only under the terminal or recursive homothetic conditions described above. Otherwise it is a one-event approximation to the full continuation action value.

The account multiplier must remain positive on every scenario assigned relevant probability:

$$
M>0.
$$

In practice the exchange liquidation constraint is normally stronger than this mathematical log-utility domain restriction.

### Classifying target-exposure changes

For each asset, compare current exposure $\ell_i$ with the Bellman target $\ell_i^*$. The economic operation is determined entirely by this transition:

| Current exposure |     Target exposure | Operation                         |
| ---------------: | ------------------: | --------------------------------- |
|              $0$ |        $\ell_i^*>0$ | enter long                        |
|              $0$ |        $\ell_i^*<0$ | enter short                       |
|       $\ell_i>0$ |   $\ell_i^*>\ell_i$ | add long exposure                 |
|       $\ell_i>0$ | $0<\ell_i^*<\ell_i$ | partially exit long               |
|       $\ell_i>0$ |                 $0$ | fully exit long                   |
|       $\ell_i>0$ |        $\ell_i^*<0$ | fully exit long, then enter short |
|       $\ell_i<0$ |   $\ell_i^*<\ell_i$ | add short exposure                |
|       $\ell_i<0$ | $\ell_i<\ell_i^*<0$ | partially exit short              |
|       $\ell_i<0$ |                 $0$ | fully exit short                  |
|       $\ell_i<0$ |        $\ell_i^*>0$ | fully exit short, then enter long |
|              any |   $\ell_i^*=\ell_i$ | hold                              |

For an exit-only lifecycle position with current exposure $\ell_{i,j}$, restrict its target to the interval between its current value and zero:

$$
\ell_{i,j}'=(1-s_{i,j})\ell_{i,j},
\qquad
0\le s_{i,j}\le1.
$$

The close fraction is

$$
s_{i,j}
=
1-\frac{\ell_{i,j}'}{\ell_{i,j}}.
$$

The long-exit, short-entry and short-exit policies may be reconstructed from a canonical long-entry problem only when the corresponding full Bellman symmetries have been proved. Otherwise all operations still use the same target-exposure formulation but are evaluated with their actual direction-specific costs, constraints and continuation states.

### Entry decisions

When current exposure is zero, the Bellman target jointly decides:

1. whether the expected utility improvement is sufficient to enter;
2. whether the sign should be long or short;
3. how much account risk and leverage to allocate;
4. whether to enter immediately or through resting orders;
5. which existing opportunities should lose capital or risk budget to make the entry feasible.

For one asset with net return $\rho$ and no state coupling, the interior CRRA condition has the form

$$
\mathbb E_n
\left[
M(\ell,\rho)^{-\gamma}
\left(
\rho-B'(\ell)-C'(\ell)
\right)
\right]
=0.
$$

For log utility without costs,

$$
\mathbb E_n
\left[
\frac{\rho}{1+\ell^*\rho}
\right]
=0.
$$

Entry remains at zero when neither directional marginal value covers transaction, funding and opportunity costs. Fixed fees, spread and minimum orders therefore create a no-entry region even when expected return is nonzero.

An entry creates a new lifecycle position. Existing lifecycle positions are not increased; if the global target increases exposure on the same side, the increment is represented by a new position. This preserves the rule that every position can only be held or reduced after creation.

### Exit decisions

For a currently active position, the unrestricted account optimizer first determines the desired aggregate target exposure. The induced reduction on a side with current global exposure $x_i\ne0$ is

$$
c_i
=
\max\left(
0,
|x_i|-|x_i^*|
\right)
$$

when the target remains on the same side. Its global close fraction is

$$
s_i
=
\frac{c_i}{|x_i|}.
$$

The event-policy regions describe how this target is reached:

1. **Hold region:** $s_i=0$.
2. **Indecision/minimum-order region:** compare waiting against the smallest executable reduction using the continuation action value.
3. **Interior region:** use the Bellman- or CRRA-optimal fractional target.
4. **Maximum-close region:** reduce by the largest currently feasible amount.

For a genuine per-event maximum size, execute the maximum and reoptimize at the next event if the desired reduction remains larger. If the maximum applies only per submitted order and several orders may execute immediately without changing cost or impact, splitting the order removes it as an economic constraint.

For a minimum order, the feasible adjustment is disconnected:

$$
\Delta q\in\{0\}\cup[q_{\min},q_{\max}].
$$

If the unconstrained desired reduction is below $q_{\min}$, do not mechanically round it upward. Compare the Bellman value of waiting with the value of executing $q_{\min}$.

### From global exposure changes to lifecycle positions

Convert target leverage to target asset units using current equity and price:

$$
x_i^*
=
\frac{\ell_i^*W_n}{p_{i,n}}.
$$

Define

$$
\Delta x_i=x_i^*-x_i.
$$

Apply the constructive decomposition:

1. **Increase on the same side.** Create a new position with size $|\Delta x_i|$ and the current target sign. Attach its entry order and cost attribution.
2. **Reduction without crossing zero.** Allocate the reduction across positions on that side. Under pro-rata reduction,

   $$
   q_{i,j}'=(1-s_i)q_{i,j}.
   $$

   FIFO, LIFO or value-based matching may be used instead, but only pro-rata makes every position follow the same local fractional policy.
3. **Crossing zero.** Close every position on the old side, reconcile their realized PnL and liabilities, then create a new position for the residual exposure on the opposite side.
4. **No change.** Keep every active position unchanged apart from accumulated funding, maintenance and lifecycle metadata.

This decomposition is exact accounting:

$$
x_i^*=\sum_j\sigma_{i,j}q_{i,j}'.
$$

It does not imply that the positions independently had enough capital to execute their proposed actions. Joint feasibility was already enforced by the account optimizer.

### Local proposals and the global coordinator

A scalable implementation may let each position or asset model produce a local value curve

$$
Q_{i,j}(a_{i,j};X_n)
$$

or scenario-dependent PnL curve rather than one irrevocable action. The global coordinator then chooses the jointly feasible combination:

$$
\max_{\{a_{i,j}\}}
\mathbb E_n
\left[
U_\gamma
\left(
W_n+\sum_{i,j}\Pi_{i,j}(a_{i,j},Y)-C_{global}(a,Y)
\right)
\right].
$$

For an approximately additive constrained problem, introduce shadow prices $\lambda$ for shared resources. A local subproblem becomes

$$
\max_{a_{i,j}}
\left[
Q_{i,j}(a_{i,j})
-\lambda^{\mathsf T}g_{i,j}(a_{i,j})
\right],
$$

and the coordinator adjusts $\lambda$ until aggregate collateral, borrow and execution constraints are satisfied. This is conditional decomposition, not unconditional independence.

Positions may propose mutually conflicting actions. Only the netted and globally projected account action is sent to the exchange. Internal borrowing between virtual positions is unnecessary unless it represents a real position-specific credit rule; ordinary shared collateral belongs to the account.

### Order construction and netting

After lifecycle decomposition, convert desired position changes into physical orders:

1. Sum all desired buys and sells per asset and execution venue.
2. Net opposing virtual trades when doing so preserves the intended lifecycle accounting.
3. Apply price, quantity and notional precision.
4. Enforce minimum and maximum order restrictions.
5. Estimate spread, fee, impact, funding and borrow consequences again using the executable order.
6. If the executable action differs materially from the Bellman target, project or reoptimize rather than silently rounding.
7. Submit market, limit or conditional orders according to the selected execution action $u_n^*$.

Virtual positions can cross or hedge for attribution while the physical execution layer submits only the net trade. If separate gross positions have different borrow, liquidation or legal treatment, they cannot be reduced to net exposure and must remain explicit in account feasibility.

### Fill reconciliation

The intended target is not the new state until orders fill. For every fill:

1. update asset and quote balances;
2. update liabilities and collateral;
3. allocate the fill to lifecycle entries or exits according to the chosen matching rule;
4. attribute fees, funding and realized PnL;
5. update remaining order quantities;
6. recompute equity, leverage and liquidation distance;
7. construct the new compressed account state.

Partial fills create an intermediate state and trigger reoptimization when their effect is material. The policy should never assume that the requested target exposure was achieved without reconciliation.

### Complete decision loop

The full online policy is:

1. **Observe:** ingest prices, balances, liabilities, orders, fills and funding.
2. **Compress:** update each asset's anchor, current run and elapsed duration.
3. **Predict:** produce the joint endpoint-duration scenario kernel.
4. **Value:** evaluate candidate signed target exposures with account-level expected utility and Bellman continuation.
5. **Constrain:** enforce global collateral, leverage, borrow, liquidation, order and risk feasibility.
6. **Select:** choose the jointly optimal target exposure and execution action.
7. **Classify:** translate every current-to-target exposure change into hold, long entry, long exit, short entry, short exit or reversal.
8. **Decompose:** create new lifecycle positions for exposure increases and reduce existing positions for exposure decreases.
9. **Net and execute:** construct the feasible physical order set.
10. **Reconcile:** update the account only from actual fills and charges.
11. **Repeat:** advance at the next portfolio decision event.

In compact form,

$$
\boxed{
\text{raw history}
\longrightarrow
X_n
\longrightarrow
\mathcal P_n
\longrightarrow
a_n^*
\longrightarrow
\boldsymbol\ell_n^*
\longrightarrow
\Delta\mathbf x_n
\longrightarrow
\text{position changes}
\longrightarrow
\text{net orders}
\longrightarrow
X_{n+1}.
}
$$

### Conditions for global optimality

The construction represents the optimal policy for the original portfolio problem when:

1. $X_n$ is a sufficient Markov state;
2. the compressed joint event kernel preserves every reward- and continuation-relevant property of the raw process;
3. the Bellman objective matches the intended terminal utility or additive log-wealth objective;
4. $\mathcal A(X_n)$ exactly represents global feasibility;
5. the target-exposure-to-order mapping preserves the selected action or feeds execution deviations back into reoptimization;
6. lifecycle decomposition preserves aggregate exposure and all economically relevant costs;
7. every use of long-short or base-quote symmetry satisfies the complete Bellman symmetry conditions.

The position decomposition itself is exact bookkeeping under weaker assumptions. What requires the stronger conditions is the claim that independently generated local decisions, compressed forecasts, or symmetry-reconstructed actions are also globally Bellman-optimal.
