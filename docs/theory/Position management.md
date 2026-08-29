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

Note that the overall PnL is the sum of each position's PnL, and RoI is a mix of each of the position's RoIs. Moreover, an optimal policy will have every position closed with best possible RoI. But on the other hand, it will also have only one active position at a time, because at each optimal transition it would fully move all of the equity to the other side.

Nevertheless it allows us to consider a restricted problem of optimally opening and then closing that single position, instead of managing whole portfolio with arbitrary possible action sequences.

Note that opening price primarily affects drawdown resulting from the position, while closing price affects its final return.

We can restrict ourselves according to properties of the optimal policy:
1. We close only when PnL>0
2. Positions are independent - we dont need to consider global availability for opening closing.
3. We only need to maximize RoI and minimize drawdown per position.
4. Size of the position is irrelevant.

Thus we consider lifecycle of a single position. Lets say we open at price $p_o$, time $t_o$ and close at $p_c$, time $t_c$. In the interval $[t_o,t_c]$ there are two prices $p_h$ at time $t_h$ and $p_l$ at $t_l$ for highest and lowest prices. For optimal policy these are equal to close and open prices. The difference corresponds to missed return and drawdown.

Assume we entered at lowest price. Then the task is simply to close optimally. Using restriction (1) we need to only consider the case when $p_c>p_o$. Thus, assume we are in such a state and considering to close at price $p_t$. For simplicity lets also assume that every next step is strictly opposite in direction. Another way to put it is that every step is accumulation of same direction changes.

Now assume there is a distribution $P(p_{t+1})$ for the next price. Under no friction, for $p_{t+1}>p_t$ we trivially hold the position for the next tick and get more of the return. Otherwise we can close and reopen at the lower price. It can be written down as a more general "it gets better before it gets worse" rule.

With friction it gets more complicated. Now for any action to become actually beneficial the price must move past the friction costs. Lets consider two simpler actions for now - fully closing or not at all.

If we hold, that means the return will be bigger than if we sold and reopened later. Since holding doesnt cost anything, if price rises we trivially can benefit from it. Thus if $p_{t+1}>p_t$, we still hold.

In the opposite case holding should be more favorable then selling and buying later. Assume that roundtrip cost is $c$, then it is more beneficial to sell if price drops below $p_t-c$ . The band $[p_t, p_t-c]$ is the ambiguous case - if price is within it, then neither case is trivially benefitial. If we hold and price stays within this limit, then we are worse, but selling is also worse because it does not fall low enough so we can benefit.

For the inbetween case it seems to be better to partially close such that $c<-\Delta p_t$ and we can control that with order size.

Next addition that complicates the optimal policy, is that there is a minimal order size, which in turn creates a cap on how small the roundtrip cost we can make. We are back to square one, when price is within $[p_t, p_t-c_{min}]$.

we could split the policy further by how the price moves:
1. strictly outside $[p_t, p_t-c_{max}]$ - hold if $p_{t+1}>p_t$, sell if $p_{t+1}<p_t-c_{max}$
2. within $[p_t-c_{min}, p_t-c_{max}]$ - sell $\Delta p_t=c=f*s*p_t+f*s*p_{t+1}$, $s=\frac {\Delta p_t} {f(p_{t+1}+p_t)}$. Notice how this formula works even for prev and next case if we constrain $s \in [0,1]$.
3. within $[p_t, p_t-c_{min}]$:
	1. second move larger than first $|p_{t+2}|>|p_{t+1}|$, then necessarily $p_{t+2}>p_t$. If we hold we are sure to get better position at the cost of the intermediate drawdown of size $\Delta p_t$. Selling and buying at $p_{t+1}$ is always worse, since we pay roundtrip cost at the same price movement. That places break-even $p_{t+2}$ strictly above holding's break even $p_t$. Thus we always hold.
	2. In case it is smaller, then neither is better, we need to look at the next price move. Next move is same direction, and if in total it moves us below $p_t-c_{min}>p_{t+3}$, then now its better to sell.
	3. Thus we can conclude that staying in this range necessarily requires looking further into the future.
	4. By induction it is preferrable to hold when $p_{t+h}>p_t$ and $p_{\tau}>p_t-c_{min}$ for any $\tau \in [t,t+h]$. Similarly for selling - it is preferrable when $p_{t+h}>p_t-c_{min}$ and $p_t>p_{\tau}$ for any $\tau \in [t,t+h]$.
	5. As $h \to \infty$, if we are still within the range, then the asset can be considered frozen, as there are no useful opportunities. In which case it is better to invest in another, more volatile asset. And that means selling all and never touching it again.

For high frequency trading the last decision branch becomes increasingly important, since more and more deltas are within this range.

A reasonable assumption is that probability of price path $p_{t:t+h}$ fully staying within $[p_t,p_t-c_{min}]$ goes to 0 as $h \to \infty$. You could write it down as:
$$\prod_{t=T}^{H}p(p_{t}\in[p_{T},p_{T}-c_{min}])\to0$$
In this case we can safely ignore that case, which means the rest of the policy covers all possible price paths optimally.

(\*) Note that from this policy it follows that we can partition all positions by continuous holding and partial close decisions.

Maintenance costs also become increasingly important, since now holding can also be costly, making overall costs constantly grow with time. Lets call holding costs $c_h$, then policy changes as follows:
1. strictly outside $[p_t+c_h, p_t-c_{max}]$ - hold if $p_{t+1}>p_t+c_h$, sell if $p_{t+1}<p_t-c_{max}$
2. within $[p_t-c_{min}, p_t-c_{max}]$ - $c=f*s*p_t+(1-s)*c_h+f*s*p_{t+1}$, $s=\frac {\Delta p_t-c_h} {f(p_{t+1}+p_t)-c_h}$.
3. within $[p_t+c_h, p_t-c_{min}]$:
	1. second move larger than first by $c_h$, $|p_{t+2}|>|p_{t+1}|+c_h$, then necessarily $p_{t+2}>p_t+c_h$. If we hold we are sure to get better position at the cost of the intermediate drawdown of size $\Delta p_t$. Selling and buying at $p_{t+1}$ is worse if $c_h<c$, since we pay roundtrip cost at the same price movement. The condition $c_h<c$ is guaranteed to be false by construction. That places break-even $p_{t+2}$ strictly above holding's break even $p_t$. Thus we always hold.
	2. In case it is smaller, then neither is better, we need to look at the next price move. Next move is same direction, and if in total it moves us below $p_t-c_{min}>p_{t+3}$, then now its better to sell.
	3. Thus we can conclude that staying in this range necessarily requires looking further into the future.
	4. As we look deeper, the cost of holding, as well as partial close increases linearly with $h$.
	5. By induction it is preferrable to hold when $p_{t+h}>p_t+h*c_h$ and $p_{\tau}>p_t-c_{min}$ for any $\tau \in [t,t+h]$. Similarly for selling - it is preferrable when $p_{t+h}>p_t-c_{min}$ and $p_t+h*c_h>p_{\tau}$ for any $\tau \in [t,t+h]$.
	6. Note that $c_{min}=f*s_{min}*p_t+(1-s_{min})*h*c_h+f*s_{min}*p_{t+1}$ increases with $h$ a bit slower than the pure holding cost itself.
	7. As $h \to \infty$, same reasoning applies, but even more aggresively, since the "frozen" range increases as time moves on.

We could naturally express action in this case as sell fraction $s_t$ at each time step. Each branch has its own probability in terms of next price probabilities, which means we can compute expectation of $s_t$ for each time step.

(\*) Notice that it naturally expresses the idea that continuously selling partially as price moves down.

(\*) Notice that we also can apply basically the same policy for short positions. And if we consider 0 exposure position as short with respect to asset, then we can apply it for determining entry points as well.

(\*) The expected action probably minimizes drawdown and maximizes reward under given distribution. 

