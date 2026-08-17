For a trading strategy we can define an oracle as an optimal decision policy given future history.

For the oracle it is easiest to act over the space of available exposures $e \in E=[E_-, E_+]$, since it is scale independent from actual equity.
Oracle should be able to handle a wider range of effective exposures $e_{eff} \in E_{eff}=[E_{eff-},E_{eff+}]$, where $A \subset P$, since as price moves, effective exposure also moves. Leaving the set $P$ will result in liquidation.

The usual setup is to assume $e_t=0$, but it is not required. In this way we can model situations when we might have picked up from an older state or decided to withdraw/close some of the positions. We might already have some exposure and it might not be optimal to immediately remove it.

We may also interpret it as "forcing" exposure at time $t$ and then proceeding optimally.

Note that since we optimize *returns* and not simply equity, we can safely scale everything by $Q_t$, effectively setting it to 1. 
# Transition rules
Before we move to the oracle policy, we need to define how oracle can transition between states each time step.

It can be split in five phases:
1. Targeting phase - decide what should be the exposure for the duration of the time step.
2. Rebalancing phase - given target exposure rebalance portfolio to match it.
3. Maintenance phase - apply maintenance costs due to debt, financing, etc.
4. Liquidation phase - check liquidation conditions. If they are met, the equity goes to 0 and return becomes negative.
5. Finalization phase - once we arrive at the last time step, oracle should simulate moving to 0 exposure.
### Targeting phase
Suppose we have some policy $\pi_t(e_t\to e)$ that defines how we should favor transitioning from $e_t$ to $e$. Then the optimal decision is: $$e'=\arg\max\limits_{e} \pi_t(e_t\to e)$$
### Rebalancing phase
Once we have decided on the target exposure $e_r$, we should rebalance portfolio $\phi_t$ to $\phi_t'$ such that after applying fees we will be at the current target. For that we need to consider buying and selling cases separately.
#### Buying
Let $dq$ be the cost and $da$ be the received amount. Then small size market order will guarantee that:$$da=\frac {(1-f)dq} p$$Where $f$ is the friction (fees, slippage, price impact). Equity will reduce proportionally to the cost:$$Q'=Q-f*dq$$And asset value $u$ will change according to this:$$u'=a*p+(1-f)*dq$$Relating these two relations we get an expression for $dq$:$$dq=\frac {u-p*a} {1-f+f*a}$$That means we buy if the condition $u>p*a$ is true.

#### Selling
Let $da$ be the cost and $dq$ be the received amount. Then small size market order will guarantee that:
$$dq=(1-f)p*da$$Where $f$ is the friction (fees, slippage, price impact). Equity will reduce proportionally to the cost, just like with buying:$$Q_t'=Q_t-f*p*da$$And asset value $u$ will change according to this:$$u'=p*(a-da)$$Relating these two relations we get an expression for $dq$:$$da=\frac {p*a-u} {p*(1-f*a)}$$That means we sell if the condition $u<p*a$ is true.

#### Transition
If we combine it together, the fee-aware rebalance transition map at price $p_t$ is:
$$
\operatorname{reb}((q,a),e)=
\begin{cases}
\left(q-s,\ a+\dfrac{(1-f)s}{p_t}\right),
& e>x,\quad s=\dfrac{Q(e-x)}{1-f+fe}\\[8pt]
\left(q+(1-f)s,\ a-\dfrac{s}{p_t}\right),
& e<x,\quad s=\dfrac{Q(x-e)}{1-fe}\\[8pt]
(q,a), & e=x
\end{cases}
$$
In the buy branch $s$ is gross quote spent. In the sell branch $s$ is gross asset value sold.
### Maintenance phase
This is the cost of simply holding the existing position, simply labeled as the mapping $\operatorname{maint}(q,a)$. There are two models that correspond to borrowing and funding mechanisms, which we will name correspondingly.

The transition simply $\operatorname{maint}(q,a)$ to the rebalanced portfolio to get $\phi''$.

We can also iterate it over the holding time $H$, assuming no liquidation:
$$\begin{aligned}
\operatorname{maint}_H(\psi_t)=\operatorname{maint}_{H-1}(\operatorname{maint}(q, a))
\end{aligned}$$

#### Borrowing maintenance
For convenience define $z^+=\max(0,z)$, $z^-=\max(0,-z)$. Then the portfolio $\psi'$ will update according to debt maintenance rules:
$$
\operatorname{maint}_H(q,a)=
\left(q^+-(1+r_q)^Hq^-,\ a^+-(1+r_a)^Ha^-\right)
$$
Usually $r_q=r_a=r_{debt}$.
#### Funding maintenance
For simplicity we can also assume funding costs instead of borrowing costs. These scale linearly with asset position sizes independent of the sign, which yields this expression:
$$
\operatorname{maint}_H(q,a)=
\left(q,\ (1+r_a)^Ha\right)
$$
#### Variable rates
The rates $r_q$, $r_a$ may vary with time, in that case their cumulative holding result is the product:
$$
r^H_q=\prod_{h\le H}(1+r^{t-h}_q)=r^{H-1}_q(1+r^{t-H}_q)
$$
### Liquidation phase
Final phase is the liquidation check after rebalancing, maintenance and the time step.

Portfolio is liquidated simply when $e_{eff}\not\in [E^-_{eff},E^+_{eff}]$ or $Q_{eff}\leq0$. 
If liquidated, $a_{t+1}=0$ and $q_{t+1}=Q_{eff}$. 
Otherwise $a_{t+1}=a''$ and $q_{t+1}=q''$.

Then after the price move to $p_{t+1}$ the liquidation transition can be expressed as:
$$
liq(q,a)=
\begin{cases}
(Q_{eff},0),
& Q\leq0\ \lor\ Q_{eff}\leq0\ \lor\ e_{eff}\notin E_{eff}\\
(q,a), & \text{otherwise}
\end{cases}
$$

For simplification we can assume $Q_{eff}=0$ for any liquidation:
$$
liq(q,a)=
\begin{cases}
(0,0),
& Q\leq0\ \lor\ Q_{eff}\leq0\ \lor\ e_{eff}\notin E_{eff}\\
(q,a), & \text{otherwise}
\end{cases}
$$
### Finalization phase
Finally, once oracle is done, it closes the remaining portfolio by applying the same fee-aware rebalance rule with target exposure $0$:
$$(q_{t+T}^{flat},a_{t+T}^{flat})=\operatorname{reb}_{p_{t+T},f}((q_{t+T},a_{t+T}),0)$$


### Recursion over exposure state
We can express the transition rules a bit nicer in the equity-exposure space with transform $\varPsi_t$:
$$\begin{aligned}
\varPsi_t(q,a)=\left(q+ap_t,\frac{ap_t}{q+ap_t}\right)\\
\varPsi_t^{-1}(Q,e)=\left(Q(1-e),\frac{Qe}{p_t}\right) \\
\end{aligned}
$$
The nice thing about it is that transitions like price movement, liquidation, or changing exposure are much simpler to express generally in this state space.

Most transition phases are homogeneous in equity: they simply scale $(q,a)$ by some $Q$ which does not change overall exposure. So we only need to track the scalar equity multiplier and the next exposure is practically unchanged.

The equity-exposure space introduces an ambiguity when equity is 0 - all exposures become equivalent. In this case we choose 0 exposure as canonical.
#### Rebalancing phase
First collapse the rebalance phase. It is by definition does not change exposure, which means we can define $R_t(x\to e)$ as:
$$
\operatorname{reb}_{p_t,f}(\varPsi_t^{-1}(Q,e),e')
=\varPsi_t(QR_t(e\to e'),e')
$$
Substituting the buy/sell rebalance formulas gives:
$$
R_t(x\to e)=
\begin{cases}
\dfrac{1-f+fx}{1-f+fe}, & e>x\\[8pt]
\dfrac{1-fx}{1-fe}, & e<x\\[8pt]
1, & e=x
\end{cases}
$$
This is defined only when the relevant denominator and resulting equity are positive; otherwise the branch is infeasible.

#### Maintenance phase
After rebalancing we have a maintenance phase. It follows this general form:
$$
\varPsi_t
\left(\operatorname{maint}_H(\varPsi_t^{-1}(Q,e))\right)=
\left(Qh(e),\ \frac{k(e)}{h(e)}\right)
$$
Where $h_t$ is the equity change and $k_t$ is the exposure change.

The two models have these definitions for equity and exposure changes:
$$
\begin{aligned}
k_{H}^{borrow}(e)&=e^+-(1+r_a)^He^-\\
h_{H}^{borrow}(e)&=(1-e)^++e^+-(1+r_q)^H(e^--(1-e)^-)\\
k_{H}^{fund}(e)&=(1+r_a)^He\\
h_{H}^{fund}(e)&=(1-e)+(1+r_a)^He
\end{aligned}
$$
#### Liquidation and finalization phase
The liquidation and finalization almost does not change:
$$
\varPsi_t
\left(\operatorname{liq}(\varPsi_t^{-1}(Q,e))\right)=
\begin{cases}
(Q_{eff},0),
& Q\leq0\ \lor\ Q_{eff}\leq0\ \lor\ e_{eff}\notin E_{eff}\\
(Q,e), & \text{otherwise}
\end{cases}
$$
It is either identity, or an effective transition to 0 exposure.

We could also derive pure equity change:
$$\eta_{liq}(Q,e)=
\begin{cases}
Q_{eff}/Q,
& Q\leq0\ \lor\ Q_{eff}\leq0\ \lor\ e_{eff}\notin E_{eff}\\
1, & \text{otherwise}
\end{cases}$$
#### Price transition
For direct quote-asset space we didn't need any explicit price transition, since it is basically an identity. But in equity-exposure state it changes the values:$$
\begin{aligned}
\delta_{t+1}(e) &=1+r_{t+1}e\\
\zeta_{t+1}(e)&=(1+r_{t+1})/\delta_{t+1}(e)
\end{aligned}
$$
$$
\begin{aligned}
\varPsi_{t+1}(\varPsi_t^{-1}(Q,e))
&=\left(Q\left(1-e+\frac{p_{t+1}}{p_t}e\right),\frac{\frac{e}{p_t}p_{t+1}}{1-e+\frac{e}{p_t}p_{t+1}}\right)\\
&=\left(Q\left(1-e+(1+r_{t+1})e\right),\frac{(1+r_{t+1})e}{1-e+(1+r_{t+1})e}\right)\\
&=\left(Q\left(1+r_{t+1}e\right),\frac{(1+r_{t+1})e}{1+r_{t+1}e}\right)\\
&=\left(Q\delta_{t+1}(e),\frac{(1+r_{t+1})e}{\delta_{t+1}(e)}\right)\\
&=\left(Q\delta_{t+1}(e),e\zeta_{t+1}(e)\right)
\end{aligned}
$$
#### Log equity
Under these definitions we can separate equity evolution as this, given current and target exposures $e$ and $e'$:
$$\begin{aligned}
Q_t'&=Q_t*R_{t}(e_t\to e_t')*h_{t}(e_t')\\
Q_{t+1}&=Q_t'*\eta_{liq}(Q_t',e_t'')*\delta_{t+1}(e_t'')
\end{aligned}
$$
Since all changes are multiplicative, we can take logarithm and linearize the transition:
$$
\begin{aligned}
\log Q_t'&=\log Q_t+\log R_{t}(e_t\to e_t')+\log h_{t}(e_t')\\
\log Q_{t+1}&=\log Q_t'+\log \eta_{liq}(Q_t', e_t'')+\log \delta_{t+1}(e_t'')
\end{aligned}
$$
Assuming $Q_{eff}=0$, we also get:
$$
\log \eta_{liq}(Q, e)=
\begin{cases}
-\infty,
& Q\leq0\ \lor\ Q_{eff}\leq0\ \lor\ e_{eff}\notin E_{eff}\\
0, & \text{otherwise}
\end{cases}$$
The $e'$ and $e''$ are exposures after transition and after holding respectively. So the equity transition can be separated from exposure, but not completely, they are still coupled, since they are evolving together. But on the other hand, exposure *is almost* separated from equity:
$$\begin{aligned}
e_t'&=e_{t}+\Delta e\\
e_t''&=k(e_t')/h(e_t')\\
e_{t+1}&=e_t''*\eta_{liq}(Q_t',e_t'')*\zeta_{t+1}(e_t'')
\end{aligned}
$$
Only liquidation couples them, but otherwise it is completely independent.
What's also useful, is that any liquidation path maps to $-\infty$.

### Log return recursion
Now we can define *return* $V$ of the particular state under some policy, horizon $T$ at time $t$:
$$V_{t,H}=\log \frac {Q_{t+H}} {Q_t}$$
We can parametrize further with a "discount" of the return $\gamma$, initial holding time $H$, and action delay time $D$ and get these equations for the oracle value recursion $V_{t,H,T}$:
$$
\begin{aligned}
V_{t,0}(x)&=\log R_t(x\to0)\\
V_{t,k}(x)&=\max\limits_{e\in E}
\left[
\log R_t(x\to e)+h_{t,D'}(e)+\gamma V_{t+D',k-D'}(e)
\right],
\quad D'=\min(D,k)\\
V_{t,H,T}(e)&=h_{t,H'}(e)+V_{t+H',T-H'}(e),
\quad H'=\min(H,T)
\end{aligned}
$$

# Bellman equation
We follow RL problem statement.

The set of actions are exposure transitions, identified by target exposure, so $a\in\mathcal A(s)=E$ effectively.
The set of rewards is essentially a set of possible return, so $r\in\mathcal R=\mathbb{R} \cup \{-\infty\}$.

Next we define what is the state set of states of our problem. The market can be generally described as aggregate result of many actors trading within a platform that manages execution. Each actor is defined by its portfolio and set of orders it offers, which define limit order book (LOB) microstructure. Agent is just another actor in this system, and actors are mostly indistinguishable from other agents.

That kind of suggests that whatever policy we choose, it will probably need to have capacity for describing each of the actors, or at least their aggregate behavior, which may be somewhat simpler. If we could estimate market participants count and capacity per actor, then we might manage to predict required model capacity.

In the simplest case we might define state as the LOB that evolves independently, since our market impact is negligible in case of small equity. And the agents state itself - portfolio describing allocation of equity across assets. For simplicity assume only one asset.

As mentioned before it may also include a set of pending limit orders, but for simplicity we omit them as well for now.

Some actors that participate in the market may act based on the historic states, so that means we might want to include all of the prior LOB history as well.

To formalize - set of states is, in most general case, a cartesian product of market states, limit order states and portfolio states $\mathcal S=\mathcal S_{market}\times \mathcal S_{portfolio} \times \mathcal S_{orders}$. And in the particular case considered here it is simplified to $\mathcal S=(market: \mathcal S_{market},\ quote: \mathbb{R},\ asset: \mathbb{R})$

The agent's set of actions is the set of target portfolio and orders states $\mathcal A(s)=\mathcal S_{portfolio}\times \mathcal S_{orders}$, and it is independent of the current state as a whole. That also implies that it is a composite action, consisting of portfolio action and order action. But it still may refer to it for convenience, since it is nice for noop expression. We might want to also express actions as portfolio state deltas, which may feel more natural, but harder to properly manage as action space. So instead we opt for expressing the end result we want and then deriving the necessary changes.

The agent's reward is mainly defined by the return we get by keeping the selected state. That is defined by market movement and order execution, if any. The reward is then the delta between prev equity and the new, calculated as mark-to-market value of all assets+canceling of pending orders. We could generically say its $\mathcal R=\mathbb R$, but more concretely its derived from market return.

Now to the transition distribution. Its effective behavior is mostly described by what we already discussed in prev section. To fit the RL formulation we can a bit adjust it.

First, market changes and portfolio changes are independent, and further more market changes are independent of our actions. Returns and portfolio changes are deterministic in terms of the changes to the state and action. The order state changes are not deterministic in market change/action, essentially because it is not guaranteed to execute fully even if price actually touched it. That suggests the following composition:
$$
p(s', r | s, a)=p_{m}(s_m'|s_m)p_{o}(s_o'|s_o, a_o, s_m')\delta_{s_p'}\delta_r
$$
$$s_p'=F(s_p, a_p)$$
$$r=R(\Delta s_m, \Delta s_p)$$
The delta terms correspond to the two deterministic components, while others correspond to market and order dynamics.

Next lets consider score. It is a simple weighted sum of returns we got. Since returns are deterministic, the score is also deterministic. The weight can describe two useful characteristics - validity of current estimates and value of immediate returns vs delayed ones. Thus we get the following for the score:
$$g_t=r_t+\gamma w_{t+1} g_{t+1}$$
For a perfect trader both $\gamma$ and $w_{t+1}$ are 1, but once we get into approximation/training realm, these might become more useful.

Next is action value function. From definition, it describes expected score given state and action. Since returns are deterministic that reduces to exact score:
$$
q(s, a)=\mathbb{E}[G_t\ |\ s, a]=\sum_{s'\in\mathcal S, r\in\mathcal R} p(s',r\ |\ s, a)[r+w*v(s')]
$$
# Policy
We define policy as a distribution over all possible current and target exposures:$$\pi_{t,T}(e_t\to e)=\frac {R_{t,T}(e_t\to e)} {\int_{E_-}^{E_+}R_{t,T}(e_t\to e)de}$$Where $R_{t,T}$ is a return from moving to the exposure $e$. The return itself is defined simply as oracle value minus transition cost:$$\begin{aligned}
R_{t,T}(e_t\to e)=V_{t,T}(e)-C_t(e_t\to e)
\end{aligned}$$


Note that the optimal path will always choose target exposure that corresponds to the mode of this distribution. Furthermore, it can assign the optimal exposure to move to, given any initial exposure and already accounts for the cost of transition as $C_t$ term.

For training we might want to also add a temperature parameter $\tau$:
$$\pi_{t,T}(e_t\to e)=\frac {R_{t,T}(e_t\to e)^{1/\tau}} {\int_{E_-}^{E_+}R_{t,T}(e_t\to e)^{1/\tau}de}$$

The distribution might become almost uniform if we get timesteps too small. That will cause smaller returns per step and more steps overall in a given time window. That basically means oracle can immediately fix any bad entry exposure, which will remove negative effects of bad choices. That also makes it incredibly high frequency in time - oracle can fix the issue with a small adjustment immediately and keep it on the optimal path.

Another issue is that as amount of timesteps $T$ grows the oracle has more and more opportunities to find profits and make frequent trades, which makes overall policy high frequency at each step, while smoother, low frequency distribution might be preferrable. Temperature can help with this, but it blurs rather than removes the unnecessary detail.

To fix these issue we may restrict oracle decision making. 

First introduce holding time $H$ - how long starting from the initial time should oracle passively wait. That forces it to delay "fixing" bad exposure and instead work with what we get after holding. That smoothens the distribution along its decision points and add more detail to it since now bad exposure can carry much more impact when met with unfavorable price move.

Second introduce resolution $R$ - how fine the oracle's candle view is. That forces oracle to make coarser decisions simply because there are less candles to work with and the get larger due to accumulation of inner movement. By controlling it we can control smoothness more directly, because it starts to isolate the most impactful actions. Otherwise small fluctuations that oracle can exploit will add more frequency to the distribution, that is not necessarily helping to choose optimally.

Now we need 



      26. Let h_t,k(a) and d_t,k(a) be the log wealth multiplier and drifted exposure after passively holding a for k price moves, and let R_t(x->b) be the rebalance wealth multiplier.

      27. The faithful recurrence is V_t,0(x)=ln R_t(x->0), V_t,k(x)=max_b[ln R_t(x->b)+h_t,1(b)+V_t+1,k-1(d_t,1(b))], and Q_t,H,T(a)=h_t,H'(a)+V_t+H',T-H'(d_t,H'(a)), with H'=min(H,T,remaining moves).

      28. H applies only to the initially forced target. The optimal continuation may rebalance every candle and the final state closes to exact zero exposure.

   29. Note that we can have asset vectors instead of singular values, encoding multiple assets per position. The evolution procedure idea is mostly the same, and oracle's exposure is chosen only for the asset where there is the most abs return and 0 for the rest. The assets each can have separate leverages that they must maintain, each define maintenance margin. The portfolio equity must be above the sum of all margins. Rebalancing between two assets incurs double fees, so we generally trade with the quote to rebalance. For now it is not needed, but the current implementation must be future proofed for this case.

30. Strategy defines a distribution over possible exposures, lets call it s_t(a). it decides which exposure is most preferable given the current state at this point in time. Then the bot will execute this strategy by choosing a single exposure a_t and rebalancing to match it. the chosen execution exposure is called a_t=exec(s_t(a)).

31. it is then used to compare strategy with the oracle - pick best possible return exposure and compare with the perfect return corresponding to the chosen exposure. the difference between best and strategy returns is called strategy regret, which yields this formula:

   32. R_t(a) = max_A(Q_t(A)) - Q_t(a)

   33. This can be computed either as regret over the next time T, or as regret until the end of the current evaluation window. The first case might be more versatile, as the former is a special case

34. p_t(a) is the oracle's preference for the exposure a at time t.

   35. p_t(a)=exp(-R_t(a)/temp)/int(exp(-R_t(A)/temp)dA)

36. we compute objective as oracle value distillation over all example windows

   37. L​=−sum(t=1..N,w_t\*[int(p_t(a)\*log(s_t(a))da)])

   38. w_t=W_t/mean_batch(W_t)

   39. D_t=sum_x E[a-x|x] / (sum_x E[abs(a-x)|x]+eps), computed once per complete timestamp example from the exact cutoff-applied raw oracle map over every visible current-exposure/action cell and stored as aligned dataset metadata.

   40. Build p_1m from completed UTC one-minute closes only. At second 59 it is

      exactly aligned; otherwise use the latest completed minute (a conservative

      1-59 second shift) so a 60-minute-delay target contains no hidden

      within-minute ordering.

   41. W_t=(eps+abs(D_t)*persistenceMultiplier)*(1+lambda_resolution*JSD(p_1s,p_1m)).

      The completed-minute target and its visible-range JSD are stored

      separately so lambda_resolution can change without rebuilding either

      oracle.

      This is a ratio of the global signed and absolute displacement integrals, not a mean of separately normalized rows, so each row contributes in proportion to its expected actionable distance.

      42. Important same-side advice accumulates causal, decaying evidence so each later repeated advice receives a larger bounded multiplier.

      43. Opposite advice and long discontinuities reset the persistence evidence; no future timestamp may increase an older timestamp's weight.

      44. Persist the resulting causal unnormalized whole-example W_t in the dataset; training may only normalize it by the current batch mean and must not reconstruct it from fitted parameters.

   45. The configurable mixed objective is L_mix=L_CE+lambda_H*entropyGap-lambda_S*stateMI-lambda_O*oracleMI.

      46. entropyGap is the distance-imbalance-weighted squared positive excess max(0,H(s_t)-H(p_t))/log(|A|).

      47. stateMI uses the normalized Gaussian total/conditional variance decomposition of s_t.

      48. oracleMI can use the normalized Gaussian correlation approximation or precise normalized categorical MI over soft exposure bins.

      49. Any component with lambda=0 is skipped; precise oracle MI retains p_t(a) and runs a separate binned GPU reduction.

50. can we use the exposure distribution for the bot execution specifically? i think we can use variance of the distribution around the realized target exposure as confidence.

51. We can also extend the value function to account for limit orders, which would allow us to use it as prediction of the future price.

   52. limit order is defined in relative terms from current state. now the oracle could choose between making market, limit, both, or nothing.

   53. it generally just outputs what is the preferred final state of the bot state (exposure and pending order), and then execution engine calculates the actual actions needed to achieve that from current state.

   54.  note that we need only one order to be modelled for the oracle. the limit order and market order value follow a bit different value calculations, since limit orders are passive - we dont do anything with them until they execute.

   55.  the tradeoff between market and limit captures the tradeoff between immediate profit and opportunity cost.

   56.  but this idea is for future iterations, not for now.

57.  the limit order model:

    58.  [7/26/2026 12:16 AM] Roman Храновський: Currently i compute a regret for each forced target exposure and use that as a distribution to be learned for the strategy. And values are computed as holding target distribution for H time, then continuing optimally for T time. Regret is then the difference between the optimal target exposure and the actual chosen target exposure.

    59.  I want to design similar regret but for limit orders. i think he premise should be similar. Assume we create a limit order at chosen relative price from current in percents and a reserved exposure. If reserved exposure is borrowed we count borrowing fees each step we hold it before the execution. The reserved amount cant be used for market orders which defines opportunity cost (maybe computed in a similar way to regret). But executing limit order has less fees (potentially 0) than market orders. Then we compute regret as difference between optimal limit order and the chosen one. The optimal one balances opportunity cost such that we get the most profit. We also assume that after limit order is done we act as perfect margin trader.

    60.  The limit order exposure delta is signed - negative is sell, positive is buy.

    61.  The oracle can trade optimally with unreserved assets during lifetime of the lo.

    62.  That essentially scales the optimal market trade return by 1-a

    63.  Then it can trade optimally with post execution equity

    64.  The limit order either executed until the duration T passed, or is cancelled at that time. That is the value horizon

    65.  If candle fully crosses the target price, we execute it at that price.

    66.  The "no order" is identified as any lo with size 0

    67. Limit orders can execute at wicks, while market orders assumed to execute at close basically

    68. Limit price is always positive

    69. Value of the lo is the same way as the mo = final equity over initial

    70. Regret is difference between best value and chosen

    71. Best value is the one where we setr just below wick top at every significant turn. That benefits both from volatility and from reduced fees

    72. We can decide if making limit order is profitable by comparing with empty lo?
