For a trading strategy we can define an oracle as an optimal decision policy given future history.
## Oracle value
We define oracle value as the total return oracle can get over some time horizon $T$ starting from exposure $e$:$$V_{t,T}(e)=\frac {Q_{t+T}} {Q_t}$$This quantity is at the core of the oracle policy, since it directly tries to maximize it.
Note that since we optimize *returns* and not simply equity, we can safely scale everything by $Q_t$, effectively setting it to 1. 

For the oracle it is easiest to act over the space of available exposures $e \in E=[E_-, E_+]$, since it is scale independent from actual equity.
Oracle should be able to handle a wider range of effective exposures $e_{eff} \in E_{eff}=[E_{eff-},E_{eff+}]$, where $A \subset P$, since as price moves, effective exposure also moves. Leaving the set $P$ will result in liquidation.

The usual setup is to assume $e_t=0$, but it is not required. In this way we can model situations when we might have picked up from an older state or decided to withdraw/close some of the positions. We might already have some exposure and it might not be optimal to immediately remove it.

We may also interpret it as "forcing" exposure at time $t$ and then proceeding optimally.

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
For convenience define $z^+=\max(0,z)$, $z^-=\max(0,-z)$. Then the portfolio $\psi'$ will update according to debt maintenance rules:
$$
\operatorname{maint}(q,a)=
\left(q^+-(1+r_q)q^-,\ a^+-(1+r_a)a^-\right)
$$
Usually $r_q=r_a=r_{debt}$.
The transition simply applies it to rebalanced portfolio to get $\phi''$.

For simplicity we can also assume funding costs instead of borrowing costs. These scale linearly with asset position sizes independent of the sign, which yields this expression:
$$
\operatorname{maint}(q,a)=
\left(q,\ (1+r_a)a\right)
$$
### Liquidation phase
Final phase is the liquidation check after rebalancing, maintenance and the time step.

Portfolio is liquidated simply when $e_{eff}\not\in [E^-_{eff},E^+_{eff}]$ or $Q_{liq}\leq0$. 
If liquidated, $a_{t+1}=0$ and $q_{t+1}=Q_{liq}$. 
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
### Finalization phase
Finally, once oracle is done, it closes the remaining portfolio by applying the same fee-aware rebalance rule with target exposure $0$:
$$(q_{t+T}^{flat},a_{t+T}^{flat})=\operatorname{reb}_{p_{t+T},f}((q_{t+T},a_{t+T}),0)$$


### Value recursion over exposure state
We can express the transition rules a bit nicer in the equity-exposure space with transform $\Phi_t$:
$$\begin{aligned}
\Phi_t(q,a)=\left(q+ap_t,\frac{ap_t}{q+ap_t}\right)\\
\Phi_t^{-1}(Q,e)=\left(Q(1-e),\frac{Qe}{p_t}\right) \\
\end{aligned}
$$
The nice thing about it is that transitions like price movement, liquidation, or changing exposure are much simpler in this state space.

Most transition phases are homogeneous in equity: they simply scale $(q,a)$ by some $Q$ which does not change overall exposure. So we only need to track the scalar equity multiplier and the next exposure is practically unchanged.

### Rebalancing phase
First collapse the rebalance phase. It is by definition does not change exposure, which means we can define $R_t(x\to e)$ as:
$$
\operatorname{reb}_{p_t,f}(\Phi_t^{-1}(Q,e),e')
=\Phi_t(R_t(e\to e')*Q,e')
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

### Maintenance phase
After rebalancing we have a maintenance phase. 
$$
\operatorname{maint}(\Phi_t^{-1}(Q,e))=
\left(Q((1-e)^+-(1+r_q)(1-e)^-),\ \frac{Q}{p_t}(e^+-(1+r_a)e^-)\right)
$$
$$
h_t(e)=((1-e)^++e^+-(1+r_q)(e^--(1-e)^-))
$$$$
\Phi_t
\left(\operatorname{maint}(\Phi_t^{-1}(Q,e))\right)=
\left(Qh_t(e),\ \frac{e^+-(1+r_a)e^-}{h_t(e)}\right)
$$

Now collapse passive holding after the target exposure has already been reached. With unit post-rebalance equity, target exposure $e$ corresponds to quote amount $1-e$ and base amount $e/p_t$ at $p_t$. During holding these amounts are not rebalanced. Owned amounts stay constant, borrowed amounts grow by maintenance, and exposure is re-marked at every intermediate price. For $j$ passive price moves define maintenance factors:
$$
\rho^q_{t,j}=\prod_{i=0}^{j-1}(1+r^q_{t+i}),
\qquad
\rho^a_{t,j}=\prod_{i=0}^{j-1}(1+r^a_{t+i})
$$
For constant rates these are $(1+r_q)^j$ and $(1+r_a)^j$. The surviving quote amount, surviving base amount, and marked asset value after $j$ moves are:
$$
\begin{aligned}
B_{t,j}(e)&=(1-e)^+-\rho^q_{t,j}(1-e)^-\\
b_{t,j}(e)&=\frac{e^+-\rho^a_{t,j}e^-}{p_t}\\
A_{t,j}(e)&=p_{t+j}b_{t,j}(e)
=\frac{p_{t+j}}{p_t}\left(e^+-\rho^a_{t,j}e^-\right)
\end{aligned}
$$
So the hold path is not constant in exposure. Its marked wealth, liquidation close wealth, marked exposure, and liquidation-test exposure after each intermediate move are:
$$
\begin{aligned}
W_{t,j}(e)&=B_{t,j}(e)+A_{t,j}(e)\\
Z_{t,j}(e)&=B_{t,j}(e)+L_f(A_{t,j}(e))\\
d_{t,j}(e)&=\frac{A_{t,j}(e)}{W_{t,j}(e)}\\
\epsilon_{t,j}(e)&=\frac{L_f(A_{t,j}(e))}{Z_{t,j}(e)}
\end{aligned}
$$
Without maintenance this reduces to $d_{t,j}(e)=p_{t+j}e/(p_t(1-e)+p_{t+j}e)$, which changes with price unless $e=0$ or $e=1$.

If for any intermediate move $i\in[1,k]$ we have $W_{t,i}(e)\leq0$, $Z_{t,i}(e)\leq0$, or $\epsilon_{t,i}(e)\notin E_{\operatorname{eff}}$, then the hold is liquidated and has value $-\infty$. Otherwise the $k$-move collapsed hold is:
$$
\begin{aligned}
h_{t,k}(e)&=\log W_{t,k}(e)
\end{aligned}
$$
with $h_{t,0}(e)=0$ and $d_{t,0}(e)=e$.

The collapsed composition is therefore:
$$
\operatorname{hold}_{t,k}\left(\operatorname{reb}_{p_t,f}(\Phi_t(Q,x),e)\right)
=\Phi_{t+k}\left(QR_t(x\to e)e^{h_{t,k}(e)},d_{t,k}(e)\right)
$$
or, in log-value form:
$$
\log\frac{Q_{t+k}}{Q_t}
=\log R_t(x\to e)+h_{t,k}(e)
$$

Then the oracle recurrence is:
$$
\begin{aligned}
V_{t,0}(x)&=\log R_t(x\to0)\\
V_{t,k}(x)&=\max\limits_{e\in E}
\left[
\log R_t(x\to e)+h_{t,1}(e)+V_{t+1,k-1}(d_{t,1}(e))
\right]\\
F_{t,H,T}(e)&=h_{t,H'}(e)+V_{t+H',T-H'}(d_{t,H'}(e)),
\quad H'=\min(H,T,\text{remaining moves})
\end{aligned}
$$
Liquidated branches have value $-\infty$. The terminal condition is exactly the finalization phase: rebalance the last surviving state to zero exposure.

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
