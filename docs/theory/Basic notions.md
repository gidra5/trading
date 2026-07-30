Here we discuss and define basic notions and the framework within which we define everything else. That is mostly a model of the portfolio and the market.
### Portfolio
The natural state space is space of raw quantities. Lets call base currency $q$ for quote and the other currencies as $\mathbf a$ for asset vector. For a single asset case we write simply $a$. 

A tuple $\phi=(q, \mathbf a)$ defines a portfolio. For every portfolio $\phi$ we define equity as:$$
\begin{aligned}
Q=q+u \\
u=\mathbf a \cdot \mathbf{p}
\end{aligned}
$$where $\mathbf{p}$ is a vector of prices for each of the assets relative to base currency, while $u$ is mark-to-market value of the assets. It defines total worth of the portfolio at a given time.

We can have negative values for each element of a portfolio, which will indicate debt in that element.

Define effective asset value after fees:$$
\begin{aligned}
u_{eff}&=u*(1-f) && \text{if }u>0 \\
u_{eff}&=\frac {u}{(1-f)} && \text{if }u<0
\end{aligned}
$$
From that we derive effective equity: $$Q_{eff}=q+u_{eff}$$And effective exposure:$$e_{eff}=u_{eff}/Q_{eff}$$
### Exposure
Exposure is measuring the fraction of equity that is stored in the asset: $$\mathbf e=\frac {\mathbf a \odot p} Q$$Similarly to asset quantity, if we have a single asset we write its exposure simply as $e$. If exposure is 0, we can say that equity is independent of the asset or we are not exposed to the asset.
There is also total exposure:$$e_a=\sum_i \mathbf e_i=\frac {\mathbf a \cdot p} Q$$The rest of the equity we call unexposed fraction:$$e_q=\frac q Q$$They are constrained to add up to 1:$$e_q+e_a=1$$That means every exposure element is usually within a normal exposure range $\mathbf e_i \in [0,1]$.
When either exposure or unexposed fraction are outside this range, we say that portfolio is leveraged. Since exposures are constrained, leveraging implies that there is debt somewhere, and that may come with liabilities.

A tuple $\Psi=(e_q,\mathbf e)$ we call portfolio exposure. From exposure definitions we get this relation between it and simple portfolio:$$\Psi=\frac \psi Q$$
### Portfolio evolution
Portfolio by itself is inert - all of its assets usually don't change their quantities over time. But if there is debt somewhere, or the asset is a derivative of some sort, we usually need to pay maintenance costs $m_t$ periodically or at each time step:$$a_{t+1}=(1-m_t)\cdot a_t$$Usually it is expressed in basis points (bps) which can be thought of as a fraction of a percent:$$1\ bps=0.01\%=0.0001$$
### Order execution
Besides the maintenance costs, portfolio can change as a result of order execution. There are two basic orders we can execute:
1. Market orders $M(\mathbf a_m)$ - exchange asset amounts $\mathbf a_m$ with quote immediately at prices $\mathbf p_t$. Negative amounts equate to selling them, positive equate to buying.
2. Limit orders $L(\mathbf p, \mathbf a_{l})$ - reserve asset amounts $\mathbf a_l$ for an exchange exactly at prices $\mathbf p$. Same amounts convention applies here.

Limit orders define market microstructure, while market orders consume it. They can also be called makers (for limit orders) and takers (for market orders).

Once either is executed the portfolio changes as follows:$$(q_{t+1},a_{t+1})=(q_t+(1-f_{t,q})*p_ea_e,a_t-(1-f_{t,a})*a_e)$$Where $p_e$ is the execution price, $a_e$ is executed order size and $f_{t,q}$, $f_{t,a}$ are fees on received asset/quote, measured in bps. 

For market orders $a_e=a_m$, while $p_e$ generally moves away from $p_t$ proportionally to price impact and market depth. For limit orders it is the opposite - $p_e$ is known to be $p$, while $a_e$ may not be equal to $a_l$. 

For small order sizes these caveats are insignificant and can be assumed false, meaning $p_e=p_t$ for market orders and $m_e=m_l$ for limit orders.
### Markets
We call $\omega \in \Omega$ a market in some universe of all markets $\Omega$, defined by a set of limit orders $L_i=(p_i,a_i) \in \omega$. 
Lets define $p_{bid}$ and $p_{ask}$ as best buy/sell prices available on the market:
$$
\begin{aligned}
p_{bid} &= \max\limits_{i :\ a_i > 0} p_i \\
p_{ask} &= \min\limits_{i :\ a_i < 0} p_i
\end{aligned}
$$
From these quantities we can define spread $s$ and mid-price $m$:
$$
\begin{aligned}
s &= p_{ask} - p_{bid} \\
m &= \frac{p_{ask} + p_{bid}}{2}
\end{aligned}
$$
Lets call $\omega_{ask}(p)$ and $\omega_{bid}(p)$ an ask-bid depth at a price level $p$:$$\begin{aligned}
\omega_{bid}(p)=\sum\limits_{i:\ p<p_i\leq p_{bid}} a_i \\
\omega_{ask}(p)=\sum\limits_{i:\ p_{ask}\leq p_i<p} a_i
\end{aligned}$$Similarly $p_{buy}$ and $p_{sell}$ are prices after buy/sell of a given size $a$:$$\begin{aligned}
p_{buy}(a)&=\{p:\omega_{bid}(p)=a\} \\
p_{sell}(a)&=\{p:\omega_{ask}(p)=a\}
\end{aligned}$$Price impact of an market order $M(a)$ can be found as:$$\Delta p=p_{sell}(a_+)-p_{buy}(a_-)$$and effective execution price:$$p_{eff}=\sum\limits_i\frac {a_i p_i} {a}$$Where $i$ ranges over executed limit orders.
### Price
At any time $t$ we can have price $p_t$ for the asset in terms of quote. That is the rate of exchange between them.

Given a price history, we can define a return of a single time step as relative increase of price:$$r_t=\frac {p_t-p_{t-1}} {p_{t-1}}$$