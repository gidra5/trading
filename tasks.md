- add strategies/algos like dca, grid, rebalancing, arbitrage, etc
- search for a good portfolio that will lower risks while keeping profits based on correlations - the assets should be orthogonal or compensate variance of each other.
- portfolio management experiments: initial strategy-portfolio harness exists; still need cached multi-symbol data for inverse-vol allocation, correlation-cluster caps, volatility-targeted leverage, market-neutral pairs, robust min-variance baskets, turnover-aware rebalancing, funding-aware overlays, and stress-mode deleveraging. Details are in docs/strategy-research.md and docs/experiment-plan.md.
- randomized model. lets say we look at the price according to some poisson process. At each observation we decide one of the following - open new long/short position, close existing position, or do nothing. We should pick the action based on which will yield best change in equity or break-even price.
- 
- add prediction market support
- is cache size limit per pair? It should be total cache size
- how is random week/length backtest compute equity and return?
- maybe we can do a kind of backpropagation to improve the parameters.

- estimate how much the quality of peaks and size distributions affect the performance. add a confidence score before creating positions/closing to guide the actual decisions.
- accomodate in some way average candle size at various levels of granularity. Use this as the expected move over the window size.

1. Treat strategy as a derivative. Given some bot executing a strategy, track proportion of investment that it makes - what part of its leveraged equity invested where. Run another bot that monitors that proportion and can accept its own initial equity, that will be invested proportionally to the initial strategy, or even against it. This basically defines an investable derivative over which we can run another bot. Investing in that derivative proportionally invests it into the asset, and realising profit proportionally reduces the investment.
2. Portfolio trading. We can borrow from other asset positions internally. This incurs conversion cost that should be accounted the same way fees are, but double since it is double converted.
3.  Get assets sorted by 24h abs change, volume/market cap, and keep portfolio consisting of best 10 entries
4.  Look for negatively correlated assets with good sharpe and mix to improve overall sharpe. Leverage the better one to keep profits?

Entry grid - detect incoming bottom, extrapolate to decide bottom price, setup grid. Track filled orders. On partial fill we can still allow position exits, but only over currently filled part. Once partial entry is fully exited and we still have on filled entry orders, look for trend direction, if its opposite or too weak, cancel rest. Must maximize entry size. maybw makes sense for exit as well.

sell sigma=b+a*ln(e^x+c), a some constant, x derivative of higher level sma
buy sigma=b+a*ln(e^-x+c)

extrapolate other signals used for sizing.in the same manner

3 representative interavals, but sideways with a rally inbetween defied expectations. 
st “sideways but choppy” candidates:

static with different sigmas (0.1, 0.3), 7d

Window	Close/Open	Low	High	Span	Movement score
2022-07-28..2022-08-03	-0.59%	22582.13	24668	9.09%	highest OHLC churn
2022-05-14..2022-05-20	-0.29%	28630	31460	9.66%	highest close-to-close churn
2021-12-14..2021-12-20	+0.45%	45456	49500	8.66%	very choppy
2021-09-08..2021-09-14	+0.52%	43370	47399.97	8.60%	very choppy
2023-03-18..2023-03-24	+0.22%	26578	28868.05	8.36%	choppy, near-flat close

Regime	Window	Market move	Static sigmas	Return	Max DD	Trades
Uptrend	2023-03-11..2023-03-17	+35.95%	buy 0.3, sell 0.1	+17.32%	5.32%	1,778
Sideways	2026-04-22..2026-04-28	+0.009%	buy 0.1, sell 0.1	+0.27%	0.35%	160
Downtrend	2022-06-12..2022-06-18	-33.26%	buy 0.1, sell 0.3	+29.58%	7.79%	3,533

I found these 3d UTC intervals. Definitions used:

`returnPct` = end vs start  
`biasPct` = 15m average price vs start/end midpoint  
`turns05` = number of 0.5% zigzag turns on 15m closes, used as churn proxy

| case                           | interval                 |        ret |      span |      bias | turns05 |        low |        high |
| ------------------------------ | ------------------------ | ---------: | --------: | --------: | ------: | ---------: | ----------: |
| uptrend, low churn             | `2024-02-24..2024-02-26` |  `+7.355%` |  `8.523%` | `-1.715%` |     `4` |    `50585` |     `54910` |
| uptrend, high churn            | `2022-06-19..2022-06-21` |  `+9.239%` | `19.834%` | `+1.854%` |    `71` | `17960.41` |     `21723` |
| downtrend, low churn           | `2023-06-03..2023-06-05` |  `-5.559%` |  `7.587%` | `+1.626%` |     `4` |    `25388` |  `27455.02` |
| downtrend, high churn          | `2022-06-13..2022-06-15` | `-15.017%` | `25.529%` | `-8.069%` |    `94` | `20111.62` |  `26895.84` |
| sideways, high bias, churn     | `2021-10-19..2021-10-21` |  `+0.302%` |  `9.157%` | `+3.062%` |    `38` | `61322.22` |     `67000` |
| sideways, high bias, low churn | `2025-02-14..2025-02-16` |  `-0.507%` |  `2.877%` | `+0.954%` |     `6` | `96046.18` |     `98826` |
| sideways, low bias, churn      | `2024-07-07..2024-07-09` |  `-0.309%` |  `7.194%` | `-1.966%` |    `40` | `54260.16` |  `58449.46` |
| sideways, low bias, low churn  | `2025-07-04..2025-07-06` |  `-0.348%` |  `2.302%` | `-0.959%` |     `0` |   `107245` | `109767.59` |
| sideways, mid bias, churn      | `2024-01-02..2024-01-04` |  `-0.064%` | `11.611%` | `+0.084%` |    `36` |    `40750` |  `45879.63` |
| sideways, mid bias, low churn  | `2023-09-15..2023-09-17` |  `+0.018%` |  `2.504%` | `+0.006%` |     `4` |    `26224` |     `26888` |

duration	trend	interval	market Sharpe	market move	bot return	bot max DD	bot ann. Sharpe
3d	up	2024-11-09..2024-11-11	2.806	+15.865%	+5.559%	1.690%	14.909
3d	up	2023-12-03..2023-12-05	2.707	+11.718%	+2.642%	2.224%	8.140
3d	down	2026-06-01..2026-06-03	2.842	-12.938%	+8.955%	2.231%	22.167
3d	down	2023-03-07..2023-03-09	2.608	-9.135%	+3.516%	2.059%	9.828
7d	up	2023-11-29..2023-12-05	2.959	+16.538%	+6.331%	2.082%	9.809
7d	up	2024-11-05..2024-11-11	3.454	+30.653%	+5.269%	3.765%	6.039
7d	down	2023-03-03..2023-03-09	2.540	-13.224%	+3.269%	1.936%	5.096
7d	down	2026-05-27..2026-06-02	2.626	-12.076%	+2.525%	4.195%	4.088

duration	interval	market Sharpe	market move	bot return
3d	2022-06-11..2022-06-13	2.602	-22.702%	-3.455%
7d	2022-06-07..2022-06-13

For running one:

```bash
npx tsx scripts/check-sigma-borrow-matrix.ts \
  --start-date 2024-01-02 --end-date 2024-01-04 \
  --sigma-mode static --buy-sigma 0.1 --sell-sigma 0.1 \
  --mode both --long-borrow-depth 999 --short-borrow-depth 999
```

add "parallel" giid strategy that would place limit orders with fixed interval between prices in some range (short/mid price range) and some price distribution among them. Then assume mean reversion/adjust based on high window sma the bias. the exact mechanics are this:
1. place grid of limit orders accordigng to price range, mean, and size distibution
2. when price crosses long order we place a short order at the cell we left. symmentrically for short orders.
3. long grids assume the trend is upwards and accumulate long position as grid crosses any cell and then sell it when price crosses grid cell against assume trend. the short is symmtric
4. neutral grids assume the trend is mean reversing an create short grid above mid, and long below.

also estimate peaks/valleys based on orderbook depth. identify support/resistance levels based on order concentration and predict extrema around them.

- develop strategy
  - while we can attempt to define them mechanically, the market is inherently unpredictable, so it makes sense to approach it with ml - train a model to decide buy/sell/size signals that maximize profit.
  - the optimal strategy will maximize utility from peaks and valleys, while avoiding loosing too much profit on fees.
  - pick actions that will also improve break even prices for the existing positions.
  - if we assume some expected long-term range of volatility, we can make leveraged positions if the liquidation price is outside of it.
  - make grids of limit orders that will capitalize on future movements and not simply monitor the price. that way we dont need to guess exact valleys/peaks, and instead do that implicitly via grid.
  - once we see the reversal of the trend, we can cancel further orders and setup position closing orders going up.
  - grid orders should be balanced, so that approximately half of the orders are long and half are short.
  - grid orders should be placed within expected price range, so that we don't waste freezing money on too improbable price movements.
  - all of that should make sure we get the most out volatility - candle has small body, but very big wicks.
  - use the resulting "balance" as a derivative asset useful for creating portfolios.
  - use boxing of positions - they should basically be closed until some fixed time and have definite take profit/stop loss prices.
  - after we get to some suitable level of performance, explore portfolio management strategies. hedging, rebalancing, correlation reduction, maybe something else.
  - the strategies currently account for jagged up or down trend, but possibly breaks when the trend changes. Need to account for transition periods as well.
  - for portfolio, the perfect trader would pick to move all equity to the asset with best abs move. As an approximation, we can extend borrowing logic to work across assets, and at the entry point we borrow from the worst performing position across all assets.

PORT=3002 TRADING_DATA_DIR=/var/lib/trading/stage
TRADING_MARKET_ID=usdm-futures:SOLUSDT TRADING_BINANCE_PAPER_ENABLED=true TRADING_BINANCE_PAPER_MODE=usdm-futures-testnet TRADING_SHORT_MARGIN_MODEL=futures-margin TRADING_MAX_LEVERAGE=100 TRADING_EXCHANGE_ACCOUNT_GUARD_HARD_STOP=false TRADING_BINANCE_PAPER_AUTO_SUBMIT=true pm2 restart trading-server --update-env

TRADING_WEB_PORT=4174 TRADING_BACKEND_URL=http://207.180.247.128:3002 pm2 restart trading-web --update-env

PORT=3001 TRADING_DATA_DIR=/var/lib/trading/prod TRADING_MARKET_ID=usdm-futures:SOLUSDT TRADING_SHORT_MARGIN_MODEL=futures-margin TRADING_MAX_LEVERAGE=100 TRADING_EXCHANGE_ACCOUNT_GUARD_HARD_STOP=false TRADING_STARTING_QUOTE=50 pm2 restart trading-server-prod --update-env

TRADING_WEB_PORT=4173 TRADING_BACKEND_URL=http://207.180.247.128:3001 pm2 restart trading-web-prod --update-env

pm2 restart trading-server trading-web trading-server-prod trading-web-prod --update-env

PORT=3001 TRADING_DATA_DIR=/var/lib/trading/prod npm run start -w @trading/server
TRADING_WEB_PORT=4173 TRADING_BACKEND_URL=http://127.0.0.1:3001 npm run start -w @trading/web

PORT=3002 TRADING_DATA_DIR=/var/lib/trading/stage npm run start -w @trading/server
TRADING_WEB_PORT=4174 TRADING_BACKEND_URL=http://127.0.0.1:3002 npm run start -w @trading/web

npm run build -w @trading/server -w @trading/bot-algo -w @trading/web

it looks like the state transition chart below the main graph is not accurate and can miss some transitions.

this https://chatgpt.com/c/6a538661-de74-83ed-9f46-856d994d4031

get evaluation closer to this
https://chatgpt.com/c/6a58d7b3-dd44-83eb-b172-a7aec93cf050
https://chatgpt.com/c/6a58fb11-2eb0-83ed-8bfb-807dca191a58
https://chatgpt.com/c/6a58fd5c-bda4-83eb-bd94-3d7c853ac950

we need to adjust the oracle evaluation:
1. for a given window and a point in it, it should generate a range of returns based on some initial exposure, lets call that Q_t(a), the window is implied. Currently it is as if we assume 0 initial exposure, but in general it can be any other value. That models "picking up" from whatever is the current state, not just from clean quote-only portfolio. that matters because moving money back incurs friction. Or rather there should simply be a procedure to do this computation for a given interval and exposure, it can even be defined recursively/iteratively. You could also interpret it as forcing the exposure a at time t and then computing oracle's perfect return afterwards. 
   1. For that lets define a few common sequences: P_t is the price at time t, r_t=P_t/P_{t-1}-1 is the return at time t, E_t is the equity at time t
   2. E_t can be defined as evolution of simple portfolio with the quote and asset j_t=(q_t,u_t). Then mark-to-market value is y_t=u_t*P_t and E_t=q_t+y_t is the total equity at time t.
   3. f_t here is the friction term, generally accounting for fees, slippage, etc. at time t.
   4. based on j_t we also define exposure to the asset as a_t=y_t/E_t
   5. The evolution of j_t is defined as the recurrence over (q,u,a,P,f) tuple: 
      1. Rebalancing phase
         1. Assume we are buying the asset. 
            1. dq is the cost and du is the received amount. that means du=(1-f)*dq/P
            2. the equity then is E^+_t=E-f*dq
            3. and with y^+_t=P\*u+(1-f)*dq
            4. we get dq=(a\*E-P\*u)/(1-f+f*a)
            5. that is true when (a\*E-P\*u)>0
         2. Assume we are selling the asset. 
            1. du is the cost and dq is the received amount. that means dq=P\*(1-f)\*du
            2. the equity then is E^+_t=E-f\*P\*du
            3. and with y^+_t=P\*(u-du)
            4. we get du=(P\*u-a\*E)/(P*(1-f*a))
            5. that is true when (a\*E-P\*u)<0
      2. Maintenance phase:
         1. define [x]=max(0,x)
         2. q^-_{t+1}=(1+r_lend)\*[q^+_t]-(1+r_borrow)\*[-q^+_t]
         3. u^-_{t+1}=[u^+_t]-(1+r_borrow)\*[-u^+_t]
      3. Liquidation phase:
         1. u_liq=(1-f_{t+1})y^-\_{t+1} if u^-_{t+1}>0 and u_liq=y^-\_{t+1}/(1-f_{t+1}) if u^-_{t+1}<0
         2. E_liq=q^-_{t+1}+u_liq
         3. a_eff=u_liq/E_liq
         4. Liquidated when abs(a_eff)>L_max or E_liq<=0
         5. alternatively, liquidated when maintenance margin ratio E_liq/abs(u_liq) is less than 1/L_max
         6. L_max is max effective leverage.
         7. If liquidated, u_{t+1}=0 and q_{t+1}=E_liq
         8. otherwise u_{t+1}=u^-_{t+1} and q_{t+1}=q^-_{t+1}
   6. The oracle defines exposure based on the future returns:
      1. a_t=L^+ if r_t>0 and a_t=-L^- if r_t<0
   7. Then the definition for oracle's return is simply the log return over initial and final equity:
      1. Q_t(a)=ln E_T/E_t
   8. Note that we can have asset vectors instead of singular values, encoding multiple assets per position. The evolution procedure idea is mostly the same, and oracle's exposure is chosen only for the asset where there is the most abs return and 0 for the rest. The assets each can have separate leverages that they must maintain, each define maintenance margin. The portfolio equity must be above the sum of all margins. Rebalancing between two assets incurs double fees, so we generally trade with the quote to rebalance. For now it is not needed, but the current implementation must be future proofed for this case.
   9. bellman equation???
2. Strategy defines a distribution over possible exposures, lets call it s_t(a). it decides which exposure is most preferable given the current state at this point in time. Then the bot will execute this strategy by choosing a single exposure a_t and rebalancing to match it. the chosen execution exposure is called a_t=exec(s_t(a)).
3. it is then used to compare strategy with the oracle - pick best possible return exposure and compare with the perfect return corresponding to the chosen exposure. the difference between best and strategy returns is called strategy regret, which yields this formula:
   1. R_t(a) = max_A(Q_t(A)) - Q_t(a)
   2. This can be computed either as regret over the next time T, or as regret until the end of the current evaluation window. The first case might be more versatile, as the former is a special case
   3. we also might want to compute regret of waiting until t'>t:
   4. Rw_t(a, t')=R_t(a)-R_t'(a)
4. p_t(a) is the oracle's preference for the exposure a at time t.
   1. p_t(a)=exp(-R_t(a)/temp)/int(exp(-R_t(A)/temp)dA)
5. we compute objective as oracle value distillation over all example windows
   1. L​=−sum(t=1..N,w_t\*[int(p_t(a)\*log(s_t(a))da)])
   2. w_t=W_t/sum(W_t)
   3. W_t=eps+R_t(a_t)/median_a(R_t(a)) 
   4. or W_t=max_A(Q_t(a))-min_A(Q_t(a))
6. can we use the exposure distribution for the bot execution specifically? i think we can use variance of the distribution around the realized target exposure as confidence. 
7. We can also extend the value function to account for limit orders, which would allow us to use it as prediction of the future price. 
   1. limit order is defined in relative terms from current state. now the oracle could choose between making market, limit, both, or nothing. 
   2. it generally just outputs what is the preferred final state of the bot state (exposure and pending order), and then execution engine calculates the actual actions needed to achieve that from current state. 
   3.  note that we need only one order to be modelled for the oracle. the limit order and market order value follow a bit different value calculations, since limit orders are passive - we dont do anything with them until they execute. 
   4.  the tradeoff between market and limit captures the tradeoff between immediate profit and opportunity cost.
   5.  but this idea is for future iterations, not for now.


1. what would be needed to move current strategy/signal to a backprop based learning engine? can we efficiently mix it with current genetic approach? In principal i think its possible, we just need to replace all discrete decisions with continuous ones based on soft function like sigmoid.
9. can we adapt the current strategy to the framework that was developed? that will predict action preference instead of a single action.
10. maybe it is time for actual neural network to be trained. it should probably be autoregressive at least, possibly an llm like transformer architecture.
11. train the model on progressively larger intervals based on amounts of oracle signals it contains. start from 1 signal, fit as much as we can to it and then extend up to the next signal, repeat.