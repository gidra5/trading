- add strategies/algos like dca, grid, rebalancing, arbitrage, etc
- search for a good portfolio that will lower risks while keeping profits based on correlations - the assets should be orthogonal or compensate variance of each other.
- portfolio management experiments: initial strategy-portfolio harness exists; still need cached multi-symbol data for inverse-vol allocation, correlation-cluster caps, volatility-targeted leverage, market-neutral pairs, robust min-variance baskets, turnover-aware rebalancing, funding-aware overlays, and stress-mode deleveraging. Details are in docs/strategy-research.md and docs/experiment-plan.md.
- randomized model. lets say we look at the price according to some poisson process. At each observation we decide one of the following - open new long/short position, close existing position, or do nothing. We should pick the action based on which will yield best change in equity or break-even price.
- 
- add prediction market support
- is cache size limit per pair? It should be total cache size
- how is random week/length backtest compute equity and return?
- maybe run some kind of genetic algorithm to finetune the parameters. use quick 1month backtests to verify quality and guide the search.
- maybe we can do a kind of backpropagation to improve the parameters.

- estimate how much the quality of peaks and size distributions affect the performance. add a confidence score before creating positions/closing to guide the actual decisions.
- accomodate in some way average candle size at various levels of granularity. Use this as the expected move over the window size.

- avoid position drift by using the actual order results from binance - how much was actually sold and bought and use that as ground truth. Ideally in live run we should use binance account info, position info, order info, etc, as the source of truth. that should make sure that there is no drift possible.
- as base for extrema pick sma window in which there is enough movement on average to get over round trip cost (buy-sell fees).

1. use simple `dS_m=(p_t-p_{t-m})/m` and `dS_m^2=(p_t-p_{t-m}-(p_{t-1}-p_{t-m-1}))/m` to get sma first and second derivatives
2. anticipate the extrema for early entry. the extrema will happen at least half window size in future.
3. use derivatives to estimate order of SMAs with `d_{m,n}=(m-n)/2*dS_m+m(m-n)/6*dS_m^2`.
4. Measure entry error. We can see good entries post fact and compare with actual entries. We can probably get this by using shifted SMAs by half window. We may compute it incrementally during the run. Explore ways to improve entry time.
5. Treat strategy as a derivative. Given some bot executing a strategy, track proportion of investment that it makes - what part of its leveraged equity invested where. Run another bot that monitors that proportion and can accept its own initial equity, that will be invested proportionally to the initial strategy, or even against it. This basically defines an investable derivative over which we can run another bot. Investing in that derivative proportionally invests it into the asset, and realising profit proportionally reduces the investment.
6. Portfolio trading. We can borrow from other asset positions internally. This incurs conversion cost that should be accounted the same way fees are, but double since it is double converted.
7.  Get assets sorted by 24h abs change, volume/market cap, and keep portfolio consisting of best 10 entries
8.  Look for negatively correlated assets with good sharpe and mix to improve overall sharpe. Leverage the better one to keep profits?
9.  Exit signal should be more permissive/absolute?
  
add a button to run the backtest from live bot start date to now.

Hedge mode reconcile warning

Half the confirmation windows
equal sigmas
for Long Range Bounds use largest suitable candle width instead of 1m granularity
try ema instead of sma

Entry grid - detect incoming bottom, extrapolate to decide bottom price, setup grid. Track filled orders. On partial fill we can still allow position exits, but only over currently filled part. Once partial entry is fully exited and we still have on filled entry orders, look for trend direction, if its opposite or too weak, cancel rest. Must maximize entry size

Before transitioning trend to sideways we wait until rate crosses 0. Then if it gets through either rate threshold or absolute threshold relative to point when we entered sideways trend, then we transition into directional trend. The qbs threshold is about the size of fees

0.1	0.3	0/999 both sides swings from +16 on 0/0 to -20.67%
0.3	0.3	0/0 both sides suddenly drops even more to -41.81%, the deeper borrows only make things worse up to -60.73% in 999/0 case. the 0.1 case is basically the same but scaled down.
the best results are short-side only - 0.3 sigma with +21.02%, and 0.1 sigma with +4.31%. buy sigma and depth are irrelevant here, since longs are not created at all.

sell sigma=ae^-x, a some constant, x derivative higher level sma
buy sigma=ae^x

the single side case need an interpretation to be cleared. because it simply controls how large the orders are based on current trend strength.

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
