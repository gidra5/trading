- add strategies/algos like dca, grid, rebalancing, arbitrage, etc
- search for a good portfolio that will lower risks while keeping profits based on correlations - the assets should be orthogonal or compensate variance of each other.
- portfolio management experiments: initial strategy-portfolio harness exists; still need cached multi-symbol data for inverse-vol allocation, correlation-cluster caps, volatility-targeted leverage, market-neutral pairs, robust min-variance baskets, turnover-aware rebalancing, funding-aware overlays, and stress-mode deleveraging. Details are in docs/strategy-research.md and docs/experiment-plan.md.
- randomized model. lets say we look at the price according to some poisson process. At each observation we decide one of the following - open new long/short position, close existing position, or do nothing. We should pick the action based on which will yield best change in equity or break-even price.
- closing all long positions if there are short positions borrowing from them should not be allowed if we get over the leverage limit. the leftover fraction is basically reserved for the already open opposing positions.
- authorize bots to be able to trade actual money.
- add prediction market support
- add proper live-exchange leverage integration for futures and spot margin; paper
  simulator now has a `shortMarginModel = "futures-margin"` option for collateral-backed
  standalone shorts
- is cache size limit per pair? It should be total cache size
- how is random week/length backtest compute equity and return?


- develop strategy
  - while we can attempt to define them mechanically, the market is inherently unpredictable, so it makes sense to approach it with ml - train a model to decide buy/sell/size signals that maximize profit.
  - accomodate in some way average candle size at various levels of granularity
  - the optimal strategy will maximize utility from peaks and valleys, while avoiding loosing to much profit on fees.
  - every trade is supposed to be profitable, so we should calculate break even price and only close positions if we are below/above it. 
  - We may partially close positions, in that case the break even changes.
  - given current price, we can compute how much we need to close the position so that we will break even outside the expected price range. that way we can maximize our exposure, while controlling risk.
  - if we dont have enough capital, or we opened positions against the trend, we should be able to borrow from these bad positions and basically "improve" them in case we were actually wrong.
  - we can "accept" losses on the position that was borrowed from as a way of reducing amount of positions.
  - we can cap losses on bad positions with stop loss setup at some percentage of the break even price. It should primarily borrow to create new position, and not just close the bad one.
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
