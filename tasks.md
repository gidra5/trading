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
2. use derivatives to estimate order of SMAs with `d_{m,n}=(m-n)/2*dS_m+m(m-n)/6*dS_m^2`.
3. Treat strategy as a derivative. Given some bot executing a strategy, track proportion of investment that it makes - what part of its leveraged equity invested where. Run another bot that monitors that proportion and can accept its own initial equity, that will be invested proportionally to the initial strategy, or even against it. This basically defines an investable derivative over which we can run another bot. Investing in that derivative proportionally invests it into the asset, and realising profit proportionally reduces the investment.
4. Portfolio trading. We can borrow from other asset positions internally. This incurs conversion cost that should be accounted the same way fees are, but double since it is double converted.
5.  Get assets sorted by 24h abs change, volume/market cap, and keep portfolio consisting of best 10 entries
6.  Look for negatively correlated assets with good sharpe and mix to improve overall sharpe. Leverage the better one to keep profits?

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

the lots state should be in bot strategy layer, updated on each signal/related order executoin. on entry we create lot with specified parameters once we successfully executed order. on exit we close the lot, partially or fully depending on exit grid execution state. the resulting state is persisted. it should not need be reconciled/reconstructed from history, but saved and restored as part of live bot state.

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




- refactor the project. 
  - The bot module should have an explicit interface for the trading api it uses and should only call into it. the server should implement different versions for it and inject/pass the correct one depending on the provider - live, paper, or backtest. The bot only manages strategy state and internal lots, submits provider-neutral order requests to the trading api, and updates lots only from provider-normalized order events/fills. Order submission itself must not mutate lots or assume funds/order state changed.
  - the abstract trading api interface should have something like that:
    - create stop-market order - size, price, side.
    - create stop-limit order - size, limit price, stop price, side. 
    - create limit order - size, price, side.
    - create market order - size, side.
    - cancel order - accept order id, return success boolean
    - trading api is abstracted from particular symbol.
    - order methods return order status/result, including accepted/rejected and pending (for stop orders when condition is not yet triggered) state.
    - get price history - retrieves latest `count` candles given candle `interval`.
    - get equity - available for trade and reserved amounts of quote/asset, unleveraged equity amounts
    - get friction - slippage, price impact, etc accounted in generic friction number.
  - trading api execution implementations, all on the server side:
    - live trading - adapts the api to the trading api interface for the selected backend (testnet, demo, prod as well as futures/spot). it is the layer that does actual http requests. To support extended limit of orders also support manual execution on ticks with market orders. this is the only server side state that is used when we hit limits on the number of orders we can place directly on the exchange.
    - paper trading - still use binance price ticks, but simulate accounting, leverage, borrowing, liquidation and limit/market orders locally based on streamed price ticks. Select accounting models like spot borrowing or futures margin, etc. This should be a faithful model of binance accounting and execution.
  - backtesting - uses paper trading but with synthetic price ticks.
  - the bot interface should be something like that:
    - on tick - next price update/candle and timestamp for it.
    - on order - order updates (status, fill, partial fill, cancel, etc)
    - order/fill events echo the bot intent id/client order id and provider order id.
    - fill events include exact filled quote amount, filled asset amount, friction/fee, remaining amount, and terminal/non-terminal state.
    - versioned snapshot
    - restore from snapshot
    - get metrics - only metrics derived from internal strategy state/lots, not account/market stats
    - update config
    - get diagnostics - derive UI/backtest explanation snapshot from current state/context:
      - latest indicator values
      - signal gates and blockers as enum/code values with numeric/raw context, not formatted UI labels/details
      - last signal/reason
      - planned orders
      - entry risk details
      - positions, lots
    - warmup to populate indicators - requests needed amount of candles to fill the indicators and basically runs indicator tick on every candle.
    - constructor
  - strategy interface should be something like that:
    - entry signal - the bot should now make long/short entry with given size, leverarge, (anticipated) price and confidence. The bot then creates entry grids based on this signal and config.
    - exit signal - the bot should now make long/short exit based on positions, (anticipated) price and confidence. The bot then creates exit grids based on this signal and config.
  - strategy state should something look like that:
    - config with indicator parameters
    - list of indictors used by the strategy
      - SMAs
      - EMAs
      - price ranges
      - avg candle sizes
      - rates/derivatives, etc
  - the internal state should something look like that:
    - config
    - strategy
    - lots, for each:
      - id
      - side
      - quote
      - asset
      - break even price
      - internal borrow, list of lots
        - lot id
        - amount borrowed
      - external borrow amount, implying leverage on this specific lot
      - nullable exit grid:
        - list of provider-owned stop limit order refs, for each:
          - bot intent id/client order id
          - provider order id
          - size
          - price
          - filled quote/asset amount
          - terminal state
        - creation price
      - maybe also track filled amounts of quote/asset per whole lot
    - saturation check derived from config cooldown and windows
  - config
    - strategy config
    - sizing/risk parameters:
      - max target leverage as a strategy cap, after provider max leverage is applied by server
      - min and max allowed trade quote, after provider min/max order rules are applied by server
      - entry/close sizing rates, like buy and sell rate
    - exit grid parameters:
      - grid order count
      - max price step between neighboring grid orders
      - size distribution (geometric/linear)
      - size fraction per grid order for geometric distribution
      - grid reset strategy (above prev anchor, above last executed grid order)
    - position lifecycle parameters:
      - optional lot lifetime
      - optional stop loss policy
      - optional take profit policy
    - internal borrow/lot accounting parameters:
      - lender quote and asset locking 
      - borrower profit share to lender
    - should not be in bot config:
      - api keys/secrets
      - symbol/base/quote identity; server/provider owns stream identity and creates separate bot instances per market
      - strategy algorithm tag; use a concrete bot/strategy implementation instead of switching behavior inside one bot by config
      - provider mode, live/paper/backtest
      - generic preferred order type
      - provider order lifecycle settings like time in force
      - generic stale order timeout; cancel orders when strategy state invalidates them instead
      - global max open orders; use concrete strategy limits like exit grid order count instead
      - generic limit offset unless a specific strategy explicitly computes non-grid limit orders from current price
      - forced close behavior like whether to close unprofitable lots; this is a server command option, not strategy config
      - real account balances
      - symbol filters/precision/min notional/max leverage as fetched from provider
      - market stream settings
      - persistence/storage settings
      - reset/status ownership
      - diagnostics/event history retention; server/backtest/ui owns this, bot only derives diagnostics that depend on internals
      - UI labels/formatted text for diagnostics, gates, order plans, chart annotations, and event messages; bot should expose enum/code values plus raw numbers/timestamps and let UI/server format them
  - reset/status change events are owned by server, not by bot
  - market/account stats are owned by server, not by bot
  - account balances are owned by providers
  - metrics are mostly owned by server, only those metrics that directly reference internal strategy state are owned by bot (like lot/exit grid related metrics)
  - the server should use binance as source of truth for base account/order/position state (order status, account equity, open position size) for live trading, simulate them for paper and backtest. Basically implement the trading api interface for each of the providers.
  - On frontend split app into multiple files containing separate components instead of mixing many domains in one file.
  - make a single source of truth for default bot config. The bot module should not have any explicit defaults, the server should own the defaults and use them for backtest scripts and server backtest runs.
  - avoid position drift by making provider order events the only source of truth for accepted/rejected/cancelled/filled order state. The bot should not store exact account balances, reserve balances itself, or derive lots from incomplete order state. It should read tradable equity through the trading api and update internal lots only from provider-normalized order events with exact filled quote/asset amounts.
  - exit grid orders are provider-owned orders referenced by the bot. The bot only tracks their ids, intended lot/grid relation, filled amount, and terminal state; it does not mutate provider order state directly.
  - any exchange resync after reconnects, missed websocket events, or process restart is owned by the provider/server adapter. It should emit the same normalized order events into the bot. This is not bot-level reconciliation.
  - The live mode uses single api key and secret, not separate for binance paper trading and binance live trading. If we use testnet, we simply pass testnet keys. if we use prod, we pass prod keys. 
- as base for extrema pick sma window in which there is enough movement on average to get over round trip cost (buy-sell fees).
- Simulation and live trading mismatch
Simulation says we should already have 16% profit, but we only have 2%. The trades roughly match but maybe there were more in the simulation than or live, or they were having different size. Run live bot, record enough activity to reproduce in a backtest, compare with simulation on the same interval.
