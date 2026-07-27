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

ML model based on MLP:
1. Historic inputs:
   1. normalize into log returns
   2. standardized candle shape
      1. open-close return
      2. max deviation up from middle
      3. max deviation down from middle
      4. relative log volume over slow volume EMA 
   3. for each candle size last candle is the latest candle that may be partially complete
   4. last 64 1s candles
   5. no last 1s candle fill fraction, assume its the finest granularity
   6. last 64 1m candles
   7. last 1m candle fill fraction
   8. last 32 1h candles
   9. last 1h candle fill fraction
   10. last 32 1d candles
   11. last 1d candle fill fraction
   12. last 16 1M candles
   13. last 1M candle fill fraction
   14. last 16 3M candles
   15. last 3M candle fill fraction
2. teacher and execution configuration, fixed for the current model and not
   included in its network inputs:
   1. fee rate
   2. min/max usable leverage
   3. min/max effective leverage
   4. maintenance costs
3.  Architecture:
    1.  accept the 901 historic market inputs only
    2.  network depth 16
    3.  per layer 1024 neurons
    4. predict 255 raw base-action logits on the stored effective-range grid.
       Normalize and score them on visible leverage, then apply the exact
       deterministic fee/current-exposure transition used by the oracle. Keep
       the eight-parameter quadratic fitter only as a diagnostic.
4.  Optimize the loss function directly against the stored raw oracle policy. All
    distribution objectives use the visible current/target exposure surface:
    conditional cross entropy + probability MSE and action-only cross entropy
    + probability MSE, with oracle MI as a separately reported objective. All
    four distribution weights are 1, oracle MI has weight 0.5, and excess
    entropy is currently disabled. There is no parameter-MSE term for the
    direct output. Oracle MI computes predicted/teacher Gaussian correlation
    over batch time independently at each current exposure before averaging
    states.
    Measure weight sensitivity with the configured resolution-VI joint screen:
    16 simultaneous low/high combinations plus the production-weight center,
    crossed with every configured delay. Report all main effects and 10
    pairwise weight interactions independently at each delay so delay × weight
    dependence is measured rather than assumed away.
    After the response screen, run the resumable one-epoch curriculum pilot:
    start from the best 60-minute models, search every 33-way next-weight step
    along `[30, 30, 1, 1, 0, 0]` minutes, collapse empirically
    policy-equivalent states, and retain a KL-mean/KL-standard-deviation Pareto
    beam. Persist every branch metric and lineage while exporting only selected
    delay finalists.
    Replace the pilot for the final study with the adaptive absolute-weight
    curriculum. Enumerate every six-way direction from absolute levels
    `[0, 0.25, 1, 4]`, require at least one of CE/probability-MSE/parameter-MSE
    to remain nonzero, and do not multiply candidates by production base
    weights. Rank the complete canonical space with per-loss gradients and a
    projected validation-KL Hessian, calibrate it with measured Gaussian-process
    residual probes, and promote only from actual multi-fidelity validation.
    Continue from 60 minutes to one second at arbitrary integer-second delay
    steps, dwelling to restore delay-specific accuracy and backtracking to the
    last accepted model when a transition plateaus.
5.  Examples are every candle in inspector windows
    1. Pair each prediction-time input with the oracle target at
       `predictionTime - predictionDelayMs`; the current experiment uses a
       configurable 60-second delay/hindsight window.
    2. Assign train/validation/test by prediction time. Persist complete
       timestamp-keyed input days and oracle days independently, then represent
       a delay with lightweight row-offset/stride pairings so completed
       components can be reused across delay experiments.
    3. Screen prediction delays `[0, 1, 30, 60]` minutes at every one of the 33
       loss-weight design points (132 combinations), with identical seeds and
       training settings. At 60 minutes the complete
       one-hour teacher value horizon is historical; treat its remaining error
       as feature compression + model/optimization + teacher-fit error, rather
       than claiming it is model approximation alone.
6.  Validation/testing on last 1M worth of 1s candles.


Alternatives:
1.  PatchTST
2.  Decision Transformer
3.  iTransformer
4.  encoder(-decoder)s
5.  LSTM

Insufficient margin trades should not happen

Timestamp (UTC)	Usable-range KL after deep fitting	Action-mean RMSE
2022-06-15 06:39:59	0.147	6.98×
2022-06-16 16:58:59	0.045	3.01×
2022-06-16 16:59:59	0.032	2.04×
2025-06-05 02:11:59	0.058	5.89×
2025-11-05 16:16:59	0.072	3.24×

measure confidence as entropy of the predicted distribution - the more uniform it is, the more uncertain it is. that means confidence corresponds to lower entropy. since entropy is in range 0-inf, we should convert it to 0-1 range with exp(-entropy/temperature).

the 60m delay cant match the distributions, even though the model has all the information.

ML model based on MLP:
3.  Architecture:
    1.  accept 60 historic simple market returns
    2.  network depth 16
    3.  encoder-like shrinking of the layers. Start from 1024 and shrink to 255 in steps of (1024-255)/16=48 per layer.
    4. 255 output neurons predict raw base-action logits on the stored allowed-range grid.
4.  Optimize the loss function directly against the stored raw oracle policy.
    1.  the model should learn to derive the oracle distribution from the input candles
    2.  the loss is mutual information
5.  Examples are every candle in inspector windows
    - Pair features with oracle distributions 
    - 60 1m candle close values (returns) with the oracle distributions computed over the same candles.
6.  validation on a separate, non overlapping set of 1m candles.
7.  at least as many examples as there are parameters in the model.
8.  testing on last 1M worth of 1m candles.


can we somehow use the difference between two successive oracle distributions? Since they are limited in horizon, this isolates the effect of the two candles that leave and enter the window.

try finding simpler learning tasks and then gradually move to the full model:
1. add prediction delay
2. add oracle moving average
3. freezing inner layers after/before some layer.
4. add candle size scaling - the larger the candle, the more of a "rough" idea the oracle has about optimal path. That is basically a proxy for oracle transition path - the more opportunities oracle has for transitions, the more intricate distribution becomes. If oracle only can do one action for the full value horizon, then its decision policy is very simple - depending on the next candle either enter, exit or stay, depending on current position. The candle span tells us how uncertain it is going to be.
5. add temperature scaling - the higher means more diffuse and even distribution
   1. Lets add gradual temperature learning. Start training with high temperature (like 0.1-0.5) and once we saturate scale it down with some factor (like 0.75) until we hit target temperature of 0.01. Then on each new delay set it proportional to initial KL loss in some way, such that we will have the distributions much closer overall (within the target KL). Thus a condition for reducing delay becomes KL<0.15 and temp=0.01. Note that setting initial temp to already equal target temp

6. test a simple kind of encoder - get full minute candle data and encode into PDF of actions that should match the oracle distribution.
7. make a joint model with encoder and diffusion model. the encoder will condition the diffusion model, and diffusion model will generate the oracle distribution. We can use temperature as our "noise" parameter, since it evens out the distributions.
8. codex suggests freezing the parameters of hidden layers, and only train the output layer for a while, when we move to the next delay.
9. Maybe it makes sense to design architecture around trading intuition? Like for example main elements of a strategy are trend estimator, mean-reversion estimator, (anticipatory) entry estimator, volatility estimator (direct or through er, volume), and confidence estimator (position sizing and entry), which are used to determine overall market dynamics and what should we do exactly. The idea for using them is mostly as follows:
   1. follow large scale trend as they are slow changing
   2. when we see unnatural deviation of the price we can trade against the trend, expecting mean reversion
   3. volatility to measure certainty and importance of particular changes. It also presents us many opportunities for profit, since movement range can be larger than fees and quickly accumulate returns.
   4. dont trade persistent moves, prefer entering and exiting at extrema
   5. if market state is uncertain, dont overcommit to a single entry. scale signal according to confidence in its quality.