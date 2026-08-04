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
   4. We cover multiple scales of candle sizes: 1s, 1m, 1h, 1d, 1M, 3M
   5. last 64 1s candles
   6. no last 1s candle fill fraction, assume its the finest granularity
   7. last 64 1m candles
   8. last 1m candle fill fraction
   9. last 32 1h candles
   10. last 1h candle fill fraction
   11. last 32 1d candles
   12. last 1d candle fill fraction
   13. last 16 1M candles
   14. last 1M candle fill fraction
   15. last 16 3M candles
   16. last 3M candle fill fraction
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
3.  iTransformer, ipatch
4.  encoder(-decoder)s
5.  LSTM
6.  DLinear https://arxiv.org/html/2606.27282v1?utm_source=chatgpt.com https://arxiv.org/abs/2305.10721?utm_source=chatgpt.com https://arxiv.org/html/2403.14587v2 https://arxiv.org/pdf/2205.13504
7.  TiDE https://arxiv.org/abs/2304.08424
8.  DUET
9.  TLOB https://arxiv.org/html/2502.15757v3
10.  TQNet https://arxiv.org/abs/2505.12917?utm_source=chatgpt.com
11.  MoE https://proceedings.mlr.press/v238/ni24a.html
12.  FITS https://arxiv.org/abs/2307.03756?utm_source=chatgpt.com
13.  TSMixer (https://arxiv.org/pdf/2303.06053, https://arxiv.org/abs/2405.14616?utm_source=chatgpt.com)
14.  cmos https://proceedings.mlr.press/v267/si25a.html
15.  sparsetsf https://proceedings.mlr.press/v235/lin24n.html?utm_source=chatgpt.com
16.  GTR

Insufficient margin trades should not happen

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

7. make a joint model with encoder and diffusion model. the encoder will condition the diffusion model, and diffusion model will generate the oracle distribution. We can use temperature as our "noise" parameter, since it evens out the distributions.
8. codex suggests freezing the parameters of hidden layers, and only train the output layer for a while, when we move to the next delay.
9. Maybe it makes sense to design architecture around trading intuition? Like for example main elements of a strategy are trend estimator, mean-reversion estimator, (anticipatory) entry estimator, volatility estimator (direct or through er, volume), and confidence estimator (position sizing and entry), which are used to determine overall market dynamics and what should we do exactly. The idea for using them is mostly as follows:
   1. follow large scale trend as they are slow changing
   2. when we see unnatural deviation of the price we can trade against the trend, expecting mean reversion
   3. volatility to measure certainty and importance of particular changes. It also presents us many opportunities for profit, since movement range can be larger than fees and quickly accumulate returns.
   4. dont trade persistent moves, prefer entering and exiting at extrema
   5. if market state is uncertain, dont overcommit to a single entry. scale signal according to confidence in its quality.

x

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

train a joint model with predictor-oracle policy.

Lets try this arch:
1. Split into 2-3 parts: (encoder ->) predictor -> (adapter ->) decoder to oracle policy
2. Encoder accepts the input, transforms it into some latent representation, which is used by the predictor to predict next market state in it, then adapter translates into something that decoder can use for oracle policy.
3. Encoder and adapter might be unnecessary, if predictor and decoder perform better by taking these tasks on their own.
4. input basis is still unclear.
5. For decoder i think we found quite good arch. based on the oracle modelling.
6. Predictor is probably what will be the harder part, based on existing, possibly modifier, arch.
7. We can search for it by similarly testing direct predition of the market - give it some input and only require it to predict x candles/closes. If we find accurate model that generalizes to validation, then we can join decoder and predictor together and train them better.
8. We still need to check other training strategies besides simple backpropagation training on direct examples.

The plan can be something like this:
1. For decoder:
   1. Simple MLP seems to suffice when given already sufficient information.
   2. There are many experiments I've ran and some of them are still unfinished, because i wanted to run all of them up to 200 epochs and then decide which of them are worth keeping based on if they genuenly plateaued/diverged/overfit or they are still improving at a similar pace to the training improvements. if they diverged/overfit then we can be sure to drop them, and choose a few of the best from the other ones.
   4. For this one the most important metric is validation base-action KL, which measures how well the model fits the oracle policy. The lower the better, and preferably at least 0.1 +- 0.01
   5. The input for it is basically 60 normalized closes for the window that the oracle used, in the interval (t, t+60]. The oracle is 1h horizon / 1m delay / 1m hold / temperature 0.01 at some time t.
3. For predictor:
   1. Simpler tasks - prediction delay, bigger input candle scaling (less noisy information), smaller prediction horizon.
   2. Arch that will manage prediction is much harder to determine, so we probably need to try each separately
   3. Analyze performance and think of improvements.
   4. The task for this part is simple - given sequence of closes predict next fixed horizon of closes.
   5. At this layer we should also decide which features are worth passing into the predictor, in which amounts and formats. Open/close/high/low/volume, averages, indicators, etc. It is not obvious which of these are useful for the model to learn based on, since indicators/averages are usually derived from them. But on the other hand we might actually benefit from decomposing into them so that they provide some unique and clear information instead of it being "aggregated" into one price movement. But they should be actual non interfering decompositions that add up precisely to reconstruct the original price movement. Lets try these:
      1. Plain history of closes. Certainly must be converted into returns to keep the model abstracted from the scale.
         1. possibly log returns
         2. then these (log) returns can be normalized further in time to have mean 0 and variance 1.
         3. then these time-normalized (log) returns can be train dataset/batch-normalized further to have mean 0 and variance 1.
      2. Decompose history of closes into multiple MA as additive features.
         1. Pass in a history of slowest MA directly. Then the difference between it and the next slowest MA, then the difference between that and the next slowest MA after it, and so on, until last MA and direct candle close.
         2. How many MAs to use? Lets try 1-5.
         3. we can make these convolutions initialized to match each sequence kind so that these are learnable parameters.
      3. History of candles. Not only the closes, but also high and low relative to it.
      4. History of full OHLCV candles. Volumes probably also need to be normalized into relative values.
      5. Multiple candle sizes similar to what we tried before.
      6. lengths of histories similar to what we tried before.
      7. Maybe some other signal if its available. Like for limit order book
      8. Some more complex decomposition into indicators for trend, volatility, etc that is equivalent to direct candles. Maybe FFT of the sequence, its supposed to be equivalent to the original sequence.
      9. Combinations of all of the above. For each signal we can pass multiple candle sizes, MAs, OHLCV, normalize.
   6. It should probably simply predict the same thing that the decoder accepted as input - next 60 1m candle closes/returns.
   7. And probably should optimize square error loss
4. Check best candidates in a joint e2e training. The predictor can pass its results into the decoder though some adapter.
   1. Just like with the modules, we have some choices for adapter, inputs and outputs (these are the same as for separate modules).
   2. The adapter can be a simple MLP, or non existent, or something more complex.
   3. There are also multiple choices in how to train joint model:
     1. Train from scratch end to end to facilitate task specific latent representation
     2. Train in stages:
        1. Train decoder on true future paths until saturated.
        2. Freeze decoder.
        3. Train predictor using path loss plus decoder-policy KL.
        4. Train decoder on noisy and predictor-generated paths to address distribution shift.
        5. Jointly fine-tune with a smaller decoder learning rate.
        6. Retain the best raw-validation-KL checkpoint throughout.
5. All training should happen over the same dataset used for decoder currently, around 553k training examples and 256k validation examples.
6. Each element may also benefit from residual connections as well, possibly with attention like mechanisms.
7. In general we want to decide which arch best suits decoder and predictor, and then try to combine them into a single model. these are a kind of proof of capability, which are then combined to learn together.

Implemented as the causal experiment documented in
`docs/joint-price-oracle.md`: exact same-timestamp 1s input pairing against the
verified 1h-horizon/1m-delay/1m-hold target, 3,600-close context and forecast,
hybrid learned trailing patches/MA, reversible trend/residual normalization,
DLinear plus separate TiDE streams, and a forecast-only 101-usable-action
policy head. The learned policy is available to the regular bot as
`learned-oracle-1s`; chronological holdout acceptance gates live opt-in.

Training result (2026-08-01): v3 is durably paused after epoch 17, with epoch
10 selected by transition-conditioned validation KL. The exported ONNX matches
PyTorch within `6.68e-6` logits. Validation-only calibration selected a safe
flat policy; forced active calibration lost `28.87%` over five validation days
because turnover costs dominated its small gross directional edge. Keep live
activation opt-in and capped at 1x; this model is integrated but not accepted
as profitable. The untouched 30-day test confirmed the rejection: 43,200
decisions, zero conditioned learned actions/trades, `0%` return, versus 4,470
nonzero conditioned hindsight decisions.

Capability-screen result (2026-08-02, finalized): completed matched decoder,
predictor, feature, resolution, objective, causal-architecture, and curriculum
screens without reading sealed test payloads. Promote only
`return-oracle-decoder-learned-radius-direct-long-v1` for a saturation run;
its 1,600-epoch ceiling and 160-stale-epoch stop cover the late-convergence
regime demonstrated by the preserved epoch-1,345 / KL-0.071195 decoder.

Do not promote a candle-only predictor or joint causal policy. The strongest
one, v28's exact six-hour TCN, reached raw validation KL 0.967229 at epoch 7,
but its exact-row gain over v18 was only 0.001848 and it diverged through
epoch 39. Train-selected six-hour close summaries improved the untouched
validation half by only 0.000827; intraminute one-second OHLCV improved it by
only 0.000239, versus the 0.002 feature-integration gate. Direct learning of
the original soft oracle distribution remains the correct objective;
prototype assignments, softened-then-sharpened targets, and clustering do not
add future information. Full evidence and next gates are in
`docs/joint-price-oracle-capability-screen-2026-08-02.md`.

Spot aggressor-flow information gate (2026-08-02): ingested 420 immutable,
checksum-attributed Binance BTCUSDT `aggTrades` days covering all train/
validation targets and predecessor context, with zero sealed-test dates.
Across 85 causal buy/sell-flow features plus matched OHLCV/activity controls,
the frozen-v18 residual audit selected zero validation fusion weight: embargoed
within-audit holdout KL remained 1.002405121 and the gain was 0.000000 versus
the 0.002 gate. Do not train a neural Spot trade-flow branch. The next bounded
information screen must use a genuinely distinct causal source such as futures
positioning/basis/liquidation or related-market context.

USD-M positioning information gate (2026-08-02): ingested the exact 420
checksum-pinned Binance BTCUSDT five-minute metrics days with a full-bin causal
lag, independent nullable-field masks/ages, zero backward timestamp rounding,
and zero sealed-test references. The frozen-v18 paired residual audit selected
zero validation fusion weight: holdout KL remained 1.002405121 and full
validation KL remained 0.969834926, for 0.000000 gain versus the 0.002 gate.
Do not train a neural open-interest/crowding branch. Screen completed USD-M
perp price/volume versus Spot basis/flow next; continue to optimize direct raw
oracle KL rather than prototype or cluster assignments.

USD-M/Spot basis-flow information gate (2026-08-02): joined the exact 420
checksum-pinned USD-M one-minute kline days with the exact 420 Spot aggTrade
days using completed minute k-1, explicit missing/no-trade/live state, causal
ages, and zero sealed-test references. The paired frozen-v18 residual audit
selected basis change, but its train-selected signed-table backoff was zero;
holdout KL remained 1.002405121 and full validation KL remained 0.969834926,
for 0.000000 gain versus the 0.002 gate. Do not train a neural basis/flow
branch. Screen strictly validated USD-M historical order-book depth next;
continue to train directly against the original oracle distribution with raw
KL, never cluster assignments.
