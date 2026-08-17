# Recommended causal model-input basis

Date: 2026-08-17  
Objective: predict the complete BTCUSDT return distribution from 1s through 1h.

## Short answer

Do not pass every examined feature to the model. The matched 15m neural experiment became 0.233% worse when all 231 forward-market inputs were added together. The defensible basis is the smallest direct joint winner at each horizon, plus only those external coordinates that survive an untouched transfer block.

The five-year direct joint search against the unconditional return distribution selected:

| target | required endogenous basis | primary bits/target | untouched transfer bits/target | blocks |
|---|---|---:|---:|---:|
| 1s | completed 1s range; trailing 60s active count; latest two 1s returns; trailing 60s realized volatility; completed 1s close location | 0.485670 | 0.468705 | 4/4 |
| 1m | completed 1m range; trailing 15m and 60m realized volatility | within 0.001 of the 0.210343 maximum | positive | 4/4 |
| 15m | trailing 15m, 60m, and 240m realized volatility | 0.159805 | 0.159899 | 4/4 |
| 1h | trailing 30m and 240m realized volatility | 0.120519 | 0.134533 | 4/4 |

The exact maximum at 1m also contains trailing 30m realized volatility, but removing it costs only 0.000385 primary bits/target and produces the smaller basis above. The earlier RSI/EMA/last-hour-volume core was selected conditionally after fixing previous return and with a different state construction. It remains a useful ablation branch, but it is not the smallest direct unconditional winner.

## Production input contract now

| branch | coordinate | causal construction / lookback | route to forecast heads | evidence |
|---|---|---|---|---|
| return state | signed log-return lags | latest two completed 1s returns; additionally pass `is_zero` for the latest return | all, strongest at 1s | lag 2 adds 0.006351 primary / 0.005604 transfer bits beyond the prior five-coordinate 1s basis, positive in all four blocks |
| activity state | active count and time since last nonzero return | trailing 10s and 60s counts; clipped zero-run age | 1s–1m | required to model the exact-zero gate; importance rapidly decays above 1m |
| volatility state | realized variance and mean absolute return | trailing 5s, 15s, 1m, 5m, 15m, and 1h | all | absolute-return dependence remains material at every measured scale |
| price core | RSI(2s) | Wilder gain/loss EMA with alpha 1/2 | primarily 1s–15s | selected, 0.071437005 marginal bits at its selection step |
| price core | EMA acceleration(2s, 1s) | current 2s EMA slope minus its previous 1s slope | primarily 1s–15s | selected, 0.046348059 conditional marginal bits |
| price core | EMA slope(8s, 8s) | `10000 * log(EMA8_t / EMA8_t-8) / 8` | primarily 1s–15s | selected, 0.024508952 conditional marginal bits |
| optional volume-regime proxy | last completed 1h `log1p(base_volume)` | do not expose the partial current 1h bar | mainly the 1s distribution head; omit from the minimal basis when direct range/activity/volatility state is present | added 0.098816076 bits after the older indicator-only core, but was not selected by the later direct global subset search |
| candle shape | completed 1s log high-low range | `10000 * log(high / low)` | 1s–1m | selected, 0.055313147 conditional marginal bits |
| local multiscale state | normalized Haar adjacent-return contrast | `(r[t-1]-r[t]) / (sqrt(2) * sqrt(sum(r[t-j]^2, j=0..15)))` | 1s sign head | after fixing lag 2, adds 0.003792 primary / 0.003392 transfer bits; positive in all four blocks; optional for a sufficiently expressive raw-history encoder |
| optional path state | signed 16s efficiency ratios | `sum(r)/sum(abs(r))` and `sum(r)/sqrt(sum(r^2))`, over the latest 16 completed seconds | 1s sign head only | after fixing lag 2, the variance-normalized form adds 0.000734 primary / 0.002347 transfer bits, but loses the parsimonious joint comparison to the Haar contrast; do not require both |
| spot trade flow | last aggressor side | categorical sell/no-trade/buy for the last completed 1s and the 2s-old bin | 1s–15s; allow 1m ablation | selected; 0.12525481 primary / 0.10207176 transfer bits at age 1s; signal vanishes by age 3s |
| spot trade flow | taker quote-volume and trade-count imbalance | `(buy-sell)/(buy+sell)` over the latest completed 1s; optionally EMA over 2s and 8s | 1s; 8s EMA is a 1m ablation | latest quote-volume imbalance adds 0.023082 / 0.036053 bits in the 90-day screen; aggregate-count imbalance is selected jointly in the 30-day search at +0.013598 / +0.044549 conditional bits |
| provisional spot book | L1 and top-5 quantity imbalance | `(bid_qty - ask_qty)/(bid_qty + ask_qty)` from latest snapshot strictly before the origin; discard after 5s | 1s sign head | dedicated audit: 0.014954 / 0.022972 bits; later common-coverage screen: 0.017525 / -0.002830, so longer confirmation is required |
| futures activity | last completed 1m log trade count | Binance USD-M 1m candle, available only after close | 1s–1m | stable individual addition after the five-coordinate core: 0.031064383 / 0.030789436 bits |
| futures activity | last completed 1m high-low range | Binance USD-M completed 1m candle | 1s–1m | stable individual addition after the core: 0.023361213 / 0.027566250 bits; condition jointly with trade count before final promotion |
| cross-market regime | ETH realized volatility | trailing 30m for the 1m head; trailing 60m for 5m and 15m heads | 1m, 5m, 15m | selected: 0.069524 / 0.036094 bits at 1m; stable but smaller through 15m |
| metadata | age and observed mask for every asynchronous source | age in seconds, clipped/log-scaled; separate Boolean observed flag | all | prevents stale values from being interpreted as current measurements |
| calendar | cyclic UTC second/minute/hour and day-of-week | sine/cosine pairs, known before prediction | all | low-cost causal control for activity and volatility seasonality |

The activity and volatility rows are deterministic state summaries rather than separately claimed marginal winners after the complete core. They are retained because the return-distribution work shows that the zero gate and persistent variance state are necessary to reproduce aggregation from 1s to slower scales.

## Why volume is not in the minimal direct basis

Volume is informative, but mostly as an indirect activity/volatility-regime measurement rather than a directional measurement. The apparent contradiction comes from conditioning on different baselines:

| test | primary / transfer information | interpretation |
|---|---:|---|
| completed 1h log volume for the next 1s return, alone | 0.220979 / 0.271570 bits | strong regime information |
| completed 1h log volume added after the older RSI/EMA price core | +0.098816 bits on the chronological holdouts | useful when direct activity and volatility state is absent |
| completed 1m log volume, alone, for the next 1m return | 0.074002 / 0.078866 bits | predicts distribution scale, not primarily sign |
| completed 1m log volume, alone, for the next 1h return | 0.049156 / 0.050809 bits | still informative at 1h, but weaker than the selected multiscale-volatility pair at 0.120519 / 0.134533 bits |
| last 1s spot quote volume after the richer price/volume/range baseline | -0.003177 / +0.000181 bits | no stable residual information |

The global subset search therefore chooses completed range, active-return count, and realized volatility instead of raw volume: they measure the state through which volume is useful more directly. Total unsigned volume also cannot distinguish balanced two-sided trading from directional pressure, is fragmented across venues, and changes structurally with market participation and price level. Trade count, aggressor side, signed imbalance, and order-book flow should not be conflated with raw volume; those composition/direction features can remain predictive after conditioning and are retained separately above.

## Provisional branch: collect and ablate, but do not require yet

| coordinate | lookback | heads | current evidence |
|---|---|---|---|
| normalized L1 order-flow imbalance, bid/ask additions, cancellations, and queue depletion | latest, 1s, 5s | 1s–1m | snapshot OFI proxy positive overall but one chronological block negative |
| Coinbase trailing return | 5s | 1s–15s | 0.017814736 primary / 0.018444009 transfer bits on sparse monthly samples |
| Coinbase realized volatility | 5m | 1m | 0.013721870 / 0.0060156912 bits on sparse monthly samples |
| liquidation count | 15m | 1s, 5s | 0.0070383517 / 0.0075641338 bits on sparse monthly samples |
| liquidation count | 1h | 15s, 1m | 0.0071318950 / 0.0061818179 bits on sparse monthly samples |

These inputs should be connected through gated adapters or source dropout so the production model remains usable when the feed is missing. Promote them only after they improve a joint chronological ablation, not because their individual bits are positive.

## Research-only inputs until live evidence matures

- Deribit 1d/7d/30d ATM IV, 25-delta skew, term slopes, implied-minus-realized volatility, OI imbalance, strike distance, and expiry time.
- GDELT intensity, breadth, sentiment, novelty, and sentiment shocks.
- True mempool count, vbytes, fee distribution, projected-block depth, and arrival shocks.
- Macro surprise and ETF/treasury-flow inputs when point-in-time data becomes available.

These can be logged and scored by the research pipeline, but they should not enter the production basis merely because they are available.

## Exclude from the current basis

- Futures premium/basis level, funding changes, positioning ratios, and open-interest changes: no stable conditional distribution gain was found.
- Slow Binance futures percentage-depth snapshots: the best full-distribution score was negative and unstable.
- Public revised on-chain/flow proxies: no stable 15m–1h result and point-in-time leakage remains a concern.
- A large bank of overlapping RSI/MACD/EMA variants: the selected core captures most tested price-indicator information; adding redundant coordinates increases sample and optimization cost.
- Ordinary and fractional Fourier summaries and complex Morlet coefficients: causal FFT energy, normalized complex bins, fractional-DFT orders 0.25/0.5/0.75, and complex Morlet coordinates at 16/64/256 samples added no stable full-distribution information at 1m, 15m, or 1h. Spectral energy was strong alone but redundant with realized volatility. At 1s, a raw second-lag return explains the complete-distribution gain; retain only the normalized Haar contrast above for the factorized sign head.
- Daily Cboe VIX and hourly BTC DVOL: levels, changes, absolute shocks, and implied-minus-BTC-realized spreads added no stable conditional information from 1m through 1h. Reserve them for slower risk/leverage controls, not the required short-horizon density basis.

## Encoding

1. Keep continuous values continuous. Fit a median/IQR or empirical-quantile transform on training history only; clip extreme normalized values rather than clipping the raw market event.
2. Encode exact-zero return separately from signed magnitude. Encode aggressor side as a three-state categorical value, not as an arbitrary continuous number.
3. Every asynchronous or optional source gets both a value and `age`/`observed` fields. Never silently forward-fill without exposing age.
4. Completed-bar features update only after the bar closes. Never pass a partial 1m or 1h value under a completed-bar feature name.
5. If one model serves multiple horizons, pass a horizon embedding and route horizon-specific coordinates to the relevant output head. Do not force the 1h head to use transient 1s microstructure.

## Training versus feature lookback

Feature lookback and estimation history are different:

- The input coordinates above need at most 1h of causal state, apart from recursive indicator warm-up.
- The exact discrete five-coordinate model had its lowest total held-out log loss with a rolling 90-day training window. Use 90 days as the initial refit window and retain 180 days as the principal ablation.
- A learned continuous model can use longer history with recency weighting, but its choice must be made by chronological validation rather than by pooling future regimes.

## Nested coverage tiers

The search now deliberately trades calendar span for source breadth:

| tier | common span | sources jointly available | role |
|---|---:|---|---|
| endogenous | 5 years | BTC returns, range, close location, volume, indicators, calendar | required stable core |
| archived market | 90–180 days plus a separated 64-day transfer block | spot aggTrade flow, Binance perpetual candles/metrics, BTC/ETH/SOL/BNB/DOGE minute candles | promotion-quality external tests |
| recent broad | 30 days, 2026-07-18 through 2026-08-16 | all preceding public market sources plus recent local Binance spot book | fast discovery; 16d train, 7d primary, 7d untouched transfer |
| live-only | growing from 2026-08-17 | cross-exchange books, Deribit option surface/OI, liquidations, GDELT, mempool | early screening only until another chronological block exists |

The recent broad search considered 147 coordinates jointly. Its promotion decisions are:

| target | 30-day result | decision |
|---|---|---|
| 1s | last aggressor side, aggregate-count imbalance, VWAP gap, RSI(2s), and 1s range transfer positively; book spread has negative leave-one-out transfer value | retain flow features as a fast gated overlay; keep book spread provisional |
| 1m | trailing 60m BTC realized volatility plus last completed Binance perpetual 1m log trade count: 0.293908 primary / 0.312165 transfer bits, 4/4 blocks | promote futures trade count for the 1m head |
| 15m | trailing 240m realized volatility only; one transfer half is negative | no new external promotion; use the five-year volatility basis |
| 1h | selected mixture scores 0.160572 primary but -0.228559 transfer bits | reject all recent external additions at 1h; 30 days is insufficient |

Thirty days is therefore reasonable for discovering fast microstructure and 1m regime features. It is not sufficient evidence for a 1h production feature: the split contains only 381 train, 168 primary, and 167 transfer non-overlapping 1h outcomes.

## Remaining gap before calling this globally optimal

The endogenous basis and the recent 147-coordinate overlay have now been searched jointly inside explicit finite candidate universes. This is not a mathematical optimum over arbitrary transforms: unrestricted mutual information is monotone and would select every causal input. The remaining work is to repeat the 30-day broad test on later non-overlapping months and to accumulate enough live-only history for options, news, cross-exchange books, liquidations, and mempool features.

Full long-history results: `docs/experiments/global-return-feature-basis-2026-08-17.md`.  
Full recent 30-day results: `docs/experiments/global-return-feature-basis-30d-2026-08-17.md`.
Fourier feature audit: `docs/experiments/fourier-return-feature-information-2026-08-17.md`.
Imbalance and volatility-index audit: `docs/experiments/imbalance-and-volatility-index-audit-2026-08-17.md`.
