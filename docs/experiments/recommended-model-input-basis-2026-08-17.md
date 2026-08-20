# Recommended causal model-input basis

Date: 2026-08-17  
Objective: predict the complete BTCUSDT return distribution from 1s through 1h.

## Short answer

> **2026-08-19 availability-aware update:** the current per-component contract for 1s, 1m, 15m, and 1h is in [all-feature-availability-audit-2026-08-19.md](all-feature-availability-audit-2026-08-19.md). It reconciles the 147-coordinate recent basis, 309 macro transforms, 76 clean live inputs, 102 spectral transforms, 3,471 dense EMA/RSI variants per larger target, public slow sources, and the overlapping 231-coordinate neural representation. The tables below remain the long-history core and taxonomy, but the newer document is canonical for exact routed inputs and availability fallbacks.

Do not pass every examined feature to the model. The matched 15m neural experiment became 0.233% worse when all 231 forward-market inputs were added together. The defensible basis is the smallest direct joint winner at each horizon, plus only those external coordinates that survive an untouched transfer block.

The five-year direct joint search against the unconditional return distribution selected:

| target | required endogenous basis | primary bits/target | untouched transfer bits/target | blocks |
|---|---|---:|---:|---:|
| 1s | completed 1s range; trailing 60s active count; latest two 1s returns; trailing 60s realized volatility; completed 1s close location | 0.485670 | 0.468705 | 4/4 |
| 1m | completed 1m range; trailing 15m and 60m realized volatility | within 0.001 of the 0.210343 maximum | positive | 4/4 |
| 15m | trailing 15m, 60m, and 240m realized volatility | 0.159805 | 0.159899 | 4/4 |
| 1h | trailing 30m and 240m realized volatility | 0.120519 | 0.134533 | 4/4 |

The exact maximum at 1m also contains trailing 30m realized volatility, but removing it costs only 0.000385 primary bits/target and produces the smaller basis above. The earlier RSI/EMA/last-hour-volume core was selected conditionally after fixing previous return and with a different state construction. It remains a useful ablation branch, but it is not the smallest direct unconditional winner.

## Feature taxonomy: basic/derived and asset-specific/general

These are two independent classifications:

- **Basic** means an atomic causally observed field: an exchange trade, quote, completed candle field, published economic value, timestamp, or article. An exchange-computed completed candle is treated as a basic observation at our acquisition boundary.
- **Derived** means a deterministic coordinate computed from one or more basic observations: a return, lag, rolling statistic, imbalance, indicator, normalized difference, surprise, or age.
- **Asset-specific** means the instantiated value depends on BTC, a BTC venue/instrument/network, or a related market chosen specifically to forecast BTC. The formula can still be portable: BTC realized volatility is asset-specific here even though realized volatility can be calculated for any asset.
- **General** means the value does not change when the target asset changes: UTC time, a global macro release, or general availability metadata. A BTC-filtered news count is asset-specific; the underlying unfiltered article stream is general.

Basic observations are the reproducible source layer, not a recommendation to pass every raw field to the model. The compact production tensor should normally contain the selected derived coordinates plus only those basic categorical/event fields that survived selection.

### Current production and provisional inputs by quadrant

| level | scope | observations or model coordinates | exact role | horizons | status |
|---|---|---|---|---|---|
| basic | asset-specific | completed BTC spot 1s OHLC and last two closes | source for returns, range, and close location; raw OHLC need not also be passed after these are derived | 1s–1m | required source |
| basic | asset-specific | BTC spot aggregate-trade side, quote quantity, and aggregate count | last aggressor side can be passed categorically; quantities/counts feed signed imbalances | 1s–1m | selected |
| basic | asset-specific | completed Binance BTC perpetual 1m trade count and OHLC | pass robustly scaled log trade count; derive completed range | 1s–1m | selected, range still needs joint ablation |
| basic | asset-specific | completed ETH spot candles | source for relative ETH/BTC volatility state | 1m–15m | selected context source |
| basic | asset-specific | synchronized BTC venue bids, asks, quantities, and update timestamps | source for imbalance, microprice, spread, depth flow, and cross-venue state | primarily 1s | provisional live branch |
| basic | general | UTC prediction timestamp | source for cyclic calendar coordinates | all | control source |
| basic | general | source receive/update timestamp and whether a value was observed | source for age and availability coordinates | all asynchronous branches | required metadata source |
| derived | asset-specific | latest two BTC log returns and latest-return exact-zero flag | local signed state and zero gate | primarily 1s | required |
| derived | asset-specific | active-return fractions/counts over 10s and 60s; clipped `log1p` zero-run age | separates no-update probability from active-return magnitude | 1s–1m | required fast branch |
| derived | asset-specific | BTC realized-volatility anchor/differences over 5s, 15s, 1m, 5m, 15m, 30m, 1h, and 4h as routed by head | conditional distribution scale and volatility clustering | all | required |
| derived | asset-specific | completed 1s log range and close location, normalized by the 60s volatility anchor | local candle shape | 1s–1m | required |
| derived | asset-specific | RSI(2s), EMA acceleration(2s, 1s), EMA slope(8s, 8s), normalized Haar contrast | compact local path encoding | primarily 1s–15s | selected fast branch; redundant for a capable raw-history encoder |
| derived | asset-specific | aggressor-side lags and aggregate-count/quote-volume imbalance over 1s, optionally 2s/8s EMA | executed directional flow | 1s–15s | selected; quote-volume channel optional |
| derived | asset-specific | robust log futures trade count and completed futures range | derivative-market activity regime | 1s–1m | selected/provisional joint pair |
| derived | asset-specific | ETH/BTC log-volatility differences at 30m and 60m | cross-market magnitude regime | 1m–15m | selected |
| derived | asset-specific | L1 imbalance, normalized microprice offset, spread, top-five imbalance, per-second added/removed depth, cross-venue dispersion, and futures basis | instantaneous liquidity/order-flow state | primarily 1s | measured early; provisional positive; confirmation pending |
| derived | general | sine/cosine UTC second, minute, hour, and day-of-week phases | activity/volatility seasonality without discontinuous integer encodings | all | low-cost control |
| derived | general | observed mask and clipped `log1p(age_seconds)` for each asynchronous source | prevents missing/stale values from masquerading as current values | all asynchronous branches | required metadata |
| derived | general | forecast-horizon embedding | tells a shared model which conditional distribution is requested | all | required only for a shared multi-horizon model |

There is deliberately no required **basic-general economic/news value** or **derived-general macro/news signal** in the production tensor yet. Those families were measured or are collecting, but none has passed the long-window conditional transfer requirement.

### Examined feature families by quadrant

This registry classifies the broader research space as well as the current production basis, so a raw source is not confused with a transformation derived from it.

| quadrant | basic source families | derived families examined | current disposition |
|---|---|---|---|
| basic + asset-specific | BTC spot OHLCV/trade count; spot aggregate trades; Binance/Coinbase/Kraken/Deribit books; BTC perpetual OHLCV/mark/index/funding/OI/liquidations; ETH/SOL/BNB/DOGE candles; Deribit BTC option quotes/OI; Bitcoin mempool; exchange/whale/miner/ETF/treasury flows; margin borrow rates | none—the entries in this column are acquisition fields | retain required/selected sources above; books and fast liquidation/Deribit trade streams are measured-early; continue confirmation plus low-cadence option-surface/mempool collection; borrow rates remain credential-blocked |
| derived + asset-specific | the preceding asset-specific sources | signed returns/lags/zero flags; active counts and zero gaps; range/close location; realized volatility; RSI/MACD/EMA value/slope/acceleration; efficiency ratio; Haar/Fourier/fractional-Fourier/wavelet summaries; signed trade flow; VWAP gaps; book imbalance/microprice/spread/OFI/depth changes; cross-venue return/dispersion/lead-lag; futures premium/funding changes; liquidation bursts; IV/skew/term structure/implied-minus-realized/OI/strike-distance/expiry features; mempool and flow shocks/z-scores | only the production rows above are selected; book and Deribit perpetual-flow features are measured-early positive; BTC liquidation and Deribit option-trade flow are measured-early inconclusive; full option-surface and mempool features remain research-only; large indicator and spectral banks, slow depth, funding/premium, revised flow proxies, and daily DVOL are excluded from the required basis |
| basic + general | UTC timestamp; unfiltered news articles; scheduled macro release actual/consensus/revision fields; global rates, yield-curve, FX, equity, credit, CPI, labor, production, and GDP levels | none—the entries in this column are published observations | timestamp is required; macro/news observations are research-only and must retain publication/vintage time |
| derived + general | the preceding general sources plus per-source timestamps | cyclic calendar encoding; observed/age masks; standardized macro surprises and revisions; yield slopes/changes; broad-market returns/volatility; news intensity/breadth/sentiment/novelty/shocks | calendar and availability metadata are retained; macro/news transformations are not in the required basis because gains did not survive long-window conditioning or point-in-time data is incomplete |

### What should actually enter the model

For the compact tabular implementation, pass the selected **derived asset-specific** coordinates, the selected basic aggressor-side category, and the **derived general** calendar/availability controls. Do not duplicate them with all underlying raw values. Keep basic observations in storage so features can be reconstructed and new transformations can be tested.

For a sequence encoder, it is reasonable to pass a short basic BTC candle/trade sequence and let the network reconstruct RSI/EMA/Haar-like local dynamics. Still pass explicit multiscale volatility, zero/activity state, asynchronous age/masks, and slower external coordinates: those require much longer or irregular context than the short sequence should be expected to infer reliably.

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

## Operationally canonical alternative

Keep the production input contract above as the evidence-preserving set. The table below is a second, more compact encoding ranked first by predictive evidence and then by normalization, acquisition effort, and feed reliability. `N/E/R` score normalization, ease of acquisition, and reliability from 1 (poor) to 5 (best).

For a window of length (L), define the causal log-RMS volatility anchor

$$
v_L(t)=\frac{1}{2}\log\left(\epsilon+\frac{1}{L}\sum_{j=0}^{L-1}r_{t-j}^2\right).
$$

Passing one anchor together with log-volatility differences is an invertible reparameterization of the original volatility levels, apart from the fixed numerical floor. It improves scale behavior without discarding the volatility level that carries most of the predictive information.

| rank | canonical input group | better-behaved encoding | replaces or consolidates | source and causal delay | N/E/R | use |
|---:|---|---|---|---|:---:|---|
| 1 | multiscale volatility state | anchor `v_60m`; differences `v_15m-v_60m`, `v_30m-v_60m`, `v_240m-v_60m`; analogous 5s/15s/60s differences for the fast head | multiple highly correlated positive volatility levels; exact given the anchor | spot candles through the origin | 5/5/5 | required at every horizon |
| 2 | return and activity state | `r[t]/exp(v_60s)`, `r[t-1]/exp(v_60s)`, exact-zero flags, active fractions `count/L`, and `log1p(zero_age)`; retain `v_60s` | raw return scale, raw active counts, and unbounded delay | spot 1s candles through the origin | 5/5/5 | required for 1s; activity can be dropped progressively above 1m |
| 3 | candle shape | close location in `[-1,1]`; `log1p(range/exp(v_60s))` plus the volatility anchor | raw positive range and redundant absolute scale | completed spot candle | 5/5/5 | required at 1s–1m |
| 4 | executed directional flow | last aggressor side in `{-1,0,+1}` plus aggregate-count imbalance in `[-1,1]`; expose no-trade separately | quote/base imbalance duplicates and raw/aggregate count duplicates; quote-volume imbalance becomes an optional trade-size channel | Binance spot aggregate trades, latest completed 1s | 5/4/4 | preferred external 1s branch |
| 5 | local price dynamics | volatility-normalized Haar contrast; RSI mapped to `[-1,1]`; EMA slope and acceleration divided by matching RMS volatility | unscaled RSI/EMA/Haar bank; all remain derivable from raw return history | spot candles, no external feed | 4/5/5 | use for a compact tabular model; omit redundant indicators for a capable sequence encoder |
| 6 | futures activity | robustly scale `log1p(completed_1m_trade_count)` with training median/IQR; keep completed range only if joint ablation confirms it | futures quote volume, correlated at 0.930 with trade count | Binance USD-M completed 1m candle | 4/4/5 | 1s–1m only |
| 7 | cross-market volatility | `v_ETH,30m-v_BTC,30m` and `v_ETH,60m-v_BTC,60m`, with the BTC anchors already present | absolute ETH volatility, correlated at 0.918 with BTC 30m volatility | completed ETH and BTC spot candles | 5/4/4 | 1m–15m |
| 8 | calendar and availability | sine/cosine phase pairs; Boolean observed flag; clipped `log1p(age_seconds)` | raw integer time fields and silent forward fills | local clock plus source timestamps | 5/5/5 | cheap control at all horizons; lower predictive priority than ranks 1–7 |
| 9 | provisional book state | L1 quantity imbalance in `[-1,1]`, normalized microprice offset, observed flag, and clipped age | L1 notional imbalance and top-2/5/10 imbalance bank | synchronized spot book snapshot strictly before origin, discard after 5s | 5/2/2 | gated 1s sign branch only |

The redundancy measurements use 42,901 common recent origins; book correlations use the 18,954 origins with an observed fresh book:

| candidate pair | Pearson correlation | canonical choice |
|---|---:|---|
| spot taker quote- versus base-volume imbalance | 1.000000 | quote-notional imbalance if a size channel is retained |
| raw-trade versus aggregate-trade count imbalance | 0.976228 | aggregate-count imbalance |
| quote-volume versus aggregate-count imbalance | 0.893261 | aggregate count required; quote volume optional |
| L1 quantity versus L1 notional book imbalance | 1.000000 | L1 quantity imbalance |
| L1 versus top-2/top-5/top-10 quantity imbalance | 0.998728 / 0.996067 / 0.990188 | L1 only until deeper shape proves incremental value |
| futures log trade count versus log quote volume | 0.929814 | trade count |
| ETH versus BTC 30m realized volatility | 0.917902 | log ETH/BTC volatility ratio plus the existing BTC anchor |

This alternative is an encoding and operational simplification, not a new claim that every transformed coordinate has independently passed the feature search. The volatility-anchor, ratio, fraction, and fixed robust-scaling changes preserve the underlying coordinates. Dropping correlated members such as quote volume or deeper book levels should still be confirmed in the learned model's chronological ablation.

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

## Measured-early provisional branch: confirm and jointly ablate

| coordinate | lookback | heads | current evidence |
|---|---|---|---|
| normalized L1 order-flow imbalance, bid/ask additions, cancellations, and queue depletion | latest, 1s, 5s | 1s–1m | snapshot OFI proxy positive overall but one chronological block negative |
| Coinbase trailing return | 5s | 1s–15s | 0.017814736 primary / 0.018444009 transfer bits on sparse monthly samples |
| Coinbase realized volatility | 5m | 1m | 0.013721870 / 0.0060156912 bits on sparse monthly samples |
| liquidation count | 15m | 1s, 5s | 0.0070383517 / 0.0075641338 bits on sparse monthly samples |
| liquidation count | 1h | 15s, 1m | 0.0071318950 / 0.0061818179 bits on sparse monthly samples |
| Deribit perpetual trade activity | trailing 5s and 15s count/amount | 5s–1m | current 20h live screen: roughly 0.011–0.051 bits/target across strict rows; strongest clean live family, but likely contains magnitude/volatility-regime information and still needs joint ablation |
| Binance displayed-depth churn | latest completed 1s | 5s–1m | current 20h live screen: 0.023151 at 5s, 0.025787 at 15s, and 0.033485 at 1m, all 6/6 positive blocks and above shuffled controls |
| rolling spot/perpetual book state | 5s–60s L1 imbalance, cross-venue dispersion/spread | 1s–1m | several smaller strict early rows, about 0.003–0.017 bits/target; retain as gated confirmation candidates |

These inputs should be connected through gated adapters or source dropout so the production model remains usable when the feed is missing. Promote them only after they improve a joint chronological ablation, not because their individual bits are positive.

The first causal live fast-feature screen is recorded in `docs/experiments/live-fast-feature-early-screen-2026-08-19.md`. These families are no longer in a "to be checked" state. The frozen checkpoint has 29.701 observed target hours; compact books cover 20.155h / 72,268 rows, with 1,235 BTCUSDT liquidations, 91,227 Deribit perpetual trades, and 13,306 option trades. It uses completed-second receive-time alignment, a 60/40 chronological split, six evaluation blocks, and shuffled-feature controls. BTC liquidation and Deribit option-trade flow produced no strict early winner; that is an inconclusive measured result, not a rejection. Historical Kraken compact-book rows are excluded because only 69.8% were structurally valid; the depth-truncation bug was fixed on 2026-08-19, and only post-fix Kraken observations may be used in later confirmation runs. Automated 3-day and 7-day reruns are confirmation tests.

### Separate output heads prefer different inputs

A follow-up component screen at **30.933 observed target hours / 21.387 book hours** tests every live input separately against inactivity, active sign, four active-magnitude thresholds, sign inside small/large magnitude subsets, magnitude thresholds conditional on sign, and a joint zero/sign/magnitude state. Thresholds are fitted from training-only active-return quantiles. The numbers below are conditional held-out bits per eligible target beyond previous same-horizon return and trailing absolute-return state; they are marginal one-input results, not a jointly selected basis.

At 1s, the Q25/Q50/Q75 active thresholds all sit near one BTC price tick (about 0.0015–0.0016 bps here), so they should be treated as one micro-move regime rather than three distinct economic thresholds. Q90, about 0.285 bps, is the first clearly separated 1s tail event.

| horizon and output | best early input | bits/eligible target | lower block bound | interpretation |
|---|---|---:|---:|---|
| 1s inactivity | Binance displayed-depth churn | 0.020951 | 0.001622 | queue activity helps predict whether price changes at all |
| 1s active sign | Binance displayed-depth churn | 0.022469 | 0.005611 | short-lived directional information exists after conditioning on activity |
| 1s active magnitude at least Q90, about 0.285 bps | Deribit perpetual trade amount, 5s | 0.098163 | 0.005674 | derivative activity is much more informative for the tail than for ordinary sign |
| 1s joint zero/sign/magnitude state | Binance spot spread | 0.264401 | 0.058455 | spread captures price-update/tick-size and magnitude state; confirm post-regime before promotion |
| 15s active sign | Binance spot L1 imbalance mean, 15s | 0.025671 | 0.000974 | book direction beats the activity features for this head |
| 15s active magnitude at least Q90, about 1.693 bps | BTC liquidation count, 60s | 0.111302 | 0.004475 | liquidations become visible when the target is specifically a large move |
| 1m active sign | Coinbase L1 imbalance mean, 5s | 0.009581 | -0.000094 | no strict early winner for unconditional minute direction |
| 1m active magnitude at least Q75, about 2.076 bps | Deribit perpetual trade count, 60s | 0.111362 | 0.035880 | activity robustly identifies the minute volatility/tail regime |
| 1m active magnitude at least Q90, about 3.352 bps | Binance perpetual basis mean, 15s | 0.185320 | 0.022824 | basis is a tail-state input here, not a general directional input |
| 1m joint zero/sign/magnitude state | Deribit perpetual trade count, 60s | 0.113909 | 0.018114 | best compact joint-state input in this early block |

Therefore the model should not force one shared external feature subset into every output. Use separately gated heads for inactivity, direction, magnitude/tail thresholds, and the joint density, with shared raw observations underneath. BTC liquidations and Deribit option-trade flow had no strict winner for the earlier four-bin whole-return target but do have conditional component candidates; this changes their status from “globally unhelpful” to “potentially specialized,” still subject to the 3-day/7-day confirmation and joint redundancy tests. The complete 6,232 component/input scores are stored in `data/benchmarks/live-fast-feature-early-screen.json`.

The marginal winners above have now been followed by a joint redundancy-aware search on 72,814 common-coverage seconds. Its separate per-head input contracts, exact lookbacks, primary selection scores, untouched transfer scores, and leave-one-out contributions are recorded in `docs/experiments/live-component-feature-bases-2026-08-19.md`. Use that document—not the marginal-winner table—as the current candidate live overlay contract.

## Research-only inputs until live evidence matures

- Deribit 1d/7d/30d ATM IV, 25-delta skew, term slopes, implied-minus-realized volatility, OI imbalance, strike distance, and expiry time.
- GDELT intensity, breadth, sentiment, novelty, and sentiment shocks.
- True mempool count, vbytes, fee distribution, projected-block depth, and arrival shocks.
- Macro surprise and ETF/treasury-flow inputs when point-in-time data becomes available.
- Global rates/yield curves, FX, credit spreads, CPI/labor/production state, and quarterly GDP. The 36-series causal screen now covers the US, euro area, UK, China, Japan, India, and Russia. Recent discovery blocks contain euro 2y-yield, UK production, and China CPI candidates, but those specific gains all disappear after conditioning on the established multiscale volatility/cross-market basis. None of 309 transformations survives the 2021-2026 long-window test. Slow releases also have few independent updates and current histories are revised rather than vintage-correct. If revisited, use a gated rolling-regime adapter rather than required core inputs.

These can be logged and scored by the research pipeline, but they should not enter the production basis merely because they are available.

## Exclude from the current basis

- Futures premium/basis level, 5,919 official BTCUSDT settled-funding observations (levels, changes, and 1/3/7/30-day means), positioning ratios, and open-interest changes: no stable conditional distribution gain was found. Historical pre-settlement predicted funding was not available in this test.
- Binance BTC/USDT margin borrow rates: unavailable without signed USER_DATA credentials and therefore not selected. If obtained, test them first as financing-cost/risk inputs as well as return-density inputs.
- Slow Binance futures percentage-depth snapshots: the best full-distribution score was negative and unstable.
- Public revised on-chain/flow proxies: no stable 15m–1h result and point-in-time leakage remains a concern.
- A large bank of overlapping RSI/MACD/EMA variants: the dense larger-horizon audit tested 3,471 EMA-value/slope/acceleration and RSI parameter/lag combinations per target. All families were informative alone, including delayed signals, but none of the 10,413 target-specific additions produced positive transfer-year information after the selected volatility/range basis. Use these only for the fast 1s branch or as ablations for a raw-history encoder.
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
| live-only fast | 29.701h target / 20.155h books at the frozen first checkpoint | cross-exchange books, Binance BTC liquidations, Deribit perpetual and option trade flow | first screen complete; confirmation-only at 3 and 7 observed days |
| live-only slow | growing from 2026-08-17 | Deribit full option surface/OI, GDELT, true mempool | still accumulating toward the first useful low-cadence chronological test |

The recent broad search considered 147 coordinates jointly. Its promotion decisions are:

| target | 30-day result | decision |
|---|---|---|
| 1s | last aggressor side, aggregate-count imbalance, VWAP gap, RSI(2s), and 1s range transfer positively; book spread has negative leave-one-out transfer value | retain flow features as a fast gated overlay; keep book spread provisional |
| 1m | trailing 60m BTC realized volatility plus last completed Binance perpetual 1m log trade count: 0.293908 primary / 0.312165 transfer bits, 4/4 blocks | promote futures trade count for the 1m head |
| 15m | trailing 240m realized volatility only; one transfer half is negative | no new external promotion; use the five-year volatility basis |
| 1h | selected mixture scores 0.160572 primary but -0.228559 transfer bits | reject all recent external additions at 1h; 30 days is insufficient |

Thirty days is therefore reasonable for discovering fast microstructure and 1m regime features. It is not sufficient evidence for a 1h production feature: the split contains only 381 train, 168 primary, and 167 transfer non-overlapping 1h outcomes.

## Remaining gap before calling this globally optimal

The endogenous basis and the recent 147-coordinate overlay have now been searched jointly inside explicit finite candidate universes. This is not a mathematical optimum over arbitrary transforms: unrestricted mutual information is monotone and would select every causal input. The remaining work is to repeat the 30-day broad test on later non-overlapping months; confirm the already-measured books, liquidation, and Deribit trade-flow results at 3 and 7 observed days; and accumulate enough live-only history for full option surfaces/OI, news, and true mempool features.

Full long-history results: `docs/experiments/global-return-feature-basis-2026-08-17.md`.  
Full recent 30-day results: `docs/experiments/global-return-feature-basis-30d-2026-08-17.md`.
Fourier feature audit: `docs/experiments/fourier-return-feature-information-2026-08-17.md`.
Imbalance and volatility-index audit: `docs/experiments/imbalance-and-volatility-index-audit-2026-08-17.md`.
Dense lagged EMA/RSI audit: `docs/experiments/dense-lagged-indicator-audit-2026-08-18.md`.
