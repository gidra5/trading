# All-feature availability audit and prediction basis

Generated `2026-08-19T19:27:02.231Z`.

## Result

All inventories are now reconciled by semantic source rather than added as if every tensor coordinate were independent. The 231-input neural tensor is an encoding of already counted sources, while the dense EMA/RSI and spectral inventories are derived transforms of candles.

The production default is the archive-backed input list below. Local-book features remain optional even when useful because they cannot be reconstructed for arbitrary history. Slow macro/flow inputs stay out because their recent gains did not survive the long conditional test.

## Prediction heads checked

| Horizon | Heads | Confirmed | Primary-only | Insufficient |
|---|---:|---:|---:|---:|
| 1s | 19 | 15 | 4 | 0 |
| 1m | 19 | 19 | 0 | 0 |
| 15m | 19 | 10 | 9 | 0 |
| 1h | 19 | 8 | 10 | 1 |

The 19 heads are inactivity; active sign; joint zero/sign/magnitude; four active-magnitude thresholds; sign in three large- and three small-magnitude regimes; and three magnitude thresholds conditional on each sign.

## Exhaustive inventory ledger

| Inventory | Coordinates | Targets checked | Evidence coverage | Acquisition | Availability | Conclusion |
|---|---:|---|---|---|---:|---|
| recent-broad | 147 | all 19 component heads at 1s, 1m, 15m, 60m | 30d chronological | mixed; 118 archive-backed + 29 local book | 0.94 | selected per head |
| long endogenous basis | 34 | full return distribution at 1m, 15m, 60m; 10-coordinate 1s branch | 2021-07-25..2026-07-25 | candle-derived | 1.00 | promotion/stability reference |
| dense EMA/RSI grid | 3,471 | full return distribution separately at 1m, 15m, 60m | 2021-07-25..2026-07-25 | candle-derived | 1.00 | no stable conditional addition after volatility/range basis |
| 1s technical signals | 166 | activity, sign and magnitude targets | long 1s history | candle-derived | 1.00 | representatives retained only when selected |
| Fourier/FrFT/wavelet | 102 | full distribution, sign, magnitude at 1s, 1m, 15m, 60m | 2021-07-25..2026-07-25 | candle-derived | 1.00 | only three 1s controls survive transfer/tolerance |
| cross-market public | 104 | full/sign/magnitude at 1m, 15m, 60m | multi-year | free kline archives | 0.96 | ETH volatility survives and is represented in broad bases |
| global macro | 309 | full/sign/magnitude at 1m, 15m, 60m | 2021-03-24..2026-08-19 plus warmup | official public, slow cadence | 0.90 | recent discoveries fail long-window conditional stability |
| funding | 18 | full/sign/magnitude at 1m, 15m, 60m | 2021-03-24..2026-08-19 | official public API | 0.96 | no stable addition |
| DVOL | 17 | full/sign/magnitude at 15m, 60m | 2021-03-24..2026-08-19 | official public API | 0.94 | no stable addition |
| VIX | 9 | full/sign/magnitude at 1m, 15m, 60m | 2021-03-24..2026-08-18 | FRED public | 0.94 | no stable addition |
| Coin Metrics network/flows | 37 | full/sign/magnitude at 15m, 60m | 2021-03-24..2026-08-18 | community public API; revisions | 0.78 | no stable addition |
| community exchange/whale/miner flows | 145 | full/sign/magnitude at 15m, 60m | 2022-11-27..2026-08-19 | public derived archive; revisions | 0.65 | no stable addition |
| mempool mining proxies | 36 | full/sign/magnitude at 15m, 60m | 2023-08-19..2026-08-19 | public proxy history | 0.75 | no stable addition; exact mempool is live-only |
| official futures percentage depth | 25 | full/zero/sign for next 1s | 368 official archive days | free Binance archive, sparse dates | 0.82 | one narrow active-sign result; optional until joint common-window ablation |
| Tardis cross-venue quotes/liquidations | 118 | full/sign/magnitude at 1s, 1m, 15m, 60m (plus intermediate horizons) | 10 independent first-of-month UTC days | free monthly samples | 0.60 | no stable candidate at any horizon |
| GDELT news | 8 | early live full-return screen; broad component join pending complete history | 1,162 valid historical buckets through 2026-08-01 plus live | public rate-limited API | 0.70 | not selected; historical backfill checkpointed after HTTP 429 |
| Deribit option surface | 20 | early live full-return screen | 2 fixed surface snapshots plus continuing live summaries | official current public API; no retrospective surface | 0.40 | insufficient evidence; 12 option-trade-flow coordinates are separately included in fast live clean |
| fast live clean | 76 | all 19 heads at 1s, 5s, 15s, 60s | 2026-08-18T20:58:20.000Z..2026-08-19T18:40:33.000Z | collector-only | 0.45 | experimental overlay; no fallback promotion over broad confirmed bases |
| matched forward neural tensor | 231 | 15m neural ablation | same sources as forward-market audit | representation coordinates, not 231 new sources | 0.90 | all-at-once tensor was worse; do not count as a separate feature universe |

Counts overlap intentionally: a lagged EMA, Fourier coefficient, and neural tensor slot can all derive from the same candle. They are separate candidate transforms, not separate data sources.

## Availability-aware selection effect

Within the 0.001-bit near-tie band, availability changed 12 of 75 eligible broad selections. The mean/max predictive sacrifice was 0.000438/0.000932 bits per target.

## Recommended model inputs

### 1s

Confirmed component heads: 15.

| Input | Family | Parameters | Lookback | Causal delay | Availability | Used by heads |
|---|---|---|---|---|---|---|
| active-count-60s | activity | window=60s | 60s | through origin | core candle-derived | inactive |
| close-location-1s | candle shape | normalized OHLC location | 1s | latest completed second | core candle-derived | sign_given_large_q75 |
| ema-acceleration-2s-1s | price dynamics | period=2s,horizon=1s | recursive | through origin | core candle-derived | inactive, sign_given_small_q75 |
| previous-return-1s | return history | lag=1s | 1s | latest completed second | core candle-derived | sign_given_small_q25, sign_given_small_q50 |
| range-1m | minute candle shape | 10000*log(high/low) | 1m | latest completed minute | core candle-derived | large_q75_given_active, large_q90_given_active |
| range-1s | candle shape | 10000*log(high/low) | 1s | latest completed second | core candle-derived | joint_zero_sign_magnitude, large_q25_given_active |
| realized-volatility-30m | minute volatility | sqrt(sum(r_1m^2)),window=30m | 30m | through origin | core candle-derived | inactive, large_q75_given_active, large_q90_given_active, large_q75_given_negative, large_q90_given_negative |
| realized-volatility-60s | volatility | sqrt(sum(r_1s^2)) | 60s | through origin | core candle-derived | large_q90_given_negative, large_q90_given_positive |
| return-difference-1s-control | return dynamics | r(t)-r(t-1s) | 2s | through origin | core candle-derived | sign |
| return-lag-2s-control | return history | lag=2s | 2s | through origin | core candle-derived | full_distribution |
| rsi-2s | price dynamics | period=2s | recursive | through origin | core candle-derived | large_q75_given_negative |
| signed-variance-efficiency-16s | path efficiency | signed efficiency of variance over 16s | 16s | through origin | core candle-derived | sign |
| futures-basis-deviation-5m | basis | futures basis deviation from EMA(5m) | 5m | after 1m candle close | official futures archive | sign_given_active, sign_given_large_q75, sign_given_large_q90 |
| futures-range-1m | futures candle | futures range, last completed 1m | 1m | after 1m candle close | official futures archive | large_q75_given_positive |
| spot-flow-last-side-1s | trade sequence | spot last aggressor side, last 1s | 1s | latest completed second | official aggregate-trade archive | sign_given_active, joint_zero_sign_magnitude, sign_given_small_q25, sign_given_small_q50, sign_given_small_q75, large_q75_given_negative, large_q90_given_negative, large_q75_given_positive, large_q90_given_positive |
| spot-flow-last-side-lag-2s | lagged trade sequence | spot last aggressor side, 2s old | 2s | latest completed second | official aggregate-trade archive | sign_given_small_q25, sign_given_small_q50, sign_given_small_q75 |
| spot-flow-maximum-skew-1s | trade size | spot maximum aggregate-size skew, last 1s | 1s | latest completed second | official aggregate-trade archive | sign_given_active |
| spot-flow-quantity-squared-skew-1s | trade size | spot aggregate-size-squared skew, last 1s | 1s | latest completed second | official aggregate-trade archive | large_q75_given_positive, large_q90_given_positive |
| spot-flow-raw-per-aggregate-1s | trade structure | spot raw trades per aggregate, last 1s | 1s | latest completed second | official aggregate-trade archive | large_q75_given_active, large_q90_given_active |
| spot-flow-trade-count-imbalance-lag-2s | lagged aggressor direction | spot raw-trade count imbalance, 2s old | 2s | latest completed second | official aggregate-trade archive | sign_given_large_q90 |
| spot-flow-trade-imbalance-ema-2 | aggressor direction | spot raw-trade imbalance EMA(2s) | latest completed observation | latest completed second | official aggregate-trade archive | sign_given_large_q75, sign_given_large_q90 |
| top-position-minus-account-log | positioning spread | top-position minus top-account log ratio | latest completed observation | full 5m publication lag | official futures archive | large_q25_given_active |

Optional scarce overlays (the model must also support a missing mask and an archive-only fallback):

| spot-book-spread-bps | spot book top of book | bid-ask spread | latest snapshot | strictly before boundary; maximum age 5s | optional local snapshot | joint_zero_sign_magnitude, large_q25_given_active |

### 1m

Confirmed component heads: 19.

| Input | Family | Parameters | Lookback | Causal delay | Availability | Used by heads |
|---|---|---|---|---|---|---|
| active-count-60s | activity | window=60s | 60s | through origin | core candle-derived | large_q90_given_negative |
| close-location-1s | candle shape | normalized OHLC location | 1s | latest completed second | core candle-derived | sign_given_large_q50 |
| completed-1h-log-volume | volume regime | UTC-aligned 1h | 1h | after hour close | core candle-derived | large_q50_given_active |
| ema-slope-8s-8s | price dynamics | period=8s,horizon=8s | recursive | through origin | core candle-derived | inactive, sign_given_active, joint_zero_sign_magnitude, large_q25_given_active |
| previous-return-1s | return history | lag=1s | 1s | latest completed second | core candle-derived | sign_given_large_q90 |
| realized-volatility-15m | minute volatility | sqrt(sum(r_1m^2)),window=15m | 15m | through origin | core candle-derived | large_q90_given_active, large_q75_given_positive, large_q90_given_positive |
| realized-volatility-30m | minute volatility | sqrt(sum(r_1m^2)),window=30m | 30m | through origin | core candle-derived | large_q90_given_negative, large_q50_given_positive |
| realized-volatility-60m | minute volatility | sqrt(sum(r_1m^2)),window=60m | 60m | through origin | core candle-derived | inactive, joint_zero_sign_magnitude, large_q25_given_active, large_q75_given_active, large_q50_given_negative, large_q75_given_negative |
| realized-volatility-60s | volatility | sqrt(sum(r_1s^2)) | 60s | through origin | core candle-derived | large_q75_given_negative |
| rsi-2s | price dynamics | period=2s | recursive | through origin | core candle-derived | sign_given_small_q25 |
| eth-realized-volatility-30m | cross-market volatility | window=30m | 30m | through origin | official alt kline archive | large_q50_given_active, large_q75_given_active, large_q90_given_active, large_q50_given_negative, large_q75_given_positive, large_q90_given_positive |
| eth-realized-volatility-60m | cross-market volatility | window=60m | 60m | through origin | official alt kline archive | large_q75_given_negative, large_q50_given_positive |
| futures-basis-deviation-15m | basis | futures basis deviation from EMA(15m) | 15m | after 1m candle close | official futures archive | sign_given_active, sign_given_large_q75, sign_given_small_q50, sign_given_small_q75 |
| futures-basis-deviation-5m | basis | futures basis deviation from EMA(5m) | 5m | after 1m candle close | official futures archive | sign_given_large_q50 |
| futures-close-location-1m | futures candle | futures close location, last completed 1m | 1m | after 1m candle close | official futures archive | sign_given_large_q75, sign_given_small_q25 |
| futures-log-trade-count-1m | futures activity | futures log trade count, last completed 1m | 1m | after 1m candle close | official futures archive | inactive, joint_zero_sign_magnitude, large_q75_given_active, large_q90_given_active, large_q90_given_negative, large_q75_given_positive, large_q90_given_positive |
| futures-range-1m | futures candle | futures range, last completed 1m | 1m | after 1m candle close | official futures archive | large_q25_given_active, large_q50_given_active, large_q50_given_negative, large_q50_given_positive |
| spot-flow-last-side-1s | trade sequence | spot last aggressor side, last 1s | 1s | latest completed second | official aggregate-trade archive | sign_given_active, sign_given_small_q25, sign_given_small_q50, sign_given_small_q75 |
| spot-flow-trade-imbalance-ema-2 | aggressor direction | spot raw-trade imbalance EMA(2s) | latest completed observation | latest completed second | official aggregate-trade archive | sign_given_large_q50, sign_given_large_q75, sign_given_large_q90 |
| top-account-minus-global-log | positioning spread | top-account minus global log ratio | latest completed observation | full 5m publication lag | official futures archive | sign_given_large_q90 |
| top-account-ratio-log-deviation-24h | positioning ratio | top-trader account long/short deviation from EMA(24h) | 24h | full 5m publication lag | official futures archive | sign_given_small_q50, sign_given_small_q75 |

### 15m

Confirmed component heads: 10.

| Input | Family | Parameters | Lookback | Causal delay | Availability | Used by heads |
|---|---|---|---|---|---|---|
| active-count-60s | activity | window=60s | 60s | through origin | core candle-derived | large_q75_given_negative, large_q90_given_negative, large_q50_given_positive |
| completed-1h-log-volume | volume regime | UTC-aligned 1h | 1h | after hour close | core candle-derived | large_q75_given_active, large_q50_given_positive |
| range-1m | minute candle shape | 10000*log(high/low) | 1m | latest completed minute | core candle-derived | joint_zero_sign_magnitude |
| range-1s | candle shape | 10000*log(high/low) | 1s | latest completed second | core candle-derived | sign_given_large_q50 |
| realized-volatility-30m | minute volatility | sqrt(sum(r_1m^2)),window=30m | 30m | through origin | core candle-derived | large_q90_given_positive |
| realized-volatility-60m | minute volatility | sqrt(sum(r_1m^2)),window=60m | 60m | through origin | core candle-derived | joint_zero_sign_magnitude, large_q75_given_active, large_q90_given_active, large_q90_given_negative |
| doge-realized-volatility-60m | cross-market volatility | window=60m | 60m | through origin | official alt kline archive | large_q90_given_positive |
| doge-return-5m | cross-market return | window=5m | 5m | through origin | official alt kline archive | sign_given_large_q50, sign_given_large_q75 |
| eth-realized-volatility-60m | cross-market volatility | window=60m | 60m | through origin | official alt kline archive | joint_zero_sign_magnitude, large_q50_given_negative, large_q75_given_negative |
| eth-return-1m | cross-market return | window=1m | 1m | latest completed minute | official alt kline archive | large_q90_given_positive |
| eth-return-5m | cross-market return | window=5m | 5m | through origin | official alt kline archive | large_q90_given_negative |
| futures-close-location-1m | futures candle | futures close location, last completed 1m | 1m | after 1m candle close | official futures archive | large_q50_given_positive |
| futures-log-trade-count-1m | futures activity | futures log trade count, last completed 1m | 1m | after 1m candle close | official futures archive | large_q75_given_active, large_q90_given_active |
| open-interest-value-log-change-15m | open interest | open interest value log change, 15m | 15m | full 5m publication lag | official futures archive | large_q50_given_negative |
| open-interest-value-log-change-60m | open interest | open interest value log change, 60m | 60m | full 5m publication lag | official futures archive | large_q90_given_active |
| spot-flow-quote-imbalance-ema-128 | aggressor direction | spot taker quote imbalance EMA(128s) | latest completed observation | latest completed second | official aggregate-trade archive | sign_given_large_q75 |
| spot-flow-vwap-gap-1s | price pressure | spot buyer-minus-seller VWAP gap, last 1s | 1s | latest completed second | official aggregate-trade archive | sign_given_large_q50, sign_given_large_q75 |
| top-account-minus-global-log | positioning spread | top-account minus global log ratio | latest completed observation | full 5m publication lag | official futures archive | large_q50_given_negative, large_q75_given_negative |

### 1h

Confirmed component heads: 8.

| Input | Family | Parameters | Lookback | Causal delay | Availability | Used by heads |
|---|---|---|---|---|---|---|
| active-count-60s | activity | window=60s | 60s | through origin | core candle-derived | large_q75_given_negative |
| range-1m | minute candle shape | 10000*log(high/low) | 1m | latest completed minute | core candle-derived | large_q25_given_active, large_q75_given_active |
| range-1s | candle shape | 10000*log(high/low) | 1s | latest completed second | core candle-derived | large_q90_given_active |
| realized-volatility-240m | minute volatility | sqrt(sum(r_1m^2)),window=240m | 240m | through origin | core candle-derived | large_q25_given_active, sign_given_large_q50 |
| realized-volatility-30m | minute volatility | sqrt(sum(r_1m^2)),window=30m | 30m | through origin | core candle-derived | large_q90_given_active |
| doge-realized-volatility-30m | cross-market volatility | window=30m | 30m | through origin | official alt kline archive | large_q50_given_negative, large_q90_given_negative |
| futures-quote-volume-surprise-1m | futures activity | futures quote-volume surprise vs EMA(60m) | 1m | after 1m candle close | official futures archive | inactive |
| open-interest-value-log-change-5m | open interest | open interest value log change, 5m | 5m | full 5m publication lag | official futures archive | large_q75_given_active |
| spot-flow-quote-imbalance-ema-128 | aggressor direction | spot taker quote imbalance EMA(128s) | latest completed observation | latest completed second | official aggregate-trade archive | sign_given_large_q50 |
| top-account-minus-global-log | positioning spread | top-account minus global log ratio | latest completed observation | full 5m publication lag | official futures archive | large_q50_given_negative, large_q75_given_negative, large_q90_given_negative |

Optional scarce overlays (the model must also support a missing mask and an archive-only fallback):

| spot-book-delta-log-quantity-l10 | spot book book change | snapshot change in log top-10 quantity | latest snapshot | strictly before boundary; maximum age 5s | optional local snapshot | sign_given_large_q50 |

## Backfill ledger

| Artifact | Rows | First | Last | Checksum | Failed series |
|---|---:|---|---|---|---:|
| binance-btcusdt-funding-rates.json | 5,925 | 2021-03-24T00:00:00.002Z | 2026-08-19T16:00:00.004Z | yes | 0 |
| coinmetrics-btc-network-flows-1d.json | 1,974 | 2021-03-24T00:00:00.000Z | 2026-08-18T00:00:00.000Z | yes | 0 |
| community-crypto-market-daily.json | 1,362 | 2022-11-27T00:00:00.000Z | 2026-08-19T00:00:00.000Z | yes | 0 |
| deribit-btc-dvol-1h.json | 47,396 | 2021-03-24T00:00:00.000Z | 2026-08-19T19:00:00.000Z | yes | 0 |
| fred-cboe-vix-1d.json | 1,389 | 2021-03-24T00:00:00.000Z | 2026-08-18T00:00:00.000Z | yes | 0 |
| global-macro-state.json | 21,733 | 2020-02-18T00:00:00.000Z | 2026-08-19T00:00:00.000Z | yes | 0 |
| mempool-btc-mining-proxies-3y.json | 2,195 | 2023-08-19T21:00:14.000Z | 2026-08-19T15:52:09.000Z | yes | 0 |

Binance archive days 2026-08-17 and 2026-08-18 were added for BTC 1s/1m, ETH/SOL/BNB/DOGE 1m, spot aggregate trades, futures klines, and futures metrics. 2026-08-19 daily archives were not yet published; no partial day was stored as a complete archive.

The official Binance futures book-depth scope is complete at 368 available days. Tardis has 10/10/10/10/10/10 reduced monthly samples per source.

GDELT remains checkpointed after 1,162 valid rows through 2026-08-02T00:00:00.000Z; the API returned HTTP 429. missing intervals remain missing; incomplete checkpoint is not treated as a finished dataset.

## What is deliberately excluded

- Macro, funding, DVOL, VIX, Coin Metrics, community flows, and mempool proxies are not required inputs. Recent macro discoveries did not survive the 2021-2026 conditional stability test; all other slow families had no stable selected addition.
- Dense EMA/RSI variants are not concatenated. All 3,471 variants per larger target were screened, but none produced a stable conditional addition after the volatility/range basis.
- The 1m FrFT entropy addition is below the 0.001-bit tolerance; the 1h FrFT candidate fails transfer. The 1s negative-transfer FFT magnitude feature is excluded.
- The 76 clean live inputs remain an experimental overlay. They cover only about 20 hours and did not rescue a broad component head that lacked confirmation at the matching 1s or 60s horizon.

## Limits

- The broad component search scores all coordinates marginally, then searches every subset up to size three among 12 family-aware finalists; it is not a proof over arbitrary transforms or larger subsets.
- Slow public sources cannot be identified at genuinely independent 1s cadence; their 1s role is regime conditioning and is redundant with contemporaneous volatility in current evidence.
- The 30-day and roughly 20-hour confirmations are regime-limited. Live-only inputs remain optional until multiple independent windows accrue.

Machine-readable audit: `data/benchmarks/all-feature-availability-audit.json`.
Detailed per-head broad search: `docs/experiments/tiered-component-feature-bases-2026-08-19.md`.
Detailed live-only search: `docs/experiments/live-component-feature-bases-2026-08-19.md`.

