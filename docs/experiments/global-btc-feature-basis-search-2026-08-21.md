# Global BTC predictive feature-basis search

Generated `2026-08-21T13:40:05.895058+00:00` from the canonical registry, chronological search, transfer robustness audit, and full streamed KKT scan.

## Decision

The robust production union contains **471 raw inputs**. Every promoted correction is KKT-certified over the complete eligible registry; rejected horizons keep their incumbent inputs.

A correction is retained only when its optimizer converged, it did not regress by more than 0.001 bits on transfer, and the 95% whole-day bootstrap interval for its transfer gain is wholly positive. If quality is statistically indistinguishable, the smaller incumbent wins.

The final correction model serializes only the promoted 1s and 1m heads. The 15m and 1h rows below are carried from the prior four-horizon audit because their broad corrections failed transfer; their smaller incumbent inputs remain the production contract.

Among incumbent-safe paths, statistical equivalence uses the paired one-standard-error rule on chronological-fold gains; availability, acquisition cost, and support size then choose the operationally preferable path.

## Exact candidate registry

| Quantity | Count |
|---|---:|
| Assets in union | 261 |
| Raw ledger coordinates | 1,000,171 |
| Duplicate aliases/overlaps | 10,087 |
| Canonical unique coordinates | **990,084** |
| Canonical templates | 4,903 |

| Inventory | Raw coordinates |
|---|---:|
| coinmetrics | 37 |
| community-flows | 145 |
| cross-market-public | 104 |
| dense-minute-indicators | 892,047 |
| deribit-option-surface | 20 |
| dvol | 17 |
| fast-live | 76 |
| full-funding-grid | 4,242 |
| gdelt-news | 8 |
| global-macro | 309 |
| long-endogenous | 8,226 |
| mempool-proxy | 36 |
| representative-cross-asset | 31,043 |
| second-technical | 23,240 |
| spectral-1m | 26,214 |
| spectral-1s | 14,280 |
| tardis-cross-venue | 118 |
| vix | 9 |

The 3,471 dense minute variants already include the declared lag grid. Templates expand only over assets with source coverage; the registry is therefore not `261 × templates`.

## Search and validation design

Model class: incumbent smoothed joint-state log probabilities plus an additive four-bin group-lasso log-odds correction.

Candidate working set: **4,748** active-set coordinates, constructed from full-registry gradient screens while always retaining the incumbent inputs. The full registry is streamed for KKT checks rather than materialized as a dense 42,901 × 990,084 matrix.

Selection protocol: three expanding chronological folds; each horizon independently must remain within 0.001 bits of its incumbent on every fold. The seven-day transfer segment is confirmation-only and never selects a penalty or support.

| Horizon | λ / λmax | Mean fold bits | Worst fold bits | Fold support union | Final correction groups | Final stationarity | Converged |
|---|---:|---:|---:|---:|---:|---:|---|
| 1s | 0.1900 | 0.433565 | 0.417303 | 198 | 148 | 0.00009910 | yes |
| 1m | 0.2500 | 0.237896 | 0.193351 | 984 | 335 | 0.00009816 | yes |
| 15m | 0.3500 | 0.196716 | 0.158106 | 825 | 718 | 0.00009487 | yes |
| 1h | 0.2900 | 0.252363 | 0.170267 | 586 | 480 | 0.00009257 | yes |

### Predictive-quality versus operational tie-break

The best-mean path is shown before the predeclared one-standard-error equivalence rule. Within that equivalence set, selection uses bottleneck availability, maximum acquisition cost, and support size in that order; mean availability is secondary because adding an always-present input cannot improve joint model availability.

| Horizon | Best-mean λ | Best mean bits | Operational λ | Operational mean bits | Mean-bit difference | Operational fold-union size |
|---|---:|---:|---:|---:|---:|---:|
| 1s | 0.0900 | 0.464847 | 0.1900 | 0.433565 | -0.031282 | 198 |
| 1m | 0.2200 | 0.241469 | 0.2500 | 0.237896 | -0.003573 | 984 |
| 15m | 0.2900 | 0.208847 | 0.3500 | 0.196716 | -0.012131 | 825 |
| 1h | 0.1600 | 0.311874 | 0.2900 | 0.252363 | -0.059511 | 586 |

## Transfer robustness

| Horizon | Gain over incumbent, bits | Positive days | Day-bootstrap 95% interval | P(gain > 0) | Promote correction |
|---|---:|---:|---:|---:|---|
| 1s | 0.121694 | 7/7 | [0.089497, 0.147817] | 1.0000 | yes |
| 1m | 0.039923 | 7/7 | [0.014661, 0.070821] | 1.0000 | yes |
| 15m | -0.112224 | 0/7 | [-0.141570, -0.079012] | 0.0000 | no; keep incumbent |
| 1h | -0.091794 | 2/7 | [-0.194861, 0.002703] | 0.0293 | no; keep incumbent |

## Production input contract by prediction horizon

Every row is a completed, causally available value at the prediction origin. Missingness/age channels named explicitly in an ID are inputs, not imputed future observations.

### 1s: 150 inputs

| Coordinate | Family | Construction | Lookback | Delay | Source policy | Declared availability | Empirical availability | Acquisition cost |
|---|---|---|---|---|---|---:|---:|---:|
| `asset/ake/binance-preferred/1m/ema-slope-2048m-512m-lag-1440m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.972 | 0 |
| `asset/arm/binance-preferred/1m/fft-log-energy-16m` | Fourier energy | Hann-windowed rFFT of signed log returns | 16m | through latest completed candle | candle-derived | 1.000 | 1.000 | 0 |
| `asset/asml/binance-preferred/1m/log-quote-volume-1m` | cross-asset activity regime | log1p quote volume | 1m | latest completed minute | binance-archive | 0.960 | 1.000 | 1 |
| `asset/axl/binance-preferred/1m/ema-slope-8192m-128m-lag-1440m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.972 | 0 |
| `asset/b2/binance-preferred/1m/ema-slope-512m-1024m-lag-30m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/b2/binance-preferred/1m/ema-slope-512m-1024m-lag-60m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/bnc/binance-preferred/1m/ema-slope-2048m-1024m-lag-480m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.994 | 0 |
| `asset/bnc/binance-preferred/1m/ema-slope-8192m-256m-lag-60m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/broccolif3b/binance-preferred/1m/ema-slope-8192m-2m-lag-60m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-preferred/1m/ema-slope-4096m-2m-lag-1m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-preferred/1m/ema-slope-4096m-4m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-preferred/1m/range-1m` | minute candle shape | 10000*log(high/low) | 1m | latest completed minute | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-preferred/1m/realized-volatility-15m` | minute volatility | sqrt(sum(r_1m^2)),window=15m | 15m | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-preferred/1m/realized-volatility-60m` | minute volatility | sqrt(sum(r_1m^2)),window=60m | 60m | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-preferred/1m/rsi-8192m` | RSI | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-spot/1m/spot-book-spread-bps` | spot book top of book | bid-ask spread | latest snapshot | strictly before boundary; maximum age 5s | live-only | 0.450 | — | 5 |
| `asset/btc/binance-spot/1m/spot-flow-trade-imbalance-ema-2` | aggressor direction | spot raw-trade imbalance EMA(2s) | latest completed observation | latest completed second | binance-archive | 0.960 | 1.000 | 1 |
| `asset/btc/binance-spot/1s/ema-acceleration-2s-1s` | cross-asset fast EMA acceleration | ema acceleration 2s 1s | recursive | latest completed second at minute origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-spot/1s/ema-acceleration-512s-2s` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-spot/1s/ema-acceleration-8192s-2s` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-spot/1s/ema-distance-2s` | EMA | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-spot/1s/ema-distance-4s` | EMA | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-spot/1s/ema-slope-2s-1s` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-spot/1s/ema-slope-2s-2s` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-spot/1s/ema-slope-2s-4s` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-spot/1s/ema-slope-4s-1s` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-spot/1s/fft-log-energy-16s` | Fourier energy | Hann-windowed rFFT of signed log returns | 16s | through latest completed candle | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-spot/1s/fft-log-energy-256s` | Fourier energy | Hann-windowed rFFT of signed log returns | 256s | through latest completed candle | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-spot/1s/range-1s` | candle shape | 10000*log(high/low) | 1s | latest completed second | binance-archive | 0.960 | 1.000 | 1 |
| `asset/btc/binance-spot/1s/realized-volatility-60s` | volatility | sqrt(sum(r_1s^2)) | 60s | through origin | binance-archive | 0.960 | 1.000 | 1 |
| `asset/btc/binance-spot/1s/spot-flow-first-side-1s` | trade sequence | spot first aggressor side, last 1s | 1s | latest completed second | binance-archive | 0.960 | 1.000 | 1 |
| `asset/btc/binance-spot/1s/spot-flow-flip-rate-1s` | trade sequence | spot aggressor-side flip rate, last 1s | 1s | latest completed second | binance-archive | 0.960 | 1.000 | 1 |
| `asset/btc/binance-spot/1s/spot-flow-last-side-1s` | trade sequence | spot last aggressor side, last 1s | 1s | latest completed second | binance-archive | 0.960 | 1.000 | 1 |
| `asset/btc/binance-spot/1s/spot-flow-last-side-lag-2s` | lagged trade sequence | spot last aggressor side, 2s old | 2s | latest completed second | binance-archive | 0.960 | 1.000 | 1 |
| `asset/btc/binance-spot/1s/spot-flow-log-trade-count-1s` | activity | spot log raw-trade count, last 1s | 1s | latest completed second | binance-archive | 0.960 | 1.000 | 1 |
| `asset/btc/binance-spot/1s/spot-flow-raw-per-aggregate-1s` | trade structure | spot raw trades per aggregate, last 1s | 1s | latest completed second | binance-archive | 0.960 | 1.000 | 1 |
| `asset/btc/binance-spot/1s/spot-flow-trade-surprise-1s` | activity | spot trade-count surprise vs EMA(32s) | 1s | latest completed second | binance-archive | 0.960 | 1.000 | 1 |
| `asset/btc/binance-usdm/1m/futures-basis-change-1m` | basis | futures basis change, 1m | 1m | after 1m candle close | binance-futures-archive | 0.950 | 1.000 | 1 |
| `asset/btc/binance-usdm/1m/futures-basis-deviation-15m` | basis | futures basis deviation from EMA(15m) | 15m | after 1m candle close | binance-futures-archive | 0.950 | 1.000 | 1 |
| `asset/btc/binance-usdm/1m/futures-basis-deviation-5m` | basis | futures basis deviation from EMA(5m) | 5m | after 1m candle close | binance-futures-archive | 0.950 | 1.000 | 1 |
| `asset/btc/binance-usdm/1m/futures-basis-deviation-60m` | basis | futures basis deviation from EMA(60m) | 60m | after 1m candle close | binance-futures-archive | 0.950 | 1.000 | 1 |
| `asset/btc/binance-usdm/1m/futures-close-location-1m` | futures candle | futures close location, last completed 1m | 1m | after 1m candle close | binance-futures-archive | 0.950 | 1.000 | 1 |
| `asset/btc/binance-usdm/1m/futures-log-quote-volume-1m` | futures activity | futures log quote volume, last completed 1m | 1m | after 1m candle close | binance-futures-archive | 0.950 | 1.000 | 1 |
| `asset/btc/binance-usdm/1m/futures-log-trade-count-1m` | futures activity | futures log trade count, last completed 1m | 1m | after 1m candle close | binance-futures-archive | 0.950 | 1.000 | 1 |
| `asset/btc/binance-usdm/1m/futures-range-1m` | futures candle | futures range, last completed 1m | 1m | after 1m candle close | binance-futures-archive | 0.950 | 1.000 | 1 |
| `asset/btc/binance-usdm/1m/futures-relative-return-1m` | cross-market return | futures-minus-spot return, last completed 1m | 1m | after 1m candle close | binance-futures-archive | 0.950 | 1.000 | 1 |
| `asset/cat/binance-preferred/1m/ema-slope-4096m-1024m-lag-1440m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.972 | 0 |
| `asset/comp/binance-preferred/1m/ema-slope-8192m-1024m-lag-120m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/comp/binance-preferred/1m/ema-slope-8192m-1024m-lag-60m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/crwd/binance-preferred/1m/ema-slope-2048m-1024m-lag-15m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/crwd/binance-preferred/1m/ema-slope-2048m-1024m-lag-30m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/dgb/binance-preferred/1m/ema-slope-512m-1024m-lag-240m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/eth/binance-preferred/1m/ema-distance-8192m` | EMA value | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/eth/binance-preferred/1m/ema-distance-8192m-lag-2m` | EMA value | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/eth/binance-preferred/1m/ema-slope-8192m-16m-lag-1m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/eth/binance-preferred/1m/ema-slope-8192m-64m-lag-1m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/eth/binance-preferred/1m/ema-slope-8192m-64m-lag-60m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/eth/binance-preferred/1m/ema-slope-8192m-8m-lag-1m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/eth/binance-preferred/1m/realized-volatility-30m` | minute volatility | sqrt(sum(r_1m^2)),window=30m | 30m | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/eth/binance-spot/1s/ema-distance-16s` | EMA | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/eth/binance-spot/1s/ema-distance-2s` | EMA | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/eth/binance-spot/1s/ema-distance-4s` | EMA | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/eth/binance-spot/1s/ema-slope-2s-1s` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/eth/binance-spot/1s/ema-slope-2s-2s` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/eth/binance-spot/1s/ema-slope-4s-2s` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/eth/binance-spot/1s/fft-log-energy-256s` | Fourier energy | Hann-windowed rFFT of signed log returns | 256s | through latest completed candle | candle-derived | 1.000 | 1.000 | 0 |
| `asset/eth/binance-spot/1s/log-trade-count-1s` | cross-asset fast activity regime | log trade count 1s | 1s | latest completed second at minute origin | binance-archive | 0.960 | 1.000 | 1 |
| `asset/eth/binance-spot/1s/range-1s` | candle shape | 10000*log(high/low) | 1s | latest completed second | binance-archive | 0.960 | 1.000 | 1 |
| `asset/eth/binance-spot/1s/realized-volatility-5s` | cross-asset fast volatility | realized volatility 5s | 5s | latest completed second at minute origin | binance-archive | 0.960 | 1.000 | 1 |
| `asset/eth/binance-spot/1s/rsi-2s-ema-alpha-2-over-3` | cross-asset fast RSI | rsi 2s | recursive | latest completed second at minute origin | binance-archive | 0.960 | 1.000 | 1 |
| `asset/eth/binance-usdm/5m/top-position-minus-account-long-short` | cross-asset futures positioning | log top-position ratio minus log top-account ratio | latest 5m bucket | one completed 5m publication lag | binance-futures-archive | 0.950 | 1.000 | 1 |
| `asset/folks/binance-preferred/1m/ema-slope-4096m-512m-lag-240m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/folks/binance-preferred/1m/ema-slope-8192m-256m-lag-120m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/folks/binance-preferred/1m/ema-slope-8192m-256m-lag-8m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/glmr/binance-preferred/1m/ema-slope-2048m-512m-lag-1440m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.972 | 0 |
| `asset/hpe/binance-preferred/1m/range-1m` | minute candle shape | 10000*log(high/low) | 1m | latest completed minute | candle-derived | 1.000 | 1.000 | 0 |
| `asset/hpe/binance-preferred/1m/realized-volatility-60m` | minute volatility | sqrt(sum(r_1m^2)),window=60m | 60m | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/hype/binance-preferred/1m/realized-volatility-15m` | minute volatility | sqrt(sum(r_1m^2)),window=15m | 15m | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/ibm/binance-usdm/1m/book-log-depth-5pct` | cross-asset futures book shape | log depth 5pct | latest snapshot | latest snapshot in completed minute | binance-archive | 0.960 | 1.000 | 1 |
| `asset/ibm/binance-usdm/1m/book-log-notional-5pct` | cross-asset futures book shape | log notional 5pct | latest snapshot | latest snapshot in completed minute | binance-archive | 0.960 | 1.000 | 1 |
| `asset/ksm/binance-usdm/funding-event/funding-absolute-mean-9` | Funding pressure | — | 9 settlement(s), normally 3.0d | one minute after settlement | binance-futures-archive | 0.950 | 0.916 | 1 |
| `asset/met/binance-preferred/1m/ema-slope-8192m-1024m-lag-1m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/minimax/binance-usdm/funding-event/funding-mean-21` | Funding state | — | 21 settlement(s), normally 7.0d | one minute after settlement | binance-futures-archive | 0.950 | 0.782 | 1 |
| `asset/natgas/binance-preferred/1m/fft-log-energy-256m` | Fourier energy | Hann-windowed rFFT of signed log returns | 256m | through latest completed candle | candle-derived | 1.000 | 1.000 | 0 |
| `asset/natgas/binance-usdm/1m/book-log-notional-5pct` | cross-asset futures book shape | log notional 5pct | latest snapshot | latest snapshot in completed minute | binance-archive | 0.960 | 1.000 | 1 |
| `asset/night/binance-preferred/1m/ema-acceleration-8192m-1024m-lag-480m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 0.994 | 0 |
| `asset/nxpc/binance-preferred/1m/ema-slope-128m-1024m-lag-240m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/onds/binance-preferred/1m/ema-acceleration-4096m-1m-lag-120m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/onds/binance-preferred/1m/range-1m` | minute candle shape | 10000*log(high/low) | 1m | latest completed minute | candle-derived | 1.000 | 1.000 | 0 |
| `asset/onds/binance-preferred/1m/rsi-4096m` | RSI | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/onds/binance-preferred/1m/rsi-4096m-lag-1m` | RSI | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/openai/binance-preferred/1m/ema-slope-2048m-16m-lag-4m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/openai/binance-preferred/1m/ema-slope-2048m-8m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/orcl/binance-preferred/1m/active-fraction-60m` | cross-asset activity | fraction of nonzero minute returns | 60m | latest completed minute | binance-archive | 0.960 | 1.000 | 1 |
| `asset/pumpbtc/binance-usdm/funding-event/funding-absolute-mean-9` | Funding pressure | — | 9 settlement(s), normally 3.0d | one minute after settlement | binance-futures-archive | 0.950 | 0.961 | 1 |
| `asset/qkc/binance-preferred/1m/ema-slope-8192m-1024m-lag-120m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/qkc/binance-preferred/1m/ema-slope-8192m-1024m-lag-30m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/qkc/binance-preferred/1m/ema-slope-8192m-1024m-lag-4m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/qkc/binance-preferred/1m/ema-slope-8192m-1024m-lag-60m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/qkc/binance-preferred/1m/ema-slope-8192m-512m-lag-60m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/ray/binance-preferred/1m/ema-slope-2048m-32m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/rif/binance-usdm/5m/top-position-long-short-log-level` | cross-asset futures positioning | log top-position-long-short | latest 5m bucket | one completed 5m publication lag | binance-futures-archive | 0.950 | 1.000 | 1 |
| `asset/rivn/binance-preferred/1m/ema-slope-4096m-32m-lag-1440m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.972 | 0 |
| `asset/rivn/binance-preferred/1m/realized-volatility-5m` | cross-asset volatility | sqrt sum squared minute log returns | 5m | latest completed minute | candle-derived | 1.000 | 1.000 | 0 |
| `asset/sapien/binance-preferred/1m/ema-acceleration-4096m-1024m-lag-120m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/sapien/binance-preferred/1m/ema-slope-2048m-1024m-lag-960m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.983 | 0 |
| `asset/sats/binance-preferred/1m/ema-acceleration-8192m-1m-lag-1440m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 0.972 | 0 |
| `asset/sats/binance-preferred/1m/ema-distance-4096m-lag-1440m` | EMA value | — | recursive | through origin | candle-derived | 1.000 | 0.972 | 0 |
| `asset/sats/binance-preferred/1m/ema-slope-2048m-4m-lag-1440m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.972 | 0 |
| `asset/sats/binance-preferred/1m/ema-slope-4096m-16m-lag-1440m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.972 | 0 |
| `asset/sats/binance-preferred/1m/ema-slope-4096m-512m-lag-1440m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.972 | 0 |
| `asset/sats/binance-preferred/1m/ema-slope-4096m-8m-lag-1440m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.972 | 0 |
| `asset/sign/binance-preferred/1m/rsi-4096m-lag-240m` | RSI | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/sign/binance-preferred/1m/rsi-8192m-lag-120m` | RSI | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/sign/binance-preferred/1m/rsi-8192m-lag-2m` | RSI | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/snow/binance-usdm/funding-event/funding-absolute-mean-9` | Funding pressure | — | 9 settlement(s), normally 3.0d | one minute after settlement | binance-futures-archive | 0.950 | 0.916 | 1 |
| `asset/sol/binance-preferred/1m/realized-volatility-30m` | minute volatility | sqrt(sum(r_1m^2)),window=30m | 30m | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/soph/binance-usdm/5m/top-position-minus-account-long-short` | cross-asset futures positioning | log top-position ratio minus log top-account ratio | latest 5m bucket | one completed 5m publication lag | binance-futures-archive | 0.950 | 0.999 | 1 |
| `asset/sportfun/binance-usdm/funding-event/funding-absolute-mean-3` | Funding pressure | — | 3 settlement(s), normally 1.0d | one minute after settlement | binance-futures-archive | 0.950 | 0.994 | 1 |
| `asset/the/binance-usdm/funding-event/funding-absolute-mean-21` | Funding pressure | — | 21 settlement(s), normally 7.0d | one minute after settlement | binance-futures-archive | 0.950 | 0.894 | 1 |
| `asset/tko/binance-preferred/1m/ema-slope-128m-1024m-lag-240m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/truth/binance-preferred/1m/ema-slope-4096m-8m-lag-8m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/truth/binance-preferred/1m/ema-slope-8192m-64m-lag-60m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/tst/binance-preferred/1m/ema-slope-4096m-1024m-lag-1440m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.972 | 0 |
| `asset/tst/binance-preferred/1m/ema-slope-4096m-1024m-lag-960m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.983 | 0 |
| `asset/uai/binance-usdm/funding-event/funding-absolute-mean-9` | Funding pressure | — | 9 settlement(s), normally 3.0d | one minute after settlement | binance-futures-archive | 0.950 | 0.961 | 1 |
| `asset/ub/binance-preferred/1m/ema-slope-4096m-1024m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/ub/binance-preferred/1m/ema-slope-4096m-1024m-lag-1m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/ub/binance-preferred/1m/ema-slope-4096m-256m-lag-120m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/ub/binance-preferred/1m/ema-slope-4096m-512m-lag-60m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/v/binance-preferred/1m/ema-slope-8192m-4m-lag-30m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/v/binance-preferred/1m/ema-slope-8192m-64m-lag-15m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/v/binance-preferred/1m/ema-slope-8192m-8m-lag-15m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/vanry/binance-preferred/1m/ema-slope-8192m-256m-lag-240m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/vanry/binance-preferred/1m/ema-slope-8192m-256m-lag-480m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.994 | 0 |
| `asset/wal/binance-usdm/funding-event/funding-mean-9` | Funding state | — | 9 settlement(s), normally 3.0d | one minute after settlement | binance-futures-archive | 0.950 | 0.961 | 1 |
| `asset/wen/binance-preferred/1m/completed-5m-log-volume` | volume regime | UTC-aligned 5m | 5m | updates only after bar close | candle-derived | 1.000 | 1.000 | 0 |
| `asset/wen/binance-preferred/1m/ema-slope-4096m-512m-lag-960m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.983 | 0 |
| `asset/wen/binance-preferred/1m/ema-slope-8192m-1024m-lag-480m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.994 | 0 |
| `asset/wen/binance-preferred/1m/log-mean-trade-notional-1m` | cross-asset trade structure | log quote volume per trade | 1m | latest completed minute | binance-archive | 0.960 | 1.000 | 1 |
| `asset/wen/binance-preferred/1m/log-trade-count-1m` | cross-asset activity regime | log1p trade count | 1m | latest completed minute | binance-archive | 0.960 | 1.000 | 1 |
| `asset/xag/binance-preferred/1m/realized-volatility-30m` | minute volatility | sqrt(sum(r_1m^2)),window=30m | 30m | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/xag/binance-usdm/5m/open-interest-log-level` | cross-asset futures positioning | log open-interest | latest 5m bucket | one completed 5m publication lag | binance-futures-archive | 0.950 | 0.999 | 1 |
| `asset/xag/binance-usdm/5m/open-interest-value-log-level` | cross-asset futures positioning | log open-interest-value | latest 5m bucket | one completed 5m publication lag | binance-futures-archive | 0.950 | 0.999 | 1 |
| `asset/xec/binance-preferred/1m/ema-slope-2048m-64m-lag-1m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/xrp/binance-preferred/1m/log-trade-count-1m` | cross-asset activity regime | log1p trade count | 1m | latest completed minute | binance-archive | 0.960 | 1.000 | 1 |
| `asset/zest/binance-usdm/5m/global-long-short-log-level` | cross-asset futures positioning | log global-long-short | latest 5m bucket | one completed 5m publication lag | binance-futures-archive | 0.950 | 0.999 | 1 |
| `asset/zhipu/binance-preferred/1m/rsi-4096m-lag-2m` | RSI | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/zhipu/binance-preferred/1m/rsi-4096m-lag-480m` | RSI | — | recursive | through origin | candle-derived | 1.000 | 0.994 | 0 |
| `asset/zhipu/binance-usdm/5m/top-account-long-short-log-level` | cross-asset futures positioning | log top-account-long-short | latest 5m bucket | one completed 5m publication lag | binance-futures-archive | 0.950 | 0.983 | 1 |

### 1m: 338 inputs

| Coordinate | Family | Construction | Lookback | Delay | Source policy | Declared availability | Empirical availability | Acquisition cost |
|---|---|---|---|---|---|---:|---:|---:|
| `asset/ace/binance-preferred/1m/ema-acceleration-2048m-64m-lag-2m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/ace/binance-preferred/1m/ema-slope-8m-128m-lag-8m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/ach/binance-preferred/1m/ema-acceleration-32m-1m-lag-8m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/ai/binance-preferred/1m/ema-acceleration-32m-2m-lag-60m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/ai/binance-preferred/1m/ema-slope-8m-2m-lag-60m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/ai/binance-preferred/1m/realized-volatility-60m` | minute volatility | sqrt(sum(r_1m^2)),window=60m | 60m | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/aia/binance-preferred/1m/ema-slope-512m-16m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/aigensyn/binance-preferred/1m/ema-acceleration-16m-4m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/ake/binance-preferred/1m/ema-acceleration-2m-64m-lag-15m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/alice/binance-preferred/1m/ema-acceleration-64m-4m-lag-4m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/allo/binance-preferred/1m/ema-acceleration-4096m-64m-lag-480m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 0.994 | 0 |
| `asset/allo/binance-preferred/1m/ema-acceleration-8192m-64m-lag-480m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 0.994 | 0 |
| `asset/alpine/binance-preferred/1m/ema-slope-4m-512m-lag-30m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/alpine/binance-preferred/1m/rsi-4m-lag-1440m` | RSI | — | recursive | through origin | candle-derived | 1.000 | 0.972 | 0 |
| `asset/anthropic/binance-preferred/1m/ema-acceleration-16m-32m-lag-480m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 0.994 | 0 |
| `asset/anthropic/binance-preferred/1m/ema-acceleration-8m-1m-lag-240m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/anthropic/binance-preferred/1m/ema-slope-8192m-32m-lag-1440m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.972 | 0 |
| `asset/anthropic/binance-preferred/1m/ema-slope-8192m-8m-lag-1440m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.972 | 0 |
| `asset/ark/binance-preferred/1m/ema-acceleration-128m-8m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/ark/binance-spot/1s/fft-dominant-frequency-256s` | Fourier shape | Hann-windowed rFFT of signed log returns | 256s | through latest completed candle | candle-derived | 1.000 | 1.000 | 0 |
| `asset/arm/binance-preferred/1m/ema-acceleration-8m-2m-lag-120m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/arm/binance-preferred/1m/ema-slope-32m-16m-lag-60m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/arm/binance-spot/1s/frft-0p5-entropy-256s` | fractional Fourier shape | Hann-windowed rFFT of signed log returns | 256s | through latest completed candle | candle-derived | 1.000 | 1.000 | 0 |
| `asset/arpa/binance-preferred/1m/ema-slope-128m-1024m-lag-1440m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.972 | 0 |
| `asset/arx/binance-preferred/1m/ema-slope-128m-256m-lag-120m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/asml/binance-preferred/1m/fft-k4-real-256m` | Fourier complex coefficients | Hann-windowed rFFT of signed log returns | 256m | through latest completed candle | candle-derived | 1.000 | 1.000 | 0 |
| `asset/asml/binance-preferred/1m/haar-mid-energy-share-16m` | wavelet energy shape | Hann-windowed rFFT of signed log returns | 16m | through latest completed candle | candle-derived | 1.000 | 1.000 | 0 |
| `asset/at/binance-preferred/1m/ema-acceleration-512m-1m-lag-960m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 0.983 | 0 |
| `asset/avgo/binance-preferred/1m/ema-acceleration-4m-128m-lag-1440m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 0.972 | 0 |
| `asset/avgo/binance-preferred/1m/ema-acceleration-8m-4m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/avgo/binance-preferred/1m/fft-log-energy-16m` | Fourier energy | Hann-windowed rFFT of signed log returns | 16m | through latest completed candle | candle-derived | 1.000 | 1.000 | 0 |
| `asset/avgo/binance-spot/1s/fft-log-energy-256s` | Fourier energy | Hann-windowed rFFT of signed log returns | 256s | through latest completed candle | candle-derived | 1.000 | 1.000 | 0 |
| `asset/awe/binance-preferred/1m/ema-slope-512m-1024m-lag-1440m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.972 | 0 |
| `asset/axl/binance-preferred/1m/ema-acceleration-4m-32m-lag-30m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/b2/binance-preferred/1m/ema-acceleration-2048m-2m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/b2/binance-preferred/1m/ema-acceleration-4096m-2m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/baba/binance-preferred/1m/ema-distance-16m-lag-1440m` | EMA value | — | recursive | through origin | candle-derived | 1.000 | 0.972 | 0 |
| `asset/baba/binance-preferred/1m/ema-slope-16m-1m-lag-1440m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.972 | 0 |
| `asset/ban/binance-preferred/1m/ema-slope-32m-16m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/bank/binance-preferred/1m/ema-acceleration-8192m-1024m-lag-240m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/bank/binance-preferred/1m/ema-slope-128m-16m-lag-8m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/bas/binance-preferred/1m/ema-acceleration-8m-4m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/bas/binance-preferred/1m/ema-slope-4096m-16m-lag-60m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/beat/binance-preferred/1m/ema-acceleration-16m-32m-lag-8m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/beat/binance-preferred/1m/ema-acceleration-4096m-512m-lag-120m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/beat/binance-preferred/1m/ema-slope-2m-4m-lag-30m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/bless/binance-preferred/1m/ema-acceleration-32m-2m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/bnc/binance-preferred/1m/ema-slope-2m-32m-lag-15m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/brkb/binance-preferred/1m/ema-slope-64m-64m-lag-960m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.983 | 0 |
| `asset/broccolif3b/binance-preferred/1m/ema-slope-2m-8m-lag-120m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/bsb/binance-preferred/1m/ema-acceleration-4m-64m-lag-8m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-preferred/1m/ema-acceleration-2048m-8m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-preferred/1m/ema-acceleration-8192m-1m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-preferred/1m/ema-acceleration-8192m-2m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-preferred/1m/ema-slope-2m-2m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-preferred/1m/ema-slope-2m-4m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-preferred/1m/ema-slope-64m-2m-lag-60m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-preferred/1m/realized-volatility-60m` | minute volatility | sqrt(sum(r_1m^2)),window=60m | 60m | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-preferred/1m/rsi-128m-lag-4m` | RSI | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-preferred/1m/rsi-14m-lag-60m` | RSI | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-spot/1s/ema-acceleration-128s-1s` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-spot/1s/ema-acceleration-512s-1s` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-spot/1s/ema-acceleration-8192s-2s` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-spot/1s/ema-distance-1024s` | EMA | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-spot/1s/ema-distance-256s` | EMA | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-spot/1s/ema-distance-2s` | EMA | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-spot/1s/ema-distance-4s` | EMA | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-spot/1s/ema-slope-2s-1s` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-spot/1s/ema-slope-2s-2s` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-spot/1s/ema-slope-4s-1s` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-spot/1s/ema-slope-8s-8s` | price dynamics | period=8s,horizon=8s | recursive | through origin | candle-derived | 1.000 | — | 0 |
| `asset/btc/binance-spot/1s/frft-0p5-entropy-256s` | fractional Fourier shape | Hann-windowed rFFT of signed log returns | 256s | through latest completed candle | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-spot/1s/macd-histogram-3-7-3-1s-cadence` | MACD | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-spot/1s/macd-histogram-6-13-5-1s-cadence` | MACD | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-spot/1s/rsi-4s` | RSI | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-spot/1s/rsi-8s` | RSI | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-spot/1s/spot-flow-last-side-1s` | trade sequence | spot last aggressor side, last 1s | 1s | latest completed second | binance-archive | 0.960 | 1.000 | 1 |
| `asset/btc/binance-usdm/1m/futures-basis-change-1m` | basis | futures basis change, 1m | 1m | after 1m candle close | binance-futures-archive | 0.950 | 1.000 | 1 |
| `asset/btc/binance-usdm/1m/futures-basis-deviation-15m` | basis | futures basis deviation from EMA(15m) | 15m | after 1m candle close | binance-futures-archive | 0.950 | 1.000 | 1 |
| `asset/btc/binance-usdm/1m/futures-basis-deviation-5m` | basis | futures basis deviation from EMA(5m) | 5m | after 1m candle close | binance-futures-archive | 0.950 | 1.000 | 1 |
| `asset/btc/binance-usdm/1m/futures-basis-deviation-60m` | basis | futures basis deviation from EMA(60m) | 60m | after 1m candle close | binance-futures-archive | 0.950 | 1.000 | 1 |
| `asset/btc/binance-usdm/1m/futures-close-location-1m` | futures candle | futures close location, last completed 1m | 1m | after 1m candle close | binance-futures-archive | 0.950 | 1.000 | 1 |
| `asset/btc/binance-usdm/1m/futures-log-trade-count-1m` | futures activity | futures log trade count, last completed 1m | 1m | after 1m candle close | binance-futures-archive | 0.950 | 1.000 | 1 |
| `asset/btc/binance-usdm/1m/futures-relative-return-1m` | cross-market return | futures-minus-spot return, last completed 1m | 1m | after 1m candle close | binance-futures-archive | 0.950 | 1.000 | 1 |
| `asset/bulla/binance-preferred/1m/ema-acceleration-2m-1m-lag-960m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 0.983 | 0 |
| `asset/bx/binance-preferred/1m/fft-log-energy-16m` | Fourier energy | Hann-windowed rFFT of signed log returns | 16m | through latest completed candle | candle-derived | 1.000 | 1.000 | 0 |
| `asset/bx/binance-preferred/1m/realized-volatility-2m` | volatility | sqrt(sum(r_1m^2)), window=2m | 2m | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/bz/binance-preferred/1m/ema-slope-4m-2m-lag-960m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.983 | 0 |
| `asset/cat/binance-preferred/1m/ema-acceleration-2048m-128m-lag-120m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/chip/binance-preferred/1m/ema-acceleration-2048m-2m-lag-30m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/chip/binance-spot/1s/frft-0p25-entropy-64s` | fractional Fourier shape | Hann-windowed rFFT of signed log returns | 64s | through latest completed candle | candle-derived | 1.000 | 1.000 | 0 |
| `asset/city/binance-preferred/1m/ema-slope-4m-1024m-lag-240m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/comp/binance-preferred/1m/realized-volatility-5m` | cross-asset volatility | sqrt sum squared minute log returns | 5m | latest completed minute | candle-derived | 1.000 | 1.000 | 0 |
| `asset/crwd/binance-preferred/1m/ema-slope-8m-4m-lag-1m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/ctr/binance-preferred/1m/ema-slope-16m-4m-lag-1m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/ctsi/binance-preferred/1m/ema-slope-128m-32m-lag-120m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/dcr/binance-preferred/1m/ema-acceleration-32m-1m-lag-2m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/dcr/binance-preferred/1m/ema-slope-128m-8m-lag-30m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/dexe/binance-spot/1s/realized-volatility-5s` | cross-asset fast volatility | realized volatility 5s | 5s | latest completed second at minute origin | binance-archive | 0.960 | 1.000 | 1 |
| `asset/dgb/binance-preferred/1m/ema-slope-2m-256m-lag-960m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.983 | 0 |
| `asset/dgb/binance-preferred/1m/ema-slope-4m-16m-lag-2m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/dgb/binance-preferred/1m/ema-slope-8m-32m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/dgb/binance-preferred/1m/ema-slope-8m-32m-lag-1m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/dolo/binance-preferred/1m/fft-dominant-frequency-16m` | Fourier shape | Hann-windowed rFFT of signed log returns | 16m | through latest completed candle | candle-derived | 1.000 | 1.000 | 0 |
| `asset/eden/binance-spot/1s/frft-0p25-entropy-256s` | fractional Fourier shape | Hann-windowed rFFT of signed log returns | 256s | through latest completed candle | candle-derived | 1.000 | 1.000 | 0 |
| `asset/era/binance-preferred/1m/ema-acceleration-16m-512m-lag-480m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 0.994 | 0 |
| `asset/esports/binance-preferred/1m/ema-acceleration-128m-512m-lag-4m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/esports/binance-preferred/1m/ema-acceleration-128m-512m-lag-8m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/esports/binance-preferred/1m/ema-slope-2048m-512m-lag-120m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/esports/binance-preferred/1m/ema-slope-4096m-256m-lag-30m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/eth/binance-preferred/1m/completed-5m-log-volume` | volume regime | UTC-aligned 5m | 5m | updates only after bar close | candle-derived | 1.000 | 1.000 | 0 |
| `asset/eth/binance-preferred/1m/ema-slope-128m-1024m-lag-960m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.983 | 0 |
| `asset/eth/binance-preferred/1m/return-5m` | cross-asset return | log close return over 5 completed minutes | 5m | latest completed minute | candle-derived | 1.000 | 1.000 | 0 |
| `asset/eth/binance-spot/1s/ema-acceleration-128s-2s` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/eth/binance-spot/1s/ema-distance-16s` | EMA | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/eth/binance-spot/1s/ema-distance-2s` | EMA | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/eth/binance-spot/1s/ema-distance-4s` | EMA | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/eth/binance-spot/1s/ema-distance-8s` | EMA | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/eth/binance-spot/1s/ema-log-change-8s-ema-over-8s` | cross-asset fast EMA slope | ema slope 8s 8s | recursive | latest completed second at minute origin | binance-archive | 0.960 | 1.000 | 1 |
| `asset/eth/binance-spot/1s/ema-slope-2s-1s` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/eth/binance-spot/1s/ema-slope-2s-4s` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/eth/binance-spot/1s/ema-slope-8s-8s` | price dynamics | period=8s,horizon=8s | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/eth/binance-spot/1s/macd-line-3-7-3-1s-cadence` | MACD | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/eth/binance-spot/1s/zero-run-age-1s` | cross-asset fast activity | zero run age 1s | recursive | latest completed second at minute origin | binance-archive | 0.960 | 1.000 | 1 |
| `asset/evaa/binance-preferred/1m/ema-acceleration-4096m-512m-lag-480m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 0.994 | 0 |
| `asset/evaa/binance-preferred/1m/ema-slope-128m-16m-lag-1m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/evaa/binance-preferred/1m/ema-slope-512m-16m-lag-2m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/evaa/binance-preferred/1m/rsi-512m` | RSI | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/fight/binance-preferred/1m/ema-slope-512m-8m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/folks/binance-preferred/1m/ema-slope-2048m-32m-lag-120m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/form/binance-preferred/1m/ema-acceleration-8m-16m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/fwdi/binance-preferred/1m/ema-acceleration-16m-16m-lag-4m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/g/binance-preferred/1m/ema-acceleration-8m-1m-lag-480m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 0.994 | 0 |
| `asset/gas/binance-preferred/1m/ema-acceleration-64m-4m-lag-120m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/genius/binance-preferred/1m/ema-slope-4096m-512m-lag-960m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.983 | 0 |
| `asset/glmr/binance-preferred/1m/ema-acceleration-64m-128m-lag-60m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/glw/binance-preferred/1m/ema-slope-64m-128m-lag-120m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/glw/binance-preferred/1m/log-quote-volume-1m` | cross-asset activity regime | log1p quote volume | 1m | latest completed minute | binance-archive | 0.960 | 1.000 | 1 |
| `asset/glw/binance-spot/1s/fft-log-energy-256s` | Fourier energy | Hann-windowed rFFT of signed log returns | 256s | through latest completed candle | candle-derived | 1.000 | 1.000 | 0 |
| `asset/gram/binance-preferred/1m/ema-acceleration-2048m-64m-lag-120m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/gram/binance-preferred/1m/ema-slope-128m-32m-lag-120m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/gua/binance-preferred/1m/ema-slope-64m-32m-lag-1m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/hana/binance-preferred/1m/ema-slope-16m-64m-lag-8m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/hd/binance-preferred/1m/ema-acceleration-512m-16m-lag-2m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/hei/binance-preferred/1m/ema-acceleration-128m-512m-lag-240m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/hemi/binance-preferred/1m/ema-acceleration-8m-1m-lag-60m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/hemi/binance-preferred/1m/ema-slope-32m-256m-lag-240m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/hmstr/binance-spot/1s/fft-dominant-frequency-64s` | Fourier shape | Hann-windowed rFFT of signed log returns | 64s | through latest completed candle | candle-derived | 1.000 | 1.000 | 0 |
| `asset/hpe/binance-preferred/1m/realized-volatility-30m` | minute volatility | sqrt(sum(r_1m^2)),window=30m | 30m | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/huma/binance-preferred/1m/ema-acceleration-4m-128m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/huma/binance-preferred/1m/ema-acceleration-64m-16m-lag-8m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/hype/binance-preferred/1m/ema-acceleration-2m-16m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/hype/binance-preferred/1m/ema-slope-2m-2m-lag-1m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/hype/binance-preferred/1m/ema-slope-64m-32m-lag-480m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.994 | 0 |
| `asset/hyundai/binance-preferred/1m/realized-volatility-5m` | cross-asset volatility | sqrt sum squared minute log returns | 5m | latest completed minute | candle-derived | 1.000 | 1.000 | 0 |
| `asset/ibm/binance-preferred/1m/ema-acceleration-16m-32m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/ibm/binance-preferred/1m/ema-slope-4m-2m-lag-60m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/ibm/binance-spot/1s/frft-0p5-entropy-256s` | fractional Fourier shape | Hann-windowed rFFT of signed log returns | 256s | through latest completed candle | candle-derived | 1.000 | 1.000 | 0 |
| `asset/icnt/binance-preferred/1m/ema-acceleration-2048m-32m-lag-30m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/icnt/binance-preferred/1m/ema-slope-512m-512m-lag-1440m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.972 | 0 |
| `asset/icx/binance-preferred/1m/ema-acceleration-2048m-1m-lag-240m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/icx/binance-preferred/1m/ema-acceleration-2048m-2m-lag-240m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/idol/binance-preferred/1m/morlet-slow-imag-256m` | complex wavelet coefficients | Hann-windowed rFFT of signed log returns | 256m | through latest completed candle | candle-derived | 1.000 | 1.000 | 0 |
| `asset/ilv/binance-preferred/1m/ema-slope-128m-32m-lag-120m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/iota/binance-preferred/1m/ema-acceleration-32m-4m-lag-480m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 0.994 | 0 |
| `asset/iota/binance-preferred/1m/ema-acceleration-4m-8m-lag-30m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/iota/binance-preferred/1m/ema-slope-32m-32m-lag-4m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/iotx/binance-preferred/1m/ema-acceleration-4096m-8m-lag-120m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/iotx/binance-preferred/1m/ema-slope-8192m-32m-lag-1440m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.972 | 0 |
| `asset/iotx/binance-preferred/1m/realized-volatility-5m` | cross-asset volatility | sqrt sum squared minute log returns | 5m | latest completed minute | candle-derived | 1.000 | 1.000 | 0 |
| `asset/irys/binance-preferred/1m/ema-slope-2m-8m-lag-15m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/joe/binance-preferred/1m/ema-acceleration-4096m-128m-lag-30m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/jto/binance-preferred/1m/ema-distance-64m-lag-480m` | EMA value | — | recursive | through origin | candle-derived | 1.000 | 0.994 | 0 |
| `asset/juv/binance-preferred/1m/rsi-4m` | RSI | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/koma/binance-preferred/1m/ema-acceleration-64m-64m-lag-480m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 0.994 | 0 |
| `asset/koma/binance-preferred/1m/ema-acceleration-8192m-256m-lag-960m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 0.983 | 0 |
| `asset/koma/binance-preferred/1m/ema-slope-4096m-1024m-lag-120m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/kstr/binance-preferred/1m/ema-acceleration-16m-1m-lag-240m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/kstr/binance-preferred/1m/ema-slope-64m-512m-lag-480m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.994 | 0 |
| `asset/light/binance-preferred/1m/ema-acceleration-4m-1m-lag-1m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/lyn/binance-preferred/1m/rsi-4096m-lag-240m` | RSI | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/manta/binance-preferred/1m/ema-acceleration-4m-512m-lag-960m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 0.983 | 0 |
| `asset/mbl/binance-preferred/1m/ema-slope-512m-64m-lag-30m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/mbl/binance-preferred/1m/rsi-8192m-lag-120m` | RSI | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/melania/binance-preferred/1m/ema-slope-128m-4m-lag-4m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/melania/binance-preferred/1m/ema-slope-128m-8m-lag-2m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/met/binance-preferred/1m/ema-slope-4m-32m-lag-240m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/metis/binance-preferred/1m/ema-slope-16m-4m-lag-60m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/mira/binance-preferred/1m/ema-acceleration-32m-1m-lag-4m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/mira/binance-preferred/1m/morlet-slow-real-16m` | complex wavelet coefficients | Hann-windowed rFFT of signed log returns | 16m | through latest completed candle | candle-derived | 1.000 | 1.000 | 0 |
| `asset/mito/binance-preferred/1m/ema-acceleration-4096m-2m-lag-1m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/mmt/binance-preferred/1m/ema-slope-512m-32m-lag-240m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/mon/binance-preferred/1m/ema-slope-8m-256m-lag-2m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/msft/binance-preferred/1m/active-fraction-15m` | cross-asset activity | fraction of nonzero minute returns | 15m | latest completed minute | binance-archive | 0.960 | 1.000 | 1 |
| `asset/msft/binance-preferred/1m/ema-slope-2m-2m-lag-480m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.994 | 0 |
| `asset/mtl/binance-preferred/1m/ema-distance-2048m-lag-60m` | EMA value | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/myx/binance-preferred/1m/ema-acceleration-128m-256m-lag-60m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/myx/binance-preferred/1m/ema-slope-128m-256m-lag-960m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.983 | 0 |
| `asset/myx/binance-preferred/1m/rsi-8192m-lag-1440m` | RSI | — | recursive | through origin | candle-derived | 1.000 | 0.972 | 0 |
| `asset/night/binance-preferred/1m/ema-acceleration-4096m-256m-lag-60m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/night/binance-preferred/1m/ema-slope-128m-8m-lag-240m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/night/binance-preferred/1m/ema-slope-8m-4m-lag-2m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/nok/binance-spot/1s/frft-0p75-entropy-256s` | fractional Fourier shape | Hann-windowed rFFT of signed log returns | 256s | through latest completed candle | candle-derived | 1.000 | 1.000 | 0 |
| `asset/now/binance-preferred/1m/ema-acceleration-512m-1m-lag-1m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/now/binance-preferred/1m/ema-slope-2m-512m-lag-1m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/nxpc/binance-preferred/1m/ema-acceleration-2048m-16m-lag-15m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/nxpc/binance-preferred/1m/rsi-1024m` | RSI | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/on/binance-preferred/1m/ema-acceleration-16m-8m-lag-4m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/on/binance-preferred/1m/ema-acceleration-512m-64m-lag-960m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 0.983 | 0 |
| `asset/on/binance-preferred/1m/ema-distance-4096m-lag-480m` | EMA value | — | recursive | through origin | candle-derived | 1.000 | 0.994 | 0 |
| `asset/on/binance-preferred/1m/ema-slope-4096m-1024m-lag-120m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/open/binance-preferred/1m/ema-acceleration-64m-1m-lag-480m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 0.994 | 0 |
| `asset/opn/binance-preferred/1m/ema-acceleration-4m-128m-lag-960m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 0.983 | 0 |
| `asset/orcl/binance-preferred/1m/ema-acceleration-8192m-512m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/order/binance-preferred/1m/ema-acceleration-4096m-2m-lag-240m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/order/binance-preferred/1m/ema-acceleration-8192m-2m-lag-240m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/osmo/binance-preferred/1m/ema-acceleration-16m-128m-lag-30m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/parti/binance-preferred/1m/rsi-32m-lag-120m` | RSI | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/pha/binance-preferred/1m/ema-acceleration-2m-512m-lag-30m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/pha/binance-preferred/1m/ema-acceleration-4m-512m-lag-30m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/pha/binance-preferred/1m/ema-slope-64m-512m-lag-1m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/pha/binance-preferred/1m/ema-slope-64m-512m-lag-2m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/pieverse/binance-preferred/1m/ema-acceleration-4096m-64m-lag-60m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/pivx/binance-preferred/1m/ema-distance-64m-lag-30m` | EMA value | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/pivx/binance-preferred/1m/ema-slope-64m-2m-lag-30m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/pixel/binance-preferred/1m/ema-slope-4096m-4m-lag-1440m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.972 | 0 |
| `asset/pixel/binance-spot/1s/frft-0p25-entropy-256s` | fractional Fourier shape | Hann-windowed rFFT of signed log returns | 256s | through latest completed candle | candle-derived | 1.000 | 1.000 | 0 |
| `asset/porto/binance-preferred/1m/ema-acceleration-4096m-512m-lag-60m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/porto/binance-preferred/1m/ema-acceleration-512m-128m-lag-15m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/porto/binance-preferred/1m/ema-acceleration-8192m-512m-lag-30m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/porto/binance-preferred/1m/ema-acceleration-8m-128m-lag-1440m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 0.972 | 0 |
| `asset/pyr/binance-preferred/1m/ema-acceleration-128m-1m-lag-30m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/pyr/binance-preferred/1m/ema-acceleration-16m-32m-lag-4m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/pyth/binance-preferred/1m/ema-slope-16m-512m-lag-120m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/qkc/binance-preferred/1m/ema-distance-4m-lag-1m` | EMA value | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/qkc/binance-preferred/1m/ema-slope-4m-1m-lag-1m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/rave/binance-usdm/5m/top-position-long-short-log-change-5m` | cross-asset futures positioning | log change in top-position-long-short over 5m | 5m | one completed 5m publication lag | binance-futures-archive | 0.950 | 1.000 | 1 |
| `asset/rave/binance-usdm/funding-event/funding-absolute-mean-9` | Funding pressure | — | 9 settlement(s), normally 3.0d | one minute after settlement | binance-futures-archive | 0.950 | 0.961 | 1 |
| `asset/ray/binance-preferred/1m/ema-slope-2048m-8m-lag-120m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/red/binance-preferred/1m/ema-acceleration-8m-4m-lag-1m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/req/binance-preferred/1m/ema-acceleration-128m-512m-lag-1m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/req/binance-preferred/1m/ema-acceleration-128m-512m-lag-4m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/req/binance-preferred/1m/ema-slope-2048m-4m-lag-120m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/rivn/binance-preferred/1m/completed-15m-log-volume` | volume regime | UTC-aligned 15m | 15m | updates only after bar close | candle-derived | 1.000 | 1.000 | 0 |
| `asset/rivn/binance-preferred/1m/ema-acceleration-4096m-256m-lag-60m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/rivn/binance-preferred/1m/log-trade-count-1m` | cross-asset activity regime | log1p trade count | 1m | latest completed minute | binance-archive | 0.960 | 1.000 | 1 |
| `asset/robo/binance-preferred/1m/rsi-4m-lag-120m` | RSI | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/rpl/binance-preferred/1m/ema-slope-2m-2m-lag-60m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/rpl/binance-preferred/1m/ema-slope-2m-4m-lag-60m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/safe/binance-usdm/1m/book-depth-imbalance-3pct` | cross-asset futures book imbalance | base-depth bid/ask imbalance within +/-3% | latest snapshot | latest snapshot in completed minute | binance-archive | 0.960 | 1.000 | 1 |
| `asset/sats/binance-preferred/1m/ema-slope-4096m-1024m-lag-120m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/sats/binance-preferred/1m/log-quote-volume-1m` | cross-asset activity regime | log1p quote volume | 1m | latest completed minute | binance-archive | 0.960 | 1.000 | 1 |
| `asset/sats/binance-preferred/1m/realized-volatility-2m` | volatility | sqrt(sum(r_1m^2)), window=2m | 2m | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/skl/binance-preferred/1m/ema-acceleration-4096m-32m-lag-30m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/skr/binance-preferred/1m/ema-slope-4m-8m-lag-2m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/snx/binance-preferred/1m/completed-15m-log-volume` | volume regime | UTC-aligned 15m | 15m | updates only after bar close | candle-derived | 1.000 | 1.000 | 0 |
| `asset/snxx/binance-preferred/1m/ema-acceleration-128m-64m-lag-960m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 0.983 | 0 |
| `asset/snxx/binance-preferred/1m/ema-acceleration-4m-256m-lag-120m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/snxx/binance-preferred/1m/ema-acceleration-8m-256m-lag-120m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/sol/binance-preferred/1m/fft-k1-real-16m` | Fourier complex coefficients | Hann-windowed rFFT of signed log returns | 16m | through latest completed candle | candle-derived | 1.000 | 1.000 | 0 |
| `asset/soph/binance-preferred/1m/ema-slope-16m-1024m-lag-240m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/soph/binance-preferred/1m/ema-slope-512m-256m-lag-240m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/space/binance-preferred/1m/ema-acceleration-16m-64m-lag-1440m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 0.972 | 0 |
| `asset/spell/binance-preferred/1m/morlet-slow-real-16m` | complex wavelet coefficients | Hann-windowed rFFT of signed log returns | 16m | through latest completed candle | candle-derived | 1.000 | 1.000 | 0 |
| `asset/sportfun/binance-preferred/1m/ema-acceleration-16m-2m-lag-30m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/sportfun/binance-preferred/1m/ema-acceleration-512m-16m-lag-4m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/stable/binance-preferred/1m/ema-acceleration-32m-64m-lag-2m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/star/binance-preferred/1m/ema-slope-32m-8m-lag-2m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/stg/binance-preferred/1m/ema-acceleration-2048m-32m-lag-120m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/tac/binance-preferred/1m/ema-slope-512m-512m-lag-2m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/tac/binance-preferred/1m/ema-slope-512m-512m-lag-30m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/tac/binance-preferred/1m/ema-slope-512m-512m-lag-4m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/tac/binance-preferred/1m/ema-slope-8m-16m-lag-2m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/take/binance-preferred/1m/ema-slope-32m-256m-lag-30m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/take/binance-preferred/1m/ema-slope-8m-256m-lag-30m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/tfuel/binance-preferred/1m/ema-slope-4096m-128m-lag-120m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/tko/binance-preferred/1m/ema-acceleration-4m-1m-lag-120m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/tko/binance-preferred/1m/ema-acceleration-512m-1m-lag-120m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/tnsr/binance-preferred/1m/fft-k4-imag-256m` | Fourier complex coefficients | Hann-windowed rFFT of signed log returns | 256m | through latest completed candle | candle-derived | 1.000 | 1.000 | 0 |
| `asset/tradoor/binance-preferred/1m/fft-k2-imag-64m` | Fourier complex coefficients | Hann-windowed rFFT of signed log returns | 64m | through latest completed candle | candle-derived | 1.000 | 1.000 | 0 |
| `asset/trx/binance-preferred/1m/ema-acceleration-2048m-512m-lag-480m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 0.994 | 0 |
| `asset/tst/binance-preferred/1m/ema-slope-4096m-32m-lag-480m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.994 | 0 |
| `asset/tst/binance-preferred/1m/ema-slope-512m-1024m-lag-1440m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.972 | 0 |
| `asset/turtle/binance-preferred/1m/ema-acceleration-2048m-64m-lag-480m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 0.994 | 0 |
| `asset/turtle/binance-spot/1s/fft-low-power-share-256s` | Fourier shape | Hann-windowed rFFT of signed log returns | 256s | through latest completed candle | candle-derived | 1.000 | 1.000 | 0 |
| `asset/tut/binance-preferred/1m/ema-acceleration-2m-1m` | cross-asset EMA acceleration | second difference of EMA(2m) | 2m recursive + 2m | latest completed minute | candle-derived | 1.000 | 1.000 | 0 |
| `asset/tut/binance-preferred/1m/ema-acceleration-4m-1m-lag-15m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/tut/binance-preferred/1m/ema-slope-128m-4m-lag-240m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/tut/binance-preferred/1m/ema-slope-32m-512m-lag-960m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.983 | 0 |
| `asset/ub/binance-preferred/1m/ema-slope-4m-16m-lag-480m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.994 | 0 |
| `asset/uma/binance-spot/1s/rsi-2s` | price dynamics | period=2s | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/unicode-e9be99e899be/binance-preferred/1m/ema-acceleration-64m-8m-lag-8m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/unicode-e9be99e899be/binance-preferred/1m/rsi-4m-lag-120m` | RSI | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/useless/binance-usdm/5m/open-interest-log-change-15m` | cross-asset futures positioning | log change in open-interest over 15m | 15m | one completed 5m publication lag | binance-futures-archive | 0.950 | 0.999 | 1 |
| `asset/v/binance-preferred/1m/ema-acceleration-32m-1m-lag-1m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/v/binance-preferred/1m/ema-distance-64m-lag-2m` | EMA value | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/v/binance-preferred/1m/realized-volatility-5m` | cross-asset volatility | sqrt sum squared minute log returns | 5m | latest completed minute | candle-derived | 1.000 | 1.000 | 0 |
| `asset/v/binance-preferred/1m/zero-run-age` | cross-asset activity | log1p consecutive exact-zero minute returns | recursive | latest completed minute | binance-archive | 0.960 | 1.000 | 1 |
| `asset/vanry/binance-preferred/1m/ema-slope-4096m-4m-lag-120m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/vtho/binance-preferred/1m/ema-acceleration-2m-8m-lag-480m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 0.994 | 0 |
| `asset/vtho/binance-preferred/1m/ema-slope-2m-512m-lag-960m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.983 | 0 |
| `asset/wal/binance-preferred/1m/ema-acceleration-8m-8m-lag-1440m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 0.972 | 0 |
| `asset/wen/binance-preferred/1m/ema-acceleration-64m-512m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/wen/binance-preferred/1m/ema-acceleration-64m-512m-lag-1m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/win/binance-spot/1s/haar-coarse-energy-share-64s` | wavelet energy shape | Hann-windowed rFFT of signed log returns | 64s | through latest completed candle | candle-derived | 1.000 | 1.000 | 0 |
| `asset/wlfi/binance-preferred/1m/ema-acceleration-16m-256m-lag-240m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/xag/binance-preferred/1m/completed-5m-log-volume` | volume regime | UTC-aligned 5m | 5m | updates only after bar close | candle-derived | 1.000 | 1.000 | 0 |
| `asset/xan/binance-preferred/1m/ema-acceleration-64m-32m-lag-2m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/xan/binance-preferred/1m/rsi-16m-lag-2m` | RSI | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/xec/binance-preferred/1m/ema-slope-4096m-128m-lag-60m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/xec/binance-preferred/1m/ema-slope-4096m-64m-lag-120m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/xec/binance-preferred/1m/ema-slope-8192m-64m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/xec/binance-preferred/1m/ema-slope-8192m-64m-lag-1m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/xno/binance-preferred/1m/ema-acceleration-512m-512m-lag-60m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/xno/binance-preferred/1m/log-trade-count-1m` | cross-asset activity regime | log1p trade count | 1m | latest completed minute | binance-archive | 0.960 | 1.000 | 1 |
| `asset/xno/binance-preferred/1m/morlet-fast-imag-16m` | complex wavelet coefficients | Hann-windowed rFFT of signed log returns | 16m | through latest completed candle | candle-derived | 1.000 | 1.000 | 0 |
| `asset/xny/binance-preferred/1m/ema-slope-512m-128m-lag-960m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 0.983 | 0 |
| `asset/xpin/binance-preferred/1m/ema-acceleration-2048m-256m-lag-1440m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 0.972 | 0 |
| `asset/xpin/binance-preferred/1m/ema-acceleration-8192m-512m-lag-1440m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 0.972 | 0 |
| `asset/xrp/binance-preferred/1m/ema-acceleration-128m-2m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/xrp/binance-preferred/1m/ema-distance-128m-lag-1m` | EMA value | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/xrp/binance-preferred/1m/ema-slope-128m-1m-lag-1m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/xrp/binance-preferred/1m/ema-slope-32m-2m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/xrp/binance-preferred/1m/ema-slope-4m-2m-lag-1m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/xrp/binance-spot/1s/ema-acceleration-2048s-2s` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/xrp/binance-spot/1s/fft-log-energy-256s` | Fourier energy | Hann-windowed rFFT of signed log returns | 256s | through latest completed candle | candle-derived | 1.000 | 1.000 | 0 |
| `asset/yb/binance-preferred/1m/realized-volatility-30m` | minute volatility | sqrt(sum(r_1m^2)),window=30m | 30m | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/zama/binance-preferred/1m/ema-acceleration-8192m-128m-lag-120m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/zama/binance-preferred/1m/taker-imbalance-ema-60m` | cross-asset aggressor flow | EMA(60m) taker quote imbalance | 60m recursive | latest completed minute | binance-archive | 0.960 | 1.000 | 1 |
| `asset/zbt/binance-preferred/1m/ema-acceleration-32m-32m-lag-240m` | EMA acceleration | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/zbt/binance-preferred/1m/ema-slope-4m-512m-lag-2m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/zhipu/binance-preferred/1m/ema-distance-32m-lag-15m` | EMA value | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/zhipu/binance-preferred/1m/ema-slope-32m-1m-lag-15m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/zkp/binance-preferred/1m/ema-slope-2048m-8m-lag-8m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/zm/binance-preferred/1m/ema-slope-16m-4m-lag-1m` | EMA slope | — | recursive | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/zm/binance-preferred/1m/realized-volatility-60m` | minute volatility | sqrt(sum(r_1m^2)),window=60m | 60m | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/zm/binance-preferred/1m/rsi-4096m-lag-960m` | RSI | — | recursive | through origin | candle-derived | 1.000 | 0.983 | 0 |
| `asset/zm/binance-preferred/1m/zero-run-age` | cross-asset activity | log1p consecutive exact-zero minute returns | recursive | latest completed minute | binance-archive | 0.960 | 1.000 | 1 |

### 15m: 3 inputs

| Coordinate | Family | Construction | Lookback | Delay | Source policy | Declared availability | Empirical availability | Acquisition cost |
|---|---|---|---|---|---|---:|---:|---:|
| `asset/btc/binance-preferred/1m/range-1m` | minute candle shape | 10000*log(high/low) | 1m | latest completed minute | candle-derived | 1.000 | 1.000 | 0 |
| `asset/btc/binance-preferred/1m/realized-volatility-60m` | minute volatility | sqrt(sum(r_1m^2)),window=60m | 60m | through origin | candle-derived | 1.000 | 1.000 | 0 |
| `asset/eth/binance-preferred/1m/realized-volatility-60m` | minute volatility | sqrt(sum(r_1m^2)),window=60m | 60m | through origin | candle-derived | 1.000 | — | 0 |

### 1h: 2 inputs

| Coordinate | Family | Construction | Lookback | Delay | Source policy | Declared availability | Empirical availability | Acquisition cost |
|---|---|---|---|---|---|---:|---:|---:|
| `asset/btc/binance-preferred/1m/completed-1h-log-volume` | volume regime | UTC-aligned 1h | 1h | after hour close | binance-archive | 0.960 | — | 1 |
| `asset/btc/binance-preferred/1m/range-1m` | minute candle shape | 10000*log(high/low) | 1m | latest completed minute | candle-derived | 1.000 | 1.000 | 0 |

## Full-registry KKT audit

Providers scanned: `base, dense, external, funding, long, representative, spectral1m, spectral1s, technical1s`. Unique emitted coordinates: **989,335**. Complete stream: **yes**.

| Horizon | Production correction | Regularization | Scanned | Maximum omitted-group violation | Maximum active-stationarity residual | Certified |
|---|---|---:|---:|---:|---:|---|
| 1s | required | 0.00900192 | 989,187 | 0.00010000 | 0.00009910 | yes |
| 1m | required | 0.00599053 | 989,000 | 0.00010000 | 0.00009815 | yes |

## Meaning of the global certificate

The proof boundary is deliberately narrow: fixed causal quantile partitions, the incumbent smoothed joint-state conditional distribution, and additive four-bin categorical corrections with a group-lasso penalty. For a selected penalty, a complete nonpositive KKT scan proves the global convex optimum in that model class across the eligible canonical registry. It does not prove an optimum over arbitrary neural interactions, alternative discretizations, revised/non-point-in-time data, or the short live-only inventories.

Feature discovery used the first 23 days and chronological folds within it, so it is not fully nested feature-selection cross-validation. The final seven days remained untouched through the first confirmation. Later KKT expansions and the operational tie-break used only training residuals and stored fold paths, but this report reuses the already revealed transfer segment; its whole-day resampling is robustness evidence, not a new pristine holdout. A later calendar block is still required before deployment promotion.

## Reproducibility

- Registry: `data/benchmarks/global-feature-registry.json`
- Expanded working set: `data/runtime-cache/global-btc-expanded-working-set-v4/manifest.json`
- Search: `data/benchmarks/global-btc-v4-production-final-search.json`
- Transfer audit: `data/benchmarks/global-btc-v4-production-final-transfer-robustness.json`
- KKT audit: `data/benchmarks/global-btc-v4-production-final-kkt-merged.json`
- Rejected-horizon incumbent search: `data/benchmarks/global-btc-v3-operational-search.json`
- Rejected-horizon robustness: `data/benchmarks/global-btc-v3-operational-transfer-robustness.json`

```powershell
npm run analysis:feature-registry
npm run analysis:global-basis-search:per-horizon -- --working-set data/runtime-cache/global-btc-expanded-working-set-v4 --all-coordinates --horizons 1s,1m --equivalence-rule paired-one-se
npm run analysis:global-basis-search:refine-support -- --search data/benchmarks/global-btc-v4-production-search.json --model data/benchmarks/global-btc-v4-production-model.npz --working-set data/runtime-cache/global-btc-expanded-working-set-v4 --output-search data/benchmarks/global-btc-v4-production-final-search.json --output-model data/benchmarks/global-btc-v4-production-final-model.npz --horizon 1s --device cpu
npm run analysis:global-basis-search:transfer-robustness -- --search data/benchmarks/global-btc-v4-production-final-search.json --model data/benchmarks/global-btc-v4-production-final-model.npz --output data/benchmarks/global-btc-v4-production-final-transfer-robustness.json
# Run the three disjoint KKT provider partitions with all training rows, then:
npm run analysis:global-basis-search:kkt-merge -- --search data/benchmarks/global-btc-v4-production-final-search.json --model data/benchmarks/global-btc-v4-production-final-model.npz --parts data/benchmarks/global-btc-v4-production-final-kkt-core.json data/benchmarks/global-btc-v4-production-final-kkt-spectral1m.json data/benchmarks/global-btc-v4-production-final-kkt-external-second.json --output data/benchmarks/global-btc-v4-production-final-kkt-merged.json --horizons 1s,1m
npm run analysis:global-basis-search:report
```
