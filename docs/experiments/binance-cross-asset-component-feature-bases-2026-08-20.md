# Binance cross-asset BTC component feature bases

Generated `2026-08-20T18:37:25Z`.

This experiment replicates every source-supported candle, activity, volatility, technical, spectral, flow, positioning, funding, and book-depth family across the latest study's largest 1-minute Binance basis. It then globally refits each component from the union of the existing 147-coordinate inventory and the replicated cross-market inventory; the previous basis is a benchmark, not a frozen baseline.

## Scope

- Source market universe: 713 economic assets.
- Latest 1-minute independent-scale basis: 257 assets.
- Explicit additions: ETH, SOL, XRP, HYPE.
- Requested assets: 261 total; 260 non-BTC markets attempted; 260 non-BTC markets contributed at least one eligible coordinate.
- BTC remains in the existing 147-coordinate inventory and is excluded only from the replicated cross-market copy.
- Wholly excluded for insufficient recent coverage: none; individual low-coverage source tiers are excluded by the 95% rule below.
- Candidate coordinates: 31,043 (147 existing + 30,896 replicated cross-market).
- Window: 2026-07-18 through 2026-08-16 UTC; chronological train, primary, and untouched transfer blocks match the recent broad-basis audit.
- Alignment: each stored `:59` origin uses the candle from that same UTC minute via floor-to-minute indexing; the following minute is excluded.

## Backfilled source coverage

| Source tier | Markets/assets with usable coverage | Causal availability |
|---|---:|---|
| 1m spot | 144 | completed minute |
| 1m USD-M futures | 236 | completed minute |
| paired spot + futures | 119 | completed minute |
| spot 1s fast state | 140 | latest completed second |
| USD-M OI/positioning metrics | 196 | one completed 5m publication lag |
| USD-M settled funding | 236 | latest settled event |
| USD-M 12-band percentage depth | 232 | latest complete snapshot in completed minute |

## Decision summary

The enlarged search does **not** justify replacing the current component bases wholesale. The selected refits win strongly on the primary selection week but lose on average on the untouched transfer week at every horizon, which is direct evidence of winner's-curse/multiple-testing overfit from searching 31,043 coordinates.

| Horizon | Mean previous transfer bits | Mean refit transfer bits | Mean change | Refit beats previous | Positive-transfer cross-market leads that beat previous |
|---|---:|---:|---:|---:|---:|
| 1s | 0.156864 | 0.068014 | -0.088851 | 9/19 | 3/19 |
| 1m | 0.083404 | 0.080605 | -0.002798 | 5/19 | 4/19 |
| 15m | 0.058922 | 0.050956 | -0.007965 | 7/19 | 4/19 |
| 1h | 0.038059 | -0.038989 | -0.077048 | 4/19 | 4/19 |

The last column is a discovery shortlist, not a production promotion: looking at the transfer week to identify those rows consumes it for that decision, so they require a new later holdout or rolling-window confirmation.

Among the four explicit additions, ETH appears repeatedly and supplies the only positive-transfer benchmark-beating leads; SOL is never selected, while the selected XRP and HYPE rows do not beat their previous bases on transfer.

## Globally refitted component bases

`positive vs marginal` means the refit has positive total and both half-block scores versus the no-feature marginal predictor. It does **not** mean it beats the previous basis; use the final change column for that comparison.

| Horizon | BTC target component | Transfer validation | Previous basis (transfer bits) | Refitted extended basis | Primary bits | Transfer bits | Change vs previous transfer |
|---|---|---|---|---|---:|---:|---:|
| 1s | P(inactive) | positive vs marginal | active-count-60s, realized-volatility-30m, ema-acceleration-2s-1s (0.048499) | active-count-60s, realized-volatility-30m, ema-acceleration-2s-1s | 0.028371 | 0.048499 | +0.000000 |
| 1s | P(positive \| active) | positive vs marginal | spot-flow-last-side-1s, spot-flow-maximum-skew-1s, futures-basis-deviation-5m (0.194885) | spot-flow-last-side-1s, eth__rsi-2s, futures-close-location-1m | 0.174450 | 0.167557 | -0.027328 |
| 1s | P(zero/sign/magnitude quartile) | failed transfer | spot-flow-last-side-1s, spot-book-spread-bps, range-1s (0.131251) | open__top-position-minus-account-long-short, spot-flow-last-side-1s, range-1s | 0.329455 | -0.237869 | -0.369120 |
| 1s | P(\|R\| >= Q25 \| active) | positive vs marginal | spot-book-spread-bps, range-1s, top-position-minus-account-log (0.259687) | xag__open-interest-value-log-level, spot-book-spread-bps, range-1s | 0.447091 | 0.115118 | -0.144568 |
| 1s | P(\|R\| >= Q50 \| active) | failed transfer | spot-book-spread-bps, eth-realized-volatility-60m, ema-slope-8s-8s (-0.075216) | o__top-account-long-short-log-level, realized-volatility-30m, futures-range-1m | 0.251034 | -0.630550 | -0.555334 |
| 1s | P(\|R\| >= Q75 \| active) | positive vs marginal | range-1m, realized-volatility-30m, spot-flow-raw-per-aggregate-1s (0.085051) | realized-volatility-60s, realized-volatility-30m, xag__completed-volume-60m | 0.095157 | 0.089258 | +0.004206 |
| 1s | P(\|R\| >= Q90 \| active) | positive vs marginal | range-1m, realized-volatility-30m, spot-flow-raw-per-aggregate-1s (0.036285) | realized-volatility-60s, realized-volatility-15m, eth-realized-volatility-60m | 0.038432 | 0.038655 | +0.002369 |
| 1s | P(positive \| active, \|R\| >= Q50) | failed transfer | futures-basis-deviation-15m, spot-flow-trade-imbalance-ema-8, spot-book-quantity-imbalance-l10 (-0.024848) | futures-basis-deviation-15m, spot-flow-trade-imbalance-ema-8, eth__rsi-2s | 0.191374 | -0.006901 | +0.017947 |
| 1s | P(positive \| active, \|R\| >= Q75) | positive vs marginal | spot-flow-trade-imbalance-ema-2, futures-basis-deviation-5m, close-location-1s (0.230369) | futures-basis-deviation-15m, spot-flow-trade-imbalance-ema-2, eth__futures-basis-deviation-15m | 0.275892 | 0.261977 | +0.031607 |
| 1s | P(positive \| active, \|R\| >= Q90) | positive vs marginal | futures-basis-deviation-5m, spot-flow-trade-imbalance-ema-2, spot-flow-trade-count-imbalance-lag-2s (0.166221) | futures-basis-deviation-15m, spot-flow-trade-imbalance-ema-2, eth__futures-basis-deviation-15m | 0.281479 | 0.248382 | +0.082162 |
| 1s | P(positive \| active, \|R\| < Q25) | positive vs marginal | spot-flow-last-side-1s, previous-return-1s, spot-flow-last-side-lag-2s (0.725519) | spot-flow-last-side-1s, previous-return-1s, spot-flow-last-side-lag-2s | 0.670456 | 0.725519 | +0.000000 |
| 1s | P(positive \| active, \|R\| < Q50) | positive vs marginal | spot-flow-last-side-1s, previous-return-1s, spot-flow-last-side-lag-2s (0.703901) | spot-flow-last-side-1s, previous-return-1s, spot-flow-last-side-lag-2s | 0.622870 | 0.703901 | +0.000000 |
| 1s | P(positive \| active, \|R\| < Q75) | positive vs marginal | spot-flow-last-side-1s, ema-acceleration-2s-1s, spot-flow-last-side-lag-2s (0.348716) | spot-flow-last-side-1s, previous-return-1s, spot-flow-last-side-lag-2s | 0.351167 | 0.377987 | +0.029272 |
| 1s | P(\|R\| >= Q50 \| negative) | failed transfer | spot-flow-last-side-1s, rsi-2s, spot-book-age-ms (-0.038033) | o__top-account-long-short-log-level, spot-flow-trade-count-imbalance-1s, rsi-2s | 0.351129 | -0.542350 | -0.504317 |
| 1s | P(\|R\| >= Q75 \| negative) | positive vs marginal | spot-flow-last-side-1s, realized-volatility-30m, rsi-2s (0.105934) | spot-flow-trade-count-imbalance-1s, realized-volatility-60s, spot-flow-last-side-1s | 0.180071 | 0.138855 | +0.032922 |
| 1s | P(\|R\| >= Q90 \| negative) | positive vs marginal | spot-flow-last-side-1s, realized-volatility-30m, realized-volatility-60s (0.043713) | realized-volatility-60s, spot-flow-aggregate-count-imbalance-1s, close-location-1s | 0.055517 | 0.056209 | +0.012496 |
| 1s | P(\|R\| >= Q50 \| positive) | failed transfer | spot-flow-trade-imbalance-ema-2, spot-book-observed, eth-realized-volatility-60m (-0.183437) | spot-flow-last-side-1s, spot-flow-trade-imbalance-ema-2, o__top-account-long-short-log-level | 0.425899 | -0.490608 | -0.307172 |
| 1s | P(\|R\| >= Q75 \| positive) | positive vs marginal | spot-flow-quantity-squared-skew-1s, futures-range-1m, spot-flow-last-side-1s (0.162213) | spot-flow-last-side-1s, rsi-2s, realized-volatility-60m | 0.182186 | 0.173997 | +0.011783 |
| 1s | P(\|R\| >= Q90 \| positive) | positive vs marginal | realized-volatility-60s, spot-flow-quantity-squared-skew-1s, spot-flow-last-side-1s (0.059712) | spot-flow-last-side-1s, eth__range-1m, asml__realized-volatility-60m | 0.062482 | 0.054623 | -0.005089 |
| 1m | P(inactive) | positive vs marginal | realized-volatility-60m, futures-log-trade-count-1m, ema-slope-8s-8s (0.112905) | snxx__realized-volatility-60m, eth__active-fraction-60m, ema-slope-8s-8s | 0.108180 | 0.106649 | -0.006256 |
| 1m | P(positive \| active) | positive vs marginal | futures-basis-deviation-15m, ema-slope-8s-8s, spot-flow-last-side-1s (0.019661) | futures-basis-deviation-15m, eth__rsi-2s, eth__rsi-2m | 0.017317 | 0.016593 | -0.003068 |
| 1m | P(zero/sign/magnitude quartile) | positive vs marginal | realized-volatility-60m, futures-log-trade-count-1m, ema-slope-8s-8s (0.273035) | realized-volatility-60m, futures-log-trade-count-1m, ema-slope-8s-8s | 0.248659 | 0.273035 | +0.000000 |
| 1m | P(\|R\| >= Q25 \| active) | positive vs marginal | realized-volatility-60m, futures-range-1m, ema-slope-8s-8s (0.124348) | mvll__realized-volatility-30m, range-1m, eth__realized-volatility-15m | 0.112429 | 0.123077 | -0.001270 |
| 1m | P(\|R\| >= Q50 \| active) | positive vs marginal | eth-realized-volatility-30m, futures-range-1m, completed-1h-log-volume (0.117043) | eth__realized-volatility-60m, futures-range-1m, snxx__active-fraction-60m | 0.118662 | 0.116739 | -0.000304 |
| 1m | P(\|R\| >= Q75 \| active) | positive vs marginal | realized-volatility-60m, eth-realized-volatility-30m, futures-log-trade-count-1m (0.102705) | eth-realized-volatility-60m, futures-log-trade-count-1m, snxx__active-fraction-60m | 0.092172 | 0.104641 | +0.001936 |
| 1m | P(\|R\| >= Q90 \| active) | positive vs marginal | futures-log-trade-count-1m, eth-realized-volatility-30m, realized-volatility-15m (0.067588) | futures-log-trade-count-1m, eth__realized-volatility-30m, glw__completed-volume-60m | 0.060886 | 0.065548 | -0.002040 |
| 1m | P(positive \| active, \|R\| >= Q50) | positive vs marginal | futures-basis-deviation-5m, spot-flow-trade-imbalance-ema-2, close-location-1s (0.031431) | futures-basis-deviation-5m, eth__rsi-2s, futures-close-location-1m | 0.028378 | 0.031963 | +0.000532 |
| 1m | P(positive \| active, \|R\| >= Q75) | positive vs marginal | futures-basis-deviation-15m, spot-flow-trade-imbalance-ema-2, futures-close-location-1m (0.045146) | futures-basis-deviation-60m, eth__rsi-2s, eth__close-location-1s | 0.023415 | 0.038908 | -0.006238 |
| 1m | P(positive \| active, \|R\| >= Q90) | positive vs marginal | spot-flow-trade-imbalance-ema-2, previous-return-1s, top-account-minus-global-log (0.032012) | spot-flow-trade-imbalance-ema-2, eth__futures-basis-deviation-15m, mbl__macd-histogram-3-10-4 | 0.030690 | 0.029271 | -0.002742 |
| 1m | P(positive \| active, \|R\| < Q25) | positive vs marginal | spot-flow-last-side-1s, rsi-2s, futures-close-location-1m (0.055315) | spot-flow-trade-imbalance-ema-128, rsi-2s, futures-close-location-1m | 0.042539 | 0.047763 | -0.007551 |
| 1m | P(positive \| active, \|R\| < Q50) | positive vs marginal | futures-basis-deviation-15m, spot-flow-last-side-1s, top-account-ratio-log-deviation-24h (0.014097) | spot-flow-trade-imbalance-ema-128, futures-basis-deviation-15m, previous-return-1s | 0.017491 | 0.015225 | +0.001128 |
| 1m | P(positive \| active, \|R\| < Q75) | positive vs marginal | futures-basis-deviation-15m, spot-flow-last-side-1s, top-account-ratio-log-deviation-24h (0.013465) | futures-basis-deviation-15m, eth__rsi-2s, eth__rsi-2m | 0.016150 | 0.012018 | -0.001447 |
| 1m | P(\|R\| >= Q50 \| negative) | positive vs marginal | realized-volatility-60m, eth-realized-volatility-30m, futures-range-1m (0.118969) | futures-range-1m, kstr__completed-volume-60m, hype__realized-volatility-30m | 0.119373 | 0.106351 | -0.012618 |
| 1m | P(\|R\| >= Q75 \| negative) | positive vs marginal | realized-volatility-60m, eth-realized-volatility-60m, realized-volatility-60s (0.096052) | realized-volatility-60m, realized-volatility-60s, xag__completed-volume-60m | 0.089395 | 0.093500 | -0.002552 |
| 1m | P(\|R\| >= Q90 \| negative) | positive vs marginal | futures-log-trade-count-1m, realized-volatility-30m, active-count-60s (0.069110) | eth__realized-volatility-15m, realized-volatility-60s, glw__completed-volume-60m | 0.058400 | 0.069364 | +0.000254 |
| 1m | P(\|R\| >= Q50 \| positive) | positive vs marginal | realized-volatility-30m, eth-realized-volatility-60m, futures-range-1m (0.118242) | realized-volatility-30m, futures-range-1m, xag__completed-volume-60m | 0.118903 | 0.109060 | -0.009182 |
| 1m | P(\|R\| >= Q75 \| positive) | positive vs marginal | futures-log-trade-count-1m, realized-volatility-15m, eth-realized-volatility-30m (0.108300) | eth-realized-volatility-30m, futures-log-trade-count-1m, glw__book-log-depth-5pct | 0.094801 | 0.104947 | -0.003353 |
| 1m | P(\|R\| >= Q90 \| positive) | positive vs marginal | futures-log-trade-count-1m, eth-realized-volatility-30m, realized-volatility-15m (0.065243) | futures-log-trade-count-1m, realized-volatility-15m, glw__book-log-depth-5pct | 0.063674 | 0.066845 | +0.001601 |
| 15m | P(inactive) | positive vs marginal | global-ratio-log-level, sol-return-1m (0.004111) | ake__completed-volume-60m | 0.001311 | 0.002877 | -0.001234 |
| 15m | P(positive \| active) | failed transfer | bnb-return-1m, spot-flow-trade-imbalance-ema-8, realized-volatility-60s (-0.000672) | xno__return-kurtosis-60m, red__taker-imbalance-ema-15m, g__haar-contrast-64m | 0.019363 | -0.018292 | -0.017620 |
| 15m | P(zero/sign/magnitude quartile) | failed transfer | realized-volatility-60m, eth-realized-volatility-60m, range-1m (0.144491) | mvll__realized-volatility-30m, realized-volatility-60s, eth__completed-volume-60m | 0.168460 | 0.115185 | -0.029306 |
| 15m | P(\|R\| >= Q25 \| active) | failed transfer | eth-realized-volatility-30m, realized-volatility-60s, completed-1h-log-volume (0.067842) | mvll__active-fraction-60m, eth-realized-volatility-30m, range-1m | 0.103733 | 0.077616 | +0.009774 |
| 15m | P(\|R\| >= Q50 \| active) | failed transfer | eth-realized-volatility-30m, futures-log-trade-count-1m, completed-1h-log-volume (0.110826) | mvll__active-fraction-60m, realized-volatility-60m, orcl__realized-volatility-60m | 0.147335 | 0.116507 | +0.005681 |
| 15m | P(\|R\| >= Q75 \| active) | positive vs marginal | realized-volatility-60m, futures-log-trade-count-1m, completed-1h-log-volume (0.085488) | realized-volatility-60m, mvll__range-1m, asml__realized-volatility-15m | 0.088370 | 0.089394 | +0.003906 |
| 15m | P(\|R\| >= Q90 \| active) | failed transfer | realized-volatility-60m, futures-log-trade-count-1m, open-interest-value-log-change-60m (0.042047) | mvll__realized-volatility-15m, doge-realized-volatility-30m, asml__log-trade-count-1m | 0.067047 | 0.044152 | +0.002105 |
| 15m | P(positive \| active, \|R\| >= Q50) | failed transfer | doge-return-5m, range-1s, spot-flow-vwap-gap-1s (0.015473) | hana__top-account-long-short-log-change-5m, mon__book-depth-imbalance-5pct, eth__rsi-8m | 0.040610 | -0.008797 | -0.024269 |
| 15m | P(positive \| active, \|R\| >= Q75) | failed transfer | spot-flow-quote-imbalance-ema-128, doge-return-5m, spot-flow-vwap-gap-1s (0.016568) | kaito__active-fraction-15m, yfi__fourier-energy-k1-16m, snxx__realized-volatility-15m | 0.054533 | -0.025011 | -0.041578 |
| 15m | P(positive \| active, \|R\| >= Q90) | failed transfer | top-account-minus-global-log, spot-flow-quote-imbalance-ema-128, spot-book-delta-log-quantity-l10 (-0.051081) | rivn__taker-imbalance-ema-15m, opn__book-notional-imbalance-4pct, b2__signed-variance-efficiency-16m | 0.127559 | -0.122297 | -0.071216 |
| 15m | P(positive \| active, \|R\| < Q25) | failed transfer | open-interest-log-change-15m, bnb-realized-volatility-60m, spot-flow-quote-imbalance-lag-3s (-0.010342) | huma__fourier-imag-k1-64m, ta__completed-volume-60m, yb__macd-histogram-12-26-9 | 0.031408 | -0.016730 | -0.006387 |
| 15m | P(positive \| active, \|R\| < Q50) | failed transfer | top-position-minus-account-log, spot-flow-quote-imbalance-lag-3s, realized-volatility-15m (-0.002462) | ake__realized-volatility-240m, arpa__open-interest-log-level, xrp__funding-absolute-rate | 0.032119 | -0.010531 | -0.008069 |
| 15m | P(positive \| active, \|R\| < Q75) | failed transfer | open-interest-log-change-15m, global-ratio-log-deviation-24h, realized-volatility-15m (-0.005963) | zkp__open-interest-value-deviation-24h, huma__taker-imbalance-ema-60m, msft__active-fraction-60m | 0.021117 | -0.007199 | -0.001237 |
| 15m | P(\|R\| >= Q50 \| negative) | positive vs marginal | eth-realized-volatility-60m, open-interest-value-log-change-15m, top-account-minus-global-log (0.241420) | the__top-account-minus-global-long-short, doge-realized-volatility-60m, nxpc__realized-volatility-240m | 0.200320 | 0.166467 | -0.074953 |
| 15m | P(\|R\| >= Q75 \| negative) | positive vs marginal | eth-realized-volatility-60m, active-count-60s, top-account-minus-global-log (0.142699) | ibm__book-log-notional-5pct, bank__completed-volume-60m, cat__book-log-notional-5pct | 0.119020 | 0.112361 | -0.030338 |
| 15m | P(\|R\| >= Q90 \| negative) | positive vs marginal | realized-volatility-60m, active-count-60s, eth-return-5m (0.085612) | nxpc__realized-volatility-240m, tst__book-log-notional-1pct, hype__realized-volatility-60m | 0.072499 | 0.051183 | -0.034429 |
| 15m | P(\|R\| >= Q50 \| positive) | positive vs marginal | completed-1h-log-volume, active-count-60s, futures-close-location-1m (0.118303) | sofi__active-fraction-60m, doge-realized-volatility-60m, gua__open-interest-log-level | 0.147974 | 0.219331 | +0.101028 |
| 15m | P(\|R\| >= Q75 \| positive) | positive vs marginal | doge-realized-volatility-60m, completed-1h-log-volume, futures-taker-quote-imbalance-1m (0.070639) | gua__open-interest-log-level, sofi__active-fraction-60m, doge-realized-volatility-60m | 0.106880 | 0.124919 | +0.054280 |
| 15m | P(\|R\| >= Q90 \| positive) | positive vs marginal | doge-realized-volatility-60m, realized-volatility-30m, eth-return-1m (0.044517) | avgo__realized-volatility-30m, arm__book-log-depth-5pct, syn__book-notional-imbalance-5pct | 0.069800 | 0.057037 | +0.012520 |
| 1h | P(inactive) | positive vs marginal | futures-quote-volume-surprise-1m (0.000120) | pivx__realized-volatility-60m | 0.000429 | 0.000398 | +0.000277 |
| 1h | P(positive \| active) | failed transfer | top-position-ratio-log-deviation-24h, realized-volatility-60s (-0.024412) | manta__open-interest-deviation-24h, snow__active-fraction-60m, alice__rsi-32m | 0.075230 | -0.051604 | -0.027192 |
| 1h | P(zero/sign/magnitude quartile) | failed transfer | completed-1h-log-volume, range-1m (0.066274) | sofi__taker-imbalance-ema-60m, completed-1h-log-volume, sapien__top-account-long-short-log-level | 0.226940 | -0.190291 | -0.256565 |
| 1h | P(\|R\| >= Q25 \| active) | failed transfer | realized-volatility-240m, range-1m (0.081954) | bx__book-log-depth-1pct, eth__realized-volatility-15s, completed-1h-log-volume | 0.119654 | 0.041992 | -0.039961 |
| 1h | P(\|R\| >= Q50 \| active) | failed transfer | eth-realized-volatility-60m, completed-1h-log-volume (0.058678) | bank__log-trade-count-1m, bttc__efficiency-ratio-60m, ema-slope-8s-8s | 0.187314 | -0.026170 | -0.084848 |
| 1h | P(\|R\| >= Q75 \| active) | failed transfer | range-1m, open-interest-value-log-change-5m (0.036078) | bz__active-fraction-60m, range-1m, bttc__efficiency-ratio-60m | 0.129956 | 0.011592 | -0.024486 |
| 1h | P(\|R\| >= Q90 \| active) | failed transfer | realized-volatility-30m, range-1s (0.062264) | ibm__book-log-notional-5pct, eth__realized-volatility-15m, zbt__top-account-minus-global-long-short | 0.099545 | -0.061989 | -0.124253 |
| 1h | P(positive \| active, \|R\| >= Q50) | failed transfer | spot-flow-quote-imbalance-ema-128, spot-book-delta-log-quantity-l10, realized-volatility-240m (0.018183) | ub__return-kurtosis-60m, hana__completed-volume-60m, tst__taker-base-imbalance-1s | 0.119067 | -0.014876 | -0.033059 |
| 1h | P(positive \| active, \|R\| >= Q75) | failed transfer | top-account-minus-global-log, realized-volatility-240m (-0.046913) | stg__book-log-depth-5pct, mon__rsi-32m, fight__global-long-short-log-change-240m | 0.213681 | -0.143269 | -0.096356 |
| 1h | P(positive \| active, \|R\| >= Q90) | failed transfer | none (0.000000) | zk__active-fraction-60m, open__open-interest-value-log-change-60m, snxx__completed-volume-60m | 0.427628 | -0.582312 | -0.582312 |
| 1h | P(positive \| active, \|R\| < Q25) | failed transfer | global-ratio-log-level, open-interest-log-change-15m (-0.012255) | rave__open-interest-value-deviation-24h, b2__completed-volume-60m, crwd__return-kurtosis-60m | 0.099725 | -0.034989 | -0.022734 |
| 1h | P(positive \| active, \|R\| < Q50) | failed transfer | global-ratio-log-level, futures-basis-deviation-15m (-0.005780) | ctsi__return-kurtosis-60m, glw__completed-volume-60m, bttc__efficiency-ratio-60m | 0.091402 | -0.099515 | -0.093734 |
| 1h | P(positive \| active, \|R\| < Q75) | failed transfer | futures-range-1m, futures-taker-imbalance-ema-60m (-0.003738) | cati__funding-change, space__open-interest-value-deviation-24h, onds__trade-count-surprise-60m | 0.069010 | -0.016691 | -0.012953 |
| 1h | P(\|R\| >= Q50 \| negative) | failed transfer | doge-realized-volatility-30m, top-account-minus-global-log (0.192024) | bank__completed-volume-60m, nxpc__realized-volatility-240m, jto__open-interest-log-level | 0.343087 | -0.009500 | -0.201525 |
| 1h | P(\|R\| >= Q75 \| negative) | failed transfer | top-account-minus-global-log, active-count-60s (0.116903) | jto__open-interest-log-level, bttc__efficiency-ratio-60m, bz__log-mean-trade-notional-1m | 0.218996 | 0.046312 | -0.070590 |
| 1h | P(\|R\| >= Q90 \| negative) | failed transfer | top-account-minus-global-log, doge-realized-volatility-30m (0.049522) | nxpc__realized-volatility-240m, yb__global-long-short-log-level, zama__top-position-minus-account-long-short | 0.140185 | 0.010357 | -0.039165 |
| 1h | P(\|R\| >= Q50 \| positive) | positive vs marginal | doge-realized-volatility-30m, futures-taker-imbalance-ema-15m (0.072066) | bank__completed-volume-60m, dexe__realized-volatility-60m, ibm__book-log-notional-5pct | 0.226405 | 0.202813 | +0.130747 |
| 1h | P(\|R\| >= Q75 \| positive) | positive vs marginal | bnb-realized-volatility-30m, spot-flow-flip-rate-1s (0.044852) | rave__funding-absolute-rate, mtl__top-position-minus-account-long-short, avgo__realized-volatility-60m | 0.164226 | 0.120821 | +0.075969 |
| 1h | P(\|R\| >= Q90 \| positive) | positive vs marginal | doge-realized-volatility-30m, futures-taker-imbalance-ema-60m (0.017294) | minimax__book-notional-imbalance-5pct, doge-realized-volatility-30m, pixel__open-interest-log-level | 0.121089 | 0.056130 | +0.038836 |

## Exact selected input contract

Each row is one input to one prediction head. The delay is part of the contract; values must never be joined earlier than that boundary.

| Horizon | Target component | Input | Asset/scope | Family | Lookback | Delay | Source availability score |
|---|---|---|---|---|---|---|---:|
| 1s | P(inactive) | active-count-60s | BTC/general baseline inventory | activity | 60s | through origin | 1.00 |
| 1s | P(inactive) | realized-volatility-30m | BTC/general baseline inventory | minute volatility | 30m | through origin | 1.00 |
| 1s | P(inactive) | ema-acceleration-2s-1s | BTC/general baseline inventory | price dynamics | recursive | through origin | 1.00 |
| 1s | P(positive \| active) | spot-flow-last-side-1s | BTC/general baseline inventory | trade sequence | 1s | latest completed second | 0.95 |
| 1s | P(positive \| active) | eth__rsi-2s | ETH | cross-asset fast RSI | recursive | latest completed second at minute origin | 0.95 |
| 1s | P(positive \| active) | futures-close-location-1m | BTC/general baseline inventory | futures candle | 1m | after 1m candle close | 0.95 |
| 1s | P(zero/sign/magnitude quartile) | open__top-position-minus-account-long-short | OPEN | cross-asset futures positioning | latest 5m bucket | one completed 5m publication lag | 0.93 |
| 1s | P(zero/sign/magnitude quartile) | spot-flow-last-side-1s | BTC/general baseline inventory | trade sequence | 1s | latest completed second | 0.95 |
| 1s | P(zero/sign/magnitude quartile) | range-1s | BTC/general baseline inventory | candle shape | 1s | latest completed second | 1.00 |
| 1s | P(\|R\| >= Q25 \| active) | xag__open-interest-value-log-level | XAG | cross-asset futures positioning | latest 5m bucket | one completed 5m publication lag | 0.93 |
| 1s | P(\|R\| >= Q25 \| active) | spot-book-spread-bps | BTC/general baseline inventory | spot book top of book | latest snapshot | strictly before boundary; maximum age 5s | 0.45 |
| 1s | P(\|R\| >= Q25 \| active) | range-1s | BTC/general baseline inventory | candle shape | 1s | latest completed second | 1.00 |
| 1s | P(\|R\| >= Q50 \| active) | o__top-account-long-short-log-level | O | cross-asset futures positioning | latest 5m bucket | one completed 5m publication lag | 0.93 |
| 1s | P(\|R\| >= Q50 \| active) | realized-volatility-30m | BTC/general baseline inventory | minute volatility | 30m | through origin | 1.00 |
| 1s | P(\|R\| >= Q50 \| active) | futures-range-1m | BTC/general baseline inventory | futures candle | 1m | after 1m candle close | 0.95 |
| 1s | P(\|R\| >= Q75 \| active) | realized-volatility-60s | BTC/general baseline inventory | volatility | 60s | through origin | 1.00 |
| 1s | P(\|R\| >= Q75 \| active) | realized-volatility-30m | BTC/general baseline inventory | minute volatility | 30m | through origin | 1.00 |
| 1s | P(\|R\| >= Q75 \| active) | xag__completed-volume-60m | XAG | cross-asset volume regime | 60m | latest completed minute | 0.96 |
| 1s | P(\|R\| >= Q90 \| active) | realized-volatility-60s | BTC/general baseline inventory | volatility | 60s | through origin | 1.00 |
| 1s | P(\|R\| >= Q90 \| active) | realized-volatility-15m | BTC/general baseline inventory | minute volatility | 15m | through origin | 1.00 |
| 1s | P(\|R\| >= Q90 \| active) | eth-realized-volatility-60m | BTC/general baseline inventory | cross-market volatility | 60m | through origin | 0.96 |
| 1s | P(positive \| active, \|R\| >= Q50) | futures-basis-deviation-15m | BTC/general baseline inventory | basis | 15m | after 1m candle close | 0.95 |
| 1s | P(positive \| active, \|R\| >= Q50) | spot-flow-trade-imbalance-ema-8 | BTC/general baseline inventory | aggressor direction | latest completed observation | latest completed second | 0.95 |
| 1s | P(positive \| active, \|R\| >= Q50) | eth__rsi-2s | ETH | cross-asset fast RSI | recursive | latest completed second at minute origin | 0.95 |
| 1s | P(positive \| active, \|R\| >= Q75) | futures-basis-deviation-15m | BTC/general baseline inventory | basis | 15m | after 1m candle close | 0.95 |
| 1s | P(positive \| active, \|R\| >= Q75) | spot-flow-trade-imbalance-ema-2 | BTC/general baseline inventory | aggressor direction | latest completed observation | latest completed second | 0.95 |
| 1s | P(positive \| active, \|R\| >= Q75) | eth__futures-basis-deviation-15m | ETH | cross-asset futures basis | 15m recursive | latest completed minute | 0.96 |
| 1s | P(positive \| active, \|R\| >= Q90) | futures-basis-deviation-15m | BTC/general baseline inventory | basis | 15m | after 1m candle close | 0.95 |
| 1s | P(positive \| active, \|R\| >= Q90) | spot-flow-trade-imbalance-ema-2 | BTC/general baseline inventory | aggressor direction | latest completed observation | latest completed second | 0.95 |
| 1s | P(positive \| active, \|R\| >= Q90) | eth__futures-basis-deviation-15m | ETH | cross-asset futures basis | 15m recursive | latest completed minute | 0.96 |
| 1s | P(positive \| active, \|R\| < Q25) | spot-flow-last-side-1s | BTC/general baseline inventory | trade sequence | 1s | latest completed second | 0.95 |
| 1s | P(positive \| active, \|R\| < Q25) | previous-return-1s | BTC/general baseline inventory | return history | 1s | latest completed second | 1.00 |
| 1s | P(positive \| active, \|R\| < Q25) | spot-flow-last-side-lag-2s | BTC/general baseline inventory | lagged trade sequence | 2s | latest completed second | 0.95 |
| 1s | P(positive \| active, \|R\| < Q50) | spot-flow-last-side-1s | BTC/general baseline inventory | trade sequence | 1s | latest completed second | 0.95 |
| 1s | P(positive \| active, \|R\| < Q50) | previous-return-1s | BTC/general baseline inventory | return history | 1s | latest completed second | 1.00 |
| 1s | P(positive \| active, \|R\| < Q50) | spot-flow-last-side-lag-2s | BTC/general baseline inventory | lagged trade sequence | 2s | latest completed second | 0.95 |
| 1s | P(positive \| active, \|R\| < Q75) | spot-flow-last-side-1s | BTC/general baseline inventory | trade sequence | 1s | latest completed second | 0.95 |
| 1s | P(positive \| active, \|R\| < Q75) | previous-return-1s | BTC/general baseline inventory | return history | 1s | latest completed second | 1.00 |
| 1s | P(positive \| active, \|R\| < Q75) | spot-flow-last-side-lag-2s | BTC/general baseline inventory | lagged trade sequence | 2s | latest completed second | 0.95 |
| 1s | P(\|R\| >= Q50 \| negative) | o__top-account-long-short-log-level | O | cross-asset futures positioning | latest 5m bucket | one completed 5m publication lag | 0.93 |
| 1s | P(\|R\| >= Q50 \| negative) | spot-flow-trade-count-imbalance-1s | BTC/general baseline inventory | aggressor direction | 1s | latest completed second | 0.95 |
| 1s | P(\|R\| >= Q50 \| negative) | rsi-2s | BTC/general baseline inventory | price dynamics | recursive | through origin | 1.00 |
| 1s | P(\|R\| >= Q75 \| negative) | spot-flow-trade-count-imbalance-1s | BTC/general baseline inventory | aggressor direction | 1s | latest completed second | 0.95 |
| 1s | P(\|R\| >= Q75 \| negative) | realized-volatility-60s | BTC/general baseline inventory | volatility | 60s | through origin | 1.00 |
| 1s | P(\|R\| >= Q75 \| negative) | spot-flow-last-side-1s | BTC/general baseline inventory | trade sequence | 1s | latest completed second | 0.95 |
| 1s | P(\|R\| >= Q90 \| negative) | realized-volatility-60s | BTC/general baseline inventory | volatility | 60s | through origin | 1.00 |
| 1s | P(\|R\| >= Q90 \| negative) | spot-flow-aggregate-count-imbalance-1s | BTC/general baseline inventory | aggressor direction | 1s | latest completed second | 0.95 |
| 1s | P(\|R\| >= Q90 \| negative) | close-location-1s | BTC/general baseline inventory | candle shape | 1s | latest completed second | 1.00 |
| 1s | P(\|R\| >= Q50 \| positive) | spot-flow-last-side-1s | BTC/general baseline inventory | trade sequence | 1s | latest completed second | 0.95 |
| 1s | P(\|R\| >= Q50 \| positive) | spot-flow-trade-imbalance-ema-2 | BTC/general baseline inventory | aggressor direction | latest completed observation | latest completed second | 0.95 |
| 1s | P(\|R\| >= Q50 \| positive) | o__top-account-long-short-log-level | O | cross-asset futures positioning | latest 5m bucket | one completed 5m publication lag | 0.93 |
| 1s | P(\|R\| >= Q75 \| positive) | spot-flow-last-side-1s | BTC/general baseline inventory | trade sequence | 1s | latest completed second | 0.95 |
| 1s | P(\|R\| >= Q75 \| positive) | rsi-2s | BTC/general baseline inventory | price dynamics | recursive | through origin | 1.00 |
| 1s | P(\|R\| >= Q75 \| positive) | realized-volatility-60m | BTC/general baseline inventory | minute volatility | 60m | through origin | 1.00 |
| 1s | P(\|R\| >= Q90 \| positive) | spot-flow-last-side-1s | BTC/general baseline inventory | trade sequence | 1s | latest completed second | 0.95 |
| 1s | P(\|R\| >= Q90 \| positive) | eth__range-1m | ETH | cross-asset candle shape | 1m | latest completed minute | 0.96 |
| 1s | P(\|R\| >= Q90 \| positive) | asml__realized-volatility-60m | ASML | cross-asset volatility | 60m | latest completed minute | 0.96 |
| 1m | P(inactive) | snxx__realized-volatility-60m | SNXX | cross-asset volatility | 60m | latest completed minute | 0.96 |
| 1m | P(inactive) | eth__active-fraction-60m | ETH | cross-asset activity | 60m | latest completed minute | 0.96 |
| 1m | P(inactive) | ema-slope-8s-8s | BTC/general baseline inventory | price dynamics | recursive | through origin | 1.00 |
| 1m | P(positive \| active) | futures-basis-deviation-15m | BTC/general baseline inventory | basis | 15m | after 1m candle close | 0.95 |
| 1m | P(positive \| active) | eth__rsi-2s | ETH | cross-asset fast RSI | recursive | latest completed second at minute origin | 0.95 |
| 1m | P(positive \| active) | eth__rsi-2m | ETH | cross-asset RSI | 2m recursive | latest completed minute | 0.96 |
| 1m | P(zero/sign/magnitude quartile) | realized-volatility-60m | BTC/general baseline inventory | minute volatility | 60m | through origin | 1.00 |
| 1m | P(zero/sign/magnitude quartile) | futures-log-trade-count-1m | BTC/general baseline inventory | futures activity | 1m | after 1m candle close | 0.95 |
| 1m | P(zero/sign/magnitude quartile) | ema-slope-8s-8s | BTC/general baseline inventory | price dynamics | recursive | through origin | 1.00 |
| 1m | P(\|R\| >= Q25 \| active) | mvll__realized-volatility-30m | MVLL | cross-asset volatility | 30m | latest completed minute | 0.96 |
| 1m | P(\|R\| >= Q25 \| active) | range-1m | BTC/general baseline inventory | minute candle shape | 1m | latest completed minute | 1.00 |
| 1m | P(\|R\| >= Q25 \| active) | eth__realized-volatility-15m | ETH | cross-asset volatility | 15m | latest completed minute | 0.96 |
| 1m | P(\|R\| >= Q50 \| active) | eth__realized-volatility-60m | ETH | cross-asset volatility | 60m | latest completed minute | 0.96 |
| 1m | P(\|R\| >= Q50 \| active) | futures-range-1m | BTC/general baseline inventory | futures candle | 1m | after 1m candle close | 0.95 |
| 1m | P(\|R\| >= Q50 \| active) | snxx__active-fraction-60m | SNXX | cross-asset activity | 60m | latest completed minute | 0.96 |
| 1m | P(\|R\| >= Q75 \| active) | eth-realized-volatility-60m | BTC/general baseline inventory | cross-market volatility | 60m | through origin | 0.96 |
| 1m | P(\|R\| >= Q75 \| active) | futures-log-trade-count-1m | BTC/general baseline inventory | futures activity | 1m | after 1m candle close | 0.95 |
| 1m | P(\|R\| >= Q75 \| active) | snxx__active-fraction-60m | SNXX | cross-asset activity | 60m | latest completed minute | 0.96 |
| 1m | P(\|R\| >= Q90 \| active) | futures-log-trade-count-1m | BTC/general baseline inventory | futures activity | 1m | after 1m candle close | 0.95 |
| 1m | P(\|R\| >= Q90 \| active) | eth__realized-volatility-30m | ETH | cross-asset volatility | 30m | latest completed minute | 0.96 |
| 1m | P(\|R\| >= Q90 \| active) | glw__completed-volume-60m | GLW | cross-asset volume regime | 60m | latest completed minute | 0.96 |
| 1m | P(positive \| active, \|R\| >= Q50) | futures-basis-deviation-5m | BTC/general baseline inventory | basis | 5m | after 1m candle close | 0.95 |
| 1m | P(positive \| active, \|R\| >= Q50) | eth__rsi-2s | ETH | cross-asset fast RSI | recursive | latest completed second at minute origin | 0.95 |
| 1m | P(positive \| active, \|R\| >= Q50) | futures-close-location-1m | BTC/general baseline inventory | futures candle | 1m | after 1m candle close | 0.95 |
| 1m | P(positive \| active, \|R\| >= Q75) | futures-basis-deviation-60m | BTC/general baseline inventory | basis | 60m | after 1m candle close | 0.95 |
| 1m | P(positive \| active, \|R\| >= Q75) | eth__rsi-2s | ETH | cross-asset fast RSI | recursive | latest completed second at minute origin | 0.95 |
| 1m | P(positive \| active, \|R\| >= Q75) | eth__close-location-1s | ETH | cross-asset fast candle shape | 1s | latest completed second at minute origin | 0.95 |
| 1m | P(positive \| active, \|R\| >= Q90) | spot-flow-trade-imbalance-ema-2 | BTC/general baseline inventory | aggressor direction | latest completed observation | latest completed second | 0.95 |
| 1m | P(positive \| active, \|R\| >= Q90) | eth__futures-basis-deviation-15m | ETH | cross-asset futures basis | 15m recursive | latest completed minute | 0.96 |
| 1m | P(positive \| active, \|R\| >= Q90) | mbl__macd-histogram-3-10-4 | MBL | cross-asset MACD | 10m recursive | latest completed minute | 0.96 |
| 1m | P(positive \| active, \|R\| < Q25) | spot-flow-trade-imbalance-ema-128 | BTC/general baseline inventory | aggressor direction | latest completed observation | latest completed second | 0.95 |
| 1m | P(positive \| active, \|R\| < Q25) | rsi-2s | BTC/general baseline inventory | price dynamics | recursive | through origin | 1.00 |
| 1m | P(positive \| active, \|R\| < Q25) | futures-close-location-1m | BTC/general baseline inventory | futures candle | 1m | after 1m candle close | 0.95 |
| 1m | P(positive \| active, \|R\| < Q50) | spot-flow-trade-imbalance-ema-128 | BTC/general baseline inventory | aggressor direction | latest completed observation | latest completed second | 0.95 |
| 1m | P(positive \| active, \|R\| < Q50) | futures-basis-deviation-15m | BTC/general baseline inventory | basis | 15m | after 1m candle close | 0.95 |
| 1m | P(positive \| active, \|R\| < Q50) | previous-return-1s | BTC/general baseline inventory | return history | 1s | latest completed second | 1.00 |
| 1m | P(positive \| active, \|R\| < Q75) | futures-basis-deviation-15m | BTC/general baseline inventory | basis | 15m | after 1m candle close | 0.95 |
| 1m | P(positive \| active, \|R\| < Q75) | eth__rsi-2s | ETH | cross-asset fast RSI | recursive | latest completed second at minute origin | 0.95 |
| 1m | P(positive \| active, \|R\| < Q75) | eth__rsi-2m | ETH | cross-asset RSI | 2m recursive | latest completed minute | 0.96 |
| 1m | P(\|R\| >= Q50 \| negative) | futures-range-1m | BTC/general baseline inventory | futures candle | 1m | after 1m candle close | 0.95 |
| 1m | P(\|R\| >= Q50 \| negative) | kstr__completed-volume-60m | KSTR | cross-asset volume regime | 60m | latest completed minute | 0.96 |
| 1m | P(\|R\| >= Q50 \| negative) | hype__realized-volatility-30m | HYPE | cross-asset volatility | 30m | latest completed minute | 0.96 |
| 1m | P(\|R\| >= Q75 \| negative) | realized-volatility-60m | BTC/general baseline inventory | minute volatility | 60m | through origin | 1.00 |
| 1m | P(\|R\| >= Q75 \| negative) | realized-volatility-60s | BTC/general baseline inventory | volatility | 60s | through origin | 1.00 |
| 1m | P(\|R\| >= Q75 \| negative) | xag__completed-volume-60m | XAG | cross-asset volume regime | 60m | latest completed minute | 0.96 |
| 1m | P(\|R\| >= Q90 \| negative) | eth__realized-volatility-15m | ETH | cross-asset volatility | 15m | latest completed minute | 0.96 |
| 1m | P(\|R\| >= Q90 \| negative) | realized-volatility-60s | BTC/general baseline inventory | volatility | 60s | through origin | 1.00 |
| 1m | P(\|R\| >= Q90 \| negative) | glw__completed-volume-60m | GLW | cross-asset volume regime | 60m | latest completed minute | 0.96 |
| 1m | P(\|R\| >= Q50 \| positive) | realized-volatility-30m | BTC/general baseline inventory | minute volatility | 30m | through origin | 1.00 |
| 1m | P(\|R\| >= Q50 \| positive) | futures-range-1m | BTC/general baseline inventory | futures candle | 1m | after 1m candle close | 0.95 |
| 1m | P(\|R\| >= Q50 \| positive) | xag__completed-volume-60m | XAG | cross-asset volume regime | 60m | latest completed minute | 0.96 |
| 1m | P(\|R\| >= Q75 \| positive) | eth-realized-volatility-30m | BTC/general baseline inventory | cross-market volatility | 30m | through origin | 0.96 |
| 1m | P(\|R\| >= Q75 \| positive) | futures-log-trade-count-1m | BTC/general baseline inventory | futures activity | 1m | after 1m candle close | 0.95 |
| 1m | P(\|R\| >= Q75 \| positive) | glw__book-log-depth-5pct | GLW | cross-asset futures book shape | latest snapshot | latest snapshot in completed minute | 0.90 |
| 1m | P(\|R\| >= Q90 \| positive) | futures-log-trade-count-1m | BTC/general baseline inventory | futures activity | 1m | after 1m candle close | 0.95 |
| 1m | P(\|R\| >= Q90 \| positive) | realized-volatility-15m | BTC/general baseline inventory | minute volatility | 15m | through origin | 1.00 |
| 1m | P(\|R\| >= Q90 \| positive) | glw__book-log-depth-5pct | GLW | cross-asset futures book shape | latest snapshot | latest snapshot in completed minute | 0.90 |
| 15m | P(inactive) | ake__completed-volume-60m | AKE | cross-asset volume regime | 60m | latest completed minute | 0.96 |
| 15m | P(positive \| active) | xno__return-kurtosis-60m | XNO | cross-asset return shape | 60m | latest completed minute | 0.96 |
| 15m | P(positive \| active) | red__taker-imbalance-ema-15m | RED | cross-asset aggressor flow | 15m recursive | latest completed minute | 0.96 |
| 15m | P(positive \| active) | g__haar-contrast-64m | G | cross-asset wavelet | 64m | latest completed minute | 0.96 |
| 15m | P(zero/sign/magnitude quartile) | mvll__realized-volatility-30m | MVLL | cross-asset volatility | 30m | latest completed minute | 0.96 |
| 15m | P(zero/sign/magnitude quartile) | realized-volatility-60s | BTC/general baseline inventory | volatility | 60s | through origin | 1.00 |
| 15m | P(zero/sign/magnitude quartile) | eth__completed-volume-60m | ETH | cross-asset volume regime | 60m | latest completed minute | 0.96 |
| 15m | P(\|R\| >= Q25 \| active) | mvll__active-fraction-60m | MVLL | cross-asset activity | 60m | latest completed minute | 0.96 |
| 15m | P(\|R\| >= Q25 \| active) | eth-realized-volatility-30m | BTC/general baseline inventory | cross-market volatility | 30m | through origin | 0.96 |
| 15m | P(\|R\| >= Q25 \| active) | range-1m | BTC/general baseline inventory | minute candle shape | 1m | latest completed minute | 1.00 |
| 15m | P(\|R\| >= Q50 \| active) | mvll__active-fraction-60m | MVLL | cross-asset activity | 60m | latest completed minute | 0.96 |
| 15m | P(\|R\| >= Q50 \| active) | realized-volatility-60m | BTC/general baseline inventory | minute volatility | 60m | through origin | 1.00 |
| 15m | P(\|R\| >= Q50 \| active) | orcl__realized-volatility-60m | ORCL | cross-asset volatility | 60m | latest completed minute | 0.96 |
| 15m | P(\|R\| >= Q75 \| active) | realized-volatility-60m | BTC/general baseline inventory | minute volatility | 60m | through origin | 1.00 |
| 15m | P(\|R\| >= Q75 \| active) | mvll__range-1m | MVLL | cross-asset candle shape | 1m | latest completed minute | 0.96 |
| 15m | P(\|R\| >= Q75 \| active) | asml__realized-volatility-15m | ASML | cross-asset volatility | 15m | latest completed minute | 0.96 |
| 15m | P(\|R\| >= Q90 \| active) | mvll__realized-volatility-15m | MVLL | cross-asset volatility | 15m | latest completed minute | 0.96 |
| 15m | P(\|R\| >= Q90 \| active) | doge-realized-volatility-30m | BTC/general baseline inventory | cross-market volatility | 30m | through origin | 0.96 |
| 15m | P(\|R\| >= Q90 \| active) | asml__log-trade-count-1m | ASML | cross-asset activity regime | 1m | latest completed minute | 0.96 |
| 15m | P(positive \| active, \|R\| >= Q50) | hana__top-account-long-short-log-change-5m | HANA | cross-asset futures positioning | 5m | one completed 5m publication lag | 0.93 |
| 15m | P(positive \| active, \|R\| >= Q50) | mon__book-depth-imbalance-5pct | MON | cross-asset futures book imbalance | latest snapshot | latest snapshot in completed minute | 0.90 |
| 15m | P(positive \| active, \|R\| >= Q50) | eth__rsi-8m | ETH | cross-asset RSI | 8m recursive | latest completed minute | 0.96 |
| 15m | P(positive \| active, \|R\| >= Q75) | kaito__active-fraction-15m | KAITO | cross-asset activity | 15m | latest completed minute | 0.96 |
| 15m | P(positive \| active, \|R\| >= Q75) | yfi__fourier-energy-k1-16m | YFI | cross-asset spectral energy | 16m | latest completed minute | 0.96 |
| 15m | P(positive \| active, \|R\| >= Q75) | snxx__realized-volatility-15m | SNXX | cross-asset volatility | 15m | latest completed minute | 0.96 |
| 15m | P(positive \| active, \|R\| >= Q90) | rivn__taker-imbalance-ema-15m | RIVN | cross-asset aggressor flow | 15m recursive | latest completed minute | 0.96 |
| 15m | P(positive \| active, \|R\| >= Q90) | opn__book-notional-imbalance-4pct | OPN | cross-asset futures book imbalance | latest snapshot | latest snapshot in completed minute | 0.90 |
| 15m | P(positive \| active, \|R\| >= Q90) | b2__signed-variance-efficiency-16m | B2 | cross-asset path efficiency | 16m | latest completed minute | 0.96 |
| 15m | P(positive \| active, \|R\| < Q25) | huma__fourier-imag-k1-64m | HUMA | cross-asset spectral phase | 64m | latest completed minute | 0.96 |
| 15m | P(positive \| active, \|R\| < Q25) | ta__completed-volume-60m | TA | cross-asset volume regime | 60m | latest completed minute | 0.96 |
| 15m | P(positive \| active, \|R\| < Q25) | yb__macd-histogram-12-26-9 | YB | cross-asset MACD | 26m recursive | latest completed minute | 0.96 |
| 15m | P(positive \| active, \|R\| < Q50) | ake__realized-volatility-240m | AKE | cross-asset volatility | 240m | latest completed minute | 0.96 |
| 15m | P(positive \| active, \|R\| < Q50) | arpa__open-interest-log-level | ARPA | cross-asset futures positioning | latest 5m bucket | one completed 5m publication lag | 0.93 |
| 15m | P(positive \| active, \|R\| < Q50) | xrp__funding-absolute-rate | XRP | cross-asset funding | latest event / 24h | latest settled event | 0.95 |
| 15m | P(positive \| active, \|R\| < Q75) | zkp__open-interest-value-deviation-24h | ZKP | cross-asset futures positioning | 24h recursive | one completed 5m publication lag | 0.93 |
| 15m | P(positive \| active, \|R\| < Q75) | huma__taker-imbalance-ema-60m | HUMA | cross-asset aggressor flow | 60m recursive | latest completed minute | 0.96 |
| 15m | P(positive \| active, \|R\| < Q75) | msft__active-fraction-60m | MSFT | cross-asset activity | 60m | latest completed minute | 0.96 |
| 15m | P(\|R\| >= Q50 \| negative) | the__top-account-minus-global-long-short | THE | cross-asset futures positioning | latest 5m bucket | one completed 5m publication lag | 0.93 |
| 15m | P(\|R\| >= Q50 \| negative) | doge-realized-volatility-60m | BTC/general baseline inventory | cross-market volatility | 60m | through origin | 0.96 |
| 15m | P(\|R\| >= Q50 \| negative) | nxpc__realized-volatility-240m | NXPC | cross-asset volatility | 240m | latest completed minute | 0.96 |
| 15m | P(\|R\| >= Q75 \| negative) | ibm__book-log-notional-5pct | IBM | cross-asset futures book shape | latest snapshot | latest snapshot in completed minute | 0.90 |
| 15m | P(\|R\| >= Q75 \| negative) | bank__completed-volume-60m | BANK | cross-asset volume regime | 60m | latest completed minute | 0.96 |
| 15m | P(\|R\| >= Q75 \| negative) | cat__book-log-notional-5pct | CAT | cross-asset futures book shape | latest snapshot | latest snapshot in completed minute | 0.90 |
| 15m | P(\|R\| >= Q90 \| negative) | nxpc__realized-volatility-240m | NXPC | cross-asset volatility | 240m | latest completed minute | 0.96 |
| 15m | P(\|R\| >= Q90 \| negative) | tst__book-log-notional-1pct | TST | cross-asset futures book shape | latest snapshot | latest snapshot in completed minute | 0.90 |
| 15m | P(\|R\| >= Q90 \| negative) | hype__realized-volatility-60m | HYPE | cross-asset volatility | 60m | latest completed minute | 0.96 |
| 15m | P(\|R\| >= Q50 \| positive) | sofi__active-fraction-60m | SOFI | cross-asset activity | 60m | latest completed minute | 0.85 |
| 15m | P(\|R\| >= Q50 \| positive) | doge-realized-volatility-60m | BTC/general baseline inventory | cross-market volatility | 60m | through origin | 0.96 |
| 15m | P(\|R\| >= Q50 \| positive) | gua__open-interest-log-level | GUA | cross-asset futures positioning | latest 5m bucket | one completed 5m publication lag | 0.93 |
| 15m | P(\|R\| >= Q75 \| positive) | gua__open-interest-log-level | GUA | cross-asset futures positioning | latest 5m bucket | one completed 5m publication lag | 0.93 |
| 15m | P(\|R\| >= Q75 \| positive) | sofi__active-fraction-60m | SOFI | cross-asset activity | 60m | latest completed minute | 0.85 |
| 15m | P(\|R\| >= Q75 \| positive) | doge-realized-volatility-60m | BTC/general baseline inventory | cross-market volatility | 60m | through origin | 0.96 |
| 15m | P(\|R\| >= Q90 \| positive) | avgo__realized-volatility-30m | AVGO | cross-asset volatility | 30m | latest completed minute | 0.96 |
| 15m | P(\|R\| >= Q90 \| positive) | arm__book-log-depth-5pct | ARM | cross-asset futures book shape | latest snapshot | latest snapshot in completed minute | 0.90 |
| 15m | P(\|R\| >= Q90 \| positive) | syn__book-notional-imbalance-5pct | SYN | cross-asset futures book imbalance | latest snapshot | latest snapshot in completed minute | 0.90 |
| 1h | P(inactive) | pivx__realized-volatility-60m | PIVX | cross-asset volatility | 60m | latest completed minute | 0.96 |
| 1h | P(positive \| active) | manta__open-interest-deviation-24h | MANTA | cross-asset futures positioning | 24h recursive | one completed 5m publication lag | 0.93 |
| 1h | P(positive \| active) | snow__active-fraction-60m | SNOW | cross-asset activity | 60m | latest completed minute | 0.96 |
| 1h | P(positive \| active) | alice__rsi-32m | ALICE | cross-asset RSI | 32m recursive | latest completed minute | 0.96 |
| 1h | P(zero/sign/magnitude quartile) | sofi__taker-imbalance-ema-60m | SOFI | cross-asset aggressor flow | 60m recursive | latest completed minute | 0.85 |
| 1h | P(zero/sign/magnitude quartile) | completed-1h-log-volume | BTC/general baseline inventory | volume regime | 1h | after hour close | 1.00 |
| 1h | P(zero/sign/magnitude quartile) | sapien__top-account-long-short-log-level | SAPIEN | cross-asset futures positioning | latest 5m bucket | one completed 5m publication lag | 0.93 |
| 1h | P(\|R\| >= Q25 \| active) | bx__book-log-depth-1pct | BX | cross-asset futures book shape | latest snapshot | latest snapshot in completed minute | 0.90 |
| 1h | P(\|R\| >= Q25 \| active) | eth__realized-volatility-15s | ETH | cross-asset fast volatility | 15s | latest completed second at minute origin | 0.95 |
| 1h | P(\|R\| >= Q25 \| active) | completed-1h-log-volume | BTC/general baseline inventory | volume regime | 1h | after hour close | 1.00 |
| 1h | P(\|R\| >= Q50 \| active) | bank__log-trade-count-1m | BANK | cross-asset activity regime | 1m | latest completed minute | 0.96 |
| 1h | P(\|R\| >= Q50 \| active) | bttc__efficiency-ratio-60m | BTTC | cross-asset path efficiency | 60m | latest completed minute | 0.96 |
| 1h | P(\|R\| >= Q50 \| active) | ema-slope-8s-8s | BTC/general baseline inventory | price dynamics | recursive | through origin | 1.00 |
| 1h | P(\|R\| >= Q75 \| active) | bz__active-fraction-60m | BZ | cross-asset activity | 60m | latest completed minute | 0.96 |
| 1h | P(\|R\| >= Q75 \| active) | range-1m | BTC/general baseline inventory | minute candle shape | 1m | latest completed minute | 1.00 |
| 1h | P(\|R\| >= Q75 \| active) | bttc__efficiency-ratio-60m | BTTC | cross-asset path efficiency | 60m | latest completed minute | 0.96 |
| 1h | P(\|R\| >= Q90 \| active) | ibm__book-log-notional-5pct | IBM | cross-asset futures book shape | latest snapshot | latest snapshot in completed minute | 0.90 |
| 1h | P(\|R\| >= Q90 \| active) | eth__realized-volatility-15m | ETH | cross-asset volatility | 15m | latest completed minute | 0.96 |
| 1h | P(\|R\| >= Q90 \| active) | zbt__top-account-minus-global-long-short | ZBT | cross-asset futures positioning | latest 5m bucket | one completed 5m publication lag | 0.93 |
| 1h | P(positive \| active, \|R\| >= Q50) | ub__return-kurtosis-60m | UB | cross-asset return shape | 60m | latest completed minute | 0.96 |
| 1h | P(positive \| active, \|R\| >= Q50) | hana__completed-volume-60m | HANA | cross-asset volume regime | 60m | latest completed minute | 0.96 |
| 1h | P(positive \| active, \|R\| >= Q50) | tst__taker-base-imbalance-1s | TST | cross-asset fast aggressor flow | 1s | latest completed second at minute origin | 0.95 |
| 1h | P(positive \| active, \|R\| >= Q75) | stg__book-log-depth-5pct | STG | cross-asset futures book shape | latest snapshot | latest snapshot in completed minute | 0.90 |
| 1h | P(positive \| active, \|R\| >= Q75) | mon__rsi-32m | MON | cross-asset RSI | 32m recursive | latest completed minute | 0.96 |
| 1h | P(positive \| active, \|R\| >= Q75) | fight__global-long-short-log-change-240m | FIGHT | cross-asset futures positioning | 240m | one completed 5m publication lag | 0.93 |
| 1h | P(positive \| active, \|R\| >= Q90) | zk__active-fraction-60m | ZK | cross-asset activity | 60m | latest completed minute | 0.96 |
| 1h | P(positive \| active, \|R\| >= Q90) | open__open-interest-value-log-change-60m | OPEN | cross-asset futures positioning | 60m | one completed 5m publication lag | 0.93 |
| 1h | P(positive \| active, \|R\| >= Q90) | snxx__completed-volume-60m | SNXX | cross-asset volume regime | 60m | latest completed minute | 0.96 |
| 1h | P(positive \| active, \|R\| < Q25) | rave__open-interest-value-deviation-24h | RAVE | cross-asset futures positioning | 24h recursive | one completed 5m publication lag | 0.93 |
| 1h | P(positive \| active, \|R\| < Q25) | b2__completed-volume-60m | B2 | cross-asset volume regime | 60m | latest completed minute | 0.96 |
| 1h | P(positive \| active, \|R\| < Q25) | crwd__return-kurtosis-60m | CRWD | cross-asset return shape | 60m | latest completed minute | 0.96 |
| 1h | P(positive \| active, \|R\| < Q50) | ctsi__return-kurtosis-60m | CTSI | cross-asset return shape | 60m | latest completed minute | 0.96 |
| 1h | P(positive \| active, \|R\| < Q50) | glw__completed-volume-60m | GLW | cross-asset volume regime | 60m | latest completed minute | 0.96 |
| 1h | P(positive \| active, \|R\| < Q50) | bttc__efficiency-ratio-60m | BTTC | cross-asset path efficiency | 60m | latest completed minute | 0.96 |
| 1h | P(positive \| active, \|R\| < Q75) | cati__funding-change | CATI | cross-asset funding | latest event / 24h | latest settled event | 0.95 |
| 1h | P(positive \| active, \|R\| < Q75) | space__open-interest-value-deviation-24h | SPACE | cross-asset futures positioning | 24h recursive | one completed 5m publication lag | 0.93 |
| 1h | P(positive \| active, \|R\| < Q75) | onds__trade-count-surprise-60m | ONDS | cross-asset activity regime | 60m recursive | latest completed minute | 0.96 |
| 1h | P(\|R\| >= Q50 \| negative) | bank__completed-volume-60m | BANK | cross-asset volume regime | 60m | latest completed minute | 0.96 |
| 1h | P(\|R\| >= Q50 \| negative) | nxpc__realized-volatility-240m | NXPC | cross-asset volatility | 240m | latest completed minute | 0.96 |
| 1h | P(\|R\| >= Q50 \| negative) | jto__open-interest-log-level | JTO | cross-asset futures positioning | latest 5m bucket | one completed 5m publication lag | 0.93 |
| 1h | P(\|R\| >= Q75 \| negative) | jto__open-interest-log-level | JTO | cross-asset futures positioning | latest 5m bucket | one completed 5m publication lag | 0.93 |
| 1h | P(\|R\| >= Q75 \| negative) | bttc__efficiency-ratio-60m | BTTC | cross-asset path efficiency | 60m | latest completed minute | 0.96 |
| 1h | P(\|R\| >= Q75 \| negative) | bz__log-mean-trade-notional-1m | BZ | cross-asset trade structure | 1m | latest completed minute | 0.96 |
| 1h | P(\|R\| >= Q90 \| negative) | nxpc__realized-volatility-240m | NXPC | cross-asset volatility | 240m | latest completed minute | 0.96 |
| 1h | P(\|R\| >= Q90 \| negative) | yb__global-long-short-log-level | YB | cross-asset futures positioning | latest 5m bucket | one completed 5m publication lag | 0.93 |
| 1h | P(\|R\| >= Q90 \| negative) | zama__top-position-minus-account-long-short | ZAMA | cross-asset futures positioning | latest 5m bucket | one completed 5m publication lag | 0.93 |
| 1h | P(\|R\| >= Q50 \| positive) | bank__completed-volume-60m | BANK | cross-asset volume regime | 60m | latest completed minute | 0.96 |
| 1h | P(\|R\| >= Q50 \| positive) | dexe__realized-volatility-60m | DEXE | cross-asset volatility | 60m | latest completed minute | 0.96 |
| 1h | P(\|R\| >= Q50 \| positive) | ibm__book-log-notional-5pct | IBM | cross-asset futures book shape | latest snapshot | latest snapshot in completed minute | 0.90 |
| 1h | P(\|R\| >= Q75 \| positive) | rave__funding-absolute-rate | RAVE | cross-asset funding | latest event / 24h | latest settled event | 0.95 |
| 1h | P(\|R\| >= Q75 \| positive) | mtl__top-position-minus-account-long-short | MTL | cross-asset futures positioning | latest 5m bucket | one completed 5m publication lag | 0.93 |
| 1h | P(\|R\| >= Q75 \| positive) | avgo__realized-volatility-60m | AVGO | cross-asset volatility | 60m | latest completed minute | 0.96 |
| 1h | P(\|R\| >= Q90 \| positive) | minimax__book-notional-imbalance-5pct | MINIMAX | cross-asset futures book imbalance | latest snapshot | latest snapshot in completed minute | 0.90 |
| 1h | P(\|R\| >= Q90 \| positive) | doge-realized-volatility-30m | BTC/general baseline inventory | cross-market volatility | 30m | through origin | 0.96 |
| 1h | P(\|R\| >= Q90 \| positive) | pixel__open-interest-log-level | PIXEL | cross-asset futures positioning | latest 5m bucket | one completed 5m publication lag | 0.93 |

## Aggregate held-out comparison

| Horizon | Heads | Positive-vs-marginal heads | Heads using cross-market input | Mean previous transfer bits | Mean refit transfer bits | Mean change |
|---|---:|---:|---:|---:|---:|---:|
| 1s | 19 | 14 | 11 | 0.156864 | 0.068014 | -0.088851 |
| 1m | 19 | 19 | 16 | 0.083404 | 0.080605 | -0.002798 |
| 15m | 19 | 8 | 19 | 0.058922 | 0.050956 | -0.007965 |
| 1h | 19 | 4 | 19 | 0.038059 | -0.038989 | -0.077048 |

## Feature-family coverage

| Examined family | Coordinates |
|---|---:|
| activity | 7 |
| aggressor change | 1 |
| aggressor direction | 12 |
| basis | 5 |
| candle shape | 2 |
| cross-asset EMA acceleration | 256 |
| cross-asset EMA slope | 256 |
| cross-asset EMA value | 768 |
| cross-asset MACD | 520 |
| cross-asset RSI | 780 |
| cross-asset activity | 1,040 |
| cross-asset activity regime | 1,024 |
| cross-asset aggressor flow | 1,300 |
| cross-asset candle shape | 516 |
| cross-asset fast EMA acceleration | 139 |
| cross-asset fast EMA slope | 139 |
| cross-asset fast RSI | 139 |
| cross-asset fast activity | 556 |
| cross-asset fast activity regime | 278 |
| cross-asset fast aggressor flow | 695 |
| cross-asset fast candle shape | 278 |
| cross-asset fast return | 278 |
| cross-asset fast volatility | 417 |
| cross-asset fast wavelet | 139 |
| cross-asset funding | 932 |
| cross-asset funding availability | 233 |
| cross-asset futures basis | 575 |
| cross-asset futures book change | 1,155 |
| cross-asset futures book imbalance | 2,310 |
| cross-asset futures book shape | 2,310 |
| cross-asset futures positioning | 7,437 |
| cross-asset path efficiency | 780 |
| cross-asset return | 1,024 |
| cross-asset return shape | 520 |
| cross-asset spectral energy | 520 |
| cross-asset spectral phase | 1,040 |
| cross-asset trade structure | 256 |
| cross-asset venue activity | 115 |
| cross-asset venue lead-lag | 115 |
| cross-asset volatility | 1,280 |
| cross-asset volume regime | 256 |
| cross-asset wavelet | 520 |
| cross-market activity | 1 |
| cross-market return | 9 |
| cross-market volatility | 8 |
| futures activity | 5 |
| futures candle | 2 |
| futures flow | 5 |
| futures price | 1 |
| lagged aggressor direction | 6 |
| lagged trade sequence | 3 |
| minute candle shape | 1 |
| minute volatility | 4 |
| open interest | 11 |
| positioning ratio | 16 |
| positioning spread | 2 |
| price dynamics | 3 |
| price pressure | 1 |
| return history | 1 |
| source metadata | 2 |
| spot book book change | 7 |
| spot book depth shape | 5 |
| spot book liquidity | 4 |
| spot book order flow | 1 |
| spot book queue imbalance | 8 |
| spot book top of book | 2 |
| timing | 2 |
| trade sequence | 3 |
| trade size | 4 |
| trade structure | 1 |
| volatility | 1 |
| volume regime | 1 |

## Canonical asset universe

This is the latest 257-asset independent 1m basis plus the four explicit additions. A blank rank marks an explicit addition rather than a member of the source basis.

| Rank | Asset | Explicit addition | Preferred market | Available products |
|---:|---|---|---|---|
| 1 | BTC | no | spot | spot:BTCUSDT, usdm-futures:BTCUSDT |
| 2 | XNO | no | spot | spot:XNOUSDT |
| 3 | BTTC | no | spot | spot:BTTCUSDT |
| 4 | RIF | no | spot | spot:RIFUSDT, usdm-futures:RIFUSDT |
| 5 | B2 | no | usdm-futures | usdm-futures:B2USDT |
| 6 | BANK | no | spot | spot:BANKUSDT, usdm-futures:BANKUSDT |
| 7 | PYR | no | spot | spot:PYRUSDT |
| 8 | ONE | no | spot | spot:ONEUSDT, usdm-futures:ONEUSDT |
| 9 | DEXE | no | spot | spot:DEXEUSDT, usdm-futures:DEXEUSDT |
| 10 | AKE | no | usdm-futures | usdm-futures:AKEUSDT |
| 11 | ON | no | usdm-futures | usdm-futures:ONUSDT |
| 12 | ERA | no | spot | spot:ERAUSDT, usdm-futures:ERAUSDT |
| 13 | BLESS | no | usdm-futures | usdm-futures:BLESSUSDT |
| 14 | BROCCOLIF3B | no | usdm-futures | usdm-futures:BROCCOLIF3BUSDT |
| 15 | SHAZ | no | usdm-futures | usdm-futures:SHAZUSDT |
| 16 | ESPORTS | no | usdm-futures | usdm-futures:ESPORTSUSDT |
| 17 | DODO | no | spot | spot:DODOUSDT |
| 18 | MIRA | no | spot | spot:MIRAUSDT, usdm-futures:MIRAUSDT |
| 19 | SAPIEN | no | spot | spot:SAPIENUSDT, usdm-futures:SAPIENUSDT |
| 20 | DGB | no | spot | spot:DGBUSDT |
| 21 | SNXX | no | usdm-futures | spot:SNXXBUSDT, usdm-futures:SNXXUSDT |
| 22 | LAB | no | usdm-futures | usdm-futures:LABUSDT |
| 23 | QNTB | no | spot | spot:QNTBUSDT |
| 24 | AIA | no | usdm-futures | usdm-futures:AIAUSDT |
| 25 | STBL | no | usdm-futures | usdm-futures:STBLUSDT |
| 26 | NIGHT | no | spot | spot:NIGHTUSDT, usdm-futures:NIGHTUSDT |
| 27 | BAS | no | usdm-futures | usdm-futures:BASUSDT |
| 28 | ZAMA | no | spot | spot:ZAMAUSDT, usdm-futures:ZAMAUSDT |
| 29 | SPELL | no | spot | spot:SPELLUSDT, usdm-futures:SPELLUSDT |
| 30 | STAR | no | usdm-futures | usdm-futures:STARUSDT |
| 31 | NAORIS | no | usdm-futures | usdm-futures:NAORISUSDT |
| 32 | BSB | no | usdm-futures | usdm-futures:BSBUSDT |
| 33 | CAT | no | spot | spot:1000CATUSDT, usdm-futures:CATUSDT |
| 34 | BULLA | no | usdm-futures | usdm-futures:BULLAUSDT |
| 35 | EVAA | no | usdm-futures | usdm-futures:EVAAUSDT |
| 36 | BEAT | no | usdm-futures | usdm-futures:BEATUSDT |
| 37 | RAD | no | spot | spot:RADUSDT |
| 38 | ZHIPU | no | usdm-futures | usdm-futures:ZHIPUUSDT |
| 39 | XNY | no | usdm-futures | usdm-futures:XNYUSDT |
| 40 | OPN | no | spot | spot:OPNUSDT, usdm-futures:OPNUSDT |
| 41 | MBL | no | spot | spot:MBLUSDT |
| 42 | HANA | no | usdm-futures | usdm-futures:HANAUSDT |
| 43 | B | no | usdm-futures | usdm-futures:BUSDT |
| 44 | WIN | no | spot | spot:WINUSDT |
| 45 | 龙虾 | no | usdm-futures | usdm-futures:龙虾USDT |
| 46 | TKO | no | spot | spot:TKOUSDT |
| 47 | IBM | no | spot | spot:IBMBUSDT, usdm-futures:IBMUSDT |
| 48 | MINIMAX | no | usdm-futures | usdm-futures:MINIMAXUSDT |
| 49 | ANTHROPIC | no | usdm-futures | usdm-futures:ANTHROPICUSDT |
| 50 | XPIN | no | usdm-futures | usdm-futures:XPINUSDT |
| 51 | ALLO | no | spot | spot:ALLOUSDT, usdm-futures:ALLOUSDT |
| 52 | NOK | no | spot | spot:NOKBUSDT, usdm-futures:NOKUSDT |
| 53 | TA | no | usdm-futures | usdm-futures:TAUSDT |
| 54 | HEI | no | spot | spot:HEIUSDT, usdm-futures:HEIUSDT |
| 55 | SPORTFUN | no | usdm-futures | usdm-futures:SPORTFUNUSDT |
| 56 | BAN | no | usdm-futures | usdm-futures:BANUSDT |
| 57 | TAKE | no | usdm-futures | usdm-futures:TAKEUSDT |
| 58 | THE | no | spot | spot:THEUSDT, usdm-futures:THEUSDT |
| 59 | AI | no | spot | spot:AIUSDT |
| 60 | GUN | no | spot | spot:GUNUSDT, usdm-futures:GUNUSDT |
| 61 | GUA | no | usdm-futures | usdm-futures:GUAUSDT |
| 62 | IDOL | no | usdm-futures | usdm-futures:IDOLUSDT |
| 63 | ZBT | no | spot | spot:ZBTUSDT, usdm-futures:ZBTUSDT |
| 64 | TRUTH | no | usdm-futures | usdm-futures:TRUTHUSDT |
| 65 | REQ | no | spot | spot:REQUSDT |
| 66 | SOMI | no | spot | spot:SOMIUSDT, usdm-futures:SOMIUSDT |
| 67 | NVO | no | usdm-futures | usdm-futures:NVOUSDT |
| 68 | 币安人生 | no | spot | spot:币安人生USDT, usdm-futures:币安人生USDT |
| 69 | IOTA | no | spot | spot:IOTAUSDT, usdm-futures:IOTAUSDT |
| 70 | CITY | no | spot | spot:CITYUSDT |
| 71 | WLFI | no | spot | spot:WLFIUSDT, usdm-futures:WLFIUSDT |
| 72 | ARPA | no | spot | spot:ARPAUSDT, usdm-futures:ARPAUSDT |
| 73 | AVGO | no | spot | spot:AVGOBUSDT, usdm-futures:AVGOUSDT |
| 74 | FWDI | no | usdm-futures | usdm-futures:FWDIUSDT |
| 75 | ACH | no | spot | spot:ACHUSDT, usdm-futures:ACHUSDT |
| 76 | NXPC | no | spot | spot:NXPCUSDT, usdm-futures:NXPCUSDT |
| 77 | YB | no | spot | spot:YBUSDT, usdm-futures:YBUSDT |
| 78 | SKL | no | spot | spot:SKLUSDT, usdm-futures:SKLUSDT |
| 79 | SNOW | no | usdm-futures | usdm-futures:SNOWUSDT |
| 80 | ILV | no | spot | spot:ILVUSDT, usdm-futures:ILVUSDT |
| 81 | XAN | no | usdm-futures | usdm-futures:XANUSDT |
| 82 | SNX | no | spot | spot:SNXUSDT, usdm-futures:SNXUSDT |
| 83 | OPENAI | no | usdm-futures | usdm-futures:OPENAIUSDT |
| 84 | SPACE | no | usdm-futures | usdm-futures:SPACEUSDT |
| 85 | GRAM | no | spot | spot:GRAMUSDT, usdm-futures:GRAMUSDT |
| 86 | ZEST | no | usdm-futures | usdm-futures:ZESTUSDT |
| 87 | PYTH | no | spot | spot:PYTHUSDT, usdm-futures:PYTHUSDT |
| 88 | ASML | no | usdm-futures | usdm-futures:ASMLUSDT |
| 89 | SOFI | no | usdm-futures | usdm-futures:SOFIUSDT |
| 90 | ID | no | spot | spot:IDUSDT, usdm-futures:IDUSDT |
| 91 | VTHO | no | spot | spot:VTHOUSDT, usdm-futures:VTHOUSDT |
| 92 | HOT | no | spot | spot:HOTUSDT, usdm-futures:HOTUSDT |
| 93 | WAXP | no | spot | spot:WAXPUSDT, usdm-futures:WAXPUSDT |
| 94 | CATI | no | spot | spot:CATIUSDT, usdm-futures:CATIUSDT |
| 95 | AWE | no | spot | spot:AWEUSDT, usdm-futures:AWEUSDT |
| 96 | BRKB | no | usdm-futures | usdm-futures:BRKBUSDT |
| 97 | AIGENSYN | no | spot | spot:AIGENSYNUSDT, usdm-futures:AIGENSYNUSDT |
| 98 | PROM | no | spot | spot:PROMUSDT, usdm-futures:PROMUSDT |
| 99 | BTW | no | usdm-futures | usdm-futures:BTWUSDT |
| 100 | PIVX | no | spot | spot:PIVXUSDT |
| 101 | ONDS | no | usdm-futures | usdm-futures:ONDSUSDT |
| 102 | FIGHT | no | usdm-futures | usdm-futures:FIGHTUSDT |
| 103 | BILL | no | usdm-futures | usdm-futures:BILLUSDT |
| 104 | POWER | no | usdm-futures | usdm-futures:POWERUSDT |
| 105 | MYX | no | usdm-futures | usdm-futures:MYXUSDT |
| 106 | ACM | no | spot | spot:ACMUSDT |
| 107 | T | no | spot | spot:TUSDT, usdm-futures:TUSDT |
| 108 | BSP | no | usdm-futures | usdm-futures:BSPUSDT |
| 109 | MET | no | spot | spot:METUSDT, usdm-futures:METUSDT |
| 110 | KOMA | no | usdm-futures | usdm-futures:KOMAUSDT |
| 111 | GENIUS | no | spot | spot:GENIUSUSDT, usdm-futures:GENIUSUSDT |
| 112 | BAR | no | spot | spot:BARUSDT |
| 113 | STABLE | no | usdm-futures | usdm-futures:STABLEUSDT |
| 114 | LYN | no | usdm-futures | usdm-futures:LYNUSDT |
| 115 | CYS | no | usdm-futures | usdm-futures:CYSUSDT |
| 116 | SYN | no | spot | spot:SYNUSDT, usdm-futures:SYNUSDT |
| 117 | CTR | no | usdm-futures | usdm-futures:CTRUSDT |
| 118 | ROBO | no | spot | spot:ROBOUSDT, usdm-futures:ROBOUSDT |
| 119 | BANANAS31 | no | spot | spot:BANANAS31USDT, usdm-futures:BANANAS31USDT |
| 120 | GPS | no | spot | spot:GPSUSDT, usdm-futures:GPSUSDT |
| 121 | MAVIA | no | usdm-futures | usdm-futures:MAVIAUSDT |
| 122 | IRYS | no | usdm-futures | usdm-futures:IRYSUSDT |
| 123 | TST | no | spot | spot:TSTUSDT, usdm-futures:TSTUSDT |
| 124 | FLUID | no | usdm-futures | usdm-futures:FLUIDUSDT |
| 125 | OSMO | no | spot | spot:OSMOUSDT |
| 126 | Q | no | usdm-futures | usdm-futures:QUSDT |
| 127 | PHA | no | spot | spot:PHAUSDT, usdm-futures:PHAUSDT |
| 128 | PUMPBTC | no | usdm-futures | usdm-futures:PUMPBTCUSDT |
| 129 | NOM | no | spot | spot:NOMUSDT, usdm-futures:NOMUSDT |
| 130 | KSM | no | spot | spot:KSMUSDT, usdm-futures:KSMUSDT |
| 131 | ZK | no | spot | spot:ZKUSDT, usdm-futures:ZKUSDT |
| 132 | CIEN | no | usdm-futures | usdm-futures:CIENUSDT |
| 133 | TFUEL | no | spot | spot:TFUELUSDT |
| 134 | HK1810 | no | usdm-futures | usdm-futures:HK1810USDT |
| 135 | WEN | no | usdm-futures | usdm-futures:WENUSDT |
| 136 | WAL | no | spot | spot:WALUSDT, usdm-futures:WALUSDT |
| 137 | TURTLE | no | spot | spot:TURTLEUSDT, usdm-futures:TURTLEUSDT |
| 138 | RIVN | no | usdm-futures | usdm-futures:RIVNUSDT |
| 139 | BZ | no | usdm-futures | usdm-futures:BZUSDT |
| 140 | CC | no | usdm-futures | usdm-futures:CCUSDT |
| 141 | O | no | usdm-futures | usdm-futures:OUSDT |
| 142 | MITO | no | spot | spot:MITOUSDT, usdm-futures:MITOUSDT |
| 143 | XMR | no | usdm-futures | usdm-futures:XMRUSDT |
| 144 | PORTAL | no | spot | spot:PORTALUSDT, usdm-futures:PORTALUSDT |
| 145 | HMSTR | no | spot | spot:HMSTRUSDT, usdm-futures:HMSTRUSDT |
| 146 | ARM | no | spot | spot:ARMBUSDT, usdm-futures:ARMUSDT |
| 147 | BX | no | usdm-futures | usdm-futures:BXUSDT |
| 148 | YFI | no | spot | spot:YFIUSDT, usdm-futures:YFIUSDT |
| 149 | AT | no | spot | spot:ATUSDT, usdm-futures:ATUSDT |
| 150 | NATGAS | no | usdm-futures | usdm-futures:NATGASUSDT |
| 151 | FORM | no | spot | spot:FORMUSDT, usdm-futures:FORMUSDT |
| 152 | QKC | no | spot | spot:QKCUSDT |
| 153 | USTC | no | spot | spot:USTCUSDT, usdm-futures:USTCUSDT |
| 154 | ICX | no | spot | spot:ICXUSDT, usdm-futures:ICXUSDT |
| 155 | UBER | no | usdm-futures | usdm-futures:UBERUSDT |
| 156 | SIGN | no | spot | spot:SIGNUSDT, usdm-futures:SIGNUSDT |
| 157 | NOW | no | usdm-futures | usdm-futures:NOWUSDT |
| 158 | SAFE | no | usdm-futures | usdm-futures:SAFEUSDT |
| 159 | GLMR | no | spot | spot:GLMRUSDT |
| 160 | HEMI | no | spot | spot:HEMIUSDT, usdm-futures:HEMIUSDT |
| 161 | AIOT | no | usdm-futures | usdm-futures:AIOTUSDT |
| 162 | INX | no | usdm-futures | usdm-futures:INXUSDT |
| 163 | ICNT | no | usdm-futures | usdm-futures:ICNTUSDT |
| 164 | JTO | no | spot | spot:JTOUSDT, usdm-futures:JTOUSDT |
| 165 | SOPH | no | spot | spot:SOPHUSDT, usdm-futures:SOPHUSDT |
| 166 | AXL | no | spot | spot:AXLUSDT, usdm-futures:AXLUSDT |
| 167 | HYUNDAI | no | usdm-futures | usdm-futures:HYUNDAIUSDT |
| 168 | MANTA | no | spot | spot:MANTAUSDT, usdm-futures:MANTAUSDT |
| 169 | SATS | no | spot | spot:1000SATSUSDT, usdm-futures:1000SATSUSDT |
| 170 | LLY | no | usdm-futures | usdm-futures:LLYUSDT |
| 171 | PORTO | no | spot | spot:PORTOUSDT |
| 172 | RECALL | no | usdm-futures | usdm-futures:RECALLUSDT |
| 173 | DIA | no | spot | spot:DIAUSDT, usdm-futures:DIAUSDT |
| 174 | TRADOOR | no | usdm-futures | usdm-futures:TRADOORUSDT |
| 175 | FOLKS | no | usdm-futures | usdm-futures:FOLKSUSDT |
| 176 | BNC | no | usdm-futures | usdm-futures:BNCUSDT |
| 177 | EDEN | no | spot | spot:EDENUSDT, usdm-futures:EDENUSDT |
| 178 | RAY | no | spot | spot:RAYUSDT |
| 179 | REZ | no | spot | spot:REZUSDT, usdm-futures:REZUSDT |
| 180 | RESOLV | no | spot | spot:RESOLVUSDT, usdm-futures:RESOLVUSDT |
| 181 | PIEVERSE | no | usdm-futures | usdm-futures:PIEVERSEUSDT |
| 182 | DCR | no | spot | spot:DCRUSDT |
| 183 | ORCL | no | usdm-futures | spot:ORCLBUSDT, usdm-futures:ORCLUSDT |
| 184 | CRWD | no | usdm-futures | usdm-futures:CRWDUSDT |
| 185 | MOCA | no | usdm-futures | usdm-futures:MOCAUSDT |
| 186 | XAG | no | usdm-futures | usdm-futures:XAGUSDT |
| 187 | ZKP | no | spot | spot:ZKPUSDT, usdm-futures:ZKPUSDT |
| 188 | GAS | no | spot | spot:GASUSDT, usdm-futures:GASUSDT |
| 189 | G | no | spot | spot:GUSDT, usdm-futures:GUSDT |
| 190 | FHE | no | usdm-futures | usdm-futures:FHEUSDT |
| 191 | HD | no | usdm-futures | usdm-futures:HDUSDT |
| 192 | MSFT | no | spot | spot:MSFTBUSDT, usdm-futures:MSFTUSDT |
| 193 | BABA | no | spot | spot:BABABUSDT, usdm-futures:BABAUSDT |
| 194 | ARK | no | spot | spot:ARKUSDT, usdm-futures:ARKUSDT |
| 195 | CYBER | no | spot | spot:CYBERUSDT, usdm-futures:CYBERUSDT |
| 196 | KAITO | no | spot | spot:KAITOUSDT, usdm-futures:KAITOUSDT |
| 197 | XEC | no | spot | spot:XECUSDT, usdm-futures:1000XECUSDT |
| 198 | HUMA | no | spot | spot:HUMAUSDT, usdm-futures:HUMAUSDT |
| 199 | MMT | no | spot | spot:MMTUSDT, usdm-futures:MMTUSDT |
| 200 | DOLO | no | spot | spot:DOLOUSDT, usdm-futures:DOLOUSDT |
| 201 | RED | no | spot | spot:REDUSDT, usdm-futures:REDUSDT |
| 202 | CHIP | no | spot | spot:CHIPUSDT, usdm-futures:CHIPUSDT |
| 203 | ALPINE | no | spot | spot:ALPINEUSDT, usdm-futures:ALPINEUSDT |
| 204 | TRIA | no | usdm-futures | usdm-futures:TRIAUSDT |
| 205 | 我踏马来了 | no | usdm-futures | usdm-futures:我踏马来了USDT |
| 206 | CTSI | no | spot | spot:CTSIUSDT, usdm-futures:CTSIUSDT |
| 207 | TRX | no | spot | spot:TRXUSDT, usdm-futures:TRXUSDT |
| 208 | TNSR | no | spot | spot:TNSRUSDT, usdm-futures:TNSRUSDT |
| 209 | WMT | no | usdm-futures | usdm-futures:WMTUSDT |
| 210 | UMA | no | spot | spot:UMAUSDT, usdm-futures:UMAUSDT |
| 211 | ARX | no | usdm-futures | usdm-futures:ARXUSDT |
| 212 | ZM | no | usdm-futures | usdm-futures:ZMUSDT |
| 213 | RPL | no | spot | spot:RPLUSDT, usdm-futures:RPLUSDT |
| 214 | LIGHT | no | usdm-futures | usdm-futures:LIGHTUSDT |
| 215 | JELLYJELLY | no | usdm-futures | usdm-futures:JELLYJELLYUSDT |
| 216 | HPE | no | usdm-futures | usdm-futures:HPEUSDT |
| 217 | PARTI | no | spot | spot:PARTIUSDT, usdm-futures:PARTIUSDT |
| 218 | GLW | no | spot | spot:GLWBUSDT, usdm-futures:GLWUSDT |
| 219 | IOTX | no | spot | spot:IOTXUSDT, usdm-futures:IOTXUSDT |
| 220 | ATH | no | usdm-futures | usdm-futures:ATHUSDT |
| 221 | PIXEL | no | spot | spot:PIXELUSDT, usdm-futures:PIXELUSDT |
| 222 | V | no | usdm-futures | usdm-futures:VUSDT |
| 223 | UAI | no | usdm-futures | usdm-futures:UAIUSDT |
| 224 | UB | no | usdm-futures | usdm-futures:UBUSDT |
| 225 | ACE | no | spot | spot:ACEUSDT, usdm-futures:ACEUSDT |
| 226 | RAVE | no | usdm-futures | usdm-futures:RAVEUSDT |
| 227 | SQD | no | usdm-futures | usdm-futures:SQDUSDT |
| 228 | VANRY | no | spot | spot:VANRYUSDT, usdm-futures:VANRYUSDT |
| 229 | WET | no | usdm-futures | usdm-futures:WETUSDT |
| 230 | RIVER | no | usdm-futures | usdm-futures:RIVERUSDT |
| 231 | JOE | no | spot | spot:JOEUSDT, usdm-futures:JOEUSDT |
| 232 | ENSO | no | spot | spot:ENSOUSDT, usdm-futures:ENSOUSDT |
| 233 | CSCO | no | usdm-futures | usdm-futures:CSCOUSDT |
| 234 | STG | no | spot | spot:STGUSDT, usdm-futures:STGUSDT |
| 235 | COMP | no | spot | spot:COMPUSDT, usdm-futures:COMPUSDT |
| 236 | MTL | no | spot | spot:MTLUSDT, usdm-futures:MTLUSDT |
| 237 | METIS | no | spot | spot:METISUSDT, usdm-futures:METISUSDT |
| 238 | STRC | no | usdm-futures | usdm-futures:STRCUSDT |
| 239 | MVLL | no | usdm-futures | spot:MVLLBUSDT, usdm-futures:MVLLUSDT |
| 240 | MON | no | usdm-futures | usdm-futures:MONUSDT |
| 241 | SKR | no | usdm-futures | usdm-futures:SKRUSDT |
| 242 | COAI | no | usdm-futures | usdm-futures:COAIUSDT |
| 243 | M | no | usdm-futures | usdm-futures:MUSDT |
| 244 | ORDER | no | usdm-futures | usdm-futures:ORDERUSDT |
| 245 | TUT | no | spot | spot:TUTUSDT, usdm-futures:TUTUSDT |
| 246 | KSTR | no | usdm-futures | usdm-futures:KSTRUSDT |
| 247 | KERNEL | no | spot | spot:KERNELUSDT, usdm-futures:KERNELUSDT |
| 248 | MELANIA | no | usdm-futures | usdm-futures:MELANIAUSDT |
| 249 | NEXO | no | spot | spot:NEXOUSDT |
| 250 | OPEN | no | spot | spot:OPENUSDT, usdm-futures:OPENUSDT |
| 251 | TAC | no | usdm-futures | usdm-futures:TACUSDT |
| 252 | ETHW | no | usdm-futures | usdm-futures:ETHWUSDT |
| 253 | KAT | no | spot | spot:KATUSDT, usdm-futures:KATUSDT |
| 254 | USELESS | no | usdm-futures | usdm-futures:USELESSUSDT |
| 255 | PHAROS | no | usdm-futures | usdm-futures:PHAROSUSDT |
| 256 | ALICE | no | spot | spot:ALICEUSDT, usdm-futures:ALICEUSDT |
| 257 | JUV | no | spot | spot:JUVUSDT |
|  | ETH | yes | spot | spot:ETHUSDT, usdm-futures:ETHUSDT |
|  | SOL | yes | spot | spot:SOLUSDT, usdm-futures:SOLUSDT |
|  | XRP | yes | spot | spot:XRPUSDT, usdm-futures:XRPUSDT |
|  | HYPE | yes | usdm-futures | usdm-futures:HYPEUSDT |

## Availability boundary

The per-asset replication covers official completed Binance spot/USD-M klines, spot 1-second fast state, USD-M 5-minute positioning/open-interest metrics, settled funding, paired-venue basis/lead-lag, and percentage-depth snapshots wherever each source exists. Historical order additions/cancellations, liquidation events, exact aggregate-trade sequence/size shape, options surfaces, chain-specific flows, and non-Binance books are not uniformly available for this universe and are not fabricated; the corresponding BTC-only/live audits remain explicit separate tiers.

The near-tie rule prefers reproducible candle/archive sources over local/live-only feeds and incorporates observed 30-day coverage. It does not manufacture pre-listing history: an asset-specific coordinate is unavailable before that market was listed. This is another reason the recent cross-asset winners remain discovery candidates and the older established production basis is retained.

All candidate marginals are scored; joint subset enumeration is exact only inside each head's 16 finalists and size-three limit. Transfer data never chooses candidates or subsets. A `primary-only` row is discovery evidence, not a production promotion.

Machine-readable catalog and results: `data/benchmarks/binance-cross-asset-component-feature-bases.json`.
