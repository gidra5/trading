# Tiered component feature bases

Generated `2026-08-19T19:23:55Z`.

This search evaluates every coordinate in the recent broad tier separately for each return component. The final transfer block is not used for selection. Within 0.001 primary bits, broader availability wins.

## Dataset and selection

- `recent`: 42,901 origins, 147 coordinates; core-candle-derived=15, free-official-archive=103, local-snapshot-only=29.

The search is exact only inside each component's 12 family-aware finalists and subsets of size at most three; all broad coordinates are nevertheless scored marginally before finalist selection.

The availability rule changed 12 of 75 eligible selections. Its mean/max held-out primary-score sacrifice was 0.000438/0.000932 bits per target.

## Selected bases

| Horizon | Component | Status | Selected inputs | Primary bits | Transfer bits | Availability min/mean |
|---|---|---|---|---:|---:|---:|
| 1s | P(inactive) | confirmed-early | active-count-60s, realized-volatility-30m, ema-acceleration-2s-1s | 0.028371 | 0.048499 | 1.00/1.00 |
| 1s | P(positive \| active) | confirmed-early | spot-flow-last-side-1s, spot-flow-maximum-skew-1s, futures-basis-deviation-5m | 0.172683 | 0.194169 | 0.95/0.95 |
| 1s | P(zero/sign/magnitude quartile) | confirmed-early | spot-flow-last-side-1s, spot-book-spread-bps, range-1s | 0.312569 | 0.131251 | 0.45/0.80 |
| 1s | P(\|R\| >= Q25 \| active) | confirmed-early | spot-book-spread-bps, range-1s, top-position-minus-account-log | 0.348349 | 0.270913 | 0.45/0.80 |
| 1s | P(\|R\| >= Q50 \| active) | primary-only | spot-book-spread-bps, eth-realized-volatility-60m, ema-slope-8s-8s | 0.212103 | -0.091990 | 0.45/0.80 |
| 1s | P(\|R\| >= Q75 \| active) | confirmed-early | range-1m, realized-volatility-30m, spot-flow-raw-per-aggregate-1s | 0.100621 | 0.090122 | 0.95/0.98 |
| 1s | P(\|R\| >= Q90 \| active) | confirmed-early | range-1m, realized-volatility-30m, spot-flow-raw-per-aggregate-1s | 0.041545 | 0.037497 | 0.95/0.98 |
| 1s | P(positive \| active, \|R\| >= Q50) | primary-only | futures-basis-deviation-15m, spot-flow-trade-imbalance-ema-8, spot-book-quantity-imbalance-l10 | 0.178537 | -0.021789 | 0.45/0.78 |
| 1s | P(positive \| active, \|R\| >= Q75) | confirmed-early | spot-flow-trade-imbalance-ema-2, futures-basis-deviation-5m, close-location-1s | 0.266914 | 0.255133 | 0.95/0.97 |
| 1s | P(positive \| active, \|R\| >= Q90) | confirmed-early | futures-basis-deviation-5m, spot-flow-trade-imbalance-ema-2, spot-flow-trade-count-imbalance-lag-2s | 0.265737 | 0.191855 | 0.95/0.95 |
| 1s | P(positive \| active, \|R\| < Q25) | confirmed-early | spot-flow-last-side-1s, previous-return-1s, spot-flow-last-side-lag-2s | 0.670456 | 0.725519 | 0.95/0.97 |
| 1s | P(positive \| active, \|R\| < Q50) | confirmed-early | spot-flow-last-side-1s, previous-return-1s, spot-flow-last-side-lag-2s | 0.622870 | 0.703901 | 0.95/0.97 |
| 1s | P(positive \| active, \|R\| < Q75) | confirmed-early | spot-flow-last-side-1s, ema-acceleration-2s-1s, spot-flow-last-side-lag-2s | 0.364863 | 0.381930 | 0.95/0.97 |
| 1s | P(\|R\| >= Q50 \| negative) | primary-only | spot-flow-last-side-1s, rsi-2s, spot-book-age-ms | 0.371014 | -0.111002 | 0.45/0.80 |
| 1s | P(\|R\| >= Q75 \| negative) | confirmed-early | spot-flow-last-side-1s, realized-volatility-30m, rsi-2s | 0.211060 | 0.149155 | 0.95/0.98 |
| 1s | P(\|R\| >= Q90 \| negative) | confirmed-early | spot-flow-last-side-1s, realized-volatility-30m, realized-volatility-60s | 0.064074 | 0.053842 | 0.95/0.98 |
| 1s | P(\|R\| >= Q50 \| positive) | primary-only | spot-flow-trade-imbalance-ema-2, spot-book-observed, eth-realized-volatility-60m | 0.291765 | -0.168894 | 0.45/0.79 |
| 1s | P(\|R\| >= Q75 \| positive) | confirmed-early | spot-flow-quantity-squared-skew-1s, futures-range-1m, spot-flow-last-side-1s | 0.161586 | 0.154974 | 0.95/0.95 |
| 1s | P(\|R\| >= Q90 \| positive) | confirmed-early | realized-volatility-60s, spot-flow-quantity-squared-skew-1s, spot-flow-last-side-1s | 0.061852 | 0.057899 | 0.95/0.97 |
| 1m | P(inactive) | confirmed-early | realized-volatility-60m, futures-log-trade-count-1m, ema-slope-8s-8s | 0.104052 | 0.112905 | 0.95/0.98 |
| 1m | P(positive \| active) | confirmed-early | futures-basis-deviation-15m, ema-slope-8s-8s, spot-flow-last-side-1s | 0.016258 | 0.019160 | 0.95/0.97 |
| 1m | P(zero/sign/magnitude quartile) | confirmed-early | realized-volatility-60m, futures-log-trade-count-1m, ema-slope-8s-8s | 0.248659 | 0.273035 | 0.95/0.98 |
| 1m | P(\|R\| >= Q25 \| active) | confirmed-early | realized-volatility-60m, futures-range-1m, ema-slope-8s-8s | 0.107511 | 0.119324 | 0.95/0.98 |
| 1m | P(\|R\| >= Q50 \| active) | confirmed-early | eth-realized-volatility-30m, futures-range-1m, completed-1h-log-volume | 0.113508 | 0.114354 | 0.95/0.97 |
| 1m | P(\|R\| >= Q75 \| active) | confirmed-early | realized-volatility-60m, eth-realized-volatility-30m, futures-log-trade-count-1m | 0.090420 | 0.099950 | 0.95/0.97 |
| 1m | P(\|R\| >= Q90 \| active) | confirmed-early | futures-log-trade-count-1m, eth-realized-volatility-30m, realized-volatility-15m | 0.060430 | 0.068020 | 0.95/0.97 |
| 1m | P(positive \| active, \|R\| >= Q50) | confirmed-early | futures-basis-deviation-5m, spot-flow-trade-imbalance-ema-2, close-location-1s | 0.021823 | 0.030909 | 0.95/0.97 |
| 1m | P(positive \| active, \|R\| >= Q75) | confirmed-early | futures-basis-deviation-15m, spot-flow-trade-imbalance-ema-2, futures-close-location-1m | 0.019735 | 0.048729 | 0.95/0.95 |
| 1m | P(positive \| active, \|R\| >= Q90) | confirmed-early | spot-flow-trade-imbalance-ema-2, previous-return-1s, top-account-minus-global-log | 0.023591 | 0.038184 | 0.95/0.97 |
| 1m | P(positive \| active, \|R\| < Q25) | confirmed-early | spot-flow-last-side-1s, rsi-2s, futures-close-location-1m | 0.047643 | 0.057260 | 0.95/0.97 |
| 1m | P(positive \| active, \|R\| < Q50) | confirmed-early | futures-basis-deviation-15m, spot-flow-last-side-1s, top-account-ratio-log-deviation-24h | 0.018559 | 0.014397 | 0.95/0.95 |
| 1m | P(positive \| active, \|R\| < Q75) | confirmed-early | futures-basis-deviation-15m, spot-flow-last-side-1s, top-account-ratio-log-deviation-24h | 0.016274 | 0.013518 | 0.95/0.95 |
| 1m | P(\|R\| >= Q50 \| negative) | confirmed-early | realized-volatility-60m, eth-realized-volatility-30m, futures-range-1m | 0.116733 | 0.112190 | 0.95/0.97 |
| 1m | P(\|R\| >= Q75 \| negative) | confirmed-early | realized-volatility-60m, eth-realized-volatility-60m, realized-volatility-60s | 0.084216 | 0.096192 | 0.96/0.99 |
| 1m | P(\|R\| >= Q90 \| negative) | confirmed-early | futures-log-trade-count-1m, realized-volatility-30m, active-count-60s | 0.058251 | 0.068463 | 0.95/0.98 |
| 1m | P(\|R\| >= Q50 \| positive) | confirmed-early | realized-volatility-30m, eth-realized-volatility-60m, futures-range-1m | 0.115235 | 0.115557 | 0.95/0.97 |
| 1m | P(\|R\| >= Q75 \| positive) | confirmed-early | futures-log-trade-count-1m, realized-volatility-15m, eth-realized-volatility-30m | 0.093678 | 0.109281 | 0.95/0.97 |
| 1m | P(\|R\| >= Q90 \| positive) | confirmed-early | futures-log-trade-count-1m, eth-realized-volatility-30m, realized-volatility-15m | 0.063844 | 0.063991 | 0.95/0.97 |
| 15m | P(inactive) | primary-only | global-ratio-log-level, sol-return-1m | 0.000485 | 0.004111 | 0.95/0.95 |
| 15m | P(positive \| active) | primary-only | bnb-return-1m, spot-flow-trade-imbalance-ema-8, realized-volatility-60s | 0.008657 | -0.000547 | 0.95/0.97 |
| 15m | P(zero/sign/magnitude quartile) | confirmed-early | realized-volatility-60m, eth-realized-volatility-60m, range-1m | 0.138274 | 0.144491 | 0.96/0.99 |
| 15m | P(\|R\| >= Q25 \| active) | primary-only | eth-realized-volatility-30m, realized-volatility-60s, completed-1h-log-volume | 0.096235 | 0.067762 | 0.96/0.99 |
| 15m | P(\|R\| >= Q50 \| active) | primary-only | eth-realized-volatility-30m, futures-log-trade-count-1m, completed-1h-log-volume | 0.134595 | 0.110857 | 0.95/0.97 |
| 15m | P(\|R\| >= Q75 \| active) | confirmed-early | realized-volatility-60m, futures-log-trade-count-1m, completed-1h-log-volume | 0.072578 | 0.085492 | 0.95/0.98 |
| 15m | P(\|R\| >= Q90 \| active) | confirmed-early | realized-volatility-60m, futures-log-trade-count-1m, open-interest-value-log-change-60m | 0.055128 | 0.042010 | 0.95/0.97 |
| 15m | P(positive \| active, \|R\| >= Q50) | confirmed-early | doge-return-5m, range-1s, spot-flow-vwap-gap-1s | 0.022062 | 0.012584 | 0.95/0.97 |
| 15m | P(positive \| active, \|R\| >= Q75) | confirmed-early | spot-flow-quote-imbalance-ema-128, doge-return-5m, spot-flow-vwap-gap-1s | 0.031972 | 0.016011 | 0.95/0.95 |
| 15m | P(positive \| active, \|R\| >= Q90) | primary-only | top-account-minus-global-log, spot-flow-quote-imbalance-ema-128, spot-book-delta-log-quantity-l10 | 0.069256 | -0.028218 | 0.45/0.78 |
| 15m | P(positive \| active, \|R\| < Q25) | primary-only | open-interest-log-change-15m, bnb-realized-volatility-60m, spot-flow-quote-imbalance-lag-3s | 0.018149 | -0.018144 | 0.95/0.95 |
| 15m | P(positive \| active, \|R\| < Q50) | primary-only | top-position-minus-account-log, spot-flow-quote-imbalance-lag-3s, realized-volatility-15m | 0.017375 | 0.000570 | 0.95/0.97 |
| 15m | P(positive \| active, \|R\| < Q75) | primary-only | open-interest-log-change-15m, global-ratio-log-deviation-24h, realized-volatility-15m | 0.009046 | -0.011400 | 0.95/0.97 |
| 15m | P(\|R\| >= Q50 \| negative) | confirmed-early | eth-realized-volatility-60m, open-interest-value-log-change-15m, top-account-minus-global-log | 0.129134 | 0.236618 | 0.95/0.95 |
| 15m | P(\|R\| >= Q75 \| negative) | confirmed-early | eth-realized-volatility-60m, active-count-60s, top-account-minus-global-log | 0.080309 | 0.141219 | 0.95/0.97 |
| 15m | P(\|R\| >= Q90 \| negative) | confirmed-early | realized-volatility-60m, active-count-60s, eth-return-5m | 0.061315 | 0.084151 | 0.96/0.99 |
| 15m | P(\|R\| >= Q50 \| positive) | confirmed-early | completed-1h-log-volume, active-count-60s, futures-close-location-1m | 0.077601 | 0.121452 | 0.95/0.98 |
| 15m | P(\|R\| >= Q75 \| positive) | primary-only | doge-realized-volatility-60m, completed-1h-log-volume, futures-taker-quote-imbalance-1m | 0.074688 | 0.071293 | 0.95/0.97 |
| 15m | P(\|R\| >= Q90 \| positive) | confirmed-early | doge-realized-volatility-60m, realized-volatility-30m, eth-return-1m | 0.055764 | 0.044565 | 0.96/0.97 |
| 1h | P(inactive) | confirmed-early | futures-quote-volume-surprise-1m | 0.000137 | 0.000120 | 0.95/0.95 |
| 1h | P(positive \| active) | primary-only | top-position-ratio-log-deviation-24h, realized-volatility-60s | 0.027183 | -0.024515 | 0.95/0.97 |
| 1h | P(zero/sign/magnitude quartile) | primary-only | completed-1h-log-volume, range-1m | 0.136755 | 0.066274 | 1.00/1.00 |
| 1h | P(\|R\| >= Q25 \| active) | confirmed-early | realized-volatility-240m, range-1m | 0.104772 | 0.081867 | 1.00/1.00 |
| 1h | P(\|R\| >= Q50 \| active) | primary-only | eth-realized-volatility-60m, completed-1h-log-volume | 0.127822 | 0.058653 | 0.96/0.98 |
| 1h | P(\|R\| >= Q75 \| active) | confirmed-early | range-1m, open-interest-value-log-change-5m | 0.080358 | 0.036077 | 0.95/0.97 |
| 1h | P(\|R\| >= Q90 \| active) | confirmed-early | realized-volatility-30m, range-1s | 0.060862 | 0.062266 | 1.00/1.00 |
| 1h | P(positive \| active, \|R\| >= Q50) | confirmed-early | spot-flow-quote-imbalance-ema-128, spot-book-delta-log-quantity-l10, realized-volatility-240m | 0.037305 | 0.038741 | 0.45/0.80 |
| 1h | P(positive \| active, \|R\| >= Q75) | primary-only | top-account-minus-global-log, realized-volatility-240m | 0.055035 | -0.051210 | 0.95/0.97 |
| 1h | P(positive \| active, \|R\| >= Q90) | insufficient-evaluation | none | 0.000000 | 0.000000 | n/a |
| 1h | P(positive \| active, \|R\| < Q25) | primary-only | global-ratio-log-level, open-interest-log-change-15m | 0.029455 | -0.017783 | 0.95/0.95 |
| 1h | P(positive \| active, \|R\| < Q50) | primary-only | global-ratio-log-level, futures-basis-deviation-15m | 0.017434 | -0.003318 | 0.95/0.95 |
| 1h | P(positive \| active, \|R\| < Q75) | primary-only | futures-range-1m, futures-taker-imbalance-ema-60m | 0.017377 | -0.004127 | 0.95/0.95 |
| 1h | P(\|R\| >= Q50 \| negative) | confirmed-early | doge-realized-volatility-30m, top-account-minus-global-log | 0.169504 | 0.183025 | 0.95/0.95 |
| 1h | P(\|R\| >= Q75 \| negative) | confirmed-early | top-account-minus-global-log, active-count-60s | 0.079717 | 0.116974 | 0.95/0.97 |
| 1h | P(\|R\| >= Q90 \| negative) | confirmed-early | top-account-minus-global-log, doge-realized-volatility-30m | 0.053395 | 0.049622 | 0.95/0.95 |
| 1h | P(\|R\| >= Q50 \| positive) | primary-only | doge-realized-volatility-30m, futures-taker-imbalance-ema-15m | 0.106960 | 0.074719 | 0.95/0.95 |
| 1h | P(\|R\| >= Q75 \| positive) | primary-only | bnb-realized-volatility-30m, spot-flow-flip-rate-1s | 0.040930 | 0.051009 | 0.95/0.95 |
| 1h | P(\|R\| >= Q90 \| positive) | primary-only | doge-realized-volatility-30m, futures-taker-imbalance-ema-60m | 0.046471 | 0.016638 | 0.95/0.95 |

## Interpretation

A selected live-snapshot feature is retained only when no archive-backed subset is within the predictive tolerance. `primary-only` is discovery evidence, not a production input. `confirmed-early` still covers only this recent regime.

Machine-readable results: `data\benchmarks\tiered-component-feature-bases.json`.
