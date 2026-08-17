# Joint 30-day feature-subset discovery across recent public and local sources — 2026-08-17

## Meaning of global

Unrestricted mutual information is monotone, so every causal input is a trivial maximizer. This experiment instead searches for the jointly best finite basis under a declared model and search universe. Every subset is scored directly against the unconditional return distribution; no previous feature baseline is fixed.

The broad screen chooses 10 finalists while retaining the best representative of every family. Every subset of up to 6 finalists is then scored. Finalists and subsets are selected only on the primary selection period, with positivity required in both primary half-blocks. The later transfer period is untouched until the final confirmation.

## Results

| return horizon | broad candidates | exhaustive finalists | tested subsets | selected features | primary bits | transfer bits | positive blocks |
|---:|---:|---:|---:|---|---:|---:|---:|
| 1s | 147 | 10 | 847 | spot-flow-last-side-1s, spot-book-spread-bps, rsi-2s, spot-flow-aggregate-count-imbalance-1s, range-1s, spot-flow-vwap-gap-1s | 0.52316028 | 0.32970502 | 4/4 |
| 1m | 147 | 10 | 847 | realized-volatility-60m, futures-log-trade-count-1m | 0.29390816 | 0.31216479 | 4/4 |
| 15m | 147 | 10 | 847 | realized-volatility-240m | 0.12764332 | 0.10656683 | 3/4 |
| 1h | 147 | 10 | 847 | realized-volatility-60m, open-interest-value-log-change-5m, spot-book-delta-spread-bps, spot-flow-raw-per-aggregate-1s | 0.16057234 | -0.22855908 | 2/4 |

## 1s return

Observations: 22,801 train, 10,080 primary, 10,020 transfer.

Primary maximum: `spot-flow-last-side-1s, spot-book-spread-bps, rsi-2s, spot-flow-aggregate-count-imbalance-1s, range-1s, spot-flow-vwap-gap-1s` at 0.52316028 primary / 0.32970502 transfer bits.

Validation-stable selection: `spot-flow-last-side-1s, spot-book-spread-bps, rsi-2s, spot-flow-aggregate-count-imbalance-1s, range-1s, spot-flow-vwap-gap-1s` at 0.52316028 primary / 0.32970502 untouched transfer bits; primary/transfer half-blocks 0.475862, 0.570459, 0.362226, 0.297184.

Smallest subset within 0.001 bits of the primary maximum: `spot-flow-last-side-1s, spot-book-spread-bps, rsi-2s, spot-flow-aggregate-count-imbalance-1s, range-1s, spot-flow-vwap-gap-1s`.

| input | family | parameters | lookback | availability delay | conditional primary bits | conditional transfer bits |
|---|---|---|---|---|---:|---:|
| spot last aggressor side, last 1s (`spot-flow-last-side-1s`) | trade sequence | spot last aggressor side, last 1s | 1s | latest completed second | 0.06830095 | 0.07607499 |
| bid-ask spread (`spot-book-spread-bps`) | spot book top of book | bid-ask spread | latest snapshot | strictly before boundary; maximum age 5s | 0.12324772 | -0.05793303 |
| RSI (`rsi-2s`) | price dynamics | period=2s | recursive | through origin | 0.02567132 | 0.02356159 |
| spot aggregate-trade count imbalance, last 1s (`spot-flow-aggregate-count-imbalance-1s`) | aggressor direction | spot aggregate-trade count imbalance, last 1s | 1s | latest completed second | 0.01359752 | 0.04454864 |
| Completed 1s range (`range-1s`) | candle shape | 10000*log(high/low) | 1s | latest completed second | 0.03049031 | 0.04958526 |
| spot buyer-minus-seller VWAP gap, last 1s (`spot-flow-vwap-gap-1s`) | price pressure | spot buyer-minus-seller VWAP gap, last 1s | 1s | latest completed second | 0.03286505 | 0.00847712 |

## 1m return

Observations: 22,801 train, 10,080 primary, 10,020 transfer.

Primary maximum: `realized-volatility-60m, futures-log-trade-count-1m` at 0.29390816 primary / 0.31216479 transfer bits.

Validation-stable selection: `realized-volatility-60m, futures-log-trade-count-1m` at 0.29390816 primary / 0.31216479 untouched transfer bits; primary/transfer half-blocks 0.086298, 0.501518, 0.134071, 0.490258.

Smallest subset within 0.001 bits of the primary maximum: `realized-volatility-60m, futures-log-trade-count-1m`.

| input | family | parameters | lookback | availability delay | conditional primary bits | conditional transfer bits |
|---|---|---|---|---|---:|---:|
| Realized volatility (`realized-volatility-60m`) | minute volatility | sqrt(sum(r_1m^2)),window=60m | 60m | through origin | 0.06487766 | 0.06300029 |
| futures log trade count, last completed 1m (`futures-log-trade-count-1m`) | futures activity | futures log trade count, last completed 1m | 1m | after 1m candle close | 0.04813377 | 0.05928074 |

## 15m return

Observations: 1,521 train, 672 primary, 668 transfer.

Primary maximum: `realized-volatility-240m` at 0.12764332 primary / 0.10656683 transfer bits.

Validation-stable selection: `realized-volatility-240m` at 0.12764332 primary / 0.10656683 untouched transfer bits; primary/transfer half-blocks 0.032700, 0.222586, -0.014352, 0.227486.

Smallest subset within 0.001 bits of the primary maximum: `realized-volatility-240m`.

| input | family | parameters | lookback | availability delay | conditional primary bits | conditional transfer bits |
|---|---|---|---|---|---:|---:|
| Realized volatility (`realized-volatility-240m`) | minute volatility | sqrt(sum(r_1m^2)),window=240m | 240m | through origin | 0.12764332 | 0.10656683 |

## 1h return

Observations: 381 train, 168 primary, 167 transfer.

Primary maximum: `realized-volatility-60m, open-interest-value-log-change-5m, spot-book-delta-spread-bps, spot-flow-raw-per-aggregate-1s` at 0.16057234 primary / -0.22855908 transfer bits.

Validation-stable selection: `realized-volatility-60m, open-interest-value-log-change-5m, spot-book-delta-spread-bps, spot-flow-raw-per-aggregate-1s` at 0.16057234 primary / -0.22855908 untouched transfer bits; primary/transfer half-blocks 0.011694, 0.309451, -0.371291, -0.087526.

Smallest subset within 0.001 bits of the primary maximum: `realized-volatility-60m, open-interest-value-log-change-5m, spot-book-delta-spread-bps, spot-flow-raw-per-aggregate-1s`.

| input | family | parameters | lookback | availability delay | conditional primary bits | conditional transfer bits |
|---|---|---|---|---|---:|---:|
| Realized volatility (`realized-volatility-60m`) | minute volatility | sqrt(sum(r_1m^2)),window=60m | 60m | through origin | 0.08133432 | -0.04056204 |
| open interest value log change, 5m (`open-interest-value-log-change-5m`) | open interest | open interest value log change, 5m | 5m | full 5m publication lag | 0.01701370 | -0.18141191 |
| snapshot change in spread (`spot-book-delta-spread-bps`) | spot book book change | snapshot change in spread | latest snapshot | strictly before boundary; maximum age 5s | 0.00542900 | -0.06405515 |
| spot raw trades per aggregate, last 1s (`spot-flow-raw-per-aggregate-1s`) | trade structure | spot raw trades per aggregate, last 1s | 1s | latest completed second | 0.05006195 | 0.11086968 |

## Limits

- `Global` means exhaustive only inside the explicitly reported finalist universe and maximum subset size, not over every mathematical transform of history.
- Every candidate in this report is scored only on the exact common timestamp coverage declared by the dataset manifest; results do not imply stability outside that calendar span.
- Quartile feature cells and a categorical return distribution make the search exact and inspectable, but a continuous model may exploit information inside cells.
- Reported conditional contributions remove one coordinate from the final joint subset. They are not standalone feature scores and need not sum exactly because features interact.

Machine-readable results are stored in `data/benchmarks/global-return-feature-basis-30d.json`.
