# Joint chronological feature-subset search for BTC return distributions — 2026-08-17

## Meaning of global

Unrestricted mutual information is monotone, so every causal input is a trivial maximizer. This experiment instead searches for the jointly best finite basis under a declared model and search universe. Every subset is scored directly against the unconditional return distribution; no previous feature baseline is fixed.

The broad screen chooses 10 finalists while retaining the best representative of every family. Every subset of up to 6 finalists is then scored. Finalists and subsets are selected only on the primary selection period, with positivity required in both primary half-blocks. The later transfer period is untouched until the final confirmation.

## Results

| return horizon | broad candidates | exhaustive finalists | tested subsets | selected features | primary bits | transfer bits | positive blocks |
|---:|---:|---:|---:|---|---:|---:|---:|
| 1s | 10 | 10 | 847 | range-1s, active-count-60s, previous-return-1s, realized-volatility-60s, close-location-1s | 0.47931874 | 0.46310058 | 4/4 |
| 1m | 34 | 10 | 847 | realized-volatility-30m, range-1m, realized-volatility-15m, realized-volatility-60m | 0.21034339 | 0.22499481 | 4/4 |
| 15m | 34 | 10 | 847 | realized-volatility-60m, realized-volatility-15m, realized-volatility-240m | 0.15980478 | 0.15989883 | 4/4 |
| 1h | 34 | 10 | 847 | realized-volatility-30m, realized-volatility-240m | 0.12051894 | 0.13453297 | 4/4 |

## 1s return

Observations: 1,578,180 train, 525,600 primary, 525,600 transfer.

Primary maximum: `range-1s, active-count-60s, previous-return-1s, realized-volatility-60s, close-location-1s` at 0.47931874 primary / 0.46310058 transfer bits.

Validation-stable selection: `range-1s, active-count-60s, previous-return-1s, realized-volatility-60s, close-location-1s` at 0.47931874 primary / 0.46310058 untouched transfer bits; primary/transfer half-blocks 0.473325, 0.485312, 0.491080, 0.435121.

Smallest subset within 0.001 bits of the primary maximum: `range-1s, active-count-60s, previous-return-1s, realized-volatility-60s, close-location-1s`.

| input | family | parameters | lookback | availability delay | conditional primary bits | conditional transfer bits |
|---|---|---|---|---|---:|---:|
| Completed 1s high-low log range (`range-1s`) | candle shape | 10000*log(high/low) | 1s | latest completed second | 0.25377486 | 0.23540688 |
| Active-return count (`active-count-60s`) | activity | window=60s | 60s | trailing through origin | 0.07261417 | 0.06076274 |
| Previous 1s signed log return (`previous-return-1s`) | return history | lag=1s | 1s | latest completed second | 0.02041253 | 0.02221065 |
| Realized volatility (`realized-volatility-60s`) | volatility | sqrt(sum(r^2)), window=60s | 60s | trailing through origin | 0.02512570 | 0.01947858 |
| Completed 1s close location (`close-location-1s`) | candle shape | (2*close-high-low)/(high-low) | 1s | latest completed second | 0.07939705 | 0.05704037 |

## 1m return

Observations: 394,500 train, 131,400 primary, 131,385 transfer.

Primary maximum: `realized-volatility-30m, range-1m, realized-volatility-15m, realized-volatility-60m` at 0.21034339 primary / 0.22499481 transfer bits.

Validation-stable selection: `realized-volatility-30m, range-1m, realized-volatility-15m, realized-volatility-60m` at 0.21034339 primary / 0.22499481 untouched transfer bits; primary/transfer half-blocks 0.184474, 0.236212, 0.260460, 0.189530.

Smallest subset within 0.001 bits of the primary maximum: `range-1m, realized-volatility-15m, realized-volatility-60m`.

| input | family | parameters | lookback | availability delay | conditional primary bits | conditional transfer bits |
|---|---|---|---|---|---:|---:|
| Realized volatility (`realized-volatility-30m`) | volatility | sqrt(sum(r_1m^2)), window=30m | 30m | through origin | 0.00038489 | 0.00020101 |
| Completed 1m high-low log range (`range-1m`) | candle shape | 10000*log(high/low) | 1m | latest completed minute | 0.00902207 | 0.01465284 |
| Realized volatility (`realized-volatility-15m`) | volatility | sqrt(sum(r_1m^2)), window=15m | 15m | through origin | 0.00235774 | 0.00244964 |
| Realized volatility (`realized-volatility-60m`) | volatility | sqrt(sum(r_1m^2)), window=60m | 60m | through origin | 0.00321700 | 0.00382756 |

## 15m return

Observations: 98,557 train, 32,828 primary, 32,824 transfer.

Primary maximum: `realized-volatility-60m, realized-volatility-15m, realized-volatility-240m` at 0.15980478 primary / 0.15989883 transfer bits.

Validation-stable selection: `realized-volatility-60m, realized-volatility-15m, realized-volatility-240m` at 0.15980478 primary / 0.15989883 untouched transfer bits; primary/transfer half-blocks 0.145498, 0.174112, 0.182308, 0.137490.

Smallest subset within 0.001 bits of the primary maximum: `realized-volatility-60m, realized-volatility-15m, realized-volatility-240m`.

| input | family | parameters | lookback | availability delay | conditional primary bits | conditional transfer bits |
|---|---|---|---|---|---:|---:|
| Realized volatility (`realized-volatility-60m`) | volatility | sqrt(sum(r_1m^2)), window=60m | 60m | through origin | 0.00285449 | 0.00492039 |
| Realized volatility (`realized-volatility-15m`) | volatility | sqrt(sum(r_1m^2)), window=15m | 15m | through origin | 0.01046833 | 0.00409586 |
| Realized volatility (`realized-volatility-240m`) | volatility | sqrt(sum(r_1m^2)), window=240m | 240m | through origin | 0.00282739 | 0.00050018 |

## 1h return

Observations: 26,282 train, 8,754 primary, 8,753 transfer.

Primary maximum: `realized-volatility-30m, realized-volatility-240m` at 0.12051894 primary / 0.13453297 transfer bits.

Validation-stable selection: `realized-volatility-30m, realized-volatility-240m` at 0.12051894 primary / 0.13453297 untouched transfer bits; primary/transfer half-blocks 0.095148, 0.145890, 0.163605, 0.105467.

Smallest subset within 0.001 bits of the primary maximum: `realized-volatility-30m, realized-volatility-240m`.

| input | family | parameters | lookback | availability delay | conditional primary bits | conditional transfer bits |
|---|---|---|---|---|---:|---:|
| Realized volatility (`realized-volatility-30m`) | volatility | sqrt(sum(r_1m^2)), window=30m | 30m | through origin | 0.02334124 | 0.02465614 |
| Realized volatility (`realized-volatility-240m`) | volatility | sqrt(sum(r_1m^2)), window=240m | 240m | through origin | 0.00713321 | 0.00793192 |

## Limits

- `Global` means exhaustive only inside the explicitly reported finalist universe and maximum subset size, not over every mathematical transform of history.
- Every candidate in this report is scored only on the exact common timestamp coverage declared by the dataset manifest; results do not imply stability outside that calendar span.
- Quartile feature cells and a categorical return distribution make the search exact and inspectable, but a continuous model may exploit information inside cells.
- Reported conditional contributions remove one coordinate from the final joint subset. They are not standalone feature scores and need not sum exactly because features interact.

Machine-readable results are stored in `data/benchmarks/global-return-feature-basis.json`.
