# Extended price, volume, and multiscale information basis

Generated 2026-08-16T17:08:04.542Z for BTCUSDT next-one-second return distributions.

## Result

Adding **last completed 1h log(1 + volume)** to the prior price core-three raises held-out information from 0.14229402 to 0.24111009 bits/target beyond the latest-return state.

After conditioning on that volume regime, the best remaining fifth coordinate is **1s high-low range**, adding 0.055313147 bits/target for a cumulative 0.29642324 bits/target. It is positive in 4/4 annual holdouts.

| order | coordinate | marginal bits | cumulative bits |
|---:|---|---:|---:|
| 1–3 | price core: RSI(2s), EMA acceleration(2,1), EMA slope(8,8) | 0.14229402 | 0.14229402 |
| 4 | last completed 1h log(1 + volume) | 0.098816076 | 0.24111009 |
| 5 | 1s high-low range | 0.055313147 | 0.29642324 |

## Fifth-coordinate alternatives

| rank | candidate | scale | marginal bits | cumulative bits | positive years |
|---:|---|---|---:|---:|---:|
| 1 | 1s high-low range | 1s | 0.055313147 | 0.29642324 | 4/4 |
| 2 | 1s close location in range | 1s | 0.046766347 | 0.28787644 | 4/4 |
| 3 | last completed 5m high-low range | 5m | 0.024950850 | 0.26606094 | 4/4 |
| 4 | last completed 1m high-low range | 1m | 0.019745602 | 0.26085569 | 4/4 |
| 5 | last completed 5s close location in range | 5s | 0.016908646 | 0.25801874 | 4/4 |
| 6 | last completed 5s high-low range | 5s | 0.0093670667 | 0.25047716 | 4/4 |
| 7 | last completed 15m high-low range | 15m | 0.025120506 | 0.26623060 | 3/4 |
| 8 | last completed 5m log(1 + volume) | 5m | 0.019290866 | 0.26040096 | 3/4 |
| 9 | last completed 1m log(1 + volume) | 1m | 0.018282086 | 0.25939218 | 3/4 |
| 10 | last completed 4h log(1 + volume) | 4h | 0.018118174 | 0.25915710 | 3/4 |
| 11 | last completed 15m log(1 + volume) | 15m | 0.014438009 | 0.25554810 | 3/4 |
| 12 | last completed 15s log(1 + volume) | 15s | 0.013407719 | 0.25451781 | 3/4 |
| 13 | last completed 5s log(1 + volume) | 5s | 0.0089993338 | 0.25010943 | 3/4 |
| 14 | last completed 5s absolute return | 5s | 0.0069697523 | 0.24807984 | 3/4 |
| 15 | last completed 15s high-low range | 15s | 0.0067312366 | 0.24784133 | 3/4 |
| 16 | last completed 1m absolute return | 1m | 0.0050294244 | 0.24613952 | 3/4 |
| 17 | Price−EMA(8s) (price-basis benchmark) | 1s | 0.0045535264 | 0.24566362 | 3/4 |
| 18 | last completed 15s absolute return | 15s | 0.0029409038 | 0.24405100 | 3/4 |
| 19 | EMA acceleration(n=2s,k=2s) (price-basis benchmark) | 1s | 0.0012044384 | 0.24231453 | 3/4 |
| 20 | last completed 15s close location in range | 15s | 0.0016463120 | 0.24275640 | 2/4 |
| 21 | 1s log-volume surprise vs EMA(128s) | 1s | -0.000048795589 | 0.24106130 | 2/4 |
| 22 | 1s log-volume surprise vs EMA(32s) | 1s | -0.00013735426 | 0.24097274 | 2/4 |
| 23 | 1s log-volume surprise vs EMA(512s) | 1s | -0.0014299819 | 0.23968011 | 2/4 |
| 24 | 1s log-volume surprise vs EMA(8s) | 1s | -0.0024811251 | 0.23862897 | 2/4 |
| 25 | 1s log-volume surprise vs EMA(2048s) | 1s | -0.0039550059 | 0.23715509 | 2/4 |

The fifth-step counts condition on the exact quartile cross of the previous-return state, all three price coordinates, the selected hourly-volume coordinate, and each candidate. All probabilities for a test year are estimated only from earlier years.

## Limits

- The fifth-step scan carries the strongest 24 stable candidates from the prior exhaustive external-feature screen, plus the two prior price benchmarks.
- This is greedy forward selection, not a proof of the globally optimal nonlinear feature subset.
- Inputs are quartile-quantized for exact interaction counts; a learned continuous model can retain more detail.
- The spot archive supplies base volume but not taker imbalance, trade count, order book, spread, or derivatives flow.

Complete values are stored in `data/benchmarks/extended-market-information-basis.json`.
