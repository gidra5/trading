# Spot top-10 order-book information about the next 1s return

Generated 2026-08-16T20:34:38.839Z. This study uses 790 324 locally recorded BTCUSDT spot top-10 snapshots and matching official spot 1s candles.

## Result

The best primary full-distribution result is **L1 quantity imbalance** with the **all pre-cutoff spot-book history** fit: 0.014954081 bits/target in July 31–August 5 and 0.022972245 in the later August 10–15 block. Its four chronological sub-block scores are 0.016155519, 0.013872892, 0.026490155, 0.021333555.

There are 20 feature/window combinations positive in all four chronological sub-blocks. The strongest is **L1 quantity imbalance** (all pre-cutoff spot-book history), with 0.014954081 primary and 0.022972245 transfer bits/target.

For active-return sign, the most stable result is **top-10 quantity imbalance** (72 hours): 0.043498845 primary and 0.054948709 transfer bits per active target.

For the winning L1 feature's training quartiles, P(positive | active) runs 37.222% → 46.738% → 53.063% → 62.713% from the most ask-heavy to the most bid-heavy cell. The zero-return probabilities are 44.463% → 50.532% → 50.351% → 45.038%.

## Primary ranking

| rank | feature | history | family | primary full bits | transfer full bits | primary sign bits | transfer sign bits | positive sub-blocks |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| 1 | L1 quantity imbalance | all pre-cutoff spot-book history | queue imbalance | 0.014954081 | 0.022972245 | 0.042584936 | 0.053853854 | 4/4 |
| 2 | L1 notional imbalance | all pre-cutoff spot-book history | queue imbalance | 0.014954081 | 0.022972245 | 0.042584936 | 0.053853854 | 4/4 |
| 3 | top-10 quantity imbalance | all pre-cutoff spot-book history | queue imbalance | 0.014908323 | 0.022457860 | 0.042919301 | 0.054084585 | 4/4 |
| 4 | top-10 notional imbalance | all pre-cutoff spot-book history | queue imbalance | 0.014908323 | 0.022457564 | 0.042919301 | 0.054084585 | 4/4 |
| 5 | top-2 quantity imbalance | all pre-cutoff spot-book history | queue imbalance | 0.014866644 | 0.022760168 | 0.042441076 | 0.053699705 | 4/4 |
| 6 | top-2 notional imbalance | all pre-cutoff spot-book history | queue imbalance | 0.014866644 | 0.022760168 | 0.042441076 | 0.053699705 | 4/4 |
| 7 | top-5 quantity imbalance | all pre-cutoff spot-book history | queue imbalance | 0.014817515 | 0.022597186 | 0.042453855 | 0.053828455 | 4/4 |
| 8 | top-5 notional imbalance | all pre-cutoff spot-book history | queue imbalance | 0.014817515 | 0.022597186 | 0.042453855 | 0.053828455 | 4/4 |
| 9 | microprice offset from midpoint | all pre-cutoff spot-book history | top of book | 0.014774729 | 0.022566391 | 0.042603808 | 0.053403195 | 4/4 |
| 10 | L1 quantity imbalance | 72 hours | queue imbalance | 0.011293592 | 0.019676178 | 0.043184199 | 0.054548604 | 4/4 |
| 11 | L1 notional imbalance | 72 hours | queue imbalance | 0.011293592 | 0.019676178 | 0.043184199 | 0.054548604 | 4/4 |
| 12 | top-2 quantity imbalance | 72 hours | queue imbalance | 0.011224555 | 0.019437001 | 0.043077401 | 0.054415838 | 4/4 |
| 13 | top-2 notional imbalance | 72 hours | queue imbalance | 0.011224555 | 0.019437001 | 0.043077401 | 0.054415838 | 4/4 |
| 14 | top-5 quantity imbalance | 72 hours | queue imbalance | 0.011120167 | 0.019061889 | 0.043115148 | 0.054559911 | 4/4 |
| 15 | top-5 notional imbalance | 72 hours | queue imbalance | 0.011120167 | 0.019061889 | 0.043115148 | 0.054559911 | 4/4 |
| 16 | top-10 quantity imbalance | 72 hours | queue imbalance | 0.011035873 | 0.019017934 | 0.043498845 | 0.054948709 | 4/4 |
| 17 | top-10 notional imbalance | 72 hours | queue imbalance | 0.011035873 | 0.019017630 | 0.043498845 | 0.054948709 | 4/4 |
| 18 | microprice offset from midpoint | 72 hours | top of book | 0.010976127 | 0.019224877 | 0.043022246 | 0.054045794 | 4/4 |
| 19 | snapshot change in spread | 72 hours | book change | 0.0081994925 | 0.012848440 | 0.015704627 | 0.020927401 | 4/4 |
| 20 | snapshot change in spread | all pre-cutoff spot-book history | book change | 0.0061341940 | 0.0097239977 | 0.013430557 | 0.016740999 | 4/4 |
| 21 | normalized L1 order-flow imbalance | 72 hours | order flow | 0.0058477776 | 0.020244919 | 0.023176534 | 0.036705796 | 3/4 |
| 22 | normalized L1 order-flow imbalance | all pre-cutoff spot-book history | order flow | 0.0055639782 | 0.017477646 | 0.019673673 | 0.029825091 | 3/4 |
| 23 | snapshot-to-snapshot midpoint return | 72 hours | book change | 0.0047108662 | 0.0071941806 | 0.013327870 | 0.014427509 | 3/4 |
| 24 | snapshot-to-snapshot midpoint return | all pre-cutoff spot-book history | book change | 0.0038824209 | 0.0050130861 | 0.012033928 | 0.010893418 | 3/4 |
| 25 | snapshot change in spread | 24 hours | book change | 0.0022734491 | 0.0055456925 | 0.015046067 | 0.018161901 | 2/4 |
| 26 | snapshot-to-snapshot midpoint return | 24 hours | book change | -0.00026669982 | 0.00072295759 | 0.011578542 | 0.0099386288 | 2/4 |
| 27 | snapshot change in log top-10 quantity | all pre-cutoff spot-book history | book change | -0.0021763315 | 0.0078632235 | 0.011700552 | 0.019396461 | 2/4 |
| 28 | snapshot change in top-10 imbalance | all pre-cutoff spot-book history | book change | -0.0022331467 | 0.0077885051 | 0.0079588828 | 0.015730799 | 2/4 |
| 29 | snapshot change in L1 imbalance | all pre-cutoff spot-book history | book change | -0.0023300319 | 0.0072823611 | 0.0071957774 | 0.014993465 | 2/4 |
| 30 | snapshot change in top-5 imbalance | all pre-cutoff spot-book history | book change | -0.0023510172 | 0.0075275339 | 0.0074106961 | 0.015583011 | 2/4 |

## Timestamp-lag sensitivity

To allow for exchange-to-recorder delivery delay, the winning feature was rescored after requiring the latest snapshot to predate the target boundary by at least the stated amount. The probability table remains the same frozen fit.

| minimum snapshot age | primary observations | primary bits | transfer observations | transfer bits |
|---:|---:|---:|---:|---:|
| 100ms | 420 314 | 0.015009064 | 198 749 | 0.021928882 |
| 250ms | 419 054 | 0.015057215 | 181 954 | 0.020796125 |
| 500ms | 414 728 | 0.015205898 | 171 875 | 0.019880873 |
| 1000ms | 113 738 | 0.0080407326 | 50 412 | 0.016530099 |

## Coverage and causal split

The frozen cutoff is 2026-07-31T00:00:00.000Z. The primary test has 421 125 fresh-book targets (199 470 / 221 655); transfer has 215 479 (68 476 / 147 003). 809 292 seconds are excluded because the book is missing/stale or the candle basis is not ready.

Only snapshots strictly earlier than a target second are used; snapshots older than 5 seconds are discarded.
Snapshot-change features reset after gaps longer than 5 seconds.
All book quartiles and probability tables are fitted only from pre-cutoff data and then frozen.

## Contrast with futures percentage-depth

The earlier futures result had -0.0015874785 bits/target for its numerically best full-distribution feature; its best active-sign result was 0.0027487982 bits/active target while remaining -0.0034141679 on the full distribution.
The futures study uses 30-second cumulative percentage-depth and a much longer history, so this is a directional rather than controlled source comparison.

## Limits

- The local spot stream spans only about three weeks and contains multi-hour and multi-day gaps; usable observations are limited to fresh-snapshot periods.
- This is top-10 depth, not a complete depth ladder, and it records snapshots rather than every exchange depth update.
- Only one training cutoff and two later calendar blocks are available; positive results need confirmation on a longer independently recorded spot history.
- The comparison with archived futures percentage-depth is not apples-to-apples because source cadence, depth representation, dates, and history length differ.
- No latency beyond causal timestamp ordering, fees, queue position, spread crossing, fills, or market impact are modeled.

Complete values are stored in `data/benchmarks/spot-order-book-return-information.json`.
