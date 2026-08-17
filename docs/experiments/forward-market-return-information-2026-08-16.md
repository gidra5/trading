# Forward-market information about the next 1s spot return

Generated 2026-08-16T20:59:08.491Z. The screen tests 87 causal features from spot aggressor flow, futures basis/flow, and futures positioning after the existing five-coordinate price/volume/range basis.

## Result by source

| source | features | stable full improvements | best primary full bits | transfer full bits | best active-sign feature | primary sign bits | transfer sign bits |
|---|---:|---:|---:|---:|---|---:|---:|
| spot trade flow | 38 | 15 | 0.12525481 | 0.10207176 | spot last aggressor side, last 1s | 0.10756243 | 0.083953081 |
| futures basis/flow | 20 | 2 | 0.031064383 | 0.030789436 | futures log trade count, last completed 1m | 0.024140759 | 0.020510367 |
| futures positioning | 29 | 0 | -0.0092945587 | -0.0038442896 | open interest value log change, 15m | 0.0012173474 | 0.0023870819 |

Across all sources, **spot last aggressor side, last 1s** is the strongest feature positive in both primary halves and transfer: 0.12525481 primary and 0.10207176 transfer bits/target.

## Spot-flow age decay

The last-side and imbalance coordinates were repeated with older completed seconds while leaving the model and targets unchanged.

| flow age | last-side primary bits | last-side transfer bits | trade-count-imbalance primary bits | quote-imbalance primary bits |
|---:|---:|---:|---:|---:|
| 1s | 0.12525481 | 0.10207176 | 0.039090400 | 0.023082268 |
| 2s | 0.087281126 | 0.076346837 | 0.014993694 | 0.0098760742 |
| 3s | -0.0046537989 | -0.00057302263 | -0.011448902 | -0.012395578 |
| 5s | -0.0063587526 | -0.0020483097 | -0.013984016 | -0.013379089 |

The signal remains large at 2s but is gone by 3s. It is therefore a short-lived microstructure feature, not a persistent directional forecast.

## Ninety-day primary ranking

| rank | feature | source | family | full bits | zero bits | active-sign bits | half 1 | half 2 | transfer | stable blocks |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | spot last aggressor side, last 1s | spot trade flow | trade sequence | 0.12525481 | 0.0010662343 | 0.10756243 | 0.12060228 | 0.12975723 | 0.10207176 | 3/3 |
| 2 | spot last aggressor side, 2s old | spot trade flow | lagged trade sequence | 0.087281126 | 0.00047839218 | 0.095698829 | 0.090472668 | 0.084192547 | 0.076346837 | 3/3 |
| 3 | spot aggregate-trade count imbalance, last 1s | spot trade flow | aggressor direction | 0.039090400 | 0.0012797386 | 0.039267978 | 0.040584673 | 0.037644333 | 0.047474991 | 3/3 |
| 4 | spot raw-trade count imbalance, last 1s | spot trade flow | aggressor direction | 0.037960626 | 0.0010163329 | 0.037743128 | 0.039006576 | 0.036948420 | 0.045447608 | 3/3 |
| 5 | futures log trade count, last completed 1m | futures basis/flow | futures activity | 0.031064383 | 0.0050705264 | 0.024140759 | 0.021456299 | 0.040362499 | 0.030789436 | 3/3 |
| 6 | futures range, last completed 1m | futures basis/flow | futures candle | 0.023361213 | 0.0043411516 | 0.018106066 | 0.017608184 | 0.028928642 | 0.027566250 | 3/3 |
| 7 | spot taker quote imbalance, last 1s | spot trade flow | aggressor direction | 0.023082268 | 0.00024923712 | 0.028749132 | 0.024451328 | 0.021757375 | 0.036052984 | 3/3 |
| 8 | spot taker base imbalance, last 1s | spot trade flow | aggressor direction | 0.023074307 | 0.00024874956 | 0.028749335 | 0.024436059 | 0.021756486 | 0.036050402 | 3/3 |
| 9 | spot aggregate-size-squared skew, last 1s | spot trade flow | trade size | 0.021093332 | 0.00028161839 | 0.027628726 | 0.022723061 | 0.019516179 | 0.034489230 | 3/3 |
| 10 | spot maximum aggregate-size skew, last 1s | spot trade flow | trade size | 0.020122241 | 0.00012241667 | 0.027356653 | 0.021977085 | 0.018327237 | 0.034040584 | 3/3 |
| 11 | spot raw-trade count imbalance, 2s old | spot trade flow | lagged aggressor direction | 0.014993694 | -0.00080992343 | 0.025311398 | 0.020604620 | 0.0095637823 | 0.025580639 | 3/3 |
| 12 | spot buyer-minus-seller arrival centroid, last 1s | spot trade flow | timing | 0.010399995 | 0.00055486689 | 0.017696247 | 0.013042016 | 0.0078432082 | 0.014739747 | 3/3 |
| 13 | spot raw-trade imbalance EMA(2s) | spot trade flow | aggressor direction | 0.010399668 | 0.00039389929 | 0.010897100 | 0.013181459 | 0.0077076205 | 0.016224590 | 3/3 |
| 14 | spot taker quote imbalance, 2s old | spot trade flow | lagged aggressor direction | 0.0098760742 | -0.00090350105 | 0.020841123 | 0.014057711 | 0.0058293419 | 0.021728111 | 3/3 |
| 15 | spot log raw-trade count, last 1s | spot trade flow | activity | 0.0085935808 | 0.0014716074 | 0.013131613 | 0.0077979174 | 0.0093635752 | 0.0050034533 | 3/3 |
| 16 | spot trade-count surprise vs EMA(32s) | spot trade flow | activity | 0.0062942336 | 0.00010845155 | 0.0095704606 | 0.0062978484 | 0.0062907355 | 0.0083495115 | 3/3 |
| 17 | futures log quote volume, last completed 1m | futures basis/flow | futures activity | 0.0061617089 | 0.0022406512 | 0.0093948109 | 0.014053592 | -0.0014755734 | 0.0094150228 | 2/3 |
| 18 | spot raw-trade imbalance EMA(8s) | spot trade flow | aggressor direction | 0.0044179216 | 0.00030532985 | 0.017108147 | 0.011177323 | -0.0021234139 | 0.010874607 | 2/3 |
| 19 | spot raw trades per aggregate, last 1s | spot trade flow | trade structure | 0.0041850013 | 0.00048706494 | 0.0088456101 | 0.0038322304 | 0.0045263913 | 0.0041973457 | 3/3 |
| 20 | spot first aggressor side, last 1s | spot trade flow | trade sequence | 0.00016400820 | 0.00036400485 | 0.0040048671 | 0.0017431827 | -0.0013642203 | 0.0064868570 | 2/3 |
| 21 | spot two-sided activity, last 1s | spot trade flow | activity | -0.000024946408 | 0.0011661980 | 0.00068643877 | 0.0014338148 | -0.0014366463 | 0.00023872876 | 2/3 |
| 22 | spot buyer-minus-seller VWAP gap, last 1s | spot trade flow | price pressure | -0.0020844524 | -0.00039348660 | 0.00077772952 | -0.00072466720 | -0.0034003693 | -0.00061338724 | 0/3 |
| 23 | spot taker quote imbalance EMA(2s) | spot trade flow | aggressor direction | -0.0021479201 | -0.00062474969 | 0.0022872735 | 0.00049829194 | -0.0047087623 | 0.0048274806 | 2/3 |
| 24 | spot aggressor-side flip rate, last 1s | spot trade flow | trade sequence | -0.0024258477 | 0.0010627296 | 0.00081882239 | -0.00016627092 | -0.0046125279 | -0.00054147965 | 0/3 |
| 25 | spot log quote volume, last 1s | spot trade flow | activity | -0.0031770910 | 0.00018115529 | 0.0038034009 | -0.00068637383 | -0.0055874548 | -0.0047720150 | 0/3 |
| 26 | spot last aggressor side, 3s old | spot trade flow | lagged trade sequence | -0.0046537989 | -0.00035131959 | 0.0013383680 | -0.0022766171 | -0.0069542902 | -0.00057302263 | 0/3 |
| 27 | futures return, last completed 1m | futures basis/flow | futures price | -0.0053679862 | -0.000026760555 | 0.0040017750 | -0.0025893929 | -0.0080569388 | 0.0028678587 | 1/3 |
| 28 | spot last aggressor side, 5s old | spot trade flow | lagged trade sequence | -0.0063587526 | -0.000094866321 | 0.00046642537 | -0.0038271143 | -0.0088087174 | -0.0020483097 | 0/3 |
| 29 | spot raw-trade imbalance EMA(32s) | spot trade flow | aggressor direction | -0.0066411577 | -0.00036446648 | 0.0027838133 | -0.0016895403 | -0.011433030 | 0.0017837412 | 1/3 |
| 30 | spot taker quote imbalance EMA(8s) | spot trade flow | aggressor direction | -0.0074628188 | -0.00058036740 | 0.0023635191 | -0.0035446542 | -0.011254579 | -0.0016322511 | 0/3 |
| 31 | spot last-trade position within prior second | spot trade flow | timing | -0.0081027295 | -0.000034088059 | -0.000049611207 | -0.0038647602 | -0.012203977 | -0.0027835611 | 0/3 |
| 32 | spot quote-volume surprise vs EMA(32s) | spot trade flow | activity | -0.0089361846 | -0.00083108272 | -0.0010772357 | -0.0056182881 | -0.012147042 | -0.0016907429 | 0/3 |
| 33 | OI-value/OI implied-price basis to spot | futures positioning | open interest | -0.0092945587 | -0.00088596787 | -0.00071423907 | -0.0071395239 | -0.011380070 | -0.0038442896 | 0/3 |
| 34 | open interest value log change, 60m | futures positioning | open interest | -0.0093257850 | -0.00078150979 | 0.0010647364 | -0.0054160965 | -0.013109342 | -0.00053476864 | 0/3 |
| 35 | futures trade-count surprise vs EMA(60m) | futures basis/flow | futures activity | -0.0095540876 | -0.00078558784 | 0.00015494262 | -0.0054759271 | -0.013500682 | -0.0013094233 | 0/3 |
| 36 | open interest value log change, 15m | futures positioning | open interest | -0.0096593110 | -0.00067971820 | 0.0012173474 | -0.0065724228 | -0.012646613 | -0.00098473956 | 0/3 |
| 37 | open interest value log change, 5m | futures positioning | open interest | -0.010077007 | -0.00073215268 | 0.0010818544 | -0.0067175497 | -0.013328084 | -0.0011771880 | 0/3 |
| 38 | spot taker quote imbalance EMA(32s) | spot trade flow | aggressor direction | -0.010099441 | -0.00071744248 | 0.00051166032 | -0.0065815055 | -0.013503884 | -0.0045569130 | 0/3 |
| 39 | spot aggregate-size HHI, last 1s | spot trade flow | trade size | -0.010704568 | -0.0010980020 | -0.00030648702 | -0.0075154342 | -0.013790816 | -0.0050768057 | 0/3 |
| 40 | top-trader account long/short log change, 5m | futures positioning | positioning ratio | -0.011061505 | -0.0010573660 | -0.00029302679 | -0.0082713801 | -0.013761617 | -0.0029810438 | 0/3 |

## Training-history sensitivity

The table follows the overall strongest stable feature, **spot last aggressor side, last 1s**, across histories before the same frozen primary test.

| training history | full bits | active-sign bits |
|---:|---:|---:|
| 30 days | 0.11902352 | 0.10768162 |
| 60 days | 0.12336165 | 0.10732674 |
| 90 days | 0.12525481 | 0.10756243 |
| 180 days | 0.12636348 | 0.10972118 |

## Relation to the existing 15m model test

The earlier matched 15m neural test added all 231 forward-market inputs together and was 0.233% worse than its candle-only control.
This individual information screen can reveal a useful coordinate even when adding every feature to a finite neural model is counterproductive; it still does not establish that combining the winners will improve the 15m model.

## Causal alignment

- Spot aggressor flow uses only the immediately preceding completed 1s bin and its causal EMAs.
- Futures kline features update only after the corresponding 1m candle closes.
- Futures positioning metrics retain a full 5m lag before use.

## Limits

- The source corpus contains a long 2025 block and a separated 2026 transfer block, but not continuous five-year coverage.
- Individual quartile screens identify marginal information; correlated winners from the same source are not additive.
- Five-minute derivatives metrics are repeated between releases, so chronological block stability matters more than a naive independent-sample significance estimate.
- The 1s target test does not replace a dedicated execution backtest and excludes fees, latency, fills, spread, and impact.

Complete values are stored in `data/benchmarks/forward-market-return-information.json`.
