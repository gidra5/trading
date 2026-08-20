# Dense lagged EMA and RSI audit — 2026-08-18

## Key findings

- All four indicator families carry standalone distributional information at the larger targets. The strongest EMA time scale increases from roughly 512 minutes for 1m/15m returns to 2,048 minutes for 1h returns.
- Delayed indicators remain standalone-informative because the volatility regime persists. Even the best one-day-lagged candidate remains positive in both years at every target.
- None of the 10,413 target-specific conditional candidates is positive in the untouched transfer year or in any complete set of four half-year blocks. The existing multiscale volatility/range basis therefore remains the selected input basis.
- Exact parameter winners should not be over-interpreted when quartile states are identical or nearly identical. In particular, one-minute EMA slope and normalized price-minus-EMA value are monotone-equivalent and receive the same score.

## Experiment

This audit tests the full signed-return distribution at 1m, 15m, and 1h horizons. Indicator parameters and signal delays are selected only on the primary year; the following transfer year is untouched confirmation.

The screen contains 3,471 candidates per target: RSI periods [2, 4, 8, 14, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]; EMA periods [2, 4, 8, 16, 32, 64, 128, 512, 2048, 4096, 8192]; EMA difference horizons [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]; and whole-signal lags [0, 1, 2, 4, 8, 15, 30, 60, 120, 240, 480, 960, 1440] minutes.

`EMA value` means the stationary normalized deviation `10000 * log(price / EMA)`, not the nonstationary absolute price level. Slope is the per-minute EMA log change over the stated difference horizon. Acceleration is the difference between adjacent slopes of that length.

Standalone bits compare an indicator with the unconditional distribution. Conditional bits compare the existing volatility/range basis plus the indicator with the basis alone. Negative conditional bits mean the extra histogram coordinate hurts held-out log loss.

## Results

| target | family | primary-selected parameters | standalone primary / transfer bits | conditional primary / transfer bits | conditional positive blocks |
|---:|---|---|---:|---:|---:|
| 1m | ema-value | period=128m, signal lag=0m | 0.04853425 / 0.06296392 | -0.00483383 / -0.00692019 | 0/4 |
| 1m | ema-slope | period=128m, difference=1m, signal lag=0m | 0.04853425 / 0.06296392 | -0.00483383 / -0.00692019 | 0/4 |
| 1m | ema-acceleration | period=2048m, difference=1m, signal lag=0m | 0.04530305 / 0.05692393 | -0.00538010 / -0.00565177 | 0/4 |
| 1m | rsi | period=32m, signal lag=0m | 0.00095529 / 0.00188963 | -0.00703530 / -0.00935802 | 0/4 |
| 15m | ema-value | period=4m, signal lag=0m | 0.02761170 / 0.03425045 | -0.00841006 / -0.00815019 | 0/4 |
| 15m | ema-slope | period=4m, difference=8m, signal lag=0m | 0.02952466 / 0.03592209 | -0.00710052 / -0.00982867 | 0/4 |
| 15m | ema-acceleration | period=2m, difference=256m, signal lag=1m | 0.02372438 / 0.02589139 | -0.00809367 / -0.01151825 | 0/4 |
| 15m | rsi | period=256m, signal lag=1m | 0.01202483 / 0.00634704 | -0.00905757 / -0.01166768 | 0/4 |
| 1h | ema-value | period=512m, signal lag=960m | 0.00503005 / 0.00547565 | -0.00760565 / -0.01166676 | 0/4 |
| 1h | ema-slope | period=2m, difference=4m, signal lag=0m | 0.02775613 / 0.03147803 | -0.00734064 / -0.01349874 | 0/4 |
| 1h | ema-acceleration | period=512m, difference=512m, signal lag=120m | 0.00900552 / 0.00931973 | -0.00777678 / -0.01297292 | 0/4 |
| 1h | rsi | period=4096m, signal lag=1440m | 0.00169947 / 0.00290142 | -0.00882835 / -0.01490899 | 0/4 |

## Best standalone parameters

These rows answer whether the indicator family predicts anything by itself, independently of whether it adds to the current basis.

| target | family | primary-selected parameters | primary bits | untouched transfer bits | positive blocks |
|---:|---|---|---:|---:|---:|
| 1m | ema-value | period=512m, signal lag=0m | 0.05272402 | 0.05258046 | 4/4 |
| 1m | ema-slope | period=512m, difference=1m, signal lag=0m | 0.05272402 | 0.05258046 | 4/4 |
| 1m | ema-acceleration | period=512m, difference=1m, signal lag=0m | 0.04536849 | 0.05676123 | 4/4 |
| 1m | rsi | period=512m, signal lag=1m | 0.01652134 | 0.00607703 | 4/4 |
| 15m | ema-value | period=512m, signal lag=0m | 0.04388935 | 0.04335507 | 4/4 |
| 15m | ema-slope | period=512m, difference=4m, signal lag=4m | 0.04389463 | 0.04050073 | 4/4 |
| 15m | ema-acceleration | period=8192m, difference=4m, signal lag=0m | 0.03078076 | 0.03503549 | 4/4 |
| 15m | rsi | period=1024m, signal lag=1m | 0.01652428 | 0.00692965 | 4/4 |
| 1h | ema-value | period=2048m, signal lag=2m | 0.03153863 | 0.02536401 | 4/4 |
| 1h | ema-slope | period=2048m, difference=1m, signal lag=2m | 0.03153863 | 0.02536401 | 4/4 |
| 1h | ema-acceleration | period=128m, difference=4m, signal lag=1m | 0.02749158 | 0.02097798 | 4/4 |
| 1h | rsi | period=1024m, signal lag=0m | 0.01257288 | 0.00373211 | 3/4 |

## Lag profile

Each row selects the strongest conditional candidate at that fixed lag using only the primary period. This separates a useful delayed signal from merely finding a favorable lag in the transfer data.

### 1m

#### Standalone

| lag | selected family and parameters | primary bits | untouched transfer bits | positive blocks |
|---:|---|---:|---:|---:|
| 0m | ema-value; period=512m, signal lag=0m | 0.05272402 | 0.05258046 | 4/4 |
| 1m | ema-value; period=512m, signal lag=1m | 0.05190577 | 0.05151519 | 4/4 |
| 2m | ema-value; period=512m, signal lag=2m | 0.05131194 | 0.05161624 | 4/4 |
| 4m | ema-value; period=512m, signal lag=4m | 0.04995177 | 0.04967670 | 4/4 |
| 8m | ema-value; period=512m, signal lag=8m | 0.04875180 | 0.04653312 | 4/4 |
| 15m | ema-slope; period=512m, difference=2m, signal lag=15m | 0.04591138 | 0.04422717 | 4/4 |
| 30m | ema-value; period=512m, signal lag=30m | 0.04033679 | 0.03990106 | 4/4 |
| 60m | ema-value; period=512m, signal lag=60m | 0.03343211 | 0.03360138 | 4/4 |
| 120m | ema-slope; period=512m, difference=4m, signal lag=120m | 0.02603399 | 0.02446571 | 4/4 |
| 240m | ema-slope; period=32m, difference=1024m, signal lag=240m | 0.01827295 | 0.01567730 | 4/4 |
| 480m | ema-slope; period=2048m, difference=1024m, signal lag=480m | 0.01379864 | 0.01448276 | 4/4 |
| 960m | ema-slope; period=512m, difference=512m, signal lag=960m | 0.01550652 | 0.01208299 | 4/4 |
| 1440m | ema-acceleration; period=128m, difference=1m, signal lag=1440m | 0.01055273 | 0.01321590 | 4/4 |

#### Conditional on the selected volatility/range basis

| lag | selected family and parameters | conditional primary bits | untouched transfer bits | positive blocks |
|---:|---|---:|---:|---:|
| 0m | ema-value; period=128m, signal lag=0m | -0.00483383 | -0.00692019 | 0/4 |
| 1m | ema-value; period=128m, signal lag=1m | -0.00534614 | -0.00706528 | 0/4 |
| 2m | ema-slope; period=128m, difference=2m, signal lag=2m | -0.00571626 | -0.00763222 | 0/4 |
| 4m | ema-slope; period=128m, difference=256m, signal lag=4m | -0.00577883 | -0.00683102 | 0/4 |
| 8m | ema-slope; period=128m, difference=256m, signal lag=8m | -0.00587961 | -0.00661253 | 0/4 |
| 15m | ema-slope; period=64m, difference=256m, signal lag=15m | -0.00627020 | -0.00666098 | 0/4 |
| 30m | ema-slope; period=32m, difference=256m, signal lag=30m | -0.00591143 | -0.00695742 | 0/4 |
| 60m | ema-acceleration; period=2m, difference=64m, signal lag=60m | -0.00580635 | -0.00844399 | 0/4 |
| 120m | ema-slope; period=8m, difference=64m, signal lag=120m | -0.00639786 | -0.00813720 | 0/4 |
| 240m | ema-slope; period=2048m, difference=512m, signal lag=240m | -0.00659399 | -0.00784127 | 0/4 |
| 480m | ema-slope; period=512m, difference=512m, signal lag=480m | -0.00657234 | -0.00827470 | 0/4 |
| 960m | ema-slope; period=512m, difference=256m, signal lag=960m | -0.00664020 | -0.00944007 | 0/4 |
| 1440m | ema-slope; period=8m, difference=128m, signal lag=1440m | -0.00686180 | -0.00878128 | 0/4 |

### 15m

#### Standalone

| lag | selected family and parameters | primary bits | untouched transfer bits | positive blocks |
|---:|---|---:|---:|---:|
| 0m | ema-value; period=512m, signal lag=0m | 0.04388935 | 0.04335507 | 4/4 |
| 1m | ema-value; period=512m, signal lag=1m | 0.04382658 | 0.04129234 | 4/4 |
| 2m | ema-value; period=512m, signal lag=2m | 0.04359267 | 0.04072119 | 4/4 |
| 4m | ema-slope; period=512m, difference=4m, signal lag=4m | 0.04389463 | 0.04050073 | 4/4 |
| 8m | ema-slope; period=512m, difference=2m, signal lag=8m | 0.04190667 | 0.03970813 | 4/4 |
| 15m | ema-value; period=512m, signal lag=15m | 0.04059636 | 0.03748973 | 4/4 |
| 30m | ema-slope; period=512m, difference=2m, signal lag=30m | 0.03638787 | 0.03223891 | 4/4 |
| 60m | ema-slope; period=512m, difference=4m, signal lag=60m | 0.02914676 | 0.02559206 | 4/4 |
| 120m | ema-value; period=2048m, signal lag=120m | 0.02430161 | 0.02132228 | 4/4 |
| 240m | ema-slope; period=512m, difference=128m, signal lag=240m | 0.01774316 | 0.01176875 | 4/4 |
| 480m | ema-slope; period=2m, difference=1024m, signal lag=480m | 0.01271766 | 0.01246368 | 4/4 |
| 960m | ema-slope; period=512m, difference=512m, signal lag=960m | 0.01209301 | 0.01349197 | 4/4 |
| 1440m | ema-slope; period=8192m, difference=2m, signal lag=1440m | 0.00751100 | 0.01285889 | 3/4 |

#### Conditional on the selected volatility/range basis

| lag | selected family and parameters | conditional primary bits | untouched transfer bits | positive blocks |
|---:|---|---:|---:|---:|
| 0m | ema-slope; period=4m, difference=8m, signal lag=0m | -0.00710052 | -0.00982867 | 0/4 |
| 1m | ema-slope; period=512m, difference=8m, signal lag=1m | -0.00770070 | -0.00981518 | 0/4 |
| 2m | ema-slope; period=512m, difference=8m, signal lag=2m | -0.00742372 | -0.00926681 | 0/4 |
| 4m | ema-acceleration; period=8m, difference=256m, signal lag=4m | -0.00813905 | -0.01002615 | 0/4 |
| 8m | ema-slope; period=512m, difference=2m, signal lag=8m | -0.00837226 | -0.01051254 | 0/4 |
| 15m | ema-slope; period=32m, difference=128m, signal lag=15m | -0.00839522 | -0.01107226 | 0/4 |
| 30m | ema-slope; period=16m, difference=256m, signal lag=30m | -0.00865811 | -0.01224752 | 0/4 |
| 60m | ema-slope; period=512m, difference=16m, signal lag=60m | -0.00892838 | -0.01062437 | 0/4 |
| 120m | ema-slope; period=512m, difference=512m, signal lag=120m | -0.00839044 | -0.00960362 | 0/4 |
| 240m | ema-slope; period=2048m, difference=256m, signal lag=240m | -0.00877741 | -0.00942797 | 0/4 |
| 480m | ema-acceleration; period=16m, difference=2m, signal lag=480m | -0.00892691 | -0.01221336 | 0/4 |
| 960m | ema-acceleration; period=16m, difference=8m, signal lag=960m | -0.00844242 | -0.01246879 | 0/4 |
| 1440m | ema-acceleration; period=512m, difference=128m, signal lag=1440m | -0.00891051 | -0.01225730 | 0/4 |

### 1h

#### Standalone

| lag | selected family and parameters | primary bits | untouched transfer bits | positive blocks |
|---:|---|---:|---:|---:|
| 0m | ema-slope; period=2048m, difference=4m, signal lag=0m | 0.03070591 | 0.02588684 | 4/4 |
| 1m | ema-slope; period=2048m, difference=2m, signal lag=1m | 0.03090865 | 0.02473202 | 4/4 |
| 2m | ema-value; period=2048m, signal lag=2m | 0.03153863 | 0.02536401 | 4/4 |
| 4m | ema-slope; period=2048m, difference=8m, signal lag=4m | 0.02963805 | 0.02422801 | 4/4 |
| 8m | ema-value; period=2048m, signal lag=8m | 0.03026439 | 0.02463885 | 4/4 |
| 15m | ema-slope; period=2048m, difference=2m, signal lag=15m | 0.02825755 | 0.02587596 | 4/4 |
| 30m | ema-slope; period=2048m, difference=4m, signal lag=30m | 0.02612944 | 0.02194369 | 4/4 |
| 60m | ema-slope; period=512m, difference=8m, signal lag=60m | 0.02351527 | 0.01941672 | 4/4 |
| 120m | ema-slope; period=2048m, difference=2m, signal lag=120m | 0.01860239 | 0.01509723 | 4/4 |
| 240m | ema-slope; period=2048m, difference=256m, signal lag=240m | 0.01430386 | 0.01430006 | 4/4 |
| 480m | ema-acceleration; period=2m, difference=512m, signal lag=480m | 0.01343572 | 0.01066277 | 4/4 |
| 960m | ema-slope; period=2048m, difference=4m, signal lag=960m | 0.01409849 | 0.00804680 | 4/4 |
| 1440m | ema-acceleration; period=512m, difference=4m, signal lag=1440m | 0.00825080 | 0.01066234 | 4/4 |

#### Conditional on the selected volatility/range basis

| lag | selected family and parameters | conditional primary bits | untouched transfer bits | positive blocks |
|---:|---|---:|---:|---:|
| 0m | ema-slope; period=2m, difference=4m, signal lag=0m | -0.00734064 | -0.01349874 | 0/4 |
| 1m | ema-slope; period=2m, difference=1024m, signal lag=1m | -0.00918180 | -0.01608345 | 0/4 |
| 2m | ema-slope; period=16m, difference=2m, signal lag=2m | -0.00909529 | -0.01413985 | 0/4 |
| 4m | ema-slope; period=8m, difference=1024m, signal lag=4m | -0.01015194 | -0.01460450 | 0/4 |
| 8m | ema-slope; period=2m, difference=1024m, signal lag=8m | -0.01048970 | -0.01340683 | 0/4 |
| 15m | ema-acceleration; period=32m, difference=32m, signal lag=15m | -0.00929810 | -0.01756122 | 0/4 |
| 30m | ema-slope; period=2048m, difference=64m, signal lag=30m | -0.01169554 | -0.01693645 | 0/4 |
| 60m | ema-acceleration; period=4m, difference=1m, signal lag=60m | -0.00799080 | -0.00956201 | 0/4 |
| 120m | ema-acceleration; period=512m, difference=512m, signal lag=120m | -0.00777678 | -0.01297292 | 0/4 |
| 240m | ema-acceleration; period=64m, difference=64m, signal lag=240m | -0.00844738 | -0.01988703 | 0/4 |
| 480m | ema-slope; period=16m, difference=4m, signal lag=480m | -0.00856560 | -0.01737550 | 0/4 |
| 960m | ema-value; period=512m, signal lag=960m | -0.00760565 | -0.01166676 | 0/4 |
| 1440m | rsi; period=4096m, signal lag=1440m | -0.00882835 | -0.01490899 | 0/4 |

## Interpretation rules

- A parameter/lag winner is not trusted merely because it maximizes the primary score. It should remain positive in the untouched transfer year and preferably all four half-year blocks.
- Thousands of correlated candidates create selection optimism in the primary maximum. The transfer score is the evidence for or against survival.
- The four-bin histogram model measures distributional information, including volatility and tails; it is not limited to mean-return direction.
- The complete candidate table, including every tested lag, is retained in the machine-readable artifact.

Machine-readable results: `data/benchmarks/dense-lagged-indicator-audit.json`.
