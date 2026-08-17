# One-second log-return time reversibility

Generated 2026-08-15T22:00:08.062Z. This analysis covers 157,766,399 adjacent BTCUSDT one-second log returns from 2021-07-25T00:00:00.000Z through 2026-07-25T00:00:00.000Z. Exact zero is a separate state with probability 35.727134%.

## Result

Clearly resolved but practically modest one-step time asymmetry at the 65-state resolution.

Forward and backward marginals being identical is only stationarity. Reversibility additionally requires every ordered block distribution to equal its reversed distribution.

## Interpretation

- At one second, reversal JS is 0.0018228105 bits, 208.8 times the orientation-null mean. Total variation above its null mean is 3.2722%, while the raw optimal arrow classifier reaches only 51.7258%. The arrow is unambiguous statistically but modest as predictive information.
- The effect decays sharply: at 60 seconds, TV above null is only 0.0583% and arrow classification is 50.1202%. The measurable arrow is predominantly immediate market microstructure.
- Histories accumulate additional direction information. A five-return block reaches 54.5111% raw arrow accuracy; its TV is 9.0222% versus a 2.2203% sampling floor.
- Both-active transitions contribute about 67.11% of one-step reversal JS. Therefore the effect is not merely the zero-gap process, although exact-zero boundaries also carry directional information.
- The largest currents involve very small returns, zero transitions, and immediate sign reversals. That pattern is consistent with tick-size/price-grid mechanics and bid-ask bounce; it should not be interpreted directly as tradable directional return correlation.

## Pairwise detailed balance by lag

The primary representation has 64 optimized transformed-return cells plus the exact-zero state.

| lag | JS forward vs reverse (bits) | null JS mean | JS above null | total variation | TV above null | arrow classifier | entropy production (bits) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1s | 0.0018228105 | 0.0000087311970 | 0.0018140793 | 0.034516387 | 0.032722153 | 51.725819% | 0.0073141574 |
| 2s | 0.00025925373 | 0.0000088343408 | 0.00025041939 | 0.010292997 | 0.0084926571 | 50.514650% | 0.0010379697 |
| 5s | 0.000076959264 | 0.0000088164353 | 0.000068142829 | 0.0050209362 | 0.0032105761 | 50.251047% | 0.00030783692 |
| 10s | 0.000049412527 | 0.0000087974051 | 0.000040615122 | 0.0038574693 | 0.0020433058 | 50.192873% | 0.00019761772 |
| 30s | 0.000026286426 | 0.0000088308158 | 0.000017455610 | 0.0026876260 | 0.00086844758 | 50.134381% | 0.00010498342 |
| 60s | 0.000021441995 | 0.0000088263279 | 0.000012615668 | 0.0024038524 | 0.00058314139 | 50.120193% | 0.000085598696 |
| 300s | 0.000024611325 | 0.0000088279256 | 0.000015783400 | 0.0025069961 | 0.00067990922 | 50.125350% | 0.000098292358 |
| 900s | 0.000028897788 | 0.0000088463334 | 0.000020051455 | 0.0028242740 | 0.00099201775 | 50.141214% | 0.00011536315 |

`arrow classifier` is the best possible accuracy from the discretized block alone when forward and reversed orientations are equally likely. Chance is 50%.

## Consecutive history reversal

Longer histories use 16 approximately equal-mass active-return cells plus zero.

| block length | horizon | JS (bits) | null JS mean | JS above null | total variation | TV above null | arrow classifier |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1s | 0.0012191387 | 6.2182609e-7 | 0.0012185169 | 0.030012259 | 0.029476194 | 51.500613% |
| 3 | 2s | 0.0035766036 | 0.000010622022 | 0.0035659816 | 0.047256191 | 0.045482348 | 52.362810% |
| 4 | 3s | 0.0067210050 | 0.00018310994 | 0.0065378950 | 0.069101553 | 0.062450824 | 53.455078% |
| 5 | 4s | 0.012440109 | 0.0024993532 | 0.0099407559 | 0.090222344 | 0.068018962 | 54.511117% |

## One-step decomposition

| subset | share of transitions | JS (bits) | JS above null | total variation | TV above null | arrow classifier |
|---|---:|---:|---:|---:|---:|---:|
| bothReturnsActive | 46.400991% | 0.0026362592 | 0.0026180632 | 0.045253251 | 0.041991874 | 52.262663% |
| exactlyOneReturnIsZero | 35.743752% | 0.0016773844 | 0.0016765786 | 0.037820400 | 0.037034467 | 51.891020% |

## Annual stability of the one-step result

| window | transitions | JS (bits) | total variation | arrow classifier |
|---|---:|---:|---:|---:|
| 2021-07-25 to 2022-07-25 | 31,535,998 | 0.0036565793 | 0.048146122 | 52.407306% |
| 2022-07-25 to 2023-07-25 | 31,535,999 | 0.0015606321 | 0.031998003 | 51.599900% |
| 2023-07-25 to 2024-07-25 | 31,622,399 | 0.0047277655 | 0.051114876 | 52.555744% |
| 2024-07-25 to 2025-07-25 | 31,535,999 | 0.0016575628 | 0.029982465 | 51.499123% |
| 2025-07-25 to 2026-07-25 | 31,535,999 | 0.0023430221 | 0.040385180 | 52.019259% |

## Largest one-step probability currents

Positive current means the first direction occurs more often than its exact reverse.

| from | to | forward count | reverse count | net probability |
|---|---|---:|---:|---:|
| [-0.00386725, -0.00329019) bps excluding exact zero | exact zero | 1,677,625 | 1,471,805 | 0.0013045871 |
| [0.00333955, 0.00373927) bps excluding exact zero | [-0.00386725, -0.00329019) bps excluding exact zero | 888,283 | 771,478 | 0.00074036678 |
| exact zero | [0.00333955, 0.00373927) bps excluding exact zero | 1,082,869 | 968,783 | 0.00072313244 |
| [0.00103828, 0.00123085) bps excluding exact zero | exact zero | 1,201,462 | 1,095,725 | 0.00067021242 |
| exact zero | [-0.00132313, -0.00112012) bps excluding exact zero | 888,754 | 793,767 | 0.00060207371 |
| [0.00123085, 0.00143979) bps excluding exact zero | exact zero | 776,148 | 703,594 | 0.00045988246 |
| exact zero | [0.00373927, 0.00529294) bps excluding exact zero | 596,763 | 527,333 | 0.00044008104 |
| exact zero | [-0.000977971, -0.000828339) bps excluding exact zero | 1,413,693 | 1,354,659 | 0.00037418614 |
| exact zero | [-0.00152438, -0.00132313) bps excluding exact zero | 1,237,236 | 1,179,898 | 0.00036343607 |
| [0.00373927, 0.00529294) bps excluding exact zero | [-0.00386725, -0.00329019) bps excluding exact zero | 292,261 | 236,042 | 0.00035634331 |
| exact zero | [0.00213486, 0.00240525) bps excluding exact zero | 726,260 | 670,494 | 0.00035347197 |
| [-0.00132313, -0.00112012) bps excluding exact zero | [0.00103828, 0.00123085) bps excluding exact zero | 454,238 | 400,378 | 0.00034139082 |

## Conditional kernels

The machine-readable artifact contains the full 65 by 65 forward kernel `P(next | current)` and Bayes-reversed kernel `P(previous | current)`, along with state boundaries. Their difference is the measured detailed-balance violation, not a violation of Bayes' theorem.

## Methodological notes

- JS and total variation remain finite when a block orientation has no observed reverse. Entropy production uses a Jeffreys 0.5 count to avoid infinite plug-in KL.
- The orientation null fixes each block/reverse-block orbit count and assigns its directions with probability one half. Moments are exact for orbit totals through 256 and use their asymptotic normal/chi-square forms above that.
- Longer-block scores naturally have a larger sampling floor; `JS above null` subtracts the measured orientation-null mean.
- Pairwise symmetry does not prove full reversibility. The length-three through length-five tests look for higher-order arrows of time.

## Reproducibility

```text
node --conditions=development --import tsx scripts/analyze-return-reversibility.ts
```

The full conditional matrices, state definitions, null summaries, annual results, and probability currents are stored in `data/benchmarks/one-second-return-reversibility.json`.
