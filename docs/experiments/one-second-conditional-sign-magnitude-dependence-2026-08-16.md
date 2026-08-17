# Conditional one-second sign–magnitude dependence

Generated 2026-08-15T22:33:06.162Z. The analysis uses 101,400,987 active BTCUSDT one-second target returns from 2021-07-25T00:00:00.000Z through 2026-07-25T00:00:00.000Z. Every history state uses only returns strictly before its target.

## Result

At least one tested history reveals material conditional sign-magnitude coupling.

Conditional independence requires both `P(sign | magnitude, history) = P(sign | history)` and `P(magnitude | sign, history) = P(magnitude | history)`. Conditional mutual information measures these equivalent failures in sample; rolling gains test whether they persist into later years.

## History comparison at 32 magnitude cells

| history | states | CMI (bits) | CMI above bias | conditional TV | sign entropy explained | rolling sign gain (bits) | rolling magnitude gain (bits) | rolling sign-accuracy gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No history | 1 | 0.000015455054 | 0.000015234525 | 0.0018803408 | 0.001523% | -0.000057240719 | -0.000057240719 | 0.069716 pp |
| Previous sign | 3 | 0.079074329 | 0.079073668 | 0.12527569 | 7.963245% | 0.054138924 | 0.054138924 | 2.712679 pp |
| Last 3 signs | 27 | 0.11783424 | 0.11782829 | 0.14486787 | 12.604205% | 0.079999096 | 0.079999096 | 0.739667 pp |
| Last 5 signs | 243 | 0.12348546 | 0.12343188 | 0.14203871 | 13.649708% | 0.081272115 | 0.081272115 | 1.330753 pp |
| Previous return (17 states) | 17 | 0.080763580 | 0.080759917 | 0.083618359 | 8.677169% | 0.063054497 | 0.063054497 | 2.348113 pp |
| Last 2 returns (9 states each) | 81 | 0.11130823 | 0.11129037 | 0.10968347 | 12.546739% | 0.064336645 | 0.064336645 | -0.551414 pp |
| Last 3 returns (9 states each) | 729 | 0.12321544 | 0.12305548 | 0.11787284 | 14.356249% | 0.069459140 | 0.069459140 | -0.944052 pp |

## Resolution robustness for Last 5 signs

| magnitude cells | CMI (bits) | null bias | CMI above bias | conditional TV | sign entropy explained |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.11896568 | 0.000025929844 | 0.11893975 | 0.13965323 | 13.152947% |
| 32 | 0.12348546 | 0.000053588345 | 0.12343188 | 0.14203871 | 13.649708% |
| 64 | 0.12582806 | 0.00010890535 | 0.12571915 | 0.14405758 | 13.902647% |

## Previous-sign state breakdown

This separates the three states inside the simplest nontrivial history. The target itself is always active; `Previous zero` means only that the immediately preceding one-second return was zero.

| history state | active targets | P(positive) | P(positive | magnitude) range | TV of magnitude laws by sign | CMI above bias | rolling log-score gain |
|---|---:|---:|---:|---:|---:|---:|
| Previous negative | 36,626,385 | 55.6910% | 39.585%–89.375% | 34.2810% | 0.10997480 | 0.076738841 |
| Previous zero | 28,195,815 | 49.9746% | 47.631%–53.256% | 2.5094% | 0.00070038039 | 0.00020990779 |
| Previous positive | 36,578,787 | 44.0980% | 10.133%–59.988% | 34.1162% | 0.10854435 | 0.076154097 |

## Rolling annual holdouts

Each annual target window uses all preceding annual windows to estimate its conditional tables. Positive gains favor coupling sign and magnitude; negative gains favor conditional factorization.

### No history

| test year index | active targets | sign log-loss gain (bits) | magnitude log-loss gain (bits) | sign-accuracy gain |
|---:|---:|---:|---:|---:|
| 1 | 26,003,486 | -0.00017841363 | -0.00017841363 | -0.046986 pp |
| 2 | 17,439,786 | -0.000021653220 | -0.000021653220 | 0.002729 pp |
| 3 | 18,160,597 | -9.3229602e-7 | -9.3229602e-7 | 0.094501 pp |
| 4 | 16,961,724 | 0.000031646771 | 0.000031646771 | 0.290967 pp |

### Previous sign

| test year index | active targets | sign log-loss gain (bits) | magnitude log-loss gain (bits) | sign-accuracy gain |
|---:|---:|---:|---:|---:|
| 1 | 26,003,486 | -0.0098298337 | -0.0098298337 | -5.823346 pp |
| 2 | 17,439,786 | 0.10402002 | 0.10402002 | 5.181222 pp |
| 3 | 18,160,597 | 0.033788913 | 0.033788913 | 4.626995 pp |
| 4 | 16,961,724 | 0.12270881 | 0.12270881 | 11.211248 pp |

### Last 3 signs

| test year index | active targets | sign log-loss gain (bits) | magnitude log-loss gain (bits) | sign-accuracy gain |
|---:|---:|---:|---:|---:|
| 1 | 26,003,486 | 0.022904833 | 0.022904833 | -2.494239 pp |
| 2 | 17,439,786 | 0.16043097 | 0.16043097 | 4.997768 pp |
| 3 | 18,160,597 | 0.00063548213 | 0.00063548213 | -4.459468 pp |
| 4 | 16,961,724 | 0.16980281 | 0.16980281 | 6.885969 pp |

### Last 5 signs

| test year index | active targets | sign log-loss gain (bits) | magnitude log-loss gain (bits) | sign-accuracy gain |
|---:|---:|---:|---:|---:|
| 1 | 26,003,486 | 0.032791273 | 0.032791273 | -1.291504 pp |
| 2 | 17,439,786 | 0.16330501 | 0.16330501 | 6.409052 pp |
| 3 | 18,160,597 | -0.017956281 | -0.017956281 | -5.661972 pp |
| 4 | 16,961,724 | 0.17749357 | 0.17749357 | 7.616401 pp |

### Previous return (17 states)

| test year index | active targets | sign log-loss gain (bits) | magnitude log-loss gain (bits) | sign-accuracy gain |
|---:|---:|---:|---:|---:|
| 1 | 26,003,486 | 0.016886895 | 0.016886895 | 5.152036 pp |
| 2 | 17,439,786 | 0.12186974 | 0.12186974 | 1.348250 pp |
| 3 | 18,160,597 | 0.035693962 | 0.035693962 | 0.170925 pp |
| 4 | 16,961,724 | 0.10265407 | 0.10265407 | 1.408625 pp |

### Last 2 returns (9 states each)

| test year index | active targets | sign log-loss gain (bits) | magnitude log-loss gain (bits) | sign-accuracy gain |
|---:|---:|---:|---:|---:|
| 1 | 26,003,486 | 0.026431633 | 0.026431633 | 1.691596 pp |
| 2 | 17,439,786 | 0.15385323 | 0.15385323 | 1.015064 pp |
| 3 | 18,160,597 | -0.052700946 | -0.052700946 | -7.773285 pp |
| 4 | 16,961,724 | 0.15571797 | 0.15571797 | 2.131588 pp |

### Last 3 returns (9 states each)

| test year index | active targets | sign log-loss gain (bits) | magnitude log-loss gain (bits) | sign-accuracy gain |
|---:|---:|---:|---:|---:|
| 1 | 26,003,486 | 0.030011354 | 0.030011354 | 1.244833 pp |
| 2 | 17,439,786 | 0.16329579 | 0.16329579 | 1.247882 pp |
| 3 | 18,160,597 | -0.063964464 | -0.063964464 | -9.655723 pp |
| 4 | 16,961,724 | 0.17630802 | 0.17630802 | 2.773946 pp |

## History-state definitions

- **No history:** Unconditional sign-magnitude dependence baseline.
- **Previous sign:** Previous return is negative, exactly zero, or positive.
- **Last 3 signs:** Ordered ternary sign/zero history of the previous three returns.
- **Last 5 signs:** Ordered ternary sign/zero history of the previous five returns.
- **Previous return (17 states):** Previous return: zero or sign crossed with eight magnitude cells.
- **Last 2 returns (9 states each):** Ordered two-return history: zero or sign crossed with four magnitude cells.
- **Last 3 returns (9 states each):** Ordered three-return history: zero or sign crossed with four magnitude cells.

## Reproducibility

```text
node --conditions=development --import tsx scripts/analyze-return-conditional-sign-magnitude.ts
```

The complete history/resolution metrics and annual rolling evaluations are stored in `data/benchmarks/one-second-conditional-sign-magnitude-dependence.json`.
