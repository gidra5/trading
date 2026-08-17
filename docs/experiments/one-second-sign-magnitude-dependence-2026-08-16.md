# One-second return sign–magnitude dependence

Generated 2026-08-15T22:10:03.956Z. The analysis uses 101,400,987 nonzero BTCUSDT one-second log returns from 2021-07-25T00:00:00.000Z through 2026-07-25T00:00:00.000Z. The 56,365,412 exact-zero returns are excluded because zero has no sign.

## Result

Sign and magnitude are measurably but weakly dependent at the measured resolution.

At 64 magnitude cells, mutual information is 0.000017718092 bits versus a 4.4817013e-7-bit plug-in null bias. Magnitude raises in-sample optimal sign accuracy by 0.127491 percentage points, of which 0.000629 points are expected from cell-selection noise.

At the finest 2,048-cell audit, bias-corrected mutual information is 0.000031474197 bits and sign-accuracy gain above its null expectation is 0.088216 percentage points.

## Interpretation

- Magnitude explains only 0.003147% of sign entropy at the finest audited resolution. Unconditionally, sign and magnitude are therefore extremely close to independent, though not exactly independent.
- At 64 cells, the two conditional magnitude distributions differ by only 0.3948% TV. Positive-return magnitudes are slightly smaller at the median but slightly larger in the far tail, so the residual relationship is weak and non-monotone.
- Dependence is not stable across years: corrected magnitude-only sign gain ranges from 0.0274 to 0.3765 percentage points.
- A factorized sign and magnitude output is a sound unconditional baseline. It should remain possible for a conditional history model to introduce a small sign–magnitude coupling rather than enforcing exact independence.

## Resolution robustness

| magnitude cells | mutual information (bits) | MI above bias | JS from product (bits) | TV above null | Cramer's V | accuracy gain above null |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.0000092669994 | 0.0000092172027 | 0.0000023167565 | 0.0013925386 | 0.0035842349 | 0.090092 pp |
| 16 | 0.000013481779 | 0.000013375072 | 0.0000033704794 | 0.0014939414 | 0.0043231276 | 0.091669 pp |
| 32 | 0.000015455054 | 0.000015234525 | 0.0000038637995 | 0.0016562298 | 0.0046287107 | 0.114749 pp |
| 64 | 0.000017718092 | 0.000017269922 | 0.0000044295708 | 0.0016568582 | 0.0049560184 | 0.126862 pp |
| 128 | 0.000022923338 | 0.000022019883 | 0.0000057309570 | 0.0016904584 | 0.0056371619 | 0.137162 pp |
| 256 | 0.000026816659 | 0.000025002637 | 0.0000067044623 | 0.0015907807 | 0.0060970164 | 0.136046 pp |
| 512 | 0.000031021118 | 0.000027385961 | 0.0000077557345 | 0.0014183100 | 0.0065575196 | 0.126023 pp |
| 1024 | 0.000038385573 | 0.000031108143 | 0.0000095971931 | 0.0012272506 | 0.0072943691 | 0.112481 pp |
| 2048 | 0.000046036169 | 0.000031474197 | 0.000011510493 | 0.00093974305 | 0.0079880552 | 0.088216 pp |

## Conditional magnitude distributions

On the fixed 64-cell view, the magnitude laws conditional on negative and positive sign have JS 0.000017718159 bits and total variation 0.0039476082. Mean magnitude is 0.50990756 bps after negative returns and 0.51167577 bps after positive returns.

| magnitude quantile | negative (bps) | positive (bps) | positive / negative |
|---|---:|---:|---:|
| p50 | 0.10963359 | 0.10852347 | 0.98987428 |
| p90 | 1.4787074 | 1.4781888 | 0.99964931 |
| p99 | 4.0923266 | 4.1359053 | 1.0106489 |
| p999 | 8.8071685 | 8.9686037 | 1.0183300 |

## Linear correlations

- Correlation of sign with magnitude: 0.00092750716.
- Correlation of sign with log magnitude: -0.0012826165.

Zero correlation would not prove independence; mutual information and the full conditional-bin comparison are the primary tests.

## Annual stability

| window | active returns | positive probability | MI above bias (bits) | TV above null | accuracy gain above null |
|---|---:|---:|---:|---:|---:|
| 2021-07-25 to 2022-07-25 | 22,835,394 | 49.694917% | 0.00010692364 | 0.0029518921 | 0.190655 pp |
| 2022-07-25 to 2023-07-25 | 26,003,486 | 49.871475% | 0.0000043107646 | 0.00045415691 | 0.027400 pp |
| 2023-07-25 to 2024-07-25 | 17,439,786 | 49.952173% | 0.000026557889 | 0.0010972024 | 0.092192 pp |
| 2024-07-25 to 2025-07-25 | 18,160,597 | 50.165234% | 0.000050194333 | 0.0022504925 | 0.184544 pp |
| 2025-07-25 to 2026-07-25 | 16,961,724 | 49.998850% | 0.00012024211 | 0.0037713127 | 0.376513 pp |

## Largest dependence contributions

| magnitude interval (bps) | mass | positive probability | deviation from baseline | MI contribution (bits) |
|---|---:|---:|---:|---:|
| [0.084473832, 0.10903240) | 1.56295% | 49.21800% | -0.70151 pp | 0.0000022194181 |
| [0.038430807, 0.060380859) | 1.56295% | 49.32924% | -0.59027 pp | 0.0000015713310 |
| [3.5058221, +inf) | 1.56203% | 50.46805% | 0.54854 pp | 0.0000013561680 |
| [0.018223366, 0.038430807) | 1.56252% | 49.37810% | -0.54141 pp | 0.0000013216124 |
| [0.10903240, 0.13554608) | 1.56329% | 49.53368% | -0.38583 pp | 6.7148918e-7 |
| [0.060380859, 0.084473832) | 1.56125% | 49.55174% | -0.36777 pp | 6.0931417e-7 |
| [0.0012231643, 0.0013478181) | 1.56196% | 50.27642% | 0.35691 pp | 5.7409233e-7 |
| [0.0011414723, 0.0012231643) | 1.56363% | 50.27016% | 0.35065 pp | 5.5473656e-7 |
| [0.92633177, 0.99786605) | 1.55991% | 50.26385% | 0.34434 pp | 5.3367792e-7 |
| [0.0010411318, 0.0010891506) | 1.55959% | 50.25774% | 0.33823 pp | 5.1479866e-7 |
| [0.00086514598, 0.00090878437) | 1.56380% | 50.25206% | 0.33255 pp | 4.9900928e-7 |
| [0.0059368793, 0.018223366) | 1.56307% | 50.24259% | 0.32308 pp | 4.7076705e-7 |

## Method

The independence null uses the observed global sign probability, not an assumed 50/50 split. Magnitudes are first accumulated on a 131,072-cell logarithmic grid, then combined into approximately equal-mass cells. This preserves the central tick-scale structure and far tails while allowing the same source scan to be audited at several resolutions.

## Reproducibility

```text
node --conditions=development --import tsx scripts/analyze-return-sign-magnitude.ts
```

The complete magnitude edges, signed counts, conditional probabilities, annual results, and per-cell information contributions are stored in `data/benchmarks/one-second-sign-magnitude-dependence.json`.
