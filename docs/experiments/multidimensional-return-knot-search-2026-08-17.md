# Density-tempered multidimensional one-second return point clouds

Generated 2026-08-17T21:15:40.472971Z.

## Acceptance criteria

The direct JS-optimized 1D 32-knot reference has JS 0.0154573928 bits. The joint fit must not exceed this per active coordinate. KL is not calculated.

The point cloud must also beat the 1D 32-knot quantizer on every measured conditional mean and conditional median operation. Exact zero masks remain separate mixture components.

Beta affects only knot allocation after the invertible mapping. K-means observations are weighted by the smoothed mapped density raised to beta minus one. Smaller beta spends relatively fewer centers in the concentrated middle while leaving the mapping and tails geometrically unchanged. Final bandwidths and weights target the true distribution.

## Exact zero-mask split

| dimension | observations | all zero | any zero | all active |
|---:|---:|---:|---:|---:|
| 2 | 157,766,399 | 18.1871% | 54.1242% | 45.8758% |
| 3 | 157,766,398 | 9.9131% | 64.8488% | 35.1512% |

## 2D all-active component

Training sample: 487,702; final-year sample: 310,916.

### 1D 32-knot conditional-operation baselines

| target | conditioning coordinates/cells | mean RMSE bps | median MAE bps |
|---|---:|---:|---:|
| r2 | r1 (24) | 0.0029862566 | 0.0021474385 |

### Transform and allocation-beta screening at 384 points

| family | beta | allocation ESS | JS/d | conditional mean RMSEs | conditional median MAEs | max ratio |
|---|---:|---:|---:|---|---|---:|
| factorized | 0.5 | 70.154% | 0.039093295 | 0.03941379 | 0.01693163 | 13.1984 |
| factorized | 0.625 | 76.608% | 0.035159493 | 0.03810498 | 0.01806774 | 12.7601 |
| factorized | 0.75 | 84.626% | 0.035790954 | 0.03717967 | 0.01676361 | 12.4503 |
| factorized | 0.875 | 94.064% | 0.034237115 | 0.03494241 | 0.01467035 | 11.7011 |
| factorized | 1 | 100.000% | 0.033642866 | 0.03536575 | 0.0163116 | 11.8428 |
| scalar | 0.5 | 68.099% | 0.037454652 | 0.05767455 | 0.01122769 | 19.3133 |
| scalar | 0.625 | 74.997% | 0.03614547 | 0.0297236 | 0.005760876 | 9.95346 |
| scalar | 0.75 | 83.754% | 0.036184898 | 0.02717733 | 0.006196945 | 9.1008 |
| scalar | 0.875 | 93.895% | 0.035369394 | 0.03300552 | 0.006168162 | 11.0525 |
| scalar | 1 | 100.000% | 0.02581937 | 0.02754979 | 0.003927704 | 9.22553 |
| vector | 0.5 | 68.099% | 0.038440547 | 0.02704897 | 0.007824472 | 9.05782 |
| vector | 0.625 | 74.997% | 0.036313088 | 0.03782646 | 0.007870371 | 12.6668 |
| vector | 0.75 | 83.754% | 0.035616759 | 0.02663625 | 0.006915067 | 8.91961 |
| vector | 0.875 | 93.895% | 0.023887243 | 0.02734737 | 0.003849405 | 9.15774 |
| vector | 1 | 100.000% | 0.022628325 | 0.03313665 | 0.003943002 | 11.0964 |

Selected: **vector**, beta **0.75**.

### Knot-count search

| points | beta | fit JS/d | final-year JS/d | conditional mean RMSEs | conditional median MAEs | max ratio | passes |
|---:|---:|---:|---:|---|---|---:|---:|
| 32 | 0.75 | 0.10408429 | 0.21922666 | 0.04971189 | 0.02435915 | 16.6469 | no |
| 64 | 0.75 | 0.062030344 | 0.15300132 | 0.03917961 | 0.0154711 | 13.12 | no |
| 96 | 0.625 | 0.05538665 | 0.14210124 | 0.03947275 | 0.01365698 | 13.2181 | no |
| 144 | 0.5 | 0.051433647 | 0.13519405 | 0.03694467 | 0.01142699 | 12.3716 | no |
| 256 | 0.625 | 0.041400579 | 0.12018344 | 0.05019711 | 0.01029518 | 16.8094 | no |
| 384 | 0.75 | 0.035616759 | 0.11293427 | 0.02663625 | 0.006915067 | 8.91961 | no |
| 576 | 0.875 | 0.019362215 | 0.084015612 | 0.02476533 | 0.003524715 | 8.2931 | no |
| 768 | 0.875 | 0.013584285 | 0.080642564 | 0.01894351 | 0.002152125 | 6.34356 | no |
| 1,024 | 1 | 0.014551926 | 0.081880639 | 0.01487263 | 0.00107106 | 4.98036 | no |
| 1,536 | 1 | 0.0086063467 | 0.075780521 | 0.01399752 | 0.001014829 | 4.68731 | no |
| 2,048 | 1 | 0.0094713059 | 0.07373552 | 0.007359349 | 0.0003974103 | 2.46441 | no |
| 3,072 | 0.875 | 0.013705655 | 0.083857203 | 0.00279553 | 7.845842e-05 | 1 | yes |
| 4,096 | 0.75 | 0.010363293 | 0.078627455 | 0.00223185 | 7.786675e-05 | 1 | yes |

Smallest tested passing point cloud: **3072**. First passing count confirmed by the next tested size: **3072**. Best tested count: **3072** with maximum normalized error 1.

### Complete-five-year refit

| transform | beta | points | JS/d | conditional mean RMSEs | conditional median MAEs | max ratio | passes |
|---|---:|---:|---:|---|---|---:|---:|
| vector | 0.875 | 3,072 | 0.010199559 | 0.002193804 | 0.0001045444 | 1 | yes |

Post-asinh matrix: `[[3.153724060260527,0.0],[0.0,3.150166233004714]]`

## 3D all-active component

Training sample: 385,414; final-year sample: 191,415.

### 1D 32-knot conditional-operation baselines

| target | conditioning coordinates/cells | mean RMSE bps | median MAE bps |
|---|---:|---:|---:|
| r2 | r1 (24) | 0.0043661994 | 0.0017530396 |
| r3 | r1,r2 (144) | 0.013787194 | 0.0056333789 |

### Transform and allocation-beta screening at 512 points

| family | beta | allocation ESS | JS/d | conditional mean RMSEs | conditional median MAEs | max ratio |
|---|---:|---:|---:|---|---|---:|
| factorized | 0.6 | 81.696% | 0.023216094 | 0.05066831 / 0.1109801 | 0.02482757 / 0.04434148 | 14.1626 |
| factorized | 0.7 | 86.886% | 0.021893829 | 0.04996778 / 0.1215832 | 0.02331996 / 0.04301631 | 13.3026 |
| factorized | 0.8 | 92.328% | 0.021272221 | 0.04896885 / 0.1281582 | 0.02105525 / 0.04735066 | 12.0107 |
| factorized | 0.9 | 97.408% | 0.020372757 | 0.05288623 / 0.129553 | 0.0221566 / 0.04527676 | 12.639 |
| factorized | 1 | 100.000% | 0.020059759 | 0.04845808 / 0.1253527 | 0.01987991 / 0.04400021 | 11.3402 |
| scalar | 0.6 | 78.768% | 0.027651261 | 0.05193411 / 0.07598816 | 0.01497996 / 0.01652548 | 11.8946 |
| scalar | 0.7 | 84.901% | 0.026898499 | 0.0416238 / 0.07451506 | 0.01275014 / 0.01807502 | 9.53319 |
| scalar | 0.8 | 91.317% | 0.025941421 | 0.06210273 / 0.06768795 | 0.01499553 / 0.01777083 | 14.2235 |
| scalar | 0.9 | 97.152% | 0.025712874 | 0.03441071 / 0.06550646 | 0.01178341 / 0.01810052 | 7.88116 |
| scalar | 1 | 100.000% | 0.025357291 | 0.04801106 / 0.0662511 | 0.01421644 / 0.01788673 | 10.9961 |
| vector | 0.6 | 78.770% | 0.02782806 | 0.05703449 / 0.06720968 | 0.01606442 / 0.01977546 | 13.0627 |
| vector | 0.7 | 84.903% | 0.026596385 | 0.04552985 / 0.07281689 | 0.0124929 / 0.01873383 | 10.4278 |
| vector | 0.8 | 91.318% | 0.026061314 | 0.03806426 / 0.06724429 | 0.0117869 / 0.01765872 | 8.71794 |
| vector | 0.9 | 97.152% | 0.025657067 | 0.0363163 / 0.0648881 | 0.01095298 / 0.01940596 | 8.3176 |
| vector | 1 | 100.000% | 0.024917022 | 0.0439121 / 0.0674024 | 0.01293668 / 0.01795685 | 10.0573 |

Selected: **scalar**, beta **0.9**.

### Knot-count search

| points | beta | fit JS/d | final-year JS/d | conditional mean RMSEs | conditional median MAEs | max ratio | passes |
|---:|---:|---:|---:|---|---|---:|---:|
| 64 | 0.9 | 0.047061528 | 0.13703181 | 0.04624068 / 0.1168951 | 0.02406621 / 0.05261301 | 13.7283 | no |
| 128 | 0.9 | 0.039536417 | 0.11738225 | 0.03512798 / 0.09178534 | 0.01757658 / 0.03534178 | 10.0263 | no |
| 256 | 0.9 | 0.033070671 | 0.10745527 | 0.04622627 / 0.07661429 | 0.01573479 / 0.02263098 | 10.5873 | no |
| 384 | 0.8 | 0.029415288 | 0.10180352 | 0.03877367 / 0.07005573 | 0.01386421 / 0.01903351 | 8.88042 | no |
| 512 | 0.9 | 0.025712874 | 0.096071095 | 0.03441071 / 0.06550646 | 0.01178341 / 0.01810052 | 7.88116 | no |
| 768 | 0.8 | 0.022380872 | 0.09019975 | 0.03623105 / 0.06650672 | 0.009046882 / 0.01654396 | 8.29807 | no |
| 1,024 | 0.8 | 0.019762159 | 0.086064899 | 0.03282677 / 0.06261146 | 0.009366215 / 0.01506671 | 7.51839 | no |
| 1,536 | 0.9 | 0.014289309 | 0.075858054 | 0.02960084 / 0.05918362 | 0.006511518 / 0.01123281 | 6.77954 | no |
| 2,048 | 0.8 | 0.013321145 | 0.075658466 | 0.02857208 / 0.0557583 | 0.005868836 / 0.009584532 | 6.54392 | no |
| 3,072 | 0.9 | 0.01223528 | 0.075600126 | 0.02528029 / 0.05259423 | 0.003730298 / 0.00817991 | 5.79 | no |
| 4,096 | 1 | 0.014457873 | 0.080297561 | 0.01887406 / 0.03675082 | 0.002028778 / 0.006936466 | 4.32277 | no |
| 6,144 | 0.9 | 0.010864662 | 0.074361282 | 0.01810503 / 0.03832695 | 0.001637762 / 0.006997556 | 4.14663 | no |
| 8,192 | 0.9 | 0.011519021 | 0.078167034 | 0.01432282 / 0.02321027 | 0.0008439194 / 0.005636796 | 3.28039 | no |
| 12,288 | 0.9 | 0.012649902 | 0.082340164 | 0.009661117 / 0.01762202 | 0.0005651625 / 0.002294943 | 2.21271 | no |
| 16,384 | 1 | 0.0099942458 | 0.078048987 | 0.01067153 / 0.02040801 | 0.0003857878 / 0.002487568 | 2.44412 | no |
| 24,576 | 0.9 | 0.0053250177 | 0.070292871 | 0.006382677 / 0.01598865 | 0.0002623442 / 0.002406293 | 1.46184 | no |
| 32,768 | 1 | 0.0036725907 | 0.066447029 | 0.005836836 / 0.01303808 | 0.0001655554 / 0.001901011 | 1.33682 | no |

Smallest tested passing point cloud: **none**. First passing count confirmed by the next tested size: **none**. Best tested count: **32768** with maximum normalized error 1.33682.

### Complete-five-year refit

| transform | beta | points | JS/d | conditional mean RMSEs | conditional median MAEs | max ratio | passes |
|---|---:|---:|---:|---|---|---:|---:|
| scalar | 1 | 32,768 | 0.00326654 | 0.004654318 / 0.01473798 | 0.0002107424 / 0.001897244 | 1.23672 | no |

Post-asinh matrix: `[[3.0759326992014087,0.0,0.0],[0.0,3.0759326992014087,0.0],[0.0,0.0,3.0759326992014087]]`

## Conclusions

- In the four-year representation fit, the smallest tested 2D pass is 3,072 points. The first pass confirmed by the next tested size is 3072.
- No tested 3D count passes; the best is 32,768 points at 1.33682x the strictest 1D32 threshold.
- The complete-five-year 2D refit passes at 3,072 points with r2 conditional-mean RMSE (0.921737x).
- The complete-five-year 3D refit does not pass at 32,768 points with r2 conditional-mean RMSE (1.23672x).
- Screening / selected-count / complete-five-year allocation betas are 0.75 / 0.875 / 0.875 in 2D and 0.9 / 1 / 1 in 3D. Beta does not participate in the forward or inverse mapping; it only changes finite point placement.

## Interpretation

- Fit JS measures representation error. Final-year JS and conditional-operation errors also contain temporal distribution drift and are reported but do not select knot count.
- Conditional statistics are calculated from the continuous triangular mixture, not from nearest-center reconstructions.
- Counts are the smallest tested passing configurations in the recorded search bracket, not proofs over every omitted integer.
- Point-cloud fits are not nested across counts. A higher count can score worse because k-means initialization, local bandwidths, and operation-aware weight calibration are refit.
- When no complete-five-year candidate passes, the final section retains the tested candidate with the lowest maximum normalized error rather than relabeling it as passing.

## Reproduction

```text
npm run analysis:return-multidimensional-knots
```

The machine artifact retains covariance and transform parameters plus the final centers, bandwidths, and weights.
