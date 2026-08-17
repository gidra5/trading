# Beta-companded multidimensional one-second return point clouds

Generated 2026-08-17T18:38:18.630371Z.

## Acceptance criteria

The direct JS-optimized 1D 32-knot reference has JS 0.0154573928 bits. The joint fit must not exceed this per active coordinate. KL is not calculated.

The point cloud must also beat the 1D 32-knot quantizer on every measured conditional mean and conditional median operation. Exact zero masks remain separate mixture components.

The beta transform targets a Jacobian proportional to the pilot density raised to beta. Beta=1 favors density uniformization; smaller beta retains more adaptive center allocation in dense regions.

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

### Transform and beta screening at 384 points

| family | beta | escort ESS | JS/d | conditional mean RMSEs | conditional median MAEs | max ratio |
|---|---:|---:|---:|---|---|---:|
| factorized | 0.5 | 2.297% | 0.019886742 | 0.03184561 | 0.01292587 | 10.6641 |
| factorized | 0.625 | 9.226% | 0.027160675 | 0.03443706 | 0.01312757 | 11.5318 |
| factorized | 0.75 | 40.079% | 0.027794968 | 0.03322534 | 0.01392801 | 11.1261 |
| factorized | 0.875 | 87.088% | 0.031876577 | 0.03584738 | 0.01411316 | 12.0041 |
| factorized | 1 | 100.000% | 0.033644773 | 0.03770231 | 0.01574441 | 12.6253 |
| scalar | 0.5 | 1.084% | 0.014444472 | 0.03087712 | 0.01115172 | 10.3397 |
| scalar | 0.625 | 4.368% | 0.015187849 | 0.03096139 | 0.008779755 | 10.368 |
| scalar | 0.75 | 27.655% | 0.022218627 | 0.03940321 | 0.008197908 | 13.1949 |
| scalar | 0.875 | 84.009% | 0.026263239 | 0.02861868 | 0.005792185 | 9.58346 |
| scalar | 1 | 100.000% | 0.02581937 | 0.02731962 | 0.003936761 | 9.14845 |
| vector | 0.5 | 1.084% | 0.012757926 | 0.02950695 | 0.009244604 | 9.88091 |
| vector | 0.625 | 4.369% | 0.015931798 | 0.03228044 | 0.009890718 | 10.8097 |
| vector | 0.75 | 27.657% | 0.030895652 | 0.04422689 | 0.01082251 | 14.8101 |
| vector | 0.875 | 84.009% | 0.033913017 | 0.03155325 | 0.008354404 | 10.5662 |
| vector | 1 | 100.000% | 0.022628325 | 0.03310608 | 0.003947411 | 11.0861 |

Selected: **scalar**, beta **1**.

### Knot-count search

| points | fit JS/d | final-year JS/d | conditional mean RMSEs | conditional median MAEs | max ratio | passes |
|---:|---:|---:|---|---|---:|---:|
| 32 | 0.096875753 | 0.20952516 | 0.06785208 | 0.0250021 | 22.7215 | no |
| 64 | 0.065619248 | 0.15082943 | 0.07343473 | 0.02188499 | 24.5909 | no |
| 96 | 0.057221544 | 0.1375038 | 0.0593677 | 0.0161401 | 19.8803 | no |
| 144 | 0.048172389 | 0.13180275 | 0.06162401 | 0.01228912 | 20.6359 | no |
| 256 | 0.039214822 | 0.11802987 | 0.04156334 | 0.008307955 | 13.9182 | no |
| 384 | 0.02581937 | 0.093418441 | 0.02731962 | 0.003936761 | 9.14845 | no |
| 576 | 0.016172466 | 0.079161339 | 0.02538365 | 0.003263025 | 8.50016 | no |
| 768 | 0.011495745 | 0.071854619 | 0.02015704 | 0.001947684 | 6.74994 | no |
| 1,024 | 0.012717193 | 0.085613463 | 0.01610471 | 0.001229382 | 5.39294 | no |
| 1,536 | 0.0091381659 | 0.077800817 | 0.01462531 | 0.0009769644 | 4.89754 | no |
| 2,048 | 0.010349778 | 0.079120006 | 0.007403161 | 0.000378919 | 2.47908 | no |
| 3,072 | 0.0051942904 | 0.069440765 | 0.002127771 | 7.259135e-05 | 1 | yes |
| 4,096 | 0.0033944719 | 0.062648346 | 0.001538825 | 5.966892e-05 | 1 | yes |

Smallest tested passing point cloud: **3072**. Best tested count: **3072** with maximum normalized error 1.

### Complete-five-year refit

| transform | beta | points | JS/d | conditional mean RMSEs | conditional median MAEs | max ratio | passes |
|---|---:|---:|---:|---|---|---:|---:|
| scalar | 1 | 3,072 | 0.0075191624 | 0.002273075 | 9.773715e-05 | 1 | yes |

Post-asinh matrix: `[[3.151944289114927,0.0],[0.0,3.151944289114927]]`

## 3D all-active component

Training sample: 385,414; final-year sample: 191,415.

### 1D 32-knot conditional-operation baselines

| target | conditioning coordinates/cells | mean RMSE bps | median MAE bps |
|---|---:|---:|---:|
| r2 | r1 (24) | 0.0043661994 | 0.0017530396 |
| r3 | r1,r2 (144) | 0.013787194 | 0.0056333789 |

### Transform and beta screening at 512 points

| family | beta | escort ESS | JS/d | conditional mean RMSEs | conditional median MAEs | max ratio |
|---|---:|---:|---:|---|---|---:|
| factorized | 0.6 | 1.570% | 0.014604166 | 0.03439444 / 0.08101427 | 0.01777297 / 0.02658499 | 10.1384 |
| factorized | 0.7 | 6.283% | 0.014463488 | 0.03545188 / 0.07350395 | 0.01807457 / 0.02764676 | 10.3104 |
| factorized | 0.8 | 33.045% | 0.017014813 | 0.04398702 / 0.08272218 | 0.0213919 / 0.02702335 | 12.2027 |
| factorized | 0.9 | 85.112% | 0.01988777 | 0.04064122 / 0.1162776 | 0.0216554 / 0.03457556 | 12.3531 |
| factorized | 1 | 100.000% | 0.020059474 | 0.04924153 / 0.1219679 | 0.01984177 / 0.04385867 | 11.3185 |
| scalar | 0.6 | 0.856% | 0.014295662 | 0.03327847 / 0.0710543 | 0.0170056 / 0.02592766 | 9.70064 |
| scalar | 0.7 | 3.165% | 0.015256974 | 0.09211548 / 0.09666645 | 0.01839467 / 0.02686734 | 21.0974 |
| scalar | 0.8 | 22.053% | 0.019328812 | 0.05643162 / 0.1206153 | 0.0188164 / 0.02556589 | 12.9247 |
| scalar | 0.9 | 81.440% | 0.02250962 | 0.04171757 / 0.07209104 | 0.01492816 / 0.02079612 | 9.55466 |
| scalar | 1 | 100.000% | 0.025357291 | 0.04802441 / 0.06737474 | 0.01430309 / 0.01817388 | 10.9991 |
| vector | 0.6 | 0.858% | 0.012971329 | 0.03637947 / 0.07366705 | 0.01900187 / 0.02601376 | 10.8394 |
| vector | 0.7 | 3.169% | 0.014708848 | 0.05252497 / 0.08786881 | 0.01914277 / 0.02647862 | 12.0299 |
| vector | 0.8 | 22.066% | 0.01907199 | 0.07944001 / 0.1243893 | 0.01894355 / 0.02525271 | 18.1943 |
| vector | 0.9 | 81.443% | 0.022655974 | 0.04948935 / 0.07160551 | 0.01574189 / 0.02023458 | 11.3347 |
| vector | 1 | 100.000% | 0.024917022 | 0.04501356 / 0.06480447 | 0.01297179 / 0.01779391 | 10.3096 |

Selected: **scalar**, beta **0.9**.

### Knot-count search

| points | fit JS/d | final-year JS/d | conditional mean RMSEs | conditional median MAEs | max ratio | passes |
|---:|---:|---:|---|---|---:|---:|
| 64 | 0.035403007 | 0.11666612 | 0.06141735 / 0.1002944 | 0.02699367 / 0.03582359 | 15.3982 | no |
| 128 | 0.032614938 | 0.10651702 | 0.07412243 / 0.09194987 | 0.02623512 / 0.03062623 | 16.9764 | no |
| 256 | 0.027531575 | 0.099533282 | 0.04510315 / 0.07539033 | 0.01838251 / 0.0240192 | 10.4861 | no |
| 384 | 0.02448706 | 0.093643924 | 0.04416787 / 0.07068761 | 0.01654616 / 0.02141733 | 10.1159 | no |
| 512 | 0.02250962 | 0.089662149 | 0.04171757 / 0.07209104 | 0.01492816 / 0.02079612 | 9.55466 | no |
| 768 | 0.020068063 | 0.084563661 | 0.03948816 / 0.06784483 | 0.0138187 / 0.01869739 | 9.04406 | no |
| 1,024 | 0.016897754 | 0.078964456 | 0.03429255 / 0.06768731 | 0.01188066 / 0.01715502 | 7.8541 | no |
| 1,536 | 0.012639863 | 0.074387494 | 0.03649887 / 0.06791628 | 0.009602718 / 0.0153023 | 8.35942 | no |
| 2,048 | 0.010753503 | 0.068084335 | 0.02978444 / 0.06244614 | 0.006951002 / 0.01237086 | 6.82159 | no |
| 3,072 | 0.01058448 | 0.071096003 | 0.02408367 / 0.05500808 | 0.004805636 / 0.009969524 | 5.51593 | no |
| 4,096 | 0.0095848467 | 0.068416513 | 0.02581465 / 0.05257635 | 0.004173953 / 0.009953742 | 5.91239 | no |
| 6,144 | 0.014836224 | 0.07681938 | 0.01782266 / 0.04502661 | 0.001322079 / 0.005835095 | 4.08196 | no |
| 8,192 | 0.012367846 | 0.076086126 | 0.01551004 / 0.02764122 | 0.00125288 / 0.006202098 | 3.5523 | no |
| 12,288 | 0.013158195 | 0.079591396 | 0.01288883 / 0.02382499 | 0.0005582529 / 0.003350604 | 2.95196 | no |
| 16,384 | 0.01041328 | 0.076046249 | 0.01657509 / 0.02528812 | 0.0005948895 / 0.00306079 | 3.79623 | no |
| 24,576 | 0.0062831998 | 0.067985297 | 0.007184592 / 0.01721606 | 0.0003275534 / 0.002141467 | 1.6455 | no |
| 32,768 | 0.0046902824 | 0.065417023 | 0.006971617 / 0.01485863 | 0.0002235383 / 0.001765544 | 1.59672 | no |

Smallest tested passing point cloud: **none**. Best tested count: **32768** with maximum normalized error 1.59672.

### Complete-five-year refit

| transform | beta | points | JS/d | conditional mean RMSEs | conditional median MAEs | max ratio | passes |
|---|---:|---:|---:|---|---|---:|---:|
| scalar | 0.9 | 32,768 | 0.0039651179 | 0.006352019 / 0.01782905 | 0.0002088064 / 0.002003405 | 1.68783 | no |

Post-asinh matrix: `[[2.2637793872433547,0.0,0.0],[0.0,2.2637793872433547,0.0],[0.0,0.0,2.2637793872433547]]`

## Interpretation

- Fit JS measures representation error. Final-year JS and conditional-operation errors also contain temporal distribution drift and are reported but do not select knot count.
- Conditional statistics are calculated from the continuous triangular mixture, not from nearest-center reconstructions.
- Counts are the smallest tested passing configurations in the recorded search bracket, not proofs over every omitted integer.

## Reproduction

```text
npm run analysis:return-multidimensional-knots
```

The machine artifact retains covariance and transform parameters plus the final centers, bandwidths, and weights.
