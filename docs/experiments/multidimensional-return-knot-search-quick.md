# Beta-companded multidimensional one-second return point clouds

Generated 2026-08-17T18:14:39.230775Z.

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

### Transform and beta screening at 128 points

| family | beta | escort ESS | JS/d | conditional mean RMSEs | conditional median MAEs | max ratio |
|---|---:|---:|---:|---|---|---:|
| factorized | 0.5 | 2.341% | 0.10273959 | 0.05252405 | 0.02188293 | 17.5886 |
| factorized | 0.625 | 9.221% | 0.11995449 | 0.03193813 | 0.04090063 | 19.0462 |
| factorized | 0.75 | 39.794% | 0.10966833 | 0.04442179 | 0.04057347 | 18.8939 |
| factorized | 0.875 | 86.975% | 0.099810306 | 0.03515868 | 0.01192975 | 11.7735 |
| factorized | 1 | 100.000% | 0.11301887 | 0.03490612 | 0.01493736 | 11.6889 |
| scalar | 0.5 | 0.945% | 0.064244808 | 0.0306553 | 0.003589275 | 10.2655 |
| scalar | 0.625 | 3.919% | 0.12023356 | 0.03639796 | 0.01990782 | 12.1885 |
| scalar | 0.75 | 26.505% | 0.13690867 | 0.04391683 | 0.01070108 | 14.7063 |
| scalar | 0.875 | 83.803% | 0.10876548 | 0.02399645 | 0.008642788 | 8.03563 |
| scalar | 1 | 100.000% | 0.11072734 | 0.02300513 | 0.004651439 | 7.70367 |
| vector | 0.5 | 0.943% | 0.091421712 | 0.02406413 | 0.0122057 | 8.05829 |
| vector | 0.625 | 3.908% | 0.10127122 | 0.03399792 | 0.0348787 | 16.242 |
| vector | 0.75 | 26.457% | 0.16846332 | 0.05038655 | 0.03385199 | 16.8728 |
| vector | 0.875 | 83.783% | 0.10645743 | 0.02629688 | 0.01101278 | 8.80597 |
| vector | 1 | 100.000% | 0.16676688 | 0.03436949 | 0.05045027 | 23.4932 |

Selected: **scalar**, beta **1**.

### Knot-count search

| points | fit JS/d | final-year JS/d | conditional mean RMSEs | conditional median MAEs | max ratio | passes |
|---:|---:|---:|---|---|---:|---:|
| 64 | 0.17120569 | 0.23189216 | 0.02782368 | 0.009183252 | 11.076 | no |
| 128 | 0.11072734 | 0.18786117 | 0.02300513 | 0.004651439 | 7.70367 | no |
| 256 | 0.08029782 | 0.17511615 | 0.02356855 | 0.01389995 | 7.89234 | no |
| 512 | 0.061600315 | 0.13921303 | 0.01462693 | 0.00790584 | 4.89808 | no |

Smallest tested passing point cloud: **none**. Best tested count: **512** with maximum normalized error 4.89808.

### Complete-five-year refit

| transform | beta | points | JS/d | conditional mean RMSEs | conditional median MAEs | max ratio | passes |
|---|---:|---:|---:|---|---|---:|---:|
| scalar | 1 | 512 | 0.059197439 | 0.008708355 | 0.01029502 | 3.82972 | no |

Post-asinh matrix: `[[3.1648103486329133,0.0],[0.0,3.1648103486329133]]`

## 3D all-active component

Training sample: 385,414; final-year sample: 191,415.

### 1D 32-knot conditional-operation baselines

| target | conditioning coordinates/cells | mean RMSE bps | median MAE bps |
|---|---:|---:|---:|
| r2 | r1 (24) | 0.0043661994 | 0.0017530396 |
| r3 | r1,r2 (144) | 0.013787194 | 0.0056333789 |

### Transform and beta screening at 128 points

| family | beta | escort ESS | JS/d | conditional mean RMSEs | conditional median MAEs | max ratio |
|---|---:|---:|---:|---|---|---:|
| factorized | 0.6 | 1.409% | 0.063845284 | 0.03672455 / 0.1863477 | 0.01890182 / 0.07301724 | 1e+06 |
| factorized | 0.7 | 5.746% | 0.069752197 | 0.06842498 / 0.1639852 | 0.01613546 / 0.06255991 | 1e+06 |
| factorized | 0.8 | 31.928% | 0.11582915 | 0.05981454 / 0.6401758 | 0.02015347 / 0.1349827 | 46.4326 |
| factorized | 0.9 | 84.967% | 0.090562725 | 0.07978175 / 0.1354886 | 0.03124486 / 0.07829741 | 1e+06 |
| factorized | 1 | 100.000% | 0.093438526 | 0.04474939 / 0.1084551 | 0.03706813 / 0.04266589 | 1e+06 |
| scalar | 0.6 | 0.837% | 0.09508257 | 0.07676684 / 0.1959057 | 0.023888 / 0.09854453 | 1e+06 |
| scalar | 0.7 | 3.110% | 0.076622391 | 0.07748554 / 0.1364663 | 0.02589956 / 0.05911536 | 1e+06 |
| scalar | 0.8 | 21.934% | 0.11388408 | 0.06392665 / 0.201014 | 0.01421195 / 0.06017328 | 1e+06 |
| scalar | 0.9 | 81.489% | 0.073837576 | 0.06273689 / 0.1089863 | 0.01190064 / 0.02852733 | 14.3688 |
| scalar | 1 | 100.000% | 0.11444467 | 0.03496279 / 0.1682072 | 0.01435608 / 0.07700367 | 1e+06 |
| vector | 0.6 | 0.831% | 0.067393849 | 0.07117777 / 0.1385264 | 0.02107859 / 0.0644502 | 1e+06 |
| vector | 0.7 | 3.080% | 0.11211581 | 0.08790871 / 0.1805686 | 0.01122372 / 0.06389618 | 1e+06 |
| vector | 0.8 | 21.778% | 0.075397641 | 0.04501957 / 0.1360831 | 0.01459431 / 0.06577788 | 11.6765 |
| vector | 0.9 | 81.399% | 0.08493868 | 0.0604132 / 0.1789251 | 0.01882447 / 0.07165018 | 1e+06 |
| vector | 1 | 100.000% | 0.097886956 | 0.0321904 / 0.1214206 | 0.00949019 / 0.07199844 | 12.7807 |

Selected: **vector**, beta **0.8**.

### Knot-count search

| points | fit JS/d | final-year JS/d | conditional mean RMSEs | conditional median MAEs | max ratio | passes |
|---:|---:|---:|---|---|---:|---:|
| 64 | 0.083128803 | 0.15223694 | 0.1430505 / 0.2634204 | 0.02616011 / 0.09276993 | 1e+06 | no |
| 128 | 0.075397641 | 0.1432373 | 0.04501957 / 0.1360831 | 0.01459431 / 0.06577788 | 11.6765 | no |
| 256 | 0.062476864 | 0.1331525 | 0.1156111 / 0.3915314 | 0.01897469 / 0.0387198 | 1e+06 | no |
| 512 | 0.069772259 | 0.12703735 | 0.04079076 / 0.1713258 | 0.00878077 / 0.03243165 | 12.4264 | no |

Smallest tested passing point cloud: **none**. Best tested count: **128** with maximum normalized error 11.6765.

### Complete-five-year refit

| transform | beta | points | JS/d | conditional mean RMSEs | conditional median MAEs | max ratio | passes |
|---|---:|---:|---:|---|---|---:|---:|
| vector | 0.8 | 512 | 0.062066086 | 1.159744 / 0.3210365 | 0.007209865 / 0.03129418 | 308.162 | no |

Post-asinh matrix: `[[1.5244350591292744,0.0,0.0],[0.0,1.5135359525279108,0.0],[0.0,0.0,1.5359530942946493]]`

## Interpretation

- Fit JS measures representation error. Final-year JS and conditional-operation errors also contain temporal distribution drift and are reported but do not select knot count.
- Conditional statistics are calculated from the continuous triangular mixture, not from nearest-center reconstructions.
- Counts are the smallest tested passing configurations in the recorded search bracket, not proofs over every omitted integer.

## Reproduction

```text
npm run analysis:return-multidimensional-knots
```

The machine artifact retains covariance and transform parameters plus the final centers, bandwidths, and weights.
