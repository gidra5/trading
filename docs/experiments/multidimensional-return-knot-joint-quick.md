# Joint multidimensional return-knot optimization

Generated 2026-08-17T22:13:33.446362Z.

The r2|r1 conditional-mean RMSE is disabled as an optimization loss and acceptance criterion, but remains in every diagnostic table. All other density and conditional operation thresholds remain active.

## 2D

Initial: 3,072 points, beta 0.875, active score 1, joint objective 0.605085.

Final: **2,304 points**, beta **0.875**, active score **1**, joint objective **0.155871**, passes.

### Fixed-point checks

| points | sweeps | converged | covariance step | transform step | beta step |
|---:|---:|---:|---:|---:|---:|
| 3,072 | 1 | safety cap | 0.07 | 0.06 | 0.125 |
| 2,304 | 1 | safety cap | 0.07 | 0.06 | 0.125 |

| metric | ratio to 1D32 | active | passes |
|---|---:|---:|---:|
| density JS/d | 0.169977 | yes | yes |
| r2 conditional mean | 12.9093 | no | no |
| r2 conditional median | 0.0474498 | yes | yes |

### Accepted/rejected iteration trace

| stage | points | beta | objective | active score | passes |
|---|---:|---:|---:|---:|---:|
| stored verified initial | 3,072 | 0.875 | 0.605085 | 1 | yes |
| common-fidelity warm refit | 3,072 | 0.875 | 0.208889 | 1 | yes |
| covariance shape 1/2 | 3,072 | 0.875 | 0.208889 | 1 | yes |
| covariance shape 2/2 | 3,072 | 0.875 | 0.208889 | 1 | yes |
| post-asinh transform 1/4 | 3,072 | 0.875 | 0.208889 | 1 | yes |
| post-asinh transform 2/4 | 3,072 | 0.875 | 0.208889 | 1 | yes |
| post-asinh transform 3/4 | 3,072 | 0.875 | 0.208889 | 1 | yes |
| post-asinh transform 4/4 | 3,072 | 0.875 | 0.208889 | 1 | yes |
| allocation beta | 3,072 | 0.75 | 0.195742 | 1 | yes |
| knot refit | 3,072 | 0.75 | 0.195742 | 1 | yes |
| fixed-point safety cap reached | 3,072 | 0.75 | 0.195742 | 1 | yes |
| prune proposal 3072->2304 | 2,304 | 0.875 | 0.25169 | 1 | yes |
| covariance shape 1/2 | 2,304 | 0.75 | 0.368445 | 1 | yes |
| covariance shape 2/2 | 2,304 | 0.75 | 0.368445 | 1 | yes |
| post-asinh transform 1/4 | 2,304 | 0.75 | 0.368445 | 1 | yes |
| post-asinh transform 2/4 | 2,304 | 0.75 | 0.368445 | 1 | yes |
| post-asinh transform 3/4 | 2,304 | 0.75 | 0.368445 | 1 | yes |
| post-asinh transform 4/4 | 2,304 | 0.75 | 0.368445 | 1 | yes |
| allocation beta | 2,304 | 0.875 | 0.25967 | 1 | yes |
| knot refit | 2,304 | 0.875 | 0.25169 | 1 | yes |
| fixed-point safety cap reached | 2,304 | 0.875 | 0.25169 | 1 | yes |
| accepted prune | 2,304 | 0.875 | 0.25169 | 1 | yes |

Covariance adjustment from empirical whitening: `[[1.0,0.0],[0.0,1.0]]`

Post-asinh matrix: `[[3.153724060260527,0.0],[0.0,3.150166233004714]]`

## Reproduction

```text
node scripts/run-ml-python.mjs ml/joint_optimize_multidimensional_return_knots.py
```
