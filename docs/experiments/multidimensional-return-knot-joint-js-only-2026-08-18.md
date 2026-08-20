# Joint multidimensional return-knot optimization

Generated 2026-08-18T08:29:36.710645Z.

Only density Jensen-Shannon divergence is used for optimization and acceptance. Conditional means, medians, and coverage remain in the diagnostic tables but do not affect fitting or pruning.

## 2D

Initial: 3,072 points, beta 0.875, active score 0.65985, joint objective 0.65985.

Final: **460 points**, beta **1**, active score **0.973626**, joint objective **0.973626**, passes.

### Fixed-point checks

| points | sweeps | converged | covariance step | transform step | beta step |
|---:|---:|---:|---:|---:|---:|
| 3,072 | 4 | yes | 0.0175 | 0.015 | 0.03125 |
| 2,304 | 3 | yes | 0.0175 | 0.015 | 0.03125 |
| 2,304 | 6 | yes | 0.0175 | 0.015 | 0.03125 |
| 1,728 | 7 | yes | 0.0175 | 0.015 | 0.03125 |
| 1,296 | 5 | yes | 0.0175 | 0.015 | 0.03125 |
| 972 | 5 | yes | 0.0175 | 0.015 | 0.03125 |
| 729 | 12 | yes | 0.0175 | 0.015 | 0.03125 |
| 729 | 4 | yes | 0.0175 | 0.015 | 0.03125 |
| 547 | 7 | yes | 0.0175 | 0.015 | 0.03125 |
| 547 | 7 | yes | 0.0175 | 0.015 | 0.03125 |
| 638 | 8 | yes | 0.0175 | 0.015 | 0.03125 |
| 547 | 5 | yes | 0.0175 | 0.015 | 0.03125 |
| 547 | 8 | yes | 0.0175 | 0.015 | 0.03125 |
| 456 | 3 | yes | 0.0175 | 0.015 | 0.03125 |
| 456 | 4 | yes | 0.0175 | 0.015 | 0.03125 |
| 502 | 3 | yes | 0.0175 | 0.015 | 0.03125 |
| 457 | 3 | yes | 0.0175 | 0.015 | 0.03125 |
| 457 | 5 | yes | 0.0175 | 0.015 | 0.03125 |
| 480 | 3 | yes | 0.0175 | 0.015 | 0.03125 |
| 458 | 3 | yes | 0.0175 | 0.015 | 0.03125 |
| 458 | 3 | yes | 0.0175 | 0.015 | 0.03125 |
| 469 | 5 | yes | 0.0175 | 0.015 | 0.03125 |
| 458 | 6 | yes | 0.0175 | 0.015 | 0.03125 |
| 458 | 4 | yes | 0.0175 | 0.015 | 0.03125 |
| 464 | 5 | yes | 0.0175 | 0.015 | 0.03125 |
| 459 | 6 | yes | 0.0175 | 0.015 | 0.03125 |
| 459 | 5 | yes | 0.0175 | 0.015 | 0.03125 |
| 462 | 4 | yes | 0.0175 | 0.015 | 0.03125 |
| 462 | 7 | yes | 0.0175 | 0.015 | 0.03125 |
| 463 | 7 | yes | 0.0175 | 0.015 | 0.03125 |
| 463 | 9 | yes | 0.0175 | 0.015 | 0.03125 |

| metric | ratio to 1D32 | active | passes |
|---|---:|---:|---:|
| density JS/d | 0.973626 | yes | yes |
| r2 conditional mean | 11.2162 | no | no |
| r2 conditional median | 1.2044 | no | no |

### Accepted/rejected iteration trace

| stage | points | beta | objective | active score | passes |
|---|---:|---:|---:|---:|---:|
| stored verified initial | 3,072 | 0.875 | 0.65985 | 0.65985 | yes |
| common-fidelity warm refit | 3,072 | 0.875 | 0.145082 | 0.145082 | yes |
| covariance shape 1/2 | 3,072 | 0.875 | 0.145082 | 0.145082 | yes |
| covariance shape 2/2 | 3,072 | 0.875 | 0.145082 | 0.145082 | yes |
| post-asinh transform 1/4 | 3,072 | 0.875 | 0.145082 | 0.145082 | yes |
| post-asinh transform 2/4 | 3,072 | 0.875 | 0.145082 | 0.145082 | yes |
| post-asinh transform 3/4 | 3,072 | 0.875 | 0.145082 | 0.145082 | yes |
| post-asinh transform 4/4 | 3,072 | 0.875 | 0.145082 | 0.145082 | yes |
| allocation beta | 3,072 | 1 | 0.140913 | 0.140913 | yes |
| knot refit | 3,072 | 1 | 0.140913 | 0.140913 | yes |
| covariance shape 1/2 | 3,072 | 1 | 0.140913 | 0.140913 | yes |
| covariance shape 2/2 | 3,072 | 1 | 0.140913 | 0.140913 | yes |
| post-asinh transform 1/4 | 3,072 | 1 | 0.140913 | 0.140913 | yes |
| post-asinh transform 2/4 | 3,072 | 1 | 0.140913 | 0.140913 | yes |
| post-asinh transform 3/4 | 3,072 | 1 | 0.140913 | 0.140913 | yes |
| post-asinh transform 4/4 | 3,072 | 1 | 0.140913 | 0.140913 | yes |
| allocation beta | 3,072 | 1 | 0.140913 | 0.140913 | yes |
| knot refit (cached stationary) | 3,072 | 1 | 0.140913 | 0.140913 | yes |
| refined parameter-search resolution | 3,072 | 1 | 0.140913 | 0.140913 | yes |
| covariance shape 1/2 | 3,072 | 1 | 0.140913 | 0.140913 | yes |
| covariance shape 2/2 | 3,072 | 1 | 0.140913 | 0.140913 | yes |
| post-asinh transform 1/4 | 3,072 | 1 | 0.140913 | 0.140913 | yes |
| post-asinh transform 2/4 | 3,072 | 1 | 0.140913 | 0.140913 | yes |
| post-asinh transform 3/4 | 3,072 | 1 | 0.140913 | 0.140913 | yes |
| post-asinh transform 4/4 | 3,072 | 1 | 0.140913 | 0.140913 | yes |
| allocation beta | 3,072 | 1 | 0.140913 | 0.140913 | yes |
| knot refit (cached stationary) | 3,072 | 1 | 0.140913 | 0.140913 | yes |
| refined parameter-search resolution | 3,072 | 1 | 0.140913 | 0.140913 | yes |
| covariance shape 1/2 | 3,072 | 1 | 0.140913 | 0.140913 | yes |
| covariance shape 2/2 | 3,072 | 1 | 0.140913 | 0.140913 | yes |
| post-asinh transform 1/4 | 3,072 | 1 | 0.140913 | 0.140913 | yes |
| post-asinh transform 2/4 | 3,072 | 1 | 0.140913 | 0.140913 | yes |
| post-asinh transform 3/4 | 3,072 | 1 | 0.140913 | 0.140913 | yes |
| post-asinh transform 4/4 | 3,072 | 1 | 0.140913 | 0.140913 | yes |
| allocation beta | 3,072 | 1 | 0.140913 | 0.140913 | yes |
| knot refit (cached stationary) | 3,072 | 1 | 0.140913 | 0.140913 | yes |
| fixed point reached | 3,072 | 1 | 0.140913 | 0.140913 | yes |
| cold restart 3072->2304 | 2,304 | 1 | 0.232594 | 0.232594 | yes |
| covariance shape 1/2 | 2,304 | 1 | 0.240967 | 0.240967 | yes |
| covariance shape 2/2 | 2,304 | 1 | 0.240967 | 0.240967 | yes |
| post-asinh transform 1/4 | 2,304 | 1 | 0.240967 | 0.240967 | yes |
| post-asinh transform 2/4 | 2,304 | 1 | 0.240967 | 0.240967 | yes |
| post-asinh transform 3/4 | 2,304 | 1 | 0.240967 | 0.240967 | yes |
| post-asinh transform 4/4 | 2,304 | 1 | 0.240967 | 0.240967 | yes |
| allocation beta | 2,304 | 0.875 | 0.238911 | 0.238911 | yes |
| knot refit | 2,304 | 0.875 | 0.238911 | 0.238911 | yes |
| covariance shape 1/2 | 2,304 | 0.875 | 0.238911 | 0.238911 | yes |
| covariance shape 2/2 | 2,304 | 0.875 | 0.238911 | 0.238911 | yes |
| post-asinh transform 1/4 | 2,304 | 0.875 | 0.238911 | 0.238911 | yes |
| post-asinh transform 2/4 | 2,304 | 0.875 | 0.238911 | 0.238911 | yes |
| post-asinh transform 3/4 | 2,304 | 0.875 | 0.238911 | 0.238911 | yes |
| post-asinh transform 4/4 | 2,304 | 0.875 | 0.238911 | 0.238911 | yes |
| allocation beta | 2,304 | 1 | 0.235232 | 0.235232 | yes |
| knot refit | 2,304 | 1 | 0.233991 | 0.233991 | yes |
| covariance shape 1/2 | 2,304 | 1 | 0.233991 | 0.233991 | yes |
| covariance shape 2/2 | 2,304 | 1 | 0.233991 | 0.233991 | yes |
| post-asinh transform 1/4 | 2,304 | 1 | 0.233991 | 0.233991 | yes |
| post-asinh transform 2/4 | 2,304 | 1 | 0.233991 | 0.233991 | yes |
| post-asinh transform 3/4 | 2,304 | 1 | 0.233991 | 0.233991 | yes |
| post-asinh transform 4/4 | 2,304 | 1 | 0.233991 | 0.233991 | yes |
| allocation beta | 2,304 | 1 | 0.233991 | 0.233991 | yes |
| knot refit | 2,304 | 1 | 0.232594 | 0.232594 | yes |
| covariance shape 1/2 | 2,304 | 1 | 0.232594 | 0.232594 | yes |
| covariance shape 2/2 | 2,304 | 1 | 0.232594 | 0.232594 | yes |
| post-asinh transform 1/4 | 2,304 | 1 | 0.232594 | 0.232594 | yes |
| post-asinh transform 2/4 | 2,304 | 1 | 0.232594 | 0.232594 | yes |
| post-asinh transform 3/4 | 2,304 | 1 | 0.232594 | 0.232594 | yes |
| post-asinh transform 4/4 | 2,304 | 1 | 0.232594 | 0.232594 | yes |
| allocation beta | 2,304 | 1 | 0.232594 | 0.232594 | yes |
| knot refit | 2,304 | 1 | 0.232594 | 0.232594 | yes |
| refined parameter-search resolution | 2,304 | 1 | 0.232594 | 0.232594 | yes |
| covariance shape 1/2 | 2,304 | 1 | 0.232594 | 0.232594 | yes |
| covariance shape 2/2 | 2,304 | 1 | 0.232594 | 0.232594 | yes |
| post-asinh transform 1/4 | 2,304 | 1 | 0.232594 | 0.232594 | yes |
| post-asinh transform 2/4 | 2,304 | 1 | 0.232594 | 0.232594 | yes |
| post-asinh transform 3/4 | 2,304 | 1 | 0.232594 | 0.232594 | yes |
| post-asinh transform 4/4 | 2,304 | 1 | 0.232594 | 0.232594 | yes |
| allocation beta | 2,304 | 1 | 0.232594 | 0.232594 | yes |
| knot refit (cached stationary) | 2,304 | 1 | 0.232594 | 0.232594 | yes |
| refined parameter-search resolution | 2,304 | 1 | 0.232594 | 0.232594 | yes |
| covariance shape 1/2 | 2,304 | 1 | 0.232594 | 0.232594 | yes |
| covariance shape 2/2 | 2,304 | 1 | 0.232594 | 0.232594 | yes |
| post-asinh transform 1/4 | 2,304 | 1 | 0.232594 | 0.232594 | yes |
| post-asinh transform 2/4 | 2,304 | 1 | 0.232594 | 0.232594 | yes |
| post-asinh transform 3/4 | 2,304 | 1 | 0.232594 | 0.232594 | yes |
| post-asinh transform 4/4 | 2,304 | 1 | 0.232594 | 0.232594 | yes |
| allocation beta | 2,304 | 1 | 0.232594 | 0.232594 | yes |
| knot refit (cached stationary) | 2,304 | 1 | 0.232594 | 0.232594 | yes |
| fixed point reached | 2,304 | 1 | 0.232594 | 0.232594 | yes |
| prune proposal 3072->2304 | 2,304 | 1 | 0.232594 | 0.232594 | yes |
| covariance shape 1/2 | 2,304 | 1 | 1.92972 | 1.92972 | no |
| covariance shape 2/2 | 2,304 | 1 | 1.92972 | 1.92972 | no |
| post-asinh transform 1/4 | 2,304 | 1 | 1.92972 | 1.92972 | no |
| post-asinh transform 2/4 | 2,304 | 1 | 1.92972 | 1.92972 | no |
| post-asinh transform 3/4 | 2,304 | 1 | 1.92972 | 1.92972 | no |
| post-asinh transform 4/4 | 2,304 | 1 | 1.92972 | 1.92972 | no |
| allocation beta | 2,304 | 1 | 1.92972 | 1.92972 | no |
| knot refit | 2,304 | 1 | 1.92972 | 1.92972 | no |
| refined parameter-search resolution | 2,304 | 1 | 1.92972 | 1.92972 | no |
| covariance shape 1/2 | 2,304 | 1 | 1.92972 | 1.92972 | no |
| covariance shape 2/2 | 2,304 | 1 | 1.92972 | 1.92972 | no |
| post-asinh transform 1/4 | 2,304 | 1 | 1.92972 | 1.92972 | no |
| post-asinh transform 2/4 | 2,304 | 1 | 1.92972 | 1.92972 | no |
| post-asinh transform 3/4 | 2,304 | 1 | 1.92972 | 1.92972 | no |
| post-asinh transform 4/4 | 2,304 | 1 | 1.92972 | 1.92972 | no |
| allocation beta | 2,304 | 1 | 1.92972 | 1.92972 | no |
| knot refit (cached stationary) | 2,304 | 1 | 1.92972 | 1.92972 | no |
| refined parameter-search resolution | 2,304 | 1 | 1.92972 | 1.92972 | no |
| covariance shape 1/2 | 2,304 | 1 | 1.92972 | 1.92972 | no |
| covariance shape 2/2 | 2,304 | 1 | 1.92972 | 1.92972 | no |
| post-asinh transform 1/4 | 2,304 | 1 | 1.92972 | 1.92972 | no |
| post-asinh transform 2/4 | 2,304 | 1 | 1.92972 | 1.92972 | no |
| post-asinh transform 3/4 | 2,304 | 1 | 1.92972 | 1.92972 | no |
| post-asinh transform 4/4 | 2,304 | 1 | 1.92972 | 1.92972 | no |
| allocation beta | 2,304 | 1 | 1.92972 | 1.92972 | no |
| knot refit (cached stationary) | 2,304 | 1 | 1.92972 | 1.92972 | no |
| fixed point reached | 2,304 | 1 | 1.92972 | 1.92972 | no |
| accepted prune | 2,304 | 1 | 0.232594 | 0.232594 | yes |
| prune proposal 2304->1728 | 1,728 | 1 | 0.498471 | 0.498471 | yes |
| covariance shape 1/2 | 1,728 | 1 | 1.83988 | 1.83988 | no |
| covariance shape 2/2 | 1,728 | 1 | 1.83988 | 1.83988 | no |
| post-asinh transform 1/4 | 1,728 | 1 | 1.83988 | 1.83988 | no |
| post-asinh transform 2/4 | 1,728 | 1 | 1.83988 | 1.83988 | no |
| post-asinh transform 3/4 | 1,728 | 1 | 1.83988 | 1.83988 | no |
| post-asinh transform 4/4 | 1,728 | 1 | 1.83988 | 1.83988 | no |
| allocation beta | 1,728 | 0.875 | 0.702156 | 0.702156 | yes |
| knot refit | 1,728 | 0.875 | 0.53356 | 0.53356 | yes |
| covariance shape 1/2 | 1,728 | 0.875 | 0.53356 | 0.53356 | yes |
| covariance shape 2/2 | 1,728 | 0.875 | 0.53356 | 0.53356 | yes |
| post-asinh transform 1/4 | 1,728 | 0.875 | 0.53356 | 0.53356 | yes |
| post-asinh transform 2/4 | 1,728 | 0.875 | 0.53356 | 0.53356 | yes |
| post-asinh transform 3/4 | 1,728 | 0.875 | 0.53356 | 0.53356 | yes |
| post-asinh transform 4/4 | 1,728 | 0.875 | 0.53356 | 0.53356 | yes |
| allocation beta | 1,728 | 1 | 0.514861 | 0.514861 | yes |
| knot refit | 1,728 | 1 | 0.506929 | 0.506929 | yes |
| covariance shape 1/2 | 1,728 | 1 | 0.506929 | 0.506929 | yes |
| covariance shape 2/2 | 1,728 | 1 | 0.506929 | 0.506929 | yes |
| post-asinh transform 1/4 | 1,728 | 1 | 0.506929 | 0.506929 | yes |
| post-asinh transform 2/4 | 1,728 | 1 | 0.506929 | 0.506929 | yes |
| post-asinh transform 3/4 | 1,728 | 1 | 0.506929 | 0.506929 | yes |
| post-asinh transform 4/4 | 1,728 | 1 | 0.506929 | 0.506929 | yes |
| allocation beta | 1,728 | 0.875 | 0.505743 | 0.505743 | yes |
| knot refit | 1,728 | 0.875 | 0.501905 | 0.501905 | yes |
| covariance shape 1/2 | 1,728 | 0.875 | 0.501905 | 0.501905 | yes |
| covariance shape 2/2 | 1,728 | 0.875 | 0.501905 | 0.501905 | yes |
| post-asinh transform 1/4 | 1,728 | 0.875 | 0.501905 | 0.501905 | yes |
| post-asinh transform 2/4 | 1,728 | 0.875 | 0.501905 | 0.501905 | yes |
| post-asinh transform 3/4 | 1,728 | 0.875 | 0.501905 | 0.501905 | yes |
| post-asinh transform 4/4 | 1,728 | 0.875 | 0.501905 | 0.501905 | yes |
| allocation beta | 1,728 | 1 | 0.499113 | 0.499113 | yes |
| knot refit | 1,728 | 1 | 0.498471 | 0.498471 | yes |
| covariance shape 1/2 | 1,728 | 1 | 0.498471 | 0.498471 | yes |
| covariance shape 2/2 | 1,728 | 1 | 0.498471 | 0.498471 | yes |
| post-asinh transform 1/4 | 1,728 | 1 | 0.498471 | 0.498471 | yes |
| post-asinh transform 2/4 | 1,728 | 1 | 0.498471 | 0.498471 | yes |
| post-asinh transform 3/4 | 1,728 | 1 | 0.498471 | 0.498471 | yes |
| post-asinh transform 4/4 | 1,728 | 1 | 0.498471 | 0.498471 | yes |
| allocation beta | 1,728 | 1 | 0.498471 | 0.498471 | yes |
| knot refit | 1,728 | 1 | 0.498471 | 0.498471 | yes |
| refined parameter-search resolution | 1,728 | 1 | 0.498471 | 0.498471 | yes |
| covariance shape 1/2 | 1,728 | 1 | 0.498471 | 0.498471 | yes |
| covariance shape 2/2 | 1,728 | 1 | 0.498471 | 0.498471 | yes |
| post-asinh transform 1/4 | 1,728 | 1 | 0.498471 | 0.498471 | yes |
| post-asinh transform 2/4 | 1,728 | 1 | 0.498471 | 0.498471 | yes |
| post-asinh transform 3/4 | 1,728 | 1 | 0.498471 | 0.498471 | yes |
| post-asinh transform 4/4 | 1,728 | 1 | 0.498471 | 0.498471 | yes |
| allocation beta | 1,728 | 1 | 0.498471 | 0.498471 | yes |
| knot refit (cached stationary) | 1,728 | 1 | 0.498471 | 0.498471 | yes |
| refined parameter-search resolution | 1,728 | 1 | 0.498471 | 0.498471 | yes |
| covariance shape 1/2 | 1,728 | 1 | 0.498471 | 0.498471 | yes |
| covariance shape 2/2 | 1,728 | 1 | 0.498471 | 0.498471 | yes |
| post-asinh transform 1/4 | 1,728 | 1 | 0.498471 | 0.498471 | yes |
| post-asinh transform 2/4 | 1,728 | 1 | 0.498471 | 0.498471 | yes |
| post-asinh transform 3/4 | 1,728 | 1 | 0.498471 | 0.498471 | yes |
| post-asinh transform 4/4 | 1,728 | 1 | 0.498471 | 0.498471 | yes |
| allocation beta | 1,728 | 1 | 0.498471 | 0.498471 | yes |
| knot refit (cached stationary) | 1,728 | 1 | 0.498471 | 0.498471 | yes |
| fixed point reached | 1,728 | 1 | 0.498471 | 0.498471 | yes |
| accepted prune | 1,728 | 1 | 0.498471 | 0.498471 | yes |
| prune proposal 1728->1296 | 1,296 | 1 | 0.618645 | 0.618645 | yes |
| covariance shape 1/2 | 1,296 | 1 | 0.675028 | 0.675028 | yes |
| covariance shape 2/2 | 1,296 | 1 | 0.675028 | 0.675028 | yes |
| post-asinh transform 1/4 | 1,296 | 1 | 0.675028 | 0.675028 | yes |
| post-asinh transform 2/4 | 1,296 | 1 | 0.675028 | 0.675028 | yes |
| post-asinh transform 3/4 | 1,296 | 1 | 0.675028 | 0.675028 | yes |
| post-asinh transform 4/4 | 1,296 | 1 | 0.675028 | 0.675028 | yes |
| allocation beta | 1,296 | 0.875 | 0.626647 | 0.626647 | yes |
| knot refit | 1,296 | 0.875 | 0.626647 | 0.626647 | yes |
| covariance shape 1/2 | 1,296 | 0.875 | 0.626647 | 0.626647 | yes |
| covariance shape 2/2 | 1,296 | 0.875 | 0.626647 | 0.626647 | yes |
| post-asinh transform 1/4 | 1,296 | 0.875 | 0.626647 | 0.626647 | yes |
| post-asinh transform 2/4 | 1,296 | 0.875 | 0.626647 | 0.626647 | yes |
| post-asinh transform 3/4 | 1,296 | 0.875 | 0.626647 | 0.626647 | yes |
| post-asinh transform 4/4 | 1,296 | 0.875 | 0.626647 | 0.626647 | yes |
| allocation beta | 1,296 | 1 | 0.618645 | 0.618645 | yes |
| knot refit | 1,296 | 1 | 0.618645 | 0.618645 | yes |
| covariance shape 1/2 | 1,296 | 1 | 0.618645 | 0.618645 | yes |
| covariance shape 2/2 | 1,296 | 1 | 0.618645 | 0.618645 | yes |
| post-asinh transform 1/4 | 1,296 | 1 | 0.618645 | 0.618645 | yes |
| post-asinh transform 2/4 | 1,296 | 1 | 0.618645 | 0.618645 | yes |
| post-asinh transform 3/4 | 1,296 | 1 | 0.618645 | 0.618645 | yes |
| post-asinh transform 4/4 | 1,296 | 1 | 0.618645 | 0.618645 | yes |
| allocation beta | 1,296 | 1 | 0.618645 | 0.618645 | yes |
| knot refit (cached stationary) | 1,296 | 1 | 0.618645 | 0.618645 | yes |
| refined parameter-search resolution | 1,296 | 1 | 0.618645 | 0.618645 | yes |
| covariance shape 1/2 | 1,296 | 1 | 0.618645 | 0.618645 | yes |
| covariance shape 2/2 | 1,296 | 1 | 0.618645 | 0.618645 | yes |
| post-asinh transform 1/4 | 1,296 | 1 | 0.618645 | 0.618645 | yes |
| post-asinh transform 2/4 | 1,296 | 1 | 0.618645 | 0.618645 | yes |
| post-asinh transform 3/4 | 1,296 | 1 | 0.618645 | 0.618645 | yes |
| post-asinh transform 4/4 | 1,296 | 1 | 0.618645 | 0.618645 | yes |
| allocation beta | 1,296 | 1 | 0.618645 | 0.618645 | yes |
| knot refit (cached stationary) | 1,296 | 1 | 0.618645 | 0.618645 | yes |
| refined parameter-search resolution | 1,296 | 1 | 0.618645 | 0.618645 | yes |
| covariance shape 1/2 | 1,296 | 1 | 0.618645 | 0.618645 | yes |
| covariance shape 2/2 | 1,296 | 1 | 0.618645 | 0.618645 | yes |
| post-asinh transform 1/4 | 1,296 | 1 | 0.618645 | 0.618645 | yes |
| post-asinh transform 2/4 | 1,296 | 1 | 0.618645 | 0.618645 | yes |
| post-asinh transform 3/4 | 1,296 | 1 | 0.618645 | 0.618645 | yes |
| post-asinh transform 4/4 | 1,296 | 1 | 0.618645 | 0.618645 | yes |
| allocation beta | 1,296 | 1 | 0.618645 | 0.618645 | yes |
| knot refit (cached stationary) | 1,296 | 1 | 0.618645 | 0.618645 | yes |
| fixed point reached | 1,296 | 1 | 0.618645 | 0.618645 | yes |
| accepted prune | 1,296 | 1 | 0.618645 | 0.618645 | yes |
| prune proposal 1296->972 | 972 | 1 | 0.798675 | 0.798675 | yes |
| covariance shape 1/2 | 972 | 1 | 0.839327 | 0.839327 | yes |
| covariance shape 2/2 | 972 | 1 | 0.839327 | 0.839327 | yes |
| post-asinh transform 1/4 | 972 | 1 | 0.839327 | 0.839327 | yes |
| post-asinh transform 2/4 | 972 | 1 | 0.839327 | 0.839327 | yes |
| post-asinh transform 3/4 | 972 | 1 | 0.839327 | 0.839327 | yes |
| post-asinh transform 4/4 | 972 | 1 | 0.839327 | 0.839327 | yes |
| allocation beta | 972 | 1 | 0.839327 | 0.839327 | yes |
| knot refit | 972 | 1 | 0.805806 | 0.805806 | yes |
| covariance shape 1/2 | 972 | 1 | 0.805806 | 0.805806 | yes |
| covariance shape 2/2 | 972 | 1 | 0.805806 | 0.805806 | yes |
| post-asinh transform 1/4 | 972 | 1 | 0.805806 | 0.805806 | yes |
| post-asinh transform 2/4 | 972 | 1 | 0.805806 | 0.805806 | yes |
| post-asinh transform 3/4 | 972 | 1 | 0.805806 | 0.805806 | yes |
| post-asinh transform 4/4 | 972 | 1 | 0.805806 | 0.805806 | yes |
| allocation beta | 972 | 1 | 0.805806 | 0.805806 | yes |
| knot refit | 972 | 1 | 0.798675 | 0.798675 | yes |
| covariance shape 1/2 | 972 | 1 | 0.798675 | 0.798675 | yes |
| covariance shape 2/2 | 972 | 1 | 0.798675 | 0.798675 | yes |
| post-asinh transform 1/4 | 972 | 1 | 0.798675 | 0.798675 | yes |
| post-asinh transform 2/4 | 972 | 1 | 0.798675 | 0.798675 | yes |
| post-asinh transform 3/4 | 972 | 1 | 0.798675 | 0.798675 | yes |
| post-asinh transform 4/4 | 972 | 1 | 0.798675 | 0.798675 | yes |
| allocation beta | 972 | 1 | 0.798675 | 0.798675 | yes |
| knot refit | 972 | 1 | 0.798675 | 0.798675 | yes |
| refined parameter-search resolution | 972 | 1 | 0.798675 | 0.798675 | yes |
| covariance shape 1/2 | 972 | 1 | 0.798675 | 0.798675 | yes |
| covariance shape 2/2 | 972 | 1 | 0.798675 | 0.798675 | yes |
| post-asinh transform 1/4 | 972 | 1 | 0.798675 | 0.798675 | yes |
| post-asinh transform 2/4 | 972 | 1 | 0.798675 | 0.798675 | yes |
| post-asinh transform 3/4 | 972 | 1 | 0.798675 | 0.798675 | yes |
| post-asinh transform 4/4 | 972 | 1 | 0.798675 | 0.798675 | yes |
| allocation beta | 972 | 1 | 0.798675 | 0.798675 | yes |
| knot refit (cached stationary) | 972 | 1 | 0.798675 | 0.798675 | yes |
| refined parameter-search resolution | 972 | 1 | 0.798675 | 0.798675 | yes |
| covariance shape 1/2 | 972 | 1 | 0.798675 | 0.798675 | yes |
| covariance shape 2/2 | 972 | 1 | 0.798675 | 0.798675 | yes |
| post-asinh transform 1/4 | 972 | 1 | 0.798675 | 0.798675 | yes |
| post-asinh transform 2/4 | 972 | 1 | 0.798675 | 0.798675 | yes |
| post-asinh transform 3/4 | 972 | 1 | 0.798675 | 0.798675 | yes |
| post-asinh transform 4/4 | 972 | 1 | 0.798675 | 0.798675 | yes |
| allocation beta | 972 | 1 | 0.798675 | 0.798675 | yes |
| knot refit (cached stationary) | 972 | 1 | 0.798675 | 0.798675 | yes |
| fixed point reached | 972 | 1 | 0.798675 | 0.798675 | yes |
| accepted prune | 972 | 1 | 0.798675 | 0.798675 | yes |
| cold restart 972->729 | 729 | 1 | 0.749058 | 0.749058 | yes |
| covariance shape 1/2 | 729 | 1 | 0.751293 | 0.751293 | yes |
| covariance shape 2/2 | 729 | 1 | 0.751293 | 0.751293 | yes |
| post-asinh transform 1/4 | 729 | 1 | 0.751293 | 0.751293 | yes |
| post-asinh transform 2/4 | 729 | 1 | 0.751293 | 0.751293 | yes |
| post-asinh transform 3/4 | 729 | 1 | 0.751293 | 0.751293 | yes |
| post-asinh transform 4/4 | 729 | 1 | 0.751293 | 0.751293 | yes |
| allocation beta | 729 | 1 | 0.751293 | 0.751293 | yes |
| knot refit | 729 | 1 | 0.749058 | 0.749058 | yes |
| covariance shape 1/2 | 729 | 1 | 0.749058 | 0.749058 | yes |
| covariance shape 2/2 | 729 | 1 | 0.749058 | 0.749058 | yes |
| post-asinh transform 1/4 | 729 | 1 | 0.749058 | 0.749058 | yes |
| post-asinh transform 2/4 | 729 | 1 | 0.749058 | 0.749058 | yes |
| post-asinh transform 3/4 | 729 | 1 | 0.749058 | 0.749058 | yes |
| post-asinh transform 4/4 | 729 | 1 | 0.749058 | 0.749058 | yes |
| allocation beta | 729 | 1 | 0.749058 | 0.749058 | yes |
| knot refit | 729 | 1 | 0.749058 | 0.749058 | yes |
| refined parameter-search resolution | 729 | 1 | 0.749058 | 0.749058 | yes |
| covariance shape 1/2 | 729 | 1 | 0.749058 | 0.749058 | yes |
| covariance shape 2/2 | 729 | 1 | 0.749058 | 0.749058 | yes |
| post-asinh transform 1/4 | 729 | 1 | 0.749058 | 0.749058 | yes |
| post-asinh transform 2/4 | 729 | 1 | 0.749058 | 0.749058 | yes |
| post-asinh transform 3/4 | 729 | 1 | 0.749058 | 0.749058 | yes |
| post-asinh transform 4/4 | 729 | 1 | 0.749058 | 0.749058 | yes |
| allocation beta | 729 | 1 | 0.749058 | 0.749058 | yes |
| knot refit (cached stationary) | 729 | 1 | 0.749058 | 0.749058 | yes |
| refined parameter-search resolution | 729 | 1 | 0.749058 | 0.749058 | yes |
| covariance shape 1/2 | 729 | 1 | 0.749058 | 0.749058 | yes |
| covariance shape 2/2 | 729 | 1 | 0.749058 | 0.749058 | yes |
| post-asinh transform 1/4 | 729 | 1 | 0.749058 | 0.749058 | yes |
| post-asinh transform 2/4 | 729 | 1 | 0.749058 | 0.749058 | yes |
| post-asinh transform 3/4 | 729 | 1 | 0.749058 | 0.749058 | yes |
| post-asinh transform 4/4 | 729 | 1 | 0.749058 | 0.749058 | yes |
| allocation beta | 729 | 1 | 0.749058 | 0.749058 | yes |
| knot refit (cached stationary) | 729 | 1 | 0.749058 | 0.749058 | yes |
| fixed point reached | 729 | 1 | 0.749058 | 0.749058 | yes |
| prune proposal 972->729 | 729 | 1 | 0.749058 | 0.749058 | yes |
| covariance shape 1/2 | 729 | 1 | 1.23182 | 1.23182 | no |
| covariance shape 2/2 | 729 | 1 | 1.23182 | 1.23182 | no |
| post-asinh transform 1/4 | 729 | 1 | 1.23182 | 1.23182 | no |
| post-asinh transform 2/4 | 729 | 1 | 1.23182 | 1.23182 | no |
| post-asinh transform 3/4 | 729 | 1 | 1.23182 | 1.23182 | no |
| post-asinh transform 4/4 | 729 | 1 | 1.23182 | 1.23182 | no |
| allocation beta | 729 | 0.875 | 1.20833 | 1.20833 | no |
| knot refit | 729 | 0.875 | 1.17866 | 1.17866 | no |
| covariance shape 1/2 | 729 | 0.875 | 1.17866 | 1.17866 | no |
| covariance shape 2/2 | 729 | 0.875 | 1.17866 | 1.17866 | no |
| post-asinh transform 1/4 | 729 | 0.875 | 1.17866 | 1.17866 | no |
| post-asinh transform 2/4 | 729 | 0.875 | 1.17866 | 1.17866 | no |
| post-asinh transform 3/4 | 729 | 0.875 | 1.17866 | 1.17866 | no |
| post-asinh transform 4/4 | 729 | 0.875 | 1.17866 | 1.17866 | no |
| allocation beta | 729 | 1 | 1.13617 | 1.13617 | no |
| knot refit | 729 | 1 | 1.10586 | 1.10586 | no |
| covariance shape 1/2 | 729 | 1 | 1.10586 | 1.10586 | no |
| covariance shape 2/2 | 729 | 1 | 1.10586 | 1.10586 | no |
| post-asinh transform 1/4 | 729 | 1 | 1.10586 | 1.10586 | no |
| post-asinh transform 2/4 | 729 | 1 | 1.10586 | 1.10586 | no |
| post-asinh transform 3/4 | 729 | 1 | 1.10586 | 1.10586 | no |
| post-asinh transform 4/4 | 729 | 1 | 1.10586 | 1.10586 | no |
| allocation beta | 729 | 1 | 1.10586 | 1.10586 | no |
| knot refit | 729 | 1 | 1.09587 | 1.09587 | no |
| covariance shape 1/2 | 729 | 1 | 1.09587 | 1.09587 | no |
| covariance shape 2/2 | 729 | 1 | 1.09587 | 1.09587 | no |
| post-asinh transform 1/4 | 729 | 1 | 1.09587 | 1.09587 | no |
| post-asinh transform 2/4 | 729 | 1 | 1.09587 | 1.09587 | no |
| post-asinh transform 3/4 | 729 | 1 | 1.09587 | 1.09587 | no |
| post-asinh transform 4/4 | 729 | 1 | 1.09587 | 1.09587 | no |
| allocation beta | 729 | 1 | 1.09587 | 1.09587 | no |
| knot refit | 729 | 1 | 1.08355 | 1.08355 | no |
| covariance shape 1/2 | 729 | 1 | 1.08355 | 1.08355 | no |
| covariance shape 2/2 | 729 | 1 | 1.08355 | 1.08355 | no |
| post-asinh transform 1/4 | 729 | 1 | 1.08355 | 1.08355 | no |
| post-asinh transform 2/4 | 729 | 1 | 1.08355 | 1.08355 | no |
| post-asinh transform 3/4 | 729 | 1 | 1.08355 | 1.08355 | no |
| post-asinh transform 4/4 | 729 | 1 | 1.08355 | 1.08355 | no |
| allocation beta | 729 | 1 | 1.08355 | 1.08355 | no |
| knot refit | 729 | 1 | 1.07927 | 1.07927 | no |
| covariance shape 1/2 | 729 | 1 | 1.07927 | 1.07927 | no |
| covariance shape 2/2 | 729 | 1 | 1.07927 | 1.07927 | no |
| post-asinh transform 1/4 | 729 | 1 | 1.07927 | 1.07927 | no |
| post-asinh transform 2/4 | 729 | 1 | 1.07927 | 1.07927 | no |
| post-asinh transform 3/4 | 729 | 1 | 1.07927 | 1.07927 | no |
| post-asinh transform 4/4 | 729 | 1 | 1.07927 | 1.07927 | no |
| allocation beta | 729 | 1 | 1.07927 | 1.07927 | no |
| knot refit | 729 | 1 | 1.07569 | 1.07569 | no |
| covariance shape 1/2 | 729 | 1 | 1.07569 | 1.07569 | no |
| covariance shape 2/2 | 729 | 1 | 1.07569 | 1.07569 | no |
| post-asinh transform 1/4 | 729 | 1 | 1.07569 | 1.07569 | no |
| post-asinh transform 2/4 | 729 | 1 | 1.07569 | 1.07569 | no |
| post-asinh transform 3/4 | 729 | 1 | 1.07569 | 1.07569 | no |
| post-asinh transform 4/4 | 729 | 1 | 1.07569 | 1.07569 | no |
| allocation beta | 729 | 1 | 1.07569 | 1.07569 | no |
| knot refit | 729 | 1 | 1.07205 | 1.07205 | no |
| covariance shape 1/2 | 729 | 1 | 1.07205 | 1.07205 | no |
| covariance shape 2/2 | 729 | 1 | 1.07205 | 1.07205 | no |
| post-asinh transform 1/4 | 729 | 1 | 1.07205 | 1.07205 | no |
| post-asinh transform 2/4 | 729 | 1 | 1.07205 | 1.07205 | no |
| post-asinh transform 3/4 | 729 | 1 | 1.07205 | 1.07205 | no |
| post-asinh transform 4/4 | 729 | 1 | 1.07205 | 1.07205 | no |
| allocation beta | 729 | 1 | 1.07205 | 1.07205 | no |
| knot refit | 729 | 1 | 1.06849 | 1.06849 | no |
| covariance shape 1/2 | 729 | 1 | 1.06849 | 1.06849 | no |
| covariance shape 2/2 | 729 | 1 | 1.06849 | 1.06849 | no |
| post-asinh transform 1/4 | 729 | 1 | 1.06849 | 1.06849 | no |
| post-asinh transform 2/4 | 729 | 1 | 1.06849 | 1.06849 | no |
| post-asinh transform 3/4 | 729 | 1 | 1.06849 | 1.06849 | no |
| post-asinh transform 4/4 | 729 | 1 | 1.06849 | 1.06849 | no |
| allocation beta | 729 | 1 | 1.06849 | 1.06849 | no |
| knot refit | 729 | 1 | 1.06217 | 1.06217 | no |
| covariance shape 1/2 | 729 | 1 | 1.06217 | 1.06217 | no |
| covariance shape 2/2 | 729 | 1 | 1.06217 | 1.06217 | no |
| post-asinh transform 1/4 | 729 | 1 | 1.06217 | 1.06217 | no |
| post-asinh transform 2/4 | 729 | 1 | 1.06217 | 1.06217 | no |
| post-asinh transform 3/4 | 729 | 1 | 1.06217 | 1.06217 | no |
| post-asinh transform 4/4 | 729 | 1 | 1.06217 | 1.06217 | no |
| allocation beta | 729 | 1 | 1.06217 | 1.06217 | no |
| knot refit | 729 | 1 | 1.06217 | 1.06217 | no |
| refined parameter-search resolution | 729 | 1 | 1.06217 | 1.06217 | no |
| covariance shape 1/2 | 729 | 1 | 1.06217 | 1.06217 | no |
| covariance shape 2/2 | 729 | 1 | 1.06217 | 1.06217 | no |
| post-asinh transform 1/4 | 729 | 1 | 1.06217 | 1.06217 | no |
| post-asinh transform 2/4 | 729 | 1 | 1.06217 | 1.06217 | no |
| post-asinh transform 3/4 | 729 | 1 | 1.06217 | 1.06217 | no |
| post-asinh transform 4/4 | 729 | 1 | 1.06217 | 1.06217 | no |
| allocation beta | 729 | 1 | 1.06217 | 1.06217 | no |
| knot refit (cached stationary) | 729 | 1 | 1.06217 | 1.06217 | no |
| refined parameter-search resolution | 729 | 1 | 1.06217 | 1.06217 | no |
| covariance shape 1/2 | 729 | 1 | 1.06217 | 1.06217 | no |
| covariance shape 2/2 | 729 | 1 | 1.06217 | 1.06217 | no |
| post-asinh transform 1/4 | 729 | 1 | 1.06217 | 1.06217 | no |
| post-asinh transform 2/4 | 729 | 1 | 1.06217 | 1.06217 | no |
| post-asinh transform 3/4 | 729 | 1 | 1.06217 | 1.06217 | no |
| post-asinh transform 4/4 | 729 | 1 | 1.06217 | 1.06217 | no |
| allocation beta | 729 | 1 | 1.06217 | 1.06217 | no |
| knot refit (cached stationary) | 729 | 1 | 1.06217 | 1.06217 | no |
| fixed point reached | 729 | 1 | 1.06217 | 1.06217 | no |
| accepted prune | 729 | 1 | 0.749058 | 0.749058 | yes |
| cold restart 729->547 | 547 | 1 | 1.00784 | 1.00784 | no |
| covariance shape 1/2 | 547 | 1 | 1.02825 | 1.02825 | no |
| covariance shape 2/2 | 547 | 1 | 1.02825 | 1.02825 | no |
| post-asinh transform 1/4 | 547 | 1 | 1.02825 | 1.02825 | no |
| post-asinh transform 2/4 | 547 | 1 | 1.02825 | 1.02825 | no |
| post-asinh transform 3/4 | 547 | 1 | 1.02825 | 1.02825 | no |
| post-asinh transform 4/4 | 547 | 1 | 1.02825 | 1.02825 | no |
| allocation beta | 547 | 1 | 1.02825 | 1.02825 | no |
| knot refit | 547 | 1 | 1.02649 | 1.02649 | no |
| covariance shape 1/2 | 547 | 1 | 1.02649 | 1.02649 | no |
| covariance shape 2/2 | 547 | 1 | 1.02649 | 1.02649 | no |
| post-asinh transform 1/4 | 547 | 1 | 1.02649 | 1.02649 | no |
| post-asinh transform 2/4 | 547 | 1 | 1.02649 | 1.02649 | no |
| post-asinh transform 3/4 | 547 | 1 | 1.02649 | 1.02649 | no |
| post-asinh transform 4/4 | 547 | 1 | 1.02649 | 1.02649 | no |
| allocation beta | 547 | 1 | 1.02649 | 1.02649 | no |
| knot refit | 547 | 1 | 1.01636 | 1.01636 | no |
| covariance shape 1/2 | 547 | 1 | 1.01636 | 1.01636 | no |
| covariance shape 2/2 | 547 | 1 | 1.01636 | 1.01636 | no |
| post-asinh transform 1/4 | 547 | 1 | 1.01636 | 1.01636 | no |
| post-asinh transform 2/4 | 547 | 1 | 1.01636 | 1.01636 | no |
| post-asinh transform 3/4 | 547 | 1 | 1.01636 | 1.01636 | no |
| post-asinh transform 4/4 | 547 | 1 | 1.01636 | 1.01636 | no |
| allocation beta | 547 | 1 | 1.01636 | 1.01636 | no |
| knot refit | 547 | 1 | 1.01039 | 1.01039 | no |
| covariance shape 1/2 | 547 | 1 | 1.01039 | 1.01039 | no |
| covariance shape 2/2 | 547 | 1 | 1.01039 | 1.01039 | no |
| post-asinh transform 1/4 | 547 | 1 | 1.01039 | 1.01039 | no |
| post-asinh transform 2/4 | 547 | 1 | 1.01039 | 1.01039 | no |
| post-asinh transform 3/4 | 547 | 1 | 1.01039 | 1.01039 | no |
| post-asinh transform 4/4 | 547 | 1 | 1.01039 | 1.01039 | no |
| allocation beta | 547 | 1 | 1.01039 | 1.01039 | no |
| knot refit | 547 | 1 | 1.00784 | 1.00784 | no |
| covariance shape 1/2 | 547 | 1 | 1.00784 | 1.00784 | no |
| covariance shape 2/2 | 547 | 1 | 1.00784 | 1.00784 | no |
| post-asinh transform 1/4 | 547 | 1 | 1.00784 | 1.00784 | no |
| post-asinh transform 2/4 | 547 | 1 | 1.00784 | 1.00784 | no |
| post-asinh transform 3/4 | 547 | 1 | 1.00784 | 1.00784 | no |
| post-asinh transform 4/4 | 547 | 1 | 1.00784 | 1.00784 | no |
| allocation beta | 547 | 1 | 1.00784 | 1.00784 | no |
| knot refit | 547 | 1 | 1.00784 | 1.00784 | no |
| refined parameter-search resolution | 547 | 1 | 1.00784 | 1.00784 | no |
| covariance shape 1/2 | 547 | 1 | 1.00784 | 1.00784 | no |
| covariance shape 2/2 | 547 | 1 | 1.00784 | 1.00784 | no |
| post-asinh transform 1/4 | 547 | 1 | 1.00784 | 1.00784 | no |
| post-asinh transform 2/4 | 547 | 1 | 1.00784 | 1.00784 | no |
| post-asinh transform 3/4 | 547 | 1 | 1.00784 | 1.00784 | no |
| post-asinh transform 4/4 | 547 | 1 | 1.00784 | 1.00784 | no |
| allocation beta | 547 | 1 | 1.00784 | 1.00784 | no |
| knot refit (cached stationary) | 547 | 1 | 1.00784 | 1.00784 | no |
| refined parameter-search resolution | 547 | 1 | 1.00784 | 1.00784 | no |
| covariance shape 1/2 | 547 | 1 | 1.00784 | 1.00784 | no |
| covariance shape 2/2 | 547 | 1 | 1.00784 | 1.00784 | no |
| post-asinh transform 1/4 | 547 | 1 | 1.00784 | 1.00784 | no |
| post-asinh transform 2/4 | 547 | 1 | 1.00784 | 1.00784 | no |
| post-asinh transform 3/4 | 547 | 1 | 1.00784 | 1.00784 | no |
| post-asinh transform 4/4 | 547 | 1 | 1.00784 | 1.00784 | no |
| allocation beta | 547 | 1 | 1.00784 | 1.00784 | no |
| knot refit (cached stationary) | 547 | 1 | 1.00784 | 1.00784 | no |
| fixed point reached | 547 | 1 | 1.00784 | 1.00784 | no |
| prune proposal 729->547 | 547 | 1 | 1.00784 | 1.00784 | no |
| covariance shape 1/2 | 547 | 1 | 2.60324 | 2.60324 | no |
| covariance shape 2/2 | 547 | 1 | 2.60324 | 2.60324 | no |
| post-asinh transform 1/4 | 547 | 1 | 2.60324 | 2.60324 | no |
| post-asinh transform 2/4 | 547 | 1 | 2.60324 | 2.60324 | no |
| post-asinh transform 3/4 | 547 | 1 | 2.60324 | 2.60324 | no |
| post-asinh transform 4/4 | 547 | 1 | 2.60324 | 2.60324 | no |
| allocation beta | 547 | 0.875 | 2.47907 | 2.47907 | no |
| knot refit | 547 | 0.875 | 2.46923 | 2.46923 | no |
| covariance shape 1/2 | 547 | 0.875 | 2.46923 | 2.46923 | no |
| covariance shape 2/2 | 547 | 0.875 | 2.46923 | 2.46923 | no |
| post-asinh transform 1/4 | 547 | 0.875 | 2.46923 | 2.46923 | no |
| post-asinh transform 2/4 | 547 | 0.875 | 2.46923 | 2.46923 | no |
| post-asinh transform 3/4 | 547 | 0.875 | 2.46923 | 2.46923 | no |
| post-asinh transform 4/4 | 547 | 0.875 | 2.46923 | 2.46923 | no |
| allocation beta | 547 | 1 | 2.46657 | 2.46657 | no |
| knot refit | 547 | 1 | 2.46257 | 2.46257 | no |
| covariance shape 1/2 | 547 | 1 | 2.46257 | 2.46257 | no |
| covariance shape 2/2 | 547 | 1 | 2.46257 | 2.46257 | no |
| post-asinh transform 1/4 | 547 | 1 | 2.46257 | 2.46257 | no |
| post-asinh transform 2/4 | 547 | 1 | 2.46257 | 2.46257 | no |
| post-asinh transform 3/4 | 547 | 1 | 2.46257 | 2.46257 | no |
| post-asinh transform 4/4 | 547 | 1 | 2.46257 | 2.46257 | no |
| allocation beta | 547 | 1 | 2.46257 | 2.46257 | no |
| knot refit | 547 | 1 | 2.45844 | 2.45844 | no |
| covariance shape 1/2 | 547 | 1 | 2.45844 | 2.45844 | no |
| covariance shape 2/2 | 547 | 1 | 2.45844 | 2.45844 | no |
| post-asinh transform 1/4 | 547 | 1 | 2.45844 | 2.45844 | no |
| post-asinh transform 2/4 | 547 | 1 | 2.45844 | 2.45844 | no |
| post-asinh transform 3/4 | 547 | 1 | 2.45844 | 2.45844 | no |
| post-asinh transform 4/4 | 547 | 1 | 2.45844 | 2.45844 | no |
| allocation beta | 547 | 1 | 2.45844 | 2.45844 | no |
| knot refit | 547 | 1 | 2.45538 | 2.45538 | no |
| covariance shape 1/2 | 547 | 1 | 2.45538 | 2.45538 | no |
| covariance shape 2/2 | 547 | 1 | 2.45538 | 2.45538 | no |
| post-asinh transform 1/4 | 547 | 1 | 2.45538 | 2.45538 | no |
| post-asinh transform 2/4 | 547 | 1 | 2.45538 | 2.45538 | no |
| post-asinh transform 3/4 | 547 | 1 | 2.45538 | 2.45538 | no |
| post-asinh transform 4/4 | 547 | 1 | 2.45538 | 2.45538 | no |
| allocation beta | 547 | 1 | 2.45538 | 2.45538 | no |
| knot refit | 547 | 1 | 2.45538 | 2.45538 | no |
| refined parameter-search resolution | 547 | 1 | 2.45538 | 2.45538 | no |
| covariance shape 1/2 | 547 | 1 | 2.45538 | 2.45538 | no |
| covariance shape 2/2 | 547 | 1 | 2.45538 | 2.45538 | no |
| post-asinh transform 1/4 | 547 | 1 | 2.45538 | 2.45538 | no |
| post-asinh transform 2/4 | 547 | 1 | 2.45538 | 2.45538 | no |
| post-asinh transform 3/4 | 547 | 1 | 2.45538 | 2.45538 | no |
| post-asinh transform 4/4 | 547 | 1 | 2.45538 | 2.45538 | no |
| allocation beta | 547 | 1 | 2.45538 | 2.45538 | no |
| knot refit (cached stationary) | 547 | 1 | 2.45538 | 2.45538 | no |
| refined parameter-search resolution | 547 | 1 | 2.45538 | 2.45538 | no |
| covariance shape 1/2 | 547 | 1 | 2.45538 | 2.45538 | no |
| covariance shape 2/2 | 547 | 1 | 2.45538 | 2.45538 | no |
| post-asinh transform 1/4 | 547 | 1 | 2.45538 | 2.45538 | no |
| post-asinh transform 2/4 | 547 | 1 | 2.45538 | 2.45538 | no |
| post-asinh transform 3/4 | 547 | 1 | 2.45538 | 2.45538 | no |
| post-asinh transform 4/4 | 547 | 1 | 2.45538 | 2.45538 | no |
| allocation beta | 547 | 1 | 2.45538 | 2.45538 | no |
| knot refit (cached stationary) | 547 | 1 | 2.45538 | 2.45538 | no |
| fixed point reached | 547 | 1 | 2.45538 | 2.45538 | no |
| rejected prune | 547 | 1 | 1.00784 | 1.00784 | no |
| prune proposal 729->638 | 638 | 1 | 0.727593 | 0.727593 | yes |
| covariance shape 1/2 | 638 | 1 | 0.74484 | 0.74484 | yes |
| covariance shape 2/2 | 638 | 1 | 0.74484 | 0.74484 | yes |
| post-asinh transform 1/4 | 638 | 1 | 0.74484 | 0.74484 | yes |
| post-asinh transform 2/4 | 638 | 1 | 0.74484 | 0.74484 | yes |
| post-asinh transform 3/4 | 638 | 1 | 0.74484 | 0.74484 | yes |
| post-asinh transform 4/4 | 638 | 1 | 0.74484 | 0.74484 | yes |
| allocation beta | 638 | 1 | 0.74484 | 0.74484 | yes |
| knot refit | 638 | 1 | 0.742609 | 0.742609 | yes |
| covariance shape 1/2 | 638 | 1 | 0.742609 | 0.742609 | yes |
| covariance shape 2/2 | 638 | 1 | 0.742609 | 0.742609 | yes |
| post-asinh transform 1/4 | 638 | 1 | 0.742609 | 0.742609 | yes |
| post-asinh transform 2/4 | 638 | 1 | 0.742609 | 0.742609 | yes |
| post-asinh transform 3/4 | 638 | 1 | 0.742609 | 0.742609 | yes |
| post-asinh transform 4/4 | 638 | 1 | 0.742609 | 0.742609 | yes |
| allocation beta | 638 | 1 | 0.742609 | 0.742609 | yes |
| knot refit | 638 | 1 | 0.738534 | 0.738534 | yes |
| covariance shape 1/2 | 638 | 1 | 0.738534 | 0.738534 | yes |
| covariance shape 2/2 | 638 | 1 | 0.738534 | 0.738534 | yes |
| post-asinh transform 1/4 | 638 | 1 | 0.738534 | 0.738534 | yes |
| post-asinh transform 2/4 | 638 | 1 | 0.738534 | 0.738534 | yes |
| post-asinh transform 3/4 | 638 | 1 | 0.738534 | 0.738534 | yes |
| post-asinh transform 4/4 | 638 | 1 | 0.738534 | 0.738534 | yes |
| allocation beta | 638 | 1 | 0.738534 | 0.738534 | yes |
| knot refit | 638 | 1 | 0.735236 | 0.735236 | yes |
| covariance shape 1/2 | 638 | 1 | 0.735236 | 0.735236 | yes |
| covariance shape 2/2 | 638 | 1 | 0.735236 | 0.735236 | yes |
| post-asinh transform 1/4 | 638 | 1 | 0.735236 | 0.735236 | yes |
| post-asinh transform 2/4 | 638 | 1 | 0.735236 | 0.735236 | yes |
| post-asinh transform 3/4 | 638 | 1 | 0.735236 | 0.735236 | yes |
| post-asinh transform 4/4 | 638 | 1 | 0.735236 | 0.735236 | yes |
| allocation beta | 638 | 1 | 0.735236 | 0.735236 | yes |
| knot refit | 638 | 1 | 0.732998 | 0.732998 | yes |
| covariance shape 1/2 | 638 | 1 | 0.732998 | 0.732998 | yes |
| covariance shape 2/2 | 638 | 1 | 0.732998 | 0.732998 | yes |
| post-asinh transform 1/4 | 638 | 1 | 0.732998 | 0.732998 | yes |
| post-asinh transform 2/4 | 638 | 1 | 0.732998 | 0.732998 | yes |
| post-asinh transform 3/4 | 638 | 1 | 0.732998 | 0.732998 | yes |
| post-asinh transform 4/4 | 638 | 1 | 0.732998 | 0.732998 | yes |
| allocation beta | 638 | 1 | 0.732998 | 0.732998 | yes |
| knot refit | 638 | 1 | 0.727593 | 0.727593 | yes |
| covariance shape 1/2 | 638 | 1 | 0.727593 | 0.727593 | yes |
| covariance shape 2/2 | 638 | 1 | 0.727593 | 0.727593 | yes |
| post-asinh transform 1/4 | 638 | 1 | 0.727593 | 0.727593 | yes |
| post-asinh transform 2/4 | 638 | 1 | 0.727593 | 0.727593 | yes |
| post-asinh transform 3/4 | 638 | 1 | 0.727593 | 0.727593 | yes |
| post-asinh transform 4/4 | 638 | 1 | 0.727593 | 0.727593 | yes |
| allocation beta | 638 | 1 | 0.727593 | 0.727593 | yes |
| knot refit | 638 | 1 | 0.727593 | 0.727593 | yes |
| refined parameter-search resolution | 638 | 1 | 0.727593 | 0.727593 | yes |
| covariance shape 1/2 | 638 | 1 | 0.727593 | 0.727593 | yes |
| covariance shape 2/2 | 638 | 1 | 0.727593 | 0.727593 | yes |
| post-asinh transform 1/4 | 638 | 1 | 0.727593 | 0.727593 | yes |
| post-asinh transform 2/4 | 638 | 1 | 0.727593 | 0.727593 | yes |
| post-asinh transform 3/4 | 638 | 1 | 0.727593 | 0.727593 | yes |
| post-asinh transform 4/4 | 638 | 1 | 0.727593 | 0.727593 | yes |
| allocation beta | 638 | 1 | 0.727593 | 0.727593 | yes |
| knot refit (cached stationary) | 638 | 1 | 0.727593 | 0.727593 | yes |
| refined parameter-search resolution | 638 | 1 | 0.727593 | 0.727593 | yes |
| covariance shape 1/2 | 638 | 1 | 0.727593 | 0.727593 | yes |
| covariance shape 2/2 | 638 | 1 | 0.727593 | 0.727593 | yes |
| post-asinh transform 1/4 | 638 | 1 | 0.727593 | 0.727593 | yes |
| post-asinh transform 2/4 | 638 | 1 | 0.727593 | 0.727593 | yes |
| post-asinh transform 3/4 | 638 | 1 | 0.727593 | 0.727593 | yes |
| post-asinh transform 4/4 | 638 | 1 | 0.727593 | 0.727593 | yes |
| allocation beta | 638 | 1 | 0.727593 | 0.727593 | yes |
| knot refit (cached stationary) | 638 | 1 | 0.727593 | 0.727593 | yes |
| fixed point reached | 638 | 1 | 0.727593 | 0.727593 | yes |
| accepted prune | 638 | 1 | 0.727593 | 0.727593 | yes |
| cold restart 638->547 | 547 | 1 | 0.95197 | 0.95197 | yes |
| covariance shape 1/2 | 547 | 1 | 0.984498 | 0.984498 | yes |
| covariance shape 2/2 | 547 | 1 | 0.984498 | 0.984498 | yes |
| post-asinh transform 1/4 | 547 | 1 | 0.984498 | 0.984498 | yes |
| post-asinh transform 2/4 | 547 | 1 | 0.984498 | 0.984498 | yes |
| post-asinh transform 3/4 | 547 | 1 | 0.984498 | 0.984498 | yes |
| post-asinh transform 4/4 | 547 | 1 | 0.984498 | 0.984498 | yes |
| allocation beta | 547 | 1 | 0.984498 | 0.984498 | yes |
| knot refit | 547 | 1 | 0.971805 | 0.971805 | yes |
| covariance shape 1/2 | 547 | 1 | 0.971805 | 0.971805 | yes |
| covariance shape 2/2 | 547 | 1 | 0.971805 | 0.971805 | yes |
| post-asinh transform 1/4 | 547 | 1 | 0.971805 | 0.971805 | yes |
| post-asinh transform 2/4 | 547 | 1 | 0.971805 | 0.971805 | yes |
| post-asinh transform 3/4 | 547 | 1 | 0.971805 | 0.971805 | yes |
| post-asinh transform 4/4 | 547 | 1 | 0.971805 | 0.971805 | yes |
| allocation beta | 547 | 1 | 0.971805 | 0.971805 | yes |
| knot refit | 547 | 1 | 0.961086 | 0.961086 | yes |
| covariance shape 1/2 | 547 | 1 | 0.961086 | 0.961086 | yes |
| covariance shape 2/2 | 547 | 1 | 0.961086 | 0.961086 | yes |
| post-asinh transform 1/4 | 547 | 1 | 0.961086 | 0.961086 | yes |
| post-asinh transform 2/4 | 547 | 1 | 0.961086 | 0.961086 | yes |
| post-asinh transform 3/4 | 547 | 1 | 0.961086 | 0.961086 | yes |
| post-asinh transform 4/4 | 547 | 1 | 0.961086 | 0.961086 | yes |
| allocation beta | 547 | 1 | 0.961086 | 0.961086 | yes |
| knot refit | 547 | 1 | 0.957361 | 0.957361 | yes |
| covariance shape 1/2 | 547 | 1 | 0.957361 | 0.957361 | yes |
| covariance shape 2/2 | 547 | 1 | 0.957361 | 0.957361 | yes |
| post-asinh transform 1/4 | 547 | 1 | 0.957361 | 0.957361 | yes |
| post-asinh transform 2/4 | 547 | 1 | 0.957361 | 0.957361 | yes |
| post-asinh transform 3/4 | 547 | 1 | 0.957361 | 0.957361 | yes |
| post-asinh transform 4/4 | 547 | 1 | 0.957361 | 0.957361 | yes |
| allocation beta | 547 | 1 | 0.957361 | 0.957361 | yes |
| knot refit | 547 | 1 | 0.95522 | 0.95522 | yes |
| covariance shape 1/2 | 547 | 1 | 0.95522 | 0.95522 | yes |
| covariance shape 2/2 | 547 | 1 | 0.95522 | 0.95522 | yes |
| post-asinh transform 1/4 | 547 | 1 | 0.95522 | 0.95522 | yes |
| post-asinh transform 2/4 | 547 | 1 | 0.95522 | 0.95522 | yes |
| post-asinh transform 3/4 | 547 | 1 | 0.95522 | 0.95522 | yes |
| post-asinh transform 4/4 | 547 | 1 | 0.95522 | 0.95522 | yes |
| allocation beta | 547 | 1 | 0.95522 | 0.95522 | yes |
| knot refit | 547 | 1 | 0.95197 | 0.95197 | yes |
| covariance shape 1/2 | 547 | 1 | 0.95197 | 0.95197 | yes |
| covariance shape 2/2 | 547 | 1 | 0.95197 | 0.95197 | yes |
| post-asinh transform 1/4 | 547 | 1 | 0.95197 | 0.95197 | yes |
| post-asinh transform 2/4 | 547 | 1 | 0.95197 | 0.95197 | yes |
| post-asinh transform 3/4 | 547 | 1 | 0.95197 | 0.95197 | yes |
| post-asinh transform 4/4 | 547 | 1 | 0.95197 | 0.95197 | yes |
| allocation beta | 547 | 1 | 0.95197 | 0.95197 | yes |
| knot refit | 547 | 1 | 0.95197 | 0.95197 | yes |
| refined parameter-search resolution | 547 | 1 | 0.95197 | 0.95197 | yes |
| covariance shape 1/2 | 547 | 1 | 0.95197 | 0.95197 | yes |
| covariance shape 2/2 | 547 | 1 | 0.95197 | 0.95197 | yes |
| post-asinh transform 1/4 | 547 | 1 | 0.95197 | 0.95197 | yes |
| post-asinh transform 2/4 | 547 | 1 | 0.95197 | 0.95197 | yes |
| post-asinh transform 3/4 | 547 | 1 | 0.95197 | 0.95197 | yes |
| post-asinh transform 4/4 | 547 | 1 | 0.95197 | 0.95197 | yes |
| allocation beta | 547 | 1 | 0.95197 | 0.95197 | yes |
| knot refit (cached stationary) | 547 | 1 | 0.95197 | 0.95197 | yes |
| refined parameter-search resolution | 547 | 1 | 0.95197 | 0.95197 | yes |
| covariance shape 1/2 | 547 | 1 | 0.95197 | 0.95197 | yes |
| covariance shape 2/2 | 547 | 1 | 0.95197 | 0.95197 | yes |
| post-asinh transform 1/4 | 547 | 1 | 0.95197 | 0.95197 | yes |
| post-asinh transform 2/4 | 547 | 1 | 0.95197 | 0.95197 | yes |
| post-asinh transform 3/4 | 547 | 1 | 0.95197 | 0.95197 | yes |
| post-asinh transform 4/4 | 547 | 1 | 0.95197 | 0.95197 | yes |
| allocation beta | 547 | 1 | 0.95197 | 0.95197 | yes |
| knot refit (cached stationary) | 547 | 1 | 0.95197 | 0.95197 | yes |
| fixed point reached | 547 | 1 | 0.95197 | 0.95197 | yes |
| prune proposal 638->547 | 547 | 1 | 0.95197 | 0.95197 | yes |
| covariance shape 1/2 | 547 | 1 | 2.59611 | 2.59611 | no |
| covariance shape 2/2 | 547 | 1 | 2.59611 | 2.59611 | no |
| post-asinh transform 1/4 | 547 | 1 | 2.59611 | 2.59611 | no |
| post-asinh transform 2/4 | 547 | 1 | 2.59611 | 2.59611 | no |
| post-asinh transform 3/4 | 547 | 1 | 2.59611 | 2.59611 | no |
| post-asinh transform 4/4 | 547 | 1 | 2.59611 | 2.59611 | no |
| allocation beta | 547 | 0.875 | 2.45759 | 2.45759 | no |
| knot refit | 547 | 0.875 | 2.45759 | 2.45759 | no |
| covariance shape 1/2 | 547 | 0.875 | 2.45759 | 2.45759 | no |
| covariance shape 2/2 | 547 | 0.875 | 2.45759 | 2.45759 | no |
| post-asinh transform 1/4 | 547 | 0.875 | 2.45759 | 2.45759 | no |
| post-asinh transform 2/4 | 547 | 0.875 | 2.45759 | 2.45759 | no |
| post-asinh transform 3/4 | 547 | 0.875 | 2.45759 | 2.45759 | no |
| post-asinh transform 4/4 | 547 | 0.875 | 2.45759 | 2.45759 | no |
| allocation beta | 547 | 1 | 2.4503 | 2.4503 | no |
| knot refit | 547 | 1 | 2.4503 | 2.4503 | no |
| covariance shape 1/2 | 547 | 1 | 2.4503 | 2.4503 | no |
| covariance shape 2/2 | 547 | 1 | 2.4503 | 2.4503 | no |
| post-asinh transform 1/4 | 547 | 1 | 2.4503 | 2.4503 | no |
| post-asinh transform 2/4 | 547 | 1 | 2.4503 | 2.4503 | no |
| post-asinh transform 3/4 | 547 | 1 | 2.4503 | 2.4503 | no |
| post-asinh transform 4/4 | 547 | 1 | 2.4503 | 2.4503 | no |
| allocation beta | 547 | 1 | 2.4503 | 2.4503 | no |
| knot refit (cached stationary) | 547 | 1 | 2.4503 | 2.4503 | no |
| refined parameter-search resolution | 547 | 1 | 2.4503 | 2.4503 | no |
| covariance shape 1/2 | 547 | 1 | 2.4503 | 2.4503 | no |
| covariance shape 2/2 | 547 | 1 | 2.4503 | 2.4503 | no |
| post-asinh transform 1/4 | 547 | 1 | 2.4503 | 2.4503 | no |
| post-asinh transform 2/4 | 547 | 1 | 2.4503 | 2.4503 | no |
| post-asinh transform 3/4 | 547 | 1 | 2.4503 | 2.4503 | no |
| post-asinh transform 4/4 | 547 | 1 | 2.4503 | 2.4503 | no |
| allocation beta | 547 | 1 | 2.4503 | 2.4503 | no |
| knot refit (cached stationary) | 547 | 1 | 2.4503 | 2.4503 | no |
| refined parameter-search resolution | 547 | 1 | 2.4503 | 2.4503 | no |
| covariance shape 1/2 | 547 | 1 | 2.4503 | 2.4503 | no |
| covariance shape 2/2 | 547 | 1 | 2.4503 | 2.4503 | no |
| post-asinh transform 1/4 | 547 | 1 | 2.4503 | 2.4503 | no |
| post-asinh transform 2/4 | 547 | 1 | 2.4503 | 2.4503 | no |
| post-asinh transform 3/4 | 547 | 1 | 2.4503 | 2.4503 | no |
| post-asinh transform 4/4 | 547 | 1 | 2.4503 | 2.4503 | no |
| allocation beta | 547 | 1 | 2.4503 | 2.4503 | no |
| knot refit (cached stationary) | 547 | 1 | 2.4503 | 2.4503 | no |
| fixed point reached | 547 | 1 | 2.4503 | 2.4503 | no |
| accepted prune | 547 | 1 | 0.95197 | 0.95197 | yes |
| cold restart 547->456 | 456 | 0.875 | 1.17794 | 1.17794 | no |
| covariance shape 1/2 | 456 | 1 | 1.18027 | 1.18027 | no |
| covariance shape 2/2 | 456 | 1 | 1.18027 | 1.18027 | no |
| post-asinh transform 1/4 | 456 | 1 | 1.18027 | 1.18027 | no |
| post-asinh transform 2/4 | 456 | 1 | 1.18027 | 1.18027 | no |
| post-asinh transform 3/4 | 456 | 1 | 1.18027 | 1.18027 | no |
| post-asinh transform 4/4 | 456 | 1 | 1.18027 | 1.18027 | no |
| allocation beta | 456 | 0.875 | 1.17794 | 1.17794 | no |
| knot refit | 456 | 0.875 | 1.17794 | 1.17794 | no |
| covariance shape 1/2 | 456 | 0.875 | 1.17794 | 1.17794 | no |
| covariance shape 2/2 | 456 | 0.875 | 1.17794 | 1.17794 | no |
| post-asinh transform 1/4 | 456 | 0.875 | 1.17794 | 1.17794 | no |
| post-asinh transform 2/4 | 456 | 0.875 | 1.17794 | 1.17794 | no |
| post-asinh transform 3/4 | 456 | 0.875 | 1.17794 | 1.17794 | no |
| post-asinh transform 4/4 | 456 | 0.875 | 1.17794 | 1.17794 | no |
| allocation beta | 456 | 0.875 | 1.17794 | 1.17794 | no |
| knot refit (cached stationary) | 456 | 0.875 | 1.17794 | 1.17794 | no |
| refined parameter-search resolution | 456 | 0.875 | 1.17794 | 1.17794 | no |
| covariance shape 1/2 | 456 | 0.875 | 1.17794 | 1.17794 | no |
| covariance shape 2/2 | 456 | 0.875 | 1.17794 | 1.17794 | no |
| post-asinh transform 1/4 | 456 | 0.875 | 1.17794 | 1.17794 | no |
| post-asinh transform 2/4 | 456 | 0.875 | 1.17794 | 1.17794 | no |
| post-asinh transform 3/4 | 456 | 0.875 | 1.17794 | 1.17794 | no |
| post-asinh transform 4/4 | 456 | 0.875 | 1.17794 | 1.17794 | no |
| allocation beta | 456 | 0.875 | 1.17794 | 1.17794 | no |
| knot refit (cached stationary) | 456 | 0.875 | 1.17794 | 1.17794 | no |
| refined parameter-search resolution | 456 | 0.875 | 1.17794 | 1.17794 | no |
| covariance shape 1/2 | 456 | 0.875 | 1.17794 | 1.17794 | no |
| covariance shape 2/2 | 456 | 0.875 | 1.17794 | 1.17794 | no |
| post-asinh transform 1/4 | 456 | 0.875 | 1.17794 | 1.17794 | no |
| post-asinh transform 2/4 | 456 | 0.875 | 1.17794 | 1.17794 | no |
| post-asinh transform 3/4 | 456 | 0.875 | 1.17794 | 1.17794 | no |
| post-asinh transform 4/4 | 456 | 0.875 | 1.17794 | 1.17794 | no |
| allocation beta | 456 | 0.875 | 1.17794 | 1.17794 | no |
| knot refit (cached stationary) | 456 | 0.875 | 1.17794 | 1.17794 | no |
| fixed point reached | 456 | 0.875 | 1.17794 | 1.17794 | no |
| prune proposal 547->456 | 456 | 1 | 1.02528 | 1.02528 | no |
| covariance shape 1/2 | 456 | 1 | 1.02528 | 1.02528 | no |
| covariance shape 2/2 | 456 | 1 | 1.02528 | 1.02528 | no |
| post-asinh transform 1/4 | 456 | 1 | 1.02528 | 1.02528 | no |
| post-asinh transform 2/4 | 456 | 1 | 1.02528 | 1.02528 | no |
| post-asinh transform 3/4 | 456 | 1 | 1.02528 | 1.02528 | no |
| post-asinh transform 4/4 | 456 | 1 | 1.02528 | 1.02528 | no |
| allocation beta | 456 | 1 | 1.02528 | 1.02528 | no |
| knot refit | 456 | 1 | 1.02528 | 1.02528 | no |
| refined parameter-search resolution | 456 | 1 | 1.02528 | 1.02528 | no |
| covariance shape 1/2 | 456 | 1 | 1.02528 | 1.02528 | no |
| covariance shape 2/2 | 456 | 1 | 1.02528 | 1.02528 | no |
| post-asinh transform 1/4 | 456 | 1 | 1.02528 | 1.02528 | no |
| post-asinh transform 2/4 | 456 | 1 | 1.02528 | 1.02528 | no |
| post-asinh transform 3/4 | 456 | 1 | 1.02528 | 1.02528 | no |
| post-asinh transform 4/4 | 456 | 1 | 1.02528 | 1.02528 | no |
| allocation beta | 456 | 1 | 1.02528 | 1.02528 | no |
| knot refit (cached stationary) | 456 | 1 | 1.02528 | 1.02528 | no |
| refined parameter-search resolution | 456 | 1 | 1.02528 | 1.02528 | no |
| covariance shape 1/2 | 456 | 1 | 1.02528 | 1.02528 | no |
| covariance shape 2/2 | 456 | 1 | 1.02528 | 1.02528 | no |
| post-asinh transform 1/4 | 456 | 1 | 1.02528 | 1.02528 | no |
| post-asinh transform 2/4 | 456 | 1 | 1.02528 | 1.02528 | no |
| post-asinh transform 3/4 | 456 | 1 | 1.02528 | 1.02528 | no |
| post-asinh transform 4/4 | 456 | 1 | 1.02528 | 1.02528 | no |
| allocation beta | 456 | 1 | 1.02528 | 1.02528 | no |
| knot refit (cached stationary) | 456 | 1 | 1.02528 | 1.02528 | no |
| fixed point reached | 456 | 1 | 1.02528 | 1.02528 | no |
| rejected prune | 456 | 1 | 1.02528 | 1.02528 | no |
| prune proposal 547->502 | 502 | 1 | 0.952118 | 0.952118 | yes |
| covariance shape 1/2 | 502 | 1 | 0.952118 | 0.952118 | yes |
| covariance shape 2/2 | 502 | 1 | 0.952118 | 0.952118 | yes |
| post-asinh transform 1/4 | 502 | 1 | 0.952118 | 0.952118 | yes |
| post-asinh transform 2/4 | 502 | 1 | 0.952118 | 0.952118 | yes |
| post-asinh transform 3/4 | 502 | 1 | 0.952118 | 0.952118 | yes |
| post-asinh transform 4/4 | 502 | 1 | 0.952118 | 0.952118 | yes |
| allocation beta | 502 | 1 | 0.952118 | 0.952118 | yes |
| knot refit | 502 | 1 | 0.952118 | 0.952118 | yes |
| refined parameter-search resolution | 502 | 1 | 0.952118 | 0.952118 | yes |
| covariance shape 1/2 | 502 | 1 | 0.952118 | 0.952118 | yes |
| covariance shape 2/2 | 502 | 1 | 0.952118 | 0.952118 | yes |
| post-asinh transform 1/4 | 502 | 1 | 0.952118 | 0.952118 | yes |
| post-asinh transform 2/4 | 502 | 1 | 0.952118 | 0.952118 | yes |
| post-asinh transform 3/4 | 502 | 1 | 0.952118 | 0.952118 | yes |
| post-asinh transform 4/4 | 502 | 1 | 0.952118 | 0.952118 | yes |
| allocation beta | 502 | 1 | 0.952118 | 0.952118 | yes |
| knot refit (cached stationary) | 502 | 1 | 0.952118 | 0.952118 | yes |
| refined parameter-search resolution | 502 | 1 | 0.952118 | 0.952118 | yes |
| covariance shape 1/2 | 502 | 1 | 0.952118 | 0.952118 | yes |
| covariance shape 2/2 | 502 | 1 | 0.952118 | 0.952118 | yes |
| post-asinh transform 1/4 | 502 | 1 | 0.952118 | 0.952118 | yes |
| post-asinh transform 2/4 | 502 | 1 | 0.952118 | 0.952118 | yes |
| post-asinh transform 3/4 | 502 | 1 | 0.952118 | 0.952118 | yes |
| post-asinh transform 4/4 | 502 | 1 | 0.952118 | 0.952118 | yes |
| allocation beta | 502 | 1 | 0.952118 | 0.952118 | yes |
| knot refit (cached stationary) | 502 | 1 | 0.952118 | 0.952118 | yes |
| fixed point reached | 502 | 1 | 0.952118 | 0.952118 | yes |
| accepted prune | 502 | 1 | 0.952118 | 0.952118 | yes |
| cold restart 502->457 | 457 | 1 | 1.17 | 1.17 | no |
| covariance shape 1/2 | 457 | 1 | 1.23854 | 1.23854 | no |
| covariance shape 2/2 | 457 | 1 | 1.23854 | 1.23854 | no |
| post-asinh transform 1/4 | 457 | 1 | 1.23854 | 1.23854 | no |
| post-asinh transform 2/4 | 457 | 1 | 1.23854 | 1.23854 | no |
| post-asinh transform 3/4 | 457 | 1 | 1.23854 | 1.23854 | no |
| post-asinh transform 4/4 | 457 | 1 | 1.23854 | 1.23854 | no |
| allocation beta | 457 | 0.875 | 1.22404 | 1.22404 | no |
| knot refit | 457 | 0.875 | 1.2006 | 1.2006 | no |
| covariance shape 1/2 | 457 | 0.875 | 1.2006 | 1.2006 | no |
| covariance shape 2/2 | 457 | 0.875 | 1.2006 | 1.2006 | no |
| post-asinh transform 1/4 | 457 | 0.875 | 1.2006 | 1.2006 | no |
| post-asinh transform 2/4 | 457 | 0.875 | 1.2006 | 1.2006 | no |
| post-asinh transform 3/4 | 457 | 0.875 | 1.2006 | 1.2006 | no |
| post-asinh transform 4/4 | 457 | 0.875 | 1.2006 | 1.2006 | no |
| allocation beta | 457 | 1 | 1.17972 | 1.17972 | no |
| knot refit | 457 | 1 | 1.17 | 1.17 | no |
| covariance shape 1/2 | 457 | 1 | 1.17 | 1.17 | no |
| covariance shape 2/2 | 457 | 1 | 1.17 | 1.17 | no |
| post-asinh transform 1/4 | 457 | 1 | 1.17 | 1.17 | no |
| post-asinh transform 2/4 | 457 | 1 | 1.17 | 1.17 | no |
| post-asinh transform 3/4 | 457 | 1 | 1.17 | 1.17 | no |
| post-asinh transform 4/4 | 457 | 1 | 1.17 | 1.17 | no |
| allocation beta | 457 | 1 | 1.17 | 1.17 | no |
| knot refit | 457 | 1 | 1.17 | 1.17 | no |
| refined parameter-search resolution | 457 | 1 | 1.17 | 1.17 | no |
| covariance shape 1/2 | 457 | 1 | 1.17 | 1.17 | no |
| covariance shape 2/2 | 457 | 1 | 1.17 | 1.17 | no |
| post-asinh transform 1/4 | 457 | 1 | 1.17 | 1.17 | no |
| post-asinh transform 2/4 | 457 | 1 | 1.17 | 1.17 | no |
| post-asinh transform 3/4 | 457 | 1 | 1.17 | 1.17 | no |
| post-asinh transform 4/4 | 457 | 1 | 1.17 | 1.17 | no |
| allocation beta | 457 | 1 | 1.17 | 1.17 | no |
| knot refit (cached stationary) | 457 | 1 | 1.17 | 1.17 | no |
| refined parameter-search resolution | 457 | 1 | 1.17 | 1.17 | no |
| covariance shape 1/2 | 457 | 1 | 1.17 | 1.17 | no |
| covariance shape 2/2 | 457 | 1 | 1.17 | 1.17 | no |
| post-asinh transform 1/4 | 457 | 1 | 1.17 | 1.17 | no |
| post-asinh transform 2/4 | 457 | 1 | 1.17 | 1.17 | no |
| post-asinh transform 3/4 | 457 | 1 | 1.17 | 1.17 | no |
| post-asinh transform 4/4 | 457 | 1 | 1.17 | 1.17 | no |
| allocation beta | 457 | 1 | 1.17 | 1.17 | no |
| knot refit (cached stationary) | 457 | 1 | 1.17 | 1.17 | no |
| fixed point reached | 457 | 1 | 1.17 | 1.17 | no |
| prune proposal 502->457 | 457 | 1 | 1.10209 | 1.10209 | no |
| covariance shape 1/2 | 457 | 1 | 1.10209 | 1.10209 | no |
| covariance shape 2/2 | 457 | 1 | 1.10209 | 1.10209 | no |
| post-asinh transform 1/4 | 457 | 1 | 1.10209 | 1.10209 | no |
| post-asinh transform 2/4 | 457 | 1 | 1.10209 | 1.10209 | no |
| post-asinh transform 3/4 | 457 | 1 | 1.10209 | 1.10209 | no |
| post-asinh transform 4/4 | 457 | 1 | 1.10209 | 1.10209 | no |
| allocation beta | 457 | 1 | 1.10209 | 1.10209 | no |
| knot refit | 457 | 1 | 1.10209 | 1.10209 | no |
| refined parameter-search resolution | 457 | 1 | 1.10209 | 1.10209 | no |
| covariance shape 1/2 | 457 | 1 | 1.10209 | 1.10209 | no |
| covariance shape 2/2 | 457 | 1 | 1.10209 | 1.10209 | no |
| post-asinh transform 1/4 | 457 | 1 | 1.10209 | 1.10209 | no |
| post-asinh transform 2/4 | 457 | 1 | 1.10209 | 1.10209 | no |
| post-asinh transform 3/4 | 457 | 1 | 1.10209 | 1.10209 | no |
| post-asinh transform 4/4 | 457 | 1 | 1.10209 | 1.10209 | no |
| allocation beta | 457 | 1 | 1.10209 | 1.10209 | no |
| knot refit (cached stationary) | 457 | 1 | 1.10209 | 1.10209 | no |
| refined parameter-search resolution | 457 | 1 | 1.10209 | 1.10209 | no |
| covariance shape 1/2 | 457 | 1 | 1.10209 | 1.10209 | no |
| covariance shape 2/2 | 457 | 1 | 1.10209 | 1.10209 | no |
| post-asinh transform 1/4 | 457 | 1 | 1.10209 | 1.10209 | no |
| post-asinh transform 2/4 | 457 | 1 | 1.10209 | 1.10209 | no |
| post-asinh transform 3/4 | 457 | 1 | 1.10209 | 1.10209 | no |
| post-asinh transform 4/4 | 457 | 1 | 1.10209 | 1.10209 | no |
| allocation beta | 457 | 1 | 1.10209 | 1.10209 | no |
| knot refit (cached stationary) | 457 | 1 | 1.10209 | 1.10209 | no |
| fixed point reached | 457 | 1 | 1.10209 | 1.10209 | no |
| rejected prune | 457 | 1 | 1.10209 | 1.10209 | no |
| prune proposal 502->480 | 480 | 1 | 0.96694 | 0.96694 | yes |
| covariance shape 1/2 | 480 | 1 | 0.96694 | 0.96694 | yes |
| covariance shape 2/2 | 480 | 1 | 0.96694 | 0.96694 | yes |
| post-asinh transform 1/4 | 480 | 1 | 0.96694 | 0.96694 | yes |
| post-asinh transform 2/4 | 480 | 1 | 0.96694 | 0.96694 | yes |
| post-asinh transform 3/4 | 480 | 1 | 0.96694 | 0.96694 | yes |
| post-asinh transform 4/4 | 480 | 1 | 0.96694 | 0.96694 | yes |
| allocation beta | 480 | 1 | 0.96694 | 0.96694 | yes |
| knot refit | 480 | 1 | 0.96694 | 0.96694 | yes |
| refined parameter-search resolution | 480 | 1 | 0.96694 | 0.96694 | yes |
| covariance shape 1/2 | 480 | 1 | 0.96694 | 0.96694 | yes |
| covariance shape 2/2 | 480 | 1 | 0.96694 | 0.96694 | yes |
| post-asinh transform 1/4 | 480 | 1 | 0.96694 | 0.96694 | yes |
| post-asinh transform 2/4 | 480 | 1 | 0.96694 | 0.96694 | yes |
| post-asinh transform 3/4 | 480 | 1 | 0.96694 | 0.96694 | yes |
| post-asinh transform 4/4 | 480 | 1 | 0.96694 | 0.96694 | yes |
| allocation beta | 480 | 1 | 0.96694 | 0.96694 | yes |
| knot refit (cached stationary) | 480 | 1 | 0.96694 | 0.96694 | yes |
| refined parameter-search resolution | 480 | 1 | 0.96694 | 0.96694 | yes |
| covariance shape 1/2 | 480 | 1 | 0.96694 | 0.96694 | yes |
| covariance shape 2/2 | 480 | 1 | 0.96694 | 0.96694 | yes |
| post-asinh transform 1/4 | 480 | 1 | 0.96694 | 0.96694 | yes |
| post-asinh transform 2/4 | 480 | 1 | 0.96694 | 0.96694 | yes |
| post-asinh transform 3/4 | 480 | 1 | 0.96694 | 0.96694 | yes |
| post-asinh transform 4/4 | 480 | 1 | 0.96694 | 0.96694 | yes |
| allocation beta | 480 | 1 | 0.96694 | 0.96694 | yes |
| knot refit (cached stationary) | 480 | 1 | 0.96694 | 0.96694 | yes |
| fixed point reached | 480 | 1 | 0.96694 | 0.96694 | yes |
| accepted prune | 480 | 1 | 0.96694 | 0.96694 | yes |
| cold restart 480->458 | 458 | 1 | 1.1962 | 1.1962 | no |
| covariance shape 1/2 | 458 | 1 | 1.1962 | 1.1962 | no |
| covariance shape 2/2 | 458 | 1 | 1.1962 | 1.1962 | no |
| post-asinh transform 1/4 | 458 | 1 | 1.1962 | 1.1962 | no |
| post-asinh transform 2/4 | 458 | 1 | 1.1962 | 1.1962 | no |
| post-asinh transform 3/4 | 458 | 1 | 1.1962 | 1.1962 | no |
| post-asinh transform 4/4 | 458 | 1 | 1.1962 | 1.1962 | no |
| allocation beta | 458 | 1 | 1.1962 | 1.1962 | no |
| knot refit | 458 | 1 | 1.1962 | 1.1962 | no |
| refined parameter-search resolution | 458 | 1 | 1.1962 | 1.1962 | no |
| covariance shape 1/2 | 458 | 1 | 1.1962 | 1.1962 | no |
| covariance shape 2/2 | 458 | 1 | 1.1962 | 1.1962 | no |
| post-asinh transform 1/4 | 458 | 1 | 1.1962 | 1.1962 | no |
| post-asinh transform 2/4 | 458 | 1 | 1.1962 | 1.1962 | no |
| post-asinh transform 3/4 | 458 | 1 | 1.1962 | 1.1962 | no |
| post-asinh transform 4/4 | 458 | 1 | 1.1962 | 1.1962 | no |
| allocation beta | 458 | 1 | 1.1962 | 1.1962 | no |
| knot refit (cached stationary) | 458 | 1 | 1.1962 | 1.1962 | no |
| refined parameter-search resolution | 458 | 1 | 1.1962 | 1.1962 | no |
| covariance shape 1/2 | 458 | 1 | 1.1962 | 1.1962 | no |
| covariance shape 2/2 | 458 | 1 | 1.1962 | 1.1962 | no |
| post-asinh transform 1/4 | 458 | 1 | 1.1962 | 1.1962 | no |
| post-asinh transform 2/4 | 458 | 1 | 1.1962 | 1.1962 | no |
| post-asinh transform 3/4 | 458 | 1 | 1.1962 | 1.1962 | no |
| post-asinh transform 4/4 | 458 | 1 | 1.1962 | 1.1962 | no |
| allocation beta | 458 | 1 | 1.1962 | 1.1962 | no |
| knot refit (cached stationary) | 458 | 1 | 1.1962 | 1.1962 | no |
| fixed point reached | 458 | 1 | 1.1962 | 1.1962 | no |
| prune proposal 480->458 | 458 | 1 | 1.13889 | 1.13889 | no |
| covariance shape 1/2 | 458 | 1 | 1.13889 | 1.13889 | no |
| covariance shape 2/2 | 458 | 1 | 1.13889 | 1.13889 | no |
| post-asinh transform 1/4 | 458 | 1 | 1.13889 | 1.13889 | no |
| post-asinh transform 2/4 | 458 | 1 | 1.13889 | 1.13889 | no |
| post-asinh transform 3/4 | 458 | 1 | 1.13889 | 1.13889 | no |
| post-asinh transform 4/4 | 458 | 1 | 1.13889 | 1.13889 | no |
| allocation beta | 458 | 1 | 1.13889 | 1.13889 | no |
| knot refit | 458 | 1 | 1.13889 | 1.13889 | no |
| refined parameter-search resolution | 458 | 1 | 1.13889 | 1.13889 | no |
| covariance shape 1/2 | 458 | 1 | 1.13889 | 1.13889 | no |
| covariance shape 2/2 | 458 | 1 | 1.13889 | 1.13889 | no |
| post-asinh transform 1/4 | 458 | 1 | 1.13889 | 1.13889 | no |
| post-asinh transform 2/4 | 458 | 1 | 1.13889 | 1.13889 | no |
| post-asinh transform 3/4 | 458 | 1 | 1.13889 | 1.13889 | no |
| post-asinh transform 4/4 | 458 | 1 | 1.13889 | 1.13889 | no |
| allocation beta | 458 | 1 | 1.13889 | 1.13889 | no |
| knot refit (cached stationary) | 458 | 1 | 1.13889 | 1.13889 | no |
| refined parameter-search resolution | 458 | 1 | 1.13889 | 1.13889 | no |
| covariance shape 1/2 | 458 | 1 | 1.13889 | 1.13889 | no |
| covariance shape 2/2 | 458 | 1 | 1.13889 | 1.13889 | no |
| post-asinh transform 1/4 | 458 | 1 | 1.13889 | 1.13889 | no |
| post-asinh transform 2/4 | 458 | 1 | 1.13889 | 1.13889 | no |
| post-asinh transform 3/4 | 458 | 1 | 1.13889 | 1.13889 | no |
| post-asinh transform 4/4 | 458 | 1 | 1.13889 | 1.13889 | no |
| allocation beta | 458 | 1 | 1.13889 | 1.13889 | no |
| knot refit (cached stationary) | 458 | 1 | 1.13889 | 1.13889 | no |
| fixed point reached | 458 | 1 | 1.13889 | 1.13889 | no |
| rejected prune | 458 | 1 | 1.13889 | 1.13889 | no |
| prune proposal 480->469 | 469 | 1 | 0.932444 | 0.932444 | yes |
| covariance shape 1/2 | 469 | 1 | 0.940994 | 0.940994 | yes |
| covariance shape 2/2 | 469 | 1 | 0.940994 | 0.940994 | yes |
| post-asinh transform 1/4 | 469 | 1 | 0.940994 | 0.940994 | yes |
| post-asinh transform 2/4 | 469 | 1 | 0.940994 | 0.940994 | yes |
| post-asinh transform 3/4 | 469 | 1 | 0.940994 | 0.940994 | yes |
| post-asinh transform 4/4 | 469 | 1 | 0.940994 | 0.940994 | yes |
| allocation beta | 469 | 1 | 0.940994 | 0.940994 | yes |
| knot refit | 469 | 1 | 0.938722 | 0.938722 | yes |
| covariance shape 1/2 | 469 | 1 | 0.938722 | 0.938722 | yes |
| covariance shape 2/2 | 469 | 1 | 0.938722 | 0.938722 | yes |
| post-asinh transform 1/4 | 469 | 1 | 0.938722 | 0.938722 | yes |
| post-asinh transform 2/4 | 469 | 1 | 0.938722 | 0.938722 | yes |
| post-asinh transform 3/4 | 469 | 1 | 0.938722 | 0.938722 | yes |
| post-asinh transform 4/4 | 469 | 1 | 0.938722 | 0.938722 | yes |
| allocation beta | 469 | 1 | 0.938722 | 0.938722 | yes |
| knot refit | 469 | 1 | 0.932444 | 0.932444 | yes |
| covariance shape 1/2 | 469 | 1 | 0.932444 | 0.932444 | yes |
| covariance shape 2/2 | 469 | 1 | 0.932444 | 0.932444 | yes |
| post-asinh transform 1/4 | 469 | 1 | 0.932444 | 0.932444 | yes |
| post-asinh transform 2/4 | 469 | 1 | 0.932444 | 0.932444 | yes |
| post-asinh transform 3/4 | 469 | 1 | 0.932444 | 0.932444 | yes |
| post-asinh transform 4/4 | 469 | 1 | 0.932444 | 0.932444 | yes |
| allocation beta | 469 | 1 | 0.932444 | 0.932444 | yes |
| knot refit | 469 | 1 | 0.932444 | 0.932444 | yes |
| refined parameter-search resolution | 469 | 1 | 0.932444 | 0.932444 | yes |
| covariance shape 1/2 | 469 | 1 | 0.932444 | 0.932444 | yes |
| covariance shape 2/2 | 469 | 1 | 0.932444 | 0.932444 | yes |
| post-asinh transform 1/4 | 469 | 1 | 0.932444 | 0.932444 | yes |
| post-asinh transform 2/4 | 469 | 1 | 0.932444 | 0.932444 | yes |
| post-asinh transform 3/4 | 469 | 1 | 0.932444 | 0.932444 | yes |
| post-asinh transform 4/4 | 469 | 1 | 0.932444 | 0.932444 | yes |
| allocation beta | 469 | 1 | 0.932444 | 0.932444 | yes |
| knot refit (cached stationary) | 469 | 1 | 0.932444 | 0.932444 | yes |
| refined parameter-search resolution | 469 | 1 | 0.932444 | 0.932444 | yes |
| covariance shape 1/2 | 469 | 1 | 0.932444 | 0.932444 | yes |
| covariance shape 2/2 | 469 | 1 | 0.932444 | 0.932444 | yes |
| post-asinh transform 1/4 | 469 | 1 | 0.932444 | 0.932444 | yes |
| post-asinh transform 2/4 | 469 | 1 | 0.932444 | 0.932444 | yes |
| post-asinh transform 3/4 | 469 | 1 | 0.932444 | 0.932444 | yes |
| post-asinh transform 4/4 | 469 | 1 | 0.932444 | 0.932444 | yes |
| allocation beta | 469 | 1 | 0.932444 | 0.932444 | yes |
| knot refit (cached stationary) | 469 | 1 | 0.932444 | 0.932444 | yes |
| fixed point reached | 469 | 1 | 0.932444 | 0.932444 | yes |
| accepted prune | 469 | 1 | 0.932444 | 0.932444 | yes |
| cold restart 469->458 | 458 | 1 | 1.18163 | 1.18163 | no |
| covariance shape 1/2 | 458 | 1 | 1.1858 | 1.1858 | no |
| covariance shape 2/2 | 458 | 1 | 1.1858 | 1.1858 | no |
| post-asinh transform 1/4 | 458 | 1 | 1.1858 | 1.1858 | no |
| post-asinh transform 2/4 | 458 | 1 | 1.1858 | 1.1858 | no |
| post-asinh transform 3/4 | 458 | 1 | 1.1858 | 1.1858 | no |
| post-asinh transform 4/4 | 458 | 1 | 1.1858 | 1.1858 | no |
| allocation beta | 458 | 1 | 1.1858 | 1.1858 | no |
| knot refit | 458 | 1 | 1.18163 | 1.18163 | no |
| covariance shape 1/2 | 458 | 1 | 1.18163 | 1.18163 | no |
| covariance shape 2/2 | 458 | 1 | 1.18163 | 1.18163 | no |
| post-asinh transform 1/4 | 458 | 1 | 1.18163 | 1.18163 | no |
| post-asinh transform 2/4 | 458 | 1 | 1.18163 | 1.18163 | no |
| post-asinh transform 3/4 | 458 | 1 | 1.18163 | 1.18163 | no |
| post-asinh transform 4/4 | 458 | 1 | 1.18163 | 1.18163 | no |
| allocation beta | 458 | 1 | 1.18163 | 1.18163 | no |
| knot refit | 458 | 1 | 1.18163 | 1.18163 | no |
| refined parameter-search resolution | 458 | 1 | 1.18163 | 1.18163 | no |
| covariance shape 1/2 | 458 | 1 | 1.18163 | 1.18163 | no |
| covariance shape 2/2 | 458 | 1 | 1.18163 | 1.18163 | no |
| post-asinh transform 1/4 | 458 | 1 | 1.18163 | 1.18163 | no |
| post-asinh transform 2/4 | 458 | 1 | 1.18163 | 1.18163 | no |
| post-asinh transform 3/4 | 458 | 1 | 1.18163 | 1.18163 | no |
| post-asinh transform 4/4 | 458 | 1 | 1.18163 | 1.18163 | no |
| allocation beta | 458 | 1 | 1.18163 | 1.18163 | no |
| knot refit (cached stationary) | 458 | 1 | 1.18163 | 1.18163 | no |
| refined parameter-search resolution | 458 | 1 | 1.18163 | 1.18163 | no |
| covariance shape 1/2 | 458 | 1 | 1.18163 | 1.18163 | no |
| covariance shape 2/2 | 458 | 1 | 1.18163 | 1.18163 | no |
| post-asinh transform 1/4 | 458 | 1 | 1.18163 | 1.18163 | no |
| post-asinh transform 2/4 | 458 | 1 | 1.18163 | 1.18163 | no |
| post-asinh transform 3/4 | 458 | 1 | 1.18163 | 1.18163 | no |
| post-asinh transform 4/4 | 458 | 1 | 1.18163 | 1.18163 | no |
| allocation beta | 458 | 1 | 1.18163 | 1.18163 | no |
| knot refit (cached stationary) | 458 | 1 | 1.18163 | 1.18163 | no |
| fixed point reached | 458 | 1 | 1.18163 | 1.18163 | no |
| prune proposal 469->458 | 458 | 1 | 1.05956 | 1.05956 | no |
| covariance shape 1/2 | 458 | 1 | 1.0735 | 1.0735 | no |
| covariance shape 2/2 | 458 | 1 | 1.0735 | 1.0735 | no |
| post-asinh transform 1/4 | 458 | 1 | 1.0735 | 1.0735 | no |
| post-asinh transform 2/4 | 458 | 1 | 1.0735 | 1.0735 | no |
| post-asinh transform 3/4 | 458 | 1 | 1.0735 | 1.0735 | no |
| post-asinh transform 4/4 | 458 | 1 | 1.0735 | 1.0735 | no |
| allocation beta | 458 | 1 | 1.0735 | 1.0735 | no |
| knot refit | 458 | 1 | 1.06476 | 1.06476 | no |
| covariance shape 1/2 | 458 | 1 | 1.06476 | 1.06476 | no |
| covariance shape 2/2 | 458 | 1 | 1.06476 | 1.06476 | no |
| post-asinh transform 1/4 | 458 | 1 | 1.06476 | 1.06476 | no |
| post-asinh transform 2/4 | 458 | 1 | 1.06476 | 1.06476 | no |
| post-asinh transform 3/4 | 458 | 1 | 1.06476 | 1.06476 | no |
| post-asinh transform 4/4 | 458 | 1 | 1.06476 | 1.06476 | no |
| allocation beta | 458 | 1 | 1.06476 | 1.06476 | no |
| knot refit | 458 | 1 | 1.0634 | 1.0634 | no |
| covariance shape 1/2 | 458 | 1 | 1.0634 | 1.0634 | no |
| covariance shape 2/2 | 458 | 1 | 1.0634 | 1.0634 | no |
| post-asinh transform 1/4 | 458 | 1 | 1.0634 | 1.0634 | no |
| post-asinh transform 2/4 | 458 | 1 | 1.0634 | 1.0634 | no |
| post-asinh transform 3/4 | 458 | 1 | 1.0634 | 1.0634 | no |
| post-asinh transform 4/4 | 458 | 1 | 1.0634 | 1.0634 | no |
| allocation beta | 458 | 1 | 1.0634 | 1.0634 | no |
| knot refit | 458 | 1 | 1.0634 | 1.0634 | no |
| refined parameter-search resolution | 458 | 1 | 1.0634 | 1.0634 | no |
| covariance shape 1/2 | 458 | 1 | 1.0634 | 1.0634 | no |
| covariance shape 2/2 | 458 | 1 | 1.0634 | 1.0634 | no |
| post-asinh transform 1/4 | 458 | 1 | 1.0634 | 1.0634 | no |
| post-asinh transform 2/4 | 458 | 1 | 1.0634 | 1.0634 | no |
| post-asinh transform 3/4 | 458 | 1 | 1.0634 | 1.0634 | no |
| post-asinh transform 4/4 | 458 | 1 | 1.0634 | 1.0634 | no |
| allocation beta | 458 | 1 | 1.0634 | 1.0634 | no |
| knot refit (cached stationary) | 458 | 1 | 1.0634 | 1.0634 | no |
| refined parameter-search resolution | 458 | 1 | 1.0634 | 1.0634 | no |
| covariance shape 1/2 | 458 | 1 | 1.0634 | 1.0634 | no |
| covariance shape 2/2 | 458 | 1 | 1.0634 | 1.0634 | no |
| post-asinh transform 1/4 | 458 | 1 | 1.0634 | 1.0634 | no |
| post-asinh transform 2/4 | 458 | 1 | 1.06151 | 1.06151 | no |
| post-asinh transform 3/4 | 458 | 1 | 1.06151 | 1.06151 | no |
| post-asinh transform 4/4 | 458 | 1 | 1.06151 | 1.06151 | no |
| allocation beta | 458 | 1 | 1.06151 | 1.06151 | no |
| knot refit | 458 | 1 | 1.05956 | 1.05956 | no |
| covariance shape 1/2 | 458 | 1 | 1.05956 | 1.05956 | no |
| covariance shape 2/2 | 458 | 1 | 1.05956 | 1.05956 | no |
| post-asinh transform 1/4 | 458 | 1 | 1.05956 | 1.05956 | no |
| post-asinh transform 2/4 | 458 | 1 | 1.05956 | 1.05956 | no |
| post-asinh transform 3/4 | 458 | 1 | 1.05956 | 1.05956 | no |
| post-asinh transform 4/4 | 458 | 1 | 1.05956 | 1.05956 | no |
| allocation beta | 458 | 1 | 1.05956 | 1.05956 | no |
| knot refit | 458 | 1 | 1.05956 | 1.05956 | no |
| fixed point reached | 458 | 1 | 1.05956 | 1.05956 | no |
| rejected prune | 458 | 1 | 1.05956 | 1.05956 | no |
| prune proposal 469->464 | 464 | 1 | 0.982207 | 0.982207 | yes |
| covariance shape 1/2 | 464 | 1 | 0.987888 | 0.987888 | yes |
| covariance shape 2/2 | 464 | 1 | 0.987888 | 0.987888 | yes |
| post-asinh transform 1/4 | 464 | 1 | 0.987888 | 0.987888 | yes |
| post-asinh transform 2/4 | 464 | 1 | 0.987888 | 0.987888 | yes |
| post-asinh transform 3/4 | 464 | 1 | 0.987888 | 0.987888 | yes |
| post-asinh transform 4/4 | 464 | 1 | 0.987888 | 0.987888 | yes |
| allocation beta | 464 | 1 | 0.987888 | 0.987888 | yes |
| knot refit | 464 | 1 | 0.987888 | 0.987888 | yes |
| refined parameter-search resolution | 464 | 1 | 0.987888 | 0.987888 | yes |
| covariance shape 1/2 | 464 | 1 | 0.987888 | 0.987888 | yes |
| covariance shape 2/2 | 464 | 1 | 0.987888 | 0.987888 | yes |
| post-asinh transform 1/4 | 464 | 1 | 0.987888 | 0.987888 | yes |
| post-asinh transform 2/4 | 464 | 1 | 0.987888 | 0.987888 | yes |
| post-asinh transform 3/4 | 464 | 1 | 0.987888 | 0.987888 | yes |
| post-asinh transform 4/4 | 464 | 1 | 0.987888 | 0.987888 | yes |
| allocation beta | 464 | 1 | 0.987888 | 0.987888 | yes |
| knot refit (cached stationary) | 464 | 1 | 0.987888 | 0.987888 | yes |
| refined parameter-search resolution | 464 | 1 | 0.987888 | 0.987888 | yes |
| covariance shape 1/2 | 464 | 1 | 0.987888 | 0.987888 | yes |
| covariance shape 2/2 | 464 | 1 | 0.987888 | 0.987888 | yes |
| post-asinh transform 1/4 | 464 | 1 | 0.987888 | 0.987888 | yes |
| post-asinh transform 2/4 | 464 | 1 | 0.986742 | 0.986742 | yes |
| post-asinh transform 3/4 | 464 | 1 | 0.986742 | 0.986742 | yes |
| post-asinh transform 4/4 | 464 | 1 | 0.986742 | 0.986742 | yes |
| allocation beta | 464 | 1 | 0.986742 | 0.986742 | yes |
| knot refit | 464 | 1 | 0.985121 | 0.985121 | yes |
| covariance shape 1/2 | 464 | 1 | 0.985121 | 0.985121 | yes |
| covariance shape 2/2 | 464 | 1 | 0.985121 | 0.985121 | yes |
| post-asinh transform 1/4 | 464 | 1 | 0.985121 | 0.985121 | yes |
| post-asinh transform 2/4 | 464 | 1 | 0.985121 | 0.985121 | yes |
| post-asinh transform 3/4 | 464 | 1 | 0.985121 | 0.985121 | yes |
| post-asinh transform 4/4 | 464 | 1 | 0.985121 | 0.985121 | yes |
| allocation beta | 464 | 1 | 0.985121 | 0.985121 | yes |
| knot refit | 464 | 1 | 0.982207 | 0.982207 | yes |
| covariance shape 1/2 | 464 | 1 | 0.982207 | 0.982207 | yes |
| covariance shape 2/2 | 464 | 1 | 0.982207 | 0.982207 | yes |
| post-asinh transform 1/4 | 464 | 1 | 0.982207 | 0.982207 | yes |
| post-asinh transform 2/4 | 464 | 1 | 0.982207 | 0.982207 | yes |
| post-asinh transform 3/4 | 464 | 1 | 0.982207 | 0.982207 | yes |
| post-asinh transform 4/4 | 464 | 1 | 0.982207 | 0.982207 | yes |
| allocation beta | 464 | 1 | 0.982207 | 0.982207 | yes |
| knot refit | 464 | 1 | 0.982207 | 0.982207 | yes |
| fixed point reached | 464 | 1 | 0.982207 | 0.982207 | yes |
| accepted prune | 464 | 1 | 0.982207 | 0.982207 | yes |
| cold restart 464->459 | 459 | 0.875 | 1.13272 | 1.13272 | no |
| covariance shape 1/2 | 459 | 1 | 1.14897 | 1.14897 | no |
| covariance shape 2/2 | 459 | 1 | 1.14897 | 1.14897 | no |
| post-asinh transform 1/4 | 459 | 1 | 1.14897 | 1.14897 | no |
| post-asinh transform 2/4 | 459 | 1 | 1.14897 | 1.14897 | no |
| post-asinh transform 3/4 | 459 | 1 | 1.14897 | 1.14897 | no |
| post-asinh transform 4/4 | 459 | 1 | 1.14897 | 1.14897 | no |
| allocation beta | 459 | 0.875 | 1.13695 | 1.13695 | no |
| knot refit | 459 | 0.875 | 1.13695 | 1.13695 | no |
| covariance shape 1/2 | 459 | 0.875 | 1.13695 | 1.13695 | no |
| covariance shape 2/2 | 459 | 0.875 | 1.13695 | 1.13695 | no |
| post-asinh transform 1/4 | 459 | 0.875 | 1.13695 | 1.13695 | no |
| post-asinh transform 2/4 | 459 | 0.875 | 1.13695 | 1.13695 | no |
| post-asinh transform 3/4 | 459 | 0.875 | 1.13695 | 1.13695 | no |
| post-asinh transform 4/4 | 459 | 0.875 | 1.13695 | 1.13695 | no |
| allocation beta | 459 | 0.875 | 1.13695 | 1.13695 | no |
| knot refit (cached stationary) | 459 | 0.875 | 1.13695 | 1.13695 | no |
| refined parameter-search resolution | 459 | 0.875 | 1.13695 | 1.13695 | no |
| covariance shape 1/2 | 459 | 0.875 | 1.13695 | 1.13695 | no |
| covariance shape 2/2 | 459 | 0.875 | 1.13695 | 1.13695 | no |
| post-asinh transform 1/4 | 459 | 0.875 | 1.13695 | 1.13695 | no |
| post-asinh transform 2/4 | 459 | 0.875 | 1.13695 | 1.13695 | no |
| post-asinh transform 3/4 | 459 | 0.875 | 1.13695 | 1.13695 | no |
| post-asinh transform 4/4 | 459 | 0.875 | 1.13695 | 1.13695 | no |
| allocation beta | 459 | 0.875 | 1.13695 | 1.13695 | no |
| knot refit (cached stationary) | 459 | 0.875 | 1.13695 | 1.13695 | no |
| refined parameter-search resolution | 459 | 0.875 | 1.13695 | 1.13695 | no |
| covariance shape 1/2 | 459 | 0.875 | 1.13695 | 1.13695 | no |
| covariance shape 2/2 | 459 | 0.875 | 1.13695 | 1.13695 | no |
| post-asinh transform 1/4 | 459 | 0.875 | 1.13272 | 1.13272 | no |
| post-asinh transform 2/4 | 459 | 0.875 | 1.13272 | 1.13272 | no |
| post-asinh transform 3/4 | 459 | 0.875 | 1.13272 | 1.13272 | no |
| post-asinh transform 4/4 | 459 | 0.875 | 1.13272 | 1.13272 | no |
| allocation beta | 459 | 0.875 | 1.13272 | 1.13272 | no |
| knot refit | 459 | 0.875 | 1.13272 | 1.13272 | no |
| covariance shape 1/2 | 459 | 0.875 | 1.13272 | 1.13272 | no |
| covariance shape 2/2 | 459 | 0.875 | 1.13272 | 1.13272 | no |
| post-asinh transform 1/4 | 459 | 0.875 | 1.13272 | 1.13272 | no |
| post-asinh transform 2/4 | 459 | 0.875 | 1.13272 | 1.13272 | no |
| post-asinh transform 3/4 | 459 | 0.875 | 1.13272 | 1.13272 | no |
| post-asinh transform 4/4 | 459 | 0.875 | 1.13272 | 1.13272 | no |
| allocation beta | 459 | 0.875 | 1.13272 | 1.13272 | no |
| knot refit (cached stationary) | 459 | 0.875 | 1.13272 | 1.13272 | no |
| fixed point reached | 459 | 0.875 | 1.13272 | 1.13272 | no |
| prune proposal 464->459 | 459 | 0.96875 | 1.12598 | 1.12598 | no |
| covariance shape 1/2 | 459 | 1 | 1.16263 | 1.16263 | no |
| covariance shape 2/2 | 459 | 1 | 1.16263 | 1.16263 | no |
| post-asinh transform 1/4 | 459 | 1 | 1.16263 | 1.16263 | no |
| post-asinh transform 2/4 | 459 | 1 | 1.16263 | 1.16263 | no |
| post-asinh transform 3/4 | 459 | 1 | 1.16263 | 1.16263 | no |
| post-asinh transform 4/4 | 459 | 1 | 1.16263 | 1.16263 | no |
| allocation beta | 459 | 0.875 | 1.15862 | 1.15862 | no |
| knot refit | 459 | 0.875 | 1.15862 | 1.15862 | no |
| covariance shape 1/2 | 459 | 0.875 | 1.15862 | 1.15862 | no |
| covariance shape 2/2 | 459 | 0.875 | 1.15862 | 1.15862 | no |
| post-asinh transform 1/4 | 459 | 0.875 | 1.15862 | 1.15862 | no |
| post-asinh transform 2/4 | 459 | 0.875 | 1.15862 | 1.15862 | no |
| post-asinh transform 3/4 | 459 | 0.875 | 1.15862 | 1.15862 | no |
| post-asinh transform 4/4 | 459 | 0.875 | 1.15862 | 1.15862 | no |
| allocation beta | 459 | 1 | 1.14586 | 1.14586 | no |
| knot refit | 459 | 1 | 1.13842 | 1.13842 | no |
| covariance shape 1/2 | 459 | 1 | 1.13842 | 1.13842 | no |
| covariance shape 2/2 | 459 | 1 | 1.13842 | 1.13842 | no |
| post-asinh transform 1/4 | 459 | 1 | 1.13842 | 1.13842 | no |
| post-asinh transform 2/4 | 459 | 1 | 1.13842 | 1.13842 | no |
| post-asinh transform 3/4 | 459 | 1 | 1.13842 | 1.13842 | no |
| post-asinh transform 4/4 | 459 | 1 | 1.13842 | 1.13842 | no |
| allocation beta | 459 | 1 | 1.13842 | 1.13842 | no |
| knot refit | 459 | 1 | 1.13842 | 1.13842 | no |
| refined parameter-search resolution | 459 | 1 | 1.13842 | 1.13842 | no |
| covariance shape 1/2 | 459 | 1 | 1.13842 | 1.13842 | no |
| covariance shape 2/2 | 459 | 1 | 1.13842 | 1.13842 | no |
| post-asinh transform 1/4 | 459 | 1 | 1.13842 | 1.13842 | no |
| post-asinh transform 2/4 | 459 | 1 | 1.13842 | 1.13842 | no |
| post-asinh transform 3/4 | 459 | 1 | 1.13842 | 1.13842 | no |
| post-asinh transform 4/4 | 459 | 1 | 1.13842 | 1.13842 | no |
| allocation beta | 459 | 1 | 1.13842 | 1.13842 | no |
| knot refit (cached stationary) | 459 | 1 | 1.13842 | 1.13842 | no |
| refined parameter-search resolution | 459 | 1 | 1.13842 | 1.13842 | no |
| covariance shape 1/2 | 459 | 1 | 1.13842 | 1.13842 | no |
| covariance shape 2/2 | 459 | 1 | 1.13842 | 1.13842 | no |
| post-asinh transform 1/4 | 459 | 1 | 1.13656 | 1.13656 | no |
| post-asinh transform 2/4 | 459 | 1 | 1.13656 | 1.13656 | no |
| post-asinh transform 3/4 | 459 | 1 | 1.13656 | 1.13656 | no |
| post-asinh transform 4/4 | 459 | 1 | 1.13656 | 1.13656 | no |
| allocation beta | 459 | 0.96875 | 1.12598 | 1.12598 | no |
| knot refit | 459 | 0.96875 | 1.12598 | 1.12598 | no |
| covariance shape 1/2 | 459 | 0.96875 | 1.12598 | 1.12598 | no |
| covariance shape 2/2 | 459 | 0.96875 | 1.12598 | 1.12598 | no |
| post-asinh transform 1/4 | 459 | 0.96875 | 1.12598 | 1.12598 | no |
| post-asinh transform 2/4 | 459 | 0.96875 | 1.12598 | 1.12598 | no |
| post-asinh transform 3/4 | 459 | 0.96875 | 1.12598 | 1.12598 | no |
| post-asinh transform 4/4 | 459 | 0.96875 | 1.12598 | 1.12598 | no |
| allocation beta | 459 | 0.96875 | 1.12598 | 1.12598 | no |
| knot refit (cached stationary) | 459 | 0.96875 | 1.12598 | 1.12598 | no |
| fixed point reached | 459 | 0.96875 | 1.12598 | 1.12598 | no |
| rejected prune | 459 | 0.96875 | 1.12598 | 1.12598 | no |
| cold restart 464->462 | 462 | 1 | 1.09252 | 1.09252 | no |
| covariance shape 1/2 | 462 | 1 | 1.13015 | 1.13015 | no |
| covariance shape 2/2 | 462 | 1 | 1.13015 | 1.13015 | no |
| post-asinh transform 1/4 | 462 | 1 | 1.13015 | 1.13015 | no |
| post-asinh transform 2/4 | 462 | 1 | 1.13015 | 1.13015 | no |
| post-asinh transform 3/4 | 462 | 1 | 1.13015 | 1.13015 | no |
| post-asinh transform 4/4 | 462 | 1 | 1.13015 | 1.13015 | no |
| allocation beta | 462 | 1 | 1.13015 | 1.13015 | no |
| knot refit | 462 | 1 | 1.12545 | 1.12545 | no |
| covariance shape 1/2 | 462 | 1 | 1.12545 | 1.12545 | no |
| covariance shape 2/2 | 462 | 1 | 1.12545 | 1.12545 | no |
| post-asinh transform 1/4 | 462 | 1 | 1.12545 | 1.12545 | no |
| post-asinh transform 2/4 | 462 | 1 | 1.12545 | 1.12545 | no |
| post-asinh transform 3/4 | 462 | 1 | 1.12545 | 1.12545 | no |
| post-asinh transform 4/4 | 462 | 1 | 1.12545 | 1.12545 | no |
| allocation beta | 462 | 1 | 1.12545 | 1.12545 | no |
| knot refit | 462 | 1 | 1.11693 | 1.11693 | no |
| covariance shape 1/2 | 462 | 1 | 1.11693 | 1.11693 | no |
| covariance shape 2/2 | 462 | 1 | 1.11693 | 1.11693 | no |
| post-asinh transform 1/4 | 462 | 1 | 1.11693 | 1.11693 | no |
| post-asinh transform 2/4 | 462 | 1 | 1.11693 | 1.11693 | no |
| post-asinh transform 3/4 | 462 | 1 | 1.11693 | 1.11693 | no |
| post-asinh transform 4/4 | 462 | 1 | 1.11693 | 1.11693 | no |
| allocation beta | 462 | 1 | 1.11693 | 1.11693 | no |
| knot refit | 462 | 1 | 1.10404 | 1.10404 | no |
| covariance shape 1/2 | 462 | 1 | 1.10404 | 1.10404 | no |
| covariance shape 2/2 | 462 | 1 | 1.10404 | 1.10404 | no |
| post-asinh transform 1/4 | 462 | 1 | 1.10404 | 1.10404 | no |
| post-asinh transform 2/4 | 462 | 1 | 1.10404 | 1.10404 | no |
| post-asinh transform 3/4 | 462 | 1 | 1.10404 | 1.10404 | no |
| post-asinh transform 4/4 | 462 | 1 | 1.10404 | 1.10404 | no |
| allocation beta | 462 | 1 | 1.10404 | 1.10404 | no |
| knot refit | 462 | 1 | 1.10404 | 1.10404 | no |
| refined parameter-search resolution | 462 | 1 | 1.10404 | 1.10404 | no |
| covariance shape 1/2 | 462 | 1 | 1.10404 | 1.10404 | no |
| covariance shape 2/2 | 462 | 1 | 1.10404 | 1.10404 | no |
| post-asinh transform 1/4 | 462 | 1 | 1.10404 | 1.10404 | no |
| post-asinh transform 2/4 | 462 | 1 | 1.10404 | 1.10404 | no |
| post-asinh transform 3/4 | 462 | 1 | 1.10404 | 1.10404 | no |
| post-asinh transform 4/4 | 462 | 1 | 1.10404 | 1.10404 | no |
| allocation beta | 462 | 1 | 1.10404 | 1.10404 | no |
| knot refit (cached stationary) | 462 | 1 | 1.10404 | 1.10404 | no |
| refined parameter-search resolution | 462 | 1 | 1.10404 | 1.10404 | no |
| covariance shape 1/2 | 462 | 1 | 1.10404 | 1.10404 | no |
| covariance shape 2/2 | 462 | 1 | 1.10404 | 1.10404 | no |
| post-asinh transform 1/4 | 462 | 1 | 1.09252 | 1.09252 | no |
| post-asinh transform 2/4 | 462 | 1 | 1.09252 | 1.09252 | no |
| post-asinh transform 3/4 | 462 | 1 | 1.09252 | 1.09252 | no |
| post-asinh transform 4/4 | 462 | 1 | 1.09252 | 1.09252 | no |
| allocation beta | 462 | 1 | 1.09252 | 1.09252 | no |
| knot refit | 462 | 1 | 1.09252 | 1.09252 | no |
| covariance shape 1/2 | 462 | 1 | 1.09252 | 1.09252 | no |
| covariance shape 2/2 | 462 | 1 | 1.09252 | 1.09252 | no |
| post-asinh transform 1/4 | 462 | 1 | 1.09252 | 1.09252 | no |
| post-asinh transform 2/4 | 462 | 1 | 1.09252 | 1.09252 | no |
| post-asinh transform 3/4 | 462 | 1 | 1.09252 | 1.09252 | no |
| post-asinh transform 4/4 | 462 | 1 | 1.09252 | 1.09252 | no |
| allocation beta | 462 | 1 | 1.09252 | 1.09252 | no |
| knot refit (cached stationary) | 462 | 1 | 1.09252 | 1.09252 | no |
| fixed point reached | 462 | 1 | 1.09252 | 1.09252 | no |
| prune proposal 464->462 | 462 | 1 | 1.02699 | 1.02699 | no |
| covariance shape 1/2 | 462 | 1 | 1.03757 | 1.03757 | no |
| covariance shape 2/2 | 462 | 1 | 1.03757 | 1.03757 | no |
| post-asinh transform 1/4 | 462 | 1 | 1.03757 | 1.03757 | no |
| post-asinh transform 2/4 | 462 | 1 | 1.03757 | 1.03757 | no |
| post-asinh transform 3/4 | 462 | 1 | 1.03757 | 1.03757 | no |
| post-asinh transform 4/4 | 462 | 1 | 1.03757 | 1.03757 | no |
| allocation beta | 462 | 1 | 1.03757 | 1.03757 | no |
| knot refit | 462 | 1 | 1.02699 | 1.02699 | no |
| covariance shape 1/2 | 462 | 1 | 1.02699 | 1.02699 | no |
| covariance shape 2/2 | 462 | 1 | 1.02699 | 1.02699 | no |
| post-asinh transform 1/4 | 462 | 1 | 1.02699 | 1.02699 | no |
| post-asinh transform 2/4 | 462 | 1 | 1.02699 | 1.02699 | no |
| post-asinh transform 3/4 | 462 | 1 | 1.02699 | 1.02699 | no |
| post-asinh transform 4/4 | 462 | 1 | 1.02699 | 1.02699 | no |
| allocation beta | 462 | 1 | 1.02699 | 1.02699 | no |
| knot refit | 462 | 1 | 1.02699 | 1.02699 | no |
| refined parameter-search resolution | 462 | 1 | 1.02699 | 1.02699 | no |
| covariance shape 1/2 | 462 | 1 | 1.02699 | 1.02699 | no |
| covariance shape 2/2 | 462 | 1 | 1.02699 | 1.02699 | no |
| post-asinh transform 1/4 | 462 | 1 | 1.02699 | 1.02699 | no |
| post-asinh transform 2/4 | 462 | 1 | 1.02699 | 1.02699 | no |
| post-asinh transform 3/4 | 462 | 1 | 1.02699 | 1.02699 | no |
| post-asinh transform 4/4 | 462 | 1 | 1.02699 | 1.02699 | no |
| allocation beta | 462 | 1 | 1.02699 | 1.02699 | no |
| knot refit (cached stationary) | 462 | 1 | 1.02699 | 1.02699 | no |
| refined parameter-search resolution | 462 | 1 | 1.02699 | 1.02699 | no |
| covariance shape 1/2 | 462 | 1 | 1.02699 | 1.02699 | no |
| covariance shape 2/2 | 462 | 1 | 1.02699 | 1.02699 | no |
| post-asinh transform 1/4 | 462 | 1 | 1.02699 | 1.02699 | no |
| post-asinh transform 2/4 | 462 | 1 | 1.02699 | 1.02699 | no |
| post-asinh transform 3/4 | 462 | 1 | 1.02699 | 1.02699 | no |
| post-asinh transform 4/4 | 462 | 1 | 1.02699 | 1.02699 | no |
| allocation beta | 462 | 1 | 1.02699 | 1.02699 | no |
| knot refit (cached stationary) | 462 | 1 | 1.02699 | 1.02699 | no |
| fixed point reached | 462 | 1 | 1.02699 | 1.02699 | no |
| rejected prune | 462 | 1 | 1.02699 | 1.02699 | no |
| cold restart 464->463 | 463 | 1 | 1.08137 | 1.08137 | no |
| covariance shape 1/2 | 463 | 1 | 1.16081 | 1.16081 | no |
| covariance shape 2/2 | 463 | 1 | 1.16081 | 1.16081 | no |
| post-asinh transform 1/4 | 463 | 1 | 1.16081 | 1.16081 | no |
| post-asinh transform 2/4 | 463 | 1 | 1.16081 | 1.16081 | no |
| post-asinh transform 3/4 | 463 | 1 | 1.16081 | 1.16081 | no |
| post-asinh transform 4/4 | 463 | 1 | 1.16081 | 1.16081 | no |
| allocation beta | 463 | 0.875 | 1.15003 | 1.15003 | no |
| knot refit | 463 | 0.875 | 1.13956 | 1.13956 | no |
| covariance shape 1/2 | 463 | 0.875 | 1.13956 | 1.13956 | no |
| covariance shape 2/2 | 463 | 0.875 | 1.13956 | 1.13956 | no |
| post-asinh transform 1/4 | 463 | 0.875 | 1.13956 | 1.13956 | no |
| post-asinh transform 2/4 | 463 | 0.875 | 1.13956 | 1.13956 | no |
| post-asinh transform 3/4 | 463 | 0.875 | 1.13956 | 1.13956 | no |
| post-asinh transform 4/4 | 463 | 0.875 | 1.13956 | 1.13956 | no |
| allocation beta | 463 | 1 | 1.13023 | 1.13023 | no |
| knot refit | 463 | 1 | 1.11587 | 1.11587 | no |
| covariance shape 1/2 | 463 | 1 | 1.11587 | 1.11587 | no |
| covariance shape 2/2 | 463 | 1 | 1.11587 | 1.11587 | no |
| post-asinh transform 1/4 | 463 | 1 | 1.11587 | 1.11587 | no |
| post-asinh transform 2/4 | 463 | 1 | 1.11587 | 1.11587 | no |
| post-asinh transform 3/4 | 463 | 1 | 1.11587 | 1.11587 | no |
| post-asinh transform 4/4 | 463 | 1 | 1.11587 | 1.11587 | no |
| allocation beta | 463 | 1 | 1.11587 | 1.11587 | no |
| knot refit | 463 | 1 | 1.11108 | 1.11108 | no |
| covariance shape 1/2 | 463 | 1 | 1.11108 | 1.11108 | no |
| covariance shape 2/2 | 463 | 1 | 1.11108 | 1.11108 | no |
| post-asinh transform 1/4 | 463 | 1 | 1.11108 | 1.11108 | no |
| post-asinh transform 2/4 | 463 | 1 | 1.11108 | 1.11108 | no |
| post-asinh transform 3/4 | 463 | 1 | 1.11108 | 1.11108 | no |
| post-asinh transform 4/4 | 463 | 1 | 1.11108 | 1.11108 | no |
| allocation beta | 463 | 1 | 1.11108 | 1.11108 | no |
| knot refit | 463 | 1 | 1.09426 | 1.09426 | no |
| covariance shape 1/2 | 463 | 1 | 1.09426 | 1.09426 | no |
| covariance shape 2/2 | 463 | 1 | 1.09426 | 1.09426 | no |
| post-asinh transform 1/4 | 463 | 1 | 1.09426 | 1.09426 | no |
| post-asinh transform 2/4 | 463 | 1 | 1.09426 | 1.09426 | no |
| post-asinh transform 3/4 | 463 | 1 | 1.09426 | 1.09426 | no |
| post-asinh transform 4/4 | 463 | 1 | 1.09426 | 1.09426 | no |
| allocation beta | 463 | 1 | 1.09426 | 1.09426 | no |
| knot refit | 463 | 1 | 1.08471 | 1.08471 | no |
| covariance shape 1/2 | 463 | 1 | 1.08471 | 1.08471 | no |
| covariance shape 2/2 | 463 | 1 | 1.08471 | 1.08471 | no |
| post-asinh transform 1/4 | 463 | 1 | 1.08471 | 1.08471 | no |
| post-asinh transform 2/4 | 463 | 1 | 1.08471 | 1.08471 | no |
| post-asinh transform 3/4 | 463 | 1 | 1.08471 | 1.08471 | no |
| post-asinh transform 4/4 | 463 | 1 | 1.08471 | 1.08471 | no |
| allocation beta | 463 | 1 | 1.08471 | 1.08471 | no |
| knot refit | 463 | 1 | 1.08137 | 1.08137 | no |
| covariance shape 1/2 | 463 | 1 | 1.08137 | 1.08137 | no |
| covariance shape 2/2 | 463 | 1 | 1.08137 | 1.08137 | no |
| post-asinh transform 1/4 | 463 | 1 | 1.08137 | 1.08137 | no |
| post-asinh transform 2/4 | 463 | 1 | 1.08137 | 1.08137 | no |
| post-asinh transform 3/4 | 463 | 1 | 1.08137 | 1.08137 | no |
| post-asinh transform 4/4 | 463 | 1 | 1.08137 | 1.08137 | no |
| allocation beta | 463 | 1 | 1.08137 | 1.08137 | no |
| knot refit | 463 | 1 | 1.08137 | 1.08137 | no |
| refined parameter-search resolution | 463 | 1 | 1.08137 | 1.08137 | no |
| covariance shape 1/2 | 463 | 1 | 1.08137 | 1.08137 | no |
| covariance shape 2/2 | 463 | 1 | 1.08137 | 1.08137 | no |
| post-asinh transform 1/4 | 463 | 1 | 1.08137 | 1.08137 | no |
| post-asinh transform 2/4 | 463 | 1 | 1.08137 | 1.08137 | no |
| post-asinh transform 3/4 | 463 | 1 | 1.08137 | 1.08137 | no |
| post-asinh transform 4/4 | 463 | 1 | 1.08137 | 1.08137 | no |
| allocation beta | 463 | 1 | 1.08137 | 1.08137 | no |
| knot refit (cached stationary) | 463 | 1 | 1.08137 | 1.08137 | no |
| refined parameter-search resolution | 463 | 1 | 1.08137 | 1.08137 | no |
| covariance shape 1/2 | 463 | 1 | 1.08137 | 1.08137 | no |
| covariance shape 2/2 | 463 | 1 | 1.08137 | 1.08137 | no |
| post-asinh transform 1/4 | 463 | 1 | 1.08137 | 1.08137 | no |
| post-asinh transform 2/4 | 463 | 1 | 1.08137 | 1.08137 | no |
| post-asinh transform 3/4 | 463 | 1 | 1.08137 | 1.08137 | no |
| post-asinh transform 4/4 | 463 | 1 | 1.08137 | 1.08137 | no |
| allocation beta | 463 | 1 | 1.08137 | 1.08137 | no |
| knot refit (cached stationary) | 463 | 1 | 1.08137 | 1.08137 | no |
| fixed point reached | 463 | 1 | 1.08137 | 1.08137 | no |
| prune proposal 464->463 | 463 | 0.96875 | 1.01533 | 1.01533 | no |
| covariance shape 1/2 | 463 | 1 | 1.03565 | 1.03565 | no |
| covariance shape 2/2 | 463 | 1 | 1.03565 | 1.03565 | no |
| post-asinh transform 1/4 | 463 | 1 | 1.03565 | 1.03565 | no |
| post-asinh transform 2/4 | 463 | 1 | 1.03565 | 1.03565 | no |
| post-asinh transform 3/4 | 463 | 1 | 1.03565 | 1.03565 | no |
| post-asinh transform 4/4 | 463 | 1 | 1.03565 | 1.03565 | no |
| allocation beta | 463 | 1 | 1.03565 | 1.03565 | no |
| knot refit | 463 | 1 | 1.0265 | 1.0265 | no |
| covariance shape 1/2 | 463 | 1 | 1.0265 | 1.0265 | no |
| covariance shape 2/2 | 463 | 1 | 1.0265 | 1.0265 | no |
| post-asinh transform 1/4 | 463 | 1 | 1.0265 | 1.0265 | no |
| post-asinh transform 2/4 | 463 | 1 | 1.0265 | 1.0265 | no |
| post-asinh transform 3/4 | 463 | 1 | 1.0265 | 1.0265 | no |
| post-asinh transform 4/4 | 463 | 1 | 1.0265 | 1.0265 | no |
| allocation beta | 463 | 1 | 1.0265 | 1.0265 | no |
| knot refit | 463 | 1 | 1.02351 | 1.02351 | no |
| covariance shape 1/2 | 463 | 1 | 1.02351 | 1.02351 | no |
| covariance shape 2/2 | 463 | 1 | 1.02351 | 1.02351 | no |
| post-asinh transform 1/4 | 463 | 1 | 1.02351 | 1.02351 | no |
| post-asinh transform 2/4 | 463 | 1 | 1.02351 | 1.02351 | no |
| post-asinh transform 3/4 | 463 | 1 | 1.02351 | 1.02351 | no |
| post-asinh transform 4/4 | 463 | 1 | 1.02351 | 1.02351 | no |
| allocation beta | 463 | 1 | 1.02351 | 1.02351 | no |
| knot refit | 463 | 1 | 1.02082 | 1.02082 | no |
| covariance shape 1/2 | 463 | 1 | 1.02082 | 1.02082 | no |
| covariance shape 2/2 | 463 | 1 | 1.02082 | 1.02082 | no |
| post-asinh transform 1/4 | 463 | 1 | 1.02082 | 1.02082 | no |
| post-asinh transform 2/4 | 463 | 1 | 1.02082 | 1.02082 | no |
| post-asinh transform 3/4 | 463 | 1 | 1.02082 | 1.02082 | no |
| post-asinh transform 4/4 | 463 | 1 | 1.02082 | 1.02082 | no |
| allocation beta | 463 | 1 | 1.02082 | 1.02082 | no |
| knot refit | 463 | 1 | 1.02082 | 1.02082 | no |
| refined parameter-search resolution | 463 | 1 | 1.02082 | 1.02082 | no |
| covariance shape 1/2 | 463 | 1 | 1.02082 | 1.02082 | no |
| covariance shape 2/2 | 463 | 1 | 1.02082 | 1.02082 | no |
| post-asinh transform 1/4 | 463 | 1 | 1.02082 | 1.02082 | no |
| post-asinh transform 2/4 | 463 | 1 | 1.02082 | 1.02082 | no |
| post-asinh transform 3/4 | 463 | 1 | 1.02082 | 1.02082 | no |
| post-asinh transform 4/4 | 463 | 1 | 1.02082 | 1.02082 | no |
| allocation beta | 463 | 1 | 1.02082 | 1.02082 | no |
| knot refit (cached stationary) | 463 | 1 | 1.02082 | 1.02082 | no |
| refined parameter-search resolution | 463 | 1 | 1.02082 | 1.02082 | no |
| covariance shape 1/2 | 463 | 1 | 1.02082 | 1.02082 | no |
| covariance shape 2/2 | 463 | 1 | 1.02082 | 1.02082 | no |
| post-asinh transform 1/4 | 463 | 1 | 1.01953 | 1.01953 | no |
| post-asinh transform 2/4 | 463 | 1 | 1.01953 | 1.01953 | no |
| post-asinh transform 3/4 | 463 | 1 | 1.01953 | 1.01953 | no |
| post-asinh transform 4/4 | 463 | 1 | 1.01953 | 1.01953 | no |
| allocation beta | 463 | 0.96875 | 1.01533 | 1.01533 | no |
| knot refit | 463 | 0.96875 | 1.01533 | 1.01533 | no |
| covariance shape 1/2 | 463 | 0.96875 | 1.01533 | 1.01533 | no |
| covariance shape 2/2 | 463 | 0.96875 | 1.01533 | 1.01533 | no |
| post-asinh transform 1/4 | 463 | 0.96875 | 1.01533 | 1.01533 | no |
| post-asinh transform 2/4 | 463 | 0.96875 | 1.01533 | 1.01533 | no |
| post-asinh transform 3/4 | 463 | 0.96875 | 1.01533 | 1.01533 | no |
| post-asinh transform 4/4 | 463 | 0.96875 | 1.01533 | 1.01533 | no |
| allocation beta | 463 | 0.96875 | 1.01533 | 1.01533 | no |
| knot refit (cached stationary) | 463 | 0.96875 | 1.01533 | 1.01533 | no |
| fixed point reached | 463 | 0.96875 | 1.01533 | 1.01533 | no |
| rejected prune | 463 | 0.96875 | 1.01533 | 1.01533 | no |
| high-fidelity warm 464->348 | 348 | 1 | 2.80637 | 2.80637 | no |
| high-fidelity cold 1/4 464->348 | 348 | 1 | 1.90723 | 1.90723 | no |
| high-fidelity cold 2/4 464->348 | 348 | 1 | 1.93844 | 1.93844 | no |
| high-fidelity cold 3/4 464->348 | 348 | 1 | 2.67652 | 2.67652 | no |
| high-fidelity cold 4/4 464->348 | 348 | 1 | 1.90864 | 1.90864 | no |
| high-fidelity proposal 464->348 | 348 | 1 | 1.90723 | 1.90723 | no |
| rejected high-fidelity prune | 348 | 1 | 1.90723 | 1.90723 | no |
| high-fidelity warm 464->406 | 406 | 1 | 2.55056 | 2.55056 | no |
| high-fidelity cold 1/4 464->406 | 406 | 1 | 1.61991 | 1.61991 | no |
| high-fidelity cold 2/4 464->406 | 406 | 1 | 1.57981 | 1.57981 | no |
| high-fidelity cold 3/4 464->406 | 406 | 1 | 1.68763 | 1.68763 | no |
| high-fidelity cold 4/4 464->406 | 406 | 1 | 1.65217 | 1.65217 | no |
| high-fidelity proposal 464->406 | 406 | 1 | 1.57981 | 1.57981 | no |
| rejected high-fidelity prune | 406 | 1 | 1.57981 | 1.57981 | no |
| high-fidelity warm 464->435 | 435 | 1 | 2.53734 | 2.53734 | no |
| high-fidelity cold 1/4 464->435 | 435 | 1 | 1.59374 | 1.59374 | no |
| high-fidelity cold 2/4 464->435 | 435 | 1 | 1.79325 | 1.79325 | no |
| high-fidelity cold 3/4 464->435 | 435 | 1 | 1.6203 | 1.6203 | no |
| high-fidelity cold 4/4 464->435 | 435 | 1 | 1.63215 | 1.63215 | no |
| high-fidelity proposal 464->435 | 435 | 1 | 1.59374 | 1.59374 | no |
| rejected high-fidelity prune | 435 | 1 | 1.59374 | 1.59374 | no |
| high-fidelity warm 464->450 | 450 | 1 | 2.52757 | 2.52757 | no |
| high-fidelity cold 1/4 464->450 | 450 | 1 | 1.53784 | 1.53784 | no |
| high-fidelity cold 2/4 464->450 | 450 | 1 | 1.57404 | 1.57404 | no |
| high-fidelity cold 3/4 464->450 | 450 | 1 | 1.47679 | 1.47679 | no |
| high-fidelity cold 4/4 464->450 | 450 | 1 | 1.31262 | 1.31262 | no |
| high-fidelity proposal 464->450 | 450 | 1 | 1.31262 | 1.31262 | no |
| rejected high-fidelity prune | 450 | 1 | 1.31262 | 1.31262 | no |
| high-fidelity warm 464->457 | 457 | 1 | 1.06796 | 1.06796 | no |
| high-fidelity cold 1/4 464->457 | 457 | 1 | 1.46922 | 1.46922 | no |
| high-fidelity cold 2/4 464->457 | 457 | 1 | 1.48108 | 1.48108 | no |
| high-fidelity cold 3/4 464->457 | 457 | 1 | 1.34934 | 1.34934 | no |
| high-fidelity cold 4/4 464->457 | 457 | 1 | 1.26803 | 1.26803 | no |
| high-fidelity proposal 464->457 | 457 | 1 | 1.06796 | 1.06796 | no |
| rejected high-fidelity prune | 457 | 1 | 1.06796 | 1.06796 | no |
| high-fidelity warm 464->461 | 461 | 1 | 0.961796 | 0.961796 | yes |
| high-fidelity proposal 464->461 | 461 | 1 | 0.961796 | 0.961796 | yes |
| accepted high-fidelity prune | 461 | 1 | 0.961796 | 0.961796 | yes |
| high-fidelity warm 461->458 | 458 | 1 | 1.08694 | 1.08694 | no |
| high-fidelity cold 1/4 461->458 | 458 | 1 | 1.50307 | 1.50307 | no |
| high-fidelity cold 2/4 461->458 | 458 | 1 | 1.38049 | 1.38049 | no |
| high-fidelity cold 3/4 461->458 | 458 | 1 | 1.42827 | 1.42827 | no |
| high-fidelity cold 4/4 461->458 | 458 | 1 | 1.32455 | 1.32455 | no |
| high-fidelity proposal 461->458 | 458 | 1 | 1.08694 | 1.08694 | no |
| rejected high-fidelity prune | 458 | 1 | 1.08694 | 1.08694 | no |
| high-fidelity warm 461->460 | 460 | 1 | 0.973626 | 0.973626 | yes |
| high-fidelity proposal 461->460 | 460 | 1 | 0.973626 | 0.973626 | yes |
| accepted high-fidelity prune | 460 | 1 | 0.973626 | 0.973626 | yes |
| high-fidelity warm 460->459 | 459 | 1 | 1.02881 | 1.02881 | no |
| high-fidelity cold 1/4 460->459 | 459 | 1 | 1.35092 | 1.35092 | no |
| high-fidelity cold 2/4 460->459 | 459 | 1 | 1.46619 | 1.46619 | no |
| high-fidelity cold 3/4 460->459 | 459 | 1 | 1.42568 | 1.42568 | no |
| high-fidelity cold 4/4 460->459 | 459 | 1 | 1.41364 | 1.41364 | no |
| high-fidelity proposal 460->459 | 459 | 1 | 1.02881 | 1.02881 | no |
| rejected high-fidelity prune | 459 | 1 | 1.02881 | 1.02881 | no |

Covariance adjustment from empirical whitening: `[[1.0,0.0],[0.0,1.0]]`

Post-asinh matrix: `[[3.120450536670439,0.0],[0.0,3.1837566165838824]]`

## 3D

Initial: 32,768 points, beta 1, active score 0.211325, joint objective 0.211325.

Final: **961 points**, beta **1**, active score **0.95985**, joint objective **0.95985**, passes.

### Fixed-point checks

| points | sweeps | converged | covariance step | transform step | beta step |
|---:|---:|---:|---:|---:|---:|
| 32,768 | 5 | yes | 0.0125 | 0.01125 | 0.025 |
| 28,672 | 6 | yes | 0.0125 | 0.01125 | 0.025 |
| 24,576 | 5 | yes | 0.0125 | 0.01125 | 0.025 |
| 20,480 | 5 | yes | 0.0125 | 0.01125 | 0.025 |
| 16,384 | 5 | yes | 0.0125 | 0.01125 | 0.025 |
| 12,288 | 3 | yes | 0.0125 | 0.01125 | 0.025 |
| 9,216 | 4 | yes | 0.0125 | 0.01125 | 0.025 |
| 6,912 | 5 | yes | 0.0125 | 0.01125 | 0.025 |
| 5,184 | 4 | yes | 0.0125 | 0.01125 | 0.025 |
| 3,888 | 5 | yes | 0.0125 | 0.01125 | 0.025 |
| 2,916 | 6 | yes | 0.0125 | 0.01125 | 0.025 |
| 2,916 | 9 | yes | 0.0125 | 0.01125 | 0.025 |
| 2,187 | 6 | yes | 0.0125 | 0.01125 | 0.025 |
| 2,187 | 10 | yes | 0.0125 | 0.01125 | 0.025 |
| 2,552 | 5 | yes | 0.0125 | 0.01125 | 0.025 |
| 2,552 | 5 | yes | 0.0125 | 0.01125 | 0.025 |
| 2,188 | 6 | yes | 0.0125 | 0.01125 | 0.025 |
| 2,188 | 5 | yes | 0.0125 | 0.01125 | 0.025 |
| 2,370 | 4 | yes | 0.0125 | 0.01125 | 0.025 |
| 2,188 | 5 | yes | 0.0125 | 0.01125 | 0.025 |
| 2,188 | 8 | yes | 0.0125 | 0.01125 | 0.025 |
| 2,279 | 5 | yes | 0.0125 | 0.01125 | 0.025 |
| 2,188 | 5 | yes | 0.0125 | 0.01125 | 0.025 |
| 2,188 | 10 | yes | 0.0125 | 0.01125 | 0.025 |
| 2,234 | 3 | yes | 0.0125 | 0.01125 | 0.025 |
| 2,234 | 8 | yes | 0.0125 | 0.01125 | 0.025 |
| 2,257 | 5 | yes | 0.0125 | 0.01125 | 0.025 |
| 2,257 | 5 | yes | 0.0125 | 0.01125 | 0.025 |
| 2,268 | 7 | yes | 0.0125 | 0.01125 | 0.025 |
| 2,257 | 3 | yes | 0.0125 | 0.01125 | 0.025 |
| 2,257 | 5 | yes | 0.0125 | 0.01125 | 0.025 |
| 2,263 | 3 | yes | 0.0125 | 0.01125 | 0.025 |
| 2,263 | 6 | yes | 0.0125 | 0.01125 | 0.025 |
| 2,266 | 6 | yes | 0.0125 | 0.01125 | 0.025 |
| 2,266 | 4 | yes | 0.0125 | 0.01125 | 0.025 |
| 2,267 | 3 | yes | 0.0125 | 0.01125 | 0.025 |
| 2,266 | 6 | yes | 0.0125 | 0.01125 | 0.025 |
| 2,266 | 7 | yes | 0.0125 | 0.01125 | 0.025 |

| metric | ratio to 1D32 | active | passes |
|---|---:|---:|---:|
| density JS/d | 0.95985 | yes | yes |
| r2 conditional mean | 10.2484 | no | no |
| r2 conditional median | 4.20829 | no | no |
| r3 conditional mean | 4.03983 | no | no |
| r3 conditional median | 2.6827 | no | no |

### Accepted/rejected iteration trace

| stage | points | beta | objective | active score | passes |
|---|---:|---:|---:|---:|---:|
| stored verified initial | 32,768 | 1 | 0.211325 | 0.211325 | yes |
| common-fidelity warm refit | 32,768 | 1 | 0.112561 | 0.112561 | yes |
| covariance shape 1/5 | 32,768 | 1 | 0.112561 | 0.112561 | yes |
| covariance shape 2/5 | 32,768 | 1 | 0.112561 | 0.112561 | yes |
| covariance shape 3/5 | 32,768 | 1 | 0.112561 | 0.112561 | yes |
| covariance shape 4/5 | 32,768 | 1 | 0.112561 | 0.112561 | yes |
| covariance shape 5/5 | 32,768 | 1 | 0.112561 | 0.112561 | yes |
| post-asinh transform 1/9 | 32,768 | 1 | 0.112561 | 0.112561 | yes |
| post-asinh transform 2/9 | 32,768 | 1 | 0.112561 | 0.112561 | yes |
| post-asinh transform 3/9 | 32,768 | 1 | 0.112561 | 0.112561 | yes |
| post-asinh transform 4/9 | 32,768 | 1 | 0.112561 | 0.112561 | yes |
| post-asinh transform 5/9 | 32,768 | 1 | 0.112561 | 0.112561 | yes |
| post-asinh transform 6/9 | 32,768 | 1 | 0.112561 | 0.112561 | yes |
| post-asinh transform 7/9 | 32,768 | 1 | 0.112561 | 0.112561 | yes |
| post-asinh transform 8/9 | 32,768 | 1 | 0.112561 | 0.112561 | yes |
| post-asinh transform 9/9 | 32,768 | 1 | 0.112561 | 0.112561 | yes |
| allocation beta | 32,768 | 0.9 | 0.109973 | 0.109973 | yes |
| knot refit | 32,768 | 0.9 | 0.109973 | 0.109973 | yes |
| covariance shape 1/5 | 32,768 | 0.9 | 0.109973 | 0.109973 | yes |
| covariance shape 2/5 | 32,768 | 0.9 | 0.109973 | 0.109973 | yes |
| covariance shape 3/5 | 32,768 | 0.9 | 0.109973 | 0.109973 | yes |
| covariance shape 4/5 | 32,768 | 0.9 | 0.109973 | 0.109973 | yes |
| covariance shape 5/5 | 32,768 | 0.9 | 0.109973 | 0.109973 | yes |
| post-asinh transform 1/9 | 32,768 | 0.9 | 0.109973 | 0.109973 | yes |
| post-asinh transform 2/9 | 32,768 | 0.9 | 0.109973 | 0.109973 | yes |
| post-asinh transform 3/9 | 32,768 | 0.9 | 0.109973 | 0.109973 | yes |
| post-asinh transform 4/9 | 32,768 | 0.9 | 0.109973 | 0.109973 | yes |
| post-asinh transform 5/9 | 32,768 | 0.9 | 0.109973 | 0.109973 | yes |
| post-asinh transform 6/9 | 32,768 | 0.9 | 0.109973 | 0.109973 | yes |
| post-asinh transform 7/9 | 32,768 | 0.9 | 0.109973 | 0.109973 | yes |
| post-asinh transform 8/9 | 32,768 | 0.9 | 0.109973 | 0.109973 | yes |
| post-asinh transform 9/9 | 32,768 | 0.9 | 0.109973 | 0.109973 | yes |
| allocation beta | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| knot refit | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| covariance shape 1/5 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| covariance shape 2/5 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| covariance shape 3/5 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| covariance shape 4/5 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| covariance shape 5/5 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| post-asinh transform 1/9 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| post-asinh transform 2/9 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| post-asinh transform 3/9 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| post-asinh transform 4/9 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| post-asinh transform 5/9 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| post-asinh transform 6/9 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| post-asinh transform 7/9 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| post-asinh transform 8/9 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| post-asinh transform 9/9 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| allocation beta | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| knot refit (cached stationary) | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| refined parameter-search resolution | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| covariance shape 1/5 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| covariance shape 2/5 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| covariance shape 3/5 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| covariance shape 4/5 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| covariance shape 5/5 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| post-asinh transform 1/9 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| post-asinh transform 2/9 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| post-asinh transform 3/9 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| post-asinh transform 4/9 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| post-asinh transform 5/9 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| post-asinh transform 6/9 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| post-asinh transform 7/9 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| post-asinh transform 8/9 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| post-asinh transform 9/9 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| allocation beta | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| knot refit (cached stationary) | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| refined parameter-search resolution | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| covariance shape 1/5 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| covariance shape 2/5 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| covariance shape 3/5 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| covariance shape 4/5 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| covariance shape 5/5 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| post-asinh transform 1/9 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| post-asinh transform 2/9 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| post-asinh transform 3/9 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| post-asinh transform 4/9 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| post-asinh transform 5/9 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| post-asinh transform 6/9 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| post-asinh transform 7/9 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| post-asinh transform 8/9 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| post-asinh transform 9/9 | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| allocation beta | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| knot refit (cached stationary) | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| fixed point reached | 32,768 | 0.8 | 0.10976 | 0.10976 | yes |
| prune proposal 32768->28672 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| covariance shape 1/5 | 28,672 | 0.8 | 0.310777 | 0.310777 | yes |
| covariance shape 2/5 | 28,672 | 0.8 | 0.310777 | 0.310777 | yes |
| covariance shape 3/5 | 28,672 | 0.8 | 0.310777 | 0.310777 | yes |
| covariance shape 4/5 | 28,672 | 0.8 | 0.310777 | 0.310777 | yes |
| covariance shape 5/5 | 28,672 | 0.8 | 0.310777 | 0.310777 | yes |
| post-asinh transform 1/9 | 28,672 | 0.8 | 0.310777 | 0.310777 | yes |
| post-asinh transform 2/9 | 28,672 | 0.8 | 0.310777 | 0.310777 | yes |
| post-asinh transform 3/9 | 28,672 | 0.8 | 0.310777 | 0.310777 | yes |
| post-asinh transform 4/9 | 28,672 | 0.8 | 0.310777 | 0.310777 | yes |
| post-asinh transform 5/9 | 28,672 | 0.8 | 0.310777 | 0.310777 | yes |
| post-asinh transform 6/9 | 28,672 | 0.8 | 0.310777 | 0.310777 | yes |
| post-asinh transform 7/9 | 28,672 | 0.8 | 0.310777 | 0.310777 | yes |
| post-asinh transform 8/9 | 28,672 | 0.8 | 0.310777 | 0.310777 | yes |
| post-asinh transform 9/9 | 28,672 | 0.8 | 0.310777 | 0.310777 | yes |
| allocation beta | 28,672 | 0.9 | 0.197801 | 0.197801 | yes |
| knot refit | 28,672 | 0.9 | 0.141351 | 0.141351 | yes |
| covariance shape 1/5 | 28,672 | 0.9 | 0.141351 | 0.141351 | yes |
| covariance shape 2/5 | 28,672 | 0.9 | 0.141351 | 0.141351 | yes |
| covariance shape 3/5 | 28,672 | 0.9 | 0.141351 | 0.141351 | yes |
| covariance shape 4/5 | 28,672 | 0.9 | 0.141351 | 0.141351 | yes |
| covariance shape 5/5 | 28,672 | 0.9 | 0.141351 | 0.141351 | yes |
| post-asinh transform 1/9 | 28,672 | 0.9 | 0.141351 | 0.141351 | yes |
| post-asinh transform 2/9 | 28,672 | 0.9 | 0.141351 | 0.141351 | yes |
| post-asinh transform 3/9 | 28,672 | 0.9 | 0.141351 | 0.141351 | yes |
| post-asinh transform 4/9 | 28,672 | 0.9 | 0.141351 | 0.141351 | yes |
| post-asinh transform 5/9 | 28,672 | 0.9 | 0.141351 | 0.141351 | yes |
| post-asinh transform 6/9 | 28,672 | 0.9 | 0.141351 | 0.141351 | yes |
| post-asinh transform 7/9 | 28,672 | 0.9 | 0.141351 | 0.141351 | yes |
| post-asinh transform 8/9 | 28,672 | 0.9 | 0.141351 | 0.141351 | yes |
| post-asinh transform 9/9 | 28,672 | 0.9 | 0.141351 | 0.141351 | yes |
| allocation beta | 28,672 | 1 | 0.124283 | 0.124283 | yes |
| knot refit | 28,672 | 1 | 0.117682 | 0.117682 | yes |
| covariance shape 1/5 | 28,672 | 1 | 0.117682 | 0.117682 | yes |
| covariance shape 2/5 | 28,672 | 1 | 0.117682 | 0.117682 | yes |
| covariance shape 3/5 | 28,672 | 1 | 0.117682 | 0.117682 | yes |
| covariance shape 4/5 | 28,672 | 1 | 0.117682 | 0.117682 | yes |
| covariance shape 5/5 | 28,672 | 1 | 0.117682 | 0.117682 | yes |
| post-asinh transform 1/9 | 28,672 | 1 | 0.117682 | 0.117682 | yes |
| post-asinh transform 2/9 | 28,672 | 1 | 0.117682 | 0.117682 | yes |
| post-asinh transform 3/9 | 28,672 | 1 | 0.117682 | 0.117682 | yes |
| post-asinh transform 4/9 | 28,672 | 1 | 0.117682 | 0.117682 | yes |
| post-asinh transform 5/9 | 28,672 | 1 | 0.117682 | 0.117682 | yes |
| post-asinh transform 6/9 | 28,672 | 1 | 0.117682 | 0.117682 | yes |
| post-asinh transform 7/9 | 28,672 | 1 | 0.117682 | 0.117682 | yes |
| post-asinh transform 8/9 | 28,672 | 1 | 0.117682 | 0.117682 | yes |
| post-asinh transform 9/9 | 28,672 | 1 | 0.117682 | 0.117682 | yes |
| allocation beta | 28,672 | 1 | 0.117682 | 0.117682 | yes |
| knot refit | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| covariance shape 1/5 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| covariance shape 2/5 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| covariance shape 3/5 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| covariance shape 4/5 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| covariance shape 5/5 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| post-asinh transform 1/9 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| post-asinh transform 2/9 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| post-asinh transform 3/9 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| post-asinh transform 4/9 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| post-asinh transform 5/9 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| post-asinh transform 6/9 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| post-asinh transform 7/9 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| post-asinh transform 8/9 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| post-asinh transform 9/9 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| allocation beta | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| knot refit | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| refined parameter-search resolution | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| covariance shape 1/5 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| covariance shape 2/5 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| covariance shape 3/5 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| covariance shape 4/5 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| covariance shape 5/5 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| post-asinh transform 1/9 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| post-asinh transform 2/9 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| post-asinh transform 3/9 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| post-asinh transform 4/9 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| post-asinh transform 5/9 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| post-asinh transform 6/9 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| post-asinh transform 7/9 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| post-asinh transform 8/9 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| post-asinh transform 9/9 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| allocation beta | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| knot refit (cached stationary) | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| refined parameter-search resolution | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| covariance shape 1/5 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| covariance shape 2/5 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| covariance shape 3/5 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| covariance shape 4/5 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| covariance shape 5/5 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| post-asinh transform 1/9 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| post-asinh transform 2/9 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| post-asinh transform 3/9 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| post-asinh transform 4/9 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| post-asinh transform 5/9 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| post-asinh transform 6/9 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| post-asinh transform 7/9 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| post-asinh transform 8/9 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| post-asinh transform 9/9 | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| allocation beta | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| knot refit (cached stationary) | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| fixed point reached | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| accepted prune | 28,672 | 1 | 0.117524 | 0.117524 | yes |
| prune proposal 28672->24576 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| covariance shape 1/5 | 24,576 | 1 | 0.148708 | 0.148708 | yes |
| covariance shape 2/5 | 24,576 | 1 | 0.148708 | 0.148708 | yes |
| covariance shape 3/5 | 24,576 | 1 | 0.148708 | 0.148708 | yes |
| covariance shape 4/5 | 24,576 | 1 | 0.148708 | 0.148708 | yes |
| covariance shape 5/5 | 24,576 | 1 | 0.148708 | 0.148708 | yes |
| post-asinh transform 1/9 | 24,576 | 1 | 0.148708 | 0.148708 | yes |
| post-asinh transform 2/9 | 24,576 | 1 | 0.148708 | 0.148708 | yes |
| post-asinh transform 3/9 | 24,576 | 1 | 0.148708 | 0.148708 | yes |
| post-asinh transform 4/9 | 24,576 | 1 | 0.148708 | 0.148708 | yes |
| post-asinh transform 5/9 | 24,576 | 1 | 0.148708 | 0.148708 | yes |
| post-asinh transform 6/9 | 24,576 | 1 | 0.148708 | 0.148708 | yes |
| post-asinh transform 7/9 | 24,576 | 1 | 0.148708 | 0.148708 | yes |
| post-asinh transform 8/9 | 24,576 | 1 | 0.148708 | 0.148708 | yes |
| post-asinh transform 9/9 | 24,576 | 1 | 0.148708 | 0.148708 | yes |
| allocation beta | 24,576 | 0.9 | 0.147993 | 0.147993 | yes |
| knot refit | 24,576 | 0.9 | 0.147719 | 0.147719 | yes |
| covariance shape 1/5 | 24,576 | 0.9 | 0.147719 | 0.147719 | yes |
| covariance shape 2/5 | 24,576 | 0.9 | 0.147719 | 0.147719 | yes |
| covariance shape 3/5 | 24,576 | 0.9 | 0.147719 | 0.147719 | yes |
| covariance shape 4/5 | 24,576 | 0.9 | 0.147719 | 0.147719 | yes |
| covariance shape 5/5 | 24,576 | 0.9 | 0.147719 | 0.147719 | yes |
| post-asinh transform 1/9 | 24,576 | 0.9 | 0.147719 | 0.147719 | yes |
| post-asinh transform 2/9 | 24,576 | 0.9 | 0.147719 | 0.147719 | yes |
| post-asinh transform 3/9 | 24,576 | 0.9 | 0.147719 | 0.147719 | yes |
| post-asinh transform 4/9 | 24,576 | 0.9 | 0.147719 | 0.147719 | yes |
| post-asinh transform 5/9 | 24,576 | 0.9 | 0.147719 | 0.147719 | yes |
| post-asinh transform 6/9 | 24,576 | 0.9 | 0.147719 | 0.147719 | yes |
| post-asinh transform 7/9 | 24,576 | 0.9 | 0.147719 | 0.147719 | yes |
| post-asinh transform 8/9 | 24,576 | 0.9 | 0.147719 | 0.147719 | yes |
| post-asinh transform 9/9 | 24,576 | 0.9 | 0.147719 | 0.147719 | yes |
| allocation beta | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| knot refit | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| covariance shape 1/5 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| covariance shape 2/5 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| covariance shape 3/5 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| covariance shape 4/5 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| covariance shape 5/5 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| post-asinh transform 1/9 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| post-asinh transform 2/9 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| post-asinh transform 3/9 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| post-asinh transform 4/9 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| post-asinh transform 5/9 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| post-asinh transform 6/9 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| post-asinh transform 7/9 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| post-asinh transform 8/9 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| post-asinh transform 9/9 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| allocation beta | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| knot refit (cached stationary) | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| refined parameter-search resolution | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| covariance shape 1/5 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| covariance shape 2/5 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| covariance shape 3/5 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| covariance shape 4/5 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| covariance shape 5/5 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| post-asinh transform 1/9 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| post-asinh transform 2/9 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| post-asinh transform 3/9 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| post-asinh transform 4/9 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| post-asinh transform 5/9 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| post-asinh transform 6/9 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| post-asinh transform 7/9 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| post-asinh transform 8/9 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| post-asinh transform 9/9 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| allocation beta | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| knot refit (cached stationary) | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| refined parameter-search resolution | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| covariance shape 1/5 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| covariance shape 2/5 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| covariance shape 3/5 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| covariance shape 4/5 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| covariance shape 5/5 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| post-asinh transform 1/9 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| post-asinh transform 2/9 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| post-asinh transform 3/9 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| post-asinh transform 4/9 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| post-asinh transform 5/9 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| post-asinh transform 6/9 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| post-asinh transform 7/9 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| post-asinh transform 8/9 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| post-asinh transform 9/9 | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| allocation beta | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| knot refit (cached stationary) | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| fixed point reached | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| accepted prune | 24,576 | 1 | 0.147549 | 0.147549 | yes |
| prune proposal 24576->20480 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| covariance shape 1/5 | 20,480 | 1 | 0.203495 | 0.203495 | yes |
| covariance shape 2/5 | 20,480 | 1 | 0.203495 | 0.203495 | yes |
| covariance shape 3/5 | 20,480 | 1 | 0.203495 | 0.203495 | yes |
| covariance shape 4/5 | 20,480 | 1 | 0.203495 | 0.203495 | yes |
| covariance shape 5/5 | 20,480 | 1 | 0.203495 | 0.203495 | yes |
| post-asinh transform 1/9 | 20,480 | 1 | 0.203495 | 0.203495 | yes |
| post-asinh transform 2/9 | 20,480 | 1 | 0.203495 | 0.203495 | yes |
| post-asinh transform 3/9 | 20,480 | 1 | 0.203495 | 0.203495 | yes |
| post-asinh transform 4/9 | 20,480 | 1 | 0.203495 | 0.203495 | yes |
| post-asinh transform 5/9 | 20,480 | 1 | 0.203495 | 0.203495 | yes |
| post-asinh transform 6/9 | 20,480 | 1 | 0.203495 | 0.203495 | yes |
| post-asinh transform 7/9 | 20,480 | 1 | 0.203495 | 0.203495 | yes |
| post-asinh transform 8/9 | 20,480 | 1 | 0.203495 | 0.203495 | yes |
| post-asinh transform 9/9 | 20,480 | 1 | 0.203495 | 0.203495 | yes |
| allocation beta | 20,480 | 0.9 | 0.20245 | 0.20245 | yes |
| knot refit | 20,480 | 0.9 | 0.202155 | 0.202155 | yes |
| covariance shape 1/5 | 20,480 | 0.9 | 0.202155 | 0.202155 | yes |
| covariance shape 2/5 | 20,480 | 0.9 | 0.202155 | 0.202155 | yes |
| covariance shape 3/5 | 20,480 | 0.9 | 0.202155 | 0.202155 | yes |
| covariance shape 4/5 | 20,480 | 0.9 | 0.202155 | 0.202155 | yes |
| covariance shape 5/5 | 20,480 | 0.9 | 0.202155 | 0.202155 | yes |
| post-asinh transform 1/9 | 20,480 | 0.9 | 0.202155 | 0.202155 | yes |
| post-asinh transform 2/9 | 20,480 | 0.9 | 0.202155 | 0.202155 | yes |
| post-asinh transform 3/9 | 20,480 | 0.9 | 0.202155 | 0.202155 | yes |
| post-asinh transform 4/9 | 20,480 | 0.9 | 0.202155 | 0.202155 | yes |
| post-asinh transform 5/9 | 20,480 | 0.9 | 0.202155 | 0.202155 | yes |
| post-asinh transform 6/9 | 20,480 | 0.9 | 0.202155 | 0.202155 | yes |
| post-asinh transform 7/9 | 20,480 | 0.9 | 0.202155 | 0.202155 | yes |
| post-asinh transform 8/9 | 20,480 | 0.9 | 0.202155 | 0.202155 | yes |
| post-asinh transform 9/9 | 20,480 | 0.9 | 0.202155 | 0.202155 | yes |
| allocation beta | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| knot refit | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| covariance shape 1/5 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| covariance shape 2/5 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| covariance shape 3/5 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| covariance shape 4/5 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| covariance shape 5/5 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| post-asinh transform 1/9 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| post-asinh transform 2/9 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| post-asinh transform 3/9 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| post-asinh transform 4/9 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| post-asinh transform 5/9 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| post-asinh transform 6/9 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| post-asinh transform 7/9 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| post-asinh transform 8/9 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| post-asinh transform 9/9 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| allocation beta | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| knot refit (cached stationary) | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| refined parameter-search resolution | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| covariance shape 1/5 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| covariance shape 2/5 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| covariance shape 3/5 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| covariance shape 4/5 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| covariance shape 5/5 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| post-asinh transform 1/9 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| post-asinh transform 2/9 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| post-asinh transform 3/9 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| post-asinh transform 4/9 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| post-asinh transform 5/9 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| post-asinh transform 6/9 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| post-asinh transform 7/9 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| post-asinh transform 8/9 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| post-asinh transform 9/9 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| allocation beta | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| knot refit (cached stationary) | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| refined parameter-search resolution | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| covariance shape 1/5 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| covariance shape 2/5 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| covariance shape 3/5 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| covariance shape 4/5 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| covariance shape 5/5 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| post-asinh transform 1/9 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| post-asinh transform 2/9 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| post-asinh transform 3/9 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| post-asinh transform 4/9 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| post-asinh transform 5/9 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| post-asinh transform 6/9 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| post-asinh transform 7/9 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| post-asinh transform 8/9 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| post-asinh transform 9/9 | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| allocation beta | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| knot refit (cached stationary) | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| fixed point reached | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| accepted prune | 20,480 | 1 | 0.201928 | 0.201928 | yes |
| prune proposal 20480->16384 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| covariance shape 1/5 | 16,384 | 1 | 0.291714 | 0.291714 | yes |
| covariance shape 2/5 | 16,384 | 1 | 0.291714 | 0.291714 | yes |
| covariance shape 3/5 | 16,384 | 1 | 0.291714 | 0.291714 | yes |
| covariance shape 4/5 | 16,384 | 1 | 0.291714 | 0.291714 | yes |
| covariance shape 5/5 | 16,384 | 1 | 0.291714 | 0.291714 | yes |
| post-asinh transform 1/9 | 16,384 | 1 | 0.291714 | 0.291714 | yes |
| post-asinh transform 2/9 | 16,384 | 1 | 0.291714 | 0.291714 | yes |
| post-asinh transform 3/9 | 16,384 | 1 | 0.291714 | 0.291714 | yes |
| post-asinh transform 4/9 | 16,384 | 1 | 0.291714 | 0.291714 | yes |
| post-asinh transform 5/9 | 16,384 | 1 | 0.291714 | 0.291714 | yes |
| post-asinh transform 6/9 | 16,384 | 1 | 0.291714 | 0.291714 | yes |
| post-asinh transform 7/9 | 16,384 | 1 | 0.291714 | 0.291714 | yes |
| post-asinh transform 8/9 | 16,384 | 1 | 0.291714 | 0.291714 | yes |
| post-asinh transform 9/9 | 16,384 | 1 | 0.291714 | 0.291714 | yes |
| allocation beta | 16,384 | 0.9 | 0.290814 | 0.290814 | yes |
| knot refit | 16,384 | 0.9 | 0.290814 | 0.290814 | yes |
| covariance shape 1/5 | 16,384 | 0.9 | 0.290814 | 0.290814 | yes |
| covariance shape 2/5 | 16,384 | 0.9 | 0.290814 | 0.290814 | yes |
| covariance shape 3/5 | 16,384 | 0.9 | 0.290814 | 0.290814 | yes |
| covariance shape 4/5 | 16,384 | 0.9 | 0.290814 | 0.290814 | yes |
| covariance shape 5/5 | 16,384 | 0.9 | 0.290814 | 0.290814 | yes |
| post-asinh transform 1/9 | 16,384 | 0.9 | 0.290814 | 0.290814 | yes |
| post-asinh transform 2/9 | 16,384 | 0.9 | 0.290814 | 0.290814 | yes |
| post-asinh transform 3/9 | 16,384 | 0.9 | 0.290814 | 0.290814 | yes |
| post-asinh transform 4/9 | 16,384 | 0.9 | 0.290814 | 0.290814 | yes |
| post-asinh transform 5/9 | 16,384 | 0.9 | 0.290814 | 0.290814 | yes |
| post-asinh transform 6/9 | 16,384 | 0.9 | 0.290814 | 0.290814 | yes |
| post-asinh transform 7/9 | 16,384 | 0.9 | 0.290814 | 0.290814 | yes |
| post-asinh transform 8/9 | 16,384 | 0.9 | 0.290814 | 0.290814 | yes |
| post-asinh transform 9/9 | 16,384 | 0.9 | 0.290814 | 0.290814 | yes |
| allocation beta | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| knot refit | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| covariance shape 1/5 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| covariance shape 2/5 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| covariance shape 3/5 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| covariance shape 4/5 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| covariance shape 5/5 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| post-asinh transform 1/9 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| post-asinh transform 2/9 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| post-asinh transform 3/9 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| post-asinh transform 4/9 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| post-asinh transform 5/9 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| post-asinh transform 6/9 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| post-asinh transform 7/9 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| post-asinh transform 8/9 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| post-asinh transform 9/9 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| allocation beta | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| knot refit (cached stationary) | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| refined parameter-search resolution | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| covariance shape 1/5 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| covariance shape 2/5 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| covariance shape 3/5 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| covariance shape 4/5 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| covariance shape 5/5 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| post-asinh transform 1/9 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| post-asinh transform 2/9 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| post-asinh transform 3/9 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| post-asinh transform 4/9 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| post-asinh transform 5/9 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| post-asinh transform 6/9 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| post-asinh transform 7/9 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| post-asinh transform 8/9 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| post-asinh transform 9/9 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| allocation beta | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| knot refit (cached stationary) | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| refined parameter-search resolution | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| covariance shape 1/5 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| covariance shape 2/5 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| covariance shape 3/5 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| covariance shape 4/5 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| covariance shape 5/5 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| post-asinh transform 1/9 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| post-asinh transform 2/9 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| post-asinh transform 3/9 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| post-asinh transform 4/9 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| post-asinh transform 5/9 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| post-asinh transform 6/9 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| post-asinh transform 7/9 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| post-asinh transform 8/9 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| post-asinh transform 9/9 | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| allocation beta | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| knot refit (cached stationary) | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| fixed point reached | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| accepted prune | 16,384 | 1 | 0.290164 | 0.290164 | yes |
| prune proposal 16384->12288 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| covariance shape 1/5 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| covariance shape 2/5 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| covariance shape 3/5 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| covariance shape 4/5 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| covariance shape 5/5 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| post-asinh transform 1/9 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| post-asinh transform 2/9 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| post-asinh transform 3/9 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| post-asinh transform 4/9 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| post-asinh transform 5/9 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| post-asinh transform 6/9 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| post-asinh transform 7/9 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| post-asinh transform 8/9 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| post-asinh transform 9/9 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| allocation beta | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| knot refit | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| refined parameter-search resolution | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| covariance shape 1/5 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| covariance shape 2/5 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| covariance shape 3/5 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| covariance shape 4/5 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| covariance shape 5/5 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| post-asinh transform 1/9 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| post-asinh transform 2/9 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| post-asinh transform 3/9 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| post-asinh transform 4/9 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| post-asinh transform 5/9 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| post-asinh transform 6/9 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| post-asinh transform 7/9 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| post-asinh transform 8/9 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| post-asinh transform 9/9 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| allocation beta | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| knot refit (cached stationary) | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| refined parameter-search resolution | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| covariance shape 1/5 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| covariance shape 2/5 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| covariance shape 3/5 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| covariance shape 4/5 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| covariance shape 5/5 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| post-asinh transform 1/9 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| post-asinh transform 2/9 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| post-asinh transform 3/9 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| post-asinh transform 4/9 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| post-asinh transform 5/9 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| post-asinh transform 6/9 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| post-asinh transform 7/9 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| post-asinh transform 8/9 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| post-asinh transform 9/9 | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| allocation beta | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| knot refit (cached stationary) | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| fixed point reached | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| accepted prune | 12,288 | 1 | 0.393515 | 0.393515 | yes |
| prune proposal 12288->9216 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| covariance shape 1/5 | 9,216 | 1 | 0.498076 | 0.498076 | yes |
| covariance shape 2/5 | 9,216 | 1 | 0.498076 | 0.498076 | yes |
| covariance shape 3/5 | 9,216 | 1 | 0.498076 | 0.498076 | yes |
| covariance shape 4/5 | 9,216 | 1 | 0.498076 | 0.498076 | yes |
| covariance shape 5/5 | 9,216 | 1 | 0.498076 | 0.498076 | yes |
| post-asinh transform 1/9 | 9,216 | 1 | 0.498076 | 0.498076 | yes |
| post-asinh transform 2/9 | 9,216 | 1 | 0.498076 | 0.498076 | yes |
| post-asinh transform 3/9 | 9,216 | 1 | 0.498076 | 0.498076 | yes |
| post-asinh transform 4/9 | 9,216 | 1 | 0.498076 | 0.498076 | yes |
| post-asinh transform 5/9 | 9,216 | 1 | 0.498076 | 0.498076 | yes |
| post-asinh transform 6/9 | 9,216 | 1 | 0.498076 | 0.498076 | yes |
| post-asinh transform 7/9 | 9,216 | 1 | 0.498076 | 0.498076 | yes |
| post-asinh transform 8/9 | 9,216 | 1 | 0.498076 | 0.498076 | yes |
| post-asinh transform 9/9 | 9,216 | 1 | 0.498076 | 0.498076 | yes |
| allocation beta | 9,216 | 1 | 0.498076 | 0.498076 | yes |
| knot refit | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| covariance shape 1/5 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| covariance shape 2/5 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| covariance shape 3/5 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| covariance shape 4/5 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| covariance shape 5/5 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| post-asinh transform 1/9 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| post-asinh transform 2/9 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| post-asinh transform 3/9 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| post-asinh transform 4/9 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| post-asinh transform 5/9 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| post-asinh transform 6/9 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| post-asinh transform 7/9 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| post-asinh transform 8/9 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| post-asinh transform 9/9 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| allocation beta | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| knot refit | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| refined parameter-search resolution | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| covariance shape 1/5 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| covariance shape 2/5 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| covariance shape 3/5 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| covariance shape 4/5 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| covariance shape 5/5 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| post-asinh transform 1/9 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| post-asinh transform 2/9 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| post-asinh transform 3/9 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| post-asinh transform 4/9 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| post-asinh transform 5/9 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| post-asinh transform 6/9 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| post-asinh transform 7/9 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| post-asinh transform 8/9 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| post-asinh transform 9/9 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| allocation beta | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| knot refit (cached stationary) | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| refined parameter-search resolution | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| covariance shape 1/5 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| covariance shape 2/5 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| covariance shape 3/5 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| covariance shape 4/5 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| covariance shape 5/5 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| post-asinh transform 1/9 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| post-asinh transform 2/9 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| post-asinh transform 3/9 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| post-asinh transform 4/9 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| post-asinh transform 5/9 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| post-asinh transform 6/9 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| post-asinh transform 7/9 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| post-asinh transform 8/9 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| post-asinh transform 9/9 | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| allocation beta | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| knot refit (cached stationary) | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| fixed point reached | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| accepted prune | 9,216 | 1 | 0.497122 | 0.497122 | yes |
| prune proposal 9216->6912 | 6,912 | 0.975 | 0.599196 | 0.599196 | yes |
| covariance shape 1/5 | 6,912 | 1 | 0.608469 | 0.608469 | yes |
| covariance shape 2/5 | 6,912 | 1 | 0.608469 | 0.608469 | yes |
| covariance shape 3/5 | 6,912 | 1 | 0.608469 | 0.608469 | yes |
| covariance shape 4/5 | 6,912 | 1 | 0.608469 | 0.608469 | yes |
| covariance shape 5/5 | 6,912 | 1 | 0.608469 | 0.608469 | yes |
| post-asinh transform 1/9 | 6,912 | 1 | 0.608469 | 0.608469 | yes |
| post-asinh transform 2/9 | 6,912 | 1 | 0.608469 | 0.608469 | yes |
| post-asinh transform 3/9 | 6,912 | 1 | 0.608469 | 0.608469 | yes |
| post-asinh transform 4/9 | 6,912 | 1 | 0.608469 | 0.608469 | yes |
| post-asinh transform 5/9 | 6,912 | 1 | 0.608469 | 0.608469 | yes |
| post-asinh transform 6/9 | 6,912 | 1 | 0.608469 | 0.608469 | yes |
| post-asinh transform 7/9 | 6,912 | 1 | 0.608469 | 0.608469 | yes |
| post-asinh transform 8/9 | 6,912 | 1 | 0.608469 | 0.608469 | yes |
| post-asinh transform 9/9 | 6,912 | 1 | 0.608469 | 0.608469 | yes |
| allocation beta | 6,912 | 1 | 0.608469 | 0.608469 | yes |
| knot refit | 6,912 | 1 | 0.606243 | 0.606243 | yes |
| covariance shape 1/5 | 6,912 | 1 | 0.606243 | 0.606243 | yes |
| covariance shape 2/5 | 6,912 | 1 | 0.606243 | 0.606243 | yes |
| covariance shape 3/5 | 6,912 | 1 | 0.606243 | 0.606243 | yes |
| covariance shape 4/5 | 6,912 | 1 | 0.606243 | 0.606243 | yes |
| covariance shape 5/5 | 6,912 | 1 | 0.606243 | 0.606243 | yes |
| post-asinh transform 1/9 | 6,912 | 1 | 0.606243 | 0.606243 | yes |
| post-asinh transform 2/9 | 6,912 | 1 | 0.606243 | 0.606243 | yes |
| post-asinh transform 3/9 | 6,912 | 1 | 0.606243 | 0.606243 | yes |
| post-asinh transform 4/9 | 6,912 | 1 | 0.606243 | 0.606243 | yes |
| post-asinh transform 5/9 | 6,912 | 1 | 0.606243 | 0.606243 | yes |
| post-asinh transform 6/9 | 6,912 | 1 | 0.606243 | 0.606243 | yes |
| post-asinh transform 7/9 | 6,912 | 1 | 0.606243 | 0.606243 | yes |
| post-asinh transform 8/9 | 6,912 | 1 | 0.606243 | 0.606243 | yes |
| post-asinh transform 9/9 | 6,912 | 1 | 0.606243 | 0.606243 | yes |
| allocation beta | 6,912 | 1 | 0.606243 | 0.606243 | yes |
| knot refit | 6,912 | 1 | 0.606243 | 0.606243 | yes |
| refined parameter-search resolution | 6,912 | 1 | 0.606243 | 0.606243 | yes |
| covariance shape 1/5 | 6,912 | 1 | 0.606243 | 0.606243 | yes |
| covariance shape 2/5 | 6,912 | 1 | 0.606243 | 0.606243 | yes |
| covariance shape 3/5 | 6,912 | 1 | 0.606243 | 0.606243 | yes |
| covariance shape 4/5 | 6,912 | 1 | 0.606243 | 0.606243 | yes |
| covariance shape 5/5 | 6,912 | 1 | 0.606243 | 0.606243 | yes |
| post-asinh transform 1/9 | 6,912 | 1 | 0.606243 | 0.606243 | yes |
| post-asinh transform 2/9 | 6,912 | 1 | 0.606243 | 0.606243 | yes |
| post-asinh transform 3/9 | 6,912 | 1 | 0.606243 | 0.606243 | yes |
| post-asinh transform 4/9 | 6,912 | 1 | 0.606243 | 0.606243 | yes |
| post-asinh transform 5/9 | 6,912 | 1 | 0.606243 | 0.606243 | yes |
| post-asinh transform 6/9 | 6,912 | 1 | 0.606243 | 0.606243 | yes |
| post-asinh transform 7/9 | 6,912 | 1 | 0.606243 | 0.606243 | yes |
| post-asinh transform 8/9 | 6,912 | 1 | 0.606243 | 0.606243 | yes |
| post-asinh transform 9/9 | 6,912 | 1 | 0.606243 | 0.606243 | yes |
| allocation beta | 6,912 | 1 | 0.606243 | 0.606243 | yes |
| knot refit (cached stationary) | 6,912 | 1 | 0.606243 | 0.606243 | yes |
| refined parameter-search resolution | 6,912 | 1 | 0.606243 | 0.606243 | yes |
| covariance shape 1/5 | 6,912 | 1 | 0.606243 | 0.606243 | yes |
| covariance shape 2/5 | 6,912 | 1 | 0.605449 | 0.605449 | yes |
| covariance shape 3/5 | 6,912 | 1 | 0.605449 | 0.605449 | yes |
| covariance shape 4/5 | 6,912 | 1 | 0.605449 | 0.605449 | yes |
| covariance shape 5/5 | 6,912 | 1 | 0.605449 | 0.605449 | yes |
| post-asinh transform 1/9 | 6,912 | 1 | 0.605449 | 0.605449 | yes |
| post-asinh transform 2/9 | 6,912 | 1 | 0.605449 | 0.605449 | yes |
| post-asinh transform 3/9 | 6,912 | 1 | 0.605449 | 0.605449 | yes |
| post-asinh transform 4/9 | 6,912 | 1 | 0.605449 | 0.605449 | yes |
| post-asinh transform 5/9 | 6,912 | 1 | 0.605449 | 0.605449 | yes |
| post-asinh transform 6/9 | 6,912 | 1 | 0.605449 | 0.605449 | yes |
| post-asinh transform 7/9 | 6,912 | 1 | 0.605449 | 0.605449 | yes |
| post-asinh transform 8/9 | 6,912 | 1 | 0.605449 | 0.605449 | yes |
| post-asinh transform 9/9 | 6,912 | 1 | 0.605449 | 0.605449 | yes |
| allocation beta | 6,912 | 0.975 | 0.599196 | 0.599196 | yes |
| knot refit | 6,912 | 0.975 | 0.599196 | 0.599196 | yes |
| covariance shape 1/5 | 6,912 | 0.975 | 0.599196 | 0.599196 | yes |
| covariance shape 2/5 | 6,912 | 0.975 | 0.599196 | 0.599196 | yes |
| covariance shape 3/5 | 6,912 | 0.975 | 0.599196 | 0.599196 | yes |
| covariance shape 4/5 | 6,912 | 0.975 | 0.599196 | 0.599196 | yes |
| covariance shape 5/5 | 6,912 | 0.975 | 0.599196 | 0.599196 | yes |
| post-asinh transform 1/9 | 6,912 | 0.975 | 0.599196 | 0.599196 | yes |
| post-asinh transform 2/9 | 6,912 | 0.975 | 0.599196 | 0.599196 | yes |
| post-asinh transform 3/9 | 6,912 | 0.975 | 0.599196 | 0.599196 | yes |
| post-asinh transform 4/9 | 6,912 | 0.975 | 0.599196 | 0.599196 | yes |
| post-asinh transform 5/9 | 6,912 | 0.975 | 0.599196 | 0.599196 | yes |
| post-asinh transform 6/9 | 6,912 | 0.975 | 0.599196 | 0.599196 | yes |
| post-asinh transform 7/9 | 6,912 | 0.975 | 0.599196 | 0.599196 | yes |
| post-asinh transform 8/9 | 6,912 | 0.975 | 0.599196 | 0.599196 | yes |
| post-asinh transform 9/9 | 6,912 | 0.975 | 0.599196 | 0.599196 | yes |
| allocation beta | 6,912 | 0.975 | 0.599196 | 0.599196 | yes |
| knot refit (cached stationary) | 6,912 | 0.975 | 0.599196 | 0.599196 | yes |
| fixed point reached | 6,912 | 0.975 | 0.599196 | 0.599196 | yes |
| accepted prune | 6,912 | 0.975 | 0.599196 | 0.599196 | yes |
| prune proposal 6912->5184 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| covariance shape 1/5 | 5,184 | 0.975 | 0.700822 | 0.700822 | yes |
| covariance shape 2/5 | 5,184 | 0.975 | 0.700822 | 0.700822 | yes |
| covariance shape 3/5 | 5,184 | 0.975 | 0.700822 | 0.700822 | yes |
| covariance shape 4/5 | 5,184 | 0.975 | 0.700822 | 0.700822 | yes |
| covariance shape 5/5 | 5,184 | 0.975 | 0.700822 | 0.700822 | yes |
| post-asinh transform 1/9 | 5,184 | 0.975 | 0.700822 | 0.700822 | yes |
| post-asinh transform 2/9 | 5,184 | 0.975 | 0.700822 | 0.700822 | yes |
| post-asinh transform 3/9 | 5,184 | 0.975 | 0.700822 | 0.700822 | yes |
| post-asinh transform 4/9 | 5,184 | 0.975 | 0.700822 | 0.700822 | yes |
| post-asinh transform 5/9 | 5,184 | 0.975 | 0.700822 | 0.700822 | yes |
| post-asinh transform 6/9 | 5,184 | 0.975 | 0.700822 | 0.700822 | yes |
| post-asinh transform 7/9 | 5,184 | 0.975 | 0.700822 | 0.700822 | yes |
| post-asinh transform 8/9 | 5,184 | 0.975 | 0.700822 | 0.700822 | yes |
| post-asinh transform 9/9 | 5,184 | 0.975 | 0.700822 | 0.700822 | yes |
| allocation beta | 5,184 | 1 | 0.693922 | 0.693922 | yes |
| knot refit | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| covariance shape 1/5 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| covariance shape 2/5 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| covariance shape 3/5 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| covariance shape 4/5 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| covariance shape 5/5 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| post-asinh transform 1/9 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| post-asinh transform 2/9 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| post-asinh transform 3/9 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| post-asinh transform 4/9 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| post-asinh transform 5/9 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| post-asinh transform 6/9 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| post-asinh transform 7/9 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| post-asinh transform 8/9 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| post-asinh transform 9/9 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| allocation beta | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| knot refit | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| refined parameter-search resolution | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| covariance shape 1/5 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| covariance shape 2/5 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| covariance shape 3/5 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| covariance shape 4/5 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| covariance shape 5/5 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| post-asinh transform 1/9 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| post-asinh transform 2/9 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| post-asinh transform 3/9 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| post-asinh transform 4/9 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| post-asinh transform 5/9 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| post-asinh transform 6/9 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| post-asinh transform 7/9 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| post-asinh transform 8/9 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| post-asinh transform 9/9 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| allocation beta | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| knot refit (cached stationary) | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| refined parameter-search resolution | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| covariance shape 1/5 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| covariance shape 2/5 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| covariance shape 3/5 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| covariance shape 4/5 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| covariance shape 5/5 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| post-asinh transform 1/9 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| post-asinh transform 2/9 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| post-asinh transform 3/9 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| post-asinh transform 4/9 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| post-asinh transform 5/9 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| post-asinh transform 6/9 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| post-asinh transform 7/9 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| post-asinh transform 8/9 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| post-asinh transform 9/9 | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| allocation beta | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| knot refit (cached stationary) | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| fixed point reached | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| accepted prune | 5,184 | 1 | 0.691704 | 0.691704 | yes |
| prune proposal 5184->3888 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| covariance shape 1/5 | 3,888 | 1 | 0.931678 | 0.931678 | yes |
| covariance shape 2/5 | 3,888 | 1 | 0.931678 | 0.931678 | yes |
| covariance shape 3/5 | 3,888 | 1 | 0.931678 | 0.931678 | yes |
| covariance shape 4/5 | 3,888 | 1 | 0.931678 | 0.931678 | yes |
| covariance shape 5/5 | 3,888 | 1 | 0.931678 | 0.931678 | yes |
| post-asinh transform 1/9 | 3,888 | 1 | 0.931678 | 0.931678 | yes |
| post-asinh transform 2/9 | 3,888 | 1 | 0.931678 | 0.931678 | yes |
| post-asinh transform 3/9 | 3,888 | 1 | 0.931678 | 0.931678 | yes |
| post-asinh transform 4/9 | 3,888 | 1 | 0.931678 | 0.931678 | yes |
| post-asinh transform 5/9 | 3,888 | 1 | 0.931678 | 0.931678 | yes |
| post-asinh transform 6/9 | 3,888 | 1 | 0.931678 | 0.931678 | yes |
| post-asinh transform 7/9 | 3,888 | 1 | 0.931678 | 0.931678 | yes |
| post-asinh transform 8/9 | 3,888 | 1 | 0.931678 | 0.931678 | yes |
| post-asinh transform 9/9 | 3,888 | 1 | 0.931678 | 0.931678 | yes |
| allocation beta | 3,888 | 0.9 | 0.926926 | 0.926926 | yes |
| knot refit | 3,888 | 0.9 | 0.923906 | 0.923906 | yes |
| covariance shape 1/5 | 3,888 | 0.9 | 0.923906 | 0.923906 | yes |
| covariance shape 2/5 | 3,888 | 0.9 | 0.923906 | 0.923906 | yes |
| covariance shape 3/5 | 3,888 | 0.9 | 0.923906 | 0.923906 | yes |
| covariance shape 4/5 | 3,888 | 0.9 | 0.923906 | 0.923906 | yes |
| covariance shape 5/5 | 3,888 | 0.9 | 0.923906 | 0.923906 | yes |
| post-asinh transform 1/9 | 3,888 | 0.9 | 0.923906 | 0.923906 | yes |
| post-asinh transform 2/9 | 3,888 | 0.9 | 0.923906 | 0.923906 | yes |
| post-asinh transform 3/9 | 3,888 | 0.9 | 0.923906 | 0.923906 | yes |
| post-asinh transform 4/9 | 3,888 | 0.9 | 0.923906 | 0.923906 | yes |
| post-asinh transform 5/9 | 3,888 | 0.9 | 0.923906 | 0.923906 | yes |
| post-asinh transform 6/9 | 3,888 | 0.9 | 0.923906 | 0.923906 | yes |
| post-asinh transform 7/9 | 3,888 | 0.9 | 0.923906 | 0.923906 | yes |
| post-asinh transform 8/9 | 3,888 | 0.9 | 0.923906 | 0.923906 | yes |
| post-asinh transform 9/9 | 3,888 | 0.9 | 0.923906 | 0.923906 | yes |
| allocation beta | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| knot refit | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| covariance shape 1/5 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| covariance shape 2/5 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| covariance shape 3/5 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| covariance shape 4/5 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| covariance shape 5/5 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| post-asinh transform 1/9 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| post-asinh transform 2/9 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| post-asinh transform 3/9 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| post-asinh transform 4/9 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| post-asinh transform 5/9 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| post-asinh transform 6/9 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| post-asinh transform 7/9 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| post-asinh transform 8/9 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| post-asinh transform 9/9 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| allocation beta | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| knot refit (cached stationary) | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| refined parameter-search resolution | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| covariance shape 1/5 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| covariance shape 2/5 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| covariance shape 3/5 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| covariance shape 4/5 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| covariance shape 5/5 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| post-asinh transform 1/9 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| post-asinh transform 2/9 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| post-asinh transform 3/9 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| post-asinh transform 4/9 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| post-asinh transform 5/9 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| post-asinh transform 6/9 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| post-asinh transform 7/9 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| post-asinh transform 8/9 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| post-asinh transform 9/9 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| allocation beta | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| knot refit (cached stationary) | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| refined parameter-search resolution | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| covariance shape 1/5 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| covariance shape 2/5 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| covariance shape 3/5 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| covariance shape 4/5 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| covariance shape 5/5 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| post-asinh transform 1/9 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| post-asinh transform 2/9 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| post-asinh transform 3/9 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| post-asinh transform 4/9 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| post-asinh transform 5/9 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| post-asinh transform 6/9 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| post-asinh transform 7/9 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| post-asinh transform 8/9 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| post-asinh transform 9/9 | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| allocation beta | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| knot refit (cached stationary) | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| fixed point reached | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| accepted prune | 3,888 | 1 | 0.914286 | 0.914286 | yes |
| cold restart 3888->2916 | 2,916 | 1 | 0.907406 | 0.907406 | yes |
| covariance shape 1/5 | 2,916 | 1 | 0.947793 | 0.947793 | yes |
| covariance shape 2/5 | 2,916 | 1 | 0.947793 | 0.947793 | yes |
| covariance shape 3/5 | 2,916 | 1 | 0.947793 | 0.947793 | yes |
| covariance shape 4/5 | 2,916 | 1 | 0.947793 | 0.947793 | yes |
| covariance shape 5/5 | 2,916 | 1 | 0.947793 | 0.947793 | yes |
| post-asinh transform 1/9 | 2,916 | 1 | 0.947793 | 0.947793 | yes |
| post-asinh transform 2/9 | 2,916 | 1 | 0.947793 | 0.947793 | yes |
| post-asinh transform 3/9 | 2,916 | 1 | 0.947793 | 0.947793 | yes |
| post-asinh transform 4/9 | 2,916 | 1 | 0.947793 | 0.947793 | yes |
| post-asinh transform 5/9 | 2,916 | 1 | 0.947793 | 0.947793 | yes |
| post-asinh transform 6/9 | 2,916 | 1 | 0.947793 | 0.947793 | yes |
| post-asinh transform 7/9 | 2,916 | 1 | 0.947793 | 0.947793 | yes |
| post-asinh transform 8/9 | 2,916 | 1 | 0.947793 | 0.947793 | yes |
| post-asinh transform 9/9 | 2,916 | 1 | 0.947793 | 0.947793 | yes |
| allocation beta | 2,916 | 0.9 | 0.943093 | 0.943093 | yes |
| knot refit | 2,916 | 0.9 | 0.939141 | 0.939141 | yes |
| covariance shape 1/5 | 2,916 | 0.9 | 0.939141 | 0.939141 | yes |
| covariance shape 2/5 | 2,916 | 0.9 | 0.939141 | 0.939141 | yes |
| covariance shape 3/5 | 2,916 | 0.9 | 0.939141 | 0.939141 | yes |
| covariance shape 4/5 | 2,916 | 0.9 | 0.939141 | 0.939141 | yes |
| covariance shape 5/5 | 2,916 | 0.9 | 0.939141 | 0.939141 | yes |
| post-asinh transform 1/9 | 2,916 | 0.9 | 0.939141 | 0.939141 | yes |
| post-asinh transform 2/9 | 2,916 | 0.9 | 0.939141 | 0.939141 | yes |
| post-asinh transform 3/9 | 2,916 | 0.9 | 0.939141 | 0.939141 | yes |
| post-asinh transform 4/9 | 2,916 | 0.9 | 0.939141 | 0.939141 | yes |
| post-asinh transform 5/9 | 2,916 | 0.9 | 0.939141 | 0.939141 | yes |
| post-asinh transform 6/9 | 2,916 | 0.9 | 0.939141 | 0.939141 | yes |
| post-asinh transform 7/9 | 2,916 | 0.9 | 0.939141 | 0.939141 | yes |
| post-asinh transform 8/9 | 2,916 | 0.9 | 0.939141 | 0.939141 | yes |
| post-asinh transform 9/9 | 2,916 | 0.9 | 0.939141 | 0.939141 | yes |
| allocation beta | 2,916 | 1 | 0.934268 | 0.934268 | yes |
| knot refit | 2,916 | 1 | 0.930973 | 0.930973 | yes |
| covariance shape 1/5 | 2,916 | 1 | 0.930973 | 0.930973 | yes |
| covariance shape 2/5 | 2,916 | 1 | 0.930973 | 0.930973 | yes |
| covariance shape 3/5 | 2,916 | 1 | 0.930973 | 0.930973 | yes |
| covariance shape 4/5 | 2,916 | 1 | 0.930973 | 0.930973 | yes |
| covariance shape 5/5 | 2,916 | 1 | 0.930973 | 0.930973 | yes |
| post-asinh transform 1/9 | 2,916 | 1 | 0.930973 | 0.930973 | yes |
| post-asinh transform 2/9 | 2,916 | 1 | 0.930973 | 0.930973 | yes |
| post-asinh transform 3/9 | 2,916 | 1 | 0.930973 | 0.930973 | yes |
| post-asinh transform 4/9 | 2,916 | 1 | 0.930973 | 0.930973 | yes |
| post-asinh transform 5/9 | 2,916 | 1 | 0.930973 | 0.930973 | yes |
| post-asinh transform 6/9 | 2,916 | 1 | 0.930973 | 0.930973 | yes |
| post-asinh transform 7/9 | 2,916 | 1 | 0.930973 | 0.930973 | yes |
| post-asinh transform 8/9 | 2,916 | 1 | 0.930973 | 0.930973 | yes |
| post-asinh transform 9/9 | 2,916 | 1 | 0.930973 | 0.930973 | yes |
| allocation beta | 2,916 | 1 | 0.930973 | 0.930973 | yes |
| knot refit | 2,916 | 1 | 0.930973 | 0.930973 | yes |
| refined parameter-search resolution | 2,916 | 1 | 0.930973 | 0.930973 | yes |
| covariance shape 1/5 | 2,916 | 1 | 0.929384 | 0.929384 | yes |
| covariance shape 2/5 | 2,916 | 1 | 0.929384 | 0.929384 | yes |
| covariance shape 3/5 | 2,916 | 1 | 0.929384 | 0.929384 | yes |
| covariance shape 4/5 | 2,916 | 1 | 0.929384 | 0.929384 | yes |
| covariance shape 5/5 | 2,916 | 1 | 0.929384 | 0.929384 | yes |
| post-asinh transform 1/9 | 2,916 | 1 | 0.929384 | 0.929384 | yes |
| post-asinh transform 2/9 | 2,916 | 1 | 0.926866 | 0.926866 | yes |
| post-asinh transform 3/9 | 2,916 | 1 | 0.925479 | 0.925479 | yes |
| post-asinh transform 4/9 | 2,916 | 1 | 0.925479 | 0.925479 | yes |
| post-asinh transform 5/9 | 2,916 | 1 | 0.925479 | 0.925479 | yes |
| post-asinh transform 6/9 | 2,916 | 1 | 0.925479 | 0.925479 | yes |
| post-asinh transform 7/9 | 2,916 | 1 | 0.925479 | 0.925479 | yes |
| post-asinh transform 8/9 | 2,916 | 1 | 0.925479 | 0.925479 | yes |
| post-asinh transform 9/9 | 2,916 | 1 | 0.925479 | 0.925479 | yes |
| allocation beta | 2,916 | 0.95 | 0.918067 | 0.918067 | yes |
| knot refit | 2,916 | 0.95 | 0.916389 | 0.916389 | yes |
| covariance shape 1/5 | 2,916 | 0.95 | 0.916389 | 0.916389 | yes |
| covariance shape 2/5 | 2,916 | 0.95 | 0.916389 | 0.916389 | yes |
| covariance shape 3/5 | 2,916 | 0.95 | 0.916389 | 0.916389 | yes |
| covariance shape 4/5 | 2,916 | 0.95 | 0.916389 | 0.916389 | yes |
| covariance shape 5/5 | 2,916 | 0.95 | 0.916389 | 0.916389 | yes |
| post-asinh transform 1/9 | 2,916 | 0.95 | 0.916389 | 0.916389 | yes |
| post-asinh transform 2/9 | 2,916 | 0.95 | 0.916389 | 0.916389 | yes |
| post-asinh transform 3/9 | 2,916 | 0.95 | 0.916389 | 0.916389 | yes |
| post-asinh transform 4/9 | 2,916 | 0.95 | 0.916389 | 0.916389 | yes |
| post-asinh transform 5/9 | 2,916 | 0.95 | 0.916389 | 0.916389 | yes |
| post-asinh transform 6/9 | 2,916 | 0.95 | 0.916389 | 0.916389 | yes |
| post-asinh transform 7/9 | 2,916 | 0.95 | 0.916389 | 0.916389 | yes |
| post-asinh transform 8/9 | 2,916 | 0.95 | 0.916389 | 0.916389 | yes |
| post-asinh transform 9/9 | 2,916 | 0.95 | 0.916389 | 0.916389 | yes |
| allocation beta | 2,916 | 1 | 0.91292 | 0.91292 | yes |
| knot refit | 2,916 | 1 | 0.91292 | 0.91292 | yes |
| covariance shape 1/5 | 2,916 | 1 | 0.91292 | 0.91292 | yes |
| covariance shape 2/5 | 2,916 | 1 | 0.91292 | 0.91292 | yes |
| covariance shape 3/5 | 2,916 | 1 | 0.91292 | 0.91292 | yes |
| covariance shape 4/5 | 2,916 | 1 | 0.91292 | 0.91292 | yes |
| covariance shape 5/5 | 2,916 | 1 | 0.91292 | 0.91292 | yes |
| post-asinh transform 1/9 | 2,916 | 1 | 0.91292 | 0.91292 | yes |
| post-asinh transform 2/9 | 2,916 | 1 | 0.91292 | 0.91292 | yes |
| post-asinh transform 3/9 | 2,916 | 1 | 0.91292 | 0.91292 | yes |
| post-asinh transform 4/9 | 2,916 | 1 | 0.91292 | 0.91292 | yes |
| post-asinh transform 5/9 | 2,916 | 1 | 0.91292 | 0.91292 | yes |
| post-asinh transform 6/9 | 2,916 | 1 | 0.91292 | 0.91292 | yes |
| post-asinh transform 7/9 | 2,916 | 1 | 0.91292 | 0.91292 | yes |
| post-asinh transform 8/9 | 2,916 | 1 | 0.91292 | 0.91292 | yes |
| post-asinh transform 9/9 | 2,916 | 1 | 0.91292 | 0.91292 | yes |
| allocation beta | 2,916 | 1 | 0.91292 | 0.91292 | yes |
| knot refit (cached stationary) | 2,916 | 1 | 0.91292 | 0.91292 | yes |
| refined parameter-search resolution | 2,916 | 1 | 0.91292 | 0.91292 | yes |
| covariance shape 1/5 | 2,916 | 1 | 0.91292 | 0.91292 | yes |
| covariance shape 2/5 | 2,916 | 1 | 0.91292 | 0.91292 | yes |
| covariance shape 3/5 | 2,916 | 1 | 0.91292 | 0.91292 | yes |
| covariance shape 4/5 | 2,916 | 1 | 0.91292 | 0.91292 | yes |
| covariance shape 5/5 | 2,916 | 1 | 0.91292 | 0.91292 | yes |
| post-asinh transform 1/9 | 2,916 | 1 | 0.91292 | 0.91292 | yes |
| post-asinh transform 2/9 | 2,916 | 1 | 0.911244 | 0.911244 | yes |
| post-asinh transform 3/9 | 2,916 | 1 | 0.911244 | 0.911244 | yes |
| post-asinh transform 4/9 | 2,916 | 1 | 0.911244 | 0.911244 | yes |
| post-asinh transform 5/9 | 2,916 | 1 | 0.911244 | 0.911244 | yes |
| post-asinh transform 6/9 | 2,916 | 1 | 0.911244 | 0.911244 | yes |
| post-asinh transform 7/9 | 2,916 | 1 | 0.911244 | 0.911244 | yes |
| post-asinh transform 8/9 | 2,916 | 1 | 0.911244 | 0.911244 | yes |
| post-asinh transform 9/9 | 2,916 | 1 | 0.911244 | 0.911244 | yes |
| allocation beta | 2,916 | 0.975 | 0.910018 | 0.910018 | yes |
| knot refit | 2,916 | 0.975 | 0.910018 | 0.910018 | yes |
| covariance shape 1/5 | 2,916 | 0.975 | 0.910018 | 0.910018 | yes |
| covariance shape 2/5 | 2,916 | 0.975 | 0.910018 | 0.910018 | yes |
| covariance shape 3/5 | 2,916 | 0.975 | 0.910018 | 0.910018 | yes |
| covariance shape 4/5 | 2,916 | 0.975 | 0.910018 | 0.910018 | yes |
| covariance shape 5/5 | 2,916 | 0.975 | 0.910018 | 0.910018 | yes |
| post-asinh transform 1/9 | 2,916 | 0.975 | 0.910018 | 0.910018 | yes |
| post-asinh transform 2/9 | 2,916 | 0.975 | 0.910018 | 0.910018 | yes |
| post-asinh transform 3/9 | 2,916 | 0.975 | 0.910018 | 0.910018 | yes |
| post-asinh transform 4/9 | 2,916 | 0.975 | 0.910018 | 0.910018 | yes |
| post-asinh transform 5/9 | 2,916 | 0.975 | 0.910018 | 0.910018 | yes |
| post-asinh transform 6/9 | 2,916 | 0.975 | 0.910018 | 0.910018 | yes |
| post-asinh transform 7/9 | 2,916 | 0.975 | 0.910018 | 0.910018 | yes |
| post-asinh transform 8/9 | 2,916 | 0.975 | 0.910018 | 0.910018 | yes |
| post-asinh transform 9/9 | 2,916 | 0.975 | 0.910018 | 0.910018 | yes |
| allocation beta | 2,916 | 1 | 0.908359 | 0.908359 | yes |
| knot refit | 2,916 | 1 | 0.907406 | 0.907406 | yes |
| covariance shape 1/5 | 2,916 | 1 | 0.907406 | 0.907406 | yes |
| covariance shape 2/5 | 2,916 | 1 | 0.907406 | 0.907406 | yes |
| covariance shape 3/5 | 2,916 | 1 | 0.907406 | 0.907406 | yes |
| covariance shape 4/5 | 2,916 | 1 | 0.907406 | 0.907406 | yes |
| covariance shape 5/5 | 2,916 | 1 | 0.907406 | 0.907406 | yes |
| post-asinh transform 1/9 | 2,916 | 1 | 0.907406 | 0.907406 | yes |
| post-asinh transform 2/9 | 2,916 | 1 | 0.907406 | 0.907406 | yes |
| post-asinh transform 3/9 | 2,916 | 1 | 0.907406 | 0.907406 | yes |
| post-asinh transform 4/9 | 2,916 | 1 | 0.907406 | 0.907406 | yes |
| post-asinh transform 5/9 | 2,916 | 1 | 0.907406 | 0.907406 | yes |
| post-asinh transform 6/9 | 2,916 | 1 | 0.907406 | 0.907406 | yes |
| post-asinh transform 7/9 | 2,916 | 1 | 0.907406 | 0.907406 | yes |
| post-asinh transform 8/9 | 2,916 | 1 | 0.907406 | 0.907406 | yes |
| post-asinh transform 9/9 | 2,916 | 1 | 0.907406 | 0.907406 | yes |
| allocation beta | 2,916 | 1 | 0.907406 | 0.907406 | yes |
| knot refit | 2,916 | 1 | 0.907406 | 0.907406 | yes |
| fixed point reached | 2,916 | 1 | 0.907406 | 0.907406 | yes |
| prune proposal 3888->2916 | 2,916 | 1 | 0.907406 | 0.907406 | yes |
| covariance shape 1/5 | 2,916 | 1 | 1.03992 | 1.03992 | no |
| covariance shape 2/5 | 2,916 | 1 | 1.03992 | 1.03992 | no |
| covariance shape 3/5 | 2,916 | 1 | 1.03992 | 1.03992 | no |
| covariance shape 4/5 | 2,916 | 1 | 1.03992 | 1.03992 | no |
| covariance shape 5/5 | 2,916 | 1 | 1.03992 | 1.03992 | no |
| post-asinh transform 1/9 | 2,916 | 1 | 1.03992 | 1.03992 | no |
| post-asinh transform 2/9 | 2,916 | 1 | 1.03992 | 1.03992 | no |
| post-asinh transform 3/9 | 2,916 | 1 | 1.03992 | 1.03992 | no |
| post-asinh transform 4/9 | 2,916 | 1 | 1.03992 | 1.03992 | no |
| post-asinh transform 5/9 | 2,916 | 1 | 1.03992 | 1.03992 | no |
| post-asinh transform 6/9 | 2,916 | 1 | 1.03992 | 1.03992 | no |
| post-asinh transform 7/9 | 2,916 | 1 | 1.03992 | 1.03992 | no |
| post-asinh transform 8/9 | 2,916 | 1 | 1.03992 | 1.03992 | no |
| post-asinh transform 9/9 | 2,916 | 1 | 1.03992 | 1.03992 | no |
| allocation beta | 2,916 | 0.9 | 1.0381 | 1.0381 | no |
| knot refit | 2,916 | 0.9 | 1.03291 | 1.03291 | no |
| covariance shape 1/5 | 2,916 | 0.9 | 1.03291 | 1.03291 | no |
| covariance shape 2/5 | 2,916 | 0.9 | 1.03291 | 1.03291 | no |
| covariance shape 3/5 | 2,916 | 0.9 | 1.03291 | 1.03291 | no |
| covariance shape 4/5 | 2,916 | 0.9 | 1.03291 | 1.03291 | no |
| covariance shape 5/5 | 2,916 | 0.9 | 1.03291 | 1.03291 | no |
| post-asinh transform 1/9 | 2,916 | 0.9 | 1.03291 | 1.03291 | no |
| post-asinh transform 2/9 | 2,916 | 0.9 | 1.03291 | 1.03291 | no |
| post-asinh transform 3/9 | 2,916 | 0.9 | 1.03291 | 1.03291 | no |
| post-asinh transform 4/9 | 2,916 | 0.9 | 1.03291 | 1.03291 | no |
| post-asinh transform 5/9 | 2,916 | 0.9 | 1.03291 | 1.03291 | no |
| post-asinh transform 6/9 | 2,916 | 0.9 | 1.03291 | 1.03291 | no |
| post-asinh transform 7/9 | 2,916 | 0.9 | 1.03291 | 1.03291 | no |
| post-asinh transform 8/9 | 2,916 | 0.9 | 1.03291 | 1.03291 | no |
| post-asinh transform 9/9 | 2,916 | 0.9 | 1.03291 | 1.03291 | no |
| allocation beta | 2,916 | 1 | 1.0233 | 1.0233 | no |
| knot refit | 2,916 | 1 | 1.02191 | 1.02191 | no |
| covariance shape 1/5 | 2,916 | 1 | 1.02191 | 1.02191 | no |
| covariance shape 2/5 | 2,916 | 1 | 1.02191 | 1.02191 | no |
| covariance shape 3/5 | 2,916 | 1 | 1.02191 | 1.02191 | no |
| covariance shape 4/5 | 2,916 | 1 | 1.02191 | 1.02191 | no |
| covariance shape 5/5 | 2,916 | 1 | 1.02191 | 1.02191 | no |
| post-asinh transform 1/9 | 2,916 | 1 | 1.02191 | 1.02191 | no |
| post-asinh transform 2/9 | 2,916 | 1 | 1.02191 | 1.02191 | no |
| post-asinh transform 3/9 | 2,916 | 1 | 1.02191 | 1.02191 | no |
| post-asinh transform 4/9 | 2,916 | 1 | 1.02191 | 1.02191 | no |
| post-asinh transform 5/9 | 2,916 | 1 | 1.02191 | 1.02191 | no |
| post-asinh transform 6/9 | 2,916 | 1 | 1.02191 | 1.02191 | no |
| post-asinh transform 7/9 | 2,916 | 1 | 1.02191 | 1.02191 | no |
| post-asinh transform 8/9 | 2,916 | 1 | 1.02191 | 1.02191 | no |
| post-asinh transform 9/9 | 2,916 | 1 | 1.02191 | 1.02191 | no |
| allocation beta | 2,916 | 1 | 1.02191 | 1.02191 | no |
| knot refit | 2,916 | 1 | 1.01986 | 1.01986 | no |
| covariance shape 1/5 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| covariance shape 2/5 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| covariance shape 3/5 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| covariance shape 4/5 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| covariance shape 5/5 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| post-asinh transform 1/9 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| post-asinh transform 2/9 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| post-asinh transform 3/9 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| post-asinh transform 4/9 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| post-asinh transform 5/9 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| post-asinh transform 6/9 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| post-asinh transform 7/9 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| post-asinh transform 8/9 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| post-asinh transform 9/9 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| allocation beta | 2,916 | 1 | 1.01986 | 1.01986 | no |
| knot refit | 2,916 | 1 | 1.01986 | 1.01986 | no |
| refined parameter-search resolution | 2,916 | 1 | 1.01986 | 1.01986 | no |
| covariance shape 1/5 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| covariance shape 2/5 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| covariance shape 3/5 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| covariance shape 4/5 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| covariance shape 5/5 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| post-asinh transform 1/9 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| post-asinh transform 2/9 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| post-asinh transform 3/9 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| post-asinh transform 4/9 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| post-asinh transform 5/9 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| post-asinh transform 6/9 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| post-asinh transform 7/9 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| post-asinh transform 8/9 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| post-asinh transform 9/9 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| allocation beta | 2,916 | 1 | 1.01986 | 1.01986 | no |
| knot refit (cached stationary) | 2,916 | 1 | 1.01986 | 1.01986 | no |
| refined parameter-search resolution | 2,916 | 1 | 1.01986 | 1.01986 | no |
| covariance shape 1/5 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| covariance shape 2/5 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| covariance shape 3/5 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| covariance shape 4/5 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| covariance shape 5/5 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| post-asinh transform 1/9 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| post-asinh transform 2/9 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| post-asinh transform 3/9 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| post-asinh transform 4/9 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| post-asinh transform 5/9 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| post-asinh transform 6/9 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| post-asinh transform 7/9 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| post-asinh transform 8/9 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| post-asinh transform 9/9 | 2,916 | 1 | 1.01986 | 1.01986 | no |
| allocation beta | 2,916 | 1 | 1.01986 | 1.01986 | no |
| knot refit (cached stationary) | 2,916 | 1 | 1.01986 | 1.01986 | no |
| fixed point reached | 2,916 | 1 | 1.01986 | 1.01986 | no |
| accepted prune | 2,916 | 1 | 0.907406 | 0.907406 | yes |
| cold restart 2916->2187 | 2,187 | 1 | 1.05408 | 1.05408 | no |
| covariance shape 1/5 | 2,187 | 1 | 1.07186 | 1.07186 | no |
| covariance shape 2/5 | 2,187 | 1 | 1.07186 | 1.07186 | no |
| covariance shape 3/5 | 2,187 | 1 | 1.07186 | 1.07186 | no |
| covariance shape 4/5 | 2,187 | 1 | 1.07186 | 1.07186 | no |
| covariance shape 5/5 | 2,187 | 1 | 1.07186 | 1.07186 | no |
| post-asinh transform 1/9 | 2,187 | 1 | 1.07186 | 1.07186 | no |
| post-asinh transform 2/9 | 2,187 | 1 | 1.07186 | 1.07186 | no |
| post-asinh transform 3/9 | 2,187 | 1 | 1.07186 | 1.07186 | no |
| post-asinh transform 4/9 | 2,187 | 1 | 1.07186 | 1.07186 | no |
| post-asinh transform 5/9 | 2,187 | 1 | 1.07186 | 1.07186 | no |
| post-asinh transform 6/9 | 2,187 | 1 | 1.07186 | 1.07186 | no |
| post-asinh transform 7/9 | 2,187 | 1 | 1.07186 | 1.07186 | no |
| post-asinh transform 8/9 | 2,187 | 1 | 1.07186 | 1.07186 | no |
| post-asinh transform 9/9 | 2,187 | 1 | 1.07186 | 1.07186 | no |
| allocation beta | 2,187 | 1 | 1.07186 | 1.07186 | no |
| knot refit | 2,187 | 1 | 1.06794 | 1.06794 | no |
| covariance shape 1/5 | 2,187 | 1 | 1.06794 | 1.06794 | no |
| covariance shape 2/5 | 2,187 | 1 | 1.06794 | 1.06794 | no |
| covariance shape 3/5 | 2,187 | 1 | 1.06794 | 1.06794 | no |
| covariance shape 4/5 | 2,187 | 1 | 1.06794 | 1.06794 | no |
| covariance shape 5/5 | 2,187 | 1 | 1.06794 | 1.06794 | no |
| post-asinh transform 1/9 | 2,187 | 1 | 1.06794 | 1.06794 | no |
| post-asinh transform 2/9 | 2,187 | 1 | 1.06794 | 1.06794 | no |
| post-asinh transform 3/9 | 2,187 | 1 | 1.06794 | 1.06794 | no |
| post-asinh transform 4/9 | 2,187 | 1 | 1.06794 | 1.06794 | no |
| post-asinh transform 5/9 | 2,187 | 1 | 1.06794 | 1.06794 | no |
| post-asinh transform 6/9 | 2,187 | 1 | 1.06794 | 1.06794 | no |
| post-asinh transform 7/9 | 2,187 | 1 | 1.06794 | 1.06794 | no |
| post-asinh transform 8/9 | 2,187 | 1 | 1.06794 | 1.06794 | no |
| post-asinh transform 9/9 | 2,187 | 1 | 1.06794 | 1.06794 | no |
| allocation beta | 2,187 | 1 | 1.06794 | 1.06794 | no |
| knot refit | 2,187 | 1 | 1.06487 | 1.06487 | no |
| covariance shape 1/5 | 2,187 | 1 | 1.06487 | 1.06487 | no |
| covariance shape 2/5 | 2,187 | 1 | 1.06487 | 1.06487 | no |
| covariance shape 3/5 | 2,187 | 1 | 1.06487 | 1.06487 | no |
| covariance shape 4/5 | 2,187 | 1 | 1.06487 | 1.06487 | no |
| covariance shape 5/5 | 2,187 | 1 | 1.06487 | 1.06487 | no |
| post-asinh transform 1/9 | 2,187 | 1 | 1.06487 | 1.06487 | no |
| post-asinh transform 2/9 | 2,187 | 1 | 1.06487 | 1.06487 | no |
| post-asinh transform 3/9 | 2,187 | 1 | 1.06487 | 1.06487 | no |
| post-asinh transform 4/9 | 2,187 | 1 | 1.06487 | 1.06487 | no |
| post-asinh transform 5/9 | 2,187 | 1 | 1.06487 | 1.06487 | no |
| post-asinh transform 6/9 | 2,187 | 1 | 1.06487 | 1.06487 | no |
| post-asinh transform 7/9 | 2,187 | 1 | 1.06487 | 1.06487 | no |
| post-asinh transform 8/9 | 2,187 | 1 | 1.06487 | 1.06487 | no |
| post-asinh transform 9/9 | 2,187 | 1 | 1.06487 | 1.06487 | no |
| allocation beta | 2,187 | 1 | 1.06487 | 1.06487 | no |
| knot refit | 2,187 | 1 | 1.06231 | 1.06231 | no |
| covariance shape 1/5 | 2,187 | 1 | 1.06231 | 1.06231 | no |
| covariance shape 2/5 | 2,187 | 1 | 1.06231 | 1.06231 | no |
| covariance shape 3/5 | 2,187 | 1 | 1.06231 | 1.06231 | no |
| covariance shape 4/5 | 2,187 | 1 | 1.06231 | 1.06231 | no |
| covariance shape 5/5 | 2,187 | 1 | 1.06231 | 1.06231 | no |
| post-asinh transform 1/9 | 2,187 | 1 | 1.06231 | 1.06231 | no |
| post-asinh transform 2/9 | 2,187 | 1 | 1.06231 | 1.06231 | no |
| post-asinh transform 3/9 | 2,187 | 1 | 1.06231 | 1.06231 | no |
| post-asinh transform 4/9 | 2,187 | 1 | 1.06231 | 1.06231 | no |
| post-asinh transform 5/9 | 2,187 | 1 | 1.06231 | 1.06231 | no |
| post-asinh transform 6/9 | 2,187 | 1 | 1.06231 | 1.06231 | no |
| post-asinh transform 7/9 | 2,187 | 1 | 1.06231 | 1.06231 | no |
| post-asinh transform 8/9 | 2,187 | 1 | 1.06231 | 1.06231 | no |
| post-asinh transform 9/9 | 2,187 | 1 | 1.06231 | 1.06231 | no |
| allocation beta | 2,187 | 1 | 1.06231 | 1.06231 | no |
| knot refit | 2,187 | 1 | 1.06055 | 1.06055 | no |
| covariance shape 1/5 | 2,187 | 1 | 1.06055 | 1.06055 | no |
| covariance shape 2/5 | 2,187 | 1 | 1.06055 | 1.06055 | no |
| covariance shape 3/5 | 2,187 | 1 | 1.06055 | 1.06055 | no |
| covariance shape 4/5 | 2,187 | 1 | 1.06055 | 1.06055 | no |
| covariance shape 5/5 | 2,187 | 1 | 1.06055 | 1.06055 | no |
| post-asinh transform 1/9 | 2,187 | 1 | 1.06055 | 1.06055 | no |
| post-asinh transform 2/9 | 2,187 | 1 | 1.06055 | 1.06055 | no |
| post-asinh transform 3/9 | 2,187 | 1 | 1.06055 | 1.06055 | no |
| post-asinh transform 4/9 | 2,187 | 1 | 1.06055 | 1.06055 | no |
| post-asinh transform 5/9 | 2,187 | 1 | 1.06055 | 1.06055 | no |
| post-asinh transform 6/9 | 2,187 | 1 | 1.06055 | 1.06055 | no |
| post-asinh transform 7/9 | 2,187 | 1 | 1.06055 | 1.06055 | no |
| post-asinh transform 8/9 | 2,187 | 1 | 1.06055 | 1.06055 | no |
| post-asinh transform 9/9 | 2,187 | 1 | 1.06055 | 1.06055 | no |
| allocation beta | 2,187 | 1 | 1.06055 | 1.06055 | no |
| knot refit | 2,187 | 1 | 1.05929 | 1.05929 | no |
| covariance shape 1/5 | 2,187 | 1 | 1.05929 | 1.05929 | no |
| covariance shape 2/5 | 2,187 | 1 | 1.05929 | 1.05929 | no |
| covariance shape 3/5 | 2,187 | 1 | 1.05929 | 1.05929 | no |
| covariance shape 4/5 | 2,187 | 1 | 1.05929 | 1.05929 | no |
| covariance shape 5/5 | 2,187 | 1 | 1.05929 | 1.05929 | no |
| post-asinh transform 1/9 | 2,187 | 1 | 1.05929 | 1.05929 | no |
| post-asinh transform 2/9 | 2,187 | 1 | 1.05929 | 1.05929 | no |
| post-asinh transform 3/9 | 2,187 | 1 | 1.05929 | 1.05929 | no |
| post-asinh transform 4/9 | 2,187 | 1 | 1.05929 | 1.05929 | no |
| post-asinh transform 5/9 | 2,187 | 1 | 1.05929 | 1.05929 | no |
| post-asinh transform 6/9 | 2,187 | 1 | 1.05929 | 1.05929 | no |
| post-asinh transform 7/9 | 2,187 | 1 | 1.05929 | 1.05929 | no |
| post-asinh transform 8/9 | 2,187 | 1 | 1.05929 | 1.05929 | no |
| post-asinh transform 9/9 | 2,187 | 1 | 1.05929 | 1.05929 | no |
| allocation beta | 2,187 | 1 | 1.05929 | 1.05929 | no |
| knot refit | 2,187 | 1 | 1.05929 | 1.05929 | no |
| refined parameter-search resolution | 2,187 | 1 | 1.05929 | 1.05929 | no |
| covariance shape 1/5 | 2,187 | 1 | 1.05929 | 1.05929 | no |
| covariance shape 2/5 | 2,187 | 1 | 1.05929 | 1.05929 | no |
| covariance shape 3/5 | 2,187 | 1 | 1.05929 | 1.05929 | no |
| covariance shape 4/5 | 2,187 | 1 | 1.05929 | 1.05929 | no |
| covariance shape 5/5 | 2,187 | 1 | 1.05929 | 1.05929 | no |
| post-asinh transform 1/9 | 2,187 | 1 | 1.05929 | 1.05929 | no |
| post-asinh transform 2/9 | 2,187 | 1 | 1.05929 | 1.05929 | no |
| post-asinh transform 3/9 | 2,187 | 1 | 1.05929 | 1.05929 | no |
| post-asinh transform 4/9 | 2,187 | 1 | 1.05929 | 1.05929 | no |
| post-asinh transform 5/9 | 2,187 | 1 | 1.05929 | 1.05929 | no |
| post-asinh transform 6/9 | 2,187 | 1 | 1.05929 | 1.05929 | no |
| post-asinh transform 7/9 | 2,187 | 1 | 1.05929 | 1.05929 | no |
| post-asinh transform 8/9 | 2,187 | 1 | 1.05929 | 1.05929 | no |
| post-asinh transform 9/9 | 2,187 | 1 | 1.05929 | 1.05929 | no |
| allocation beta | 2,187 | 1 | 1.05929 | 1.05929 | no |
| knot refit (cached stationary) | 2,187 | 1 | 1.05929 | 1.05929 | no |
| refined parameter-search resolution | 2,187 | 1 | 1.05929 | 1.05929 | no |
| covariance shape 1/5 | 2,187 | 1 | 1.05929 | 1.05929 | no |
| covariance shape 2/5 | 2,187 | 1 | 1.05929 | 1.05929 | no |
| covariance shape 3/5 | 2,187 | 1 | 1.05929 | 1.05929 | no |
| covariance shape 4/5 | 2,187 | 1 | 1.05929 | 1.05929 | no |
| covariance shape 5/5 | 2,187 | 1 | 1.05929 | 1.05929 | no |
| post-asinh transform 1/9 | 2,187 | 1 | 1.05929 | 1.05929 | no |
| post-asinh transform 2/9 | 2,187 | 1 | 1.05748 | 1.05748 | no |
| post-asinh transform 3/9 | 2,187 | 1 | 1.05748 | 1.05748 | no |
| post-asinh transform 4/9 | 2,187 | 1 | 1.05748 | 1.05748 | no |
| post-asinh transform 5/9 | 2,187 | 1 | 1.05748 | 1.05748 | no |
| post-asinh transform 6/9 | 2,187 | 1 | 1.05748 | 1.05748 | no |
| post-asinh transform 7/9 | 2,187 | 1 | 1.05748 | 1.05748 | no |
| post-asinh transform 8/9 | 2,187 | 1 | 1.05748 | 1.05748 | no |
| post-asinh transform 9/9 | 2,187 | 1 | 1.05748 | 1.05748 | no |
| allocation beta | 2,187 | 0.975 | 1.05614 | 1.05614 | no |
| knot refit | 2,187 | 0.975 | 1.05614 | 1.05614 | no |
| covariance shape 1/5 | 2,187 | 0.975 | 1.05614 | 1.05614 | no |
| covariance shape 2/5 | 2,187 | 0.975 | 1.05614 | 1.05614 | no |
| covariance shape 3/5 | 2,187 | 0.975 | 1.05614 | 1.05614 | no |
| covariance shape 4/5 | 2,187 | 0.975 | 1.05614 | 1.05614 | no |
| covariance shape 5/5 | 2,187 | 0.975 | 1.05614 | 1.05614 | no |
| post-asinh transform 1/9 | 2,187 | 0.975 | 1.05614 | 1.05614 | no |
| post-asinh transform 2/9 | 2,187 | 0.975 | 1.05614 | 1.05614 | no |
| post-asinh transform 3/9 | 2,187 | 0.975 | 1.05614 | 1.05614 | no |
| post-asinh transform 4/9 | 2,187 | 0.975 | 1.05614 | 1.05614 | no |
| post-asinh transform 5/9 | 2,187 | 0.975 | 1.05614 | 1.05614 | no |
| post-asinh transform 6/9 | 2,187 | 0.975 | 1.05614 | 1.05614 | no |
| post-asinh transform 7/9 | 2,187 | 0.975 | 1.05614 | 1.05614 | no |
| post-asinh transform 8/9 | 2,187 | 0.975 | 1.05614 | 1.05614 | no |
| post-asinh transform 9/9 | 2,187 | 0.975 | 1.05614 | 1.05614 | no |
| allocation beta | 2,187 | 1 | 1.05408 | 1.05408 | no |
| knot refit | 2,187 | 1 | 1.05408 | 1.05408 | no |
| covariance shape 1/5 | 2,187 | 1 | 1.05408 | 1.05408 | no |
| covariance shape 2/5 | 2,187 | 1 | 1.05408 | 1.05408 | no |
| covariance shape 3/5 | 2,187 | 1 | 1.05408 | 1.05408 | no |
| covariance shape 4/5 | 2,187 | 1 | 1.05408 | 1.05408 | no |
| covariance shape 5/5 | 2,187 | 1 | 1.05408 | 1.05408 | no |
| post-asinh transform 1/9 | 2,187 | 1 | 1.05408 | 1.05408 | no |
| post-asinh transform 2/9 | 2,187 | 1 | 1.05408 | 1.05408 | no |
| post-asinh transform 3/9 | 2,187 | 1 | 1.05408 | 1.05408 | no |
| post-asinh transform 4/9 | 2,187 | 1 | 1.05408 | 1.05408 | no |
| post-asinh transform 5/9 | 2,187 | 1 | 1.05408 | 1.05408 | no |
| post-asinh transform 6/9 | 2,187 | 1 | 1.05408 | 1.05408 | no |
| post-asinh transform 7/9 | 2,187 | 1 | 1.05408 | 1.05408 | no |
| post-asinh transform 8/9 | 2,187 | 1 | 1.05408 | 1.05408 | no |
| post-asinh transform 9/9 | 2,187 | 1 | 1.05408 | 1.05408 | no |
| allocation beta | 2,187 | 1 | 1.05408 | 1.05408 | no |
| knot refit (cached stationary) | 2,187 | 1 | 1.05408 | 1.05408 | no |
| fixed point reached | 2,187 | 1 | 1.05408 | 1.05408 | no |
| prune proposal 2916->2187 | 2,187 | 1 | 1.05408 | 1.05408 | no |
| covariance shape 1/5 | 2,187 | 1 | 1.57426 | 1.57426 | no |
| covariance shape 2/5 | 2,187 | 1 | 1.57426 | 1.57426 | no |
| covariance shape 3/5 | 2,187 | 1 | 1.57426 | 1.57426 | no |
| covariance shape 4/5 | 2,187 | 1 | 1.57426 | 1.57426 | no |
| covariance shape 5/5 | 2,187 | 1 | 1.57426 | 1.57426 | no |
| post-asinh transform 1/9 | 2,187 | 1 | 1.57426 | 1.57426 | no |
| post-asinh transform 2/9 | 2,187 | 1 | 1.57426 | 1.57426 | no |
| post-asinh transform 3/9 | 2,187 | 1 | 1.57426 | 1.57426 | no |
| post-asinh transform 4/9 | 2,187 | 1 | 1.57426 | 1.57426 | no |
| post-asinh transform 5/9 | 2,187 | 1 | 1.57426 | 1.57426 | no |
| post-asinh transform 6/9 | 2,187 | 1 | 1.57426 | 1.57426 | no |
| post-asinh transform 7/9 | 2,187 | 1 | 1.57426 | 1.57426 | no |
| post-asinh transform 8/9 | 2,187 | 1 | 1.57426 | 1.57426 | no |
| post-asinh transform 9/9 | 2,187 | 1 | 1.57426 | 1.57426 | no |
| allocation beta | 2,187 | 0.9 | 1.4482 | 1.4482 | no |
| knot refit | 2,187 | 0.9 | 1.44243 | 1.44243 | no |
| covariance shape 1/5 | 2,187 | 0.9 | 1.44243 | 1.44243 | no |
| covariance shape 2/5 | 2,187 | 0.9 | 1.44243 | 1.44243 | no |
| covariance shape 3/5 | 2,187 | 0.9 | 1.44243 | 1.44243 | no |
| covariance shape 4/5 | 2,187 | 0.9 | 1.44243 | 1.44243 | no |
| covariance shape 5/5 | 2,187 | 0.9 | 1.44243 | 1.44243 | no |
| post-asinh transform 1/9 | 2,187 | 0.9 | 1.44243 | 1.44243 | no |
| post-asinh transform 2/9 | 2,187 | 0.9 | 1.44243 | 1.44243 | no |
| post-asinh transform 3/9 | 2,187 | 0.9 | 1.44243 | 1.44243 | no |
| post-asinh transform 4/9 | 2,187 | 0.9 | 1.44243 | 1.44243 | no |
| post-asinh transform 5/9 | 2,187 | 0.9 | 1.44243 | 1.44243 | no |
| post-asinh transform 6/9 | 2,187 | 0.9 | 1.44243 | 1.44243 | no |
| post-asinh transform 7/9 | 2,187 | 0.9 | 1.44243 | 1.44243 | no |
| post-asinh transform 8/9 | 2,187 | 0.9 | 1.44243 | 1.44243 | no |
| post-asinh transform 9/9 | 2,187 | 0.9 | 1.44243 | 1.44243 | no |
| allocation beta | 2,187 | 1 | 1.42175 | 1.42175 | no |
| knot refit | 2,187 | 1 | 1.41918 | 1.41918 | no |
| covariance shape 1/5 | 2,187 | 1 | 1.41918 | 1.41918 | no |
| covariance shape 2/5 | 2,187 | 1 | 1.41918 | 1.41918 | no |
| covariance shape 3/5 | 2,187 | 1 | 1.41918 | 1.41918 | no |
| covariance shape 4/5 | 2,187 | 1 | 1.41918 | 1.41918 | no |
| covariance shape 5/5 | 2,187 | 1 | 1.41918 | 1.41918 | no |
| post-asinh transform 1/9 | 2,187 | 1 | 1.41918 | 1.41918 | no |
| post-asinh transform 2/9 | 2,187 | 1 | 1.41918 | 1.41918 | no |
| post-asinh transform 3/9 | 2,187 | 1 | 1.41918 | 1.41918 | no |
| post-asinh transform 4/9 | 2,187 | 1 | 1.41918 | 1.41918 | no |
| post-asinh transform 5/9 | 2,187 | 1 | 1.41918 | 1.41918 | no |
| post-asinh transform 6/9 | 2,187 | 1 | 1.41918 | 1.41918 | no |
| post-asinh transform 7/9 | 2,187 | 1 | 1.41918 | 1.41918 | no |
| post-asinh transform 8/9 | 2,187 | 1 | 1.41918 | 1.41918 | no |
| post-asinh transform 9/9 | 2,187 | 1 | 1.41918 | 1.41918 | no |
| allocation beta | 2,187 | 1 | 1.41918 | 1.41918 | no |
| knot refit | 2,187 | 1 | 1.41719 | 1.41719 | no |
| covariance shape 1/5 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| covariance shape 2/5 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| covariance shape 3/5 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| covariance shape 4/5 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| covariance shape 5/5 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| post-asinh transform 1/9 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| post-asinh transform 2/9 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| post-asinh transform 3/9 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| post-asinh transform 4/9 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| post-asinh transform 5/9 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| post-asinh transform 6/9 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| post-asinh transform 7/9 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| post-asinh transform 8/9 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| post-asinh transform 9/9 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| allocation beta | 2,187 | 1 | 1.41719 | 1.41719 | no |
| knot refit | 2,187 | 1 | 1.41719 | 1.41719 | no |
| refined parameter-search resolution | 2,187 | 1 | 1.41719 | 1.41719 | no |
| covariance shape 1/5 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| covariance shape 2/5 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| covariance shape 3/5 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| covariance shape 4/5 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| covariance shape 5/5 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| post-asinh transform 1/9 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| post-asinh transform 2/9 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| post-asinh transform 3/9 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| post-asinh transform 4/9 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| post-asinh transform 5/9 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| post-asinh transform 6/9 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| post-asinh transform 7/9 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| post-asinh transform 8/9 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| post-asinh transform 9/9 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| allocation beta | 2,187 | 1 | 1.41719 | 1.41719 | no |
| knot refit (cached stationary) | 2,187 | 1 | 1.41719 | 1.41719 | no |
| refined parameter-search resolution | 2,187 | 1 | 1.41719 | 1.41719 | no |
| covariance shape 1/5 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| covariance shape 2/5 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| covariance shape 3/5 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| covariance shape 4/5 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| covariance shape 5/5 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| post-asinh transform 1/9 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| post-asinh transform 2/9 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| post-asinh transform 3/9 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| post-asinh transform 4/9 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| post-asinh transform 5/9 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| post-asinh transform 6/9 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| post-asinh transform 7/9 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| post-asinh transform 8/9 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| post-asinh transform 9/9 | 2,187 | 1 | 1.41719 | 1.41719 | no |
| allocation beta | 2,187 | 1 | 1.41719 | 1.41719 | no |
| knot refit (cached stationary) | 2,187 | 1 | 1.41719 | 1.41719 | no |
| fixed point reached | 2,187 | 1 | 1.41719 | 1.41719 | no |
| rejected prune | 2,187 | 1 | 1.05408 | 1.05408 | no |
| cold restart 2916->2552 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| covariance shape 1/5 | 2,552 | 1 | 1.01701 | 1.01701 | no |
| covariance shape 2/5 | 2,552 | 1 | 1.01701 | 1.01701 | no |
| covariance shape 3/5 | 2,552 | 1 | 1.01701 | 1.01701 | no |
| covariance shape 4/5 | 2,552 | 1 | 1.01701 | 1.01701 | no |
| covariance shape 5/5 | 2,552 | 1 | 1.01701 | 1.01701 | no |
| post-asinh transform 1/9 | 2,552 | 1 | 1.01701 | 1.01701 | no |
| post-asinh transform 2/9 | 2,552 | 1 | 1.01701 | 1.01701 | no |
| post-asinh transform 3/9 | 2,552 | 1 | 1.01701 | 1.01701 | no |
| post-asinh transform 4/9 | 2,552 | 1 | 1.01701 | 1.01701 | no |
| post-asinh transform 5/9 | 2,552 | 1 | 1.01701 | 1.01701 | no |
| post-asinh transform 6/9 | 2,552 | 1 | 1.01701 | 1.01701 | no |
| post-asinh transform 7/9 | 2,552 | 1 | 1.01701 | 1.01701 | no |
| post-asinh transform 8/9 | 2,552 | 1 | 1.01701 | 1.01701 | no |
| post-asinh transform 9/9 | 2,552 | 1 | 1.01701 | 1.01701 | no |
| allocation beta | 2,552 | 0.9 | 1.01191 | 1.01191 | no |
| knot refit | 2,552 | 0.9 | 1.00728 | 1.00728 | no |
| covariance shape 1/5 | 2,552 | 0.9 | 1.00728 | 1.00728 | no |
| covariance shape 2/5 | 2,552 | 0.9 | 1.00728 | 1.00728 | no |
| covariance shape 3/5 | 2,552 | 0.9 | 1.00728 | 1.00728 | no |
| covariance shape 4/5 | 2,552 | 0.9 | 1.00728 | 1.00728 | no |
| covariance shape 5/5 | 2,552 | 0.9 | 1.00728 | 1.00728 | no |
| post-asinh transform 1/9 | 2,552 | 0.9 | 1.00728 | 1.00728 | no |
| post-asinh transform 2/9 | 2,552 | 0.9 | 1.00728 | 1.00728 | no |
| post-asinh transform 3/9 | 2,552 | 0.9 | 1.00728 | 1.00728 | no |
| post-asinh transform 4/9 | 2,552 | 0.9 | 1.00728 | 1.00728 | no |
| post-asinh transform 5/9 | 2,552 | 0.9 | 1.00728 | 1.00728 | no |
| post-asinh transform 6/9 | 2,552 | 0.9 | 1.00728 | 1.00728 | no |
| post-asinh transform 7/9 | 2,552 | 0.9 | 1.00728 | 1.00728 | no |
| post-asinh transform 8/9 | 2,552 | 0.9 | 1.00728 | 1.00728 | no |
| post-asinh transform 9/9 | 2,552 | 0.9 | 1.00728 | 1.00728 | no |
| allocation beta | 2,552 | 1 | 0.997253 | 0.997253 | yes |
| knot refit | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| covariance shape 1/5 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| covariance shape 2/5 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| covariance shape 3/5 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| covariance shape 4/5 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| covariance shape 5/5 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| post-asinh transform 1/9 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| post-asinh transform 2/9 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| post-asinh transform 3/9 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| post-asinh transform 4/9 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| post-asinh transform 5/9 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| post-asinh transform 6/9 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| post-asinh transform 7/9 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| post-asinh transform 8/9 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| post-asinh transform 9/9 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| allocation beta | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| knot refit | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| refined parameter-search resolution | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| covariance shape 1/5 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| covariance shape 2/5 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| covariance shape 3/5 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| covariance shape 4/5 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| covariance shape 5/5 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| post-asinh transform 1/9 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| post-asinh transform 2/9 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| post-asinh transform 3/9 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| post-asinh transform 4/9 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| post-asinh transform 5/9 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| post-asinh transform 6/9 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| post-asinh transform 7/9 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| post-asinh transform 8/9 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| post-asinh transform 9/9 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| allocation beta | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| knot refit (cached stationary) | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| refined parameter-search resolution | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| covariance shape 1/5 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| covariance shape 2/5 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| covariance shape 3/5 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| covariance shape 4/5 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| covariance shape 5/5 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| post-asinh transform 1/9 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| post-asinh transform 2/9 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| post-asinh transform 3/9 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| post-asinh transform 4/9 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| post-asinh transform 5/9 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| post-asinh transform 6/9 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| post-asinh transform 7/9 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| post-asinh transform 8/9 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| post-asinh transform 9/9 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| allocation beta | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| knot refit (cached stationary) | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| fixed point reached | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| prune proposal 2916->2552 | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| covariance shape 1/5 | 2,552 | 1 | 1.11764 | 1.11764 | no |
| covariance shape 2/5 | 2,552 | 1 | 1.11764 | 1.11764 | no |
| covariance shape 3/5 | 2,552 | 1 | 1.11764 | 1.11764 | no |
| covariance shape 4/5 | 2,552 | 1 | 1.11764 | 1.11764 | no |
| covariance shape 5/5 | 2,552 | 1 | 1.11764 | 1.11764 | no |
| post-asinh transform 1/9 | 2,552 | 1 | 1.11764 | 1.11764 | no |
| post-asinh transform 2/9 | 2,552 | 1 | 1.11764 | 1.11764 | no |
| post-asinh transform 3/9 | 2,552 | 1 | 1.11764 | 1.11764 | no |
| post-asinh transform 4/9 | 2,552 | 1 | 1.11764 | 1.11764 | no |
| post-asinh transform 5/9 | 2,552 | 1 | 1.11764 | 1.11764 | no |
| post-asinh transform 6/9 | 2,552 | 1 | 1.11764 | 1.11764 | no |
| post-asinh transform 7/9 | 2,552 | 1 | 1.11764 | 1.11764 | no |
| post-asinh transform 8/9 | 2,552 | 1 | 1.11764 | 1.11764 | no |
| post-asinh transform 9/9 | 2,552 | 1 | 1.11764 | 1.11764 | no |
| allocation beta | 2,552 | 0.9 | 1.03329 | 1.03329 | no |
| knot refit | 2,552 | 0.9 | 1.03182 | 1.03182 | no |
| covariance shape 1/5 | 2,552 | 0.9 | 1.03182 | 1.03182 | no |
| covariance shape 2/5 | 2,552 | 0.9 | 1.03182 | 1.03182 | no |
| covariance shape 3/5 | 2,552 | 0.9 | 1.03182 | 1.03182 | no |
| covariance shape 4/5 | 2,552 | 0.9 | 1.03182 | 1.03182 | no |
| covariance shape 5/5 | 2,552 | 0.9 | 1.03182 | 1.03182 | no |
| post-asinh transform 1/9 | 2,552 | 0.9 | 1.03182 | 1.03182 | no |
| post-asinh transform 2/9 | 2,552 | 0.9 | 1.03182 | 1.03182 | no |
| post-asinh transform 3/9 | 2,552 | 0.9 | 1.03182 | 1.03182 | no |
| post-asinh transform 4/9 | 2,552 | 0.9 | 1.03182 | 1.03182 | no |
| post-asinh transform 5/9 | 2,552 | 0.9 | 1.03182 | 1.03182 | no |
| post-asinh transform 6/9 | 2,552 | 0.9 | 1.03182 | 1.03182 | no |
| post-asinh transform 7/9 | 2,552 | 0.9 | 1.03182 | 1.03182 | no |
| post-asinh transform 8/9 | 2,552 | 0.9 | 1.03182 | 1.03182 | no |
| post-asinh transform 9/9 | 2,552 | 0.9 | 1.03182 | 1.03182 | no |
| allocation beta | 2,552 | 1 | 1.02708 | 1.02708 | no |
| knot refit | 2,552 | 1 | 1.02708 | 1.02708 | no |
| covariance shape 1/5 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| covariance shape 2/5 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| covariance shape 3/5 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| covariance shape 4/5 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| covariance shape 5/5 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| post-asinh transform 1/9 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| post-asinh transform 2/9 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| post-asinh transform 3/9 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| post-asinh transform 4/9 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| post-asinh transform 5/9 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| post-asinh transform 6/9 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| post-asinh transform 7/9 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| post-asinh transform 8/9 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| post-asinh transform 9/9 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| allocation beta | 2,552 | 1 | 1.02708 | 1.02708 | no |
| knot refit (cached stationary) | 2,552 | 1 | 1.02708 | 1.02708 | no |
| refined parameter-search resolution | 2,552 | 1 | 1.02708 | 1.02708 | no |
| covariance shape 1/5 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| covariance shape 2/5 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| covariance shape 3/5 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| covariance shape 4/5 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| covariance shape 5/5 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| post-asinh transform 1/9 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| post-asinh transform 2/9 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| post-asinh transform 3/9 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| post-asinh transform 4/9 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| post-asinh transform 5/9 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| post-asinh transform 6/9 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| post-asinh transform 7/9 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| post-asinh transform 8/9 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| post-asinh transform 9/9 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| allocation beta | 2,552 | 1 | 1.02708 | 1.02708 | no |
| knot refit (cached stationary) | 2,552 | 1 | 1.02708 | 1.02708 | no |
| refined parameter-search resolution | 2,552 | 1 | 1.02708 | 1.02708 | no |
| covariance shape 1/5 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| covariance shape 2/5 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| covariance shape 3/5 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| covariance shape 4/5 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| covariance shape 5/5 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| post-asinh transform 1/9 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| post-asinh transform 2/9 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| post-asinh transform 3/9 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| post-asinh transform 4/9 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| post-asinh transform 5/9 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| post-asinh transform 6/9 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| post-asinh transform 7/9 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| post-asinh transform 8/9 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| post-asinh transform 9/9 | 2,552 | 1 | 1.02708 | 1.02708 | no |
| allocation beta | 2,552 | 1 | 1.02708 | 1.02708 | no |
| knot refit (cached stationary) | 2,552 | 1 | 1.02708 | 1.02708 | no |
| fixed point reached | 2,552 | 1 | 1.02708 | 1.02708 | no |
| accepted prune | 2,552 | 1 | 0.996155 | 0.996155 | yes |
| cold restart 2552->2188 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| covariance shape 1/5 | 2,188 | 1 | 1.08018 | 1.08018 | no |
| covariance shape 2/5 | 2,188 | 1 | 1.08018 | 1.08018 | no |
| covariance shape 3/5 | 2,188 | 1 | 1.08018 | 1.08018 | no |
| covariance shape 4/5 | 2,188 | 1 | 1.08018 | 1.08018 | no |
| covariance shape 5/5 | 2,188 | 1 | 1.08018 | 1.08018 | no |
| post-asinh transform 1/9 | 2,188 | 1 | 1.08018 | 1.08018 | no |
| post-asinh transform 2/9 | 2,188 | 1 | 1.08018 | 1.08018 | no |
| post-asinh transform 3/9 | 2,188 | 1 | 1.08018 | 1.08018 | no |
| post-asinh transform 4/9 | 2,188 | 1 | 1.08018 | 1.08018 | no |
| post-asinh transform 5/9 | 2,188 | 1 | 1.08018 | 1.08018 | no |
| post-asinh transform 6/9 | 2,188 | 1 | 1.08018 | 1.08018 | no |
| post-asinh transform 7/9 | 2,188 | 1 | 1.08018 | 1.08018 | no |
| post-asinh transform 8/9 | 2,188 | 1 | 1.08018 | 1.08018 | no |
| post-asinh transform 9/9 | 2,188 | 1 | 1.08018 | 1.08018 | no |
| allocation beta | 2,188 | 0.9 | 1.07666 | 1.07666 | no |
| knot refit | 2,188 | 0.9 | 1.07177 | 1.07177 | no |
| covariance shape 1/5 | 2,188 | 0.9 | 1.07177 | 1.07177 | no |
| covariance shape 2/5 | 2,188 | 0.9 | 1.07177 | 1.07177 | no |
| covariance shape 3/5 | 2,188 | 0.9 | 1.07177 | 1.07177 | no |
| covariance shape 4/5 | 2,188 | 0.9 | 1.07177 | 1.07177 | no |
| covariance shape 5/5 | 2,188 | 0.9 | 1.07177 | 1.07177 | no |
| post-asinh transform 1/9 | 2,188 | 0.9 | 1.07177 | 1.07177 | no |
| post-asinh transform 2/9 | 2,188 | 0.9 | 1.07177 | 1.07177 | no |
| post-asinh transform 3/9 | 2,188 | 0.9 | 1.07177 | 1.07177 | no |
| post-asinh transform 4/9 | 2,188 | 0.9 | 1.07177 | 1.07177 | no |
| post-asinh transform 5/9 | 2,188 | 0.9 | 1.07177 | 1.07177 | no |
| post-asinh transform 6/9 | 2,188 | 0.9 | 1.07177 | 1.07177 | no |
| post-asinh transform 7/9 | 2,188 | 0.9 | 1.07177 | 1.07177 | no |
| post-asinh transform 8/9 | 2,188 | 0.9 | 1.07177 | 1.07177 | no |
| post-asinh transform 9/9 | 2,188 | 0.9 | 1.07177 | 1.07177 | no |
| allocation beta | 2,188 | 1 | 1.06499 | 1.06499 | no |
| knot refit | 2,188 | 1 | 1.0632 | 1.0632 | no |
| covariance shape 1/5 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| covariance shape 2/5 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| covariance shape 3/5 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| covariance shape 4/5 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| covariance shape 5/5 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| post-asinh transform 1/9 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| post-asinh transform 2/9 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| post-asinh transform 3/9 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| post-asinh transform 4/9 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| post-asinh transform 5/9 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| post-asinh transform 6/9 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| post-asinh transform 7/9 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| post-asinh transform 8/9 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| post-asinh transform 9/9 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| allocation beta | 2,188 | 1 | 1.0632 | 1.0632 | no |
| knot refit | 2,188 | 1 | 1.0632 | 1.0632 | no |
| refined parameter-search resolution | 2,188 | 1 | 1.0632 | 1.0632 | no |
| covariance shape 1/5 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| covariance shape 2/5 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| covariance shape 3/5 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| covariance shape 4/5 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| covariance shape 5/5 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| post-asinh transform 1/9 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| post-asinh transform 2/9 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| post-asinh transform 3/9 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| post-asinh transform 4/9 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| post-asinh transform 5/9 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| post-asinh transform 6/9 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| post-asinh transform 7/9 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| post-asinh transform 8/9 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| post-asinh transform 9/9 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| allocation beta | 2,188 | 1 | 1.0632 | 1.0632 | no |
| knot refit (cached stationary) | 2,188 | 1 | 1.0632 | 1.0632 | no |
| refined parameter-search resolution | 2,188 | 1 | 1.0632 | 1.0632 | no |
| covariance shape 1/5 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| covariance shape 2/5 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| covariance shape 3/5 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| covariance shape 4/5 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| covariance shape 5/5 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| post-asinh transform 1/9 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| post-asinh transform 2/9 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| post-asinh transform 3/9 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| post-asinh transform 4/9 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| post-asinh transform 5/9 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| post-asinh transform 6/9 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| post-asinh transform 7/9 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| post-asinh transform 8/9 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| post-asinh transform 9/9 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| allocation beta | 2,188 | 1 | 1.0632 | 1.0632 | no |
| knot refit (cached stationary) | 2,188 | 1 | 1.0632 | 1.0632 | no |
| fixed point reached | 2,188 | 1 | 1.0632 | 1.0632 | no |
| prune proposal 2552->2188 | 2,188 | 1 | 1.0632 | 1.0632 | no |
| covariance shape 1/5 | 2,188 | 1 | 1.37579 | 1.37579 | no |
| covariance shape 2/5 | 2,188 | 1 | 1.37579 | 1.37579 | no |
| covariance shape 3/5 | 2,188 | 1 | 1.37579 | 1.37579 | no |
| covariance shape 4/5 | 2,188 | 1 | 1.37579 | 1.37579 | no |
| covariance shape 5/5 | 2,188 | 1 | 1.37579 | 1.37579 | no |
| post-asinh transform 1/9 | 2,188 | 1 | 1.37579 | 1.37579 | no |
| post-asinh transform 2/9 | 2,188 | 1 | 1.37579 | 1.37579 | no |
| post-asinh transform 3/9 | 2,188 | 1 | 1.37579 | 1.37579 | no |
| post-asinh transform 4/9 | 2,188 | 1 | 1.37579 | 1.37579 | no |
| post-asinh transform 5/9 | 2,188 | 1 | 1.37579 | 1.37579 | no |
| post-asinh transform 6/9 | 2,188 | 1 | 1.37579 | 1.37579 | no |
| post-asinh transform 7/9 | 2,188 | 1 | 1.37579 | 1.37579 | no |
| post-asinh transform 8/9 | 2,188 | 1 | 1.37579 | 1.37579 | no |
| post-asinh transform 9/9 | 2,188 | 1 | 1.37579 | 1.37579 | no |
| allocation beta | 2,188 | 0.9 | 1.35423 | 1.35423 | no |
| knot refit | 2,188 | 0.9 | 1.34696 | 1.34696 | no |
| covariance shape 1/5 | 2,188 | 0.9 | 1.34696 | 1.34696 | no |
| covariance shape 2/5 | 2,188 | 0.9 | 1.34696 | 1.34696 | no |
| covariance shape 3/5 | 2,188 | 0.9 | 1.34696 | 1.34696 | no |
| covariance shape 4/5 | 2,188 | 0.9 | 1.34696 | 1.34696 | no |
| covariance shape 5/5 | 2,188 | 0.9 | 1.34696 | 1.34696 | no |
| post-asinh transform 1/9 | 2,188 | 0.9 | 1.34696 | 1.34696 | no |
| post-asinh transform 2/9 | 2,188 | 0.9 | 1.34696 | 1.34696 | no |
| post-asinh transform 3/9 | 2,188 | 0.9 | 1.34696 | 1.34696 | no |
| post-asinh transform 4/9 | 2,188 | 0.9 | 1.34696 | 1.34696 | no |
| post-asinh transform 5/9 | 2,188 | 0.9 | 1.34696 | 1.34696 | no |
| post-asinh transform 6/9 | 2,188 | 0.9 | 1.34696 | 1.34696 | no |
| post-asinh transform 7/9 | 2,188 | 0.9 | 1.34696 | 1.34696 | no |
| post-asinh transform 8/9 | 2,188 | 0.9 | 1.34696 | 1.34696 | no |
| post-asinh transform 9/9 | 2,188 | 0.9 | 1.34696 | 1.34696 | no |
| allocation beta | 2,188 | 1 | 1.33376 | 1.33376 | no |
| knot refit | 2,188 | 1 | 1.3301 | 1.3301 | no |
| covariance shape 1/5 | 2,188 | 1 | 1.3301 | 1.3301 | no |
| covariance shape 2/5 | 2,188 | 1 | 1.3301 | 1.3301 | no |
| covariance shape 3/5 | 2,188 | 1 | 1.3301 | 1.3301 | no |
| covariance shape 4/5 | 2,188 | 1 | 1.3301 | 1.3301 | no |
| covariance shape 5/5 | 2,188 | 1 | 1.3301 | 1.3301 | no |
| post-asinh transform 1/9 | 2,188 | 1 | 1.3301 | 1.3301 | no |
| post-asinh transform 2/9 | 2,188 | 1 | 1.3301 | 1.3301 | no |
| post-asinh transform 3/9 | 2,188 | 1 | 1.3301 | 1.3301 | no |
| post-asinh transform 4/9 | 2,188 | 1 | 1.3301 | 1.3301 | no |
| post-asinh transform 5/9 | 2,188 | 1 | 1.3301 | 1.3301 | no |
| post-asinh transform 6/9 | 2,188 | 1 | 1.3301 | 1.3301 | no |
| post-asinh transform 7/9 | 2,188 | 1 | 1.3301 | 1.3301 | no |
| post-asinh transform 8/9 | 2,188 | 1 | 1.3301 | 1.3301 | no |
| post-asinh transform 9/9 | 2,188 | 1 | 1.3301 | 1.3301 | no |
| allocation beta | 2,188 | 1 | 1.3301 | 1.3301 | no |
| knot refit | 2,188 | 1 | 1.32833 | 1.32833 | no |
| covariance shape 1/5 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| covariance shape 2/5 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| covariance shape 3/5 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| covariance shape 4/5 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| covariance shape 5/5 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| post-asinh transform 1/9 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| post-asinh transform 2/9 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| post-asinh transform 3/9 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| post-asinh transform 4/9 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| post-asinh transform 5/9 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| post-asinh transform 6/9 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| post-asinh transform 7/9 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| post-asinh transform 8/9 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| post-asinh transform 9/9 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| allocation beta | 2,188 | 1 | 1.32833 | 1.32833 | no |
| knot refit | 2,188 | 1 | 1.32833 | 1.32833 | no |
| refined parameter-search resolution | 2,188 | 1 | 1.32833 | 1.32833 | no |
| covariance shape 1/5 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| covariance shape 2/5 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| covariance shape 3/5 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| covariance shape 4/5 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| covariance shape 5/5 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| post-asinh transform 1/9 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| post-asinh transform 2/9 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| post-asinh transform 3/9 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| post-asinh transform 4/9 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| post-asinh transform 5/9 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| post-asinh transform 6/9 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| post-asinh transform 7/9 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| post-asinh transform 8/9 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| post-asinh transform 9/9 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| allocation beta | 2,188 | 1 | 1.32833 | 1.32833 | no |
| knot refit (cached stationary) | 2,188 | 1 | 1.32833 | 1.32833 | no |
| refined parameter-search resolution | 2,188 | 1 | 1.32833 | 1.32833 | no |
| covariance shape 1/5 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| covariance shape 2/5 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| covariance shape 3/5 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| covariance shape 4/5 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| covariance shape 5/5 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| post-asinh transform 1/9 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| post-asinh transform 2/9 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| post-asinh transform 3/9 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| post-asinh transform 4/9 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| post-asinh transform 5/9 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| post-asinh transform 6/9 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| post-asinh transform 7/9 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| post-asinh transform 8/9 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| post-asinh transform 9/9 | 2,188 | 1 | 1.32833 | 1.32833 | no |
| allocation beta | 2,188 | 1 | 1.32833 | 1.32833 | no |
| knot refit (cached stationary) | 2,188 | 1 | 1.32833 | 1.32833 | no |
| fixed point reached | 2,188 | 1 | 1.32833 | 1.32833 | no |
| rejected prune | 2,188 | 1 | 1.0632 | 1.0632 | no |
| prune proposal 2552->2370 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| covariance shape 1/5 | 2,370 | 1 | 0.999581 | 0.999581 | yes |
| covariance shape 2/5 | 2,370 | 1 | 0.999581 | 0.999581 | yes |
| covariance shape 3/5 | 2,370 | 1 | 0.999581 | 0.999581 | yes |
| covariance shape 4/5 | 2,370 | 1 | 0.999581 | 0.999581 | yes |
| covariance shape 5/5 | 2,370 | 1 | 0.999581 | 0.999581 | yes |
| post-asinh transform 1/9 | 2,370 | 1 | 0.999581 | 0.999581 | yes |
| post-asinh transform 2/9 | 2,370 | 1 | 0.999581 | 0.999581 | yes |
| post-asinh transform 3/9 | 2,370 | 1 | 0.999581 | 0.999581 | yes |
| post-asinh transform 4/9 | 2,370 | 1 | 0.999581 | 0.999581 | yes |
| post-asinh transform 5/9 | 2,370 | 1 | 0.999581 | 0.999581 | yes |
| post-asinh transform 6/9 | 2,370 | 1 | 0.999581 | 0.999581 | yes |
| post-asinh transform 7/9 | 2,370 | 1 | 0.999581 | 0.999581 | yes |
| post-asinh transform 8/9 | 2,370 | 1 | 0.999581 | 0.999581 | yes |
| post-asinh transform 9/9 | 2,370 | 1 | 0.999581 | 0.999581 | yes |
| allocation beta | 2,370 | 1 | 0.999581 | 0.999581 | yes |
| knot refit | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| covariance shape 1/5 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| covariance shape 2/5 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| covariance shape 3/5 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| covariance shape 4/5 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| covariance shape 5/5 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| post-asinh transform 1/9 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| post-asinh transform 2/9 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| post-asinh transform 3/9 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| post-asinh transform 4/9 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| post-asinh transform 5/9 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| post-asinh transform 6/9 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| post-asinh transform 7/9 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| post-asinh transform 8/9 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| post-asinh transform 9/9 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| allocation beta | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| knot refit | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| refined parameter-search resolution | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| covariance shape 1/5 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| covariance shape 2/5 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| covariance shape 3/5 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| covariance shape 4/5 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| covariance shape 5/5 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| post-asinh transform 1/9 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| post-asinh transform 2/9 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| post-asinh transform 3/9 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| post-asinh transform 4/9 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| post-asinh transform 5/9 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| post-asinh transform 6/9 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| post-asinh transform 7/9 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| post-asinh transform 8/9 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| post-asinh transform 9/9 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| allocation beta | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| knot refit (cached stationary) | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| refined parameter-search resolution | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| covariance shape 1/5 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| covariance shape 2/5 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| covariance shape 3/5 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| covariance shape 4/5 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| covariance shape 5/5 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| post-asinh transform 1/9 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| post-asinh transform 2/9 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| post-asinh transform 3/9 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| post-asinh transform 4/9 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| post-asinh transform 5/9 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| post-asinh transform 6/9 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| post-asinh transform 7/9 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| post-asinh transform 8/9 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| post-asinh transform 9/9 | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| allocation beta | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| knot refit (cached stationary) | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| fixed point reached | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| accepted prune | 2,370 | 1 | 0.998551 | 0.998551 | yes |
| cold restart 2370->2188 | 2,188 | 1 | 1.05888 | 1.05888 | no |
| covariance shape 1/5 | 2,188 | 1 | 1.0854 | 1.0854 | no |
| covariance shape 2/5 | 2,188 | 1 | 1.0854 | 1.0854 | no |
| covariance shape 3/5 | 2,188 | 1 | 1.0854 | 1.0854 | no |
| covariance shape 4/5 | 2,188 | 1 | 1.0854 | 1.0854 | no |
| covariance shape 5/5 | 2,188 | 1 | 1.0854 | 1.0854 | no |
| post-asinh transform 1/9 | 2,188 | 1 | 1.0854 | 1.0854 | no |
| post-asinh transform 2/9 | 2,188 | 1 | 1.0854 | 1.0854 | no |
| post-asinh transform 3/9 | 2,188 | 1 | 1.0854 | 1.0854 | no |
| post-asinh transform 4/9 | 2,188 | 1 | 1.0854 | 1.0854 | no |
| post-asinh transform 5/9 | 2,188 | 1 | 1.0854 | 1.0854 | no |
| post-asinh transform 6/9 | 2,188 | 1 | 1.0854 | 1.0854 | no |
| post-asinh transform 7/9 | 2,188 | 1 | 1.0854 | 1.0854 | no |
| post-asinh transform 8/9 | 2,188 | 1 | 1.0854 | 1.0854 | no |
| post-asinh transform 9/9 | 2,188 | 1 | 1.0854 | 1.0854 | no |
| allocation beta | 2,188 | 0.9 | 1.08387 | 1.08387 | no |
| knot refit | 2,188 | 0.9 | 1.08387 | 1.08387 | no |
| covariance shape 1/5 | 2,188 | 0.9 | 1.08387 | 1.08387 | no |
| covariance shape 2/5 | 2,188 | 0.9 | 1.08387 | 1.08387 | no |
| covariance shape 3/5 | 2,188 | 0.9 | 1.08387 | 1.08387 | no |
| covariance shape 4/5 | 2,188 | 0.9 | 1.08387 | 1.08387 | no |
| covariance shape 5/5 | 2,188 | 0.9 | 1.08387 | 1.08387 | no |
| post-asinh transform 1/9 | 2,188 | 0.9 | 1.08387 | 1.08387 | no |
| post-asinh transform 2/9 | 2,188 | 0.9 | 1.08387 | 1.08387 | no |
| post-asinh transform 3/9 | 2,188 | 0.9 | 1.08387 | 1.08387 | no |
| post-asinh transform 4/9 | 2,188 | 0.9 | 1.08387 | 1.08387 | no |
| post-asinh transform 5/9 | 2,188 | 0.9 | 1.08387 | 1.08387 | no |
| post-asinh transform 6/9 | 2,188 | 0.9 | 1.08387 | 1.08387 | no |
| post-asinh transform 7/9 | 2,188 | 0.9 | 1.08387 | 1.08387 | no |
| post-asinh transform 8/9 | 2,188 | 0.9 | 1.08387 | 1.08387 | no |
| post-asinh transform 9/9 | 2,188 | 0.9 | 1.08387 | 1.08387 | no |
| allocation beta | 2,188 | 1 | 1.07472 | 1.07472 | no |
| knot refit | 2,188 | 1 | 1.07181 | 1.07181 | no |
| covariance shape 1/5 | 2,188 | 1 | 1.07181 | 1.07181 | no |
| covariance shape 2/5 | 2,188 | 1 | 1.07181 | 1.07181 | no |
| covariance shape 3/5 | 2,188 | 1 | 1.07181 | 1.07181 | no |
| covariance shape 4/5 | 2,188 | 1 | 1.07181 | 1.07181 | no |
| covariance shape 5/5 | 2,188 | 1 | 1.07181 | 1.07181 | no |
| post-asinh transform 1/9 | 2,188 | 1 | 1.07181 | 1.07181 | no |
| post-asinh transform 2/9 | 2,188 | 1 | 1.07181 | 1.07181 | no |
| post-asinh transform 3/9 | 2,188 | 1 | 1.07181 | 1.07181 | no |
| post-asinh transform 4/9 | 2,188 | 1 | 1.07181 | 1.07181 | no |
| post-asinh transform 5/9 | 2,188 | 1 | 1.07181 | 1.07181 | no |
| post-asinh transform 6/9 | 2,188 | 1 | 1.07181 | 1.07181 | no |
| post-asinh transform 7/9 | 2,188 | 1 | 1.07181 | 1.07181 | no |
| post-asinh transform 8/9 | 2,188 | 1 | 1.07181 | 1.07181 | no |
| post-asinh transform 9/9 | 2,188 | 1 | 1.07181 | 1.07181 | no |
| allocation beta | 2,188 | 1 | 1.07181 | 1.07181 | no |
| knot refit | 2,188 | 1 | 1.06962 | 1.06962 | no |
| covariance shape 1/5 | 2,188 | 1 | 1.06962 | 1.06962 | no |
| covariance shape 2/5 | 2,188 | 1 | 1.06962 | 1.06962 | no |
| covariance shape 3/5 | 2,188 | 1 | 1.06962 | 1.06962 | no |
| covariance shape 4/5 | 2,188 | 1 | 1.06962 | 1.06962 | no |
| covariance shape 5/5 | 2,188 | 1 | 1.06962 | 1.06962 | no |
| post-asinh transform 1/9 | 2,188 | 1 | 1.06962 | 1.06962 | no |
| post-asinh transform 2/9 | 2,188 | 1 | 1.06962 | 1.06962 | no |
| post-asinh transform 3/9 | 2,188 | 1 | 1.06962 | 1.06962 | no |
| post-asinh transform 4/9 | 2,188 | 1 | 1.06962 | 1.06962 | no |
| post-asinh transform 5/9 | 2,188 | 1 | 1.06962 | 1.06962 | no |
| post-asinh transform 6/9 | 2,188 | 1 | 1.06962 | 1.06962 | no |
| post-asinh transform 7/9 | 2,188 | 1 | 1.06962 | 1.06962 | no |
| post-asinh transform 8/9 | 2,188 | 1 | 1.06962 | 1.06962 | no |
| post-asinh transform 9/9 | 2,188 | 1 | 1.06962 | 1.06962 | no |
| allocation beta | 2,188 | 1 | 1.06962 | 1.06962 | no |
| knot refit | 2,188 | 1 | 1.06962 | 1.06962 | no |
| refined parameter-search resolution | 2,188 | 1 | 1.06962 | 1.06962 | no |
| covariance shape 1/5 | 2,188 | 1 | 1.06962 | 1.06962 | no |
| covariance shape 2/5 | 2,188 | 1 | 1.06962 | 1.06962 | no |
| covariance shape 3/5 | 2,188 | 1 | 1.06962 | 1.06962 | no |
| covariance shape 4/5 | 2,188 | 1 | 1.06962 | 1.06962 | no |
| covariance shape 5/5 | 2,188 | 1 | 1.06962 | 1.06962 | no |
| post-asinh transform 1/9 | 2,188 | 1 | 1.06962 | 1.06962 | no |
| post-asinh transform 2/9 | 2,188 | 1 | 1.06962 | 1.06962 | no |
| post-asinh transform 3/9 | 2,188 | 1 | 1.06962 | 1.06962 | no |
| post-asinh transform 4/9 | 2,188 | 1 | 1.06962 | 1.06962 | no |
| post-asinh transform 5/9 | 2,188 | 1 | 1.06962 | 1.06962 | no |
| post-asinh transform 6/9 | 2,188 | 1 | 1.06962 | 1.06962 | no |
| post-asinh transform 7/9 | 2,188 | 1 | 1.06962 | 1.06962 | no |
| post-asinh transform 8/9 | 2,188 | 1 | 1.06962 | 1.06962 | no |
| post-asinh transform 9/9 | 2,188 | 1 | 1.06962 | 1.06962 | no |
| allocation beta | 2,188 | 1 | 1.06962 | 1.06962 | no |
| knot refit (cached stationary) | 2,188 | 1 | 1.06962 | 1.06962 | no |
| refined parameter-search resolution | 2,188 | 1 | 1.06962 | 1.06962 | no |
| covariance shape 1/5 | 2,188 | 1 | 1.06703 | 1.06703 | no |
| covariance shape 2/5 | 2,188 | 1 | 1.06703 | 1.06703 | no |
| covariance shape 3/5 | 2,188 | 1 | 1.06703 | 1.06703 | no |
| covariance shape 4/5 | 2,188 | 1 | 1.06703 | 1.06703 | no |
| covariance shape 5/5 | 2,188 | 1 | 1.06703 | 1.06703 | no |
| post-asinh transform 1/9 | 2,188 | 1 | 1.06703 | 1.06703 | no |
| post-asinh transform 2/9 | 2,188 | 1 | 1.06703 | 1.06703 | no |
| post-asinh transform 3/9 | 2,188 | 1 | 1.06703 | 1.06703 | no |
| post-asinh transform 4/9 | 2,188 | 1 | 1.06703 | 1.06703 | no |
| post-asinh transform 5/9 | 2,188 | 1 | 1.06703 | 1.06703 | no |
| post-asinh transform 6/9 | 2,188 | 1 | 1.06703 | 1.06703 | no |
| post-asinh transform 7/9 | 2,188 | 1 | 1.06703 | 1.06703 | no |
| post-asinh transform 8/9 | 2,188 | 1 | 1.06703 | 1.06703 | no |
| post-asinh transform 9/9 | 2,188 | 1 | 1.06703 | 1.06703 | no |
| allocation beta | 2,188 | 0.975 | 1.0628 | 1.0628 | no |
| knot refit | 2,188 | 0.975 | 1.06173 | 1.06173 | no |
| covariance shape 1/5 | 2,188 | 0.975 | 1.06173 | 1.06173 | no |
| covariance shape 2/5 | 2,188 | 0.975 | 1.06173 | 1.06173 | no |
| covariance shape 3/5 | 2,188 | 0.975 | 1.06173 | 1.06173 | no |
| covariance shape 4/5 | 2,188 | 0.975 | 1.06173 | 1.06173 | no |
| covariance shape 5/5 | 2,188 | 0.975 | 1.06173 | 1.06173 | no |
| post-asinh transform 1/9 | 2,188 | 0.975 | 1.06173 | 1.06173 | no |
| post-asinh transform 2/9 | 2,188 | 0.975 | 1.06173 | 1.06173 | no |
| post-asinh transform 3/9 | 2,188 | 0.975 | 1.06173 | 1.06173 | no |
| post-asinh transform 4/9 | 2,188 | 0.975 | 1.06173 | 1.06173 | no |
| post-asinh transform 5/9 | 2,188 | 0.975 | 1.06173 | 1.06173 | no |
| post-asinh transform 6/9 | 2,188 | 0.975 | 1.06173 | 1.06173 | no |
| post-asinh transform 7/9 | 2,188 | 0.975 | 1.06173 | 1.06173 | no |
| post-asinh transform 8/9 | 2,188 | 0.975 | 1.06173 | 1.06173 | no |
| post-asinh transform 9/9 | 2,188 | 0.975 | 1.06173 | 1.06173 | no |
| allocation beta | 2,188 | 1 | 1.05888 | 1.05888 | no |
| knot refit | 2,188 | 1 | 1.05888 | 1.05888 | no |
| covariance shape 1/5 | 2,188 | 1 | 1.05888 | 1.05888 | no |
| covariance shape 2/5 | 2,188 | 1 | 1.05888 | 1.05888 | no |
| covariance shape 3/5 | 2,188 | 1 | 1.05888 | 1.05888 | no |
| covariance shape 4/5 | 2,188 | 1 | 1.05888 | 1.05888 | no |
| covariance shape 5/5 | 2,188 | 1 | 1.05888 | 1.05888 | no |
| post-asinh transform 1/9 | 2,188 | 1 | 1.05888 | 1.05888 | no |
| post-asinh transform 2/9 | 2,188 | 1 | 1.05888 | 1.05888 | no |
| post-asinh transform 3/9 | 2,188 | 1 | 1.05888 | 1.05888 | no |
| post-asinh transform 4/9 | 2,188 | 1 | 1.05888 | 1.05888 | no |
| post-asinh transform 5/9 | 2,188 | 1 | 1.05888 | 1.05888 | no |
| post-asinh transform 6/9 | 2,188 | 1 | 1.05888 | 1.05888 | no |
| post-asinh transform 7/9 | 2,188 | 1 | 1.05888 | 1.05888 | no |
| post-asinh transform 8/9 | 2,188 | 1 | 1.05888 | 1.05888 | no |
| post-asinh transform 9/9 | 2,188 | 1 | 1.05888 | 1.05888 | no |
| allocation beta | 2,188 | 1 | 1.05888 | 1.05888 | no |
| knot refit (cached stationary) | 2,188 | 1 | 1.05888 | 1.05888 | no |
| fixed point reached | 2,188 | 1 | 1.05888 | 1.05888 | no |
| prune proposal 2370->2188 | 2,188 | 1 | 1.05888 | 1.05888 | no |
| covariance shape 1/5 | 2,188 | 1 | 1.38391 | 1.38391 | no |
| covariance shape 2/5 | 2,188 | 1 | 1.38391 | 1.38391 | no |
| covariance shape 3/5 | 2,188 | 1 | 1.38391 | 1.38391 | no |
| covariance shape 4/5 | 2,188 | 1 | 1.38391 | 1.38391 | no |
| covariance shape 5/5 | 2,188 | 1 | 1.38391 | 1.38391 | no |
| post-asinh transform 1/9 | 2,188 | 1 | 1.38391 | 1.38391 | no |
| post-asinh transform 2/9 | 2,188 | 1 | 1.38391 | 1.38391 | no |
| post-asinh transform 3/9 | 2,188 | 1 | 1.38391 | 1.38391 | no |
| post-asinh transform 4/9 | 2,188 | 1 | 1.38391 | 1.38391 | no |
| post-asinh transform 5/9 | 2,188 | 1 | 1.38391 | 1.38391 | no |
| post-asinh transform 6/9 | 2,188 | 1 | 1.38391 | 1.38391 | no |
| post-asinh transform 7/9 | 2,188 | 1 | 1.38391 | 1.38391 | no |
| post-asinh transform 8/9 | 2,188 | 1 | 1.38391 | 1.38391 | no |
| post-asinh transform 9/9 | 2,188 | 1 | 1.38391 | 1.38391 | no |
| allocation beta | 2,188 | 0.9 | 1.35891 | 1.35891 | no |
| knot refit | 2,188 | 0.9 | 1.3522 | 1.3522 | no |
| covariance shape 1/5 | 2,188 | 0.9 | 1.3522 | 1.3522 | no |
| covariance shape 2/5 | 2,188 | 0.9 | 1.3522 | 1.3522 | no |
| covariance shape 3/5 | 2,188 | 0.9 | 1.3522 | 1.3522 | no |
| covariance shape 4/5 | 2,188 | 0.9 | 1.3522 | 1.3522 | no |
| covariance shape 5/5 | 2,188 | 0.9 | 1.3522 | 1.3522 | no |
| post-asinh transform 1/9 | 2,188 | 0.9 | 1.3522 | 1.3522 | no |
| post-asinh transform 2/9 | 2,188 | 0.9 | 1.3522 | 1.3522 | no |
| post-asinh transform 3/9 | 2,188 | 0.9 | 1.3522 | 1.3522 | no |
| post-asinh transform 4/9 | 2,188 | 0.9 | 1.3522 | 1.3522 | no |
| post-asinh transform 5/9 | 2,188 | 0.9 | 1.3522 | 1.3522 | no |
| post-asinh transform 6/9 | 2,188 | 0.9 | 1.3522 | 1.3522 | no |
| post-asinh transform 7/9 | 2,188 | 0.9 | 1.3522 | 1.3522 | no |
| post-asinh transform 8/9 | 2,188 | 0.9 | 1.3522 | 1.3522 | no |
| post-asinh transform 9/9 | 2,188 | 0.9 | 1.3522 | 1.3522 | no |
| allocation beta | 2,188 | 1 | 1.33852 | 1.33852 | no |
| knot refit | 2,188 | 1 | 1.33577 | 1.33577 | no |
| covariance shape 1/5 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| covariance shape 2/5 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| covariance shape 3/5 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| covariance shape 4/5 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| covariance shape 5/5 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| post-asinh transform 1/9 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| post-asinh transform 2/9 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| post-asinh transform 3/9 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| post-asinh transform 4/9 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| post-asinh transform 5/9 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| post-asinh transform 6/9 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| post-asinh transform 7/9 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| post-asinh transform 8/9 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| post-asinh transform 9/9 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| allocation beta | 2,188 | 1 | 1.33577 | 1.33577 | no |
| knot refit | 2,188 | 1 | 1.33577 | 1.33577 | no |
| refined parameter-search resolution | 2,188 | 1 | 1.33577 | 1.33577 | no |
| covariance shape 1/5 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| covariance shape 2/5 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| covariance shape 3/5 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| covariance shape 4/5 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| covariance shape 5/5 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| post-asinh transform 1/9 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| post-asinh transform 2/9 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| post-asinh transform 3/9 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| post-asinh transform 4/9 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| post-asinh transform 5/9 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| post-asinh transform 6/9 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| post-asinh transform 7/9 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| post-asinh transform 8/9 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| post-asinh transform 9/9 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| allocation beta | 2,188 | 1 | 1.33577 | 1.33577 | no |
| knot refit (cached stationary) | 2,188 | 1 | 1.33577 | 1.33577 | no |
| refined parameter-search resolution | 2,188 | 1 | 1.33577 | 1.33577 | no |
| covariance shape 1/5 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| covariance shape 2/5 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| covariance shape 3/5 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| covariance shape 4/5 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| covariance shape 5/5 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| post-asinh transform 1/9 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| post-asinh transform 2/9 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| post-asinh transform 3/9 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| post-asinh transform 4/9 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| post-asinh transform 5/9 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| post-asinh transform 6/9 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| post-asinh transform 7/9 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| post-asinh transform 8/9 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| post-asinh transform 9/9 | 2,188 | 1 | 1.33577 | 1.33577 | no |
| allocation beta | 2,188 | 1 | 1.33577 | 1.33577 | no |
| knot refit (cached stationary) | 2,188 | 1 | 1.33577 | 1.33577 | no |
| fixed point reached | 2,188 | 1 | 1.33577 | 1.33577 | no |
| rejected prune | 2,188 | 1 | 1.05888 | 1.05888 | no |
| prune proposal 2370->2279 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| covariance shape 1/5 | 2,279 | 1 | 1.01122 | 1.01122 | no |
| covariance shape 2/5 | 2,279 | 1 | 1.01122 | 1.01122 | no |
| covariance shape 3/5 | 2,279 | 1 | 1.01122 | 1.01122 | no |
| covariance shape 4/5 | 2,279 | 1 | 1.01122 | 1.01122 | no |
| covariance shape 5/5 | 2,279 | 1 | 1.01122 | 1.01122 | no |
| post-asinh transform 1/9 | 2,279 | 1 | 1.01122 | 1.01122 | no |
| post-asinh transform 2/9 | 2,279 | 1 | 1.01122 | 1.01122 | no |
| post-asinh transform 3/9 | 2,279 | 1 | 1.01122 | 1.01122 | no |
| post-asinh transform 4/9 | 2,279 | 1 | 1.01122 | 1.01122 | no |
| post-asinh transform 5/9 | 2,279 | 1 | 1.01122 | 1.01122 | no |
| post-asinh transform 6/9 | 2,279 | 1 | 1.01122 | 1.01122 | no |
| post-asinh transform 7/9 | 2,279 | 1 | 1.01122 | 1.01122 | no |
| post-asinh transform 8/9 | 2,279 | 1 | 1.01122 | 1.01122 | no |
| post-asinh transform 9/9 | 2,279 | 1 | 1.01122 | 1.01122 | no |
| allocation beta | 2,279 | 0.9 | 1.00628 | 1.00628 | no |
| knot refit | 2,279 | 0.9 | 1.00262 | 1.00262 | no |
| covariance shape 1/5 | 2,279 | 0.9 | 1.00262 | 1.00262 | no |
| covariance shape 2/5 | 2,279 | 0.9 | 1.00262 | 1.00262 | no |
| covariance shape 3/5 | 2,279 | 0.9 | 1.00262 | 1.00262 | no |
| covariance shape 4/5 | 2,279 | 0.9 | 1.00262 | 1.00262 | no |
| covariance shape 5/5 | 2,279 | 0.9 | 1.00262 | 1.00262 | no |
| post-asinh transform 1/9 | 2,279 | 0.9 | 1.00262 | 1.00262 | no |
| post-asinh transform 2/9 | 2,279 | 0.9 | 1.00262 | 1.00262 | no |
| post-asinh transform 3/9 | 2,279 | 0.9 | 1.00262 | 1.00262 | no |
| post-asinh transform 4/9 | 2,279 | 0.9 | 1.00262 | 1.00262 | no |
| post-asinh transform 5/9 | 2,279 | 0.9 | 1.00262 | 1.00262 | no |
| post-asinh transform 6/9 | 2,279 | 0.9 | 1.00262 | 1.00262 | no |
| post-asinh transform 7/9 | 2,279 | 0.9 | 1.00262 | 1.00262 | no |
| post-asinh transform 8/9 | 2,279 | 0.9 | 1.00262 | 1.00262 | no |
| post-asinh transform 9/9 | 2,279 | 0.9 | 1.00262 | 1.00262 | no |
| allocation beta | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| knot refit | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| covariance shape 1/5 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| covariance shape 2/5 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| covariance shape 3/5 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| covariance shape 4/5 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| covariance shape 5/5 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| post-asinh transform 1/9 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| post-asinh transform 2/9 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| post-asinh transform 3/9 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| post-asinh transform 4/9 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| post-asinh transform 5/9 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| post-asinh transform 6/9 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| post-asinh transform 7/9 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| post-asinh transform 8/9 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| post-asinh transform 9/9 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| allocation beta | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| knot refit (cached stationary) | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| refined parameter-search resolution | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| covariance shape 1/5 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| covariance shape 2/5 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| covariance shape 3/5 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| covariance shape 4/5 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| covariance shape 5/5 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| post-asinh transform 1/9 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| post-asinh transform 2/9 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| post-asinh transform 3/9 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| post-asinh transform 4/9 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| post-asinh transform 5/9 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| post-asinh transform 6/9 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| post-asinh transform 7/9 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| post-asinh transform 8/9 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| post-asinh transform 9/9 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| allocation beta | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| knot refit (cached stationary) | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| refined parameter-search resolution | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| covariance shape 1/5 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| covariance shape 2/5 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| covariance shape 3/5 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| covariance shape 4/5 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| covariance shape 5/5 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| post-asinh transform 1/9 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| post-asinh transform 2/9 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| post-asinh transform 3/9 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| post-asinh transform 4/9 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| post-asinh transform 5/9 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| post-asinh transform 6/9 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| post-asinh transform 7/9 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| post-asinh transform 8/9 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| post-asinh transform 9/9 | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| allocation beta | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| knot refit (cached stationary) | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| fixed point reached | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| accepted prune | 2,279 | 1 | 0.997003 | 0.997003 | yes |
| cold restart 2279->2188 | 2,188 | 1 | 1.04409 | 1.04409 | no |
| covariance shape 1/5 | 2,188 | 1 | 1.07851 | 1.07851 | no |
| covariance shape 2/5 | 2,188 | 1 | 1.07851 | 1.07851 | no |
| covariance shape 3/5 | 2,188 | 1 | 1.07851 | 1.07851 | no |
| covariance shape 4/5 | 2,188 | 1 | 1.07851 | 1.07851 | no |
| covariance shape 5/5 | 2,188 | 1 | 1.07851 | 1.07851 | no |
| post-asinh transform 1/9 | 2,188 | 1 | 1.07851 | 1.07851 | no |
| post-asinh transform 2/9 | 2,188 | 1 | 1.07851 | 1.07851 | no |
| post-asinh transform 3/9 | 2,188 | 1 | 1.07851 | 1.07851 | no |
| post-asinh transform 4/9 | 2,188 | 1 | 1.07851 | 1.07851 | no |
| post-asinh transform 5/9 | 2,188 | 1 | 1.07851 | 1.07851 | no |
| post-asinh transform 6/9 | 2,188 | 1 | 1.07851 | 1.07851 | no |
| post-asinh transform 7/9 | 2,188 | 1 | 1.07851 | 1.07851 | no |
| post-asinh transform 8/9 | 2,188 | 1 | 1.07851 | 1.07851 | no |
| post-asinh transform 9/9 | 2,188 | 1 | 1.07851 | 1.07851 | no |
| allocation beta | 2,188 | 1 | 1.07851 | 1.07851 | no |
| knot refit | 2,188 | 1 | 1.07277 | 1.07277 | no |
| covariance shape 1/5 | 2,188 | 1 | 1.07277 | 1.07277 | no |
| covariance shape 2/5 | 2,188 | 1 | 1.07277 | 1.07277 | no |
| covariance shape 3/5 | 2,188 | 1 | 1.07277 | 1.07277 | no |
| covariance shape 4/5 | 2,188 | 1 | 1.07277 | 1.07277 | no |
| covariance shape 5/5 | 2,188 | 1 | 1.07277 | 1.07277 | no |
| post-asinh transform 1/9 | 2,188 | 1 | 1.07277 | 1.07277 | no |
| post-asinh transform 2/9 | 2,188 | 1 | 1.07277 | 1.07277 | no |
| post-asinh transform 3/9 | 2,188 | 1 | 1.07277 | 1.07277 | no |
| post-asinh transform 4/9 | 2,188 | 1 | 1.07277 | 1.07277 | no |
| post-asinh transform 5/9 | 2,188 | 1 | 1.07277 | 1.07277 | no |
| post-asinh transform 6/9 | 2,188 | 1 | 1.07277 | 1.07277 | no |
| post-asinh transform 7/9 | 2,188 | 1 | 1.07277 | 1.07277 | no |
| post-asinh transform 8/9 | 2,188 | 1 | 1.07277 | 1.07277 | no |
| post-asinh transform 9/9 | 2,188 | 1 | 1.07277 | 1.07277 | no |
| allocation beta | 2,188 | 1 | 1.07277 | 1.07277 | no |
| knot refit | 2,188 | 1 | 1.06944 | 1.06944 | no |
| covariance shape 1/5 | 2,188 | 1 | 1.06944 | 1.06944 | no |
| covariance shape 2/5 | 2,188 | 1 | 1.06944 | 1.06944 | no |
| covariance shape 3/5 | 2,188 | 1 | 1.06944 | 1.06944 | no |
| covariance shape 4/5 | 2,188 | 1 | 1.06944 | 1.06944 | no |
| covariance shape 5/5 | 2,188 | 1 | 1.06944 | 1.06944 | no |
| post-asinh transform 1/9 | 2,188 | 1 | 1.06944 | 1.06944 | no |
| post-asinh transform 2/9 | 2,188 | 1 | 1.06944 | 1.06944 | no |
| post-asinh transform 3/9 | 2,188 | 1 | 1.06944 | 1.06944 | no |
| post-asinh transform 4/9 | 2,188 | 1 | 1.06944 | 1.06944 | no |
| post-asinh transform 5/9 | 2,188 | 1 | 1.06944 | 1.06944 | no |
| post-asinh transform 6/9 | 2,188 | 1 | 1.06944 | 1.06944 | no |
| post-asinh transform 7/9 | 2,188 | 1 | 1.06944 | 1.06944 | no |
| post-asinh transform 8/9 | 2,188 | 1 | 1.06944 | 1.06944 | no |
| post-asinh transform 9/9 | 2,188 | 1 | 1.06944 | 1.06944 | no |
| allocation beta | 2,188 | 1 | 1.06944 | 1.06944 | no |
| knot refit | 2,188 | 1 | 1.0675 | 1.0675 | no |
| covariance shape 1/5 | 2,188 | 1 | 1.0675 | 1.0675 | no |
| covariance shape 2/5 | 2,188 | 1 | 1.0675 | 1.0675 | no |
| covariance shape 3/5 | 2,188 | 1 | 1.0675 | 1.0675 | no |
| covariance shape 4/5 | 2,188 | 1 | 1.0675 | 1.0675 | no |
| covariance shape 5/5 | 2,188 | 1 | 1.0675 | 1.0675 | no |
| post-asinh transform 1/9 | 2,188 | 1 | 1.0675 | 1.0675 | no |
| post-asinh transform 2/9 | 2,188 | 1 | 1.0675 | 1.0675 | no |
| post-asinh transform 3/9 | 2,188 | 1 | 1.0675 | 1.0675 | no |
| post-asinh transform 4/9 | 2,188 | 1 | 1.0675 | 1.0675 | no |
| post-asinh transform 5/9 | 2,188 | 1 | 1.0675 | 1.0675 | no |
| post-asinh transform 6/9 | 2,188 | 1 | 1.0675 | 1.0675 | no |
| post-asinh transform 7/9 | 2,188 | 1 | 1.0675 | 1.0675 | no |
| post-asinh transform 8/9 | 2,188 | 1 | 1.0675 | 1.0675 | no |
| post-asinh transform 9/9 | 2,188 | 1 | 1.0675 | 1.0675 | no |
| allocation beta | 2,188 | 1 | 1.0675 | 1.0675 | no |
| knot refit | 2,188 | 1 | 1.06421 | 1.06421 | no |
| covariance shape 1/5 | 2,188 | 1 | 1.06421 | 1.06421 | no |
| covariance shape 2/5 | 2,188 | 1 | 1.06421 | 1.06421 | no |
| covariance shape 3/5 | 2,188 | 1 | 1.06421 | 1.06421 | no |
| covariance shape 4/5 | 2,188 | 1 | 1.06421 | 1.06421 | no |
| covariance shape 5/5 | 2,188 | 1 | 1.06421 | 1.06421 | no |
| post-asinh transform 1/9 | 2,188 | 1 | 1.06421 | 1.06421 | no |
| post-asinh transform 2/9 | 2,188 | 1 | 1.06421 | 1.06421 | no |
| post-asinh transform 3/9 | 2,188 | 1 | 1.06421 | 1.06421 | no |
| post-asinh transform 4/9 | 2,188 | 1 | 1.06421 | 1.06421 | no |
| post-asinh transform 5/9 | 2,188 | 1 | 1.06421 | 1.06421 | no |
| post-asinh transform 6/9 | 2,188 | 1 | 1.06421 | 1.06421 | no |
| post-asinh transform 7/9 | 2,188 | 1 | 1.06421 | 1.06421 | no |
| post-asinh transform 8/9 | 2,188 | 1 | 1.06421 | 1.06421 | no |
| post-asinh transform 9/9 | 2,188 | 1 | 1.06421 | 1.06421 | no |
| allocation beta | 2,188 | 1 | 1.06421 | 1.06421 | no |
| knot refit | 2,188 | 1 | 1.06258 | 1.06258 | no |
| covariance shape 1/5 | 2,188 | 1 | 1.06258 | 1.06258 | no |
| covariance shape 2/5 | 2,188 | 1 | 1.06258 | 1.06258 | no |
| covariance shape 3/5 | 2,188 | 1 | 1.06258 | 1.06258 | no |
| covariance shape 4/5 | 2,188 | 1 | 1.06258 | 1.06258 | no |
| covariance shape 5/5 | 2,188 | 1 | 1.06258 | 1.06258 | no |
| post-asinh transform 1/9 | 2,188 | 1 | 1.06258 | 1.06258 | no |
| post-asinh transform 2/9 | 2,188 | 1 | 1.06258 | 1.06258 | no |
| post-asinh transform 3/9 | 2,188 | 1 | 1.06258 | 1.06258 | no |
| post-asinh transform 4/9 | 2,188 | 1 | 1.06258 | 1.06258 | no |
| post-asinh transform 5/9 | 2,188 | 1 | 1.06258 | 1.06258 | no |
| post-asinh transform 6/9 | 2,188 | 1 | 1.06258 | 1.06258 | no |
| post-asinh transform 7/9 | 2,188 | 1 | 1.06258 | 1.06258 | no |
| post-asinh transform 8/9 | 2,188 | 1 | 1.06258 | 1.06258 | no |
| post-asinh transform 9/9 | 2,188 | 1 | 1.06258 | 1.06258 | no |
| allocation beta | 2,188 | 1 | 1.06258 | 1.06258 | no |
| knot refit | 2,188 | 1 | 1.06258 | 1.06258 | no |
| refined parameter-search resolution | 2,188 | 1 | 1.06258 | 1.06258 | no |
| covariance shape 1/5 | 2,188 | 1 | 1.06258 | 1.06258 | no |
| covariance shape 2/5 | 2,188 | 1 | 1.06258 | 1.06258 | no |
| covariance shape 3/5 | 2,188 | 1 | 1.06258 | 1.06258 | no |
| covariance shape 4/5 | 2,188 | 1 | 1.06258 | 1.06258 | no |
| covariance shape 5/5 | 2,188 | 1 | 1.06258 | 1.06258 | no |
| post-asinh transform 1/9 | 2,188 | 1 | 1.06258 | 1.06258 | no |
| post-asinh transform 2/9 | 2,188 | 1 | 1.05963 | 1.05963 | no |
| post-asinh transform 3/9 | 2,188 | 1 | 1.05963 | 1.05963 | no |
| post-asinh transform 4/9 | 2,188 | 1 | 1.05963 | 1.05963 | no |
| post-asinh transform 5/9 | 2,188 | 1 | 1.05963 | 1.05963 | no |
| post-asinh transform 6/9 | 2,188 | 1 | 1.05963 | 1.05963 | no |
| post-asinh transform 7/9 | 2,188 | 1 | 1.05963 | 1.05963 | no |
| post-asinh transform 8/9 | 2,188 | 1 | 1.05963 | 1.05963 | no |
| post-asinh transform 9/9 | 2,188 | 1 | 1.05963 | 1.05963 | no |
| allocation beta | 2,188 | 0.95 | 1.05146 | 1.05146 | no |
| knot refit | 2,188 | 0.95 | 1.04743 | 1.04743 | no |
| covariance shape 1/5 | 2,188 | 0.95 | 1.04743 | 1.04743 | no |
| covariance shape 2/5 | 2,188 | 0.95 | 1.04743 | 1.04743 | no |
| covariance shape 3/5 | 2,188 | 0.95 | 1.04743 | 1.04743 | no |
| covariance shape 4/5 | 2,188 | 0.95 | 1.04743 | 1.04743 | no |
| covariance shape 5/5 | 2,188 | 0.95 | 1.04743 | 1.04743 | no |
| post-asinh transform 1/9 | 2,188 | 0.95 | 1.04743 | 1.04743 | no |
| post-asinh transform 2/9 | 2,188 | 0.95 | 1.04743 | 1.04743 | no |
| post-asinh transform 3/9 | 2,188 | 0.95 | 1.04743 | 1.04743 | no |
| post-asinh transform 4/9 | 2,188 | 0.95 | 1.04743 | 1.04743 | no |
| post-asinh transform 5/9 | 2,188 | 0.95 | 1.04743 | 1.04743 | no |
| post-asinh transform 6/9 | 2,188 | 0.95 | 1.04743 | 1.04743 | no |
| post-asinh transform 7/9 | 2,188 | 0.95 | 1.04743 | 1.04743 | no |
| post-asinh transform 8/9 | 2,188 | 0.95 | 1.04743 | 1.04743 | no |
| post-asinh transform 9/9 | 2,188 | 0.95 | 1.04743 | 1.04743 | no |
| allocation beta | 2,188 | 1 | 1.04409 | 1.04409 | no |
| knot refit | 2,188 | 1 | 1.04409 | 1.04409 | no |
| covariance shape 1/5 | 2,188 | 1 | 1.04409 | 1.04409 | no |
| covariance shape 2/5 | 2,188 | 1 | 1.04409 | 1.04409 | no |
| covariance shape 3/5 | 2,188 | 1 | 1.04409 | 1.04409 | no |
| covariance shape 4/5 | 2,188 | 1 | 1.04409 | 1.04409 | no |
| covariance shape 5/5 | 2,188 | 1 | 1.04409 | 1.04409 | no |
| post-asinh transform 1/9 | 2,188 | 1 | 1.04409 | 1.04409 | no |
| post-asinh transform 2/9 | 2,188 | 1 | 1.04409 | 1.04409 | no |
| post-asinh transform 3/9 | 2,188 | 1 | 1.04409 | 1.04409 | no |
| post-asinh transform 4/9 | 2,188 | 1 | 1.04409 | 1.04409 | no |
| post-asinh transform 5/9 | 2,188 | 1 | 1.04409 | 1.04409 | no |
| post-asinh transform 6/9 | 2,188 | 1 | 1.04409 | 1.04409 | no |
| post-asinh transform 7/9 | 2,188 | 1 | 1.04409 | 1.04409 | no |
| post-asinh transform 8/9 | 2,188 | 1 | 1.04409 | 1.04409 | no |
| post-asinh transform 9/9 | 2,188 | 1 | 1.04409 | 1.04409 | no |
| allocation beta | 2,188 | 1 | 1.04409 | 1.04409 | no |
| knot refit (cached stationary) | 2,188 | 1 | 1.04409 | 1.04409 | no |
| refined parameter-search resolution | 2,188 | 1 | 1.04409 | 1.04409 | no |
| covariance shape 1/5 | 2,188 | 1 | 1.04409 | 1.04409 | no |
| covariance shape 2/5 | 2,188 | 1 | 1.04409 | 1.04409 | no |
| covariance shape 3/5 | 2,188 | 1 | 1.04409 | 1.04409 | no |
| covariance shape 4/5 | 2,188 | 1 | 1.04409 | 1.04409 | no |
| covariance shape 5/5 | 2,188 | 1 | 1.04409 | 1.04409 | no |
| post-asinh transform 1/9 | 2,188 | 1 | 1.04409 | 1.04409 | no |
| post-asinh transform 2/9 | 2,188 | 1 | 1.04409 | 1.04409 | no |
| post-asinh transform 3/9 | 2,188 | 1 | 1.04409 | 1.04409 | no |
| post-asinh transform 4/9 | 2,188 | 1 | 1.04409 | 1.04409 | no |
| post-asinh transform 5/9 | 2,188 | 1 | 1.04409 | 1.04409 | no |
| post-asinh transform 6/9 | 2,188 | 1 | 1.04409 | 1.04409 | no |
| post-asinh transform 7/9 | 2,188 | 1 | 1.04409 | 1.04409 | no |
| post-asinh transform 8/9 | 2,188 | 1 | 1.04409 | 1.04409 | no |
| post-asinh transform 9/9 | 2,188 | 1 | 1.04409 | 1.04409 | no |
| allocation beta | 2,188 | 1 | 1.04409 | 1.04409 | no |
| knot refit (cached stationary) | 2,188 | 1 | 1.04409 | 1.04409 | no |
| fixed point reached | 2,188 | 1 | 1.04409 | 1.04409 | no |
| prune proposal 2279->2188 | 2,188 | 1 | 1.04409 | 1.04409 | no |
| covariance shape 1/5 | 2,188 | 1 | 1.37772 | 1.37772 | no |
| covariance shape 2/5 | 2,188 | 1 | 1.37772 | 1.37772 | no |
| covariance shape 3/5 | 2,188 | 1 | 1.37772 | 1.37772 | no |
| covariance shape 4/5 | 2,188 | 1 | 1.37772 | 1.37772 | no |
| covariance shape 5/5 | 2,188 | 1 | 1.37772 | 1.37772 | no |
| post-asinh transform 1/9 | 2,188 | 1 | 1.37772 | 1.37772 | no |
| post-asinh transform 2/9 | 2,188 | 1 | 1.37772 | 1.37772 | no |
| post-asinh transform 3/9 | 2,188 | 1 | 1.37772 | 1.37772 | no |
| post-asinh transform 4/9 | 2,188 | 1 | 1.37772 | 1.37772 | no |
| post-asinh transform 5/9 | 2,188 | 1 | 1.37772 | 1.37772 | no |
| post-asinh transform 6/9 | 2,188 | 1 | 1.37772 | 1.37772 | no |
| post-asinh transform 7/9 | 2,188 | 1 | 1.37772 | 1.37772 | no |
| post-asinh transform 8/9 | 2,188 | 1 | 1.37772 | 1.37772 | no |
| post-asinh transform 9/9 | 2,188 | 1 | 1.37772 | 1.37772 | no |
| allocation beta | 2,188 | 0.9 | 1.35963 | 1.35963 | no |
| knot refit | 2,188 | 0.9 | 1.35212 | 1.35212 | no |
| covariance shape 1/5 | 2,188 | 0.9 | 1.35212 | 1.35212 | no |
| covariance shape 2/5 | 2,188 | 0.9 | 1.35212 | 1.35212 | no |
| covariance shape 3/5 | 2,188 | 0.9 | 1.35212 | 1.35212 | no |
| covariance shape 4/5 | 2,188 | 0.9 | 1.35212 | 1.35212 | no |
| covariance shape 5/5 | 2,188 | 0.9 | 1.35212 | 1.35212 | no |
| post-asinh transform 1/9 | 2,188 | 0.9 | 1.35212 | 1.35212 | no |
| post-asinh transform 2/9 | 2,188 | 0.9 | 1.35212 | 1.35212 | no |
| post-asinh transform 3/9 | 2,188 | 0.9 | 1.35212 | 1.35212 | no |
| post-asinh transform 4/9 | 2,188 | 0.9 | 1.35212 | 1.35212 | no |
| post-asinh transform 5/9 | 2,188 | 0.9 | 1.35212 | 1.35212 | no |
| post-asinh transform 6/9 | 2,188 | 0.9 | 1.35212 | 1.35212 | no |
| post-asinh transform 7/9 | 2,188 | 0.9 | 1.35212 | 1.35212 | no |
| post-asinh transform 8/9 | 2,188 | 0.9 | 1.35212 | 1.35212 | no |
| post-asinh transform 9/9 | 2,188 | 0.9 | 1.35212 | 1.35212 | no |
| allocation beta | 2,188 | 1 | 1.33368 | 1.33368 | no |
| knot refit | 2,188 | 1 | 1.3287 | 1.3287 | no |
| covariance shape 1/5 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| covariance shape 2/5 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| covariance shape 3/5 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| covariance shape 4/5 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| covariance shape 5/5 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| post-asinh transform 1/9 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| post-asinh transform 2/9 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| post-asinh transform 3/9 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| post-asinh transform 4/9 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| post-asinh transform 5/9 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| post-asinh transform 6/9 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| post-asinh transform 7/9 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| post-asinh transform 8/9 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| post-asinh transform 9/9 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| allocation beta | 2,188 | 1 | 1.3287 | 1.3287 | no |
| knot refit | 2,188 | 1 | 1.3287 | 1.3287 | no |
| refined parameter-search resolution | 2,188 | 1 | 1.3287 | 1.3287 | no |
| covariance shape 1/5 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| covariance shape 2/5 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| covariance shape 3/5 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| covariance shape 4/5 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| covariance shape 5/5 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| post-asinh transform 1/9 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| post-asinh transform 2/9 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| post-asinh transform 3/9 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| post-asinh transform 4/9 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| post-asinh transform 5/9 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| post-asinh transform 6/9 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| post-asinh transform 7/9 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| post-asinh transform 8/9 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| post-asinh transform 9/9 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| allocation beta | 2,188 | 1 | 1.3287 | 1.3287 | no |
| knot refit (cached stationary) | 2,188 | 1 | 1.3287 | 1.3287 | no |
| refined parameter-search resolution | 2,188 | 1 | 1.3287 | 1.3287 | no |
| covariance shape 1/5 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| covariance shape 2/5 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| covariance shape 3/5 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| covariance shape 4/5 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| covariance shape 5/5 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| post-asinh transform 1/9 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| post-asinh transform 2/9 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| post-asinh transform 3/9 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| post-asinh transform 4/9 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| post-asinh transform 5/9 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| post-asinh transform 6/9 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| post-asinh transform 7/9 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| post-asinh transform 8/9 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| post-asinh transform 9/9 | 2,188 | 1 | 1.3287 | 1.3287 | no |
| allocation beta | 2,188 | 1 | 1.3287 | 1.3287 | no |
| knot refit (cached stationary) | 2,188 | 1 | 1.3287 | 1.3287 | no |
| fixed point reached | 2,188 | 1 | 1.3287 | 1.3287 | no |
| rejected prune | 2,188 | 1 | 1.04409 | 1.04409 | no |
| cold restart 2279->2234 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| covariance shape 1/5 | 2,234 | 1 | 1.06385 | 1.06385 | no |
| covariance shape 2/5 | 2,234 | 1 | 1.06385 | 1.06385 | no |
| covariance shape 3/5 | 2,234 | 1 | 1.06385 | 1.06385 | no |
| covariance shape 4/5 | 2,234 | 1 | 1.06385 | 1.06385 | no |
| covariance shape 5/5 | 2,234 | 1 | 1.06385 | 1.06385 | no |
| post-asinh transform 1/9 | 2,234 | 1 | 1.06385 | 1.06385 | no |
| post-asinh transform 2/9 | 2,234 | 1 | 1.06385 | 1.06385 | no |
| post-asinh transform 3/9 | 2,234 | 1 | 1.06385 | 1.06385 | no |
| post-asinh transform 4/9 | 2,234 | 1 | 1.06385 | 1.06385 | no |
| post-asinh transform 5/9 | 2,234 | 1 | 1.06385 | 1.06385 | no |
| post-asinh transform 6/9 | 2,234 | 1 | 1.06385 | 1.06385 | no |
| post-asinh transform 7/9 | 2,234 | 1 | 1.06385 | 1.06385 | no |
| post-asinh transform 8/9 | 2,234 | 1 | 1.06385 | 1.06385 | no |
| post-asinh transform 9/9 | 2,234 | 1 | 1.06385 | 1.06385 | no |
| allocation beta | 2,234 | 0.9 | 1.06015 | 1.06015 | no |
| knot refit | 2,234 | 0.9 | 1.05614 | 1.05614 | no |
| covariance shape 1/5 | 2,234 | 0.9 | 1.05614 | 1.05614 | no |
| covariance shape 2/5 | 2,234 | 0.9 | 1.05614 | 1.05614 | no |
| covariance shape 3/5 | 2,234 | 0.9 | 1.05614 | 1.05614 | no |
| covariance shape 4/5 | 2,234 | 0.9 | 1.05614 | 1.05614 | no |
| covariance shape 5/5 | 2,234 | 0.9 | 1.05614 | 1.05614 | no |
| post-asinh transform 1/9 | 2,234 | 0.9 | 1.05614 | 1.05614 | no |
| post-asinh transform 2/9 | 2,234 | 0.9 | 1.05614 | 1.05614 | no |
| post-asinh transform 3/9 | 2,234 | 0.9 | 1.05614 | 1.05614 | no |
| post-asinh transform 4/9 | 2,234 | 0.9 | 1.05614 | 1.05614 | no |
| post-asinh transform 5/9 | 2,234 | 0.9 | 1.05614 | 1.05614 | no |
| post-asinh transform 6/9 | 2,234 | 0.9 | 1.05614 | 1.05614 | no |
| post-asinh transform 7/9 | 2,234 | 0.9 | 1.05614 | 1.05614 | no |
| post-asinh transform 8/9 | 2,234 | 0.9 | 1.05614 | 1.05614 | no |
| post-asinh transform 9/9 | 2,234 | 0.9 | 1.05614 | 1.05614 | no |
| allocation beta | 2,234 | 1 | 1.0452 | 1.0452 | no |
| knot refit | 2,234 | 1 | 1.04297 | 1.04297 | no |
| covariance shape 1/5 | 2,234 | 1 | 1.04297 | 1.04297 | no |
| covariance shape 2/5 | 2,234 | 1 | 1.04297 | 1.04297 | no |
| covariance shape 3/5 | 2,234 | 1 | 1.04297 | 1.04297 | no |
| covariance shape 4/5 | 2,234 | 1 | 1.04297 | 1.04297 | no |
| covariance shape 5/5 | 2,234 | 1 | 1.04297 | 1.04297 | no |
| post-asinh transform 1/9 | 2,234 | 1 | 1.04297 | 1.04297 | no |
| post-asinh transform 2/9 | 2,234 | 1 | 1.04297 | 1.04297 | no |
| post-asinh transform 3/9 | 2,234 | 1 | 1.04297 | 1.04297 | no |
| post-asinh transform 4/9 | 2,234 | 1 | 1.04297 | 1.04297 | no |
| post-asinh transform 5/9 | 2,234 | 1 | 1.04297 | 1.04297 | no |
| post-asinh transform 6/9 | 2,234 | 1 | 1.04297 | 1.04297 | no |
| post-asinh transform 7/9 | 2,234 | 1 | 1.04297 | 1.04297 | no |
| post-asinh transform 8/9 | 2,234 | 1 | 1.04297 | 1.04297 | no |
| post-asinh transform 9/9 | 2,234 | 1 | 1.04297 | 1.04297 | no |
| allocation beta | 2,234 | 1 | 1.04297 | 1.04297 | no |
| knot refit | 2,234 | 1 | 1.04023 | 1.04023 | no |
| covariance shape 1/5 | 2,234 | 1 | 1.04023 | 1.04023 | no |
| covariance shape 2/5 | 2,234 | 1 | 1.04023 | 1.04023 | no |
| covariance shape 3/5 | 2,234 | 1 | 1.04023 | 1.04023 | no |
| covariance shape 4/5 | 2,234 | 1 | 1.04023 | 1.04023 | no |
| covariance shape 5/5 | 2,234 | 1 | 1.04023 | 1.04023 | no |
| post-asinh transform 1/9 | 2,234 | 1 | 1.04023 | 1.04023 | no |
| post-asinh transform 2/9 | 2,234 | 1 | 1.04023 | 1.04023 | no |
| post-asinh transform 3/9 | 2,234 | 1 | 1.04023 | 1.04023 | no |
| post-asinh transform 4/9 | 2,234 | 1 | 1.04023 | 1.04023 | no |
| post-asinh transform 5/9 | 2,234 | 1 | 1.04023 | 1.04023 | no |
| post-asinh transform 6/9 | 2,234 | 1 | 1.04023 | 1.04023 | no |
| post-asinh transform 7/9 | 2,234 | 1 | 1.04023 | 1.04023 | no |
| post-asinh transform 8/9 | 2,234 | 1 | 1.04023 | 1.04023 | no |
| post-asinh transform 9/9 | 2,234 | 1 | 1.04023 | 1.04023 | no |
| allocation beta | 2,234 | 1 | 1.04023 | 1.04023 | no |
| knot refit | 2,234 | 1 | 1.0391 | 1.0391 | no |
| covariance shape 1/5 | 2,234 | 1 | 1.0391 | 1.0391 | no |
| covariance shape 2/5 | 2,234 | 1 | 1.0391 | 1.0391 | no |
| covariance shape 3/5 | 2,234 | 1 | 1.0391 | 1.0391 | no |
| covariance shape 4/5 | 2,234 | 1 | 1.0391 | 1.0391 | no |
| covariance shape 5/5 | 2,234 | 1 | 1.0391 | 1.0391 | no |
| post-asinh transform 1/9 | 2,234 | 1 | 1.0391 | 1.0391 | no |
| post-asinh transform 2/9 | 2,234 | 1 | 1.0391 | 1.0391 | no |
| post-asinh transform 3/9 | 2,234 | 1 | 1.0391 | 1.0391 | no |
| post-asinh transform 4/9 | 2,234 | 1 | 1.0391 | 1.0391 | no |
| post-asinh transform 5/9 | 2,234 | 1 | 1.0391 | 1.0391 | no |
| post-asinh transform 6/9 | 2,234 | 1 | 1.0391 | 1.0391 | no |
| post-asinh transform 7/9 | 2,234 | 1 | 1.0391 | 1.0391 | no |
| post-asinh transform 8/9 | 2,234 | 1 | 1.0391 | 1.0391 | no |
| post-asinh transform 9/9 | 2,234 | 1 | 1.0391 | 1.0391 | no |
| allocation beta | 2,234 | 1 | 1.0391 | 1.0391 | no |
| knot refit | 2,234 | 1 | 1.03767 | 1.03767 | no |
| covariance shape 1/5 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| covariance shape 2/5 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| covariance shape 3/5 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| covariance shape 4/5 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| covariance shape 5/5 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| post-asinh transform 1/9 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| post-asinh transform 2/9 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| post-asinh transform 3/9 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| post-asinh transform 4/9 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| post-asinh transform 5/9 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| post-asinh transform 6/9 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| post-asinh transform 7/9 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| post-asinh transform 8/9 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| post-asinh transform 9/9 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| allocation beta | 2,234 | 1 | 1.03767 | 1.03767 | no |
| knot refit | 2,234 | 1 | 1.03767 | 1.03767 | no |
| refined parameter-search resolution | 2,234 | 1 | 1.03767 | 1.03767 | no |
| covariance shape 1/5 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| covariance shape 2/5 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| covariance shape 3/5 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| covariance shape 4/5 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| covariance shape 5/5 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| post-asinh transform 1/9 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| post-asinh transform 2/9 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| post-asinh transform 3/9 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| post-asinh transform 4/9 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| post-asinh transform 5/9 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| post-asinh transform 6/9 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| post-asinh transform 7/9 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| post-asinh transform 8/9 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| post-asinh transform 9/9 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| allocation beta | 2,234 | 1 | 1.03767 | 1.03767 | no |
| knot refit (cached stationary) | 2,234 | 1 | 1.03767 | 1.03767 | no |
| refined parameter-search resolution | 2,234 | 1 | 1.03767 | 1.03767 | no |
| covariance shape 1/5 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| covariance shape 2/5 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| covariance shape 3/5 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| covariance shape 4/5 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| covariance shape 5/5 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| post-asinh transform 1/9 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| post-asinh transform 2/9 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| post-asinh transform 3/9 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| post-asinh transform 4/9 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| post-asinh transform 5/9 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| post-asinh transform 6/9 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| post-asinh transform 7/9 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| post-asinh transform 8/9 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| post-asinh transform 9/9 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| allocation beta | 2,234 | 1 | 1.03767 | 1.03767 | no |
| knot refit (cached stationary) | 2,234 | 1 | 1.03767 | 1.03767 | no |
| fixed point reached | 2,234 | 1 | 1.03767 | 1.03767 | no |
| prune proposal 2279->2234 | 2,234 | 1 | 1.03767 | 1.03767 | no |
| covariance shape 1/5 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| covariance shape 2/5 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| covariance shape 3/5 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| covariance shape 4/5 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| covariance shape 5/5 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| post-asinh transform 1/9 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| post-asinh transform 2/9 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| post-asinh transform 3/9 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| post-asinh transform 4/9 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| post-asinh transform 5/9 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| post-asinh transform 6/9 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| post-asinh transform 7/9 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| post-asinh transform 8/9 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| post-asinh transform 9/9 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| allocation beta | 2,234 | 1 | 1.10924 | 1.10924 | no |
| knot refit | 2,234 | 1 | 1.10924 | 1.10924 | no |
| refined parameter-search resolution | 2,234 | 1 | 1.10924 | 1.10924 | no |
| covariance shape 1/5 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| covariance shape 2/5 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| covariance shape 3/5 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| covariance shape 4/5 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| covariance shape 5/5 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| post-asinh transform 1/9 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| post-asinh transform 2/9 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| post-asinh transform 3/9 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| post-asinh transform 4/9 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| post-asinh transform 5/9 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| post-asinh transform 6/9 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| post-asinh transform 7/9 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| post-asinh transform 8/9 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| post-asinh transform 9/9 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| allocation beta | 2,234 | 1 | 1.10924 | 1.10924 | no |
| knot refit (cached stationary) | 2,234 | 1 | 1.10924 | 1.10924 | no |
| refined parameter-search resolution | 2,234 | 1 | 1.10924 | 1.10924 | no |
| covariance shape 1/5 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| covariance shape 2/5 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| covariance shape 3/5 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| covariance shape 4/5 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| covariance shape 5/5 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| post-asinh transform 1/9 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| post-asinh transform 2/9 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| post-asinh transform 3/9 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| post-asinh transform 4/9 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| post-asinh transform 5/9 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| post-asinh transform 6/9 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| post-asinh transform 7/9 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| post-asinh transform 8/9 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| post-asinh transform 9/9 | 2,234 | 1 | 1.10924 | 1.10924 | no |
| allocation beta | 2,234 | 1 | 1.10924 | 1.10924 | no |
| knot refit (cached stationary) | 2,234 | 1 | 1.10924 | 1.10924 | no |
| fixed point reached | 2,234 | 1 | 1.10924 | 1.10924 | no |
| rejected prune | 2,234 | 1 | 1.03767 | 1.03767 | no |
| cold restart 2279->2257 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| covariance shape 1/5 | 2,257 | 1 | 1.0535 | 1.0535 | no |
| covariance shape 2/5 | 2,257 | 1 | 1.0535 | 1.0535 | no |
| covariance shape 3/5 | 2,257 | 1 | 1.0535 | 1.0535 | no |
| covariance shape 4/5 | 2,257 | 1 | 1.0535 | 1.0535 | no |
| covariance shape 5/5 | 2,257 | 1 | 1.0535 | 1.0535 | no |
| post-asinh transform 1/9 | 2,257 | 1 | 1.0535 | 1.0535 | no |
| post-asinh transform 2/9 | 2,257 | 1 | 1.0535 | 1.0535 | no |
| post-asinh transform 3/9 | 2,257 | 1 | 1.0535 | 1.0535 | no |
| post-asinh transform 4/9 | 2,257 | 1 | 1.0535 | 1.0535 | no |
| post-asinh transform 5/9 | 2,257 | 1 | 1.0535 | 1.0535 | no |
| post-asinh transform 6/9 | 2,257 | 1 | 1.0535 | 1.0535 | no |
| post-asinh transform 7/9 | 2,257 | 1 | 1.0535 | 1.0535 | no |
| post-asinh transform 8/9 | 2,257 | 1 | 1.0535 | 1.0535 | no |
| post-asinh transform 9/9 | 2,257 | 1 | 1.0535 | 1.0535 | no |
| allocation beta | 2,257 | 0.9 | 1.05206 | 1.05206 | no |
| knot refit | 2,257 | 0.9 | 1.04692 | 1.04692 | no |
| covariance shape 1/5 | 2,257 | 0.9 | 1.04692 | 1.04692 | no |
| covariance shape 2/5 | 2,257 | 0.9 | 1.04692 | 1.04692 | no |
| covariance shape 3/5 | 2,257 | 0.9 | 1.04692 | 1.04692 | no |
| covariance shape 4/5 | 2,257 | 0.9 | 1.04692 | 1.04692 | no |
| covariance shape 5/5 | 2,257 | 0.9 | 1.04692 | 1.04692 | no |
| post-asinh transform 1/9 | 2,257 | 0.9 | 1.04692 | 1.04692 | no |
| post-asinh transform 2/9 | 2,257 | 0.9 | 1.04692 | 1.04692 | no |
| post-asinh transform 3/9 | 2,257 | 0.9 | 1.04692 | 1.04692 | no |
| post-asinh transform 4/9 | 2,257 | 0.9 | 1.04692 | 1.04692 | no |
| post-asinh transform 5/9 | 2,257 | 0.9 | 1.04692 | 1.04692 | no |
| post-asinh transform 6/9 | 2,257 | 0.9 | 1.04692 | 1.04692 | no |
| post-asinh transform 7/9 | 2,257 | 0.9 | 1.04692 | 1.04692 | no |
| post-asinh transform 8/9 | 2,257 | 0.9 | 1.04692 | 1.04692 | no |
| post-asinh transform 9/9 | 2,257 | 0.9 | 1.04692 | 1.04692 | no |
| allocation beta | 2,257 | 1 | 1.03447 | 1.03447 | no |
| knot refit | 2,257 | 1 | 1.03447 | 1.03447 | no |
| covariance shape 1/5 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| covariance shape 2/5 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| covariance shape 3/5 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| covariance shape 4/5 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| covariance shape 5/5 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| post-asinh transform 1/9 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| post-asinh transform 2/9 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| post-asinh transform 3/9 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| post-asinh transform 4/9 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| post-asinh transform 5/9 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| post-asinh transform 6/9 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| post-asinh transform 7/9 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| post-asinh transform 8/9 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| post-asinh transform 9/9 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| allocation beta | 2,257 | 1 | 1.03447 | 1.03447 | no |
| knot refit (cached stationary) | 2,257 | 1 | 1.03447 | 1.03447 | no |
| refined parameter-search resolution | 2,257 | 1 | 1.03447 | 1.03447 | no |
| covariance shape 1/5 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| covariance shape 2/5 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| covariance shape 3/5 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| covariance shape 4/5 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| covariance shape 5/5 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| post-asinh transform 1/9 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| post-asinh transform 2/9 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| post-asinh transform 3/9 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| post-asinh transform 4/9 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| post-asinh transform 5/9 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| post-asinh transform 6/9 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| post-asinh transform 7/9 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| post-asinh transform 8/9 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| post-asinh transform 9/9 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| allocation beta | 2,257 | 1 | 1.03447 | 1.03447 | no |
| knot refit (cached stationary) | 2,257 | 1 | 1.03447 | 1.03447 | no |
| refined parameter-search resolution | 2,257 | 1 | 1.03447 | 1.03447 | no |
| covariance shape 1/5 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| covariance shape 2/5 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| covariance shape 3/5 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| covariance shape 4/5 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| covariance shape 5/5 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| post-asinh transform 1/9 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| post-asinh transform 2/9 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| post-asinh transform 3/9 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| post-asinh transform 4/9 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| post-asinh transform 5/9 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| post-asinh transform 6/9 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| post-asinh transform 7/9 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| post-asinh transform 8/9 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| post-asinh transform 9/9 | 2,257 | 1 | 1.03447 | 1.03447 | no |
| allocation beta | 2,257 | 1 | 1.03447 | 1.03447 | no |
| knot refit (cached stationary) | 2,257 | 1 | 1.03447 | 1.03447 | no |
| fixed point reached | 2,257 | 1 | 1.03447 | 1.03447 | no |
| prune proposal 2279->2257 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| covariance shape 1/5 | 2,257 | 1 | 1.02838 | 1.02838 | no |
| covariance shape 2/5 | 2,257 | 1 | 1.02838 | 1.02838 | no |
| covariance shape 3/5 | 2,257 | 1 | 1.02838 | 1.02838 | no |
| covariance shape 4/5 | 2,257 | 1 | 1.02838 | 1.02838 | no |
| covariance shape 5/5 | 2,257 | 1 | 1.02838 | 1.02838 | no |
| post-asinh transform 1/9 | 2,257 | 1 | 1.02838 | 1.02838 | no |
| post-asinh transform 2/9 | 2,257 | 1 | 1.02838 | 1.02838 | no |
| post-asinh transform 3/9 | 2,257 | 1 | 1.02838 | 1.02838 | no |
| post-asinh transform 4/9 | 2,257 | 1 | 1.02838 | 1.02838 | no |
| post-asinh transform 5/9 | 2,257 | 1 | 1.02838 | 1.02838 | no |
| post-asinh transform 6/9 | 2,257 | 1 | 1.02838 | 1.02838 | no |
| post-asinh transform 7/9 | 2,257 | 1 | 1.02838 | 1.02838 | no |
| post-asinh transform 8/9 | 2,257 | 1 | 1.02838 | 1.02838 | no |
| post-asinh transform 9/9 | 2,257 | 1 | 1.02838 | 1.02838 | no |
| allocation beta | 2,257 | 0.9 | 1.02191 | 1.02191 | no |
| knot refit | 2,257 | 0.9 | 1.02191 | 1.02191 | no |
| covariance shape 1/5 | 2,257 | 0.9 | 1.02191 | 1.02191 | no |
| covariance shape 2/5 | 2,257 | 0.9 | 1.02191 | 1.02191 | no |
| covariance shape 3/5 | 2,257 | 0.9 | 1.02191 | 1.02191 | no |
| covariance shape 4/5 | 2,257 | 0.9 | 1.02191 | 1.02191 | no |
| covariance shape 5/5 | 2,257 | 0.9 | 1.02191 | 1.02191 | no |
| post-asinh transform 1/9 | 2,257 | 0.9 | 1.02191 | 1.02191 | no |
| post-asinh transform 2/9 | 2,257 | 0.9 | 1.02191 | 1.02191 | no |
| post-asinh transform 3/9 | 2,257 | 0.9 | 1.02191 | 1.02191 | no |
| post-asinh transform 4/9 | 2,257 | 0.9 | 1.02191 | 1.02191 | no |
| post-asinh transform 5/9 | 2,257 | 0.9 | 1.02191 | 1.02191 | no |
| post-asinh transform 6/9 | 2,257 | 0.9 | 1.02191 | 1.02191 | no |
| post-asinh transform 7/9 | 2,257 | 0.9 | 1.02191 | 1.02191 | no |
| post-asinh transform 8/9 | 2,257 | 0.9 | 1.02191 | 1.02191 | no |
| post-asinh transform 9/9 | 2,257 | 0.9 | 1.02191 | 1.02191 | no |
| allocation beta | 2,257 | 1 | 1.01372 | 1.01372 | no |
| knot refit | 2,257 | 1 | 1.01372 | 1.01372 | no |
| covariance shape 1/5 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| covariance shape 2/5 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| covariance shape 3/5 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| covariance shape 4/5 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| covariance shape 5/5 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| post-asinh transform 1/9 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| post-asinh transform 2/9 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| post-asinh transform 3/9 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| post-asinh transform 4/9 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| post-asinh transform 5/9 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| post-asinh transform 6/9 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| post-asinh transform 7/9 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| post-asinh transform 8/9 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| post-asinh transform 9/9 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| allocation beta | 2,257 | 1 | 1.01372 | 1.01372 | no |
| knot refit (cached stationary) | 2,257 | 1 | 1.01372 | 1.01372 | no |
| refined parameter-search resolution | 2,257 | 1 | 1.01372 | 1.01372 | no |
| covariance shape 1/5 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| covariance shape 2/5 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| covariance shape 3/5 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| covariance shape 4/5 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| covariance shape 5/5 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| post-asinh transform 1/9 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| post-asinh transform 2/9 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| post-asinh transform 3/9 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| post-asinh transform 4/9 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| post-asinh transform 5/9 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| post-asinh transform 6/9 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| post-asinh transform 7/9 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| post-asinh transform 8/9 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| post-asinh transform 9/9 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| allocation beta | 2,257 | 1 | 1.01372 | 1.01372 | no |
| knot refit (cached stationary) | 2,257 | 1 | 1.01372 | 1.01372 | no |
| refined parameter-search resolution | 2,257 | 1 | 1.01372 | 1.01372 | no |
| covariance shape 1/5 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| covariance shape 2/5 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| covariance shape 3/5 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| covariance shape 4/5 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| covariance shape 5/5 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| post-asinh transform 1/9 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| post-asinh transform 2/9 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| post-asinh transform 3/9 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| post-asinh transform 4/9 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| post-asinh transform 5/9 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| post-asinh transform 6/9 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| post-asinh transform 7/9 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| post-asinh transform 8/9 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| post-asinh transform 9/9 | 2,257 | 1 | 1.01372 | 1.01372 | no |
| allocation beta | 2,257 | 1 | 1.01372 | 1.01372 | no |
| knot refit (cached stationary) | 2,257 | 1 | 1.01372 | 1.01372 | no |
| fixed point reached | 2,257 | 1 | 1.01372 | 1.01372 | no |
| rejected prune | 2,257 | 1 | 1.01372 | 1.01372 | no |
| prune proposal 2279->2268 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| covariance shape 1/5 | 2,268 | 1 | 1.01113 | 1.01113 | no |
| covariance shape 2/5 | 2,268 | 1 | 1.01113 | 1.01113 | no |
| covariance shape 3/5 | 2,268 | 1 | 1.01113 | 1.01113 | no |
| covariance shape 4/5 | 2,268 | 1 | 1.01113 | 1.01113 | no |
| covariance shape 5/5 | 2,268 | 1 | 1.01113 | 1.01113 | no |
| post-asinh transform 1/9 | 2,268 | 1 | 1.01113 | 1.01113 | no |
| post-asinh transform 2/9 | 2,268 | 1 | 1.01113 | 1.01113 | no |
| post-asinh transform 3/9 | 2,268 | 1 | 1.01113 | 1.01113 | no |
| post-asinh transform 4/9 | 2,268 | 1 | 1.01113 | 1.01113 | no |
| post-asinh transform 5/9 | 2,268 | 1 | 1.01113 | 1.01113 | no |
| post-asinh transform 6/9 | 2,268 | 1 | 1.01113 | 1.01113 | no |
| post-asinh transform 7/9 | 2,268 | 1 | 1.01113 | 1.01113 | no |
| post-asinh transform 8/9 | 2,268 | 1 | 1.01113 | 1.01113 | no |
| post-asinh transform 9/9 | 2,268 | 1 | 1.01113 | 1.01113 | no |
| allocation beta | 2,268 | 1 | 1.01113 | 1.01113 | no |
| knot refit | 2,268 | 1 | 1.00785 | 1.00785 | no |
| covariance shape 1/5 | 2,268 | 1 | 1.00785 | 1.00785 | no |
| covariance shape 2/5 | 2,268 | 1 | 1.00785 | 1.00785 | no |
| covariance shape 3/5 | 2,268 | 1 | 1.00785 | 1.00785 | no |
| covariance shape 4/5 | 2,268 | 1 | 1.00785 | 1.00785 | no |
| covariance shape 5/5 | 2,268 | 1 | 1.00785 | 1.00785 | no |
| post-asinh transform 1/9 | 2,268 | 1 | 1.00785 | 1.00785 | no |
| post-asinh transform 2/9 | 2,268 | 1 | 1.00785 | 1.00785 | no |
| post-asinh transform 3/9 | 2,268 | 1 | 1.00785 | 1.00785 | no |
| post-asinh transform 4/9 | 2,268 | 1 | 1.00785 | 1.00785 | no |
| post-asinh transform 5/9 | 2,268 | 1 | 1.00785 | 1.00785 | no |
| post-asinh transform 6/9 | 2,268 | 1 | 1.00785 | 1.00785 | no |
| post-asinh transform 7/9 | 2,268 | 1 | 1.00785 | 1.00785 | no |
| post-asinh transform 8/9 | 2,268 | 1 | 1.00785 | 1.00785 | no |
| post-asinh transform 9/9 | 2,268 | 1 | 1.00785 | 1.00785 | no |
| allocation beta | 2,268 | 1 | 1.00785 | 1.00785 | no |
| knot refit | 2,268 | 1 | 1.00354 | 1.00354 | no |
| covariance shape 1/5 | 2,268 | 1 | 1.00354 | 1.00354 | no |
| covariance shape 2/5 | 2,268 | 1 | 1.00354 | 1.00354 | no |
| covariance shape 3/5 | 2,268 | 1 | 1.00354 | 1.00354 | no |
| covariance shape 4/5 | 2,268 | 1 | 1.00354 | 1.00354 | no |
| covariance shape 5/5 | 2,268 | 1 | 1.00354 | 1.00354 | no |
| post-asinh transform 1/9 | 2,268 | 1 | 1.00354 | 1.00354 | no |
| post-asinh transform 2/9 | 2,268 | 1 | 1.00354 | 1.00354 | no |
| post-asinh transform 3/9 | 2,268 | 1 | 1.00354 | 1.00354 | no |
| post-asinh transform 4/9 | 2,268 | 1 | 1.00354 | 1.00354 | no |
| post-asinh transform 5/9 | 2,268 | 1 | 1.00354 | 1.00354 | no |
| post-asinh transform 6/9 | 2,268 | 1 | 1.00354 | 1.00354 | no |
| post-asinh transform 7/9 | 2,268 | 1 | 1.00354 | 1.00354 | no |
| post-asinh transform 8/9 | 2,268 | 1 | 1.00354 | 1.00354 | no |
| post-asinh transform 9/9 | 2,268 | 1 | 1.00354 | 1.00354 | no |
| allocation beta | 2,268 | 1 | 1.00354 | 1.00354 | no |
| knot refit | 2,268 | 1 | 0.999525 | 0.999525 | yes |
| covariance shape 1/5 | 2,268 | 1 | 0.999525 | 0.999525 | yes |
| covariance shape 2/5 | 2,268 | 1 | 0.999525 | 0.999525 | yes |
| covariance shape 3/5 | 2,268 | 1 | 0.999525 | 0.999525 | yes |
| covariance shape 4/5 | 2,268 | 1 | 0.999525 | 0.999525 | yes |
| covariance shape 5/5 | 2,268 | 1 | 0.999525 | 0.999525 | yes |
| post-asinh transform 1/9 | 2,268 | 1 | 0.999525 | 0.999525 | yes |
| post-asinh transform 2/9 | 2,268 | 1 | 0.999525 | 0.999525 | yes |
| post-asinh transform 3/9 | 2,268 | 1 | 0.999525 | 0.999525 | yes |
| post-asinh transform 4/9 | 2,268 | 1 | 0.999525 | 0.999525 | yes |
| post-asinh transform 5/9 | 2,268 | 1 | 0.999525 | 0.999525 | yes |
| post-asinh transform 6/9 | 2,268 | 1 | 0.999525 | 0.999525 | yes |
| post-asinh transform 7/9 | 2,268 | 1 | 0.999525 | 0.999525 | yes |
| post-asinh transform 8/9 | 2,268 | 1 | 0.999525 | 0.999525 | yes |
| post-asinh transform 9/9 | 2,268 | 1 | 0.999525 | 0.999525 | yes |
| allocation beta | 2,268 | 1 | 0.999525 | 0.999525 | yes |
| knot refit | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| covariance shape 1/5 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| covariance shape 2/5 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| covariance shape 3/5 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| covariance shape 4/5 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| covariance shape 5/5 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| post-asinh transform 1/9 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| post-asinh transform 2/9 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| post-asinh transform 3/9 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| post-asinh transform 4/9 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| post-asinh transform 5/9 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| post-asinh transform 6/9 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| post-asinh transform 7/9 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| post-asinh transform 8/9 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| post-asinh transform 9/9 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| allocation beta | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| knot refit | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| refined parameter-search resolution | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| covariance shape 1/5 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| covariance shape 2/5 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| covariance shape 3/5 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| covariance shape 4/5 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| covariance shape 5/5 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| post-asinh transform 1/9 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| post-asinh transform 2/9 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| post-asinh transform 3/9 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| post-asinh transform 4/9 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| post-asinh transform 5/9 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| post-asinh transform 6/9 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| post-asinh transform 7/9 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| post-asinh transform 8/9 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| post-asinh transform 9/9 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| allocation beta | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| knot refit (cached stationary) | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| refined parameter-search resolution | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| covariance shape 1/5 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| covariance shape 2/5 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| covariance shape 3/5 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| covariance shape 4/5 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| covariance shape 5/5 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| post-asinh transform 1/9 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| post-asinh transform 2/9 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| post-asinh transform 3/9 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| post-asinh transform 4/9 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| post-asinh transform 5/9 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| post-asinh transform 6/9 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| post-asinh transform 7/9 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| post-asinh transform 8/9 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| post-asinh transform 9/9 | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| allocation beta | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| knot refit (cached stationary) | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| fixed point reached | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| accepted prune | 2,268 | 1 | 0.997097 | 0.997097 | yes |
| cold restart 2268->2257 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| covariance shape 1/5 | 2,257 | 1 | 1.06611 | 1.06611 | no |
| covariance shape 2/5 | 2,257 | 1 | 1.06611 | 1.06611 | no |
| covariance shape 3/5 | 2,257 | 1 | 1.06611 | 1.06611 | no |
| covariance shape 4/5 | 2,257 | 1 | 1.06611 | 1.06611 | no |
| covariance shape 5/5 | 2,257 | 1 | 1.06611 | 1.06611 | no |
| post-asinh transform 1/9 | 2,257 | 1 | 1.06611 | 1.06611 | no |
| post-asinh transform 2/9 | 2,257 | 1 | 1.06611 | 1.06611 | no |
| post-asinh transform 3/9 | 2,257 | 1 | 1.06611 | 1.06611 | no |
| post-asinh transform 4/9 | 2,257 | 1 | 1.06611 | 1.06611 | no |
| post-asinh transform 5/9 | 2,257 | 1 | 1.06611 | 1.06611 | no |
| post-asinh transform 6/9 | 2,257 | 1 | 1.06611 | 1.06611 | no |
| post-asinh transform 7/9 | 2,257 | 1 | 1.06611 | 1.06611 | no |
| post-asinh transform 8/9 | 2,257 | 1 | 1.06611 | 1.06611 | no |
| post-asinh transform 9/9 | 2,257 | 1 | 1.06611 | 1.06611 | no |
| allocation beta | 2,257 | 0.9 | 1.06299 | 1.06299 | no |
| knot refit | 2,257 | 0.9 | 1.05838 | 1.05838 | no |
| covariance shape 1/5 | 2,257 | 0.9 | 1.05838 | 1.05838 | no |
| covariance shape 2/5 | 2,257 | 0.9 | 1.05838 | 1.05838 | no |
| covariance shape 3/5 | 2,257 | 0.9 | 1.05838 | 1.05838 | no |
| covariance shape 4/5 | 2,257 | 0.9 | 1.05838 | 1.05838 | no |
| covariance shape 5/5 | 2,257 | 0.9 | 1.05838 | 1.05838 | no |
| post-asinh transform 1/9 | 2,257 | 0.9 | 1.05838 | 1.05838 | no |
| post-asinh transform 2/9 | 2,257 | 0.9 | 1.05838 | 1.05838 | no |
| post-asinh transform 3/9 | 2,257 | 0.9 | 1.05838 | 1.05838 | no |
| post-asinh transform 4/9 | 2,257 | 0.9 | 1.05838 | 1.05838 | no |
| post-asinh transform 5/9 | 2,257 | 0.9 | 1.05838 | 1.05838 | no |
| post-asinh transform 6/9 | 2,257 | 0.9 | 1.05838 | 1.05838 | no |
| post-asinh transform 7/9 | 2,257 | 0.9 | 1.05838 | 1.05838 | no |
| post-asinh transform 8/9 | 2,257 | 0.9 | 1.05838 | 1.05838 | no |
| post-asinh transform 9/9 | 2,257 | 0.9 | 1.05838 | 1.05838 | no |
| allocation beta | 2,257 | 1 | 1.05263 | 1.05263 | no |
| knot refit | 2,257 | 1 | 1.05263 | 1.05263 | no |
| covariance shape 1/5 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| covariance shape 2/5 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| covariance shape 3/5 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| covariance shape 4/5 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| covariance shape 5/5 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| post-asinh transform 1/9 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| post-asinh transform 2/9 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| post-asinh transform 3/9 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| post-asinh transform 4/9 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| post-asinh transform 5/9 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| post-asinh transform 6/9 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| post-asinh transform 7/9 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| post-asinh transform 8/9 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| post-asinh transform 9/9 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| allocation beta | 2,257 | 1 | 1.05263 | 1.05263 | no |
| knot refit (cached stationary) | 2,257 | 1 | 1.05263 | 1.05263 | no |
| refined parameter-search resolution | 2,257 | 1 | 1.05263 | 1.05263 | no |
| covariance shape 1/5 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| covariance shape 2/5 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| covariance shape 3/5 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| covariance shape 4/5 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| covariance shape 5/5 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| post-asinh transform 1/9 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| post-asinh transform 2/9 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| post-asinh transform 3/9 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| post-asinh transform 4/9 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| post-asinh transform 5/9 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| post-asinh transform 6/9 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| post-asinh transform 7/9 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| post-asinh transform 8/9 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| post-asinh transform 9/9 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| allocation beta | 2,257 | 1 | 1.05263 | 1.05263 | no |
| knot refit (cached stationary) | 2,257 | 1 | 1.05263 | 1.05263 | no |
| refined parameter-search resolution | 2,257 | 1 | 1.05263 | 1.05263 | no |
| covariance shape 1/5 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| covariance shape 2/5 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| covariance shape 3/5 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| covariance shape 4/5 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| covariance shape 5/5 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| post-asinh transform 1/9 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| post-asinh transform 2/9 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| post-asinh transform 3/9 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| post-asinh transform 4/9 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| post-asinh transform 5/9 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| post-asinh transform 6/9 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| post-asinh transform 7/9 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| post-asinh transform 8/9 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| post-asinh transform 9/9 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| allocation beta | 2,257 | 1 | 1.05263 | 1.05263 | no |
| knot refit (cached stationary) | 2,257 | 1 | 1.05263 | 1.05263 | no |
| fixed point reached | 2,257 | 1 | 1.05263 | 1.05263 | no |
| prune proposal 2268->2257 | 2,257 | 1 | 1.05263 | 1.05263 | no |
| covariance shape 1/5 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| covariance shape 2/5 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| covariance shape 3/5 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| covariance shape 4/5 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| covariance shape 5/5 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| post-asinh transform 1/9 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| post-asinh transform 2/9 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| post-asinh transform 3/9 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| post-asinh transform 4/9 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| post-asinh transform 5/9 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| post-asinh transform 6/9 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| post-asinh transform 7/9 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| post-asinh transform 8/9 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| post-asinh transform 9/9 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| allocation beta | 2,257 | 1 | 1.11493 | 1.11493 | no |
| knot refit | 2,257 | 1 | 1.11493 | 1.11493 | no |
| refined parameter-search resolution | 2,257 | 1 | 1.11493 | 1.11493 | no |
| covariance shape 1/5 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| covariance shape 2/5 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| covariance shape 3/5 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| covariance shape 4/5 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| covariance shape 5/5 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| post-asinh transform 1/9 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| post-asinh transform 2/9 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| post-asinh transform 3/9 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| post-asinh transform 4/9 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| post-asinh transform 5/9 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| post-asinh transform 6/9 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| post-asinh transform 7/9 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| post-asinh transform 8/9 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| post-asinh transform 9/9 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| allocation beta | 2,257 | 1 | 1.11493 | 1.11493 | no |
| knot refit (cached stationary) | 2,257 | 1 | 1.11493 | 1.11493 | no |
| refined parameter-search resolution | 2,257 | 1 | 1.11493 | 1.11493 | no |
| covariance shape 1/5 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| covariance shape 2/5 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| covariance shape 3/5 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| covariance shape 4/5 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| covariance shape 5/5 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| post-asinh transform 1/9 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| post-asinh transform 2/9 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| post-asinh transform 3/9 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| post-asinh transform 4/9 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| post-asinh transform 5/9 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| post-asinh transform 6/9 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| post-asinh transform 7/9 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| post-asinh transform 8/9 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| post-asinh transform 9/9 | 2,257 | 1 | 1.11493 | 1.11493 | no |
| allocation beta | 2,257 | 1 | 1.11493 | 1.11493 | no |
| knot refit (cached stationary) | 2,257 | 1 | 1.11493 | 1.11493 | no |
| fixed point reached | 2,257 | 1 | 1.11493 | 1.11493 | no |
| rejected prune | 2,257 | 1 | 1.05263 | 1.05263 | no |
| cold restart 2268->2263 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| covariance shape 1/5 | 2,263 | 1 | 1.07594 | 1.07594 | no |
| covariance shape 2/5 | 2,263 | 1 | 1.07594 | 1.07594 | no |
| covariance shape 3/5 | 2,263 | 1 | 1.07594 | 1.07594 | no |
| covariance shape 4/5 | 2,263 | 1 | 1.07594 | 1.07594 | no |
| covariance shape 5/5 | 2,263 | 1 | 1.07594 | 1.07594 | no |
| post-asinh transform 1/9 | 2,263 | 1 | 1.07594 | 1.07594 | no |
| post-asinh transform 2/9 | 2,263 | 1 | 1.07594 | 1.07594 | no |
| post-asinh transform 3/9 | 2,263 | 1 | 1.07594 | 1.07594 | no |
| post-asinh transform 4/9 | 2,263 | 1 | 1.07594 | 1.07594 | no |
| post-asinh transform 5/9 | 2,263 | 1 | 1.07594 | 1.07594 | no |
| post-asinh transform 6/9 | 2,263 | 1 | 1.07594 | 1.07594 | no |
| post-asinh transform 7/9 | 2,263 | 1 | 1.07594 | 1.07594 | no |
| post-asinh transform 8/9 | 2,263 | 1 | 1.07594 | 1.07594 | no |
| post-asinh transform 9/9 | 2,263 | 1 | 1.07594 | 1.07594 | no |
| allocation beta | 2,263 | 0.9 | 1.07479 | 1.07479 | no |
| knot refit | 2,263 | 0.9 | 1.07083 | 1.07083 | no |
| covariance shape 1/5 | 2,263 | 0.9 | 1.07083 | 1.07083 | no |
| covariance shape 2/5 | 2,263 | 0.9 | 1.07083 | 1.07083 | no |
| covariance shape 3/5 | 2,263 | 0.9 | 1.07083 | 1.07083 | no |
| covariance shape 4/5 | 2,263 | 0.9 | 1.07083 | 1.07083 | no |
| covariance shape 5/5 | 2,263 | 0.9 | 1.07083 | 1.07083 | no |
| post-asinh transform 1/9 | 2,263 | 0.9 | 1.07083 | 1.07083 | no |
| post-asinh transform 2/9 | 2,263 | 0.9 | 1.07083 | 1.07083 | no |
| post-asinh transform 3/9 | 2,263 | 0.9 | 1.07083 | 1.07083 | no |
| post-asinh transform 4/9 | 2,263 | 0.9 | 1.07083 | 1.07083 | no |
| post-asinh transform 5/9 | 2,263 | 0.9 | 1.07083 | 1.07083 | no |
| post-asinh transform 6/9 | 2,263 | 0.9 | 1.07083 | 1.07083 | no |
| post-asinh transform 7/9 | 2,263 | 0.9 | 1.07083 | 1.07083 | no |
| post-asinh transform 8/9 | 2,263 | 0.9 | 1.07083 | 1.07083 | no |
| post-asinh transform 9/9 | 2,263 | 0.9 | 1.07083 | 1.07083 | no |
| allocation beta | 2,263 | 1 | 1.06169 | 1.06169 | no |
| knot refit | 2,263 | 1 | 1.06005 | 1.06005 | no |
| covariance shape 1/5 | 2,263 | 1 | 1.06005 | 1.06005 | no |
| covariance shape 2/5 | 2,263 | 1 | 1.06005 | 1.06005 | no |
| covariance shape 3/5 | 2,263 | 1 | 1.06005 | 1.06005 | no |
| covariance shape 4/5 | 2,263 | 1 | 1.06005 | 1.06005 | no |
| covariance shape 5/5 | 2,263 | 1 | 1.06005 | 1.06005 | no |
| post-asinh transform 1/9 | 2,263 | 1 | 1.06005 | 1.06005 | no |
| post-asinh transform 2/9 | 2,263 | 1 | 1.06005 | 1.06005 | no |
| post-asinh transform 3/9 | 2,263 | 1 | 1.06005 | 1.06005 | no |
| post-asinh transform 4/9 | 2,263 | 1 | 1.06005 | 1.06005 | no |
| post-asinh transform 5/9 | 2,263 | 1 | 1.06005 | 1.06005 | no |
| post-asinh transform 6/9 | 2,263 | 1 | 1.06005 | 1.06005 | no |
| post-asinh transform 7/9 | 2,263 | 1 | 1.06005 | 1.06005 | no |
| post-asinh transform 8/9 | 2,263 | 1 | 1.06005 | 1.06005 | no |
| post-asinh transform 9/9 | 2,263 | 1 | 1.06005 | 1.06005 | no |
| allocation beta | 2,263 | 1 | 1.06005 | 1.06005 | no |
| knot refit | 2,263 | 1 | 1.05888 | 1.05888 | no |
| covariance shape 1/5 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| covariance shape 2/5 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| covariance shape 3/5 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| covariance shape 4/5 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| covariance shape 5/5 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| post-asinh transform 1/9 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| post-asinh transform 2/9 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| post-asinh transform 3/9 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| post-asinh transform 4/9 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| post-asinh transform 5/9 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| post-asinh transform 6/9 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| post-asinh transform 7/9 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| post-asinh transform 8/9 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| post-asinh transform 9/9 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| allocation beta | 2,263 | 1 | 1.05888 | 1.05888 | no |
| knot refit | 2,263 | 1 | 1.05888 | 1.05888 | no |
| refined parameter-search resolution | 2,263 | 1 | 1.05888 | 1.05888 | no |
| covariance shape 1/5 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| covariance shape 2/5 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| covariance shape 3/5 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| covariance shape 4/5 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| covariance shape 5/5 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| post-asinh transform 1/9 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| post-asinh transform 2/9 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| post-asinh transform 3/9 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| post-asinh transform 4/9 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| post-asinh transform 5/9 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| post-asinh transform 6/9 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| post-asinh transform 7/9 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| post-asinh transform 8/9 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| post-asinh transform 9/9 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| allocation beta | 2,263 | 1 | 1.05888 | 1.05888 | no |
| knot refit (cached stationary) | 2,263 | 1 | 1.05888 | 1.05888 | no |
| refined parameter-search resolution | 2,263 | 1 | 1.05888 | 1.05888 | no |
| covariance shape 1/5 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| covariance shape 2/5 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| covariance shape 3/5 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| covariance shape 4/5 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| covariance shape 5/5 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| post-asinh transform 1/9 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| post-asinh transform 2/9 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| post-asinh transform 3/9 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| post-asinh transform 4/9 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| post-asinh transform 5/9 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| post-asinh transform 6/9 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| post-asinh transform 7/9 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| post-asinh transform 8/9 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| post-asinh transform 9/9 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| allocation beta | 2,263 | 1 | 1.05888 | 1.05888 | no |
| knot refit (cached stationary) | 2,263 | 1 | 1.05888 | 1.05888 | no |
| fixed point reached | 2,263 | 1 | 1.05888 | 1.05888 | no |
| prune proposal 2268->2263 | 2,263 | 1 | 1.05888 | 1.05888 | no |
| covariance shape 1/5 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| covariance shape 2/5 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| covariance shape 3/5 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| covariance shape 4/5 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| covariance shape 5/5 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| post-asinh transform 1/9 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| post-asinh transform 2/9 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| post-asinh transform 3/9 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| post-asinh transform 4/9 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| post-asinh transform 5/9 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| post-asinh transform 6/9 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| post-asinh transform 7/9 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| post-asinh transform 8/9 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| post-asinh transform 9/9 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| allocation beta | 2,263 | 1 | 1.11486 | 1.11486 | no |
| knot refit | 2,263 | 1 | 1.11486 | 1.11486 | no |
| refined parameter-search resolution | 2,263 | 1 | 1.11486 | 1.11486 | no |
| covariance shape 1/5 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| covariance shape 2/5 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| covariance shape 3/5 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| covariance shape 4/5 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| covariance shape 5/5 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| post-asinh transform 1/9 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| post-asinh transform 2/9 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| post-asinh transform 3/9 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| post-asinh transform 4/9 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| post-asinh transform 5/9 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| post-asinh transform 6/9 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| post-asinh transform 7/9 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| post-asinh transform 8/9 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| post-asinh transform 9/9 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| allocation beta | 2,263 | 1 | 1.11486 | 1.11486 | no |
| knot refit (cached stationary) | 2,263 | 1 | 1.11486 | 1.11486 | no |
| refined parameter-search resolution | 2,263 | 1 | 1.11486 | 1.11486 | no |
| covariance shape 1/5 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| covariance shape 2/5 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| covariance shape 3/5 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| covariance shape 4/5 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| covariance shape 5/5 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| post-asinh transform 1/9 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| post-asinh transform 2/9 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| post-asinh transform 3/9 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| post-asinh transform 4/9 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| post-asinh transform 5/9 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| post-asinh transform 6/9 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| post-asinh transform 7/9 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| post-asinh transform 8/9 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| post-asinh transform 9/9 | 2,263 | 1 | 1.11486 | 1.11486 | no |
| allocation beta | 2,263 | 1 | 1.11486 | 1.11486 | no |
| knot refit (cached stationary) | 2,263 | 1 | 1.11486 | 1.11486 | no |
| fixed point reached | 2,263 | 1 | 1.11486 | 1.11486 | no |
| rejected prune | 2,263 | 1 | 1.05888 | 1.05888 | no |
| cold restart 2268->2266 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| covariance shape 1/5 | 2,266 | 1 | 1.04654 | 1.04654 | no |
| covariance shape 2/5 | 2,266 | 1 | 1.04654 | 1.04654 | no |
| covariance shape 3/5 | 2,266 | 1 | 1.04654 | 1.04654 | no |
| covariance shape 4/5 | 2,266 | 1 | 1.04654 | 1.04654 | no |
| covariance shape 5/5 | 2,266 | 1 | 1.04654 | 1.04654 | no |
| post-asinh transform 1/9 | 2,266 | 1 | 1.04654 | 1.04654 | no |
| post-asinh transform 2/9 | 2,266 | 1 | 1.04654 | 1.04654 | no |
| post-asinh transform 3/9 | 2,266 | 1 | 1.04654 | 1.04654 | no |
| post-asinh transform 4/9 | 2,266 | 1 | 1.04654 | 1.04654 | no |
| post-asinh transform 5/9 | 2,266 | 1 | 1.04654 | 1.04654 | no |
| post-asinh transform 6/9 | 2,266 | 1 | 1.04654 | 1.04654 | no |
| post-asinh transform 7/9 | 2,266 | 1 | 1.04654 | 1.04654 | no |
| post-asinh transform 8/9 | 2,266 | 1 | 1.04654 | 1.04654 | no |
| post-asinh transform 9/9 | 2,266 | 1 | 1.04654 | 1.04654 | no |
| allocation beta | 2,266 | 1 | 1.04654 | 1.04654 | no |
| knot refit | 2,266 | 1 | 1.04234 | 1.04234 | no |
| covariance shape 1/5 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| covariance shape 2/5 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| covariance shape 3/5 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| covariance shape 4/5 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| covariance shape 5/5 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| post-asinh transform 1/9 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| post-asinh transform 2/9 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| post-asinh transform 3/9 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| post-asinh transform 4/9 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| post-asinh transform 5/9 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| post-asinh transform 6/9 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| post-asinh transform 7/9 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| post-asinh transform 8/9 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| post-asinh transform 9/9 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| allocation beta | 2,266 | 1 | 1.04234 | 1.04234 | no |
| knot refit | 2,266 | 1 | 1.04234 | 1.04234 | no |
| refined parameter-search resolution | 2,266 | 1 | 1.04234 | 1.04234 | no |
| covariance shape 1/5 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| covariance shape 2/5 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| covariance shape 3/5 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| covariance shape 4/5 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| covariance shape 5/5 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| post-asinh transform 1/9 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| post-asinh transform 2/9 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| post-asinh transform 3/9 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| post-asinh transform 4/9 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| post-asinh transform 5/9 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| post-asinh transform 6/9 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| post-asinh transform 7/9 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| post-asinh transform 8/9 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| post-asinh transform 9/9 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| allocation beta | 2,266 | 1 | 1.04234 | 1.04234 | no |
| knot refit (cached stationary) | 2,266 | 1 | 1.04234 | 1.04234 | no |
| refined parameter-search resolution | 2,266 | 1 | 1.04234 | 1.04234 | no |
| covariance shape 1/5 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| covariance shape 2/5 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| covariance shape 3/5 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| covariance shape 4/5 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| covariance shape 5/5 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| post-asinh transform 1/9 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| post-asinh transform 2/9 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| post-asinh transform 3/9 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| post-asinh transform 4/9 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| post-asinh transform 5/9 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| post-asinh transform 6/9 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| post-asinh transform 7/9 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| post-asinh transform 8/9 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| post-asinh transform 9/9 | 2,266 | 1 | 1.04234 | 1.04234 | no |
| allocation beta | 2,266 | 1 | 1.04234 | 1.04234 | no |
| knot refit (cached stationary) | 2,266 | 1 | 1.04234 | 1.04234 | no |
| fixed point reached | 2,266 | 1 | 1.04234 | 1.04234 | no |
| prune proposal 2268->2266 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| covariance shape 1/5 | 2,266 | 1 | 1.0099 | 1.0099 | no |
| covariance shape 2/5 | 2,266 | 1 | 1.0099 | 1.0099 | no |
| covariance shape 3/5 | 2,266 | 1 | 1.0099 | 1.0099 | no |
| covariance shape 4/5 | 2,266 | 1 | 1.0099 | 1.0099 | no |
| covariance shape 5/5 | 2,266 | 1 | 1.0099 | 1.0099 | no |
| post-asinh transform 1/9 | 2,266 | 1 | 1.0099 | 1.0099 | no |
| post-asinh transform 2/9 | 2,266 | 1 | 1.0099 | 1.0099 | no |
| post-asinh transform 3/9 | 2,266 | 1 | 1.0099 | 1.0099 | no |
| post-asinh transform 4/9 | 2,266 | 1 | 1.0099 | 1.0099 | no |
| post-asinh transform 5/9 | 2,266 | 1 | 1.0099 | 1.0099 | no |
| post-asinh transform 6/9 | 2,266 | 1 | 1.0099 | 1.0099 | no |
| post-asinh transform 7/9 | 2,266 | 1 | 1.0099 | 1.0099 | no |
| post-asinh transform 8/9 | 2,266 | 1 | 1.0099 | 1.0099 | no |
| post-asinh transform 9/9 | 2,266 | 1 | 1.0099 | 1.0099 | no |
| allocation beta | 2,266 | 1 | 1.0099 | 1.0099 | no |
| knot refit | 2,266 | 1 | 1.0078 | 1.0078 | no |
| covariance shape 1/5 | 2,266 | 1 | 1.0078 | 1.0078 | no |
| covariance shape 2/5 | 2,266 | 1 | 1.0078 | 1.0078 | no |
| covariance shape 3/5 | 2,266 | 1 | 1.0078 | 1.0078 | no |
| covariance shape 4/5 | 2,266 | 1 | 1.0078 | 1.0078 | no |
| covariance shape 5/5 | 2,266 | 1 | 1.0078 | 1.0078 | no |
| post-asinh transform 1/9 | 2,266 | 1 | 1.0078 | 1.0078 | no |
| post-asinh transform 2/9 | 2,266 | 1 | 1.0078 | 1.0078 | no |
| post-asinh transform 3/9 | 2,266 | 1 | 1.0078 | 1.0078 | no |
| post-asinh transform 4/9 | 2,266 | 1 | 1.0078 | 1.0078 | no |
| post-asinh transform 5/9 | 2,266 | 1 | 1.0078 | 1.0078 | no |
| post-asinh transform 6/9 | 2,266 | 1 | 1.0078 | 1.0078 | no |
| post-asinh transform 7/9 | 2,266 | 1 | 1.0078 | 1.0078 | no |
| post-asinh transform 8/9 | 2,266 | 1 | 1.0078 | 1.0078 | no |
| post-asinh transform 9/9 | 2,266 | 1 | 1.0078 | 1.0078 | no |
| allocation beta | 2,266 | 1 | 1.0078 | 1.0078 | no |
| knot refit | 2,266 | 1 | 1.00528 | 1.00528 | no |
| covariance shape 1/5 | 2,266 | 1 | 1.00528 | 1.00528 | no |
| covariance shape 2/5 | 2,266 | 1 | 1.00528 | 1.00528 | no |
| covariance shape 3/5 | 2,266 | 1 | 1.00528 | 1.00528 | no |
| covariance shape 4/5 | 2,266 | 1 | 1.00528 | 1.00528 | no |
| covariance shape 5/5 | 2,266 | 1 | 1.00528 | 1.00528 | no |
| post-asinh transform 1/9 | 2,266 | 1 | 1.00528 | 1.00528 | no |
| post-asinh transform 2/9 | 2,266 | 1 | 1.00528 | 1.00528 | no |
| post-asinh transform 3/9 | 2,266 | 1 | 1.00528 | 1.00528 | no |
| post-asinh transform 4/9 | 2,266 | 1 | 1.00528 | 1.00528 | no |
| post-asinh transform 5/9 | 2,266 | 1 | 1.00528 | 1.00528 | no |
| post-asinh transform 6/9 | 2,266 | 1 | 1.00528 | 1.00528 | no |
| post-asinh transform 7/9 | 2,266 | 1 | 1.00528 | 1.00528 | no |
| post-asinh transform 8/9 | 2,266 | 1 | 1.00528 | 1.00528 | no |
| post-asinh transform 9/9 | 2,266 | 1 | 1.00528 | 1.00528 | no |
| allocation beta | 2,266 | 1 | 1.00528 | 1.00528 | no |
| knot refit | 2,266 | 1 | 1.00326 | 1.00326 | no |
| covariance shape 1/5 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| covariance shape 2/5 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| covariance shape 3/5 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| covariance shape 4/5 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| covariance shape 5/5 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| post-asinh transform 1/9 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| post-asinh transform 2/9 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| post-asinh transform 3/9 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| post-asinh transform 4/9 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| post-asinh transform 5/9 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| post-asinh transform 6/9 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| post-asinh transform 7/9 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| post-asinh transform 8/9 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| post-asinh transform 9/9 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| allocation beta | 2,266 | 1 | 1.00326 | 1.00326 | no |
| knot refit | 2,266 | 1 | 1.00326 | 1.00326 | no |
| refined parameter-search resolution | 2,266 | 1 | 1.00326 | 1.00326 | no |
| covariance shape 1/5 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| covariance shape 2/5 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| covariance shape 3/5 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| covariance shape 4/5 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| covariance shape 5/5 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| post-asinh transform 1/9 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| post-asinh transform 2/9 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| post-asinh transform 3/9 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| post-asinh transform 4/9 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| post-asinh transform 5/9 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| post-asinh transform 6/9 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| post-asinh transform 7/9 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| post-asinh transform 8/9 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| post-asinh transform 9/9 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| allocation beta | 2,266 | 1 | 1.00326 | 1.00326 | no |
| knot refit (cached stationary) | 2,266 | 1 | 1.00326 | 1.00326 | no |
| refined parameter-search resolution | 2,266 | 1 | 1.00326 | 1.00326 | no |
| covariance shape 1/5 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| covariance shape 2/5 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| covariance shape 3/5 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| covariance shape 4/5 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| covariance shape 5/5 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| post-asinh transform 1/9 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| post-asinh transform 2/9 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| post-asinh transform 3/9 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| post-asinh transform 4/9 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| post-asinh transform 5/9 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| post-asinh transform 6/9 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| post-asinh transform 7/9 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| post-asinh transform 8/9 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| post-asinh transform 9/9 | 2,266 | 1 | 1.00326 | 1.00326 | no |
| allocation beta | 2,266 | 1 | 1.00326 | 1.00326 | no |
| knot refit (cached stationary) | 2,266 | 1 | 1.00326 | 1.00326 | no |
| fixed point reached | 2,266 | 1 | 1.00326 | 1.00326 | no |
| rejected prune | 2,266 | 1 | 1.00326 | 1.00326 | no |
| prune proposal 2268->2267 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| covariance shape 1/5 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| covariance shape 2/5 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| covariance shape 3/5 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| covariance shape 4/5 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| covariance shape 5/5 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| post-asinh transform 1/9 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| post-asinh transform 2/9 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| post-asinh transform 3/9 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| post-asinh transform 4/9 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| post-asinh transform 5/9 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| post-asinh transform 6/9 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| post-asinh transform 7/9 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| post-asinh transform 8/9 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| post-asinh transform 9/9 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| allocation beta | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| knot refit | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| refined parameter-search resolution | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| covariance shape 1/5 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| covariance shape 2/5 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| covariance shape 3/5 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| covariance shape 4/5 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| covariance shape 5/5 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| post-asinh transform 1/9 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| post-asinh transform 2/9 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| post-asinh transform 3/9 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| post-asinh transform 4/9 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| post-asinh transform 5/9 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| post-asinh transform 6/9 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| post-asinh transform 7/9 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| post-asinh transform 8/9 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| post-asinh transform 9/9 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| allocation beta | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| knot refit (cached stationary) | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| refined parameter-search resolution | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| covariance shape 1/5 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| covariance shape 2/5 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| covariance shape 3/5 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| covariance shape 4/5 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| covariance shape 5/5 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| post-asinh transform 1/9 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| post-asinh transform 2/9 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| post-asinh transform 3/9 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| post-asinh transform 4/9 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| post-asinh transform 5/9 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| post-asinh transform 6/9 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| post-asinh transform 7/9 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| post-asinh transform 8/9 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| post-asinh transform 9/9 | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| allocation beta | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| knot refit (cached stationary) | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| fixed point reached | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| accepted prune | 2,267 | 1 | 0.996908 | 0.996908 | yes |
| cold restart 2267->2266 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| covariance shape 1/5 | 2,266 | 1 | 1.0805 | 1.0805 | no |
| covariance shape 2/5 | 2,266 | 1 | 1.0805 | 1.0805 | no |
| covariance shape 3/5 | 2,266 | 1 | 1.0805 | 1.0805 | no |
| covariance shape 4/5 | 2,266 | 1 | 1.0805 | 1.0805 | no |
| covariance shape 5/5 | 2,266 | 1 | 1.0805 | 1.0805 | no |
| post-asinh transform 1/9 | 2,266 | 1 | 1.0805 | 1.0805 | no |
| post-asinh transform 2/9 | 2,266 | 1 | 1.0805 | 1.0805 | no |
| post-asinh transform 3/9 | 2,266 | 1 | 1.0805 | 1.0805 | no |
| post-asinh transform 4/9 | 2,266 | 1 | 1.0805 | 1.0805 | no |
| post-asinh transform 5/9 | 2,266 | 1 | 1.0805 | 1.0805 | no |
| post-asinh transform 6/9 | 2,266 | 1 | 1.0805 | 1.0805 | no |
| post-asinh transform 7/9 | 2,266 | 1 | 1.0805 | 1.0805 | no |
| post-asinh transform 8/9 | 2,266 | 1 | 1.0805 | 1.0805 | no |
| post-asinh transform 9/9 | 2,266 | 1 | 1.0805 | 1.0805 | no |
| allocation beta | 2,266 | 0.9 | 1.07477 | 1.07477 | no |
| knot refit | 2,266 | 0.9 | 1.06538 | 1.06538 | no |
| covariance shape 1/5 | 2,266 | 0.9 | 1.06538 | 1.06538 | no |
| covariance shape 2/5 | 2,266 | 0.9 | 1.06538 | 1.06538 | no |
| covariance shape 3/5 | 2,266 | 0.9 | 1.06538 | 1.06538 | no |
| covariance shape 4/5 | 2,266 | 0.9 | 1.06538 | 1.06538 | no |
| covariance shape 5/5 | 2,266 | 0.9 | 1.06538 | 1.06538 | no |
| post-asinh transform 1/9 | 2,266 | 0.9 | 1.06538 | 1.06538 | no |
| post-asinh transform 2/9 | 2,266 | 0.9 | 1.06538 | 1.06538 | no |
| post-asinh transform 3/9 | 2,266 | 0.9 | 1.06538 | 1.06538 | no |
| post-asinh transform 4/9 | 2,266 | 0.9 | 1.06538 | 1.06538 | no |
| post-asinh transform 5/9 | 2,266 | 0.9 | 1.06538 | 1.06538 | no |
| post-asinh transform 6/9 | 2,266 | 0.9 | 1.06538 | 1.06538 | no |
| post-asinh transform 7/9 | 2,266 | 0.9 | 1.06538 | 1.06538 | no |
| post-asinh transform 8/9 | 2,266 | 0.9 | 1.06538 | 1.06538 | no |
| post-asinh transform 9/9 | 2,266 | 0.9 | 1.06538 | 1.06538 | no |
| allocation beta | 2,266 | 1 | 1.05549 | 1.05549 | no |
| knot refit | 2,266 | 1 | 1.05323 | 1.05323 | no |
| covariance shape 1/5 | 2,266 | 1 | 1.05323 | 1.05323 | no |
| covariance shape 2/5 | 2,266 | 1 | 1.05323 | 1.05323 | no |
| covariance shape 3/5 | 2,266 | 1 | 1.05323 | 1.05323 | no |
| covariance shape 4/5 | 2,266 | 1 | 1.05323 | 1.05323 | no |
| covariance shape 5/5 | 2,266 | 1 | 1.05323 | 1.05323 | no |
| post-asinh transform 1/9 | 2,266 | 1 | 1.05323 | 1.05323 | no |
| post-asinh transform 2/9 | 2,266 | 1 | 1.05323 | 1.05323 | no |
| post-asinh transform 3/9 | 2,266 | 1 | 1.05323 | 1.05323 | no |
| post-asinh transform 4/9 | 2,266 | 1 | 1.05323 | 1.05323 | no |
| post-asinh transform 5/9 | 2,266 | 1 | 1.05323 | 1.05323 | no |
| post-asinh transform 6/9 | 2,266 | 1 | 1.05323 | 1.05323 | no |
| post-asinh transform 7/9 | 2,266 | 1 | 1.05323 | 1.05323 | no |
| post-asinh transform 8/9 | 2,266 | 1 | 1.05323 | 1.05323 | no |
| post-asinh transform 9/9 | 2,266 | 1 | 1.05323 | 1.05323 | no |
| allocation beta | 2,266 | 1 | 1.05323 | 1.05323 | no |
| knot refit | 2,266 | 1 | 1.05098 | 1.05098 | no |
| covariance shape 1/5 | 2,266 | 1 | 1.05098 | 1.05098 | no |
| covariance shape 2/5 | 2,266 | 1 | 1.05098 | 1.05098 | no |
| covariance shape 3/5 | 2,266 | 1 | 1.05098 | 1.05098 | no |
| covariance shape 4/5 | 2,266 | 1 | 1.05098 | 1.05098 | no |
| covariance shape 5/5 | 2,266 | 1 | 1.05098 | 1.05098 | no |
| post-asinh transform 1/9 | 2,266 | 1 | 1.05098 | 1.05098 | no |
| post-asinh transform 2/9 | 2,266 | 1 | 1.05098 | 1.05098 | no |
| post-asinh transform 3/9 | 2,266 | 1 | 1.05098 | 1.05098 | no |
| post-asinh transform 4/9 | 2,266 | 1 | 1.05098 | 1.05098 | no |
| post-asinh transform 5/9 | 2,266 | 1 | 1.05098 | 1.05098 | no |
| post-asinh transform 6/9 | 2,266 | 1 | 1.05098 | 1.05098 | no |
| post-asinh transform 7/9 | 2,266 | 1 | 1.05098 | 1.05098 | no |
| post-asinh transform 8/9 | 2,266 | 1 | 1.05098 | 1.05098 | no |
| post-asinh transform 9/9 | 2,266 | 1 | 1.05098 | 1.05098 | no |
| allocation beta | 2,266 | 1 | 1.05098 | 1.05098 | no |
| knot refit | 2,266 | 1 | 1.04938 | 1.04938 | no |
| covariance shape 1/5 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| covariance shape 2/5 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| covariance shape 3/5 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| covariance shape 4/5 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| covariance shape 5/5 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| post-asinh transform 1/9 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| post-asinh transform 2/9 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| post-asinh transform 3/9 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| post-asinh transform 4/9 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| post-asinh transform 5/9 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| post-asinh transform 6/9 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| post-asinh transform 7/9 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| post-asinh transform 8/9 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| post-asinh transform 9/9 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| allocation beta | 2,266 | 1 | 1.04938 | 1.04938 | no |
| knot refit | 2,266 | 1 | 1.04938 | 1.04938 | no |
| refined parameter-search resolution | 2,266 | 1 | 1.04938 | 1.04938 | no |
| covariance shape 1/5 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| covariance shape 2/5 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| covariance shape 3/5 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| covariance shape 4/5 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| covariance shape 5/5 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| post-asinh transform 1/9 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| post-asinh transform 2/9 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| post-asinh transform 3/9 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| post-asinh transform 4/9 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| post-asinh transform 5/9 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| post-asinh transform 6/9 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| post-asinh transform 7/9 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| post-asinh transform 8/9 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| post-asinh transform 9/9 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| allocation beta | 2,266 | 1 | 1.04938 | 1.04938 | no |
| knot refit (cached stationary) | 2,266 | 1 | 1.04938 | 1.04938 | no |
| refined parameter-search resolution | 2,266 | 1 | 1.04938 | 1.04938 | no |
| covariance shape 1/5 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| covariance shape 2/5 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| covariance shape 3/5 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| covariance shape 4/5 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| covariance shape 5/5 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| post-asinh transform 1/9 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| post-asinh transform 2/9 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| post-asinh transform 3/9 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| post-asinh transform 4/9 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| post-asinh transform 5/9 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| post-asinh transform 6/9 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| post-asinh transform 7/9 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| post-asinh transform 8/9 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| post-asinh transform 9/9 | 2,266 | 1 | 1.04938 | 1.04938 | no |
| allocation beta | 2,266 | 1 | 1.04938 | 1.04938 | no |
| knot refit (cached stationary) | 2,266 | 1 | 1.04938 | 1.04938 | no |
| fixed point reached | 2,266 | 1 | 1.04938 | 1.04938 | no |
| prune proposal 2267->2266 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| covariance shape 1/5 | 2,266 | 1 | 1.01036 | 1.01036 | no |
| covariance shape 2/5 | 2,266 | 1 | 1.01036 | 1.01036 | no |
| covariance shape 3/5 | 2,266 | 1 | 1.01036 | 1.01036 | no |
| covariance shape 4/5 | 2,266 | 1 | 1.01036 | 1.01036 | no |
| covariance shape 5/5 | 2,266 | 1 | 1.01036 | 1.01036 | no |
| post-asinh transform 1/9 | 2,266 | 1 | 1.01036 | 1.01036 | no |
| post-asinh transform 2/9 | 2,266 | 1 | 1.01036 | 1.01036 | no |
| post-asinh transform 3/9 | 2,266 | 1 | 1.01036 | 1.01036 | no |
| post-asinh transform 4/9 | 2,266 | 1 | 1.01036 | 1.01036 | no |
| post-asinh transform 5/9 | 2,266 | 1 | 1.01036 | 1.01036 | no |
| post-asinh transform 6/9 | 2,266 | 1 | 1.01036 | 1.01036 | no |
| post-asinh transform 7/9 | 2,266 | 1 | 1.01036 | 1.01036 | no |
| post-asinh transform 8/9 | 2,266 | 1 | 1.01036 | 1.01036 | no |
| post-asinh transform 9/9 | 2,266 | 1 | 1.01036 | 1.01036 | no |
| allocation beta | 2,266 | 1 | 1.01036 | 1.01036 | no |
| knot refit | 2,266 | 1 | 1.0082 | 1.0082 | no |
| covariance shape 1/5 | 2,266 | 1 | 1.0082 | 1.0082 | no |
| covariance shape 2/5 | 2,266 | 1 | 1.0082 | 1.0082 | no |
| covariance shape 3/5 | 2,266 | 1 | 1.0082 | 1.0082 | no |
| covariance shape 4/5 | 2,266 | 1 | 1.0082 | 1.0082 | no |
| covariance shape 5/5 | 2,266 | 1 | 1.0082 | 1.0082 | no |
| post-asinh transform 1/9 | 2,266 | 1 | 1.0082 | 1.0082 | no |
| post-asinh transform 2/9 | 2,266 | 1 | 1.0082 | 1.0082 | no |
| post-asinh transform 3/9 | 2,266 | 1 | 1.0082 | 1.0082 | no |
| post-asinh transform 4/9 | 2,266 | 1 | 1.0082 | 1.0082 | no |
| post-asinh transform 5/9 | 2,266 | 1 | 1.0082 | 1.0082 | no |
| post-asinh transform 6/9 | 2,266 | 1 | 1.0082 | 1.0082 | no |
| post-asinh transform 7/9 | 2,266 | 1 | 1.0082 | 1.0082 | no |
| post-asinh transform 8/9 | 2,266 | 1 | 1.0082 | 1.0082 | no |
| post-asinh transform 9/9 | 2,266 | 1 | 1.0082 | 1.0082 | no |
| allocation beta | 2,266 | 1 | 1.0082 | 1.0082 | no |
| knot refit | 2,266 | 1 | 1.0053 | 1.0053 | no |
| covariance shape 1/5 | 2,266 | 1 | 1.0053 | 1.0053 | no |
| covariance shape 2/5 | 2,266 | 1 | 1.0053 | 1.0053 | no |
| covariance shape 3/5 | 2,266 | 1 | 1.0053 | 1.0053 | no |
| covariance shape 4/5 | 2,266 | 1 | 1.0053 | 1.0053 | no |
| covariance shape 5/5 | 2,266 | 1 | 1.0053 | 1.0053 | no |
| post-asinh transform 1/9 | 2,266 | 1 | 1.0053 | 1.0053 | no |
| post-asinh transform 2/9 | 2,266 | 1 | 1.0053 | 1.0053 | no |
| post-asinh transform 3/9 | 2,266 | 1 | 1.0053 | 1.0053 | no |
| post-asinh transform 4/9 | 2,266 | 1 | 1.0053 | 1.0053 | no |
| post-asinh transform 5/9 | 2,266 | 1 | 1.0053 | 1.0053 | no |
| post-asinh transform 6/9 | 2,266 | 1 | 1.0053 | 1.0053 | no |
| post-asinh transform 7/9 | 2,266 | 1 | 1.0053 | 1.0053 | no |
| post-asinh transform 8/9 | 2,266 | 1 | 1.0053 | 1.0053 | no |
| post-asinh transform 9/9 | 2,266 | 1 | 1.0053 | 1.0053 | no |
| allocation beta | 2,266 | 1 | 1.0053 | 1.0053 | no |
| knot refit | 2,266 | 1 | 1.00316 | 1.00316 | no |
| covariance shape 1/5 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| covariance shape 2/5 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| covariance shape 3/5 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| covariance shape 4/5 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| covariance shape 5/5 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| post-asinh transform 1/9 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| post-asinh transform 2/9 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| post-asinh transform 3/9 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| post-asinh transform 4/9 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| post-asinh transform 5/9 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| post-asinh transform 6/9 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| post-asinh transform 7/9 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| post-asinh transform 8/9 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| post-asinh transform 9/9 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| allocation beta | 2,266 | 1 | 1.00316 | 1.00316 | no |
| knot refit | 2,266 | 1 | 1.00316 | 1.00316 | no |
| refined parameter-search resolution | 2,266 | 1 | 1.00316 | 1.00316 | no |
| covariance shape 1/5 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| covariance shape 2/5 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| covariance shape 3/5 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| covariance shape 4/5 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| covariance shape 5/5 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| post-asinh transform 1/9 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| post-asinh transform 2/9 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| post-asinh transform 3/9 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| post-asinh transform 4/9 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| post-asinh transform 5/9 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| post-asinh transform 6/9 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| post-asinh transform 7/9 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| post-asinh transform 8/9 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| post-asinh transform 9/9 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| allocation beta | 2,266 | 1 | 1.00316 | 1.00316 | no |
| knot refit (cached stationary) | 2,266 | 1 | 1.00316 | 1.00316 | no |
| refined parameter-search resolution | 2,266 | 1 | 1.00316 | 1.00316 | no |
| covariance shape 1/5 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| covariance shape 2/5 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| covariance shape 3/5 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| covariance shape 4/5 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| covariance shape 5/5 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| post-asinh transform 1/9 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| post-asinh transform 2/9 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| post-asinh transform 3/9 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| post-asinh transform 4/9 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| post-asinh transform 5/9 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| post-asinh transform 6/9 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| post-asinh transform 7/9 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| post-asinh transform 8/9 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| post-asinh transform 9/9 | 2,266 | 1 | 1.00316 | 1.00316 | no |
| allocation beta | 2,266 | 1 | 1.00316 | 1.00316 | no |
| knot refit (cached stationary) | 2,266 | 1 | 1.00316 | 1.00316 | no |
| fixed point reached | 2,266 | 1 | 1.00316 | 1.00316 | no |
| rejected prune | 2,266 | 1 | 1.00316 | 1.00316 | no |
| high-fidelity warm 2267->1701 | 1,701 | 1 | 0.63478 | 0.63478 | yes |
| high-fidelity proposal 2267->1701 | 1,701 | 1 | 0.63478 | 0.63478 | yes |
| accepted high-fidelity prune | 1,701 | 1 | 0.63478 | 0.63478 | yes |
| high-fidelity warm 1701->1276 | 1,276 | 1 | 0.756085 | 0.756085 | yes |
| high-fidelity proposal 1701->1276 | 1,276 | 1 | 0.756085 | 0.756085 | yes |
| accepted high-fidelity prune | 1,276 | 1 | 0.756085 | 0.756085 | yes |
| high-fidelity warm 1276->957 | 957 | 1 | 1.25593 | 1.25593 | no |
| high-fidelity cold 1276->957 | 957 | 1 | 1.17369 | 1.17369 | no |
| high-fidelity proposal 1276->957 | 957 | 1 | 1.17369 | 1.17369 | no |
| rejected high-fidelity prune | 957 | 1 | 1.17369 | 1.17369 | no |
| high-fidelity warm 1276->1117 | 1,117 | 1 | 0.900466 | 0.900466 | yes |
| high-fidelity proposal 1276->1117 | 1,117 | 1 | 0.900466 | 0.900466 | yes |
| accepted high-fidelity prune | 1,117 | 1 | 0.900466 | 0.900466 | yes |
| high-fidelity warm 1117->958 | 958 | 1 | 1.0031 | 1.0031 | no |
| high-fidelity cold 1117->958 | 958 | 1 | 1.17358 | 1.17358 | no |
| high-fidelity proposal 1117->958 | 958 | 1 | 1.0031 | 1.0031 | no |
| rejected high-fidelity prune | 958 | 1 | 1.0031 | 1.0031 | no |
| high-fidelity warm 1117->1038 | 1,038 | 1 | 1.03968 | 1.03968 | no |
| high-fidelity cold 1117->1038 | 1,038 | 1 | 1.25455 | 1.25455 | no |
| high-fidelity proposal 1117->1038 | 1,038 | 1 | 1.03968 | 1.03968 | no |
| rejected high-fidelity prune | 1,038 | 1 | 1.03968 | 1.03968 | no |
| high-fidelity warm 1117->1078 | 1,078 | 1 | 0.902404 | 0.902404 | yes |
| high-fidelity proposal 1117->1078 | 1,078 | 1 | 0.902404 | 0.902404 | yes |
| accepted high-fidelity prune | 1,078 | 1 | 0.902404 | 0.902404 | yes |
| high-fidelity warm 1078->1039 | 1,039 | 1 | 1.00494 | 1.00494 | no |
| high-fidelity cold 1078->1039 | 1,039 | 1 | 0.946742 | 0.946742 | yes |
| high-fidelity proposal 1078->1039 | 1,039 | 1 | 0.946742 | 0.946742 | yes |
| accepted high-fidelity prune | 1,039 | 1 | 0.946742 | 0.946742 | yes |
| high-fidelity warm 1039->1000 | 1,000 | 1 | 1.13617 | 1.13617 | no |
| high-fidelity cold 1039->1000 | 1,000 | 1 | 1.06298 | 1.06298 | no |
| high-fidelity proposal 1039->1000 | 1,000 | 1 | 1.06298 | 1.06298 | no |
| rejected high-fidelity prune | 1,000 | 1 | 1.06298 | 1.06298 | no |
| high-fidelity warm 1039->1020 | 1,020 | 1 | 0.949733 | 0.949733 | yes |
| high-fidelity proposal 1039->1020 | 1,020 | 1 | 0.949733 | 0.949733 | yes |
| accepted high-fidelity prune | 1,020 | 1 | 0.949733 | 0.949733 | yes |
| high-fidelity warm 1020->1001 | 1,001 | 1 | 1.21254 | 1.21254 | no |
| high-fidelity cold 1020->1001 | 1,001 | 1 | 1.13905 | 1.13905 | no |
| high-fidelity proposal 1020->1001 | 1,001 | 1 | 1.13905 | 1.13905 | no |
| rejected high-fidelity prune | 1,001 | 1 | 1.13905 | 1.13905 | no |
| high-fidelity warm 1020->1011 | 1,011 | 1 | 1.05823 | 1.05823 | no |
| high-fidelity cold 1020->1011 | 1,011 | 1 | 0.973866 | 0.973866 | yes |
| high-fidelity proposal 1020->1011 | 1,011 | 1 | 0.973866 | 0.973866 | yes |
| accepted high-fidelity prune | 1,011 | 1 | 0.973866 | 0.973866 | yes |
| high-fidelity warm 1011->1002 | 1,002 | 1 | 1.00824 | 1.00824 | no |
| high-fidelity cold 1011->1002 | 1,002 | 1 | 1.11948 | 1.11948 | no |
| high-fidelity proposal 1011->1002 | 1,002 | 1 | 1.00824 | 1.00824 | no |
| rejected high-fidelity prune | 1,002 | 1 | 1.00824 | 1.00824 | no |
| high-fidelity warm 1011->1007 | 1,007 | 1 | 0.947343 | 0.947343 | yes |
| high-fidelity proposal 1011->1007 | 1,007 | 1 | 0.947343 | 0.947343 | yes |
| accepted high-fidelity prune | 1,007 | 1 | 0.947343 | 0.947343 | yes |
| high-fidelity warm 1007->1003 | 1,003 | 1 | 1.0255 | 1.0255 | no |
| high-fidelity cold 1007->1003 | 1,003 | 1 | 0.945247 | 0.945247 | yes |
| high-fidelity proposal 1007->1003 | 1,003 | 1 | 0.945247 | 0.945247 | yes |
| accepted high-fidelity prune | 1,003 | 1 | 0.945247 | 0.945247 | yes |
| high-fidelity warm 1003->999 | 999 | 1 | 1.24903 | 1.24903 | no |
| high-fidelity cold 1003->999 | 999 | 1 | 1.08189 | 1.08189 | no |
| high-fidelity proposal 1003->999 | 999 | 1 | 1.08189 | 1.08189 | no |
| rejected high-fidelity prune | 999 | 1 | 1.08189 | 1.08189 | no |
| high-fidelity warm 1003->1001 | 1,001 | 1 | 1.03977 | 1.03977 | no |
| high-fidelity cold 1003->1001 | 1,001 | 1 | 0.973345 | 0.973345 | yes |
| high-fidelity proposal 1003->1001 | 1,001 | 1 | 0.973345 | 0.973345 | yes |
| accepted high-fidelity prune | 1,001 | 1 | 0.973345 | 0.973345 | yes |
| high-fidelity warm 1001->999 | 999 | 1 | 1.10415 | 1.10415 | no |
| high-fidelity cold 1001->999 | 999 | 1 | 0.980295 | 0.980295 | yes |
| high-fidelity proposal 1001->999 | 999 | 1 | 0.980295 | 0.980295 | yes |
| accepted high-fidelity prune | 999 | 1 | 0.980295 | 0.980295 | yes |
| high-fidelity warm 999->997 | 997 | 1 | 1.04943 | 1.04943 | no |
| high-fidelity cold 999->997 | 997 | 1 | 0.956228 | 0.956228 | yes |
| high-fidelity proposal 999->997 | 997 | 1 | 0.956228 | 0.956228 | yes |
| accepted high-fidelity prune | 997 | 1 | 0.956228 | 0.956228 | yes |
| high-fidelity warm 997->995 | 995 | 1 | 1.11429 | 1.11429 | no |
| high-fidelity cold 997->995 | 995 | 1 | 1.119 | 1.119 | no |
| high-fidelity proposal 997->995 | 995 | 1 | 1.11429 | 1.11429 | no |
| rejected high-fidelity prune | 995 | 1 | 1.11429 | 1.11429 | no |
| high-fidelity warm 997->996 | 996 | 1 | 1.11385 | 1.11385 | no |
| high-fidelity cold 997->996 | 996 | 1 | 1.00493 | 1.00493 | no |
| high-fidelity proposal 997->996 | 996 | 1 | 1.00493 | 1.00493 | no |
| rejected high-fidelity prune | 996 | 1 | 1.00493 | 1.00493 | no |
| high-fidelity warm 997->748 | 748 | 1 | 1.42176 | 1.42176 | no |
| high-fidelity cold 1/4 997->748 | 748 | 1 | 1.40403 | 1.40403 | no |
| high-fidelity cold 2/4 997->748 | 748 | 1 | 1.46458 | 1.46458 | no |
| high-fidelity cold 3/4 997->748 | 748 | 1 | 1.47593 | 1.47593 | no |
| high-fidelity cold 4/4 997->748 | 748 | 1 | 1.48315 | 1.48315 | no |
| high-fidelity proposal 997->748 | 748 | 1 | 1.40403 | 1.40403 | no |
| rejected high-fidelity prune | 748 | 1 | 1.40403 | 1.40403 | no |
| high-fidelity warm 997->873 | 873 | 1 | 1.17143 | 1.17143 | no |
| high-fidelity cold 1/4 997->873 | 873 | 1 | 1.19969 | 1.19969 | no |
| high-fidelity cold 2/4 997->873 | 873 | 1 | 1.27151 | 1.27151 | no |
| high-fidelity cold 3/4 997->873 | 873 | 1 | 1.08037 | 1.08037 | no |
| high-fidelity cold 4/4 997->873 | 873 | 1 | 1.27389 | 1.27389 | no |
| high-fidelity proposal 997->873 | 873 | 1 | 1.08037 | 1.08037 | no |
| rejected high-fidelity prune | 873 | 1 | 1.08037 | 1.08037 | no |
| high-fidelity warm 997->935 | 935 | 1 | 1.14466 | 1.14466 | no |
| high-fidelity cold 1/4 997->935 | 935 | 1 | 1.14975 | 1.14975 | no |
| high-fidelity cold 2/4 997->935 | 935 | 1 | 1.17484 | 1.17484 | no |
| high-fidelity cold 3/4 997->935 | 935 | 1 | 1.1352 | 1.1352 | no |
| high-fidelity cold 4/4 997->935 | 935 | 1 | 1.0588 | 1.0588 | no |
| high-fidelity proposal 997->935 | 935 | 1 | 1.0588 | 1.0588 | no |
| rejected high-fidelity prune | 935 | 1 | 1.0588 | 1.0588 | no |
| high-fidelity warm 997->966 | 966 | 1 | 1.23029 | 1.23029 | no |
| high-fidelity cold 1/4 997->966 | 966 | 1 | 1.1199 | 1.1199 | no |
| high-fidelity cold 2/4 997->966 | 966 | 1 | 1.19653 | 1.19653 | no |
| high-fidelity cold 3/4 997->966 | 966 | 1 | 1.11871 | 1.11871 | no |
| high-fidelity cold 4/4 997->966 | 966 | 1 | 1.11349 | 1.11349 | no |
| high-fidelity proposal 997->966 | 966 | 1 | 1.11349 | 1.11349 | no |
| rejected high-fidelity prune | 966 | 1 | 1.11349 | 1.11349 | no |
| high-fidelity warm 997->982 | 982 | 1 | 0.944542 | 0.944542 | yes |
| high-fidelity proposal 997->982 | 982 | 1 | 0.944542 | 0.944542 | yes |
| accepted high-fidelity prune | 982 | 1 | 0.944542 | 0.944542 | yes |
| high-fidelity warm 982->967 | 967 | 1 | 1.16705 | 1.16705 | no |
| high-fidelity cold 1/4 982->967 | 967 | 1 | 1.14359 | 1.14359 | no |
| high-fidelity cold 2/4 982->967 | 967 | 1 | 0.978443 | 0.978443 | yes |
| high-fidelity cold 3/4 982->967 | 967 | 1 | 0.988921 | 0.988921 | yes |
| high-fidelity cold 4/4 982->967 | 967 | 1 | 1.13017 | 1.13017 | no |
| high-fidelity proposal 982->967 | 967 | 1 | 0.978443 | 0.978443 | yes |
| accepted high-fidelity prune | 967 | 1 | 0.978443 | 0.978443 | yes |
| high-fidelity warm 967->952 | 952 | 1 | 1.10901 | 1.10901 | no |
| high-fidelity cold 1/4 967->952 | 952 | 1 | 1.01912 | 1.01912 | no |
| high-fidelity cold 2/4 967->952 | 952 | 1 | 1.11507 | 1.11507 | no |
| high-fidelity cold 3/4 967->952 | 952 | 1 | 1.03797 | 1.03797 | no |
| high-fidelity cold 4/4 967->952 | 952 | 1 | 1.00351 | 1.00351 | no |
| high-fidelity proposal 967->952 | 952 | 1 | 1.00351 | 1.00351 | no |
| rejected high-fidelity prune | 952 | 1 | 1.00351 | 1.00351 | no |
| high-fidelity warm 967->960 | 960 | 1 | 1.25696 | 1.25696 | no |
| high-fidelity cold 1/4 967->960 | 960 | 1 | 1.01756 | 1.01756 | no |
| high-fidelity cold 2/4 967->960 | 960 | 1 | 1.13019 | 1.13019 | no |
| high-fidelity cold 3/4 967->960 | 960 | 1 | 1.0225 | 1.0225 | no |
| high-fidelity cold 4/4 967->960 | 960 | 1 | 1.11408 | 1.11408 | no |
| high-fidelity proposal 967->960 | 960 | 1 | 1.01756 | 1.01756 | no |
| rejected high-fidelity prune | 960 | 1 | 1.01756 | 1.01756 | no |
| high-fidelity warm 967->964 | 964 | 1 | 1.00837 | 1.00837 | no |
| high-fidelity cold 1/4 967->964 | 964 | 1 | 1.11172 | 1.11172 | no |
| high-fidelity cold 2/4 967->964 | 964 | 1 | 0.969061 | 0.969061 | yes |
| high-fidelity cold 3/4 967->964 | 964 | 1 | 1.09646 | 1.09646 | no |
| high-fidelity cold 4/4 967->964 | 964 | 1 | 1.12077 | 1.12077 | no |
| high-fidelity proposal 967->964 | 964 | 1 | 0.969061 | 0.969061 | yes |
| accepted high-fidelity prune | 964 | 1 | 0.969061 | 0.969061 | yes |
| high-fidelity warm 964->961 | 961 | 1 | 1.10963 | 1.10963 | no |
| high-fidelity cold 1/4 964->961 | 961 | 1 | 1.1746 | 1.1746 | no |
| high-fidelity cold 2/4 964->961 | 961 | 1 | 1.03787 | 1.03787 | no |
| high-fidelity cold 3/4 964->961 | 961 | 1 | 1.15322 | 1.15322 | no |
| high-fidelity cold 4/4 964->961 | 961 | 1 | 1.16336 | 1.16336 | no |
| high-fidelity proposal 964->961 | 961 | 1 | 1.03787 | 1.03787 | no |
| rejected high-fidelity prune | 961 | 1 | 1.03787 | 1.03787 | no |
| high-fidelity warm 964->963 | 963 | 1 | 0.938584 | 0.938584 | yes |
| high-fidelity proposal 964->963 | 963 | 1 | 0.938584 | 0.938584 | yes |
| accepted high-fidelity prune | 963 | 1 | 0.938584 | 0.938584 | yes |
| high-fidelity warm 963->962 | 962 | 1 | 1.05592 | 1.05592 | no |
| high-fidelity cold 1/4 963->962 | 962 | 1 | 1.12285 | 1.12285 | no |
| high-fidelity cold 2/4 963->962 | 962 | 1 | 0.967314 | 0.967314 | yes |
| high-fidelity cold 3/4 963->962 | 962 | 1 | 1.1627 | 1.1627 | no |
| high-fidelity cold 4/4 963->962 | 962 | 1 | 1.06621 | 1.06621 | no |
| high-fidelity proposal 963->962 | 962 | 1 | 0.967314 | 0.967314 | yes |
| accepted high-fidelity prune | 962 | 1 | 0.967314 | 0.967314 | yes |
| high-fidelity warm 962->961 | 961 | 1 | 1.10118 | 1.10118 | no |
| high-fidelity cold 1/4 962->961 | 961 | 1 | 0.95985 | 0.95985 | yes |
| high-fidelity cold 2/4 962->961 | 961 | 1 | 1.18326 | 1.18326 | no |
| high-fidelity cold 3/4 962->961 | 961 | 1 | 1.01804 | 1.01804 | no |
| high-fidelity cold 4/4 962->961 | 961 | 1 | 1.07133 | 1.07133 | no |
| high-fidelity proposal 962->961 | 961 | 1 | 0.95985 | 0.95985 | yes |
| accepted high-fidelity prune | 961 | 1 | 0.95985 | 0.95985 | yes |
| high-fidelity warm 961->960 | 960 | 1 | 1.06054 | 1.06054 | no |
| high-fidelity cold 1/4 961->960 | 960 | 1 | 1.09275 | 1.09275 | no |
| high-fidelity cold 2/4 961->960 | 960 | 1 | 1.02225 | 1.02225 | no |
| high-fidelity cold 3/4 961->960 | 960 | 1 | 1.00946 | 1.00946 | no |
| high-fidelity cold 4/4 961->960 | 960 | 1 | 1.07369 | 1.07369 | no |
| high-fidelity proposal 961->960 | 960 | 1 | 1.00946 | 1.00946 | no |
| rejected high-fidelity prune | 960 | 1 | 1.00946 | 1.00946 | no |

Covariance adjustment from empirical whitening: `[[1.0178348443250573,0.0,0.0],[0.0,1.0088780126085894,0.0],[0.0,0.0,0.973831971293259]]`

Post-asinh matrix: `[[3.0515608712881273,0.0,0.0],[0.0,3.125261859311369,0.0],[0.0,0.0,3.051560871288127]]`

## Reproduction

```text
node scripts/run-ml-python.mjs ml/joint_optimize_multidimensional_return_knots.py
```
