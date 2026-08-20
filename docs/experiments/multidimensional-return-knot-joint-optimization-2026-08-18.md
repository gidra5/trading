# Joint multidimensional return-knot optimization

Generated 2026-08-18T05:33:09.759829Z.

The r2|r1 conditional-mean RMSE is disabled as an optimization loss and acceptance criterion, but remains in every diagnostic table. All other density and conditional operation thresholds remain active.

## 2D

Initial: 3,072 points, beta 0.875, active score 1, joint objective 0.605085.

Final: **618 points**, beta **1**, active score **1**, joint objective **0.78735**, passes.

### Fixed-point checks

| points | sweeps | converged | covariance step | transform step | beta step |
|---:|---:|---:|---:|---:|---:|
| 3,072 | 6 | yes | 0.0175 | 0.015 | 0.03125 |
| 2,304 | 8 | yes | 0.0175 | 0.015 | 0.03125 |
| 1,728 | 6 | yes | 0.0175 | 0.015 | 0.03125 |
| 1,296 | 5 | yes | 0.0175 | 0.015 | 0.03125 |
| 972 | 8 | yes | 0.0175 | 0.015 | 0.03125 |
| 729 | 6 | yes | 0.0175 | 0.015 | 0.03125 |
| 547 | 12 | yes | 0.0175 | 0.015 | 0.03125 |
| 547 | 9 | yes | 0.0175 | 0.015 | 0.03125 |
| 638 | 5 | yes | 0.0175 | 0.015 | 0.03125 |
| 638 | 5 | yes | 0.0175 | 0.015 | 0.03125 |
| 684 | 5 | yes | 0.0175 | 0.015 | 0.03125 |
| 639 | 5 | yes | 0.0175 | 0.015 | 0.03125 |
| 639 | 6 | yes | 0.0175 | 0.015 | 0.03125 |
| 662 | 4 | yes | 0.0175 | 0.015 | 0.03125 |
| 640 | 4 | yes | 0.0175 | 0.015 | 0.03125 |
| 618 | 6 | yes | 0.0175 | 0.015 | 0.03125 |
| 618 | 6 | yes | 0.0175 | 0.015 | 0.03125 |
| 629 | 4 | yes | 0.0175 | 0.015 | 0.03125 |
| 618 | 5 | yes | 0.0175 | 0.015 | 0.03125 |

| metric | ratio to 1D32 | active | passes |
|---|---:|---:|---:|
| density JS/d | 0.858612 | yes | yes |
| r2 conditional mean | 22.3148 | no | no |
| r2 conditional median | 0.0520239 | yes | yes |

### Accepted/rejected iteration trace

| stage | points | beta | objective | active score | passes |
|---|---:|---:|---:|---:|---:|
| stored verified initial | 3,072 | 0.875 | 0.605085 | 1 | yes |
| common-fidelity warm refit | 3,072 | 0.875 | 0.155152 | 1 | yes |
| covariance shape 1/2 | 3,072 | 0.875 | 0.155152 | 1 | yes |
| covariance shape 2/2 | 3,072 | 0.875 | 0.155152 | 1 | yes |
| post-asinh transform 1/4 | 3,072 | 0.875 | 0.155152 | 1 | yes |
| post-asinh transform 2/4 | 3,072 | 0.875 | 0.155152 | 1 | yes |
| post-asinh transform 3/4 | 3,072 | 0.875 | 0.155152 | 1 | yes |
| post-asinh transform 4/4 | 3,072 | 0.875 | 0.155152 | 1 | yes |
| allocation beta | 3,072 | 1 | 0.141067 | 1 | yes |
| knot refit | 3,072 | 1 | 0.141067 | 1 | yes |
| covariance shape 1/2 | 3,072 | 1 | 0.141067 | 1 | yes |
| covariance shape 2/2 | 3,072 | 1 | 0.141067 | 1 | yes |
| post-asinh transform 1/4 | 3,072 | 1 | 0.141067 | 1 | yes |
| post-asinh transform 2/4 | 3,072 | 1 | 0.141067 | 1 | yes |
| post-asinh transform 3/4 | 3,072 | 1 | 0.141067 | 1 | yes |
| post-asinh transform 4/4 | 3,072 | 1 | 0.141067 | 1 | yes |
| allocation beta | 3,072 | 0.875 | 0.138245 | 1 | yes |
| knot refit | 3,072 | 0.875 | 0.138245 | 1 | yes |
| covariance shape 1/2 | 3,072 | 0.875 | 0.138245 | 1 | yes |
| covariance shape 2/2 | 3,072 | 0.875 | 0.138245 | 1 | yes |
| post-asinh transform 1/4 | 3,072 | 0.875 | 0.138245 | 1 | yes |
| post-asinh transform 2/4 | 3,072 | 0.875 | 0.138245 | 1 | yes |
| post-asinh transform 3/4 | 3,072 | 0.875 | 0.138245 | 1 | yes |
| post-asinh transform 4/4 | 3,072 | 0.875 | 0.138245 | 1 | yes |
| allocation beta | 3,072 | 0.875 | 0.138245 | 1 | yes |
| knot refit | 3,072 | 0.875 | 0.138245 | 1 | yes |
| refined parameter-search resolution | 3,072 | 0.875 | 0.138245 | 1 | yes |
| covariance shape 1/2 | 3,072 | 0.875 | 0.138245 | 1 | yes |
| covariance shape 2/2 | 3,072 | 0.875 | 0.138245 | 1 | yes |
| post-asinh transform 1/4 | 3,072 | 0.875 | 0.138245 | 1 | yes |
| post-asinh transform 2/4 | 3,072 | 0.875 | 0.138245 | 1 | yes |
| post-asinh transform 3/4 | 3,072 | 0.875 | 0.138245 | 1 | yes |
| post-asinh transform 4/4 | 3,072 | 0.875 | 0.138245 | 1 | yes |
| allocation beta | 3,072 | 0.9375 | 0.134424 | 1 | yes |
| knot refit | 3,072 | 0.9375 | 0.134424 | 1 | yes |
| covariance shape 1/2 | 3,072 | 0.9375 | 0.134424 | 1 | yes |
| covariance shape 2/2 | 3,072 | 0.9375 | 0.134424 | 1 | yes |
| post-asinh transform 1/4 | 3,072 | 0.9375 | 0.134424 | 1 | yes |
| post-asinh transform 2/4 | 3,072 | 0.9375 | 0.134424 | 1 | yes |
| post-asinh transform 3/4 | 3,072 | 0.9375 | 0.134424 | 1 | yes |
| post-asinh transform 4/4 | 3,072 | 0.9375 | 0.134424 | 1 | yes |
| allocation beta | 3,072 | 0.9375 | 0.134424 | 1 | yes |
| knot refit | 3,072 | 0.9375 | 0.134424 | 1 | yes |
| refined parameter-search resolution | 3,072 | 0.9375 | 0.134424 | 1 | yes |
| covariance shape 1/2 | 3,072 | 0.9375 | 0.134424 | 1 | yes |
| covariance shape 2/2 | 3,072 | 0.9375 | 0.134424 | 1 | yes |
| post-asinh transform 1/4 | 3,072 | 0.9375 | 0.134424 | 1 | yes |
| post-asinh transform 2/4 | 3,072 | 0.9375 | 0.134424 | 1 | yes |
| post-asinh transform 3/4 | 3,072 | 0.9375 | 0.134424 | 1 | yes |
| post-asinh transform 4/4 | 3,072 | 0.9375 | 0.134424 | 1 | yes |
| allocation beta | 3,072 | 0.9375 | 0.134424 | 1 | yes |
| knot refit | 3,072 | 0.9375 | 0.134424 | 1 | yes |
| fixed point reached | 3,072 | 0.9375 | 0.134424 | 1 | yes |
| prune proposal 3072->2304 | 2,304 | 0.96875 | 0.203416 | 1 | yes |
| covariance shape 1/2 | 2,304 | 0.9375 | 0.250624 | 1 | yes |
| covariance shape 2/2 | 2,304 | 0.9375 | 0.250624 | 1 | yes |
| post-asinh transform 1/4 | 2,304 | 0.9375 | 0.250624 | 1 | yes |
| post-asinh transform 2/4 | 2,304 | 0.9375 | 0.250624 | 1 | yes |
| post-asinh transform 3/4 | 2,304 | 0.9375 | 0.250624 | 1 | yes |
| post-asinh transform 4/4 | 2,304 | 0.9375 | 0.250624 | 1 | yes |
| allocation beta | 2,304 | 0.8125 | 0.232601 | 1 | yes |
| knot refit | 2,304 | 0.8125 | 0.231483 | 1 | yes |
| covariance shape 1/2 | 2,304 | 0.8125 | 0.231483 | 1 | yes |
| covariance shape 2/2 | 2,304 | 0.8125 | 0.231483 | 1 | yes |
| post-asinh transform 1/4 | 2,304 | 0.8125 | 0.231483 | 1 | yes |
| post-asinh transform 2/4 | 2,304 | 0.8125 | 0.231483 | 1 | yes |
| post-asinh transform 3/4 | 2,304 | 0.8125 | 0.231483 | 1 | yes |
| post-asinh transform 4/4 | 2,304 | 0.8125 | 0.231483 | 1 | yes |
| allocation beta | 2,304 | 0.9375 | 0.223073 | 1 | yes |
| knot refit | 2,304 | 0.9375 | 0.216441 | 1 | yes |
| covariance shape 1/2 | 2,304 | 0.9375 | 0.216441 | 1 | yes |
| covariance shape 2/2 | 2,304 | 0.9375 | 0.216441 | 1 | yes |
| post-asinh transform 1/4 | 2,304 | 0.9375 | 0.216441 | 1 | yes |
| post-asinh transform 2/4 | 2,304 | 0.9375 | 0.216441 | 1 | yes |
| post-asinh transform 3/4 | 2,304 | 0.9375 | 0.216441 | 1 | yes |
| post-asinh transform 4/4 | 2,304 | 0.9375 | 0.216441 | 1 | yes |
| allocation beta | 2,304 | 0.8125 | 0.213159 | 1 | yes |
| knot refit | 2,304 | 0.8125 | 0.212854 | 1 | yes |
| covariance shape 1/2 | 2,304 | 0.8125 | 0.212854 | 1 | yes |
| covariance shape 2/2 | 2,304 | 0.8125 | 0.212854 | 1 | yes |
| post-asinh transform 1/4 | 2,304 | 0.8125 | 0.212854 | 1 | yes |
| post-asinh transform 2/4 | 2,304 | 0.8125 | 0.212854 | 1 | yes |
| post-asinh transform 3/4 | 2,304 | 0.8125 | 0.212854 | 1 | yes |
| post-asinh transform 4/4 | 2,304 | 0.8125 | 0.212854 | 1 | yes |
| allocation beta | 2,304 | 0.9375 | 0.210589 | 1 | yes |
| knot refit | 2,304 | 0.9375 | 0.208003 | 1 | yes |
| covariance shape 1/2 | 2,304 | 0.9375 | 0.208003 | 1 | yes |
| covariance shape 2/2 | 2,304 | 0.9375 | 0.208003 | 1 | yes |
| post-asinh transform 1/4 | 2,304 | 0.9375 | 0.208003 | 1 | yes |
| post-asinh transform 2/4 | 2,304 | 0.9375 | 0.208003 | 1 | yes |
| post-asinh transform 3/4 | 2,304 | 0.9375 | 0.208003 | 1 | yes |
| post-asinh transform 4/4 | 2,304 | 0.9375 | 0.208003 | 1 | yes |
| allocation beta | 2,304 | 0.9375 | 0.208003 | 1 | yes |
| knot refit | 2,304 | 0.9375 | 0.208003 | 1 | yes |
| refined parameter-search resolution | 2,304 | 0.9375 | 0.208003 | 1 | yes |
| covariance shape 1/2 | 2,304 | 0.9375 | 0.208003 | 1 | yes |
| covariance shape 2/2 | 2,304 | 0.9375 | 0.208003 | 1 | yes |
| post-asinh transform 1/4 | 2,304 | 0.9375 | 0.208003 | 1 | yes |
| post-asinh transform 2/4 | 2,304 | 0.9375 | 0.208003 | 1 | yes |
| post-asinh transform 3/4 | 2,304 | 0.9375 | 0.208003 | 1 | yes |
| post-asinh transform 4/4 | 2,304 | 0.9375 | 0.208003 | 1 | yes |
| allocation beta | 2,304 | 0.9375 | 0.208003 | 1 | yes |
| knot refit | 2,304 | 0.9375 | 0.208003 | 1 | yes |
| refined parameter-search resolution | 2,304 | 0.9375 | 0.208003 | 1 | yes |
| covariance shape 1/2 | 2,304 | 0.9375 | 0.208003 | 1 | yes |
| covariance shape 2/2 | 2,304 | 0.9375 | 0.208003 | 1 | yes |
| post-asinh transform 1/4 | 2,304 | 0.9375 | 0.208003 | 1 | yes |
| post-asinh transform 2/4 | 2,304 | 0.9375 | 0.208003 | 1 | yes |
| post-asinh transform 3/4 | 2,304 | 0.9375 | 0.208003 | 1 | yes |
| post-asinh transform 4/4 | 2,304 | 0.9375 | 0.208003 | 1 | yes |
| allocation beta | 2,304 | 0.96875 | 0.203416 | 1 | yes |
| knot refit | 2,304 | 0.96875 | 0.203416 | 1 | yes |
| covariance shape 1/2 | 2,304 | 0.96875 | 0.203416 | 1 | yes |
| covariance shape 2/2 | 2,304 | 0.96875 | 0.203416 | 1 | yes |
| post-asinh transform 1/4 | 2,304 | 0.96875 | 0.203416 | 1 | yes |
| post-asinh transform 2/4 | 2,304 | 0.96875 | 0.203416 | 1 | yes |
| post-asinh transform 3/4 | 2,304 | 0.96875 | 0.203416 | 1 | yes |
| post-asinh transform 4/4 | 2,304 | 0.96875 | 0.203416 | 1 | yes |
| allocation beta | 2,304 | 0.96875 | 0.203416 | 1 | yes |
| knot refit | 2,304 | 0.96875 | 0.203416 | 1 | yes |
| fixed point reached | 2,304 | 0.96875 | 0.203416 | 1 | yes |
| accepted prune | 2,304 | 0.96875 | 0.203416 | 1 | yes |
| prune proposal 2304->1728 | 1,728 | 0.75 | 0.257983 | 1 | yes |
| covariance shape 1/2 | 1,728 | 0.96875 | 0.2868 | 1 | yes |
| covariance shape 2/2 | 1,728 | 0.96875 | 0.2868 | 1 | yes |
| post-asinh transform 1/4 | 1,728 | 0.96875 | 0.2868 | 1 | yes |
| post-asinh transform 2/4 | 1,728 | 0.96875 | 0.2868 | 1 | yes |
| post-asinh transform 3/4 | 1,728 | 0.96875 | 0.2868 | 1 | yes |
| post-asinh transform 4/4 | 1,728 | 0.96875 | 0.2868 | 1 | yes |
| allocation beta | 1,728 | 0.84375 | 0.268207 | 1 | yes |
| knot refit | 1,728 | 0.84375 | 0.268207 | 1 | yes |
| covariance shape 1/2 | 1,728 | 0.84375 | 0.268207 | 1 | yes |
| covariance shape 2/2 | 1,728 | 0.84375 | 0.268207 | 1 | yes |
| post-asinh transform 1/4 | 1,728 | 0.84375 | 0.268207 | 1 | yes |
| post-asinh transform 2/4 | 1,728 | 0.84375 | 0.268207 | 1 | yes |
| post-asinh transform 3/4 | 1,728 | 0.84375 | 0.268207 | 1 | yes |
| post-asinh transform 4/4 | 1,728 | 0.84375 | 0.268207 | 1 | yes |
| allocation beta | 1,728 | 0.71875 | 0.264153 | 1 | yes |
| knot refit | 1,728 | 0.71875 | 0.264153 | 1 | yes |
| covariance shape 1/2 | 1,728 | 0.71875 | 0.264153 | 1 | yes |
| covariance shape 2/2 | 1,728 | 0.71875 | 0.264153 | 1 | yes |
| post-asinh transform 1/4 | 1,728 | 0.71875 | 0.264153 | 1 | yes |
| post-asinh transform 2/4 | 1,728 | 0.71875 | 0.264153 | 1 | yes |
| post-asinh transform 3/4 | 1,728 | 0.71875 | 0.264153 | 1 | yes |
| post-asinh transform 4/4 | 1,728 | 0.71875 | 0.264153 | 1 | yes |
| allocation beta | 1,728 | 0.71875 | 0.264153 | 1 | yes |
| knot refit | 1,728 | 0.71875 | 0.264153 | 1 | yes |
| refined parameter-search resolution | 1,728 | 0.71875 | 0.264153 | 1 | yes |
| covariance shape 1/2 | 1,728 | 0.71875 | 0.264153 | 1 | yes |
| covariance shape 2/2 | 1,728 | 0.71875 | 0.264153 | 1 | yes |
| post-asinh transform 1/4 | 1,728 | 0.71875 | 0.264153 | 1 | yes |
| post-asinh transform 2/4 | 1,728 | 0.71875 | 0.264153 | 1 | yes |
| post-asinh transform 3/4 | 1,728 | 0.71875 | 0.264153 | 1 | yes |
| post-asinh transform 4/4 | 1,728 | 0.71875 | 0.264153 | 1 | yes |
| allocation beta | 1,728 | 0.71875 | 0.264153 | 1 | yes |
| knot refit | 1,728 | 0.71875 | 0.264153 | 1 | yes |
| refined parameter-search resolution | 1,728 | 0.71875 | 0.264153 | 1 | yes |
| covariance shape 1/2 | 1,728 | 0.71875 | 0.264153 | 1 | yes |
| covariance shape 2/2 | 1,728 | 0.71875 | 0.264153 | 1 | yes |
| post-asinh transform 1/4 | 1,728 | 0.71875 | 0.264153 | 1 | yes |
| post-asinh transform 2/4 | 1,728 | 0.71875 | 0.264153 | 1 | yes |
| post-asinh transform 3/4 | 1,728 | 0.71875 | 0.264153 | 1 | yes |
| post-asinh transform 4/4 | 1,728 | 0.71875 | 0.264153 | 1 | yes |
| allocation beta | 1,728 | 0.75 | 0.257983 | 1 | yes |
| knot refit | 1,728 | 0.75 | 0.257983 | 1 | yes |
| covariance shape 1/2 | 1,728 | 0.75 | 0.257983 | 1 | yes |
| covariance shape 2/2 | 1,728 | 0.75 | 0.257983 | 1 | yes |
| post-asinh transform 1/4 | 1,728 | 0.75 | 0.257983 | 1 | yes |
| post-asinh transform 2/4 | 1,728 | 0.75 | 0.257983 | 1 | yes |
| post-asinh transform 3/4 | 1,728 | 0.75 | 0.257983 | 1 | yes |
| post-asinh transform 4/4 | 1,728 | 0.75 | 0.257983 | 1 | yes |
| allocation beta | 1,728 | 0.75 | 0.257983 | 1 | yes |
| knot refit | 1,728 | 0.75 | 0.257983 | 1 | yes |
| fixed point reached | 1,728 | 0.75 | 0.257983 | 1 | yes |
| accepted prune | 1,728 | 0.75 | 0.257983 | 1 | yes |
| prune proposal 1728->1296 | 1,296 | 1 | 0.314331 | 1 | yes |
| covariance shape 1/2 | 1,296 | 0.75 | 1.0943 | 1.11377 | no |
| covariance shape 2/2 | 1,296 | 0.75 | 1.0943 | 1.11377 | no |
| post-asinh transform 1/4 | 1,296 | 0.75 | 1.0943 | 1.11377 | no |
| post-asinh transform 2/4 | 1,296 | 0.75 | 1.0943 | 1.11377 | no |
| post-asinh transform 3/4 | 1,296 | 0.75 | 1.0943 | 1.11377 | no |
| post-asinh transform 4/4 | 1,296 | 0.75 | 1.0943 | 1.11377 | no |
| allocation beta | 1,296 | 0.875 | 0.681776 | 1 | yes |
| knot refit | 1,296 | 0.875 | 0.646185 | 1 | yes |
| covariance shape 1/2 | 1,296 | 0.875 | 0.646185 | 1 | yes |
| covariance shape 2/2 | 1,296 | 0.875 | 0.646185 | 1 | yes |
| post-asinh transform 1/4 | 1,296 | 0.875 | 0.646185 | 1 | yes |
| post-asinh transform 2/4 | 1,296 | 0.875 | 0.646185 | 1 | yes |
| post-asinh transform 3/4 | 1,296 | 0.875 | 0.646185 | 1 | yes |
| post-asinh transform 4/4 | 1,296 | 0.875 | 0.646185 | 1 | yes |
| allocation beta | 1,296 | 1 | 0.419655 | 1 | yes |
| knot refit | 1,296 | 1 | 0.314331 | 1 | yes |
| covariance shape 1/2 | 1,296 | 1 | 0.314331 | 1 | yes |
| covariance shape 2/2 | 1,296 | 1 | 0.314331 | 1 | yes |
| post-asinh transform 1/4 | 1,296 | 1 | 0.314331 | 1 | yes |
| post-asinh transform 2/4 | 1,296 | 1 | 0.314331 | 1 | yes |
| post-asinh transform 3/4 | 1,296 | 1 | 0.314331 | 1 | yes |
| post-asinh transform 4/4 | 1,296 | 1 | 0.314331 | 1 | yes |
| allocation beta | 1,296 | 1 | 0.314331 | 1 | yes |
| knot refit | 1,296 | 1 | 0.314331 | 1 | yes |
| refined parameter-search resolution | 1,296 | 1 | 0.314331 | 1 | yes |
| covariance shape 1/2 | 1,296 | 1 | 0.314331 | 1 | yes |
| covariance shape 2/2 | 1,296 | 1 | 0.314331 | 1 | yes |
| post-asinh transform 1/4 | 1,296 | 1 | 0.314331 | 1 | yes |
| post-asinh transform 2/4 | 1,296 | 1 | 0.314331 | 1 | yes |
| post-asinh transform 3/4 | 1,296 | 1 | 0.314331 | 1 | yes |
| post-asinh transform 4/4 | 1,296 | 1 | 0.314331 | 1 | yes |
| allocation beta | 1,296 | 1 | 0.314331 | 1 | yes |
| knot refit | 1,296 | 1 | 0.314331 | 1 | yes |
| refined parameter-search resolution | 1,296 | 1 | 0.314331 | 1 | yes |
| covariance shape 1/2 | 1,296 | 1 | 0.314331 | 1 | yes |
| covariance shape 2/2 | 1,296 | 1 | 0.314331 | 1 | yes |
| post-asinh transform 1/4 | 1,296 | 1 | 0.314331 | 1 | yes |
| post-asinh transform 2/4 | 1,296 | 1 | 0.314331 | 1 | yes |
| post-asinh transform 3/4 | 1,296 | 1 | 0.314331 | 1 | yes |
| post-asinh transform 4/4 | 1,296 | 1 | 0.314331 | 1 | yes |
| allocation beta | 1,296 | 1 | 0.314331 | 1 | yes |
| knot refit | 1,296 | 1 | 0.314331 | 1 | yes |
| fixed point reached | 1,296 | 1 | 0.314331 | 1 | yes |
| accepted prune | 1,296 | 1 | 0.314331 | 1 | yes |
| prune proposal 1296->972 | 972 | 0.875 | 0.445112 | 1 | yes |
| covariance shape 1/2 | 972 | 1 | 0.530072 | 1 | yes |
| covariance shape 2/2 | 972 | 1 | 0.530072 | 1 | yes |
| post-asinh transform 1/4 | 972 | 1 | 0.530072 | 1 | yes |
| post-asinh transform 2/4 | 972 | 1 | 0.530072 | 1 | yes |
| post-asinh transform 3/4 | 972 | 1 | 0.530072 | 1 | yes |
| post-asinh transform 4/4 | 972 | 1 | 0.530072 | 1 | yes |
| allocation beta | 972 | 0.875 | 0.489806 | 1 | yes |
| knot refit | 972 | 0.875 | 0.466611 | 1 | yes |
| covariance shape 1/2 | 972 | 0.875 | 0.466611 | 1 | yes |
| covariance shape 2/2 | 972 | 0.875 | 0.466611 | 1 | yes |
| post-asinh transform 1/4 | 972 | 0.875 | 0.466611 | 1 | yes |
| post-asinh transform 2/4 | 972 | 0.875 | 0.466611 | 1 | yes |
| post-asinh transform 3/4 | 972 | 0.875 | 0.466611 | 1 | yes |
| post-asinh transform 4/4 | 972 | 0.875 | 0.466611 | 1 | yes |
| allocation beta | 972 | 1 | 0.460619 | 1 | yes |
| knot refit | 972 | 1 | 0.455076 | 1 | yes |
| covariance shape 1/2 | 972 | 1 | 0.455076 | 1 | yes |
| covariance shape 2/2 | 972 | 1 | 0.455076 | 1 | yes |
| post-asinh transform 1/4 | 972 | 1 | 0.455076 | 1 | yes |
| post-asinh transform 2/4 | 972 | 1 | 0.455076 | 1 | yes |
| post-asinh transform 3/4 | 972 | 1 | 0.455076 | 1 | yes |
| post-asinh transform 4/4 | 972 | 1 | 0.455076 | 1 | yes |
| allocation beta | 972 | 0.875 | 0.449625 | 1 | yes |
| knot refit | 972 | 0.875 | 0.449625 | 1 | yes |
| covariance shape 1/2 | 972 | 0.875 | 0.449625 | 1 | yes |
| covariance shape 2/2 | 972 | 0.875 | 0.449625 | 1 | yes |
| post-asinh transform 1/4 | 972 | 0.875 | 0.449625 | 1 | yes |
| post-asinh transform 2/4 | 972 | 0.875 | 0.449625 | 1 | yes |
| post-asinh transform 3/4 | 972 | 0.875 | 0.449625 | 1 | yes |
| post-asinh transform 4/4 | 972 | 0.875 | 0.449625 | 1 | yes |
| allocation beta | 972 | 0.875 | 0.449625 | 1 | yes |
| knot refit | 972 | 0.875 | 0.449625 | 1 | yes |
| refined parameter-search resolution | 972 | 0.875 | 0.449625 | 1 | yes |
| covariance shape 1/2 | 972 | 0.875 | 0.449625 | 1 | yes |
| covariance shape 2/2 | 972 | 0.875 | 0.449625 | 1 | yes |
| post-asinh transform 1/4 | 972 | 0.875 | 0.449625 | 1 | yes |
| post-asinh transform 2/4 | 972 | 0.875 | 0.449625 | 1 | yes |
| post-asinh transform 3/4 | 972 | 0.875 | 0.449625 | 1 | yes |
| post-asinh transform 4/4 | 972 | 0.875 | 0.449625 | 1 | yes |
| allocation beta | 972 | 0.875 | 0.449625 | 1 | yes |
| knot refit | 972 | 0.875 | 0.449625 | 1 | yes |
| refined parameter-search resolution | 972 | 0.875 | 0.449625 | 1 | yes |
| covariance shape 1/2 | 972 | 0.875 | 0.449625 | 1 | yes |
| covariance shape 2/2 | 972 | 0.875 | 0.449625 | 1 | yes |
| post-asinh transform 1/4 | 972 | 0.875 | 0.449625 | 1 | yes |
| post-asinh transform 2/4 | 972 | 0.875 | 0.449625 | 1 | yes |
| post-asinh transform 3/4 | 972 | 0.875 | 0.449625 | 1 | yes |
| post-asinh transform 4/4 | 972 | 0.875 | 0.449625 | 1 | yes |
| allocation beta | 972 | 0.84375 | 0.449127 | 1 | yes |
| knot refit | 972 | 0.84375 | 0.449127 | 1 | yes |
| covariance shape 1/2 | 972 | 0.84375 | 0.449127 | 1 | yes |
| covariance shape 2/2 | 972 | 0.84375 | 0.449127 | 1 | yes |
| post-asinh transform 1/4 | 972 | 0.84375 | 0.449127 | 1 | yes |
| post-asinh transform 2/4 | 972 | 0.84375 | 0.449127 | 1 | yes |
| post-asinh transform 3/4 | 972 | 0.84375 | 0.449127 | 1 | yes |
| post-asinh transform 4/4 | 972 | 0.84375 | 0.449127 | 1 | yes |
| allocation beta | 972 | 0.875 | 0.445112 | 1 | yes |
| knot refit | 972 | 0.875 | 0.445112 | 1 | yes |
| covariance shape 1/2 | 972 | 0.875 | 0.445112 | 1 | yes |
| covariance shape 2/2 | 972 | 0.875 | 0.445112 | 1 | yes |
| post-asinh transform 1/4 | 972 | 0.875 | 0.445112 | 1 | yes |
| post-asinh transform 2/4 | 972 | 0.875 | 0.445112 | 1 | yes |
| post-asinh transform 3/4 | 972 | 0.875 | 0.445112 | 1 | yes |
| post-asinh transform 4/4 | 972 | 0.875 | 0.445112 | 1 | yes |
| allocation beta | 972 | 0.875 | 0.445112 | 1 | yes |
| knot refit | 972 | 0.875 | 0.445112 | 1 | yes |
| fixed point reached | 972 | 0.875 | 0.445112 | 1 | yes |
| accepted prune | 972 | 0.875 | 0.445112 | 1 | yes |
| prune proposal 972->729 | 729 | 0.96875 | 0.846157 | 1 | yes |
| covariance shape 1/2 | 729 | 0.875 | 1.65797 | 1.80593 | no |
| covariance shape 2/2 | 729 | 0.875 | 1.65797 | 1.80593 | no |
| post-asinh transform 1/4 | 729 | 0.875 | 1.65797 | 1.80593 | no |
| post-asinh transform 2/4 | 729 | 0.875 | 1.65797 | 1.80593 | no |
| post-asinh transform 3/4 | 729 | 0.875 | 1.65797 | 1.80593 | no |
| post-asinh transform 4/4 | 729 | 0.875 | 1.65797 | 1.80593 | no |
| allocation beta | 729 | 1 | 1.20955 | 1.29856 | no |
| knot refit | 729 | 1 | 0.909325 | 1 | yes |
| covariance shape 1/2 | 729 | 1 | 0.909325 | 1 | yes |
| covariance shape 2/2 | 729 | 1 | 0.909325 | 1 | yes |
| post-asinh transform 1/4 | 729 | 1 | 0.909325 | 1 | yes |
| post-asinh transform 2/4 | 729 | 1 | 0.909325 | 1 | yes |
| post-asinh transform 3/4 | 729 | 1 | 0.909325 | 1 | yes |
| post-asinh transform 4/4 | 729 | 1 | 0.909325 | 1 | yes |
| allocation beta | 729 | 1 | 0.909325 | 1 | yes |
| knot refit | 729 | 1 | 0.854902 | 1 | yes |
| covariance shape 1/2 | 729 | 1 | 0.854902 | 1 | yes |
| covariance shape 2/2 | 729 | 1 | 0.854902 | 1 | yes |
| post-asinh transform 1/4 | 729 | 1 | 0.854902 | 1 | yes |
| post-asinh transform 2/4 | 729 | 1 | 0.854902 | 1 | yes |
| post-asinh transform 3/4 | 729 | 1 | 0.854902 | 1 | yes |
| post-asinh transform 4/4 | 729 | 1 | 0.854902 | 1 | yes |
| allocation beta | 729 | 1 | 0.854902 | 1 | yes |
| knot refit | 729 | 1 | 0.854902 | 1 | yes |
| refined parameter-search resolution | 729 | 1 | 0.854902 | 1 | yes |
| covariance shape 1/2 | 729 | 1 | 0.854902 | 1 | yes |
| covariance shape 2/2 | 729 | 1 | 0.854902 | 1 | yes |
| post-asinh transform 1/4 | 729 | 1 | 0.854902 | 1 | yes |
| post-asinh transform 2/4 | 729 | 1 | 0.854902 | 1 | yes |
| post-asinh transform 3/4 | 729 | 1 | 0.854902 | 1 | yes |
| post-asinh transform 4/4 | 729 | 1 | 0.854902 | 1 | yes |
| allocation beta | 729 | 1 | 0.854902 | 1 | yes |
| knot refit | 729 | 1 | 0.854902 | 1 | yes |
| refined parameter-search resolution | 729 | 1 | 0.854902 | 1 | yes |
| covariance shape 1/2 | 729 | 1 | 0.854902 | 1 | yes |
| covariance shape 2/2 | 729 | 1 | 0.854902 | 1 | yes |
| post-asinh transform 1/4 | 729 | 1 | 0.854902 | 1 | yes |
| post-asinh transform 2/4 | 729 | 1 | 0.854902 | 1 | yes |
| post-asinh transform 3/4 | 729 | 1 | 0.854902 | 1 | yes |
| post-asinh transform 4/4 | 729 | 1 | 0.854902 | 1 | yes |
| allocation beta | 729 | 0.96875 | 0.846157 | 1 | yes |
| knot refit | 729 | 0.96875 | 0.846157 | 1 | yes |
| covariance shape 1/2 | 729 | 0.96875 | 0.846157 | 1 | yes |
| covariance shape 2/2 | 729 | 0.96875 | 0.846157 | 1 | yes |
| post-asinh transform 1/4 | 729 | 0.96875 | 0.846157 | 1 | yes |
| post-asinh transform 2/4 | 729 | 0.96875 | 0.846157 | 1 | yes |
| post-asinh transform 3/4 | 729 | 0.96875 | 0.846157 | 1 | yes |
| post-asinh transform 4/4 | 729 | 0.96875 | 0.846157 | 1 | yes |
| allocation beta | 729 | 0.96875 | 0.846157 | 1 | yes |
| knot refit | 729 | 0.96875 | 0.846157 | 1 | yes |
| fixed point reached | 729 | 0.96875 | 0.846157 | 1 | yes |
| accepted prune | 729 | 0.96875 | 0.846157 | 1 | yes |
| cold restart 729->547 | 547 | 1 | 1.32562 | 1.3409 | no |
| covariance shape 1/2 | 547 | 0.96875 | 1.64736 | 1.76461 | no |
| covariance shape 2/2 | 547 | 0.96875 | 1.64736 | 1.76461 | no |
| post-asinh transform 1/4 | 547 | 0.96875 | 1.64736 | 1.76461 | no |
| post-asinh transform 2/4 | 547 | 0.96875 | 1.64736 | 1.76461 | no |
| post-asinh transform 3/4 | 547 | 0.96875 | 1.64736 | 1.76461 | no |
| post-asinh transform 4/4 | 547 | 0.96875 | 1.64736 | 1.76461 | no |
| allocation beta | 547 | 1 | 1.56207 | 1.67669 | no |
| knot refit | 547 | 1 | 1.41654 | 1.49105 | no |
| covariance shape 1/2 | 547 | 1 | 1.41654 | 1.49105 | no |
| covariance shape 2/2 | 547 | 1 | 1.41654 | 1.49105 | no |
| post-asinh transform 1/4 | 547 | 1 | 1.41654 | 1.49105 | no |
| post-asinh transform 2/4 | 547 | 1 | 1.41654 | 1.49105 | no |
| post-asinh transform 3/4 | 547 | 1 | 1.41654 | 1.49105 | no |
| post-asinh transform 4/4 | 547 | 1 | 1.41654 | 1.49105 | no |
| allocation beta | 547 | 1 | 1.41654 | 1.49105 | no |
| knot refit | 547 | 1 | 1.38848 | 1.45314 | no |
| covariance shape 1/2 | 547 | 1 | 1.38848 | 1.45314 | no |
| covariance shape 2/2 | 547 | 1 | 1.38848 | 1.45314 | no |
| post-asinh transform 1/4 | 547 | 1 | 1.38848 | 1.45314 | no |
| post-asinh transform 2/4 | 547 | 1 | 1.38848 | 1.45314 | no |
| post-asinh transform 3/4 | 547 | 1 | 1.38848 | 1.45314 | no |
| post-asinh transform 4/4 | 547 | 1 | 1.38848 | 1.45314 | no |
| allocation beta | 547 | 1 | 1.38848 | 1.45314 | no |
| knot refit | 547 | 1 | 1.38623 | 1.45342 | no |
| covariance shape 1/2 | 547 | 1 | 1.38623 | 1.45342 | no |
| covariance shape 2/2 | 547 | 1 | 1.38623 | 1.45342 | no |
| post-asinh transform 1/4 | 547 | 1 | 1.38623 | 1.45342 | no |
| post-asinh transform 2/4 | 547 | 1 | 1.38623 | 1.45342 | no |
| post-asinh transform 3/4 | 547 | 1 | 1.38623 | 1.45342 | no |
| post-asinh transform 4/4 | 547 | 1 | 1.38623 | 1.45342 | no |
| allocation beta | 547 | 1 | 1.38623 | 1.45342 | no |
| knot refit | 547 | 1 | 1.36677 | 1.42659 | no |
| covariance shape 1/2 | 547 | 1 | 1.36677 | 1.42659 | no |
| covariance shape 2/2 | 547 | 1 | 1.36677 | 1.42659 | no |
| post-asinh transform 1/4 | 547 | 1 | 1.36677 | 1.42659 | no |
| post-asinh transform 2/4 | 547 | 1 | 1.36677 | 1.42659 | no |
| post-asinh transform 3/4 | 547 | 1 | 1.36677 | 1.42659 | no |
| post-asinh transform 4/4 | 547 | 1 | 1.36677 | 1.42659 | no |
| allocation beta | 547 | 1 | 1.36677 | 1.42659 | no |
| knot refit | 547 | 1 | 1.36003 | 1.41788 | no |
| covariance shape 1/2 | 547 | 1 | 1.36003 | 1.41788 | no |
| covariance shape 2/2 | 547 | 1 | 1.36003 | 1.41788 | no |
| post-asinh transform 1/4 | 547 | 1 | 1.36003 | 1.41788 | no |
| post-asinh transform 2/4 | 547 | 1 | 1.36003 | 1.41788 | no |
| post-asinh transform 3/4 | 547 | 1 | 1.36003 | 1.41788 | no |
| post-asinh transform 4/4 | 547 | 1 | 1.36003 | 1.41788 | no |
| allocation beta | 547 | 1 | 1.36003 | 1.41788 | no |
| knot refit | 547 | 1 | 1.36003 | 1.41788 | no |
| refined parameter-search resolution | 547 | 1 | 1.36003 | 1.41788 | no |
| covariance shape 1/2 | 547 | 1 | 1.36003 | 1.41788 | no |
| covariance shape 2/2 | 547 | 1 | 1.36003 | 1.41788 | no |
| post-asinh transform 1/4 | 547 | 1 | 1.36003 | 1.41788 | no |
| post-asinh transform 2/4 | 547 | 1 | 1.36003 | 1.41788 | no |
| post-asinh transform 3/4 | 547 | 1 | 1.36003 | 1.41788 | no |
| post-asinh transform 4/4 | 547 | 1 | 1.36003 | 1.41788 | no |
| allocation beta | 547 | 1 | 1.36003 | 1.41788 | no |
| knot refit | 547 | 1 | 1.36003 | 1.41788 | no |
| refined parameter-search resolution | 547 | 1 | 1.36003 | 1.41788 | no |
| covariance shape 1/2 | 547 | 1 | 1.34174 | 1.37927 | no |
| covariance shape 2/2 | 547 | 1 | 1.34174 | 1.37927 | no |
| post-asinh transform 1/4 | 547 | 1 | 1.32562 | 1.3409 | no |
| post-asinh transform 2/4 | 547 | 1 | 1.32562 | 1.3409 | no |
| post-asinh transform 3/4 | 547 | 1 | 1.32562 | 1.3409 | no |
| post-asinh transform 4/4 | 547 | 1 | 1.32562 | 1.3409 | no |
| allocation beta | 547 | 1 | 1.32562 | 1.3409 | no |
| knot refit | 547 | 1 | 1.32562 | 1.3409 | no |
| covariance shape 1/2 | 547 | 1 | 1.32562 | 1.3409 | no |
| covariance shape 2/2 | 547 | 1 | 1.32562 | 1.3409 | no |
| post-asinh transform 1/4 | 547 | 1 | 1.32562 | 1.3409 | no |
| post-asinh transform 2/4 | 547 | 1 | 1.32562 | 1.3409 | no |
| post-asinh transform 3/4 | 547 | 1 | 1.32562 | 1.3409 | no |
| post-asinh transform 4/4 | 547 | 1 | 1.32562 | 1.3409 | no |
| allocation beta | 547 | 1 | 1.32562 | 1.3409 | no |
| knot refit | 547 | 1 | 1.32562 | 1.3409 | no |
| fixed point reached | 547 | 1 | 1.32562 | 1.3409 | no |
| prune proposal 729->547 | 547 | 1 | 1.32562 | 1.3409 | no |
| covariance shape 1/2 | 547 | 0.96875 | 1.64173 | 1.77374 | no |
| covariance shape 2/2 | 547 | 0.96875 | 1.64173 | 1.77374 | no |
| post-asinh transform 1/4 | 547 | 0.96875 | 1.64173 | 1.77374 | no |
| post-asinh transform 2/4 | 547 | 0.96875 | 1.64173 | 1.77374 | no |
| post-asinh transform 3/4 | 547 | 0.96875 | 1.64173 | 1.77374 | no |
| post-asinh transform 4/4 | 547 | 0.96875 | 1.64173 | 1.77374 | no |
| allocation beta | 547 | 1 | 1.53235 | 1.65829 | no |
| knot refit | 547 | 1 | 1.49905 | 1.62089 | no |
| covariance shape 1/2 | 547 | 1 | 1.49905 | 1.62089 | no |
| covariance shape 2/2 | 547 | 1 | 1.49905 | 1.62089 | no |
| post-asinh transform 1/4 | 547 | 1 | 1.49905 | 1.62089 | no |
| post-asinh transform 2/4 | 547 | 1 | 1.49905 | 1.62089 | no |
| post-asinh transform 3/4 | 547 | 1 | 1.49905 | 1.62089 | no |
| post-asinh transform 4/4 | 547 | 1 | 1.49905 | 1.62089 | no |
| allocation beta | 547 | 1 | 1.49905 | 1.62089 | no |
| knot refit | 547 | 1 | 1.46521 | 1.57653 | no |
| covariance shape 1/2 | 547 | 1 | 1.46521 | 1.57653 | no |
| covariance shape 2/2 | 547 | 1 | 1.46521 | 1.57653 | no |
| post-asinh transform 1/4 | 547 | 1 | 1.46521 | 1.57653 | no |
| post-asinh transform 2/4 | 547 | 1 | 1.46521 | 1.57653 | no |
| post-asinh transform 3/4 | 547 | 1 | 1.46521 | 1.57653 | no |
| post-asinh transform 4/4 | 547 | 1 | 1.46521 | 1.57653 | no |
| allocation beta | 547 | 0.875 | 1.4628 | 1.57027 | no |
| knot refit | 547 | 0.875 | 1.4628 | 1.57027 | no |
| covariance shape 1/2 | 547 | 0.875 | 1.4628 | 1.57027 | no |
| covariance shape 2/2 | 547 | 0.875 | 1.4628 | 1.57027 | no |
| post-asinh transform 1/4 | 547 | 0.875 | 1.4628 | 1.57027 | no |
| post-asinh transform 2/4 | 547 | 0.875 | 1.4628 | 1.57027 | no |
| post-asinh transform 3/4 | 547 | 0.875 | 1.4628 | 1.57027 | no |
| post-asinh transform 4/4 | 547 | 0.875 | 1.4628 | 1.57027 | no |
| allocation beta | 547 | 1 | 1.41516 | 1.49874 | no |
| knot refit | 547 | 1 | 1.4085 | 1.49954 | no |
| covariance shape 1/2 | 547 | 1 | 1.4085 | 1.49954 | no |
| covariance shape 2/2 | 547 | 1 | 1.4085 | 1.49954 | no |
| post-asinh transform 1/4 | 547 | 1 | 1.4085 | 1.49954 | no |
| post-asinh transform 2/4 | 547 | 1 | 1.4085 | 1.49954 | no |
| post-asinh transform 3/4 | 547 | 1 | 1.4085 | 1.49954 | no |
| post-asinh transform 4/4 | 547 | 1 | 1.4085 | 1.49954 | no |
| allocation beta | 547 | 1 | 1.4085 | 1.49954 | no |
| knot refit | 547 | 1 | 1.40515 | 1.49696 | no |
| covariance shape 1/2 | 547 | 1 | 1.40515 | 1.49696 | no |
| covariance shape 2/2 | 547 | 1 | 1.40515 | 1.49696 | no |
| post-asinh transform 1/4 | 547 | 1 | 1.40515 | 1.49696 | no |
| post-asinh transform 2/4 | 547 | 1 | 1.40515 | 1.49696 | no |
| post-asinh transform 3/4 | 547 | 1 | 1.40515 | 1.49696 | no |
| post-asinh transform 4/4 | 547 | 1 | 1.40515 | 1.49696 | no |
| allocation beta | 547 | 1 | 1.40515 | 1.49696 | no |
| knot refit | 547 | 1 | 1.39381 | 1.48934 | no |
| covariance shape 1/2 | 547 | 1 | 1.39381 | 1.48934 | no |
| covariance shape 2/2 | 547 | 1 | 1.39381 | 1.48934 | no |
| post-asinh transform 1/4 | 547 | 1 | 1.39381 | 1.48934 | no |
| post-asinh transform 2/4 | 547 | 1 | 1.39381 | 1.48934 | no |
| post-asinh transform 3/4 | 547 | 1 | 1.39381 | 1.48934 | no |
| post-asinh transform 4/4 | 547 | 1 | 1.39381 | 1.48934 | no |
| allocation beta | 547 | 1 | 1.39381 | 1.48934 | no |
| knot refit | 547 | 1 | 1.39238 | 1.48865 | no |
| covariance shape 1/2 | 547 | 1 | 1.39238 | 1.48865 | no |
| covariance shape 2/2 | 547 | 1 | 1.39238 | 1.48865 | no |
| post-asinh transform 1/4 | 547 | 1 | 1.39238 | 1.48865 | no |
| post-asinh transform 2/4 | 547 | 1 | 1.39238 | 1.48865 | no |
| post-asinh transform 3/4 | 547 | 1 | 1.39238 | 1.48865 | no |
| post-asinh transform 4/4 | 547 | 1 | 1.39238 | 1.48865 | no |
| allocation beta | 547 | 1 | 1.39238 | 1.48865 | no |
| knot refit | 547 | 1 | 1.38896 | 1.48501 | no |
| covariance shape 1/2 | 547 | 1 | 1.38896 | 1.48501 | no |
| covariance shape 2/2 | 547 | 1 | 1.38896 | 1.48501 | no |
| post-asinh transform 1/4 | 547 | 1 | 1.38896 | 1.48501 | no |
| post-asinh transform 2/4 | 547 | 1 | 1.38896 | 1.48501 | no |
| post-asinh transform 3/4 | 547 | 1 | 1.38896 | 1.48501 | no |
| post-asinh transform 4/4 | 547 | 1 | 1.38896 | 1.48501 | no |
| allocation beta | 547 | 1 | 1.38896 | 1.48501 | no |
| knot refit | 547 | 1 | 1.38721 | 1.48313 | no |
| covariance shape 1/2 | 547 | 1 | 1.38721 | 1.48313 | no |
| covariance shape 2/2 | 547 | 1 | 1.38721 | 1.48313 | no |
| post-asinh transform 1/4 | 547 | 1 | 1.38721 | 1.48313 | no |
| post-asinh transform 2/4 | 547 | 1 | 1.38721 | 1.48313 | no |
| post-asinh transform 3/4 | 547 | 1 | 1.38721 | 1.48313 | no |
| post-asinh transform 4/4 | 547 | 1 | 1.38721 | 1.48313 | no |
| allocation beta | 547 | 1 | 1.38721 | 1.48313 | no |
| knot refit | 547 | 1 | 1.38721 | 1.48313 | no |
| refined parameter-search resolution | 547 | 1 | 1.38721 | 1.48313 | no |
| covariance shape 1/2 | 547 | 1 | 1.38721 | 1.48313 | no |
| covariance shape 2/2 | 547 | 1 | 1.38721 | 1.48313 | no |
| post-asinh transform 1/4 | 547 | 1 | 1.38721 | 1.48313 | no |
| post-asinh transform 2/4 | 547 | 1 | 1.38721 | 1.48313 | no |
| post-asinh transform 3/4 | 547 | 1 | 1.38721 | 1.48313 | no |
| post-asinh transform 4/4 | 547 | 1 | 1.38721 | 1.48313 | no |
| allocation beta | 547 | 1 | 1.38721 | 1.48313 | no |
| knot refit | 547 | 1 | 1.38721 | 1.48313 | no |
| refined parameter-search resolution | 547 | 1 | 1.38721 | 1.48313 | no |
| covariance shape 1/2 | 547 | 1 | 1.38721 | 1.48313 | no |
| covariance shape 2/2 | 547 | 1 | 1.38721 | 1.48313 | no |
| post-asinh transform 1/4 | 547 | 1 | 1.38721 | 1.48313 | no |
| post-asinh transform 2/4 | 547 | 1 | 1.38721 | 1.48313 | no |
| post-asinh transform 3/4 | 547 | 1 | 1.38721 | 1.48313 | no |
| post-asinh transform 4/4 | 547 | 1 | 1.38721 | 1.48313 | no |
| allocation beta | 547 | 1 | 1.38721 | 1.48313 | no |
| knot refit | 547 | 1 | 1.38721 | 1.48313 | no |
| fixed point reached | 547 | 1 | 1.38721 | 1.48313 | no |
| rejected prune | 547 | 1 | 1.32562 | 1.3409 | no |
| cold restart 729->638 | 638 | 1 | 1.15042 | 1.1991 | no |
| covariance shape 1/2 | 638 | 0.96875 | 1.33194 | 1.43722 | no |
| covariance shape 2/2 | 638 | 0.96875 | 1.33194 | 1.43722 | no |
| post-asinh transform 1/4 | 638 | 0.96875 | 1.33194 | 1.43722 | no |
| post-asinh transform 2/4 | 638 | 0.96875 | 1.33194 | 1.43722 | no |
| post-asinh transform 3/4 | 638 | 0.96875 | 1.33194 | 1.43722 | no |
| post-asinh transform 4/4 | 638 | 0.96875 | 1.33194 | 1.43722 | no |
| allocation beta | 638 | 1 | 1.17306 | 1.23853 | no |
| knot refit | 638 | 1 | 1.17306 | 1.23853 | no |
| covariance shape 1/2 | 638 | 1 | 1.17306 | 1.23853 | no |
| covariance shape 2/2 | 638 | 1 | 1.17306 | 1.23853 | no |
| post-asinh transform 1/4 | 638 | 1 | 1.17306 | 1.23853 | no |
| post-asinh transform 2/4 | 638 | 1 | 1.17306 | 1.23853 | no |
| post-asinh transform 3/4 | 638 | 1 | 1.17306 | 1.23853 | no |
| post-asinh transform 4/4 | 638 | 1 | 1.17306 | 1.23853 | no |
| allocation beta | 638 | 1 | 1.17306 | 1.23853 | no |
| knot refit | 638 | 1 | 1.17306 | 1.23853 | no |
| refined parameter-search resolution | 638 | 1 | 1.17306 | 1.23853 | no |
| covariance shape 1/2 | 638 | 1 | 1.17306 | 1.23853 | no |
| covariance shape 2/2 | 638 | 1 | 1.17306 | 1.23853 | no |
| post-asinh transform 1/4 | 638 | 1 | 1.17306 | 1.23853 | no |
| post-asinh transform 2/4 | 638 | 1 | 1.17306 | 1.23853 | no |
| post-asinh transform 3/4 | 638 | 1 | 1.17306 | 1.23853 | no |
| post-asinh transform 4/4 | 638 | 1 | 1.17306 | 1.23853 | no |
| allocation beta | 638 | 1 | 1.17306 | 1.23853 | no |
| knot refit | 638 | 1 | 1.17306 | 1.23853 | no |
| refined parameter-search resolution | 638 | 1 | 1.17306 | 1.23853 | no |
| covariance shape 1/2 | 638 | 1 | 1.15042 | 1.1991 | no |
| covariance shape 2/2 | 638 | 1 | 1.15042 | 1.1991 | no |
| post-asinh transform 1/4 | 638 | 1 | 1.15042 | 1.1991 | no |
| post-asinh transform 2/4 | 638 | 1 | 1.15042 | 1.1991 | no |
| post-asinh transform 3/4 | 638 | 1 | 1.15042 | 1.1991 | no |
| post-asinh transform 4/4 | 638 | 1 | 1.15042 | 1.1991 | no |
| allocation beta | 638 | 1 | 1.15042 | 1.1991 | no |
| knot refit | 638 | 1 | 1.15042 | 1.1991 | no |
| covariance shape 1/2 | 638 | 1 | 1.15042 | 1.1991 | no |
| covariance shape 2/2 | 638 | 1 | 1.15042 | 1.1991 | no |
| post-asinh transform 1/4 | 638 | 1 | 1.15042 | 1.1991 | no |
| post-asinh transform 2/4 | 638 | 1 | 1.15042 | 1.1991 | no |
| post-asinh transform 3/4 | 638 | 1 | 1.15042 | 1.1991 | no |
| post-asinh transform 4/4 | 638 | 1 | 1.15042 | 1.1991 | no |
| allocation beta | 638 | 1 | 1.15042 | 1.1991 | no |
| knot refit | 638 | 1 | 1.15042 | 1.1991 | no |
| fixed point reached | 638 | 1 | 1.15042 | 1.1991 | no |
| prune proposal 729->638 | 638 | 1 | 1.15042 | 1.1991 | no |
| covariance shape 1/2 | 638 | 0.96875 | 1.26215 | 1.32129 | no |
| covariance shape 2/2 | 638 | 0.96875 | 1.26215 | 1.32129 | no |
| post-asinh transform 1/4 | 638 | 0.96875 | 1.26215 | 1.32129 | no |
| post-asinh transform 2/4 | 638 | 0.96875 | 1.26215 | 1.32129 | no |
| post-asinh transform 3/4 | 638 | 0.96875 | 1.26215 | 1.32129 | no |
| post-asinh transform 4/4 | 638 | 0.96875 | 1.26215 | 1.32129 | no |
| allocation beta | 638 | 0.96875 | 1.26215 | 1.32129 | no |
| knot refit | 638 | 0.96875 | 1.23941 | 1.29204 | no |
| covariance shape 1/2 | 638 | 0.96875 | 1.23941 | 1.29204 | no |
| covariance shape 2/2 | 638 | 0.96875 | 1.23941 | 1.29204 | no |
| post-asinh transform 1/4 | 638 | 0.96875 | 1.23941 | 1.29204 | no |
| post-asinh transform 2/4 | 638 | 0.96875 | 1.23941 | 1.29204 | no |
| post-asinh transform 3/4 | 638 | 0.96875 | 1.23941 | 1.29204 | no |
| post-asinh transform 4/4 | 638 | 0.96875 | 1.23941 | 1.29204 | no |
| allocation beta | 638 | 0.96875 | 1.23941 | 1.29204 | no |
| knot refit | 638 | 0.96875 | 1.23941 | 1.29204 | no |
| refined parameter-search resolution | 638 | 0.96875 | 1.23941 | 1.29204 | no |
| covariance shape 1/2 | 638 | 0.96875 | 1.23941 | 1.29204 | no |
| covariance shape 2/2 | 638 | 0.96875 | 1.23941 | 1.29204 | no |
| post-asinh transform 1/4 | 638 | 0.96875 | 1.23941 | 1.29204 | no |
| post-asinh transform 2/4 | 638 | 0.96875 | 1.23941 | 1.29204 | no |
| post-asinh transform 3/4 | 638 | 0.96875 | 1.23941 | 1.29204 | no |
| post-asinh transform 4/4 | 638 | 0.96875 | 1.23941 | 1.29204 | no |
| allocation beta | 638 | 0.96875 | 1.23941 | 1.29204 | no |
| knot refit | 638 | 0.96875 | 1.23941 | 1.29204 | no |
| refined parameter-search resolution | 638 | 0.96875 | 1.23941 | 1.29204 | no |
| covariance shape 1/2 | 638 | 0.96875 | 1.23941 | 1.29204 | no |
| covariance shape 2/2 | 638 | 0.96875 | 1.23941 | 1.29204 | no |
| post-asinh transform 1/4 | 638 | 0.96875 | 1.23672 | 1.31107 | no |
| post-asinh transform 2/4 | 638 | 0.96875 | 1.23672 | 1.31107 | no |
| post-asinh transform 3/4 | 638 | 0.96875 | 1.23672 | 1.31107 | no |
| post-asinh transform 4/4 | 638 | 0.96875 | 1.23672 | 1.31107 | no |
| allocation beta | 638 | 0.96875 | 1.23672 | 1.31107 | no |
| knot refit | 638 | 0.96875 | 1.23672 | 1.31107 | no |
| covariance shape 1/2 | 638 | 0.96875 | 1.23672 | 1.31107 | no |
| covariance shape 2/2 | 638 | 0.96875 | 1.23672 | 1.31107 | no |
| post-asinh transform 1/4 | 638 | 0.96875 | 1.23672 | 1.31107 | no |
| post-asinh transform 2/4 | 638 | 0.96875 | 1.23672 | 1.31107 | no |
| post-asinh transform 3/4 | 638 | 0.96875 | 1.23672 | 1.31107 | no |
| post-asinh transform 4/4 | 638 | 0.96875 | 1.23672 | 1.31107 | no |
| allocation beta | 638 | 0.96875 | 1.23672 | 1.31107 | no |
| knot refit | 638 | 0.96875 | 1.23672 | 1.31107 | no |
| fixed point reached | 638 | 0.96875 | 1.23672 | 1.31107 | no |
| rejected prune | 638 | 1 | 1.15042 | 1.1991 | no |
| prune proposal 729->684 | 684 | 0.96875 | 0.912566 | 1 | yes |
| covariance shape 1/2 | 684 | 0.96875 | 1.2623 | 1.31894 | no |
| covariance shape 2/2 | 684 | 0.96875 | 1.2623 | 1.31894 | no |
| post-asinh transform 1/4 | 684 | 0.96875 | 1.2623 | 1.31894 | no |
| post-asinh transform 2/4 | 684 | 0.96875 | 1.2623 | 1.31894 | no |
| post-asinh transform 3/4 | 684 | 0.96875 | 1.2623 | 1.31894 | no |
| post-asinh transform 4/4 | 684 | 0.96875 | 1.2623 | 1.31894 | no |
| allocation beta | 684 | 1 | 1.05868 | 1.12882 | no |
| knot refit | 684 | 1 | 1.05868 | 1.12882 | no |
| covariance shape 1/2 | 684 | 1 | 1.05868 | 1.12882 | no |
| covariance shape 2/2 | 684 | 1 | 1.05868 | 1.12882 | no |
| post-asinh transform 1/4 | 684 | 1 | 1.05868 | 1.12882 | no |
| post-asinh transform 2/4 | 684 | 1 | 1.05868 | 1.12882 | no |
| post-asinh transform 3/4 | 684 | 1 | 1.05868 | 1.12882 | no |
| post-asinh transform 4/4 | 684 | 1 | 1.05868 | 1.12882 | no |
| allocation beta | 684 | 1 | 1.05868 | 1.12882 | no |
| knot refit | 684 | 1 | 1.05868 | 1.12882 | no |
| refined parameter-search resolution | 684 | 1 | 1.05868 | 1.12882 | no |
| covariance shape 1/2 | 684 | 1 | 1.05868 | 1.12882 | no |
| covariance shape 2/2 | 684 | 1 | 1.05868 | 1.12882 | no |
| post-asinh transform 1/4 | 684 | 1 | 1.05868 | 1.12882 | no |
| post-asinh transform 2/4 | 684 | 1 | 1.05868 | 1.12882 | no |
| post-asinh transform 3/4 | 684 | 1 | 1.05868 | 1.12882 | no |
| post-asinh transform 4/4 | 684 | 1 | 1.05868 | 1.12882 | no |
| allocation beta | 684 | 1 | 1.05868 | 1.12882 | no |
| knot refit | 684 | 1 | 1.05868 | 1.12882 | no |
| refined parameter-search resolution | 684 | 1 | 1.05868 | 1.12882 | no |
| covariance shape 1/2 | 684 | 1 | 1.05868 | 1.12882 | no |
| covariance shape 2/2 | 684 | 1 | 1.05868 | 1.12882 | no |
| post-asinh transform 1/4 | 684 | 1 | 1.05868 | 1.12882 | no |
| post-asinh transform 2/4 | 684 | 1 | 1.05868 | 1.12882 | no |
| post-asinh transform 3/4 | 684 | 1 | 1.05868 | 1.12882 | no |
| post-asinh transform 4/4 | 684 | 1 | 1.05868 | 1.12882 | no |
| allocation beta | 684 | 0.96875 | 1.04818 | 1.14226 | no |
| knot refit | 684 | 0.96875 | 0.912566 | 1 | yes |
| covariance shape 1/2 | 684 | 0.96875 | 0.912566 | 1 | yes |
| covariance shape 2/2 | 684 | 0.96875 | 0.912566 | 1 | yes |
| post-asinh transform 1/4 | 684 | 0.96875 | 0.912566 | 1 | yes |
| post-asinh transform 2/4 | 684 | 0.96875 | 0.912566 | 1 | yes |
| post-asinh transform 3/4 | 684 | 0.96875 | 0.912566 | 1 | yes |
| post-asinh transform 4/4 | 684 | 0.96875 | 0.912566 | 1 | yes |
| allocation beta | 684 | 0.96875 | 0.912566 | 1 | yes |
| knot refit | 684 | 0.96875 | 0.912566 | 1 | yes |
| fixed point reached | 684 | 0.96875 | 0.912566 | 1 | yes |
| accepted prune | 684 | 0.96875 | 0.912566 | 1 | yes |
| cold restart 684->639 | 639 | 0.59375 | 1.24275 | 1.34967 | no |
| covariance shape 1/2 | 639 | 0.96875 | 1.30786 | 1.34493 | no |
| covariance shape 2/2 | 639 | 0.96875 | 1.30786 | 1.34493 | no |
| post-asinh transform 1/4 | 639 | 0.96875 | 1.30786 | 1.34493 | no |
| post-asinh transform 2/4 | 639 | 0.96875 | 1.30786 | 1.34493 | no |
| post-asinh transform 3/4 | 639 | 0.96875 | 1.30786 | 1.34493 | no |
| post-asinh transform 4/4 | 639 | 0.96875 | 1.30786 | 1.34493 | no |
| allocation beta | 639 | 0.84375 | 1.28806 | 1.30412 | no |
| knot refit | 639 | 0.84375 | 1.26809 | 1.26851 | no |
| covariance shape 1/2 | 639 | 0.84375 | 1.26809 | 1.26851 | no |
| covariance shape 2/2 | 639 | 0.84375 | 1.26809 | 1.26851 | no |
| post-asinh transform 1/4 | 639 | 0.84375 | 1.26809 | 1.26851 | no |
| post-asinh transform 2/4 | 639 | 0.84375 | 1.26809 | 1.26851 | no |
| post-asinh transform 3/4 | 639 | 0.84375 | 1.26809 | 1.26851 | no |
| post-asinh transform 4/4 | 639 | 0.84375 | 1.26809 | 1.26851 | no |
| allocation beta | 639 | 0.71875 | 1.25387 | 1.32967 | no |
| knot refit | 639 | 0.71875 | 1.25387 | 1.32967 | no |
| covariance shape 1/2 | 639 | 0.71875 | 1.25387 | 1.32967 | no |
| covariance shape 2/2 | 639 | 0.71875 | 1.25387 | 1.32967 | no |
| post-asinh transform 1/4 | 639 | 0.71875 | 1.25387 | 1.32967 | no |
| post-asinh transform 2/4 | 639 | 0.71875 | 1.25387 | 1.32967 | no |
| post-asinh transform 3/4 | 639 | 0.71875 | 1.25387 | 1.32967 | no |
| post-asinh transform 4/4 | 639 | 0.71875 | 1.25387 | 1.32967 | no |
| allocation beta | 639 | 0.59375 | 1.24275 | 1.34967 | no |
| knot refit | 639 | 0.59375 | 1.24275 | 1.34967 | no |
| covariance shape 1/2 | 639 | 0.59375 | 1.24275 | 1.34967 | no |
| covariance shape 2/2 | 639 | 0.59375 | 1.24275 | 1.34967 | no |
| post-asinh transform 1/4 | 639 | 0.59375 | 1.24275 | 1.34967 | no |
| post-asinh transform 2/4 | 639 | 0.59375 | 1.24275 | 1.34967 | no |
| post-asinh transform 3/4 | 639 | 0.59375 | 1.24275 | 1.34967 | no |
| post-asinh transform 4/4 | 639 | 0.59375 | 1.24275 | 1.34967 | no |
| allocation beta | 639 | 0.59375 | 1.24275 | 1.34967 | no |
| knot refit | 639 | 0.59375 | 1.24275 | 1.34967 | no |
| refined parameter-search resolution | 639 | 0.59375 | 1.24275 | 1.34967 | no |
| covariance shape 1/2 | 639 | 0.59375 | 1.24275 | 1.34967 | no |
| covariance shape 2/2 | 639 | 0.59375 | 1.24275 | 1.34967 | no |
| post-asinh transform 1/4 | 639 | 0.59375 | 1.24275 | 1.34967 | no |
| post-asinh transform 2/4 | 639 | 0.59375 | 1.24275 | 1.34967 | no |
| post-asinh transform 3/4 | 639 | 0.59375 | 1.24275 | 1.34967 | no |
| post-asinh transform 4/4 | 639 | 0.59375 | 1.24275 | 1.34967 | no |
| allocation beta | 639 | 0.59375 | 1.24275 | 1.34967 | no |
| knot refit | 639 | 0.59375 | 1.24275 | 1.34967 | no |
| refined parameter-search resolution | 639 | 0.59375 | 1.24275 | 1.34967 | no |
| covariance shape 1/2 | 639 | 0.59375 | 1.24275 | 1.34967 | no |
| covariance shape 2/2 | 639 | 0.59375 | 1.24275 | 1.34967 | no |
| post-asinh transform 1/4 | 639 | 0.59375 | 1.24275 | 1.34967 | no |
| post-asinh transform 2/4 | 639 | 0.59375 | 1.24275 | 1.34967 | no |
| post-asinh transform 3/4 | 639 | 0.59375 | 1.24275 | 1.34967 | no |
| post-asinh transform 4/4 | 639 | 0.59375 | 1.24275 | 1.34967 | no |
| allocation beta | 639 | 0.59375 | 1.24275 | 1.34967 | no |
| knot refit | 639 | 0.59375 | 1.24275 | 1.34967 | no |
| fixed point reached | 639 | 0.59375 | 1.24275 | 1.34967 | no |
| prune proposal 684->639 | 639 | 0.8125 | 1.20455 | 1.22677 | no |
| covariance shape 1/2 | 639 | 0.96875 | 1.37659 | 1.47304 | no |
| covariance shape 2/2 | 639 | 0.96875 | 1.37659 | 1.47304 | no |
| post-asinh transform 1/4 | 639 | 0.96875 | 1.33822 | 1.34259 | no |
| post-asinh transform 2/4 | 639 | 0.96875 | 1.33822 | 1.34259 | no |
| post-asinh transform 3/4 | 639 | 0.96875 | 1.33822 | 1.34259 | no |
| post-asinh transform 4/4 | 639 | 0.96875 | 1.33822 | 1.34259 | no |
| allocation beta | 639 | 0.84375 | 1.25805 | 1.28486 | no |
| knot refit | 639 | 0.84375 | 1.23871 | 1.24837 | no |
| covariance shape 1/2 | 639 | 0.84375 | 1.23871 | 1.24837 | no |
| covariance shape 2/2 | 639 | 0.84375 | 1.23871 | 1.24837 | no |
| post-asinh transform 1/4 | 639 | 0.84375 | 1.23871 | 1.24837 | no |
| post-asinh transform 2/4 | 639 | 0.84375 | 1.23871 | 1.24837 | no |
| post-asinh transform 3/4 | 639 | 0.84375 | 1.23871 | 1.24837 | no |
| post-asinh transform 4/4 | 639 | 0.84375 | 1.23871 | 1.24837 | no |
| allocation beta | 639 | 0.84375 | 1.23871 | 1.24837 | no |
| knot refit | 639 | 0.84375 | 1.23871 | 1.24837 | no |
| refined parameter-search resolution | 639 | 0.84375 | 1.23871 | 1.24837 | no |
| covariance shape 1/2 | 639 | 0.84375 | 1.23871 | 1.24837 | no |
| covariance shape 2/2 | 639 | 0.84375 | 1.23871 | 1.24837 | no |
| post-asinh transform 1/4 | 639 | 0.84375 | 1.23871 | 1.24837 | no |
| post-asinh transform 2/4 | 639 | 0.84375 | 1.23871 | 1.24837 | no |
| post-asinh transform 3/4 | 639 | 0.84375 | 1.23871 | 1.24837 | no |
| post-asinh transform 4/4 | 639 | 0.84375 | 1.23871 | 1.24837 | no |
| allocation beta | 639 | 0.84375 | 1.23871 | 1.24837 | no |
| knot refit | 639 | 0.84375 | 1.23871 | 1.24837 | no |
| refined parameter-search resolution | 639 | 0.84375 | 1.23871 | 1.24837 | no |
| covariance shape 1/2 | 639 | 0.84375 | 1.23871 | 1.24837 | no |
| covariance shape 2/2 | 639 | 0.84375 | 1.23871 | 1.24837 | no |
| post-asinh transform 1/4 | 639 | 0.84375 | 1.23871 | 1.24837 | no |
| post-asinh transform 2/4 | 639 | 0.84375 | 1.23871 | 1.24837 | no |
| post-asinh transform 3/4 | 639 | 0.84375 | 1.23871 | 1.24837 | no |
| post-asinh transform 4/4 | 639 | 0.84375 | 1.23871 | 1.24837 | no |
| allocation beta | 639 | 0.8125 | 1.20455 | 1.22677 | no |
| knot refit | 639 | 0.8125 | 1.20455 | 1.22677 | no |
| covariance shape 1/2 | 639 | 0.8125 | 1.20455 | 1.22677 | no |
| covariance shape 2/2 | 639 | 0.8125 | 1.20455 | 1.22677 | no |
| post-asinh transform 1/4 | 639 | 0.8125 | 1.20455 | 1.22677 | no |
| post-asinh transform 2/4 | 639 | 0.8125 | 1.20455 | 1.22677 | no |
| post-asinh transform 3/4 | 639 | 0.8125 | 1.20455 | 1.22677 | no |
| post-asinh transform 4/4 | 639 | 0.8125 | 1.20455 | 1.22677 | no |
| allocation beta | 639 | 0.8125 | 1.20455 | 1.22677 | no |
| knot refit | 639 | 0.8125 | 1.20455 | 1.22677 | no |
| fixed point reached | 639 | 0.8125 | 1.20455 | 1.22677 | no |
| rejected prune | 639 | 0.8125 | 1.20455 | 1.22677 | no |
| prune proposal 684->662 | 662 | 1 | 0.87096 | 1 | yes |
| covariance shape 1/2 | 662 | 0.96875 | 0.916537 | 1 | yes |
| covariance shape 2/2 | 662 | 0.96875 | 0.916537 | 1 | yes |
| post-asinh transform 1/4 | 662 | 0.96875 | 0.916537 | 1 | yes |
| post-asinh transform 2/4 | 662 | 0.96875 | 0.916537 | 1 | yes |
| post-asinh transform 3/4 | 662 | 0.96875 | 0.916537 | 1 | yes |
| post-asinh transform 4/4 | 662 | 0.96875 | 0.916537 | 1 | yes |
| allocation beta | 662 | 1 | 0.87096 | 1 | yes |
| knot refit | 662 | 1 | 0.87096 | 1 | yes |
| covariance shape 1/2 | 662 | 1 | 0.87096 | 1 | yes |
| covariance shape 2/2 | 662 | 1 | 0.87096 | 1 | yes |
| post-asinh transform 1/4 | 662 | 1 | 0.87096 | 1 | yes |
| post-asinh transform 2/4 | 662 | 1 | 0.87096 | 1 | yes |
| post-asinh transform 3/4 | 662 | 1 | 0.87096 | 1 | yes |
| post-asinh transform 4/4 | 662 | 1 | 0.87096 | 1 | yes |
| allocation beta | 662 | 1 | 0.87096 | 1 | yes |
| knot refit | 662 | 1 | 0.87096 | 1 | yes |
| refined parameter-search resolution | 662 | 1 | 0.87096 | 1 | yes |
| covariance shape 1/2 | 662 | 1 | 0.87096 | 1 | yes |
| covariance shape 2/2 | 662 | 1 | 0.87096 | 1 | yes |
| post-asinh transform 1/4 | 662 | 1 | 0.87096 | 1 | yes |
| post-asinh transform 2/4 | 662 | 1 | 0.87096 | 1 | yes |
| post-asinh transform 3/4 | 662 | 1 | 0.87096 | 1 | yes |
| post-asinh transform 4/4 | 662 | 1 | 0.87096 | 1 | yes |
| allocation beta | 662 | 1 | 0.87096 | 1 | yes |
| knot refit | 662 | 1 | 0.87096 | 1 | yes |
| refined parameter-search resolution | 662 | 1 | 0.87096 | 1 | yes |
| covariance shape 1/2 | 662 | 1 | 0.87096 | 1 | yes |
| covariance shape 2/2 | 662 | 1 | 0.87096 | 1 | yes |
| post-asinh transform 1/4 | 662 | 1 | 0.87096 | 1 | yes |
| post-asinh transform 2/4 | 662 | 1 | 0.87096 | 1 | yes |
| post-asinh transform 3/4 | 662 | 1 | 0.87096 | 1 | yes |
| post-asinh transform 4/4 | 662 | 1 | 0.87096 | 1 | yes |
| allocation beta | 662 | 1 | 0.87096 | 1 | yes |
| knot refit | 662 | 1 | 0.87096 | 1 | yes |
| fixed point reached | 662 | 1 | 0.87096 | 1 | yes |
| accepted prune | 662 | 1 | 0.87096 | 1 | yes |
| prune proposal 662->640 | 640 | 0.875 | 0.894046 | 1 | yes |
| covariance shape 1/2 | 640 | 1 | 0.901631 | 1 | yes |
| covariance shape 2/2 | 640 | 1 | 0.901631 | 1 | yes |
| post-asinh transform 1/4 | 640 | 1 | 0.901631 | 1 | yes |
| post-asinh transform 2/4 | 640 | 1 | 0.901631 | 1 | yes |
| post-asinh transform 3/4 | 640 | 1 | 0.901631 | 1 | yes |
| post-asinh transform 4/4 | 640 | 1 | 0.901631 | 1 | yes |
| allocation beta | 640 | 0.875 | 0.894046 | 1 | yes |
| knot refit | 640 | 0.875 | 0.894046 | 1 | yes |
| covariance shape 1/2 | 640 | 0.875 | 0.894046 | 1 | yes |
| covariance shape 2/2 | 640 | 0.875 | 0.894046 | 1 | yes |
| post-asinh transform 1/4 | 640 | 0.875 | 0.894046 | 1 | yes |
| post-asinh transform 2/4 | 640 | 0.875 | 0.894046 | 1 | yes |
| post-asinh transform 3/4 | 640 | 0.875 | 0.894046 | 1 | yes |
| post-asinh transform 4/4 | 640 | 0.875 | 0.894046 | 1 | yes |
| allocation beta | 640 | 0.875 | 0.894046 | 1 | yes |
| knot refit | 640 | 0.875 | 0.894046 | 1 | yes |
| refined parameter-search resolution | 640 | 0.875 | 0.894046 | 1 | yes |
| covariance shape 1/2 | 640 | 0.875 | 0.894046 | 1 | yes |
| covariance shape 2/2 | 640 | 0.875 | 0.894046 | 1 | yes |
| post-asinh transform 1/4 | 640 | 0.875 | 0.894046 | 1 | yes |
| post-asinh transform 2/4 | 640 | 0.875 | 0.894046 | 1 | yes |
| post-asinh transform 3/4 | 640 | 0.875 | 0.894046 | 1 | yes |
| post-asinh transform 4/4 | 640 | 0.875 | 0.894046 | 1 | yes |
| allocation beta | 640 | 0.875 | 0.894046 | 1 | yes |
| knot refit | 640 | 0.875 | 0.894046 | 1 | yes |
| refined parameter-search resolution | 640 | 0.875 | 0.894046 | 1 | yes |
| covariance shape 1/2 | 640 | 0.875 | 0.894046 | 1 | yes |
| covariance shape 2/2 | 640 | 0.875 | 0.894046 | 1 | yes |
| post-asinh transform 1/4 | 640 | 0.875 | 0.894046 | 1 | yes |
| post-asinh transform 2/4 | 640 | 0.875 | 0.894046 | 1 | yes |
| post-asinh transform 3/4 | 640 | 0.875 | 0.894046 | 1 | yes |
| post-asinh transform 4/4 | 640 | 0.875 | 0.894046 | 1 | yes |
| allocation beta | 640 | 0.875 | 0.894046 | 1 | yes |
| knot refit | 640 | 0.875 | 0.894046 | 1 | yes |
| fixed point reached | 640 | 0.875 | 0.894046 | 1 | yes |
| accepted prune | 640 | 0.875 | 0.894046 | 1 | yes |
| cold restart 640->618 | 618 | 1 | 1.05207 | 1.11578 | no |
| covariance shape 1/2 | 618 | 0.875 | 1.44702 | 1.48024 | no |
| covariance shape 2/2 | 618 | 0.875 | 1.44702 | 1.48024 | no |
| post-asinh transform 1/4 | 618 | 0.875 | 1.44702 | 1.48024 | no |
| post-asinh transform 2/4 | 618 | 0.875 | 1.44702 | 1.48024 | no |
| post-asinh transform 3/4 | 618 | 0.875 | 1.44702 | 1.48024 | no |
| post-asinh transform 4/4 | 618 | 0.875 | 1.44702 | 1.48024 | no |
| allocation beta | 618 | 1 | 1.3616 | 1.44996 | no |
| knot refit | 618 | 1 | 1.32131 | 1.40364 | no |
| covariance shape 1/2 | 618 | 1 | 1.32131 | 1.40364 | no |
| covariance shape 2/2 | 618 | 1 | 1.32131 | 1.40364 | no |
| post-asinh transform 1/4 | 618 | 1 | 1.23211 | 1.30791 | no |
| post-asinh transform 2/4 | 618 | 1 | 1.23211 | 1.30791 | no |
| post-asinh transform 3/4 | 618 | 1 | 1.23211 | 1.30791 | no |
| post-asinh transform 4/4 | 618 | 1 | 1.23211 | 1.30791 | no |
| allocation beta | 618 | 0.875 | 1.16608 | 1.17597 | no |
| knot refit | 618 | 0.875 | 1.16608 | 1.17597 | no |
| covariance shape 1/2 | 618 | 0.875 | 1.16608 | 1.17597 | no |
| covariance shape 2/2 | 618 | 0.875 | 1.16608 | 1.17597 | no |
| post-asinh transform 1/4 | 618 | 0.875 | 1.16608 | 1.17597 | no |
| post-asinh transform 2/4 | 618 | 0.875 | 1.16608 | 1.17597 | no |
| post-asinh transform 3/4 | 618 | 0.875 | 1.16608 | 1.17597 | no |
| post-asinh transform 4/4 | 618 | 0.875 | 1.16608 | 1.17597 | no |
| allocation beta | 618 | 1 | 1.06243 | 1.09492 | no |
| knot refit | 618 | 1 | 1.05207 | 1.11578 | no |
| covariance shape 1/2 | 618 | 1 | 1.05207 | 1.11578 | no |
| covariance shape 2/2 | 618 | 1 | 1.05207 | 1.11578 | no |
| post-asinh transform 1/4 | 618 | 1 | 1.05207 | 1.11578 | no |
| post-asinh transform 2/4 | 618 | 1 | 1.05207 | 1.11578 | no |
| post-asinh transform 3/4 | 618 | 1 | 1.05207 | 1.11578 | no |
| post-asinh transform 4/4 | 618 | 1 | 1.05207 | 1.11578 | no |
| allocation beta | 618 | 1 | 1.05207 | 1.11578 | no |
| knot refit | 618 | 1 | 1.05207 | 1.11578 | no |
| refined parameter-search resolution | 618 | 1 | 1.05207 | 1.11578 | no |
| covariance shape 1/2 | 618 | 1 | 1.05207 | 1.11578 | no |
| covariance shape 2/2 | 618 | 1 | 1.05207 | 1.11578 | no |
| post-asinh transform 1/4 | 618 | 1 | 1.05207 | 1.11578 | no |
| post-asinh transform 2/4 | 618 | 1 | 1.05207 | 1.11578 | no |
| post-asinh transform 3/4 | 618 | 1 | 1.05207 | 1.11578 | no |
| post-asinh transform 4/4 | 618 | 1 | 1.05207 | 1.11578 | no |
| allocation beta | 618 | 1 | 1.05207 | 1.11578 | no |
| knot refit | 618 | 1 | 1.05207 | 1.11578 | no |
| refined parameter-search resolution | 618 | 1 | 1.05207 | 1.11578 | no |
| covariance shape 1/2 | 618 | 1 | 1.05207 | 1.11578 | no |
| covariance shape 2/2 | 618 | 1 | 1.05207 | 1.11578 | no |
| post-asinh transform 1/4 | 618 | 1 | 1.05207 | 1.11578 | no |
| post-asinh transform 2/4 | 618 | 1 | 1.05207 | 1.11578 | no |
| post-asinh transform 3/4 | 618 | 1 | 1.05207 | 1.11578 | no |
| post-asinh transform 4/4 | 618 | 1 | 1.05207 | 1.11578 | no |
| allocation beta | 618 | 1 | 1.05207 | 1.11578 | no |
| knot refit | 618 | 1 | 1.05207 | 1.11578 | no |
| fixed point reached | 618 | 1 | 1.05207 | 1.11578 | no |
| prune proposal 640->618 | 618 | 1 | 1.05207 | 1.11578 | no |
| covariance shape 1/2 | 618 | 0.875 | 1.48237 | 1.53383 | no |
| covariance shape 2/2 | 618 | 0.875 | 1.48237 | 1.53383 | no |
| post-asinh transform 1/4 | 618 | 0.875 | 1.48237 | 1.53383 | no |
| post-asinh transform 2/4 | 618 | 0.875 | 1.48237 | 1.53383 | no |
| post-asinh transform 3/4 | 618 | 0.875 | 1.48237 | 1.53383 | no |
| post-asinh transform 4/4 | 618 | 0.875 | 1.48237 | 1.53383 | no |
| allocation beta | 618 | 1 | 1.45975 | 1.46627 | no |
| knot refit | 618 | 1 | 1.40666 | 1.44237 | no |
| covariance shape 1/2 | 618 | 1 | 1.40666 | 1.44237 | no |
| covariance shape 2/2 | 618 | 1 | 1.40666 | 1.44237 | no |
| post-asinh transform 1/4 | 618 | 1 | 1.40666 | 1.44237 | no |
| post-asinh transform 2/4 | 618 | 1 | 1.40666 | 1.44237 | no |
| post-asinh transform 3/4 | 618 | 1 | 1.40666 | 1.44237 | no |
| post-asinh transform 4/4 | 618 | 1 | 1.40666 | 1.44237 | no |
| allocation beta | 618 | 1 | 1.40666 | 1.44237 | no |
| knot refit | 618 | 1 | 1.36857 | 1.41651 | no |
| covariance shape 1/2 | 618 | 1 | 1.36857 | 1.41651 | no |
| covariance shape 2/2 | 618 | 1 | 1.36857 | 1.41651 | no |
| post-asinh transform 1/4 | 618 | 1 | 1.36857 | 1.41651 | no |
| post-asinh transform 2/4 | 618 | 1 | 1.36857 | 1.41651 | no |
| post-asinh transform 3/4 | 618 | 1 | 1.36857 | 1.41651 | no |
| post-asinh transform 4/4 | 618 | 1 | 1.36857 | 1.41651 | no |
| allocation beta | 618 | 1 | 1.36857 | 1.41651 | no |
| knot refit | 618 | 1 | 1.36067 | 1.41815 | no |
| covariance shape 1/2 | 618 | 1 | 1.36067 | 1.41815 | no |
| covariance shape 2/2 | 618 | 1 | 1.36067 | 1.41815 | no |
| post-asinh transform 1/4 | 618 | 1 | 1.36067 | 1.41815 | no |
| post-asinh transform 2/4 | 618 | 1 | 1.36067 | 1.41815 | no |
| post-asinh transform 3/4 | 618 | 1 | 1.36067 | 1.41815 | no |
| post-asinh transform 4/4 | 618 | 1 | 1.36067 | 1.41815 | no |
| allocation beta | 618 | 1 | 1.36067 | 1.41815 | no |
| knot refit | 618 | 1 | 1.36067 | 1.41815 | no |
| refined parameter-search resolution | 618 | 1 | 1.36067 | 1.41815 | no |
| covariance shape 1/2 | 618 | 1 | 1.36067 | 1.41815 | no |
| covariance shape 2/2 | 618 | 1 | 1.36067 | 1.41815 | no |
| post-asinh transform 1/4 | 618 | 1 | 1.36067 | 1.41815 | no |
| post-asinh transform 2/4 | 618 | 1 | 1.36067 | 1.41815 | no |
| post-asinh transform 3/4 | 618 | 1 | 1.36067 | 1.41815 | no |
| post-asinh transform 4/4 | 618 | 1 | 1.36067 | 1.41815 | no |
| allocation beta | 618 | 1 | 1.36067 | 1.41815 | no |
| knot refit | 618 | 1 | 1.36067 | 1.41815 | no |
| refined parameter-search resolution | 618 | 1 | 1.36067 | 1.41815 | no |
| covariance shape 1/2 | 618 | 1 | 1.36067 | 1.41815 | no |
| covariance shape 2/2 | 618 | 1 | 1.36067 | 1.41815 | no |
| post-asinh transform 1/4 | 618 | 1 | 1.36067 | 1.41815 | no |
| post-asinh transform 2/4 | 618 | 1 | 1.36067 | 1.41815 | no |
| post-asinh transform 3/4 | 618 | 1 | 1.36067 | 1.41815 | no |
| post-asinh transform 4/4 | 618 | 1 | 1.36067 | 1.41815 | no |
| allocation beta | 618 | 1 | 1.36067 | 1.41815 | no |
| knot refit | 618 | 1 | 1.36067 | 1.41815 | no |
| fixed point reached | 618 | 1 | 1.36067 | 1.41815 | no |
| rejected prune | 618 | 1 | 1.05207 | 1.11578 | no |
| prune proposal 640->629 | 629 | 1 | 0.799143 | 1 | yes |
| covariance shape 1/2 | 629 | 0.875 | 2.22204 | 2.41735 | no |
| covariance shape 2/2 | 629 | 0.875 | 2.22204 | 2.41735 | no |
| post-asinh transform 1/4 | 629 | 0.875 | 2.22204 | 2.41735 | no |
| post-asinh transform 2/4 | 629 | 0.875 | 2.22204 | 2.41735 | no |
| post-asinh transform 3/4 | 629 | 0.875 | 2.22204 | 2.41735 | no |
| post-asinh transform 4/4 | 629 | 0.875 | 2.22204 | 2.41735 | no |
| allocation beta | 629 | 1 | 1.03569 | 1.0878 | no |
| knot refit | 629 | 1 | 0.799143 | 1 | yes |
| covariance shape 1/2 | 629 | 1 | 0.799143 | 1 | yes |
| covariance shape 2/2 | 629 | 1 | 0.799143 | 1 | yes |
| post-asinh transform 1/4 | 629 | 1 | 0.799143 | 1 | yes |
| post-asinh transform 2/4 | 629 | 1 | 0.799143 | 1 | yes |
| post-asinh transform 3/4 | 629 | 1 | 0.799143 | 1 | yes |
| post-asinh transform 4/4 | 629 | 1 | 0.799143 | 1 | yes |
| allocation beta | 629 | 1 | 0.799143 | 1 | yes |
| knot refit | 629 | 1 | 0.799143 | 1 | yes |
| refined parameter-search resolution | 629 | 1 | 0.799143 | 1 | yes |
| covariance shape 1/2 | 629 | 1 | 0.799143 | 1 | yes |
| covariance shape 2/2 | 629 | 1 | 0.799143 | 1 | yes |
| post-asinh transform 1/4 | 629 | 1 | 0.799143 | 1 | yes |
| post-asinh transform 2/4 | 629 | 1 | 0.799143 | 1 | yes |
| post-asinh transform 3/4 | 629 | 1 | 0.799143 | 1 | yes |
| post-asinh transform 4/4 | 629 | 1 | 0.799143 | 1 | yes |
| allocation beta | 629 | 1 | 0.799143 | 1 | yes |
| knot refit | 629 | 1 | 0.799143 | 1 | yes |
| refined parameter-search resolution | 629 | 1 | 0.799143 | 1 | yes |
| covariance shape 1/2 | 629 | 1 | 0.799143 | 1 | yes |
| covariance shape 2/2 | 629 | 1 | 0.799143 | 1 | yes |
| post-asinh transform 1/4 | 629 | 1 | 0.799143 | 1 | yes |
| post-asinh transform 2/4 | 629 | 1 | 0.799143 | 1 | yes |
| post-asinh transform 3/4 | 629 | 1 | 0.799143 | 1 | yes |
| post-asinh transform 4/4 | 629 | 1 | 0.799143 | 1 | yes |
| allocation beta | 629 | 1 | 0.799143 | 1 | yes |
| knot refit | 629 | 1 | 0.799143 | 1 | yes |
| fixed point reached | 629 | 1 | 0.799143 | 1 | yes |
| accepted prune | 629 | 1 | 0.799143 | 1 | yes |
| prune proposal 629->618 | 618 | 1 | 0.872906 | 1 | yes |
| covariance shape 1/2 | 618 | 1 | 1.21611 | 1.29497 | no |
| covariance shape 2/2 | 618 | 1 | 1.21611 | 1.29497 | no |
| post-asinh transform 1/4 | 618 | 1 | 1.21611 | 1.29497 | no |
| post-asinh transform 2/4 | 618 | 1 | 1.21611 | 1.29497 | no |
| post-asinh transform 3/4 | 618 | 1 | 1.21611 | 1.29497 | no |
| post-asinh transform 4/4 | 618 | 1 | 1.21611 | 1.29497 | no |
| allocation beta | 618 | 0.875 | 1.02011 | 1.05115 | no |
| knot refit | 618 | 0.875 | 1.02011 | 1.05115 | no |
| covariance shape 1/2 | 618 | 0.875 | 1.02011 | 1.05115 | no |
| covariance shape 2/2 | 618 | 0.875 | 1.02011 | 1.05115 | no |
| post-asinh transform 1/4 | 618 | 0.875 | 1.02011 | 1.05115 | no |
| post-asinh transform 2/4 | 618 | 0.875 | 1.02011 | 1.05115 | no |
| post-asinh transform 3/4 | 618 | 0.875 | 1.02011 | 1.05115 | no |
| post-asinh transform 4/4 | 618 | 0.875 | 1.02011 | 1.05115 | no |
| allocation beta | 618 | 1 | 0.872906 | 1 | yes |
| knot refit | 618 | 1 | 0.872906 | 1 | yes |
| covariance shape 1/2 | 618 | 1 | 0.872906 | 1 | yes |
| covariance shape 2/2 | 618 | 1 | 0.872906 | 1 | yes |
| post-asinh transform 1/4 | 618 | 1 | 0.872906 | 1 | yes |
| post-asinh transform 2/4 | 618 | 1 | 0.872906 | 1 | yes |
| post-asinh transform 3/4 | 618 | 1 | 0.872906 | 1 | yes |
| post-asinh transform 4/4 | 618 | 1 | 0.872906 | 1 | yes |
| allocation beta | 618 | 1 | 0.872906 | 1 | yes |
| knot refit | 618 | 1 | 0.872906 | 1 | yes |
| refined parameter-search resolution | 618 | 1 | 0.872906 | 1 | yes |
| covariance shape 1/2 | 618 | 1 | 0.872906 | 1 | yes |
| covariance shape 2/2 | 618 | 1 | 0.872906 | 1 | yes |
| post-asinh transform 1/4 | 618 | 1 | 0.872906 | 1 | yes |
| post-asinh transform 2/4 | 618 | 1 | 0.872906 | 1 | yes |
| post-asinh transform 3/4 | 618 | 1 | 0.872906 | 1 | yes |
| post-asinh transform 4/4 | 618 | 1 | 0.872906 | 1 | yes |
| allocation beta | 618 | 1 | 0.872906 | 1 | yes |
| knot refit | 618 | 1 | 0.872906 | 1 | yes |
| refined parameter-search resolution | 618 | 1 | 0.872906 | 1 | yes |
| covariance shape 1/2 | 618 | 1 | 0.872906 | 1 | yes |
| covariance shape 2/2 | 618 | 1 | 0.872906 | 1 | yes |
| post-asinh transform 1/4 | 618 | 1 | 0.872906 | 1 | yes |
| post-asinh transform 2/4 | 618 | 1 | 0.872906 | 1 | yes |
| post-asinh transform 3/4 | 618 | 1 | 0.872906 | 1 | yes |
| post-asinh transform 4/4 | 618 | 1 | 0.872906 | 1 | yes |
| allocation beta | 618 | 1 | 0.872906 | 1 | yes |
| knot refit | 618 | 1 | 0.872906 | 1 | yes |
| fixed point reached | 618 | 1 | 0.872906 | 1 | yes |
| accepted prune | 618 | 1 | 0.872906 | 1 | yes |

Covariance adjustment from empirical whitening: `[[1.0,0.0],[0.0,1.0]]`

Post-asinh matrix: `[[3.153724060260527,0.0],[0.0,3.150166233004714]]`

## 3D

Initial: 32,768 points, beta 1, active score 1, joint objective 0.822184.

Final: **31,360 points**, beta **1**, active score **1**, joint objective **0.823976**, passes.

### Fixed-point checks

| points | sweeps | converged | covariance step | transform step | beta step |
|---:|---:|---:|---:|---:|---:|
| 32,768 | 5 | yes | 0.0125 | 0.01125 | 0.025 |
| 28,672 | 4 | yes | 0.0125 | 0.01125 | 0.025 |
| 28,672 | 4 | yes | 0.0125 | 0.01125 | 0.025 |
| 30,720 | 4 | yes | 0.0125 | 0.01125 | 0.025 |
| 30,720 | 3 | yes | 0.0125 | 0.01125 | 0.025 |
| 31,744 | 3 | yes | 0.0125 | 0.01125 | 0.025 |
| 30,720 | 6 | yes | 0.0125 | 0.01125 | 0.025 |
| 30,720 | 3 | yes | 0.0125 | 0.01125 | 0.025 |
| 31,232 | 3 | yes | 0.0125 | 0.01125 | 0.025 |
| 31,232 | 5 | yes | 0.0125 | 0.01125 | 0.025 |
| 31,488 | 3 | yes | 0.0125 | 0.01125 | 0.025 |
| 31,232 | 4 | yes | 0.0125 | 0.01125 | 0.025 |
| 31,232 | 5 | yes | 0.0125 | 0.01125 | 0.025 |
| 31,360 | 3 | yes | 0.0125 | 0.01125 | 0.025 |
| 31,232 | 3 | yes | 0.0125 | 0.01125 | 0.025 |
| 31,232 | 6 | yes | 0.0125 | 0.01125 | 0.025 |
| 31,296 | 6 | yes | 0.0125 | 0.01125 | 0.025 |
| 31,232 | 6 | yes | 0.0125 | 0.01125 | 0.025 |
| 31,232 | 5 | yes | 0.0125 | 0.01125 | 0.025 |

| metric | ratio to 1D32 | active | passes |
|---|---:|---:|---:|
| density JS/d | 0.171328 | yes | yes |
| r2 conditional mean | 5.83969 | no | no |
| r2 conditional median | 0.0813732 | yes | yes |
| r3 conditional mean | 0.979876 | yes | yes |
| r3 conditional median | 0.243949 | yes | yes |

### Accepted/rejected iteration trace

| stage | points | beta | objective | active score | passes |
|---|---:|---:|---:|---:|---:|
| stored verified initial | 32,768 | 1 | 0.822184 | 1 | yes |
| common-fidelity warm refit | 32,768 | 1 | 0.771549 | 1 | yes |
| covariance shape 1/5 | 32,768 | 1 | 0.771549 | 1 | yes |
| covariance shape 2/5 | 32,768 | 1 | 0.771549 | 1 | yes |
| covariance shape 3/5 | 32,768 | 1 | 0.771549 | 1 | yes |
| covariance shape 4/5 | 32,768 | 1 | 0.771549 | 1 | yes |
| covariance shape 5/5 | 32,768 | 1 | 0.771549 | 1 | yes |
| post-asinh transform 1/9 | 32,768 | 1 | 0.771549 | 1 | yes |
| post-asinh transform 2/9 | 32,768 | 1 | 0.771549 | 1 | yes |
| post-asinh transform 3/9 | 32,768 | 1 | 0.771549 | 1 | yes |
| post-asinh transform 4/9 | 32,768 | 1 | 0.771549 | 1 | yes |
| post-asinh transform 5/9 | 32,768 | 1 | 0.771549 | 1 | yes |
| post-asinh transform 6/9 | 32,768 | 1 | 0.771549 | 1 | yes |
| post-asinh transform 7/9 | 32,768 | 1 | 0.771549 | 1 | yes |
| post-asinh transform 8/9 | 32,768 | 1 | 0.771549 | 1 | yes |
| post-asinh transform 9/9 | 32,768 | 1 | 0.771549 | 1 | yes |
| allocation beta | 32,768 | 0.9 | 0.709175 | 1 | yes |
| knot refit | 32,768 | 0.9 | 0.709175 | 1 | yes |
| covariance shape 1/5 | 32,768 | 0.9 | 0.709175 | 1 | yes |
| covariance shape 2/5 | 32,768 | 0.9 | 0.709175 | 1 | yes |
| covariance shape 3/5 | 32,768 | 0.9 | 0.709175 | 1 | yes |
| covariance shape 4/5 | 32,768 | 0.9 | 0.709175 | 1 | yes |
| covariance shape 5/5 | 32,768 | 0.9 | 0.709175 | 1 | yes |
| post-asinh transform 1/9 | 32,768 | 0.9 | 0.709175 | 1 | yes |
| post-asinh transform 2/9 | 32,768 | 0.9 | 0.709175 | 1 | yes |
| post-asinh transform 3/9 | 32,768 | 0.9 | 0.709175 | 1 | yes |
| post-asinh transform 4/9 | 32,768 | 0.9 | 0.709175 | 1 | yes |
| post-asinh transform 5/9 | 32,768 | 0.9 | 0.709175 | 1 | yes |
| post-asinh transform 6/9 | 32,768 | 0.9 | 0.709175 | 1 | yes |
| post-asinh transform 7/9 | 32,768 | 0.9 | 0.709175 | 1 | yes |
| post-asinh transform 8/9 | 32,768 | 0.9 | 0.709175 | 1 | yes |
| post-asinh transform 9/9 | 32,768 | 0.9 | 0.709175 | 1 | yes |
| allocation beta | 32,768 | 1 | 0.69944 | 1 | yes |
| knot refit | 32,768 | 1 | 0.669942 | 1 | yes |
| covariance shape 1/5 | 32,768 | 1 | 0.669942 | 1 | yes |
| covariance shape 2/5 | 32,768 | 1 | 0.669942 | 1 | yes |
| covariance shape 3/5 | 32,768 | 1 | 0.669942 | 1 | yes |
| covariance shape 4/5 | 32,768 | 1 | 0.669942 | 1 | yes |
| covariance shape 5/5 | 32,768 | 1 | 0.669942 | 1 | yes |
| post-asinh transform 1/9 | 32,768 | 1 | 0.669942 | 1 | yes |
| post-asinh transform 2/9 | 32,768 | 1 | 0.669942 | 1 | yes |
| post-asinh transform 3/9 | 32,768 | 1 | 0.669942 | 1 | yes |
| post-asinh transform 4/9 | 32,768 | 1 | 0.669942 | 1 | yes |
| post-asinh transform 5/9 | 32,768 | 1 | 0.669942 | 1 | yes |
| post-asinh transform 6/9 | 32,768 | 1 | 0.669942 | 1 | yes |
| post-asinh transform 7/9 | 32,768 | 1 | 0.669942 | 1 | yes |
| post-asinh transform 8/9 | 32,768 | 1 | 0.669942 | 1 | yes |
| post-asinh transform 9/9 | 32,768 | 1 | 0.669942 | 1 | yes |
| allocation beta | 32,768 | 1 | 0.669942 | 1 | yes |
| knot refit | 32,768 | 1 | 0.669942 | 1 | yes |
| refined parameter-search resolution | 32,768 | 1 | 0.669942 | 1 | yes |
| covariance shape 1/5 | 32,768 | 1 | 0.669942 | 1 | yes |
| covariance shape 2/5 | 32,768 | 1 | 0.669942 | 1 | yes |
| covariance shape 3/5 | 32,768 | 1 | 0.669942 | 1 | yes |
| covariance shape 4/5 | 32,768 | 1 | 0.669942 | 1 | yes |
| covariance shape 5/5 | 32,768 | 1 | 0.669942 | 1 | yes |
| post-asinh transform 1/9 | 32,768 | 1 | 0.669942 | 1 | yes |
| post-asinh transform 2/9 | 32,768 | 1 | 0.669942 | 1 | yes |
| post-asinh transform 3/9 | 32,768 | 1 | 0.669942 | 1 | yes |
| post-asinh transform 4/9 | 32,768 | 1 | 0.669942 | 1 | yes |
| post-asinh transform 5/9 | 32,768 | 1 | 0.669942 | 1 | yes |
| post-asinh transform 6/9 | 32,768 | 1 | 0.669942 | 1 | yes |
| post-asinh transform 7/9 | 32,768 | 1 | 0.669942 | 1 | yes |
| post-asinh transform 8/9 | 32,768 | 1 | 0.669942 | 1 | yes |
| post-asinh transform 9/9 | 32,768 | 1 | 0.669942 | 1 | yes |
| allocation beta | 32,768 | 1 | 0.669942 | 1 | yes |
| knot refit (cached stationary) | 32,768 | 1 | 0.669942 | 1 | yes |
| refined parameter-search resolution | 32,768 | 1 | 0.669942 | 1 | yes |
| covariance shape 1/5 | 32,768 | 1 | 0.669942 | 1 | yes |
| covariance shape 2/5 | 32,768 | 1 | 0.669942 | 1 | yes |
| covariance shape 3/5 | 32,768 | 1 | 0.669942 | 1 | yes |
| covariance shape 4/5 | 32,768 | 1 | 0.669942 | 1 | yes |
| covariance shape 5/5 | 32,768 | 1 | 0.669942 | 1 | yes |
| post-asinh transform 1/9 | 32,768 | 1 | 0.669942 | 1 | yes |
| post-asinh transform 2/9 | 32,768 | 1 | 0.669942 | 1 | yes |
| post-asinh transform 3/9 | 32,768 | 1 | 0.669942 | 1 | yes |
| post-asinh transform 4/9 | 32,768 | 1 | 0.669942 | 1 | yes |
| post-asinh transform 5/9 | 32,768 | 1 | 0.669942 | 1 | yes |
| post-asinh transform 6/9 | 32,768 | 1 | 0.669942 | 1 | yes |
| post-asinh transform 7/9 | 32,768 | 1 | 0.669942 | 1 | yes |
| post-asinh transform 8/9 | 32,768 | 1 | 0.669942 | 1 | yes |
| post-asinh transform 9/9 | 32,768 | 1 | 0.669942 | 1 | yes |
| allocation beta | 32,768 | 1 | 0.669942 | 1 | yes |
| knot refit (cached stationary) | 32,768 | 1 | 0.669942 | 1 | yes |
| fixed point reached | 32,768 | 1 | 0.669942 | 1 | yes |
| cold restart 32768->28672 | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| covariance shape 1/5 | 28,672 | 1 | 0.928105 | 1.10277 | no |
| covariance shape 2/5 | 28,672 | 1 | 0.928105 | 1.10277 | no |
| covariance shape 3/5 | 28,672 | 1 | 0.928105 | 1.10277 | no |
| covariance shape 4/5 | 28,672 | 1 | 0.928105 | 1.10277 | no |
| covariance shape 5/5 | 28,672 | 1 | 0.928105 | 1.10277 | no |
| post-asinh transform 1/9 | 28,672 | 1 | 0.928105 | 1.10277 | no |
| post-asinh transform 2/9 | 28,672 | 1 | 0.928105 | 1.10277 | no |
| post-asinh transform 3/9 | 28,672 | 1 | 0.928105 | 1.10277 | no |
| post-asinh transform 4/9 | 28,672 | 1 | 0.928105 | 1.10277 | no |
| post-asinh transform 5/9 | 28,672 | 1 | 0.928105 | 1.10277 | no |
| post-asinh transform 6/9 | 28,672 | 1 | 0.928105 | 1.10277 | no |
| post-asinh transform 7/9 | 28,672 | 1 | 0.928105 | 1.10277 | no |
| post-asinh transform 8/9 | 28,672 | 1 | 0.928105 | 1.10277 | no |
| post-asinh transform 9/9 | 28,672 | 1 | 0.928105 | 1.10277 | no |
| allocation beta | 28,672 | 1 | 0.928105 | 1.10277 | no |
| knot refit | 28,672 | 1 | 0.928105 | 1.10277 | no |
| refined parameter-search resolution | 28,672 | 1 | 0.928105 | 1.10277 | no |
| covariance shape 1/5 | 28,672 | 1 | 0.928105 | 1.10277 | no |
| covariance shape 2/5 | 28,672 | 1 | 0.928105 | 1.10277 | no |
| covariance shape 3/5 | 28,672 | 1 | 0.928105 | 1.10277 | no |
| covariance shape 4/5 | 28,672 | 1 | 0.928105 | 1.10277 | no |
| covariance shape 5/5 | 28,672 | 1 | 0.928105 | 1.10277 | no |
| post-asinh transform 1/9 | 28,672 | 1 | 0.928105 | 1.10277 | no |
| post-asinh transform 2/9 | 28,672 | 1 | 0.928105 | 1.10277 | no |
| post-asinh transform 3/9 | 28,672 | 1 | 0.928105 | 1.10277 | no |
| post-asinh transform 4/9 | 28,672 | 1 | 0.928105 | 1.10277 | no |
| post-asinh transform 5/9 | 28,672 | 1 | 0.928105 | 1.10277 | no |
| post-asinh transform 6/9 | 28,672 | 1 | 0.928105 | 1.10277 | no |
| post-asinh transform 7/9 | 28,672 | 1 | 0.928105 | 1.10277 | no |
| post-asinh transform 8/9 | 28,672 | 1 | 0.928105 | 1.10277 | no |
| post-asinh transform 9/9 | 28,672 | 1 | 0.928105 | 1.10277 | no |
| allocation beta | 28,672 | 0.95 | 0.92514 | 1.09934 | no |
| knot refit | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| covariance shape 1/5 | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| covariance shape 2/5 | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| covariance shape 3/5 | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| covariance shape 4/5 | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| covariance shape 5/5 | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| post-asinh transform 1/9 | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| post-asinh transform 2/9 | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| post-asinh transform 3/9 | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| post-asinh transform 4/9 | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| post-asinh transform 5/9 | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| post-asinh transform 6/9 | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| post-asinh transform 7/9 | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| post-asinh transform 8/9 | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| post-asinh transform 9/9 | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| allocation beta | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| knot refit | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| refined parameter-search resolution | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| covariance shape 1/5 | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| covariance shape 2/5 | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| covariance shape 3/5 | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| covariance shape 4/5 | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| covariance shape 5/5 | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| post-asinh transform 1/9 | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| post-asinh transform 2/9 | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| post-asinh transform 3/9 | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| post-asinh transform 4/9 | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| post-asinh transform 5/9 | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| post-asinh transform 6/9 | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| post-asinh transform 7/9 | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| post-asinh transform 8/9 | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| post-asinh transform 9/9 | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| allocation beta | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| knot refit (cached stationary) | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| fixed point reached | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| prune proposal 32768->28672 | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| covariance shape 1/5 | 28,672 | 1 | 2.0703 | 2.38268 | no |
| covariance shape 2/5 | 28,672 | 1 | 2.0703 | 2.38268 | no |
| covariance shape 3/5 | 28,672 | 1 | 2.0703 | 2.38268 | no |
| covariance shape 4/5 | 28,672 | 1 | 2.0703 | 2.38268 | no |
| covariance shape 5/5 | 28,672 | 1 | 2.0703 | 2.38268 | no |
| post-asinh transform 1/9 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| post-asinh transform 2/9 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| post-asinh transform 3/9 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| post-asinh transform 4/9 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| post-asinh transform 5/9 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| post-asinh transform 6/9 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| post-asinh transform 7/9 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| post-asinh transform 8/9 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| post-asinh transform 9/9 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| allocation beta | 28,672 | 1 | 2.00458 | 2.3515 | no |
| knot refit | 28,672 | 1 | 2.00458 | 2.3515 | no |
| covariance shape 1/5 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| covariance shape 2/5 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| covariance shape 3/5 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| covariance shape 4/5 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| covariance shape 5/5 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| post-asinh transform 1/9 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| post-asinh transform 2/9 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| post-asinh transform 3/9 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| post-asinh transform 4/9 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| post-asinh transform 5/9 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| post-asinh transform 6/9 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| post-asinh transform 7/9 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| post-asinh transform 8/9 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| post-asinh transform 9/9 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| allocation beta | 28,672 | 1 | 2.00458 | 2.3515 | no |
| knot refit (cached stationary) | 28,672 | 1 | 2.00458 | 2.3515 | no |
| refined parameter-search resolution | 28,672 | 1 | 2.00458 | 2.3515 | no |
| covariance shape 1/5 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| covariance shape 2/5 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| covariance shape 3/5 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| covariance shape 4/5 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| covariance shape 5/5 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| post-asinh transform 1/9 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| post-asinh transform 2/9 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| post-asinh transform 3/9 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| post-asinh transform 4/9 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| post-asinh transform 5/9 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| post-asinh transform 6/9 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| post-asinh transform 7/9 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| post-asinh transform 8/9 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| post-asinh transform 9/9 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| allocation beta | 28,672 | 1 | 2.00458 | 2.3515 | no |
| knot refit (cached stationary) | 28,672 | 1 | 2.00458 | 2.3515 | no |
| refined parameter-search resolution | 28,672 | 1 | 2.00458 | 2.3515 | no |
| covariance shape 1/5 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| covariance shape 2/5 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| covariance shape 3/5 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| covariance shape 4/5 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| covariance shape 5/5 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| post-asinh transform 1/9 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| post-asinh transform 2/9 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| post-asinh transform 3/9 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| post-asinh transform 4/9 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| post-asinh transform 5/9 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| post-asinh transform 6/9 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| post-asinh transform 7/9 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| post-asinh transform 8/9 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| post-asinh transform 9/9 | 28,672 | 1 | 2.00458 | 2.3515 | no |
| allocation beta | 28,672 | 1 | 2.00458 | 2.3515 | no |
| knot refit (cached stationary) | 28,672 | 1 | 2.00458 | 2.3515 | no |
| fixed point reached | 28,672 | 1 | 2.00458 | 2.3515 | no |
| rejected prune | 28,672 | 0.95 | 0.922995 | 1.0968 | no |
| cold restart 32768->30720 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| covariance shape 1/5 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| covariance shape 2/5 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| covariance shape 3/5 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| covariance shape 4/5 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| covariance shape 5/5 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| post-asinh transform 1/9 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| post-asinh transform 2/9 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| post-asinh transform 3/9 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| post-asinh transform 4/9 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| post-asinh transform 5/9 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| post-asinh transform 6/9 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| post-asinh transform 7/9 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| post-asinh transform 8/9 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| post-asinh transform 9/9 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| allocation beta | 30,720 | 1 | 1.61726 | 1.92326 | no |
| knot refit | 30,720 | 1 | 1.61726 | 1.92326 | no |
| refined parameter-search resolution | 30,720 | 1 | 1.61726 | 1.92326 | no |
| covariance shape 1/5 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| covariance shape 2/5 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| covariance shape 3/5 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| covariance shape 4/5 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| covariance shape 5/5 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| post-asinh transform 1/9 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| post-asinh transform 2/9 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| post-asinh transform 3/9 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| post-asinh transform 4/9 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| post-asinh transform 5/9 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| post-asinh transform 6/9 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| post-asinh transform 7/9 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| post-asinh transform 8/9 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| post-asinh transform 9/9 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| allocation beta | 30,720 | 1 | 1.61726 | 1.92326 | no |
| knot refit (cached stationary) | 30,720 | 1 | 1.61726 | 1.92326 | no |
| refined parameter-search resolution | 30,720 | 1 | 1.61726 | 1.92326 | no |
| covariance shape 1/5 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| covariance shape 2/5 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| covariance shape 3/5 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| covariance shape 4/5 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| covariance shape 5/5 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| post-asinh transform 1/9 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| post-asinh transform 2/9 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| post-asinh transform 3/9 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| post-asinh transform 4/9 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| post-asinh transform 5/9 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| post-asinh transform 6/9 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| post-asinh transform 7/9 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| post-asinh transform 8/9 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| post-asinh transform 9/9 | 30,720 | 1 | 1.61726 | 1.92326 | no |
| allocation beta | 30,720 | 1 | 1.61726 | 1.92326 | no |
| knot refit (cached stationary) | 30,720 | 1 | 1.61726 | 1.92326 | no |
| fixed point reached | 30,720 | 1 | 1.61726 | 1.92326 | no |
| prune proposal 32768->30720 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| covariance shape 1/5 | 30,720 | 1 | 1.68992 | 2.00967 | no |
| covariance shape 2/5 | 30,720 | 1 | 1.68992 | 2.00967 | no |
| covariance shape 3/5 | 30,720 | 1 | 1.68992 | 2.00967 | no |
| covariance shape 4/5 | 30,720 | 1 | 1.68992 | 2.00967 | no |
| covariance shape 5/5 | 30,720 | 1 | 1.68992 | 2.00967 | no |
| post-asinh transform 1/9 | 30,720 | 1 | 1.68992 | 2.00967 | no |
| post-asinh transform 2/9 | 30,720 | 1 | 1.68992 | 2.00967 | no |
| post-asinh transform 3/9 | 30,720 | 1 | 1.68992 | 2.00967 | no |
| post-asinh transform 4/9 | 30,720 | 1 | 1.68992 | 2.00967 | no |
| post-asinh transform 5/9 | 30,720 | 1 | 1.68992 | 2.00967 | no |
| post-asinh transform 6/9 | 30,720 | 1 | 1.68992 | 2.00967 | no |
| post-asinh transform 7/9 | 30,720 | 1 | 1.68992 | 2.00967 | no |
| post-asinh transform 8/9 | 30,720 | 1 | 1.68992 | 2.00967 | no |
| post-asinh transform 9/9 | 30,720 | 1 | 1.68992 | 2.00967 | no |
| allocation beta | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| knot refit | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| covariance shape 1/5 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| covariance shape 2/5 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| covariance shape 3/5 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| covariance shape 4/5 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| covariance shape 5/5 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| post-asinh transform 1/9 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| post-asinh transform 2/9 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| post-asinh transform 3/9 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| post-asinh transform 4/9 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| post-asinh transform 5/9 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| post-asinh transform 6/9 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| post-asinh transform 7/9 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| post-asinh transform 8/9 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| post-asinh transform 9/9 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| allocation beta | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| knot refit (cached stationary) | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| refined parameter-search resolution | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| covariance shape 1/5 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| covariance shape 2/5 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| covariance shape 3/5 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| covariance shape 4/5 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| covariance shape 5/5 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| post-asinh transform 1/9 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| post-asinh transform 2/9 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| post-asinh transform 3/9 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| post-asinh transform 4/9 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| post-asinh transform 5/9 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| post-asinh transform 6/9 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| post-asinh transform 7/9 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| post-asinh transform 8/9 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| post-asinh transform 9/9 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| allocation beta | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| knot refit (cached stationary) | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| refined parameter-search resolution | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| covariance shape 1/5 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| covariance shape 2/5 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| covariance shape 3/5 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| covariance shape 4/5 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| covariance shape 5/5 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| post-asinh transform 1/9 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| post-asinh transform 2/9 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| post-asinh transform 3/9 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| post-asinh transform 4/9 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| post-asinh transform 5/9 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| post-asinh transform 6/9 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| post-asinh transform 7/9 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| post-asinh transform 8/9 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| post-asinh transform 9/9 | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| allocation beta | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| knot refit (cached stationary) | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| fixed point reached | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| rejected prune | 30,720 | 0.9 | 1.38666 | 1.64903 | no |
| prune proposal 32768->31744 | 31,744 | 1 | 0.692432 | 1 | yes |
| covariance shape 1/5 | 31,744 | 1 | 0.692432 | 1 | yes |
| covariance shape 2/5 | 31,744 | 1 | 0.692432 | 1 | yes |
| covariance shape 3/5 | 31,744 | 1 | 0.692432 | 1 | yes |
| covariance shape 4/5 | 31,744 | 1 | 0.692432 | 1 | yes |
| covariance shape 5/5 | 31,744 | 1 | 0.692432 | 1 | yes |
| post-asinh transform 1/9 | 31,744 | 1 | 0.692432 | 1 | yes |
| post-asinh transform 2/9 | 31,744 | 1 | 0.692432 | 1 | yes |
| post-asinh transform 3/9 | 31,744 | 1 | 0.692432 | 1 | yes |
| post-asinh transform 4/9 | 31,744 | 1 | 0.692432 | 1 | yes |
| post-asinh transform 5/9 | 31,744 | 1 | 0.692432 | 1 | yes |
| post-asinh transform 6/9 | 31,744 | 1 | 0.692432 | 1 | yes |
| post-asinh transform 7/9 | 31,744 | 1 | 0.692432 | 1 | yes |
| post-asinh transform 8/9 | 31,744 | 1 | 0.692432 | 1 | yes |
| post-asinh transform 9/9 | 31,744 | 1 | 0.692432 | 1 | yes |
| allocation beta | 31,744 | 1 | 0.692432 | 1 | yes |
| knot refit | 31,744 | 1 | 0.692432 | 1 | yes |
| refined parameter-search resolution | 31,744 | 1 | 0.692432 | 1 | yes |
| covariance shape 1/5 | 31,744 | 1 | 0.692432 | 1 | yes |
| covariance shape 2/5 | 31,744 | 1 | 0.692432 | 1 | yes |
| covariance shape 3/5 | 31,744 | 1 | 0.692432 | 1 | yes |
| covariance shape 4/5 | 31,744 | 1 | 0.692432 | 1 | yes |
| covariance shape 5/5 | 31,744 | 1 | 0.692432 | 1 | yes |
| post-asinh transform 1/9 | 31,744 | 1 | 0.692432 | 1 | yes |
| post-asinh transform 2/9 | 31,744 | 1 | 0.692432 | 1 | yes |
| post-asinh transform 3/9 | 31,744 | 1 | 0.692432 | 1 | yes |
| post-asinh transform 4/9 | 31,744 | 1 | 0.692432 | 1 | yes |
| post-asinh transform 5/9 | 31,744 | 1 | 0.692432 | 1 | yes |
| post-asinh transform 6/9 | 31,744 | 1 | 0.692432 | 1 | yes |
| post-asinh transform 7/9 | 31,744 | 1 | 0.692432 | 1 | yes |
| post-asinh transform 8/9 | 31,744 | 1 | 0.692432 | 1 | yes |
| post-asinh transform 9/9 | 31,744 | 1 | 0.692432 | 1 | yes |
| allocation beta | 31,744 | 1 | 0.692432 | 1 | yes |
| knot refit (cached stationary) | 31,744 | 1 | 0.692432 | 1 | yes |
| refined parameter-search resolution | 31,744 | 1 | 0.692432 | 1 | yes |
| covariance shape 1/5 | 31,744 | 1 | 0.692432 | 1 | yes |
| covariance shape 2/5 | 31,744 | 1 | 0.692432 | 1 | yes |
| covariance shape 3/5 | 31,744 | 1 | 0.692432 | 1 | yes |
| covariance shape 4/5 | 31,744 | 1 | 0.692432 | 1 | yes |
| covariance shape 5/5 | 31,744 | 1 | 0.692432 | 1 | yes |
| post-asinh transform 1/9 | 31,744 | 1 | 0.692432 | 1 | yes |
| post-asinh transform 2/9 | 31,744 | 1 | 0.692432 | 1 | yes |
| post-asinh transform 3/9 | 31,744 | 1 | 0.692432 | 1 | yes |
| post-asinh transform 4/9 | 31,744 | 1 | 0.692432 | 1 | yes |
| post-asinh transform 5/9 | 31,744 | 1 | 0.692432 | 1 | yes |
| post-asinh transform 6/9 | 31,744 | 1 | 0.692432 | 1 | yes |
| post-asinh transform 7/9 | 31,744 | 1 | 0.692432 | 1 | yes |
| post-asinh transform 8/9 | 31,744 | 1 | 0.692432 | 1 | yes |
| post-asinh transform 9/9 | 31,744 | 1 | 0.692432 | 1 | yes |
| allocation beta | 31,744 | 1 | 0.692432 | 1 | yes |
| knot refit (cached stationary) | 31,744 | 1 | 0.692432 | 1 | yes |
| fixed point reached | 31,744 | 1 | 0.692432 | 1 | yes |
| accepted prune | 31,744 | 1 | 0.692432 | 1 | yes |
| cold restart 31744->30720 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| covariance shape 1/5 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| covariance shape 2/5 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| covariance shape 3/5 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| covariance shape 4/5 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| covariance shape 5/5 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| post-asinh transform 1/9 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| post-asinh transform 2/9 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| post-asinh transform 3/9 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| post-asinh transform 4/9 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| post-asinh transform 5/9 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| post-asinh transform 6/9 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| post-asinh transform 7/9 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| post-asinh transform 8/9 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| post-asinh transform 9/9 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| allocation beta | 30,720 | 1 | 1.6094 | 1.91391 | no |
| knot refit | 30,720 | 1 | 1.6094 | 1.91391 | no |
| refined parameter-search resolution | 30,720 | 1 | 1.6094 | 1.91391 | no |
| covariance shape 1/5 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| covariance shape 2/5 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| covariance shape 3/5 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| covariance shape 4/5 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| covariance shape 5/5 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| post-asinh transform 1/9 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| post-asinh transform 2/9 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| post-asinh transform 3/9 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| post-asinh transform 4/9 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| post-asinh transform 5/9 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| post-asinh transform 6/9 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| post-asinh transform 7/9 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| post-asinh transform 8/9 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| post-asinh transform 9/9 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| allocation beta | 30,720 | 1 | 1.6094 | 1.91391 | no |
| knot refit (cached stationary) | 30,720 | 1 | 1.6094 | 1.91391 | no |
| refined parameter-search resolution | 30,720 | 1 | 1.6094 | 1.91391 | no |
| covariance shape 1/5 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| covariance shape 2/5 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| covariance shape 3/5 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| covariance shape 4/5 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| covariance shape 5/5 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| post-asinh transform 1/9 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| post-asinh transform 2/9 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| post-asinh transform 3/9 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| post-asinh transform 4/9 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| post-asinh transform 5/9 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| post-asinh transform 6/9 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| post-asinh transform 7/9 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| post-asinh transform 8/9 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| post-asinh transform 9/9 | 30,720 | 1 | 1.6094 | 1.91391 | no |
| allocation beta | 30,720 | 1 | 1.6094 | 1.91391 | no |
| knot refit (cached stationary) | 30,720 | 1 | 1.6094 | 1.91391 | no |
| fixed point reached | 30,720 | 1 | 1.6094 | 1.91391 | no |
| prune proposal 31744->30720 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| covariance shape 1/5 | 30,720 | 1 | 1.70154 | 2.02348 | no |
| covariance shape 2/5 | 30,720 | 1 | 1.70154 | 2.02348 | no |
| covariance shape 3/5 | 30,720 | 1 | 1.70154 | 2.02348 | no |
| covariance shape 4/5 | 30,720 | 1 | 1.70154 | 2.02348 | no |
| covariance shape 5/5 | 30,720 | 1 | 1.70154 | 2.02348 | no |
| post-asinh transform 1/9 | 30,720 | 1 | 1.70154 | 2.02348 | no |
| post-asinh transform 2/9 | 30,720 | 1 | 1.70154 | 2.02348 | no |
| post-asinh transform 3/9 | 30,720 | 1 | 1.70154 | 2.02348 | no |
| post-asinh transform 4/9 | 30,720 | 1 | 1.70154 | 2.02348 | no |
| post-asinh transform 5/9 | 30,720 | 1 | 1.70154 | 2.02348 | no |
| post-asinh transform 6/9 | 30,720 | 1 | 1.70154 | 2.02348 | no |
| post-asinh transform 7/9 | 30,720 | 1 | 1.70154 | 2.02348 | no |
| post-asinh transform 8/9 | 30,720 | 1 | 1.70154 | 2.02348 | no |
| post-asinh transform 9/9 | 30,720 | 1 | 1.70154 | 2.02348 | no |
| allocation beta | 30,720 | 0.9 | 1.41182 | 1.67895 | no |
| knot refit | 30,720 | 0.9 | 1.40261 | 1.66798 | no |
| covariance shape 1/5 | 30,720 | 0.9 | 1.40261 | 1.66798 | no |
| covariance shape 2/5 | 30,720 | 0.9 | 1.40261 | 1.66798 | no |
| covariance shape 3/5 | 30,720 | 0.9 | 1.40261 | 1.66798 | no |
| covariance shape 4/5 | 30,720 | 0.9 | 1.40261 | 1.66798 | no |
| covariance shape 5/5 | 30,720 | 0.9 | 1.40261 | 1.66798 | no |
| post-asinh transform 1/9 | 30,720 | 0.9 | 1.40261 | 1.66798 | no |
| post-asinh transform 2/9 | 30,720 | 0.9 | 1.40261 | 1.66798 | no |
| post-asinh transform 3/9 | 30,720 | 0.9 | 1.40261 | 1.66798 | no |
| post-asinh transform 4/9 | 30,720 | 0.9 | 1.40261 | 1.66798 | no |
| post-asinh transform 5/9 | 30,720 | 0.9 | 1.40261 | 1.66798 | no |
| post-asinh transform 6/9 | 30,720 | 0.9 | 1.40261 | 1.66798 | no |
| post-asinh transform 7/9 | 30,720 | 0.9 | 1.40261 | 1.66798 | no |
| post-asinh transform 8/9 | 30,720 | 0.9 | 1.40261 | 1.66798 | no |
| post-asinh transform 9/9 | 30,720 | 0.9 | 1.40261 | 1.66798 | no |
| allocation beta | 30,720 | 0.8 | 1.39191 | 1.65525 | no |
| knot refit | 30,720 | 0.8 | 1.36878 | 1.62776 | no |
| covariance shape 1/5 | 30,720 | 0.8 | 1.36878 | 1.62776 | no |
| covariance shape 2/5 | 30,720 | 0.8 | 1.36878 | 1.62776 | no |
| covariance shape 3/5 | 30,720 | 0.8 | 1.36878 | 1.62776 | no |
| covariance shape 4/5 | 30,720 | 0.8 | 1.36878 | 1.62776 | no |
| covariance shape 5/5 | 30,720 | 0.8 | 1.36878 | 1.62776 | no |
| post-asinh transform 1/9 | 30,720 | 0.8 | 1.36878 | 1.62776 | no |
| post-asinh transform 2/9 | 30,720 | 0.8 | 1.36878 | 1.62776 | no |
| post-asinh transform 3/9 | 30,720 | 0.8 | 1.36878 | 1.62776 | no |
| post-asinh transform 4/9 | 30,720 | 0.8 | 1.36878 | 1.62776 | no |
| post-asinh transform 5/9 | 30,720 | 0.8 | 1.36878 | 1.62776 | no |
| post-asinh transform 6/9 | 30,720 | 0.8 | 1.36878 | 1.62776 | no |
| post-asinh transform 7/9 | 30,720 | 0.8 | 1.36878 | 1.62776 | no |
| post-asinh transform 8/9 | 30,720 | 0.8 | 1.36878 | 1.62776 | no |
| post-asinh transform 9/9 | 30,720 | 0.8 | 1.36878 | 1.62776 | no |
| allocation beta | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| knot refit | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| covariance shape 1/5 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| covariance shape 2/5 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| covariance shape 3/5 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| covariance shape 4/5 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| covariance shape 5/5 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| post-asinh transform 1/9 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| post-asinh transform 2/9 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| post-asinh transform 3/9 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| post-asinh transform 4/9 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| post-asinh transform 5/9 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| post-asinh transform 6/9 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| post-asinh transform 7/9 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| post-asinh transform 8/9 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| post-asinh transform 9/9 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| allocation beta | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| knot refit (cached stationary) | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| refined parameter-search resolution | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| covariance shape 1/5 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| covariance shape 2/5 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| covariance shape 3/5 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| covariance shape 4/5 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| covariance shape 5/5 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| post-asinh transform 1/9 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| post-asinh transform 2/9 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| post-asinh transform 3/9 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| post-asinh transform 4/9 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| post-asinh transform 5/9 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| post-asinh transform 6/9 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| post-asinh transform 7/9 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| post-asinh transform 8/9 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| post-asinh transform 9/9 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| allocation beta | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| knot refit (cached stationary) | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| refined parameter-search resolution | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| covariance shape 1/5 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| covariance shape 2/5 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| covariance shape 3/5 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| covariance shape 4/5 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| covariance shape 5/5 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| post-asinh transform 1/9 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| post-asinh transform 2/9 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| post-asinh transform 3/9 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| post-asinh transform 4/9 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| post-asinh transform 5/9 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| post-asinh transform 6/9 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| post-asinh transform 7/9 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| post-asinh transform 8/9 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| post-asinh transform 9/9 | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| allocation beta | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| knot refit (cached stationary) | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| fixed point reached | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| rejected prune | 30,720 | 0.7 | 1.36329 | 1.62123 | no |
| cold restart 31744->31232 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| covariance shape 1/5 | 31,232 | 1 | 0.944593 | 1.12327 | no |
| covariance shape 2/5 | 31,232 | 1 | 0.944593 | 1.12327 | no |
| covariance shape 3/5 | 31,232 | 1 | 0.944593 | 1.12327 | no |
| covariance shape 4/5 | 31,232 | 1 | 0.944593 | 1.12327 | no |
| covariance shape 5/5 | 31,232 | 1 | 0.944593 | 1.12327 | no |
| post-asinh transform 1/9 | 31,232 | 1 | 0.944593 | 1.12327 | no |
| post-asinh transform 2/9 | 31,232 | 1 | 0.944593 | 1.12327 | no |
| post-asinh transform 3/9 | 31,232 | 1 | 0.944593 | 1.12327 | no |
| post-asinh transform 4/9 | 31,232 | 1 | 0.944593 | 1.12327 | no |
| post-asinh transform 5/9 | 31,232 | 1 | 0.944593 | 1.12327 | no |
| post-asinh transform 6/9 | 31,232 | 1 | 0.944593 | 1.12327 | no |
| post-asinh transform 7/9 | 31,232 | 1 | 0.944593 | 1.12327 | no |
| post-asinh transform 8/9 | 31,232 | 1 | 0.944593 | 1.12327 | no |
| post-asinh transform 9/9 | 31,232 | 1 | 0.944593 | 1.12327 | no |
| allocation beta | 31,232 | 0.9 | 0.940912 | 1.1189 | no |
| knot refit | 31,232 | 0.9 | 0.940912 | 1.1189 | no |
| covariance shape 1/5 | 31,232 | 0.9 | 0.940912 | 1.1189 | no |
| covariance shape 2/5 | 31,232 | 0.9 | 0.940912 | 1.1189 | no |
| covariance shape 3/5 | 31,232 | 0.9 | 0.940912 | 1.1189 | no |
| covariance shape 4/5 | 31,232 | 0.9 | 0.940912 | 1.1189 | no |
| covariance shape 5/5 | 31,232 | 0.9 | 0.940912 | 1.1189 | no |
| post-asinh transform 1/9 | 31,232 | 0.9 | 0.940912 | 1.1189 | no |
| post-asinh transform 2/9 | 31,232 | 0.9 | 0.940912 | 1.1189 | no |
| post-asinh transform 3/9 | 31,232 | 0.9 | 0.940912 | 1.1189 | no |
| post-asinh transform 4/9 | 31,232 | 0.9 | 0.940912 | 1.1189 | no |
| post-asinh transform 5/9 | 31,232 | 0.9 | 0.940912 | 1.1189 | no |
| post-asinh transform 6/9 | 31,232 | 0.9 | 0.940912 | 1.1189 | no |
| post-asinh transform 7/9 | 31,232 | 0.9 | 0.940912 | 1.1189 | no |
| post-asinh transform 8/9 | 31,232 | 0.9 | 0.940912 | 1.1189 | no |
| post-asinh transform 9/9 | 31,232 | 0.9 | 0.940912 | 1.1189 | no |
| allocation beta | 31,232 | 1 | 0.939225 | 1.11689 | no |
| knot refit | 31,232 | 1 | 0.939225 | 1.11689 | no |
| covariance shape 1/5 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| covariance shape 2/5 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| covariance shape 3/5 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| covariance shape 4/5 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| covariance shape 5/5 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| post-asinh transform 1/9 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| post-asinh transform 2/9 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| post-asinh transform 3/9 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| post-asinh transform 4/9 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| post-asinh transform 5/9 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| post-asinh transform 6/9 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| post-asinh transform 7/9 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| post-asinh transform 8/9 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| post-asinh transform 9/9 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| allocation beta | 31,232 | 1 | 0.939225 | 1.11689 | no |
| knot refit (cached stationary) | 31,232 | 1 | 0.939225 | 1.11689 | no |
| refined parameter-search resolution | 31,232 | 1 | 0.939225 | 1.11689 | no |
| covariance shape 1/5 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| covariance shape 2/5 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| covariance shape 3/5 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| covariance shape 4/5 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| covariance shape 5/5 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| post-asinh transform 1/9 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| post-asinh transform 2/9 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| post-asinh transform 3/9 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| post-asinh transform 4/9 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| post-asinh transform 5/9 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| post-asinh transform 6/9 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| post-asinh transform 7/9 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| post-asinh transform 8/9 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| post-asinh transform 9/9 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| allocation beta | 31,232 | 1 | 0.939225 | 1.11689 | no |
| knot refit (cached stationary) | 31,232 | 1 | 0.939225 | 1.11689 | no |
| refined parameter-search resolution | 31,232 | 1 | 0.939225 | 1.11689 | no |
| covariance shape 1/5 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| covariance shape 2/5 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| covariance shape 3/5 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| covariance shape 4/5 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| covariance shape 5/5 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| post-asinh transform 1/9 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| post-asinh transform 2/9 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| post-asinh transform 3/9 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| post-asinh transform 4/9 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| post-asinh transform 5/9 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| post-asinh transform 6/9 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| post-asinh transform 7/9 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| post-asinh transform 8/9 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| post-asinh transform 9/9 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| allocation beta | 31,232 | 1 | 0.939225 | 1.11689 | no |
| knot refit (cached stationary) | 31,232 | 1 | 0.939225 | 1.11689 | no |
| fixed point reached | 31,232 | 1 | 0.939225 | 1.11689 | no |
| prune proposal 31744->31232 | 31,232 | 1 | 0.939225 | 1.11689 | no |
| covariance shape 1/5 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| covariance shape 2/5 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| covariance shape 3/5 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| covariance shape 4/5 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| covariance shape 5/5 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| post-asinh transform 1/9 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| post-asinh transform 2/9 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| post-asinh transform 3/9 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| post-asinh transform 4/9 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| post-asinh transform 5/9 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| post-asinh transform 6/9 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| post-asinh transform 7/9 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| post-asinh transform 8/9 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| post-asinh transform 9/9 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| allocation beta | 31,232 | 1 | 0.946358 | 1.12529 | no |
| knot refit | 31,232 | 1 | 0.946358 | 1.12529 | no |
| refined parameter-search resolution | 31,232 | 1 | 0.946358 | 1.12529 | no |
| covariance shape 1/5 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| covariance shape 2/5 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| covariance shape 3/5 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| covariance shape 4/5 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| covariance shape 5/5 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| post-asinh transform 1/9 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| post-asinh transform 2/9 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| post-asinh transform 3/9 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| post-asinh transform 4/9 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| post-asinh transform 5/9 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| post-asinh transform 6/9 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| post-asinh transform 7/9 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| post-asinh transform 8/9 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| post-asinh transform 9/9 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| allocation beta | 31,232 | 1 | 0.946358 | 1.12529 | no |
| knot refit (cached stationary) | 31,232 | 1 | 0.946358 | 1.12529 | no |
| refined parameter-search resolution | 31,232 | 1 | 0.946358 | 1.12529 | no |
| covariance shape 1/5 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| covariance shape 2/5 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| covariance shape 3/5 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| covariance shape 4/5 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| covariance shape 5/5 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| post-asinh transform 1/9 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| post-asinh transform 2/9 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| post-asinh transform 3/9 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| post-asinh transform 4/9 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| post-asinh transform 5/9 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| post-asinh transform 6/9 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| post-asinh transform 7/9 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| post-asinh transform 8/9 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| post-asinh transform 9/9 | 31,232 | 1 | 0.946358 | 1.12529 | no |
| allocation beta | 31,232 | 1 | 0.946358 | 1.12529 | no |
| knot refit (cached stationary) | 31,232 | 1 | 0.946358 | 1.12529 | no |
| fixed point reached | 31,232 | 1 | 0.946358 | 1.12529 | no |
| rejected prune | 31,232 | 1 | 0.939225 | 1.11689 | no |
| prune proposal 31744->31488 | 31,488 | 1 | 0.689049 | 1 | yes |
| covariance shape 1/5 | 31,488 | 1 | 0.689049 | 1 | yes |
| covariance shape 2/5 | 31,488 | 1 | 0.689049 | 1 | yes |
| covariance shape 3/5 | 31,488 | 1 | 0.689049 | 1 | yes |
| covariance shape 4/5 | 31,488 | 1 | 0.689049 | 1 | yes |
| covariance shape 5/5 | 31,488 | 1 | 0.689049 | 1 | yes |
| post-asinh transform 1/9 | 31,488 | 1 | 0.689049 | 1 | yes |
| post-asinh transform 2/9 | 31,488 | 1 | 0.689049 | 1 | yes |
| post-asinh transform 3/9 | 31,488 | 1 | 0.689049 | 1 | yes |
| post-asinh transform 4/9 | 31,488 | 1 | 0.689049 | 1 | yes |
| post-asinh transform 5/9 | 31,488 | 1 | 0.689049 | 1 | yes |
| post-asinh transform 6/9 | 31,488 | 1 | 0.689049 | 1 | yes |
| post-asinh transform 7/9 | 31,488 | 1 | 0.689049 | 1 | yes |
| post-asinh transform 8/9 | 31,488 | 1 | 0.689049 | 1 | yes |
| post-asinh transform 9/9 | 31,488 | 1 | 0.689049 | 1 | yes |
| allocation beta | 31,488 | 1 | 0.689049 | 1 | yes |
| knot refit | 31,488 | 1 | 0.689049 | 1 | yes |
| refined parameter-search resolution | 31,488 | 1 | 0.689049 | 1 | yes |
| covariance shape 1/5 | 31,488 | 1 | 0.689049 | 1 | yes |
| covariance shape 2/5 | 31,488 | 1 | 0.689049 | 1 | yes |
| covariance shape 3/5 | 31,488 | 1 | 0.689049 | 1 | yes |
| covariance shape 4/5 | 31,488 | 1 | 0.689049 | 1 | yes |
| covariance shape 5/5 | 31,488 | 1 | 0.689049 | 1 | yes |
| post-asinh transform 1/9 | 31,488 | 1 | 0.689049 | 1 | yes |
| post-asinh transform 2/9 | 31,488 | 1 | 0.689049 | 1 | yes |
| post-asinh transform 3/9 | 31,488 | 1 | 0.689049 | 1 | yes |
| post-asinh transform 4/9 | 31,488 | 1 | 0.689049 | 1 | yes |
| post-asinh transform 5/9 | 31,488 | 1 | 0.689049 | 1 | yes |
| post-asinh transform 6/9 | 31,488 | 1 | 0.689049 | 1 | yes |
| post-asinh transform 7/9 | 31,488 | 1 | 0.689049 | 1 | yes |
| post-asinh transform 8/9 | 31,488 | 1 | 0.689049 | 1 | yes |
| post-asinh transform 9/9 | 31,488 | 1 | 0.689049 | 1 | yes |
| allocation beta | 31,488 | 1 | 0.689049 | 1 | yes |
| knot refit (cached stationary) | 31,488 | 1 | 0.689049 | 1 | yes |
| refined parameter-search resolution | 31,488 | 1 | 0.689049 | 1 | yes |
| covariance shape 1/5 | 31,488 | 1 | 0.689049 | 1 | yes |
| covariance shape 2/5 | 31,488 | 1 | 0.689049 | 1 | yes |
| covariance shape 3/5 | 31,488 | 1 | 0.689049 | 1 | yes |
| covariance shape 4/5 | 31,488 | 1 | 0.689049 | 1 | yes |
| covariance shape 5/5 | 31,488 | 1 | 0.689049 | 1 | yes |
| post-asinh transform 1/9 | 31,488 | 1 | 0.689049 | 1 | yes |
| post-asinh transform 2/9 | 31,488 | 1 | 0.689049 | 1 | yes |
| post-asinh transform 3/9 | 31,488 | 1 | 0.689049 | 1 | yes |
| post-asinh transform 4/9 | 31,488 | 1 | 0.689049 | 1 | yes |
| post-asinh transform 5/9 | 31,488 | 1 | 0.689049 | 1 | yes |
| post-asinh transform 6/9 | 31,488 | 1 | 0.689049 | 1 | yes |
| post-asinh transform 7/9 | 31,488 | 1 | 0.689049 | 1 | yes |
| post-asinh transform 8/9 | 31,488 | 1 | 0.689049 | 1 | yes |
| post-asinh transform 9/9 | 31,488 | 1 | 0.689049 | 1 | yes |
| allocation beta | 31,488 | 1 | 0.689049 | 1 | yes |
| knot refit (cached stationary) | 31,488 | 1 | 0.689049 | 1 | yes |
| fixed point reached | 31,488 | 1 | 0.689049 | 1 | yes |
| accepted prune | 31,488 | 1 | 0.689049 | 1 | yes |
| cold restart 31488->31232 | 31,232 | 0.925 | 1.03708 | 1.23325 | no |
| covariance shape 1/5 | 31,232 | 1 | 1.05022 | 1.24889 | no |
| covariance shape 2/5 | 31,232 | 1 | 1.05022 | 1.24889 | no |
| covariance shape 3/5 | 31,232 | 1 | 1.05022 | 1.24889 | no |
| covariance shape 4/5 | 31,232 | 1 | 1.05022 | 1.24889 | no |
| covariance shape 5/5 | 31,232 | 1 | 1.05022 | 1.24889 | no |
| post-asinh transform 1/9 | 31,232 | 1 | 1.05022 | 1.24889 | no |
| post-asinh transform 2/9 | 31,232 | 1 | 1.05022 | 1.24889 | no |
| post-asinh transform 3/9 | 31,232 | 1 | 1.05022 | 1.24889 | no |
| post-asinh transform 4/9 | 31,232 | 1 | 1.05022 | 1.24889 | no |
| post-asinh transform 5/9 | 31,232 | 1 | 1.05022 | 1.24889 | no |
| post-asinh transform 6/9 | 31,232 | 1 | 1.05022 | 1.24889 | no |
| post-asinh transform 7/9 | 31,232 | 1 | 1.05022 | 1.24889 | no |
| post-asinh transform 8/9 | 31,232 | 1 | 1.05022 | 1.24889 | no |
| post-asinh transform 9/9 | 31,232 | 1 | 1.05022 | 1.24889 | no |
| allocation beta | 31,232 | 0.9 | 1.04042 | 1.23723 | no |
| knot refit | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| covariance shape 1/5 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| covariance shape 2/5 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| covariance shape 3/5 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| covariance shape 4/5 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| covariance shape 5/5 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| post-asinh transform 1/9 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| post-asinh transform 2/9 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| post-asinh transform 3/9 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| post-asinh transform 4/9 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| post-asinh transform 5/9 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| post-asinh transform 6/9 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| post-asinh transform 7/9 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| post-asinh transform 8/9 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| post-asinh transform 9/9 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| allocation beta | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| knot refit | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| refined parameter-search resolution | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| covariance shape 1/5 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| covariance shape 2/5 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| covariance shape 3/5 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| covariance shape 4/5 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| covariance shape 5/5 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| post-asinh transform 1/9 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| post-asinh transform 2/9 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| post-asinh transform 3/9 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| post-asinh transform 4/9 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| post-asinh transform 5/9 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| post-asinh transform 6/9 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| post-asinh transform 7/9 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| post-asinh transform 8/9 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| post-asinh transform 9/9 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| allocation beta | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| knot refit (cached stationary) | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| refined parameter-search resolution | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| covariance shape 1/5 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| covariance shape 2/5 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| covariance shape 3/5 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| covariance shape 4/5 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| covariance shape 5/5 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| post-asinh transform 1/9 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| post-asinh transform 2/9 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| post-asinh transform 3/9 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| post-asinh transform 4/9 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| post-asinh transform 5/9 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| post-asinh transform 6/9 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| post-asinh transform 7/9 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| post-asinh transform 8/9 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| post-asinh transform 9/9 | 31,232 | 0.9 | 1.03856 | 1.23502 | no |
| allocation beta | 31,232 | 0.925 | 1.03708 | 1.23325 | no |
| knot refit | 31,232 | 0.925 | 1.03708 | 1.23325 | no |
| covariance shape 1/5 | 31,232 | 0.925 | 1.03708 | 1.23325 | no |
| covariance shape 2/5 | 31,232 | 0.925 | 1.03708 | 1.23325 | no |
| covariance shape 3/5 | 31,232 | 0.925 | 1.03708 | 1.23325 | no |
| covariance shape 4/5 | 31,232 | 0.925 | 1.03708 | 1.23325 | no |
| covariance shape 5/5 | 31,232 | 0.925 | 1.03708 | 1.23325 | no |
| post-asinh transform 1/9 | 31,232 | 0.925 | 1.03708 | 1.23325 | no |
| post-asinh transform 2/9 | 31,232 | 0.925 | 1.03708 | 1.23325 | no |
| post-asinh transform 3/9 | 31,232 | 0.925 | 1.03708 | 1.23325 | no |
| post-asinh transform 4/9 | 31,232 | 0.925 | 1.03708 | 1.23325 | no |
| post-asinh transform 5/9 | 31,232 | 0.925 | 1.03708 | 1.23325 | no |
| post-asinh transform 6/9 | 31,232 | 0.925 | 1.03708 | 1.23325 | no |
| post-asinh transform 7/9 | 31,232 | 0.925 | 1.03708 | 1.23325 | no |
| post-asinh transform 8/9 | 31,232 | 0.925 | 1.03708 | 1.23325 | no |
| post-asinh transform 9/9 | 31,232 | 0.925 | 1.03708 | 1.23325 | no |
| allocation beta | 31,232 | 0.925 | 1.03708 | 1.23325 | no |
| knot refit (cached stationary) | 31,232 | 0.925 | 1.03708 | 1.23325 | no |
| fixed point reached | 31,232 | 0.925 | 1.03708 | 1.23325 | no |
| prune proposal 31488->31232 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| covariance shape 1/5 | 31,232 | 1 | 0.995256 | 1.18345 | no |
| covariance shape 2/5 | 31,232 | 1 | 0.995256 | 1.18345 | no |
| covariance shape 3/5 | 31,232 | 1 | 0.995256 | 1.18345 | no |
| covariance shape 4/5 | 31,232 | 1 | 0.995256 | 1.18345 | no |
| covariance shape 5/5 | 31,232 | 1 | 0.995256 | 1.18345 | no |
| post-asinh transform 1/9 | 31,232 | 1 | 0.995256 | 1.18345 | no |
| post-asinh transform 2/9 | 31,232 | 1 | 0.995256 | 1.18345 | no |
| post-asinh transform 3/9 | 31,232 | 1 | 0.995256 | 1.18345 | no |
| post-asinh transform 4/9 | 31,232 | 1 | 0.995256 | 1.18345 | no |
| post-asinh transform 5/9 | 31,232 | 1 | 0.995256 | 1.18345 | no |
| post-asinh transform 6/9 | 31,232 | 1 | 0.995256 | 1.18345 | no |
| post-asinh transform 7/9 | 31,232 | 1 | 0.995256 | 1.18345 | no |
| post-asinh transform 8/9 | 31,232 | 1 | 0.995256 | 1.18345 | no |
| post-asinh transform 9/9 | 31,232 | 1 | 0.995256 | 1.18345 | no |
| allocation beta | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| knot refit | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| covariance shape 1/5 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| covariance shape 2/5 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| covariance shape 3/5 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| covariance shape 4/5 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| covariance shape 5/5 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| post-asinh transform 1/9 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| post-asinh transform 2/9 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| post-asinh transform 3/9 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| post-asinh transform 4/9 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| post-asinh transform 5/9 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| post-asinh transform 6/9 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| post-asinh transform 7/9 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| post-asinh transform 8/9 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| post-asinh transform 9/9 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| allocation beta | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| knot refit (cached stationary) | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| refined parameter-search resolution | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| covariance shape 1/5 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| covariance shape 2/5 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| covariance shape 3/5 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| covariance shape 4/5 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| covariance shape 5/5 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| post-asinh transform 1/9 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| post-asinh transform 2/9 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| post-asinh transform 3/9 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| post-asinh transform 4/9 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| post-asinh transform 5/9 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| post-asinh transform 6/9 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| post-asinh transform 7/9 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| post-asinh transform 8/9 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| post-asinh transform 9/9 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| allocation beta | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| knot refit (cached stationary) | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| refined parameter-search resolution | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| covariance shape 1/5 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| covariance shape 2/5 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| covariance shape 3/5 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| covariance shape 4/5 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| covariance shape 5/5 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| post-asinh transform 1/9 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| post-asinh transform 2/9 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| post-asinh transform 3/9 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| post-asinh transform 4/9 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| post-asinh transform 5/9 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| post-asinh transform 6/9 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| post-asinh transform 7/9 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| post-asinh transform 8/9 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| post-asinh transform 9/9 | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| allocation beta | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| knot refit (cached stationary) | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| fixed point reached | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| rejected prune | 31,232 | 0.9 | 0.954809 | 1.13532 | no |
| prune proposal 31488->31360 | 31,360 | 1 | 0.705168 | 1 | yes |
| covariance shape 1/5 | 31,360 | 1 | 0.705168 | 1 | yes |
| covariance shape 2/5 | 31,360 | 1 | 0.705168 | 1 | yes |
| covariance shape 3/5 | 31,360 | 1 | 0.705168 | 1 | yes |
| covariance shape 4/5 | 31,360 | 1 | 0.705168 | 1 | yes |
| covariance shape 5/5 | 31,360 | 1 | 0.705168 | 1 | yes |
| post-asinh transform 1/9 | 31,360 | 1 | 0.705168 | 1 | yes |
| post-asinh transform 2/9 | 31,360 | 1 | 0.705168 | 1 | yes |
| post-asinh transform 3/9 | 31,360 | 1 | 0.705168 | 1 | yes |
| post-asinh transform 4/9 | 31,360 | 1 | 0.705168 | 1 | yes |
| post-asinh transform 5/9 | 31,360 | 1 | 0.705168 | 1 | yes |
| post-asinh transform 6/9 | 31,360 | 1 | 0.705168 | 1 | yes |
| post-asinh transform 7/9 | 31,360 | 1 | 0.705168 | 1 | yes |
| post-asinh transform 8/9 | 31,360 | 1 | 0.705168 | 1 | yes |
| post-asinh transform 9/9 | 31,360 | 1 | 0.705168 | 1 | yes |
| allocation beta | 31,360 | 1 | 0.705168 | 1 | yes |
| knot refit | 31,360 | 1 | 0.705168 | 1 | yes |
| refined parameter-search resolution | 31,360 | 1 | 0.705168 | 1 | yes |
| covariance shape 1/5 | 31,360 | 1 | 0.705168 | 1 | yes |
| covariance shape 2/5 | 31,360 | 1 | 0.705168 | 1 | yes |
| covariance shape 3/5 | 31,360 | 1 | 0.705168 | 1 | yes |
| covariance shape 4/5 | 31,360 | 1 | 0.705168 | 1 | yes |
| covariance shape 5/5 | 31,360 | 1 | 0.705168 | 1 | yes |
| post-asinh transform 1/9 | 31,360 | 1 | 0.705168 | 1 | yes |
| post-asinh transform 2/9 | 31,360 | 1 | 0.705168 | 1 | yes |
| post-asinh transform 3/9 | 31,360 | 1 | 0.705168 | 1 | yes |
| post-asinh transform 4/9 | 31,360 | 1 | 0.705168 | 1 | yes |
| post-asinh transform 5/9 | 31,360 | 1 | 0.705168 | 1 | yes |
| post-asinh transform 6/9 | 31,360 | 1 | 0.705168 | 1 | yes |
| post-asinh transform 7/9 | 31,360 | 1 | 0.705168 | 1 | yes |
| post-asinh transform 8/9 | 31,360 | 1 | 0.705168 | 1 | yes |
| post-asinh transform 9/9 | 31,360 | 1 | 0.705168 | 1 | yes |
| allocation beta | 31,360 | 1 | 0.705168 | 1 | yes |
| knot refit (cached stationary) | 31,360 | 1 | 0.705168 | 1 | yes |
| refined parameter-search resolution | 31,360 | 1 | 0.705168 | 1 | yes |
| covariance shape 1/5 | 31,360 | 1 | 0.705168 | 1 | yes |
| covariance shape 2/5 | 31,360 | 1 | 0.705168 | 1 | yes |
| covariance shape 3/5 | 31,360 | 1 | 0.705168 | 1 | yes |
| covariance shape 4/5 | 31,360 | 1 | 0.705168 | 1 | yes |
| covariance shape 5/5 | 31,360 | 1 | 0.705168 | 1 | yes |
| post-asinh transform 1/9 | 31,360 | 1 | 0.705168 | 1 | yes |
| post-asinh transform 2/9 | 31,360 | 1 | 0.705168 | 1 | yes |
| post-asinh transform 3/9 | 31,360 | 1 | 0.705168 | 1 | yes |
| post-asinh transform 4/9 | 31,360 | 1 | 0.705168 | 1 | yes |
| post-asinh transform 5/9 | 31,360 | 1 | 0.705168 | 1 | yes |
| post-asinh transform 6/9 | 31,360 | 1 | 0.705168 | 1 | yes |
| post-asinh transform 7/9 | 31,360 | 1 | 0.705168 | 1 | yes |
| post-asinh transform 8/9 | 31,360 | 1 | 0.705168 | 1 | yes |
| post-asinh transform 9/9 | 31,360 | 1 | 0.705168 | 1 | yes |
| allocation beta | 31,360 | 1 | 0.705168 | 1 | yes |
| knot refit (cached stationary) | 31,360 | 1 | 0.705168 | 1 | yes |
| fixed point reached | 31,360 | 1 | 0.705168 | 1 | yes |
| accepted prune | 31,360 | 1 | 0.705168 | 1 | yes |
| cold restart 31360->31232 | 31,232 | 1 | 1.03578 | 1.23167 | no |
| covariance shape 1/5 | 31,232 | 1 | 1.03983 | 1.23647 | no |
| covariance shape 2/5 | 31,232 | 1 | 1.03983 | 1.23647 | no |
| covariance shape 3/5 | 31,232 | 1 | 1.03983 | 1.23647 | no |
| covariance shape 4/5 | 31,232 | 1 | 1.03983 | 1.23647 | no |
| covariance shape 5/5 | 31,232 | 1 | 1.03983 | 1.23647 | no |
| post-asinh transform 1/9 | 31,232 | 1 | 1.03983 | 1.23647 | no |
| post-asinh transform 2/9 | 31,232 | 1 | 1.03983 | 1.23647 | no |
| post-asinh transform 3/9 | 31,232 | 1 | 1.03983 | 1.23647 | no |
| post-asinh transform 4/9 | 31,232 | 1 | 1.03983 | 1.23647 | no |
| post-asinh transform 5/9 | 31,232 | 1 | 1.03983 | 1.23647 | no |
| post-asinh transform 6/9 | 31,232 | 1 | 1.03983 | 1.23647 | no |
| post-asinh transform 7/9 | 31,232 | 1 | 1.03983 | 1.23647 | no |
| post-asinh transform 8/9 | 31,232 | 1 | 1.03983 | 1.23647 | no |
| post-asinh transform 9/9 | 31,232 | 1 | 1.03983 | 1.23647 | no |
| allocation beta | 31,232 | 0.9 | 1.03854 | 1.23495 | no |
| knot refit | 31,232 | 0.9 | 1.03854 | 1.23495 | no |
| covariance shape 1/5 | 31,232 | 0.9 | 1.03854 | 1.23495 | no |
| covariance shape 2/5 | 31,232 | 0.9 | 1.03854 | 1.23495 | no |
| covariance shape 3/5 | 31,232 | 0.9 | 1.03854 | 1.23495 | no |
| covariance shape 4/5 | 31,232 | 0.9 | 1.03854 | 1.23495 | no |
| covariance shape 5/5 | 31,232 | 0.9 | 1.03854 | 1.23495 | no |
| post-asinh transform 1/9 | 31,232 | 0.9 | 1.03854 | 1.23495 | no |
| post-asinh transform 2/9 | 31,232 | 0.9 | 1.03854 | 1.23495 | no |
| post-asinh transform 3/9 | 31,232 | 0.9 | 1.03854 | 1.23495 | no |
| post-asinh transform 4/9 | 31,232 | 0.9 | 1.03854 | 1.23495 | no |
| post-asinh transform 5/9 | 31,232 | 0.9 | 1.03854 | 1.23495 | no |
| post-asinh transform 6/9 | 31,232 | 0.9 | 1.03854 | 1.23495 | no |
| post-asinh transform 7/9 | 31,232 | 0.9 | 1.03854 | 1.23495 | no |
| post-asinh transform 8/9 | 31,232 | 0.9 | 1.03854 | 1.23495 | no |
| post-asinh transform 9/9 | 31,232 | 0.9 | 1.03854 | 1.23495 | no |
| allocation beta | 31,232 | 0.9 | 1.03854 | 1.23495 | no |
| knot refit (cached stationary) | 31,232 | 0.9 | 1.03854 | 1.23495 | no |
| refined parameter-search resolution | 31,232 | 0.9 | 1.03854 | 1.23495 | no |
| covariance shape 1/5 | 31,232 | 0.9 | 1.03854 | 1.23495 | no |
| covariance shape 2/5 | 31,232 | 0.9 | 1.03854 | 1.23495 | no |
| covariance shape 3/5 | 31,232 | 0.9 | 1.03854 | 1.23495 | no |
| covariance shape 4/5 | 31,232 | 0.9 | 1.03854 | 1.23495 | no |
| covariance shape 5/5 | 31,232 | 0.9 | 1.03854 | 1.23495 | no |
| post-asinh transform 1/9 | 31,232 | 0.9 | 1.03854 | 1.23495 | no |
| post-asinh transform 2/9 | 31,232 | 0.9 | 1.03854 | 1.23495 | no |
| post-asinh transform 3/9 | 31,232 | 0.9 | 1.03854 | 1.23495 | no |
| post-asinh transform 4/9 | 31,232 | 0.9 | 1.03854 | 1.23495 | no |
| post-asinh transform 5/9 | 31,232 | 0.9 | 1.03854 | 1.23495 | no |
| post-asinh transform 6/9 | 31,232 | 0.9 | 1.03854 | 1.23495 | no |
| post-asinh transform 7/9 | 31,232 | 0.9 | 1.03854 | 1.23495 | no |
| post-asinh transform 8/9 | 31,232 | 0.9 | 1.03854 | 1.23495 | no |
| post-asinh transform 9/9 | 31,232 | 0.9 | 1.03854 | 1.23495 | no |
| allocation beta | 31,232 | 0.95 | 1.0372 | 1.23335 | no |
| knot refit | 31,232 | 0.95 | 1.0372 | 1.23335 | no |
| covariance shape 1/5 | 31,232 | 0.95 | 1.0372 | 1.23335 | no |
| covariance shape 2/5 | 31,232 | 0.95 | 1.0372 | 1.23335 | no |
| covariance shape 3/5 | 31,232 | 0.95 | 1.0372 | 1.23335 | no |
| covariance shape 4/5 | 31,232 | 0.95 | 1.0372 | 1.23335 | no |
| covariance shape 5/5 | 31,232 | 0.95 | 1.0372 | 1.23335 | no |
| post-asinh transform 1/9 | 31,232 | 0.95 | 1.0372 | 1.23335 | no |
| post-asinh transform 2/9 | 31,232 | 0.95 | 1.0372 | 1.23335 | no |
| post-asinh transform 3/9 | 31,232 | 0.95 | 1.0372 | 1.23335 | no |
| post-asinh transform 4/9 | 31,232 | 0.95 | 1.0372 | 1.23335 | no |
| post-asinh transform 5/9 | 31,232 | 0.95 | 1.0372 | 1.23335 | no |
| post-asinh transform 6/9 | 31,232 | 0.95 | 1.0372 | 1.23335 | no |
| post-asinh transform 7/9 | 31,232 | 0.95 | 1.0372 | 1.23335 | no |
| post-asinh transform 8/9 | 31,232 | 0.95 | 1.0372 | 1.23335 | no |
| post-asinh transform 9/9 | 31,232 | 0.95 | 1.0372 | 1.23335 | no |
| allocation beta | 31,232 | 1 | 1.03578 | 1.23167 | no |
| knot refit | 31,232 | 1 | 1.03578 | 1.23167 | no |
| covariance shape 1/5 | 31,232 | 1 | 1.03578 | 1.23167 | no |
| covariance shape 2/5 | 31,232 | 1 | 1.03578 | 1.23167 | no |
| covariance shape 3/5 | 31,232 | 1 | 1.03578 | 1.23167 | no |
| covariance shape 4/5 | 31,232 | 1 | 1.03578 | 1.23167 | no |
| covariance shape 5/5 | 31,232 | 1 | 1.03578 | 1.23167 | no |
| post-asinh transform 1/9 | 31,232 | 1 | 1.03578 | 1.23167 | no |
| post-asinh transform 2/9 | 31,232 | 1 | 1.03578 | 1.23167 | no |
| post-asinh transform 3/9 | 31,232 | 1 | 1.03578 | 1.23167 | no |
| post-asinh transform 4/9 | 31,232 | 1 | 1.03578 | 1.23167 | no |
| post-asinh transform 5/9 | 31,232 | 1 | 1.03578 | 1.23167 | no |
| post-asinh transform 6/9 | 31,232 | 1 | 1.03578 | 1.23167 | no |
| post-asinh transform 7/9 | 31,232 | 1 | 1.03578 | 1.23167 | no |
| post-asinh transform 8/9 | 31,232 | 1 | 1.03578 | 1.23167 | no |
| post-asinh transform 9/9 | 31,232 | 1 | 1.03578 | 1.23167 | no |
| allocation beta | 31,232 | 1 | 1.03578 | 1.23167 | no |
| knot refit (cached stationary) | 31,232 | 1 | 1.03578 | 1.23167 | no |
| refined parameter-search resolution | 31,232 | 1 | 1.03578 | 1.23167 | no |
| covariance shape 1/5 | 31,232 | 1 | 1.03578 | 1.23167 | no |
| covariance shape 2/5 | 31,232 | 1 | 1.03578 | 1.23167 | no |
| covariance shape 3/5 | 31,232 | 1 | 1.03578 | 1.23167 | no |
| covariance shape 4/5 | 31,232 | 1 | 1.03578 | 1.23167 | no |
| covariance shape 5/5 | 31,232 | 1 | 1.03578 | 1.23167 | no |
| post-asinh transform 1/9 | 31,232 | 1 | 1.03578 | 1.23167 | no |
| post-asinh transform 2/9 | 31,232 | 1 | 1.03578 | 1.23167 | no |
| post-asinh transform 3/9 | 31,232 | 1 | 1.03578 | 1.23167 | no |
| post-asinh transform 4/9 | 31,232 | 1 | 1.03578 | 1.23167 | no |
| post-asinh transform 5/9 | 31,232 | 1 | 1.03578 | 1.23167 | no |
| post-asinh transform 6/9 | 31,232 | 1 | 1.03578 | 1.23167 | no |
| post-asinh transform 7/9 | 31,232 | 1 | 1.03578 | 1.23167 | no |
| post-asinh transform 8/9 | 31,232 | 1 | 1.03578 | 1.23167 | no |
| post-asinh transform 9/9 | 31,232 | 1 | 1.03578 | 1.23167 | no |
| allocation beta | 31,232 | 1 | 1.03578 | 1.23167 | no |
| knot refit (cached stationary) | 31,232 | 1 | 1.03578 | 1.23167 | no |
| fixed point reached | 31,232 | 1 | 1.03578 | 1.23167 | no |
| prune proposal 31360->31232 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| covariance shape 1/5 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| covariance shape 2/5 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| covariance shape 3/5 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| covariance shape 4/5 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| covariance shape 5/5 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| post-asinh transform 1/9 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| post-asinh transform 2/9 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| post-asinh transform 3/9 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| post-asinh transform 4/9 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| post-asinh transform 5/9 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| post-asinh transform 6/9 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| post-asinh transform 7/9 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| post-asinh transform 8/9 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| post-asinh transform 9/9 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| allocation beta | 31,232 | 1 | 0.948267 | 1.12753 | no |
| knot refit | 31,232 | 1 | 0.948267 | 1.12753 | no |
| refined parameter-search resolution | 31,232 | 1 | 0.948267 | 1.12753 | no |
| covariance shape 1/5 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| covariance shape 2/5 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| covariance shape 3/5 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| covariance shape 4/5 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| covariance shape 5/5 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| post-asinh transform 1/9 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| post-asinh transform 2/9 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| post-asinh transform 3/9 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| post-asinh transform 4/9 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| post-asinh transform 5/9 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| post-asinh transform 6/9 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| post-asinh transform 7/9 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| post-asinh transform 8/9 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| post-asinh transform 9/9 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| allocation beta | 31,232 | 1 | 0.948267 | 1.12753 | no |
| knot refit (cached stationary) | 31,232 | 1 | 0.948267 | 1.12753 | no |
| refined parameter-search resolution | 31,232 | 1 | 0.948267 | 1.12753 | no |
| covariance shape 1/5 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| covariance shape 2/5 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| covariance shape 3/5 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| covariance shape 4/5 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| covariance shape 5/5 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| post-asinh transform 1/9 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| post-asinh transform 2/9 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| post-asinh transform 3/9 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| post-asinh transform 4/9 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| post-asinh transform 5/9 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| post-asinh transform 6/9 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| post-asinh transform 7/9 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| post-asinh transform 8/9 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| post-asinh transform 9/9 | 31,232 | 1 | 0.948267 | 1.12753 | no |
| allocation beta | 31,232 | 1 | 0.948267 | 1.12753 | no |
| knot refit (cached stationary) | 31,232 | 1 | 0.948267 | 1.12753 | no |
| fixed point reached | 31,232 | 1 | 0.948267 | 1.12753 | no |
| rejected prune | 31,232 | 1 | 0.948267 | 1.12753 | no |
| prune proposal 31360->31296 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| covariance shape 1/5 | 31,296 | 1 | 0.786592 | 1 | yes |
| covariance shape 2/5 | 31,296 | 1 | 0.786592 | 1 | yes |
| covariance shape 3/5 | 31,296 | 1 | 0.786592 | 1 | yes |
| covariance shape 4/5 | 31,296 | 1 | 0.786592 | 1 | yes |
| covariance shape 5/5 | 31,296 | 1 | 0.786592 | 1 | yes |
| post-asinh transform 1/9 | 31,296 | 1 | 0.786592 | 1 | yes |
| post-asinh transform 2/9 | 31,296 | 1 | 0.786592 | 1 | yes |
| post-asinh transform 3/9 | 31,296 | 1 | 0.786592 | 1 | yes |
| post-asinh transform 4/9 | 31,296 | 1 | 0.786592 | 1 | yes |
| post-asinh transform 5/9 | 31,296 | 1 | 0.786592 | 1 | yes |
| post-asinh transform 6/9 | 31,296 | 1 | 0.786592 | 1 | yes |
| post-asinh transform 7/9 | 31,296 | 1 | 0.786592 | 1 | yes |
| post-asinh transform 8/9 | 31,296 | 1 | 0.786592 | 1 | yes |
| post-asinh transform 9/9 | 31,296 | 1 | 0.786592 | 1 | yes |
| allocation beta | 31,296 | 0.9 | 0.774874 | 1 | yes |
| knot refit | 31,296 | 0.9 | 0.751131 | 1 | yes |
| covariance shape 1/5 | 31,296 | 0.9 | 0.751131 | 1 | yes |
| covariance shape 2/5 | 31,296 | 0.9 | 0.751131 | 1 | yes |
| covariance shape 3/5 | 31,296 | 0.9 | 0.751131 | 1 | yes |
| covariance shape 4/5 | 31,296 | 0.9 | 0.751131 | 1 | yes |
| covariance shape 5/5 | 31,296 | 0.9 | 0.751131 | 1 | yes |
| post-asinh transform 1/9 | 31,296 | 0.9 | 0.751131 | 1 | yes |
| post-asinh transform 2/9 | 31,296 | 0.9 | 0.751131 | 1 | yes |
| post-asinh transform 3/9 | 31,296 | 0.9 | 0.751131 | 1 | yes |
| post-asinh transform 4/9 | 31,296 | 0.9 | 0.751131 | 1 | yes |
| post-asinh transform 5/9 | 31,296 | 0.9 | 0.751131 | 1 | yes |
| post-asinh transform 6/9 | 31,296 | 0.9 | 0.751131 | 1 | yes |
| post-asinh transform 7/9 | 31,296 | 0.9 | 0.751131 | 1 | yes |
| post-asinh transform 8/9 | 31,296 | 0.9 | 0.751131 | 1 | yes |
| post-asinh transform 9/9 | 31,296 | 0.9 | 0.751131 | 1 | yes |
| allocation beta | 31,296 | 0.8 | 0.734718 | 1 | yes |
| knot refit | 31,296 | 0.8 | 0.711655 | 1 | yes |
| covariance shape 1/5 | 31,296 | 0.8 | 0.711655 | 1 | yes |
| covariance shape 2/5 | 31,296 | 0.8 | 0.711655 | 1 | yes |
| covariance shape 3/5 | 31,296 | 0.8 | 0.711655 | 1 | yes |
| covariance shape 4/5 | 31,296 | 0.8 | 0.711655 | 1 | yes |
| covariance shape 5/5 | 31,296 | 0.8 | 0.711655 | 1 | yes |
| post-asinh transform 1/9 | 31,296 | 0.8 | 0.711655 | 1 | yes |
| post-asinh transform 2/9 | 31,296 | 0.8 | 0.711655 | 1 | yes |
| post-asinh transform 3/9 | 31,296 | 0.8 | 0.711655 | 1 | yes |
| post-asinh transform 4/9 | 31,296 | 0.8 | 0.711655 | 1 | yes |
| post-asinh transform 5/9 | 31,296 | 0.8 | 0.711655 | 1 | yes |
| post-asinh transform 6/9 | 31,296 | 0.8 | 0.711655 | 1 | yes |
| post-asinh transform 7/9 | 31,296 | 0.8 | 0.711655 | 1 | yes |
| post-asinh transform 8/9 | 31,296 | 0.8 | 0.711655 | 1 | yes |
| post-asinh transform 9/9 | 31,296 | 0.8 | 0.711655 | 1 | yes |
| allocation beta | 31,296 | 0.7 | 0.70579 | 1 | yes |
| knot refit | 31,296 | 0.7 | 0.70579 | 1 | yes |
| covariance shape 1/5 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| covariance shape 2/5 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| covariance shape 3/5 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| covariance shape 4/5 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| covariance shape 5/5 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| post-asinh transform 1/9 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| post-asinh transform 2/9 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| post-asinh transform 3/9 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| post-asinh transform 4/9 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| post-asinh transform 5/9 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| post-asinh transform 6/9 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| post-asinh transform 7/9 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| post-asinh transform 8/9 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| post-asinh transform 9/9 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| allocation beta | 31,296 | 0.7 | 0.70579 | 1 | yes |
| knot refit (cached stationary) | 31,296 | 0.7 | 0.70579 | 1 | yes |
| refined parameter-search resolution | 31,296 | 0.7 | 0.70579 | 1 | yes |
| covariance shape 1/5 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| covariance shape 2/5 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| covariance shape 3/5 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| covariance shape 4/5 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| covariance shape 5/5 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| post-asinh transform 1/9 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| post-asinh transform 2/9 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| post-asinh transform 3/9 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| post-asinh transform 4/9 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| post-asinh transform 5/9 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| post-asinh transform 6/9 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| post-asinh transform 7/9 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| post-asinh transform 8/9 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| post-asinh transform 9/9 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| allocation beta | 31,296 | 0.7 | 0.70579 | 1 | yes |
| knot refit (cached stationary) | 31,296 | 0.7 | 0.70579 | 1 | yes |
| refined parameter-search resolution | 31,296 | 0.7 | 0.70579 | 1 | yes |
| covariance shape 1/5 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| covariance shape 2/5 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| covariance shape 3/5 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| covariance shape 4/5 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| covariance shape 5/5 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| post-asinh transform 1/9 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| post-asinh transform 2/9 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| post-asinh transform 3/9 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| post-asinh transform 4/9 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| post-asinh transform 5/9 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| post-asinh transform 6/9 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| post-asinh transform 7/9 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| post-asinh transform 8/9 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| post-asinh transform 9/9 | 31,296 | 0.7 | 0.70579 | 1 | yes |
| allocation beta | 31,296 | 0.7 | 0.70579 | 1 | yes |
| knot refit (cached stationary) | 31,296 | 0.7 | 0.70579 | 1 | yes |
| fixed point reached | 31,296 | 0.7 | 0.70579 | 1 | yes |
| accepted prune | 31,296 | 0.7 | 0.70579 | 1 | yes |
| cold restart 31296->31232 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| covariance shape 1/5 | 31,232 | 0.7 | 0.952689 | 1.13293 | no |
| covariance shape 2/5 | 31,232 | 0.7 | 0.952689 | 1.13293 | no |
| covariance shape 3/5 | 31,232 | 0.7 | 0.952689 | 1.13293 | no |
| covariance shape 4/5 | 31,232 | 0.7 | 0.952689 | 1.13293 | no |
| covariance shape 5/5 | 31,232 | 0.7 | 0.952689 | 1.13293 | no |
| post-asinh transform 1/9 | 31,232 | 0.7 | 0.952689 | 1.13293 | no |
| post-asinh transform 2/9 | 31,232 | 0.7 | 0.952689 | 1.13293 | no |
| post-asinh transform 3/9 | 31,232 | 0.7 | 0.952689 | 1.13293 | no |
| post-asinh transform 4/9 | 31,232 | 0.7 | 0.952689 | 1.13293 | no |
| post-asinh transform 5/9 | 31,232 | 0.7 | 0.952689 | 1.13293 | no |
| post-asinh transform 6/9 | 31,232 | 0.7 | 0.952689 | 1.13293 | no |
| post-asinh transform 7/9 | 31,232 | 0.7 | 0.952689 | 1.13293 | no |
| post-asinh transform 8/9 | 31,232 | 0.7 | 0.952689 | 1.13293 | no |
| post-asinh transform 9/9 | 31,232 | 0.7 | 0.952689 | 1.13293 | no |
| allocation beta | 31,232 | 0.8 | 0.944032 | 1.12264 | no |
| knot refit | 31,232 | 0.8 | 0.938249 | 1.11576 | no |
| covariance shape 1/5 | 31,232 | 0.8 | 0.938249 | 1.11576 | no |
| covariance shape 2/5 | 31,232 | 0.8 | 0.938249 | 1.11576 | no |
| covariance shape 3/5 | 31,232 | 0.8 | 0.938249 | 1.11576 | no |
| covariance shape 4/5 | 31,232 | 0.8 | 0.938249 | 1.11576 | no |
| covariance shape 5/5 | 31,232 | 0.8 | 0.938249 | 1.11576 | no |
| post-asinh transform 1/9 | 31,232 | 0.8 | 0.938249 | 1.11576 | no |
| post-asinh transform 2/9 | 31,232 | 0.8 | 0.938249 | 1.11576 | no |
| post-asinh transform 3/9 | 31,232 | 0.8 | 0.938249 | 1.11576 | no |
| post-asinh transform 4/9 | 31,232 | 0.8 | 0.938249 | 1.11576 | no |
| post-asinh transform 5/9 | 31,232 | 0.8 | 0.938249 | 1.11576 | no |
| post-asinh transform 6/9 | 31,232 | 0.8 | 0.938249 | 1.11576 | no |
| post-asinh transform 7/9 | 31,232 | 0.8 | 0.938249 | 1.11576 | no |
| post-asinh transform 8/9 | 31,232 | 0.8 | 0.938249 | 1.11576 | no |
| post-asinh transform 9/9 | 31,232 | 0.8 | 0.938249 | 1.11576 | no |
| allocation beta | 31,232 | 0.8 | 0.938249 | 1.11576 | no |
| knot refit | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| covariance shape 1/5 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| covariance shape 2/5 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| covariance shape 3/5 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| covariance shape 4/5 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| covariance shape 5/5 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| post-asinh transform 1/9 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| post-asinh transform 2/9 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| post-asinh transform 3/9 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| post-asinh transform 4/9 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| post-asinh transform 5/9 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| post-asinh transform 6/9 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| post-asinh transform 7/9 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| post-asinh transform 8/9 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| post-asinh transform 9/9 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| allocation beta | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| knot refit | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| refined parameter-search resolution | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| covariance shape 1/5 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| covariance shape 2/5 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| covariance shape 3/5 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| covariance shape 4/5 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| covariance shape 5/5 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| post-asinh transform 1/9 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| post-asinh transform 2/9 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| post-asinh transform 3/9 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| post-asinh transform 4/9 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| post-asinh transform 5/9 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| post-asinh transform 6/9 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| post-asinh transform 7/9 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| post-asinh transform 8/9 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| post-asinh transform 9/9 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| allocation beta | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| knot refit (cached stationary) | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| refined parameter-search resolution | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| covariance shape 1/5 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| covariance shape 2/5 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| covariance shape 3/5 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| covariance shape 4/5 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| covariance shape 5/5 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| post-asinh transform 1/9 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| post-asinh transform 2/9 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| post-asinh transform 3/9 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| post-asinh transform 4/9 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| post-asinh transform 5/9 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| post-asinh transform 6/9 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| post-asinh transform 7/9 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| post-asinh transform 8/9 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| post-asinh transform 9/9 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| allocation beta | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| knot refit (cached stationary) | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| fixed point reached | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| prune proposal 31296->31232 | 31,232 | 0.8 | 0.936655 | 1.11386 | no |
| covariance shape 1/5 | 31,232 | 0.7 | 0.953461 | 1.13375 | no |
| covariance shape 2/5 | 31,232 | 0.7 | 0.953461 | 1.13375 | no |
| covariance shape 3/5 | 31,232 | 0.7 | 0.953461 | 1.13375 | no |
| covariance shape 4/5 | 31,232 | 0.7 | 0.953461 | 1.13375 | no |
| covariance shape 5/5 | 31,232 | 0.7 | 0.953461 | 1.13375 | no |
| post-asinh transform 1/9 | 31,232 | 0.7 | 0.953461 | 1.13375 | no |
| post-asinh transform 2/9 | 31,232 | 0.7 | 0.953461 | 1.13375 | no |
| post-asinh transform 3/9 | 31,232 | 0.7 | 0.953461 | 1.13375 | no |
| post-asinh transform 4/9 | 31,232 | 0.7 | 0.953461 | 1.13375 | no |
| post-asinh transform 5/9 | 31,232 | 0.7 | 0.953461 | 1.13375 | no |
| post-asinh transform 6/9 | 31,232 | 0.7 | 0.953461 | 1.13375 | no |
| post-asinh transform 7/9 | 31,232 | 0.7 | 0.953461 | 1.13375 | no |
| post-asinh transform 8/9 | 31,232 | 0.7 | 0.953461 | 1.13375 | no |
| post-asinh transform 9/9 | 31,232 | 0.7 | 0.953461 | 1.13375 | no |
| allocation beta | 31,232 | 0.8 | 0.951036 | 1.13088 | no |
| knot refit | 31,232 | 0.8 | 0.947078 | 1.12619 | no |
| covariance shape 1/5 | 31,232 | 0.8 | 0.947078 | 1.12619 | no |
| covariance shape 2/5 | 31,232 | 0.8 | 0.947078 | 1.12619 | no |
| covariance shape 3/5 | 31,232 | 0.8 | 0.947078 | 1.12619 | no |
| covariance shape 4/5 | 31,232 | 0.8 | 0.947078 | 1.12619 | no |
| covariance shape 5/5 | 31,232 | 0.8 | 0.947078 | 1.12619 | no |
| post-asinh transform 1/9 | 31,232 | 0.8 | 0.947078 | 1.12619 | no |
| post-asinh transform 2/9 | 31,232 | 0.8 | 0.947078 | 1.12619 | no |
| post-asinh transform 3/9 | 31,232 | 0.8 | 0.947078 | 1.12619 | no |
| post-asinh transform 4/9 | 31,232 | 0.8 | 0.947078 | 1.12619 | no |
| post-asinh transform 5/9 | 31,232 | 0.8 | 0.947078 | 1.12619 | no |
| post-asinh transform 6/9 | 31,232 | 0.8 | 0.947078 | 1.12619 | no |
| post-asinh transform 7/9 | 31,232 | 0.8 | 0.947078 | 1.12619 | no |
| post-asinh transform 8/9 | 31,232 | 0.8 | 0.947078 | 1.12619 | no |
| post-asinh transform 9/9 | 31,232 | 0.8 | 0.947078 | 1.12619 | no |
| allocation beta | 31,232 | 0.9 | 0.945135 | 1.12388 | no |
| knot refit | 31,232 | 0.9 | 0.945135 | 1.12388 | no |
| covariance shape 1/5 | 31,232 | 0.9 | 0.945135 | 1.12388 | no |
| covariance shape 2/5 | 31,232 | 0.9 | 0.945135 | 1.12388 | no |
| covariance shape 3/5 | 31,232 | 0.9 | 0.945135 | 1.12388 | no |
| covariance shape 4/5 | 31,232 | 0.9 | 0.945135 | 1.12388 | no |
| covariance shape 5/5 | 31,232 | 0.9 | 0.945135 | 1.12388 | no |
| post-asinh transform 1/9 | 31,232 | 0.9 | 0.945135 | 1.12388 | no |
| post-asinh transform 2/9 | 31,232 | 0.9 | 0.945135 | 1.12388 | no |
| post-asinh transform 3/9 | 31,232 | 0.9 | 0.945135 | 1.12388 | no |
| post-asinh transform 4/9 | 31,232 | 0.9 | 0.945135 | 1.12388 | no |
| post-asinh transform 5/9 | 31,232 | 0.9 | 0.945135 | 1.12388 | no |
| post-asinh transform 6/9 | 31,232 | 0.9 | 0.945135 | 1.12388 | no |
| post-asinh transform 7/9 | 31,232 | 0.9 | 0.945135 | 1.12388 | no |
| post-asinh transform 8/9 | 31,232 | 0.9 | 0.945135 | 1.12388 | no |
| post-asinh transform 9/9 | 31,232 | 0.9 | 0.945135 | 1.12388 | no |
| allocation beta | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| knot refit | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| covariance shape 1/5 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| covariance shape 2/5 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| covariance shape 3/5 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| covariance shape 4/5 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| covariance shape 5/5 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| post-asinh transform 1/9 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| post-asinh transform 2/9 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| post-asinh transform 3/9 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| post-asinh transform 4/9 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| post-asinh transform 5/9 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| post-asinh transform 6/9 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| post-asinh transform 7/9 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| post-asinh transform 8/9 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| post-asinh transform 9/9 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| allocation beta | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| knot refit (cached stationary) | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| refined parameter-search resolution | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| covariance shape 1/5 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| covariance shape 2/5 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| covariance shape 3/5 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| covariance shape 4/5 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| covariance shape 5/5 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| post-asinh transform 1/9 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| post-asinh transform 2/9 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| post-asinh transform 3/9 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| post-asinh transform 4/9 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| post-asinh transform 5/9 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| post-asinh transform 6/9 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| post-asinh transform 7/9 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| post-asinh transform 8/9 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| post-asinh transform 9/9 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| allocation beta | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| knot refit (cached stationary) | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| refined parameter-search resolution | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| covariance shape 1/5 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| covariance shape 2/5 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| covariance shape 3/5 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| covariance shape 4/5 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| covariance shape 5/5 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| post-asinh transform 1/9 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| post-asinh transform 2/9 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| post-asinh transform 3/9 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| post-asinh transform 4/9 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| post-asinh transform 5/9 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| post-asinh transform 6/9 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| post-asinh transform 7/9 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| post-asinh transform 8/9 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| post-asinh transform 9/9 | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| allocation beta | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| knot refit (cached stationary) | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| fixed point reached | 31,232 | 0.8 | 0.942958 | 1.12128 | no |
| rejected prune | 31,232 | 0.8 | 0.936655 | 1.11386 | no |

Covariance adjustment from empirical whitening: `[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]`

Post-asinh matrix: `[[3.0759326992014087,0.0,0.0],[0.0,3.0759326992014087,0.0],[0.0,0.0,3.0759326992014087]]`

## Reproduction

```text
node scripts/run-ml-python.mjs ml/joint_optimize_multidimensional_return_knots.py
```
