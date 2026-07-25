# Exposure-value oracle execution model

The displayed value-oracle return is one coherent full-window control result:

`oracle return = exp(Q_0(initialExposure)) - 1 = E_T / E_0 - 1`.

The selected window supplies all prices. Training targets default to a fixed 60-second holding period and one-hour rolling value horizon. Truncate mode shortens targets near the selected window's end. The separately reported oracle return remains one coherent full-window policy: the last candle is its terminal state and forcibly rebalances to exact zero exposure.

## Bellman execution

At each decision, the oracle may rebalance once to a target from the configured exposure grid. It then leaves the resulting quote and asset quantities alone for `H` price moves. Marked exposure is allowed to drift with price and maintenance; there are no intermediate target-restoring trades and therefore no intermediate rebalancing friction. At the next decision the oracle compares every grid target with an explicit **no-trade** continuation from its exact drifted exposure, and trades only when the continuation value after friction is better. The final continuation value for current exposure `a` is the log of its fee-aware rebalance factor to zero.

The soft training target is transition-aware. Let `F_t(a)` be the forced-action value after selecting target `a`, including its untouched `H`-candle evolution and optimal continuation. For any input/current exposure `x`, the action value and policy are

`Q_t(x,a) = log R(x→a) + F_t(a)`

`p_oracle,t(a|x) = softmax_a(Q_t(x,a) / oracleTemperature)`,

where `R(x→a)` is the exact fee-aware equity factor. Thus the target is a two-argument current-to-target policy, not one unconditioned target distribution. The axes deliberately use different grids: target actions `a` stay on the executable `[-100, 100]` grid, while current states `x` span the complete `[-maxEffectiveExposure, +maxEffectiveExposure]` interval, including the default liquidation boundary at `±250`. The state grid uses the same point count as an odd action grid (or one extra point for an even grid), always containing zero and both boundaries without multiplying per-candle training work. The operational prior over `x` is uniform across this full state grid, so drift or prior strategy errors remain inside the learned state space.

The CPU solver retains square-root-spaced Bellman checkpoints and recomputes each block while reconstructing the policy. Its memory is `O(actionGrid * sqrt(decisions))`; separable buy/sell prefix/suffix scans reduce every conditional statistic over both axes in `O(actionGrid + stateGrid)` rather than materializing their product. The CUDA solver computes untouched forced-hold transitions in parallel, initializes the terminal closeout in an ordered kernel, runs parallel residue chains with the same separable scans, and reconstructs decisions against the exact marked current exposure rather than a nearest grid state.

## Holding and continuation factorization

The rolling target is already evaluated as two separate terms:

`F_t(a) = H_t(a) + V_(t+H)(drift_t(a))`.

Use normalized quote and marked-asset amounts immediately after the action,

`m(a) = [Q, A] = [1 - a, a]`.

During a passive hold their signs do not change, so each financing category
has two precomputable multipliers: `g_q` for quote and `g_a` for marked asset.
`g_a` includes the endpoint price ratio and, for a short, compounded asset
borrow maintenance. `g_q` applies the quote lend or borrow maintenance factor.
The exact surviving endpoint is therefore

`[Q', A'] = [g_q * (1 - a), g_a * a]`,

`W_t(a) = Q' + A' = g_q * (1 - a) + g_a * a`,

`H_t(a) = log W_t(a)`,

`drift_t(a) = A' / W_t(a)`.

Thus passive holding is a linear map in the two amount coordinates and its
wealth is affine in starting exposure within each financing category. Only
the final log score and the conversion back to exposure are nonlinear. This
remains exact when maintenance compounds: the debt quantity is merely
multiplied by a precomputable constant for the interval.

The implementation precomputes the price extrema and endpoint ratio once per
timestamp and financing category, then materializes the 255 hold values and
endpoint exposures once. They are reused by every rolling-horizon level.
Intermediate price extrema reduce liquidation to a continuous feasible
starting-exposure interval. Actions outside it have invalid value; no
liquidation branch is retained as a candidate continuation.

Fixed-target proportional-fee transitions have the same amount-space
factorization. If `[Q, A]` is the current marked amount vector, `f` is friction,
and `G(y)` is future wealth per unit equity after choosing target `y`, then

`sell(x -> y): G(y) * [Q + (1 - f) * A] / (1 - f * y)`, for `y <= x`,

`buy(x -> y): G(y) * [(1 - f) * Q + A] / (1 - f + f * y)`, for `y >= x`.

For a selected target sequence `y_0, ..., y_n`, form one amount row per held
interval,

`B_k = [g_(q,k) * (1 - y_k), g_(a,k) * y_k]`.

Stacking these rows produces an `n by 2` transition-path matrix. Dotting each
row with the appropriate buy/sell fee coefficient for `y_(k+1)` produces one
scalar wealth multiplier per transition. Their product (or, more stably, the
sum of their logarithms) reduces the complete path. If the first target is
forced from normalized equity, the reduction yields one scalar value for that
initial action. If the path starts from arbitrary current amounts, the suffix
product scales the first fee coefficient and the entire path reduces to one
terminal `[c_q, c_a]` tuple.

Consequently, a batch of `N` already-selected paths can be represented as an
`N by n by 2` amount tensor and reduced cheaply into `N` action values or `N`
coefficient tuples. No continuous exposure envelope is needed merely to
evaluate those paths.

Selecting the optimal successor of every path row remains a max operation over
the alternative targets. Maximizing all discrete targets is equivalent to
querying a prepared sell-prefix maximum and buy-suffix maximum. The
implementation already uses the log-domain form of those two linear scans
instead of a dense 255 by 255 transition matrix. It evaluates the resulting
layered path DAG directly without materializing all backpointer paths.

The remaining rolling-target approximation is the lookup after passive drift:
the current implementation linearly interpolates the sampled continuation log
values at `drift_t(a)`. Querying the prefix/suffix aggregates directly at that
exact exposure would remove this interpolation. For the unconditional
255-action target, this requires only the discrete path row for each forced
initial action; it does not require representing the optimal policy over every
possible continuous starting exposure. An exact continuous no-trade policy
would still be an upper envelope of amount-space linear terms, but that is a
broader object than the unconditional training target.

A 60-candle one-minute path contains at most 60 passive-hold intervals and 59
subsequent transitions. This does not strictly cap the number of distinct
affine branches across all possible initial target exposures, because different
initial targets can select different suffix paths, but it makes the
unconditional row substantially smaller in practice. Across 867 sampled
one-minute rows from three market regimes, a piecewise-affine relative-wealth
encoding used about 11 segments on average and 31 at worst for absolute wealth
error `1e-6`. At error `3e-6`, it used about 6.5 segments on average and 21 at
worst; every sampled modal action was preserved, mean row KL was `2.4e-9`, and
worst row KL was `1.8e-8`. A CSR row of `(endIndex, intercept, slope)` Float32
segments would average about 63 bytes before compression, compared with 1,020
bytes for 255 Float32 probabilities. It can be decoded directly on the training
GPU without retaining the transition paths.

The distribution-only CUDA path follows each complete horizon diagonal in
warp-local memory. Its final level performs the softmax directly from the
warp's forced-action registers, avoiding an intermediate device table and a
second full-memory pass.

The compact fused path must not allocate the general Float64 Bellman tables.
On the 255-action, 60-second hold, 3,600-second horizon production case this
first removed about 866 MiB of incremental device allocation (approximately
1,377 MiB down to 511 MiB in a concurrent-training profile). Fusing the final
softmax removes another 88,128,000-byte forced-action table, bringing the same
allocation estimate to about 427 MiB. The generated probability bytes remain
identical to the prior compact implementation.

The final base log-density is empirically sparse enough for an optional lossy
storage representation, but not an exact handful-of-lines representation
under the current discrete interpolation contract. Across 768 sampled
one-second rows from three market regimes, adaptive piecewise-linear
log-probability interpolation over the full 255-action support required about
17 knots on average for mean row KL `6.1e-5`. Restricting to the trained
`[-100, 100]` support required about 6.4 knots for mean row KL `3.2e-5`.
This is promising as a separately versioned approximate streaming format; it
must not replace the retained raw oracle probabilities until an end-to-end
model-quality comparison accepts the approximation.

Exact zero must be present in the grid. This makes staying in cash an executable baseline. The solver rejects a `Q_0` result below the immediate-cash baseline; with zero initial exposure and non-negative maintenance inputs, terminal return therefore cannot be negative. A nonzero input exposure can lose only the unavoidable cost of reaching the cash baseline when no profitable path exists.

Terminal optimality does not imply monotonic marked equity. Friction is charged on traded notional, so its equity effect scales with exposure: at `100×` exposure and `0.175%` friction, closing to cash costs `17.5%` of equity and a direct `+100× → -100×` rebalance costs about `29.8%`. A terminal-return optimizer may accept several such interim costs before a later gain, so peak-to-trough drawdown can be much larger than the raw friction rate even though `Q_0(0)` is positive. Guaranteeing a friction-sized drawdown would require a separate pathwise drawdown constraint or a risk-adjusted objective; it is not a property of the stated terminal `Q_0` objective.

The inspector uses the same reconstructed exposure path for the price-chart Oracle band and the exposure/equity chart. When detail candles are loaded for a zoomed viewport, their exact exposure and equity samples replace the overview samples over the same timestamps.

Minimum order size, quantity step, and notional filters are deliberately not part of this scale-free oracle or predictor. The value process continues to track relative equity from an arbitrary unit starting value. Actual balances and fixed exchange order limits remain execution-layer concerns; adding them here would make the policy depend on an absolute initial balance and would destroy the current homogeneous exposure state.

## Binance comparison

The implementation follows the recurrence in `tasks.md`, with these Binance-oriented interpretations:

- Rebalances and forced liquidation debit the configured friction, corresponding to execution commission. Binance reports commission and commission asset on fills.
- Borrow maintenance inputs are hourly in the UI and optimizer. They are converted to the equivalent compounded candle rate before the recurrence. Binance states that margin interest begins immediately and is accrued hourly; live rates can change, while an oracle run uses the supplied constant rate.
- Negative quote and negative asset balances are treated as liabilities. Binance repayment applies to interest before principal; the homogeneous exposure state tracks their combined economic cost rather than a separate repayment ledger.
- The configured maximum effective exposure is a research approximation for forced liquidation. Binance liquidation is based on account margin level and differs between Classic and Pro modes, with collateral haircuts and account-specific thresholds. It should not be presented as an exact Binance liquidation-engine replica.

Primary references:

- [Binance Margin Trading Best Practice](https://developers.binance.com/en/docs/products/margin-trading/best-practice)
- [Binance Margin Trading Introduction](https://developers.binance.com/en/docs/products/margin-trading/Introduction)
- [Binance Margin Common Definition](https://developers.binance.com/en/docs/products/margin-trading/common-definition)
