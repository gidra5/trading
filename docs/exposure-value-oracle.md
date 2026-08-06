# Exposure-value oracle execution model

The displayed value-oracle return is one coherent full-window control result:

`oracle return = exp(Q_0(initialExposure)) - 1 = E_T / E_0 - 1`.

The selected window supplies all prices. Training targets default to a fixed 60-second holding period and one-hour rolling value horizon. Truncate mode shortens targets near the selected window's end. The separately reported oracle return remains one coherent full-window policy: the last candle is its terminal state and forcibly rebalances to exact zero exposure.

## Bellman execution

At each scored timestamp, the rolling oracle forces target `a` for `H` price moves. Marked exposure is allowed to drift with price and maintenance during that initial hold; there are no intermediate target-restoring trades and therefore no intermediate rebalancing friction. After the forced hold, the perfect continuation may choose a new target on every candle until the total `T`-move horizon is exhausted. It then closes the remaining exposure to exact cash. This forced-`H`, one-candle-continuation, terminal-closeout definition is the schema-v5 raw-target contract.

Let `h_(t,k)(a)` and `d_(t,k)(a)` be the log return and drifted exposure obtained by passively holding target `a` from `t` for `k` moves. Let `R_t(x -> b)` be the exact fee-aware equity factor for rebalancing current exposure `x` to target `b`. The independent recurrence is

`V_(t,0)(x) = log R_t(x -> 0)`,

`V_(t,k)(x) = max_b [log R_t(x -> b) + h_(t,1)(b) + V_(t+1,k-1)(d_(t,1)(b))]`,

`F_(t,H,T)(a) = h_(t,H')(a) + V_(t+H',T-H')(d_(t,H')(a))`,

where `H' = min(H, T, remaining price moves)`. Equivalently, the one-candle hold/drift can be folded into a time-indexed transition return. Crucially, `H` applies only to the initially forced action; it is not reapplied at every continuation decision.

The coherent full-window inspector path remains a separate policy reconstruction problem. It compares every grid target with an explicit **no-trade** continuation from its drifted exposure and trades only when the continuation after friction is better.

The soft training target is transition-aware. Let `F_t(a)` be the forced-action value after selecting target `a`, including its untouched `H`-candle evolution and optimal continuation. For any input/current exposure `x`, the action value and policy are

`Q_t(x,a) = log R(x→a) + F_t(a)`

`p_oracle,t(a|x) = softmax_a(Q_t(x,a) / oracleTemperature)`,

where `R(x→a)` is the exact fee-aware equity factor. Thus the target is a two-argument current-to-target policy, not one unconditioned target distribution. The axes deliberately use different grids: target actions `a` stay on the executable `[-100, 100]` grid, while current states `x` span the complete `[-maxEffectiveExposure, +maxEffectiveExposure]` interval, including the default liquidation boundary at `±250`. The state grid uses the same point count as an odd action grid (or one extra point for an even grid), always containing zero and both boundaries without multiplying per-candle training work. The operational prior over `x` is uniform across this full state grid, so drift or prior strategy errors remain inside the learned state space.

The CPU solver retains square-root-spaced Bellman checkpoints and recomputes each block while reconstructing the policy. Its memory is `O(actionGrid * sqrt(decisions))`; separable buy/sell prefix/suffix scans reduce every conditional statistic over both axes in `O(actionGrid + stateGrid)` rather than materializing their product. The CUDA solver computes untouched forced-hold transitions in parallel, initializes the terminal closeout in an ordered kernel, runs parallel residue chains with the same separable scans, and reconstructs decisions against the exact marked current exposure rather than a nearest grid state.

## Holding and continuation factorization

The initial forced action is evaluated as two separate terms:

`F_(t,H,T)(a) = H_(t,H)(a) + V_(t+H,T-H)(drift_(t,H)(a))`.

Use normalized quote and marked-asset amounts immediately after the action,

`m(a) = [Q, A] = [1 - a, a]`.

During a passive hold their signs do not change, so each financing category
has two precomputable multipliers: `g_q` for quote and `g_a` for marked asset.
`g_a` includes the endpoint price ratio and, for a short, compounded asset
borrow maintenance. `g_q` is one for owned quote and applies the compounded
quote-borrow maintenance factor only to negative quote debt.
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
liquidation branch is retained as a candidate continuation. Their action value
is `-Infinity`, which maps to probability exactly zero. Finite negative log
returns remain valid action values and participate normally in the softmax.

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

The rolling target queries those prefix/suffix aggregates directly at the exact
`drift_t(a)` exposure. It does not interpolate a continuation sampled at nearby
grid states. Each horizon level therefore passes its forced-action row to the
preceding level; two linear scans turn that row into the complete sell/buy
transition envelope needed by all 255 exact endpoint queries. For the
unconditional 255-action target this retains only the discrete path row for
each forced initial action. It does not represent the optimal policy over every
possible continuous starting exposure. This distinction is intentional: the
full-window no-trade inspector path remains a separate coherent control
problem, while persisted raw training rows use the exact transition-row
contract.

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
second full-memory pass. The same exact transition-row query is used by the
non-fused rolling path and the general statistics path, so inspector statistics
and persisted unconditional targets share one Bellman definition.

## Differentiable training implementation

`ml/differentiable_exposure_value_oracle.py` implements the same single-window
base-action oracle as a batched PyTorch module without replacing the production
CPU or CUDA generators. It accepts either positive relative prices or future
log returns, retains the hard Bellman maximum and hard liquidation boundary,
and returns action values, logits, and normalized probabilities inside the
autograd graph.

The forward pass is exact rather than a log-sum-exp relaxation. It is therefore
piecewise differentiable: gradients follow the selected continuation branch,
while ties, buy/sell crossings, and liquidation crossings are nonsmooth. The
separable prefix/suffix continuation scan keeps the normal path linear in the
number of actions; unusual grids with invalid fee denominators use a dense
correctness fallback. The Python tests compare fixed and seeded-random paths
against an independent literal amount-space brute-force implementation and
check an oracle-policy gradient against a central finite difference.

Oracle input prices and probability output cross the native boundary through a
reused pair of pinned host staging slots. Slots alternate between calls, while
the outer dataset worker pipeline can persist/compress one shared output as the
next day is prepared. Distribution-only calls also skip the eleven unused
diagnostic-column transfers.

The compact fused path must not allocate the general Float64 Bellman tables.
On the 255-action, 60-second hold, 3,600-second horizon production case this
first removed about 866 MiB of incremental device allocation (approximately
1,377 MiB down to 511 MiB in a concurrent-training profile). Fusing the final
softmax removes another 88,128,000-byte forced-action table, bringing the same
allocation estimate to about 427 MiB. The generated probability bytes remain
identical to the prior compact implementation.

The final base log-density is empirically sparse enough for an optional lossy
storage representation, but not an exact handful-of-lines representation.
Across 768 sampled
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
