# Exposure-value oracle execution model

The displayed value-oracle return is one coherent full-window control result:

`oracle return = exp(Q_0(initialExposure)) - 1 = E_T / E_0 - 1`.

The selected window supplies all prices. The value horizon defaults to **Full selected window**, so a three-day one-second example uses all 259,199 price moves between its 259,200 candles without requiring the user to enter `259200`. The last candle is the terminal state and forcibly rebalances to exact zero exposure. A fixed rolling horizon remains available for experiments; soft-Q distributions used by the distillation loss remain separate from the reported full-window path.

## Bellman execution

At each decision, the oracle may rebalance once to a target from the configured exposure grid. It then leaves the resulting quote and asset quantities alone for `H` price moves. Marked exposure is allowed to drift with price and maintenance; there are no intermediate target-restoring trades and therefore no intermediate rebalancing friction. At the next decision the oracle compares every grid target with an explicit **no-trade** continuation from its exact drifted exposure, and trades only when the continuation value after friction is better. The final continuation value for current exposure `a` is the log of its fee-aware rebalance factor to zero.

The soft training target is transition-aware. Let `F_t(a)` be the forced-action value after selecting target `a`, including its untouched `H`-candle evolution and optimal continuation. For any input/current exposure `x`, the action value and policy are

`Q_t(x,a) = log R(x→a) + F_t(a)`

`p_oracle,t(a|x) = softmax_a(Q_t(x,a) / oracleTemperature)`,

where `R(x→a)` is the exact fee-aware equity factor. Thus the target is a two-argument current-to-target policy, not one unconditioned target distribution. The exposure grid is used for both arguments during training. The operational prior over input `x` is uniform, so sudden external changes can move execution to any represented current exposure without leaving the learned state space.

The CPU solver retains square-root-spaced Bellman checkpoints and recomputes each block while reconstructing the policy. Its memory is `O(grid * sqrt(decisions))`; its separable buy/sell scans avoid a quadratic target search. The CUDA solver computes untouched forced-hold transitions in parallel, initializes the terminal closeout in an ordered kernel, runs parallel residue chains with the same separable scans, and reconstructs decisions against the exact marked current exposure rather than a nearest grid state.

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
