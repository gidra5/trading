# Local market refinement and distribution support

## Paired local refresh

`oracle_local_market_refresh.py` recomputes two Bellman layers against the same archived distant continuation. The actual root receives a fresh NSF forecast. At each sampled next history, the audit compares forecasting from that history with forecasting from its matched retained history. Both calls use the same random draws. Retained price moves are reanchored to the actual account price; the audit does not introduce a jump back to the retained node's price level.

The three variants are the archived continuation, a locally resampled retained-history continuation, and a locally refreshed sampled-history continuation. Future menus and the terminal date stay fixed. Root selection searches the catalog and retained-order subsets; the logged action, which can include an interior sample, is cross-scored separately. This is a local sensitivity measurement, not a reference to the full continuous optimum.

Three tests pass: identical histories give identical fields, an omitted predictive feature changes continuation value under common random draws, and reanchoring preserves identical relative laws across different price levels.

The first learned-model audit used the fine 41-exposure field at offsets 30, 300 and 600 seconds, with two random seeds, both the actual carried account and a cash account, and eight conditional samples. It took **67.09 seconds on CPU**. All 12 state/seed cases selected the same action in every variant. The six logged actions also matched the freshly selected catalog action. The measured cross-scored action gap was zero in these cases.

The largest fresh-versus-retained best log-value shift was **0.01541 basis points** for the carried account and **0.00688 basis points** for cash. Fresh versus retained conditional mean returns differed by approximately 0.0043–0.0121 basis points across the six batches. The price-coordinate projection discrepancy itself reached 16.83 basis points; it is not the same quantity as conditional return-law error after reanchoring.

Artifacts: `data/backtests/oracle-dense-local-refresh-audit/results.json`. These checks do not cover accumulated projection error through all remaining layers, rare unsampled outcomes, other price paths, or continuous action coverage.

```powershell
.venv-ml/Scripts/python.exe ml/audit_oracle_local_market_refresh.py --map data/backtests/oracle-dense-exposure41-replay/initial --replay data/backtests/oracle-dense-exposure41-prepared --output data/backtests/oracle-dense-local-refresh-audit-new --offsets 30 300 600 --replicates 2
```

## A missing support assumption for gamma=1

The earlier bounded-price hypothesis is not enforced by the NSF generator. Its Gaussian price emission has strictly positive scale. `next_observations` exponentiates it, rounds to the price tick, and applies only a lower price clamp. The ideal Gaussian predictive law therefore retains an unbounded upper price tail. Finite sampled trees do not cover that support.

This distinction is consequential for log utility. `CRRAUtility.lean` explicitly excludes zero wealth from the finite-utility domain when gamma is at least one; zero wealth must have negative-infinite log utility. The numeric Oracle kernel likewise rejects a liquidation branch. Its liquidation check occurs before pending fills.

For a post-decision short account with quote balance `q > 0`, funded asset quantity `a_f < 0`, taker fee `c`, and lower maintenance bound `-L`, effective equity after a price move to `P` is

\[
\widetilde W(P)=q-\frac{-a_f}{1-c}P.
\]

The maintenance boundary occurs at

\[
P_{\rm liq}=\frac{Lq(1-c)}{(1+L)(-a_f)}.
\]

Above this boundary, Oracle liquidates the account. An ideal positive-scale Gaussian log-price law gives every upper-tail interval beyond that finite boundary positive probability. With a finite terminal horizon and bounded aggregate quote flow, terminal cash has a finite upper bound, so its positive log-utility part is bounded. A positive-probability zero-wealth event therefore gives negative-infinite expected log utility, rather than a small finite tail correction. Equivalently, a policy requiring support-wide survival cannot admit such a post-state.

`oracle_log_support.py` constructs a tick-aligned witness and verifies it through the actual `move_arrive` execution function. Four tests cover the short boundary, leveraged-long downside boundary, funding/fees and nonleveraged inventory. In the fine replay, the first short state at update 10 has a maintenance threshold of **97,247.8661 USDT**. The execution kernel confirms liquidation at **97,247.89 USDT**. This is a support check; no probability magnitude has been estimated. The audit is in `oracle-dense-exposure41-prepared/log-support-audit.json`.

A specified hard upper price bound could remove this particular short-inventory witness if it lies below the boundary. A larger bound may retain it. Pending-order outcomes, downside support, minimum trade sizes and strict terminal closing need their own feasibility checks. Absence of the simple inventory witness is not a complete admissibility certificate.

The user has been asked for the missing numerical BTCUSDT upper bound. The running replay continues to evaluate the existing finite sampled law; it is not being silently changed to another objective or support. Agreement between finite-grid implementations must not be described as an optimality certificate for the ideal unbounded NSF law.
