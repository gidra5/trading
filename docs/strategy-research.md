# Strategy Research Notes

The target remains the "perfect margin trader": full long on every upward move, full
short on every downward move, scaled by maximum leverage. This is an oracle benchmark,
not an achievable live strategy, because the real bot only sees past/current data and
pays fees, spread, slippage, funding, and liquidation risk.

The codebase now keeps one automated strategy: `legacy-valley-peak`. Removed strategy
families and deleted adaptive controls are no longer available as runtime choices or
benchmark controls. Research should therefore improve legacy valley/peak itself, its
execution model, its sizing, its filters, or the data it uses.

## Strategy Components

Every strategy improvement should be evaluated as an improvement to one of three basic
components:

1. Signal quality: decide when price is near a useful low or high and therefore when
   the bot should buy, sell, hold, or ignore the move.
2. Sizing: choose how much exposure to open or close based on current equity, borrowed
   balances, open orders, existing lots, volatility, and remaining opportunity budget.
3. Execution price: choose whether to trade immediately or wait for a better level,
   while accounting for fees, spread, slippage, missed fills, and stale-order risk.

The better these components are, the closer the realized strategy can move toward the
perfect-margin oracle. The signal decides whether there is an opportunity. The sizing
procedure decides how much of that opportunity to take. The execution price decides
whether the expected edge survives trading costs and fill risk.

Optimal size is not simply maximum leverage on every signal. It should preserve enough
capital to keep buying a deeper dip, but not preserve so much that the strategy fails to
extract profit from the current dip. Optimal execution price has the same tension: trade
too frequently and fees eat the edge; wait too long and volatility opportunities are
missed.

## Current Legacy Summary

Legacy valley/peak is a rolling-average turning-point strategy. The current default is
the relaxed per-lot peak exit-grid execution model:

- detects local valleys from clamped rolling-average derivative sign changes
- opens long lots at confirmed valleys with market-style entries
- detects local peaks from clamped rolling-average derivative sign changes
- places resettable targeted sell ladders between each lot's tracked peak and
  break-even sell price
- cancels stale unfilled orders after `staleOrderMs`
- tracks quote/base reserves while orders are open
- records fees, realized PnL, long average entry, drawdown, and win rate when orders fill
- does not open automated shorts

In the simulator those ladder orders use a below-price trigger, which is closer to
stop-ladder behavior than normal exchange sell-limit behavior. The older limit-only mode
still exists as a configuration path, but it is no longer the default.

The detailed implementation flow and current default parameters are documented in
[Algorithms](algorithms.md).

## Validation Standard

A legacy valley/peak change is not promotable unless it improves robust evidence, not
just one recent window. Preferred validation:

- cycle-wide random-length BTCUSDT windows
- folded grid search for parameter changes
- oracle capture versus the perfect-margin benchmark
- average, median, and P10 return
- max drawdown and risk-adjusted return
- trade count, fee burden, and stale-order rate
- explicit note of sample count, date range, leverage guard, cooldown, limit offset, and
  seed

The benchmark harness now produces legacy-only comparisons. Grid search should compare
legacy parameter variants, not separate strategy families.

## Current Observations

Legacy valley/peak remains useful but fragile:

- The detector can identify local turns, but it can buy too early during persistent
  downside trends.
- Resting limit orders improve nominal price but introduce missed-fill risk after fast
  reversals.
- Confirmation windows reduce noise but can delay exits.
- Gaussian sizing keeps largest trades near smooth derivative turns, but sigma and
  spend-rate settings are easy to overfit.
- Fees and stale orders remain large enough that high-churn parameter sets should be
  rejected quickly.
- Current BTCUSDT tests show three useful exit-grid regimes. On the 2026-05-23 to
  2026-06-21 fixed window, aggregate grid returned `-2.16%` with `5.16%` drawdown,
  per-lot strict returned `-10.18%` with `17.94%` drawdown, per-lot filled-grid
  relaxed returned `-10.20%` with `17.94%` drawdown, and default legacy returned
  `-14.99%` with `21.65%` drawdown.
- On the last-year 2025-06-22 to 2026-06-21 BTCUSDT window, aggregate grid was the
  safest variant (`-11.06%` return, `13.41%` drawdown). Default legacy returned
  `-33.77%`, while per-lot strict returned `-38.46%` and per-lot filled-grid relaxed
  returned `-38.96%` with `49.02%` drawdown.
- On 24 random 7-120 day BTCUSDT windows over the local 2021-07-16 to 2026-06-21
  cache, aggregate grid had the best downside profile (`0.69%` average return,
  `2.57%` average drawdown, `-5.18%` worst sample). Per-lot filled-grid relaxed had
  the best average return among grid variants (`1.73%`, `11.05%` average drawdown,
  `-12.44%` worst sample), slightly ahead of per-lot strict (`1.64%`). Default legacy
  averaged `-0.22%` with `13.96%` average drawdown and a `-25.92%` worst sample.
- On six folded BTCUSDT windows from 2021-07-16 to 2026-06-21, aggregate grid ranked
  first by risk-adjusted return (`2.67%` average return, `2.385` average risk return,
  `4.93%` average drawdown, `-10.19%` worst fold). Per-lot filled-grid relaxed ranked
  seventh (`14.06%` average return, `1.529` average risk return, `29.15%` average
  drawdown, `-42.24%` worst fold), ahead of per-lot strict by risk-adjusted return
  but behind default legacy on this folded run.
- Full folded validation became practical after replacing repeated position-ledger
  rebuilds with an incremental long-lot cache inside the strategy. Random-window runtime
  for per-lot rows dropped from roughly `40-43s` per row to `16-18s` per row on the same
  sample set.
- Portfolio mode over the four strategy equity curves favored a rolling-winner strategy
  mix (`316.19%` return, `71.39%` drawdown, `4.429` risk return). Inverse-vol had the
  best drawdown-adjusted shape (`158.02%` return, `42.07%` drawdown, `3.756` risk
  return). This is strategy allocation evidence only; multi-symbol allocation was
  skipped because the cached symbol set did not yield enough aligned usable series for
  that routine.

The most important next improvement is better signal qualification. The strategy needs
to know whether a detected valley/peak has enough expected move after costs and fill
risk before it opens or exits exposure.

## Research Backlog

| Direction | Why it may help legacy | Main risk |
| --- | --- | --- |
| Limit-execution tuning | Jointly tune `limitOffsetBps`, stale timing, and order count to improve fill quality. | Can overfit synthetic OHLC fill paths. |
| Peak exit-grid tuning | Tune grid order count, price distribution, size distribution, sell fraction, reset policy, stale timing, and entry sizing. | Current below-trigger simulator may overstate fills after gaps. |
| Triple-barrier labels | Label candidate entries by take-profit, stop-loss, timeout, and net return after costs. | Easy to overfit barrier and horizon settings. |
| Cost-aware accept/reject model | Learn when a detected valley/peak is worth trading after fees and missed-fill risk. | Data leakage and unstable calibration. |
| Funding and premium features | Perpetual funding can dominate flat price movement and crowded positioning. | Funding extremes can persist for long periods. |
| Open-interest and liquidation features | Helps distinguish real continuation from weak squeeze/liquidation moves. | Exchange data quality and interpretation are regime-dependent. |
| Order-flow imbalance | May improve very short-horizon entry timing around detected valleys/peaks. | Requires realistic spread, latency, depth, and partial-fill simulation. |
| Volatility targeting | Reduce exposure when realized volatility makes long inventory fragile. | Can reduce size before profitable volatility expansion. |
| Correlation-aware portfolio layer | Avoid running many copies of the same BTC-beta exposure across symbols. | Needs synchronized multi-symbol history. |
| Anti-overfit reporting | Track trials, folds, and statistical confidence before promoting parameters. | Adds process overhead, but prevents false winners. |

## Near-Term Implementation Steps

1. Tune peak exit-grid parameters across random-length and folded validation.
2. Compare aggregate low-drawdown grid against per-lot filled-grid relaxed as separate
   risk profiles rather than forcing one winner too early.
3. Add benchmark columns for created, filled, stale-cancelled, and leverage-cancelled
   orders.
4. Tune limit offset and stale timing across folded/random-length validation.
5. Add triple-barrier labels around legacy candidate entries.
6. Train a small accept/reject model using only features available at signal time.
7. Add funding, mark/index premium, open interest, and liquidation history for futures
   markets.
8. Cache enough liquid symbols to evaluate legacy as a portfolio-level allocator.
9. Delay RL until the simulator includes funding, liquidation, latency, spread, and
   partial-fill assumptions.

## References

| Area | Reference | Useful idea | Application here |
| --- | --- | --- | --- |
| Backtest overfitting | [Bailey, Borwein, Lopez de Prado, Zhu - The Probability of Backtest Overfitting](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253) | Strategy search can select false winners. | Track number of trials, walk-forward splits, and later PBO-style reporting. |
| Deflated Sharpe | [Bailey and Lopez de Prado - The Deflated Sharpe Ratio](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551) | Sharpe needs correction for selection bias and non-normal returns. | Add confidence checks before promoting legacy parameters. |
| Triple-barrier labeling | [Mlfin.py triple-barrier and meta-labeling docs](https://mlfinpy.readthedocs.io/en/latest/Labelling.html) | Labels should include profit, stop, and timeout outcomes. | Add cost-aware labels for legacy valley/peak candidates. |
| Time-series momentum | [Moskowitz, Ooi, Pedersen - Time Series Momentum](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2089463) | Past return can predict continuation across futures markets. | Add trend/regime filters before trading a valley or peak. |
| Volatility timing | [Moreira and Muir - Volatility Managed Portfolios](https://www.nber.org/papers/w22208) | Reducing exposure when volatility is high can improve risk-adjusted returns. | Add volatility throttles around legacy sizing. |
| Order-flow imbalance | [Cont, Kukanov, Stoikov - The Price Impact of Order Book Events](https://arxiv.org/abs/1011.6402) | Bid/ask order-flow imbalance can explain short-horizon price changes. | Add order-book features only after execution simulation is realistic. |
| Perpetual futures mechanics | [He, Manela, Ross, von Wachter - Fundamentals of Perpetual Futures](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4301150) | Funding links perpetual futures and spot prices. | Include funding and basis in PnL and features. |
| Binance funding data | [Binance funding-rate FAQ](https://www.binance.com/en/support/faq/detail/360033525031) and [funding-rate history API](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History) | Funding history is available from the exchange. | Cache funding beside candles. |
| Online portfolio selection | [Li and Hoi - Online Portfolio Selection: A Survey](https://arxiv.org/abs/1212.2129) | Sequential allocation can be benchmarked with simple online allocators. | Use portfolio allocation as a later layer around legacy signals. |
| Covariance shrinkage | [Ledoit and Wolf - A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices](https://www.ledoit.net/ole1a.pdf) | Sample covariance is unstable with short history. | Use shrinkage before multi-symbol risk allocation. |
| RL caution | [Deep Reinforcement Learning in Quantitative Algorithmic Trading: A Review](https://arxiv.org/abs/2106.00123) | Many DRL trading studies use unrealistic settings. | Keep RL behind simulator realism and robust validation. |
