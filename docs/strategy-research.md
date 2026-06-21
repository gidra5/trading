# Strategy Research Notes

The target is to approach the "perfect margin trader": full long on upward moves, full
short on downward moves, scaled by max leverage. In practice this is an upper-bound
benchmark, not an achievable strategy, because real systems see only past and current
market data and pay spread, fees, slippage, funding, and liquidation risk.

## Candidate Families

| Candidate | Why it is worth testing | Main failure mode |
| --- | --- | --- |
| Multi-timeframe trend following | Closest simple heuristic to long-up and short-down behavior. | Choppy markets, late entries, high turnover. |
| Volatility breakout | Attempts to capture large impulse moves after compression. | Fake breakouts and wick-driven entries. |
| Mean reversion with regime filter | Can complement trend systems in sideways markets. | Gets crushed when a real trend starts. |
| Funding/premium filter | Helps avoid crowded perp positioning or size against extremes. | Funding can stay extreme for long periods. |
| Open-interest confirmation | Distinguishes stronger trend continuation from weak short-covering/long-liquidation moves. | OI interpretation is regime-dependent. |
| Order-flow imbalance | Can predict very short-horizon direction from book/trade pressure. | Requires realistic spread, latency, queue, and fill modeling. |
| Regime-switching ensemble | Allocates between trend, reversion, breakout, and flat states. | Easy to overfit regime boundaries. |
| Cost-aware classifier | Predicts future return net of fees/slippage and trades only above threshold. | Data leakage and calibration drift. |
| Triple-barrier meta-labeling | Labels whether proposed trades hit TP before SL/timeout. | Needs careful walk-forward validation. |
| Bandit/RL sizing overlay | Learns position sizing from edge, volatility, and drawdown state. | Exploits simulator weaknesses if the simulator is unrealistic. |

## Implemented First

The first implementation slice adds real long/short paper mechanics:

- market-style paper fills for automated strategies
- signed base inventory, where negative base is a short
- average short entry tracking
- realized PnL for short closes
- absolute exposure metrics
- fast leverage precheck before full ledger validation
- deterministic OHLC tick replay through the same strategy path used by dashboard historical backtests

Three bidirectional research baselines and one master strategy were added:

- `trend-following`
- `volatility-breakout`
- `mean-reversion`
- `master-adaptive`

The first three are intentionally simple and parameterized from the dashboard. The
master strategy combines their signed exposure signals and is the current target for
autonomous improvement.

## Benchmark Observations

Default benchmark command:

```bash
npm run benchmark:strategies
```

This now runs deterministic random-length samples across the available BTC cycle by
default. Recent fixed-window tests are still useful as smoke checks, but they are not
promotion evidence because a single market regime can make passive long or passive short
look artificially strong.

`Risk Ret` is `returnPct / maxDrawdownPct`; `Sharpe` is annualized from the sampled
equity curve with zero risk-free rate.

Detailed experiment commands, acceptance rules, grid-search results, and portfolio
results are recorded in [Experiment Plan](experiment-plan.md).

Local BTCUSDT 1m, 30 cached day files from 2026-05-22 through 2026-06-20, 3x max
leverage, target cap $25,500, initial short debt cap $19,600, 300s cooldown:

| Strategy | Return | Net PnL | Max DD | Risk Ret | Sharpe | Trades | Win Rate | Oracle Capture |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline always flat | 0.00% | $0.00 | 0.00% | - | - | 0 | 0.0% | 0.000% |
| Baseline always long | -47.11% | -$4,711.03 | 60.74% | -0.776 | -3.642 | 1 | 0.0% | -26.249% |
| Baseline always short | 35.52% | $3,552.42 | 13.55% | 2.622 | 6.632 | 1 | 0.0% | 19.793% |
| Moving average | -39.99% | -$3,998.58 | 39.99% | -1.000 | -40.971 | 4,519 | 16.1% | -22.279% |
| Legacy valley/peak | -15.70% | -$1,569.67 | 23.50% | -0.668 | -4.804 | 258 | 7.2% | -8.746% |
| Trend following L/S | -61.84% | -$6,184.28 | 61.84% | -1.000 | -73.217 | 2,056 | 7.0% | -34.458% |
| Vol breakout L/S | -17.22% | -$1,721.90 | 17.22% | -1.000 | -23.038 | 332 | 22.2% | -9.594% |
| Mean reversion L/S | -72.65% | -$7,265.09 | 72.65% | -1.000 | -129.283 | 2,373 | 2.7% | -40.480% |

The active strategies do not beat the passive short control on this window. The result is
useful because it proves the mechanics, the benchmark harness, and the cost assumptions
are harsh enough to reject weak signals.

Last-year benchmark:

```bash
npm run benchmark:strategies -- --mode year
```

Local BTCUSDT 1m, 365 cached day files from 2025-06-21 through 2026-06-20, 3x
max leverage, target cap $25,500, initial short debt cap $19,600, 300s cooldown:

| Strategy | Return | Net PnL | Max DD | Risk Ret | Sharpe | Trades | Win Rate | Oracle Capture |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline always flat | 0.00% | $0.00 | 0.00% | - | - | 0 | 0.0% | 0.000% |
| Baseline always long | -98.91% | -$9,890.78 | 103.79% | -0.953 | 1.126 | 1 | 0.0% | -4.080% |
| Baseline always short | 75.34% | $7,533.73 | 45.52% | 1.655 | 1.125 | 1 | 0.0% | 3.107% |
| Moving average | -99.74% | -$9,974.50 | 99.74% | -1.000 | -13.761 | 17,489 | 15.7% | -4.114% |
| Legacy valley/peak | -35.98% | -$3,597.54 | 48.08% | -0.748 | -0.952 | 4,918 | 41.0% | -1.484% |
| Trend following L/S | -99.97% | -$9,996.69 | 99.97% | -1.000 | -27.298 | 16,557 | 17.9% | -4.123% |
| Vol breakout L/S | -85.62% | -$8,561.53 | 85.62% | -1.000 | -16.541 | 3,197 | 25.0% | -3.531% |
| Mean reversion L/S | -100.00% | -$9,999.97 | 100.00% | -1.000 | -62.639 | 21,695 | 7.1% | -4.125% |

Random-length benchmark:

```bash
npm run benchmark:strategies -- --mode random-lengths
```

Local BTCUSDT 1m, 40 deterministic samples, 1-30 day windows, 365 day lookback,
seed 1337, 3x max leverage, 300s cooldown:

| Strategy | Samples | Profitable | Avg Return | Avg Net PnL | Avg PnL/day | Avg Max DD | Avg Risk Ret | Avg Sharpe | Avg Trades | Avg Capture | Best | Worst |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline always flat | 40 | 0/40 | 0.00% | $0.00 | $0.00 | 0.00% | - | - | 0.0 | 0.000% | 0.00% | 0.00% |
| Baseline always long | 40 | 17/40 | -5.87% | -$587.05 | -$18.07 | 21.99% | 0.372 | 0.517 | 1.0 | 150.410% | 27.98% | -67.63% |
| Baseline always short | 40 | 19/40 | 3.83% | $382.63 | $5.92 | 13.25% | 0.694 | 0.594 | 1.0 | -137.020% | 51.30% | -22.20% |
| Moving average | 40 | 0/40 | -12.99% | -$1,299.07 | -$97.36 | 13.24% | -0.965 | -31.721 | 1,830.4 | -129.510% | -1.57% | -35.09% |
| Legacy valley/peak | 40 | 19/40 | -2.55% | -$255.24 | -$10.59 | 7.33% | 0.395 | 0.287 | 175.7 | 4.945% | 6.49% | -22.95% |
| Trend following L/S | 40 | 0/40 | -24.49% | -$2,448.66 | -$187.36 | 24.56% | -0.991 | -65.727 | 700.5 | -210.048% | -1.87% | -78.15% |
| Vol breakout L/S | 40 | 2/40 | -5.74% | -$574.50 | -$42.86 | 6.12% | -0.836 | -18.539 | 103.5 | -48.714% | 0.34% | -28.78% |
| Mean reversion L/S | 40 | 0/40 | -41.81% | -$4,180.99 | -$354.63 | 41.82% | -1.000 | -148.066 | 1,086.4 | -683.772% | -8.97% | -71.56% |

The random-length and year-scale checks strengthen the earlier conclusion: the first
bidirectional mechanics are useful infrastructure, but these naive signal rules are not
viable strategy candidates without better labels, filters, and search. The first folded
grid search also failed: 48 trend/breakout/reversion parameter candidates had zero
profitable folds. Strategy-portfolio allocation reduced losses but did not become
profitable on true candle-level periods.

### 2026-06-21 Agent Iteration 1

Validation frame: `random-lengths`, BTCUSDT 1m, 48 samples, 7-120 day windows,
1,825 day lookback, seed 7331, 3x max leverage.

The controls show that this sample set contains mixed regimes but still has strong BTC
long drift: always-long averaged `21.66%` return with 28/48 profitable samples, while
always-short averaged `-17.34%`. The active L/S strategies all lost money after
fees/slippage and rebalancing: trend-following `-65.83%`, volatility breakout
`-29.36%`, mean reversion `-74.99%`, and current master-adaptive `-32.12%`. Legacy
valley/peak remained the only active strategy with positive average return at `4.22%`,
but its median return was `0.00%`, so the edge is not broadly distributed.

The new random-window table reports median and P10 return because the average alone hid
tail fragility. For the current master, median return was `-25.89%` and P10 was
`-67.11%`, confirming that the failure is not one bad window.

Defensive master variants were tested on the same 48 random windows. Lower exposure
improved average return from `-32.12%` to `-9.05%`; a breakout-only defensive variant
improved it to `-6.35%` with 3/48 profitable samples and much lower average trades.
Neither beat always-flat, so neither should become the default. Full-cycle six-fold
checks also rejected them for promotion: breakout-only defensive averaged `-35.35%`
with 0/6 profitable folds, and defensive low-exposure averaged `-47.08%` with 0/6
profitable folds.

Implementation outcome: keep the master default unchanged, add the defensive variants
only as explicit grid-search hypotheses, add oracle capture to grid output, and make
`--only` filter grid candidates so future agents can run focused full-cycle checks
without scanning the entire naive grid.

### 2026-06-21 Agent Iteration 2

Validation frame: `random-lengths`, BTCUSDT 1m, 48 samples, 7-120 day windows,
1,825 day lookback, seed 7332, 3x max leverage. The benchmark was rerun after adding
median and P10 return columns to the random-length table.

This seed is more directionally balanced than iteration 1: always-long averaged
`-0.74%` return with median `-4.60%`, while always-short averaged `-0.12%` with median
`2.85%`. Both passive controls had large lower tails (`-66.47%` P10 for long and
`-45.94%` P10 for short), so neither passive direction is robust despite being far less
bad than the active timing rules.

The active algorithms again show the cost and churn failure mode. Trend-following made
about `4,652` trades per sample and averaged `-78.31%`; mean reversion made about
`5,102` trades and averaged `-83.09%`; moving average made about `9,431` trades and
averaged `-62.73%`. Volatility breakout traded less, about `809` trades per sample, but
still averaged `-39.41%` with zero profitable samples. Master-adaptive averaged
`-43.14%`, median `-40.67%`, P10 `-68.81%`, and zero profitable samples, so the ensemble
still mostly combines losing child signals rather than filtering them.

Implementation outcome: do not change the master default. The justified change is a
validation-harness improvement: random-length output now includes median and P10 return
so future candidates must clear both average and lower-tail robustness checks before
promotion.

## Research And Theory References

The practical theme across the literature is that signal discovery, sizing, validation,
and portfolio allocation should be separated. The "perfect margin trader" remains an
oracle upper bound; real candidates should be judged by net return, drawdown, Sharpe,
risk-adjusted return, turnover, liquidation risk, and oracle capture after costs.

| Area | Reference | Useful idea | Application here |
| --- | --- | --- | --- |
| Time-series momentum | [Moskowitz, Ooi, Pedersen - Time Series Momentum](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2089463) | Own past return can predict continuation across futures markets over medium horizons. | Build multi-timeframe trend features and benchmark them against always-long/short/random baselines. |
| Crypto momentum | [Cryptocurrency Volume-Weighted Time Series Momentum](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4825389) | Crypto momentum may depend on market-wide and volume-weighted effects, not only single-symbol price history. | Add cross-sectional crypto baskets and volume-weighted market regime features before optimizing individual pairs. |
| Volatility timing | [Moreira and Muir - Volatility Managed Portfolios](https://www.nber.org/papers/w22208) | Reducing exposure when volatility is high can improve risk-adjusted returns for factors. | Add portfolio-level volatility targeting and per-strategy leverage throttles. |
| Growth-optimal sizing | [Kelly - A New Interpretation of Information Rate](https://www.princeton.edu/~wbialek/rome/refs/kelly_56.pdf) | Bet size should be tied to edge and uncertainty; full Kelly is fragile under estimation error. | Use fractional Kelly-style sizing only after a calibrated probability/edge model exists. |
| Mean-variance portfolio theory | [Markowitz - Portfolio Selection](https://www.jstor.org/stable/2975974) | Portfolio risk depends on covariance, not just individual asset risk. | Implement multi-symbol portfolio backtests with covariance, correlation clusters, and exposure caps. |
| Covariance shrinkage | [Ledoit and Wolf - A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices](https://www.ledoit.net/ole1a.pdf) | Sample covariance is unstable when assets are many or history is short; shrinkage improves conditioning. | Use shrinkage covariance before min-variance, risk-parity, or cluster-cap allocation. |
| Risk parity | [AQR - Understanding Risk Parity](https://www.aqr.com/-/media/AQR/Documents/Insights/White-Papers/Understanding-Risk-Parity.pdf) | Allocate by risk contribution instead of capital amount. | Add inverse-vol and equal-risk-contribution portfolio baselines. |
| Online portfolio selection | [Li and Hoi - Online Portfolio Selection: A Survey](https://arxiv.org/abs/1212.2129) | Portfolio selection can be framed as sequential allocation with follow-the-winner, follow-the-loser, pattern matching, and meta-learning families. | Benchmark simple online allocators before complex RL allocators. |
| Universal portfolios | [Cover - Universal Portfolios](https://isl.stanford.edu/~cover/papers/paper93.pdf) | Constant-rebalanced portfolios can be used as a strong theoretical benchmark for adaptive allocation. | Add constant-rebalanced and universal-style portfolio baselines across liquid symbols. |
| Backtest overfitting | [Bailey, Borwein, Lopez de Prado, Zhu - The Probability of Backtest Overfitting](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253) | Strategy search can select false winners; combinatorially symmetric cross-validation estimates overfit probability. | Track number of trials, add walk-forward splits, and later add CSCV/PBO reporting for grid searches. |
| Deflated Sharpe | [Bailey and Lopez de Prado - The Deflated Sharpe Ratio](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551) | Sharpe needs correction for selection bias, non-normal returns, and multiple testing. | Add DSR or similar significance checks before promoting a strategy to paper/live trading. |
| Triple-barrier labeling | [Mlfin.py triple-barrier and meta-labeling docs](https://mlfinpy.readthedocs.io/en/latest/Labelling.html) | Labels should consider take-profit, stop-loss, timeout, and volatility instead of fixed-horizon direction only. | Build trade outcome labels for proposed long/short entries and train accept/reject meta-models. |
| Order-flow imbalance | [Cont, Kukanov, Stoikov - The Price Impact of Order Book Events](https://arxiv.org/abs/1011.6402) | Short-horizon price changes can be driven by bid/ask order-flow imbalance and market depth. | Add order-book features only after the simulator models spread, latency, slippage, and partial fills. |
| Limit-order market making | [Avellaneda and Stoikov - High-frequency Trading in a Limit Order Book](https://people.orie.cornell.edu/sfs33/LimitOrderBook.pdf) | Market making is an inventory-risk and quote-placement problem, not just a direction problem. | Useful if the bot later places passive maker quotes instead of market-style strategy fills. |
| Order-book simulation | [Cont, Stoikov, Talreja - A Stochastic Model for Order Book Dynamics](https://www.columbia.edu/~ww2040/orderbook.pdf) | Queue dynamics matter for fill probability and short-horizon execution research. | Use as guidance for a more realistic execution simulator before trusting high-frequency strategies. |
| Perpetual futures mechanics | [He, Manela, Ross, von Wachter - Fundamentals of Perpetual Futures](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4301150) | Perps differ from dated futures because funding links contract and spot prices. | Add funding, mark/index premium, and basis features to futures strategies and backtests. |
| Binance funding data | [Binance funding-rate FAQ](https://www.binance.com/en/support/faq/detail/360033525031) and [funding-rate history API](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History) | Funding depends on interest and premium components and is available historically through the exchange API. | Cache funding history beside candles so net PnL includes funding payments. |
| Regime detection | [Regime-Switching Factor Investing with Hidden Markov Models](https://www.mdpi.com/1911-8074/13/12/311) | Hidden regimes can select or suppress strategy families. | Test HMM-style filters for trend, reversion, high-volatility, and flat regimes. |
| RL trading frameworks | [FinRL](https://arxiv.org/abs/2111.09395) | DRL needs a full trading environment with friction, liquidity, constraints, and reproducible pipeline design. | Use FinRL as architecture reference, not as proof that RL should be first. |
| RL caution | [Deep Reinforcement Learning in Quantitative Algorithmic Trading: A Review](https://arxiv.org/abs/2106.00123) | Many DRL trading studies are proof-of-concept, use unrealistic settings, and lack real-time validation. | Keep RL behind simulator realism, robust validation, and simpler allocator baselines. |

Near-term research priorities from these references:

1. Add baselines and anti-overfit reporting before doing large parameter searches.
2. Add volatility targeting, inverse-vol allocation, and correlation-cluster caps before ML.
3. Add funding/basis/open-interest data for perpetual futures.
4. Add triple-barrier labels for long/short entries and train a small accept/reject model.
5. Add order-flow imbalance only when order-book capture and execution simulation are realistic.
6. Treat RL as a later sizing or allocation layer, not the first source of entry signals.

## Next Experiments

Priority order:

1. Cache enough liquid symbols to run true multi-symbol portfolio experiments.
2. Add cost-aware labels: future return must exceed fee, slippage, and spread.
3. Add funding, mark/index premium, and open-interest data for futures markets.
4. Add triple-barrier labels around proposed trades and train a simple classifier to accept/reject them.
5. Add anti-overfit reporting for parameter searches: trial count, fold metadata, and later Deflated Sharpe / PBO style statistics.
6. Improve the ensemble allocator so it can choose flat or passive controls when no active strategy has a recent edge.
7. Only attempt RL after the simulator includes funding, liquidation, latency, spread, and partial fill assumptions.

## Portfolio Management Experiments

Single-symbol alpha is not enough for a durable margin system. The portfolio layer should decide where capital, leverage, and risk budget go across symbols, strategy families, and market regimes.

| Experiment | What to test | Why it may help | Main risk |
| --- | --- | --- | --- |
| Inverse-volatility allocation | Allocate lower notional to high-volatility symbols and higher notional to stable symbols so each market contributes similar realized risk. | Reduces the chance that one noisy symbol dominates account drawdown. | Can overweight quiet markets just before volatility expands. |
| Correlation-cluster caps | Group symbols by rolling correlation and cap gross exposure per cluster. Prefer orthogonal assets or opposite-pair exposure. | Avoids accidentally running many copies of the same BTC-beta trade. | Correlations can jump toward 1.0 during market stress. |
| Volatility-targeted leverage | Scale total portfolio leverage up/down to target a realized volatility band and reduce leverage after volatility shocks. | Makes leverage conditional on market risk instead of fixed. | Slow realized-volatility estimates may react too late. |
| Drawdown-sensitive capital allocator | Reduce capital for symbols or strategies with recent drawdown, then restore gradually after recovery. | Stops failing strategies from consuming margin indefinitely. | Can cut exposure at the worst time if drawdown is temporary noise. |
| Rolling risk-adjusted winner allocation | Allocate more capital to strategies with better rolling Sharpe, risk-adjusted return, and drawdown profile. | Turns benchmark metrics into live capital-routing signals. | Chases recent winners and may overfit regime-specific behavior. |
| Market-neutral pair baskets | Long stronger symbols and short weaker correlated symbols, optionally hedging BTC/ETH beta. | Can profit from relative strength while reducing broad market direction risk. | Spread trades can diverge for long periods and still liquidate leveraged accounts. |
| Robust minimum-variance basket | Estimate covariance with shrinkage and choose weights that minimize portfolio variance under leverage, turnover, and per-symbol caps. | Provides a defensive default when directional edge is weak. | Low variance does not guarantee positive return after fees. |
| Turnover-aware rebalancing bands | Rebalance only when target weights drift enough to overcome fees, spread, and slippage. | Prevents the portfolio layer from paying away edge through constant resizing. | Wider bands can leave unintended concentration during fast moves. |
| Funding-aware exposure overlay | Penalize positions with expensive funding and favor trades where funding supports the position direction. | Important for perpetual futures where carry can dominate flat price movement. | Funding can flip quickly during crowded moves. |
| Stress-mode deleveraging | Detect correlation spikes, liquidity drops, and volatility breaks; reduce gross exposure or force flat. | Protects the account in liquidation-prone environments. | False positives may miss profitable high-volatility trends. |

Portfolio benchmarks should report account-level return, risk-adjusted return, Sharpe, max drawdown, turnover, leverage utilization, liquidation distance, per-symbol PnL contribution, per-strategy PnL contribution, and exposure by correlation cluster. These experiments need synchronized multi-symbol historical data and a portfolio backtest engine with shared margin, funding, slippage, and liquidation assumptions.
