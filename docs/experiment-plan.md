# Experiment Plan

This plan turns the research backlog into repeatable CLI experiments. The goal is not
to declare a production strategy yet; it is to reject weak ideas quickly, find whether
any signal beats simple market-regime controls, and make later ML/RL work less likely
to overfit.

## Acceptance Rules

A candidate is not interesting unless it beats cheap controls on the same window:

- always flat
- always long
- always short
- deterministic random sign, opt-in because it is intentionally churn-heavy

Promotable candidates should show positive net PnL, positive risk-adjusted return,
acceptable max drawdown, and stable behavior across random-length samples from the
whole available BTC cycle. A parameter candidate that only wins one fold, one recent
30-day window, or the current directional regime is treated as overfit until proven
otherwise.

## Implemented Experiment Modes

Run the normal strategy benchmark:

```bash
npm run benchmark:strategies
```

The default benchmark is now a cycle-wide random-length BTCUSDT test: 48 deterministic
samples, 7-120 day windows, 1,825 day lookback, seed 1337. The script chooses the
deepest local candle cache for the requested symbol, so the newer
`data/historical/spot-btcusdt/btcusdt/1m` cache is used automatically when it has more
history than the older flat `data/historical/btcusdt/1m` cache.

Run an explicit recent-window smoke test only when needed:

```bash
npm run benchmark:strategies -- --mode days --days 30
```

Run folded parameter search:

```bash
npm run benchmark:strategies -- --mode grid-search --days 1825 --grid-folds 6 --grid-limit 12
```

Run strategy-ensemble portfolio allocation:

```bash
npm run benchmark:strategies -- --mode portfolio --days 1825 --portfolio-rebalance-candles 40 --portfolio-lookback-candles 120
```

Run the expensive churn control explicitly:

```bash
npm run benchmark:strategies -- --days 3 --include-random-sign --only random
```

The script also supports multi-symbol portfolio allocation through `--mode portfolio`
and `--symbols BTCUSDT,ETHUSDT,...`. It skips uncached symbols cleanly. Many non-BTC
symbols have shallow local 1m caches, but cycle-scale multi-symbol allocation is still
blocked until enough liquid symbols have comparable 4-5 year history.

## Autonomous Experiment Loop

Run one prompt dry-run to inspect what the loop will ask an agent to do:

```bash
npm run experiment:loop -- --max-iterations 1 --dry-run
```

Run indefinitely:

```bash
npm run experiment:loop
```

Each iteration runs a cycle-wide random-length benchmark, stores the output under
`data/experiments/agent-loop`, prompts `codex exec` to review the evidence and improve
the `master-adaptive` algorithm, then runs `npm run typecheck`. The loop deliberately
forbids promoting changes from a simple 30-day test because the current recent market
trend can make passive short exposure look better than it is across a full BTC cycle.

Useful overrides:

```bash
TRADING_EXPERIMENT_AGENT_COMMAND="codex exec -C . --sandbox workspace-write -" npm run experiment:loop
npm run experiment:loop -- --sleep-sec 300 --agent-timeout-min 240
npm run experiment:loop -- --benchmark-command "npm run benchmark:strategies -- --mode random-lengths --lookback-days 1825 --min-window-days 14 --max-window-days 180 --samples 48 --seed {seed}"
```

## Experiment 1: Market-Regime Controls

Purpose: determine whether active strategies beat simple directional exposure.

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

Conclusion: the market window was strongly favorable to passive short exposure. None of
the active rules beat that control. This means the active rules are not yet adding
detectable timing edge.

## Experiment 2: Folded Parameter Search

Purpose: test whether simple parameter changes rescue trend, breakout, or mean reversion.

Grid: 48 candidates across trend-following, volatility breakout, and mean reversion.
Validation: 3 contiguous folds over the same 30-day BTCUSDT 1m window.

Top 12 candidates:

| Rank | Candidate | Algo | Folds | Profitable | Avg Return | Avg Risk Ret | Avg Sharpe | Avg Max DD | Avg Trades | Worst |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Breakout l288/b12/x0.35 | volatility-breakout | 3 | 0/3 | -11.46% | -0.997 | -30.484 | 11.50% | 191.0 | -14.79% |
| 2 | Breakout l288/b12/x0.2 | volatility-breakout | 3 | 0/3 | -6.64% | -0.997 | -30.617 | 6.66% | 190.7 | -8.58% |
| 3 | Reversion w96/z2.4/t60/x0.15 | mean-reversion | 3 | 0/3 | -23.83% | -0.999 | -147.601 | 23.85% | 858.0 | -27.70% |
| 4 | Reversion w96/z2.4/t60/x0.25 | mean-reversion | 3 | 0/3 | -36.67% | -0.999 | -147.346 | 36.70% | 859.3 | -42.00% |
| 5 | Reversion w96/z1.8/t30/x0.15 | mean-reversion | 3 | 0/3 | -30.28% | -1.000 | -168.539 | 30.29% | 1,175.0 | -34.46% |
| 6 | Reversion w96/z1.8/t60/x0.15 | mean-reversion | 3 | 0/3 | -35.38% | -1.000 | -196.294 | 35.39% | 1,440.3 | -38.35% |
| 7 | Reversion w96/z1.8/t30/x0.25 | mean-reversion | 3 | 0/3 | -45.31% | -1.000 | -168.233 | 45.32% | 1,176.3 | -50.74% |
| 8 | Reversion w96/z1.8/t60/x0.25 | mean-reversion | 3 | 0/3 | -51.83% | -1.000 | -195.793 | 51.84% | 1,443.7 | -55.44% |
| 9 | Reversion w48/z1.8/t30/x0.15 | mean-reversion | 3 | 0/3 | -44.63% | -1.000 | -251.268 | 44.64% | 1,596.3 | -49.22% |
| 10 | Reversion w48/z1.8/t60/x0.15 | mean-reversion | 3 | 0/3 | -48.52% | -1.000 | -290.376 | 48.53% | 1,811.0 | -51.22% |
| 11 | Reversion w48/z1.8/t30/x0.25 | mean-reversion | 3 | 0/3 | -62.73% | -1.000 | -251.049 | 62.73% | 1,597.7 | -67.82% |
| 12 | Reversion w48/z1.8/t60/x0.25 | mean-reversion | 3 | 0/3 | -67.04% | -1.000 | -289.823 | 67.05% | 1,812.7 | -69.90% |

Conclusion: no tested parameter candidate is viable. Lower-turnover breakout is least
bad, but still loses every fold. The next search should add better features or labels,
not merely widen this naive grid.

## Experiment 3: Strategy-Portfolio Allocation

Purpose: test whether capital allocation across strategies improves the account-level
result.

Local BTCUSDT 1m, same 30-day window, full candle-level equity curves, 1x portfolio
gross leverage, 40-candle rebalance, 120-candle lookback:

| Portfolio | Subject | Return | Net PnL | Max DD | Risk Ret | Sharpe | Turnover | Periods |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Equal strategy mix | strategies | -33.86% | -$3,385.80 | 33.97% | -0.997 | -22.417 | 1.00 | 41,918 |
| Inverse-vol strategy mix | strategies | -47.42% | -$4,741.65 | 47.42% | -1.000 | -94.960 | 300.30 | 41,918 |
| Rolling winner strategy mix | strategies | -1.55% | -$155.46 | 32.27% | -0.048 | 0.365 | 589.79 | 41,918 |
| Drawdown-guard strategy mix | strategies | -1.36% | -$135.72 | 15.08% | -0.090 | -0.171 | 478.73 | 41,918 |

Conclusion: strategy allocation reduced losses but did not turn the system profitable
once measured on true candle-level periods. It is still useful as a risk-control layer,
but it cannot compensate for weak underlying strategies.

## Experiment 4: Longer Robustness Checks

Last-year BTCUSDT 1m, 365 cached day files from 2025-06-21 through 2026-06-20:

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

Random-length BTCUSDT 1m, 40 deterministic samples, 1-30 day windows, 365-day
lookback, seed 1337:

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

## Next Implementation Steps

1. Cache at least 10 liquid symbols for 1m candles so multi-symbol portfolio experiments
   can run.
2. Add funding, mark/index premium, and open interest to futures backtests.
3. Build triple-barrier labels for proposed entries and train a small accept/reject model.
4. Add anti-overfit reporting: trial count, walk-forward split metadata, and later
   Deflated Sharpe / PBO style reporting.
5. Replace naive strategy signals with cost-aware labels or order-flow/funding features
   before doing larger parameter searches.
