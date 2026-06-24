# Experiment Plan

The experiment plan centers on one automated strategy: `legacy-valley-peak`. The goal is
to improve that strategy toward the perfect-margin-trader oracle while rejecting overfit
parameter changes quickly.

## Acceptance Rules

A legacy valley/peak change is not interesting unless it improves robust evidence on the
same data frame:

- positive or improved net PnL
- better perfect-margin capture
- lower max drawdown or better risk-adjusted return
- stable average, median, and P10 return across random-length samples
- acceptable trade count, fee burden, and unfilled-order behavior
- no obvious dependence on one recent trend window

A parameter candidate that only wins one fold, one recent 30-day window, or one seed is
treated as overfit until proven otherwise.

## Implemented Experiment Modes

Run the normal legacy benchmark:

```bash
npm run benchmark:strategies
```

Run an explicit recent-window smoke test:

```bash
npm run benchmark:strategies -- --mode days --days 30
```

Run folded legacy parameter search:

```bash
npm run benchmark:strategies -- --mode grid-search --days 1825 --grid-folds 6 --grid-limit 12
```

Run random-length validation:

```bash
npm run benchmark:strategies -- --mode random-lengths --lookback-days 1825 --min-window-days 14 --max-window-days 180 --samples 48 --seed 1337
```

Run multi-symbol portfolio experiments when enough symbols are cached:

```bash
npm run benchmark:strategies -- --mode portfolio --days 1825 --portfolio-rebalance-candles 40 --portfolio-lookback-candles 120
```

The script chooses the deepest local candle cache for the requested symbol.

## Autonomous Experiment Loop

Dry-run one prompt:

```bash
npm run experiment:loop -- --max-iterations 1 --dry-run
```

Run the loop:

```bash
npm run experiment:loop
```

Each iteration runs a cycle-wide random-length benchmark, stores output under
`data/experiments/agent-loop`, prompts `codex exec` to review the evidence and improve
legacy valley/peak, then runs `npm run typecheck`.

Useful overrides:

```bash
TRADING_EXPERIMENT_AGENT_COMMAND="codex exec -C . --sandbox workspace-write -" npm run experiment:loop
npm run experiment:loop -- --sleep-sec 300 --agent-timeout-min 240
npm run experiment:loop -- --benchmark-command "npm run benchmark:strategies -- --mode random-lengths --lookback-days 1825 --min-window-days 14 --max-window-days 180 --samples 48 --seed {seed}"
```

## Current Legacy Experiment Axes

| Axis | Parameters | Purpose |
| --- | --- | --- |
| Clip size | `legacyValleyPeak.maxTradeQuote` | Balance opportunity capture against drawdown and idle cash. |
| Warmup | `legacyValleyPeak.saturationSec` | Avoid early trades before rolling averages are meaningful. |
| Gaussian sizing | `buySigma`, `sellSigma`, buy/sell rates | Control how aggressively derivative shape maps to size. |
| Signal source | `buyDataIndex`, `sellDataIndex` | Shift the primary valley/peak detector between timeframes. |
| Confirmation | `buyConfirmationOffsets`, `sellConfirmationOffsets` | Decide how much broader-window confirmation is required. |
| Limit execution | `limitOffsetBps`, `staleOrderMs` | Balance fill probability against entry/exit price improvement. |
| Internal borrow chains | `longBorrowDepth`, `shortBorrowDepth` | Compare disabled, one-hop, and multi-hop reuse of opposing positions. |
| Exposure cap | `maxPositionQuote`, `minOrderQuote` | Avoid oversized long inventory and tiny fee-inefficient trades. |

## Experiment Log Template

Use this shape for new entries:

````md
## YYYY-MM-DD: Experiment Name

Command:

```bash
npm run benchmark:strategies -- ...
```

Frame:
- Symbol/interval:
- Date range:
- Samples/folds:
- Seed:
- Leverage/cooldown/fees/limit offset:

Result:
| Candidate | Samples/Folds | Profitable | Avg Return | Median | P10 | Max DD | Risk Ret | Sharpe | Trades | Capture |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

Conclusion:
- Promote / reject / keep testing:
- Reason:
- Follow-up:
````

## Next Experiments

1. Tune limit-offset and stale-order timing jointly; the legacy strategy is sensitive to
   whether orders fill after a detected turn.
2. Add better reporting for created, filled, cancelled, and stale orders in benchmark
   summaries.
3. Add triple-barrier labels for legacy valley/peak candidate entries.
4. Train a small cost-aware accept/reject model using only signal-time features.
5. Add funding, mark/index premium, open interest, and liquidation features.
6. Add walk-forward fold metadata and trial counts to benchmark output.
7. Add volatility-targeted sizing and portfolio-level exposure caps.
