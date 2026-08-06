# Oracle-distribution path model: one-minute trading backtest

Date: 2026-08-06

## Setup

The standard 34-window hindsight-oracle suite was replayed on canonical
BTCUSDT one-minute candles, replacing each hindsight oracle distribution with
the causal distribution produced by
`oracle-distribution-path-15m-two-layer-glu-mean-p50-v1`.

At each completed one-minute candle, the model receives the 120 log returns
ending at that candle, predicts the next 15 one-minute returns, and evaluates
that path with the same exact differentiable oracle used in training. The
resulting 101-action distribution is passed to the existing trading simulator.
Inference runs through one persistent Python/CUDA worker and distributions are
held only in process memory; no prediction cache is written to disk. The large
fit window is retained temporarily in memory so its four contained folds can
reuse it.

Static confidence is `0.75`. All other execution settings remain at their
defaults, including 0.175% friction, the 10 bps/hour borrow rates, the
confidence leverage floor, and the learned-oracle maximum leverage of 1x.

The latest-three-month window ends on 2026-07-24 because that is the newest day
with a complete contiguous preceding three-month one-minute history. Later
local shards contain gaps.

## Result

| Metric | Result |
| --- | ---: |
| Standard windows | 34 |
| Loaded/replayed candles | 1,018,080 |
| Policy decisions | 1,018,079 |
| Raw model modes outside cash | 165,390 (16.25%) |
| Modes outside cash after transition-cost conditioning | 0 (0%) |
| Signals emitted | 0 |
| Trades | 0 |
| Profitable windows | 0 / 34 |
| Return in every window | 0% |
| Mean model confidence after static scaling | 40.01% |

The provider and cadence are functioning: every expected timestamp except the
terminal replay boundary reached the policy, and the raw model distribution
preferred nonzero exposure on 16.25% of decisions. However, none of those
preferences remained modal after the existing policy conditioned the base
distribution on current exposure and charged the initial rebalance friction.
The strategy therefore correctly remained in cash.

This means the distribution-level KL improvement measured during training is
not large enough to create an executable edge under the current fee and
slippage assumptions. Static confidence 0.75 only scales leverage after a
nonzero conditioned target exists, so changing it cannot turn these cash
decisions into trades.

## Runtime and five-year estimate

The complete command took 38.5 seconds wall time. Internal measurements were:

| Stage | Observations | Time | Throughput |
| --- | ---: | ---: | ---: |
| One-time model-worker startup | - | 1.77 s | - |
| Direct model + oracle inference | 672,480 unique candles | 10.28 s excluding startup | 65,435 candles/s |
| Simulator replay | 1,018,080 candles | 23.65 s | 43,055 candles/s |

The suite infers fewer unique candles than it replays because the four fit
folds reuse their enclosing fit-window distributions in memory.

Five 365-day years contain 2,628,000 one-minute candles. Linear extrapolation
from the measured steady rates gives:

- model plus oracle inference: 40.2 seconds;
- simulator replay: 61.0 seconds;
- model-worker startup: 1.8 seconds; and
- core total: about 103 seconds, or 1 minute 43 seconds.

Allowing for loading 1,825 daily shards, allocation, and garbage collection,
approximately two minutes is a reasonable estimate on this machine. A single
five-year in-memory run would be memory-heavy: probabilities alone occupy
about 1.06 GB in each process during transfer, before candle objects and model
working memory. Chunked stateful replay would reduce that peak if the complete
cycle is run later.

## Reproduce

```powershell
npm run mlp:backtest:oracle-distribution-path-1m -- --quiet-replay
```

Machine-readable results are in
`data/benchmarks/oracle-distribution-path-1m-suite-diagnostics-2026-08-06.json`.
