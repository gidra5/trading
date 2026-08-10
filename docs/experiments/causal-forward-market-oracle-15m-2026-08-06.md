# Causal forward-market features for the 15-minute oracle (2026-08-06)

## Question

Do causal trade-flow and derivatives-market inputs improve the 15-minute
oracle-distribution model over the same model trained on candle history alone?

## Inputs and causal alignment

The experiment added 231 inputs to the existing 771-feature multiscale OHLCV
history:

- 85 spot aggregate-trade-flow features;
- 66 USD-M futures basis and taker-flow features;
- 80 USD-M futures positioning features, including open interest and
  long/short ratios.

Every example is aligned to a completed 1-minute candle. Spot flow ends at the
same completed minute, futures basis/flow uses the just-closed futures minute,
and Binance's 5-minute positioning metrics retain a full 5-minute availability
lag. The target remains the exact next-15-minute, 101-action differentiable
oracle distribution with a one-minute decision delay and 0.175% transition
friction.

The earlier frozen-v18 residual screens were used only as inexpensive input
audits, not as 15-minute model-selection evidence. Spot flow, futures
basis/flow, and futures positioning each selected effectively zero residual
weight on their untouched validation portions. The existing order-book-depth
audit is incomplete and its source covers only 368 days, so depth was not
included in this first matched experiment.

## Matched training contract

The candle-only control and joint model use exactly the same timestamps,
targets, architecture, seed, optimizer, temperature curriculum, and 32-epoch
budget. Only the inputs differ:

| Split | Examples |
| --- | ---: |
| Train | 265,290 |
| Validation | 258,105 |
| Sealed test | 14,400 |

The smaller intersection relative to the preceding candle-only experiment is
caused by requiring all three added sources and their predecessor day. Feature
normalization is fitted on the training split only.

An initial joint run exposed a normalization defect: features that were
constant in training received a `1e-6` scale but became nonconstant in
validation, creating normalized magnitudes near one million and an enormous
regularization term. That run is invalid and is excluded below. The corrected
dataset assigns a neutral scale of 1.0 to forward features with training
standard deviation below `1e-4`; 26 inputs were neutralized this way. The
largest validation magnitude then fell to 266, comparable to the base input
block.

## Training result

Both corrected runs completed all 32 epochs and selected epoch 3. Lower is
better for every metric in this table.

| Validation metric | Candle control (771) | Joint forward (1,002) |
| --- | ---: | ---: |
| Selection objective: mean KL + p50 KL | **2.689429** | 2.695700 |
| Mean raw KL | **1.388794** | 1.393034 |
| Raw KL p50 | **1.300635** | 1.302666 |
| Raw KL p90 | **2.449761** | 2.463641 |
| Raw KL p95 | **2.721381** | 2.732923 |

The joint model's validation objective is 0.233% higher (worse), so the
candle-only control wins the predeclared validation selection. As a secondary
check performed after selection, its sealed-test-subset objective is also
lower: 2.524939 versus 2.527886. Both schedules stalled at target temperature
0.168468 rather than reaching the production temperature 0.01.

## Backtest

The validation winner and the forward-feature candidate were both run through
the standard model-driven backtest. They used static confidence 75, the bot's
other default parameters, direct inference, and a 16-day in-memory feature LRU
with no persistent prediction cache.

| Metric | Candle control | Joint forward |
| --- | ---: | ---: |
| Covered standard windows | 33 | 33 |
| Oracle decisions | 887,039 | 887,039 |
| Raw nonzero modal decisions | 2,143 (0.242%) | 205,633 (23.182%) |
| Mean predicted confidence | 30.937% | 30.951% |
| Nonzero after transition-cost conditioning | 0 | 0 |
| Signals / trades | 0 / 0 | 0 / 0 |
| Return | 0% in every window | 0% in every window |
| Model inference duration | 55.289 s | 121.513 s |
| Whole suite wall duration | 76.047 s | 142.953 s |

`latest-3m` remains outside the frozen rich-feature corpus and is excluded, as
in the prior 33-window suite.

## Interpretation

The tested trade-flow, basis/taker-flow, and positioning inputs do not improve
the 15-minute oracle-distribution objective under this architecture and data
intersection. They make the raw modal action directional much more often, but
the predicted advantage is never large enough to survive transition-cost
conditioning. Consequently neither model emits a signal or trade.

This is a negative result for these specific aggregated features, not evidence
that all order-flow information is useless. Important untested variants include
causally available order-book depth, event-level USD-M futures flow, mark/index
premium and funding history, and an objective trained directly through the
transition-cost-conditioned trading decision.

Artifacts:

- Feature builder: `ml/forward_market_features.py`
- Dataset builder: `ml/prepare_forward_market_oracle_dataset.py`
- Direct inference server: `ml/serve_forward_market_oracle.py`
- Control plan: `ml/training-plans/causal-forward-market-oracle-15m-control-v1.json`
- Joint plan: `ml/training-plans/causal-forward-market-oracle-15m-joint-normalized-v2.json`
- Control result: `data/training/runs/causal-forward-market-oracle-15m-control-v1/result.json`
- Corrected joint result: `data/training/runs/causal-forward-market-oracle-15m-joint-normalized-v2/result.json`
- Control backtest: `data/benchmarks/causal-forward-market-oracle-15m-control-suite-2026-08-06.json`
- Joint backtest: `data/benchmarks/causal-forward-market-oracle-15m-joint-normalized-v2-suite-2026-08-06.json`
