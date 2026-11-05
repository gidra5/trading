# Historical evaluation of the revised plane-value planner

**The first historical test hour lost 28.7570 USDT (-0.287570%), after 13.6591
USDT in fees.** Both numerical and independent exact-rational execution audits
passed. This is an actual historical replay result, not simulated CE.

This experiment evaluates the revised controller on actual archived BTCUSDT
prices. It is separate from the earlier 64-path simulation and its geometric
mean return. The selected interval is the first complete test hour, fixed
before measuring returns; the NSF checkpoint and trading menu are frozen.

## Completed result

The preselected interval is **2026-08-10 00:00:01–01:00:01 UTC**. It contains
3,600 actual one-second decisions and starts and ends with no position or orders.

| Metric | Result |
|---|---:|
| Initial cash | 10,000.0000 USDT |
| Terminal cash | **9,971.2430 USDT** |
| Net return | **-0.287570%** |
| Maximum marked-equity drawdown | **1.280677%** |
| Market execution fees | 9.7601 USDT |
| Maker execution fees | 3.8991 USDT |
| Total execution-price fees | **13.6591 USDT** |
| Market trades | 5 |
| Limit placements / fills / cancellations | 39 / 5 / 34 |
| Liquidations | 0 |
| Residual inventory / pending orders | 0 / 0 |
| Underlying BTCUSDT price change, without fees | +0.129415% |
| Cash benchmark | 0% |

The origin graph took 541.14 seconds to build, its continuation values took
167.29 seconds, and historical replay took 107.39 seconds. Fresh one-second
forecast generation consumed 1.47 seconds of replay time. Mean decision time,
including continuation mapping, was 29.18 ms. These timings use the local
RTX 3070 and CPU together; they are not a latency benchmark for rebuilding
the whole graph every second.

The position was predominantly short during an hour whose final BTC price was
higher. The measured projection distance is also substantial: fresh predicted
child prices differed from their assigned continuation-node prices by **15.15
bps on average**, with a **64.29 bps maximum**. The graph has known dispersion
loss, and this reused-continuation adapter adds off-graph approximation. This
run therefore measures this implemented controller; it does not isolate the
trained distribution's error or demonstrate the continuous optimum's return.

The earlier +0.185733% simulation result used model-generated paths at a
different, validation origin. It is not a matched historical performance
estimate for this hour. The historical result above does not demonstrate
profitability; one fixed hour also does not establish full-archive performance.

![Historical account return and exposure](E:/Projects/trading/data/backtests/oracle-plane-history-test-hour0/equity.png)

## Forecasting and decision procedure

The origin graph has 3,600 consecutive one-second transitions, 256 market
representatives per layer and 64 conditional draws per parent. It is generated
from only the two observations available at the beginning of the episode.
The graph is built once for this test hour, with its continuation planes
computed backward to the declared flat terminal state.

Every historical second uses a fresh 64-draw one-second NSF prediction from
that second's two completed observable rows. Each predicted child is mapped to
the corresponding-depth graph representative using the original joint metric:
all 118 standardized coordinates of the two 59-feature observations, plus
their two log prices scaled by one basis point. The selected representative
supplies a continuation function. The child's actual sampled price is used in
the Oracle transition; it is not snapped to the representative price.

The controller executes each candidate against the same conditional outcomes,
averages the continuation log values, and then selects an action. Historical
next prices are supplied to execution only after the action has been selected.
Forecast origins can be batched because inputs and predictions remain row-local.
The graph, value computation and checkpoint contain no observations from the
future of the episode origin. Fresh one-step predictions contain no observations
after their respective decision times.

This is a necessary out-of-graph extension of the simulated controller. It
reuses the same action-selection implementation in `ml/oracle_plane_policy.py`.
It does not rebuild a complete one-hour graph every second, and no upper bound
computed on the origin graph is claimed to transfer to the historical path.
Projection errors are saved and reported explicitly.

## Execution settings

- Gamma 1; initial cash 10,000; 1-second decisions.
- Market exposure targets from -2 to +2 with 0.01 increments; zero trade and
  cancellation remain available.
- One limit slot. Gross order fractions 0.1/0.25/0.5/1, capped by actual size
  restrictions; passive offsets 1/10/100 price ticks. No joint nonzero market
  trade and new limit placement in this evaluated menu.
- Received-amount maker/taker fees 0.02%/0.05%, reservation leverage 2,
  effective maintenance bounds [-3,3], quote minimum 5, aggregate quote cap
  10,000 per decision, price tick 0.01.
- Oracle's signed-asset funding map with factor 1 each second, matching the
  simulation. Actual exchange funding payments are not included.
- Arrival fills before cancellation/rebalancing; liquidation before the next
  arrival; pending orders reserve collateral; both maker fills and market
  trades consume the shared quote cap.
- The last eight decisions use valid capped closing, including valid dust
  top-up/close sequences where needed. The required terminal state is zero
  asset and an empty order book. Infeasible replay paths are reported as
  failures, not assigned a fabricated terminal return.

The mark source is the existing immutable **spot BTCUSDT one-second close**
archive. The execution model is the declared Oracle model, rather than a replay
of exchange queues, spread, impact or actual perpetual-funding records.

## Validation and artifacts

The shared policy, observation projection and graph regressions pass 45 Python
tests. These include averaging before maximizing, retention of fresh off-graph
prices for execution, forecast-row independence, chunk invariance, immutable
input handling, order capacity, and valid terminal closing. The independent
TypeScript Oracle execution suite has 20 passing tests.

The completed historical trace passed all 3,600 float64 conformance checks
against the current Oracle engine. A separate exact-rational TypeScript replay
accepted every actual action and terminal close, ending at 9,971.242960469956
USDT versus 9,971.242960469952 in Python. The largest numerical state/trade
difference was 1.82e-11. Fees and trade/order counts matched. A separate nine-step
fixture also exercised a price-induced dust position, valid top-up and complete
close in both implementations. These checks verify execution, not optimality
of the learned continuation or validity of its graph approximation.

Source snapshots and hashes are saved with each new artifact:

- `data/backtests/oracle-accuracy-graph-test-hour0/`: origin-only graph and saved observable representatives.
- `data/backtests/oracle-accuracy-value-test-hour0/`: continuation planes and graph-specific account upper.
- `data/backtests/oracle-plane-history-test-hour0/`: historical marks, actual decisions, execution trace, fees, projection errors and report.

The replay is run with:

```powershell
.venv-ml/Scripts/python.exe ml/backtest_oracle_plane_history.py --graph data/backtests/oracle-accuracy-graph-test-hour0 --value data/backtests/oracle-accuracy-value-test-hour0 --output data/backtests/oracle-plane-history-test-hour0
.venv-ml/Scripts/python.exe ml/audit_certified_graph_execution.py --run data/backtests/oracle-plane-history-test-hour0
node --conditions=development --import tsx scripts/audit-gpu-replay.ts data/backtests/oracle-plane-history-test-hour0
```
