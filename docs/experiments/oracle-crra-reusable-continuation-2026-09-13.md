# Reusable continuation for one-second Oracle planning

A learned, horizon-conditioned continuation value makes the full archived
one-second replay practical: **144,533 decisions in 238.53 seconds** on the
RTX 3070. The selected policy stayed entirely cash, ending at **$10,000 with
0% return**. This is an approximate gamma-one policy; neither the earlier
five-stage 95.55% continuation certificate nor a new uniform optimality
certificate has been established for this per-second model.

## Policy construction

At every observed second, the planner draws 64 conditional one-second
outcomes from the frozen sampled-skip latent NSF. For each feasible candidate
action it executes the Oracle transition numerically on those same outcomes,
evaluates the reusable value with one fewer remaining second, averages, and
then chooses the largest action value:

    Qhat_h(s,a) = (1/64) sum_j Vhat_(h-1)(Oracle(s,a,X_j))
    ahat_h(s)   = argmax_a Qhat_h(s,a).

The horizon is min(3600, seconds until the declared backtest end). The
network represents future sequential decisions; the runtime does not build
and solve a 3,600-layer scenario tree. The model is trained on one-second
Bellman backups, including recursively generated future observations.

The value network receives two completed rows of all 59 production features,
their prices and clock features, plus the account, pending order and remaining
horizon. Account inputs retain absolute wealth and order-size ratios,
exposure, limit price, notional, reserved equity and activation age. Market
coordinates pass through a learned 32-dimensional representation. Neither
sample IDs nor hidden sampled latent states are available to the policy.
These are the same two-observation conditional kernels used by the existing
NSF planner; this does not establish sufficiency for arbitrary longer market
histories.

Execution uses the existing TensorOracle and these declared restrictions:

- Exposure targets [-2,2] spaced by 0.01, plus zero trade and cancellation.
- One pending limit slot; new limits one price tick away with notional 25%
  of current equity, subject to the existing size and reservation rules.
- No simultaneous nonzero market trade and new limit placement in this menu.
- Maker fee 0.02%, taker fee 0.05%, reserve leverage 2, maintenance range [-3,3],
  minimum quote flow $5, aggregate maximum $10,000 per rebalance, price tick $0.01.
- Original signed-asset funding transition; this experiment uses factor 1
  at each second, as did the preceding fixed-parameter comparison.
- Arrival fills, liquidation, fee-aware rebalancing and the aggregate cap
  are checked at every second. A final close must actually leave zero asset
  and no orders; dust or capped residuals are infeasible terminal outcomes.

At horizon zero, the network is bypassed and actual terminal execution is
evaluated. Ruin has negative-infinite log utility. At positive horizons, the
network's smooth initialization uses the unrestricted fee-close value, but
this is an approximation, not a claim that such a close is executable.
A valid immediate close followed by cash supplies a lower policy where
available. Approximate positive-horizon values can still misjudge whether an
account can reach a valid flat terminal state.

## Training and selection

The frozen NSF checkpoint is
`data/backtests/oracle-latent-nsf-sampled-skips/latentSampledSkipFlow.pt`, SHA256
`80fdc3df78a5f60d9980b236f9bbaa296904dc562bd64195394831120dde01db`.

The continuation network was trained for 3,000 updates with batch size 64,
eight outcomes per backup, AdamW and an EMA target network. Half the market
batch restarts at actual training origins; half follows recursively sampled
one-second observations. Account wealth is sampled from $2,500 to $20,000,
with cash, inventory and pending-order cases. Half the sampled horizons are
1–64 seconds and half 1–3,600 seconds. The test archive is not used for
training or checkpoint selection.

Selection minimizes mean absolute one-step Bellman residual over 96 fixed
validation accounts with 32 conditional outcomes. The best snapshot was at
step 1,000; the complete 3,000-update run took 162.97 seconds.

| Validation quantity | Result |
| --- | ---: |
| Initial mean absolute log residual | 0.0000135961 |
| Selected mean absolute log residual | 0.0000132245 |
| Selected maximum absolute log residual | 0.0000366466 |
| Finite validation cases | 94 / 96 |
| Nonfinite training targets excluded over the full run | 175 |

The improvement in this validation metric is only about 2.7%. Excluded
nonfinite targets are counted, not converted to finite losses. A small
sampled residual alone does not show that long-horizon opportunity value
has been learned accurately.

Selected value checkpoint: `data/backtests/oracle-continuation-v2/value.pt`,
SHA256 `f4c711120cc9aa9733c6acd3b87f187088c0cde344666dd08b2a0ab0c8766ad9`.
The training plan's source hashes match saved files in its `source/` directory.
The replay plan separately records the runtime sources, including the physical
one-second input adapter; archived training forecast origins have a four-second
stride and must not be mistaken for adjacent replay seconds.

## Backtests and measured computation

| Complete replay | Decisions | Replay time | NSF forecasting | Return | Trades / orders |
| --- | ---: | ---: | ---: | ---: | ---: |
| One-hour fixture, flat end | 3,600 | 6.273 s | 0.640 s | 0% | 0 / 0 |
| Full 40h08m53s archive, rolling one-hour plan | 144,533 | 238.529 s | 16.594 s | 0% | 0 / 0 |

Both replays start and finish at $10,000, with no drawdown, fees, liquidation,
or terminal residual. Independent exact-rational Oracle replay accepts every
actual action and reports zero numerical state/trade discrepancy. Since the
realized trace is all cash, this audit does not exercise traded account paths;
targeted execution tests cover those cases.

Forecast origins are batched in groups of 64. Each row uses only its own
completed history. When the actual account is cash with an empty book,
counterfactual cash decisions at the same wealth can also be batched. A
cached decision is used only if the actual account exactly matches its cash
account. All candidates are still evaluated; this does not assume waiting
is optimal or read a later forecast to choose an earlier action. Tests check
that changing another forecast row cannot change the earlier backup.

All decisions in these runs used this cash batching. Mean amortized solve
time was 1.513 ms/decision; p95 was 1.634 ms. These are batch throughput
numbers, not single-request live latency. The separate 120-second unbatched
profile averaged 16.951 ms per solve. Timings exclude initial checkpoint/data
loading and include the replay loop and trace writing. Performance on an
actively trading trajectory may differ.

## Error bounds for this per-second model

There are two separate errors: replacing 48 hours by one hour, and replacing
the one-hour Bellman solution by a learned continuation. The second does not
resolve the first.

The new Lean module `TradingTheory/OracleContinuationValue.lean` applies the
existing Oracle shadow argument at **every** tick. For a flat-ending run of
n transitions with aggregate quote flow at most M per rebalance, including
the terminal rebalance, it proves

    W_terminal <= max(initial_quote,0) + (n+1) M.

The bound includes fees, funding, liquidation, limit fills and arbitrary
feasible continuous actions, with positive grid prices. Log monotonicity
gives the corresponding log-value bound. This pathwise envelope applies to
the per-second law without dropping seconds or bounding prices by observed
sample extrema.

For $10,000 initial cash, M=$10,000 and n=172,800 seconds, the upper is
$1,728,020,000. The independent stay-cash policy gives a certified $10,000
lower certainty equivalent at one hour; the replay's cash decision does
not prove that this is the one-hour optimum. Thus the currently established
distribution-independent upper on C_48h/C_1h - 1 is **172,801**, or
**17,280,100%**. This is much too loose to certify 95% retained value and is
not an estimate of the actual tail contribution.

The evaluator also checks 24h, 47h59m59s and 48h. This envelope certifies
epsilon < 0.1% only at 48h, by the empty-tail identity. It neither identifies
the earliest sufficient depth nor proves that an earlier depth fails.
Obtaining a useful number requires a tighter bound under the recursive NSF
law and a certified prefix value; the earlier five-stage distribution bound
does not supply them.

For the reusable value approximation, the same Lean module proves that an
exact terminal anchor and upper/lower Bellman residual inequalities bounded
uniformly by delta yield

    V*_H - V_policy,H <= 2 H delta
    CE*_H / CE_policy,H - 1 <= exp(2 H delta) - 1.

For H=3,600 and a 0.1% target, a sufficient condition is

    delta < log(1.001) / 7,200 = 1.38819490706e-7.

The code computes a conservative rational threshold from the existing
outward log interval. The selected validation maximum is about 264 times
this threshold. More fundamentally, these 94 finite sampled cases do not
establish a uniform residual bound. The theorem requires the inequalities
over all relevant finite-value states, remaining horizons and competing
actions, plus the selected policy's lower backup. Sampling and action-menu
errors must be covered if comparing against the full-law continuous optimum.
The proof supplies a sufficient certificate contract; it does not prove
convergence of this neural training procedure or certify this checkpoint.

## Verification and reproduction

All 17 targeted Python tests pass, covering gamma-one action selection,
nonanticipative averaging, exact terminal handling, ruin, account/clock
inputs, chunking, physical-second indexing and execution constraints.
The full Lean build and axiom audit pass. Numerical Python execution is
tested against the rational implementation; it is not extracted from Lean.
No live bot default or exchange execution was changed by this experiment.

Artifacts:

- `data/backtests/oracle-continuation-v2/`: training plan, history, source snapshots and weights.
- `data/backtests/oracle-continuation-hour/`: complete one-hour fixture and exact replay.
- `data/backtests/oracle-continuation-full/`: complete archive trace, report and exact replay.
- `data/backtests/oracle-continuation-bound/report.json`: exact cap bounds, residual threshold and certificate status.
- `data/backtests/oracle-continuation-lean.log`: successful build and axiom audit.

Use fresh output directories when reproducing:

```powershell
.venv-ml/Scripts/python.exe ml/train_oracle_continuation.py --output data/backtests/new-continuation --steps 3000
.venv-ml/Scripts/python.exe ml/backtest_oracle_continuation.py --value data/backtests/new-continuation/value.pt --output data/backtests/new-continuation-replay
node --conditions=development --import tsx scripts/audit-gpu-replay.ts data/backtests/new-continuation-replay
node --conditions=development --import tsx scripts/evaluate-oracle-continuation-bound.ts data/backtests/new-continuation-replay data/backtests/new-continuation-bound
.venv-ml/Scripts/python.exe ml/test_oracle_continuation_value.py
.venv-ml/Scripts/python.exe ml/test_gpu_oracle_bellman.py
node scripts/theory-lean.mjs check
```
