# Raw execution prices in the dense account map

The subsequent [checked tail-repair implementation](oracle-dense-tail-repair-2026-09-14.md) adds residual accounting, a further actual-arrival feasibility correction, and a rolling backtest runner. The measurements here describe the preceding execution kernel and should not be treated as current cached values.

The market cloud now stores each conditional sample's actual price separately from its continuation-node assignment. Compression may choose a nearby observable state for continuation lookup; it no longer changes the mark used for execution. This is a correction to the model being evaluated, not just a numerical acceleration.

## Execution and continuation

`ScenarioGraph.sampled_prices` preserves the NSF draws before medoid projection. `MarketCloud.sampled_prices` carries them through archive round trips and rolling updates. CPU and CUDA backups use these prices for funding, mark-to-market equity, liquidation checks, pending-order crossing and closing constraints.

When an arriving account's actual mark differs from its retained node's mark, the next conditional row is interpreted as relative moves and reanchored to that actual mark, then rounded to the configured price tick. For saved parent mark `p_node`, sampled next price `p_sample` and actual current mark `p`, the next execution mark is `tick * max(1, round(p * (p_sample / p_node) / tick))`. Both interior closing fallbacks and actual root queries use this rule. This prevents an artificial jump back to the representative's absolute price level.

Continuation interpolation still uses the retained node. Consequently, preserving execution prices does not remove the error from approximating the conditional market state, nor establish optimality over continuous actions. Old archives without raw draws are rejected with a regeneration instruction; their projected prices cannot reconstruct the missing draws.

## Measurements

The new complete initial map is `data/backtests/oracle-dense-raw-price-gpu`. It uses the existing NSF checkpoint, 3,600 one-second layers, eight market nodes, all eight conditional draws, three wealth nodes, 315 account cells and the existing candidate catalog. Fresh market construction took **53.37 seconds**. The CUDA value sweep took **90.87 seconds** including initialization and export. The initial action was cash, with estimated log value `log(10000)`. This is a planning measurement, not a realized trading return.

The market reuse audit `oracle-dense-raw-price-rollout` advanced through 120 actual observations. Median market refresh was **43.50 ms**; all 120 refreshed root rows had **zero execution-price RMSE and zero mean-return shift relative to their raw samples**. The retained continuation nodes still differed from those samples: node-price discrepancy reached 2.66 bps RMSE and 2.55 bps mean-return shift. These latter measurements now describe lookup approximation rather than an altered execution law.

The old projected and corrected initial maps retained identical node prices and destinations. Among 3,664,268 paired finite values, the maximum change was 13.91 bps in normalized log value and RMS change was 2.40 bps. Their finite-support patterns differed at 471,684 entries. Execution at different marks can change whether orders fill and whether a valid terminal close is available, so old feasibility/value caches must not be reused as corrected results.

## GPU profile

`profile_oracle_dense_gpu.py` profiles a warmed 60-layer full-sample sweep on the RTX 3070. Three unprofiled runs took 1.3850, 1.3827 and 1.3837 seconds. The profiled candidate kernel consumed 1.371 seconds of device time; the mean kernel took 1.15 ms and host-to-device copies took about 0.10 ms. This places the bottleneck inside GPU candidate evaluation, rather than in CPU scheduling or transfers.

Conditional price factors are now resident on the GPU for the whole sweep. A further experiment packed feasible candidates before continuation evaluation. It passed the conformance tests but increased median 60-layer runtime from 1.3837 to 1.4101 seconds, so it was removed. The baseline and rejected experiment's profiles are saved as `profile.json` and `profile-packed.json` in the initial-map directory. Execution remains double precision.

## Verification and remaining evaluation

Regression coverage includes raw draws opposite to their representative's direction, limit crossing and liquidation under unprojected marks, reanchored subsequent closing prices, graph-to-cloud preservation, rolling source immutability, archive preservation and rejection of old archives. These supplement the existing CPU/CUDA execution, sample-reuse and terminal-flat tests.

The final production-kernel regression run passed **47 tests** in 3.42 seconds of test execution (7.23 seconds including process startup). The rejected packing variant had also passed these tests; its rejection was based on measured runtime.

`audit_oracle_dense_market_values.py` compares retained and freshly rebuilt continuation grids at the next actual observation. Both use the same raw root draws and all conditional samples across the full 3,600-step horizon. Its result measures a finite-grid discrepancy; it cannot serve as a certificate against the underlying continuous market law.

The measured comparison is `data/backtests/oracle-dense-raw-price-market-values`. Across 110 paired root proposals, the maximum absolute log-value discrepancy was **0.01140 bps**, and the range of action-relative errors was **0.01380 bps**. There were zero finite-support disagreements. Both maps chose cash; the retained map's selected action had zero regret against the evaluated fresh-grid proposals. The retained and rebuilt value sweeps took 88.61 and 88.32 seconds respectively. Retaining the market grid reduces forecast work, but the current implementation still recomputes its value field.

## Corrected Monte Carlo reuse

`data/backtests/oracle-dense-raw-price-paired` retests the guarded cash-centered estimator with raw execution marks and rolling interior market kernels. Initial construction and both full-sample reference fields were freshly computed. The complete experiment took 453.64 seconds. Each update requested two conditional samples; the feasibility guard increased the effective mean to 5.32 out of eight, retaining about 33.5% of cached outcomes.

| Update | Refresh | Maximum root log-value error | Root action-relative error range | Selected action's reference regret | Full-field maximum error |
| --- | ---: | ---: | ---: | ---: | ---: |
| +1 second | 61.69 s | 0.1748 bps | 0.1749 bps | 0 bps | 1.945 bps |
| +2 seconds | 61.51 s | 1.6539 bps | 1.6559 bps | 0.8083 bps | 5.384 bps |

Finite-support disagreements were zero for both entire fields. Field RMS errors were 0.0050 and 0.0152 bps. At the second update the estimator placed one pending order; the reference chose cash. The reported regret is a difference in forecast log utility, not a realized loss or return. These results are substantially better than the previous projected-price/rebuilt-grid reuse experiment, but two updates cannot establish long-run stability. Both the raw-price correction and retaining conditional kernels changed this experiment, so the improvement cannot be attributed exclusively to one of them. Paired reuse remains explicitly selected with `--estimator paired`.

## Tail propagation diagnostic

`audit_oracle_dense_tail.py` verifies that interior market nodes, raw conditional rows and absolute-time candidate schedules align before comparing shifted full-sample fields. In `oracle-dense-raw-price-market-values/tail-audit.json`, 2,482 of 3,598 aligned interior layers were bitwise unchanged, treating matching NaNs as the same missing-cell marker. The first change was at depth 2,483.

| Allowed normalized log-value difference | Last qualifying boundary | Tail layers after that boundary |
| --- | ---: | ---: |
| 0 | 2,482 | 1,118 |
| 1e-10 | 3,042 | 558 |
| 1e-8 | 3,294 | 306 |
| 1e-7 | 3,410 | 190 |
| 1e-6 (0.01 bps) | 3,481 | 119 |

At each listed boundary, all earlier aligned layers also stayed within the stated tolerance, with no support differences. These are offline measurements obtained from completed full sweeps. An adaptive tail-repair algorithm has not yet replaced those sweeps.

The useful mathematical property is nonexpansiveness of a shared finite-grid Bellman backup on fields with equal finite support: positive interpolation weights with total at most one, conditional probability averaging and maximization cannot amplify a uniform continuation-value error in exact arithmetic. This suggests a checked stopping rule, rather than assuming distant rewards are negligible. Repeated updates would additionally need to account for residual errors introduced at earlier copied/repaired boundaries, as well as floating-point error. A tolerance measured once must not be silently treated as a persistent certificate.

A fully refreshed realized-hour backtest remains too slow for the requested few-hour budget: 90.87 seconds per update would extrapolate to about **91 hours of value computation** for 3,600 actual decisions, even with cheap market reuse. No long rolling backtest or profitability claim follows from these measurements. Stable value reuse and continuation-grid accuracy remain unresolved.

```powershell
.venv-ml/Scripts/python.exe ml/benchmark_oracle_dense_map.py --output data/backtests/oracle-dense-raw-price-new --horizon 3600 --updates 0 --backend cuda --budget 8
.venv-ml/Scripts/python.exe ml/audit_oracle_dense_rollout.py --source data/backtests/oracle-dense-raw-price-new --output data/backtests/oracle-dense-raw-rollout-new --updates 120
.venv-ml/Scripts/python.exe ml/profile_oracle_dense_gpu.py --source data/backtests/oracle-dense-raw-price-new --output data/backtests/oracle-dense-raw-price-new/profile.json --depth 60
.venv-ml/Scripts/python.exe ml/audit_oracle_dense_market_values.py --source data/backtests/oracle-dense-raw-price-new --output data/backtests/oracle-dense-raw-market-values-new
```
