# Reusing dense-map estimates and market transitions

The later [raw execution-price correction](oracle-dense-raw-execution-prices-2026-09-14.md) removes the price replacement described below. The measurements in this report remain historical results of the projected-price implementation; its value caches are not corrected-model results.

The implementation now has persisted per-outcome controls, fresh correction samples, a shared cash baseline, and an all-outcome feasibility guard. The guarded estimator passes its conformance tests and preserves the census field's finite support in the measured experiment. Its value errors remain too large to accept it as the accurate controller. The existing sampled CUDA sweep remains the default; reuse requires `--estimator paired`.

## Corrected finite-row estimator

At one fixed parent/account point, let `g_b` be the current Bellman outcome value on edge b, including the maximization after the next observation. Let `r_b` be the corresponding value for a shared zero-order cash account at the middle wealth node. All B reference outcomes are computed. The stored control `c_b` is an older estimate of `g_b-r_b`, indexed by the canonical edge, not by its rank in the current permutation.

For a fresh uniform subset I of n distinct edges, the estimate is

\[
\widehat m=\frac1B\sum_b r_b+\frac1B\sum_b c_b
+\frac1n\sum_{b\in I}\bigl[(g_b-r_b)-c_b\bigr].
\]

Conditional on the current outcome functions and the existing controls, uniform inclusion probabilities give `E[estimate] = mean(g)`. With a census it reduces to the current ordinary mean. This is a local identity for the finite row. It does not prove that a sequence of noisy maximizations produces an unbiased optimal value or that the sampled market cloud represents the true market accurately.

Only selected control entries are replaced by their new centered outcomes. The exact mean of the resulting stored vector remains known, even though its components can come from different observations. It is this exact control mean that is reused, not the previous noisy value estimate. Probability weights remain uniform; no likelihood-ratio or Bayesian SIR interpretation is asserted.

Candidate catalogs remain keyed to absolute time. Correction permutations also include the new origin and a persisted generation counter, ensuring a fresh stream when the same timestamp is revisited. Old arrays remain immutable. `outcomes.npy` and `reuse.json` preserve the controls, their representation and generation across reloads. For this configuration the outcome file adds **1,741,824,000 bytes** to each archived map.

Cash centering cancels changes in continuation level shared by all account states. A regression test changes that common level differently at each child market node and verifies that every tested partial sample gives the same result as the census at interior wealth points. This cancellation alone did not stabilize the full experiment.

## Feasibility guard

A finite-value mean cannot safely sample away an outcome with value minus infinity. Old nonfinite controls already force a census. The final implementation also checks every current conditional arrival before allowing a partial update. Every arrival must satisfy the execution checks and admit a valid immediate full close, including the traffic consumed by committed fills. Otherwise the complete value backup is evaluated.

The close is a feasibility witness, not a restriction on the policy selected by the value backup. The actual maximization can choose any of the configured candidates.

For the configured finite grid, this preserves the census finite/nonfinite pattern by backward induction. A partially evaluated row has an explicit finite close-and-remain-cash policy on every edge. For a fully evaluated row, finiteness depends on the same action feasibility, interpolation-neighbor support and closing fallback as the census. Assuming the child support patterns agree, so do these full rows. The endpoint uses the same valid-close rule. This argument concerns support of the numerical construction; it does not certify feasibility of every continuum state between grid representatives.

## Measured updates

The same frozen one-second NSF clouds, 3,600-step horizon, 315 account cells, three wealth nodes and eight conditional outcomes were used. Partial updates request two outcomes; the guard can raise this to eight. The final run is `data/backtests/oracle-dense-map-paired-guarded-verified`.

| Observation | Guarded refresh | Effective outcomes per scheduled point | Root maximum paired log-value error | Selected action's census regret | Field support disagreements |
| --- | ---: | ---: | ---: | ---: | ---: |
| +1 s | 50.8 s | 4.30 | 60.33 bps | 0 bps | 0 |
| +2 s | 48.6 s | 4.30 | 496.19 bps | 2.71 bps | 0 |

Approximately 46.2% of cached outcome entries were retained per update. Across all wealth/account/market/time points, the maximum paired field errors were 276.5 and 11,393.0 bps; RMS errors were 69.4 and 2,056.9 bps. Root action-relative error ranges were 60.33 and 496.19 bps, so the discrepancy cannot be dismissed solely as a common value offset. These are errors against the finite-cloud census, not errors against the continuous optimum. The two observed decisions are insufficient to establish a regret bound for future decisions.

The complete census initial map and reference maps were reused from `oracle-dense-map-paired-centered` after checking their execution/search contracts, market arrays, sample counts and completion markers. Their original sweep times were 88.9 seconds initially and 85.9/86.2 seconds for the references. Copy/load times in the final run are explicitly labeled as cached results, not fresh solver timings. Current reference front actions were recomputed for the actual account reached by the new run.

Earlier results are retained for diagnosis. Raw paired corrections in `oracle-dense-map-paired-fresh` reached 1,624 bps maximum root error on the second update. Cash centering alone in `oracle-dense-map-paired-centered` reduced that to 426 bps, but those runs also had 12,271 and 11,011 full-field support disagreements. The final guard removed the support disagreements; it did not solve the value-estimation instability. The initial `oracle-dense-map-paired` run was stopped when the non-fresh sampling-stream issue was found. `oracle-dense-map-paired-guarded` stopped during configuration validation before computation.

## Reusing conditional market transitions

`ml/oracle_dense_rollout.py` implements a separate diagnostic. With a fixed NSF model, an unchanged hypothetical observation pair at an unchanged absolute timestamp has an unchanged conditional kernel. The procedure freshly predicts the new actual root, shifts retained interior states/conditional rows, appends a newly predicted tail and recomputes all marginal masses from the new root. It does not keep the old unconditional future distribution. Earlier input arrays are unchanged.

This makes market refresh cheap, but mapping the new root's samples to the old observable-state grid changes their prices. A 120-observation causal audit in `data/backtests/oracle-dense-rollout-reuse` took **5.59 seconds** in total, with a **41.7 ms** median refresh. It predicted nine parent rows and reused 28,784 interior rows per refresh. Mass normalization stayed within numerical roundoff.

Root price-projection RMSE averaged **1.01 bps**, with a **2.66 bps** maximum. The absolute mean-return shift reached **2.55 bps**. Every audited root exceeded 0.1 bps RMSE; 50 exceeded 1 bp. These thresholds are descriptive diagnostics, not economic error certificates. The shifted grid has not been connected as the default forecast source or used to claim a successful trading backtest.

## Remaining work

Value reuse still needs a stable, accurate update rule. The supported-state correction removed one concrete failure mode but has not met the accuracy requirement. Market reuse needs to preserve the actually sampled transition prices rather than replacing them by retained representative prices, while controlling the resulting continuation approximation. The current grid, finite candidate menu and interpolation still require refinement/error evaluation beyond CPU/GPU agreement.

Even at roughly 49 seconds per value refresh, a full hour of one-second rolling decisions would take roughly **49 hours for value updates alone** under linear extrapolation. Adding the previously measured 53.8-second full forecast construction would bring that estimate to about **103 hours**. These are extrapolations, not measured backtests. The faster market-grid diagnostic cannot yet be substituted without accepting its projection error. Consequently, a long fully refreshed backtest has not been launched under the requested few-hour compute budget. The overall implementation/evaluation goal remains unfinished.

## Commands

```powershell
.venv-ml/Scripts/python.exe ml/benchmark_oracle_dense_map.py --output data/backtests/oracle-dense-map-paired-new --horizon 3600 --width 8 --branches 8 --budget 2 --updates 2 --candidates 16 --market-cache data/backtests/oracle-dense-map-hour --census-cache data/backtests/oracle-dense-map-paired-centered --backend cuda --estimator paired
.venv-ml/Scripts/python.exe ml/report_oracle_dense_reuse.py data/backtests/oracle-dense-map-paired-new
.venv-ml/Scripts/python.exe ml/audit_oracle_dense_rollout.py --source data/backtests/oracle-dense-map-hour --output data/backtests/oracle-dense-rollout-new --updates 120
```

Omit `--census-cache` to recompute the initial and reference maps. All **42 tests passed** in the final regression run. They cover the local estimator identity, fresh persisted generations, canonical-edge cache replacement, unchanged source arrays, cash-level cancellation, unseen liquidation, multistep book-support equality and the rolling market kernel/mass construction, alongside the existing execution and value-map checks.
