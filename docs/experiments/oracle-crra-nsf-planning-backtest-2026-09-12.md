# NSF-driven Oracle planning and backtest

Implemented a computable scenario-tree Bellman planner using the sampled-skip latent NSF, and backtested it over the complete archived forecast interval, 2026-08-10T00:00:01+00:00 through 2026-08-11T16:08:54+00:00: 144,533 one-second marks, 161 action decisions, and one fixed terminal date.

The tested planners lost money. All numerical Bellman maximizations are exact for their supplied finite trees and finite menus. This is not a certified epsilon-optimal replay of the unrestricted continuous, one-second construction: approximation errors against that construction remain unquantified.

## Counterfactual forecasting and information

Checkpoint: `data/backtests/oracle-latent-nsf-sampled-skips/latentSampledSkipFlow.pt`, selected at epoch 4 on validation. Forecast weights are unchanged. The 256-dimensional latent model receives raw production features directly; no direct59 summary is used.

At each actual decision, encode the two completed observed feature/price rows. For each future branch, sample the inferred latent posterior, propagate it with NSF to that branch time, and sample the Gaussian observation decoder for price and all 59 features. Discard the hidden latent samples. At child nodes, rerun the causal encoder on the two most recent synthetic observations and generate their conditional forecasts.

This defines an observable forecast kernel from the learned encoder, transition, and decoder. It is an approximation to market laws; the variational encoder is not asserted to be an exact Bayesian filter of the latent generative process. Gaussian simulated feature vectors may also violate identities of real production features. No forecast issued later on the realized history is read by an earlier tree.

Continuation decisions can distinguish different OBSERVED feature vectors at the same price. The tree canonicalizer now merges branches by price plus observable feature information. Identical observable histories are merged before optimization; hidden latent/scenario labels are not valid distinguishing information. Tests verify both cases.

Branch samples use independent Gaussian draws across nodes with antithetic pairs within each node. Each branch has equal rational probability. Samples are an empirical approximation, not a bounded-tail quadrature certificate. More samples can change the optimized policy; the measured two-versus-four-branch difference is material.

## Horizon and execution

Actual action decisions are 900 seconds apart. Every initial tree has three transitions: the next decision, an intermediate future time, and the final backtest time. Near the endpoint, duplicate times are removed. Every modeled interval stays within the NSF's 24-hour query range. Increasing `--planning-level` adds nested future observation/action times through the same endpoint. Increasing it sufficiently includes every second; reducing the actual `--stride` to one second is also required to approach the original decision schedule.

The coarse trees check future maintenance and fills only at their modeled times. Actual replay checks every second, including arrival fills and liquidation. Synthetic encoder updates likewise see coarse observation intervals, whereas actual decisions use the last two consecutive seconds. These omitted future observations, fills, maintenance checks, and action opportunities are approximation error, not proved irrelevant. NSF's flow regularizer does not guarantee exact consistency between coarse direct transitions and many one-second transitions.

Execution reuses the existing exact rational Oracle implementation:

- Initial equity $10,000; gamma 0, expected terminal wealth.
- Entry exposure [-2,2], maintenance exposure [-3,3], reserve leverage 2.
- Maker fee 0.0002, taker fee 0.0005; separate zero-fee diagnostic.
- Quote-order minimum $5, aggregate per-decision maximum $10,000, price step $0.01, one active order slot.
- Signed-asset funding factor 1. This run does not charge historical funding; there is no substituted borrowing-interest model.
- Valid capped terminal close after arrival fills and cancellations; remaining inventory is marked and retained.

Prices are the existing archived BTCUSDT spot series, using these configured Oracle account parameters. The tests are model experiments, not claims to reproduce all historical futures-exchange parameters.

The fast market menu initially targets [-2,0,2]; refinement adds -1 and 1. Actual fees and quote caps determine the realized exposure. Zero quote flow is a no-op. Limit orders use one-tick passive prices and a quarter-equity notional subject to the cap. Cancellation subsets are available. The coarse menu selects market adjustment or placement; `--joint-actions 1` enables its full Cartesian product.

`exhaustiveOracleMenu(level)` supplies a stronger refinement family. It enumerates signed quote amounts and notionals equal to current equity times m/d, 1<=m,d<=level, all passive grid offsets through level, every cancellation subset, and ordered placement lists up to the account slot bound. Invalid controls are rejected by Oracle execution and continuation checks. Increasing level includes every rational physical control at any rational account state; it does not silently clamp an enumerated action. `--enumeration-level` augments the fast menu with this family, including simultaneous actions. Level 1 was also backtested.

## Formal approximation contract

New Lean module: `TradingTheory.ComputableBellmanApproximation`.

1. `menu_greedy_bound`: if a menu loses at most eta in true Bellman score relative to every feasible action, and numerical scores have uniform absolute error rho, its maximizing action loses at most eta + 2*rho.
2. `policy_error_bound`: a validated per-backup loss delta accumulates to at most N*delta over a fixed N-decision horizon.
3. `tolerance_allocation`: delta <= epsilon/(N+1) suffices for total loss <= epsilon, including N=0.
4. `exhaustive_search_approaches`: nested exhaustive search approaches a supremum if the enumerated feasible controls are value-dense. Value density is an explicit hypothesis; an exposure grid alone does not imply it across liquidation or dust discontinuities.

For a fixed finite scenario tree with rational Oracle parameters, fixed grid order prices and bounded book size, execution can be partitioned into finitely many rational polyhedral regimes for trade/placement variables. Within each regime, balances and gamma-0 objective are affine. Rational points are dense within a nonempty rational regime, including approximation from the correct side of strict dust/liquidation boundaries. This motivates rational rather than only dyadic exposure enumeration for continuous control on that finite tree. The complete Oracle-to-polyhedra/value-density bridge is a mathematical argument here, not a completed Lean refinement proof of the enumerator.

Approaching the original full-distribution policy additionally requires control of scenario-law/integration error, rare-event feasibility, omitted ticks, and wealth tails. Finite Bellman values or suitable integrability bounds must be established for that target model. Those conditions are not automatic consequences of NSF, and this run supplies no validated uniform eta or rho. Therefore the new lemmas provide the precise convergence/error contract, not an unconditional end-to-end certificate for this backtest. They do not bound the difference between the learned model and the real market.

## Results

| Configuration | Net return | Maximum drawdown | Fees at execution prices | Market orders | Limit fills |
|---|---:|---:|---:|---:|---:|
| 2 branches, coarse actions | -4.0167% | 5.3234% | $12.6843 | 12 | 3 |
| 2 branches, finer actions | -4.0167% | 5.3234% | $12.6843 | 12 | 3 |
| 2 branches, rational enumeration level 1 | -4.0167% | 5.3234% | $12.6843 | 12 | 3 |
| 4 branches, coarse actions | -2.9940% | 4.0692% | $12.0279 | 6 | 4 |
| 2 branches, zero fees | -1.5958% | 2.5709% | $0.0000 | 5 | 0 |

Cash returned 0%. Oracle buy-and-hold with the same fees and valid terminal close returned -2.1017%; the underlying gross price return was -2.0038%.

The finer exposure menu and first exhaustive rational level produced the same realized trades and wealth as the original two-branch menu. Four branches changed the policy and improved the result by about 1.02 percentage points, but it still underperformed buy-and-hold and cash. This is evidence of approximation sensitivity, not numerical convergence.

The zero-fee policy also lost money. The fee and zero-fee results are different optimized policies, so their wealth difference cannot be interpreted as fees paid on the same trades. Average absolute exposure was about 1.82 with fees versus 1.00 without fees in the two-branch runs. No tested run liquidated. Terminal inventories can remain substantial because the final close shares the configured $10,000 cap; they are included in wealth.

These historical test data were already inspected during model development. Configurations here are diagnostics, with no model retraining, calibration fitting, or promotion based on their test returns.

## Verification and artifacts

- The entire Lean library builds and its axiom audit passes, including all four new lemmas.
- Lean's exact checker accepts four root Bellman certificates for the two branch counts and two exposure grids, plus 108 existing Oracle conformance cases. These certificates verify finite-menu maximization/wealth arithmetic; they do not establish continuous coverage or forecast correctness, and the transition conformance cases are not a proof about every foreign-code execution.
- 34 Oracle policy/execution/refinement tests pass, including the two new rational-menu tests and observation-information merge test. Two Python rollout tests verify nested fixed-endpoint schedules and independence from other realized-history rows. The Oracle TypeScript typecheck passes.
- Every replay uses exact rational account arithmetic and validates actual committed fills. Source snapshots and their hashes are retained per run, including the earlier runner revision used for the first diagnostics.

Key artifacts: `data/backtests/oracle-nsf-planning/comparison.json`, per-run `decisions.json` and `report.json`, `forecast-plan.json`, `forecasts.jsonl`, `search-audit.json`, and `lean-verification.json`. Four-branch forecasts/replays are under `data/backtests/oracle-nsf-planning-branches4`.

```powershell
.venv-ml/Scripts/python.exe ml/export_nsf_planning_trees.py --output data/backtests/nsf-new-run
node --conditions=development --import tsx scripts/backtest-nsf-oracle.ts --source data/backtests/nsf-new-run
node --conditions=development --import tsx scripts/backtest-nsf-oracle.ts --source data/backtests/nsf-new-run --enumeration-level 1 --output data/backtests/nsf-new-run/enumerated
```
