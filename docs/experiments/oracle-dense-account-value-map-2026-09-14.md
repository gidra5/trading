# Dense account-state value map

The implementation allocates every configured account cell at every second of a one-hour continuation. Each cell contains value estimates at sampled market states, conditional-sample dispersion and a compute allocation. Account-cell observation counts and residual statistics guide later refreshes. This is an experimental numerical controller; it has not been connected to live trading.

“Pending orders” means the account's own limit orders. There is no exchange-depth, queue, matching-engine or partial-fill simulation. A passive order activates on the next tick and fills completely at its limit when the observed move crosses it, according to `Oracle.md`.

## Representation

The experiment has 315 account cells: seven exposure nodes, eight order-bin types, and 45 canonical multisets containing zero, one or two orders. Equal permutations share a cell; duplicate orders retain their multiplicity. Each order has two numerical coordinates, so the continuous coordinates of a K-order account have dimension **1 + 2K** before conditioning on wealth.

An order's price coordinate is `z = x/(1+x)`, with `x = abs(limit/currentPrice - 1)/0.0001`. Two bins span `0 <= z < 1`. Four signed notional/equity bins span `[-2,2]`; the sign selects buy or sell. Price and size bin centers are representatives. The seven exposure nodes are `[-2.5,-2,-1,0,1,2,2.5]`. The outer exposure cells extend to maintenance bounds `[-3,3]`, but interpolation requires an exposure inside the node range. Invalid and unsupported representatives remain allocated and explicitly marked.

Wealth has separate nodes at $5,000, $10,000 and $20,000. Absolute minimum and maximum order sizes make wealth essential. The initial account has $10,000 cash. The numerical state includes quote balance, asset quantity, price and the complete pending-order multiset. Small fixed arrays are used inside the execution kernel; permutations do not create additional value-map cells.

Each second has up to eight sampled market representatives, shared by reference across account cells, and eight equally weighted conditional samples per market parent. A representative contains two observed feature rows and their prices. Child rows remain conditional on their own parent. Unconditional market masses are used for market compression and diagnostics, never substituted for these conditional laws.

`values.npy` and `variance.npy` have axes `(3600, 3 wealth nodes, 315 account cells, 8 market nodes)`. Together they occupy 435,456,000 bytes (415.3 MiB), excluding market observations and other diagnostics. Dispersion is a sample statistic, not a confidence certificate; the forecast generator uses antithetic samples.

## Value and execution

The field stores post-decision expected terminal log wealth minus log current equity. Root search executes the actual fee- and cap-aware transition first, then looks up that actual post-state. Capped targets can land in a different cell. Root search includes cell representatives and samples from cell interiors, as well as holding, valid full closing, and retaining or cancelling subsets of existing orders.

Completed experiment fields can be reopened with `DenseValueMap.load(directory)`. It checks the execution-kernel hash and completion markers and maps the value arrays read-only. A refresh writes a new map and can use the loaded field as its previous estimate. The restart test recovers the same decision value without rebuilding the map.

The continuation uses bilinear wealth/exposure interpolation and constant interpolation within order bins. Future action search always includes the zero-order exposure targets, holding and every retained-order subset, plus sampled placement candidates. The experiment requests 16 representative candidates per depth before these mandatory actions. The sample catalog rotates with absolute time. This is a finite candidate approximation, even though the root proposals can have continuous sizes and prices before tick rounding.

The Numba execution kernel applies signed asset funding, liquidation before arrival fills, crossing-based maker fills, received-currency fees, shared aggregate quote-flow caps, per-trade minimums, entry/maintenance exposure checks and shared collateral reservations. New orders activate at the next tick. Pending orders that do not fill persist in the continuation lookup. Terminal inventory must close with a valid trade under the same cap; residuals are not erased.

An explicit additional continuation evaluates closing at the next arrival and then staying in cash. It checks all edges of the finite conditional row, including committed fills before closing. This recovers usable values for actual small residuals when grid interpolation has an unsupported neighbor. A regression test covers closing a short in two capped decisions. This fallback does not certify the interpolated component of the value estimate.

The benchmark uses maker fee 0.0002, taker fee 0.0005, entry exposure `[-2,2]`, maintenance exposure `[-3,3]`, quote-flow bounds $5–$10,000 and a $0.01 tick. Funding multiplier is 1 in this experiment; nontrivial funding is covered by execution-conformance tests. These are the existing experimental Oracle parameters, not an assertion about a particular account's current exchange settings.

## Observation and refresh

After an actual new observation:

1. Apply that price move to the chosen account and its pending orders.
2. Evaluate counterfactual one-step TD residuals for the old front proposals, using the old depth-one continuation and its unchanged terminal date. The observed market feature pair is projected to the closest old market representative. Assign each residual to its actual post-state cell and aggregate duplicates so one observation counts once per cell.
3. Shift old cell values and dispersion by one second for use as priorities. Residual mean and dispersion also affect priority.
4. Allocate between two and eight conditional outcomes per cell, averaging four in the reduced-sample experiment. Weighted sampling of available compute slots preserves the floor, cap and total budget.
5. Recompute target means backward under the updated conditional market law. Each cell uses a uniformly selected subset of its conditional sample row. Profitability does not change market-outcome probabilities.

The residuals are correlated, bootstrapped diagnostics. They are used for computation priority, not added to every value as if they were known forecast bias. Existing particles are not SIR-reweighted: no likelihood ratio has been derived for doing that here. Absolute-time random seeds couple successive forecast constructions, but market quantization can change representatives and sample pairing. This implementation refreshes the full value field; it does not yet retain old target means through a statistically corrected delta-only estimator.

## Validation and interpretation

The tests check unordered-book identity, multiplicity, dense cell coverage and within-cell sampling; single-order execution against the existing Torch Oracle with nontrivial funding; multi-order cap aggregation; next-tick activation; reservations; liquidation timing; terminal residual rejection; retention of pending orders in value lookup; conditional averaging before action selection; deterministic two-step value; bounded root trades; two-stage short closing; and observation updates that leave old values and market probabilities unchanged.

A fresh eight-outcome sweep on the same frozen market cloud is the reference for the reduced-sample sweep. Comparisons evaluate identical actual post-states. They measure additional sampling/backup differences within this grid, market cloud and candidate search. They do **not** measure error against the true continuous optimal policy, the uncompressed NSF law or the actual market law.

In particular, finiteness of an interpolated value is not a feasibility certificate for the represented continuum. Hard order-size and terminal boundaries can intersect a cell. The explicit closing fallback and final-query checks cover specific failure modes, while a general error/feasibility certificate remains unestablished. No new claim of Lean-verified optimality is made for this interpolation scheme.

The heatmap displays average log gain across finite market samples and separately displays the finite-sample fraction. Its display average is not the conditional expectation used to choose an action. Missing estimates can arise from interpolation support and the finite candidate search as well as actual execution constraints, so a grey cell does not prove economic infeasibility.

The rolling smoke replay covers only the initial decision and two subsequent real observations. Each map looks 3,600 seconds ahead. That is different from replaying a full hour of realized decisions with a refreshed map each second. No predicted certainty-equivalent wealth or short-run marked equity should be read as a completed terminal-wealth backtest.

## Frozen-map replay

An additional full realized-hour replay uses each archived map with decreasing time to its fixed endpoint. New observations select a nearby market representative; the NSF and value field are not refreshed in this experiment. Relative next-price moves are anchored to the actual current price and rounded to the tick, avoiding an artificial jump back to the old cloud's absolute price. At the last real decision, actual candidate closing values on every conditional price sample replace interpolation. These query corrections were added after the field-construction benchmark started; its archived source snapshot identifies the construction code, while the replay uses the corrected query interface.

The initial replay exposed those two query issues and failed its terminal-close check; it is retained as a failed diagnostic in `data/backtests/oracle-dense-map-frozen-replay`. With the corrections, replays using `update-0`, `update-1` and `reference-1` all ended flat at **$10,000, 0% return**, with no market trades or limit fills. Each replay evaluated 3,600 decisions in approximately **1.44–1.50 seconds**, excluding approximately 0.45–0.59 seconds of loading and Python startup. This demonstrates inexpensive queries against an existing map, not an inexpensive forecast/value refresh. Unfilled order placement and cancellation can occur even when there are no fills.

The corrected results are in `data/backtests/oracle-dense-map-frozen-replay-corrected`, `data/backtests/oracle-dense-map-frozen-replay-update1` and `data/backtests/oracle-dense-map-frozen-replay-reference1`. These substantially overlapping hours are implementation checks, not evidence of profitable or broadly reliable trading.

## Measured results

The completed experiment is `data/backtests/oracle-dense-map-hour-verified`. It uses the test observations beginning **2026-08-10 00:00:01 UTC**. The initial eight-outcome map took **284.4 seconds**. The preliminary GPU construction of its market cloud took **53.8 seconds**; the verification subsequently reused the frozen cloud. Hardware was a Ryzen 7 3800XT (eight CPU worker threads) and an RTX 3070.

| Observed update | Reduced-sample sweep | Eight-sample reference sweep | Maximum paired root log-value difference | Reference value gap of selected action |
| --- | ---: | ---: | ---: | ---: |
| +1 second | 242.2 s | 280.0 s | 3.43 bps | 0 bps |
| +2 seconds | 259.2 s | 262.6 s | 3.24 bps | 0 bps |

The reduced-sample sweeps requested four outcomes per cell on average, with a floor of two and a cap of eight. The closing fallback still evaluates every conditional edge. Allocation and mandatory searches also remain work, so halving the requested sample budget did not halve runtime: the measured sweep savings were **13.5% and 1.3%**, approximately **7.6% combined**. This is not yet a cheap delta-only refresh.

Both reduced-sample updates selected two resting buy orders with zero current exposure; the corresponding reference selected cash. The selected orders **tied in reference value** with the reference's preferred action—they were not identical decisions. There were no finite-coverage disagreements among the compared root proposals. Across the entire $10,000 field, the first update had 167 finite-coverage disagreements and a maximum finite paired log-value difference of 4.48 bps; the second had 147 disagreements and a maximum difference of 4.21 bps. Thus the root comparison cannot be promoted to a uniform feasibility or accuracy guarantee.

At the initial root, 167 of the 945 wealth/account representatives passed the static execution-state checks; 125 had finite continuation estimates. At $10,000 specifically, those counts were 49 and 40. Most grey cells are statically invalid, but some have missing continuation support. All 945 cells are allocated at every second regardless.

The final value-query implementation passes **16 new tests**, and the **10 existing Torch Oracle tests** also pass. The complete construction, two reduced-sample updates and their two reference sweeps took **1,332 seconds** with cached market clouds. The heatmap and full-field sampling comparisons are saved beside the results. The current experiment provides a functioning dense-map prototype and reproducible diagnostics; it does not establish proximity to the continuous optimal policy or profitability.

## Run

The sweep now selects CUDA when available. The measurements above describe the earlier CPU implementation. See [the GPU comparison](oracle-dense-map-gpu-2026-09-14.md) for acceleration and CPU/GPU numerical validation; `--backend cpu` selects the reference implementation.

```powershell
.venv-ml/Scripts/python.exe ml/benchmark_oracle_dense_map.py --output data/backtests/oracle-dense-map-new --horizon 3600 --width 8 --branches 8 --budget 4 --updates 2 --candidates 16
.venv-ml/Scripts/python.exe ml/report_oracle_dense_map.py data/backtests/oracle-dense-map-new
.venv-ml/Scripts/python.exe -m unittest discover -s ml -p test_oracle_account_grid.py -v
.venv-ml/Scripts/python.exe -m unittest discover -s ml -p test_oracle_dense_value_map.py -v
```

The final verification uses `--market-cache data/backtests/oracle-dense-map-hour` to reuse the exact frozen forecast laws from the preliminary run while evaluating the completed account-map kernel. Cache compatibility checks include the root observations, forecast settings, generator hashes and model checkpoint. Source snapshots and hashes are stored with each experiment.
