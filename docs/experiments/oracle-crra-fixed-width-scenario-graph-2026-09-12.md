# Fixed-width NSF market scenario graph

Implemented a scenario graph with at most 64 market nodes per future depth.
On the RTX 3070, constructing 64 consecutive one-second layers took 2.338
seconds, stored 4,097 market nodes including the root and 258,112 sampled
edges, and used about 80 MiB of peak PyTorch GPU allocations. This measures
market graph construction, not a 64-step Bellman solve or a trading backtest.

The sampled-skip latent NSF checkpoint is unchanged:
`80fdc3df78a5f60d9980b236f9bbaa296904dc562bd64195394831120dde01db`.

## Compression procedure

Each market node carries the two observed feature rows and prices used by the
current learned NSF kernel, plus their times. It does not carry a sampled
latent variable or an unobservable scenario identifier.

At every depth:

1. Generate 64 conditional next observations from each current representative.
2. Fit at most 64 representatives to the resulting joint observation pairs.
3. Assign every sample to a representative, preserving its probability mass.
4. Keep each parent's own outgoing probabilities and destination indices.
5. Use the representatives as the forecast inputs for the next depth.

The fitter uses weighted Lloyd clustering and selects an actual sample near
each center. Selecting samples preserves the price tick grid and the sampled
joint feature coordinates. These are still synthetic NSF observations; they
need not satisfy all identities of production market features. Exact duplicate
coordinates share a representative even when the node cap is not binding.

Fitting weights are marginal node probabilities, used to allocate representation
capacity. They do not replace conditional transition rows. Every parent keeps
64 equally weighted sampled edges; several can point to the same representative.
Equivalently, transition probability from parent i to representative j is the
number of that parent's samples assigned to j, divided by 64. No sample mass
is dropped, but its price and feature values can change under the projection.

The metric uses the two log prices in units of one basis point. The 118 feature
coordinates use the model's training normalization, with the entire feature
block divided by sqrt(118). Thus many feature coordinates do not overwhelm
small economically relevant price differences. This metric is a heuristic,
not a proven value-preserving distance.

## What remains uncompressed

The GPU Bellman evaluator now accepts graph destination indices. Its value is
still a function of both market node and full account state:

\[
V_t(m,s)=\max_{a\in A_t(m,s)}
\sum_b w_t(m,b)V_{t+1}(d_t(m,b),T_t(s,a,p_t(m,b))).
\]

Here s includes cash, inventory, prices needed for execution, and pending
orders. T applies the existing Oracle arrival/action/marking/liquidation rules.
Feasibility includes having a valid continuation under every sampled branch.

Two paths can reach the same market representative with different holdings or
orders. Their account values are not interchangeable. Market compression gives
O(D K) market-node storage and O(D K B) sampled-edge storage for depth D, width K,
and samples B. The current account search can still expand exponentially with
actions and depth. It does not yet memoize or approximate the full account value
function. Bounding the entire planner requires an additional approximation over
account states or continuation values.

Even with a bounded layer width, storing every depth remains linear in the
horizon. The benchmark does not establish memory or runtime feasibility for a
complete 40-hour one-second graph on this GPU.

## Compression measurements

For 64 one-second layers and width 64, local compression errors measured against
the samples generated immediately before each projection were:

| Diagnostic | Result |
|---|---:|
| Mean over layers of probability-weighted log-price RMS error | 0.292 bps |
| Largest absolute conditional mean-return shift, over parents and layers | 1.049 bps |
| Maximum retained nodes at any layer | 64 |

These are local projection errors within an already approximated rollout.
They are not errors against the true market law, cumulative path-distribution
bounds, or bounds on policy regret. Small price errors can cross an order or
liquidation threshold. The final diagnostics also record changes in move sign;
roughly a quarter of the very small sampled one-second moves change sign under
projection. A small RMS error alone does not establish economic sufficiency.

An initial feature-dominated metric used the NSF decoder's 100-bps log-price
scale without balancing the feature block. Its mean local price error was
1.43 bps and its worst conditional mean shift was 7.85 bps. Giving prices more
weight materially reduced those errors. It did not eliminate them.

## Small decision comparisons

Compared three archived forecast origins and two two-transition schedules:
[1,2] seconds and [900,1800] seconds. Each parent generated four samples, so
width 16 was uncompressed. Root and continuation exposure spacing was 0.1,
with the existing one-slot limit menu and gamma zero. Tested widths 16,8,4,1.
These short terminal horizons are compression diagnostics, not replacements
for the fixed whole-backtest terminal objective.

For each compressed root choice, an additional audit evaluated that first
action under the uncompressed sampled law, allowing optimal continuation under
that reference law. Widths 8 and 4 retained the reference root action in these
six small cases. Width 1 changed the root action in two cases, losing $7.95 and
$11.84 of expected terminal wealth relative to the $10,000 reference optimum.
All audited root choices were feasible under the reference law.

This directly demonstrates that aggressive compression can change the optimal
decision. The six reference cases all chose cash initially, so this is a limited
test set, not evidence of broad policy preservation. Synthetic tests separately
exercise profitable contingent decisions and verify that conditional rows are
not replaced by marginal distributions.

## Verification and scope

Six scenario-graph tests pass: bounded width and mass conservation; safe
duplicate handling; equality of graph Bellman values with its explicitly
unrolled graph; preservation of conditional-information distinctions; bounded
NSF generation; and exact agreement with the existing NSF tree when no
compression occurs. The eight earlier GPU execution tests also pass. After
adding graph support, the existing 36 rational reference cases still agreed
within $1.82e-12 in expected wealth.

The unrolled-graph equality verifies evaluation of the compressed model. It
does not identify that model with the uncompressed NSF law. Exact merging would
require equivalent admissible decisions, payoffs, and transitions into future
equivalence classes. Geometric similarity of feature vectors does not establish
those conditions. No new Lean optimality or approximation certificate is claimed.

Source files:

- `ml/nsf_scenario_graph.py`: representative fitting and graph generation.
- `ml/gpu_oracle_bellman.py`: destination-index graph evaluation.
- `ml/test_nsf_scenario_graph.py`: graph and compression tests.
- `ml/benchmark_nsf_scenario_graph.py`: construction and decision measurements.
- `ml/audit_scenario_graph_actions.py`: root choices evaluated under the reference law.

Final measurements, diagnostics, source snapshots, and action audits are in
`data/backtests/oracle-nsf-scenario-graph-final/`. The earlier
`oracle-nsf-scenario-graph/` directory records the feature-dominated metric;
`oracle-nsf-scenario-graph-price-aware/` records the intermediate price-aware
version before the duplicate-observation guard was extended to uncompressed
layers. Those directories retain their original source snapshots.

```powershell
.venv-ml/Scripts/python.exe ml/benchmark_nsf_scenario_graph.py --depth 64 --output data/backtests/new-scenario-graph
.venv-ml/Scripts/python.exe ml/audit_scenario_graph_actions.py --source data/backtests/new-scenario-graph
.venv-ml/Scripts/python.exe -m unittest discover -s ml -p test_nsf_scenario_graph.py
```
