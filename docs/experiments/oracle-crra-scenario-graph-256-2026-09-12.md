# 256-node NSF scenario graph

Increasing the market graph cap from 64 to 256 nodes per depth improved
compression accuracy and agreement with reference Bellman values, at about
4.6 times the construction runtime. It did not change the first action in the
18 matched account cases tested. These are graph and finite-policy diagnostics,
not completed trading backtests.

The checkpoint, compression metric, 64 conditional samples per parent,
64 consecutive one-second layers, and RTX 3070 are unchanged. The retained
node cap is the variable under comparison. The root initially generates only
64 samples, so the 256-node graph reaches its full width at the second layer.
The benchmark CLI now exposes width, sample count, construction/comparison
selection, and initial exposures. No forecasting or execution kernels changed.

## Construction and rollout projection errors

| Measurement | 64 nodes | 256 nodes |
|---|---:|---:|
| Construction time, 64 future seconds | 2.289 s | 10.506 s |
| Peak PyTorch GPU allocation | 79.5 MiB | 272.5 MiB |
| Stored market nodes, including root | 4,097 | 16,193 |
| Stored sampled edges | 258,112 | 1,019,968 |
| Mean local log-price RMS projection error | 0.2894 bps | 0.2595 bps |
| Mean probability of a projected move changing sign | 23.80% | 17.93% |
| Worst conditional mean-return shift across the rollout | 1.297 bps | 1.761 bps |

Average local price error improves by about 10%. The worst conditional mean
shift does not improve. After the second layer these graphs follow different
projected states, and the 256-node graph exposes four times as many parent
states per layer. The rollout maxima are therefore not paired measurements
of the same input population. They nevertheless show why more nodes alone
do not establish a uniform accuracy or optimality guarantee.

The clustering uses two observed prices in one-basis-point log units and the
model-normalized feature block balanced by its RMS. It retains sampled
representatives and each parent's own conditional transition row. The existing
geometric clustering heuristic is not a nested or value-optimal quantizer.

## Comparisons using identical pre-compression samples

For three origins, generated two transitions with 64 samples per parent.
Width 4,096 is uncompressed; widths 256 and 64 compress the same 4,096 second-layer
samples. All three widths retain all 64 first-layer observations. This isolates
second-layer projection from divergence between longer compressed rollouts.

| Mean second-layer price RMS error across origins | 64 nodes | 256 nodes | Reduction |
|---|---:|---:|---:|
| Decision offsets +1, +2 seconds | 0.3303 bps | 0.2207 bps | 33.2% |
| Decision offsets +900, +1800 seconds | 2.5810 bps | 1.1815 bps | 54.2% |

The mean over origins of the worst parent mean-return shift also improves:
0.3230 to 0.0780 bps for the one-second transitions, and 0.8846 to 0.3592 bps
for the 15-minute transitions. These are errors relative to generated NSF
samples, not forecast errors against realized prices.

## Policy and value comparisons

Tested the same three origins and two schedules, starting from cash, +1 exposure,
and -1 exposure: 18 account cases. Each starts with $10,000 marked wealth and
an empty pending book. The finite action menu uses exposure spacing 0.1,
the existing 25%-notional one-tick limit placements, one pending-order slot,
and cancellation choices. Gamma remains zero, with the same Oracle fees,
leverage, maintenance, order caps, and terminal close. The short terminal
schedules here are evaluation fixtures, not the fixed terminal of a full replay.

| Relative to the uncompressed 4,096-node sampled reference | 64 nodes | 256 nodes |
|---|---:|---:|
| Matching first-action descriptors | 18/18 | 18/18 |
| Mean absolute optimal-wealth value error | $0.05377 | $0.00873 |
| Maximum absolute value error | $0.19955 | $0.03659 |
| Maximum first-action regret under the reference law | $0 | $0 |

Each width selected a nonzero market adjustment in three cases and a limit
placement in six cases. Cash-starting cases remained inactive. This extends
the comparison beyond only cash decisions, but remains a small test set.

For the first-action regret audit, execute the compressed graph's chosen root
action under the uncompressed sampled law, then allow optimal reference
continuation. Thus zero first-action regret does not certify the entire
compressed continuation policy. The value error compares each graph's own
optimal value; it is not realized P&L or an error bound against the true law.

Mean value error falls by about 84%, but no first decision improves because
both widths already select the reference action in every tested case. There
is no established improvement in trading returns, full-interval planning, or
global epsilon-optimality.

## Artifacts and reproduction

- `data/backtests/oracle-nsf-scenario-graph-256/`: 256-node construction and
  initial cash comparisons.
- `data/backtests/oracle-nsf-scenario-graph-64-matched/`: 64-node construction
  using the same benchmark source and parameters.
- `data/backtests/oracle-nsf-scenario-graph-256-accounts/`: 54 solves covering
  the three widths and 18 account cases, plus root-action audits.
- Each directory preserves the source snapshots used for its run.

```powershell
.venv-ml/Scripts/python.exe ml/benchmark_nsf_scenario_graph.py --width 256 --exposures 0,1,-1 --output data/backtests/new-graph-256
.venv-ml/Scripts/python.exe ml/audit_scenario_graph_actions.py --source data/backtests/new-graph-256
.venv-ml/Scripts/python.exe ml/benchmark_nsf_scenario_graph.py --width 64 --construction-only --output data/backtests/new-graph-64
```

The 256 cap applies to market nodes. Account-state search remains uncompressed;
the additional nodes do not make a 64-step or 40-hour Bellman solve bounded by
256 total states per layer. The measurements support using 256 when more
accurate graph values justify its extra construction cost, not treating it as
an already better trading policy.
