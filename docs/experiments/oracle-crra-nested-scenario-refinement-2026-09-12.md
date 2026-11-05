# Nested 64-to-256 scenario refinement

Implemented and tested nested refinement. It provides a stronger local
compression guarantee, but this particular splitter was less accurate and
slower than the existing independent 256-node fitter. Independent clustering
remains the default; nested refinement is available with `--refinement nested`.

The frozen latent NSF checkpoint and price-aware metric are unchanged. The
construction benchmark uses 64 future one-second layers, 64 samples per parent,
and at most 256 market nodes per layer on the RTX 3070.

## Split rule and its guarantee

For each fixed candidate sample batch:

1. Fit the existing 64 coarse representatives and assignments.
2. Select the cell with the largest probability-weighted squared error.
3. Add a sampled representative from that cell, prioritizing weighted error.
4. Move only members of that cell that are strictly closer to the new point.
5. Retain the old representative and repeat until reaching the node cap or
   exhausting distinct points with positive error.

The fine representative list starts with all coarse representatives. Each
fine cell records its coarse ancestor. `NestedPartition` exposes both
assignments, both representative sets, and the ancestor map; `ScenarioGraph`
retains these records in `refinements`.

For sample x with old representative c and proposed new representative c',
the new squared distance is either the old distance or its minimum with the
distance to c'. Therefore, in exact arithmetic,

\[
d_{\rm new}(x)^2\le d_{\rm old}(x)^2.
\]

This holds pointwise and hence for any fixed nonnegative sample weights.
Cells never cross their old boundaries, so mapping a fine assignment through
its ancestor exactly recovers the coarse assignment. This also recovers coarse
conditional edge masses for the same parent/sample batch.

The distance combines two log-price coordinates scaled by one basis point and
the RMS-balanced, training-normalized feature coordinates. Its monotonicity
does not imply monotonicity of every individual coordinate, conditional mean,
tail probability, or Bellman value. A unit test gives an explicit example in
which the combined distance improves while the price coordinate worsens.

## Measurements

| Measurement | Independent 64 | Independent 256 | Nested 64→256 |
|---|---:|---:|---:|
| Generate 64 one-second layers | 2.289 s | 10.506 s | 12.386 s |
| Peak PyTorch GPU allocation | 79.5 MiB | 272.5 MiB | 275.7 MiB |
| Mean local rollout price RMS error | 0.2894 bps | 0.2595 bps | 0.2676 bps |
| Mean absolute optimal-wealth error, matched cases | $0.05377 | $0.00873 | $0.03817 |
| Maximum absolute optimal-wealth error | $0.19955 | $0.03659 | $0.12577 |
| Matching reference first actions | 18/18 | 18/18 | 18/18 |

The nested run retained 16,193 market nodes including the root and 1,019,968
sampled edges. The extra memory includes the recoverable coarse assignments.
Every layer passed the ancestry check, and the largest observed pointwise
increase in the combined metric was zero. Mean probability-weighted squared
metric error on each layer's own candidate population fell from 1.15749 before
splitting to 1.06639 after splitting, a reduction of about 7.9%.

For a paired comparison, all methods were also given the same second-layer
samples, with the first layer left uncompressed:

| Mean second-layer price RMS error | Independent 64 | Independent 256 | Nested 64→256 |
|---|---:|---:|---:|
| Offsets +1, +2 seconds | 0.3303 bps | 0.2207 bps | 0.2932 bps |
| Offsets +900, +1800 seconds | 2.5810 bps | 1.1815 bps | 1.6190 bps |

Nested refinement improves on the coarse representation in these price tests,
but independent 256-node clustering does better. Keeping old representatives
and forbidding reassignment across coarse cells restricts the fit. The new
splitter is a heuristic, so these results do not rule out better nested methods.

## Policy comparison and interpretation

The 18 cases comprise three forecast origins, two two-transition schedules,
and initial exposures 0, +1, and -1, each with $10,000 marked wealth and no
pending orders. The action grid is 0.1; the existing one-slot limit-order menu,
Oracle execution constraints, fees, and gamma zero are unchanged. Width 4,096
is the uncompressed two-transition reference with 64 samples per parent.

All first-action descriptors matched the reference. Each method selected three
nonzero market adjustments and six limit placements. Evaluating the nested
root choice under the reference sampled law with optimal reference continuation
gave zero first-action regret in every case. The saved uncompressed reference
values match those from the independent-clustering experiment exactly at
reported precision.

The guarantee is local to a fixed sample population. At later depths a
256-node rollout generates forecasts from different representatives than a
standalone 64-node rollout. Its locally recoverable coarse projection is not
automatically that earlier standalone graph. Keeping coarse representatives
also does not make the two forecast laws identical.

Similarly, retaining a coarse policy as a candidate prevents a reduction in
the *optimized value* only when candidates are evaluated under the same law
and objective. Comparing the optimal values of two different compressed laws
does not establish that guarantee. The current implementation exposes the
coarse projection; it does not select between laws using an oracle or claim
that the fine law is more accurate in every case.

These are construction and finite-policy diagnostics, not a complete backtest.
The short terminal schedules are test fixtures. Account states are still
uncompressed, and no global approximation certificate or new Lean theorem is
claimed. The earlier suggestion that nesting would be a better construction
should be read as a testable option: it strengthens one property but did not
improve the value estimates relative to independent 256-node clustering here.

## Verification and reproduction

Five new tests cover coarse recovery, pointwise metric improvement, prefix
consistency as nodes are added, duplicate handling, the coordinate-error
counterexample, and access to graph fallback assignments. The six existing
scenario-graph tests and eight GPU execution tests also pass, for 19 tests.
Construction checks the two refinement invariants at every generated layer.

```powershell
.venv-ml/Scripts/python.exe ml/benchmark_nsf_scenario_graph.py --width 256 --refinement nested --exposures 0,1,-1 --output data/backtests/new-nested-graph
.venv-ml/Scripts/python.exe ml/audit_scenario_graph_actions.py --source data/backtests/new-nested-graph
.venv-ml/Scripts/python.exe -m unittest discover -s ml -p test_nested_scenario_graph.py
```

Final artifacts are in `data/backtests/oracle-nsf-scenario-graph-nested-final/`,
including source snapshots, per-layer diagnostics, 54 finite solves, and root
action audits. `oracle-nsf-scenario-graph-nested-256/` is the earlier trial before
the graph API retained the fallback records; its source snapshot is preserved.
