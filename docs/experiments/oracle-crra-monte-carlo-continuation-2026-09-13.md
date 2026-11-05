# Monte Carlo continuation updates

**Plain Monte Carlo with eight draws per conditional node met the 0.1%
accuracy target in this study.** Across three seeds, its maximum continuous
value-error bound was **0.060532%**, its continuous action-loss bound was
**0.047050%**, and it matched **99–100%** of the 100 tested actions. It is now
the default estimator; the paired correction remains an explicit comparison
option.

The plain update averaged **91.97 seconds**, versus **157.95 seconds** for the
full 64-outcome calculation: approximately **1.72× faster**. This remains too
slow for a full per-second historical replay. The implementation still visits
every graph node.

This experiment estimates the change to the cached continuation when the
one-hour finite forecast graph is extended by one second. It tests sample
counts against a complete backward calculation, retaining the existing Oracle
fees, funding factor 1, maintenance/liquidation rules, aggregate order cap,
one pending limit slot and actual flat terminal requirement in action scoring.

## Estimator

The cached market graph has 64 equally weighted outcomes at each node. Plain
Monte Carlo draws a uniform subset without replacement and performs a backward
backup under that empirical conditional law. It reuses the market graph but
recomputes the value functions. Using all 64 outcomes recovers the full law.

The optional paired estimator instead uses the same subset for old and new
value computations:

    raw_planes = cached_full_old
               + sampled_backup(estimated_new_child)
               - sampled_backup(cached_old_child).

Then scale both coordinates of each row together so that its cash value is

    full_old_cash + sampled_new_cash - sampled_old_cash.

Taking the minimum of componentwise-corrected planes does not otherwise
preserve this scalar control variate when minimizing faces change. This
centering preserves the raw correction's exposure shape; that shape and the
multistep nonlinear estimate still require empirical accuracy checks.

The extra terminal layer is evaluated with all outcomes. Corrections are
propagated backwards through every one-second layer. With all 64 outcomes the
algorithm recovers the full backward result. With identical child inputs the
correction is exactly zero. Smaller sample counts share nested permutation
prefixes; independent seed runs use independent permutations.

The paired control-variate construction aims to avoid replacing a
well-computed value by an independently noisy estimate of its entire magnitude.
The existing backup also chooses support faces and fee closures. Those
nonlinear operations mean the complete estimator is **not claimed unbiased**.
Its corrected planes are numerical estimates, **not certified upper bounds**.
Any loss of the fee wedge is counted, not silently projected away.

The initial uncentered experiment was stopped after its first completed
one-sample trial showed 16.37% maximum value error despite only 0.00211%
maximum action loss. Its frozen sources and partial results remain in
`data/backtests/oracle-mc-continuation-test-hour0`. The centered experiment
reuses the unaffected complete reference from that directory.

No sample identity is visible to the policy. Action values are averaged before
choosing the action. The action comparison itself uses all 64 outcomes, so this
study isolates errors in continuation updates rather than adding another noisy
root-action search.

## Predeclared accuracy checks

- Baseline graph: first test hour, 256 nodes per layer, 3,600 one-second steps.
- Updated endpoint: 3,601 steps from the old origin, allowing 3,600 steps after
  advancing one existing graph step.
- Continuation support: 65 points, the same rational order-aware construction.
- Samples per conditional node: 1, 4 and 16; three independent seeds each.
- Reference: all 64 outcomes, additionally checked against the independent
  complete calculation from the incremental-update experiment.
- Comparison: also estimate an entire value using one sample without the
  paired correction, and retain the original cache without updating it.
- Accuracy target: maximum measured action loss below 0.1% relative CE.

Action checks use the current 0.01 exposure menu, zero trade, cancellation and
limit placements at 1/10/100 ticks with 0.1/0.25/0.5/1 equity fractions. Existing
restrictions on simultaneous market execution and limit placement remain.

At depths 0, 600, 1,800 and 3,000, the benchmark selects low, middle and high
representative prices (only the single root at depth zero). Each has ten
predeclared account cases: cash, positive/negative exposure, near-cap exposure,
pending buy/sell orders, and three account sizes. This gives 100 action cases.
Continuation errors are also evaluated at all market nodes of the corresponding
child layers on 41 exposures from -2 to 2. These probes are not a formal uniform
bound between grid points.

For an estimated selected action `a_hat`, the measured loss is

    100 * (1 - exp(Q_reference(a_hat) - max_a Q_reference(a))).

The reference uses the existing approximate plane continuation, so this is
loss relative to that controller's action-value scores. It is not a claim of
0.1% error relative to the true optimal NSF policy or realized-market wealth.
Exact action agreement is reported separately: changing between nearly tied
actions need not entail a material value loss.

## Continuous exposure check

An additional audit compares the reference and estimated log-value curves at
the union of their support-envelope breakpoints. Between consecutive
breakpoints, the ratio of their positive affine wealth functions is monotone
or constant, so no exposure grid is needed to find its extrema.

The audit subtracts the value error at a fixed cash reference account separately
for each market node. This isolates the part of the error that can affect
actions. Let that market-only shift be `g(m)` and suppose

    |V_est(s,m) - V_ref(s,m) - g(m)| <= epsilon.

Because the market kernel is exogenous and shared by all actions, `E[g(m')]`
is a common shift of every action value. The two-error argmax argument gives

    Q_ref(best) - Q_ref(selected) <= 2*epsilon
    relative CE loss <= 1 - exp(-2*epsilon).

This requires identical feasible actions, finite positive continuation values,
and continuation exposures inside the checked interval. The audit evaluates
all feasible benchmark-action outcomes to verify that interval premise.
It reports both the graph's previously derived reachable interval
`[-2.017, 2.017]` and the broader maintenance interval `[-3,3]`.

These are float64 numerical bounds on differences between stored continuation
functions, at all represented market nodes of the four declared child depths.
They are not new Lean certificates, high-probability statements about other
random seeds, or bounds on the true optimal NSF policy.

## Results

Worst results across three independent seeds, using the reachable exposure
interval for the continuous bounds:

| Estimator | Draws / node | Value-error bound | Action-loss bound | Tested action agreement |
| --- | ---: | ---: | ---: | ---: |
| Paired | 1 | 5.345225% | 0.184265% | 82–94% |
| Paired | 4 | 0.305946% | 0.106632% | 93–96% |
| Paired | 8 | 0.077596% | 0.084409% | 92–98% |
| Paired | 16 | 0.043496% | 0.060720% | 97–99% |
| **Plain** | **8** | **0.060532%** | **0.047050%** | **99–100%** |

The largest observed action loss for plain eight-draw MC was **0.000215%**.
The continuous bound is larger because it covers every represented market
node and every exposure in the interval, including untested account cases.
The domain audit checked **1,760,640** feasible one-step outcomes; exposures
were between **-2.000718 and 2.000248**, inside the checked interval.

Plain eight-draw updates took **90.76, 93.08 and 92.06 seconds** in isolated
runs. An isolated paired eight-draw update took **188.36 seconds**. It does not
provide a cold-start advantage over the **157.95-second** full calculation.
Paired warm-cache times remain estimates, not implemented rolling timings.

Eight draws are sufficient in this study, not a proved universal minimum.
Plain four-draw updates were not tested; the four-draw failures above concern
the paired estimator. Individual action losses do not have to decrease
monotonically with sample count.

The unchanged cache was better than either sampling method for this particular
one-second extension: **all 100 actions matched**, its value-error bound was
**0.007711%**, and its action-loss bound was **0.001101%**. This is an offline
comparison using the full reference. It does not yet supply a cheap online
certificate that decides when cache reuse is safe at a new forecast origin.

All **62 related tests pass**, including recovery of the complete empirical
law by the default plain estimator, full-census equivalence, cache immutability,
sampling nesting, and continuous-envelope error checks.

![Monte Carlo continuation accuracy](E:/Projects/trading/data/backtests/oracle-mc-continuation-comparison/accuracy.png)

The connected curves show the paired estimator; the green diamond shows the
plain eight-draw action-loss bound. The dotted line is the unchanged-cache
bound. [Combined data and provenance](E:/Projects/trading/data/backtests/oracle-mc-continuation-comparison/summary.json).

## Runtime scope

The sample count is **per node**, not a count of complete rollout paths. The
current implementation still visits roughly 921,000 conditional nodes during
an update. It reduces work inside each conditional backup; it does not yet
implement a sparse path-only root estimator.

For the paired estimator, cold runtime includes computing both sampled backups. The old sampled backup
could be cached while its state, child values, sample selection and operator
remain unchanged. Reports separately time that component and give the remaining
time as a potential warm-update estimate. That estimate excludes the additional
cache storage/read costs and is not a measured end-to-end rolling runtime.

The experiment does not rebuild the original market graph, train a model,
refresh graph coverage for new observations, or run a historical trading
backtest. Original fixed-graph sampling/quantization and policy-approximation
errors remain outside this sampling-error study.

An eight-draw follow-up uses the same reference, seeds and accounts. It runs in
parallel with the later sixteen-draw trials on the eight-core host; their
timings include this shared workload and are not isolated latency measurements.

## Artifacts and reproduction

Implementation: `ml/oracle_mc_continuation.py`.
Runner: `ml/benchmark_oracle_mc_continuation.py`.
Tests: `ml/test_oracle_mc_continuation.py`.
Continuous interval audit: `ml/audit_oracle_mc_continuation.py` and
`ml/oracle_mc_error.py`, with tests in `ml/test_oracle_mc_error.py`.

Current CLI for the selected estimator:

    .venv-ml/Scripts/python.exe ml/benchmark_oracle_mc_continuation.py --graph data/backtests/oracle-accuracy-graph-test-hour0 --value data/backtests/oracle-accuracy-value-test-hour0 --extension data/backtests/oracle-incremental-value-test-hour0 --output data/backtests/oracle-mc-plain-repeat --reference data/backtests/oracle-mc-continuation-test-hour0 --estimator plain --samples 8 --seeds 20260930 20260931 20260932

Use `--estimator paired` for paired corrections and `--include-comparison` to
include one opposite-estimator comparison. Historical plans retain the earlier
CLI flags; their frozen `source/benchmark_oracle_mc_continuation.py` runners
remain available for exact reproduction.

Use a new output directory for a repeat. The directory contains frozen sources
and hashes, the full-reference sampled layers, account cases, every trial's
estimated layers, per-depth diagnostics, action comparisons and summary report.

Artifacts include:

- `oracle-mc-continuation-centered-test-hour0`: paired 1/4/16, three seeds.
- `oracle-mc-continuation-eight-test-hour0`: paired eight, three seeds.
- `oracle-mc-continuation-eight-isolated-test-hour0`: isolated paired/plain
  eight, seed 20260930.
- `oracle-mc-continuation-plain-eight-test-hour0`: plain eight, the other seeds.
- `oracle-mc-continuation-comparison`: combined JSON, table and figure.

These directories are under `data/backtests/`. This work did not change the
live bot, forecast checkpoint or earlier historical backtest results.
