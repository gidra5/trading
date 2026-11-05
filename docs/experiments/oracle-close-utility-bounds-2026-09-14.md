# Closing-utility shortcut experiment

The current GPU Bellman kernel checks next-tick survival and closing utility for every candidate against every sampled next price. This includes repeated funding, marking, order-crossing and logarithm calculations. A separate CPU prototype tests whether some of that work can be replaced by a narrow utility enclosure. The running planner does not use it.

## Conditions and bound

Consider a post-rebalance account with cash `q`, inventory `a`, next-tick funding multiplier `k`, and taker fee `f`. If no pending order crosses anywhere in the finite next-price row, the wealth after a valid complete close is affine in the next price:

\[
X(p)=q+cp,\qquad
c=\begin{cases}(1-f)ka,&a\ge0,\\ka/(1-f),&a<0.\end{cases}
\]

The prototype checks both extreme sampled, tick-rounded prices using the existing arrival kernel. Both must survive maintenance and allow complete closing within the minimum and aggregate maximum quote-size constraints. Any fill at either extreme disables the shortcut. For a fixed pending limit, crossing on an intermediate sampled move implies crossing on the corresponding extreme move, so checking the extremes is sufficient to exclude all fills. With constant funded inventory and cash, raw equity and closing wealth are affine; effective exposure is monotone wherever closing wealth is positive. Endpoint checks therefore cover the interval's maintenance and close-size constraints as well.

For positive closing wealth in `[lo, hi]`, mean `mu` and variance `v`, Taylor's theorem and the zero mean of `X-mu` give

\[
\log\mu-\frac{v}{2lo^2}
\le \mathbb E\log X
\le \log\mu-\frac{v}{2hi^2}.
\]

The expectation here is over the same finite, equally weighted conditional row as the reference calculation. Price rounding happens before its moments are computed. The shortcut returns the lower endpoint only when the interval width is at most the configured tolerance; otherwise the audit uses the full existing calculation. It cannot skip a possible liquidation or replace an infeasible close with a finite value.

Float64 padding is an operation-scale allowance, not directed rounding. The real-arithmetic inequality is established above, but the executable enclosure is not a machine-verified certificate. This distinction must remain explicit if it is integrated.

## Measured candidate coverage and cost

`data/backtests/oracle-close-bounds-candidates/results.json` records 20,000 feasible post-rebalance candidates drawn from the corrected one-hour archived map. Generation includes representative account states, sampled arrivals, fresh catalog orders, retained order subsets and exposure changes. It required 287,450 proposals. These random candidates are not weighted by the actual GPU workload.

| Pending orders | Candidates | Exact finite closes | Shortcut accepted |
| --- | ---: | ---: | ---: |
| 0 | 8,270 | 5,162 | 5,162 |
| 1 | 7,154 | 4,416 | 3,548 |
| 2 | 4,576 | 2,372 | 1,440 |
| Total | 20,000 | 11,950 | 10,150 |

At tolerance `1e-9`, there were no enclosure violations or changes in finite/infeasible closing support. The largest observed finite-value difference was `9.281464485866309e-12` log utility. Median single-threaded CPU time per batch fell from 13.687 ms to 11.153 ms, a 1.227x speedup. Compilation and candidate generation are excluded from those timings. This is a CPU closing-kernel measurement, not a GPU or end-to-end planner speedup.

If a future implementation establishes a uniform per-stage error of `1e-9`, unchanged feasible support and nonexpansive backups would give an accumulated error no greater than `H * 1e-9` against that same finite operator. At 3,600 stages this is `3.6e-6` log value (approximately 0.036 basis points of certainty equivalent). Selecting the best approximate action can add a factor of two to the value-gap bound. These conditional bounds do not cover market sampling, account interpolation, finite action menus, learned-model error or unverified floating-point arithmetic.

## Reproduction and adoption gate

```powershell
.venv-ml/Scripts/python.exe -m unittest discover -s ml -p test_oracle_close_bounds.py -q
.venv-ml/Scripts/python.exe ml/audit_oracle_close_bounds.py --map data/backtests/oracle-dense-repair-terminal-90m/initial --output data/backtests/oracle-close-bounds-candidates-new
```

The CPU prototype is in `ml/oracle_close_bounds.py`. Before adoption it needs an isolated GPU implementation benchmark, comparison against the exact full-field operator including pending fills and terminal feasibility, and explicit accumulation of its approximation allowance in repair diagnostics. The current long replay retains its original operator and hashes.

## Prefer eliminating unused action work first

A subsequent audit identified a candidate optimization that does not approximate closing values. In the current GPU kernel, `_post` runs across the complete power-of-two action vector; `in_menu & feasible` masks its result afterward. Thus padded, unavailable and infeasible action lanes still perform continuation calculations.

`ml/audit_oracle_action_work.py` sampled 5,000 valid account/market/outcome arrivals from the archived map and reproduced the actual future candidate schedule. They produced 320,000 continuation lanes, of which 196,648 were in the menu, 96,885 were feasible, and 49,475 described distinct post-account states. Only **30.28%** of the lanes were feasible; **15.46%** were distinct feasible post-account states. These are counts of arithmetic lanes, not measured runtime fractions.

For the existing post-decision operator, actions reaching exactly the same complete account state at the same market node have identical continuation values. Removing infeasible lanes and retaining one representative of duplicate post-states therefore preserves the candidate maximum. This requires comparing all inventory, cash and pending-order coordinates; matching only target exposure is insufficient. Fees and the aggregate trade constraint are applied before considering candidates equivalent.

GPU compaction has its own cost and may change occupancy. It needs a benchmark and conformance tests before adoption, but it is the next optimization to test before adding the utility enclosure's numerical approximation. The current replay remains unchanged.

The standalone prototype is now in `ml/oracle_gpu_action_compaction.py`. It sorts hashed states, verifies all coordinates before dropping an adjacent duplicate, and evaluates the smallest supported power-of-two prefix containing the retained actions. Hash collisions may retain extra work; they cannot justify dropping unequal states. Padding the original vector and masking after scoring would not remove the expensive computation, hence the explicit smaller-vector dispatch.

Four Triton-interpreter CPU tests pass, checking empty menus, single feasible candidates, duplicate states, forced hash collisions, signed zero and pending-order coordinate changes. They also verify that compaction preserves a state-dependent scoring maximum. Offline CUDA compilation for sm_86 succeeds for both the original and compacted versions of the actual Oracle post-state function, including its ordinary and final-decision forms. This confirms compilation, not GPU execution correctness or speed. Compilation records are in `data/backtests/oracle-action-compaction-compile-v3/results.json`.

The isolated GPU tests and benchmark have completed. All four GPU tests passed. On 2,000 sampled tasks, full and compacted scoring agreed exactly, and the checked CPU reference subset also had zero error. Ordinary post-state scoring had a median speedup of **0.9981x** (effectively unchanged). Final-decision scoring had a median speedup of **0.3230x**, with substantial timing variation. These timings include compaction but exclude initial action construction. The prototype is **not enabled** in the planner: the reduced lane count did not produce a measured speed benefit. Results are in `data/backtests/oracle-action-compaction-gpu/results.json`.

```powershell
.venv-ml/Scripts/python.exe ml/audit_oracle_action_work.py --map data/backtests/oracle-dense-repair-terminal-90m/initial --output data/backtests/oracle-action-lane-audit-new
.venv-ml/Scripts/python.exe ml/benchmark_oracle_action_compaction.py --map data/backtests/oracle-dense-repair-terminal-90m/initial --output data/backtests/oracle-action-compaction-gpu-new
```
