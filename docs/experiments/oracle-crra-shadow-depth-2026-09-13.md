# Tighter continuation bounds and the 0.1% cutoff search

On the unchanged five-stage sampled NSF law, the one-hour-to-48-hour relative
contribution now has the continuous-control interval **2.6177%–4.6604%** at the
primary origin. The previous interval was 1.7954%–4.9964%. Bellman refinement
tightens the upper bound to **3.4823% for the declared finite action menu**.

The earliest certified checkpoint with **epsilon < 0.1% is 48 hours (172,800
seconds; five modeled transitions)** at all three origins. Every earlier
checkpoint is rejected by a lower bound, including 24.5 hours, where the
continuous contribution is at least 0.893%. At 48 hours the omitted contribution
is exactly zero because the cutoff equals the full horizon.

This is a complete search of the existing forecast checkpoints, not a claim
about the earliest physical second between them. The experiment does not
certify the unsampled NSF law or execution at omitted per-second timestamps.

## Quantity and unchanged experiment

For the fixed current action a and T = 48 hours, we evaluate

    epsilon_H = (Q_T(a) - Q_H(a)) / Q_H(a).

Q is expected terminal wealth, with gamma = 0 and a valid flat ending at each
evaluated horizon. The solver stores utility wealth minus one; the evaluator
adds one before forming these wealth ratios. This quantity is the contribution
of extending the horizon, not the regret of the current action or numerical
slack in an upper bound.

The initial action is selected by the one-hour finite policy and then held
fixed across every cutoff. It is zero trade with no orders at all three origins.
The remaining policy is reoptimized for each cutoff. Flatness requires an
actual valid terminal close; an uncloseable residual makes that branch infeasible.

Inputs are frozen from `data/backtests/oracle-shadow-tail-20260913-v2/`:

- Checkpoints: 900, 2250, 3600, 88200, 172800 seconds (32 terminal paths).
- Forecast SHA256: `28db55bd43d205ace04d7a044b02051e6a502fac58637dbffb3def0c3cf2434f`.
- NSF checkpoint SHA256: `80fdc3df78a5f60d9980b236f9bbaa296904dc562bd64195394831120dde01db`.
- Initial cash $10,000; entry exposure [-2,2]; maintenance [-3,3].
- Aggregate quote cap $10,000 per decision; order minimum $5; price tick 0.01.
- Taker fee 0.0005; maker fee 0.0002; unit signed-asset funding; one limit slot.
- Finite menu: targets [-2,-1,0,1,2], zero trade, cancellation, and passive
  limits for 25% of wealth one tick away; a decision uses market trade or placement.

Origins are 2026-08-10 00:00:01 UTC and 300/900 seconds later. Recomputed
48-hour finite values are checked for exact equality with the previous report.
Forecasts at future realized observations are not read.

## Results

Primary origin, with the same current action at every cutoff:

| Cutoff H | Modeled transitions | Finite Q_H | Continuous epsilon lower | Continuous epsilon upper | Refined finite-menu epsilon upper |
| --- | ---: | ---: | ---: | ---: | ---: |
| 15 minutes | 1 | $10,000.000000 | 2.617734% | 4.660402% | 3.482259% |
| 37 minutes 30 seconds | 2 | $10,000.000000 | 2.617734% | 4.660402% | 3.482259% |
| 1 hour | 3 | $10,000.000000 | 2.617734% | 4.660402% | 3.482259% |
| 24 hours 30 minutes | 4 | $10,048.935940 | 0.892958% | 4.150730% | 2.978325% |
| 48 hours | 5 | $10,261.773366 | 0% | 0% | 0% |

The exact finite-menu contributions are 2.617734% at one hour and 2.118010%
at 24.5 hours. These exact comparisons use a full finite Bellman solve for
verification; the shadow upper-bound algorithms do not require that solve.

| Origin | Previous continuous 1h upper | New continuous 1h upper | Refined finite 1h upper | First checkpoint below 0.1% |
| --- | ---: | ---: | ---: | --- |
| 0 s | 4.996444% | 4.660402% | 3.482259% | 48 h |
| 300 s | 4.995123% | 4.659375% | 3.481751% | 48 h |
| 900 s | 4.991250% | 4.656339% | 3.479761% | 48 h |

In particular, failing the 0.1% target at 24.5 hours is not merely a loose
upper-bound calculation: a feasible longer policy exceeds even the continuous
shorter-horizon upper by more than the threshold. The search does not assume
monotonicity of epsilon. At T the zero result uses the identity Q_T/Q_T - 1 = 0,
not a claim that the numerical upper and lower value bounds have converged.

## Improvements to the bound

### Directional fill coverage

For shadow capital S_z(q,a) = q + a*z, received-amount trade fees give gains
per absolute quote flow

    buy:  (1-fee)*z/p - 1
    sell: 1 - z/((1-fee)*p).

The older upper relaxation allowed maker buys and sells anywhere inside a
crossed price interval. Oracle only fills buys on a downward move and sells
on an upward move. Moreover, a crossed buy executes at or above the new mark,
and a crossed sell at or below it. For z >= 0, replacing either execution
price by the new mark can only increase its shadow gain. Therefore each
frame now covers only the eligible maker direction at that mark, plus both
market directions and zero. All fills and trades still share ONE quote cap.

This changes the primary 48-hour continuous upper from $10,499.644432 to
$10,466.040160. It retains arbitrary valid continuous quantities and limit
prices at the modeled timestamps; no finite-menu coverage assumption is used.

### Conditional shadow prices

A second bound chooses terminal reference z_T = p_T and backs up

    z_parent = funding * E[z_child | current observable node].

Inventory after the current rebalance is chosen before observing the child.
Thus its expected funded shadow value equals its current shadow value.
Trading adds at most M*c at each node, where c covers the eligible trade
coefficients above. Summing expected node charges supplies an upper bound.

Liquidation needs an additional guard, rather than following from normalized
probabilities alone. Let L be the maximum of one and the absolute entry and
maintenance bounds. Every feasible post-rebalance account has

    W >= 0,  |a*p| <= L*W.

Physical trades respect the entry band. Without a physical trade, the
previously surviving account is preserved; Oracle's effective maintenance
band implies the stated raw bound when L >= 1. Absorbing ruin also satisfies it.
The implementation checks the initial account and, at every edge, requires

    L * |funding*z_child/p_parent - 1| <= 1.

Then the funded shadow value is

    W + a*p_parent*(funding*z_child/p_parent - 1) >= 0.

Consequently zero-account liquidation cannot increase it. If any edge or
initial leverage condition fails, this candidate is unavailable; the evaluator
falls back to the pathwise bound. It never substitutes zero for a failed bound.

At the primary origin the maximum scaled edge move is 0.116383, below one.
The full-horizon conditional candidate is $11,191.909216, so the pathwise
bound is tighter there. For the one-hour prefix, however, the conditional
candidate is exactly $10,000. Staying cash attains it, proving that the
continuous Q_1h is $10,000 and improving the relative contribution's lower bound.

Given feasible lower values L_H,L_T and continuous upper values U_H,U_T,
the reported interval is

    max(0, L_T/U_H - 1) <= epsilon_H <= U_T/L_H - 1.

All denominators here are positive. In this experiment, a flat-ending shorter
policy can stay cash through the additional stages, so the contribution is
nonnegative. This also explains the zero clipping in the lower endpoint.

### Finite-menu Bellman refinement

At a nonterminal node, start with the smaller continuous shadow bound B_0.
Refinement over the declared finite menu takes

    B_(d+1)(s) = min(B_0(s), max_a E[B_d(next(s,a))]).

True leaves always use Oracle's actual valid terminal close, with infeasibility
represented as bottom. Each refinement remains above that finite-menu optimum
and cannot increase the preceding bound. Sufficient depth gives the exact
finite-menu value. This refinement does not give an upper bound for continuous
prefix actions outside the declared menu.

| Evaluated account depth d | Primary 48h finite upper | Numerical gap above exact finite value |
| --- | ---: | ---: |
| 0 | $10,466.040160 | 1.990560% |
| 1 | $10,466.040160 | 1.990560% |
| 2 | $10,440.688939 | 1.743515% |
| 3 | $10,433.929373 | 1.677644% |
| 4 | $10,348.225934 | 0.842472% |

Depth d is numerical account-search depth, separate from the cutoff H used in
epsilon_H. The 0.842472% numerical gap must not be reported as the omitted
continuation contribution. The depth-four bound takes about 2.9 seconds at
the primary origin, excluding forecasting and the verification solve.

## Verification and artifacts

`OracleShadowTailBound.lean` now proves the tighter directional crossed-fill
coverage against the actual Oracle fill machine. The new
`OracleMartingaleShadowBound.lean` proves raw leverage bounds from entry and
effective maintenance constraints, guarded nonnegative funded shadow value,
liquidation nonincrease, the conditional expected-shadow inequality, and
intersection of valid bounds. Their public results are imported and axiom
audited. These are formal proof ingredients for the numerical algorithm;
the TypeScript tree traversal and finite solver are not extracted Lean code.

All 49 relevant TypeScript tests pass, including comparisons against exact
Bellman values with pending fills, signed funding and short positions;
liquidation-guard fallback; nested refinement; and infeasible terminal dust.
The Oracle TypeScript typecheck and complete Lean build/axiom audit pass.

Results with exact rational values and source hashes:
`data/backtests/oracle-shadow-depth-20260913-v2/{plan,report}.json`.
The preceding run is preserved separately. Verification logs:
`data/backtests/oracle-shadow-depth-tests-all.log` and
`data/backtests/oracle-shadow-depth-lean-v3.log`.

Reproduce from the frozen input into a new output directory:

```powershell
node --conditions=development --import tsx scripts/evaluate-oracle-shadow-depth.ts data/backtests/oracle-shadow-tail-20260913-v2 data/backtests/new-shadow-depth
node --conditions=development --import tsx --test packages/bot-algo/test/oracle-shadow-tail.test.ts packages/bot-algo/test/oracle-tail-bound.test.ts packages/bot-algo/test/oracle-execution.test.ts packages/bot-algo/test/oracle-crra-policy.test.ts
node node_modules/typescript/bin/tsc -p packages/bot-algo/tsconfig.oracle.json --noEmit
node scripts/theory-lean.mjs check
```

The earlier 16,920,299% per-second envelope counted a capped trade at every
omitted second. It concerns a larger execution model. This experiment's
4.6604% result cannot replace that certificate without also covering the
omitted timestamps and the unsampled market law.
