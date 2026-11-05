# A fee-aware bound for the omitted continuation

The new bound reduces the one-hour-to-48-hour relative allowance from
**499.0535% to 4.9964% on one matched five-stage sampled law**, about a hundredfold
improvement. It accounts for the inventory liability behind sale proceeds,
original fees, signed-asset funding, limit fills and liquidation.

For the same law and fixed initial cash action, continuous controls have a
relative continuation contribution between **1.7954% and 4.9964%**. Thus 0.1%
is already impossible in this example, even with a tighter upper bound. This
interval concerns expected terminal wealth, not regret of the initial action.

These numbers do **not** replace the earlier bound for an unrestricted
per-second 48-hour process. The calculation covers the sampled law's modeled
decision times. The distinction between those models is retained below.

## What changed mathematically

The cash-only envelope grants up to M dollars of extra terminal wealth at every
decision. A sale can raise quote by M while creating an offsetting short
liability; ignoring that liability made the envelope very loose.

Instead value an account using a nonnegative reference price z:

    S_z(q,a) = q + a z.

For an actual Oracle quote trade dq at price p, with received-amount multiplier
b = 1 - fee, the change in this value is exactly

    buy:   |dq| (b z/p - 1)
    sell:  |dq| (1 - z/(b p)).

Consequently, if c bounds zero and both expressions in parentheses, then

    S_z(after trade) <= S_z(before trade) + |dq| c.

At a decision, apply this to all arrival fills and the market trade together.
Their aggregate absolute quote traffic is at most M, giving at most M c extra
shadow value. The executable calculation uses taker fees at the current mark.
For a non-flat move, every crossed limit price lies between the previous and
current marks. It additionally covers maker buys at the smaller endpoint and
maker sells at the larger endpoint. This deliberately permits both directions
throughout that interval, which is an upper relaxation of the actual fills.
Flat moves add no maker-fill allowance.

Reservations, order slots, minimum sizes, price grids, leverage and position
lifecycles can only remove executions from this relaxation. No omitted control
can exceed the bound simply by choosing a quantity outside the tested action
menu. The terminal closing decision retains its aggregate cap and fees.

For funding multiplier k, choose references satisfying k z_next = z_now. Then

    S_z_next(q, k a) = S_z_now(q,a).

This uses Oracle's original signed-asset funding map. With cumulative funding
F_i from the starting account, the reference at tick i is z/F_i. The numerical
experiment has unit funding, matching the current replay configuration.

Liquidation resets the account to zero. Starting with max(S_z(initial), 0) and
adding nonnegative allowances therefore bounds shadow value through every
liquidation as well. At a valid flat ending, shadow value equals actual cash
for every z. Along any covered complete price path:

    terminal wealth <= B(path,z)
      = max(initial quote + initial asset*z, 0) + sum_i M_i c_i(z/F_i).

The current first action is executed exactly before starting this bound. The
sum includes subsequent decisions and the actual terminal arrival/close.

For each finite sampled path, the implementation chooses the best nonnegative
reference from the breakpoints of this convex piecewise-linear expression.
Every reference gives an upper bound, so a suboptimal reference would remain
safe. z = 0 recovers the cash envelope and is always a candidate: minimizing
cannot worsen that bound on the same execution schedule.

Taking the exact finite expectation of these pathwise upper bounds yields a
bound for every observable policy. Choosing the reference with hindsight is
legitimate here because it tightens an upper bound; it is not an executable
trading policy or an aggregation of clairvoyant trading decisions.

## Matched forecast experiment

The earlier one-hour tree is preserved exactly, including its prices, weights
and decoded observable features. Its offsets are [900, 2250, 3600] seconds.
Two further conditional NSF transitions extend it to [88200, 172800] seconds,
each 84,600 seconds long, within the model's query range. There are five
transitions and 32 equally weighted terminal paths per origin.

The checkpoint and account rules are the same as in the previous experiment:
$10,000 initial cash, gamma zero, exposure entry band [-2,2], maintenance
[-3,3], aggregate quote cap $10,000, minimum $5, price tick 0.01, maker fee
0.0002, taker fee 0.0005, unit funding and one active limit slot. The finite
lower policy uses the current market/limit menu. The upper bound covers all
valid continuous controls at the modeled times.

Unlike the earlier independent horizon queries, this is one consistent finite
law with a shared one-hour prefix. No realized future observations are read.
Origins are 2026-08-10 00:00:01 UTC and 300/900 seconds later; each starts in cash.

| Origin | One-hour finite value | 48-hour finite value, same first action | 48-hour continuous upper | Relative upper |
| --- | ---: | ---: | ---: | ---: |
| 0 s | $10,000 | $10,261.773366 | $10,499.644432 | 4.996444% |
| 300 s | $10,000 | $10,261.726983 | $10,499.512283 | 4.995123% |
| 900 s | $10,000 | $10,261.541055 | $10,499.124975 | 4.991250% |

The continuous one-hour upper bounds are respectively $10,080.779972,
$10,080.744644 and $10,080.595918. For the primary origin this gives

    10000 <= Q_1h_continuous <= 10080.779972
    10261.773366 <= Q_48h_continuous <= 10499.644432

and hence

    1.795430% <= (Q_48h_continuous - Q_1h_continuous) / Q_1h_continuous
              <= 4.996444%.

The lower interval endpoint uses a feasible finite-menu 48-hour policy and an
upper bound on the continuous one-hour value. It does not mistake the finite
one-hour optimum for the continuous optimum. Both use the same fixed first
action, zero trade. The finite-menu optimum itself gains 2.617734% by using
the longer continuation.

Computing the two shadow upper bounds takes about 0.082 seconds at the primary
origin. The exact full finite-menu Bellman solve takes about 2.77 seconds. The
upper-bound calculation does not need that solve; the latter was run separately
to check the result and supply a feasible lower policy. Forecast extension
takes about four seconds on the GPU and is excluded from both timings.

## Comparison with the earlier millions-percent result

The earlier 16,920,299% figure allowed a separate capped decision at each of
the 169,200 omitted physical seconds. For a fair comparison on the new finite
law, there are only two omitted modeled transitions after its exact one-hour
prefix. The old formula on that same schedule is

    (39905.352211 + 2*10000 - 10000) / 10000 = 499.053522%.

The reduction from 499.053522% to 4.996444% comes from the new mathematics.
Reducing the count of physical decisions to modeled stages is a separate model
restriction and must not be described as a proof improvement.

The new theorem also applies to per-second execution, provided its coefficients
cover execution prices at every one of those seconds. The current coarse tree
does not bound price excursions, intermediate fills or other decisions inside
its long jumps. Nor do its sampled extrema cover the unsampled NSF distribution.
Thus no improved numeric certificate for the full per-second, unsampled NSF
problem is claimed. That requires bounds or controlled integration over the
missing market trajectories; it does not require a full account Bellman solve.

In a genuinely narrow fee-covered price band, all the coefficients can be zero,
and cash-starting terminal wealth is bounded by initial cash regardless of
duration. For wider paths the computed coefficients are positive. The method
does not assume universal time decay, a price ceiling from sampled extrema,
or a discounted objective.

## Proof, implementation and reproduction

`TradingTheory/OracleShadowTailBound.lean` proves twelve public results:
trade/fill/rebalance bounds, endpoint interval coverage, funding cancellation,
liquidation-safe capital propagation, and a bound for complete flat-ending
Oracle runs. They are imported into the theory library and axiom audit.

The numerical evaluator uses exact rational arithmetic. Its proof ingredients
are formalized; the TypeScript execution and breakpoint-search implementation
have not themselves been translated into Lean. Forty-four TypeScript tests
pass, including exact Bellman comparisons with funding and root limit fills,
liquidation, zero-gain fee-covered paths and the cash-bound fallback. The Oracle
TypeScript project typecheck passes. The full Lean build and public-result
axiom audit passed; see the saved verification log.

Completed results and provenance are in
`data/backtests/oracle-shadow-tail-20260913-v2/`. The first run lacks the added
continuous-prefix upper/lower interval; its otherwise matching results are
preserved. Logs: `data/backtests/oracle-shadow-tail-{lean,tests,typecheck}.log`.

```powershell
.venv-ml/Scripts/python.exe ml/export_oracle_shadow_tail.py --output data/backtests/new-shadow-tail
node --conditions=development --import tsx scripts/evaluate-oracle-shadow-tail.ts data/backtests/new-shadow-tail
node --conditions=development --import tsx --test packages/bot-algo/test/oracle-shadow-tail.test.ts packages/bot-algo/test/oracle-tail-bound.test.ts packages/bot-algo/test/oracle-execution.test.ts packages/bot-algo/test/oracle-crra-policy.test.ts
node node_modules/typescript/bin/tsc -p packages/bot-algo/tsconfig.oracle.json --noEmit
node scripts/theory-lean.mjs check
```
