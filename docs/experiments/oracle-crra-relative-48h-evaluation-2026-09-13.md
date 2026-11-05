# Relative continuation bound: one hour within 48 hours

The evaluated one-hour wealth is $10,000 at all three tested forecast origins.
At the first origin, the computed cash-cap envelope is **16,920,299.0535%** of
that wealth. This envelope cannot certify a useful early cutoff at a 0.1%
tolerance. It is an upper bound calculation, not a measurement of actual
truncation error.

## Model and calculation scope

This uses the configuration of the latest complete, flat-ending NSF replay:

- Sampled-skip latent NSF checkpoint, SHA-256
  `80fdc3df78a5f60d9980b236f9bbaa296904dc562bd64195394831120dde01db`.
- Initial cash $10,000, no holdings or pending orders; gamma zero.
- Entry exposure [-2, 2], maintenance [-3, 3], reserve leverage 2.
- Taker fee 0.0005, maker fee 0.0002, funding multiplier 1.
- Quote minimum $5, aggregate maximum $10,000 per decision, price tick 0.01.
- One limit slot; placement notional 25% of wealth, one tick from current price.
- Market targets [-2, -1, 0, 1, 2], zero trade, cancellations and limit placements.
  The replay menu chooses market adjustment or placement at each decision.
- Two sampled conditional branches and three hypothetical trading stages per
  horizon query. At one hour the price offsets are [900, 2250, 3600] seconds.
  Decisions occur before these transitions, with a valid flat close at the end.

The primary forecast origin is 2026-08-10 00:00:01 UTC. Two additional origins
are 300 and 900 seconds later, each evaluated with a fresh cash account rather
than an account carried forward from a replay. Fresh hypothetical forecasts
were generated on the GPU; no realized future features were used. Account
Bellman calculations use exact rational arithmetic on the quantized finite law.

The later 256-node graph work is a separate compression benchmark, not the
latest complete replay. This experiment does not establish an upper certificate
for unrestricted continuous actions, missing intermediate decision times, or
the unsampled NSF law. It also is not a new 48-hour realized backtest.

## One-hour result

For the action selected by the short solve, calculate both:

- Q_H: expected wealth with a valid flat close at H. Gamma-zero utility is
  wealth minus one, so one is added back to the solver's utility value.
- U_H: a second Bellman backup through the same prefix, replacing terminal
  wealth by max(frontier quote, 0) + M. It fixes the same first action and
  optimizes subsequent prefix actions, permitting open inventory at the cutoff.

The omitted tail is counted conservatively as one capped decision per physical
second. With T = 172800 seconds, H = 3600 seconds, and M = $10,000:

    r = T - H = 169200
    r M = $1,692,000,000
    epsilon_bound = [U_H - Q_H + r M] / Q_H.

| Origin offset | Selected first trade | Q_1h | U_1h | Relative envelope, percent |
| --- | --- | ---: | ---: | ---: |
| 0 s | Zero | $10,000 | $39,905.352211 | 16,920,299.053522% |
| 300 s | Zero | $10,000 | $39,905.355667 | 16,920,299.053557% |
| 900 s | Zero | $10,000 | $39,905.574138 | 16,920,299.055741% |

The primary result is

    epsilon_bound = (29905.35221102604 + 1692000000) / 10000
                  = 169202.99053522112
                  = 16,920,299.053522112%.

The $10,000 denominator here is the computed finite planner value, not an
assumed substitute for that value. The upper calculation is loose: it rewards
raising quote through sales without imposing the final cost of buying back
the liability at the cutoff. Future fees, leverage and liquidation remain in
the prefix transitions; the tail envelope mostly uses the quote-flow cap.

The corresponding formal comparison requires the full problem to have these
same prefix laws, decisions and menus, with the tail covered by the cash cap.
It cannot cover additional trades or fills in skipped parts of the prefix.
For the unrestricted per-second Oracle model, a separately valid but looser
bound follows from the root cash theorem and the feasible cash policy:

    Q_48h <= 10000 + (172800 + 1) * 10000 = $1,728,020,000
    Q_1h(first action cash) >= $10,000
    relative contribution <= 172801 = 17,280,100%.

This last comparison applies independently of the forecast approximation,
under the stated uniform aggregate cap and original funding model.

## Depth required for an envelope below 0.1%

These are the primary origin's other queried horizons:

| Evaluated horizon | Computed Q_H | Cash envelope, percent |
| --- | ---: | ---: |
| 1 h | $10,000.000000 | 16,920,299.053522% |
| 6 h | $10,005.306489 | 15,112,278.079525% |
| 24 h | $10,081.066088 | 8,570,813.274474% |
| 47 h 57 min 8 s | $10,231.159561 | 17,004.610573% |
| 47 h 59 min 59 s | $10,231.555256 | 290.947407% |
| 48 h | $10,231.556747 | 193.210513% |

At **48 hours**, the omitted continuation is empty, so its contribution is
**exactly zero**. That identity uses the actual terminal contract; the loose
cash envelope in the last row still includes its terminal relaxation gap.
Neither result eliminates action-grid, sampled-law or skipped-tick errors.

There is also an analytical reason no partial whole-second horizon can pass
this particular envelope for the current three-stage query family. Its prefix
wealth is at most initial cash + (3 + 1) M = $50,000. Its prefix relaxation gap
is nonnegative. Thus, whenever at least one second remains,

    epsilon_bound >= M / Q_H >= 10000 / 50000 = 20% > 0.1%.

Consequently, **48 hours (172,800 seconds) is the only depth in this query
family that passes by eliminating the omitted tail entirely**. This does not
prove that a 48-hour numerical plan approximates the per-second optimum.
The queried schedules change with H, and the learned multi-step laws need not
be consistent marginals of one common per-second process. In particular, the
difference between the tabulated 48-hour and one-hour wealth values is not an
estimate of the same-law truncation error.

For a genuine per-second prefix solve, a weaker necessary depth can be derived
without solving it. Even if its prefix relaxation gap were zero and its value
attained the largest value allowed by the cash theorem, passing would require

    (172800 - H) * 10000 < 0.001 * [10000 + (H + 1) * 10000].

The smallest integer satisfying this strict inequality is **H = 172628
seconds = 47 hours 57 minutes 8 seconds**, leaving 172 seconds. This is only
a necessary depth for this envelope, not a sufficient depth or a measured
accuracy guarantee. Actual values substantially below that maximal wealth
bound require more depth or a tighter upper bound.

## Implementation and verification

`packages/bot-algo/src/oracle-tail-bound.ts` adds the finite prefix upper backup,
exact relative envelope and necessary-depth calculation. It implements the
finite specialization of the calculation described in
[the relative-tail proof report](oracle-crra-relative-linear-tail-2026-09-13.md).
No new Lean theorem or executable-to-Lean refinement proof is claimed here.

39 TypeScript tests passed, including fixed-first-action handling, relaxed
frontier closing, exact full-continuation comparisons with fees/funding/limits,
relative units, and the strict depth threshold. The Oracle TypeScript project
typecheck passed. The previous Lean theorem files were not changed.

Completed artifacts: `data/backtests/oracle-relative-48h-20260913-v3/` contains
the forecast samples, source/checkpoint hashes and exact rational results.
The earlier two export directories precede fixes to script startup and decimal
parsing; they have no completed report. Verification logs are
`data/backtests/oracle-relative-48h-{tests,typecheck}.log`.

```powershell
.venv-ml/Scripts/python.exe ml/export_oracle_tail_bound.py --output data/backtests/new-tail-evaluation
node --conditions=development --import tsx scripts/evaluate-oracle-tail-bound.ts data/backtests/new-tail-evaluation
node --conditions=development --import tsx --test packages/bot-algo/test/oracle-tail-bound.test.ts packages/bot-algo/test/oracle-execution.test.ts packages/bot-algo/test/oracle-crra-policy.test.ts
node node_modules/typescript/bin/tsc -p packages/bot-algo/tsconfig.oracle.json --noEmit
```
