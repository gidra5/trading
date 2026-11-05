# Gamma-one continuation comparison

Changing gamma from zero to one barely changes the continuation result on the
same five-stage sampled NSF law. At the primary origin, the one-hour continuous
relative upper is **4.6540%**, compared with **4.6604%** for gamma zero. The
earliest evaluated checkpoint satisfying epsilon < 0.1% remains **48 hours**.

Gamma one optimizes expected log terminal wealth. Percentages below compare
certainty-equivalent wealth rather than mean terminal wealth. All Oracle
execution rules, forecast prices, probabilities, limits and terminal-flat
requirements remain the same as in the preceding experiment.

## Objective and percentage definition

For initial wealth unit W0 = $10,000, fixed current action a, and horizon H,

    V_H(a) = sup_policy E[log(W_H / W0)]
    C_H(a) = W0 * exp(V_H(a))
    epsilon_H = (C_48h(a) - C_H(a)) / C_H(a)
              = exp(V_48h(a) - V_H(a)) - 1.

C_H is certainty-equivalent wealth: the guaranteed wealth having the same log
utility as the optimal policy's terminal wealth distribution. Normalization by
W0 does not change policy rankings or this relative contribution. In contrast,
dividing two log values would depend on the chosen currency unit and can have
a zero denominator. The one-hour normalized optimal log value here is zero.

Log utility penalizes downside; any positive probability of zero terminal
wealth gives negative-infinite expected log utility. It introduces neither
time discounting nor a guarantee that distant continuation matters less.
The horizon calculation still compares complete flat-ending policies.

## Matched results

Primary origin, 2026-08-10 00:00:01 UTC:

| Cutoff H | Gamma-one finite certainty equivalent | Continuous relative lower | Continuous relative upper | Refined finite-menu relative upper |
| --- | ---: | ---: | ---: | ---: |
| 15 minutes | $10,000.000000 | 2.613101% | 4.653991% | 3.474178% |
| 37 minutes 30 seconds | $10,000.000000 | 2.613101% | 4.653991% | 3.474178% |
| 1 hour | $10,000.000000 | 2.613101% | 4.653991% | 3.474178% |
| 24 hours 30 minutes | $10,048.889755 | 0.888612% | 4.144830% | 2.970757% |
| 48 hours | $10,261.310106 | 0% | 0% | 0% |

Every earlier checkpoint is rejected by a lower bound, including 24.5 hours.
At 48 hours epsilon = 0 follows from the empty-tail identity; it is not a claim
that the continuous numerical upper and lower value bounds coincide. The
five modeled offsets are [900,2250,3600,88200,172800] seconds. This search does
not identify a precise crossing between them, or cover omitted per-second
execution and unsampled NSF outcomes.

| Origin | Gamma-zero 1h upper (mean wealth) | Gamma-one 1h upper (certainty equivalent) | Gamma-one finite C_48h | First checkpoint below 0.1% |
| --- | ---: | ---: | ---: | --- |
| 0 s | 4.660402% | 4.653991% | $10,261.310106 | 48 h |
| 300 s | 4.659375% | 4.652965% | $10,261.263907 | 48 h |
| 900 s | 4.656339% | 4.649932% | $10,261.078703 | 48 h |

The one-hour-selected current action is zero trade with no orders for both
gamma values at every origin. Holding that action fixed, the gamma-one
selected finite policy's expected terminal wealth equals the previous
gamma-zero optimum at all three origins: respectively $10,261.773366,
$10,261.726983 and $10,261.541055. Equality of those aggregates does not prove
that every future action of the policies is identical. At the primary origin,
the mean exceeds the gamma-one certainty equivalent by about $0.46, explaining
why risk aversion has little numerical effect on this particular sampled law.

The one-hour continuous certainty equivalent is exactly $10,000: staying cash
attains it, and the conditional mean-wealth bound plus Jensen certifies it.
The continuous 48-hour certainty-equivalent upper is $10,465.399126.

These are model-based value calculations, not a new realized-market backtest
or evidence of calibration of the sampled law.

## Bound construction

The existing Oracle shadow bound is an upper B(omega) on actual flat terminal
wealth for every continuous feasible policy along each sampled path. Since
log is increasing,

    E[log W] <= E[log B].

Separately, any upper U on E[W], including the guarded conditional shadow
bound, gives by Jensen

    E[log W] <= log E[W] <= log U.

The evaluator takes the smaller of these upper bounds, subtracting log W0.
This transfers a wealth envelope to the log objective; it does not assume
that the mean-wealth optimizer is also the log optimizer. The original fees,
signed-asset funding, liquidation, aggregate quote cap, price grid, limit
fills and flat terminal execution remain inside the underlying wealth bounds.

Finite-menu refinement then applies nonanticipative Bellman backups to these
log upper bounds. Four account-search depths were used, as in the gamma-zero
comparison. Its 3.474178% one-hour upper covers only the declared finite
menu; the 4.653991% upper covers continuous controls at the modeled timestamps.

With log-value enclosures [L_H,U_H] and [L_T,U_T], the relative interval follows
from monotonicity of exp:

    exp(L_T-U_H)-1 <= epsilon_H <= exp(U_T-L_H)-1.

The reported lower endpoint is clipped to zero because a feasible earlier
flat close can stay cash through the remaining stages in this experiment.

## Numerical and formal verification

`oracle-log-policy.ts` adds a gamma-one finite-tree evaluator with outward
logarithm intervals. Existing rational gamma-zero/two solvers and the bot's
default gamma remain unchanged. The new evaluator is used explicitly for
this comparison.

Each rational positive terminal wealth is reduced to a power of two times a
mantissa in [1,2]. The identity

    log x = 2 * sum_(j>=0) y^(2j+1)/(2j+1),  y=(x-1)/(x+1),

is evaluated with 32 terms and outward rounding on a 2^64 fixed-point scale.
The omitted positive tail is bounded by (9/4)*3^(-65). Probability backups
and action-interval comparisons use exact rational arithmetic. A selected
policy attains or exceeds the reported lower endpoint; the maximum upper
endpoint covers all menu candidates, including overlapping intervals. Thus
unresolved numerical ties cannot silently invalidate the optimality-gap bound.

At all three origins, the full finite optimum's log interval has width below
3.524e-18. Decimal exponential values in the report are display approximations.
The actual strict epsilon test is performed as

    delta_upper < lower_bound(log(1.001)),

using rational interval comparisons, not floating-point exponentials.

The new `OracleLogTailBound.lean` proves pathwise log monotonicity, transfer
from a mean-wealth upper bound, the analytic log-series enclosure, the
certainty-equivalent ratio identity, interval propagation, and the strict
tolerance test. The full Lean build and axiom audit pass. The TypeScript
execution and integer rounding code are not extracted from Lean.

All 55 relevant TypeScript tests and the Oracle project typecheck pass.
Tests include a one-step example whose exact log-optimal exposure is 0.5
while gamma zero selects 1, currency-unit invariance, positive-probability
ruin, signed funding, limit fills, terminal dust, and nested refinement.
An independent 180-digit Python Decimal check encloses all 1,000 rational log
inputs tested over roughly 600 orders of magnitude; maximum interval width
in that stress check is 3.204e-15.

## Artifacts and reproduction

Input: `data/backtests/oracle-shadow-tail-20260913-v2/forecasts.json`, SHA256
`28db55bd43d205ace04d7a044b02051e6a502fac58637dbffb3def0c3cf2434f`.
The evaluator checks that this matches the gamma-zero comparison's input.
Results, exact rational endpoints and source hashes are stored in
`data/backtests/oracle-log-tail-20260913/{plan,report}.json`.

Verification artifacts:
`data/backtests/oracle-log-tail-tests-all.log`,
`data/backtests/oracle-log-tail-lean-v2.log`, and
`data/backtests/oracle-log-interval-audit-report.json`.

```powershell
node --conditions=development --import tsx scripts/evaluate-oracle-log-tail.ts data/backtests/oracle-shadow-tail-20260913-v2 data/backtests/new-log-tail
node --conditions=development --import tsx --test packages/bot-algo/test/oracle-log-policy.test.ts packages/bot-algo/test/oracle-shadow-tail.test.ts packages/bot-algo/test/oracle-tail-bound.test.ts packages/bot-algo/test/oracle-execution.test.ts packages/bot-algo/test/oracle-crra-policy.test.ts
node node_modules/typescript/bin/tsc -p packages/bot-algo/tsconfig.oracle.json --noEmit
node scripts/theory-lean.mjs check
```
