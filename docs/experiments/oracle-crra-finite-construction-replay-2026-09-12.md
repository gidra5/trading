# Replay of the proved finite-tree CRRA construction

The existing finite-tree/menu policy was tested over **144,533 contiguous
one-second transitions**, from **2026-08-10 00:00:01 UTC** through the requested
final timestamp **2026-08-11 16:08:54 UTC**. It selected cash at every decision:
**0% return, zero drawdown, zero market orders and zero limit placements/fills**.
Initial and terminal wealth are both 10,000 quote. There are no residuals,
cancellations or liquidations. Same-engine buy-and-hold returned **-2.101732%**.

This tests the user's accepted **restricted construction**. The account is
carried across the entire replay, with physical finalization only at its end.
The planner uses the existing **two-tick conditional tree**, shortened to one
tick for the final decision so it never optimizes past the replay end. It
evaluates optimal continuations inside that tree. Its guarantee is optimality
for each supplied finite planning problem, **not full-interval Bellman
optimality**. The separate fixed-terminal continuous-action construction audit
remains applicable; this run does not complete that unrestricted construction.

## Declared model and action restrictions

The replay calls the existing exact-rational `solveCrra` Bellman implementation
and uses its original forecast adapter. It does not use the endpoint optimizer
or impose an actual close at each forecast. Gamma is zero throughout.

| Parameter | Value |
| --- | --- |
| Market exposure targets | -2, -1, 0, 1, 2; zero trade also available |
| Entry / maintenance bands | [-2, 2] / [-3, 3] |
| Received-amount market / maker fees | 0.05% / 0.02% |
| Signed-asset funding multiplier | 1 per tick |
| Actual quote order minimum / aggregate maximum | 5 / 10,000 per decision |
| Price tick | 0.01 |
| Active limit slots / maximum new orders | 1 / 1 |
| Limit prices | Buy or sell one price tick away |
| Limit notional | Half current equity, subject to the physical cap |
| Reservation leverage | 2 |
| Cancellation choices | Every subset of the existing book |
| Forecast branching | Three active-bin representatives plus the zero atom |
| Planning horizon | Two seconds; one second on the final decision |

Market targets use exact fee-aware sizing and the remaining aggregate cap;
infeasible combinations are rejected by Oracle execution. Pending orders can
survive decisions and mature into fills. Finalization processes arrival fills,
cancels remaining orders and attempts one valid capped close, retaining residual.
The selected cash trace makes those operations trivial but does not remove them
from the candidate planning problems.

The input is the archived causal two-tick model
`stochastic-crra-production178-zeroatom-k256-v1`, not the newer direct59/base17
mixture. Its rollout updates hypothetical return lags and freezes the 58 other
observed features. Forecasts contain conditional active-bin means and are
rounded to the execution price grid; indistinguishable observations are merged
before continuation choices. This is a finite forecast model, not a proved
policy-preserving reduction of the full learned density. The latest mixture has
no corresponding future-feature rollout adapter, so it was not silently inserted
into this test.

The source remains BTCUSDT spot closes with synthetic Oracle account rules;
this is not an actual USD-M funding, spread, queue or partial-fill replay.
The forecast prefix SHA256 is
`8e51d6937d40f8568f62b94af35bef9b311c11cfbd0716c5ae46327fe391e9f4`.

## Exact cash-dominance acceleration

For a fixed terminal price P, value an account at `q + a*P`. An Oracle trade at
price p and received-amount fee f cannot increase this value whenever

```
(1-f)*P <= p    and    (1-f)*p <= P.
```

These inequalities cover both buying and selling, including signed/leveraged
holdings. Unit funding and price marking preserve this fixed-price valuation.
Zero-account liquidation preserves an upper bound by nonnegative initial cash.
Reservations change available collateral but not quote/asset balances. Limits
fill through the same received-amount trade arithmetic at their execution price.
Actual feasibility restrictions can only remove candidate traces.

The new `OracleShadowBound` Lean module proves the trade inequality, the marking
and liquidation steps, a trace-level capital bound and its terminal-wealth
conclusion. If all candidate marks and possible menu limit execution prices lie
in [lo, hi] and `(1-minFee)*hi <= lo`, the condition holds for every possible
terminal price. Therefore cash reaches the pathwise upper bound and is optimal
for any normalized probabilities on this tree. This is a sufficient dominance
certificate, not a heuristic threshold or tail deletion.

The replay applies that certificate to **132,197 decisions**. The remaining
**12,336 decisions** use the existing exhaustive rational Bellman solver.
An additional **35 bounded cases** are cross-checked by exhaustive search; their
chosen action is also the first, empty-book zero trade. No policy action is
changed by a floating-point confidence or exposure multiplier.

Eight workers evaluate disjoint issue indices. This parallelization is valid
only because every decision preserves exactly the same cash balance and empty
book. Any non-cash result aborts the specialized replay; it does not reset
inventory. The final independent verifier links all issue indices and replays
every actual transition as one continuous rational account. Optimization took
approximately **136 seconds**, excluding verification.

## Verification

- `npm run theory:check` built the theory library and passed its axiom audit,
  including the new Oracle shadow-bound lemmas and trace theorem.
- **14 focused TypeScript tests** pass, including exhaustive comparisons with
  passive orders, rejection of an insufficient envelope, and the regression
  where entering and carrying inventory beats separate one-tick roundtrips.
- The three new TypeScript files pass a focused strict typecheck.
- An independent verifier checks unique coverage of all **144,533 decisions**,
  forecast prefix and source identities, all actual price grids, and every
  rational account transition, plus terminal close and the buy-and-hold control.
- It checks each shadow envelope against every node in its supplied tree and
  the possible one-tick menu limit prices. Lean's integer checker then accepts
  **all 132,197 envelopes**. An invalid envelope is correctly rejected.
- Lean's existing CRRA checker accepts **51 sampled complete decision
  certificates**, including their supplied candidate child continuations. It
  also accepts the **108 Oracle primitive fixtures** used by that verification
  command. The unsampled exhaustive decisions are not individually exported
  to Lean.

The mathematical trace theorem is proved in Lean. Foreign-code parsing,
forecast construction and matching the complete menu to that theorem retain
their stated implementation obligations; these checks are not a universal
formal refinement of TypeScript. No theorem about the real market distribution,
continuous-action global optimum or whole-replay optimum is asserted.

## Artifacts and reproduction

`data/backtests/oracle-finite-cash-replay` contains the plan, result, per-worker
decision logs, all shadow envelopes, sampled Bellman certificates, source hashes,
exact execution verification and both Lean verification reports.

```powershell
node --conditions=development --import tsx scripts/backtest-oracle-finite-cash.ts --output data/backtests/oracle-finite-cash-rerun
node --conditions=development --import tsx scripts/verify-oracle-finite-cash.ts data/backtests/oracle-finite-cash-rerun
node scripts/verify-oracle-crra.mjs --certificates data/backtests/oracle-finite-cash-rerun/bellman-certificates.jsonl --output data/backtests/oracle-finite-cash-rerun/lean-bellman-verification.json
node scripts/verify-oracle-shadow.mjs data/backtests/oracle-finite-cash-rerun
```

No live bot configuration or forecast checkpoint was changed. There is no new
zero-fee replay in this experiment; earlier zero-fee endpoint results are not
comparators for this different controller/model pair.
