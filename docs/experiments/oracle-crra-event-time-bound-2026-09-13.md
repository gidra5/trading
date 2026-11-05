# Skipping economically quiet intervals

Measuring lookahead by relevant events instead of elapsed ticks removes the
arbitrary waiting-time obstruction in the
[flat-horizon example](oracle-crra-flat-horizon-bound-2026-09-13.md). Its delayed
opportunity contains only a fixed number of price moves regardless of how many
identical prices occur between them. That example does not disprove a bound
expressed in an appropriate event count.

This change permits a bound to remain unchanged during an irrelevant wait.
It does not by itself prove that the bound decreases at each retained event.
Exact compression and decay of the omitted continuation are separate claims.

## What can be skipped exactly

`OracleEventTimeBound.cash_wait_step` proves an identity for the original
Oracle account and pending-book machine. If quote is positive, asset and book
are empty, current and previous price equal p, the next price is p, and the
selected action makes no trade or order changes, the resulting account/book
state is identical. Funding may vary: signed-asset funding has no effect on
zero asset holdings. Fees, arrival processing and liquidation remain part of
the actual transition used in the proof.

`cash_wait_block` extends this identity to any finite list of such ticks.
`cash_wait_continuation` evaluates the exit continuation at t + duration,
preserving elapsed time. These are physical transition equalities. They do
not assert that the no-op is admissible under every custom account predicate
or optimal when alternative trades remain available.

For optimal-policy compression, the skipped interval must also preserve the
choices that matter at its exit. If B_(t:u) denotes the block's Bellman operator
with continuation F, a sufficient condition for replacing all internal
decisions by a fixed waiting rule is

    B_(t:u) F(s,h) = E[F(s_u,h_u) | s,h, wait].

Here the right side retains the actual exit state, observed history and date.
The waiting rule must be feasible, and no other feasible internal policy may
give a larger continuation value. `bellman_skip_if_dominates` proves this
condition at one original continuous Bellman decision. Applying such certified
equalities successively preserves the Bellman value. The theorem provides a
condition for a skip, not an automatic method of detecting it.

More generally, an interval can be replaced by a macro decision that retains
all feasible internal policies and their exit laws. That is an exact grouping
of decisions, but solving the macro decision may still require substantial
work. Calling a whole block one event does not make that work disappear.

## A flat price is not a sufficient detector

- Several flat ticks may be needed to accumulate or unwind a position under
  the aggregate quote cap. Deleting them and allowing one capped order changes
  the reachable exit states. The macro must retain this execution opportunity.
- Nonzero positions incur actual funding and conservative assessment during
  the block. Their exposure and liquidation status can change without a price
  move. Products of funding factors alone need not preserve intermediate checks.
- Order age, reservations, cancellation opportunities and arrival timing remain
  relevant. Pending orders cannot be treated as a cash-only account.
- Flat observations and elapsed time can change a history-dependent price law.
  The predictor state and calendar must advance through the block.

For stochastic execution, the next event time is a stopping time, rather than
a future time revealed to the current action. A macro forecast must describe
the joint law of duration and exit information/state. Realized future duration
or exit outcome cannot be supplied to an earlier action. With a fixed terminal
date T, T itself is an exit boundary if no intervening event occurs. These
stochastic macro-kernel requirements have not been implemented or instantiated
in a new Lean stopping-time policy construction here.

## A numerical error budget in event count

The new conditional theorem `event_count_error_bound` refines the existing
n*delta approximation budget. Let active(i) indicate the stages with possibly
inexact Bellman selection. Require local loss at most delta at these stages
and zero loss at every other stage, against the full continuation value. For
normalized nonnegative finite-outcome transition weights, it proves

    V_n(s) - W_policy,n(s) <= N_active(n) * delta.

The proof inducts over the original stages. Expectation preserves an inherited
error bound because the weights sum to one. An exact stage contributes zero;
an inexact stage contributes at most delta. Consequently arbitrarily many
certified exact waiting stages do not enlarge the numerical error budget.

This theorem has an explicitly supplied stage schedule, not an inferred or
state-dependent quiet detector. It is a generic finite-outcome lemma with
Bellman backup hypotheses, separate from the concrete account-wait proof.
It is not an assertion that the current NSF planner satisfies those hypotheses.

The inequality bounds approximation of a specified full continuation. It is
not a bound on how much continuation remains after N events. For the latter,
the existing certified upper/lower value gap still applies; a universal
shrinking rate in event count remains unproved. Normalization of probabilities
does not itself provide a strict contraction at retained events.

## Validation

All five public results in `TradingTheory/OracleEventTimeBound.lean` are
imported by the project and included in its public-theorem axiom audit.

    node scripts/theory-lean.mjs check

The full build and audit passed. Log:
`data/backtests/oracle-event-time-bound-lean.log`.
No execution behavior, learned model or backtest was changed in this follow-up.
