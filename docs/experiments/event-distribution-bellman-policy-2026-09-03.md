# Event-distribution Bellman policy — 2026-09-03

Status: implemented and evaluated; **not ready for live trading**.

Latest native experiment, v506–v513: preserving ordered opening acceptance and
coupled wealth reduces the first November one-lot bound from 25.54 to 0.335 bp
of slack. Complete feasible root covers expose 59 November and 381 July regions;
33 and 179 hold-equivalent regions are ruled out against verified incumbents.
This is only about 0.6% of the finite request lots, not a global H2 certificate.
July's nonconstant region remains budget-incomplete. Four predefined small
inventory seeds cannot beat November's incumbent; 5.27 million joint-path
rescoring checks validate their lower policies. All 162 focused tests and
workspace typechecks pass. Forecasts, historical returns and the outstanding
position-policy comparison are unchanged; see the native report's final section.

Earlier native experiment, v501–v505: account-region bounds preserve uncertain
acceptance, linked cash/inventory and identical-opening acceptance groups. A
17-control audit isolates fictitious value from choosing a different incoming
account for each future outcome. The grouped bound reduces one-lot mean slack
from 19.46 to 7.88 bp, but costs more and still reaches 37.53 bp of slack. The
broad-replay computation gate fails. All 159 focused tests and workspace
typechecks pass. Forecasts, backtest results and the untested status of the full
position-decision framework are unchanged; see the native report's final section.

Earlier native experiment, v496–v500: nested opening-information partitions add
a tightening route back to the original H1 problem. Two-group proposals recover
the exact best value on 111 of 113 predefined calibration probes; their bounds
detect the two misses. July's fixed-request H2 interval narrows to 0.0004471 bp
in 5.83 seconds. November's second control preserves the H1 request's ranking,
but with only 0.0645% modeled fill probability. All 155 focused tests and workspace
typechecks pass. These are value/computation results, not a globally searched H2
policy, a new backtest, or evidence against position decomposition. See the native
report's final section.

Earlier native experiment, v492–v495: an execution-consistent continuation upper
dominates all 9,629 saved H1 values, with 0.94 seconds of total query time. An
adaptive H2 backup rejects July's old request after one exact continuation solve
in 0.18 seconds, versus 6.40 seconds for its full evaluation; it rejects root hold
without a continuation solve. The winning candidate remains expensive, and the
bound's remaining information advantage prevents a global convergence claim.
All 154 focused tests and workspace typechecks pass. No forecasts, backtest
returns or position-policy conclusions change. See the native report's final
section for the relaxation proof, failed pruning experiment and next steps.

Earlier native experiment, v483–v491: the execution-aware H1 request optimizer
completes all **9,629 ordinary decisions / 28 full windows**, retaining unchanged
forecasts and old control returns. It produces **8 positive / 19 negative / 1
no-fill window**; returns are higher on 10, lower on 17 and equal on one, with
88.42% worst drawdown. Full-law action values and realized accounts reconcile.
Of 6,504 nonzero requests, 5,813 have at least 90% modeled rejection probability.
Higher fixed-model utility therefore cannot be called strategy promotion.
All 152 focused tests and workspace typechecks pass, including exhaustive
comparison on 960 small problems; exact-output specialization makes local paired
probes about 11.8x faster. An exact two-event backup preserves the ranking of three
predefined July calibration requests; direct rescoring of 916,080 paths confirms
the account-level Bellman sum. At 5–8 seconds per root candidate, a global deeper
search needs better execution-consistent continuation bounds before scaling.
This is complete H1 search under declared research execution, not deeper or
stationary optimality, or evidence against the coordinated position framework.
See the native report's final section and `event-native-execution-one-step-coverage-v490`.

Earlier native experiment, v481–v482: the compressed execution transition matches
all **19,258 saved H1/H2 transitions across 28 intervals** within $4.52e-8. The
largest old transition errors arise from next-open cancellations, not funding
precision. Three enriched laws preserve their original joint forecast projections
and weights exactly. A five-lot reduction improves expected one-event log growth
by 11.2931 bp at July's first calibration account. This is a bounded candidate
comparison, not a globally optimized or backtested replacement. Global execution-
aware action search and deeper recursion remain required; all 149 focused tests
and workspace typechecks pass. See the native report's final section.

Earlier native experiment, v477–v480: complete replays cover **27 strict-source
windows plus the March window under an explicit closure scenario**. All 9,629
ordinary H2 actions meet the 0.001 bp tolerance; forced waiting is separately
audited without a fabricated value certificate. March H2 returns -10.97% versus
H1 -6.72%; totals are 7 positive / 17 negative / 4 cash windows. All 145 focused
tests and workspace typechecks pass. This is finite-H2 research coverage under
declared execution assumptions, not full-depth or next-open optimality or a
profitable policy. Position decision decomposition remains a hypothesis for a
matched account-utility/equity/runtime comparison; current accounting equivalence
does not establish that it helps or hurts decisions. See the native report.

Earlier native experiment, v470–v476: **27 complete native inspector windows
and 9,219 actions** are certified within 0.001 bp for finite H2 under their
frozen earlier forecasts. H2 has 7 positive / 16 negative / 4 cash windows;
it earns less than H1 on 16 windows. The March 2023 window remains incomplete:
official archive-derived trade flow confirms no trades during the missing
seconds, requiring availability-aware execution rather than fabricated fills.
Recovery-bound tightening resolves the choppy-window numerical probes. All
137 focused tests and workspace typechecks pass. Full-depth, execution-consistent
optimization and the forecasting objective remain unmet; see the native report.

Earlier native experiment, v454–v469: all 1,562 original H2 actions on the full
June downtrend window and 39 calibration actions meet the 0.001 bp tolerance.
Retaining maximum order size in the continuation bound resolves 214 previously
uncertified decisions without changing their actions. Full-window H2 earns
+272.57%, below H1's +278.59%, with 49.87% drawdown; calibration loses 16.42%.
This adds a second complete native window, not stationary/all-window optimality
or a forecast promotion. Exact controls verify date-only fitting exclusions
and loading separated source intervals. All 136 focused tests and workspace
typechecks pass; details and execution-rejection analysis are in the native report.

Earlier native experiment, v442–v453: overlap-aware estimation leaves original
H1 actions unchanged and removes the two-day H2 short exposure. Weighted H2
returns +1.07%/+0.25% on calibration/test. On the full November inspector
window it earns +59.53%, below H1's +171.16%. All 357 full-window decisions
plus 35 calibration decisions meet the 0.001 bp H2 tolerance. No promotion:
the forecast still understates trend continuation, and all-window/stationary
optimality is unfinished. All 133 focused tests and workspace typechecks pass.

Earlier native experiment, v435–v441: the native day-context ablation preserves
one-second candles and all physical targets. Added estimation history yields
+0.88% calibration / +4.58% test, but the apparent entry fails day-removal and
event-chain sensitivity checks, even after matching the global prior weight.
No new policy is promoted or given deeper Bellman computation. All 131 focused
tests and workspace typechecks pass; see the final native-report section.

Earlier native experiment, v428–v434: adding three earlier estimation days
under the unchanged partition flips the bearish state's mean from -10.63 to
+3.34 bp. It removes H1/H2 entries but worsens calibration CRPS; no new policy
is promoted. A two-by-two history/sampling control and unconditional-law audit
preserve evidence for conditional duration forecasts. Ordinary and return-weighted
separate sign heads fail the calibration checks. All 130 focused tests and
workspace typechecks pass; see the final native-report section.

Earlier native optimization checkpoint, v427: 78/78 original H2 actions are
certified within 0.001 bp on the two-day screen. Calibration improves from
+1.60% H1 to +8.17% H2, while test worsens from -0.80% to -11.00%. The
finite H2 result does not establish stationary optimality or full native coverage.

Earlier native coverage checkpoint: v402 verifies 27 full windows and 6,702,007
decisions, all cash under the unchanged small run tree. The March 2023
archive has an early-close candle and 4,818 missing seconds, so that replay
remains explicitly incomplete. V401 supplies the first historical refit of
the remembered production59/base17 family: 63.02% validation / 65.04% test
active-next-second sign accuracy. The largest validation-calibrated group
mean is only 0.2273 bp against 12 bp trading cost, making cash optimal for
that one-step empirical law from a flat account. This predictor has not yet
been integrated into full-window Bellman replays. See the final sections of
`native-second-event-policy-2026-09-04.md` for sources, checks and scope.

Resolution clarification after the end-to-end review: the user questioned
the use of one-minute instead of one-second candles. The minute resolution
was an implementation choice for this baseline, not a demonstrated advantage
over native seconds. Native support is now implemented with an explicit
clock, separate 64-second features, physical-minute financial durations,
second-scale label bins and next-open execution. The first bounded native
experiments v391/v392 fit earlier data and replay one hour of the November
inspector window; both remain cash. A forecast-measure shadow-price check
v393 also certifies cash through 9,880 run events / 2,549 second steps under
these weak fixed laws. See `native-second-event-policy-2026-09-04.md` for the
contract, source hashes, scope and numerical limitations. These initial
models do not use the remembered structured predictor, and their 5x cap
differs from the minute baseline's 1x cap. They do not establish that either
resolution is better. The old minute results retain their original scope;
v394 reproduces all 1,217 saved H1 decisions after the shared-clock changes.

Architecture clarification: the initial baseline omitted the lifecycle
position decomposition from `docs/theory/Position management.md` without a
comparison demonstrating an advantage. The user clarified that account equity
is the objective and the framework is a starting hypothesis, not a requirement
to retain regardless of evidence. Restoring it unconditionally would repeat
the same mistake in the opposite direction.

An optional replay ledger now creates positions on increases, reduces them
pro rata, closes the old side on reversals, and attributes actual fills,
fees, borrowing and liquidation writeoffs. It reconciles their collapsed
quantity and wealth with the physical account. This is diagnostic accounting;
the optimizer still chooses joint account actions and the four trade flags
do not establish canonical entry/exit Bellman symmetry. Attribution defaults
to enabled for traces and disabled otherwise, and can be switched explicitly
for comparisons. No return improvement is attributed to this bookkeeping.

Compare any position-based decision policy with the account optimizer under
identical forecasts, constraints and execution. Measure account expected log
wealth/action regret, realized equity, fees, failures and computation. Preserve
the framework if it improves these outcomes or offers equal decisions with a
useful implementation benefit; simplify or omit it if it constrains useful
actions or adds unjustified cost. Independent local log-utility maximization
does not replace the account objective. The current finite-horizon numerical
certificates apply to the aggregate baseline; neither a lifecycle ledger nor
four signal flags extend those certificates to a different policy.

The first comparison, `event-policy-position-attribution-v389`, repeats all
28 saved H1 windows from v289 with the same model hashes, costs and execution.
All **1,217 decisions, 254 orders, 17 cancellations, predicted values and
account paths** exactly match the saved reference and match with attribution
enabled or disabled. Maximum final wealth reconciliation error is
**$1.437e-10**. Three alternating timing pairs after warmup total 864.99 ms
without attribution versus 879.51 ms with attribution when summing each
window's median (about 1.7% extra). Data loading and assertions are excluded;
short local timings under concurrent load do not establish a general runtime
penalty, particularly for native seconds. This is evidence of bookkeeping
equivalence and modest measured overhead, not evidence that independent
position decisions help or hurt returns. The coordinator remains the decision
reference; the ledger remains optional attribution. All **118 focused tests**
pass, including an independent cash-ledger comparison over 1,000 fills,
funding credits/debits, zero crossings and terminal settlement. Tracking the
confirmed physical post-fill balance fixed virtual-lot summation dust after
complete closes; the comparison did not change the strategy's orders.

Completed minute checkpoint, v390: **1,217/1,217 H3 decisions across all
28 non-fit inspector windows** certify within **0.001 bp**, with maximum gap
**0.000994598 bp**. Mean independent-window return is H1 **+1.7648%**,
H2 **+3.1414%**, H3 **+3.7478%**. H3 improves eleven windows, worsens ten
and leaves seven unchanged versus H2; eight still lose. Its worst window
loses **19.2431%**, worse than H2's worst loss of 15.7563%. H3 makes 511
orders, pays 2,441.23 in aggregate independent-account fees, has 23 canceled
orders and no simulated liquidations. The final June returns are +34.7497%
and +12.2708%. This closes finite H3 coverage under each frozen minute law,
not deeper/stationary convergence, execution consistency or native-second
forecast quality. No additional minute horizon expansion was started.

Previous checkpoint, v388: **855/855 H3 decisions across 26 of 28
complete non-fit inspector windows** certify within **0.001 bp** of their
global fixed-law optima; the maximum gap is **0.000994598 bp**. On the same
covered subset, mean independent-window return is H2 **+1.6724%**, H3
**+2.2277%**; ten windows worsen and eight still lose. November's larger
short loses gross P&L despite lower fees, while December's profitable trades
remain unchanged. Complete June action evaluation falls from **37.58 to
8.16 seconds** using coarser initial continuations, with the final global
tolerance unchanged. The planner refines only when its remaining interval
could close that global gap, counting both passes against the same budget.
All **115 focused tests**, 1,200 independent H3 audit queries and workspace
typechecks pass. The two dense June windows, deeper/stationary convergence and
execution consistency remain outside this checkpoint. No forecast retraining
or promotion occurred. The saved-source audit v338 additionally corrects the
remembered sign model: later structured production59/base17 models achieved
75.70–76.81% next-second validation direction accuracy, excluding zero targets.
See `structured-next-second-sign-audit-2026-09-04.md`; those models were not
tested by the earlier event-sign trials.

Previous checkpoint, v309: exact H1 and numerically bounded H2 now cover all
**28 non-fit KAMA inspector windows**, using each window's unchanged saved
v10 forecast. All **1,217 H2 decisions** certify within 0.001 bp of expected
log wealth. Mean independent-window return rises from +1.7648% to +3.1414%,
but 11 windows worsen and eight still lose. A dense-law evaluator reduces one
decision from 15.94 to 0.625 seconds and reproduces every trade in a complete
48-event window while reducing its runtime from 37.39 to 3.15 seconds. All
104 tests and workspace typechecks pass. This closes finite H2 coverage under
the stated model contract, not deeper/stationary or next-open-execution
optimality. No forecast retraining or incumbent promotion occurred.

Previous checkpoint, v286: the two-event order search returns feasible
values and numerical upper bounds under the unchanged joint forecast. It
matches exhaustive search on 3,618 small problems and certifies all 402
decisions in the first March prior origin to within 0.001 bp of expected log
utility. That full replay takes 42.5 seconds but loses **11.8242%**, versus
10.5893% for repeated one-event planning. Fees fall from $124.57 to $50.00;
holding a losing short for longer offsets the early churn improvement.
An audit separates two-event forecast residuals from controller disagreement.
All 100 focused tests and workspace typechecks pass. Finite two-event action
optimality is now measured; deeper/stationary, execution-consistent and full
28-window optimality remain open. No forecast retraining or incumbent
promotion occurred in v267–v286.

Previous checkpoint, v266: the new exact one-event lot optimizer closes the
demonstrated sizing gap. It matches exhaustive enumeration in 5,454 fixed-law
cases, an analytical five-billion-lot example, and nine full-lattice probes
of saved market forecasts. On the learned law, exact sizing improves one
March prior origin from +0.1845% to +0.2053%; most decisions remain unchanged.
The next-step marked-wealth objective is now separate from the old forced-sale
objective. It produces substantially more trading but poor prior returns;
its final May/March/January diagnostics are +0.9221%/−3.3907%/−0.3039%.
These are unconditional diagnostics, not selected replacements. All 93
focused tests and workspace typechecks pass. The multi-event optimum remains
unproven; no forecast training or incumbent promotion occurred in v258–v266.

The preceding v245–v255 exact deployed rollouts isolate next-open order
cancellations from forecast error. A precommitted quote-entry experiment removes
those cancellations and improves the held-out variant's active May origin from
+21.6343% to +24.6389%, with five fewer orders. Full-data return slightly worsens
from +23.9564% to +23.9089%; the other origins are unchanged. Both variants and
the retained incumbent remain cash on the final May window. The short-horizon
value forecasts remain inaccurate. No model training was needed for v245–v257.

The user revised the objective on September 4: first establish policy
optimality **conditional on a fixed forecasting model**, then improve the
forecast to improve returns. This supersedes the earlier requirement to keep
iterating until every market window is reliably profitable. Neither the
interpolated Bellman tables nor the fitted/controller compositions currently
certify that conditional optimum. The first independent small Bellman
reference now measures decision regret with the forecast frozen. The one-event
sizing gap is closed under its stated model contract. The next priority is
multi-event continuation with the same explicit objective and execution
contract, before modifying the forecast.

## Question and leakage contract

This experiment predicts the next decision-relevant BTC move rather than every
one-minute candle. An event ends at the first close beyond a symmetric return
barrier or at a timeout. The optional literal run clock ends on the first
observed change from the current raw-candle sign, including the first opposite
candle; a flat run ends on the first nonzero candle. It never backdates an exit
to an unobserved turning point. The predicted joint atom retains endpoint return,
duration, adverse/favorable OHLC excursion, and the next learned state.

Each inspector window gets a separately frozen model. Its 120-day nominal
training interval and 21-day policy-calibration interval end before the scored
window. In the default `suite` isolation mode, every one of the 28 non-fit KAMA inspector windows is purged from the
input and target support, including the preceding 1-day feature history. No
`fit-*` inspector window is scored. The distribution tree, Bellman depth, and
cash gate are frozen before the first scored candle is read by the simulator.
Signals use a completed close and market orders fill at the next open.

The separately labeled `causal` walk-forward mode, introduced in v35, permits
earlier inspector-window prices in training and calibration. All targets still
end strictly before the current test. This removes artificial historical gaps;
it is not the same suite-isolated experiment. No `fit-*` window is scored in
either mode.

The inspector suite has been inspected by many prior experiments. These are
out-of-fit chronological replays, but they are not a pristine final holdout.

## Model

The initial model is a small multinomial/regression CART whose leaves contain a
Dirichlet-shrunk empirical distribution. Its 12 causal features are normalized
returns over 1, 5, 15, 60, 240, and 1440 minutes, 60-minute volatility,
short/long volatility ratio, current raw-candle run sign and length, 60-minute
efficiency ratio, and relative volume. Tree depth is selected from 2, 4, and 6
on pre-window data. The useful experiment split nodes by reduction in return
mean error; the initial duration-classification objective learned event timing
but produced no economic edge.

The optional scale-only expectation calibrator performs an exponential tilt of
observed joint atoms. This preserves return support, duration, extrema, and
successor-state dependence. It failed its first blocked trial: one window began
trading more and lost 2.45%; another useful bearish mean was shrunk to zero.
It remains an explicit research option and was not promoted.

## Policy and account model

The common action is signed target exposure. Positive values are long,
negative values are short, and zero is cash. Long entry/exit and short
entry/exit booleans are derived by comparing the positive and negative parts of
the old and new target. This uses long/short reflection in the symmetric parts
of the account model while still optimizing exits from their actual current
inventory state; it does not claim the invalid flat-entry/exit symmetry warned
about in `Position management.md`.

For every learned leaf, equity grid, price grid, and current-exposure grid, the
Bellman update is

\[
V_d(s,x)=\max_a \left[\log R(x,a) +
  \mathbb E_{(r,D,s')\sim p(\cdot|s)}
  \left(\log G(a,r,D)+V_{d-1}(s',x')\right)\right].
\]

The maximization occurs inside each next state value, after the next state is
observed, but outside the current outcome expectation. This prevents hindsight
action selection. Any atom that liquidates gives expected log utility
`-Infinity`. Each finite depth ends with cash settlement, so deeper policies do
not get free terminal inventory.

The transition accounts for fees, slippage, leverage, maintenance margin,
long/short borrow cost, minimum notional and quantity, quantity step, maximum
order notional, lot rounding, exposure drift, and equity/price changes. The
minimum and maximum feasible orders are compared with waiting; small desired
orders are not blindly rounded up. The value function uses interpolation on a
5×3 equity/price grid and a configurable exposure grid, so its Bellman result
is an approximation to the declared discretized model rather than a proof of a
continuous global optimum.

The simulator marks every intervening one-minute OHLC candle, checks
liquidation intrabar, applies borrow cost, executes at the next open, and
reconciles gross long/short PnL, fees, borrow, terminal settlement, and final
cash. It records canceled orders and value-grid extrapolation.

At event boundaries, marked exposure above the target cap must be reduced.
If the maximum order cannot restore the cap in one fill, only a maximum-sized
risk-reducing clip may remain outside it. Exposure can still drift between
events; this is not a continuous leverage boundary. A regression test caught an
initial implementation that rejected holding but accidentally reintroduced it
as a zero-quantity candidate. The corrected test uses a strongly favorable
return so that waiting would otherwise win.

## Experiment sequence

The compact runs took seconds per window; only the supported variants were
scaled to all 28 windows.

| Variant | Positive | Negative | Cash | Worst return | Max drawdown | Sum of per-window log growth | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| 60 bps / 240m, 5×, depth ≤4 (3-window screen) | 1 | 0 | 2 | 0.00% | 1.11% | +0.0097 | 16s |
| 60 bps / 240m, 5×, depth ≤8 | 2 | 0 | 26 | 0.00% | 16.71% | +0.5187 | 317s |
| 120 bps / 1440m, 5×, depth ≤4 | 9 | 4 | 15 | -16.02% | 48.21% | +2.0808 | 71s |
| 120 bps / 1440m, 1×, depth ≤4 | 9 | 5 | 14 | -8.41% | 14.56% | +0.7647 | 66s |

The original 20-bps/60m classification screen traded in none of its three
windows. “Cash” in the table includes an enabled policy that chose no orders.

The sum is a diagnostic across overlapping windows, not a compound portfolio
return. The median window return is zero in all full runs because the cash gate
usually abstains.

The complete 120-bps/1× active rows are in the machine-readable summary. Its
largest useful replays were +39.16% on the seven-day June 2022 decline,
+23.23% on the November–December 2023 uptrend, and +22.57% on the three-day
known June 2022 miss. Its largest failures were -8.41% on the March 2023
seven-day decline, -4.41% on the January 2024 flat-shape window, and -3.43% on
the March 2023 choppy window.

Refining the 5× action grid from 1.0 to 0.25 exposure increments changed the
June 2022 return from +161.45% to +121.71% and drawdown from 48.21% to 32.53%.
That material sensitivity confirms that the coarse 5× result is not an
optimality certificate. The 1× cap was the more useful change: it reduced tail
loss and drawdown substantially without changing the number of winning
windows.

### Frozen-model replay and follow-up screens

Changing stride-loop alignment shifted the first training origin by one minute
between v5 and v10. This changed selected trees and policies materially, so the
v10 refit cannot isolate the leverage fix. The new replay command keeps the
exact saved v5 distributions and selects depth/cash again using only their
original preceding calibration interval. With the corrected cap, v13 produced
9 positive, 5 negative, and 14 zero-return windows; worst return remained
-8.41% and worst drawdown was 14.41%. Trades increased from 203 to 387.
The June 2022 crash return fell from +39.16% to +32.94%, and the May 2022
choppy loss grew from -0.02% to -0.65%. Enforcing the stated risk rule adds
turnover and does not repair a weak forecast.

The frozen v13 trace makes the directional failure concrete. On January 3,
2024 the same leaf predicted +9.92 bps before successive -331.43, -144.21,
and -255.41 bps events while holding approximately 1× long. In March 2023,
full-size entries predicted +39.83 to +44.24 bps before -133 to -212 bps
events. The account arithmetic explains the loss; the forecast stayed wrong.

The five diagnostic cases below are March 2023 choppy, June 2022 crash,
January 2024 flat shape, December 2023 seven-day uptrend, and June 2026
seven-day decline. Screens are research comparisons, not a selection holdout.

| Follow-up | Positive / negative / zero | Main result | Decision |
|---|---:|---|---|
| Honest chronological partition/estimation, v12 | 1 / 2 / 2 | +16.26% uptrend; -0.30% January with 11.17% drawdown | Do not scale |
| Three day-block-bagged shallow trees, v14 | 1 / 1 / 3 | +0.43% March; same January loss; lost useful trend trades | Do not scale |
| Exact inverted-chart training augmentation, v15 | 0 / 0 / 5 | Conditional means mostly shrank below fees | Do not scale |
| Zero fee/slippage diagnostic, frozen models, v16 | 1 / 2 / 2 | March -13.12%; January -2.83%; borrow still charged | Forecast failure persists without execution friction |
| Literal candle-run/flat clock, v17 | 0 / 0 / 5 | Timing skill did not yield fee-covering directional means | Do not scale |
| Actual one-second feature basis, v18 | 3 / 1 / 1 | March -3.69%; January +1.91%; uptrend +23.23%; June 2026 +4.80% | Mixed; investigate sample support before scaling |
| One-year fit with one-second basis, v19 (3 cases) | 0 / 1 / 2 | March and crash cash; January -1.78% | More history did not fix transfer; do not scale |

Honest estimation learns partitions from the first half and estimates their
laws from the later half. Forest policies average predictive laws before
maximizing; the observable state is the joint leaf signature, so future
decisions cannot learn which ensemble member secretly generated an outcome.
Inverse augmentation regenerates first-passage labels on an inverse OHLC chart;
negating endpoint returns would incorrectly preserve barrier crossing times.

### Online completed-error correction — v20 through v26

The repository's `rolling-forecast-calibration-2026-08-13.md` found that recent
completed forecast errors improved calibration more consistently than frozen
correction. That motivated a causal rolling regression of actual event return
on its original predicted mean. An 8-, 32-, or 128-event history and Bellman
depth are chosen on the preceding calibration ranges. The selected rule and
cash gate are frozen before the scored window; only the rolling completed-pair
state updates during replay. Four completed observations are required before
the first update.

The inferred mean scale selects a prebuilt distribution/policy bank. Each law
uses an exponential probability tilt on the original joint atoms, retaining
observed return support, duration, extrema, and successor dependence. Within a
Bellman rollout the selected scale is held fixed; it is reconsidered at the
next real event. This is a receding-horizon approximation, not an exact Bellman
solution on an augmented rolling-history state.

- The initial nonnegative scale bank (v20) reduced March 2023 from -3.69% to
  -1.72%, retained +20.80% in the uptrend, and chose cash in the other three
  diagnostic cases. It mostly suppressed trading.
- The signed bank `[-2,-1,-0.5,0,0.5,1,1.5,2]` (v21/v22) changed March to
  +2.81% and preserved the original +23.23% uptrend. June 2026 returned
  +3.99%, June 2022 stayed cash, but January 2024 lost -3.10%.
- A calibration drawdown penalty of 0.5 (v23) retained March and the uptrend
  while selecting cash for January. It would also reject June 2026's positive
  calibration, whose log-growth/drawdown ratio was only 0.364.
- A 0.35 penalty was then selected using these inspected diagnostic cases.
  This was explicitly suite-informed tuning, not an untouched validation.
  The broader 28-window run (v25) was stopped after its first two completed
  windows: July 2022 made no trades; **May 2022 lost -10.10% with 13.83%
  drawdown and 23 trades**. The improvement did not transfer.
- The boundary audit found a partially observed last event entering the rolling
  error history. It now enters only when the event has fully resolved, and a
  focused test covers this. The corrected frozen-model May replay (v26)
  reproduced the same -10.10% loss, so the branch remains rejected.

The useful behavior was correcting persistently wrong directional means without
destroying the 2023 trend. The failure was unstable inversion in another choppy
regime: a short calibration episode selected an aggressive response that failed
immediately outside the five-case screen. Increasing the drawdown penalty again
to hide that new loss would just continue tuning to inspected windows. The
next learner should reduce directional-estimation variance directly, for
example a regularized continuous projection with a small empirical joint-state
model, before spending more compute on adaptive policy banks.

### Regularized forecast states and latest refit — v27 through v34

Two smaller first-moment learners were implemented while keeping the same
empirical joint return/duration/extrema/successor law:

- A standardized, clipped ridge projection with penalties 0.01, 0.1, and 1.
- Histogram gradient boosting with depth-two trees, learning rate 0.05, and
  16, 64, or 256 trees selected on preceding calibration MSE.

Each forecast partitions the historical scores into at most eight states. An
exponential probability tilt makes each joint law's mean match the regularized
forecast, so shrinkage reaches the optimizer instead of being undone by a
noisy empirical bin mean. This is still a finite approximate state model.

The ridge screen (v27) produced one gain (+10.29% in June 2026), two losses
(-0.02% March 2023 and -0.30% January 2024), and three zero-return cases.
Its near-flat returns concealed 7.53% and 11.17% drawdowns from holding long
through the choppy reversals. It was not scaled.

The first boosted screens exposed a discretization bug. Requested quantiles
often fell inside tied forecast masses; rejecting those boundaries collapsed
many models into one unconditional state despite nonconstant underlying tree
predictions. The v28–v31 results therefore do **not** evaluate the intended
conditional boosted policy. The replacement chooses nearby distinct score
boundaries with minimum support. A regression test requires tied nonlinear
scores to produce distinct states and both long and short actions.

The corrected six-case screen (v32, depths up to 16) gave:

| Window | Frozen early fit | Latest eligible refit (v33) |
|---|---:|---:|
| May 2022 choppy | -3.83% | -1.09% |
| March 2023 choppy | -0.02% | -0.02% |
| June 2022 crash | 0.00% | 0.00% |
| January 2024 flat shape | -0.30% | -0.30% |
| December 2023 seven-day uptrend | +16.10% | +16.26% |
| June 2026 seven-day decline | +12.86% | +10.27% |

The latest refit is an explicit separate protocol. Hyperparameters, Bellman
depth, and the cash decision are selected with the original earlier model.
Only then is that chosen learner refit on the latest eligible trailing 120
days and frozen for the scored episode. The artifact retains both
`selectionPolicy` and the final `policy`; frozen replay calibrates the former
and scores the latter. All six calibration score arrays, chosen depths, and
cash decisions were verified unchanged, and every final target timestamp
precedes its test boundary. This does not validate a rolling retraining
procedure; that would need chronological refits inside calibration too.
The separate saved-model replay (v34) exactly reproduced May's -1.091545%
return and 9.028605% drawdown using the stored selection and refit policies.

Refitting could not make every model current. Purging all non-fit inspector
windows leaves the March 2023 choppy model's last usable outcome 15.13 days
before its test and the June 2022 crash model's last outcome 5.28 days before
its test. The other four refits end within 21–73 minutes of their test.
This suite-wide isolation should be distinguished from a live prequential
protocol that learns from all already observed prices. Any such comparison
must remain separately labeled and may never read the current test's future.

Additional causal run-amplitude, distance-from-high/low, and extreme-age inputs
were implemented as `--path-basis`; their first screen (v31) predates the tied
state fix and is not valid evidence for promoting or rejecting that feature
set. There is no full 28-window run of a corrected boosted model yet: the
six-case screen still has three losses and 11.17% maximum drawdown.

### Causal refit and event-sequence models — v35 through v39

The causal latest-refit boosted screen (v35) used all preceding prices rather
than deleting earlier inspector windows. Its six returns were -1.09%, -0.02%,
+41.96%, -0.30%, +16.26%, and +7.03%, in the table order below. It restored the
June 2022 short but left all three choppy failures. The prior 15-day March gap
disappeared; every final target ended 21–74 minutes before its test. This is
evidence that the exclusion protocol affected the result, not a profitability
solution.

Drawdown marking was corrected after v35. Earlier runs retained close equity
peaks but missed favorable intrabar peaks. Current `maxDrawdownPct` is the
conservative OHLC envelope: assume the favorable extreme precedes the adverse
extreme, with full bar borrow charged at the adverse mark. Since OHLC does not
reveal that order, it is an upper-bound convention, not an exact tick-level
drawdown. `closeDrawdownPct` separately records close-to-close equity drawdown.
Historical drawdowns above have not been silently relabeled or recomputed.
A regression test checks a 120-to-100 intrabar equity path alongside a
110-to-105 closing-equity path.

The new hidden-regime learner uses Baum-Welch on complete endpoint-aligned
event chains. Gaps and separate source series reset the sequence. Two or three
latent regimes emit the existing 15 direction/duration classes. The policy
receives a quantized posterior over regimes, not the latent regime itself.
For belief `b`, transition matrix `T`, and emissions `E`, a branch predicts
`q = b T`, gives class `c` probability `sum(q[j] E[j,c])`, and updates its next
belief proportional to `q[j] E[j,c]`. Both the Bellman kernel and replay use
the same posterior grid and deterministic update. This prevents the planner
from observing a hidden generating state that the real strategy could not see.

The four-division simplex has five states for two regimes and fifteen for
three. Each class has up to eight return strata; endpoint quadrature preserves
its probability, mean return and mean duration, while retaining the worst
observed excursions in each stratum. This remains a coarse joint quadrature,
not an exact empirical path law. The learner pools outcome sizes within a class
across regimes. It currently uses the event sequence rather than the snapshot
OHLCV feature vector. Snapshot-only inference rejects this model explicitly.

Policy replay starts with the fitted stationary prior and processes only
completed events from at most the previous day. Partly observed warmup events
are discarded; a new event anchors the scoring interval. Truncated terminal
events never update the filter. Diagnostic likelihood metrics reset at each
separate chain and do not use the replay warmup. Independent mean tilts are
rejected because they would invalidate the posterior transition law.

All screens below use 120-bps barriers, 1440-minute timeout, 1× maximum target,
10-bps fees plus 2-bps slippage, causal latest refit, and depth at most eight.
Parameters and depth are selected before each scored window.

| Window | NLL-selected hidden model v36 | MSE-selected v37 | Mirrored-series v38 |
|---|---:|---:|---:|
| May 2022 choppy | +1.19% | 0.00% | 0.00% |
| March 2023 choppy | -0.02% | -0.02% | 0.00% |
| June 2022 crash | +30.63% | 0.00% | 0.00% |
| January 2024 flat shape | -0.30% | -0.30% | 0.00% |
| December 2023 seven-day uptrend | +16.26% | +16.26% | 0.00% |
| June 2026 seven-day decline | 0.00% | 0.00% | 0.00% |

The v36 May improvement used six trades and had 2.71% OHLC-envelope drawdown.
March and January still held one long position throughout, with 7.85% and
11.18% envelope drawdowns. Their visited posterior states all predicted
positive means: +8.70 to +32.59 bps in March and +15.42 to +23.04 bps in
January. The filter did change states; the entire learned support of
conditional means remained bullish. Better duration likelihood did not give
the model useful reversal evidence. Selecting by MSE in v37 removed two useful
trades without correcting those losses.

The mirrored-series trial rebuilds the reciprocal OHLC chart, recomputes its
first-passage events, and appends it as an independent sequence. It does not
negate labels or connect the end of one chart to the beginning of the other.
This removed the unconditional drift preference but all six tests made zero
trades, including the enabled December policy. That is abstention, not a
successful symmetric strategy. None of these variants justified a 28-window
scale-up or live use.

Frozen-model replay v39 exactly reproduced v36's May and March returns,
drawdowns, depths, and orders. Replay now inherits the source risk penalty by
default. All 28 focused tests and the full workspace typecheck pass, including
sequence-gap isolation, observable posterior transitions, future-invariant
first orders, and serialized policy inference.

### Directional-change and run-progress clocks — v40 through v44

The local theory's compressed-state section requires current run anchors,
duration, extrema, and pending counter-moves. It also says an intermediate
observation that can change the optimal action must be retained as a decision.
The fixed-origin barrier and reversal-only clocks do not establish that
condition. [Algorithmic trading with directional changes](https://link.springer.com/article/10.1007/s10462-022-10307-0)
provides related work: it distinguishes continuation beyond a confirmed turn
from immediate reversal and uses confirmation duration and earlier overshoot
information. Its reported FX results are not validation for this BTC model.

`--reversal-clock` now ends at an observed close reversing by the configured
log-price threshold from the running extreme, or at the timeout. It never
executes at the earlier extreme. A one-day bounded causal reconstruction
supplies eight additional features: direction, signed run magnitude, age,
overshoot beyond confirmation, current pullback, confirmation duration, and
the previous overshoot's magnitude and duration. The one-day bound is explicit:
this is not an unlimited-history directional-change model. It makes input
purging and saved-model replay independent of how much older data was loaded.

`--progress-bps` also ends the event on that much signed log-price progress
from its current origin. This admits decisions during run extension and feeds
observed overshoot state into training. Both upward and downward prices use
the same log threshold, so reciprocal charts have matching boundaries. Tests
cover a continuation that remains profitable at reversal confirmation,
mid-run progress, reciprocal timing, future invariance, and invalid clocks.

`--model-selection utility` evaluates each predeclared model and Bellman depth
on preceding calibration net log growth minus the same drawdown penalty.
Prediction-based selection remains the default. Utility selection retains the
full calibration scores for every candidate; it increases selection degrees
of freedom and does not make the repeatedly inspected suite untouched.

All three six-window screens below use 60-bps reversal thresholds, a 1440m
timeout, endpoint-chain training, trees of depth 0/2/4, a 100-sample prior,
1× maximum target, depth at most eight, and the causal latest-refit protocol.

| Window | Reversal / MSE v40 | Reversal / utility v41 | Progress + reversal / utility v43 |
|---|---:|---:|---:|
| May 2022 choppy | 0.00% | 0.00% | 0.00% |
| March 2023 choppy | -0.43% | -0.43% | -0.02% |
| June 2022 crash | 0.00% | 0.00% | 0.00% |
| January 2024 flat shape | -2.64% | -2.64% | -2.66% |
| December 2023 seven-day uptrend | 0.00% | +15.96% | +16.18% |
| June 2026 seven-day decline | 0.00% | +9.59% | 0.00% |

Utility selection recovered useful trend trades but left the choppy failures.
The reversal model's March forecasts ranged from -9.32 to +25.75 bps; January's
from -6.26 to +18.32 bps. Unlike the earlier hidden model, these laws did visit
negative-mean states. Deeper continuation values nevertheless often favored
coasting through them rather than paying for an exit and later re-entry.
Observed losses do not alone prove that choice was numerically wrong; they
show its assumed conditional recovery did not transfer reliably.

The frozen no-refit control v42 retained each v40 calibration-selected model
instead of its latest refit. March deteriorated to -18.59%, with 59 trades and
23.13% OHLC-envelope drawdown. January improved to -0.30% while still holding
through 11.18% drawdown. Thus simply restoring the older fit is not a fix.
`research:event-policy:replay --use-selection-model` reproduces this explicit
control and rejects sources without a separate selection model.

The new `research:event-policy:audit` command is deliberately a **hindsight
diagnostic**, not a tradable result. It keeps the previously learned states
fixed, estimates their outcome laws from the scored window, and selects depth
on that same window. It also compares an unconditional law from those outcomes.
No diagnostic output carries the ordinary research-backtest contract.

| Fixed v43 states | Actual return | Hindsight unconditional law | Hindsight state-conditional law |
|---|---:|---:|---:|
| March 2023, 9 states | -0.02% | 0.00% | +2.40% |
| January 2024, 3 states | -2.66% | 0.00% | 0.00% |
| December 2023, 7 states | +16.18% | +16.26% | +16.38% |

This is an expressiveness diagnostic, not a formal upper bound on all policies.
March has usable separation under a hindsight law, but several historical
conditional means reverse sign. January's three-state representation cannot
produce a useful policy even in this generous diagnostic. The December profit
mostly comes from a known-positive market drift rather than detailed states.
Changing only the clock, increasing depth, or making the same model larger
without addressing these failures is not supported. No variant was scaled to
all 28 windows.

### Canonical run law and horizon diagnostics — v45 through v50

`--run-symmetry` pools upward and downward run observations in canonical run
coordinates. The observable run direction determines orientation; signed
return and run features reflect together, while volatility, duration, and
activity retain their values. A shared tree predicts the joint canonical
return/extrema/duration/successor law. Each leaf expands into physical up,
down, and neutral states. Down states use reciprocal returns and excursions,
with successor orientation reflected too. Neutral states use the equal mixture.
This is an explicit market-model assumption, not an account-action theorem:
actual inventory, arithmetic wealth, borrowing, and order constraints still
enter the optimizer separately. Validation checks the full reflected law and
neutral mixture, not just the mean.

The v45 screen kept the 60-bps reversal/progress clock and 1× causal latest-refit
protocol. It tried canonical tree depths 0/2/3 with a 100-sample prior, selected
model/depth on calibration utility, and limited Bellman depth to eight.

| Window | Selected canonical tree depth | Physical states | Policy depth | Test return | Test trades |
|---|---:|---:|---:|---:|---:|
| May 2022 choppy | 2 | 12 | 1, cash gate | 0.00% | 0 |
| March 2023 choppy | 2 | 12 | 6 | 0.00% | 0 |
| January 2024 flat shape | 0 | 3 | 1, cash gate | 0.00% | 0 |

March's selected model improved calibration NLL to 2.079 from an unconditional
2.388 and mean MSE skill to +0.00508, but that did not become test profit. Its
latest refit chose no exposure even though calibration had selected an active
policy. The symmetric assumption removes much of the historical physical drift
that generated the trend profits. This screen does not establish a replacement
directional edge and was not scaled to the remaining inspector windows.

Bellman construction now compiles the empirical atoms and interpolation weights
into a sparse transition operator once, then reuses it at every depth. Every
positive-probability outcome remains represented, including ruin reachable
through subnormal weights; the serialized model retains its complete empirical
law. A paired benchmark against the authoritative saved implementation used
the v45 March model (12 states, 8,740 atoms, eight depths). Across three paired
trials, median construction fell from 7.736s to 2.011s, **3.85× faster**. Maximum
value-table error was 1.35e-17, with zero finite/infinite mismatches. The earlier
v46 single timing overlapped tests and is not the speed estimate. v47 records
the paired timings and source hashes; the reference implementation is loaded
in memory from the artifact, not kept as a legacy workspace implementation.

Each depth now records the fraction and largest magnitude of changed grid
actions and the span of the finite value increment. These are diagnostics for
the learned finite grid, not a certificate of true-market optimality. A single
unchanged depth is insufficient: May's actions changed again at depth 16 after
being unchanged at eight, and last changed at 49.

Frozen-model horizon replay v48 built through depth 64. May and January still
selected cash; May's calibration scores worsened to -0.02094 at depth 32 and
-0.10652 at 64. Its sparse evaluation schedule initially omitted March's
original winning depth six, selected eight, and lost 1.02464% with three trades.
That was a selection-schedule confound. Corrected v49 retained every original
depth 1–8 before adding 16/32/64, reselected six, and reproduced zero trades and
zero return. At depth 64 all three final models had unchanged grid actions
from depth 63, but nonzero value-increment spans. Longer rollouts do not rescue
this model, and zero return is not the requested profitability.

Frozen active replay v50 also checked that the operator improvement preserved
an actual trading policy: December's v43 result remained +16.17927%, 2.39582%
OHLC-envelope drawdown, two trades, and depth eight. All 94 event trace rows
matched exactly on timestamps, states, forecasts, orders, inventory, and equity.
Only continuation-value roundoff was excluded from that exact comparison.
All 32 focused tests and the full workspace typecheck pass.

### Direction-conditioned run laws — v51 through v53

Full run reflection can discard physical directional asymmetry. The next
bounded comparison relaxed only the outcome-law assumption: the canonical
tree still shares run partitions, but each physical direction estimates its
own empirical joint law and shrinks toward its reflected pooled counterpart.
For a direction with `n` observations and direction prior `a`, its law is
`n / (n + a) * empirical + a / (n + a) * pooled`. Empty directions use the
pooled fallback. This mixes whole return/extrema/duration/successor atoms;
it does not independently adjust return means or invent successor states.
The prior is an empirical regularizer, not independent observations or a
calibrated Bayesian uncertainty estimate. Reported estimation counts retain
the real number of samples.

`trainEventRunDistribution` in `event-run-model.ts` now supplies exact
reflection and direction-conditioned modes. `--run-direction-prior 100`
selects the latter instead of `--run-symmetry`. The v51 screen used that one
fixed prior, retaining the v45 clock, tree-depth candidates, costs, depth limit,
selection rule, and causal latest-refit protocol. No prior sweep was performed.

| Window | Test return | OHLC drawdown | Trades | Test mean MSE skill |
|---|---:|---:|---:|---:|
| May 2022 choppy | -1.54% | 8.24% | 34 | -0.01509 |
| March 2023 choppy | -1.78% | 2.66% | 2 | -0.02043 |
| January 2024 flat shape | +2.84% | 3.77% | 14 | +0.01265 |

The three windows completed in 62.6s total. Partial pooling restored trading,
but two losses rule out broadening this as a demonstrated improvement.
The useful January behavior is repeated exits from declining runs and later
long re-entry: gross long PnL was +450.09 on initial equity 10,000, with 166.19
fees/slippage and +283.90 net PnL. This is more informative than the earlier constant
long drift exposure, but it is only one inspected case.

May's shorts earned -9.77 gross, then paid 141.85 fees/slippage and 2.62 borrow. The
apparently strong short-entry state predicted -23.67 bps but realized +3.49
bps per visit over eight visits. Most small adjustments restored the exposure
cap; eliminating required cap reductions would change the account contract,
not repair that forecast. March made one losing long round trip, beginning in
a state with +17.30-bps predicted return. Its next event returned -25.38 bps;
the position remained open across subsequent events and lost 153.81 gross,
plus 23.79 fees/slippage. These are weak directional estimates, not evidence that fees
were accidentally omitted from the optimizer. Small visit counts also prevent
treating individual observed sign reversals as precise population estimates.

To check whether sharing partition and estimation outcomes drove these signals,
v52 repeated the same screen with a chronological 50% honest split. The earlier
half chooses the shared partition; the later half estimates all physical and
pooled laws. This simultaneously makes estimation more recent and less data
rich, so it does not isolate those two effects. All three tests made no trades.
May and January failed calibration's active-policy gate; March passed the gate
but its latest refit chose no exposure. This removed both the bad trades and the
useful January behavior; it did not establish profitable generalization.

The existing `forward-market-return-information-2026-08-16.md` is relevant to
the feature question: its strongest spot-flow sign information disappeared by
three seconds, and its longer-horizon joint feature test did not improve the
15m model. `extended-market-information-basis-2026-08-16.md` also confirms that
the five-year spot candle archive lacks taker flow, trade count, and book data.
Those results do not justify loading a large microstructure corpus and assuming
its one-second signal predicts these much longer run events. The separate
rolling-calibration study supports updating forecast probabilities from past
errors, but the global signed-mean correction already failed v25/v26; a future
attempt would need state-conditional joint-law updates and its own causal
evaluation, not a repeat of that rejected correction.

The unresolved issue remains conditional directional information and its
transfer between estimation, calibration, and test periods. More horizon or
prior search alone is not supported. The new estimator passed the focused
physical-drift, full-joint shrinkage, empty-direction, honest-estimation, and
serialization checks; all 33 focused tests and workspace typechecks pass.
Frozen replay v53 reproduced all three v51 metric objects and every byte of
the 591 event trace rows. `reproduction-check.json` records the matching hashes.

### Recent completed-event law screen — v54

The next step tested the prior rolling-calibration hypothesis without first
paying for repeated policy construction. `EventRecentLaw` keeps a bounded
history of complete post-fit events in the fixed observable states. A state's
new law is `(a * base law + sum(recent joint atoms)) / (a + recent state count)`.
All atoms retain return, duration, OHLC extrema, and actual successor state.
The base mass is strictly positive, including protection against underflow of
an already subnormal tail probability. Observations before the base-fit cutoff,
overlapping events, future outcomes, and backward clock calls are rejected.
Exact-reflection and hidden-belief models are not independently updated.

The prequential screen compared no update with four predeclared settings:
global histories of 64/256 completed events and base strengths of 8/32 events.
It selected minimum return-mean MSE on the same preceding calibration period,
then scored only the selected rule after the final model refit. Histories reset
at the refit and across excluded calibration gaps; no training observation is
reused as a new update. NLL here uses the 15-class projection of the actual atom
law for both base and updated forecasts, with the same 1e-12 scoring floor.
It therefore is not directly comparable to earlier pseudocount-smoothed NLLs.

| Window | Calibration-selected update | Calibration MSE skill vs base | Test MSE skill vs base | Test NLL change |
|---|---|---:|---:|---:|
| May 2022 choppy | unchanged | 0 | 0 | 0 |
| March 2023 choppy | last 64 events, base strength 32 | +0.00122 | -0.00271 | -0.01052 |
| January 2024 flat shape | unchanged | 0 | 0 | 0 |

The three-case screen took 2.14s of per-window work (3.88s process time). March
improved duration/direction-class likelihood slightly but worsened the mean
forecast. At its v51 losing long entry on 2023-03-20 09:03 UTC, the state had
zero recent observations, so the +17.30369-bps forecast remained unchanged.
At the later exit it had three observations and the forecast moved from
-8.21252 to -6.26361 bps. This is evidence of a sparse-state adaptation limit,
not a verified alternative trading outcome: **v54 is a forecast screen, not a
backtest**. All candidate updates worsened May and January calibration MSE.
No Bellman-policy integration or broad computation was justified by this screen.

The source model and update settings are saved before scoring test forecasts;
source model hashes and exact implementation snapshots accompany the results.
The final subnormal-tail guard was added after this screen in the snapshot
builder, which the forecast-only screen does not invoke. The 35 focused tests and full workspace
typecheck pass. Future adaptation would need to share information across related
states or learn a predictive regime context, rather than waiting for enough
recent visits to each rare trading state independently.

### Shared recent evidence — v55

The updater now offers two explicit backoff assumptions for direction-conditioned
run models. `parent` uses recent observations from the same parent branch of the
canonical tree and the same physical run direction. `direction` uses every recent
observation with that physical run direction. Both retain the complete observed
joint atom and its actual successor; neither reflects or relabels a borrowed
observation. These are coarser conditional-law approximations, not extra independent
data. The global unique observation count is unchanged when one event informs
several state laws. Forecast traces distinguish local visits from shared visits.

The v55 screen retained all v54 candidates and added the two sharing scopes with
the same 64/256 histories and 8/32 base strengths, for 12 updates plus unchanged.
Selection again used only preceding calibration MSE. All three selected direction
sharing; no parent-scope candidate won.

| Window | Selected history / strength | Calibration mean skill vs base | Test mean skill vs base | Test NLL change |
|---|---|---:|---:|---:|
| May 2022 choppy | 256 / 32 | +0.00115 | +0.00046 | -0.04096 |
| March 2023 choppy | 64 / 32 | +0.00293 | -0.00438 | -0.01766 |
| January 2024 flat shape | 256 / 32 | +0.00370 | -0.01455 | -0.03990 |

The screen took 5.58s of per-window work, 7.77s process time. Shared evidence
improved class likelihood on all three tests, but only May had positive return-mean
skill, and that improvement was small. At March's previously identified bad entry,
35 related observations were available despite zero local visits. Its forecast
fell from +17.30369 to +15.22110 bps, still positive before the negative next event.
Thus a shortage of local visits is not the only issue; this pooling rule does not
provide a reliable reversal predictor. These forecast results do not prove either
improved or degraded trading performance, especially when duration and successor
probabilities also change.

This motivated one bounded May policy test rather than a suite-wide run. Scheduled
policy updates now preserve the original account costs, grids, event clock, and
state mapping, and are applied only at or after their recorded availability. A
regression test verifies that a future negative-law update cannot change the first
positive-law order, then changes the action when it becomes available. Updates
that change the account contract or arrive out of order fail loudly.

### Scheduled Bellman updates — v56

The bounded May replay inherited v55's calibration-selected direction-sharing
rule (256-event history, base strength 32), and rebuilt the Bellman tables after
each 32 completed events. One table set contains all depths 1–8, so calibration
reuses scheduled builds when comparing depths. Updates retain positive base tail
mass and the complete empirical outcome dependence. Each solve freezes its
current law for hypothetical future branches; it does **not** solve the augmented
belief-state Bellman problem with future learning inside every branch.

Calibration selected depth eight with log growth 0.04218 and score 0.03029 after
the same drawdown penalty. There were 33 calibration updates and eight test
updates. Source model hashes are checked against the forecast selection artifact;
update times, counts, conditional means, calibration scores, and executed orders
are saved. The one-window run took 153.85s; this is a bounded diagnostic, not
evidence that sweeping more update schedules would be productive.

| May 2022 outcome | Frozen v51 | Scheduled updates v56 |
|---|---:|---:|
| Net return | -1.54% | -0.57% |
| OHLC-envelope drawdown | 8.24% | 2.55% |
| Trades | 34 | 4 |
| Fees/slippage | 141.85 | 24.01 |
| Gross short PnL | -9.77 | -32.38 |

Updating prevented later repeated entries and reduced turnover and drawdown.
It did not create a profitable policy. Every v56 order preceded the first
refresh at **2022-05-14 22:06 UTC**: short entry at 00:00, two cap reductions,
and exit at 16:50. The resulting -57.07 net PnL was then preserved in cash for
the rest of the week. Thus the remaining loss in this case belongs to the
initial frozen policy; it cannot be attributed to a later update.

The update history intentionally starts empty after the final base refit. In
this slow event clock, collecting 32 outcomes took nearly a day. A justified
next check is to initialize recent state from a strictly pre-window warmup,
with the base estimator ending before that warmup so observations are not counted
again as new evidence. The same initialization must be used in calibration,
and the complete test window must still be scored from its first timestamp.
This has not been tested; skipping the first losing day would not be an acceptable
substitute. No broader policy run was started from this one still-negative result.
All 37 focused tests and the full workspace typecheck pass.

### Separate sign head and one Bellman improvement — v57 through v59

Correction recorded in v338: the user's remembered 75–77% next-second sign
result belongs to the later structured production59/base17 family, not the
ridge precedent below. The causal-attention model records 76.8059% validation
and 81.5012% test direction accuracy; its balanced-loss variant records
75.7012% and 80.3507%. Exact-zero targets are excluded. A separate rerun's
largest-move decile is only about 57% accurate, so raw accuracy does not
establish event-horizon sign quality. See
`structured-next-second-sign-audit-2026-09-04.md` for sources and scope.
V57–v59 did not evaluate these structured models.

The original v57 investigation used two other sign precedents. In
`next-second-return-linear-vs-glu-2026-08-04.md`, the 120-lag linear ridge model
achieved 55.2207% test direction accuracy versus 50.7012% for the one-layer GLU;
the GLU had better MSE. That ridge model was trained on return MSE, not a sign
loss. In `technical-indicator-predictiveness-2026-08-16.md`, a conditional
histogram using latest-return state and EMA acceleration(2s,1s) added 0.0579731
bits and 5.7437 percentage points of held-out **active-sign** accuracy, with
positive information gains in all four annual tests. These are different
metrics/populations, and both predict the next second. Neither establishes
directional skill on the present run-event horizon.

V57 applies the small regularized linear approach to the actual event target:
an L2 logistic head predicts `P(return > 0 | return != 0, current features)`
from the 20 existing physical run features. It does not load the one-second
checkpoint or claim to replicate its exact inputs. Magnitude never enters the
sign-training objective; zero targets are excluded. Train-only normalization,
clipped standardized features, and damped Newton fitting keep this inexpensive.
Penalties 0.01/0.1/1 and probability blends 0.5/1 compete with the unchanged
base on preceding calibration cross entropy. Fit populations and cutoffs match
v51 exactly, including the latest refit. The chosen head is saved before test.

Combination reweights the base atoms as
`P(sign | features) * P(return, duration, extrema, next state | sign, base state)`.
The zero mass and all joint values within each sign remain unchanged. Both
observed signs retain positive support. A missing conditional sign law falls
back to the original kernel. This avoids falsely assuming conditional
sign/magnitude independence, which the existing
`one-second-conditional-sign-magnitude-dependence-2026-08-16.md` explicitly
rejects. It still assumes the base state's within-sign law is adequate.

| Window | Selected penalty / blend | Test active-sign accuracy, base → new | Sign gain, bits/event | Return-mean MSE reduction |
|---|---|---:|---:|---:|
| May 2022 | 0.01 / 1 | 61.94% → 62.69% | +0.004013 | +1.744% |
| March 2023 | 0.01 / 0.5 | 60.36% → 60.81% | +0.002514 | +0.440% |
| January 2024 | 0.1 / 1 | 64.29% → 64.29% | -0.020722 | -2.557% |

The screen took 7.48s of per-window work and scored 268/222/98 complete events.
At March's bad entry, 2023-03-20 09:03 UTC, the up probability changed from
52.7999% to 47.8527% (unblended head 42.9055%). The expected return nevertheless
remained positive, falling from +17.3037 to +10.9608 bps, because the conditional
positive and negative payoffs are unequal. The realized next event was -25.3757
bps. Thus predicting the more likely sign and predicting the wealth-maximizing
action are different tasks even when the sign is corrected at this example.

V58 performs one Bellman improvement with this forecast. Conditional hold-value
tables separate negative, zero, and positive outcomes. Current sign probability
mixes those tables before maximizing over actual feasible trades. Hypothetical
later moves use the frozen base `V_(depth-1)`; the new head is evaluated again
from observed features at each real event. This is a bounded rollout
approximation, **not** recursive propagation of the sign head through simulated
features or evidence of Bellman convergence. At depth one it is the full
one-event problem with terminal cash settlement. Depths 1–8 and cash are
reselected only on preceding policy calibration, with the sign model fixed.

| Window | V51 return | V58 return | V58 drawdown | V58 trades | Selection |
|---|---:|---:|---:|---:|---|
| May 2022 | -1.5425% | 0% | 0% | 0 | cash; every calibration depth inactive |
| March 2023 | -1.7759% | 0% | 0% | 0 | depth 2 active gate; final model stays cash |
| January 2024 | +2.8390% | -2.5313% | 10.4392% | 2 | depth 8 |

V58 took 12.44s of per-window work. January entered long on January 2 at 16:30
UTC and held through terminal settlement, 3,330 minutes. During the January 3
selloff, it repeatedly predicted a rebound: at 10:39/11:12/11:25/11:52/11:56 UTC
the up probabilities were 66.73/67.01/71.90/69.44/72.46%, while the next returns
were -67.34/-70.55/-112.82/-94.99/-238.70 bps. A counterfactual decision audit
using the **same actual account and depth 8** shows that the unmodified law
would exit at these points. Thus the lost exits are caused by the changed
forecast, not solely by reselecting depth 8 instead of 7. Preserve the original
ability to exit declining runs; do not promote this sign override.

V59 sets blend to zero through the new machinery. It reselects the original
depths 8/3/7, reproduces all three returns, drawdowns, fees, positions, and order
quantities exactly over 591 trace events. Maximum action-value difference is
9.98e-18. This rules out the numerical decomposition as the cause of the changed
behavior. All 40 focused tests and the full workspace typecheck pass, including
sign-only target invariance, joint-law preservation, and retaining ruin risk
under highly confident sign forecasts.

The sign hypothesis has useful evidence, but the new head's small aggregate
gains do not establish a profitable strategy. Its calibration also reverses
between January's calibration and test periods. Before enlarging the state
space or running all 28 windows, the next bounded comparison should test
sign probabilities conditional on move-size regimes, with those regime
probabilities themselves forecast causally. Score the resulting **joint** law
and its adverse-move calibration; never provide the realized future magnitude
as an input. The component-head and conditional-dependence notes motivate this
test. It takes priority over the earlier proposed cold-start warmup experiment,
which has not been implemented.

### Size-conditioned sign, fast volatility, and gate calibration — v60 through v64

V60 implements the component factorization motivated above:
`P(size regime | x) * P(sign | size regime, x)`. A training-only absolute
log-return quantile splits ordinary and large moves; a logistic gate forecasts
the regime, and separate logistic heads forecast sign in each regime. Future
size is a label during fitting, never an inference input. The base law supplies
the joint return/duration/extrema/successor distribution within each of four
size/sign groups; its exact-zero mass stays unchanged. Unsupported groups retain
the complete base law. No group was unsupported in these scored windows.

Median/Q75 thresholds, penalties 0.01/0.1/1, and blends 0.5/1 compete with the
unchanged base. Selection uses the same 15-class joint NLL for all candidates,
so different fitted size thresholds do not change the scoring target. All three
windows selected Q75 and full replacement; May/March selected penalty 0.01,
January 0.1. Test joint NLL improved everywhere, while January's mean and adverse
probabilities worsened. Its Q75 threshold was 64.9052 log bps and the large-sign
head had 392 fitting examples.

The important diagnostic is that the large-sign head *does* predict the January
selloff's direction conditionally: at the five previously identified failures,
`P(up | large)` was 28.1/24.9/29.6/32.9/37.3%. However, the gate assigned only
22.9/26.1/32.0/32.8/39.6% to large moves. Small upward rebounds continued to
dominate the combined forecast. This narrows the problem from an unconditional
sign failure to the size gate and its regime context; it does not prove that
these five future outcomes should have been predictable.

The local `global-return-feature-basis-2026-08-17.md` found that completed-minute
range adds 0.00902 primary / 0.01465 transfer bits conditional on its volatility
basis. `recommended-model-input-basis-2026-08-17.md` also retains multiscale
volatility. The current 20 run inputs lack latest-candle range and short realized
volatility, so V61 adds a head-only candidate with `log1p(range-1m-bps)`,
`log1p(RV-5m-bps)`, and `log1p(RV-15m-bps)`. RV is the square root of summed
completed-minute squared log returns. These features stay within the existing
one-day purge; the frozen base state map is unchanged. Both original and extended
bases compete on preceding calibration. Every window selects the extended basis.

| Window | V60 joint NLL change | V61 joint NLL change | V61 mean-MSE reduction | V61 large-down Brier change |
|---|---:|---:|---:|---:|
| May 2022 | -0.011067 | -0.012375 | +1.392% | -0.001564 |
| March 2023 | -0.011807 | -0.016622 | +1.213% | +0.000702 |
| January 2024 | -0.052446 | -0.057380 | -1.958% | +0.001003 |

Changes compare the same v51 base; negative NLL/Brier changes improve the
forecast. The common large-down target is a log return at or below -60 bps,
independent of each fitted Q75 threshold. V60/V61 took 2.73/3.41s of per-window
work. V61 improves joint likelihood but still understates January's adverse risk.

V62 generalizes the conditional Bellman operator from three sign groups to
five size/sign/zero groups. It retains V58's explicit approximation: one current
backup under the new probabilities, with frozen base continuation values after
that; observed features are recomputed at every real event. No expansion to all
28 windows or deeper hypothetical sign states was performed.

| Window | V62 return | Drawdown | Trades | Calibration-selected depth |
|---|---:|---:|---:|---:|
| May 2022 | 0% | 0% | 0 | 3; active gate but final model stays cash |
| March 2023 | -0.0236% | 7.8504% | 2 | 8; long for the entire week |
| January 2024 | -3.8290% | 10.4391% | 5 | 8 |

The three-window backtest took 12.42s of per-window work. January first entered
at 01:50 UTC on January 2, added to full long at 16:30, and stayed through the
selloff. It finally exited at 12:28 UTC on January 3, then re-entered at 13:37.
The selloff's five early forecasts remained +3.07 to +4.61 bps. The later exit
and re-entry lost more than V58's continuous hold. Preserve the original v51
exits instead of treating lower average NLL as automatic policy promotion.

V63 tests a cheaper response to completed surprises, following the repository's
rolling probability-calibration work. It freezes the already selected V61 heads
and compares no update with rolling 8/32-event log-odds intercept corrections,
using strengths 4/16. The objective sums Bernoulli cross entropy of raw gate
forecasts plus `strength * offset^2 / 8`; the offset is bounded to [-8,8]. Only
completed post-fit sizes enter, and only the size gate changes. Histories reset
across excluded gaps and at final refit. Selection again uses preceding joint
NLL, with the original base also eligible. The parent head/config hash and copied
heads preserve provenance; no head is refitted for this comparison.

All three select 32 events, with strengths 16/4/16. Compared with frozen V61,
test NLL changes by -0.003065/+0.011636/+0.010584 for May/March/January. January
mean-MSE skill versus the base deteriorates from -1.958% to -2.883%, and its
large-down Brier score worsens from 0.172261 to 0.174472. At 10:39 UTC, its raw
22.44% large probability is corrected *down* to 20.16%; the 32-event history's
negative offset persists through the initial selloff and becomes positive only
after the 11:56 event. The correction still mixes older conditions with the new
burst. This is a failed forecast screen, not a new trading result. Policy replay
explicitly rejects these rolling-gate artifacts until causal updating is
implemented there; no unsupported static replay may silently ignore the gate.

V64 uses zero blend in the generalized five-group machinery. It reproduces all
three v51 returns, drawdowns, fees, exposures and order quantities exactly over
591 events, with maximum action-value difference 6.29e-18. All 43 focused tests
and full workspace typechecks pass. New checks cover opposed sign patterns in
size regimes, completed-candle feature causality, unsupported-group fallback,
and refusal to update a gate from unfinished or overlapping outcomes.

The next justified hypothesis is a learned change in event-size persistence or
arrival intensity, rather than further global probability rescaling. Recent
completed event sizes and durations could distinguish a burst from the old
32-event average. The earlier full-outcome HMM did not fix choppy-window edge;
a component-specific regime filter must demonstrate incremental held-out gate
skill before another policy integration. This remains a research hypothesis.
Neither the 28-window profitability requirement nor policy optimality is met.

### Completed-event context and utility-based policy retention — v65 through v70

V65 tests observable event-size persistence and arrival pace without adding a
hidden-state model. The head receives ten extra coordinates: history coverage,
previous-event sign/absolute log return/duration, and mean absolute log return,
mean duration, and signed efficiency over the latest 4 and 16 completed events.
At most 16 events are retained, and each whole event must originate within the
preceding day. This keeps the existing one-day purge valid. The first decision
in each fit/calibration/test episode has empty history, and excluded gaps reset
it. Current targets enter the memory only after their completion. The base
partition still receives its original 20 coordinates.

The V61 head is the incumbent. Candidates retain its Q75 threshold rule, blend,
and fast-volatility basis, and compare penalties 0.01/0.1/1 with history added
to all three components. All three windows reject that addition on preceding
joint NLL, so V65's selected forecasts exactly equal V61. No new backtest was
needed. The three-case screen took 2.67s of per-window work.

V66 adds a narrower candidate: refit only the size gate with event history and
preserve both incumbent sign components byte for byte. Components consume
explicit prefixes of the input vector; the gate uses 33 inputs and the sign
components retain 23. May and March again retain V61. January selects gate-only
history with penalty 0.01. Its calibration NLL changes from 1.669377 to 1.668255.
Test NLL changes from 2.159028 to 2.149821, mean-MSE skill versus the original
base improves from -1.958% to -0.786%, and large-down Brier improves from 0.172261
to 0.170874 (base 0.171258). This is a small gain, not recovered mean skill versus
the base. At 11:56 UTC during the selloff, the predicted large-move probability
is 52.43%, versus V61's 44.88%, and expected return falls to +0.23 bps. It still
does not predict an adverse mean before that large decline. V66 took 3.06s.

The bounded January replay V67 uses the exact same completed-event feature
contract. All 98 complete-event probability vectors match the forecast screen
exactly; expected-return differences are at most 3.20e-14 bps. It selects depth
8 and loses 0.30394%, with 11.1803% drawdown and a single full-window long
round-trip. The smaller loss relative to V62 (-3.8290%) comes from removing the
late exit/re-entry, not from restoring the original profitable early exits.

V68 checks the continuation approximation. `--project-continuation` averages
the head's probabilities over **training** feature vectors in each original
finite state and reweights its joint kernels. The Bellman tables are rebuilt
at every depth under this projected law. Current decisions still use the fine
observable head inputs; hypothetical future event memory and candle features
are averaged into the existing states. This is a finite-state projection, not
full simulation of those fine features or proof of optimality. Empty states
keep their base law. Joint returns/extrema/durations/successors remain unchanged
within groups, and zero blend preserves kernel probabilities exactly. The
projection saves both calibration and final policy artifacts with fit cutoffs
and population counts. January selects depth 6 instead of 8 but executes the
same positions and has exactly the same return/drawdown as V67.

The forecast-first workflow could still replace an economically better original
policy merely because a head had lower likelihood loss. The original January
policy's preceding calibration score is 0.089815, versus 0.065636 for the head
with frozen continuation and 0.062395 with projected continuation. V69 therefore
adds the original policy to the final utility comparison. It restores January's
original result, but selects the losing original May policy because the
projected-head candidate is poor there. V70 retains both continuation candidates
in the same preceding-calibration comparison, alongside the original and cash.
All depths 1–8 remain available. Ties prefer the original, then the head with
frozen continuation, then projected continuation. No scored-window result enters
this selection.

| Window | Best original calibration score | Best frozen-continuation head score | Best projected-continuation head score | Selected policy/depth | Test return | Drawdown |
|---|---:|---:|---:|---|---:|---:|
| May 2022 | 0.021092 | 0.035739 | -0.001892 | frozen-continuation head / 3 | 0% | 0% |
| March 2023 | 0.048363 | 0.063430 | 0.130745 | projected-continuation head / 7 | -0.02362% | 7.8504% |
| January 2024 | 0.089815 | 0.065636 | 0.062395 | original / 7 | +2.83900% | 3.7729% |

V70 took 33.04s of per-window work. May places no orders despite passing the
calibration active-policy gate. March holds a full long throughout the week;
its +21.63 gross PnL does not cover 24.00 in fees/slippage, and its intrabar risk
remains substantial. January preserves all 14 original trades, including useful
exits from declining runs. The utility selector is a useful correction, but
these results still do not meet profitable generalization across the suite.

All 45 focused tests and full workspace typechecks pass. The new tests verify
event-history chronology/lookback, unchanged frozen sign components, projected
joint-law preservation, deeper-value changes, and exact zero-blend kernels.
Forecast/replay parity is stored with V67. Source/config/head snapshots accompany
every screen and replay.

The next bottleneck is March's strong calibration utility becoming passive
long exposure in the test. Before expanding computation, inspect the predicted
action-value contributions of endpoint payoffs and successor continuation at
its entries and holds, and compare them with completed out-of-fit outcomes.
Better aggregate likelihood and more features have repeatedly failed to fix
this economic calibration error; another predictor should target the identified
action-value error rather than be promoted on NLL alone.

### Action-value audit, value-aware correction, and head age — v71 through v74

V71 reconstructs the selected March V70 policy on its preceding calibration
and scored week. Every saved scored decision and trace field reproduces exactly.
The new `eventActionValues` diagnostic exposes the same feasible lot-rounded
candidates and interpolated values used by execution. A separate direct backup
integrates each joint atom at the actual current account, separating immediate
log return/order cost from successor value. Completed-outcome backups substitute
the observed next event but retain the saved forecast continuation; they are
**not realized full-horizon wealth**. Event-aggregated borrowing is also an
approximation to the minute-by-minute execution ledger. No audit target trains
or selects the audited policy.

| March diagnostic | Preceding calibration | Scored week |
|---|---:|---:|
| Complete events | 536 | 222 |
| Events with invested selected action | 68 | 222 |
| Mean predicted immediate log reward, invested events | +5.2461 bps | +2.4431 bps |
| Mean realized immediate log reward, same events | +19.4345 bps | -0.0554 bps |
| Mean continuation advantage over immediate cash exit | -5.4899 bps | +4.8968 bps |
| Invested actions favored over cash despite lower one-event terminal value | 20 | 67 |
| Maximum absolute direct-versus-grid value error | 0.001530 bps | 0.00001364 bps |

The grid error is economically negligible here. Holding through mildly negative
next-event forecasts can be rational after exit/re-entry fees. These numbers do
not justify removing continuation or forcing an exit whenever the mean is below
zero. They identify weak immediate edge and a changed continuation incentive,
not an account-arithmetic defect. The first March entry expects +22.13 bps and
realizes -102.76 bps. Across the scored week, ordinary-up mass is forecast at
39.225% versus 37.838% realized, while large-down mass is forecast at 12.440%
versus 13.964% realized. Conditional log-return magnitudes within the four groups
are fairly close (approximately -49/+49/-83/+84 bps predicted). Relatively small
probability errors can therefore erase the entire estimated economic edge.
The audit took 21.84s; it is descriptive, without a significance claim.

The local reinforcement-learning notes recommend estimating values from
transitions. A more specific external result is
[Value-Aware Loss Function for Model-based Reinforcement Learning](https://proceedings.mlr.press/v54/farahmand17a.html),
which accounts for the planner's value function when estimating its transition
model. [Iterative Value-Aware Model Learning](https://papers.neurips.cc/paper_files/paper/2018/hash/7a2347d96752880e3d58d72e9813cc14-Abstract.html)
fits against the value functions produced during approximate value iteration.
These papers motivate an experiment; their guarantees do not establish
profitability for this undiscounted, partially observed trading approximation.

V72 implements one small value-aware correction to the existing size gate and
two sign components. It retains their training normalization, feature prefixes,
all four positive outcome probabilities, and every within-group joint path.
The fit minimizes squared Bellman **advantage over cash** error at signed maximum
exposures and depths 1/4/8. The target is the observed event's log holding return
plus the frozen continuation difference between that inventory and cash. This
removes action-independent continuation from the loss. Conditional group values
come from V70's training-only projected policies. The fit uses the original
purged training population; target RMS normalization uses training only. L2
penalties 0.01/0.1/1 anchor all parameter changes, including intercepts, to the
incumbent. This is a single fixed-planner update, not converged IterVAML.

All corrections lose to the incumbent on March's preceding calibration value
MSE: unchanged 0.0000419010; penalty 1 gives 0.0000419457, penalty 0.1 gives
0.0000422221, and penalty 0.01 gives 0.0000436789. The selected probabilities
remain unchanged, so a duplicate policy backtest is unnecessary. The bounded
screen takes 5.13s. This rejects the tested correction; it does not reject
value-aware learning with other data or representations.

V73 isolates refit components at the same March depth 7 and scored features.
Each head uses its own fitted magnitude threshold: 70.000 log bps for the older
head, 68.244 for the latest. Thus “head age” includes the gate, both sign heads,
normalization, and size threshold, rather than coefficients alone.

| Continuation model | Current head | Test return | Drawdown | Trades | Long minutes |
|---|---|---:|---:|---:|---:|
| Older selection fit | Older selection fit | -2.34983% | 3.3954% | 2 | 991 |
| Older selection fit | Latest fit | -2.34983% | 3.3954% | 2 | 991 |
| Latest fit | Older selection fit | +2.91489% | 6.0868% | 4 | 9,880 |
| Latest fit | Latest fit | -0.02362% | 7.8504% | 2 | 10,080 |

The profitable diagnostic exits at March 22 18:40 UTC and re-enters at 22:00.
At the exit, its expected return is -19.55 bps versus the latest head's -12.17.
Its next single event actually rises 62.89 bps; the benefit comes from remaining
out during the following decline. This is a discovered component interaction,
**not permission to select the profitable row using its test return**. V73
includes the full V71 audit and all four traces and takes 26.68s.

V74 tests a causal version of that lead. A lagged head always ends training
`calibrationDays + 1 = 22` days before the respective evaluation start and uses
`trainDays - 1 = 119` days of history. The same rule applies in calibration and
test. Its final head exactly reuses the older saved selection fit. The lagged
head competes with original and projected continuation, alongside all three
V70 incumbent policy choices and cash, at depths 1–8. The initial implementation
explicitly limits this option to fixed size/sign heads without event-history
or value-aware corrections. It does not silently reuse incompatible fits.

March's best preceding calibration scores are: latest projected head 0.130745
(depth 7), latest head with original continuation 0.063430 (depth 8), delayed head
with projected continuation 0.057212 (depth 8), original policy 0.048363 (depth 3),
and delayed head with original continuation 0.024208 (depth 7). All 24 incumbent
calibration scores equal V70 exactly. The delayed head loses selection; March
therefore remains -0.02362% with 7.8504% drawdown. V74 takes 13.87s. Neither the
profitable diagnostic nor the rejected delay is promoted.

All 47 focused tests and full workspace typechecks pass. New tests compare
action values with a closed-form binary log bet, exclude over-cap holding, and
verify that value-aware fitting learns synthetic signed utility while leaving
the incumbent unchanged and retaining every outcome group.

The next justified step is to evaluate refit/update choices at several earlier
forecast origins, using only information available at each origin. One 21-day
calibration interval strongly favors a policy that fails to transfer, and both
new corrections were rejected there. A small rolling-origin comparison should
measure that instability before another representation change or wider compute
run. The 28-window profitability requirement and policy optimality remain unmet.

### Rolling refit selection and conditional severity — v75 through v77

V75 evaluates the refit rules at three consecutive 21-day origins before
March's scored window: January 14, February 4, and February 25, 2023. This follows
the chronological principle in the local rolling-forecast experiments and
[Forecasting: Principles and Practice, time-series cross-validation](https://otexts.com/fpp3/tscv.html):
each fit contains only observations preceding its forecast origin, and accuracy
or utility is assessed across multiple origins. This implementation compares
refit rules within the existing tree/head hyperparameters. Their prior research
selection is not an untouched outer test.

Five families compete at depths 1–8: original joint law; current head with
original continuation; current head with projected continuation; delayed head
with original continuation; and delayed head with the current head's projected
continuation. Fresh fits use the trailing 120 days and completed targets strictly
before the origin. Delayed heads use 119 days ending 22 days before the origin.
The same rules apply in validation and final deployment. This also removes the
old one-day difference between fresh validation and deployment fit cutoffs.
Size thresholds, normalization, partitions, kernels, and Bellman values are all
refitted from earlier data at each origin. History-only gate inputs retain the
existing episode/gap resets. The current completed origin close supplies the
price grid reference, rather than an unseen next open.

Selection maximizes the mean of each origin's log growth minus 0.1 times its
maximum fractional drawdown. All three origins receive equal weight. Cash
remains eligible; original-policy candidates win exact ties. Each origin starts
with 10,000 in cash and includes terminal settlement. The final test result is
unavailable to selection. Models, fit support endpoints, all 40 candidate
scores per origin, selected-policy traces, and final artifacts are saved.

March chooses the delayed head with original continuation at depth 8. Its
three utility scores are -0.017151, -0.079772, and +0.149470, averaging +0.017516.
Only one origin is profitable, and score standard deviation is 0.096745. The
result therefore remains unstable even though pooling changes the selection.
The scored week returns +0.09216%, with 6.3627% drawdown and five trades. It exits
at March 22 18:40 UTC, re-enters at 20% exposure at 22:00, and adds to full exposure
on March 23 at 15:00. It avoids the selloff but misses much of the rebound. The
final base policy, fresh head, and delayed head exactly equal the corresponding
V51/V66 saved artifacts; the changed selector, not different deployment weights,
causes the changed trades. V75 takes 40.15s.

V76 applies the same three-origin procedure to May and January:

| Window | V70 reference return / drawdown | Rolling selection | Rolling return / drawdown | Profitable validation origins |
|---|---|---|---|---:|
| May 2022 | 0% / 0% | delayed head, original continuation, depth 3 | +0.96929% / 0.8590% | 1/3 |
| March 2023 (V75) | -0.02362% / 7.8504% | delayed head, original continuation, depth 8 | +0.09216% / 6.3627% | 1/3 |
| January 2024 | +2.83900% / 3.7729% | current head, original continuation, depth 6 | -0.92275% / 11.1803% | 3/3 |

May's policy makes one long round trip lasting 77 minutes, earns 121.05 gross,
and pays 24.12 in friction. January enters full long immediately, cuts to 60%
only at January 3 12:28, and restores full exposure at 13:37. Those late changes
lose more than simply holding; the useful original January exits are not
preserved. All three final model/head reproduction checks pass for these cases
as well. May takes 70.96s and January 39.43s. V75/V76 are **not promoted over V70**:
two improved research windows do not justify replacing the profitable January
behavior, and profitable validation origins do not establish reliable transfer.

V77 investigates the remaining January failure inside the size/sign model. The
head can raise the probability of a large move, but its within-group joint law
still comes from the original finite state. The audit bins current five-minute
realized volatility at training-only 50th/90th/99th percentiles and compares
conditional magnitudes and extreme-event frequencies. Realized sign/size is
used only to stratify completed errors, never as a forecast input. “Extreme”
means absolute log return at least twice the fitted large-move threshold.

For the final January model, the 90th–99th percentile volatility band is
49.51–229.45 bps of five-minute realized volatility:

| Population in that band | Total events | Large down events | Predicted large-down severity | Realized large-down severity | Predicted extreme frequency | Realized extreme frequency |
|---|---:|---:|---:|---:|---:|---:|
| Final training | 141 | 27 | 81.44 bps | 101.11 bps | 0.809% | 5.674% |
| January test | 20 | 6 | 81.59 bps | 153.12 bps | 0.763% | 15.0% |

In ordinary lower-volatility training states, large downward moves average
approximately 76 bps, versus predictions around 79–80. The model therefore
misses a state-dependent change in severity, not merely a global magnitude
offset. Correlation between current five-minute volatility and the magnitude
of a subsequent large event is 0.348 in final training and 0.372 in the test.
These are descriptive associations with small tail populations, not proof of
out-of-sample forecasting skill.

At January 3 11:56, the head assigns 52.43% to a large move, but conditional
downside severity is only 84.81 bps; the realized log return is -241.60 bps.
At 12:07 and 12:09, conditional downside severity is 75.84 bps, versus realized
returns of -183.69 and -258.73 bps. The preceding 21-day calibration contains
306 events but only six above its training 90th-percentile volatility threshold,
and no extreme events at all. This limits what that calibration can establish
about burst behavior. V77 takes 0.84s and fits no new trading model.

The next justified experiment is a small severity model conditional on known
volatility and candidate sign, integrating over signs at inference. It should
change probabilities within the large-move joint groups while preserving their
returns, excursions, durations, successor links, and supported tails. Both
ordinary-state errors and burst-state errors need chronological validation.
There is no evidence yet that this correction improves policy utility; existing
policies must remain in the economic comparison. All 48 focused tests and full
workspace typechecks pass. The new test verifies chronological origin boundaries
and that selection uses all origins rather than the best historical interval.

### Volatility-conditioned severity and exact current backups — v78 through v81

V78 fits a three-parameter conditional mean for excess large-event magnitude.
For threshold `T`, the target is `abs(log(1+r))*10000/T - 1`; the mean is
`exp(intercept + slope*z(log1p(RV5)) + contrast*candidateSign)`. Both candidate
signs are evaluated at inference and integrated under the existing probabilities.
Future realized sign and size are training labels only. The loss `mu-y*log(mu)`
targets a nonnegative conditional mean without assuming continuous magnitudes
are Poisson counts. Standardization, threshold and coefficients use only fit
data. Calibration compares penalties 0.01/0.1/1 and blends 0.5/1 with unchanged.

An exponential tilt changes probabilities only within each large/sign group.
Group masses and all joint returns, excursions, durations and successor states
remain intact. Targets outside existing support are clamped; even underflowed
positive ruin tails retain positive mass. This estimates a mean and chooses a
minimally tilted existing conditional law, not a newly learned full tail shape.

January selects penalty 0.01 and full blend. Calibration conditional magnitude
MSE falls from 226.18 to 156.23 bps squared, with only 62 large events. Its three
high-volatility large events worsen from 363.50 to 448.08. The refit has 392 large
training events. On the already examined January window:

| Metric | Frozen size/sign head | Added severity |
| --- | ---: | ---: |
| Large-event magnitude MSE, 31 events | 2651.67 | 1982.98 |
| High-volatility large-event MSE, 13 events | 5906.83 | 4438.95 |
| Signed-return MSE | 0.0000524915 | 0.0000522510 |
| Extreme-move Brier score | 0.039938 | 0.035825 |
| Joint 15-class NLL | 2.14982 | 2.17082 |

The magnitude improvement is 25.22%, but the signed-mean improvement is only
0.458%, and joint NLL worsens. Group-mass differences are at most 3.33e-16;
four conditional means hit support limits. The 11:56 forecast still has a
slightly positive mean (+0.185 bps) before a realized -238.70 arithmetic-bps
move. Increasing both signs' magnitude does not supply missing downside odds.
The screen takes 2.64s.

`buildEventKernelLookahead` / `decideEventKernel` add an exact current backup
for arbitrary probability changes on the same joint support. They lazily cache
per-atom holding-plus-continuation values at account grid cells, never forecast
weights. Every decision weights them with its current conditional law. Future
values remain those of the incumbent policy, so this is one improvement step,
not full recursion through future volatility features. Unit checks reproduce
the compiled and sign-conditional operators, reject changed paths, and check
that changed forecast weights cannot reuse a stale weighted value.

V79 compares original, frozen-continuation head, projected-continuation head,
and severity with each continuation, at depths 1–8. All 24 incumbent calibration
scores reproduce exactly. Original depth 7 still wins, preserving January
**+2.8390% return, 3.7729% drawdown, 14 trades**. The best severity candidate is
projected depth 6, calibration score 0.067252 versus original 0.089815. It improves
on projected-head calibration score 0.062395 but cannot justify replacing the
original policy. The complete replay takes 14.55s; lazy caches contain 7105 and
7449 account/state/depth cells for the two severity families.

V80 uses zero severity blend. All 16 severity-family calibration scores exactly
match their head-family controls, and the selected test trace is byte-identical
to v79. V81 separately audits each family's best calibration depth on the test
window; it does not select from test returns:

| Family | Depth | Test return | Drawdown | Trades |
| --- | ---: | ---: | ---: | ---: |
| Original | 7 | +2.8390% | 3.7729% | 14 |
| Head, original continuation | 8 | -0.3039% | 11.1803% | 2 |
| Head, projected continuation | 6 | -0.3039% | 11.1803% | 2 |
| Severity, original continuation | 8 | -2.0855% | 10.4392% | 3 |
| Severity, projected continuation | 6 | -2.7683% | 10.4394% | 5 |

The severity variants start smaller and miss part of the earlier rise, but still
hold through the crash. Projected severity reduces long exposure to 0.8 only at
12:20 on January 3, after the decline, and restores it two minutes later during
the rebound. The original policy's useful 10:39 exit is absent. Zero-tilt replays
at those two depths reproduce all 198 decisions' quantities and equity ledgers
exactly; maximum value differences are 8.24e-18 and 9.11e-18. These results reject
severity as a trading replacement despite its better conditional magnitude
forecast. The next bounded ablation separates the learned gate from learned
signs, allowing either to retain the original state's other conditionals.

### Separating size and sign replacement; state support — v82 through v83

V82 adds two explicitly separate ablations to the 24 v70 incumbent choices:
replace only the size gate, or replace only the two conditional signs. Both use
the original continuation and depths 1–8. No head is retrained. The gate-only
case retains the base conditional sign probabilities in each magnitude regime;
signs-only retains the base size probability. Zero mass and complete within-group
joint paths remain unchanged. Selection still uses preceding calibration log
growth minus 0.1 times drawdown, with cash eligible. A component-retention test
also checks unsupported groups and exact all-head equivalence.

| Window | Calibration winner | Test return | Drawdown | Trades |
| --- | --- | ---: | ---: | ---: |
| May 2022 | Signs only, depth 8 | 0% | 0% | 0 |
| March 2023 | Projected head, depth 7 | -0.0236% | 7.8504% | 2 |
| January 2024 | Gate only, depth 7 | -2.4875% | 11.5361% | 25 |

All incumbent calibration scores reproduce exactly. January gate-only wins
calibration score 0.110377 versus original 0.089815, but loses on test. It delays
the useful January 3 exit from 10:39 to 11:12: gate replacement raises the 10:39
expected return from -10.92 to -4.61 bps even though conditional signs are
unchanged. It enters short at 11:25 and captures part of the first decline.
However, it reverses long at 12:03 before further -182/-255-bps moves, reverses
short at 12:10 just before the rebound, and repeatedly changes sides afterward.
There are seven reversals, 252 short minutes, and one canceled order. Long PnL
is -22.24, short PnL +38.12, fees 264.47, and borrowing 0.17 on initial 10000.
The small aggregate gross gain does not pay execution costs. V82 is not promoted
over v70. Its three windows take 17.51s, 9.56s and 9.09s.

V83 extends the severity audit with observable base-state/volatility-band
counts, mean returns, durations and next volatility bands. These are descriptive
training/test diagnostics, not fitted states or a selection rule. Its reported
standard error is the ordinary sample standard deviation divided by sqrt(n),
without serial-dependence adjustment; it is not a valid confidence interval for
these event sequences. The audit takes 0.88s.

January final training has 1565 events, with 43 observed state/band combinations.
The state at 10:39–11:25 (leaf 13, volatility band 1) has 90 training events,
58% positive outcomes but **-9.42 bps mean**. Its average head prediction is
**+4.00 bps**, whereas the base predicts **-10.92 bps**. This is a concrete case
where majority sign and economic expectation differ. At higher volatility,
leaf 13 has 12 events and an empirical mean -24.28 bps with a naive 29.64-bps
standard error; the top band has only one event. Leaf 0/top band has two events,
and leaf 16/top band has none. Leaf 16/band 2, reached before two further crash
moves, has 14 training events, +20.00-bps mean and 122.7-minute average duration.
Those two test events instead last two minutes and one minute. Fine subdivision
would therefore create poorly supported future paths, not establish a reliable
burst-state law.

The next justified step is a bounded earlier-history census and pooled burst
forecast screen, with separate chronological origins, before enlarging the
planner or adding state dimensions. Pooling must demonstrate transferable sign,
magnitude and duration information; predictable volatility by itself has not
produced profitable switching. Preserve v70 as the economic reference and its
January exits. All 52 focused tests and full workspace typechecks pass. The
overall reliable-profitability goal remains unmet.

### Pooled heads and complete volatility-conditioned paths — v84 through v92

The repository's recommended input-basis notes distinguish feature lookback
from estimation history and allow longer learned-model history only after
chronological validation. V84 therefore reuses the three existing v76 origin
models (October 31, November 21 and December 12, 2023) and their independently
fitted recent heads. Added history ends before each origin; the exact recent
120-day event chain is retained, while older completed chains are appended from
the preceding year or from all cached history beginning July 2021. Thresholds,
component penalties and feature prefixes remain the recent head's values.

The January final fit grows from 1565 events/392 large events to 21,458/7328.
Above the recent training 90th-percentile RV5 threshold, coverage grows from
156 events on 48 days to 4207 events on 595 days. Critical physical states 0,
13 and 16 grow from 15/13/14 high-volatility examples to 344/343/607. These are
coverage counts, not independent sample sizes. No evaluation label is included
in its origin's fit. All history extraction takes 3–7s and each head-fitting
origin less than one second.

V84 compares replacement of large-move sign alone or all three heads, using
one-year/all-earlier training. All four pooled alternatives lose to the recent
head on mean validation joint NLL: recent 1.828016, best pooled 1.828829. V85
isolates the gate as well, with no additional model family. The all-history gate
retaining both recent sign heads wins mean NLL 1.826277. On January it changes
NLL 2.14982 -> 2.13183 and log-duration MSE 1.88792 -> 1.71588, but signed-return
MSE slightly worsens, 0.0000524915 -> 0.0000527101.

V86 tests that selected gate with original and projected continuation, retaining
all 40 v76 policy/depth candidates. Original depth-8 controls reproduce exactly
at each origin. The best pooled policy is projected depth 8, score 0.057147
versus the incumbent recent-head/original-continuation depth 6 score 0.092026.
The incumbent remains selected and reproduces its January -0.92275% result.
This does not replace the v70 reference, which was already better in January.
The economic comparison takes 22.44s. Pooled sign or gate training alone has
not fixed the behavior.

The next model changes the full joint law. `event-volatility-law.ts` crosses
the existing canonical run partition and physical direction with two observed
RV5 bands, using a cutoff fitted on recent training. Every historical path
retains its return, low/high excursion, duration and **observed next RV5 band**.
Bellman recursion can therefore transition between quiet and burst states;
it no longer treats fine volatility only as a current-step adjustment. Quiet
empirical paths initially use recent data; high-volatility paths pool older
episodes. The fixed shrinkage prior has strength 100.

V87 exposed a compression defect: global return-quantile representatives erased
some observed joint class/successor combinations. Depending on origin, 70–100
state/class cells lacked support even though another state in the same band
contained that class. V88 replaces this with stratification by joint 15-class
label and successor state. Each stratum retains its exact total prior mass,
worst low/high excursion paths, and a median representative for remaining mass.
All actual leaf observations remain exact. This approximates only the prior,
and never invents an artificial return/extremum/duration tuple. V89 adds 10%
global-prior mass to the 90% same-band prior: a quiet state must not categorically
exclude a shock merely absent from its smaller conditional sample. The empirical
variants have no unsupported evaluated classes after this change.

Although burst forecasts improve, replacing the quiet empirical law harms
overall validation NLL. V90 adds `retainQuietEventLaw`: quiet-state kernels
retain the original full return/extrema/duration/parent-successor marginal.
Their next volatility bands are recovered from the actual recent training
paths that generated each atom, including reciprocal paths. Identical path
fingerprints (initially 12 significant digits for floating returns/extrema; the
machine-precision provenance correction in v98 below replaces this) with different
next bands split mass by observed counts. Missing provenance fails loudly.
This preserves current quiet-state forecasts while allowing future Bellman
branches to enter the separately learned high-volatility law.

The hybrid wins the three-origin forecast comparison: mean NLL **1.809618**
versus recent head 1.828016, original 1.851866 and fully replaced pooled law
1.888693. On the already examined January window:

| Metric | Original law | Recent head | Hybrid joint law |
| --- | ---: | ---: | ---: |
| Joint NLL | 2.21641 | 2.14982 | **1.84957** |
| Signed-return MSE | 0.0000520820 | 0.0000524915 | **0.0000517696** |
| Log-duration MSE | 2.14841 | 1.88792 | **0.88948** |
| High-volatility log-duration MSE | 5.83614 | 4.99478 | **0.90113** |

The hybrid retains one unsupported original-law class in the first validation
origin because its quiet marginal is deliberately unchanged. The new pooled
burst law has none. The final model has 36 states; the earlier-origin partitions
have 42. Forecast extraction, fitting and scoring all four phases takes 4.57s.

V91 builds actual recursive policies at depths 1–8 and preserves all original
rolling selection candidates. Incumbent replay controls again reproduce exactly.
Within the new family, depth 1/cash wins the configured calibration utility.
The best active choice is depth 4: mean log growth 0.006675, but mean
log-growth-minus-0.1-drawdown score -0.002168. The overall incumbent remains
selected. Longer depths increase validation turnover and generally worsen
results; increasing recursion depth is not the next justified experiment.
The complete policy comparison takes 21.19s.

V92 disables the cash gate only for a clearly labeled diagnostic at depth 4,
the best active depth chosen from calibration, and reproduces its earlier
origin outcomes exactly:

| Start / phase | Return | Drawdown | Trades | Fees on initial 10000 |
| --- | ---: | ---: | ---: | ---: |
| October 31 validation | +1.8675% | 6.0226% | 28 | 342.47 |
| November 21 validation | +0.2040% | 11.3869% | 34 | 308.87 |
| December 12 validation | -0.0516% | 9.1191% | 59 | 659.49 |
| January 2 test | **+4.0135%** | **3.8873%** | **10** | **122.82** |

January preserves the original January 3 10:39 exit and 13:37 re-entry. It avoids
the original policy's additional afternoon exit/re-entry sequence, keeping the
rebound exposure and paying fewer fees (122.82 versus 166.19). It places no new
orders in the 25 high-volatility events, and takes no shorts. All January profit
is long PnL: 524.17 gross less fees. The result improves the original +2.8390%
January diagnostic, but is **not an economically selected replacement**.

The earlier origins reveal the remaining failure: substantial gross trading
profits are consumed by frequent quiet-state switching. December has 656.24
gross long-plus-short PnL and 659.49 fees. November has a small positive terminal
return with 11.39% drawdown. Forecast likelihood and duration accuracy therefore
still do not establish reliable economic performance. Next check the same
hybrid construction on the other two choppy windows and inspect losing
validation trades; avoid selecting a policy just because January improved.

All 54 focused tests and workspace typechecks pass. New checks cover completed
RV5 availability, unchanged original feature prefixes, observed successor-band
transitions, exclusion of older quiet paths, rare class/successor retention,
serialization, quiet-law marginal preservation and matching depth-1 values.
The full profitability goal remains unmet; no live strategy was replaced.

### Volatility-law transfer and action attribution — v93 through v104

The next screen applies the same frozen construction to May 2022 and March
2023. No additional quantile, prior strength, action depth or feature grid is
introduced. V93/v94 build the pooled-head reference populations needed by the
joint-law screen, using only completed targets before each origin. May's
available older history is less than one year, so its year/all populations are
identical; the selected `year-all` head modestly improves test joint NLL from
2.04799 to 2.04601 but worsens signed-return MSE. March selects `all-all`: NLL
2.35068 to 2.33049, signed MSE 0.000038866 to 0.000038553. Neither is promoted
from forecast scores alone.

V95 found a provenance lookup defect in the first May origin. One reciprocal
path's high excursion differed from its generating observation by
8.673617379884035e-19. The two representations were on opposite sides of a
12-significant-digit rounding boundary. This was a failed run, not a model
result. `retainQuietEventLaw` now uses exact round-trip numeric keys first;
unmatched atoms are checked against actual observations with the same duration
and parent successor. All return/low/high coordinates must agree within 32
machine epsilons relative to their magnitude (with a 1e-8 scale floor). A
nonmatching path still throws; no successor band is guessed. Tests accept an
ulp perturbation and reject a material excursion change. V99 repeats January:
the full serialized hybrid model is **exactly identical in all four phases**
to v90. The correction changes no prior January result.

The complete joint forecast screens take approximately four to five seconds
each. Both select the hybrid on the same three earlier origins:

| Test window | Recent head NLL | Hybrid NLL | Recent head log-duration MSE | Hybrid log-duration MSE | High-band events |
| --- | ---: | ---: | ---: | ---: | ---: |
| March 2023, v96 | 2.35068 | **2.17738** | 1.42494 | **1.21621** | 19 / 222 |
| May 2022, v98 | **2.04799** | 2.06359 | **0.71078** | 0.73109 | 6 / 268 |

March's high-band log-duration MSE falls from 3.91392 to 0.90829. Its signed
MSE does not improve: 0.000038866 to 0.000039349. May's aggregate forecast
gain does not transfer; its hybrid selection is driven by the third origin,
which contains 291 high-band events, versus 31 and 21 in the first two. The
final May evaluation contains only six such events. Forecast NLL therefore
does not supply evidence for a more reliable directional edge.

V97 and v100 rebuild depths 1–8 while retaining all 40 original candidates.
Every original head/base depth-8 control reproduces exactly in each origin.
Overall selection remains the older lag-head candidate in both windows:

| Window | Overall selection | Test return | Drawdown | Trades | Best active hybrid depth |
| --- | --- | ---: | ---: | ---: | ---: |
| March | lag-base, depth 8 | +0.09216% | 6.36265% | 5 | 1 |
| May | lag-base, depth 3 | +0.96929% | 0.85898% | 2 | 2 |

These are incumbent reproduction results, not gains caused by the new law.
The hybrid family is rejected by its own calibration cash gate. The complete
policy replays take 14.81s and 26.00s respectively. V101/v102 inspect the best
active hybrid depth without applying that gate, exclusively as a diagnostic:

| Window / phase | Return | Drawdown | Trades | Fees |
| --- | ---: | ---: | ---: | ---: |
| March, January 14 origin | -0.10080% | 1.45200% | 4 | 48.17 |
| March, February 4 origin | +0.18453% | 0.16853% | 6 | 14.44 |
| March, February 25 origin | 0 | 0 | 0 | 0 |
| March test | 0 | 0 | 0 | 0 |
| May, March 12 origin | -1.05524% | 2.20591% | 6 | 28.69 |
| May, April 2 origin | 0 | 0 | 0 | 0 |
| May, April 23 origin | -0.97257% | 1.94299% | 10 | 119.31 |
| May test | -1.94092% | 3.31189% | 23 | 188.72 |

May test gross PnL is -4.89 before 188.72 fees. All orders are in quiet states.
The hybrid's high-volatility correction alone cannot resolve the weak quiet
directional law. The March hybrid simply does not find a fee-covering test
trade. Neither outcome establishes profitability.

V103/v104 add `--actions` to the existing volatility-policy audit. For every
executed order whose next event completes inside the phase, it reconstructs
the exact feasible, lot-rounded action set; the selected value must reproduce
the saved order within 1e-12. It compares the selected action with holding
inventory and cash, separating immediate expected log reward, continuation,
and terminal-settlement value. Realized one-event backups keep the forecast
value function at the observed successor: they are **not** realized full-path
profit or an oracle policy. The diagnostic uses close-price aggregate holding
costs, whereas the backtest continues to execute next-open orders and accrue
costs minute by minute. Forced cap-restoration decisions have no feasible hold
counterfactual and are reported separately. Final settlement and incomplete
last events are absent from this action attribution.

| Validation origin | Completed orders with feasible hold | Mean forecast Q advantage over hold, bps | Mean realized next-event reward advantage, bps |
| --- | ---: | ---: | ---: |
| January model, October 31 | 28 | +6.440 | -0.879 |
| January model, November 21 | 32 | +3.969 | -8.209 |
| January model, December 12 | 44 | +8.655 | -1.285 |
| May model, March 12 | 6 | +0.852 | -6.022 |
| May model, April 23 | 10 | +0.988 | -5.750 |

The maximum absolute difference between interpolated Q and direct current
account evaluation is 0.03525 bps in the January validation audit, and 0.00069
bps in May validation. This is much smaller than the forecast-versus-outcome
error. There is no evidence here for increasing grid density.

The entry/exit decomposition matters. In the May March-12 origin, all three
entries have only +0.0354 bps forecast advantage, lose 17.85 bps on the next
event on average, and none has a positive realized one-event advantage. The
corresponding exits help by +5.80 bps on average. January's December-origin
entries predict +9.21 bps Q advantage but average -6.02 bps realized immediate
advantage; seven reversals instead average +25.36 bps realized advantage.
Simply suppressing every reversal would discard useful behavior. Suppressing
every low-advantage change is also not established: in the November origin,
the 0.1–1 bps advantage bucket has positive realized immediate advantage, while
the 5–20 bps bucket is negative. These are small, dependent samples, not a
monotonic calibration curve from which to choose a threshold.

The January test still preserves its useful pre-crash exit and profitable
rebound exposure: +4.01346%, ten trades. Its four completed exits average
+4.03 bps realized immediate advantage; its five entries average -10.40 bps
over their first event, despite the profitable whole episode. This illustrates
why one-event hindsight cannot replace the requested multi-event utility.

Repository follow-up: the Kronos experiment of August 6 already implements
same-direction confirmation/hysteresis; `Position management.md` describes
the transaction-cost no-trade region. Those are possible mechanisms, not
evidence for adding a post-hoc confirmation gate to this policy. Costs and
holding inventory are already inside its Bellman maximization. The next
correction should address quiet-state directional reliability or the joint
law's representation, and validate it on earlier origins. A threshold that
merely turns the losing model into cash does not meet the goal. If uncertainty
or a persistence state is added, it should have an explicit interpretation in
the recursive policy rather than silently overriding the chosen action.

All 54 focused tests and workspace typechecks pass after the provenance fix.
No model is promoted in this iteration. The all-window profitability objective
remains unmet.

### Complete priors and positive observed-path quadrature — v105 through v116

The next audit isolates representation error before changing the statistical
estimator again. `trainEventRunDistribution` accepts a positive `priorLimit`
(default still 128); setting it to the number of training rows retains the
entire empirical prior. `trainEventVolatilityLaw` accepts `compressPrior:false`
for the corresponding complete volatility prior. V105/v106 reconstruct every
saved run and hybrid control **exactly**, then compare complete priors using
the same observations, tree partitions, shrinkage strengths and origins.
These are exact finite training priors, not exact market forecasts.

Immediate-return distortion is small. Across January phases, the original
run prior's maximum state mean error is 0.025–0.089 bps; May's is
0.018–0.075 bps. This is not the source of the much larger directional forecast
errors in v103/v104. However, the original return-only quantile approximation
distorts successor-state probabilities and omits some observed extreme paths.
For the hybrid January state space, maximum successor total-variation error
is 0.049–0.073; the evaluation-visited average is 0.012–0.023. The complete
prior also removes the unsupported October validation outcome caused by
compression. January hybrid October NLL changes from 1.85440 to 1.79391;
final NLL changes from 1.84957 to 1.84786. Other phases have mixed, small
forecast changes. The volatility prior already stratifies by successor, so
its successor mass is preserved; its remaining compression error concerns
within-stratum outcomes and durations.

V107/v108 perform a paired economic check at the **already selected active
depth**, with no new depth search: January depth 4, May depth 2. Calibration
controls reproduce the saved returns, drawdowns and order counts exactly.

| Model / phase | Old compressed return | Complete-prior return | Complete-prior drawdown | Trades |
| --- | ---: | ---: | ---: | ---: |
| January, October 31 origin | +1.86750% | +1.86750% | 6.02259% | 28 |
| January, November 21 origin | +0.20400% | **+1.59248%** | 11.43231% | 34 |
| January, December 12 origin | -0.05162% | -0.05162% | 9.11915% | 59 |
| January test | +4.01346% | +4.01346% | 3.88732% | 10 |
| May, March 12 origin | -1.05524% | **0** | 0 | 0 |
| May, April 2 origin | 0 | 0 | 0 | 0 |
| May, April 23 origin | -0.97257% | -0.97257% | 1.94299% | 10 |
| May test, active diagnostic | -1.94092% | -1.94092% | 3.31189% | 23 |

The January fixed-depth candidate's mean calibration score rises from
-0.00216808 to **+0.00240393**. Its paired comparison with the old active depth
and cash passes, but it still does not beat the full incumbent selector's
0.092026 score. The November change starts at 15:47 on November 21: the new
policy reduces long exposure to about 0.8 instead of 0.6. At 04:07 the next
day it takes about -0.2 rather than -0.4 short exposure. Small changes to the
distribution can alter sizing near a flat optimum; the resulting improved
episode does not demonstrate a newly strong sign forecast. May's paired
calibration score improves from -0.00817689 to -0.00390542 and still selects
cash. Its active test loss is unchanged. No final-window improvement was used
to select these choices.

Complete priors take roughly 16–20 seconds per phase to build and replay in
this paired check. A more accurate compact representation is therefore useful
before scaling. The discrete positive quadrature result in
[Piazzon, Sommariva and Vianello, *Caratheodory–Tchakaloff Subsampling*](https://arxiv.org/html/1611.02065)
provides an applicable construction: an empirical measure can be represented
by a smaller positive-weight subset preserving a finite set of function
moments. The new JS `event-quadrature.ts` applies a streaming null-space support
reduction, with no additional numerical dependency. It works independently
inside each actual event-class / reciprocal event-class / successor stratum.
Every retained atom is a complete original observed path. The seven preserved
functions are mass, arithmetic return, reciprocal return, log return, squared
return, duration, and log(1 + duration). It does not claim to integrate every
Bellman continuation exactly.

V109/v110 initially pinned the lowest and highest excursion paths, and
v111/v112 reproduced the complete-prior economic results. Review identified
a missing contract for arbitrary borrowing costs: a less extreme but longer
path can uniquely cause liquidation. Those initial quadrature artifacts are
superseded by v113/v114. The current implementation pins the full adverse
excursion/duration Pareto frontiers at their original positive masses. With
nonnegative borrowing costs and the existing maintenance formula, every omitted
path is dominated in adverse excursion and duration by a retained path, so it
cannot uniquely cause ruin. A regression test includes a 1e-200-probability
long-duration ruin path whose excursion alone would not have been pinned.
Ill-conditioned reductions fall back to the whole original stratum if the
normalized moment check fails; they never silently accept bad weights.

Final representation checks across the eight phases:

- January: 183,233–226,406 full atoms become 47,084–49,995 atoms.
- May: 190,439–221,214 full atoms become 32,241–37,597 atoms.
- Maximum reported normalized moment error is below 3.2e-13; maximum
  joint class/successor mass error is below 1.9e-14.
- Maximum one-event expected log-holding error on the checked exposures
  -1, -0.5, +0.5 and +1 is below 0.000020 bps for the saved costs.
- Extraction and reduction cost approximately one second per model.

V115/v116 replay the corrected compact distributions. All eight phases have
**identical executed quantities, account equity/exposure trajectories, returns,
fees, drawdowns, borrowing, PnL, reversals and liquidation counts** to the
complete-prior references. The serialized comparison files record the checked
fields. The per-phase build-and-replay times are 4.16–5.48s for January and
3.21–3.95s for May, versus 15.90–20.14s and 17.18–19.34s respectively for full
support. Compression itself is separate from those timings. These are paired
observations on a shared machine, not a hardware-independent speed guarantee.

This establishes a useful numerical replacement for the crude prior
approximation on these cases, not a profitable final trading model. Next
integrate the complete-law-plus-quadrature construction into the forecast
comparison and rebuild all existing depths and incumbent comparisons. Preserve
the useful January exit/rebound behavior and the reduced weak entries, then
return to improving quiet-state directional reliability. Do not promote the
fixed-depth result over incumbents or infer performance on the remaining
inspector windows from these two cases.

All 55 focused tests and workspace typechecks pass. The full objective remains
active and unmet.

### Integrated quadrature selection and futures-input discovery — v117 onward

V117–v119 add the complete-prior-plus-quadrature construction to the normal
forecast screen. It uses the original frozen partitions, data ranges and
shrinkage. The January and May serialized candidate models match every model
from v113/v114 exactly. All three windows choose quadrature by mean earlier-
origin joint NLL:

| Window | Previous hybrid calibration NLL | Quadrature calibration NLL | Quadrature final NLL |
| --- | ---: | ---: | ---: |
| January | 1.809618 | **1.791166** | 1.847860 |
| May | 2.065912 | **2.065106** | 2.063234 |
| March | 2.011597 | **2.011167** | 2.177300 |

V120–v122 run the complete existing depth grid, 1–8, retaining all 40 earlier
policy candidates. All nine incumbent replay controls are exact. The result
does not justify choosing a different depth after observing the January test:
the new January family still selects depth 4, with score +0.002403935 and
test +4.013463%, 3.887323% drawdown, ten trades. May selects depth 1/cash.
March's best new candidate is depth 1 with score -0.000261801 and is also
rejected by cash. The overall selectors remain the old candidates: January
head-base depth 6 (-0.922748% test), May lag-base depth 3 (+0.969293%), and
March lag-base depth 8 (+0.092160%). These incumbent results are reproduced,
not attributed to the new law. Complete per-window policy comparisons take
28.22s, 27.43s and 15.47s respectively.

The remaining work therefore returns to directional information, rather than
more quadrature or depth tuning. The current event candles contain OHLCV only.
The repository's recommended input basis distinguishes total volume from
executed directional flow, and identifies perpetual trade count as a useful
minute-scale activity input. The data inventory contains 420 cached spot-flow
and futures-kline days, but most old dates occur only inside inspector windows;
they do not cover these rolling training and validation periods. For example,
the first May origin has only eight cached days in its 120-day training period
and none in its 21-day validation period. Using those alone would confound a
feature comparison with missing history.

For a bounded first test, the existing checksum-verifying research fetcher
backfills only BTCUSDT USD-M minute klines from November 12, 2021 through
April 1, 2022: 141 days, 9,200,263 downloaded archive bytes. Research namespaces
are separate from the oracle corpus. No aggregate-trade backfill or live book
collection is required for this test.

`event-futures-basis.ts` requires six contiguous completed minute observations
and returns explicit missingness when source rows are absent or invalid. It
does not forward-fill gaps or treat missing flow as zero. Inputs are:

- Futures/spot basis and futures-minus-spot log return over 1m and 5m.
- Log(1 + futures trade count) for the last minute and taker quote-volume
  imbalance over 1m and 5m.

V123 trains separate sign-only logistic heads on the same 4,243 completed
training events and evaluates the same 409 March-12-origin events for every
feature comparison. One original training event is excluded because the
backfilled futures warmup starts at the training boundary. The price-only
control uses the existing 20 run features plus three fast-volatility inputs.
Additional bases are futures prices, futures flow, or both. Penalties remain
0.01/0.1/1 and blends 0.5/1; this is a forecast discovery screen, not an
economic selection or final-window result. Joint conditional paths given sign
are retained when translating probabilities into mean and class forecasts.

| Best first-origin sign fit | Sign loss | Accuracy | Joint NLL | Signed-return MSE |
| --- | ---: | ---: | ---: | ---: |
| Original joint law | 0.674565 | 58.435% | 2.012969 | 0.0000391050 |
| Spot inputs | 0.669821 | 61.125% | 2.008276 | 0.0000390450 |
| Add futures prices | 0.667177 | 60.880% | 2.005643 | 0.0000388724 |
| Add futures flow | 0.668510 | 61.369% | 2.006958 | 0.0000389874 |
| Add both | **0.666330** | 61.125% | **2.004793** | **0.0000388388** |

Each best row happens to use penalty 0.01/blend 1, allowing a matched-setting
comparison. Adding both inputs improves sign loss by 0.005037 bits per event
over the spot head, positive on 15/21 days; flow alone improves by 0.001892
bits, positive on 14/21 days. Accuracy is essentially unchanged. This is a
small probability improvement, not evidence of a large tradable directional
edge. It justifies checking the next two earlier origins before any policy
integration or further expansion of the dataset.

The new causal feature test covers future mutation, missing/null rows, exact
alignment and observed zero-flow minutes. All 56 focused tests and workspace
typechecks pass at this checkpoint. No strategy is promoted.

### Futures-sign transfer, policy comparison and failure diagnosis — v124–v127

The next bounded backfill adds 42 research-only futures-kline days, April 2
through May 13, 2022. Existing canonical cache covers the final May window.
V124 and v125 repeat the identical forecast grid on the two remaining prior
origins. Futures price inputs improve the matched spot control on all three
origins; flow alone regresses on the third. Mean prior-origin sign loss selects
price inputs, penalty 0.01 and blend 1 (0.665397386), versus the best spot-only
head at penalty 0.1/blend 1 (0.667129543). The combined price/flow head at penalty
0.1 is nearly tied (0.665402230); this small difference does not establish that
flow has no information.

V126 retains all 48 incumbent candidates and adds the selected spot and futures
heads at depths 1–8. Head settings are selected only from prior-origin forecast
loss, and policy/depth selection is saved before the final replay. Continuous
head inputs are timestamped completed observations. They change the first
Bellman backup; continuation still uses the frozen joint market-state law.
Future futures features are **not** simulated recursively.

| Futures-price head, depth 7 | Return | Maximum drawdown | Trades |
| --- | ---: | ---: | ---: |
| March 12 prior origin | +2.324177% | 2.939728% | 5 |
| April 2 prior origin | 0% | 0% | 0 |
| April 23 prior origin | +0.386127% | 9.202792% | 35 |
| Final May diagnostic | **-0.563034%** | **6.163686%** | 5 |

This new family's mean calibration risk score is +0.004895702, below the
incumbent lag-base depth 3 score +0.016286367. The overall selection therefore
remains the incumbent, reproducing its +0.969293% May result. The new diagnostic
spends 2,251 minutes short, pays 24.0080 in execution costs and 1.3007 in borrow,
and loses 30.9947 gross on a 10,000 account. The comparison takes 21.97 seconds.
No new strategy is promoted.

V127 repeats the comparison with explicit forecast/replay checks. All six
origin/head comparisons have exactly equal probabilities; maximum expected-
return discrepancy is below 1.85e-13 bps. All three origin metric arrays, final
metrics and final executed-decision trace match v126 exactly. This rules out
a feature or probability integration mismatch for this experiment. Causal
observation tests reject future, stale and missing inputs and reproduce the
original input path. All 57 focused tests and workspace typechecks pass.

The saved `sign-magnitude-diagnosis.json` explains why accuracy is insufficient:

| Period | Sign accuracy | Mean size when correct | Mean size when wrong | Gross mean from following every predicted sign |
| --- | ---: | ---: | ---: | ---: |
| March 12 prior origin | 60.880% | 45.35 bps | 73.43 bps | -1.115 bps |
| April 2 prior origin | 61.743% | 45.83 bps | 70.26 bps | +1.414 bps |
| April 23 prior origin | 61.351% | 53.74 bps | 76.15 bps | +3.537 bps |
| Final May, completed events | 62.687% | 42.53 bps | 69.86 bps | +0.595 bps |

The diagnostic's invested events have 63.235% sign accuracy, predicted mean
-2.562 bps and realized mean +0.750 bps. These are descriptive event statistics,
not a return decomposition of the held-position policy. Its May 14 16:53 short
entry correctly precedes a -61.89 bps event; the subsequent small orders at
18:03 and 18:16 repair exposure-cap drift. A partial cover at 20:16 forecasts
+6.23 bps although P(up)=0.447 because the conditional magnitudes are unequal.
The final May 16 06:24 exit forecasts +12.24 bps before an actual -72.61 bps
move. Preserve the useful entry and proper cap handling as diagnostic evidence,
not hard-coded timestamp rules.

The next bounded test reuses the existing size-regime/sign factorization with
these external inputs. The wrong-sign magnitude imbalance is already present
in all three earlier origins; no threshold is chosen from the final mistakes.
This tests a specific remaining error before changing continuation states or
expanding computation. Reliable profitability across the inspector windows
remains unachieved.

### Futures size/sign factorization — v128–v132

`screen-event-futures-sign.ts --size-regimes` reuses the existing three logistic
components: P(large), P(up | ordinary), and P(up | large). The large threshold
is the training-only 75th percentile of absolute log return, unchanged from
earlier size/sign experiments. All four input bases use identical matched
events; penalties and blends remain the same small grid. V128–v130 cover all
three earlier origins, each finishing in about three seconds including loading.
V131 reruns the binary screen: every head, forecast metric and prediction array
matches v123 exactly after the shared feature-loader refactor and size extension.

Mean earlier-origin four-group log loss chooses both futures price and flow
inputs, penalty 0.1/blend 1: 1.134775914 versus spot-only 1.137421980. The selected
external model's joint NLL is 2.040871820 versus the original joint law's
2.065106022. However, signed-return MSE is 0.0000401663, slightly worse than the
binary futures head's 0.0000399974. Better size classification is not assumed
to imply better utility.

V132 retains all 64 earlier policy candidates and adds the two forecast-selected
size/sign heads at depths 1–8. All forecast probabilities reproduce exactly in
replay, with expected-return differences below 7.82e-14 bps. The best external
size/sign policy is depth 8: no trades in the first two calibration periods;
+2.349550%, 2.473653% drawdown and six trades in the third. This improves on the
binary family's selected depth-7 result there (+0.386127%, 9.202792% drawdown,
35 trades). Its mean calibration risk score rises to +0.006916692, still below
the incumbent +0.016286367. The external size/sign policy then stays entirely
in cash on the final May window. This is a rejected losing position, not new
profitable test trading. The overall incumbent and its test result are unchanged.

The complete comparison takes 23.60 seconds. All 58 focused tests pass, including
causal timestamp enforcement for both binary and size/sign external inputs.
The next controlled experiment changes only the coarse-state continuation law:
average the same head's probabilities over training features in each existing
state, rebuild Bellman tables, and retain the continuous forecast at each real
decision. Full current-head replacement must preserve the one-event forecast
and depth-1 policy; deeper behavior can then isolate the continuation change.

### Isolating coarse continuation under the futures size/sign head — v133

V133 uses `--project-continuation` with v132 as the incumbent comparison.
Projection reuses the same selected heads and training events, averages their
four probabilities within each original run/volatility state, and rebuilds
depths 1–8 with unchanged cost and account grids. All 80 prior candidates remain
eligible. The sixteen new projected candidates do not introduce any new fitted
hyperparameter. This is a coarse-state approximation, not simulation of future
continuous external inputs or a proof of Bellman convergence.

Every earlier-origin incumbent metric array is preserved exactly. All six
head/origin depth-1 economic controls match exactly, probabilities are identical,
and expected-return differences remain below 7.06e-13 bps. Thus the observed
policy differences isolate deeper continuation rather than a changed current
forecast or execution contract.

The projected external head makes no trades in any earlier origin at any depth.
The projected spot head loses at depths 3–8 in the third origin (depth 8 score
-0.035546385 for that origin); its other origins remain cash. Neither family
beats cash on mean prior-origin score. The final selected external diagnostic
is therefore depth 1/cash, and the overall incumbent is unchanged. Runtime is
66.89 seconds. The projection removed the unprojected family's useful third-
origin trades as well as its assumed continuation edge; this result does not
justify treating projection itself as an improvement.

This sequence establishes modest external sign information and an economically
useful reduction in some earlier-period churn. It does not establish a reliable
profitable policy. Further work should test whether a future-state representation
can retain useful conditional persistence without the original law's spurious
continuation incentive. Repeating arbitrary depth, threshold or leverage tuning
on the same final May window is not evidence toward that objective.

Reproduction commands for this branch (choose fresh output directories):

```powershell
node --conditions=development --import tsx scripts/screen-event-futures-sign.ts --source event-policy-quadrature-policy-may-v121 --phase origin-2022-03-12 --size-regimes --output futures-size-forecast-local
node --conditions=development --import tsx scripts/replay-event-futures-sign.ts --source event-policy-quadrature-policy-may-v121 --incumbent event-policy-futures-policy-control-v127 --forecasts event-policy-futures-size-first-v128,event-policy-futures-size-second-v129,event-policy-futures-size-third-v130 --output futures-size-policy-local
node --conditions=development --import tsx scripts/replay-event-futures-sign.ts --source event-policy-quadrature-policy-may-v121 --incumbent event-policy-futures-size-policy-v132 --forecasts event-policy-futures-size-first-v128,event-policy-futures-size-second-v129,event-policy-futures-size-third-v130 --project-continuation --output futures-projected-policy-local
```

### Multi-event continuation audit — v134

`audit-event-continuation.ts` evaluates only the three prior origins. For each
contiguous completed path of 1/2/4/8 events it compares cumulative log return
with the model expectation. The current continuous size/sign forecast is held
identical; only the subsequent original versus projected transition law changes.
The expectation is computed by repeated transition-matrix application, keeping
the first outcome's joint successor dependence. This is a forecast diagnostic,
not realized policy utility; overlapping paths are not independent observations.

| Prior origin, 8 events | Actual mean | Original predicted mean | Projected predicted mean | Original MSE | Projected MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| March 12 | +34.19 bps | -5.69 bps | -7.65 bps | 31,123.25 | 31,272.19 |
| April 2 | -33.06 bps | -0.41 bps | -3.89 bps | 25,277.59 | 25,064.24 |
| April 23 | -21.64 bps | -2.32 bps | -6.52 bps | 33,920.59 | 33,756.70 |

MSE is in squared basis points. Projected continuation is slightly better in
two origins despite making fewer profitable trades. The missing trades therefore
do not by themselves establish that projection discarded useful information.
Both models' expected returns remain weak compared with realized variation.
At one event the selected size/sign forecast has worse mean error than a zero
log-return forecast in two of three origins. The audit finishes in 1.4 seconds
including loading; it does not justify enlarging the state model yet.

### Return-weighted sign correction — v135–v138

The existing July 18 KAMA value-distillation notes already weight cross entropy
by the range of oracle action values, but that experiment's finalists also
mostly stayed in cash. More generally,
[Example-dependent cost-sensitive decision trees](https://albahnsen.github.io/files/Example-Dependent%20Cost-Sensitive%20Decision%20Trees.pdf)
describes incorporating each example's different error cost during training.
That motivates a small cost-weighted forecast test here, not a claim that its
credit/fraud results transfer to trading.

`trainEventSign` now optionally accepts nonnegative sample weights, normalized
to mean one without changing the ordinary fit's arithmetic. V135, v137 and v138
weight completed training targets by absolute arithmetic return. The resulting
score estimates

\[
q(x)=\frac{E[|r|1(r>0)\mid x]}{E[|r|\mid x]},
\]

which is **not** P(up). Given the existing conditional means mu+ > 0 and mu- < 0,
the mixture used for the joint-path forecast is

\[
p=\frac{q(-\mu_-)}{q(-\mu_-)+(1-q)\mu_+}.
\]

This inversion matches the predicted gain/loss balance conditional on the
existing magnitude estimates; it does not make those estimates exact or train
the entire log-utility curve. All original return/extrema/duration/successor
paths retain support. Replay rejects a raw weighted head rather than silently
interpreting q as an ordinary probability. A numerical test uses an 80%-positive
population whose losses are eight times its gains: ordinary probability is 0.8,
weighted score is 1/3, and inversion recovers 0.8. Weight-unit invariance and
invalid-weight checks also pass.

All three prior-origin forecast grids finish in about two seconds each. V136
reproduces every original v123 binary head, metric and prediction exactly after
the optimizer extension. Comparing all 24 new settings with all 24 previous
binary settings on the same return-weighted loss selects the **previous binary
price head**, penalty 0.01/blend 1:

| Best family | Mean return-weighted loss | Signed-return MSE |
| --- | ---: | ---: |
| Previous binary price head | **0.692830029** | **0.00003999736** |
| New weighted price head, penalty 0.1/blend 1 | 0.693028011 | 0.00004002080 |

The weighted head loses to the binary incumbent in each of the three origins
at these selected settings. It is not promoted or integrated into an additional
policy backtest. `paired-objective-comparison.json` records the full comparison.

### Martingale null for event-sign skill — v139

A stronger diagnostic asks whether the sign/size imbalance appears without any
conditional expected price return. Directional-change statistics can arise in
random-walk processes; the intrinsic-time
[agent-based model paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3240456)
discusses directional-change and overshoot laws as descriptive benchmarks.
The local consecutive-return notes likewise distinguish magnitude dependence
from signed predictive drift. The following is our controlled null construction,
not an empirical conclusion imported from those sources.

`screen-event-martingale.ts` regenerates the same event clock on four seeded
pseudorandom paths. Each minute uses historical absolute close log return a and
changes the synthetic close by an equiprobable +/-tanh(a). Under independent
random signs, the arithmetic price multiplier has expectation one exactly.
Simply flipping log-return signs would introduce positive Jensen drift, which
this construction avoids. The historical timestamp, volume and volatility
schedule is retained; candle wicks are synthetic reflected envelopes. This is
not a claim to reconstruct intraminute dynamics. Every event target and its
causal run features is regenerated, so event counts differ across paths.

The fixed spot-only head uses 20 run features plus three volatility features and
penalty 0.1. Each earlier origin trains on its preceding 120 days. The actual
control uses all available spot events, including the first event excluded by
the earlier external-data warmup; it is not asserted to be the identical old
external-matched head. No final window, parameter search or policy selector is
used. The entire 15-fit experiment takes 8.5 seconds.

| Prior origin | Actual sign accuracy | Four-seed null mean | Null range | Actual magnitude-weighted accuracy | Null weighted mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| March 12 | 61.614% | 60.463% | 57.806–64.706% | 49.103% | 49.906% |
| April 2 | 62.712% | 62.084% | 60.085–64.080% | 52.349% | 51.982% |
| April 23 | 61.163% | 57.264% | 55.453–58.915% | 52.435% | 48.980% |

The null also predicts smaller correct moves and larger wrong moves. Thus much
of the roughly 60% event-sign accuracy can be induced by the stopping rule,
without a profitable conditional mean. The third real origin exceeds all four
null seeds, so this is **not** evidence that all market information is absent.
Four seeds are diagnostic, not a formal significance test or a proof of market
efficiency. The test changes the next action: additional sign-accuracy gains
alone are no longer persuasive evidence for another policy expansion. A new
learner should show a conditional expected-return or fee-aware value improvement
against a no-drift reference before scaling.

The null transformation test checks arithmetic centering, valid candle bounds,
deterministic reproduction, future-mutation isolation and gap rejection. All 60
focused tests pass. This branch has produced no new promoted trading policy;
the full inspector-window profitability objective remains active and unmet.

### Fitted holding-value Bellman iteration — v140–v145

The martingale sign null motivated a different training target: estimate the
decision's conditional log-wealth value directly instead of optimizing another
event-sign accuracy metric. The existing distribution remains the source of
positive-probability liquidation guards. This is an additional research policy,
not evidence that the distribution model or separate sign heads are unnecessary.

`event-fitted-value.ts` fits a sequence of regularized linear holding functions:

```
H_d(features, account)
  <- E[log(holdingFactor) + V_(d-1)(nextFeatures, markedAccount) | features]
V_d(features, account)
  = max_feasible_order [log(postFeeEquity / equity) + H_d(features, postTradeAccount)]
V_0(account) = log(terminal cash-settlement factor)
```

Every training row contains a completed event and its observed successor
features. The maximum in the training target uses the **previous learned
conditional value**, not the realized best action on a future price path.
At inference only current completed features are supplied. The common
`chooseEventTrade` optimizer preserves fees, lot rounding, minimum/maximum
orders, exposure-cap repair and all four entry/exit signals. The actual May
source grid has **5 equity × 3 price × 13 exposure coordinates = 195 heads**,
with 11 target exposures from −1 to +1. At depth one the holding value is
independent of equity and price, so only 13 distinct regressions are required.
At later depths all account cells are retained. A shared Cholesky factorization
makes fitting inexpensive; this is still an interpolated approximation.

The first target uses exact terminal settlement at the marked exposure, whereas
the original compiled policy interpolates a bounded terminal grid. Accordingly,
the numerical reference test uses direct expected log settlement, not equality
to a clamped terminal-grid approximation at extreme exposures. A single
non-finite training outcome forbids its fitted cell; it is not dropped from the
regression. Separate analytical liquidation roots retain every original
positive-probability adverse path, including arbitrarily small tail mass.

This follows the supervised-backup structure of
[Ernst, Geurts and Wehenkel, *Tree-Based Batch Mode Reinforcement Learning*](https://www.jmlr.org/beta/papers/v6/ernst05a.html).
The implementation uses ridge regression instead of their tree regressors and
finite horizons instead of claiming convergence. It is not least-squares policy
iteration or a proof of the globally optimal trading policy.

**v140: cheap one-event screen.** Three earlier 21-day origins, each trained on
120 days of completed events, compare spot/run/volatility inputs (23), those
inputs plus three futures-price inputs (26), and price plus flow (29). Penalties
are 0.01, 0.1 and 1. All nine candidates are retained. Matched train/evaluation
counts are 4,243/409, 3,918/413 and 3,492/1,066. One initial training event lacks
matched external history; no evaluation event is discarded.

Mean prior-origin one-event holding-value MSE selects the **price basis,
penalty 0.1**: 0.0000401279. Its mean fractional improvement over a zero-return
reference is only **0.00002937**, or **0.00294%**. This reference includes fees
but is not a full stochastic martingale model. The tiny average improvement is
not convincing predictive evidence by itself. Every candidate stays in cash
in the first two origins. In the third, price/all penalty 0.01 returns +6.3543%
with 3.4873% drawdown and four trades; penalty 0.1 returns +2.5070% with 3.4205%
drawdown and two trades. All spot candidates and penalty-1 candidates stay in
cash. The complete 27-fit screen takes 26.1 seconds.

**v141–v142: second backup and a cash-bound correction.** With the
forecast-selected price/0.1 setting, two events return +6.8565% in the third
origin, with 4.0715% drawdown and four trades; the other origins stay in cash.
An audit finds that unconstrained regression predicts negative cash continuation
at 248/410 and 270/414 decisions in the first two origins (minimum −0.1420 and
−0.0903 basis points). Cash can remain cash indefinitely, so its continuation
must be at least zero. `predictEventHoldingGrid` now enforces this lower bound
without erasing positive future option value or clamping risky holdings.

v142 reproduces every forecast-ranking and economic policy result exactly after
the correction. Only the reported sum of predicted decision values changes in
the two inactive origins. No profitable third-origin decision depended on a
negative cash forecast. v141 is retained as the pre-correction record. Tests
cover the cash bound, direct expectation before maximization, serialization,
rare liquidation support, aligned feature availability, and first-order
invariance to future-candle changes. All **64 focused tests** and the workspace
typecheck pass.

**v143: limited depth expansion.** The same selected price/0.1 regression is
tested through four event steps. No final-window outcome enters this expansion.

| Prior evaluation origin | Depth 1 return | Depth 2 | Depth 3 | Depth 4 |
|---|---:|---:|---:|---:|
| March 12–April 2 | 0% | 0% | 0% | +1.8535% |
| April 2–23 | 0% | 0% | 0% | 0% |
| April 23–May 14 | +2.5070% | +6.8565% | **+11.8654%** | +6.3797% |

Depth three has 4.0714% worst drawdown and 12 executed trades, all in the third
origin. Depth four has nine trades/3.9893% drawdown in the first origin and 17
trades/4.4496% drawdown in the third. The added depth is not monotonically better.
The full four-depth run takes 61.9 seconds. The best mean risk-adjusted origin
score is 0.03601835 at depth three, versus 0.02074828 at depth two and the
previous incumbent's 0.01628637. It still wins in only one of three origins.

**v144: complete retained selection and final May replay.**
`replay-event-fitted-value.ts` retains every one-event setting from v140, all
new depths, and all 96 v133 incumbent candidates. Duplicate economic results
must match; the entire incumbent ranking reproduces exactly. The combined
**108-candidate** ranking selects price/0.1/depth three and is saved before final
outcomes are loaded. The selected model is then trained on 4,025 completed
pre-window events and replayed with next-open execution on May 14–21.

It makes **zero trades, returns 0%, and has 0% drawdown**. The previous
incumbent's +0.9693% remains better on this window. There are no canceled orders,
liquidations or account-grid boundary decisions in the fitted final replay.
The original joint-law final trace also reproduces exactly. Training and replay
take 14.8 seconds. This is a failed transfer of the calibration improvement,
not a new reliably profitable inspector strategy. No broader 28-window run is
justified by this result alone.

**v145: why the result changes, and which behavior matters.** A post-selection
audit replays the April 23 fit on the final May window without retraining; it
also stays entirely in cash. Thus changing the fit is not sufficient to explain
the lack of final entries. The same successful calibration model sees no
sufficiently strong opportunity in the final inputs. This older-fit replay is
explicitly a diagnostic, not an additional selected candidate.

For the fresh final fit, the best full-1× long and short three-event advantages
over waiting in cash are **−7.86 and −8.57 basis points**, respectively. Their
one-event maxima are −11.28 and −12.51 basis points. The earlier fit's best
three-event long advantage on the same market is −0.49 basis points; its actual
optimizer, including smaller orders, still chooses cash. The final period has
no clipped futures-price input. The earlier active calibration period has 17,
12 and 19 clipped observations in basis, relative 1-minute return and relative
5-minute return, respectively. The policy's profitable activity is concentrated
around larger disturbances, not broadly distributed across ordinary states.

The third origin contains four cash-to-cash long positions:

| Entry and exit, UTC | Account return during position | Net PnL |
|---|---:|---:|
| May 5 15:09–17:27 | −0.5459% | −54.59 |
| May 11 12:53–13:46 | +4.3510% | +432.72 |
| May 11 20:57–May 12 00:44 | +4.2865% | +444.85 |
| May 12 05:15–10:31 | +3.3591% | +363.56 |

The four positions produce gross long PnL 1,293.08 and fees/slippage 106.54 on
initial wealth 10,000. Two proposed size increases are canceled by the unchanged
next-open feasibility check; they are not counted as executed trades. All
entries from cash have positive three-event value while their full-1×
one-event alternatives remain negative after entry and terminal costs. The
deeper holding model can therefore preserve exposure across several events
instead of requiring immediate round-trip profitability.

Preserve the demonstrated ability to hold through some initially adverse events
and capture the longer rebound. Do not preserve every exit indiscriminately:
the May 11 13:46 exit precedes another +282.48-bp event; May 12 10:31 precedes
+88.80 bp. The May 5 trade loses despite an initially correct next-event sign,
and the May 12 07:21 increase precedes −57.29 bp before later recovery. These
are further reasons not to substitute sign accuracy for policy value. The
next unresolved issue is conditional value and continuation outside the rare
active conditions, including whether additional backups generalize or only
amplify approximation error. Do not weaken costs or select a horizon using
the already inspected final return.

Reproduction:

```powershell
node --conditions=development --import tsx scripts/screen-event-fitted-value.ts --source event-policy-quadrature-policy-may-v121 --output event-policy-fitted-value-one-v140
node --conditions=development --import tsx scripts/screen-event-fitted-value.ts --source event-policy-quadrature-policy-may-v121 --settings-source event-policy-fitted-value-one-v140 --depths 2 --output event-policy-fitted-value-cash-bound-v142
node --conditions=development --import tsx scripts/screen-event-fitted-value.ts --source event-policy-quadrature-policy-may-v121 --settings-source event-policy-fitted-value-one-v140 --depths 4 --output event-policy-fitted-value-four-v143
node --conditions=development --import tsx scripts/replay-event-fitted-value.ts --source event-policy-quadrature-policy-may-v121 --screens event-policy-fitted-value-one-v140,event-policy-fitted-value-cash-bound-v142,event-policy-fitted-value-four-v143 --incumbent event-policy-futures-projected-policy-v133 --output event-policy-fitted-value-policy-v144
node --conditions=development --import tsx scripts/audit-event-fitted-value.ts --source event-policy-fitted-value-policy-v144 --output event-policy-fitted-value-audit-v145
```

Use new output directory names when rerunning. v141's unconstrained cash results
can be inspected through its saved source snapshot; the working implementation
contains the cash correction.

### Nonlinear value corrections and relative futures basis — v146–v151

The next controlled question was whether the linear holding function missed
nonlinear interactions. The earlier v27–v35 boosted experiments estimated a
return mean, converted it into coarse states and rebuilt a path distribution.
The new experiment keeps the continuous fitted holding function and adds small
shared-partition vector regression trees to its residuals. This directly tests
the function approximation in the fitted Bellman backup. It is motivated by
the tree-based fitted-Q work cited above; it does not inherit a convergence
guarantee for this boosted, constrained, interpolated implementation.

`event-value-boost.ts` uses squared-error histogram splits, depth two,
minimum 128 training observations per leaf, and learning rate 0.05. All finite
holding coordinates share each tree's partition and retain separate leaf
values. Training-only quantile cuts snap to distinct supported boundaries so
tied inputs cannot silently collapse otherwise supported groups. Missing or
ruin-forbidden holding coordinates receive no correction and remain forbidden.
The separate hard liquidation guards and cash lower bound are unchanged.

**v146: reject the tested nonlinear correction.** Price/penalty-0.1 is fixed
from the prior forecast selection. Zero, 16 and 64 residual trees are compared
at depth one on all three earlier origins, using identical training and target
rows. The unchanged linear forecast and policy metrics reproduce exactly.

| Model | Mean validation holding MSE | Mean fractional skill vs zero-return reference | Third-origin return |
|---|---:|---:|---:|
| Linear control | 0.00004012789 | +0.00002937 | +2.5070% |
| +16 residual trees | 0.00004024118 | −0.00301377 | +2.5070% |
| +64 residual trees | 0.00004035352 | −0.00597861 | +3.0810% |

Both tree sizes reduce training error but worsen validation MSE in the first
two origins. The 64-tree policy additionally loses −0.6084% in the first origin
with 0.9874% drawdown and two trades. The third-origin return improvement is
therefore not a reason to promote it. The nine-fit screen takes 12.3 seconds;
no deeper tree-model expansion or final tree-only evaluation is performed.
Tests verify nonlinear interaction learning, supported leaf sizes, serialization,
both Bellman output shapes and preservation of rare liquidation constraints.

**v147–v148: a narrower feature proposal from existing research.**
The local cross-asset component and global BTC feature-basis notes repeatedly
identify futures-basis deviations at 5, 15 and 60 minutes among useful short-horizon
inputs. They also show serious transfer failures after broad searches; those
results do not establish event-scale trading profitability. This experiment
tests only the three deviations, using the already cached BTC futures archive:

```
basis = 10,000 * log(futuresClose / spotClose)
deviation(span) = basis - EMA_span(basis), span in {5, 15, 60}
```

Each EMA initializes at the oldest of exactly 240 completed, contiguous paired
minutes and updates through the current completed minute with alpha=2/(span+1).
This fixed causal warmup makes values independent of where a loader starts.
It is a finite-warmup implementation of the proposed feature, not an assertion
that it reproduces another experiment's indefinitely recursive EMA state.
Missing observations invalidate the feature; no forward fill is used. Constant
basis offsets cancel. Tests cover that invariance, the known response to a
one-minute impulse, future-mutation invariance and missing-history rejection.

Raw-price inputs (26 coordinates) and raw price plus the deviations (29) are
each tested at penalties 0.01, 0.1 and 1. They use the same matched cohort.
The 240-minute requirement removes four additional early training events in
the first origin: counts are now **4,239/409, 3,918/413, 3,492/1,066** for
training/validation. No validation event is discarded.

v147 initially did not encode this longer-history cohort in its candidate
identity. v148 corrects that provenance issue with `historyMinutes: 240` and
the `-m240` candidate suffix, including for the raw-price controls. Forecasts
and economic results reproduce exactly. The final replay applies the same
training-cohort requirement. v147 is a superseded naming/provenance record and
is explicitly rejected as a selection input; its results are not silently
merged with the original shorter-history price models.

The deviation/0.1 setting wins mean one-event forecast MSE, **0.00004011639**.
Compared with its matched raw-price/0.1 control, per-origin MSE gains are
+6.3635e−8, +1.1424e−8 and −3.6227e−8 (positive means improvement).
The equally weighted mean gain is only **1.2944e−8**. A 2,000-replicate
whole-day bootstrap within the three 21-day origins gives a 95% interval
**[−8.7648e−8, +1.1090e−7]**, with 61.8% positive replicates. This is a small,
uncertain research improvement, not statistical confirmation of an edge.

At depth one, deviation/0.1 stays in cash in the first two origins and returns
+1.9197% in the third with 3.4872% drawdown and two trades. The lower-penalty
deviation model loses −0.9117% in the first origin and returns +3.6832% in the
third. The best raw-price control still returns +6.3543% in the third origin.
The corrected six-setting screen takes 16.6 seconds.

**v149: limited three-event check.** The single forecast-selected
deviation/0.1 setting is extended to depth three, retaining every earlier
candidate. The first two origins stay in cash at every tested depth. Third-origin
returns are +1.9197%, +3.6832% and **+17.7489%** at depths one, two and three.
Depth three has **4.5709% drawdown**, 11 executed trades and two canceled
orders. Its mean risk-adjusted origin score is 0.05293789, versus the previous
price/0.1/depth-three score of 0.03601835. This 43.1-second experiment shows
better calibration behavior, still concentrated in one origin.

**v150: retained comparison and final replay.** All 108 v144 candidates are
retained, plus two residual-tree variants, six matched-cohort one-event
variants and two deeper deviation variants: **118 candidates total**. Every
incumbent ranking entry reproduces exactly. The deviation/0.1/depth-three
selection is saved before final data are loaded, then refit on 4,025 completed
pre-window events. On May 14–21 it makes **zero trades, returns 0%, and has 0%
drawdown**. The original joint-law trace reproduces. Final training/replay
takes 14.9 seconds. The older lag-base incumbent's +0.9693% remains better on
this inspector window; the all-window profitability goal is not achieved.

**v151: trade-level attribution.** The independent audit reproduces every
prior and final economic metric. The improved third origin contains four
cash-to-cash positions:

| Side | Entry/exit, UTC | Account return during position | Net PnL |
|---|---|---:|---:|
| Short | May 8 17:02–May 9 02:57 | +1.6461% | +164.61 |
| Long | May 11 12:53–13:46 | +4.3510% | +442.26 |
| Long | May 11 20:57–May 12 00:44 | +4.5494% | +482.55 |
| Long | May 12 05:15–13:29 | +6.1814% | +685.48 |

Compared with v143, the deviation policy avoids the losing May 5 long, adds
the May 8 short, and keeps more exposure during the last two rebounds. It
holds the final position until 13:29 rather than exiting at 10:31; the new
exit precedes a −100.43-bp event. This is behavior worth preserving in further
tests. The May 11 13:46 exit still misses the subsequent +282.48-bp event, and
the May 12 07:58 size increase still precedes a −66.55-bp event before recovery.
The short's full-1× entry advantage is only +0.37 bp; its profitable realized
result alone does not establish reliable forecast calibration.

The final fresh fit's maximum three-event full-1× long/short advantages over
cash are −7.23/−7.15 bp. In a post-selection counterfactual, keeping the earlier
April 23 deviation fit opens one final-window long, but returns **−0.0191%**
with **1.5588% drawdown**: gross PnL 22.09 is smaller than costs 24.00. The fresh
fit avoids that trade. This contradicts treating delayed refitting or simply
forcing more entries as the missing solution. The main unresolved issue is
reliable conditional value outside the active rebound conditions; neither
extra tree capacity nor these additional basis inputs establishes it.

Relevant local sources:

- `docs/experiments/binance-cross-asset-component-feature-bases-2026-08-20.md`
- `docs/experiments/global-btc-feature-basis-search-2026-08-21.md`
- `docs/experiments/extended-market-information-basis-2026-08-16.md`

Verification: **67 focused tests pass**, as does the workspace typecheck.
Reproduction uses new output directory names:

```powershell
node --conditions=development --import tsx scripts/screen-event-fitted-value.ts --source event-policy-quadrature-policy-may-v121 --settings-source event-policy-fitted-value-one-v140 --boost-trees 0,16,64 --output event-policy-fitted-value-boost-screen-v146
node --conditions=development --import tsx scripts/screen-event-fitted-value.ts --source event-policy-quadrature-policy-may-v121 --basis price,deviation --output event-policy-fitted-value-deviation-cohort-v148
node --conditions=development --import tsx scripts/screen-event-fitted-value.ts --source event-policy-quadrature-policy-may-v121 --settings-source event-policy-fitted-value-deviation-cohort-v148 --depths 3 --output event-policy-fitted-value-deviation-three-v149
node --conditions=development --import tsx scripts/replay-event-fitted-value.ts --source event-policy-quadrature-policy-may-v121 --screens event-policy-fitted-value-boost-screen-v146,event-policy-fitted-value-deviation-cohort-v148,event-policy-fitted-value-deviation-three-v149 --incumbent event-policy-fitted-value-policy-v144 --output event-policy-fitted-value-deviation-policy-v150
node --conditions=development --import tsx scripts/audit-event-fitted-value.ts --source event-policy-fitted-value-deviation-policy-v150 --output event-policy-fitted-value-deviation-audit-v151
```

### Deeper fitted values, horizon audit and reusable checkpoints — v152–v158

The deviation/0.1 setting and matched 240-minute cohort are fixed from v148.
v152 extends it to six events, retaining the earlier depths. All three prior
origins now earn positive returns after costs. v155 extends the same setting
to eight events; its depths one through six reproduce v152's coefficients,
one-event forecasts and economic metrics exactly.

| Prior evaluation, UTC | Depth 3 return | Depth 6 return / drawdown / trades | Depth 8 return / drawdown / trades |
|---|---:|---:|---:|
| March 12–April 2, 2022 | 0% | +2.7556% / 4.1161% / 11 | +0.2065% / 4.5600% / 16 |
| April 2–23, 2022 | 0% | +0.8432% / 0.7124% / 2 | +0.8432% / 0.7124% / 2 |
| April 23–May 14, 2022 | +17.7489% | +27.5087% / 5.5241% / 43 | +39.3582% / 6.2465% / 61 |

Mean log growth minus the existing drawdown penalty increases from 0.05293789
at depth three to 0.08941395 at six and 0.11027256 at eight. However, the first
origin deteriorates after depth six and its depth-eight risk-adjusted score is
negative, −0.00249737. The increasing mean is driven mainly by the third origin;
it is not uniform improvement or evidence that unlimited recursion is useful.
The six/eight-depth screens take 102.8/140.2 seconds respectively.

**v153: inspect approximation before extending it.** The new
`audit-event-fitted-horizon.ts` compares every saved horizon on the same prior
decision times and the same accounts: equity 10,000, current price, and exposure
−1, 0, +1. It loads no final-window outcomes. The audit reports cash entry
advantages, changed exposure decisions, value increments, and validation
Bellman residuals. A residual compares the current predicted holding value
with realized one-event holding return plus the *previous learned* continuation
at the observed next state. It is not an observed multi-event payoff error.

Depth six changes 3.50%, 3.62% and 6.56% of the common-account decisions relative
to depth five. Mean value increments are 0.426, 0.464 and 0.431 bp; some
predicted holding values decrease. Values are not exploding, but neither values
nor actions have converged. At depth six, common flat-account entry opportunities
number 4, 1 and 72. These are probe opportunities, not executed trade counts.
Side-specific validation residual biases are still several basis points, against
holding-return MSE around 3,460–4,597 bp². Approximation and sample noise remain
material relative to the small entry margins.

This diagnostic is motivated by [Munos and Szepesvári's fitted value iteration
analysis](https://www.jmlr.org/papers/volume9/munos08a/munos08a.pdf), which relates
performance to propagated approximation and estimation errors and the function
class's Bellman residual. Its discounted-MDP and sampling assumptions do not
establish a guarantee for this undiscounted historical-data implementation.
Nor is monotonicity asserted here: terminal settlement can use multiple maximum
order clips, ordinary decisions use one clip, and account values are interpolated.

**v154/v158: final May remains cash.** The retained comparisons contain 121
and 123 candidates. All incumbent ranking entries reproduce exactly, and the
six/eight-depth choices are saved before final data are loaded. Each refit uses
4,025 pre-window events. Both make zero trades across 269 final decisions, with
0% return and drawdown. Final fit/replay takes 34.3/48.7 seconds. More profitable
calibration behavior has not solved final-window activity or transfer.

**v156: six-depth attribution.** First-origin PnL is mainly short exposure:
gross short PnL 343.31, gross long PnL −4.41, fees 63.08 and borrowing 0.26.
The second origin makes one long position: gross PnL 108.42 and fees 24.10.
The active third origin earns gross long/short PnL 1,431.09/1,574.67, pays 251.20
in fees and 3.68 in borrowing, and spends 1,941/6,448 minutes long/short. Its
43 trades include one reversal and four canceled orders. The fresh final fit's
best full-1× long/short entry advantages remain −4.48/−3.21 bp. A diagnostic
counterfactual retaining the April 23 fit returns only +0.0987%, with 3.0749%
drawdown and four trades; it is not a selection candidate.

**v157: avoid recomputing earlier backups.** `trainEventFittedValue` now accepts
a checkpoint containing the earlier depth tables. It validates costs, account
axes, standardization, regularization, liquidation limits, boost settings and a
deterministic signature of every training transition. The signature is an
accidental-consistency check, not authentication. Old artifacts without it
remain readable for inference but cannot be resumed. The trainer copies the
table list and computes only missing backups. The screen exposes this through
`--resume-source`, which also checks source, origins and settings.

The synthetic checkpoint test reproduces an uninterrupted fit and rejects
changed targets, regularization and missing signatures. On the actual 4,239-row
first-origin dataset, resuming a genuine six-depth prefix to eight reproduces
the complete v155 model exactly, leaves the prefix unchanged and takes 14.2
seconds. That timing covers the two new backups, not the full screen/replay.
The full eight-depth v155 run was necessary because v152 predates signatures;
future compatible depth extensions can reuse the saved work. All 68 focused
tests and the bot-algo typecheck pass.

The next useful experiment is transfer to another inspector case with a small
one-event feature screen. Further May-only depth or feature searches would
increase selection exposure without resolving the observed concentration.

```powershell
node --conditions=development --import tsx scripts/screen-event-fitted-value.ts --source event-policy-quadrature-policy-may-v121 --settings-source event-policy-fitted-value-deviation-cohort-v148 --depths 6 --output event-policy-fitted-value-deviation-six-v152
node --conditions=development --import tsx scripts/audit-event-fitted-horizon.ts --source event-policy-fitted-value-deviation-six-v152 --output event-policy-fitted-value-horizon-audit-v153
node --conditions=development --import tsx scripts/replay-event-fitted-value.ts --source event-policy-quadrature-policy-may-v121 --screens event-policy-fitted-value-deviation-six-v152 --incumbent event-policy-fitted-value-deviation-policy-v150 --output event-policy-fitted-value-six-policy-v154
node --conditions=development --import tsx scripts/screen-event-fitted-value.ts --source event-policy-quadrature-policy-may-v121 --settings-source event-policy-fitted-value-deviation-cohort-v148 --depths 8 --output event-policy-fitted-value-deviation-eight-v155
node --conditions=development --import tsx scripts/audit-event-fitted-value.ts --source event-policy-fitted-value-six-policy-v154 --output event-policy-fitted-value-six-audit-v156
node --conditions=development --import tsx data/benchmarks/event-policy-fitted-value-resume-check-v157/check.ts
node --conditions=development --import tsx scripts/replay-event-fitted-value.ts --source event-policy-quadrature-policy-may-v121 --screens event-policy-fitted-value-deviation-eight-v155 --incumbent event-policy-fitted-value-six-policy-v154 --output event-policy-fitted-value-eight-policy-v158
```

### March transfer, censored-chain correction and trade attribution — v159–v167

The transfer case is `sideways-churn-2023-03`, March 18–25, 2023. It provides
another choppy market without continuing to tune the May outcome. The earlier
three 21-day origins begin January 14, February 4 and February 25. Their
120-day training histories have fewer complete event transitions than May's.
The same 1× costs, target grid and original joint-law controls are retained.

**v159: bounded data coverage.** The existing checksum-verifying
`fetch-research-forward-market.ts` fills the 168 missing futures-kline days
between September 15, 2022 and March 18, 2023, exclusive. Sixteen existing
research/canonical references are reused and verified unchanged. New archives
total **10,105,792 bytes**, fetched in 242.5 seconds. Each reference records
its archive checksum and research-only namespace; no other market sources
are downloaded. The final March week was already cached.

**v160–v163: initial results, superseded by the sampling correction below.**
The first six-setting screen compares raw futures-price inputs and those
inputs plus the three basis deviations, with penalties 0.01/0.1/1 and a
matched 240-minute cohort. Raw price/1 wins. Extending this one setting to
three depths does not improve its policy. The final retained comparison keeps
the older lag-base/depth-eight incumbent. These policy conclusions survive
the correction, but v160's and v162's forecast summaries include one spurious
last-origin row. Use v165–v167 for subsequent selection or comparison.

**v161 and the boundary correction.** The replay previously read the next
day to resolve the eventual stopping time of its final event, then clipped
execution to the evaluation boundary. Fitting and scoring excluded that later
outcome, but requiring it made strict prior-only loading impossible. Replay
now resolves events only from the prefix through the scoring boundary. An
unresolved final event is executed through that boundary and settled; it is
not fed back as a completed event. Missing scored candles and missing event
features still fail explicitly. Fitted screens and final replays no longer
load the extra day.

Nine real May checks—three origins at depths one, six and eight—reproduce
every saved trade and economic metric exactly with no post-window candles
loaded. The check takes 2.28 seconds. The regression test also changes all
post-window prices and verifies identical output, and confirms that missing
scored history still fails.

Strictly truncating the loader exposed a separate sampler defect. When
`observeMove` returned null at the end of a chain, `makeSamples` advanced by
the configured stride and could find a shorter event from a different origin
inside the unresolved run. That row is not a successor on the original chain.
The sampler now stops at a contiguous censored final chain instead of
restarting it. A focused example tests a price fluctuation that crosses a
threshold only after such an invalid restart.

For March, the removed origin is March 17 at 23:35 UTC, whose alleged endpoint
was 23:42. The actual replay chain had already started its final unfinished
event at 23:25. This correction changes the third-origin forecast count from
537 to **536**. Training rows, all fitted coefficients and every economic
metric remain identical across all 18 one-event fits and the three deeper
fits. Retained forecast rows are also exact. Reproduction records are saved
with v165/v166. This was a forecast-sampling defect, not the cause of the
observed trading losses. v164's audit already followed saved replay times and
excluded the censored outcome, so its 536-row residual calculation remains valid.

**v165: corrected forecast screen.** Training/validation counts are
**1,953/401, 1,808/351 and 1,926/536**. All settings use identical rows. The
corrected screen takes 9.32 seconds.

| One-event setting | Mean holding MSE | Mean fractional skill versus zero-return reference | Prior returns, first / second / third |
|---|---:|---:|---:|
| Raw futures price, penalty 1 | **0.00003933262** | **−0.00467984** | +0.6266% / 0% / 0% |
| Price + deviation, penalty 1 | 0.00003937505 | −0.00587812 | −2.0291% / 0% / 0% |
| Raw futures price, penalty 0.1 | 0.00004017077 | −0.02705262 | −4.6613% / −1.7899% / 0% |
| Price + deviation, penalty 0.1 | 0.00004021279 | −0.02825650 | −4.6613% / −3.7178% / +0.2671% |

The lower penalty 0.01 also loses in all three origins for either basis.
The selected setting's average fractional forecast skill is negative, about
−0.468%. The May-selected deviation/0.1 setting does not transfer into a
reliable March value forecast. Lower training error at weaker regularization
does not imply better calibration behavior.

**v166: reuse the selected checkpoint, stop at three.** A resumed screen may
now select an exactly matching subset of settings from a prior multi-setting
screen; source, phase and complete training-signature checks still apply.
The first-depth table is reused exactly and only the two missing backups are
computed. All three-depth coefficients and policy results reproduce v162,
and the corrected forecast rows match v165. This screen takes 22.91 seconds.
Depth two retains +0.6266%/0%/0%; depth three returns
**−0.5558%/0%/0%**, with 2.2700% first-origin drawdown and seven trades.
There is no basis for expanding this setting to six or eight events.

**v167: retain the complete comparison.** The replay supports using the
original joint-policy source itself as the incumbent, as well as a later
comparison derived from that source. It retains all 48 existing March
candidates and adds eight fitted candidates: **56 total**. Every incumbent
ranking entry is exact. The selected incumbent remains lag-base/depth eight,
with prior score 0.01751597 versus 0.00170716 for the best fitted candidate.
Its previously measured final +0.092160% return and 6.362654% drawdown are
carried forward unchanged, not attributed to the fitted model. The newly
refit fitted diagnostic uses all 1,697 complete training events and makes
zero trades over 223 final decisions. Its final return and drawdown are 0%.
The original joint-law final trace reproduces exactly. Fit/replay takes 1.29 seconds.

**Trade attribution and behavior to preserve.** The one-event model opens a
short on January 14 at 00:45 UTC, earns the following −86.63-bp move, and
exits at 00:51 before a +70.01-bp rebound. Net PnL is **+62.66** after 23.87
in fees and 0.0042 borrowing. That prompt exit is useful behavior.

The three-event model adds a profitable January 14 00:39–00:40 long, net
**+83.75**, but retains part of the short through the rebound. Its short
position closes only at 01:08, reducing that position's net PnL to **+4.03**.
It then enters long on January 30 at 19:10 before a −120.62-bp event and
exits a minute later for **−143.36**. These three cash-to-cash positions
reconcile exactly to final equity 9,944.42. Fees increase to 72.27. The saved
`trade-attribution.json` includes all position orders and the reconciliation.

v164 probes show the two additional long entries have estimated advantages
over waiting in cash of only **0.477 and 0.331 bp** at a common 10,000-equity
account. Both are small relative to forecast errors. The first-origin
depth-three short/long Bellman-residual biases are +3.67/−3.72 bp, with MSE
about 3,797/3,691 bp². These residuals contain a learned continuation, so they
cannot be interpreted as direct three-event return errors. The failure is
weak conditional value and a wrong hold/exit decision, not simply absence
of a sign classifier or a need for more recursion.

The local documentation search found no ready-made double-critic correction
for this pipeline. [Double Q-learning](https://proceedings.neurips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html)
separates action selection and evaluation to reduce bias from maximizing noisy
estimates; it can also underestimate values. That makes an independently fitted
continuation-value diagnostic relevant, but these results do not establish
maximization bias as the cause. Broadly making the already inactive final
policy more pessimistic is not yet a supported fix. A bounded comparison of
independent critics, using only earlier complete event blocks, should precede
any larger double-critic or nonlinear policy experiment. The zero-entry final
windows and the useful prompt short exit must remain visible in that comparison.

All **70 focused tests** pass. The workspace typecheck also passes. The new
research scripts run against the real artifacts; no model from this section
is promoted as a reliably profitable policy.

```powershell
$eventMissingDays = (Get-Content data/benchmarks/event-policy-march-futures-backfill-v159/config.json -Raw | ConvertFrom-Json).missing -join ","
node --conditions=development --import tsx scripts/fetch-research-forward-market.ts --only $eventMissingDays --sources futures-klines --quiet
node --conditions=development --import tsx data/benchmarks/event-policy-terminal-boundary-check-v161/check.ts
node --conditions=development --import tsx scripts/screen-event-fitted-value.ts --source event-policy-quadrature-policy-march-v122 --basis price,deviation --output event-policy-fitted-value-march-chain-screen-v165
node --conditions=development --import tsx scripts/screen-event-fitted-value.ts --source event-policy-quadrature-policy-march-v122 --settings-source event-policy-fitted-value-march-chain-screen-v165 --resume-source event-policy-fitted-value-march-chain-screen-v165 --depths 3 --output event-policy-fitted-value-march-chain-three-v166
node --conditions=development --import tsx scripts/replay-event-fitted-value.ts --source event-policy-quadrature-policy-march-v122 --screens event-policy-fitted-value-march-chain-screen-v165,event-policy-fitted-value-march-chain-three-v166 --incumbent event-policy-quadrature-policy-march-v122 --output event-policy-fitted-value-march-chain-policy-v167
```

Use fresh output names when rerunning. v160/v162/v163 are superseded selection
records; v165/v166/v167 contain the corrected forecast cohort.

### Separate-critic diagnosis — v168–v169

`audit-event-fitted-critics.ts` tests the action-selection/evaluation issue
identified after March transfer. The feature basis and regularization are
frozen at each window's earlier selection: raw futures price/1 for March,
deviation/0.1 for May. It splits the earlier 120-day training history into
alternating UTC-epoch-anchored 14-day blocks. A transition is retained only
when its entire 1,440-minute input support through the next event close fits
inside one block. No input or target interval crosses between the two critics.
The historical blocks remain serially related market data; this is not a claim
of statistical independence.

Three new one-event models are fitted per origin: critic A, critic B, and a
pooled control on their combined retained rows. The original full-cohort
model is reused after exact transition-signature validation. The audit also
evaluates the pointwise mean of A/B holding values. All models keep the same
account axes, costs and original hard liquidation-support guards.

At each validation event, the diagnostic considers common 10,000-equity
accounts with exposure −1, 0 and +1. Each critic chooses a feasible trade.
Its own predicted value and the other critic's value for that *same trade*
are compared with observed one-event holding plus terminal settlement. Entry
fees, borrowing, return-dependent marked exposure and settlement costs are
included. These are fixed-account utility probes, not a continuous backtest
or observed multi-event value. The experiment applies the separate-estimator
idea from the Double Q-learning paper cited above, but does not implement its
iterative learning algorithm.

| Case | Full training rows | Retained A / B rows | Validation events |
|---|---:|---:|---:|
| March, January 14 origin | 1,953 | 665 / 1,050 | 401 |
| March, February 4 origin | 1,808 | 775 / 801 | 351 |
| March, February 25 origin | 1,926 | 912 / 797 | 536 |
| May, March 12 origin | 4,239 | 2,120 / 1,881 | 409 |
| May, April 2 origin | 3,918 | 2,083 / 1,584 | 413 |
| May, April 23 origin | 3,492 | 1,885 / 1,406 | 1,066 |

**March v168:** the first origin produces 11 critic-specific entry decisions,
all rejected by the other critic. A's four entries average −35.91 bp realized
utility and B's seven average −36.91 bp. For these entries, self-evaluation
bias is +43.45 bp and cross-evaluation bias +8.57 bp; MSE falls from 9,417 to
6,769 bp². This shows selection-sensitive optimism in these smaller models.
However, the mean critic's holding MSE is worse than the full model in the
first and third origins. Its only cash entry reproduces the original useful
January 14 short. Every model stays cash in the other two origins. The complete
diagnostic takes 2.71 seconds after loading data.

**May v169:** the active third origin gives the opposite warning. A's seven
entries average +20.15 bp realized utility; B's 37 average +12.56 bp. The other
critic rejects 40 of the 44 entries. Cross-evaluation moves their bias from
−7.72 to −27.72 bp and worsens MSE from 6,366 to 7,946 bp². The mean critic's
five entries average only +5.60 bp, versus +16.77 bp for the full model's
three independent entry probes. These counts differ from continuous trading
because every probe starts from cash and settles after one event. The mean
holding forecast is worse than the full model in two of three origins. The
diagnostic takes 4.80 seconds.

Cross-evaluation therefore removes optimistic errors in one setting while
underestimating useful actions in another. More pessimism does not address
the inactive final windows. The evidence does not justify building deeper
double-critic policies or an agreement gate now. Keep the diagnostic, retain
the full-data policies as controls, and return to improving conditional inputs.

```powershell
node --conditions=development --import tsx scripts/audit-event-fitted-critics.ts --source event-policy-fitted-value-march-chain-three-v166 --output event-policy-fitted-critic-march-v168
node --conditions=development --import tsx scripts/audit-event-fitted-critics.ts --source event-policy-fitted-value-deviation-eight-v155 --output event-policy-fitted-critic-may-v169
```

### Exact-second sign dynamics and action attribution — v170–v175

The remembered one-second sign results motivate testing the actual temporal
inputs at the current event boundary. Earlier v18/v19 used five different
features from a rotating one-second sample per minute, potentially 59 seconds
old. They did not test the documented EMA2 acceleration or RSI2 at the exact
completed minute close. The immutable spot cache already contains the required
one-second history; no market download or new neural network is needed.

`event-second-dynamics.ts` derives four coordinates from exactly 64 contiguous,
completed one-second closes: last-second log return, EMA2 log-slope acceleration,
two-second EMA2 log slope per second, and centered RSI2. EMA uses alpha 2/3
seeded with the oldest close. RSI uses Wilder alpha 1/2 with gain/loss initially
zero and the following 63 arithmetic price changes. This finite initialization
matches the existing indicator engine under the same initialization; it does
not claim to reproduce its arbitrary earlier recursive state.

The last second must close exactly at the decision minute and have the same
price as the one-minute close. Missing or partial history invalidates the
feature window; observations are never carried forward. Derived daily caches
are keyed by schema, helper source hash, and both current/previous immutable
source references. Output fingerprints and helper snapshots accompany the
experiments. A source candle at 2021-12-24 04:59:54 UTC has a 362-ms close offset
instead of 999 ms despite its `closed` flag. It is left unchanged and treated
as incomplete. This makes the 05:00 minute unavailable; no sampled training or
evaluation event uses it, so no cohort rows are dropped. The first May attempt
stopped at this anomaly before creating an output directory; v171 is the
successful run after correcting the feature loader.

`--second-dynamics 0,1` compares the existing fitted basis with the same basis
plus these four inputs (`-s64`). It changes neither event labels, sample support,
account costs, action grid, nor the selected regularization. These trials add
the sign-related information to holding-value regression. The separately
trained probabilistic sign heads remain documented in v57–v139; these new trials
are not another sign-accuracy benchmark or a claim that the old 55.2207% result
transfers to this event horizon.

**March v170:** the one-event raw-price/1 control reproduces v165's models,
forecasts and trades exactly on all three origins. Adding four inputs changes
mean holding MSE from 0.0000393326 to 0.0000395402. Its first-origin return falls
from +0.6266% to −3.4078%, with drawdown rising from 1.1247% to 3.7609% and trades
from two to ten. Both other origins stay cash. This rejects the new basis for
March; no deeper new-basis screen is warranted. The model screen takes 4.04 s,
plus 25.86 s to build 184 daily derived caches on its first run.

**May v171:** the one-event deviation/0.1 control reproduces v155's first-depth
models, forecasts and trades exactly. Adding the inputs changes mean holding
MSE from 0.0000401164 to 0.0000400598. Whole-UTC-day resampling within each of
the three prior origins, with equal origin weighting and 2,000 replicates,
gives a mean MSE gain of 5.66e−8 and a 95% interval of [−1.07e−7, 2.02e−7].
The small gain is uncertain. Returns change from [0, 0, +1.9197%] to
[0, −0.3793%, +2.5070%]. The predeclared economic score improves slightly from
0.0051760 to 0.0055658. The screen takes 7.09 s after 22.03 s of cold cache
construction: 264,959 valid minute observations across 184 days.

This gives a bounded reason to test two more Bellman backups for May, not to
scale the model broadly. **v172** resumes its exact first-depth checkpoint and
fits depths two and three in 45.19 s; cached features load in 0.47 s. At depth
three, the second origin becomes profitable but the third loses substantial
profit relative to the matched old three-event model:

| May calibration origin | Existing depth 3 return | With exact seconds, depth 3 | New drawdown | New trades |
|---|---:|---:|---:|---:|
| March 12 | 0% | 0% | 0% | 0 |
| April 2 | 0% | +1.2273% | 1.6207% | 6 |
| April 23 | +17.7489% | +10.7862% | 4.4633% | 13 |

Its aggregate economic score is 0.0361822 versus 0.0529379 for the existing
depth-three control. More positive origins alone does not establish a better
policy. Extending this feature branch to depths six/eight is not justified.

**Retained comparisons:** v173 preserves all 56 March incumbent candidates and
adds the rejected one-event feature candidate. The chosen older lag-base
depth-eight policy is unchanged: +0.0922% on the final March window, with
6.3627% drawdown. The best fitted diagnostic remains the old raw-price
one-event model and stays cash. v174 preserves all 123 May incumbent candidates
and adds the three new feature depths. Selection still chooses the existing
deviation depth-eight policy. The new depth-three diagnostic, fitted on 4,025
earlier events, makes no trades over May 14–21: zero return and drawdown across
269 decisions. No final-window outcome is used to choose the candidate or its
depth. These are repeated research windows, not a fresh sealed holdout.

**Trade audit v175** reconciles every position and reproduces the prior/final
economic metrics exactly. All timestamps below are UTC. The April 2 origin adds
a short on April 3 22:05–April 4 00:27, earning +1.6127%, followed by an April 11
short losing −0.3793%. In the April 23 origin, the four cash-to-cash positions are:

| Side | Entry | Exit | Net position return |
|---|---|---|---:|
| Short | May 9 07:28 | May 9 10:55 | +1.7205% |
| Long | May 11 12:53 | May 11 13:50 | +7.1974% |
| Long | May 12 04:29 | May 12 07:45 | +2.0231% |
| Long | May 12 07:49 | May 12 13:33 | −0.4149% |

The useful behavior to preserve is holding the May 11 rebound four minutes
longer than the old policy, through another +282.48-bp price event, then exiting
before the next −143.92-bp event. However, the new policy misses the old
May 11 20:57–May 12 00:44 long, which earned +4.5494%, and changes the last
profitable May 12 position into an earlier exit followed by a losing re-entry.

`action-attribution.ts` examines five inspected timestamps using identical
10,000-equity cash/long accounts. It compares the old model, the new model, and
the new model with only its four current second inputs replaced by their
training means. This is a post-selection sensitivity probe, not a new candidate,
unbiased sample, or refit that removes the features' effects from Bellman targets.

- At May 11 13:46, old holding-versus-exit value is −1.606 bp. The new model
  gives +0.462 bp; mean substitution gives −0.362 bp. The current second inputs
  are enough to change this useful hold decision. Cash entry remains unfavorable
  in all three variants, illustrating the importance of actual inventory state.
- At May 11 20:57, the old full-long entry advantage is +0.421 bp; the new
  model gives −7.759 bp. Mean substitution restores +1.346 bp. The current
  second inputs therefore suppress the useful entry, rather than this being
  solely a shift in other refitted coefficients.
- At May 12 07:49, the new inputs turn the full-long advantage from −6.170 bp
  under mean substitution to +2.557 bp. The next price event is indeed positive
  (+80.46 bp), but the ensuing complete position loses −0.4149%. Correct next
  direction does not by itself validate the continuation and exit decisions.

The fresh final model's maximum full-long/short advantages are −7.055/−6.595 bp.
None of its four second inputs is clipped, and no event observation is missing.
Its inactivity is therefore the fitted cost-adjusted value, not a feature-join
failure. Using the earlier April 23 fit on the final window yields the same
May 16 06:24–10:14 long as the old three-event diagnostic: −0.0191% after costs,
with 1.5588% drawdown. This counterfactual is diagnostic only.

The four inputs remain an explicit research option, not the promoted policy.
The next useful question is whether short-lived directional information also
predicts persistence and value after the next event. A cheap horizon-conditioned
forecast audit should precede another deeper-policy run. The current evidence
does not support more depth, blanket critic agreement, or a cash-gate adjustment.

Verification: 71 focused tests pass, including exact indicator reconstruction,
fixed causal support, future-mutation invariance, gap rejection, and exact close
availability. All workspace package typechecks pass. Research scripts are
outside the package tsconfigs; their checks here are the executed experiments
and reproduction audits.

```powershell
node --conditions=development --import tsx scripts/screen-event-fitted-value.ts --source event-policy-quadrature-policy-march-v122 --settings-source event-policy-fitted-value-march-chain-three-v166 --second-dynamics 0,1 --output event-policy-fitted-value-exact-second-march-v170
node --conditions=development --import tsx scripts/screen-event-fitted-value.ts --source event-policy-quadrature-policy-may-v121 --settings-source event-policy-fitted-value-deviation-eight-v155 --second-dynamics 0,1 --output event-policy-fitted-value-exact-second-may-v171
node --conditions=development --import tsx scripts/screen-event-fitted-value.ts --source event-policy-quadrature-policy-may-v121 --settings-source event-policy-fitted-value-exact-second-may-v171 --resume-source event-policy-fitted-value-exact-second-may-v171 --depths 3 --output event-policy-fitted-value-exact-second-three-v172
node --conditions=development --import tsx scripts/replay-event-fitted-value.ts --source event-policy-quadrature-policy-march-v122 --screens event-policy-fitted-value-exact-second-march-v170 --incumbent event-policy-fitted-value-march-chain-policy-v167 --output event-policy-fitted-value-exact-second-march-policy-v173
node --conditions=development --import tsx scripts/replay-event-fitted-value.ts --source event-policy-quadrature-policy-may-v121 --screens event-policy-fitted-value-exact-second-three-v172 --incumbent event-policy-fitted-value-eight-policy-v158 --output event-policy-fitted-value-exact-second-may-policy-v174
node --conditions=development --import tsx scripts/audit-event-fitted-value.ts --source event-policy-fitted-value-exact-second-may-policy-v174 --output event-policy-fitted-value-exact-second-audit-v175
node --conditions=development --import tsx data/benchmarks/event-policy-fitted-value-exact-second-audit-v175/action-attribution.ts
```

The commands record the original run names, which cannot be overwritten. Use
new names for a rerun and carry those through dependent commands. The saved
configuration and source snapshots are authoritative for each historical run.

### Directional persistence audit — v176–v177

`audit-event-sign-horizons.ts` tests whether the four exact-second inputs help
predict beyond the immediate event. Each prior-origin fit compares the same
existing futures/run basis with and without the four second coordinates.
Both use fixed logistic penalty 0.1. The targets are return through event one,
two and three, plus the second and third event returns individually. Every
prediction uses only the first origin's features, including for later-event-only
targets. Each target shares a cohort with three complete contiguous events;
paths cannot cross omitted events or the training/evaluation boundary.

There are two separate losses. Ordinary logistic fitting predicts active sign.
The return-weighted version weights each training target by its absolute
arithmetic return. Its output is a weighted class score, not `P(up)`, and is
never supplied to the trading planner as an ordinary probability. This second
objective checks whether sign information survives when larger errors count
more, addressing the stopping-rule effect documented by the v139 null.

The original one-event cohorts reproduce v170/v171 exactly before requiring
three-event paths. Each loses its last two origins to that common requirement:
May training/validation counts are 4,237/407, 3,916/411 and 3,490/1,064; March's
are 1,951/399, 1,806/349 and 1,924/534. No final-window outcomes are loaded.
The 60 fits per case take 3.92 s for May and 2.05 s for March after loading.

| Target | May ordinary loss gain | May weighted loss gain | March ordinary loss gain | March weighted loss gain |
|---|---:|---:|---:|---:|
| Through event 1 | +0.000356 | +0.000754 | −0.001398 | −0.002612 |
| Through event 2 | −0.000188 | −0.000109 | −0.001456 | −0.001594 |
| Through event 3 | −0.001061 | −0.001139 | −0.000653 | −0.000695 |
| Event 2 only | −0.000303 | −0.001119 | +0.000476 | +0.000420 |
| Event 3 only | −0.000913 | −0.001089 | −0.000006 | +0.000002 |

Loss gains are equally weighted across prior origins, in nats; positive favors
the second inputs. The descriptive 2,000-replicate whole-decision-day bootstrap
includes zero for both immediate May gains. Later weighted May gains are
negative in all three origins for events two and three individually. March's
small second-event improvement is uncertain and does not improve cumulative
returns. Overlapping paths and adjacent days are dependent; these intervals
are not a formal familywise significance test.

The first event's median duration is 49/45/10 minutes in May and 40/39/23 minutes
in March. Through three events the medians are 185/160/35 and 165/161/83 minutes.
These are substantially different horizons from the remembered next-second
sign benchmark. For example, on May's active third origin, adding second inputs
changes the weighted-score direction's average gross return from 1.39 to
2.07 bp at one event, but from 6.69 to 4.45 bp through three events. These
overlapping always-directed probes omit costs and are not portfolio returns.

This rejects expanding the exact-second sign branch into longer policies.
It does not establish that every feature or every event horizon lacks signal.
The next bounded experiment changes how the existing value model evaluates its
future policy, keeping the older feature basis fixed.

The relevant local precedent is
[`forward-market-return-information-2026-08-16.md`](forward-market-return-information-2026-08-16.md):
its strongest aggressor-side signal decayed within a few seconds. That is a
different feature and target, motivating the horizon check rather than proving
its result in advance. `event-paths.ts` now supplies the common path assembler;
the original v176/v177 snapshots contain the identical helper inline. A focused
test verifies compounding, duration, truncation and rejection of broken chains.

```powershell
node --conditions=development --import tsx scripts/audit-event-sign-horizons.ts --source event-policy-fitted-value-exact-second-may-v171 --output event-policy-sign-horizons-may-v176
node --conditions=development --import tsx scripts/audit-event-sign-horizons.ts --source event-policy-fitted-value-exact-second-march-v170 --output event-policy-sign-horizons-march-v177
```

### Observed-path policy evaluation — v178–v180

The new bounded hypothesis is that repeated fitted-value targets propagate
conditional-value errors. `trainEventFittedValue` now supports a sampled-path
target mode. At depth one it is exactly the same regression as before. At
depth two or three it holds the initial inventory over the first observed
event, then chooses each future order using the already fitted table for the
remaining horizon and that future event's observed starting features. It
executes that chosen order on the following observed event, propagates equity,
price and exposure, and finally settles in cash. It does not choose an order
by maximizing the realized return of its following event.

This replaces bootstrapped continuation values with realized returns under
the earlier fitted policy. It retains regression before the current action
maximization, all 195 account-grid heads, fees, borrowing, order constraints,
the independent liquidation-support guard, and the feasible cash lower bound.
The fitted policy is frozen during a complete backup. It is trained on the same
pre-window dataset, so this does not claim out-of-fold policy evaluation or
eliminate in-sample selection bias. Observed multi-step targets also have higher
variance. Market paths are assumed exogenous to the account's orders, as in the
existing simulator; market impact is not introduced here.

Return-based policy evaluation is established RL work. For example,
[Munos et al., Safe and Efficient Off-Policy Reinforcement Learning](https://arxiv.org/abs/1606.02647)
studies multi-step returns and the efficiency/stability issues of using policy
trajectories. Our fully simulated account actions on exogenous market paths
do not implement Retrace, and its convergence results are not claimed for
this finite-data, regularized grid model.

The training signature includes every successor's features, leaf, duration,
return and excursions, plus target mode and declared path length in the
checkpoint identity. Resume rejects altered paths or changing target mode.
`--path-horizon 3 --sampled-path 0,1` creates a matched ordinary/sample-path
comparison. Both require three complete events and lose the same final two
training/validation origins. Their depth-one coefficients and predictions
match exactly. All ordinary depth-one/two/three economic metrics and orders
also reproduce the original v155 control, despite its two additional rows;
fitted order values themselves can differ.

**May v178:** six fits across three earlier origins, to depth three, take
111.75 s in total. The feature basis remains deviation/0.1 without second
inputs. The sampled two-event policy is better than its matched ordinary
two-event control, while the third sampled event is harmful:

| Origin | Ordinary depth 2 return | Sampled depth 2 return | Sampled depth 2 DD | Sampled depth 3 return | Sampled depth 3 DD |
|---|---:|---:|---:|---:|---:|
| March 12 | 0% | 0% | 0% | +0.7788% | 0.7523% |
| April 2 | 0% | +1.0666% | 1.6105% | −0.0096% | 0.6254% |
| April 23 | +3.6832% | +21.4242% | 4.7485% | +15.9460% | 11.7149% |

Depth two wins this new comparison with score 0.0661233 and 22 total trades.
This also exceeds the old depth-three score 0.0529379, but not the retained
depth-eight score 0.1102726. The sampled depth-three policy makes 62 trades in
the active origin, versus 18 at depth two; fees rise from 165.20 to 382.28.
Its training MSE rises with depth, consistent with the noisier sampled target;
this is not an error against an independently observed optimal value.

**Retained final comparison v179:** all 126 incumbent candidates reproduce;
six matched-path candidates bring the total to 132. The old depth-eight policy
remains selected. The best new diagnostic is the sampled two-event model,
refitted on 4,023 complete earlier paths. Its final May replay makes no trades
over 269 decisions. Selection is saved before reading the final outcome. The
fit/replay takes 9.44 s. No new policy is promoted as reliably profitable.

**Trade audit v180:** the sampled two-event model adds useful shorts and improves
some entry timing, while retaining an incorrect early rebound exit. On the
active origin its cash-to-cash positions are:

| Side | Entry UTC | Exit UTC | Net position return |
|---|---|---|---:|
| Short | May 1 22:57 | May 2 00:17 | −0.8489% |
| Short | May 4 19:09 | May 5 14:33 | +1.2251% |
| Short | May 8 17:02 | May 9 02:57 | +1.9645% |
| Long | May 11 12:53 | May 11 13:46 | +4.3510% |
| Short | May 11 13:50 | May 11 17:37 | +2.2985% |
| Long | May 11 21:29 | May 12 00:44 | +6.2207% |
| Long | May 12 05:15 | May 12 13:44 | +4.6394% |

The May 11 13:50 short is a useful newly learned response after the rebound.
The preceding long still exits at 13:46 and misses the next +282.48-bp rise.
The later 21:29 long avoids the original depth-three policy's earlier 20:57
entry and earns a better position return. By contrast, the sampled depth-three
model holds a May 11 21:02–May 12 14:08 long through adverse movement and loses
1.9497%; its many added trades and risk cannot be justified by the higher depth.
The saved cash-to-cash audit identifies its one interval with a reversal as
such; that mixed interval is not mislabeled as a single uninterrupted long.

The final two-event model's maximum full-long/short entry advantages are
−5.483/−6.789 bp. The previous April 23 fit also remains cash on the final
window, with maxima −4.973/−7.678 bp. Neither fit has clipped external inputs
there. Thus this inactivity is not caused solely by the final refit or missing
features. All prior/final replay metrics reconcile exactly. The remaining
question is transfer of the improved two-event policy; a bounded March
comparison is warranted, while further May depth is not.

The sampled-target regression test compares its raw cash target to the
explicit expected return of the same frozen long action on symmetric future
outcomes. It would fail if each outcome selected its own hindsight action.
It also checks unchanged depth one, exact resumed training, changed-path
rejection and insufficient-horizon rejection. All 73 focused tests and all
workspace package typechecks pass.

```powershell
node --conditions=development --import tsx scripts/screen-event-fitted-value.ts --source event-policy-quadrature-policy-may-v121 --settings-source event-policy-fitted-value-deviation-eight-v155 --depths 3 --path-horizon 3 --sampled-path 0,1 --output event-policy-fitted-value-sampled-path-may-v178
node --conditions=development --import tsx scripts/replay-event-fitted-value.ts --source event-policy-quadrature-policy-may-v121 --screens event-policy-fitted-value-sampled-path-may-v178 --incumbent event-policy-fitted-value-exact-second-may-policy-v174 --output event-policy-fitted-value-sampled-path-policy-v179
node --conditions=development --import tsx scripts/audit-event-fitted-value.ts --source event-policy-fitted-value-sampled-path-policy-v179 --output event-policy-fitted-value-sampled-path-audit-v180
```

### Observed-path transfer rejection — v181–v182

The May two-event improvement warrants a bounded transfer check, not more
depth. v181 keeps March's earlier selected raw-price/1 basis, compares ordinary
and sampled depth one/two, and requires the same three-event training cohort
used in May. It takes 25.64 s. All first-depth tables and coordinates match
between the two target modes, and the ordinary control's economic results
reproduce v166 exactly.

The sampled second depth changes the January 14 origin from +0.6266% to
−1.4502%, raises drawdown from 1.1247% to 3.7411%, and increases orders from two
to six. Both remaining origins stay cash. The six orders form one cash-to-cash
cycle with three direction reversals, not six independent round trips:

| January 14 UTC | Sampled policy action | Following event return |
|---|---|---:|
| 00:39 | Enter long | +107.98 bp |
| 00:40 | Reverse to short | +93.07 bp |
| 00:41 | Reduce short to restore leverage cap | −61.58 bp |
| 00:43 | Reverse to long | −87.36 bp |
| 00:45 | Reverse to short | −86.63 bp |
| 09:12 | Exit short | −131.80 bp |

The initial long is useful, but the next two reversals are on the wrong side
of the following move. The last short remains open until 09:12. The ordinary
policy enters only the 00:45 short and exits at 00:51 with +62.66 net PnL per
10,000 initial equity; the sampled cycle loses 145.02. This rejects applying
the sampled-target method broadly. Its apparent May improvement does not
establish transfer or remove the need to calibrate conditional action value.

v182 preserves all 57 incumbent candidates and adds the four matched-path
variants. Selection remains the old lag-base depth-eight policy, with its
unchanged final March +0.0922% return and 6.3627% drawdown. The best new fitted
diagnostic is the ordinary one-event control; it fits 1,695 complete paths and
stays cash on the final window. The rejected sampled model is not force-tested
on final outcomes. The fit/replay takes 1.28 s.

The useful next check is attribution of the large continuation changes to
training paths: are the reversal values systematic or driven by a small set of
high-variance sampled outcomes? This should precede either deeper rollouts or
another model-size increase. The earlier fitted policy is frozen during each
backup but learned in-sample; independent or chronological policy-evaluation
targets remain a possible follow-up if the attribution supports that diagnosis.

The v178/v181 generic configuration `target` sentence originally described
only the ordinary control. `target-clarification.json` records the sampled
alternative; the explicit settings, serialized `targetMode` and source
snapshots identify what actually ran. Original configuration bytes are retained
for downstream hashes, and new screens now write the complete target description.

```powershell
node --conditions=development --import tsx scripts/screen-event-fitted-value.ts --source event-policy-quadrature-policy-march-v122 --settings-source event-policy-fitted-value-march-chain-three-v166 --depths 2 --path-horizon 3 --sampled-path 0,1 --output event-policy-fitted-value-sampled-path-march-v181
node --conditions=development --import tsx scripts/replay-event-fitted-value.ts --source event-policy-quadrature-policy-march-v122 --screens event-policy-fitted-value-sampled-path-march-v181 --incumbent event-policy-fitted-value-exact-second-march-policy-v173 --output event-policy-fitted-value-sampled-path-march-policy-v182
```

### Exact sampled-target influence — v183–v184

The new `audit-event-fitted-influence.ts` reconstructs both saved models' exact
training paths, verifies their signatures and identical first-depth tables,
and attributes the second-depth change at each executed sampled-policy order.
Each comparison uses the actual sampled policy's pre-order account for both
models. It does not pretend the ordinary policy arrived with that inventory.

For fixed standardized inputs, ridge prediction is linear in its labels:
`w_i = x_i' (X'X/n + lambda D)^-1 x_query / n`, with an unpenalized intercept.
The audit combines those weights with the exact equity/price/exposure grid
interpolation. For each path, it computes the change from predicted H1
continuation to the realized return of the shared H1-selected action, including
terminal settlement. Contributions reconcile the raw action-value change to
within 1e-7 bp. A separately reported cash-floor correction reconciles actual
clamped predictions. This is fixed-design label attribution, not leave-one-out
retraining of features or of the H1 policy. Signed ridge weights are not
probabilities, and `1/sum(w_i^2)` is not a count of independent samples.

v183 audits all six March orders in 3.33 s. At the wrong January 14 00:40 UTC
long-to-short reversal, the ordinary model prefers exiting: the short's
advantage over exit is −6.1899 bp. Sampled evaluation changes it to +1.5102 bp.
The raw change is +11.1376 bp; the cash floor reduces it by 3.4375 bp.
Absolute futures/spot spread and five-minute relative return are both clipped
at five training standard deviations, contributing +6.9200 and +5.1090 bp.
This is not a claim that they exceed the empirical training range.

One November 8 path has a +457.82 bp second move, adverse to short exposure.
Its negative regression weight (−0.008236) turns its −470.69 bp target change
into a **+3.8765 bp** contribution to the current short. However, this is not
one removable outlier: the ten largest absolute contributions are only 17.69%
of total absolute contribution. The initial useful long and later wrong long
also involve broad positive/negative cancellation.

v184 audits 22 May orders in 7.78 s. Useful May entries likewise rely on
extrapolation. The profitable May 11 21:29 long shifts from −5.3911 bp under
the ordinary model to +0.5222 bp under sampled evaluation. Its negative weight
mass is 2.4065, larger than 1.0778 at March's wrong 00:40 reversal; its ten
largest absolute contributions account for only 4.98%. Some other decisions
are sensitive to particular dates: December 4 contributes +11.24 bp of the
April 21 entry's +19.07 bp net change. A blanket negative-weight or confidence
gate would therefore remove useful trades as well. No such gate is promoted.

```powershell
node --conditions=development --import tsx scripts/audit-event-fitted-influence.ts --source event-policy-fitted-value-sampled-path-march-v181 --output event-policy-fitted-influence-march-v183
node --conditions=development --import tsx scripts/audit-event-fitted-influence.ts --source event-policy-fitted-value-sampled-path-may-v178 --output event-policy-fitted-influence-may-v184
```

### Centered futures spread and matched control — v185–v189

The targeted input change removes absolute futures/spot spread and retains
relative one/five-minute returns plus deviations of spread from its completed
5/15/60-minute EMAs. All candidates require the same 240-minute paired history
and three completed events. Penalties remain the previously selected 1 for
March and 0.1 for May; depth remains two with sampled continuation. No trading
rule, cost, or order threshold changes. `eventFittedFuturesInputs` centralizes
the layout across screens, replay and audits; tests cover the old layouts and
invariance of centered inputs to the removed absolute coordinate.

March's previous raw-price basis has 26 inputs; centered has 28. v187 adds the
three deviations while retaining absolute spread (29 inputs), so comparison
with v185 isolates removing absolute spread rather than conflating removal
with adding the deviation history. All March variants use 1,951/1,806/1,924
training paths and 399/349/534 validation paths. May uses
4,237/3,916/3,490 and 407/411/1,064 respectively, matching v178.

| Depth-two variant | Prior-origin returns, chronological | Active-origin orders | Active-origin drawdown | Mean selection score |
|---|---|---:|---:|---:|
| March raw price, v181 | −1.4502%, 0, 0 | 6 | 3.7411% | negative |
| March raw plus deviations, v187 | +0.0208%, 0, 0 | 5 | 1.7606% | −0.000518 |
| March centered, v185 | +1.3006%, 0, 0 | 6 | 2.0067% | +0.003639 |
| May raw plus deviations, v178 | 0, +1.0666%, +21.4242% | 18 | 4.7485% | +0.066123 |
| May centered, v186 | 0, +0.5620%, +0.0679% | 5 | 2.9142% | +0.000828 |

The March centered model preserves the useful January 14 00:39 long, holds
through the two previous wrong reversals, reverses short at 00:45, reduces
short at 09:12 and exits at 09:18. That cash cycle earns 276.10 per 10,000
initial equity. It then makes a bad January 30 19:10 long entry before a
−120.62 bp event and exits at 19:11, losing 146.04 of the earlier gain. The
additive-deviation control avoids the earlier wrong reversals too, but misses
the initial useful long; it enters only the 00:45 short and also takes the bad
January 30 long. Thus the centered representation fixes a specific trajectory
without solving conditional direction generally.

The forecast metric does not explain an across-regime improvement: centered
one-event MSE skills are −0.6053% in March and −0.3776% in May. The screens take
14.59 s (March centered), 29.32 s (May centered) and 14.58 s (March additive
control). May's lost trading gain rejects adopting this representation broadly.

v188 retains all 61 March incumbent candidates and adds four, preserving the
old lag-base depth-eight selection and its final +0.0922% return / 6.3627%
drawdown. The best new centered depth-two diagnostic trains on 1,695 complete
paths and stays cash for all 223 final decisions. v189 retains all 132 May
candidates and adds two. The old deviation depth-eight model remains selected;
both it and the centered diagnostic stay cash. The latter trains on 4,023
complete paths and makes 269 final decisions. Fit/replay takes 4.58 s and
9.89 s respectively. Neither diagnostic is a new promoted policy.

The next bounded check is whether choosing continuation actions using H1
trained on the same observed outcomes inflates sampled targets. Attribution
does not by itself establish that bias, so measure it before increasing depth
or replacing the value learner.

```powershell
node --conditions=development --import tsx scripts/screen-event-fitted-value.ts --source event-policy-quadrature-policy-march-v122 --settings-source event-policy-fitted-value-sampled-path-march-v181 --basis centered --depths 2 --sampled-path 1 --output event-policy-fitted-value-centered-march-v185
node --conditions=development --import tsx scripts/screen-event-fitted-value.ts --source event-policy-quadrature-policy-may-v121 --settings-source event-policy-fitted-value-sampled-path-may-v178 --basis centered --depths 2 --sampled-path 1 --output event-policy-fitted-value-centered-may-v186
node --conditions=development --import tsx scripts/screen-event-fitted-value.ts --source event-policy-quadrature-policy-march-v122 --settings-source event-policy-fitted-value-sampled-path-march-v181 --basis deviation --depths 2 --sampled-path 1 --output event-policy-fitted-value-deviation-sampled-march-v187
node --conditions=development --import tsx scripts/replay-event-fitted-value.ts --source event-policy-quadrature-policy-march-v122 --screens event-policy-fitted-value-centered-march-v185,event-policy-fitted-value-deviation-sampled-march-v187 --incumbent event-policy-fitted-value-sampled-path-march-policy-v182 --output event-policy-fitted-value-centered-march-policy-v188
node --conditions=development --import tsx scripts/replay-event-fitted-value.ts --source event-policy-quadrature-policy-may-v121 --screens event-policy-fitted-value-centered-may-v186 --incumbent event-policy-fitted-value-sampled-path-policy-v179 --output event-policy-fitted-value-centered-may-policy-v189
```

### Excluded-block continuation diagnosis — v190–v191

The sign question now concerns the action selected at the next event state.
The sampled backup's H1 model was fitted on the same pre-window outcomes on
which its selected action is subsequently evaluated. v190/v191 measure that
effect before changing the backup. Six fixed calendar blocks divide each
nominal training interval. Each second-event outcome belongs to one block by
its start time. A complementary H1 fit excludes every training row whose full
one-day feature/label support overlaps that block, extending the exclusion
through any held-out event that ends beyond the block. Means and scales are
refitted too. A past-only comparison uses strictly earlier rows after two
blocks of warmup. The helper tests cover long event tails, boundary equality,
history purging, complete evaluation coverage and exclusion of future rows.

Both versions probe starting exposures −1, 0 and +1 at the account-grid center,
mark the first move, select a second-event action using H1 and then evaluate
the observed second move with fees and terminal settlement. The full model's
H1 tables, coordinates and path signatures reproduce the saved source exactly.
This is a training-target diagnostic, not a new backtest return. The three
inventory probes per path are dependent; direction accuracy counts the
long-versus-short holding-value ordering, not a calibrated sign probability.

| Source / prior origin | Full-data holding direction | Block-held-out holding direction | Change in realized continuation per probe |
|---|---:|---:|---:|
| March / January 14 | 59.00% | 55.56% | −1.572 bp |
| March / February 4 | 57.75% | 54.15% | −1.044 bp |
| March / February 25 | 54.63% | 48.91% | −1.838 bp |
| May / March 12 | 52.09% | 50.30% | −1.150 bp |
| May / April 2 | 52.09% | 50.54% | −1.067 bp |
| May / April 23 | 52.23% | 50.40% | −1.091 bp |

Past-only mean continuation differences are −1.254/−1.064/−1.736 bp in March
and −0.635/−0.782/−1.651 bp in May on their smaller matched subsets. Cash-only
differences are much smaller and mixed in sign: most of the aggregate effect
is on continuation/exit decisions from existing inventory. The mean full-H1
predicted value actually *underestimates* its realized value in these probes;
the evidence is an advantage from selecting on seen outcomes, not proof that
all fitted predictions are upward biased. Smaller training sets and temporal
regime changes confound the difference with pure statistical overfitting.

The complementary models can see later pre-origin blocks. Only their H1 heads
and normalization are held out; the shared account axes and leaf liquidation
guards retain the original full-training fit. Even the past-only head trial
therefore does not claim a fully refitted causal market model. Neither trial
uses final inspector prices. Runtimes are 3.51 s for March and 6.73 s for May.

The literature supports diagnostics rather than a universal correction here.
[Gottesman et al. (2020)](https://proceedings.mlr.press/v119/gottesman20a.html)
study influential transitions for fitted-Q evaluation, including linear and
kernel models; their removal-based influence analysis is distinct from our
fixed-design label decomposition. [Wang et al. (2024)](https://proceedings.mlr.press/v235/wang24be.html)
derive fitted-Q evaluation rates under stated completeness assumptions. Those
results do not establish that our restricted linear holding heads satisfy the
required assumptions or that splitting data makes them reliable. The bounded
implementation below is an empirical sample-splitting trial, not either
paper's estimator or a transferred guarantee.

```powershell
node --conditions=development --import tsx scripts/audit-event-continuation-crossfit.ts --source event-policy-fitted-value-sampled-path-march-v181 --output event-policy-continuation-crossfit-march-v190
node --conditions=development --import tsx scripts/audit-event-continuation-crossfit.ts --source event-policy-fitted-value-sampled-path-may-v178 --output event-policy-continuation-crossfit-may-v191
```

### Held-out actions in the sampled backup — v192–v196

`trainEventFittedValue` now accepts optional fixed continuation policies for
each observed path. At depth two, the supplied block-held-out H1 chooses the
next action before its realized return is consumed. The full-data H1 still
trains on all matched rows; only H2 labels change. External policies must have
matching costs, account grids and ruin guards. Checkpoint identity includes
their serialized fitted content and row assignment, so changing a head or
swapping its evaluation rows cannot silently resume a different experiment.
An analytic test supplies policies that both choose the wrong future side and
checks the resulting loss, rather than a hindsight maximum. Resume, changed
coefficients, missing policies and incompatible cost contracts are covered.

The `continuationFolds: 6` research setting adds `-cf6` to artifact names and is
limited to sampled depth one/two. Six complementary H1 fits are rebuilt from
the same purged blocks used in the audit. They exactly reproduce every audit
model and split. Saved reproduction checks also prove that sample counts,
H1 tables, normalization, forecast metrics and depth-one economic replay remain
identical to the original sampled controls. No fee, size, leverage, feature,
penalty, clock or execution setting changes.

| Variant | Prior-origin returns, chronological | Active-origin drawdown | Active-origin orders | Mean selection score |
|---|---|---:|---:|---:|
| March sampled H2, v181 | −1.4502%, 0, 0 | 3.7411% | 6 | negative |
| March held-out H2, v192 | +0.6266%, 0, 0 | 1.1247% | 2 | +0.001707 |
| May sampled H2, v178 | 0, +1.0666%, +21.4242% | 4.7485% | 18 | +0.066123 |
| May held-out H2, v193 | 0, 0, +6.0994% | 3.6032% | 11 | +0.018534 |

March's held-out policy executes exactly the useful January 14 00:45–00:51
short from the original H1/ordinary-H2 control, net +62.66 per 10,000 starting
equity. It avoids the losing reversal sequence but also skips the earlier
useful 00:39 long. Depth two provides no economic gain over depth one in this
trial. Runtime is 15.27 s for all three priors.

May's held-out policy retains the losing May 1 short (−0.8489%), earns only
+0.1585% in its May 8 short, preserves the May 11 12:53–13:46 long (+4.3510%),
then takes two later May 12 longs (+1.9197% and +0.4553%). It skips the May 4
short, the May 11 13:50 short and the profitable overnight May 11 long, and
enters the May 12 rebound later than the original sampled policy. These five
cash cycles reconcile exactly to +609.94 final PnL; fees are 122.63 and borrow
0.3340. Runtime is 28.87 s.

A local action comparison makes the failure concrete. Both models evaluate
the original selected order versus cash at the original actual pre-entry
account; these are counterfactual values, not reconstructed trajectories:

| Original entry UTC | Original sampled advantage | Held-out value of the same action |
|---|---:|---:|
| May 1 22:57 short, losing cycle | +2.4949 bp | +3.7320 bp |
| May 4 19:09 partial short | +0.0619 bp | −0.0065 bp |
| May 11 12:53 long, preserved | +5.1154 bp | +1.3353 bp |
| May 11 13:50 partial short | +0.1056 bp | −1.1412 bp |
| May 11 21:29 overnight long | +0.5222 bp | −6.8639 bp |
| May 12 05:15 long | +2.8472 bp | −1.6993 bp |

Thus excluded-block targets do not simply remove overconfident bad trades.
They increase confidence in the first losing short while rejecting several
profitable entries. The preserved May 11 rebound is useful behavior, but the
net comparison rejects broad adoption over existing candidates.

v194 retains all 65 March candidates and adds two. The old lag-base depth-eight
selection and final +0.0922% / 6.3627% drawdown are unchanged. New H1 wins the
tie with new H2 and stays cash on the final 223 decisions. v195 retains all 134
May candidates and adds two; the old deviation depth-eight model remains
selected and cash. Its new H2 diagnostic also stays cash across 269 decisions.
Final complete-path counts are 1,695 and 4,023; refit/replay takes 2.19 s and
10.42 s respectively.

v196 reproduces all three May prior replays and the final diagnostic metrics
exactly. Maximum final fresh-cash full-exposure advantages are −7.6527 bp for
long and −5.6269 bp for short. Keeping the previous April 23 held-out fit also
stays cash (maxima −7.6831/−7.3325 bp). This is not solely a last-refit issue.
Partial-order advantages are evaluated separately in the action audit above;
a negative full-size probe alone does not imply every partial entry is bad.

All 77 focused tests and workspace typechecks pass. No selected policy is
replaced. Before testing greater depth, the next useful attribution is the
held-out-minus-original target change at the retained wrong short and rejected
overnight long: which training states change the relative holding/exit values,
and does a small set of calendar blocks account for their opposite shifts?
This follows the observed failure rather than adding another global sign or
confidence threshold.

```powershell
node --conditions=development --import tsx scripts/screen-event-fitted-value.ts --source event-policy-quadrature-policy-march-v122 --settings-source event-policy-fitted-value-sampled-path-march-v181 --depths 2 --sampled-path 1 --continuation-folds 6 --output event-policy-fitted-value-crossfit-march-v192
node --conditions=development --import tsx scripts/screen-event-fitted-value.ts --source event-policy-quadrature-policy-may-v121 --settings-source event-policy-fitted-value-sampled-path-may-v178 --depths 2 --sampled-path 1 --continuation-folds 6 --output event-policy-fitted-value-crossfit-may-v193
node --conditions=development --import tsx scripts/replay-event-fitted-value.ts --source event-policy-quadrature-policy-march-v122 --screens event-policy-fitted-value-crossfit-march-v192 --incumbent event-policy-fitted-value-centered-march-policy-v188 --output event-policy-fitted-value-crossfit-march-policy-v194
node --conditions=development --import tsx scripts/replay-event-fitted-value.ts --source event-policy-quadrature-policy-may-v121 --screens event-policy-fitted-value-crossfit-may-v193 --incumbent event-policy-fitted-value-centered-may-policy-v189 --output event-policy-fitted-value-crossfit-may-policy-v195
node --conditions=development --import tsx scripts/audit-event-fitted-value.ts --source event-policy-fitted-value-crossfit-may-policy-v195 --output event-policy-fitted-value-crossfit-may-audit-v196
node --conditions=development --import tsx data/benchmarks/event-policy-fitted-value-crossfit-may-audit-v196/action-contrast-audit.ts
```

### Attribution of the held-out target changes — v197–v198

The influence audit now accepts `--crossfit-source`. It reconstructs every
held-out H1 fit and split, verifies them against saved artifacts, and verifies
both complete-path training signatures. Each target difference is the realized
return of the held-out-selected next action minus that of the full-data-selected
next action, including different trading costs and terminal settlement. The
first event's holding return cancels. Fixed ridge weights and account-grid
interpolation then reconstruct each H2 action contrast; a separate cash-floor
term reconciles actual predictions to within 1e-7 bp. The reference orders and
accounts remain those of the original sampled trajectory.

v197 takes 11.44 s for May; v198 takes 4.63 s for March. They attribute every
executed original sampled order, including those the held-out policy rejects.
The misleading May 1 short shifts from +2.4949 to +3.7320 bp. Its +1.2371 bp
increase is a balance of all six blocks, with +0.9796/+0.7370 bp from the last
two blocks and −1.2458 bp from the fourth. Ten largest absolute path
contributions account for only 8.82% of absolute influence. It is not one
removable training anomaly.

The profitable May 11 21:29 long shifts from +0.5222 to −6.8639 bp. All six
blocks contribute negatively to the −7.3861 bp change. The January 13–February
2 block contributes −2.1257 bp and February 22–March 14 contributes −3.0001 bp;
the others contribute −0.3864/−0.2992/−0.3352/−1.2395 bp. Absolute spread and
five-minute realized volatility account for −4.0725 and −3.6673 bp of the
coefficient-based decomposition, offset by other inputs. Ten largest paths
explain only 13.18% of absolute influence. Calendar-block deletion is therefore
not a supported selective correction for this trade.

March's wrong January 14 00:40 short shifts from +1.5102 to −1.9284 bp. The
raw change is −6.9564 bp, partly offset by +3.5178 bp from the cash floor. The
November 8 +457.82 bp next-move path now contributes −4.0666 bp via a negative
ridge weight: it had been a major source of optimism in v183. The same held-out
change also rejects the useful 00:39 long (+3.0243 to −1.7020 bp). Correcting a
measured source of extrapolation still does not identify only bad actions.

```powershell
node --conditions=development --import tsx scripts/audit-event-fitted-influence.ts --source event-policy-fitted-value-sampled-path-may-v178 --crossfit-source event-policy-fitted-value-crossfit-may-v193 --output event-policy-fitted-influence-crossfit-may-v197
node --conditions=development --import tsx scripts/audit-event-fitted-influence.ts --source event-policy-fitted-value-sampled-path-march-v181 --crossfit-source event-policy-fitted-value-crossfit-march-v192 --output event-policy-fitted-influence-crossfit-march-v198
```

### Smaller held-out blocks — v199–v204

The remaining confound is loss of training history: six calendar folds remove
20 days plus overlapping input/label support from each H1 fit. The fixed
alternative uses 20 folds of six days, with exactly the same one-day support
purge. It retains roughly 93% of rows on average across folds. Retention is not
uniform: March's busiest excluded block still leaves only 68–71% of rows, while
May's minimum is 86–89%. Event clustering makes calendar duration different from
sample count. This is one predeclared contrast, not a search over block sizes.

v199/v200 repeat the H1 diagnostic in 4.33/9.14 s. Complementary H1 direction
accuracy remains below the full-data estimate: 55.66/54.93/49.69% in March and
50.06/50.41/50.20% in May. Realized continuation differences remain
−1.255/−0.919/−1.689 bp and −1.014/−1.062/−1.149 bp per inventory probe. These
results weaken the explanation that the earlier gap was solely caused by
removing too much training history. They do not remove temporal dependence or
prove pure self-selection bias. Past-only comparisons now warm up through seven
blocks (42 days); their smaller subsets are not asserted equal to v190/v191.

v201/v202 apply those 20 complementary H1 policies to H2 labels. Full-data H1,
its normalization, the matched cohort, forecasts and economic replay remain
exactly unchanged; every complementary policy and split reproduces the
diagnostic artifacts. Costs, inputs and depth remain fixed.

| Held-out H2 | Prior-origin returns, chronological | Active-origin drawdown | Active-origin orders |
|---|---|---:|---:|
| March, six folds, v192 | +0.6266%, 0, 0 | 1.1247% | 2 |
| March, twenty folds, v201 | 0, 0, 0 | 0 | 0 |
| May, six folds, v193 | 0, 0, +6.0994% | 3.6032% | 11 |
| May, twenty folds, v202 | 0, 0, +7.1525% | 3.6032% | 16 |

The smaller blocks improve May's May 8 short to +1.6485%, but reduce exposure
to the useful May 11 rebound, earning +2.6118% instead of +4.3510%. They add an
unprofitable May 11 20:57–21:26 long (−0.2398%), still miss the profitable later
overnight position, and retain the initial losing May 1 short (−0.8489%). An
earlier May 12 05:15–07:45 long earns +3.3896%, followed by +0.4553% on the
07:58–10:31 long. Six cash cycles reconcile to +715.25 PnL, with 133.07 fees
and 0.4358 borrow. Screens take 16.53/30.89 s. More trades and a small local gain
do not establish transfer, and this is below the original sampled May result.

v203/v204 preserve all 67/136 incumbent candidates and add two each. Selection
remains March's old lag-base depth eight and May's old deviation depth eight.
The new diagnostics stay cash on final windows: March's H1 is preferred over
its inactive H2; May's H2 makes no trades. Final training uses 1,695/4,023
complete paths, with refit/replay taking 2.75/11.05 s. Existing selected final
returns and drawdowns are unchanged. No further block-count or depth search is
justified by this contrast.

```powershell
node --conditions=development --import tsx scripts/audit-event-continuation-crossfit.ts --source event-policy-fitted-value-sampled-path-march-v181 --folds 20 --output event-policy-continuation-crossfit20-march-v199
node --conditions=development --import tsx scripts/audit-event-continuation-crossfit.ts --source event-policy-fitted-value-sampled-path-may-v178 --folds 20 --output event-policy-continuation-crossfit20-may-v200
node --conditions=development --import tsx scripts/screen-event-fitted-value.ts --source event-policy-quadrature-policy-march-v122 --settings-source event-policy-fitted-value-sampled-path-march-v181 --depths 2 --sampled-path 1 --continuation-folds 20 --output event-policy-fitted-value-crossfit20-march-v201
node --conditions=development --import tsx scripts/screen-event-fitted-value.ts --source event-policy-quadrature-policy-may-v121 --settings-source event-policy-fitted-value-sampled-path-may-v178 --depths 2 --sampled-path 1 --continuation-folds 20 --output event-policy-fitted-value-crossfit20-may-v202
node --conditions=development --import tsx scripts/replay-event-fitted-value.ts --source event-policy-quadrature-policy-march-v122 --screens event-policy-fitted-value-crossfit20-march-v201 --incumbent event-policy-fitted-value-crossfit-march-policy-v194 --output event-policy-fitted-value-crossfit20-march-policy-v203
node --conditions=development --import tsx scripts/replay-event-fitted-value.ts --source event-policy-quadrature-policy-may-v121 --screens event-policy-fitted-value-crossfit20-may-v202 --incumbent event-policy-fitted-value-crossfit-may-policy-v195 --output event-policy-fitted-value-crossfit20-may-policy-v204
```

### Positive local averages and the clock baseline — v205–v211

The repeated signed-extrapolation findings motivate a different approximation
class rather than another pessimism threshold. [Ormoneit and Glynn (2002)](https://web.stanford.edu/~glynn/papers/2002/OrmoneitG02.html)
study approximate dynamic programming using local averaging, including nearest
neighbors, grids and trees. Their stability and asymptotic statements concern
their stated MDP setting; they are not guarantees for this trading pipeline.
Here the first gate is simply whether local averaging improves H1 forecasts.

`eventLocalValuePredictor` constructs uniform positive weights over 32, 128 or
512 nearest historical events. Neighbor selection depends only on Euclidean
distance in the exact training-standardized, clipped feature space used by
ridge. The same weights average both signed holding values, return, duration
and sign indicators, retaining a coherent empirical event law. They cannot turn
a worse observed label into a better prediction through a negative coefficient.
Tests check shared target-independent neighbors, bounded extrapolation,
deterministic ties, copied training inputs and invalid settings.

v205/v206 use the exact matched three-event cohort and H1 targets from
v181/v178. Path signatures, sample counts and ridge MSE reproduce; only first
event forecasts are scored, including terminal settlement but not entry
execution. They take 3.14 s for March and 6.40 s for May. No Bellman expansion,
execution backtest or final-window evaluation is performed for this family.

| Neighbors | March mean value-MSE skill vs zero return | May mean value-MSE skill vs zero return |
|---:|---:|---:|
| 32 | −4.2964% | −3.4877% |
| 128 | −0.4086% | −0.3971% |
| 512 | −0.1449% | +0.1035% |

512 neighbors has the lowest mean forecast error in each case. It improves
ridge MSE by only 0.2933% in March and 0.0187% in May. v207/v208 resample whole
decision days jointly for the two models, then average origin differences
equally. The local-minus-ridge MSE difference is −11.54 bp² in March with a
descriptive 95% interval [−35.46, +11.09], and −0.75 bp² in May with
[−30.33, +28.73]. Both include zero. These are exploratory intervals after
neighbor-count selection on reused research origins; adjacent days and origins
can remain dependent. They do not establish a significant economic gain.

The initially attractive 59–63% sign accuracy requires a stronger control.
v209 adds opposite-run baselines; v210/v211 also estimate a Laplace-smoothed
P(up | current directional-change sign) using training events only. Neither
baseline sees the next move.

| Prior origin | Local sign accuracy | Predict opposite DC state | Local prediction agreement with opposite DC state |
|---|---:|---:|---:|
| January 14, March case | 63.1579% | 62.9073% | 99.7494% |
| February 4, March case | 59.3123% | 59.3123% | 100% |
| February 25, March case | 61.0487% | 61.0487% | 100% |
| March 12, May case | 61.4251% | 61.4251% | 100% |
| April 2, May case | 62.2871% | 62.5304% | 99.7567% |
| April 23, May case | 61.2782% | 61.3722% | 99.9060% |

In every May origin, the local model's predicted sign is **exactly identical**
to the opposite raw-candle run sign. March agreement with that raw rule is
99.50–99.81%. High classification accuracy mostly recovers the asymmetric
stopping-time behavior of this clock, consistent with v139's earlier martingale
null; it does not imply that wrong predictions have equal magnitude to right
ones. The local model's holding-value direction accuracy is only 51–55%.

The direction-only probability baseline has better Brier score than the local
model in all three March origins. Local Brier improves slightly in May, but
its gain beyond the direction-only baseline is tiny in the active origin
(0.236827 versus 0.236842). For May's last training fit, the state with roughly
59.1% P(up) still has mean full-long holding value −13.31 bp after terminal
settlement, before paying entry costs. Predicting the more frequent sign is
not the same as predicting positive expected log growth.

This rejects expanding this local law into deeper Bellman training at present.
It remains an inspectable forecast experiment, not a promoted policy. All 78
focused tests pass. The next learner or input change should demonstrate
conditional expected holding-value improvement beyond the simple clock-state
control, not just another sign-accuracy increase. This preserves the original
profitability goal while avoiding another expansion of a weak forecast.

```powershell
node --conditions=development --import tsx scripts/screen-event-local-value.ts --source event-policy-fitted-value-sampled-path-march-v181 --output event-policy-local-value-march-v205
node --conditions=development --import tsx scripts/screen-event-local-value.ts --source event-policy-fitted-value-sampled-path-may-v178 --output event-policy-local-value-may-v206
node --conditions=development --import tsx scripts/audit-event-local-value.ts --source event-policy-local-value-march-v205 --output event-policy-local-value-clock-march-v210
node --conditions=development --import tsx scripts/audit-event-local-value.ts --source event-policy-local-value-may-v206 --output event-policy-local-value-clock-may-v211
```

### Completed candle close locations — v212–v216

The [feature-availability audit](all-feature-availability-audit-2026-08-19.md)
identified `futures-close-location-1m` for size-conditioned sign and magnitude
targets. The current fitted inputs had returns, run state, ranges, realized
volatility, basis and basis deviations, but no location of the close within
the last candle. This is a small, archive-backed addition worth testing before
another learner expansion. The [cross-asset study](binance-cross-asset-component-feature-bases-2026-08-20.md)
also records many discovery wins failing transfer; it does not support a broad
asset search here. A reference-file inventory found only 9/184, 16/184 and
13/184 required days of BTC spot one-second trade-flow in the May, March and
January training/prior intervals, and no ETH spot/futures one-minute references
in those intervals. These are file-presence counts, not verified payload
coverage. No new market data was downloaded.

`eventCandleShapes` computes two inputs from the completed spot and futures
minute at the exact decision timestamp: `2 * (close - low) / (high - low) - 1`.
A valid flat candle yields zero. Missing, misaligned, nonfinite, nonpositive
or inconsistent OHLC yields no observation. The helper only reads that completed
minute; tests cover future-candle mutation, scale invariance, flat observations,
invalid bounds and missing sources. The feature flag is part of the setting
name and is carried through training, replay and audit input construction.

v212/v213 compare the existing ridge model against the same model with both
locations appended. Penalty, base features, normalization method, costs,
liquidation guard and complete three-event cohort stay fixed. Both members of
each pair require valid shapes, so missing data cannot change only one arm's
sample. This is an H1 forecast gate; the sampled-path flag retains the previous
cohort and signatures but has no deeper continuation at H1. The spot and futures
feature pair is one declared test, not separate feature selection trials.

| Prior-origin suite | Control mean H1 MSE | With close locations | Change, bp² | Change versus clock-state mean, bp² |
| --- | ---: | ---: | ---: | ---: |
| March, raw futures price, penalty 1 | 0.000039327505 | 0.000039349718 | +2.22130 | +22.43950 |
| May, futures basis deviations, penalty 0.1 | 0.000040159984 | 0.000040164769 | +0.47855 | −7.59759 |

Positive changes are worse. The last column compares the new forecasts with
the training-only DC-direction group means from v210/v211. That simple control
still beats the feature model on average in March. May's improvement over the
clock model already existed without the new locations: this addition makes
the existing ridge forecast slightly worse. Shape-minus-ridge changes by
origin are `[+5.66256, +1.28567, −0.28434]` bp² for March and
`[−1.80551, +2.27181, +0.96935]` bp² for May. Paired UTC-day descriptive 95%
intervals are `[−0.26673, +4.58697]` and `[−8.30307, +9.98657]` bp². They
do not establish an improvement; overlapping paths and reused origins remain
dependent.

The controls reproduce v181/v178 H1 normalization, coefficients, complete
training signatures, forecasts and replay traces exactly. Cohorts are unchanged:
March has 1,951/1,806/1,924 training paths and 399/349/534 validation paths;
May has 4,237/3,916/3,490 and 407/411/1,064. Appending the locations flips the
predicted better holding side on 18/12/15 March observations and 45/51/50 May
observations, but **every economic replay field remains identical**. Only
reported model utility can differ. March retains the single +0.626577% H1
round trip; May retains its +1.919707% H1 round trip. Other origins stay cash.
No orders, fees, drawdowns or realized gains improve. The screens take about
6.99 and 13.30 seconds after loading.

v214/v215 then use the same two feature sets in separate ordinary and
return-weighted logistic sign heads, retaining the previous fixed penalty 0.1.
The horizon audit now accepts either an exact-second or candle-shape pair and
checks that all remaining settings match. It verifies the source cohort when
the source already requires three complete events. All predictions use inputs
at the first decision; targets are returns through events 1/2/3 and isolated
events 2/3. There is no policy selection or final-window outcome access.

| Suite | Mean first-event ordinary log-loss gain | Mean first-event return-weighted gain | Ordinary accuracy with locations, by origin |
| --- | ---: | ---: | --- |
| March | +0.00037402 | −0.00048211 | 62.9073%, 59.3123%, 61.2360% |
| May | −0.00014453 | +0.00003169 | 61.4251%, 62.5304%, 61.3722% |

Positive loss gains favor the addition. March's small ordinary-probability gain
has an interval spanning zero, while its move-size-weighted loss worsens on
average. May has no consistent first-event improvement. The ordinary sign
accuracy remains essentially unchanged; it should not be confused with a new
60% economic edge. No positive gain at a later horizon has a descriptive 95%
interval excluding zero. The sign audits take 2.02 and 3.48 seconds. This does
not show that close location is useless for every target; it rejects this
specific addition as justification for deeper event-policy training.

v216 verifies the generalized horizon-audit implementation against the earlier
March exact-second audit v177. All 60 saved head/prediction files, forecast
metrics and paired-day intervals reproduce exactly after renaming the generic
augmentation and cohort-check fields. Runtime is excluded from equality. This
is a regression check, not another feature or policy selection experiment.

Preserve the existing profitable H1 trades and the exact control contract.
Do not increase Bellman depth, widen feature searches, or backfill historical
flow/ETH solely on the strength of these results. A next change needs a
specific expected-value mechanism and a matched forecast improvement, rather
than another aggregate sign-accuracy result. No candidate is promoted and
there is no new final-window replay. The profitability objective remains open.

```powershell
node --conditions=development --import tsx scripts/screen-event-fitted-value.ts --source event-policy-quadrature-policy-march-v122 --settings-source event-policy-fitted-value-sampled-path-march-v181 --depths 1 --sampled-path 1 --candle-shape 0,1 --output event-policy-fitted-value-shape-march-v212
node --conditions=development --import tsx scripts/screen-event-fitted-value.ts --source event-policy-quadrature-policy-may-v121 --settings-source event-policy-fitted-value-sampled-path-may-v178 --depths 1 --sampled-path 1 --candle-shape 0,1 --output event-policy-fitted-value-shape-may-v213
node --conditions=development --import tsx scripts/audit-event-sign-horizons.ts --source event-policy-fitted-value-shape-march-v212 --output event-policy-sign-shape-march-v214
node --conditions=development --import tsx scripts/audit-event-sign-horizons.ts --source event-policy-fitted-value-shape-may-v213 --output event-policy-sign-shape-may-v215
node data/benchmarks/event-policy-fitted-value-shape-march-v212/check.cjs
```

### Direct holding paths and a constrained two-event option — v217–v224

The previous gates mainly assessed one-event holding forecasts or forecasts
under fitted future actions. A weak H1 forecast does not itself prove that
longer holding value cannot be predicted. The local
[`Position management`](../theory/Position%20management.md) discussion motivates
looking farther ahead when transaction costs make the immediate exit decision
ambiguous. The options framework of
[Sutton, Precup and Singh (1999)](https://people.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf)
provides a way to represent temporally extended actions alongside primitive
actions. Its policy-improvement results assume appropriate value knowledge;
they do not guarantee improvement from these fitted trading forecasts. The
local consecutive-return study also warns that persistent volatility need not
imply persistent direction. This motivates a direct forecast gate, not a large
option search or an assertion that a longer horizon must work.

**v217/v218: fixed-inventory forecast.** `eventPathHolding` marks the initial
inventory through each complete event, updating equity and exposure and charging
borrowing. At each one/two/three-event prefix it computes one hypothetical
terminal settlement. It does not rebalance to constant leverage, double-charge
exit costs, or choose future orders using realized returns. It records any
target-cap breach before continuing. These are counterfactual holding targets;
they are not automatically feasible multi-event options. In particular,
roughly 47–53% of full-short two-event validation paths breach the 1× target
cap after the first event; there are no such full-long breaches at 1×.

The direct models use the exact v181/v178 complete three-event cohorts,
features, clipped standardized coordinates and ridge penalties. Ridge
prediction weights fit six targets together: two inventory signs at three
horizons. H1 predictions match the original fitted heads within 1.8e−17, and
H1 MSE reproduces within numerical precision. Training and label availability
remain strictly before each origin. Controls are the training-only DC-state
holding means and a fixed zero-return, zero-borrow, terminal-fee forecast.
No future realized duration is used by that zero-return forecast.

| Suite | Horizon | Skill versus zero-return control | Skill versus clock-state mean | Post-first-event increment skill versus clock |
| --- | ---: | ---: | ---: | ---: |
| March | 1 | −0.4852% | −0.5567% | — |
| March | 2 | −0.5455% | −0.5037% | −0.3854% |
| March | 3 | −0.0065% | −0.1178% | −0.0609% |
| May | 1 | +0.0535% | +0.1752% | — |
| May | 2 | +0.3585% | +0.3451% | −0.0219% |
| May | 3 | +0.8014% | +0.8899% | +0.5761% |

Skills are equal-origin averages of relative MSE improvements, not return
percentages. March has no useful extension. May's three-event improvement is
positive in all origins, but still small. Screens take 2.38 and 4.84 seconds
after loading.

**v219/v220: paired errors and actionable forecast tails.** Resampling 2,000
UTC decision-day blocks within each reused origin gives the following
clock-minus-ridge MSE gains: March H2 −36.57 bp² with interval
`[−96.00, +27.87]`; March H3 −13.16 with `[−90.51, +82.39]`; May H2 +29.21
with `[−48.67, +105.48]`; May H3 +99.11 with `[−39.30, +238.80]`. These are
descriptive intervals for dependent paths, not independent evidence of
reliable profitability.

For a second diagnostic, pick the better full-long/full-short forecast only
when its holding value covers proportional entry costs. The entry log factor
is `−log(1 + maxLeverage * friction)`. These overlapping probes do not execute
an account and do not enforce lot/notional limits. Cap violations are retained
and counted rather than removed after seeing the path.

| May origin | Two-event chosen probes | Mean actual net log payoff, bp | Three-event chosen probes | Mean actual net log payoff, bp |
| --- | ---: | ---: | ---: | ---: |
| March 12 | 5 | +4.0647 | 11 | −25.3733 |
| April 2 | 2 | +55.7138 | 9 | +71.0732 |
| April 23 | 53 | +7.6831 | 142 | −5.0580 |

All March H2/H3 probes fail the entry-cost threshold. May H3's better overall
MSE does not make its selected opportunities profitable. H2 has positive mean
chosen-probe outcomes in each May origin, albeit with very few independent
opportunities. This justifies one inexpensive execution test of H2; it does
not justify scaling horizon, leverage or feature search.

**v222/v223: feasible two-event controller.** The core fitted-value trainer
now supports a `minimum-turnover` observed-path target. At the intermediate
event it uses the existing feasible-action optimizer with zero future utility:
hold if within the cap, otherwise choose the least-cost feasible reduction.
The account-dependent targets include that order's cost, marked equity and
exposure, subsequent borrowing and terminal settlement. All 195 account grid
cells are fitted at H2; H1, its normalization and its complete training
signature reproduce the reference exactly. External fitted continuation heads
cannot be combined with this mode, and checkpoint identity binds the mode.

The replay chooses an H2 action, then holds through two completed events with
mandatory intermediate cap reductions. It replans at expiry; a cash account
can replan at every event. At expiry, retaining inventory can avoid the
hypothetical closing/reopening fees of the terminal training convention.
The existing next-open fills, size/lot/notional limits, intrabar risk accounting,
borrowing, fees, order cancellations and final settlement remain in force.
This is a restricted option controller, not Bellman-optimal policy iteration.

| Suite/origin | H1 control return | Two-event option return | Option maximum drawdown | Orders |
| --- | ---: | ---: | ---: | ---: |
| May / March 12 | 0% | −0.626770% | 3.810960% | 11 |
| May / April 2 | 0% | +2.306344% | 0.832516% | 8 |
| May / April 23 | +1.919707% | +19.893711% | 6.918745% | 30 |
| March / January 14 | +0.626577% | 0% | 0% | 0 |
| March / February 4 | 0% | 0% | 0% | 0 |
| March / February 25 | 0% | 0% | 0% | 0 |

The May screen takes 20.52 seconds, March 10.19. H1 serialized replay traces
match the old controls exactly. Mandatory intermediate reduction signals are
2/0/2 across May's origins. The initial v221 run stopped before candidate
replay because an in-memory trace contained undefined optional properties
omitted by its JSON reference. Comparing the serialized trace contract fixes
that assertion; v221 is explicitly marked failed and v222 is its replacement.

**Trade attribution.** Complete cash-to-cash episodes reconcile both wealth
changes and log growth, and reconcile PnL minus fees and borrowing. The first
May origin has two winning shorts (+0.4837% and +0.6172%) but a March 27 short
loses −1.7118%. The active origin initially loses on April 25 (−0.2105%),
April 26 (−1.4730%), May 1 (−0.5058%), May 4 (−0.0411%), and a May 5 long
(−2.4554%). Its useful later behavior includes:

- May 8 short, 15:28–May 9 02:57: +3.4136%.
- May 9 short, 12:41–16:52: +3.9802%.
- May 11 rebound long, 12:53–14:12: +8.8147%, versus +4.3510% for the
  corresponding v178 sampled-policy episode ending at 13:46.
- May 11 overnight long, 21:29–May 12 00:41: +5.5508%.

These timings are UTC decision times. May's active-origin fees are $328.54
and borrowing $1.20 on $10,000 initial equity. Some profitable holds are
preserved or extended, but the new entries and increased drawdown offset the
gain over H1. Its prior score is 0.0621291, below v178's 0.0661233 and the
retained depth-eight candidate's 0.1102726. Slower replanning by itself does
not solve the entry problem.

**v224: final May comparison.** Append the one declared option candidate to
the 138 retained candidates, preserve their exact ranking, and write selection
before loading final outcomes. The incumbent remains selected. The new option
is refitted on 4,023 complete paths and independently replayed; its positive
prior score permits trading, but its forecasts choose cash on all 269 final
decisions. Both the incumbent and the new diagnostic return 0%. Final replay
takes 8.01 seconds. No final score is used to revise the candidate.

The useful next target is conditional entry value while preserving demonstrated
holding behavior. The May 11 rebound shows the cost of an early exit; the
March 27 short and May 5 long show the cost of entering on a weak longer-horizon
mean. A specific remaining approximation is cash opportunity value: the
minimum-turnover target holds cash forever within its two-event horizon, giving
it value zero, while the live cash state can reconsider entry at the next event.
Compare the value of waiting under the existing sampled continuation against
the new holding option at the wrong and preserved entries before changing the
controller. Whether that comparison corrects these trades is untested. The
final window remains flat because this candidate's forecasts do not support a
trade. Adding another horizon does not fix that.
All 82 focused tests and workspace typechecks pass. The full profitability and
optimality objective remains open.

```powershell
node --conditions=development --import tsx scripts/screen-event-hold-paths.ts --source event-policy-fitted-value-sampled-path-march-v181 --output event-policy-hold-path-march-v217
node --conditions=development --import tsx scripts/screen-event-hold-paths.ts --source event-policy-fitted-value-sampled-path-may-v178 --output event-policy-hold-path-may-v218
node --conditions=development --import tsx scripts/audit-event-hold-paths.ts --source event-policy-hold-path-march-v217 --output event-policy-hold-path-audit-march-v219
node --conditions=development --import tsx scripts/audit-event-hold-paths.ts --source event-policy-hold-path-may-v218 --output event-policy-hold-path-audit-may-v220
node --conditions=development --import tsx scripts/screen-event-hold-options.ts --source event-policy-fitted-value-sampled-path-may-v178 --output event-policy-hold-option-may-v222
node --conditions=development --import tsx scripts/screen-event-hold-options.ts --source event-policy-fitted-value-sampled-path-march-v181 --output event-policy-hold-option-march-v223
node --conditions=development --import tsx scripts/replay-event-hold-option.ts --source event-policy-hold-option-may-v222 --incumbent event-policy-fitted-value-crossfit20-may-policy-v204 --output event-policy-hold-option-may-policy-v224
node data/benchmarks/event-policy-hold-option-may-v222/trade-audit.cjs
```

### Waiting value and composition of saved controllers — v225–v235

The preceding option experiment leaves a concrete mismatch: its cash target
holds cash for the entire two-event horizon, while a cash account can decide
again at the next event. Test the existing sampled-policy value of waiting
before adding another gate, feature, or trained head. All runs in this section
use saved models; no extra training, data download, or wider parameter search
is involved.

**v225/v226: evaluate waiting on the original accounts.** At every original
H2 planning decision, reconstruct the option action exactly, then compare
the full interpolated option and sampled H2 holding values for every feasible
action. The two policies have identical H1 heads, normalization, complete
training signature, fees, leverage, order constraints and account grids.
V225 uses the full-data sampled continuation from v178; v226 uses the six-fold
held-out continuation targets from v193. Inputs are completed candles and
futures observations available at the original decision time.

Both audits inspect 403/406/943 planning accounts and 3/3/13 **executed**
flat entries across the three May origins. Both retain the side of every
first-origin and active-origin entry. The only entry replaced by cash is the
profitable April 6 long in the second origin. In v225, the sampled value of
waiting is zero at all first-origin and active-origin entries, including the
March 27 short, May 1 short and May 5 long. V226 gives the May 1 cash state
only +0.0738 bp of log value against the original entry's +3.2536 bp. It also
keeps the bad entry. Therefore this waiting-value gate does not solve the
observed entry failures. The audits take 1.20 and 1.02 seconds.

Counts use the executed `orderQuantity`. A nonzero planned `order.quantity`
can be canceled at the next open; comparing it with executed-entry counts
would falsely report new entries in the composition audit.

**Controller definition and prior work.**
[Barreto et al. (2017)](https://proceedings.neurips.cc/paper/2017/file/350db081a661525235354dd3e19b8c05-Paper.pdf)
describe generalized policy improvement by maximizing over multiple policy
action-value functions. Their guarantee assumes a uniform approximation-error
bound. No such bound has been established for these fitted trading values.
The present experiment also commits to a finite two-event controller instead
of applying their infinite-horizon greedy policy. It is an approximate
controller-composition experiment, not an application of a proven trading
improvement guarantee. The temporally extended controller follows the options
view of [Sutton, Precup and Singh (1999)](https://people.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf).

`decideFittedEventControllers` compares **complete interpolated** holding
functions on the same feasible post-order account. It does not take a maximum
at each grid cell before interpolation: that would splice different policies
into an artificial value function. The selected action maximizes order log
cost plus the larger holding value. Ties within the existing numerical
tolerance retain the first controller, the holding option.

At an option boundary the replay chooses an action and its controller. The
holding controller makes only mandatory cap reductions at the intermediate
event; the sampled controller acts with its shared H1 at that event. A sampled
cash decision also advances to H1, avoiding indefinite postponement through
repeated H2 replanning. After the second event the controller is selected
again. Holding-controller cash can reconsider at every event, matching v222.
Existing next-open execution, canceled orders, borrowing, liquidation checks,
lot/notional constraints and terminal settlement remain in effect. For the
held-out version, the H2 targets evaluate block-held-out H1 fits; execution
uses the saved full-fit H1, as in the previous held-out experiment.

**v227–v229: actual composed replays.** Both original hold and rolling
sampled-H2 traces reproduce exactly before running the composition.

| Prior suite / method | Origin 1 return | Origin 2 return | Origin 3 return | Worst drawdown | Prior score |
| --- | ---: | ---: | ---: | ---: | ---: |
| May hold option v222 | −0.626770% | +2.306344% | +19.893711% | 6.918745% | 0.0621291 |
| May rolling sampled v178 | 0% | +1.066571% | +21.424165% | 4.748533% | 0.0661233 |
| May composition v227 | −0.271834% | +0.168781% | +23.956387% | 4.982136% | 0.0687114 |
| May held-out composition v228 | −1.142429% | +1.636827% | +22.535363% | 5.114143% | 0.0662399 |
| March hold option v223 | 0% | 0% | 0% | 0% | 0 |
| March held-out composition v229 | +0.626577% | 0% | 0% | 1.124729% | 0.0017072 |

May composition uses 7/2/26 orders and pays $48.03/$4.82/$274.74 in fees
from $10,000 initial equity in each origin. Its improvement is concentrated
in the active origin. The held-out composition uses 10/5/24 orders and pays
$62.73/$38.72/$278.87. March reproduces the earlier held-out sampled
controller's economic result exactly; it does not find another profitable
origin. Initial screen times are 2.36/2.35/1.62 seconds of measured work.

**v230–v232: isolate advancement from controller choice.** A sampled policy
which alternates H2 then H1, including in cash, is a necessary control. Simply
advancing that policy gives May returns 0%, +0.596447%, +19.316707% and
score 0.0589607. Advancing its held-out version gives 0%, 0%, +5.379621%
and score 0.0159427. Both are weaker than their corresponding rolling H2
controls. The March control remains +0.626577%, 0%, 0%. Thus advancement
alone does not explain the composed May improvement. These runs also require
the v227–v229 composed traces to reproduce exactly after adding this control.
Measured screen times, including all controls, are 3.55/3.47/2.28 seconds.

**v233/v234: entry and episode attribution.** The audit reconstructs complete
cash-to-cash episodes, including any reversals inside them. Total PnL reconciles
to final wealth and to long PnL plus short PnL minus fees and borrowing; summed
episode log growth and reversals also reconcile to each replay.

The full-data composition keeps the bad March 27 entry but exits at 22:47 UTC
for −0.7519%, instead of holding until March 28 00:45 for −1.7118%.
It also skips the profitable March 19 short and April 21 long while advancing
the sampled cash controller to H1. The held-out version retains those entries,
but its March 27 short loses more, −2.2218%. In the active origin:

- Both compositions retain the May 4 short until May 5 14:33, earning about
  +3.0607% instead of the option's −0.0411% with its 05:16 exit.
- The full-data composition retains the May 1 losing entry (−0.5058%) and
  increases the May 5 long loss to −3.0709%. The held-out version skips May 1
  during a committed H1 step but retains the May 5 long loss (−2.4554%).
- Both preserve the profitable May 8 short. The May 9 short improves from
  +3.9802% to +5.0162% / +4.6668% as exposure lasts until 17:05.
- The May 11 rebound cash-to-cash episodes return +8.5143% / +8.2671%,
  compared with the hold option's +8.8147%. Both new episodes include a
  long-to-short reversal at 13:50; these returns must not be described as
  pure long-holding profits.
- The May 11 overnight long remains useful: +4.9030% / +5.5509%.
  The full-data composition also joins the May 12 morning opportunities into
  one 04:35–13:33 long, +2.5245%.

The held-out version skips the April 25, May 1, May 10 and first May 11 rebound
entries at their original timestamps while advancing H1; it enters the rebound
three minutes later. This does **not** contradict the v226 finding that H2
value comparisons kept those entry sides. Policy state and cycle timing have
changed. Skipping a losing trade is not by itself evidence of a better sign
forecast or of correctly rejecting its value.

**v235: frozen final May comparison without retraining.** Append both composed
variants and both fixed sampled controls to the 139 retained candidates,
preserving every old rank and score. Write the 143-candidate selection before
loading final candles. The incumbent remains depth eight with score 0.1102726,
above the best new score of 0.0687114. Load the compatible saved final holding,
sampled and held-out sampled models; verify identical H1 heads and training
contracts, and reproduce their original final traces exactly. All four new
diagnostics are enabled by positive prior scores but choose no orders in the
final May window: return, fees and drawdown are all zero. The incumbent's
selected final result remains zero. Measured final replay time is 0.81 seconds.

The useful behavior to retain is patient holding through the May 4/5 decline,
May 9 decline and May 11 rebounds. The unresolved problem is conditional
entry and exit value, including the loss of good trades through option timing.
The inexpensive next diagnosis is paired held-out error in the **difference**
between the two controllers' values at the same feasible account and action.
Check whether the predicted winner actually improves two-event payoff before
learning a selector, extending horizons, or interpreting extra sign accuracy
as economic skill. The present maximum can amplify approximation error and
does not justify scaling computation or claiming reliable profitability.

All 84 focused tests and workspace typechecks pass. New checks cover policy
maximization after interpolation, deterministic ties, incompatible account
contracts, and advancing a sampled cash option to its next H1, both alone and
inside the composition. Script executions verify exact controls and reconciled
economic results; no extra model training was performed.

```powershell
node --conditions=development --import tsx scripts/audit-event-hold-opportunity.ts --source event-policy-hold-option-may-v222 --output event-policy-hold-opportunity-may-v225
node --conditions=development --import tsx scripts/audit-event-hold-opportunity.ts --source event-policy-hold-option-may-v222 --sampled-source event-policy-fitted-value-crossfit-may-v193 --output event-policy-hold-opportunity-crossfit-may-v226
node --conditions=development --import tsx scripts/screen-event-controller-composition.ts --source event-policy-hold-option-may-v222 --reference event-policy-controller-composition-may-v227 --output event-policy-controller-fixed-control-may-v230
node --conditions=development --import tsx scripts/screen-event-controller-composition.ts --source event-policy-hold-option-may-v222 --sampled-source event-policy-fitted-value-crossfit-may-v193 --reference event-policy-controller-composition-crossfit-may-v228 --output event-policy-controller-fixed-control-crossfit-may-v231
node --conditions=development --import tsx scripts/screen-event-controller-composition.ts --source event-policy-hold-option-march-v223 --sampled-source event-policy-fitted-value-crossfit-march-v192 --reference event-policy-controller-composition-crossfit-march-v229 --output event-policy-controller-fixed-control-crossfit-march-v232
node --conditions=development --import tsx scripts/audit-event-controller-composition.ts --source event-policy-controller-fixed-control-may-v230 --output event-policy-controller-composition-audit-may-v233
node --conditions=development --import tsx scripts/audit-event-controller-composition.ts --source event-policy-controller-fixed-control-crossfit-may-v231 --output event-policy-controller-composition-audit-crossfit-may-v234
node --conditions=development --import tsx scripts/replay-event-controller-composition.ts --sources event-policy-controller-fixed-control-may-v230,event-policy-controller-fixed-control-crossfit-may-v231 --sampled-finals event-policy-fitted-value-sampled-path-policy-v179,event-policy-fitted-value-crossfit-may-policy-v195 --hold-final event-policy-hold-option-may-policy-v224 --incumbent event-policy-hold-option-may-policy-v224 --output event-policy-controller-composition-may-policy-v235
```

### Paired controller values and cash decision timing — v236–v244

The next test follows v235's diagnosis: determine whether the predicted
advantage of the sampled controller over the holding controller is accurate
on the same state, account and action. Do this before fitting another selector
or increasing rollout depth. The paired comparison needs no model training.

**v236–v238: observed two-event controller payoffs.**
`eventControllerHolding` starts from a post-order account, marks event one,
then chooses either the minimum-turnover action or the saved H1 action using
the newly observed state. Only after that choice does it apply event two.
It includes the intermediate order cost, marked exposure, borrowing and one
terminal settlement. A test checks the analytical wealth factors, agreement
with uninterrupted holding when no reduction is needed, unchanged intermediate
orders when the second return changes, and sticky ruin. Settlement and
intermediate fills follow the fitted-target convention, not the minute-by-minute
next-open execution convention; this is a value diagnostic, not a portfolio
backtest. Floating-point residual exposure at an exact close is checked with
a numerical tolerance.

Use the 408/412/1,065 complete two-event paths from May's three prior origins
and 400/350/535 from March. Every path ends before its origin boundary.
For each, evaluate full short, cash and full long from $10,000 equity at the
current price. Also reconstruct the composed controller's actual H2 planning
actions on their original accounts, including its chosen controller. The
full-data May / held-out May / held-out March audits reproduce
276/279/932, 288/289/679, and 255/245/535 planning actions respectively.
Intermediate H1 actions are deliberately outside that selected-H2 cohort.
All saved model hashes still match their source screens.

The target is `actual(sampled) − actual(hold)`; the forecast is the same
difference between their complete interpolated H2 values. All evaluated
forecasts and payoffs are finite; ruin probes would fail explicitly instead
of disappearing from the sample. Full-data and held-out models share the
deployed H1, so their **fixed-account actual payoff differences are identical**.
Their H2 forecasts and resulting selected accounts differ.

| Audit / fixed exposure | MSE improvement over predicting zero, bp² | Exploratory 95% interval |
| --- | ---: | ---: |
| May full-data / short | −2.4108 | [−25.9619, +20.7841] |
| May full-data / long | −2.2242 | [−16.5953, +15.2924] |
| May held-out / short | −9.6647 | [−26.1434, +6.6602] |
| May held-out / long | +3.5189 | [−10.3345, +19.8960] |
| March held-out / short | −3.9022 | [−15.2545, +6.7278] |
| March held-out / long | −12.0908 | [−23.5409, +0.2886] |

Positive improvements favor the fitted difference forecast. These intervals
use 2,000 circular two-day block resamples within each origin and equal weight
per origin. They are descriptive checks on reused, overlapping research paths,
not a new independent significance test. None establishes an advantage-MSE
improvement. The full-data May short selector adds +1.6702 bp of mean realized
holding value over always holding, but only +0.5115 bp over always using the
sampled controller, whose interval [−0.5148, +1.4672] crosses zero. March's
short selector loses −1.9159 bp against always sampled, interval
[−3.5124, −0.3467]. These are post-order holding-value differences, not returns
of a strategy entering full exposure on every overlapping probe.

The selected-account results are weaker than the favorable May portfolio
headline. In the active May origin, full-data selection has 137 exposed H2
planning probes, with only 17 differing intermediate orders. Its advantage
MSE is 6.36% worse than zero, and choosing the predicted controller loses
1.7202 bp per probe against always holding. The held-out version has 120
probes and 14 differing orders; its MSE is 0.0369% worse than zero and its
selection loses 0.0232 bp against holding. Sparse selected-account results
remain descriptive. Different termination conventions and subsequent option
cycles also prevent these two-event figures from being equated with the full
portfolio return.

**Waiting-value diagnosis.** In May's first two origins, both deployed H1
controllers leave cash untouched at every paired continuation. The actual
cash payoff difference is exactly zero, while the full-data model predicts
positive differences on 48.28%/47.82% of probes and the held-out model on
40.69%/47.09%. Those small values alter option timing without producing the
predicted subsequent trade in these samples. In the active origin, only three
cash continuation decisions trade. The full-data H2 predicts zero cash
advantage throughout; the held-out forecast selects a mixture that has a
negative realized gain. This supports testing cash cadence, not training a
more elaborate controller selector from these noisy advantage estimates.

The result is consistent with the selection problem discussed in
[Smith and Winkler (2006)](https://pubsonline.informs.org/doi/10.1287/mnsc.1050.0451):
maximizing noisy value estimates can make the selected estimate optimistic
even when individual estimators are unbiased. That paper's Bayesian adjustment
would require an error model estimated from prior data; the present audit does
not identify a validated correction. Likewise,
[Munos and Szepesvári (2008)](https://www.jmlr.org/papers/volume9/munos08a/munos08a.pdf)
analyze fitted value iteration under explicit MDP, sampling and approximation
conditions. Their bounds do not establish convergence for these reused market
paths. These sources support checking value error rather than assuming that
another maximization or recursion will improve the strategy.

**v239–v241: cash timing ablation.** Add explicit `replanCash` behavior to
the existing option replay. A flat account now replans H2 at every event,
including after a sampled cash choice. Invested options still use the same
two-event commitment and intermediate controller. No model, feature, action
value, fee, leverage or order-size parameter changes. The original committed
compositions and all other controls reproduce exactly before the ablation.
This deliberately introduces a target/execution mismatch at cash states:
the saved H2 target assumed H1 next, while cash now reconsiders H2. Therefore
it is a measured timing experiment, not exact evaluation of the changed policy.

| Prior suite / sampled model | Origin 1 return | Origin 2 return | Origin 3 return | Worst drawdown | Prior score |
| --- | ---: | ---: | ---: | ---: | ---: |
| May full-data, replan cash v239 | +0.343714% | +0.596447% | +23.956387% | 4.982136% | 0.0718622 |
| May held-out, replan cash v240 | −1.142429% | +1.636827% | +21.634305% | 5.114199% | 0.0637797 |
| March held-out, replan cash v241 | +0.626577% | 0% | 0% | 1.124729% | 0.0017072 |

The full-data variant now has positive net return **and** positive
drawdown-penalized score in all three May origins. It uses 9/4/26 orders and
pays $62.55/$28.92/$274.74 in fees. Its active-origin trace and economics are
unchanged. The held-out variant keeps the first two origins' trading results unchanged but
adds six orders in the active origin, raising its fees from $278.87 to
$334.37 and reducing its return by 0.9011 percentage points. March's economic
result remains unchanged. Measured screen times are 4.32/4.20/3.04 seconds,
including controls; paired audits took 1.70/1.59/1.23 seconds.

**v242: final May replay.** Preserve all 143 incumbent candidates and append
the two declared cash-timing candidates; do not duplicate their unchanged
fixed sampled controls. Write selection before loading final candles. The
incumbent remains selected with score 0.1102726, above 0.0718622 and
0.0637797. Reuse the saved final models and reproduce the original hold and
sampled final traces exactly. Both new candidates have positive prior scores
but choose zero orders in all 269 final decisions, returning 0% with no fees
or drawdown. The selected incumbent result also remains 0%. No retraining;
measured final replay time is 0.64 seconds.

**v243/v244: changed trades, reconciled.** Compare the cash-timing variants
with their committed compositions, not with the original hold-only strategy.
The full-data variant restores the March 19 short, +0.6172%, and April 21
long, +0.4269%. It keeps the useful March 16 short and the shorter exit from
the bad March 27 short, whose loss remains approximately −0.7519%. The
active origin keeps every entry and its complete cash-to-cash episodes.
The held-out variant restores both the losing April 25 long (−0.2105%) and
May 1 short (−0.8489%), as well as the profitable May 10 short (+0.4513%).
It enters the May 11 rebound at 12:53 rather than 12:56; the episode still
reverses short at 13:50 and returns +8.1319%. PnL, log wealth, fees, borrowing
and reversal counts reconcile in all audited origins.

This isolates a real cost of cash commitment, but does not repair the bad
entry forecasts or justify promoting the timing change as a general rule.
The full-data variant preserves useful behavior and improves the prior score;
the held-out variant demonstrates the counterexample. Keep both explicit
research results. All 85 focused tests and workspace typechecks pass.

The next substantive model change should evaluate the **deployed** controller,
including its option phase and cash cadence, before its next Bellman
improvement. The current H2 labels evaluate a one-step H1 continuation, while
deployment can retain positions over repeated option cycles. First reproduce
short observed rollouts of that frozen controller on held-out paths and
compare their selected-action errors. Only a useful result would justify
fitting the next value head or expanding computation. The full non-fit
inspector profitability objective is still open.

```powershell
node --conditions=development --import tsx scripts/audit-event-controller-values.ts --source event-policy-controller-fixed-control-may-v230 --output event-policy-controller-value-may-v236
node --conditions=development --import tsx scripts/audit-event-controller-values.ts --source event-policy-controller-fixed-control-crossfit-may-v231 --output event-policy-controller-value-crossfit-may-v237
node --conditions=development --import tsx scripts/audit-event-controller-values.ts --source event-policy-controller-fixed-control-crossfit-march-v232 --output event-policy-controller-value-crossfit-march-v238
node --conditions=development --import tsx scripts/screen-event-controller-composition.ts --source event-policy-hold-option-may-v222 --reference event-policy-controller-fixed-control-may-v230 --replan-cash --output event-policy-controller-cash-replan-may-v239
node --conditions=development --import tsx scripts/screen-event-controller-composition.ts --source event-policy-hold-option-may-v222 --sampled-source event-policy-fitted-value-crossfit-may-v193 --reference event-policy-controller-fixed-control-crossfit-may-v231 --replan-cash --output event-policy-controller-cash-replan-crossfit-may-v240
node --conditions=development --import tsx scripts/screen-event-controller-composition.ts --source event-policy-hold-option-march-v223 --sampled-source event-policy-fitted-value-crossfit-march-v192 --reference event-policy-controller-fixed-control-crossfit-march-v232 --replan-cash --output event-policy-controller-cash-replan-crossfit-march-v241
node --conditions=development --import tsx scripts/replay-event-controller-composition.ts --sources event-policy-controller-cash-replan-may-v239,event-policy-controller-cash-replan-crossfit-may-v240 --sampled-finals event-policy-fitted-value-sampled-path-policy-v179,event-policy-fitted-value-crossfit-may-policy-v195 --hold-final event-policy-hold-option-may-policy-v224 --incumbent event-policy-controller-composition-may-policy-v235 --output event-policy-controller-cash-replan-may-policy-v242
node --conditions=development --import tsx scripts/audit-event-controller-composition.ts --source event-policy-controller-cash-replan-may-v239 --output event-policy-controller-cash-replan-audit-may-v243
node --conditions=development --import tsx scripts/audit-event-controller-composition.ts --source event-policy-controller-cash-replan-crossfit-may-v240 --output event-policy-controller-cash-replan-audit-crossfit-may-v244
```

### Exact deployed rollouts and quote-entry execution — v245–v255

**v245–v247: preserve the deployed account and option phase.** The replay can
resume from observed equity, base quantity, remaining option events and chosen
controller. Its terminal settlement arithmetic is shared with the rollout
audit. For every observed state with four complete subsequent events inside
its prior origin, run the same frozen controller and require the complete
local trace to equal the original suffix. Compute hypothetical terminal
settlements at each prefix without inserting those settlements into later
prefixes. Independently replay H2 at the first state and every executed entry.
Full original metrics and traces also reproduce exactly.

The three audits cover 5,037 overlapping four-event rollouts, 20,148 reproduced
trace events, and 44 independent H2 checks across nine prior origins. They take
8.62/9.01/6.15 seconds. These are conditional evaluations at accounts visited by
the existing policy, not an off-policy action grid, a new forecast, or new
independent evidence of profitability. H2 forecasts are compared with H2
payoffs; H4 minus H2 is reported separately.

For executed entries in the active May origin, the old observed-path target
and exact H2 replay differ by only 0.0197 bp on average in the full-data model,
and 0.3593 bp in the held-out model. Canceled flat entries are the exception:
their mean absolute target discrepancies are 131.49/123.22 bp. On May 11,
20:57/21:00/21:16/21:28 UTC, next-open gaps of 16.6761/4.8445/2.7845/7.3971 bp
push the precomputed full-size quantity over the post-fee leverage cap. The
held-out replay also cancels May 12, 04:35, after a 0.1916-bp gap. This is an
execution failure to take the planned action, not a missing directional signal.
Four of those five gaps would still violate the cap after subtracting one lot.

**v248–v250: freeze quote turnover instead of base quantity on flat entries.**
Binance's [Margin new-order endpoint](https://developers.binance.com/en/docs/catalog/core-trading-margin-trading/api/rest-api/trade)
accepts `quoteOrderQty`. Its [Spot market-order specification](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/trade)
describes quote spending for BUY and quote proceeds for SELL, with executed
base quantity determined from market liquidity and lot rules. These support
the research execution contract; they do not validate its liquidity model.

At the completed decision close, retain the selected side and freeze its
planned quote turnover. At the next open, divide that committed amount by the
fill price and round down to the base lot step. Apply the existing fees,
minimum/maximum order bounds and leverage cap. Orders from an invested account
retain their fixed base quantity. The opening gap is never earned by a flat
account. Tests vary future opening prices and verify identical committed
decisions and quote amounts, correct long/short PnL, fee reconciliation,
minimum-lot rejection, and unchanged invested-account execution.

This is an explicit single-price approximation to quote-sized market orders,
not an exact Binance fill simulator: commission assets, borrow inventory,
liquidity and all symbol-specific filters are not modeled. The existing
fixed-quantity mode remains the declared comparison. Each reference's exact
trace and metrics reproduce before the new replay. No forecast, target value,
fee, leverage, or decision cadence is refitted or tuned for this experiment.

| Prior suite / sampled model | Origin 1 return | Origin 2 return | Origin 3 return | Worst drawdown | Prior score |
| --- | ---: | ---: | ---: | ---: | ---: |
| May full-data, quote entry v248 | +0.343714% | +0.596447% | +23.908945% | 4.982103% | 0.0717346 |
| May held-out, quote entry v249 | −1.142429% | +1.636827% | +24.638884% | 5.114205% | 0.0719135 |
| March held-out, quote entry v250 | +0.626577% | 0% | 0% | 1.124729% | 0.0017072 |

The first two May origins and all March economic results are unchanged.
Active May cancellations fall from 4/5 to zero. Full-data order count stays
26 and fees change from $274.74 to $274.72; held-out orders fall from 30 to
25 and fees from $334.37 to $275.80. Screen times are 4.17/4.17/2.87 seconds.

**v251/v252: cash-cycle attribution.** Both variants now enter the May 11
overnight long at 20:57 rather than 21:29. Earlier entry alone is slightly
worse: the full-data cycle falls from +4.9030% to +4.8630%; the held-out cycle
falls from +5.5508% to +5.5107%. The held-out gain comes on May 12. Filling
04:35 rather than 04:36 changes the option phase, retaining the long until
13:33 for +2.5244%. The original sequence exits at 07:45, re-enters at 07:49,
exits at 10:31, then trades again from 10:57 to 13:33; those three cycles
return +0.8743%, +0.1123%, and −0.9633%. This is a measured interaction between
execution and controller memory, not evidence that every earlier entry is
better. Existing beneficial May shorts and the midday May 11 reversal remain;
the earlier losing April/May entries also remain. PnL, log wealth, fees,
borrowing and reversals reconcile in all six audited origins.

**v253: final May replay.** Append only the two new execution candidates to
the 145 retained candidates and write all 147 ranks before reading final
candles. Both new prior scores remain below the incumbent's 0.1102726. Reuse
the saved models and exactly reproduce original hold and sampled controls.
Both new variants make zero orders across all 269 final decisions. The
selected incumbent's exact final trace and 0% return remain unchanged. The
final replay takes 0.64 seconds; no training.

**v254/v255: verify the new execution in the exact rollout.** Another 3,758
four-event rollouts reproduce 15,032 trace events and 38 independent H2 checks.
In active May, the mean absolute exact-H2 versus old-target discrepancy on
exposed planning states falls to 0.2523/0.2479 bp. These are changed visited
cohorts, so the comparison does not estimate a paired forecast improvement.
Executed-entry predicted/realized H2 means remain +3.36/−3.18 bp for full-data
and +3.41/−6.33 bp for held-out. Actual H4 means are +18.36/+15.22 bp. Closing
later can help, but those conditional overlapping observations do not justify
choosing the horizon from realized returns. Rollout times are 9.12/8.79 seconds.

The shared simulator is now a useful exact policy-evaluation primitive at
observed accounts. It is not an optimal-control certificate: it only evaluates
the supplied actions. Under the revised objective, hold the forecast fixed,
check action search and continuation approximation separately against a small
exhaustive Bellman reference, and quantify any remaining policy regret before
another forecast experiment. Fitted values for different controllers are not
automatically the Bellman optimum of a single coherent transition model.

```powershell
node --conditions=development --import tsx scripts/audit-event-policy-rollouts.ts --source event-policy-controller-cash-replan-may-v239 --output event-policy-deployed-rollout-may-v245
node --conditions=development --import tsx scripts/audit-event-policy-rollouts.ts --source event-policy-controller-cash-replan-crossfit-may-v240 --output event-policy-deployed-rollout-crossfit-may-v246
node --conditions=development --import tsx scripts/audit-event-policy-rollouts.ts --source event-policy-controller-cash-replan-crossfit-march-v241 --output event-policy-deployed-rollout-crossfit-march-v247
node --conditions=development --import tsx scripts/screen-event-controller-composition.ts --source event-policy-hold-option-may-v222 --reference event-policy-controller-cash-replan-may-v239 --replan-cash --quote-entries --output event-policy-controller-quote-entry-may-v248
node --conditions=development --import tsx scripts/screen-event-controller-composition.ts --source event-policy-hold-option-may-v222 --sampled-source event-policy-fitted-value-crossfit-may-v193 --reference event-policy-controller-cash-replan-crossfit-may-v240 --replan-cash --quote-entries --output event-policy-controller-quote-entry-crossfit-may-v249
node --conditions=development --import tsx scripts/screen-event-controller-composition.ts --source event-policy-hold-option-march-v223 --sampled-source event-policy-fitted-value-crossfit-march-v192 --reference event-policy-controller-cash-replan-crossfit-march-v241 --replan-cash --quote-entries --output event-policy-controller-quote-entry-crossfit-march-v250
node --conditions=development --import tsx scripts/audit-event-controller-composition.ts --source event-policy-controller-quote-entry-may-v248 --output event-policy-controller-quote-entry-audit-may-v251
node --conditions=development --import tsx scripts/audit-event-controller-composition.ts --source event-policy-controller-quote-entry-crossfit-may-v249 --output event-policy-controller-quote-entry-audit-crossfit-may-v252
node --conditions=development --import tsx scripts/replay-event-controller-composition.ts --sources event-policy-controller-quote-entry-may-v248,event-policy-controller-quote-entry-crossfit-may-v249 --sampled-finals event-policy-fitted-value-sampled-path-policy-v179,event-policy-fitted-value-crossfit-may-policy-v195 --hold-final event-policy-hold-option-may-policy-v224 --incumbent event-policy-controller-cash-replan-may-policy-v242 --output event-policy-controller-quote-entry-may-policy-v253
node --conditions=development --import tsx scripts/audit-event-policy-rollouts.ts --source event-policy-controller-quote-entry-may-v248 --output event-policy-deployed-rollout-quote-may-v254
node --conditions=development --import tsx scripts/audit-event-policy-rollouts.ts --source event-policy-controller-quote-entry-crossfit-may-v249 --output event-policy-deployed-rollout-quote-crossfit-may-v255
```

### Fixed-forecast optimality reference — v256/v257

The revised objective requires a model-conditioned optimum before another
forecast experiment. The existing historical-price exposure oracle is not
appropriate for that comparison: it sees the realized future and omits fixed
order minima/lots. Add an independent stochastic reference that instead uses
the frozen event kernel and enumerates **every feasible order lot** at every
node. It tracks actual equity, price and exposure without interpolation, pays
fees/borrowing, and treats every positive-probability liquidation as log ruin.
It does not call production trading, holding or Bellman helpers. Search is
bounded to 100,000 nodes and 1,000 order lots per side.

The recurrence is `V_h = max_order(log(E_after_fee/E_before) +
sum_outcomes p * (log(holding_factor) + V_(h-1)(successor_account)))`.
The order is fixed before averaging future outcomes. At zero remaining events,
settle the account. The primary comparison uses the planner's existing
proportional terminal friction on every remaining unit. A separate result
uses replay-style terminal dust when the residual order is below the minimum.
Since the event law has no opening-gap variable, both reference and planner
fill at the event decision price in this audit. This is explicitly a different
contract from the next-open market backtest.

Three comparisons separate errors: exhaustive lot optimization; production
candidate search supplied with exact continuation; and the existing
interpolated policy, evaluated on exact future accounts under that same law.
The last evaluation advances its finite horizon instead of resetting it.
This measures decision regret independently of a value estimate's numerical
error. It also prevents comparing a fitted composition against a different
forecast and calling the difference optimizer regret.

**v256, 27 queries.** The first frozen law gives +10% with probability 0.6 and
−10% with probability 0.4, with $100 equity, price $10, 5× maximum leverage,
12 bp friction per trade, 0.1-unit lots, $1 minimum and $500 maximum notional.
The optimal cash entry is 17.6 units with expected log growth 0.0156175023.
The grid chooses 19.9 units, whose exact value is 0.0153479605: **2.6954 bp
regret**. Starting from existing long/short exposure produces maximum regret
2.7531 bp. Supplying exact continuation to the same candidate set does not
fix this: the missing intermediate order size is the cause.

A two-state reversal law, evaluated through three events, has up to 0.8153 bp
decision regret and 11.7612 bp absolute value-estimate error. In the affected
state, the optimal order is 0.3 units and the grid chooses 0.4. The dust fixture
has no action regret, but charging an untradeable terminal residual changes
its value by about 2.4 bp. None of these tiny fixtures finds an additional
root-action error from interpolation after controlling for candidate choice;
that is not a general bound on interpolation error.

**v257, frozen-law grid refinement.** Increasing `actionSteps` from 5 to the
default 10 leaves the binary decision regret unchanged. At 20 steps, the
cash order becomes 17.4 units and regret falls to 0.0266 bp; the maximum over
the three initial exposures falls from 2.7531 to 0.0309 bp. Both the target
grid and exposure interpolation grid are refined, but exact candidate
evaluation isolates the improvement to the available order sizes in this
fixture. The exact reference still prefers 17.6. Do not infer exact optimality
from grid stability or automatically enlarge every real-model grid.

These screens take 0.044/0.086 seconds of measured work. Focused tests compare
the reference with the analytical zero-fee binary Kelly solution, require
cash on a fair two-event law (no future-sign oracle), retain rare liquidation
risk, reject exceeded work budgets, and distinguish dust from proportional
settlement. All 89 focused tests and workspace typechecks pass.

Next, remove the demonstrated order-size search gap with a bounded optimizer,
using the independent lot enumeration as its check, then measure the effect
with a frozen learned forecast on the non-fit inspector windows. Also align
the chosen terminal and fill conventions before claiming a model-conditioned
Bellman optimum. No forecast was retrained or promoted in this stage, and
the synthetic reference does not certify the existing full strategy.

```powershell
node --conditions=development --import tsx scripts/audit-event-bellman-optimality.ts --output event-policy-fixed-model-optimality-v256
node --conditions=development --import tsx scripts/audit-event-bellman-optimality.ts --output event-policy-fixed-model-grid-refinement-v257
```

### Exact one-event order optimization and terminal objectives — v258–v266

**Implementation.** `event-one-step.ts` optimizes the integer order-lot count,
including no order, both signs, partial changes and reversals. It preserves
fees, leverage, minimum/maximum notional, quantity minimum/step, borrowing,
maintenance liquidation and above-cap recovery. It shares only the production
holding/account definitions and signal translation; verification uses the
independently written exhaustive Bellman reference.

Split the order domain at every transaction-sign, position-sign, borrowing,
feasibility and terminal-dust boundary. Evaluate boundary-adjacent lots
directly. Inside each remaining interval, outcome-specific terminal wealth
has the form `a_j + b_j*k`, where `k` is the integer order count. All
positive-probability outcomes must survive. The expected value is a weighted
sum of logarithms of these affine wealth terms, so its discrete derivative
`F(k+1)-F(k)` is non-increasing. Binary search locates the maximizing lot;
compare all intervals and boundaries. This avoids assuming that an arbitrary
fitted critic, or a multi-event value with integer order constraints, is
concave. The solver rejects a lattice beyond safe integer precision.

The explicit terminal conventions are:

- `marked`: value wealth including remaining inventory; pay the current
  transaction and holding costs, with no assumed transaction at the next event.
- `friction`: charge proportional terminal liquidation on all inventory,
  matching the existing planner's mathematical boundary.
- `market`: charge terminal liquidation only when the residual order meets
  the minimum quantity and notional; otherwise mark the remaining dust.

**v258/v259: independent stress check.** The first run found a numerical
settlement inconsistency, not a different chosen order. Recovering quantity
from post-event exposure could put an exact minimum lot one floating-point
unit below the minimum. The reference then waived fees that the optimizer
charged. Apply the same quantity/notional tolerances already used by order
validation to terminal settlement in the reference, optimizer and simulator.
The failed run and its inputs remain in v258. V259 then matches all 3,636
queries for the two liquidation conventions. V263 adds marked wealth and
matches all **5,454** queries, with zero measured chosen-action regret against
enumeration and a value tolerance of 1e-10.

The seeded cases vary fees, leverage, borrowing, event duration, return and
intrabar extrema, minimum/maximum order size, lot size, and initial cash,
long, short and above-cap inventory. Separate focused tests check the known
fee-aware binary optimum on a five-billion-lot lattice with fewer than 200
direct order evaluations, rare ruin mass as small as 1e-300, exact minimum
lots, and a case profitable after one fee but not after forced round-trip
costs. The original 19.9-versus-17.6 sizing error is removed.

For the deliberately small randomized lattices, search overhead outweighs
the reduced order count: v263 takes 89.96 ms in the new solver versus
55.17 ms for enumeration, while evaluating 128,582 versus 421,212 orders.
The benefit is avoiding a scan of large exchange lot grids, not claiming
that binary search beats enumeration on every tiny problem.

**v260–v262: frozen learned forecasts.** Reuse the existing May/March/January
joint laws from v121/v122/v120 with all probabilities, costs and training
cutoffs unchanged. Replay depth one at the three preceding origins and the
final non-fit inspector window for each. The saved grid economics reproduce
where a depth-one result exists. Compare 4,693 grid-visited accounts and 18
additional initial long/short accounts on exactly the same inputs. Nine
independent checks enumerate 3,150,069 orders on the actual saved kernels;
all agree with the fast solver. Those checks take about 1.2–1.7 seconds each.
Complete screen times, including them, are 11.57/8.49/7.68 seconds.

With the old proportional terminal friction, only three visited decisions
change: the March suite's February 7/17/21 entries increase exposure from
approximately 0.2000 to 0.2225. Matching-account expected utility gains are
tiny, at most 0.000126 bp per entry, but realized origin return changes from
+0.184529% to +0.205320%. Both policies use six orders; drawdown changes from
0.168527% to 0.187475%. The other active March origin remains −0.100796%,
and every May/January origin and all three final windows remain cash. This
closes a real sizing gap without mistaking a realized gain for evidence of
forecast improvement.

The market-dust boundary mainly generates minimum-size trades. For example,
May's first origin enters approximately $5 positions with predicted utility
only 0.0003–0.0004 bp, then later pays closing costs. It loses 0.000190%; the
active May origin loses 0.005318% and its final window loses 0.000247% while
retaining $0.2920 dust. These small incentives arise from the terminal
valuation rule; they are not a useful new signal. The implementation and
accounting are retained as explicit contracts rather than suppressing those
actions with a threshold fitted to the observed losses.

**Why also test marked wealth?** The user's next-step log-wealth objective
does not itself require an exit at every forecast endpoint. The repo's
`handcrafted-indicator-parameter-prediction.md`, sections 3–4, describes
holding utility and a zero continuation boundary; `theory/Position
management.md` separately requires the Bellman terminal objective to match
the intended wealth objective. A forced future sale is an additional modeling
choice. [Boyd et al. (2017), section 5.2](https://web.stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf)
likewise distinguish the current executed trade from a multi-period plan and
describe an optional terminal cash constraint that accounts for unwind costs.
That supports keeping the conventions explicit; their convex framework does
not certify our integer/minimum-order problem.

**v264–v266: marked-wealth replay.** Optimize actual next-event marked wealth,
paying entry/rebalance fees and borrowing but assuming no next-event sale.
The simulator still charges all actual subsequent trades and settles inventory
at the real window end. This is a receding one-event policy, not an exact
multi-event improvement. The previous grid, friction and market traces and
all their metrics reproduce exactly before this comparison. The prior
rankings are recorded before loading final candles. No cash gate masks the
new policy's diagnostic returns.

| Frozen-law suite | Origin 1 return | Origin 2 return | Origin 3 return | Final return | Final drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| May, v264 | −16.693550% | −16.892036% | −19.149608% | +0.922135% | 9.579015% |
| March, v265 | −10.589303% | −14.534949% | +16.823723% | −3.390721% | 7.850335% |
| January, v266 | −3.128817% | −14.695077% | −7.734057% | −0.303944% | 11.180256% |

Marked-wealth prior scores are −0.217907/−0.054846/−0.104743. They do not
justify selecting this policy over cash. In May's active origin, 145 orders
and 61 reversals pay $1,383.93 in fees, but gross directional PnL also loses
about $519.29. This is not solely a fee problem. March's final window enters
long on March 20 at 09:03, forecasting +5.0701 bp after the current fee;
its next realized account move is −37.4006 bp. It then retains the position
through the window. January's final position initially benefits from its long
entry, but also remains invested through the subsequent loss. Preserve the
ability to size and hold useful exposure; multi-event continuation must
account for future exits/reversals and the costs those actions imply.

The marked screens take 9.29/4.82/4.60 seconds while reusing the earlier
expensive exhaustive checks. All 93 focused tests and workspace typechecks
pass. The three models' probabilities and the broader retained strategy remain
unchanged. These 12 replay phases are not the full 28-window suite.

**Next step under the revised goal:** use the verified one-event calculation
as the leaf of a multi-event Bellman reference, with the chosen terminal
objective explicit. Measure continuation and order-search errors separately
under the same frozen law. Do not apply the one-event concavity argument
blindly after a discrete future maximization, or replace that calculation
with a fit to realized future paths and call it exact. Next-open execution
and minute borrowing also remain distinct from the event-level forecast law.
Conditional one-event optimality is now directly checked; conditional
multi-event and full-suite optimality remain open.

```powershell
node --conditions=development --import tsx scripts/audit-event-one-step.ts --reference event-policy-fixed-model-grid-refinement-v257 --output event-policy-one-event-lot-optimizer-v258
node --conditions=development --import tsx scripts/audit-event-one-step.ts --reference event-policy-fixed-model-grid-refinement-v257 --output event-policy-one-event-lot-optimizer-v259
node --conditions=development --import tsx scripts/screen-event-one-step.ts --source event-policy-quadrature-policy-may-v121 --output event-policy-one-event-lot-may-v260
node --conditions=development --import tsx scripts/screen-event-one-step.ts --source event-policy-quadrature-policy-march-v122 --output event-policy-one-event-lot-march-v261
node --conditions=development --import tsx scripts/screen-event-one-step.ts --source event-policy-quadrature-policy-january-v120 --output event-policy-one-event-lot-january-v262
node --conditions=development --import tsx scripts/audit-event-one-step.ts --reference event-policy-fixed-model-grid-refinement-v257 --output event-policy-one-event-marked-optimizer-v263
node --conditions=development --import tsx scripts/screen-event-one-step.ts --source event-policy-quadrature-policy-may-v121 --reference event-policy-one-event-lot-may-v260 --output event-policy-one-event-marked-may-v264
node --conditions=development --import tsx scripts/screen-event-one-step.ts --source event-policy-quadrature-policy-march-v122 --reference event-policy-one-event-lot-march-v261 --output event-policy-one-event-marked-march-v265
node --conditions=development --import tsx scripts/screen-event-one-step.ts --source event-policy-quadrature-policy-january-v120 --reference event-policy-one-event-lot-january-v262 --output event-policy-one-event-marked-january-v266
```

### Two-event optimality bounds and replay — v267–v286

The new `decideEventTwoStep` evaluates each candidate root order with the
verified exact one-event continuation. Its lower value is therefore a
feasible stochastic Bellman value, not a maximum over realized future paths.
The root search covers ordinary cap-feasible integer lots and the separate
maximum-notional recovery clips. Minimum orders, fees, leverage, borrowing,
OHLC liquidation checks and the account's existing inventory remain active.
It supports marked wealth and proportional-friction terminal conventions;
minimum-order-dependent terminal dust is excluded because its discontinuity
breaks the monotonic relaxation used here.

The first interval bound constructed a portfolio with both the largest cash
and largest asset quantity attainable in the interval. This is conservative
but weak. In v268, one saved May cash state needs 8.54 seconds for eight
evaluations and still has a **6,919.87 bp** upper/lower gap. Do not scale that
method to backtests. The v267 small-lattice audit passed, which illustrates
why correctness alone was insufficient evidence for useful computation.

The replacement bound uses a continuous one-event relaxation. A shadow
execution price `m` between `1−f` and `1+f` undercharges every real order.
With cash `C`, asset quantity `Q` and price `P`, its resource budget is
`A = C + m Q P`. For any ordinary cap-feasible continuation,

```text
absolute expected log terminal wealth <= log(C + m Q P) + K(m)
```

`K(m)` solves a one-dimensional concave expected-log problem with the same
leverage, borrowing and terminal friction. Order minima/maxima, lot rounding
and maintenance constraints are relaxed only in the upper bound. Endpoint
shadow prices are precomputed; an interior marginal price gives a tight
bound in the no-trade region. Tangents bound the continuation at every root
order in an interval. A common root order is used across first-event outcomes.
The maximum of the resulting piecewise affine bounds is checked at the
cash, position and borrowing knots. Above-cap recovery clips are bounded
separately, since their feasible sets do not preserve the usual portfolio
dominance argument. Tolerances include the ordinary cap tolerance.

This is an implementation-specific dual relaxation, not a direct application
of a theorem covering the project's discrete exchange rules. The separation
between a feasible policy and a performance bound follows the approach in
[Boyd et al., Performance Bounds and Suboptimal Policies for Multi-Period Investment](https://web.stanford.edu/~boyd/papers/port_opt_bound.html).
That paper treats convex stochastic-control problems and quadratic bounds;
it does not certify this integer/minimum-order model. The forecast/policy
separation is also explicit in
[Multi-Period Trading via Convex Optimization](https://web.stanford.edu/~boyd/papers/cvx_portfolio.html).

The search retains the upper bounds of tolerance-pruned intervals. Exhausting
the evaluation budget leaves unresolved intervals and an explicit gap; it
cannot silently convert an incomplete search into an exact result. These are
floating-point numerical bounds, tested against an independent enumeration,
not directed-rounding formal proofs. Utility gaps below are in basis points
of **expected log wealth**, not realized-return basis points.

**Validation and computation.** The final v273 audit uses the nine saved
two-event fixture/account cases plus 1,800 seeded cases, two terminal
conventions and budgets of 2/16/512: **10,854 queries over 3,618 distinct
problems**. It checks that the exact optimum lies between returned bounds,
that the chosen order's exact value equals the lower value, and that larger
budgets do not worsen either bound. All 3,618 problems converge at budget
512, with zero measured action regret. Only 2,561 converge at budget 16;
the others are reported as unresolved. Cases include asymmetric two-state
transitions, minimum quantities/notionals, large fees/borrow, extrema and
above-cap recovery. The new search takes 702 ms in aggregate versus 554 ms
for exhaustive enumeration on these tiny lattices. Its benefit is avoiding
enumeration of large market lattices, not universally beating tiny oracles.

Using the incumbent action as the tangent anchor avoids repeated subdivision
around a no-trade kink. In the May first-state short/long inventory probes,
v272 needed 29/17 evaluations and 10.15/5.85 seconds. V274 needs two
evaluations for each and about 0.91/0.59 seconds, with unchanged actions.
Across v274–v276, all nine first-state cash/long/short probes certify holding
within 0.001 bp. Maximum gap is 0.000145 bp. All reuse the exact saved laws
v121/v122/v120; none retrains a forecast.

The next nine probes use the first three actual marked-H1 orders in each
first prior origin. Six certify at budget 32; three require only 33–34 total
evaluations when rerun with budget 64 (v280/v282). All nine then meet the
0.001 bp tolerance. Observations worth preserving:

- March's first short entry is rejected: waiting has H2 value **+1.30890 bp**,
  versus +0.22401 bp for entering the H1 short with optimal H1 continuation.
  Two subsequent recorded position-change states also favor holding.
- May's first long and later short-to-long reversal remain. Its first
  long-to-short reversal changes from −0.50756 to −0.50710 BTC, an improvement
  of only 0.000408 bp over the seed under this law.
- January's first long remains. The later reversal changes from −0.57799 to
  −0.57577 BTC, and its subsequent short addition from −0.00365 to −0.00140.
  Gains are small: 0.00875 bp over the reversal seed and 0.00509 bp over the
  best initial candidate for the addition. This is sizing refinement, not
  evidence of improved directional forecasts.

These are same-account policy probes. An old grid-H2 action can also appear
as a root seed, but that grid used a friction terminal; comparing it to the
marked optimum mixes objective and search differences. Do not report that
comparison as pure sizing regret.

An exact one-event shortcut now checks the two directional derivatives at
holding. For marked/friction objectives, if they bracket zero and holding
survives every positive-probability event, it is already a global optimum of
the concave order objective. This respects the long-borrowing kink at 1× and
does not apply to market-dust settlement. V283 still matches all **5,454**
independent one-event enumeration checks. In the 16-event March prefix,
v284 reproduces v281's complete H2 trace and both strategies' economic
metrics exactly, reducing H2 replay from 7.51 to **2.19 seconds**.

**Full-origin result.** The prefix covers January 14, 2023, 00:00–01:28 UTC.
It stays cash under H2 versus an H1 loss of 1.2988%. The subsequent v285 run
covers the complete January 14–February 4 prior origin, with all 402 event
decisions and the unchanged next-open simulator:

| Metric | Repeated exact H1 | Receding bounded H2 |
| --- | ---: | ---: |
| Return | −10.589303% | **−11.824201%** |
| Maximum drawdown | 15.292647% | 16.729193% |
| Fees | $124.5673 | $50.0011 |
| Borrowing | $8.4518 | $14.2763 |
| Orders / reversals | 33 / 4 | 31 / 1 |
| Gross long PnL | +$36.9297 | −$54.6738 |
| Gross short PnL | −$962.8409 | −$1,063.4689 |
| Short exposure minutes | 13,585 | 23,450 |
| Canceled orders / liquidations | 6 / 0 | 5 / 0 |

All **402 H2 decisions certify**, with maximum gap **0.000796 bp**. Replay
takes **42.50 seconds**. H1 reproduces the earlier v265 full-origin economics
exactly. Early avoidance of churn does not generalize to an improved total
return: both strategies short before the January 20 rally, while H2 retains
the short after January 25 and consequently misses the long exposure held by
H1 during the January 29 rise. Lower transaction cost is outweighed by worse
directional exposure and additional borrowing.

**Diagnosis, v286.** On the H2-visited accounts, H1 and H2 choose different
orders only five times. The largest later difference is January 25 at 23:04:
H1 would reverse a −0.9497× short to +1× long; H2 holds. Under the fixed H2
law, holding is worth **−8.79966 bp**, versus **−11.30181 bp** for that
reversal with optimal H1 continuation. The 2.50215 bp preference is far above
the 0.0000245 bp residual search gap. Finer root sizing cannot reverse this
conclusion under the same two-event model.

The audit also avoids comparing a two-event prediction to a one-event
realization. In 396 of 401 available successor pairs, the next receding-H2
order equals the planned H1 continuation. Requiring both planned orders to
fill leaves **386 aligned two-event intervals**. Their mean predicted log
gain is +0.2938 bp versus realized −6.1475 bp. On January 20 at 20:02 the
model predicts −2.2209 bp while the aligned pair realizes −273.1928 bp.
These overlapping, agreement-selected intervals are diagnostics, not an
independent accuracy estimate. Execution still uses next-open prices and
minute borrowing, unlike the decision-price aggregate event law. Nevertheless,
the large misses in these pairs cannot be attributed to a different second
policy action or a loose root optimizer.

The finite H2 value assumes optimal H1 on the next event; the receding replay
optimizes H2 again at every decision. Its per-decision certificate therefore
does **not** establish stationary-policy optimality or execution-consistent
global optimality. Next steps under the revised goal are to extend the
certified checks across the remaining frozen-law origins/windows, measure
the effect of horizon resets and deeper continuation, and reconcile the
event law with next-open execution before returning to forecast fitting.
The useful behavior is less unnecessary turnover with correct cap recovery;
the unresolved bad behavior is persistence in wrong directional exposure.
Do not promote this H2 policy on the basis of its prefix or root certificates.
All 100 focused tests and workspace typechecks pass. No incumbent changed.

Reproduction uses fresh output directories; selected examples:

```powershell
node --conditions=development --import tsx scripts/audit-event-two-step.ts --reference event-policy-fixed-model-grid-refinement-v257 --output event-policy-two-event-incumbent-bound-audit-v273 --cases 1800
node --conditions=development --import tsx scripts/probe-event-two-step.ts --source event-policy-one-event-marked-may-v264 --output event-policy-two-event-active-may-refine-v280 --budgets 64 --selection marked-orders --offset 1
node --conditions=development --import tsx scripts/audit-event-one-step.ts --reference event-policy-fixed-model-grid-refinement-v257 --output event-policy-one-event-hold-shortcut-v283
node --conditions=development --import tsx scripts/replay-event-two-step.ts --source event-policy-one-event-marked-march-v265 --output event-policy-two-event-origin-march-v285 --events 402 --budget 64
node --conditions=development --import tsx scripts/audit-event-two-step-replay.ts --source event-policy-two-event-origin-march-v285 --output event-policy-two-event-origin-march-audit-v286
```

### Horizon countdown, prepared continuation and inspector coverage — v287–v297

**Horizon consistency is a policy question.** Receding H2 plans one exact H1
continuation, then solves H2 again after the next observation. V287 compares
this with two predeclared countdown controllers on the same complete March
prior origin, January 14–February 4, 2023. One alternates H2/H1 starting at H2;
the other starts at H1. Each H1 step still optimizes from the actual newly
observed account and state. Neither precommits an order from a hypothetical
future path. The counter advances on every event, including cash and canceled
orders; its phase can be restored explicitly when resuming a replay.

| Controller | Return | Maximum drawdown | Fees | Orders / reversals |
| --- | ---: | ---: | ---: | ---: |
| Repeated H1, v285 | −10.589303% | 15.292647% | $124.5673 | 33 / 4 |
| Receding H2, v285 | −11.824201% | 16.729193% | $50.0011 | 31 / 1 |
| Countdown starting H2, v287 | **−10.302894%** | 15.292555% | $92.2562 | 30 / 3 |
| Countdown starting H1, v287 | −12.347271% | 16.729260% | $82.2768 | 34 / 2 |

Both countdowns cover 402 events and certify all 201 H2 decisions. Maximum
gaps are 0.000643/0.000798 bp; runtime is 20.77/22.00 seconds. At January 25,
23:04 UTC, the first countdown is at H1 and reverses its −0.9497× short to
approximately +1× long with a +0.76144 BTC order. The other is at H2 and holds
the short. The 2.04 percentage-point return difference between arbitrary
starting phases rejects treating the better parity as a stationary solution.
This is an identified horizon effect, not an improvement in sign forecasting.

The local `docs/theory/Position management.md` discussion of reoptimization
requires future actions to respond to the observed state. Countdown H1 does
that; changing the remaining horizon is what changes the objective. Finite
horizon effects on transaction-cost no-trade regions are also studied in
[Gennotte and Jung, Investment Strategies under Transaction Costs: The Finite Horizon Case](https://pubsonline.informs.org/doi/10.1287/mnsc.40.3.385).
Their continuous portfolio model motivates checking horizon-dependent
boundaries; it does not establish convergence for this project's minimum
orders, integer lots, maintenance checks or maximum-order recovery clips.

**Actual March inspector window.** V288 freezes the newer v122 final joint
forecast and replays March 18–25, 2023 without another model or policy
selection. H1 reproduces v265 exactly. H2 certifies all 223 decisions, with
maximum gap 0.0009975 bp, in 24.31 seconds.

| Metric | Repeated H1 | Receding H2 |
| --- | ---: | ---: |
| Return | −3.390721% | −3.724929% |
| Maximum drawdown | 7.850335% | 9.341633% |
| Fees | $23.5925 | $46.8066 |
| Orders / reversals | 2 / 0 | 10 / 1 |
| Canceled orders / liquidations | 0 / 0 | 0 / 0 |

V297 recomputes H1 on every H2-visited account. There are three action
differences. The largest is March 23 at 12:59 UTC: H1 holds a +1× long, while
H2 sells 0.69969 BTC to reach −0.98836×. Under the fixed two-event law, that
reversal has value −9.303897 bp versus −15.682490 bp for holding with optimal
H1 continuation. Its **6.378593 bp** advantage exceeds the residual search
gap by orders of magnitude. It loses the rally later that day, then benefits
from the March 24 fall. Root-search refinement cannot correct that direction
while retaining this forecast and H2 objective.

In 219 of 222 available successor pairs, the next H2 order equals the planned
H1 continuation and both orders fill. Those overlapping diagnostic intervals
have mean predicted log gain +0.288702 bp versus realized −3.022474 bp. A
March 22 long hold predicts −4.014549 bp over two events and realizes
−239.538242 bp. These agreement-selected residuals are not a fresh holdout
accuracy estimate, and the event law still omits the next-open execution gap.

**Compile the fixed one-event law once for repeated account queries.**
`event-one-step-prepared.ts` uses log-wealth homogeneity and piecewise
concavity. Within each buy/sell region, the continuous optimum in post-trade
exposure depends on the law and costs, not equity or price. The compiler
finds those two exposure targets, respecting borrowing kinks and every
positive-probability survival constraint. Each account query evaluates
integer neighbors of the targets and the feasible minimum/maximum/cap
boundaries, plus exceptional above-cap recovery clips. Every candidate is
checked against the original trade and holding rules. Holding strictly inside
the no-trade region returns early. Singular costs, unreliable very fine
lattices and unusually wide recovery tolerance bands fall back to the general
one-event solver. This is an internal continuation optimization, not a new
forecast or an approximate distribution compression.

V292 matches the independent exhaustive oracle on **3,636** marked/friction
queries with zero action regret. Preparation takes 122.20 ms; queries take
21.27 ms versus 37.39 ms for enumeration. Preparation is therefore not a
universal speedup for one account per law. Its intended use is H2's many
accounts under each fixed successor law. With the final holding shortcut,
v293 repeats 10,854 H2 queries over 3,618 small problems at budgets 2/16/512.
All 3,618 certify at budget 512 with zero action regret; unresolved smaller
budgets remain explicit. The maximum budget-512 gap is 0.00000151 bp.

The dense July 2022 inspector window provides the practical check. V295
reproduces v291's complete 48-event account/action trace and economic metrics
exactly, with maximum action-value difference zero. Runtime falls from
**124.11 to 37.39 seconds**, a 3.3× speedup. All 48 decisions still certify.
Only after this reproduction was the slower all-window process stopped.
Its five completed results remain in v291 with a `stopped.json` provenance
record; v296 continues the other 23 windows. No partial window is counted.

**Coverage baseline.** V289 restores the complete saved v10 static forecasts
for all **28 actual non-fit KAMA inspector windows**, excluding `latest` and
`fit-*`. Config intent alone is insufficient: the runner validates the live
catalog, all saved model files and results, each chronological training cutoff,
and calibration end. SHA-256 hashes identify the frozen models. These are
the older 120-bp barrier laws, not the newer three-window run-conditioned
v121/v122/v120 forecasts. Results from the two model families must stay separate.

The exact H1 replay covers **1,217 events** in 2.17 seconds. Every decision is
feasible with finite utility; 15 windows gain, eight lose, and five stay cash.
No fitting, depth selection or cash gate is applied. The complete paired H2
merge is reported below in v309; a finite H2 action certificate still does
not establish stationary or next-open-execution-consistent optimality.

All **103 focused tests** and workspace typechecks pass. New tests cover
countdown phase and cash advancement, actual-state feedback and resume,
prepared-law agreement with independent enumeration, a five-billion-lot
case, and preservation of a 1e−300-probability liquidation outcome. No forecast
retraining or incumbent promotion occurred in this checkpoint.

Reproduction (use new output names rather than overwriting artifacts):

```powershell
node --conditions=development --import tsx scripts/replay-event-two-step.ts --source event-policy-one-event-marked-march-v265 --output event-policy-two-event-countdown-march-v287 --events all --modes countdown2,countdown1 --budget 64
node --conditions=development --import tsx scripts/replay-event-two-step.ts --source event-policy-one-event-marked-march-v265 --output event-policy-two-event-final-march-v288 --phase final --events all --budget 64
node --conditions=development --import tsx scripts/audit-event-one-step.ts --reference event-policy-fixed-model-grid-refinement-v257 --output event-policy-prepared-one-event-audit-v292 --prepared
node --conditions=development --import tsx scripts/audit-event-policy-suite.ts --source event-policy-all-mean120-1x-cap-v10 --output event-policy-one-event-full-suite-v289 --depth 1
node --conditions=development --import tsx scripts/audit-event-two-step-replay.ts --source event-policy-two-event-final-march-v288 --output event-policy-two-event-final-march-audit-v297
```

### Dense-law evaluation without changing the forecast — v299–v308

The first full-suite timing check did not expose the worst computational
case. June 2022's three-state forecast has a state with 4,699 outcomes. Its
first H2 decision requires ten action evaluations and 21 interval bounds;
46,990 continuation queries repeatedly traverse dense successor laws. V299
takes **15.94 seconds for this single decision**. The v300 CPU profile finds
that repeated holding scores and marginal shadow-price calculations dominate.
Deeper recursion with this implementation would be wasteful.

First, concavity removes distant ordinary-order candidates after feasibility
checks: for each buy/sell component, only the feasible neighbors of its
continuous optimum can win. Holding and exceptional recovery clips remain
separate. The shadow upper bound skips its marginal interior-price calculation
outside the precomputed continuous no-trade region. Both endpoint shadow
bounds remain valid even if the boundary shortcut omits a useful interior
candidate; that can only loosen the bound. V301 matches all 3,636 one-event
oracle queries and reduces evaluated candidates from 49,348 to 8,299. V304
repeats the 10,854-query two-event audit with zero regret at budget 512.
However, the dense market probe only falls to **12.70 seconds** in v303.

The decisive change is `event-holding-law.ts`. In each borrowing/sign piece,
terminal holding wealth is affine in exposure. For an anchor `x0`, write

```text
H(x) = sum_i p_i * [log(w_i(x0)) + log(1 + c_i * (x - x0))]
rho = max_i abs(c_i * (x - x0))
```

Sixteen precomputed moments evaluate the log series. With probability mass
`M`, the omitted value is bounded in absolute size by
`M*rho^17 / (17*(1-rho))`; its derivative remainder is bounded by
`M*max(abs(c_i))*rho^16 / (1-rho)`. The series is used only when **both bounds
are at most 1e−17**. Otherwise the original finite sum is evaluated. This
bound concerns series truncation; floating-point summation is still subject
to the numerical padding described in v267–v286. Upper-bound use is limited
to leverage at most 10 and fee-times-leverage at most 0.5, with the existing
1e−11 pad. Extreme parameter cases keep the direct calculation.

No outcome is removed, merged or reweighted. Every positive-probability
extremum separately constrains survival, even at probability 1e−300. Near a
survival boundary, the original per-outcome liquidation check decides. The
current dense market law's returns lie approximately between −2.98% and
+2.80%, so this small, rigorously truncated series is effective at 1×.

V305 splits each atom into 32 identical equal-mass copies to activate dense
evaluation while leaving the independent oracle's distribution unchanged.
It checks 618 distinct two-event problems at three budgets: **1,854 queries**.
All 618 certify at budget 512 with zero action regret; smaller unresolved
budgets stay explicit. The new focused test also compares values, derivatives,
direct fallbacks, borrowing pieces and rare-outcome survival against the
original holding calculation. All **104 tests** and workspace typechecks pass.

The dense June decision now takes **0.625 seconds** in v306, a 25.5× speedup
over v299. It selects the same order, reproduces the executed trace exactly,
and changes the lower value by only 1.19e−18. Its numerical upper bound changes
by 5.97e−13, far below the 1e−7 stopping tolerance. V307 then reproduces every
order, account transition and economic metric of the full 48-event July
window in **3.15 seconds**, versus 37.39 seconds in v295 and 124.11 seconds
in v291. Maximum action-value difference is 3.90e−18.

After those checks, v296 was stopped with its two complete windows preserved;
its unfinished June replay is not counted. V308 runs only the remaining 21
windows. June completes in **87.43 seconds**, with all 224 actions certified.
The stopped artifacts, profiles and reproduction checks remain available.
No forecasting model, state transition or exchange constraint changed.

Reproduction:

```powershell
node --conditions=development --import tsx scripts/audit-event-two-step.ts --reference event-policy-fixed-model-grid-refinement-v257 --output event-policy-two-event-neighbor-full-audit-v304 --cases 1800
node --conditions=development --import tsx scripts/audit-event-two-step.ts --reference event-policy-fixed-model-grid-refinement-v257 --output event-policy-two-event-dense-series-audit-v305 --cases 300 --copies 32
node --conditions=development --import tsx scripts/audit-event-policy-suite.ts --source event-policy-all-mean120-1x-cap-v10 --output event-policy-two-event-regime-series-v306 --depth 2 --budget 64 --windows regime-down-2022-06 --limit-events 1
node --conditions=development --import tsx scripts/audit-event-policy-suite.ts --source event-policy-all-mean120-1x-cap-v10 --output event-policy-two-event-series-window-v307 --depth 2 --budget 64 --windows sideways-churn-2022-07
```

### Complete fixed-forecast inspector audit — v309

The merger verifies exactly one complete H1 and H2 result for every current
non-fit catalog window. H2 results comprise five windows from v291, two
from v296 and 21 from v308; stopped partial windows and duplicate performance
checks are excluded. It verifies original model-file hashes and training
boundaries, equal event clocks/states across horizons, and every stored
upper/lower gap. No chronological fitting, forecast scaling, cash gate or
depth selection is performed. This is the older complete v10 forecast
baseline, separate from the newer three-window laws used in v287/v288.

All **1,217 decisions across all 28 windows are feasible, finite and certified**
at the H2 tolerance of 0.001 bp of expected log wealth. The maximum gap is
**0.00099860244 bp**. There are no unresolved states requiring a larger budget.

| Metric | Repeated exact H1 | Receding bounded H2 |
| --- | ---: | ---: |
| Positive / negative / cash windows | 15 / 8 / 5 | 16 / 8 / 4 |
| Mean window return | +1.764751% | +3.141367% |
| Mean window log growth | 0.01397962 | 0.02665118 |
| Worst window return | −21.907112% | −15.756277% |
| Largest window drawdown | 25.722593% | 19.776520% |
| Orders | 254 | 483 |
| Fees | $2,077.5789 | $2,665.6511 |
| Borrowing | $44.7577 | $54.2025 |
| Canceled orders / liquidations | 17 / 0 | 22 / 0 |

These averages and totals describe independent $10,000 account resets over
overlapping, repeatedly inspected research windows. They are neither a
compounded portfolio return nor an independent holdout estimate. H2 improves
seven windows, worsens eleven and leaves ten unchanged. The higher mean is
concentrated in a few windows; it does not establish reliable profitability.

| Inspector window | H1 return % | H2 return % | H2 drawdown % | Certified decisions |
| --- | ---: | ---: | ---: | ---: |
| sideways-churn-2022-07 | −2.8958 | −4.0409 | 10.4599 | 48/48 |
| sideways-churn-2022-05 | 11.3758 | 9.3143 | 5.5018 | 50/50 |
| sideways-churn-2021-12 | 0.0663 | 0.0663 | 6.7104 | 58/58 |
| sideways-churn-2021-09 | 0.0000 | 0.8201 | 6.7994 | 57/57 |
| sideways-churn-2023-03 | −7.1867 | −1.9928 | 4.4162 | 48/48 |
| regime-up-2023-03 | −0.3070 | 7.3504 | 8.2592 | 90/90 |
| regime-flat-2026-04 | 0.0000 | 0.0000 | 0.0000 | 15/15 |
| regime-down-2022-06 | −11.5705 | 32.6760 | 16.5138 | 224/224 |
| shape-up-low-2024-02 | 0.0000 | 0.0000 | 0.0000 | 8/8 |
| shape-up-high-2022-06 | −9.5832 | −15.7563 | 19.7765 | 69/69 |
| shape-down-low-2023-06 | 4.5145 | −4.3027 | 6.0689 | 10/10 |
| shape-down-high-2022-06 | 3.9124 | 11.8006 | 14.7218 | 138/138 |
| shape-flat-high-bias-2021-10 | 6.5674 | 2.1703 | 6.3337 | 21/21 |
| shape-flat-high-bias-low-2025-02 | −2.5531 | −2.5531 | 2.7875 | 4/4 |
| shape-flat-low-bias-2024-07 | 0.0069 | 0.0084 | 6.5076 | 19/19 |
| shape-flat-low-bias-low-2025-07 | 0.0000 | 0.0000 | 0.0000 | 4/4 |
| shape-flat-mid-bias-2024-01 | 5.3733 | 1.6425 | 11.0243 | 24/24 |
| shape-flat-mid-bias-low-2023-09 | −0.3154 | −1.3424 | 2.4792 | 4/4 |
| sharpe-up-3d-2024-11 | 8.0577 | 8.0577 | 2.2694 | 17/17 |
| sharpe-up-3d-2023-12 | 11.4502 | 11.4502 | 2.3958 | 14/14 |
| sharpe-down-3d-2026-06 | 11.3389 | 11.2089 | 3.1291 | 18/18 |
| sharpe-down-3d-2023-03 | 0.9409 | 0.9409 | 2.1367 | 11/11 |
| sharpe-up-7d-2023-12 | 23.2259 | 23.2259 | 1.8354 | 21/21 |
| sharpe-up-7d-2024-11 | 0.8874 | −12.7710 | 14.0520 | 38/38 |
| sharpe-down-7d-2023-03 | 12.9823 | 12.9322 | 3.0158 | 21/21 |
| sharpe-down-7d-2026-06 | 5.0319 | 4.8850 | 4.1155 | 21/21 |
| failure-down-3d-2022-06 | 0.0000 | 0.0000 | 0.0000 | 64/64 |
| failure-down-7d-2022-06 | −21.9071 | −7.8322 | 16.1702 | 101/101 |

**Behavior to preserve and failures to explain.** June's regime-down result
improves by 44.2466 percentage points. H1 never shorts: its gross long PnL is
−$1,124.52. H2 enters short at the first event and later rebalances; gross
short PnL is +$3,754.56 and long PnL −$188.88. The initial short's H2 value is
5.732279 bp versus 0.113109 bp for waiting with optimal continuation. This
is a continuation-driven entry absent from H1, not a sign-head replacement.
Its higher order count and fees are visible rather than hidden by a gate.

The same mechanism fails in November 2024's seven-day up window. H1 remains
cash initially and eventually makes a profitable long. H2 immediately shorts
to −0.84375×: predicted H2 value is +1.124985 bp versus +0.024209 bp for
waiting. Its first aligned two-event realization is **−223.045589 bp**. Gross
short losses reach $1,304.15, explaining the window's deterioration from
+0.8874% to −12.7710%. The 1.100776 bp model advantage over waiting is far
above its 0.000291 bp search gap. Smaller lots cannot remove that entry while
retaining the frozen H2 law and objective.

In June 2023's low-volatility down window, H2 reverses a −0.5204× short to
approximately +1× long on June 5 at 10:26 UTC. The model prefers that reversal
by **3.878714 bp** over holding the H1 short, but its aligned two-event result
is −290.721368 bp. This loses a useful bearish exposure: the whole-window
result falls from +4.5145% to −4.3027%. It is a concrete reversal failure to
retain in future horizon and forecast comparisons.

Across the suite, H1 and H2 differ on **290 H2-visited accounts**. There are
908 successor-order agreements out of 1,189 available pairs. Requiring both
orders to fill and excluding pairs ending at a potentially truncated window
boundary leaves **851 aligned intervals**. Their mean predicted/realized log
gains are +7.015449/+9.407487 bp. This aggregate looks better than the newer
March law's diagnostic, but it still contains large failures and selected,
overlapping intervals. For example, June 7, 2022 at 22:53 predicts +3.845856
bp and realizes −458.662631 bp. Do not infer calibrated tail risk or robust
sign skill from the aggregate mean.

**Remaining conditional-optimality work.** The H2 certificate applies to a
two-event plan with H1 continuation at the decision price. Receding execution
resets its horizon, and the simulator fills at the next open; 22 planned
orders cancel. V287's phase dependence and the observed successor-policy
differences therefore remain unresolved by the completed coverage audit.
The next useful step is a bounded deeper continuation and an explicit
execution-consistent action contract, preserving the profitable June short
and testing the adverse November/June-2023 reversals on the same frozen laws.
Do not start another sign-head or profitability sweep before resolving this
policy objective. Reuse compiled fixed-law work across accounts before
expanding the event tree; the dense profiles show why naive recursion is not
yet a productive computation plan.

```powershell
node --conditions=development --import tsx scripts/merge-event-policy-suite.ts --one event-policy-one-event-full-suite-v289 --two event-policy-two-event-full-suite-v291,event-policy-two-event-full-suite-prepared-v296,event-policy-two-event-full-suite-series-v308 --output event-policy-two-event-full-suite-merged-v309
```

### Three-event action bounds and global direction checks — v310–v337

The prior turn completed finite H2 coverage, not the full conditional-policy
objective. This checkpoint freezes the same forecasts and extends the
optimality evidence to a third event. The predeclared market cases are the
first entry in `regime-down-2022-06`, where H2's short was useful, and
`sharpe-up-7d-2024-11`, where it was harmful. These are diagnostic states from
the existing research suite, not newly selected holdout windows.

**Reuse fixed-law work.** `prepareEventTwoStep` owns a snapshot of the model
and costs and reuses compiled one-event optima and shadow bounds across
account queries. Each query retains its own root orders, bounds and budget.
The root H1 seed also uses the prepared one-event solver. A test mutates the
original model/cost objects after preparation and interleaves different
accounts; the prepared search remains tied to its original forecast.

V331 reproduces all 48 July orders, account transitions, action values and
economic metrics exactly in **1.176 seconds**, versus 3.147 seconds in v307.
V332 reruns 10,854 H2 queries over 3,618 independent small problems. All
3,618 converge at budget 512 with zero measured action regret. Budget-two
certification changes slightly with the new seed (266 rather than 270);
unresolved states remain explicit. No broader H2 replay is needed to claim
this particular speedup; v309 remains the complete 28-window audit.

**Three-event action values.** `event-three-step.ts` evaluates a specified
feasible root order as its immediate log-wealth change plus the expectation
of first-event holding gain and a bounded H2 value at each successor. The
future optimizer observes the successor state before choosing its action.
Every original positive-probability outcome is checked for liquidation.
Only exactly identical successor accounts and holding factors are coalesced;
this is not a forecast approximation. A computation profile with five or
twenty outcomes leaves the full action interval at `[−Infinity, Infinity]`.
It never treats those outcomes as a representative sampled expectation.

Nearby queries can seed the full H2 search with its preceding buy/sell
targets. These seeds are projected through the actual order rules and do
not replace the upper-bound search. The independent v315 audit checks
**1,200 H3 action queries** over 100 small cases, both terminal conventions,
three root quantities and cold/warm continuation searches. Every result
brackets the exhaustive Bellman value; maximum gap is 0.000000200 bp.

The complete initial November comparison, v316, evaluates 3,684/3,695
distinct successor accounts for waiting/the H2 short. All their H2 searches
certify. The full calculations take 222.33/192.58 seconds. This cost confirms
that a naive H3 backtest over all 28 windows would be premature, despite the
earlier cheap prefix profiles. June's complete cash comparison takes 868.96
seconds in v317. Its unfinished short comparison is later replaced only after
the faster bound-assisted method is verified; the completed cash result and
a `stopped.json` provenance record remain intact.

**A global upper bound beyond the listed candidates.** Comparing only waiting
and the H2 order cannot prove root optimality. `event-multi-step-upper.ts`
therefore propagates continuous upper relaxations through the finite horizon.
With current cash `C`, current asset notional `N`, and a shadow execution
price `m = 1+d` in the bid/ask interval, the bound is

```text
F_h(C, N) <= log(C + (1+d)*N) + K_h(d)
```

Here `F_h` is absolute expected log terminal wealth. `K_h` maximizes the
expectation of the preceding upper envelope over normalized post-trade
notional, retaining borrowing and the ordinary leverage cap. Each branch's
shadow wealth is affine in that notional. Minimum/maximum orders, integer
lots and maintenance constraints are relaxed only in this upper calculation;
the feasible H3 lower policy retains them all. Pairing supporting tangents
bounds each entire concave piece, including a nonsmooth maximum. The marked
terminal is `log(C+N)`; proportional-friction settlement uses the bid/ask
minimum. The default coefficient tolerance and numerical pad are 1e−10.
Probabilities must sum to one within 1e−12 for this homogeneity calculation.
These remain floating-point numerical bounds, not directed-rounding proofs.

The exchange's exceptional maximum-order recovery trades require an explicit
guard. If the smallest recovery clip is at least twice the largest reachable
absolute asset notional, it cannot reduce exposure: it crosses the entire
position and at least its mirror while fees reduce equity. The query
propagates conservative equity, notional and price bounds through every
remaining event before using that argument. If it cannot exclude recovery
exceptions, it returns `Infinity`; the ordinary relaxation cannot silently
certify that account. The guard succeeds for both market probes.

This separates a feasible policy from its performance bound, following the
general approach in
[Boyd et al., Performance Bounds and Suboptimal Policies for Multi-Period Investment](https://stanford.edu/~boyd/papers/port_opt_bound.html)
and
[Wang, O'Donoghue and Boyd, Approximate Dynamic Programming via Iterated Bellman Inequalities](https://web.stanford.edu/~boyd/papers/adp_iter_bellman.html).
Those works motivate Bellman bounds and separate policy evaluation; they do
not prove this implementation's discrete-order or recovery-clip rules. The
homothetic shadow construction and the reachability guard above are the
project-specific derivation. They also preserve the full-objective and
feasibility requirements in `docs/theory/Position management.md`.

**Put shadow points where they matter.** A uniform nine-point fee grid gives
a November H3 upper/lower gap of 0.3393 bp. Thirty-three points still leave
0.2494 bp against the old H2-sized action and cost 9.10 seconds to compile.
Most shadow prices are uninformative. The marginal construction instead
evaluates the previous holding envelope on an exposure grid, obtains its
cash/notional supergradient, and uses their ratio as the shadow price. By
concavity, that holding portfolio maximizes the envelope under its marginal
resource price. Homogeneity gives the new coefficient directly, avoiding
another inner optimization. Both fee endpoints remain in the bound.

With 33 exposure points, compilation falls to **0.648 seconds** and the H3
gap against a refined short is 0.005335 bp. With 129 points, it takes 3.414
seconds and the gap falls to **0.000475592 bp**. The same saved joint outcomes
and probabilities are used throughout; no forecast is compressed or refit.

V322 and v324 each check 1,800 global H1/H2/H3 upper bounds on the same 100
seeded two-state problems, using uniform and marginal grids respectively.
There are no upper-bound violations. V333 also bounds each entire buy/sell
half of the root lattice: **5,400 global/directional comparisons**, 4,968
finite and 432 correctly identified as infeasible. None passes merely by
returning `Infinity`. Cases vary leverage, fees, borrowing, minimum orders
and initial long/short/cash inventory.

**November: a wrong forecast remains wrong after a third event.** Values
below are basis points of expected log wealth under the fixed forecast:

| Root quantity, BTC | H3 feasible lower value | H3 candidate upper value | Artifact |
| --- | ---: | ---: | --- |
| Wait, 0 | 0.836581 | 0.837037 | v316 |
| Original H2 short, −0.12423 | 3.456403 | 3.456747 | v316 |
| Refined short, −0.14031 | **3.687650** | 3.687906 | v321 |

The refined action is suggested by the relaxation, then evaluated through
all 3,695 discrete successor problems. The **global** H3 upper bound in v326
is **3.688126 bp**, only 0.000476 bp above this feasible lower value. Thus the
refined root order with its bounded H2 continuation is numerically within
0.001 bp of the full three-event optimum at this state, not merely the best
of three named candidates.

The directional upper bound is stronger evidence about the sign: **every
feasible nonnegative root order**, including cash and all long sizes, has H3
value at most **0.836640 bp**. The verified short exceeds this bound by about
2.85 bp. A finer root action search cannot change that directional conclusion
under the same three-event model. H3 increases the initial short from roughly
−0.844× to −0.953×, although the original short lost money in the observed
November rally. This is evidence of a forecast/horizon limitation, not a
reason to promote a more aggressive short or claim an improved backtest.

**Use the global bound to stop continuation search earlier.** An optional
compiled H2 upper envelope now checks the feasible root seeds before interval
subdivision. It is built from the same snapshot and falls back to the existing
search wherever its recovery-exclusion guard fails or its gap is too wide.
It also tightens the final gap after a budget stop. A test proves both the
no-subdivision cash case and the recovery-exception fallback.

V329 repeats the complete refined November action. Its interval overlaps
v321, with lower-value difference −3.09e−9, and all 3,695 continuations still
certify. Runtime falls from 61.49 to **39.30 seconds**. This permits a slightly
different feasible continuation within the same tolerance; it does not claim
bitwise-identical future orders. In June, the first 250 short continuations
certify in **2.68 seconds** (v328). Only then is v317's unfinished short
calculation stopped and replaced by v330. The complete cash result is reused.

**June: the useful short also survives the third event.** Waiting has a
complete H3 interval of [5.499063, 5.499773] bp in v317. The original
−0.3425 BTC short has [13.631340, 13.631737] bp in v330, after all 4,594
continuations certify in 188.53 seconds. V334's global upper bound leaves a
0.001294 bp sizing gap. Increasing the marginal grid from 129 to 257 points
in v335 barely improves that gap to 0.001282 bp, so no larger grid sweep is
justified. Instead v336 evaluates one root quantity suggested by that bound,
−0.3428 BTC. All 4,594 continuations certify in 207.45 seconds, producing
a feasible H3 lower value of **13.631966 bp** and candidate upper value of
13.632371 bp.

V337 recompiles the 257-point global bound in 7.10 seconds. Its upper value
is **13.632622 bp**, leaving **0.000656337 bp** above v336's feasible policy,
below the unchanged 0.001 bp tolerance. Every cash or long root order has
value at most **5.499317 bp**. This establishes global finite-H3 sizing and
direction at this second diagnostic state under the same frozen model.
It is not a new window backtest or evidence of stationary convergence.

All **108 focused tests** and workspace typechecks pass. No forecast fitting,
new calibration selection or incumbent promotion occurred. A finite H3
certificate at these diagnostic states still does not establish stationary
optimality, cover all 28 windows at H3, or resolve next-open execution versus
decision-price planning. The next computational requirement is reusable
feasible continuation-value bounds, so deeper policy evaluation does not
repeat thousands of nested searches at every real decision.

Reproduction (new output names are required):

```powershell
node --conditions=development --import tsx scripts/audit-event-three-step.ts --reference event-policy-two-event-neighbor-full-audit-v304 --output event-policy-three-event-action-audit-v315 --cases 100
node --conditions=development --import tsx scripts/audit-event-multi-step-upper.ts --output event-policy-recursive-directional-audit-v333 --cases 100 --method marginal --directions
node --conditions=development --import tsx scripts/probe-event-three-step.ts --source event-policy-two-event-full-suite-merged-v309 --window sharpe-up-7d-2024-11 --output event-policy-three-event-november-global-v329 --budget 64 --quantities -0.14031 --global-upper
node --conditions=development --import tsx scripts/probe-event-multi-step-upper.ts --source event-policy-three-event-november-upper-seed-v321 --output event-policy-recursive-marginal-november-fine-v326 --points 129 --method marginal
node --conditions=development --import tsx scripts/probe-event-three-step.ts --source event-policy-two-event-full-suite-merged-v309 --window regime-down-2022-06 --output event-policy-three-event-june-upper-seed-v336 --budget 64 --quantities -0.3428 --global-upper
node --conditions=development --import tsx scripts/probe-event-multi-step-upper.ts --source event-policy-three-event-june-upper-seed-v336 --output event-policy-recursive-marginal-june-final-v337 --points 257 --method marginal
```

### Faster continuations and the first complete H3 window — v339–v355

**Reduce repeated work before scaling the horizon.** The prepared H1 solver
now exposes a scalar continuation value and checks the no-trade interval
before allocating candidate orders. For each ordinary buy/sell interval,
concavity reduces the candidate set to the feasible integers surrounding its
continuous optimum. Minimum orders, caps, rare ruin and exceptional maximum
order recovery retain their original rules; singular cases use the general
solver. H2 consumes scalar values, and the compiled H2 bound supplies
directional target seeds that are evaluated under the complete original law.
The seeds never substitute a relaxed value for a feasible continuation.

V344 matches exhaustive H1 search on 3,636 cases with zero regret. V345 checks
10,854 H2 queries over 3,618 cases and three budgets; all 3,618 full-budget
queries certify with zero measured regret. The maximum full-budget gap is
0.0000015094 bp. V341 separately exercises the optional global-bound seeds.

The complete −0.14031 BTC November action falls from v329's 39.30 seconds to
v340's 18.36 seconds with bitwise-identical results and branches. Better H2
seeds slightly improve the feasible continuation in v342, then ordinary-order
interval pruning reduces it to **16.58 seconds** in v346. V346 exactly
reproduces v342's full result and branches; its feasible value is
0.0003687659029847011 and global gap against v326 is 0.000466857 bp.
In June, the complete −0.3428 BTC action falls from v336's 207.45 seconds to
91.20 seconds in v343, then **71.67 seconds** in v347. V347 exactly reproduces
v343's full result and branches. Its feasible value is 0.0013632109428665822,
with global gap 0.000512608 bp against v337. These are complete-law timings,
not sampled-outcome profiles.

**Make H3 a replayable policy with an explicit global gap.**
`prepareEventThreeStep` ranks feasible root candidates using complete-law
upper bounds, evaluates their H2 continuations, and compares the best feasible
lower value against an upper bound covering the entire root order lattice.
The root evaluation budget limits computation, not the domain of the claimed
certificate. A known policy that remains in cash through all three events can
certify without path expansion. Unsupported recovery states retain an infinite
global bound, and discrete future-order gaps remain unresolved. Replays reject
mixing this fixed-law optimizer with fitted critics or adaptive/sign forecasts.

V349, v351 and v352 each check 1,200 H3 queries against independent exhaustive
search: 100 two-state laws, two terminal contracts, three exposures and two
root budgets. V352 has 600 finite upper bounds and 26 certificates at each
budget, with zero certified regret or upper/lower violations. The other cases
correctly retain unresolved continuous-versus-discrete gaps. The focused tests
also prove that the first replayed order and its bound cannot see later prices.

**Certify holding separately from executable adjustments.** Local theory in
`docs/theory/Position management.md` explicitly requires a finite-value
comparison between holding and the minimum feasible order; a derivative alone
does not resolve the disconnected action set. The new directional root bound
intersects the continuous buy/sell domain with the exchange's minimum and
maximum constraints, while bounding holding separately. Future shadow tables
continue to relax absolute order limits and lots. A tolerance-edge regression
ensures that the upper bound retains orders admitted by the dollar tolerance;
rounding maximum notional down prematurely would incorrectly exclude them.
A singleton numerical interval remains conservatively unresolved.

**Complete July replay, then tighten only its unresolved bounds.** V348 is a
three-event prefix profile. After that profile certifies, v350 runs all 48
decisions from July 28 through August 4, 2022 with the original v10 forecast,
1× leverage, 12 bp execution cost, $5 minimum notional and the unchanged lot,
borrow and maintenance rules. It uses 129 shadow points, a 64-evaluation H2
budget and a two-candidate H3 root budget. Every chosen action evaluates all
of its forecast successors. The full replay takes **622.60 seconds** and
initially certifies 35/48 decisions; its largest global gap is 0.00324256 bp.
All 13 unresolved actions have much smaller fixed-action continuation gaps,
so more root candidates or repeated continuation evaluation is not justified.

V353 recomputes the global bounds for only those 13 saved decision states,
using 257 points and the directional order limits. It verifies the forecast
hash and retains the saved feasible lower policies, root orders and realized
metrics. All 48 certify in 14.27 seconds of additional work. V355 repeats that
refinement after the conservative exchange-tolerance fix, taking 15.58 seconds
and retaining **48/48 certificates, maximum gap 0.000994598 bp**. V355 is the
final bound artifact; v350 remains the source replay. No trades were rerun or
selected using the realized return.

| Complete July window | H2, v331 | H3, v350 + v355 bounds |
| --- | ---: | ---: |
| Return | −4.040934% | −4.350268% |
| Maximum intrabar drawdown | 10.459874% | 10.634352% |
| Executed orders, including settlement | 15 | 19 |
| Reversals | 2 | 8 |
| Fees | $109.668165 | $140.029529 |
| Borrowing | $4.062620 | $4.182638 |
| Gross long + short P&L | −$290.362657 | −$290.814642 |
| Cancellations / liquidations | 2 / 0 | 2 / 0 |
| Certified decisions | 48/48 | 48/48 |

**What changed and what should be preserved.** V354 reconciles the $30.933366
equity difference into −$0.451985 of gross P&L, −$30.361364 of additional fees
and −$0.120018 of additional borrowing. Both policies' initial entry and July
30 reversal are canceled at the next open (decisions 1 and 27). The early short
trajectory otherwise stays almost identical through decision 29. This
preserves its useful downside capture but also its losses on upward events.

The first material divergence is July 31 at 18:30 UTC: H2 reduces a long from
0.40491 to 0.15848 BTC, while H3 sells through zero to −0.02249 BTC. H3 then
holds a small short between the model's bullish leaf-2 states, with several
minimum-size adjustments, whereas H2 retains a partial long. Decisions 31–35
improve by $56.15, but decisions 37–41 worsen by $67.45. Rebuilding the long
from a short also costs more. Thus the first successful reduction should not
be discarded, and a blanket rule that preserves every H3 reversal is equally
unsupported. These segment differences describe whole account trajectories;
they are not isolated counterfactual action effects.

The frozen forecast supplies a concrete later-stage diagnosis. Excluding the
boundary-ending event, leaf 2 predicts +17.99 bp but averages −76.57 bp over
five July outcomes, with the predicted sign correct only once. Leaf 0 predicts
−7.76 bp but averages +16.56 bp over 32 outcomes, with 14 sign matches. These
small, inspected research-window summaries justify testing conditional sign
and transition forecasts after the policy objective is established; they do
not validate a new model or a sign inversion. The later structured next-second
sign family audited in v338 remains an untested candidate for that stage.

The next policy task remains deeper fixed-law coverage and convergence. H3
resets its horizon after each observed event, while each root's valuation
assumes H2 at its first successor. Its finite-horizon certificates therefore
do not certify the infinite receding policy or explain all turnover by that
horizon mismatch. Decision-price planning also remains distinct from actual
next-open fills. A complete H3 replay is now available for one of 28 windows,
with no claim of all-window H3 optimality or stationary convergence. All
**113 focused tests** and workspace typechecks pass.

Reproduction (use new output names):

```powershell
node --conditions=development --import tsx scripts/audit-event-three-step-policy.ts --source event-policy-recursive-directional-audit-v333 --output event-policy-three-event-order-limit-audit-v352 --cases 100
node --conditions=development --import tsx scripts/audit-event-policy-suite.ts --source event-policy-all-mean120-1x-cap-v10 --output event-policy-three-event-july-full-v350 --depth 3 --budget 64 --root-budget 2 --points 129 --windows sideways-churn-2022-07
node --conditions=development --import tsx scripts/refine-event-three-step-bounds.ts --source event-policy-three-event-july-full-v350 --output event-policy-three-event-july-final-bounds-v355 --points 257
node --conditions=development --import tsx scripts/compare-event-policy-depths.ts --lower event-policy-two-event-reuse-final-july-v331 --higher event-policy-three-event-july-full-v350 --bounds event-policy-three-event-july-final-bounds-v355 --output event-policy-three-event-july-comparison-v354
```

### H3 coverage, executable root bounds and continuation accuracy — v356–v375

**Measured speed improvements.** V357 profiles the already-certified
November −0.14031 BTC action. About 4.02 seconds of sampled self time appears
in the transpiler's helper-naming function; another 3.29 seconds appears in
its runtime property-setting frame. The prepared H1 solver created its
fallback closure for every account, including millions of no-trade queries.
Moving fallback, ordinary-order validation and candidate insertion helpers
outside the per-account path reduces the complete action from **16.34 to
7.51 seconds** in v358. Removing the temporary `Object.values` array from
the three-field account validation reduces it further to **5.26 seconds** in
v365. The complete values, bounds and all 3,695 branch results exactly match
v346, which took 16.58 seconds. Saved CPU profiles and reproduction checks
support these local timings; hardware-independent speedups are not claimed.

**The root bound must cover executable alternatives accurately.** V359 and
v360 expose gaps caused by continuous root orders and an infeasible holding
alternative. A short just beyond the leverage cap must trim; allowing it to
hold made the May churn decision-2 gap 0.00689 bp. The directional bound now
uses the convex hull of the ordinary executable lot interval, including
minimum order and cap endpoints. Its initial arithmetic bounds round outward,
then validate endpoints using actual fee/notional/quantity tolerances. If
endpoint arithmetic is uncertain, it retains the wider continuous bound.
Exceptional recovery still requires the existing reachability exclusion.
Holding enters the bound only when it is actually cap-feasible. A one-lot
interval is evaluated directly instead of being mistaken for an empty set.

V363 certifies all 18 decisions in the three short-window replays, including
the two February 2025 decisions that a finer grid alone failed to close.
V364 closes most gaps in the 241-decision smaller-kernel batch, but leaves six
holding decisions slightly above tolerance. Their complete holding values
already have much tighter bounds than the relaxed future-order envelope.
The optimizer therefore partitions the whole root action set into hold,
nonzero buys and nonzero sells. The maximum of the complete holding upper
value and the two nonzero-order bounds remains a global upper bound. This
closes all six remaining gaps in v368 without changing or reevaluating the
saved trades. V367 and the final v374 each check 1,200 independently enumerated
H3 problems. At each root budget, 91/600 cases certify with zero certified
regret, versus 26 before the holding partition; all 600 upper bounds are
finite. Unresolved cases remain explicit.

**Do not require unnecessary accuracy from every successor.** V366 completes
the November 2024 three-day window, then its December replay takes 210.81
seconds for decision 1 and 191.42 seconds for decision 2. The process is
identity-checked and stopped to avoid scaling that cost across the remaining
decisions. Its completed November window is retained; the partial December
trajectory is excluded from coverage. `stopped.json` records this explicitly.

The probe can now select a bounded successor range. Any offset or outcome
limit keeps the full-action value unresolved; it never reports a sampled
expectation as a complete value. V369 and v370 show cheap early and later
successor ranges. The difference is accuracy: the replay required every H2
continuation to reach half the final H3 tolerance. On the same 25 later
successor accounts, v372's 5e−8 tolerance takes **2.020 seconds**, versus
v370's **0.115 seconds** at 1e−7. Both produce identical quantities and
feasible lower values. The stricter run makes 300 action evaluations and 725
interval-bound evaluations; the sufficient-accuracy run makes 25 action
evaluations and no interval-bound evaluations.

V371 evaluates all 2,905 successors of the December entry at 1e−7 in
**2.968 seconds**. Its complete action gap is 0.000437210 bp. H3 now uses
the requested global tolerance for continuations and still certifies only
against the final whole-root global gap. It does not require every local
bound to be twice as tight. V373 reruns the stopped December window from
its original start, then completes the remaining two three-day Sharpe
windows. December's first two decisions take 6.18 and 2.94 seconds including
initial setup on the first; all **14 decisions finish in 44.25 seconds** and
certify, retaining the H2 return of +11.4502%. The partial stopped run is not
spliced into the new replay.

**Verified coverage and matched return comparison.** V375 merges only complete
windows from finished batches or explicitly stopped batches with an exact
completed-window list. It checks the current non-fit catalog, forecast hashes,
training boundaries, H2/H3 event times, numerical gaps and every refinement's
unchanged root quantity and feasible lower value. The merged parts use
successive numerical optimizer revisions under the same fixed-law objective;
each saved decision has its own certificate. They are not asserted to be
bitwise replays of the latest optimizer implementation.

| Same 16 complete windows | H1 | H2 | H3 |
| --- | ---: | ---: | ---: |
| Mean independent-window return | +1.061924% | +0.457175% | +0.760355% |
| Positive / negative / cash windows | 7 / 5 / 4 | 6 / 6 / 4 | 7 / 6 / 3 |
| Worst window | −21.907112% | −15.756277% | −15.795418% |
| Maximum intrabar drawdown | 25.722593% | 19.776520% | 19.813678% |
| Orders, including settlement | 111 | 144 | 155 |
| Fees | $1,008.820478 | $1,152.375844 | $1,183.314013 |
| Cancellations / liquidations | 3 / 0 | 5 / 0 | 4 / 0 |

All **458 H3 decisions** certify; maximum gap is **0.000994598 bp**.
Relative to H2, three windows improve, seven worsen and six are unchanged.
These means compare independent account resets over overlapping, previously
inspected research windows. They are not a compounded portfolio or a fresh
holdout, and the remaining windows must not be inferred from this subset.

**Behavior to preserve and failures to investigate.** In the February 2024
low-churn uptrend, H3 pays to enter 0.19667 BTC and holds it, while H2 stays
cash. That captures $734.04 gross less $24.83 in fees for **+7.0920%**. The
leaf's next-event forecast mean is only +4.45 bp, so the extra horizon makes
this persistent exposure worthwhile under the frozen law. This is useful
evidence for amortizing entry cost over a forecast sequence.

In the known June 2022 seven-day miss, H3 reduces early long exposure to about
0.75× and later leaves the account nearly flat rather than repeatedly
restoring H2's roughly 0.17× long. Return improves from **−7.8322% to −4.3283%**:
$329.04 better gross P&L and $21.37 less in fees offset $0.02 additional borrow.
It preserves exposure reduction during the selloff, but remains a losing
window and gives up some rebounds.

The November 2024 three-day uptrend supplies the opposite case. At November
11, 15:14 UTC, H3 sells 0.06161 BTC, roughly halving its profitable long,
after the state forecasts −3.53 bp. The next move is +130.24 bp and the rally
continues. Return falls from **+8.0577% to +4.4813%**; $358.07 of lost gross
P&L explains the deterioration, with slightly lower fees. This is a forecast
and horizon failure, unlike July's predominantly fee-driven deterioration.
The optimizer's finite-H3 certificate does not make its forecast correct.
V375's `behavior.json` records these matched trajectory differences without
calling them isolated counterfactual action effects.

The exact 12 missing H3 windows are listed in v375: the December 2021,
September 2021 and March 2023 churn windows; March 2023 up and June 2022 down
regimes; June 2022 high-churn down shape; July 2024 low-bias flat shape;
January 2024 mid-bias flat shape; and all four seven-day Sharpe windows.
Deeper/stationary convergence and decision-price versus next-open execution
remain separate open requirements. No sign-model fitting, forecast selection
or promotion occurred. All **115 focused tests** and workspace typechecks
pass; the final independent audit covers **1,200 H3 queries**.

Reproduction (new output names are required):

```powershell
node --conditions=development --import tsx scripts/audit-event-three-step-policy.ts --source event-policy-recursive-directional-audit-v333 --output event-policy-three-event-final-policy-audit-v374 --cases 100
node --conditions=development --import tsx scripts/probe-event-three-step.ts --source event-policy-two-event-full-suite-merged-v309 --window sharpe-up-3d-2023-12 --output event-policy-three-event-december-strict-tail-v372 --budget 64 --quantities 0.25317 --global-upper --offset 2000 --limit 25 --tolerance 5e-8
node --conditions=development --import tsx scripts/audit-event-policy-suite.ts --source event-policy-all-mean120-1x-cap-v10 --output event-policy-three-event-three-day-sharpe-resumed-v373 --depth 3 --budget 64 --root-budget 2 --points 129 --windows sharpe-up-3d-2023-12,sharpe-down-3d-2026-06,sharpe-down-3d-2023-03
node --conditions=development --import tsx scripts/merge-event-three-step-suite.ts --reference event-policy-two-event-full-suite-merged-v309 --three event-policy-three-event-july-full-v350,event-policy-three-event-quiet-windows-v356,event-policy-three-event-short-windows-v359,event-policy-three-event-small-kernels-v360,event-policy-three-event-three-day-sharpe-v366,event-policy-three-event-three-day-sharpe-resumed-v373 --bounds event-policy-three-event-july-final-bounds-v355,event-policy-three-event-short-root-lot-bounds-v363,event-policy-three-event-small-kernel-final-bounds-v368 --output event-policy-three-event-partial-suite-v375 --partial
```

### Coarse continuation policies and broader coverage — v376–v388

The completed checkpoint v385 covers **20/28 full non-fit windows**, with
**559/559 H3 decisions** certified at the unchanged global tolerance of
1e-7 log wealth (0.001 bp). Its maximum gap remains 0.000994598 bp. The
four seven-day Sharpe windows in v376 add 101 decisions. The original
129-point bounds certify 98; 257 points in v381 certify 100; the targeted
513-point refinement v384 closes the last November gap. These refinements
retain every saved root order, feasible lower value and backtest metric.

| Added window | H2 return | H3 return | H3 decisions | Final maximum gap, bp |
| --- | ---: | ---: | ---: | ---: |
| Uptrend, seven-day December 2023 | +23.2259% | +23.2259% | 21 | 0.000348541 |
| Uptrend, seven-day November 2024 | −12.7710% | −19.2431% | 38 | 0.000856313 |
| Downtrend, seven-day March 2023 | +12.9322% | +12.0784% | 21 | 0.000232108 |
| Downtrend, seven-day June 2026 | +4.8850% | +4.8973% | 21 | 0.000918218 |

Across the same 20-window subset, mean independent-window returns are
H1 +2.9559%, H2 +1.7793% and H3 +1.6562%. H3 improves four windows, worsens
nine and leaves seven unchanged versus H2. Seven H3 windows lose money.
These overlapping inspected windows are neither a compounded portfolio nor
new holdout evidence. Finite H3 optimality does not establish profitable
forecasts, stationary optimality, or optimal next-open execution.

The dense June profiling comparison evaluates all **4,594 coalesced
successors**, preserving the original joint law and ruin support. v377
uses continuation tolerance 1e-7 and takes 37.5818 seconds. v378 uses 1e-6
and takes 8.1648 seconds, a **4.60× reduction**. Its feasible value falls by
only 3.453e-9 log wealth. Intersecting that lower policy with the independently
saved v337 whole-action upper bound gives a **0.000547136 bp global gap**,
still below the original outer tolerance. `global-check.json` verifies the
forecast hash and account match. This is a complete-law comparison, not a
sampled or renormalized expectation.

`prepareEventThreeStep` now evaluates a candidate first with continuation
tolerance ten times the outer tolerance. It stops if the final global gap
closes. Otherwise it may repeat that candidate at the outer tolerance when
the candidate's remaining value interval could close the gap. Both complete
passes count against the same root evaluation budget. Every returned
evaluation records its continuation tolerance; duplicate root quantities
can represent different feasible continuation policies. The selected lower
value is the best actually evaluated policy. Holding upper bounds intersect
across complete passes. Neither the final tolerance nor the global action
domain is relaxed.

All **115 tests** and workspace typechecks pass. v379 independently
enumerates **1,200 H3 queries** on 100 saved small laws, with root budgets
one and four. All bounds remain valid; 91/600 queries certify under each
budget, with zero certified action regret. v380 then exercises the actual
June planner on two consecutive decisions: both certify in 20.027 seconds
including preparation, with maximum gap 0.000564773 bp. The root orders
are −0.34281 and −0.00833 BTC. This measured improvement justifies the
full June batch v383; the six other remaining windows run separately in
v382. Neither in-progress batch is counted in v385.

The merger now saves reproducible `behavior.json` diagnostics alongside
its verified coverage report. It compares requested and filled orders,
holdings, end exposures, event P&L, leaf forecast residuals, and canceled
orders. It reconciles terminal settlement separately and excludes the
window-boundary event from forecast residuals. Differences describe whole
account trajectories, not same-account causal effects of an isolated action.

The November failure is materially different from July's added turnover:
H3 loses **$658.91 more gross P&L**, saves **$13.16 in fees**, and pays
**$1.45 more borrowing**, explaining its $647.20 lower ending wealth.
It initially shorts 0.14026 BTC versus H2's 0.12423. Its bearish leaf 0
predicts −10.3620 bp but averages **+103.1328 bp over ten completed events**,
matching the realized sign only once. At the first bullish state it retains
0.09897 BTC short versus H2's 0.04299; this helps during the immediate
decline, then hurts during the subsequent rally. Longer planning intensifies
the exposure selected under this misspecified forecast. Correcting the
optimizer's small numerical residual cannot explain these realized losses.

Preserve the December seven-day behavior: H1, H2 and H3 reproduce the same
orders and +23.2259% return. March's H3 trims short exposure more aggressively
at a positive forecast state, saving on one rebound but missing subsequent
declines; it loses $71.33 gross relative to H2 and adds $14.41 fees. June
2026's small gain is partly associated with a canceled next-open add-short
order, so it is not evidence of a better forecast. The existing later
structured next-second sign models remain a candidate for the forecast
phase; these observations do not justify inserting their August 2026
checkpoints into earlier historical windows or changing the frozen law
during this policy audit.

The subsequent six-window batch v382 completes another **296 decisions**
in 272.90 seconds. Six December 2021 bounds need refinement: v386's
257-point envelope leaves them unresolved, while v387's 513-point envelope
closes all six without changing orders or feasible values. The full merged
checkpoint **v388 covers 26/28 windows and certifies 855/855 decisions**,
with maximum gap 0.000994598 bp. Only the two dense June windows remain;
their running batch v383 is excluded until completion.

| Further window | H2 return | H3 return | Decisions |
| --- | ---: | ---: | ---: |
| Choppy December 2021 | +0.0663% | +0.9095% | 58 |
| Choppy September 2021 | +0.8201% | +0.9730% | 57 |
| Choppy March 2023 | −1.9928% | −0.0236% | 48 |
| Uptrend March 2023 | +7.3504% | +20.8317% | 90 |
| Low-bias July 2024 | +0.0084% | +0.4632% | 19 |
| Mid-bias January 2024 | +1.6425% | +1.6414% | 24 |

On all 26 covered windows, H3's mean return is **+2.2277%** versus H2
**+1.6724%**, with nine improvements, ten declines and seven unchanged.
H3 has 15 positive, eight negative and three cash windows. Its 271 orders
cost $2,074.08 in fees, versus H2's 260 orders and $2,211.31; both have
13 cancellations and no liquidation. These remain independent account
resets over inspected windows.

March's uptrend supplies the largest new gain: H3 loses less on bearish
states during the rally, improving gross P&L by **$1,209.71**, saving
**$135.28 fees** and **$3.14 borrowing**. On March 16 at 20:48 UTC,
it holds only 0.01139 BTC short versus H2's 0.44295 BTC short; the next
event rises 210.20 bp. That smaller short sacrifices profit on intervening
declines, so the entire trajectory, including losses, remains recorded.
December 2021's improvement instead combines smaller early long exposure
with lower fees. These concrete behaviors should survive subsequent model
and horizon changes when supported by the conditional forecast; the
realized favorable direction cannot be used as a decision-time gate.

## Diagnosis

The retained depth-eight fitted-value branch earns positive returns in all three May
calibration origins, but remains cash on the final May inspector window.
Depth eight improves the active origin to +39.3582% while worsening the first
origin relative to depth six. Retaining an earlier fit produces only a small
final-window gain with materially larger drawdown. Under the revised objective,
the immediate unresolved issue is conditional multi-event policy optimality.
The one-event sizing gap is closed, the marked-versus-liquidated terminal
objective is explicit, and finite two-event actions now have numerical bounds
throughout one full prior origin and all 28 non-fit inspector windows.
Global three-event bounds now certify two diagnostic initial states; full
H3 coverage, stationary continuation and execution consistency remain open.
The all-window result uses the older complete v10 forecast family;
it does not substitute those models for the newer three-window forecasts.

March transfer rejects the deviation setting and additional fitted depth.
The new fitted branch stays in cash on March's final window; the retained
older policy's small positive return comes with materially larger drawdown.
Correcting final-event handling leaves those economic results unchanged.

The exact-second input trial preserves one useful rebound hold but also
suppresses a profitable entry and fails to improve the final windows. Its
matched depth-three policy is weaker overall than the existing control. The
separate-critic diagnosis likewise shows that reducing optimism can remove
profitable opportunities. These results prioritize conditional persistence and
holding/exit values over another blanket sign or pessimism gate.

The subsequent horizon audit does not find persistent gains from the new
second-level inputs. Observed-path evaluation improves May at depth two but
produces wrong reversals in March and stays cash in the final May window.
This local improvement is retained as a research result, not a reliability
claim or a reason to expand computation. Exact attribution then identifies
amplified signed extrapolation. Centering the spread fixes March's wrong
reversals but loses May's gains. Holding out each evaluated time block removes
the March losses but retains the wrong first May short while excluding useful
May entries. Training-data reuse is a measured issue, not a sufficient
explanation or a complete correction. The next attribution should compare the
specific target changes behind the opposite action-value shifts.

That attribution shows broad block contributions, not a single bad period.
Using shorter held-out blocks yields only a small May gain and loses March's
remaining trade. A positive neighbor law then removes signed extrapolation but
does not establish a useful value forecast. Its apparently strong sign accuracy
almost duplicates the cheap opposite-clock-state baseline. Future forecast
gates must include that baseline and test magnitude-weighted utility, before
any deeper rollout or larger learner is justified.

What worked:

- The event target produced sparse, fee-covering trades where the 20-bps target
  almost always selected cash.
- The bearish June 2022 state entered short at about 4× in the 5× run before
  three consecutive moves of -122, -126, and -138 bps. The model's leaf mean
  was only -18.8 bps, so the profit was caused by correctly persistent
  direction rather than foreknowledge of the realized move sizes.
- The cash action prevented losses in many windows with negative expectation
  skill. There were no liquidations or value-grid boundary decisions in the
  inspected profitable crash replay.
- Deeper Bellman recursion was useful in a few persistent regimes, while
  validation usually chose depth 1 or cash. That agrees with short-rollout
  model-based RL rather than assuming unlimited reliable recursion.

What failed:

- Directional expectation skill is small and unstable. Test MSE skill often
  changed sign across windows even when duration/class NLL improved.
- The 120-bps 5× policy held its bearish state through +208 to +236 bps
  counter-moves. It allowed exposure to drift above 5× while no new order was
  placed, producing the 48% drawdown. Lower target leverage improved this;
  event-boundary cap restoration is now enforced, but the model still lacks
  a distinct rebound/liquidity state and intraperiod risk-control decisions.
- A positive 21-day calibration episode did not reliably transfer into the
  next 3–7 days. This caused trades in four losing 5× windows and five losing
  1× windows. Optimizing the cash gate further on these known windows would be
  test-set overfitting.
- The training origins overlap because they are sampled on a fixed stride,
  while live decisions begin at preceding event endpoints. The conditional
  state is only approximately Markov; time since the preceding significant
  event and nested pullback state are omitted.
- The hidden-model branch fixes endpoint alignment and preserves posterior
  memory, but its NLL improvements mostly describe activity regimes. A model
  whose every conditional mean has the same sign still holds through adverse
  reversals. Removing historical drift alone can remove its entire trading edge.
- A model selected for mean skill can have worse likelihood, while a
  likelihood-selected model can predict almost zero mean. Prior density
  experiments documented the same mismatch between NLL and trading-relevant
  expectation checkpoints.

## Relevant prior work and next justified experiment

The current priority is conditional multi-event optimality, as stated in
v267 onward. The following notes
record the motivation for the earlier endpoint and one-second trials; they do
not supersede the later failed transfer checks.

The repository's rolling intraday study already found no robust signed-return
forecast at 15m–1h, while scale and activity were predictable. Its production
density study found useful one-step expectation skill but degrading open-loop
paths. Those observations explain both the event-duration skill and the weak
directional transfer here. The existing recommended feature basis identifies
one-second EMA slope/acceleration, active-run state, trade flow, order-book
imbalance, and cross-asset derivatives as the next causal inputs; the current
OHLCV tree uses only the long-history subset.

The next useful direction is a short branched rollout trained on a pooled
endpoint dataset with enough independent regimes, then a frozen calibration
rule tested at 1×. There is too little cached history before the earliest
2021 inspector window to assume a large global pre-suite model is available.
A new later calendar block is needed for a final untouched confirmation.
Longer Bellman depths, denser 5× grids, and more cash-gate tuning are not
justified by these results.

An immediate endpoint-chain screen was stopped early rather than scaled: two of
its first three completed choppy windows lost money (including -5.96%), one
returned +2.85%, and several purged 21-day calibration spans contained fewer
than 25 independent events. Endpoint alignment is still the right transition
contract, but this simple tree does not have enough independent large-event
samples per local window. The next model should pool a global endpoint dataset
and condition on regime, rather than shrinking local samples and pretending
they are sufficient.

A second immediate screen added one-minute analogues of the repository's
one-second EMA, RSI, range, and calendar features. The five-window diagnostic
looked safer, but the scaled run immediately lost on its first three active
choppy windows (-2.55%, -3.95%, and -9.52%). It was stopped after five windows
and the feature branch was removed. The documented one-second information does
not safely transfer by aggregating indicator formulas to one minute; the richer
branch must consume the actual one-second feature basis and retain its original
availability contract.

The actual bounded one-second basis is now supported: range, 60-second active
count, previous one-second return, 60-second realized volatility, and close
location. `data/runtime-cache/global-feature-basis` contains 2,629,380 rows
from July 2021 through July 2026, at one rotating second per minute. Features
are joined by `originTime + 1000 <= decisionTime`, must be less than one minute
old, and carry a source fingerprint. The five coordinates have at most 60s
of input support, so the existing one-day purge also covers them. No cached
target, future row, recursive indicator, or pre-trained density checkpoint is
used. The v18 five-case screen is split across two artifact directories because
the uptrend ID was initially mistyped; unknown requested IDs now fail loudly.

This matches [Model-Based Policy Optimization](https://proceedings.neurips.cc/paper/2019/hash/5faf461eff3099671ad63c6f3f094f7f-Abstract.html),
which found short branched model rollouts useful for limiting accumulated model
bias, and [Multi-Period Trading via Convex Optimization](https://stanford.edu/~boyd/papers/cvx_portfolio.html),
which plans multiple periods but executes only the first trade before
re-optimization.

## Reproduction

```bash
npm run test:event-policy
npm run research:event-policy -- --windows all --threshold-bps 120 --max-candles 1440 --criterion mean --tree-depths 2,4,6 --prior 100 --stride 30 --bellman-depth 4 --leverage 1 --output event-policy-local-reproduction
npm run research:event-policy:replay -- --source event-policy-all-mean120-1x-v5 --output event-policy-frozen-local-reproduction
```

Each output directory contains its exact config, source snapshot, frozen model
and value tables, test trade trace, per-window results, and summary. Reusing an
output name with changed code or configuration fails instead of mixing runs.
These commands use current code; exact historical reproduction requires the
corresponding saved `sources.json`. Add `--run-clock`, `--second-basis`,
`--sampling chain`, `--honesty-fraction 0.5`, or `--invert-augment` only for the
specific diagnostic branch being investigated. Inverse augmentation and cached
one-second features cannot be combined without rematerializing inverse inputs.
`--online-scale` enables the rejected signed rolling-correction research branch;
`--risk-penalty` sets the pre-window selection penalty. Frozen-model replay
refuses an incomplete source run unless completed windows are requested
explicitly. Earlier v23 replay B contains only the first two completed v22
windows because it was started before v22 finished; this behavior now fails
loudly instead of silently omitting pending source windows.
Use `--learner projection` with `--ridge-penalties` for the ridge state model,
or `--learner boost` with `--boost-iterations` and `--boost-rate` for the
boosted state model. `--projection-cells` controls the maximum number of score
states for either learner. `--refit-latest` retains a separate selection model
and final pre-test refit; it cannot currently be combined with online scaling
or mean calibration. Inverse latest-refit augmentation is supported for hidden
sequences. Use `--learner hidden --sampling chain --hidden-states 2,3
--hidden-resolution 4 --hidden-smoothing 10` for that learner; `--invert-augment`
adds the separate reciprocal-chart sequences. `--training-isolation causal`
allows all strictly preceding prices; the default `suite` keeps all inspector
windows out of estimation and calibration.
Use `--reversal-clock` for confirmed reversals and optionally `--progress-bps
60` for within-run progress boundaries. `--run-basis` adds the same bounded run
features to the ordinary barrier clock for a separate feature-only comparison.
`--run-symmetry` fits the canonical run law and requires the reversal clock and
tree learner; independent inverse augmentation and mean rescaling cannot be
combined with its exact reflection contract.
Use `--run-direction-prior 100` instead to estimate direction-specific laws
shrunk toward the shared canonical law; zero requests no shrinkage for observed
directions. `--honesty-fraction 0.5` separates partition and law-estimation data.
`--model-selection utility` ranks all model/depth candidates using calibration
policy outcomes. `npm run research:event-policy:audit -- --source <run>
--windows <ids> --output <new-name>` is explicitly hindsight-only.
Frozen replay accepts `--bellman-depth 64 --evaluation-depths
1,2,3,4,5,6,7,8,16,32,64` to build every recursion level while scoring only the
listed depths. Keep the original candidate depths when testing added horizons.
The paired operator benchmark is reproduced with:

```bash
node --conditions=development --import tsx scripts/benchmark-event-operator.ts --source event-policy-run-symmetry-screen-v45 --window sideways-churn-2023-03 --output event-policy-operator-local.json
```

The completed-event forecast screen is reproduced with:

```bash
node --conditions=development --import tsx scripts/screen-event-recent-law.ts --source event-policy-run-conditioned-screen-v51 --output event-policy-recent-law-local
```

Add `--shared-states` to include parent-branch and run-direction sharing. The
bounded policy replay uses the forecast-selected update rule and reselects depth
and the cash gate only on preceding calibration:

```bash
node --conditions=development --import tsx scripts/replay-event-recent-law.ts --forecast event-policy-shared-recent-law-forecast-v55 --windows sideways-churn-2022-05 --refresh-events 32 --output event-policy-shared-law-local
```

The separate sign head and bounded Bellman improvement are reproduced with:

```bash
node --conditions=development --import tsx scripts/screen-event-sign.ts --source event-policy-run-conditioned-screen-v51 --output event-sign-forecast-local
node --conditions=development --import tsx scripts/replay-event-sign.ts --forecast event-sign-forecast-local --output event-sign-replay-local
```

Add `--control` to the replay command, with a new output name, to keep the
original sign probabilities while exercising the same conditional operators.

Use `--size-regimes --fast-volatility-head` on the forecast screen to compare
size-conditioned heads and their short-volatility input basis. Its output can
feed the same replay command. To screen rolling gate calibration while freezing
those selected heads, use:

```bash
node --conditions=development --import tsx scripts/screen-event-sign.ts --source event-policy-run-conditioned-screen-v51 --size-regimes --gate-head-source event-policy-size-sign-fast-volatility-v61 --output event-size-gate-local
```

Rolling-gate output is forecast-only and is rejected by the policy replay.

Completed-event context and the final utility comparison are reproduced with:

```bash
node --conditions=development --import tsx scripts/screen-event-sign.ts --source event-policy-run-conditioned-screen-v51 --size-regimes --history-head-source event-policy-size-sign-fast-volatility-v61 --include-gate-history --output event-history-forecast-local
node --conditions=development --import tsx scripts/replay-event-sign.ts --forecast event-history-forecast-local --project-continuation --compare-base --compare-continuations --output event-history-policy-local
```

The action-value audit and bounded follow-up experiments are reproduced with:

```bash
node --conditions=development --import tsx scripts/audit-event-action-values.ts --source event-policy-joint-utility-selection-v70 --windows sideways-churn-2023-03 --compare-refits --output event-value-audit-local
node --conditions=development --import tsx scripts/screen-event-value-head.ts --forecast event-policy-gate-history-screen-v66 --planner-source event-policy-joint-utility-selection-v70 --windows sideways-churn-2023-03 --output event-value-head-local
node --conditions=development --import tsx scripts/replay-event-sign.ts --forecast event-policy-gate-history-screen-v66 --windows sideways-churn-2023-03 --project-continuation --compare-base --compare-continuations --compare-head-lag --output event-head-age-local
```

Rolling refit selection and the conditional-severity audit:

```bash
node --conditions=development --import tsx scripts/research-event-refits.ts --forecast event-policy-gate-history-screen-v66 --windows sideways-churn-2023-03 --output event-rolling-refit-local
node --conditions=development --import tsx scripts/research-event-refits.ts --forecast event-policy-gate-history-screen-v66 --windows sideways-churn-2022-05,shape-flat-mid-bias-2024-01 --output event-rolling-refit-more-local
node --conditions=development --import tsx scripts/audit-event-severity.ts --forecast event-policy-gate-history-screen-v66 --windows shape-flat-mid-bias-2024-01 --output event-severity-audit-local
```

Conditional severity, economic comparison, unchanged-probability control and
the diagnostic family audit:

```bash
node --conditions=development --import tsx scripts/screen-event-severity.ts --forecast event-policy-gate-history-screen-v66 --windows shape-flat-mid-bias-2024-01 --output event-severity-screen-local
node --conditions=development --import tsx scripts/replay-event-severity.ts --forecast event-severity-screen-local --incumbent event-policy-joint-utility-selection-v70 --output event-severity-replay-local
node --conditions=development --import tsx scripts/replay-event-severity.ts --forecast event-severity-screen-local --incumbent event-policy-joint-utility-selection-v70 --output event-severity-control-local --control
node --conditions=development --import tsx scripts/audit-event-severity-policy.ts --source event-severity-replay-local --output event-severity-policy-audit-local
node --conditions=development --import tsx scripts/replay-event-components.ts --source event-policy-joint-utility-selection-v70 --output event-components-local
node --conditions=development --import tsx scripts/audit-event-severity.ts --forecast event-policy-gate-history-screen-v66 --windows shape-flat-mid-bias-2024-01 --output event-volatility-state-audit-local
```

Pooled head and joint-volatility experiments, reusing saved rolling-origin fits:

```bash
node --conditions=development --import tsx scripts/screen-event-pooling.ts --source event-policy-rolling-refit-more-v76 --window shape-flat-mid-bias-2024-01 --output event-pooling-local
node --conditions=development --import tsx scripts/replay-event-pooling.ts --forecast event-pooling-local --output event-pooling-policy-local
node --conditions=development --import tsx scripts/screen-event-volatility-law.ts --forecast event-pooling-local --output event-joint-volatility-local
node --conditions=development --import tsx scripts/replay-event-volatility-law.ts --forecast event-joint-volatility-local --output event-joint-policy-local
node --conditions=development --import tsx scripts/audit-event-volatility-policy.ts --source event-joint-policy-local --output event-joint-active-audit-local
node --conditions=development --import tsx scripts/audit-event-volatility-policy.ts --source event-joint-policy-local --output event-joint-action-audit-local --actions
```

Complete-prior reference and compact representation, using existing saved fits:

```bash
node --conditions=development --import tsx scripts/audit-event-prior.ts --forecast event-policy-joint-volatility-hybrid-v90 --output event-prior-local
node --conditions=development --import tsx scripts/replay-event-prior.ts --audit event-prior-local --policy event-policy-joint-volatility-policy-v91 --output event-prior-policy-local
node --conditions=development --import tsx scripts/compress-event-prior.ts --audit event-prior-local --output event-quadrature-local
node --conditions=development --import tsx scripts/replay-event-prior.ts --audit event-quadrature-local --policy event-policy-joint-volatility-policy-v91 --model hybrid-quadrature --output event-quadrature-policy-local
```

## Artifacts

- `packages/bot-algo/src/event-distribution.ts`
- `packages/bot-algo/src/event-hidden.ts`
- `packages/bot-algo/src/event-run-model.ts`
- `packages/bot-algo/src/event-recent-law.ts`
- `packages/bot-algo/src/event-sign.ts`
- `packages/bot-algo/src/event-size-sign.ts`
- `packages/bot-algo/src/event-completed-history.ts`
- `packages/bot-algo/src/event-log-policy.ts`
- `packages/bot-algo/src/event-value-head.ts`
- `packages/bot-algo/src/event-severity.ts`
- `packages/bot-algo/src/event-volatility-law.ts`
- `packages/bot-algo/src/event-quadrature.ts`
- `scripts/research-event-policy.ts`
- `scripts/replay-event-policy.ts`
- `scripts/audit-event-policy.ts`
- `scripts/benchmark-event-operator.ts`
- `scripts/screen-event-recent-law.ts`
- `scripts/replay-event-recent-law.ts`
- `scripts/screen-event-sign.ts`
- `scripts/replay-event-sign.ts`
- `scripts/audit-event-action-values.ts`
- `scripts/screen-event-value-head.ts`
- `scripts/research-event-refits.ts`
- `scripts/audit-event-severity.ts`
- `scripts/screen-event-severity.ts`
- `scripts/replay-event-severity.ts`
- `scripts/audit-event-severity-policy.ts`
- `scripts/replay-event-components.ts`
- `scripts/screen-event-pooling.ts`
- `scripts/replay-event-pooling.ts`
- `scripts/screen-event-volatility-law.ts`
- `scripts/replay-event-volatility-law.ts`
- `scripts/audit-event-volatility-policy.ts`
- `scripts/audit-event-prior.ts`
- `scripts/replay-event-prior.ts`
- `scripts/compress-event-prior.ts`
- `scripts/event-futures-basis.ts`
- `scripts/screen-event-futures-sign.ts`
- `scripts/replay-event-futures-sign.ts`
- `scripts/audit-event-continuation.ts`
- `scripts/screen-event-martingale.ts`
- `scripts/event-second-basis.ts`
- `data/benchmarks/event-policy-all-mean60-v3/summary.json`
- `data/benchmarks/event-policy-all-mean120-v3/summary.json`
- `data/benchmarks/event-policy-all-mean120-1x-v5/summary.json`
- `data/benchmarks/event-policy-frozen-cap-v13/summary.json`
- `data/benchmarks/event-policy-second-basis-screen-v18/summary.json`
- `data/benchmarks/event-policy-second-basis-control-v18/summary.json`
- `data/benchmarks/event-policy-second-basis-year-screen-v19/summary.json`
- `data/benchmarks/event-policy-online-scale-screen-v20/summary.json`
- `data/benchmarks/event-policy-online-affine-screen-v21/summary.json`
- `data/benchmarks/event-policy-online-affine-more-v22/summary.json`
- `data/benchmarks/event-policy-all-online-affine-risk35-v25/summary.json` (stopped after two windows)
- `data/benchmarks/event-policy-online-affine-completed-only-v26/summary.json`
- `data/benchmarks/event-policy-projection-screen-v27/summary.json`
- `data/benchmarks/event-policy-boost-ties-fixed-v32/summary.json`
- `data/benchmarks/event-policy-boost-latest-refit-v33/summary.json`
- `data/benchmarks/event-policy-refit-replay-check-v34/summary.json`
- `data/benchmarks/event-policy-causal-refit-screen-v35/summary.json`
- `data/benchmarks/event-policy-hidden-sequence-screen-v36/summary.json`
- `data/benchmarks/event-policy-hidden-mean-screen-v37/summary.json`
- `data/benchmarks/event-policy-hidden-inverse-screen-v38/summary.json`
- `data/benchmarks/event-policy-hidden-replay-check-v39/summary.json`
- `data/benchmarks/event-policy-directional-change-screen-v40/summary.json`
- `data/benchmarks/event-policy-directional-utility-screen-v41/summary.json`
- `data/benchmarks/event-policy-directional-no-refit-v42/summary.json`
- `data/benchmarks/event-policy-run-progress-screen-v43/summary.json`
- `data/benchmarks/event-policy-state-law-audit-v44/summary.json` (hindsight only)
- `data/benchmarks/event-policy-run-symmetry-screen-v45/summary.json`
- `data/benchmarks/event-policy-operator-paired-v47.json`
- `data/benchmarks/event-policy-symmetric-horizon-v48/summary.json` (March omits the original winning depth)
- `data/benchmarks/event-policy-symmetric-horizon-complete-grid-v49/summary.json` (corrected March comparison)
- `data/benchmarks/event-policy-sparse-active-replay-v50/summary.json`
- `data/benchmarks/event-policy-run-conditioned-screen-v51/summary.json`
- `data/benchmarks/event-policy-run-conditioned-honest-v52/summary.json`
- `data/benchmarks/event-policy-run-conditioned-replay-v53/summary.json`
- `data/benchmarks/event-policy-run-conditioned-replay-v53/reproduction-check.json`
- `data/benchmarks/event-policy-recent-law-forecast-v54/summary.json`
- `data/benchmarks/event-policy-shared-recent-law-forecast-v55/summary.json`
- `data/benchmarks/event-policy-shared-recent-law-replay-v56/summary.json`
- `data/benchmarks/event-policy-sign-head-screen-v57/summary.json`
- `data/benchmarks/event-policy-sign-lookahead-replay-v58/summary.json`
- `data/benchmarks/event-policy-sign-lookahead-replay-v58/january-decision-audit.json`
- `data/benchmarks/event-policy-sign-lookahead-control-v59/reproduction-check.json`
- `data/benchmarks/event-policy-size-sign-screen-v60/summary.json`
- `data/benchmarks/event-policy-size-sign-fast-volatility-v61/summary.json`
- `data/benchmarks/event-policy-size-sign-lookahead-v62/summary.json`
- `data/benchmarks/event-policy-size-gate-calibration-v63/summary.json`
- `data/benchmarks/event-policy-size-sign-control-v64/reproduction-check.json`
- `data/benchmarks/event-policy-completed-history-screen-v65/summary.json`
- `data/benchmarks/event-policy-gate-history-screen-v66/summary.json`
- `data/benchmarks/event-policy-gate-history-replay-v67/summary.json`
- `data/benchmarks/event-policy-gate-history-replay-v67/forecast-replay-check.json`
- `data/benchmarks/event-policy-gate-history-projected-v68/summary.json`
- `data/benchmarks/event-policy-utility-head-selection-v69/summary.json`
- `data/benchmarks/event-policy-joint-utility-selection-v70/summary.json`
- `data/benchmarks/event-policy-action-value-audit-v71/summary.json`
- `data/benchmarks/event-policy-value-head-march-v72/summary.json`
- `data/benchmarks/event-policy-refit-action-audit-v73/summary.json`
- `data/benchmarks/event-policy-head-age-selection-v74/summary.json`
- `data/benchmarks/event-policy-rolling-refit-march-v75/summary.json`
- `data/benchmarks/event-policy-rolling-refit-march-v75/reproduction-check.json`
- `data/benchmarks/event-policy-rolling-refit-more-v76/summary.json`
- `data/benchmarks/event-policy-rolling-refit-more-v76/reproduction-check.json`
- `data/benchmarks/event-policy-severity-audit-v77/summary.json`
- `data/benchmarks/event-policy-severity-screen-v78/summary.json`
- `data/benchmarks/event-policy-severity-replay-v79/summary.json`
- `data/benchmarks/event-policy-severity-control-v80/summary.json`
- `data/benchmarks/event-policy-severity-policy-audit-v81/summary.json`
- `data/benchmarks/event-policy-component-selection-v82/summary.json`
- `data/benchmarks/event-policy-volatility-state-audit-v83/summary.json`
- `data/benchmarks/event-policy-pooled-heads-january-v84/summary.json`
- `data/benchmarks/event-policy-pooled-gate-january-v85/summary.json`
- `data/benchmarks/event-policy-pooled-policy-january-v86/summary.json`
- `data/benchmarks/event-policy-joint-volatility-forecast-v87/summary.json` (prior compression lost observed support; superseded)
- `data/benchmarks/event-policy-joint-volatility-support-v88/summary.json`
- `data/benchmarks/event-policy-joint-volatility-backoff-v89/summary.json`
- `data/benchmarks/event-policy-joint-volatility-hybrid-v90/summary.json`
- `data/benchmarks/event-policy-joint-volatility-policy-v91/summary.json`
- `data/benchmarks/event-policy-joint-volatility-active-audit-v92/summary.json`
- `data/benchmarks/event-policy-pooled-heads-may-v93/summary.json`
- `data/benchmarks/event-policy-pooled-heads-march-v94/summary.json`
- `data/benchmarks/event-policy-joint-volatility-may-v95/` (failed provenance lookup; superseded)
- `data/benchmarks/event-policy-joint-volatility-march-v96/summary.json`
- `data/benchmarks/event-policy-joint-volatility-march-policy-v97/summary.json`
- `data/benchmarks/event-policy-joint-volatility-may-fixed-v98/summary.json`
- `data/benchmarks/event-policy-joint-volatility-provenance-control-v99/reproduction-check.json`
- `data/benchmarks/event-policy-joint-volatility-may-policy-v100/summary.json`
- `data/benchmarks/event-policy-joint-volatility-march-audit-v101/summary.json`
- `data/benchmarks/event-policy-joint-volatility-may-audit-v102/summary.json`
- `data/benchmarks/event-policy-joint-volatility-action-audit-v103/summary.json`
- `data/benchmarks/event-policy-joint-volatility-may-actions-v104/summary.json`
- `data/benchmarks/event-policy-prior-audit-january-v105/summary.json`
- `data/benchmarks/event-policy-prior-audit-may-v106/summary.json`
- `data/benchmarks/event-policy-prior-replay-january-v107/summary.json`
- `data/benchmarks/event-policy-prior-replay-may-v108/summary.json`
- `data/benchmarks/event-policy-quadrature-january-v109/summary.json` (excursion-only pinning; superseded)
- `data/benchmarks/event-policy-quadrature-may-v110/summary.json` (excursion-only pinning; superseded)
- `data/benchmarks/event-policy-quadrature-replay-january-v111/summary.json` (superseded by frontier version)
- `data/benchmarks/event-policy-quadrature-replay-may-v112/summary.json` (superseded by frontier version)
- `data/benchmarks/event-policy-quadrature-frontier-january-v113/summary.json`
- `data/benchmarks/event-policy-quadrature-frontier-may-v114/summary.json`
- `data/benchmarks/event-policy-quadrature-frontier-replay-january-v115/summary.json`
- `data/benchmarks/event-policy-quadrature-frontier-replay-january-v115/reproduction-check.json`
- `data/benchmarks/event-policy-quadrature-frontier-replay-may-v116/summary.json`
- `data/benchmarks/event-policy-quadrature-frontier-replay-may-v116/reproduction-check.json`
- `data/benchmarks/event-policy-quadrature-forecast-january-v117/summary.json`
- `data/benchmarks/event-policy-quadrature-forecast-january-v117/reproduction-check.json`
- `data/benchmarks/event-policy-quadrature-forecast-may-v118/summary.json`
- `data/benchmarks/event-policy-quadrature-forecast-may-v118/reproduction-check.json`
- `data/benchmarks/event-policy-quadrature-forecast-march-v119/summary.json`
- `data/benchmarks/event-policy-quadrature-policy-january-v120/summary.json`
- `data/benchmarks/event-policy-quadrature-policy-may-v121/summary.json`
- `data/benchmarks/event-policy-quadrature-policy-march-v122/summary.json`
- `data/benchmarks/event-policy-futures-sign-first-v123/summary.json`
- `data/benchmarks/event-policy-futures-sign-first-v123/paired-day-audit.json`
- `data/benchmarks/event-policy-futures-sign-second-v124/summary.json`
- `data/benchmarks/event-policy-futures-sign-third-v125/summary.json`
- `data/benchmarks/event-policy-futures-policy-may-v126/summary.json`
- `data/benchmarks/event-policy-futures-policy-control-v127/reproduction-check.json`
- `data/benchmarks/event-policy-futures-policy-control-v127/sign-magnitude-diagnosis.json`
- `data/benchmarks/event-policy-futures-size-first-v128/summary.json`
- `data/benchmarks/event-policy-futures-size-second-v129/summary.json`
- `data/benchmarks/event-policy-futures-size-third-v130/summary.json`
- `data/benchmarks/event-policy-futures-sign-control-v131/reproduction-check.json`
- `data/benchmarks/event-policy-futures-size-policy-v132/summary.json`
- `data/benchmarks/event-policy-futures-projected-policy-v133/summary.json`
- `data/benchmarks/event-policy-futures-projected-policy-v133/reproduction-check.json`
- `data/benchmarks/event-policy-futures-continuation-audit-v134/summary.json`
- `data/benchmarks/event-policy-weighted-sign-first-v135/summary.json`
- `data/benchmarks/event-policy-weighted-sign-control-v136/reproduction-check.json`
- `data/benchmarks/event-policy-weighted-sign-second-v137/summary.json`
- `data/benchmarks/event-policy-weighted-sign-third-v138/summary.json`
- `data/benchmarks/event-policy-weighted-sign-third-v138/paired-objective-comparison.json`
- `data/benchmarks/event-policy-martingale-sign-v139/summary.json`
- `data/benchmarks/event-policy-martingale-sign-v139/paired-null-audit.json`
- `packages/bot-algo/src/event-fitted-value.ts`
- `scripts/screen-event-fitted-value.ts`
- `scripts/replay-event-fitted-value.ts`
- `scripts/audit-event-fitted-value.ts`
- `data/benchmarks/event-policy-fitted-value-one-v140/summary.json`
- `data/benchmarks/event-policy-fitted-value-one-v140/calibration-trade-audit.json`
- `data/benchmarks/event-policy-fitted-value-two-v141/summary.json`
- `data/benchmarks/event-policy-fitted-value-two-v141/cash-lower-bound-audit.json`
- `data/benchmarks/event-policy-fitted-value-cash-bound-v142/summary.json`
- `data/benchmarks/event-policy-fitted-value-cash-bound-v142/reproduction-check.json`
- `data/benchmarks/event-policy-fitted-value-four-v143/summary.json`
- `data/benchmarks/event-policy-fitted-value-policy-v144/summary.json`
- `data/benchmarks/event-policy-fitted-value-policy-v144/reproduction-check.json`
- `data/benchmarks/event-policy-fitted-value-audit-v145/summary.json`
- `data/benchmarks/event-policy-fitted-value-audit-v145/round-trip-audit.json`
- `data/benchmarks/event-policy-fitted-value-audit-v145/reproduction-check.json`
- `packages/bot-algo/src/event-value-boost.ts`
- `scripts/event-fitted-settings.ts`
- `data/benchmarks/event-policy-fitted-value-boost-screen-v146/summary.json`
- `data/benchmarks/event-policy-fitted-value-boost-screen-v146/reproduction-check.json`
- `data/benchmarks/event-policy-fitted-value-basis-deviation-v147/summary.json`
- `data/benchmarks/event-policy-fitted-value-deviation-cohort-v148/summary.json`
- `data/benchmarks/event-policy-fitted-value-deviation-cohort-v148/reproduction-check.json`
- `data/benchmarks/event-policy-fitted-value-deviation-cohort-v148/paired-day-audit.json`
- `data/benchmarks/event-policy-fitted-value-deviation-three-v149/summary.json`
- `data/benchmarks/event-policy-fitted-value-deviation-policy-v150/summary.json`
- `data/benchmarks/event-policy-fitted-value-deviation-policy-v150/reproduction-check.json`
- `data/benchmarks/event-policy-fitted-value-deviation-audit-v151/summary.json`
- `data/benchmarks/event-policy-fitted-value-deviation-audit-v151/round-trip-audit.json`
- `data/benchmarks/event-policy-fitted-value-deviation-audit-v151/reproduction-check.json`
- `scripts/audit-event-fitted-horizon.ts`
- `data/benchmarks/event-policy-fitted-value-deviation-six-v152/summary.json`
- `data/benchmarks/event-policy-fitted-value-horizon-audit-v153/summary.json`
- `data/benchmarks/event-policy-fitted-value-six-policy-v154/summary.json`
- `data/benchmarks/event-policy-fitted-value-deviation-eight-v155/summary.json`
- `data/benchmarks/event-policy-fitted-value-deviation-eight-v155/reproduction-check.json`
- `data/benchmarks/event-policy-fitted-value-six-audit-v156/summary.json`
- `data/benchmarks/event-policy-fitted-value-resume-check-v157/check.ts`
- `data/benchmarks/event-policy-fitted-value-resume-check-v157/summary.json`
- `data/benchmarks/event-policy-fitted-value-eight-policy-v158/summary.json`
- `data/benchmarks/event-policy-march-futures-backfill-v159/config.json`
- `data/benchmarks/event-policy-march-futures-backfill-v159/summary.json`
- `data/benchmarks/event-policy-fitted-value-march-screen-v160/summary.json` (superseded forecast cohort)
- `data/benchmarks/event-policy-terminal-boundary-check-v161/check.ts`
- `data/benchmarks/event-policy-terminal-boundary-check-v161/summary.json`
- `data/benchmarks/event-policy-fitted-value-march-three-v162/summary.json` (superseded forecast cohort)
- `data/benchmarks/event-policy-fitted-value-march-policy-v163/summary.json` (superseded selection source)
- `data/benchmarks/event-policy-fitted-value-march-horizon-v164/summary.json` (replay-aligned diagnostic remains valid)
- `data/benchmarks/event-policy-fitted-value-march-chain-screen-v165/summary.json`
- `data/benchmarks/event-policy-fitted-value-march-chain-screen-v165/reproduction-check.json`
- `data/benchmarks/event-policy-fitted-value-march-chain-three-v166/summary.json`
- `data/benchmarks/event-policy-fitted-value-march-chain-three-v166/reproduction-check.json`
- `data/benchmarks/event-policy-fitted-value-march-chain-three-v166/trade-attribution.json`
- `data/benchmarks/event-policy-fitted-value-march-chain-policy-v167/summary.json`
- `data/benchmarks/event-policy-fitted-value-march-chain-policy-v167/reproduction-check.json`
- `scripts/audit-event-fitted-critics.ts`
- `data/benchmarks/event-policy-fitted-critic-march-v168/summary.json`
- `data/benchmarks/event-policy-fitted-critic-may-v169/summary.json`
- `scripts/event-second-dynamics.ts`
- `data/benchmarks/event-policy-fitted-value-exact-second-march-v170/summary.json`
- `data/benchmarks/event-policy-fitted-value-exact-second-march-v170/reproduction-check.json`
- `data/benchmarks/event-policy-fitted-value-exact-second-may-v171/summary.json`
- `data/benchmarks/event-policy-fitted-value-exact-second-may-v171/reproduction-check.json`
- `data/benchmarks/event-policy-fitted-value-exact-second-may-v171/paired-day-audit.json`
- `data/benchmarks/event-policy-fitted-value-exact-second-three-v172/summary.json`
- `data/benchmarks/event-policy-fitted-value-exact-second-march-policy-v173/summary.json`
- `data/benchmarks/event-policy-fitted-value-exact-second-march-policy-v173/reproduction-check.json`
- `data/benchmarks/event-policy-fitted-value-exact-second-may-policy-v174/summary.json`
- `data/benchmarks/event-policy-fitted-value-exact-second-may-policy-v174/reproduction-check.json`
- `data/benchmarks/event-policy-fitted-value-exact-second-audit-v175/summary.json`
- `data/benchmarks/event-policy-fitted-value-exact-second-audit-v175/reproduction-check.json`
- `data/benchmarks/event-policy-fitted-value-exact-second-audit-v175/round-trip-audit.json`
- `data/benchmarks/event-policy-fitted-value-exact-second-audit-v175/action-attribution.ts`
- `data/benchmarks/event-policy-fitted-value-exact-second-audit-v175/action-attribution.json`
- `scripts/event-paths.ts`
- `scripts/audit-event-sign-horizons.ts`
- `data/benchmarks/event-policy-sign-horizons-may-v176/summary.json`
- `data/benchmarks/event-policy-sign-horizons-may-v176/paired-day-audit.json`
- `data/benchmarks/event-policy-sign-horizons-march-v177/summary.json`
- `data/benchmarks/event-policy-sign-horizons-march-v177/paired-day-audit.json`
- `data/benchmarks/event-policy-fitted-value-sampled-path-may-v178/summary.json`
- `data/benchmarks/event-policy-fitted-value-sampled-path-may-v178/reproduction-check.json`
- `data/benchmarks/event-policy-fitted-value-sampled-path-may-v178/target-clarification.json`
- `data/benchmarks/event-policy-fitted-value-sampled-path-may-v178/round-trip-audit.cjs`
- `data/benchmarks/event-policy-fitted-value-sampled-path-may-v178/round-trip-audit.json`
- `data/benchmarks/event-policy-fitted-value-sampled-path-policy-v179/summary.json`
- `data/benchmarks/event-policy-fitted-value-sampled-path-policy-v179/reproduction-check.json`
- `data/benchmarks/event-policy-fitted-value-sampled-path-audit-v180/summary.json`
- `data/benchmarks/event-policy-fitted-value-sampled-path-audit-v180/reproduction-check.json`
- `data/benchmarks/event-policy-fitted-value-sampled-path-march-v181/summary.json`
- `data/benchmarks/event-policy-fitted-value-sampled-path-march-v181/reproduction-check.json`
- `data/benchmarks/event-policy-fitted-value-sampled-path-march-v181/target-clarification.json`
- `data/benchmarks/event-policy-fitted-value-sampled-path-march-v181/trade-attribution.json`
- `data/benchmarks/event-policy-fitted-value-sampled-path-march-policy-v182/summary.json`
- `data/benchmarks/event-policy-fitted-value-sampled-path-march-policy-v182/reproduction-check.json`
- `data/benchmarks/event-policy-fitted-influence-march-v183/summary.json`
- `data/benchmarks/event-policy-fitted-influence-march-v183/weight-sign-audit.json`
- `data/benchmarks/event-policy-fitted-influence-may-v184/summary.json`
- `data/benchmarks/event-policy-fitted-influence-may-v184/weight-sign-audit.json`
- `data/benchmarks/event-policy-fitted-value-centered-march-v185/summary.json`
- `data/benchmarks/event-policy-fitted-value-centered-may-v186/summary.json`
- `data/benchmarks/event-policy-fitted-value-deviation-sampled-march-v187/summary.json`
- `data/benchmarks/event-policy-fitted-value-centered-march-policy-v188/summary.json`
- `data/benchmarks/event-policy-fitted-value-centered-march-policy-v188/reproduction-check.json`
- `data/benchmarks/event-policy-fitted-value-centered-may-policy-v189/summary.json`
- `data/benchmarks/event-policy-fitted-value-centered-may-policy-v189/reproduction-check.json`
- `scripts/event-ridge-influence.ts`
- `scripts/audit-event-fitted-influence.ts`
- `scripts/audit-event-continuation-crossfit.ts`
- `scripts/event-fitted-crossfit.ts`
- `data/benchmarks/event-policy-continuation-crossfit-march-v190/summary.json`
- `data/benchmarks/event-policy-continuation-crossfit-may-v191/summary.json`
- `data/benchmarks/event-policy-fitted-value-crossfit-march-v192/summary.json`
- `data/benchmarks/event-policy-fitted-value-crossfit-march-v192/reproduction-check.json`
- `data/benchmarks/event-policy-fitted-value-crossfit-march-v192/trade-attribution.json`
- `data/benchmarks/event-policy-fitted-value-crossfit-may-v193/summary.json`
- `data/benchmarks/event-policy-fitted-value-crossfit-may-v193/reproduction-check.json`
- `data/benchmarks/event-policy-fitted-value-crossfit-may-v193/trade-attribution.json`
- `data/benchmarks/event-policy-fitted-value-crossfit-march-policy-v194/summary.json`
- `data/benchmarks/event-policy-fitted-value-crossfit-march-policy-v194/reproduction-check.json`
- `data/benchmarks/event-policy-fitted-value-crossfit-may-policy-v195/summary.json`
- `data/benchmarks/event-policy-fitted-value-crossfit-may-policy-v195/reproduction-check.json`
- `data/benchmarks/event-policy-fitted-value-crossfit-may-audit-v196/summary.json`
- `data/benchmarks/event-policy-fitted-value-crossfit-may-audit-v196/reproduction-check.json`
- `data/benchmarks/event-policy-fitted-value-crossfit-may-audit-v196/action-contrast-audit.json`
- `data/benchmarks/event-policy-fitted-influence-crossfit-may-v197/summary.json`
- `data/benchmarks/event-policy-fitted-influence-crossfit-march-v198/summary.json`
- `data/benchmarks/event-policy-continuation-crossfit20-march-v199/summary.json`
- `data/benchmarks/event-policy-continuation-crossfit20-may-v200/summary.json`
- `data/benchmarks/event-policy-fitted-value-crossfit20-march-v201/summary.json`
- `data/benchmarks/event-policy-fitted-value-crossfit20-march-v201/reproduction-check.json`
- `data/benchmarks/event-policy-fitted-value-crossfit20-march-v201/trade-attribution.json`
- `data/benchmarks/event-policy-fitted-value-crossfit20-may-v202/summary.json`
- `data/benchmarks/event-policy-fitted-value-crossfit20-may-v202/reproduction-check.json`
- `data/benchmarks/event-policy-fitted-value-crossfit20-may-v202/trade-attribution.json`
- `data/benchmarks/event-policy-fitted-value-crossfit20-march-policy-v203/summary.json`
- `data/benchmarks/event-policy-fitted-value-crossfit20-march-policy-v203/reproduction-check.json`
- `data/benchmarks/event-policy-fitted-value-crossfit20-may-policy-v204/summary.json`
- `data/benchmarks/event-policy-fitted-value-crossfit20-may-policy-v204/reproduction-check.json`
- `scripts/event-local-value.ts`
- `scripts/screen-event-local-value.ts`
- `scripts/audit-event-local-value.ts`
- `data/benchmarks/event-policy-local-value-march-v205/summary.json`
- `data/benchmarks/event-policy-local-value-may-v206/summary.json`
- `data/benchmarks/event-policy-local-value-march-audit-v207/summary.json`
- `data/benchmarks/event-policy-local-value-may-audit-v208/summary.json`
- `data/benchmarks/event-policy-local-value-sign-baseline-may-v209/summary.json`
- `data/benchmarks/event-policy-local-value-clock-march-v210/summary.json`
- `data/benchmarks/event-policy-local-value-clock-may-v211/summary.json`
- `data/benchmarks/event-policy-fitted-value-shape-march-v212/summary.json`
- `data/benchmarks/event-policy-fitted-value-shape-march-v212/check.cjs`
- `data/benchmarks/event-policy-fitted-value-shape-march-v212/reproduction-check.json`
- `data/benchmarks/event-policy-fitted-value-shape-march-v212/archive-coverage.json`
- `data/benchmarks/event-policy-fitted-value-shape-may-v213/summary.json`
- `data/benchmarks/event-policy-fitted-value-shape-may-v213/reproduction-check.json`
- `data/benchmarks/event-policy-sign-shape-march-v214/summary.json`
- `data/benchmarks/event-policy-sign-shape-may-v215/summary.json`
- `data/benchmarks/event-policy-sign-audit-regression-v216/check.cjs`
- `data/benchmarks/event-policy-sign-audit-regression-v216/reproduction-check.json`
- `scripts/screen-event-hold-paths.ts`
- `scripts/audit-event-hold-paths.ts`
- `scripts/screen-event-hold-options.ts`
- `scripts/replay-event-hold-option.ts`
- `data/benchmarks/event-policy-hold-path-march-v217/summary.json`
- `data/benchmarks/event-policy-hold-path-may-v218/summary.json`
- `data/benchmarks/event-policy-hold-path-audit-march-v219/summary.json`
- `data/benchmarks/event-policy-hold-path-audit-may-v220/summary.json`
- `data/benchmarks/event-policy-hold-option-may-v221/failure.json`
- `data/benchmarks/event-policy-hold-option-may-v222/summary.json`
- `data/benchmarks/event-policy-hold-option-may-v222/trade-audit.cjs`
- `data/benchmarks/event-policy-hold-option-may-v222/trade-attribution.json`
- `data/benchmarks/event-policy-hold-option-march-v223/summary.json`
- `data/benchmarks/event-policy-hold-option-may-policy-v224/summary.json`
- `data/benchmarks/event-policy-hold-option-may-policy-v224/reproduction-check.json`
- `scripts/audit-event-hold-opportunity.ts`
- `scripts/screen-event-controller-composition.ts`
- `scripts/audit-event-controller-composition.ts`
- `scripts/replay-event-controller-composition.ts`
- `data/benchmarks/event-policy-hold-opportunity-may-v225/summary.json`
- `data/benchmarks/event-policy-hold-opportunity-crossfit-may-v226/summary.json`
- `data/benchmarks/event-policy-controller-composition-may-v227/summary.json`
- `data/benchmarks/event-policy-controller-composition-crossfit-may-v228/summary.json`
- `data/benchmarks/event-policy-controller-composition-crossfit-march-v229/summary.json`
- `data/benchmarks/event-policy-controller-fixed-control-may-v230/summary.json`
- `data/benchmarks/event-policy-controller-fixed-control-crossfit-may-v231/summary.json`
- `data/benchmarks/event-policy-controller-fixed-control-crossfit-march-v232/summary.json`
- `data/benchmarks/event-policy-controller-composition-audit-may-v233/summary.json`
- `data/benchmarks/event-policy-controller-composition-audit-crossfit-may-v234/summary.json`
- `data/benchmarks/event-policy-controller-composition-may-policy-v235/summary.json`
- `data/benchmarks/event-policy-controller-composition-may-policy-v235/reproduction-check.json`
- `scripts/audit-event-controller-values.ts`
- `data/benchmarks/event-policy-controller-value-may-v236/summary.json`
- `data/benchmarks/event-policy-controller-value-crossfit-may-v237/summary.json`
- `data/benchmarks/event-policy-controller-value-crossfit-march-v238/summary.json`
- `data/benchmarks/event-policy-controller-cash-replan-may-v239/summary.json`
- `data/benchmarks/event-policy-controller-cash-replan-crossfit-may-v240/summary.json`
- `data/benchmarks/event-policy-controller-cash-replan-crossfit-march-v241/summary.json`
- `data/benchmarks/event-policy-controller-cash-replan-may-policy-v242/summary.json`
- `data/benchmarks/event-policy-controller-cash-replan-may-policy-v242/reproduction-check.json`
- `data/benchmarks/event-policy-controller-cash-replan-audit-may-v243/summary.json`
- `data/benchmarks/event-policy-controller-cash-replan-audit-crossfit-may-v244/summary.json`
- `scripts/audit-event-policy-rollouts.ts`
- `data/benchmarks/event-policy-deployed-rollout-may-v245/summary.json`
- `data/benchmarks/event-policy-deployed-rollout-crossfit-may-v246/summary.json`
- `data/benchmarks/event-policy-deployed-rollout-crossfit-march-v247/summary.json`
- `data/benchmarks/event-policy-controller-quote-entry-may-v248/summary.json`
- `data/benchmarks/event-policy-controller-quote-entry-crossfit-may-v249/summary.json`
- `data/benchmarks/event-policy-controller-quote-entry-crossfit-march-v250/summary.json`
- `data/benchmarks/event-policy-controller-quote-entry-audit-may-v251/summary.json`
- `data/benchmarks/event-policy-controller-quote-entry-audit-crossfit-may-v252/summary.json`
- `data/benchmarks/event-policy-controller-quote-entry-may-policy-v253/summary.json`
- `data/benchmarks/event-policy-controller-quote-entry-may-policy-v253/reproduction-check.json`
- `data/benchmarks/event-policy-deployed-rollout-quote-may-v254/summary.json`
- `data/benchmarks/event-policy-deployed-rollout-quote-crossfit-may-v255/summary.json`
- `packages/bot-algo/test/event-bellman-reference.ts`
- `scripts/audit-event-bellman-optimality.ts`
- `data/benchmarks/event-policy-fixed-model-optimality-v256/summary.json`
- `data/benchmarks/event-policy-fixed-model-grid-refinement-v257/summary.json`
- `packages/bot-algo/src/event-one-step.ts`
- `scripts/audit-event-one-step.ts`
- `scripts/screen-event-one-step.ts`
- `data/benchmarks/event-policy-one-event-lot-optimizer-v258/failure.json`
- `data/benchmarks/event-policy-one-event-lot-optimizer-v259/summary.json`
- `data/benchmarks/event-policy-one-event-lot-may-v260/summary.json`
- `data/benchmarks/event-policy-one-event-lot-march-v261/summary.json`
- `data/benchmarks/event-policy-one-event-lot-january-v262/summary.json`
- `data/benchmarks/event-policy-one-event-marked-optimizer-v263/summary.json`
- `data/benchmarks/event-policy-one-event-marked-may-v264/summary.json`
- `data/benchmarks/event-policy-one-event-marked-march-v265/summary.json`
- `data/benchmarks/event-policy-one-event-marked-january-v266/summary.json`
- `packages/bot-algo/src/event-two-step.ts`
- `packages/bot-algo/src/event-one-step-upper.ts`
- `scripts/audit-event-two-step.ts`
- `scripts/probe-event-two-step.ts`
- `scripts/replay-event-two-step.ts`
- `scripts/audit-event-two-step-replay.ts`
- `data/benchmarks/event-policy-two-event-bound-audit-v267/summary.json`
- `data/benchmarks/event-policy-two-event-market-probe-v268/summary.json`
- `data/benchmarks/event-policy-two-event-dual-bound-audit-v269/summary.json`
- `data/benchmarks/event-policy-two-event-dual-market-probe-v270/summary.json`
- `data/benchmarks/event-policy-two-event-dual-bound-audit-v271/summary.json`
- `data/benchmarks/event-policy-two-event-dual-may-inventory-v272/summary.json`
- `data/benchmarks/event-policy-two-event-incumbent-bound-audit-v273/summary.json`
- `data/benchmarks/event-policy-two-event-incumbent-may-v274/summary.json`
- `data/benchmarks/event-policy-two-event-incumbent-march-v275/summary.json`
- `data/benchmarks/event-policy-two-event-incumbent-january-v276/summary.json`
- `data/benchmarks/event-policy-two-event-active-march-v277/summary.json`
- `data/benchmarks/event-policy-two-event-active-may-v278/summary.json`
- `data/benchmarks/event-policy-two-event-active-january-v279/summary.json`
- `data/benchmarks/event-policy-two-event-active-may-refine-v280/summary.json`
- `data/benchmarks/event-policy-two-event-prefix-march-v281/summary.json`
- `data/benchmarks/event-policy-two-event-active-january-refine-v282/summary.json`
- `data/benchmarks/event-policy-one-event-hold-shortcut-v283/summary.json`
- `data/benchmarks/event-policy-two-event-prefix-fast-march-v284/summary.json`
- `data/benchmarks/event-policy-two-event-prefix-fast-march-v284/reproduction-check.json`
- `data/benchmarks/event-policy-two-event-origin-march-v285/summary.json`
- `data/benchmarks/event-policy-two-event-origin-march-audit-v286/summary.json`
- `data/benchmarks/event-policy-two-event-countdown-march-v287/summary.json`
- `data/benchmarks/event-policy-two-event-final-march-v288/summary.json`
- `scripts/audit-event-policy-suite.ts`
- `scripts/merge-event-policy-suite.ts`
- `data/benchmarks/event-policy-one-event-full-suite-v289/summary.json`
- `data/benchmarks/event-policy-two-event-dense-profile-v290/summary.json`
- `data/benchmarks/event-policy-two-event-full-suite-v291/summary.json`
- `data/benchmarks/event-policy-two-event-full-suite-v291/stopped.json`
- `packages/bot-algo/src/event-one-step-prepared.ts`
- `data/benchmarks/event-policy-prepared-one-event-audit-v292/summary.json`
- `data/benchmarks/event-policy-prepared-two-event-audit-v293/summary.json`
- `data/benchmarks/event-policy-two-event-prepared-dense-profile-v294/summary.json`
- `data/benchmarks/event-policy-two-event-prepared-window-v295/summary.json`
- `data/benchmarks/event-policy-two-event-prepared-window-v295/reproduction-check.json`
- `data/benchmarks/event-policy-two-event-full-suite-prepared-v296/summary.json`
- `data/benchmarks/event-policy-two-event-full-suite-prepared-v296/stopped.json`
- `data/benchmarks/event-policy-two-event-final-march-audit-v297/summary.json`
- `data/benchmarks/event-policy-two-event-regime-profile-v299/summary.json`
- `data/benchmarks/event-policy-two-event-regime-cpu-v300/summary.json`
- `data/benchmarks/event-policy-prepared-neighbor-audit-v301/summary.json`
- `data/benchmarks/event-policy-two-event-neighbor-audit-v302/summary.json`
- `data/benchmarks/event-policy-two-event-regime-neighbors-v303/summary.json`
- `data/benchmarks/event-policy-two-event-neighbor-full-audit-v304/summary.json`
- `packages/bot-algo/src/event-holding-law.ts`
- `data/benchmarks/event-policy-two-event-dense-series-audit-v305/summary.json`
- `data/benchmarks/event-policy-two-event-regime-series-v306/summary.json`
- `data/benchmarks/event-policy-two-event-regime-series-v306/reproduction-check.json`
- `data/benchmarks/event-policy-two-event-series-window-v307/summary.json`
- `data/benchmarks/event-policy-two-event-series-window-v307/reproduction-check.json`
- `data/benchmarks/event-policy-two-event-full-suite-series-v308/summary.json`
- `data/benchmarks/event-policy-two-event-full-suite-merged-v309/summary.json`
- `data/benchmarks/event-policy-two-event-full-suite-merged-v309/behavior.json`
- `packages/bot-algo/src/event-three-step.ts`
- `packages/bot-algo/src/event-multi-step-upper.ts`
- `scripts/probe-event-three-step.ts`
- `scripts/probe-event-multi-step-upper.ts`
- `scripts/audit-event-three-step.ts`
- `scripts/audit-event-multi-step-upper.ts`
- `docs/experiments/structured-next-second-sign-audit-2026-09-04.md`
- `data/benchmarks/event-policy-two-event-reuse-july-v310/summary.json`
- `data/benchmarks/event-policy-three-event-june-profile-v311/summary.json`
- `data/benchmarks/event-policy-three-event-november-profile-v312/summary.json`
- `data/benchmarks/event-policy-three-event-november-warm-profile-v313/summary.json`
- `data/benchmarks/event-policy-three-event-june-warm-profile-v314/summary.json`
- `data/benchmarks/event-policy-three-event-action-audit-v315/summary.json`
- `data/benchmarks/event-policy-three-event-november-entry-v316/summary.json`
- `data/benchmarks/event-policy-three-event-june-entry-v317/summary.json`
- `data/benchmarks/event-policy-three-event-june-entry-v317/stopped.json`
- `data/benchmarks/event-policy-recursive-upper-november-v318/summary.json`
- `data/benchmarks/event-policy-recursive-upper-june-v319/summary.json`
- `data/benchmarks/event-policy-recursive-upper-november-fine-v320/summary.json`
- `data/benchmarks/event-policy-three-event-november-upper-seed-v321/summary.json`
- `data/benchmarks/event-policy-recursive-upper-audit-v322/summary.json`
- `data/benchmarks/event-policy-recursive-direction-november-v323/summary.json`
- `data/benchmarks/event-policy-recursive-marginal-upper-audit-v324/summary.json`
- `data/benchmarks/event-policy-recursive-marginal-november-v325/summary.json`
- `data/benchmarks/event-policy-recursive-marginal-november-fine-v326/summary.json`
- `data/benchmarks/event-policy-three-event-november-global-profile-v327/summary.json`
- `data/benchmarks/event-policy-three-event-june-global-profile-v328/summary.json`
- `data/benchmarks/event-policy-three-event-november-global-v329/summary.json`
- `data/benchmarks/event-policy-three-event-november-global-v329/reproduction-check.json`
- `data/benchmarks/event-policy-three-event-june-global-v330/summary.json`
- `data/benchmarks/event-policy-two-event-reuse-final-july-v331/summary.json`
- `data/benchmarks/event-policy-two-event-reuse-final-july-v331/reproduction-check.json`
- `data/benchmarks/event-policy-two-event-reuse-audit-v332/summary.json`
- `data/benchmarks/event-policy-recursive-directional-audit-v333/summary.json`
- `data/benchmarks/event-policy-recursive-marginal-june-fine-v334/summary.json`
- `data/benchmarks/event-policy-recursive-marginal-june-refined-v335/summary.json`
- `data/benchmarks/event-policy-three-event-june-upper-seed-v336/summary.json`
- `data/benchmarks/event-policy-recursive-marginal-june-final-v337/summary.json`
- `data/benchmarks/event-policy-remembered-sign-source-audit-v338/summary.json`
- `scripts/audit-event-three-step-policy.ts`
- `scripts/refine-event-three-step-bounds.ts`
- `scripts/compare-event-policy-depths.ts`
- `data/benchmarks/event-policy-two-event-scalar-audit-v339/summary.json`
- `data/benchmarks/event-policy-three-event-november-scalar-v340/summary.json`
- `data/benchmarks/event-policy-two-event-global-seed-audit-v341/summary.json`
- `data/benchmarks/event-policy-three-event-november-bound-seed-v342/summary.json`
- `data/benchmarks/event-policy-three-event-june-bound-seed-v343/summary.json`
- `data/benchmarks/event-policy-one-event-interval-audit-v344/summary.json`
- `data/benchmarks/event-policy-two-event-interval-audit-v345/summary.json`
- `data/benchmarks/event-policy-three-event-november-interval-v346/summary.json`
- `data/benchmarks/event-policy-three-event-june-interval-v347/summary.json`
- `data/benchmarks/event-policy-three-event-july-prefix-v348/summary.json`
- `data/benchmarks/event-policy-three-event-policy-audit-v349/summary.json`
- `data/benchmarks/event-policy-three-event-july-full-v350/summary.json`
- `data/benchmarks/event-policy-three-event-direction-policy-audit-v351/summary.json`
- `data/benchmarks/event-policy-three-event-order-limit-audit-v352/summary.json`
- `data/benchmarks/event-policy-three-event-july-refined-bounds-v353/summary.json`
- `data/benchmarks/event-policy-three-event-july-comparison-v354/summary.json`
- `data/benchmarks/event-policy-three-event-july-final-bounds-v355/summary.json`
- `data/benchmarks/event-policy-three-event-november-scalar-v340/reproduction-check.json`
- `data/benchmarks/event-policy-three-event-november-interval-v346/reproduction-check.json`
- `data/benchmarks/event-policy-three-event-june-interval-v347/reproduction-check.json`
- `scripts/merge-event-three-step-suite.ts`
- `data/benchmarks/event-policy-three-event-quiet-windows-v356/summary.json`
- `data/benchmarks/event-policy-three-event-november-profile-v357/summary.json`
- `data/benchmarks/event-policy-three-event-november-hoisted-v358/summary.json`
- `data/benchmarks/event-policy-three-event-short-windows-v359/summary.json`
- `data/benchmarks/event-policy-three-event-small-kernels-v360/summary.json`
- `data/benchmarks/event-policy-three-event-short-window-bounds-v361/summary.json`
- `data/benchmarks/event-policy-three-event-root-lot-audit-v362/summary.json`
- `data/benchmarks/event-policy-three-event-short-root-lot-bounds-v363/summary.json`
- `data/benchmarks/event-policy-three-event-small-kernel-bounds-v364/summary.json`
- `data/benchmarks/event-policy-three-event-november-allocation-v365/summary.json`
- `data/benchmarks/event-policy-three-event-three-day-sharpe-v366/summary.json`
- `data/benchmarks/event-policy-three-event-split-hold-audit-v367/summary.json`
- `data/benchmarks/event-policy-three-event-small-kernel-final-bounds-v368/summary.json`
- `data/benchmarks/event-policy-three-event-december-prefix-profile-v369/summary.json`
- `data/benchmarks/event-policy-three-event-december-tail-profile-v370/summary.json`
- `data/benchmarks/event-policy-three-event-december-coarse-v371/summary.json`
- `data/benchmarks/event-policy-three-event-december-strict-tail-v372/summary.json`
- `data/benchmarks/event-policy-three-event-three-day-sharpe-resumed-v373/summary.json`
- `data/benchmarks/event-policy-three-event-final-policy-audit-v374/summary.json`
- `data/benchmarks/event-policy-three-event-partial-suite-v375/summary.json`
- `data/benchmarks/event-policy-three-event-november-profile-v357/cpu-summary.json`
- `data/benchmarks/event-policy-three-event-november-hoisted-v358/cpu-summary.json`
- `data/benchmarks/event-policy-three-event-november-hoisted-v358/reproduction-check.json`
- `data/benchmarks/event-policy-three-event-november-allocation-v365/reproduction-check.json`
- `data/benchmarks/event-policy-three-event-three-day-sharpe-v366/stopped.json`
- `data/benchmarks/event-policy-three-event-december-strict-tail-v372/comparison.json`
- `data/benchmarks/event-policy-three-event-partial-suite-v375/behavior.json`
- `data/benchmarks/event-policy-three-event-seven-day-sharpe-v376/summary.json`
- `data/benchmarks/event-policy-three-event-june-allocation-v377/summary.json`
- `data/benchmarks/event-policy-three-event-june-coarse-continuation-v378/summary.json`
- `data/benchmarks/event-policy-three-event-june-coarse-continuation-v378/global-check.json`
- `data/benchmarks/event-policy-three-event-staged-audit-v379/summary.json`
- `data/benchmarks/event-policy-three-event-june-staged-prefix-v380/summary.json`
- `data/benchmarks/event-policy-three-event-seven-day-bounds-v381/summary.json`
- `data/benchmarks/event-policy-three-event-remaining-small-v382/summary.json`
- `data/benchmarks/event-policy-three-event-june-staged-full-v383/config.json`
- `data/benchmarks/event-policy-three-event-seven-day-final-bounds-v384/summary.json`
- `data/benchmarks/event-policy-three-event-twenty-window-suite-v385/summary.json`
- `data/benchmarks/event-policy-three-event-twenty-window-suite-v385/behavior.json`
- `data/benchmarks/event-policy-three-event-remaining-small-bounds-v386/summary.json`
- `data/benchmarks/event-policy-three-event-remaining-small-fine-bounds-v387/summary.json`
- `data/benchmarks/event-policy-three-event-twenty-six-window-suite-v388/summary.json`
- `data/benchmarks/event-policy-three-event-twenty-six-window-suite-v388/behavior.json`
