# Native-second event policy — historical experiments

Latest forecast/policy checkpoint, v676–v691: a strictly earlier minute-event
direction expert transfers positively to the 48 bp/four-hour native target. A
log-odds ensemble with the native return-weighted fast head reaches +2.3409%
return-MSE skill and 58.61% magnitude-weighted direction on the untouched
seven-day calibration half, but no one-event mean clears the 24 bp round trip.
Causal H2 integration reveals why that economic gate matters: the unguarded
ensemble loses 21.57% on the seven-day selection segment. Treating the eight
selection-eligible coefficient pairs as a forecast interval and scaling a 50 bp
loss budget by signal-to-interval width cuts the selected policy to two fills;
it returns +0.000050% on selection and +0.000258% on untouched holdout. On the
three-day non-fit July inspector window it loses 0.000618% with 0.000800%
drawdown. The capital rule successfully contains the adverse events, but the
ensemble remains economically rejected. The frozen native H1 policy's sparse
holding behavior should be preserved; the slow/fast law is retained as a useful
forecast candidate and uncertainty source, not an action-ranking replacement.
See the final section for exact execution results and artifacts.

Earlier forecast checkpoint, v670–v675: a calibration-only competing-risk
screen separates barrier arrival from barrier direction and weights the two
heads independently. Exact 15m/30m/60m/4h volatility state improves the first
seven-day selection block, but the selected law fails the untouched seven-day
block: return-MSE skill is -0.6912%, group NLL worsens to 1.10212 and
magnitude-weighted direction falls to 43.99%. Its largest mean is only 6.54 bp,
so no forecast clears even the 12 bp one-way cost. The screen is rejected before
test or policy replay. The result isolates the remaining problem as unstable
cost-sized direction rather than missing capital protection or barrier-arrival
modeling. See the final section for the factorization and artifacts.

Earlier checkpoint, v563–v615: the four-day law remains cash under exact
next-open H1 execution, while compact positive quadrature makes bounded H2
experiments practical. Propagating the event sign head into current and
successor kernels turns a 4.03% failed replay into a near-flat **-0.001795%**
one-day result, but fees still exceed positive gross trading PnL. A
chronologically selected sign/magnitude ensemble improves return-MSE skill from
-20.28% to -9.11% on 24 held-out decisions, then crosses a discrete Bellman
action threshold and loses 3.35% in the matching execution replay. It is
rejected. A fitted-support risk floor, based on net liquidatable high-water
equity and including the next flatten cost, blocks that catastrophic exposure.
A tighter 0.1 bp risk budget also reduces the sign-only loss. The exact H2
version carries branch-specific liquidatable-equity high-water and floor state
through its child optimizer; it returns -0.000699% with 0.000778% drawdown. A
1 bp version limits the rejected ensemble's former 3.35% loss to 0.00815%.
Full profit protection collapses the same H2 policy to cash. V611 estimates a
90% block-bootstrap interval for the base/sign blend. After correcting a second
sign-probability inversion in the experimental H2 replay, all 25 scored decisions
have intervals spanning both directions. V615 therefore forbids opening,
enlarging or reversing exposure and remains cash. These budgets and uncertainty
rules were inspected after the held-out replay and are diagnostics.
Position decomposition remains the exact lifecycle/accounting representation;
the account-level Bellman coordinator still owns the feasible net action. See
the final section for chronology, limitations and artifacts.

Earlier checkpoint, v514–v521: all **22** remaining singleton November root
regions are classified; two are exact and twenty are certified inferior, leaving
the incumbent unchanged at **0.234902 bp** for -0.72268 BTC. Twelve exact samples
inside the four unresolved intervals reach at most **0.146346 bp**, but do not
certify the unsampled lots. The first whole-interval relaxation costs 124.76 s
and returns a useless **118.603 bp** upper, so the remaining intervals are not
scaled. Target inventory is stable on none of 1,270 outcome rows; target exposure
is stable within 0.001 on only 18.43% of mass. Synthetic exhaustive tests find
**325** H2 concavity violations in 2,860 finite triples. Position decomposition
is retained for lifecycle accounting and signal classification; the full
single-asset value curve stays in the account optimizer. Exact nonanticipativity
cuts are the next computational route. Forecasts, historical returns and global
H2 status remain unchanged.

Earlier checkpoint, v506–v513: ordered acceptance and coupled financial wealth
reduce November's first one-lot bound slack from **25.54 to 0.335 bp**. The exact
H1 solver exposes complete feasible request regions: 59 in November and 381 in
July. Hold-equivalent regions eliminate **33 and 179** regions, but only about
0.6% of their finite lot domains. Four predefined small inventory seeds are
bounded below November's incumbent and their lower policies are independently
rescored on **5,274,568 joint paths**. All **162 focused tests** and workspace
typechecks pass. These are conditional optimization results, not global H2
optimality or a new profitable backtest.

Earlier checkpoint, v501–v505: account-region bounds now preserve uncertain
order acceptance, cash/inventory dependence and shared acceptance at identical
openings. All **17 predefined controls / 68 box queries** dominate the checked
original H1 values, and **159 focused tests** plus workspace typechecks pass.
The diagnostic finds a concrete information-relaxation failure: November's
one-lot region gains about **70.24 bp** by choosing its incoming account after
each outcome. Sharing acceptance at identical openings removes much of this
advantage, but the remaining one-lot upper slack reaches **37.53 bp** and queries
are too expensive. The broad-replay computation gate fails. This is a validated
research bound, not a new policy, global H2 certificate or profitable backtest.
Position-framework conclusions remain unchanged; see the final section.

Earlier checkpoint, v496–v500: nested opening-information partitions tighten
continuation bounds without changing the forecasting or execution model. Across
all **113 predeclared calibration probes / 28 laws**, two-group proposals recover
the exact best value on **111**, with their interval detecting the two misses.
The July fixed-root H2 value now reaches **0.0004471 bp** width in **5.83 s**,
using 172 exact continuations and 105 partition solves. This certifies a value
for that request, not the global root optimum. A second, November control retains
the H1 request's ranking; its apparent edge depends on a 0.0645% modeled fill
probability. All **155 focused tests** and workspace typechecks pass. Historical
returns and position-framework conclusions remain unchanged. Global root-request
search and full-depth convergence are still required; see the final section.

Earlier checkpoint, v492–v495: an execution-consistent continuation upper bound
dominates all **9,629 saved H1 optima / 28 windows**; queries take **0.94 seconds**
after 1.81 seconds of preparation. It enables selective H2 request evaluation:
July's old request is rejected against the verified incumbent after **one**
continuation solve, **0.18 s versus 6.40 s** for its full backup. Root hold needs
no continuation solves. The winning request still needs complete evaluation.
An inactive-boundary pruning experiment is not retained because its timing gain
is small. All **154 focused tests** and workspace typechecks pass. Forecasts,
historical strategy returns and position-framework conclusions are unchanged;
global deeper action search and convergence remain required. Details follow below.

Earlier checkpoint, v483–v491: globally searched execution-aware **H1** now covers
all **9,629 ordinary decisions / 28 full windows**, plus the separate March
forced wait. The original fitted joint laws and old control returns are unchanged.
Results are **8 positive / 19 negative / 1 no-fill window**, higher than the old
H1 on 10 and lower on 17, with **88.42% worst drawdown**. Every chosen value is
reproduced from the full law and beats or ties the old request at the same account;
this conditional utility improvement does not establish profitability. Most
requests have very high modeled rejection probability. The solver passes 960
exhaustive small problems; all 152 focused tests pass. Exact-output specialization
cuts paired probe time about 11.8x; full execution replays total 195.06 seconds.
Workspace typechecks pass. One exact H2 backup preserves the new H1 request's
ranking over two predefined alternatives, but costs 5–8 seconds per candidate.
Global deeper recursion and a matched position-policy comparison remain
outstanding. See the final sections and v490 full-window table.

Earlier checkpoint, v481–v482: a compressed next-open account transition
reproduces **19,258 saved H1/H2 transitions across all 28 intervals**, plus the
separate closure waits, within $4.52e-8. Three enriched empirical laws preserve
their original forecasts and weights exactly. July's first calibration probe
improves expected one-event log growth by **11.2931 bp** by reducing the request
five lots, because rejection probability falls. This is a candidate comparison,
not a global optimizer or a new profitable backtest. The next required work is
global execution-aware action search, then deeper Bellman control. All 149
focused tests and workspace typechecks pass; details are in the final section.

Earlier checkpoint, v477–v480: all 28 full native intervals now have replays:
**27 strict-source windows plus one explicitly assumed March closure scenario**.
The 9,629 optimized H2 actions meet 0.001 bp tolerance; the closure's forced
wait has separate account/availability checks and no fabricated value certificate.
March H2 loses 10.97% versus H1's 6.72% loss. Totals are 7 positive, 17 negative
and 4 cash windows; H2 underperforms H1 on 17. No forecast or decision-framework
promotion follows. All 145 focused tests and workspace typechecks pass. The
remaining optimization requirements include deeper convergence and consistent
execution, and the position decision framework still needs a matched comparison.

Earlier checkpoint, v470–v476: **27 complete native windows / 9,219 actions**
meet the finite H2 tolerance of 0.001 bp. All 28 unchanged-protocol fitting
screens pass; the remaining full window fails strict March 2023 source
validation. A trade-flow audit finds zero trades throughout the missing
seconds and a longer 11:27:24–14:00 no-trade interval. Explicit availability
and execution modeling remain necessary. H2 has 7 positive, 16 negative and
4 cash windows, and lower returns than H1 on 16 windows. This is numerical
coverage, not strategy promotion or full-depth/next-open optimality. A tighter
recovery bound clears the choppy-window probes; all 137 focused tests and
workspace typechecks pass. The complete table is in the final section.

Earlier checkpoint, v454–v469: the complete June downtrend window now has
1,562/1,562 original H2 actions certified within 0.001 bp, plus 39 calibration
actions. H2 returns +272.57% versus H1 +278.59%, with 49.87% drawdown;
calibration loses 16.42%. This is a second complete native window, not a
forecast promotion or all-window/stationary optimality. A capacity-aware
continuation bound certifies all 214 originally unresolved actions unchanged
in 4.64 seconds. Date-only fit exclusion and separated source blocks preserve
exact control models, forecasts and trades. All 136 focused tests and
workspace typechecks pass. See the final section for scope and failure analysis.

Earlier checkpoint, v442–v453: average-uniqueness weighting preserves the
original H1 actions but removes H2's damaging short exposure on the two-day
screen. Weighted H2 returns +1.07% calibration / +0.25% test. Extending the
same frozen forecast to the complete November inspector window earns +59.53%,
versus +171.16% for H1: positive profit is not an H2 performance victory.
All 357 full-window H2 actions, plus 35 calibration actions, are certified
within 0.001 bp. This is one full window and finite H2, not all-window or
stationary optimality. All 133 focused tests and workspace typechecks pass.

Earlier checkpoint, v435–v441: adding exact 12h/24h returns to native seconds
does not establish a forecast improvement. With broader estimation history it
earns +0.88% calibration / +4.58% test, but the entry disappears when one
estimation day is removed. Event-chain estimation also removes it, including
with the global prior-to-observation ratio held constant. Retain the causal
feature contract and diagnostics; do not promote this fragile entry or expand
its Bellman computation. All 131 focused tests and workspace typechecks pass.

Earlier checkpoint, v428–v434: a frozen-partition estimation ablation shows
that the bearish state's mean depends strongly on the estimated days. Adding
three earlier days flips it from -10.63 to +3.34 bp and yields cash through H2;
calibration forecast error worsens while test error improves. Event-chain
controls also stay cash. Conditional duration information survives, but neither
ordinary nor return-weighted separate sign heads clears the diagnostic checks.
No new policy is promoted; see the final section for the controlled evidence.

Earlier optimization checkpoint, v427: all 78 original native H2 actions on
the two-day screen meet the 0.001 bp value-gap tolerance. Returns are +8.17%
calibration and -11.00% test under the frozen v412 forecast. Broader native
coverage and deeper convergence remain incomplete.

Earlier coverage checkpoint: 27 full native window replays / 6.70 million decisions
remain cash under the basic tree; one official archive gap prevents complete
replay coverage. A fresh historical production59/base17 refit reaches 63.02%
validation and 65.04% test active-next-second sign accuracy. Its calibrated
one-second edge still falls below costs. See the final two sections for
v395–v402 results, data limitations and the next experiment.

The original minute baseline was a compute-driven implementation choice,
without a controlled resolution comparison. This implementation now supports
native second observations explicitly. It does not claim that seconds improve
returns merely because they match the requested starting resolution.

## Implemented contract

- `EventClock.candleIntervalMs=1000` selects native seconds. The existing
  minute contract remains the default. Native feature names cannot be used
  with a minute clock, nor minute feature names with a second clock.
- The shared loader validates complete, ordered 1s OHLCV, exact close times
  and requested coverage. Missing seconds are errors, not forward-filled rows.
- Fast features use exactly 64 completed second candles: the existing EMA2/RSI2
  dynamics, 60s realized volatility, range, close location, volume/activity,
  and sign/run age/return capped at 63s. This is an initial bounded basis,
  not the production59/base17 structured model or a claim that slow features
  are unnecessary. The pure dynamics implementation is shared with its old
  minute-boundary observation cache.
- Explicit `context` adds exact endpoint returns through four hours; explicit
  `day-context` preserves those nineteen features and appends 12h/24h returns.
  Their declared purge support is respectively four hours and one day. Neither
  contract resamples the candles or reads future values.
- A raw run ends at the first **observed** change in close-to-close sign,
  including zero. The revealing second belongs to the event payoff; there
  is no retrospective trade at the preceding extremum. A 60s timeout is a
  decision boundary even if the run has not ended. The separate next-second
  experiment uses one candle per decision.
- Outcome atoms retain joint arithmetic return, extrema, duration and next
  feature-state leaf. Financial duration remains **physical minutes**, so a
  one-second holding step uses `1/60` minute. Duration class bins default to 5s
  and 20s in the native model; explicit physical-minute boundaries support
  longer decision events. The 15-class tree label's central return interval
  includes both zero and small nonzero returns; exact zeros remain distinct
  atoms but are not a separate supervised class in this first partition.
- The replay commits orders at the completed close, attempts the same base
  quantity at the next open, applies order/cap checks at that fill price,
  and charges actual execution fees and terminal settlement. Native replay
  currently rejects the old minute-specific forecast adapters.
- Exact H1/H2/H3 solvers can now use a policy with zero precomputed grid
  tables. This avoids building unused approximate value tables.

## Frozen-model screen

Artifacts:

- `event-policy-native-second-run-screen-v391`
- `event-policy-native-next-second-screen-v392`

Both fit on November 2–3, 2024, using one training origin every 30 seconds.
They reserve November 4 for diagnostic calibration and score the first hour
of November 5 from `sharpe-up-7d-2024-11`. Complete training input/target
support is purged against the inspector catalog, including fit windows.
The calibration segment is reported but does not select or modify either
model. Each tree has depth 2, minimum leaf 128 and prior strength 32. This is
an inspected research prefix, not full-window or fresh-holdout evidence.

Each run reads 262,864 actual 1s rows and uses 5,760 training examples plus
2,880 calibration examples. Source reference hashes, code and frozen models
are saved. Costs are the current defaults: **5x maximum leverage**, 10 bp
fees plus 2 bp slippage per turnover, minimum notional 5, maximum 50,000,
quantity step/minimum 0.00001, maintenance 0.005 and 1 bp/day borrowing.
The earlier minute baseline used a 1x cap, so these are not a controlled
minute-versus-second return comparison.

| Result | Run clock | Next-second clock |
| --- | ---: | ---: |
| Completed test targets | 2,314 | 3,599 |
| Replay decisions, including boundary event | 2,315 | 3,600 |
| Median / 99th percentile event duration | 1s / 6s | 1s / 1s |
| Median absolute target return | 0.001470 bp | 0.001470 bp |
| 99th percentile absolute target return | 3.328 bp | 2.148 bp |
| Test active-direction accuracy | 48.94% | 54.35% |
| Test class NLL / unconditional NLL | 1.09750 / 1.08709 | 0.86338 / 0.85998 |
| Largest absolute conditional mean | 0.03988 bp | 0.16469 bp |
| Fitting time | 0.063s | 0.055s |
| Replay time | 0.245s | 0.517s |
| Orders / fees / return | 0 / 0 / 0% | 0 / 0 / 0% |

These weak forecasts do not justify fee-paying entries in H1. The cash
decision preserves equity; it is not evidence of a profitable predictor.
The raw one-second direction scores are not comparable to a claim that the
remembered structured model has been reproduced: that architecture and
training procedure were not used.

The existing `one-second-conditional-sign-magnitude-dependence-2026-08-16.md`
finds history-dependent sign/magnitude coupling, and the structured model's
saved decile audit shows much lower accuracy on economically larger moves.
Any separate sign model must therefore preserve conditional magnitude and
activity information. Reweighting all sizes from one headline sign score is
not justified by those results.

## Cheap finite-horizon cash check

Artifact: `event-policy-native-cash-horizons-v393`.

Instead of expanding billions of paths, form the return-weighted transition
matrix and recurse

`m_0(i) = 1`, `m_h(i) = E[(1 + R) m_(h-1)(next) | i]`.

If every multiplier through depth H stays in `[1-f, 1+f]`, the process
`price_t * m_(H-t)(state_t)` is a martingale under the **forecast probability
law** inside its execution spread, ending at the actual terminal mark.
Trades can only reduce shadow wealth; nonnegative borrowing reduces it
further. Thus expected terminal wealth from cash is at most initial wealth,
and Jensen bounds expected log growth by zero. Cash attains that value.
Order, lot and leverage constraints restrict the competitors further.
Policies with positive-probability liquidation have minus-infinite log value.

This specializes the shadow-market upper-bound argument in
[Czichowsky, Muhle-Karbe and Schachermayer, *Transaction Costs and Shadow
Prices in Discrete Time*](https://www.mat.univie.ac.at/~schachermayer/pubs/preprnts/prpr0156.pdf),
Section 2.2. It uses the forecast measure itself, not just the existence of
some equivalent martingale measure. The recurrence and its cash conclusion
are the project-specific construction; the source does not establish these
models' profitability or the validity of their forecasts.

The implementation retains return/successor dependence, requires probability
mass error below 1e-12, interprets floating weights as normalized probabilities,
and applies an explicit conservative numerical margin. This is a numerical
sufficient check, not a formally rounded interval proof. It certifies cash
for **all four initial leaves** through:

- **9,880 event steps** under the run model (0.014s calculation).
- **2,549 seconds** under the next-second model (0.002s calculation).

The first subsequent failure only means this particular construction no
longer certifies cash; it does not establish a profitable entry. These are
finite event horizons, not fixed wall-clock run horizons or stationary
optimality. They apply to a flat starting account and the frozen
decision-price law, not arbitrary existing holdings or next-open execution.

This changes the next action: deeper H2/H3 expansion cannot rescue these
forecasts from cash. Preserve the cash action and the actual fee constraints;
do not tune fees downward to manufacture trades. Broader native coverage,
better time-scale context and a calibrated joint activity/sign/size forecast
remain necessary. The remembered structured model remains a candidate for
historically valid refitting; this screen neither tests nor rejects it.

## Validation and compatibility

All **124 focused tests** pass. Native tests cover future-data independence,
missing-second rejection, run-boundary revelation, complete-support purging,
terminal censoring, borrowing units, initial-position reconciliation and
next-open cancellation. Cash-bound tests compare small exhaustive Bellman
problems, preserve return/successor dependence, and stop before a deterministic
longer hold actually becomes profitable.

`event-policy-native-clock-minute-regression-v394` repeats every saved H1
minute window after the shared-clock changes. All **1,217 decisions**, 254
orders, values, fills and account paths exactly match v289, with attribution
enabled or disabled. This verifies that native support did not silently
reinterpret the earlier minute results.

## Full native coverage and archive exception (v395–v402)

The unchanged native run tree now has **27 complete window replays**, containing
**6,702,007 decisions**, with zero orders and zero return in every window.
`event-policy-native-run-coverage-v402/summary.json` verifies model hashes,
source-reference hashes, identical settings, unique window coverage and
training/calibration exclusion from every inspector window. The least of
the 27 all-leaf cash bounds is **172 event steps**, so all cover H3.
This establishes a conditional cash result for these laws, not useful
forecasting or profitability. The 1x minute results and 5x native results
still do not constitute a controlled resolution comparison.

The missing replay is `sideways-churn-2023-03`. The loader correctly rejected
the March 24 12:39:41 UTC candle, which closes at millisecond 646. A fresh
download of the [official daily archive](https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1s/BTCUSDT-1s-2023-03-24.zip)
matches its published SHA-256 checksum and contains the same candle. It has
81,582 distinct seconds, 36 identical duplicate rows, and **4,818 missing
seconds** from 12:39:42 through 13:59:59 UTC. Cached OHLCV agrees exactly on
every observed row; missing rows are zero-volume carry candles. No immutable
data was overwritten and strict validation was not relaxed. The audit is
`event-policy-native-data-audit-v397/official-audit.json`.

The interrupted v396 batch is explicitly marked failed; only its four finished
windows were retained. V398 completed the remaining 21 unaffected windows;
v395 supplied the first two. The March window's frozen forecast separately
certifies cash through 382 steps, but it has **no completed replay**. A future
trading policy needs explicit treatment of exchange/data unavailability;
gap-filled rows must not silently become executable market quotes.

## Historical structured sign refit (v399–v401)

The remembered production59/base17 model has now been refit from fresh weights
on historical data. The dataset uses BTC spot 1s/1m, spot aggregate trade flow,
BTC futures 1m and ETH spot 1m over November 2–5, 2024. Training is November
2–3, validation November 4, test November 5. It retains 65,536 constructible
training origins, 8,191 validation origins and 8,190 test origins. Targets
are active next-second returns; all flat seconds remain in input history.
These are bounded prefixes, not full-day or full-window evaluations.

`export-next-return-production-basis.ts` accepts explicit UTC split dates,
refuses existing output directories, purges labels at split boundaries and
writes a minimal five-source history manifest. `Production59BaseHistoryDataset`
shares the existing source readers without loading unrelated 530-feature
example matrices. It verifies all 20 source-reference hashes and UTC axes.
The existing union dataset still opens its 1,115,706 examples and 530 channels.
Teacher-forced primitive reconstruction was checked at 1,006 origins, including
minute/hour boundaries; maximum absolute feature discrepancy is 1.335e-5.
Future primitive changes do not change the observed rollout context. All
50 existing structured-model unit tests pass.

Plan `event-production59-sign-nov2024-pilot-v400` keeps the original 4,130,613
parameter architecture, loss, optimizer, batch sizes, seed and EMA. The data
count is smaller than the original 256k experiment. Eight epochs took 28.8s;
continued validation improvement justified extensions to 32 and then 64.
The 24-epoch extension took 80.2s. Test metrics were not consulted for these
extensions. Validation selected **epoch 50** (zero-based index 49); further
epochs did not improve its feature MSE, so training stopped.

| Selected checkpoint | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Active next-1s direction accuracy | 77.47% | 63.02% | 65.04% |
| Return correlation | 0.2261 | 0.2068 | 0.1433 |
| Return MSE skill versus zero | 5.12% | 4.23% | 2.06% |

This is a historical refit of the correct family, not a reproduction of its
75–77% validation score on the different 2026 population. The primary saved
results are `event-production59-historical-refit-v401/selected-evaluation.json`.
The dense export covers 14,178 validation seconds and 14,634 test seconds;
it removes future labels from the model's rollout context. Its smaller-batch
inference reports 63.04% / 65.02% active direction accuracy, within 0.025
percentage points of the trainer's evaluation. Magnitude-weighted test direction
accuracy is 63.75%; dense inference takes about 1.1 seconds after loading.

Sixteen prediction-score quantile groups, with boundaries and laws fit only
on validation, preserve the empirical down/flat/up frequencies and signed
return magnitudes. Their largest absolute conditional mean is **0.2273 bp**.
The largest individual predicted next-second return in the test export is
**0.8016 bp**. Both are below the unchanged **12 bp one-way fee plus slippage**.

For a flat account and normalized exposure x, the one-step marked wealth
factor before borrowing is `1 - f*abs(x) + x*R`. Every calibrated group has
`abs(E[R]) < f`, so expected wealth cannot exceed cash for either side or any
size. Jensen implies expected log growth cannot exceed zero. The saved
`one-step-cash-check.json` records this sufficient result and its limitations:
the empirical forecast, decision-price execution, marked terminal and cash
initial state. It does not prove the actual market law, existing-position
actions or deeper horizons. The structured model is not yet integrated into
full inspector-window policy replays.

Preserve the stronger sign signal and cash action under the present one-step
law. The next useful experiment should condition larger decision-relevant
move/holding-path distributions on this signal, while checking whether its
edge persists beyond the next second. Increasing sign accuracy alone does
not resolve the cost gap. Position-decomposed decisions still require a
controlled comparison under that same forecast and execution model; the
ledger equivalence result does not answer that policy question.

## Signal persistence and cost-sized events (v403–v409)

V403 keeps the selected structured predictor and validation score bins frozen
and measures cumulative returns at fixed horizons on the same dense prefixes.
Future labels use the cached native closes. Labels overlap; the nominal row
count is not the number of independent outcomes.

| Horizon | Validation score/return correlation | Test correlation | Test magnitude-weighted sign accuracy |
| --- | ---: | ---: | ---: |
| 1 second | 0.1721 | 0.1174 | 63.75% |
| 5 seconds | 0.1476 | 0.1250 | 59.89% |
| 15 seconds | 0.1094 | 0.0856 | 56.88% |
| 60 seconds | 0.0412 | 0.0328 | 51.86% |
| 300 seconds | -0.0007 | 0.0214 | 49.75% |
| 3,600 seconds | 0.0016 | 0.0350 | 49.07% |

The one-hour conditional group means are large, but unconditional means are
also +22.63 bp in validation and +13.41 bp in test. Only four/five origins
can be retained at one-hour spacing in these prefixes. The next-second score
has little measured correlation at that horizon. These observations do not
justify treating its sign accuracy as a persistent one-hour directional edge.

`docs/theory/Position management.md`, under **Predictive target** and
**Complete portfolio-management policy**, explicitly separates the next
significant endpoint/duration distribution from raw-candle direction and
keeps account allocation separate from position accounting. The next small
experiment therefore uses **48 bp close barriers with a one-hour timeout**,
observed on native seconds. The barrier is twice the 24 bp round-trip
friction; it is not a fee reduction. The full joint atoms still retain
return, duration, intraperiod low/high and successor state.

V404/V405 fit November 2–3, reserve November 4 for diagnosis, and replay the
first 24 hours of the November 5 inspector window. The same 5,644 training
targets, 2,762 diagnostic targets and 42 completed test events are used.
There are 43 actual replay decisions, including the terminal censored event.
Both runs take about 1.5 seconds including process startup.

The feature ablation adds exact completed-close log returns at
5/15/60/300/900/3,600/14,400 seconds to the existing 12 native features.
Endpoint timestamps and the four-hour input support are validated; the
longer support participates in fit-window purging. The original feature
contract and behavior remain the default.

| Variant | Calibration H1 return | Test H1 return | Test orders | Test class NLL |
| --- | ---: | ---: | ---: | ---: |
| Fast 12 features, shared fit/estimation (v404) | 0% | 0% | 0 | 1.7321 |
| Fast + slow context, shared fit/estimation (v405) | -3.3882% | -3.0451% | 6 | 1.7443 |
| Context, separate later-day chain estimation (v407) | 0% | 0% | 0 | 1.6564 |
| Context, separate later-day stride estimation (v408) | 0% | 0% | 0 | 1.6182 |

The raw context model learns a bullish state when one-minute volatility is
high and the preceding four-hour return is below -100.27 bp. Its forecast
mean is +19.27 bp after shrinkage. There are 345 training origins in this
state, but earliest-finish interval scheduling finds a maximum of only
**10 non-overlapping label intervals**. The earlier greedy-start diagnostic
retained eight; neither count proves statistical independence. Its calibration
mean is only +1.03 bp, already below entry friction.

At 20:34:15 UTC on November 5, this law causes a nearly 5x long entry. The
policy stays exposed through subsequent losses because its expected holding
loss remains smaller than the cost of exiting, and it performs small
reductions when losses push leverage above the cap. Its final loss consists
of **182.67 gross trading loss + 121.28 fees + 0.56 borrowing**, for
**304.51 lost equity**. This is an overestimated reversal law, not a finding
that position decomposition or the account optimizer caused the loss.
Both forecast diagnostics and costed calibration replay would reject this
variant before the test day.

The corrective experiment uses the existing documented idea of separating
partition learning from law estimation. `trainEventDistribution` now accepts
an explicit estimation population; it does not use those outcomes to choose
splits. The script learns partitions on November 2, purges targets crossing
midnight, estimates kernels from November 3, and keeps November 4 diagnostic.
V407 estimates from 34 successive complete events; v408 uses 2,764 stride
origins on the same day. Both retain joint atoms and the same prior strength,
costs, leverage, order lattice and terminal execution.

Both corrections eliminate the rejected long entry. V408's control shows
that this outcome cannot be attributed to thinning alone. V407's simpler
law has an all-state cash bound through **five event steps**, so H2/H3 cannot
improve entry from cash under that particular marked, decision-price law.
V404 certifies cash through two steps. V408's sufficient bound fails at H1
because a rare state has a -12.03 bp mean; the bound ignores beneficial
restrictions such as borrowing costs, so failure does not prove a profitable
trade. Its actual test replay still stays flat.

Preserve forecast separation, correct accounting, and the strong short-horizon
sign signal. Do not promote the raw slow-context tree, or claim that cash
constitutes profitability. These results support better distribution estimation
over enough completed significant events before expanding Bellman computation.
The larger-event label grid still uses 5/20-second duration boundaries: this
is poorly matched to one-hour events and warrants a controlled duration-target
experiment rather than treating the current class loss as a complete measure
of return/duration quality. None of these one-day screens establishes full
inspector-window performance or resolves the framework comparison.

All **125 focused tests** and workspace typechecks pass. The new checks cover
native slow-context causality, endpoint alignment, four-hour purging, and
invariance of learned partitions when only separate estimation outcomes change.
Artifacts retain the initial failed test, which mutated a future close without
updating its OHLC envelope; the corrected causality fixture restores it before
testing downstream label construction.

## Duration-label control and earlier partition history (v410–v414)

The preceding duration mismatch was a hypothesis, not a diagnosed cause.
`EventClock.durationBinsMinutes` now permits explicit physical boundaries;
`--duration-bins-seconds 300,1800` selects 5/30 minutes in the native screen.
The complete clock is passed through event labeling, mean/size/sign/volatility
reweighting and quadrature. Financial holding and borrowing durations remain
unchanged. Existing default boundaries still represent five/twenty observations.

`event-crps.ts` scores weighted empirical return and duration distributions
with CRPS in basis points and seconds. It prepares sorted prefix sums once per
leaf and scores observations in logarithmic time. Independent direct pairwise
scoring verifies the weighted formula, ties, zero weights, affine scaling and
physical units. CRPS measures marginals; it does not establish correct joint
return/duration/extrema/successor dependence. Class NLL across different label
grids must not be interpreted as improvement on a common target.

V410/V411 repeat v408/v407 with only the duration bins changed. Exact structural
hash comparisons find **identical nodes, full joint kernels and test traces**.
The partition-training day contains 2,760 labels: zero at <=5 minutes, 16 at
5–30 minutes and 2,744 beyond 30 minutes. Both variants remain cash. The label
change alone therefore has no performance effect on these controls.

A bounded follow-up increases `fit-days` from two to four. Partition learning
now uses October 31–November 2, still excluding all inspector windows and
purging boundary-crossing inputs/targets. Distribution estimation remains
November 3, calibration November 4, test November 5. There are 8,520 partition
origins, 2,764 estimation origins, 2,762 calibration origins and 42 complete
test-chain events. Both runs finish in about two seconds including startup.

| Fixed comparison | Default 5/20 seconds (v412) | Explicit 5/30 minutes (v413) |
| --- | ---: | ---: |
| Calibration return CRPS, bp | 22.5479 | 22.5538 |
| Calibration duration CRPS, seconds | 613.7489 | 613.8706 |
| Calibration H1 account return | +1.6028% | +1.6028% |
| Test return CRPS, bp | 23.6371 | 23.1558 |
| Test duration CRPS, seconds | 758.0283 | 758.5315 |
| Test H1 account return | -0.8007% | -0.5999% |
| Test gross trading PnL | +42.21 | +60.02 |
| Test fees | 121.62 | 119.36 |
| Test borrowing | 0.6645 | 0.6481 |
| Test orders / cancellations | 6 / 0 | 4 / 1 |

The expanded partition has 154 / 1,488 / 6,878 labels in the new duration
classes. One one-hour-return cut changes slightly; the other splits remain
identical. Marginal calibration scores are slightly worse with the new bins,
and calibration trades produce identical returns. The smaller test loss is
not a sound reason to select the new bins after inspecting that test.

The bullish state combines elevated 60-second volatility with a preceding
one-hour fall near 136 bp. V412/V413 estimate its mean as +15.62/+16.17 bp
from 46/44 overlapping origins. Earliest-finish scheduling retains a maximum
of **three non-overlapping label intervals** in either estimation population.
The separate calibration state also has only three non-overlapping intervals;
its observed mean is +13.74/+12.93 bp. These counts do not establish independent
observations, and separate split fitting does not remove dependence within the
estimation population. The one-day profit is insufficient support for a strong
reversal law. Test gross gains fail to cover the unchanged execution costs.

V414 reconstructs all populations, verifies immutable reference hashes and
saved sample counts, compares physical-target hashes and replays calibration
from the frozen models. All four physical-target populations match across
v412/v413. `identity.json` records the exact v408/v410 and v407/v411 identities
as well as the v412/v413 difference. The sufficient all-state cash certificate
fails immediately for v412/v413; that is not a proof that their entries are
profitable or that deeper planning cannot improve them.

Preserve the explicit units, comparable scoring, separated fit/estimation,
actual account costs and saved diagnostics. Do not expand a duration-bin
search or promote either new policy. The next forecast experiment should
address support and dependence in law estimation, using additional earlier
completed events or a justified uncertainty model. A deeper-policy experiment
should first profile a small frozen-law decision set before a full replay.
Position decomposition remains an independent policy hypothesis: none of
these forecast ablations tests whether its decision rule helps account growth.

All **128 focused tests** and workspace typechecks pass. The first new test
run had two fixture errors (a one-element feature vector for a twelve-feature
model, and an assumed 60 bp threshold left at the helper's 20 bp default).
The fixtures were corrected; the initial log remains preserved. Tests also
verify duration boundaries, unchanged physical event outcomes, serialization,
recalibration and quadrature class/successor mass preservation.

## Frozen-law native H2 planning and forecast diagnosis (v415–v427)

This experiment retains the v412 forecast, costs and chronological populations.
V412 is the default-duration control; the better test loss of v413 is not used
for selection. Each day starts from cash and settles at its own boundary.
`research-native-event-planning.ts` profiles H2 on the first occurrence of each
calibration leaf plus the first held-inventory state, then performs paired H1/H2
replays with the full joint atoms. There is no forecast fit or compression.

The first H2 probe introduces a short where H1 remains cash. A 16-evaluation
budget leaves a 4,869.90 bp value gap; 128 evaluations reduce it to 3.42 bp.
The search actually converges after 142 evaluations when given a larger cap.
Five selected states all certify within 0.001 bp in under one second each.
This measured cost justifies a full-day replay rather than a broad horizon grid.

The v420 calibration replay with a 512-evaluation cap certifies 25/35 decisions.
V421 at 2,048 certifies 31/35, with identical actions and returns. V422's test
replay certifies 42/43. Rather than replay both days at a larger budget,
v425/v426 recompute the original actions' complete H2 values and refine only
the five unresolved states. They require 2,068–2,230 evaluations, roughly
1.03–1.15 seconds per state after preparation. All original actions remain
unchanged and meet tolerance. V427 matches model and trace hashes, account,
leaf, action, recomputed value and upper bound before joining the certificates.

| Frozen v412 forecast | Calibration Nov 4 | Test Nov 5 |
| --- | ---: | ---: |
| H1 account return | +1.6028% | -0.8007% |
| H2 account return | +8.1734% | -10.9986% |
| Certified H2 decisions | 35 / 35 | 43 / 43 |
| Maximum original-action value gap, bp | 0.0008764 | 0.0009697 |
| H2 gross long PnL | +11.12 | +192.75 |
| H2 gross short PnL | +934.67 | -1,046.14 |
| H2 fees | 125.00 | 242.11 |
| H2 borrowing | 3.45 | 4.37 |
| H2 orders / reversals / cancellations | 9 / 1 / 0 | 23 / 2 / 1 |
| H2 maximum drawdown | 8.66% | 20.38% |

The finite optimization statement is exact in scope: **78/78 original actions
have a numerical upper/lower gap no larger than 0.001 bp for marked-terminal
H2 under the frozen decision-price forecast and actual modeled order lattice**.
It does not prove the market law, stationary control, next-open execution
optimality, or all-window native coverage. The simulator still fixes orders
at a completed close, attempts the next open, and pays actual terminal fees.
A receding H2 replay does not implement one fixed terminal two-event plan.

The short-entry state forecasts -10.63 bp over one event and -16.38 bp over
two events. The longer forecast crosses the unchanged 12 bp entry friction.
Its completed two-event outcome mean is -38.33 bp over seven calibration
pairs but **+11.39 bp over sixteen test pairs**. Another held-inventory state
forecasts -4.66 bp over two events and realizes -3.93 bp on calibration but
+9.57 bp on test. These adjacent pairs overlap and are not independent samples.
The planner spends 1,197.45 test minutes short. Gross short losses dominate
fees, so reducing turnover alone does not explain or repair the failure.

### Bound improvement and its scope

The optional recursive global bound was initially infinite because the old
reachability check excluded exceptional recovery trades only when a maximum
order was too large to reduce exposure. The native 50,000 quote cap can instead
flatten a roughly 50,000 quote position; such a trade restores ordinary leverage
and is already represented in the continuous capped relaxation.

`event-multi-step-upper.ts` now propagates conservative lower/upper equity and
upper absolute-notional/price bounds. For a potentially over-cap account,
absolute notional is in `[L*E_min, N_max]`. A reducing clip has size between
`M_min` and `M_max`, so its residual notional is bounded by
`max(abs(L*E_min-M_max), abs(N_max-M_min))`. If this is no larger than
`L*(E_min-f*M_max)`, with a positive post-fee lower equity, every clip restores
the ordinary cap. The previous too-large-to-reduce check remains sufficient
in its own domain. Between decisions, lower wealth accounts for maximum fees,
return loss and borrowing; a nonpositive lower bound declines certification.
This proves the exclusion inductively without changing permissible trades.

Exhaustive two-event references cover both signs, flat and above-cap accounts,
marked/friction terminals, asymmetric borrowing and clips below twice the
reachable position. Insufficient clips and severe loss/fee cases still reject
the bound. Existing recovery and three-event reference tests also pass.
V423 certifies four cash-start probes using one or two action evaluations;
the expensive first probe falls from about 0.87 seconds to 0.038 seconds after
shared preparation. Held-inventory cases remain harder because the relaxation
omits the actual maximum order restriction. The v424 optional global-bound
replay is a speed diagnostic, not the primary certified replay: 25/35 decisions
meet tolerance at budget 512, and slightly different near-optimal entry lots
produce +8.1707%. It is not substituted into v427.

### Position decomposition and actual margin symmetry

The current `Position management.md` explicitly retains a shared account
allocator and lets local positions contribute value/PnL curves. Independent
per-lot log maximization would change that documented objective. Preserve the
coordinated decomposition and lifecycle accounting; the H2 evidence does not
show that they hurt performance.

A separate exact H1 counterexample tests blind long/short reflection under
actual default costs. Over a deterministic one-hour move of +12.02 bp, a flat
10,000 quote account at price 100 buys 99.88014 units, approximately 1x funded
exposure. Under the arithmetic-reflected -12.02 bp move it optimally stays cash.
Although both borrow rates are 1 bp/day, the funded long borrows nothing while
the short borrows the entire asset amount. Blindly reflecting the long order
has negative expected log growth (-0.0000021641); the correct short action has
zero. Removing both borrowing rates restores reflected optimal actions in the
same fixture. This rejects unchanged-rule sign reflection, not the lifecycle
framework or a correctly transformed full financial problem. The counterexample
is saved in v427 and covered by a focused test.

Preserve the certified account optimizer and conditional decomposition. The
next substantive forecast work should address the sparse, short estimation
history and failure of bearish state means to transfer across days. Deeper
planning alone magnifies that error. Full native inspector coverage and deeper
policy convergence remain unfinished.

All **130 focused tests** and workspace typechecks pass. New checks cover the
recovery-bound extension, margin-symmetry counterexample and causality with
the optional global H2 bound enabled. V427 contains the joined certificates,
forecast diagnostics, counterexample and verification logs.

## Estimation support, day stability and separate sign heads (v428–v434)

The next experiment changes law estimation while holding the v412 state
partition fixed. `reestimateEventTree` uses the shared joint-kernel construction,
keeps nodes/clock/features unchanged, resets stale calibration metadata, and
validates complete atom and label contracts. Its zero-extra-day control v428
reproduces the original model exactly. Tests also cover equivalent explicit
estimation, source immutability, empty-leaf fallback and invalid populations.

V429 adds October 28–30 to the original November 3 estimation data. The
partition-training period October 31–November 2 is excluded from estimation
labels. Every observation predates calibration; all inputs and labels still
exclude every inspector window, including fit windows. Historical input
features can overlap between populations, so this is disjoint-label fitting,
not a claim of statistically independent samples. Calibration remains November
4 and the repeatedly inspected test remains November 5.

V430/V431 replace stride origins with complete event chains, producing the
full two-by-two history/sampling comparison. Prior strength remains 32 in all
runs; thinning changes both the empirical sample population and its weight
relative to that prior. The chain controls therefore do not isolate overlap
alone. Every run takes roughly 1.3–2.4 seconds including process startup.

| Frozen partition / law estimation | One day, stride v428 | Four days, stride v429 | One day, chain v430 | Four days, chain v431 |
| --- | ---: | ---: | ---: | ---: |
| Estimation rows | 2,764 | 11,284 | 34 | 136 |
| Former bearish-state mean, bp | -10.6264 | +3.3421 | -3.7886 | +3.7169 |
| Same state's two-event mean, bp | -16.3762 | +6.4405 | -6.4847 | +7.4440 |
| Calibration return CRPS, bp | 22.5479 | 23.2125 | 22.6196 | 23.1261 |
| Test return CRPS, bp | 23.6371 | 22.4167 | 23.5981 | 22.7418 |
| Calibration H1 return | +1.6028% | 0% | 0% | 0% |
| Test H1 return | -0.8007% | 0% | 0% | 0% |
| Sufficient all-state cash bound, event steps | none at H1 | 2 | 4 | 2 |

The three corrected laws require no costly H2 replay to discover an entry:
their frozen-law cash bounds already certify cash through at least two events.
Their actual H1 replays remain flat. Cash is neither profitability nor evidence
that those forecasts should replace the current research baseline. In particular,
both added-history variants worsen calibration CRPS despite improving test CRPS.

V432 verifies identical physical calibration/test target hashes, tree nodes,
feature names, clocks and costs across all four cases. It also reconstructs
estimation data and compares each conditional law with the unconditional empirical
law on exactly that population. Conditional duration CRPS is better in **all
eight calibration/test comparisons**. For v429, test duration CRPS is 749.11
seconds versus 952.06 seconds unconditionally, while return CRPS is 22.42 bp
versus 23.31 bp. This supports preserving state-dependent duration/magnitude
information, while treating direction separately.

Within-day diagnostics retain only labels completed before their origin day's
end. The original bearish state's raw mean varies as follows:

| Estimation day | Global return mean, bp | State-1 return mean, bp |
| --- | ---: | ---: |
| October 28 | +9.2477 | +13.9766 |
| October 29 | +11.8502 | +11.5110 |
| October 30 | -2.3464 | -2.1193 |
| November 3 | -5.3850 | -10.7726 |

The state follows much of the day-level drift rather than defining a stable
negative return law. Its stride population grows from 21 to 97 maximally
non-overlapping label intervals after adding history; these are still not
independent observations. The rare bullish state grows from only three to five
such intervals. Adding these particular dates does not solve support for that
state. The day table excludes cross-midnight labels for interpretation, whereas
the pooled model retains all complete labels inside each declared estimation
segment; this explains the lower summed daily counts.

### Separate direction-head screen

The existing `event-sign.ts` learner supports generic feature vectors, so the
next small screen reuses it without adding a native policy adapter. Both heads
fit the original 8,520 partition-training origins using the nineteen native
features, a fixed L2 penalty of 0.1, and train-only standardization. They predict
the sign of the **48 bp barrier / one-hour-timeout event**, not next-second sign.
The v429 magnitude/duration/extrema/successor law remains frozen. Fixed blend
weights 0, 0.5 and 1 preserve zero mass and conditional sign-specific laws.

V433 uses ordinary logistic cross entropy. V434 uses absolute-return weights
and the existing conditional-magnitude inversion to convert the resulting
return-weighted score back into a sign probability. This distinction prevents
mistaking a gain/loss balance for an ordinary up probability.

| Head / blend | Calibration sign log loss | Calibration return MSE skill vs zero | Test sign log loss | Test return MSE skill vs zero |
| --- | ---: | ---: | ---: | ---: |
| Frozen v429 law | 0.69756 | -1.40% | 0.68632 | +2.42% |
| Ordinary / 0.5 | 0.69542 | -0.73% | 0.69325 | -0.20% |
| Ordinary / 1 | 0.69842 | -0.79% | 0.71755 | -5.52% |
| Return-weighted / 0.5 | 0.69735 | -0.60% | 0.69513 | -0.89% |
| Return-weighted / 1 | 0.71308 | -1.86% | 0.73378 | -8.52% |

Every head/blend has worse calibration probability loss than the constant
50/50 forecast (`log(2)=0.69315`) and worse calibration return MSE than zero.
Some improve relative to the biased base; that is insufficient evidence for
policy integration. All lose the base's small positive test MSE skill. No
trading return is attributed to these forecast-only screens. They finish in
about 1.6–1.7 seconds each including startup. This does not contradict the
stronger production59/base17 next-active-second predictor: the target horizon
and learner are different.

Prior notes already contain failed broad HMM, adaptive inversion, ridge and
boosting screens (v27–v38 and later history-gate work). Do not repeat a broad
parameter search in response to this result. The next bounded comparison
should test a longer physical market context: native features currently stop
at four hours, while the original minute basis includes a day. That difference
has not been isolated experimentally and is not evidence for choosing minute
candles. A controlled native feature ablation can test it directly while
preserving costs, event targets, fit/estimation separation and scoring dates.

All **130 focused tests** and workspace typechecks pass. The new estimator
has an exact original-model control, validated joint-law reconstruction and
source-hash audits. Broader native coverage, deeper policy convergence and
reliable profitable forecasts remain unfinished.

## Native day context and entry sensitivity (v435–v441)

The controlled feature extension appends exact 43,200s and 86,400s log returns
to the existing nineteen native features. `day-context` declares 86,400 candles
of historical support; the shared endpoint implementation preserves the old
feature prefix. Tests verify endpoint values, future isolation, bad endpoints,
insufficient history, minute-clock rejection, serialization and full-day input
purging. The default fast basis and existing four-hour contract remain explicit
controls. Candle observations and execution remain native one-second.

V435 uses the same October 31–November 2 partition dates, November 3 law
estimation, November 4 diagnostic calibration, and November 5 test prefix as
v412. The 48 bp barrier, one-hour timeout, tree depth, prior, costs and exclusion
rules are unchanged. V436 reconstructs both populations and verifies matching
physical target hashes for all four splits: 8,520 partition, 2,764 estimation,
2,762 calibration and 42 complete test events. Both test replays make 43
decisions, including their boundary-censored tail.

The new partition retains the volatility root and selects the **24-hour**
return at both child splits; the offered 12-hour feature is not selected. The
lower-volatility branch splits at -1.1835 bp, while the higher-volatility branch
splits at -179.5569 bp. Its second state receives zero November 3 estimation
rows, exposing a support problem despite ample partition-training rows.

V437 therefore repeats the already-declared earlier-history ablation: estimate
from October 28–30 plus November 3, while freezing the v435 partition. V438
compares its 11,284 estimation labels with v429 and verifies identical physical
estimation/calibration/test hashes. It also compares both conditional laws with
the same unconditional empirical law. These dates have been repeatedly
inspected; none constitutes a new independent holdout.

| Features / estimation days | Calibration return CRPS, bp | Test return CRPS, bp | Calibration H1 return | Test H1 return |
| --- | ---: | ---: | ---: | ---: |
| Four-hour context / one day, v412 | 22.5479 | 23.6371 | +1.6028% | -0.8007% |
| Day context / one day, v435 | 23.0041 | 24.9692 | +1.9222% | -4.4073% |
| Four-hour context / four days, v429 | 23.2125 | 22.4167 | 0% | 0% |
| Day context / four days, v437 | 23.0090 | 23.4816 | +0.8763% | +4.5825% |

V435 enters a short and stays exposed for 1,363.87 test minutes. Gross short
PnL is -396.38 quote, fees 42.66 and borrowing 1.69, reconciling to -440.73
quote equity change. Its bearish state's forecast is -12.27 bp, versus an
observed +6.90 bp across 23 test events. Longer context alone has not fixed
the wrong-sign forecast.

V437's sole test entry is a 0.26878 BTC long at 01:34:39 UTC, about 1.81x
exposure. It remains open for 1,345.35 minutes: gross PnL +503.16, fees 44.15
and borrowing 0.76, producing +458.25 quote. This favorable held position
does not demonstrate stable next-event prediction. Calibration/test mean-return
MSE skill is -1.83%/-3.69% versus zero. Its calibration CRPS improves over v429,
but test CRPS worsens; the same is true for duration CRPS (603.46 vs 622.15
seconds on calibration, 766.99 vs 749.11 on test). Conditional duration still
beats the unconditional law on both days, supporting preservation of that
information without claiming a reliable directional edge.

The entry state has 406 stride observations but only 20 maximally
non-overlapping label intervals. Of these rows, 399 occur on November 3;
the remaining seven occur on October 30 and admit only one non-overlapping
interval. Its estimated mean is 12.3331 bp against 12 bp one-way costs.

### Remove-day sensitivity and controlled prior weight

V439 removes every estimation label touching one UTC day, re-estimates the
full joint law and prior under the same frozen partition, and solves exact
marked H1 at a common flat account: equity 10,000, price 68,775.99. This is
estimation sensitivity, not an independent validation fold, confidence
interval, or omitted-day strategy backtest. Historical feature inputs remain
unchanged; cross-midnight target labels touching the removed day are excluded.

| Estimation data | Entry-state observations | Mean return, bp | Optimal flat-entry exposure |
| --- | ---: | ---: | ---: |
| All four days | 406 | +12.3331 | +1.8183x |
| Without October 28 | 406 | +12.1855 | +1.0000x |
| Without October 29 | 406 | +12.1574 | +0.9330x |
| Without October 30 | 399 | +11.9386 | 0 |
| Without November 3 | 7 | +12.9533 | +5.0000x |

The entry is not stable enough to promote. V440/V441 additionally replace
stride origins with complete event chains. Both use the same four estimation
days and frozen partition; they contain 136 observations, only eight in the
entry state, all on November 3. Their raw state mean is -0.6674 bp, versus
+13.0742 bp for the stride population.

V440 keeps prior strength 32. V441 sets it mechanically to
`32 * 136 / 11284 = 0.3856788373`, preserving the global prior-to-observation
ratio of v437. This is a declared control, not a tuned prior. Individual leaf
prior fractions still differ because chain and stride populations have
different state proportions.

| Event-chain estimation | Entry-state forecast, bp | Calibration/test H1 return | Sufficient cash horizon |
| --- | ---: | ---: | ---: |
| Prior 32, v440 | +2.8862 | 0% / 0% | 2 events |
| Matched global prior ratio, v441 | -0.4631 | 0% / 0% | 1 event |

Thus the disappearing entry is not explained solely by stronger global prior
weight after thinning. Chain sampling also changes the origin population;
one chain is not an independent or uniquely correct estimate. V441 improves
duration CRPS to 588.49/677.54 seconds but has negative return MSE skill on
both days. Cash results do not constitute profitability or justify promotion.

`v439/comparison.cjs` and its saved JSON verify matched physical outcomes,
feature prefixes, dates, exclusion rules, costs, frozen re-estimation nodes,
account/position PnL reconciliation and the declared prior calculation. Models,
source snapshots, forecasts, traces and sensitivity outputs remain available
in their respective artifact directories. Each fit/replay screen takes roughly
two seconds including startup. **131 focused tests** and all workspace
typechecks pass; logs are saved under v439.

Keep the useful conditional duration information and coordinated account-level
position accounting. Do not promote the profitable two-day variant or spend
more on its Bellman depth yet. The next bounded diagnostic should compare
fixed event-chain starting phases on the same estimation dates, to measure
origin sensitivity before selecting an estimation population. This neither
settles the best candle resolution nor supports discarding position
decomposition. All-window native conditional optimality, deeper convergence
and reliably profitable forecasts remain unfinished.

## Origin overlap, weighted estimation and full-window H2 (v442–v453)

### Fixed chain phases

V442 starts each estimation segment at offsets 0, 900, 1,800 and 2,700 seconds,
without shifting its end or any diagnostic targets. The v437 day-context
partition remains frozen. Each phase is fitted with prior 32 and with the
global prior-to-observation ratio matched to the stride baseline. All eight
H1 replays remain cash on both days. The three nonzero phases have seven
entry-state observations with raw mean +0.1494 bp, versus eight observations
and -0.6674 bp at phase zero. This does not restore the stride entry.

The chains contain 136–137 events and share 92–119 identical labels pairwise.
Their agreement is not independent replication: different starts can reach
the same subsequent boundaries. The phase-zero saved laws exactly reproduce
v440/v441. The sampling-origin discrepancy survives this bounded check.

### Average-uniqueness estimation

The project notes did not supply a specific label-concurrency implementation.
The [Mlfin.py sampling documentation](https://mlfinpy.readthedocs.io/en/latest/Sampling.html#sample-uniqueness)
and its [concurrency implementation](https://mlfinpy.readthedocs.io/en/latest/_modules/mlfinpy/sampling/concurrent.html)
describe averaging inverse concurrent-label counts over each label's lifespan.
This motivates a controlled empirical reweighting; it does not guarantee
unbiased forecasts or fix selection differences between event and stride
origins. No return-magnitude weighting or bootstrap parameter search is added.

`eventAverageUniqueness` integrates inverse concurrency over each completed
label's actual return intervals. A label from candle i to j contains returns
i→i+1 through j−1→j; a shared endpoint alone is not overlap. Sorted endpoints
avoid a dense sample-by-second matrix and allocation across long empty gaps.
Only estimation labels contribute to weights. A direct per-return enumeration
test checks arbitrary overlaps, duplicate labels, touching endpoints and gaps.

`reestimateEventTree` now accepts positive observation weights. Both leaf
observations and the joint shrinkage prior use them. Prior quadrature selects
weighted representatives while retaining each outcome's return, extrema,
duration and successor coupling. Unit weights reproduce old models exactly;
small integer weights agree with explicit replicated observations. Raw model
counts remain observation counts, not weight mass or effective sample size.

V443 reweights the v437 day-context law. The 11,284 observations carry total
uniqueness mass 140.5917; the entry state's 406 rows carry mass 8.0104. These
are weighting totals, not independent observation counts. Its weighted raw
mean drops from +13.0742 to +9.6156 bp. With prior 32 the forecast becomes
+4.8568 bp; matching the original global prior ratio uses prior 0.3987002 and
gives +9.3334 bp. Both H1 replays remain cash. Unlike selecting a favorable
chain phase, this preserves all stride observations while changing their weights.

| Day-context law | Calibration return CRPS, bp | Test return CRPS, bp | Calibration/test H1 return |
| --- | ---: | ---: | ---: |
| Original equal-weight v437 | 23.0090 | 23.4816 | +0.8763% / +4.5825% |
| Uniqueness, prior 32 | 22.9050 | 22.9553 | 0% / 0% |
| Uniqueness, matched global prior ratio | 22.9487 | 23.2441 | 0% / 0% |

V444 applies the same controls to the four-hour-context laws. For the four-day
v429 law, matched weighting improves return CRPS from 23.2125/22.4167 to
23.1478/22.2527 bp, while calibration MSE skill remains negative. This is
forecast evidence, not a profitable-policy result.

The original one-day v428/v412 law provides a more informative policy control.
Matched weighting lowers the bearish state's forecast from -10.6264 to
-5.9957 bp, and its two-event forecast from -16.3762 to -10.8267 bp. Its
bullish mean rises from +15.6231 to +18.3793 bp. **H1 actions and equity remain
exactly unchanged**, including entry/exit quantities, times and account paths.
The two H1 returns remain +1.6028%/-0.8007%. V445 exports this exact weighted
model with source hashes, weights, dates and prior calculation, ready for the
shared native planner. Its resolved prior is 0.4129230; there is no refit on
calibration or test outcomes.

### H2 behavior under the changed forecast

V446 profiles all four first-occurrence leaves and the first held-position
state. All five meet tolerance; the slowest probe takes 0.66 seconds. V447/V448
then replay full calibration/test days under the same weighted law. All 78
decisions certify at the initial budget of 512, with maximum gap 0.0009393 bp.
Each replay takes about 2.1–2.3 seconds after loading/preparation. V449 audits
the numerical gaps, model hashes and account reconciliation.

| Two-day diagnostic | Original H2, v427 | Weighted H2, v447/v448 |
| --- | ---: | ---: |
| Calibration return | +8.1734% | +1.0750% |
| Test return | -10.9986% | +0.2494% |
| Calibration maximum drawdown | 8.6589% | 5.1540% |
| Test maximum drawdown | 20.3806% | 4.5477% |
| Calibration short exposure, minutes | 1,032.98 | 0 |
| Test short exposure, minutes | 1,197.45 | 0 |

The changed forecast removes the adverse short exposure while retaining H1's
long-entry timing. On November 5 the weighted H2 policy buys 0.70374 BTC at
19:55:52, compared with H1's 0.71723 BTC. At 21:34:15 it sells 0.46007 BTC
as the forecast changes to leaf zero, before the next -48.36 bp event. Its
gross long PnL is +142.53 quote versus H1's +42.21, with fees 117.26 and
borrowing 0.33. Calibration loses profitable short exposure as well, so the
two-day result is a tradeoff rather than a uniform improvement.

### One complete inspector window

V450 extends the **unchanged v445 model** through the full
`sharpe-up-7d-2024-11` window, November 5–11 inclusive. Full-window native
planning now records and verifies the additional immutable replay references.
The forecast is still estimated solely from November 3, with its earlier
partition and separate November 4 calibration. No source or fit rule changes.

| Full November window | H1 | H2 |
| --- | ---: | ---: |
| Return | +171.1625% | +59.5348% |
| Final equity from 10,000 | 27,116.25 | 15,953.48 |
| Maximum drawdown | 15.8318% | 13.0644% |
| Fees | 236.94 | 313.18 |
| Orders, including terminal settlement | 19 | 21 |
| Decision events | 357 | 357 |

H2 stays long for 8,884.13 minutes and never goes short. It earns positive
equity growth, but **H1 earns substantially more**. H2's smaller exposure
reduces participation in the trend. The dominant state still predicts a
two-event mean of -6.2711 bp, whereas 253 completed overlapping pairs in this
window average +18.9481 bp. Its bullish state also misestimates two-event
returns: +17.6333 bp forecast versus -4.6175 bp over 13 pairs. These are
diagnostic averages, not independent statistical tests. Deeper planning still
acts on an inaccurate market law; positive realized profit does not cure that.

The initial full replay takes 11.06 seconds and certifies 344/357 actions.
V451/V452 recompute each of the thirteen remaining original actions and refine
its upper bound; all actions remain unchanged. They require 585–987 evaluations
and roughly 0.17–0.29 seconds each after preparation. V453 verifies all
**357/357 full-window actions**, plus **35/35 calibration actions**, within
0.001 bp; maximum gap is 0.0009869 bp. The overlapping one-day test prefix is
not counted again in this 392-decision total.

These certificates concern finite H2 under the frozen decision-price,
marked-terminal model. Actual replay retains next-open execution and
fee-paying settlement. They do not prove stationary or execution-consistent
optimality, or completion across all 28 inspector windows. This is the first
complete native cost-sized-window certificate in this experiment family.

V453's saved comparison script verifies exact zero-phase controls, exact
weighted-model export, unchanged original H1 actions/equity, identical two-day
decision boundaries and costs, full catalog-window boundaries and PnL/position
reconciliation. All **133 focused tests** and workspace typechecks pass; logs
are under v449. No policy is promoted. Preserve causal full-joint estimation,
account-coordinated position accounting, and the verified cap/fee behavior.
The next bounded coverage step should use the same declared fitting/weighting
protocol before a contrasting full inspector window, such as
`regime-down-2022-06`, with every fit outcome preceding that window. Do not
continue tuning the November prefix or choose a horizon using its return.

## Contrasting full window, admissible fitting and order-capacity bounds (v454–v469)

The next window is `regime-down-2022-06`, June 12–18 inclusive (end June 19,
00:00 UTC). Settings remain native completed **one-second candles**, a 48 bp
barrier with a 3,600-second timeout, the 19-feature four-hour context, a
depth-two tree and a frozen weighted full joint outcome law. Financial settings
remain 10 bp fees plus 2 bp slippage per traded notional, 5x leverage, 0.5%
maintenance margin, $5 minimum / $50,000 maximum order notional, 0.00001 BTC
quantity increments, and 1 bp/day borrowing on the relevant debt. These are
experimental assumptions, not a claim about current exchange terms.

### Fit exclusion and source-loading controls

V454 correctly fails before fitting: the immediately preceding estimation
day overlaps another inspector window, leaving zero admissible estimation
samples. Its directory contains a failure record and no trained-model claim.
`eventFitPeriods` now chooses the latest complete fit/calibration block whose
entire feature-history support avoids **every catalog window**, including fit
windows. Only dates and exclusions enter this rule; returns and model scores
do not. For June it moves the block back five days:

| Population | UTC dates | Samples |
|---|---|---:|
| Tree partition | June 2–4 | 8,520 |
| Frozen-leaf outcome estimation | June 5 | 2,761 |
| Diagnostic calibration | June 6 | 2,760 overlapping origins |
| Initial test screen | June 12 | 86 complete chained events |

V455 is the admissible unweighted baseline; v456 applies the already declared
average-uniqueness/matched-prior protocol. Its estimation mass is 31.2166 and
prior strength 0.361801. The mass is an overlap weight, not an independent
sample count. Calibration and test outcomes do not enter the weights.

Loading the unused gap from calibration to test is unnecessary. Native source
loading now merges requested intervals while leaving other gaps omitted. Each
loaded block retains strict one-second validation, and feature/target timestamp
checks reject any crossing between blocks. This change also applies to native
law re-estimation and the forecast, stability and sign diagnostics.

V461 reproduces the original November v412 **model, forecasts, complete trades
and positions exactly**, with no change to its fitting dates. V462 reproduces
the June v455 files exactly while loading **547,202 rather than 964,801**
candles. V464 reproduces v456's weighted model, forecasts, every weight and both
calibration/test trade and position traces exactly. V466 contains the executable
comparison and its successful result. Timing and memory metadata are excluded
from equality; neither compressed array indices nor the date helper changes
the physical data used by these controls.

### Complete replay and numerical certification

Four calibration probes in v457/v458 meet tolerance before full replay is
started. V459 replays the calibration day; v460 extends the identical frozen
forecast through the **entire** seven-day inspector window.

| Interval | H1 return | H2 return | H2 drawdown | H2 orders | H2 cancellations |
|---|---:|---:|---:|---:|---:|
| Calibration, June 6 | -16.4876% | -16.4215% | 22.0857% | 16 | 2 |
| Full test, June 12–18 | +278.5913% | +272.5697% | 49.8737% | 255 | 103 |

H2 has no long exposure, reversals or liquidations. Full-window short exposure
lasts 9,188.97 minutes, gross short PnL is $28,989.74, trading costs are
$1,666.10, borrowing is $66.66, and final equity is $37,256.97 from $10,000.
The shared position ledger reconciles full-window H2 equity to within
$2.8e-9; H1 also reconciles. That validates attribution, not independent
position optimization or canonical entry/exit symmetry.

The original full replay certifies 1,348 of 1,562 actions at a 512-evaluation
budget. Its 214 unresolved actions have maximum gap 1.8903 bp. V463 first
profiles one additional search successfully. V465 increases the budget to
4,096 for the other 213, but 196 still lack certificates after 66.61 seconds
of measured refinement work. Every selected action remains unchanged. This
is evidence to tighten the bound rather than continue increasing budgets.

The cause is a continuation relaxation that permits unlimited next-order
notional. As account equity grows, its proposed future adjustment can exceed
$50,000. Subdividing the current order interval does not remove that future
capacity advantage. `eventOneStepUpper` now takes the tighter of its existing
shadow-price bound and a holding-wealth tangent retaining the maximum order.

For a concave absolute holding value `F(C,Q)` with tangent `(gC,gQ)` at a
candidate post-order portfolio, any future order `u` changes the affine tangent
by `(gQ - P*gC)*u - gC*f*P*abs(u)`. Over `abs(u*P) <= M`, the maximum change is
`M*max(0, gQ/P-gC-gC*f, gC-gQ/P-gC*f)`. Adding this support to the tangent gives
an affine upper bound valid at **every incoming portfolio** at that price.
The candidate point is chosen by clipping the relaxed target to order capacity;
validity comes from concavity and the support maximization, not from assuming
the clipped target is optimal. Ordinary leverage bounds and exceptional
maximum-order recovery handling remain in place.

The new exhaustive regression checks cover both directions, marked and
friction terminals, borrowing, changed incoming equity/exposure, and both direct
and prepared-series holding calculations. They verify global dominance of
executable integer actions and tightness where the order maximum binds.
Existing recovery, ruin and Bellman-reference checks continue to pass.

V467 profiles a previously unresolved case: its gap closes in 46 evaluations,
where 4,096 had left 0.0434 bp. V468 then certifies **all 214 original actions**
without changing any quantity, using **2–125 evaluations each and 4.64 seconds**
of measured work. No backtest path is overwritten. V469 joins the original
trace and refinement hashes, recomputes original action values at identical
accounts, and verifies **1,562/1,562 full-window plus 39/39 calibration actions**.
Maximum final gap is **0.0009283 bp**, below the 0.001 bp tolerance.

Together with November, this covers **two complete native cost-sized windows**
(1,919 full-window decisions), plus 74 calibration decisions. Each window has
its own forecast fitted before its evaluation dates. These are finite H2
certificates under decision-price marked-terminal laws. They do not establish
stationary control, optimal next-open execution, or completion of all 28
non-fit inspector windows. The March 2023 native-data defect remains unresolved.

### Behavior worth preserving and failures to address

Full-window profit does not establish an H2 improvement: H1 earns another
6.02 percentage points with similar drawdown. Both policies hold shorts for
the same total duration. The calibration loss is also retained in the report;
the horizon was not chosen by its more favorable full-window result.

The forecast has a substantial support problem. Estimation leaves 0 and 1
have **zero observations** and use the prior, yet account for 940 of the
1,560 completed overlapping two-event pairs in the full window. The tree
distinguishes these states using four-hour and one-hour historical returns,
but June 5's estimation population does not populate them.

| Leaf | Estimated count | Forecast two-event mean | Full-window realized two-event mean | Completed pairs |
|---|---:|---:|---:|---:|
| 0 | 0 | -2.0788 bp | -6.3543 bp | 671 |
| 1 | 0 | -2.0788 bp | +5.1251 bp | 269 |
| 2 | 2,560 | +4.2660 bp | -13.5070 bp | 282 |
| 3 | 201 | -35.3344 bp | -2.7399 bp | 338 |

On calibration, leaf 3's same -35.3344 bp forecast faces +25.0957 bp realized
over 14 pairs. These overlapping descriptive samples do not constitute an
independent accuracy estimate. They show why maximizing the forecast's value
can retain a losing short through a rebound. The full-window event-boundary
drawdown runs from June 15 09:46:24 to June 16 01:14:46 UTC; intrabar replay
drawdown is slightly worse than this boundary-only diagnostic.

V469's executable behavior audit reconstructs all next-open rejections from
saved quantities and actual prices. **All 103 H2 cancellations, and all 106 H1
cancellations, violate the leverage constraint at the next open.** None is
caused by minimum/maximum order notional in this replay. That execution issue
is separate from the maximum-notional continuation-bound issue above. The
optimizer chooses against the decision price, whereas fills are checked at
the next open. A certificate for the former does not make those orders optimal
for the latter. Do not silently add a price buffer calibrated on this window.

Preserve the account objective, complete joint outcomes, explicit constraints,
cheap certified no-trade decisions, and reconciled position attribution. The
user's decomposition remains an evidence-tested starting point: the earlier
28-window ledger comparison preserves every action and equity path, while
independent per-position log maximization would change the account objective.
This June result gives no evidence that merely removing the ledger would
improve returns.

The next coverage work should retain the declared fitting/weighting protocol
and use the improved capacity bound on the remaining windows. Before treating
deeper results as execution-optimal, the conditional model must represent
next-open price/feasibility, or its execution policy must be compared under
the same forecast and account objective. Forecast improvement should address
empty leaf support and sign/continuation stability on admissible earlier data;
neither hindsight tuning of June nor an unconditional return to a preferred
framework answers those problems. All **136 focused tests** and workspace
typechecks pass; logs and exact source controls are saved under v466.

## Native coverage, recovery pruning and the March availability defect (v470–v476)

The suite runner now orchestrates the current native protocol through the
existing fit, re-estimation, probe, replay and certificate-audit tools. Children
run sequentially, release their candle arrays on exit and retain complete
logs and source/model artifacts. It does not retry failed stages silently or
select windows by return. The older raw-run suite remains reproducible from
its saved source snapshot; the working runner implements the current protocol.

V470 fits and screens all **28 non-fit inspector windows** in **88.40 seconds**.
Every window uses the declared 48 bp / 3,600-second event clock, native 1s
candles, four-hour context, separate partition/estimation days, uniqueness
weights and matched prior. V471 verifies every date exclusion, shared setting
and model hash. Its November and June models, forecasts, all weights and both
calibration/test traces exactly match v445/v456. These prefix screens alone
do not prove full source availability or deeper control.

V472 profiles calibration leaf occurrences and the first held account before
each full replay. The gate requires convergence within 0.001 bp at 512
evaluations and at most two seconds per probe. It excludes the two previously
certified full windows from duplicate replay. Of its remaining 26 requests,
24 obtain full certificates, July 2022 initially fails the probe gate, and
March 2023 fails the full native-data validator. The batch takes 782.99 seconds,
including probes, full H1 controls, H2 replays and audits.

### A second justified bound improvement

The July choppy probes spend unnecessary work on exceptional maximum-order
recovery bounds. Some candidate clips restore the ordinary leverage cap at
**every** account in a root interval; ordinary continuation bounds already
cover these. For each maximum-order clip, check positive post-fee equity and
the leverage cap at all cash/sign/borrowing critical endpoints. Between those
knots, cash and equity are affine, and absolute notional divided by positive
equity has its maximum at an endpoint. A clip that passes everywhere needs no
separate exception. Clips that can remain above the cap retain the exceptional
bound, including severe-loss cases.

V474 first tests an isolated source copy against exhaustive H2 on 84 small
problems spanning marked/friction terminals, both signs, borrowing, leverage
1/5 and maximum order sizes below/above recovery requirements. All match their
references. The four real July probes then take 0.10, 0.05, 0.26 and 0.02
seconds; previously one took 17.30 seconds without convergence. All four now
meet tolerance. One selected lot amount changes slightly within the numerical
tolerance; this is not claimed as a realized-return improvement.

The fix is applied to `event-two-step.ts` and included in the main regression
suite. The running batch's already loaded processes retain their original
solver; later children record the tightened source. V476 records the initial
root-search source hash per replay: 18 full windows use the added pruning and
9 retain earlier sources, including the two previously certified windows.
Every certificate refers to its own saved replay/refinement artifacts. Forecasts,
costs, feasible action sets and the Bellman objective do not change.

V475 retries July's computation after the fix. All **344 full-window decisions**
certify within tolerance in a 68.38-second orchestration run. Its return is
**-61.98%**, versus H1 -66.03%, with 65.14% drawdown. Passing the numerical
gate does not make this forecast profitable.

### Full-window result and scope

V476 joins the 24 new certificates, July's successful retry, and the two earlier
full-window audits. It checks unique catalog IDs, exact interval endpoints,
frozen model hashes, trace hashes, clock/features/costs and PnL/position
reconciliation. The resulting coverage is **27/28 windows and 9,219 full-window
actions**, with maximum gap **0.0009972 bp**. Calibration and overlapping test
prefix decisions are not added to this total. Separate catalog windows can
overlap in time; their reset accounts are not independent observations or a
single compounded portfolio.

H2 has **7 positive, 16 negative and 4 cash windows**. It earns more than H1
on 6, less on 16, and the same on 5. Worst drawdown is 66.03%. The exact table
is also saved under `event-native-barrier48-coverage-audit-v476/table.md`.

| Window | H1 return | H2 return | H2 drawdown | Certified decisions |
|---|---:|---:|---:|---:|
| sideways-churn-2022-07 | -66.03% | -61.98% | 65.14% | 344 |
| sideways-churn-2022-05 | 0.00% | 0.00% | 0.00% | 441 |
| sideways-churn-2021-12 | -17.21% | -17.15% | 35.13% | 400 |
| sideways-churn-2021-09 | -1.86% | -1.71% | 36.96% | 421 |
| regime-up-2023-03 | 187.10% | 268.07% | 42.88% | 576 |
| regime-flat-2026-04 | -13.55% | -13.56% | 23.41% | 204 |
| regime-down-2022-06 | 278.59% | 272.57% | 49.87% | 1562 |
| shape-up-low-2024-02 | -11.47% | -31.80% | 34.56% | 89 |
| shape-up-high-2022-06 | -46.26% | -46.70% | 66.03% | 464 |
| shape-down-low-2023-06 | 0.00% | 0.00% | 0.00% | 97 |
| shape-down-high-2022-06 | 38.63% | 36.61% | 47.28% | 999 |
| shape-flat-high-bias-2021-10 | -7.12% | -6.23% | 34.63% | 198 |
| shape-flat-high-bias-low-2025-02 | -2.99% | -2.96% | 13.11% | 81 |
| shape-flat-low-bias-2024-07 | 0.00% | 0.00% | 0.00% | 166 |
| shape-flat-low-bias-low-2025-07 | -3.40% | -3.85% | 11.15% | 74 |
| shape-flat-mid-bias-2024-01 | 0.00% | 0.00% | 0.00% | 196 |
| shape-flat-mid-bias-low-2023-09 | -0.84% | -1.73% | 8.74% | 76 |
| sharpe-up-3d-2024-11 | 53.71% | 25.22% | 8.55% | 156 |
| sharpe-up-3d-2023-12 | -44.16% | -44.87% | 48.07% | 124 |
| sharpe-down-3d-2026-06 | -6.87% | -51.77% | 52.57% | 134 |
| sharpe-down-3d-2023-03 | -34.63% | -34.63% | 41.70% | 120 |
| sharpe-up-7d-2023-12 | -54.03% | -56.12% | 59.36% | 243 |
| sharpe-up-7d-2024-11 | 171.16% | 59.53% | 13.06% | 357 |
| sharpe-down-7d-2023-03 | -38.25% | -38.42% | 46.58% | 249 |
| sharpe-down-7d-2026-06 | -6.40% | -49.41% | 51.64% | 235 |
| failure-down-3d-2022-06 | 170.50% | 167.92% | 20.79% | 486 |
| failure-down-7d-2022-06 | 204.91% | 201.34% | 28.14% | 727 |

Three concrete behavior diagnostics explain why no policy is promoted:

- In June 2026's three-day downtrend, both horizons hold longs for 4,157.65
  minutes. H2 increases exposure: gross long loss grows from $674.59 to
  $5,042.27 and costs from $12.02 to $126.47. Its dominant state predicts
  +11.1478 bp over two events, versus -15.0506 bp over 94 completed pairs.
- In February 2024's rising window, H2 extends short exposure from H1's 397.18
  minutes to all 4,320 minutes. The dominant state predicts -13.2778 bp over
  two events, whereas 61 completed pairs average +20.6900 bp.
- In March 2023's rising window, H2's extra long exposure is productive:
  +268.07% versus +187.10%. The dominant state's +21.4822 bp two-event forecast
  has the correct sign against +16.2999 bp realized over 262 pairs. Retain
  evidence that larger exposure can help when the forecast is informative;
  do not apply a universal exposure reduction solely from losing windows.

These are overlapping descriptive pairs, not independent significance tests.
Empty estimated leaves remain material: they serve 73.66% of decisions on the
three-day June 2022 failure window and 62.26% on its high-churn down-shape window.

### The remaining data window and next required work

`sideways-churn-2023-03` fails on the official early-close candle at March 24,
12:39:41.646 UTC. V473 compares the saved verified kline archive with the
existing official-archive-derived aggregate-trade stream. The stream contains
1,142,284 aggregate records with no aggregate-ID gaps. All **4,818 missing
kline seconds contain zero trades**. The full no-trade interval is longer:
**11:27:24–14:00 UTC, 9,156 whole seconds**. It contains 4,338 exchange-published
zero-volume candle rows before the missing kline section. The last recorded
trade before it is 11:27:23.146 and the first afterward is 14:00:00.062.

The [archived Binance maintenance-complete announcement](https://www.coincarp.com/exchange/announcement/binance-813a31506e9f478ea8c1058b425df87a/)
states that trading resumes at 14:00 UTC. Its original Binance link currently
redirects to the announcement index. The notice supports the resumption time;
it does not establish the precise halt onset, hypothetical market-order fills,
or margin liquidation behavior. The trade stream and candle provenance provide
the separate evidence for the observed no-trade intervals.

No source was overwritten and strict validation remains enabled. Simply filling
missing candles would permit hypothetical fills during a venue halt and would
miss the earlier published-but-untraded period. The next implementation must
distinguish price observations, carried marks and execution availability;
prevent fills while unavailable; preserve elapsed borrowing and resumed-price
risk; and avoid revealing the future halt end to the policy. Unexpected gaps
must continue to fail. Only then can this remaining window be scored under a
declared, consistent execution model.

The broader objective remains unfinished. These certificates are finite H2
under decision-price marked-terminal forecasts, while execution occurs at the
next open. They do not establish a full-depth policy fixed point, optimal
next-open actions, or canonical entry/exit symmetry. Preserve the accounting
decomposition and account objective while testing decision frameworks by regret,
equity and computation.

One fitting assumption also deserves a separate later ablation: the original
minute suite purged non-fit evaluation windows from training, whereas the native
pipeline purges every catalog window, including `fit-*`. This makes July 2025's
forecast 107 days stale. That conservatism is an implementation choice, not a
demonstrated requirement for forecasting performance. Do not retroactively
change the frozen laws in this coverage audit; evaluate a chronology-respecting
training rule separately while retaining all non-fit evaluation exclusions.

All **137 focused tests** and workspace typechecks pass, including the new
84-case recovery regression. Logs are saved under v471. Both orchestration
processes finished successfully; all per-window failures are retained explicitly.

## Explicit March market availability and evidence-based framework choice (v477–v480)

The user's criterion remains account equity under the account utility objective.
Position decomposition is a starting hypothesis, to retain, simplify or replace
based on a controlled comparison. Existing v389/v394 evidence establishes exact
lifecycle accounting equivalence, not an advantage or disadvantage for the full
position decision framework. The documented coordinator already handles shared
constraints and account utility; do not mischaracterize it as independent
per-position log maximization. Compare a proposed implementation at the same
forecast states/accounts against certified account action values, then replay it
with the same forecast, clock, costs and execution. Record utility regret, equity,
fees, failures and runtime. Equal actions with easier or faster computation can
also justify the representation. Wrong forecast signs and blind margin reflection
counterexamples do not by themselves reject this coordinated framework.

### Replay contract and controls

`event-market-availability.ts` builds an explicit replay-only marked view from
unchanged source rows. The manifest in
`event-native-march-availability-checks-v477/availability.json` declares no
execution over March 24, 2023, **11:27:24–14:00 UTC**. It hashes the immutable
candle reference, aggregate-trade reference, verified official kline archive
and v473 evidence report. The no-trade stream and archived resumption notice
motivate this scenario; they do not prove exact venue halt onset or exchange
margin/liquidation rules.

All 9,156 seconds in that interval carry the last observed mark, 28,080, and
are explicitly tagged as carried. This includes exchange-published zero-volume
rows as well as missing/partial rows; zero volume outside the declared interval
does not imply unavailability. Contradictory trades or prices, unexpected gaps,
unordered rows and malformed available seconds fail. Starting inside a closure
requires the actual preceding observed second as an anchor. No immutable source
is overwritten; default strict loading still rejects the original partial candle.
The replay-only view is rejected as ordinary observed training data.

The controller learns unavailability from the latest completed second. It then
waits until an available second completes, without receiving a scheduled future
reopening time in its predictor or Bellman search. An ordinary event is interrupted
when the first unavailable second completes. A market order attempted at an
unavailable next open is canceled without a fee or deferred fill. Per-second
borrowing, local-mark maintenance checks, and reopening gap/intrabar PnL continue.
Terminal inventory remains marked and unsettled if execution is unavailable or
the replay ends exactly at reopening without an observed available price; it is
not reported as dust or a fictional sale at the carried mark.

Wait intervals retain account and position reconciliation separately from ordinary
decision traces. They are not assigned zero expected utility or counted as H2
value certificates. Forecast diagnostics omit interrupted events and pairs across
a wait. The audit checks continuous account-time coverage, source/model hashes,
no unavailable fills, unsettled inventory and lifecycle reconciliation.

V477 verifies 86,400 marked-view seconds, exactly 9,156 tagged carries, and exact
identity of every available row to the source. All **30** unaffected March
calibration decisions, positions and account metrics match the original v472
control, both with and without the explicit scenario. Eight new tests cover
source integrity, anchors, cancellation/no deferral, borrowing, reopening risk,
terminal boundaries, future-prefix invariance, and H2 wait/certificate separation.
All **145 focused tests** and workspace typechecks pass. The terminal-reopening
edge guard was added after the full replay; its end timestamp is outside the
closure, so that guard does not alter its saved actions or settlement. Replay
source snapshots retain the exact code used for their results.

### Frozen-law full-window result

V478 uses the unchanged v470 March model
`41971eb92e912d6f46c9521869ebafd3cdf6b1766c60894730a0c9e89ec66ca3`,
native context19 features, 48 bp / 3,600-second events and identical costs.
H2 takes **10.48 seconds**; H1 plus H2 take **11.75 seconds** excluding loading.
V479 verifies all **410** ordinary H2 decisions within **0.0009779 bp**.

| March full window | H1 | H2 |
|---|---:|---:|
| Return | -6.72% | -10.97% |
| Drawdown | 33.66% | 34.60% |
| Orders, including settlement | 77 | 176 |
| Canceled orders | 18 | 50 |
| Gross long PnL | -$434.81 | -$722.77 |
| Fees and slippage | $210.02 | $346.17 |
| Borrowing | $26.78 | $27.70 |

Both hold long exposure throughout the seven days. Neither happens to attempt an
order at the halt onset, so unavailable-order rejection is established by the
synthetic tests, not claimed as an observed historical cancellation. H2 carries
1.7727 BTC through the wait. It pays $0.4208 during the unavailable marks and
retains the loss on the first resumed candle. The account falls from $10,070.97
before the wait to $9,903.90 after the resumed second; no trade occurs inside it.

The bullish leaf 3 forecasts **+21.4822 bp** over two events, while 85 contiguous
completed pairs average **-8.5700 bp**. This is descriptive evidence of a forecast
failure on this window; overlapping pairs are not independent samples. Extra
long exposure and turnover worsen realized wealth despite certified H2 action
values. It does not establish that deeper optimization or position decomposition
is intrinsically harmful.

V480 joins the prior 27-window audit and this one declared scenario, preserving
the distinction. It verifies **9,629 optimized actions** with maximum gap
**0.0009972 bp**, plus one separately audited forced-wait segment. H2 has
**7 positive, 17 negative and 4 cash windows**; relative to H1 it is higher on
6, lower on 17 and equal on 5. Worst drawdown remains **66.03%**. The full table
is `event-native-march-availability-coverage-v480/table.md`. No profitability,
stationary/full-depth optimization or exact historical exchange-execution claim
is made. Coverage of the research intervals is now available under the stated
assumptions; deeper conditional optimization and a matched position-policy test
remain outstanding before forecasting changes are promoted.

## Exact replay transitions and unchanged-law enrichment (v481–v482)

The prior turn completed interval coverage under declared assumptions. This turn
addresses a concrete obstacle to conditional policy optimality: the optimizer and
replay use different account transitions. This is a reason to fix their contract,
not to abandon the shared-account position framework. `Position management.md`
already requires event compression to preserve controlled rewards, future account
state and execution constraints, and permits local scenario PnL curves coordinated
under one account utility.

### What the original event summary omits

The old atom retains close return, duration, extrema and successor state. It lacks
the first opening gap, which changes the price and feasibility of a committed
base-quantity order. It also does not exactly reproduce the simulator's borrowing:
quote debt compounds each second, while the short's quote-denominated interest
charge depends on the open price of each second. Two paths with identical return,
duration, extrema and first open can therefore have different terminal wealth.

`event-execution-path.ts` compiles sufficient statistics for the research replay's
constant inventory between decisions. Let Q and C denote asset units and quote
cash after an accepted first-open trade and its friction. With per-second rates
bL and bS, N seconds, opens O_j, and final price P_N, terminal marked wealth is:

- Borrowed long: C(1+bL)^N + Q P_N, where Q > 0 and C < 0.
- Funded long or cash: C + Q P_N.
- Short: C + Q(P_N + bS sum_j O_j), where Q < 0.

The summary also stores min_j L_j/(1+bL)^j for the borrowed-long maintenance
inequality, and max_j[(1+m)H_j + bS sum_{k<=j} O_k] for the short inequality.
Prices are normalized to the decision close; the coefficients are bound to the
saved cost parameters. Opening-gap maintenance is checked on the old inventory
before any proposed order. Request acceptance uses the actual first-open price,
notional/quantity limits, fee-solvency and existing replay leverage rule. Rejected
requests leave the old inventory and pay no execution fee. Market settlement
preserves minimum-size dust and unavailable terminal inventory.

The compressed evaluator recovers terminal equity, actual requested-order fills,
interest and liquidation occurrence. On ruin it intentionally does not invent an
intra-event liquidation timestamp or interest accrued before that timestamp. It
does not compress drawdown or permit new discretionary actions inside an event.
Those remain separate requirements if a future policy needs them.

The model remains the research simulator's financial contract. The current
[official Binance filters](https://github.com/binance/binance-spot-api-docs/blob/master/filters.md)
distinguish market quantity filters and market-notional reference/average prices.
The replay's fixed lot limits and next-open notional checks are explicit modeling
assumptions; this work does not turn them into an exact historical exchange model.

### Full saved-path audit

`audit-native-event-execution.ts` loads the v480 coverage manifest, verifies its
saved audit/model/trace/reference hashes, and evaluates each committed request on
the same realized event used by the saved per-second replay. Future realized
paths enter this diagnostic only, never an order-selection forecast.

`event-native-execution-path-audit-v481` covers **all 28 intervals**, both H1 and
H2, **19,258 ordinary decisions**, and the two policy-specific March wait records.
It reproduces fills, cancellations, post-event inventory, terminal settlement and
account equity. Maximum equity error is **$4.5166e-8**; maximum event log-growth
error is **3.5111e-8 bp**. Loading, compilation and all comparisons take **13.53 s**.
The input 27 strict-source windows and one March availability scenario keep their
original qualifications; overlapping windows are not independent samples.

Using the *same realized paths*, the original planned-trade transition disagrees
with execution by as much as **618.1179 bp** of event log growth for July H1,
and **446.9974 bp** for July H2. These are accounting/execution-transition errors,
not forecast errors or attainable profit improvements. The largest cases are
canceled requests that the planning transition treats as executed. Errors can
have either sign; cancellation occasionally preserves more wealth.

After conditioning on the actual fill and its first-open account, the largest
simple-borrowing discrepancy over all these replays is only **0.0006511 bp**.
Preserve the more complete borrowing summary for correctness, but prioritize
order acceptance over further numerical precision work on that small difference.

### Enriching the fixed empirical law without changing its old forecasts

`reestimateEventTreeWithSources` exposes the actual estimation-row index for
every empirical atom and every weighted prior representative. The prior already
selects whole observed rows, so adding execution statistics does not require
inventing independent gaps, replacing prior atoms or altering return probabilities.
The existing re-estimator calls the same implementation and retains its output.

`compile-native-event-execution-law.ts` reconstructs only the saved estimation
periods and their feature history, with the same purges and uniqueness weights.
It verifies the entire reconstructed model, weights, and return/duration/extrema/
successor projection against the original, then saves indexed controlled paths.
Calibration and test labels do not enter this enrichment. The enriched financial
transition differs from the old approximation; its old forecast projection is
bit-for-bit identical.

V482 compiles three bounded controls:

| Saved window law | Estimation paths | Compilation time | Probe result |
|---|---:|---:|---|
| July 2022 choppy | 2,847 | 0.40 s | Three of four accounts admit a better tested request |
| November 2024 seven-day uptrend | 2,764 | 0.47 s | None of five tested accounts improves |
| March 2023 choppy | 2,825 | 0.60 s | None of four tested accounts improves |

Probes are the first calibration occurrence of each state and the first held
account, where distinct. Requests are fixed in advance: hold, the old H1 request,
half that request, and its +/-1, +/-2 and +/-5 lot neighbors. Each is evaluated
against the complete enriched empirical mixture, not the future calibration path.

At July's initial cash account, price 21,254.67 and equity 10,000, reducing the
request from **2.33839 BTC to 2.33834 BTC** lowers modeled rejection mass from
**41.2917% to 28.4129%**. Expected one-event log growth increases from
0.00334192764 to 0.00447123770, an improvement of **11.2931 bp**. This is direct
evidence that next-open execution changes the preferred request without a change
to the fitted return probabilities. The best size over the full request lattice
has not yet been found.

Two held July accounts prefer waiting by **0.9246** and **2.4809 bp** among these
candidates. Both have exposure above 5x after price movement. This exposes another
action-contract difference: the old optimizer forbids a zero request above its
leverage cap, while the replay can hold until maintenance fails. The enriched
evaluator explicitly follows the replay's entry/recovery-cap semantics. The two
wait improvements must not be reported as a comparison under identical old and
new feasible sets. A strict continuous leverage ceiling would instead require a
different replay and forced-recovery rule. The flat-account sizing result does
not depend on this distinction.

Four new tests include an independent per-second cash ledger over **1,728**
path/account/request cases, a same-old-summary/different-interest counterexample,
compounding/settlement edge cases and provenance for coincident return summaries.
All **149 focused tests** and workspace typechecks pass. No saved policy action,
forecast distribution or historical return is replaced by these diagnostics.

### Next required implementation

Build the global one-event request optimizer against the enriched mixture and
validate it against exhaustive small lattices. Order acceptance introduces
outcome-specific regions: split at fee, borrow-sign, minimum/maximum-order and
leverage/recovery boundaries, retain rejected-order holding branches, and optimize
the account's expected log wealth within each region. Use bounded profiling on
the existing calibration accounts before full replays. The fitted probabilities,
joint path correlations, global account constraints and all four trade directions
must remain shared; a local position proposal is a candidate in that joint
objective, not a replacement per-position utility. Then extend the consistent
transition to deeper Bellman recursion. Candidate probes and old H2 certificates
do not certify this new execution-aware optimum or reliable profitability.

## Global execution-aware one-event search (v483–v490)

The objective remains expected log account wealth. Position decomposition is a
starting hypothesis to judge by conditional action regret, realized account
equity, and computation under matched laws and execution. The earlier accounting
ablation tests lifecycle bookkeeping only. It is not evidence that local
scenario-PnL proposals with a global coordinator help or hurt the policy. The
present change fixes action valuation shared by either architecture.

### Global request search and its scope

`event-execution-one-step.ts` searches signed integer base-quantity requests.
Each empirical outcome applies its own next-open acceptance checks and leaves
old holdings unchanged when the request is rejected. Zero request remains
admissible above the entry cap, subject to maintenance, as in the actual research
replay. This last rule differs from the old optimizer's feasible-action contract.

The search is finite: requests beyond the maximum quantity that any opening
price can accept all fail and are equivalent to waiting. It partitions that
lattice at every outcome's minimum/maximum order, fee-solvency, leverage/recovery,
inventory-sign, borrowing-sign, and terminal-dust boundary. Numerical neighbors
of each boundary are evaluated directly. Inside each remaining region, outcome
acceptance, fee signs and funding regimes are constant. Terminal wealth and
maintenance inequalities are affine in the requested lot count. Their positive
intersection gives the surviving interval; expected log wealth is concave there.
Discrete derivative bisection and neighboring-lot checks find the best request
in each interval. The global best includes all boundaries and zero. No
continuous target is silently rounded into a different chosen trade.

This construction provides a global **H1 request optimum under the empirical
research law**, with floating-point boundary guards. An outcome that liquidates
the old position before execution makes every request infeasible. Marked and
market terminal modes preserve their respective settlement semantics; the
historical receding policy uses marked H1 and the unchanged market settlement at
the actual backtest end. It has not optimized the full remaining calendar
horizon, stationary growth rate or arbitrary event depth.

The replay receives the selected request before the next open. Trace order
equity/exposure explicitly describe the pre-order account; reported requested
turnover and cost at the decision price are diagnostics. Confirmed quantities,
fees, cancellations and position changes are recorded only after execution.
Changing an unseen next opening price leaves the committed request unchanged.
The original joint return/duration/extrema/successor projection is checked exactly
before execution-aware replay. Mixing quote requests or a changed forecast into
this mode is rejected.

### Validation and bounded computation

Three new focused tests cover **960 exhaustive small problems** with marked and
market terminals, two leverage limits, random three-path mixtures, eight account
inventories including dust and above-cap states, variable gaps, funding and
availability. The global solver agrees with brute force within 1e-11 in value;
its chosen value also agrees with the separate full transition evaluator. Other
checks cover all-unavailable openings, cap-violating holds, inconsistent laws,
and future-opening causality. All **152 focused tests** pass.

V483 first solves the July cash calibration account globally. It requests
**2.33796 BTC**, with zero modeled rejection mass and expected log growth
**0.0071498680**. The old 2.33839 BTC request has 41.29% rejection mass and
value 0.0033419276. The improvement is **38.0794 bp of expected log utility**,
not a realized profit. The initial solve evaluates 405 orders in 0.095 seconds.
V484 completes the other twelve predeclared calibration leaf/held probes.

V485 runs three complete calibration intervals with identical old controls:

| Calibration law | Decisions | Old H1 return | Execution H1 return |
|---|---:|---:|---:|
| July 2022 choppy | 67 | +41.95% | +44.04% |
| November 2024 uptrend | 35 | +1.60% | +1.74% |
| March 2023 choppy | 30 | -1.40% | -6.15% |

The slowest probe spends about a second repeatedly constructing per-path
diagnostics. Account/opening constants are now prepared once and candidate
scoring uses the same scalar transition without allocating those objects. V486
compares all thirteen full results, including requested quantities, values and
search counts, exactly against the saved pre-specialization implementation.
Three alternating paired runs reduce the sum of per-probe median times from
**2,085.82 ms to 176.53 ms**, about **11.8x**. The largest 2,403-atom probe falls
from 984.61 to 78.54 ms. These are local timings, not a deployment guarantee.
The 152 tests also pass after specialization.

V487 completes the first three full intervals. V488 independently checks all
1,243 calibration/full decisions. V489 then compiles, profiles, replays and audits
the remaining 25 windows, with a computation-only gate: all predeclared accounts
must solve completely, finitely and below two seconds each. Returns do not select
windows or determine this gate. Every window passes. Its process exits zero.

### Full native interval coverage

V490 joins all 28 windows, verifies unique catalog membership and endpoints,
saved source/law/model/trace/audit hashes, exact old joint-law projection, and
**exactly unchanged old H1 control returns** against v480. It retains **27
strict-source windows plus one explicit March availability scenario**. All
**9,629 ordinary decisions** are complete and finite; the one forced-wait segment
is separately reconciled. Every chosen value reproduces exactly using the full
path evaluator, and beats or ties the old H1 request and waiting at the same new
account. The largest realized equity reconciliation error is **$3.033e-8**.
Beating two alternatives alone is not a globality proof; that claim depends on
the partition/concavity construction and exhaustive tests above.

Full execution replays total **195.06 seconds**, evaluating 2,819,050 candidate
orders. The compact per-window table is
`data/benchmarks/event-native-execution-one-step-coverage-v490/table.md`.
Results are **8 positive, 19 negative and one no-fill window**. Relative to the
old H1 controller, returns are higher on **10**, lower on **17**, and equal on
**one**. The worst drawdown rises to **88.42%**. Windows overlap and reset their
accounts; their returns must not be compounded or treated as independent fresh
holdouts.

| Full window | Old H1 | Execution H1 | Execution drawdown |
|---|---:|---:|---:|
| July 2022 choppy | -66.03% | -57.55% | 61.07% |
| May 2022 choppy | 0.00% | +35.51% | 28.74% |
| March 2023 choppy, closure scenario | -6.72% | -26.81% | 40.28% |
| November 2024 seven-day uptrend | +171.16% | +18.91% | 33.59% |
| June 2022 regime downtrend | +278.59% | +169.43% | 62.33% |
| December 2023 seven-day uptrend | -54.03% | -82.40% | 88.42% |

### What the policy actually changes

There are **6,504 nonzero requests, 5,644 cancellations and 904 fills/orders
including terminal settlement**. Of the nonzero requests, **5,813** have at least
90% rejection probability under their fitted law; average modeled rejection mass
is **91.47%**. One zero-return window still makes 61 unsuccessful requests, so
its absence of profit is not a deliberate cash-policy success. At the newly
visited accounts, the mean expected one-event gain over the old request rule is
**9.5656 bp**. This is a conditional value diagnostic, not a return attribution.

The optimizer can use acceptance as a condition on the first opening price.
Within a finite empirical mixture, gaps near order limits select small subsets
with favorable conditional returns. This can raise model utility while producing
many rejected requests and very different holding paths. The replay assumes
next-open notional checks; it is still a declared research execution model, as
qualified in the prior section. Do not promote this conditional-execution effect
as live-exchange alpha without separately validating that contract and forecast.

November illustrates the loss of useful behavior. The original policy spends
almost all its exposure long. The new one holds shorts for 2,356 minutes,
accumulates **-$5,241.65 gross short PnL**, and pays **$671.40** in friction versus
$236.94 before. Gross long PnL is still +$7,830.41. July's short holding loses
$1,437.80 gross even though total return improves. March's gross long PnL turns
positive (+$2,906.32), but shorts lose $2,323.26 and friction rises to $3,242.58.
These are descriptions of the realized path, not grounds to remove shorts using
the already-seen test results. Preserve correct shared-account constraints,
execution reconciliation and beneficial directional persistence where justified
by the model; inspect the conditional forecast and continuation value that cause
costly reversals and high-rejection requests.

### Next conditional optimization step

The consistent H1 baseline is complete. Deeper Bellman recursion must now use
these same next-open transitions and successor accounts. Old H2 concavity bounds
cannot be reused without proof: outcome-dependent acceptance makes the global
request value discontinuous. Start with bounded calibration action evaluations
and verified continuation bounds/caches. Do not expand thousands of paths times
hundreds of candidate orders without a measured productivity gain. Keep the
forecast frozen for this phase. A matched lifecycle-proposal/global-coordinator
policy test remains separate; accounting equivalence and these mixed H1 returns
do not establish whether that decision framework helps. Neither finite H1 nor
the earlier transition-inconsistent H2 work satisfies full-depth optimality or
reliable profitability.

## A bounded execution-consistent Bellman backup (v491)

`probe-native-execution-bellman.ts` fixes July's initial calibration account and
three root candidates before looking at any realized calibration path: zero,
the saved old H1 request, and the new global H1 request. For each of the 282
first-event outcomes it applies the exact controlled transition, updates account
equity, price and exposure, then solves global execution-H1 on the outcome's
successor leaf. The backup is

`Q2(s, a) = sum_i p_i [log(E_i / E) + V1(s_i)]`.

The first event and all continuations use the same compiled v482 law and cost
contract. Second-event requests are conditional on the first completed event,
not on its unobserved future. No calibration labels enter the calculation. A
45-second per-root cap is set in advance; incomplete sums would have null values
and could not be ranked. All three values finish completely.

| Root request | Immediate expected log growth | H2 expected log growth | Time |
|---|---:|---:|---:|
| 0 BTC | 0.0000 bp | 56.6715 bp | 5.18 s |
| 2.33839 BTC, old H1 | 33.4193 bp | 113.0068 bp | 6.40 s |
| 2.33796 BTC, execution H1 | 71.4987 bp | 169.7223 bp | 8.12 s |

The execution-H1 request remains best among these three candidates. Its H2 value
exceeds the old request by **56.7155 bp** under the fixed model. This is not a
global H2 root search, an H2 policy backtest, or evidence of reliable returns.
It establishes a consistent continuation calculation and a measured lower
bound on the optimal two-event value at one saved account.

An exact-key cache reuses 111 of 846 successor-account solves; 735 distinct H1
continuations remain. An independent artifact audit rescores all **916,080**
two-event paths using the public transition evaluator. Every saved continuation
value matches exactly. Direct `log(finalEquity / originalEquity)` agrees with the
sum of event log increments within **1.46e-16**, including path-dependent fills,
funding and account constraints. The probe exits zero.

The measured per-candidate cost rules out naively repeating this backup for
hundreds of root requests at each historical decision. The next implementation
needs execution-consistent continuation bounds or shared value approximations
with independently checked error. Reusing the old decision-price concavity
certificate would skip the very acceptance discontinuities this change fixes.
No further large computation is launched on the strength of a three-candidate
ranking alone.

## Execution-consistent continuation bounds and selective backups (v492–v495)

The previous turn is progress: complete H1 coverage and a verified H2 action-value
calculation are authoritative results. The next obstacle is computational cost,
not missing permission or an unavailable market process. This turn preserves the
fixed forecast and develops a way to discard inferior H2 requests cheaply.

### An optimization not retained

V492 removes guarded lattice boundaries where no outcome's acceptance, funding,
sign or settlement regime changes. On July's initial account and all 735 distinct
saved continuation accounts, every selected request and value remains exactly
equal to the reference. Evaluated orders fall from **447,104 to 272,584**, about
39%. However, determining active boundaries consumes most of the saving: paired
times improve only about **6–10%**. The extra implementation complexity is not
retained. The production research solver is restored exactly to the v491 source;
its validation is subsequently extracted into a shared helper without changing
the search. V492 preserves the rejected experiment and its measurements for
inspection rather than leaving an unused runtime mode.

### A controlled information relaxation

`event-execution-upper.ts` supplies an upper bound on marked H1 value. It grants
the controller knowledge of the first opening-price ratio before choosing an
order. Within each equal-opening group it preserves the full conditional path
mixture and its exact funding coefficients. Future returns inside a group are
not revealed. Size minima/maxima, lots, availability restrictions and intrabar
maintenance are relaxed. Fees and pre-trade opening solvency remain. This is
optimistic information/execution used to bound value, not permission to give a
real order access to the next candle.

Write g for first-open/decision-price, e for post-open exposure, R for
close/first-open, B for compounded long-debt growth, and I for the path's short
interest integral normalized by the first open. Conditional terminal wealth
divided by post-trade equity is:

- Short, e < 0: `1 + e * (R - 1 + I)`.
- Funded long, 0 <= e <= 1: `1 + e * (R - 1)`.
- Borrowed long, e > 1: `B + e * (R - B)`.

For buy/sell fee sign s, use `t = e / (1 + s*f*e)`. Each conditional wealth ratio
after the fee-budget transformation is affine in t on the funding pieces.
Expected log wealth is concave, including the funding kinks. The two signed
optima can therefore be prepared once per opening group. Query-time exposure
limits restrict the interval; clipping the transformed optimum to that interval
does not increase its regret beyond the stored global optimization gap. Tangent
bounds and a 1e-10 numerical pad cover that gap. The largest recorded gap is
1.00000028e-10 in the July law, negligible beside information-relaxation slack.

The exposure cap is `max(entry leverage cap, abs(old opening exposure))`.
Consequently above-cap waiting and maximum-order recovery are contained in the
relaxation instead of accidentally excluded. Any surviving position at the first
open also has absolute exposure below `1 / maintenanceMargin`; this gives the
finite precomputation domain. The bound currently requires maintenance margin
greater than total proportional friction. Unsupported costs yield `Infinity`,
never an extrapolated finite certificate. The ordinary exact solver remains
available in those cases.

Rounding needs a separate allowance because the replay checks fees and leverage
before rounding resulting inventory. The relaxed order may need to pay for the
small difference to reach that exact rounded holding. A positive quote-cash
credit covers both its additional fee and collateral. This makes the actual
rounded trade feasible in the relaxed problem, including fractional initial
inventory; it is not silently assumed to be a continuous exact-lot account.
Opening ruin still yields -Infinity, since no request can repair a liquidation
that precedes its execution. Numerically unrepresentable accounts produce an
uninformative upper rather than a false finite result.

### Evidence and selective continuation solves

The existing **960 exhaustive small-lattice problems** now also check that the
new bound dominates every optimum, including marked/market settlement, gaps,
unequal borrowing, unavailable openings, dust and above-cap inventories. New
tests check that coincident openings do not reveal opposite future returns,
that distinct openings deliberately provide additional information, and that
unsupported costs and opening ruin are handled explicitly. Another test verifies
the H2 lower/upper interval against brute force, exact-cache reuse, time/work
limits, and pruning only against an achievable incumbent. All **154 focused
tests** and workspace typechecks pass.

V493 queries all 846 saved first-event outcomes of the three July candidate
requests. Preparation takes **0.089 s** and all queries **0.112 s**, versus
19.70 seconds of saved exact H2 evaluation. The resulting bounds are:

| Root request | Exact H2 value | Initial H2 upper | Can reject against 169.7223 bp? |
|---|---:|---:|---|
| 0 BTC | 56.6715 bp | 120.1745 bp | Yes |
| 2.33839 BTC, old H1 | 113.0068 bp | 170.5995 bp | Not yet |
| 2.33796 BTC, execution H1 | 169.7223 bp | 222.3855 bp | No |

`event-execution-backup.ts` starts each continuation with an executable holding
policy as a lower bound and the new relaxation as an upper. It merges identical
successor leaf/account states, so the returned contingent policy cannot choose
different actions solely because duplicate empirical atoms have different IDs.
It refines continuations in descending probability-weighted uncertainty using
the complete H1 solver, caches exact results, and stops only at full evaluation,
proof that the request cannot beat a supplied achievable incumbent, or an
explicit work/time limit. Incomplete values are null; partial sums are never
called exact Bellman values. The lower bound retains a concrete continuation
policy and can be independently executed under the full law.

V494 uses the previously fully evaluated winning July request as its incumbent:

| Request | Result | Exact continuation solves | Time | Saved full-evaluation time |
|---|---|---:|---:|---:|
| 0 BTC | Proven inferior | 0 | 0.198 s | 5.178 s |
| Old H1 | Proven inferior | 1 | 0.181 s | 6.401 s |
| Execution H1 | Fully evaluated | 277 | 10.214 s | 8.120 s |

One continuation lowers the old request's upper bound to **169.1016 bp**, below
the incumbent's 169.7223 bp. Its true saved value remains 113.0068 bp; the pruner
does not pretend to have recomputed that exact value. All three returned lower
policies are independently rescored on **916,080 joint paths**, agreeing within
1e-12. The winning request remains expensive and this run adds overhead to it;
the measured gain is in rejecting inferior requests, not a blanket claim that
every backup became faster. Its exact value agrees with v491 within 1e-17.

V495 checks the analytical upper against every saved H1 optimum across the
**28 full windows / 9,629 decisions**. All bounds are finite and dominate their
saved values. Slack ranges from **5.7229 to 186.1891 bp**. Total preparation is
1.81 seconds, queries 0.94 seconds, and the entire source-bound audit 3.37
seconds. This retains the 27 strict-source/one March scenario distinction; it
does not run or promote a new historical policy.

### Research direction and remaining work

This construction is a zero-penalty information relaxation. The general approach
and the possibility of tightening it with feasible penalties are developed by
Brown, Smith and Sun in
[Information Relaxations and Duality in Stochastic Dynamic Programs](https://people.duke.edu/~psun/bio/Information_Relaxations_and%20Duality_in_Stochastic_DPs.pdf)
(author manuscript; journal publication 2010). Their framework combines achievable
policy values with optimistic upper bounds. Zhu, Ye and Zhou's
[Solving the Dual Problems of Dynamic Programs via Regression](https://arxiv.org/abs/1610.07726)
studies regression-based feasible penalties that avoid nested conditional
simulation. These are relevant tools for tightening a bound; neither paper proves
this trading model convex or supplies a ready-made optimal policy for its
discontinuous order acceptance.

The measured positive slack shows why this upper alone cannot certify global
deeper convergence: even at a single fixed account it grants information the
real controller does not possess. Next, use it to bound root-request regions and
refine the remaining uncertainty with the same execution law. Two concrete
tightening routes are nested outcome-information partitions, whose coarsest
single group is the original H1 problem, and centered payoff/value penalties
whose expectation vanishes for every permissible nonanticipating request.
Prepare and profile these on saved calibration accounts before any broad replay.
Any root-interval tangent must first be proved valid across outcome-specific
acceptance, funding and above-cap recovery regions; the old decision-price
certificate cannot be substituted. A matched position-proposal/coordinator test
and full-depth conditional optimization remain outstanding. Forecasts and
reported historical returns are unchanged by this computational work.

## Nested information partitions and bounded continuation policies (v496–v500)

The prior bound is useful for pruning but grants every first opening-price ratio
in advance. Its persistent information advantage is too wide near the best
request. This turn adds a hierarchy that restores the original information
constraint, retains explicit lower policies, and measures its computational value.

### Coarsening returns to the original problem

`event-execution-partitions.ts` sorts the fixed atoms by opening-price ratio and
recursively splits at the nearest probability-balanced boundary between distinct
opening prices. Identical opening ratios are never split, so group identity
reveals no additional distinction between their later returns. At a given depth,
the relaxed controller knows which opening group occurs and chooses one committed
request for that group. Each group uses the **original exact H1 request solver**,
including next-open rejections, all size constraints, fees, borrowing, maintenance
and above-cap holding/recovery rules.

For a partition P, its upper value is

`U_P(s) = sum_B Pr(B) * max_a E[log(W1/W0) | B, s, a]`.

Every real common request is among the choices in every group, hence its value
cannot exceed this weighted sum. Merging groups removes information and cannot
increase the upper. One group is exactly the original request problem. This
argument does not require global concavity across acceptance discontinuities.
Probability weights and complete outcome rows are preserved; only the controller's
information is relaxed. A small numerical pad covers the weighted recombination.

The zero request and all group-optimal requests are separately evaluated against
the **full original law**. The best of these executable common requests supplies
a lower bound. Independently choosing each group's request is never substituted
for a real nonanticipating order. A coarse partition may fail to propose the true
best common request, and its upper/lower gap must then trigger refinement.

V496 first profiles the twelve predefined July leaf × cash/invested/above-cap
accounts from the earlier backup. Three local timing repetitions per case give:

| Information depth | Typical groups | Sum of median query times | Mean upper slack | Largest proposal regret |
|---|---:|---:|---:|---:|
| 0, original full problem | 1 | 220.34 ms | 0 bp | 0 bp |
| 1 | 2 | 144.58 ms | 1.8657 bp | 0 bp |
| 2 | 4 | 107.08 ms | 2.5048 bp | 0 bp |
| 3 | 8 | 100.00 ms | 6.0515 bp | 0.0141 bp |
| 4 | 15 | 126.90 ms | 10.8847 bp | 0.4368 bp |

More groups can reduce individual solve cost, but reveal more information and
widen the bound. They do not monotonically improve the common request proposals.
The two-group level is the first tightening stage, followed by the original
solver if the remaining value gap matters.

### Explicit value intervals in the Bellman backup

`event-execution-backup.ts` now accepts an explicit `valueTolerance`. A positive
tolerance permits status `certified` when a **fixed-root two-event value** lies
inside the returned interval of the requested width. `value` remains null and
`complete` remains false in that case; an approximate midpoint is not reported
as an exact value. `lowerValue` retains a concrete contingent policy whose
second-stage requests are evaluated under the full original law. `upperValue`
bounds the best possible second-stage policy for that same root request. A zero
tolerance keeps the exact-evaluation contract.

After each tightening the remaining successor states are reprioritized by
probability-weighted uncertainty. States with wide bounds receive either their
two-group solve or exact H1 fallback. Exact duplicate leaf/account states share
one action and cache entry. Pruning against an achievable incumbent and explicit
work/time budgets remain available.

V497 evaluates the same July candidates at **0.001 bp** value tolerance. The
winning request reaches an interval of **0.0009920 bp** with 132 exact and 277
partition solves, taking 6.33 seconds. Its executable lower policy is only
0.00000897 bp below the independently saved exact value. This is an action-value
certificate for a known request, not a globally optimized root action.

### All predeclared calibration probes and a detected proposal failure

V498 verifies all **113 predeclared first-leaf/held calibration probes** from the
28 unchanged compiled laws. It checks law/model/calibration-trace hashes and
compares every bound and executable proposal to the original full H1 optimizer.
The two-group and four-group bounds dominate all reference values, and coarsening
is monotone within numerical tolerance.

| Method | Query time, single passes | Exact-value proposals | Intervals within 0.001 bp | Largest proposal regret |
|---|---:|---:|---:|---:|
| Original H1 | 1,167.85 ms | 113 / 113 | 113 / 113 | 0 bp |
| Two groups | 1,105.93 ms | 111 / 113 | 86 / 113 | 0.2193 bp |
| Four groups | 1,024.32 ms | 109 / 113 | 44 / 113 | 2.6385 bp |

Preparation takes 2.60 seconds; the entire audit takes 6.52 seconds. These are
computational probes, not newly independent market samples or test-return
selection. The gain is concentrated in large kernels; adding a partition solve
and then a full solve can be slower on small kernels.

The two misses share the same June 2026 calibration account and 311-atom law,
used by both the three-day and seven-day downtrend windows. They are not two
independent failures. At equity $10,000 and price 77,322.01:

- One group, with probability 73.77%, proposes buying **0.44203 BTC**, worth
  +0.3604 bp conditionally but **-0.3213 bp under the full law**.
- The other group proposes zero. The best common request among those proposals
  is therefore zero, with value zero.
- The original global optimizer instead requests **0.64279 BTC**, worth
  **+0.2193 bp**. It has **99.5631% modeled rejection probability** and is absent
  from the finite proposal set.
- The upper/lower interval is **0.2659 bp**, so it correctly refuses a 0.001 bp
  certificate and demands further work.

This illustrates a concrete loss from replacing an action-value curve with a
small list of local optima. It is a scenario-partition experiment, not a test of
lifecycle-position decomposition. The documented position framework's global
coordinator can retain full scenario-PnL/value curves; these results do not reject
that framework. They also reinforce that the model's apparent edge can depend
on empirical acceptance filtering, which remains a separate forecasting and
execution-realism issue.

### Computation gate and the second control

The backup uses the partition stage only for kernels with at least **1,024
atoms**, based on the calibration timing/gap evidence. Smaller kernels go
directly to their exact solve when refinement is required. This gate changes
computation, never the value-bound contract or the permitted real action set.
It also sends the 311-atom proposal-failure case directly to the full optimizer.

V499 repeats July with this gate. Root hold is pruned without a continuation
solve, and the old request after one exact solve. The winning request takes
**5.83 seconds**, with **172 exact and 105 partition solves**. Its interval is
**0.0004471 bp** wide. The executable lower policy agrees with the v491 exact
value within 1e-17, and independent rescoring of all **916,080 joint paths**
checks the returned lower policies. The earlier full evaluation took 8.12 seconds;
these local timings establish a useful saving, not a universal speed guarantee.
Original source snapshots and explicit certificate semantics remain recorded.

V500 applies the original exact backup to the second predefined control law,
November 2024's initial calibration account. The old H1 request is zero, so the
predeclared set contains two distinct requests:

| November root request | Immediate log value | Exact H2 log value | Time |
|---|---:|---:|---:|
| 0 BTC | 0 bp | 0.160420 bp | 5.24 s |
| -0.72268 BTC, execution H1 | 0.064492 bp | 0.234902 bp | 0.067 s |

The short request retains its ranking, improving H2 value over zero by
**0.074482 bp** under this law. Its modeled rejection probability is **99.9355%**.
Only one successor-account solve is new relative to the cash backup; the other
1,269 outcomes reuse exact cached continuations. An independent audit rescores
**2,637,284 two-event paths**, reproduces every continuation value exactly, and
checks the Bellman composition within 5.38e-17. This is evidence about the fixed
model and cache behavior, not a claim of realized trading profit.

All **155 focused tests** and workspace typechecks pass. New checks include
coarsening bounds against exhaustive whole-law request search, unsplit identical
openings, source immutability, and explicit certified-value/budget semantics.
No historical backtest returns or fitted probabilities change in this turn.

### Next global-search requirement

The remaining step is to search **all root requests**, not merely refine values
of known candidates. A root-interval bound must cover the complete range of
successor cash, inventory and acceptance behavior. A promising construction is
to bound cash and asset units over each interval and retain explicit uncertainty
about which future orders are accepted. Under the compressed funding law,
terminal wealth is increasing in each of cash and asset units, which supplies
useful balance bounds. However, evaluating the existing optimizer at a single
optimistic account is not automatically an upper bound: changed order acceptance
can remove beneficial rejections. That effect and recovery exceptions must be
included, with the bound returning to the original H1 problem as the root interval
shrinks. Profile the interval construction on these saved calibration controls
before launching any all-window deeper replay. Full-depth conditional optimality,
the matched lifecycle-policy comparison, and later forecast improvement remain
outstanding.

## Account-region bounds and acceptance dependence (v501–v505)

The objective remains expected log **account equity** under the fixed forecasting
law. Decomposition is a candidate organization of that decision problem, judged
by matched utility, realized equity, execution costs and computation. The earlier
accounting-only parity result does not test the full local-proposal/global-
coordinator policy in `docs/theory/Position management.md`. No result below is an
ablation of that position framework.

### A bound for a range of successor accounts

`event-execution-box.ts` bounds expected **log terminal equity**, rather than log
growth, over a fixed-price cash/inventory region and all common lot requests.
It owns the original joint execution law, keeps its fees, borrowing, maintenance
and size rules, and returns an explicitly optimistic request. That request is
not an executable policy recommendation. Completion means the relaxed request
search finished; it does not certify the original region optimum or an H2 root.

The first construction, v501, uses independent balance intervals. Terminal
wealth is increasing in cash and inventory on live accounts, so their upper
endpoints dominate financial outcomes. This alone is insufficient for order
acceptance: extra wealth can remove useful rejection or an above-cap recovery.
Acceptance is therefore classified as impossible, certain or uncertain over the
whole region. Uncertain branches may take the larger valid accepted/held value.
Opening ruin before the request remains unrecoverable. All recovery-eligible
integers near the maximum-order limit are explicit search guards; unsafe numeric
domains return an unbounded result, not a false finite certificate.

A regression example has cash 21, inventory -20, price 1, leverage limit 5 and
zero fees. Buying 36 units is permitted as a maximum-size recovery, changing
exposure from -20 to +16. Adding 2.5 units of inventory leaves equity 3.5 and
exposure -5; the same order would produce +5.286 and is rejected. Thus the
ordinary-cap monotonicity argument cannot replace recovery checks.

V502 optionally restricts the region to a convex hull of supplied balance
vertices. The cash/inventory relation then survives in the two affine cap
constraints. With opening price G, request u, fee rate f and padded leverage L,
ordinary acceptance requires positive post-fee equity and

`L*(C + Q*G) - Q*G >= (u + L*f*abs(u))*G`,

`L*(C + Q*G) + Q*G >= (L*f*abs(u) - u)*G`.

Their extrema over the hull bound certain/possible acceptance. The maxima may
occur at different vertices; allowing that remains optimistic. The financial
payoff still uses dominating balances, with an explicit rounding allowance.

### Predeclared compute probes and a failed specialization

V501 and v502 use the first saved calibration occurrence of each leaf and
cash/invested/above-cap bucket from July v491 and November v500: **17 accounts**.
Each has fixed half-widths of **0, 1, 10 and 100 quantity lots**, giving 68 queries.
The original models, laws and source summaries are hash-bound. Saved centre
optima and freshly solved corner/endpoint optima are below every corresponding
bound. These are computational controls, not new independent market windows.

| Bound | Sum of query times | Mean one-lot slack vs centre | Largest one-lot slack |
|---|---:|---:|---:|
| Independent balances, v501 | 12.02 s | 26.10 bp | 80.05 bp |
| Linked balances, v502 | 11.62 s | 19.46 bp | 79.82 bp |
| Linked balances and shared opening acceptance, v505 | 15.40 s | 7.88 bp | 37.53 bp |

Zero-width cases agree with their original H1 values to the 0.00001001 bp numeric
padding. This observed agreement is not a license to reconstruct a root
singleton with different floating-point operation ordering: an actual root
singleton must use its original exact account transition.

V503 prepares the six financial branches per outcome and removes repeated
branch allocations. All **68 values, relaxed requests and search counts** are
identical to v502, but local query time rises from 11.62 to 13.23 seconds. The
specialization is **not retained**. This single comparison does not establish a
universal slowdown; it supplies no evidence to keep the extra implementation.

### Why the bound manufactures a large apparent edge

V504 enumerates the three nominal lot inventories in every linked one-lot segment
and globally solves the original H1 problem at each resulting account. It also
scores the relaxed request at each account using the independent path evaluator.
Reconstruction uses nominal lot multiples within the region's floating-point
padding; these accounts do not replace original Bellman successor objects.
Finally, it permits an explicitly clairvoyant choice of incoming account for
each later outcome while holding that request fixed. That last construction is
a diagnostic information relaxation, never a strategy.

At November's first cash control, the relaxed request is -0.72266 BTC. Across the
three accounts, acceptance varies for **99.8333%** of the probability mass.
Choosing a different account for each future outcome gains **70.2363 bp** over
the best original H1 value among all three accounts. The remaining **1.3668 bp**
comes from optimistic financial balances. Together they explain the **71.6031
bp** excess over that finite account set. July's first cash control instead has
only 0.3130% uncertain-acceptance mass and 0.00106 bp of this revealed-account
advantage. This identifies a specific information/coupling error in the bound,
not a deficient return predictor or an observed trading profit.

### Preserve identical-opening acceptance

V505 groups available outcomes with exactly the same next-opening ratio. Such
outcomes must accept or reject a committed order together: terminal return,
duration, funding and extrema cannot influence that opening decision. Unavailable
outcomes retain their separate forced rejection behavior. The bound may choose
accepted versus held for each opening group, but may no longer select only the
profitable terminal outcomes within that group.

Within each guarded financial/acceptance interval, a group's accepted expected
log wealth is concave in the request and its held value is constant. Their
difference can be nonnegative on only one interval. The search explicitly finds
both integer crossings before applying concavity to the total common-request
objective; it does not incorrectly assume that a maximum of concave branches is
concave. This preserves a complete search of the grouped relaxation.

All **68** coarsening comparisons pass; 35 bounds tighten strictly. November's
first one-lot centre slack falls from **71.6043 to 25.5394 bp**. July's first cash
control remains 0.4232 bp, while its other cash controls improve from 10.55–14.65
bp to 1.92–2.71 bp. Grouping is retained as a correctness-preserving tightening
of the research bound, but its **broad-replay compute gate fails**: remaining
slack and per-query cost are still too large for an unqualified nested search.
No all-window deeper replay is launched on this evidence.

All **159 focused tests** and workspace typechecks pass. New checks cover 120
generated laws/regions with exhaustive request lattices and 3,000 account-grid
comparisons (including repeated singleton grid points), linked-balance bounds,
funding, dust, unavailable openings, ruin and nonmonotone recovery. Eighteen
separate grouped problems compare the relaxed global request optimum against
exhaustive acceptance choices scored through the original path evaluator.
The tests also prohibit using future returns to split identical available opens.

### Next tightening requirement

The grouped relaxation still chooses acceptance independently at different
opening prices. A real ordinary-cap request has more structure: for a fixed
account and request, its positive-equity and two cap constraints are affine in
G, so accepted opening prices form an interval after the size filters. With
strictly positive cash the ordinary-cap part is a prefix; with strictly negative
cash it is a suffix or empty. Availability and maximum-size recovery must remain
explicit exceptions. Preserving this ordered acceptance structure is the next
candidate tightening, to be tested before wider computation.

One possible search bound uses tangents to each accepted group's concave log
value within a financial interval. For a fixed accepted prefix/interval, their
sum is an affine upper in the request; maximizing those affine bounds over
admissible cut points and interval endpoints supplies an optimistic interval
bound. It still needs implementation, numerical checks and measured pruning.
For small discrete incoming-account sets, exact enumeration avoids the
revealed-account relaxation altogether and provides a useful reference.

The position notes' account-level coordinator and shared scenario-PnL curves are
consistent with this lesson: retain dependencies until the joint decision is
made. The notes' endpoint/record-scan theorem concerns a specified hindsight
piecewise-linear model; it does not by itself solve stochastic next-open
acceptance with integer orders. Full root-request search, deeper convergence,
the matched position-policy comparison and later forecast improvement remain
required. All earlier historical backtest returns are unchanged.

## Ordered acceptance, coupled wealth and the root-request cover (v506–v513)

### Ordered opening decisions

V506 implements the previous section's ordered-acceptance tightening. For a
fixed account and ordinary request, accepted opening prices form an interval.
With positive incoming cash the cap part is a prefix; with negative cash it is
a suffix. The known size filters are applied first. A three-state scan maximizes
the accepted/rejected group values over a common interval, with corresponding
two-state prefix/suffix cases. Identical openings remain grouped. Unavailable
openings and possible maximum-size recovery orders retain explicit exceptions.

Recovery possibility uses a bound on the actual leverage ratio over balance
vertices, rather than combining the largest inventory with the smallest equity
from unrelated vertices. The latter caused a detected false recovery exception
in an exhaustive test and unnecessarily widened the relaxation. With positive
equity throughout a convex hull, an absolute linear-fractional leverage ratio
is bounded by its vertex maxima; numeric padding is applied before division.

Inside a guarded request interval, each accepted group has a concave log-value
curve. Its tangent is an upper bound. Each permitted acceptance sequence sums
affine tangents, so the maximum over sequences is convex in the request and
attains its interval maximum at an endpoint. A priority queue subdivides only
intervals that could exceed the best evaluated relaxed value. Explicit work
budgets retain an upper and a relaxed lower value without claiming completion.
These lower values belong to the relaxation, not to an executable account policy.

The first November one-lot centre slack falls from **25.5394 to 1.7077 bp**.
All twenty predefined November region queries close their relaxed numerical
intervals. The remaining financial slack is then addressed separately.

### Preserve the balance segment in financial wealth

On live accounts, the compressed terminal financial payoff can be written as

`W(C,Q) = C + Q*Pclose + min(C,0)*(B-1) + min(Q,0)*P*I`,

where B is quote-debt growth and I is the asset-borrow price integral. The case
with both cash and inventory negative is nonlive and cannot contribute positive
terminal wealth. The expression is a minimum of four affine funding facets.
Along a supplied two-vertex balance segment, and on a fixed request sign, each
facet has the form `a_i + b_i*k + d_i*t`, with `0 <= t <= 1`.

`event-affine-segment.ts` represents

`max_t min_i(a_i + b_i*k + d_i*t)`

as a minimum of affine supports. A single-facet support is
`a_i + b_i*k + max(0,d_i)`. Further supports mix two opposite-sign slopes with
weights making their total t-slope zero. These are the extreme choices of the
finite linear-program dual, so no numerical inner optimizer is required.
The general strong-duality and max-min reasoning is standard; see Boyd and
Vandenberghe, [Convex Optimization, sections 5.2.4 and 5.4](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf).
The funding facets and their application here are derived for this executor,
not a claim that the full discontinuous trading problem is convex.

V507 applies these supports to the linked financial segment. The existing
optimistic maintenance test remains a necessary filter; the financial maximum
may still select a different segment point for different outcomes. Acceptance
ordering and integer committed requests remain separate constraints.

| November probe | One-lot slack, ordered only | One-lot slack, coupled wealth | 100-lot slack, coupled wealth |
|---|---:|---:|---:|
| First cash / leaf 1 | 1.7077 bp | 0.3345 bp | 0.5899 bp |
| Cash / leaf 0 | 4.3720 bp | 3.0032 bp | 3.2311 bp |
| Cash / leaf 3 | 2.3412 bp | 0.9668 bp | 1.2529 bp |
| Cash / leaf 2 | 1.9510 bp | 0.5866 bp | 0.8738 bp |
| Invested / leaf 3 | 1.3561 bp | 0.0028 bp | 0.2808 bp |

The twenty controls retain valid bounds on their checked original H1 optima.
These widths are relative to the saved centre account, not proof of an exact
optimum over every account in a region. The relaxed numerical search gap is
approximately 0.00001001 bp in these queries.

### Complete first-request regions and one H2 interval probe

The existing exact H1 solver can now optionally return `requestRegions`.
Every region contains only requests with finite first-event log wealth under
the full law. Non-singleton regions have fixed acceptance and funding branches
for every outcome, so successor cash and inventory are affine in the root lot.
Numerical guard points are retained as singletons. Adjacent regions are not
merged across a branch boundary. Requests outside the finite maximum are all
equivalent to holding. Normal H1 calls keep their existing output shape.

V509 checks the two original calibration controls:

| Control | Finite lot domain | Regions | Singletons | Non-singletons | Audited first transitions |
|---|---:|---:|---:|---:|---:|
| November | 145,407 lots | 59 | 51 | 8 | 224,790 |
| July | 470,565 lots | 381 | 369 | 12 | 322,326 |

The saved H1 request and value remain exactly unchanged. Endpoint/midpoint
transition checks retain acceptance flags and affine equity/inventory, with
maximum equity interpolation error **1.28e-11 dollars**.

The first H2 region is chosen mechanically: the closest non-singleton region to
the H1 incumbent, clipped to at most 201 lots. For each first outcome, its two
endpoint accounts define the child balance segment. Summing the child upper
log-terminal-equity values and subtracting log initial equity bounds every root
request in that region. The midpoint uses exact H1 continuations, rescored via
the independent path evaluator. A soft 30-second budget leaves both complete
region values null if any first outcome remains unevaluated.

November's selected lots **[-72,471, -72,271]** all produce the same successor
accounts as a rejected order. Its H2 value is **0.160420 bp**, below the verified
incumbent's **0.234902 bp**. All 1,270 continuations reuse previously verified
values; this regional evaluation takes 0.0086 seconds after preparation.

July's selected lots **[233,594, 233,794]** require changing successor accounts.
The probe reaches **91 of 282 outcomes**, covering 61.08% probability, in **32.73
seconds**. Its complete upper and midpoint values remain **null**. The evaluated
portion has 0.08793 bp of probability-weighted upper excess over its exact
midpoint continuations, but that partial sum is not a whole-region certificate.
This run fails the broad nested-computation gate.

### Profiled changes that are and are not retained

- **V508:** scalar point evaluation preserves all twenty November upper values
  but saves only 8.67% in a single pass while duplicating financial formulas.
  It is not retained.
- **V510:** a support's log tangent also bounds the minimum of all financial
  supports across their crossings. Ordered search therefore keeps every support
  solvency boundary but does not need support-crossing partitions. Three paired
  control medians change from 111.61/2,686.89/57.57 ms to
  104.66/2,511.18/57.86 ms, with unchanged upper values. This mathematical
  simplification is retained, but the modest speed change does not clear the
  broad-computation gate. CPU profiles identify repeated group/financial
  evaluation as the dominant remaining cost.
- **V511:** precompiling ordinary acceptance spans preserves the checked bounds
  but makes the three paired queries **2.43–3.69 times slower**. It is not retained;
  the shared v510 evaluator is restored. Recorded candidate/reference sources
  preserve the comparison.

### Test small inventory seeds rather than assume their value

The remaining acceptance dependence suggests testing whether a cheap initial
inventory change improves the next integer-order decision. V512 predeclares both
signs of the smallest request meeting order minima at every available forecast
opening, and twice that size. At the November cash control these are ±0.00008
and ±0.00016 BTC. This is a new fixed-root candidate test under the original
forecasting law, not test-return selection.

| Root request | Immediate log value | Certified H2 upper | Existing incumbent H2 value |
|---|---:|---:|---:|
| -0.00008 BTC | -0.003320 bp | 0.228370 bp | 0.234902 bp |
| +0.00008 BTC | -0.009901 bp | 0.230584 bp | 0.234902 bp |
| -0.00016 BTC | -0.006641 bp | 0.233350 bp | 0.234902 bp |
| +0.00016 BTC | -0.019803 bp | 0.233944 bp | 0.234902 bp |

All requests fill in the forecast law and all four are **pruned as inferior**.
Their exact H2 values remain null because full continuation optimization was
unnecessary. Each returned lower policy is independently rescored across all
1,318,642 joint paths: **5,274,568** path checks in total. The whole experiment
takes 33.26 seconds. This rejects these four seeds at this account; it does not
reject every small inventory adjustment or establish global root optimality.

### Remove whole hold-equivalent regions

V513 checks the entire captured root covers. With constant first-execution
branches, regions whose requests never fill leave exactly the hold successor
account for every outcome. Their complete H2 values therefore equal the already
evaluated zero request. They can be eliminated against the existing achievable
incumbent without further child optimization.

| Control | Eliminated regions | Eliminated finite lots | Remaining regions | Remaining finite lots |
|---|---:|---:|---:|---:|
| November | 33 / 59 | 885 | 26: 22 singletons + 4 intervals | 144,522 |
| July | 179 / 381 | 2,857 | 202: 196 singletons + 6 intervals | 467,708 |

Only about **0.6% of finite request lots** are removed; the reduction in region
count must not be presented as equivalent progress through the whole lot domain.
All requests beyond the maximum size are also hold-equivalent. The remaining
regions are not globally optimized.

All **162 focused tests** and workspace typechecks pass. New validation includes
exhaustive acceptance sequences with unavailable/recovery exceptions, 2,100
independent primal-breakpoint checks of the segment dual supports, explicit
unfinished-subdivision semantics, and complete feasible-region coverage across
the existing 960 small execution problems. Native H1 and historical backtest
returns remain unchanged. Full root optimization, deeper Bellman convergence,
the matched position-policy comparison and subsequent forecast improvement
remain required.

## Position-coordinate decision and common-root failure modes (v514–v521)

### Finish the isolated November roots first

V514 evaluates only the 22 singleton regions left by the complete H1 cover and
hold-equivalent pruning. The existing zero and -0.72268 BTC controls first warm
the exact continuation cache. The incumbent lot -72,268 reproduces its saved H2
value, **0.2349019024 bp**. The adjacent lot -72,267 is evaluated exactly and is
lower by about **0.000001005 bp**. The other 20 singleton roots are certified
unable to beat the achievable incumbent. This takes 69.02 seconds and leaves
only four non-singleton regions:

`[-72,263,-10]`, `[10,14,520]`, `[14,525,14,555]`, and `[14,560,72,263]`.

The incumbent is still a fixed-model H2 candidate. Classifying every singleton
does not bound the requests inside those four intervals.

### Reject the scalar whole-interval bound before scaling it

V515 applies the ordered, coupled account-segment upper to the first full
interval with zero adaptive subdivisions. All 1,222 distinct live successor
segments vary across the interval, so it still initializes 1,222 child box
queries. The pass takes **124.76 seconds**. Its upper is **118.6030 bp**, compared
with the incumbent's 0.2349 bp, and 261 child boxes retain unfinished relaxed
searches. The upper remains valid, but it is far too loose to prune. The process
is stopped before repeating this cost on the other three intervals.

The dominant relaxation error is now clear. Each first-outcome branch may choose
a different root point before the branch values are summed. A target-coordinate
rename cannot restore that lost root nonanticipativity.

### Exact shape probes and failed fixed-root upper

V516 evaluates the two endpoints and midpoint of every unresolved interval with
the exact H1 continuation optimizer. All twelve requests complete in 91.82
seconds. The negative interval has H2 values **-9.9585, -3.8444, and +0.14635
bp** from left to right. The first positive interval starts at +0.14231 bp and
falls below zero; both larger positive intervals are negative at every sampled
point. The largest sample is therefore 0.14635 bp, 0.08856 bp below the incumbent.

These monotone samples motivate a shape theorem but cannot supply one. V517
evaluates the continuous opening-information child upper on a 33-point grid per
interval. At exact sample points its slack is **43.02–49.85 bp**, so every region
still has sampled upper values above the incumbent. This cheaper upper also
fails the refinement gate.

### Does the position coordinate preserve useful policy structure?

For a fixed account, current inventory plus signed request and current inventory
plus signed target are bijective on the lot lattice. A complete target-position
value curve followed by the same account-level coordinator therefore represents
the same physical single-asset decisions as the global request value curve. It
does not improve equity by itself. Lifecycle decomposition remains useful after
selection because it attributes entries, reductions, exits, fees, borrowing and
PnL without changing the net exchange order.

The possible computational benefit was that child targets might remain fixed as
the root request changes. V519 tests the exact child policies at the endpoints
and midpoint of the first large interval:

- the child request is identical on 9.77% of outcome mass;
- target inventory is identical on **0 of 1,270** outcome rows;
- intended target exposure is identical within 1e-6 on **0** rows;
- only **18.43%** of mass stays within 0.001 exposure;
- the largest target-exposure range is about **10.0003**.

Thus neither target inventory nor signed target exposure supplies a stable local
policy coordinate for this interval. Compressing the complete value curve to a
position-local optimum would discard material state dependence. This agrees with
v498, where a small list of scenario optima missed a profitable common request.
The evidence supports the current architecture: the account optimizer owns the
full value curve and feasibility; `EventPositionLedger` records the selected
physical fills and classifies lifecycle changes.

### Integer recourse invalidates the concavity shortcut

V520 tests discrete H2 concavity on 60 deterministic-seed synthetic execution
laws, eight account states per law, and the exact H1 request regions. Among 2,860
finite adjacent-lot triples inside 529 guarded regions, **325** have a positive
second difference. The largest is 0.05689 log units. Fixing the first execution
branch is therefore insufficient: the optimal integer child action creates a
discontinuous, nonconcave recourse value. Endpoint maxima, derivative bisection,
or a tangent proof over the whole H2 region would be invalid.

This matches the general mixed-integer recourse literature. Hassanzadeh and
Ralphs describe the second-stage value function as discontinuous and nonconvex
and recover exact cuts from branch-and-bound trees in
[A Generalization of Benders' Algorithm for Two-Stage Stochastic Optimization
Problems With Mixed Integer Recourse](https://optimization-online.org/2014/08/4474/).
Boland et al. compute bounds by relaxing nonanticipativity and optimizing its
Lagrangian dual in
[Combining Progressive Hedging with a Frank-Wolfe Method](https://optimization-online.org/2016/03/5391/).
Deng and Xie's
[ReLU Lagrangian cuts](https://optimization-online.org/2024/11/on-the-relu-lagrangian-cuts-for-stochastic-mixed-integer-programming/)
give a newer exact-cut route for stochastic mixed-integer programs. These papers
do not directly solve the executor's log-affine acceptance problem, but they
support the required structure: keep scenario recourse, add cuts that enforce a
common root action, and branch only where the resulting bound remains material.

V521 validates the source-bound artifacts and records the architecture decision.
The next prototype should build a nonanticipativity Lagrangian or branch-tree cut
from the existing exact one-step subproblems and test it only on the November
control. The whole-interval account box, fixed-root opening-information upper,
stable-target restriction and H2 concavity assumption have all failed measured
gates. Forecast fitting, the 28-window replay and historical returns are unchanged.
Global H2 optimization and deeper convergence remain open.

## Risk-neutral bounds, sign transfer and admissible fitting (v522–v546)

### Locate the bound slack before building a larger optimizer

V527 decomposes exact fixed-root H2 values under the unchanged November law. At
the -0.72268 BTC incumbent, the exact log value is **0.234902 bp** and Jensen's
upper from exact mean terminal equity is **0.239173 bp**, a gap of only
**0.004271 bp**. At -0.00010 BTC the corresponding values are 0.146346 and
0.148549 bp, a 0.002202 bp gap. The existing opening-informed continuation upper
has about **48.4 bp** of slack. Log utility is therefore not what makes the bound
useless; allowing a different child decision after seeing its opening is.

V525 preserves one common child request while relaxing its financial result in
arithmetic wealth. The one-bin bound prunes `[14,525,14,555]` at -13.1871 bp and
`[14,560,72,263]` at -13.2421 bp. The two intervals near cash remain unresolved
at 9.6215 and 9.6066 bp. Nine request bins in v526 cost roughly ten times as much
and improve those uppers by only about 0.04 bp, so maximum-size eligibility is
not the dominant relaxation.

V529 keeps a shared root endpoint through all first-event outcomes and prunes
`[-72,263,-36,137]` at -0.125959 bp. Near the fill switch, v534 leaves the
two-lot interval `[-11,-10]` unresolved but prunes the singleton `[-10,-10]`.
This refinement is converging to individual-lot evaluation. V535 independently
tests the required risk-neutral convexity theorem on 60 synthetic laws and eight
accounts per law: **480 of 3,133** finite adjacent-lot triples violate it, with a
minimum second difference of -0.137445. These results justify stopping this
search route. The two positive intervals and the far negative half now have
valid fixed-model certificates; the remaining intervals do not have a global H2
certificate.

### Transfer the remembered 1-second sign model to decision-sized events

V537 uses the frozen v401 validation and test predictions from the later
structured sign family. It evaluates barriers of 24 and 48 bp, timeouts of 60,
300 and 900 seconds, four feature sets and a fixed L2 penalty. The first two
thirds of validation fit each head; the final third selects a specification only
when sign information is positive in aggregate and in at least 60% of
timeout-sized blocks. Test is consulted once after selection.

Only the 24 bp, 900-second primitive-feature head passes validation: 0.015922
bits/row, four of five positive blocks, 1.521% return-MSE skill and an 18.339 bp
largest forecast mean. On test it reverses to **-0.021403 bits/row**, has 53.08%
plain and 54.65% magnitude-weighted sign accuracy, and is positive in only six
of sixteen blocks. Return-MSE skill remains +1.682%, but the maximum forecast
mean is **16.361 bp**, below the 24 bp round-trip screen, so it produces no
actionable entries. The remembered model is useful evidence about short-horizon
structure, but this transfer does not justify installing it as the event
direction or reversal head.

### Treat explicit fit windows as training data, then test the consequence

The native fitting protocol previously excluded every catalog interval, even
those named `fit-*` that are never scored. V538 adds an explicit exclusion mode
and makes `non-fit` the default: scored calibration/test windows remain purged,
while unscored fit windows can supply causal history. The old strict behavior is
still available as `all-catalog`. Only the July `shape-flat-low-bias-low-2025-07`
window materially changes because of overlap with `fit-full`; its fit-to-test
gap falls from **107 days to zero**.

The fresh latest-day estimate still predicts the wrong first scored day and
loses **7.9311%**, so merely making the fit recent does not solve the forecast.
V543 instead estimates the unchanged four-leaf law from four admissible days:
11,280 stride rows with uniqueness mass 103.1653. Calibration return CRPS falls
from 16.8463 to 15.443 and the test-prefix CRPS from 17.5498 to 13.1552, although
test return-MSE skill is still -16.331%. Its H1 policy stays cash on calibration
and test.

V544 compiles the exact execution law from all 11,280 samples. V545 then replays
all three July test days: **74 decisions, zero fills, zero return and zero
drawdown**, compared with the old exact H1 replay's -4.49% return and 11.44%
drawdown. Twelve submitted requests are canceled because their required openings
are unavailable; none fill. V546 independently checks every chosen value and
reconciles account and position transitions with zero value and equity error.

This is loss avoidance, not a profitable strategy. It also means a 28-window
rerun would be wasteful because the exclusion change affects only this one
scored window. The result narrows the next work to forecasting a cost-sized move
law with stable out-of-sample value. Position decomposition stays in the ledger
and lifecycle signal layer because it exactly conserves account wealth. It does
not impose independent per-position optima, stable-target assumptions or local
action restrictions; the account-level Bellman coordinator retains the complete
feasible action curve and remains the authority for expected log wealth.

The v522–v546 additions are covered by the focused event-policy tests,
TypeScript typechecks for `bot-algo`, `storage`, `server` and `web`, and a syntax
check of the frozen sign-transfer analysis. No live-trading decision follows
from these results. Reliable profitability, full H2 certification and deeper
convergence remain open.

## Calibration-selected law and causal direction heads (v547–v562)

### Select history length without consulting the scored window

The robust four-day estimate in v543 was compared with otherwise identical 2-,
7- and 15-day laws using calibration CRPS as the frozen selection measure. The
four-day law remains best at **15.443 bp**, versus 16.544, 15.889 and 16.333 bp.
The two-day law is the only alternative that trades in the short screen, and it
then loses 7.9311% on test. The 7- and 15-day laws remain cash. This supports the
four-day law on pre-test evidence; their scored-window behavior is diagnostic,
not part of the selection.

V550 tests a lower 24 bp event barrier on the same July chronology. Its
calibration return-MSE skill is **-4.435%** and class NLL is 1.6763 versus a
1.6032 unconditional baseline. It also remains cash on test. Reducing the event
size does not create a forecast edge and is rejected before any wider replay.

### Test direct conditional-mean learners before scaling Bellman

V551 fits shallow honest mean trees and ridge projections on the partition days,
then scores them only on calibration. None passes the preregistered requirement
of positive return-MSE skill plus lower class NLL and CRPS than the honest
unconditional law. The best ridge MSE skill is **-1.105%** at penalty 0.1. Its
largest expected move is 4.32 bp, well below execution cost, so the branch does
not justify a policy backtest.

The repository's strongest transferable 1-second direction features were then
implemented causally: completed-second aggregate trade-count imbalance and last
aggressor side. Every row requires a same-second immutable trade-flow reference
with `availableAt = openTime + 1000`; missing, future or invalid observations are
rejected. The depth-two event tree in v552 chooses neither flow feature and is
identical to the candle-only tree. In v553 the flow ridge improves modestly over
the candle ridge, reaching **-0.920%** calibration MSE skill and 15.8623 bp CRPS,
but remains negative and its largest expected move is 4.60 bp. Flow is useful
directional information, but the conditional mean is still too small and noisy
to trade.

V561–v562 add buyer-minus-seller VWAP gap, the remaining fast input promoted by
the earlier repository-wide 1-second transfer study. This is a new versioned
feature contract; the original two-flow contract remains valid for saved-model
reproduction. The VWAP law is fit and scored on calibration only and records
`scoredTestLoaded: false`. The tree again ignores flow and exactly reproduces
the candle law. The return-weighted sign blend has +2.449% calibration MSE
skill, a largest expected move of **11.944 bp**, and zero rows above the 12 bp
one-way cost. It is indistinguishable from the prior flow screen economically
and is rejected without consulting the held-out window.

### Isolate the remembered high-accuracy sign behavior

V554–v558 fit a separate penalized logistic sign head while leaving the joint
event magnitude, duration, extrema and successor law frozen. Training uses the
partition interval only. The calibration-only scan fixes four penalties
{0.01, 0.1, 1, 10}, two objectives and blends {0, 0.5, 1}. Ordinary heads fail.
Return-weighted heads are stable when blended 50/50 with the event law, with
calibration return-MSE skill from +2.315% to +2.493%. The frozen selection rule
picks penalty 0.01 and blend 0.5.

V559–v560 consult the held-out prefix once for that selected head. It obtains
**75.00% plain sign accuracy** on 24 event decisions, reproducing the kind of
headline accuracy remembered from the later sign work. The economic diagnostics
do not pass:

- magnitude-weighted direction accuracy is **59.13%**;
- sign log loss is 0.63684, but return-MSE skill is **-20.28%**;
- the largest absolute expected move is **11.974 bp**;
- zero decisions clear the **12 bp one-way** fee-plus-slippage cost, and zero
  clear the 24 bp round trip.

The unblended sign head has +5.75% test MSE skill but only 45.83% plain sign
accuracy and an 8.14 bp largest expected move. Selecting it after seeing test
would violate chronology and still would not create an actionable entry. The
separate sign model is therefore preserved as a reproducible diagnostic, not
installed in Bellman.

These results resolve the position-decomposition choice for the current branch.
For a fixed account, signed request and signed target inventory are bijective;
the representation itself cannot change the optimal action. The ledger remains
because it exactly accounts for lots, debt, fees, borrowing, entries, reductions,
exits and reversals. Independent virtual-position optima and target-agreement
rules remain excluded because v519 measured strong dependence on the full
account state. The account-level Bellman coordinator alone selects the feasible
net order from the complete action curve. In this test that choice is cash,
which preserves wealth when the 75% sign statistic has no fee-sized expected
edge.

The v547–v562 work is covered by **166 focused tests**, including causal
trade-flow availability and future-isolation checks, and TypeScript typechecks
for every workspace package. No live-trading decision follows from these
results. Reliable profitability and a stable cost-sized forecast remain open.

## Final refit, compact H2, forecast ensembles and drawdown state (v563–v615)

### Refit only after the calibration choice is frozen

V563 promotes the already selected four-day law through the calibration day and
then consults the final day. Its return MSE skill is -7.931%; exact H1 makes 12
trades, holds a 5x long and loses 11.13% peak to trough. Adding raw day context
in v564 worsens calibration. The robust day context in v565 also fails the
calibration gate. These results reject a calendar-label patch and show that a
final refit can make the selected model worse even when its structure is held
fixed.

V566 replaces raw calendar identity with a causal rolling partition. The robust
v567 law improves calibration CRPS from 14.1718 to **13.4878 bp**, but calibration
return-MSE skill is still -13.4468%. The two visited leaf means are +9.2184 and
+0.2684 bp. Exact H1 compilation in v568 produces 11,280 observed paths; v569
replays 25 final-day decisions with zero fills, fees, drawdown or return. Six
requests are canceled by unavailable next openings. The exact one-step policy
therefore remains cash in realized wealth terms.

V572 compresses the four execution leaves from 1,805/9,663/193/127 atoms to
119/154/66/59 positive observed-path atoms. It preserves the selected moments
to a maximum error of 5.24e-12. Compact H1 agrees with the full law at leaf 0;
the leaf-1 discrepancy is a 0.03093 bp value difference and defines an explicit
approximation error rather than an exact certificate. Compact H2 values at cash
are 0.02411 bp for leaf 0 and 0.29993 bp for leaf 1.

The coarse grids and singleton searches in v575–v580 retain +0.45398 BTC at
leaf 0 with 3.19581 bp and find -0.00005 BTC at leaf 1 with 0.30552 bp. Four
non-singleton request intervals remain per leaf. The guarded recursive search in
v583/v584 deliberately stops at its 180-second cap: it improves leaf 1 to
+0.00012 BTC and 0.32313 bp but does not close the global upper bounds. Full-law
rescoring in v585/v586 proves that the leaf-0 +0.45398 BTC request beats cash,
with a 3.03936 bp lower value, but the exact fixed-request upper remains 11.2517
bp. This is useful evidence for the action, not global H2 optimality.

### Propagate direction through the Bellman path

V587 applies a compact receding-H2 candidate policy with exact realized
next-open accounting. It loses **4.0298%**, pays 59.87 in fees and reaches 5.76%
drawdown. A 0.22835 BTC long stays open through the day and loses 341.31 gross.
The leaf-0 sign is correct on 8 of 10 events, yet two large adverse moves make
its realized mean negative; leaf 1 is positive on only 5 of 15 events. This
isolates stale sign and severity forecasts, rather than ledger accounting, as
the primary failure.

V588 preserves every per-decision sign forecast. The selected 50/50 sign blend
is correct on 75% of events, but the unequal positive and negative conditional
magnitudes keep its expected return positive on many events that its direction
score classifies as down. V589 reweights only the observed root and cuts the
loss to -0.00686%. V590 attaches the same causal sign calculation to every
observed estimation path; v591 compresses the enriched law. V592 then reweights
both the current root and every H2 successor. It loses only **0.001795%**, with
positive gross trading PnL of 0.10846, but 0.28682 in fees and 0.00118 in
borrowing overwhelm that edge. V593's 0.03093 bp hold deadband worsens the loss
to 0.005224% because it preserves wrong inventory. This repeats the earlier
finding that blanket hysteresis can reduce turnover while increasing adverse
persistence.

### Combine sign and magnitude by measured reliability

The exact saved one-second comparison proposed by the user cannot be reproduced
from the current files without retraining. The compressed-path model's causal
12k affine calibration recorded **6.340% validation / 7.041% test** step-1 MSE
skill, while the structured production59/base17 family recorded 75–77%
validation direction accuracy. Their original datasets and checkpoint files
have since been cleaned; aggregate result JSON remains, but aligned per-example
predictions do not. The November v401 export contains only the historical
structured model. Scaling GPU training merely to recreate an unproven stack is
therefore deferred.

V594/v595 test the same idea at the current event horizon, where aligned
predictions are available. The magnitude input is the frozen event law's
expected absolute return under the full sign-head probability. Direction is the
selected 50/50 probability blend. Candidate transforms fit on the first half of
calibration, select on the second half by return MSE, refit on all calibration,
and consult the 24-event test once. This respects the repository's measured
sign/magnitude dependence: positive and negative conditional magnitudes stay
separate rather than assuming independence.

The selected transform is

\[
\hat r = 1.579656\,(2p_{dir}-1)\,
  \left[p_{mag}E(|r|\mid r>0,x)+(1-p_{mag})E(|r|\mid r<0,x)\right].
\]

It improves full-calibration MSE skill from +2.493% to +4.002% and final test
skill from **-20.281% to -9.113%**, while retaining 75% direction accuracy. The
test maximum rises to 13.361 bp. This is a real forecast improvement, but skill
is still negative and the fitted scale moves from 0.762 on early calibration to
1.580 after the full refit, which is a stability warning.

That warning matters economically. Mapping the ensemble mean back into each
joint execution kernel and propagating it through H2 loses **3.3530%** on the
matching one-day replay in v597, with 59.86 in fees and 4.88% drawdown. The
three-day extrapolation in v596 loses 0.5864%. A small improvement in average
MSE crossed a discrete action boundary and created a roughly 0.229 BTC long.
The ensemble is rejected and the lower-loss v592 sign policy remains the useful
experimental reference.

### Preserve bankable profits and shrink uncertain actions

V598–v602 add a path-dependent risk guard around the candidate Bellman policy.
The state contains initial equity and a high-water mark of **net liquidatable
equity**, after reserving the current flatten cost. Every proposed root order is
evaluated across all fitted execution paths and reserves the cost of flattening
again at the next decision. A request is admissible only if its worst supported
next equity clears

\[
F_t = \begin{cases}
W_0(1-b), & H_t\le W_0,\\
W_0 + \rho(H_t-W_0), & H_t>W_0,
\end{cases}
\]

where `b` is initial risk capital and `rho` is the fraction of bankable peak
profit protected. If no candidate is admissible, the guard selects the request
with the highest worst supported equity. This is a safety filter around the
candidate H2 values; the hypothetical child optimizer does not yet carry the
floor as Bellman state, so this is not the globally optimal risk-constrained H2
policy.

Two implementation details are material. V598 initially omitted the next
flatten cost; v599 adds it. V600 then appeared to improve the sign policy to
-0.000170%, but its peak used marked equity. At 05:00 the apparent 0.0092 profit
was already negative after liquidation costs, so that result is rejected as a
lucky early exit. V601 uses liquidatable equity and correctly reproduces the
unconstrained v592 result under a 1 bp initial budget because every small action
fits inside that budget.

V602 is a deliberately labeled post-test diagnostic at 0.1 bp initial risk and
50% profit protection. It reduces the loss to **0.000598%**, trades from 20 to
7, fees from 0.28682 to 0.06558 and drawdown from 0.001986% to 0.000689%. Final
equity is 9,999.94018, above the fitted-support floor of 9,999.90. It cannot be
selected or promoted after seeing this day. A valid follow-up must choose `b`
and `rho` entirely on an earlier replay and then freeze them for a fresh window.

This risk state is consistent with Busseti, Ryu and Boyd's
[risk-constrained Kelly formulation](https://stanford.edu/~boyd/papers/pdf/kelly.pdf):
maximize expected log growth while constraining drawdown risk, rather than
replacing the growth objective with an unrelated score. The current hard floor
is more conservative on the fitted support. It is also model-dependent. No
nonzero position can literally guarantee preservation against unbounded gaps;
the reported guarantee covers only the saved execution paths and explicit cost
stress.

V603–v606 then carry the floor through the H2 continuation. The reusable exact
one-event optimizer accepts an absolute minimum liquidatable equity and clips
every affine order region against that floor. Exhaustive enumeration across the
existing 960 randomized account/terminal cases verifies the constrained optimum
and selected order. At every first-event outcome, H2 updates that branch's
liquidatable high-water mark, raises its child floor when the branch makes a new
peak, and globally optimizes the child action subject to the new floor. The root
action remains a compact candidate search, so this is exact conditional H2
evaluation of each candidate rather than a global root-action certificate.

V604 repeats the 0.1 bp / 50% post-test diagnostic with the complete state. It
returns **-0.000699%**, executes 7 trades, pays 0.05238 in fees and has
0.000778% maximum drawdown. Its lowest realized liquidatable endpoint is
9,999.93006, above the fixed 9,999.90 floor. Return is slightly worse than
v602's root-only -0.000598%; lower turnover did not compensate for worse gross
trade selection.

V605 applies the complete H2 guard to the rejected magnitude ensemble with a 1
bp initial budget and 50% profit protection. It returns **-0.008150%**, pays
0.26031 in fees and has 0.008227% maximum drawdown. The lowest realized
liquidatable endpoint is 9,999.18496, above the 9,999 floor. This contains the
unconstrained v597 loss of 3.3530% and modestly improves on the root-only v599
loss of 0.009585%, but it still loses money.

V606 tests the user's literal no-giveback principle by setting `rho=1` with the
same 0.1 bp initial budget. It makes **zero trades**, pays no costs, has no
drawdown and returns 0%. Every child law retains an adverse outcome and closing
costs, so preserving each new branch peak exactly removes the continuation value
that justified the entry. This is direct evidence that complete peak protection
gets in the way of account growth for this fitted law. A partial protected
fraction can trade, but its value must be selected before evaluation.

Current decision: keep the sign-conditioned joint law and successor propagation;
reject the magnitude ensemble and fixed hold deadband; retain the lifecycle
ledger; and keep the account-level coordinator. The implementation now matches
the requested risk-state structure for the two-event horizon. The next empirical
gate is still a strictly pre-test choice of `b` and `rho`, followed by a fresh
native window. Longer Bellman horizons must carry the same state at every
recursion. No live-trading policy is promoted.

### Global root coverage and explicit forecast uncertainty — v607 through v615

V607 globally enumerates every floor-admissible root lot under the compact law
with the literal `rho=1` protection rule. All 25 decisions are certified and the
policy remains cash. V608 repeats the enumeration with partial protection, but
only 17 of 25 realized states have a feasible constrained H2 continuation; its
result is not a complete H2 certificate. Transition-equivalent root requests
are then collapsed before child evaluation. V610 uses that optimization with a
1 bp initial budget and `rho=0.5`: 16,312 admissible lots collapse to 2,739
distinct first-event account transitions, and all 25 roots are evaluated in
191.52 seconds. The fitted-law optimum loses **0.008665%**, pays 0.58411 in fees
and reaches 0.009195% drawdown. This showed that missing root candidates were
not the cause of the loss under that compiled law.

A later audit found that v589–v593, v604, and v607–v610 do not represent the
intended sign law. `adjustedProbability` and `directionProbability` are already
ordinary `P(up)` values, but the experimental replay passed them through
`eventProbabilityFromReturnWeight` a second time. Their exact execution and
risk-accounting results remain valid for the distorted saved probability law;
their returns are rejected as evidence about the intended sign forecast. The
sign screen itself and the mean-mapped ensemble path are unaffected. V613 and
later use direct ordinary sign reweighting and assert that the resulting atom
mass reproduces the requested `P(up)`.

V611 fits the convex base/head expected-return blend on 2,760 calibration rows
and block-bootstraps contiguous one-hour groups. The calibration-only central
weight is 0.615621; its 90% interval is `[0.233858, 1]`. The interval contains
zero expected return on 85.11% of calibration rows and 79.17% of the 24 scored
rows. Mean robust absolute forecast is only 0.23645 bp on calibration and
0.20134 bp on the scored day; no row clears the 12 bp one-way cost. The central
scored return-MSE skill is -10.69%. The full-head endpoint happens to have
+5.75% scored MSE skill, but selecting that endpoint after observing this day
would be leakage.

The exact one-event optimizer now accepts multiple probability laws and chooses
one order that maximizes worst expected log wealth. Its ambiguity-set result is
checked against exhaustive enumeration together with the liquidatable-equity
floor in all 960 randomized account/terminal cases. H2 applies the ambiguity
set recursively to causal successor forecasts. V613 uses this corrected nested
maximin policy with a 1 bp initial budget and 50% profit protection. It still
loses **0.003049%**, makes 16 trades, pays 0.18191 in fees and reaches 0.003961%
drawdown. V614 additionally requires paired worst-case improvement over holding
under the same probability endpoint. It loses **0.002467%**, makes 19 trades
and pays 0.20791. Its traded paired advantages range from only 0.001262 to
0.134772 bp, with a 0.023351 bp mean. This is optimizer's-curse evidence: the
fitted continuation values report tiny common gains that are not stable in the
realized path.

V615 makes forecast ambiguity an exposure constraint. When the 90% ordinary
sign interval crosses 50%, a request may retain or reduce same-side quantity,
but cannot open, enlarge or reverse risk. All 25 intervals cross 50% on this
day. Starting from cash therefore removes every executable risk-increasing
request; 201 compact candidates are filtered and 75 hold-equivalent candidates
are evaluated in 6.49 seconds. The result is **0% return, zero trades, zero fees
and zero drawdown**. This implements the requested withdrawal behavior and
prevents either of the observed adverse events from consuming equity. It does
not satisfy the profit objective. The next productive step is to narrow a
strictly pre-test forecast interval with better causal sign or magnitude
information; relaxing the guard or scaling global search cannot create
fee-positive evidence from the current interval.

### Full-window withdrawal and separate selected sign inputs — v616 through v626

V616 freezes the v615 rule across the complete three-day July inspector window.
Seventy-two of 74 direction intervals span 50%. The other two still have no
positive robust value after execution costs. The policy remains cash at all 74
decisions: **0% return, zero fees and zero drawdown** in 23.75 planner seconds.
This is the desired response to unresolved uncertainty, but it also confirms
that the current forecast cannot fund a trade. The 1 bp initial loss allowance
and 50% liquidatable-profit floor remain branch state throughout H2. Exact
`rho=1` peak protection remains rejected by v606 because it removes every
entry under the two-sided saved law; protecting half of each peak prevents one
supported adverse event from erasing all accumulated profit while leaving some
risk budget for growth.

The next screen keeps the final-refit event tree as the magnitude, duration,
extrema and successor law and gives only the sign head a new versioned causal
feature basis. The added coordinates are the previous completed-second return,
normalized adjacent-return Haar contrast over 16 completed seconds, the last
aggressor side two seconds old, current taker quote-volume imbalance, its
separate-buy/sell EMA(2), and current maximum aggregate-size skew. Existing
aggregate-count imbalance and current last side remain. These were selected
before this run from the long-window feature audits; no scored-window feature
selection is used. Rich trade-flow fields retain exact close availability and
the EMA is reconstructed from 64 completed bins. A dedicated causality test
covers values, future mutation, missing fields and stale timestamps.

At penalty 0.01, the 50/50 base/head blend improves calibration
magnitude-weighted direction from **56.9822% to 57.7281%**, return-MSE skill
from **2.4935% to 2.6398%**, and one-way-cost exceedances from 7 to 12 of 2,760.
The 90% blend interval becomes `[0.238613, 1]`; ambiguous calibration rows fall
from 85.11% to **82.68%**. The first scored day is still weak: 87.5% ambiguous,
-10.72% central MSE skill and zero robust moves over 12 bp. Compiled v620 and
quadrature v621 preserve the magnitude law while evaluating the new sign
features independently. Full-window H2 v622 again has 72 ambiguous and two
unambiguous decisions. Their central expected moves are only 4.18 and 5.81 bp;
even minimum legal entries have negative worst-case value. The result is again
**0% with no trades, fees or drawdown**.

A bounded calibration-only penalty check selects 0.001 over 0.01 and 0.1 by
the declared economic diagnostics: 57.8244% magnitude-weighted direction,
2.7149% MSE skill and 17 cost exceedances. Its frozen first-day and bootstrap
runs v625–v626 still have an interval `[0.240013, 1]`, 81.01% ambiguous
calibration rows, 83.33% ambiguous test rows and no robust cost-clearing move.
Further tuning of this linear head is therefore stopped. The new features carry
some signal, but the historical production59 75–77% result is not a substitute:
that model predicts active next-second signs, and its saved largest-move decile
is only about 57% accurate. Its dedicated event-transfer screen was already
negative. The next forecast iteration needs a causal nonlinear or sequence
model trained directly on cost-sized event outcomes, with calibration
uncertainty as part of its promotion gate.

### Nonlinear mean rejection and a larger protected event target — v627 through v650

V627 initially fit its nonlinear-head blend on the magnitude law's own
estimation day, where the base law is in-sample; its zero blend weights are
therefore rejected as a split-design error. V628 corrects the chronology: a
depth-two boosted mean head trains on the three partition days, the first half
of calibration fits the convex base/head blend, and the second half selects
among 12 predeclared small configurations. The selected eight-tree head has
+3.696% second-half MSE skill and improves over the base by 13.486%, with eight
of 12 positive hourly blocks. It does not transfer in v629: first-day skill is
-10.971%, magnitude-weighted direction is 47.381%, and no forecast clears the
24 bp round trip. The head is rejected.

A selective-prediction diagnostic also fails. On calibration, the top 2% by
absolute predicted return reaches 94.74% magnitude-weighted direction but only
23.851 bp mean gross return, 0.149 bp below round-trip cost. Its sole selected
test event is wrong and loses 48.486 bp gross. Expanding sign training from
three to 15 contiguous days in v630 dilutes the current regime: calibration
weighted direction falls from 57.824% to 55.926% and MSE skill from 2.715% to
2.175%. A 30-day run is not justified.

V631 raises the barrier to 96 bp while retaining a one-hour timeout. More than
96% of calibration labels time out, so the median absolute return remains only
15.808 bp and calibration MSE skill is -11.350%. V632 instead pairs the 96 bp
barrier with a four-hour timeout. A 50/50 selected sign blend appears promising
on calibration (+12.416% MSE skill, 63.928% weighted direction and 631 of 2,400
rows over round-trip cost), but v635 reverses to -86.475% MSE skill on the six
non-overlapping first-day events. The calibrated blend-weight interval in v636
is `[0.375483, 0.784893]`; it is confidently wrong on the largest event. Blend
coefficient uncertainty alone does not cover regime error.

The four-day average-uniqueness magnitude law is stronger than that separate
head. V637 improves calibration CRPS from 38.513 to 27.611 bp, obtains +16.991%
return-MSE skill and earns +2.0026% in exact H1. The separate sign head reduces
weighted calibration direction from 67.069% to 39.382%, so it is rejected for
this target. The frozen law still fails the first scored day in v639: it enters
near 5x after a +17.30 bp event, holds through the next -97.33 bp event and
returns -4.5107%. This is exactly the small-win/large-adverse-loss pattern the
capital rule must contain.

V640–v650 compile that law to observed execution paths and test the combined
response. A 1 bp initial risk budget and 50% liquidatable-peak protection shrink
the full-window H2 policy from near 5x to less than 0.01x. The complete
zero-deadband lattice in v644 covers all 19 decisions and 10,371 admissible lots;
drawdown falls from the unconstrained control's 8.4373% to 0.00947%, though tiny
fee-dominated orders still lose 0.00724%. On preceding calibration the only
opening action has 0.24376 bp of constrained H2 advantage. Freezing a 0.20 bp
minimum-advantage deadband leaves the positive calibration replay unchanged and
removes the earlier adverse test entry. V650 covers all 19 decisions and 10,789
admissible lots and exactly reproduces v649: **+0.000028% net, 0.006352%
drawdown, three fills and 0.19692 fees** over all three July days. Once its
liquidatable equity peaks, the branch floor rises and the remaining decisions
stay cash.

This validates the mechanism, not reliable profitability. The unconstrained
full-window control eventually recovers to +1.0021%, so the 1 bp budget gives up
material upside while preventing its large interim drawdown. The next policy
iteration should make the initial budget depend on a pre-test estimate of
forecast uncertainty and skip minimum-size orders whose robust advantage cannot
absorb calibration error. A replay integration bug found during this work is
fixed: magnitude models that use trade-flow features now load and hash those
shards even when no separate sign head is attached.

The literature search found a direct recent proposal in
[Conformal Kelly](https://arxiv.org/abs/2608.01494): use a slowly estimated
conformal residual width as the scale in fractional Kelly and cut leverage when
downside coverage deteriorates. Its sealed-data coverage transferred but its
claimed growth advantage did not, so only the uncertainty mechanism is adopted.
This is also consistent with
[distributionally robust Kelly](https://arxiv.org/abs/1812.10371), which
maximizes the worst expected log growth across an ambiguity set, and the
[risk-averse calibration](https://proceedings.mlr.press/v267/kiyani25a.html)
result that a maximin action is the correct policy for a risk-averse decision
maker given a prediction set.

V651 fits a 75% split-conformal absolute-residual radius from the six ordered,
non-overlapping pre-test calibration events. The finite-sample corrected radius
is **79.189 bp**. All four leaf-mean intervals span zero; for example the
strongest 42.955 bp leaf becomes `[-36.234, 122.143]` bp. V652 propagates both
endpoints through current and successor H2 kernels and applies the same
exposure-withdrawal rule. It remains cash at all 19 decisions and returns 0%
with no fees or drawdown. This is the honest uncertainty-selected result. The
tiny positive v650 result remains a useful mechanism diagnostic, but its 0.20
bp deadband was inspected on this research window and is not promoted. More
independent pre-test cost-sized events or a substantially narrower residual law
are required before conformal sizing can permit nonzero risk.

V653 extends the chronology helper from one to a declared seven-day calibration
block while preserving one day as the default. The 96 bp/four-hour tree then
has 19,680 calibration stride origins, or roughly 42 non-overlapping event
decisions. Its apparent one-day edge disappears before test: return-MSE skill is
-8.894%, direction is 52.40% and CRPS is 27.643 bp. Applying the same four-day
average-uniqueness re-estimation in v654 worsens CRPS to 31.234 bp and loses
14.4384% in calibration H1. The 96 bp target is therefore rejected without a
second test consultation. The one-day v637 selection was a narrow favorable
slice, not stable evidence. Future event candidates must pass the broader
calibration chronology before conformal sizing or Bellman compilation.

### Confidence-scaled profit preservation — v659

V659 makes the capital floor respond continuously to the saved forecast
interval. Directional confidence is the fraction of the point forecast that
remains after moving the adverse endpoint toward zero. An interval which
touches or crosses zero has confidence zero; a point interval has confidence
one. Confidence scales both the maximum initial-risk allowance and the
unprotected share of liquidatable high-water profit. With maximum initial risk
`b`, minimum protected-profit fraction `rho`, confidence `c`, initial equity
`W0`, and high-water profit `P`, the active floor is

\[
F_t = \begin{cases}
W_0(1-cb), & P=0,\\
W_0 + [1-c(1-\rho)]P, & P>0.
\end{cases}
\]

The same calculation is carried into every H2 child using that successor
state's causal uncertainty interval. Root and child feasibility still reserve
the next flatten cost and check every saved adverse execution path. Therefore
zero confidence forces withdrawal to the current liquidatable high-water mark,
while narrower intervals smoothly restore the declared risk budget. Unit tests
cover intervals on both sides of zero, partial confidence, seed risk and profit
protection.

The frozen 75% conformal test is intentionally unchanged. All 19 July decision
intervals cross zero, so all 19 confidence values are zero, effective initial
risk is zero bp, and the protected-profit fraction is 100%. V659 therefore
returns **0% with zero trades, fees and drawdown**, versus the same unconstrained
control's +1.0021% return and 8.4373% drawdown. This is the correct action for
the measured uncertainty and prevents the two adverse events from consuming
prior equity, but it cannot turn an unresolved forecast into profitable risk.
The guarantee is conditional on the saved execution-path support; an unbounded
market gap cannot be covered by nonzero exposure. The remaining bottleneck is
still a causal cost-sized forecast whose pre-test interval excludes zero.

Artifact: `data/benchmarks/event-native-barrier96-4h-h2-conformal75-dynamic-risk-july-v659/summary.json`.

### Broader 48 bp chronology and completed-event sequence screen — v660 through v662

V660 starts a new calibration-only branch with a 48 bp barrier, one-hour
timeout, seven fit days and seven later calibration days. It uses the selected
native sign/flow basis and a separate final fit day for distribution
estimation. No July inspector candle is loaded. The 20,040 calibration stride
origins contain more observations than the 96 bp branch, but 17,586 (87.75%)
still time out. The frozen four-leaf model has **-1.1525%** calibration
return-MSE skill, **49.2437%** nonzero direction accuracy, and leaf means from
**-4.5404 to +5.1296 bp**. It fails before policy compilation.

V661 tests the specific sequence hypothesis rather than adding more price
transforms. It converts the fit and calibration populations into causal,
non-overlapping event chains: 235 fit events and 177 calibration events. Each
row can observe at most sixteen prior completed events within one day, including
their signs, sizes, durations and path efficiency. The calibration chain is
split chronologically into 88 selection and 89 untouched validation events.
None of the twelve fixed ridge mean specifications passes the selection gate.
The numerically best regularized context/history head improves the already weak
base by 0.1182% on selection but remains **-0.1157%** versus predicting zero;
weaker penalties overfit substantially.

V662 evaluates the requested separate-sign combination on the same sealed
chronology. Causal logistic heads use ordinary or return-weighted sign loss;
their probabilities are blended with the frozen leaf-specific positive and
negative magnitudes and active mass. Sixteen candidates pass the selection
gate. The selected 50/50 return-weighted blend reaches **+0.2828%** MSE skill
versus zero, **+0.5158%** improvement over the base and **+0.01164 bits** of
sign information on selection. On untouched validation it remains +2.5287%
versus zero but falls **0.8569% behind the base**, improves on only one of four
UTC days, and adds just +0.00348 sign bits. Its largest mean is **3.3223 bp**
and its 75% absolute-residual radius is **30.3243 bp**, leaving zero directional
or cost-clearing intervals. The candidate is rejected without loading the test
window.

A heteroscedastic interval refit is not run for this branch. Even a hypothetical
zero-width interval could preserve at most the 3.3223 bp point-forecast margin,
which is below the 24 bp round-trip cost. Rescaling uncertainty cannot create
economic edge absent from the mean. The documented external-feature audit also
does not provide a ready substitute: historical spot books do not cover this
2025 chronology, archived futures depth has only a narrow next-second sign
result, and the current native basis already includes the supported spot-flow,
volatility and activity inputs. The sequence branch is therefore stopped before
backtesting; the confidence-scaled capital guard remains the correct active
behavior until a forecast passes the pre-test economic gate.

Artifacts:

- `data/benchmarks/event-native-barrier48-selected-history-cal7-july-v660/summary.json`
- `data/benchmarks/event-native-barrier48-history-mean-cal7-july-v661/summary.json`
- `data/benchmarks/event-native-barrier48-history-sign-magnitude-cal7-july-v662/summary.json`

### Futures increment, softer capital sizing, and honest leaf support — v663 through v669

V663 tests archived futures information that was absent from the native spot
basis. A causal adapter exposes only the latest fully completed futures minute
paired with the matching completed spot minute. The fixed views cover futures
activity, centered basis and relative returns, and taker imbalance. A causality
test verifies that changing the unfinished or future minute cannot change the
features. The 48 bp/one-hour screen has 235 fit-chain and 177 calibration-chain
events. One activity candidate passes the first 88-event selection half, but on
the untouched 89-event half it is **1.2126% worse than the frozen law**, improves
only two of four days, and adds just 0.00103 sign-information bits over its
matched base-only head. Its largest expected move is 2.931 bp against a 30.003
bp 75% residual radius. It is rejected without loading the test window.

V664 raises the 48 bp timeout from one to four hours. The seven-day calibration
timeout share falls from 87.75% to **44.34%**, but MSE skill remains -1.9029%
and direction accuracy is 51.7581%. V665 expands calibration to 14 days. The
39,840 stride origins contain 194 non-overlapping event-chain observations;
timeout share falls to **32.21%**, but MSE skill worsens to -3.0254%, direction
is 50.5447%, and one of four tree leaves has zero separate estimation rows.
V666 repeats the incremental futures screen on 97 fit and 194 calibration-chain
events. No candidate passes selection. The best incremental-MSE specification
adds only 0.01297% over its matched base head, loses 0.21784% to the frozen law,
and has negative incremental sign information. Futures activity and basis are
therefore not promoted for either tested event clock.

V667–v668 evaluate a softer form of the requested uncertainty response. The
central forecast ranks actions while the conformal interval scales only the
loss budget by

\[
c_{\mathrm{soft}} = \frac{|\mu|}{|\mu|+r},
\]

where `r` is the maximum distance from the point forecast to either interval
endpoint. The same high-water floor and every saved adverse execution path,
including flatten cost, still constrain root and H2 child actions. With
`rho=0.5`, a supported adverse event can consume at most half of accumulated
liquidatable high-water profit, and less when uncertainty is wider. The frozen
0.20 bp action-value threshold in v667 rejects every trade and returns 0%. A
zero-threshold diagnostic in v668 permits two tiny long round trips. Both have
positive gross price P&L totaling **$0.0087355**, but **$0.0388402** of fees
make net return **-0.0003010%** with 0.0007774% drawdown. The capital rule removes
the large-loss pattern, while the existing evidence threshold correctly
rejects the remaining fee-dominated actions. Soft sizing is retained as a
diagnostic option, not promoted over the robust withdrawal rule; the scored
window had already been inspected.

V669 adds an honest covariate-support constraint to tree construction. Split
scores still use only partition outcomes, but a split is inadmissible unless
both children contain at least 128 rows from the separate estimation population.
Changing estimation outcomes while holding their features fixed leaves the tree
unchanged. On the 14-day calibration chronology the empty four-leaf partition
becomes a fully supported three-leaf partition with counts 128, 2,290 and 201.
This does not fix prediction: direction remains 50.5447%, MSE skill changes from
-3.0254% to **-3.0417%**, and return CRPS improves only from 24.1754 to
24.1668 bp. The constraint is useful structural hygiene but is rejected as a
forecast improvement. No v663–v669 result changes a live bot, and no new test
window was loaded for the forecast screens.

Artifacts:

- `data/benchmarks/event-native-barrier48-futures-sign-cal7-july-v663/summary.json`
- `data/benchmarks/event-native-barrier48-4h-selected-cal7-july-v664/forecast.json`
- `data/benchmarks/event-native-barrier48-4h-selected-cal14-july-v665/forecast.json`
- `data/benchmarks/event-native-barrier48-4h-futures-sign-cal14-july-v666/summary.json`
- `data/benchmarks/event-native-barrier96-4h-h2-conformal75-soft-sizing-july-v667/summary.json`
- `data/benchmarks/event-native-barrier96-4h-h2-conformal75-soft-sizing-zero-july-v668/summary.json`
- `data/benchmarks/event-native-barrier48-4h-selected-cal14-support128-july-v669/forecast.json`

### Competing risks and independently weighted direction — v670 through v675

The preceding joint tree asks one shallow partition to predict direction,
duration and timeout behavior together. V670 therefore factorizes each frozen
48 bp / four-hour leaf into three masses: down-barrier hit, timeout and
up-barrier hit. Two penalized logistic heads estimate
`P(barrier before timeout)` and `P(up | barrier)`. Reweighting changes only
those masses; the leaf's conditional returns, duration, extrema, paths and
successor states remain intact.

The first prototype confirms that factorization can shrink a bad mean. On its
untouched calibration half, return-MSE skill changes from -2.1192% to +0.0413%.
That apparent gain is insufficient: group NLL worsens from 1.09235 to 1.18125,
and every predicted mean falls below round-trip cost. A probability model that
collapses toward zero can improve squared error without supplying a trade.

The repository's recommended basis specifically calls for explicit multiscale
volatility. `NATIVE_SECOND_VOLATILITY_CONTEXT_FEATURES` now adds causal
15-minute, 30-minute, one-hour and four-hour rolling return volatility. It uses
one-hour log-RMS volatility as the anchor and three scale contrasts. A segmented
prefix-sum cache makes all four coordinates O(1) per origin after one linear
source pass, while rejecting history that crosses missing seconds. Tests verify
the exact rolling variances, declared four-hour support and future invariance.

V675 is the honest final screen. It fits on the pre-estimation partition interval,
selects penalties plus separate arrival/direction blend weights on the first
seven calibration days, and requires pooled NLL and MSE gains on at least 60%
of UTC-day blocks. The untouched later seven days are read once. Independent
blend weights directly test weighting the arrival/magnitude and direction
experts by their separate evidence.

| Metric | Frozen joint leaf | Selected competing risks |
| --- | ---: | ---: |
| Selection return-MSE skill | -3.6730% | +0.4109% |
| Selection group NLL | 1.19643 | 1.00838 |
| Selection magnitude-weighted direction | 47.95% | 54.83% |
| Holdout return-MSE skill | -2.1192% | -0.6912% |
| Holdout group NLL | 1.09235 | 1.10212 |
| Holdout magnitude-weighted direction | 53.54% | 43.99% |
| Holdout maximum absolute mean | 34.672 bp | 6.537 bp |
| Holdout means above 12 / 24 bp | 994 / 994 | 0 / 0 |

The selected full-weight heads reduce holdout MSE by 1.3983% relative to the
frozen law, and improve five of seven daily MSE blocks, but fail the NLL,
direction and economic gates. Arrival is regime-dependent: timeout frequency
rises from 20.38% in selection to 44.34% in holdout, while the selected head
predicts 16.68% and 29.35%. Keeping the frozen direction while changing only
arrival preserves its 53.54% weighted direction accuracy, but does not improve
mean skill. Conversely, using the new direction head creates the MSE shrinkage
by removing cost-sized means and flips the useful direction signal. This is a
clean rejection, not a policy loss requiring more Bellman computation.

Preserve the factorized evaluation and multiscale volatility implementation for
future models, but do not integrate this head into execution. The next forecast
candidate must improve cost-sized direction on chronological blocks; uncertainty
sizing already handles capital withdrawal once such an edge exists.

Artifacts:

- `data/benchmarks/event-native-barrier48-4h-competing-risks-independent-blends-july-v675/summary.json`

```powershell
node --conditions=development --import tsx scripts/screen-native-event-competing-risks.ts --source event-native-barrier48-4h-selected-cal14-july-v665 --features volatility-context --output event-native-barrier48-4h-competing-risks-independent-blends-july-v675
```

### Slower direction transfer, reliability ensemble and uncertainty withdrawal — v676 through v691

The later 75–77% structured next-second classifier did not transfer directly to
the 48 bp event horizon in v537. V676–v678 therefore test a different use of
scale: train the old fixed minute-event specification strictly before native
calibration and use only its probability that a 120 bp barrier resolves upward
within one day. The model uses 5,731 events, four leaves and only the last fully
completed minute at each native origin. The first prototype mislabeled its
forecast field and is discarded. The corrected v678 selects full slow weight
and unit log-odds scale on the first seven days.

The slow model is the first positive direction transfer in this branch. Its
untouched later-half return-MSE skill is **+2.0359%**, versus -2.1192% for the
frozen native law; group NLL improves from 1.09235 to **1.08103**, and
magnitude-weighted direction rises from 53.54% to **59.62%**. It improves MSE on
five of seven days, but NLL on only three. Its maximum mean is 9.281 bp and no
forecast clears even the 12 bp one-way cost, so slow direction alone is not an
economic policy.

V679 recreates the native return-weighted fast sign head without reading the
inspector window. V680 combines slow and fast probabilities in symmetric
log-odds space,

\[
p_{up}=\sigma\!\left(a\,\operatorname{logit}(p_{slow})
  +b\,\operatorname{logit}(p_{fast})\right).
\]

The selection rule chooses `a=1`, `b=0.5`. On the untouched later seven days,
return-MSE skill is **+2.3409%**, sign NLL is **0.68505**, and
magnitude-weighted direction is **58.61%**. MSE improves on five of seven days
and NLL on four. Six stride origins clear 12 bp, but none clears the 24 bp round
trip; the maximum mean is 13.743 bp. This supports combining direction experts
according to measured reliability, but does not yet support trading each event.

V681 corrects the execution-law compiler for `native-second-event-screen-v2`:
v665 partitions on earlier fit days but estimates its kernels from the final
`later-fit-stride` day. Reconstructing all seven fit days was both wasteful and
the wrong empirical law. The corrected compiler uses the exact 2,619 estimation
events, completes in 1.99 seconds, and reproduces every saved return, duration,
extrema and successor probability. V682 compresses the four kernels from
253/2,415/125/326 paths to 129/204/90/136 observed paths, with maximum declared
moment error `4.49e-12`. Each path carries the causal slow and fast forecasts at
its start and successor endpoint, so H2 does not assume that the current signal
persists. V692/V693 reproduce that law and compression after pinning hashes for
the slow model/config and fast head/config; a four-decision v694 replay verifies
the provenance-complete loading path.

Every coefficient pair that passed v680's first-half MSE/NLL/day-stability gate
forms an eight-model forecast set. The selected pair ranks actions. Its minimum
and maximum probabilities define the interval used by soft confidence,

\[
c=\frac{|p-0.5|}{|p-0.5|+
  \max(p-p_{low},p_{high}-p)}.
\]

The existing liquidatable-equity state then scales a maximum 50 bp initial loss
budget by `c`. After a peak, it protects
`1-c(1-rho)` of profit with `rho=0.5`. Root and successor actions must clear
that floor under every fitted execution path after reserving a flatten cost.
Thus high disagreement withdraws exposure continuously, and a supported adverse
event cannot erase all accumulated liquidatable profit.

The action results separate forecast quality from capital control:

| Replay | Native H1 control | Ensemble H2 | Trades | Fees | Max drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| Seven-day selection, no guard (v686) | +21.5580% | **-21.5651%** | 57 | $135.48 | 42.2845% |
| Seven-day selection, 50 bp soft guard (v687) | +21.5580% | **+0.000050%** | 2 | $0.0249 | 0.000223% |
| Untouched seven-day holdout, frozen guard (v689) | +8.6268% | **+0.000258%** | 2 | $0.0256 | 0.000247% |
| Three-day non-fit inspector test, frozen guard (v691) | 0% | **-0.000618%** | 4 | $0.0260 | 0.000800% |

Raising the maximum risk to 100 bp in v688 changes no fill because exchange
minimum quantity already binds. V690 keeps the native distribution for H2
ranking and uses the ensemble only for uncertainty. Even a 1,000 bp maximum
seed budget makes only two fills and returns +0.000125% on selection: a small
early profit raises the floor, after which the worst fitted loss from the
minimum order exceeds the permitted giveback. This is correct profit protection
but also shows a cold-start/minimum-lot limitation.

V691 is the decisive rejection. The guard contains the damage, but two short
round trips lose gross price P&L and fees. The second entry has only 0.00091 bp
of fitted H2 advantage and is followed by a +49.66 bp event. The other entry's
advantage is 0.00945 bp. These values are far below the previously selected
0.20 bp evidence threshold, repeating the optimizer's-curse result from v614
and the fee-dominated result from v668. Scaling leverage or enumerating more
root lots cannot make those advantages reliable.

Current decision: retain the slow expert, the reliability-weighted combination,
causal successor enrichment, lifecycle position attribution, and the
uncertainty/high-water mechanism. Do not replace the native action law with the
ensemble, and do not promote v691. The next forecast iteration should preserve
the native H1 policy's sparse multi-event holding behavior and use the ensemble
as a calibrated prior or veto only after an action-value uncertainty bound
clears costs. Exact peak protection remains intentionally partial: v606 already
showed that protecting 100% of each new peak forces cash under a two-sided law.

Artifacts:

- `data/benchmarks/event-native-barrier48-4h-slow-direction-calibrated-july-v678/summary.json`
- `data/benchmarks/event-native-barrier48-4h-direction-ensemble-july-v680/summary.json`
- `data/benchmarks/event-native-barrier48-4h-direction-ensemble-execution-law-july-v681/summary.json`
- `data/benchmarks/event-native-barrier48-4h-direction-ensemble-execution-compact-july-v682/summary.json`
- `data/benchmarks/event-native-barrier48-4h-direction-ensemble-h2-central-selection7-july-v686/summary.json`
- `data/benchmarks/event-native-barrier48-4h-direction-ensemble-h2-soft-risk50-profit50-selection7-july-v687/summary.json`
- `data/benchmarks/event-native-barrier48-4h-direction-ensemble-h2-soft-risk50-profit50-holdout7-july-v689/summary.json`
- `data/benchmarks/event-native-barrier48-4h-h2-base-with-ensemble-risk1000-profit50-selection7-july-v690/summary.json`
- `data/benchmarks/event-native-barrier48-4h-direction-ensemble-h2-soft-risk50-profit50-test3-july-v691/summary.json`
- `data/benchmarks/event-native-barrier48-4h-direction-ensemble-execution-law-provenance-july-v692/summary.json`
- `data/benchmarks/event-native-barrier48-4h-direction-ensemble-execution-compact-provenance-july-v693/summary.json`
- `data/benchmarks/event-native-barrier48-4h-direction-ensemble-h2-provenance-smoke-july-v694/summary.json`
