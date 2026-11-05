# Planning accuracy: independent account bounds and a forecast-law audit

The 0.1% accuracy target for the complete learned NSF law is **not established**.
This experiment replaces the sampled neural-residual argument with independent
account-value upper bounds, improves the simulated policy, and measures a large
distortion introduced by the fixed-width forecast graph. It does not justify
attributing the remaining performance problem solely to the trained predictor.

## Declared experiment

- Gamma 1, initial cash 10,000, zero inventory and empty book.
- One validation forecast origin, 3,600 consecutive one-second transitions.
- Frozen sampled-skip latent NSF checkpoint
  `80fdc3df78a5f60d9980b236f9bbaa296904dc562bd64195394831120dde01db`.
- 256 representative market nodes per depth, 64 conditional draws per parent:
  921,409 nodes and 58,953,792 edges in the completed graph.
- Oracle received-amount maker/taker fees 0.0002/0.0005; raw entry leverage
  [-2,2], effective maintenance [-3,3], reservation leverage 2; minimum quote
  order 5, aggregate quote cap 10,000 per decision, price tick 0.01.
- Unit signed-asset funding, matching this numerical experiment. No substitute
  borrowing charge or absolute-inventory maintenance was introduced.
- All-or-nothing crossing fills at the next decision, one active order in the
  evaluated policy, and terminal zero asset with an empty book through valid
  actual closing trades.

The graph hash is
`b48caaea27d0b2660b4e71435e0389cab70222530c4ffcf0d3b26362dce25abe`.
It is a finite conditional law, not the original unbounded NSF distribution.
The graph build took 545 seconds on this workstation. The final account-bound
calculation took 179 seconds. It uses CPU Numba hull queries and exact integer
inequality checks; policy action evaluation uses CUDA float64.

## Independent upper construction

At each market node retain globally valid account supports

\[
U(q,a,p)=\min_i (c_iq+d_iap)/2^{30}.
\]

The support grid chooses planes; it does **not** restrict the competing
policy's continuous exposure, order notional or legal limit-price choices.
For each proposed parent plane select a valid child support for every outcome.
The conditional coefficients are integer sums of child coefficients and cent
prices. A suboptimal floating-point hull selection can weaken tightness, but
each selected plane still bounds the child value. Arithmetic overflow and failed
endpoint/fee-wedge checks raise errors instead of returning a certificate.

Domination on the two leverage endpoints implies domination throughout the
continuous exposure interval. Market trading cannot increase a plane when

\[
(1-f_t)d\le c,\qquad (1-f_t)c\le d.
\]

Liquidation is handled separately: the edge-price guard ensures the moved
shadow value is nonnegative, so replacing a liquidated account by zero cannot
increase it. Terminal supports dominate actual fee-aware closing cash.
Relaxing minimum sizes, maximum traffic or the order-slot count can only enlarge
the competitor set in the upper calculation; actual evaluated actions retain
those constraints.

The predictable-order refinement keeps each limit price fixed before the next
random outcome. For a chosen child support, the expected increment per unit
gross buy notional at limit `ell` is

\[
\mathbb E[1_{\{P'\le\ell<p\}}
 (-c'+(1-f_m)d'P'/\ell)],
\]

and for a sell it is

\[
\mathbb E[1_{\{p<\ell\le P'\}}
 ((1-f_m)c'-d'P'/\ell)].
\]

With the finite price law, a best threshold for a fixed fill set is at one of
its outcome-price endpoints. Prefix/suffix sums therefore upper-bound all legal
passive prices, including prices absent from the evaluated order menu. The
largest positive coupon is charged against the shared reservation allowance.
For `|ap| <= L W`, `2W-(2/L)|ap|` is an upper envelope for available order gross
notional, including accounts outside the order-reservation cone with no book.
No fresh same-decision fills are admitted. Under unit funding, an unfilled
eligible passive order remains passive and can be retained or replaced for the
following tick without changing its next-tick crossing event.

The observed graph-wide price envelope also tightens reachable raw leverage
between physical trades. If `R=Pmax/Pmin` and entry gross leverage is at most 2,

\[
|e|\le {2R\over3-2R}\le2.015.
\]

This is used only for this graph and unit funding, with a rational outward
rounding. It is not a bound for arbitrary price laws or funding paths.
Convex mixtures of supporting planes provide another valid refinement.

Backward upper backups bound every admissible competitor under the declared
law. Jensen then gives `CE(optimal log policy) <= optimal mean wealth <= U`.
Cash is a feasible, exactly valued lower policy. The final interval is

\[
10{,}000\le CE^*\le10{,}035.1776275784.
\]

Thus the cash-policy relative CE regret is at most **0.3517763%** on this graph.
That number is an upper bound on loss, not a prediction of actual loss.

## Refinements and policy evaluation

| Upper construction | Upper excess over cash |
|---|---:|
| Basic 33-support closure | 0.559462% |
| 129 supports | 0.557572% |
| Convex support mixtures | 0.556002% |
| Direction-specific maker/taker coverage | 0.554310% |
| Predictable limit prices | 0.504640% |
| Reachable leverage 2.015 | 0.352534% |
| Reachable leverage plus hull mixtures | **0.351776%** |

The absolute-quote-cap affine experiment did not improve this: its first
version gave a loose 14.667% bound and its shared-cap/reservation revision gave
0.485039%. Those artifacts are retained as failed/looser refinements.

The continuation planes also supply a practical greedy feedback policy.
Each candidate is evaluated by **averaging its conditional children before
selecting an action**. Actual arrival fills precede querying the post-arrival
continuation. This is not a claim that greediness against an upper function
attains the upper bound.

On 64 simulated graph paths, the richer menu with exposure steps 0.1, order
fractions 0.1/0.25/0.5/1 and passive offsets 1/10/100 ticks produced:

- Mean terminal wealth 10,017.1967.
- Log certainty-equivalent return **+0.171847%**, after fees and valid closing.
- Standard error of mean log return 0.000194493 (about 0.01945 percentage points).
- 249 market trades, 1,602 placements and 653 cancellations across all paths.
- Every evaluated terminal account was flat and feasible; runtime 156 seconds.

The relative difference between the upper and this empirical CE is 0.179620%.
The empirical value is **not** a certified lower expectation. These are paths
sampled from the scenario graph, not historical trading returns or a full
backtest of the original NSF law.

The completed **0.01-exposure** refinement on the same 64 draws gave mean
terminal wealth **10,018.5833**, CE return **+0.185733%**, and a mean-log standard
error of 0.000178179. It executed 369 market trades, 1,762 placements and 724
cancellations in 469 seconds. The upper-to-empirical-CE difference is still
**0.165735%**, before accounting for evaluation uncertainty. Finer market sizing
therefore helped modestly and did not meet the 0.1% target.

An independent replay audit checked all **230,400** recorded fine-run decisions:
initial balances, continuity, the declared random graph draws, actual fill and
market fees, shared collateral, aggregate size limits, placement and activation,
funding/marking/liquidation, and flat terminal scores. It passed against the
current Oracle engine. This is float64 execution conformance, not formal
extraction or a certificate of the policy's expectation.

Execution corrections made during this experiment:

1. A valid uncapped target-zero market execution now sets its algebraically
   zero asset balance exactly to zero. Capped and genuinely undersized residuals
   are preserved.
2. The terminal procedure uses valid capped reductions and, for dust, a valid
   same-side top-up followed by a valid close. It does not waive order minima.
3. Candidate limit orders are rejected if a possible arrival fill would leave
   no feasible leverage repair within the remaining aggregate quote cap.

Earlier failed replays are preserved. The first rich-menu replay encountered
this third issue at tick 542 and is not counted as a completed result.

## The forecast graph itself loses dispersion

The audit propagates the graph's conditional probabilities through every layer
and compares its marginal returns with 64 independently sampled, recursively
re-encoded one-second NSF paths from the same checkpoint and origin.

| Horizon | Graph return standard deviation | Direct-path sample standard deviation |
|---|---:|---:|
| 60 seconds | 2.016 bps | 3.187 bps |
| 300 seconds | 3.089 bps | 6.368 bps |
| 900 seconds | 3.710 bps | 12.250 bps |
| 1,800 seconds | 4.456 bps | 15.748 bps |
| 3,600 seconds | **4.781 bps** | **22.822 bps** |

At one hour the graph has only about 21% of the direct sample's standard
deviation. Its mean is -9.389 bps, versus -7.289 bps in the direct sample
(mean standard error 2.853 bps). The comparison is diagnostic, not a uniform
statistical or Bellman-error bound. It demonstrates why account accuracy on
this graph cannot certify accuracy against the original NSF law.

The nearest-representative projection is not mean/variance preserving. Its
per-tick distortion accumulates when the projected observations are fed back
into the next transition. Finer execution alone does not repair this.

## Formal coverage and remaining work

`TradingTheory/OracleCertifiedValue.lean` proves continuous endpoint domination,
convex support mixing, outcome-dependent funded coefficient backup, liquidation
and actual rebalance inequalities, support-selection monotonicity, the drift
bound and the policy-induction contract. Existing `OracleLogTailBound` proves
the Jensen/CE step. All imported results pass the full Lean build and axiom audit.

The Python verifier has not been extracted from Lean. In particular, the
predictable-order threshold enumeration, reachable-state invariant across the
whole machine and their composition still require full implementation
refinement. The report explicitly separates checked integer inequalities from
that missing end-to-end formal connection. The focused numerical suite has
28 passing tests, including finite Bellman comparisons with actual Oracle fees,
orders and feedback; overflow rejection; and closing/repair regressions.

The saved NSF has unbounded Gaussian log-price tails. A separate diagnostic
restricting competitors to policies with finite log utility under that support
gave 0.0466344% on the finite graph. That restriction excludes held shorts and
almost all long borrowing because positive-probability zero-account liquidation
has utility minus infinity. It is a **different competitor set** from the
bounded-price problem. It is not used to claim the 0.1% target was met. The
earlier bounded-price assumption and the current unbounded model need an
explicit common support definition before such a restriction can be adopted.

What remains before claiming the requested accuracy: preserve or rigorously
bound the forecast-law approximation, obtain a lower expectation bound for the
selected policy close enough to the upper, complete the verifier refinement,
and repeat across required origins/account states. No live-bot default was
changed to an uncertified controller.

## Reproduction

The primary artifacts are under `data/backtests/`:

- `oracle-accuracy-graph-validation/`: graph, checkpoint provenance and construction diagnostics.
- `oracle-accuracy-validated-graph/`: final upper, rules, source snapshots and hashes.
- `oracle-accuracy-value-validation-reachable-mixed/`: reusable per-depth continuation planes.
- `oracle-accuracy-policy-validation-orders-v2/`: actual simulated execution arrays, source snapshots and scores.
- `oracle-accuracy-policy-validation-fine/`: completed 0.01-exposure run and passing execution audit.
- `oracle-planning-paths-validation/`: independent NSF paths.
- `oracle-planning-accuracy-audit/`: graph/direct comparison and explicit unresolved accuracy status.
- `oracle-planning-accuracy-audit-fine/`: final fine-policy comparison and remaining gap.
- `oracle-certified-value-lean-v6.log`: passing full proof build and axiom audit.

```powershell
.venv-ml/Scripts/python.exe ml/certify_oracle_graph.py --graph data/backtests/oracle-accuracy-graph-validation --output <new-output-directory> --points 65 --order-aware --reachable-leverage
.venv-ml/Scripts/python.exe ml/backtest_certified_graph_policy.py --graph data/backtests/oracle-accuracy-graph-validation --value data/backtests/oracle-accuracy-value-validation-reachable-mixed --output <new-policy-directory> --step .01 --limit-fractions .1 .25 .5 1 --limit-offsets 1 10 100
node scripts/theory-lean.mjs check
```
