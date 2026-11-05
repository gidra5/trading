# Continuous actions against the full saved forecast law

The saved output-only and joint-calibration forecasts were replayed using their full signed-Gamma log-return distributions and continuous valid market trade sizes. The predictors and calibrators were not retrained or selected again. This replaces the three-conditional-mean, target-menu controller for this comparison.

With fees, the 15-minute and 30-minute policies still remain in cash. At two hours both variants return -1.042942%, versus -1.042219% with the former controller. Continuous optimization does not establish improved realized returns: it optimizes the supplied forecast and terminal objective.

## Matched results

Same 42-hour tape, initial wealth 10,000, gamma zero, 0.05% received-amount fees per side in paid runs, quote-order minimum 5 and aggregate maximum 10,000 per decision, price step 0.01. All 151,200 actual one-second marks are processed per run. Decisions remain nonoverlapping at each forecast horizon.

| Horizon | Output-only, fees | Joint, fees | Trades, each | Output-only, zero fees | Joint, zero fees |
| --- | ---: | ---: | ---: | ---: | ---: |
| 15 minutes | 0.000000% | 0.000000% | 0 | -0.587085% | -0.587085% |
| 30 minutes | 0.000000% | 0.000000% | 0 | -0.114951% | -0.114951% |
| 2 hours | -1.042942% | -1.042942% | 3 | -2.594118% | -2.594118% |

The former output-only zero-fee 15-minute result was -0.656531%; the other zero-fee results are unchanged to reported precision. Distribution integration and action search changed together, so this is not an attribution of that difference to either change alone.

The paid two-hour policy includes a minimum-sized buy and a later sale leaving a small residual. Its final residual is approximately 0.00007555348 asset units, valued at the terminal mark. Counting fractional exposure at every decision also counts ordinary holding drift and residuals; it does not count only deliberately selected interior trade sizes.

## Objective and action domain

The action is the actual signed quote change d. Negative buys, positive sells, and d=0 is the identity action. A separate hold label is unnecessary. With b=1-f, current account (q,a), and current price p, feasible trades in this experiment are

```
{0}
union [-min(q,dq_max), -dq_min]                   when this interval is nonempty
union [dq_min, min(b*a*p,dq_max)]                 when this interval is nonempty.
```

Post-trade balances use Oracle's received-amount rules:

```
buy:  q' = q+d,  a' = a-b*d/p
sell: q' = q+d,  a' = a-d/(b*p).
```

For each possible endpoint price P, the terminal procedure sells net quote

```
D(a',P) = 0                         if b*a'*P < dq_min
          min(b*a'*P, dq_max)       otherwise.
W(P) = q' + a'*P - (f/b)*D(a',P).
```

The solver maximizes E[W(P)]. This is equivalent to maximizing gamma-zero CRRA E[W(P)-1]. The terminal close is one valid non-overshooting capped order; any residual is marked. It is a specified terminal contract, not a theorem that forced closing at each planning horizon is globally optimal.

Even with gamma zero, the order minimum and cap make the terminal cost nonlinear in price. Replacing all magnitudes by conditional means can therefore change the objective. Full conditional distributions are retained here.

## Full-law integration and continuous search

`ml/continuous_oracle_endpoint.py` reads all six parameters of the saved zero/sign/Gamma law. It integrates the probability of every central allowed price tick via CDF differences, including the exact unchanged-price atom. It does not use three representative prices or a magnitude-quantile scenario menu.

The modeled price is projected to `max(1, round(P/dp))*dp`, the positive price lattice. The first positive tick receives the otherwise sub-tick lower tail. This projection is explicit because an unbounded negative log-return distribution assigns positive mass below half a price tick.

The central range covers all but at most 1e-11 of each conditional magnitude tail. The remaining tails are not discarded or renormalized away. Their exact analytic mass and unrounded first-price moments bound gross wealth, allowing at most one price tick of rounding error per tail observation. Monotone terminal costs bound each tail's contribution, including the close cap. Central cumulative probability and first-moment arrays permit fast evaluations at arbitrary trade sizes.

The global search covers both continuous trade intervals and the isolated zero action. On either interval, post-trade gross expected wealth is affine, and expected terminal closing cost is nondecreasing in remaining asset quantity. Thus

```
upper bound on a cell = max(gross wealth at its endpoints)
                     - lower closing-cost bound at its smallest asset quantity.
```

The solver subdivides cells until no cell can exceed the best achieved lower value by more than 0.0001 quote units. It fails explicitly if its integration budget, evaluation budget, or floating-point resolution prevents this tolerance. This works with the minimum-order discontinuity and does not assume differentiability or concavity. A supremum near a dust boundary need not be attained, which is another reason to report an epsilon-optimal result rather than claim an exact argmax.

Across all 1,092 decisions the largest reported gap was approximately 0.00007276 quote units. These are numerical bounds with an explicit floating-point cushion, not directed-rounding interval certificates. The same-state former target-menu candidates were included as seeds and did not beat the chosen value beyond the numerical tolerance.

## Verification and limits

Eight tests cover price-cell probabilities and the zero atom, total mass and first moments, an independent per-outcome terminal-cost calculation, continuous search versus an independent dense sweep, a fractional dust-boundary optimum missed by the old targets, zero-action identity, finite-tail refinement, and rejection of leveraged accounts or infinite price expectations.

`scripts/verify-continuous-oracle.ts` independently replayed all 1,092 selected actions and 1,814,400 intermediate marks using the existing exact-rational Oracle execution engine. It checked feasibility, received-amount fees, aggregate size limits, grid prices, survival, final close, residuals and reported wealth/drawdown. All 12 runs passed; maximum observed numerical-versus-rational state discrepancy was about 7.28e-12.

`TradingTheory.ContinuousEndpointBounds` formalizes monotonicity of the capped/dust closing cost, its equity identity under Oracle execution, the cell upper-bound inequality, and epsilon-optimality conditional on an exhaustive valid cover. These supporting real-arithmetic lemmas do not certify Python floating-point operations, Gamma CDFs or the generated search cover. The former finite-menu Lean certificates must not be relabeled as continuous-action certificates.

All eight new numerical tests and 22 existing execution/endpoint tests passed. `npm run theory:check` built the complete theory library and passed its axiom audit, including all six new supporting lemmas.

This remains an unleveraged long/cash endpoint MPC experiment: q,a>=0, unit funding, no active orders, entry and maintenance bands [0,1]. Existing `OracleEndpointHold` proofs justify endpoint sufficiency under these assumptions. The live finite-tree bot has not been replaced by this experiment.

A leveraged position can liquidate during a path that later recovers to the same endpoint. Therefore a multi-tick endpoint marginal alone cannot determine its expected wealth under Oracle liquidation. A valid leveraged extension needs a forecast of the intervening path or a proven sufficient joint statistic, such as the relevant extrema jointly with terminal price under a fixed holding rule. Multiple horizon marginals alone do not supply that joint law. Optimizing future rebalances or limit orders additionally needs observable conditional continuations. This experiment does not invent such a law or use the realized future path as a forecast.

## Reproduction

Artifacts: `data/backtests/oracle-continuous-full-law/{plan,comparison,execution-verification}.json`, per-run decisions and reports, and `lean-check.log`. Original forecasts and historical reports remain unchanged.

```powershell
.venv-ml/Scripts/python.exe ml/backtest_continuous_oracle.py --output data/backtests/oracle-continuous-full-law-recheck
node --conditions=development --import tsx scripts/verify-continuous-oracle.ts data/backtests/oracle-continuous-full-law-recheck
$env:PYTHONPATH='ml'
.venv-ml/Scripts/python.exe -m unittest ml/test_continuous_oracle_endpoint.py
npm run theory:check
```
