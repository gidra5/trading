# Searching for a forecast horizon above the fee threshold

The first crossing in the validation-selected ridge search is **two hours**.
Its maximum forecast is **12.7158 bps (0.127158%)** on nonoverlapping validation
origins and **17.3467 bps (0.173467%)** on nonoverlapping test origins, exceeding
the configured 10.007505-bps break-even diagnostic. The signal does not hold up:
the resulting Oracle account replay loses **1.579856% after fees** over 42 hours.
This is a forecast crossing, not evidence of a profitable conditional expectation.

## What the older 0.3% candle statistic measured

The saved data support movement sizes around the recalled number:

- In [the five-year distribution study](log-return-distributions-2026-08-12.md),
  175,296 fifteen-minute observations from July 2021 through July 2026 have
  log-return standard deviation **0.283186%** and mean absolute log return
  **0.177577%**. Their unconditional signed mean is only **0.0003578%**.
- Recomputing the actual OHLC candles saved in
  [the interval charts](../interval-price-movement.html) gives mean fifteen-minute
  high-low/open ranges **0.284388%** for November 29–December 5, 2023 and
  **0.326524%** for March 7–9, 2023. Their mean absolute candle bodies are
  **0.148815%** and **0.143195%**, respectively. These are hand-selected historical
  periods, not a representative pooled sample.
- The UI statistic described in [algorithms](../algorithms.md) averages
  `(high-low)/open` over its recent synthetic candles. No archived numeric UI
  snapshot was found that establishes which statistic the recollection refers to.

In the current test split, actual complete nonoverlapping intervals give:

| Horizon | Intervals | Mean absolute endpoint move | Median absolute endpoint move |
| --- | ---: | ---: | ---: |
| 15 minutes | 168 | 0.088894% | 0.067450% |
| 30 minutes | 84 | 0.124152% | 0.085705% |
| 1 hour | 42 | 0.195040% | 0.157740% |
| 2 hours | 21 | 0.295083% | 0.2442% |

Thus the average absolute test move passes 0.1% at **30 minutes**, and the median
passes it at **one hour**. Even fifteen-minute mean absolute endpoint movement
passes 0.1% in the current training and validation splits: 0.101824% and 0.116130%.
Moves of sufficient size already exist. The unresolved requirement is predicting
their direction and magnitude well enough before entering.

For the surviving, fully closable, cash-starting long with unit funding and fee
`f = 0.0005` at both ends, positive expected gain requires

\[
\mathbb E[P_{t+H}/P_t-1\mid I_t]>(1-f)^{-2}-1
=0.0010007505003.
\]

This conditional signed expectation is different from expected absolute
movement, standard deviation, or a candle's high-low range. For example, if the
only outcomes were exactly +0.3% and -0.3%, the probability of the positive
outcome would need to exceed approximately 66.68% to pass this long-entry
threshold. A large range alone supplies no such probability estimate.

An independent fifteen-minute numerical audit integrates the selected density
directly in physical Gaussian log-return coordinates. It obtains 2.3676934818
bps versus 2.3676934816 bps from the knot calculation. The exported mean agrees
to 0.00000033 bps before price rounding. There is no factor-of-ten or percent/bps
error in the investigated example. These are floating-point integration checks,
not a Lean proof of numerical integration error.

## Forecast search

The new neural runs cover 30 minutes, 1, 2 and 4 hours, each with auxiliary
mean-loss weights 0 and 1 and a train-fitted marginal control. The fixed selection
score is validation NLL plus train-normalized excess-price mean MSE. Selected
maximum validation means are 3.3135, -0.05178, 0.02063 and 0.40494 bps.
The last three are constant epoch-zero priors; subsequent fitting did not improve
the declared score. None crosses the fee threshold. Full details and commands are
in [the neural search report](oracle-crra-horizon-search-forecasts-2026-09-11.md).
The planned 8-, 12- and 24-hour neural runs were not run after the ridge control
found a crossing; the saved plan explicitly records this amendment.

The independent CPU ridge control targets the price mean relevant to gamma zero.
It uses the same 178 causal features: 120 current/past one-second return lags and
58 observed features. Train-only normalization, conditional Gaussian active
log-return regression, train-only residual variance and a fixed train-only zero
atom define a proper distribution with finite price moments. Its expectation is
`(1-p0) * (exp(mu + residualVariance/2) - 1)`.

Each horizon compares seven ridge strengths with zero-price-mean and train-drift
controls. The candidate minimizes **validation simple-return MSE**, a different
selection criterion from the neural composite. The first ascending selected
horizon with a positive nonoverlapping validation crossing is saved before test
evaluation. All horizons share 912,403 complete training origins. Overlapping
labels do not create that many independent observations.

| Horizon | Validation-selected candidate | Largest validation forecast | Crossings / nonoverlapping origins |
| --- | --- | ---: | ---: |
| 30 minutes | Zero price mean | 0 bps | 0 / 80 |
| 1 hour | Zero price mean | 0 bps | 0 / 40 |
| 2 hours | Ridge, lambda 0.0001 | 12.7158 bps | 1 / 20 |
| 4 hours | Ridge, lambda 0.01 | 18.8520 bps | 2 / 10 |
| 8 hours | Ridge, lambda 0.001 | 23.8156 bps | 3 / 5 |
| 12 hours | Ridge, lambda 0.0001 | 27.2203 bps | 1 / 3 |
| 24 hours | Ridge, lambda 0.0001 | 15.6621 bps | 1 / 1 |

Only the first selected two-hour ridge candidate and its fixed controls are
evaluated on test. Test PnL does not select a horizon or model. Validation is
reused for regularization and horizon screening; these are exploratory results.

For the selected two-hour ridge, validation MSE improves over zero prediction by
3.2952%, but test MSE is **17.8189% worse**. Test prediction/return correlation is
**-0.2695**. Test mixed NLL is also worse than the corresponding zero-mean law:
-4.07773 versus -4.15048. The one validation crossing has realized cash-starting
full-close net return -54.588 bps. The three test crossings average -53.915 bps
net. These isolated entry diagnostics do not replace an account replay.

## Exact account replay and verification

The frozen two-hour ridge is exported without refitting or reselection. Seven
equal-probability active Gaussian bins use analytic conditional mean prices,
plus the zero atom. The maximum mean-conservation error is 2.22e-16 in gross
price-ratio units before price-grid rounding. An independent quadrature review
also verifies the conditional-bin formula.

All runs below use exactly the same **151,200 one-second marks**, from
**2026-08-10 00:00:01 UTC through 2026-08-11 18:00:01 UTC**, and 21 two-hour decisions:

| Forecast | Fees | Terminal wealth from 10,000 | Return | Market trades | Maximum drawdown |
| --- | --- | ---: | ---: | ---: | ---: |
| Selected ridge | 5 bps per side | 9,842.0144 | -1.579856% | 4 | 1.924792% |
| Selected ridge | Zero | 9,756.2270 | -2.437730% | 6 | 2.867173% |
| Selected neural prior | 5 bps per side | 10,000 | 0% | 0 | 0% |

Paid buy-and-hold returns -2.076869%; cash returns zero. The paid ridge beats
buy-and-hold during this declining market but loses to cash. Removing fees changes
the optimal actions under the supplied forecast: it takes more trades and loses
more on this realized path. This does not imply that adding fees to a fixed action
sequence improves its result.

Three test origins exceed the cash-start threshold, while the paid account holds
inventory over seven intervals. Existing inventory changes the cost of the next
action. The replay executes net adjustments and does not force a close and reopen
at every horizon boundary. All three runs finish without residual inventory.

The account uses gamma zero, nonnegative cash and asset balances, entry and
maintenance bounds `[0,1]`, unit funding, no pending orders, exposure targets
`0,0.5,1` plus hold, quote trade bounds `[5,10000]`, and a 0.01 price grid. Every
actual intermediate mark is applied. Hypothetical and final closes respect the
cap and retain marked residuals. The underlying tape is BTCUSDT **spot**, with
synthetic Oracle parameters; this is not an actual USD-M funding/order-book replay.

The existing endpoint-hold theorem applies to these restricted holdings for any
number of intermediate marks. It does not justify using endpoint laws alone for
unrestricted leveraged liquidation or limit-order fills. Conditional mean bins
preserve the price mean, but capped-close payoffs can be nonlinear inside bins;
no exact optimum for the underlying continuous distribution is inferred.

The report assembler checks declared validation minima and first crossing,
actual model/selection/forecast hashes, the full shared realized tape, interval
alignment and certificate counts. All checks pass. The backtester now accepts
`--certificate-count`; all **63 decision certificates** across these three runs
are checked, including every entry and exit. The existing **108 execution
fixtures** are checked separately. The Lean checker verifies exact comparisons
of the supplied finite scenario/menu payoffs; it does not certify forecast
calibration or global optimality of rolling horizon decisions. The three endpoint
replay tests and the independent ridge-versus-augmented-least-squares test pass.

## Artifacts and reproduction

Training/selection records and saved coefficient arrays are under
`data/training/runs/stochastic-crra-ridge-horizon-search-v1`. The distribution
exports, fixed replay plan, three account runs, combined `comparison.json`,
`decisions.jsonl` and `lean-verification.json` are under
`data/backtests/oracle-horizon-search`. The realized movement audit is
`data/backtests/stochastic-crra/horizon-movement-range-audit.json`.

Reproduce the ridge experiment into a fresh run directory to preserve its original
selection records:

```powershell
node scripts/run-ml-python.mjs ml/audit_ridge_horizon_forecasts.py --output data/training/runs/stochastic-crra-ridge-horizon-reproduction
```

Re-export and replay the existing frozen selection:

```powershell
node scripts/run-ml-python.mjs ml/export_ridge_horizon_forecast.py
npm run backtest:oracle-endpoint -- --forecasts data/backtests/oracle-horizon-search/ridge-h7200.jsonl --count 21 --certificate-count 21 --fee-scale 1 --output data/backtests/oracle-horizon-search/ridge-h7200-fees1
npm run backtest:oracle-endpoint -- --forecasts data/backtests/oracle-horizon-search/ridge-h7200.jsonl --count 21 --certificate-count 21 --fee-scale 0 --output data/backtests/oracle-horizon-search/ridge-h7200-fees0
npm run backtest:oracle-endpoint -- --forecasts data/backtests/oracle-horizon-search/oracle-crra-horizon-h7200.jsonl --count 21 --certificate-count 21 --fee-scale 1 --output data/backtests/oracle-horizon-search/neural-h7200-fees1
node scripts/report-oracle-ridge-horizon.mjs
node scripts/verify-oracle-crra.mjs --certificates data/backtests/oracle-horizon-search/decisions.jsonl --output data/backtests/oracle-horizon-search/lean-verification.json
node scripts/run-ml-python.mjs ml/audit_horizon_movement_scales.py
node scripts/run-ml-python.mjs ml/audit_horizon_price_mean.py
```

The new model remains an experiment. Shared forecasting/execution rules and the
running provider were not changed; the backtester change controls certificate
coverage only.
