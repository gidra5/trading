# Refitted ensemble: full-law H1 Oracle backtest

All six fee-paying policies stayed in cash: **0 trades, 0% return, 0%
drawdown**. Removing fees exposes useful forecast information: direct59
returned **+5.786%**, while the expanded mixture returned **+5.605%**. The
mixture's better forecast MSE did not make it the best zero-fee trading policy
on this interval. This comparison supplies no fee-paying trading improvement.

## Results

Test interval: **2026-08-10 00:00:01 through 2026-08-11 16:08:54 UTC**, about
40.15 hours. Initial wealth is 10,000 quote in every run. The zero-fee rows are
separately optimized policies, not fees subtracted after a fixed trade replay.

| Predictor | Return with 0.05% per side | Zero-fee return | Zero-fee max drawdown | Zero-fee executed orders |
| --- | ---: | ---: | ---: | ---: |
| Original joint knot | 0.000% | +5.184% | 0.098% | 40,951 |
| Current calibrated specialists | 0.000% | +5.050% | 0.085% | 34,647 |
| Previous equal pair | 0.000% | +5.466% | 0.097% | 38,853 |
| Fresh direct59 | 0.000% | **+5.786%** | **0.076%** | 38,375 |
| Fresh base17 | 0.000% | +5.376% | 0.114% | 33,664 |
| Expanded mixture | 0.000% | +5.605% | 0.084% | 37,307 |

Buy-and-hold returned **-2.102% with fees** and **-2.004% without fees**, with
maximum drawdown 2.967%. Cash returned zero. Buy-and-hold enters once at the
first issue, holds the entire interval and attempts one capped close at the
end; any residual is marked at the final price.

The expanded zero-fee mixture ended at **10,560.530834 quote-equivalent**, with
quote turnover **372,000,552.79** and average post-entry exposure **50.086%**.
Its terminal residual was 0.0000662373 asset units; equity includes this
residual. Dust and the absolute order cap account for many fractional exposure
records, so a nonzero-position count must not be interpreted as a fresh trade
count. The direct59 control ended at 10,578.567662 quote-equivalent.

The expanded mixture exceeded the previous equal pair by 0.1388 return
percentage points, but trailed direct59 by 0.1804 points, without fees. These
are descriptive results from one previously inspected interval, not a new
selection or statistical significance claim. The many zero-fee trades and
all-cash paid controls show why positive gross returns here do not establish
that the one-second signal can cover roundtrip execution costs.

## Execution contract

This compares the freshly trained direct59 and base17 distribution heads,
the validation-selected expanded mixture, the original joint knot model,
the current calibrated specialists, and their equal-weight pair. All use
the same 36,134 archived test origins. The forecasts are **one second ahead**
and were issued every **four seconds**. No missing forecasts or longer-horizon
laws are synthesized.

At each forecast issue the controller optimizes a continuous market quote
flow using the complete predicted price law. At the next second it attempts
one valid, capped close. It then retains cash and any uncloseable residual
until the next forecast issue, processing every intervening observed price.
The final tick allows that same single close, without an extra same-tick
order. Zero quote flow already means no trade; there is no separate hold action.

Parameters are initial wealth 10,000 quote, CRRA gamma zero, exposure and
maintenance bands [0, 1], unit signed-asset funding multiplier, no limit
orders, received-amount taker fee 0.05% per side, minimum actual quote order
5, maximum aggregate quote order 10,000 per decision, and price step 0.01.
Each predictor also has a zero-fee diagnostic with all other rules identical.
These are synthetic Oracle executions on the existing BTC spot tape, not a
simulation of leveraged Binance perpetual funding or limit-order fills.

This is a **one-step roundtrip endpoint policy**, with a numerical absolute
expected-wealth gap tolerance of 0.0001 quote per decision. It is not a full
multiperiod Bellman optimum. The previous 15m/30m/2h experiment used different
forecast and closing contracts, so its returns are not like-for-like controls.

## Frozen forecasts

The expanded law uses 75.6531% direct59, 18.2735% current specialists, and
6.0735% base17, selected/refitted on validation before this backtest. All six
candidates are reported; backtest returns do not change the weights or select
a deployment model. The historical test interval has been inspected in prior
experiments and is not a new untouched holdout.

The fresh heads and old joint model retain their knot densities and exact
zero-return atoms. The current specialist stream retains its archived causal
online input/output calibration and its conditional signed-Gamma log-return
law. New heads receive no additional online calibration in this comparison.
For gamma zero the relevant gross-payoff moment is E[price], not exp(E[log
price]). The terminal closing fee also depends on the distribution through
minimum-size and maximum-size thresholds.

## Integration and implementation

`ml/knot_oracle_price_law.py` integrates the complete densities into nearest
positive price ticks. It retains both tails and bounds their first moments.
The knot integration range accounts for the learned density envelope: a fixed
eight-normal-sigma range is insufficient when a tiny endpoint knot carries
material probability. No tail is silently dropped or renormalized away.

For specialist laws, analytic Gamma truncated probabilities and price moments
evaluate closing-cost bounds at the actual dust and cap thresholds. Central
price ticks and analytic tail moments determine expected price, with remaining
rounding error retained. Mixture expectations and cost bounds are weighted
sums of the component calculations. This keeps the full laws without merging
large price grids. The accelerated 100-decision preview reproduced every
variant's terminal wealth and trade count from the dense preview.

The continuous branch-and-bound solver considers zero and every feasible buy
or sell amount, including the order-minimum gap. Its bound uses affine expected
gross equity and monotone terminal closing cost. It does not reduce the action
space to a fixed exposure menu.

## Verification

All **13 focused Python tests** pass, covering existing continuous endpoint
bounds and the new knot CDF, atom, mixture expectations, tail refinement,
adversarial tiny endpoint knot, and analytic-Gamma versus dense-price-grid
bounds around dust/cap thresholds.

The independent TypeScript checker passed all **12 runs**, comprising
**433,608 forecast decisions** and **1,734,396 price advances**. It checks
source and decision-stream hashes, exact rational Oracle action feasibility,
received-amount fees, aggregate order caps, terminal closes and residuals,
account linkage, every intervening mark, final equity and drawdown. Maximum
observed local quote discrepancy was **3.638e-12**; asset discrepancy was
**5.551e-17**.

At each forecast issue the checker resets rational quote/asset values to the
recorded floating-point account after checking continuity with the previous
exact transition. This avoids unbounded rational denominator growth. It is
**local exact execution refinement**, not a wholly rational compounded replay.

The maximum reported expected-wealth optimality gap across all decisions was
**0.000006140 quote**, below the requested 0.0001 tolerance. The checker verifies
the recorded gap arithmetic and tolerance; it does not independently certify
the integration or global optimizer. The existing Lean endpoint-bound lemmas
justify the real-arithmetic bound structure given their hypotheses. SciPy
CDFs, numerical roundoff, integration bounds and this Python implementation
have **no new end-to-end Lean certificate**. Neither execution agreement nor
these numerical gaps establish full multiperiod stochastic optimality.

## Artifacts and reproduction

- `data/backtests/oracle-refitted-h1-backtest/comparison.json`: results,
  contract, frozen source hashes and benchmark returns.
- `data/backtests/oracle-refitted-h1-backtest/tape.json`: matched origins and
  complete intervening price tape.
- `data/backtests/oracle-refitted-h1-backtest/execution-verification.json`:
  completed independent Oracle checks and verifier identity.
- Each variant/fee directory: `report.json` and hashed `decisions.jsonl`,
  including input account, chosen trade, next-tick close and numerical bounds.
- `ml/backtest_refitted_ensemble.py`: replay and comparison runner.
- `ml/knot_oracle_price_law.py`: full-law integration and mixture adapter.
- `ml/test_knot_oracle_price_law.py`: integration regression tests.
- `scripts/verify-refitted-oracle-backtest.ts`: independent exact Oracle
  execution check against every recorded decision and intervening price.
- [Forecast training and selection](oracle-crra-expanded-mixture-refits-2026-09-12.md).

From the repository root, choose a fresh output directory to reproduce:

```powershell
$env:PYTHONPATH='ml'
.venv-ml/Scripts/python.exe ml/backtest_refitted_ensemble.py --output data/backtests/oracle-refitted-h1-rerun
node --conditions=development --import tsx scripts/verify-refitted-oracle-backtest.ts data/backtests/oracle-refitted-h1-rerun
```

The runner refuses to overwrite a completed comparison. No live bot settings
or model pointers are changed by this experiment.
