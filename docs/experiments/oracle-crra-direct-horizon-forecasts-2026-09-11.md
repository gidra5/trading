# Direct-horizon distribution forecasts and economic information

The retrained forecasts contain directional information, but their predicted
entry gains remain below the execution costs in the declared long/cash
comparison. Adding a proper mean-price training loss improves the selected
one-second forecast. Extending the forecast to 30 and 120 seconds does not
make the validation entry gate pass at 5 bps per side.

These are forecast diagnostics. The stateful Oracle fee/horizon backtest is a
separate experiment: diagnostic means below are not compounded strategy returns.

## Training and selection

The input is the same causal production178 basis: 120 observed one-second
log-return lags and 58 latest observed features. Each target is
`log(price[t+H] / price[t])`, checked against the sum of the next H timeline
returns. Reading actual endpoint prices preserves an exact zero-return atom
when a price path returns to its starting price; summing rounded float32
one-second returns can otherwise leave rounding noise.

All target endpoints remain within their chronological split. Normalization,
knots, Gaussian coordinate scale and the unconditional density are fit on train
only. Test outcomes and backtest profits do not enter checkpoint or model
selection.

| Horizon | Train labels | Validation labels | Test labels |
|---|---:|---:|---:|
| 1 second | 998,802 | 145,437 | 151,732 |
| 30 seconds | 998,773 | 145,408 | 151,703 |
| 120 seconds | 998,683 | 145,318 | 151,613 |

Six conditional GLU candidates were trained: H in `{1,30,120}` and auxiliary
weight lambda in `{0,1}`, seed 1337, four width-512 layers, batches of 4,096,
maximum 64 epochs and patience 12. Each horizon also includes a train-fitted
unconditional density baseline. All selected models were conditional GLUs.
The six completed training/evaluation loops took approximately 577 seconds
on the available RTX 3070, excluding input loading and export.

Write `Y = exp(R)-1` and let `m_theta(I)` be the forecast's expected Y. Training
uses

```text
mixed-law NLL + lambda * (m_theta(I) - Y)^2 / E_train[Y^2].
```

Every candidate and checkpoint is selected using the same validation score:
NLL plus normalized mean-price squared error, with coefficient 1. NLL and mean
accuracy are reported separately. The extra loss targets the quantity relevant
to a risk-neutral affine price payoff. It does not reward trade counts or
force the predicted mean across a fee threshold.

For exact integrals, adding this squared-mean loss preserves propriety because
its conditional expected value is
`Var(Y | I) + (m_theta(I) - E[Y | I])^2`. The implementation approximates the
distribution moments by quadrature. It evaluates `expm1` directly so that
sub-basis-point forecasts and their gradients are not erased by subtracting
two float32 numbers near one.

| Horizon | Lambda | Validation NLL | Validation mean-price MSE skill | Test mean-price MSE skill | Selected |
|---|---:|---:|---:|---:|---|
| 1 second | 0 | -4.937322 | 2.4000% | 3.3101% | |
| 1 second | 1 | -4.939230 | 2.9090% | 3.7202% | yes |
| 30 seconds | 0 | -6.215743 | 1.2876% | 2.2516% | yes |
| 30 seconds | 1 | -6.211039 | 1.2937% | 2.3535% | |
| 120 seconds | 0 | -6.014585 | 0.5804% | 0.3871% | yes |
| 120 seconds | 1 | -6.013737 | 0.5793% | 0.1322% | |

MSE skill is relative to predicting zero excess price return. NLLs at different
horizons concern different target laws and zero-atom frequencies; they do not
rank the economic usefulness of horizons. The 30-second auxiliary model has
slightly better mean accuracy but worse validation NLL; the declared combined
score selects lambda 0. The 120-second auxiliary model does not improve mean
accuracy. No further tuning was done to manufacture trades.

## Finite expected wealth

The earlier asinh-logistic coordinates give polynomial tails for **log** return
with positive endpoint density. An integrable log return does not imply finite
`E[exp(R)]`, which gamma 0 needs for ordinary expected-wealth comparisons.

The new family uses Gaussian-CDF coordinates `u = Phi(R / sigma)` and the
same normalized triangular knot-density architecture, plus the exact zero
atom. Its finite piecewise-linear coordinate density h is bounded by some M.
Consequently its continuous return density is bounded by M times a Gaussian
density and

```text
E[exp(t R) | nonzero component] <= M * exp(t^2 sigma^2 / 2)
```

for every real t. The zero atom also has finite exponential moments. The
Gaussian coordinate scale and knots are train-only parameters. This is a
different tail family, so the H=1 candidate is included as a control for the
family change. The new model is selected within the declared candidate family;
this experiment does not claim it is the best possible price model.

## Validation economic gate

For a fresh cash account entering a surviving long and closing fully with
valid orders, unit funding and the same fee f on each side, the expected
gross price ratio must exceed `1/(1-f)^2`. At f=0.0005 this requires an expected
return greater than **10.007505 bps**. Account constraints and an incomplete
close require the actual Oracle action-value calculation instead of applying
this full-close formula blindly.

The gate audits nonoverlapping validation origins with stride H. It requires
finite predicted means and both a strict long winner and a strict cash winner
at the declared fee. The saved `selectedModelGatePassed` is false for all
three selected models.

| Horizon | Validation contexts | Predicted mean minimum | p50 | p95 | p99 | Maximum | Above 10.007505 bps |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 second | 145,437 | -0.26973 | 0.00163 | 0.12413 | 0.20303 | 0.27176 | 0 |
| 30 seconds | 4,847 | -1.06945 | -0.01047 | 0.60523 | 0.81300 | 1.11791 | 0 |
| 120 seconds | 1,211 | -1.11426 | -0.05137 | 0.64346 | 0.96923 | 1.52972 | 0 |

All return columns are basis points. No selected validation forecast exceeds
even the 2.000300 bps full-close threshold at 1 bp per side.

| Horizon | Zero-fee long decisions | Zero-fee cash decisions | Observed return when long was predicted |
|---|---:|---:|---:|
| 1 second | 75,073 | 70,364 | +0.04583 bps |
| 30 seconds | 2,349 | 2,498 | +0.32367 bps |
| 120 seconds | 547 | 664 | +0.43840 bps |

This is empirical validation evidence of a small directional signal. It does
not establish a uniform action-value error bound, calibrated tail probabilities
or profitable trading after fees. A predicted positive margin alone is not the
error-controlled improvement theorem in
[Decision-informative distributions](../theory/Decision-informative%20distributions.md).

Checkpoint selection was persisted before test scoring. The initial running
trainer computed overlapping validation economic diagnostics before test.
The separate nonoverlapping gate audit was added afterward and reads only
validation data; it does not alter model selection. The current reproduction
trainer also persists a nonoverlapping validation gate before loading test.

## Endpoint exports and scope

Exports contain seven nonzero-return branches plus the zero atom. Each active
branch uses the conditional **gross price ratio mean** in an equal-probability
bin, rather than exponentiating a conditional mean log return. Price-ratio
mean conservation relative to the chosen quadrature is within 1.7e-15 in the
exports. A 96-versus-384-point component-moment comparison differs by at most
2.37e-8 in price ratio for the 30/120-second models; this is a numerical
convergence check, not a certified quadrature error bound.

| Horizon | Exported nonoverlapping test decisions | File |
|---|---:|---|
| 1 second | 151,732 | `data/backtests/oracle-crra-horizon-h1.jsonl` |
| 30 seconds | 5,057 | `data/backtests/oracle-crra-horizon-h30.jsonl` |
| 120 seconds | 1,264 | `data/backtests/oracle-crra-horizon-h120.jsonl` |

Each row contains `horizonSeconds`, the observed time/price, a one-level
endpoint price tree, realized endpoint time/price, and `realizedPrices` with
all H actual intermediate one-second marks. Future realized marks are replay
data and never decision inputs. All exports begin at timestamp 1786320001000.

An endpoint law does not determine leveraged liquidation or limit-order fills
inside the horizon. These exports are intended for a declared fixed-hold
long/cash comparison with nonnegative quote and asset balances, unit funding,
and no pending limits. The replay still checks all actual intermediate marks.
Finite-tree optimality is conditional on the exported law and action menu;
quadrature, price rounding, nonlinear capped-close payoffs and receding-horizon
planning need their stated approximation scope.

## Reproduction

Run from the repository root with the existing ML environment. SciPy is
already part of the forecast dependency profile. Use a fresh output directory;
the trainer refuses to overwrite an existing checkpoint.

```powershell
.venv-ml/Scripts/python.exe ml/train_horizon_price_forecast.py --output data/training/runs/stochastic-crra-direct-horizons-gaussian-reproduction
.venv-ml/Scripts/python.exe ml/audit_horizon_price_forecast.py --runs-root data/training/runs/stochastic-crra-direct-horizons-gaussian-reproduction
.venv-ml/Scripts/python.exe ml/export_horizon_price_forecast.py --runs-root data/training/runs/stochastic-crra-direct-horizons-gaussian-reproduction --output-directory data/backtests/horizon-reproduction
.venv-ml/Scripts/python.exe -m unittest discover -s ml -p test_horizon_price_forecast.py -v
```

The completed experiment artifacts are under
`data/training/runs/stochastic-crra-direct-horizons-gaussian-v2`.
Each horizon contains `model-selection.json`, `validation-economic-gate.json`,
and per-candidate plans, training logs, checkpoints and distribution results.
Eight focused ML contract tests pass, including chronological horizon bounds,
exact roundtrip zero labels, finite extreme likelihoods, stable mean gradients,
and gross-mean preservation.
