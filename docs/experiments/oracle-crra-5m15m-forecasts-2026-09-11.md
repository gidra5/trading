# Five- and fifteen-minute distribution forecasts

Neither selected forecast clears the **10.007505 bps** expected-return threshold
for a fresh full-close long with unit funding and 5 bps fees on each side.
The largest predicted mean among nonoverlapping validation decisions is
1.591 bps at five minutes and 2.368 bps at fifteen minutes. Both saved
`selectedModelGatePassed` values are false.

These are distribution and validation diagnostics. The corresponding stateful
Oracle replay compares paid and zero-fee policies separately over a common
151,200 seconds (42 hours); the averages here are not compounded bot returns.

## Fixed training protocol

The existing direct-horizon trainer and model were used without changes.
The input remains 120 past/current one-second return lags plus 58 current
observed features. Targets are actual endpoint log price ratios, checked
against the return timeline; exact equal endpoint prices retain the zero atom.
Origins whose targets cross their chronological split end are discarded.

The four candidates use H in `{300,900}` seconds and mean-loss weight lambda
in `{0,1}`. All use seed 1337, four width-512 GLU layers, batch size 4,096,
maximum 64 epochs and early-stopping patience 12. Each candidate is compared
with a train-fitted unconditional density baseline. Gaussian-CDF coordinates
retain finite exponential price moments. Feature normalization, basis knots,
coordinate scale and mean-loss normalization use training data only.

Training minimizes NLL plus lambda times the squared error of the predicted
excess-price mean, divided by the training excess-price second moment. Every
checkpoint and candidate is selected using the same validation NLL plus
normalized mean-price MSE, with coefficient one. Test scores and bot profits
do not enter selection. No distribution sharpening, trade-count loss or
additional search was used to force threshold crossings.

| Horizon | Train labels | Validation labels | Test labels | Nonoverlapping validation decisions |
|---|---:|---:|---:|---:|
| 5 minutes | 998,503 | 145,138 | 151,433 | 484 |
| 15 minutes | 997,903 | 144,538 | 150,833 | 161 |

Both horizons selected the conditional GLU with lambda 1, at epoch 1. The four
training/evaluation loops took approximately 347 seconds on the RTX 3070,
excluding input loading and export.

| Horizon | Lambda | Validation NLL | Validation mean-price MSE skill | Test mean-price MSE skill | Selected |
|---|---:|---:|---:|---:|---|
| 5 minutes | 0 | -5.609855 | 0.1863% | 0.3707% | |
| 5 minutes | 1 | -5.609574 | 0.4929% | 0.2586% | yes |
| 15 minutes | 0 | -5.073085 | -0.4488% | 0.2217% | |
| 15 minutes | 1 | -5.072136 | -0.0630% | -0.7379% | yes |

Mean-price MSE skill is relative to predicting zero excess return. The
auxiliary loss improves validation mean accuracy at both horizons, but its
selected models have lower test mean accuracy than their lambda-0 counterparts.
The fifteen-minute selected model is worse than the zero-return baseline for
mean-price MSE on both validation and test. The validation-selected checkpoint
was retained rather than choosing a different one after examining test data.

## Economic validation gate

For a surviving valid full-close long from cash, the relevant gross price
ratio threshold is `1/(1-f)^2`; at f=0.0005 it is 1.0010007505003125. This
full-close formula does not replace the account-specific Oracle action-value
calculation when order caps leave a residual position.

| Horizon | Predicted mean minimum | p50 | p95 | p99 | Maximum | Above 10.007505 bps |
|---|---:|---:|---:|---:|---:|---:|
| 5 minutes | -1.4792 | -0.1003 | 0.8180 | 1.1366 | 1.5909 | 0 / 484 |
| 15 minutes | -3.3035 | -0.2825 | 1.8351 | 2.1462 | 2.3680 | 0 / 161 |

All return columns are basis points, evaluated at stride-H validation origins.
Without fees, the five-minute model chooses long in 208 of 484 contexts and
cash in 276; observed return conditional on predicted long is +0.6384 bps.
The fifteen-minute model chooses long in 70 of 161 contexts and cash in 91;
the corresponding observed return is +3.6271 bps.

At 1 bp per side, the fifteen-minute model crosses the 2.000300 bps threshold
in 4 of 161 validation contexts; their observed mean net return is +2.9003 bps.
The five-minute model has no such validation crossing. Neither model crosses
the 4.001200 bps threshold at 2 bps per side on the nonoverlapping validation
schedule. Four observations do not establish reliable calibration or future
profitability, and no uniform action-value error bound has been established.

Across **all overlapping test origins**, the selected predicted-mean maxima
are 2.2144 bps at five minutes and 3.8429 bps at fifteen minutes. Neither
selected test forecast crosses the 10.007505 bps threshold there either.

## Exports and verification

The output directory is `data/backtests/oracle-horizons-5m15m`:

| File | Exported decisions | Decisions for common 42-hour replay |
|---|---:|---:|
| `oracle-crra-horizon-h300.jsonl` | 505 | 504 |
| `oracle-crra-horizon-h900.jsonl` | 168 | 168 |

Both files start at timestamp 1786320001000. The common replay ends at
1786471201000. Every row includes all H actual intermediate one-second marks
for replay, while the forecast tree contains only an endpoint law. Future
realized marks are never forecast inputs.

Each endpoint tree has seven nonzero-return bins plus the zero atom, using
conditional gross price-ratio means. Numerical price-mean conservation errors
are at most 1.12e-15 and 1.34e-15 respectively. The 96-versus-384-point component
moment differences are at most 3.53e-8 and 6.16e-8 in price ratio; these are
numerical convergence checks, not certified integration-error bounds.

Training and export completed their finite-input, split-boundary, endpoint
alignment, zero-atom and probability/mean checks. No shared ML core was edited.
Endpoint predictions remain scoped to the declared long/cash holding problem
with nonnegative cash and inventory, unit funding and no pending limits.
They do not supply the path law needed for leveraged liquidation or limit
fills inside a horizon.

## Reproduction

Run from the repository root. The trainer refuses to overwrite an existing
checkpoint, so use a fresh output directory for another run.

```powershell
.venv-ml/Scripts/python.exe ml/train_horizon_price_forecast.py --horizons 300 900 --mean-loss-weights 0 1 --output data/training/runs/stochastic-crra-direct-horizons-5m15m-reproduction
.venv-ml/Scripts/python.exe ml/audit_horizon_price_forecast.py --runs-root data/training/runs/stochastic-crra-direct-horizons-5m15m-reproduction --horizons 300 900
.venv-ml/Scripts/python.exe ml/export_horizon_price_forecast.py --runs-root data/training/runs/stochastic-crra-direct-horizons-5m15m-reproduction --horizons 300 900 --output-directory data/backtests/oracle-horizons-5m15m-reproduction
```

Completed plans, checkpoints, training logs, model selections and validation
gates are under
`data/training/runs/stochastic-crra-direct-horizons-5m15m-v1`.
See the [earlier direct-horizon methods note](oracle-crra-direct-horizon-forecasts-2026-09-11.md)
for the Gaussian-coordinate moment argument and the proper auxiliary loss.
