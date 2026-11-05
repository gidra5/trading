# Neural horizon search through four hours

The completed neural search did not find a validation-selected forecast with
positive expected return above **10.007505 bps**, the declared full-close
break-even threshold at 5 bps per side and unit funding. Its largest selected
validation mean was 3.3135 bps at 30 minutes.

The original ascending grid was 30 minutes, 1, 2, 4, 8, 12 and 24 hours.
The runs through four hours were completed. The remaining 8-, 12- and 24-hour
neural runs were **not run**: the coordinating experiment found a positive
validation crossing in a separate two-hour ridge control and requested
stopping this neural search after the current four-hour pair. This is not a
claim that the full neural grid was exhausted or that it found a crossing.

## Protocol and results

The unchanged direct-horizon trainer uses causal production178 inputs, actual
endpoint log-return labels, exact endpoint-zero atoms, train-only Gaussian-CDF
knots and feature normalization, and strictly within-split targets. Every
horizon trains lambda 0 and 1 versions of the proper mean-price auxiliary loss
and compares each with a train-fitted unconditional density. Seed, optimizer,
four width-512 layers, 64-epoch limit and patience 12 are unchanged.

Checkpoint/model selection uses only the declared validation NLL plus
train-normalized mean-price MSE. The planned horizon screen uses the first
selected model with any **positive** validation mean above the fee threshold;
negative extremes are reported separately. Neither test results nor backtest
profits select a model or a horizon.

| Horizon | Selected epoch | Selected lambda | All validation origins | Stride-H validation origins | Selected mean range, bps | Positive crossings |
|---|---:|---:|---:|---:|---:|---:|
| 30 minutes | 1 | 0 | 143,638 | 80 | -3.1684 to +3.3135 | 0 |
| 1 hour | 0 | 0 | 141,838 | 40 | -0.05178, constant | 0 |
| 2 hours | 0 | 0 | 138,238 | 20 | +0.02063, constant | 0 |
| 4 hours | 0 | 0 | 131,038 | 10 | +0.40494, constant | 0 |

The 1-, 2- and 4-hour selections are the epoch-zero GLU prior: their forecast
does not depend on the observed feature inputs. Lambda 0 wins the identical
epoch-zero comparison by deterministic tie order. Later trained checkpoints
did not improve the declared validation score. They were not promoted merely
because they made larger predictions.

| Horizon | Validation NLL | Validation mean-price MSE skill | Test mean-price MSE skill |
|---|---:|---:|---:|
| 30 minutes | -4.716403 | -0.6499% | +0.1422% |
| 1 hour | -4.386704 | -0.0278% | +0.1048% |
| 2 hours | -3.929662 | +0.0109% | -0.0328% |
| 4 hours | -3.317336 | +0.2007% | -0.4965% |

MSE skill is relative to predicting zero excess price return. At 30 minutes,
the stride-H validation maximum is 1.97795 bps; the all-overlapping test
maximum is 3.22210 bps. The longer selected priors have the same constant mean
on validation and test. There are no selected validation fee-threshold
crossings at which to estimate conditional realized net return; that field is
saved as null, rather than presenting a return from nonexistent events.

The original million-scale label count should not be read as a million
independent long-horizon observations. The fixed training span supplies about
277 nonoverlapping one-hour targets, 138 two-hour targets and 69 four-hour
targets. Even those can be serially dependent. The validation counts above
show the corresponding limitation for economic screening.

## Scope and artifacts

The search reuses the same validation split for checkpoint/model selection and
horizon screening. A later crossing in any model family is research discovery,
not independent confirmation of a true expected-return edge. These neural
results provide neither a uniform action-value error bound nor evidence that
a profitable forecast cannot exist at another horizon or with another model.

Plans, checkpoints and selected validation audits are under
`data/training/runs/stochastic-crra-horizon-search-v1`. Its `search-plan.json`
preserves the original grid and records the stopping amendment.
`search-results.json` explicitly records:

```json
{
  "completedHorizonsSeconds": [1800, 3600, 7200, 14400],
  "firstSelectedValidationCrossingSeconds": null,
  "allPrespecifiedHorizonsComplete": false,
  "executionStoppedByAmendment": true,
  "remainingPlannedHorizonsNotRun": [28800, 43200, 86400]
}
```

Full endpoint exports are available under
`data/backtests/oracle-horizon-search` as
`oracle-crra-horizon-h1800.jsonl`, `h3600.jsonl`, `h7200.jsonl` and `h14400.jsonl`
(all filenames retain the `oracle-crra-horizon-` prefix). They contain 84, 42,
21 and 10 decisions respectively, with all actual intermediate one-second
marks. The first three cover 42 hours; the four-hour export covers 40 hours.

The exported laws use seven active gross-price-mean bins plus the zero atom.
They support the declared long/cash holding comparison with nonnegative cash
and inventory, unit funding and no pending limits. Endpoint laws do not
determine intermediate leveraged liquidation or limit fills. Training/export
alignment and finiteness checks passed. No shared model or trainer core was
modified for this search; the new `ml/summarize_horizon_search.py` only reads
completed selections/results and writes the combined diagnostic artifact.

## Reproduction

For each completed horizon, use a fresh output root and retain the ascending
order. Inspect the selected validation result before starting another horizon.

```powershell
.venv-ml/Scripts/python.exe ml/train_horizon_price_forecast.py --horizons 1800 --mean-loss-weights 0 1 --output data/training/runs/stochastic-crra-horizon-search-reproduction
.venv-ml/Scripts/python.exe ml/audit_horizon_price_forecast.py --runs-root data/training/runs/stochastic-crra-horizon-search-reproduction --horizons 1800
.venv-ml/Scripts/python.exe ml/export_horizon_price_forecast.py --runs-root data/training/runs/stochastic-crra-horizon-search-reproduction --horizons 1800 --output-directory data/backtests/oracle-horizon-search-reproduction
```

Repeat the commands with 3600, 7200 and 14400 for the completed scope. The
existing summary can be reproduced without retraining:

```powershell
.venv-ml/Scripts/python.exe ml/summarize_horizon_search.py --runs-root data/training/runs/stochastic-crra-horizon-search-v1
```
