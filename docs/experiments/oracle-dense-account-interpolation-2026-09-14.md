# Account interpolation versus exact-state Bellman enumeration

`ml/audit_oracle_dense_interpolation.py` enumerates a two-step finite price tree and all actions in a declared catalog, carrying actual cash, inventory and pending orders through both transitions. Future decisions are selected after the first observation and before the second; they cannot select a different action for each unseen second-step outcome. Terminal cash requires a valid executed close after committed fills. This reference uses the same tested Oracle execution primitives, but does not interpolate account values.

The comparison isolates account approximation when both methods use the same action catalog. It is not a proof about continuous actions or the learned NSF law. An optional shared, denser reference catalog additionally tests omitted exposure actions against a common benchmark. Prices, order-price bins and order-size bins remain finite in both modes.

The diagnostic includes four designed paths (quiet, rising, reversal and asymmetric), plus 100 seeded random two-step laws. Each conditional row has two equally weighted outcomes. Random first-step and conditional second-step log moves have standard deviation 0.004, rounded to the 0.01 price tick. These are synthetic stress cases, not empirical forecast accuracy measurements. Initial equity is $10,000 with no exposure or orders; at most one limit order is active. Fees, reserve leverage, caps and maintenance use `Rules()`.

## Same-catalog comparison after terminal-backup correction

Results are in `data/backtests/oracle-dense-interpolation-terminal-report/results.json`.

| Exposure axis | Cases | Chosen action has no feasible continuation in exact catalog | Cases with positive finite decision gap | Mean finite CE gap | Maximum finite CE gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7 points: -2.5, -2, -1, 0, 1, 2, 2.5 | 104 | 4 | 27 | 0.1580 bps | 7.1052 bps |
| 21 points, spacing 0.25 | 104 | 0 | 13 | 0.0733 bps | 4.4248 bps |

For finite values, the reported root-decision gap is `1 - exp(Q_exact(chosen) - max Q_exact)`, expressed in basis points. The continuation in this score uses the exact-account catalog optimizer. It measures first-decision loss against that catalog, not cumulative realized backtest loss. An infeasible exact-catalog continuation is reported separately; it is neither evidence of literal bankruptcy nor proof that a finer continuous execution rule cannot close.

The four designed paths happened to produce zero root-decision gap on both grids. The randomized cases are therefore necessary: agreement on the obvious cases would miss material grid error. Finer exposures improve these aggregate diagnostics, but the two same-catalog references have different action menus, and the result is not a monotonic-refinement theorem.

The audit also exposed the missing exact-close guard in the penultimate sweep backup. That defect was fixed and independently covered by a failing-then-passing CPU/GPU regression before collecting the table above. The remaining discrepancies arise when querying the two-step field at actual post-trade accounts between representatives, or when exact catalog feasibility differs from interpolated support.

```powershell
.venv-ml/Scripts/python.exe ml/audit_oracle_dense_interpolation.py --output data/backtests/oracle-dense-interpolation-terminal-report-new --random-cases 100
```

These results do not justify attributing all poor policy performance to forecast distributions. Account/action approximation remains a separate source of error to reduce and measure.

## Common denser action reference

`data/backtests/oracle-dense-common-census-audit/results.json` repeats the same 104 laws using a shared exact-account reference with 81 exposure targets (spacing 0.0625). This axis contains both tested axes; price and size bins are unchanged. Both root catalog enumeration and future continuation use the common reference.

| Tested exposure axis | Cases with positive gap | Mean CE gap | Maximum CE gap | Chosen action infeasible in common reference |
| --- | ---: | ---: | ---: | ---: |
| 7 points | 50 / 104 | 1.4961 bps | 16.3181 bps | 0 |
| 21 points | 30 / 104 | 0.1629 bps | 2.4708 bps | 0 |

The coarse catalog's four infeasible continuations become feasible with the common denser menu. This demonstrates why they must not be described as inevitable liquidation or bankruptcy. The 7-point grid's worst common-reference case bought to the quote cap and also placed a sell limit; the better root action bought the same quantity without that sell order. It is an action-selection discrepancy, not just a numerical shift in the reported value.

The 21-point grid reduces the mean gap by about 89% in this synthetic sample. It remains an empirical comparison against a finite reference, not a certificate for the continuous optimum, long horizons, two simultaneous orders, or the actual NSF distribution. The common-reference audit took about 40 seconds of single-threaded CPU work while the GPU replay continued.

```powershell
.venv-ml/Scripts/python.exe ml/audit_oracle_dense_interpolation.py --output data/backtests/oracle-dense-common-census-audit-new --random-cases 100 --reference-exposure-count 81
```

## Which account axis benefits from refinement?

`data/backtests/oracle-dense-axis-refinement` holds the 104 laws, root accounts, order bins and 81-point exact-account reference fixed. It adds wealth representatives near the current account value, then halves exposure spacing again.

| Wealth points | Exposure spacing | Mean CE gap | Maximum CE gap | Infeasible in reference |
| --- | ---: | ---: | ---: | ---: |
| 3 | 0.25 | 0.162897 bps | 2.470811 bps | 0 |
| 11 | 0.25 | 0.162930 bps | 2.470811 bps | 0 |
| 11 | 0.125 | 0.065497 bps | 1.090108 bps | 0 |
| 3 | 0.125 | 0.065475 bps | 1.090108 bps | 0 |

The 11 wealth representatives are 5,000; 9,000; 9,500; 9,900; 9,990; 10,000; 10,010; 10,100; 10,500; 11,000; and 20,000. They add coverage close to fee-adjusted wealth while preserving the original range. In these cases they provide no practical improvement by themselves. Halving exposure spacing lowers the average gap by about 60%. This identifies exposure refinement as the more useful direction for this benchmark; it does not establish that wealth resolution is irrelevant for other account sizes or longer paths.

The two new variants took 93 seconds of single-threaded CPU work in total. The GPU backtest ran concurrently. Its implementation and settings were not changed during this comparison.

A follow-up in `oracle-dense-exposure41-audit` retained only the original three wealth points. It matched the 11-point variant's maximum gap and nearly matched its mean, while requiring 3/11 as many wealth/account representatives. It took 29.36 seconds for the 104 cases. The 41-exposure / three-wealth grid is therefore the next GPU profiling candidate.

```powershell
.venv-ml/Scripts/python.exe ml/audit_oracle_dense_axes.py --source data/backtests/oracle-dense-common-census-audit/results.json --output data/backtests/oracle-dense-axis-refinement-new
```

## Carried inventory and two pending orders

`data/backtests/oracle-dense-carried-audit` uses eight of the same finite laws and five initial account configurations: one buy; one sell; two opposing orders; long inventory with opposing orders; and short inventory with two buys at different limits. Every account starts with $10,000 marked equity and satisfies the reservation constraints. Orders can be cancelled or retained at the root; retained orders keep their actual prices and quantities.

Both grids are compared with a shared exact-account reference containing 21 exposure points and up to two orders. The 40 cases intentionally include difficult laws identified in the earlier audit, so the averages are diagnostics, not estimates over a representative market distribution.

| Tested grid | Mean first-decision CE gap | Maximum gap | Infeasible chosen continuations |
| --- | ---: | ---: | ---: |
| 7 exposure points | 5.8521 bps | 24.3811 bps | 0 |
| 21 exposure points | 0.03482 bps | 1.04269 bps | 0 |

The largest coarse-grid discrepancy occurred with existing long inventory and opposing orders. The reference added inventory up to the trade cap and cancelled the orders; the coarse solver bought less and posted a sell order. The finer grid's worst case instead missed the benefit of retaining one existing buy order. These examples confirm that the action-space diagnostics must include carried orders, not just cash-starting accounts. Both variants together took 17.96 seconds of single-threaded CPU work.

```powershell
.venv-ml/Scripts/python.exe ml/audit_oracle_dense_carried.py --source data/backtests/oracle-dense-common-census-audit/results.json --output data/backtests/oracle-dense-carried-audit-new
```

## Preparing the finer implementation benchmark

The experiment driver and GPU profiler now accept `--exposure-step`, including 0.125 and 0.01, and validate that the new axis retains all original exposure targets. The default future-candidate budget grows with the mandatory zero-order exposure targets and preserves nine additional order-bearing proposals. A fixed budget of 16 with 41 exposure points would otherwise silently omit all new-order proposals; that configuration is now rejected. Three focused configuration tests pass.

Only the exposure axis is nested. Randomly selected future order candidates need not form nested menus, so these runtime experiments cannot claim monotonic policy refinement. The full synthetic audit uses complete catalogs and a common reference specifically to avoid that ambiguity.

For 3,600 layers, eight market nodes, three wealth points and 45 canonical order books, the 41-point grid needs 2.55 GB for values and variances, compared with 0.435 GB for seven exposure points. GPU runtime and total peak memory still need measurement, including old fields retained for repair and the NSF model. The long replay continues with its frozen seven-point grid; the finer GPU profile will run after it releases the GPU.

```powershell
.venv-ml/Scripts/python.exe ml/profile_oracle_dense_gpu.py --source data/backtests/oracle-dense-raw-price-paired --output data/backtests/oracle-dense-exposure41-profile.json --depth 60 --exposure-step 0.125
```

The causal backtest driver now accepts the same `--exposure-step` and `--candidates` options. A 0.125 exposure step selects 41 account/exposure points and, by default, 50 future proposals: all 41 zero-order targets plus nine order-bearing proposals. Root search still covers every catalog cell and its configured interior sample. Resuming retains the saved grid and effective candidate count, even when the original market source was generated with the coarse account grid. The forecast cloud itself is unchanged by account-grid refinement. Five configuration tests pass, including default preservation, fine-grid resume and rejection of a budget that silently omits order proposals.

This command is available for the subsequent full replay; its practical runtime and memory budget should follow the queued GPU profile:

```powershell
.venv-ml/Scripts/python.exe ml/backtest_oracle_dense_repair.py --market-source data/backtests/oracle-dense-raw-price-paired --output data/backtests/oracle-dense-exposure41-replay --exposure-step 0.125 --seconds 3600 --repair-tolerance 1e-5 --max-wall-seconds 10800
```

## Finer root search on the saved one-hour learned-law field

The 41-exposure, 50-candidate GPU profile is complete (`data/backtests/oracle-dense-exposure41-profile.json`). Three warmed 60-depth sweeps took 12.8835, 12.9686 and 12.9714 seconds; the earlier seven-exposure profile took about 1.41 seconds. The finer full sweep is therefore about 9.2 times more expensive in this measurement. Linear extrapolation gives approximately 13 minutes for a complete 3,600-depth sweep, before forecasting/export. This is a runtime estimate, not a measured full-hour execution time. Profile peak allocation was 45,470,208 bytes for 60 depths, excluding the NSF model and a retained previous field; full replay memory must be checked separately.

`data/backtests/oracle-dense-root-exposure001` evaluates the actual initial $10,000 cash account against the corrected initial one-hour continuation map. It tests 401 exposure targets from -2 to 2 in increments of 0.01 across all 45 canonical order books. Exact rebalancing determines the resulting account; only feasible actions with finite continuation scores are retained. There were 3,715 such proposals. The original selected action remains available, so this search cannot reduce the best reported score.

The refined search still chose cash, with the same log value `9.210340371976184` and zero improvement in reported certainty equivalent. It took 11.43 seconds on CPU, including archive loading and the original root search. This checks root exposure coverage at one observed timestamp; the saved future action menus, account interpolation and market sample cloud are unchanged. It does not establish that the continuous one-hour optimum is cash or explain all later decisions.

`margins.json` now summarizes these saved proposals using `10000 * expm1(candidate_log_value - cash_log_value)`. Among 3,701 proposals carrying at least 1% absolute exposure, the best was +1% exposure without orders at **-0.07267 bps** relative to cash. The best short proposal was -1% at **-0.09999 bps**. Requiring at least 50% absolute exposure gave a best score of **-3.63313 bps**, at +50% without orders. None of those exposed proposals tied or beat cash. Thirteen finite proposals had pending orders and zero inventory; five tied cash within `1e-12` log units, and none exceeded it.

These are initial-state action-score margins under the archived approximation. They include future execution choices, fees and account interpolation, so subtracting a round-trip fee does not isolate the price model's expected return. The small 1% exposure margin also illustrates why a small field-error allowance alone need not certify a particular tiny action's ranking at every later decision. The initial archived field itself used a full finite-grid sweep with zero reuse residual.

```powershell
.venv-ml/Scripts/python.exe ml/audit_oracle_dense_root.py --map data/backtests/oracle-dense-repair-terminal-90m/initial --output data/backtests/oracle-dense-root-exposure001-new --step 0.01
```
