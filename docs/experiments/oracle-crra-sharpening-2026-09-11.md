# Distribution sharpening and the entry threshold

Sharpening can push a forecast over the entry threshold by changing its expected
payoff. It does not change the payoff of the trade under the true price process.
It is useful as calibration only when the adjusted probabilities better describe
future outcomes. The existing optimality theorem then applies to the adjusted
forecast model; it does not certify that the adjustment improved that model.

## Mean and confidence

In the surviving, valid-full-close long case with gamma zero, expected net wealth
depends on the expected price ratio. Contracting **price** outcomes toward their
mean preserves that ratio and the action ranking. Merely reducing variance
therefore cannot overcome the fee threshold in this case.

With liquidation, pending limits or a capped residual close, payoffs are
nonlinear and preserving the price mean alone need not preserve action values.
Contracting **log returns** about their log mean is different: even for a Gaussian,
`E[exp(R)] = exp(mu + variance/2)`, so lowering variance at fixed `mu` lowers the
expected price ratio.

For a concrete probability-sharpening example, suppose gross returns are
`1.002` with probability `0.6` and `0.998` with probability `0.4`. The mean return
is 4 bps. Oracle's 5-bps entry and exit fees give an expected net return on the
quote spent of approximately **-6.0015 bps**, assuming survival and a valid full
close.

Sharpen the two probabilities using `p_i^(1/T)` and normalize, with `T=0.25`.
The positive probability becomes `0.8350515` and the transformed forecast's mean
return becomes 13.4021 bps. It now predicts a profitable long, but the original
law still gives -6.0015 bps expected net return. This is why passing a predicted
switching gate alone is insufficient; the value-error condition in
[Decision-informative distributions](../theory/Decision-informative%20distributions.md)
also matters.

## Measured temperature calibration

The experiment uses the three existing selected horizon checkpoints, without
retraining or changing the running provider. The prespecified temperature grid
is `0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 4`. Lower temperatures amplify the
active knot-height logit differences. The zero-return atom probability is held
fixed to isolate the active distribution.

For a knot height logit `z_i` and triangular area `A_i`, the transformed mixture
mass is proportional to `A_i * exp(z_i/T)`. The area is retained outside the
temperature. This is a specific calibration of the model's Gaussian-coordinate
density head, not literal exponentiation of the continuous price density. The
Jacobian remains unchanged. Different coordinates or sharpening transformations
can behave differently; lower head temperature does not universally imply lower
price variance or a larger positive mean.

All temperatures are evaluated on validation data. Selection minimizes mixed-law
negative log likelihood, including the unchanged `T=1` control, before loading
test data for this experiment. Validation was already used to choose the original
checkpoints, and the uncalibrated test split was used in earlier experiments;
this is not a new untouched dataset or a preregistered profitability study.

**All three horizons selected T=1.** Every tested sharper setting worsened
validation likelihood. The selected frozen test comparison therefore reproduces
the original model, with no paid long entries.

The two-minute model nevertheless demonstrates that forced sharpening can pass
the predicted long-entry gate:

| Temperature | Validation NLL (lower is better) | Maximum predicted mean, bps | Paid long signals / 1,211 contexts | Realized net return when long, bps |
| --- | ---: | ---: | ---: | ---: |
| 1 | -6.0146 | 1.5297 | 0 | — |
| 0.1 | 18.0524 | 41.4898 | 8 | -5.7418 |
| 0.05 | 50.6243 | 62.4330 | 65 | -8.8834 |

These are nonoverlapping validation horizon labels and independent fresh-entry
payoff calculations, not a compounded account backtest. Serial dependence may
remain. The threshold is 10.007505 bps at 5 bps per side and unit funding, under
the full-close assumptions. The paid switching gate becomes true at 0.1 and
0.05, but the realized conditional averages are negative and likelihood is much
worse. Neither setting was selected for test evaluation or deployment.

The one-second model does not cross the positive threshold at any tested
temperature. Strong sharpening of the 30-second model instead moves forecasts
toward a negative peak in Gaussian coordinates; it does not produce long entries.
This diagnostic does not evaluate a short or unrestricted leveraged policy.

## Artifacts and checks

`ml/calibrate_horizon_temperature.py` saves its grid and selection rule before
evaluation and each horizon's selected temperature before loading test. It
computes expected price returns in double precision, checks finite outputs, and
asserts that temperature-one validation likelihood agrees with the original
checkpoint. All three controls passed. Existing checkpoints and execution code
are unchanged; no new Lean theorem is claimed for this calibration experiment.

Results: `data/training/runs/stochastic-crra-temperature-v1/report.json`.
Execution log: `.tools/oracle-temperature-calibration.log`.

```powershell
node scripts/run-ml-python.mjs ml/calibrate_horizon_temperature.py --output data/training/runs/stochastic-crra-temperature-reproduction
```

The measured result supports retaining the existing temperature. A different
calibration could help an underconfident model, but crossing the fee threshold
must follow improved probability or payoff estimation, rather than serve as the
criterion for changing the probabilities.
