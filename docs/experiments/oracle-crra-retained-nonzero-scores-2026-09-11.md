# Retained models and comparable nonzero-return scores

Following the shared-horizon experiment, retain the stronger existing
checkpoints. The shared trial is not promoted. The earlier joint density
remains the best surviving model on the matched one-second all-tick MSE
comparison. The calibrated separate-horizon hurdle specialists remain the
retained reference for the recent specialist experiment series. These are
forecast metric references, not new claims of trading profitability or a
live deployment change.

A subsequent fresh log-target retrain also leaves these references stronger.
Its calibrated one-second log MSE skill is 2.588907% overall and 3.988041%
conditional on nonzero outcomes. See
`oracle-crra-log-target-retrain-2026-09-11.md` for matched comparisons,
training selection and the verified backtests.

The [corrected log retrain with independent per-horizon copies](oracle-crra-per-horizon-log-retrain-2026-09-11.md)
restores calibrated one-second log MSE skill to 3.512772% overall, 5.729022%
on nonzero outcomes with the normal forecast, and 5.375497% with the
conditional-on-nonzero forecast. These closely match the existing specialist
reference; the earlier joint density still leads overall and active-conditional
MSE. The shared trial above should not be treated as the per-horizon log retrain.

## Same one-second test population

Both models were evaluated on 36,134 common test origins, of which 15,505
have a nonzero next-one-second return. The scores below use simple returns
and a zero-prediction baseline. Skill is `100*(1-MSE/mean(actualReturn^2))`,
computed over the row's stated population.

| Metric | Earlier joint density | Calibrated separate specialists |
| --- | ---: | ---: |
| All-tick MSE skill | 3.698803% | 3.512464% |
| Same deployed forecast, scored only on nonzero returns | 5.401430% | 5.729108% |
| Conditional-on-nonzero forecast, scored on nonzero returns | 6.296005% | 5.374942% |

For zero probability `p0`, conditional UP probability `p`, and conditional
UP/DOWN magnitude means `mu,md`, the conditional-on-nonzero forecast is
`c=p*mu-(1-p)*md`. The deployed full-distribution mean is `(1-p0)*c`.
Filtering scoring rows to realized nonzero returns leaves the deployed
forecast unchanged; conditioning the forecast removes the zero-probability
multiplier. Those are different measurements and can rank models differently.
The conditional score is a diagnostic; future activity is not known when
the bot forecasts or trades.

For comparison with the earlier matched active-conditional audit, use
**6.296% for the joint density and 5.375% for the calibrated specialists**.
The historical roughly 6.5–7% headline used log returns and a different
active-return sample from deleted checkpoints. It is not an exact matched
comparison to these simple-return figures.

The calibrated specialists' nonzero sign accuracy at one second is 84.4244%.
Their active conditional/deployed MSE skills at the other horizons are:

| Horizon | Nonzero examples | Conditional forecast | Deployed forecast |
| --- | ---: | ---: | ---: |
| 15 minutes | 36,113 | -1.760520% | -1.757650% |
| 30 minutes | 36,098 | -6.647092% | -6.642795% |
| 2 hours | 36,131 | 0.919358% | 0.919991% |

## Training targets and matched log-return rescoring

The earlier joint model learns a density of log returns
`X=log(P[t+h]/P[t])`. Its training objective combines log-return density
negative log likelihood with normalized **simple-return mean MSE**:
`(E[exp(X)-1] - (exp(actualX)-1))^2`. It was not trained solely on
log-return MSE.

The newer specialists instead use **simple-return magnitude targets**
`R=P[t+h]/P[t]-1`. Sign pretraining uses active-only cross entropy; size
pretraining uses sign-conditioned simple-magnitude MSE. Hurdle fine-tuning
and online calibration use the Gamma-UP/Beta-DOWN likelihood on simple
returns. Zero and sign labels are invariant under the log transform, but
magnitude targets and their losses are not. Historical log-return inputs
do not make these output targets log returns. This target change should
have been stated explicitly when introducing the specialists.

Both retained models have now been rescored on actual log returns over
the same 36,134 origins and 15,505 nonzero outcomes. Forecasts integrate
`E[log(1+R)]` under each unchanged predictive law; they do not use the
different quantity `log(1+E[R])`. Skill uses the same zero baseline formula
as above, with actual and predicted log returns.

| Log-return MSE skill | Earlier joint density | Calibrated separate specialists |
| --- | ---: | ---: |
| All-tick deployed forecast | 3.698769% | 3.513035% |
| Deployed forecast, scored only on nonzero returns | 5.401253% | 5.728953% |
| Conditional-on-nonzero forecast, scored on nonzero returns | 6.295851% | 5.375472% |

At one second, changing evaluation coordinates has a negligible effect.
It does not explain the much larger gap to the historical headline. The
historical sample and deleted checkpoints still prevent an exact matched
comparison; this rescoring also does not establish what retraining the
specialists on log-magnitude targets would achieve.

Reproducible audits are `ml/audit_joint_log_returns.py` and
`ml/audit_hurdle_log_returns.py`, with outputs in
`data/backtests/oracle-online-calibration/joint-log-return-audit.json` and
`data/backtests/oracle-online-calibration/hurdle-log-return-audit.json`.
Both agree on origin and target hashes, reproduce the prior simple-return
scores, and check numerical integration convergence. No weights,
calibration settings, or model selections changed for this rescoring.

## Retained artifact references

- Joint one-second checkpoint:
  `data/training/runs/stochastic-crra-direct-horizons-gaussian-v2/h1/lambda1/best.pt`,
  SHA-256 `2382fd195fd62203137a44b4752189013cd60c0d3b2e683d2d03e57fc9a1e258`.
- Joint matched audit:
  `data/backtests/oracle-neural-specialists/joint-marginal-audit.json`.
- Specialist neural selection:
  `data/training/runs/stochastic-crra-hurdle-nll-v1/selection.json`,
  SHA-256 `8ac7d25215c733b284d26cf97c09b741ffc81cc786eae20ef195547286777317`.
- Specialist calibration selection:
  `data/training/runs/stochastic-crra-online-calibration-v1/selection.json`,
  SHA-256 `3a2987241eace925c2a1281f936e5d641524bba048d6322fa482a04c3d694f63`.
  Preserve validation-preferred full calibration at one second and
  probability-only calibration at 900, 1800 and 7200 seconds.

The specialist one-second metrics were independently recomputed from the
hashed saved forecast NPZ and observed market prices, matching the recorded
results exactly. The joint checkpoint remains present and its hash matches
the earlier audit. No weights, calibration settings, forecasts or trading
configuration were changed for this metric comparison.
