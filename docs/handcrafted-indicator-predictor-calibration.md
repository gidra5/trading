# Handcrafted indicator predictor calibration

Generated on 2026-07-20 with `npm run calibrate:indicator-predictor`.

The calibration uses all 33 static inspector windows. The dynamic **Latest history** window is
excluded deliberately so it remains an out-of-sample inspection target. Every fitted window
contributes eight evenly spaced causal samples after a three-day warmup, for 264 samples total.
The latest window ends at the current UTC-day boundary and has no configured day-count ceiling;
when a requested day is absent locally, the inspector fetches and verifies its Binance daily
archive before evaluation and surfaces the fetch error if the archive is unavailable.

The target uses the same configuration as the inspector default:

- 60-second candles and mandatory holding period;
- one-hour rolling value horizon;
- 31 calibration-grid targets from -100 to +100 exposure;
- 0.175% friction;
- 0.01 oracle temperature; and
- no quote-debt or asset-debt borrow maintenance.

The primary loss is realized oracle regret at the exposure selected by the forecast predictor.
Distribution cross-entropy is retained as a diagnostic and is not optimized. This follows the
decision objective in the design guide and prevents a diffuse distribution from scoring well
without selecting a useful exposure.

The calibration command also evaluates the same candidate pool separately on each static window
and coordinate-refines the winner. Running it with `--write-presets` regenerates the inspector's
one global forecast preset plus 33 one-minute local hindsight presets. A local preset is useful as
an upper-bound diagnostic on its own window; it is not out-of-sample evidence.

## Result

| Metric | Value |
|---|---:|
| Default mean decision regret | 0.10687857 |
| Calibrated mean decision regret | 0.10527748 |
| Relative regret reduction | 1.50% |
| Calibrated diagnostic cross-entropy | 11.89985132 |

| Parameter | Calibrated value |
|---|---:|
| Drift-estimation half-life | 2,657,296.395 ms |
| Drift-forecast half-life | 1,316,056.907 ms |
| Drift shrinkage/scale | 0.631158797 |
| Variance-estimation half-life | 29,220,215.857 ms |
| Long-run variance half-life | 28,897,920.875 ms |
| Variance-forecast half-life | 7,716,496.504 ms |

The improvement is real but modest. The latest-history mode should therefore be treated as an
important out-of-sample check rather than assuming the handcrafted drift signal has a strong edge.
The heatmap's optional per-candle fit is an even stronger hindsight diagnostic: it searches the
same bounded six-parameter space directly against the selected candle's conditional oracle
cross-entropy and never contributes to aggregate causal accuracy metrics.
