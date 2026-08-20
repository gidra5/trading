# Production-basis next-return density GLU

Date: 2026-08-17  
Status: complete

## Setup

- Same 178-feature, 65,536-clean-example train/validation/test matrix as the
  matched scalar production-basis experiment.
- Four fused GLU layers of width 512 with static C and no dropout or decay.
- 256 fixed global return knots under the frozen asinh-logistic transform.
- The model emits conditional log-density heights and is trained by
  return-space negative log likelihood (NLL).
- The canonical point prediction is the exact expectation of the reconstructed
  knot distribution.
- 512 epochs with six independently retained checkpoints: best train and
  validation MSE, correlation, and NLL.
- 6,088,968 total parameters.

## Single-return checkpoint comparison

| checkpoint | epoch | validation skill | validation corr. | validation NLL | test skill | test corr. | test NLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| train MSE | 218 | -9.283% | 0.03239 | -5.31469 | -13.991% | 0.05208 | -11.20156 |
| validation MSE | 35 | +3.260% | 0.18986 | -12.74733 | +4.813% | 0.22592 | -15.45282 |
| train correlation | 192 | -9.276% | 0.03444 | -5.87206 | -13.996% | 0.05042 | -11.47062 |
| validation correlation | 31 | +3.205% | 0.19271 | -12.78771 | +4.761% | 0.22863 | -15.46642 |
| train NLL | 512 | -16.234% | 0.03736 | -2.31864 | -26.862% | 0.05206 | -9.66093 |
| validation NLL | 9 | +0.534% | 0.16686 | **-13.41960** | +0.896% | 0.19419 | -15.05588 |

The NLL-selected and expectation-selected checkpoints are materially
different. Best validation NLL occurs at epoch 9, while best expectation MSE
and correlation occur at epochs 35 and 31. Selecting by validation MSE makes
the expectation outperform the matched scalar model: +3.260% versus +2.694%
validation skill and +4.813% versus +4.200% test skill. Correlation also rises
from 0.169/0.205 to 0.190/0.226 on validation/test.

The best-train checkpoints confirm complete memorization does not generalize.
They approach 98% train expectation skill but have negative held-out skill and
substantially worse NLL than the fixed global distribution.

## Autoregressive expectation

The expectation is appended to the input history for step 2 and step 3. The
table uses the best-validation-MSE checkpoint.

| split | step | MSE skill | correlation | direction |
|---|---:|---:|---:|---:|
| validation | 1 | +3.259% | 0.18984 | 48.552% |
| validation | 2 | +0.325% | 0.08562 | 63.152% |
| validation | 3 | -0.498% | 0.06760 | 46.945% |
| test | 1 | +4.813% | 0.22591 | 46.147% |
| test | 2 | +0.649% | 0.12031 | 66.393% |
| test | 3 | -1.316% | 0.09305 | 45.256% |

For cleaned 15-minute open-loop episodes, pooled MSE skill is -7.812% on
validation and -16.189% on test. Thus the density expectation retains useful
skill for one extra active return but still accumulates error over full
episodes.

The best-validation-NLL checkpoint produces much more conservative
expectations: its pooled 15-minute skill is -0.105% validation and -0.149%
test, essentially the zero-return baseline. This is better rollout MSE than
the expectation-selected checkpoint, but it achieves that by predicting very
small means rather than forecasting the path accurately; pooled correlations
remain near zero.

## Pre-validation calibration

Calibration uses 16,384 clean active returns from 2026-08-02 12:00 UTC until
the quota is reached, strictly after the retained training prefix and before
the 2026-08-03 validation boundary. The test split remains untouched until all
parameters are frozen. Every checkpoint receives three variants:

- scale-only expectation calibration;
- affine expectation calibration;
- density-mass temperature calibration.

| checkpoint | scale | scaled validation skill | scaled test skill | temperature | temperature-calibrated validation/test NLL |
|---|---:|---:|---:|---:|---:|
| validation MSE | 1.4132 | +3.602% | +5.054% | 1.8472 | -12.85757 / -14.72490 |
| validation correlation | 1.5117 | **+3.703%** | **+5.198%** | 1.7851 | -12.91148 / -14.80463 |
| validation NLL | 7.9562 | +2.673% | +3.718% | 0.9635 | **-13.42987 / -15.09327** |

The affine intercepts are tiny and perform almost identically to scale-only
calibration. For the expectation objective, scale-only calibration of the
best-validation-correlation checkpoint is strongest. Temperature calibration
answers a different question: the best-validation-NLL checkpoint needs almost
no temperature change and remains the strongest calibrated validation NLL.
Temperature values fitted to the expectation-selected checkpoints are above
one and improve validation NLL, but worsen test NLL, indicating a distribution
shift in sharpness.

## Artifacts

- Plan: `ml/training-plans/next-return-production-basis-density-4l-65k-k256-v1.json`
- Trainer: `ml/train_feature_augmented_return_density.py`
- Calibration exporter: `scripts/export-next-return-production-basis.ts --calibration-only`
- Calibrator: `ml/calibrate_feature_augmented_return_density.py`
- Multi-step evaluator: `ml/evaluate_feature_augmented_next_return.py`
- Run: `data/training/runs/next-return-production-basis-density-4l-65k-k256-v1`
