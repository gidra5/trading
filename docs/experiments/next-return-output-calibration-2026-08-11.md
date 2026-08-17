# Next-return output calibration — 2026-08-11

## Question

Can a frozen scalar output calibration recover held-out MSE skill from the
baseline next-second GLU scaling diagnostics without changing their weak but
positive correlation?

## Protocol

- Models: the six static-C baselines that vary only model depth/width and the
  fixed contiguous training-example count.
- Training subsets: start on 2026-04-01 and end no later than
  2026-04-13T03:16:15Z.
- Calibration: 604,680 one-second examples from 2026-05-25 through
  2026-05-31. The final calendar day retains the existing 121-second purge.
- Validation: 2026-06-01 through 2026-06-30.
- Test: 2026-07-01 through 2026-07-23, untouched until calibration parameters
  were frozen.
- Fit: weighted ordinary least squares. Both a scale-only transform
  `a * prediction` and an affine transform `a * prediction + b` were recorded.
- Artifact: each run stores
  `state/output-calibration-pre-validation-7d.json`.

Jointly training the two calibration parameters with the network was rejected:
the existing output projection and bias can already represent the same
transform under the training MSE. A frozen post-hoc fit provides distinct
chronological evidence and has an exact closed-form solution.

## Scale-only results

| Model | Fit examples | Scale | Calibration corr. | Raw validation skill | Calibrated validation skill | Raw test skill | Calibrated test skill |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4L × 512, 65k | 65,536 | 0.03260 | 0.02752 | -22.3069% | +0.0621% | -48.9799% | +0.0692% |
| 4L × 512, 128k | 131,072 | 0.03718 | 0.02551 | -18.2811% | +0.0712% | -34.8544% | +0.0796% |
| 4L × 512, 256k | 262,144 | 0.04885 | 0.03034 | -16.1229% | +0.0743% | -29.0189% | +0.0820% |
| 4L × 512, 512k | 524,288 | 0.04646 | 0.02788 | -16.8519% | +0.0649% | -28.2513% | +0.0679% |
| 8L × 512, 512k | 524,288 | 0.15320 | 0.03557 | -1.9175% | +0.1093% | -3.1814% | **+0.1209%** |
| 8L × 1024, 1,024k | 1,048,576 | 0.10191 | 0.02665 | -3.6715% | +0.0528% | -4.8515% | +0.0595% |

## Interpretation

Scale-only calibration converts negative raw skill into small positive skill
on both future periods for all six models. It does not create information:
correlation and sign ordering are unchanged. It removes excessive forecast
amplitude, and the remaining attainable MSE gain is approximately the squared
correlation.

The 8-layer width-512 model remains strongest. Its frozen May scale of 0.1532
produces +0.1093% validation skill and +0.1209% test skill. Scaling width and
data again reduces the calibrated test skill to +0.0595%, confirming the
earlier diminishing-return result.

The affine intercept is not stable. May has a small negative target mean while
July has a small positive mean; carrying May's negative intercept forward
slightly worsens test MSE and severely distorts direction accuracy. The
recommended transform is therefore scale-only, with intercept fixed at zero.
