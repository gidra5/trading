# High-dimensional consecutive-return knot scaling (4D-15D)

Generated 2026-08-18T09:40:08.523019Z.

The dense full-grid JS used through 3D grows exponentially. This experiment instead uses deterministic sliced JS across coordinate, adjacent sum/difference, and random Cramer-Wold projections. The three group means receive equal weight, and the threshold is calibrated to the final exact 2D/3D boundaries.

Calibrated balanced mean sliced-JS threshold: **0.0059991285 bits**.

| dimensions | train / validation observations | all-active mass | knots | train balanced JS | validation balanced JS | p90 / max train JS |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 320,000 / 200,590 | 28.730204% | 3,455 | 0.0059031794 | 0.052546222 | 0.0054443419 / 0.011021042 |
| 5 | 320,000 / 195,694 | 24.542483% | 11,206 | 0.0059755622 | 0.05318238 | 0.0057091151 / 0.011345238 |
| 6 | 320,000 / 196,223 | 21.736826% | 29,058 | 0.0059715815 | 0.054755544 | 0.006217072 / 0.012246653 |
| 7 | 320,000 / 208,531 | 19.750557% | >32,768 | 0.0067554665 | 0.060062052 | 0.0064885468 / 0.019285619 |
| 8 | 320,000 / 225,031 | 18.294220% | >32,768 | 0.0076743563 | 0.064677798 | 0.0080368586 / 0.015832704 |
| 9 | 320,000 / 297,967 | 17.183652% | >32,768 | 0.0083713272 | 0.073895177 | 0.0097809648 / 0.018046255 |
| 10 | 320,000 / 320,000 | 16.311045% | >32,768 | 0.0090406579 | 0.08149335 | 0.012436919 / 0.018389106 |
| 11 | 320,000 / 304,500 | 15.604876% | >32,768 | 0.0089962807 | 0.086975191 | 0.014974694 / 0.017225954 |
| 12 | 320,000 / 320,000 | 15.017918% | >32,768 | 0.0094520579 | 0.093736638 | 0.015100962 / 0.016105102 |
| 13 | 320,000 / 320,000 | 14.519106% | >32,768 | 0.0094688072 | 0.099683509 | 0.01491854 / 0.016395037 |
| 14 | 320,000 / 320,000 | 14.087174% | >32,768 | 0.0095998912 | 0.1067469 | 0.015380183 / 0.016101049 |
| 15 | 320,000 / 320,000 | 13.706986% | >32,768 | 0.0096494283 | 0.11206163 | 0.016311388 / 0.017396849 |

## Interpretation limits

- These counts are one-knot training boundaries under the recorded deterministic multistart protocol; validation is diagnostic.
- A leading `>` means the fit still failed at the maximum tested count, so only a lower bound is established.
- Sliced JS tests the joint law through projections but is not numerically identical to full-grid JS.
- Only the all-active continuous component is fitted. Exact-zero mask probabilities are retained separately.
- Conditional mean/median errors are not acceptance criteria in this JS-only experiment.

## Reproduction

```text
node scripts/run-ml-python.mjs ml/scale_high_dimensional_return_knots.py
```
