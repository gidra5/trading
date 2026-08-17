# Fourier, fractional-Fourier, and wavelet features for BTC return distributions — 2026-08-17

## Result

The audit applies causal FFT, unitary fractional-DFT powers at orders 0.25/0.5/0.75, orthonormal Haar decompositions, and complex causal Morlet filters to trailing signed returns. Complex coordinates are supplied as separate normalized real and imaginary values. It tests 16, 64, and 256-sample windows at each native input cadence. Selection uses only the primary period; the later transfer year is untouched until confirmation.

| target | fixed baseline | selected time-frequency additions | primary gain (bits) | transfer gain (bits) | transfer conclusion |
|---:|---|---|---:|---:|---|
| 1s | range-1s, active-count-60s, previous-return-1s, realized-volatility-60s, close-location-1s | return-lag-2s-control | 0.00635117 | 0.00560449 | confirmed |
| 1m | realized-volatility-30m, range-1m, realized-volatility-15m, realized-volatility-60m | none | 0.00000000 | 0.00000000 | not confirmed |
| 15m | realized-volatility-60m, realized-volatility-15m, realized-volatility-240m | none | 0.00000000 | 0.00000000 | not confirmed |
| 1h | realized-volatility-30m, realized-volatility-240m | none | 0.00000000 | 0.00000000 | not confirmed |

Fourier energy is not intrinsically new: by Parseval's theorem it is a re-expression of return energy and therefore closely overlaps realized volatility. A useful result requires normalized complex coefficients, fractional-domain structure, or wavelet localization to remain positive after conditioning on the fixed basis and on the untouched transfer period.

## Efficiency-ratio audit

| target | objective | best efficiency coordinate | standalone primary/transfer | conditional primary/transfer | conditional blocks |
|---:|---|---|---:|---:|---|
| 1s | full distribution | signed-path-efficiency-16s | 0.015384 / 0.010297 | -0.001530 / -0.002517 | 0.00342, -0.00648, -0.00624, 0.00121 |
| 1s | magnitude | path-efficiency-16s | 0.055435 / 0.058185 | -0.002999 / -0.002661 | -0.00259, -0.00341, -0.00194, -0.00338 |
| 1s | sign | signed-variance-efficiency-16s | 0.008182 / 0.007316 | 0.005504 / 0.006501 | 0.00515, 0.00586, 0.00807, 0.00493 |
| 1m | full distribution | path-efficiency-16m | 0.000648 / 0.000599 | -0.007563 / -0.009337 | -0.00747, -0.00766, -0.00710, -0.01157 |
| 1m | magnitude | path-efficiency-16m | 0.000783 / 0.000935 | -0.006845 / -0.009128 | -0.00676, -0.00693, -0.00709, -0.01117 |
| 1m | sign | signed-variance-efficiency-64m | 0.000059 / 0.000355 | -0.000982 / -0.000669 | -0.00112, -0.00084, -0.00065, -0.00069 |
| 15m | full distribution | signed-variance-efficiency-256m | 0.004370 / 0.004809 | -0.010495 / -0.012078 | -0.00926, -0.01173, -0.01589, -0.00827 |
| 15m | magnitude | signed-path-efficiency-16m | 0.000199 / 0.000196 | -0.010532 / -0.011525 | -0.00953, -0.01153, -0.00933, -0.01372 |
| 15m | sign | signed-variance-efficiency-64m | 0.000037 / 0.000600 | -0.000784 / -0.000365 | 0.00043, -0.00199, -0.00117, 0.00044 |
| 1h | full distribution | path-efficiency-64m | -0.000454 / -0.003650 | -0.014444 / -0.020372 | -0.01559, -0.01329, -0.02385, -0.01690 |
| 1h | magnitude | signed-path-efficiency-16m | -0.001088 / -0.000932 | -0.013010 / -0.018326 | -0.01165, -0.01437, -0.01767, -0.01899 |
| 1h | sign | signed-variance-efficiency-16m | 0.000801 / 0.001487 | -0.000192 / 0.000426 | 0.00149, -0.00187, 0.00008, 0.00077 |

## 1s target

Observations: 1,578,180 train, 525,600 primary, 525,600 transfer.

| objective | selected time-frequency additions | primary gain | transfer gain | primary/transfer half-block gains |
|---|---|---:|---:|---|
| full distribution | return-lag-2s-control | 0.00635117 | 0.00560449 | 0.007505, 0.005197, 0.006287, 0.004922 |
| magnitude | fft-log-energy-256s | 0.00141410 | -0.00083701 | 0.002155, 0.000673, -0.000856, -0.000818 |
| sign | return-difference-1s-control, signed-variance-efficiency-16s | 0.00967145 | 0.01007605 | 0.009143, 0.010200, 0.011511, 0.008641 |

After fixing the second-lag return in the baseline:

| objective | lag-2 primary/transfer gain | remaining selected path features | additional primary/transfer gain |
|---|---:|---|---:|
| full distribution | 0.00635117 / 0.00560449 | none | 0.00000000 / 0.00000000 |
| magnitude | 0.00095658 / 0.00061454 | none | 0.00000000 / 0.00000000 |
| sign | 0.00429403 / 0.00326145 | haar-latest-detail-l1-16s | 0.00379160 / 0.00339153 |

Individual 1s sign coordinates after fixing lag 2:

| coordinate | additional primary/transfer bits | blocks |
|---|---:|---|
| haar-latest-detail-l1-16s | 0.00379160 / 0.00339153 | 0.003261, 0.004322, 0.005575, 0.001208 |
| haar-latest-detail-l2-256s | 0.00231485 / 0.00403769 | 0.001613, 0.003016, 0.004733, 0.003342 |
| signed-variance-efficiency-16s | 0.00073360 / 0.00234682 | 0.000509, 0.000958, 0.003611, 0.001083 |
| signed-path-efficiency-16s | 0.00070495 / 0.00216927 | 0.000455, 0.000955, 0.003607, 0.000732 |

For the 1s sign head, the surviving coefficient after fixing lag 2 is

$$
d_{1,16}(t)=\frac{r_{t-1}-r_t}{\sqrt{2}\,\sqrt{\sum_{j=0}^{15}r_{t-j}^2}}.
$$

The tested 16-s efficiency coordinates were Kaufman's path ratio, its signed form, and a variance-normalized signed form:

$$
ER_{16}(t)=\frac{|\sum_{j=0}^{15}r_{t-j}|}{\sum_{j=0}^{15}|r_{t-j}|},\qquad SER_{16}(t)=\frac{\sum_{j=0}^{15}r_{t-j}}{\sum_{j=0}^{15}|r_{t-j}|},\qquad E_{16}(t)=\frac{\sum_{j=0}^{15}r_{t-j}}{\sqrt{\sum_{j=0}^{15}r_{t-j}^2}}.
$$

Thus the complete-distribution gain is primarily ordinary second-lag information. The efficiency ratios retain a small, stable 1s sign gain after lag 2, but the normalized adjacent-return Haar contrast is stronger and is the parsimonious selection. No efficiency coordinate survives for the full-distribution or magnitude heads, or at 1m through 1h.

Top full-distribution time-frequency coordinates, each added separately to the fixed baseline:

| feature | family | lookback | standalone primary/transfer | conditional primary/transfer | conditional blocks |
|---|---|---:|---:|---:|---|
| return-lag-2s-control | time-domain controls | 256s | 0.144435 / 0.176544 | 0.006351 / 0.005604 | 0.00751, 0.00520, 0.00629, 0.00492 |
| haar-latest-detail-l1-256s | wavelet local coefficients | 256s | 0.129415 / 0.141360 | 0.005767 / 0.005349 | 0.00675, 0.00478, 0.00430, 0.00640 |
| haar-latest-detail-l1-64s | wavelet local coefficients | 64s | 0.106631 / 0.109951 | 0.002701 / 0.001288 | 0.00501, 0.00039, -0.00157, 0.00414 |
| fft-log-energy-16s | Fourier energy | 16s | 0.125757 / 0.126972 | -0.001056 / 0.000032 | -0.00008, -0.00204, -0.00197, 0.00204 |
| signed-path-efficiency-16s | signed path efficiency | 16s | 0.015384 / 0.010297 | -0.001530 / -0.002517 | 0.00342, -0.00648, -0.00624, 0.00121 |
| signed-variance-efficiency-16s | signed variance efficiency | 16s | 0.010438 / 0.005954 | -0.001654 / -0.002665 | 0.00356, -0.00687, -0.00688, 0.00155 |
| frft-0p25-entropy-64s | fractional Fourier shape | 64s | 0.120674 / 0.127161 | -0.001708 / -0.001641 | 0.00082, -0.00423, -0.00628, 0.00300 |
| frft-0p25-entropy-256s | fractional Fourier shape | 256s | 0.163547 / 0.183683 | -0.002467 / -0.003552 | 0.00231, -0.00724, -0.00927, 0.00216 |
| haar-fine-energy-share-16s | wavelet energy shape | 16s | 0.018876 / 0.018399 | -0.002566 / -0.002245 | -0.00174, -0.00340, -0.00327, -0.00122 |
| path-efficiency-16s | path efficiency | 16s | 0.049024 / 0.048744 | -0.002958 / -0.002660 | -0.00188, -0.00403, -0.00331, -0.00201 |

## 1m target

Observations: 394,500 train, 131,400 primary, 131,385 transfer.

| objective | selected time-frequency additions | primary gain | transfer gain | primary/transfer half-block gains |
|---|---|---:|---:|---|
| full distribution | none | 0.00000000 | 0.00000000 | 0.000000, 0.000000, 0.000000, 0.000000 |
| magnitude | none | 0.00000000 | 0.00000000 | 0.000000, 0.000000, 0.000000, 0.000000 |
| sign | frft-0p25-entropy-256m | 0.00032087 | 0.00004811 | 0.000258, 0.000384, 0.001070, -0.000974 |

Top full-distribution time-frequency coordinates, each added separately to the fixed baseline:

| feature | family | lookback | standalone primary/transfer | conditional primary/transfer | conditional blocks |
|---|---|---:|---:|---:|---|
| fft-log-energy-64m | Fourier energy | 64m | 0.176982 / 0.185397 | -0.002737 / -0.002887 | -0.00318, -0.00229, -0.00230, -0.00347 |
| fft-log-energy-16m | Fourier energy | 16m | 0.180171 / 0.187221 | -0.002962 / -0.003682 | -0.00305, -0.00288, -0.00315, -0.00422 |
| fft-log-energy-256m | Fourier energy | 256m | 0.140150 / 0.153586 | -0.004280 / -0.004351 | -0.00500, -0.00356, -0.00322, -0.00548 |
| frft-0p25-entropy-256m | fractional Fourier shape | 256m | 0.012468 / 0.010049 | -0.005289 / -0.005855 | -0.00521, -0.00537, -0.00272, -0.00899 |
| frft-0p25-entropy-64m | fractional Fourier shape | 64m | 0.014122 / 0.012664 | -0.005291 / -0.005037 | -0.00542, -0.00516, -0.00287, -0.00721 |
| frft-0p5-entropy-64m | fractional Fourier shape | 64m | 0.006017 / 0.005399 | -0.006492 / -0.006940 | -0.00751, -0.00548, -0.00527, -0.00861 |
| haar-latest-detail-l3-64m | wavelet local coefficients | 64m | 0.000498 / 0.000968 | -0.007012 / -0.008140 | -0.00698, -0.00704, -0.00761, -0.00867 |
| frft-0p75-k1-imag-256m | fractional Fourier complex coefficients | 256m | -0.000225 / -0.000205 | -0.007277 / -0.008446 | -0.00629, -0.00826, -0.00766, -0.00923 |
| frft-0p25-entropy-16m | fractional Fourier shape | 16m | 0.006845 / 0.006985 | -0.007393 / -0.006870 | -0.00775, -0.00703, -0.00509, -0.00865 |
| path-efficiency-16m | path efficiency | 16m | 0.000648 / 0.000599 | -0.007563 / -0.009337 | -0.00747, -0.00766, -0.00710, -0.01157 |

## 15m target

Observations: 98,557 train, 32,828 primary, 32,824 transfer.

| objective | selected time-frequency additions | primary gain | transfer gain | primary/transfer half-block gains |
|---|---|---:|---:|---|
| full distribution | none | 0.00000000 | 0.00000000 | 0.000000, 0.000000, 0.000000, 0.000000 |
| magnitude | none | 0.00000000 | 0.00000000 | 0.000000, 0.000000, 0.000000, 0.000000 |
| sign | none | 0.00000000 | 0.00000000 | 0.000000, 0.000000, 0.000000, 0.000000 |

Top full-distribution time-frequency coordinates, each added separately to the fixed baseline:

| feature | family | lookback | standalone primary/transfer | conditional primary/transfer | conditional blocks |
|---|---|---:|---:|---:|---|
| fft-log-energy-64m | Fourier energy | 64m | 0.143708 / 0.152231 | -0.001954 / -0.004259 | -0.00235, -0.00156, -0.00407, -0.00444 |
| fft-log-energy-256m | Fourier energy | 256m | 0.120363 / 0.124256 | -0.002965 / -0.003921 | -0.00330, -0.00263, -0.00298, -0.00486 |
| fft-log-energy-16m | Fourier energy | 16m | 0.140825 / 0.137197 | -0.004389 / -0.004194 | -0.00358, -0.00519, -0.00414, -0.00424 |
| frft-0p25-entropy-256m | fractional Fourier shape | 256m | 0.009581 / 0.006639 | -0.007734 / -0.010831 | -0.00758, -0.00789, -0.00860, -0.01306 |
| frft-0p5-entropy-256m | fractional Fourier shape | 256m | 0.005346 / 0.003523 | -0.009500 / -0.009555 | -0.00819, -0.01081, -0.00577, -0.01334 |
| fft-low-power-share-256m | Fourier shape | 256m | -0.000445 / -0.000200 | -0.009881 / -0.010841 | -0.00966, -0.01011, -0.00947, -0.01221 |
| frft-0p25-entropy-16m | fractional Fourier shape | 16m | 0.003932 / 0.004277 | -0.009964 / -0.009417 | -0.01126, -0.00867, -0.01018, -0.00865 |
| frft-0p5-k1-real-16m | fractional Fourier complex coefficients | 16m | 0.000161 / -0.000587 | -0.010265 / -0.012586 | -0.01250, -0.00803, -0.01352, -0.01165 |
| frft-0p75-k1-real-16m | fractional Fourier complex coefficients | 16m | -0.000122 / -0.000856 | -0.010455 / -0.016308 | -0.01190, -0.00901, -0.01583, -0.01679 |
| signed-variance-efficiency-256m | signed variance efficiency | 256m | 0.004370 / 0.004809 | -0.010495 / -0.012078 | -0.00926, -0.01173, -0.01589, -0.00827 |

## 1h target

Observations: 26,282 train, 8,754 primary, 8,753 transfer.

| objective | selected time-frequency additions | primary gain | transfer gain | primary/transfer half-block gains |
|---|---|---:|---:|---|
| full distribution | none | 0.00000000 | 0.00000000 | 0.000000, 0.000000, 0.000000, 0.000000 |
| magnitude | none | 0.00000000 | 0.00000000 | 0.000000, 0.000000, 0.000000, 0.000000 |
| sign | frft-0p75-k1-real-64m | 0.00076706 | -0.00117936 | 0.001438, 0.000096, -0.000610, -0.001749 |

Top full-distribution time-frequency coordinates, each added separately to the fixed baseline:

| feature | family | lookback | standalone primary/transfer | conditional primary/transfer | conditional blocks |
|---|---|---:|---:|---:|---|
| fft-log-energy-64m | Fourier energy | 64m | 0.112167 / 0.128130 | -0.003122 / -0.002230 | -0.00239, -0.00385, -0.00180, -0.00266 |
| fft-log-energy-256m | Fourier energy | 256m | 0.098905 / 0.108059 | -0.003799 / -0.002255 | -0.00466, -0.00293, -0.00158, -0.00293 |
| fft-log-energy-16m | Fourier energy | 16m | 0.109390 / 0.118064 | -0.008162 / -0.006677 | -0.00844, -0.00789, -0.00657, -0.00679 |
| morlet-slow-imag-256m | complex wavelet coefficients | 256m | 0.000293 / -0.000163 | -0.009953 / -0.016017 | -0.01432, -0.00559, -0.01995, -0.01208 |
| morlet-fast-imag-256m | complex wavelet coefficients | 256m | 0.000412 / -0.001805 | -0.010855 / -0.013892 | -0.01236, -0.00935, -0.01587, -0.01192 |
| frft-0p25-k1-imag-256m | fractional Fourier complex coefficients | 256m | -0.000754 / -0.000070 | -0.011188 / -0.015734 | -0.01357, -0.00881, -0.01457, -0.01690 |
| frft-0p75-k1-real-256m | fractional Fourier complex coefficients | 256m | -0.000774 / -0.000064 | -0.011194 / -0.015712 | -0.01360, -0.00879, -0.01459, -0.01684 |
| fft-k1-real-256m | Fourier complex coefficients | 256m | -0.001202 / -0.000800 | -0.011587 / -0.013187 | -0.00941, -0.01376, -0.01305, -0.01332 |
| frft-0p5-k1-imag-256m | fractional Fourier complex coefficients | 256m | -0.001150 / -0.000788 | -0.011771 / -0.013239 | -0.00949, -0.01405, -0.01330, -0.01318 |
| frft-0p5-entropy-256m | fractional Fourier shape | 256m | 0.002782 / 0.002337 | -0.012375 / -0.008431 | -0.00985, -0.01490, -0.00210, -0.01476 |

## Interpretation limits

- The FFT windows end at the latest completed candle, so no future sample enters a feature.
- Quartile feature cells detect robust distribution changes but can miss smooth information that a neural sequence model could exploit.
- This audit tests fixed-window transform coordinates, not a learned spectral/wavelet layer or arbitrary transform-parameter search.
- Selection is conditional on the previously fixed global basis. Negative transfer gain means the apparent primary-period addition should not be promoted.

Machine-readable results are stored in `data/benchmarks/fourier-return-feature-information.json`.
