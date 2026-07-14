# Volume-weighted efficiency-ratio search

The added causal ER parameterization is:

`weight_i = (volume_i / EMA(volume)_i) ^ efficiencyVolumePower`

`ER = abs(sum(weight_i * priceChange_i)) / sum(abs(weight_i * priceChange_i))`

The EMA at candle `i` includes that completed candle. Power zero makes every weight one and
recovers standard ER exactly. The EMA duration and power are searched independently from the
existing post-ER volume adjustment.

## Standard-ER worst-window baseline

The two-restart, 2,048-member, 64-generation deep run on
`shape-flat-low-bias-low-2025-07` at 1 second completed in 2h 6m 50s. It improved the
previous shallow window optimum from 46.36% to 55.12% (+8.76 points).

| metric | result |
|---|---:|
| score | 55.12% |
| timing F1 | 42.52% |
| exposure agreement | 75.69% |
| cleanliness | 68.97% |
| candidate signals | 29 |
| matched / extra | 20 / 9 |

The selected configuration used `hold`, a static 37.795 bps/hour threshold, confidence
agreement, nearly full side maxima, finite Gaussian widths, and a 95.18% confirmation mix
with a 95% quality floor. Its confirmation was dominated by price-overextension and RSI;
DMI was exactly disabled.

## Causal ablations

Each row changes only the named part of the selected configuration and re-evaluates the
same 518,400 scored one-second candles and oracle path.

| variant | score | delta | signals | interpretation |
|---|---:|---:|---:|---|
| selected baseline | 55.1157% | — | 29 | reference |
| full confidence | 55.6283% | +0.5125 | 29 | fractional Gaussian strength was harmful |
| full sizing | 55.5935% | +0.4778 | 29 | effectively equivalent at full exposure |
| no EMA confirmation | 52.7304% | -2.3854 | 29 | retain EMA contribution/gate |
| no RSI confirmation | 10.0000% | -45.1157 | 0 | retain RSI; quality floor depends on it |
| no acceleration | 55.1100% | -0.0058 | 29 | remove acceleration dimension |
| no overextension | 52.8951% | -2.2206 | 25 | retain distance/overextension |
| no confirmation | 52.8956% | -2.2202 | 25 | confirmation is useful on this window |
| no post-ER volume adjustment | 55.1144% | -0.0014 | 29 | remove old volume adjustment in the next deep run |
| flat state handling | 21.4953% | -33.6205 | 68 | retain hold |
| hysteresis state handling | 21.0592% | -34.0566 | 68 | retain hold |

Combining full exposure, sizing-independent confidence, zero acceleration, zero DMI, zero
post-ER volume adjustment, static thresholding, and `hold` scores 55.5887% before any new
optimization. This is 0.4730 points above the selected baseline, so it is the lower bound
for the pruned ER-volume deep run.

## Search policy

The next worst-window deep search fixes only dimensions supported as inactive by both the
ablation and prior 1-second window results:

- `hold` state handling and static thresholding;
- full signal strength with confidence agreement;
- no DMI, acceleration, hysteresis-release, adaptive-noise, or post-ER volume dimensions.

It continues searching KAMA durations and power, the base threshold, confirmation mix and
quality floor, overextension, EMA, RSI, and the two new ER-volume parameters. The later
global 1-second search remains fully parameterized, including all state, threshold,
agreement, sizing, volume, and confirmation alternatives.
