# Derivative Threshold Grid Benchmark

Generated: 2026-07-05 19:02
Raw results: `data/benchmarks/derivative-threshold-grid-2026-07-05-180107.jsonl`

## Scope

- Market: BTCUSDT 1m spot candles, UTC day-inclusive intervals from `tasks.md`.
- Strategy: static sigma `buySigma=0.1`, `sellSigma=0.1`, mode both, futures-margin shorts, borrow depths `999/999`, max leverage `1x`.
- Sources: `kama`.
- Threshold multipliers: `1, 2, 4`; default 60s threshold is `0.25`, so `4x` makes it `1.0`.
- Hysteresis inner ratios: `0, 0.25, 0.5, 0.75, 1`; ratio `0` is zero-cross exit, ratio `1` exits at the outer threshold.
- KAMA: `erLen=20`, `fastLen=20`, `slowLen=200`, `power=1`.

## Top Combinations

| Source | Outer | Mode | Inner | Avg return | Vs baseline | Median | Positive | Avg DD | Max DD | Avg trades |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `kama` | 4x | `hysteresis` | 0.00 | +0.292% | +0.455% | +0.000% | 9/28 | 0.598% | 5.714% | 10.2 |
| `kama` | 2x | `deadband` |  | +0.247% | +0.410% | +0.000% | 11/28 | 0.492% | 5.833% | 14.5 |
| `kama` | 2x | `hysteresis` | 1.00 | +0.247% | +0.410% | +0.000% | 11/28 | 0.492% | 5.833% | 14.5 |
| `kama` | 2x | `hysteresis` | 0.50 | -0.005% | +0.158% | +0.000% | 14/28 | 0.853% | 8.634% | 39.2 |
| `kama` | 4x | `deadband` |  | -0.087% | +0.076% | +0.000% | 2/28 | 0.096% | 2.495% | 1.1 |
| `kama` | 4x | `hysteresis` | 1.00 | -0.087% | +0.076% | +0.000% | 2/28 | 0.096% | 2.495% | 1.1 |
| `kama` | 4x | `hysteresis` | 0.25 | -0.096% | +0.067% | +0.000% | 11/28 | 0.244% | 3.559% | 11.4 |
| `kama` | 2x | `hysteresis` | 0.75 | -0.126% | +0.037% | +0.000% | 14/28 | 0.943% | 9.627% | 21.3 |
| `kama` | 4x | `hysteresis` | 0.75 | -0.143% | +0.020% | +0.000% | 3/28 | 0.209% | 3.217% | 1.6 |
| `kama` | 1x | `deadband` |  | -0.163% | +0.000% | +0.117% | 17/28 | 1.960% | 15.333% | 106.6 |
| `kama` | 1x | `hysteresis` | 1.00 | -0.163% | +0.000% | +0.117% | 17/28 | 1.960% | 15.333% | 106.6 |
| `kama` | 4x | `hysteresis` | 0.50 | -0.230% | -0.067% | +0.000% | 6/28 | 0.309% | 5.774% | 3.7 |
| `kama` | 2x | `hysteresis` | 0.25 | -0.483% | -0.320% | +0.000% | 13/28 | 1.781% | 20.083% | 65.4 |
| `kama` | 1x | `hysteresis` | 0.75 | -0.664% | -0.501% | +0.002% | 15/28 | 2.225% | 16.434% | 150.4 |
| `kama` | 2x | `hysteresis` | 0.00 | -1.526% | -1.363% | -0.002% | 11/28 | 3.432% | 24.151% | 76 |
| `kama` | 1x | `hysteresis` | 0.50 | -1.909% | -1.746% | -0.169% | 14/28 | 3.949% | 20.505% | 238 |
| `kama` | 1x | `hysteresis` | 0.25 | -4.177% | -4.014% | -2.446% | 8/28 | 5.840% | 22.961% | 309.8 |
| `kama` | 1x | `hysteresis` | 0.00 | -6.256% | -6.093% | -5.210% | 5/28 | 7.979% | 22.353% | 272.8 |

## All Combinations

| Source | Outer | Mode | Inner | Avg return | Vs baseline | Median | Positive | Avg DD | Max DD | Avg trades |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `kama` | 4x | `hysteresis` | 0.00 | +0.292% | +0.455% | +0.000% | 9/28 | 0.598% | 5.714% | 10.2 |
| `kama` | 2x | `deadband` |  | +0.247% | +0.410% | +0.000% | 11/28 | 0.492% | 5.833% | 14.5 |
| `kama` | 2x | `hysteresis` | 1.00 | +0.247% | +0.410% | +0.000% | 11/28 | 0.492% | 5.833% | 14.5 |
| `kama` | 2x | `hysteresis` | 0.50 | -0.005% | +0.158% | +0.000% | 14/28 | 0.853% | 8.634% | 39.2 |
| `kama` | 4x | `deadband` |  | -0.087% | +0.076% | +0.000% | 2/28 | 0.096% | 2.495% | 1.1 |
| `kama` | 4x | `hysteresis` | 1.00 | -0.087% | +0.076% | +0.000% | 2/28 | 0.096% | 2.495% | 1.1 |
| `kama` | 4x | `hysteresis` | 0.25 | -0.096% | +0.067% | +0.000% | 11/28 | 0.244% | 3.559% | 11.4 |
| `kama` | 2x | `hysteresis` | 0.75 | -0.126% | +0.037% | +0.000% | 14/28 | 0.943% | 9.627% | 21.3 |
| `kama` | 4x | `hysteresis` | 0.75 | -0.143% | +0.020% | +0.000% | 3/28 | 0.209% | 3.217% | 1.6 |
| `kama` | 1x | `deadband` |  | -0.163% | +0.000% | +0.117% | 17/28 | 1.960% | 15.333% | 106.6 |
| `kama` | 1x | `hysteresis` | 1.00 | -0.163% | +0.000% | +0.117% | 17/28 | 1.960% | 15.333% | 106.6 |
| `kama` | 4x | `hysteresis` | 0.50 | -0.230% | -0.067% | +0.000% | 6/28 | 0.309% | 5.774% | 3.7 |
| `kama` | 2x | `hysteresis` | 0.25 | -0.483% | -0.320% | +0.000% | 13/28 | 1.781% | 20.083% | 65.4 |
| `kama` | 1x | `hysteresis` | 0.75 | -0.664% | -0.501% | +0.002% | 15/28 | 2.225% | 16.434% | 150.4 |
| `kama` | 2x | `hysteresis` | 0.00 | -1.526% | -1.363% | -0.002% | 11/28 | 3.432% | 24.151% | 76 |
| `kama` | 1x | `hysteresis` | 0.50 | -1.909% | -1.746% | -0.169% | 14/28 | 3.949% | 20.505% | 238 |
| `kama` | 1x | `hysteresis` | 0.25 | -4.177% | -4.014% | -2.446% | 8/28 | 5.840% | 22.961% | 309.8 |
| `kama` | 1x | `hysteresis` | 0.00 | -6.256% | -6.093% | -5.210% | 5/28 | 7.979% | 22.353% | 272.8 |

