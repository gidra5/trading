# Binance portfolio basis

Generated 2026-07-23T16:11:28.148Z.

## Scope

- Products: all
- Product listings discovered: spot 1376, usdm-futures 846, coinm-futures 30, options 10
- Numeraire: USDT
- Daily log-return window: 2025-07-23 through 2026-07-22 (365 samples)
- Correlation: pearson
- Universe: 412 eligible assets from 715 active candidate symbols
- Quality filter: at least 80.0% non-zero daily returns and a complete window
- Stable assets excluded: yes

## Selected basis

The selector uses column-pivoted QR on standardized return vectors. Residual is the fraction of an asset's return-vector norm not explained by earlier basis assets. Correlations are shown as absolute values because both +1 and -1 are linearly redundant, not orthogonal.

| # | Asset | Canonical market | Products | Residual | Max |r| to earlier | Closest earlier | Median daily quote volume | Annualized vol |
| -: | --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| 1 | BTC | BTCUSDT (spot) | spot, usdm-futures, coinm-futures, options | 100.0% | 0.000 | - | 1.4B | 43.2% |
| 2 | VELVET | VELVETUSDT (usdm-futures) | usdm-futures | 100.0% | 0.006 | BTCUSDT | 3.2M | 276.6% |
| 3 | BULLA | BULLAUSDT (usdm-futures) | usdm-futures | 99.9% | 0.027 | VELVETUSDT | 2.7M | 384.7% |
| 4 | JELLYJELLY | JELLYJELLYUSDT (usdm-futures) | usdm-futures | 99.8% | 0.041 | BTCUSDT | 7.5M | 258.8% |
| 5 | SIREN | SIRENUSDT (usdm-futures) | usdm-futures | 99.7% | 0.068 | VELVETUSDT | 6.4M | 413.1% |
| 6 | JST | JSTUSDT (spot) | spot, usdm-futures | 99.5% | 0.091 | BTCUSDT | 2.2M | 61.1% |
| 7 | ALCH | ALCHUSDT (usdm-futures) | usdm-futures | 99.1% | 0.091 | BTCUSDT | 3.7M | 151.1% |
| 8 | TA | TAUSDT (usdm-futures) | usdm-futures | 98.9% | 0.110 | BTCUSDT | 4.6M | 202.7% |
| 9 | M | MUSDT (usdm-futures) | usdm-futures | 98.7% | 0.126 | SIRENUSDT | 11.6M | 196.4% |
| 10 | BR | BRUSDT (usdm-futures) | usdm-futures | 98.2% | 0.121 | BTCUSDT | 1.5M | 191.2% |
| 11 | OG | OGUSDT (spot) | spot, usdm-futures | 97.9% | 0.184 | BTCUSDT | 2.1M | 131.0% |
| 12 | FHE | FHEUSDT (usdm-futures) | usdm-futures | 97.7% | 0.151 | BTCUSDT | 4.8M | 253.3% |

## Diagnostics

- Mean pairwise absolute correlation: 0.044
- Maximum pairwise absolute correlation: 0.184
- Mean whole-market projection R²: 37.1%
- Median whole-market projection R²: 35.3%
- 10th-percentile whole-market projection R²: 14.7%
- Minimum whole-market projection R²: 4.8%

### Selected-asset correlation matrix

| Asset | BTC | VELVET | BULLA | JELLYJELLY | SIREN | JST | ALCH | TA | M | BR | OG | FHE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | 1.000 | 0.006 | 0.023 | 0.041 | 0.017 | 0.091 | 0.091 | 0.110 | 0.057 | 0.121 | 0.184 | 0.151 |
| VELVET | 0.006 | 1.000 | 0.027 | -0.036 | 0.068 | 0.018 | -0.054 | 0.033 | -0.048 | 0.029 | -0.060 | 0.060 |
| BULLA | 0.023 | 0.027 | 1.000 | 0.017 | 0.013 | 0.032 | -0.022 | 0.042 | -0.014 | 0.021 | -0.002 | 0.053 |
| JELLYJELLY | 0.041 | -0.036 | 0.017 | 1.000 | -0.001 | 0.009 | 0.029 | -0.061 | 0.048 | 0.037 | 0.003 | 0.005 |
| SIREN | 0.017 | 0.068 | 0.013 | -0.001 | 1.000 | -0.005 | -0.022 | -0.021 | 0.126 | 0.031 | 0.002 | -0.012 |
| JST | 0.091 | 0.018 | 0.032 | 0.009 | -0.005 | 1.000 | -0.067 | 0.063 | 0.033 | -0.031 | 0.011 | -0.024 |
| ALCH | 0.091 | -0.054 | -0.022 | 0.029 | -0.022 | -0.067 | 1.000 | 0.018 | 0.026 | 0.118 | 0.058 | 0.054 |
| TA | 0.110 | 0.033 | 0.042 | -0.061 | -0.021 | 0.063 | 0.018 | 1.000 | 0.021 | 0.080 | 0.038 | 0.106 |
| M | 0.057 | -0.048 | -0.014 | 0.048 | 0.126 | 0.033 | 0.026 | 0.021 | 1.000 | -0.008 | 0.026 | 0.007 |
| BR | 0.121 | 0.029 | 0.021 | 0.037 | 0.031 | -0.031 | 0.118 | 0.080 | -0.008 | 1.000 | 0.070 | -0.036 |
| OG | 0.184 | -0.060 | -0.002 | 0.003 | 0.002 | 0.011 | 0.058 | 0.038 | 0.026 | 0.070 | 1.000 | 0.043 |
| FHE | 0.151 | 0.060 | 0.053 | 0.005 | -0.012 | -0.024 | 0.054 | 0.106 | 0.007 | -0.036 | 0.043 | 1.000 |

### Least-covered eligible assets

A low R² here means the current basis does not explain much of that asset. Increase `--size` if these residuals matter.

| Asset | Symbol | Projection R² | Residual | Closest basis | Correlation |
| --- | --- | ---: | ---: | --- | ---: |
| TNSR | TNSRUSDT | 4.8% | 97.6% | BTCUSDT | 0.189 |
| PIPPIN | PIPPINUSDT | 4.9% | 97.5% | BTCUSDT | 0.151 |
| DEXE | DEXEUSDT | 5.0% | 97.4% | BTCUSDT | 0.167 |
| B2 | B2USDT | 5.1% | 97.4% | BTCUSDT | 0.206 |
| AGT | AGTUSDT | 5.1% | 97.4% | TAUSDT | 0.123 |
| ARC | ARCUSDT | 5.2% | 97.4% | TAUSDT | 0.133 |
| ZEREBRO | ZEREBROUSDT | 5.4% | 97.2% | BTCUSDT | 0.155 |
| H | HUSDT | 5.6% | 97.2% | BTCUSDT | 0.153 |
| STO | STOUSDT | 5.7% | 97.1% | BTCUSDT | 0.187 |
| IDOL | IDOLUSDT | 5.8% | 97.1% | BTCUSDT | 0.140 |
| BAN | BANUSDT | 6.0% | 96.9% | BTCUSDT | 0.184 |
| TAC | TACUSDT | 6.0% | 96.9% | BTCUSDT | 0.137 |
| ALPINE | ALPINEUSDT | 6.2% | 96.8% | BTCUSDT | 0.160 |
| AIN | AINUSDT | 6.3% | 96.8% | BTCUSDT | 0.155 |
| HOME | HOMEUSDT | 6.5% | 96.7% | BTCUSDT | 0.163 |

## Interpretation

This is an empirical basis of historical return directions, not a portfolio allocation and not a profitability claim. Selection is sensitive to the window, listings, liquidity, and correlation regime. Validate stability across adjacent windows before using it for capital allocation.

