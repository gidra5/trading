# Binance portfolio basis

Generated 2026-07-23T17:57:14.986Z.

## Scope

- Products: all
- Product listings discovered: spot 3659, usdm-futures 846, coinm-futures 30, options 1550
- Listing statuses: TRADING 3677, BREAK 2283, SETTLING 123, PENDING_TRADING 2
- Numeraire: USDT
- Daily log-return window: 2026-06-23 through 2026-07-22 (30 samples)
- Correlation: pearson
- Sizing: automatic until median R² >= 80.0% and 10th-percentile R² >= 50.0% (maximum 128)
- Full catalog: 6085 listings, 3677 active listings, 1126 deduplicated economic assets
- Return universe: 633 eligible assets from 715 active continuously priced candidates
- Quality filter: at least 80.0% non-zero daily returns and a complete window
- Stable assets excluded: yes

## Selected basis

The selector uses column-pivoted QR on standardized return vectors. Residual is the fraction of an asset's return-vector norm not explained by earlier basis assets. Correlations are shown as absolute values because both +1 and -1 are linearly redundant, not orthogonal.

| # | Asset | Canonical market | Products | Residual | Max |r| to earlier | Closest earlier | Median daily quote volume | Annualized vol |
| -: | --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| 1 | BTC | BTCUSDT (spot) | spot, usdm-futures, coinm-futures, options | 100.0% | 0.000 | - | 1.2B | 32.3% |
| 2 | UBER | UBERUSDT (usdm-futures) | usdm-futures | 100.0% | 0.000 | BTCUSDT | 330.4K | 38.1% |
| 3 | DODOX | DODOXUSDT (usdm-futures) | usdm-futures | 100.0% | 0.008 | UBERUSDT | 3.9M | 199.8% |
| 4 | XPIN | XPINUSDT (usdm-futures) | usdm-futures | 99.9% | 0.024 | DODOXUSDT | 2.9M | 124.6% |
| 5 | JCT | JCTUSDT (usdm-futures) | usdm-futures | 99.3% | 0.095 | UBERUSDT | 3.2M | 198.9% |
| 6 | SKR | SKRUSDT (usdm-futures) | usdm-futures | 99.2% | 0.083 | XPINUSDT | 1.7M | 68.5% |
| 7 | AKE | AKEUSDT (usdm-futures) | usdm-futures | 99.0% | 0.101 | DODOXUSDT | 3.6M | 537.2% |
| 8 | ACE | ACEUSDT (spot) | spot, usdm-futures | 98.4% | 0.109 | AKEUSDT | 318.9K | 176.1% |
| 9 | EDGE | EDGEUSDT (usdm-futures) | usdm-futures | 97.8% | 0.131 | XPINUSDT | 13.1M | 169.9% |
| 10 | HUMA | HUMAUSDT (spot) | spot, usdm-futures | 97.7% | 0.151 | UBERUSDT | 499.3K | 99.8% |
| 11 | AIGENSYN | AIGENSYNUSDT (spot) | spot, usdm-futures | 96.5% | 0.219 | UBERUSDT | 1.5M | 143.0% |
| 12 | MANA | MANAUSDT (spot) | spot, usdm-futures | 94.3% | 0.241 | BTCUSDT | 370.9K | 66.9% |
| 13 | PIEVERSE | PIEVERSEUSDT (usdm-futures) | usdm-futures | 92.5% | 0.189 | BTCUSDT | 2.9M | 99.7% |
| 14 | SKYAI | SKYAIUSDT (usdm-futures) | usdm-futures | 91.2% | 0.175 | EDGEUSDT | 22.7M | 329.9% |
| 15 | T | TUSDT (spot) | spot, usdm-futures | 88.6% | 0.264 | DODOXUSDT | 468.8K | 148.0% |
| 16 | TRUTH | TRUTHUSDT (usdm-futures) | usdm-futures | 87.8% | 0.281 | AIGENSYNUSDT | 1.9M | 116.3% |
| 17 | FLUID | FLUIDUSDT (usdm-futures) | usdm-futures | 85.8% | 0.299 | BTCUSDT | 649.4K | 110.1% |
| 18 | BAN | BANUSDT (usdm-futures) | usdm-futures | 81.8% | 0.377 | XPINUSDT | 2.2M | 51.7% |
| 19 | UAI | UAIUSDT (usdm-futures) | usdm-futures | 78.4% | 0.279 | SKYAIUSDT | 4.4M | 145.6% |
| 20 | SAPIEN | SAPIENUSDT (spot) | spot, usdm-futures | 77.5% | 0.317 | ACEUSDT | 330.2K | 73.6% |

## Diagnostics

- Basis size selected: 20
- Coverage target reached: yes
- Mean pairwise absolute correlation: 0.088
- Maximum pairwise absolute correlation: 0.377
- Mean whole-market projection R²: 79.4%
- Median whole-market projection R²: 81.1%
- 10th-percentile whole-market projection R²: 63.2%
- Minimum whole-market projection R²: 40.2%

### Coverage by basis size

| Size | Mean R² | Median R² | 10th percentile R² | Minimum R² |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 22.6% | 17.9% | 0.6% | 0.0% |
| 5 | 33.8% | 32.1% | 9.7% | 1.6% |
| 10 | 50.6% | 50.1% | 27.5% | 7.0% |
| 15 | 64.4% | 65.1% | 42.4% | 22.8% |
| 20 | 79.4% | 81.1% | 63.2% | 40.2% |

### Selected-asset correlation matrix

| Asset | BTC | UBER | DODOX | XPIN | JCT | SKR | AKE | ACE | EDGE | HUMA | AIGENSYN | MANA | PIEVERSE | SKYAI | T | TRUTH | FLUID | BAN | UAI | SAPIEN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | 1.000 | 0.000 | -0.005 | -0.016 | -0.016 | -0.074 | -0.054 | 0.096 | -0.041 | 0.053 | -0.027 | 0.241 | 0.189 | 0.113 | 0.024 | -0.046 | 0.299 | 0.114 | -0.090 | 0.129 |
| UBER | 0.000 | 1.000 | -0.008 | -0.020 | -0.095 | 0.030 | 0.034 | 0.055 | -0.082 | -0.151 | -0.219 | -0.076 | -0.015 | -0.028 | 0.104 | 0.019 | 0.046 | 0.116 | -0.126 | -0.215 |
| DODOX | -0.005 | -0.008 | 1.000 | 0.024 | 0.048 | 0.043 | 0.101 | -0.104 | 0.106 | -0.056 | 0.051 | -0.074 | -0.069 | -0.127 | 0.264 | 0.144 | 0.015 | -0.271 | -0.121 | 0.083 |
| XPIN | -0.016 | -0.020 | 0.024 | 1.000 | -0.042 | -0.083 | 0.017 | 0.012 | -0.131 | 0.057 | 0.097 | -0.023 | 0.025 | -0.010 | 0.202 | 0.023 | -0.023 | -0.377 | 0.121 | 0.080 |
| JCT | -0.016 | -0.095 | 0.048 | -0.042 | 1.000 | 0.019 | 0.067 | -0.005 | -0.031 | -0.073 | 0.004 | 0.106 | -0.112 | -0.140 | -0.075 | -0.160 | -0.216 | 0.168 | 0.061 | 0.109 |
| SKR | -0.074 | 0.030 | 0.043 | -0.083 | 0.019 | 1.000 | 0.018 | -0.019 | 0.009 | -0.007 | -0.049 | 0.069 | 0.188 | -0.005 | -0.044 | 0.032 | -0.152 | 0.015 | 0.223 | -0.124 |
| AKE | -0.054 | 0.034 | 0.101 | 0.017 | 0.067 | 0.018 | 1.000 | -0.109 | -0.052 | -0.049 | 0.034 | 0.043 | 0.018 | 0.172 | -0.064 | 0.197 | -0.055 | -0.071 | -0.187 | -0.021 |
| ACE | 0.096 | 0.055 | -0.104 | 0.012 | -0.005 | -0.019 | -0.109 | 1.000 | -0.019 | -0.074 | -0.001 | 0.112 | 0.058 | 0.162 | -0.080 | -0.063 | 0.088 | 0.031 | -0.236 | 0.317 |
| EDGE | -0.041 | -0.082 | 0.106 | -0.131 | -0.031 | 0.009 | -0.052 | -0.019 | 1.000 | 0.008 | -0.044 | 0.006 | 0.106 | -0.175 | -0.163 | -0.145 | -0.116 | 0.075 | 0.171 | 0.036 |
| HUMA | 0.053 | -0.151 | -0.056 | 0.057 | -0.073 | -0.007 | -0.049 | -0.074 | 0.008 | 1.000 | -0.026 | 0.008 | -0.115 | 0.003 | -0.053 | -0.168 | -0.071 | -0.044 | -0.002 | 0.033 |
| AIGENSYN | -0.027 | -0.219 | 0.051 | 0.097 | 0.004 | -0.049 | 0.034 | -0.001 | -0.044 | -0.026 | 1.000 | -0.092 | -0.066 | -0.031 | -0.114 | 0.281 | -0.074 | -0.070 | 0.028 | 0.009 |
| MANA | 0.241 | -0.076 | -0.074 | -0.023 | 0.106 | 0.069 | 0.043 | 0.112 | 0.006 | 0.008 | -0.092 | 1.000 | 0.138 | 0.152 | 0.107 | -0.117 | 0.229 | 0.268 | -0.005 | -0.011 |
| PIEVERSE | 0.189 | -0.015 | -0.069 | 0.025 | -0.112 | 0.188 | 0.018 | 0.058 | 0.106 | -0.115 | -0.066 | 0.138 | 1.000 | -0.052 | -0.020 | -0.016 | 0.023 | -0.002 | 0.126 | -0.036 |
| SKYAI | 0.113 | -0.028 | -0.127 | -0.010 | -0.140 | -0.005 | 0.172 | 0.162 | -0.175 | 0.003 | -0.031 | 0.152 | -0.052 | 1.000 | -0.036 | 0.153 | 0.095 | -0.020 | -0.279 | 0.134 |
| T | 0.024 | 0.104 | 0.264 | 0.202 | -0.075 | -0.044 | -0.064 | -0.080 | -0.163 | -0.053 | -0.114 | 0.107 | -0.020 | -0.036 | 1.000 | 0.008 | 0.072 | -0.079 | -0.226 | 0.054 |
| TRUTH | -0.046 | 0.019 | 0.144 | 0.023 | -0.160 | 0.032 | 0.197 | -0.063 | -0.145 | -0.168 | 0.281 | -0.117 | -0.016 | 0.153 | 0.008 | 1.000 | 0.180 | -0.070 | -0.080 | -0.293 |
| FLUID | 0.299 | 0.046 | 0.015 | -0.023 | -0.216 | -0.152 | -0.055 | 0.088 | -0.116 | -0.071 | -0.074 | 0.229 | 0.023 | 0.095 | 0.072 | 0.180 | 1.000 | 0.068 | 0.078 | -0.123 |
| BAN | 0.114 | 0.116 | -0.271 | -0.377 | 0.168 | 0.015 | -0.071 | 0.031 | 0.075 | -0.044 | -0.070 | 0.268 | -0.002 | -0.020 | -0.079 | -0.070 | 0.068 | 1.000 | 0.040 | 0.118 |
| UAI | -0.090 | -0.126 | -0.121 | 0.121 | 0.061 | 0.223 | -0.187 | -0.236 | 0.171 | -0.002 | 0.028 | -0.005 | 0.126 | -0.279 | -0.226 | -0.080 | 0.078 | 0.040 | 1.000 | -0.191 |
| SAPIEN | 0.129 | -0.215 | 0.083 | 0.080 | 0.109 | -0.124 | -0.021 | 0.317 | 0.036 | 0.033 | 0.009 | -0.011 | -0.036 | 0.134 | 0.054 | -0.293 | -0.123 | 0.118 | -0.191 | 1.000 |

### Least-covered eligible assets

A low R² here means the current basis does not explain much of that asset. Increase `--size` if these residuals matter.

| Asset | Symbol | Projection R² | Residual | Closest basis | Correlation |
| --- | --- | ---: | ---: | --- | ---: |
| DEXE | DEXEUSDT | 40.2% | 77.3% | ACEUSDT | 0.427 |
| STXX | STXXUSDT | 43.8% | 75.0% | UBERUSDT | -0.317 |
| VANRY | VANRYUSDT | 46.1% | 73.4% | JCTUSDT | -0.295 |
| SLX | SLXUSDT | 47.5% | 72.5% | BTCUSDT | -0.377 |
| TREE | TREEUSDT | 48.5% | 71.8% | BTCUSDT | 0.449 |
| SMCI | SMCIUSDT | 48.8% | 71.5% | ACEUSDT | -0.327 |
| CRM | CRMUSDT | 49.6% | 71.0% | BANUSDT | 0.406 |
| SENT | SENTUSDT | 50.1% | 70.6% | XPINUSDT | -0.409 |
| PYR | PYRUSDT | 50.7% | 70.2% | XPINUSDT | 0.454 |
| PROMPT | PROMPTUSDT | 50.9% | 70.1% | BTCUSDT | 0.454 |
| ERA | ERAUSDT | 51.5% | 69.6% | BTCUSDT | 0.422 |
| COHR | COHRUSDT | 51.7% | 69.5% | TRUTHUSDT | 0.345 |
| OPEN | OPENUSDT | 51.8% | 69.5% | TRUTHUSDT | 0.492 |
| COST | COSTUSDT | 51.8% | 69.4% | XPINUSDT | -0.269 |
| SAHARA | SAHARAUSDT | 52.0% | 69.3% | BTCUSDT | 0.414 |

## Interpretation

This is an empirical basis of historical return directions, not a portfolio allocation and not a profitability claim. Selection is sensitive to the window, listings, liquidity, and correlation regime. Validate stability across adjacent windows before using it for capital allocation.

