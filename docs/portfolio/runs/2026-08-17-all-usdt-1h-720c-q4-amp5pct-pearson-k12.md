# Binance portfolio basis

Generated 2026-08-20T16:37:20.253Z.

## Scope

- Products: all
- Product listings discovered: spot 3681, usdm-futures 872, coinm-futures 30, options 1848
- Listing statuses: TRADING 3983, BREAK 2320, SETTLING 127, PENDING_TRADING 1
- Numeraire: USDT
- Return window: 2026-07-19T00:00Z through 2026-08-17T23:00Z
- Sampling: exactly 720 1h log returns (30 days)
- Correlation: pearson
- Pivot tie-break: among candidates within 5.0% of the maximum unexplained variance, select the largest mean absolute 1h return
- Sizing: fixed at 12 assets
- Full catalog: 6431 listings, 3983 active listings, 1175 deduplicated economic assets
- Return universe: 80 eligible assets from 735 active continuously priced candidates
- Quality filter: complete window, trades or quote volume on at least 50.0% of UTC days, no stale-price run longer than 48 hours
- TradFi session handling: zero-volume off-session candles do not extend stale-price runs
- Stable assets excluded: yes

## Selected basis

The selector uses column-pivoted QR on standardized return vectors. Residual is the fraction of an asset's return-vector norm not explained by earlier basis assets. Orthogonality remains primary; mean absolute candle return only chooses among near-equivalent residual candidates at this scale. Correlations are shown as absolute values because both +1 and -1 are linearly redundant, not orthogonal.

| # | Asset | Canonical market | Products | Residual | Max abs corr to earlier | Closest earlier | Mean abs return | Traded days | Max stale hours | Median daily quote volume | Annualized vol |
| -: | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | BTC | BTCUSDT (spot) | spot, usdm-futures, coinm-futures, options | 100.0% | 0.000 | - | 18.46 bp | 100.0% | 0 | 846.5M | 25.4% |
| 2 | BANK | BANKUSDT (spot) | spot, usdm-futures | 100.0% | 0.000 | BTCUSDT | 285.27 bp | 100.0% | 2 | 22.2M | 480.7% |
| 3 | BEAT | BEATUSDT (usdm-futures) | usdm-futures | 99.9% | 0.035 | BTCUSDT | 283.14 bp | 100.0% | 1 | 99.5M | 401.5% |
| 4 | AKE | AKEUSDT (usdm-futures) | usdm-futures | 99.8% | 0.057 | BTCUSDT | 273.36 bp | 100.0% | 0 | 189.8M | 400.4% |
| 5 | ON | ONUSDT (usdm-futures) | usdm-futures | 99.7% | 0.075 | BANKUSDT | 266.17 bp | 100.0% | 1 | 43.8M | 372.0% |
| 6 | BTW | BTWUSDT (usdm-futures) | usdm-futures | 99.7% | 0.063 | BTCUSDT | 234.59 bp | 100.0% | 1 | 67.1M | 348.7% |
| 7 | BLESS | BLESSUSDT (usdm-futures) | usdm-futures | 99.2% | 0.097 | BTCUSDT | 232.70 bp | 100.0% | 0 | 20.4M | 376.8% |
| 8 | ESPORTS | ESPORTSUSDT (usdm-futures) | usdm-futures | 99.4% | 0.070 | BANKUSDT | 224.14 bp | 100.0% | 2 | 22.7M | 404.9% |
| 9 | US | USUSDT (usdm-futures) | usdm-futures | 99.7% | 0.049 | BLESSUSDT | 202.40 bp | 100.0% | 0 | 23.3M | 337.0% |
| 10 | CAP | CAPUSDT (usdm-futures) | usdm-futures | 99.2% | 0.069 | AKEUSDT | 197.14 bp | 100.0% | 1 | 26M | 261.5% |
| 11 | SKYAI | SKYAIUSDT (usdm-futures) | usdm-futures | 98.0% | 0.109 | BTCUSDT | 190.05 bp | 100.0% | 1 | 14.7M | 273.2% |
| 12 | KOMA | KOMAUSDT (usdm-futures) | usdm-futures | 99.5% | 0.060 | BLESSUSDT | 183.33 bp | 100.0% | 1 | 5.3M | 341.9% |

## Diagnostics

- Basis size selected: 12
- Coverage target reached: yes
- Mean pairwise absolute correlation: 0.033
- Maximum pairwise absolute correlation: 0.109
- Mean whole-market projection R²: 29.7%
- Median whole-market projection R²: 16.5%
- 10th-percentile whole-market projection R²: 2.0%
- Minimum whole-market projection R²: 1.0%

### Coverage by basis size

| Size | Mean R² | Median R² | 10th percentile R² | Minimum R² |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 14.6% | 6.6% | 0.0% | 0.0% |
| 5 | 20.2% | 10.0% | 0.6% | 0.3% |
| 10 | 27.0% | 12.1% | 1.5% | 0.7% |
| 12 | 29.7% | 16.5% | 2.0% | 1.0% |

### Selected-asset correlation matrix

| Asset | BTC | BANK | BEAT | AKE | ON | BTW | BLESS | ESPORTS | US | CAP | SKYAI | KOMA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | 1.000 | -0.000 | -0.035 | -0.057 | -0.003 | 0.063 | 0.097 | 0.048 | -0.024 | 0.040 | 0.109 | 0.019 |
| BANK | -0.000 | 1.000 | 0.020 | 0.011 | 0.075 | -0.015 | -0.008 | -0.070 | 0.037 | -0.065 | -0.001 | 0.031 |
| BEAT | -0.035 | 0.020 | 1.000 | -0.037 | 0.012 | 0.012 | -0.048 | 0.001 | 0.022 | -0.020 | -0.089 | 0.030 |
| AKE | -0.057 | 0.011 | -0.037 | 1.000 | -0.008 | 0.022 | 0.018 | -0.005 | 0.003 | 0.069 | -0.035 | 0.024 |
| ON | -0.003 | 0.075 | 0.012 | -0.008 | 1.000 | 0.031 | -0.049 | 0.061 | 0.020 | 0.006 | 0.027 | 0.018 |
| BTW | 0.063 | -0.015 | 0.012 | 0.022 | 0.031 | 1.000 | -0.024 | 0.000 | -0.025 | 0.025 | 0.079 | 0.013 |
| BLESS | 0.097 | -0.008 | -0.048 | 0.018 | -0.049 | -0.024 | 1.000 | 0.028 | 0.049 | -0.051 | 0.034 | -0.060 |
| ESPORTS | 0.048 | -0.070 | 0.001 | -0.005 | 0.061 | 0.000 | 0.028 | 1.000 | 0.006 | 0.015 | 0.035 | 0.024 |
| US | -0.024 | 0.037 | 0.022 | 0.003 | 0.020 | -0.025 | 0.049 | 0.006 | 1.000 | -0.006 | 0.101 | 0.032 |
| CAP | 0.040 | -0.065 | -0.020 | 0.069 | 0.006 | 0.025 | -0.051 | 0.015 | -0.006 | 1.000 | 0.023 | 0.042 |
| SKYAI | 0.109 | -0.001 | -0.089 | -0.035 | 0.027 | 0.079 | 0.034 | 0.035 | 0.101 | 0.023 | 1.000 | 0.013 |
| KOMA | 0.019 | 0.031 | 0.030 | 0.024 | 0.018 | 0.013 | -0.060 | 0.024 | 0.032 | 0.042 | 0.013 | 1.000 |

### Least-covered eligible assets

A low R² here means the current basis does not explain much of that asset. Increase `--size` if these residuals matter.

| Asset | Symbol | Projection R² | Residual | Closest basis | Correlation |
| --- | --- | ---: | ---: | --- | ---: |
| KAITO | KAITOUSDT | 1.0% | 99.5% | SKYAIUSDT | -0.053 |
| VELVET | VELVETUSDT | 1.0% | 99.5% | CAPUSDT | 0.059 |
| ALLO | ALLOUSDT | 1.1% | 99.5% | BANKUSDT | 0.051 |
| BULLA | BULLAUSDT | 1.1% | 99.4% | BTWUSDT | 0.063 |
| DODOX | DODOXUSDT | 1.1% | 99.4% | BEATUSDT | -0.065 |
| H | HUSDT | 1.2% | 99.4% | BLESSUSDT | 0.053 |
| MINIMAX | MINIMAXUSDT | 1.8% | 99.1% | BTCUSDT | 0.103 |
| NATGAS | NATGASUSDT | 1.9% | 99.1% | BLESSUSDT | -0.068 |
| HOME | HOMEUSDT | 2.0% | 99.0% | BTCUSDT | 0.089 |
| AIO | AIOUSDT | 2.2% | 98.9% | BTCUSDT | 0.111 |
| ZAMA | ZAMAUSDT | 2.5% | 98.7% | ONUSDT | 0.095 |
| U | UUSDT | 2.5% | 98.7% | ONUSDT | -0.087 |
| UB | UBUSDT | 2.6% | 98.7% | BLESSUSDT | 0.105 |
| ZHIPU | ZHIPUUSDT | 2.6% | 98.7% | BTCUSDT | 0.114 |
| GWEI | GWEIUSDT | 2.6% | 98.7% | SKYAIUSDT | 0.110 |

## Interpretation

This is an empirical basis of historical return directions, not a portfolio allocation and not a profitability claim. Selection is sensitive to the window, listings, liquidity, and correlation regime. Validate stability across adjacent windows before using it for capital allocation.

