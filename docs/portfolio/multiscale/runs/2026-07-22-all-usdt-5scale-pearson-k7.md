# Binance multiscale portfolio basis

Generated 2026-07-23T18:38:58.816Z.

## Scope and method

- Scale views: 365d × 1d, 30d × 4h, 14d × 1h, 7d × 15m, 1d × 1m
- Union universe: 700 economic assets
- Selected basis: 7 assets
- Compression: 99.0%
- Coverage target reached: no
- Construction: equal-weight direct sum of the five standardized correlation views, followed by column-pivoted QR
- Missing histories contribute no scale block, so persistent multi-horizon assets receive more selection norm than one-scale-only assets

## Coverage by scale

| Scale | Samples | Eligible | Individual basis | Joint basis available | Median R² | P10 R² | Unselected median | Unselected P10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 365d × 1d | 365 | 406 | 117 | 5 | NaN% | NaN% | NaN% | NaN% |
| 30d × 4h | 180 | 638 | 115 | 7 | NaN% | NaN% | NaN% | NaN% |
| 14d × 1h | 336 | 671 | 210 | 7 | NaN% | NaN% | NaN% | NaN% |
| 7d × 15m | 672 | 691 | 321 | 7 | NaN% | NaN% | NaN% | NaN% |
| 1d × 1m | 1440 | 700 | 506 | 7 | NaN% | NaN% | NaN% | NaN% |

## Joint diagnostics

- Joint median R²: NaN%
- Joint lower-decile R²: NaN%
- Unselected joint median R²: NaN%
- Unselected joint lower-decile R²: NaN%
- Mean pairwise absolute joint correlation: 0.054
- Maximum pairwise absolute joint correlation: 0.182

## Selected assets

| # | Asset | Residual | Available scales | Selected by individual scales |
| -: | --- | ---: | --- | --- |
| 1 | BTC | 100.0% | structural, swing, adaptive, intraday, microstructure | structural, swing, adaptive, intraday, microstructure |
| 2 | ATM | 99.8% | structural, swing, adaptive, intraday, microstructure | structural, swing, adaptive, intraday, microstructure |
| 3 | AGT | 99.8% | structural, swing, adaptive, intraday, microstructure | structural, adaptive, intraday, microstructure |
| 4 | AERGO | 97.7% | structural, swing, adaptive, intraday, microstructure | structural, adaptive, intraday, microstructure |
| 5 | ACE | 95.8% | structural, swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 6 | AAPL | 99.4% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 7 | 0G | 96.5% | swing, adaptive, intraday, microstructure | intraday, microstructure |

## Joint coverage curve

| Size | Mean R² | Median R² | P10 R² | Minimum R² |
| ---: | ---: | ---: | ---: | ---: |
| 1 | NaN% | NaN% | 8.2% | NaN% |
| 7 | NaN% | NaN% | NaN% | NaN% |

## Least-covered assets

| Asset | Joint R² | Available scales |
| --- | ---: | --- |
| 0G | 100.0% | swing, adaptive, intraday, microstructure |
| 1INCH | NaN% | structural, swing, adaptive, intraday, microstructure |
| 1MBABYDOGE | NaN% | structural, swing, adaptive, intraday, microstructure |
| 2Z | NaN% | swing, adaptive, intraday, microstructure |
| 4 | NaN% | swing, adaptive, intraday, microstructure |
| A | NaN% | structural, adaptive, intraday, microstructure |
| AAOI | NaN% | intraday, microstructure |
| AAPL | NaN% | swing, adaptive, intraday, microstructure |
| AAVE | NaN% | structural, swing, adaptive, intraday, microstructure |
| ACE | NaN% | structural, swing, adaptive, intraday, microstructure |
| ACH | NaN% | structural, swing, adaptive, intraday, microstructure |
| ACM | NaN% | structural, swing, adaptive, intraday, microstructure |
| ACT | NaN% | structural, swing, adaptive, intraday, microstructure |
| ACU | NaN% | swing, adaptive, intraday, microstructure |
| ACX | NaN% | structural, swing, adaptive, intraday, microstructure |
| ADA | NaN% | structural, swing, adaptive, intraday, microstructure |
| ADBE | NaN% | swing, adaptive, intraday, microstructure |
| ADX | NaN% | structural, swing, adaptive, intraday, microstructure |
| AERGO | NaN% | structural, swing, adaptive, intraday, microstructure |
| AERO | NaN% | microstructure |
