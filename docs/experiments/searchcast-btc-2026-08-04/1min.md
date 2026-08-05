# SearchCast-style BTCUSDT analysis: 1m

> **Result.** At the one-candle horizon, the selected context is **128 1m candles**, using local standard normalization, alpha **1.000e-06**, and none augmentation. Held-out normalized MSE changes by **-4.35%** relative to persistence. Evidence quality at this scale is **usable**.

## Scope and adaptation

This report applies the transferable analysis from [How Good Can Linear Models Be for Time-Series Forecasting?](https://arxiv.org/html/2606.27282v1) to canonical spot BTCUSDT **log close** at 1m resolution. The model is multi-output Ridge regression. Search covers context length, global versus trailing-window normalization, standard versus robust scaling, time/frequency/no augmentation, and the paper's 21-value alpha grid. Candidate preprocessors are scored with chronological expanding-window validation; the final 20% of time is sealed until evaluation.

BTC close is a single target, so cross-series grouping is not applicable. The paper's forecast-horizon grouping question is represented here by independent tuning at several horizon cutoffs and the fitted relation `L* = a H^b`.

## Dataset statistics

| Statistic | Value |
|---|---:|
| Candles analyzed | 2,629,440 |
| Period | 2021-07-25T00:00:00Z to 2026-07-24T23:59:00Z |
| Source rows read | 2,629,440 / 2,629,440 (100.00%) |
| Price range | $15,476.00 to $126,199.63 |
| Mean / median log return | +0.0024 / +0.0000 bps |
| Return standard deviation | 7.474 bps |
| Mean absolute return | 4.510 bps |
| Annualized volatility | 54.20% |
| Skewness / excess kurtosis | -0.243 / +109.448 |
| Mean high-low range | 7.670 bps |
| Median candle volume | 15.8884 BTC |

All canonical one-minute rows in the common continuous window.

![1m BTC data profile](charts/1min-data-profile.png)

Return autocorrelation measures linear predictability; absolute-return autocorrelation measures volatility clustering. The latter can be persistent even when signed returns are close to serially uncorrelated.

## Held-out forecasting results

All MSE values are divided by the development-window log-price variance. RMSE and MAE are log-price errors in basis points.

| H (candles) | Test windows | Persistence MSE | Global Ridge MSE | Local-full Ridge MSE | Tuned Ridge MSE | Tuned RMSE | Direction | Gain vs persistence |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1800 | 1.18672e-06 | 1.30331e-06 | 1.2241e-06 | 1.23829e-06 | 6.07 bps | 49.1% | -4.35% |
| 5 | 1800 | 3.32152e-06 | 3.52235e-06 | 3.38981e-06 | 3.51397e-06 | 10.23 bps | 49.7% | -5.79% |
| 15 | 1800 | 8.79854e-06 | 8.97118e-06 | 8.94843e-06 | 9.00715e-06 | 16.38 bps | 49.3% | -2.37% |
| 30 | 1800 | 1.67913e-05 | 1.72416e-05 | 1.69916e-05 | 1.68749e-05 | 22.43 bps | 48.4% | -0.50% |
| 60 | 1800 | 3.76706e-05 | 3.78742e-05 | 3.78715e-05 | 3.82856e-05 | 33.78 bps | 49.8% | -1.63% |

![1m model comparison](charts/1min-model-comparison.png)

A negative gain means tuned Ridge did not beat the no-change forecast on the sealed test period. Such a result is useful: it bounds how much tradable directional information this linear close-only setup exposes.

## Learned dataset-specific parameters

| H | L | Local/global | Method | r | Effective local points | Alpha | Augmentation | Sigma | CV MSE |
|---:|---:|---|---|---:|---:|---:|---|---:|---:|
| 1 | 128 | local | standard | 0.3162 | 41 | 1.000e-06 | none | 0 | 5.49897e-06 |
| 5 | 8 | global | standard | 1.0000 | n/a | 0.0039811 | none | 0 | 1.38425e-05 |
| 15 | 8 | global | standard | 1.0000 | n/a | 6.310e-05 | none | 0 | 2.78491e-05 |
| 30 | 8 | global | standard | 1.0000 | n/a | 0.0014125 | none | 0 | 4.94069e-05 |
| 60 | 8 | global | standard | 1.0000 | n/a | 1.778e-04 | none | 0 | 9.84452e-05 |

The fitted relation is **L* = 62.17 H^-0.633** with R-squared **0.674**. A positive exponent means longer forecast horizons selected more context; a negative exponent means old history became less useful as the target moved farther out.

![1m lookback versus horizon](charts/1min-lookback-horizon.png)

![1m selected preprocessing parameters](charts/1min-hyperparameters.png)

## What the model uses

![1m Ridge weight magnitudes](charts/1min-weights.png)

The heatmap shows endpoint-forecast coefficients after preprocessing. Bright recent lags indicate local momentum/mean-reversion structure; isolated older bands suggest recurring phase anchors. White/blank left regions are outside the selected context.

![1m held-out forecast example](charts/1min-forecast.png)

The forecast plot is one deterministic held-out example at the longest evaluated horizon. It is diagnostic, not a hand-picked claim about average performance; the table above is the aggregate test result.

## Implications for later studies

- Selected lookback shrinks with horizon (`b=-0.633`).
- Local normalization wins 1/5 horizon cells; robust scaling wins 0/5.
- Noise augmentation wins 0/5 horizon cells.
- The one-step selected lookback is 128 candles versus the project's current nominal 64 candles at this scale.
- The selected lookback touches a searched boundary at H=5, 15, 30, 60; those L values are censored optima, not precise interior estimates.

The selected parameters should be treated as search priors, not fixed production truth. A later model should center its context and normalization search near these values, while retaining neighboring candidates and walk-forward validation.

## Limitations

- The target is univariate log close; the paper's cross-series grouping sweep is not applicable.
- Search uses deterministic joint trials rather than Optuna TPE; the searched axes and 21-value alpha loop match the paper.
- No nonlinear baseline is trained; the comparison isolates preprocessing gains against persistence and fixed Ridge baselines.

Fees, leverage, slippage, and position transitions are intentionally absent. Forecast accuracy is not trading profitability, especially at 1s and 1m where errors can be smaller than execution costs.

## Reproduce

```powershell
npm run analysis:searchcast-btc -- --scales 1m
```

Machine-readable details, all CV folds, and top trials are in [`results/1min.json`](results/1min.json).
