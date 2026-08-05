# SearchCast-style BTCUSDT analysis: 1M

> **Result.** At the one-candle horizon, the selected context is **12 1M candles**, using local standard normalization, alpha **0.70795**, and none augmentation. Held-out normalized MSE changes by **-17.25%** relative to persistence. Evidence quality at this scale is **exploratory**.

## Scope and adaptation

This report applies the transferable analysis from [How Good Can Linear Models Be for Time-Series Forecasting?](https://arxiv.org/html/2606.27282v1) to canonical spot BTCUSDT **log close** at 1M resolution. The model is multi-output Ridge regression. Search covers context length, global versus trailing-window normalization, standard versus robust scaling, time/frequency/no augmentation, and the paper's 21-value alpha grid. Candidate preprocessors are scored with chronological expanding-window validation; the final 20% of time is sealed until evaluation.

BTC close is a single target, so cross-series grouping is not applicable. The paper's forecast-horizon grouping question is represented here by independent tuning at several horizon cutoffs and the fitted relation `L* = a H^b`.

## Dataset statistics

| Statistic | Value |
|---|---:|
| Candles analyzed | 59 |
| Period | 2021-08-01T00:00:00Z to 2026-06-01T00:00:00Z |
| Source rows read | 2,629,440 / 2,629,440 (100.00%) |
| Price range | $15,476.00 to $126,199.63 |
| Mean / median log return | +37.7352 / +38.4825 bps |
| Return standard deviation | 1559.747 bps |
| Mean absolute return | 1219.373 bps |
| Annualized volatility | 54.03% |
| Skewness / excess kurtosis | +0.029 / +0.557 |
| Mean high-low range | 2418.784 bps |
| Median candle volume | 1,141,403.6799 BTC |

Complete UTC calendar months derived from common-window 1m candles.

![1M BTC data profile](charts/1month-data-profile.png)

Return autocorrelation measures linear predictability; absolute-return autocorrelation measures volatility clustering. The latter can be persistent even when signed returns are close to serially uncorrelated.

## Held-out forecasting results

All MSE values are divided by the development-window log-price variance. RMSE and MAE are log-price errors in basis points.

| H (candles) | Test windows | Persistence MSE | Global Ridge MSE | Local-full Ridge MSE | Tuned Ridge MSE | Tuned RMSE | Direction | Gain vs persistence |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12 | 0.052473 | 0.0632921 | 0.0615238 | 0.0615238 | 1238.61 bps | 41.7% | -17.25% |
| 2 | 11 | 0.0791102 | 0.0871805 | 0.112522 | 0.112522 | 1635.63 bps | 54.5% | -42.23% |
| 3 | 10 | 0.109955 | 0.0799147 | 0.192948 | 0.192948 | 2089.38 bps | 30.0% | -75.48% |
| 6 | 7 | 0.354279 | 0.128681 | 0.482084 | 0.482084 | 3120.20 bps | 14.3% | -36.07% |

![1M model comparison](charts/1month-model-comparison.png)

A negative gain means tuned Ridge did not beat the no-change forecast on the sealed test period. Such a result is useful: it bounds how much tradable directional information this linear close-only setup exposes.

## Learned dataset-specific parameters

| H | L | Local/global | Method | r | Effective local points | Alpha | Augmentation | Sigma | CV MSE |
|---:|---:|---|---|---:|---:|---:|---|---:|---:|
| 1 | 12 | local | standard | 1.0000 | 12 | 0.70795 | none | 0 | 0.171235 |
| 2 | 12 | local | standard | 1.0000 | 12 | 0.70795 | none | 0 | 0.362765 |
| 3 | 12 | local | standard | 1.0000 | 12 | 0.089125 | none | 0 | 0.635656 |
| 6 | 6 | local | standard | 1.0000 | 6 | 1.000e-06 | none | 0 | 0.458598 |

The fitted relation is **L* = 14.03 H^-0.368** with R-squared **0.634**. A positive exponent means longer forecast horizons selected more context; a negative exponent means old history became less useful as the target moved farther out.

![1M lookback versus horizon](charts/1month-lookback-horizon.png)

![1M selected preprocessing parameters](charts/1month-hyperparameters.png)

## What the model uses

![1M Ridge weight magnitudes](charts/1month-weights.png)

The heatmap shows endpoint-forecast coefficients after preprocessing. Bright recent lags indicate local momentum/mean-reversion structure; isolated older bands suggest recurring phase anchors. White/blank left regions are outside the selected context.

![1M held-out forecast example](charts/1month-forecast.png)

The forecast plot is one deterministic held-out example at the longest evaluated horizon. It is diagnostic, not a hand-picked claim about average performance; the table above is the aggregate test result.

## Implications for later studies

- Selected lookback shrinks with horizon (`b=-0.368`).
- Local normalization wins 4/4 horizon cells; robust scaling wins 0/4.
- Noise augmentation wins 0/4 horizon cells.
- The one-step selected lookback is 12 candles versus the project's current nominal 16 candles at this scale.
- The selected lookback touches a searched boundary at H=1, 2, 3, 6; those L values are censored optima, not precise interior estimates.

The selected parameters should be treated as search priors, not fixed production truth. A later model should center its context and normalization search near these values, while retaining neighboring candidates and walk-forward validation.

## Limitations

- The target is univariate log close; the paper's cross-series grouping sweep is not applicable.
- Search uses deterministic joint trials rather than Optuna TPE; the searched axes and 21-value alpha loop match the paper.
- No nonlinear baseline is trained; the comparison isolates preprocessing gains against persistence and fixed Ridge baselines.
- Only 59 complete 1M candles exist in the common window; estimates are exploratory and high variance.
- At least one horizon has fewer than 30 held-out windows, so test metrics are descriptive rather than inferential.

Fees, leverage, slippage, and position transitions are intentionally absent. Forecast accuracy is not trading profitability, especially at 1s and 1m where errors can be smaller than execution costs.

## Reproduce

```powershell
npm run analysis:searchcast-btc -- --scales 1M
```

Machine-readable details, all CV folds, and top trials are in [`results/1month.json`](results/1month.json).
