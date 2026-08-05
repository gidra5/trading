# SearchCast Exchange-Rate replication

Date: 2026-08-05

## Outcome

The official SearchCast implementation reproduces the paper's Exchange-Rate row essentially exactly. The selected `series_group_size=4` run has mean MSE **0.341036**, compared with the paper's rounded **0.341**.

However, the same test windows scored with a repeat-the-last-value persistence forecast have mean MSE **0.341002**. The replicated SearchCast result therefore has **-0.00988% MSE skill versus persistence**: it is effectively tied and very slightly worse in aggregate.

This is the central reason the published Exchange numbers cannot be compared directly with our return-prediction skill. The benchmark forecasts slowly varying exchange-rate **levels**, for which copying the last level is already extremely strong. Our earlier experiments forecast returns, where the zero-return baseline is much harder to beat.

## Reproduction contract

- Source: [official SakanaAI/SearchCast repository](https://github.com/SakanaAI/SearchCast)
- Pinned commit: `9a12b22525d787c0e0f919b2bd5b26fec5d64d03`
- Paper: [How Good Can Linear Models Be for Time-Series Forecasting?](https://arxiv.org/html/2606.27282v2)
- Dataset: the official `exchange_rate.csv`, SHA-256 `48b4d9d3d508f5104162e85b9a6042e3557fde11aa9f2944eba8c0d0efc89842`
- Data: 7,588 daily observations from 1990-01-01 through 2010-10-10, with eight exchange-rate series
- Split: chronological 70% train, 10% validation, 20% test
- Targets: direct multi-step level forecasts for every lead from 1 through 720; reported MSE at 96, 192, 336, and 720 steps averages every lead up to that cutoff
- Context length: searched over 19 powers-of-two-spaced choices from 32 through 2,048
- Ridge regularization: searched over 21 values
- Local normalization: mean/std from a searched recent fraction of each input window; its scale is also supplied to the linear model
- Augmentation: searched among no augmentation, time-domain noise, and frequency-domain noise
- Selection: 20 Optuna trials with three chronological expanding folds; forecast leads are grouped by 24
- Cross-series setup: pooled Ridge models, with official series-group sizes 1, 2, 4, and 8 swept
- Randomness: the official canonical reproduction command supplies no seed, so last decimals may vary between runs

The official code first standardizes every series using training-split statistics. This determines the unit of the reported MSE. The window-local normalization is applied after that for model fitting and is reversed for evaluation.

## Exact results

| Series group size | H=96 MSE | H=192 MSE | H=336 MSE | H=720 MSE | Mean MSE |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.083498 | 0.171088 | 0.307684 | 0.821795 | 0.346016 |
| 2 | 0.083028 | 0.169956 | 0.305638 | 0.811861 | 0.342621 |
| **4 (selected)** | **0.081163** | **0.167092** | **0.305408** | **0.810480** | **0.341036** |
| 8 | 0.081175 | 0.167186 | 0.306436 | 0.814677 | 0.342368 |
| Paper | 0.081 | 0.167 | 0.305 | 0.811 | 0.341 |

The best grouping by mean MSE is 4, and all four cutoff results agree with the paper after its three-decimal rounding.

## Persistence audit

Persistence repeats the final context level at every future lead. It uses the exact standardized data, split boundaries, contexts, targets, and aggregation used by the official evaluator.

| Horizon | SearchCast MSE | Persistence MSE | SearchCast skill vs persistence |
|---:|---:|---:|---:|
| 96 | 0.081163 | 0.081126 | -0.04538% |
| 192 | 0.167092 | 0.167119 | +0.01592% |
| 336 | 0.305408 | 0.305700 | +0.09534% |
| 720 | 0.810480 | 0.810064 | -0.05135% |
| **Mean** | **0.341036** | **0.341002** | **-0.00988%** |

The model marginally wins at horizons 192 and 336 and marginally loses at 96 and 720. None of these differences is large enough to establish useful predictive skill without repeated seeds and uncertainty estimates.

The paper's reported 14.8% improvement is relative to its OLS baseline, not persistence. In this run the official global linear baseline averages 0.393517 MSE, so beating that baseline does not imply beating a no-change forecast.

## Interpretation for BTC experiments

To make our next BTC comparison faithful to this research setup, the primary input and target should be log-price or price **levels**, not precomputed returns:

1. Give the model a contiguous level history.
2. Directly predict the full future level path rather than autoregressively appending predicted returns.
3. Apply per-window normalization using only the input context and reverse it on the output.
4. Report the paper-style standardized-level MSE for comparability.
5. Also report MSE skill against persistence and return/direction metrics. Without these, a visually strong level-MSE result can contain no trading edge.

This should be treated as a benchmark-control experiment, not evidence that raw levels are a better trading target.

## Reproduction

From the repository root:

```powershell
npm run analysis:searchcast-exchange
```

The command clones the pinned official implementation into ignored tooling storage if necessary, verifies the dataset hash, runs any missing series-group sweep, computes the persistence audit, and writes the machine-readable summary to `data/training/runs/searchcast-exchange-official-9a12b225/summary.json`.

To intentionally repeat all stochastic sweeps:

```powershell
npm run analysis:searchcast-exchange -- --rerun
```
