# Five-day BTC return prediction at 1d resolution

Date: 2026-08-05

## Question

Can the existing two-layer normalized GLU predict the next five daily BTCUSDT log returns from 120 daily positions, and how many telescoping moving-average levels should it receive?

## Inputs

Every model consumes 120 daily-spaced positions. Four common-corpus variants were trained:

| Maximum window | Input terms at every daily position | MA levels |
| --- | --- | ---: |
| None / raw (`1d`) | `return_1d` | 0 |
| `1w` | `ma_1w`, `return_1d - ma_1w` | 1 |
| `1M` | `ma_1M`, `ma_1w - ma_1M`, `return_1d - ma_1w` | 2 |
| `3M` | `ma_3M`, `ma_1M - ma_3M`, `ma_1w - ma_1M`, `return_1d - ma_1w` | 3 |

The MA windows are 7, 30, and 90 daily candles. The terms telescope exactly back to the corresponding raw daily log return. All components are passed jointly to one model; there are no separately trained component forecasters.

Each feature component and each of its 120 lag positions has a mean and standard deviation fitted only on the training split. The model directly emits five raw daily log returns after the output normalization is reversed.

## Common corpus and split

- Source: complete UTC BTCUSDT daily closes derived from the canonical one-minute candle references.
- Full close series: 2021-07-25 through 2026-07-24.
- Common warm-up: every depth starts after enough history for both 120 daily positions and the widest 90-day MA.
- Train: 881 forecasts, 2022-02-20 through 2024-07-19.
- Validation: 361 forecasts, 2024-07-24 through 2025-07-19.
- Sealed test: 362 forecasts, 2025-07-24 through 2026-07-20; the fifth target of the final forecast ends on 2026-07-24.
- Four forecasts are purged at each split boundary so targets do not cross into the next split.

The validation and test boundaries are 60% and 80% of the full daily close series, matching the final-20% boundary used by the earlier SearchCast-style BTC daily analysis. Past context from an earlier split is allowed because it is observable at prediction time; future targets never cross a boundary.

## Model and training

- Two normalized-GLU layers, width 512 each.
- Dropout 0.05 with dropout rate 0.5.
- Training-position input normalization.
- Five direct daily-return outputs.
- Loss: normalized per-candle MSE plus normalized cumulative five-day-return MSE.
- Hybrid Muon/AdamW optimizer, learning rate `1e-4`.
- Maximum 32 epochs; early stopping patience 8 on the combined validation objective.
- Fixed seed 1337 and BF16 CUDA training.
- Only the validation-selected MA depth is evaluated on test.

## Validation depth screen

Positive skill means lower per-candle MSE than predicting zero daily return.

| Maximum window | Parameters | Best epoch | Epochs run | Validation MSE | MSE skill vs zero | Direction | Correlation | Validation objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| None / raw (`1d`) | 2,750,985 | 1 | 9 | 0.000628353 | +0.2843% | 52.3546% | 0.0153 | 1.479997 |
| **`1w`** | **2,873,865** | **2** | **10** | **0.000628015** | **+0.3380%** | **53.1302%** | **0.0246** | **1.477525** |
| `1M` | 2,996,745 | 1 | 9 | 0.000628645 | +0.2379% | 52.3546% | -0.0138 | 1.481304 |
| `3M` | 3,119,625 | 1 | 9 | 0.000630020 | +0.0197% | 52.2992% | -0.0699 | 1.490924 |

The one-week decomposition wins both the combined training objective and aggregate per-candle MSE skill. All variants early-stop long before 32 epochs, and the selected checkpoints occur at epoch 1 or 2.

The selected one-week model's validation MSE skill by future day is:

| +1d | +2d | +3d | +4d | +5d |
| ---: | ---: | ---: | ---: | ---: |
| +0.3052% | +0.3111% | +0.4606% | +0.3339% | +0.2788% |

## Sealed test result for the selected one-week model

| Metric | Result |
| --- | ---: |
| Aggregate five-output MSE | 0.000518092 |
| Zero-return MSE | 0.000516675 |
| Aggregate MSE skill vs zero | **-0.2743%** |
| Per-candle direction accuracy | 50.1657% |
| Per-candle correlation | 0.0474 |
| Five-day cumulative-return MSE | 0.00222901 |
| Zero cumulative-return MSE | 0.00219872 |
| Five-day cumulative MSE skill | **-1.3774%** |

Per-day test metrics:

| Lead | MSE skill vs zero | Direction | Correlation |
| --- | ---: | ---: | ---: |
| +1d | -0.1009% | 51.3812% | 0.0699 |
| +2d | -0.2260% | 50.2762% | 0.0582 |
| +3d | -0.3225% | 50.2762% | 0.0437 |
| +4d | -0.4045% | 49.1713% | 0.0221 |
| +5d | -0.3172% | 49.7238% | 0.0426 |

The small positive per-candle validation edge does not survive the sealed period. Cumulative five-day skill is +2.0308% on validation but reverses to -1.3774% on test, so it also fails to generalize.

## Same-period comparison with the daily SearchCast-style Ridge

For a one-day target, level-forecast error and return-forecast error are algebraically identical after subtracting the known current log close. Re-evaluating the selected three-lag SearchCast-style Ridge on the same 362 daily targets gives:

| Model, +1d target | MSE | MSE skill vs zero | Direction | Correlation |
| --- | ---: | ---: | ---: | ---: |
| Two-layer GLU, selected `1w` input | **0.000516060** | **-0.1009%** | **51.3812%** | **0.0699** |
| SearchCast-style Ridge | 0.000524108 | -1.6620% | 49.7238% | 0.0277 |
| Zero return | **0.000515539** | 0% | n/a | n/a |

The GLU is materially better than the SearchCast-style Ridge on the identical days, but neither beats zero-return persistence on one-day MSE.

## Reproduce

```powershell
npm run mlp:experiment:multiscale-daily
```

The matrix runner resumes completed cells. Result JSON and checkpoints are under `data/training/runs/multiscale-candle-1d-*-joint-5path-32epoch-daily-v2/`.
