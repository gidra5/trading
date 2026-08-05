# Multiscale moving-average next-return experiment

Date: 2026-08-04

## Question

Can a two-layer normalized GLU improve the next five one-second return path by
decomposing each return into causal moving-average frequency bands?

Two combination strategies are compared at every maximum window:

1. `separate-components`: one independent width-512, two-layer GLU forecasts
   each component from its own 120-second history. Component forecasts are
   summed to form the raw-return forecast. Each branch is trained against its
   own component target; checkpoint selection uses the summed raw-return
   validation objective.
2. `joint-input`: all component histories are concatenated and passed to one
   width-512, two-layer GLU, which directly forecasts the raw five-return path.

Both strategies use fixed training-set, per-component, per-lag-position means
and standard deviations. There is no per-example normalization or reversible
local output transform.

## Windows

Fixed durations are used:

| Label | Duration |
|---|---:|
| `1m` | 60 seconds |
| `1h` | 3,600 seconds |
| `1d` | 86,400 seconds |
| `1w` | 604,800 seconds |
| `1M` | 2,592,000 seconds (30 days) |
| `3M` | 7,776,000 seconds (90 days) |

For a return ending at second `t`, the moving average is computed without a
long rolling scan:

`MA_W(t) = (log(close_t) - log(close_(t-W))) / W`

This is exactly the mean of the `W` one-second log returns ending at `t`.

## Telescoping components

For maximum window `1d`, for example, the stored components are:

1. `MA_1d`
2. `MA_1h - MA_1d`
3. `MA_1m - MA_1h`
4. `return_1s - MA_1m`

Their sum is `return_1s`. The other cases use the same construction:

| Maximum | Components |
|---|---:|
| `1m` | 2 |
| `1h` | 3 |
| `1d` | 4 |
| `1w` | 5 |
| `1M` | 6 |
| `3M` | 7 |

The high-frequency residual is calculated after the coarser bands are stored
as float32, so reconstruction is preserved to float32 summation precision.
Tests cover synthetic inputs and real BTCUSDT days at all six levels.

## Training protocol

- Input history: 120 one-second component values.
- Forecast horizon: five one-second returns/components.
- GLU depth and width: two layers, width 512.
- Epoch cap: 32; early-stopping patience: 8.
- Objective: normalized candle MSE plus normalized cumulative-return MSE.
- All other path-summary losses have zero weight and remain diagnostics.
- Test evaluation: disabled and sealed.
- The source split assignments and existing 125-second cross-split purge are
  retained. Moving averages are causal and use only market state available at
  prediction time.
- Source candle history is continuous from 2021-07-25 through 2026-07-24.
  Only the `3M` case trims rows that lack a complete 90-day lookback.

## Matrix size

| Maximum | Separate parameters | Joint parameters | Train examples | Validation examples |
|---|---:|---:|---:|---:|
| `1m` | 5,501,970 | 2,873,865 | 33,219,925 | 15,505,675 |
| `1h` | 8,252,955 | 2,996,745 | 33,219,925 | 15,505,675 |
| `1d` | 11,003,940 | 3,119,625 | 33,219,925 | 15,505,675 |
| `1w` | 13,754,925 | 3,242,505 | 33,219,925 | 15,505,675 |
| `1M` | 16,505,910 | 3,365,385 | 33,219,925 | 15,505,675 |
| `3M` | 19,256,895 | 3,488,265 | 32,269,525 | 15,073,925 |

## Execution status

The resumable 12-run matrix is active. The first `1m` separate-component run
passed its full normalization, training, validation, and checkpoint smoke test.
It reached 99-100% GPU utilization without an out-of-memory error. Final
validation comparisons will be added after early stopping completes all runs.

Matrix status:
`data/training/runs/multiscale-telescoping-glu-5s-2layer-32epoch-v1/state/matrix-status.json`

## Artifacts

- Model: `ml/multiscale_next_return.py`
- Streaming dataset: `ml/multiscale_next_return_dataset.py`
- Trainer: `ml/train_multiscale_next_return.py`
- Matrix runner: `ml/run_multiscale_next_return_matrix.py`
- Tests: `ml/test_multiscale_next_return.py`
- Plan: `ml/training-plans/multiscale-telescoping-glu-5s-2layer-32epoch-v1.json`
