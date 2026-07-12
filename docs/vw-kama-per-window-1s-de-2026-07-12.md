# VW-KAMA per-window optimization

Generated: 2026-07-12T17:18:58.120Z
Algorithm: de; 64 population/trials; 6 generations; 1s.

These are hindsight upper-bound configurations for comparison, not deployable validation results.

Every selected preset is scored against the global candidates at the same window and candle scale; the global score is therefore a hard lower bound.

| window | scale | score | preset |
|---|---:|---:|---|
| sideways-churn-2022-07 | 1s | 56.41% | window-sideways-churn-2022-07-1s |
| sideways-churn-2022-05 | 1s | 61.42% | window-sideways-churn-2022-05-1s |
| sideways-churn-2021-12 | 1s | 57.86% | window-sideways-churn-2021-12-1s |
| sideways-churn-2021-09 | 1s | 61.17% | window-sideways-churn-2021-09-1s |
| sideways-churn-2023-03 | 1s | 61.01% | window-sideways-churn-2023-03-1s |
| regime-up-2023-03 | 1s | 65.64% | window-regime-up-2023-03-1s |
| regime-flat-2026-04 | 1s | 46.89% | window-regime-flat-2026-04-1s |
| regime-down-2022-06 | 1s | 70.83% | window-regime-down-2022-06-1s |
| shape-up-low-2024-02 | 1s | 51.54% | window-shape-up-low-2024-02-1s |
| shape-up-high-2022-06 | 1s | 69.54% | window-shape-up-high-2022-06-1s |
| shape-down-low-2023-06 | 1s | 59.72% | window-shape-down-low-2023-06-1s |
| shape-down-high-2022-06 | 1s | 74.55% | window-shape-down-high-2022-06-1s |
| shape-flat-high-bias-2021-10 | 1s | 63.21% | window-shape-flat-high-bias-2021-10-1s |
| shape-flat-high-bias-low-2025-02 | 1s | 49.68% | window-shape-flat-high-bias-low-2025-02-1s |
| shape-flat-low-bias-2024-07 | 1s | 59.16% | window-shape-flat-low-bias-2024-07-1s |
| shape-flat-low-bias-low-2025-07 | 1s | 46.36% | window-shape-flat-low-bias-low-2025-07-1s |
| shape-flat-mid-bias-2024-01 | 1s | 62.94% | window-shape-flat-mid-bias-2024-01-1s |
| shape-flat-mid-bias-low-2023-09 | 1s | 48.74% | window-shape-flat-mid-bias-low-2023-09-1s |
| sharpe-up-3d-2024-11 | 1s | 61.73% | window-sharpe-up-3d-2024-11-1s |
| sharpe-up-3d-2023-12 | 1s | 57.5% | window-sharpe-up-3d-2023-12-1s |
| sharpe-down-3d-2026-06 | 1s | 58.98% | window-sharpe-down-3d-2026-06-1s |
| sharpe-down-3d-2023-03 | 1s | 62.07% | window-sharpe-down-3d-2023-03-1s |
| sharpe-up-7d-2023-12 | 1s | 54.27% | window-sharpe-up-7d-2023-12-1s |
| sharpe-up-7d-2024-11 | 1s | 58.7% | window-sharpe-up-7d-2024-11-1s |
| sharpe-down-7d-2023-03 | 1s | 58.67% | window-sharpe-down-7d-2023-03-1s |
| sharpe-down-7d-2026-06 | 1s | 53.43% | window-sharpe-down-7d-2026-06-1s |
| failure-down-3d-2022-06 | 1s | 70.68% | window-failure-down-3d-2022-06-1s |
| failure-down-7d-2022-06 | 1s | 66.03% | window-failure-down-7d-2022-06-1s |
