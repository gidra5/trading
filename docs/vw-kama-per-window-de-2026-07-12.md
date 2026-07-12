# VW-KAMA per-window optimization

Generated: 2026-07-12T17:05:34.194Z
Algorithm: de; 256 population/trials; 8 generations; 1m, 5m, 15m.

These are hindsight upper-bound configurations for comparison, not deployable validation results.

Every selected preset is scored against the global candidates at the same window and candle scale; the global score is therefore a hard lower bound.

| window | scale | score | preset |
|---|---:|---:|---|
| sideways-churn-2022-07 | 1m | 53.51% | window-sideways-churn-2022-07-1m |
| sideways-churn-2022-07 | 5m | 47.65% | window-sideways-churn-2022-07-5m |
| sideways-churn-2022-07 | 15m | 38.74% | window-sideways-churn-2022-07-15m |
| sideways-churn-2022-05 | 1m | 55.69% | window-sideways-churn-2022-05-1m |
| sideways-churn-2022-05 | 5m | 50.11% | window-sideways-churn-2022-05-5m |
| sideways-churn-2022-05 | 15m | 39.39% | window-sideways-churn-2022-05-15m |
| sideways-churn-2021-12 | 1m | 53.93% | window-sideways-churn-2021-12-1m |
| sideways-churn-2021-12 | 5m | 47.59% | window-sideways-churn-2021-12-5m |
| sideways-churn-2021-12 | 15m | 40.24% | window-sideways-churn-2021-12-15m |
| sideways-churn-2021-09 | 1m | 55.84% | window-sideways-churn-2021-09-1m |
| sideways-churn-2021-09 | 5m | 48.18% | window-sideways-churn-2021-09-5m |
| sideways-churn-2021-09 | 15m | 40.41% | window-sideways-churn-2021-09-15m |
| sideways-churn-2023-03 | 1m | 52.69% | window-sideways-churn-2023-03-1m |
| sideways-churn-2023-03 | 5m | 48.75% | window-sideways-churn-2023-03-5m |
| sideways-churn-2023-03 | 15m | 39.86% | window-sideways-churn-2023-03-15m |
| regime-up-2023-03 | 1m | 58.2% | window-regime-up-2023-03-1m |
| regime-up-2023-03 | 5m | 48.79% | window-regime-up-2023-03-5m |
| regime-up-2023-03 | 15m | 38.88% | window-regime-up-2023-03-15m |
| regime-flat-2026-04 | 1m | 43.16% | window-regime-flat-2026-04-1m |
| regime-flat-2026-04 | 5m | 40.3% | window-regime-flat-2026-04-5m |
| regime-flat-2026-04 | 15m | 34.74% | window-regime-flat-2026-04-15m |
| regime-down-2022-06 | 1m | 65.12% | window-regime-down-2022-06-1m |
| regime-down-2022-06 | 5m | 57.39% | window-regime-down-2022-06-5m |
| regime-down-2022-06 | 15m | 41.87% | window-regime-down-2022-06-15m |
| shape-up-low-2024-02 | 1m | 50.85% | window-shape-up-low-2024-02-1m |
| shape-up-low-2024-02 | 5m | 45.96% | window-shape-up-low-2024-02-5m |
| shape-up-low-2024-02 | 15m | 41.93% | window-shape-up-low-2024-02-15m |
| shape-up-high-2022-06 | 1m | 65.28% | window-shape-up-high-2022-06-1m |
| shape-up-high-2022-06 | 5m | 54.03% | window-shape-up-high-2022-06-5m |
| shape-up-high-2022-06 | 15m | 43.37% | window-shape-up-high-2022-06-15m |
| shape-down-low-2023-06 | 1m | 56.75% | window-shape-down-low-2023-06-1m |
| shape-down-low-2023-06 | 5m | 48.78% | window-shape-down-low-2023-06-5m |
| shape-down-low-2023-06 | 15m | 42.43% | window-shape-down-low-2023-06-15m |
| shape-down-high-2022-06 | 1m | 68.57% | window-shape-down-high-2022-06-1m |
| shape-down-high-2022-06 | 5m | 60.96% | window-shape-down-high-2022-06-5m |
| shape-down-high-2022-06 | 15m | 45.87% | window-shape-down-high-2022-06-15m |
| shape-flat-high-bias-2021-10 | 1m | 59.22% | window-shape-flat-high-bias-2021-10-1m |
| shape-flat-high-bias-2021-10 | 5m | 50.25% | window-shape-flat-high-bias-2021-10-5m |
| shape-flat-high-bias-2021-10 | 15m | 40.34% | window-shape-flat-high-bias-2021-10-15m |
| shape-flat-high-bias-low-2025-02 | 1m | 50.77% | window-shape-flat-high-bias-low-2025-02-1m |
| shape-flat-high-bias-low-2025-02 | 5m | 52.88% | window-shape-flat-high-bias-low-2025-02-5m |
| shape-flat-high-bias-low-2025-02 | 15m | 39.52% | window-shape-flat-high-bias-low-2025-02-15m |
| shape-flat-low-bias-2024-07 | 1m | 56.14% | window-shape-flat-low-bias-2024-07-1m |
| shape-flat-low-bias-2024-07 | 5m | 51.93% | window-shape-flat-low-bias-2024-07-5m |
| shape-flat-low-bias-2024-07 | 15m | 42.21% | window-shape-flat-low-bias-2024-07-15m |
| shape-flat-low-bias-low-2025-07 | 1m | 48.05% | window-shape-flat-low-bias-low-2025-07-1m |
| shape-flat-low-bias-low-2025-07 | 5m | 56.04% | window-shape-flat-low-bias-low-2025-07-5m |
| shape-flat-low-bias-low-2025-07 | 15m | 37.87% | window-shape-flat-low-bias-low-2025-07-15m |
| shape-flat-mid-bias-2024-01 | 1m | 58.02% | window-shape-flat-mid-bias-2024-01-1m |
| shape-flat-mid-bias-2024-01 | 5m | 52.76% | window-shape-flat-mid-bias-2024-01-5m |
| shape-flat-mid-bias-2024-01 | 15m | 40.29% | window-shape-flat-mid-bias-2024-01-15m |
| shape-flat-mid-bias-low-2023-09 | 1m | 49.69% | window-shape-flat-mid-bias-low-2023-09-1m |
| shape-flat-mid-bias-low-2023-09 | 5m | 48.17% | window-shape-flat-mid-bias-low-2023-09-5m |
| shape-flat-mid-bias-low-2023-09 | 15m | 43.52% | window-shape-flat-mid-bias-low-2023-09-15m |
| sharpe-up-3d-2024-11 | 1m | 57.74% | window-sharpe-up-3d-2024-11-1m |
| sharpe-up-3d-2024-11 | 5m | 50.35% | window-sharpe-up-3d-2024-11-5m |
| sharpe-up-3d-2024-11 | 15m | 38.88% | window-sharpe-up-3d-2024-11-15m |
| sharpe-up-3d-2023-12 | 1m | 53.92% | window-sharpe-up-3d-2023-12-1m |
| sharpe-up-3d-2023-12 | 5m | 50.08% | window-sharpe-up-3d-2023-12-5m |
| sharpe-up-3d-2023-12 | 15m | 39.49% | window-sharpe-up-3d-2023-12-15m |
| sharpe-down-3d-2026-06 | 1m | 56.27% | window-sharpe-down-3d-2026-06-1m |
| sharpe-down-3d-2026-06 | 5m | 50.38% | window-sharpe-down-3d-2026-06-5m |
| sharpe-down-3d-2026-06 | 15m | 41.08% | window-sharpe-down-3d-2026-06-15m |
| sharpe-down-3d-2023-03 | 1m | 53.98% | window-sharpe-down-3d-2023-03-1m |
| sharpe-down-3d-2023-03 | 5m | 45.1% | window-sharpe-down-3d-2023-03-5m |
| sharpe-down-3d-2023-03 | 15m | 37.75% | window-sharpe-down-3d-2023-03-15m |
| sharpe-up-7d-2023-12 | 1m | 49.67% | window-sharpe-up-7d-2023-12-1m |
| sharpe-up-7d-2023-12 | 5m | 46.1% | window-sharpe-up-7d-2023-12-5m |
| sharpe-up-7d-2023-12 | 15m | 34.65% | window-sharpe-up-7d-2023-12-15m |
| sharpe-up-7d-2024-11 | 1m | 56.79% | window-sharpe-up-7d-2024-11-1m |
| sharpe-up-7d-2024-11 | 5m | 47.94% | window-sharpe-up-7d-2024-11-5m |
| sharpe-up-7d-2024-11 | 15m | 37.72% | window-sharpe-up-7d-2024-11-15m |
| sharpe-down-7d-2023-03 | 1m | 52.51% | window-sharpe-down-7d-2023-03-1m |
| sharpe-down-7d-2023-03 | 5m | 42.77% | window-sharpe-down-7d-2023-03-5m |
| sharpe-down-7d-2023-03 | 15m | 34.63% | window-sharpe-down-7d-2023-03-15m |
| sharpe-down-7d-2026-06 | 1m | 51.78% | window-sharpe-down-7d-2026-06-1m |
| sharpe-down-7d-2026-06 | 5m | 43.36% | window-sharpe-down-7d-2026-06-5m |
| sharpe-down-7d-2026-06 | 15m | 37.02% | window-sharpe-down-7d-2026-06-15m |
| failure-down-3d-2022-06 | 1m | 67.13% | window-failure-down-3d-2022-06-1m |
| failure-down-3d-2022-06 | 5m | 60.56% | window-failure-down-3d-2022-06-5m |
| failure-down-3d-2022-06 | 15m | 45.37% | window-failure-down-3d-2022-06-15m |
| failure-down-7d-2022-06 | 1m | 59.92% | window-failure-down-7d-2022-06-1m |
| failure-down-7d-2022-06 | 5m | 54.48% | window-failure-down-7d-2022-06-5m |
| failure-down-7d-2022-06 | 15m | 41.74% | window-failure-down-7d-2022-06-15m |
