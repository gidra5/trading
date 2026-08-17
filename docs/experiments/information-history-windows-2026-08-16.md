# Historical-window sensitivity of the next-1s distribution model

Generated 2026-08-16T19:13:42.913Z for BTCUSDT. All windows score exactly the same future targets from 2023-07-25T00:00:00.000Z through 2026-07-25T00:00:00.000Z.

## Result

The lowest held-out log loss uses **90 trailing days**: 2.5971050 bits/target. It retains 0.099565912 bits/target beyond a previous-return-only model and encounters an unseen five-feature context for 0.228% of targets.

| training history | model log loss | history-only log loss | gain beyond history | geometric probability gain | unseen context targets |
|---|---:|---:|---:|---:|---:|
| 7 trailing days | 2.6592777 | 2.6334778 | -0.025799841 | -1.772% | 1.332% |
| 30 trailing days | 2.6004029 | 2.6566815 | 0.056278605 | 3.978% | 0.477% |
| 90 trailing days | 2.5971050 | 2.6966710 | 0.099565912 | 7.145% | 0.228% |
| 180 trailing days | 2.6202134 | 2.7439416 | 0.12372815 | 8.955% | 0.173% |
| 365 trailing days | 2.6679003 | 2.8271843 | 0.15928395 | 11.673% | 0.139% |
| 730 trailing days | 2.7068963 | 2.9226098 | 0.21571357 | 16.128% | 0.032% |
| all available prior history | 2.7331794 | 3.0038940 | 0.27071453 | 20.641% | 0.001% |

Lower log loss is better. The information-gain column compares the complete five-feature table against a previous-return-only table trained on the same amount of history, so it does not reward a longer window merely for estimating the unconditional distribution more accurately.

## By future epoch

### Epoch 2

| history | log loss | gain beyond history | unseen contexts |
|---|---:|---:|---:|
| 7 trailing days | 2.6202681 | -0.035984697 | 1.720% |
| 30 trailing days | 2.5838083 | 0.057959054 | 0.816% |
| 90 trailing days | 2.6236678 | 0.12500545 | 0.434% |
| 180 trailing days | 2.7104519 | 0.17136659 | 0.385% |
| 365 trailing days | 2.8337947 | 0.24464673 | 0.366% |
| 730 trailing days | 2.8409086 | 0.31034903 | 0.078% |
| all available prior history | 2.8091321 | 0.28375957 | 0.001% |

### Epoch 3

| history | log loss | gain beyond history | unseen contexts |
|---|---:|---:|---:|
| 7 trailing days | 2.7756278 | -0.027093125 | 1.110% |
| 30 trailing days | 2.6983628 | 0.056139906 | 0.290% |
| 90 trailing days | 2.6698636 | 0.090052294 | 0.113% |
| 180 trailing days | 2.6603226 | 0.10342677 | 0.056% |
| 365 trailing days | 2.6876025 | 0.13033803 | 0.024% |
| 730 trailing days | 2.7879802 | 0.21739649 | 0.010% |
| all available prior history | 2.8241017 | 0.28024539 | 0.001% |

### Epoch 4

| history | log loss | gain beyond history | unseen contexts |
|---|---:|---:|---:|
| 7 trailing days | 2.5820440 | -0.014293789 | 1.166% |
| 30 trailing days | 2.5190830 | 0.054732249 | 0.325% |
| 90 trailing days | 2.4977108 | 0.083570276 | 0.136% |
| 180 trailing days | 2.4896185 | 0.096260526 | 0.077% |
| 365 trailing days | 2.4818492 | 0.10263316 | 0.027% |
| 730 trailing days | 2.4914327 | 0.11913583 | 0.009% |
| all available prior history | 2.5660963 | 0.24810289 | 0.001% |

## Method limits

- This isolates the history used for probability-table estimation; it does not refit the indicator or quantization transforms for every window.
- The exact five-coordinate quartile table is deliberately high-dimensional, so short histories expose the real sparsity cost of this representation.
- A hierarchical-backoff or learned continuous model can use short recent windows more efficiently than exact cells.

Complete values are stored in `data/benchmarks/information-history-windows.json`.
