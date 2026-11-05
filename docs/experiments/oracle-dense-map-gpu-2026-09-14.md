# GPU dense account-map sweep

The dense Bellman sweep now runs on CUDA by default when a CUDA device is available. NSF market forecasting was already on the GPU; the preceding dense account sweep used eight Numba CPU workers. This change accelerates the same approximate controller. It does not change its grid, action search, sampled price law or interpolation accuracy.

## Parallel work

`ml/oracle_book_gpu.py` uses Triton to evaluate independent account cells, market representatives and conditional outcomes concurrently. Candidate actions are parallel lanes within each outcome. It first maximizes over actions after the hypothetical next observation, then averages over that observation's conditional law. Maximization and expectation have the same order as the CPU Bellman implementation.

All 3,600 continuation layers remain resident on the device during the backward sweep. Layers have a temporal dependency: layer t needs the completed layer t+1. They are evaluated in that order. Each layer launches a candidate-evaluation kernel followed by a conditional-mean/dispersion kernel. Completed value and dispersion arrays are copied back once at the end.

Every configured account cell remains allocated. A preprocessing pass uses the reference representative-state checks at the prices in the saved cloud to exclude impossible representatives from GPU scheduling. In this experiment 167 of 945 wealth/account representatives need work. The remaining cells receive explicit nonfinite values. This does not drop valid actions or coarsen the map. Price-dependent validity is still checked inside the GPU kernel.

Execution arithmetic and stored values use double precision, with fused floating-point contraction disabled. The GPU implements the same fees, funding, liquidation timing, aggregate caps, reservation checks, limit crossing and terminal closing rules. CPU code remains available as the numerical reference and for machines without CUDA. Selecting CUDA explicitly never silently falls back to CPU.

Root queries, observation-error statistics, random candidate/sample schedules and adaptive compute allocations remain CPU work. The measured bottleneck was the backward sweep. At one sampled point during the final sweep, `nvidia-smi` reported 98% GPU utilization and 2,373 MiB total device memory use on the RTX 3070; these are instantaneous device-wide readings, not isolated peak-memory measurements.

## Reproducible comparison

The CPU archive is `data/backtests/oracle-dense-map-hour-verified`. The final GPU archive is `data/backtests/oracle-dense-map-gpu-final`. Both use the exact saved market clouds in `data/backtests/oracle-dense-map-hour`: 3,600 one-second layers, eight market representatives, eight conditional outcomes, 315 account cells at three wealth nodes, and the same rotating candidate catalogs. The initial sweep and reference sweeps use all eight outcomes; subsequent adaptive sweeps average four.

Times below measure value sweeps, including initialization, scheduling and exporting the completed fields. They exclude constructing the forecast cloud, which was cached for this comparison. The initial CPU sweep took 284.4 seconds; the optimized initial GPU sweep took 87.9 seconds, a 3.24-fold speedup. This is the compute for one map covering a one-hour future, not a full realized-hour backtest with 3,600 forecast refreshes.

| Sweep | CPU, eight workers | RTX 3070 | Speedup |
| --- | ---: | ---: | ---: |
| Initial, eight outcomes | 284.4 s | 87.9 s | 3.24x |
| Update +1 s, adaptive four | 242.2 s | 77.9 s | 3.11x |
| Update +1 s, reference eight | 280.0 s | 84.6 s | 3.31x |
| Update +2 s, adaptive four | 259.2 s | 77.9 s | 3.33x |
| Update +2 s, reference eight | 262.6 s | 84.6 s | 3.10x |

The complete five-sweep experiment took 417.0 seconds with cached forecasts, compared with 1,332 seconds for the CPU experiment. The initial CUDA sweep includes one-time initialization/compilation overhead. These are single-run wall-clock measurements on the same machine, not a distribution over repeated timing trials.

The earlier unfiltered GPU work schedule is archived separately in `data/backtests/oracle-dense-map-gpu`. Its initial sweep took 106.7 seconds. It is an intermediate performance measurement, not the default implementation.

`ml/audit_oracle_dense_gpu.py` compares the saved execution/search contracts, market arrays and sample allocations before reporting differences over the entire value and dispersion fields. The report is written as `cpu-gpu-comparison.json` in the candidate archive. This checks CPU/GPU numerical agreement; neither implementation is a certificate of continuous-policy optimality.

All five final fields used identical market arrays and conditional-sample allocations to the CPU archive. Across **20,592,704 finite value entries**, the maximum absolute difference was **5.33e-15** in normalized log value. Finite masks and nonfinite marker kinds agreed everywhere. The largest variance difference was **2.13e-18** (rounded upward). Initial cash and subsequent two-pending-buy selections matched the preceding experiment; the adaptive choices still tied with cash under the full-sample reference value.

Four dedicated GPU conformance tests compare all layers against the CPU kernel. They cover cash/directional actions, up to three pending orders, nonzero fees and funding, aggregate order caps, liquidation, reduced conditional sampling, and price-dependent order validity near the minimum price tick.

These four tests, eight account-grid tests, eight value-map tests and ten existing Torch Oracle tests all pass: **30 tests**, 7.0 seconds of test execution in the final run.

The final GPU map was also loaded into the existing frozen-map realized-hour replay. It ended flat at $10,000 (0% return), with no market trades, limit fills or execution failures, matching the earlier initial-map replay. Its 3,600 decisions took 1.13 seconds, excluding 0.44 seconds of loading and Python startup. Results are in `data/backtests/oracle-dense-map-gpu-frozen-replay`. Forecasts and the map were not refreshed during this replay; it tests using the exported GPU field, not a rolling one-hour forecast refresh every second.

The GPU acceleration described here recomputes target means over the whole horizon. A subsequent [reuse experiment](oracle-dense-map-reuse-2026-09-14.md) adds corrected per-outcome reuse and feasibility checks, but has not yet met the accuracy requirement; it remains opt-in.

## Commands

```powershell
.venv-ml/Scripts/python.exe ml/benchmark_oracle_dense_map.py --output data/backtests/oracle-dense-map-gpu-new --horizon 3600 --width 8 --branches 8 --budget 4 --updates 2 --candidates 16 --market-cache data/backtests/oracle-dense-map-hour --backend cuda
.venv-ml/Scripts/python.exe ml/audit_oracle_dense_gpu.py data/backtests/oracle-dense-map-hour-verified data/backtests/oracle-dense-map-gpu-new
.venv-ml/Scripts/python.exe -m unittest discover -s ml -p test_oracle_book_gpu.py -v
```

Omit `--backend` to select CUDA automatically when available, or pass `--backend cpu` for the reference sweep. The experiment records its backend and source snapshot.
