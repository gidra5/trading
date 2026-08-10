# FinCast, TiRex-2, and Chronos-2 on BTCUSDT 1m

Date: 2026-08-07

Status: dense forecast benchmark, exhaustive bot-policy calibration, and the permitted held-out bot validation are complete.

## Question

Can FinCast, TiRex-2, or Chronos-2 predict the next 15 one-minute BTCUSDT candles well enough to improve candle MSE, return correlation, and the bot's oracle-action distribution, then survive the real trading simulator with fees and risk?

## Leakage contract

- Forecast horizon: exactly 15 consecutive one-minute candles.
- Test set: all 11,232 unique 15-minute forecast origins in the 28 static non-fit windows exposed by the KAMA inspector.
- Excluded: `fit-full`, `fit-1` through `fit-4`, and dynamic `latest`/latest 3M.
- Model/input selection used only the sparse and 217-origin calibration screens.
- Chronos-2 LoRA trained on 2021-07-01 through 2021-07-31 and selected on 2,976 August 2021 origins. Both ranges end before the first test window on 2021-09-08.
- Once the dense test began, no model, representation, context length, or checkpoint was changed from test results.
- Forecast rows use only the candles strictly before each target. Their decision time is one millisecond before the first target candle opens.

## Reproducible model setup

| Candidate | Official source | Source commit | Checkpoint | Pinned revision | Runtime |
|---|---|---|---|---|---|
| FinCast | <https://github.com/vincent05r/FinCast-fts> | `488b19d1d85fa2b3d4b93469530cefdcf1cc97a4` | `Vincent05R/FinCast` | `2d7d90b159db8961d27c2cf165d51195902ef92b` | CUDA, official PyTorch decoder, FP16 with rare FP32 fallback |
| TiRex-2 | <https://github.com/NX-AI/tirex-2> | `ad7ce6a2a0cb639ea58eedc3afc472e7e5b2bae0` | `NX-AI/TiRex-2` | `05e5b26db52bfb256f1ae1bdf785589850482de3` | CUDA with the official pure-PyTorch native kernels on Windows |
| Chronos-2 | <https://github.com/amazon-science/chronos-forecasting> | `7dc4435706a4454feb79df44ca9f33631f3027bf` | `amazon/chronos-2` | `29ec3766d36d6f73f0696f85560a422f50e8498c` | CUDA FP32 |

TiRex-2 was selected instead of the original TiRex because TiRex-2 is multivariate and Apache-2.0, while the original public model is univariate and uses a community license. All sources and weights are pinned in `.tools/forecast-models/manifest.json`; FinCast's 3.97 GB weight file is additionally checked against the publisher's SHA-256.

Setup and smoke commands:

```text
npm run forecast-models:setup
npm run forecast-models:smoke
```

The adapters expose one common tensor contract: point forecast `[batch, 5, 15]` plus native 0.1-0.9 quantiles `[batch, 5, 15, 9]`.

## Frozen zero-shot choices

The screen compared raw OHLCV, close-anchored log OHLCV, and structurally encoded candle returns at contexts 128, 256, and 512.

| Model | Frozen context | Representation | 217-origin candle skill | 217-origin horizon correlation |
|---|---:|---|---:|---:|
| FinCast | 128 | anchored log | -6.510% | 0.1111 |
| TiRex-2 | 256 | anchored log | -6.520% | -0.1339 |
| Chronos-2 | 256 | raw | +0.977% | 0.1648 |

These figures were calibration-only and were not treated as final evidence. The reversal in the dense results below demonstrates why.

## Chronos-2 LoRA

Chronos-2 is the only selected release with a complete official fine-tuning API. FinCast publishes experimental PEFT components without a complete official training entry point; TiRex-2's public release is inference-only.

Four leakage-safe Chronos-2 schedules were compared on August 2021:

| LoRA schedule | Candle skill | Minute-return correlation | Horizon correlation |
|---|---:|---:|---:|
| Base model | -1.891% | 0.01280 | 0.00985 |
| 100 steps, 1e-5 | -1.777% | 0.01353 | 0.01052 |
| 300 steps, 1e-5 | -1.668% | 0.01330 | 0.01081 |
| 100 steps, 1e-4 | -1.135% | 0.01391 | 0.02065 |
| **300 steps, 1e-4** | **+0.046%** | 0.01033 | **0.02967** |

The winner was frozen before dense testing. Actual training took about 56 seconds for 300 steps on the local GPU; the merged production checkpoint is self-contained under `data/models/forecast-foundation/chronos2-btc-pretest-lora/`.

## Dense untouched forecast results

All metrics below aggregate the same 11,232 origins. Candle MSE is mean squared error over anchored log OHLC. Skill is `1 - model MSE / persistence MSE`; positive is better than repeating the last close. Oracle KL is forward KL from the realized oracle-action distribution to the forecast-derived distribution; lower is better. The nine native quantile trajectories are used as equal-weight distribution support. Invalid raw OHLC quantiles are repaired by the same KQSP projection used in the Kronos study.

| Variant | Candle MSE | Skill vs persistence | Candle corr. | 1m return corr. | 15m return corr. | Horizon direction | Oracle KL | CRPS | Raw valid OHLC | Repaired valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FinCast zero-shot | 7.4105e-6 | -3.972% | 0.1096 | -0.00344 | -0.00237 | 51.14% | 1.3995 | 0.0011816 | 49.98% | 100% |
| TiRex-2 zero-shot | 7.3980e-6 | -3.797% | 0.1003 | 0.00249 | -0.01047 | 51.48% | **1.3910** | **0.0011666** | 52.90% | 100% |
| Chronos-2 zero-shot | 7.3901e-6 | -3.685% | 0.0966 | -0.00198 | -0.02948 | 50.61% | 1.4225 | 0.0011751 | 78.74% | 100% |
| **Chronos-2 BTC LoRA** | **7.2656e-6** | **-1.938%** | **0.1108** | **0.00969** | -0.02153 | 50.75% | 1.5398 | 0.0011743 | **81.91%** | 100% |

Main conclusions:

- None of the candidates beats candle persistence over the complete test corpus.
- LoRA generalizes in the narrow sense that it improves Chronos-2 candle MSE by 1.75 percentage points of skill and turns minute-return correlation positive. It does not produce positive 15-minute correlation and worsens oracle KL.
- TiRex-2 has the best probabilistic CRPS and oracle KL, but its point/horizon signal is still not useful.
- Direction accuracy slightly above 50% coexists with near-zero or negative correlation; this is too small and poorly calibrated to imply trading value.
- KQSP guarantees valid output candles after repair, but it does not create predictive skill.

Per-window robustness is also weak. FinCast, TiRex-2, base Chronos-2, and tuned Chronos-2 beat persistence in only 2, 4, 3, and 9 of the 28 windows respectively. Every model's worst skill occurs in `shape-down-low-2023-06`; tuned Chronos-2 falls to -14.31% there.

## Real bot integration

Each 125 MB causal forecast artifact passed the real bot simulator's strict parser and replay smoke. The simulator consumed all 672 forecast decisions in the selected pre-cutoff window, generated actual fills, charged configured fees and maintenance, and recorded no liquidations.

| Variant | Aggressive smoke return | Fills | Fees | Maintenance | Max drawdown | Liquidations |
|---|---:|---:|---:|---:|---:|---:|
| FinCast | -3.049% | 21 | 100.79 | 38.94 | 3.61% | 0 |
| TiRex-2 | -1.058% | 13 | 77.18 | 39.54 | 3.07% | 0 |
| Chronos-2 base | +0.085% | 15 | 36.78 | 3.93 | 0.49% | 0 |
| Chronos-2 BTC LoRA | -0.829% | 7 | 45.72 | 0.00 | 1.53% | 0 |

This smoke is integration evidence, not a profitability result. Separate exhaustive 990-policy searches were completed for all four artifacts. Policies were ranked on 11 pre-2024 episodes, had to independently pass two post-training/pre-validation confirmation episodes, and only then were allowed to be replayed on the final eight inspector windows (merged into six non-overlapping market episodes).

| Variant | Broad-gate candidates | Confirmation-gate candidates | Passed both | Final replay allowed |
|---|---:|---:|---:|---:|
| FinCast | 3 | 168 | 0 | No |
| TiRex-2 | 0 | 84 | 0 | No |
| Chronos-2 base | 0 | 124 | 0 | No |
| Chronos-2 BTC LoRA | at least 1 | at least 1 | 1 | Yes |

The frozen tuned-Chronos policy uses the median 15-minute horizon return, requires six consecutive same-direction decisions and at least 20 bps predicted movement, and otherwise holds. It passed calibration narrowly: +0.00139% geometric mean across 11 selection episodes, with activity in 6/11 and profit in 3/6 active episodes; confirmation was +0.0979% geometric mean, active and profitable in 1/2 episodes.

On the one-time final replay it made $887.61 net across six independent episodes: +1.4274% geometric mean episode return, 18 fills, $24.42 fees, 1.52% maximum drawdown, and zero liquidations. It exceeded the best constant-direction control, constant long at 1x, by 0.316 percentage points of geometric mean return. However, every trade occurred in a single November 2024 uptrend episode, which returned +8.876%; the policy did nothing in the other five episodes. Consequently the final aggregate fails the same breadth/activity eligibility rule. Across the eight overlapping inspector windows the aggregate was +1.0886%, active in only 2/8 windows.

This is a genuine held-out positive-PnL result, but not a robust or deployable one. It demonstrates that the adapter and policy can produce useful fills in one regime; it does not show performance across market conditions.

## Artifacts

- Dense metrics and all 28 per-window blocks: `data/benchmarks/forecast-models-dense-all-windows-2026-08-07.json`
- Causal forecasts: `data/benchmarks/forecast-model-forecasts-2026-08-07/*.json`
- LoRA selection: `data/benchmarks/chronos2-btc-lora-grid-pretest-validation-2026-08-07.json`
- Integration smokes: `data/benchmarks/*-bot-integration-smoke-2026-08-07.json`
- Frozen tuned-Chronos policy: `data/benchmarks/chronos2-btc-lora-bot-policy-2026-08-07.json`
- Tuned-Chronos held-out replay: `data/benchmarks/kronos-bot-validation-15ff9244cda5.json`
- Rejected-model calibration evidence: `data/benchmarks/*-bot-policy-2026-08-07.json.progress.json`
- Kronos report: `docs/experiments/kronos-report-2026-08-07.md`

## Current recommendation

Do not deploy any of these models as a standalone BTCUSDT 1m trading signal. Chronos-2 LoRA is the strongest point-forecast candidate and TiRex-2 is the strongest probabilistic candidate, but neither clears the persistence or oracle-distribution bars. The tuned-Chronos policy's held-out profit is concentrated entirely in one uptrend episode and fails the precommitted robustness gate. If retained, use their outputs only as candidate features in a separately calibrated ensemble whose final bot policy is evaluated under the existing sealed-window protocol.
