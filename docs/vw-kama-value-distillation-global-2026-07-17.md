# Volume-weighted KAMA oracle approximation search

Generated: 2026-07-17T18:36:37.400Z
Score version: 4
Raw results: `data/benchmarks/vw-kama-value-distillation-global-2026-07-17.jsonl`

The fit ranking never reads validation or holdout scores. The best validated volume-aware and canonical (both volume powers are zero) candidates are the only candidates evaluated on holdout.

## Observed outcome

- The volume-aware finalist returned **-0.74%** on the median holdout case; the canonical finalist returned **-0.18%**. Their median maximum drawdowns were 0.75% and 0.32% respectively.
- The executable hindsight oracle returned **+5,401.26%** on the same median holdout cases, with 0.35% median maximum drawdown. This is a perfect-information upper bound, not a tradable result.
- Oracle entropy is roughly 3.0 nats against a maximum of `log(21) = 3.045`, so `temperature=0.01` makes most target distributions nearly uniform. With the strategy represented by a fixed `sigma=0.15` Gaussian, cross-entropy is minimized by keeping its center near zero. The search therefore found low-exposure strategies instead of directional predictors.
- This pass validates the plumbing and exposes a loss-calibration failure. A useful next iteration should emit an actual learnable distribution (including width), reduce or adapt oracle temperature, or distill value/advantage rather than fitting a fixed-width Gaussian around the current discrete strategy output.

## Data and objective

- Source: `data/historical/spot-btcusdt/btcusdt/1m` (1m); target scales: 1m, 5m, 15m, 1h.
- Windows: fit 2025-03-19..2025-05-17, 2025-05-18..2025-07-16, 2025-07-17..2025-09-14, 2025-09-15..2025-11-13; validation 2025-11-14..2026-01-12, 2026-01-13..2026-03-13; test 2026-03-14..2026-05-12, 2026-05-13..2026-07-11.
- Each continuous segment reserves 3d before scoring.
- Candidate evaluation uses 12 persistent shared-memory worker threads, cross-generation score caching, and stage-wide prepared columnar candle/oracle caches.
- Generation zero evaluates all 4 warm genomes plus 384 Latin-hypercube genomes, then selects 384 by 70% score / 30% parameter novelty. Warm sources: `data/benchmarks/vw-kama-global-presets.json`.
- Search uses 2 independent restart(s), adaptive current-to-pbest differential evolution, family/agreement/confirmation-mask islands with cross-island migration, rotating fit folds with a full-fit pass every 4 generations, and 3 shrinking elite-refinement round(s).
- Signal: completed-candle volume-weighted KAMA derivative rate with flat, hold, or hysteresis state handling. ER optionally weights every price move by `(volume / causal volume EMA)^ER volume power`; zero recovers standard ER. A second volume-aware KAMA supplies the mean-reversion baseline: it has independent ER/fast/slow periods and shares the strategy's ER-volume, KAMA-power, and post-ER volume behavior. Its causal volatility-normalized distance follows KAMA below the suppression threshold, goes flat between thresholds, and reverses KAMA at the reversal threshold. A causal logistic confirmation combines KAMA acceleration, price overextension, independent slow-EMA trend, RSI, and ADX-strength-weighted DMI direction, then scales or filters the signal. Sizing mode price-marks the fraction, while confidence mode holds it as uncertainty until the next signal.
- Signal memory: after the first signal, a candidate state change emits only when the current close is strictly more than 17.5 bps from the last emitted signal price; rejected changes retain the prior state. This is the same friction used by the oracle.
- Matching is one chronological one-to-one alignment by resulting state. It maximizes total timing credit, so extra candidate transitions reduce precision and uncovered oracle transitions reduce recall.
- Search objective: value-distillation; negative weighted cross-entropy -CE(p_oracle, s_candidate); p is derived from friction-aware future value and weighted by max(Q)-min(Q). Cleanliness is matched / (matched + extra); the displayed noise/signal ratio is extra / matched.
- Oracle exposures: 21 points over [-1, 1], value temperature 0.01, candidate sigma 0.15; opportunity weight is max(Q)-min(Q)+0.000001.
- Candidate fitness is the negative of median/P90 cross-entropy, equally weighted; every scale/window case has equal weight.

## Holdout finalists

| family | id | strategy | robust CE | median CE | P90 CE | median KL | exp(-KL) | strategy return | oracle return | strategy DD | oracle DD | signal score | agreement | signals/day |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| volume | v00031 | confidence | 9.74408 | 9.50577 | 9.98238 | 6.48174 | 0.15% | -0.74% | 5401.26% | 0.75% | 0.35% | 10.65% | 0.15% | 23.26 |
| canonical | k00044 | sizing | 9.74948 | 9.50677 | 9.9922 | 6.48273 | 0.15% | -0.18% | 5401.26% | 0.32% | 0.35% | 11.16% | 0.21% | 0.94 |

## Validation finalists

| family | id | strategy | robust CE | median CE | P90 CE | median KL | exp(-KL) | strategy return | oracle return | strategy DD | oracle DD | signal score | agreement | signals/day |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical | k00044 | sizing | 9.92059 | 9.57434 | 10.26685 | 6.55963 | 0.14% | -0.05% | 33002.75% | 0.25% | 0.35% | 10.64% | 0.1% | 0.72 |
| volume | v00031 | confidence | 9.92061 | 9.57419 | 10.26703 | 6.55948 | 0.14% | -0.97% | 33002.75% | 0.97% | 0.35% | 11.66% | 0.14% | 28.71 |
| volume | v00373 | confidence | 9.92152 | 9.5744 | 10.26865 | 6.55969 | 0.14% | 0% | 33002.75% | 0% | 0.35% | 20% | 0% | 0 |
| volume | v00030 | confidence | 9.92166 | 9.57428 | 10.26905 | 6.55957 | 0.14% | -0.23% | 33002.75% | 0.24% | 0.35% | 13.77% | 0.04% | 20.9 |
| volume | v00004 | sizing | 9.92179 | 9.57441 | 10.26917 | 6.5597 | 0.14% | 0% | 33002.75% | 0% | 0.35% | 11.13% | 0% | 10.4 |
| canonical | k00047 | sizing | 9.92226 | 9.5744 | 10.27013 | 6.55969 | 0.14% | 0% | 33002.75% | 0% | 0.35% | 16.69% | 0% | 0.02 |
| canonical | k00065 | confidence | 9.92226 | 9.5744 | 10.27013 | 6.55969 | 0.14% | 0% | 33002.75% | 0% | 0.35% | 20% | 0% | 0 |
| volume | v00021 | confidence | 9.92283 | 9.57348 | 10.27218 | 6.55877 | 0.14% | -0.97% | 33002.75% | 0.99% | 0.35% | 12.02% | 0.17% | 8.86 |
| canonical | k00049 | confidence | 9.92294 | 9.57542 | 10.27047 | 6.56071 | 0.14% | -0.16% | 33002.75% | 0.23% | 0.35% | 11.35% | 0.09% | 3.88 |
| canonical | k00063 | confidence | 9.92299 | 9.57548 | 10.27049 | 6.56078 | 0.14% | -0.54% | 33002.75% | 0.59% | 0.35% | 11.19% | 0.11% | 5.75 |
| canonical | k00105 | confidence | 9.92383 | 9.57702 | 10.27064 | 6.56231 | 0.14% | -0.49% | 33002.75% | 0.52% | 0.35% | 8.26% | 0.08% | 2.43 |
| volume | v00020 | confidence | 9.93632 | 9.57471 | 10.29793 | 6.56 | 0.14% | -0.01% | 33002.75% | 0.02% | 0.35% | 10.15% | 0.01% | 10.44 |

## Best fit candidates

| family | id | strategy | robust CE | median CE | P90 CE | median KL | exp(-KL) | strategy return | oracle return | strategy DD | oracle DD | signal score | agreement | signals/day |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| volume | v00020 | confidence | 9.70385 | 9.48415 | 9.92354 | 6.45761 | 0.16% | 0% | 2793.73% | 0.05% | 0.35% | 9.25% | 0.01% | 9.13 |
| volume | v00373 | confidence | 9.70465 | 9.48452 | 9.92478 | 6.45797 | 0.16% | 0% | 2793.73% | 0% | 0.35% | 20% | 0% | 0 |
| canonical | k00105 | confidence | 9.70489 | 9.48633 | 9.92345 | 6.45978 | 0.16% | -0.42% | 2793.73% | 0.46% | 0.35% | 7.24% | 0.19% | 1.91 |
| volume | v00021 | confidence | 9.70505 | 9.4823 | 9.92779 | 6.45576 | 0.16% | -0.74% | 2793.73% | 0.75% | 0.35% | 10.85% | 0.24% | 6.93 |
| volume | v00031 | confidence | 9.70538 | 9.48335 | 9.92742 | 6.4568 | 0.16% | -0.85% | 2793.73% | 0.86% | 0.35% | 9.99% | 0.16% | 20.52 |
| volume | v00004 | sizing | 9.70608 | 9.48452 | 9.92764 | 6.45797 | 0.16% | 0% | 2793.73% | 0% | 0.35% | 10.24% | 0% | 8.86 |
| volume | v00030 | confidence | 9.70693 | 9.48418 | 9.92969 | 6.45763 | 0.16% | -0.18% | 2793.73% | 0.18% | 0.35% | 11.31% | 0.04% | 16.25 |
| canonical | k00049 | confidence | 9.70694 | 9.48292 | 9.93097 | 6.45637 | 0.16% | -0.11% | 2793.73% | 0.18% | 0.35% | 11.06% | 0.15% | 2.44 |
| canonical | k00047 | sizing | 9.70699 | 9.48303 | 9.93096 | 6.45648 | 0.16% | 0% | 2793.73% | 0% | 0.35% | 16.69% | 0% | 0.03 |
| canonical | k00044 | sizing | 9.70713 | 9.48412 | 9.93015 | 6.45757 | 0.16% | 0% | 2793.73% | 0.17% | 0.35% | 11.06% | 0.15% | 0.71 |
| canonical | k00063 | confidence | 9.70715 | 9.48355 | 9.93076 | 6.457 | 0.16% | -0.34% | 2793.73% | 0.4% | 0.35% | 11.02% | 0.24% | 2.99 |
| canonical | k00065 | confidence | 9.70716 | 9.48336 | 9.93096 | 6.45681 | 0.16% | 0% | 2793.73% | 0% | 0.35% | 20% | 0% | 0 |

## Finalist parameters

| family | id | agreement | efficiency | ER volume EMA/power | fast | slow | power | volume | cap | volume power | base threshold | state mode | noise lookback | noise multiplier | buy max | sell max | buy sigma | sell sigma |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| volume | v00031 | confidence | 60000ms | 3939234ms / 2.666 | 5463ms | 5232333ms | 1.123 | 2419983ms | 2.64 | 1.915 | 69.396 bps/hour | flat | 75391753ms | 0.104 | 8.23% | 97.8% | 58.573 | 14.743 |
| canonical | k00044 | sizing | 1085736ms | 1m / 0 | 29017ms | 86400000ms | 0.713 | 1m | 1 | 0 | 12.835 bps/hour | hold | 11472165ms | 3.157 | 5% | 54.28% | 251.505 | 0.459 |

## Finalist confirmation parameters

| candidate | mix | minimum quality | acceleration lookback | distance lookback | acceleration weight | overextension weight | bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| v00031 | 51.48% | 35.46% | 21600000ms | 63988ms | 0 | 0 | 0.496 |
| k00044 | 16.88% | 95% | 327770ms | 21600000ms | 5 | 5 | -3.519 |

| candidate | hysteresis release | slow EMA | EMA tolerance | EMA weight/gate | RSI period | RSI tolerance/weight | DMI period | DMI weight | ADX threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v00031 | 22.64% | 5983593ms | 43.607 bps/h | 1.183 / 2.97% | 297366ms | 5.45 / 3.155 | 715513ms | 1.776 | 9.35 |
| k00044 | 0% | 15483049ms | 19.815 bps/h | 4.401 / 66.36% | 21600000ms | 20 / 0.05 | 1028342ms | 0 | 31.3 |

## Finalist mean-reversion parameters

| candidate | ER | fast | slow | volatility | suppress at | reverse at |
|---|---:|---:|---:|---:|---:|---:|
| v00031 | 1m | 1m | 5m | 1m | 0σ | 0σ |
| k00044 | 15640074ms | 407654ms | 407654ms | 20048561ms | 0.93σ | 2.235σ |

## Holdout cases

| candidate | scale/window | cross-entropy | KL | exp(-KL) | strategy return | oracle return | strategy DD | oracle DD | opportunity | signal score | agreement | signals/day |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v00031 | 1m:test-1 | 9.38688 | 6.34672 | 0.18% | -0.59% | 14616.59% | 0.59% | 0.35% | 0.002833 | 11.13% | 0.05% | 56.4 |
| v00031 | 1m:test-2 | 9.38497 | 6.34518 | 0.18% | -0.49% | 44325.5% | 0.49% | 0.35% | 0.002894 | 12.32% | 0.03% | 62.95 |
| v00031 | 5m:test-1 | 9.44331 | 6.41067 | 0.16% | -1.15% | 5815.67% | 1.16% | 0.35% | 0.003387 | 10.67% | 0.17% | 31.4 |
| v00031 | 5m:test-2 | 9.45046 | 6.41933 | 0.16% | -1.46% | 12450.84% | 1.46% | 0.35% | 0.003526 | 11.61% | 0.14% | 35.14 |
| v00031 | 15m:test-1 | 9.56109 | 6.54415 | 0.14% | -0.9% | 2566.55% | 0.91% | 0.35% | 0.004299 | 9.56% | 0.27% | 13.63 |
| v00031 | 15m:test-2 | 9.60704 | 6.5971 | 0.14% | -1.22% | 4986.85% | 1.23% | 0.35% | 0.00458 | 10.62% | 0.24% | 15.11 |
| v00031 | 1h:test-1 | 9.91107 | 6.94463 | 0.1% | -0.22% | 889.33% | 0.26% | 0.35% | 0.006741 | 5.87% | 0.04% | 1.95 |
| v00031 | 1h:test-2 | 10.14878 | 7.219 | 0.07% | -0.04% | 1425.94% | 0.13% | 0.35% | 0.007415 | 6.74% | 0.18% | 2.64 |
| k00044 | 1m:test-1 | 9.38697 | 6.34681 | 0.18% | -0.39% | 14616.59% | 0.4% | 0.35% | 0.002833 | 11.31% | 0.13% | 5.62 |
| k00044 | 1m:test-2 | 9.38511 | 6.34533 | 0.18% | -0.35% | 44325.5% | 0.37% | 0.35% | 0.002894 | 12.15% | 0.09% | 6 |
| k00044 | 5m:test-1 | 9.44387 | 6.41123 | 0.16% | -0.23% | 5815.67% | 0.32% | 0.35% | 0.003387 | 11% | 0.39% | 1.85 |
| k00044 | 5m:test-2 | 9.45059 | 6.41946 | 0.16% | -0.19% | 12450.84% | 0.32% | 0.35% | 0.003526 | 11.93% | 0.29% | 1.6 |
| k00044 | 15m:test-1 | 9.56295 | 6.54601 | 0.14% | -0.01% | 2566.55% | 0.21% | 0.35% | 0.004299 | 8.56% | 0.5% | 0.28 |
| k00044 | 15m:test-2 | 9.60698 | 6.59704 | 0.14% | -0.1% | 4986.85% | 0.13% | 0.35% | 0.00458 | 8.09% | 0.11% | 0.17 |
| k00044 | 1h:test-1 | 9.91546 | 6.94902 | 0.1% | 0% | 889.33% | 0% | 0.35% | 0.006741 | 20% | 0% | 0 |
| k00044 | 1h:test-2 | 10.17126 | 7.24148 | 0.07% | -0.18% | 1425.94% | 0.66% | 0.35% | 0.007415 | 0.64% | 1.06% | 0.02 |
