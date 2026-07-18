# Volume-weighted KAMA oracle approximation search

Generated: 2026-07-18T11:00:28.873Z
Score version: 8
Raw results: `data/benchmarks/vw-kama-value-distillation-global-2026-07-18.jsonl`

The fit ranking never reads validation or holdout scores. Holdout evaluates only the best validated volume-aware/canonical candidates and the fixed current-production signal baseline.

## Data and objective

- Source: `data/historical/spot-btcusdt/btcusdt/1m` (1m); target scales: 1m, 5m, 15m, 1h.
- Windows: fit 2025-03-19..2025-05-17, 2025-05-18..2025-07-16, 2025-07-17..2025-09-14, 2025-09-15..2025-11-13; validation 2025-11-14..2026-01-12, 2026-01-13..2026-03-13; test 2026-03-14..2026-05-12, 2026-05-13..2026-07-11.
- Each continuous segment reserves 3d before scoring.
- Candidate evaluation uses 12 persistent shared-memory worker threads, cross-generation score caching, and stage-wide prepared columnar candle/oracle caches.
- Generation zero evaluates all 4 warm genomes plus 384 Latin-hypercube genomes, then selects 384 by 70% score / 30% parameter novelty. Warm sources: `data/benchmarks/vw-kama-global-presets.json`.
- Search uses 2 independent restart(s), adaptive current-to-pbest differential evolution, family/agreement/confirmation-mask islands with cross-island migration, rotating fit folds with a full-fit pass every 4 generations, and 3 shrinking elite-refinement round(s).
- Signal: completed-candle volume-weighted KAMA derivative rate with flat, hold, or hysteresis state handling. The rate calculation, threshold clamp, adaptive-noise threshold, and consumed-edge friction memory are shared directly with the live PeakValleyStrategy. ER optionally weights every price move by `(volume / causal volume EMA)^ER volume power`; zero recovers standard ER. A second volume-aware KAMA supplies the optional mean-reversion baseline: it has independent ER/fast/slow periods and shares the strategy's ER-volume, KAMA-power, and post-ER volume behavior. Its causal volatility-normalized distance follows KAMA below the suppression threshold, goes flat between thresholds, and reverses KAMA at the reversal threshold. A causal logistic confirmation can combine KAMA acceleration, price overextension, independent slow-EMA trend, RSI, and ADX-strength-weighted DMI direction, then scale or filter the signal. Those experimental layers are disabled for `production-current`. Sizing mode price-marks the fraction, while confidence mode holds it as uncertainty until the next signal.
- Signal memory: after the first signal, a candidate state change emits only when the current close moves strictly past oracle friction × its searched candidate fraction (`0..1`); rejected changes retain the prior state. Oracle friction is 17.5 bps.
- Matching is one chronological one-to-one alignment by resulting state. It maximizes total timing credit, so extra candidate transitions reduce precision and uncovered oracle transitions reduce recall.
- Search objective: value-distillation; negative weighted cross-entropy -CE(p_oracle, s_candidate); p is derived from friction-aware future value and weighted by max(Q)-min(Q). Cleanliness is matched / (matched + extra); the displayed noise/signal ratio is extra / matched.
- Oracle exposures: 150 tradable targets over [-100, 100] with |effective exposure| <= 250, 1s fixed forced-exposure horizon, value temperature 0.01; candidate strategy temperature 0.000001..0.1, quadratic scale 0..1000000, and quadratic volatility window 1m..1d; opportunity weight is max(Q)-min(Q)+0.000001.
- Candidate fitness is the negative of median/P90 cross-entropy, equally weighted; every scale/window case has equal weight.

## Holdout finalists and production signal baseline

| family | id | strategy | robust CE | median CE | P90 CE | median KL | exp(-KL) | strategy return | oracle return | strategy DD | oracle DD | signal score | agreement | signals/day |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| volume | v00023 | confidence | 5.80796 | 5.02601 | 6.58991 | 2.62342 | 7.35% | 0% | 3.5728421555697718e+143% | 0.33% | 26.55% | 18.17% | 0.32% | 0.08 |
| canonical | k00033 | sizing | 5.91862 | 5.03448 | 6.80277 | 2.62405 | 7.35% | 0% | 3.5728421555697718e+143% | 0% | 26.55% | 20% | 0% | 0 |
| canonical | production-current | sizing | 238.86469 | 110.97159 | 366.75779 | 108.56115 | 0% | -93.91% | 3.5728421555697718e+143% | 93.95% | 26.55% | 34.73% | 39.32% | 22.49 |

## Validation finalists

| family | id | strategy | robust CE | median CE | P90 CE | median KL | exp(-KL) | strategy return | oracle return | strategy DD | oracle DD | signal score | agreement | signals/day |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| volume | v00023 | confidence | 6.04309 | 5.02187 | 7.06431 | 2.77613 | 6.27% | 0% | 1.903971301373628e+197% | 0.76% | 28.24% | 19.02% | 0.82% | 0.08 |
| volume | v00029 | confidence | 6.04346 | 5.02092 | 7.066 | 2.77504 | 6.27% | 0% | 1.903971301373628e+197% | 0% | 28.24% | 20% | 0% | 0 |
| volume | v00002 | sizing | 6.04396 | 5.0217 | 7.06622 | 2.77578 | 6.27% | 0% | 1.903971301373628e+197% | 0% | 28.24% | 20% | 0% | 0 |
| volume | v00316 | confidence | 6.04686 | 5.02771 | 7.066 | 2.782 | 6.23% | 0% | 1.903971301373628e+197% | 0% | 28.24% | 18.72% | 0% | 0.04 |
| volume | v00134 | confidence | 6.05252 | 5.03148 | 7.07356 | 2.78574 | 6.21% | 0% | 1.903971301373628e+197% | 0% | 28.24% | 20% | 0% | 0 |
| volume | v00353 | confidence | 6.0603 | 5.04612 | 7.07448 | 2.79973 | 6.14% | -67.5% | 1.903971301373628e+197% | 67.71% | 28.24% | 11.99% | 8.84% | 19.77 |
| canonical | k00033 | sizing | 6.16177 | 5.02914 | 7.29441 | 2.78343 | 6.22% | 0% | 1.903971301373628e+197% | 0% | 28.24% | 20% | 0% | 0 |
| canonical | k00036 | sizing | 6.16177 | 5.02914 | 7.29441 | 2.78343 | 6.22% | 0% | 1.903971301373628e+197% | 0% | 28.24% | 20% | 0% | 0 |
| canonical | k00043 | sizing | 6.16177 | 5.02914 | 7.29441 | 2.78343 | 6.22% | -1.04% | 1.903971301373628e+197% | 5.03% | 28.24% | 17.69% | 2.5% | 0.16 |
| canonical | k00046 | sizing | 6.16177 | 5.02914 | 7.29441 | 2.78343 | 6.22% | 0% | 1.903971301373628e+197% | 0% | 28.24% | 20% | 0% | 0 |
| canonical | k00159 | sizing | 6.1618 | 5.02919 | 7.29441 | 2.78349 | 6.22% | 0% | 1.903971301373628e+197% | 0% | 28.24% | 20% | 0% | 0 |
| canonical | k00259 | sizing | 6.16351 | 5.03261 | 7.29441 | 2.78691 | 6.21% | 0% | 1.903971301373628e+197% | 0% | 28.24% | 20% | 0% | 0 |
| canonical | production-current | sizing | 301.04929 | 147.74512 | 454.35346 | 145.49941 | 0% | -97.44% | 1.903971301373628e+197% | 97.45% | 28.24% | 38.13% | 41.83% | 27.38 |

## Best fit candidates

| family | id | strategy | robust CE | median CE | P90 CE | median KL | exp(-KL) | strategy return | oracle return | strategy DD | oracle DD | signal score | agreement | signals/day |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| volume | v00002 | sizing | 5.70404 | 5.01549 | 6.3926 | 2.51407 | 8.11% | 0% | 1.912962460302393e+118% | 0% | 27.24% | 20% | 0% | 0 |
| volume | v00316 | confidence | 5.70484 | 5.01678 | 6.3929 | 2.51406 | 8.11% | 0% | 1.912962460302393e+118% | 0.88% | 27.24% | 13.87% | 1.75% | 0.07 |
| volume | v00023 | confidence | 5.7077 | 5.01495 | 6.40045 | 2.51434 | 8.11% | 0% | 1.912962460302393e+118% | 0.26% | 27.24% | 20% | 0.54% | 0.03 |
| volume | v00029 | confidence | 5.70809 | 5.01525 | 6.40093 | 2.51367 | 8.11% | 0% | 1.912962460302393e+118% | 0% | 27.24% | 20% | 0% | 0 |
| volume | v00134 | confidence | 5.71453 | 5.01849 | 6.41058 | 2.51937 | 8.07% | 0% | 1.912962460302393e+118% | 0% | 27.24% | 20% | 0% | 0 |
| volume | v00353 | confidence | 5.72123 | 5.02338 | 6.41908 | 2.52689 | 8.02% | -44.57% | 1.912962460302393e+118% | 44.72% | 27.24% | 9.95% | 7.99% | 12.05 |
| volume | v00247 | confidence | 5.72439 | 5.02718 | 6.42159 | 2.53203 | 7.98% | -11.29% | 1.912962460302393e+118% | 11.29% | 27.24% | 8.09% | 3.15% | 6.41 |
| volume | v00009 | sizing | 5.72487 | 5.01686 | 6.43288 | 2.5147 | 8.11% | -0.99% | 1.912962460302393e+118% | 1.76% | 27.24% | 7.79% | 1.16% | 1.25 |
| volume | v00227 | sizing | 5.7322 | 5.07369 | 6.39072 | 2.55806 | 7.78% | 0% | 1.912962460302393e+118% | 0% | 27.24% | 20% | 0% | 0 |
| volume | v00121 | sizing | 5.73444 | 5.07455 | 6.39434 | 2.558 | 7.78% | 0% | 1.912962460302393e+118% | 0% | 27.24% | 20% | 0% | 0 |
| volume | v00075 | sizing | 5.74888 | 5.08844 | 6.40932 | 2.57098 | 7.68% | 0% | 1.912962460302393e+118% | 0% | 27.24% | 20% | 0% | 0 |
| volume | v00366 | sizing | 5.75461 | 5.09185 | 6.41736 | 2.57435 | 7.65% | 0% | 1.912962460302393e+118% | 0% | 27.24% | 20% | 0% | 0 |

## Finalist parameters

| family | id | agreement | efficiency | ER volume EMA/power | fast | slow | power | volume | cap | volume power | base threshold | state mode | noise lookback | noise multiplier | buy max | sell max | buy sigma | sell sigma |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| volume | v00023 | confidence | 7200000ms | 1m / 0 | 1800000ms | 24165856ms | 5 | 164680ms | 1 | 2.254 | 0.05 bps/hour | hold | 85828612ms | 8 | 5% | 7.02% | 0.01 | 241.663 |
| canonical | k00033 | sizing | 7200000ms | 1m / 0 | 1797341ms | 86400000ms | 5 | 1m | 1 | 0 | 2000 bps/hour | flat | 300000ms | 0 | 100% | 16.15% | 20.625 | 19.43 |
| canonical | production-current | sizing | 14m | 1h / 0 | 28m | 153m | 0.49 | 130m | 2.65 | 0 | 67.567 bps/hour | flat | 1h | 0 | 100% | 100% | 1000000000000 | 1000000000000 |

## Finalist distribution and friction parameters

| candidate | noise response | candidate friction | strategy temperature | quadratic scale | quadratic volatility window |
|---|---|---:|---:|---:|---:|
| v00023 | inverse | 0.01119 | 0.1 | 0 | 69987169ms |
| k00033 | inverse | 0 | 0.1 | 0 | 115327ms |
| production-current | proportional | 1 | 0.001 | 0 | 1h |

## Finalist confirmation parameters

| candidate | mix | minimum quality | acceleration lookback | distance lookback | acceleration weight | overextension weight | bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| v00023 | 7.61% | 2.03% | 1992471ms | 21600000ms | 2.538 | 0.464 | 4.671 |
| k00033 | 0% | 0% | 3270829ms | 21600000ms | 0 | 0 | 5 |
| production-current | 0% | 0% | 1h | 1h | 1 | 1 | 0 |

| candidate | hysteresis release | slow EMA | EMA tolerance | EMA weight/gate | RSI period | RSI tolerance/weight | DMI period | DMI weight | ADX threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v00023 | 99.93% | 328719ms | 281.544 bps/h | 0 / 0% | 6490237ms | 16.7 / 4.915 | 20238629ms | 0 | 39.48 |
| k00033 | 100% | 300000ms | 0.1 bps/h | 0 / 0% | 21600000ms | 12.45 / 0 | 120000ms | 0 | 50 |
| production-current | 0% | 1h | 0 bps/h | 0 / 0% | 14m | 0 / 0 | 14m | 0 | 20 |

## Finalist mean-reversion parameters

| candidate | ER | fast | slow | volatility | suppress at | reverse at |
|---|---:|---:|---:|---:|---:|---:|
| v00023 | 1m | 1m | 5m | 1m | 0σ | 0σ |
| k00033 | 334953ms | 21333568ms | 21333568ms | 9609447ms | 0.73σ | 1.993σ |
| production-current | 1h | 15m | 1h | 1h | 1σ | 0σ |

## Holdout cases

| candidate | scale/window | H | cross-entropy | KL | exp(-KL) | strategy return | oracle return | strategy DD | oracle DD | opportunity | signal score | agreement | signals/day |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v00023 | 1m:test-1 | 1m | 5.01063 | 2.07991 | 12.49% | 0% | 3.8735118007927644e+170% | 0% | 45.21% | 0.209385 | 20% | 0% | 0 |
| v00023 | 1m:test-2 | 1m | 5.01063 | 2.11542 | 12.06% | 0% | 4.728715252228284e+211% | 0% | 35.31% | 0.218138 | 20% | 0% | 0 |
| v00023 | 5m:test-1 | 5m | 5.01067 | 2.38762 | 9.18% | 0% | 7.145684311133585e+143% | 0% | 29.82% | 0.28039 | 20% | 0% | 0 |
| v00023 | 5m:test-2 | 5m | 5.01066 | 2.46164 | 8.53% | 0% | 5.524825742681639e+169% | 0% | 26.83% | 0.304011 | 20% | 0% | 0 |
| v00023 | 15m:test-1 | 15m | 5.05704 | 2.78519 | 6.17% | -0.87% | 1.0315535176719662e+112% | 1.23% | 26.27% | 0.421175 | 14.39% | 0.64% | 0.17 |
| v00023 | 15m:test-2 | 15m | 5.04134 | 2.86303 | 5.71% | 0.92% | 5.958536947525781e+131% | 0.66% | 26.12% | 0.477944 | 16.35% | 2.11% | 0.2 |
| v00023 | 1h:test-1 | 1h | 6.13897 | 4.31058 | 1.34% | -3.33% | 6.470510885381901e+72% | 3.33% | 25.97% | 0.905198 | 12.12% | 1.71% | 3.62 |
| v00023 | 1h:test-2 | 1h | 7.6421 | 5.8878 | 0.28% | -2.82% | 1.265787967519293e+83% | 2.9% | 25.97% | 1.019082 | 11.97% | 1.77% | 4.05 |
| k00033 | 1m:test-1 | 1m | 5.01063 | 2.07991 | 12.49% | 0% | 3.8735118007927644e+170% | 0% | 45.21% | 0.209385 | 20% | 0% | 0 |
| k00033 | 1m:test-2 | 1m | 5.01063 | 2.11542 | 12.06% | 0% | 4.728715252228284e+211% | 0% | 35.31% | 0.218138 | 20% | 0% | 0 |
| k00033 | 5m:test-1 | 5m | 5.01069 | 2.38764 | 9.18% | 0% | 7.145684311133585e+143% | 0% | 29.82% | 0.28039 | 20% | 0% | 0 |
| k00033 | 5m:test-2 | 5m | 5.01072 | 2.4617 | 8.53% | 0% | 5.524825742681639e+169% | 0% | 26.83% | 0.304011 | 20% | 0% | 0 |
| k00033 | 15m:test-1 | 15m | 5.05825 | 2.7864 | 6.16% | 0% | 1.0315535176719662e+112% | 0% | 26.27% | 0.421175 | 20% | 0% | 0 |
| k00033 | 15m:test-2 | 15m | 5.05875 | 2.88044 | 5.61% | 0% | 5.958536947525781e+131% | 0% | 26.12% | 0.477944 | 20% | 0% | 0 |
| k00033 | 1h:test-1 | 1h | 6.42233 | 4.59395 | 1.01% | 0% | 6.470510885381901e+72% | 0% | 25.97% | 0.905198 | 20% | 0% | 0 |
| k00033 | 1h:test-2 | 1h | 7.69045 | 5.93615 | 0.26% | 0% | 1.265787967519293e+83% | 0% | 25.97% | 1.019082 | 20% | 0% | 0 |
| production-current | 1m:test-1 | 1m | 15.61741 | 12.68669 | 0% | -99.69% | 3.8735118007927644e+170% | 99.69% | 45.21% | 0.209385 | 34.99% | 39.51% | 41.65 |
| production-current | 1m:test-2 | 1m | 16.86959 | 13.97437 | 0% | -99.76% | 4.728715252228284e+211% | 99.76% | 35.31% | 0.218138 | 36.7% | 39.97% | 45.11 |
| production-current | 5m:test-1 | 5m | 57.91172 | 55.28867 | 0% | -97.91% | 7.145684311133585e+143% | 97.91% | 29.82% | 0.28039 | 35.73% | 42.19% | 28.63 |
| production-current | 5m:test-2 | 5m | 65.91256 | 63.36354 | 0% | -98.7% | 5.524825742681639e+169% | 98.71% | 26.83% | 0.304011 | 37.56% | 43.32% | 31.42 |
| production-current | 15m:test-1 | 15m | 156.03061 | 153.75876 | 0% | -86.22% | 1.0315535176719662e+112% | 86.23% | 26.27% | 0.421175 | 32.65% | 37.43% | 14.27 |
| production-current | 15m:test-2 | 15m | 182.85759 | 180.67928 | 0% | -89.91% | 5.958536947525781e+131% | 89.99% | 26.12% | 0.477944 | 34.46% | 39.12% | 16.34 |
| production-current | 1h:test-1 | 1h | 312.57218 | 310.7438 | 0% | -29% | 6.470510885381901e+72% | 30.7% | 25.97% | 0.905198 | 19.37% | 21.46% | 2.48 |
| production-current | 1h:test-2 | 1h | 493.19087 | 491.43657 | 0% | -18.83% | 1.265787967519293e+83% | 21.3% | 25.97% | 1.019082 | 18.06% | 18.83% | 3.09 |
