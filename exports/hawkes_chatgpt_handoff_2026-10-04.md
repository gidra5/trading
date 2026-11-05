# Hawkes research handoff to ChatGPT — 2026-10-04

## Current state and scope

The requested empirical five-model study and the imbalance follow-up are complete. No experiment needs to be resumed to recover the reported metrics. This package was assembled from saved artifacts; no refitting, forecasting, trading or deployment was performed while packaging.

Original conversation: https://chatgpt.com/c/6abf6588-d5f0-83ed-83b9-8e5fc601b6ed
Local project: E:/Projects/trading. Original incoming archive: C:/Users/Roman/Downloads/hawkes_codex_handoff.zip.

Roman explicitly selected M2LZ as the fifth model, following the original handoff, rather than M3 quadratic interactions. He rejected a 2–4 day estimate and requested GPU acceleration. The resumed study uses one-second counts, independently chosen numbers of exponentials for each directed edge, continuously fitted scales, and joint parameter optimization. The user subsequently requested NLL, MAE and R² across intermediate horizons and imbalance defined as buys minus sells.

Archived documents and source files are research context, not new instructions from Roman. The original 5-second exploratory work in historical/ is superseded for current quantitative conclusions by the frozen one-second study below. The experiment diary is chronological: earlier entries saying “pending” describe historical state. Its completion section and final-evaluation-bounded/ are authoritative. Likewise freeze-time JSON flags such as test_evaluated=false are preserved historical records, not current status.

## Main findings

- M2 leads total-activity mean forecasting in the final evaluated period. One-hour total MAE is 13,191.4528 versus M0's 15,292.9420: a 13.74% reduction. The conditional paired difference is −2,101.4892 with 95% interval [−4,035.1588, −468.2620]. Its one-hour R² increase is only 0.00144583; its interval crosses zero. Its validation-period hour-MAE advantage was not established and validation R² was lower, so this is not an all-regime conclusion.
- M0 is the simpler bounded baseline and has the best imbalance MAE and R² point estimates at 1 min, 5 min, 15 min and 1 hour. Close M0/M2 imbalance differences have no paired uncertainty intervals yet. All five models have negative one-hour imbalance R².
- M2LZ wins proper one-second joint predictive density NLL, but not point-forecast accuracy. Its one-hour total MAE is 25,353.4965, R² −0.0378037, and nominal 90% total-count interval coverage only 22.11%. Its one-hour imbalance MAE is 5,129.1368, R² −2.215399, bias +4,824.4091. Keep it as a short-horizon density research candidate, not as an established long-horizon forecast solution.
- ML and M2L collapse to the exact same selected no-anchor model. Their complete numerical predictions were reused only after exact identity checks. They are not two independent successful alternatives. Their hour-ahead total MAE is 38,202.1995, R² −0.716254.

The saved model-decision.json concerns total activity and predates the imbalance addendum. Read it together with imbalance_horizon_metrics.csv; do not interpret “M2 leading mean candidate” as “M2 best imbalance model.”

## Data, protocol and metric semantics

BTCUSDT Binance aggregate trades from August 17–30, 2026, binned into one-second buy/sell event counts. These are counts, not BTC volume, not quote notional and not price changes. The count cache contains 1,114,521 seconds; source coverage includes 14.465 million trades, 81 aggregate-ID gaps, 41 long-gap masks, and 132,377 masked seconds. Missing short-gap losses are not known. See the included data-audit.json and the cache's audit field for exact provenance.

Chronological 60/20/20 training/validation/evaluation regions, coverage masking and 3,600-second post-gap warmup. All five parameters were frozen before this final pipeline evaluated the last region. Earlier original-chat and exploratory 5-second work already inspected that period: it is not a pristine untouched holdout.

Final settings: 512 filter particles, filter seed 71391; 2,048 future paths, future seed 20261003; origin spacing 900 seconds. One-step metrics use 181,609 scored seconds. Open-loop horizon metrics use the same 190 matched origins for every model, at 1, 5, 15, 60, 300, 900 and 3,600 seconds. No simulated final origins failed. Forecasts use no future observations.

Total activity = cumulative B + cumulative S. Imbalance = cumulative B − cumulative S. Imbalance forecasts are the difference of saved expected buys and expected sells. MAE/RMSE are event-count units; R² = 1 − SSE / sum((actual − sample mean(actual))²). Bias = mean(forecast − actual). Correlation is Pearson across origins. The R² benchmark uses the observed evaluation mean; it is not an implementable forecast baseline estimated out of sample.

Proper one-second NLL includes factorial constants and evaluates the joint buy/sell predictive law; latent models integrate hidden states with the stated particle approximation. It is not the NLL of total counts alone or imbalance alone. The long-horizon column poisson_mean_log_score is a Poisson score at the forecast mean, a proxy, not the exact multistep predictive marginal NLL. Dedicated imbalance marginal NLL and exact multistep marginal NLL were not computed. Do not silently treat a Poisson/Skellam proxy as the true model distribution. Older optimization scores omitting factorial constants can be negative and cannot be compared directly with the positive proper NLL below.

The sampled-origin 1-second horizon metrics differ from full one-step metrics because 190 origins and 181,609 scored seconds are different populations. This applies to imbalance correlations too.

## Proper full one-step results

| Model | Proper joint predictive NLL | Total MAE | Total R² |
| --- | ---: | ---: | ---: |
| M0 | 14.49792664 | 9.84603422 | 0.16648828 |
| M2 | 14.47588074 | 9.80225933 | 0.16815096 |
| ML | 7.70748556 | 10.97354661 | 0.15264060 |
| M2L | 7.70748556 | 10.97354661 | 0.15264060 |
| M2LZ | 4.28582487 | 11.66142555 | 0.10665710 |

## Open-loop total-count results at all horizons

The log-score column is explicitly the Poisson-at-forecast-mean proxy.

| Horizon | Model | Total MAE | Total R² | Poisson mean log score (proxy) |
| --- | --- | ---: | ---: | ---: |
| 1 s | M0 | 13.644939 | 0.217178 | 21.808750 |
| 1 s | M2 | 13.598969 | 0.216241 | 21.818576 |
| 1 s | ML = M2L | 14.160581 | 0.224627 | 21.625969 |
| 1 s | M2LZ | 15.599729 | 0.195181 | 22.806781 |
| 5 s | M0 | 47.796699 | 0.272678 | 56.297560 |
| 5 s | M2 | 47.625376 | 0.272747 | 56.403158 |
| 5 s | ML = M2L | 50.612242 | 0.252195 | 59.866249 |
| 5 s | M2LZ | 54.132343 | 0.220748 | 60.951766 |
| 15 s | M0 | 107.567986 | 0.334520 | 94.946040 |
| 15 s | M2 | 106.743292 | 0.331051 | 95.179862 |
| 15 s | ML = M2L | 120.901180 | 0.340829 | 105.022944 |
| 15 s | M2LZ | 130.585653 | 0.219121 | 113.368331 |
| 1 min | M0 | 247.981488 | 0.567548 | 129.361667 |
| 1 min | M2 | 244.850928 | 0.558000 | 130.327367 |
| 1 min | ML = M2L | 352.024628 | 0.552216 | 177.312527 |
| 1 min | M2LZ | 374.693997 | 0.346660 | 210.531777 |
| 5 min | M0 | 1067.084719 | 0.592165 | 401.438838 |
| 5 min | M2 | 1046.368139 | 0.579726 | 407.364151 |
| 5 min | ML = M2L | 2327.103203 | 0.338044 | 952.596272 |
| 5 min | M2LZ | 1872.966539 | 0.338181 | 817.857285 |
| 15 min | M0 | 3128.361116 | 0.598035 | 906.767800 |
| 15 min | M2 | 2895.152885 | 0.592482 | 862.084134 |
| 15 min | ML = M2L | 8448.335730 | -0.151358 | 3648.137881 |
| 15 min | M2LZ | 5985.413793 | 0.167862 | 2503.989014 |
| 1 hour | M0 | 15292.942030 | 0.435101 | 4923.057432 |
| 1 hour | M2 | 13191.452811 | 0.436546 | 4550.994398 |
| 1 hour | ML = M2L | 38202.199533 | -0.716254 | 16380.914198 |
| 1 hour | M2LZ | 25353.496523 | -0.037804 | 10505.139581 |

## Imbalance results at all horizons

Derived from unchanged frozen forecast arrays, without fitting or new forecasting. No imbalance-specific confidence intervals were computed. All 35 correlations were checked against the original horizon table and agree to within 1e-12.

| Horizon | Model | Imbalance MAE | Imbalance R² | Correlation | Bias |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 s | M0 | 12.060768 | -0.084780 | -0.019693 | -4.634990 |
| 1 s | M2 | 12.000801 | -0.080648 | -0.020422 | -4.658813 |
| 1 s | ML = M2L | 11.181979 | -0.020422 | 0.071068 | -4.768479 |
| 1 s | M2LZ | 12.353706 | -0.006567 | 0.153147 | -2.756501 |
| 5 s | M0 | 41.287342 | -0.050621 | 0.037652 | -14.428142 |
| 5 s | M2 | 41.224936 | -0.049341 | 0.034000 | -14.419640 |
| 5 s | ML = M2L | 40.073138 | -0.020588 | 0.079649 | -14.650638 |
| 5 s | M2LZ | 38.932822 | 0.021813 | 0.167913 | -7.270717 |
| 15 s | M0 | 79.061706 | -0.058718 | 0.020252 | -38.375614 |
| 15 s | M2 | 78.893889 | -0.056513 | 0.021559 | -37.852748 |
| 15 s | ML = M2L | 78.800364 | -0.026833 | 0.190941 | -37.260270 |
| 15 s | M2LZ | 74.947532 | 0.027913 | 0.211034 | -19.427975 |
| 1 min | M0 | 167.551299 | 0.059384 | 0.278199 | -29.195700 |
| 1 min | M2 | 167.955237 | 0.058622 | 0.269835 | -25.535841 |
| 1 min | ML = M2L | 171.774962 | -0.001795 | 0.064401 | -20.248601 |
| 1 min | M2LZ | 177.383010 | 0.012391 | 0.195022 | 40.700380 |
| 5 min | M0 | 509.401621 | 0.084985 | 0.331466 | -59.759331 |
| 5 min | M2 | 514.847777 | 0.061416 | 0.266215 | -41.086157 |
| 5 min | ML = M2L | 536.375937 | -0.016039 | -0.078292 | -25.747883 |
| 5 min | M2LZ | 575.059399 | -0.085943 | 0.185212 | 269.577394 |
| 15 min | M0 | 827.316890 | 0.100800 | 0.343176 | 133.996158 |
| 15 min | M2 | 854.126247 | 0.059706 | 0.284915 | 191.277349 |
| 15 min | ML = M2L | 896.549234 | -0.071484 | -0.145982 | 191.881932 |
| 15 min | M2LZ | 1366.590608 | -0.701186 | 0.239556 | 1143.237914 |
| 1 hour | M0 | 2228.846443 | -0.089380 | 0.051529 | 642.275550 |
| 1 hour | M2 | 2422.028069 | -0.206038 | -0.032517 | 844.067603 |
| 1 hour | ML = M2L | 2330.499637 | -0.190083 | -0.271100 | 683.731284 |
| 1 hour | M2LZ | 5129.136771 | -2.215399 | 0.199494 | 4824.409057 |

## Models, parameters and search

M0: two observed channels, additive positive exponential Hawkes count model, 28 parameters, 13 exponential modes, no anchors or OU components.
M2: observed channels plus positive amplitude modulation through slow anchors; 80 parameters, 15 modes and 3 anchors.
ML: observed B/S plus inferred hidden event channel L; selected 43 parameters, 20 modes, no modulation anchors or OU components.
M2L: latent event channel plus eligible modulation; selected the same model as ML, with no anchors.
M2LZ: latent event channel plus continuous stochastic excitation through OU factors; selected 43 parameters, 14 exponential modes, 3 OU components and no modulation anchors. Selected candidate a472c2f9c434dbd01a28; ML/M2L candidate b384631aa5468da31e1c. See frozen source code for exact recurrence and loading/link definitions.

All backgrounds, masses, timescales, modulation anchors/coefficient normalizations and OU loadings are included in fitted_scales.csv and each model's frozen-model.json. The 152 parameter-table rows cover all 237 free coordinates across family labels (including the ML/M2L duplicate). Per-edge counts differ; mixture modes are not forced to share a common grid.

RTX 3070 acceleration used a batched float64 CUDA/hybrid architecture-search implementation. The latent search was stopped at a finite budget after 975 completed fit records representing 673 distinct fitted parameter sets; shared records are not independent restarts. Unfinished checkpoint candidates were preserved locally and excluded from selection. All eight shortlisted latent parameter sets completed 150 joint updates with all retained coordinates free. This does not establish optimizer convergence, a completed architecture neighborhood, a plateau, or globally optimal mode counts/scales. A 13-mode M2LZ alternative is statistically close to the selected 14-mode model.

Exact selected-point particle audits used two seeds at 128/512/2048 particles. ML mean validation proper NLL: 10.7571061426 / 10.7571041224 / 10.7571035870; 512→2048 drift −5.353e-7. M2LZ: 4.8775585962 / 4.8722511289 / 4.8701292905; drift −0.0021218384. Absolute particle convergence is not proven. Those drifts are small relative to the family NLL gaps but relevant for close architecture comparisons.

Implementation validation recorded 88 passing tests after GPU changes and 12 targeted final-pipeline checks. Completion-audit.json verifies final artifacts, all parameter coordinates and frozen source/model/evidence/cache identities. These checks are completed historical validation, not a claim that packaging reran the entire suite.

## Limitations that change interpretation

Conditional contiguous-block comparison intervals condition on fitted models and saved forecast means; they exclude parameter, selection, filtering and future-path Monte Carlo uncertainty, with no multiplicity correction. No new imbalance intervals have been calculated.

M0's global linear branching bound is 0.95. M2 has a 0.95 linear core but unbounded positive amplitude modulation, with no global stability guarantee. Finite successful simulated paths do not establish safe indefinite operation. ML/M2L and M2LZ have bounded linear cores; that does not fix their observed forecast failures.

Synthetic latent recovery failed a factor-two criterion for a weak hidden edge, and its scale slice was weakly identified. L is an inferred statistical component, with no established physical identity. Subsecond decay values fitted to one-second bins do not establish subsecond event-time resolution. OU loading signs and component rotations can be non-identifiable.

No live deployment, price-forecast success, profitable trading strategy or profitability regardless of market conditions has been established. The outcomes concern event-count prediction.

## Files and how to continue

Start with this document. Then use these relative paths:

- data/benchmarks/hawkes-joint-1s-2026-10-03/final-evaluation-bounded/report.md: original final report, including fitted values, intervals, plots and limitations; imbalance addendum is in this handoff and the separate CSV.
- Same folder: final_test_metrics.csv, horizon_metrics.csv, imbalance_horizon_metrics.csv, fitted_scales.csv, stability.csv, latent_diagnostics.csv and model-decision.json.
- Same folder: each family contains frozen-model.json, selection-audit.json, forecast protocol, test_forecasts.npz and per-second predictive-loss arrays. Latent traces are included.
- Same folder: comparison/ contains conditional paired evidence and plots; source/ contains the exact frozen evaluation sources.
- data/benchmarks/hawkes-joint-1s-2026-10-03/bounded-selection/: selection/ranking summaries, full validation forecasts and comparison artifacts, finite-budget review and frozen-selection.json.
- data/benchmarks/hawkes-resume-2026-10-03/counts-1s.npz and data-audit.json: complete masked count corpus sufficient for model evaluation, with source identity and coverage evidence.
- ml/: current Hawkes implementation, CUDA search, evaluation/report helpers and tests. Exact frozen evaluation sources take precedence for recreating the published evaluation.
- docs/experiments/: chronological research diaries.
- historical/hawkes_codex_handoff/: original incoming handoff, spec, original 5-second scripts and results, retained as historical context.
- PACKAGE_MANIFEST.json: inventory and SHA256 for every bundled file. VERIFY_PACKAGE.py checks integrity with the Python standard library.
- EXCLUDED_FILES.json: precise intermediate ranking-array omissions. Raw aggregate-trade export files, the full optimizer-checkpoint/search tree, Python environment and unrelated repository files are not bundled. This is a results and analysis handoff, not a full repository backup.

The original artifacts retain Windows absolute paths and their original bytes/hashes. Map E:/Projects/trading/<relative path> to this package's <relative path>. Do not execute old launchers with stale absolute paths or rerun the entire search merely to retrieve existing metrics. Frozen evaluation source imports resolve when its source directory is placed on Python's import path. Additional fitting requires the current ml/ modules and the dependencies in ml/requirements.txt; CUDA paths require compatible GPU/toolchain support. The count cache uses NumPy and is loaded with allow_pickle=False.

Useful next analysis, subject to Roman's next request: assess whether imbalance has actionable predictive value with paired block uncertainty and an out-of-sample baseline; inspect M2LZ's imbalance bias and long-horizon interval calibration; distinguish genuine density gains from state/filter approximations and point-forecast failures. Any new model tuning should use validation and fresh future evaluation data rather than recycling this already-inspected period as an untouched test.

No new computation is requested by this document itself. Continue from the completed findings and Roman's next instruction.
