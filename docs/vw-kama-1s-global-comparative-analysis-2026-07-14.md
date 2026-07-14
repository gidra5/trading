# VW-KAMA 1-second optimization analysis

Status: final; all experiment artifacts and score calculations independently audited.

## Comparable score definition

All comparisons in this report use score version 3:

`score = 0.2 × F1 + 0.6 × exposure agreement + 0.2 × signal cleanliness`

The historical per-window and deep-search reports were produced with score version 2. Their displayed headline scores are retained as experiment history, but are not directly compared with the current global run. The 28 one-second per-window presets were therefore evaluated again, unchanged, with the current evaluator and saved in `data/benchmarks/vw-kama-window-presets-v3-rescored.json`.

## Completed per-window search

The per-window run optimized each history independently, with hindsight. It is an upper-bound diagnostic for model capacity, not a deployable validation result. The original run covered 28 windows at seven scales; this analysis uses its 28 one-second presets.

Artifact verification found the completion marker after 13,541,533 ms, 196/196 unique preset IDs in the seven-scale preset file, and exactly 28 one-second presets in the score-v3 rescore. Recomputed score-v3 values match `0.2 × F1 + 0.6 × agreement + 0.2 × cleanliness` exactly, and every preset satisfies `signals = matched + extra` and `oracle = matched + missed`.

Exact score-v3 rescore of those one-second presets:

| metric | minimum | median/mean | maximum |
|---|---:|---:|---:|
| score | 53.38% | 62.82% median; 61.84% mean | 70.85% |
| exposure agreement | 53.77% | 63.03% mean | 77.54% |
| F1 | 33.84% | 53.94% mean | 77.59% |
| cleanliness | 41.51% | 66.14% mean | 83.33% |

The strongest unchanged preset is `shape-down-high-2022-06` at 70.85%: 65.64% agreement, 77.59% F1, and 79.72% cleanliness. The weakest is `sharpe-down-7d-2023-03` at 53.38%: 53.77% agreement, 49.65% F1, and 55.95% cleanliness.

Even this hindsight-per-window upper bound remains far from the oracle: the best score leaves a 29.15-point gap and the mean leaves a 38.16-point gap. Because parameters were optimized on the same windows, this residual is primarily a limitation of the current causal VW-KAMA/confirmation family and its event timing—not global parameter generalization. The composite score measures path resemblance rather than trading return, so the gap should not be interpreted as percentage profit left unrealized.

This dispersion is important: a single VW-KAMA family can approximate the oracle much better in some regimes than others, and parameters selected independently for each window vary materially. A dynamic parameterization may therefore have value, but the hindsight scores do not prove that a causal regime selector can choose the right configuration.

The selected parameter distributions confirm that this is not just score noise:

| parameter | minimum | median | maximum |
|---|---:|---:|---:|
| ER window | 112.75 s | 22.91 min | 2 h |
| fast period | 1 min | 9.85 min | 1 h |
| slow period | 30 min | 66.34 min | 12 h |
| KAMA exponent | 0.383 | 0.643 | 1.091 |
| post-ER volume exponent | 0 | 0 | 3 |
| deadband | 0.1 bps/h | 242.79 bps/h | 300 bps/h |
| adaptive lookback | 15 min | 3.92 h | 12 h |
| noise multiplier | 0 | 5.20 | 8 |

The ranges frequently reach search boundaries, particularly deadband, slow period, volume exponent, and noise multiplier. That means the per-window optima are useful evidence for dynamic behavior, but not reliable estimates of unconstrained parameter optima. A future per-regime search should widen or reparameterize those ranges before treating boundary selections literally.

## Threshold and state behavior

All 28 selected one-second presets use `hold` deadband state. None selected `flat` or hysteresis as its final state policy in the older per-window search.

- 15 use adaptive thresholds and 13 use static thresholds.
- Adaptive presets average 62.50% score; static presets average 61.07%.
- Adaptive thresholds improve average F1 and cleanliness in this sample; static thresholds have slightly higher mean exposure agreement.
- 27 use sizing agreement. The deep rerun is the sole confidence-mode preset.
- Only the later deep rerun uses the new volume-weighted efficiency ratio and confirmation mixture, so the older per-window selection frequencies cannot establish whether those features generalize.

Interpretation of the newer global parameters:

- An adaptive threshold is the base deadband plus smoothed absolute KAMA-rate noise times the learned noise multiplier. It therefore widens in noisy periods.
- `hold` preserves the prior direction inside the deadband; it does not create a flat transition. `hysteresis` preserves the direction only until the rate falls below its learned release fraction, after which the ordinary three-state threshold applies.
- Mean-reversion strength rises smoothly as price distance from its EMA moves from `0.5 × threshold` to `1.5 × threshold`, measured in EMA volatility units. Its blend is `1 - 2 × mix × strength`: a mix at or below 0.5 can only attenuate trend signals, while a mix above 0.5 can reverse them at sufficiently large deviations.
- Confirmation evidence combines rate acceleration, negative overextension, slow-EMA alignment, RSI, and ADX-weighted DMI through a logistic score. The learned mix blends this score with an unconditional quality of one; minimum quality can suppress a transition entirely, while sizing/confidence mode determines how accepted quality affects exposure agreement.

## Deep worst-window searches

The original worst one-second window was `shape-flat-low-bias-low-2025-07` (`2025-07-04..2025-07-06`). Two much larger hindsight searches were run on it:

1. The first deep run raised its historical score-v2 result from 46.36% to 55.12%.
2. Adding volume-weighted ER raised it again to 55.62% under score v2.

Both logs end with completion markers and preset writes: the first deep run completed in 7,610,307 ms and the weighted-ER rerun in 6,031,420 ms. Their selected preset and the archived baseline were independently rescored from the stored parameters under score v3 before the comparisons below.

The selected final deep configuration uses:

- ER window 112.75 seconds;
- ER volume EMA 468.38 seconds and exponent 0.05, very close to standard ER (`0`);
- fast/slow KAMA periods 35.0 and 35.88 minutes;
- KAMA exponent 0.854;
- no post-ER volume adjustment (`volumePower = 0`, cap 1);
- static 37.795 bps/hour threshold with `hold` state;
- confidence agreement;
- strong confirmation mixture (0.952) and minimum quality (0.95);
- distance confirmation dominant, weak EMA confirmation, RSI enabled, and DMI weight zero.

Under the current score-v3 evaluator, that unchanged final preset scores 65.81%: 77.54% exposure agreement, 34.37% F1, and 62.07% cleanliness. It emits 29 signals for 21 oracle transitions: 18 matched, 11 extra, and 3 missed. Its median absolute timing error is 572.5 seconds and P90 is 4,241.8 seconds.

For a direct score-v3 comparison, the archived pre-deep baseline was re-evaluated unchanged on the same candles and saved as `data/benchmarks/vw-kama-deep-worst-window-baseline-v3-rescored.json`. It scores 56.72%: 67.68% agreement, 30.57% F1, and 50.00% cleanliness. The deep search therefore adds 9.09 score points, 9.86 agreement points, 3.80 F1 points, and 12.07 cleanliness points. Signals fall from 38 to 29 and extras from 19 to 11, at the cost of one fewer matched transition and one additional miss. Median lag is almost unchanged (543.0 to 572.5 seconds), while P90 improves from 5,019.0 to 4,241.8 seconds.

The high score is therefore mainly an exposure-agreement result, not a close one-to-one reproduction of oracle transitions. The optimizer learned to spend much of the window in the correct exposure while accepting sparse and relatively late event matches. That is consistent with the new 60% agreement weight.

The deep result still leaves a 34.19-point composite gap and a 22.46-point exposure-agreement gap to the oracle. Extra search depth substantially improves this window, but does not make the current signal family an oracle-equivalent representation.

The weighted-ER exponent of 0.05 is weak evidence against large volume weighting for this particular low-volatility window. DMI being driven to zero is similarly evidence that DMI did not help this window. Neither result is sufficient to remove those parameters from the global search because both are single-window hindsight observations.

## Global score-v3 search

The global optimizer is restricted to one-second candles and searches all parameter families, including static/adaptive thresholding, hold/flat/hysteresis state, sizing/confidence agreement, volume-weighted ER, post-ER volume adjustment, EMA/RSI/DMI confirmations, and mean-reversion blending.

Training partitions:

- fit: 2021-09 through 2023-12 representative windows;
- validation: 2024-01 through 2024-11;
- holdout/test: 2025-02 through 2026-07.

It uses two 2,048-member differential-evolution restarts, 32 generations each, rotating two-window fit folds, a full-fit generation every fourth generation, and four elite-refinement rounds. The initial population is warm-started with 852 unique candidates from the previous global and per-window searches.

Evolution telemetry:

| restart | generation | fold | best | median | diversity |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 59.38% | 43.88% | 0.439 |
| 1 | 2 | 2 | 59.83% | 46.52% | 0.442 |
| 1 | 3 | 3 | 61.56% | 50.83% | 0.442 |
| 1 | 4 | full fit | 60.51% | 51.25% | 0.432 |
| 1 | 5 | 2 | 60.09% | 52.04% | 0.439 |
| 1 | 6 | 3 | 61.56% | 55.47% | 0.427 |
| 1 | 7 | 1 | 60.38% | 54.21% | 0.432 |
| 1 | 8 | full fit | 60.58% | 55.84% | 0.417 |
| 1 | 9 | 3 | 61.56% | 57.98% | 0.416 |
| 1 | 10 | 1 | 60.38% | 56.54% | 0.427 |
| 1 | 11 | 2 | 60.18% | 57.10% | 0.415 |
| 1 | 12 | full fit | 60.58% | 58.28% | 0.427 |
| 1 | 13 | 1 | 60.47% | 58.13% | 0.413 |
| 1 | 14 | 2 | 60.40% | 58.31% | 0.406 |
| 1 | 15 | 3 | 61.57% | 60.26% | 0.410 |
| 1 | 16 | full fit | 60.81% | 59.33% | 0.411 |
| 1 | 17 | 2 | 60.40% | 58.89% | 0.392 |
| 1 | 18 | 3 | 61.60% | 60.60% | 0.391 |
| 1 | 19 | 1 | 60.56% | 59.25% | 0.409 |
| 1 | 20 | full fit | 60.90% | 59.78% | 0.417 |
| 1 | 21 | 3 | 61.61% | 60.77% | 0.399 |
| 1 | 22 | 1 | 60.80% | 59.44% | 0.415 |
| 1 | 23 | 2 | 60.64% | 59.38% | 0.407 |
| 1 | 24 | full fit | 60.92% | 59.96% | 0.416 |
| 1 | 25 | 1 | 61.03% | 59.60% | 0.404 |
| 1 | 26 | 2 | 60.77% | 59.52% | 0.419 |
| 1 | 27 | 3 | 61.68% | 60.88% | 0.418 |
| 1 | 28 | full fit | 60.92% | 60.04% | 0.417 |
| 1 | 29 | 2 | 60.68% | 59.55% | 0.423 |
| 1 | 30 | 3 | 61.93% | 60.92% | 0.418 |
| 1 | 31 | 1 | 60.80% | 59.67% | 0.424 |
| 1 | 32 | full fit | 60.92% | 60.12% | 0.414 |
| 2 | 1 | 1 | 45.59% | 20.00% | 0.468 |
| 2 | 2 | 2 | 45.82% | 21.19% | 0.470 |
| 2 | 3 | 3 | 54.07% | 24.87% | 0.481 |
| 2 | 4 | full fit | 53.74% | 26.36% | 0.484 |
| 2 | 5 | 2 | 53.39% | 28.82% | 0.478 |
| 2 | 6 | 3 | 55.31% | 34.35% | 0.490 |
| 2 | 7 | 1 | 56.22% | 35.53% | 0.492 |
| 2 | 8 | full fit | 57.63% | 38.53% | 0.491 |
| 2 | 9 | 3 | 62.01% | 43.10% | 0.485 |
| 2 | 10 | 1 | 59.48% | 43.17% | 0.487 |
| 2 | 11 | 2 | 59.78% | 45.39% | 0.485 |
| 2 | 12 | full fit | 60.44% | 47.57% | 0.489 |
| 2 | 13 | 1 | 59.48% | 49.23% | 0.500 |
| 2 | 14 | 2 | 59.78% | 50.41% | 0.497 |
| 2 | 15 | 3 | 62.01% | 53.10% | 0.494 |
| 2 | 16 | full fit | 60.44% | 53.33% | 0.484 |
| 2 | 17 | 2 | 59.78% | 53.74% | 0.485 |
| 2 | 18 | 3 | 62.38% | 55.62% | 0.471 |
| 2 | 19 | 1 | 60.26% | 55.46% | 0.470 |
| 2 | 20 | full fit | 60.72% | 55.78% | 0.478 |
| 2 | 21 | 3 | 62.38% | 57.76% | 0.469 |
| 2 | 22 | 1 | 60.27% | 56.62% | 0.474 |
| 2 | 23 | 2 | 59.91% | 56.64% | 0.465 |
| 2 | 24 | full fit | 60.93% | 57.90% | 0.448 |
| 2 | 25 | 1 | 60.89% | 57.69% | 0.447 |
| 2 | 26 | 2 | 60.41% | 57.72% | 0.440 |
| 2 | 27 | 3 | 62.38% | 60.34% | 0.429 |
| 2 | 28 | full fit | 61.02% | 58.98% | 0.435 |
| 2 | 29 | 2 | 60.49% | 58.28% | 0.438 |
| 2 | 30 | 3 | 62.46% | 60.81% | 0.435 |
| 2 | 31 | 1 | 61.01% | 58.64% | 0.437 |
| 2 | 32 | full fit | 61.24% | 59.47% | 0.434 |

Between the generation-8 and generation-12 full-fit checkpoints, the best score stayed at 60.58%, while the population median increased by 2.44 percentage points and diversity rose from 0.417 to 0.427. The search is improving the broader population but has not yet found a new full-fit champion.

Generation 16 converted that population improvement into a new 60.81% full-fit best, 0.23 percentage points above generation 12. Its 59.33% median is 3.49 points above generation 8, while diversity remains nonzero at 0.411.

Generation 20 improved the full-fit frontier again to 60.90%. The 0.09-point gain over generation 16 is smaller than the preceding checkpoint gain, suggesting convergence is beginning, but the 0.417 diversity and continuing fold-level improvements do not yet justify early termination.

Generation 24 reached 60.92% full fit, only 0.02 points above generation 20, while the median rose to 59.96%. The first restart is now close to a local plateau, although diversity remains 0.416 and the second independent restart is still required before judging search-space convergence.

Generation 25 reached 61.03% on fold 1, 0.23 points above the previous fold-1 checkpoint at generation 22. This is evidence of continuing local improvement, but is not comparable to the 60.92% full-fit frontier until the candidate is evaluated across all fit windows.

Generation 26 reached 60.77% on fold 2, 0.13 points above generation 23, while diversity rose from 0.407 to 0.419. Both rotating folds have therefore continued to improve after the generation-24 full-fit checkpoint without collapsing population diversity.

Generation 27 reached 61.68% on fold 3, a new fold-3 best by 0.07 points. Its 60.88% median is the strongest fold median in restart 1, with diversity still at 0.418. All three folds improved after generation 24; generation 28 determines whether those local gains survive simultaneous full-fit evaluation.

Generation 28 left the full-fit frontier unchanged at 60.92%, although the median rose from 59.96% at generation 24 to 60.04%. The fold-specific improvements did not combine into a better all-window candidate. Restart 1 is therefore converging around a broad local plateau, making the independent second restart meaningful rather than redundant.

Generation 29 reached 60.68% on fold 2, below the generation-26 fold best of 60.77%, while its median rose slightly to 59.55% and diversity to 0.423. It adds exploration but no new fold frontier.

Generation 30 reached 61.93% on fold 3, improving its previous frontier by 0.25 points. The median reached 60.92%, the strongest of restart 1, while diversity remained 0.418. This is meaningful local progress, but generation 32 must still show that it survives all-window evaluation.

Generation 31 reached 60.80% on fold 1, below its generation-25 frontier of 61.03%, but with a higher 59.67% median and restart-high diversity of 0.424. Restart 1 therefore closes its rotating folds with a broad population but no uniform fold improvement.

Generation 32 closed restart 1 at the unchanged 60.92% full-fit frontier. Its 60.12% median is the strongest full-fit median, 0.16 points above generation 24, but the champion has not improved since generation 24. Restart 1 has converged to a broad plateau rather than a single collapsed genome.

Restart 2 generation 1 begins at 45.59% on fold 1, with a 20.00% median and 0.468 diversity. The large gap from restart 1 confirms that the second restart is genuinely independent rather than another warm-seed continuation. Its early score is not comparable evidence of final quality; its purpose is to search a different basin before full-fit selection merges both restarts.

Restart 2 generation 2 improves to 45.82% on fold 2, with its median rising to 21.19% and diversity to 0.470. The independent population is improving without premature convergence, but remains far below the warm-started basin.

Restart 2 generation 3 jumps to 54.07% on fold 3, an 8.25-point one-generation gain. Its median reaches 24.87% and diversity rises further to 0.481. The population has found a materially stronger basin without losing exploration; generation 4 provides its first full-fit comparison.

Restart 2 generation 4 scores 53.74% on full fit, with a 26.36% median and 0.484 diversity. Its first all-window frontier is 7.18 points below restart 1's 60.92%, but it has completed only four generations and retains substantially more diversity than restart 1. The result supports continuing the independent search rather than merging it early.

Restart 2 generation 5 reaches 53.39% on fold 2, 7.57 points above its generation-2 fold checkpoint. Its median rises to 28.82%, while diversity remains high at 0.478. The independent population is converging rapidly from its random start.

Restart 2 generation 6 reaches 55.31% on fold 3, 1.24 points above generation 3. Its median jumps by 9.48 points to 34.35%, and diversity reaches a run-high 0.490. Improvement is spreading through the population rather than being confined to one champion.

Restart 2 generation 7 reaches 56.22% on fold 1, 10.63 points above its first-generation fold-1 result. The median rises to 35.53% and diversity to 0.492, both restart highs. The independent basin is still materially weaker than restart 1, but it continues to improve without collapsing before its second full-fit checkpoint.

Restart 2 generation 8 reaches 57.63% on full fit, 3.89 points above its generation-4 checkpoint but still 3.29 points below restart 1's 60.92% frontier. Its full-fit median rises from 26.36% to 38.53%, while diversity remains high at 0.491. The second basin is converging rapidly and still exploring broadly, but it has not yet displaced the warm-started solution.

Restart 2 generation 9 reaches 62.01% on its two-window rotation, 0.08 points above restart 1's best corresponding rotation result at generation 30. Its median remains much lower at 43.10%, with diversity at 0.485, so this is a strong frontier candidate rather than a broadly competitive population. The result becomes globally meaningful only if it survives the generation-12 full-fit checkpoint.

Restart 2 generation 10 reaches 59.48% on the next rotation, 1.55 points below restart 1's best corresponding result of 61.03%. Its median is nearly unchanged at 43.17% and diversity remains high at 0.487. The generation-9 frontier has therefore not generalized across the next rotation yet; generations 11 and 12 still determine whether it contributes to a stronger all-window candidate.

Restart 2 generation 11 reaches 59.78% on the third rotation, 0.99 points below restart 1's best corresponding result of 60.77%. Its median improves by 2.22 points to 45.39%, while diversity remains 0.485. Restart 2 now leads narrowly on one rotation and trails on two; generation 12 determines whether the generation-9 frontier improves its 57.63% all-window result.

Restart 2 generation 12 reaches 60.44% on full fit, 2.81 points above its generation-8 checkpoint and only 0.48 points below restart 1's 60.92% frontier. Its 47.57% median remains 12.55 points below restart 1's final 60.12% full-fit median, but diversity is substantially higher at 0.489 versus 0.414. Restart 2 has therefore found a genuinely competitive frontier while retaining much more unexplored population variation; it is no longer merely a weak control restart.

Restart 2 generation 13 leaves its rotation frontier unchanged at 59.48%, while the median rises by 6.06 points from the corresponding generation-10 checkpoint to 49.23%. Diversity reaches a restart high of 0.500. The population is improving broadly without producing a new frontier on this rotation.

Restart 2 generation 14 leaves its rotation frontier unchanged at 59.78%, while the median rises by 5.02 points from the corresponding generation-11 checkpoint to 50.41%. Diversity remains high at 0.497. Two consecutive rotations now show broad population improvement without a new frontier.

Restart 2 generation 15 leaves its rotation frontier unchanged at 62.01%, while the median rises by 10.00 points from the corresponding generation-9 checkpoint to 53.10%. Diversity remains high at 0.494. All three rotations preserved their existing frontiers but materially improved their medians before the generation-16 full-fit checkpoint.

Restart 2 generation 16 leaves its full-fit frontier unchanged at 60.44%, still 0.48 points below restart 1's 60.92%. Its full-fit median rises by 5.76 points from generation 12 to 53.33%, while diversity remains high at 0.484. The rotation-level population gains therefore generalized into a materially stronger all-window population, but not yet into a better champion.

Restart 2 generation 17 leaves its fold-2 frontier unchanged at 59.78%, while the median rises from 50.41% at the corresponding generation-14 checkpoint to 53.74%. Diversity remains nearly unchanged at 0.485. The restart continues to improve the broader population, but has not yet found a stronger rotation champion after its generation-16 full-fit checkpoint.

Restart 2 generation 18 improves the fold-3 frontier from 62.01% to 62.38%, while the corresponding median rises from 53.10% at generation 15 to 55.62%. Diversity declines moderately from 0.494 to 0.471 but remains substantial. This is restart 2's strongest rotation result so far; generation 20 must still establish whether it improves the 60.44% all-window frontier.

Restart 2 generation 19 improves its fold-1 frontier from 59.48% to 60.26%, while the corresponding median rises from 49.23% at generation 13 to 55.46%. It remains 0.77 points below restart 1's 61.03% fold-1 best, but the simultaneous fold-1 and fold-3 gains give generation 20 a credible chance to improve restart 2's 60.44% full-fit frontier.

Restart 2 generation 20 converts those rotation gains into a 60.72% full-fit frontier, 0.28 points above generation 16 and only 0.20 points below restart 1's 60.92% best. Its full-fit median rises by 2.45 points to 55.78%, while diversity increases slightly to 0.478. The independent basin is now genuinely competitive across all fit windows, although restart 1 still supplies the overall fit champion.

Restart 2 generation 21 leaves the fold-3 frontier unchanged at 62.38%, while the corresponding median rises by 2.14 points from generation 18 to 57.76%. Diversity remains nearly unchanged at 0.469. Improvement is again spreading through the population without producing a new rotation champion.

Restart 2 generation 22 nudges the fold-1 frontier from 60.26% to 60.27%, while the corresponding median rises by 1.16 points to 56.62%. Diversity increases slightly to 0.474. The 0.01-point frontier gain is negligible, so this checkpoint is better read as continued population consolidation than a materially stronger champion.

Restart 2 generation 23 improves the fold-2 frontier from 59.78% to 59.91%, while the corresponding median rises by 2.90 points from generation 17 to 56.64%. Diversity remains substantial at 0.465. The frontier still trails restart 1's 60.77% fold-2 best by 0.86 points, so generation 24 must establish whether the broad population gains improve full-fit selection.

Restart 2 generation 24 reaches a new overall full-fit frontier of 60.93%, 0.21 points above its generation-20 checkpoint and 0.01 points above restart 1's 60.92% best. The full-fit median rises by 2.12 points to 57.90%, while diversity remains substantial at 0.448. The independent restart has therefore displaced the warm-started champion rather than merely matching it, although the 0.01-point lead is too small to treat as a meaningful quality difference before exact validation and holdout rescoring.

Restart 2 generation 25 improves its fold-1 frontier from 60.27% to 60.89%, while the median rises by 1.07 points from generation 22 to 57.69%. It is now only 0.14 points below restart 1's 61.03% fold-1 best, with nearly unchanged diversity at 0.447. The post-checkpoint population therefore improved on the next rotation without immediately collapsing around the new full-fit champion.

Restart 2 generation 26 improves its fold-2 frontier from 59.91% to 60.41%, while the median rises by 1.08 points from generation 23 to 57.72%. It remains 0.36 points below restart 1's 60.77% fold-2 best, but all three rotating frontiers are now within 0.36 points of or above restart 1's corresponding results.

Restart 2 generation 27 leaves its 62.38% fold-3 frontier unchanged, while the median rises by 2.58 points from generation 21 to 60.34%. Diversity declines moderately to 0.429 but remains broad. The full population has caught up substantially before generation 28's all-window checkpoint, even though this rotation did not produce a new champion.

Restart 2 generation 28 reaches a new overall full-fit frontier of 61.02%, 0.09 points above generation 24 and 0.10 points above restart 1's 60.92% best. The full-fit median rises by 1.08 points to 58.98%, while diversity remains substantial at 0.435. This is a small but genuine all-window improvement; its practical significance still depends on exact validation and holdout behavior rather than the fit-set lead alone.

Restart 2 generation 29 improves its fold-2 frontier from 60.41% to 60.49%, while the median rises by 0.56 points from generation 26 to 58.28%. It remains 0.28 points below restart 1's 60.77% fold-2 best, with diversity nearly unchanged at 0.438. The small frontier gain is evidence of continued local progress, not a material quality change by itself.

Restart 2 generation 30 improves its fold-3 frontier from 62.38% to 62.46%, 0.53 points above restart 1's 61.93% fold-3 best. The median rises by 0.47 points from generation 27 to 60.81%, while diversity increases slightly to 0.435. Restart 2 therefore enters its final fold and full-fit checkpoint with a clearly stronger fold-3 frontier and a broadly competitive population.

Restart 2 generation 31 improves its fold-1 frontier from 60.89% to 61.01%, ending only 0.02 points below restart 1's 61.03% fold-1 best. The median rises by 0.95 points from generation 25 to 58.64%, while diversity remains nearly unchanged at 0.437. Restart 2 therefore reaches the final full-fit checkpoint with two stronger rotation frontiers and the third effectively tied.

Restart 2 generation 32 reaches a final evolutionary full-fit frontier of 61.24%, 0.22 points above generation 28 and 0.32 points above restart 1's 60.92% best. The full-fit median rises by 0.49 points to 59.47%, while diversity remains substantial at 0.434. Restart 2 therefore finishes with both a stronger champion and a broad population; elite refinement, validation, and untouched holdout rescoring must determine whether that fit advantage generalizes.

The two independent restarts and four local-refinement rounds ultimately produced 2,058 unique candidates for authoritative all-fit rescoring. The optimizer then evaluated 24 candidates on validation and exactly two family finalists on the untouched holdout. The complete run took 72,336,335 ms (20 hours 5 minutes 36 seconds).

## Audited global result

The family finalists are:

| split | candidate | score | F1 | agreement | cleanliness | noise/signal | signals/day | timing P50 | timing P90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fit | volume `v00129` | 60.91% | 43.05% | 59.61% | 85.78% | 0.166 | 67.00 | 402.5 s | — |
| validation | volume `v00129` | 59.97% | 37.77% | 62.55% | 77.17% | 0.296 | 47.37 | 466.0 s | 1,597.5 s |
| holdout | volume `v00129` | **51.74%** | 30.13% | **59.70%** | **56.97%** | 0.820 | 27.42 | **683.5 s** | **2,599.1 s** |
| fit | canonical `k00459` | 60.92% | 48.92% | 57.19% | 81.80% | 0.222 | 75.65 | 308.0 s | — |
| validation | canonical `k00459` | **60.30%** | **44.83%** | 61.15% | 76.38% | 0.313 | 57.33 | **348.0 s** | **1,298.0 s** |
| holdout | canonical `k00459` | 49.88% | **32.51%** | 55.40% | 59.66% | 0.702 | 27.20 | 721.3 s | 2,842.3 s |

Validation selected canonical by only 0.33 score points, but the untouched holdout reverses the ordering: volume wins by 1.87 points. Volume also retains 4.29 points more exposure agreement and lower lag. Canonical retains 2.24 points more precision and 2.38 points more F1 on holdout, but those event-level gains do not offset its poorer exposure path under the 60%-agreement objective.

Neither finalist is robust enough to be treated as a learned production configuration. From validation to holdout, volume loses 8.23 score points and canonical loses 10.42. Cleanliness falls by 20.20 and 16.72 points respectively, while noise/signal rises from 0.296 to 0.820 for volume and from 0.313 to 0.702 for canonical. Median timing error grows by 217.5 seconds for volume and 373.3 seconds for canonical. The largest failure is therefore event generalization: both produce a substantially noisier and later transition path in future regimes.

The two holdout windows also differ sharply. Volume scores 49.50% on `test-1` and 56.98% on `test-2`; canonical scores 47.18% and 56.18%. The first future window is the main robustness failure for both families, so the aggregate is not hiding uniformly mediocre behavior—it is averaging one especially weak regime with one materially better regime.

## Learned behavior

The volume finalist is structurally simple. It uses a 20.50-minute ER window, a volume-weighted ER exponent of 0.529, fast/slow KAMA periods of 25.75/28.30 minutes, a static 1.005 bps/hour threshold, and `hold`. It disables post-ER volume adjustment, all confirmations, and mean-reversion blending. This is the better holdout candidate.

The canonical finalist is structurally much more aggressive. It uses standard ER, 11.08/42.64-minute KAMA periods, the minimum searched static threshold of 0.05 bps/hour, `hold`, 100% confirmation mixing with a 95% quality gate, active slow-EMA and DMI evidence, and 100% mean-reversion blending around a very short mean. Acceleration, distance, and RSI weights are zero. Its validation edge disappears on holdout, so the complex confirmation/mean-reversion combination is not robust evidence of improvement.

Across all 24 validation candidates:

- every candidate uses `hold`;
- 22 use a static threshold and only two use an adaptive threshold;
- 15 use confidence scoring and nine use sizing;
- 13 retain mean-reversion, 14 retain confirmation mixing, and 10 retain volume-weighted ER;
- only seven retain post-ER volume adjustment.

This concentration is strong evidence for `hold` and static thresholds within the current search domain. It is weaker evidence for confidence scoring, confirmations, mean reversion, and volume-weighted ER because validation contains several competing forms and the two final families choose opposite structures. The better holdout result from a nonzero ER volume exponent justifies keeping weighted ER in future searches, but does not establish 0.529 as a stable production value. Both finalists set post-ER volume power to zero, making that separate volume adjustment a reasonable candidate for temporary removal or conditional search.

Several selected values lie on boundaries: the canonical base threshold is at its lower bound, confirmation quality is at 95%, and mean-reversion mixing is 100%. These should be interpreted as model pressure, not precise parameter estimates. Before widening those ranges, the feature should first survive rolling validation; otherwise a wider range only gives the optimizer more room to overfit.

## Global, per-window, and oracle comparison

| experiment | parameter knowledge | score | agreement | interpretation |
|---|---|---:|---:|---|
| oracle | future action path | 100% | 100% | non-causal target |
| best one-second per-window preset | same-window hindsight | 70.85% | 65.64% | best of 28 independently fitted windows |
| mean one-second per-window preset | same-window hindsight | 61.84% | 63.03% | current family’s average in-sample capacity |
| deep worst-window preset | same-window hindsight, much larger search | 65.81% | 77.54% | specialized upper-bound diagnostic |
| global volume finalist, validation | past fit only | 59.97% | 62.55% | model selection split |
| global volume finalist, holdout | fit and validation only | **51.74%** | **59.70%** | best honest future result |

The per-window and deep scores are not deployment estimates: each uses the evaluated window to select parameters. Their usefulness is diagnostic. The deep run’s 77.54% agreement shows that the signal family can represent the exposure path of one difficult window much better when specialized, while its 34.37% F1 and 572.5-second median lag show that even this specialized result does not reproduce oracle transitions closely.

The global holdout result leaves a 48.26-point composite gap and a 40.30-point exposure-agreement gap to the oracle. Some of that gap is the unavoidable difference between a causal signal and a hindsight oracle, but the per-window dispersion and holdout degradation show that regime generalization is currently the larger actionable problem. More generations within the same global objective are unlikely to close it: restart 2 improved full-fit from 60.44% to 61.24%, yet future performance remained near 50%.

## Conclusions

1. Use `v00129` as the current comparison baseline, not as a production-ready strategy. It is the less fragile of the two honest finalists.
2. Keep volume-weighted ER available. Remove or conditionally enable the separate post-ER volume adjustment unless a rolling test demonstrates value.
3. Keep `hold` and static thresholding as the baseline. Reintroduce adaptive thresholds only through regime-conditioned or walk-forward experiments, because they did not survive final selection here.
4. Do not adopt the canonical finalist’s full mean-reversion and confirmation stack as a default. Its validation result was marginally better, but its holdout result was worse and degraded more.
5. Optimize the next experiment for rolling generalization: choose parameters on a past window, score the immediately following window, and aggregate multiple such train-next pairs. This directly targets how long a learned configuration remains useful and prevents a single validation period from becoming another training set.
6. Report exposure-path quality and transition quality separately. The deep run proves they can diverge substantially; a composite score alone can hide a strategy that spends time on the correct side but changes position too late or too noisily.

The remaining oracle gap should be interpreted as both model error and causality cost. The per-window optimum has hindsight over parameters, while the perfect-margin oracle also has hindsight over every action. A causal global signal cannot be expected to close that gap completely, but the present holdout gap is large enough that improving causal regime adaptation matters more than further fine-tuning this single static configuration.

## Verification

`npm run audit:kama` independently verified the score-v3 formula, split isolation, aggregate median/P10 values, candidate/finalist stage counts, canonical ER invariants, warm-start population, full-fit schedule, generated presets, and the completed 72,336,335 ms runtime. The audit confirms 2,058 fit candidates, 24 validation candidates, and exactly two untouched-test evaluations.
