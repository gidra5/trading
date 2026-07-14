# Causal KAMA oracle-signal search

This experiment fits completed-candle KAMA state changes to the close-only perfect-margin oracle. It ranks signal timing and exposure-path similarity, not trading return, execution price, leverage, or P&L.

The volume-weighted KAMA uses only information available at the candle close:

`ER move weight_i = (volume_i / EMA(volume)_i) ^ efficiencyVolumePower`

`ER = abs(sum(ER move weight_i * price change_i)) / sum(abs(ER move weight_i * price change_i))`

`effective ER = clamp(ER * relativeVolume ^ volumePower, 0, 1)`

`alpha = (slowAlpha + effective ER * (fastAlpha - slowAlpha)) ^ kamaPower`

The ER volume EMA includes the current completed candle and no future observations.
`efficiencyVolumePower = 0` recovers the standard unweighted ER exactly. Setting both
that power and the post-ER `volumePower` to zero is the canonical KAMA control.

## Fractional signal strength

Accepted buy and sell signals use separate maximum strengths and Gaussian widths:

`signal fraction = side maximum * exp(-0.5 * (VW-KAMA rate / side sigma) ^ 2)`

The rate is the same causal, completed-candle VW-KAMA rate used for the signal. Agreement
can interpret the resulting strength in two ways:

- `sizing`: a partial position is anchored at its signal price. If
  `r = current price / signal price`, its marked effective exposure is
  `f*r/(1-f+f*r)` when long and `-f*r/(1+f-f*r)` when short. Explicit zero and full
  allocations remain exactly `0` and `±1`.
- `confidence`: strength remains fixed until the next accepted signal. It is probability
  mass on the signaled direction, with `1-f` assigned to flat/uncertain. A matching oracle
  direction earns `f`, oracle-flat earns `1-f`, and the opposite direction earns zero.

The oracle always uses full directional exposure. Optimizer runs may search both agreement
strategies or restrict them with `--agreement-modes`.

## Causal peak/valley confirmation

The evaluator and optimizer combine causal clues before accepting a directional transition:

`acceleration = smoothed KAMA-rate change / EWMA(abs(KAMA-rate change))`

`overextension = (price / KAMA - 1) / EWMA(abs(price / KAMA - 1))`

They can also use:

- an independent slow price EMA rate, with a countertrend tolerance;
- RSI direction around 50, with a configurable neutral tolerance;
- DMI direction (`+DI - -DI`) weighted by how far ADX is above its trend threshold.

ADX supplies strength rather than direction: weak-ADX DMI contributes almost nothing,
while strong ADX makes aligned or opposing DMI direction matter. For direction `d`
(`+1` long, `-1` short), the raw confirmation is:

`logistic(bias + acceleration term - overextension term + EMA term + RSI term + ADX-weighted DMI term)`

This rewards acceleration in the proposed direction and penalizes entering after price is
already extended in that direction. `quality = 1 - mix + mix * rawConfirmation`; quality
scales the existing Gaussian signal fraction, and an optional minimum-quality floor
suppresses weak transitions. `mix = 0` returns exactly `quality = 1`, preserving the
unconfirmed signal and its warmup behavior when the independent EMA gate is also zero.

The EMA gate can additionally penalize countertrend signals independently of the logistic
mix. At full strength it is a hard gate: an opposing position closes to flat, then the
candidate waits for EMA confirmation before opening the new side. This avoids interpreting
“ignore a short” as “continue holding long.”

The KAMA state clamp also supports hysteresis. A flat state requires the outer rate
threshold to enter; an existing direction remains active down to
`outer threshold * release ratio`, provided the rate still has the same sign. A sign
reversal releases immediately unless it crosses the opposite outer threshold.

All lookbacks are physical durations converted to samples at each candle scale. All inputs
use the current or earlier completed candles; centered extrema and future prices are not
used. This confirmation is currently an inspector/search experiment, not part of the live
peak/valley strategy.

## Signal memory

Candidate signals are causal and stateful. The first state change is accepted. After that, a proposed state change emits only when

`abs(close / lastSignalPrice - 1) > friction`

Rejected changes retain both the accepted state and its anchor price. The search, inspector, and strategy share this exact strict predicate. Search friction is `0.00175` (17.5 bps), the same value used by the target oracle. Runtime bot configs derive it from configured fee plus market slippage.

On the former `v0092` one-minute configuration, memory reduced validation signal density from `107.15` to `51.42` signals/day (`-52.0%`). Precision rose, but recall fell enough to reduce the objective from `57.61%` to `53.18%`, so parameters were searched again.

## Objective and selection

Candidate and oracle transitions use one chronological, one-to-one alignment by resulting
state. The alignment maximizes total timing credit for pairs inside the two-hour window:

`2 ^ (-abs(lag) / 10 minutes)`

Each transition can appear in only one pair. Unmatched candidate transitions reduce
precision, uncovered oracle transitions reduce recall, and timing error is reported only
for paired transitions.

`case score = 0.2 * timing F1 + 0.6 * path agreement + 0.2 * signal cleanliness`

`noise/signal ratio = extra candidate transitions / matched candidate transitions`

`signal cleanliness = matched / (matched + extra) = 1 / (1 + noise/signal ratio)`

Path agreement is the marked fraction overlapping the oracle: `clamp(candidate, 0, 1)`
for an oracle long, `clamp(-candidate, 0, 1)` for an oracle short, and
`1 - clamp(abs(candidate), 0, 1)` while the oracle is flat. The objective equally weights
median and P10 case scores. Fit ranks candidates, validation selects one finalist per
family, and only those finalists touch holdout. Defaults are chosen by validation, never
holdout.

The agreement-dominant objective is score version 3. Reports generated under score
versions 1 or 2 are retained as historical results and are not numerically comparable;
their configurations remain valid warm starts because every candidate is re-evaluated
under the current objective.

## Data

The one-minute search uses continuous local BTCUSDT history:

- Fit: 2021-08-01 through 2023-12-31 in four windows.
- Validation: 2024-01-01 through 2024-12-31 in two windows.
- Holdout: 2025-01-01 through 2026-07-10 in two windows.

The scale-general search uses `17,447,982` local one-second candles covering the representative `tasks.md` regimes. It evaluates `1s`, `5s`, `15s`, `1m`, `5m`, `15m`, and `1h` aggregations across three fit, two validation, and two holdout windows. Every sparse continuous segment reserves a fixed 72-hour causal lead-in before scoring.

The search loader processes one continuous segment at a time and retains only compact candidate statistics. This keeps the full 3 GiB source search below 1 GiB RSS without changing grouping, scores, ranking, or selection.

## Latest scale-general signal search

The historical score-version-2 `60% F1 / 30% agreement / 10% cleanliness` differential-evolution
search used a 256-member population for eight generations across all seven scales and
searched both sizing and confidence agreement strategies.
It ranked candidates on three fit windows, selected one finalist per family on two
later validation windows, and read the two holdout windows only for those finalists.
The full run completed in 32 minutes 58 seconds.
Validation retained the full-sizing volume-aware baseline `v0182`:

| parameter | value |
|---|---:|
| ER | 265846 ms |
| fast | 1113418 ms |
| slow | 64800000 ms |
| KAMA power | 0.62547 |
| volume EMA | 283525 ms |
| volume cap | 5.08117 |
| volume power | 0.48543 |
| derivative policy | hold |
| threshold | 30 bps/hour |
| threshold mode | adaptive, zero noise multiplier |

| candidate | validation objective | holdout objective | validation F1 | validation agreement | validation cleanliness |
|---|---:|---:|---:|---:|---:|
| canonical `k0050` | 40.80% | 30.81% | 43.55% | 62.72% | 60.42% |
| volume `v0182` | 41.39% | 31.74% | 43.61% | 62.93% | 63.83% |

The modest holdout drop is real out-of-sample degradation, not a ranking leak.
No evolved fractional-sizing or confidence candidate beat the full-sizing baseline on
validation. Because 100% strength is identical in the two agreement strategies, their
baseline variants tied and deterministic ID ordering retained `sizing`.
The inspector now treats `v0182` as the global comparison preset, while the live
strategy runtime remains unchanged. See the
[score-version-2 chronological search report](vw-kama-global-score-v2-de-2026-07-12.md).

### Historical 60/40 search

The one-to-one `60% F1 / 40% agreement` search evaluated 256 broad trials and 256
refinement trials across all seven scales. Validation selected volume-aware candidate
`v0016`:

| parameter | searched duration/value | one-minute samples/value |
|---|---:|---:|
| ER | 1856036 ms | 31 samples |
| fast | 1873481 ms | 31 samples |
| slow | 3701806 ms | 62 samples |
| KAMA power | 0.61357 | 0.61357 |
| volume EMA | 440028 ms | 7 samples |
| volume cap | 1.97271 | 1.97271 |
| volume power | 1.72738 | 1.72738 |
| derivative policy | hold | hold |
| threshold | 0.44351 bps/hour | 0.44351 bps/hour |
| signal friction | 17.5 bps | fee + slippage |

| candidate | validation objective | holdout objective | validation F1 | agreement | signals/day |
|---|---:|---:|---:|---:|---:|
| old runtime `k0021` | 34.59% | 26.77% | 31.50% | 63.82% | 82.75 |
| broad volume `v0153` | 39.50% | 32.34% | 43.03% | 60.32% | 59.42 |
| refined canonical `k0016` | 39.90% | 31.57% | 42.65% | 62.19% | 54.80 |
| refined volume `v0016` | 39.95% | 30.88% | 43.00% | 62.08% | 54.87 |

Validation selected `v0016` without using holdout ranking. Its `0.05` percentage-point
validation edge over paired canonical `k0016` is negligible, while `k0016` is stronger
on holdout, so this search does not establish a robust benefit from volume weighting.
The runtime defaults remain `k0021` until the new signal is explicitly adopted.

The specialized one-minute search remains stronger at that scale: canonical `k0214` reached `55.48%` validation and `52.27%` holdout. It is not the default because it generalizes poorly to coarse candle scales. The current shared configuration still degrades materially at `15m` and `1h`; dedicated per-scale configurations are the clearest next experiment.

The strategy uses no confirmation averages for KAMA decisions. Flat-to-direction transitions create entries, direction-to-flat transitions create exits, and direct flips create the paired exit and entry. KAMA memory warms from only the KAMA-required suffix, while unrelated averages warm independently. Historical backtests load causal pre-window candles and exclude them from measured results.

## Interactive inspector

Open `#/kama-inspector` from the dashboard. Parameter changes rerun the selected representative window immediately. The page supports all seven candle scales, exact narrow-range one-second chart data, independently toggleable overlays, oracle/candidate state strips, matched and missed transitions, timing metrics, and a shared oracle/signal friction control.

Zoom detail returns server-computed KAMA values with the same parameters and causal warmup
as the evaluator, so the indicator line and transitions use the same resolution. The
metrics expose one-to-one matched, extra candidate, and missed oracle transition counts;
timing error is calculated only for matched pairs.

## Adaptive threshold and evolutionary search

The optional causal threshold is:

`base bps/hour + noise multiplier * EWMA(abs(current KAMA rate - previous KAMA rate))`

The lookback is a physical duration and is converted to samples at each candle scale. The
same state and snapshot logic is used by the evaluator and live strategy. Static mode is
exactly the old behavior.

The optimizer now uses adaptive current-to-pbest differential evolution over continuous
parameters and categorical modes. Family, agreement mode, and every confirmation-feature
combination have explicit islands. Each island retains a small diversity floor; the remaining
population competes globally, with continuous cross-island migration. Successful mutations
adapt the island's differential weight and crossover rate. Rotating fit folds reduce repeated
screening cost, periodic full-fit generations populate a hall of fame, and shrinking local
perturbations refine the final elites.

Generation zero evaluates every warm seed plus an equally large Latin-hypercube pool, then
selects by 70% score and 30% normalized parameter-space novelty. Nearest-selected novelty is
updated incrementally, preserving the selection rule without cubic distance recomputation.
Independent deterministic
restarts are merged and selected on the full fit set, so the final candidate count remains
bounded. Candidate/window scores are cached across generations. Candidate evaluation uses a
persistent bounded process pool with deterministic longest-processing-time shards: KAMA,
threshold, and confirmation warmups estimate work, then long-lookback genomes are assigned
first to the least-loaded worker. This avoids correlating Latin-hypercube order with a core;
bounded single-window searches also retain prepared candles and oracle paths in each worker.
Structured convergence telemetry records best/median score,
population diversity, unique candidates, adaptive F/CR, and the active fold for every
generation. Results are restored to original order before selection.

Search evaluation only warms and updates enabled features. Static-threshold candidates do not
inherit adaptive-threshold lookback, and disabled acceleration, distance, EMA, RSI, and DMI
parameters cannot change feed start or score. Full inspector traces still compute diagnostic
series for display.

The surviving population is evaluated by the full fit/validation/holdout path. Independent
per-window searches use the same bounded pool at the window level and emit inspector presets.
They are hindsight upper bounds for diagnosing parameter non-stationarity, not validation
results suitable for deployment.

`--seed-candidates` warm-starts generation zero from prior fit-result JSONL files and/or
inspector preset arrays. Exact candidates are deduplicated; volume-aware seeds receive a
canonical counterpart for a controlled family comparison. A single-window per-window run uses
the pool for candidate shards; multi-window runs use it for independent windows, avoiding
nested process pools.

The current 256-member, eight-generation per-window run covered all 28 representative
windows at `1m`, `5m`, and `15m`. It evaluates 512 evolving family candidates plus five
fixed global baselines per generation. Presets are selected independently per candle scale,
with the best global score enforced as a hard lower bound. After adding the 10% signal
cleanliness component and refreshing against the new global winner, all 84 selected
presets retained that lower bound; nine selected adaptive thresholds. Scores ranged from
`34.63%` to `68.57%`; see
[the per-window report](vw-kama-per-window-de-2026-07-12.md).
Hovering an oracle transition in the inspector now links and highlights its exact one-to-one
candidate match, when one exists.

An exact `1s` follow-up used the 3 GiB one-second cache, warm-started every window from its
existing `1m`/`5m`/`15m` winners, and evolved 64 genomes for six generations. It produced
28 additional scale-specific presets without replacing the coarser presets. All exceeded
the best global lower bound. Under the 60/30/10 objective, 16 presets use adaptive
thresholds. Scores range from `46.36%` to `74.55%`,
with a `59.44%` median. See
[the exact 1s report](vw-kama-per-window-1s-de-2026-07-12.md).

Detailed outputs:

The new broad, refined, and baseline reports share the current one-to-one evaluator and
objective. Earlier search reports are retained as historical many-to-one results and are
not directly comparable.

- [One-to-one 60/40 refined search](vw-kama-one-to-one-60-40-refined-2026-07-12.md)
- [Score-v2 sizing/confidence global search](vw-kama-global-score-v2-de-2026-07-12.md)
- [Fresh global 60/30/10 differential-evolution search](vw-kama-global-60-30-10-de-2026-07-12.md)
- [Per-window differential-evolution comparison](vw-kama-per-window-de-2026-07-12.md)
- [Exact 1s per-window differential-evolution comparison](vw-kama-per-window-1s-de-2026-07-12.md)
- [One-to-one 60/40 broad search](vw-kama-one-to-one-60-40-broad-2026-07-12.md)
- [One-to-one 60/40 runtime baseline](vw-kama-one-to-one-60-40-baseline-2026-07-12.md)
- [One-to-one alignment smoke verification](vw-kama-one-to-one-smoke-2026-07-12.md)
- [Refined multi-scale memory search](vw-kama-memory-multiscale-refined-2026-07-12.md)
- [Rounded runtime configuration verification](vw-kama-memory-selected-rounded-2026-07-12.md)
- [Broad multi-scale memory search](vw-kama-memory-multiscale-2026-07-12.md)
- [Refined one-minute memory search](vw-kama-memory-refined-1m-2026-07-12.md)
- [Pre-memory one-minute search](vw-kama-signal-only-1m-refined-2026-07-11.md)
