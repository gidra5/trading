# Kronos on BTCUSDT 1-minute candles: interim report

Date: 2026-08-07

Status: paused with resumable dense inference at 1,090 of 11,232 origins
(9.70%). The final frozen-policy held-out bot validation has not run, so this
report does not claim that Kronos is profitable.

## Executive summary

Kronos is not a strong point forecaster for this BTCUSDT 15x1-minute task so
far. On the clean policy-calibration windows, every episode-purged base variant
had worse candle MSE than persistence. The selected 50/50 mixture of a clean
BTC-adapted base predictor and the pretrained base predictor gave up additional
point MSE in exchange for the best calibration price IC, horizon IC, and oracle
distribution KL among the clean candidates.

The probabilistic output is more promising than the point forecast. Retaining
20 stochastic paths improved distribution estimates compared with 10 paths,
and an expected-utility policy derived from the complete path distribution has
driven profitable real simulator fills in several calibration diagnostics. The
strongest precommitted temporal-forward diagnostic returned +7.13% net over 44
hours while all constant-direction controls lost. That is useful evidence, but
it is only one forward slice inside a calibration episode and is not the final
six-episode held-out result.

Raw Kronos paths frequently contain invalid OHLC candles. KQSP makes evaluated
quantile candles 100% valid, but this is a geometry repair rather than an
accuracy improvement. The execution oracle consumes sampled close paths, so
high/low ordering defects do not contaminate its action values.

## Scope and reproducibility

- Market: Binance spot BTCUSDT.
- Cadence: one-minute candles.
- Forecast: 15 consecutive candles, with a new decision every 15 minutes.
- Context selected by calibration: 512 candles.
- Models installed at pinned revisions: Kronos mini, small, and base, plus the
  frozen public tokenizers.
- Evaluation coverage: the same 28 static non-fit windows exposed by the
  inspector UI. The four fit folds and latest three-month window are excluded.
- Sampling selected by calibration: temperature 0.8, top-p 0.9, 20 retained
  paths.
- Final model candidate: 10 paths from the episode-purged BTC-tuned base model
  and 10 paths from pretrained base.
- Checkpoint selection and bot-policy selection use only pre-held-out data.
  Overlapping UI windows are merged into independent market episodes for PnL
  ranking.
- Forecast rows are causal: context ends before the first target candle, and
  artifacts contain no realized targets.

The public Kronos paper evaluated five-minute and coarser data. This one-minute
experiment is therefore domain adaptation, not a reproduction of the paper's
headline numbers.

## Fine-tuning

The clean base predictor was trained on BTCUSDT one-minute sequences with a
frozen pretrained tokenizer and the upstream hierarchical next-token
cross-entropy objective. Normalization uses only the 512 historical candles.
An exclusion plan removes every training or validation sequence that touches a
policy-calibration UI episode.

Training range: 2021-07-01 through 2023-12-31. Chronological validation range:
2024-01-01 through 2024-06-30. Later inspector windows start on or after
2024-07-01.

The clean two-epoch tune improved total validation token loss from 2.425293 to
2.397107 (1.16%) and future-15 token cross-entropy from 2.474837 to 2.361090
(4.60%). Those gains did not translate into better candle MSE. A sequential
tokenizer-plus-predictor tune improved tokenizer reconstruction but degraded
later candle skill to -67.04% and price IC to 0.0322, so it was rejected.
Horizon-weighted predictor objectives also improved point MSE while discarding
roughly half the useful horizon correlation and price IC, so the upstream
weight-zero objective was retained.

## Clean sparse forecast results

The following are macro means over the 20 policy-calibration UI windows, with
identical target-time seeds and four origins per window. Positive candle skill
means lower MSE than persistence; lower oracle KL and CRPS are better.

| Configuration | Candle skill | Return correlation | Horizon IC | Price IC | Oracle KL | CRPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pretrained base | -4.44% | 0.0364 | 0.1829 | 0.0941 | 1.7195 | 0.001312 |
| Clean tuned base | -14.82% | 0.0462 | 0.1212 | 0.0751 | 1.7319 | 0.001350 |
| 75/25 tuned/pretrained | -17.05% | 0.0226 | 0.1160 | 0.1121 | 1.6558 | 0.001357 |
| Selected 50/50 mixture | -22.51% | 0.0111 | 0.1837 | 0.1327 | 1.5857 | 0.001353 |

On the eight later model-comparison windows, clean tuned alone had the best
candle skill (-12.99%) and CRPS (0.000593), while the selected 50/50 mixture
had the best oracle KL (0.7181). These later forecast metrics were inspected
during architecture comparison; later bot PnL remains untouched.

The earlier all-model zero-shot screen found that mini was the only public
checkpoint to narrowly beat persistence on sparse candle MSE (+0.85%). Small
scored -23.02% and base -10.24%. Base nevertheless had the best zero-shot
oracle KL (1.4190) and was selected for adaptation.

## Sampling and candle validity

Twenty retained paths are a measured compromise, not an arbitrary large
sample count. On the paired zero-shot screen, increasing 10 to 20 paths reduced
oracle KL from 1.8949 to 1.5589 for mini and from 1.8064 to 1.3988 for small.
Sampling reduces Monte Carlo estimation noise; it cannot correct a biased
conditional forecast.

Raw sampled paths often violate high >= max(open, close) or low <= min(open,
close). The implementation applies the two-stage KQSP projection to marginal
OHLC quantiles: Euclidean projection within each candle followed by isotonic
projection across quantiles. Repaired quantiles are 100% valid in every
completed screen. Tests cover the paper examples, randomized constraints,
idempotence, and isolation of close-only execution values.

## Bot evidence

All execution diagnostics use the production `GridTradingBot` and
`SimulatedTradingApi`, including actual entry/exit fills, fees, slippage,
borrow maintenance, drawdown, forced exit, and liquidation behavior.

The most important completed diagnostic was fixed before its target rows were
inspected. A policy selected on the first 176 September 2021 decisions was
replayed unchanged on the next non-overlapping 176 decisions:

- net return: +7.1310%;
- actual fills: 6;
- fees: $317.58;
- maintenance: $539.77;
- maximum drawdown: 10.02%;
- liquidations: 0;
- best constant long/short control: -2.47%.

Shorter replays of the same frozen policy were highly endpoint-sensitive: the
next 60 decisions lost 0.65%, while the next 90 returned +8.82% with the same
four-fill position sequence. This is why only predefined complete episodes are
eligible for the final conclusion.

Several complete calibration episodes also contain profitable locally selected
policies, but those are optimization diagnostics and must not be presented as
out-of-sample profitability. The final selector ranks candidates across 11
pretraining-era episodes, requires a separate pass on two January/February
2024 confirmation episodes, freezes one content-addressed policy, and then
evaluates it once on six later held-out episodes representing all eight later
UI windows.

## Paused exhaustive run

The dense run evaluates every eligible 15-minute origin across all 28 requested
windows. Its durable checkpoint contains 1,090 of 11,232 origins. On this early
chronological subset, the ensemble mean currently has:

| Metric | Interim value |
| --- | ---: |
| Candle MSE skill vs persistence | -8.33% |
| Candle correlation | 0.1244 |
| One-minute return correlation | 0.0146 |
| 15-minute horizon IC | 0.0521 |
| Horizon direction accuracy | 52.57% |
| Price IC | 0.0326 |
| Oracle forward KL | 1.7597 |
| Sample-path CRPS | 0.001452 |
| Origins with any invalid raw path | 81.10% |
| Repaired quantile OHLC validity | 100% |

These values are provisional because the checkpoint is chronological and only
9.70% complete. No Kronos process is running. The default float32 pipeline can
resume the matching checkpoint with:

```bash
npm run kronos:final-pipeline
```

An opt-in mixed-precision acceleration was implemented but its paired GPU
quality/speed screen was interrupted when work was paused. It is not selected
for the final artifact.

## Current conclusion

Kronos should not currently be treated as a reliable candle point predictor or
a proven profitable bot. Fine-tuning improved its language-model objective but
did not consistently improve downstream MSE. The worthwhile hypothesis is
narrower: its retained path distribution may encode weak ranking and
expected-utility information that a low-turnover, causally calibrated bot can
exploit. The precommitted forward diagnostic supports testing that hypothesis;
only completion of the frozen six-episode held-out validation can accept or
reject it.

## Artifacts

- Detailed experiment notebook:
  `docs/experiments/kronos-btcusdt-1m-15-candle-2026-08-06.md`
- Clean selected sparse metrics:
  `data/benchmarks/kronos-base-policy-holdout-v2-deterministic-t08-n20-n4.json`
- Precommitted forward bot report:
  `data/benchmarks/kronos-bot-confirmation-clean-v2-sideways-churn-2021-09-after176-next176.json`
- Resumable dense checkpoint:
  `data/benchmarks/kronos-base-policy-holdout-ensemble-dense-execution-metrics.base-policy-holdout-pretrained-ensemble.progress.json`
- Exclusion plan:
  `ml/training-plans/kronos-btcusdt-1m-policy-holdout-v1.json`

References: [Kronos repository](https://github.com/shiyu-coder/Kronos),
[Kronos paper](https://arxiv.org/abs/2508.02739), and
[KQSP paper](https://arxiv.org/abs/2607.26792v1).
