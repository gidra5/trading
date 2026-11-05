# Causal online calibration of the hurdle specialists

This experiment adds online calibration to the latest frozen hurdle NLL
checkpoints. The neural networks and the conditional distribution shapes stay
fixed. Calibration adapts probability intercepts and, in one variant,
conditional mean sizes using only outcomes already observed at the forecast
time. The experiment and all validation candidate settings were recorded
before evaluation.

The validation-selected calibration improves all-tick test mean-MSE skill
at all four horizons. At one second, skill rises from **3.337% to 3.512%**.
Most of that gain comes from adapting the zero probability to the test
period's higher zero frequency. Test likelihood improves at one second and
30 minutes but worsens at 15 minutes and two hours for these same selected
modes. The frozen baseline reproduces the previous experiment's metrics
exactly. **Fee-paying backtest results are unchanged**: 15- and 30-minute
policies remain in cash and the two-hour policy loses 1.042%. This run shows
a modest forecast correction without an observed trading benefit after fees.

## Forecast results

Mean-MSE skill is `100*(1-MSE/mean(actualReturn^2))` on all 36,134 common test
origins. Positive skill beats predicting a zero return; negative skill does
not. These are observed differences on a previously inspected period,
without a claim of statistically established improvement.

| Horizon | Frozen | Zero only | Both probabilities | Full calibration | Validation-preferred mode |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 second | +3.337% | +3.474% | +3.486% | +3.512% | Full |
| 15 minutes | -2.021% | -2.021% | -1.758% | -1.513% | Probabilities |
| 30 minutes | -7.612% | -7.612% | -6.645% | -6.346% | Probabilities |
| 2 hours | +0.673% | +0.673% | +0.920% | +1.021% | Probabilities |

Full calibration has the best observed test mean-MSE in each row, but
validation does **not** select it at the longer horizons. Changing those
choices after seeing this table would be test-based selection. The selected
one-second result is a gain of 0.175 percentage points in skill, equivalent
to about 0.181% less MSE, and remains below the previously measured joint
model's matched 3.699% skill.

| Horizon | Test full NLL, frozen | Zero only | Both probabilities | Full calibration |
| --- | ---: | ---: | ---: | ---: |
| 1 second | -3.999551 | -4.004890 | -4.004947 | -4.004881 |
| 15 minutes | -5.361137 | -5.361136 | -5.359991 | -5.363947 |
| 30 minutes | -5.026308 | -5.026308 | -5.026785 | -5.030933 |
| 2 hours | -4.238063 | -4.238063 | -4.235554 | -4.238944 |

Lower NLL is better. NLL scores the full hurdle law, including class
probabilities and conditional size densities; MSE scores its mean. Their
rankings can differ. Probability calibration at 15 minutes and two hours
worsens conditional sign log loss even while mean MSE improves. This does
not support a claim that every aspect of the distribution improves.

At one second, all three calibrated variants select the same zero-intercept
settings. Average predicted zero probability rises from **52.2801% to
56.5082%**, against **57.0903%** observed. Zero Brier error falls from
**0.240269 to 0.237695**. Zero-only calibration supplies about 78% of the
full variant's mean-MSE skill gain. With both probability intercepts,
conditional active-UP log loss improves from 0.383986 to 0.383853, while
classification accuracy changes from 84.4502% to 84.4244%. The update mostly
corrects probability levels; it does not create a large new directional
signal. Zeros are rare at the longer horizons, so zero-only calibration has
little effect there.

## Backtest results

All 24 replays finished on the same 42-hour tape, with gamma zero and the
unchanged controller. Every calibration mode has the same after-fee result:

| Horizon | Return after fees, all four modes | Market trades, including final close |
| --- | ---: | ---: |
| 15 minutes | 0.000% | 0 |
| 30 minutes | 0.000% | 0 |
| 2 hours | -1.042% | 2 |

The actual replay queries have zero full-close entry-threshold crossings at
15 and 30 minutes, and one at two hours, for every mode. The validation-
preferred variants' largest predicted mean returns are 5.136, 4.981 and
15.133 bps, respectively. The ordinary received-amount round-trip threshold
is 10.0075 bps. Calibration does not turn the shorter-horizon forecasts into
fee-covering entries on this replay schedule.

| Horizon | Zero fees, frozen | Zero only | Both probabilities | Full calibration |
| --- | ---: | ---: | ---: | ---: |
| 15 minutes | -1.694% | -1.694% | -1.860% | -1.777% |
| 30 minutes | -0.180% | -0.180% | +0.008% | +0.005% |
| 2 hours | -2.594% | -2.594% | -2.594% | -2.594% |

Without fees, probability calibration changes trade timing: 15-minute market
trades increase from 86 to 88, while 30-minute trades fall from 46 to 40.
The two-hour zero-fee decisions remain unchanged at 11 market trades. The
tiny positive 30-minute zero-fee return does not carry over to the model
with ordinary execution fees.

## Frozen validation choices

Every full-calibration variant selected a one-hour half-life and ridge
precision 32. Probability-only variants selected one hour with ridge 32
at one second and ridge 8 at each longer horizon. Zero-only selected
`(half-life hours, ridge)` of `(1,32)`, `(16,8)`, `(4,8)` and `(1,32)` for
the four horizons in order.

| Horizon | Frozen validation NLL | Preferred validation NLL | Preferred mode |
| --- | ---: | ---: | --- |
| 1 second | -3.988965 | -3.994443 | Full |
| 15 minutes | -5.106417 | -5.107624 | Probabilities |
| 30 minutes | -4.729420 | -4.730101 | Probabilities |
| 2 hours | -3.860703 | -3.861099 | Probabilities |

All choices were recorded in `selection.json` at
`2026-09-11T17:25:33.814642+00:00`, before test caching. Its SHA-256 is
`3a2987241eace925c2a1281f936e5d641524bba048d6322fa482a04c3d694f63`.
The plan hash is
`e09290903c9d49c9d6c5118c47a152a0b4a2d0e0bbab00c7a13eeb6a3859b2bb`.
Validation inference, grid evaluation and saving took 245.81 seconds; test
inference, calibration and exports took 62.36 seconds. No neural training
was performed.

## Fixed comparison

The source is `data/training/runs/stochastic-crra-hurdle-nll-v1/selection.json`,
SHA-256 `8ac7d25215c733b284d26cf97c09b741ffc81cc786eae20ef195547286777317`.
Only its selected fine-tuned model at each of 1, 900, 1800 and 7200 seconds is
used. There is no neural retraining or test-based checkpoint selection.

| Mode | Parameters that may change |
| --- | --- |
| `none` | None; frozen baseline |
| `zero` | Zero-probability logit intercept |
| `probabilities` | Zero and conditional-active-UP logit intercepts |
| `full` | Both probability intercepts, UP log-mean scale and DOWN log-odds-mean scale |

For each nonbaseline mode and horizon, validation compares half-lives of
1, 4 and 16 hours crossed with ridge precisions 8 and 32. The lowest full
hurdle validation NLL fixes that mode's settings. A validation-preferred mode
is also recorded, using deterministic ties that prefer the frozen baseline.
All settings and all four horizons' choices freeze before loading test data.
Every mode is reported, so the test period does not select a winner.

The dataset, two observed production59 feature states at `t-1,t`, 34,560
validation accuracy origins and 36,134 test accuracy origins match the prior
experiment. Accuracy uses the common maximum-horizon-complete origins at a
four-second stride. The canonical calibration stream also issues each
horizon's additional complete forecasts beyond this common accuracy cutoff,
allowing the entire 42-hour replay to use one consistent stream.

## Causal timing and overlap

Every split starts from identity calibration. A raw forecast issued at `t`
may update calibration only at `t+H`, when its endpoint return is observed.
At a query timestamp, already matured labels are processed first, in order;
the current raw forecast is then calibrated. Labels with maturity equal to
the query timestamp are available. One-second labels are processed at their
actual one-second maturity even though forecasts issue every four seconds.

Every issued forecast contributes once. Queries are read-only: choosing a
different metric or replay query schedule cannot add observations or change
the state. Forecasts and backtest exports are slices of the same saved
canonical calibrated stream. Later observations cannot rewrite older
forecasts. Split-terminal labels are flushed for the final state only and
do not affect any earlier prediction.

Overlapping horizon labels receive weight `min(1,4/H)`; statistics decay by
elapsed wall time. This weighting is a practical correction for heavily
overlapping returns, not a proof of independent observations or effective
sample size. Cold starts and slow availability of independent long-horizon
outcomes limit adaptation on this short evaluation period.

## Calibration update

Let the four coefficients be `theta=(b0,bs,ku,kd)`. The frozen model predicts
zero and active-UP logits and conditional magnitude means `mu,md`:

```text
zeroLogit' = zeroLogit + b0
activeUpLogit' = activeUpLogit + bs
mu' = mu * exp(ku)
md' = sigmoid(logit(md) + kd)
```

Numerical means remain at least `1e-12`; DOWN means remain at most
`1-1e-12`. Logit intercepts are bounded by ±2 and size coefficients by
±log(2). The UP Gamma and DOWN Beta shape parameters remain unchanged.
Zero calibration uses every label; sign calibration uses only nonzero
labels; each size coefficient uses only labels of its direction.

For a matured observation, let `g` be its per-coefficient negative-log-
likelihood gradient and `F` the corresponding Fisher information at the
current coefficients. With time decay `d`, label weight `w` and fixed ridge
precision `lambda`, the private sufficient statistics update as

```text
A <- d*A + w*F
B <- d*B + w*(F*theta - g)
theta <- clip(B/(A + lambda), coefficient bounds)
```

This is a discounted local Fisher approximation to regularized likelihood,
not an exact batch likelihood minimizer. Aging the data statistics while
keeping the zero-centered prior fixed moves old calibration toward identity.
Queries calculate that decay without mutating the event-anchored state.

The UP log-mean gradient and information are `alpha*(1-y/mu')` and
`alpha`. For DOWN, with `v=-y` and `beta=alpha*(1-md')/md'`, they are
`beta*(digamma(beta+alpha)-digamma(beta)+log(1-v))` and
`beta^2*(trigamma(beta)-trigamma(beta+alpha))`. The observed size-loss
Hessian equals `F-g`, so it is not interchangeable with expected Fisher
information. Large-beta expressions use checked asymptotic differences to
avoid cancellation. Physical-clamp masks compare DOWN means in logit space
so roundoff at the exact minimum mean cannot disable adaptation.

## Replay and proof scope

The fixed replay plan runs all four modes at 15 minutes, 30 minutes and two
hours, with both ordinary and zero fees: 24 replays, each on the identical
151,200-second observed tape. Gamma is zero. The unchanged endpoint policy
uses a restricted long/cash target menu `0,0.5,1` plus hold, unit funding,
no pending orders, actual one-second intermediate price marks, quote order
caps and price quantization. Ordinary received-amount taker fees are 0.05%
per side. Final closing obeys the existing valid-order and residual rules.

Forecast exports collapse each direction's magnitude distribution to its
conditional mean, producing DOWN/zero/UP support. This preserves the first
moment but discards magnitude spread. The result is a comparison for this
restricted controller, not an unrestricted multiasset, limit-order or
long-horizon optimal-policy backtest. The observed BTCUSDT spot tape with
these synthetic Oracle parameters is not a replay of a live perpetual
funding book.

Lean checks each supplied finite-menu payoff comparison under that model;
it does not prove that calibration improves forecast accuracy or that the
learned distribution is the true market law. This chronological test period
has already been inspected in earlier experiments. Settings are selected
only on validation, but these findings still need a fresh forward period
before any deployment decision.

## Verification

All **10 calibration tests pass**. They cover independent likelihood
autodiff and high-precision Fisher comparisons, exact identity and immutable
inputs, enabled-head masks, discount/update equations, physical boundaries,
one update per origin, rejection of past queries, exact maturity equality,
one-second delayed labels, query-schedule independence, cold resets, future-
suffix perturbation and deliberate corrupted-trace detection.

The independent CPU audit checked **1,168,900 canonical queries across 32
split/horizon/mode trajectories**, plus 87 artifact hashes. Every query's
incorporated-label count and latest maturity matched a separate timestamp
formula. No query used a label from its future. Cold starts, final label
flushes, mode masks, frozen distribution shapes and coefficient bounds all
passed. An independent odds-formula reconstruction matched transformed
parameters with maximum absolute difference `8.674e-18`. Independently
recomputed mean-MSE agrees with the reported metric; uncalibrated mean skill
differs from the frozen neural baseline by exactly zero on every split and
horizon.

The audit also verifies that selected settings and preferred modes are the
minimum validation NLL candidates in the registered order. Its artifact,
`data/backtests/oracle-online-calibration/diagnostics.json`, has SHA-256
`13960c48aa62f5ebbdaeff9767c012d5de0f0a040dcb0655af5bdad12163a837`.

The replay artifact checks passed: all 12 exports retain the frozen
checkpoint/selection/cache/trace hashes, correct mode settings and causal
replay snapshots, valid three-point laws and the identical complete observed
price tape. **Lean passed all 2,184 decision certificates across all 24
replays, plus all 108 execution conformance fixtures.** Results are saved in
`data/backtests/oracle-online-calibration/lean-verification.json`. These
checks preserve the existing finite-menu decision guarantee for the supplied
forecast laws; they do not establish statistical calibration or true-market
optimality.

## Reproduction and artifacts

New implementation:

- `ml/online_hurdle_calibration.py`: state and likelihood updates.
- `ml/evaluate_online_hurdle_calibration.py`: frozen inference cache,
  validation selection, causal test streams, metrics and exports.
- `ml/test_online_hurdle_calibration.py`: analytic, boundary and causality
  regression checks.
- `ml/audit_online_hurdle_calibration.py`: independent saved-trace timing,
  transformation, frozen-baseline parity and metric audit.
- `scripts/report-oracle-online-calibration.mjs`: selection, hash, maturity,
  tape and replay consistency checks and complete certificate collection.

Recorded artifacts live under
`data/training/runs/stochastic-crra-online-calibration-v1/` and
`data/backtests/oracle-online-calibration/`.

```powershell
node scripts/run-ml-python.mjs -m unittest ml/test_online_hurdle_calibration.py
node scripts/run-ml-python.mjs ml/evaluate_online_hurdle_calibration.py --prepare-only
node scripts/run-ml-python.mjs ml/evaluate_online_hurdle_calibration.py --run-prepared --validation-only
node scripts/run-ml-python.mjs ml/evaluate_online_hurdle_calibration.py --evaluate-test
node scripts/run-ml-python.mjs ml/audit_online_hurdle_calibration.py
node scripts/run-oracle-neural-specialists.mjs --output data/backtests/oracle-online-calibration
```

The plan and test cache refuse overwriting/repeated evaluation. Reproduction
requires new output directories passed via `--output` and `--forecasts`,
with a corresponding replay plan.
