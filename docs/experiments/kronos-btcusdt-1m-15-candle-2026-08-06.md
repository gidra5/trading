# Kronos BTCUSDT 1m, 15-candle forecast experiment

Date: 2026-08-06

## Scope

This experiment installs every public Kronos predictor (`mini`, `small`, and
`base`) at pinned source/model revisions and forecasts 15 consecutive BTCUSDT
one-minute candles. The benchmark derives its 28 non-fit date ranges from the
same inspector-window definitions used by the web UI. The four fit folds and
the latest three-month window are intentionally excluded.

Kronos itself was published and evaluated at 5-minute and coarser intervals.
The one-minute results below are therefore a local domain-adaptation experiment,
not a reproduction of the paper's reported accuracy.

## Correct probabilistic protocol

The paper's price/return configuration uses temperature `0.6`, top-p `0.9`, and
an average of multiple paths. Local calibration found temperature `0.8` with
20 retained paths to be the best joint candle/distribution setting on the
deterministic screen. The benchmark does not call the upstream convenience
path that discards samples after averaging. It retains every path and computes:

- ensemble mean, minimum-distance OHLC-projected mean, ensemble median, and
  KQSP median point forecasts; the projected mean is the primary because it is
  identical to the more accurate mean on already-valid rows and minimally
  repairs the roughly 0.4% of sparse-screen mean candles that were invalid;
- candle and close-path anchored-log MSE against persistence;
- close-return MSE, direction accuracy, and correlations;
- the paper-aligned per-path price IC/RankIC and horizon-return IC/RankIC;
- empirical quantiles, pinball loss, coverage, interval width, and CRPS;
- the mean oracle distribution across sampled return paths and its forward KL
  from the realized oracle distribution.

Raw sampled candles can violate OHLC constraints. [KQSP](https://arxiv.org/html/2607.26792v1)
applies exact Euclidean projection to every quantile OHLC row followed by an
unweighted PAVA/isotonic projection down each feature's quantiles. This matches
the paper's two-stage, parameter-free method—including its published numerical
examples—and makes reported quantile candles 100% valid. Randomized property
tests also verify both constraints, idempotence, and a correction no larger
than one-sided clipping. It is a constraint repair, not a claim that the
underlying price prediction became more accurate. The execution oracle consumes
only each sampled close path, so invalid high/low ordering cannot alter its
action values; a regression test holds close fixed, deliberately corrupts the
other OHLC fields, and verifies bit-identical vote and expected-utility
distributions. Projecting individual raw paths would instead move valid close
forecasts unnecessarily.

## Policy-episode-purged fine-tuning

Predictor adaptation uses BTCUSDT 1-minute windows with 512 historical candles,
15 targets, and the additional next-token position required by teacher forcing.
Normalization statistics come strictly from the 512 historical candles, which
matches inference and prevents target leakage. The pretrained tokenizer is
frozen; AdamW updates the predictor with the upstream two-level next-token
cross-entropy objective.

- training: 2021-07-01 through 2023-12-31;
- chronological validation: 2024-01-01 through 2024-06-30;
- later chronological subset: inspector windows starting on or after 2024-07-01.

An audit of the original light pilot found that 437 of its 4,000 deterministic
training sequences intersected a policy-calibration UI window, touching all 18
train-era windows. That does not leak the final post-July 2024 validation, but
it makes earlier policy PnL partly in-sample to the adapted predictor. The final
checkpoint therefore uses the content-addressed
`kronos-btcusdt-1m-policy-holdout-v1.json` exclusion plan. It is generated from
the same UI catalog and merges the 20 pre-validation windows into 13 episodes;
no training or validation sequence may touch any of their candles. This removes
120,997 possible training starts and 9,694 validation starts. A cross-language
test requires the plan to remain exactly equal to the current inspector catalog.

The clean two-epoch base retrain lowered deterministic chronological-validation
loss from `2.425293` to `2.397107` (`1.16%`) and its future-position component
from `2.474837` to `2.361090` (`4.60%`). Its 409,264,008-byte predictor has
SHA-256 `ed1ba73a21ec027b0a0471724c48c8a259724c679234da3ac3dd8eb3e81bd1db`.
These token losses are checkpoint diagnostics only; the paired downstream
forecast screen and causal bot gates determine whether the checkpoint is used.

The tokenizer stays frozen because Kronos pretraining already includes Binance
one-minute data, predictor adaptation produced a strong gain, and KQSP handles
the invalid-candle output contract. A sequential ablation fine-tuned the
tokenizer first and then retrained the predictor against its new vocabulary.
Tokenizer validation reconstruction MSE improved from `0.004415` to `0.003872`,
but later-window candle skill fell to `-67.04%` and price IC to `0.0322`.
Improved token reconstruction therefore did not improve forecasting, and that
pair was rejected.

The paired screens and early bot diagnostics below used the original pilot to
choose architecture, sampling, and policy families. They remain useful
exploratory evidence, including on genuinely later windows, but the original
checkpoint and its interrupted 3,810-row dense export are explicitly rejected
for final policy selection. Final calibration and validation use only the
episode-purged v2 checkpoint and its newly generated forecast artifact.

## Paired calibration-screen results

These values use the same seed, 20 paths, temperature 0.8, and 106 unique
origins (four memberships in each of 28 inspector windows). They are useful for
model/configuration selection; the exhaustive run is a separate artifact. This
first screen predates deterministic per-batch seeding, but each zero-shot/tuned
pair still started from the same seed and used identical batching.

| predictor | candle skill vs persistence | price IC | horizon IC | return correlation | oracle KL | CRPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mini, zero-shot | +0.85% | -0.0180 | 0.1662 | 0.0696 | 1.6570 | 0.001122 |
| small, zero-shot | -23.02% | 0.0409 | 0.0644 | 0.0116 | 1.5724 | 0.001162 |
| base, zero-shot | -10.24% | 0.0378 | 0.0910 | 0.0737 | 1.4190 | 0.001126 |
| small, light predictor tune | -5.60% | 0.0649 | 0.0788 | 0.0539 | 1.4506 | 0.001084 |
| base, light predictor tune | **+2.09%** | **0.0686** | 0.0371 | **0.0946** | **1.3405** | **0.001064** |

On the eight truly post-pretraining inspector windows, light base adaptation
changes macro candle skill from -66.29% to -0.59%, price IC from 0.0402 to
0.1092, oracle KL from 0.7383 to 0.6179, and CRPS from 0.000698 to 0.000557.
Horizon IC remains negative (-0.0997), so this is not evidence of a profitable
standalone trading signal.

### Deterministic confirmation

The winning base pair was rerun after adding target-time-derived batch seeds and
within-model resume. Both checkpoints therefore receive exactly paired random
draws independent of interruption. On the same 106 origins, zero-shot versus
lightly tuned base produced:

| metric | zero-shot base | tuned base |
| --- | ---: | ---: |
| candle skill vs persistence | -5.05% | **+3.00%** |
| candle anchored-log correlation | 0.2513 | **0.2648** |
| price IC | 0.0662 | **0.0712** |
| horizon-return IC | **0.1285** | 0.1022 |
| return MSE skill vs zero | +0.05% | **+0.21%** |
| oracle KL | 1.4811 | **1.4709** |
| CRPS | 0.001101 | **0.001064** |

Across post-June-2024 windows, tuning changes macro candle skill from -57.35%
to -20.72%, price IC from 0.0686 to 0.1404, horizon IC from -0.4038 to
-0.3478, exposure MAE from 34.49 to 28.85, and CRPS from 0.000702 to
0.000587. Post-period oracle KL moves slightly in the wrong direction (0.8513
to 0.8731), so the tuned checkpoint wins on the overall metric balance rather
than every individual measurement.

A longer small-model run reached better validation token loss and almost zero
aggregate candle-skill deficit, but degraded correlation and unseen-window
metrics. Checkpoint selection must therefore use the downstream forecast table,
not token loss alone. Mini adaptation was also rejected because it worsened
candle and horizon metrics.

### Predictor ensemble

A deterministic 50/50 path ensemble retains ten samples from the tuned base
predictor and ten from the pretrained base predictor. Compared with the tuned
predictor alone, it gives up some point-forecast skill but improves the
distributional oracle objective:

| configuration | candle skill | price IC | horizon IC | return correlation | oracle KL | CRPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tuned base | +3.00% | 0.0712 | 0.1022 | 0.1211 | 1.4709 | 0.001064 |
| 50/50 tuned/pretrained | -0.82% | **0.0906** | 0.0956 | 0.1198 | **1.3166** | 0.001082 |

The 50/50 ensemble is selected for the bot's probabilistic oracle distribution;
the tuned checkpoint remains the better point-forecast candidate. A 75/25
ensemble was rejected because it neither retained tuned candle skill nor
matched the 50/50 oracle improvement.

### Clean-checkpoint downstream confirmation

After purging the policy episodes and retraining, the pretrained, clean tuned,
75/25, and 50/50 configurations were rerun with identical target-time seeds.
The mixture was selected strictly on the 20 pre-July-2024 calibration windows:

| configuration | candle skill | return correlation | horizon IC | price IC | oracle KL | CRPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pretrained | **-4.44%** | 0.0364 | 0.1829 | 0.0941 | 1.7195 | **0.001312** |
| clean tuned | -14.82% | **0.0462** | 0.1212 | 0.0751 | 1.7319 | 0.001350 |
| 75/25 tuned/pretrained | -17.05% | 0.0226 | 0.1160 | 0.1121 | 1.6558 | 0.001357 |
| **50/50 tuned/pretrained** | -22.51% | 0.0111 | **0.1837** | **0.1327** | **1.5857** | 0.001353 |

The clean 50/50 mixture therefore remains the bot configuration: it has the
best calibration price IC and oracle KL and preserves the best horizon IC,
even though it is not the best point-MSE model. On the eight later windows,
which were inspected only after this calibration choice, clean tuned alone has
the best candle skill (`-12.99%`) and CRPS (`0.000593`), while 50/50 has the
best oracle KL (`0.7181`). This is the expected separation between the point
forecast and probabilistic trading objectives, not a blanket fine-tuning win.

### Sample-count sanity check

The earlier zero-shot calibration screen evaluated the same 217 unique origins
with 10 and 20 retained paths. Increasing to 20 improved the oracle forward KL
from `1.8949` to `1.5589` for mini and from `1.8064` to `1.3988` for small. It
also improved candle skill from `-8.91%` to `-6.80%` and from `-20.25%` to
`-15.82%`, respectively. Not every metric moved monotonically (for example,
small price IC fell from `0.1075` to `0.0747`), so this is not a claim that
arbitrarily many samples improve the underlying model. Twenty paths are the
selected compute/Monte-Carlo compromise; sampling reduces distribution
estimation noise but cannot repair a biased conditional forecast.

### Context-length calibration

The 512-candle context was compared with 128, 256, and 384 candles using the
same deterministic seed policy, 20 retained paths, and four origins per
inspector window. Selection used only the 20 policy-calibration windows before
2024-07-01. Values below are macro means over those windows:

| lookback | candle skill | horizon correlation | horizon direction | price IC | oracle KL | CRPS | origins/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | -5.89% | 0.0461 | 57.50% | 0.0669 | 1.7411 | 0.001397 | 0.944 |
| 256 | -27.74% | -0.0449 | 55.00% | 0.0569 | 1.9171 | 0.001360 | 0.511 |
| 384 | -43.01% | 0.0756 | 56.25% | 0.0398 | 2.0999 | 0.001397 | 0.336 |
| **512** | -25.40% | **0.1863** | **58.75%** | **0.1154** | **1.5722** | 0.001362 | 0.279 |

The 128-candle model has the best candle MSE and is much faster, but loses most
of the horizon and oracle signal needed by the bot. The full 512-candle context
therefore remains selected. Batch size four was also measured and did not
materially improve sustained 512-candle throughput: on the paired 28-origin
execution smoke it reached `0.255` origins/s versus `0.248` for batch two, only
`2.7%` faster. The clean dense export uses explicit batch size two as the stable
memory/throughput compromise. Requested batch layout is part of the run
signature because changing it also changes target-time-grouped stochastic
draws; a mismatched artifact therefore cannot be resumed accidentally.

### Horizon-weighted fine-tuning ablation

The predictor trainer can add the cross-entropy of the exact future 15 token
positions to the upstream all-position objective. We tested weights `0.25` and
`1.0`, then paired each checkpoint 50/50 with pretrained base under the same
screen. Calibration-window macro metrics were:

| forecast loss weight | candle skill | horizon correlation | horizon direction | price IC | oracle KL | CRPS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **0 (upstream objective)** | -25.40% | **0.1863** | 58.75% | **0.1154** | 1.5722 | 0.001362 |
| 0.25 | -6.95% | 0.1036 | 58.75% | 0.0209 | 1.6005 | 0.001367 |
| 1.0 | -8.91% | 0.1024 | 58.75% | 0.0531 | **1.5577** | **0.001357** |

Both weighted objectives improve validation token loss and candle-level MSE,
but discard roughly half of the useful horizon correlation and price IC without
improving direction accuracy. They are rejected; the original predictor pilot
is retained only as exploratory evidence. The episode-purged v2 checkpoint uses
the upstream weight-zero objective and regenerates its dense export from scratch.

## Causal bot integration

Every forecast row contains only information available at the preceding candle
close: `decisionTime = targetStartTime - 1`. The artifact excludes actual or
realized target fields and the TypeScript loader rejects unknown row fields,
off-cadence origins, missing inspector-window memberships, and incomplete dense
coverage.

The `-100..+100` evaluation oracle models the project's perfect-margin target,
but it is not directly executable by the production bot's 5x leverage cap.
Conditioning that native 100x action grid on entry friction selected zero for
all 106 sparse forecasts. The export now also computes an executable
`-5..+5` oracle from the same retained stochastic paths, using the same 17.5 bp
fee-plus-slippage friction. Its scored base action is held for the complete
15-candle decision interval. An earlier export incorrectly held it for one
candle and then allowed a clairvoyant minute-by-minute Bellman continuation;
the bot cannot rebalance between 15-minute forecasts. On 570 forecast centers,
that mismatch changed the expected-exposure direction 12.6% of the time. The
run signature includes this configuration, the artifact loader rejects the old
cadence, and the corrected dense export restarts from origin zero. This
preserves the perfect-margin distribution for evaluation while giving the bot
a correctly scaled and cadence-matched action distribution.

The export retains two executable distributions. The oracle-vote distribution
is the arithmetic mean of the per-sampled-path perfect-action distributions.
The expected-utility distribution averages each action's terminal log wealth
across all 20 sampled paths and then applies the oracle softmax; this is the
Kelly-style action under forecast uncertainty. The sampled oracle-vote
and expected-utility arrays are deliberately stored before entry-transition
conditioning. At each live or replay decision, the bot reconditions that base
distribution on its actual current exposure and the same 17.5 bp friction,
then scales the resulting native `-5..+5` target. This preserves the
exposure-dependent cost of holding, reducing, or reversing a position instead
of assuming that every forecast begins flat.

The sampled oracle-vote
distribution is broad: its raw modal action is often zero or an endpoint even
when the distribution mean carries a more stable directional vote. Policy
calibration therefore compares both complete distributions, the rounded mean
and sign of each executable distribution, and the horizon mean/median
directions. Weak forecasts can either flatten or hold the
current position, and a one-to-twelve-decision same-direction confirmation gate
provides causal turnover hysteresis. The raw bounded grid contains 504 unique
policies. Full-confidence candidates additionally compare the bot's established
`0.75` gradual-expansion cap against an unrestricted one-step expansion, bringing
the raw grid to 630 unique policies without multiplying lower-confidence
equivalents.

A second, optional signal family corrects stable Kronos bias with a 16-feature
ridge mapper from forecast-only summaries to the realized 15-minute log return.
Six regularization strengths are evaluated against a sample-count-normalized
ridge objective. Every return used for policy
selection is strictly out of fold across the 11 episodes that end before
predictor training stopped: the mapper that predicts one merged market episode
is trained on the other ten, so no row or overlapping inspector window can
train its own signal. The mapper is then fit on those 11 episodes and frozen
before either January/February 2024 confirmation episode is evaluated. Its
content-addressed coefficients are reused unchanged for final validation. With
the gradual-expansion alternative, this adds 360
calibrated-return policies for 990 total candidates. Calibrated return remains a
candidate only and is discarded when its out-of-fold bot score loses to the raw
Kronos policies.

The integration smoke test runs the real `GridTradingBot` and
`SimulatedTradingApi`, not an analytical PnL shortcut. It has produced persisted
entry and market-exit fills with fees, maintenance, realized PnL, and no
liquidation. Its one sparse forecast was intentionally held for almost a week
and lost money due to maintenance; it is execution proof, not the final trading
result. The dense artifact supplies a new causal decision every 15 minutes.

The first clean-checkpoint progress snapshot supplied 176 consecutive decisions
from the September 2021 episode. Its explicitly non-freezable, locally selected
diagnostic policy made seven actual bot fills and returned `+15.89%` after
`$324.94` fees and `$425.47` maintenance, with `6.41%` maximum drawdown and no
liquidation. Constant long and short controls at both 1x and 5x all lost money
on the same slice. Fill records exactly match the reported trade count and fee
total. This proves the leakage-free forecast can drive profitable execution on
one partial calibration slice, but local selection makes it smoke evidence only;
it cannot contribute to frozen-policy selection or held-out validation.

The same locally selected policy was then frozen and replayed from a fresh
account on the next 60 non-overlapping decisions (15 hours). It made four fills
and lost `0.65%` net after `$168.40` fees and `$273.50` maintenance, with
`8.25%` drawdown and no liquidation. Its pre-cost edge was positive and it still
beat every constant control (best control `-1.04%`), but costs erased the gain.
This temporal confirmation weakens the headline local result and demonstrates
why the final selector evaluates turnover and maintenance across 13 complete
episodes. The confirmation result is immutable diagnostic evidence and cannot
modify the final policy grid or validation.

Extending that same frozen confirmation to the next 90 decisions (22.5 hours),
without changing the policy or its start, preserved the same four-fill position
sequence but ended at `+8.82%` net after `$166.75` fees and `$464.62`
maintenance. Maximum drawdown was `10.02%`, there was no liquidation, and all
constant controls lost money (best `-1.91%`). The 60-row loss and 90-row profit
together expose strong forced-exit/end-time sensitivity; only the predefined
complete episode boundaries are eligible for final conclusions.

Before further rows were inspected, the definitive temporal-confirmation length
was fixed at 176 decisions to match the policy-selection slice. On exactly the
next 176 non-overlapping decisions (44 hours), the unchanged policy made six
actual fills and returned `+7.13%` net after `$317.58` fees and `$539.77`
maintenance. Maximum drawdown was `10.02%`, there was no liquidation, all
forecast rows were consumed, and every constant control lost money (best
`-2.47%`). The source slice ends exactly where confirmation begins, and the
content-addressed confirmation report has SHA-256
`93f9d9bd32cf66c08330c12e154247f464e1378f729b8c16dce3274bf1d14cd5`.
This is real forward execution evidence within one calibration episode, not a
substitute for the six independent final episodes.

An explicitly non-freezable v2 progress diagnostic replayed the first 150
consecutive forecasts (37.5 hours) of the September 2021 calibration episode.
Across all 464 policies, the best diagnostic policy produced two actual fills,
`+14.80%` net return, `4.93%` maximum drawdown, `$135.75` fees, `$213.64`
maintenance, and no liquidation. The best expected-utility policy returned
`+3.64%`, versus `+3.30%` for the best raw oracle-vote policy, so the additional
aggregation is behaviorally useful. This was one well-timed short rather than
broad forecast accuracy: mean-path MSE skill was `-9.47%`, horizon correlation
was `0.0094`, and direction accuracy was `55.3%`. The slice is execution and
pathology evidence only; it is not used as proof of general profitability or as
the frozen policy.

Once the first complete window became durable, the same diagnostic was repeated
over all 672 scheduled origins (the complete September 8--15, 2021 episode).
The winning expected-utility policy produced 10 real simulator fills, two
profitable round trips, and `+7.88%` net return after `$119.45` fees and
`$255.04` maintenance. Maximum drawdown was `6.72%` and there was no
liquidation. The best raw oracle-vote, horizon-mean, and horizon-median families
returned `+6.03%`, `+3.41%`, and `+1.15%`, respectively. Raw point prediction
remained weak across the full window: close-path MSE skill versus persistence
was `-9.36%`, horizon-return skill versus zero was `-14.70%`, horizon
correlation was `0.0348`, and direction accuracy was `51.93%`. This supports
the probabilistic expected-utility integration, but it is still a
single-window diagnostic and is not eligible for final policy selection.

The next complete independent calibration episode, October 19--22, 2021,
contained 288 origins. Its local oracle-sign winner made four simulator fills
and returned `+15.57%` net, with `19.37%` maximum drawdown, `$160.93` fees,
`$1,519.06` maintenance, and no liquidation. The best complete
expected-utility policy returned `+1.77%` with `5.20%` drawdown. Point forecasts
again did not explain the local trading result: close-path skill was `-18.83%`,
horizon-return skill was `-15.49%`, horizon correlation was `0.0521`, and
direction accuracy was `50.69%`. The differing first- and second-episode
winners demonstrate why neither local result is frozen; selection waits for
the complete multi-episode calibration.

The first episode's expected-utility winner ranked tenth on the second episode
without modification and returned another `+1.77%` at `5.20%` drawdown. Its
two-episode independent-equity geometric mean is therefore `+4.78%`. This is a
more useful early robustness observation than either locally selected maximum,
but two episodes are still insufficient for freezing the policy.

The third complete independent episode, December 14--21, 2021, contained 672
origins. Its local horizon-median winner made nine simulator fills and returned
`+11.17%` net after `$339.79` fees and `$3,153.29` maintenance, with `18.11%`
maximum drawdown and no liquidation. The best expected-utility policy returned
`+5.01%` with `4.20%` drawdown; in fact, the best policy from every raw signal
family was profitable on this episode. Point prediction remained weak:
close-path skill was `-8.84%`, horizon-return skill was `-8.64%`, horizon
correlation was `0.1007`, and direction accuracy was `54.02%`. This is another
useful execution diagnostic, not a locally selectable final policy.

The fourth complete independent episode, May 14--21, 2022, also contained 672
origins. Its local expected-utility-mean winner made 18 simulator fills and
returned `+11.12%` net after `$164.02` fees and `$521.40` maintenance. Maximum
drawdown was `4.16%` and there was no liquidation. The best candidate from
every raw signal family was profitable, while all four constant-exposure
controls lost money. Forecast metrics were particularly poor: close-path skill
was `-16.23%`, horizon-return skill was `-24.89%`, horizon correlation was
`-0.0474`, and direction accuracy was `50.60%`. The complete result replacing
the earlier partial diagnostic again shows that bot utility and turnover must
be evaluated independently of point-forecast MSE.

The fifth complete independent episode, June 7--14, 2022, was the earlier known
selloff miss and contained 672 origins. Its local horizon-mean winner made four
simulator fills and returned `+32.55%` net after `$274.76` fees and `$2,296.58`
maintenance, with `12.00%` maximum drawdown and no liquidation. The raw
execution-sign and oracle-vote family winners returned `+25.12%` and `+6.87%`;
the best expected-utility vote returned only `+0.12%`, while the
expected-utility-mean and expected-utility-sign family winners lost money.
Point prediction was still below simple baselines: candle skill was `-14.78%`,
close-path skill was `-17.45%`, horizon-return skill was `-20.75%`, horizon
correlation was `-0.0066`, and horizon direction accuracy was `52.68%`. Raw
sample paths contained at least one invalid candle on `85.42%` of origins, while
the KQSP projection produced `100%` valid quantile candles. This fixes the
earlier execution miss locally but is still a per-window diagnostic, not
out-of-sample evidence for the policy that will be frozen across all calibration
episodes.

The overlapping June 11--14 three-day known-miss UI window was also replayed
over all 288 origins. Its locally selected execution-utility-sign policy made
110 fills and returned `+25.76%` after `$2,499.48` fees and `$1,115.65`
maintenance, with `30.03%` maximum drawdown and no liquidation. This is not
incremental evidence of signal value: constant -5x returned `+82.39%` over the
same selloff with similar `29.96%` drawdown, while constant -1x returned
`+16.68%`. Candle, close-path, and horizon-return skills were `-10.24%`,
`-13.49%`, and `-14.96%`; horizon correlation was `0.0485` and direction
accuracy was `50.35%`. It remains a separately visible UI-window report, but the
calibration merges it with the seven-day June episode so the overlapping price
move cannot be counted twice.

The overlapping June 13--16 high-churn down-shape UI window contained 288
origins. Its locally selected expected-utility policy made 19 fills and returned
`+10.05%` after `$194.61` fees and `$69.17` maintenance, with `9.02%` maximum
drawdown and no liquidation. Every raw signal family except the two equivalent
mean-distribution sources had a profitable local candidate. The best constant
control, -1x, returned `+4.16%`; +1x, +5x, and -5x returned `-15.36%`, `-71.91%`,
and `-7.35%`. Point accuracy remained weak: candle, close-path, and
horizon-return skills were `-6.25%`, `-9.86%`, and `-12.06%`, horizon
correlation was `0.0301`, and direction accuracy was `50.69%`. This is useful
probabilistic-execution evidence, but it overlaps the same June episode and
cannot select the final policy by itself.

The overlapping June 12--19 down-regime UI window contained 672 origins. Its
local oracle-vote winner made 32 fills and returned `+24.95%` after `$274.25`
fees and `$798.98` maintenance, with `13.59%` maximum drawdown and no
liquidation. Constant -1x returned `+15.95%`; constant -5x returned only
`+19.01%` despite 1,759 fills, `$11,957.56` combined costs, and `57.87%`
drawdown. The new `0.75` expansion-cap variant of the winning family returned
`+24.09%` with the same 32 fills and slightly lower `13.05%` drawdown. Candle,
close-path, and horizon-return skills were still negative at `-8.57%`,
`-11.80%`, and `-13.78%`; horizon correlation was `0.0165` and direction
accuracy was `51.79%`. The execution result therefore beats the trivial regime
controls even though point forecasting remains weak, but the overlapping window
still merges into the single June calibration episode.

The adjacent June 19--22 high-churn up-shape UI window contained 288 origins.
Its local oracle-vote winner used twelve-decision confirmation, made six actual
bot fills, and returned `+1.91%` after `$40.02` fees and `$158.44` maintenance,
with `3.33%` maximum drawdown and no liquidation. The expected-utility family
also remained profitable at `+1.13%`; horizon mean was barely positive and the
other point/sign families lost money. Constant +1x returned `+8.87%`, so this
particular result did not beat passive knowledge of the upward regime. It is
positive execution and regime-direction evidence only, and its contiguous date
range is merged into the same independent June calibration episode.

Constant-exposure controls show that these calibration diagnostics are not
explained by a trivial directional position. On the first episode, constant +1x returned only
`+0.30%` and constant -1x lost `-19.22%`; the ±5x controls lost `-53.88%` and
`-69.96%`. On the second episode, +1x and -1x lost `-0.10%` and `-9.24%`, while
the ±5x controls lost `-34.86%` and `-46.69%`. These controls continuously
retarget exposure through the real bot rather than approximating PnL from the
endpoint price.
On the third episode, +1x returned `+0.22%`, -1x lost `-19.17%`, and the
constant +5x and -5x controls lost `-55.93%` and `-68.65%`, respectively.
On the fourth episode, the +1x, -1x, +5x, and -5x controls returned `-0.79%`,
`-18.47%`, `-60.49%`, and `-69.01%`, respectively.
On the fifth selloff episode, +1x lost `-28.48%`, while -1x returned `+12.30%`.
The +5x control lost `-89.81%`; -5x returned `+34.80%`, but with `54.02%`
maximum drawdown, 1,138 fills, and much larger costs than the local Kronos
winner.
On the adjacent up-shape window, +1x returned `+8.87%`, while -1x, +5x, and
-5x returned `-17.26%`, `-16.98%`, and `-59.45%`, respectively; the local
Kronos policy was profitable but did not beat the low-leverage long control.

Policy calibration uses only the 20 inspector windows before 2024-07-01. Because
several 3-day, 7-day, shape, and regime windows overlap the same market move,
it merges them into 13 non-overlapping chronological episodes rather
than counting the same PnL multiple times. All 20 individual UI-window reports
are still emitted. The 11 episodes ending before the predictor's 2024-01-01
training cutoff rank candidates. A candidate must have positive geometric mean return, trade
in at least half of the episodes, profit in at least half of its active
episodes, average at least one fill per episode, and survive the explicit
downside, drawdown, median, and chronological-fold penalties. Any liquidation
disqualifies a policy outright. The same complete activity, profitability,
fill, and no-liquidation gate must then pass independently on the two
January/February 2024 episodes excluded from predictor training. Those two
episodes never contribute to candidate scores or return-calibrator coefficients;
they are a genuine guard rather than a second objective to maximize. The selected policy is
frozen into a forecast-run-specific
artifact before the eight later windows are run once as six independent
episodes for headline bot-PnL validation; all eight UI-window reports are also
emitted. Those later bot returns are not used for policy selection, although
their forecast-quality metrics were inspected during earlier model comparison.
The validation filename is fixed by the forecast run signature and an existing
canonical result is never overwritten, so supplying a different `--output`
cannot bypass the one-shot guard.
The frozen policy additionally stores the SHA-256 of the complete forecast
artifact, preventing a different Monte Carlo export from being substituted even
if it shares the same older run-signature metadata.
The one-shot validation report reciprocally stores the SHA-256 of the complete
policy artifact, binding its fills not only to the selected parameter ID but to
the exact ranking episodes, confirmation evidence, gates, and optional learned
coefficients that authorized the run.
The canonical report is still written when the frozen policy has zero fills,
liquidates, or loses money; explicit coverage, fill, liquidation, and positive
PnL gates preserve an unfavorable result instead of turning it into a rerunnable
exception.
After publication, the final orchestrator independently verifies all six merged
episodes, all eight inspector memberships, complete forecast consumption,
nonempty fill arrays, fill/trade counts, aggregate fee and maintenance totals,
zero liquidations, and all four directional controls. A held-out loss remains a
valid immutable report; it is surfaced rather than converted into a retry.
The same one-shot invocation also records constant long and short controls at
1x and 5x through the identical bot simulator. They run only after the policy
is frozen and are never candidates, but distinguish genuine signal value from
a period that merely rewarded constant directional exposure.

## Reproduce

```bash
npm run kronos:setup
npm run kronos:benchmark -- --models all --temperature 0.8 --sample-count 20

npm run kronos:finetune -- --model base --epochs 2 \
  --train-samples-per-epoch 2000 --validation-samples 256 \
  --batch-size 1 --gradient-accumulation 32 \
  --exclusion-plan ml/training-plans/kronos-btcusdt-1m-policy-holdout-v1.json \
  --output-dir .tools/Kronos-finetuned/btcusdt-1m-base-policy-holdout-v2

npm run kronos:benchmark -- --models base \
  --model-checkpoint .tools/Kronos-finetuned/btcusdt-1m-base-policy-holdout-v2/best_model \
  --model-label base-policy-holdout-v2 --temperature 0.8 --sample-count 20

npm run kronos:benchmark -- --models base \
  --model-checkpoint .tools/Kronos-finetuned/btcusdt-1m-base-policy-holdout-v2/best_model \
  --ensemble-predictor-checkpoint pretrained --ensemble-sample-count 10 \
  --model-label base-policy-holdout-pretrained-ensemble \
  --temperature 0.8 --sample-count 20 --batch-size 2 \
  --forecast-output data/benchmarks/kronos-base-policy-holdout-ensemble-dense-execution-forecasts.json \
  --output data/benchmarks/kronos-base-policy-holdout-ensemble-dense-execution-metrics.json

# Or resume and chain dense inference, calibration, and one-shot validation:
npm run kronos:final-pipeline

# Re-run the strict completion audit against explicit canonical artifacts:
npm run kronos:audit -- \
  --forecasts data/benchmarks/kronos-base-policy-holdout-ensemble-dense-execution-forecasts.json \
  --metrics data/benchmarks/kronos-base-policy-holdout-ensemble-dense-execution-metrics.json \
  --policy data/benchmarks/KRONOS_POLICY.json \
  --validation data/benchmarks/KRONOS_VALIDATION.json

npm run kronos:backtest -- --phase=calibrate \
  --forecasts=data/benchmarks/kronos-base-policy-holdout-ensemble-dense-execution-forecasts.json

npm run kronos:backtest -- --phase=validate \
  --forecasts=data/benchmarks/kronos-base-policy-holdout-ensemble-dense-execution-forecasts.json \
  --policy=data/benchmarks/KRONOS_POLICY.json
```

The benchmark writes atomic accumulator progress every two minutes and durable
per-model partials next to its JSON output. It resumes matching runs by default.
Every batch seed is a SHA-256 derivation of the benchmark seed and exact target
timestamps, so a resumed run generates the same stochastic paths. The selected
checkpoint manifest records source revisions, ranges, hyperparameters, and
validation history. Atomic publication retries transient Windows destination
read locks with bounded backoff. This was exercised in practice when a monitor
briefly blocked `os.replace`: the fully serialized 3,360-row temporary state was
validated across every accumulator, promoted with the 3,330-row checkpoint kept
as rollback, and the identical run signature then advanced successfully to a
new 3,390-row checkpoint. The TypeScript policy calibrator uses the same bounded
retry discipline for its resumable candidate-grid progress, with a regression
test that injects repeated destination-lock failures before a successful publish.
The final pipeline invokes `kronos:audit` after validation. That audit verifies
the exact five pinned model/tokenizer snapshots, all-model sparse metrics,
exclusion-plan and clean-checkpoint hashes, every dense causal origin and UI
window membership, KQSP validity, the frozen policy's forecast hash, the exact
held-out split, every bot fill, aggregate fees and maintenance, directional
controls, and the no-liquidation and truthful-profit gates. It writes a compact
`kronos-final-completion-audit-v1` artifact keyed by the dense run signature.

## Calibration artifacts

- `data/benchmarks/kronos-calibration-all-t08-n20-n4.json`
- `data/benchmarks/kronos-finetuned-small-pilot-t08-n20-n4.json`
- `data/benchmarks/kronos-finetuned-small-final-t08-n20-n4.json`
- `data/benchmarks/kronos-finetuned-mini-final-t08-n20-n4.json`
- `data/benchmarks/kronos-finetuned-base-pilot-t08-n20-n4.json`
- `data/benchmarks/kronos-base-zero-deterministic-t08-n20-n4.json`
- `data/benchmarks/kronos-base-tuned-deterministic-t08-n20-n4.json`
- `data/benchmarks/kronos-base-sequential-deterministic-t08-n20-n4.json`
- `data/benchmarks/kronos-base-ensemble-deterministic-t08-n20-n4.json`
- `data/benchmarks/kronos-base-ensemble75-deterministic-t08-n20-n4.json`
- `data/benchmarks/kronos-bot-execution-oracle-smoke.json`
