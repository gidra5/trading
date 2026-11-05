# Forecast revisions at a fixed future timestamp

The current latent NSF usually revises a distant price distribution only
slightly after one additional observed second. This supports testing Monte
Carlo corrections with shared random draws. It does not establish a bound on
the change in the optimal trading action or its Bellman value.

For a target initially one hour ahead, the median distribution shift was
**0.1936 bps**, versus a forecast standard deviation of **30.813 bps**. The
95th-percentile shift was **0.8674 bps**. Large observed moves are exceptions:
the 32 largest-move stress cases had a median one-hour revision of **5.948 bps**
and maximum **12.751 bps**.

The one-second rollout kernel also supports effective coupling. Across two
random seeds, the median variance of a one-hour terminal-price correction was
**0.0002645 times** the variance of a fresh terminal-price draw. This is about
**3,780 times less variance per draw**, conditional on an accurately known old
expectation. Threshold events benefited much less and sometimes had no gain.

## What was compared

For each real origin `t` and lead `h`, compare

    law(price at t+h | observations through t)
    law(price at t+h | observations through t+1).

The second query uses horizon `h-1`, and both distributions use the same price
units and target timestamp. Both encoders receive only the two completed
observations available at their own origin. The new observation is used only
in the later forecast.

The trained checkpoint is unchanged:

    data/backtests/oracle-latent-nsf-sampled-skips/latentSampledSkipFlow.pt
    SHA256 80fdc3df78a5f60d9980b236f9bbaa296904dc562bd64195394831120dde01db

Two distinct forecast laws were tested:

1. **Direct marginal:** ask the NSF for the entire requested horizon in one
   transition, integrating posterior, transition and decoder uncertainty.
2. **Planner's one-second kernel:** repeatedly sample all decoded market
   observations, re-encode their two-row history, and advance exactly one
   second. No graph compression or representative-node selection is used.

They are not assumed to be the same law. For example, the median one-hour
standard deviation was about 31 bps for direct queries and 25 bps for these
rollouts. The direct statistics cannot be substituted for rollout statistics
in a planning-accuracy claim.

## Direct marginal experiment

The main sample contains **2,400 adjacent-second pairs**: forty seeded,
stratified 60-second windows across the available test interval, with enough
remaining history to keep the one-hour endpoint within that interval. Each
pair has its own target `t+h`; that target is held fixed within the comparison.
This is not a measurement of accumulated drift over sixty successive updates
to one common endpoint. Another **32 pairs** are the largest absolute observed
one-second moves in the eligible interval. Stress selection is retrospective
and is reported separately, not treated as an unbiased frequency estimate.

Each forecast uses 512 latent mixture components and 4,096 observation draws.
Old and new forecasts use identical Sobol coordinates. Distribution shift is
the empirical 1-Wasserstein distance in log price, multiplied by 10,000:
the mean absolute separation of equally ranked samples, in basis points.
It measures the whole one-dimensional distribution, not just its mean.

| Target initially ahead | Median shift (bps) | 95th percentile (bps) | Median forecast SD (bps) |
| --- | ---: | ---: | ---: |
| 2 seconds | 0.1327 | 0.4361 | 0.583 |
| 5 seconds | 0.0819 | 0.4361 | 0.934 |
| 10 seconds | 0.0585 | 0.4364 | 1.348 |
| 1 minute | 0.0256 | 0.4341 | 3.518 |
| 5 minutes | 0.0306 | 0.4268 | 8.300 |
| 15 minutes | 0.0652 | 0.5166 | 14.852 |
| 1 hour | 0.1936 | 0.8674 | 30.813 |

One basis point is 0.01%. Thus the median one-hour revision corresponds to
about **0.00194%** of price, while its forecast standard deviation is about
**0.308%**. After subtracting each distribution's own mean, the median one-hour
shape distance was **0.0503 bps**. Much of the revision is a price-level shift.

Closeness in basis points is not closeness for every event. At the two-second
lead, the 95th percentile of the absolute revision in probability below the
fixed old spot was **35.30 percentage points**; at one hour it was **1.20
percentage points**. Near-expiry distributions are narrow, so a small price
revision can materially change a threshold probability.

Wasserstein calculations use unrounded price draws. Fixed-price CDF
comparisons use the existing 0.01 tick rule. The recursive experiment rounds
every simulated price to that tick, as the planner does.

### Sampling resolution

The same 96 routine cases plus all 32 stress cases were repeated with 2,048
latent components. The maximum change in the measured Wasserstein revision
across these cases and all horizons was **0.000723 bps**. This is an empirical
refinement check, not a certified error bound for the underlying density.

Independent seeds were also used to compare two estimates of the *same old
forecast*. At one hour their median distance was **0.3741 bps**, larger than
the typical actual revision. Fresh independent sampling can therefore obscure
the changes we are trying to measure. Shared randomness cancels much of this
noise. These are Sobol diagnostics; their raw-draw variance ratios are not
claimed to be exact QMC integration-error ratios.

## Recursive one-second coupling

The rollout check uses sixteen declared routine origins and the eight largest
stress moves, **512 independent paths per origin**, repeated with two seeds.
The old forecast first simulates its unobserved next second. The new forecast
starts at the actual next observation. From that matching absolute timestamp
onward, old and new paths use the same independent Gaussian innovations in the
posterior, latent transition, and full observation decoder. They continue
through every second to the target. This is coupling by shared randomness,
not importance reweighting and not conditioning old particles on having
predicted the actual observation exactly.

For each scalar statistic `f`, the diagnostic is

    Var(f(new sample) - f(old sample)) / Var(f(new sample)).

The paired correction estimates the difference between the two expectations.
Both marginals remain the intended forecast laws. The denominator is the
per-draw variance for a fresh MC estimate of the new expectation.

Median ratios across the sixteen routine origins and two seeds:

| Initial lead | Terminal price | Upside beyond +10 bps | Above +10 bps event |
| --- | ---: | ---: | ---: |
| 2 seconds | 0.9981 | unresolved | unresolved |
| 5 seconds | 0.2410 | unresolved | unresolved |
| 10 seconds | 0.1082 | unresolved | unresolved |
| 1 minute | 0.01650 | 0.1586 | zero observed median |
| 5 minutes | 0.003215 | 0.009034 | 0.09558 |
| 15 minutes | 0.001052 | 0.002210 | 0.06550 |
| 1 hour | 0.0002645 | 0.0005480 | 0.01912 |

The upside statistic is `max(price/old_spot - 1 - 0.001, 0)` and the event is
`price/old_spot > 1.001`. They are sensitivity probes, not Oracle wealth or
actual limit-order fill indicators. A terminal crossing event does not record
intermediate path crossings. Unresolved or zero empirical variances at rare
thresholds do not establish zero tail probability or zero true error.

At one hour the routine median gains are approximately **3,780x**, **1,825x**,
and **52x** in per-draw variance for those three statistics. In stress cases,
the event's median ratio rises to **0.563**, and its maximum is **1.207**:
some corrections are noisier than a fresh event estimate. An adaptive method
must examine the actual statistic it is estimating.

The second seed also checks the whole generated two-row feature vector in
training-standardized units. The median paired feature RMS difference falls
from **0.941** at the two-second target to **0.00000530** at five seconds and
about **0.000000126** at ten seconds. Under shared innovations, this particular
model rapidly forgets most feature differences between the two roots. This is
a measured property of its generated feature process, not evidence that real
market information becomes irrelevant after five seconds. The price-level
difference persists.

Independent-half path comparisons at one hour have several-basis-point noise,
so the small rollout Wasserstein distances are not treated as precise estimates
of the true marginal distance. The paired statistic-variance measurements and
their second-seed agreement are the more useful rollout diagnostics.

## Implication for MC updates

For any valid coupling `X ~ old law`, `Y ~ new law` and integrable statistic,

    E_new[f] = E_old[f] + E[f(Y) - f(X)].

An implementation can keep an accurate old expectation and its random-number
seeds, then estimate only the correction using a smaller paired batch. If the
old value has error `e_old`, that error remains part of the new estimate;
reducing correction noise does not erase it. Fresh independent corrections
also accumulate uncertainty over repeated updates. Refresh decisions should
track that accumulated uncertainty and estimated correction variance.

The evidence supports testing this construction. It does **not** establish
the same variance reduction for a maximized Bellman function, a changed
account/order state, or a full trading trajectory with fills and liquidation.
The continuation representation must be evaluated on those changed states.
Nor does endpoint-distribution similarity by itself justify path importance
sampling or quantify a runtime speedup. This experiment performs no action
optimization and no historical trading backtest.

The previous MC experiment sampled edges of an unchanged finite graph after
appending a terminal second. This experiment instead measures forecast changes
after **actual new observations**, using uncompressed model draws. Its coupled
scalar corrections are a different object from the earlier coefficientwise
correction to supporting planes.

![Forecast revisions and coupling variance](E:/Projects/trading/data/backtests/oracle-nsf-forecast-revisions-comparison/revisions.png)

## Implementation and reproduction

Direct queries completed in **67.95 seconds**. Each complete paired rollout
study took about **100.5 seconds** for its 24 origins and 512 paths. These are
diagnostic runtimes, not trading-policy update latencies. The checkpoint,
trained forecast law, live bot and earlier backtests were not changed.

**18 related tests pass**, including exact agreement of the new GPU-oriented
one-second kernel with the existing planner sampler, fixed-target timing,
Wasserstein translation, ties, and empirical-CDF edge cases.

Implementation files:

- `ml/audit_nsf_forecast_revisions.py`
- `ml/audit_nsf_rollout_revisions.py`
- `ml/forecast_revision_metrics.py`
- `ml/report_nsf_forecast_revisions.py`
- `ml/test_forecast_revision_metrics.py`
- `ml/test_nsf_rollout_revisions.py`

Commands, run from the project root with new output directories:

```powershell
.venv-ml/Scripts/python.exe ml/audit_nsf_forecast_revisions.py --output data/backtests/revision-repeat
.venv-ml/Scripts/python.exe ml/audit_nsf_rollout_revisions.py --direct data/backtests/revision-repeat --output data/backtests/rollout-revision-repeat
```

Saved artifacts under `data/backtests`:

- `oracle-nsf-forecast-revisions-test`: direct queries, component refinement,
  same-law independent sampling control, selected timestamps, frozen sources.
- `oracle-nsf-rollout-revisions-test`: first rollout seed and endpoint paths.
- `oracle-nsf-rollout-revisions-repeat-test`: second seed and feature checks.
- `oracle-nsf-forecast-revisions-comparison`: combined JSON, tables and figure.

[Combined numeric results](E:/Projects/trading/data/backtests/oracle-nsf-forecast-revisions-comparison/summary.json)
