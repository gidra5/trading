# Later structured next-second sign models

Update: the identified family has now been refit from fresh weights on
November 2024 data. The bounded historical experiment reaches 63.02%
validation / 65.04% test active-next-second sign accuracy, but its calibrated
one-second mean returns remain below trading costs. See the historical refit
section of `native-second-event-policy-2026-09-04.md` and artifact v401.
The source audit below describes the earlier saved 2026 experiments.

The September 4 source audit corrects the earlier identification of the user's
remembered 75–77% next-second direction result as the 55.22% ridge model.
Saved results in the later production59/base17 structured family match that
range. The available evidence identifies the family; it does not establish
which individual checkpoint the user remembers.

These are existing results, with no retraining or new inference. The compact
audit records the complete run IDs, result and plan paths, source SHA-256
hashes, metrics, and saved magnitude deciles:
`data/benchmarks/event-policy-remembered-sign-source-audit-v338/summary.json`.

| Model variant | Train direction | Validation direction | Test direction | Selected epoch, zero-based |
| --- | ---: | ---: | ---: | ---: |
| Base17 causal self-attention | 78.7906% | **76.8059%** | **81.5012%** | 33 |
| Balanced return/derived/primitive loss | 76.4828% | **75.7012%** | **80.3507%** | 24 |
| Derivative-of-Gaussian basis, Gram auxiliary loss | **77.2383%** | **75.9957%** | **81.1609%** | 98 |
| Separate direction-decile reproduction | 76.9543% | 74.3400% | 79.4412% | 31 |

All four have 4,130,613 parameters. They consume two chronological 59-feature
states, predict 17 primitive coordinates for each of two subsequent seconds,
and derive the complete future feature states from those primitives. The
headline above scores only the first future second's signed log return.
Checkpoint selection uses validation feature-state MSE, not sign accuracy or
test performance. The first two variants use causal attention in layer 8;
the Gram variant replaces that component with its specified basis operator.

The dataset excludes exact-zero targets from every split, retaining zero
returns in observed histories. Thus these are direction scores conditional
on an active target, not three-class down/flat/up accuracy over all seconds.
Each model scores 256,000 training and 65,534 validation/test examples after
two-step constructibility checks. The dataset split boundaries are July 18,
August 3, August 10, and August 17, 2026; fixed example counts need not span
each complete date interval. The trainer's `FeatureMetrics.add` compares
`prediction >= 0` with `target >= 0` on the return channel.

The separate magnitude-decile reproduction is a different training run and
selected checkpoint. It must not be substituted for the original model's
unmeasured conditional scores:

| Reproduction diagnostic | Validation | Test |
| --- | ---: | ---: |
| Overall direction accuracy | 74.3400% | 79.4412% |
| Direction accuracy weighted by absolute realized return | 57.8739% | 57.4023% |
| Direction accuracy on largest 10% of realized moves | 57.2475% | 56.8966% |
| Mean absolute return in that largest decile | 1.6893 bp | 1.2116 bp |
| Next-second MSE skill versus zero | 2.2895% | 2.3712% |

The first six validation deciles and first seven test deciles have average
absolute returns near 0.0015–0.0016 bp. Their high sign accuracy explains much
of the difference between raw and magnitude-weighted accuracy. Magnitude
deciles are retrospective diagnostics, not an available trading gate.

This supports retaining the family as a candidate for a dedicated calibrated
sign forecast. It does not establish sign probabilities, run/barrier-horizon
accuracy, or profitable actions after costs. The prior event-sign experiments
did not evaluate these structured checkpoints, so their rejection does not
reject this family. When forecast work resumes, measure probability quality
and direction at the policy's decision horizon before combining it with the
conditional size/duration law. For historical inspector windows, models and
calibration must be fitted only on permitted preceding data; the saved
August 2026 checkpoints cannot be inserted into earlier-window backtests.

The ongoing policy-first objective remains unchanged. This audit corrects
source identification and preserves a concrete forecast candidate without
changing the frozen law used for Bellman optimality checks.
