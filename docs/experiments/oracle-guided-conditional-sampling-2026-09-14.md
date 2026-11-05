# Guided conditional sampling for the dense value map

The existing planner samples eight next-price outcomes at each evaluated market node and time layer. Candidate future actions share those outcomes. Its continuation table already includes sampling and decisions at later layers; it is not an eight-complete-path approximation to the whole hour. The experiments here reduce the current conditional evaluation to four fresh draws where safe, while retaining the precomputed deeper table.

## Implementation and invariants

`oracle_selected_outcomes.py` evaluates requested outcome indices on the GPU for arbitrary attainable post-decision states. Each selected outcome applies Oracle arrival/funding/fills and maximizes the existing next-decision menu against the supplied continuation field. It uses the same execution/interpolation primitives and candidate lane order as the dense GPU sweep. Unselected outcomes skip that expensive maximization.

Before partial sampling, a cheap check evaluates all eight arrivals and verifies an attainable immediate full close on each. Missing feasibility witnesses or nonfinite cached controls force an eight-outcome census. This protects feasibility under the finite eight-edge law; it does not certify survival over the ideal unbounded NSF support.

`oracle_guided_sampling.py` forms a proposal proportional to absolute deviations of cached values from their row mean, with a 20% uniform exploration mixture. It favors both tails and retains positive probability for every edge. Four indices are sampled with replacement using common random numbers across action rows. Importance weights correct the changed probabilities. The exact mean of the cached finite control vector plus sampled current-minus-cached corrections estimates the current row mean. Cash-reference centering removes shared continuation levels; that reference always receives a census. Only evaluated edges update the cached control vector.

The local expectation identity assumes a fixed continuation evaluator; maximization, interpolation, future-table approximation and model uncertainty are separate error sources. Per-update correction variances are not global action-confidence certificates. Six tests cover exhaustive weighted-index expectations, coverage of both tails, nonfinite/support fallbacks, cache isolation, CPU/GPU execution agreement through terminal/nonterminal decisions, and coverage of the original front's actions.

`oracle_root_candidates.py` generates attainable catalog, interior, retained-order and valid-close proposals before value evaluation. Stable descriptor IDs support cache reuse as actual account states change. A descriptor is not assumed to execute an identical trade at every observation.

## Fixed-target experiment

`oracle-dense-guided-root/results.json` compares the initial fine map's cached controls with the root of the saved two-update rolling profile. There are 408 candidate rows, eight conditional edges, and 100 seeds. Only 237 rows initially qualify for partial evaluation. It took 18.15 seconds including repeated refinements.

The complete root census takes a median 13.82 ms. Four-sample updates take approximately 9.2 ms. Two-tail action agreement with the eight-edge reference is 76% after one update, 99% after two, and 100% after four/eight. Uniform sampling achieves 78%, 66%, 85%, and 98%, respectively. Generations refine one fixed target; they are not later real observations. These results alone do not establish stability as forecasts and account state change.

## Advancing-observation experiment

`audit_oracle_guided_sequence.py` evaluates 24 consecutive observations of the completed fine replay, including its first limit fills. Every step receives a fresh causal NSF next-price forecast. Proposals are generated from the actual arrived account. Each sampler retains its own control history, updated only by its selected outcomes and required censuses. A separate eight-edge reference grades action choices without populating the sampler cache. The distant field remains the initial map's continuation to the same fixed terminal date.

There are 16 seeds per method, with the first observation serving as the initial census. The following 23 observations give 368 graded decisions per method:

| Metric | Uniform four-draw updates | Two-tail four-draw updates |
| --- | ---: | ---: |
| Median sampling-update time | 19.54 ms | 19.35 ms |
| Mean evaluated outcomes per row, including fallbacks | 5.891 | 5.891 |
| Exact action-state agreement | 47.55% | 50.54% |
| Mean action-value loss against eight-edge reference | 0.26184 bps | 0.22274 bps |
| 95th percentile loss | 1.18017 bps | 1.03642 bps |
| Maximum loss | 2.59711 bps | 1.80416 bps |
| RMS log-value error | 0.37941 bps | 0.37082 bps |

Action-value loss is `10000 * (1 - exp(Q_reference(selected) - max Q_reference))`, a relative certainty-equivalent loss for the log objective. It is not a realized trading loss. Forecast/proposal/evaluator preparation costs about 70–90 ms per warm observation. Full root censuses are generally 23–27 ms; the fill tick has very few attainable proposals and is much cheaper. The entire study takes 18.26 seconds. Artifact: `data/backtests/oracle-dense-guided-sequence/results.json` with individual records in `trials.json`.

The guided method has modest gains in this short audit. These root experiments do not replace backward table updates. The subsequent [resident incremental table experiment](oracle-incremental-table-2026-09-14.md) applies the sampler inside the continuation field. Its timing and accuracy are evaluated separately: these root timings must not be presented as an end-to-end one-hour-table speed or an optimal-policy certificate.
