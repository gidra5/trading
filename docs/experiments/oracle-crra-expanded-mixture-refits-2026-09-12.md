# Expanded distribution mixture and historical refits

The user requested a small set of older models with useful, distinct strengths,
fresh snapshots, and a test of whether they improve the current distribution
mixture. This experiment uses next-second BTC log returns. It is a forecast
comparison, not a trading-profit or policy-optimality result.

**Completed:** adding fresh historical models increased test all-tick log MSE
skill from **4.197% to 4.614%**. Most of the gain comes from direct59, which
reaches **4.530% alone** and is also the validation-selected likelihood winner.

## Historical selection

Two native families were selected from their recorded **validation** results:

| Family | Historical strength | Fresh training |
|---|---|---|
| Direct production59, five layer-4 branches, layer-8 attention | About 6.4% next-second nonzero validation MSE skill | Original two-step, 59-feature standardized MSE; 256,000 native training examples |
| Base17 primitive reconstruction, layer-8 attention | 76.8% nonzero validation direction accuracy | Original primitive-to-production59 reconstruction loss; 256,000 native training examples |

These are the original feature-prediction objectives, not another retraining of
the later sign/size specialist readouts. Both use fresh seed 20260912, up to 64
epochs, the original Muon/AdamW setup and EMA with a four-epoch half-life.
Native checkpoint selection remains minimum validation **feature-state** MSE.
The native training/evaluation population excludes zero targets; those headline
numbers must not be compared directly with the all-tick mixture scores.

The removed union530 source container is replaced by the restored production59
dataset's equivalent five-primitive-history manifest. Its 150 immutable source
references were verified. No market data was synthesized or downloaded.

## Probability heads and common evaluation

Each native model is frozen. A width-128 GLU maps its two predicted standardized
feature states to a full triangular-knot density in Gaussian-CDF log-return
coordinates, plus a separate exact-zero atom. These are new probability heads;
the historical native models did not themselves provide full price laws.
The Gaussian coordinate gives finite exponential price moments. The old
asinh-logistic log-tail law is excluded because its expected price can diverge.

Head training uses all outcomes, including zeros, in the later training tail
strictly after the native model's last two-step training target. It uses cadence
four seconds and at most 131,072 examples. The loss is mixed-law NLL plus expected
**log-return** MSE divided by the training log-return second moment. Selection
uses only the first chronological validation half, with a maximum of 64 epochs
and patience 12. The original native weights remain frozen throughout.

The first screen uses uncalibrated new heads; the archived online-calibrated
forecasts remain controls. No existing calibration or bot configuration is
changed. New-head means use 1536-point quadrature per half-knot and are checked
against 768 points to absolute tolerance 1e-12 in log return. Likelihood includes
the zero atom and the density Jacobian. It is not exchange price-cell likelihood.

All distribution comparisons use identical origins: 34,560 validation and
36,134 test examples (15,505 nonzero test outcomes). Prices, return coordinates,
timestamps and forecast ordering are checked. Simple-return Gamma/Beta laws
are integrated and transformed to log coordinates, including their nonzero
likelihood Jacobian; their means are not merely passed through `log1p`.

## Mixture protocol

For complete laws, use `P = sum_i w_i P_i`, with nonnegative weights summing to
one. This preserves every component's continuous distribution and atom. The
law's expected log return is exactly the weighted sum of component expectations.

Fit simplex weights on the first chronological validation half. Select the
candidate pool on the second half; refit its weights on full validation and
freeze them before test scoring. Fit and select MSE and NLL pools separately.
Include the original pair, singleton models, all archived models, each new
family with the original pair, both new families with the original pair, and
all candidates. Equal averaging and the existing two-regime rule are controls.

The historical test has already been inspected in earlier work. Base-model
checkpoint selection also used validation. These are exploratory comparisons,
not a fresh untouched holdout or a strictly nested estimate of the entire model
search. No test outcome selects the new weights or candidate pool.

For base prediction `m`, candidate prediction `g`, target `Y`, and weight `a`,
the change in mean squared error is exactly:

`2 a E[(m-Y)(g-m)] + a² E[(g-m)²]`.

A small addition helps iff the first expectation is negative. Lower correlation
or good standalone accuracy alone does not establish this. Empirical validation
improvement does not guarantee improvement on a future population.

## Completed archived-model screen

Thirteen compatible saved forecasts were screened: the two current components,
the likelihood-trained knot alternative, four per-horizon log specialist
variants, two shared log variants, two per-horizon Gamma/Beta variants and two
shared Gamma/Beta variants.

The likelihood-trained knot mean correlates 0.995 with the main knot mean on
validation. The shared simple-return models are more different (about 0.66
correlation) but weaker. Several raw/calibrated simple/log variants are nearly
duplicates and should not be counted as independent sources of evidence.

| Archived control | Test all-tick log MSE skill |
|---|---:|
| Main joint knot model | 3.699% |
| Current specialists | 3.446% |
| Original pair, equal weights | 4.197% |
| Archived screen's validation-selected MSE pool | 4.162% |

The selected archived pool assigns 43.23% to the main knot model, 48.34% to the
current specialists and 8.43% to the raw fine-tuned per-horizon log specialists.
Likelihood selection retains the main knot model alone. This screen does not
establish a gain over equal averaging. The historical refits are evaluated
separately below.

## Completed fresh refits

Both native runs completed 64 epochs and retained their best feature-validation
checkpoints. The probability heads each used 125,293 later training examples;
their training begins at origin 497750, after the native last target 497749.

| Native model | Selected native epoch, zero-based | Native nonzero validation MSE skill | Native nonzero validation direction | Selected head epoch |
|---|---:|---:|---:|---:|
| Direct59, five branches | 33 | 6.213% | 57.121% | 2 |
| Base17 reconstruction | 47 | 4.581% | 77.386% | 5 |

The following table uses the common **all-tick** test origins and each model's
complete probability law. Nonzero skill scores the same deployed expectation
restricted to nonzero outcomes, without renormalizing away the zero probability.
Lower NLL is better; negative density NLL is valid.

| Forecast | All-tick log MSE skill | Nonzero deployed log MSE skill | Mixed-law log-return NLL |
|---|---:|---:|---:|
| Original joint knot | 3.699% | 5.401% | -5.903018 |
| Current calibrated specialists | 3.446% | 5.756% | -4.005978 |
| Original pair, equal weights | 4.197% | 5.934% | -5.699083 |
| Fresh direct59 + probability head | 4.530% | 6.381% | **-6.029284** |
| Fresh base17 + probability head | 4.094% | 5.228% | -6.007046 |
| Validation-selected expanded MSE mixture | **4.614%** | 6.369% | -5.993936 |

Expanded weights after full-validation refitting:

- **75.6531%** fresh direct59 law;
- **18.2735%** current calibrated specialists;
- **6.0735%** fresh base17 law;
- zero numerical weight for the other archived candidates, including the old
  joint knot model.

The selection-half MSE skill is 4.4744%, versus 4.0732% for the earlier
two-regime rule and 4.4054% for direct59 alone. Several pools tie on this half:
their fitted weights reduce to the same direct59/specialist pair. The declared
candidate ordering chooses the all-candidate pool; full-validation refitting
then assigns base17 its small weight. **This tie does not establish that adding
base17 is necessary.** The final mixture has only three nonzero components,
but the strongest evidence is for direct59's contribution.

Against equal averaging, the gain is **0.4173 skill percentage points**, or
about **0.436% lower MSE**. The exploratory paired hour-block bootstrap uses 41
blocks and gives a 95% interval of **[0.1414, 0.6832] skill points**. Against
direct59 alone the gain is only 0.0844 points; a post-scoring diagnostic interval
is **[-0.0119, 0.1815]**, so a further mixture benefit over that model is uncertain.
These intervals do not account for the full historical model-search process.

For **NLL**, held-out validation selects direct59 alone. Its selection-half NLL
is -5.716650; the broader NLL-fitted pools score worse on that half. Do not select
the MSE mixture as though it also won the likelihood comparison.

The fresh laws also retain directional information: test active-sign accuracy
from their conditional sign probabilities is 85.811% for direct59 and 85.734%
for base17. These are probability-head metrics on the common origins, rather
than signs of the native feature models' point expectations.

On validation, direct59's expected-return correlation is 0.839 with the old
knot model and 0.835 with the current specialists. Coarsened distribution TV
has median 0.093 versus the knot law and 0.192 versus the specialist law; base17
medians are 0.095 and 0.187. These use the prior outcome-gate partition plus the
separate atom and both tails, and are lower bounds on continuous-law TV. They
must not be compared as exact distances with earlier finer-partition figures.

## Verification and scope

All six focused tests pass. Both saved native/head combinations reproduce
2,560 archived validation examples with **zero difference** in native outputs
and head logits using CUDA, high matmul precision and the archived batch size.
The inference adapter checks native-plan, native-checkpoint and head-checkpoint
identity, freezes both models, and returns the atom and complete knot masses.
CPU native inference is not bitwise equivalent; this experiment's final scoring
uses the common CUDA profile. A preliminary CPU-head score was retained as a
diagnostic, then rescored on CUDA without changing training or selection.

The bot has not been switched to these models. No trading backtest, live
execution claim, or new Lean certificate is implied by these forecast scores.
The snapshots and selected mixture are ready for a separate policy comparison.

## Artifacts

- `ml/expanded_distribution_mixture.py`: archived inventory scoring and pooling.
- `ml/refit_ensemble_distribution_heads.py`: frozen native forecasts and full-law heads.
- `ml/evaluate_refitted_ensemble.py`: expanded validation selection and scoring.
- `ml/refitted_price_forecast.py`: reload native/head snapshots and infer complete
  laws from two observed production59 states.
- `ml/training-plans/oracle-ensemble-refit-{direct59,base17}-20260912-v1.json`:
  fresh native plans; training uses the existing structured trainer.
- `data/backtests/oracle-expanded-mixture`: plans, selections, source hashes,
  forecast laws, matched predictions, scores and training logs.
  Final files: `refitted-selection.json`, `refitted-evaluation.json`, and each
  refit's `head.pt`, `evaluation.json`, `inference-adapter-audit.json`.
- `data/training/runs/oracle-ensemble-refit-{direct59,base17}-20260912-v1`:
  native checkpoint pointers and validation-selected immutable model objects.

Six focused tests cover simplex fitting, mixture normalization and boundary
weights, the simple/log Jacobian and zero atom, head predictions independent of
the future outcome being scored, and the forecast-only two-regime rule.
