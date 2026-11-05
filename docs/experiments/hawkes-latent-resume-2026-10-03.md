# Resuming the B/S/latent Hawkes experiment

This resumes [Recall SDE Data Analysis](https://chatgpt.com/c/6abf6588-d5f0-83ed-83b9-8e5fc601b6ed).
The retrieved final turns reported residual-derived latent screening results and
an unsuccessful attempt at generative latent inference. Those results do not
establish a third Hawkes channel. The initial implementation here was reconstructed
before the user supplied the handoff archive; the chat retrieval tool did not
provide those files.

## Authoritative research context received afterward

The user subsequently provided `C:/Users/Roman/Downloads/hawkes_codex_handoff.zip`.
Its full `SPEC.md`, `CODEX_TASK.md`, README and reference scripts have now been read.
The unchanged package is preserved under
`data/research/hawkes-handoff/hawkes_codex_handoff/`, with archive provenance in
`data/research/hawkes-handoff/provenance.json`. Documents in the archive are
research context; the reference scripts are not the requested final implementation.

The handoff defines a broader milestone than the exploratory run reported here:

- Use **1-second** counts for the main architecture search.
- Fit independent source/target kernels with independent mode counts `K_ij`.
- Optimize continuous timescales, refitting all existing modes after every
  forward addition or backward deletion.
- Compare M0/M2/ML/M2L with fairly optimized architectures; prefer a linear
  stability margin around 0.95 rather than this screen's 0.995 limit.
- Jointly fit observed and latent parameters instead of only rescaling the
  observed coefficient blocks.
- Evaluate causal **open-loop** forecasts at 1/5/15/60/300/900/3600 seconds,
  plus likelihood, RMSE, correlation, R-squared and imbalance diagnostics.
- Report latent identifiability and initialization sensitivity, serialized
  parameters, fitted scales, stability and model-selection histories.

This five-second finite-bank comparison **does not satisfy that milestone**.
Its scores must remain preliminary, rather than being presented as the decisive
comparison requested in the handoff. The one-second cache and the coverage audit
are reusable inputs. Historical test results have already been inspected in both
the original chat and this exploratory run; subsequent work must freeze decisions
using train/validation and disclose that history, rather than claim a previously
unseen test set.

The supplied cross-scale screen uses a 14-scale bank from 5s to 43200s, NNLS mean
fitting, and products of standardized state traces. This reconstructed screen
uses five scales, Poisson likelihood fitting, and recursively filtered
event-amplitude products. Along with coverage masking, these differences explain
why reproducing the earlier reported numbers is a separate sanity check.

The next implementation milestone starts with an independently parameterized,
stable one-second M0 and forward/backward model-order refits on train/validation.
The nonlinear and generative latent comparisons then extend that selected
implementation, followed by the required open-loop horizon evaluation.

## Source and coverage

The matching dataset is
`data/exports/BTCUSDT-aggTrades-live-2026-08-17_2026-08-30.jsonl`.
Its adjacent metadata manifest records the following:

- Binance spot BTCUSDT aggregate trades; an event can contain multiple fills.
- August 17, 2026 09:34:49.910 UTC through August 30, 2026 07:10:09.636 UTC.
- 14,464,973 records, spanning 1,114,521 one-second bins.
- JSONL SHA-256: `7c18da2e810924ebffedaf5156fb1d54f6a31cdee80895d37a0059a6e808d4fd`.
- 81 aggregate-ID discontinuities and 884,285 missing aggregate IDs.
- The gzip export is 94,887,212 bytes, matching the previous chat's roughly 95 MB archive.

The parser checks both the row count and the complete JSONL hash before caching.
The source is a WebSocket recording, not a complete exchange history. There are
41 inter-record gaps longer than 60 seconds. A conservative heuristic excludes
both boundary seconds and the interval between them: 132,377 seconds in total.
The two largest gaps are 92,040 and 34,025 seconds. This mask cannot identify all
shorter interruptions or reconstruct missing events; it is not a certified
observation-coverage contract.

## Implemented comparison

`ml/hawkes_count_experiment.py` implements four count models: linear B/S,
M2 event-response modulation, linear B/S/L, and M2 plus L. These are five-second
discrete Poisson count processes. They approximate the proposed exponential
Hawkes dynamics but do **not** implement exact continuous-time event likelihoods
or within-bin cascades.

For observed events, use exponential moving states with times 10, 40, 160, 640,
and 2560 seconds. Every feature uses only previous bins. M2 filters products of
the current source count and a slower state known before that bin; the product
first affects the following bin. This adds 40 nonnegative modulation features
to the baseline's 11 features. This reconstruction need not be the same feature
bank as the unavailable earlier code.

The latent model evolves its own integer Poisson counts:

```
E[L_t | past] = muL + aLB*xB_t + aLS*xS_t + aLL*z_t
z_(t+1) = d*z_t + (1-d)*L_t
E[B_t | past] = scaled observed B baseline/kernel + gB*z_t
E[S_t | past] = scaled observed S baseline/kernel + gS*z_t
```

Here `d=exp(-5/tauL)`. B/S and L counts in the same bin are conditionally
independent given the previous state. A bootstrap particle filter integrates
over L histories. The observed predictive distribution is recorded before
conditioning on the target count, and latent events are sampled from their
generative transition, not from B/S residuals. All five directed B/S/L couplings
are fitted, along with the latent background. The observed baseline and kernel
blocks are only rescaled; their individual coefficients are not jointly refitted
with the latent parameters.

Particle inference for missing events is grounded in methods such as
[Shelton, Qin and Shetty (2018)](https://ojs.aaai.org/index.php/AAAI/article/view/12116)
and [Mei, Qin and Eisner (2019)](https://proceedings.mlr.press/v97/mei19a.html).
This script is a simpler binned bootstrap filter, not a reproduction of either paper.

## Evaluation and limitations

Use the same elapsed-time chronological 60/20/20 split, with no shuffle. Clear
model memory at masked gaps and exclude a 7680-second warmup after each reset
from fitting and scoring. Training, validation and test then contain 105,401,
34,921 and 34,689 five-second bins respectively. This changes the scoring set
from the previous chat; the old losses and MAEs are not directly comparable.

Observed coefficient fitting uses nonnegative Poisson likelihood optimization.
Latent fitting uses three multiplicative coordinate sweeps with 32 particles;
tauL is selected from 10/40/160 seconds using validation likelihood. Evaluate
the selected model with 128 particles and three independently seeded filters.
This optimizer is a restricted local search; parameter or model-order
convergence has not been established. Reported likelihoods omit the count
factorial term, which is identical across models on the same scoring set.

The first unconstrained latent fit had a linear branching spectral radius of
1.024 and was rejected. The revised fitter requires the complete linear B/S/L
branching core to have spectral radius below 0.995. This establishes stability
for the purely linear count process. It does not establish global stability for
the positive nonlinear M2 feedback, which remains a structural screening model.

## Reproduction and saved state

From the repository root:

```powershell
.venv-ml/Scripts/python.exe ml/hawkes_count_experiment.py --latent --particles 32 --sweeps 3
.venv-ml/Scripts/python.exe ml/hawkes_count_experiment.py --audit-particles 512
.venv-ml/Scripts/python.exe -m unittest discover -s ml -p test_hawkes_count_experiment.py
```

Artifacts live under `data/benchmarks/hawkes-resume-2026-10-03/`: the checked
one-second cache, coverage audit, fitted observed weights/predictions, per-tau
coordinate checkpoints, logs and `report.json`. Rerunning with the same settings
reuses completed fits and resumes particle optimization at the next coordinate.
Changed particle/sweep settings require a new output directory once checkpoints
exist. These data artifacts are intentionally ignored by Git.

Six focused tests check causality, gap resets, the Poisson gradient, exact
recovery of the observed likelihood when L is disconnected, and instability
through a cross-channel loop despite subcritical diagonal coefficients. A
synthetic hidden-event example checks that the filter improves likelihood over
a stationary Poisson control when latent events actually generate the data.

## Completed constrained fit

Latent rows below average three 512-particle audit evaluations. Lower is better for
both metrics. These scores use the new coverage mask and feature construction.

| Model | Test Poisson loss, without factorial | Test total-count MAE |
| --- | ---: | ---: |
| Linear B/S | -135.5140 | 35.5873 |
| M2 | -135.5577 | **35.4976** |
| Linear B/S/L | **-136.7274** | 36.0917 |
| M2 + L | -136.6390 | 36.2996 |

Both latent models select a 10-second latent mode. The fitted linear B/S/L
branching radius is 0.98667; M2+L's linear-core radius is 0.97894. Observed
coefficient optimization reports convergence for both linear and M2 fits.

M2 reduces held-out total-count MAE by about 0.25%. Adding L improves mixture
likelihood substantially but worsens count MAE: roughly 1.42% versus the linear
baseline, and 2.26% versus M2. Adding M2 to the fitted latent model improves
neither test metric in this restricted search. This supports further testing of
latent distributional structure, not a conclusion that a hidden channel is
identified or that the proposed continuous-time model has been validated.

Particle degeneracy is present: effective sample size can reach one even with
128 particles. At 512 particles, ESS is below 10% of the particle count on about
14% of scored test bins. Increasing particles from 128 to 512 moves linear+L
test loss from -136.5874 to -136.7274, and M2+L from -136.5137 to -136.6390.
The ranking survives, but the numerical likelihood has not converged. The
audit does not refit or select another model on the test set.

Validation: all six focused tests passed, both observed fits reported optimizer
convergence, both latent models completed all three scales/sweeps, and a repeat
run reused the completed fits successfully.

## Next unresolved scientific comparison

A decisive test still requires continuous-time latent-event inference, jointly
optimized observed kernels, numerical particle convergence, synthetic recovery
controls, and evaluation with a verified observation mask. Count likelihood or
MAE improvements alone do not establish profitable trading behavior.
