# Joint one-second five-model comparison — in progress

The requested end state is a completed M0/M2/ML/M2L/M2LZ comparison, independently
optimized mixture orders and continuous scales, jointly fitted parameters,
causal forecasts through one hour, full fitted parameter reports, and evidence
for which mechanisms add value. The fifth model is M2LZ, confirmed by the user.

The user confirmed M2LZ as the fifth model. The five-second work in
`hawkes-latent-resume-2026-10-03.md` is preliminary and
does not meet this objective. No final scientific recommendation has been made.

## Implemented and running

- `ml/hawkes_research.py`: independent B/S kernels, analytic derivatives for
  every mass/background/continuous scale, constrained joint likelihood fitting,
  three scale initializations, forward addition and backward deletion with all
  parameters refitted, candidate caching and selection histories.
- `ml/hawkes_modulation.py`: the handoff's linear slow-state amplitude modulation, jointly optimized
  fast and anchor scales/amplitudes, separate edge and slow-anchor order search.
  Gates favor anchors at least four times slower than the event trace. The
  additive core is constrained below 0.95; this does not establish nonlinear
  stability. An optional bounded-tanh variant has a **dominating** branching
  matrix constraint, but it is separate from the main handoff comparison.
- `ml/hawkes_synthetic.py`: seeded two-channel/two-mode recovery control that
  runs the actual addition/deletion/refitting algorithm.
- Exact linear conditional-mean open-loop forecasting through augmented state
  transitions at 1/5/15/60/300/900/3600 seconds. Forecasts use future expected
  events, never future observations. M2 has a tested generative Monte Carlo
  rollout. Numerically divergent paths invalidate affected forecast means;
  surviving paths are never averaged as if they were the complete distribution.
  Latent/diffusion rollouts still need their own implementations and controls.

Source counts: the previously hash-checked local one-second cache, exactly
14,464,973 observed aggregate trades. Recording coverage is imperfect. Keep
the >60s gap exclusion, initialize traces after gaps using **training-only**
mean counts, and use a common fixed one-hour warmup for every candidate. Using
3*max(tau) as a model-specific warmup would change the scoring set during scale
optimization and unfairly compare models. This conditional initialization is
explicit; it is not an assertion of complete market observations.

## Selection protocol

Chronological elapsed-time 60/20/20 split. Selection objects contain only train
and validation arrays; test outcomes are unavailable to fitting/search routines.
The original chat and preliminary experiments already reported test outcomes;
that historical inspection must be disclosed in the final report.

Candidate selection minimizes validation NLL plus `n_parameters/n_validation`.
For every proposed extra mode, try initial scales 1/100/10000 seconds and jointly
refit all scales and amplitudes. Every deletion also jointly refits all retained
parameters. Maximum order is six modes per directed kernel. M2 also searches
up to three slow anchors per observed source. Stop on lack of forward/backward
validation improvement; an iteration limit is not successful completion.

Linear kernels use discrete normalized jumps `(1-exp(-1/tau))*count`. Their
integrated offspring mass is the fitted mass, not mass*tau. The safety margin is
0.95. All optimization is local and multistart: the result can establish best
validated parameters found, not a mathematically certified global optimum.

Fit objectives omit log-factorials, constant across candidates. Final one-step
NLL must restore them. Long-horizon Poisson mean scores must be labeled as such:
they are not exact marginal Hawkes count-distribution likelihoods. All model
comparisons must use common forecast origins and report MAE/RMSE/R²/correlation,
side-wise errors and imbalance correlation.

## Remaining required work

1. Complete M0/M2 architecture search and synthetic recovery/stability controls.
2. Implement joint generative ML/M2L fitting, including independent B/S/L kernel
   orders/scales and the observed parameters. Frozen/rescaled observed kernels
   do not meet the requirement. Validate hidden-channel recovery and particle
   likelihood convergence and initialization sensitivity.
3. Implement the fifth continuous stochastic excitation model, distinguish
   diffusion states from latent integer events, and optimize its state counts,
   continuous scales and diffusion parameters jointly with the other kernels.
4. Complete causal open-loop rollout comparisons through 3600s and numerical
   accuracy checks; select stopping/architecture using validation only.
5. Freeze all five fitted model selections, then perform the final comparison,
   uncertainty estimates, fitted scale/mass and latent-identifiability reports.

Particle parameter fitting can use an established partially observed process
method such as [iterated filtering](https://kingaa.github.io/manuals/pomp/html/mif2.html).
The final implementation and its numerical convergence must be demonstrated;
the reference alone is not evidence that these latent fits have been completed.

Outputs live under `data/benchmarks/hawkes-joint-1s-2026-10-03/`. Model searches
write candidate JSONs, `model_selection.csv`, `selection-history.json` and
serialized selected-model parameters. `selection_complete` remains false until
the forward/backward search actually terminates. Final test and parameter CSVs
are required after every model has been fitted and frozen.

## Commands

```powershell
.venv-ml/Scripts/python.exe ml/hawkes_research.py
.venv-ml/Scripts/python.exe ml/hawkes_modulation.py
.venv-ml/Scripts/python.exe ml/hawkes_synthetic.py
.venv-ml/Scripts/python.exe -m unittest discover -s ml -p test_hawkes_research.py
.venv-ml/Scripts/python.exe -m unittest discover -s ml -p test_hawkes_modulation.py
```

These commands currently cover the observed models and their controls only.
They are not a completed five-model experiment runner.

## Completed M0 selection and current evidence

The M0 forward/backward search stopped on no further penalized validation
improvement. It selected 13 modes, with independent directed mixtures:

| Source → target | Selected scales (seconds) |
| --- | --- |
| B → B | 0.399, 3.053, 27.150, 119.100, 980.656 |
| S → B | 36.869, 7556.882 |
| B → S | 33.520 |
| S → S | 0.486, 3.704, 28.028, 272.962, 1958.228 |

The backgrounds are approximately 0.31119 buy and 0.31006 sell events/second.
The integrated branching spectral radius is 0.95. Complete fitted masses and
scales are in `M0/selected-M0.json`, with 28 jointly optimized parameters and
validation objective -18.865375 before restoring the common factorial term.
None of the mode-count ceilings was reached. Sub-second fitted scales are
parameters of the discrete one-second approximation, not resolved event-time
kernel measurements.

The validation open-loop comparison uses 194 common origins at a 900-second
stride, each with a fully covered/scored one-hour future. M0's one-hour total
MAE is 12693.455 events, R² is 0.43059 and correlation is 0.67976. These are
validation results; the final five-model test comparison is pending. One-second
primary metrics also use all scored validation seconds, separately from this
coarser set of common forecast origins.

The seeded synthetic control recovered the correct orders `(2,0,0,2)`, pruning
the nonexistent cross edges. True buy scales `(0.8,80)` were estimated as
`(0.8084,75.1991)`, and true sell scales `(2,50)` as `(1.9490,49.0208)`.
The search therefore exercises actual order/continuous-scale recovery rather
than testing only a known architecture's trace implementation.

Seven joint-M0/forecast tests and six M2 gradient/filter/rollout tests passed.
The real-data M2 search and validation rollouts have now completed. The
remaining three models have not yet been finalized. This is progress toward
the full objective, not completion of the requested five-model study.

## Joint hidden-channel implementation and running controls

`ml/hawkes_latent.py` now implements an actual three-channel Poisson Hawkes
filter. At each second it integrates the preceding integer hidden count against
the current buy/sell likelihood and samples that count's posterior. It searches
the log-concave posterior support with bounded discarded tail mass rather than
truncating at a prior credible interval. Rare impulses that an ordinary bootstrap
filter would miss are therefore included in the observation likelihood. The
forecast mean is recorded before conditioning on the current observation.

Fitting uses all-parameter simultaneous perturbations with common random
numbers, Adam moments and validation checkpoints; see the original
[SPSA method](https://www.jhuapl.edu/spsa/pdf-spsa/spall_tac92.pdf).
Every background, directed mixture amplitude and continuous timescale is free.
Modulation coefficients/slow scales and OU scales/loadings enter the same raw
parameter vector when those mechanisms are enabled. This is not the staged
rescaling or residual-proxy screen. Finite iteration exhaustion is explicitly
reported as unverified convergence; implementing the optimizer is not evidence
that it has found an optimum.

Two real-data ML fits are running at 16 particles and 120 joint updates, starting
with frequent 10-second hidden events and rare 3600-second macro events. The
second starts at latent background 0.0001, outgoing integrated masses 1000 and
incoming masses 0.000001; these are starting values, not estimates. Hidden-edge
amplitudes can reach 1e6, so a small common weight ceiling does not exclude rare
long-scale impulses. The normalized three-channel core is projected to radius
at most 0.95. Bounds and latent degeneracy/scale sensitivity remain audit items.

`ml/hawkes_latent_audit.py` evaluates a fixed serialized checkpoint with several
particle counts and independent seeds, including marginal NLL, mean-prediction
errors, ESS, hidden-count activity and parameter-bound/scale diagnostics. The
first fast-start checkpoint's validation objective moved from approximately
-18.87921 at 32 particles to -18.87976 in the first 128-particle replicate. That
change is material relative to its small margin against M2. The audit is still
running; neither ESS nor this one replicate establishes convergence. The
audited model is frozen inside the audit protocol, even while the fitting job
continues to change its own checkpoint.

`ml/hawkes_latent_synthetic.py` is running two joint fits on a seeded known hidden
stream. Only observed counts enter fitting. It reports individual mass/scale
and background ratios, correlation with the true hidden count one bin earlier,
and recovery failures. This control uses a known seven-edge architecture;
mixture-order recovery remains a separate requirement.

`ml/hawkes_latent_search.py` adds or deletes modes separately on all nine
directed mixtures and refits the complete retained parameter vector. M2L also
searches B/S/L slow anchors; M2LZ searches independently scaled OU components
with signed loadings. It keeps architecture plateau, count ceilings and
unverified optimizer convergence separate. These latent architecture searches
have not yet been run to completion.

The implemented continuous extension uses independent stationary unit-variance
OU states with discrete decay `exp(-1/tau)` and innovation standard deviation
`sqrt(1-exp(-2/tau))`. Positive background excitation is
`mu_i * exp(sum_q loading[q,i]*z_q - 0.5*sum_q loading[q,i]^2)`.
Its unconditional mean is `mu_i`; signed loadings allow activity and imbalance
variation. This specifies one positive-intensity M2LZ member rather than
implicitly replacing integer L events with a Gaussian residual. Gaussian states
use a mixture of prior draws (25%) and guided Gaussian draws (75%), with the
exact Gaussian prior/proposal mixture correction in particle weights. Guidance
uses approximate diagonal Fisher curvature; the observation likelihood still
integrates integer latent counts. Including the prior component keeps an
alternative hidden-event explanation of a burst represented in the proposal.
The predictive background integrates the Gaussian transition
analytically before the observation is read. OU states with different decays
are retained separately. Real-data M2LZ fitting has not been launched; its
conditional model definition and particle accuracy still need empirical checks.

`ml/hawkes_latent_forecasts.py` snapshots the filtering law before each forecast
origin. ML conditional count means through one hour use exact augmented matrix
powers, including all latent feedback edges. M2L/M2LZ use causal generative paths;
the simulated total counts provide 5/50/95% quantiles, and integrated intensities
provide mean forecasts and Monte Carlo errors. These intervals condition on
fitted parameters and exclude parameter uncertainty. Any numerical runaway
invalidates that horizon rather than dropping failed paths from its average.
Hidden-model real-data horizon comparisons are still pending.

All 35 Hawkes tests passed after these additions: six count-screen controls,
seven observed linear controls, six modulation controls, eight hidden/continuous
filter controls, five hidden-forecast controls and three architecture-layout
controls. The hidden-count integral agrees with exhaustive enumeration; the
Poisson-lognormal likelihood agrees with independent Gaussian quadrature;
forecast snapshots are causal and hidden linear means agree with direct
recursion through 3600 seconds. The simulated linear forecast also agrees with
the exact mean within Monte Carlo error. These are implementation checks, not
the final five-model empirical comparison.

## M2 selection and current latent diagnostics

M2 stopped on no further penalized validation improvement, selecting 15 modes
with directed orders `(5,2,3,5)` and 80 joint parameters. The backgrounds are
0.14350 buy and 0.25636 sell events/second. Its three anchor scales are 2.82875s
from buys and 1.11882s/35.78662s from sells. The full modes, masses and modulation
matrix are stored in `M2/selected-M2.json`. None of the mode/anchor count
ceilings was reached. Its additive core radius is 0.95; the linear, unbounded
amplitude link still has no global nonlinear stability guarantee. No numerical
runaway occurred among the 512 simulated future paths at each of 194 origins.

These are validation comparisons, not the frozen five-model test result:

| Metric | M0 | M2 |
| --- | ---: | ---: |
| One-step NLL including factorials | 21.091124 | 21.077464 |
| One-step total MAE | 15.121101 | 15.098225 |
| One-step total R² | 0.130985 | 0.130760 |
| One-hour total MAE | 12693.455 | 12648.918 |
| One-hour total R² | 0.430593 | 0.393215 |

The one-hour MAE improvement is small (0.351%); R² and correlation worsen.
M2's one-hour Monte Carlo errors average 36.12 buy and 31.88 sell events per
origin. Particle/path refinement and paired uncertainty checks remain necessary
before turning the small MAE change into a recommendation.

A 5000-replicate paired validation block bootstrap, using contiguous origin
runs of at most six hours and stopping before coverage/time gaps, gave a 95%
interval of approximately `[-1217.9, 538.7]` events for M2-minus-M0 one-hour
MAE and `[-0.1101, 0.0177]` for the R² difference. Both include zero. This is
exploratory uncertainty on selection data, conditional on the stored forecast
means; it excludes parameter and Monte Carlo uncertainty. Details are in
`M0-M2-paired-validation.json`. The small MAE gain does not establish that M2
improves one-hour forecasting over M0 on this validation period.

The fixed ML checkpoint audit completed two independent replicates each at
32 and 128 particles. Mean validation objectives were -18.87921149 and
-18.87976577, respectively, with between-seed standard deviations 0.00009278
and 0.00000548. A 512-particle audit of exactly the same stored candidate is
running. This is an accuracy check on a provisional fit, not a fitted optimum.
The sparse/slow ML start completed its first 120-update fit but remains worse
than the fast start; its numerical and initialization sensitivity still require
refinement. The fast ML fit and first full-parameter M2L fit are running.

Both 300-update synthetic hidden fits completed. Hidden-count correlations
were 0.94496 and 0.94537, but both failed the predeclared factor-two parameter
recovery criterion. In the first fit, outgoing hidden-to-observed masses/scales
were within roughly 4% of truth; one weak sell-to-hidden scale was just below
half its true value. The second fit gave a sell-to-hidden mass about 4.42 times
truth, despite similarly good hidden activity recovery. At 256 particles, true
parameters had training objective 0.576937 versus 0.578425/0.579061 for the
fits; there is still an optimization gap. Two further full-parameter synthetic
refinements (128 particles, 600 updates) are running. The failures remain
recorded in the original control outputs; prediction/state recovery alone does
not establish parameter recovery or convergence.

Additional commands:

```powershell
.venv-ml/Scripts/python.exe ml/hawkes_latent.py --particles 16 --iterations 120 --tau 10 --output data/benchmarks/hawkes-joint-1s-2026-10-03/ML/start-fast
.venv-ml/Scripts/python.exe ml/hawkes_latent.py --particles 16 --iterations 120 --tau 3600 --mu-latent 0.0001 --out-gain 1000 --in-gain 0.000001 --output data/benchmarks/hawkes-joint-1s-2026-10-03/ML/start-sparse-slow
.venv-ml/Scripts/python.exe ml/hawkes_latent_synthetic.py
.venv-ml/Scripts/python.exe -m unittest discover -s ml -p 'test_hawkes*.py'
```

## Faster latent joint fits and forecast tradeoff

The preconditioned joint stochastic-window optimizer is now implemented in
`ml/hawkes_latent_minibatch.py`. Every background, directed mass and continuous
scale, modulation coefficient, anchor scale, OU scale and signed loading remains
free. Observed histories are reconstructed exactly at the current scales from
the entire prefix. Hidden histories use adaptive burn-in based on their decay
and feedback; difficult configurations fall back to the full particle prefix.
Matched perturbations now share the longer required context. Full training and
validation likelihood checkpoints guard against misleading short-window gains.
Finite iteration fits still do not establish convergence. All 43 Hawkes tests
passed, including continuous-history, context, sampling, and mixed OU/L checks.

The ML 1000-update fit completed, with training objective -32.321875 and
validation objective -27.121388 before factorials. A new fixed-parameter
32/128/512-particle audit and independent directed-mixture architecture search
are running on this stronger candidate. The older candidate's completed
512-particle audit gave mean validation objective -18.88018078; it does not
validate the newer candidate or the optimizer.

Provisional ML forecasts used its frozen iteration-899 checkpoint, 128 filtering
particles, 512 future paths and the same 194 validation origins as M0/M2.
One-step marginal NLL was 12.860555, substantially below M0's 21.091124, but
one-step total MAE was 15.559638 and R² 0.118099. One-hour total MAE was
22794.458 and R² 0.073114, substantially worse than M0's 12693.455 and
0.430593. The simulated 90% one-hour interval covered only 9.79% of origins.
There were no failed paths. ML uses an exact conditional first moment for
point forecasts; reported Monte Carlo standard errors describe the simulated
paths used for intervals. Likelihood improvement therefore does not establish
better long-horizon forecasting or calibration. These are selection-period
results from an unfinished architecture, not final test evidence.

The nested ML-seeded M2L 1000-update fit completed with validation objective
-26.050293. Its provisional horizon evaluation is running. A weaker M2-seeded
window refinement was stopped deliberately after 266 updates because the
nested ML start was substantially stronger; all saved checkpoints remain.
M2LZ's first short-OU 500-update fit completed with validation objective
-24.478916; its horizon evaluation is running. A separate regime-OU start
continues. Both classes still require current best nested baselines, independent
kernel/anchor/diffusion count search, and optimizer/numerical verification.

The required next steps remain numerical convergence and hidden recovery,
joint latent architecture fitting/refinement for ML/M2L/M2LZ, stable long-horizon
comparisons, all-five model freezing, then final held-out metrics and the
recommendation. No test metrics have been computed in this pipeline; fitting
and selection receive only training and validation observations.

## Progress update, 2026-10-03 13:37 UTC

This update supersedes earlier statements that the provisional forecasts and
fixed-candidate audits are still running. The experiment remains incomplete.
All 53 Hawkes tests pass. The first frozen five-model validation comparison is
saved under `provisional-comparison/`, including metrics, forecast bias,
provenance and `forecast_comparison.png`. Its M2LZ row is the earlier short-OU
fit; the stronger fast-OU result below is a separate candidate.

M0 remains the strongest provisional one-hour point forecast by R². M2's small
MAE advantage has no established paired-bootstrap improvement. ML's earlier
frozen candidate overpredicts hour totals by 27.88%. The stronger ML fit's
32/128/512-particle audit completed: mean validation objectives are
-27.12138906, -27.12139057 and -27.12139125. This verifies its likelihood
numerically, but does not establish optimal parameters or mixture counts.

The provisional M2L simulation encountered one runaway path among 194 origins
with 512 paths each. Aggregate 900-second and one-hour point metrics remain
unavailable; failed paths are not removed to create a survivor-only score.
The core spectral-radius cap does not bound the unbounded multiplicative
modulation globally. The mirrored ML start and regime-OU M2LZ start finished
with worse validation likelihood than the stronger retained candidates.

Fast buy-background OU excitation gave a substantially stronger M2LZ density
fit. The 1000-update candidate evaluated with 256 filtering particles and 512
future paths has one-step marginal NLL 5.310372, total MAE 14.241692 and
R² 0.104057. Its one-hour MAE is 22176.745 and R² -0.569683, with 90% interval
coverage 10.31% and no failed paths. These are validation metrics, not final
test results. A fixed earlier checkpoint was audited at 32/128/512 particles;
the density gain survives refinement, but the low-particle objective shift
exceeds tiny architecture-selection differences and requires final reranking.

For models with zero modulation, forecasts now integrate OU immigration and
the linear Hawkes first moment exactly, conditional on the empirical filtering
posterior. Closed-form and simulation checks pass. The fast-OU model's poor
hour forecast persists with this calculation. Future simulations still provide
intervals and nonlinear forecasts. A higher-particle fast-OU parameter
refinement and a separate fast-background plus slow-hidden-immigration OU fit
are running.

Directed mixture searches for ML, M2L and M2LZ are running in their
`architecture-search-controlled/` directories, with up to eight modes per
directed edge. New-mode proposals include amplitudes calibrated to the target
training rate, so rare hidden sources can produce a material initial effect.
Every architecture comparison also refits the unchanged incumbent for an
equal optimization budget. All retained parameters remain jointly free.
Earlier proposal searches were deliberately superseded with their best
checkpoints preserved. No latent architecture is yet certified complete.

The original and refined 30000-second synthetic hidden controls failed the
predeclared factor-two parameter-recovery criterion. In the new 300000-second
control, the first start completed with hidden-state correlation 0.939966 and
background ratios near one, but still fails recovery because sell-to-hidden
scale is 1.526 seconds versus truth 6 seconds. The second start is running.
A 256-particle fixed-other-parameter likelihood slice over that weak scale is
running to inspect the failure. It is not a profile likelihood, joint refit or
confidence region; neither weak identification nor an optimization gap is
assumed before examining its results.

The final-test runner is implemented and tested. It requires all five frozen
models, verified checksums and supporting selection audits before loading
test observations. No real BTC final-test evaluation or ready manifest has
been produced. Remaining work includes controlled count searches and deletion
checks, convergence and numerical reranking, model freezing, final held-out
forecasts, paired uncertainty and the parameter/report artifacts. Historical
inspection of this test period by the old chat and exploratory runs must still
be disclosed.

The current planning estimate is another 24–48 hours for the complete
comparison, with convergence uncertainty. Standalone running refinements are
expected to finish sooner; their completion does not finish architecture
selection.

## Weak-scale and numerical comparison update, 2026-10-03 14:00 UTC

The 256-particle sell-to-hidden likelihood slice completed both replicates.
Moving the scale from the fitted 1.526 seconds to the generating 6 seconds,
holding other parameters fixed, changes mean training NLL by -0.00004209 and
-0.00003232. Both conditional paired-block intervals exclude zero. Validation
changes are +0.00002706 and +0.00000516, with intervals including zero. This
is evidence of a small remaining training optimization gap at this particle
count, alongside weak validation discrimination. It does not establish a
parameter confidence region or overturn the failed recovery threshold.
The stored figure is
`synthetic-ML-long/sell-to-hidden-scale-slice/likelihood_scale_slice.png`.

An independent optimizer control now computes every raw-parameter gradient
coordinate using matched finite differences on common training windows and
random numbers, then updates the entire parameter vector simultaneously.
All probes share the longest conservative hidden context. Backgrounds, masses,
continuous scales, modulation and diffusion parameters remain free. A
100-update, 128-particle control with 256-particle full-data checkpoints is
running from the original stronger synthetic fitted point in
`synthetic-ML-long/coordinate-refine-start-0/`. No convergence flag is inferred
from a finite iteration count. Tests check a coupled objective, boundaries,
shared histories and the exact observable Poisson gradient with a disconnected
hidden coordinate.

The M2LZ search's first frozen numerical comparison cohort is running in
`M2LZ/ranking-controlled-first-cohort/`. It includes the two strongest completed
coarse candidates at cohort creation plus the original and equally refitted
incumbent controls, evaluated at 128/512 particles and two matched seeds.
Source parameters and the search CSV are frozen with fingerprints; later
search progress cannot silently alter this cohort. Paired validation intervals
use contiguous scored blocks and stop at coverage gaps. This early cohort is
not a final architecture decision or optimization certificate.

A frozen fast-plus-slow-hidden-OU candidate is also being forecast at 256
filtering particles and 512 future paths in
`M2LZ/fast-plus-slow-L-provisional-horizons/`. This check considers useful
horizons even though its coarse likelihood is worse than the fast-only fit.
The ongoing slow-OU fit cannot modify that forecast snapshot. The final-test
period remains unopened by this pipeline. All 59 Hawkes tests pass.

## Completed slow-diffusion forecasts and shared fits, 2026-10-03 14:40 UTC

The higher-particle fast-only parameter refinement finished 1000 updates,
selecting its final checkpoint at 64 particles: validation objective
-34.746284, 43 parameters and core spectral radius 0.718930. This is a joint
fit, not a convergence certificate or a completed count search.

The frozen fast-plus-slow-hidden-OU iteration-149 forecast completed at 256
filtering particles and 512 future paths. One-step marginal NLL is 5.304033,
total MAE 14.046590 and R² 0.097438. One-hour MAE is 17987.111 and R²
-0.104520, compared with 22176.745 and -0.569683 for the earlier fast-only
candidate. One-hour 90% interval coverage increases from 10.31% to 78.87%.
Both variants have no failed paths; point forecasts use exact additive first
moments conditional on the empirical particle posterior. The slow candidate's
512-particle likelihood audit is running, so numerical uncertainty remains.

A 5000-replicate paired bootstrap of contiguous forecast-origin blocks up to
six hours, stopping at coverage gaps, gives the slow-minus-fast one-hour MAE
change -4189.633 with interval [-6628.081, -1949.411]. Its R² change is
0.465163 with interval [0.327074, 0.695917]. However, slow-minus-M0 one-hour
MAE is +5293.656 with interval [1423.617, 9689.887], and its R² change is
-0.535113 with interval [-0.807512, -0.299106]. These intervals are conditional
on the stored validation fits and means; they exclude parameter, selection,
filtering and path Monte Carlo uncertainty. The shorter-horizon comparisons,
source fingerprints and figure are in `M2LZ/diffusion-horizon-comparison/`.
They support this provisional tradeoff, not a final model recommendation.

The first frozen M2LZ ranking cohort now has both 128-particle replicates.
The additional buy-to-buy component, initialized at 10000 seconds and jointly
refitted to 6360.273 seconds with integrated mass 0.022228, improves validation
NLL over the equal-budget unchanged-model control by 0.003491 and 0.003541.
Conditional one-hour-block intervals exclude zero in both replicates. The
512-particle evaluations are still needed before accepting this count change.

Both 300000-second synthetic starts completed. Hidden-count correlations are
0.939966 and 0.939611, but both fail the original all-parameters-within-twofold
criterion. Start 0 misses the weak sell-to-hidden scale; start 1 overestimates
its integrated mass by 2.775-fold. Successful state tracking does not establish
full parameter recovery. The independent all-coordinate joint refinement of
start 0 remains running; its first checkpoints have not improved on the seed.

ML and M2L were repeating identical no-anchor joint fits. Their searches now
share completed fits keyed by the full initial model, corpus and optimizer
configuration. OS file locks permit one producer and release on process exit;
there is no stale-process inference from a lock file. Fifty-one completed
candidate artifacts were published to the shared cache. Both owned searches
were deliberately stopped and resumed from saved checkpoints with unchanged
proposals, optimizer settings, scoring and scientific protocol. Their cached
results remain in their original directories, with reuse provenance recorded.
One search now computes an identical candidate while the other reuses it.
The M2LZ search and other fits were left running.

The frozen final-test runner now also produces paired horizon comparisons and
the common forecast plot after all five evaluations. No actual BTC test data
have been evaluated by this pipeline. All 65 Hawkes tests pass, including
paired comparisons, numerical failure retention and concurrent fit sharing.

Update, 2026-10-03 16:02 UTC: completion estimate revised

M0 and M2 validation searches are complete. The ML/M2L shared searches and
M2LZ controlled search remain in their first forward round; the selection
CSVs currently contain 44 and 19 evaluated rows for ML and M2LZ respectively.
The latest M2LZ joint candidate required 4897 seconds for 150 updates.
At current throughput, allow roughly another 48-96 hours for the remaining
searches, numerical review, frozen final evaluation and report. This is a
low-confidence planning range, not a deadline; 24-48 hours was optimistic.
No actual final test has been evaluated in this pipeline.

The slow hidden-immigration OU fit completed 300 updates. Its frozen earlier
149-update point now has a second forecast evaluation: 512 filter particles,
seed 91337, 512 future paths. Hour MAE is 18057.0214 and R2 -0.1202482,
compared with 17987.1115 and -0.1045200 at 256 particles/seed 71391.
Similar aggregate scores do not establish per-origin numerical precision:
the absolute change in total forecast means averages 1688.5326, with 95th
percentile 5017.0509 and maximum 16090.0260. Both runs have no failed paths.
The fixed-point 128/512-particle likelihood audit also completed, with proper
512-particle NLL 5.3027674 and 5.3030388 across its two seeds.

The dual-fast OU alternative completed 500 joint updates. Its frozen
validation forecast evaluation completed all 194 origins: marginal one-step
NLL 5.0138736, MAE 14.7959315, R2 0.0950086; at one hour MAE 17876.4818,
R2 -0.1264793, correlation 0.6724863 and 90% interval coverage 0.2371134.
No forecast paths failed. This point improves probabilistic density, while
its hour point accuracy remains worse than M0. Its higher-particle audit is
still running, as is the common slow background OU joint fit. These are
provisional fitted points, not completed architecture selection results.
Seven focused forecast tests passed after exposing the filter seed argument.

Update, 2026-10-03 16:04 UTC: matched diffusion comparisons

The expanded frozen comparison includes M0, M2, the earlier fast OU point,
the slow-hidden-OU point evaluated at 512 particles/seed 91337, and dual-fast
OU. Targets and all 194 origins are identical, and input file hashes are
recorded. The plot was visually inspected. Paired fixed contiguous blocks
(up to six hours, stopping at coverage gaps) give hour MAE increases versus
M0 of 5363.5664 [1552.8054, 9773.2659] for slow OU and 5183.0268
[1484.2762, 9458.5955] for dual-fast OU. Dual-fast minus slow-OU MAE is
-180.5396 [-2257.4604, 2604.3713], which does not establish a ranking.
These intervals exclude parameter, selection, filtering and future-path
Monte Carlo uncertainty; they are exploratory validation evidence.

The older M0-M2-paired-validation artifact uses moving contiguous runs with
a truncated final draw, whereas the expanded comparison uses fixed blocks
and variable origin counts. The resulting intervals differ: fixed-block hour
M2-minus-M0 MAE is -44.5366 [-980.6426, 729.0178], and R2 difference is
-0.0373784 [-0.0996239, -0.0029446]. The older moving-run R2 interval includes
zero. Do not treat this bootstrap-dependent result as a robust improvement
or conflate the two uncertainty methods. Neither shows an MAE improvement.

The stronger dual-fast fit now seeds a separate controlled full architecture
search with the same 150-update all-parameter joint-refit budgets and
independent directed-edge proposals. Its count/anchor/OU additions and all
backward deletions remain required: better parameter fitting in this basin
alone does not establish its optimal mixture order. The original one-OU
search continues, preserving its conditional structure comparisons.


Update, 2026-10-03 16:10 UTC: final-evaluation resume verification

The final-test runner previously reused any existing forecast NPZ accompanied
by a protocol file. It now verifies the saved model, corpus identity, test
split, stride, and applicable filter/path counts and seeds against the frozen
settings. Mismatches are rejected; missing forecast/metric/latent artifacts
are recomputed under the unchanged settings. The new regression test covers
foreign models, corpus, splits, particle budgets, seeds and strides, and
incomplete outputs. All 66 Hawkes tests pass (3.254 seconds). This does not
constitute actual final-test evaluation, which remains pending.

The dual-fast-OU audit produced its first 512-particle replicate: validation
marginal NLL 5.0107291330, ESS mean fraction 0.5788723, ESS below 10% on
3.1710% of scored seconds. The 128-particle NLL is around 5.0187; that drift
is larger than several mixture-selection margins. A separate frozen-point
2048-particle/two-seed audit is now running. No parameters are refitted by
these audits. The original 512-particle second replicate also remains live.

The synthetic all-coordinate refinement's current best checkpoint is update
19: train NLL 0.5055631226 and validation NLL 0.5280516460 at 256 particles.
This is a modest improvement over the original fit, contrary to the earlier
blanket statement that no checkpoint improved the seed. The 100-update run
is still live at update 72; parameter-recovery success has not been established.


Update, 2026-10-03 16:27 UTC: report generation and completed audits

The first frozen M2LZ shortlist audit is complete: four models, 128 and 512
particles, two seeds. At 512 particles the added-long-BB model's mean raw
validation NLL is -34.7713535400, versus -34.7677875703 for the equal-budget
unchanged-architecture control. Control-minus-added-mode differences are
0.0035503084 [0.0009568082, 0.0059663756] and 0.0035816309
[0.0010081320, 0.0059941006] across the two seeds. The ordering survives
higher-particle evaluation. This is conditional evidence in the original
one-OU basin, not final M2LZ selection; the stronger dual-OU search remains
live and architecture stopping has not been established.

The dual-fast-OU 128/512-particle audit also completed. At 512 particles the
proper validation marginal NLL is 5.0107291330 / 5.0107582214, with raw NLL
seed SD 0.0000205686. Its 128-to-512 mean raw NLL shift is -0.0076111927,
so matching seeds alone does not establish particle convergence. The separate
2048-particle two-seed audit remains running.

ml/hawkes_report.py now builds the final human-readable report only from a
completed frozen five-model evaluation and a separately reviewed scientific
decision. It lists one-step marginal NLL/MAE/R2, every horizon through one
hour, paired hour intervals, failures, every directed mixture count, all
backgrounds/masses/scales/modulation coefficients/OU loadings, stability and
latent-state diagnostics. It records historical test inspection and avoids
claims of global optimality or physical latent identity. Input and generator
hashes accompany the report. The latent-state figure averages inferred
quantities within scored minutes and leaves empty minutes as gaps. Its layout
was visually checked on a clearly separate toy rendering; no actual test
observations were opened. Two report regression tests cover incomplete
outputs, hidden failures and parameter/limit disclosure. All 68 Hawkes tests
pass (4.022 seconds). The actual final report remains ungenerated because
selection, numerical and stopping reviews are still unfinished.

The frozen evaluation now also copies the verified selection manifest, each
model's selection audit, and its complete architecture CSV into the result
bundle. The report links these copies and fingerprints them, so its scientific
review remains available with the exported artifacts. All 68 tests still pass
after this addition (4.266 seconds).

Update, 2026-10-03 16:35 UTC: dual-fast BB addition controls

All three relative-amplitude B-to-B additions in the dual-fast search have
completed joint refits. Their coarse selection scores (32-particle full-data
checkpoints) are -34.9561132314, -34.9603879428 and -34.9632765724 for
initial scales 1, 100 and 10000 seconds, versus -34.9671430606 for the
unchanged architecture with the same 150-update refit budget. None beats
that control. This differs from the gain of an added long BB mode in the
original one-OU basin, and reinforces the need to optimize mixture counts
within each model rather than reuse one model's selected order. This is
provisional coarse-particle evidence, not a completed higher-precision
ranking or architecture stopping decision. All searches remain live.

Update, 2026-10-03 16:54 UTC: paired one-step likelihood evidence

All three forecast evaluators now save per-scored-second marginal predictive
NLL (including factorial constants), with aligned global second indices.
For latent models this is the particle-integrated likelihood, distinct from
Poisson scoring at the predictive mean. The frozen final runner verifies
trace completeness on resume and produces paired NLL intervals for all five
models. Bootstrap computation uses block sums and scored-second weights,
preserving gaps and unequal block lengths without allocating per-bin draws.
Intervals are pointwise, conditional on fitted parameters, exclude selection
and filtering uncertainty, and do not adjust for multiple comparisons.

The regression test matches block-sum intervals to explicit resampling on an
unequal-block example, rejects mismatched times and nonfinite losses, and
checks trace averages against reported scores. The full suite passed 69 tests
(30.654 seconds, including compilation); the subsequently added end-to-end
latent marginal-versus-mean-score check and all seven existing forecast tests
passed together (23.629 seconds). Thus 70 distinct current Hawkes tests pass.
Only toy test corpora were used for these checks; the actual final test has
not been opened by this pipeline.

In the dual-fast architecture search, the rate-calibrated long BB addition
165b72e9d9b2e0330932 now has coarse score -34.9797428551, versus the
unchanged equal-budget control -34.9671430606. Its extra BB mass is
0.0451903694 and its jointly optimized scale is 11563.1700 seconds; all
49 parameters were free. This initial amplitude differs from the earlier
three relative-amplitude BB proposals that failed to beat the control.
Higher-particle ranking and backward selection remain required. Its frozen
256-particle/512-path validation horizon evaluation is now running to test
whether the NLL improvement also helps useful forecast horizons.

Update, 2026-10-03 18:01 UTC: completed synthetic refinement and new forecasts

The independent synthetic all-coordinate refinement completed 100 updates
with all 17 parameters free. Its selected final checkpoint (99) has train raw
NLL 0.5055516764 and validation raw NLL 0.5280039758 at 256 particles.
A separate frozen 512-particle assessment gives hidden-count correlation
0.9398600318 and count MAE 0.0130643245. Background ratios are
0.9932200/0.9943894/1.0846954. The strict original factor-two parameter
criterion still fails: S-to-L mass is 0.0012347055 versus 0.002 and scale is
2.24603494s versus 6s (scale ratio 0.3743392). B-to-L mass/scale ratios are
1.2412905/0.6924517; L-to-L ratios are 0.9452623/1.0550936. Good filtered
state recovery does not establish parameter identification or optimizer
convergence. The assessment first failed on an incorrect archive field name;
that script error was corrected to the documented hidden field, and only
the frozen assessment was rerun. It did not change the completed fit.

Frozen M0 and M2 validation losses were independently reconstructed from
their selected parameters and checked to reproduce their recorded NLLs.
Across 195016 scored seconds, M2-minus-M0 NLL is -0.0136598641 with
pointwise conditional six-hour-block interval [-0.0206331774, -0.0075949895].
The small probabilistic gain is supported, but its hour MAE gain remains
unestablished. These traces do not evaluate the final test.

The dual-fast long-BB point completed all 194 forecast origins at 256 filter
particles and 512 paths: marginal NLL 4.9555178802, one-step MAE 14.6401300,
R2 0.0699055; hour MAE 18372.9810, R2 -0.2280582, correlation 0.5820203,
90% coverage 0.3505155 and no failed paths. Against M0, its paired NLL change
is -16.1356059 [-18.5409669, -13.7158332], but hour MAE change is
+5679.5260 [1305.4693, 10964.7637] and R2 change is -0.6586510
[-0.9855042, -0.2947204]. The new six-series horizon comparison plot was
visually checked. Comparisons are conditional on stored fits and means,
exclude numerical/parameter/selection uncertainty, and use validation data.
The dual-fast shortlist ranking (128/512 particles, two seeds) remains live.

The ML search found a much stronger short L-to-B component via a calibrated
initial amplitude, followed by joint refitting and a B-to-B deletion. Frozen
candidate 1db69c1f899e51ee4602 has 41 parameters, edge counts
(4,2,2,1,6,1,1,1,1), and core radius 0.95. Its two L-to-B modes have
mass/scale (0.09748807,4.2603668s) and (10.374235,0.2113328s).
Its validation forecast evaluation completed: NLL 10.8496646759, one-step
MAE 15.8104137, R2 0.0680050; hour MAE 19820.6284, R2 0.1129462,
correlation 0.6650252, 90% coverage 0.4381443, no failed paths. This point
improves the earlier ML fit but still trails M0 in hour point accuracy.
The ML and M2L searches continue; this is not a frozen final selection.

The older dual-fast point's first 2048-particle audit replicate has proper
NLL 5.0076804163, versus 5.0107291330 at 512 particles with the same seed.
The remaining shift is -0.0030487167; its second 2048-particle replicate is
still running. A 512-particle/512-path forecast evaluation now also runs for
the completed refined single-fast-OU fit, whose parameters differ from the
earlier single-OU forecast snapshot. Its simulation uses two CPU threads to
limit competition with the searches. No existing job was stopped for this.

The final report now displays horizon Poisson-mean log scores explicitly,
labelled as proxies rather than exact marginal multistep NLL. The two report
tests pass after this display change. Actual final-test scores and the final
model decision remain pending. The count cache contains the full period;
selection and these validation routines expose/use only its training and
validation prefix. Earlier shorthand about not opening the test refers to
not evaluating it in this pipeline, not to the physical contents of the cache.

Update, 2026-10-03 18:22 UTC: matched updated latent-model comparison

The reproducible `compare_updated_latent_validation.py` compares M0, M2,
ML candidate 1db69c1f899e51ee4602 and dual-fast M2LZ candidate
165b72e9d9b2e0330932. All four forecast archives have identical 194 origins,
targets and horizons; all marginal-loss traces use the same 195016 scored
validation seconds and reproduce their recorded NLLs. Results and source
hashes are in `updated-latent-four-model-validation`; the plot was visually
checked. This comparison does not refit models or evaluate the final test.

ML-minus-M0 marginal NLL is -10.2414591 with six-hour-block conditional
95% interval [-11.4769886, -8.9154135]. Its hour MAE change is +7127.1734
[3662.0386, 10685.0567], and hour R2 change is -0.3176467
[-1.1798707, -0.1356167]. Thus this improved ML point has a supported
one-step density improvement and worse hour point forecasts.

M2LZ-minus-ML marginal NLL is -5.8941468 [-7.0990194, -4.7834540]. Its
hour MAE change is -1447.6474 [-8108.9161, 6166.2398] and hour R2 change
is -0.3410043 [-0.8238038, 0.8797241], so an hour improvement over ML is
not established. These are pointwise conditional sampling intervals;
parameter, selection and filtering uncertainty are excluded.

Live-process checks confirm all eight current jobs remain running. ML has
advanced to forward round 1 (90 completed candidates): its unchanged
41-parameter joint-refit control 35f92a6021d777014aeb has raw validation NLL
-29.1586281, better than the forecasted point above. This is further evidence
that the forecasted point is provisional and parameter stopping remains
unresolved. M2L's first forward pass has 56 completed candidates and its
best remains the no-anchor short L-to-B addition. The original and dual-fast
M2LZ searches have 33 and 16 completed candidates respectively. No final
architecture or optimizer convergence is claimed.

Update, 2026-10-03 18:28 UTC: dual-fast particle audit completed

The original 500-update dual-fast point completed both 2048-particle seeds.
Proper validation NLLs are 5.0076804163 and 5.0077415955, mean
5.0077110059. Raw mean NLLs at 128/512/2048 particles are
-34.9381440487/-34.9457552414/-34.9487879127. Successive changes are
-0.0076111927 and -0.0030326713; raw seed standard deviations are
0.0004256431/0.0000205686/0.0000432602. Small seed variation does not
eliminate systematic particle-count drift. The improvement is small relative
to broad model-class margins, but close architecture comparisons and
forecast-state accuracy need review on the actual selected parameter point.
No blanket convergence claim or new audit of this superseded point follows.
`M2LZ/dual-fast-particle-precision-review.json` preserves the comparison and
source hashes. Seven other compute jobs remain live.

Parameter diagnostics also flag nearly coincident BB scales (42.63/42.21s),
SS scales (750.01/837.74s), and OU scales (1.16484/1.05441s). These are
diagnostic evidence for backward selection and identifiability review, not
proof that deleting a mode preserves fit. OU loading directions and overall
signs are not uniquely interpretable physical mechanisms.

The common slow-background candidate reached update 149 with raw train NLL
-40.1522750 and validation NLL -34.8431968 at 64 checkpoint particles.
Its 200-update fit remains live; cross-candidate numerical settings must be
matched before using this checkpoint to select a final architecture.

Update, 2026-10-03 18:33 UTC: refined single-OU forecasts completed

The completed 1000-update refined single-fast-OU point (43 parameters) was
evaluated using 512 filter particles and 512 future paths. Its proper one-step
validation NLL is 5.2089111010, MAE 14.5496703 and R2 0.1040964. Across
all 194 hour origins, MAE is 18991.9780, R2 -0.2868428, correlation
0.6766368 and 90% total-count interval coverage 0.2525773. No origin or
path failed. The mean simulation standard error is 64.3885; point forecasts
use exact additive conditional moments, so this is a simulation diagnostic,
not their uncertainty. It is a different parameter point from the older
single-OU forecast and was independently evaluated.

`compare_refined_ou_validation.py` adds this point to a matched comparison
with M0, M2, the updated ML point and the evaluated dual-fast long-BB point.
The complete results and source hashes are in `refined-OU-validation-comparison`;
the plot was visually checked. The single-OU point improves marginal NLL
over M0 by -15.8822127 [-18.2659890, -13.5018756], yet its hour MAE is
worse by +6298.5230 [1533.6830, 12009.9076].

Dual-fast long-BB minus refined single-OU NLL is -0.2533932
[-0.3003073, -0.2135071]. Hour MAE changes by -618.9971
[-1198.8475, 30.6396], so a MAE improvement is not established; hour R2
changes by +0.0587846 [0.0275954, 0.1211482]. All intervals are pointwise
conditional fixed-block sampling intervals and exclude parameter, selection
and filtering uncertainty. The two points use different filter particle counts
(256 and 512), so the final precise comparison must match numerical settings.
Their validation density margin is much larger than the particle-count shifts
seen in the earlier dual-fast audit; that audit is still not a precision proof
for this newer point.

The refined single-OU posterior mean hidden intensity is 0.2627312, preceding
hidden count mean 0.2577922 and active probability mean 0.1276400. Posterior
state correlations using the current observation do not establish physical
latent identification or predictive information beyond the likelihood and
forecast tests. Six compute jobs remain live; architecture selection, parameter
stopping review and final test are still incomplete.

Update, 2026-10-03 18:39 UTC: matched-particle OU likelihood comparison

The completed 512-particle seed-71391 likelihood trace for candidate
165b72e9d9b2e0330932 was reused from the frozen ranking cohort. The new
`compare_matched_ou_losses.py` checks the corpus identity, serialized model,
particle count and seed, finite scored times, and reproduction of the stored
raw ranking score before adding factorial constants. It reruns neither a fit
nor a filter and records source hashes. Its derived metrics contain only the
marginal likelihood; predictive means and point metrics were not recalculated.

At matched 512 particles and seed 71391, the dual-fast long-BB proper NLL
is 4.9532106155 versus refined single-OU 5.2089111010. The paired difference
is -0.2557004855 with conditional six-hour-block interval
[-0.3030524505, -0.2155960671]. Thus the density improvement persists
after matching these numerical settings. Results are in
`M2LZ/matched-512-OU-validation-loss-comparison`. This does not match the
filter settings of their stored horizon forecasts or prove final selection.

The NLL comparison now also accepts a likelihood-only cache without demanding
a Poisson-at-mean score. The four comparison tests pass (0.066s), including
an added assertion that such a cache gives exactly the same paired result.
There is no new scientific fit or count metric in this code change.

M2L completed its remaining first-pass anchors and has caught up through the
shared no-anchor ML fits into round 1. It has over 100 completed candidates;
its subsequent modulation proposals remain to be evaluated on the improved
ML parameter point. No modulation gain or final collapse is established yet.

Update, 2026-10-03 19:13 UTC: verified filtering performance improvement

The OU proposal reused a single gradient workspace rather than allocating a
copied/negated array on every guidance iteration. The proposal equations,
importance correction, model parameters and random draw order are unchanged.
`benchmark_ou_workspace.py` compares the earlier allocation strategy with the
production implementation on a 10000-bin training slice at 16 particles for
the fitted single-OU and dual-OU models. All predictions, likelihoods, posterior
diagnostics and six saved-state arrays are bitwise identical, including an
injected gap reset and four forecast origins.

The initial benchmark measured median speedups 1.2217/1.4001; a repeat after
applying the production change measured 1.2849/1.4656. The latter correspond
to roughly 22%/32% lower runtime on these slices. Measurements are local
benchmarks, not a guarantee of the full study's remaining runtime. Results and
source hashes are in `OU-workspace-benchmark.json`; the allocation oracle is
used only by the benchmark, and production has one implementation.

All 71 current Hawkes tests passed in 31.896s. The new filter test compares
two- and four-component Gaussian likelihoods to independent quadrature and
checks that forecasts still use the prior mean. Existing likelihood, causality,
gap/reset, warmup, recursion, joint-refit and frozen-evaluation tests also passed.
Only toy corpora were used for this suite; the benchmark uses training data.

The two M2LZ count searches were deliberately resumed to load the verified
workspace change, with their original model inputs, particle counts, 150-update
refit budgets, seeds and cached completed fits. Candidate
52f641667ad2e2bd9991 resumed after update 70; candidate
eb707a2826ff135179e7 resumed after update 145 and completed its 150 updates.
`OU-workspace-resume-checkpoints.json` and `OU-workspace-resume-evidence.json`
record the saved and advanced counters. This was a planned performance
resumption, not a restart triggered by an observation timeout. Optimizer
checkpoint format and mathematical method are unchanged.

The slow-background fit and frozen ranking cohort continue from their existing
processes. The slow fit has reached 195/200 updates. The dual-fast long-BB
point's second 512-particle seed has raw validation NLL -35.0032000226,
close to -35.0032883031 for the first seed. Remaining cohort points and all
architecture/stopping reviews still need to finish before the final test.

Update, 2026-10-03 19:19 UTC: common slow-background fit completed

The existing common slow-background M2LZ seed completed all 200 joint updates.
Its selected final checkpoint (199) has raw train NLL -40.1558071697 and
validation NLL -34.8442171269 at 64 checkpoint particles, 51 jointly free
parameters and additive core radius 0.6339589909. This finishes that fit's
scheduled budget; its optimizer-convergence flag remains false.

A frozen evaluation of this selected point now runs with 512 filter particles,
512 future paths and the same 194 fully covered validation hour origins. The
output directory is `M2LZ/common-slow-OU-provisional-horizons`. It uses the
verified workspace implementation and two rollout CPU threads. The source fit
is retained unchanged. This evaluation is needed to assess the persistent
immigration-state hypothesis; no final architecture choice follows from its
coarse fitting score alone. Actual final-test evaluation remains pending.

### 2026-10-03 20:14 UTC: readiness and ETA update

The M0 and M2 validation forecasts were reproduced with current model/corpus/split bindings before completing their individual selection audits (217 and 145 successful joint candidate fits respectively). Both components are ready for the eventual frozen comparison; the five-model selection is still incomplete and this pipeline has not evaluated the final test. The final-test verifier now requires the reviewed audit to bind to the exact model label, model hash and selection CSV hash. There are 72 distinct currently passing tests (the full 71-test suite followed by the updated six final-test checks and two report checks).

The common slow-background OU pilot completed at 512 filter particles and 512 future paths: proper one-step NLL 5.1045316; hour MAE 15868.7196, R2 0.3038617, 90% interval coverage 0.9175258, with no failed paths. Compared with the matched 512-particle dual-fast long-BB candidate, its one-step NLL is worse by 0.1513210 (conditional paired 95% interval 0.1223457 to 0.1801021). Its hour MAE point estimate improves by 2504.2614, but the interval (-9756.9333 to 3632.9218) includes zero. Relative to M0, its hour MAE is worse by 3175.2646 (59.5594 to 6422.3658). This remains an informative validation tradeoff, not a finalized architecture or a physical identification claim.

The frozen four-candidate dual-fast reranking completed. The long-BB candidate remains best at both 128 and 512 particles across two seeds; this does not establish convergence of the ongoing architecture search. Four search processes remain live. Planning ETA remains 48–96 additional hours, with low confidence because remaining search rounds depend on validation improvements. Final numerical reviews, five-model freeze, final evaluation and report remain pending.

### Frozen dual-fast initial-cohort review

`review_dual_cohort.py` reproduced all 16 cached likelihood means on the 195016 scored validation seconds and verified the frozen CSV, candidate model hashes and model serialization against the corpus. `M2LZ/dual-first-cohort-review.json` records these source fingerprints and the parameter diagnostics. No new fit, filter or test evaluation was run for this review.

The ranking is identical at 128 and 512 particles: long-BB candidate 165b72e9d9b2e0330932, calibrated 100-second BB candidate 9ea3d30b21f5d2a839be, equal-budget unchanged-architecture control 4f7f15405c65d5c747f1, original fit 45743312d8010af525ff. At 512 particles the long-BB raw NLL mean is -35.0032441629, with seed SD 0.0000624. The control minus long-BB differences are 0.0123761 and 0.0125061 for the two seeds, with conditional paired intervals [0.0078908,0.0165394] and [0.0081557,0.0165944]. These support this local addition against the unchanged-architecture control.

The long-BB raw NLL shifts by -0.0065457 from 128 to 512 particles; the runner-up shifts by -0.0065956. Relative ordering survives, but absolute likelihood convergence remains unestablished. This frozen initial cohort does not cover the unfinished architecture searches and is not readiness evidence for a final M2LZ selection.

### 2026-10-03 22:07 UTC: CUDA continuation

The four latent architecture searches now share one batched float64 CUDA optimizer on the RTX 3070. Independent SPSA perturbations and chronological windows are batched; time remains sequential within each particle filter. All backgrounds, directed masses and continuous scales, modulation coefficients/anchor scales and OU scales/loadings remain jointly free. Candidate neighborhoods, 150-update budgets, equal-budget controls, score weighting, conditioning contexts and greedy addition/deletion logic are preserved. Full initial/final likelihood checkpoints still use the CPU reference at 32 particles and seed 71391, preserving comparability with completed candidates. CUDA uses addressed Philox particle draws, so the resumed optimizer trajectory is explicitly different from the original CPU particle RNG; its numerical transition is recorded rather than treated as bitwise reuse.

`gpu-migration-snapshot/audit.json` preserves hashes of 389 completed fit files and snapshots the four unfinished optimizer states. Those fits resume at iterations 119 (ML), 33 (M2L), 119 (single-OU M2LZ) and 133 (dual-fast M2LZ). Their raw parameters, Adam moments, optimizer RNG state, budget and configuration are retained. Each candidate's `accelerator-resume.json` binds the transition to the exact source checkpoint and implementation hashes. Completed CPU fits remain in the original shared cache; new CUDA fits are deduplicated within the GPU coordinator and are not published as bitwise-identical CPU cache entries. `gpu-migration-verification.json` checks that completed fit bytes are unchanged and the four transition hashes/configurations match their snapshots.

The independent CUDA reference tests cover integer latent counts, rare impulses, large Poisson intensities, observed/hidden anchors, linear/bounded modulation, up to four OU components, gap resets, variable input segments, causal predictions and chunk/batch invariance. A compact weighted-loss output reproduces the full filter, including repeated sampled windows, while avoiding gigabytes of per-proposal time-series outputs. Optimizer tests check CPU agreement in an exact observed case, interrupted resume including all moments/RNG state, and CPU filter bitwise invariance under concurrent threads. Scheduler tests check shared-job deduplication, exception propagation and equality of serial/batched greedy decisions and CSVs. The complete Hawkes suite passes **83 tests** (`gpu-migration-tests.log`).

Training-only 10,000-bin, 16-particle benchmarks on frozen fitted models measured 6.02–6.85x throughput over serial CPU filtering at batch 32; dual-OU batches 64/128 measured 10.57x/18.39x. A single GPU filter was slower than one CPU filter, which is why candidate batching is necessary. These are filter benchmarks, not full-study speedup estimates. The live continuation's first three cycles advanced 60 fits in 75.265, 91.547 and 90.594 seconds; GPU utilization was 100%, using about 1.4 GiB including the desktop. By the third cycle the four resumed fits had advanced to iterations 122, 36, 122 and 136, with all 389 completed source fit hashes still matching. Extrapolating these cycles gives roughly three to four hours for this candidate batch, before subsequent data-dependent neighborhoods and final numerical/forecast reviews. The previous 48–96-hour CPU planning estimate is superseded by the GPU continuation; a precise completion ETA remains unestablished. The five-model selection and actual final test are still pending.

### 2026-10-03 22:23 UTC: completed 43-parameter ML forecast

Completed 43-parameter ML candidate `3701a71ba6e206ac80ff` was evaluated at 512 filter particles and 512 generative paths on the same 194 covered validation hour origins. `ML43-validation-review.json` verifies the model, corpus, origins, targets and zero failed paths. Source fit SHA256 is `039255481763adfae7026218cac4a9f49f6b6826c73bae88662d8922ce3a8aa5`; forecast SHA256 is `40472c0bd627017dd3b36015b2416cc85bab4f3aa90e526f837f38154f354912`.

Its proper one-step predictive NLL is 10.7840879, MAE 15.8653079 and R2 0.1096636. Compared with M0, NLL changes by -10.3070359 (conditional paired 95% interval [-11.5537474,-8.9665896]). The hour mean MAE is 26518.7646 and R2 -0.2277718; versus M0, MAE is worse by 13825.3096 ([8079.2689,18720.0112]) and R2 changes by -0.6583647 ([-3.1466317,-0.1996583]). Hour interval coverage is 0.1907216. The additive core stationary rates are [11.1989642,10.8083939,0.7475115] counts/s. Mean hour forecasts total 70909.6112 against actual 52872.0619, a +34.1155% mean bias. Additive point forecasts use exact conditional first moments, so this mean forecast deterioration is not caused by the 512-path Monte Carlo mean estimate.

`ML43-six-point-validation-comparison` stores horizon metrics, paired differences and a visually checked scientific plot for M0, M2, ML43 and the three completed M2LZ pilots. Proper-NLL paired comparisons include only the matched 512-particle latent evaluations: the dual-fast long-BB pilot's stored forecast uses 256 filter particles and is excluded from that matched NLL comparison while remaining explicitly provisional in the horizon plot. All comparisons use fixed source fits, retain all origins and exclude parameter/selection/filtering uncertainty. The GPU searches remain live; this completed ML point can still be superseded. No five-model freeze, final test or final keep/drop decision follows from this interim comparison.

### First completed resumed CUDA budget

Dual-fast candidate `2b80c241680ac778b930` finished its 150-update budget after resuming from iteration 133. The full CPU checkpoint at iteration 149 selected raw validation NLL -34.9705912030 with 49 free parameters and two OU components. This is worse than the existing long-BB candidate's comparable coarse score; it does not promote a new architecture winner. `completed-gpu-checkpoint-review.json` verifies iteration completion, the CPU checkpoint, best-model decoding and the minimum of recorded full checkpoints. The GPU coordinator publishes fit-result files when the whole current fit batch finishes; this interim evidence distinguishes a completed per-fit budget from batch publication, architecture selection and numerical convergence. Other fits remain live.

### 2026-10-03 23:20 UTC: hybrid dispatch and refined synthetic review

Parallel CPU measurements on the same training-only 10000-bin blocks and frozen models reproduce all six serial filter outputs bitwise. Six CPU threads evaluate dual-OU batches of 2/4/8/16 in 0.297/0.484/0.578/0.906 seconds (`CPU-parallel-batch-benchmark.json`); the archived single CUDA evaluation took 1.094 seconds. Small groups therefore use the CPU reference while larger groups retain CUDA. Dispatch keeps each plus/minus trajectory pair on the same random generator, including pairs split by a group boundary. CPU tasks overlap CUDA work. The per-update log records actual CPU-NumPy and CUDA-Philox trajectory counts. This is a numerical backend transition, not a different scientific likelihood, parameter set, conditioning context, score weighting or update budget.

`hybrid-migration-snapshot` preserves the original CUDA sources, 389 completed fit hashes and all 60 pending optimizer states at iterations 40–150. Each `backend-migration.json` binds the new source/dispatch fingerprint to the exact old checkpoint and transition bytes. Raw parameters, moments, optimizer RNG and original CPU-to-CUDA provenance are retained. `hybrid-migration-verification.json` verifies all 60 transitions/configurations and unchanged completed fits. An asynchronously finished initial CPU checkpoint is restored from `best-validation.json` when its history row was not yet committed at interruption; tests verify this recovery and rejection of missing/mismatched transition manifests. The complete Hawkes suite now passes **88 tests** (`hybrid-migration-tests.log`). Live timing varies with sampled contexts; the first hybrid cycles remain about 80–92 seconds, so no additional end-to-end speedup is claimed from these few updates. The runner has 57 active fits after three saved budgets reached 150 updates.

The refined synthetic sell-to-hidden sensitivity slice completed at 512 particles, two seeds and five scales after the all-coordinate joint fit. At the generating scale of 6s versus fitted 2.24603494s, training NLL changes are -3.1450e-6 and -7.5122e-6; both paired 95% intervals include zero. Validation changes are +3.7433e-5 and +2.2027e-5; only the first interval excludes zero. `synthetic-ML-long/refined-scale-slice-review.json` binds the model/slice hashes, and its scientific plot was visually checked. This supports weak fixed-parameter sensitivity, not joint profile confidence bounds, optimizer convergence or physical identification. The predeclared factor-two recovery test remains failed.

The preserved CPU selection CSV also confirms that ML subsequently favored 41-parameter control `18bd11b15cd6c05ac4ff` with coarse raw NLL -29.1728000615 (score -29.1725898223), slightly better than the evaluated 43-parameter point. A 512-particle/512-path forecast of that actual completed incumbent now runs separately in `ML/controlled-41-parameter-18bd11-provisional-horizons`. Neither point is promoted to the final selection while architecture search and stopping review remain incomplete. Actual final-test evaluation remains pending.
### 2026-10-03 23:31 UTC: current 41-parameter ML validation completed

The incumbent `18bd11b15cd6c05ac4ff` completed validation with 512 filter particles and 512 future paths on the same 194 covered hour origins, with zero failed paths. `ML41-validation-review.json` binds the exact source model (SHA256 `54a15ce64f36d20d50dbb193d21195dcd1808b25948c8186637ff9a43c2513c5`) and forecast archive (SHA256 `3536fba7d3c9ce5127c3491d72ba054779e5c9e335e0ce72a998ad3474c8a660`). `ML41-ML43-validation-comparison` contains the matched paired comparisons and checked figure.

Proper one-step marginal NLL is 10.7837031, MAE 15.8590598 and R2 0.1096355. Compared with M0, NLL improves by 10.3074206, with a conditional paired 95% interval of 8.9663865 to 11.5544429. Relative to ML43, the NLL change is -0.0003848 and its interval includes zero (-0.0008491 to 0.0000593).

At one hour, ML41 has MAE 26171.1549, R2 -0.2015644 and 90% interval coverage 0.1958763. Its total mean bias is +17494.6876 events (+33.09%). Compared with M0, hour MAE is worse by 13477.6999 (paired interval 7826.7886 to 18289.2612). Compared with ML43, MAE improves by 347.6098 (212.9972 to 449.4266), but the poor long-horizon result persists. The additive forecast means are exact, so their bias is not Monte Carlo mean noise. These are completed incumbent validation results, not a final ML choice or an optimizer-convergence claim. Latent architecture searches continue on the GPU; the final test remains unevaluated in this pipeline.

### 2026-10-03 23:42 UTC: current ML particle-accuracy audit started

A CPU reference audit of the exact incumbent `18bd11b15cd6c05ac4ff` now runs at 128, 512 and 2048 particles, with seeds 71391 and 91337, alongside the GPU architecture search. `ML/audit-18bd11-128-512-2048/source-bindings.json` records the model and implementation hashes; the model hash agrees with the completed forecast review. This uses only the training and validation corpus and performs no fitting. The two completed 128-particle runs give proper validation NLL 10.7837033094 and 10.7837050779, closely matching the existing 512-particle forecast result. The remaining four replicates are pending. `review_ml41_particle_accuracy.py` will verify the exact bindings and summarize changes across particle counts after completion; architecture selection and optimization stopping remain separate requirements.

The two 512-particle replicates subsequently completed with proper NLL 10.7837031151 and 10.7837040343. The first exactly reproduces the completed validation forecast's one-step metrics; the second remains finite and differs by only 0.0000009191. The two-seed mean change from 128 to 512 particles is approximately -0.0000006190. Both 2048-particle replicates remain pending, so the numerical review is incomplete. The GPU search has advanced to iterations 68–101 across 57 active fits without changing its joint-refit budgets or search protocol.

### 2026-10-03 23:57 UTC: complete incumbent parameter export

`completed-incumbent-parameters/all_parameters.csv` exports 153 rows covering every fitted background, directed exponential mass and continuous scale, anchor scale, modulation coefficient and normalization reference, and OU scale/loading for the preserved completed-fit incumbents. `directed_mixture_counts.csv` includes every possible directed edge, including empty mixtures; `model_dimensions.csv` gives the jointly free parameter counts. The source model hashes match the M0/M2 reviewed audits and the original completed-fit migration snapshot. Exact model serializations, source fingerprints and CSV hashes are in `manifest.json`; the README records units and interpretation.

The exported points have 28/80/41/43/49 joint parameters for M0/M2/ML/M2L/M2LZ respectively. Their exponential counts are 13/15/19/20/19; M2 has three anchors, while the completed M2L point has no modulation anchors and reduces to an ML model. M2LZ has two OU components. These exports make all incumbent parameters reviewable without claiming global optimality or a completed latent search. They are not the final frozen five-model parameter report, and no final-test evaluation was performed.

### 2026-10-04 00:09 UTC: first 2048-particle ML replicate completed

The current ML incumbent's seed-71391 audit completed at 2048 particles in 1097.687 seconds. Proper validation NLL is 10.7837042592, a change of +0.0000011440 from the same seed at 512 particles. Mean ESS fraction is 0.9998975060, with no scored seconds below 10% ESS; all scored likelihoods are finite. This small measured change supports numerical stability of the large likelihood gain over M0 for this fixed point. The second 2048-particle seed is still running, so the two-seed numerical review remains incomplete. Audit source fingerprints are checked against their original binding before progress recording, rather than overwritten on refresh. The GPU architecture fits remain live and have reached iterations 81–114 of 150; optimization stopping, architecture selection, five-model freeze and final test remain pending.

### 2026-10-04: five completed search incumbents and nested baselines

`completed-incumbent-validation-comparison` compares M0, M2, ML41, the M2L search's collapsed ML43 point, and dual-fast long-BB M2LZ. Exact serialized parameter equality allows the M2L point to reuse the completed ML43 forecasts by reference; `M2L43-collapsed-validation-review.json` records both source hashes and the forecast binding. All five series have identical covered origins and targets. The comparison figure was visually checked. Proper one-step NLL comparisons use 512 particles for all latent points: 21.0911238, 21.0774639, 10.7837031, 10.7840879 and 4.9532106 respectively. M2LZ's matched density is derived from its existing frozen-cohort 512-particle loss trace with the common observed-count factorial term. Its existing horizon forecast remains a separately identified 256-filter-particle, 512-future-path pilot.

The completed points' hour MAE/R2 values are M0 12693.4550/0.4305929, M2 12648.9184/0.3932145, ML41 26171.1549/-0.2015644, M2L-search ML43 26518.7646/-0.2277718, and M2LZ 18372.9810/-0.2280582. Relative to ML41, the matched M2LZ one-step NLL improvement is 5.8304925 with conditional paired 95% interval 4.7275919 to 7.0239626. Density gains still do not establish better hour-ahead point forecasts. No final-test observations or new fits were used.

`nested-completed-candidate-pool` separately verifies all 389 preserved completed source hashes, joint-fit budgets, corpus bindings and checkpoint particle settings, then admits ML unchanged into M2L/M2LZ and M2L unchanged into M2LZ. The strongest eligible M2L point is consequently the 41-parameter ML incumbent, not its own search's slightly worse 43-parameter point. Candidate aliases `0a593ed57bd74132d6db` and `18bd11b15cd6c05ac4ff` have the identical fitted-model file hash. Final selection must retain these optimized lower-family baselines; relabeling an identical model provides no statistical improvement. The pool excludes unfinished accelerator fits and remains provisional.

### 2026-10-04 00:28 UTC: completed ML41 numerical review

All six fixed-parameter CPU reference runs completed for candidate `18bd11b15cd6c05ac4ff` at 128, 512 and 2048 particles with seeds 71391 and 91337. `ML41-particle-accuracy-review.json` verifies the original model and implementation fingerprints, exact corpus identity, each stored replicate, and the aggregate summary. The two-seed proper-NLL means are 10.7837041937, 10.7837035747 and 10.7837043065 respectively. The 512-to-2048 mean change is +0.0000007318; the individual seed changes are +0.0000011440 and +0.0000003196. Seed SD at 2048 particles is 0.0000000670.

These measured changes are negligible relative to the roughly 10.3074 NLL gain over M0 for this fixed incumbent. They also do not account for the existing roughly 0.0003848 ML41/ML43 point difference, whose conditional paired sampling interval still includes zero. This numerical evidence is specific to the audited parameters; it does not establish optimizer convergence, physical identification, unfinished architecture selection or accuracy of every latent candidate. The audit process exited successfully and is removed from the running-job status. The GPU search remains live at iterations 95–128 across 57 active fits. The final model freeze and actual final-test evaluation remain pending.

The fixed dual-fast long-BB M2LZ incumbent `165b72e9d9b2e0330932` now receives its own 2048-particle reference audit, with seeds 71391 and 91337 evaluated in two concurrent CPU threads while the GPU search continues. Both jobs share readonly model/corpus inputs and write independent per-seed outputs. Existing bitwise concurrent-CPU tests and benchmarks cover this filter implementation. `M2LZ/audit-165b72-2048-parallel/source-bindings.json` records the exact model, corpus and implementation hashes. Its existing frozen-cohort 128/512-particle loss traces supply the lower-particle comparisons; no filter states or mean-error metrics are invented for those cached traces. This audits a strong retained reference point and does not freeze the unfinished architecture selection.

### 2026-10-04 00:40 UTC: ML validation stopping evidence

`ML41-validation-stopping-review.json` verifies the completed deletion fit `0a593ed57bd74132d6db` and unchanged-architecture refit control `18bd11b15cd6c05ac4ff`, their original operations, checkpoint hashes, parameter dimensions and 150-update budgets. The control's initial serialized model is exactly the retained ML41 point. Both runs start from identical raw vectors and use optimizer seed 20261003; they are not independent multistart trials.

At the 32-particle full checkpoints, training raw NLL improves from -34.6220145085 to -34.6321346805 (change -0.0101201721), while validation raw NLL worsens from -29.1728000615 to -29.1566150265 (change +0.0161850350). The resulting fitted files are identical and retain the initial validation checkpoint. This explains the recorded validation stopping decision after an all-coordinate refit. Only initial and final full checkpoints were evaluated, so no intermediate validation monotonicity or optimizer convergence is inferred. The architecture search and two-seed high-particle M2LZ audit are still running; this partial stopping review does not make ML ready for final freezing.

The resumed M2L candidate `15fdbaab25bab96cf6bf` subsequently completed all 150 joint updates and its full CPU validation checkpoint. Its 61 free parameters include one latent-source anchor at 300 seconds. The best checkpoint remains the initial point, raw validation NLL -29.1489179511; the final point is worse at -29.1378043919. The retained no-anchor ML41 baseline has raw NLL -29.1728000615 and fewer parameters, so this candidate does not improve the completed eligible M2L pool. `completed-gpu-checkpoint-review.json` verifies the best-model/raw-vector agreement, checkpoint minimum and exact source hashes. All four originally interrupted fits now have verified completed budgets; newly queued neighborhood candidates, architecture stopping and final evaluation remain pending.

### 2026-10-04 01:23 UTC: completed dual-incumbent high-particle audit

Both 2048-particle reference runs of fixed dual-fast long-BB candidate `165b72e9d9b2e0330932` completed successfully in parallel, taking 2495.156 and 2504.984 seconds. Proper validation NLL is 4.9510805892 and 4.9511606395. The two-seed mean is 4.9511206143, compared with 4.9532547557 at 512 particles and 4.9598004130 at 128 particles. The successive changes are -0.0065456573 and -0.0021341414; the 2048-particle seed SD is 0.0000566041.

Mean ESS fraction is about 0.5831, with about 3.10% of scored seconds below 10% ESS. Both scored likelihoods remain finite. The one-step total-count MAEs are 14.6387407 and 14.6386422, with R2 0.0700400 and 0.0700907. These agree closely with the lower-particle mean-error pilot, while the density score still exhibits systematic particle-count drift. `M2LZ/audit-165b72-2048-parallel/review.json` binds the exact model, corpus, four cached lower-particle loss traces, two new high-particle results and implementation hashes; all file bindings were independently checked after completion. The audit is removed from the live-job status.

The density gain over ML remains large, but this fixed-point audit does not prove absolute particle convergence or rank unfinished nearby candidates. Its existing hour forecast still uses 256 filter particles and is identified as a pilot. The GPU architecture batch continues with 56 fits at update 137 of 150; final architecture/stopping reviews, model freeze and actual final-test evaluation remain pending.


### 2026-10-04 02:03 UTC: first accelerator cohort and three-OU shortlist

The first 60-fit accelerator cohort has completed and published its full CPU checkpoints. M2L independently reached the same 41-parameter no-anchor point as ML. The dual-fast M2LZ lineage now has three leading 51-parameter, three-OU candidates: `dda17b4d6971f5ed89e9`, `165876e3c67c3b9e10d3`, and `8101483264904025659a`. Their coarse penalized validation scores are -34.9902430551, -34.9884714831 and -34.9842865430; the preserved two-OU incumbent scores -34.9797428551. These particle-32 checkpoint margins require matched higher-particle review.

`M2LZ/ranking-three-OU-first-cohort/protocol.json` freezes those three candidates, the initial seed, the unchanged-architecture joint-refit control, and the previous two-OU incumbent. The wrapper evaluates 12 new likelihood traces at 128/512 particles and two seeds using four CPU threads; 12 unchanged reference traces are copied from their exact-model, exact-corpus prior cohort. Code/source bindings and atomic cache publication protect resumability. The next 64-fit GPU neighborhood batch runs concurrently at about 14–18 seconds per update. No new fitting is done in this ranking and no test observations are loaded. The old 48–96 hour planning estimate is superseded; the complete ETA depends on the remaining architecture rounds.

The final evaluation now reuses a completed latent-family forecast when another frozen family has exactly identical serialized parameters, corpus identity and evaluation settings. Numerical artifacts remain byte-identical; diagnostic and table family labels are rewritten, and source hashes are recorded in an explicit reuse manifest. This avoids repeating the entire ML filter/forecast if M2L still selects the same model at freezing. Changed parameters, particle counts or corpus identities and incomplete source outputs are rejected. All seven final-evaluation tests pass, including the new reuse and rejection checks. This change does not touch the running search or its fitting backend.

The first three-OU candidate's two 128-particle replicates completed in 194.563 and 201.031 seconds. Against the preserved two-OU incumbent at the same particle counts and seeds, raw validation NLL changes are -0.0146586341 and -0.0142995716. Conditional paired scored-hour-block intervals are [-0.0261239794, -0.0057978432] and [-0.0264189722, -0.0048826303]. The observed-count factorial constants cancel in these differences. This supports the candidate's density gain at 128 particles; the 512-particle runs and other frozen candidates remain pending. The two report tests also pass after the exact-model reuse change.

`final-evaluation-plan.json` fixes the common latent filter at 512 particles, 2048 future paths, 900-second origin spacing, and horizons 1/5/15/60/300/900/3600 seconds. Its stated numerical rationale retains the validation particle-drift disclosures and the distinction between density and point-forecast accuracy. Before test loading, the final runner now requires the five-model freeze to bind this plan's path and SHA256, verifies the file hash, and rejects mismatched evaluation budgets, seeds, horizons, warmup or corpus hash. The plan is copied into final results. Ten targeted final-evaluation/report tests pass, including rejection before test access. No model has been frozen and no test data opened by these changes.

### 2026-10-04 02:16 UTC: three-OU candidate matched-particle evidence and forecasts

The activity candidate `dda17b4d6971f5ed89e9` now has both 512-particle replicate losses. Proper validation NLL is 4.9372691022 and 4.9373179340, mean 4.9372935181. At matching seeds and particles, changes from the preserved two-OU incumbent are -0.0159415133 and -0.0159809620; paired scored-hour-block intervals are [-0.0288511116, -0.0061064192] and [-0.0287366903, -0.0061928420]. Its 128-to-512 two-seed mean NLL change is -0.0080277921. Absolute particle convergence remains unproven. The next imbalance candidate is worse by 0.0054175449 and 0.0056530273 at 128 particles; its 512-particle runs and the third candidate remain pending.

`run_three_ou_forecasts.py` freezes this exact completed candidate and implementation fingerprints for validation-only forecasts at 512 filter particles, 512 future paths and the shared 900-second origins. It checks corpus identity against the frozen ranking cohort. Two Numba rollout threads limit contention with the four-thread CPU ranking and the hybrid GPU search. This run tests whether the supported density gain improves the one-hour point forecast; it is not final model selection and does not load test observations. Session 17620, actual process 29320.

### 2026-10-04 02:29 UTC: second accelerator batch publication

The second 64-fit batch completed its CPU full-data checkpoints and published the finished fits. ML now has 215 completed candidates; new `733c5ba0b1a22599021f`, adding a rate-calibrated L-to-L exponential, has 43 parameters and coarse penalized score -29.1773493999. It improves the previous ML41 score -29.1725898223. M2L still has 191 published candidates with the original ML41 point as its own incumbent; the improved no-anchor ML43 point remains eligible for M2L at final nested selection. The single-start M2LZ lineage has 86 completed candidates and improved to 47-parameter deletion `c1295ee688b2568ae80c`, score -34.8240020066. The dual lineage still leads with the three-OU activity point, score -34.9902430551. A third 64-fit batch is now running. These remain coarse search results, with selected-point numerical/stopping reviews and the final freeze pending.

Both 512-particle short-imbalance replicates now trail the activity candidate by 0.0079019211 and 0.0078312854 NLL per scored second. Paired intervals are [0.0045583417, 0.0120194027] and [0.0046909048, 0.0114885785]. The long-imbalance 512-particle checks remain pending. The activity fit retained its final joint checkpoint: train raw NLL improved from -40.2569288928 to -40.2898146085 and validation from -34.9448820698 to -34.9905045721 across the 150 updates. Its three OU scales are 1.1090167, 0.9589486 and 129.8748099 seconds; it has 18 directed exponential modes and 51 free parameters. No optimizer-convergence claim is made.

### 2026-10-04 02:34 UTC: complete three-OU shortlist and horizon review

All 24 frozen-cohort likelihood replicates completed successfully, and the activity candidate ranks first at both 128 and 512 particles. Its completed forecast run matches the cached 512-particle seed-71391 predictive loss trace exactly, and all 194 origins match M0 with no path failures. One-step NLL/MAE/R2 are 4.9372691022/14.5068708802/0.0908851577. Hour MAE is 18284.0088, R2 -0.2088788, mean bias -23.97%, and 90% coverage 31.44%. Against M0, hour MAE is worse by 5590.5538 [1275.9066, 10745.7560], and R2 by 0.6394717 [0.2883701, 0.9542918]. The hour MAE gain over the prior two-OU reference is only 88.9722; its interval includes zero. The matched 512-particle NLL gain is 0.0159415 [0.0019070, 0.0352309].

`review_three_ou_results.py` verifies source/code/corpus bindings, every cohort source and loss artifact, joint-fit checkpoint selection, and forecast alignment; `M2LZ/three-OU-validation-review.json` stores the evidence. `three-OU-validation-comparison/README.md` explains the comparison, including the old two-OU horizon filter's 256 particles versus the new point's 512. Its scientific plot was visually checked. Both completed processes exited successfully and were removed from live status. A two-seed 2048-particle fixed-point audit now runs in session 58098 (actual process 36148), reusing the four exact frozen lower-particle losses. The architecture search remains active; final freezing and test evaluation remain pending.

### Final five-model freeze publication command

`ml/hawkes_freeze_selection.py` consumes a JSON request containing `cache`, `evaluation_plan`, and all five family entries with `model`, `selection_audit`, and `selection_csv` paths. Paths may be relative to the request file. It binds every source hash, the exact evaluation plan, the corpus byte hash and the builder source hash. The same final-evaluation validators verify hierarchy restrictions, review completeness and source consistency before an immutable manifest is atomically published; test observations are never loaded. Repeating the same verified request is idempotent, and a differing existing freeze is preserved. Unfinished audits produce no freeze. Twelve targeted publication/evaluation/report tests pass. No real five-model freeze was created: the architecture and selected-point numerical reviews remain pending.

### 2026-10-04 03:07 UTC: third accelerator batch publication

The third 64-fit accelerator batch completed all CPU full-data checkpoints and published its fits. M2L now has 242 completed candidates and independently reaches ML's no-anchor 43-parameter point `733c5ba0b1a22599021f`, coarse penalized score -29.1773493999. ML has 215 completed candidates. The single-OU-start lineage has 107 completed candidates and now selects 45-parameter deletion `00f585ee620f4931959a`, score -34.8588771138. The dual lineage has 84 completed candidates and improves to three-OU deletion `76ae63b74f489e6beed2`, 17 exponential modes, 49 parameters and score -35.0085642318. Its completed 51-parameter refit control `dc73444cd7f8d8d695b5` also improves on the previous reference, with raw score -35.0039133545. The completed-budget verifier independently checked 189 checkpoint/model records.

A fourth batch of 50 fits is now running. The completed three-OU cohort/forecast comparison and live 2048-particle audit remain bound to the earlier 51-parameter reference; they do not assert numerical accuracy or horizon metrics for the new 49-parameter point. Final selected-point reranking, stopping, numerical accuracy and freezing remain pending.

### 2026-10-04 03:28 UTC: completed three-OU reference accuracy audit

Both 2048-particle reference jobs for fixed candidate `dda17b4d6971f5ed89e9` completed successfully, in 2923.641 and 2928.546 seconds. Proper NLL is 4.9336807555 and 4.9335824230; mean 4.9336315892, seed SD 0.0000695316. Mean changes from 128 to 512 and 512 to 2048 particles are -0.0080277921 and -0.0036619288. At matching 2048-particle seeds, the three-OU reference improves on the prior two-OU reference by 0.0173998337 and 0.0175782165. Mean ESS is about 59.1%, with about 2.92% of scored seconds below 10% ESS; likelihoods remain finite.

The source model, wrapper, cohort protocol, four implementation modules and all six referenced result files were independently verified by SHA256 after successful process exit. Session 58098 is removed from live status. This quantifies remaining likelihood approximation drift and supports the fixed-reference density gain; it does not establish absolute particle convergence or accuracy of the newer 49-parameter search point. Its completed horizon comparison remains a 512-filter-particle pilot. The GPU search continues through the fourth batch's full-data checkpoints; final model selections and test evaluation remain pending.

### 2026-10-04 03:36 UTC: fourth accelerator batch publication

The fourth 50-fit batch published its completed joint fits. ML and M2L now both lead with deletion `fba675d2818cdfe979cf`, 41 parameters and coarse penalized score -29.1895022545; this is a different, improved parameter set from the previously audited ML41 reference. Their tables contain 236 and 263 completed candidates. The single-start M2LZ lineage has 127 completed candidates and improves to 43-parameter deletion `2706c5b1567b93f4a677`, score -34.8687506339. The dual lineage has 105 completed candidates and selects deletion `d2364f07902e8b2ff69f`: 47 parameters, 16 exponential modes, three OU components, and score -35.0291610889. The budget/checkpoint verifier now covers 304 completed records; shared model records across families are not independent fits.

A fifth batch of 59 fits is active. Neither latent-family architecture search has stopped. The old ML41/three-OU reference accuracy and forecast artifacts remain valid for their exact parameter sets; their metrics are not assigned to these improved points. Final selected-point numerical checks, optimization stopping review, freeze and final test remain pending.

### 2026-10-04 13:15 UTC: confirmed runner stop and checkpoint resume

The old process 20076 and unified execution session 14676 are both absent. Its log ends at 06:39 UTC with a 64-fit batch at update 78 and no recorded traceback. The GPU remains RTX 3070 and the corpus byte hash is unchanged. Before resuming, `resume-2026-10-04-1315-snapshot/audit.json` records 975 completed result hashes and snapshots 64 unfinished optimizer states at update 78. Current backend fingerprints match the saved transitions. All completed result hashes and all copied partial snapshot hashes were independently checked after launch; none changed.

The same plan, batch size 64 and six CPU workers now run in session 94064, actual process 11660. The new log is `hybrid-resume-2026-10-04-1315.log`. Saved raw parameters, Adam moments and RNG state are reused. Completed-fit replay also reuses the stored outcomes while reconstructing the greedy search; newly queued fits can share batches with resumed ones. The restart follows a confirmed terminal runner, not an observation timeout. No backend settings were changed. At the stop, ML/M2L tables led with 43-parameter `b384631aa5468da31e1c`, score -29.1991650733, and the dual M2LZ lineage led with 43-parameter `a472c2f9c434dbd01a28`, score -35.0610965508. Their exact final-point numerical/horizon checks and all five-model freezing remain pending.

### Current ML/M2L reference numerical audit after resume

A two-seed CPU reference audit now evaluates exact 43-parameter point `b384631aa5468da31e1c` at 128/512/2048 particles while the resumed architecture search runs. ML and M2L share this same serialized no-anchor model. `ML/audit-b38463-128-512-2048-parallel/source-bindings.json` records the source model, corpus identity and byte hash, implementation and wrapper fingerprints. The fitted progress corpus identity was independently checked against the audit input. Session 5947, actual process 41012. This audit performs no fitting or test evaluation and remains specific to this reference if the ongoing search finds a better point.


### Bounded architecture search after runtime objection

The user rejected a multi-day runtime. The active GPU search was stopped at preserved per-update optimizer checkpoints on 2026-10-04. Completed 150-update fits are retained; unfinished fits are excluded from selection. This is a finite compute-budget stop, not a forward/backward plateau or an optimization convergence certificate. The frozen completed-fit pool and shortlist are in `bounded-selection/completed-fit-pool.json`. Final validation reranking, exact-point particle checks, five-model parameter freeze and final evaluation remain pending.


### Bounded ML numerical and horizon review

Exact completed point `b384631aa5468da31e1c` has finished two-seed audits at 128, 512 and 2,048 particles. Mean proper validation NLL is 10.757106142629013, 10.757104122355969 and 10.757103587049006 respectively; 512-to-2048 drift is -5.353069632008101e-7. All source, code and artifact hashes were checked. P512/512-path validation forecasts cover 194 identical origins, with no failed origins: one-hour total MAE 25399.293669860424, R2 -0.202024895169705, 90% interval coverage 0.13402061855670103. This point improves density but remains materially worse than M0 for hour-ahead means. Final shortlist selection and final evaluation are still pending.


### Fixed completed-fit ranking and five-model validation

The fixed eight-point cohort completed all 32 evaluations (128/512 particles, seeds 71391/91337). ML and M2L both select exact point `b384631aa5468da31e1c`, 43 parameters, 20 exponential modes and no anchors. M2LZ selects `a472c2f9c434dbd01a28`, 43 parameters, 14 modes and three OU components. Mean M2LZ predictive NLL is 4.877558596228274 at P128 and 4.872251128870288 at P512. The 13-mode alternative `1a68be04919a1c44ae59` has paired 1-hour block intervals overlapping zero, so unique superiority of the selected mixture count is not established.

Selected M2LZ P512/512-path forecasts completed 194 validation origins without failures: hour MAE 17167.936551386134, R2 0.07101128087542585, nominal-90% interval coverage 0.5824742268041238. Its hour MAE is worse than M0 by 4474.481541320591 (conditional 95% interval [1847.3336386611338, 7057.221155363636]); its R2 is lower by 0.3595815939721535 (interval [-0.5004275273125799, -0.10739420911468535]). Five-model validation comparisons are in `bounded-selection/validation-comparison`; ML/M2L reuse the identical model predictions by reference. The exact M2LZ P2048 audit and frozen final comparison remain pending.


### All-five finite-budget parameter freeze

M2LZ exact-point P2048 audits completed: seed 71391 NLL 4.87015239593766, seed 91337 NLL 4.870106185034864; mean 4.870129290486261. P512-to-P2048 drift is -0.0021218383840269084, so likelihood approximation is quantified, not treated as absolutely converged. Module, model, wrapper and result hashes were checked. All five evidence audits passed the freeze validator; `bounded-selection/frozen-selection.json` binds models, selection evidence, cache and the preselected final evaluation plan. The actual final evaluation has started with 512 filter particles, 2048 future paths and 900-second forecast spacing. Parameters are fixed; no post-freeze fitting is allowed. This is a compute-budget study, without architecture plateau, mathematical convergence, globally optimal counts or physical latent identifiability claims. Historical exploratory test inspection remains disclosed.


### Completed empirical five-model study

The frozen final evaluation completed all five models on 181609 scored seconds and 190 identical forecast origins, with P512 filtering and 2048 future paths. All frozen source, evidence, cache and plan hashes are unchanged. ML/M2L exact numerical reuse was verified. The report includes 35 model/horizon rows, 152 parameter rows covering every frozen coordinate, all per-edge mixture counts, conditional paired intervals, inferred-state plots and scientific limitations. `final-evaluation-bounded/completion-audit.json` records the closing checks.

Final one-second NLL / MAE / R2: M0 14.497926640631718 / 9.846034223093575 / 0.1664882783781112; M2 14.475880742249473 / 9.802259332932328 / 0.16815095846717887; ML and M2L 7.707485564827357 / 10.973546607336111 / 0.1526406020511496; M2LZ 4.285824870414678 / 11.661425548271831 / 0.10665710135208784. Hour MAE / R2: M0 15292.942030148066 / 0.4351006035615007; M2 13191.452811240444 / 0.43654643231911383; ML/M2L 38202.19953267393 / -0.7162541905607083; M2LZ 25353.496522524507 / -0.03780370541063127. M2 hour-MAE improvement over M0 is -2101.489218907619 with conditional 95% interval [-4035.1587903194777, -468.26204377913473]; the R2 increase is not established. M2LZ hour nominal-90% interval coverage is 0.22105263157894736 despite its density advantage. No simulated final origins failed.

Decision: retain M2 as the leading mean-forecast candidate, M0 as the simpler globally bounded baseline/fallback, and M2LZ for short-horizon density research. Drop ML/M2L as separate candidates. The final report is `final-evaluation-bounded/report.md`, with full precision in `fitted_scales.csv` and each family's frozen-model JSON. Finite compute-budget stopping, unresolved mathematical/global/count/physical identifiability, non-pristine historical test inspection, conditional uncertainty and M2's absent global modulation bound remain explicit. No live model deployment or trading-profitability claim was made. The requested empirical experiments and reporting are complete.


### Imbalance follow-up: buy count minus sell count

Derived imbalance MAE, RMSE, R2, Pearson correlation and bias directly from the unchanged frozen `test_forecasts.npz` files for all five models and seven horizons. No fitting or forecasting was repeated. Results are `final-evaluation-bounded/imbalance_horizon_metrics.csv`; formulas and source hashes are in `imbalance_metrics.provenance.json`. These use 190 sampled origins, distinct from the original full one-step correlations on 181609 scored seconds. No imbalance marginal NLL or new uncertainty intervals were computed.

M0 has the lowest point-estimate imbalance MAE and highest R2 among the models at 1m, 5m, 15m and 1h. Its MAE/R2 are 167.5513/0.059384, 509.4016/0.084985, 827.3169/0.100800 and 2228.8464/-0.089380. All models have negative one-hour imbalance R2; M2LZ has hour MAE 5129.1368, R2 -2.215399 and positive bias 4824.4091. The earlier M2 preference concerns total-activity means, whereas M0 is the strongest imbalance candidate by these point estimates. Close M0/M2 differences have not been assessed with paired uncertainty intervals for imbalance.
