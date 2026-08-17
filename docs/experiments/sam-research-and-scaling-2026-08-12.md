# Sharpness-aware minimization research and empirical scaling

> **Status, 2026-08-12.** Standard globally normalized SAM improves the
> fixed-budget training fit of the four-layer width-512 GLU most consistently
> around `rho = 0.1`. Larger radii do not merely suppress fitting: they create a
> long delayed-escape phase followed by ordinary fast fitting. On the current
> noisy next-second forecasting task, however, the resulting raw validation
> improvement remains small and inconsistent, and no completed run establishes
> a robust forecasting edge.

This note connects the current SAM literature to the controlled next-second
GLU diagnostics and records the empirical scaling laws extracted from their
training trajectories. The equations below are descriptive fits to this one
architecture, optimizer, batch size, dataset construction, and training budget.
They are not general SAM laws.

## Experimental question

The diagnostic asks two separate questions:

1. Can the normalized GLU fit an increasingly large fixed set of next-second
   returns?
2. Does SAM change that fit/generalization trade-off, and how does its
   perturbation radius change the time required to start fitting?

Each example contains the previous 120 completed one-second close log returns
and targets the next completed one-second close log return. Inputs use the
training-wide per-lag-position mean and standard deviation. The controlled
four-layer model has width 512, static centering matrices, learned normalization
radii, 3,801,609 trainable parameters, and 5,898,761 stored parameters.

Unless explicitly stated otherwise, every run uses:

- fixed contiguous subsets beginning on 2026-04-01;
- `N = 65,536`, `131,072`, `262,144`, or `524,288` examples;
- 512 epochs and batch size 4,096;
- fixed learning rate `1e-4`;
- the hybrid Muon/AdamW optimizer with no weight decay;
- no dropout, no early stopping, and float32 computation;
- deterministic full-training-subset metrics after every epoch; and
- the common 2026-06-01 through 2026-06-30 validation split containing
  2,591,880 examples.

The run plans are enumerated in
[`ml/training-matrices/sam-memorization-rho-sweep-v1.json`](../../ml/training-matrices/sam-memorization-rho-sweep-v1.json).
The no-SAM capacity baseline and model definition are documented in
[`next-second-glu-recent-4m-replication-2026-08-07.md`](next-second-glu-recent-4m-replication-2026-08-07.md).

## SAM used here

For weights \(w\), training loss \(L\), and radius \(\rho\), the implementation
uses ordinary non-adaptive SAM with a single globally normalized perturbation:

\[
g=\nabla_w L(w), \qquad
\epsilon=\rho\frac{g}{\lVert g\rVert_2},
\]

\[
g_{\mathrm{SAM}}=\nabla_w L(w+\epsilon).
\]

The original weights are restored and the base optimizer applies
\(g_{\mathrm{SAM}}\). Thus every optimizer update requires two forward/backward
passes. The perturbation is normalized over all trainable parameters together;
it is not ASAM and does not apply layerwise \(\mu\mathrm{P}^2\) scaling.

## Research overview

### Foundation and dynamics

The [original SAM paper](https://research.google/pubs/sharpness-aware-minimization-for-efficiently-improving-generalization/)
frames training as minimizing the worst loss within a radius-\(\rho\) weight
neighborhood. It reports improved generalization, but the practical update
doubles gradient computation and does not prescribe an efficient late-start or
radius schedule.

[The Dynamics of SAM](https://jmlr.org/papers/v24/23-043.html) describes SAM as
bouncing across narrow ravines while drifting toward wider minima. A separate
[stability analysis](https://arxiv.org/abs/2301.06308) shows that SAM can escape
saddle points more slowly than SGD and identifies higher momentum and smaller
batches as useful escape mechanisms. Our high-radius plateaus and abrupt escape
are consistent with these mechanisms, but the training traces alone do not
establish that the GLU is at a saddle or ravine.

### Avoiding the early cost

[Late-training SAM](https://proceedings.iclr.cc/paper_files/paper/2025/hash/35d5ad984cc0ddd84c6f1c177a2066e5-Abstract-Conference.html)
finds that a few SAM epochs at the end of ordinary training can reach nearly the
same generalization and sharpness as full-run SAM. It identifies an initial
escape from the SGD solution followed by rapid convergence to a flatter minimum
inside the same valley. This is the most direct literature-supported way to
avoid paying for the long early SAM phase observed here.

[SAMPa](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5bf2b802e24106064dc547ae9283bb0c-Abstract-Conference.html)
parallelizes SAM's two gradients and can approach a twofold SAM speedup when two
devices and negligible communication cost are available.
[Momentum-SAM](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4fb596fd04a0eaa7bdd78aae24943a8e-Abstract-Conference.html)
instead uses accumulated optimizer momentum as the perturbation direction and
removes SAM's extra gradient evaluation. ESAM and related gradient-reuse or
sample-selection methods are earlier alternatives for reducing, rather than
eliminating, the additional pass.

### Stability and perturbation quality

[Lookahead SAM](https://proceedings.mlr.press/v235/yu24q.html) targets SAM's
oscillation near saddle points with a lookahead trajectory.
[Stable SAM](https://jmlr.org/beta/papers/v26/24-0065.html) renormalizes the
descent gradient to the ascent-gradient norm, aiming to prevent saddle trapping
and widen the stable learning-rate regime. Both are directly relevant if the
observed plateau is an optimization instability rather than useful
regularization.

[Unified SAM](https://proceedings.iclr.cc/paper_files/paper/2025/hash/4a46787131e2f1d8cf429362f6e2a6ec-Abstract-Conference.html)
unifies normalized and unnormalized SAM and provides convergence results under
more general stochastic-noise and sampling assumptions. The normalized update
matters here: the current experiments use normalized SAM, and the empirical laws
should not be transferred to unnormalized SAM.

The newest direct algorithmic proposals include
[XSAM](https://arxiv.org/abs/2603.10048), a March 2026 preprint that explicitly
estimates the local maximum direction, and
[ED-SAM](https://www.sciencedirect.com/science/article/pii/S0893608026006465),
which combines energy-adjusted perturbations with a direction-corrected update.
They address the possibility that the standard one-step perturbation points in
a harmful or inaccurate direction. They are newer and less established than
the peer-reviewed late-training, scaling, and stability results above.

### Scaling with model size

[\(\mu\mathrm{P}^2\)](https://proceedings.neurips.cc/paper_files/paper/2024/file/449a016a6ce6fba3fe50d05482abf836-Paper-Conference.pdf)
is the most important scaling result for the current implementation. It shows
that ordinary globally normalized SAM becomes effectively last-layer-only as
width tends to infinity, even when global hyperparameters are tuned. Its
layerwise perturbation scaling keeps every layer effectively perturbed and
allows the joint optimum of learning rate and \(\rho\) to transfer across model
widths. Consequently, a radius fitted on the width-512 GLU should not be assumed
to transfer to width 1,024 or to a materially different depth.

[Critical Influence of Overparameterization on SAM](https://proceedings.mlr.press/v286/shin25a.html)
finds that SAM generally benefits more from overparameterization, especially
with noise or sparsity, but still requires sufficient regularization. This is
compatible with a larger model providing more flat solutions, while
\(\mu\mathrm{P}^2\) explains why the perturbation itself must also be rescaled.

## Measured training regimes

Let \(M_e\) be deterministic training normalized MSE after epoch \(e\), and
define the first crossing

\[
E_q=\min\{e:M_e<q\}.
\]

`E_0.99` is used as the operational end of the initial plateau: it is the first
epoch whose training NMSE is at least 1% below the zero-prediction scale.
`E_0.90` is a more conservative escape marker. A missing crossing is
right-censored at the 512-epoch limit.

The four-layer width-512 measurements are:

| Examples | rho | E0.99 | E0.90 | E0.50 | Best train NMSE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 65,536 | 0.05 | 4 | 13 | 27 | 0.000632 |
| 65,536 | 0.10 | 7 | 21 | 36 | **0.000541** |
| 65,536 | 0.30 | 180 | 231 | 263 | 0.000730 |
| 131,072 | 0.05 | 2 | 10 | 20 | 0.001045 |
| 131,072 | 0.10 | 4 | 14 | 27 | **0.000873** |
| 131,072 | 0.30 | 147 | 188 | 217 | 0.002091 |
| 131,072 | 0.50 | 395 | 456 | 509 | 0.482971 |
| 131,072 | 1.00 | >511 | >511 | >511 | 0.997127 |
| 262,144 | 0.05 | 1 | 8 | 18 | 0.002380 |
| 262,144 | 0.10 | 2 | 12 | 24 | **0.002032** |
| 262,144 | 0.30 | 100 | 148 | 178 | 0.005721 |
| 262,144 | 0.50 | 217 | 286 | 339 | 0.045343 |
| 262,144 | 1.00 | 356 | >511 | >511 | 0.950060 |
| 524,288 | 0.05 | 1 | 7 | 17 | 0.006117 |
| 524,288 | 0.10 | 2 | 12 | 24 | **0.005943** |
| 524,288 | 0.30 | 81 | 122 | 150 | 0.019357 |
| 524,288 | 0.50 | 119 | 187 | 239 | 0.060672 |
| 524,288 | 1.00 | 205 | 364 | 479 | 0.389888 |

There is a visible regime boundary between `rho = 0.1` and `rho = 0.3`.
At or below 0.1, `E0.99 <= 7` and `E0.90 <= 21` for every dataset size. At or
above 0.3, the delay dominates a substantial fraction of the 512-epoch budget.
A single power law across both regimes would hide this transition and should
not be used.

## Empirical scaling laws

### High-radius plateau width

An ordinary least-squares fit in log space over the uncensored four-layer
points with `rho >= 0.3` gives:

\[
\boxed{
E_{0.99}\approx 776\,\rho^{1.01}
\left(\frac{N}{65{,}536}\right)^{-0.54}
}
\]

with log-space \(R^2=0.864\) over nine observations. Within the measured
high-radius regime, initial plateau epochs are therefore approximately linear
in \(\rho\), while doubling the dataset reduces the number of plateau epochs by
about \(2^{-0.54}=0.69\).

The more conservative crossing produces a similar law:

\[
\boxed{
E_{0.90}\approx 980\,\rho^{1.07}
\left(\frac{N}{65{,}536}\right)^{-0.41}
}
\]

with log-space \(R^2=0.906\) over eight uncensored observations.

These laws explain why more training data appeared to counteract the
high-radius delay when progress was plotted by epoch. Every epoch contains
\(N/4096\) optimizer updates. In optimizer-update units, the first law becomes

\[
S_{0.99}\approx 12{,}400\,\rho^{1.01}
\left(\frac{N}{65{,}536}\right)^{0.46}.
\]

Thus a larger dataset shortens the plateau in epochs but still increases the
number of optimizer updates, examples processed, and approximate wall-clock
work needed to escape. The dataset is not making each update intrinsically
faster; it is providing more updates per epoch.

### Learning after escape

For the same uncensored high-radius runs, the epochs needed to move from
training NMSE 0.9 to 0.5 fit

\[
\boxed{
E_{0.50}-E_{0.90}\approx
125\,\rho^{1.16}
\left(\frac{N}{65{,}536}\right)^{-0.04}
}
\]

with log-space \(R^2=0.997\) over eight observations. The near-zero dataset
exponent is the strongest measured regularity: after escape, early fitting
progress is approximately dataset-pass limited rather than optimizer-update
limited. In optimizer steps the same relation is approximately proportional to
\(N^{0.96}\).

Larger \(\rho\) does not yield a larger per-epoch decay rate in this phase. It
takes about 30 epochs at `rho = 0.3`, 52--53 epochs at `rho = 0.5`, and 115
epochs at `rho = 1.0` to go from 0.9 to 0.5 where those crossings are observed.
The curves can nevertheless look dramatically faster immediately after escape
because a long nearly flat segment is followed by a coherent descent.

### Fixed-budget interpolation capacity

For the no-SAM four-layer baseline, the best residual training NMSE after at
most 512 epochs follows

\[
M_{\mathrm{best}}\approx 7.41\times10^{-4}
\left(\frac{N}{65{,}536}\right)^{1.10},
\qquad R^2_{\log}=0.995.
\]

The endpoint ratio gives the previously reported approximate `N^1.08` law;
`1.10` is the regression exponent using all four points. This rising residual
is a fixed-model, fixed-epoch memorization-capacity curve, not a statistical
generalization scaling law.

Matched SAM curves are:

\[
\begin{aligned}
\rho=0.05:&\quad
M_{\mathrm{best}}\approx5.60\times10^{-4}
\left(\frac{N}{65{,}536}\right)^{1.10},\\
\rho=0.10:&\quad
M_{\mathrm{best}}\approx4.63\times10^{-4}
\left(\frac{N}{65{,}536}\right)^{1.16},\\
\rho=0.30:&\quad
M_{\mathrm{best}}\approx7.09\times10^{-4}
\left(\frac{N}{65{,}536}\right)^{1.56}.
\end{aligned}
\]

Their respective log-space \(R^2\) values are 0.983, 0.973, and 0.998.
At `rho = 0.1`, SAM reduces residual NMSE relative to no SAM by 31.5%, 39.5%,
40.4%, and 20.9% as `N` grows from 65k to 512k. At `rho = 0.3`, the delayed
phase consumes enough of the fixed budget that scaling becomes much worse.
This is why `rho = 0.1` is the current fixed-budget interpolation optimum even
though larger radii can eventually begin learning.

### Model-size effect

At 512k examples, increasing depth from four to eight width-512 layers changes
the measured low-radius behavior as follows:

| rho | Model | E0.90 | Best train NMSE | Raw validation skill | Validation correlation |
| ---: | --- | ---: | ---: | ---: | ---: |
| none | 4L x 512 | -- | 0.007509 | -16.8519% | 0.02614 |
| 0.05 | 4L x 512 | 7 | 0.006117 | -12.2830% | 0.02469 |
| 0.10 | 4L x 512 | 12 | 0.005943 | -10.7500% | 0.02462 |
| none | 8L x 512 | -- | 0.006646 | **-1.9175%** | **0.03373** |
| 0.05 | 8L x 512 | 4 | 0.006060 | -2.1961% | 0.03273 |
| 0.10 | 8L x 512 | 9 | 0.005689 | -2.2037% | 0.03085 |

Extra depth shortens the low-radius fitting transition and slightly lowers
training residual, but it does not improve validation relative to the already
stronger eight-layer no-SAM baseline. The partial eight-layer `rho = 0.3` run
crossed NMSE 0.99/0.90/0.50 at epochs 103/137/168, versus 81/122/150 for four
layers. Model-size effects are therefore radius-dependent and not monotonic.
There are too few matched depths and widths to fit a defensible model-size
exponent. This is also exactly where globally normalized SAM is least
transferable according to the \(\mu\mathrm{P}^2\) result.

## Validation results

The raw validation MSE skill versus predicting zero is:

| Examples | No SAM | rho 0.05 | rho 0.10 | rho 0.30 | rho 0.50 | rho 1.00 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 65,536 | -22.31% | -8.13% | **-6.62%** | -7.87% | -- | -- |
| 131,072 | -18.28% | -9.48% | -7.44% | **-6.82%** | -7.81% | +0.25%* |
| 262,144 | -16.12% | -11.08% | -9.33% | -7.49% | -11.54% | **+1.35%*** |
| 524,288 | -16.85% | -12.28% | -10.75% | **-8.23%** | -9.33% | -10.32% |

`*` The 128k `rho = 1.0` run never escaped and the 256k run crossed only NMSE
0.99. These validation numbers are real checkpoint measurements, but they
describe a nearly-zero/strongly underfit phase rather than the phase reached by
the lower-radius runs. They cannot establish that a fully trained `rho = 1.0`
model would retain the same validation result.

For the four-layer model, SAM substantially reduces excessive raw output error,
but most runs still have negative raw MSE skill and correlations only around
0.02--0.04. Higher-radius underfit checkpoints sometimes have better
correlation, reaching 0.119 for the 256k `rho = 1.0` checkpoint, but fixed-epoch
comparisons then confound radius with training phase.

The eight-layer no-SAM model remains the stronger completed validation result;
`rho = 0.05` and `0.1` make its raw validation NMSE about 0.27--0.28% worse.
SAM has therefore not yet demonstrated additional generalization for the larger
model. The separate chronological output-calibration study shows that negative
raw MSE skill can largely be excessive amplitude, but SAM checkpoints have not
yet received the same sealed calibration/test treatment. See
[`next-return-output-calibration-2026-08-11.md`](next-return-output-calibration-2026-08-11.md).

Validation was run on each stored best-training checkpoint. Intermediate
checkpoints at `E0.99`, `E0.90`, and `E0.50` were not retained, so a genuinely
phase-matched validation comparison cannot be reconstructed from the current
artifacts. Future sweeps must checkpoint these crossings rather than infer
phase-specific generalization from the final models.

## Conclusions

1. **Large-radius SAM delays fitting rather than preventing it.** The delay has
   a sharp knee between `rho = 0.1` and `0.3` in this setup.
2. **High-radius plateau epochs are approximately linear in rho.** More data
   reduces the delay in epochs, but not in optimizer steps or wall-clock work.
3. **Post-escape progress is approximately pass-limited.** Dataset size has
   almost no effect on the epochs needed to move from NMSE 0.9 to 0.5.
4. **`rho = 0.1` is the current four-layer fixed-budget training optimum.** It
   lowers interpolation residual at every tested dataset size without spending
   most of the 512 epochs on the delayed phase.
5. **Training fit and forecasting generalization remain different problems.**
   The SAM sweep improves raw validation calibration for the weaker four-layer
   models but does not create a stable, material predictive edge.
6. **The radius does not transfer cleanly across model scales.** The eight-layer
   observations and the \(\mu\mathrm{P}^2\) result both argue for layerwise
   perturbation scaling before extrapolating to larger models.

## Recommended next experiments

In priority order:

1. Train the ordinary optimizer first and switch to SAM only late in training,
   comparing equal total forward/backward compute rather than equal epochs.
2. Save and validate checkpoints at `E0.99`, `E0.90`, `E0.50`, and fixed numbers
   of post-escape epochs so radius comparisons are phase matched.
3. Implement \(\mu\mathrm{P}^2\)-style layerwise perturbation scaling and test
   whether one radius transfers between four/eight layers and width 512/1,024.
4. Test Stable SAM or Lookahead SAM at `rho = 0.3--1.0` to determine whether the
   initial plateau is avoidable instability.
5. Benchmark Momentum-SAM against ordinary late-start SAM when compute cost is
   included in the result.
6. Repeat selected cells over multiple seeds. The present laws are fitted to
   one seed and should be treated as hypotheses until replicated.

## Reproducibility and limitations

- Training result: `data/training/runs/<run-id>/state/result.json`.
- Validation result:
  `data/training/runs/<run-id>/state/validation-current-best.json`.
- Epoch trajectory: `data/training/runs/<run-id>/logs/training.jsonl`.
- Plateau fits: ordinary least squares of `log(epoch)` on `log(rho)` and
  `log(N / 65,536)`, using only observed crossings.
- Right-censored crossings are reported but excluded from regression.
- The laws use a single seed, one target, one market period, one batch size, one
  learning rate, and mostly one model size.
- Nested contiguous datasets change both sample count and calendar coverage.
- Training NMSE scaling mixes representation capacity with a fixed epoch
  budget; it is not a population-risk scaling law.
- Raw validation metrics are not trading returns and do not include fees,
  spread, latency, turnover, or execution constraints.

