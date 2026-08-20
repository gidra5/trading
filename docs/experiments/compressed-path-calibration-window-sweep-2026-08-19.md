# Compressed-path calibration window sweep — 2026-08-19

## Setup

- Model: compressed path density, 15 active 1-second returns, immediate 59-feature state, 512k clean examples, no SAM, 4-epoch weight EMA.
- Primary comparison checkpoint: best validation correlation (epoch 17 in the UI).
- Calibration data remains strictly before validation; test is untouched by fitting.
- Windows: 64k, 32k, 16k, 12k, 8k, 4k, 2k, 1k, 512, 256, 64, and 16 trailing clean active-return examples.
- Methods: affine/cubic, log/arithmetic, shared/per-step, and static/online.
- Static variants fit once on the trailing calibration window. Online variants refit causally from the trailing realized active-return window at every evaluation event.

## Pooled MSE skill by window

Values are validation / test. Arithmetic-return variants are shown because log-return variants are effectively identical while the fits are stable (maximum validation-skill difference through the 4k window is 0.0027 percentage points).

| Window | Affine shared | Affine per-step | Online affine shared | Online affine per-step | Cubic shared | Cubic per-step | Online cubic shared | Online cubic per-step |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64k | 0.509% / 0.803% | 0.538% / 0.827% | 0.531% / 0.787% | **0.570% / 0.819%** | 0.495% / 0.791% | 0.534% / 0.849% | 0.529% / 0.793% | 0.569% / 0.846% |
| 32k | 0.508% / 0.789% | 0.535% / 0.808% | 0.523% / 0.786% | **0.557% / 0.828%** | 0.472% / 0.766% | 0.481% / 0.818% | 0.518% / 0.792% | 0.539% / 0.843% |
| 16k | 0.503% / 0.789% | 0.527% / 0.811% | 0.523% / 0.784% | 0.550% / 0.814% | 0.503% / 0.789% | 0.114% / 0.676% | 0.518% / 0.786% | 0.505% / 0.689% |
| 12k | 0.503% / 0.774% | 0.527% / 0.789% | 0.525% / 0.781% | **0.552% / 0.800%** | 0.499% / 0.770% | 0.104% / 0.645% | 0.520% / 0.779% | 0.421% / 0.677% |
| 8k | **0.518% / 0.788%** | **0.544% / 0.797%** | 0.522% / 0.779% | 0.543% / 0.798% | 0.488% / 0.762% | 0.017% / 0.609% | 0.517% / 0.780% | 0.385% / 0.552% |
| 4k | 0.518% / 0.776% | 0.529% / 0.747% | 0.507% / 0.772% | 0.515% / 0.752% | 0.441% / 0.720% | -0.347% / 0.306% | 0.500% / 0.763% | 0.241% / 0.020% |
| 2k | 0.461% / 0.583% | 0.402% / 0.475% | 0.471% / 0.750% | 0.437% / 0.709% | 0.452% / 0.575% | -55.629% / -14.951% | 0.457% / 0.748% | -0.654% / -0.540% |
| 1k | 0.314% / 0.269% | 0.120% / 0.082% | 0.381% / 0.615% | 0.229% / 0.487% | 0.154% / 0.199% | -432.649% / -62.759% | 0.350% / 0.590% | -48.916% / -3.159% |
| 512 | 0.398% / 0.685% | -0.002% / 0.179% | 0.168% / 0.406% | -0.107% / 0.158% | -0.215% / 0.359% | -5,111.815% / -854.464% | 0.107% / 0.349% | -283.717% / -10.568% |
| 256 | 0.207% / 0.248% | -1.510% / -2.089% | -0.156% / 0.073% | -0.736% / -0.534% | -0.095% / 0.186% | -1,463.148% / -264.907% | -0.288% / -0.095% | -737.017% / -1,114.062% |
| 64 | -2.350% / -5.764% | -3.756% / -7.610% | -2.214% / -2.156% | -4.835% / -4.938% | -7.464% / -7.906% | -716,786.971% / -97,437.836% | -3.631% / -5.577% | -10,114.592% / -15,352.490% |
| 16 | -0.734% / -2.222% | -137.048% / -138.655% | -9.112% / -9.564% | -37.324% / -37.053% | -5.724% / -4.811% | catastrophically unstable | -1,244.367% / -1,151.299% | catastrophically unstable |

## Findings

1. **The useful range extends through 64k:** online per-step affine continues improving in validation pooled MSE skill from 0.550% at 16k to 0.557% at 32k and 0.570% at 64k. Its validation/test pooled correlation also rises to 0.076085/0.091245 at 64k.
2. **Step-1 skill does not share the pooled optimum:** online per-step affine still has its best step-1 validation MSE skill at 12k (6.340% validation / 7.041% test). The 32k and 64k values are 6.315%/6.995% and 6.275%/7.023%, respectively.
3. **Shared affine is the safest low-data method:** it remains positive at every window down to 256, though performance becomes noisy below 2k.
4. **Per-step affine needs more data:** it helps at 4k and above, but loses to shared affine by 2k and becomes harmful below roughly 512–256.
5. **Cubic needs the larger windows:** per-step cubic is competitive at 32k–64k, but was already worse at 16k and becomes numerically catastrophic as the window shrinks.
6. **Online refitting changes the failure boundary and benefits from long history:** online affine remains useful around 1k–2k, while its best pooled result is now at 64k.
7. **Log versus arithmetic domain is immaterial at practical windows:** 1-second returns are small enough that the two transforms agree closely. Their divergence at tiny windows is a symptom of unstable high-order fitting, not useful domain information.

## Recommendation

Use **online per-step affine with a 64k window** for the best pooled path metrics. If step-1 MSE skill is the primary objective, retain the **12k online per-step affine** fit. Prefer **shared affine** if the available clean calibration history may fall below about 4k. Do not use per-step cubic at short windows without much stronger regularization or hierarchical shrinkage.

## Joint matrix-affine calibration

A later extension fits the complete expected-return vector jointly:

\[
\hat{\mathbf y}_{cal}=A\hat{\mathbf y}+\mathbf b,
\qquad A\in\mathbb R^{15\times15}.
\]

This has 240 fitted coefficients instead of 30 for independent per-step affine fits. A ridge of \(10^{-3}\) is applied after independently standardizing every input and output step. The online form refits only from forecast paths whose complete 15-return target has already resolved.

Results below use the best-validation-correlation checkpoint and log-return calibration; arithmetic-return results are effectively identical.

| Window | Static matrix MSE skill | Online matrix MSE skill | Online matrix correlation | Online matrix step-1 MSE skill |
|---:|---:|---:|---:|---:|
| 16k | -0.711% / -0.813% | 0.378% / 0.598% | 0.067469 / 0.080010 | 6.213% / 6.900% |
| 32k | 0.259% / 0.464% | 0.477% / 0.776% | 0.072191 / 0.088017 | 6.223% / 6.950% |
| 64k | 0.422% / 0.739% | **0.546% / 0.831%** | **0.075674 / 0.091350** | **6.231% / 7.000%** |

The matrix transform clearly needs the longer window, but it still loses on validation to 64k online per-step affine: 0.546% versus 0.570% pooled MSE skill and 0.075674 versus 0.076085 correlation. Its small descriptive test advantage is not sufficient to select it after the validation comparison. Cross-step mixing therefore adds variance without a demonstrated validation benefit at the tested ridge.

All three preserved checkpoint criteria were backfilled. The dashboard exposes the window selector beside the calibration-method selector.
