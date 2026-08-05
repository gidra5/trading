# Cumulative-return MSE skill audit

Date: 2026-08-05

## Definition and correction

Every number in this audit uses the same definition:

`100 * (1 - model cumulative MSE / zero cumulative MSE)`

The cumulative return is the compounded simple return used by the training code:

`expm1(sum(log returns))`

An earlier calculation for the five-day model incorrectly used `sum(log returns)` in the zero-baseline denominator while comparing it with the model's compounded-return MSE. That produced a false `+4.03%` daily test result. The correct five-day test skill is **-1.3774%**.

## One-second candle resolution

These are validation results unless a test value is explicitly shown.

| Horizon | Model | Validation cumulative skill | Test cumulative skill |
| ---: | --- | ---: | ---: |
| 1s | Linear Ridge | +0.3241% | +1.0990% |
| 1s | One-layer GLU | +0.7200% | +1.9796% |
| 2s | Linear, candle + cumulative loss | +0.3841% | sealed |
| 2s | One-layer GLU, long run | **+0.9986%** | sealed |
| 3s | Linear, candle + cumulative loss | +0.3372% | sealed |
| 3s | One-layer GLU | **+0.8512%** | sealed |
| 5s | Linear, candle + cumulative loss | +0.2419% | sealed |
| 5s | One-layer GLU | +0.7431% | sealed |
| 5s | Two-layer GLU | **+0.9152%** | sealed |
| 5s | Three-layer GLU | +0.8987% | sealed |
| 15s | Equal-summary one-layer GLU | +0.3629% | +1.0218% |
| 15s | 2x-cumulative one-layer GLU | **+0.3999%** | +0.7928% |
| 15s | 2x-cumulative linear model | +0.0285% | +0.5492% |

The three 15-second test values use different, deliberately disjoint test blocks. They are evidence that each beat zero on its own cumulative target, but are not a direct ranking of the models on identical examples.

## Five-output 1m candle models

Validation only. Each model consumes 120 one-minute positions and predicts five one-minute returns.

| Maximum MA | Cumulative skill |
| --- | ---: |
| Raw only | -0.0442% |
| **Through 1h** | **-0.0313%** |
| Through 1d | -0.0387% |
| Through 1w | -0.0492% |
| Through 1M | -0.0570% |
| Through 3M | -0.0698% |

None beats zero cumulative return.

## Five-output 1h candle models

Validation only. Each model consumes 120 one-hour positions and predicts five one-hour returns.

| Maximum MA | Cumulative skill |
| --- | ---: |
| Raw only | -0.3330% |
| Through 1d | -0.3435% |
| Through 1w | -0.2966% |
| **Through 1M** | **-0.1872%** |
| Through 3M | -0.4909% |

None beats zero cumulative return.

## Five-output 1d candle models

| Maximum MA | Validation cumulative skill | Test cumulative skill |
| --- | ---: | ---: |
| Raw only | +1.7453% | not evaluated |
| **Through 1w** | **+2.0308%** | **-1.3774%** |
| Through 1M | +1.6138% | not evaluated |
| Through 3M | +0.5128% | not evaluated |

Only the validation-selected one-week model was opened on test. Its cumulative validation edge did not generalize.

## One-step 1m models

For one-output models, cumulative skill equals ordinary return MSE skill.

| Model | Validation skill | Test skill |
| --- | ---: | ---: |
| Linear Ridge | +0.0283% | -0.0312% |
| 16-layer GLU | +0.0107% | -0.0139% |
| One-layer GLU | +0.0433% | -0.1190% |

## SearchCast-style one-step Ridge models

For a one-step level forecast, subtracting the known current log price makes level error and return error identical. These are sealed-test skills against persistence/zero return from the earlier SearchCast-style BTC study.

| Candle resolution | Test cumulative skill |
| --- | ---: |
| 1s | -2.8332% |
| 1m | -4.3454% |
| 1h | -3.0784% |
| 1d | -1.7265% |

On the exact 362-day subset used by the five-output daily GLU, the SearchCast-style daily Ridge scores -1.6620%, versus -0.1009% for the GLU's first daily lead.

## Overall pattern

- At one-second resolution, cumulative validation skill is consistently positive and peaks around 1% for the best 2-5 second GLUs.
- The available one-second tests at horizons 1 and 15 are also positive, although the 15-second tests use different market blocks.
- Five-candle cumulative prediction becomes negative at 1-minute and 1-hour resolutions.
- The five-day model has positive validation cumulative skill but negative sealed-test skill.
- SearchCast-style one-step Ridge models are negative at every BTC resolution tested.

Normalized cumulative MSE values from older reports are not themselves skills. This audit recomputes the zero baseline from the exact target examples and uses compounded returns on both sides.
