# Two-, three-, and five-second return-path screen

Date: 2026-08-04

## Question

Compare a linear model and the one-layer normalized GLU when predicting the
next `T` BTCUSDT one-second log returns from the preceding 120 one-second log
returns, for `T = 2`, `3`, and `5`.

Every run uses the cumulative-weighted path objective:

`candle normalized MSE + (mean + variance + minimum + maximum + 2 * cumulative return) / 6`

The cumulative return is `expm1(sum(log returns))`. Every candle lead and path
summary is normalized using statistics from the training split only.

## Training optimization

The runner was optimized before this screen:

- Batches now span daily shards instead of emitting a short batch at every
  shard boundary.
- Rows remain contiguous inside each shuffled daily shard, avoiding expensive
  random advanced indexing.
- The training batch increased from 65,536 to 262,144 examples, reducing a
  full training epoch from 507 optimizer steps to 127.
- One reusable host batch buffer prevents allocation growth across epochs.
- CPU window preparation and transfer-stream copies overlap the previously
  queued GPU step without retaining unbounded pinned staging buffers.
- Per-batch scalar reads of loss and gradient norm were removed because they
  forced CUDA synchronization. Non-finite training is still rejected by the
  full validation result at the end of each epoch.
- Metric accumulation performs direct reductions and omits per-lead training
  metrics, while validation and test retain the complete metric set.
- A GLU reuses the matching linear run's train-only normalization, avoiding a
  second pass over 33.2 million training examples.

After warm-up, the GLUs reached 98–100% instantaneous GPU utilization on the
RTX 3070. The final 5-second GLU median steady epoch was 23.38 seconds versus
about 32.8 seconds before optimization. The 5-second linear median steady epoch
was 8.05 seconds versus about 19 seconds before optimization. The linear model
is too small to saturate a GPU; its speedup comes from fewer batches and less
data-pipeline overhead.

## Dataset and model sizes

| Horizon | Train | Validation | Test | Embargo | Linear parameters | GLU parameters |
|---:|---:|---:|---:|---:|---:|---:|
| 2s | 33,219,946 | 15,505,750 | 15,000 | 122s | 242 | 1,174,532 |
| 3s | 33,219,939 | 15,505,725 | 15,000 | 123s | 363 | 1,175,045 |
| 5s | 33,219,925 | 15,505,675 | 15,000 | 125s | 605 | 1,176,071 |

The GLU has one fused width-512 layer, learned value/gate metric matrices, and
one `T`-return output head. The linear model has exactly `120 * T + T`
parameters.

The six sealed test slices are disjoint 15,000-example chronological blocks.
Their source-test tail offsets are 2,500,000, 2,515,000, 2,530,000, 2,545,000,
2,560,000, and 2,575,000 examples for linear-5s, GLU-5s, linear-3s, GLU-3s,
linear-2s, and GLU-2s respectively. Test metrics therefore describe different
market periods and are not paired architecture comparisons.

## Shared-validation results

Within each horizon, linear and GLU validation metrics cover the same rows and
are the appropriate architecture comparison.

| Horizon | Model | Best epoch | Composite objective | Candle MSE skill vs zero | Direction accuracy | Correlation | Cumulative normalized MSE |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5s | Linear | 10 | 1.896057 | -3.3238% | 51.5655% | 0.01424 | 1.005390 |
| 5s | GLU | 15 | **1.877620** | -3.7486% | **52.3233%** | **0.02338** | **1.000226** |
| 3s | Linear | 4 | 1.884950 | -2.1022% | 50.0794% | 0.02043 | 1.004988 |
| 3s | GLU | 15 | **1.871830** | -2.1482% | **52.1584%** | **0.03492** | **0.999243** |
| 2s | Linear | 11 | 1.877556 | -0.7217% | 50.8352% | 0.02856 | 1.005189 |
| 2s | GLU | 15 | **1.867337** | **-0.7188%** | **51.0388%** | **0.04775** | **1.000030** |

The GLU improves the requested composite objective at every horizon: 0.97% at
5s, 0.70% at 3s, and 0.54% at 2s relative to the matching linear objective.
It also has higher validation correlation and direction accuracy in all three
comparisons.

However, every validation candle MSE skill remains negative. Even the best
case, the 2-second GLU, is 0.719% worse than always predicting zero returns.
The GLU's composite improvement mostly comes from matching the min/max path
summaries and bringing mean/cumulative normalized error close to 1. It does not
represent accurate prediction of the entire ordered return path.

## Fresh disjoint test results

| Horizon | Model | Composite objective | Candle MSE skill vs zero | Direction accuracy | Correlation | Cumulative normalized MSE |
|---:|---|---:|---:|---:|---:|---:|
| 5s | Linear | 2.543230 | -3.3491% | 51.5107% | 0.01470 | 1.375876 |
| 5s | GLU | 0.712111 | -1.9201% | 53.4853% | 0.06636 | 0.446603 |
| 3s | Linear | 0.440810 | -2.1660% | 50.0378% | 0.02486 | 0.237355 |
| 3s | GLU | 0.162852 | **+0.2250%** | 51.4533% | **0.10847** | 0.089106 |
| 2s | Linear | 0.149092 | -0.4288% | 50.6333% | 0.04110 | 0.080280 |
| 2s | GLU | 0.155840 | -0.2324% | 49.3800% | 0.08230 | 0.085347 |

Only the 3-second GLU beats the zero-return MSE baseline on its test block, by
0.225%. It also has the strongest test correlation, 0.1085. This is an
interesting lead, not a confirmed winner: the test block contains only 15,000
seconds, is distinct from every other model's block, and has much lower return
variance than the long train/validation corpus.

The numerically small test objectives and cumulative normalized errors are
caused partly by this distribution shift and training-derived normalization.
They must not be read as percentage skill or compared directly across the
different test periods.

## Test MSE skill by lead

| Horizon/model | +1s | +2s | +3s | +4s | +5s |
|---|---:|---:|---:|---:|---:|
| 5s linear | -3.592% | -3.669% | -4.442% | -0.028% | -5.013% |
| 5s GLU | **+1.110%** | **+0.664%** | -5.894% | **+0.120%** | -5.597% |
| 3s linear | -1.223% | -2.543% | -2.732% | — | — |
| 3s GLU | **+0.895%** | **+0.291%** | -0.511% | — | — |
| 2s linear | -0.115% | -0.742% | — | — | — |
| 2s GLU | -0.270% | -0.195% | — | — | — |

The 5-second GLU has positive test skill at leads 1, 2, and 4 but loses more at
leads 3 and 5, making its aggregate skill negative. The 3-second GLU is the
cleanest test path: positive first and second leads and only a small loss at
the third lead. Validation remains the guard against over-interpreting these
short disjoint test periods.

## Conclusion

Shortening the horizon reduces the validation candle-MSE penalty: aggregate
skill improves from roughly -3% to -4% at 5s, to about -2.1% at 3s, to about
-0.72% at 2s. The nonlinear GLU consistently improves correlation and the
composite path-summary objective, but approximately 1.17 million parameters do
not yet produce positive validation MSE skill over the 242–605 parameter
linear baselines.

The most defensible next experiment is a paired validation screen around the
2–3 second range with a weaker summary term or prefix-cumulative losses. The
already-opened six test blocks should not be reused for choosing that loss.

## Artifacts

Plans:

- `ml/training-plans/horizon-screen-linear-5s-v1.json`
- `ml/training-plans/horizon-screen-glu-5s-v1.json`
- `ml/training-plans/horizon-screen-linear-3s-v1.json`
- `ml/training-plans/horizon-screen-glu-3s-v1.json`
- `ml/training-plans/horizon-screen-linear-2s-v1.json`
- `ml/training-plans/horizon-screen-glu-2s-v1.json`

Results:

- `data/training/runs/horizon-screen-linear-5s-v1/state/result.json`
- `data/training/runs/horizon-screen-glu-5s-v1/state/result.json`
- `data/training/runs/horizon-screen-linear-3s-v1/state/result.json`
- `data/training/runs/horizon-screen-glu-3s-v1/state/result.json`
- `data/training/runs/horizon-screen-linear-2s-v1/state/result.json`
- `data/training/runs/horizon-screen-glu-2s-v1/state/result.json`

