# Latent NSF with skip targets sampled across training sequences

Implemented the requested sequence-wide skip-target sampling and retrained the 256-dimensional calendar-aware latent NSF. This retains raw observed feature inputs, all three horizon features, advancing UTC clock features, uniform integer horizons 1..86400, and the probabilistic price decoder. There is no direct59 prediction-summary input.

## Objective and sequence construction

The paper's latent objective adds a skip KL that compares a directly predicted future latent law with the posterior inferred from observations through that future time. Targets are sampled among nonadjacent observations. See [the authors' latent NSF objective](https://nkiyohara.github.io/nsf-neurips2025/).

The previous implementation used just the designated origin-to-endpoint skip in a four-observation window. This run uses eight observations at offsets

`[-1, 0, floor(h/6), floor(2h/6), ..., floor(5h/6), h]`.

The forecast-origin posterior remains at index 1 and uses only the first two observations. Later observations supply training targets, not information for the current forecast.

For each distinct source time `i` with at least two later distinct times, independently sample one target `j` uniformly among the next two to four distinct observation intervals. The target set is truncated at the end of the sequence. With eight distinct observations, sources 0 through 5 participate:

| Source index | Possible target indices |
|---|---|
| 0 | 2, 3, 4 |
| 1 | 3, 4, 5 |
| 2 | 4, 5, 6 |
| 3 | 5, 6, 7 |
| 4 | 6, 7 |
| 5 | 7 |

Duplicate timestamps for very short horizons are padding: they neither update the encoder nor create extra skip candidates. Nonadjacency is measured in distinct observations, not padded row indices. Actual transition durations are timestamp differences in seconds, and the transition receives each source's advancing calendar time.

A sampled term is `log q(z_j | observations through j) - log K(z_j | z_i, t_i, t_j-t_i)`, evaluated at samples from the causal source and target posteriors. Both encoder and transition remain differentiable. We average sampled terms over eligible sources within each sequence and then over the batch. This estimates the corresponding mean of per-source expectations, not a uniform average over all possible pairs. The unit tests compare the sampling expectation against exact enumeration.

The maximum skip span of four observation intervals, eight-point window construction, boundary truncation, and source averaging are explicit local choices. This follows the paper's sampled-skip mechanism, not a claim to reproduce its benchmark configuration exactly. In particular, the paper writes a sum over source positions; our averaging controls loss scale as the number of valid positions changes.

## Comparison protocol

Two arms are trained from the same initialization using the same origin/horizon draws and eight-observation windows:

- `latentAdjacent`: reconstruction plus adjacent KL, weight 0.1; no skip or flow loss.
- `latentSampledSkipFlow`: additionally sampled skip KL, weight 0.1, and bidirectional flow loss, weight 0.01.

The adjacent-only arm is a matched control for the changed window length. Comparing directly against the previous four-point experiment changes both sequence construction and skip supervision. The second arm adds skip and flow together, so their separate effects are not identified.

The model has a 256-dimensional stochastic latent state, 256-wide networks, and 1,741,432 forecast parameters. All feature normalizers use training data. The same 103,693 training origins and 14,760 validation origins have complete 24-hour endpoints inside their own splits.

Each epoch draws 32,768 windows, batch size 1024, AdamW learning rate 0.0003. Maximum epochs 32, minimum 16, patience 10. Selection uses the same fixed 2048 uniform origin/horizon validation pairs as the preceding calendar-aware experiment and minimizes forecast CRPS(log return)/sqrt(h). Selected checkpoints freeze before test evaluation.

Evaluation uses 256 latent components and integrates the decoder's Gaussian law. MSE skill is relative to the zero-log-return mean; reported CRPS uses 512 observation draws. The old historical test is already inspected, and long-horizon targets overlap. No online calibration, trading backtest, or deployment is included.

## Results

Both arms selected epoch 4 and stopped after epoch 16, each seeing 524,288 training windows. Training took 236.5 seconds. Initial validation scaled CRPS was 0.390682 bps. Selected scores were 0.359251 for adjacent-only and 0.356479 for sampled skip+flow. The validation origin/horizon arrays exactly match the previous calendar-aware run; its original single-skip+flow selected score was 0.365468.

Test MSE skill percentages, computed from decoder-integrated mean log return:

| Horizon | Previous single skip + flow, 4 observations | Sampled skip + flow, 8 observations | Adjacent control, 8 observations |
|---|---:|---:|---:|
| 1 second | -12.9236 | 0.1765 | -34.8037 |
| 5 minutes | 0.1913 | 0.3001 | -0.2309 |
| 15 minutes | 0.2262 | 1.4993 | 0.7751 |
| 2 hours | -0.7415 | 5.3148 | 9.9003 |
| 6 hours | -3.0400 | 0.7171 | -0.7670 |
| 24 hours | -53.8889 | -79.2213 | -112.7564 |

Test CRPS in log-return bps, lower is better:

| Horizon | Previous single skip + flow | Sampled skip + flow | Adjacent control |
|---|---:|---:|---:|
| 1 second | 0.1705 | 0.1461 | 0.1795 |
| 5 minutes | 4.7642 | 4.7492 | 4.7742 |
| 15 minutes | 8.0414 | 8.0046 | 8.0284 |
| 2 hours | 23.5034 | 22.9296 | 22.3414 |
| 6 hours | 51.0439 | 49.9419 | 50.9535 |
| 24 hours | 102.8405 | 115.3987 | 168.7797 |

The new sampled-skip configuration improves over the previous single-skip configuration through six hours, but the 24-hour prediction remains poor and worsens. Against the matched eight-observation control, sampled skip+flow improves every reported horizon except two hours. This does not establish an overall winner: the adjacent control reaches 9.90% two-hour skill, while the previous four-observation adjacent model had 9.41%. The retained direct59 predictor remains much stronger at one second (5.19% on the matched strided sample).

There are 4517 test origins at one second and 2042 at every longer horizon. These are historical, previously inspected data with overlapping targets and one training seed. No statistical significance or real-market profitability is established. The change is retained as the active sampled-skip implementation, but no existing best deployed predictor is replaced.

The 512-component numerical cross-check on 128 origins differed from the 256-component integrated mean by at most 0.053 bps RMS across these two arms and horizons. This observed integration discrepancy is far smaller than the 24-hour forecast error.

## Verification

Eleven latent NSF tests pass. Three added checks verify uniform target frequencies for each eligible source against explicit enumeration, rejection of duplicate/adjacent timestamps, and finite differentiable losses for eight-observation sequences. Existing tests cover causal encoding, advancing clock conditioning, zero-duration identity, inverse flow Jacobians, decoder uncertainty, and arbitrary-horizon queries.

Selected-checkpoint audits verify training source hashes and checkpoint hashes. Changing all future teacher observations leaves both current-posterior parameters bit-identical. A 1234.5-second forecast retains both latent variation and decoder conditional variance, and changing the clock changes predictions. Sampling audits cover horizons 1, 2, 3, 6, 7200, and 86400 seconds and confirm unique eligible sources, strictly later targets, and two-to-four distinct observation intervals per skip.

## Reproduction

```powershell
.venv-ml/Scripts/python.exe -m unittest discover -s ml -p test_latent_neural_stochastic_flow.py
.venv-ml/Scripts/python.exe ml/train_latent_neural_stochastic_flow.py
.venv-ml/Scripts/python.exe ml/evaluate_latent_neural_stochastic_flow.py
.venv-ml/Scripts/python.exe ml/audit_latent_neural_stochastic_flow.py
```

Outputs are in `data/backtests/oracle-latent-nsf-sampled-skips`. Training refuses to overwrite an existing run. The preceding calendar-aware experiment's sources are archived under its own `source` directory, and this run retains its own source snapshot separately.
