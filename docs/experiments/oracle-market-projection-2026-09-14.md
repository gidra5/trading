# Matching retained market samples by their conditional forecasts

The dense planner's current rolling map matches new hypothetical observation pairs to retained market nodes using standardized features and log prices relative to one common price anchor. The NSF encoder instead receives each pair's prices relative to that pair's final price. A large difference in absolute price level can therefore dominate the matching metric without implying an equally large difference in the model's conditional return law. Exact account execution still uses the actual sampled price, and cached conditional moves are reanchored there.

`ml/audit_oracle_market_projection.py` compares three matching rules on a fixed archived market cloud:

1. **Absolute:** the current common-price-anchor metric.
2. **Relative:** the same standardized features, with prices normalized within each observation pair.
3. **Posterior:** Euclidean distance between the encoder's concatenated latent means and standard deviations. For diagonal Gaussian posteriors, this is their 2-Wasserstein distance: coupling both distributions with the same independent standard-normal coordinates gives squared distance `sum((mu1-mu2)^2 + (std1-std2)^2)`, attaining the coordinatewise covariance bound.

All comparisons use the same trained latent NSF checkpoint and the same generated hypothetical states. Each selected retained observation pair is reanchored at the actual hypothetical mark before forecasting. Common random numbers couple the predictions from the actual and selected states. This compares conditional **model laws**, not predictions with realized future returns, trading value, or GPU-generated samples.

## Repeated comparison

`data/backtests/oracle-market-projection-replicated` covers ten observed offsets (1, 30, 120, 300, 600, 1200, 1800, 2400, 3000 and 3500 seconds), eight independent sample replicates per offset, eight hypothetical next states per replicate, and 128 conditional price samples per state. Samples are equally weighted. The table averages the empirical Wasserstein distance of log-return samples, expressed in basis points, over 80 offset/replicate groups.

| Model query horizon | Absolute | Relative | Posterior |
| --- | ---: | ---: | ---: |
| 1 second | 0.01844 bps | 0.01925 bps | 0.01548 bps |
| 60 seconds | 0.11384 bps | 0.10230 bps | 0.03796 bps |
| 3,600 seconds | 1.79320 bps | 1.60258 bps | 0.56442 bps |

Posterior matching reduced the offset-averaged distribution difference at 9/10 offsets for the one-second query and 10/10 offsets for the longer queries. These are ten distinct observed offsets, not 80 independent market histories. The complete CPU-only experiment took 45.84 seconds and did not change the concurrent GPU backtest.

The result is mixed for the conditional mean, which matters directly to trading:

| Model query horizon | Absolute mean error | Relative mean error | Posterior mean error |
| --- | ---: | ---: | ---: |
| 1 second | 0.00914 bps | 0.01137 bps | 0.01254 bps |
| 60 seconds | 0.03976 bps | 0.03460 bps | 0.01549 bps |
| 3,600 seconds | 1.46017 bps | 1.25786 bps | 0.41676 bps |

These are mean absolute differences between conditional model means. Posterior matching improves the longer-horizon means but worsens this one-second mean measure. It is a candidate for a subsequent value/backtest comparison, not an established improvement to the trading policy.

The 3,600-second query is the NSF's direct horizon-conditioned forecast from the compared states. It is not a simulation of 3,600 sequential one-second decisions and does not certify the one-hour Bellman approximation. Likewise, a large geometric price-projection diagnostic alone is not a forecast or value error bound.

Three focused tests pass: invariance of the relative coordinates to independent common rescaling within each price pair, a case where relative matching recovers the identical return at a different price level, and permutation invariance of the empirical distribution distance. The running replay retains its original matching rule until this alternative is evaluated at the policy-value level.

```powershell
.venv-ml/Scripts/python.exe ml/audit_oracle_market_projection.py --source-map data/backtests/oracle-dense-repair-terminal-90m/initial --output data/backtests/oracle-market-projection-replicated-new --offsets 1 30 120 300 600 1200 1800 2400 3000 3500 --replicates 8 --branches 128 --horizons 1 60 3600
```

## Effect on root action values

`data/backtests/oracle-projection-value-audit` recomputes one complete Bellman layer for each matching rule using the same fresh conditional root samples. The archived continuation field, future candidate schedule and original terminal date are held fixed. It tests seven observed offsets, two sample replicates and four account configurations: cash, long inventory with a buy order, short inventory with a sell order, and two opposing orders. All accounts have $10,000 marked equity.

| Matching rule | Changed actions versus absolute | Maximum absolute shift in best log value | Mean absolute shift |
| --- | ---: | ---: | ---: |
| Relative | 0 / 56 | 0.05765 bps | 0.00594 bps |
| Posterior | 0 / 56 | 0.07361 bps | 0.00834 bps |

All selected post-account states matched exactly. Scoring each selected action under the absolute-metric field also gave zero decision gap. This latter comparison is a sensitivity check, not regret against a true optimum. The experiment took 5.32 seconds on CPU, excluding Python startup/import time.

The original terminal date means the remaining horizon decreases with the queried offset; this is not a rolling one-hour replay. Root searches use the finite catalog and retained-order subsets. The result does not cover every continuous action, market state or future map rebuild. It provides no evidence that replacing the current matching rule would improve the trading decisions in these cases. Exposure refinement remains the better-supported implementation priority, while the latent-posterior metric remains a candidate for broader comparisons.

```powershell
.venv-ml/Scripts/python.exe ml/audit_oracle_projection_values.py --map data/backtests/oracle-dense-repair-terminal-90m/initial --output data/backtests/oracle-projection-value-audit-new
```
