# Direct knot price distributions from latent NSF

The latent NSF now has an additional learned price-distribution head. Given
observed history and a horizon in seconds, it outputs the same representation
as our Gaussian-coordinate knot forecasts: a probability of exactly zero log
return and normalized triangular-knot masses for the remaining distribution.
Generating these parameters needs no latent Monte Carlo samples.

## Construction

`ml/direct_latent_nsf.py` retains the 256-dimensional latent NSF and reads its
complete causal Gaussian posterior parameters, both means and standard
deviations. The direct head also receives the existing three horizon features
`[h/21600, tanh(h/21600), log1p(h)]`, plus daily and weekly clock features at the
origin and endpoint. It uses a two-layer width-256 normalized GLU head and a
train-fitted knot basis. This experiment has 255 distinct knots after
deduplication.

For horizon `h > 0`, the active log-return density is

\[
f_h(r) = \frac{1}{\sigma\sqrt h}\,
  \phi\!\left(\frac{r}{\sigma\sqrt h}\right)
  \sum_i \frac{m_i}{A_i}
       b_i\!\left(\Phi\!\left(\frac{r}{\sigma\sqrt h}\right)\right).
\]

Here `m_i` are conditional component masses, `A_i` are triangular basis areas,
and `b_i` are the unit-height triangular basis functions. The complete law is
`p0 * delta(0) + (1-p0) * f_h`. The square-root horizon scale is a coordinate
choice; it does not assume independent or Brownian market increments. The
weights themselves depend on horizon and history.

Queries accept scalar or per-row horizons, including fractional seconds in
`[0,86400]`. At zero horizon the law is exactly the known current price.
Float64 prices and Unix timestamps preserve tick and clock resolution.

`ml/knot_price_distribution.py` exposes mixed likelihood, continuous and
tick-projected CDFs, tick probabilities, inverse-CDF sampling, and price/log
return moments. CDFs integrate the piecewise-linear density analytically;
moments use explicit quadrature. The distribution includes its Gaussian tails.
Its full parameters also fit `knot_oracle_price_law.knot_price_law`; consumers
need not reduce it to a mean or a finite set of price representatives.

## Relationship to the stochastic flow

This head is an approximate learned marginal, not an exact marginalization
of the latent transition. A zero atom is also a new representational ability:
the original latent decoder only obtains unchanged prices through tick
rounding. Training a direct head does not inherit an exact flow-consistency
theorem from the latent NSF.

The initial experiment freezes the previously selected
`oracle-latent-nsf-sampled-skips/latentSampledSkipFlow.pt`. It trains the head on
observed tick negative log likelihood and a CDF projection penalty against
the frozen NSF's sampled price law. The penalty uses 64 latent components,
34 price boundaries and 2,048 training origin/horizon pairs. It encourages
agreement but does not certify uniform agreement or exact time consistency.

The standalone checkpoint contains both the direct head and all original NSF
weights. Exact equality of the retained NSF weights is checked after saving
and loading. The implementation permits differentiation through the encoder,
but the reported experiment deliberately fits only the added head.

Hypothetical full market states still come from `model.nsf.transition` and
`model.nsf.emission`. A direct price marginal alone does not determine their
joint law or the next posterior. The existing planning/backtest forecast route
has not been silently replaced with this separately fitted marginal.

## Training and use

Training uses the established, separated train/validation origins and immutable
price history. Inputs read only the previous and current observations. Knots
and feature normalization are fitted on training data only. Half the horizon
draws are uniform integers from 1 through 86,400 seconds; the other half are
uniform over `1,5,15,60,300,900,3600,7200,21600,86400`. This provides explicit
short-horizon coverage without the earlier tanh-derivative sampling rule.

Checkpoint selection minimizes tick NLL on 4,096 fixed validation pairs.
Final comparisons use held-out test observations, with no test checkpoint
selection. This historical test was inspected in previous experiments, and
long-horizon labels overlap, so it is not a new independent performance trial.

```python
from pathlib import Path
import torch
from direct_latent_nsf import load_direct_latent_nsf

model, metadata = load_direct_latent_nsf(
    Path('data/backtests/oracle-direct-latent-nsf-20260921-projection100/model.pt'),
    torch.device('cuda'))
with torch.inference_mode():
    # features [B,T,59]; prices, elapsed, timestamps [B,T].
    # Prices and Unix timestamps are float64; elapsed and horizon are seconds.
    law = model.distribution(features, prices, elapsed, 1234.5, timestamps)
    zero, conditional_masses = law.zero_probability, law.masses
    probability_at_current_tick = law.tick_probability(prices[:, -1])
    cdf_at_price = law.cdf(query_prices)
    expected_log_return = law.mean_log_return()
```

Training and evaluation entry points are `ml/train_direct_latent_nsf.py` and
`ml/evaluate_direct_latent_nsf.py`. They refuse to overwrite completed output
directories, record source/checkpoint hashes, and check preservation of the
parent dynamics.

## Verification and results

Results are recorded after the training, reload and matched evaluation finish.
