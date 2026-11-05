"""Direct price marginals on a causal NSF posterior, with its dynamics retained.

The head predicts a zero atom and a Gaussian-coordinate triangular-knot law.
It integrates no latent samples at inference. It is an additional learned
marginal, not an exact analytic marginalization of the latent transition.
"""
from __future__ import annotations

from pathlib import Path
import torch
from torch import nn

from horizon_price_forecast import HorizonPriceForecast
from latent_neural_stochastic_flow import LatentNSF, transition_time_features
from knot_price_distribution import KnotPriceDistribution
from return_knot_density import triangular_basis_areas

CONTRACT = 'latent-nsf-direct-zero-gaussian-knot-price-v1'


class DirectLatentNSF(nn.Module):
    def __init__(self, nsf: LatentNSF, basis: dict, input_mean: torch.Tensor,
                 input_std: torch.Tensor, widths=(256, 256), maximum_seconds=86400.):
        super().__init__()
        if not 0 < maximum_seconds <= 86400:
            raise ValueError('maximum horizon must be in (0,86400]')
        if input_mean.shape != (2*nsf.dimensions+11,):
            raise ValueError('normalization must cover posterior and time features')
        self.nsf = nsf
        self.maximum_seconds = float(maximum_seconds)
        self.head = HorizonPriceForecast(input_mean, input_std, basis, tuple(widths))

    def posterior(self, features, prices, elapsed, timestamps):
        if features.ndim != 3 or features.shape[-1] != 59 or not features.shape[1]:
            raise ValueError('features must be nonempty [batch,history,59]')
        shape = features.shape[:2]
        if prices.shape != shape or elapsed.shape != shape or timestamps.shape != shape:
            raise ValueError('price, elapsed and timestamp histories must match features')
        if timestamps.dtype != torch.float64 or prices.dtype != torch.float64:
            raise ValueError('prices and Unix timestamps must be float64')
        if not bool(torch.isfinite(features).all() & torch.isfinite(prices).all()
                    & (prices > 0).all() & torch.isfinite(elapsed).all()
                    & (elapsed >= 0).all() & torch.isfinite(timestamps).all()):
            raise ValueError('invalid observed history')
        if not torch.allclose(timestamps[:, 1:]-timestamps[:, :-1], elapsed[:, 1:].double(),
                              atol=1e-6, rtol=0):
            raise ValueError('timestamps and elapsed seconds disagree')
        # Re-anchor at the known price, preserving the NSF training convention.
        obs = self.nsf.observations(features, (prices/prices[:, -1:]).log())
        means, stds = self.nsf.encode(obs, elapsed, timestamps)
        return means[:, -1], stds[:, -1]

    def horizons(self, horizon_seconds, reference):
        h = torch.as_tensor(horizon_seconds, device=reference.device, dtype=reference.dtype)
        if h.ndim == 0:
            h = h.expand(len(reference))
        if h.shape != (len(reference),) or not bool((torch.isfinite(h) & (h >= 0)
                                                    & (h <= self.maximum_seconds)).all()):
            raise ValueError('horizon must be scalar or [batch] finite seconds in range')
        return h

    def conditioning(self, means, stds, horizon_seconds, timestamps_seconds):
        if means.ndim != 2 or means.shape[-1] != self.nsf.dimensions or stds.shape != means.shape:
            raise ValueError('posterior must contain [batch,latent] means and scales')
        if not bool(torch.isfinite(means).all() & torch.isfinite(stds).all() & (stds >= 0).all()):
            raise ValueError('invalid posterior parameters')
        if timestamps_seconds.shape != (len(means),) or timestamps_seconds.dtype != torch.float64 \
                or not bool(torch.isfinite(timestamps_seconds).all()):
            raise ValueError('float64 Unix forecast-origin timestamps required')
        h = self.horizons(horizon_seconds, means)
        # The complete Gaussian posterior parameters are inputs; taking only
        # its mean would discard the encoder's uncertainty.
        time = transition_time_features(h/self.nsf.time_scale,
                                       timestamps_seconds/self.nsf.time_scale, means.dtype)
        return torch.cat((means, stds, time), -1)

    def logits_from_posterior(self, means, stds, horizon_seconds, timestamps_seconds):
        return self.head(self.conditioning(means, stds, horizon_seconds, timestamps_seconds))

    def forward(self, features, prices, elapsed, horizon_seconds, timestamps_seconds):
        means, stds = self.posterior(features, prices, elapsed, timestamps_seconds)
        return self.logits_from_posterior(means, stds, horizon_seconds, timestamps_seconds[:, -1])

    def law_from_logits(self, logits, current_price, horizon_seconds):
        h = self.horizons(horizon_seconds, logits).double()
        raw = logits.double()
        # Normalize in float64 so the CDF and likelihood retain narrow tick mass.
        knots = self.head.knots.double()
        areas = triangular_basis_areas(knots)
        masses = (raw[:, 1:]+areas.log()).softmax(-1)
        zero = torch.where(h == 0, torch.ones_like(h), raw[:, 0].sigmoid())
        return KnotPriceDistribution(current_price.double(), zero, masses,
                                     knots, areas,
                                     self.head.sigma*h.sqrt())

    def distribution_from_posterior(self, means, stds, current_price, horizon_seconds, timestamps_seconds):
        logits = self.logits_from_posterior(means, stds, horizon_seconds, timestamps_seconds)
        return self.law_from_logits(logits, current_price, horizon_seconds)

    def distribution(self, features, prices, elapsed, horizon_seconds, timestamps_seconds):
        logits = self(features, prices, elapsed, horizon_seconds, timestamps_seconds)
        return self.law_from_logits(logits, prices[:, -1], horizon_seconds)

    def probabilities(self, features, prices, elapsed, horizon_seconds, timestamps_seconds):
        law = self.distribution(features, prices, elapsed, horizon_seconds, timestamps_seconds)
        return law.zero_probability, law.masses

    def checkpoint(self):
        return {'contract': CONTRACT, 'model': self.state_dict(), 'basis': self.head.basis,
                'inputMean': self.head.feature_mean, 'inputStd': self.head.feature_std,
                'featureMean': self.nsf.feature_mean, 'featureStd': self.nsf.feature_std,
                'dimensions': self.nsf.dimensions, 'latentWidth': self.nsf.width,
                'widths': self.head.widths, 'maximumSeconds': self.maximum_seconds}


def load_direct_latent_nsf(file: Path, device):
    saved = torch.load(file, map_location='cpu', weights_only=False)
    if saved.get('contract') != CONTRACT:
        raise ValueError('unsupported direct latent NSF checkpoint')
    nsf = LatentNSF(saved['featureMean'], saved['featureStd'], saved['dimensions'], saved['latentWidth'])
    model = DirectLatentNSF(nsf, saved['basis'], saved['inputMean'], saved['inputStd'],
                           tuple(saved['widths']), saved['maximumSeconds'])
    model.load_state_dict(saved['model'])
    return model.to(device).eval(), saved
