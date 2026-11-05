"""A full endpoint log-return law conditioned on an integer horizon in seconds.

This supplies marginals P(log(P[t+h]/P[t]) | I[t], h), not a joint path law.
The sqrt(h) target coordinate is invertible; it does not impose Brownian
increments, independent increments, or a horizon-invariant distribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch import Tensor, nn
from horizon_price_forecast import HorizonPriceForecast
from stochastic_price_forecast import component_log_masses

CONTRACT = 'target-seconds-tanh-gaussian-knot-law-v1'


@dataclass(frozen=True)
class HorizonRange:
    scale_seconds: float = 21600.0
    maximum_seconds: int = 86400

    def __post_init__(self):
        if not np.isfinite(self.scale_seconds) or self.scale_seconds <= 0:
            raise ValueError('horizon scale must be finite and positive')
        if isinstance(self.maximum_seconds, bool) or not isinstance(self.maximum_seconds, int) or self.maximum_seconds < 1:
            raise ValueError('maximum horizon must be a positive integer')

    def probabilities(self) -> np.ndarray:
        # Integrate the derivative over each integer bin (h-1, h].
        edges = np.tanh(np.arange(self.maximum_seconds+1)/self.scale_seconds)
        return np.diff(edges)/edges[-1]

    def sample(self, rng: np.random.Generator, count: int) -> np.ndarray:
        u = rng.random(count)*np.tanh(self.maximum_seconds/self.scale_seconds)
        return np.clip(np.ceil(self.scale_seconds*np.arctanh(u)), 1, self.maximum_seconds).astype(np.int64)

    def normalize(self, h: Tensor) -> Tensor:
        if h.dtype == torch.bool or not bool(torch.all(torch.isfinite(h) & (h >= 1) & (h <= self.maximum_seconds) & (h == h.round()))):
            raise ValueError('horizon must be integer seconds inside the trained range')
        return torch.tanh(h/self.scale_seconds)


def complete_origins(origins: np.ndarray, last_price_index: int, horizon_range: HorizonRange) -> np.ndarray:
    """Common origins for all horizons: sampling never depends on future availability."""
    return origins[origins+horizon_range.maximum_seconds <= last_price_index]


class TargetHorizonForecast(nn.Module):
    def __init__(self, feature_mean: Tensor, feature_std: Tensor, basis: dict,
                 horizon_range: HorizonRange = HorizonRange(), widths: tuple[int, ...] = (128, 128)):
        super().__init__()
        self.horizon_range, self.widths = horizon_range, widths
        self.feature_count = len(feature_mean)
        # Keep tanh's scale intact rather than standardizing it again.
        self.head = HorizonPriceForecast(torch.cat((feature_mean, torch.zeros(1))),
            torch.cat((feature_std, torch.ones(1))), basis, widths)

    def forward(self, features: Tensor, horizon_seconds: Tensor) -> Tensor:
        h = torch.as_tensor(horizon_seconds, device=features.device)
        if h.ndim == 0:
            h = h.expand(len(features))
        if features.ndim != 2 or features.shape[1] != self.feature_count or h.shape != (len(features),):
            raise ValueError('expected features [batch, channels] and scalar or [batch] horizon')
        z = self.horizon_range.normalize(h).to(features.dtype)
        return self.head(torch.cat((features, z[:, None]), dim=1))

    def log_prob_from_logits(self, logits: Tensor, returns: Tensor, horizons: Tensor) -> Tensor:
        self.horizon_range.normalize(horizons)
        scale = horizons.double().sqrt()
        # Float64 CDF avoids loss of resolution near narrow central knots.
        return self.head.log_prob_from_logits(logits.double(), returns.double()/scale) - torch.where(
            returns == 0, 0., scale.log())

    def expectation_from_logits(self, logits: Tensor, horizons: Tensor) -> Tensor:
        return self.head.expectation_from_logits(logits)*horizons.to(logits.dtype).sqrt()

    @torch.inference_mode()
    def probabilities(self, features: Tensor, horizon_seconds: Tensor) -> tuple[Tensor, Tensor]:
        logits = self(features, horizon_seconds)
        masses = component_log_masses(logits[:, 1:].double(), self.head.areas.double()).exp()
        return logits[:, 0].double().sigmoid(), masses/masses.sum(dim=1, keepdim=True)

    @torch.inference_mode()
    def quantile_returns(self, masses: np.ndarray, horizons: np.ndarray, branches: int) -> np.ndarray:
        self.horizon_range.normalize(torch.as_tensor(horizons))
        return self.head.quantile_returns(masses, branches)*np.sqrt(horizons)[:, None]


def load_target_horizon_forecast(file: Path, device: torch.device) -> tuple[TargetHorizonForecast, dict]:
    payload = torch.load(file, map_location='cpu', weights_only=False)
    if payload.get('contract') != CONTRACT:
        raise ValueError('unsupported target-horizon checkpoint')
    model = TargetHorizonForecast(payload['featureMean'], payload['featureStd'], payload['basis'],
        HorizonRange(**payload['horizonRange']), tuple(payload['widths']))
    model.load_state_dict(payload['model'])
    return model.to(device).eval(), payload


class ObservedTargetHorizonForecast(nn.Module):
    """Causal native-feature adapter; no future feature cache is read at inference."""
    def __init__(self, native: nn.Module, forecast: TargetHorizonForecast):
        super().__init__()
        self.native, self.forecast = native, forecast
        self.requires_grad_(False)
        self.eval()

    def forward(self, observed_states: Tensor, horizon_seconds: Tensor) -> Tensor:
        if observed_states.ndim != 3 or observed_states.shape[1:] != (2, 59):
            raise ValueError('expected two observed production59 states')
        return self.forecast(self.native(observed_states).flatten(1), horizon_seconds)


def load_observed_target_horizon_forecast(file: Path, device: torch.device) -> ObservedTargetHorizonForecast:
    from refitted_price_forecast import load_refitted_forecast
    from audit_forecast_agreement import sha
    forecast, payload = load_target_horizon_forecast(file, device)
    source = Path(payload['nativeDirectory'])
    if sha(source/'plan.json') != payload['nativePlanSha256']:
        raise ValueError('native forecast provenance changed')
    native = load_refitted_forecast(source, device).native
    return ObservedTargetHorizonForecast(native, forecast)
