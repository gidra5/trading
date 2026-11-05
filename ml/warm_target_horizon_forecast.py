"""Add a horizon input to an existing price head without changing its H=1 law."""
from __future__ import annotations
from pathlib import Path
import torch
from torch import nn, Tensor
from horizon_price_forecast import HorizonPriceForecast, load_horizon_forecast
from return_oracle_ce import fused_glu
from target_horizon_forecast import HorizonRange, TargetHorizonForecast

CONTRACT = 'warm-target-seconds-tanh-gaussian-knot-law-v1'


class WarmTargetHorizonForecast(TargetHorizonForecast):
    def __init__(self, head: HorizonPriceForecast, horizon_range: HorizonRange = HorizonRange()):
        # Preserve the head's architecture, parameters, normalizers and complete law.
        nn.Module.__init__(self)
        if head.dropout.p != 0 or head.input_normalization != 'training-position':
            raise ValueError('warm adapter requires the deterministic training-position price head')
        self.head, self.horizon_range = head, horizon_range
        self.feature_count, self.widths = head.feature_count, head.widths
        self.horizon_weight = nn.Parameter(torch.zeros_like(head.layers[0].bias))

    def forward(self, features: Tensor, horizon_seconds: Tensor) -> Tensor:
        h = torch.as_tensor(horizon_seconds, device=features.device)
        if h.ndim == 0:
            h = h.expand(len(features))
        if features.ndim != 2 or features.shape[1] != self.feature_count or h.shape != (len(features),):
            raise ValueError('expected features [batch, channels] and scalar or [batch] horizon')
        z = self.horizon_range.normalize(h).to(features.dtype)
        z1 = self.horizon_range.normalize(torch.ones_like(h)).to(features.dtype)
        hidden = self.head.normalize_features(features)
        for i, layer in enumerate(self.head.layers):
            preactivation = layer(hidden)
            if i == 0:
                # Separate multiply/add keeps the parent's matrix multiply unchanged.
                # This is algebraically an extra input column with zero initial weights.
                preactivation = preactivation+(z-z1)[:, None]*self.horizon_weight
            hidden, _, _ = fused_glu(preactivation, self.head.value_centering_normalizers[i],
                self.head.gate_centering_normalizers[i], self.head.value_norm_biases[i],
                self.head.gate_norm_biases[i], self.head.value_transforms[i], self.head.gate_transforms[i])
        return self.head.output(hidden)


def load_warm_forecast(file: Path, device: torch.device) -> tuple[WarmTargetHorizonForecast, dict]:
    saved = torch.load(file, map_location='cpu', weights_only=False)
    if saved.get('contract') != CONTRACT:
        raise ValueError('unsupported warm horizon checkpoint')
    head = HorizonPriceForecast(saved['featureMean'], saved['featureStd'], saved['basis'], tuple(saved['widths']))
    model = WarmTargetHorizonForecast(head, HorizonRange(**saved['horizonRange']))
    model.load_state_dict(saved['model'])
    return model.to(device).eval(), saved


def load_warm_observed_forecast(file: Path, device: torch.device):
    from target_horizon_forecast import ObservedTargetHorizonForecast
    from refitted_price_forecast import load_refitted_forecast
    from audit_forecast_agreement import sha
    model, saved = load_warm_forecast(file, device)
    root = Path(saved['nativeDirectory'])
    if sha(root/'plan.json') != saved['nativePlanSha256']:
        raise ValueError('native source changed')
    return ObservedTargetHorizonForecast(load_refitted_forecast(root, device).native, model)
