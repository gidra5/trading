"""Observed-state adapter returning an NSF price law for requested seconds."""
from pathlib import Path
import json
import torch
from torch import nn
from audit_forecast_agreement import sha
from neural_stochastic_flow import load_nsf
from refitted_price_forecast import load_refitted_forecast


class NSFPriceForecast(nn.Module):
    def __init__(self, native, flow):
        super().__init__()
        self.native, self.flow = native, flow
        self.requires_grad_(False)
        self.eval()

    @torch.inference_mode()
    def distribution(self, observed_states, current_price, ewma_volatility, horizon_seconds):
        """Volatility is causal EWMA standard deviation of 1s log returns, half-life 300s.

        Input shape [batch,2,59]; prices and volatility [batch]; horizon scalar
        or [batch], in seconds. Return a full price marginal with sample/CDF APIs.
        """
        if observed_states.ndim != 3 or observed_states.shape[1:] != (2, 59):
            raise ValueError('expected two observed production59 feature states')
        if ewma_volatility.shape != (len(observed_states),) or not bool(torch.all(
                torch.isfinite(ewma_volatility) & (ewma_volatility >= 0))):
            raise ValueError('finite nonnegative causal volatility required')
        if not bool(torch.isfinite(observed_states).all()):
            raise ValueError('nonfinite observed features')
        features = self.native(observed_states).flatten(1)
        log_vol = (ewma_volatility.clamp_min(1e-6)/.0001).log()
        return self.flow.distribution(features, log_vol, current_price, horizon_seconds)


def load_nsf_price_forecast(checkpoint: Path, device: torch.device):
    checkpoint = Path(checkpoint)
    flow, saved = load_nsf(checkpoint, device)
    plan_file = checkpoint.parent/'plan.json'
    if sha(plan_file) != saved['planSha256']:
        raise ValueError('NSF plan changed')
    plan = json.loads(plan_file.read_text())
    native_root = Path(plan['nativeDirectory'])
    if sha(native_root/'plan.json') != plan['nativePlanSha256']:
        raise ValueError('native feature model provenance changed')
    return NSFPriceForecast(load_refitted_forecast(native_root, device).native, flow)
