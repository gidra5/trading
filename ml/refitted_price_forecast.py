"""Inference adapter for frozen native feature models and their price-law heads."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from torch import nn, Tensor
from evaluate_stopped_structured_feature_process import model_from_state
from horizon_price_forecast import load_horizon_forecast
from stochastic_price_forecast import component_log_masses
from trading_storage import load_torch_checkpoint
from audit_forecast_agreement import sha


class RefittedPriceForecast(nn.Module):
    def __init__(self, native: nn.Module, head: nn.Module):
        super().__init__()
        self.native, self.head = native, head
        self.requires_grad_(False)
        self.eval()

    def forward(self, observed_states: Tensor) -> Tensor:
        if observed_states.ndim != 3 or observed_states.shape[1:] != (2, 59):
            raise ValueError('expected two observed production59 states per example')
        return self.head(self.native(observed_states).flatten(1))

    @torch.inference_mode()
    def probabilities(self, observed_states: Tensor) -> tuple[Tensor, Tensor]:
        logits = self(observed_states)
        masses = component_log_masses(logits[:, 1:], self.head.areas).exp().double()
        return logits[:, 0].sigmoid().double(), masses/masses.sum(dim=1, keepdim=True)


def load_refitted_forecast(directory: Path, device: torch.device) -> RefittedPriceForecast:
    plan = json.loads((directory/'plan.json').read_text())
    selection = json.loads((directory/'selection.json').read_text())
    if plan['contract'] != 'frozen-historical-feature-knot-head-v1':
        raise ValueError('unsupported refitted forecast')
    pointer = Path(plan['nativeCheckpoint'])
    if json.loads(pointer.read_text()) != plan['nativeCheckpointPointer']:
        raise ValueError('native snapshot changed after head fitting')
    native_plan_file = Path(plan['nativePlan'])
    if sha(native_plan_file) != plan['nativePlanSha256'] or sha(directory/'head.pt') != selection['headCheckpointSha256']:
        raise ValueError('refitted forecast source hash changed')
    native_plan = json.loads(native_plan_file.read_text())
    saved = load_torch_checkpoint(pointer, map_location='cpu', weights_only=False)
    native = model_from_state(native_plan, saved['model'], device)
    head, _ = load_horizon_forecast(directory/'head.pt', device)
    return RefittedPriceForecast(native, head)
