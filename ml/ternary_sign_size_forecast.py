"""Three-state sign and conditional-size experts with a shared physical mean.

Pretraining identifies calibrated components only to the extent supported by
its data and losses. After signed-mean-only fine-tuning, their product remains
a valid finite return law, but the heads are latent factors unless their
individual calibration is established separately.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from neural_sign_size_forecast import HORIZONS, StructuredSpecialist

CONTRACT = "structured-ternary-sign-size-v1"
DOWN, ZERO, UP = 0, 1, 2


def physical_downside_cap(dtype: torch.dtype) -> float:
    """A loss below one even after rounding to the physical output dtype."""
    return min(1-1e-12, 1-torch.finfo(dtype).eps)


def ternary_sign_labels(returns: Tensor) -> Tensor:
    """Finite simple-return labels in DOWN, ZERO, UP order, retaining exact zero."""
    if not bool(torch.isfinite(returns).all()):
        raise ValueError("ternary labels require finite returns")
    return (returns > 0).long() - (returns < 0).long() + ZERO


class TernarySignSpecialist(StructuredSpecialist):
    """Historical base17 sign backbone with three logits at every horizon."""

    contract = CONTRACT

    def __init__(self, input_mean: Tensor, input_std: Tensor,
                 horizons: Sequence[int] = HORIZONS) -> None:
        super().__init__("sign", input_mean, input_std, horizons)
        self.readout = nn.Linear(self.readout.in_features, 3*len(self.horizons))
        nn.init.zeros_(self.readout.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        return super().forward(inputs).reshape(inputs.shape[0], len(self.horizons), 3)


class CombinedSignSize(nn.Module):
    """A single differentiable return law, shared by training and inference.

``scales`` are UP/DOWN conditional simple-return scales. Their floating dtype
is retained, so physical outputs use PyTorch's normal dtype promotion. The
public ``downside_cap`` reports the corresponding representable upper loss.
"""

    contract = CONTRACT

    def __init__(self, sign: TernarySignSpecialist, size: StructuredSpecialist,
                 scales: Tensor) -> None:
        super().__init__()
        if not isinstance(sign, TernarySignSpecialist) or size.task != "size":
            raise ValueError("combined law requires ternary sign and conditional-size specialists")
        if sign.horizons != size.horizons:
            raise ValueError("specialist horizon orders must agree")
        if {id(p) for p in sign.parameters()} & {id(p) for p in size.parameters()}:
            raise ValueError("the two specialists must own separate parameters")
        if scales.shape != (len(sign.horizons), 2) or not scales.is_floating_point() \
                or not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any()):
            raise ValueError("scales must be positive finite floating [horizons,UP/DOWN]")
        self.sign, self.size = sign, size
        self.horizons = sign.horizons
        self.register_buffer("scales", scales.detach().clone())

    @property
    def downside_cap(self) -> float:
        dtype = torch.promote_types(self.size.readout.weight.dtype, self.scales.dtype)
        return physical_downside_cap(dtype)

    def components(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        probabilities = self.sign(inputs).softmax(dim=-1)
        physical = self.size(inputs)*self.scales
        up = physical[..., 0]
        down = physical[..., 1].clamp_max(physical_downside_cap(physical.dtype))
        return probabilities, up, down

    def forward(self, inputs: Tensor) -> Tensor:
        probabilities, up, down = self.components(inputs)
        return probabilities[..., UP]*up - probabilities[..., DOWN]*down

    def muon_parameters(self) -> tuple[Tensor, ...]:
        return (*self.sign.muon_parameters(), *self.size.muon_parameters())


def select_horizon_pair(sign: TernarySignSpecialist, size: StructuredSpecialist,
                        scales: Tensor, horizon: int) -> CombinedSignSize:
    """Copy one pair of selected heads and both full backbones for fine-tuning.

The sources can come from different selected epochs. Their weights, buffers,
modes and scales are retained; optimizer state and other horizon heads are
not transferred. Neither source is mutated or shares parameters with the copy.
"""
    if sign.horizons != size.horizons or horizon not in sign.horizons:
        raise ValueError("selected horizon must occur in both aligned specialists")
    if scales.shape != (len(sign.horizons), 2):
        raise ValueError("source scales must follow the full specialist horizon order")
    index = sign.horizons.index(horizon)
    selected_sign = TernarySignSpecialist(sign.backbone.input_mean, sign.backbone.input_std, (horizon,))
    selected_size = StructuredSpecialist("size", size.backbone.input_mean, size.backbone.input_std, (horizon,))
    for source, selected, width in ((sign, selected_sign, 3), (size, selected_size, 2)):
        selected.to(device=source.readout.weight.device, dtype=source.readout.weight.dtype)
        selected.backbone.load_state_dict(source.backbone.state_dict())
        with torch.no_grad():
            selected.readout.weight.copy_(source.readout.weight[index*width:(index+1)*width])
            selected.readout.bias.copy_(source.readout.bias[index*width:(index+1)*width])
        selected.train(source.training)
    return CombinedSignSize(selected_sign, selected_size, scales[index:index+1])
