"""Independent zero, active-sign and magnitude experts with a proper likelihood.

The return law has an exact zero atom, a Gamma upward magnitude, and a Beta
downward magnitude on (0,1). Conditional means match the exported size heads.
The endpoint policy's three-point approximation discards conditional spread.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from neural_sign_size_forecast import StructuredSpecialist
from ternary_sign_size_forecast import physical_downside_cap

CONTRACT = "structured-hurdle-sign-size-nll-v1"
MINIMUM_MEAN = 1e-12
MINIMUM_SHAPE, MAXIMUM_SHAPE = 1e-3, 100.


def hurdle_log_probabilities(parameters: Tensor) -> Tensor:
    """Stable DOWN/ZERO/UP log masses from zero and active-sign logits."""
    zero, sign = parameters[..., 0], parameters[..., 1]
    active = F.logsigmoid(-zero)
    return torch.stack((active+F.logsigmoid(-sign), F.logsigmoid(zero),
                        active+F.logsigmoid(sign)), dim=-1)


def hurdle_components(parameters: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    return hurdle_log_probabilities(parameters).exp(), parameters[..., 2], parameters[..., 3]


def hurdle_mean(parameters: Tensor) -> Tensor:
    p, up, down = hurdle_components(parameters)
    return p[..., 2]*up-p[..., 0]*down


def log_gamma_increment(base: Tensor, increment: Tensor) -> Tensor:
    """log Gamma(base+increment)-log Gamma(base), avoiding large-base cancellation.

For base > 1e6 and increment/base < 1e-3, use the first three inverse-base
terms of the log-Gamma ratio expansion. The model bounds increment by 100;
the first omitted term is below 1e-14 at this branch boundary. This computes
the same Beta density to floating precision, rather than changing its support.
"""
    a, b = increment, base
    direct = torch.lgamma(b+a)-torch.lgamma(b)
    series = (a*torch.log(b) + a*(a-1)/(2*b)
              - a*(2*a*a-3*a+1)/(12*b.square())
              + a.square()*(a-1).square()/(12*b.pow(3)))
    return torch.where((b > 1e6) & (a/b < 1e-3), series, direct)


def hurdle_loss_parts(parameters: Tensor, returns: Tensor) -> dict[str, Tensor]:
    """Masked per-example likelihood terms; average their sum over ALL rows.

Dataset validation must require finite returns greater than -1. Safe values
are substituted before evaluating logs on inactive branches, so exact zeros
and opposite-sign targets do not create undefined operations or gradients.
"""
    zero_logit, sign_logit = parameters[..., 0], parameters[..., 1]
    up, down, alpha_up, alpha_down = (parameters[..., k] for k in range(2, 6))
    positive, negative = returns > 0, returns < 0
    active = positive | negative
    zero_loss = F.binary_cross_entropy_with_logits(zero_logit, (~active).to(zero_logit.dtype), reduction="none")
    sign_loss = F.binary_cross_entropy_with_logits(sign_logit, positive.to(sign_logit.dtype), reduction="none")
    upward = torch.where(positive, returns, torch.ones_like(returns))
    downward = torch.where(negative, -returns, torch.full_like(returns, .5))
    rate = alpha_up/up
    up_nll = (torch.lgamma(alpha_up)-alpha_up*torch.log(rate)
              -(alpha_up-1)*torch.log(upward)+rate*upward)
    beta = alpha_down*(1-down)/down
    log_normalizer = torch.lgamma(alpha_down)-log_gamma_increment(beta, alpha_down)
    down_nll = log_normalizer-(alpha_down-1)*torch.log(downward)-(beta-1)*torch.log1p(-downward)
    zeros = torch.zeros_like(returns)
    return {"zero": zero_loss, "sign": torch.where(active, sign_loss, zeros),
            "up": torch.where(positive, up_nll, zeros), "down": torch.where(negative, down_nll, zeros)}


def hurdle_nll_elements(parameters: Tensor, returns: Tensor) -> Tensor:
    return sum(hurdle_loss_parts(parameters, returns).values())


class HurdleSignSize(nn.Module):
    """Three separately owned backbones, with global per-horizon size shapes."""

    contract = CONTRACT

    def __init__(self, zero: StructuredSpecialist, sign: StructuredSpecialist,
                 size: StructuredSpecialist, scales: Tensor, shapes: Tensor | None = None) -> None:
        super().__init__()
        if zero.task != "sign" or sign.task != "sign" or size.task != "size":
            raise ValueError("zero/sign experts must emit logits and the size expert positive means")
        if zero.horizons != sign.horizons or zero.horizons != size.horizons:
            raise ValueError("all expert horizon orders must agree")
        groups = [{id(p) for p in expert.parameters()} for expert in (zero, sign, size)]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
            raise ValueError("experts must own separate parameters")
        if scales.shape != (len(zero.horizons), 2) or not scales.is_floating_point() \
                or not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any()):
            raise ValueError("scales must be finite positive floating [horizons,UP/DOWN]")
        self.zero, self.sign, self.size = zero, sign, size
        self.horizons = zero.horizons
        self.register_buffer("scales", scales.detach().clone().double())
        self.raw_shapes = nn.Parameter(torch.zeros_like(self.scales))
        self.set_shapes_(torch.ones_like(self.scales) if shapes is None else shapes)

    @property
    def downside_cap(self) -> float:
        return physical_downside_cap(self.scales.dtype)

    @property
    def shapes(self) -> Tensor:
        return MINIMUM_SHAPE+(MAXIMUM_SHAPE-MINIMUM_SHAPE)*self.raw_shapes.sigmoid()

    @torch.no_grad()
    def set_shapes_(self, shapes: Tensor) -> None:
        if shapes.shape != self.scales.shape or not bool(torch.isfinite(shapes).all()) \
                or bool(((shapes <= MINIMUM_SHAPE) | (shapes >= MAXIMUM_SHAPE)).any()):
            raise ValueError("initial shapes must lie strictly within the shape bounds")
        fraction = (shapes.to(self.raw_shapes)-MINIMUM_SHAPE)/(MAXIMUM_SHAPE-MINIMUM_SHAPE)
        self.raw_shapes.copy_(torch.logit(fraction))

    def forward(self, inputs: Tensor) -> Tensor:
        zero, sign = self.zero(inputs).double(), self.sign(inputs).double()
        physical = self.size(inputs)*self.scales
        up = physical[..., 0].clamp_min(MINIMUM_MEAN)
        down = physical[..., 1].clamp(MINIMUM_MEAN, self.downside_cap)
        alpha = self.shapes.expand(inputs.shape[0], -1, -1)
        return torch.stack((zero, sign, up, down, alpha[..., 0], alpha[..., 1]), dim=-1)

    def components(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return hurdle_components(self(inputs))

    def mean(self, inputs: Tensor) -> Tensor:
        return hurdle_mean(self(inputs))

    def muon_parameters(self) -> tuple[Tensor, ...]:
        return (*self.zero.muon_parameters(), *self.sign.muon_parameters(), *self.size.muon_parameters())
