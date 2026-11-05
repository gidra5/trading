"""Three shared-horizon experts for a zero atom and signed Gamma log returns.

The UP and DOWN heads predict positive conditional log-magnitude means. A smooth
UP Gamma scale bound preserves finite first and second price moments. Log-return
means and price-return means are separate analytic quantities throughout.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from hurdle_sign_size_forecast import hurdle_log_probabilities
from neural_sign_size_forecast import StructuredSpecialist

CONTRACT = "structured-log-hurdle-sign-size-gamma-v1"
MINIMUM_MEAN = 1e-12
MINIMUM_SHAPE, MAXIMUM_SHAPE = 1e-3, 100.
MAXIMUM_UP_MEAN_RATIO = .49


def log_hurdle_log_probabilities(parameters: Tensor) -> Tensor:
    """DOWN/ZERO/UP masses; the sign head conditions on a nonzero outcome."""
    return hurdle_log_probabilities(parameters)


def log_hurdle_components(parameters: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    return log_hurdle_log_probabilities(parameters).exp(), parameters[..., 2], parameters[..., 3]


def log_hurdle_mean(parameters: Tensor) -> Tensor:
    """E[X] for X=log(P_next/P_now), including the exact zero atom."""
    probability, up, down = log_hurdle_components(parameters)
    return probability[..., 2]*up-probability[..., 0]*down


def log_hurdle_conditional_log_gross_means(parameters: Tensor, power: float = 1.) -> tuple[Tensor, Tensor]:
    """log E[exp(power*X)|UP/DOWN] for positive powers at most two.

UP Gamma scale <= .49 implies its moment-generating function exists at both
one and two. At shape <= 100, even the largest second gross moment is finite
in float64: (1-2*.49)^(-100) < 1e170. DOWN has all positive gross moments.
These are logs OF gross means, distinct from E[X].
"""
    if not 0 < power <= 2:
        raise ValueError("the declared law guarantees positive price moments only through power two")
    up, down, alpha_up, alpha_down = (parameters[..., k] for k in range(2, 6))
    up_value = -alpha_up*torch.log1p(-power*up/alpha_up)
    down_value = -alpha_down*torch.log1p(power*down/alpha_down)
    return up_value, down_value


def log_hurdle_gross_components(parameters: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Masses and conditional E[P_next/P_now] for UP and DOWN.

Use the DOWN gross mean directly when constructing endpoint prices; computing
1 minus its simple-loss magnitude can erase a very small positive price.
"""
    probability = log_hurdle_log_probabilities(parameters).exp()
    up, down = log_hurdle_conditional_log_gross_means(parameters)
    return probability, up.exp(), down.exp()


def log_hurdle_simple_components(parameters: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Masses and positive UP/DOWN simple-return conditional magnitudes."""
    probability = log_hurdle_log_probabilities(parameters).exp()
    up, down = log_hurdle_conditional_log_gross_means(parameters)
    return probability, torch.expm1(up), -torch.expm1(down)


def log_hurdle_simple_mean(parameters: Tensor) -> Tensor:
    """E[expm1(X)]; generally not expm1(E[X])."""
    probability, up, down = log_hurdle_simple_components(parameters)
    return probability[..., 2]*up-probability[..., 0]*down


def log_hurdle_loss_parts(parameters: Tensor, returns: Tensor) -> dict[str, Tensor]:
    """Per-row log-return NLL with zero, active sign and actual-side Gamma terms.

Finite log targets can have either sign with unbounded magnitude; in particular
the old simple-return lower bound of -1 does not apply. Inactive Gamma branches
receive safe positive arguments before their outputs are masked.
"""
    zero_logit, sign_logit = parameters[..., 0], parameters[..., 1]
    positive, negative = returns > 0, returns < 0
    active = positive | negative
    zeros = torch.zeros_like(returns)
    zero_loss = F.binary_cross_entropy_with_logits(zero_logit, (~active).to(zero_logit.dtype), reduction="none")
    sign_loss = F.binary_cross_entropy_with_logits(sign_logit, positive.to(sign_logit.dtype), reduction="none")
    parts = {"zero": zero_loss, "sign": torch.where(active, sign_loss, zeros)}
    for name, mask, mean, alpha in (("up", positive, parameters[..., 2], parameters[..., 4]),
                                     ("down", negative, parameters[..., 3], parameters[..., 5])):
        magnitude = torch.where(mask, returns.abs(), torch.ones_like(returns))
        rate = alpha/mean
        nll = torch.lgamma(alpha)-alpha*torch.log(rate)-(alpha-1)*torch.log(magnitude)+rate*magnitude
        parts[name] = torch.where(mask, nll, zeros)
    return parts


def log_hurdle_nll_elements(parameters: Tensor, returns: Tensor) -> Tensor:
    return sum(log_hurdle_loss_parts(parameters, returns).values())


class LogHurdleSignSize(nn.Module):
    """One zero, one sign and one size backbone, each emitting every horizon.

UP mean = alpha*c*tanh(raw_mean/(alpha*c)), c=.49. This smooth bound has an
identity derivative at zero and a cubic leading distortion. It bounds Gamma
scale, not realized log returns: the modeled positive tail remains unbounded.
The DOWN log magnitude uses an ordinary positive Gamma mean with no price-loss
cap. Shapes are global per horizon/direction; expert weights remain independent.
"""

    contract = CONTRACT

    def __init__(self, zero: StructuredSpecialist, sign: StructuredSpecialist,
                 size: StructuredSpecialist, scales: Tensor, shapes: Tensor | None = None) -> None:
        super().__init__()
        if zero.task != "sign" or sign.task != "sign" or size.task != "size":
            raise ValueError("zero/sign experts must emit logits and size must emit positive means")
        if zero.horizons != sign.horizons or zero.horizons != size.horizons:
            raise ValueError("all expert horizon orders must agree")
        groups = [{id(p) for p in expert.parameters()} for expert in (zero, sign, size)]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
            raise ValueError("experts must own separate parameters")
        if scales.shape != (len(zero.horizons), 2) or not scales.is_floating_point() \
                or not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any()):
            raise ValueError("log-magnitude scales must be finite positive floating [horizons,UP/DOWN]")
        self.zero, self.sign, self.size = zero, sign, size
        self.horizons = zero.horizons
        self.register_buffer("scales", scales.detach().clone().double())
        self.raw_shapes = nn.Parameter(torch.zeros_like(self.scales))
        self.set_shapes_(torch.ones_like(self.scales) if shapes is None else shapes)

    @property
    def shapes(self) -> Tensor:
        return MINIMUM_SHAPE+(MAXIMUM_SHAPE-MINIMUM_SHAPE)*self.raw_shapes.sigmoid()

    @torch.no_grad()
    def set_shapes_(self, shapes: Tensor) -> None:
        if shapes.shape != self.scales.shape or not bool(torch.isfinite(shapes).all()) \
                or bool(((shapes <= MINIMUM_SHAPE) | (shapes >= MAXIMUM_SHAPE)).any()):
            raise ValueError("initial shapes must lie strictly within the configured bounds")
        fraction = (shapes.to(self.raw_shapes)-MINIMUM_SHAPE)/(MAXIMUM_SHAPE-MINIMUM_SHAPE)
        self.raw_shapes.copy_(torch.logit(fraction))

    def forward(self, inputs: Tensor) -> Tensor:
        zero, sign = self.zero(inputs).double(), self.sign(inputs).double()
        raw_means = self.size(inputs).double()*self.scales
        alpha = self.shapes.expand(inputs.shape[0], -1, -1)
        cap = alpha[..., 0]*MAXIMUM_UP_MEAN_RATIO
        up = (cap*torch.tanh(raw_means[..., 0]/cap)).clamp_min(MINIMUM_MEAN)
        down = raw_means[..., 1].clamp_min(MINIMUM_MEAN)
        return torch.stack((zero, sign, up, down, alpha[..., 0], alpha[..., 1]), dim=-1)

    def components(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return log_hurdle_components(self(inputs))

    def simple_components(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return log_hurdle_simple_components(self(inputs))

    def mean(self, inputs: Tensor) -> Tensor:
        return log_hurdle_mean(self(inputs))

    def simple_mean(self, inputs: Tensor) -> Tensor:
        return log_hurdle_simple_mean(self(inputs))

    def muon_parameters(self) -> tuple[Tensor, ...]:
        return (*self.zero.muon_parameters(), *self.sign.muon_parameters(), *self.size.muon_parameters())
