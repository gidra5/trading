"""Batched zero-atom price laws with Gaussian-coordinate triangular densities.

The conditional density is a normalized piecewise-linear function of
u=Phi(log(P/current)/sigma). CDFs and inverse CDFs integrate that density
analytically; moments use Gauss-Legendre quadrature over every knot interval.
The infinite Gaussian tails remain part of the law, including for sampling.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
import torch
from torch import Tensor

from return_knot_density import triangular_basis_areas


@lru_cache(maxsize=16)
def _quadrature(order: int, device: torch.device) -> tuple[Tensor, Tensor]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return (torch.as_tensor(nodes, dtype=torch.float64, device=device),
            torch.as_tensor(weights, dtype=torch.float64, device=device))


def _log_positive(value: Tensor) -> Tensor:
    # Mask before log so a zero-probability unused branch has finite gradients.
    return torch.where(value > 0, value.clamp_min(torch.finfo(value.dtype).tiny).log(), -torch.inf)


@dataclass(frozen=True)
class KnotPriceDistribution:
    current_price: Tensor
    zero_probability: Tensor
    masses: Tensor
    knots: Tensor
    areas: Tensor
    log_return_sigma: Tensor

    def __post_init__(self) -> None:
        tensors = (self.current_price, self.zero_probability, self.masses,
                   self.knots, self.areas, self.log_return_sigma)
        if any(not isinstance(x, Tensor) or not x.is_floating_point() for x in tensors):
            raise ValueError('price-law parameters must be floating tensors')
        if any(x.device != self.current_price.device for x in tensors):
            raise ValueError('price-law tensors must share a device')
        if any(not bool(torch.isfinite(x).all()) for x in tensors):
            raise ValueError('price-law parameters must be finite')
        batch = self.current_price.shape
        if len(batch) != 1 or not batch[0] or self.zero_probability.shape != batch or self.log_return_sigma.shape != batch:
            raise ValueError('price, zero probability and sigma must be nonempty [batch]')
        if self.knots.ndim != 1 or self.areas.shape != self.knots.shape or self.masses.shape != (batch[0], self.knots.numel()):
            raise ValueError('expected knots/areas [K] and conditional masses [batch,K]')
        expected_areas = triangular_basis_areas(self.knots.double())
        if self.knots[0] != 0 or self.knots[-1] != 1 or not torch.allclose(self.areas.double(), expected_areas, rtol=2e-6, atol=0):
            raise ValueError('knots must cover [0,1] and areas must be their triangular basis areas')
        if bool((self.current_price <= 0).any()) or bool((self.log_return_sigma < 0).any()):
            raise ValueError('positive current prices and nonnegative sigmas required')
        if bool(((self.zero_probability < 0) | (self.zero_probability > 1)).any()) or bool((self.masses < 0).any()):
            raise ValueError('invalid zero or conditional probabilities')
        if not torch.allclose(self.masses.double().sum(-1), torch.ones_like(self.current_price, dtype=torch.float64), rtol=0, atol=2e-6):
            raise ValueError('conditional knot masses must sum to one')

    def _values(self, values, name: str) -> tuple[Tensor, bool]:
        value = torch.as_tensor(values, dtype=torch.float64, device=self.current_price.device)
        if value.ndim == 0:
            value = value.expand(len(self.current_price))
        if value.ndim not in (1, 2) or value.shape[0] != len(self.current_price):
            raise ValueError(f'{name} must be scalar, [batch], or [batch,N]')
        if bool(torch.isnan(value).any()):
            raise ValueError(f'{name} cannot contain NaN')
        return (value[:, None], True) if value.ndim == 1 else (value, False)

    def _heights(self) -> Tensor:
        # Remove floating-point normalization residue without detaching gradients.
        mass = self.masses.double()
        return mass / mass.sum(-1, keepdim=True) / self.areas.double()

    def _active_cdf(self, units: Tensor) -> Tensor:
        knots, heights = self.knots.double(), self._heights()
        widths = knots.diff()
        prefix = torch.cat((heights.new_zeros((len(heights), 1)),
                            ((heights[:, :-1] + heights[:, 1:]) * widths / 2).cumsum(-1)), -1)
        index = (torch.searchsorted(knots, units.contiguous(), right=True) - 1).clamp(0, len(knots) - 2)
        left, right = heights.gather(1, index), heights.gather(1, index + 1)
        distance = units - knots[index]
        result = prefix.gather(1, index) + left * distance + (right - left) * distance.square() / (2 * widths[index])
        return torch.where(units <= 0, 0., torch.where(units >= 1, 1., result.clamp(0, 1)))

    def log_return_cdf(self, returns) -> Tensor:
        values, squeeze = self._values(returns, 'log returns')
        sigma = self.log_return_sigma.double()[:, None]
        active = self._active_cdf(torch.special.ndtr(values / torch.where(sigma > 0, sigma, 1.)))
        atom = (values >= 0).double()
        p0 = self.zero_probability.double()[:, None]
        result = torch.where(sigma == 0, atom, p0 * atom + (1 - p0) * active)
        return result[:, 0] if squeeze else result

    @staticmethod
    def _tick_size(tick_size: float) -> float:
        if isinstance(tick_size, bool) or not math.isfinite(tick_size) or tick_size <= 0:
            raise ValueError('tick size must be positive and finite')
        return tick_size

    def cdf(self, price, tick_size: float | None = .01) -> Tensor:
        """CDF of raw prices, or max(one tick, nearest tick) projected prices.

        Nearest-tick ties round upward. The zero atom is projected separately,
        so an anchor exactly on a half-tick obeys the same rule as samples.
        """
        prices, squeeze = self._values(price, 'prices')
        current = self.current_price.double()[:, None]
        if tick_size is None:
            returns = (prices.clamp_min(torch.finfo(prices.dtype).tiny) / current).log()
            result = torch.where(prices <= 0, 0., self.log_return_cdf(returns))
        else:
            tick = self._tick_size(tick_size)
            index = torch.floor(prices / tick + 1e-8)
            boundary = ((index + .5) * tick).clamp_min(torch.finfo(prices.dtype).tiny)
            sigma = self.log_return_sigma.double()[:, None]
            active = self._active_cdf(torch.special.ndtr((boundary / current).log() / torch.where(sigma > 0, sigma, 1.)))
            atom = (torch.floor(current / tick + .5).clamp_min(1) <= index).double()
            p0 = self.zero_probability.double()[:, None]
            result = torch.where(index < 1, 0., torch.where(sigma == 0, atom, p0 * atom + (1 - p0) * active))
        return result[:, 0] if squeeze else result

    def tick_probability(self, price, tick_size: float = .01) -> Tensor:
        tick = self._tick_size(tick_size)
        prices, squeeze = self._values(price, 'prices')
        result = (self.cdf(prices, tick) - self.cdf(prices - tick, tick)).clamp_min(0)
        return result[:, 0] if squeeze else result

    def log_prob(self, returns) -> Tensor:
        """Mixed log likelihood: atom mass at zero and Lebesgue density elsewhere.

        A zero-sigma row is exactly delta(0), with log mass 0 at zero and
        minus infinity elsewhere. Use tick_probability for quantized prices.
        """
        values, squeeze = self._values(returns, 'log returns')
        sigma = self.log_return_sigma.double()[:, None]
        safe_sigma = torch.where(sigma > 0, sigma, 1.)
        z = values / safe_sigma
        units = torch.special.ndtr(z)
        knots, heights = self.knots.double(), self._heights()
        index = (torch.searchsorted(knots, units.contiguous(), right=True) - 1).clamp(0, len(knots) - 2)
        width = knots[index + 1] - knots[index]
        fraction = ((units - knots[index]) / width).clamp(0, 1)
        # At a Gaussian-coordinate endpoint, Phi can round to zero or one.
        # Evaluate the vanishing hat weight through log Phi in the tail so
        # a zero endpoint height does not turn positive tail density into zero.
        log_left_weight = torch.where(knots[index + 1] == 1,
            torch.special.log_ndtr(-z) - width.log(), _log_positive(1 - fraction))
        log_right_weight = torch.where(knots[index] == 0,
            torch.special.log_ndtr(z) - width.log(), _log_positive(fraction))
        left = _log_positive(heights.gather(1, index)) + log_left_weight
        right = _log_positive(heights.gather(1, index + 1)) + log_right_weight
        supported = torch.isfinite(left) | torch.isfinite(right)
        log_height = torch.logaddexp(torch.where(supported, left, 0.), torch.where(supported, right, 0.))
        log_height = torch.where(supported, log_height, -torch.inf)
        active = log_height - .5 * z.square() - safe_sigma.log() - .5 * math.log(2 * math.pi)
        p0 = self.zero_probability.double()[:, None]
        result = torch.where(values == 0, _log_positive(p0), _log_positive(1 - p0) + active)
        result = torch.where(sigma == 0, torch.where(values == 0, 0., -torch.inf), result)
        return result[:, 0] if squeeze else result

    def log_return_density(self, returns) -> Tensor:
        """Mixed-law likelihood; exact zero observations score the zero atom."""
        return self.log_prob(returns)

    def _active_quantile_units(self, quantiles: Tensor) -> Tensor:
        knots, heights = self.knots.double(), self._heights()
        widths = knots.diff()
        interval_mass = (heights[:, :-1] + heights[:, 1:]) * widths / 2
        prefix = torch.cat((heights.new_zeros((len(heights), 1)), interval_mass.cumsum(-1)), -1)
        index = torch.searchsorted(prefix[:, 1:].contiguous(), quantiles.contiguous(), right=True).clamp_max(len(knots) - 2)
        local = (quantiles - prefix.gather(1, index)).clamp_min(0)
        left = heights.gather(1, index)
        slope = (heights.gather(1, index + 1) - left) / widths[index]
        radical = (left.square() + 2 * slope * local).clamp_min(0).sqrt()
        distance = 2 * local / (left + radical).clamp_min(torch.finfo(left.dtype).tiny)
        units = (knots[index] + distance).maximum(knots[index]).minimum(knots[index + 1])
        positive = interval_mass > 0
        lower = knots[positive.long().argmax(-1)][:, None]
        upper = knots[len(widths) - positive.flip(-1).long().argmax(-1)][:, None]
        return torch.where(quantiles <= 0, lower, torch.where(quantiles >= 1, upper, units))

    def quantile_log_returns(self, quantiles) -> Tensor:
        q, squeeze = self._values(quantiles, 'quantiles')
        if bool(((q < 0) | (q > 1)).any()):
            raise ValueError('quantiles must lie in [0,1]')
        p0 = self.zero_probability.double()[:, None]
        active_weight = 1 - p0
        lower_atom = active_weight * self._active_cdf(torch.full_like(q, .5))
        inside_atom = (p0 > 0) & (q >= lower_atom) & (q <= lower_atom + p0)
        active_q = torch.where(q <= lower_atom, q, q - p0) / active_weight.clamp_min(torch.finfo(q.dtype).tiny)
        unit = self._active_quantile_units(active_q.clamp(0, 1))
        sigma = self.log_return_sigma.double()[:, None]
        result = torch.special.ndtri(unit) * torch.where(sigma > 0, sigma, 1.)
        result = torch.where(inside_atom | (sigma == 0) | (p0 == 1), 0., result)
        return result[:, 0] if squeeze else result

    def sample_prices(self, samples: int = 512, seed: int = 37, tick_size: float | None = .01) -> Tensor:
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
            raise ValueError('positive integer sample count required')
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError('nonnegative integer seed required')
        q = torch.quasirandom.SobolEngine(1, scramble=True, seed=seed).draw(samples, dtype=torch.float64).to(self.current_price.device)
        epsilon = torch.finfo(q.dtype).eps
        q = q[:, 0].clamp(epsilon, 1 - epsilon).expand(len(self.current_price), -1)
        price = self.current_price.double()[:, None] * self.quantile_log_returns(q).exp()
        if tick_size is None:
            return price
        tick = self._tick_size(tick_size)
        return torch.floor(price / tick + .5).clamp_min(1) * tick

    def _moment(self, gross: bool, quadrature_order: int) -> Tensor:
        if isinstance(quadrature_order, bool) or not isinstance(quadrature_order, int) or quadrature_order < 8:
            raise ValueError('quadrature order must be an integer of at least eight')
        nodes, weights = _quadrature(quadrature_order, self.current_price.device)
        knots, heights = self.knots.double(), self._heights()
        widths = knots.diff()
        fraction = (nodes + 1) / 2
        unit = knots[:-1, None] + widths[:, None] * fraction
        standard = torch.special.ndtri(unit)
        sigma = self.log_return_sigma.double()
        result = []
        # Bound peak GPU memory when requesting many batched price moments.
        for start in range(0, len(sigma), 128):
            density = heights[start:start + 128, :-1, None] * (1 - fraction) + heights[start:start + 128, 1:, None] * fraction
            value = sigma[start:start + 128, None, None] * standard
            value = value.expm1() if gross else value
            result.append((value * density * weights * widths[None, :, None] / 2).sum((-1, -2)))
        return (1 - self.zero_probability.double()) * torch.cat(result)

    def mean_log_return(self, quadrature_order: int = 96) -> Tensor:
        """Numerical E[log(P/current)] integrating all knot intervals."""
        return self._moment(False, quadrature_order)

    def mean_price(self, quadrature_order: int = 96) -> Tensor:
        """Numerical unrounded E[P]; Gaussian tails give finite true moments.

        Quadrature is an approximation, not a tail truncation of the law.
        Increase quadrature_order to check numerical moment convergence.
        """
        return self.current_price.double() * (1 + self._moment(True, quadrature_order))
