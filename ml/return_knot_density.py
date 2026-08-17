from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


RETURN_TO_BPS = 10_000.0


@dataclass(frozen=True)
class ReturnTransform:
    alpha: float
    location_bps: float
    scale_bps: float

    def validate(self) -> None:
        if not math.isfinite(self.alpha) or self.alpha <= 1:
            raise ValueError("return transform alpha must be finite and above one")
        if not math.isfinite(self.location_bps):
            raise ValueError("return transform location must be finite")
        if not math.isfinite(self.scale_bps) or self.scale_bps <= 0:
            raise ValueError("return transform scale must be finite and positive")


@dataclass(frozen=True)
class KnotDensityContract:
    transform: ReturnTransform
    knots_unit: np.ndarray
    prior_component_masses: np.ndarray
    source_file: str
    source_fit: str

    @classmethod
    def load(cls, file: Path, *, fit: str = "kl") -> KnotDensityContract:
        value = json.loads(file.read_text(encoding="utf-8"))
        transform_value = value["transform"]
        transform = ReturnTransform(
            alpha=float(transform_value["alpha"]),
            location_bps=float(transform_value["locationBps"]),
            scale_bps=float(transform_value["scaleBps"]),
        )
        transform.validate()
        selected = value["fits"][fit]
        knots = np.asarray(selected["knotsUnit"], dtype=np.float64).copy()
        knots[0] = 0.0
        knots[-1] = 1.0
        result = cls(
            transform=transform,
            knots_unit=knots,
            prior_component_masses=np.asarray(
                selected["componentWeights"], dtype=np.float64
            ),
            source_file=str(file),
            source_fit=fit,
        )
        result.validate()
        return result

    def validate(self) -> None:
        knots = self.knots_unit
        masses = self.prior_component_masses
        if knots.ndim != 1 or knots.size < 3 \
                or masses.shape != knots.shape \
                or not np.isfinite(knots).all() \
                or not np.isfinite(masses).all() \
                or not math.isclose(float(knots[0]), 0.0, abs_tol=1e-12) \
                or not math.isclose(float(knots[-1]), 1.0, abs_tol=1e-12) \
                or bool((np.diff(knots) <= 0).any()) \
                or bool((masses <= 0).any()) \
                or not math.isclose(float(masses.sum()), 1.0, abs_tol=1e-8):
            raise ValueError("invalid fixed return-knot density contract")


def triangular_basis_areas(knots: Tensor) -> Tensor:
    if knots.ndim != 1 or knots.numel() < 3:
        raise ValueError("density knots must be a one-dimensional sequence")
    gaps = knots[1:] - knots[:-1]
    if not bool(torch.isfinite(knots).all()) or bool((gaps <= 0).any()):
        raise ValueError("density knots must be finite and strictly increasing")
    areas = torch.empty_like(knots)
    areas[0] = gaps[0] / 2
    areas[-1] = gaps[-1] / 2
    areas[1:-1] = (gaps[:-1] + gaps[1:]) / 2
    return areas


def transform_returns_to_unit(
    returns: Tensor,
    transform: ReturnTransform,
) -> tuple[Tensor, Tensor]:
    """Return transformed coordinates and log |du/dr| for raw log returns."""
    transform.validate()
    values = returns.float()
    z = (
        values * RETURN_TO_BPS - transform.location_bps
    ) / transform.scale_bps
    latent = transform.alpha * torch.asinh(z)
    unit = torch.sigmoid(latent)
    # Keep the density calculation finite for representable extreme values.
    epsilon = torch.finfo(unit.dtype).eps
    stable_unit = unit.clamp(epsilon, 1 - epsilon)
    log_jacobian = (
        math.log(
            transform.alpha * RETURN_TO_BPS / transform.scale_bps
        )
        + torch.log(stable_unit)
        + torch.log1p(-stable_unit)
        - 0.5 * torch.log1p(z.square())
    )
    return stable_unit, log_jacobian


def inverse_unit_to_returns(
    unit: np.ndarray,
    transform: ReturnTransform,
) -> np.ndarray:
    values = np.asarray(unit, dtype=np.float64)
    logit = np.log(values) - np.log1p(-values)
    return (
        transform.location_bps
        + transform.scale_bps * np.sinh(logit / transform.alpha)
    ) / RETURN_TO_BPS


def component_log_masses(raw_density_logits: Tensor, areas: Tensor) -> Tensor:
    """Normalize knot-height logits and return their triangular-basis masses.

    If ``h_i = exp(logit_i) / sum_j(A_j * exp(logit_j))``, the mass carried by
    basis ``i`` is ``q_i = A_i * h_i``.  The area therefore appears once in
    the shared density normalizer; this log-mass form is convenient for NLL
    evaluation and expectations.
    """
    if raw_density_logits.shape[-1] != areas.numel():
        raise ValueError("one density logit is required for every knot")
    return torch.log_softmax(
        raw_density_logits.float() + torch.log(areas.float()), dim=-1
    )


def interpolated_log_density_unit(
    log_masses: Tensor,
    unit_targets: Tensor,
    knots: Tensor,
    areas: Tensor,
) -> Tensor:
    """Evaluate the normalized piecewise-linear density at each target."""
    if log_masses.ndim < 2 \
            or unit_targets.shape != log_masses.shape[:-1] \
            or knots.shape != areas.shape \
            or log_masses.shape[-1] != knots.numel():
        raise ValueError("invalid knot-density evaluation shapes")
    output_shape = unit_targets.shape
    targets = unit_targets.float().reshape(-1).clamp(0, 1)
    flat_log_masses = log_masses.float().reshape(-1, log_masses.shape[-1])
    knots = knots.float()
    areas = areas.float()
    interval = torch.bucketize(targets.contiguous(), knots[1:-1].contiguous())
    left = knots[interval]
    right = knots[interval + 1]
    fraction = ((targets - left) / (right - left)).clamp(0, 1)
    log_heights = flat_log_masses - torch.log(areas)
    left_log_height = log_heights.gather(1, interval[:, None]).squeeze(1)
    right_log_height = log_heights.gather(
        1, (interval + 1)[:, None]
    ).squeeze(1)
    return torch.logaddexp(
        left_log_height + torch.log1p(-fraction),
        right_log_height + torch.log(fraction),
    ).reshape(output_shape)


def return_negative_log_likelihood(
    raw_density_logits: Tensor,
    targets: Tensor,
    knots: Tensor,
    areas: Tensor,
    transform: ReturnTransform,
) -> tuple[Tensor, Tensor, Tensor]:
    log_masses = component_log_masses(raw_density_logits, areas)
    unit, log_jacobian = transform_returns_to_unit(targets, transform)
    log_density_unit = interpolated_log_density_unit(
        log_masses, unit, knots, areas
    )
    return -(log_density_unit + log_jacobian), log_masses, log_density_unit


def component_return_means(
    knots: np.ndarray,
    transform: ReturnTransform,
    *,
    quadrature_order: int = 64,
) -> np.ndarray:
    """Integrate r(u) against every normalized triangular basis."""
    if quadrature_order < 8:
        raise ValueError("return expectation quadrature needs at least 8 points")
    knot_tensor = torch.from_numpy(np.asarray(knots, dtype=np.float64))
    areas = triangular_basis_areas(knot_tensor).numpy()
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    result = np.zeros(knots.size, dtype=np.float64)
    for interval in range(knots.size - 1):
        left = float(knots[interval])
        right = float(knots[interval + 1])
        midpoint = (left + right) / 2
        half_width = (right - left) / 2
        unit = midpoint + half_width * nodes
        returns = inverse_unit_to_returns(unit, transform)
        fraction = (unit - left) / (right - left)
        scaled_weights = weights * half_width
        result[interval] += float(
            np.sum(scaled_weights * returns * (1 - fraction))
        )
        result[interval + 1] += float(
            np.sum(scaled_weights * returns * fraction)
        )
    return result / areas


def mode_grid(
    knots: np.ndarray,
    transform: ReturnTransform,
    *,
    points_per_interval: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute candidates for the mode of the density in return space."""
    if points_per_interval < 2:
        raise ValueError("mode grid needs at least two points per interval")
    fractions = np.arange(points_per_interval, dtype=np.float64) \
        / points_per_interval
    unit = np.concatenate([
        left + (right - left) * fractions
        for left, right in zip(knots[:-1], knots[1:], strict=True)
    ] + [np.asarray([1.0], dtype=np.float64)])
    epsilon = np.finfo(np.float64).eps
    stable = np.clip(unit, epsilon, 1 - epsilon)
    z = np.sinh(
        (np.log(stable) - np.log1p(-stable)) / transform.alpha
    )
    returns = (
        transform.location_bps + transform.scale_bps * z
    ) / RETURN_TO_BPS
    jacobian = (
        transform.alpha * RETURN_TO_BPS / transform.scale_bps
        * stable * (1 - stable)
        / np.sqrt(1 + z * z)
    )
    interval = np.searchsorted(knots[1:-1], unit, side="right")
    left = knots[interval]
    right = knots[interval + 1]
    fraction = np.clip((unit - left) / (right - left), 0, 1)
    return returns, jacobian, interval.astype(np.int64), fraction
