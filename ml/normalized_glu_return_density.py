from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from normalized_glu_next_return import NormalizedGluNextReturn
from return_knot_density import (
    KnotDensityContract,
    component_log_masses,
    component_return_means,
    mode_grid,
    triangular_basis_areas,
)


ARCHITECTURE_CONTRACT = (
    "next-return-fixed-knot-piecewise-linear-density-normalized-glu-v1"
)


class NormalizedGluReturnDensity(NormalizedGluNextReturn):
    architecture_contract = ARCHITECTURE_CONTRACT

    def __init__(
        self,
        feature_mean: Tensor,
        feature_std: Tensor,
        density: KnotDensityContract,
        return_count: int = 1,
        **kwargs,
    ) -> None:
        density.validate()
        if isinstance(return_count, bool) or int(return_count) < 1:
            raise ValueError("density return count must be positive")
        self.return_count = int(return_count)
        knot_count = int(density.knots_unit.size)
        self.knot_count = knot_count
        super().__init__(
            feature_mean,
            feature_std,
            torch.zeros(knot_count * self.return_count),
            torch.ones(knot_count * self.return_count),
            **kwargs,
        )
        knots = torch.from_numpy(density.knots_unit).float()
        areas = triangular_basis_areas(knots)
        prior_masses = torch.from_numpy(
            density.prior_component_masses
        ).float()
        means = torch.from_numpy(component_return_means(
            density.knots_unit, density.transform
        )).float()
        mode_returns, mode_jacobian, mode_intervals, mode_fractions = mode_grid(
            density.knots_unit, density.transform
        )
        self.register_buffer("density_knots_unit", knots)
        self.register_buffer("density_basis_areas", areas)
        self.register_buffer("density_prior_masses", prior_masses)
        self.register_buffer("density_component_return_means", means)
        self.register_buffer(
            "density_mode_returns", torch.from_numpy(mode_returns).float()
        )
        self.register_buffer(
            "density_mode_jacobian", torch.from_numpy(mode_jacobian).float()
        )
        self.register_buffer(
            "density_mode_intervals", torch.from_numpy(mode_intervals).long()
        )
        self.register_buffer(
            "density_mode_fractions", torch.from_numpy(mode_fractions).float()
        )
        self.prefix_conditioner = torch.nn.Parameter(
            torch.zeros(max(0, self.return_count - 1), 2, knot_count)
        )
        self.density_transform = density.transform
        self.density_source_file = density.source_file
        self.density_source_fit = density.source_fit
        # Raw outputs are log density heights. The shared normalizer is
        # sum_i(area_i * exp(logit_i)); expressing its result as component
        # masses is convenient but does not apply area twice. This
        # initialization exactly reconstructs the selected global density.
        prior_heights = prior_masses / areas
        with torch.no_grad():
            self.output.weight.zero_()
            self.output.bias.copy_(
                torch.log(prior_heights).repeat(self.return_count)
            )

    def _base_density_logits(self, features: Tensor) -> Tensor:
        logits = super().forward_standardized(features)
        if self.return_count == 1:
            return logits
        return logits.reshape(features.shape[0], self.return_count, self.knot_count)

    def _prefix_features(self, returns: Tensor) -> Tensor:
        z = (
            returns.float() * 10_000.0
            - self.density_transform.location_bps
        ) / self.density_transform.scale_bps
        signed = torch.tanh(torch.asinh(z) / self.density_transform.alpha)
        return torch.stack((signed, signed.abs()), dim=-1)

    def _conditioned_logits(
        self, base_logits: Tensor, prefix_returns: Tensor
    ) -> Tensor:
        if self.return_count == 1:
            return base_logits
        if prefix_returns.shape != base_logits.shape[:-1]:
            raise ValueError("path prefix must contain one return per path lead")
        prefix = self._prefix_features(prefix_returns)
        conditioned: list[Tensor] = []
        for lead in range(self.return_count):
            value = base_logits[:, lead]
            for previous in range(lead):
                lag = lead - previous - 1
                value = value + prefix[:, previous] @ self.prefix_conditioner[lag]
            conditioned.append(value)
        return torch.stack(conditioned, dim=1)

    def teacher_forced_density_logits(
        self, features: Tensor, targets: Tensor
    ) -> Tensor:
        """Evaluate the joint path density using each realized prefix."""
        base_logits = self._base_density_logits(features)
        if self.return_count == 1:
            return base_logits
        return self._conditioned_logits(base_logits, targets)

    def raw_density_logits(self, features: Tensor) -> Tensor:
        """Roll out conditional logits along the recursively expected path."""
        base_logits = self._base_density_logits(features)
        if self.return_count == 1:
            return base_logits
        generated: list[Tensor] = []
        logits: list[Tensor] = []
        for lead in range(self.return_count):
            value = base_logits[:, lead]
            for previous, previous_return in enumerate(generated):
                lag = lead - previous - 1
                prefix = self._prefix_features(previous_return)
                value = value + prefix @ self.prefix_conditioner[lag]
            log_masses = component_log_masses(
                value, self.density_basis_areas
            )
            expectation = (
                log_masses.exp() @ self.density_component_return_means
            )
            logits.append(value)
            generated.append(expectation)
        return torch.stack(logits, dim=1)

    def forward(self, features: Tensor) -> Tensor:
        return self.raw_density_logits(features)

    def log_component_masses(self, features: Tensor) -> Tensor:
        return component_log_masses(
            self.raw_density_logits(features), self.density_basis_areas
        )

    def point_predictions_from_log_masses(
        self,
        log_masses: Tensor,
        *,
        include_mode: bool = True,
        mode_candidate_batch_size: int = 256,
    ) -> tuple[Tensor, Tensor | None]:
        masses = log_masses.exp()
        expectation = masses @ self.density_component_return_means
        if not include_mode:
            return expectation, None
        if mode_candidate_batch_size < 1:
            raise ValueError("mode candidate batch size must be positive")
        leading_shape = masses.shape[:-1]
        flat_masses = masses.reshape(-1, masses.shape[-1])
        heights = flat_masses / self.density_basis_areas
        best_density = torch.full(
            (flat_masses.shape[0],), -torch.inf,
            dtype=heights.dtype, device=heights.device,
        )
        selected = torch.zeros(
            flat_masses.shape[0], dtype=torch.long, device=heights.device
        )
        candidate_count = int(self.density_mode_returns.numel())
        for start in range(0, candidate_count, mode_candidate_batch_size):
            end = min(candidate_count, start + mode_candidate_batch_size)
            interval = self.density_mode_intervals[start:end]
            fraction = self.density_mode_fractions[start:end]
            density_unit = (
                heights[:, interval] * (1 - fraction)
                + heights[:, interval + 1] * fraction
            )
            density_return = (
                density_unit * self.density_mode_jacobian[start:end]
            )
            block_density, block_index = density_return.max(dim=1)
            improved = block_density > best_density
            best_density = torch.where(improved, block_density, best_density)
            selected = torch.where(improved, block_index + start, selected)
        mode = self.density_mode_returns[selected].reshape(leading_shape)
        return expectation, mode

    def prior_point_predictions(
        self,
    ) -> tuple[float, float] | tuple[Tensor, Tensor]:
        log_masses = torch.log(self.density_prior_masses)[None, None, :].expand(
            1, self.return_count, -1
        )
        expectation, mode = self.point_predictions_from_log_masses(log_masses)
        if mode is None:
            raise RuntimeError("prior mode was not evaluated")
        if self.return_count == 1:
            return float(expectation[0, 0]), float(mode[0, 0])
        return expectation[0], mode[0]
