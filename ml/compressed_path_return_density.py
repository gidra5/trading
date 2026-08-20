from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from exact_tensor_return_density import TensorPathGluBlock
from return_knot_density import (
    KnotDensityContract,
    component_return_means,
    interpolated_log_density_unit,
    transform_returns_to_unit,
    triangular_basis_areas,
)


ARCHITECTURE_CONTRACT = (
    "compressed-fifteen-active-return-recurrent-low-rank-path-density-v3"
)


@dataclass(frozen=True)
class CompressedPathOutput:
    log_masses: tuple[Tensor, ...]
    expectations: Tensor
    knots_unit: tuple[Tensor, ...] | None = None
    areas_unit: tuple[Tensor, ...] | None = None
    component_means: tuple[Tensor, ...] | None = None
    arithmetic_component_means: tuple[Tensor, ...] | None = None


def batched_triangular_basis_areas(knots: Tensor) -> Tensor:
    """Return triangular-basis areas for one ordered knot grid per example."""
    if knots.ndim != 2 or knots.shape[1] < 3:
        raise ValueError("dynamic density knots must have shape [batch, knots]")
    gaps = knots[:, 1:] - knots[:, :-1]
    if not bool(torch.isfinite(knots).all()) or bool((gaps <= 0).any()):
        raise ValueError("dynamic density knots must be finite and increasing")
    areas = torch.empty_like(knots)
    areas[:, 0] = gaps[:, 0] / 2
    areas[:, -1] = gaps[:, -1] / 2
    areas[:, 1:-1] = (gaps[:, :-1] + gaps[:, 1:]) / 2
    return areas


def batched_interpolated_log_density_unit(
    log_masses: Tensor,
    unit_targets: Tensor,
    knots: Tensor,
    areas: Tensor,
) -> Tensor:
    """Evaluate a piecewise-linear density on a per-example knot grid."""
    if log_masses.ndim != 2 or unit_targets.ndim != 1 \
            or knots.shape != log_masses.shape \
            or areas.shape != knots.shape \
            or unit_targets.shape[0] != log_masses.shape[0]:
        raise ValueError("invalid dynamic knot-density evaluation shapes")
    targets = unit_targets.float().clamp(0, 1)
    # Searchsorted's discrete interval choice need not be differentiable; the
    # selected endpoints and interpolation fraction remain differentiable.
    interval = (targets[:, None] >= knots[:, 1:-1]).sum(dim=1)
    left = knots.gather(1, interval[:, None]).squeeze(1)
    right = knots.gather(1, (interval + 1)[:, None]).squeeze(1)
    fraction = ((targets - left) / (right - left)).clamp(0, 1)
    epsilon = torch.finfo(fraction.dtype).eps
    stable_fraction = fraction.clamp(epsilon, 1 - epsilon)
    log_heights = log_masses.float() - torch.log(areas.float())
    left_log_height = log_heights.gather(
        1, interval[:, None]
    ).squeeze(1)
    right_log_height = log_heights.gather(
        1, (interval + 1)[:, None]
    ).squeeze(1)
    return torch.logaddexp(
        left_log_height + torch.log1p(-stable_fraction),
        right_log_height + torch.log(stable_fraction),
    )


def _identity_output(block: TensorPathGluBlock) -> None:
    """Make a GLU a live representation transform instead of a zero head."""
    with torch.no_grad():
        nn.init.eye_(block.output.weight)


class CompressedPathReturnDensity(nn.Module):
    """Recurrently propagate a compressed distribution over return histories.

    Stage ``s`` owns two normalized vectors with width ``Q_s``:

    * ``q_s`` is the compressed distribution over histories reaching the stage;
    * ``g_s`` is emitted by the recurrent distribution-state embedding.

    A learned constant destination factor ``b_s`` expands ``g_s`` into the
    rank-one transition-logit matrix ``g_s b_s^T``. Row-wise softmax gives a
    valid transition matrix ``T_s`` and ``p_s = q_s T_s`` is the predicted
    marginal distribution for return ``s``. The next compressed history is
    produced from exactly ``concat(q_s, g_s)``. The final Q projection is
    therefore consumed by the final return head; no terminal state is made.
    """

    architecture_contract = ARCHITECTURE_CONTRACT

    def __init__(
        self,
        feature_mean: Tensor,
        feature_std: Tensor,
        densities: tuple[KnotDensityContract, ...],
        *,
        market_width: int,
        state_widths: tuple[int, ...],
        initial_radius: float,
        minimum_radius: float,
        learnable_centering: bool,
    ) -> None:
        super().__init__()
        if feature_mean.ndim != 1 or feature_std.shape != feature_mean.shape \
                or bool((feature_std <= 0).any()):
            raise ValueError("invalid immediate-feature normalization")
        if len(densities) != len(state_widths) or len(densities) < 2:
            raise ValueError("one density grid is required for every path step")
        if any(len(density.knots_unit) != width for density, width in zip(
            densities, state_widths, strict=True
        )):
            raise ValueError("density knot counts must match path-state widths")
        for density in densities:
            density.validate()
        self.return_count = len(state_widths)
        self.market_width = int(market_width)
        self.state_widths = tuple(int(value) for value in state_widths)
        self.density_transform = densities[0].transform
        if any(density.transform != self.density_transform for density in densities):
            raise ValueError("all path steps must share one return transform")
        self.register_buffer("feature_mean", feature_mean.float().clone())
        self.register_buffer("feature_std", feature_std.float().clone())

        self.market = TensorPathGluBlock(
            feature_mean.numel(), self.market_width, self.market_width,
            initial_radius=initial_radius,
            minimum_radius=minimum_radius,
            learnable_centering=learnable_centering,
            output_bias=torch.zeros(self.market_width),
        )
        _identity_output(self.market)

        first_width = self.state_widths[0]
        first_prior = torch.from_numpy(
            densities[0].prior_component_masses
        ).float()
        self.initial_distribution_state = TensorPathGluBlock(
            self.market_width, first_width, first_width,
            initial_radius=initial_radius,
            minimum_radius=minimum_radius,
            learnable_centering=learnable_centering,
            output_bias=torch.zeros(first_width),
        )
        _identity_output(self.initial_distribution_state)
        self.initial_history = TensorPathGluBlock(
            self.market_width, first_width, first_width,
            initial_radius=initial_radius,
            minimum_radius=minimum_radius,
            learnable_centering=learnable_centering,
            output_bias=torch.log(first_prior),
        )

        self.distribution_state_blocks = nn.ModuleList()
        self.transition_factor_blocks = nn.ModuleList()
        self.history_down_projects = nn.ModuleList()
        self.destination_factors = nn.ParameterList()

        for step, (width, density) in enumerate(zip(
            self.state_widths, densities, strict=True
        )):
            factor = TensorPathGluBlock(
                width, width, width,
                initial_radius=initial_radius,
                minimum_radius=minimum_radius,
                learnable_centering=learnable_centering,
                output_bias=torch.ones(width),
            )
            _identity_output(factor)
            self.transition_factor_blocks.append(factor)
            self.destination_factors.append(nn.Parameter(torch.log(
                torch.from_numpy(density.prior_component_masses).float()
            )))

            if step + 1 == self.return_count:
                continue
            next_width = self.state_widths[step + 1]
            state_block = TensorPathGluBlock(
                width, next_width, next_width,
                initial_radius=initial_radius,
                minimum_radius=minimum_radius,
                learnable_centering=learnable_centering,
                output_bias=torch.zeros(next_width),
            )
            _identity_output(state_block)
            self.distribution_state_blocks.append(state_block)
            next_prior = torch.from_numpy(
                densities[step + 1].prior_component_masses
            ).float()
            history_block = TensorPathGluBlock(
                2 * width, next_width, next_width,
                initial_radius=initial_radius,
                minimum_radius=minimum_radius,
                learnable_centering=learnable_centering,
                output_bias=torch.log(next_prior),
            )
            _identity_output(history_block)
            self.history_down_projects.append(history_block)

        for step, density in enumerate(densities):
            knots = torch.from_numpy(density.knots_unit).float()
            self.register_buffer(f"knots_{step}", knots)
            self.register_buffer(
                f"areas_{step}", triangular_basis_areas(knots)
            )
            self.register_buffer(
                f"means_{step}",
                torch.from_numpy(component_return_means(
                    density.knots_unit, density.transform
                )).float(),
            )

    def muon_parameters(self) -> tuple[Tensor, ...]:
        blocks = (
            self.market,
            self.initial_distribution_state,
            self.initial_history,
            *self.distribution_state_blocks,
            *self.transition_factor_blocks,
            *self.history_down_projects,
        )
        return tuple(
            parameter for block in blocks for parameter in block.muon_parameters()
        )

    def forward(self, features: Tensor) -> CompressedPathOutput:
        if features.ndim != 2 or features.shape[1] != self.feature_mean.numel():
            raise ValueError("compressed path model expects [batch, feature] input")
        normalized = (features.float() - self.feature_mean) / self.feature_std
        market = self.market(normalized)
        distribution_state = self.initial_distribution_state(market)
        q = torch.softmax(self.initial_history(market), dim=1)
        log_masses: list[Tensor] = []
        expectations: list[Tensor] = []

        for step, (factor_block, destination_factor) in enumerate(zip(
            self.transition_factor_blocks,
            self.destination_factors,
            strict=True,
        )):
            transition_factor = factor_block(distribution_state)
            transition_logits = (
                transition_factor[:, :, None]
                * destination_factor.float()[None, None, :]
            )
            transition = torch.softmax(transition_logits, dim=2)
            marginal = torch.bmm(q[:, None, :], transition).squeeze(1)
            marginal = marginal / marginal.sum(dim=1, keepdim=True).clamp_min(1e-12)
            log_mass = torch.log(marginal.clamp_min(1e-30))
            log_masses.append(log_mass)
            expectations.append(marginal @ self.means(step))

            if step + 1 < self.return_count:
                q = torch.softmax(self.history_down_projects[step](
                    torch.cat((q, transition_factor), dim=1)
                ), dim=1)
                distribution_state = self.distribution_state_blocks[step](
                    distribution_state
                )

        return CompressedPathOutput(
            log_masses=tuple(log_masses),
            expectations=torch.stack(expectations, dim=1),
        )

    def knots(self, step: int) -> Tensor:
        return getattr(self, f"knots_{step}")

    def areas(self, step: int) -> Tensor:
        return getattr(self, f"areas_{step}")

    def means(self, step: int) -> Tensor:
        return getattr(self, f"means_{step}")


def path_log_density_terms(
    output: CompressedPathOutput,
    targets: Tensor,
    model: CompressedPathReturnDensity,
) -> Tensor:
    if targets.ndim != 2 or targets.shape[1] != model.return_count:
        raise ValueError("path targets have the wrong shape")
    unit, log_jacobian = transform_returns_to_unit(
        targets, model.density_transform
    )
    terms = []
    for step, log_masses in enumerate(output.log_masses):
        if output.knots_unit is None or output.areas_unit is None:
            unit_log_density = interpolated_log_density_unit(
                log_masses, unit[:, step], model.knots(step), model.areas(step)
            )
        else:
            unit_log_density = batched_interpolated_log_density_unit(
                log_masses,
                unit[:, step],
                output.knots_unit[step],
                output.areas_unit[step],
            )
        terms.append(unit_log_density + log_jacobian[:, step])
    return torch.stack(terms, dim=1)
