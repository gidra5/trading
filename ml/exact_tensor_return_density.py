from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from next_return_dataset import HISTORY_RETURN_COUNT
from return_knot_density import (
    KnotDensityContract,
    component_log_masses,
    component_return_means,
    transform_returns_to_unit,
    triangular_basis_areas,
)
from return_oracle_ce import LearnableCenteringNorm, fused_glu


ARCHITECTURE_CONTRACT = (
    "three-return-exact-32-knot-joint-tensor-three-single-layer-"
    "normalized-glu-blocks-v1"
)


class TensorPathGluBlock(nn.Module):
    def __init__(
        self,
        input_width: int,
        hidden_width: int,
        output_width: int,
        *,
        initial_radius: float,
        minimum_radius: float,
        learnable_centering: bool,
        output_bias: Tensor,
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(input_width, 2 * hidden_width)
        self.value_centering = LearnableCenteringNorm(
            hidden_width,
            denominator_family="sqrt",
            initial_scale=initial_radius,
            minimum_scale=minimum_radius,
            learnable_centering=learnable_centering,
        )
        self.gate_centering = LearnableCenteringNorm(
            hidden_width,
            denominator_family="sqrt",
            initial_scale=initial_radius,
            minimum_scale=minimum_radius,
            learnable_centering=learnable_centering,
        )
        self.value_bias = nn.Parameter(torch.zeros(hidden_width))
        self.gate_bias = nn.Parameter(torch.zeros(hidden_width))
        self.value_transform = nn.Linear(hidden_width, hidden_width, bias=False)
        self.gate_transform = nn.Linear(hidden_width, hidden_width, bias=False)
        self.output = nn.Linear(hidden_width, output_width)
        nn.init.kaiming_normal_(self.projection.weight, nonlinearity="linear")
        nn.init.zeros_(self.projection.bias)
        nn.init.eye_(self.value_transform.weight)
        nn.init.eye_(self.gate_transform.weight)
        nn.init.zeros_(self.output.weight)
        with torch.no_grad():
            self.output.bias.copy_(output_bias)

    def forward(self, values: Tensor) -> Tensor:
        hidden, _raw_value, _raw_gate = fused_glu(
            self.projection(values),
            self.value_centering,
            self.gate_centering,
            self.value_bias,
            self.gate_bias,
            self.value_transform,
            self.gate_transform,
        )
        return self.output(hidden)

    def muon_parameters(self) -> tuple[Tensor, ...]:
        return (
            self.projection.weight,
            self.value_transform.weight,
            self.gate_transform.weight,
        )


@dataclass(frozen=True)
class ExactTensorPathOutput:
    conditional_log_masses: tuple[Tensor, Tensor, Tensor]
    joint_component_masses: tuple[Tensor, Tensor, Tensor]
    expectations: Tensor


def temperature_scaled_output(
    output: ExactTensorPathOutput,
    model: "ExactTensorReturnDensity",
    temperature: float | Tensor,
) -> ExactTensorPathOutput:
    """Rescale every normalized conditional mass distribution."""
    value = torch.as_tensor(
        temperature,
        dtype=output.conditional_log_masses[0].dtype,
        device=output.conditional_log_masses[0].device,
    )
    if value.numel() != 1 or not bool(torch.isfinite(value)) \
            or not bool(value > 0):
        raise ValueError("exact tensor temperature must be finite and positive")
    conditional = tuple(
        torch.log_softmax(log_masses / value, dim=-1)
        for log_masses in output.conditional_log_masses
    )
    joint1 = conditional[0].exp()
    joint2 = joint1[:, :, None] * conditional[1].exp()
    joint3 = joint2[:, :, :, None] * conditional[2].exp()
    joints = (joint1, joint2, joint3)
    marginals = (
        joint1,
        joint2.sum(dim=1),
        joint3.sum(dim=(1, 2)),
    )
    expectations = torch.stack(tuple(
        marginal @ model.density_component_return_means
        for marginal in marginals
    ), dim=1)
    return ExactTensorPathOutput(
        conditional_log_masses=conditional,
        joint_component_masses=joints,
        expectations=expectations,
    )


class ExactTensorReturnDensity(nn.Module):
    architecture_contract = ARCHITECTURE_CONTRACT
    return_count = 3

    def __init__(
        self,
        feature_mean: Tensor,
        feature_std: Tensor,
        density: KnotDensityContract,
        *,
        hidden_width: int = 512,
        initial_radius: float,
        minimum_radius: float,
        learnable_centering: bool,
    ) -> None:
        super().__init__()
        density.validate()
        knot_count = int(density.knots_unit.size)
        if knot_count != 32:
            raise ValueError("exact tensor experiment requires 32 knots")
        if feature_mean.shape != (HISTORY_RETURN_COUNT,) \
                or feature_std.shape != feature_mean.shape:
            raise ValueError("exact tensor model requires 120 feature statistics")
        self.knot_count = knot_count
        self.hidden_width = int(hidden_width)
        self.density_transform = density.transform
        knots = torch.from_numpy(density.knots_unit).float()
        areas = triangular_basis_areas(knots)
        prior_masses = torch.from_numpy(
            density.prior_component_masses
        ).float()
        component_means = torch.from_numpy(component_return_means(
            density.knots_unit, density.transform
        )).float()
        self.register_buffer("feature_mean", feature_mean.float().clone())
        self.register_buffer("feature_std", feature_std.float().clone())
        self.register_buffer("density_knots_unit", knots)
        self.register_buffer("density_basis_areas", areas)
        self.register_buffer("density_prior_masses", prior_masses)
        self.register_buffer(
            "density_component_return_means", component_means
        )
        prior_height_logits = torch.log(prior_masses / areas)
        blocks: list[TensorPathGluBlock] = []
        for rank in range(1, self.return_count + 1):
            prefix_width = 0 if rank == 1 else knot_count ** (rank - 1)
            output_width = knot_count ** rank
            bias = prior_height_logits.repeat(knot_count ** (rank - 1))
            blocks.append(TensorPathGluBlock(
                HISTORY_RETURN_COUNT + prefix_width,
                self.hidden_width,
                output_width,
                initial_radius=initial_radius,
                minimum_radius=minimum_radius,
                learnable_centering=learnable_centering,
                output_bias=bias,
            ))
        self.blocks = nn.ModuleList(blocks)

    def muon_parameters(self) -> tuple[Tensor, ...]:
        return tuple(
            parameter
            for block in self.blocks
            for parameter in block.muon_parameters()
        )

    def _normalized_history(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != HISTORY_RETURN_COUNT:
            raise ValueError("exact tensor model expects [batch, 120] histories")
        return (features.float() - self.feature_mean) / self.feature_std

    def forward(self, features: Tensor) -> ExactTensorPathOutput:
        history = self._normalized_history(features)
        batch = history.shape[0]
        logits1 = self.blocks[0](history).reshape(batch, self.knot_count)
        log_mass1 = component_log_masses(logits1, self.density_basis_areas)
        joint1 = log_mass1.exp()

        block2_input = torch.cat((
            history,
            joint1 * self.knot_count,
        ), dim=1)
        logits2 = self.blocks[1](block2_input).reshape(
            batch, self.knot_count, self.knot_count
        )
        log_mass2 = component_log_masses(logits2, self.density_basis_areas)
        joint2 = joint1[:, :, None] * log_mass2.exp()

        block3_input = torch.cat((
            history,
            joint2.reshape(batch, -1) * self.knot_count ** 2,
        ), dim=1)
        logits3 = self.blocks[2](block3_input).reshape(
            batch, self.knot_count, self.knot_count, self.knot_count
        )
        log_mass3 = component_log_masses(logits3, self.density_basis_areas)
        joint3 = joint2[:, :, :, None] * log_mass3.exp()

        marginals = (
            joint1,
            joint2.sum(dim=1),
            joint3.sum(dim=(1, 2)),
        )
        expectations = torch.stack(tuple(
            marginal @ self.density_component_return_means
            for marginal in marginals
        ), dim=1)
        return ExactTensorPathOutput(
            conditional_log_masses=(log_mass1, log_mass2, log_mass3),
            joint_component_masses=(joint1, joint2, joint3),
            expectations=expectations,
        )


def multilinear_grid_value(values: Tensor, coordinates: Tensor, knots: Tensor) -> Tensor:
    """Interpolate one K-sized tensor axis for every supplied coordinate."""
    if values.ndim != coordinates.shape[1] + 1 \
            or values.shape[0] != coordinates.shape[0]:
        raise ValueError("grid values and coordinates have incompatible ranks")
    current = values
    for axis in range(coordinates.shape[1]):
        coordinate = coordinates[:, axis].clamp(knots[0], knots[-1])
        right = torch.searchsorted(knots, coordinate, right=True).clamp(
            1, knots.numel() - 1
        )
        left = right - 1
        fraction = (coordinate - knots[left]) / (knots[right] - knots[left])
        remaining_shape = current.shape[2:]
        gather_shape = (current.shape[0], 1, *remaining_shape)
        left_index = left.reshape(current.shape[0], 1, *([1] * len(remaining_shape)))
        right_index = right.reshape(current.shape[0], 1, *([1] * len(remaining_shape)))
        left_value = torch.gather(
            current, 1, left_index.expand(gather_shape)
        ).squeeze(1)
        right_value = torch.gather(
            current, 1, right_index.expand(gather_shape)
        ).squeeze(1)
        blend_shape = (current.shape[0], *([1] * len(remaining_shape)))
        blend = fraction.reshape(blend_shape)
        current = left_value * (1 - blend) + right_value * blend
    return current


def exact_tensor_path_log_density(
    output: ExactTensorPathOutput,
    targets: Tensor,
    model: ExactTensorReturnDensity,
) -> Tensor:
    if targets.ndim != 2 or targets.shape[1] != model.return_count:
        raise ValueError("exact tensor targets must contain three returns")
    unit, log_jacobian = transform_returns_to_unit(
        targets, model.density_transform
    )
    joint_log_densities: list[Tensor] = []
    for rank, joint_masses in enumerate(output.joint_component_masses, start=1):
        area_product = model.density_basis_areas
        for _axis in range(1, rank):
            area_product = area_product[..., None] \
                * model.density_basis_areas
        joint_heights = joint_masses / area_product
        density_unit = multilinear_grid_value(
            joint_heights, unit[:, :rank], model.density_knots_unit
        ).clamp_min(torch.finfo(joint_heights.dtype).tiny)
        joint_log_densities.append(
            torch.log(density_unit) + log_jacobian[:, :rank].sum(dim=1)
        )
    conditional_terms = [joint_log_densities[0]]
    conditional_terms.extend(
        joint_log_densities[index] - joint_log_densities[index - 1]
        for index in range(1, len(joint_log_densities))
    )
    return torch.stack(conditional_terms, dim=1)
