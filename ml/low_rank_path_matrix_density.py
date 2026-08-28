from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from compressed_path_return_density import (
    CompressedPathOutput,
    batched_triangular_basis_areas,
)
from exact_tensor_return_density import TensorPathGluBlock
from return_knot_density import (
    RETURN_TO_BPS,
    KnotDensityContract,
    component_return_means,
    transform_returns_to_unit,
    triangular_basis_areas,
)
from return_oracle_ce import LearnableCenteringNorm


ARCHITECTURE_CONTRACT = "dynamic-low-rank-path-matrix-density-v1"
DIRECT_FACTORIZED_ARCHITECTURE_CONTRACT = (
    "direct-unpacked-factorized-path-matrix-density-v1"
)
CYCLIC_DENSE_COMPRESSED_ARCHITECTURE_CONTRACT = (
    "cyclic-dense-compressed-path-matrix-density-v1"
)
JOINT_PREFIX_CONTRACTED_CYCLIC_ARCHITECTURE_CONTRACT = (
    "joint-prefix-contracted-cyclic-path-matrix-density-v1"
)


@dataclass(frozen=True)
class CyclicPathState:
    """Complete recurrent state needed by one cyclic path-density step."""

    market: Tensor
    paths: Tensor
    normalization_location: Tensor | None
    normalization_scale: Tensor | None


def _batched_component_basis_densities(
    unit_targets: Tensor,
    knots: Tensor,
    areas: Tensor,
) -> Tensor:
    """Evaluate every normalized triangular component at one target.

    The returned rows have shape ``[batch, component]``.  Only the two basis
    functions adjacent to the target are nonzero.  Keeping the individual
    component likelihoods is what lets the joint-prefix forward contraction
    update latent prefix probabilities without sampling a return.
    """
    if unit_targets.ndim != 1 or knots.ndim != 2 or areas.shape != knots.shape \
            or unit_targets.shape[0] != knots.shape[0]:
        raise ValueError("invalid component-basis density shapes")
    targets = unit_targets.float().clamp(0, 1)
    interval = (targets[:, None] >= knots[:, 1:-1]).sum(dim=1)
    left = knots.gather(1, interval[:, None]).squeeze(1)
    right = knots.gather(1, (interval + 1)[:, None]).squeeze(1)
    fraction = ((targets - left) / (right - left)).clamp(0, 1)
    values = torch.zeros_like(knots)
    values.scatter_(
        1,
        interval[:, None],
        ((1 - fraction) / areas.gather(1, interval[:, None]).squeeze(1))[:, None],
    )
    values.scatter_add_(
        1,
        (interval + 1)[:, None],
        (fraction / areas.gather(1, (interval + 1)[:, None]).squeeze(1))[:, None],
    )
    return values


@dataclass(frozen=True)
class CyclicPathEmission:
    """One predicted return distribution and the state that emitted it."""

    conditioned_market: Tensor
    paths: Tensor
    log_masses: Tensor
    expectation: Tensor
    knots_unit: Tensor
    areas_unit: Tensor
    component_means: Tensor
    arithmetic_component_means: Tensor
    next_state: CyclicPathState


def _hidden_width(input_width: int, output_width: int, cap: int) -> int:
    return min((int(input_width) + int(output_width)) // 2, int(cap))


def _block(
    input_width: int,
    output_width: int,
    *,
    hidden_width_cap: int,
    initial_radius: float,
    minimum_radius: float,
    learnable_centering: bool,
    output_bias: Tensor,
    live_output: bool,
) -> TensorPathGluBlock:
    block = TensorPathGluBlock(
        input_width,
        _hidden_width(input_width, output_width, hidden_width_cap),
        output_width,
        initial_radius=initial_radius,
        minimum_radius=minimum_radius,
        learnable_centering=learnable_centering,
        output_bias=output_bias,
    )
    if live_output:
        with torch.no_grad():
            nn.init.eye_(block.output.weight)
    return block


class DynamicLowRankMatrixHead(nn.Module):
    """Generate ``base + A(x) @ B(x) / sqrt(rank)`` without emitting X*Y."""

    def __init__(
        self,
        input_width: int,
        rows: int,
        columns: int,
        rank: int,
        *,
        base: Tensor,
        hidden_width_cap: int,
        initial_radius: float,
        minimum_radius: float,
        learnable_centering: bool,
    ) -> None:
        super().__init__()
        if rows <= 0 or columns <= 0 or rank <= 0:
            raise ValueError("dynamic matrix dimensions must be positive")
        if base.shape != (rows, columns):
            raise ValueError("dynamic matrix base has the wrong shape")
        self.rows = int(rows)
        self.columns = int(columns)
        self.rank = int(rank)
        self.base = nn.Parameter(base.float().clone())
        left_count = self.rows * self.rank
        right = torch.randn(self.rank, self.columns) * 0.02
        output_bias = torch.cat((torch.zeros(left_count), right.flatten()))
        self.generator = _block(
            input_width,
            self.rank * (self.rows + self.columns),
            hidden_width_cap=hidden_width_cap,
            initial_radius=initial_radius,
            minimum_radius=minimum_radius,
            learnable_centering=learnable_centering,
            output_bias=output_bias,
            live_output=False,
        )
        # Make the conditional residual live without disturbing its near-base
        # initialization. Both factors must not start at exactly zero.
        nn.init.normal_(self.generator.output.weight, std=1e-4)

    def forward(self, values: Tensor) -> Tensor:
        factors = self.generator(values)
        split = self.rows * self.rank
        left = factors[:, :split].reshape(-1, self.rows, self.rank)
        right = factors[:, split:].reshape(-1, self.rank, self.columns)
        return self.base[None, :, :] + torch.bmm(left, right) / math.sqrt(
            self.rank
        )


class RankFactorizedLinear(nn.Module):
    """A dense-width linear map represented by independent rank factors."""

    def __init__(
        self,
        input_width: int,
        output_width: int,
        rank: int,
        *,
        bias: bool,
        residual_identity: bool = False,
        right_initial_std: float = 1.0,
    ) -> None:
        super().__init__()
        if input_width <= 0 or output_width <= 0 or rank <= 0:
            raise ValueError("factorized linear dimensions must be positive")
        if rank > min(input_width, output_width):
            raise ValueError("factor rank exceeds the matrix rank ceiling")
        if residual_identity and input_width != output_width:
            raise ValueError("residual identity requires a square linear map")
        self.input_width = int(input_width)
        self.output_width = int(output_width)
        self.rank = int(rank)
        self.residual_identity = bool(residual_identity)
        self.left = nn.Parameter(torch.empty(self.rank, self.input_width))
        self.right = nn.Parameter(torch.empty(self.output_width, self.rank))
        self.bias = nn.Parameter(torch.zeros(self.output_width)) if bias else None
        nn.init.normal_(self.left, std=1 / math.sqrt(self.input_width))
        nn.init.normal_(self.right, std=float(right_initial_std))

    def forward(self, values: Tensor) -> Tensor:
        low_rank = torch.nn.functional.linear(
            torch.nn.functional.linear(values, self.left),
            self.right,
        ) / math.sqrt(self.rank)
        if self.bias is not None:
            low_rank = low_rank + self.bias
        return values + low_rank if self.residual_identity else low_rank

    def muon_parameters(self) -> tuple[Tensor, Tensor]:
        return self.left, self.right


def _grouped_factorized_pair(
    first: RankFactorizedLinear,
    second: RankFactorizedLinear,
    first_values: Tensor,
    second_values: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Evaluate two independent factorized maps as two grouped GEMMs."""
    if first.input_width != second.input_width \
            or first.output_width != second.output_width \
            or first.rank != second.rank \
            or first.residual_identity != second.residual_identity:
        raise ValueError("grouped factorized maps must have matching shapes")
    if first_values.ndim != 2 or first_values.shape[1] != first.input_width:
        raise ValueError("grouped factorized input has the wrong shape")
    if second_values is None:
        inputs = first_values.unsqueeze(0).expand(2, -1, -1)
    else:
        if second_values.shape != first_values.shape:
            raise ValueError("grouped factorized branch inputs must match")
        inputs = torch.stack((first_values, second_values), dim=0)

    left = torch.stack((first.left, second.left), dim=0)
    latent = torch.bmm(inputs, left.transpose(1, 2))
    right = torch.stack((first.right, second.right), dim=0)
    outputs = torch.bmm(latent, right.transpose(1, 2)) / math.sqrt(first.rank)

    if first.bias is not None or second.bias is not None:
        if first.bias is None or second.bias is None:
            raise ValueError("grouped factorized biases must be present on both maps")
        outputs = outputs + torch.stack((first.bias, second.bias), dim=0)[:, None, :]
    if first.residual_identity:
        outputs = outputs + inputs
    return outputs[0], outputs[1]


class UnpackedFactorizedGluBlock(nn.Module):
    """Expected-width GNGLU with independently factorized branch matrices."""

    def __init__(
        self,
        input_width: int,
        hidden_width: int,
        output_width: int,
        rank: int,
        *,
        initial_radius: float,
        minimum_radius: float,
        learnable_centering: bool,
        output_bias: Tensor,
    ) -> None:
        super().__init__()
        if output_bias.shape != (output_width,):
            raise ValueError("factorized GNGLU output bias has the wrong shape")
        options = {"bias": True, "right_initial_std": 1.0}
        self.value_projection = RankFactorizedLinear(
            input_width, hidden_width, rank, **options,
        )
        self.gate_projection = RankFactorizedLinear(
            input_width, hidden_width, rank, **options,
        )
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
        self.value_transform = RankFactorizedLinear(
            hidden_width, hidden_width, rank,
            bias=False, residual_identity=True, right_initial_std=1e-3,
        )
        self.gate_transform = RankFactorizedLinear(
            hidden_width, hidden_width, rank,
            bias=False, residual_identity=True, right_initial_std=1e-3,
        )
        self.output = RankFactorizedLinear(
            hidden_width, output_width, rank,
            bias=True, right_initial_std=1e-4,
        )
        with torch.no_grad():
            assert self.output.bias is not None
            self.output.bias.copy_(output_bias)

    def forward(self, values: Tensor) -> Tensor:
        raw_value, raw_gate = _grouped_factorized_pair(
            self.value_projection,
            self.gate_projection,
            values,
        )
        value = self.value_centering(raw_value)
        gate = self.gate_centering(raw_gate)
        value, gate = _grouped_factorized_pair(
            self.value_transform,
            self.gate_transform,
            value,
            gate,
        )
        value = value + self.value_bias.to(dtype=value.dtype)
        gate = gate + self.gate_bias.to(dtype=gate.dtype)
        hidden = value * torch.sigmoid(gate)
        return self.output(hidden)

    def muon_parameters(self) -> tuple[Tensor, ...]:
        return (
            *self.value_projection.muon_parameters(),
            *self.gate_projection.muon_parameters(),
            *self.value_transform.muon_parameters(),
            *self.gate_transform.muon_parameters(),
            *self.output.muon_parameters(),
        )


class DirectFactorizedMatrixHead(nn.Module):
    """Emit a full matrix through an expected-width factorized GNGLU."""

    def __init__(
        self,
        input_width: int,
        rows: int,
        columns: int,
        rank: int,
        *,
        hidden_width_threshold: int,
        output_bias: Tensor,
        initial_radius: float,
        minimum_radius: float,
        learnable_centering: bool,
    ) -> None:
        super().__init__()
        self.rows = int(rows)
        self.columns = int(columns)
        output_width = self.rows * self.columns
        hidden_width = (int(input_width) + output_width) // 2
        block_options = {
            "initial_radius": initial_radius,
            "minimum_radius": minimum_radius,
            "learnable_centering": learnable_centering,
            "output_bias": output_bias.reshape(-1),
        }
        self.generator = (
            UnpackedFactorizedGluBlock(
                input_width,
                hidden_width,
                output_width,
                rank,
                **block_options,
            )
            if hidden_width > hidden_width_threshold else
            TensorPathGluBlock(
                input_width,
                hidden_width,
                output_width,
                **block_options,
            )
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.generator(values).reshape(
            -1, self.rows, self.columns
        )


class DynamicLowRankPathMatrixDensity(nn.Module):
    """Propagate a compressed joint path distribution through matrix heads.

    ``P_s`` contains D path embeddings of width Q. The market state emits a
    path-query matrix and a return-transition matrix as full learned bases plus
    rank-L conditional residuals. ``q_s`` scores paths, ``T_s`` gives the
    return distribution for each path, and ``J_s = q_s T_s`` is supplied to
    the next path projection together with ``P_s``, the market state, and the
    current dynamic output-point representation.
    """

    architecture_contract = ARCHITECTURE_CONTRACT

    def __init__(
        self,
        feature_mean: Tensor,
        feature_std: Tensor,
        density: KnotDensityContract,
        *,
        market_width: int,
        path_embedding_width: int,
        path_count: int,
        return_count: int,
        matrix_rank: int,
        hidden_width_cap: int,
        initial_radius: float,
        minimum_radius: float,
        learnable_centering: bool,
        quadrature_order: int = 16,
        target_normalization_variance_floor: float | None = None,
        target_normalization_center: bool = True,
    ) -> None:
        super().__init__()
        if feature_mean.ndim != 1 or feature_std.shape != feature_mean.shape \
                or bool((feature_std <= 0).any()):
            raise ValueError("invalid lagged-feature normalization")
        density.validate()
        if return_count < 1 or path_embedding_width <= 0 or path_count <= 0:
            raise ValueError("invalid path-matrix dimensions")
        if matrix_rank <= 0 or hidden_width_cap <= 0:
            raise ValueError("invalid path-matrix rank or hidden-width cap")
        if quadrature_order < 8:
            raise ValueError("dynamic density quadrature needs at least 8 points")

        self.return_count = int(return_count)
        self.market_width = int(market_width)
        self.path_embedding_width = int(path_embedding_width)
        self.path_count = int(path_count)
        self.matrix_rank = int(matrix_rank)
        self.hidden_width_cap = int(hidden_width_cap)
        self.output_width = len(density.knots_unit)
        if target_normalization_variance_floor is not None \
                and target_normalization_variance_floor <= 0:
            raise ValueError("target-normalization variance floor must be positive")
        self.target_normalization_variance_floor = (
            None if target_normalization_variance_floor is None
            else float(target_normalization_variance_floor)
        )
        self.target_normalization_center = bool(target_normalization_center)
        self.state_widths = (self.path_count,) * self.return_count
        self.density_transform = density.transform
        self.register_buffer("feature_mean", feature_mean.float().clone())
        self.register_buffer("feature_std", feature_std.float().clone())

        options = {
            "hidden_width_cap": self.hidden_width_cap,
            "initial_radius": initial_radius,
            "minimum_radius": minimum_radius,
            "learnable_centering": learnable_centering,
        }
        self.input_encoder = _block(
            feature_mean.numel(), self.market_width,
            output_bias=torch.zeros(self.market_width), live_output=True,
            **options,
        )
        self.market_transitions = nn.ModuleList([
            _block(
                self.market_width, self.market_width,
                output_bias=torch.zeros(self.market_width), live_output=True,
                **options,
            )
            for _ in range(self.return_count - 1)
        ])

        path_template = torch.randn(
            self.path_count, self.path_embedding_width
        ) * 0.05
        path_template[:, 0] = math.sqrt(self.path_embedding_width)
        self.initial_paths = nn.Linear(
            self.market_width,
            self.path_count * self.path_embedding_width,
        )
        nn.init.normal_(self.initial_paths.weight, std=1e-4)
        with torch.no_grad():
            self.initial_paths.bias.copy_(path_template.flatten())

        prior = torch.from_numpy(density.prior_component_masses).float()
        log_prior = torch.log(prior.clamp_min(torch.finfo(prior.dtype).tiny))
        return_base = torch.randn(
            self.path_embedding_width, self.output_width
        ) * 0.01
        return_base[0, :] = log_prior
        query_base = torch.randn(
            self.path_count, self.path_embedding_width
        ) * 0.01

        self.query_heads = nn.ModuleList([
            DynamicLowRankMatrixHead(
                self.market_width,
                self.path_count,
                self.path_embedding_width,
                self.matrix_rank,
                base=query_base,
                **options,
            )
            for _ in range(self.return_count)
        ])
        self.return_heads = nn.ModuleList([
            DynamicLowRankMatrixHead(
                self.market_width,
                self.path_embedding_width,
                self.output_width,
                self.matrix_rank,
                base=return_base,
                **options,
            )
            for _ in range(self.return_count)
        ])

        fixed_knots = torch.from_numpy(density.knots_unit).float()
        fixed_gaps = fixed_knots[1:] - fixed_knots[:-1]
        inverse_softplus_one = math.log(math.expm1(1.0))
        point_bias = torch.cat((
            torch.log(fixed_gaps),
            torch.tensor([inverse_softplus_one], dtype=torch.float32),
        ))
        self.point_heads = nn.ModuleList([
            _block(
                self.market_width,
                self.output_width,
                output_bias=point_bias,
                live_output=False,
                **options,
            )
            for _ in range(self.return_count)
        ])
        for head in self.point_heads:
            nn.init.normal_(head.output.weight, std=1e-4)

        recurrent_input_width = (
            self.path_count * self.path_embedding_width
            + self.market_width
            + self.path_count * self.output_width
            + self.output_width
        )
        self.path_transitions = nn.ModuleList([
            DynamicLowRankMatrixHead(
                recurrent_input_width,
                self.path_count,
                self.path_embedding_width,
                self.matrix_rank,
                base=path_template,
                **options,
            )
            for _ in range(self.return_count - 1)
        ])

        nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
        self.register_buffer("quadrature_nodes", torch.from_numpy(nodes).float())
        self.register_buffer(
            "quadrature_weights", torch.from_numpy(weights).float()
        )
        self.register_buffer("density_knots", fixed_knots)
        self.register_buffer("density_areas", triangular_basis_areas(fixed_knots))
        self.register_buffer(
            "density_means",
            torch.from_numpy(component_return_means(
                density.knots_unit, density.transform
            )).float(),
        )

    def muon_parameters(self) -> tuple[Tensor, ...]:
        blocks = (
            self.input_encoder,
            *self.market_transitions,
            *(head.generator for head in self.query_heads),
            *(head.generator for head in self.return_heads),
            *self.point_heads,
            *(head.generator for head in self.path_transitions),
        )
        return tuple(
            parameter for block in blocks for parameter in block.muon_parameters()
        )

    def _conditional_target_normalization(
        self, features: Tensor
    ) -> tuple[Tensor | None, Tensor | None]:
        if self.target_normalization_variance_floor is None:
            return None, None
        if features.shape[1] < 2:
            raise ValueError("normalized target model lacks mean/variance inputs")
        location = (
            features[:, -2].float()
            if self.target_normalization_center
            else torch.zeros_like(features[:, -2].float())
        )
        scale = torch.sqrt(features[:, -1].float().clamp_min(
            self.target_normalization_variance_floor
        ))
        return location, scale

    def _dynamic_density_grid(
        self,
        raw: Tensor,
        location: Tensor | None = None,
        scale: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        temperature = torch.nn.functional.softplus(raw[:, -1]) + 1e-4
        gap_probabilities = torch.softmax(
            raw[:, :-1] / temperature[:, None], dim=1
        )
        # A free softmax can underflow an interval to zero after one strong NLL
        # update. Besides making the piecewise-linear basis invalid, values
        # smaller than float32 resolution disappear when cumulatively added
        # near one. Keep every interval resolvable while preserving a learned,
        # exactly normalized grid.
        minimum_gap = 1e-4
        gaps = (
            minimum_gap
            + (1 - minimum_gap * gap_probabilities.shape[1])
            * gap_probabilities
        )
        knots = torch.cat((
            torch.zeros(raw.shape[0], 1, dtype=raw.dtype, device=raw.device),
            torch.cumsum(gaps, dim=1),
        ), dim=1)
        areas = batched_triangular_basis_areas(knots, validate_values=False)

        left = knots[:, :-1, None]
        right = knots[:, 1:, None]
        midpoint = (left + right) / 2
        half_width = (right - left) / 2
        unit = midpoint + half_width * self.quadrature_nodes[None, None, :]
        epsilon = torch.finfo(unit.dtype).eps
        stable = unit.clamp(epsilon, 1 - epsilon)
        logit = torch.log(stable) - torch.log1p(-stable)
        normalized_returns = (
            self.density_transform.location_bps
            + self.density_transform.scale_bps
            * torch.sinh(logit / self.density_transform.alpha)
        ) / RETURN_TO_BPS
        if location is None and scale is None:
            returns = normalized_returns
        elif location is not None and scale is not None:
            returns = (
                location[:, None, None]
                + scale[:, None, None] * normalized_returns
            )
        else:
            raise ValueError("incomplete conditional target normalization")
        arithmetic = torch.expm1(returns)
        fraction = (self.quadrature_nodes + 1) / 2
        scaled_weights = self.quadrature_weights[None, None, :] * half_width

        def component_means(values: Tensor) -> Tensor:
            left_value = (
                scaled_weights * values * (1 - fraction)[None, None, :]
            ).sum(dim=2)
            right_value = (
                scaled_weights * values * fraction[None, None, :]
            ).sum(dim=2)
            numerator = torch.zeros_like(knots)
            numerator[:, :-1] += left_value
            numerator[:, 1:] += right_value
            return numerator / areas

        return knots, areas, component_means(returns), component_means(arithmetic)

    def forward(self, features: Tensor) -> CompressedPathOutput:
        if features.ndim != 2 or features.shape[1] != self.feature_mean.numel():
            raise ValueError("path-matrix model expects [batch, 3 * feature] input")
        location, target_scale = self._conditional_target_normalization(features)
        normalized = (features.float() - self.feature_mean) / self.feature_std
        market = self.input_encoder(normalized)
        paths = self.initial_paths(market).reshape(
            -1, self.path_count, self.path_embedding_width
        )

        log_masses: list[Tensor] = []
        expectations: list[Tensor] = []
        knots_by_step: list[Tensor] = []
        areas_by_step: list[Tensor] = []
        means_by_step: list[Tensor] = []
        arithmetic_means_by_step: list[Tensor] = []
        scale = math.sqrt(self.path_embedding_width)

        for step in range(self.return_count):
            query = self.query_heads[step](market)
            q = torch.softmax((paths * query).sum(dim=2) / scale, dim=1)
            return_matrix = self.return_heads[step](market)
            transition = torch.softmax(
                torch.bmm(paths, return_matrix) / scale, dim=2
            )
            joint = q[:, :, None] * transition
            marginal = joint.sum(dim=1)
            marginal = marginal / marginal.sum(dim=1, keepdim=True).clamp_min(
                1e-12
            )

            point_state = self.point_heads[step](market)
            knots, areas, means, arithmetic_means = self._dynamic_density_grid(
                point_state, location, target_scale
            )
            log_masses.append(torch.log(marginal.clamp_min(1e-30)))
            expectations.append((marginal * means).sum(dim=1))
            knots_by_step.append(knots)
            areas_by_step.append(areas)
            means_by_step.append(means)
            arithmetic_means_by_step.append(arithmetic_means)

            if step + 1 < self.return_count:
                recurrent = torch.cat((
                    paths.flatten(1),
                    market,
                    joint.flatten(1),
                    point_state,
                ), dim=1)
                paths = self.path_transitions[step](recurrent)
                market = self.market_transitions[step](market)

        return CompressedPathOutput(
            log_masses=tuple(log_masses),
            expectations=torch.stack(expectations, dim=1),
            knots_unit=tuple(knots_by_step),
            areas_unit=tuple(areas_by_step),
            component_means=tuple(means_by_step),
            arithmetic_component_means=tuple(arithmetic_means_by_step),
            normalization_location=location,
            normalization_scale=target_scale,
        )

    def knots(self, step: int) -> Tensor:
        if step < 0 or step >= self.return_count:
            raise IndexError(step)
        return self.density_knots

    def areas(self, step: int) -> Tensor:
        if step < 0 or step >= self.return_count:
            raise IndexError(step)
        return self.density_areas

    def means(self, step: int) -> Tensor:
        if step < 0 or step >= self.return_count:
            raise IndexError(step)
        return self.density_means


class DirectFactorizedPathMatrixDensity(DynamicLowRankPathMatrixDensity):
    """Full path matrices from unpacked rank-factorized expected-width GLUs."""

    architecture_contract = DIRECT_FACTORIZED_ARCHITECTURE_CONTRACT

    def __init__(
        self,
        feature_mean: Tensor,
        feature_std: Tensor,
        density: KnotDensityContract,
        *,
        market_width: int,
        path_embedding_width: int,
        path_count: int,
        return_count: int,
        factor_rank: int,
        hidden_width_threshold: int,
        initial_radius: float,
        minimum_radius: float,
        learnable_centering: bool,
        quadrature_order: int = 16,
        target_normalization_variance_floor: float | None = None,
        target_normalization_center: bool = True,
    ) -> None:
        # Reuse the common market encoder, point heads, density transform, and
        # path propagation. The temporary rank-one heads are replaced below
        # before the model is transferred to its training device.
        super().__init__(
            feature_mean,
            feature_std,
            density,
            market_width=market_width,
            path_embedding_width=path_embedding_width,
            path_count=path_count,
            return_count=return_count,
            matrix_rank=1,
            hidden_width_cap=hidden_width_threshold,
            initial_radius=initial_radius,
            minimum_radius=minimum_radius,
            learnable_centering=learnable_centering,
            quadrature_order=quadrature_order,
            target_normalization_variance_floor=(
                target_normalization_variance_floor
            ),
            target_normalization_center=target_normalization_center,
        )
        if factor_rank <= 0 or hidden_width_threshold <= 0:
            raise ValueError("invalid direct factor rank or width threshold")
        self.factor_rank = int(factor_rank)
        self.hidden_width_threshold = int(hidden_width_threshold)
        self.matrix_rank = self.factor_rank

        prior = torch.from_numpy(density.prior_component_masses).float()
        log_prior = torch.log(prior.clamp_min(torch.finfo(prior.dtype).tiny))
        return_bias = torch.zeros(
            self.path_embedding_width, self.output_width
        )
        return_bias[0, :] = log_prior
        query_bias = torch.randn(
            self.path_count, self.path_embedding_width
        ) * 0.01
        path_bias = self.initial_paths.bias.detach().reshape(
            self.path_count, self.path_embedding_width
        ).clone()
        options = {
            "rank": self.factor_rank,
            "hidden_width_threshold": self.hidden_width_threshold,
            "initial_radius": initial_radius,
            "minimum_radius": minimum_radius,
            "learnable_centering": learnable_centering,
        }
        self.query_heads = nn.ModuleList([
            DirectFactorizedMatrixHead(
                self.market_width,
                self.path_count,
                self.path_embedding_width,
                output_bias=query_bias,
                **options,
            )
            for _ in range(self.return_count)
        ])
        self.return_heads = nn.ModuleList([
            DirectFactorizedMatrixHead(
                self.market_width,
                self.path_embedding_width,
                self.output_width,
                output_bias=return_bias,
                **options,
            )
            for _ in range(self.return_count)
        ])
        recurrent_input_width = (
            self.path_count * self.path_embedding_width
            + self.market_width
            + self.path_count * self.output_width
            + self.output_width
        )
        self.path_transitions = nn.ModuleList([
            DirectFactorizedMatrixHead(
                recurrent_input_width,
                self.path_count,
                self.path_embedding_width,
                output_bias=path_bias,
                **options,
            )
            for _ in range(self.return_count - 1)
        ])


class CyclicDenseCompressedPathMatrixDensity(DynamicLowRankPathMatrixDensity):
    """Dense path-density stages reused cyclically across the forecast horizon."""

    architecture_contract = CYCLIC_DENSE_COMPRESSED_ARCHITECTURE_CONTRACT

    def __init__(
        self,
        feature_mean: Tensor,
        feature_std: Tensor,
        density: KnotDensityContract,
        *,
        market_width: int,
        path_embedding_width: int,
        path_count: int,
        return_count: int,
        stage_block_count: int,
        path_compression_width: int,
        joint_compression_width: int,
        initial_radius: float,
        minimum_radius: float,
        learnable_centering: bool,
        input_dropout_probability: float = 0.0,
        embedding_dropout_probability: float = 0.0,
        quadrature_order: int = 16,
        target_normalization_variance_floor: float | None = None,
        target_normalization_center: bool = True,
    ) -> None:
        if stage_block_count <= 0 or stage_block_count > return_count:
            raise ValueError("cyclic stage-block count must be within the horizon")
        if path_compression_width <= 0 or joint_compression_width <= 0:
            raise ValueError("cyclic path compression widths must be positive")
        if not 0.0 <= float(input_dropout_probability) < 1.0:
            raise ValueError("input dropout probability must be in [0, 1)")
        if not 0.0 <= float(embedding_dropout_probability) < 1.0:
            raise ValueError("embedding dropout probability must be in [0, 1)")

        # Build the common density transform and buffers with small temporary
        # heads, then replace every stage-dependent module below. This keeps
        # one canonical implementation of the dynamic output grid.
        super().__init__(
            feature_mean,
            feature_std,
            density,
            market_width=market_width,
            path_embedding_width=path_embedding_width,
            path_count=path_count,
            return_count=return_count,
            matrix_rank=1,
            hidden_width_cap=256,
            initial_radius=initial_radius,
            minimum_radius=minimum_radius,
            learnable_centering=learnable_centering,
            quadrature_order=quadrature_order,
            target_normalization_variance_floor=(
                target_normalization_variance_floor
            ),
            target_normalization_center=target_normalization_center,
        )
        self.stage_block_count = int(stage_block_count)
        self.path_compression_width = int(path_compression_width)
        self.joint_compression_width = int(joint_compression_width)
        self.input_dropout_probability = float(input_dropout_probability)
        self.embedding_dropout_probability = float(embedding_dropout_probability)
        self.horizon_embedding = nn.Parameter(torch.zeros(
            self.return_count, self.market_width
        ))

        def dense_block(
            input_width: int,
            output_width: int,
            output_bias: Tensor,
            *,
            live_output: bool = False,
        ) -> TensorPathGluBlock:
            return _block(
                input_width,
                output_width,
                hidden_width_cap=(input_width + output_width) // 2,
                initial_radius=initial_radius,
                minimum_radius=minimum_radius,
                learnable_centering=learnable_centering,
                output_bias=output_bias,
                live_output=live_output,
            )

        self.input_encoder = dense_block(
            feature_mean.numel(), self.market_width,
            torch.zeros(self.market_width), live_output=True,
        )
        self.market_transitions = nn.ModuleList([
            dense_block(
                self.market_width, self.market_width,
                torch.zeros(self.market_width), live_output=True,
            )
            for _ in range(self.stage_block_count)
        ])

        prior = torch.from_numpy(density.prior_component_masses).float()
        log_prior = torch.log(prior.clamp_min(torch.finfo(prior.dtype).tiny))
        query_bias = torch.randn(
            self.path_count, self.path_embedding_width
        ) * 0.01
        return_bias = torch.zeros(self.path_embedding_width, self.output_width)
        return_bias[0, :] = log_prior
        path_bias = self.initial_paths.bias.detach().clone()

        fixed_knots = torch.from_numpy(density.knots_unit).float()
        fixed_gaps = fixed_knots[1:] - fixed_knots[:-1]
        inverse_softplus_one = math.log(math.expm1(1.0))
        point_bias = torch.cat((
            torch.log(fixed_gaps),
            torch.tensor([inverse_softplus_one], dtype=torch.float32),
        ))

        self.query_heads = nn.ModuleList([
            dense_block(
                self.market_width,
                self.path_count * self.path_embedding_width,
                query_bias.flatten(),
            )
            for _ in range(self.stage_block_count)
        ])
        self.return_heads = nn.ModuleList([
            dense_block(
                self.market_width,
                self.path_embedding_width * self.output_width,
                return_bias.flatten(),
            )
            for _ in range(self.stage_block_count)
        ])
        self.point_heads = nn.ModuleList([
            dense_block(self.market_width, self.output_width, point_bias)
            for _ in range(self.stage_block_count)
        ])
        self.path_compressors = nn.ModuleList([
            dense_block(
                self.path_count * self.path_embedding_width,
                self.path_compression_width,
                torch.zeros(self.path_compression_width),
            )
            for _ in range(self.stage_block_count)
        ])
        self.joint_compressors = nn.ModuleList([
            dense_block(
                self.path_count * self.output_width,
                self.joint_compression_width,
                torch.zeros(self.joint_compression_width),
            )
            for _ in range(self.stage_block_count)
        ])
        recurrent_input_width = (
            self.path_compression_width
            + self.market_width
            + self.joint_compression_width
            + self.output_width
        )
        self.path_transitions = nn.ModuleList([
            dense_block(
                recurrent_input_width,
                self.path_count * self.path_embedding_width,
                path_bias,
            )
            for _ in range(self.stage_block_count)
        ])

        # Keep every conditional branch live at initialization while retaining
        # the distribution priors encoded by the output biases.
        for block in (
            *self.query_heads,
            *self.return_heads,
            *self.point_heads,
            *self.path_compressors,
            *self.joint_compressors,
            *self.path_transitions,
        ):
            nn.init.normal_(block.output.weight, std=1e-4)

    def muon_parameters(self) -> tuple[Tensor, ...]:
        blocks = (
            self.input_encoder,
            *self.market_transitions,
            *self.query_heads,
            *self.return_heads,
            *self.point_heads,
            *self.path_compressors,
            *self.joint_compressors,
            *self.path_transitions,
        )
        return tuple(
            parameter for block in blocks for parameter in block.muon_parameters()
        )

    def initial_recurrent_state(self, features: Tensor) -> CyclicPathState:
        if features.ndim != 2 or features.shape[1] != self.feature_mean.numel():
            raise ValueError("cyclic path model expects [batch, lagged feature]")
        location, target_scale = self._conditional_target_normalization(features)
        normalized = (features.float() - self.feature_mean) / self.feature_std
        normalized = torch.nn.functional.dropout(
            normalized,
            p=self.input_dropout_probability,
            training=self.training,
        )
        market = self.input_encoder(normalized)
        market = torch.nn.functional.dropout(
            market,
            p=self.embedding_dropout_probability,
            training=self.training,
        )
        paths = self.initial_paths(market).reshape(
            -1, self.path_count, self.path_embedding_width
        )
        return CyclicPathState(market, paths, location, target_scale)

    def recurrent_step(
        self, state: CyclicPathState, step: int
    ) -> CyclicPathEmission:
        """Apply the learned one-step density and latent-state transition.

        The stage blocks and finite horizon embeddings are reused cyclically.
        This makes the transition callable beyond the supervised density
        horizon without introducing any new parameters.
        """
        if step < 0:
            raise IndexError(step)
        horizon = int(step) % self.return_count
        stage = int(step) % self.stage_block_count
        conditioned_market = (
            state.market + self.horizon_embedding[horizon][None, :]
        )
        scale = math.sqrt(self.path_embedding_width)
        query = self.query_heads[stage](conditioned_market).reshape(
            -1, self.path_count, self.path_embedding_width
        )
        q = torch.softmax((state.paths * query).sum(dim=2) / scale, dim=1)
        return_matrix = self.return_heads[stage](conditioned_market).reshape(
            -1, self.path_embedding_width, self.output_width
        )
        transition = torch.softmax(
            torch.bmm(state.paths, return_matrix) / scale, dim=2
        )
        joint = q[:, :, None] * transition
        marginal = joint.sum(dim=1)
        marginal = marginal / marginal.sum(dim=1, keepdim=True).clamp_min(1e-12)

        point_state = self.point_heads[stage](conditioned_market)
        knots, areas, means, arithmetic_means = self._dynamic_density_grid(
            point_state,
            state.normalization_location,
            state.normalization_scale,
        )
        compressed_paths = self.path_compressors[stage](state.paths.flatten(1))
        compressed_joint = self.joint_compressors[stage](joint.flatten(1))
        recurrent = torch.cat((
            compressed_paths,
            conditioned_market,
            compressed_joint,
            point_state,
        ), dim=1)
        next_paths = self.path_transitions[stage](recurrent).reshape(
            -1, self.path_count, self.path_embedding_width
        )
        next_market = self.market_transitions[stage](conditioned_market)
        next_state = CyclicPathState(
            next_market,
            next_paths,
            state.normalization_location,
            state.normalization_scale,
        )
        return CyclicPathEmission(
            conditioned_market=conditioned_market,
            paths=state.paths,
            log_masses=torch.log(marginal.clamp_min(1e-30)),
            expectation=(marginal * means).sum(dim=1),
            knots_unit=knots,
            areas_unit=areas,
            component_means=means,
            arithmetic_component_means=arithmetic_means,
            next_state=next_state,
        )

    def recurrent_rollout(
        self, features: Tensor, steps: int | None = None
    ) -> tuple[CyclicPathEmission, ...]:
        count = self.return_count if steps is None else int(steps)
        if count < 1:
            raise ValueError("recurrent rollout requires at least one step")
        state = self.initial_recurrent_state(features)
        emissions: list[CyclicPathEmission] = []
        for step in range(count):
            emission = self.recurrent_step(state, step)
            emissions.append(emission)
            state = emission.next_state
        return tuple(emissions)

    @staticmethod
    def output_from_emissions(
        emissions: tuple[CyclicPathEmission, ...] | list[CyclicPathEmission],
        joint_log_density_terms: Tensor | None = None,
    ) -> CompressedPathOutput:
        if len(emissions) < 1:
            raise ValueError("cannot assemble an empty density rollout")
        first_state = emissions[0].next_state
        return CompressedPathOutput(
            log_masses=tuple(value.log_masses for value in emissions),
            expectations=torch.stack(
                tuple(value.expectation for value in emissions), dim=1
            ),
            knots_unit=tuple(value.knots_unit for value in emissions),
            areas_unit=tuple(value.areas_unit for value in emissions),
            component_means=tuple(value.component_means for value in emissions),
            arithmetic_component_means=tuple(
                value.arithmetic_component_means for value in emissions
            ),
            normalization_location=first_state.normalization_location,
            normalization_scale=first_state.normalization_scale,
            joint_log_density_terms=joint_log_density_terms,
        )

    def forward(self, features: Tensor) -> CompressedPathOutput:
        emissions = self.recurrent_rollout(features, self.return_count)
        return self.output_from_emissions(emissions)


class JointPrefixContractedCyclicPathMatrixDensity(
    CyclicDenseCompressedPathMatrixDensity
):
    """Score realized paths through the existing market-decoded recurrence.

    Forecasting retains the original all-prefix stream exactly: market state
    produces a path query, similarity against ``P_t`` decodes ``q_t``, the
    row-conditional return matrix produces ``J_t = q_t T_t``, and the complete
    ``J_t`` is compressed into the next embedded path state.

    For joint NLL, a second differentiable contraction follows the realized
    prefix.  At every step it uses the same market query to decode probability
    from its contracted path embeddings, contracts the observed continuous
    return through ``J_t``, normalizes the resulting path/component posterior,
    and feeds that posterior through the same embedding transition.  This is
    target contraction, not sampled autoregressive rollout.
    """

    architecture_contract = JOINT_PREFIX_CONTRACTED_CYCLIC_ARCHITECTURE_CONTRACT
    requires_joint_targets = True

    def __init__(
        self,
        *args,
        recurrent_activation_checkpointing: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.recurrent_activation_checkpointing = bool(
            recurrent_activation_checkpointing
        )
        self._compiled_joint_recurrent_step = None
        self._compiled_contracted_joint_step = None

    def compile_shared_recurrent_step(
        self,
        *,
        mode: str,
        dynamic: bool = False,
    ) -> None:
        """Compile one shared transition instead of the unrolled horizon."""
        if self.stage_block_count != 1:
            raise ValueError(
                "shared recurrent-step compilation requires one stage block"
            )
        compile_arguments = {
            "fullgraph": False,
            "dynamic": bool(dynamic),
        }
        if mode == "default":
            compile_arguments["options"] = {"triton.cudagraphs": False}
        elif mode == "max-autotune-no-cudagraphs":
            compile_arguments["mode"] = mode
        else:
            raise ValueError(f"unsupported recurrent compile mode: {mode}")
        self._compiled_joint_recurrent_step = torch.compile(
            self._joint_recurrent_step_tensors,
            **compile_arguments,
        )
        self._compiled_contracted_joint_step = torch.compile(
            self._contracted_joint_step_tensors,
            **compile_arguments,
        )

    def _step_distribution(
        self,
        state: CyclicPathState,
        step: int,
    ) -> tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor,
        Tensor, Tensor, Tensor, Tensor,
    ]:
        if step < 0:
            raise IndexError(step)
        horizon = int(step) % self.return_count
        stage = int(step) % self.stage_block_count
        conditioned_market = (
            state.market + self.horizon_embedding[horizon][None, :]
        )
        return self._step_distribution_at_stage(
            state,
            conditioned_market,
            stage,
        )

    def _step_distribution_at_stage(
        self,
        state: CyclicPathState,
        conditioned_market: Tensor,
        stage: int,
    ) -> tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor,
        Tensor, Tensor, Tensor, Tensor,
    ]:
        scale = math.sqrt(self.path_embedding_width)
        query = self.query_heads[stage](conditioned_market).reshape(
            -1, self.path_count, self.path_embedding_width
        )
        prefix_masses = torch.softmax(
            (state.paths * query).sum(dim=2) / scale,
            dim=1,
        )
        return_matrix = self.return_heads[stage](conditioned_market).reshape(
            -1, self.path_embedding_width, self.output_width
        )
        conditional_masses = torch.softmax(
            torch.bmm(state.paths, return_matrix) / scale,
            dim=2,
        )
        joint = prefix_masses[:, :, None] * conditional_masses
        marginal = joint.sum(dim=1)
        marginal = marginal / marginal.sum(dim=1, keepdim=True).clamp_min(1e-12)
        point_state = self.point_heads[stage](conditioned_market)
        knots, areas, means, arithmetic_means = self._dynamic_density_grid(
            point_state,
            state.normalization_location,
            state.normalization_scale,
        )
        return (
            conditioned_market,
            conditional_masses,
            joint,
            marginal,
            point_state,
            knots,
            areas,
            means,
            arithmetic_means,
        )

    def _transition_with_joint(
        self,
        state: CyclicPathState,
        step: int,
        conditioned_market: Tensor,
        point_state: Tensor,
        joint: Tensor,
    ) -> CyclicPathState:
        stage = int(step) % self.stage_block_count
        compressed_paths = self.path_compressors[stage](state.paths.flatten(1))
        compressed_joint = self.joint_compressors[stage](joint.flatten(1))
        recurrent = torch.cat((
            compressed_paths,
            conditioned_market,
            compressed_joint,
            point_state,
        ), dim=1)
        next_paths = self.path_transitions[stage](recurrent).reshape(
            -1, self.path_count, self.path_embedding_width
        )
        next_market = self.market_transitions[stage](conditioned_market)
        return CyclicPathState(
            next_market,
            next_paths,
            state.normalization_location,
            state.normalization_scale,
        )

    def _emission(
        self,
        state: CyclicPathState,
        step: int,
        distribution: tuple[
            Tensor, Tensor, Tensor, Tensor, Tensor,
            Tensor, Tensor, Tensor, Tensor,
        ],
    ) -> CyclicPathEmission:
        (
            conditioned_market,
            _conditional,
            joint,
            marginal,
            point_state,
            knots,
            areas,
            means,
            arithmetic_means,
        ) = distribution
        next_state = self._transition_with_joint(
            state, step, conditioned_market, point_state, joint
        )
        return CyclicPathEmission(
            conditioned_market=conditioned_market,
            paths=state.paths,
            log_masses=torch.log(marginal.clamp_min(1e-30)),
            expectation=(marginal * means).sum(dim=1),
            knots_unit=knots,
            areas_unit=areas,
            component_means=means,
            arithmetic_component_means=arithmetic_means,
            next_state=next_state,
        )

    def _joint_recurrent_step_tensors(
        self,
        forecast_market: Tensor,
        forecast_paths: Tensor,
        contracted_market: Tensor,
        contracted_paths: Tensor,
        normalization_location: Tensor,
        normalization_scale: Tensor,
        unit_target: Tensor,
        target_log_jacobian: Tensor,
        horizon_embedding: Tensor,
        stage: int,
    ) -> tuple[Tensor, ...]:
        """Run one exact forecast/contraction transition using tensor I/O.

        Tensor-only boundaries let activation checkpointing discard the
        internals of this shared recurrent step and recompute them during the
        reverse pass.  Its numerical function is also used by the ordinary
        unrolled path, so checkpointing cannot silently select a different
        recurrence or loss.
        """
        location = (
            None if normalization_location.numel() == 0
            else normalization_location
        )
        scale = (
            None if normalization_scale.numel() == 0
            else normalization_scale
        )
        forecast_state = CyclicPathState(
            forecast_market, forecast_paths, location, scale
        )
        contracted_state = CyclicPathState(
            contracted_market, contracted_paths, location, scale
        )
        forecast_conditioned_market = (
            forecast_state.market + horizon_embedding[None, :]
        )
        forecast_distribution = self._step_distribution_at_stage(
            forecast_state,
            forecast_conditioned_market,
            stage,
        )
        emission = self._emission(
            forecast_state, stage, forecast_distribution
        )
        contracted_distribution = self._step_distribution_at_stage(
            contracted_state,
            contracted_state.market + horizon_embedding[None, :],
            stage,
        )
        (
            contracted_conditioned_market,
            _conditional,
            contracted_joint,
            _marginal,
            contracted_point_state,
            contracted_knots,
            contracted_areas,
            _means,
            _arithmetic_means,
        ) = contracted_distribution
        component_density = _batched_component_basis_densities(
            unit_target,
            contracted_knots,
            contracted_areas,
        )
        expanded_density = contracted_joint * component_density[:, None, :]
        density_unit = expanded_density.sum(dim=(1, 2)).clamp_min(
            torch.finfo(expanded_density.dtype).tiny
        )
        joint_log_density_term = torch.log(
            density_unit
        ) + target_log_jacobian
        posterior_expanded = expanded_density / density_unit[:, None, None]
        next_contracted_state = self._transition_with_joint(
            contracted_state,
            stage,
            contracted_conditioned_market,
            contracted_point_state,
            posterior_expanded,
        )
        return (
            emission.conditioned_market,
            emission.log_masses,
            emission.expectation,
            emission.knots_unit,
            emission.areas_unit,
            emission.component_means,
            emission.arithmetic_component_means,
            emission.next_state.market,
            emission.next_state.paths,
            joint_log_density_term,
            next_contracted_state.market,
            next_contracted_state.paths,
        )

    def _contracted_joint_step_tensors(
        self,
        contracted_market: Tensor,
        contracted_paths: Tensor,
        normalization_location: Tensor,
        normalization_scale: Tensor,
        unit_target: Tensor,
        target_log_jacobian: Tensor,
        horizon_embedding: Tensor,
        stage: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Advance only the realized-prefix state required by joint NLL."""
        location = (
            None if normalization_location.numel() == 0
            else normalization_location
        )
        scale = (
            None if normalization_scale.numel() == 0
            else normalization_scale
        )
        state = CyclicPathState(
            contracted_market, contracted_paths, location, scale
        )
        conditioned_market = state.market + horizon_embedding[None, :]
        distribution = self._step_distribution_at_stage(
            state,
            conditioned_market,
            stage,
        )
        (
            _conditioned_market,
            _conditional,
            joint,
            marginal,
            point_state,
            knots,
            areas,
            means,
            _arithmetic_means,
        ) = distribution
        component_density = _batched_component_basis_densities(
            unit_target,
            knots,
            areas,
        )
        expanded_density = joint * component_density[:, None, :]
        density_unit = expanded_density.sum(dim=(1, 2)).clamp_min(
            torch.finfo(expanded_density.dtype).tiny
        )
        joint_log_density_term = torch.log(
            density_unit
        ) + target_log_jacobian
        posterior_expanded = expanded_density / density_unit[:, None, None]
        next_state = self._transition_with_joint(
            state,
            stage,
            conditioned_market,
            point_state,
            posterior_expanded,
        )
        expectation = (marginal * means).sum(dim=1)
        return (
            expectation,
            joint_log_density_term,
            next_state.market,
            next_state.paths,
        )

    def contracted_joint_training_terms(
        self,
        features: Tensor,
        targets: Tensor,
        *,
        step_count: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return the exact teacher-contracted joint terms without marginals.

        The unconditioned forecast stream has no path to a pure joint-NLL
        objective. Omitting it here changes neither the joint density terms nor
        any gradient of their mean. Full forecast marginals remain available
        through ``forward`` for evaluation and auxiliary expectation losses.
        """
        if targets.ndim != 2 or targets.shape != (
            features.shape[0], self.return_count
        ):
            raise ValueError("joint-prefix targets have the wrong shape")
        active_steps = self.return_count if step_count is None else int(step_count)
        if active_steps < 1 or active_steps > self.return_count:
            raise ValueError("contracted joint step count is out of range")
        state = self.initial_recurrent_state(features)
        density_targets = targets.float()
        affine_log_jacobian: Tensor | float = 0.0
        if state.normalization_location is not None \
                or state.normalization_scale is not None:
            if state.normalization_location is None \
                    or state.normalization_scale is None:
                raise ValueError("incomplete conditional target normalization")
            density_targets = (
                density_targets - state.normalization_location[:, None]
            ) / state.normalization_scale[:, None]
            affine_log_jacobian = -torch.log(
                state.normalization_scale
            )[:, None]
        unit_targets, target_log_jacobian = transform_returns_to_unit(
            density_targets, self.density_transform
        )
        target_log_jacobian = target_log_jacobian + affine_log_jacobian
        empty_state_value = state.market.new_empty(0)
        normalization_location = (
            empty_state_value
            if state.normalization_location is None
            else state.normalization_location
        )
        normalization_scale = (
            empty_state_value
            if state.normalization_scale is None
            else state.normalization_scale
        )
        terms: list[Tensor] = []
        first_expectation: Tensor | None = None
        for step in range(active_steps):
            step_function = self._contracted_joint_step_tensors
            stage = int(step) % self.stage_block_count
            if self.training and self._compiled_contracted_joint_step is not None:
                step_function = self._compiled_contracted_joint_step
                stage = 0

            def recurrent_step(
                market: Tensor,
                paths: Tensor,
                location: Tensor,
                scale: Tensor,
                unit_target: Tensor,
                log_jacobian: Tensor,
                horizon_embedding: Tensor,
                *,
                active_step=step_function,
                active_stage: int = stage,
            ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
                return active_step(
                    market,
                    paths,
                    location,
                    scale,
                    unit_target,
                    log_jacobian,
                    horizon_embedding,
                    active_stage,
                )

            arguments = (
                state.market,
                state.paths,
                normalization_location,
                normalization_scale,
                unit_targets[:, step],
                target_log_jacobian[:, step],
                self.horizon_embedding[int(step) % self.return_count],
            )
            if self.recurrent_activation_checkpointing \
                    and self.training and torch.is_grad_enabled():
                values = activation_checkpoint(
                    recurrent_step,
                    *arguments,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                values = recurrent_step(*arguments)
            expectation, term, next_market, next_paths = values
            if first_expectation is None:
                first_expectation = expectation
            terms.append(term)
            state = CyclicPathState(
                next_market,
                next_paths,
                state.normalization_location,
                state.normalization_scale,
            )
        assert first_expectation is not None
        return torch.stack(terms, dim=1), first_expectation

    def forward(
        self,
        features: Tensor,
        targets: Tensor | None = None,
    ) -> CompressedPathOutput:
        if targets is not None and (
            targets.ndim != 2
            or targets.shape != (features.shape[0], self.return_count)
        ):
            raise ValueError("joint-prefix targets have the wrong shape")
        if targets is None:
            return super().forward(features)
        forecast_state = self.initial_recurrent_state(features)
        contracted_state = forecast_state
        emissions: list[CyclicPathEmission] = []
        joint_terms: list[Tensor] = []
        density_targets = targets.float()
        affine_log_jacobian: Tensor | float = 0.0
        if forecast_state.normalization_location is not None \
                or forecast_state.normalization_scale is not None:
            if forecast_state.normalization_location is None \
                    or forecast_state.normalization_scale is None:
                raise ValueError("incomplete conditional target normalization")
            density_targets = (
                density_targets - forecast_state.normalization_location[:, None]
            ) / forecast_state.normalization_scale[:, None]
            affine_log_jacobian = -torch.log(
                forecast_state.normalization_scale
            )[:, None]
        unit_targets, target_log_jacobian = transform_returns_to_unit(
            density_targets, self.density_transform
        )
        target_log_jacobian = target_log_jacobian + affine_log_jacobian

        empty_state_value = forecast_state.market.new_empty(0)
        normalization_location = (
            empty_state_value
            if forecast_state.normalization_location is None
            else forecast_state.normalization_location
        )
        normalization_scale = (
            empty_state_value
            if forecast_state.normalization_scale is None
            else forecast_state.normalization_scale
        )

        for step in range(self.return_count):
            step_function = self._joint_recurrent_step_tensors
            stage = int(step) % self.stage_block_count
            if self.training and self._compiled_joint_recurrent_step is not None:
                # The horizon embedding is a tensor input to the one compiled
                # stage, so all 15 leads reuse exactly the same graph and the
                # same arithmetic as the ordinary unrolled implementation.
                step_function = self._compiled_joint_recurrent_step
                stage = 0

            def recurrent_step(
                forecast_market: Tensor,
                forecast_paths: Tensor,
                contracted_market: Tensor,
                contracted_paths: Tensor,
                location: Tensor,
                scale: Tensor,
                unit_target: Tensor,
                log_jacobian: Tensor,
                horizon_embedding: Tensor,
                *,
                active_step=step_function,
                active_stage: int = stage,
            ) -> tuple[Tensor, ...]:
                return active_step(
                    forecast_market,
                    forecast_paths,
                    contracted_market,
                    contracted_paths,
                    location,
                    scale,
                    unit_target,
                    log_jacobian,
                    horizon_embedding,
                    active_stage,
                )

            step_arguments = (
                forecast_state.market,
                forecast_state.paths,
                contracted_state.market,
                contracted_state.paths,
                normalization_location,
                normalization_scale,
                unit_targets[:, step],
                target_log_jacobian[:, step],
                self.horizon_embedding[int(step) % self.return_count],
            )
            if self.recurrent_activation_checkpointing \
                    and self.training and torch.is_grad_enabled():
                step_values = activation_checkpoint(
                    recurrent_step,
                    *step_arguments,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                step_values = recurrent_step(*step_arguments)
            (
                conditioned_market,
                log_masses,
                expectation,
                knots,
                areas,
                means,
                arithmetic_means,
                next_forecast_market,
                next_forecast_paths,
                joint_log_density_term,
                next_contracted_market,
                next_contracted_paths,
            ) = step_values
            next_forecast_state = CyclicPathState(
                next_forecast_market,
                next_forecast_paths,
                forecast_state.normalization_location,
                forecast_state.normalization_scale,
            )
            emissions.append(CyclicPathEmission(
                conditioned_market=conditioned_market,
                paths=forecast_state.paths,
                log_masses=log_masses,
                expectation=expectation,
                knots_unit=knots,
                areas_unit=areas,
                component_means=means,
                arithmetic_component_means=arithmetic_means,
                next_state=next_forecast_state,
            ))
            joint_terms.append(joint_log_density_term)
            contracted_state = CyclicPathState(
                next_contracted_market,
                next_contracted_paths,
                contracted_state.normalization_location,
                contracted_state.normalization_scale,
            )
            forecast_state = next_forecast_state

        return self.output_from_emissions(
            emissions,
            torch.stack(joint_terms, dim=1),
        )
