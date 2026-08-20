from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn

from compressed_path_return_density import (
    CompressedPathOutput,
    batched_triangular_basis_areas,
)
from exact_tensor_return_density import TensorPathGluBlock
from return_knot_density import (
    RETURN_TO_BPS,
    KnotDensityContract,
    component_return_means,
    triangular_basis_areas,
)


ARCHITECTURE_CONTRACT = "dynamic-low-rank-path-matrix-density-v1"


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
    ) -> None:
        super().__init__()
        if feature_mean.ndim != 1 or feature_std.shape != feature_mean.shape \
                or bool((feature_std <= 0).any()):
            raise ValueError("invalid lagged-feature normalization")
        density.validate()
        if return_count < 2 or path_embedding_width <= 0 or path_count <= 0:
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

    def _dynamic_density_grid(
        self, raw: Tensor
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
        areas = batched_triangular_basis_areas(knots)

        left = knots[:, :-1, None]
        right = knots[:, 1:, None]
        midpoint = (left + right) / 2
        half_width = (right - left) / 2
        unit = midpoint + half_width * self.quadrature_nodes[None, None, :]
        epsilon = torch.finfo(unit.dtype).eps
        stable = unit.clamp(epsilon, 1 - epsilon)
        logit = torch.log(stable) - torch.log1p(-stable)
        returns = (
            self.density_transform.location_bps
            + self.density_transform.scale_bps
            * torch.sinh(logit / self.density_transform.alpha)
        ) / RETURN_TO_BPS
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
                point_state
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
