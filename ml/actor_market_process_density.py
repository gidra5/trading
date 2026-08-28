from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional

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


ARCHITECTURE_CONTRACT = (
    "recurrent-actor-market-process-density-packed-linear-heads-v2"
)


def _logit(probability: float) -> float:
    if not 0 < probability < 1:
        raise ValueError("a logit probability must be strictly between zero and one")
    return math.log(probability) - math.log1p(-probability)


def _dense_block(
    input_width: int,
    output_width: int,
    *,
    initial_radius: float,
    minimum_radius: float,
    learnable_centering: bool,
    output_bias: Tensor,
    output_initialization: str,
) -> TensorPathGluBlock:
    block = TensorPathGluBlock(
        input_width,
        (int(input_width) + int(output_width)) // 2,
        output_width,
        initial_radius=initial_radius,
        minimum_radius=minimum_radius,
        learnable_centering=learnable_centering,
        output_bias=output_bias,
    )
    with torch.no_grad():
        if output_initialization == "identity":
            nn.init.eye_(block.output.weight)
        elif output_initialization == "small":
            nn.init.normal_(block.output.weight, std=1e-4)
        else:
            raise ValueError(f"unsupported GNGLU output initialization: {output_initialization}")
    return block


def minimum_top_probability(probabilities: Tensor, minimum: Tensor) -> Tensor:
    """Move only the mass required to make every row's winner meet ``minimum``.

    The relative probabilities of all non-winning actions are preserved. The
    discrete winner is piecewise constant, while the mass transfer remains
    differentiable with respect to probabilities and actor certainty.
    """
    if probabilities.ndim < 1 or minimum.shape != (*probabilities.shape[:-1], 1):
        raise ValueError("minimum-confidence tensors have incompatible shapes")
    top, index = probabilities.max(dim=-1, keepdim=True)
    target = torch.maximum(top, minimum)
    transfer = (target - top) / (1 - top).clamp_min(1e-6)
    result = probabilities * (1 - transfer)
    return result.scatter_add(-1, index, transfer)


class ActorMarketProcessDensity(nn.Module):
    """Shared recurrent actor/market process with a triangular output density.

    Actor and market slots are initialized from three lagged feature rows. One
    market-process cell is reused for every forecast step. Actors affect market
    slots only through categorical action mixtures and a bounded pair impact;
    aggregated market deltas generate actor-specific reward messages. Updated
    actor and market slots become the next step's recurrent state.
    """

    architecture_contract = ARCHITECTURE_CONTRACT

    def __init__(
        self,
        feature_mean: Tensor,
        feature_std: Tensor,
        density: KnotDensityContract,
        *,
        embedding_width: int,
        actor_width: int,
        actor_decision_width: int,
        market_width: int,
        actor_count: int,
        market_count: int,
        action_count: int,
        action_basis_width: int,
        reward_width: int,
        return_count: int,
        certainty_maximum: float,
        initial_radius: float,
        minimum_radius: float,
        learnable_centering: bool,
        quadrature_order: int = 16,
    ) -> None:
        super().__init__()
        if feature_mean.ndim != 1 or feature_std.shape != feature_mean.shape \
                or bool((feature_std <= 0).any()):
            raise ValueError("invalid lagged-feature normalization")
        dimensions = (
            embedding_width,
            actor_width,
            actor_decision_width,
            market_width,
            actor_count,
            market_count,
            action_count,
            action_basis_width,
            reward_width,
            return_count,
        )
        if any(int(value) <= 0 for value in dimensions) or return_count < 2:
            raise ValueError("actor-market dimensions must be positive")
        density.validate()
        if action_count < 2 or quadrature_order < 8:
            raise ValueError("actor-market action or quadrature width is too small")
        minimum_certainty = 1 / int(action_count)
        if not minimum_certainty < certainty_maximum < 1:
            raise ValueError("certainty maximum must be between uniform and one")

        self.embedding_width = int(embedding_width)
        self.actor_width = int(actor_width)
        self.actor_decision_width = int(actor_decision_width)
        self.market_width = int(market_width)
        self.actor_count = int(actor_count)
        self.market_count = int(market_count)
        self.action_count = int(action_count)
        self.action_basis_width = int(action_basis_width)
        self.reward_width = int(reward_width)
        self.return_count = int(return_count)
        self.output_width = len(density.knots_unit)
        self.state_widths = (self.output_width,) * self.return_count
        self.minimum_certainty = minimum_certainty
        self.certainty_maximum = float(certainty_maximum)
        self.density_transform = density.transform
        self.register_buffer("feature_mean", feature_mean.float().clone())
        self.register_buffer("feature_std", feature_std.float().clone())

        options = {
            "initial_radius": initial_radius,
            "minimum_radius": minimum_radius,
            "learnable_centering": learnable_centering,
        }
        self.input_encoder = _dense_block(
            feature_mean.numel(), self.embedding_width,
            output_bias=torch.zeros(self.embedding_width),
            output_initialization="identity", **options,
        )

        actor_template = torch.randn(self.actor_count, self.actor_width) * 0.05
        market_template = torch.randn(self.market_count, self.market_width) * 0.05
        self.initial_actors = _dense_block(
            self.embedding_width, self.actor_count * self.actor_width,
            output_bias=actor_template.flatten(),
            output_initialization="small", **options,
        )
        self.initial_markets = _dense_block(
            self.embedding_width, self.market_count * self.market_width,
            output_bias=market_template.flatten(),
            output_initialization="small", **options,
        )

        importance_bias = torch.full(
            (self.market_count,), _logit(1 / self.market_count)
        )
        impact_bias = torch.full(
            (self.market_count,), _logit(1 / self.actor_count)
        )
        action_value_bias = torch.randn(
            self.action_count, self.market_width
        ) * 0.05
        reward_map_bias = torch.randn(
            self.market_width, self.reward_width
        ) * 0.01
        self.actor_decision_trunk = _dense_block(
            self.actor_width, self.actor_decision_width,
            output_bias=torch.zeros(self.actor_decision_width),
            output_initialization="identity", **options,
        )
        packed_bias = torch.cat((
            importance_bias,
            reward_map_bias.flatten(),
            torch.zeros(1),
            action_value_bias.flatten(),
            impact_bias,
        ))
        self.actor_decision_projection = nn.Linear(
            self.actor_decision_width, packed_bias.numel()
        )
        nn.init.normal_(self.actor_decision_projection.weight, std=1e-4)
        with torch.no_grad():
            self.actor_decision_projection.bias.copy_(packed_bias)

        offset = 0
        self.importance_slice = slice(offset, offset + self.market_count)
        offset = self.importance_slice.stop
        self.reward_slice = slice(
            offset, offset + self.market_width * self.reward_width
        )
        offset = self.reward_slice.stop
        self.certainty_slice = slice(offset, offset + 1)
        offset = self.certainty_slice.stop
        self.action_value_slice = slice(
            offset, offset + self.action_count * self.market_width
        )
        offset = self.action_value_slice.stop
        self.impact_slice = slice(offset, offset + self.market_count)
        if self.impact_slice.stop != packed_bias.numel():
            raise RuntimeError("packed actor-decision projection width is inconsistent")
        self.actor_transition = _dense_block(
            self.actor_width + self.reward_width, self.actor_width,
            output_bias=torch.zeros(self.actor_width),
            output_initialization="identity", **options,
        )
        self.market_transition = _dense_block(
            2 * self.market_width, self.market_width,
            output_bias=torch.zeros(self.market_width),
            output_initialization="identity", **options,
        )

        self.action_vectors = nn.Parameter(torch.randn(
            self.action_count, self.action_basis_width
        ))
        self.action_to_market_delta = nn.Parameter(torch.empty(
            self.action_basis_width, self.market_width
        ))
        nn.init.normal_(self.action_to_market_delta, std=0.01)

        self.output_embedding = _dense_block(
            self.market_count * self.market_width, self.embedding_width,
            output_bias=torch.zeros(self.embedding_width),
            output_initialization="identity", **options,
        )
        prior = torch.from_numpy(density.prior_component_masses).float()
        log_prior = torch.log(prior.clamp_min(torch.finfo(prior.dtype).tiny))
        fixed_knots = torch.from_numpy(density.knots_unit).float()
        fixed_gaps = fixed_knots[1:] - fixed_knots[:-1]
        inverse_softplus_one = math.log(math.expm1(1.0))
        point_bias = torch.cat((
            torch.log(fixed_gaps),
            torch.tensor([inverse_softplus_one], dtype=torch.float32),
        ))
        self.output_projection = nn.Linear(
            self.embedding_width, 2 * self.output_width
        )
        nn.init.normal_(self.output_projection.weight, std=1e-4)
        with torch.no_grad():
            self.output_projection.bias.copy_(torch.cat((log_prior, point_bias)))

        nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
        self.register_buffer("quadrature_nodes", torch.from_numpy(nodes).float())
        self.register_buffer("quadrature_weights", torch.from_numpy(weights).float())
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
            self.initial_actors,
            self.initial_markets,
            self.actor_decision_trunk,
            self.actor_transition,
            self.market_transition,
            self.output_embedding,
        )
        return tuple(
            parameter for block in blocks for parameter in block.muon_parameters()
        )

    def initial_state(self, features: Tensor) -> tuple[Tensor, Tensor]:
        _embedded, actors, markets = self.initial_state_with_embedding(features)
        return actors, markets

    def initial_state_with_embedding(
        self, features: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if features.ndim != 2 or features.shape[1] != self.feature_mean.numel():
            raise ValueError("actor-market model expects [batch, 3 * feature] input")
        normalized = (features.float() - self.feature_mean) / self.feature_std
        embedded = self.input_encoder(normalized)
        actors = self.initial_actors(embedded).reshape(
            -1, self.actor_count, self.actor_width
        )
        markets = self.initial_markets(embedded).reshape(
            -1, self.market_count, self.market_width
        )
        return embedded, actors, markets

    def actor_decisions(
        self, actors: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch = actors.shape[0]
        actor_rows = actors.reshape(batch * self.actor_count, self.actor_width)
        decision_rows = self.actor_decision_trunk(actor_rows)
        packed = self.actor_decision_projection(decision_rows)
        importance = torch.sigmoid(packed[:, self.importance_slice]).reshape(
            batch, self.actor_count, self.market_count
        )
        reward_maps = packed[:, self.reward_slice].reshape(
            batch, self.actor_count, self.market_width, self.reward_width
        )
        raw_certainty = packed[:, self.certainty_slice].reshape(
            batch, self.actor_count, 1, 1
        )
        action_values = packed[:, self.action_value_slice].reshape(
            batch, self.actor_count, self.action_count, self.market_width
        )
        impact = torch.sigmoid(packed[:, self.impact_slice]).reshape(
            batch, self.actor_count, self.market_count
        )
        return importance, reward_maps, raw_certainty, action_values, impact

    def action_distribution(
        self, action_values: Tensor, raw_certainty: Tensor, markets: Tensor
    ) -> Tensor:
        scores = torch.einsum(
            "bipm,bjm->bijp",
            functional.normalize(action_values, dim=-1),
            functional.normalize(markets, dim=-1),
        )
        probabilities = torch.softmax(scores, dim=-1)
        minimum = self.minimum_certainty + (
            self.certainty_maximum - self.minimum_certainty
        ) * torch.sigmoid(raw_certainty)
        minimum = minimum.expand(-1, -1, self.market_count, -1)
        return minimum_top_probability(probabilities, minimum)

    def process_step(self, actors: Tensor, markets: Tensor) -> tuple[Tensor, Tensor]:
        expected_actor_shape = (actors.shape[0], self.actor_count, self.actor_width)
        expected_market_shape = (actors.shape[0], self.market_count, self.market_width)
        if actors.shape != expected_actor_shape or markets.shape != expected_market_shape:
            raise ValueError("actor-market recurrent state has the wrong shape")
        batch = actors.shape[0]
        importance, reward_maps, raw_certainty, action_values, impact = (
            self.actor_decisions(actors)
        )
        distribution = self.action_distribution(
            action_values, raw_certainty, markets
        )
        action_vectors = functional.normalize(self.action_vectors, dim=1)
        actions = torch.einsum("bijp,pd->bijd", distribution, action_vectors)

        pair_delta = torch.einsum(
            "bijd,dm->bijm", actions, self.action_to_market_delta
        )
        market_delta = (pair_delta * impact[..., None]).sum(dim=1)

        market_input = torch.cat((markets, market_delta), dim=2)
        next_markets = self.market_transition(
            market_input.reshape(batch * self.market_count, 2 * self.market_width)
        ).reshape(batch, self.market_count, self.market_width)

        pair_rewards = torch.einsum(
            "bjm,bimr->bijr", market_delta, reward_maps
        )
        actor_rewards = (pair_rewards * importance[..., None]).sum(dim=2)
        actor_input = torch.cat((actors, actor_rewards), dim=2)
        next_actors = self.actor_transition(
            actor_input.reshape(
                batch * self.actor_count,
                self.actor_width + self.reward_width,
            )
        ).reshape(batch, self.actor_count, self.actor_width)
        return next_actors, next_markets

    def _dynamic_density_grid(
        self, raw: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        temperature = functional.softplus(raw[:, -1]) + 1e-4
        gap_probabilities = torch.softmax(
            raw[:, :-1] / temperature[:, None], dim=1
        )
        minimum_gap = 1e-4
        gaps = minimum_gap + (
            1 - minimum_gap * gap_probabilities.shape[1]
        ) * gap_probabilities
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

    def output_distribution(
        self, markets: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        embedded = self.output_embedding(markets.flatten(1))
        mass_logits, point_parameters = self.output_projection(embedded).chunk(2, dim=1)
        marginal = torch.softmax(mass_logits, dim=1)
        knots, areas, means, arithmetic_means = self._dynamic_density_grid(
            point_parameters
        )
        return marginal, knots, areas, means, arithmetic_means

    def forward(self, features: Tensor) -> CompressedPathOutput:
        actors, markets = self.initial_state(features)
        log_masses: list[Tensor] = []
        expectations: list[Tensor] = []
        knots_by_step: list[Tensor] = []
        areas_by_step: list[Tensor] = []
        means_by_step: list[Tensor] = []
        arithmetic_means_by_step: list[Tensor] = []

        for _step in range(self.return_count):
            actors, markets = self.process_step(actors, markets)
            marginal, knots, areas, means, arithmetic_means = (
                self.output_distribution(markets)
            )
            log_masses.append(torch.log(marginal.clamp_min(1e-30)))
            expectations.append((marginal * means).sum(dim=1))
            knots_by_step.append(knots)
            areas_by_step.append(areas)
            means_by_step.append(means)
            arithmetic_means_by_step.append(arithmetic_means)

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
