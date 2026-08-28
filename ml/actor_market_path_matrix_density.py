from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from actor_market_process_density import ActorMarketProcessDensity, _dense_block
from compressed_path_return_density import CompressedPathOutput
from return_knot_density import KnotDensityContract


ARCHITECTURE_CONTRACT = (
    "recurrent-actor-market-cyclic-dense-compressed-path-matrix-density-v1"
)


class ActorMarketPathMatrixDensity(ActorMarketProcessDensity):
    """Condition one shared recurrent path-matrix head on actor/market state.

    The actor-market cell supplies the recurrent world state. At every horizon,
    actor and market slots are first compressed independently to W-wide vectors,
    then fused through a 2W-to-W block. The fused vector conditions the path
    query, conditional-return matrix, dynamic output grid, and next compressed
    path state.
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
        path_embedding_width: int,
        path_count: int,
        return_count: int,
        stage_block_count: int,
        path_compression_width: int,
        joint_compression_width: int,
        certainty_maximum: float,
        initial_radius: float,
        minimum_radius: float,
        learnable_centering: bool,
        quadrature_order: int = 16,
    ) -> None:
        if stage_block_count <= 0 or stage_block_count > return_count:
            raise ValueError("hybrid stage-block count must be within the horizon")
        if path_embedding_width <= 0 or path_count <= 0:
            raise ValueError("hybrid path dimensions must be positive")
        if path_compression_width <= 0 or joint_compression_width <= 0:
            raise ValueError("hybrid compression widths must be positive")

        super().__init__(
            feature_mean,
            feature_std,
            density,
            embedding_width=embedding_width,
            actor_width=actor_width,
            actor_decision_width=actor_decision_width,
            market_width=market_width,
            actor_count=actor_count,
            market_count=market_count,
            action_count=action_count,
            action_basis_width=action_basis_width,
            reward_width=reward_width,
            return_count=return_count,
            certainty_maximum=certainty_maximum,
            initial_radius=initial_radius,
            minimum_radius=minimum_radius,
            learnable_centering=learnable_centering,
            quadrature_order=quadrature_order,
        )

        # Replace the market-only marginal head inherited from the actor-market
        # implementation. The hybrid has one joint actor+market -> W bottleneck
        # and lets the path matrix produce all output masses.
        del self.output_embedding
        del self.output_projection
        self.path_embedding_width = int(path_embedding_width)
        self.path_count = int(path_count)
        self.stage_block_count = int(stage_block_count)
        self.path_compression_width = int(path_compression_width)
        self.joint_compression_width = int(joint_compression_width)
        options = {
            "initial_radius": initial_radius,
            "minimum_radius": minimum_radius,
            "learnable_centering": learnable_centering,
        }
        self.actor_state_compressor = _dense_block(
            self.actor_count * self.actor_width,
            self.embedding_width,
            output_bias=torch.zeros(self.embedding_width),
            output_initialization="identity",
            **options,
        )
        self.market_state_compressor = _dense_block(
            self.market_count * self.market_width,
            self.embedding_width,
            output_bias=torch.zeros(self.embedding_width),
            output_initialization="identity",
            **options,
        )
        self.state_fusion = _dense_block(
            2 * self.embedding_width,
            self.embedding_width,
            output_bias=torch.zeros(self.embedding_width),
            output_initialization="identity",
            **options,
        )
        self.horizon_embedding = nn.Parameter(torch.zeros(
            self.return_count, self.embedding_width
        ))

        path_template = torch.randn(
            self.path_count, self.path_embedding_width
        ) * 0.05
        path_template[:, 0] = math.sqrt(self.path_embedding_width)
        self.initial_paths = nn.Linear(
            self.embedding_width,
            self.path_count * self.path_embedding_width,
        )
        nn.init.normal_(self.initial_paths.weight, std=1e-4)
        with torch.no_grad():
            self.initial_paths.bias.copy_(path_template.flatten())

        prior = torch.from_numpy(density.prior_component_masses).float()
        log_prior = torch.log(prior.clamp_min(torch.finfo(prior.dtype).tiny))
        query_bias = torch.randn(
            self.path_count, self.path_embedding_width
        ) * 0.01
        return_bias = torch.zeros(self.path_embedding_width, self.output_width)
        return_bias[0, :] = log_prior
        path_bias = path_template.flatten()
        fixed_knots = torch.from_numpy(density.knots_unit).float()
        fixed_gaps = fixed_knots[1:] - fixed_knots[:-1]
        inverse_softplus_one = math.log(math.expm1(1.0))
        point_bias = torch.cat((
            torch.log(fixed_gaps),
            torch.tensor([inverse_softplus_one], dtype=torch.float32),
        ))

        def dense_block(
            input_width: int, output_width: int, output_bias: Tensor
        ):
            block = _dense_block(
                input_width,
                output_width,
                output_bias=output_bias,
                output_initialization="small",
                **options,
            )
            return block

        self.query_heads = nn.ModuleList([
            dense_block(
                self.embedding_width,
                self.path_count * self.path_embedding_width,
                query_bias.flatten(),
            )
            for _ in range(self.stage_block_count)
        ])
        self.return_heads = nn.ModuleList([
            dense_block(
                self.embedding_width,
                self.path_embedding_width * self.output_width,
                return_bias.flatten(),
            )
            for _ in range(self.stage_block_count)
        ])
        self.point_heads = nn.ModuleList([
            dense_block(self.embedding_width, self.output_width, point_bias)
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
            + self.embedding_width
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

    def muon_parameters(self) -> tuple[Tensor, ...]:
        blocks = (
            self.input_encoder,
            self.initial_actors,
            self.initial_markets,
            self.actor_decision_trunk,
            self.actor_transition,
            self.market_transition,
            self.actor_state_compressor,
            self.market_state_compressor,
            self.state_fusion,
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

    def compress_state(self, actors: Tensor, markets: Tensor) -> Tensor:
        if actors.shape[0] != markets.shape[0]:
            raise ValueError("actor and market batches differ")
        actor_state = self.actor_state_compressor(actors.flatten(1))
        market_state = self.market_state_compressor(markets.flatten(1))
        return self.state_fusion(torch.cat((actor_state, market_state), dim=1))

    def forward(self, features: Tensor) -> CompressedPathOutput:
        embedded, actors, markets = self.initial_state_with_embedding(features)
        paths = self.initial_paths(embedded).reshape(
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
            stage = step % self.stage_block_count
            actors, markets = self.process_step(actors, markets)
            fused_state = self.compress_state(actors, markets)
            conditioned_state = (
                fused_state + self.horizon_embedding[step][None, :]
            )
            query = self.query_heads[stage](conditioned_state).reshape(
                -1, self.path_count, self.path_embedding_width
            )
            q = torch.softmax((paths * query).sum(dim=2) / scale, dim=1)
            return_matrix = self.return_heads[stage](conditioned_state).reshape(
                -1, self.path_embedding_width, self.output_width
            )
            transition = torch.softmax(
                torch.bmm(paths, return_matrix) / scale, dim=2
            )
            joint = q[:, :, None] * transition
            marginal = joint.sum(dim=1)
            marginal = marginal / marginal.sum(dim=1, keepdim=True).clamp_min(
                1e-12
            )

            point_state = self.point_heads[stage](conditioned_state)
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
                compressed_paths = self.path_compressors[stage](paths.flatten(1))
                compressed_joint = self.joint_compressors[stage](joint.flatten(1))
                recurrent = torch.cat((
                    compressed_paths,
                    conditioned_state,
                    compressed_joint,
                    point_state,
                ), dim=1)
                paths = self.path_transitions[stage](recurrent).reshape(
                    -1, self.path_count, self.path_embedding_width
                )

        return CompressedPathOutput(
            log_masses=tuple(log_masses),
            expectations=torch.stack(expectations, dim=1),
            knots_unit=tuple(knots_by_step),
            areas_unit=tuple(areas_by_step),
            component_means=tuple(means_by_step),
            arithmetic_component_means=tuple(arithmetic_means_by_step),
        )
