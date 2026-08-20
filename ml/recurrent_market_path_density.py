from __future__ import annotations

from torch import Tensor, nn
import torch

from compressed_path_return_density import CompressedPathOutput
from exact_tensor_return_density import TensorPathGluBlock
from return_knot_density import (
    KnotDensityContract,
    component_return_means,
    triangular_basis_areas,
)


ARCHITECTURE_CONTRACT = (
    "recurrent-market-compressed-history-fixed-output-path-density-v1"
)


def _hidden_width(input_width: int, output_width: int) -> int:
    """Use the integer midpoint width from the architecture specification."""
    return (int(input_width) + int(output_width)) // 2


def _live_block(
    input_width: int,
    output_width: int,
    *,
    initial_radius: float,
    minimum_radius: float,
    learnable_centering: bool,
    output_bias: Tensor,
) -> TensorPathGluBlock:
    block = TensorPathGluBlock(
        input_width,
        _hidden_width(input_width, output_width),
        output_width,
        initial_radius=initial_radius,
        minimum_radius=minimum_radius,
        learnable_centering=learnable_centering,
        output_bias=output_bias,
    )
    with torch.no_grad():
        nn.init.eye_(block.output.weight)
    return block


class RecurrentMarketPathDensity(nn.Module):
    """Propagate market and compressed-history states into fixed-knot marginals.

    The input encoder consumes three consecutive feature rows.  A market state
    of constant width is advanced once per forecast step.  At step ``s`` it
    emits a dynamic transition factor with width ``Q_s``.  Its outer product
    with a learned output factor, plus a learned output bias, defines a
    row-normalized ``Q_s x O`` transition.  Multiplying the compressed history
    distribution by that transition yields the step's fixed-``O`` marginal.
    The next history state is obtained only from ``concat(q_s, g_s)``; it does
    not feed back into the market-state recurrence.
    """

    architecture_contract = ARCHITECTURE_CONTRACT

    def __init__(
        self,
        feature_mean: Tensor,
        feature_std: Tensor,
        density: KnotDensityContract,
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
            raise ValueError("invalid lagged-feature normalization")
        if len(state_widths) < 2 or any(int(width) <= 0 for width in state_widths):
            raise ValueError("at least two positive history-state widths are required")
        density.validate()
        self.return_count = len(state_widths)
        self.market_width = int(market_width)
        self.state_widths = tuple(int(width) for width in state_widths)
        self.output_width = len(density.knots_unit)
        self.density_transform = density.transform
        self.register_buffer("feature_mean", feature_mean.float().clone())
        self.register_buffer("feature_std", feature_std.float().clone())

        options = {
            "initial_radius": initial_radius,
            "minimum_radius": minimum_radius,
            "learnable_centering": learnable_centering,
        }
        self.input_encoder = _live_block(
            feature_mean.numel(), self.market_width,
            output_bias=torch.zeros(self.market_width), **options,
        )
        first_width = self.state_widths[0]
        self.initial_history = _live_block(
            self.market_width, first_width,
            output_bias=torch.zeros(first_width), **options,
        )
        self.market_transitions = nn.ModuleList([
            _live_block(
                self.market_width, self.market_width,
                output_bias=torch.zeros(self.market_width), **options,
            )
            for _ in self.state_widths
        ])
        self.transition_factor_blocks = nn.ModuleList([
            _live_block(
                self.market_width, width,
                output_bias=torch.ones(width), **options,
            )
            for width in self.state_widths
        ])
        self.history_down_projects = nn.ModuleList([
            _live_block(
                2 * width, next_width,
                output_bias=torch.zeros(next_width), **options,
            )
            for width, next_width in zip(
                self.state_widths[:-1], self.state_widths[1:], strict=True
            )
        ])

        prior = torch.from_numpy(density.prior_component_masses).float()
        self.destination_factors = nn.ParameterList([
            nn.Parameter(torch.log(prior.clone())) for _ in self.state_widths
        ])
        self.destination_biases = nn.ParameterList([
            nn.Parameter(torch.zeros_like(prior)) for _ in self.state_widths
        ])
        knots = torch.from_numpy(density.knots_unit).float()
        self.register_buffer("density_knots", knots)
        self.register_buffer("density_areas", triangular_basis_areas(knots))
        self.register_buffer(
            "density_means",
            torch.from_numpy(component_return_means(
                density.knots_unit, density.transform
            )).float(),
        )

    def muon_parameters(self) -> tuple[Tensor, ...]:
        blocks = (
            self.input_encoder,
            self.initial_history,
            *self.market_transitions,
            *self.transition_factor_blocks,
            *self.history_down_projects,
        )
        return tuple(
            parameter for block in blocks for parameter in block.muon_parameters()
        )

    def forward(self, features: Tensor) -> CompressedPathOutput:
        if features.ndim != 2 or features.shape[1] != self.feature_mean.numel():
            raise ValueError("recurrent market model expects [batch, 3 * feature] input")
        normalized = (features.float() - self.feature_mean) / self.feature_std
        market = self.input_encoder(normalized)
        q = torch.softmax(self.initial_history(market), dim=1)
        log_masses: list[Tensor] = []
        expectations: list[Tensor] = []

        for step, (market_block, factor_block, destination, bias) in enumerate(zip(
            self.market_transitions,
            self.transition_factor_blocks,
            self.destination_factors,
            self.destination_biases,
            strict=True,
        )):
            market = market_block(market)
            factor = factor_block(market)
            transition = torch.softmax(
                factor[:, :, None] * destination.float()[None, None, :]
                + bias.float()[None, None, :],
                dim=2,
            )
            marginal = torch.bmm(q[:, None, :], transition).squeeze(1)
            marginal = marginal / marginal.sum(dim=1, keepdim=True).clamp_min(1e-12)
            log_masses.append(torch.log(marginal.clamp_min(1e-30)))
            expectations.append(marginal @ self.density_means)
            if step + 1 < self.return_count:
                q = torch.softmax(self.history_down_projects[step](
                    torch.cat((q, factor), dim=1)
                ), dim=1)

        return CompressedPathOutput(
            log_masses=tuple(log_masses),
            expectations=torch.stack(expectations, dim=1),
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
