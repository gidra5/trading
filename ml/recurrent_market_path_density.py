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
RESIDUAL_ARCHITECTURE_CONTRACT = (
    "recurrent-market-compressed-history-residual-output-path-density-v1"
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
    emits a dynamic transition factor with width ``Q_s``.  A learned source
    expansion maps each coordinate into the configured transition rank, and a
    learned rank-by-output factor plus output bias defines a row-normalized
    ``Q_s x O`` transition.  Rank one retains the original outer-product
    construction.  Multiplying the compressed history distribution by that
    transition yields the step's fixed-``O`` marginal.
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
        transition_rank: int = 1,
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
        self.transition_rank = int(transition_rank)
        if self.transition_rank <= 0:
            raise ValueError("transition rank must be positive")
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
        log_prior = torch.log(prior.clone())
        if self.transition_rank == 1:
            self.source_factors = None
            self.destination_factors = nn.ParameterList([
                nn.Parameter(log_prior.clone()) for _ in self.state_widths
            ])
        else:
            self.source_factors = nn.ParameterList()
            self.destination_factors = nn.ParameterList()
            for width in self.state_widths:
                source = torch.ones(width, self.transition_rank)
                source_noise = torch.randn_like(source) * 0.01
                source_noise.sub_(source_noise.mean(dim=1, keepdim=True))
                source.add_(source_noise)
                self.source_factors.append(nn.Parameter(source))
                self.destination_factors.append(nn.Parameter(
                    log_prior[None, :].expand(
                        self.transition_rank, -1
                    ).clone() / self.transition_rank
                ))
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
            marginal = self.contract_transition(
                step, q, factor, destination, bias
            )
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

    def conditional_transition(
        self,
        step: int,
        factor: Tensor,
        destination: Tensor,
        bias: Tensor,
    ) -> Tensor:
        logits = self.transition_logits(
            step, factor, destination, bias
        )
        return torch.softmax(
            logits,
            dim=2,
        )

    def transition_logits(
        self,
        step: int,
        factor: Tensor,
        destination: Tensor,
        bias: Tensor,
    ) -> Tensor:
        if self.transition_rank == 1:
            return (
                factor[:, :, None] * destination.float()[None, None, :]
                + bias.float()[None, None, :]
            )
        if self.source_factors is None:
            raise RuntimeError("ranked transition is missing source factors")
        source = self.source_factors[step].float()
        expanded = factor[:, :, None] * source[None, :, :]
        return torch.matmul(expanded, destination.float()) \
            + bias.float()[None, None, :]

    def contract_transition(
        self,
        step: int,
        q: Tensor,
        factor: Tensor,
        destination: Tensor,
        bias: Tensor,
    ) -> Tensor:
        transition = self.conditional_transition(
            step, factor, destination, bias
        )
        return torch.bmm(q[:, None, :], transition).squeeze(1)

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


class ResidualRecurrentMarketPathDensity(RecurrentMarketPathDensity):
    """Mix a learned full baseline transition with the rank-one conditional.

    Every compressed state owns a learned baseline row ``F'`` and a learned
    mixture logit ``b'``.  Both branches are normalized before their convex
    mixture, so every resulting transition row remains a probability
    distribution without a second softmax::

        F = softmax(F')
        P = softmax(A B + b)
        lambda = sigmoid(b')
        T = (1 - lambda) F + lambda P

    The baseline starts near the global return prior with small row-specific
    noise.  This preserves the prior initialization while breaking row
    symmetry so the compressed-state coordinates can specialize immediately.
    """

    architecture_contract = RESIDUAL_ARCHITECTURE_CONTRACT

    def __init__(
        self,
        feature_mean: Tensor,
        feature_std: Tensor,
        density: KnotDensityContract,
        *,
        market_width: int,
        state_widths: tuple[int, ...],
        transition_rank: int = 1,
        initial_radius: float,
        minimum_radius: float,
        learnable_centering: bool,
    ) -> None:
        super().__init__(
            feature_mean,
            feature_std,
            density,
            market_width=market_width,
            state_widths=state_widths,
            transition_rank=transition_rank,
            initial_radius=initial_radius,
            minimum_radius=minimum_radius,
            learnable_centering=learnable_centering,
        )
        prior = torch.from_numpy(density.prior_component_masses).float()
        log_prior = torch.log(prior.clamp_min(torch.finfo(prior.dtype).tiny))
        self.baseline_logits = nn.ParameterList()
        self.mixture_logits = nn.ParameterList()
        for width in self.state_widths:
            baseline = log_prior[None, :].expand(width, -1).clone()
            baseline.add_(torch.randn_like(baseline) * 0.01)
            self.baseline_logits.append(nn.Parameter(baseline))
            self.mixture_logits.append(nn.Parameter(torch.zeros(width)))

    def conditional_transition(
        self,
        step: int,
        factor: Tensor,
        destination: Tensor,
        bias: Tensor,
    ) -> Tensor:
        learned = super().conditional_transition(step, factor, destination, bias)
        baseline = torch.softmax(self.baseline_logits[step].float(), dim=1)
        mixture = torch.sigmoid(self.mixture_logits[step].float())[None, :, None]
        return (1.0 - mixture) * baseline[None, :, :] + mixture * learned

    def contract_transition(
        self,
        step: int,
        q: Tensor,
        factor: Tensor,
        destination: Tensor,
        bias: Tensor,
    ) -> Tensor:
        """Contract the exact mixture without materializing batched ``T``.

        Algebraically this is ``q @ ((1-lambda)F + lambda P)``.  Applying the
        row gate to ``q`` first avoids allocating the extra ``[B, Q, O]``
        mixture tensor and its backward intermediates.
        """
        learned = super().conditional_transition(
            step, factor, destination, bias
        )
        baseline = torch.softmax(self.baseline_logits[step].float(), dim=1)
        mixture = torch.sigmoid(self.mixture_logits[step].float())[None, :]
        baseline_marginal = (q * (1.0 - mixture)) @ baseline
        learned_marginal = torch.bmm(
            (q * mixture)[:, None, :], learned
        ).squeeze(1)
        return baseline_marginal + learned_marginal
