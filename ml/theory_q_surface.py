from __future__ import annotations

import copy
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from distribution_q_policy import (
    drifted_exposure,
    holding_log_reward_bps,
    rebalance_log_reward_bps,
)
from exact_tensor_return_density import TensorPathGluBlock
from low_rank_path_matrix_density import CyclicPathEmission


ARCHITECTURE_CONTRACT = "recurrent-return-action-q-prime-surface-v1"


@dataclass(frozen=True)
class TheoryCurriculum:
    temperature: float
    friction_bps: float
    horizon_seconds: float
    step_seconds: float
    effective_steps: float
    discount: float


def theory_curriculum(
    epoch: int, *, total_epochs: int = 512, temperature_epochs: int = 64
) -> TheoryCurriculum:
    if total_epochs <= temperature_epochs or temperature_epochs < 2:
        raise ValueError("invalid q' curriculum lengths")
    index = min(max(int(epoch), 0), total_epochs - 1)
    if index < temperature_epochs:
        progress = index / (temperature_epochs - 1)
        temperature = 0.5 * (0.01 / 0.5) ** progress
        friction_bps = 0.0
        horizon_seconds = 1.0
        step_seconds = 1.0
    else:
        progress = (
            (index - temperature_epochs)
            / (total_epochs - temperature_epochs - 1)
        )
        temperature = 0.01
        friction_bps = 1.0 * 17.5 ** progress
        horizon_seconds = 900.0 ** progress
        # Increase the physical decision interval from one second to one
        # minute while T grows from one second to fifteen minutes. This keeps
        # dt=T at the start of the phase and finishes with 15 effective steps.
        step_seconds = 60.0 ** progress
    effective_steps = horizon_seconds / step_seconds
    discount = (effective_steps - 1.0) / (effective_steps + 1.0)
    return TheoryCurriculum(
        temperature=temperature,
        friction_bps=friction_bps,
        horizon_seconds=horizon_seconds,
        step_seconds=step_seconds,
        effective_steps=effective_steps,
        discount=discount,
    )


class RecurrentQPrimeSurface(nn.Module):
    """Emit q'(market log-return state, target exposure) on an O x A grid."""

    def __init__(
        self,
        market_width: int,
        return_count: int,
        action_count: int,
        *,
        dropout: float = 0.05,
        initial_radius: float = 0.0031622776601683794,
        minimum_radius: float = 0.0001,
    ) -> None:
        super().__init__()
        if market_width < 1 or return_count < 3 or action_count < 3:
            raise ValueError("invalid q-prime surface dimensions")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.market_width = int(market_width)
        self.return_count = int(return_count)
        self.action_count = int(action_count)
        self.input_dropout = nn.Dropout(float(dropout))
        output_width = self.return_count * self.action_count
        self.surface = TensorPathGluBlock(
            self.market_width,
            (self.market_width + output_width) // 2,
            output_width,
            initial_radius=float(initial_radius),
            minimum_radius=float(minimum_radius),
            learnable_centering=False,
            output_bias=torch.zeros(output_width),
        )
        # TensorPathGluBlock starts with a zero output projection. Make the
        # state-dependent q' path trainable on the first optimization step.
        nn.init.normal_(self.surface.output.weight, mean=0.0, std=1e-4)

    def forward(self, market: Tensor) -> Tensor:
        values = self.surface(self.input_dropout(market.float()))
        return values.reshape(-1, self.return_count, self.action_count)

    def target_copy(self) -> "RecurrentQPrimeSurface":
        result = copy.deepcopy(self)
        result.eval()
        result.requires_grad_(False)
        return result


def normalized_logsumexp(
    values: Tensor, temperature: float | Tensor, *, dim: int = -1
) -> Tensor:
    """Temperature-scaled log mean exp, the uniform-action smooth maximum."""
    tau = torch.as_tensor(temperature, dtype=values.dtype, device=values.device)
    if tau.numel() != 1 or not bool(torch.isfinite(tau)) or float(tau) <= 0:
        raise ValueError("NLSE temperature must be finite and positive")
    return tau * (
        torch.logsumexp(values / tau, dim=dim)
        - math.log(values.shape[dim])
    )


def interpolate_return_axis(
    surface: Tensor,
    return_grid: Tensor,
    log_returns: Tensor,
) -> Tensor:
    """Piecewise-linear reconstruction of an O x A q' surface in return."""
    if surface.ndim != 3 or return_grid.shape != surface.shape[:2]:
        raise ValueError("q' surface and return grid shapes differ")
    if log_returns.ndim != 2 or log_returns.shape[0] != surface.shape[0]:
        raise ValueError("log-return query has the wrong shape")
    indexes = torch.searchsorted(
        return_grid.contiguous(), log_returns.contiguous(), right=True
    ).clamp(1, return_grid.shape[1] - 1)
    left_index = indexes - 1
    left_return = return_grid.gather(1, left_index)
    right_return = return_grid.gather(1, indexes)
    fraction = (
        (log_returns - left_return)
        / (right_return - left_return).clamp_min(1e-12)
    ).clamp(0, 1)
    action_count = surface.shape[2]
    left = surface.gather(
        1, left_index[:, :, None].expand(-1, -1, action_count)
    )
    right = surface.gather(
        1, indexes[:, :, None].expand(-1, -1, action_count)
    )
    return left + fraction[:, :, None] * (right - left)


def interpolate_action_axis(
    action_values: Tensor,
    action_grid: Tensor,
    exposures: Tensor,
) -> Tensor:
    """Piecewise-linear reconstruction on the uniform exposure axis."""
    if action_values.shape[:-1] != exposures.shape \
            or action_values.shape[-1] != action_grid.numel():
        raise ValueError("action reconstruction shapes differ")
    minimum = action_grid[0]
    spacing = action_grid[1] - action_grid[0]
    coordinate = ((exposures - minimum) / spacing).clamp(
        0, action_grid.numel() - 1
    )
    left_index = coordinate.floor().long().clamp_max(action_grid.numel() - 2)
    fraction = coordinate - left_index
    left = action_values.gather(-1, left_index.unsqueeze(-1)).squeeze(-1)
    right = action_values.gather(
        -1, (left_index + 1).unsqueeze(-1)
    ).squeeze(-1)
    return left + fraction * (right - left)


def reconstruct_q_prime(
    surface: Tensor,
    return_grid: Tensor,
    log_returns: Tensor,
    action_grid: Tensor,
    exposures: Tensor,
) -> Tensor:
    return interpolate_action_axis(
        interpolate_return_axis(surface, return_grid, log_returns),
        action_grid,
        exposures,
    )


@dataclass(frozen=True)
class QPrimeTarget:
    values: Tensor
    effective_temperature: float


@torch.no_grad()
def q_prime_bellman_target(
    current: CyclicPathEmission,
    next_surface: Tensor,
    next_return_grid: Tensor,
    actions: Tensor,
    *,
    discount: float,
    temperature: float,
    effective_steps: float,
    friction_bps: float,
) -> QPrimeTarget:
    """Theory-doc q' target on the return-component x exposure grid."""
    if next_surface.shape != (
        current.log_masses.shape[0],
        current.log_masses.shape[1],
        actions.numel(),
    ):
        raise ValueError("next q' surface has the wrong shape")
    if effective_steps < 1 or not math.isfinite(effective_steps):
        raise ValueError("effective step count must be finite and at least one")
    effective_temperature = float(temperature) * math.sqrt(effective_steps)
    state_returns = current.component_means
    next_at_state = interpolate_return_axis(
        next_surface, next_return_grid, state_returns
    )
    current_actions = actions.view(1, -1)
    next_actions = actions.view(1, 1, -1)
    targets: list[Tensor] = []
    for state_index in range(state_returns.shape[1]):
        returns = state_returns[:, state_index:state_index + 1]
        drift = drifted_exposure(current_actions, returns).unsqueeze(2)
        transition_reward = rebalance_log_reward_bps(
            drift, next_actions, friction_bps
        ) / 10_000.0
        continuation = normalized_logsumexp(
            transition_reward + next_at_state[:, state_index:state_index + 1, :],
            effective_temperature,
            dim=2,
        ).squeeze(1)
        immediate = holding_log_reward_bps(current_actions, returns) / 10_000.0
        targets.append(immediate + float(discount) * continuation)
    return QPrimeTarget(
        values=torch.stack(targets, dim=1),
        effective_temperature=effective_temperature,
    )


def expected_action_values(
    emission: CyclicPathEmission, surface: Tensor
) -> Tensor:
    if surface.shape[:2] != emission.log_masses.shape:
        raise ValueError("q' surface does not match transition distribution")
    probabilities = emission.log_masses.exp()
    return (probabilities[:, :, None] * surface).sum(dim=1)
