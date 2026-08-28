from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from compressed_path_return_density import CompressedPathOutput
from exact_tensor_return_density import TensorPathGluBlock
from low_rank_path_matrix_density import CyclicPathEmission


ARCHITECTURE_CONTRACT = "joint-path-distribution-model-based-fitted-q-v3"


def recurrent_q_feature_width(
    market_width: int,
    path_count: int,
    path_embedding_width: int,
    knot_count: int,
) -> int:
    return (
        int(market_width)
        + int(path_count) * int(path_embedding_width)
        + 2 * int(knot_count)
        + 2
    )


def recurrent_q_features(emission: CyclicPathEmission) -> Tensor:
    """Encode the complete latent state plus its one-step return marginal."""
    probability = emission.log_masses.exp()
    entropy = -(
        probability * emission.log_masses
    ).sum(dim=1, keepdim=True)
    return torch.cat((
        emission.conditioned_market,
        emission.paths.flatten(1),
        probability,
        emission.component_means * 10_000.0,
        emission.expectation[:, None] * 10_000.0,
        entropy,
    ), dim=1)


def distribution_feature_width(return_count: int, knot_count: int) -> int:
    # Per horizon: probability and return value for every component, followed
    # by expectation and entropy. One final coordinate carries log scale.
    return int(return_count) * (2 * int(knot_count) + 2) + 1


def distribution_features(output: CompressedPathOutput) -> Tensor:
    """Losslessly encode each predicted marginal triangular mixture."""
    if output.component_means is None:
        raise ValueError("Q policy requires component return means")
    if len(output.log_masses) != len(output.component_means):
        raise ValueError("distribution output horizons differ")
    horizons: list[Tensor] = []
    for log_mass, component_mean, expectation in zip(
        output.log_masses,
        output.component_means,
        output.expectations.unbind(dim=1),
        strict=True,
    ):
        probability = log_mass.exp()
        entropy = -(probability * log_mass).sum(dim=1, keepdim=True)
        horizons.append(torch.cat((
            probability,
            component_mean * 10_000.0,
            expectation[:, None] * 10_000.0,
            entropy,
        ), dim=1))
    if output.normalization_scale is None:
        log_scale = torch.zeros_like(output.expectations[:, :1])
    else:
        log_scale = output.normalization_scale.float().clamp_min(1e-12).log()
        log_scale = log_scale.reshape(output.expectations.shape[0], 1)
    return torch.cat((*horizons, log_scale), dim=1)


class DistributionQNetwork(nn.Module):
    """One shared nonlinear distribution encoder and a linear action head."""

    def __init__(
        self,
        input_width: int,
        action_count: int,
        *,
        state_width: int = 256,
        input_mean: Tensor | None = None,
        input_std: Tensor | None = None,
        initial_radius: float = 0.0031622776601683794,
        minimum_radius: float = 0.0001,
    ) -> None:
        super().__init__()
        if input_width < 1 or action_count < 2 or state_width < 1:
            raise ValueError("invalid Q-network width")
        hidden_width = (int(input_width) + int(state_width)) // 2
        self.encoder = TensorPathGluBlock(
            int(input_width),
            hidden_width,
            int(state_width),
            initial_radius=float(initial_radius),
            minimum_radius=float(minimum_radius),
            learnable_centering=False,
            output_bias=torch.zeros(state_width),
        )
        self.action_head = nn.Linear(state_width, action_count)
        # TensorPathGluBlock deliberately starts with a zero output projection.
        # A second zero matrix here would make the whole state-dependent path
        # permanently untrainable: neither matrix could receive a gradient.
        nn.init.normal_(self.action_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.action_head.bias)
        mean = torch.zeros(input_width) if input_mean is None else input_mean.float()
        std = torch.ones(input_width) if input_std is None else input_std.float()
        if mean.shape != (input_width,) or std.shape != (input_width,):
            raise ValueError("Q-network input statistics have the wrong shape")
        self.register_buffer("input_mean", mean)
        self.register_buffer("input_std", std.clamp_min(1e-6))

    def forward(self, encoded_distribution: Tensor) -> Tensor:
        normalized = (
            encoded_distribution.float() - self.input_mean
        ) / self.input_std
        return self.action_head(self.encoder(normalized))

    def target_copy(self) -> "DistributionQNetwork":
        result = copy.deepcopy(self)
        result.requires_grad_(False)
        return result


def rebalance_log_reward_bps(
    current_exposure: Tensor,
    target_exposure: Tensor,
    friction_bps: float,
) -> Tensor:
    cost_fraction = (
        float(friction_bps) / 10_000.0
        * (target_exposure - current_exposure).abs()
    )
    return torch.log1p(-cost_fraction.clamp_max(1 - 1e-7)) * 10_000.0


def holding_log_reward_bps(exposure: Tensor, log_return: Tensor) -> Tensor:
    wealth_multiplier = 1 + exposure * torch.expm1(log_return)
    return torch.log(wealth_multiplier.clamp_min(1e-12)) * 10_000.0


def drifted_exposure(exposure: Tensor, log_return: Tensor) -> Tensor:
    price_multiplier = torch.exp(log_return)
    wealth_multiplier = 1 + exposure * (price_multiplier - 1)
    return exposure * price_multiplier / wealth_multiplier.clamp_min(1e-12)


@dataclass(frozen=True)
class BellmanTarget:
    values: Tensor
    greedy_next_actions: Tensor


@torch.no_grad()
def full_information_double_q_target(
    online_next_values: Tensor,
    target_next_values: Tensor,
    realized_log_return: Tensor,
    actions: Tensor,
    *,
    discount: float,
    friction_bps: float,
) -> BellmanTarget:
    """Return one target for every counterfactual current target exposure."""
    if online_next_values.shape != target_next_values.shape:
        raise ValueError("online and target Q matrices differ")
    if online_next_values.ndim != 2 \
            or online_next_values.shape[1] != actions.numel():
        raise ValueError("Q matrix does not match action grid")
    if realized_log_return.shape != (online_next_values.shape[0],):
        raise ValueError("one realized return is required per transition")
    next_action = actions.view(1, 1, -1)
    returns = realized_log_return.view(-1, 1)
    drift = drifted_exposure(actions.view(1, -1), returns).unsqueeze(2)
    continuation_cost = rebalance_log_reward_bps(
        drift, next_action, friction_bps
    )
    online_candidates = online_next_values[:, None, :] + continuation_cost
    greedy = online_candidates.argmax(dim=2)
    target_candidates = target_next_values[:, None, :] + continuation_cost
    continuation = target_candidates.gather(
        2, greedy.unsqueeze(2)
    ).squeeze(2)
    immediate = holding_log_reward_bps(actions.view(1, -1), returns)
    return BellmanTarget(
        values=immediate + float(discount) * continuation,
        greedy_next_actions=greedy,
    )


@torch.no_grad()
def predicted_distribution_double_q_target(
    online_next_values: Tensor,
    target_next_values: Tensor,
    log_masses: Tensor,
    component_log_returns: Tensor,
    actions: Tensor,
    *,
    discount: float,
    friction_bps: float,
) -> BellmanTarget:
    """Integrate a Bellman backup over the model's predicted return mixture.

    The learned market/path transition supplies the next state. For every
    possible current exposure, this integrates its holding reward and its
    outcome-dependent rebalance cost over the current state's return density.
    """
    if online_next_values.shape != target_next_values.shape:
        raise ValueError("online and target Q matrices differ")
    if online_next_values.ndim != 2 \
            or online_next_values.shape[1] != actions.numel():
        raise ValueError("Q matrix does not match action grid")
    if log_masses.shape != component_log_returns.shape \
            or log_masses.ndim != 2 \
            or log_masses.shape[0] != online_next_values.shape[0]:
        raise ValueError("predicted return mixture has the wrong shape")

    probabilities = log_masses.exp()
    probabilities = probabilities / probabilities.sum(
        dim=1, keepdim=True
    ).clamp_min(1e-12)
    current_actions = actions.view(1, -1)
    next_actions = actions.view(1, 1, -1)
    values = torch.zeros_like(online_next_values)
    greedy_by_component: list[Tensor] = []
    for component in range(log_masses.shape[1]):
        returns = component_log_returns[:, component:component + 1]
        drift = drifted_exposure(current_actions, returns).unsqueeze(2)
        continuation_cost = rebalance_log_reward_bps(
            drift, next_actions, friction_bps
        )
        greedy = (
            online_next_values[:, None, :] + continuation_cost
        ).argmax(dim=2)
        continuation = (
            target_next_values[:, None, :] + continuation_cost
        ).gather(2, greedy.unsqueeze(2)).squeeze(2)
        immediate = holding_log_reward_bps(current_actions, returns)
        values += probabilities[:, component:component + 1] * (
            immediate + float(discount) * continuation
        )
        greedy_by_component.append(greedy)
    return BellmanTarget(
        values=values,
        greedy_next_actions=torch.stack(greedy_by_component, dim=2),
    )


def greedy_policy(
    post_rebalance_q: Tensor,
    current_exposure: Tensor,
    actions: Tensor,
    *,
    friction_bps: float,
) -> Tensor:
    if post_rebalance_q.ndim != 2 \
            or post_rebalance_q.shape[1] != actions.numel():
        raise ValueError("Q matrix does not match action grid")
    current = current_exposure.reshape(-1, 1)
    if current.shape[0] not in {1, post_rebalance_q.shape[0]}:
        raise ValueError("current exposure batch differs from Q batch")
    values = post_rebalance_q + rebalance_log_reward_bps(
        current, actions.view(1, -1), friction_bps
    )
    return values.argmax(dim=1)


@torch.no_grad()
def polyak_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    value = float(tau)
    if not 0 < value <= 1:
        raise ValueError("target update tau must be in (0, 1]")
    for target_parameter, online_parameter in zip(
        target.parameters(), online.parameters(), strict=True
    ):
        target_parameter.lerp_(online_parameter, value)
