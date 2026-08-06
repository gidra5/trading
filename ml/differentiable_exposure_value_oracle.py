"""Exact-forward differentiable exposure-value oracle.

This is the PyTorch counterpart of the production exposure-value Bellman
oracle.  It intentionally keeps the production implementation unchanged and
operates on one future-price window (or a batch of windows) at a time.

The hard Bellman maximum, buy/sell boundary, and liquidation boundary make the
oracle piecewise smooth.  PyTorch propagates the exact almost-everywhere
subgradient through the selected Bellman branches.  No training-only
softening or straight-through estimator is applied here, so the forward values
and probabilities retain the brute-force oracle contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class DifferentiableExposureValueOracleConfig:
    """Execution and Bellman settings for one future-price window."""

    holding_period_steps: int = 1
    decision_delay_steps: int = 1
    value_horizon_steps: int = 1
    friction: float = 0.0
    grid_size: int = 255
    temperature: float = 0.01
    min_exposure: float = -1.0
    max_exposure: float = 1.0
    max_effective_exposure: float = 250.0
    quote_borrow_rate: float = 0.0
    asset_borrow_rate: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "holding_period_steps",
            "decision_delay_steps",
            "value_horizon_steps",
            "grid_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.holding_period_steps < 1:
            raise ValueError("holding_period_steps must be positive")
        if self.decision_delay_steps < 1:
            raise ValueError("decision_delay_steps must be positive")
        if self.value_horizon_steps < self.holding_period_steps:
            raise ValueError(
                "value_horizon_steps must cover the holding period"
            )
        if not 3 <= self.grid_size <= 65_535:
            raise ValueError("grid_size must be in [3, 65,535]")
        if not math.isfinite(self.min_exposure) \
                or not math.isfinite(self.max_exposure) \
                or self.min_exposure >= 0 \
                or self.max_exposure <= 0:
            raise ValueError("exposure bounds must contain zero")
        zero_position = (
            -self.min_exposure
            / (self.max_exposure - self.min_exposure)
            * (self.grid_size - 1)
        )
        if abs(zero_position - round(zero_position)) > 1e-9:
            raise ValueError("the action grid must contain exact zero")
        if not math.isfinite(self.max_effective_exposure) \
                or self.max_effective_exposure < max(
                    abs(self.min_exposure),
                    abs(self.max_exposure),
                ):
            raise ValueError(
                "max_effective_exposure must cover the action grid"
            )
        for name in (
            "friction",
            "quote_borrow_rate",
            "asset_borrow_rate",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.friction >= 1:
            raise ValueError("friction must be less than one")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")


class DifferentiableExposureValueOracleOutput(NamedTuple):
    """Base-action values and their temperature-normalized distribution."""

    action_values: Tensor
    logits: Tensor
    probabilities: Tensor


class DifferentiableExposureValueOracle(nn.Module):
    """Evaluate the exact hard-Bellman oracle inside an autograd graph.

    ``forward`` accepts positive prices shaped ``[time]`` or
    ``[batch, time]``.  The first price is decision time and at most
    ``value_horizon_steps`` following moves are consumed.  The output contains
    the base policy before current-exposure transaction conditioning, matching
    the stored raw-oracle target contract.

    The continuation query uses the same separable fee structure as the
    production oracle and therefore costs O(batch * decisions * actions).  A
    dense differentiable fallback retains correctness for unusual exposure
    grids whose fee denominators are not separable.
    """

    def __init__(
        self,
        config: DifferentiableExposureValueOracleConfig,
    ) -> None:
        super().__init__()
        self.config = config
        grid = (
            config.min_exposure
            + torch.arange(config.grid_size, dtype=torch.float64)
            / (config.grid_size - 1)
            * (config.max_exposure - config.min_exposure)
        )
        zero_position = round(
            -config.min_exposure
            / (config.max_exposure - config.min_exposure)
            * (config.grid_size - 1)
        )
        # Match the production contract's exact executable cash action.
        grid[zero_position] = 0
        self.register_buffer("grid", grid, persistent=True)
        self._zero_index = int(zero_position)
        self._separable_rebalance_costs = (
            1 - config.friction * config.max_exposure > 0
            and 1 - config.friction
            + config.friction * config.min_exposure > 0
        )

    def forward(self, prices: Tensor) -> DifferentiableExposureValueOracleOutput:
        """Return oracle action values, logits, and probabilities."""
        batched_prices, squeeze = self._validated_prices(prices)
        horizon = min(
            self.config.value_horizon_steps,
            batched_prices.shape[1] - 1,
        )
        grid = self.grid.to(
            device=batched_prices.device,
            dtype=batched_prices.dtype,
        )
        action_values = self._action_values(
            batched_prices[:, :horizon + 1],
            grid,
        )
        logits = action_values / self.config.temperature
        probabilities = torch.softmax(logits, dim=-1)
        if squeeze:
            action_values = action_values.squeeze(0)
            logits = logits.squeeze(0)
            probabilities = probabilities.squeeze(0)
        return DifferentiableExposureValueOracleOutput(
            action_values=action_values,
            logits=logits,
            probabilities=probabilities,
        )

    def forward_from_log_returns(
        self,
        log_returns: Tensor,
    ) -> DifferentiableExposureValueOracleOutput:
        """Evaluate a scale-free path represented by successive log returns."""
        if log_returns.ndim not in (1, 2):
            raise ValueError("log_returns must have shape [time] or [batch, time]")
        if not log_returns.is_floating_point():
            raise ValueError("log_returns must be floating point")
        prefix_shape = (*log_returns.shape[:-1], 1)
        initial = torch.ones(
            prefix_shape,
            device=log_returns.device,
            dtype=log_returns.dtype,
        )
        relative_prices = torch.cat((
            initial,
            torch.exp(torch.cumsum(log_returns, dim=-1)),
        ), dim=-1)
        return self(relative_prices)

    def _validated_prices(self, prices: Tensor) -> tuple[Tensor, bool]:
        if prices.ndim not in (1, 2):
            raise ValueError("prices must have shape [time] or [batch, time]")
        if not prices.is_floating_point():
            raise ValueError("prices must be floating point")
        if prices.shape[-1] < 2:
            raise ValueError("prices must contain at least two timestamps")
        if not bool(torch.isfinite(prices).all().item()) \
                or not bool((prices > 0).all().item()):
            raise ValueError("prices must be positive and finite")
        return (prices.unsqueeze(0), True) if prices.ndim == 1 else (prices, False)

    def _action_values(self, prices: Tensor, grid: Tensor) -> Tensor:
        batch_size = prices.shape[0]
        horizon = prices.shape[1] - 1
        holding_steps = min(self.config.holding_period_steps, horizon)
        continuation_steps = horizon - holding_steps

        next_forced = torch.full(
            (batch_size, grid.numel()),
            -torch.inf,
            device=prices.device,
            dtype=prices.dtype,
        )
        next_forced[:, self._zero_index] = 0

        boundaries = [holding_steps]
        remaining = continuation_steps
        while remaining > 0:
            duration = min(self.config.decision_delay_steps, remaining)
            boundaries.append(boundaries[-1] + duration)
            remaining -= duration

        for block in range(len(boundaries) - 2, -1, -1):
            start = boundaries[block]
            end = boundaries[block + 1]
            holding_value, endpoint_exposure, feasible = self._holding_outcome(
                prices,
                start,
                end,
                grid,
            )
            continuation = self._optimal_continuation(
                next_forced,
                endpoint_exposure,
                grid,
            )
            next_forced = torch.where(
                feasible,
                holding_value + continuation,
                torch.full_like(holding_value, -torch.inf),
            )

        holding_value, endpoint_exposure, feasible = self._holding_outcome(
            prices,
            0,
            holding_steps,
            grid,
        )
        continuation = self._optimal_continuation(
            next_forced,
            endpoint_exposure,
            grid,
        )
        return torch.where(
            feasible,
            holding_value + continuation,
            torch.full_like(holding_value, -torch.inf),
        )

    def _holding_outcome(
        self,
        prices: Tensor,
        start: int,
        end: int,
        grid: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return normalized hold value, drifted exposure, and feasibility.

        Borrowed quantities have fixed signs throughout a passive hold.  After
        factoring the applicable maintenance multiplier, wealth is affine in
        one adjusted price ratio.  Consequently, the endpoint and the single
        adverse adjusted-price extremum exactly summarize the whole block.
        """
        if end <= start:
            raise ValueError("holding blocks must contain at least one move")
        duration = end - start
        start_price = prices[:, start:start + 1]
        relative_path = prices[:, start + 1:end + 1] / start_price
        steps = torch.arange(
            1,
            duration + 1,
            device=prices.device,
            dtype=prices.dtype,
        ).unsqueeze(0)
        quote_log_rate = math.log1p(self.config.quote_borrow_rate)
        asset_log_rate = math.log1p(self.config.asset_borrow_rate)
        short_adjusted_path = relative_path * torch.exp(steps * asset_log_rate)
        quote_borrow_adjusted_path = (
            relative_path * torch.exp(-steps * quote_log_rate)
        )

        short = grid < 0
        quote_borrowed = grid > 1
        endpoint_relative = relative_path[:, -1:].expand(-1, grid.numel())
        endpoint_short = short_adjusted_path[:, -1:].expand_as(endpoint_relative)
        endpoint_quote_borrow = (
            quote_borrow_adjusted_path[:, -1:].expand_as(endpoint_relative)
        )
        adjusted_endpoint = torch.where(
            short.unsqueeze(0),
            endpoint_short,
            torch.where(
                quote_borrowed.unsqueeze(0),
                endpoint_quote_borrow,
                endpoint_relative,
            ),
        )

        adverse_short = short_adjusted_path.amax(dim=1, keepdim=True)
        adverse_owned = relative_path.amin(dim=1, keepdim=True)
        adverse_quote_borrow = quote_borrow_adjusted_path.amin(
            dim=1,
            keepdim=True,
        )
        adverse_adjusted = torch.where(
            short.unsqueeze(0),
            adverse_short,
            torch.where(
                quote_borrowed.unsqueeze(0),
                adverse_quote_borrow,
                adverse_owned,
            ),
        )

        actions = grid.unsqueeze(0)
        quote = 1 - actions
        endpoint_asset_value = actions * adjusted_endpoint
        endpoint_equity = quote + endpoint_asset_value
        quote_scale_log = torch.where(
            quote_borrowed.unsqueeze(0),
            torch.full_like(endpoint_equity, duration * quote_log_rate),
            torch.zeros_like(endpoint_equity),
        )

        adverse_asset_value = actions * adverse_adjusted
        marked_adverse_equity = quote + adverse_asset_value
        liquidated_asset_value = torch.where(
            adverse_asset_value >= 0,
            adverse_asset_value * (1 - self.config.friction),
            adverse_asset_value / (1 - self.config.friction),
        )
        liquidation_equity = quote + liquidated_asset_value
        safe_liquidation_equity = torch.where(
            liquidation_equity.abs() > torch.finfo(prices.dtype).tiny,
            liquidation_equity,
            torch.ones_like(liquidation_equity),
        )
        liquidation_exposure = (
            liquidated_asset_value / safe_liquidation_equity
        )
        feasible = (
            (liquidation_equity > 0)
            & torch.isfinite(liquidation_equity)
            & (
                liquidation_exposure.abs()
                <= self.config.max_effective_exposure
            )
            & (marked_adverse_equity > 0)
            & torch.isfinite(marked_adverse_equity)
            & (endpoint_equity > 0)
            & torch.isfinite(endpoint_equity)
        )

        safe_endpoint_equity = torch.where(
            endpoint_equity > 0,
            endpoint_equity,
            torch.ones_like(endpoint_equity),
        )
        holding_value = quote_scale_log + torch.log(safe_endpoint_equity)
        endpoint_exposure = torch.where(
            feasible,
            endpoint_asset_value / safe_endpoint_equity,
            torch.zeros_like(endpoint_asset_value),
        )
        return holding_value, endpoint_exposure, feasible

    def _optimal_continuation(
        self,
        next_forced: Tensor,
        current_exposure: Tensor,
        grid: Tensor,
    ) -> Tensor:
        if not self._separable_rebalance_costs:
            return self._dense_optimal_continuation(
                next_forced,
                current_exposure,
                grid,
            )

        friction = self.config.friction
        sell_denominator = 1 - friction * grid
        buy_denominator = 1 - friction + friction * grid
        prefix = torch.cummax(
            next_forced - torch.log(sell_denominator).unsqueeze(0),
            dim=-1,
        ).values
        suffix = torch.flip(torch.cummax(
            torch.flip(
                next_forced - torch.log(buy_denominator).unsqueeze(0),
                dims=(-1,),
            ),
            dim=-1,
        ).values, dims=(-1,))

        cursor = torch.searchsorted(grid, current_exposure, right=True)
        sell_index = (cursor - 1).clamp(0, grid.numel() - 1)
        buy_index = cursor.clamp(0, grid.numel() - 1)
        sell_numerator = 1 - friction * current_exposure
        buy_numerator = 1 - friction + friction * current_exposure
        safe_sell_numerator = torch.where(
            sell_numerator > 0,
            sell_numerator,
            torch.ones_like(sell_numerator),
        )
        safe_buy_numerator = torch.where(
            buy_numerator > 0,
            buy_numerator,
            torch.ones_like(buy_numerator),
        )
        sell = (
            torch.log(safe_sell_numerator)
            + torch.gather(prefix, 1, sell_index)
        )
        buy = (
            torch.log(safe_buy_numerator)
            + torch.gather(suffix, 1, buy_index)
        )
        sell = torch.where(
            (cursor > 0) & (sell_numerator > 0),
            sell,
            torch.full_like(sell, -torch.inf),
        )
        buy = torch.where(
            (cursor < grid.numel())
            & (buy_numerator > 0),
            buy,
            torch.full_like(buy, -torch.inf),
        )
        return torch.maximum(sell, buy)

    def _dense_optimal_continuation(
        self,
        next_forced: Tensor,
        current_exposure: Tensor,
        grid: Tensor,
    ) -> Tensor:
        friction = self.config.friction
        current = current_exposure.unsqueeze(-1)
        target = grid.reshape(1, 1, -1)
        difference = target - current
        buy_denominator = 1 - friction + friction * target
        sell_denominator = 1 - friction * target
        buy_denominator_valid = buy_denominator > 0
        sell_denominator_valid = sell_denominator > 0
        safe_buy_denominator = torch.where(
            buy_denominator_valid,
            buy_denominator,
            torch.ones_like(buy_denominator),
        )
        safe_sell_denominator = torch.where(
            sell_denominator_valid,
            sell_denominator,
            torch.ones_like(sell_denominator),
        )
        buy_factor = 1 - friction * difference / safe_buy_denominator
        sell_factor = 1 - friction * (-difference) / safe_sell_denominator
        factor = torch.where(
            difference > 0,
            buy_factor,
            torch.where(difference < 0, sell_factor, torch.ones_like(difference)),
        )
        denominator_valid = torch.where(
            difference > 0,
            buy_denominator_valid,
            torch.where(
                difference < 0,
                sell_denominator_valid,
                torch.ones_like(difference, dtype=torch.bool),
            ),
        )
        valid = denominator_valid & (factor > 0) & torch.isfinite(factor) \
            & torch.isfinite(next_forced).unsqueeze(1)
        safe_factor = torch.where(valid, factor, torch.ones_like(factor))
        candidates = torch.where(
            valid,
            torch.log(safe_factor) + next_forced.unsqueeze(1),
            torch.full_like(factor, -torch.inf),
        )
        return candidates.amax(dim=-1)
