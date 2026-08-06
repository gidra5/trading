from __future__ import annotations

import math
import random
import unittest
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from differentiable_exposure_value_oracle import (
    DifferentiableExposureValueOracle,
    DifferentiableExposureValueOracleConfig,
)


@dataclass(frozen=True)
class OracleFixture:
    name: str
    prices: tuple[float, ...]
    config: DifferentiableExposureValueOracleConfig


# These are the same contract fixtures used by the TypeScript optimized CPU
# and CUDA comparisons against exposure-value-oracle-reference.ts.
FIXTURES = (
    OracleFixture(
        name="off-grid drift and frequent successor switches",
        prices=(100, 107, 94, 112, 89, 118, 103, 121),
        config=DifferentiableExposureValueOracleConfig(
            holding_period_steps=1,
            decision_delay_steps=1,
            value_horizon_steps=6,
            friction=0.0075,
            grid_size=9,
            min_exposure=-1.5,
            max_exposure=1.5,
            max_effective_exposure=4,
            temperature=0.025,
            quote_borrow_rate=0.0003,
            asset_borrow_rate=0.0004,
        ),
    ),
    OracleFixture(
        name="partial final holding block",
        prices=(91, 96, 93, 104, 99, 111, 102, 115),
        config=DifferentiableExposureValueOracleConfig(
            holding_period_steps=2,
            decision_delay_steps=2,
            value_horizon_steps=5,
            friction=0.0025,
            grid_size=7,
            min_exposure=-2,
            max_exposure=2,
            max_effective_exposure=4,
            temperature=0.01,
            quote_borrow_rate=0.0002,
            asset_borrow_rate=0.00025,
        ),
    ),
    OracleFixture(
        name="leveraged actions crossing liquidation boundaries",
        prices=(100, 58, 143, 72, 160, 80),
        config=DifferentiableExposureValueOracleConfig(
            holding_period_steps=2,
            decision_delay_steps=1,
            value_horizon_steps=5,
            friction=0.004,
            grid_size=9,
            min_exposure=-4,
            max_exposure=4,
            max_effective_exposure=4,
            temperature=0.04,
            quote_borrow_rate=0.0002,
            asset_borrow_rate=0.0003,
        ),
    ),
)


class DifferentiableExposureValueOracleTest(unittest.TestCase):
    def test_matches_literal_amount_space_brute_force(self) -> None:
        for fixture in FIXTURES:
            with self.subTest(fixture=fixture.name):
                oracle = DifferentiableExposureValueOracle(fixture.config)
                actual = oracle(torch.tensor(
                    fixture.prices,
                    dtype=torch.float64,
                ))
                expected_values, expected_probabilities = brute_force_oracle(
                    fixture.prices,
                    fixture.config,
                )
                expected_values_tensor = torch.tensor(
                    expected_values,
                    dtype=torch.float64,
                )
                finite = torch.isfinite(expected_values_tensor)
                self.assertTrue(torch.equal(
                    torch.isfinite(actual.action_values),
                    finite,
                ))
                torch.testing.assert_close(
                    actual.action_values[finite],
                    expected_values_tensor[finite],
                    atol=2e-12,
                    rtol=0,
                )
                torch.testing.assert_close(
                    actual.probabilities,
                    torch.tensor(expected_probabilities, dtype=torch.float64),
                    atol=2e-12,
                    rtol=0,
                )

    def test_batched_paths_match_independent_evaluation(self) -> None:
        fixture = FIXTURES[0]
        first = torch.tensor(fixture.prices[:7], dtype=torch.float64)
        second = first * torch.tensor(
            [1.0, 0.99, 1.01, 0.98, 1.02, 0.97, 1.01],
            dtype=torch.float64,
        )
        oracle = DifferentiableExposureValueOracle(fixture.config)
        batched = oracle(torch.stack((first, second))).probabilities
        torch.testing.assert_close(batched[0], oracle(first).probabilities)
        torch.testing.assert_close(batched[1], oracle(second).probabilities)

    def test_seeded_random_paths_match_brute_force(self) -> None:
        generator = random.Random(7_139)
        for case in range(24):
            length = generator.randint(3, 8)
            holding = generator.randint(1, min(3, length - 1))
            horizon = generator.randint(holding, length - 1)
            maximum = float(generator.choice((1, 2, 4)))
            prices = [100.0]
            for _ in range(length - 1):
                prices.append(
                    prices[-1] * math.exp(generator.uniform(-0.45, 0.45))
                )
            config = DifferentiableExposureValueOracleConfig(
                holding_period_steps=holding,
                decision_delay_steps=generator.randint(1, 3),
                value_horizon_steps=horizon,
                friction=generator.uniform(0, 0.02),
                grid_size=2 * generator.randint(1, 4) + 1,
                min_exposure=-maximum,
                max_exposure=maximum,
                max_effective_exposure=max(
                    maximum,
                    generator.choice((4.0, 8.0)),
                ),
                temperature=generator.uniform(0.01, 0.1),
                quote_borrow_rate=generator.uniform(0, 0.002),
                asset_borrow_rate=generator.uniform(0, 0.002),
            )
            actual = DifferentiableExposureValueOracle(config)(torch.tensor(
                prices,
                dtype=torch.float64,
            ))
            expected_values, expected_probabilities = brute_force_oracle(
                tuple(prices),
                config,
            )
            expected_values_tensor = torch.tensor(
                expected_values,
                dtype=torch.float64,
            )
            finite = torch.isfinite(expected_values_tensor)
            with self.subTest(case=case):
                self.assertTrue(torch.equal(
                    torch.isfinite(actual.action_values),
                    finite,
                ))
                torch.testing.assert_close(
                    actual.action_values[finite],
                    expected_values_tensor[finite],
                    atol=5e-12,
                    rtol=0,
                )
                torch.testing.assert_close(
                    actual.probabilities,
                    torch.tensor(expected_probabilities, dtype=torch.float64),
                    atol=5e-12,
                    rtol=0,
                )

    def test_dense_fee_fallback_matches_brute_force(self) -> None:
        config = DifferentiableExposureValueOracleConfig(
            holding_period_steps=1,
            decision_delay_steps=1,
            value_horizon_steps=3,
            friction=0.4,
            grid_size=5,
            min_exposure=-2,
            max_exposure=2,
            max_effective_exposure=4,
            temperature=0.1,
        )
        prices = (100.0, 105.0, 95.0, 110.0)
        actual = DifferentiableExposureValueOracle(config)(torch.tensor(
            prices,
            dtype=torch.float64,
        ))
        expected_values, expected_probabilities = brute_force_oracle(
            prices,
            config,
        )
        expected_values_tensor = torch.tensor(
            expected_values,
            dtype=torch.float64,
        )
        finite = torch.isfinite(expected_values_tensor)
        self.assertTrue(torch.equal(
            torch.isfinite(actual.action_values),
            finite,
        ))
        torch.testing.assert_close(
            actual.action_values[finite],
            expected_values_tensor[finite],
            atol=2e-12,
            rtol=0,
        )
        torch.testing.assert_close(
            actual.probabilities,
            torch.tensor(expected_probabilities, dtype=torch.float64),
            atol=2e-12,
            rtol=0,
        )

    def test_log_return_entry_point_matches_relative_prices(self) -> None:
        fixture = FIXTURES[1]
        prices = torch.tensor(fixture.prices[:6], dtype=torch.float64)
        log_returns = torch.diff(torch.log(prices))
        oracle = DifferentiableExposureValueOracle(fixture.config)
        direct = oracle(prices / prices[0])
        reconstructed = oracle.forward_from_log_returns(log_returns)
        torch.testing.assert_close(
            reconstructed.action_values,
            direct.action_values,
            atol=2e-12,
            rtol=0,
        )
        torch.testing.assert_close(
            reconstructed.probabilities,
            direct.probabilities,
            atol=2e-12,
            rtol=0,
        )

    def test_float32_forward_remains_close_to_brute_force(self) -> None:
        fixture = FIXTURES[0]
        actual = DifferentiableExposureValueOracle(fixture.config)(torch.tensor(
            fixture.prices,
            dtype=torch.float32,
        ))
        _, expected_probabilities = brute_force_oracle(
            fixture.prices,
            fixture.config,
        )
        torch.testing.assert_close(
            actual.probabilities,
            torch.tensor(expected_probabilities, dtype=torch.float32),
            atol=2e-5,
            rtol=2e-5,
        )

    def test_policy_gradient_matches_central_finite_difference(self) -> None:
        config = DifferentiableExposureValueOracleConfig(
            holding_period_steps=1,
            decision_delay_steps=1,
            value_horizon_steps=5,
            friction=0.003,
            grid_size=7,
            min_exposure=-1.5,
            max_exposure=1.5,
            max_effective_exposure=4,
            temperature=0.08,
            quote_borrow_rate=0.0001,
            asset_borrow_rate=0.00015,
        )
        oracle = DifferentiableExposureValueOracle(config)
        prices = torch.tensor(
            [100.0, 102.0, 98.5, 104.0, 101.5, 106.0],
            dtype=torch.float64,
            requires_grad=True,
        )
        weights = torch.tensor(
            [-0.7, 0.2, 0.9, -0.1, 0.4, -0.5, 0.8],
            dtype=torch.float64,
        )

        loss = (oracle(prices).probabilities * weights).sum()
        loss.backward()
        self.assertIsNotNone(prices.grad)
        assert prices.grad is not None
        self.assertTrue(torch.isfinite(prices.grad).all())
        self.assertGreater(float(prices.grad[1:].abs().sum()), 0)

        epsilon = 1e-4
        comparison_index = 3
        upper = prices.detach().clone()
        lower = prices.detach().clone()
        upper[comparison_index] += epsilon
        lower[comparison_index] -= epsilon
        finite_difference = (
            (oracle(upper).probabilities * weights).sum()
            - (oracle(lower).probabilities * weights).sum()
        ) / (2 * epsilon)
        torch.testing.assert_close(
            prices.grad[comparison_index],
            finite_difference,
            atol=2e-8,
            rtol=2e-5,
        )

    def test_infeasible_actions_have_negative_infinite_value_and_zero_mass(
        self,
    ) -> None:
        config = DifferentiableExposureValueOracleConfig(
            holding_period_steps=1,
            decision_delay_steps=1,
            value_horizon_steps=1,
            friction=0.001,
            grid_size=5,
            min_exposure=-2,
            max_exposure=2,
            max_effective_exposure=4,
            temperature=0.02,
        )
        result = DifferentiableExposureValueOracle(config)(torch.tensor(
            [100.0, 60.0],
            dtype=torch.float64,
        ))
        self.assertTrue(torch.isfinite(result.action_values[3]))
        self.assertLess(float(result.action_values[3]), 0)
        self.assertGreater(float(result.probabilities[3]), 0)
        self.assertTrue(torch.isneginf(result.action_values[4]))
        self.assertEqual(float(result.probabilities[4]), 0)


def brute_force_oracle(
    prices: tuple[float, ...],
    config: DifferentiableExposureValueOracleConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Literal amount-space reference independent of tensor Bellman helpers."""
    horizon = min(config.value_horizon_steps, len(prices) - 1)
    final_time = horizon
    prices = prices[:final_time + 1]
    grid = np.asarray([
        config.min_exposure
        + index / (config.grid_size - 1)
        * (config.max_exposure - config.min_exposure)
        for index in range(config.grid_size)
    ], dtype=np.float64)
    zero_position = round(
        -config.min_exposure
        / (config.max_exposure - config.min_exposure)
        * (config.grid_size - 1)
    )
    grid[zero_position] = 0

    def equity(portfolio: tuple[float, float], price: float) -> float:
        quote, asset = portfolio
        return quote + asset * price

    def at_exposure(exposure: float, price: float) -> tuple[float, float]:
        return 1 - exposure, exposure / price

    def rebalance(
        portfolio: tuple[float, float],
        price: float,
        target: float,
    ) -> tuple[float, float] | None:
        quote, asset = portfolio
        current_equity = equity(portfolio, price)
        if not current_equity > 0 or not math.isfinite(current_equity):
            return None
        current_exposure = asset * price / current_equity
        difference = target - current_exposure
        if abs(difference) <= np.finfo(np.float64).eps:
            return quote, asset
        if difference > 0:
            denominator = 1 - config.friction + config.friction * target
            if not denominator > 0:
                return None
            gross_quote = current_equity * difference / denominator
            return (
                quote - gross_quote,
                asset + (1 - config.friction) * gross_quote / price,
            )
        denominator = 1 - config.friction * target
        if not denominator > 0:
            return None
        gross_asset_value = current_equity * -difference / denominator
        return (
            quote + (1 - config.friction) * gross_asset_value,
            asset - gross_asset_value / price,
        )

    def hold(
        start: int,
        end: int,
        portfolio: tuple[float, float],
    ) -> tuple[float, float] | None:
        quote, asset = portfolio
        for time in range(start, end):
            if quote < 0:
                quote *= 1 + config.quote_borrow_rate
            if asset < 0:
                asset *= 1 + config.asset_borrow_rate
            asset_value = asset * prices[time + 1]
            marked_equity = quote + asset_value
            liquidated_asset_value = (
                asset_value * (1 - config.friction)
                if asset_value >= 0
                else asset_value / (1 - config.friction)
            )
            liquidation_equity = quote + liquidated_asset_value
            liquidated = (
                not liquidation_equity > 0
                or not math.isfinite(liquidation_equity)
                or abs(liquidated_asset_value / liquidation_equity)
                > config.max_effective_exposure
                or not marked_equity > 0
                or not math.isfinite(marked_equity)
            )
            if liquidated:
                return None
        return quote, asset

    @lru_cache(maxsize=None)
    def decision_forced_value(
        time: int,
        target_index: int,
        remaining_steps: int,
    ) -> float:
        duration = min(
            config.decision_delay_steps,
            remaining_steps,
            final_time - time,
        )
        held = hold(
            time,
            time + duration,
            at_exposure(float(grid[target_index]), prices[time]),
        )
        if held is None:
            return -math.inf
        return continuation_value(
            time + duration,
            held,
            remaining_steps - duration,
        )

    def continuation_value(
        time: int,
        portfolio: tuple[float, float],
        remaining_steps: int,
    ) -> float:
        if time >= final_time or remaining_steps <= 0:
            closed = rebalance(portfolio, prices[time], 0)
            if closed is None:
                return -math.inf
            final_equity = equity(closed, prices[time])
            return math.log(final_equity) if final_equity > 0 else -math.inf
        best = -math.inf
        for target_index, target in enumerate(grid):
            rebalanced = rebalance(portfolio, prices[time], float(target))
            if rebalanced is None:
                continue
            rebalanced_equity = equity(rebalanced, prices[time])
            forced = decision_forced_value(
                time,
                target_index,
                remaining_steps,
            )
            if rebalanced_equity > 0 and math.isfinite(forced):
                best = max(best, math.log(rebalanced_equity) + forced)
        return best

    initial_holding_steps = min(config.holding_period_steps, horizon)
    continuation_steps = horizon - initial_holding_steps
    values = np.full(config.grid_size, -math.inf, dtype=np.float64)
    for target_index, target in enumerate(grid):
        held = hold(
            0,
            initial_holding_steps,
            at_exposure(float(target), prices[0]),
        )
        if held is not None:
            values[target_index] = continuation_value(
                initial_holding_steps,
                held,
                continuation_steps,
            )
    maximum = float(np.max(values))
    weights = np.where(
        np.isfinite(values),
        np.exp((values - maximum) / config.temperature),
        0,
    )
    return values, weights / weights.sum()


if __name__ == "__main__":
    unittest.main()
