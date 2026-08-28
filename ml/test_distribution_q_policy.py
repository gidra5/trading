from __future__ import annotations

import unittest

import torch

from distribution_q_policy import (
    DistributionQNetwork,
    drifted_exposure,
    full_information_double_q_target,
    greedy_policy,
    holding_log_reward_bps,
    predicted_distribution_double_q_target,
    rebalance_log_reward_bps,
)
from train_distribution_q_policy import linear_curriculum


class DistributionQPolicyTest(unittest.TestCase):
    def test_linear_curriculum_reaches_target_on_epoch_63(self) -> None:
        schedule = {
            "type": "linear", "start": 0.01, "end": 1.0, "epochs": 64
        }
        self.assertEqual(linear_curriculum(0, schedule), 0.01)
        self.assertAlmostEqual(linear_curriculum(63, schedule), 1.0)
        self.assertAlmostEqual(linear_curriculum(64, schedule), 1.0)

    def test_zero_q_start_still_has_a_live_state_gradient_path(self) -> None:
        torch.manual_seed(7)
        model = DistributionQNetwork(12, 5, state_width=8)
        inputs = torch.randn(16, 12)
        targets = torch.randn(16, 5)

        loss = torch.nn.functional.smooth_l1_loss(model(inputs), targets)
        loss.backward()

        self.assertGreater(
            float(model.encoder.output.weight.grad.norm()), 0.0
        )
        self.assertGreater(float(model.action_head.bias.grad.norm()), 0.0)

    def test_flat_exposure_has_zero_holding_reward(self) -> None:
        reward = holding_log_reward_bps(
            torch.tensor([0.0]), torch.tensor([0.1])
        )
        self.assertEqual(float(reward), 0.0)

    def test_long_exposure_matches_underlying_log_return(self) -> None:
        value = torch.tensor([0.0125])
        reward = holding_log_reward_bps(torch.ones(1), value)
        torch.testing.assert_close(
            reward, value * 10_000, rtol=5e-6, atol=5e-4
        )

    def test_exposure_drift_is_identity_for_flat_price(self) -> None:
        exposure = torch.tensor([-1.0, -0.25, 0.0, 0.5, 1.0])
        torch.testing.assert_close(
            drifted_exposure(exposure, torch.zeros_like(exposure)), exposure
        )

    def test_rebalance_cost_is_zero_only_without_turnover(self) -> None:
        current = torch.tensor([0.0, 0.5])
        target = torch.tensor([0.0, -0.5])
        reward = rebalance_log_reward_bps(current, target, 17.5)
        self.assertEqual(float(reward[0]), 0.0)
        self.assertLess(float(reward[1]), 0.0)

    def test_full_information_backup_uses_all_actions(self) -> None:
        actions = torch.tensor([-1.0, 0.0, 1.0])
        online = torch.tensor([[0.0, 1.0, 3.0]])
        target = torch.tensor([[0.0, 2.0, 4.0]])
        result = full_information_double_q_target(
            online, target, torch.tensor([0.01]), actions,
            discount=0.9, friction_bps=0.0,
        )
        self.assertEqual(result.values.shape, (1, 3))
        self.assertTrue(torch.equal(
            result.greedy_next_actions, torch.full((1, 3), 2)
        ))
        expected = holding_log_reward_bps(
            actions, torch.full_like(actions, 0.01)
        ) + 0.9 * 4.0
        torch.testing.assert_close(result.values[0], expected)

    def test_predicted_one_hot_distribution_matches_realized_backup(self) -> None:
        actions = torch.tensor([-1.0, 0.0, 1.0])
        online = torch.tensor([[0.0, 1.0, 3.0]])
        target = torch.tensor([[0.0, 2.0, 4.0]])
        realized = torch.tensor([0.01])
        direct = full_information_double_q_target(
            online, target, realized, actions,
            discount=0.9, friction_bps=3.0,
        )
        integrated = predicted_distribution_double_q_target(
            online,
            target,
            torch.tensor([[0.0, -100.0]]),
            torch.tensor([[0.01, -0.25]]),
            actions,
            discount=0.9,
            friction_bps=3.0,
        )
        torch.testing.assert_close(integrated.values, direct.values)
        torch.testing.assert_close(
            integrated.greedy_next_actions[:, :, 0],
            direct.greedy_next_actions,
        )

    def test_predicted_distribution_integrates_component_rewards(self) -> None:
        actions = torch.tensor([-1.0, 0.0, 1.0])
        zeros = torch.zeros(1, 3)
        probabilities = torch.tensor([[0.25, 0.75]])
        returns = torch.tensor([[-0.01, 0.02]])
        result = predicted_distribution_double_q_target(
            zeros,
            zeros,
            probabilities.log(),
            returns,
            actions,
            discount=0.0,
            friction_bps=0.0,
        )
        expected = sum(
            probabilities[:, index:index + 1]
            * holding_log_reward_bps(
                actions.view(1, -1), returns[:, index:index + 1]
            )
            for index in range(2)
        )
        torch.testing.assert_close(result.values, expected)

    def test_policy_accounts_for_current_exposure_cost(self) -> None:
        actions = torch.tensor([-1.0, 0.0, 1.0])
        q = torch.tensor([[1.0, 0.0, 1.1]])
        choice = greedy_policy(
            q, torch.tensor([-1.0]), actions, friction_bps=17.5
        )
        self.assertEqual(int(choice), 0)


if __name__ == "__main__":
    unittest.main()
