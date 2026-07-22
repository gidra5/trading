from __future__ import annotations

import unittest

import torch

from mlp_model import (
    LossWeights,
    PolicySupport,
    TimeWeighting,
    distance_imbalance_time_weights,
    fitted_teacher_loss,
    persistent_distance_imbalance_time_weights,
    surface_probability_mse_per_example,
)


class DistanceImbalanceTimeWeightTests(unittest.TestCase):
    def test_distance_weighting_distinguishes_unequal_tail_lengths(self) -> None:
        actions = torch.tensor([-1.0, 3.0])
        current = torch.tensor([[0.0]])
        probability = torch.tensor([[[0.5, 0.5]]])

        weight = distance_imbalance_time_weights(
            probability,
            actions,
            current,
            TimeWeighting(distance_epsilon=0.0, minimum_weight=1e-6),
        )

        self.assertAlmostEqual(float(weight[0] - 1e-6), 0.5, places=6)

    def test_state_aggregation_takes_absolute_value_after_mean(self) -> None:
        actions = torch.tensor([-1.0, 0.0, 1.0])
        current = torch.tensor([[0.0, 0.0]])
        probability = torch.tensor([[
            [0.0, 0.0, 1.0],
            [0.25, 0.5, 0.25],
        ]])

        weight = distance_imbalance_time_weights(
            probability,
            actions,
            current,
            TimeWeighting(distance_epsilon=0.0, minimum_weight=0.1),
        )

        self.assertAlmostEqual(float(weight[0]), 0.6, places=6)

    def test_opposite_state_imbalances_cancel(self) -> None:
        actions = torch.tensor([-1.0, 1.0])
        current = torch.tensor([[0.0, 0.0]])
        probability = torch.tensor([[
            [0.0, 1.0],
            [1.0, 0.0],
        ]])

        weight = distance_imbalance_time_weights(
            probability,
            actions,
            current,
            TimeWeighting(distance_epsilon=0.0, minimum_weight=0.1),
        )

        self.assertAlmostEqual(float(weight[0]), 0.1, places=6)

    def test_repeated_same_side_advice_grows_causally(self) -> None:
        weights = persistent_distance_imbalance_time_weights(
            torch.tensor([0.5, 0.5, 0.5]),
            torch.tensor([60_000, 120_000, 180_000]),
            60_000,
            TimeWeighting(distance_epsilon=0.0, minimum_weight=1e-6),
        )

        self.assertAlmostEqual(float(weights[0] - 1e-6), 0.5, places=6)
        self.assertAlmostEqual(float(weights[1] - 1e-6), 0.625, places=6)
        self.assertAlmostEqual(float(weights[2] - 1e-6), 0.75, places=6)

    def test_opposite_advice_and_long_gaps_reset_persistence(self) -> None:
        weights = persistent_distance_imbalance_time_weights(
            torch.tensor([0.5, 0.5, -0.5, -0.5]),
            torch.tensor([60_000, 120_000, 180_000, 5_000_000]),
            60_000,
            TimeWeighting(distance_epsilon=0.0, minimum_weight=1e-6),
        )

        self.assertGreater(float(weights[1]), float(weights[0]))
        self.assertAlmostEqual(float(weights[2]), float(weights[0]), places=6)
        self.assertAlmostEqual(float(weights[3]), float(weights[0]), places=6)


class VisibleProbabilityMseTests(unittest.TestCase):
    def test_latent_only_logit_changes_do_not_affect_visible_mse(self) -> None:
        visible = torch.tensor([False, True, True, True, False])
        target_logits = torch.zeros((1, 1, 5))
        predicted_logits = target_logits.clone()
        predicted_logits[..., 0] = 20.0
        predicted_logits[..., 4] = -20.0

        target = torch.softmax(target_logits[..., visible], dim=-1)
        predicted = torch.softmax(predicted_logits[..., visible], dim=-1)
        mse = surface_probability_mse_per_example(predicted, target)

        self.assertEqual(float(mse[0]), 0.0)

    def test_visible_logit_changes_affect_visible_mse(self) -> None:
        visible = torch.tensor([False, True, True, True, False])
        target_logits = torch.zeros((1, 1, 5))
        predicted_logits = target_logits.clone()
        predicted_logits[..., 1] = 2.0

        target = torch.softmax(target_logits[..., visible], dim=-1)
        predicted = torch.softmax(predicted_logits[..., visible], dim=-1)
        mse = surface_probability_mse_per_example(predicted, target)

        self.assertGreater(float(mse[0]), 0.0)

    def test_teacher_cutoff_mask_keeps_distribution_loss_finite(self) -> None:
        actions = torch.linspace(-100.0, 100.0, 151)
        current = torch.linspace(-100.0, 100.0, 17).view(1, -1)
        predicted = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, -5.0]])
        target = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -14.0, 14.0]])

        metrics = fitted_teacher_loss(
            predicted,
            target,
            actions,
            current,
            PolicySupport(-250.0, 250.0, -100.0, 100.0, 0.00175, 0.01),
            torch.ones(8),
            LossWeights(),
            TimeWeighting(),
        )

        self.assertTrue(bool(torch.isfinite(metrics["loss"])))


if __name__ == "__main__":
    unittest.main()
