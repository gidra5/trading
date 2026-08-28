from __future__ import annotations

import math
import unittest

import torch

from theory_q_surface import (
    RecurrentQPrimeSurface,
    interpolate_action_axis,
    interpolate_return_axis,
    normalized_logsumexp,
    theory_curriculum,
)


class TheoryQSurfaceTest(unittest.TestCase):
    def test_curriculum_has_requested_phase_boundaries(self) -> None:
        start = theory_curriculum(0)
        temperature_end = theory_curriculum(63)
        friction_start = theory_curriculum(64)
        final = theory_curriculum(511)
        self.assertAlmostEqual(start.temperature, 0.5)
        self.assertEqual(start.friction_bps, 0.0)
        self.assertAlmostEqual(temperature_end.temperature, 0.01)
        self.assertEqual(temperature_end.discount, 0.0)
        self.assertAlmostEqual(friction_start.friction_bps, 1.0)
        self.assertAlmostEqual(friction_start.horizon_seconds, 1.0)
        self.assertAlmostEqual(final.temperature, 0.01)
        self.assertAlmostEqual(final.friction_bps, 17.5)
        self.assertAlmostEqual(final.horizon_seconds, 900.0)
        self.assertAlmostEqual(final.step_seconds, 60.0)
        self.assertAlmostEqual(final.effective_steps, 15.0)
        self.assertAlmostEqual(final.discount, 14.0 / 16.0)

    def test_nlse_is_normalized_and_approaches_max(self) -> None:
        equal = torch.full((3, 7), 2.5)
        torch.testing.assert_close(
            normalized_logsumexp(equal, 0.4), torch.full((3,), 2.5)
        )
        values = torch.tensor([[0.0, 1.0, 3.0]])
        low = normalized_logsumexp(values, 0.001)
        self.assertLess(float(low), 3.0)
        self.assertGreater(float(low), 2.99)

    def test_return_reconstruction_is_linear(self) -> None:
        grid = torch.tensor([[-0.1, 0.0, 0.2]])
        surface = torch.tensor([[[0.0, 1.0], [2.0, 3.0], [6.0, 7.0]]])
        result = interpolate_return_axis(
            surface, grid, torch.tensor([[-0.05, 0.1]])
        )
        torch.testing.assert_close(
            result, torch.tensor([[[1.0, 2.0], [4.0, 5.0]]])
        )

    def test_action_reconstruction_uses_uniform_exposure_axis(self) -> None:
        actions = torch.linspace(-1.0, 1.0, 5)
        values = actions.square().reshape(1, 1, -1)
        result = interpolate_action_axis(
            values, actions, torch.tensor([[-0.75]])
        )
        self.assertAlmostEqual(float(result), (1.0 + 0.25) / 2)

    def test_q_surface_dropout_and_first_step_gradients_are_live(self) -> None:
        torch.manual_seed(3)
        model = RecurrentQPrimeSurface(8, 5, 7, dropout=0.05)
        self.assertAlmostEqual(model.input_dropout.p, 0.05)
        inputs = torch.randn(16, 8)
        target = torch.randn(16, 5, 7)
        loss = torch.nn.functional.smooth_l1_loss(model(inputs), target)
        loss.backward()
        self.assertTrue(math.isfinite(float(loss.detach())))
        assert model.surface.output.weight.grad is not None
        self.assertGreater(float(model.surface.output.weight.grad.norm()), 0.0)


if __name__ == "__main__":
    unittest.main()
