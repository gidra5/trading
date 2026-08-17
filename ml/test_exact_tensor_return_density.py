from __future__ import annotations

import unittest

import numpy as np
import torch

from exact_tensor_return_density import (
    ExactTensorReturnDensity,
    exact_tensor_path_log_density,
    multilinear_grid_value,
    temperature_scaled_output,
)
from return_knot_density import KnotDensityContract, ReturnTransform


def density_contract() -> KnotDensityContract:
    knots = np.linspace(0, 1, 32, dtype=np.float64)
    masses = np.ones(32, dtype=np.float64) / 32
    return KnotDensityContract(
        ReturnTransform(alpha=4, location_bps=0, scale_bps=2.3),
        knots,
        masses,
        "test",
        "test",
    )


class ExactTensorReturnDensityTest(unittest.TestCase):
    def test_multilinear_interpolation(self) -> None:
        knots = torch.tensor([0.0, 0.5, 1.0])
        first, second = torch.meshgrid(knots, knots, indexing="ij")
        values = (first + 2 * second)[None, :, :]
        actual = multilinear_grid_value(
            values, torch.tensor([[0.25, 0.75]]), knots
        )
        torch.testing.assert_close(actual, torch.tensor([1.75]))

    def test_prior_initialization_builds_normalized_joint(self) -> None:
        model = ExactTensorReturnDensity(
            torch.zeros(120),
            torch.ones(120),
            density_contract(),
            hidden_width=8,
            initial_radius=0.01,
            minimum_radius=0.0001,
            learnable_centering=False,
        )
        output = model(torch.zeros((2, 120)))
        for joint in output.joint_component_masses:
            flattened = joint.reshape(joint.shape[0], -1)
            torch.testing.assert_close(
                flattened.sum(dim=1), torch.ones(2)
            )
        expected = model.density_prior_masses[None, :].expand(2, -1)
        torch.testing.assert_close(output.joint_component_masses[0], expected)
        torch.testing.assert_close(
            output.joint_component_masses[1],
            expected[:, :, None] * expected[:, None, :],
        )

    def test_path_nll_backpropagates_into_every_block(self) -> None:
        model = ExactTensorReturnDensity(
            torch.zeros(120),
            torch.ones(120),
            density_contract(),
            hidden_width=8,
            initial_radius=0.01,
            minimum_radius=0.0001,
            learnable_centering=False,
        )
        features = torch.randn((3, 120)) * 1e-4
        targets = torch.randn((3, 3)) * 1e-4
        path_terms = exact_tensor_path_log_density(
            model(features), targets, model
        )
        (-path_terms.mean()).backward()
        for block in model.blocks:
            self.assertIsNotNone(block.output.weight.grad)
            self.assertGreater(float(block.output.weight.grad.abs().sum()), 0)

    def test_joint_path_terms_telescope_to_full_tensor_density(self) -> None:
        model = ExactTensorReturnDensity(
            torch.zeros(120), torch.ones(120), density_contract(),
            hidden_width=8, initial_radius=0.01, minimum_radius=0.0001,
            learnable_centering=False,
        )
        targets = torch.tensor([[1e-5, -2e-5, 3e-5]])
        output = model(torch.zeros((1, 120)))
        terms = exact_tensor_path_log_density(output, targets, model)
        self.assertEqual(terms.shape, (1, 3))
        # At initialization all three conditionals reproduce the same global
        # prior, so their log densities match for equal target values.
        equal_targets = torch.full((1, 3), 1e-5)
        equal_terms = exact_tensor_path_log_density(
            output, equal_targets, model
        )
        torch.testing.assert_close(
            equal_terms[:, 0], equal_terms[:, 1], rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            equal_terms[:, 1], equal_terms[:, 2], rtol=1e-5, atol=1e-5
        )

    def test_temperature_scaling_preserves_normalized_joint_tensors(self) -> None:
        model = ExactTensorReturnDensity(
            torch.zeros(120), torch.ones(120), density_contract(),
            hidden_width=8, initial_radius=0.01, minimum_radius=0.0001,
            learnable_centering=False,
        )
        output = temperature_scaled_output(
            model(torch.randn((2, 120))), model, 0.5
        )
        for joint in output.joint_component_masses:
            torch.testing.assert_close(
                joint.reshape(2, -1).sum(dim=1), torch.ones(2)
            )


if __name__ == "__main__":
    unittest.main()
