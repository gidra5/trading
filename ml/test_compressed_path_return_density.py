from __future__ import annotations

from pathlib import Path
import unittest

import torch

from compressed_path_return_density import (
    CompressedPathReturnDensity,
    path_log_density_terms,
)
from return_knot_density import KnotDensityContract


class CompressedPathReturnDensityTest(unittest.TestCase):
    def make_model(self) -> CompressedPathReturnDensity:
        source = Path("data/benchmarks/one-second-return-knot-scaling-v1.json")
        densities = tuple(
            KnotDensityContract.load(source, fit=str(width))
            for width in (8, 16, 32)
        )
        return CompressedPathReturnDensity(
            torch.zeros(5), torch.ones(5), densities,
            market_width=16, state_widths=(8, 16, 32),
            initial_radius=0.0031622776601683794,
            minimum_radius=1e-4, learnable_centering=False,
        )

    def test_all_marginals_are_normalized_and_trainable(self) -> None:
        model = self.make_model()
        output = model(torch.randn(7, 5))
        self.assertEqual(output.expectations.shape, (7, 3))
        for value in output.log_masses:
            torch.testing.assert_close(value.exp().sum(dim=1), torch.ones(7))
        loss = -path_log_density_terms(
            output, torch.randn(7, 3) * 1e-4, model
        ).mean()
        loss.backward()
        self.assertTrue(any(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ))

    def test_market_encoder_unlocks_input_dependent_density_learning(self) -> None:
        torch.manual_seed(7)
        model = self.make_model()
        torch.testing.assert_close(
            model.market.output.weight,
            torch.eye(model.market_width),
        )
        features = torch.randn(11, 5)
        targets = torch.randn(11, 3) * 1e-4
        loss = -path_log_density_terms(model(features), targets, model).mean()
        loss.backward()
        head_gradient = model.initial_history.output.weight.grad
        self.assertIsNotNone(head_gradient)
        self.assertGreater(float(head_gradient.norm()), 0.0)

        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-1e-3)
        model.zero_grad(set_to_none=True)
        probe = torch.randn(11, 5, requires_grad=True)
        output = model(probe)
        probe_loss = -path_log_density_terms(output, targets, model).mean()
        probe_loss.backward()
        self.assertGreater(float(probe.grad.norm()), 0.0)
        self.assertGreater(
            float(output.expectations.detach().std(dim=0).max()), 0.0
        )

    def test_every_stage_projection_contributes_to_an_output(self) -> None:
        torch.manual_seed(11)
        model = self.make_model()
        targets = torch.randn(9, 3) * 1e-4
        loss = -path_log_density_terms(
            model(torch.randn(9, 5)), targets, model
        ).mean()
        loss.backward()
        for block in model.history_down_projects:
            self.assertIsNotNone(block.output.weight.grad)
            self.assertGreater(float(block.output.weight.grad.norm()), 0.0)
        for block in model.distribution_state_blocks:
            self.assertIsNotNone(block.output.weight.grad)
            self.assertGreater(float(block.output.weight.grad.norm()), 0.0)


if __name__ == "__main__":
    unittest.main()
