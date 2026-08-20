from __future__ import annotations

from pathlib import Path
import unittest

import torch

from compressed_path_return_density import path_log_density_terms
from recurrent_market_path_density import RecurrentMarketPathDensity
from return_knot_density import KnotDensityContract


class RecurrentMarketPathDensityTest(unittest.TestCase):
    def make_model(self) -> RecurrentMarketPathDensity:
        density = KnotDensityContract.load(
            Path("data/benchmarks/one-second-return-knot-scaling-v1.json"),
            fit="32",
        )
        return RecurrentMarketPathDensity(
            torch.zeros(15), torch.ones(15), density,
            market_width=16, state_widths=(8, 12, 12),
            initial_radius=0.0031622776601683794,
            minimum_radius=1e-4, learnable_centering=False,
        )

    def test_fixed_output_marginals_are_normalized(self) -> None:
        model = self.make_model()
        output = model(torch.randn(7, 15))
        self.assertEqual(output.expectations.shape, (7, 3))
        self.assertEqual([value.shape for value in output.log_masses], [
            (7, 32), (7, 32), (7, 32),
        ])
        for value in output.log_masses:
            torch.testing.assert_close(value.exp().sum(dim=1), torch.ones(7))

    def test_every_nonterminal_path_has_finite_gradient(self) -> None:
        torch.manual_seed(19)
        model = self.make_model()
        targets = torch.randn(9, 3) * 1e-4
        loss = -path_log_density_terms(
            model(torch.randn(9, 15)), targets, model
        ).mean()
        loss.backward()
        for blocks in (
            model.market_transitions,
            model.transition_factor_blocks,
            model.history_down_projects,
        ):
            for block in blocks:
                self.assertIsNotNone(block.output.weight.grad)
                self.assertTrue(bool(torch.isfinite(block.output.weight.grad).all()))
                self.assertGreater(float(block.output.weight.grad.norm()), 0.0)
        for parameter in (*model.destination_factors, *model.destination_biases):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.norm()), 0.0)


if __name__ == "__main__":
    unittest.main()
