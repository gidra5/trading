from __future__ import annotations

from pathlib import Path
import unittest

import torch

from compressed_path_return_density import path_log_density_terms
from recurrent_market_path_density import (
    RecurrentMarketPathDensity,
    ResidualRecurrentMarketPathDensity,
)
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

    def make_residual_model(self) -> ResidualRecurrentMarketPathDensity:
        density = KnotDensityContract.load(
            Path("data/benchmarks/one-second-return-knot-scaling-v1.json"),
            fit="32",
        )
        return ResidualRecurrentMarketPathDensity(
            torch.zeros(15), torch.ones(15), density,
            market_width=16, state_widths=(8, 12, 12),
            initial_radius=0.0031622776601683794,
            minimum_radius=1e-4, learnable_centering=False,
        )

    def make_ranked_residual_model(self) -> ResidualRecurrentMarketPathDensity:
        density = KnotDensityContract.load(
            Path("data/benchmarks/one-second-return-knot-scaling-v1.json"),
            fit="32",
        )
        return ResidualRecurrentMarketPathDensity(
            torch.zeros(15), torch.ones(15), density,
            market_width=16, state_widths=(8, 12, 12), transition_rank=4,
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

    def test_residual_transition_is_normalized_and_fully_trainable(self) -> None:
        torch.manual_seed(23)
        model = self.make_residual_model()
        output = model(torch.randn(9, 15))
        for value in output.log_masses:
            torch.testing.assert_close(value.exp().sum(dim=1), torch.ones(9))
        targets = torch.randn(9, 3) * 1e-4
        loss = -path_log_density_terms(output, targets, model).mean()
        loss.backward()
        for parameter in (*model.baseline_logits, *model.mixture_logits):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(float(parameter.grad.norm()), 0.0)

    def test_residual_contraction_matches_explicit_transition(self) -> None:
        torch.manual_seed(29)
        model = self.make_residual_model()
        batch = 5
        step = 1
        width = model.state_widths[step]
        q = torch.softmax(torch.randn(batch, width), dim=1)
        factor = torch.randn(batch, width)
        destination = model.destination_factors[step]
        bias = model.destination_biases[step]
        explicit = torch.bmm(
            q[:, None, :],
            model.conditional_transition(step, factor, destination, bias),
        ).squeeze(1)
        contracted = model.contract_transition(
            step, q, factor, destination, bias
        )
        torch.testing.assert_close(contracted, explicit, rtol=1e-5, atol=1e-7)

    def test_ranked_residual_transition_is_trainable(self) -> None:
        torch.manual_seed(31)
        model = self.make_ranked_residual_model()
        output = model(torch.randn(7, 15))
        for value in output.log_masses:
            torch.testing.assert_close(value.exp().sum(dim=1), torch.ones(7))
        loss = -path_log_density_terms(
            output, torch.randn(7, 3) * 1e-4, model
        ).mean()
        loss.backward()
        self.assertIsNotNone(model.source_factors)
        for parameter in (
            *model.source_factors,
            *model.destination_factors,
            *model.baseline_logits,
            *model.mixture_logits,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(float(parameter.grad.norm()), 0.0)


if __name__ == "__main__":
    unittest.main()
