from __future__ import annotations

from pathlib import Path
import unittest

import torch

from actor_market_process_density import (
    ActorMarketProcessDensity,
    minimum_top_probability,
)
from compressed_path_return_density import path_log_density_terms
from return_knot_density import KnotDensityContract


class ActorMarketProcessDensityTest(unittest.TestCase):
    def make_model(self) -> ActorMarketProcessDensity:
        density = KnotDensityContract.load(
            Path("data/benchmarks/one-second-return-knot-scaling-v1.json"),
            fit="32",
        )
        return ActorMarketProcessDensity(
            torch.zeros(15),
            torch.ones(15),
            density,
            embedding_width=16,
            actor_width=8,
            actor_decision_width=16,
            market_width=10,
            actor_count=3,
            market_count=4,
            action_count=4,
            action_basis_width=5,
            reward_width=3,
            return_count=3,
            certainty_maximum=0.95,
            initial_radius=0.0031622776601683794,
            minimum_radius=1e-4,
            learnable_centering=False,
        )

    def test_confidence_projection_preserves_probability_rows(self) -> None:
        probabilities = torch.tensor([
            [[0.40, 0.30, 0.20, 0.10], [0.70, 0.10, 0.10, 0.10]],
        ])
        minimum = torch.tensor([[[0.60], [0.60]]])
        result = minimum_top_probability(probabilities, minimum)
        torch.testing.assert_close(result.sum(dim=-1), torch.ones(1, 2))
        torch.testing.assert_close(result.amax(dim=-1), torch.tensor([[0.60, 0.70]]))
        torch.testing.assert_close(
            result[0, 0, 1:] / result[0, 0, 1:].sum(),
            probabilities[0, 0, 1:] / probabilities[0, 0, 1:].sum(),
        )

    def test_recurrent_outputs_form_triangular_densities(self) -> None:
        torch.manual_seed(17)
        model = self.make_model()
        output = model(torch.randn(5, 15))
        self.assertEqual(output.expectations.shape, (5, 3))
        self.assertEqual(len(output.log_masses), 3)
        self.assertIsNotNone(output.knots_unit)
        for masses, knots in zip(
            output.log_masses, output.knots_unit or (), strict=True
        ):
            self.assertEqual(masses.shape, (5, 32))
            torch.testing.assert_close(masses.exp().sum(dim=1), torch.ones(5))
            self.assertTrue(bool((knots[:, 1:] > knots[:, :-1]).all()))

    def test_actor_market_shapes_and_certainty_contract(self) -> None:
        torch.manual_seed(19)
        model = self.make_model()
        actors, markets = model.initial_state(torch.randn(6, 15))
        self.assertEqual(actors.shape, (6, 3, 8))
        self.assertEqual(markets.shape, (6, 4, 10))
        importance, rewards, certainty, action_values, impact = (
            model.actor_decisions(actors)
        )
        self.assertEqual(importance.shape, (6, 3, 4))
        self.assertEqual(rewards.shape, (6, 3, 10, 3))
        self.assertEqual(impact.shape, (6, 3, 4))
        distribution = model.action_distribution(
            action_values, certainty, markets
        )
        self.assertEqual(distribution.shape, (6, 3, 4, 4))
        torch.testing.assert_close(
            distribution.sum(dim=3), torch.ones(6, 3, 4)
        )
        self.assertTrue(bool(
            (distribution.amax(dim=3) >= model.minimum_certainty).all()
        ))
        next_actors, next_markets = model.process_step(actors, markets)
        self.assertEqual(next_actors.shape, actors.shape)
        self.assertEqual(next_markets.shape, markets.shape)

    def test_every_architecture_path_receives_finite_gradient(self) -> None:
        torch.manual_seed(23)
        model = self.make_model()
        features = torch.randn(7, 15)
        targets = torch.randn(7, 3) * 1e-4
        loss = -path_log_density_terms(model(features), targets, model).mean()
        loss.backward()
        parameters = (
            model.initial_actors.output.weight,
            model.initial_markets.output.weight,
            model.actor_decision_trunk.output.weight,
            model.actor_decision_projection.weight,
            model.actor_transition.output.weight,
            model.market_transition.output.weight,
            model.action_vectors,
            model.action_to_market_delta,
            model.output_embedding.output.weight,
            model.output_projection.weight,
        )
        for parameter in parameters:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(float(parameter.grad.norm()), 0.0)


if __name__ == "__main__":
    unittest.main()
