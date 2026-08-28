from __future__ import annotations

from pathlib import Path
import unittest

import torch

from actor_market_path_matrix_density import ActorMarketPathMatrixDensity
from compressed_path_return_density import path_log_density_terms
from return_knot_density import KnotDensityContract


class ActorMarketPathMatrixDensityTest(unittest.TestCase):
    def make_model(self) -> ActorMarketPathMatrixDensity:
        density = KnotDensityContract.load(
            Path("data/benchmarks/one-second-return-knot-scaling-v1.json"),
            fit="32",
        )
        return ActorMarketPathMatrixDensity(
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
            path_embedding_width=4,
            path_count=6,
            return_count=3,
            stage_block_count=1,
            path_compression_width=8,
            joint_compression_width=12,
            certainty_maximum=0.95,
            initial_radius=0.0031622776601683794,
            minimum_radius=1e-4,
            learnable_centering=False,
        )

    def test_joint_actor_market_state_conditions_shared_path_block(self) -> None:
        torch.manual_seed(29)
        model = self.make_model()
        self.assertEqual(
            model.actor_state_compressor.projection.in_features,
            model.actor_count * model.actor_width,
        )
        self.assertEqual(
            model.market_state_compressor.projection.in_features,
            model.market_count * model.market_width,
        )
        self.assertEqual(
            model.state_fusion.projection.in_features,
            2 * model.embedding_width,
        )
        self.assertEqual(len(model.query_heads), 1)
        self.assertEqual(len(model.path_transitions), 1)

        query_calls = 0
        state_calls = 0

        def count_query(_module, _inputs, _output) -> None:
            nonlocal query_calls
            query_calls += 1

        def count_state(_module, _inputs, _output) -> None:
            nonlocal state_calls
            state_calls += 1

        handles = (
            model.query_heads[0].register_forward_hook(count_query),
            model.state_fusion.register_forward_hook(count_state),
        )
        try:
            output = model(torch.randn(5, 15))
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(query_calls, model.return_count)
        self.assertEqual(state_calls, model.return_count)
        self.assertEqual(output.expectations.shape, (5, 3))
        for masses in output.log_masses:
            torch.testing.assert_close(masses.exp().sum(dim=1), torch.ones(5))

    def test_actor_market_and_path_branches_receive_gradient(self) -> None:
        torch.manual_seed(31)
        model = self.make_model()
        features = torch.randn(7, 15)
        targets = torch.randn(7, 3) * 1e-4
        loss = -path_log_density_terms(model(features), targets, model).mean()
        loss.backward()
        parameters = (
            model.initial_actors.output.weight,
            model.initial_markets.output.weight,
            model.actor_decision_projection.weight,
            model.actor_transition.output.weight,
            model.market_transition.output.weight,
            model.actor_state_compressor.output.weight,
            model.market_state_compressor.output.weight,
            model.state_fusion.output.weight,
            model.initial_paths.weight,
            model.query_heads[0].output.weight,
            model.return_heads[0].output.weight,
            model.point_heads[0].output.weight,
            model.path_compressors[0].output.weight,
            model.joint_compressors[0].output.weight,
            model.path_transitions[0].output.weight,
            model.horizon_embedding,
        )
        for parameter in parameters:
            self.assertIsNotNone(parameter.grad)
            assert parameter.grad is not None
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(float(parameter.grad.norm()), 0.0)


if __name__ == "__main__":
    unittest.main()
