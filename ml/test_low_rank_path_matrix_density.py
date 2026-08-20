from __future__ import annotations

import unittest
from pathlib import Path

import torch

from compressed_path_return_density import path_log_density_terms
from low_rank_path_matrix_density import DynamicLowRankPathMatrixDensity
from return_knot_density import KnotDensityContract


class DynamicLowRankPathMatrixDensityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        repo = Path(__file__).resolve().parents[1]
        cls.density = KnotDensityContract.load(
            repo / "data/benchmarks/one-second-return-knot-scaling-v1.json",
            fit="32",
        )

    def make_model(self) -> DynamicLowRankPathMatrixDensity:
        return DynamicLowRankPathMatrixDensity(
            torch.zeros(15),
            torch.ones(15),
            self.density,
            market_width=16,
            path_embedding_width=4,
            path_count=6,
            return_count=3,
            matrix_rank=2,
            hidden_width_cap=16,
            initial_radius=0.0031622776601683794,
            minimum_radius=0.0001,
            learnable_centering=False,
        )

    def test_forward_emits_normalized_dynamic_densities(self) -> None:
        model = self.make_model()
        output = model(torch.randn(5, 15))

        self.assertEqual(output.expectations.shape, (5, 3))
        self.assertEqual(len(output.log_masses), 3)
        self.assertIsNotNone(output.knots_unit)
        self.assertIsNotNone(output.areas_unit)
        self.assertIsNotNone(output.component_means)
        assert output.knots_unit is not None
        assert output.areas_unit is not None
        for log_masses, knots, areas in zip(
            output.log_masses,
            output.knots_unit,
            output.areas_unit,
            strict=True,
        ):
            self.assertEqual(log_masses.shape, (5, 32))
            self.assertTrue(torch.allclose(
                log_masses.exp().sum(dim=1), torch.ones(5), atol=1e-5
            ))
            self.assertTrue(torch.allclose(knots[:, 0], torch.zeros(5)))
            self.assertTrue(torch.allclose(knots[:, -1], torch.ones(5)))
            self.assertTrue(bool(((knots[:, 1:] - knots[:, :-1]) > 0).all()))
            self.assertTrue(bool((areas > 0).all()))

    def test_nll_reaches_dynamic_matrices_points_and_recurrence(self) -> None:
        torch.manual_seed(7)
        model = self.make_model()
        features = torch.randn(7, 15)
        targets = torch.randn(7, 3) * 1e-4
        loss = -path_log_density_terms(
            model(features), targets, model
        ).mean()
        loss.backward()

        parameters = {
            "query": model.query_heads[0].generator.output.weight,
            "return": model.return_heads[0].generator.output.weight,
            "points": model.point_heads[0].output.weight,
            "recurrent": model.path_transitions[0].generator.output.weight,
            "next-market": model.market_transitions[0].projection.weight,
        }
        for name, parameter in parameters.items():
            self.assertIsNotNone(parameter.grad, name)
            assert parameter.grad is not None
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0, name)

    def test_nll_gradient_is_finite_at_dynamic_knot_locations(self) -> None:
        model = self.make_model()
        features = torch.randn(4, 15)
        preliminary = model(features)
        assert preliminary.knots_unit is not None
        unit = preliminary.knots_unit[0][:, 8].detach()
        transform = model.density_transform
        stable = unit.clamp(torch.finfo(unit.dtype).eps, 1 - torch.finfo(unit.dtype).eps)
        z = torch.sinh((torch.log(stable) - torch.log1p(-stable)) / transform.alpha)
        return_at_knot = (
            transform.location_bps + transform.scale_bps * z
        ) / 10000
        targets = return_at_knot[:, None].expand(-1, 3).clone()
        loss = -path_log_density_terms(model(features), targets, model).mean()
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))


if __name__ == "__main__":
    unittest.main()
