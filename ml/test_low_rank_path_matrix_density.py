from __future__ import annotations

import copy
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from compressed_path_return_density import path_log_density_terms
from low_rank_path_matrix_density import (
    CyclicDenseCompressedPathMatrixDensity,
    DirectFactorizedPathMatrixDensity,
    DynamicLowRankPathMatrixDensity,
    JointPrefixContractedCyclicPathMatrixDensity,
)
from return_knot_density import KnotDensityContract
from return_oracle_ce import fused_glu


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

    def test_direct_factorized_model_uses_unpacked_expected_widths(self) -> None:
        model = DirectFactorizedPathMatrixDensity(
            torch.zeros(15),
            torch.ones(15),
            self.density,
            market_width=16,
            path_embedding_width=4,
            path_count=6,
            return_count=3,
            factor_rank=4,
            hidden_width_threshold=16,
            initial_radius=0.0031622776601683794,
            minimum_radius=0.0001,
            learnable_centering=False,
        )
        path_block = model.path_transitions[0].generator
        expected_input = 6 * 4 + 16 + 6 * 32 + 32
        expected_hidden = (expected_input + 6 * 4) // 2
        self.assertEqual(
            path_block.value_projection.output_width,
            expected_hidden,
        )
        self.assertIsNot(
            path_block.value_projection.left,
            path_block.gate_projection.left,
        )
        self.assertEqual(path_block.value_projection.rank, 4)
        self.assertIsNone(path_block.value_centering.weight)

        block_input = torch.randn(
            3,
            path_block.value_projection.input_width,
        )
        reference_hidden, _raw_value, _raw_gate = fused_glu(
            torch.cat((
                path_block.value_projection(block_input),
                path_block.gate_projection(block_input),
            ), dim=1),
            path_block.value_centering,
            path_block.gate_centering,
            path_block.value_bias,
            path_block.gate_bias,
            path_block.value_transform,
            path_block.gate_transform,
        )
        torch.testing.assert_close(
            path_block(block_input),
            path_block.output(reference_hidden),
            atol=1e-6,
            rtol=1e-5,
        )

        features = torch.randn(5, 15)
        targets = torch.randn(5, 3) * 1e-4
        output = model(features)
        self.assertEqual(output.expectations.shape, (5, 3))
        for log_masses in output.log_masses:
            torch.testing.assert_close(
                log_masses.exp().sum(dim=1),
                torch.ones(5),
                atol=1e-5,
                rtol=1e-5,
            )
        (-path_log_density_terms(output, targets, model).mean()).backward()
        for parameter in (
            path_block.value_projection.left,
            path_block.gate_projection.left,
            path_block.output.right,
        ):
            self.assertIsNotNone(parameter.grad)
            assert parameter.grad is not None
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_single_cyclic_block_is_shared_across_every_step(self) -> None:
        model = CyclicDenseCompressedPathMatrixDensity(
            torch.zeros(15),
            torch.ones(15),
            self.density,
            market_width=16,
            path_embedding_width=4,
            path_count=6,
            return_count=3,
            stage_block_count=1,
            path_compression_width=8,
            joint_compression_width=12,
            initial_radius=0.0031622776601683794,
            minimum_radius=0.0001,
            learnable_centering=False,
        )
        shared_groups = (
            model.market_transitions,
            model.query_heads,
            model.return_heads,
            model.point_heads,
            model.path_compressors,
            model.joint_compressors,
            model.path_transitions,
        )
        self.assertTrue(all(len(group) == 1 for group in shared_groups))

        calls = 0

        def count_query_calls(_module, _inputs, _output) -> None:
            nonlocal calls
            calls += 1

        handle = model.query_heads[0].register_forward_hook(count_query_calls)
        try:
            features = torch.randn(5, 15)
            targets = torch.randn(5, 3) * 1e-4
            output = model(features)
        finally:
            handle.remove()
        self.assertEqual(calls, model.return_count)
        self.assertEqual(output.expectations.shape, (5, 3))
        (-path_log_density_terms(output, targets, model).mean()).backward()
        self.assertIsNotNone(model.query_heads[0].output.weight.grad)
        self.assertIsNotNone(model.horizon_embedding.grad)
        assert model.horizon_embedding.grad is not None
        self.assertTrue(bool(torch.isfinite(model.horizon_embedding.grad).all()))

    def test_single_step_cyclic_model_emits_one_density(self) -> None:
        model = CyclicDenseCompressedPathMatrixDensity(
            torch.zeros(15),
            torch.ones(15),
            self.density,
            market_width=16,
            path_embedding_width=4,
            path_count=6,
            return_count=1,
            stage_block_count=1,
            path_compression_width=8,
            joint_compression_width=12,
            initial_radius=0.0031622776601683794,
            minimum_radius=0.0001,
            learnable_centering=False,
        )
        features = torch.randn(5, 15)
        targets = torch.randn(5, 1) * 1e-4
        output = model(features)
        self.assertEqual(output.expectations.shape, (5, 1))
        self.assertEqual(len(output.log_masses), 1)
        torch.testing.assert_close(
            output.log_masses[0].exp().sum(dim=1),
            torch.ones(5),
            atol=1e-5,
            rtol=1e-5,
        )
        (-path_log_density_terms(output, targets, model).mean()).backward()
        self.assertIsNotNone(model.query_heads[0].output.weight.grad)

    def test_cyclic_model_exposes_complete_recurrent_transition(self) -> None:
        model = CyclicDenseCompressedPathMatrixDensity(
            torch.zeros(15),
            torch.ones(15),
            self.density,
            market_width=16,
            path_embedding_width=4,
            path_count=6,
            return_count=3,
            stage_block_count=1,
            path_compression_width=8,
            joint_compression_width=12,
            initial_radius=0.0031622776601683794,
            minimum_radius=0.0001,
            learnable_centering=False,
        )
        features = torch.randn(5, 15)
        emissions = model.recurrent_rollout(features, 4)
        output = model(features)

        self.assertEqual(len(emissions), 4)
        self.assertEqual(emissions[0].paths.shape, (5, 6, 4))
        self.assertEqual(emissions[0].conditioned_market.shape, (5, 16))
        torch.testing.assert_close(
            emissions[1].paths, emissions[0].next_state.paths
        )
        torch.testing.assert_close(
            emissions[1].conditioned_market,
            emissions[0].next_state.market + model.horizon_embedding[1],
        )
        for actual, exposed in zip(
            output.log_masses, emissions[:3], strict=True
        ):
            torch.testing.assert_close(actual, exposed.log_masses)

    def test_cyclic_model_applies_conditional_return_normalization(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        density = KnotDensityContract.load(
            repo / "data/benchmarks/"
            "one-second-trailing-2h-normalized-return-density-k32-v1.json",
            fit="32",
        )
        model = CyclicDenseCompressedPathMatrixDensity(
            torch.zeros(17),
            torch.ones(17),
            density,
            market_width=16,
            path_embedding_width=4,
            path_count=6,
            return_count=3,
            stage_block_count=1,
            path_compression_width=8,
            joint_compression_width=12,
            initial_radius=0.0031622776601683794,
            minimum_radius=0.0001,
            learnable_centering=False,
            target_normalization_variance_floor=1e-16,
        )
        features = torch.randn(5, 17)
        features[:, -2] = torch.linspace(-2e-6, 2e-6, 5)
        features[:, -1] = torch.linspace(4e-10, 2e-9, 5)
        output = model(features)
        self.assertIsNotNone(output.normalization_location)
        self.assertIsNotNone(output.normalization_scale)
        assert output.normalization_location is not None
        assert output.normalization_scale is not None
        torch.testing.assert_close(output.normalization_location, features[:, -2])
        torch.testing.assert_close(
            output.normalization_scale, torch.sqrt(features[:, -1])
        )

        normalized_targets = torch.randn(5, 3)
        raw_targets = (
            features[:, -2, None]
            + torch.sqrt(features[:, -1, None]) * normalized_targets
        )
        raw_terms = path_log_density_terms(output, raw_targets, model)
        normalized_terms = path_log_density_terms(
            replace(
                output,
                normalization_location=None,
                normalization_scale=None,
            ),
            normalized_targets,
            model,
        )
        torch.testing.assert_close(
            raw_terms,
            normalized_terms - 0.5 * torch.log(features[:, -1, None]),
            atol=2e-5,
            rtol=1e-5,
        )

    def test_log_price_normalization_scales_without_return_centering(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        density = KnotDensityContract.load(
            repo / "data/benchmarks/"
            "one-second-trailing-2h-log-price-normalized-difference-"
            "density-k32-v1.json",
            fit="32",
        )
        model = CyclicDenseCompressedPathMatrixDensity(
            torch.zeros(17),
            torch.ones(17),
            density,
            market_width=16,
            path_embedding_width=4,
            path_count=6,
            return_count=3,
            stage_block_count=1,
            path_compression_width=8,
            joint_compression_width=12,
            initial_radius=0.0031622776601683794,
            minimum_radius=0.0001,
            learnable_centering=False,
            target_normalization_variance_floor=1e-16,
            target_normalization_center=False,
        )
        features = torch.randn(5, 17)
        features[:, -2] = torch.linspace(10.9, 11.1, 5)
        features[:, -1] = torch.linspace(4e-7, 2e-5, 5)
        output = model(features)
        assert output.normalization_location is not None
        assert output.normalization_scale is not None
        torch.testing.assert_close(
            output.normalization_location, torch.zeros(5)
        )
        torch.testing.assert_close(
            output.normalization_scale, torch.sqrt(features[:, -1])
        )

        normalized_targets = torch.randn(5, 3) * 0.05
        raw_targets = torch.sqrt(features[:, -1, None]) * normalized_targets
        raw_terms = path_log_density_terms(output, raw_targets, model)
        normalized_terms = path_log_density_terms(
            replace(
                output,
                normalization_location=None,
                normalization_scale=None,
            ),
            normalized_targets,
            model,
        )
        torch.testing.assert_close(
            raw_terms,
            normalized_terms - 0.5 * torch.log(features[:, -1, None]),
            atol=2e-5,
            rtol=1e-5,
        )

    def test_cyclic_input_dropout_is_training_only(self) -> None:
        model = CyclicDenseCompressedPathMatrixDensity(
            torch.zeros(17),
            torch.ones(17),
            self.density,
            market_width=16,
            path_embedding_width=4,
            path_count=6,
            return_count=1,
            stage_block_count=1,
            path_compression_width=8,
            joint_compression_width=12,
            initial_radius=0.0031622776601683794,
            minimum_radius=0.0001,
            learnable_centering=False,
            input_dropout_probability=0.5,
        )
        features = torch.randn(32, 17)
        captured: list[torch.Tensor] = []
        hook = model.input_encoder.register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[0].detach().clone())
        )
        model.eval()
        model(features)
        model(features)
        torch.testing.assert_close(captured[-2], captured[-1])
        model.train()
        torch.manual_seed(1)
        model(features)
        torch.manual_seed(2)
        model(features)
        hook.remove()
        self.assertFalse(torch.equal(captured[-2], captured[-1]))
        self.assertGreater(int((captured[-1] == 0).sum()), 0)

    def test_cyclic_embedding_dropout_is_after_input_encoder_and_training_only(
        self,
    ) -> None:
        model = CyclicDenseCompressedPathMatrixDensity(
            torch.zeros(17),
            torch.ones(17),
            self.density,
            market_width=16,
            path_embedding_width=4,
            path_count=6,
            return_count=1,
            stage_block_count=1,
            path_compression_width=8,
            joint_compression_width=12,
            initial_radius=0.0031622776601683794,
            minimum_radius=0.0001,
            learnable_centering=False,
            embedding_dropout_probability=0.5,
        )
        features = torch.randn(32, 17)
        encoder_inputs: list[torch.Tensor] = []
        embedding_outputs: list[torch.Tensor] = []
        input_hook = model.input_encoder.register_forward_pre_hook(
            lambda _module, inputs: encoder_inputs.append(inputs[0].detach().clone())
        )
        embedding_hook = model.initial_paths.register_forward_pre_hook(
            lambda _module, inputs: embedding_outputs.append(
                inputs[0].detach().clone()
            )
        )
        model.eval()
        model(features)
        model(features)
        torch.testing.assert_close(encoder_inputs[-2], encoder_inputs[-1])
        torch.testing.assert_close(embedding_outputs[-2], embedding_outputs[-1])
        model.train()
        torch.manual_seed(1)
        model(features)
        torch.manual_seed(2)
        model(features)
        input_hook.remove()
        embedding_hook.remove()
        torch.testing.assert_close(encoder_inputs[-2], encoder_inputs[-1])
        self.assertFalse(torch.equal(embedding_outputs[-2], embedding_outputs[-1]))
        self.assertGreater(int((embedding_outputs[-1] == 0).sum()), 0)

    def make_joint_prefix_model(
        self,
        return_count: int = 3,
    ) -> JointPrefixContractedCyclicPathMatrixDensity:
        return JointPrefixContractedCyclicPathMatrixDensity(
            torch.zeros(15),
            torch.ones(15),
            self.density,
            market_width=16,
            path_embedding_width=4,
            path_count=6,
            return_count=return_count,
            stage_block_count=1,
            path_compression_width=8,
            joint_compression_width=12,
            initial_radius=0.0031622776601683794,
            minimum_radius=0.0001,
            learnable_centering=False,
        )

    def test_joint_prefix_contraction_preserves_distributional_forecast(
        self,
    ) -> None:
        torch.manual_seed(17)
        model = self.make_joint_prefix_model()
        features = torch.randn(5, 15)
        targets = torch.randn(5, 3) * 1e-4
        forecast = model(features)
        contracted = model(features, targets)
        self.assertIsNone(forecast.joint_log_density_terms)
        self.assertEqual(contracted.joint_log_density_terms.shape, targets.shape)
        torch.testing.assert_close(
            forecast.expectations, contracted.expectations
        )
        for left, right in zip(
            forecast.log_masses, contracted.log_masses, strict=True
        ):
            torch.testing.assert_close(left, right)
            torch.testing.assert_close(
                right.exp().sum(dim=1), torch.ones(features.shape[0])
            )

    def test_joint_prefix_terms_condition_on_the_realized_prefix(self) -> None:
        torch.manual_seed(19)
        model = self.make_joint_prefix_model()
        with torch.no_grad():
            model.return_heads[0].output.weight.normal_(0, 0.2)
            model.joint_compressors[0].output.weight.normal_(0, 0.2)
            model.path_transitions[0].output.weight.normal_(0, 0.2)
        features = torch.randn(5, 15)
        first = torch.randn(5, 3) * 1e-4
        changed_prefix = first.clone()
        changed_prefix[:, 0] += 3e-4
        first_terms = path_log_density_terms(
            model(features, first), first, model
        )
        changed_terms = path_log_density_terms(
            model(features, changed_prefix), changed_prefix, model
        )
        # Lead one has the same target in both paths.  Its density changes only
        # because lead zero changed the contracted prefix probabilities.
        self.assertGreater(
            float((
                first_terms[:, 1] - changed_terms[:, 1]
            ).detach().abs().max()),
            1e-6,
        )

    def test_joint_prefix_nll_backpropagates_through_embedding_compression(
        self,
    ) -> None:
        torch.manual_seed(23)
        model = self.make_joint_prefix_model()
        features = torch.randn(5, 15)
        targets = torch.randn(5, 3) * 1e-4
        loss = -path_log_density_terms(
            model(features, targets), targets, model
        ).mean()
        loss.backward()
        gradient = model.joint_compressors[0].output.weight.grad
        self.assertIsNotNone(gradient)
        assert gradient is not None
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.abs().sum()), 0)

    def test_recurrent_activation_recomputation_matches_15_step_unroll(
        self,
    ) -> None:
        torch.manual_seed(29)
        unrolled = self.make_joint_prefix_model(return_count=15)
        recomputed = copy.deepcopy(unrolled)
        recomputed.recurrent_activation_checkpointing = True
        # Exercise the same step-zero/market-offset path used by the compiled
        # shared-step wrapper without paying compiler startup in a unit test.
        recomputed._compiled_joint_recurrent_step = (
            recomputed._joint_recurrent_step_tensors
        )
        unrolled.train()
        recomputed.train()
        unrolled_features = torch.randn(2, 15, requires_grad=True)
        recomputed_features = unrolled_features.detach().clone().requires_grad_(
            True
        )
        targets = torch.randn(2, 15) * 1e-4

        unrolled_output = unrolled(unrolled_features, targets)
        recomputed_output = recomputed(recomputed_features, targets)
        torch.testing.assert_close(
            unrolled_output.expectations,
            recomputed_output.expectations,
        )
        assert unrolled_output.joint_log_density_terms is not None
        assert recomputed_output.joint_log_density_terms is not None
        torch.testing.assert_close(
            unrolled_output.joint_log_density_terms,
            recomputed_output.joint_log_density_terms,
        )
        for unrolled_masses, recomputed_masses in zip(
            unrolled_output.log_masses,
            recomputed_output.log_masses,
            strict=True,
        ):
            torch.testing.assert_close(unrolled_masses, recomputed_masses)

        unrolled_loss = -unrolled_output.joint_log_density_terms.mean()
        recomputed_loss = -recomputed_output.joint_log_density_terms.mean()
        torch.testing.assert_close(unrolled_loss, recomputed_loss)
        unrolled_loss.backward()
        recomputed_loss.backward()
        torch.testing.assert_close(
            unrolled_features.grad,
            recomputed_features.grad,
            rtol=1e-5,
            atol=1e-7,
        )
        unrolled_parameters = dict(unrolled.named_parameters())
        recomputed_parameters = dict(recomputed.named_parameters())
        self.assertEqual(
            unrolled_parameters.keys(), recomputed_parameters.keys()
        )
        for name, unrolled_parameter in unrolled_parameters.items():
            recomputed_parameter = recomputed_parameters[name]
            self.assertEqual(
                unrolled_parameter.grad is None,
                recomputed_parameter.grad is None,
                name,
            )
            if unrolled_parameter.grad is not None:
                torch.testing.assert_close(
                    unrolled_parameter.grad,
                    recomputed_parameter.grad,
                    rtol=1e-5,
                    atol=1e-7,
                    msg=lambda message, parameter=name: (
                        f"{parameter}: {message}"
                    ),
                )

    def test_contracted_only_joint_nll_matches_full_15_step_unroll(
        self,
    ) -> None:
        torch.manual_seed(31)
        unrolled = self.make_joint_prefix_model(return_count=15)
        contracted = copy.deepcopy(unrolled)
        contracted.recurrent_activation_checkpointing = True
        # Exercise the compiled-step boundary in eager mode so this test
        # proves the optimized execution path, including recomputation,
        # without depending on compiler availability in the test runner.
        contracted._compiled_contracted_joint_step = (
            contracted._contracted_joint_step_tensors
        )
        unrolled.train()
        contracted.train()
        unrolled_features = torch.randn(2, 15, requires_grad=True)
        contracted_features = (
            unrolled_features.detach().clone().requires_grad_(True)
        )
        targets = torch.randn(2, 15) * 1e-4

        unrolled_output = unrolled(unrolled_features, targets)
        assert unrolled_output.joint_log_density_terms is not None
        contracted_terms, first_expectation = (
            contracted.contracted_joint_training_terms(
                contracted_features,
                targets,
            )
        )
        torch.testing.assert_close(
            contracted_terms,
            unrolled_output.joint_log_density_terms,
        )
        torch.testing.assert_close(
            first_expectation,
            unrolled_output.expectations[:, 0],
        )

        unrolled_loss = -unrolled_output.joint_log_density_terms.mean()
        contracted_loss = -contracted_terms.mean()
        torch.testing.assert_close(contracted_loss, unrolled_loss)
        unrolled_loss.backward()
        contracted_loss.backward()
        torch.testing.assert_close(
            contracted_features.grad,
            unrolled_features.grad,
            rtol=1e-5,
            atol=1e-7,
        )
        unrolled_parameters = dict(unrolled.named_parameters())
        contracted_parameters = dict(contracted.named_parameters())
        self.assertEqual(unrolled_parameters.keys(), contracted_parameters.keys())
        for name, unrolled_parameter in unrolled_parameters.items():
            contracted_parameter = contracted_parameters[name]
            self.assertEqual(
                unrolled_parameter.grad is None,
                contracted_parameter.grad is None,
                name,
            )
            if unrolled_parameter.grad is not None:
                torch.testing.assert_close(
                    contracted_parameter.grad,
                    unrolled_parameter.grad,
                    rtol=1e-5,
                    atol=1e-7,
                    msg=lambda message, parameter=name: (
                        f"{parameter}: {message}"
                    ),
                )


if __name__ == "__main__":
    unittest.main()
