from __future__ import annotations

import math
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from return_oracle_ce import (
    BRANCH_NORMALIZATION_EPSILON,
    BRANCH_NORMALIZATION_INITIAL_RADIUS,
    HIDDEN_LAYER_COUNT,
    HIDDEN_WIDTHS,
    INPUT_RETURN_COUNT,
    LearnableCenteringNorm,
    OUTPUT_ACTION_COUNT,
    ReturnOracleMlp,
    centering_matrix_constraint_components,
    distribution_layer_components,
    dropout_gate_probability,
    fold_input_normalization_for_export,
    fused_glu,
    oracle_policy_cross_entropy,
    oracle_policy_objective,
    parameter_count,
    simple_return_features,
    soft_layer_norm_components,
    soft_weight_bound_penalty,
)
from train_return_oracle_ce import (
    Segment,
    SourceShard,
    add_batch_invariant_regularizers,
    compact_completed_minute_batch_rows,
    completed_minute_row_indices,
    completed_minute_simple_return_rows,
    copy_component_to_tensor,
    group_segments_by_target,
    group_batches,
    hybrid_optimizer_parameters,
    replace_file_with_retry,
)


class ReturnOracleCeTest(unittest.TestCase):
    @staticmethod
    def model(dropout: float = 0) -> ReturnOracleMlp:
        return ReturnOracleMlp(
            torch.zeros(INPUT_RETURN_COUNT),
            torch.ones(INPUT_RETURN_COUNT),
            dropout=dropout,
        )

    def test_architecture_matches_wide_hourglass_contract(self) -> None:
        model = self.model()
        self.assertEqual(HIDDEN_LAYER_COUNT, 8)
        self.assertEqual(
            HIDDEN_WIDTHS,
            (256,) * 8,
        )
        self.assertEqual(model.layers[0].in_features, INPUT_RETURN_COUNT)
        self.assertEqual(model.layers[0].out_features, 512)
        self.assertEqual(model.layers[-1].in_features, HIDDEN_WIDTHS[-2])
        self.assertEqual(model.layers[-1].out_features, 512)
        self.assertEqual(model.output.in_features, 256)
        self.assertEqual(model.output.out_features, OUTPUT_ACTION_COUNT)
        self.assertEqual(parameter_count(model), 2_594_831)
        self.assertEqual(
            [
                normalizer.normalized_shape
                for normalizer in model.value_centering_normalizers
            ],
            [(width,) for width in HIDDEN_WIDTHS],
        )
        self.assertEqual(
            [
                normalizer.normalized_shape
                for normalizer in model.gate_centering_normalizers
            ],
            [(width,) for width in HIDDEN_WIDTHS],
        )
        self.assertEqual(
            sum(
                isinstance(module, LearnableCenteringNorm)
                for module in model.modules()
            ),
            2 * HIDDEN_LAYER_COUNT,
        )
        self.assertTrue(all(
            id(value.weight) == id(gate.weight)
            for value, gate in zip(
                model.value_centering_normalizers,
                model.gate_centering_normalizers,
                strict=True,
            )
        ))
        self.assertTrue(all(
            normalizer.weight.requires_grad
            and normalizer.raw_scale.requires_grad
            and tuple(normalizer.weight.shape) == (
                normalizer.normalized_shape[0],
                normalizer.normalized_shape[0],
            )
            for normalizer in (
                *model.value_centering_normalizers,
                *model.gate_centering_normalizers,
            )
        ))
        self.assertTrue(all(
            torch.allclose(
                normalizer.scale().detach(),
                torch.tensor(BRANCH_NORMALIZATION_INITIAL_RADIUS),
            )
            for normalizer in (
                *model.value_centering_normalizers,
                *model.gate_centering_normalizers,
            )
        ))
        self.assertTrue(all(
            value_bias.requires_grad
            and gate_bias.requires_grad
            and tuple(value_bias.shape) == (width,)
            and tuple(gate_bias.shape) == (width,)
            for width, value_bias, gate_bias in zip(
                HIDDEN_WIDTHS,
                model.value_norm_biases,
                model.gate_norm_biases,
                strict=True,
            )
        ))
        self.assertTrue(all(
            torch.allclose(
                normalizer.weight.detach(),
                torch.eye(width) - torch.full(
                    (width, width),
                    1 / width,
                ),
            )
            for widths, normalizers in (
                (HIDDEN_WIDTHS, model.value_centering_normalizers),
                (HIDDEN_WIDTHS, model.gate_centering_normalizers),
            )
            for width, normalizer in zip(
                widths, normalizers, strict=True,
            )
        ))
        self.assertEqual(
            [transform.in_features for transform in model.value_transforms],
            list(HIDDEN_WIDTHS),
        )
        self.assertEqual(
            [transform.out_features for transform in model.gate_transforms],
            list(HIDDEN_WIDTHS),
        )
        self.assertTrue(all(
            transform.bias is None
            and torch.equal(
                transform.weight.detach(),
                torch.eye(transform.in_features),
            )
            for transform in (
                *model.value_transforms,
                *model.gate_transforms,
            )
        ))
        self.assertTrue(all(
            id(value.weight) != id(gate.weight)
            for value, gate in zip(
                model.value_transforms,
                model.gate_transforms,
                strict=True,
            )
        ))
        idempotence, symmetry = centering_matrix_constraint_components(
            normalizer.weight
            for normalizer in (
                *model.value_centering_normalizers,
                *model.gate_centering_normalizers,
            )
        )
        self.assertLess(float(idempotence.detach()), 1e-12)
        self.assertEqual(float(symmetry.detach()), 0.0)
        (
            logits,
            mean_penalty,
            variance_penalty,
            distribution_sum_penalty,
            distribution_negative_penalty,
        ) = (
            model.forward_with_regularizers(
                torch.zeros(3, INPUT_RETURN_COUNT)
            )
        )
        self.assertEqual(tuple(logits.shape), (3, OUTPUT_ACTION_COUNT))
        self.assertEqual(tuple(mean_penalty.shape), (3,))
        self.assertEqual(tuple(variance_penalty.shape), (3,))
        self.assertEqual(tuple(distribution_sum_penalty.shape), (3,))
        self.assertEqual(tuple(distribution_negative_penalty.shape), (3,))

    def test_fused_glu_projection_absorbs_the_input_transform(
        self,
    ) -> None:
        model = self.model()
        input_widths = (INPUT_RETURN_COUNT, *HIDDEN_WIDTHS[:-1])
        for input_width, layer in zip(
            input_widths,
            model.layers,
            strict=True,
        ):
            self.assertEqual(layer.in_features, input_width)
            self.assertGreater(
                float(layer.weight.detach().abs().sum()),
                0,
            )

    def test_centering_can_be_fixed_at_the_canonical_projector(self) -> None:
        model = ReturnOracleMlp(
            torch.zeros(INPUT_RETURN_COUNT),
            torch.ones(INPUT_RETURN_COUNT),
            dropout=0,
            learnable_centering=False,
        )
        self.assertTrue(all(
            not normalizer.weight.requires_grad
            for normalizer in model.value_centering_normalizers
        ))
        self.assertEqual(parameter_count(model), 2_070_543)
        muon_parameters, adamw_parameters = hybrid_optimizer_parameters(model)
        self.assertEqual(
            sum(parameter.numel() for parameter in muon_parameters),
            1_996_800,
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in adamw_parameters),
            73_743,
        )

    def test_hybrid_optimizer_routes_projection_and_a_matrices_to_muon(
        self,
    ) -> None:
        model = self.model()
        muon_parameters, adamw_parameters = hybrid_optimizer_parameters(
            model
        )
        self.assertEqual(
            tuple(id(parameter) for parameter in muon_parameters),
            (
                tuple(id(layer.weight) for layer in model.layers)
                + tuple(
                    id(transform.weight)
                    for transform in model.value_transforms
                )
                + tuple(
                    id(transform.weight)
                    for transform in model.gate_transforms
                )
            ),
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in muon_parameters),
            1_996_800,
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in adamw_parameters),
            598_031,
        )
        adamw_ids = {id(parameter) for parameter in adamw_parameters}
        self.assertIn(id(model.output.weight), adamw_ids)
        self.assertIn(id(model.output.bias), adamw_ids)
        self.assertTrue(all(
            id(layer.bias) in adamw_ids
            for layer in model.layers
        ))
        self.assertTrue(all(
            id(bias) in adamw_ids
            for bias in (
                *model.value_norm_biases,
                *model.gate_norm_biases,
            )
        ))
        self.assertTrue(all(
            id(normalizer.weight) in adamw_ids
            for normalizer in (
                *model.value_centering_normalizers,
                *model.gate_centering_normalizers,
            )
        ))

    def test_fused_glu_uses_independent_value_and_gate_halves(self) -> None:
        projected = torch.tensor([[
            2.0,
            -1.0,
            0.0,
            float(torch.log(torch.tensor(3.0))),
        ]])
        hidden, value, gate_logits = fused_glu(projected)
        self.assertTrue(torch.equal(value, torch.tensor([[2.0, -1.0]])))
        self.assertTrue(
            torch.allclose(
                gate_logits,
                torch.tensor([[0.0, float(torch.log(torch.tensor(3.0)))]]),
            )
        )
        self.assertTrue(
            torch.allclose(hidden, torch.tensor([[1.0, -0.75]]))
        )

    def test_soft_layer_norm_penalty_observes_pre_norm_branches(self) -> None:
        model = self.model()
        features = torch.randn(4, INPUT_RETURN_COUNT)
        (
            _logits,
            mean_penalty,
            variance_penalty,
            _distribution_sum_penalty,
            _distribution_negative_penalty,
        ) = model.forward_with_regularizers(features)
        self.assertGreater(float(mean_penalty.mean().detach()), 0)
        self.assertGreater(float(variance_penalty.mean().detach()), 0)

    def test_branches_use_independent_learnable_c_then_bias_and_full_a(
        self,
    ) -> None:
        projected = torch.tensor([
            [1.0, 2.0, 4.0, 8.0, -3.0, -1.0, 2.0, 7.0],
            [-2.0, 0.0, 3.0, 5.0, 1.0, 3.0, 6.0, 10.0],
        ])
        value_centering_normalizer = LearnableCenteringNorm(4)
        gate_centering_normalizer = LearnableCenteringNorm(4)
        with torch.no_grad():
            gate_centering_normalizer.weight.mul_(0.75)
        value_bias = torch.nn.Parameter(torch.tensor([0.1, 0.2, 0.3, 0.4]))
        gate_bias = torch.nn.Parameter(
            torch.tensor([-0.4, -0.3, -0.2, -0.1])
        )
        value_transform = torch.nn.Linear(4, 4, bias=False)
        gate_transform = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            value_transform.weight.copy_(2 * torch.eye(4))
            gate_transform.weight.copy_(0.5 * torch.eye(4))
        hidden, raw_value, raw_gate = fused_glu(
            projected,
            value_centering_normalizer,
            gate_centering_normalizer,
            value_bias,
            gate_bias,
            value_transform,
            gate_transform,
        )
        expected_value_centered = torch.nn.functional.linear(
            raw_value,
            value_centering_normalizer.weight,
        )
        expected_gate_centered = torch.nn.functional.linear(
            raw_gate,
            gate_centering_normalizer.weight,
        )
        value_u = (
            expected_value_centered.square().mean(dim=-1, keepdim=True)
            .sqrt()
            / value_centering_normalizer.scale()
        )
        gate_u = (
            expected_gate_centered.square().mean(dim=-1, keepdim=True)
            .sqrt()
            / gate_centering_normalizer.scale()
        )
        expected_value_norm = expected_value_centered / (
            value_centering_normalizer.scale()
            * (1.0 + value_u.square()).sqrt()
        )
        expected_gate_norm = expected_gate_centered / (
            gate_centering_normalizer.scale()
            * (1.0 + gate_u.square()).sqrt()
        )
        expected_value = value_transform(expected_value_norm) + value_bias
        expected_gate = gate_transform(expected_gate_norm) + gate_bias
        self.assertTrue(torch.equal(raw_value, projected[:, :4]))
        self.assertTrue(torch.equal(raw_gate, projected[:, 4:]))
        self.assertTrue(torch.allclose(
            hidden,
            expected_value * torch.sigmoid(expected_gate),
        ))
        self.assertTrue(value_centering_normalizer.weight.requires_grad)
        self.assertTrue(gate_centering_normalizer.weight.requires_grad)
        self.assertTrue(value_centering_normalizer.raw_scale.requires_grad)
        self.assertTrue(gate_centering_normalizer.raw_scale.requires_grad)
        self.assertIsNot(
            value_centering_normalizer.weight,
            gate_centering_normalizer.weight,
        )
        self.assertTrue(value_bias.requires_grad)
        self.assertTrue(gate_bias.requires_grad)
        self.assertEqual(parameter_count(value_centering_normalizer), 17)
        self.assertEqual(parameter_count(gate_centering_normalizer), 17)
        self.assertEqual(parameter_count(value_transform), 16)
        self.assertEqual(parameter_count(gate_transform), 16)

    def test_centering_constraints_measure_projector_and_symmetry(self) -> None:
        exact = LearnableCenteringNorm(4)
        exact_idempotence, exact_symmetry = (
            centering_matrix_constraint_components([exact.weight])
        )
        self.assertLess(float(exact_idempotence.detach()), 1e-12)
        self.assertEqual(float(exact_symmetry.detach()), 0.0)

        asymmetric_projector = torch.tensor([
            [1.0, 1.0],
            [0.0, 0.0],
        ], requires_grad=True)
        idempotence, symmetry = centering_matrix_constraint_components(
            [asymmetric_projector]
        )
        self.assertEqual(float(idempotence.detach()), 0.0)
        self.assertGreater(float(symmetry.detach()), 0.0)

        symmetric_nonprojector = (
            0.5 * torch.eye(2)
        ).requires_grad_()
        idempotence, symmetry = centering_matrix_constraint_components(
            [symmetric_nonprojector]
        )
        self.assertGreater(float(idempotence.detach()), 0.0)
        self.assertEqual(float(symmetry.detach()), 0.0)
        idempotence.backward()
        self.assertTrue(bool(
            torch.isfinite(symmetric_nonprojector.grad).all()
        ))

    def test_denominator_families_have_required_limits_and_scale_gradient(
        self,
    ) -> None:
        squared_u = torch.tensor([0.0, 1e-8, 1.0, 1e8])
        for family in ("sqrt", "tanh"):
            normalizer = LearnableCenteringNorm(
                4,
                denominator_family=family,
            )
            denominator = normalizer.inverse_denominator(
                squared_u
            ).reciprocal()
            self.assertEqual(float(denominator[0]), 1.0)
            self.assertTrue(bool((denominator > 0).all()))
            self.assertAlmostEqual(
                float(denominator[-1] / squared_u[-1].sqrt()),
                1.0,
                places=4,
            )

        normalizer = LearnableCenteringNorm(4)
        normalized = normalizer(torch.tensor([[
            -4.0,
            -1.0,
            2.0,
            7.0,
        ]]))
        normalized.square().sum().backward()
        self.assertIsNotNone(normalizer.raw_scale.grad)
        self.assertTrue(bool(torch.isfinite(normalizer.raw_scale.grad)))
        self.assertNotEqual(float(normalizer.raw_scale.grad), 0.0)

    def test_sqrt_family_initialization_is_exact_prior_hard_rms(self) -> None:
        normalizer = LearnableCenteringNorm(4)
        hidden = torch.tensor([
            [-4.0, -1.0, 2.0, 7.0],
            [1e-4, -2e-4, 3e-4, -2e-4],
        ])
        centered = torch.nn.functional.linear(
            hidden,
            normalizer.weight,
        )
        expected = centered / (
            centered.square().mean(dim=-1, keepdim=True)
            + BRANCH_NORMALIZATION_EPSILON
        ).sqrt()
        self.assertTrue(torch.allclose(
            normalizer(hidden),
            expected,
            rtol=1e-5,
            atol=1e-6,
        ))

    def test_dropout_rate_is_split_across_pass_and_layer_gates(self) -> None:
        gate_probability = dropout_gate_probability(0.5)
        self.assertAlmostEqual(gate_probability, 0.5 ** 0.5)
        self.assertAlmostEqual(gate_probability * gate_probability, 0.5)
        self.assertEqual(dropout_gate_probability(0), 0)
        self.assertEqual(dropout_gate_probability(1), 1)
        with self.assertRaises(ValueError):
            dropout_gate_probability(1.01)

    def test_zero_dropout_rate_disables_training_randomness(self) -> None:
        model = ReturnOracleMlp(
            torch.zeros(INPUT_RETURN_COUNT),
            torch.ones(INPUT_RETURN_COUNT),
            dropout=0.5,
            dropout_rate=0,
        )
        model.train()
        features = torch.randn(2, INPUT_RETURN_COUNT)
        self.assertTrue(torch.equal(model(features), model(features)))

    def test_first_projection_receives_training_standardized_returns(
        self,
    ) -> None:
        feature_mean = torch.linspace(
            -0.002,
            0.003,
            INPUT_RETURN_COUNT,
        )
        feature_std = torch.linspace(
            0.001,
            0.004,
            INPUT_RETURN_COUNT,
        )
        model = ReturnOracleMlp(
            feature_mean,
            feature_std,
            dropout=0,
        )
        features = torch.stack((
            torch.linspace(-0.003, 0.004, INPUT_RETURN_COUNT),
            torch.linspace(0.01, 0.04, INPUT_RETURN_COUNT).square(),
        ))
        captured: list[torch.Tensor] = []
        handle = model.layers[0].register_forward_pre_hook(
            lambda _module, arguments: captured.append(arguments[0].detach())
        )
        try:
            model(features)
        finally:
            handle.remove()
        self.assertEqual(len(captured), 1)
        expected = (features - feature_mean) / feature_std
        self.assertTrue(
            torch.allclose(captured[0], expected, atol=1e-6, rtol=1e-6)
        )

    def test_branch_norms_and_a_transforms_follow_projection_order(self) -> None:
        model = self.model()
        projected: list[torch.Tensor] = []
        value_norm_inputs: list[torch.Tensor] = []
        gate_norm_inputs: list[torch.Tensor] = []
        normalized_values: list[torch.Tensor] = []
        normalized_gates: list[torch.Tensor] = []
        value_transform_inputs: list[torch.Tensor] = []
        gate_transform_inputs: list[torch.Tensor] = []
        layer_handle = model.layers[0].register_forward_hook(
            lambda _module, _arguments, output: projected.append(
                output.detach()
            )
        )
        value_norm_input_handle = (
            model.value_centering_normalizers[0].register_forward_pre_hook(
                lambda _module, arguments: value_norm_inputs.append(
                    arguments[0].detach()
                )
            )
        )
        gate_norm_input_handle = (
            model.gate_centering_normalizers[0].register_forward_pre_hook(
                lambda _module, arguments: gate_norm_inputs.append(
                    arguments[0].detach()
                )
            )
        )
        value_norm_output_handle = (
            model.value_centering_normalizers[0].register_forward_hook(
                lambda _module, _arguments, output: (
                    normalized_values.append(output.detach())
                )
            )
        )
        gate_norm_output_handle = (
            model.gate_centering_normalizers[0].register_forward_hook(
                lambda _module, _arguments, output: (
                    normalized_gates.append(output.detach())
                )
            )
        )
        value_transform_handle = (
            model.value_transforms[0].register_forward_pre_hook(
                lambda _module, arguments: value_transform_inputs.append(
                    arguments[0].detach()
                )
            )
        )
        gate_transform_handle = (
            model.gate_transforms[0].register_forward_pre_hook(
                lambda _module, arguments: gate_transform_inputs.append(
                    arguments[0].detach()
                )
            )
        )
        try:
            model(torch.randn(3, INPUT_RETURN_COUNT))
        finally:
            layer_handle.remove()
            value_norm_input_handle.remove()
            gate_norm_input_handle.remove()
            value_norm_output_handle.remove()
            gate_norm_output_handle.remove()
            value_transform_handle.remove()
            gate_transform_handle.remove()
        self.assertEqual(len(projected), 1)
        self.assertEqual(len(value_norm_inputs), 1)
        self.assertEqual(len(gate_norm_inputs), 1)
        self.assertEqual(len(normalized_values), 1)
        self.assertEqual(len(normalized_gates), 1)
        expected_value, expected_gate = projected[0].chunk(2, dim=-1)
        self.assertTrue(
            torch.equal(value_norm_inputs[0], expected_value)
        )
        self.assertTrue(
            torch.equal(gate_norm_inputs[0], expected_gate)
        )
        self.assertTrue(torch.equal(
            value_transform_inputs[0],
            normalized_values[0],
        ))
        self.assertTrue(torch.equal(
            gate_transform_inputs[0],
            normalized_gates[0],
        ))

    def test_export_folds_input_normalization_into_first_layer(
        self,
    ) -> None:
        torch.manual_seed(31)
        feature_mean = torch.linspace(
            -0.002,
            0.003,
            INPUT_RETURN_COUNT,
        )
        feature_std = torch.linspace(
            0.0005,
            0.0015,
            INPUT_RETURN_COUNT,
        )
        model = ReturnOracleMlp(
            feature_mean,
            feature_std,
            dropout=0.5,
            dropout_rate=1,
        ).eval()
        raw_features = torch.randn(7, INPUT_RETURN_COUNT) * 0.001
        original_weight = model.layers[0].weight.detach().clone()
        original_bias = model.layers[0].bias.detach().clone()
        expected_logits = model(raw_features)

        exported = fold_input_normalization_for_export(model)
        actual_logits = exported(raw_features)

        expected_weight = original_weight / feature_std.unsqueeze(0)
        expected_bias = original_bias - expected_weight @ feature_mean
        self.assertTrue(torch.allclose(
            exported.layers[0].weight,
            expected_weight,
        ))
        self.assertTrue(torch.allclose(
            exported.layers[0].bias,
            expected_bias,
        ))
        self.assertTrue(torch.allclose(
            actual_logits,
            expected_logits,
            atol=2e-5,
            rtol=2e-5,
        ))
        self.assertTrue(torch.equal(
            model.layers[0].weight,
            original_weight,
        ))
        self.assertTrue(torch.equal(model.layers[0].bias, original_bias))
        self.assertNotIn("feature_mean", exported.state_dict())
        self.assertNotIn("feature_std", exported.state_dict())

    def test_simple_returns_use_adjacent_close_boundaries(self) -> None:
        closes = torch.ones(2, INPUT_RETURN_COUNT + 1)
        closes[0] = torch.arange(1, INPUT_RETURN_COUNT + 2)
        features = simple_return_features(closes)
        self.assertEqual(tuple(features.shape), (2, INPUT_RETURN_COUNT))
        self.assertAlmostEqual(float(features[0, 0]), 1.0)
        self.assertAlmostEqual(
            float(features[0, -1]),
            (INPUT_RETURN_COUNT + 1) / INPUT_RETURN_COUNT - 1,
            places=6,
        )
        self.assertTrue(bool((features[1] == 0).all()))

    def test_batch_invariant_regularizers_restore_evaluation_loss(
        self,
    ) -> None:
        metrics = {
            "loss": 3.5,
            "softWeightBound": 0.0,
            "centeringIdempotence": 0.0,
            "centeringSymmetry": 0.0,
            "centeringConstraint": 0.0,
        }
        regularizers = {
            "softWeightBound": 0.25,
            "centeringIdempotence": 0.1,
            "centeringSymmetry": 0.2,
            "centeringConstraint": 0.3,
            "lossAddition": 0.3025,
        }
        result = add_batch_invariant_regularizers(metrics, regularizers)
        self.assertAlmostEqual(result["loss"], 3.8025)
        self.assertEqual(result["softWeightBound"], 0.25)
        self.assertEqual(result["centeringIdempotence"], 0.1)
        self.assertEqual(result["centeringSymmetry"], 0.2)
        self.assertEqual(result["centeringConstraint"], 0.3)
        self.assertEqual(metrics["loss"], 3.5)

    def test_gradient_accumulation_groups_keep_tail(self) -> None:
        batches = list(group_batches(iter(range(5)), 2))
        self.assertEqual(batches, [(0, 1), (2, 3), (4,)])
        with self.assertRaises(ValueError):
            list(group_batches(iter(()), 0))

    def test_completed_minute_rows_match_oracle_alignment(self) -> None:
        rows = np.asarray([0, 58, 59, 60, 118, 119, 86_399])
        mapped = completed_minute_row_indices(rows)
        np.testing.assert_array_equal(
            mapped,
            np.asarray([0, 0, 1, 1, 1, 2, 1_440]),
        )

    def test_completed_minute_features_use_matching_close_path(self) -> None:
        previous = np.arange(1, 86_401, dtype=np.float64)
        current = np.arange(86_401, 172_801, dtype=np.float64)
        features = completed_minute_simple_return_rows(previous, current)
        self.assertEqual(tuple(features.shape), (1_441, INPUT_RETURN_COUNT))
        self.assertAlmostEqual(
            float(features[0, 0]),
            (82_860 / 82_800) - 1,
        )
        self.assertAlmostEqual(
            float(features[0, -1]),
            (86_400 / 86_340) - 1,
        )
        self.assertAlmostEqual(
            float(features[1, -1]),
            (86_460 / 86_400) - 1,
        )

    def test_completed_minute_batch_compaction_preserves_multiplicity(
        self,
    ) -> None:
        feature_rows, target_rows, multiplicities = (
            compact_completed_minute_batch_rows(50, 3_650, 80)
        )
        np.testing.assert_array_equal(feature_rows, np.asarray([0, 1, 2]))
        np.testing.assert_array_equal(target_rows, np.asarray([60, 61, 62]))
        np.testing.assert_array_equal(
            multiplicities,
            np.asarray([9, 60, 11], dtype=np.float32),
        )
        self.assertEqual(float(multiplicities.sum()), 80)

    def test_cross_entropy_reaches_oracle_entropy_for_exact_policy(self) -> None:
        generator = torch.Generator().manual_seed(19)
        logits = torch.randn(7, OUTPUT_ACTION_COUNT, generator=generator)
        target = torch.softmax(logits, dim=-1)
        metrics = oracle_policy_cross_entropy(logits, target)
        self.assertAlmostEqual(
            float(metrics["loss"]),
            float(metrics["targetEntropy"]),
            places=6,
        )
        self.assertLess(abs(float(metrics["baseKlDivergence"])), 1e-6)
        self.assertLess(abs(float(metrics["reverseKlDivergence"])), 1e-6)
        self.assertLess(abs(float(metrics["entropyGap"])), 1e-6)
        self.assertLess(abs(float(metrics["entropySharpness"])), 1e-10)
        self.assertLess(abs(float(metrics["probabilityMse"])), 1e-10)

    def test_cross_entropy_penalizes_wrong_policy(self) -> None:
        target = torch.zeros(2, OUTPUT_ACTION_COUNT)
        target[:, 10] = 1
        correct = torch.full_like(target, -30)
        correct[:, 10] = 30
        wrong = torch.full_like(target, -30)
        wrong[:, 200] = 30
        correct_loss = oracle_policy_cross_entropy(correct, target)["loss"]
        wrong_loss = oracle_policy_cross_entropy(wrong, target)["loss"]
        self.assertLess(float(correct_loss), 1e-6)
        self.assertGreater(float(wrong_loss), 50)

    def test_cross_entropy_has_direct_soft_target_logit_gradient(self) -> None:
        generator = torch.Generator().manual_seed(29)
        logits = torch.randn(
            3,
            OUTPUT_ACTION_COUNT,
            generator=generator,
            requires_grad=True,
        )
        target = torch.softmax(
            torch.randn(3, OUTPUT_ACTION_COUNT, generator=generator),
            dim=-1,
        )
        loss = oracle_policy_cross_entropy(logits, target)["loss"]
        loss.backward()
        expected = (torch.softmax(logits.detach(), dim=-1) - target) / 3
        self.assertTrue(
            torch.allclose(logits.grad, expected, atol=1e-7, rtol=1e-5)
        )

    def test_reverse_kl_uses_prediction_oracle_mixture(self) -> None:
        prediction_mixture = 1e-2
        logits = torch.linspace(
            -2,
            2,
            OUTPUT_ACTION_COUNT,
        ).unsqueeze(0)
        target = torch.zeros_like(logits)
        target[:, 10] = 1
        metrics = oracle_policy_cross_entropy(
            logits,
            target,
            reverse_kl_prediction_mixture=prediction_mixture,
        )
        predicted_log = torch.log_softmax(logits, dim=-1)
        predicted = predicted_log.exp()
        reference = (
            (1 - prediction_mixture) * target
            + prediction_mixture * predicted
        )
        expected = (
            predicted * (predicted_log - reference.log())
        ).sum()
        self.assertTrue(
            torch.allclose(
                metrics["reverseKlDivergence"],
                expected,
                atol=1e-7,
                rtol=1e-6,
            )
        )
        self.assertTrue(torch.isfinite(metrics["reverseKlDivergence"]))

    def test_entropy_sharpness_only_penalizes_excess_entropy(self) -> None:
        uniform_logits = torch.zeros(1, OUTPUT_ACTION_COUNT)
        sharp_target = torch.zeros_like(uniform_logits)
        sharp_target[:, 10] = 1
        diffuse = oracle_policy_cross_entropy(
            uniform_logits,
            sharp_target,
        )
        expected_gap = math.log(OUTPUT_ACTION_COUNT)
        self.assertAlmostEqual(
            float(diffuse["entropyGap"]),
            expected_gap,
            places=5,
        )
        self.assertAlmostEqual(
            float(diffuse["entropySharpness"]),
            expected_gap ** 2,
            places=4,
        )

        sharp_logits = torch.full_like(uniform_logits, -30)
        sharp_logits[:, 10] = 30
        diffuse_target = torch.full_like(
            sharp_logits,
            1 / OUTPUT_ACTION_COUNT,
        )
        sharp = oracle_policy_cross_entropy(
            sharp_logits,
            diffuse_target,
        )
        self.assertLess(float(sharp["entropyGap"]), 0)
        self.assertEqual(float(sharp["entropySharpness"]), 0)

    def test_objective_applies_corrected_independent_output_gates(self) -> None:
        logits = torch.linspace(
            -1,
            1,
            OUTPUT_ACTION_COUNT,
        ).unsqueeze(0)
        target = torch.zeros_like(logits)
        target[:, 100] = 1
        zero_per_example = torch.zeros(1)
        def objective(
            reverse_gate: float,
            entropy_gate: float,
        ) -> dict[str, torch.Tensor]:
            return oracle_policy_objective(
                logits,
                target,
                zero_per_example,
                zero_per_example,
                zero_per_example,
                zero_per_example,
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0),
                reverse_kl_weight=0.02,
                reverse_kl_prediction_mixture=1e-2,
                entropy_sharpness_weight=0.02,
                reverse_kl_gate=torch.tensor(reverse_gate),
                entropy_sharpness_gate=torch.tensor(entropy_gate),
                reverse_kl_scale=10.0,
                entropy_sharpness_scale=10.0,
                soft_layer_norm_weight=0,
                soft_weight_bound_weight=0,
                distribution_sum_weight=0,
                distribution_negative_weight=0,
                centering_idempotence_weight=0,
                centering_symmetry_weight=0,
            )

        reverse_metrics = objective(1.0, 0.0)
        reverse_expected = (
            reverse_metrics["crossEntropy"]
            + 0.2 * reverse_metrics["reverseKlDivergence"]
        )
        self.assertTrue(
            torch.allclose(
                reverse_metrics["loss"],
                reverse_expected,
                atol=1e-7,
                rtol=1e-6,
            )
        )
        self.assertEqual(float(reverse_metrics["reverseKlGate"]), 1.0)
        self.assertEqual(
            float(reverse_metrics["entropySharpnessGate"]),
            0.0,
        )

        entropy_metrics = objective(0.0, 1.0)
        entropy_expected = (
            entropy_metrics["crossEntropy"]
            + 0.2 * entropy_metrics["entropySharpness"]
        )
        self.assertTrue(
            torch.allclose(
                entropy_metrics["loss"],
                entropy_expected,
                atol=1e-7,
                rtol=1e-6,
            )
        )

        expected_metrics = objective(0.1, 0.1)
        expected_validation_loss = (
            expected_metrics["crossEntropy"]
            + 0.02 * expected_metrics["reverseKlDivergence"]
            + 0.02 * expected_metrics["entropySharpness"]
        )
        self.assertTrue(
            torch.allclose(
                expected_metrics["loss"],
                expected_validation_loss,
                atol=1e-7,
                rtol=1e-6,
            )
        )

    def test_multiplicity_weighting_matches_expanded_rows_exactly(self) -> None:
        generator = torch.Generator().manual_seed(31)
        logits = torch.randn(
            3,
            OUTPUT_ACTION_COUNT,
            generator=generator,
            requires_grad=True,
        )
        target = torch.softmax(
            torch.randn(3, OUTPUT_ACTION_COUNT, generator=generator),
            dim=-1,
        )
        multiplicities = torch.tensor([9, 60, 11], dtype=torch.float32)
        compact = oracle_policy_cross_entropy(
            logits,
            target,
            multiplicities,
        )
        compact["loss"].backward()
        compact_gradient = logits.grad.detach().clone()

        expanded_logits = logits.detach().clone().requires_grad_(True)
        counts = multiplicities.to(dtype=torch.int64)
        expanded = oracle_policy_cross_entropy(
            expanded_logits.repeat_interleave(counts, dim=0),
            target.repeat_interleave(counts, dim=0),
        )
        expanded["loss"].backward()
        self.assertTrue(
            torch.allclose(
                compact["loss"],
                expanded["loss"],
                atol=1e-7,
                rtol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                compact["reverseKlDivergence"],
                expanded["reverseKlDivergence"],
                atol=1e-7,
                rtol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                compact["entropySharpness"],
                expanded["entropySharpness"],
                atol=1e-7,
                rtol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                compact_gradient,
                expanded_logits.grad,
                atol=1e-7,
                rtol=1e-5,
            )
        )

    def test_soft_layer_norm_components_use_per_example_statistics(
        self,
    ) -> None:
        hidden = torch.tensor([
            [-1.0, 1.0],
            [2.0, 2.0],
        ])
        mean_penalty, variance_penalty = soft_layer_norm_components(hidden)
        self.assertTrue(torch.equal(mean_penalty, torch.tensor([0.0, 4.0])))
        self.assertTrue(
            torch.equal(variance_penalty, torch.tensor([0.0, 1.0]))
        )

    def test_distribution_layer_components_use_sum_and_negative_mass(
        self,
    ) -> None:
        hidden = torch.tensor([
            [0.25, 0.75],
            [-1.0, 0.5],
        ])
        sum_penalty, negative_penalty = distribution_layer_components(hidden)
        self.assertTrue(
            torch.equal(sum_penalty, torch.tensor([0.0, 0.5625]))
        )
        self.assertTrue(
            torch.equal(negative_penalty, torch.tensor([0.0, 0.5]))
        )

    def test_soft_layer_norm_is_weighted_and_added_to_cross_entropy(
        self,
    ) -> None:
        logits = torch.zeros(2, OUTPUT_ACTION_COUNT)
        target = torch.full_like(logits, 1 / OUTPUT_ACTION_COUNT)
        multiplicities = torch.tensor([1.0, 3.0])
        metrics = oracle_policy_objective(
            logits,
            target,
            mean_penalties=torch.tensor([0.0, 4.0]),
            variance_penalties=torch.tensor([1.0, 3.0]),
            distribution_sum_penalties=torch.tensor([4.0, 8.0]),
            distribution_negative_penalties=torch.tensor([1.0, 3.0]),
            soft_weight_bound_penalty=torch.tensor(2.0),
            centering_idempotence_penalty=torch.tensor(0.25),
            centering_symmetry_penalty=torch.tensor(0.5),
            example_weights=multiplicities,
            cross_entropy_weight=1.0,
            soft_layer_norm_weight=0.01,
            soft_weight_bound_weight=0.5,
            variance_weight=2.0,
            distribution_sum_weight=0.1,
            distribution_negative_weight=0.2,
            centering_idempotence_weight=2.0,
            centering_symmetry_weight=3.0,
        )
        self.assertAlmostEqual(
            float(metrics["softLayerNormMeanPenalty"]),
            3.0,
        )
        self.assertAlmostEqual(
            float(metrics["softLayerNormVariancePenalty"]),
            2.5,
        )
        self.assertAlmostEqual(float(metrics["softLayerNorm"]), 8.0)
        self.assertAlmostEqual(
            float(metrics["distributionLayerSumPenalty"]),
            7.0,
        )
        self.assertAlmostEqual(
            float(metrics["distributionLayerNegativePenalty"]),
            2.5,
        )
        self.assertAlmostEqual(float(metrics["distributionLayer"]), 1.2)
        self.assertAlmostEqual(
            float(metrics["centeringIdempotence"]),
            0.25,
        )
        self.assertAlmostEqual(float(metrics["centeringSymmetry"]), 0.5)
        self.assertAlmostEqual(
            float(metrics["centeringConstraint"]),
            2.0,
        )
        self.assertAlmostEqual(
            float(metrics["loss"]),
            float(metrics["crossEntropy"]) + 0.08 + 1.0 + 1.2 + 2.0,
            places=6,
        )

    def test_soft_weight_bound_only_materially_penalizes_excess(
        self,
    ) -> None:
        inside = torch.tensor([[-0.5, 0.25]], requires_grad=True)
        outside = torch.tensor([[-2.0, 1.5]], requires_grad=True)
        inside_penalty = soft_weight_bound_penalty(
            [inside],
            desired_magnitude=1.0,
            sharpness=10.0,
            absolute_epsilon=1e-8,
        )
        outside_penalty = soft_weight_bound_penalty(
            [outside],
            desired_magnitude=1.0,
            sharpness=10.0,
            absolute_epsilon=1e-8,
        )
        relaxed_penalty = soft_weight_bound_penalty(
            [outside],
            desired_magnitude=2.0,
            sharpness=10.0,
            absolute_epsilon=1e-8,
        )
        self.assertLess(float(inside_penalty.detach()), 1e-6)
        self.assertGreater(float(outside_penalty.detach()), 0.5)
        self.assertLess(
            float(relaxed_penalty.detach()),
            float(outside_penalty.detach()),
        )
        outside_penalty.backward()
        self.assertLess(float(outside.grad[0, 0]), 0)
        self.assertGreater(float(outside.grad[0, 1]), 0)

    def test_segments_are_grouped_by_target_without_reordering(self) -> None:
        def segment(component: str, day: int) -> Segment:
            shard = SourceShard(
                split="train",
                date=f"2026-01-{day:02d}",
                count=10,
                feature_file=Path(f"features-{day}"),
                feature_row_offset=0,
                feature_row_stride=1,
                target_file=Path(component),
                target_row_offset=0,
                target_row_stride=1,
                prediction_time_start=day * 100_000,
                oracle_target_time_start=day * 100_000 - 3_600_000,
            )
            return Segment(shard, 0, 10)

        first = segment("a.zst", 1)
        second = segment("b.zst", 2)
        third = segment("a.zst", 3)
        groups = group_segments_by_target([first, second, third])
        self.assertEqual(groups, [[first, third], [second]])

    def test_component_copy_preserves_dtype_and_values(self) -> None:
        source = np.arange(12, dtype=np.float16).reshape(3, 4)
        copied = copy_component_to_tensor(source, torch.float16, False)
        source.fill(-1)
        self.assertEqual(copied.dtype, torch.float16)
        self.assertTrue(torch.equal(copied, torch.arange(12).reshape(3, 4)))

    def test_atomic_replace_retries_transient_permission_error(self) -> None:
        temporary = Path("status.json.tmp")
        destination = Path("status.json")
        with patch(
            "train_return_oracle_ce.os.replace",
            side_effect=(PermissionError("locked"), None),
        ) as replace, patch("train_return_oracle_ce.time.sleep") as sleep:
            replace_file_with_retry(temporary, destination, attempts=2)
        self.assertEqual(replace.call_count, 2)
        sleep.assert_called_once()


if __name__ == "__main__":
    unittest.main()
