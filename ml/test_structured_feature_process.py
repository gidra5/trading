from __future__ import annotations

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from materialize_sparse_timeline_episode_dataset import (
    episode_lengths,
    sparse_episode_origins,
)
from normalized_glu_next_return import optimizer_parameter_groups
from structured_feature_process import (
    CausalKnotBasisGramLossLayer8,
    CausalSelfAttentionLayer8,
    DualStateGatedExchangeCell,
    LowRankTensorPathGluBlock,
    StructuredSharedIoFeatureProcess,
)
from structured_union530_base import _safe_divide
from train_structured_feature_process import (
    COMPARABLE_EVALUATION_SCOPE,
    FeatureMetricAccumulator,
    FeatureSequenceMetricAccumulator,
    StructuredFeatureSequenceDataset,
    production59_balanced_objective,
    weighted_standardized_mse,
)


class StructuredFeatureProcessTest(unittest.TestCase):
    def make_model(
        self,
        input_steps: int,
        output_steps: int,
        *,
        linear_rank: int | None = None,
        layer8_attention: dict[str, int | str] | None = None,
        layer8_function_approximator: dict[str, object] | None = None,
        recurrent_memory: dict[str, object] | None = None,
    ):
        return StructuredSharedIoFeatureProcess(
            torch.zeros(5),
            torch.ones(5),
            torch.zeros(5),
            torch.ones(5),
            input_steps=input_steps,
            output_steps=output_steps,
            feature_width=12,
            market_width=7,
            prefix_width=6,
            feature_distribution_width=8,
            extended_prefix_width=10,
            next_feature_distribution_width=9,
            initial_radius=0.0031622776601683794,
            minimum_radius=1e-4,
            learnable_centering=False,
            linear_rank=linear_rank,
            layer8_attention=layer8_attention,
            layer8_function_approximator=layer8_function_approximator,
            recurrent_memory=recurrent_memory,
        )

    def test_layer8_causal_attention_uses_prefix_nf_states(self) -> None:
        attention = {
            "type": "causal-self-attention-v1",
            "queryWidth": 12,
            "keyWidth": 12,
            "valueWidth": 12,
            "outputWidth": 12,
        }
        torch.manual_seed(23)
        one_step = self.make_model(2, 1, layer8_attention=attention)
        two_step_model = self.make_model(2, 2, layer8_attention=attention)
        two_step_model.load_state_dict(one_step.state_dict())
        self.assertIsInstance(
            two_step_model.layer8, CausalSelfAttentionLayer8
        )

        inputs = torch.randn(7, 2, 5)
        first = one_step.trace(inputs)
        second = two_step_model.trace(inputs)
        torch.testing.assert_close(
            first.expected_feature_embeddings[0],
            second.expected_feature_embeddings[0],
        )
        self.assertEqual(second.outputs.shape, (7, 2, 5))

        loss = second.outputs.square().mean()
        loss.backward()
        for parameter in two_step_model.layer8.muon_parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_every_repeated_gnglu_layer_carries_private_hidden_memory(
        self,
    ) -> None:
        attention = {
            "type": "causal-self-attention-v1",
            "queryWidth": 12,
            "keyWidth": 12,
            "valueWidth": 12,
            "outputWidth": 12,
        }
        memory = {
            "type": "dual-state-gated-exchange-v1",
            "hiddenWidth": 4,
            "layers": [1, 2, 3, 4, 5, 6, 7, 9, 10, 11],
            "activation": "sigmoid",
            "mix": "a*t+b*(t-1)",
        }
        torch.manual_seed(31)
        one_output = self.make_model(
            4, 1,
            layer8_attention=attention,
            recurrent_memory=memory,
        )
        many_outputs = self.make_model(
            4, 5,
            layer8_attention=attention,
            recurrent_memory=memory,
        )
        many_outputs.load_state_dict(one_output.state_dict())
        for number in (1, 2, 3, 4, 5, 6, 7, 9, 10, 11):
            layer = getattr(many_outputs, f"layer{number}")
            self.assertIsInstance(layer, DualStateGatedExchangeCell)
            self.assertEqual(layer.hidden_width, 4)
        self.assertIsInstance(
            many_outputs.layer8, CausalSelfAttentionLayer8
        )

        inputs = torch.randn(6, 4, 5)
        first = one_output(inputs)
        repeated = many_outputs(inputs)
        torch.testing.assert_close(first[:, 0], repeated[:, 0])
        torch.testing.assert_close(repeated, many_outputs(inputs))
        self.assertEqual(repeated.shape, (6, 5, 5))
        repeated.square().mean().backward()
        for number in (1, 2, 3, 4, 5, 6, 7, 9, 10, 11):
            layer = getattr(many_outputs, f"layer{number}")
            for parameter in layer.muon_parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_dual_state_cell_uses_signed_exchange_mix(self) -> None:
        first = torch.tensor([[2.0, -1.0]])
        second = torch.tensor([[4.0, 3.0]])
        gate = torch.tensor([[0.25, 0.75]])
        actual = DualStateGatedExchangeCell.signed_mix(
            first, second, gate
        )
        torch.testing.assert_close(
            actual,
            first * gate + second * (gate - 1.0),
        )

    def test_rollout_safe_divide_has_finite_zero_denominator_gradient(
        self,
    ) -> None:
        numerator = torch.tensor([2.0, 3.0], requires_grad=True)
        denominator = torch.tensor([0.0, 4.0], requires_grad=True)
        result = _safe_divide(numerator, denominator)
        torch.testing.assert_close(result, torch.tensor([0.0, 0.75]))
        result.sum().backward()
        assert numerator.grad is not None and denominator.grad is not None
        self.assertTrue(bool(torch.isfinite(numerator.grad).all()))
        self.assertTrue(bool(torch.isfinite(denominator.grad).all()))

    def test_balanced_production59_objective_weights_and_masks_boundaries(
        self,
    ) -> None:
        derived_prediction = torch.ones(2, 2, 59)
        derived_prediction[:, :, 0] = 2.0
        derived_target = torch.zeros_like(derived_prediction)
        base_prediction = torch.full((2, 2, 17), 3.0)
        base_target = torch.zeros_like(base_prediction)
        # Only row 0, step 2 completes a minute. Masked large errors elsewhere
        # must not enter direct minute-coordinate supervision.
        base_prediction[:, :, 14:] = 100.0
        base_prediction[0, 1, 14:] = 5.0
        base_prediction.requires_grad_()
        config = {
            "weights": {
                "return": 0.5,
                "otherDerivedFeatures": 0.25,
                "primitiveCoordinates": 0.25,
            }
        }
        components = production59_balanced_objective(
            derived_prediction,
            derived_target,
            base_prediction,
            base_target,
            torch.ones(2),
            {"secondIndex": torch.tensor([57, 10])},
            config,
        )
        expected_primitive = (2 * 2 * 14 * 9 + 3 * 25) / (2 * 2 * 14 + 3)
        self.assertAlmostEqual(float(components["return"]), 4.0)
        self.assertAlmostEqual(float(components["otherDerivedFeatures"]), 1.0)
        self.assertAlmostEqual(
            float(components["primitiveCoordinates"].detach()),
            expected_primitive,
            places=5,
        )
        self.assertAlmostEqual(
            float(components["objective"].detach()),
            0.5 * 4.0 + 0.25 + 0.25 * expected_primitive,
            places=5,
        )
        components["objective"].backward()
        self.assertIsNotNone(base_prediction.grad)
        assert base_prediction.grad is not None
        self.assertGreater(
            float(base_prediction.grad[0, 1, 14:].abs().sum()), 0.0
        )
        self.assertEqual(
            float(base_prediction.grad[0, 0, 14:].abs().sum()), 0.0
        )
        self.assertEqual(
            float(base_prediction.grad[1, :, 14:].abs().sum()), 0.0
        )

    def test_layer8_knot_basis_is_causal_and_penalizes_gram(self) -> None:
        function = {
            "type": "causal-dog-knot-basis-gram-loss-v1",
            "pointWidth": 12,
            "knotWidth": 12,
            "valueWidth": 12,
            "outputWidth": 12,
            "normalizationEpsilon": 1e-6,
            "kernel": {
                "type": "normalized-distance-dog-v1",
                "bandwidth": math.sqrt(2.0),
            },
            "orthogonalizationLoss": {
                "type": "gram-identity-mean-square-v1",
                "gramEstimator": "current-batch-causal-prefix-v1",
                "weight": 1.0,
            },
        }
        torch.manual_seed(29)
        one_step = self.make_model(
            2, 1, layer8_function_approximator=function
        )
        two_steps = self.make_model(
            2, 2, layer8_function_approximator=function
        )
        two_steps.load_state_dict(one_step.state_dict())
        self.assertIsInstance(
            two_steps.layer8, CausalKnotBasisGramLossLayer8
        )

        inputs = torch.randn(512, 2, 5)
        first = one_step.trace(inputs)
        second = two_steps.trace(inputs)
        torch.testing.assert_close(
            first.expected_feature_embeddings[0],
            second.expected_feature_embeddings[0],
        )
        sequence = torch.stack(
            second.next_feature_distribution_states, dim=1
        )
        all_weights, gram_loss = \
            two_steps.layer8.basis_weights_and_gram_identity_loss(sequence)
        weights = all_weights[-1]
        points = torch.nn.functional.normalize(
            two_steps.layer8.query(sequence), dim=-1, eps=1e-6
        )
        knots = torch.nn.functional.normalize(
            two_steps.layer8.key(sequence), dim=-1, eps=1e-6
        )
        torch.testing.assert_close(
            weights,
            two_steps.layer8.kernel(points[:, -1], knots),
        )
        gram = weights.transpose(0, 1) @ weights / weights.shape[0]
        expected_loss = (
            gram - torch.eye(gram.shape[0])
        ).square().mean()
        torch.testing.assert_close(gram_loss, expected_loss)
        prediction, model_gram_loss = \
            two_steps.forward_with_auxiliary_loss(inputs)
        self.assertTrue(bool(torch.isfinite(prediction).all()))
        self.assertTrue(bool(torch.isfinite(model_gram_loss)))
        (prediction.square().mean() + model_gram_loss).backward()
        for parameter in two_steps.layer8.muon_parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_knot_basis_kernel_has_local_negative_lobe(self) -> None:
        layer = CausalKnotBasisGramLossLayer8(
            2, 2, 2, 2, 2,
            kernel_bandwidth=math.sqrt(2.0),
            normalization_epsilon=1e-6,
        )
        point = torch.tensor([[1.0, 0.0]])
        knots = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
        basis = layer.kernel(point, knots)
        self.assertAlmostEqual(float(basis[0, 0]), 1.0, places=6)
        self.assertLess(float(basis[0, 1]), 0.0)

    def test_low_rank_gnglu_uses_requested_rank_up_to_matrix_ceiling(self) -> None:
        model = self.make_model(2, 2, linear_rank=4)
        self.assertTrue(all(
            isinstance(getattr(model, f"layer{index}"), LowRankTensorPathGluBlock)
            for index in range(1, 12)
        ))
        self.assertEqual(model.layer1.projection.requested_rank, 4)
        self.assertEqual(model.layer1.projection.rank, 4)
        self.assertEqual(model.layer2.output.rank, 4)

        inputs = torch.randn(7, 2, 5)
        targets = torch.randn(7, 2, 5)
        prediction = model(inputs)
        loss = weighted_standardized_mse(
            prediction,
            model.standardized_targets(targets),
            torch.ones(7),
        )
        loss.backward()
        self.assertTrue(bool(torch.isfinite(prediction).all()))
        for parameter in model.muon_parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_low_rank_caps_narrow_io_without_redundant_factors(self) -> None:
        model = self.make_model(1, 1, linear_rank=256)
        self.assertEqual(model.layer1.projection.requested_rank, 256)
        self.assertEqual(model.layer1.projection.rank, 5)
        self.assertEqual(model.layer2.output.requested_rank, 256)
        self.assertEqual(model.layer2.output.rank, 5)

    def test_reuses_numbered_layers_across_all_io_steps(self) -> None:
        model = self.make_model(3, 4)
        calls = {index: 0 for index in range(1, 12)}
        handles = []
        for index in calls:
            def hook(_module, _inputs, _output, *, index=index):
                calls[index] += 1
            handles.append(getattr(model, f"layer{index}").register_forward_hook(hook))
        try:
            trace = model.trace(torch.randn(2, 3, 5))
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(trace.outputs.shape, (2, 4, 5))
        self.assertEqual(len(trace.feature_embeddings), 3)
        self.assertEqual(len(trace.market_states), 6)
        self.assertEqual(len(trace.next_feature_distribution_states), 4)
        self.assertEqual(calls, {
            1: 3,
            2: 4,
            3: 3,
            4: 6,
            5: 2,
            6: 2,
            7: 4,
            8: 4,
            9: 3,
            10: 3,
            11: 3,
        })

    def test_one_input_one_output_active_path_has_exact_gradient(self) -> None:
        torch.manual_seed(11)
        model = self.make_model(1, 1)
        inputs = torch.randn(7, 1, 5)
        targets = torch.randn(7, 1, 5)
        prediction = model(inputs)
        loss = weighted_standardized_mse(
            prediction,
            model.standardized_targets(targets),
            torch.ones(7),
        )
        loss.backward()
        self.assertEqual(prediction.shape, targets.shape)
        self.assertTrue(bool(torch.isfinite(prediction).all()))
        for block in (
            model.layer1,
            model.initial_market,
            model.initial_prefix,
            model.layer3,
            model.layer4,
            model.layer7,
            model.layer8,
            model.layer2,
        ):
            gradient = block.output.weight.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(float(gradient.norm()), 0.0)

    def test_optimizer_routing_covers_every_trainable_parameter_once(self) -> None:
        model = self.make_model(1, 1)
        muon, adamw = optimizer_parameter_groups(model)
        routed = {id(value) for value in (*muon, *adamw)}
        trainable = {id(value) for value in model.parameters() if value.requires_grad}
        self.assertEqual(routed, trainable)
        self.assertTrue(muon)
        self.assertTrue(adamw)

    def test_metrics_headline_raw_return_and_retain_feature_state(self) -> None:
        metrics = FeatureMetricAccumulator(
            torch.tensor([0.0, 0.0]), torch.tensor([2.0, 4.0])
        )
        metrics.add(
            torch.tensor([[[1.0, 1.0]], [[-1.0, 3.0]]]),
            torch.tensor([[[2.0, -1.0]], [[-2.0, 1.0]]]),
            torch.ones(2),
        )
        result = metrics.result()
        headline = result["nextReturn"]
        self.assertEqual(headline["evaluationScope"], COMPARABLE_EVALUATION_SCOPE)
        self.assertEqual(headline["examples"], 2)
        self.assertAlmostEqual(headline["normalizedMse"], 0.25)
        self.assertAlmostEqual(headline["mse"], 1.0)
        self.assertAlmostEqual(headline["zeroBaselineMse"], 4.0)
        self.assertAlmostEqual(headline["mseSkillVsZero"], 0.75)
        self.assertAlmostEqual(headline["directionAccuracy"], 1.0)
        self.assertAlmostEqual(headline["correlation"], 1.0)
        self.assertEqual(result["featureState"]["featureCount"], 2)

    def test_sequence_metrics_headline_step_one_and_report_every_step(self) -> None:
        metrics = FeatureSequenceMetricAccumulator(
            torch.tensor([0.0, 0.0]),
            torch.tensor([2.0, 4.0]),
            output_steps=2,
        )
        metrics.add(
            torch.tensor([
                [[1.0, 1.0], [0.0, 0.0]],
                [[-1.0, 3.0], [0.0, 0.0]],
            ]),
            torch.tensor([
                [[2.0, -1.0], [4.0, 0.0]],
                [[-2.0, 1.0], [-4.0, 0.0]],
            ]),
            torch.ones(2),
        )
        result = metrics.result()
        self.assertEqual(len(result["perStepNextReturn"]), 2)
        self.assertAlmostEqual(result["nextReturn"]["normalizedMse"], 0.25)
        self.assertAlmostEqual(
            result["perStepNextReturn"][1]["normalizedMse"], 4.0
        )
        self.assertAlmostEqual(result["returnPath"]["normalizedMse"], 2.125)
        self.assertEqual(len(result["perStepFeatureState"]), 2)


class StructuredFeatureSequenceDatasetTest(unittest.TestCase):
    def test_sparse_episode_sampling_is_deterministic_and_sequential(self) -> None:
        origins, episodes = sparse_episode_origins(
            timeline_rows=100_001,
            example_count=25_000,
            target_episode_seconds=3_600,
            seed=19,
        )
        repeated, _ = sparse_episode_origins(
            timeline_rows=100_001,
            example_count=25_000,
            target_episode_seconds=3_600,
            seed=19,
        )
        np.testing.assert_array_equal(origins, repeated)
        self.assertEqual(origins.size, 25_000)
        self.assertEqual(len(episodes), 7)
        self.assertEqual(sum(episode_lengths(25_000, 3_600)), 25_000)
        self.assertLessEqual(
            max(item["examples"] for item in episodes)
            - min(item["examples"] for item in episodes),
            1,
        )
        for episode in episodes:
            selected = origins[
                (origins >= episode["startOrigin"])
                & (origins < episode["stopOriginExclusive"])
            ]
            self.assertEqual(selected.size, episode["examples"])
            self.assertTrue(bool(np.all(np.diff(selected) == 1)))

    def test_targets_are_complete_feature_rows_after_each_origin(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            files: dict[str, dict[str, str]] = {}
            timeline_rows: dict[str, int] = {}
            counts: dict[str, int] = {}
            expected_timeline = None
            for split_index, split in enumerate(("train", "validation", "test")):
                timeline = (
                    np.arange(30, dtype=np.float32).reshape(10, 3)
                    + split_index * 100
                )
                origins = np.asarray([2, 5], dtype="<i4")
                timeline_file = f"{split}.timeline-features.f32"
                origins_file = f"{split}.origins.i32"
                timeline.astype("<f4").tofile(root / timeline_file)
                origins.tofile(root / origins_file)
                files[split] = {
                    "timelineFeatures": timeline_file,
                    "origins": origins_file,
                }
                timeline_rows[split] = 10
                counts[split] = 2
                if split == "train":
                    expected_timeline = timeline
            (root / "manifest.json").write_text(json.dumps({
                "storageLayout": "temporal-channel-timeline-v1",
                "temporalChannelCount": 3,
                "examplesBySplit": counts,
                "timelineRowsBySplit": timeline_rows,
                "files": files,
            }), encoding="utf-8")
            dataset = StructuredFeatureSequenceDataset(
                root, input_steps=2, output_steps=2, train_examples=2
            )
            inputs, targets = dataset._examples(
                "train", np.asarray([0], dtype=np.int64)
            )
            assert expected_timeline is not None
            np.testing.assert_array_equal(inputs[0], expected_timeline[[1, 2]])
            np.testing.assert_array_equal(targets[0], expected_timeline[[3, 4]])
            dataset.close()


if __name__ == "__main__":
    unittest.main()
