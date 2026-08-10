from __future__ import annotations

import unittest

import numpy as np
import torch

from next_return_dataset import (
    DAY_SECONDS,
    EXAMPLE_SPAN_MS,
    HISTORY_RETURN_COUNT,
    SECOND_MS,
    daily_log_return_examples,
    example_rows,
    select_example_shards,
    validate_split_disjointness,
)
from normalized_glu_next_return import (
    depth_width_parameter_assignments,
    NormalizedGluNextReturn,
    PER_SEQUENCE_INPUT_NORMALIZATION,
    PER_SEQUENCE_REVERSIBLE_INPUT_NORMALIZATION,
    PER_SEQUENCE_REVERSIBLE_WITH_STATS_INPUT_NORMALIZATION,
    optimizer_parameter_groups,
)
from train_next_return_memorization import (
    assert_inactive_parameter_values_unchanged,
    mask_inactive_parameter_updates,
    snapshot_inactive_parameter_values,
    validate_plan as validate_memorization_plan,
)


def source_shard(
    split: str,
    start: int,
    count: int,
    *,
    row_offset: int = 0,
) -> dict:
    return {
        "split": split,
        "predictionTimeStart": start,
        "count": count,
        "date": "2024-01-01",
        "featureRowOffset": row_offset,
        "featureRowStride": 1,
    }


class NextReturnDatasetTest(unittest.TestCase):
    def test_daily_examples_align_history_and_immediate_target(self) -> None:
        log_step = 1e-6
        previous = np.exp(
            np.arange(-DAY_SECONDS, 0, dtype=np.float64) * log_step
        )
        current = np.exp(
            np.arange(0, DAY_SECONDS, dtype=np.float64) * log_step
        )
        following = np.exp(
            np.arange(DAY_SECONDS, 2 * DAY_SECONDS, dtype=np.float64) * log_step
        )
        history, target = daily_log_return_examples(
            previous,
            current,
            following,
        )
        self.assertEqual(history.shape, (DAY_SECONDS, HISTORY_RETURN_COUNT))
        self.assertEqual(target.shape, (DAY_SECONDS,))
        np.testing.assert_allclose(history, log_step, rtol=0, atol=1e-8)
        np.testing.assert_allclose(target, log_step, rtol=0, atol=1e-8)

    def test_every_source_second_is_a_distinct_unit_weight_example(self) -> None:
        rows, weights = example_rows(7, 181)
        np.testing.assert_array_equal(rows, np.arange(7, 188))
        np.testing.assert_array_equal(weights, np.ones(181))

    def test_oracle_style_selection_uses_121_second_embargo_and_test_tail(self) -> None:
        second = SECOND_MS
        manifest = {
            "shards": [
                source_shard("train", 1_000 * second, 100),
                source_shard("validation", 1_100 * second, 100),
                source_shard("validation", 1_400 * second, 100),
                source_shard("test", 2_000 * second, 300, row_offset=500),
            ],
        }
        selected = select_example_shards(manifest, test_count=120)
        validate_split_disjointness(selected)
        self.assertEqual(len(selected["validation"]), 1)
        self.assertEqual(
            selected["validation"][0].decision_time_start,
            1_400 * second,
        )
        self.assertEqual(selected["test"][0].count, 120)
        self.assertEqual(selected["test"][0].row_offset, 680)
        self.assertEqual(EXAMPLE_SPAN_MS, 121 * SECOND_MS)


class NormalizedGluNextReturnTest(unittest.TestCase):
    def test_memorization_plan_accepts_scaled_layer_width(self) -> None:
        validate_memorization_plan({
            "id": "scaled-width",
            "datasetDir": "data/training/datasets/scaled-width",
            "runDir": "data/training/runs/scaled-width",
            "historyDir": "data/market/immutable/refs/candles/example",
            "subset": {
                "type": "fixed-contiguous",
                "date": "2026-04-01",
                "examples": 1_048_576,
            },
            "architecture": {
                "widths": [1024] * 8,
                "dropout": 0,
                "dropoutRate": 0,
                "learnableCentering": False,
            },
            "training": {
                "epochs": 512,
                "batchSize": 4096,
                "evaluationBatchSize": 16384,
                "learningRate": 1e-4,
                "targetNormalizedMse": 1e-4,
                "mixedPrecision": "float32",
                "device": "cuda",
            },
        })

    def build_model(self) -> NormalizedGluNextReturn:
        return NormalizedGluNextReturn(
            torch.zeros(HISTORY_RETURN_COUNT),
            torch.ones(HISTORY_RETURN_COUNT),
            torch.tensor(2e-6),
            torch.tensor(7e-4),
            widths=(16,),
            dropout=0,
        )

    def test_scalar_head_starts_at_training_mean_and_backpropagates(self) -> None:
        model = self.build_model()
        features = torch.randn(5, HISTORY_RETURN_COUNT)
        prediction = model(features)
        self.assertEqual(prediction.shape, (5,))
        torch.testing.assert_close(prediction, torch.full((5,), 2e-6))
        prediction.square().mean().backward()
        self.assertIsNotNone(model.output.weight.grad)
        self.assertGreater(float(model.output.weight.grad.abs().sum()), 0)

    def test_value_and_gate_centering_are_independent_and_routed_to_adamw(self) -> None:
        model = self.build_model()
        self.assertEqual(len(model.layers), 1)
        for value, gate in zip(
            model.value_centering_normalizers,
            model.gate_centering_normalizers,
            strict=True,
        ):
            self.assertIsNot(value.weight, gate.weight)
        muon, adamw = optimizer_parameter_groups(model)
        muon_ids = {id(value) for value in muon}
        adamw_ids = {id(value) for value in adamw}
        self.assertFalse(muon_ids & adamw_ids)
        self.assertIn(id(model.layers[0].weight), muon_ids)
        self.assertIn(id(model.value_centering_normalizers[0].weight), adamw_ids)
        self.assertIn(id(model.output.weight), adamw_ids)

    def test_multiple_glu_layers_are_stacked_and_trainable(self) -> None:
        model = NormalizedGluNextReturn(
            torch.zeros(HISTORY_RETURN_COUNT),
            torch.ones(HISTORY_RETURN_COUNT),
            torch.zeros(5),
            torch.ones(5),
            widths=(16, 12),
            dropout=0,
        )
        self.assertEqual(model.widths, (16, 12))
        self.assertEqual(len(model.layers), 2)
        self.assertEqual(model.layers[0].in_features, HISTORY_RETURN_COUNT)
        self.assertEqual(model.layers[0].out_features, 32)
        self.assertEqual(model.layers[1].in_features, 16)
        self.assertEqual(model.layers[1].out_features, 24)
        prediction = model(torch.randn(7, HISTORY_RETURN_COUNT))
        self.assertEqual(prediction.shape, (7, 5))
        prediction.square().mean().backward()
        self.assertIsNotNone(model.layers[1].weight.grad)

    def test_centering_can_be_fixed_at_the_canonical_projector(self) -> None:
        model = NormalizedGluNextReturn(
            torch.zeros(HISTORY_RETURN_COUNT),
            torch.ones(HISTORY_RETURN_COUNT),
            torch.tensor(0.0),
            torch.tensor(1.0),
            widths=(16,),
            learnable_centering=False,
            dropout=0,
        )
        expected = torch.eye(16) - torch.full((16, 16), 1 / 16)
        for normalizer in (
            *model.value_centering_normalizers,
            *model.gate_centering_normalizers,
        ):
            self.assertFalse(normalizer.weight.requires_grad)
            torch.testing.assert_close(normalizer.weight, expected)
            self.assertTrue(normalizer.raw_scale.requires_grad)
        _muon, adamw = optimizer_parameter_groups(model)
        adamw_ids = {id(parameter) for parameter in adamw}
        self.assertNotIn(
            id(model.value_centering_normalizers[0].weight), adamw_ids
        )

    def test_depth_width_partition_freezes_three_quadrants_exactly(self) -> None:
        model = NormalizedGluNextReturn(
            torch.zeros(HISTORY_RETURN_COUNT),
            torch.ones(HISTORY_RETURN_COUNT),
            torch.tensor(0.0),
            torch.tensor(1.0),
            widths=(4, 4, 4, 4),
            learnable_centering=False,
            dropout=0,
        )
        assignments = depth_width_parameter_assignments(model)
        counts = [
            sum(
                int((assignment == index).sum())
                for _parameter, assignment in assignments
            )
            for index in range(4)
        ]
        self.assertTrue(all(count > 0 for count in counts))
        self.assertEqual(
            sum(counts),
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
        )
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters()
             if parameter.requires_grad),
            lr=1e-2,
            weight_decay=0,
        )
        before = {
            id(parameter): parameter.detach().clone()
            for parameter, _assignment in assignments
        }
        loss = (model(torch.randn(8, HISTORY_RETURN_COUNT)) - 1).square().mean()
        loss.backward()
        frozen = snapshot_inactive_parameter_values(assignments, 2)
        mask_inactive_parameter_updates(
            assignments, 2, (optimizer,)
        )
        optimizer.step()
        assert_inactive_parameter_values_unchanged(frozen)
        active_changed = False
        for parameter, assignment in assignments:
            original = before[id(parameter)]
            inactive = assignment != 2
            torch.testing.assert_close(parameter[inactive], original[inactive])
            active = ~inactive
            if active.any() and not torch.equal(
                parameter.detach()[active], original[active]
            ):
                active_changed = True
        self.assertTrue(active_changed)

    def test_rejects_an_empty_glu_stack(self) -> None:
        with self.assertRaisesRegex(ValueError, "one or more GLU layers"):
            NormalizedGluNextReturn(
                torch.zeros(HISTORY_RETURN_COUNT),
                torch.ones(HISTORY_RETURN_COUNT),
                torch.tensor(0.0),
                torch.tensor(1.0),
                widths=(),
            )

    def test_per_sequence_input_normalization_uses_each_history_only(self) -> None:
        model = NormalizedGluNextReturn(
            torch.full((HISTORY_RETURN_COUNT,), 100.0),
            torch.full((HISTORY_RETURN_COUNT,), 7.0),
            torch.tensor(0.0),
            torch.tensor(1.0),
            widths=(16,),
            input_normalization=PER_SEQUENCE_INPUT_NORMALIZATION,
            dropout=0,
        )
        features = torch.stack((
            torch.arange(HISTORY_RETURN_COUNT, dtype=torch.float32),
            3 * torch.arange(HISTORY_RETURN_COUNT, dtype=torch.float32) + 17,
            torch.full((HISTORY_RETURN_COUNT,), 5.0),
        ))
        normalized = model.normalize_features(features)
        torch.testing.assert_close(
            normalized[:2].mean(dim=1), torch.zeros(2), atol=1e-6, rtol=0
        )
        torch.testing.assert_close(
            normalized[:2].square().mean(dim=1),
            torch.ones(2),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(normalized[2], torch.zeros(HISTORY_RETURN_COUNT))

    def test_reversible_per_sequence_output_uses_input_mean_and_std(self) -> None:
        model = NormalizedGluNextReturn(
            torch.full((HISTORY_RETURN_COUNT,), 100.0),
            torch.full((HISTORY_RETURN_COUNT,), 7.0),
            torch.full((3,), -50.0),
            torch.full((3,), 20.0),
            widths=(16,),
            input_normalization=PER_SEQUENCE_REVERSIBLE_INPUT_NORMALIZATION,
            dropout=0,
        )
        with torch.no_grad():
            model.output.bias.fill_(2.0)
        features = torch.stack((
            torch.arange(HISTORY_RETURN_COUNT, dtype=torch.float32),
            3 * torch.arange(HISTORY_RETURN_COUNT, dtype=torch.float32) + 17,
        ))
        sequence_mean = features.mean(dim=1, keepdim=True)
        sequence_std = (
            (features - sequence_mean).square().mean(dim=1, keepdim=True).sqrt()
        )
        prediction = model(features)
        expected = (sequence_mean + 2 * sequence_std).expand(-1, 3)
        torch.testing.assert_close(prediction, expected)

    def test_reversible_sequence_stats_are_explicit_side_features(self) -> None:
        model = NormalizedGluNextReturn(
            torch.zeros(HISTORY_RETURN_COUNT),
            torch.full((HISTORY_RETURN_COUNT,), 2.0),
            torch.zeros(3),
            torch.ones(3),
            widths=(16,),
            input_normalization=(
                PER_SEQUENCE_REVERSIBLE_WITH_STATS_INPUT_NORMALIZATION
            ),
            dropout=0,
        )
        features = torch.arange(
            HISTORY_RETURN_COUNT, dtype=torch.float32
        ).unsqueeze(0)
        normalized = model.normalize_features(features)
        sequence_mean = features.mean()
        sequence_std = (features - sequence_mean).square().mean().sqrt()
        self.assertEqual(normalized.shape, (1, HISTORY_RETURN_COUNT + 2))
        self.assertEqual(model.layers[0].in_features, HISTORY_RETURN_COUNT + 2)
        torch.testing.assert_close(
            normalized[0, -2], sequence_mean / 2.0
        )
        torch.testing.assert_close(
            normalized[0, -1], sequence_std / 2.0 - 1.0
        )
        prediction = model(features)
        torch.testing.assert_close(
            prediction,
            sequence_mean.expand(1, 3),
        )


if __name__ == "__main__":
    unittest.main()
