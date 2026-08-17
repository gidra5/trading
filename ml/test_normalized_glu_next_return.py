from __future__ import annotations

import unittest

import numpy as np
import torch

from next_return_dataset import (
    DAY_SECONDS,
    EXAMPLE_SPAN_MS,
    HISTORY_RETURN_COUNT,
    SECOND_MS,
    ExampleShard,
    daily_causal_volatility,
    daily_log_return_examples,
    example_rows,
    select_example_shards,
    validate_split_disjointness,
)
from normalized_glu_next_return import (
    CAUSAL_VOLATILITY_INPUT_NORMALIZATION,
    depth_width_parameter_assignments,
    NormalizedGluNextReturn,
    PER_SEQUENCE_INPUT_NORMALIZATION,
    PER_SEQUENCE_REVERSIBLE_INPUT_NORMALIZATION,
    PER_SEQUENCE_REVERSIBLE_WITH_STATS_INPUT_NORMALIZATION,
    optimizer_parameter_groups,
)
from train_next_return_memorization import (
    assert_inactive_parameter_values_unchanged,
    causal_volatility_variant,
    dropout_variant,
    l2_variant,
    mask_inactive_parameter_updates,
    matrix_l2_penalty,
    optimizer_weight_decay_variant,
    runner_contract,
    snapshot_inactive_parameter_values,
    validate_plan as validate_memorization_plan,
    swa_variant,
)
from train_normalized_glu_next_return import NextReturnDataset, uniform_group_cvar
from swa import EpochSwaSweep


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
    def test_uniform_group_cvar_averages_the_requested_worst_mass(self) -> None:
        losses = torch.tensor([10.0, 4.0, 1.0, 0.0])
        self.assertEqual(float(uniform_group_cvar(losses, 0.25)), 10.0)
        self.assertEqual(float(uniform_group_cvar(losses, 0.50)), 7.0)
        self.assertEqual(float(uniform_group_cvar(losses, 0.75)), 5.0)
        self.assertEqual(float(uniform_group_cvar(losses, 1.00)), 3.75)

    def test_exact_zero_targets_are_excluded_from_counts_and_batches(self) -> None:
        dataset = NextReturnDataset.__new__(NextReturnDataset)
        dataset.shards = {
            "train": [ExampleShard("train", 0, 5, "2024-01-01", 0)]
        }
        dataset.horizon_return_count = 1
        dataset.row_stride = 1
        dataset.exclude_zero_targets = True
        history = np.arange(5 * HISTORY_RETURN_COUNT, dtype=np.float32).reshape(
            5, HISTORY_RETURN_COUNT
        )
        target = np.array([0, -1, 0, 2, 3], dtype=np.float32)
        dataset._component = lambda _day: (history, target)

        self.assertEqual(dataset.logical_count("train"), 3)
        batches = list(dataset.iter_batches(
            "train", 8, shuffle=False, seed=1
        ))
        self.assertEqual(len(batches), 1)
        features, targets, weights = batches[0]
        torch.testing.assert_close(targets, torch.tensor([-1.0, 2.0, 3.0]))
        torch.testing.assert_close(weights, torch.ones(3))
        torch.testing.assert_close(
            features[:, 0], torch.tensor([120.0, 360.0, 480.0])
        )

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

    def test_long_causal_volatility_ends_before_the_target(self) -> None:
        previous_returns = np.full(DAY_SECONDS, 2e-4, dtype=np.float64)
        current_returns = np.full(DAY_SECONDS, 3e-4, dtype=np.float64)
        previous = np.exp(np.cumsum(previous_returns))
        current = previous[-1] * np.exp(np.cumsum(current_returns))
        volatility = daily_causal_volatility(
            previous, current, window=7_200
        )
        expected_first = np.sqrt(
            (7_199 * (2e-4 ** 2) + 3e-4 ** 2) / 7_200
        )
        self.assertEqual(volatility.shape, (DAY_SECONDS,))
        self.assertAlmostEqual(float(volatility[0]), expected_first, places=10)
        self.assertAlmostEqual(float(volatility[7_199]), 3e-4, places=10)

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
    def test_memorization_plan_accepts_scaled_width_and_dropout(self) -> None:
        plan = {
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
                "dropout": 0.05,
                "dropoutRate": 0.5,
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
        }
        validate_memorization_plan(plan)
        self.assertIn("dropout-regularization", runner_contract(plan))

        plan["architecture"]["dropoutRate"] = 0
        with self.assertRaisesRegex(ValueError, "architecture is invalid"):
            validate_memorization_plan(plan)

    def test_dropout_variant_uses_independent_run_and_dataset_paths(self) -> None:
        source = {
            "id": "pure",
            "label": "Pure",
            "datasetDir": "data/training/datasets/pure",
            "runDir": "data/training/runs/pure",
            "architecture": {"dropout": 0, "dropoutRate": 0},
        }
        variant = dropout_variant(
            source,
            probability=0.05,
            application_rate=0.5,
            suffix="dropout-p05-rate50-v1",
        )
        self.assertEqual(source["architecture"]["dropout"], 0)
        self.assertEqual(variant["architecture"]["dropout"], 0.05)
        self.assertTrue(variant["runDir"].endswith("dropout-p05-rate50-v1"))
        self.assertTrue(variant["datasetDir"].endswith("dropout-p05-rate50-v1"))

    def test_causal_volatility_variant_uses_independent_paths(self) -> None:
        source = {
            "id": "pure",
            "label": "Pure",
            "datasetDir": "data/training/datasets/pure",
            "runDir": "data/training/runs/pure",
            "architecture": {"dropout": 0, "dropoutRate": 0},
        }
        variant = causal_volatility_variant(
            source, window=15, suffix="causal-volatility-w15-v1"
        )
        self.assertNotIn("inputNormalization", source["architecture"])
        self.assertEqual(
            variant["architecture"]["inputNormalization"],
            CAUSAL_VOLATILITY_INPUT_NORMALIZATION,
        )
        self.assertEqual(variant["architecture"]["volatilityWindow"], 15)
        self.assertTrue(variant["runDir"].endswith("causal-volatility-w15-v1"))
        long_variant = causal_volatility_variant(
            source, window=14_400, suffix="causal-volatility-w14400-v1"
        )
        self.assertEqual(long_variant["architecture"]["volatilityWindow"], 14_400)

    def test_swa_variant_uses_one_shared_sweep_run(self) -> None:
        source = {
            "id": "pure",
            "label": "Pure",
            "datasetDir": "data/training/datasets/pure",
            "runDir": "data/training/runs/pure",
            "architecture": {"dropout": 0, "dropoutRate": 0},
            "training": {},
        }
        variant = swa_variant(
            source, suffix="swa-sweep-v1"
        )
        self.assertNotIn("swa", source["training"])
        self.assertEqual(
            variant["training"]["swa"]["updateInterval"],
            "epoch-end",
        )
        self.assertTrue(
            variant["runDir"].endswith("swa-sweep-v1")
        )

    def test_l2_variant_adds_an_explicit_loss_contract(self) -> None:
        source = {
            "id": "pure",
            "label": "Pure",
            "datasetDir": "data/training/datasets/pure",
            "runDir": "data/training/runs/pure",
            "architecture": {"dropout": 0, "dropoutRate": 0},
            "training": {},
        }
        variant = l2_variant(
            source, rate=1e-4, suffix="l2-rate-1e-4-v1"
        )
        self.assertNotIn("l2Regularization", source["training"])
        self.assertEqual(
            variant["training"]["l2Regularization"],
            {
                "type": "explicit-loss-term",
                "coefficient": 1e-4,
                "parameters": "all-trainable-matrices",
                "reduction": "half-sum-squared",
            },
        )

    def test_l2_penalty_includes_weights_but_not_biases(self) -> None:
        model = torch.nn.Linear(2, 1, bias=True)
        with torch.no_grad():
            model.weight.fill_(2)
            model.bias.fill_(100)
        torch.testing.assert_close(
            matrix_l2_penalty(model), torch.tensor(4.0)
        )

    def test_optimizer_weight_decay_variant_updates_both_groups(self) -> None:
        source = {
            "id": "pure",
            "label": "Pure",
            "datasetDir": "data/training/datasets/pure",
            "runDir": "data/training/runs/pure",
            "architecture": {"dropout": 0, "dropoutRate": 0},
            "training": {
                "optimizer": {
                    "muon": {"weightDecay": 0},
                    "adamw": {"weightDecay": 0},
                },
            },
        }
        variant = optimizer_weight_decay_variant(
            source, rate=1e-4, suffix="optimizer-wd-1e-4-v1"
        )
        self.assertEqual(
            source["training"]["optimizer"]["muon"]["weightDecay"], 0
        )
        self.assertEqual(
            variant["training"]["optimizer"]["muon"]["weightDecay"], 1e-4
        )
        self.assertEqual(
            variant["training"]["optimizer"]["adamw"]["weightDecay"], 1e-4
        )

    def test_epoch_swa_tracks_start_fraction_candidates(self) -> None:
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        sweep = EpochSwaSweep(model, maximum_epochs=4)
        for epoch, value in enumerate((1.0, 2.0, 3.0, 4.0)):
            with torch.no_grad():
                model.weight.fill_(value)
            sweep.update(model, epoch=epoch)
        states = sweep.candidate_states()
        torch.testing.assert_close(
            states["swa-start-50-percent"]["weight"],
            torch.tensor([[3.0]]),
        )
        torch.testing.assert_close(
            states["swa-start-75-percent"]["weight"],
            torch.tensor([[3.5]]),
        )
        torch.testing.assert_close(
            states["swa-start-90-percent"]["weight"],
            torch.tensor([[4.0]]),
        )
        restored = EpochSwaSweep(model, maximum_epochs=4)
        restored.load_state_dict(sweep.state_dict())
        self.assertEqual(restored.completed_epochs, 4)
        torch.testing.assert_close(
            restored.candidate_states()["swa-start-50-percent"]["weight"],
            torch.tensor([[3.0]]),
        )

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

    def test_causal_volatility_uses_only_the_requested_history_tail(self) -> None:
        model = NormalizedGluNextReturn(
            torch.zeros(HISTORY_RETURN_COUNT),
            torch.ones(HISTORY_RETURN_COUNT),
            torch.tensor(0.0),
            torch.tensor(0.001),
            widths=(16,),
            input_normalization=CAUSAL_VOLATILITY_INPUT_NORMALIZATION,
            volatility_window=2,
            dropout=0,
        )
        with torch.no_grad():
            model.output.bias.fill_(2.0)
        features = torch.zeros((1, HISTORY_RETURN_COUNT))
        features[0, 0] = 100.0
        features[0, -2:] = torch.tensor([-0.0003, 0.0004])
        expected_volatility = torch.sqrt(torch.tensor(1.35e-7))
        torch.testing.assert_close(
            model.causal_volatility(features), expected_volatility.reshape(1)
        )
        normalized = model.normalize_features(features)
        torch.testing.assert_close(
            normalized[0, -2:], features[0, -2:] / expected_volatility
        )
        torch.testing.assert_close(
            model(features), (2 * expected_volatility).reshape(1)
        )

    def test_causal_volatility_requires_a_valid_window(self) -> None:
        with self.assertRaisesRegex(ValueError, "window must be in"):
            NormalizedGluNextReturn(
                torch.zeros(HISTORY_RETURN_COUNT),
                torch.ones(HISTORY_RETURN_COUNT),
                torch.tensor(0.0),
                torch.tensor(1.0),
                widths=(16,),
                input_normalization=CAUSAL_VOLATILITY_INPUT_NORMALIZATION,
                dropout=0,
            )

    def test_long_causal_volatility_uses_external_scale_without_extra_features(
        self,
    ) -> None:
        model = NormalizedGluNextReturn(
            torch.zeros(HISTORY_RETURN_COUNT),
            torch.ones(HISTORY_RETURN_COUNT),
            torch.tensor(0.0),
            torch.tensor(0.001),
            widths=(16,),
            input_normalization=CAUSAL_VOLATILITY_INPUT_NORMALIZATION,
            volatility_window=14_400,
            dropout=0,
        )
        self.assertEqual(model.layers[0].in_features, HISTORY_RETURN_COUNT)
        features = torch.full((2, HISTORY_RETURN_COUNT), 2e-4)
        rms = torch.tensor([3e-4, 4e-4])
        expected = torch.sqrt(rms.square() + 1e-8)
        torch.testing.assert_close(model.causal_volatility(features, rms), expected)
        with self.assertRaisesRegex(ValueError, "require an external"):
            model(features)

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
