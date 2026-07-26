from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch
import zstandard

from mlp_model import (
    DirectLossWeights,
    FEATURE_SCHEMA_VERSION,
    INPUT_FEATURE_COUNT,
    ExposureMlp,
    LossWeights,
    PolicySupport,
    TimeWeighting,
    conditional_transaction_transition,
    distance_imbalance_time_weights,
    direct_oracle_loss,
    fitted_teacher_loss,
    gaussian_oracle_mutual_information,
    gaussian_temporal_mutual_information,
    materialize_raw_oracle_policy_map,
    persistent_distance_imbalance_time_weights,
    surface_probability_mse_per_example,
)
from train_mlp import (
    CompactMinuteFeatureComponent,
    FittedPolicyDataset,
    RuntimeMinuteOracleRows,
    Shard,
    TemporalBlockBatchSampler,
    cached_training_normalization,
    cached_training_parameter_scale,
    continuation_learning_rate_multiplier,
    merge_weighted_moments,
    resume_contract_is_monotonic_extension,
    scheduler_resume_steps,
    validate_dataset_manifest,
    weighted_standard_deviation,
    weighted_variance,
)


class MarketOnlyInputContractTests(unittest.TestCase):
    def test_model_uses_only_901_market_features(self) -> None:
        self.assertEqual(FEATURE_SCHEMA_VERSION, 6)
        self.assertEqual(INPUT_FEATURE_COUNT, 901)

    def test_resume_contract_allows_only_monotonic_horizon_extension(self) -> None:
        original = {
            "datasetVersion": 12,
            "epochs": 64,
            "patience": 8,
        }
        extended = {
            **original,
            "epochs": 256,
            "patience": 16,
        }
        self.assertTrue(resume_contract_is_monotonic_extension(
            original,
            extended,
        ))
        self.assertFalse(resume_contract_is_monotonic_extension(
            original,
            {**extended, "datasetVersion": 13},
        ))
        self.assertFalse(resume_contract_is_monotonic_extension(
            original,
            {**extended, "patience": 4},
        ))
        target_only = {
            "datasetVersion": 12,
            "epochs": 256,
            "patience": None,
            "earlyStopping": "target-only",
        }
        self.assertTrue(resume_contract_is_monotonic_extension(
            target_only,
            {**target_only, "epochs": 512},
        ))
        self.assertFalse(resume_contract_is_monotonic_extension(
            target_only,
            {**target_only, "patience": 16},
        ))

    def test_scheduler_resume_rebases_changed_batches_per_epoch(self) -> None:
        checkpoint = {
            "epoch": 51,
            "globalStep": 19_240,
            "scheduler": {"last_epoch": 19_240},
        }

        self.assertEqual(
            scheduler_resume_steps(checkpoint, optimizer_steps_per_epoch=77),
            (19_240, 4_004),
        )

    def test_scheduler_resume_keeps_partial_current_epoch(self) -> None:
        checkpoint = {
            "epoch": 456,
            "globalStep": 50_573,
            "scheduler": {"last_epoch": 35_182},
        }

        self.assertEqual(
            scheduler_resume_steps(checkpoint, optimizer_steps_per_epoch=77),
            (35_182, 35_182),
        )

    def test_continuation_schedule_is_smooth_and_monotonic(self) -> None:
        start = 172_723
        total = 983_000
        initial = 0.2612729393
        values = [
            continuation_learning_rate_multiplier(
                step,
                total,
                start,
                initial,
            )
            for step in (start, start + 1, 400_000, total)
        ]
        self.assertAlmostEqual(values[0], initial)
        self.assertLess(abs(values[1] - initial), 1e-9)
        self.assertTrue(all(
            left >= right
            for left, right in zip(values, values[1:])
        ))
        self.assertAlmostEqual(values[-1], 0.05)


    def test_runtime_minute_targets_follow_latest_completed_minute(self) -> None:
        day = 1_750_000_000_000 // 86_400_000 * 86_400_000
        probabilities = np.repeat(
            np.arange(1_441, dtype=np.float32)[:, None],
            8,
            axis=1,
        )
        shard = Shard(
            Path("."),
            4,
            "features",
            0,
            1,
            "parameters",
            "metrics",
            "raw",
            "minute",
            "resolution",
            58,
            1,
            "base-weights",
            "weights",
            day + 60_000,
            day + 58_999,
        )
        targets = RuntimeMinuteOracleRows({day: probabilities}, shard, 8)

        np.testing.assert_array_equal(
            targets[:4][:, 0],
            np.asarray([0, 1, 1, 1], dtype=np.float32),
        )


class DirectOracleTrainingPathTests(unittest.TestCase):
    def test_base_ce_and_pmse_match_direct_distribution_reference(self) -> None:
        generator = torch.Generator().manual_seed(7)
        actions = torch.linspace(-250.0, 250.0, 255)
        target = torch.softmax(torch.randn(5, 255, generator=generator), dim=-1)
        logits = torch.randn(5, 255, generator=generator, requires_grad=True)
        time_weights = torch.tensor([1.0, 3.0, 0.5, 2.0, 4.0])
        weights = DirectLossWeights(
            cross_entropy=1.0,
            probability_mse=0.75,
            excess_entropy=0.0,
            temporal_mutual_information=0.0,
            oracle_mutual_information=0.0,
        )

        actual = direct_oracle_loss(
            logits,
            target,
            actions,
            torch.linspace(-100.0, 100.0, 31).view(1, -1),
            PolicySupport(-250.0, 250.0, -100.0, 100.0, 0.00175, 0.01),
            weights,
            time_weights,
            torch.arange(5, dtype=torch.int64) * 1_000,
            1_000,
            include_diagnostics=False,
            required_loss_terms=frozenset(("cross_entropy", "probability_mse")),
        )
        visible = (actions >= -100.0) & (actions <= 100.0)
        predicted_log = torch.log_softmax(
            logits.masked_fill(~visible, torch.finfo(torch.float32).min),
            dim=-1,
        )
        predicted = predicted_log.exp()
        visible_target = target.masked_fill(~visible, 0.0)
        visible_target = visible_target / visible_target.sum(dim=-1, keepdim=True)
        normalized_weight = time_weights / time_weights.mean()
        cross_entropy = (
            normalized_weight * -(visible_target * predicted_log).sum(dim=-1)
        ).sum() / normalized_weight.sum()
        probability_mse = (
            normalized_weight
            * (visible_target - predicted).square().mean(dim=-1)
        ).sum() / normalized_weight.sum()
        expected = cross_entropy + 0.75 * probability_mse

        torch.testing.assert_close(actual["probabilityMse"], probability_mse)
        torch.testing.assert_close(actual["loss"], expected)

    def test_oracle_mi_matches_base_distribution_moments(self) -> None:
        generator = torch.Generator().manual_seed(23)
        actions = torch.linspace(-250.0, 250.0, 255)
        current = torch.linspace(-100.0, 100.0, 31).view(1, -1)
        support = PolicySupport(
            -250.0,
            250.0,
            -100.0,
            100.0,
            0.00175,
            0.01,
        )
        logits = torch.randn(8, 255, generator=generator)
        target = torch.softmax(torch.randn(8, 255, generator=generator), dim=-1)
        times = torch.arange(8, dtype=torch.int64) * 1_000
        time_weights = torch.linspace(1.0, 2.0, 8)
        result = direct_oracle_loss(
            logits,
            target,
            actions,
            current,
            support,
            DirectLossWeights(
                cross_entropy=0.0,
                probability_mse=0.0,
                excess_entropy=0.0,
                temporal_mutual_information=0.0,
                oracle_mutual_information=1.0,
            ),
            time_weights,
            times,
            1_000,
            include_diagnostics=False,
            required_loss_terms=frozenset(("oracle_mutual_information",)),
        )
        visible = (actions >= -100.0) & (actions <= 100.0)
        predicted = torch.softmax(
            logits.masked_fill(~visible, torch.finfo(torch.float32).min),
            dim=-1,
        )
        normalized_target = target.masked_fill(~visible, 0.0)
        normalized_target = (
            normalized_target
            / normalized_target.sum(dim=-1, keepdim=True)
        )
        action = actions.view(1, -1)
        expected = gaussian_oracle_mutual_information(
            (predicted * action).sum(dim=-1, keepdim=True),
            (predicted * action.square()).sum(dim=-1, keepdim=True),
            (normalized_target * action).sum(dim=-1, keepdim=True),
            (normalized_target * action.square()).sum(dim=-1, keepdim=True),
            times,
            time_weights / time_weights.mean(),
            1_000,
            actions.numel(),
        )

        torch.testing.assert_close(result["oracleMutualInformation"], expected)
        torch.testing.assert_close(result["loss"], -expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_compiled_base_loss_is_finite_under_fp16_autocast(self) -> None:
        device = torch.device("cuda")
        actions = torch.linspace(-250.0, 250.0, 255, device=device)
        current = torch.linspace(
            -100.0,
            100.0,
            31,
            device=device,
        ).view(1, -1)
        support = PolicySupport(
            -250.0,
            250.0,
            -100.0,
            100.0,
            0.00175,
            0.01,
        )
        weights = DirectLossWeights(
            cross_entropy=torch.tensor(1.0, device=device),
            probability_mse=torch.tensor(1.0, device=device),
            excess_entropy=torch.tensor(0.0, device=device),
            temporal_mutual_information=torch.tensor(0.0, device=device),
            oracle_mutual_information=torch.tensor(0.5, device=device),
        )
        required = frozenset((
            "cross_entropy",
            "probability_mse",
            "oracle_mutual_information",
        ))

        def objective(logits, target, time_weights, times):
            return direct_oracle_loss(
                logits,
                target,
                actions,
                current,
                support,
                weights,
                time_weights,
                times,
                1_000,
                include_diagnostics=False,
                required_loss_terms=required,
            )["loss"]

        compiled = torch.compile(objective, mode="reduce-overhead", fullgraph=False)
        logits = (
            torch.randn(64, 255, device=device) * 4.0
        ).clamp(-18.0, 18.0).requires_grad_(True)
        target = torch.softmax(torch.randn(64, 255, device=device), dim=-1)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = compiled(
                logits,
                target,
                torch.ones(64, device=device),
                torch.arange(64, device=device, dtype=torch.int64) * 1_000,
            )
        loss.backward()
        torch.cuda.synchronize()

        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertGreater(float(logits.grad.abs().max()), 1e-8)

    def test_compact_training_path_preserves_loss_and_gradient(self) -> None:
        generator = torch.Generator().manual_seed(14)
        actions = torch.linspace(-250.0, 250.0, 255)
        current = torch.linspace(-100.0, 100.0, 31).view(1, -1).expand(4, -1)
        target = torch.softmax(torch.randn(4, 255, generator=generator), dim=-1)
        logits = torch.randn(4, 255, generator=generator, requires_grad=True)
        weights = DirectLossWeights(
            cross_entropy=1.0,
            probability_mse=1.0,
            excess_entropy=0.0,
            temporal_mutual_information=0.0,
            oracle_mutual_information=0.5,
        )
        arguments = (
            target,
            actions,
            current,
            PolicySupport(-250.0, 250.0, -100.0, 100.0, 0.00175, 0.01),
            weights,
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.arange(4, dtype=torch.int64) * 1_000,
            1_000,
        )

        full = direct_oracle_loss(logits, *arguments)
        full["loss"].backward()
        full_gradient = logits.grad.detach().clone()
        logits.grad = None
        compact = direct_oracle_loss(
            logits,
            *arguments,
            include_diagnostics=False,
            required_loss_terms=frozenset((
                "cross_entropy",
                "probability_mse",
                "oracle_mutual_information",
            )),
        )
        compact["loss"].backward()

        torch.testing.assert_close(compact["loss"], full["loss"])
        torch.testing.assert_close(logits.grad, full_gradient)
        self.assertNotIn("klDivergence", compact)
        self.assertNotIn("temporalMutualInformation", compact)
        self.assertIn("oracleMutualInformation", compact)

    def test_indexed_minute_targets_and_precomputed_transition_preserve_loss(
        self,
    ) -> None:
        generator = torch.Generator().manual_seed(41)
        actions = torch.linspace(-250.0, 250.0, 255)
        current = torch.linspace(-100.0, 100.0, 31).view(1, -1)
        unique_targets = torch.softmax(
            torch.randn(3, 255, generator=generator),
            dim=-1,
        )
        target_rows = torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.int64)
        expanded_targets = unique_targets[target_rows]
        logits = torch.randn(6, 255, generator=generator)
        weights = DirectLossWeights(
            cross_entropy=1.0,
            probability_mse=1.0,
            excess_entropy=0.0,
            temporal_mutual_information=0.0,
            oracle_mutual_information=0.5,
        )
        common = (
            actions,
            current.expand(6, -1),
            PolicySupport(-250.0, 250.0, -100.0, 100.0, 0.00175, 0.01),
            weights,
            torch.ones(6),
            torch.arange(6, dtype=torch.int64) * 1_000,
            1_000,
        )
        expanded = direct_oracle_loss(logits, expanded_targets, *common)
        indexed = direct_oracle_loss(
            logits,
            unique_targets,
            actions,
            current,
            *common[2:],
            target_row_indices=target_rows,
            transaction_transition=conditional_transaction_transition(
                actions,
                current,
                common[2],
            ),
        )

        for name in expanded:
            torch.testing.assert_close(indexed[name], expanded[name])


class ValidationMetricAggregationTests(unittest.TestCase):
    def test_weighted_moments_merge_across_minibatches(self) -> None:
        weight, mean, centered_square_sum = merge_weighted_moments(
            torch.tensor(2.0),
            torch.tensor(1.0),
            torch.tensor(0.5),
            torch.tensor(3.0),
            torch.tensor(4.0),
            torch.tensor(1.5),
        )

        torch.testing.assert_close(weight, torch.tensor(5.0))
        torch.testing.assert_close(mean, torch.tensor(2.8))
        torch.testing.assert_close(centered_square_sum, torch.tensor(12.8))
        torch.testing.assert_close(
            weighted_standard_deviation(centered_square_sum, weight),
            torch.tensor(1.6),
        )
        torch.testing.assert_close(
            weighted_variance(centered_square_sum, weight),
            torch.tensor(2.56),
        )
        model = ExposureMlp(torch.zeros(INPUT_FEATURE_COUNT), torch.ones(INPUT_FEATURE_COUNT))
        self.assertEqual(model.layers[0].in_features, INPUT_FEATURE_COUNT)

    def test_temporal_batches_ignore_component_boundaries_but_preserve_time_gaps(self) -> None:
        dataset = type("Dataset", (), {"temporal_runs": [(0, 9), (9, 12)]})()
        sampler = TemporalBlockBatchSampler(dataset, batch_size=3, shuffle=False)

        self.assertEqual(
            [list(block) for block in sampler],
            [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
        )

    def test_minute_batches_coalesce_compressed_daily_storage(self) -> None:
        dataset = type("Dataset", (), {
            "temporal_runs": [(0, 8), (8, 10)],
            "storage_runs": [
                (0, 2, "day-a"),
                (2, 4, "day-b"),
                (4, 8, "day-c"),
                (8, 10, "day-d"),
            ],
            "group_batches_by_storage": True,
            "coalesce_batches_across_storage": True,
        })()
        sampler = TemporalBlockBatchSampler(dataset, batch_size=6, shuffle=False)

        self.assertEqual(
            [list(block) for block in sampler],
            [list(range(0, 6)), [6, 7], [8, 9]],
        )

    def test_raw_compressed_batches_remain_grouped_by_storage(self) -> None:
        dataset = type("Dataset", (), {
            "temporal_runs": [(0, 8)],
            "storage_runs": [
                (0, 2, "day-a"),
                (2, 8, "day-b"),
            ],
            "group_batches_by_storage": True,
            "coalesce_batches_across_storage": False,
        })()
        sampler = TemporalBlockBatchSampler(dataset, batch_size=6, shuffle=False)

        self.assertEqual(
            [list(block) for block in sampler],
            [[0, 1], list(range(2, 8))],
        )

    def test_minute_dataset_reads_exact_compact_feature_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "inputs.features.f16.zst"
            compact_file = root / "inputs.minute.f16"
            compact = np.zeros((1_440, INPUT_FEATURE_COUNT), dtype="<f2")
            compact[0, 0] = 11
            compact[1, 0] = 22
            compact.tofile(compact_file)
            np.zeros((2, 8), dtype="<f4").tofile(root / "parameters.f32")
            np.zeros((2, 8), dtype="<f4").tofile(root / "raw-oracle.f32")
            np.zeros((2, 8), dtype="<f4").tofile(root / "minute-oracle.f32")
            np.zeros((2, 7), dtype="<f4").tofile(root / "metrics.f32")
            np.zeros(2, dtype="<f4").tofile(root / "resolution.f32")
            np.ones(2, dtype="<f4").tofile(root / "weights.f32")
            manifest = {
                "featureCount": INPUT_FEATURE_COUNT,
                "featureSchemaVersion": FEATURE_SCHEMA_VERSION,
                "teacherParameterCount": 8,
                "teacherMetricCount": 7,
                "actionCount": 8,
                "samplingIntervalMs": 60_000,
                "shards": [{
                    "split": "train",
                    "count": 2,
                    "features": source.name,
                    "featureRowOffset": 59,
                    "featureRowStride": 60,
                    "teacherParameters": "parameters.f32",
                    "teacherMetrics": "metrics.f32",
                    "rawOracleProbabilities": "raw-oracle.f32",
                    "minuteOracleProbabilities": "minute-oracle.f32",
                    "resolutionDivergence": "resolution.f32",
                    "oracleRowOffset": 0,
                    "oracleRowStride": 1,
                    "baseTimeWeights": "weights.f32",
                    "timeWeights": "weights.f32",
                    "predictionTimeStart": 59_999,
                }],
            }
            dataset = FittedPolicyDataset(
                manifest,
                root,
                "train",
                target="minuteOracleProbabilities",
                compact_minute_features={
                    source: CompactMinuteFeatureComponent(compact_file, 59),
                },
            )

            batch = dataset.__getitems__(range(0, 2))
            torch.testing.assert_close(
                batch[0][:, 0],
                torch.tensor([11.0, 22.0]),
            )
            self.assertTrue(dataset.coalesce_batches_across_storage)
            self.assertEqual(dataset.storage_runs[0][2], str(compact_file))
            dataset.close()

    def test_temporal_validation_fraction_is_deterministic_and_stratified(self) -> None:
        dataset = type("Dataset", (), {"temporal_runs": [(0, 12), (12, 24)]})()
        sampler = TemporalBlockBatchSampler(
            dataset, batch_size=3, shuffle=False, sample_fraction=0.5,
        )

        self.assertEqual(
            [list(block) for block in sampler],
            [[1, 2, 3], [7, 8, 9], [13, 14, 15], [19, 20, 21]],
        )
        self.assertEqual(sampler.example_count, 12)

    def test_weighted_training_subset_is_fixed_and_preserves_fraction(self) -> None:
        dataset = type("Dataset", (), {
            "temporal_runs": [(0, 100)],
            "mean_time_weight": lambda self, block:
                100.0 if block.start >= 40 else 1.0,
        })()
        first = TemporalBlockBatchSampler(
            dataset,
            batch_size=16,
            shuffle=False,
            sample_fraction=0.2,
            weighted_sample=True,
            seed=7,
        )
        second = TemporalBlockBatchSampler(
            dataset,
            batch_size=16,
            shuffle=False,
            sample_fraction=0.2,
            weighted_sample=True,
            seed=7,
        )

        self.assertEqual(first.example_count, 20)
        self.assertEqual([list(block) for block in first], [
            list(block) for block in second
        ])
        self.assertTrue(any(block.start >= 40 for block in first))

    def test_dataset_pairs_independent_components_by_offset_and_delay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = np.zeros((5, INPUT_FEATURE_COUNT), dtype="<f2")
            features[:, 0] = np.arange(5)
            targets = np.repeat(np.arange(5, dtype="<f4")[:, None], 8, axis=1)
            minute_targets = np.repeat(
                np.array([10, 11], dtype="<f4")[:, None], 8, axis=1
            )
            metrics = np.zeros((5, 7), dtype="<f4")
            features.tofile(root / "inputs.f16")
            targets.tofile(root / "oracle-parameters.f32")
            targets.tofile(root / "raw-oracle.f32")
            minute_targets.tofile(root / "minute-oracle.f32")
            metrics.tofile(root / "oracle-metrics.f32")
            np.zeros(2, dtype="<f4").tofile(root / "resolution.f32")
            np.ones(2, dtype="<f4").tofile(root / "base-weights.f32")
            np.ones(2, dtype="<f4").tofile(root / "weights.f32")
            manifest = {
                "featureCount": INPUT_FEATURE_COUNT,
                "teacherParameterCount": 8,
                "teacherMetricCount": 7,
                "actionCount": 8,
                "samplingIntervalMs": 2_000,
                "shards": [{
                    "split": "train",
                    "count": 2,
                    "features": "inputs.f16",
                    "featureRowOffset": 1,
                    "featureRowStride": 2,
                    "teacherParameters": "oracle-parameters.f32",
                    "teacherMetrics": "oracle-metrics.f32",
                    "rawOracleProbabilities": "raw-oracle.f32",
                    "minuteOracleProbabilities": "minute-oracle.f32",
                    "resolutionDivergence": "resolution.f32",
                    "oracleRowOffset": 0,
                    "oracleRowStride": 2,
                    "baseTimeWeights": "base-weights.f32",
                    "timeWeights": "weights.f32",
                    "predictionTimeStart": 60_999,
                }],
            }
            dataset = FittedPolicyDataset(manifest, root, "train")

            first = dataset[0]
            second = dataset[1]
            self.assertEqual(float(first[0][0]), 1.0)
            self.assertEqual(float(second[0][0]), 3.0)
            self.assertEqual(float(first[1][0]), 0.0)
            self.assertEqual(float(second[1][0]), 2.0)
            self.assertEqual(int(first[3]), 60_999)
            self.assertEqual(int(second[3]), 62_999)

            batch = dataset.__getitems__(range(0, 2))
            torch.testing.assert_close(batch[0][:, 0], torch.tensor([1.0, 3.0]))
            torch.testing.assert_close(batch[1][:, 0], torch.tensor([0.0, 2.0]))
            torch.testing.assert_close(batch[3], torch.tensor([60_999, 62_999]))

            minute_dataset = FittedPolicyDataset(
                manifest,
                root,
                "train",
                target="minuteOracleProbabilities",
            )
            self.assertEqual(float(minute_dataset[0][1][0]), 10.0)
            self.assertEqual(float(minute_dataset[1][1][0]), 11.0)
            minute_dataset.close()
            dataset.close()

    def test_dataset_streams_compressed_persisted_minute_days(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            action_count = 8
            np.zeros((3, INPUT_FEATURE_COUNT), dtype="<f2").tofile(root / "inputs.f16")
            np.zeros((61, 8), dtype="<f4").tofile(root / "parameters.f32")
            np.zeros((61, 8), dtype="<f4").tofile(root / "raw-oracle.f32")
            np.zeros((61, 7), dtype="<f4").tofile(root / "metrics.f32")
            np.zeros(3, dtype="<f4").tofile(root / "resolution.f32")
            np.ones(3, dtype="<f4").tofile(root / "weights.f32")
            minute = np.repeat(
                np.arange(1_441, dtype="<f4")[:, None],
                action_count,
                axis=1,
            )
            (root / "minute-oracle.f32.zst").write_bytes(
                zstandard.ZstdCompressor(level=3).compress(minute.tobytes())
            )
            manifest = {
                "featureCount": INPUT_FEATURE_COUNT,
                "teacherParameterCount": 8,
                "teacherMetricCount": 7,
                "actionCount": action_count,
                "samplingIntervalMs": 1_000,
                "minuteOracleMap": {
                    "storage": "persisted-per-minute-day",
                    "factorCompression": "zstd",
                    "storedRowsPerUtcDay": 1_441,
                },
                "shards": [{
                    "split": "train",
                    "count": 3,
                    "features": "inputs.f16",
                    "featureRowOffset": 0,
                    "featureRowStride": 1,
                    "teacherParameters": "parameters.f32",
                    "teacherMetrics": "metrics.f32",
                    "rawOracleProbabilities": "raw-oracle.f32",
                    "minuteOracleProbabilities": "minute-oracle.f32.zst",
                    "resolutionDivergence": "resolution.f32",
                    "oracleRowOffset": 58,
                    "oracleRowStride": 1,
                    "baseTimeWeights": "weights.f32",
                    "timeWeights": "weights.f32",
                    "predictionTimeStart": 58_000,
                }],
            }
            dataset = FittedPolicyDataset(
                manifest,
                root,
                "train",
                target="minuteOracleProbabilities",
            )
            batch = dataset.__getitems__(range(0, 3))
            torch.testing.assert_close(
                batch[1][:, 0],
                torch.tensor([0.0, 1.0, 1.0]),
            )
            compact_dataset = FittedPolicyDataset(
                manifest,
                root,
                "train",
                target="minuteOracleProbabilities",
                compact_minute_targets=True,
            )
            compact_batch = compact_dataset.__getitems__(range(0, 3))
            self.assertEqual(compact_batch[1].shape, (2, action_count))
            torch.testing.assert_close(
                compact_batch[5],
                torch.tensor([0, 1, 1]),
            )
            torch.testing.assert_close(
                compact_batch[1][compact_batch[5]][:, 0],
                batch[1][:, 0],
            )
            compact_dataset.close()
            dataset.close()

    def test_dataset_coalesces_temporal_runs_across_component_views(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.zeros((4, INPUT_FEATURE_COUNT), dtype="<f2").tofile(root / "inputs.f16")
            np.zeros((4, 8), dtype="<f4").tofile(root / "parameters.f32")
            np.zeros((4, 8), dtype="<f4").tofile(root / "raw-oracle.f32")
            np.zeros((4, 8), dtype="<f4").tofile(root / "minute-oracle.f32")
            np.zeros(4, dtype="<f4").tofile(root / "resolution.f32")
            np.zeros((4, 7), dtype="<f4").tofile(root / "metrics.f32")
            np.ones(2, dtype="<f4").tofile(root / "weights-a.f32")
            np.ones(1, dtype="<f4").tofile(root / "weights-b.f32")
            np.ones(1, dtype="<f4").tofile(root / "weights-c.f32")
            np.ones(4, dtype="<f4").tofile(root / "base-weights.f32")

            def shard(count: int, offset: int, start: int, weights: str) -> dict:
                return {
                    "split": "train",
                    "count": count,
                    "features": "inputs.f16",
                    "featureRowOffset": offset,
                    "featureRowStride": 1,
                    "teacherParameters": "parameters.f32",
                    "teacherMetrics": "metrics.f32",
                    "rawOracleProbabilities": "raw-oracle.f32",
                    "minuteOracleProbabilities": "minute-oracle.f32",
                    "resolutionDivergence": "resolution.f32",
                    "oracleRowOffset": offset,
                    "oracleRowStride": 1,
                    "baseTimeWeights": "base-weights.f32",
                    "timeWeights": weights,
                    "predictionTimeStart": start,
                }

            manifest = {
                "featureCount": INPUT_FEATURE_COUNT,
                "teacherParameterCount": 8,
                "teacherMetricCount": 7,
                "actionCount": 8,
                "samplingIntervalMs": 1_000,
                "shards": [
                    shard(2, 0, 999, "weights-a.f32"),
                    shard(1, 2, 2_999, "weights-b.f32"),
                    shard(1, 3, 9_999, "weights-c.f32"),
                ],
            }
            dataset = FittedPolicyDataset(manifest, root, "train")

            self.assertEqual(dataset.temporal_runs, [(0, 3), (3, 4)])
            batch = dataset.__getitems__(range(1, 4))
            self.assertEqual(tuple(batch[0].shape), (3, INPUT_FEATURE_COUNT))
            torch.testing.assert_close(batch[3], torch.tensor([1_999, 2_999, 9_999]))
            dataset.close()

    def test_training_statistics_are_reused_from_valid_caches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = np.arange(3 * INPUT_FEATURE_COUNT, dtype=np.float32).reshape(
                3, INPUT_FEATURE_COUNT
            )
            targets = np.arange(3 * 8, dtype=np.float32).reshape(3, 8)
            shard = type("Shard", (), {"count": 3})()
            dataset = type("Dataset", (), {
                "parts": [(shard, features, targets, None, None, None)],
                "__len__": lambda self: 3,
            })()
            feature_cache = root / "features.npz"
            target_cache = root / "targets.npz"

            mean, std = cached_training_normalization(dataset, feature_cache)
            scale = cached_training_parameter_scale(dataset, target_cache)
            features[:] = np.nan
            targets[:] = np.nan
            cached_mean, cached_std = cached_training_normalization(dataset, feature_cache)
            cached_scale = cached_training_parameter_scale(dataset, target_cache)

            torch.testing.assert_close(cached_mean, mean)
            torch.testing.assert_close(cached_std, std)
            torch.testing.assert_close(cached_scale, scale)

class RawOracleDatasetContractTests(unittest.TestCase):
    def test_manifest_requires_lossless_255_by_255_raw_oracle_maps(self) -> None:
        grid = torch.linspace(-250.0, 250.0, 255).tolist()
        manifest = {
            "version": 8,
            "featureSchemaVersion": FEATURE_SCHEMA_VERSION,
            "featureCount": INPUT_FEATURE_COUNT,
            "teacherParameterCount": 8,
            "teacherMetricCount": 7,
            "teacherMetricNames": [
                "crossEntropy",
                "klDivergence",
                "meanSquaredError",
                "iterations",
                "restarts",
                "converged",
                "distanceImbalance",
            ],
            "samplingIntervalMs": 1_000,
            "predictionDelayMs": 60_000,
            "timestampPairing": {
                "oracleTargetTime": "predictionTime - predictionDelayMs",
                "splitAssignment": "predictionTime",
                "responseLagMs": 60_000,
            },
            "componentLayout": {"version": 1, "storeId": "test-components-v1"},
            "actionCount": 255,
            "grid": grid,
            "currentGrid": grid,
            "policySupport": {
                "latent_lower": -250.0,
                "latent_upper": 250.0,
            },
            "rawOracleMap": {
                "materializedDtype": "float32",
                "materializedLayout":
                    "row-major [example, currentExposure, targetExposure]",
                "shape": [255, 255],
                "currentExposureGrid": "currentGrid",
                "targetExposureGrid": "grid",
                "normalized": True,
                "losslessEncoding":
                    "base-probabilities-plus-deterministic-transaction-transition-v1",
                "factorDtype": "float32",
                "factorLayout": "row-major [example, targetExposure]",
                "factorShape": [255],
                "factorFileField": "rawOracleProbabilities",
                "optionalHardCutoffCoordinates": "teacherParameters[6:8]",
            },
            "minuteOracleMap": {
                "factorDtype": "float32",
                "factorLayout": "row-major [example, targetExposure]",
                "factorShape": [255],
                "factorFileField": "minuteOracleProbabilities",
                "normalized": True,
                "resolutionDivergence": {
                    "metric": "Jensen-Shannon divergence",
                    "fileField": "resolutionDivergence",
                },
            },
            "exampleWeighting": {
                "dtype": "float32",
                "layout": "row-major [example]",
                "fileField": "timeWeights",
                "baseFileField": "baseTimeWeights",
                "distanceImbalanceMetadataField":
                    "teacherMetrics.distanceImbalance",
                "storedWeights": "causal unnormalized example weights",
                "trainingTransform": "divide each batch by its mean only",
            },
            "shards": [{
                "rawOracleProbabilities":
                    "shards/train-2026-07-09.raw-oracle-probabilities.f32",
                "minuteOracleProbabilities":
                    "shards/train-2026-07-09.minute-oracle-probabilities.f32",
                "resolutionDivergence":
                    "shards/train-2026-07-09.resolution-jsd.f32",
                "timeWeights": "shards/train-2026-07-09.time-weights.f32",
                "baseTimeWeights": "shards/train-2026-07-09.base-time-weights.f32",
            }],
        }

        validate_dataset_manifest(manifest)
        del manifest["shards"][0]["rawOracleProbabilities"]
        with self.assertRaisesRegex(ValueError, "raw oracle map contract"):
            validate_dataset_manifest(manifest)

    def test_factorized_map_materializes_every_current_action_pair(self) -> None:
        actions = torch.tensor([-1.0, 0.0, 1.0])
        current = torch.tensor([-1.0, 0.0, 1.0])
        base = torch.tensor([[0.2, 0.3, 0.5], [0.0, 0.4, 0.6]])
        support = PolicySupport(-1.0, 1.0, -1.0, 1.0, 0.0, 0.01)

        materialized = materialize_raw_oracle_policy_map(
            base, actions, current, support
        )

        self.assertEqual(tuple(materialized.shape), (2, 3, 3))
        torch.testing.assert_close(
            materialized.sum(dim=-1), torch.ones((2, 3))
        )
        torch.testing.assert_close(
            materialized[0], base[0].expand(3, -1)
        )
        self.assertTrue(bool((materialized[1, :, 0] == 0).all()))

        cutoff = materialize_raw_oracle_policy_map(
            base,
            actions,
            current,
            support,
            torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        )
        self.assertTrue(bool((cutoff[:, :, 0] == 0).all()))
        self.assertTrue(bool((cutoff[:, :, 2] == 0).all()))
        torch.testing.assert_close(cutoff.sum(dim=-1), torch.ones((2, 3)))


class DistanceImbalanceTimeWeightTests(unittest.TestCase):
    def test_distance_weighting_distinguishes_unequal_tail_lengths(self) -> None:
        actions = torch.tensor([-1.0, 3.0])
        current = torch.tensor([[0.0]])
        probability = torch.tensor([[[0.5, 0.5]]])

        weight = distance_imbalance_time_weights(
            probability,
            actions,
            current,
            TimeWeighting(distance_epsilon=0.0, minimum_weight=1e-6),
        )

        self.assertAlmostEqual(float(weight[0] - 1e-6), 0.5, places=6)

    def test_state_aggregation_uses_global_distance_ratio(self) -> None:
        actions = torch.tensor([-1.0, 0.0, 1.0])
        current = torch.tensor([[0.0, 0.0]])
        probability = torch.tensor([[
            [0.0, 0.0, 1.0],
            [0.25, 0.5, 0.25],
        ]])

        weight = distance_imbalance_time_weights(
            probability,
            actions,
            current,
            TimeWeighting(distance_epsilon=0.0, minimum_weight=0.1),
        )

        self.assertAlmostEqual(float(weight[0]), 0.1 + 1.0 / 1.5, places=6)

    def test_rows_are_weighted_by_expected_action_distance(self) -> None:
        actions = torch.tensor([-3.0, -1.0, 1.0, 3.0])
        current = torch.tensor([[0.0, 0.0]])
        probability = torch.tensor([[
            [0.0, 0.0, 1.0, 0.0],
            [0.5, 0.0, 0.0, 0.5],
        ]])

        weight = distance_imbalance_time_weights(
            probability,
            actions,
            current,
            TimeWeighting(distance_epsilon=0.0, minimum_weight=0.1),
        )

        # The +1 row contributes +1/4 after the symmetric distance-3 row is
        # included. Averaging normalized row imbalances would incorrectly
        # produce +1/2.
        self.assertAlmostEqual(float(weight[0]), 0.35, places=6)

    def test_opposite_state_imbalances_cancel(self) -> None:
        actions = torch.tensor([-1.0, 1.0])
        current = torch.tensor([[0.0, 0.0]])
        probability = torch.tensor([[
            [0.0, 1.0],
            [1.0, 0.0],
        ]])

        weight = distance_imbalance_time_weights(
            probability,
            actions,
            current,
            TimeWeighting(distance_epsilon=0.0, minimum_weight=0.1),
        )

        self.assertAlmostEqual(float(weight[0]), 0.1, places=6)

    def test_repeated_same_side_advice_grows_causally(self) -> None:
        weights = persistent_distance_imbalance_time_weights(
            torch.tensor([0.5, 0.5, 0.5]),
            torch.tensor([60_000, 120_000, 180_000]),
            60_000,
            TimeWeighting(distance_epsilon=0.0, minimum_weight=1e-6),
        )

        self.assertAlmostEqual(float(weights[0] - 1e-6), 0.5, places=6)
        self.assertAlmostEqual(float(weights[1] - 1e-6), 0.625, places=6)
        self.assertAlmostEqual(float(weights[2] - 1e-6), 0.75, places=6)

    def test_opposite_advice_and_long_gaps_reset_persistence(self) -> None:
        weights = persistent_distance_imbalance_time_weights(
            torch.tensor([0.5, 0.5, -0.5, -0.5]),
            torch.tensor([60_000, 120_000, 180_000, 5_000_000]),
            60_000,
            TimeWeighting(distance_epsilon=0.0, minimum_weight=1e-6),
        )

        self.assertGreater(float(weights[1]), float(weights[0]))
        self.assertAlmostEqual(float(weights[2]), float(weights[0]), places=6)
        self.assertAlmostEqual(float(weights[3]), float(weights[0]), places=6)


class VisibleProbabilityMseTests(unittest.TestCase):
    def test_latent_only_logit_changes_do_not_affect_visible_mse(self) -> None:
        visible = torch.tensor([False, True, True, True, False])
        target_logits = torch.zeros((1, 1, 5))
        predicted_logits = target_logits.clone()
        predicted_logits[..., 0] = 20.0
        predicted_logits[..., 4] = -20.0

        target = torch.softmax(target_logits[..., visible], dim=-1)
        predicted = torch.softmax(predicted_logits[..., visible], dim=-1)
        mse = surface_probability_mse_per_example(predicted, target)

        self.assertEqual(float(mse[0]), 0.0)

    def test_visible_logit_changes_affect_visible_mse(self) -> None:
        visible = torch.tensor([False, True, True, True, False])
        target_logits = torch.zeros((1, 1, 5))
        predicted_logits = target_logits.clone()
        predicted_logits[..., 1] = 2.0

        target = torch.softmax(target_logits[..., visible], dim=-1)
        predicted = torch.softmax(predicted_logits[..., visible], dim=-1)
        mse = surface_probability_mse_per_example(predicted, target)

        self.assertGreater(float(mse[0]), 0.0)

    def test_teacher_cutoff_mask_keeps_distribution_loss_finite(self) -> None:
        actions = torch.linspace(-100.0, 100.0, 255)
        current = torch.linspace(-100.0, 100.0, 31).view(1, -1)
        predicted = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, -5.0]])
        target = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -14.0, 14.0]])

        metrics = fitted_teacher_loss(
            predicted,
            target,
            actions,
            current,
            PolicySupport(-250.0, 250.0, -100.0, 100.0, 0.00175, 0.01),
            torch.ones(8),
            LossWeights(),
            torch.ones(1),
            torch.tensor([1_000]),
            1_000,
        )

        self.assertTrue(bool(torch.isfinite(metrics["loss"])))

    def test_training_only_normalizes_persisted_example_weights(self) -> None:
        actions = torch.linspace(-10.0, 10.0, 31)
        current = torch.linspace(-10.0, 10.0, 7).view(1, -1).expand(2, -1)
        predicted = torch.tensor([
            [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, -14.0, 14.0],
            [0.0, 0.0, -0.2, 0.1, 0.0, 0.0, -14.0, 14.0],
        ])
        target = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -14.0, 14.0],
            [0.0, 0.0, 0.3, 0.0, 0.0, 0.0, -14.0, 14.0],
        ])
        arguments = (
            predicted,
            target,
            actions,
            current,
            PolicySupport(-10.0, 10.0, -10.0, 10.0, 0.001, 0.01),
            torch.ones(8),
            LossWeights(),
        )

        times = torch.tensor([1_000, 2_000])
        original = fitted_teacher_loss(
            *arguments, torch.tensor([1.0, 3.0]), times, 1_000
        )
        rescaled = fitted_teacher_loss(
            *arguments, torch.tensor([10.0, 30.0]), times, 1_000
        )

        torch.testing.assert_close(original["loss"], rescaled["loss"])
        torch.testing.assert_close(original["klDivergence"], rescaled["klDivergence"])
        torch.testing.assert_close(
            original["klDivergenceStdDev"],
            rescaled["klDivergenceStdDev"],
        )
        torch.testing.assert_close(
            original["deploymentKlDivergence"],
            rescaled["deploymentKlDivergence"],
        )
        torch.testing.assert_close(
            original["deploymentKlDivergenceStdDev"],
            rescaled["deploymentKlDivergenceStdDev"],
        )
        torch.testing.assert_close(
            original["klWeightSum"] * 10,
            rescaled["klWeightSum"],
        )
        torch.testing.assert_close(
            original["klCenteredSquareSum"] * 10,
            rescaled["klCenteredSquareSum"],
        )
        torch.testing.assert_close(
            original["deploymentKlWeightSum"] * 10,
            rescaled["deploymentKlWeightSum"],
        )
        torch.testing.assert_close(
            original["deploymentKlCenteredSquareSum"] * 10,
            rescaled["deploymentKlCenteredSquareSum"],
        )

    def test_deployment_kl_penalizes_predicted_cutoff_disagreement(self) -> None:
        actions = torch.linspace(-10.0, 10.0, 31)
        current = torch.linspace(-10.0, 10.0, 7).view(1, -1)
        target = torch.tensor([
            [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, -14.0, 14.0],
        ])
        predicted = target.clone()
        predicted[:, 6:] = 0.0
        metrics = fitted_teacher_loss(
            predicted,
            target,
            actions,
            current,
            PolicySupport(-10.0, 10.0, -10.0, 10.0, 0.001, 0.01),
            torch.ones(8),
            LossWeights(),
            torch.ones(1),
            torch.tensor([1_000]),
            1_000,
        )

        torch.testing.assert_close(
            metrics["klDivergence"],
            torch.zeros_like(metrics["klDivergence"]),
        )
        self.assertTrue(bool(torch.isfinite(metrics["deploymentKlDivergence"])))
        self.assertGreater(float(metrics["deploymentKlDivergence"]), 0.1)

    def test_reported_kl_std_dev_is_weighted_across_examples(self) -> None:
        actions = torch.linspace(-10.0, 10.0, 31)
        current = torch.linspace(-10.0, 10.0, 7).view(1, -1)
        target = torch.tensor([
            [0.0, 0.0, -0.1, 0.0, 0.0, 0.0, -14.0, 14.0],
        ])
        predicted = torch.tensor([
            [0.0, 0.0, 0.4, 0.0, 0.0, 0.0, -14.0, 14.0],
        ])
        common = (
            actions,
            current,
            PolicySupport(-10.0, 10.0, -10.0, 10.0, 0.001, 0.01),
            torch.ones(8),
            LossWeights(),
        )
        matching = fitted_teacher_loss(
            target,
            target,
            *common,
            torch.ones(1),
            torch.tensor([1_000]),
            1_000,
        )
        mismatching = fitted_teacher_loss(
            predicted,
            target,
            *common,
            torch.ones(1),
            torch.tensor([2_000]),
            1_000,
        )
        combined = fitted_teacher_loss(
            torch.cat((target, predicted)),
            target.expand(2, -1),
            actions,
            current.expand(2, -1),
            common[2],
            common[3],
            common[4],
            torch.tensor([1.0, 3.0]),
            torch.tensor([1_000, 2_000]),
            1_000,
        )

        first = matching["klDivergence"]
        second = mismatching["klDivergence"]
        expected_mean = (first + 3 * second) / 4
        expected_std_dev = (
            ((first - expected_mean).square()
             + 3 * (second - expected_mean).square()) / 4
        ).sqrt()
        torch.testing.assert_close(combined["klDivergence"], expected_mean)
        torch.testing.assert_close(
            combined["klDivergenceStdDev"],
            expected_std_dev,
        )

    def test_reported_kl_removes_irreducible_target_entropy(self) -> None:
        actions = torch.linspace(-10.0, 10.0, 31)
        current = torch.linspace(-10.0, 10.0, 7).view(1, -1)
        predicted = torch.tensor([
            [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, -14.0, 14.0],
        ])
        target = torch.tensor([
            [0.0, 0.0, -0.1, 0.0, 0.0, 0.0, -14.0, 14.0],
        ])
        cross_entropy_only = LossWeights(
            cross_entropy=1.0,
            probability_mse=0.0,
            parameter_mse=0.0,
            excess_entropy=0.0,
            temporal_mutual_information=0.0,
            oracle_mutual_information=0.0,
        )

        metrics = fitted_teacher_loss(
            predicted,
            target,
            actions,
            current,
            PolicySupport(-10.0, 10.0, -10.0, 10.0, 0.001, 0.01),
            torch.ones(8),
            cross_entropy_only,
            torch.ones(1),
            torch.tensor([1_000]),
            1_000,
        )

        torch.testing.assert_close(
            metrics["klDivergence"],
            metrics["loss"] - metrics["targetEntropy"],
        )

    def test_temporal_information_rewards_only_real_teacher_changes(self) -> None:
        actions = torch.linspace(-10.0, 10.0, 31)
        current = torch.linspace(-10.0, 10.0, 7).view(1, -1).expand(2, -1)
        changing = torch.tensor([
            [0.0, 0.0, -5.0, 0.0, 0.0, 0.0, -14.0, 14.0],
            [0.0, 0.0, 5.0, 0.0, 0.0, 0.0, -14.0, 14.0],
        ])
        stationary = changing[:1].expand(2, -1).clone()
        weights = LossWeights(
            cross_entropy=0.0,
            probability_mse=0.0,
            parameter_mse=0.0,
            excess_entropy=0.0,
            temporal_mutual_information=1.0,
            oracle_mutual_information=0.0,
        )
        arguments = (
            actions,
            current,
            PolicySupport(-10.0, 10.0, -10.0, 10.0, 0.001, 0.01),
            torch.ones(8),
            weights,
            torch.ones(2),
        )

        changed = fitted_teacher_loss(
            changing, changing, *arguments, torch.tensor([1_000, 2_000]), 1_000
        )
        unchanged = fitted_teacher_loss(
            stationary, stationary, *arguments, torch.tensor([1_000, 2_000]), 1_000
        )
        gapped = fitted_teacher_loss(
            changing, changing, *arguments, torch.tensor([1_000, 3_000]), 1_000
        )

        self.assertGreater(float(changed["temporalMutualInformation"]), 0.0)
        torch.testing.assert_close(
            changed["temporalMutualInformation"],
            changed["targetTemporalMutualInformation"],
        )
        torch.testing.assert_close(
            changed["temporalMutualInformationReward"],
            changed["targetTemporalMutualInformation"],
        )
        self.assertEqual(float(unchanged["temporalMutualInformation"]), 0.0)
        self.assertEqual(float(gapped["temporalMutualInformation"]), 0.0)
        self.assertLess(float(changed["loss"]), 0.0)

    def test_temporal_information_uses_gaussian_time_variance_ratio(self) -> None:
        means = torch.tensor([[-1.0], [1.0]])
        seconds = torch.tensor([[2.0], [2.0]])

        reward, predicted, teacher, count = gaussian_temporal_mutual_information(
            means,
            seconds,
            means,
            seconds,
            torch.tensor([1_000, 2_000]),
            torch.ones(2),
            1_000,
            31,
        )

        expected = 0.5 * torch.log(torch.tensor(2.0)) / torch.log(torch.tensor(31.0))
        torch.testing.assert_close(reward, expected)
        torch.testing.assert_close(predicted, expected)
        torch.testing.assert_close(teacher, expected)
        self.assertEqual(float(count), 2.0)

    def test_oracle_information_uses_per_exposure_time_correlation(self) -> None:
        means = torch.tensor([[-1.0], [1.0]])
        seconds = torch.tensor([[2.0], [2.0]])

        information = gaussian_oracle_mutual_information(
            means,
            seconds,
            means,
            seconds,
            torch.tensor([1_000, 2_000]),
            torch.ones(2),
            1_000,
            31,
        )

        expected = -0.5 * torch.log(torch.tensor(0.75)) / torch.log(torch.tensor(31.0))
        torch.testing.assert_close(information, expected)


if __name__ == "__main__":
    unittest.main()
