from __future__ import annotations

import argparse
import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

from joint_price_oracle import (
    OUTPUT_ACTION_COUNT,
    CausalPatchAggregate,
    HybridTrendDecomposition,
    JointLossWeights,
    JointPriceOracleModel,
    ReversibleInstanceStandardizer,
    causal_moving_average,
    joint_price_oracle_objective,
    joint_price_oracle_policy_only_objective,
)
from train_joint_price_oracle import (
    SECOND_MS,
    CausalSegment,
    PrefetchedBatchIterator,
    causal_close_windows,
    atomic_torch_save,
    configuration_fingerprint,
    load_resume_checkpoint,
    purge_cross_split_windows,
    resolve_training_config,
    snapshot_resolved_plan,
    take_tail,
    validate_experiment_arguments,
)


class JointPriceOracleTest(unittest.TestCase):
    @staticmethod
    def model(
        *,
        context_length: int = 12,
        forecast_horizon: int = 4,
        variable_count: int = 1,
    ) -> JointPriceOracleModel:
        return JointPriceOracleModel(
            context_length=context_length,
            forecast_horizon=forecast_horizon,
            variable_count=variable_count,
            moving_average_window=4,
            patch_length=4,
            linear_rank=4,
            aggregate_mode="hybrid",
            tide_hidden_width=16,
            tide_layer_count=2,
            policy_hidden_width=16,
            policy_layer_count=2,
            dropout=0,
        )

    def test_moving_average_is_trailing_and_causal(self) -> None:
        values = torch.tensor([1.0, 2.0, 3.0, 8.0]).view(1, 4, 1)
        average = causal_moving_average(values, 3)
        expected = torch.tensor([
            1.0,
            4.0 / 3.0,
            2.0,
            13.0 / 3.0,
        ]).view(1, 4, 1)
        self.assertTrue(torch.allclose(average, expected))

        changed = values.clone()
        changed[:, -1] = -100
        changed_average = causal_moving_average(changed, 3)
        self.assertTrue(torch.equal(
            average[:, :-1],
            changed_average[:, :-1],
        ))

    def test_patch_aggregate_starts_as_the_matching_moving_average(
        self,
    ) -> None:
        aggregate = CausalPatchAggregate(variable_count=2, patch_length=4)
        values = torch.randn(3, 11, 2)
        expected = causal_moving_average(values, 4)
        self.assertTrue(torch.allclose(
            aggregate(values),
            expected,
            atol=1e-6,
            rtol=1e-6,
        ))
        self.assertTrue(torch.allclose(
            aggregate.weights(),
            torch.full((2, 4), 0.25),
        ))

    def test_hybrid_decomposition_reconstructs_every_close(self) -> None:
        decomposition = HybridTrendDecomposition(
            variable_count=2,
            moving_average_window=3,
            patch_length=5,
            aggregate_mode="hybrid",
        )
        values = torch.randn(4, 12, 2)
        trend, residual = decomposition(values)
        self.assertTrue(torch.allclose(
            trend + residual,
            values,
            atol=1e-7,
            rtol=1e-7,
        ))

    def test_reversible_normalization_is_per_example_and_variable(
        self,
    ) -> None:
        standardizer = ReversibleInstanceStandardizer(epsilon=1e-8)
        values = torch.randn(3, 20, 2) * torch.tensor([2.0, 9.0]) \
            + torch.tensor([100.0, -30.0])
        normalized, mean, standard_deviation = standardizer.normalize(values)
        self.assertTrue(torch.allclose(
            normalized.mean(dim=1),
            torch.zeros(3, 2),
            atol=2e-5,
            rtol=0,
        ))
        self.assertTrue(torch.allclose(
            normalized.var(dim=1, correction=0),
            torch.ones(3, 2),
            atol=2e-5,
            rtol=0,
        ))
        self.assertTrue(torch.allclose(
            standardizer.denormalize(
                normalized,
                mean,
                standard_deviation,
            ),
            values,
            atol=2e-5,
            rtol=2e-6,
        ))

    def test_untrained_forecast_is_exact_no_change_persistence(self) -> None:
        model = self.model().eval()
        closes = torch.linspace(99.0, 104.0, 24).reshape(2, 12, 1)
        output = model.forward_with_forecast(closes)
        expected = closes[:, -1:, :].expand(-1, 4, -1)
        self.assertTrue(torch.allclose(
            output.predicted_closes,
            expected,
            atol=2e-4,
            rtol=2e-6,
        ))
        self.assertTrue(torch.allclose(
            output.predicted_log_movements,
            torch.zeros_like(output.predicted_log_movements),
            atol=2e-6,
            rtol=0,
        ))

    def test_multivariate_forecast_and_policy_shapes(self) -> None:
        model = self.model(variable_count=2).eval()
        closes = torch.rand(3, 12, 2) * 10 + 100
        output = model.forward_with_forecast(closes)
        self.assertEqual(
            tuple(output.predicted_closes.shape),
            (3, 4, 2),
        )
        self.assertEqual(
            tuple(output.predicted_movements.shape),
            (3, 4, 2),
        )
        self.assertEqual(
            tuple(output.policy_logits.shape),
            (3, OUTPUT_ACTION_COUNT),
        )

    def test_policy_gradient_flows_through_forecast_decoder(self) -> None:
        torch.manual_seed(7)
        model = self.model()
        closes = torch.rand(3, 12, 1) * 10 + 100
        output = model.forward_with_forecast(closes)
        target = torch.softmax(torch.randn(3, OUTPUT_ACTION_COUNT), dim=-1)
        policy_loss = -(
            target
            * torch.log_softmax(output.policy_logits, dim=-1)
        ).sum(dim=-1).mean()
        policy_loss.backward()
        decoder_gradient = (
            model.forecast_backbone.trend_tide
            .temporal_decoder.weight.grad
        )
        self.assertIsNotNone(decoder_gradient)
        assert decoder_gradient is not None
        self.assertGreater(float(decoder_gradient.abs().sum()), 0)

    def test_glu_paths_share_a_fixed_canonical_centering_matrix(
        self,
    ) -> None:
        layer = self.model().forecast_backbone.trend_tide.layers[0]
        self.assertIs(
            layer.value_normalizer.weight,
            layer.gate_normalizer.weight,
        )
        self.assertIs(
            layer.value_normalizer.weight,
            layer.residual_value_normalizer.weight,
        )
        self.assertIs(
            layer.value_normalizer.weight,
            layer.residual_gate_normalizer.weight,
        )
        self.assertFalse(layer.value_normalizer.weight.requires_grad)
        width = layer.output_width
        expected = (
            torch.eye(width) - torch.full((width, width), 1.0 / width)
        )
        self.assertTrue(torch.allclose(
            layer.value_normalizer.weight,
            expected,
        ))

    def test_joint_objective_rewards_exact_policy_and_forecast(self) -> None:
        torch.manual_seed(11)
        model = self.model().eval()
        closes = torch.linspace(100, 101, 24).reshape(2, 12, 1)
        output = model.forward_with_forecast(closes)
        target_policy = torch.softmax(
            output.policy_logits.detach(),
            dim=-1,
        )
        exact_future = output.predicted_closes.detach()
        metrics = joint_price_oracle_objective(
            output,
            closes,
            exact_future,
            target_policy,
            JointLossWeights(
                policy_cross_entropy=1,
                forecast=1,
                soft_layer_norm=0,
            ),
        )
        self.assertAlmostEqual(
            float(metrics["forecastLoss"].detach()),
            0.0,
        )
        self.assertAlmostEqual(
            float(metrics["klDivergence"].detach()),
            0.0,
            places=5,
        )
        self.assertAlmostEqual(
            float(metrics["directionAccuracy"].detach()),
            1.0,
        )

    def test_policy_only_raw_metrics_match_full_objective_exactly(self) -> None:
        torch.manual_seed(12)
        model = self.model().eval()
        closes = torch.rand(4, 12, 1) * 20 + 100
        output = model.forward_with_forecast(closes)
        target_policy = torch.rand(4, OUTPUT_ACTION_COUNT)
        example_weights = torch.tensor([0.5, 1.0, 2.0, 4.0])
        weights = JointLossWeights(
            policy_cross_entropy=1,
            conditioned_policy_cross_entropy=0,
            forecast=0,
            soft_layer_norm=0,
        )
        full = joint_price_oracle_objective(
            output,
            closes,
            output.predicted_closes.detach(),
            target_policy,
            weights,
            example_weights=example_weights,
        )
        policy_only = joint_price_oracle_policy_only_objective(
            output.policy_logits,
            target_policy,
            weights,
            example_weights=example_weights,
        )
        for name in (
            "loss",
            "crossEntropy",
            "klDivergence",
            "probabilityMse",
        ):
            self.assertTrue(torch.equal(full[name], policy_only[name]), name)
        for name in (
            "conditionedCrossEntropy",
            "conditionedKlDivergence",
            "forecastLoss",
            "nextMovementRmse",
            "directionAccuracy",
            "softLayerNorm",
            "softLayerNormMeanPenalty",
            "softLayerNormVariancePenalty",
        ):
            self.assertEqual(float(policy_only[name]), 0.0, name)

        with self.assertRaisesRegex(ValueError, "weights to be zero"):
            joint_price_oracle_policy_only_objective(
                output.policy_logits,
                target_policy,
                JointLossWeights(forecast=1, soft_layer_norm=0),
            )

    def test_conditioned_policy_objective_matches_exact_policy(self) -> None:
        torch.manual_seed(13)
        model = self.model().eval()
        closes = torch.linspace(100, 101, 24).reshape(2, 12, 1)
        output = model.forward_with_forecast(closes)
        target_policy = torch.softmax(
            output.policy_logits.detach(),
            dim=-1,
        )
        metrics = joint_price_oracle_objective(
            output,
            closes,
            output.predicted_closes.detach(),
            target_policy,
            JointLossWeights(
                policy_cross_entropy=1,
                conditioned_policy_cross_entropy=1,
                forecast=1,
                soft_layer_norm=0,
            ),
            action_grid=torch.linspace(-98, 98, OUTPUT_ACTION_COUNT),
            policy_friction=0.00175,
            policy_temperature=0.01,
        )
        self.assertAlmostEqual(
            float(metrics["conditionedKlDivergence"].detach()),
            0.0,
            places=5,
        )

    def test_causal_windows_end_input_at_t_and_begin_label_at_t_plus_one(
        self,
    ) -> None:
        closes = np.arange(1, 10, dtype=np.float64)
        inputs, future = causal_close_windows(
            closes,
            count=3,
            context_length=4,
            forecast_horizon=3,
        )
        np.testing.assert_array_equal(inputs, np.asarray([
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6],
        ], dtype=np.float32))
        np.testing.assert_array_equal(future, np.asarray([
            [5, 6, 7],
            [6, 7, 8],
            [7, 8, 9],
        ], dtype=np.float32))

    def test_split_purge_removes_cross_split_context_and_future(
        self,
    ) -> None:
        target = Path("target")
        segments = {
            "train": [CausalSegment(
                "train",
                999,
                10,
                target,
                0,
            )],
            "validation": [CausalSegment(
                "validation",
                10 * SECOND_MS + 999,
                10,
                target,
                10,
            )],
            "test": [CausalSegment(
                "test",
                30 * SECOND_MS + 999,
                10,
                target,
                30,
            )],
        }
        purged = purge_cross_split_windows(
            segments,
            context_length=3,
            forecast_horizon=2,
        )
        # Train t+1 and t+2 labels may not enter validation at second 10.
        self.assertEqual(purged["train"][0].prediction_time_end, 7_999)
        # Validation context may not contain the train split through second 9.
        self.assertEqual(
            purged["validation"][0].prediction_time_start,
            12_999,
        )

    def test_take_tail_preserves_target_row_alignment(self) -> None:
        target = Path("target")
        segments = [
            CausalSegment("test", 999, 5, target, 0),
            CausalSegment("test", 10_999, 5, target, 10),
        ]
        tail = take_tail(segments, 7)
        self.assertEqual(len(tail), 2)
        self.assertEqual(tail[0].prediction_time_start, 3_999)
        self.assertEqual(tail[0].target_row_offset, 3)
        self.assertEqual(tail[0].count, 2)
        self.assertEqual(tail[1], segments[1])

    def test_resume_checkpoint_requires_exact_model_and_data_contract(
        self,
    ) -> None:
        model = self.model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        model_config = {"test": "small"}
        plan = {"id": "joint-test"}
        training_fingerprint = configuration_fingerprint({"epochs": 8})
        from joint_price_oracle import (
            ARCHITECTURE_CONTRACT,
            parameter_count,
        )
        from train_joint_price_oracle import DATA_CONTRACT

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": 3,
            "globalStep": 17,
            "bestValidation": 0.4,
            "bestEpoch": 2,
            "staleEpochs": 1,
            "parameterCount": parameter_count(model),
            "architectureContract": ARCHITECTURE_CONTRACT,
            "dataContract": DATA_CONTRACT,
            "datasetFingerprint": "dataset-a",
            "modelConfig": model_config,
            "planId": plan["id"],
            "trainingConfigFingerprint": training_fingerprint,
            "rng": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            file = (
                Path(directory)
                / "data"
                / "training"
                / "runs"
                / "joint-test"
                / "checkpoints"
                / "last.json"
            )
            atomic_torch_save(checkpoint, file)
            resumed = load_resume_checkpoint(
                file,
                model,
                optimizer,
                scheduler,
                plan,
                model_config,
                "dataset-a",
                parameter_count(model),
                torch.device("cpu"),
                training_config_fingerprint=training_fingerprint,
            )
            self.assertEqual(resumed, (4, 17, 0.4, 2, 1))
            with self.assertRaises(ValueError):
                load_resume_checkpoint(
                    file,
                    model,
                    optimizer,
                    scheduler,
                    plan,
                    model_config,
                    "changed-dataset",
                    parameter_count(model),
                    torch.device("cpu"),
                    training_config_fingerprint=training_fingerprint,
                )
            with self.assertRaises(ValueError):
                load_resume_checkpoint(
                    file,
                    model,
                    optimizer,
                    scheduler,
                    plan,
                    model_config,
                    "dataset-a",
                    parameter_count(model),
                    torch.device("cpu"),
                    training_config_fingerprint="changed-training",
                )

    def test_resume_allows_only_known_unfingerprinted_joint_runs(
        self,
    ) -> None:
        model = self.model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        model_config = {"test": "small"}
        from joint_price_oracle import (
            ARCHITECTURE_CONTRACT,
            parameter_count,
        )
        from train_joint_price_oracle import DATA_CONTRACT

        plan = {
            "id": (
                "joint-price-oracle-tide-rlinear-dlinear-v2-1h-1m-1m"
            ),
        }
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": 16,
            "globalStep": 8556,
            "bestValidation": 0.989,
            "bestEpoch": 15,
            "staleEpochs": 1,
            "parameterCount": parameter_count(model),
            "architectureContract": ARCHITECTURE_CONTRACT,
            "dataContract": DATA_CONTRACT,
            "datasetFingerprint": "dataset-a",
            "modelConfig": model_config,
            "planId": plan["id"],
            "rng": {},
            "interrupted": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            file = (
                Path(directory)
                / "data"
                / "training"
                / "runs"
                / plan["id"]
                / "checkpoints"
                / "last.json"
            )
            atomic_torch_save(checkpoint, file)
            resumed = load_resume_checkpoint(
                file,
                model,
                optimizer,
                scheduler,
                plan,
                model_config,
                "dataset-a",
                parameter_count(model),
                torch.device("cpu"),
                training_config_fingerprint="new-fingerprint",
            )
            self.assertEqual(resumed, (17, 8556, 0.989, 15, 1))

            checkpoint["planId"] = "unknown-legacy-run"
            plan["id"] = "unknown-legacy-run"
            atomic_torch_save(checkpoint, file)
            with self.assertRaises(ValueError):
                load_resume_checkpoint(
                    file,
                    model,
                    optimizer,
                    scheduler,
                    plan,
                    model_config,
                    "dataset-a",
                    parameter_count(model),
                    torch.device("cpu"),
                    training_config_fingerprint="new-fingerprint",
                )

    def test_training_config_resolution_and_plan_snapshot_are_stable(
        self,
    ) -> None:
        training = {
            "epochs": 8,
            "lossWeights": {
                "policyCrossEntropy": 1,
                "forecast": 1,
                "softLayerNorm": 0.01,
            },
        }
        resolved = resolve_training_config(training)
        self.assertEqual(
            resolved["lossWeights"]["conditionedPolicyCrossEntropy"],
            0.0,
        )
        fingerprint = configuration_fingerprint(resolved)
        self.assertEqual(fingerprint, configuration_fingerprint(resolved))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "data" / "training" / "runs" / "run-a"
            plan_file = root / "plans" / "run-a.json"
            plan = {
                "id": "run-a",
                "runDir": "data/training/runs/run-a",
                "historyDir": "data/history",
                "training": training,
            }
            snapshot = snapshot_resolved_plan(
                run_dir,
                plan_file,
                plan,
                root,
                resolved,
                fingerprint,
            )
            self.assertTrue(snapshot.is_file())
            self.assertEqual(
                snapshot_resolved_plan(
                    run_dir,
                    plan_file,
                    plan,
                    root,
                    resolved,
                    fingerprint,
                ),
                snapshot,
            )
            changed = resolve_training_config({**training, "epochs": 9})
            with self.assertRaises(ValueError):
                snapshot_resolved_plan(
                    run_dir,
                    plan_file,
                    {**plan, "training": {**training, "epochs": 9}},
                    root,
                    changed,
                    configuration_fingerprint(changed),
                )

    def test_maximum_batches_requires_isolated_disposable_smoke_run(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runs = Path(directory) / "data" / "training" / "runs"
            normal = runs / "normal"
            smoke = runs / "disposable-smoke" / "smoke-a"

            def arguments(**overrides):
                values = {
                    "maximum_batches": None,
                    "disposable_smoke": False,
                    "stop_after_epoch": None,
                    "evaluate_test": False,
                    "check_data": False,
                }
                values.update(overrides)
                return argparse.Namespace(**values)

            validate_experiment_arguments(arguments(), normal, runs)
            with self.assertRaises(ValueError):
                validate_experiment_arguments(
                    arguments(maximum_batches=2),
                    normal,
                    runs,
                )
            validate_experiment_arguments(
                arguments(maximum_batches=2, disposable_smoke=True),
                smoke,
                runs,
            )
            with self.assertRaises(ValueError):
                validate_experiment_arguments(
                    arguments(maximum_batches=2, disposable_smoke=True),
                    normal,
                    runs,
                )
            with self.assertRaises(ValueError):
                validate_experiment_arguments(
                    arguments(stop_after_epoch=3, evaluate_test=True),
                    normal,
                    runs,
                )

    def test_cpu_batch_prefetch_preserves_order(self) -> None:
        expected = [
            (
                torch.full((1,), index),
                torch.full((1,), index + 10),
                torch.full((1,), index + 20),
            )
            for index in range(4)
        ]
        with PrefetchedBatchIterator(iter(expected), 2) as prefetched:
            actual = list(prefetched)
        self.assertEqual(len(actual), len(expected))
        for actual_batch, expected_batch in zip(
            actual,
            expected,
            strict=True,
        ):
            for actual_value, expected_value in zip(
                actual_batch,
                expected_batch,
                strict=True,
            ):
                self.assertTrue(torch.equal(actual_value, expected_value))


if __name__ == "__main__":
    unittest.main()
